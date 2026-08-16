"""把不可变 retrieved 物化结果展开为可审计的分层语义审核队列。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rag.knowledge import ArticleRepository

try:
    from . import materialize_rag_sft_retrieved as materializer
    from . import prepare_rag_sft as preparation
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import materialize_rag_sft_retrieved as materializer
    from dataset import prepare_rag_sft as preparation


RAG_SFT_ROOT = Path(__file__).resolve().parent / "RAG-SFT"
DEFAULT_CANONICAL_AUTHORING = (
    RAG_SFT_ROOT / "authoring" / "rag-sft-canonical-v1.jsonl"
)
DEFAULT_ARTICLE_INDEX = Path(__file__).resolve().parents[2] / "rag" / "chunk" / "article_index.jsonl"
DEFAULT_MATERIALIZATION_DIR = (
    RAG_SFT_ROOT / "retrieved" / "materialization-canonical-v1-20260809"
)
DEFAULT_OUTPUT_DIR = (
    RAG_SFT_ROOT / "retrieved" / "review-canonical-v1-20260809"
)

DETERMINISTIC_FILENAME = "rag-sft-retrieval-deterministic-decisions.jsonl"
QUEUE_FILENAME = "rag-sft-retrieval-review-queue.jsonl"
PILOT_FILENAME = "rag-sft-retrieval-review-pilot.jsonl"
MANIFEST_FILENAME = "rag-sft-retrieval-review-manifest.json"
HASH_FILENAME = "rag-sft-retrieval-review-manifest.sha256"

SUPPORTED_DISPOSITIONS = {
    "duplicate_existing",
    "needs_positive_summary",
    "retrieved_answer_approved",
    "retrieved_answer_draft",
    "retrieved_refusal_draft",
}
PILOT_TARGETS = {
    "answer_same_law_extra": 12,
    "answer_cross_law_only": 8,
    "refusal_budget_dropped_gt": 10,
    "refusal_top5_missing_gt": 10,
}


class RagSftRetrievalReviewQueueError(RuntimeError):
    """retrieved 审核队列无法安全构造。"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    path = path.resolve()
    result: dict[str, object] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if records is not None:
        result["records"] = records
    return result


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftRetrievalReviewQueueError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise RagSftRetrievalReviewQueueError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagSftRetrievalReviewQueueError(
                        f"{description}不允许空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RagSftRetrievalReviewQueueError(
                        f"{description}第 {line_number} 条必须是 object"
                    )
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftRetrievalReviewQueueError):
            raise
        raise RagSftRetrievalReviewQueueError(
            f"无法读取{description}: {path}"
        ) from error
    return records


def _unique(records, field: str, description: str) -> dict[str, dict[str, Any]]:
    result = {}
    for position, record in enumerate(records, start=1):
        value = record.get(field)
        if not isinstance(value, str) or not value or value in result:
            raise RagSftRetrievalReviewQueueError(
                f"{description}第 {position} 条 {field} 无效或重复"
            )
        result[value] = record
    return result


def _verify_materialization(directory: Path) -> tuple[dict[str, Any], Path, Path]:
    directory = directory.resolve()
    report_path = directory / materializer.REPORT_FILENAME
    locator_path = directory / materializer.LOCATOR_FILENAME
    provisional_path = directory / materializer.AUTHORING_FILENAME
    hash_path = directory / materializer.HASH_FILENAME
    expected_paths = {
        materializer.AUTHORING_FILENAME: provisional_path,
        materializer.LOCATOR_FILENAME: locator_path,
        materializer.REPORT_FILENAME: report_path,
    }
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftRetrievalReviewQueueError("无法读取物化 SHA-256 清单") from error
    parsed = {}
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2 or parts[1] in parsed:
            raise RagSftRetrievalReviewQueueError("物化 SHA-256 清单格式无效")
        parsed[parts[1]] = parts[0]
    if set(parsed) != set(expected_paths):
        raise RagSftRetrievalReviewQueueError("物化 SHA-256 清单文件集合无效")
    for name, path in expected_paths.items():
        if not path.is_file() or sha256_file(path) != parsed[name]:
            raise RagSftRetrievalReviewQueueError(f"物化文件身份已变化: {name}")
    report = _load_json(report_path, "物化报告")
    if (
        report.get("pipeline") != "rag_sft_retrieved_materialization"
        or report.get("complete") is not True
        or report.get("readiness", {}).get("real_retrieval_executed") is not True
    ):
        raise RagSftRetrievalReviewQueueError("物化报告状态无效")
    return report, locator_path, provisional_path


def _bound_authoring(
    path: Path, report: dict[str, Any]
) -> list[dict[str, Any]]:
    path = path.resolve()
    metadata = report.get("input", {}).get("authoring")
    if (
        not isinstance(metadata, dict)
        or metadata.get("path") != str(path)
        or metadata.get("bytes") != path.stat().st_size
        or metadata.get("sha256") != sha256_file(path)
    ):
        raise RagSftRetrievalReviewQueueError("canonical authoring 与物化报告不一致")
    return _load_jsonl(path, "canonical authoring")


def _article_payload(repository: ArticleRepository, chunk_id: str) -> dict[str, str]:
    try:
        article = repository.get_by_chunk_id(chunk_id)
    except KeyError as error:
        raise RagSftRetrievalReviewQueueError(f"法条索引缺少 {chunk_id}") from error
    return {
        "chunk_id": article.chunk_id,
        "law_name": article.law_name,
        "article_no": article.article_no,
        "content": article.content,
    }


def _decision(locator: dict[str, Any], action: str, reason: str) -> dict[str, object]:
    return {
        "source_id": locator["source_id"],
        "derived_id": locator["derived_id"],
        "locator_sha256": canonical_digest(locator),
        "decision": {
            "source_id": locator["source_id"],
            "decision": action,
            "supporting_extra_chunk_ids": [],
            "non_supporting_extra_chunk_ids": [],
            "replacement_summary": None,
            "reason": reason,
        },
    }


def _review_lane(locator: dict[str, Any]) -> str:
    disposition = locator["disposition"]
    if disposition == "retrieved_answer_draft":
        required_laws = {
            item.split("#", 1)[0] for item in locator["required_chunk_ids"]
        }
        extra_laws = {
            item.split("#", 1)[0] for item in locator["extra_non_gt_chunk_ids"]
        }
        return (
            "answer_same_law_extra"
            if required_laws & extra_laws
            else "answer_cross_law_only"
        )
    if disposition != "retrieved_refusal_draft":
        raise RagSftRetrievalReviewQueueError(f"无法分层 disposition: {disposition}")
    retrieved_ids = {
        item["chunk_id"] for item in locator.get("retrieved_candidates", [])
    }
    return (
        "refusal_budget_dropped_gt"
        if set(locator["required_chunk_ids"]).issubset(retrieved_ids)
        else "refusal_top5_missing_gt"
    )


def _queue_record(
    locator: dict[str, Any],
    parent: dict[str, Any],
    repository: ArticleRepository,
) -> dict[str, object]:
    required_ids = tuple(locator["required_chunk_ids"])
    packaged_ids = tuple(locator["packaged_chunk_ids"])
    ranked = {item["chunk_id"]: item for item in locator["retrieved_candidates"]}
    required = [_article_payload(repository, item) for item in required_ids]
    packaged = []
    for chunk_id in packaged_ids:
        item: dict[str, object] = _article_payload(repository, chunk_id)
        item["is_required"] = chunk_id in required_ids
        diagnostics = ranked.get(chunk_id)
        if not isinstance(diagnostics, dict):
            raise RagSftRetrievalReviewQueueError(
                f"{locator['source_id']} 包内法条缺少检索诊断: {chunk_id}"
            )
        item["rrf_rank"] = diagnostics["rrf_rank"]
        item["rrf_score"] = diagnostics["rrf_score"]
        item["rerank_score"] = diagnostics["rerank_score"]
        packaged.append(item)
    return {
        "source_id": locator["source_id"],
        "derived_id": locator["derived_id"],
        "locator_sha256": canonical_digest(locator),
        "lane": _review_lane(locator),
        "query_original": parent["query_original"],
        "target_summary": parent["target"]["summary"],
        "parent_evidence_source": parent["evidence_source"],
        "required_evidence": required,
        "packaged_evidence": packaged,
        "missing_required_chunk_ids": [
            item for item in required_ids if item not in packaged_ids
        ],
        "extra_non_gt_chunk_ids": list(locator["extra_non_gt_chunk_ids"]),
        "support_spans": list(parent["support_spans"]),
        "review_requirements": {
            "partition_every_extra": True,
            "answer_must_have_no_supporting_extra": True,
            "refusal_must_have_no_alternative_complete_answer": True,
            "target_conflict_requires_escalation": True,
        },
    }


def _pilot(queue: list[dict[str, object]]) -> list[dict[str, object]]:
    selected = []
    for lane, target in PILOT_TARGETS.items():
        candidates = [item for item in queue if item["lane"] == lane]
        candidates.sort(
            key=lambda item: hashlib.sha256(
                str(item["source_id"]).encode("utf-8")
            ).hexdigest()
        )
        selected.extend(candidates[:target])
    selected.sort(key=lambda item: item["source_id"])
    return selected


def _write_jsonl(values: list[dict[str, object]]) -> str:
    return "".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for item in values
    )


def _publish(
    output_dir: Path,
    deterministic: list[dict[str, object]],
    queue: list[dict[str, object]],
    pilot: list[dict[str, object]],
    manifest: dict[str, object],
) -> None:
    if output_dir.exists():
        raise RagSftRetrievalReviewQueueError(f"审核输出目录必须是新目录: {output_dir}")
    output_dir.mkdir(parents=True)
    payloads = (
        (DETERMINISTIC_FILENAME, _write_jsonl(deterministic)),
        (QUEUE_FILENAME, _write_jsonl(queue)),
        (PILOT_FILENAME, _write_jsonl(pilot)),
        (
            MANIFEST_FILENAME,
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        ),
    )
    partials = []
    published = []
    try:
        for filename, payload in payloads:
            partial = output_dir / f"{filename}.partial"
            final = output_dir / filename
            partial.write_text(payload, encoding="utf-8", newline="\n")
            partials.append((partial, final))
        hash_payload = "".join(
            f"{sha256_file(partial)}  {final.name}\n" for partial, final in partials
        )
        hash_partial = output_dir / f"{HASH_FILENAME}.partial"
        hash_partial.write_text(hash_payload, encoding="utf-8", newline="\n")
        for partial, final in partials:
            partial.replace(final)
            published.append(final)
        hash_partial.replace(output_dir / HASH_FILENAME)
    except (OSError, UnicodeError, ValueError) as error:
        for path in [
            *(partial for partial, _ in partials),
            output_dir / f"{HASH_FILENAME}.partial",
            *reversed(published),
        ]:
            path.unlink(missing_ok=True)
        try:
            output_dir.rmdir()
        except OSError:
            pass
        raise RagSftRetrievalReviewQueueError("无法发布 retrieved 审核队列") from error


def build_retrieval_review_queue(
    *,
    canonical_authoring_path: Path,
    article_index_path: Path,
    materialization_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """验证物化身份并发布确定性决定、语义队列和分层 pilot。"""

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RagSftRetrievalReviewQueueError(f"审核输出目录必须是新目录: {output_dir}")
    report, locator_path, provisional_path = _verify_materialization(
        Path(materialization_dir)
    )
    canonical_records = _bound_authoring(Path(canonical_authoring_path), report)
    canonical_by_id = _unique(canonical_records, "id", "canonical authoring")
    locators = _load_jsonl(locator_path, "retrieved locator")
    provisional = _load_jsonl(provisional_path, "retrieved provisional")
    provisional_by_id = _unique(provisional, "id", "retrieved provisional")
    expected_count = report.get("scope", {}).get("eligible_records")
    if len(locators) != expected_count:
        raise RagSftRetrievalReviewQueueError("locator 数量与物化报告不一致")
    if set(locator.get("disposition") for locator in locators) - SUPPORTED_DISPOSITIONS:
        raise RagSftRetrievalReviewQueueError("物化结果含未闭合的失败 disposition")

    article_index_path = Path(article_index_path).resolve()
    repository = ArticleRepository.from_jsonl(article_index_path)
    deterministic = []
    queue = []
    disposition_counts: Counter[str] = Counter()
    for locator in locators:
        source_id = locator.get("source_id")
        if source_id not in canonical_by_id:
            raise RagSftRetrievalReviewQueueError(f"{source_id} 缺少 canonical 父记录")
        if locator.get("derived_id") != materializer._derived_id(source_id):
            raise RagSftRetrievalReviewQueueError(f"{source_id} derived_id 无效")
        parent = canonical_by_id[source_id]
        preparation._validate_record_shape(parent, source_id)
        disposition = locator["disposition"]
        disposition_counts[disposition] += 1
        if disposition == "duplicate_existing":
            deterministic.append(
                _decision(locator, "exclude_duplicate", "模型可见 conversations 已存在。")
            )
        elif disposition == "retrieved_answer_approved":
            deterministic.append(
                _decision(locator, "approve_answer", "全部必要 GT 入包且不存在 extra。")
            )
        elif disposition == "needs_positive_summary":
            deterministic.append(
                _decision(
                    locator,
                    "exclude_incomplete_answer",
                    "父记录没有已审核正向 summary，按冻结策略隔离且不新写答案。",
                )
            )
        else:
            queue.append(_queue_record(locator, parent, repository))

    expected_provisional_ids = {
        locator["derived_id"]
        for locator in locators
        if locator["disposition"]
        in {
            "retrieved_answer_approved",
            "retrieved_answer_draft",
            "retrieved_refusal_draft",
        }
    }
    if set(provisional_by_id) != expected_provisional_ids:
        raise RagSftRetrievalReviewQueueError("provisional 与 locator 输出身份不闭合")
    if len(deterministic) + len(queue) != len(locators):
        raise RagSftRetrievalReviewQueueError("确定性决定与语义队列数量不闭合")
    pilot = _pilot(queue)
    queue_counts = Counter(item["lane"] for item in queue)
    pilot_counts = Counter(item["lane"] for item in pilot)
    manifest: dict[str, object] = {
        "pipeline": "rag_sft_retrieval_review_queue",
        "policy": {
            "duplicate_existing": "deterministic_exclude",
            "retrieved_answer_approved": "deterministic_approve",
            "needs_positive_summary": "deterministic_isolate_without_authoring",
            "semantic_review_roles": ["legal_support", "adversarial_boundary"],
            "user_reviews_only_disagreement_or_high_risk": True,
        },
        "input": {
            "canonical_authoring": _identity(
                Path(canonical_authoring_path), records=len(canonical_records)
            ),
            "article_index": _identity(article_index_path),
            "materialization_report": _identity(
                Path(materialization_dir) / materializer.REPORT_FILENAME
            ),
            "locators": _identity(locator_path, records=len(locators)),
            "provisional_authoring": _identity(
                provisional_path, records=len(provisional)
            ),
        },
        "records": {
            "locators": len(locators),
            "dispositions": dict(sorted(disposition_counts.items())),
            "deterministic_decisions": len(deterministic),
            "semantic_review_queue": len(queue),
            "review_queue_by_lane": dict(sorted(queue_counts.items())),
            "pilot": len(pilot),
            "pilot_by_lane": dict(sorted(pilot_counts.items())),
        },
        "outputs": {
            "deterministic_decisions": DETERMINISTIC_FILENAME,
            "review_queue": QUEUE_FILENAME,
            "pilot": PILOT_FILENAME,
            "manifest": MANIFEST_FILENAME,
            "sha256_manifest": HASH_FILENAME,
        },
        "readiness": {
            "review_queue_ready": True,
            "semantic_reviews_complete": False,
            "training_ready": False,
        },
        "complete": True,
    }
    _publish(output_dir, deterministic, queue, pilot, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-authoring", type=Path, default=DEFAULT_CANONICAL_AUTHORING
    )
    parser.add_argument("--article-index", type=Path, default=DEFAULT_ARTICLE_INDEX)
    parser.add_argument(
        "--materialization-dir", type=Path, default=DEFAULT_MATERIALIZATION_DIR
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        manifest = build_retrieval_review_queue(
            canonical_authoring_path=args.canonical_authoring,
            article_index_path=args.article_index,
            materialization_dir=args.materialization_dir,
            output_dir=args.output_dir,
        )
    except RagSftRetrievalReviewQueueError as error:
        parser.error(str(error))
    print(
        f"[完成] 确定性决定 {manifest['records']['deterministic_decisions']} 条，"
        f"语义队列 {manifest['records']['semantic_review_queue']} 条，"
        f"pilot {manifest['records']['pilot']} 条"
    )
    print(f"产物目录: {args.output_dir}")


if __name__ == "__main__":
    main()


__all__ = [
    "RagSftRetrievalReviewQueueError",
    "build_retrieval_review_queue",
    "canonical_digest",
]
