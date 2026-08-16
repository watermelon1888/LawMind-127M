"""从 v2 真实检索候选生成可追溯的 HN 全文审核工作包。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_ROOT = Path(__file__).resolve().parent
RAG_SFT_ROOT = DATASET_ROOT / "RAG-SFT"
DEFAULT_MATERIALIZATION_DIR = (
    RAG_SFT_ROOT / "retrieved" / "materialization-v2-original-top5-20260813"
)
DEFAULT_ORACLE_MANIFEST = RAG_SFT_ROOT / "review" / "v2" / "oracle-clean-v2" / "manifest.json"
DEFAULT_ARTICLE_INDEX = Path(__file__).resolve().parents[2] / "rag" / "chunk" / "article_index.jsonl"
DEFAULT_OUTPUT_DIR = RAG_SFT_ROOT / "review" / "v2" / "hn-semantic-review-v1-20260813"
BATCH_SIZE = 25


class RagSftV2HnReviewPackageError(RuntimeError):
    """HN 审核工作包无法与冻结输入可靠绑定。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, records: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if records is not None:
        result["records"] = records
    return result


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftV2HnReviewPackageError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise RagSftV2HnReviewPackageError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagSftV2HnReviewPackageError(
                        f"{description}不允许空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RagSftV2HnReviewPackageError(
                        f"{description}第 {line_number} 条必须是 object"
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftV2HnReviewPackageError):
            raise
        raise RagSftV2HnReviewPackageError(f"无法读取{description}: {path}") from error
    if not rows:
        raise RagSftV2HnReviewPackageError(f"{description}不能为空")
    return rows


def _verify_identity(metadata: object, description: str) -> Path:
    if not isinstance(metadata, dict):
        raise RagSftV2HnReviewPackageError(f"缺少{description}身份")
    path_value = metadata.get("path")
    if (
        not isinstance(path_value, str)
        or type(metadata.get("bytes")) is not int
        or not isinstance(metadata.get("sha256"), str)
    ):
        raise RagSftV2HnReviewPackageError(f"{description}身份字段无效")
    path = Path(path_value).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != metadata["bytes"]
        or _sha256(path) != metadata["sha256"]
    ):
        raise RagSftV2HnReviewPackageError(f"{description}身份已变化")
    return path


def _index_unique(
    rows: list[dict[str, Any]], key: str, description: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(rows, start=1):
        value = row.get(key)
        if not isinstance(value, str) or not value.strip() or value in result:
            raise RagSftV2HnReviewPackageError(
                f"{description}第 {position} 条 {key} 无效或重复"
            )
        result[value] = row
    return result


def _load_canonical(oracle_manifest: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, object]]]:
    inputs = oracle_manifest.get("inputs")
    batches = inputs.get("canonical_batches") if isinstance(inputs, dict) else None
    if (
        oracle_manifest.get("pipeline") != "rag_sft_v2_oracle_clean_materialization"
        or oracle_manifest.get("release_status") != "stage_5_oracle_clean_audited"
        or not isinstance(batches, list)
        or not batches
    ):
        raise RagSftV2HnReviewPackageError("Oracle clean manifest 身份无效")
    rows: list[dict[str, Any]] = []
    identities: list[dict[str, object]] = []
    for index, batch in enumerate(batches, start=1):
        metadata = batch.get("canonical_authoring") if isinstance(batch, dict) else None
        path = _verify_identity(metadata, f"canonical batch {index}")
        batch_rows = _load_jsonl(path, f"canonical batch {index}")
        if isinstance(metadata, dict) and metadata.get("records") != len(batch_rows):
            raise RagSftV2HnReviewPackageError(f"canonical batch {index} 记录数不一致")
        rows.extend(batch_rows)
        identities.append(_identity(path, len(batch_rows)))
    return _index_unique(rows, "query_id", "canonical authoring"), identities


def _index_articles(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for position, row in enumerate(rows, start=1):
        chunk_id = row.get("chunk_id")
        law_name = row.get("law_name")
        article_no = row.get("article_no")
        content = row.get("content")
        if (
            not isinstance(chunk_id, str)
            or not chunk_id
            or chunk_id in result
            or not isinstance(law_name, str)
            or not law_name
            or not isinstance(article_no, str)
            or not article_no
            or not isinstance(content, str)
            or not content
        ):
            raise RagSftV2HnReviewPackageError(f"法条索引第 {position} 条无效")
        result[chunk_id] = {
            "chunk_id": chunk_id,
            "law_name": law_name,
            "article_no": article_no,
            "content": content,
        }
    return result


def _derived_required(canonical: dict[str, Any]) -> list[str]:
    claims = canonical.get("claims")
    if not isinstance(claims, list) or not claims:
        raise RagSftV2HnReviewPackageError("canonical claims 无效")
    result: list[str] = []
    for claim in claims:
        support = claim.get("support") if isinstance(claim, dict) else None
        if not isinstance(support, list) or not support:
            raise RagSftV2HnReviewPackageError("canonical claim support 无效")
        for item in support:
            chunk_id = item.get("chunk_id") if isinstance(item, dict) else None
            if not isinstance(chunk_id, str) or not chunk_id:
                raise RagSftV2HnReviewPackageError("canonical support chunk_id 无效")
            if chunk_id not in result:
                result.append(chunk_id)
    return result


def _priority_key(candidate: dict[str, Any]) -> tuple[int, int, int, str]:
    visible = candidate["visible_chunk_ids"]
    required = set(candidate["required_chunk_ids"])
    gt_positions = [index for index, chunk_id in enumerate(visible) if chunk_id in required]
    first_gt_position = min(gt_positions)
    interleaved = any(
        visible[index] not in required and any(
            later in required for later in visible[index + 1 :]
        )
        for index in range(len(visible))
    )
    non_gt_count = len(candidate["non_gt_chunk_ids"])
    return (
        0 if first_gt_position > 0 else 1,
        0 if interleaved else 1,
        non_gt_count,
        candidate["query_id"],
    )


def _work_item(
    candidate: dict[str, Any],
    canonical: dict[str, Any],
    articles: dict[str, dict[str, str]],
) -> dict[str, object]:
    visible = candidate.get("visible_chunk_ids")
    required = candidate.get("required_chunk_ids")
    non_gt = candidate.get("non_gt_chunk_ids")
    if (
        not isinstance(visible, list)
        or not 1 <= len(visible) <= 5
        or len(visible) != len(set(visible))
        or not isinstance(required, list)
        or not set(required).issubset(visible)
        or not isinstance(non_gt, list)
        or non_gt != [chunk_id for chunk_id in visible if chunk_id not in set(required)]
    ):
        raise RagSftV2HnReviewPackageError(
            f"HN 候选结构无效: {candidate.get('query_id')}"
        )
    derived = _derived_required(canonical)
    if derived != required or canonical.get("query_original") != candidate.get("query_original", canonical.get("query_original")):
        raise RagSftV2HnReviewPackageError(
            f"HN 候选与 canonical 不闭合: {candidate.get('query_id')}"
        )
    evidence: list[dict[str, object]] = []
    retrieved = candidate.get("retrieved_candidates")
    retrieved_by_id = {
        item.get("chunk_id"): item
        for item in retrieved
        if isinstance(item, dict) and isinstance(item.get("chunk_id"), str)
    } if isinstance(retrieved, list) else {}
    required_set = set(required)
    for position, chunk_id in enumerate(visible, start=1):
        article = articles.get(chunk_id)
        if article is None:
            raise RagSftV2HnReviewPackageError(f"候选法条不存在: {chunk_id}")
        scores = retrieved_by_id.get(chunk_id, {})
        evidence.append(
            {
                **article,
                "evidence_id": f"E{position}",
                "is_required_gt": chunk_id in required_set,
                "rrf_rank": scores.get("rrf_rank"),
                "rrf_score": scores.get("rrf_score"),
                "rerank_score": scores.get("rerank_score"),
            }
        )
    gt_positions = [index for index, chunk_id in enumerate(visible, start=1) if chunk_id in required_set]
    non_gt_positions = [index for index, chunk_id in enumerate(visible, start=1) if chunk_id not in required_set]
    return {
        "variant_id": candidate["variant_id"],
        "query_id": candidate["query_id"],
        "query_original": canonical["query_original"],
        "claims": canonical["claims"],
        "summary": canonical["summary"],
        "required_chunk_ids": required,
        "visible_chunk_ids": visible,
        "evidence": evidence,
        "priority_features": {
            "first_gt_position": min(gt_positions),
            "gt_positions": gt_positions,
            "non_gt_positions": non_gt_positions,
            "non_gt_count": len(non_gt),
            "has_non_gt_before_gt": min(non_gt_positions) < min(gt_positions),
            "has_interleaved_non_gt": any(
                position < max(gt_positions) for position in non_gt_positions
            ),
        },
        "review_template": {
            "non_gt_reviews": [
                {
                    "chunk_id": chunk_id,
                    "label": "pending",
                    "confusion_basis": "",
                    "decisive_mismatch": "",
                    "textual_basis": [],
                    "reason": "",
                }
                for chunk_id in non_gt
            ],
            "package_decision": "pending",
            "package_reason": "",
        },
    }


def _jsonl_payload(rows: list[dict[str, object]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    )


def _publish(output_dir: Path, payloads: list[tuple[str, str]]) -> None:
    if output_dir.exists():
        raise RagSftV2HnReviewPackageError(f"输出目录必须是新目录: {output_dir}")
    output_dir.mkdir(parents=True)
    partials: list[tuple[Path, Path]] = []
    try:
        for filename, payload in payloads:
            partial = output_dir / f"{filename}.partial"
            final = output_dir / filename
            partial.write_text(payload, encoding="utf-8", newline="\n")
            partials.append((partial, final))
        hash_partial = output_dir / "manifest.sha256.partial"
        hash_partial.write_text(
            "".join(f"{_sha256(partial)}  {final.name}\n" for partial, final in partials),
            encoding="ascii",
            newline="\n",
        )
        for partial, final in partials:
            partial.replace(final)
        hash_partial.replace(output_dir / "manifest.sha256")
    except (OSError, UnicodeError) as error:
        raise RagSftV2HnReviewPackageError("无法发布 HN 审核工作包") from error


def build_hn_review_package(
    *,
    materialization_dir: Path,
    oracle_manifest_path: Path,
    article_index_path: Path,
    output_dir: Path,
    batch_size: int = BATCH_SIZE,
) -> dict[str, object]:
    """发布只读审核工作项，不生成任何语义裁决。"""

    if type(batch_size) is not int or batch_size <= 0:
        raise RagSftV2HnReviewPackageError("batch_size 必须是正整数")
    materialization_dir = Path(materialization_dir).resolve()
    oracle_manifest_path = Path(oracle_manifest_path).resolve()
    article_index_path = Path(article_index_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RagSftV2HnReviewPackageError(f"输出目录必须是新目录: {output_dir}")

    materialization_report_path = materialization_dir / "report.json"
    candidates_path = materialization_dir / "hn-candidates.jsonl"
    report = _load_json(materialization_report_path, "retrieved materialization report")
    candidates = _load_jsonl(candidates_path, "HN 结构候选")
    if (
        report.get("pipeline") != "rag_sft_v2_retrieved_materialization"
        or report.get("release_status") != "stage_7_retrieved_candidates"
        or report.get("records", {}).get("candidates") != len(candidates)
    ):
        raise RagSftV2HnReviewPackageError("retrieved materialization 身份无效")
    oracle_manifest = _load_json(oracle_manifest_path, "Oracle clean manifest")
    canonical, canonical_identities = _load_canonical(oracle_manifest)
    article_rows = _load_jsonl(article_index_path, "法条索引")
    articles = _index_articles(article_rows)
    indexed_candidates = _index_unique(candidates, "query_id", "HN 结构候选")
    missing = set(indexed_candidates) - set(canonical)
    if missing:
        raise RagSftV2HnReviewPackageError(f"HN 候选缺少 canonical: {sorted(missing)[0]}")

    ordered_candidates = sorted(candidates, key=_priority_key)
    work_items = [
        _work_item(candidate, canonical[candidate["query_id"]], articles)
        for candidate in ordered_candidates
    ]
    batches = [
        work_items[offset : offset + batch_size]
        for offset in range(0, len(work_items), batch_size)
    ]
    priority_counts = Counter(
        "gt_not_first" if item["priority_features"]["first_gt_position"] > 1 else "gt_first"
        for item in work_items
    )
    payloads: list[tuple[str, str]] = [("work-items.jsonl", _jsonl_payload(work_items))]
    batch_metadata: list[dict[str, object]] = []
    for index, batch in enumerate(batches, start=1):
        filename = f"review-batch-{index:02d}.jsonl"
        payloads.append((filename, _jsonl_payload(batch)))
        batch_metadata.append(
            {
                "batch": index,
                "filename": filename,
                "records": len(batch),
                "first_query_id": batch[0]["query_id"],
                "last_query_id": batch[-1]["query_id"],
                "gt_not_first": sum(
                    item["priority_features"]["first_gt_position"] > 1 for item in batch
                ),
            }
        )
    manifest: dict[str, object] = {
        "pipeline": "rag_sft_v2_hn_semantic_review_packaging",
        "release_status": "semantic_review_work_items",
        "inputs": {
            "materialization_report": _identity(materialization_report_path),
            "hn_candidates": _identity(candidates_path, len(candidates)),
            "oracle_manifest": _identity(oracle_manifest_path),
            "canonical_authoring": canonical_identities,
            "article_index": _identity(article_index_path, len(article_rows)),
        },
        "policy": {
            "semantic_labels_emitted": False,
            "full_article_text_included": True,
            "retrieval_order_preserved": True,
            "priority_order": ["gt_not_first", "interleaved", "fewer_non_gt", "query_id"],
            "batch_size": batch_size,
        },
        "records": {
            "candidates": len(work_items),
            "batches": len(batches),
            "priority": dict(sorted(priority_counts.items())),
            "non_gt_evidence": sum(
                item["priority_features"]["non_gt_count"] for item in work_items
            ),
        },
        "batches": batch_metadata,
        "readiness": {
            "semantic_review_work_items_ready": True,
            "hn_semantic_review_complete": False,
            "training_ready": False,
        },
        "complete": True,
    }
    payloads.append(("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"))
    _publish(output_dir, payloads)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-dir", type=Path, default=DEFAULT_MATERIALIZATION_DIR)
    parser.add_argument("--oracle-manifest", type=Path, default=DEFAULT_ORACLE_MANIFEST)
    parser.add_argument("--article-index", type=Path, default=DEFAULT_ARTICLE_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    try:
        manifest = build_hn_review_package(
            materialization_dir=args.materialization_dir,
            oracle_manifest_path=args.oracle_manifest,
            article_index_path=args.article_index,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
    except RagSftV2HnReviewPackageError as error:
        parser.error(str(error))
    print(json.dumps(manifest["records"], ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = ["RagSftV2HnReviewPackageError", "build_hn_review_package"]
