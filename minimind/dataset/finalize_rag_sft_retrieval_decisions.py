"""严格汇编 retrieved RAG-SFT 的确定性、双审共识与用户终裁。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from . import adjudicate_rag_sft_retrieved as adjudicator
    from . import build_rag_sft_retrieval_review_queue as queue_builder
    from . import validate_rag_sft_retrieval_full_reviews as full_validator
    from . import validate_rag_sft_retrieval_pilot_reviews as review_validator
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import adjudicate_rag_sft_retrieved as adjudicator
    from dataset import build_rag_sft_retrieval_review_queue as queue_builder
    from dataset import validate_rag_sft_retrieval_full_reviews as full_validator
    from dataset import validate_rag_sft_retrieval_pilot_reviews as review_validator


RAG_SFT_ROOT = Path(__file__).resolve().parent / "RAG-SFT"
DEFAULT_REVIEW_DIR = RAG_SFT_ROOT / "retrieved" / "review-canonical-v1-20260809"

FINAL_DECISIONS_FILENAME = "rag-sft-retrieval-final-decisions.jsonl"
REPORT_FILENAME = "rag-sft-retrieval-final-decisions.json"
HASH_FILENAME = "rag-sft-retrieval-final-decisions.sha256"

EXPECTED_LOCATORS = 574
EXPECTED_DETERMINISTIC = 79
EXPECTED_SEMANTIC = 495
_SHA256_RE = re.compile(r"[0-9a-f]{64}$")
_WRAPPED_FIELDS = {"source_id", "derived_id", "locator_sha256", "decision"}
_CONSENSUS_FIELDS = _WRAPPED_FIELDS | {"consensus_mode"}
_USER_REVIEW_FIELDS = {
    "source_id",
    "derived_id",
    "locator_sha256",
    "lane",
    "reason_codes",
    "legal_support_review",
    "adversarial_boundary_review",
}
_ESCALATION_REASON_CODES = {
    "target_conflict",
    "alternative_complete_answer",
    "review_disagreement",
}


class RagSftRetrievalFinalizationError(RuntimeError):
    """retrieved 决策输入不闭合、身份失配或无法安全发布。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
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
        raise RagSftRetrievalFinalizationError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise RagSftRetrievalFinalizationError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(path: Path, description: str, *, allow_empty: bool = False) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagSftRetrievalFinalizationError(
                        f"{description}不允许空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RagSftRetrievalFinalizationError(
                        f"{description}第 {line_number} 条必须是 object"
                    )
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftRetrievalFinalizationError):
            raise
        raise RagSftRetrievalFinalizationError(f"无法读取{description}: {path}") from error
    if not records and not allow_empty:
        raise RagSftRetrievalFinalizationError(f"{description}不能为空")
    return records


def _unique_by(records: list[dict[str, Any]], field: str, description: str) -> dict[str, dict[str, Any]]:
    result = {}
    for position, record in enumerate(records, start=1):
        value = record.get(field)
        if not isinstance(value, str) or not value or value in result:
            raise RagSftRetrievalFinalizationError(
                f"{description}第 {position} 条 {field} 无效或重复"
            )
        result[value] = record
    return result


def _verify_identity(metadata: object, path: Path, description: str, *, records: int | None = None) -> None:
    if not isinstance(metadata, dict):
        raise RagSftRetrievalFinalizationError(f"{description}身份信息无效")
    expected = _identity(path, records=records)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RagSftRetrievalFinalizationError(f"{description}身份不一致")


def _verify_hash_manifest(directory: Path, filename: str, paths: dict[str, Path], description: str) -> None:
    try:
        lines = (directory / filename).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftRetrievalFinalizationError(f"无法读取{description} SHA-256 清单") from error
    parsed = {}
    for line_number, line in enumerate(lines, start=1):
        parts = line.split("  ", 1)
        if (
            len(parts) != 2
            or not _SHA256_RE.fullmatch(parts[0])
            or not parts[1]
            or Path(parts[1]).name != parts[1]
            or parts[1] in parsed
        ):
            raise RagSftRetrievalFinalizationError(
                f"{description} SHA-256 清单格式无效: {line_number}"
            )
        parsed[parts[1]] = parts[0]
    if set(parsed) != set(paths):
        raise RagSftRetrievalFinalizationError(f"{description} SHA-256 文件集合无效")
    for name, path in paths.items():
        if not path.is_file() or parsed[name] != _sha256(path):
            raise RagSftRetrievalFinalizationError(
                f"{description}文件身份已变化: {name}"
            )


def _verify_wrapped_decisions(
    records: list[dict[str, Any]],
    *,
    locators: dict[str, dict[str, Any]],
    description: str,
    expected_fields: set[str] = _WRAPPED_FIELDS,
) -> dict[str, dict[str, Any]]:
    by_id = _unique_by(records, "source_id", description)
    decisions = {}
    for source_id, record in by_id.items():
        if set(record) != expected_fields:
            raise RagSftRetrievalFinalizationError(f"{description} {source_id} 包装字段无效")
        locator = locators.get(source_id)
        if locator is None:
            raise RagSftRetrievalFinalizationError(f"{description}引用未知 locator: {source_id}")
        if record["derived_id"] != locator.get("derived_id"):
            raise RagSftRetrievalFinalizationError(f"{description} {source_id} derived_id 不一致")
        if record["locator_sha256"] != queue_builder.canonical_digest(locator):
            raise RagSftRetrievalFinalizationError(f"{description} {source_id} locator 哈希不一致")
        decision = record["decision"]
        if not isinstance(decision, dict) or decision.get("source_id") != source_id:
            raise RagSftRetrievalFinalizationError(f"{description} {source_id} decision 身份无效")
        decisions[source_id] = decision
    return decisions


def _verify_review_bundle(review_dir: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    review_dir = Path(review_dir).resolve()
    queue_path = review_dir / queue_builder.QUEUE_FILENAME
    try:
        manifest_path, manifest = review_validator._verify_review_bundle(
            queue_path, expected_filename=queue_builder.QUEUE_FILENAME
        )
    except review_validator.RagSftRetrievalPilotReviewError as error:
        raise RagSftRetrievalFinalizationError(str(error)) from error
    records = manifest.get("records", {})
    if (
        records.get("locators") != EXPECTED_LOCATORS
        or records.get("deterministic_decisions") != EXPECTED_DETERMINISTIC
        or records.get("semantic_review_queue") != EXPECTED_SEMANTIC
    ):
        raise RagSftRetrievalFinalizationError("review bundle 固定数量合同不一致")
    queue = _load_jsonl(queue_path, "全量 review queue")
    if len(queue) != EXPECTED_SEMANTIC:
        raise RagSftRetrievalFinalizationError("review queue 数量不符合固定合同")
    queue_by_id = _unique_by(queue, "source_id", "全量 review queue")
    locator_metadata = manifest.get("input", {}).get("locators")
    if not isinstance(locator_metadata, dict) or not isinstance(locator_metadata.get("path"), str):
        raise RagSftRetrievalFinalizationError("review bundle 缺少 locator 身份")
    locator_path = Path(locator_metadata["path"]).resolve()
    locators = _load_jsonl(locator_path, "retrieved locator")
    if len(locators) != EXPECTED_LOCATORS:
        raise RagSftRetrievalFinalizationError("locator 数量不符合固定合同")
    _verify_identity(locator_metadata, locator_path, "review bundle locator", records=len(locators))
    locator_by_id = _unique_by(locators, "source_id", "retrieved locator")
    deterministic_path = review_dir / queue_builder.DETERMINISTIC_FILENAME
    deterministic_wrapped = _load_jsonl(deterministic_path, "确定性决定")
    if len(deterministic_wrapped) != EXPECTED_DETERMINISTIC:
        raise RagSftRetrievalFinalizationError("确定性决定数量不符合固定合同")
    deterministic = _verify_wrapped_decisions(
        deterministic_wrapped, locators=locator_by_id, description="确定性决定"
    )
    if set(deterministic) & set(queue_by_id) or set(deterministic) | set(queue_by_id) != set(locator_by_id):
        raise RagSftRetrievalFinalizationError("确定性决定、语义队列与 locator 未精确闭合")
    return manifest_path, manifest, locators, locator_by_id, queue, deterministic


def _verify_full_review_bundle(
    full_review_dir: Path,
    *,
    review_manifest_path: Path,
    review_manifest: dict[str, Any],
    queue: list[dict[str, Any]],
    locators: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Path]]:
    directory = Path(full_review_dir).resolve()
    paths = {
        full_validator.CONSENSUS_FILENAME: directory / full_validator.CONSENSUS_FILENAME,
        full_validator.DECISIONS_FILENAME: directory / full_validator.DECISIONS_FILENAME,
        full_validator.USER_REVIEW_FILENAME: directory / full_validator.USER_REVIEW_FILENAME,
        full_validator.REPORT_FILENAME: directory / full_validator.REPORT_FILENAME,
    }
    _verify_hash_manifest(directory, full_validator.HASH_FILENAME, paths, "full-review bundle")
    report = _load_json(paths[full_validator.REPORT_FILENAME], "full-review report")
    if report.get("pipeline") != "rag_sft_retrieval_full_review_validation" or report.get("complete") is not True:
        raise RagSftRetrievalFinalizationError("full-review report 状态无效")
    _verify_identity(report.get("input", {}).get("review_bundle_manifest"), review_manifest_path, "full-review review manifest")
    _verify_identity(report.get("input", {}).get("review_queue"), review_manifest_path.parent / queue_builder.QUEUE_FILENAME, "full-review review queue", records=len(queue))
    records = report.get("records", {})
    if records.get("review_queue") != EXPECTED_SEMANTIC:
        raise RagSftRetrievalFinalizationError("full-review queue 数量不符合固定合同")
    consensus_wrapped = _load_jsonl(paths[full_validator.CONSENSUS_FILENAME], "full-review 共识", allow_empty=True)
    consensus = _verify_wrapped_decisions(
        consensus_wrapped,
        locators=locators,
        description="full-review 共识",
        expected_fields=_CONSENSUS_FIELDS,
    )
    flat_consensus = _load_jsonl(paths[full_validator.DECISIONS_FILENAME], "full-review 扁平决定", allow_empty=True)
    if flat_consensus != [item["decision"] for item in consensus_wrapped]:
        raise RagSftRetrievalFinalizationError("full-review 扁平决定与共识包装不一致")
    user_review = _load_jsonl(paths[full_validator.USER_REVIEW_FILENAME], "full-review 用户终裁队列", allow_empty=True)
    queue_by_id = _unique_by(queue, "source_id", "全量 review queue")
    user_by_id = _unique_by(user_review, "source_id", "full-review 用户终裁队列") if user_review else {}
    for position, (source_id, item) in enumerate(user_by_id.items(), start=1):
        if set(item) != _USER_REVIEW_FIELDS:
            raise RagSftRetrievalFinalizationError(
                f"用户终裁队列 {source_id} 顶层字段无效"
            )
        queue_item = queue_by_id.get(source_id)
        if queue_item is None:
            raise RagSftRetrievalFinalizationError(f"用户终裁队列引用未知 source_id: {source_id}")
        if (
            item.get("derived_id") != queue_item.get("derived_id")
            or item.get("locator_sha256") != queue_item.get("locator_sha256")
            or item.get("lane") != queue_item.get("lane")
        ):
            raise RagSftRetrievalFinalizationError(f"用户终裁队列 {source_id} 未绑定当前 review queue")
        reason_codes = item.get("reason_codes")
        if (
            not isinstance(reason_codes, list)
            or not reason_codes
            or len(reason_codes) != len(set(reason_codes))
            or not set(reason_codes).issubset(_ESCALATION_REASON_CODES)
        ):
            raise RagSftRetrievalFinalizationError(f"用户终裁队列 {source_id} reason_codes 无效")
        for role, field in (
            ("legal_support", "legal_support_review"),
            ("adversarial_boundary", "adversarial_boundary_review"),
        ):
            review = item[field]
            if not isinstance(review, dict):
                raise RagSftRetrievalFinalizationError(
                    f"用户终裁队列 {source_id} {field} 无效"
                )
            try:
                review_validator._validate_review(
                    review,
                    position=position,
                    expected_role=role,
                    pilot_by_id=queue_by_id,
                )
            except review_validator.RagSftRetrievalPilotReviewError as error:
                raise RagSftRetrievalFinalizationError(str(error)) from error
    if set(consensus) & set(user_by_id) or set(consensus) | set(user_by_id) != set(queue_by_id):
        raise RagSftRetrievalFinalizationError("full-review 共识与用户终裁队列未精确闭合")
    if (
        records.get("consensus") != len(consensus)
        or records.get("user_review_required") != len(user_review)
        or len(consensus) + len(user_review) != EXPECTED_SEMANTIC
    ):
        raise RagSftRetrievalFinalizationError("full-review 记录数不闭合")
    return consensus, user_review, paths


def _load_user_decisions(path: Path | None, user_review: list[dict[str, Any]], locators: dict[str, dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, object] | None]:
    expected_ids = {item["source_id"] for item in user_review}
    if path is None:
        if expected_ids:
            raise RagSftRetrievalFinalizationError("用户终裁队列非空时必须提供显式六字段决定")
        return {}, None
    path = Path(path).resolve()
    records = _load_jsonl(path, "用户终裁决定", allow_empty=True)
    decisions = _unique_by(records, "source_id", "用户终裁决定") if records else {}
    if set(decisions) != expected_ids:
        raise RagSftRetrievalFinalizationError("用户终裁决定必须精确覆盖用户终裁队列")
    for source_id, decision in decisions.items():
        try:
            adjudicator._validate_decision(decision, locators[source_id])
        except adjudicator.RagSftRetrievedAdjudicationError as error:
            raise RagSftRetrievalFinalizationError(str(error)) from error
    return decisions, _identity(path, records=len(records))


def _write_jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for record in records
    )


def _publish(output_dir: Path, decisions: list[dict[str, Any]], report: dict[str, object]) -> None:
    if output_dir.exists():
        raise RagSftRetrievalFinalizationError("最终决定输出目录必须是新目录")
    output_dir.mkdir(parents=True)
    payloads = (
        (FINAL_DECISIONS_FILENAME, _write_jsonl(decisions)),
        (REPORT_FILENAME, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
    )
    partials = []
    published = []
    try:
        for filename, payload in payloads:
            partial = output_dir / f"{filename}.partial"
            final = output_dir / filename
            partial.write_text(payload, encoding="utf-8", newline="\n")
            partials.append((partial, final))
        hash_partial = output_dir / f"{HASH_FILENAME}.partial"
        hash_partial.write_text(
            "".join(f"{_sha256(partial)}  {final.name}\n" for partial, final in partials),
            encoding="utf-8",
            newline="\n",
        )
        for partial, final in partials:
            partial.replace(final)
            published.append(final)
        hash_partial.replace(output_dir / HASH_FILENAME)
    except (OSError, UnicodeError, ValueError) as error:
        for path in [*(partial for partial, _ in partials), output_dir / f"{HASH_FILENAME}.partial", *reversed(published)]:
            path.unlink(missing_ok=True)
        try:
            output_dir.rmdir()
        except OSError:
            pass
        raise RagSftRetrievalFinalizationError("无法发布最终 retrieved 决定") from error


def finalize_rag_sft_retrieval_decisions(
    *, review_dir: Path, full_review_dir: Path, user_decisions_path: Path | None, output_dir: Path
) -> dict[str, object]:
    """验证 79/495/574 决策合同，并按 locator 顺序发布扁平六字段决定。"""

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RagSftRetrievalFinalizationError("最终决定输出目录必须是新目录")
    review_manifest_path, review_manifest, locator_records, locators, queue, deterministic = _verify_review_bundle(review_dir)
    consensus, user_review, full_paths = _verify_full_review_bundle(
        full_review_dir,
        review_manifest_path=review_manifest_path,
        review_manifest=review_manifest,
        queue=queue,
        locators=locators,
    )
    user_decisions, user_identity = _load_user_decisions(user_decisions_path, user_review, locators)
    merged = {**deterministic, **consensus, **user_decisions}
    if set(merged) != set(locators):
        raise RagSftRetrievalFinalizationError("最终决定未精确覆盖全部 locator")
    ordered = []
    counts: Counter[str] = Counter()
    for locator in locator_records:
        decision = merged[locator["source_id"]]
        try:
            adjudicator._validate_decision(decision, locator)
        except adjudicator.RagSftRetrievedAdjudicationError as error:
            raise RagSftRetrievalFinalizationError(str(error)) from error
        ordered.append(decision)
        counts[decision["decision"]] += 1
    report: dict[str, object] = {
        "pipeline": "rag_sft_retrieval_decision_finalization",
        "policy": {
            "fixed_record_contract": {"locators": EXPECTED_LOCATORS, "deterministic": EXPECTED_DETERMINISTIC, "semantic": EXPECTED_SEMANTIC},
            "locator_order_preserved": True,
            "user_review_requires_explicit_six_field_decision": True,
            "authoring_emitted": False,
        },
        "input": {
            "review_bundle_manifest": _identity(review_manifest_path),
            "locators": _identity(Path(review_manifest["input"]["locators"]["path"]), records=len(locator_records)),
            "deterministic_decisions": _identity(Path(review_dir).resolve() / queue_builder.DETERMINISTIC_FILENAME, records=len(deterministic)),
            "full_review": {name: _identity(path) for name, path in full_paths.items()},
            "user_decisions": user_identity,
        },
        "records": {
            "locators": len(locator_records),
            "deterministic": len(deterministic),
            "semantic_consensus": len(consensus),
            "semantic_user_review": len(user_review),
            "published": len(ordered),
            "decisions": dict(sorted(counts.items())),
        },
        "validation": {
            "review_bundle_verified": True,
            "full_review_bundle_verified": True,
            "all_decisions_follow_adjudication_schema": True,
            "all_locators_covered_once": True,
            "output_order_matches_locator_order": True,
        },
        "outputs": {"flat_decisions": FINAL_DECISIONS_FILENAME, "report": REPORT_FILENAME, "sha256_manifest": HASH_FILENAME},
        "complete": True,
    }
    _publish(output_dir, ordered, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--full-review-dir", type=Path, required=True)
    parser.add_argument("--user-decisions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = finalize_rag_sft_retrieval_decisions(
            review_dir=args.review_dir,
            full_review_dir=args.full_review_dir,
            user_decisions_path=args.user_decisions,
            output_dir=args.output_dir,
        )
    except RagSftRetrievalFinalizationError as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["RagSftRetrievalFinalizationError", "finalize_rag_sft_retrieval_decisions"]
