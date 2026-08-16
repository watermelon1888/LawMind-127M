"""发布经语义与 GT 兼容性审核后的 Query-SFT pilot 候选池。"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .audit_query_sft_pilot_inputs import (
        DEFAULT_AUDIT_FILENAME,
        DEFAULT_DRAFT_FILENAME,
    )
    from .prepare_query_sft_pilot import (
        AUDIT_REFERENCE_FILENAME,
        BLIND_QUEUE_FILENAME,
        DEFAULT_OUTPUT_DIR,
        HASH_FILENAME as WORK_PACKAGE_HASH_FILENAME,
        MANIFEST_FILENAME as WORK_PACKAGE_MANIFEST_FILENAME,
    )
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset.audit_query_sft_pilot_inputs import (
        DEFAULT_AUDIT_FILENAME,
        DEFAULT_DRAFT_FILENAME,
    )
    from dataset.prepare_query_sft_pilot import (
        AUDIT_REFERENCE_FILENAME,
        BLIND_QUEUE_FILENAME,
        DEFAULT_OUTPUT_DIR,
        HASH_FILENAME as WORK_PACKAGE_HASH_FILENAME,
        MANIFEST_FILENAME as WORK_PACKAGE_MANIFEST_FILENAME,
    )


DEFAULT_REVIEW_LEDGER_FILENAME = "query-sft-pilot-v1-input-semantic-gt-review.jsonl"
DEFAULT_MANIFEST_FILENAME = "query-sft-pilot-v1-input-review-manifest.json"
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / DEFAULT_MANIFEST_FILENAME

_BLIND_FIELDS = {"pilot_id", "source_id", "authoring_type", "source_query"}
_DRAFT_FIELDS = {"pilot_id", "source_id", "authoring_type", "query_original"}
_REFERENCE_FIELDS = {
    "pilot_id",
    "source_id",
    "authoring_type",
    "coverage_domain",
    "source_query",
    "required_chunk_ids",
    "source_record_sha256",
}
_REVIEW_FIELDS = {"pilot_id", "source_id", "review_decision", "reason"}
_REVIEW_DECISIONS = {"approved", "rejected"}


class QuerySftPilotInputReviewError(RuntimeError):
    """Query-SFT pilot 输入审核材料不完整或彼此不一致。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if records is not None:
        result["records"] = records
    return result


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftPilotInputReviewError(
            f"无法读取{description}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise QuerySftPilotInputReviewError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise QuerySftPilotInputReviewError(
                        f"{description}不允许空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise QuerySftPilotInputReviewError(
                        f"{description}第 {line_number} 条必须是 object"
                    )
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftPilotInputReviewError):
            raise
        raise QuerySftPilotInputReviewError(
            f"无法读取{description}: {path}"
        ) from error
    if not records:
        raise QuerySftPilotInputReviewError(f"{description}不能为空")
    return records


def _verify_adjacent_hash(path: Path, description: str) -> Path:
    hash_path = path.with_suffix(".sha256")
    expected = f"{_sha256_file(path)}  {path.name}"
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftPilotInputReviewError(
            f"无法读取{description}相邻 SHA-256: {hash_path}"
        ) from error
    if lines != [expected]:
        raise QuerySftPilotInputReviewError(f"{description}相邻 SHA-256 无效")
    return hash_path


def _verify_work_package_hashes(directory: Path) -> dict[str, Path]:
    paths = {
        "manifest": directory / WORK_PACKAGE_MANIFEST_FILENAME,
        "blind_queue": directory / BLIND_QUEUE_FILENAME,
        "audit_reference": directory / AUDIT_REFERENCE_FILENAME,
    }
    hash_path = directory / WORK_PACKAGE_HASH_FILENAME
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftPilotInputReviewError(
            "无法读取 pilot 工作包 SHA-256"
        ) from error
    actual = {}
    for line in lines:
        digest, separator, name = line.partition("  ")
        if not separator or not digest or not name or name in actual:
            raise QuerySftPilotInputReviewError("pilot 工作包 SHA-256 格式无效")
        actual[name] = digest
    if set(actual) != {path.name for path in paths.values()}:
        raise QuerySftPilotInputReviewError("pilot 工作包 SHA-256 条目不完整")
    for path in paths.values():
        if actual[path.name] != _sha256_file(path):
            raise QuerySftPilotInputReviewError(
                f"pilot 工作包身份已变化: {path.name}"
            )
    paths["hash_manifest"] = hash_path
    return paths


def _normalize_reason(value: object, position: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuerySftPilotInputReviewError(f"审核账本第 {position} 条 reason 无效")
    normalized = unicodedata.normalize("NFC", value.strip())
    if normalized != value or "\n" in value or "\r" in value:
        raise QuerySftPilotInputReviewError(
            f"审核账本第 {position} 条 reason 必须为 NFC 单行文本"
        )
    return normalized


def _validate_review_ledger(
    records: list[dict[str, Any]], *, expected_by_pilot: dict[str, dict[str, Any]]
) -> dict[str, dict[str, str]]:
    by_pilot: dict[str, dict[str, str]] = {}
    for position, record in enumerate(records, start=1):
        if set(record) != _REVIEW_FIELDS:
            raise QuerySftPilotInputReviewError(
                f"审核账本第 {position} 条字段必须精确匹配 schema"
            )
        pilot_id = record.get("pilot_id")
        source_id = record.get("source_id")
        decision = record.get("review_decision")
        if pilot_id not in expected_by_pilot or pilot_id in by_pilot:
            raise QuerySftPilotInputReviewError(
                f"审核账本 pilot_id 不存在或重复: {pilot_id}"
            )
        if source_id != expected_by_pilot[pilot_id]["source_id"]:
            raise QuerySftPilotInputReviewError(
                f"审核账本 source_id 与输入不一致: {pilot_id}"
            )
        if decision not in _REVIEW_DECISIONS:
            raise QuerySftPilotInputReviewError(
                f"审核账本 review_decision 无效: {pilot_id}"
            )
        by_pilot[pilot_id] = {
            "source_id": source_id,
            "review_decision": decision,
            "reason": _normalize_reason(record.get("reason"), position),
        }
    if set(by_pilot) != set(expected_by_pilot):
        missing = sorted(set(expected_by_pilot) - set(by_pilot))
        raise QuerySftPilotInputReviewError(f"审核账本缺少输入记录: {missing}")
    return by_pilot


def finalize_query_sft_pilot_input_review(
    *,
    work_package_dir: Path,
    draft_path: Path,
    input_audit_path: Path,
    review_ledger_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """绑定已审核输入并发布可用于教师候选构造的 39 条候选池。"""

    work_package_dir = Path(work_package_dir).resolve()
    draft_path = Path(draft_path).resolve()
    input_audit_path = Path(input_audit_path).resolve()
    review_ledger_path = Path(review_ledger_path).resolve()
    output_path = Path(output_path).resolve()
    output_hash_path = output_path.with_suffix(".sha256")
    if output_path.exists() or output_hash_path.exists():
        raise QuerySftPilotInputReviewError(f"发布输出已存在: {output_path}")

    package_paths = _verify_work_package_hashes(work_package_dir)
    draft_hash_path = _verify_adjacent_hash(draft_path, "实际输入草稿")
    audit_hash_path = _verify_adjacent_hash(input_audit_path, "输入结构审计报告")
    review_hash_path = _verify_adjacent_hash(review_ledger_path, "语义与 GT 审核账本")
    manifest = _load_json(package_paths["manifest"], "pilot 工作包 manifest")
    if (
        manifest.get("pipeline") != "query_sft_pilot_work_package_v1"
        or manifest.get("release_status") != "pilot_authoring_work_package"
        or manifest.get("complete") is not True
        or manifest.get("readiness", {}).get("blind_authoring_queue_ready") is not True
    ):
        raise QuerySftPilotInputReviewError("pilot 工作包状态无效")

    blind = _load_jsonl(package_paths["blind_queue"], "盲构造队列")
    reference = _load_jsonl(package_paths["audit_reference"], "GT 审核引用")
    draft = _load_jsonl(draft_path, "实际输入草稿")
    if len(blind) != len(reference) or len(blind) != len(draft):
        raise QuerySftPilotInputReviewError("盲构造、GT 审核引用与实际输入数量不一致")
    if manifest.get("records", {}).get("selected") != len(blind):
        raise QuerySftPilotInputReviewError("pilot 工作包数量与 manifest 不一致")

    draft_by_pilot: dict[str, dict[str, Any]] = {}
    for position, (blind_record, reference_record, draft_record) in enumerate(
        zip(blind, reference, draft, strict=True), start=1
    ):
        if set(blind_record) != _BLIND_FIELDS:
            raise QuerySftPilotInputReviewError(f"盲构造队列第 {position} 条字段无效")
        if set(reference_record) != _REFERENCE_FIELDS:
            raise QuerySftPilotInputReviewError(f"GT 审核引用第 {position} 条字段无效")
        if set(draft_record) != _DRAFT_FIELDS:
            raise QuerySftPilotInputReviewError(f"实际输入草稿第 {position} 条字段无效")
        pilot_id = blind_record.get("pilot_id")
        source_id = blind_record.get("source_id")
        if (
            not isinstance(pilot_id, str)
            or pilot_id in draft_by_pilot
            or reference_record.get("pilot_id") != pilot_id
            or draft_record.get("pilot_id") != pilot_id
            or reference_record.get("source_id") != source_id
            or draft_record.get("source_id") != source_id
            or draft_record.get("authoring_type") != blind_record.get("authoring_type")
            or reference_record.get("authoring_type") != blind_record.get("authoring_type")
        ):
            raise QuerySftPilotInputReviewError(
                f"第 {position} 条 pilot 输入与工作包映射不一致"
            )
        draft_by_pilot[pilot_id] = draft_record

    input_audit = _load_json(input_audit_path, "输入结构审计报告")
    if (
        input_audit.get("pipeline") != "query_sft_pilot_input_audit_v1"
        or input_audit.get("complete") is not True
        or input_audit.get("readiness", {}).get("query_input_draft_structural_ready")
        is not True
        or input_audit.get("validation", {}).get("required_gt_not_emitted") is not True
        or input_audit.get("records", {}).get("draft") != len(draft)
        or input_audit.get("records", {}).get("evaluation_question_overlap") != 0
        or input_audit.get("records", {}).get("other_source_query_collisions") != 0
    ):
        raise QuerySftPilotInputReviewError("输入结构审计报告状态无效")
    audit_draft = input_audit.get("inputs", {}).get("draft")
    if not isinstance(audit_draft, dict) or (
        audit_draft.get("sha256") != _sha256_file(draft_path)
        or audit_draft.get("bytes") != draft_path.stat().st_size
        or audit_draft.get("records") != len(draft)
    ):
        raise QuerySftPilotInputReviewError("输入结构审计报告未绑定当前实际输入草稿")

    review = _validate_review_ledger(
        _load_jsonl(review_ledger_path, "语义与 GT 审核账本"),
        expected_by_pilot=draft_by_pilot,
    )
    approved = [pilot_id for pilot_id, item in review.items() if item["review_decision"] == "approved"]
    rejected = [pilot_id for pilot_id, item in review.items() if item["review_decision"] == "rejected"]
    if not approved or not rejected:
        raise QuerySftPilotInputReviewError("pilot 必须同时保留通过和拒绝审核轨迹")

    candidate_pool = [
        {
            "pilot_id": pilot_id,
            "query_original": draft_by_pilot[pilot_id]["query_original"],
        }
        for pilot_id in approved
    ]
    output: dict[str, object] = {
        "pipeline": "query_sft_pilot_input_review_v1",
        "release_status": "pilot_teacher_candidate_input",
        "inputs": {
            "work_package_manifest": _identity(package_paths["manifest"]),
            "work_package_hash_manifest": _identity(package_paths["hash_manifest"]),
            "blind_authoring_queue": _identity(package_paths["blind_queue"], records=len(blind)),
            "audit_reference": _identity(package_paths["audit_reference"], records=len(reference)),
            "draft": {**_identity(draft_path, records=len(draft)), "hash_manifest": _identity(draft_hash_path)},
            "input_audit": {**_identity(input_audit_path), "hash_manifest": _identity(audit_hash_path)},
            "semantic_gt_review": {**_identity(review_ledger_path, records=len(review)), "hash_manifest": _identity(review_hash_path)},
        },
        "policy": {
            "teacher_visible_fields": ["pilot_id", "query_original"],
            "teacher_cannot_read_required_gt": True,
            "review_rejected_records_excluded": True,
            "candidate_pool_omits_source_id_authoring_type_and_required_gt": True,
        },
        "records": {
            "reviewed": len(review),
            "approved": len(approved),
            "rejected": len(rejected),
            "approved_pilot_ids": approved,
            "rejected_pilot_ids": rejected,
            "teacher_candidate_inputs": candidate_pool,
            "review_decision_counts": dict(sorted(Counter(item["review_decision"] for item in review.values()).items())),
        },
        "validation": {
            "work_package_identity_bound": True,
            "draft_identity_bound": True,
            "input_audit_identity_bound": True,
            "semantic_gt_review_identity_bound": True,
            "every_input_reviewed_once": True,
            "review_source_mapping_matches_input": True,
            "input_structural_audit_passed": True,
            "evaluation_isolation_passed": True,
            "teacher_candidate_inputs_exclude_gt": True,
        },
        "readiness": {
            "query_input_semantic_gt_review_complete": True,
            "teacher_candidate_work_package_ready": True,
            "teacher_generation_completed": False,
            "retrieval_evaluation_ready": False,
            "query_sft_training_ready": False,
        },
        "complete": True,
    }
    payload = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        output_path.write_text(payload, encoding="utf-8", newline="\n")
        output_hash_path.write_text(
            f"{_sha256_file(output_path)}  {output_path.name}\n",
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, UnicodeError) as error:
        output_path.unlink(missing_ok=True)
        output_hash_path.unlink(missing_ok=True)
        raise QuerySftPilotInputReviewError("无法发布输入审核 manifest") from error
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-package-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--draft", type=Path, default=DEFAULT_OUTPUT_DIR / DEFAULT_DRAFT_FILENAME
    )
    parser.add_argument(
        "--input-audit", type=Path, default=DEFAULT_OUTPUT_DIR / DEFAULT_AUDIT_FILENAME
    )
    parser.add_argument(
        "--review-ledger", type=Path, default=DEFAULT_OUTPUT_DIR / DEFAULT_REVIEW_LEDGER_FILENAME
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    try:
        result = finalize_query_sft_pilot_input_review(
            work_package_dir=args.work_package_dir,
            draft_path=args.draft,
            input_audit_path=args.input_audit,
            review_ledger_path=args.review_ledger,
            output_path=args.output,
        )
    except QuerySftPilotInputReviewError as error:
        parser.error(str(error))
    print(
        f"[完成] 输入审核绑定 reviewed={result['records']['reviewed']} "
        f"approved={result['records']['approved']} rejected={result['records']['rejected']}"
    )
    print(f"输入审核 manifest: {args.output}")


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_MANIFEST_FILENAME",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_REVIEW_LEDGER_FILENAME",
    "QuerySftPilotInputReviewError",
    "finalize_query_sft_pilot_input_review",
]
