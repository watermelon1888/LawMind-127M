"""审计通过独立审核的全量 Query-SFT 候选文本、去重与评估隔离。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rag.query.enhancement import parse_and_validate_query_enhancement

try:
    from .audit_query_sft_pilot_inputs import question_digest
    from .audit_query_sft_full_teacher_candidates import (
        AUDIT_FILENAME as PROTOCOL_AUDIT_FILENAME,
        DEFAULT_OUTPUT_DIR as DEFAULT_TEACHER_WORK_PACKAGE_DIR,
        RESULT_FILENAME,
    )
    from .finalize_query_sft_full_candidate_semantic_review import (
        AUDIT_FILENAME as SEMANTIC_AUDIT_FILENAME,
        LEDGER_FILENAME,
    )
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset.audit_query_sft_pilot_inputs import question_digest
    from dataset.audit_query_sft_full_teacher_candidates import (
        AUDIT_FILENAME as PROTOCOL_AUDIT_FILENAME,
        DEFAULT_OUTPUT_DIR as DEFAULT_TEACHER_WORK_PACKAGE_DIR,
        RESULT_FILENAME,
    )
    from dataset.finalize_query_sft_full_candidate_semantic_review import (
        AUDIT_FILENAME as SEMANTIC_AUDIT_FILENAME,
        LEDGER_FILENAME,
    )


DEFAULT_SEMANTIC_REVIEW_DIR = DEFAULT_TEACHER_WORK_PACKAGE_DIR / "query-sft-v1-candidate-semantic-review-work-package"
DEFAULT_EVALUATION_EXCLUSIONS = Path(__file__).resolve().parent / "RAG-SFT" / "manifests" / "evaluation-exclusions-project-rag-v2.json"
REPORT_FILENAME = "query-sft-v1-candidate-text-audit.json"
RESULT_FIELDS = {"candidate_id", "work_id", "raw_output"}
LEDGER_FIELDS = {"candidate_id", "review_decision", "reason"}


class QuerySftFullCandidateTextAuditError(RuntimeError):
    """表示全量 Query-SFT 候选存在文本碰撞、评估泄漏或上游身份不一致。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    if records is not None:
        result["records"] = records
    return result


def _verify_sidecar(path: Path, label: str) -> Path:
    sidecar = path.with_suffix(".sha256")
    try:
        values = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftFullCandidateTextAuditError(f"无法读取{label} SHA-256") from error
    if values != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftFullCandidateTextAuditError(f"{label} SHA-256 无效")
    return sidecar


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftFullCandidateTextAuditError(f"无法读取{label}") from error
    if not isinstance(value, dict):
        raise QuerySftFullCandidateTextAuditError(f"{label}必须是 JSON 对象")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    raise QuerySftFullCandidateTextAuditError(f"{label}不允许空行: {number}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise QuerySftFullCandidateTextAuditError(f"{label}第 {number} 条必须是对象")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftFullCandidateTextAuditError):
            raise
        raise QuerySftFullCandidateTextAuditError(f"无法读取{label}") from error
    return rows


def audit_query_sft_full_candidate_texts(
    *,
    teacher_work_package_dir: Path = DEFAULT_TEACHER_WORK_PACKAGE_DIR,
    semantic_review_dir: Path = DEFAULT_SEMANTIC_REVIEW_DIR,
    evaluation_exclusions_path: Path = DEFAULT_EVALUATION_EXCLUSIONS,
    output_path: Path | None = None,
) -> dict[str, object]:
    """仅审计候选文本与摘要，不读取或输出任何 required GT。"""

    teacher_directory = Path(teacher_work_package_dir).resolve()
    review_directory = Path(semantic_review_dir).resolve()
    exclusions_path = Path(evaluation_exclusions_path).resolve()
    output = Path(output_path or review_directory / REPORT_FILENAME).resolve()
    if output.exists() or output.with_suffix(".sha256").exists():
        raise QuerySftFullCandidateTextAuditError("候选文本审计输出已存在")
    result_path = teacher_directory / RESULT_FILENAME
    protocol_path = teacher_directory / PROTOCOL_AUDIT_FILENAME
    ledger_path = review_directory / LEDGER_FILENAME
    semantic_audit_path = review_directory / SEMANTIC_AUDIT_FILENAME
    result_hash = _verify_sidecar(result_path, "教师候选结果")
    protocol_hash = _verify_sidecar(protocol_path, "教师候选协议审计")
    ledger_hash = _verify_sidecar(ledger_path, "候选语义审核账本")
    semantic_audit_hash = _verify_sidecar(semantic_audit_path, "候选语义审核报告")
    exclusions_hash = _verify_sidecar(exclusions_path, "评估排除清单")
    protocol = _load_json(protocol_path, "教师候选协议审计")
    semantic = _load_json(semantic_audit_path, "候选语义审核报告")
    exclusions = _load_json(exclusions_path, "评估排除清单")
    if (
        protocol.get("complete") is not True
        or protocol.get("records", {}).get("protocol_valid_candidates") != 1722
        or semantic.get("complete") is not True
        or semantic.get("records", {}).get("teacher_candidates") != 1722
        or exclusions.get("complete_for_formal_sft") is not True
    ):
        raise QuerySftFullCandidateTextAuditError("上游候选或评估隔离状态无效")
    digests = exclusions.get("question_sha256")
    if not isinstance(digests, list) or len(digests) != 563 or len(digests) != len(set(digests)) or any(not isinstance(digest, str) for digest in digests):
        raise QuerySftFullCandidateTextAuditError("评估排除摘要集合无效")
    evaluation_digests = set(digests)

    decisions = {}
    for row in _load_jsonl(ledger_path, "候选语义审核账本"):
        candidate_id = row.get("candidate_id")
        if (
            set(row) != LEDGER_FIELDS
            or not isinstance(candidate_id, str)
            or candidate_id in decisions
            or row.get("review_decision") not in {"approved", "rejected"}
            or not isinstance(row.get("reason"), str)
        ):
            raise QuerySftFullCandidateTextAuditError("候选语义审核账本映射无效")
        decisions[candidate_id] = row["review_decision"]
    if len(decisions) != 1722:
        raise QuerySftFullCandidateTextAuditError("候选语义审核账本数量无效")

    targets = {}
    all_candidates = set()
    accepted = 0
    text_overlap = 0
    for row in _load_jsonl(result_path, "教师候选结果"):
        candidate_id = row.get("candidate_id")
        if set(row) != RESULT_FIELDS or not isinstance(candidate_id, str) or candidate_id in all_candidates or candidate_id not in decisions:
            raise QuerySftFullCandidateTextAuditError("教师候选结果与审核账本映射无效")
        all_candidates.add(candidate_id)
        if decisions[candidate_id] == "rejected":
            continue
        try:
            enhancement = parse_and_validate_query_enhancement(row["raw_output"])
        except (TypeError, ValueError) as error:
            raise QuerySftFullCandidateTextAuditError(f"通过审核的候选协议无效: {candidate_id}") from error
        target = json.dumps(
            {"rewrite": enhancement.rewrite, "expansion_terms": list(enhancement.expansion_terms), "subqueries": list(enhancement.subqueries)},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        previous_work_id = targets.get(target)
        if previous_work_id is not None and previous_work_id != row["work_id"]:
            raise QuerySftFullCandidateTextAuditError(
                f"不同原始 query 的规范化 target 重复: {candidate_id} 与 {previous_work_id}"
            )
        targets[target] = row["work_id"]
        text_overlap += sum(question_digest(text) in evaluation_digests for text in [enhancement.rewrite, *enhancement.expansion_terms, *enhancement.subqueries])
        accepted += 1
    if set(decisions) != all_candidates:
        raise QuerySftFullCandidateTextAuditError("候选结果未与审核账本闭合")
    if text_overlap:
        raise QuerySftFullCandidateTextAuditError("通过审核的候选文本命中评估排除摘要")

    report = {
        "pipeline": "query_sft_full_candidate_text_audit_v1",
        "release_status": "frozen_retrieval_selection_pending",
        "inputs": {
            "teacher_candidate_results": {**_identity(result_path, records=len(all_candidates)), "hash_manifest": _identity(result_hash)},
            "teacher_candidate_protocol_audit": {**_identity(protocol_path), "hash_manifest": _identity(protocol_hash)},
            "semantic_gt_review_ledger": {**_identity(ledger_path, records=len(decisions)), "hash_manifest": _identity(ledger_hash)},
            "semantic_gt_review_audit": {**_identity(semantic_audit_path), "hash_manifest": _identity(semantic_audit_hash)},
            "evaluation_exclusions": {**_identity(exclusions_path), "hash_manifest": _identity(exclusions_hash)},
        },
        "records": {"teacher_candidates": len(all_candidates), "approved": accepted, "rejected": len(all_candidates) - accepted, "unique_approved_normalized_targets_across_work_ids": len(targets), "evaluation_text_overlap": 0},
        "validation": {"semantic_gt_review_identity_bound": True, "approved_targets_unique_across_work_ids": True, "approved_target_texts_excluded_from_evaluation": True, "required_gt_not_read_or_emitted": True},
        "readiness": {"candidate_protocol_semantic_dedup_isolation_complete": True, "frozen_retrieval_selection_ready": True, "query_sft_training_ready": False},
        "complete": True,
    }
    try:
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        output.with_suffix(".sha256").write_text(f"{_sha256_file(output)}  {output.name}\n", encoding="utf-8", newline="\n")
    except (OSError, UnicodeError) as error:
        raise QuerySftFullCandidateTextAuditError("无法发布候选文本审计报告") from error
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-work-package-dir", type=Path, default=DEFAULT_TEACHER_WORK_PACKAGE_DIR)
    parser.add_argument("--semantic-review-dir", type=Path, default=DEFAULT_SEMANTIC_REVIEW_DIR)
    parser.add_argument("--evaluation-exclusions", type=Path, default=DEFAULT_EVALUATION_EXCLUSIONS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = audit_query_sft_full_candidate_texts(
            teacher_work_package_dir=args.teacher_work_package_dir,
            semantic_review_dir=args.semantic_review_dir,
            evaluation_exclusions_path=args.evaluation_exclusions,
            output_path=args.output,
        )
    except QuerySftFullCandidateTextAuditError as error:
        parser.error(str(error))
    print(f"QUERY_SFT_FULL_CANDIDATE_TEXT_AUDIT_OK approved={report['records']['approved']} rejected={report['records']['rejected']}")


if __name__ == "__main__":
    main()
