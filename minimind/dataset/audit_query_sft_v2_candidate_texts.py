"""审计 Query-SFT v2 通过语义审核候选的文本去重与评估隔离。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rag.query.enhancement import parse_and_validate_query_enhancement

from .audit_query_sft_pilot_inputs import question_digest
from .audit_query_sft_v2_teacher_candidates import RESULT_FILENAME
from .finalize_query_sft_v2_semantic_review import (
    AUDIT_FILENAME as SEMANTIC_AUDIT_FILENAME,
    LEDGER_FILENAME,
)
from .query_sft_v2_contract import SEMANTIC_REVIEW_FIELDS


DATASET_ROOT = Path(__file__).resolve().parent
DEFAULT_TEACHER_RESULTS = (
    DATASET_ROOT
    / "QUERY-POOL"
    / "full"
    / "query-sft-v2-local-teacher-generation-r1"
    / RESULT_FILENAME
)
DEFAULT_SEMANTIC_REVIEW_DIR = (
    DATASET_ROOT / "QUERY-POOL" / "full" / "query-sft-v2-semantic-quality-r2a"
)
DEFAULT_EVALUATION_EXCLUSIONS = (
    DATASET_ROOT / "RAG-SFT" / "manifests" / "evaluation-exclusions-project-rag-v2.json"
)
REPORT_FILENAME = "query-sft-v2-candidate-text-audit.json"
RESULT_FIELDS = ("candidate_id", "work_id", "raw_output")


class QuerySftV2CandidateTextAuditError(RuntimeError):
    """表示 v2 候选存在文本碰撞、评估泄漏或上游身份不一致。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if records is not None:
        value["records"] = records
    return value


def _verify_sidecar(path: Path, label: str) -> Path:
    sidecar = path.with_suffix(".sha256")
    try:
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftV2CandidateTextAuditError(f"无法读取{label} SHA-256") from error
    if lines != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftV2CandidateTextAuditError(f"{label} SHA-256 无效")
    return sidecar


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftV2CandidateTextAuditError(f"无法读取{label}") from error
    if not isinstance(value, dict):
        raise QuerySftV2CandidateTextAuditError(f"{label}必须是 JSON 对象")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    raise QuerySftV2CandidateTextAuditError(f"{label}不允许空行：{number}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise QuerySftV2CandidateTextAuditError(f"{label}第 {number} 条必须是对象")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftV2CandidateTextAuditError):
            raise
        raise QuerySftV2CandidateTextAuditError(f"无法读取{label}") from error
    return rows


def audit_query_sft_v2_candidate_texts(
    *,
    teacher_results_path: Path = DEFAULT_TEACHER_RESULTS,
    semantic_review_dir: Path = DEFAULT_SEMANTIC_REVIEW_DIR,
    evaluation_exclusions_path: Path = DEFAULT_EVALUATION_EXCLUSIONS,
    output_path: Path | None = None,
) -> dict[str, object]:
    """只审计文本和摘要，不读取或输出 required GT。"""

    results_path = Path(teacher_results_path).resolve()
    review_dir = Path(semantic_review_dir).resolve()
    exclusions_path = Path(evaluation_exclusions_path).resolve()
    output = Path(output_path or review_dir / REPORT_FILENAME).resolve()
    if output.exists() or output.with_suffix(".sha256").exists():
        raise QuerySftV2CandidateTextAuditError("v2 候选文本审计输出已存在")
    ledger_path = review_dir / LEDGER_FILENAME
    semantic_audit_path = review_dir / SEMANTIC_AUDIT_FILENAME
    results_hash = _verify_sidecar(results_path, "教师候选结果")
    ledger_hash = _verify_sidecar(ledger_path, "语义审核账本")
    semantic_audit_hash = _verify_sidecar(semantic_audit_path, "语义审核报告")
    exclusions_hash = _verify_sidecar(exclusions_path, "评估排除清单")
    semantic = _load_json(semantic_audit_path, "语义审核报告")
    exclusions = _load_json(exclusions_path, "评估排除清单")
    if (
        semantic.get("pipeline") != "query_sft_v2_semantic_quality_audit"
        or semantic.get("complete") is not True
        or semantic.get("readiness", {}).get("semantic_quality_ready") is not True
        or semantic.get("records", {}).get("teacher_candidates") != 1722
        or exclusions.get("complete_for_formal_sft") is not True
    ):
        raise QuerySftV2CandidateTextAuditError("上游语义审核或评估隔离状态无效")
    digests = exclusions.get("question_sha256")
    if (
        not isinstance(digests, list)
        or len(digests) != 563
        or len(digests) != len(set(digests))
        or any(not isinstance(value, str) for value in digests)
    ):
        raise QuerySftV2CandidateTextAuditError("评估排除摘要集合无效")
    evaluation_digests = set(digests)

    decisions: dict[str, str] = {}
    for row in _load_jsonl(ledger_path, "语义审核账本"):
        candidate_id = row.get("candidate_id")
        if (
            tuple(row) != SEMANTIC_REVIEW_FIELDS
            or not isinstance(candidate_id, str)
            or candidate_id in decisions
            or row.get("review_decision") not in {"approved", "rejected"}
        ):
            raise QuerySftV2CandidateTextAuditError("语义审核账本映射无效")
        decisions[candidate_id] = row["review_decision"]
    if len(decisions) != 1722:
        raise QuerySftV2CandidateTextAuditError("语义审核账本数量无效")

    targets: dict[str, str] = {}
    all_candidates: set[str] = set()
    accepted = 0
    text_overlap = 0
    for row in _load_jsonl(results_path, "教师候选结果"):
        candidate_id = row.get("candidate_id")
        if (
            tuple(row) != RESULT_FIELDS
            or not isinstance(candidate_id, str)
            or candidate_id in all_candidates
            or candidate_id not in decisions
        ):
            raise QuerySftV2CandidateTextAuditError("教师候选结果与审核账本映射无效")
        all_candidates.add(candidate_id)
        if decisions[candidate_id] == "rejected":
            continue
        try:
            enhancement = parse_and_validate_query_enhancement(row["raw_output"])
        except (TypeError, ValueError) as error:
            raise QuerySftV2CandidateTextAuditError(f"通过语义审核的候选协议无效：{candidate_id}") from error
        target = json.dumps(
            {
                "rewrite": enhancement.rewrite,
                "expansion_terms": list(enhancement.expansion_terms),
                "subqueries": list(enhancement.subqueries),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        previous_work_id = targets.get(target)
        if previous_work_id is not None and previous_work_id != row["work_id"]:
            raise QuerySftV2CandidateTextAuditError(
                f"不同原始 query 的规范化 target 重复：{candidate_id} 与 {previous_work_id}"
            )
        targets[target] = row["work_id"]
        text_overlap += sum(
            question_digest(text) in evaluation_digests
            for text in [enhancement.rewrite, *enhancement.expansion_terms, *enhancement.subqueries]
        )
        accepted += 1
    if set(decisions) != all_candidates:
        raise QuerySftV2CandidateTextAuditError("教师候选结果未与审核账本闭合")
    if text_overlap:
        raise QuerySftV2CandidateTextAuditError("通过语义审核的候选文本命中评估排除摘要")

    report = {
        "schema_version": "1.0",
        "pipeline": "query_sft_v2_candidate_text_audit",
        "release_status": "frozen_retrieval_selection_pending",
        "inputs": {
            "teacher_candidate_results": {
                **_identity(results_path, records=len(all_candidates)),
                "hash_manifest": _identity(results_hash),
            },
            "semantic_review_ledger": {
                **_identity(ledger_path, records=len(decisions)),
                "hash_manifest": _identity(ledger_hash),
            },
            "semantic_quality_audit": {
                **_identity(semantic_audit_path),
                "hash_manifest": _identity(semantic_audit_hash),
            },
            "evaluation_exclusions": {
                **_identity(exclusions_path),
                "hash_manifest": _identity(exclusions_hash),
            },
        },
        "records": {
            "teacher_candidates": len(all_candidates),
            "approved": accepted,
            "rejected": len(all_candidates) - accepted,
            "unique_approved_normalized_targets_across_work_ids": len(targets),
            "evaluation_text_overlap": 0,
        },
        "validation": {
            "semantic_quality_identity_bound": True,
            "approved_targets_unique_across_work_ids": True,
            "approved_target_texts_excluded_from_evaluation": True,
            "required_gt_not_read_or_emitted": True,
        },
        "readiness": {
            "candidate_text_isolation_complete": True,
            "frozen_retrieval_selection_ready": False,
            "training_ready": False,
        },
        "complete": True,
    }
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        output.with_suffix(".sha256").write_text(
            f"{_sha256_file(output)}  {output.name}\n", encoding="utf-8", newline="\n"
        )
    except (OSError, UnicodeError) as error:
        raise QuerySftV2CandidateTextAuditError("无法发布 v2 候选文本审计报告") from error
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-results", type=Path, default=DEFAULT_TEACHER_RESULTS)
    parser.add_argument("--semantic-review-dir", type=Path, default=DEFAULT_SEMANTIC_REVIEW_DIR)
    parser.add_argument("--evaluation-exclusions", type=Path, default=DEFAULT_EVALUATION_EXCLUSIONS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = audit_query_sft_v2_candidate_texts(
            teacher_results_path=args.teacher_results,
            semantic_review_dir=args.semantic_review_dir,
            evaluation_exclusions_path=args.evaluation_exclusions,
            output_path=args.output,
        )
    except QuerySftV2CandidateTextAuditError as error:
        parser.error(str(error))
    print(
        "QUERY_SFT_V2_CANDIDATE_TEXT_AUDIT_OK "
        f"approved={report['records']['approved']} rejected={report['records']['rejected']}"
    )


if __name__ == "__main__":
    main()
