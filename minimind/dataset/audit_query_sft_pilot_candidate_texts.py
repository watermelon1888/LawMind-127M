"""审计通过语义复核的 Query-SFT pilot 候选去重与评估隔离。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rag.query.enhancement import parse_and_validate_query_enhancement

try:
    from .audit_query_sft_pilot_inputs import question_digest
    from .prepare_query_sft_pilot_teacher_candidates import NOOP_FILENAME
    from .prepare_query_sft_pilot_candidate_semantic_review import (
        AUDIT_FILENAME as SEMANTIC_AUDIT_FILENAME,
        LEDGER_FILENAME,
    )
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset.audit_query_sft_pilot_inputs import question_digest
    from dataset.prepare_query_sft_pilot_teacher_candidates import NOOP_FILENAME
    from dataset.prepare_query_sft_pilot_candidate_semantic_review import (
        AUDIT_FILENAME as SEMANTIC_AUDIT_FILENAME,
        LEDGER_FILENAME,
    )


DEFAULT_EVALUATION_EXCLUSIONS = (
    Path(__file__).resolve().parent
    / "RAG-SFT"
    / "manifests"
    / "evaluation-exclusions-project-rag-v2.json"
)
AUDIT_FILENAME = "query-sft-pilot-v1-candidate-text-audit.json"
_RESULT_FIELDS = {"candidate_id", "pilot_id", "raw_output"}
_LEDGER_FIELDS = {"candidate_id", "pilot_id", "review_decision", "reason"}
_NOOP_FIELDS = {"candidate_id", "pilot_id", "raw_output"}


class QuerySftPilotCandidateTextAuditError(RuntimeError):
    """Query-SFT pilot 候选存在重复、评估碰撞或输入身份缺失。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    identity: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if records is not None:
        identity["records"] = records
    return identity


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftPilotCandidateTextAuditError(
            f"无法读取{description}: {path}"
        ) from error
    if not isinstance(result, dict):
        raise QuerySftPilotCandidateTextAuditError(f"{description}必须是 JSON object")
    return result


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    result = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise QuerySftPilotCandidateTextAuditError(
                        f"{description}不允许空行: {line_number}"
                    )
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise QuerySftPilotCandidateTextAuditError(
                        f"{description}第 {line_number} 条必须是 object"
                    )
                result.append(item)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftPilotCandidateTextAuditError):
            raise
        raise QuerySftPilotCandidateTextAuditError(
            f"无法读取{description}: {path}"
        ) from error
    if not result:
        raise QuerySftPilotCandidateTextAuditError(f"{description}不能为空")
    return result


def _verify_adjacent_hash(path: Path, description: str) -> Path:
    hash_path = path.with_suffix(".sha256")
    try:
        values = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftPilotCandidateTextAuditError(
            f"无法读取{description}相邻 SHA-256: {hash_path}"
        ) from error
    if values != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftPilotCandidateTextAuditError(f"{description}相邻 SHA-256 无效")
    return hash_path


def _canonical_target(raw_output: str) -> tuple[str, list[str]]:
    enhancement = parse_and_validate_query_enhancement(raw_output)
    target = json.dumps(
        {
            "rewrite": enhancement.rewrite,
            "expansion_terms": list(enhancement.expansion_terms),
            "subqueries": list(enhancement.subqueries),
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return target, [
        enhancement.rewrite,
        *enhancement.expansion_terms,
        *enhancement.subqueries,
    ]


def audit_query_sft_pilot_candidate_texts(
    *,
    result_path: Path,
    noop_path: Path,
    semantic_review_dir: Path,
    evaluation_exclusions_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """只审核文本身份与隔离，绝不读取或输出 required GT。"""

    result_path = Path(result_path).resolve()
    noop_path = Path(noop_path).resolve()
    semantic_review_dir = Path(semantic_review_dir).resolve()
    evaluation_exclusions_path = Path(evaluation_exclusions_path).resolve()
    output_path = Path(output_path).resolve()
    output_hash_path = output_path.with_suffix(".sha256")
    if output_path.exists() or output_hash_path.exists():
        raise QuerySftPilotCandidateTextAuditError(f"审计输出已存在: {output_path}")
    semantic_audit_path = semantic_review_dir / SEMANTIC_AUDIT_FILENAME
    ledger_path = semantic_review_dir / LEDGER_FILENAME
    result_hash_path = _verify_adjacent_hash(result_path, "教师候选结果")
    noop_hash_path = _verify_adjacent_hash(noop_path, "确定性 no-op")
    semantic_audit_hash_path = _verify_adjacent_hash(semantic_audit_path, "语义审核报告")
    ledger_hash_path = _verify_adjacent_hash(ledger_path, "语义审核账本")
    exclusions_hash_path = _verify_adjacent_hash(
        evaluation_exclusions_path, "评估排除清单"
    )
    semantic_audit = _load_json(semantic_audit_path, "语义审核报告")
    if (
        semantic_audit.get("pipeline")
        != "query_sft_pilot_candidate_semantic_gt_review_v1"
        or semantic_audit.get("complete") is not True
        or semantic_audit.get("records", {}).get("approved") != 114
        or semantic_audit.get("validation", {}).get("required_gt_not_emitted")
        is not True
    ):
        raise QuerySftPilotCandidateTextAuditError("语义审核报告状态无效")
    exclusions = _load_json(evaluation_exclusions_path, "评估排除清单")
    digests = exclusions.get("question_sha256")
    if (
        exclusions.get("complete_for_formal_sft") is not True
        or not isinstance(digests, list)
        or not digests
        or len(digests) != len(set(digests))
    ):
        raise QuerySftPilotCandidateTextAuditError("评估排除清单状态无效")
    evaluation_digests = set(digests)

    decisions = {}
    for position, item in enumerate(_load_jsonl(ledger_path, "语义审核账本"), start=1):
        if set(item) != _LEDGER_FIELDS:
            raise QuerySftPilotCandidateTextAuditError(
                f"语义审核账本第 {position} 条字段无效"
            )
        candidate_id = item.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or candidate_id in decisions
            or item.get("review_decision") not in {"approved", "rejected"}
        ):
            raise QuerySftPilotCandidateTextAuditError(
                f"语义审核账本第 {position} 条候选无效或重复"
            )
        decisions[candidate_id] = item["review_decision"]

    accepted = []
    all_candidate_ids = set()
    targets = {}
    text_collision_count = 0
    for position, item in enumerate(_load_jsonl(result_path, "教师候选结果"), start=1):
        if set(item) != _RESULT_FIELDS:
            raise QuerySftPilotCandidateTextAuditError(
                f"教师候选结果第 {position} 条字段无效"
            )
        candidate_id = item.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or candidate_id in all_candidate_ids
            or candidate_id not in decisions
            or not isinstance(item.get("raw_output"), str)
        ):
            raise QuerySftPilotCandidateTextAuditError(
                f"教师候选结果第 {position} 条映射无效"
            )
        all_candidate_ids.add(candidate_id)
        if decisions[candidate_id] != "approved":
            continue
        try:
            target, texts = _canonical_target(item["raw_output"])
        except (TypeError, ValueError) as error:
            raise QuerySftPilotCandidateTextAuditError(
                f"通过候选协议无效: {candidate_id}"
            ) from error
        if target in targets:
            raise QuerySftPilotCandidateTextAuditError(
                f"通过候选的规范化目标重复: {candidate_id} 与 {targets[target]}"
            )
        targets[target] = candidate_id
        text_collision_count += sum(
            question_digest(text) in evaluation_digests for text in texts
        )
        accepted.append(candidate_id)
    if set(decisions) != all_candidate_ids:
        raise QuerySftPilotCandidateTextAuditError("语义审核账本与教师候选结果不一致")

    noop_records = _load_jsonl(noop_path, "确定性 no-op")
    noop_ids = set()
    for position, item in enumerate(noop_records, start=1):
        if set(item) != _NOOP_FIELDS or not isinstance(item.get("candidate_id"), str):
            raise QuerySftPilotCandidateTextAuditError(
                f"确定性 no-op 第 {position} 条字段无效"
            )
        if item["candidate_id"] in noop_ids:
            raise QuerySftPilotCandidateTextAuditError("确定性 no-op candidate_id 重复")
        noop_ids.add(item["candidate_id"])
        try:
            _, texts = _canonical_target(item["raw_output"])
        except (TypeError, ValueError) as error:
            raise QuerySftPilotCandidateTextAuditError(
                f"确定性 no-op 协议无效: {item['candidate_id']}"
            ) from error
        text_collision_count += sum(
            question_digest(text) in evaluation_digests for text in texts
        )
    if len(noop_ids) != 39:
        raise QuerySftPilotCandidateTextAuditError("确定性 no-op 数量必须为 39")
    if text_collision_count:
        raise QuerySftPilotCandidateTextAuditError("候选文本命中评估排除清单")

    report: dict[str, object] = {
        "pipeline": "query_sft_pilot_candidate_text_audit_v1",
        "inputs": {
            "teacher_candidate_results": {
                **_identity(result_path, records=len(all_candidate_ids)),
                "hash_manifest": _identity(result_hash_path),
            },
            "deterministic_noop_candidates": {
                **_identity(noop_path, records=len(noop_records)),
                "hash_manifest": _identity(noop_hash_path),
            },
            "candidate_semantic_gt_review": {
                **_identity(semantic_audit_path),
                "hash_manifest": _identity(semantic_audit_hash_path),
            },
            "candidate_semantic_gt_ledger": {
                **_identity(ledger_path, records=len(decisions)),
                "hash_manifest": _identity(ledger_hash_path),
            },
            "evaluation_exclusions": {
                **_identity(evaluation_exclusions_path),
                "hash_manifest": _identity(exclusions_hash_path),
            },
        },
        "records": {
            "teacher_candidates_total": len(all_candidate_ids),
            "teacher_candidates_approved": len(accepted),
            "teacher_candidates_rejected": len(all_candidate_ids) - len(accepted),
            "deterministic_noop_candidates": len(noop_ids),
            "unique_approved_normalized_targets": len(targets),
            "evaluation_text_overlap": 0,
        },
        "validation": {
            "semantic_gt_review_identity_bound": True,
            "teacher_results_and_noop_identities_bound": True,
            "every_teacher_candidate_has_one_semantic_decision": True,
            "approved_normalized_targets_unique": True,
            "approved_and_noop_texts_excluded_from_evaluation": True,
            "required_gt_not_read_or_emitted": True,
        },
        "readiness": {
            "candidate_protocol_semantic_dedup_isolation_complete": True,
            "frozen_retrieval_comparison_ready": True,
            "query_sft_training_ready": False,
        },
        "complete": True,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
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
        raise QuerySftPilotCandidateTextAuditError("无法发布候选文本审计报告") from error
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--noop", type=Path, required=True)
    parser.add_argument("--semantic-review-dir", type=Path, required=True)
    parser.add_argument(
        "--evaluation-exclusions", type=Path, default=DEFAULT_EVALUATION_EXCLUSIONS
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = audit_query_sft_pilot_candidate_texts(
            result_path=args.results,
            noop_path=args.noop,
            semantic_review_dir=args.semantic_review_dir,
            evaluation_exclusions_path=args.evaluation_exclusions,
            output_path=args.output,
        )
    except QuerySftPilotCandidateTextAuditError as error:
        parser.error(str(error))
    print(
        "[完成] 候选文本审计 "
        f"approved={report['records']['teacher_candidates_approved']} "
        f"noop={report['records']['deterministic_noop_candidates']}"
    )
    print(f"审计报告: {args.output}")


if __name__ == "__main__":
    main()


__all__ = [
    "AUDIT_FILENAME",
    "QuerySftPilotCandidateTextAuditError",
    "audit_query_sft_pilot_candidate_texts",
]
