"""发布经人工语义与 GT 兼容性复核的 Query-SFT pilot 教师候选账本。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .audit_query_sft_pilot_teacher_candidates import (
        DEFAULT_AUDIT_FILENAME as DEFAULT_PROTOCOL_AUDIT_FILENAME,
        DEFAULT_RESULT_FILENAME,
    )
    from .prepare_query_sft_pilot import (
        AUDIT_REFERENCE_FILENAME,
        DEFAULT_OUTPUT_DIR as DEFAULT_INPUT_WORK_PACKAGE_DIR,
        HASH_FILENAME as INPUT_WORK_PACKAGE_HASH_FILENAME,
        MANIFEST_FILENAME as INPUT_WORK_PACKAGE_MANIFEST_FILENAME,
    )
    from .prepare_query_sft_pilot_teacher_candidates import (
        DEFAULT_OUTPUT_DIR as DEFAULT_TEACHER_WORK_PACKAGE_DIR,
        HASH_FILENAME as TEACHER_WORK_PACKAGE_HASH_FILENAME,
        MANIFEST_FILENAME as TEACHER_WORK_PACKAGE_MANIFEST_FILENAME,
    )
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset.audit_query_sft_pilot_teacher_candidates import (
        DEFAULT_AUDIT_FILENAME as DEFAULT_PROTOCOL_AUDIT_FILENAME,
        DEFAULT_RESULT_FILENAME,
    )
    from dataset.prepare_query_sft_pilot import (
        AUDIT_REFERENCE_FILENAME,
        DEFAULT_OUTPUT_DIR as DEFAULT_INPUT_WORK_PACKAGE_DIR,
        HASH_FILENAME as INPUT_WORK_PACKAGE_HASH_FILENAME,
        MANIFEST_FILENAME as INPUT_WORK_PACKAGE_MANIFEST_FILENAME,
    )
    from dataset.prepare_query_sft_pilot_teacher_candidates import (
        DEFAULT_OUTPUT_DIR as DEFAULT_TEACHER_WORK_PACKAGE_DIR,
        HASH_FILENAME as TEACHER_WORK_PACKAGE_HASH_FILENAME,
        MANIFEST_FILENAME as TEACHER_WORK_PACKAGE_MANIFEST_FILENAME,
    )


LEDGER_FILENAME = "query-sft-pilot-v1-candidate-semantic-gt-review.jsonl"
AUDIT_FILENAME = "query-sft-pilot-v1-candidate-semantic-gt-review-audit.json"

# 这些是本次逐组人工复核后的明确淘汰结论，而不是自动语义分类规则。
REJECTED_REASONS = {
    "query_sft_pilot:0006/teacher-3": (
        "将“还得做什么”具体化为“减损义务”，引入原问题未明确的法律事项。"
    ),
    "query_sft_pilot:0008/teacher-2": (
        "将“伤得很重”收缩为法定伤情等级“重伤”，改变了原始事实表述的确定性。"
    ),
    "query_sft_pilot:0025/teacher-3": (
        "遗漏“购买五个月”的明确时间条件，不能视为完整等价改写。"
    ),
}
_APPROVED_REASON = "改写保留原始法律事项与条件，未新增事实、法律结论或无关检索方向。"
_REFERENCE_FIELDS = {
    "pilot_id",
    "source_id",
    "authoring_type",
    "coverage_domain",
    "source_query",
    "required_chunk_ids",
    "source_record_sha256",
}
_RESULT_FIELDS = {"candidate_id", "pilot_id", "raw_output"}
_LEDGER_FIELDS = {"candidate_id", "pilot_id", "review_decision", "reason"}


class QuerySftPilotCandidateReviewError(RuntimeError):
    """Query-SFT pilot 候选语义审核材料不完整或身份不一致。"""


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
        raise QuerySftPilotCandidateReviewError(
            f"无法读取{description}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise QuerySftPilotCandidateReviewError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise QuerySftPilotCandidateReviewError(
                        f"{description}不允许空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise QuerySftPilotCandidateReviewError(
                        f"{description}第 {line_number} 条必须是 object"
                    )
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftPilotCandidateReviewError):
            raise
        raise QuerySftPilotCandidateReviewError(
            f"无法读取{description}: {path}"
        ) from error
    if not records:
        raise QuerySftPilotCandidateReviewError(f"{description}不能为空")
    return records


def _verify_adjacent_hash(path: Path, description: str) -> Path:
    hash_path = path.with_suffix(".sha256")
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftPilotCandidateReviewError(
            f"无法读取{description}相邻 SHA-256: {hash_path}"
        ) from error
    if lines != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftPilotCandidateReviewError(f"{description}相邻 SHA-256 无效")
    return hash_path


def _verify_package_hashes(
    directory: Path, *, manifest_filename: str, hash_filename: str, required_names: set[str], description: str
) -> tuple[Path, Path]:
    manifest_path = directory / manifest_filename
    hash_path = directory / hash_filename
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftPilotCandidateReviewError(
            f"无法读取{description} SHA-256"
        ) from error
    found = {}
    for line in lines:
        digest, separator, filename = line.partition("  ")
        if not separator or not digest or not filename or filename in found:
            raise QuerySftPilotCandidateReviewError(f"{description} SHA-256 格式无效")
        found[filename] = digest
    if not required_names.issubset(found):
        raise QuerySftPilotCandidateReviewError(f"{description} SHA-256 条目不完整")
    for filename in required_names:
        path = directory / filename
        if found[filename] != _sha256_file(path):
            raise QuerySftPilotCandidateReviewError(
                f"{description}身份已变化: {filename}"
            )
    return manifest_path, hash_path


def _jsonl_payload(records: list[dict[str, str]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in records
    )


def prepare_query_sft_pilot_candidate_semantic_review(
    *,
    input_work_package_dir: Path,
    teacher_work_package_dir: Path,
    result_path: Path,
    protocol_audit_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """将人工复核结论写入账本，并发布不会包含 GT 的审计报告。"""

    input_work_package_dir = Path(input_work_package_dir).resolve()
    teacher_work_package_dir = Path(teacher_work_package_dir).resolve()
    result_path = Path(result_path).resolve()
    protocol_audit_path = Path(protocol_audit_path).resolve()
    output_dir = Path(output_dir).resolve()
    ledger_path = output_dir / LEDGER_FILENAME
    ledger_hash_path = ledger_path.with_suffix(".sha256")
    audit_path = output_dir / AUDIT_FILENAME
    audit_hash_path = audit_path.with_suffix(".sha256")
    if output_dir.exists():
        raise QuerySftPilotCandidateReviewError(f"输出目录必须不存在: {output_dir}")

    input_manifest_path, input_hash_path = _verify_package_hashes(
        input_work_package_dir,
        manifest_filename=INPUT_WORK_PACKAGE_MANIFEST_FILENAME,
        hash_filename=INPUT_WORK_PACKAGE_HASH_FILENAME,
        required_names={INPUT_WORK_PACKAGE_MANIFEST_FILENAME, AUDIT_REFERENCE_FILENAME},
        description="输入工作包",
    )
    teacher_manifest_path, teacher_hash_path = _verify_package_hashes(
        teacher_work_package_dir,
        manifest_filename=TEACHER_WORK_PACKAGE_MANIFEST_FILENAME,
        hash_filename=TEACHER_WORK_PACKAGE_HASH_FILENAME,
        required_names={TEACHER_WORK_PACKAGE_MANIFEST_FILENAME},
        description="教师候选工作包",
    )
    result_hash_path = _verify_adjacent_hash(result_path, "教师候选结果")
    protocol_hash_path = _verify_adjacent_hash(protocol_audit_path, "教师候选协议审计")

    input_manifest = _load_json(input_manifest_path, "输入工作包 manifest")
    if input_manifest.get("pipeline") != "query_sft_pilot_work_package_v1":
        raise QuerySftPilotCandidateReviewError("输入工作包状态无效")
    references = _load_jsonl(
        input_work_package_dir / AUDIT_REFERENCE_FILENAME, "GT 审核引用"
    )
    reference_ids = set()
    for position, reference in enumerate(references, start=1):
        if set(reference) != _REFERENCE_FIELDS or not isinstance(
            reference.get("pilot_id"), str
        ):
            raise QuerySftPilotCandidateReviewError(
                f"GT 审核引用第 {position} 条字段无效"
            )
        reference_ids.add(reference["pilot_id"])

    teacher_manifest = _load_json(teacher_manifest_path, "教师候选工作包 manifest")
    if (
        teacher_manifest.get("pipeline")
        != "query_sft_pilot_teacher_candidate_work_package_v1"
        or teacher_manifest.get("records", {}).get("teacher_candidate_requests")
        != 117
    ):
        raise QuerySftPilotCandidateReviewError("教师候选工作包状态无效")
    protocol = _load_json(protocol_audit_path, "教师候选协议审计")
    if (
        protocol.get("pipeline")
        != "query_sft_pilot_teacher_candidate_protocol_audit_v1"
        or protocol.get("complete") is not True
        or protocol.get("records", {}).get("protocol_valid_candidates") != 117
    ):
        raise QuerySftPilotCandidateReviewError("教师候选协议审计状态无效")
    results = _load_jsonl(result_path, "教师候选结果")
    candidates = []
    seen = set()
    for position, result in enumerate(results, start=1):
        if set(result) != _RESULT_FIELDS:
            raise QuerySftPilotCandidateReviewError(
                f"教师候选结果第 {position} 条字段无效"
            )
        candidate_id = result.get("candidate_id")
        pilot_id = result.get("pilot_id")
        if (
            not isinstance(candidate_id, str)
            or candidate_id in seen
            or not isinstance(pilot_id, str)
            or pilot_id not in reference_ids
        ):
            raise QuerySftPilotCandidateReviewError(
                f"教师候选结果第 {position} 条映射无效"
            )
        seen.add(candidate_id)
        candidates.append({"candidate_id": candidate_id, "pilot_id": pilot_id})
    if len(candidates) != 117:
        raise QuerySftPilotCandidateReviewError("教师候选结果数量无效")
    unknown_rejections = set(REJECTED_REASONS) - seen
    if unknown_rejections:
        raise QuerySftPilotCandidateReviewError(
            f"人工淘汰结论不属于当前候选: {sorted(unknown_rejections)}"
        )

    ledger = [
        {
            "candidate_id": candidate["candidate_id"],
            "pilot_id": candidate["pilot_id"],
            "review_decision": (
                "rejected"
                if candidate["candidate_id"] in REJECTED_REASONS
                else "approved"
            ),
            "reason": REJECTED_REASONS.get(
                candidate["candidate_id"], _APPROVED_REASON
            ),
        }
        for candidate in candidates
    ]
    decision_counts = Counter(item["review_decision"] for item in ledger)
    if decision_counts != {"approved": 114, "rejected": 3}:
        raise AssertionError("人工复核账本的预期决策数量已变化")
    ledger_payload = _jsonl_payload(ledger)
    audit: dict[str, object] = {
        "pipeline": "query_sft_pilot_candidate_semantic_gt_review_v1",
        "inputs": {
            "input_work_package_manifest": _identity(input_manifest_path),
            "input_work_package_hash_manifest": _identity(input_hash_path),
            "gt_audit_reference": _identity(
                input_work_package_dir / AUDIT_REFERENCE_FILENAME,
                records=len(references),
            ),
            "teacher_work_package_manifest": _identity(teacher_manifest_path),
            "teacher_work_package_hash_manifest": _identity(teacher_hash_path),
            "teacher_candidate_results": {
                **_identity(result_path, records=len(results)),
                "hash_manifest": _identity(result_hash_path),
            },
            "teacher_candidate_protocol_audit": {
                **_identity(protocol_audit_path),
                "hash_manifest": _identity(protocol_hash_path),
            },
        },
        "records": {
            "teacher_candidates": len(candidates),
            "approved": decision_counts["approved"],
            "rejected": decision_counts["rejected"],
            "rejected_candidate_ids": sorted(REJECTED_REASONS),
        },
        "validation": {
            "input_and_teacher_work_package_identities_bound": True,
            "teacher_protocol_audit_passed": True,
            "all_candidates_reviewed_once": True,
            "candidate_to_pilot_mapping_matches_gt_review_reference": True,
            "semantic_and_gt_compatibility_review_complete": True,
            "required_gt_not_emitted": True,
        },
        "readiness": {
            "candidate_semantic_gt_review_complete": True,
            "candidate_selection_retrieval_evaluation_ready": True,
            "query_sft_training_ready": False,
        },
        "complete": True,
    }
    audit_payload = json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        output_dir.mkdir(parents=True)
        ledger_path.write_text(ledger_payload, encoding="utf-8", newline="\n")
        ledger_hash_path.write_text(
            f"{_sha256_file(ledger_path)}  {ledger_path.name}\n",
            encoding="utf-8",
            newline="\n",
        )
        audit_path.write_text(audit_payload, encoding="utf-8", newline="\n")
        audit_hash_path.write_text(
            f"{_sha256_file(audit_path)}  {audit_path.name}\n",
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, UnicodeError) as error:
        for path in (audit_hash_path, audit_path, ledger_hash_path, ledger_path):
            path.unlink(missing_ok=True)
        try:
            output_dir.rmdir()
        except OSError:
            pass
        raise QuerySftPilotCandidateReviewError("无法发布候选语义审核账本") from error
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-work-package-dir", type=Path, default=DEFAULT_INPUT_WORK_PACKAGE_DIR
    )
    parser.add_argument(
        "--teacher-work-package-dir",
        type=Path,
        default=DEFAULT_TEACHER_WORK_PACKAGE_DIR,
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_TEACHER_WORK_PACKAGE_DIR / DEFAULT_RESULT_FILENAME,
    )
    parser.add_argument(
        "--protocol-audit",
        type=Path,
        default=DEFAULT_TEACHER_WORK_PACKAGE_DIR / DEFAULT_PROTOCOL_AUDIT_FILENAME,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_TEACHER_WORK_PACKAGE_DIR / "query-sft-pilot-v1-candidate-semantic-review",
    )
    args = parser.parse_args()
    try:
        audit = prepare_query_sft_pilot_candidate_semantic_review(
            input_work_package_dir=args.input_work_package_dir,
            teacher_work_package_dir=args.teacher_work_package_dir,
            result_path=args.results,
            protocol_audit_path=args.protocol_audit,
            output_dir=args.output_dir,
        )
    except QuerySftPilotCandidateReviewError as error:
        parser.error(str(error))
    print(
        "[完成] 候选语义与 GT 复核 "
        f"approved={audit['records']['approved']} rejected={audit['records']['rejected']}"
    )
    print(f"审核报告: {args.output_dir / AUDIT_FILENAME}")


if __name__ == "__main__":
    main()


__all__ = [
    "AUDIT_FILENAME",
    "LEDGER_FILENAME",
    "QuerySftPilotCandidateReviewError",
    "REJECTED_REASONS",
    "prepare_query_sft_pilot_candidate_semantic_review",
]
