"""合并并审计全量 Query-SFT 教师候选的协议与队列覆盖。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rag.query.enhancement import parse_and_validate_query_enhancement

try:
    from .prepare_query_sft_full_teacher_candidates import DEFAULT_OUTPUT_DIR
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset.prepare_query_sft_full_teacher_candidates import DEFAULT_OUTPUT_DIR


QUEUE_FIELDS = {"candidate_id", "work_id", "query_original"}
RESULT_FIELDS = {"candidate_id", "work_id", "raw_output"}
MANIFEST_FILENAME = "query-sft-v1-teacher-candidate-work-package.json"
HASH_FILENAME = "query-sft-v1-teacher-candidate-work-package.sha256"
RESULT_FILENAME = "query-sft-v1-teacher-candidate-results-r1.jsonl"
AUDIT_FILENAME = "query-sft-v1-teacher-candidate-protocol-audit-r1.json"
REPAIR_WORK_PACKAGE_DIRNAME = "teacher-protocol-repairs-r1"
REPAIR_MANIFEST_FILENAME = "query-sft-v1-teacher-protocol-repair-work-package.json"
REPAIR_HASH_FILENAME = "query-sft-v1-teacher-protocol-repair-work-package.sha256"
REPAIR_QUEUE_FIELDS = {"candidate_id", "work_id", "query_original", "failure_code"}

RESULT_SOURCES = {
    (1, 1): "teacher-c-r1-batch-01-slot-1.jsonl", (1, 2): "teacher-c-r1-batch-01-slot-2.jsonl", (1, 3): "teacher-c-r1-batch-01-slot-3.jsonl",
    (2, 1): "teacher-d-r1-batch-02-slot-1.jsonl", (2, 2): "teacher-d-r2-batch-02-slot-2.jsonl", (2, 3): "teacher-d-r1-batch-02-slot-3.jsonl",
    (3, 1): "teacher-c-r1-batch-03-slot-1.jsonl", (3, 2): "teacher-e-r1-batch-03-slot-2.jsonl", (3, 3): "teacher-a-r1-batch-03-slot-3.jsonl",
    (4, 0): "teacher-b-r1-batch-04.jsonl", (5, 0): "teacher-b-r1-batch-05.jsonl", (6, 0): "teacher-b-r1-batch-06.jsonl",
    (7, 1): "teacher-a-r1-batch-07-slot-1.jsonl", (7, 2): "teacher-a-r1-batch-07-slot-2.jsonl", (7, 3): "teacher-a-r1-batch-07-slot-3.jsonl",
    (8, 1): "teacher-a-r1-batch-08-slot-1.jsonl", (8, 2): "teacher-a-r1-batch-08-slot-2.jsonl", (8, 3): "teacher-a-r1-batch-08-slot-3.jsonl",
}

REPAIR_RESULT_SOURCES = {
    1: "teacher-repair-a-r1-part-01.jsonl",
    2: "teacher-repair-d-r1-part-02.jsonl",
    3: "teacher-repair-e-r1-part-03.jsonl",
}


class QuerySftFullTeacherAuditError(RuntimeError):
    """表示教师候选未满足全量协议或队列闭合条件。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
    if records is not None:
        value["records"] = records
    return value


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftFullTeacherAuditError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise QuerySftFullTeacherAuditError(f"{description}必须是 JSON 对象")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    raise QuerySftFullTeacherAuditError(f"{description}不允许空行: {number}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise QuerySftFullTeacherAuditError(f"{description}第 {number} 条必须是对象")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftFullTeacherAuditError):
            raise
        raise QuerySftFullTeacherAuditError(f"无法读取{description}: {path}") from error
    return rows


def _verify_work_package(directory: Path) -> Path:
    hash_path = directory / HASH_FILENAME
    try:
        entries = {name: digest for digest, name in (line.split("  ", 1) for line in hash_path.read_text(encoding="utf-8").splitlines())}
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise QuerySftFullTeacherAuditError("教师工作包 SHA-256 格式无效") from error
    if len(entries) != 10:
        raise QuerySftFullTeacherAuditError("教师工作包 SHA-256 条目数无效")
    for name, digest in entries.items():
        path = directory / name
        if not path.is_file() or _sha256_file(path) != digest:
            raise QuerySftFullTeacherAuditError(f"教师工作包身份已变化: {name}")
    manifest_path = directory / MANIFEST_FILENAME
    manifest = _load_json(manifest_path, "教师工作包 manifest")
    if manifest.get("pipeline") != "query_sft_full_teacher_candidate_work_package_v1" or manifest.get("records", {}).get("teacher_candidate_requests") != 1722 or manifest.get("complete") is not True:
        raise QuerySftFullTeacherAuditError("教师工作包状态无效")
    return hash_path


def _verify_repair_work_package(directory: Path) -> Path:
    hash_path = directory / REPAIR_HASH_FILENAME
    try:
        entries = {
            name: digest
            for digest, name in (line.split("  ", 1) for line in hash_path.read_text(encoding="utf-8").splitlines())
        }
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise QuerySftFullTeacherAuditError("协议修订工作包 SHA-256 格式无效") from error
    if len(entries) != 4:
        raise QuerySftFullTeacherAuditError("协议修订工作包 SHA-256 条目数无效")
    for name, digest in entries.items():
        path = directory / name
        if not path.is_file() or _sha256_file(path) != digest:
            raise QuerySftFullTeacherAuditError(f"协议修订工作包身份已变化: {name}")
    manifest = _load_json(directory / REPAIR_MANIFEST_FILENAME, "协议修订工作包 manifest")
    if (
        manifest.get("pipeline") != "query_sft_full_teacher_protocol_repair_work_package_v1"
        or manifest.get("records", {}).get("protocol_invalid_candidates") != 46
        or manifest.get("records", {}).get("parts") != 3
        or manifest.get("complete") is not True
    ):
        raise QuerySftFullTeacherAuditError("协议修订工作包状态无效")
    return hash_path


def _load_repair_overrides(
    repair_directory: Path, expected: dict[str, dict[str, str]]
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, object]], dict[str, object]]:
    repair_hash_path = _verify_repair_work_package(repair_directory)
    overrides: dict[str, dict[str, str]] = {}
    identities: dict[str, dict[str, object]] = {}
    expected_ids: set[str] = set()
    for part, filename in REPAIR_RESULT_SOURCES.items():
        queue_path = repair_directory / "repair-queue" / f"part-{part:02d}.jsonl"
        queue_rows = _load_jsonl(queue_path, f"协议修订队列第 {part} 份")
        queue_ids = set()
        for queue_row in queue_rows:
            candidate_id = queue_row.get("candidate_id")
            if (
                set(queue_row) != REPAIR_QUEUE_FIELDS
                or candidate_id not in expected
                or candidate_id in queue_ids
                or queue_row.get("work_id") != expected[candidate_id]["work_id"]
                or queue_row.get("query_original") != expected[candidate_id]["query_original"]
                or queue_row.get("failure_code") != "query_enhancement_protocol_invalid"
            ):
                raise QuerySftFullTeacherAuditError(f"协议修订队列第 {part} 份映射无效")
            queue_ids.add(candidate_id)
        result_path = repair_directory / "teacher-results" / filename
        result_rows = _load_jsonl(result_path, f"协议修订结果 {filename}")
        if len(result_rows) != len(queue_ids):
            raise QuerySftFullTeacherAuditError(f"协议修订结果 {filename} 数量不闭合")
        for row in result_rows:
            candidate_id = row.get("candidate_id")
            if (
                set(row) != RESULT_FIELDS
                or candidate_id not in queue_ids
                or candidate_id in overrides
                or row.get("work_id") != expected[candidate_id]["work_id"]
                or not isinstance(row.get("raw_output"), str)
            ):
                raise QuerySftFullTeacherAuditError(f"协议修订结果 {filename} 映射无效")
            try:
                parse_and_validate_query_enhancement(row["raw_output"])
            except (TypeError, ValueError) as error:
                raise QuerySftFullTeacherAuditError(f"协议修订结果 {filename} 不符合 Query Enhancement 协议: {candidate_id}") from error
            overrides[candidate_id] = row
        expected_ids.update(queue_ids)
        identities[filename] = _identity(result_path, records=len(result_rows))
    if len(expected_ids) != 46 or set(overrides) != expected_ids:
        raise QuerySftFullTeacherAuditError("协议修订结果未完整覆盖冻结的 46 个失败槽位")
    return overrides, identities, _identity(repair_hash_path)


def audit_query_sft_full_teacher_candidates(
    *,
    work_package_dir: Path = DEFAULT_OUTPUT_DIR,
    repair_work_package_dir: Path | None = None,
    output_path: Path | None = None,
    audit_path: Path | None = None,
) -> dict[str, object]:
    """合并冻结教师结果与受控修订覆盖，并严格验证 1722 个槽位。"""

    directory = Path(work_package_dir).resolve()
    output_path = Path(output_path or directory / RESULT_FILENAME).resolve()
    audit_path = Path(audit_path or directory / AUDIT_FILENAME).resolve()
    if any(path.exists() for path in (output_path, output_path.with_suffix(".sha256"), audit_path, audit_path.with_suffix(".sha256"))):
        raise QuerySftFullTeacherAuditError("教师候选审计输出已存在")
    work_hash_path = _verify_work_package(directory)
    expected: dict[str, dict[str, str]] = {}
    ordered_ids = []
    for batch in range(1, 9):
        for row in _load_jsonl(directory / "teacher-queue" / f"batch-{batch:02d}.jsonl", f"第 {batch} 批教师队列"):
            if set(row) != QUEUE_FIELDS or not isinstance(row.get("candidate_id"), str) or row["candidate_id"] in expected:
                raise QuerySftFullTeacherAuditError("教师队列字段或 candidate_id 无效")
            expected[row["candidate_id"]] = row
            ordered_ids.append(row["candidate_id"])
    if len(expected) != 1722:
        raise QuerySftFullTeacherAuditError("教师队列总数必须为 1722")
    repair_directory = Path(repair_work_package_dir or directory / REPAIR_WORK_PACKAGE_DIRNAME).resolve()
    overrides, repair_identities, repair_hash_identity = _load_repair_overrides(repair_directory, expected)
    results: dict[str, dict[str, str]] = {}
    identities = {}
    noop_count = 0
    for (batch, slot), filename in RESULT_SOURCES.items():
        path = directory / "teacher-results" / filename
        rows = _load_jsonl(path, f"教师结果 {filename}")
        first = (batch - 1) * 72 + 1
        last = min(batch * 72, 574)
        expected_ids = {
            candidate_id
            for candidate_id, queue_row in expected.items()
            if first <= int(queue_row["work_id"].rsplit(":", 1)[1]) <= last
            and (slot == 0 or candidate_id.endswith(f"/teacher-{slot}"))
        }
        if len(rows) != len(expected_ids):
            raise QuerySftFullTeacherAuditError(f"教师结果 {filename} 槽位数量不闭合")
        for row in rows:
            if set(row) != RESULT_FIELDS or row.get("candidate_id") not in expected_ids or row.get("candidate_id") in results or row.get("work_id") != expected[row["candidate_id"]]["work_id"] or not isinstance(row.get("raw_output"), str):
                raise QuerySftFullTeacherAuditError(f"教师结果 {filename} 映射无效")
            final_row = overrides.get(row["candidate_id"], row)
            try:
                enhanced = parse_and_validate_query_enhancement(final_row["raw_output"])
            except (TypeError, ValueError) as error:
                raise QuerySftFullTeacherAuditError(
                    f"教师结果 {filename} 不符合 Query Enhancement 协议，且没有有效修订: {row['candidate_id']}"
                ) from error
            source_query = expected[row["candidate_id"]]["query_original"]
            if enhanced.rewrite == source_query and not enhanced.expansion_terms and not enhanced.subqueries:
                noop_count += 1
            results[row["candidate_id"]] = final_row
        identities[filename] = _identity(path, records=len(rows))
    if set(results) != set(expected):
        raise QuerySftFullTeacherAuditError("教师结果未覆盖全部队列槽位")
    payload = "".join(json.dumps(results[candidate_id], ensure_ascii=False, separators=(",", ":")) + "\n" for candidate_id in ordered_ids)
    report = {"pipeline": "query_sft_full_teacher_candidate_protocol_audit_v1", "release_status": "full_candidate_semantic_gt_review_pending", "inputs": {"work_package_hash_manifest": _identity(work_hash_path), "teacher_result_fragments": identities, "repair_work_package_hash_manifest": repair_hash_identity, "repair_result_fragments": repair_identities}, "outputs": {"teacher_candidate_results": {"path": str(output_path), "records": 1722, "bytes": len(payload.encode("utf-8")), "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}}, "records": {"expected_teacher_candidates": 1722, "protocol_repaired_candidates": len(overrides), "protocol_valid_candidates": 1722, "teacher_output_noop_candidates": noop_count, "non_noop_candidates": 1722 - noop_count}, "validation": {"work_package_identity_bound": True, "repair_work_package_identity_bound": True, "all_teacher_requests_returned_once": True, "candidate_queue_mapping_preserved": True, "all_raw_outputs_strict_protocol_valid": True, "required_gt_not_read_or_emitted": True}, "readiness": {"teacher_generation_protocol_audit_complete": True, "semantic_gt_candidate_review_ready": True, "frozen_retrieval_selection_complete": False, "query_sft_training_ready": False}, "complete": True}
    try:
        output_path.write_text(payload, encoding="utf-8", newline="\n")
        output_path.with_suffix(".sha256").write_text(f"{_sha256_file(output_path)}  {output_path.name}\n", encoding="utf-8", newline="\n")
        audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        audit_path.with_suffix(".sha256").write_text(f"{_sha256_file(audit_path)}  {audit_path.name}\n", encoding="utf-8", newline="\n")
    except (OSError, UnicodeError) as error:
        raise QuerySftFullTeacherAuditError("无法发布全量教师候选协议审计") from error
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-package-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repair-work-package-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    try:
        result = audit_query_sft_full_teacher_candidates(work_package_dir=args.work_package_dir, repair_work_package_dir=args.repair_work_package_dir, output_path=args.output, audit_path=args.audit)
    except QuerySftFullTeacherAuditError as error:
        parser.error(str(error))
    print(f"QUERY_SFT_FULL_TEACHER_PROTOCOL_AUDIT_OK candidates={result['records']['protocol_valid_candidates']}")


if __name__ == "__main__":
    main()
