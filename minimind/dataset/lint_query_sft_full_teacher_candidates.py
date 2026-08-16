"""收集全量 Query-SFT 教师候选的协议失败项，发布最小修订队列。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rag.query.enhancement import parse_and_validate_query_enhancement

try:
    from .audit_query_sft_full_teacher_candidates import (
        DEFAULT_OUTPUT_DIR,
        QUEUE_FIELDS,
        RESULT_FIELDS,
        RESULT_SOURCES,
        _identity,
        _load_jsonl,
        _verify_work_package,
    )
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset.audit_query_sft_full_teacher_candidates import (
        DEFAULT_OUTPUT_DIR,
        QUEUE_FIELDS,
        RESULT_FIELDS,
        RESULT_SOURCES,
        _identity,
        _load_jsonl,
        _verify_work_package,
    )


REPAIR_QUEUE_FILENAME = "query-sft-v1-teacher-protocol-repair-queue-r1.jsonl"
REPORT_FILENAME = "query-sft-v1-teacher-protocol-lint-r1.json"
REPAIR_FIELDS = {"candidate_id", "work_id", "query_original", "failure_code"}


class QuerySftFullTeacherLintError(RuntimeError):
    """表示教师候选片段无法按冻结队列进行协议检查。"""


def _batch_ids(expected: dict[str, dict[str, str]], batch: int, slot: int) -> set[str]:
    first = (batch - 1) * 72 + 1
    last = min(batch * 72, 574)
    return {
        candidate_id
        for candidate_id, row in expected.items()
        if first <= int(row["work_id"].rsplit(":", 1)[1]) <= last
        and (slot == 0 or candidate_id.endswith(f"/teacher-{slot}"))
    }


def lint_query_sft_full_teacher_candidates(
    *, work_package_dir: Path = DEFAULT_OUTPUT_DIR, output_dir: Path | None = None
) -> dict[str, object]:
    """保留全部结构合格片段，逐项收集运行时协议失败而不生成训练候选。"""

    directory = Path(work_package_dir).resolve()
    output_dir = Path(output_dir or directory).resolve()
    queue_path = output_dir / REPAIR_QUEUE_FILENAME
    report_path = output_dir / REPORT_FILENAME
    if any(path.exists() for path in (queue_path, queue_path.with_suffix(".sha256"), report_path, report_path.with_suffix(".sha256"))):
        raise QuerySftFullTeacherLintError("教师协议 lint 输出已存在")
    try:
        work_hash = _verify_work_package(directory)
    except Exception as error:
        raise QuerySftFullTeacherLintError("教师工作包身份无效") from error
    expected: dict[str, dict[str, str]] = {}
    order = []
    for batch in range(1, 9):
        for row in _load_jsonl(directory / "teacher-queue" / f"batch-{batch:02d}.jsonl", f"第 {batch} 批教师队列"):
            if set(row) != QUEUE_FIELDS or not isinstance(row.get("candidate_id"), str) or row["candidate_id"] in expected:
                raise QuerySftFullTeacherLintError("教师队列字段或 candidate_id 无效")
            expected[row["candidate_id"]] = row
            order.append(row["candidate_id"])
    if len(expected) != 1722:
        raise QuerySftFullTeacherLintError("教师队列总数必须为 1722")
    seen = set()
    repair_rows = []
    fragments = {}
    valid = 0
    for (batch, slot), filename in RESULT_SOURCES.items():
        path = directory / "teacher-results" / filename
        rows = _load_jsonl(path, f"教师结果 {filename}")
        expected_ids = _batch_ids(expected, batch, slot)
        if len(rows) != len(expected_ids):
            raise QuerySftFullTeacherLintError(f"教师结果 {filename} 槽位数量不闭合")
        for row in rows:
            candidate_id = row.get("candidate_id")
            if (
                set(row) != RESULT_FIELDS
                or candidate_id not in expected_ids
                or candidate_id in seen
                or row.get("work_id") != expected[candidate_id]["work_id"]
                or not isinstance(row.get("raw_output"), str)
            ):
                raise QuerySftFullTeacherLintError(f"教师结果 {filename} 映射无效")
            seen.add(candidate_id)
            try:
                parse_and_validate_query_enhancement(row["raw_output"])
            except (TypeError, ValueError):
                repair_rows.append({"candidate_id": candidate_id, "work_id": row["work_id"], "query_original": expected[candidate_id]["query_original"], "failure_code": "query_enhancement_protocol_invalid"})
            else:
                valid += 1
        fragments[filename] = _identity(path, records=len(rows))
    if set(seen) != set(expected):
        raise QuerySftFullTeacherLintError("教师结果未覆盖全部槽位")
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in repair_rows)
    report = {
        "pipeline": "query_sft_full_teacher_protocol_lint_v1",
        "inputs": {"work_package_hash_manifest": _identity(work_hash), "teacher_result_fragments": fragments},
        "outputs": {"repair_queue": {"path": str(queue_path), "records": len(repair_rows), "bytes": len(payload.encode("utf-8")), "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}},
        "records": {"teacher_candidates": 1722, "protocol_valid": valid, "protocol_invalid": len(repair_rows)},
        "validation": {"all_teacher_slots_structurally_covered": True, "protocol_invalid_candidates_isolated": True, "required_gt_not_read_or_emitted": True},
        "readiness": {"minimal_protocol_repair_ready": bool(repair_rows), "full_protocol_audit_complete": not repair_rows, "semantic_gt_candidate_review_ready": not repair_rows, "query_sft_training_ready": False},
        "complete": True,
    }
    try:
        queue_path.write_text(payload, encoding="utf-8", newline="\n")
        queue_path.with_suffix(".sha256").write_text(f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()}  {queue_path.name}\n", encoding="utf-8", newline="\n")
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        report_path.with_suffix(".sha256").write_text(f"{hashlib.sha256(report_path.read_bytes()).hexdigest()}  {report_path.name}\n", encoding="utf-8", newline="\n")
    except (OSError, UnicodeError) as error:
        raise QuerySftFullTeacherLintError("无法发布教师协议修订队列") from error
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-package-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        report = lint_query_sft_full_teacher_candidates(work_package_dir=args.work_package_dir, output_dir=args.output_dir)
    except QuerySftFullTeacherLintError as error:
        parser.error(str(error))
    print(f"QUERY_SFT_FULL_TEACHER_PROTOCOL_LINT_OK valid={report['records']['protocol_valid']} invalid={report['records']['protocol_invalid']}")


if __name__ == "__main__":
    main()
