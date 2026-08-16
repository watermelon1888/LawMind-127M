"""从已审核的全量 Query-SFT 输入发布 GT 隔离的教师候选工作包。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from rag.query.enhancement import QUERY_ENHANCEMENT_SCHEMA, QUERY_ENHANCEMENT_SYSTEM_PROMPT
except ImportError:  # 支持从 MiniMind 根目录运行。
    from ..rag.query.enhancement import QUERY_ENHANCEMENT_SCHEMA, QUERY_ENHANCEMENT_SYSTEM_PROMPT

try:
    from .assemble_query_sft_full_inputs import DEFAULT_OUTPUT_PATH as DEFAULT_INPUT_PATH
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset.assemble_query_sft_full_inputs import DEFAULT_OUTPUT_PATH as DEFAULT_INPUT_PATH


TEACHER_CANDIDATE_COUNT = 3
DEFAULT_INPUT_REPORT = DEFAULT_INPUT_PATH.parent / "query-sft-v1-input-structural-audit.json"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_PATH.parent / "query-sft-v1-teacher-candidate-work-package"
NOOP_FILENAME = "query-sft-v1-deterministic-noop-candidates.jsonl"
MANIFEST_FILENAME = "query-sft-v1-teacher-candidate-work-package.json"
HASH_FILENAME = "query-sft-v1-teacher-candidate-work-package.sha256"
INPUT_FIELDS = {"work_id", "source_id", "authoring_type", "query_original"}
QUEUE_FIELDS = {"candidate_id", "work_id", "query_original"}
NOOP_FIELDS = {"candidate_id", "work_id", "raw_output"}


class QuerySftFullTeacherPreparationError(RuntimeError):
    """表示全量 Query-SFT 教师候选工作包无法安全发布。"""


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


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftFullTeacherPreparationError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise QuerySftFullTeacherPreparationError(f"{description}必须是 JSON 对象")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    raise QuerySftFullTeacherPreparationError(f"输入候选不允许空行: {line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise QuerySftFullTeacherPreparationError(f"输入候选第 {line_number} 条必须是对象")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftFullTeacherPreparationError):
            raise
        raise QuerySftFullTeacherPreparationError(f"无法读取输入候选: {path}") from error
    return rows


def _verify_sidecar(path: Path, description: str) -> Path:
    sidecar = path.with_suffix(".sha256")
    try:
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftFullTeacherPreparationError(f"无法读取{description} SHA-256") from error
    if lines != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftFullTeacherPreparationError(f"{description} SHA-256 无效")
    return sidecar


def _payload(rows: list[dict[str, str]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)


def prepare_query_sft_full_teacher_candidates(
    *,
    input_path: Path = DEFAULT_INPUT_PATH,
    input_report_path: Path = DEFAULT_INPUT_REPORT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    """为每条全量输入发布三个彼此隔离的教师槽位和一个 no-op 基线。"""

    input_path = Path(input_path).resolve()
    input_report_path = Path(input_report_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise QuerySftFullTeacherPreparationError(f"输出目录必须不存在: {output_dir}")
    input_hash_path = _verify_sidecar(input_path, "全量输入候选")
    report_hash_path = _verify_sidecar(input_report_path, "全量输入审计报告")
    report = _load_json(input_report_path, "全量输入审计报告")
    if (
        report.get("pipeline") != "query_sft_full_input_structural_audit_v1"
        or report.get("release_status") != "full_teacher_candidate_input"
        or report.get("records", {}).get("final_input") != 574
        or report.get("readiness", {}).get("teacher_candidate_generation_ready") is not True
        or report.get("validation", {}).get("required_gt_not_emitted") is not True
        or report.get("complete") is not True
    ):
        raise QuerySftFullTeacherPreparationError("全量输入审计报告状态无效")
    input_identity = report.get("outputs", {}).get("input_candidate")
    if input_identity is not None and (
        input_identity.get("sha256") != _sha256_file(input_path) or input_identity.get("bytes") != input_path.stat().st_size
    ):
        raise QuerySftFullTeacherPreparationError("全量输入候选身份未与审计报告绑定")
    rows = _load_jsonl(input_path)
    if len(rows) != 574:
        raise QuerySftFullTeacherPreparationError("全量输入候选数量必须为 574")
    seen = set()
    for position, row in enumerate(rows, 1):
        if set(row) != INPUT_FIELDS or not isinstance(row.get("work_id"), str) or not isinstance(row.get("query_original"), str) or not row["query_original"] or row["work_id"] in seen:
            raise QuerySftFullTeacherPreparationError(f"全量输入候选第 {position} 条字段无效")
        seen.add(row["work_id"])
    payloads: dict[str, str] = {}
    noops = []
    for batch in range(1, 9):
        start = (batch - 1) * 72
        batch_rows = rows[start : start + 72]
        queue = []
        for row in batch_rows:
            for slot in range(1, TEACHER_CANDIDATE_COUNT + 1):
                queue.append({"candidate_id": f"{row['work_id']}/teacher-{slot}", "work_id": row["work_id"], "query_original": row["query_original"]})
            noops.append({"candidate_id": f"{row['work_id']}/noop", "work_id": row["work_id"], "raw_output": json.dumps({"rewrite": row["query_original"], "expansion_terms": [], "subqueries": []}, ensure_ascii=False, separators=(",", ":"), allow_nan=False)})
        if any(set(row) != QUEUE_FIELDS for row in queue):
            raise AssertionError("教师队列字段意外变化")
        payloads[f"teacher-queue/batch-{batch:02d}.jsonl"] = _payload(queue)
    if len(noops) != 574 or len({row["candidate_id"] for row in noops}) != 574:
        raise AssertionError("no-op 基线数量或身份无效")
    payloads[NOOP_FILENAME] = _payload(noops)
    manifest = {
        "pipeline": "query_sft_full_teacher_candidate_work_package_v1",
        "release_status": "full_teacher_candidate_generation_pending",
        "inputs": {"input_candidate": {**_identity(input_path, records=574), "hash_manifest": _identity(input_hash_path)}, "input_audit": {**_identity(input_report_path), "hash_manifest": _identity(report_hash_path)}},
        "generation_protocol": {"teacher_visible_record_fields": ["work_id", "query_original"], "teacher_candidates_per_input": 3, "candidate_slots_are_independent": True, "teacher_prompt_builder": "rag.query.enhancement.build_query_enhancement_prompt", "system_prompt_sha256": hashlib.sha256(QUERY_ENHANCEMENT_SYSTEM_PROMPT.encode("utf-8")).hexdigest(), "output_schema": QUERY_ENHANCEMENT_SCHEMA, "forbidden_teacher_context": ["source_id", "authoring_type", "required_chunk_ids", "law_text", "answer", "evaluation_question", "retrieval_score", "other_candidate_output"]},
        "records": {"inputs": 574, "teacher_candidate_requests": 1722, "deterministic_noop_candidates": 574, "batches": 8},
        "validation": {"input_identity_bound": True, "teacher_queue_contains_only_original_query": True, "three_independent_teacher_slots_per_input": True, "deterministic_noop_has_strict_protocol_shape": True, "teacher_prompt_excludes_gt_and_retrieval_feedback": True},
        "readiness": {"teacher_candidate_work_package_ready": True, "teacher_generation_completed": False, "candidate_protocol_audit_completed": False, "frozen_retrieval_selection_complete": False, "query_sft_training_ready": False},
        "complete": True,
    }
    payloads[MANIFEST_FILENAME] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    try:
        output_dir.mkdir(parents=True)
        for name, payload in payloads.items():
            path = output_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8", newline="\n")
        (output_dir / HASH_FILENAME).write_text("".join(f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()}  {name}\n" for name, payload in payloads.items()), encoding="utf-8", newline="\n")
    except (OSError, UnicodeError) as error:
        raise QuerySftFullTeacherPreparationError("无法发布全量教师候选工作包") from error
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--input-report", type=Path, default=DEFAULT_INPUT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        manifest = prepare_query_sft_full_teacher_candidates(input_path=args.input, input_report_path=args.input_report, output_dir=args.output_dir)
    except QuerySftFullTeacherPreparationError as error:
        parser.error(str(error))
    print(f"QUERY_SFT_FULL_TEACHER_WORK_PACKAGE_OK requests={manifest['records']['teacher_candidate_requests']}")


if __name__ == "__main__":
    main()
