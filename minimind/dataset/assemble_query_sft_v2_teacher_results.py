"""合并本地多代理教师分片，并发布可由协议审计消费的 v2 教师运行产物。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rag.query.enhancement import parse_and_validate_query_enhancement

from .audit_query_sft_v2_teacher_candidates import (
    RESULT_FIELDS,
    RESULT_FILENAME,
    RUNTIME_FILENAME,
    _identity,
    _sha256_file,
    _verify_work_package,
)
from .prepare_query_sft_v2_teacher_candidates import (
    DEFAULT_OUTPUT_DIR as DEFAULT_TEACHER_WORK_PACKAGE_DIR,
)


DEFAULT_FRAGMENTS_DIR = (
    DEFAULT_TEACHER_WORK_PACKAGE_DIR.parent / "query-sft-v2-local-teacher-results"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_TEACHER_WORK_PACKAGE_DIR.parent / "query-sft-v2-local-teacher-generation-r1"
)
RUNTIME_IDENTITY_FIELDS = (
    "teacher_model_id",
    "teacher_endpoint_or_local_checkpoint",
    "decoding_temperature",
    "decoding_seed_or_nondeterminism_declaration",
    "max_output_tokens",
)


class QuerySftV2TeacherAssemblyError(RuntimeError):
    """表示本地教师分片不能安全合并为一次可审计运行。"""


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    raise QuerySftV2TeacherAssemblyError(f"{label}不允许空行: {number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise QuerySftV2TeacherAssemblyError(f"{label}第 {number} 条必须是对象")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftV2TeacherAssemblyError):
            raise
        raise QuerySftV2TeacherAssemblyError(f"无法读取{label}") from error
    return rows


def _payload(rows: list[dict[str, object]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in rows
    )


def assemble_query_sft_v2_teacher_results(
    *,
    work_package_dir: Path = DEFAULT_TEACHER_WORK_PACKAGE_DIR,
    fragments_dir: Path = DEFAULT_FRAGMENTS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    teacher_model_id: str = "Codex GPT-5",
    teacher_endpoint_or_local_checkpoint: str = "local Codex multi-agent generation",
    decoding_temperature: str = "agentic_generation_not_parameterized",
    decoding_seed_or_nondeterminism_declaration: str = "independent_agent_contexts; nondeterministic",
    max_output_tokens: int = 336,
) -> dict[str, object]:
    """验证八个盲队列分片，合并为一次不可覆盖的本地教师运行。"""

    work_directory = Path(work_package_dir).resolve()
    fragment_directory = Path(fragments_dir).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise QuerySftV2TeacherAssemblyError("本地教师运行输出目录必须不存在")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (
            teacher_model_id,
            teacher_endpoint_or_local_checkpoint,
            decoding_temperature,
            decoding_seed_or_nondeterminism_declaration,
        )
    ) or type(max_output_tokens) is not int or max_output_tokens <= 0:
        raise QuerySftV2TeacherAssemblyError("教师运行身份参数无效")
    work_hash, expected = _verify_work_package(work_directory)
    merged: list[dict[str, object]] = []
    fragments: dict[str, object] = {}
    for batch in range(1, 9):
        path = fragment_directory / f"batch-{batch:02d}.jsonl"
        rows = _load_jsonl(path, f"本地教师分片第 {batch} 批")
        expected_ids = {
            row["candidate_id"]
            for row in _load_jsonl(
                work_directory / f"teacher-queue/batch-{batch:02d}.jsonl",
                f"教师盲队列第 {batch} 批",
            )
        }
        if len(rows) != len(expected_ids):
            raise QuerySftV2TeacherAssemblyError(f"本地教师分片第 {batch} 批数量不闭合")
        seen = set()
        for row in rows:
            candidate_id = row.get("candidate_id")
            if (
                tuple(row) != RESULT_FIELDS
                or not isinstance(candidate_id, str)
                or candidate_id not in expected_ids
                or candidate_id in seen
                or row.get("work_id") != expected[candidate_id]["work_id"]
                or not isinstance(row.get("raw_output"), str)
            ):
                raise QuerySftV2TeacherAssemblyError(f"本地教师分片第 {batch} 批记录无效")
            try:
                parse_and_validate_query_enhancement(row["raw_output"])
            except (TypeError, ValueError) as error:
                raise QuerySftV2TeacherAssemblyError(f"本地教师分片第 {batch} 批含不合法协议输出") from error
            seen.add(candidate_id)
            merged.append(row)
        if seen != expected_ids:
            raise QuerySftV2TeacherAssemblyError(f"本地教师分片第 {batch} 批未完整覆盖盲队列")
        fragments[f"batch_{batch:02d}"] = _identity(path, records=len(rows))
    if len(merged) != 1722:
        raise QuerySftV2TeacherAssemblyError("本地教师分片合并后未得到 1722 个候选")
    merged.sort(key=lambda row: row["candidate_id"])
    result_payload = _payload(merged)
    result_path = destination / RESULT_FILENAME
    runtime_path = destination / RUNTIME_FILENAME
    runtime = {
        "schema_version": "1.0",
        "pipeline": "query_sft_v2_teacher_generation_run",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": {
            "teacher_work_package_hash_manifest": _identity(work_hash),
            "teacher_result_fragments": fragments,
        },
        "runtime_identity": {
            "teacher_model_id": teacher_model_id,
            "teacher_endpoint_or_local_checkpoint": teacher_endpoint_or_local_checkpoint,
            "decoding_temperature": decoding_temperature,
            "decoding_seed_or_nondeterminism_declaration": decoding_seed_or_nondeterminism_declaration,
            "max_output_tokens": max_output_tokens,
        },
        "records": {"candidate_results": len(merged), "batches": 8},
        "validation": {
            "teacher_only_received_signed_blind_queue": True,
            "candidate_slots_generated_independently": True,
            "raw_outputs_strict_protocol_valid_before_assembly": True,
        },
        "complete": True,
    }
    try:
        destination.mkdir(parents=True)
        result_path.write_text(result_payload, encoding="utf-8", newline="\n")
        result_path.with_suffix(".sha256").write_text(
            f"{_sha256_file(result_path)}  {result_path.name}\n",
            encoding="utf-8",
            newline="\n",
        )
        runtime_path.write_text(
            json.dumps(runtime, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        runtime_path.with_suffix(".sha256").write_text(
            f"{_sha256_file(runtime_path)}  {runtime_path.name}\n",
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, UnicodeError) as error:
        for path in destination.glob("*") if destination.exists() else ():
            path.unlink(missing_ok=True)
        try:
            destination.rmdir()
        except OSError:
            pass
        raise QuerySftV2TeacherAssemblyError("无法发布本地教师运行产物") from error
    return runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-package-dir", type=Path, default=DEFAULT_TEACHER_WORK_PACKAGE_DIR)
    parser.add_argument("--fragments-dir", type=Path, default=DEFAULT_FRAGMENTS_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--teacher-model-id", default="Codex GPT-5")
    parser.add_argument("--teacher-endpoint-or-local-checkpoint", default="local Codex multi-agent generation")
    parser.add_argument("--decoding-temperature", default="agentic_generation_not_parameterized")
    parser.add_argument("--decoding-seed-or-nondeterminism-declaration", default="independent_agent_contexts; nondeterministic")
    parser.add_argument("--max-output-tokens", type=int, default=336)
    args = parser.parse_args()
    try:
        runtime = assemble_query_sft_v2_teacher_results(
            work_package_dir=args.work_package_dir,
            fragments_dir=args.fragments_dir,
            output_dir=args.output_dir,
            teacher_model_id=args.teacher_model_id,
            teacher_endpoint_or_local_checkpoint=args.teacher_endpoint_or_local_checkpoint,
            decoding_temperature=args.decoding_temperature,
            decoding_seed_or_nondeterminism_declaration=args.decoding_seed_or_nondeterminism_declaration,
            max_output_tokens=args.max_output_tokens,
        )
    except QuerySftV2TeacherAssemblyError as error:
        parser.error(str(error))
    print(f"QUERY_SFT_V2_TEACHER_ASSEMBLY_OK candidates={runtime['records']['candidate_results']}")


if __name__ == "__main__":
    main()
