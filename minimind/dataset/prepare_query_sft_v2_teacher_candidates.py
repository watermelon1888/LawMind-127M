"""为 Query-SFT v2 发布 GT 隔离的教师候选队列与 no-op 基线。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rag.query.enhancement import QUERY_ENHANCEMENT_SCHEMA, QUERY_ENHANCEMENT_SYSTEM_PROMPT

from .prepare_query_sft_v2 import (
    DEFAULT_OUTPUT_DIR as DEFAULT_WORK_PACKAGE_DIR,
    HASH_FILENAME as INPUT_HASH_FILENAME,
    INPUT_FILENAME,
    WORK_PACKAGE_FILENAME,
)


DEFAULT_OUTPUT_DIR = DEFAULT_WORK_PACKAGE_DIR / "query-sft-v2-teacher-candidate-work-package"
MANIFEST_FILENAME = "query-sft-v2-teacher-candidate-work-package.json"
HASH_FILENAME = "query-sft-v2-teacher-candidate-work-package.sha256"
NOOP_FILENAME = "query-sft-v2-deterministic-noop-candidates.jsonl"
INPUT_FIELDS = ("work_id", "source_id", "authoring_type", "query_original")
QUEUE_FIELDS = ("candidate_id", "work_id", "query_original")
NOOP_FIELDS = ("candidate_id", "work_id", "raw_output")


class QuerySftV2TeacherPreparationError(RuntimeError):
    """表示 Query-SFT v2 教师候选队列无法安全发布。"""


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


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftV2TeacherPreparationError(f"无法读取{label}") from error
    if not isinstance(value, dict):
        raise QuerySftV2TeacherPreparationError(f"{label}必须是 JSON 对象")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    raise QuerySftV2TeacherPreparationError(f"教师输入不允许空行: {number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise QuerySftV2TeacherPreparationError(f"教师输入第 {number} 条必须是对象")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftV2TeacherPreparationError):
            raise
        raise QuerySftV2TeacherPreparationError("无法读取教师输入") from error
    return rows


def _verify_package(directory: Path) -> Path:
    hash_path = directory / INPUT_HASH_FILENAME
    try:
        entries = {
            name: digest
            for line in hash_path.read_text(encoding="utf-8").splitlines()
            for digest, name in [line.split("  ", 1)]
        }
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise QuerySftV2TeacherPreparationError("v2 工作包 SHA-256 清单无效") from error
    required = {INPUT_FILENAME, WORK_PACKAGE_FILENAME}
    if not required <= set(entries):
        raise QuerySftV2TeacherPreparationError("v2 工作包缺少必要身份")
    for name, digest in entries.items():
        path = directory / name
        if not path.is_file() or _sha256_file(path) != digest:
            raise QuerySftV2TeacherPreparationError(f"v2 工作包身份已变化: {name}")
    manifest = _load_json(directory / WORK_PACKAGE_FILENAME, "v2 工作包 manifest")
    if (
        manifest.get("pipeline") != "query_sft_v2_work_package"
        or manifest.get("records", {}).get("query_inputs") != 574
        or manifest.get("readiness", {}).get("teacher_candidate_generation_ready") is not True
        or manifest.get("complete") is not True
    ):
        raise QuerySftV2TeacherPreparationError("v2 工作包状态无效")
    return hash_path


def _payload(rows: list[dict[str, object]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in rows
    )


def prepare_query_sft_v2_teacher_candidates(
    *, work_package_dir: Path = DEFAULT_WORK_PACKAGE_DIR, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> dict[str, object]:
    """为每条 v2 输入发布三个独立教师槽位和确定性 no-op。"""

    work_package_dir = Path(work_package_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise QuerySftV2TeacherPreparationError("v2 教师工作包输出目录必须不存在")
    work_hash = _verify_package(work_package_dir)
    inputs = _load_jsonl(work_package_dir / INPUT_FILENAME)
    if len(inputs) != 574:
        raise QuerySftV2TeacherPreparationError("v2 教师输入必须为 574 条")
    seen = set()
    for position, row in enumerate(inputs, 1):
        if tuple(row) != INPUT_FIELDS or not isinstance(row.get("work_id"), str) or row["work_id"] in seen or not isinstance(row.get("query_original"), str) or not row["query_original"]:
            raise QuerySftV2TeacherPreparationError(f"v2 教师输入第 {position} 条无效")
        seen.add(row["work_id"])

    payloads: dict[str, str] = {}
    noops = []
    for batch in range(1, 9):
        rows = inputs[(batch - 1) * 72 : batch * 72]
        queue = []
        for row in rows:
            for slot in range(1, 4):
                queue.append({"candidate_id": f"{row['work_id']}/teacher-{slot}", "work_id": row["work_id"], "query_original": row["query_original"]})
            noops.append({"candidate_id": f"{row['work_id']}/noop", "work_id": row["work_id"], "raw_output": json.dumps({"rewrite": row["query_original"], "expansion_terms": [], "subqueries": []}, ensure_ascii=False, separators=(",", ":"))})
        if any(tuple(row) != QUEUE_FIELDS for row in queue):
            raise AssertionError("v2 教师队列字段顺序意外变化")
        payloads[f"teacher-queue/batch-{batch:02d}.jsonl"] = _payload(queue)
    if len(noops) != 574 or any(tuple(row) != NOOP_FIELDS for row in noops):
        raise AssertionError("v2 no-op 基线无效")
    payloads[NOOP_FILENAME] = _payload(noops)
    manifest = {
        "schema_version": "1.0",
        "pipeline": "query_sft_v2_teacher_candidate_work_package",
        "release_status": "teacher_generation_pending",
        "inputs": {"v2_work_package_hash_manifest": _identity(work_hash), "input_candidate": _identity(work_package_dir / INPUT_FILENAME, records=574)},
        "generation_protocol": {
            "teacher_visible_record_fields": ["candidate_id", "work_id", "query_original"],
            "teacher_candidates_per_input": 3,
            "candidate_slots_are_independent": True,
            "prompt_builder": "rag.query.enhancement.build_query_enhancement_prompt",
            "system_prompt_sha256": hashlib.sha256(QUERY_ENHANCEMENT_SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "output_schema": QUERY_ENHANCEMENT_SCHEMA,
            "runtime_identity_required_before_execution": [
                "teacher_model_id",
                "teacher_endpoint_or_local_checkpoint",
                "decoding_temperature",
                "decoding_seed_or_nondeterminism_declaration",
                "max_output_tokens",
            ],
            "forbidden_teacher_context": ["source_id", "required_chunk_ids", "法条正文", "法律答案", "评估题", "检索结果", "检索分数", "旧版候选", "authoring_type"],
        },
        "records": {"query_inputs": 574, "teacher_candidate_requests": 1722, "deterministic_noop_candidates": 574},
        "readiness": {"teacher_generation_ready": True, "semantic_review_ready": False, "frozen_retrieval_selection_ready": False, "training_ready": False},
        "complete": True,
    }
    payloads[MANIFEST_FILENAME] = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        output_dir.mkdir(parents=True)
        for name, payload in payloads.items():
            path = output_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8", newline="\n")
        (output_dir / HASH_FILENAME).write_text(
            "".join(f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()}  {name}\n" for name, payload in payloads.items()),
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, UnicodeError) as error:
        raise QuerySftV2TeacherPreparationError("无法发布 v2 教师候选工作包") from error
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-package-dir", type=Path, default=DEFAULT_WORK_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        manifest = prepare_query_sft_v2_teacher_candidates(
            work_package_dir=args.work_package_dir, output_dir=args.output_dir
        )
    except QuerySftV2TeacherPreparationError as error:
        parser.error(str(error))
    print(f"QUERY_SFT_V2_TEACHER_WORK_PACKAGE_OK candidates={manifest['records']['teacher_candidate_requests']}")


if __name__ == "__main__":
    main()
