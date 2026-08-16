"""将全量教师协议失败项拆分为 GT 隔离的最小修订队列。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .lint_query_sft_full_teacher_candidates import (
        DEFAULT_OUTPUT_DIR,
        REPAIR_FIELDS,
        REPAIR_QUEUE_FILENAME,
        REPORT_FILENAME,
    )
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset.lint_query_sft_full_teacher_candidates import (
        DEFAULT_OUTPUT_DIR,
        REPAIR_FIELDS,
        REPAIR_QUEUE_FILENAME,
        REPORT_FILENAME,
    )


DEFAULT_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "teacher-protocol-repairs-r1"
MANIFEST_FILENAME = "query-sft-v1-teacher-protocol-repair-work-package.json"
HASH_FILENAME = "query-sft-v1-teacher-protocol-repair-work-package.sha256"


class QuerySftFullTeacherRepairPreparationError(RuntimeError):
    """表示教师协议修订工作包无法安全发布。"""


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


def _verify_sidecar(path: Path, label: str) -> Path:
    sidecar = path.with_suffix(".sha256")
    try:
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftFullTeacherRepairPreparationError(f"无法读取{label} SHA-256") from error
    if lines != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftFullTeacherRepairPreparationError(f"{label} SHA-256 无效")
    return sidecar


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftFullTeacherRepairPreparationError(f"无法读取{label}") from error
    if not isinstance(value, dict):
        raise QuerySftFullTeacherRepairPreparationError(f"{label}必须是 JSON 对象")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                raise QuerySftFullTeacherRepairPreparationError(f"修订队列不允许空行: {number}")
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != REPAIR_FIELDS:
                raise QuerySftFullTeacherRepairPreparationError(f"修订队列第 {number} 条字段无效")
            rows.append(row)
    return rows


def prepare_query_sft_full_teacher_protocol_repairs(
    *,
    lint_dir: Path = DEFAULT_OUTPUT_DIR.parent,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    """发布三个小队列，修订者仅可见协议失败项的原始 query。"""

    lint_dir = Path(lint_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise QuerySftFullTeacherRepairPreparationError("修订工作包输出目录必须不存在")
    queue_path = lint_dir / REPAIR_QUEUE_FILENAME
    report_path = lint_dir / REPORT_FILENAME
    queue_hash = _verify_sidecar(queue_path, "协议修订队列")
    report_hash = _verify_sidecar(report_path, "协议 lint 报告")
    report = _load_json(report_path, "协议 lint 报告")
    rows = _load_jsonl(queue_path)
    if report.get("pipeline") != "query_sft_full_teacher_protocol_lint_v1" or report.get("records", {}).get("protocol_invalid") != len(rows) or len(rows) != 46:
        raise QuerySftFullTeacherRepairPreparationError("协议 lint 报告与修订队列不一致")
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise QuerySftFullTeacherRepairPreparationError("协议修订队列 candidate_id 重复")
    boundaries = (16, 31, 46)
    payloads: dict[str, str] = {}
    plan = []
    start = 0
    for index, end in enumerate(boundaries, 1):
        chunk = rows[start:end]
        name = f"repair-queue/part-{index:02d}.jsonl"
        payloads[name] = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in chunk)
        plan.append({"part": index, "records": len(chunk), "queue": name, "candidate_id_first": chunk[0]["candidate_id"], "candidate_id_last": chunk[-1]["candidate_id"]})
        start = end
    manifest = {
        "pipeline": "query_sft_full_teacher_protocol_repair_work_package_v1",
        "release_status": "teacher_protocol_repair_pending",
        "inputs": {"lint_report": {**_identity(report_path), "hash_manifest": _identity(report_hash)}, "repair_queue": {**_identity(queue_path, records=len(rows)), "hash_manifest": _identity(queue_hash)}},
        "policy": {"repairer_visible_fields": ["candidate_id", "work_id", "query_original", "failure_code"], "forbidden_context": ["source_id", "authoring_type", "required_chunk_ids", "law_text", "answer", "evaluation_question", "retrieval_score", "other_candidate_output"], "output_fields": ["candidate_id", "work_id", "raw_output"], "repair_must_pass_runtime_protocol": True},
        "records": {"protocol_invalid_candidates": len(rows), "parts": len(plan)},
        "parts": plan,
        "readiness": {"minimal_protocol_repair_ready": True, "semantic_gt_candidate_review_ready": False, "query_sft_training_ready": False},
        "complete": True,
    }
    payloads[MANIFEST_FILENAME] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    output_dir.mkdir(parents=True)
    for name, payload in payloads.items():
        path = output_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8", newline="\n")
    (output_dir / HASH_FILENAME).write_text("".join(f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()}  {name}\n" for name, payload in payloads.items()), encoding="utf-8", newline="\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lint-dir", type=Path, default=DEFAULT_OUTPUT_DIR.parent)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        result = prepare_query_sft_full_teacher_protocol_repairs(lint_dir=args.lint_dir, output_dir=args.output_dir)
    except QuerySftFullTeacherRepairPreparationError as error:
        parser.error(str(error))
    print(f"QUERY_SFT_FULL_TEACHER_PROTOCOL_REPAIR_PACKAGE_OK records={result['records']['protocol_invalid_candidates']}")


if __name__ == "__main__":
    main()
