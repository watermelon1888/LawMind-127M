"""发布全量 Query-SFT 教师候选的独立语义与 GT 审核工作包。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rag.query.enhancement import parse_and_validate_query_enhancement

try:
    from .audit_query_sft_full_teacher_candidates import (
        AUDIT_FILENAME,
        DEFAULT_OUTPUT_DIR as DEFAULT_TEACHER_WORK_PACKAGE_DIR,
        RESULT_FILENAME,
    )
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset.audit_query_sft_full_teacher_candidates import (
        AUDIT_FILENAME,
        DEFAULT_OUTPUT_DIR as DEFAULT_TEACHER_WORK_PACKAGE_DIR,
        RESULT_FILENAME,
    )


DEFAULT_INPUT_WORK_PACKAGE_DIR = DEFAULT_TEACHER_WORK_PACKAGE_DIR.parent
DEFAULT_OUTPUT_DIR = DEFAULT_TEACHER_WORK_PACKAGE_DIR / "query-sft-v1-candidate-semantic-review-work-package"
INPUT_FILENAME = "query-sft-v1-input-candidate-r1.jsonl"
MANIFEST_FILENAME = "query-sft-v1-candidate-semantic-review-work-package.json"
HASH_FILENAME = "query-sft-v1-candidate-semantic-review-work-package.sha256"
INPUT_FIELDS = {"work_id", "source_id", "authoring_type", "query_original"}
REFERENCE_FIELDS = {"work_id", "source_id", "source_query", "required_chunk_ids"}
RESULT_FIELDS = {"candidate_id", "work_id", "raw_output"}
REVIEW_FIELDS = {"candidate_id", "work_id", "query_original", "source_query", "required_chunk_ids", "candidate_target"}


class QuerySftFullCandidateReviewPreparationError(RuntimeError):
    """表示全量候选语义与 GT 审核工作包无法安全发布。"""


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
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftFullCandidateReviewPreparationError(f"无法读取{label} SHA-256") from error
    if lines != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftFullCandidateReviewPreparationError(f"{label} SHA-256 无效")
    return sidecar


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftFullCandidateReviewPreparationError(f"无法读取{label}") from error
    if not isinstance(value, dict):
        raise QuerySftFullCandidateReviewPreparationError(f"{label}必须是 JSON 对象")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    raise QuerySftFullCandidateReviewPreparationError(f"{label}不允许空行: {number}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise QuerySftFullCandidateReviewPreparationError(f"{label}第 {number} 条必须是对象")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftFullCandidateReviewPreparationError):
            raise
        raise QuerySftFullCandidateReviewPreparationError(f"无法读取{label}") from error
    return rows


def _payload(rows: list[dict[str, object]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)


def prepare_query_sft_full_candidate_semantic_review(
    *,
    input_work_package_dir: Path = DEFAULT_INPUT_WORK_PACKAGE_DIR,
    teacher_work_package_dir: Path = DEFAULT_TEACHER_WORK_PACKAGE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    """为八个独立审核批次绑定原始 query、候选目标和已审核的 GT 标识。"""

    input_directory = Path(input_work_package_dir).resolve()
    teacher_directory = Path(teacher_work_package_dir).resolve()
    output_directory = Path(output_dir).resolve()
    if output_directory.exists():
        raise QuerySftFullCandidateReviewPreparationError("候选语义审核工作包输出目录必须不存在")
    input_path = input_directory / INPUT_FILENAME
    result_path = teacher_directory / RESULT_FILENAME
    protocol_path = teacher_directory / AUDIT_FILENAME
    input_hash = _verify_sidecar(input_path, "正式输入候选")
    result_hash = _verify_sidecar(result_path, "教师候选结果")
    protocol_hash = _verify_sidecar(protocol_path, "教师候选协议审计")
    protocol = _load_json(protocol_path, "教师候选协议审计")
    if (
        protocol.get("pipeline") != "query_sft_full_teacher_candidate_protocol_audit_v1"
        or protocol.get("complete") is not True
        or protocol.get("records", {}).get("protocol_valid_candidates") != 1722
        or protocol.get("records", {}).get("protocol_repaired_candidates") != 46
    ):
        raise QuerySftFullCandidateReviewPreparationError("教师候选协议审计状态无效")

    inputs: dict[str, dict[str, str]] = {}
    for row in _load_jsonl(input_path, "正式输入候选"):
        work_id = row.get("work_id")
        if set(row) != INPUT_FIELDS or not isinstance(work_id, str) or work_id in inputs:
            raise QuerySftFullCandidateReviewPreparationError("正式输入候选映射无效")
        inputs[work_id] = row
    if len(inputs) != 574:
        raise QuerySftFullCandidateReviewPreparationError("正式输入候选数量必须为 574")

    references: dict[str, dict[str, Any]] = {}
    for batch in range(1, 9):
        for row in _load_jsonl(input_directory / "audit-reference" / f"batch-{batch:02d}.jsonl", f"GT 审核引用第 {batch} 批"):
            work_id = row.get("work_id")
            required = row.get("required_chunk_ids")
            if (
                set(row) != REFERENCE_FIELDS
                or work_id not in inputs
                or work_id in references
                or row.get("source_id") != inputs[work_id]["source_id"]
                or not isinstance(required, list)
                or not 1 <= len(required) <= 3
                or any(not isinstance(chunk_id, str) or not chunk_id for chunk_id in required)
            ):
                raise QuerySftFullCandidateReviewPreparationError("GT 审核引用映射无效")
            references[work_id] = row
    if set(references) != set(inputs):
        raise QuerySftFullCandidateReviewPreparationError("GT 审核引用未覆盖全部正式输入")

    results = _load_jsonl(result_path, "教师候选结果")
    batches: dict[int, list[dict[str, object]]] = {batch: [] for batch in range(1, 9)}
    seen = set()
    for row in results:
        candidate_id = row.get("candidate_id")
        work_id = row.get("work_id")
        if set(row) != RESULT_FIELDS or not isinstance(candidate_id, str) or candidate_id in seen or work_id not in inputs:
            raise QuerySftFullCandidateReviewPreparationError("教师候选结果映射无效")
        try:
            enhancement = parse_and_validate_query_enhancement(row["raw_output"])
        except (TypeError, ValueError) as error:
            raise QuerySftFullCandidateReviewPreparationError(f"教师候选未通过运行时协议: {candidate_id}") from error
        index = int(work_id.rsplit(":", 1)[1])
        batch = (index - 1) // 72 + 1
        batches[batch].append(
            {
                "candidate_id": candidate_id,
                "work_id": work_id,
                "query_original": inputs[work_id]["query_original"],
                "source_query": references[work_id]["source_query"],
                "required_chunk_ids": references[work_id]["required_chunk_ids"],
                "candidate_target": {
                    "rewrite": enhancement.rewrite,
                    "expansion_terms": list(enhancement.expansion_terms),
                    "subqueries": list(enhancement.subqueries),
                },
            }
        )
        seen.add(candidate_id)
    if len(seen) != 1722 or any(set(row) != REVIEW_FIELDS for rows in batches.values() for row in rows):
        raise QuerySftFullCandidateReviewPreparationError("教师候选审核视图未闭合")

    payloads = {}
    parts = []
    for batch, rows in batches.items():
        rows.sort(key=lambda row: row["candidate_id"])
        expected_records = 216 if batch < 8 else 210
        if len(rows) != expected_records:
            raise QuerySftFullCandidateReviewPreparationError(f"候选审核第 {batch} 批数量无效")
        name = f"review-queue/batch-{batch:02d}.jsonl"
        payloads[name] = _payload(rows)
        parts.append({"batch": batch, "records": len(rows), "queue": name})
    manifest = {
        "pipeline": "query_sft_full_candidate_semantic_gt_review_work_package_v1",
        "release_status": "independent_semantic_gt_review_pending",
        "inputs": {
            "input_candidate": {**_identity(input_path, records=574), "hash_manifest": _identity(input_hash)},
            "teacher_candidate_results": {**_identity(result_path, records=1722), "hash_manifest": _identity(result_hash)},
            "teacher_candidate_protocol_audit": {**_identity(protocol_path), "hash_manifest": _identity(protocol_hash)},
        },
        "policy": {
            "reviewer_visible_fields": ["candidate_id", "work_id", "query_original", "source_query", "required_chunk_ids", "candidate_target"],
            "reviewer_output_fields": ["candidate_id", "review_decision", "reason"],
            "reject_reasons": ["meaning_narrowed", "gt_leakage", "ambiguity_improperly_resolved", "unsupported_fact", "unnatural_text"],
            "forbidden": ["检索结果", "检索分数", "评估题", "其他审核者结论"],
        },
        "records": {"teacher_candidates": 1722, "batches": len(parts)},
        "batches": parts,
        "readiness": {"independent_semantic_gt_review_ready": True, "frozen_retrieval_selection_ready": False, "query_sft_training_ready": False},
        "complete": True,
    }
    payloads[MANIFEST_FILENAME] = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    output_directory.mkdir(parents=True)
    for name, payload in payloads.items():
        path = output_directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8", newline="\n")
    (output_directory / HASH_FILENAME).write_text(
        "".join(f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()}  {name}\n" for name, payload in payloads.items()),
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-work-package-dir", type=Path, default=DEFAULT_INPUT_WORK_PACKAGE_DIR)
    parser.add_argument("--teacher-work-package-dir", type=Path, default=DEFAULT_TEACHER_WORK_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        manifest = prepare_query_sft_full_candidate_semantic_review(
            input_work_package_dir=args.input_work_package_dir,
            teacher_work_package_dir=args.teacher_work_package_dir,
            output_dir=args.output_dir,
        )
    except QuerySftFullCandidateReviewPreparationError as error:
        parser.error(str(error))
    print(f"QUERY_SFT_FULL_CANDIDATE_SEMANTIC_REVIEW_PACKAGE_OK candidates={manifest['records']['teacher_candidates']}")


if __name__ == "__main__":
    main()
