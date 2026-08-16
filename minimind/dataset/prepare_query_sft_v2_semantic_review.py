"""将 Query-SFT v2 教师结果和 GT 审核引用组装为独立语义审核队列。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rag.query.enhancement import parse_and_validate_query_enhancement

from .prepare_query_sft_v2 import (
    DEFAULT_OUTPUT_DIR as DEFAULT_INPUT_WORK_PACKAGE_DIR,
    HASH_FILENAME as INPUT_HASH_FILENAME,
    INPUT_FILENAME,
)
from .prepare_query_sft_v2_teacher_candidates import (
    DEFAULT_OUTPUT_DIR as DEFAULT_TEACHER_WORK_PACKAGE_DIR,
    HASH_FILENAME as TEACHER_HASH_FILENAME,
    MANIFEST_FILENAME as TEACHER_MANIFEST_FILENAME,
)
from .audit_query_sft_v2_teacher_candidates import AUDIT_FILENAME as TEACHER_AUDIT_FILENAME


DEFAULT_OUTPUT_DIR = DEFAULT_TEACHER_WORK_PACKAGE_DIR / "query-sft-v2-semantic-review-work-package"
RESULTS_FILENAME = "query-sft-v2-teacher-candidate-results.jsonl"
MANIFEST_FILENAME = "query-sft-v2-semantic-review-work-package.json"
HASH_FILENAME = "query-sft-v2-semantic-review-work-package.sha256"
INPUT_FIELDS = ("work_id", "source_id", "authoring_type", "query_original")
REFERENCE_FIELDS = ("work_id", "source_id", "authoring_type", "query_original", "required_chunk_ids")
RESULT_FIELDS = ("candidate_id", "work_id", "raw_output")
QUEUE_FIELDS = ("candidate_id", "work_id", "authoring_type", "query_original", "required_chunk_ids", "candidate_target")


class QuerySftV2SemanticReviewPreparationError(RuntimeError):
    """表示 Query-SFT v2 独立语义审核队列无法安全发布。"""


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


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftV2SemanticReviewPreparationError(f"无法读取{label}") from error
    if not isinstance(value, dict):
        raise QuerySftV2SemanticReviewPreparationError(f"{label}必须是 JSON 对象")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    raise QuerySftV2SemanticReviewPreparationError(f"{label}不允许空行: {number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise QuerySftV2SemanticReviewPreparationError(f"{label}第 {number} 条必须是对象")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftV2SemanticReviewPreparationError):
            raise
        raise QuerySftV2SemanticReviewPreparationError(f"无法读取{label}") from error
    return rows


def _verify_package(directory: Path, *, hash_filename: str, manifest_filename: str, pipeline: str) -> Path:
    hash_path = directory / hash_filename
    try:
        entries = {
            name: digest
            for line in hash_path.read_text(encoding="utf-8").splitlines()
            for digest, name in [line.split("  ", 1)]
        }
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise QuerySftV2SemanticReviewPreparationError("上游工作包 SHA-256 清单无效") from error
    if pipeline == "query_sft_v2_work_package":
        expected_names = {
            INPUT_FILENAME,
            manifest_filename,
            *(f"blind/batch-{batch:02d}.jsonl" for batch in range(1, 9)),
            *(f"audit-reference/batch-{batch:02d}.jsonl" for batch in range(1, 9)),
        }
    elif pipeline == "query_sft_v2_teacher_candidate_work_package":
        expected_names = {
            manifest_filename,
            "query-sft-v2-deterministic-noop-candidates.jsonl",
            *(f"teacher-queue/batch-{batch:02d}.jsonl" for batch in range(1, 9)),
        }
    else:
        raise QuerySftV2SemanticReviewPreparationError("未知的上游工作包协议")
    if set(entries) != expected_names:
        raise QuerySftV2SemanticReviewPreparationError("上游工作包 SHA-256 清单未精确覆盖全部受签文件")
    for name, digest in entries.items():
        path = directory / name
        if not path.is_file() or _sha256_file(path) != digest:
            raise QuerySftV2SemanticReviewPreparationError(f"上游工作包身份已变化: {name}")
    manifest = _load_json(directory / manifest_filename, "上游工作包 manifest")
    if manifest.get("pipeline") != pipeline or manifest.get("complete") is not True:
        raise QuerySftV2SemanticReviewPreparationError("上游工作包状态无效")
    return hash_path


def _payload(rows: list[dict[str, object]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)


def prepare_query_sft_v2_semantic_review(
    *,
    input_work_package_dir: Path = DEFAULT_INPUT_WORK_PACKAGE_DIR,
    teacher_work_package_dir: Path = DEFAULT_TEACHER_WORK_PACKAGE_DIR,
    teacher_results_path: Path | None = None,
    teacher_protocol_audit_path: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    """发布只向独立审核者暴露候选 target 与 GT 的 v2 审核队列。"""

    input_directory = Path(input_work_package_dir).resolve()
    teacher_directory = Path(teacher_work_package_dir).resolve()
    result_path = Path(teacher_results_path or teacher_directory / RESULTS_FILENAME).resolve()
    protocol_path = Path(teacher_protocol_audit_path or teacher_directory / TEACHER_AUDIT_FILENAME).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise QuerySftV2SemanticReviewPreparationError("v2 语义审核输出目录必须不存在")
    input_hash = _verify_package(
        input_directory,
        hash_filename=INPUT_HASH_FILENAME,
        manifest_filename="query-sft-v2-work-package.json",
        pipeline="query_sft_v2_work_package",
    )
    teacher_hash = _verify_package(
        teacher_directory,
        hash_filename=TEACHER_HASH_FILENAME,
        manifest_filename=TEACHER_MANIFEST_FILENAME,
        pipeline="query_sft_v2_teacher_candidate_work_package",
    )
    try:
        protocol_hash = protocol_path.with_suffix(".sha256")
        protocol_lines = protocol_hash.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftV2SemanticReviewPreparationError("无法读取教师候选协议审计 SHA-256") from error
    if protocol_lines != [f"{_sha256_file(protocol_path)}  {protocol_path.name}"]:
        raise QuerySftV2SemanticReviewPreparationError("教师候选协议审计 SHA-256 无效")
    protocol = _load_json(protocol_path, "教师候选协议审计")
    if (
        protocol.get("pipeline") != "query_sft_v2_teacher_candidate_protocol_audit"
        or protocol.get("records", {}).get("teacher_candidates") != 1722
        or protocol.get("validation", {}).get("all_raw_outputs_strict_protocol_valid") is not True
        or protocol.get("validation", {}).get("required_gt_not_read_or_emitted") is not True
        or protocol.get("complete") is not True
    ):
        raise QuerySftV2SemanticReviewPreparationError("教师候选协议审计状态无效")
    audited_results = protocol.get("inputs", {}).get("teacher_candidate_results")
    if (
        not isinstance(audited_results, dict)
        or audited_results.get("sha256") != _sha256_file(result_path)
        or audited_results.get("records") != 1722
    ):
        raise QuerySftV2SemanticReviewPreparationError("教师候选结果未与协议审计交叉核验")
    inputs = {}
    for row in _load_jsonl(input_directory / INPUT_FILENAME, "v2 输入"):
        work_id = row.get("work_id")
        if tuple(row) != INPUT_FIELDS or not isinstance(work_id, str) or work_id in inputs:
            raise QuerySftV2SemanticReviewPreparationError("v2 输入映射无效")
        inputs[work_id] = row
    references = {}
    for batch in range(1, 9):
        for row in _load_jsonl(input_directory / "audit-reference" / f"batch-{batch:02d}.jsonl", f"第 {batch} 批 GT 审核引用"):
            work_id = row.get("work_id")
            if (
                tuple(row) != REFERENCE_FIELDS
                or work_id not in inputs
                or work_id in references
                or any(row[key] != inputs[work_id][key] for key in INPUT_FIELDS)
            ):
                raise QuerySftV2SemanticReviewPreparationError("v2 GT 审核引用映射无效")
            references[work_id] = row
    if len(inputs) != 574 or set(inputs) != set(references):
        raise QuerySftV2SemanticReviewPreparationError("v2 输入与 GT 审核引用未闭合")
    results = _load_jsonl(result_path, "v2 教师结果")
    expected_ids = {f"{work_id}/teacher-{slot}" for work_id in inputs for slot in range(1, 4)}
    queues: dict[int, list[dict[str, object]]] = {batch: [] for batch in range(1, 9)}
    seen = set()
    for row in results:
        candidate_id = row.get("candidate_id")
        work_id = row.get("work_id")
        if (
            tuple(row) != RESULT_FIELDS
            or candidate_id not in expected_ids
            or candidate_id in seen
            or work_id not in inputs
            or not isinstance(candidate_id, str)
            or not candidate_id.startswith(f"{work_id}/teacher-")
        ):
            raise QuerySftV2SemanticReviewPreparationError("v2 教师结果映射无效")
        try:
            target = parse_and_validate_query_enhancement(row["raw_output"])
        except (TypeError, ValueError) as error:
            raise QuerySftV2SemanticReviewPreparationError(f"v2 教师候选不符合严格协议: {candidate_id}") from error
        index = int(work_id.rsplit(":", 1)[1])
        queues[(index - 1) // 72 + 1].append(
            {
                "candidate_id": candidate_id,
                "work_id": work_id,
                "authoring_type": inputs[work_id]["authoring_type"],
                "query_original": inputs[work_id]["query_original"],
                "required_chunk_ids": references[work_id]["required_chunk_ids"],
                "candidate_target": {"rewrite": target.rewrite, "expansion_terms": list(target.expansion_terms), "subqueries": list(target.subqueries)},
            }
        )
        seen.add(candidate_id)
    if seen != expected_ids:
        raise QuerySftV2SemanticReviewPreparationError("v2 教师结果必须完整覆盖 1722 个槽位")
    payloads = {}
    batches = []
    for batch, rows in queues.items():
        rows.sort(key=lambda row: row["candidate_id"])
        expected = 216 if batch < 8 else 210
        if len(rows) != expected or any(tuple(row) != QUEUE_FIELDS for row in rows):
            raise QuerySftV2SemanticReviewPreparationError(f"v2 第 {batch} 批审核队列无效")
        name = f"review-queue/batch-{batch:02d}.jsonl"
        payloads[name] = _payload(rows)
        batches.append({"batch": batch, "records": len(rows), "queue": name})
    manifest = {
        "schema_version": "1.0",
        "pipeline": "query_sft_v2_semantic_review_work_package",
        "release_status": "independent_semantic_review_pending",
        "inputs": {"v2_work_package_hash_manifest": _identity(input_hash), "teacher_work_package_hash_manifest": _identity(teacher_hash), "teacher_results": _identity(result_path, records=1722), "teacher_protocol_audit": {**_identity(protocol_path), "hash_manifest": _identity(protocol_hash)}},
        "review_protocol": {
            "reviewer_visible_fields": list(QUEUE_FIELDS),
            "reviewer_output_fields": ["candidate_id", "work_id", "review_decision", "semantic_preserved", "unsupported_fact_absent", "ambiguity_not_resolved", "natural_text", "authoring_type_fulfilled", "subquery_coverage_complete", "explicit_multi_matter_exception", "reason"],
            "approved_requires_all_quality_flags_true": True,
            "forbidden": ["检索结果", "检索分数", "评估题", "旧版选择结果", "其他审核者结论"],
        },
        "records": {"teacher_candidates": 1722, "batches": 8},
        "batches": batches,
        "readiness": {"independent_semantic_review_ready": True, "frozen_retrieval_selection_ready": False, "training_ready": False},
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
        raise QuerySftV2SemanticReviewPreparationError("无法发布 v2 语义审核工作包") from error
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-work-package-dir", type=Path, default=DEFAULT_INPUT_WORK_PACKAGE_DIR)
    parser.add_argument("--teacher-work-package-dir", type=Path, default=DEFAULT_TEACHER_WORK_PACKAGE_DIR)
    parser.add_argument("--teacher-results", type=Path)
    parser.add_argument("--teacher-protocol-audit", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        manifest = prepare_query_sft_v2_semantic_review(
            input_work_package_dir=args.input_work_package_dir,
            teacher_work_package_dir=args.teacher_work_package_dir,
            teacher_results_path=args.teacher_results,
            teacher_protocol_audit_path=args.teacher_protocol_audit,
            output_dir=args.output_dir,
        )
    except QuerySftV2SemanticReviewPreparationError as error:
        parser.error(str(error))
    print(f"QUERY_SFT_V2_SEMANTIC_REVIEW_PACKAGE_OK candidates={manifest['records']['teacher_candidates']}")


if __name__ == "__main__":
    main()
