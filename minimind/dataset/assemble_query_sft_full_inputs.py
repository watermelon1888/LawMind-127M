"""合并并审计全量 Query-SFT 盲构造输入及其独立审核结果。"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .build_sft_evaluation_exclusions import question_digest
    from .prepare_query_sft_full import DEFAULT_OUTPUT_DIR
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset.build_sft_evaluation_exclusions import question_digest
    from dataset.prepare_query_sft_full import DEFAULT_OUTPUT_DIR


DATASET_ROOT = Path(__file__).resolve().parent
QUERY_POOL_ROOT = DATASET_ROOT / "QUERY-POOL"
DEFAULT_EVALUATION_EXCLUSIONS = (
    DATASET_ROOT / "RAG-SFT" / "manifests" / "evaluation-exclusions-project-rag-v2.json"
)
DEFAULT_OUTPUT_PATH = DEFAULT_OUTPUT_DIR / "query-sft-v1-input-candidate.jsonl"
DEFAULT_REPORT_PATH = DEFAULT_OUTPUT_DIR / "query-sft-v1-input-structural-audit.json"

WORK_PACKAGE_MANIFEST = "query-sft-v1-work-package.json"
WORK_PACKAGE_HASH = "query-sft-v1-work-package.sha256"
DRAFT_FIELDS = {"work_id", "source_id", "source_query", "authoring_type", "query_original"}
REVIEW_FIELDS = {"work_id", "source_id", "query_original", "review_decision", "reason_code"}
CORRECTION_FIELDS = {
    "work_id",
    "source_id",
    "source_query",
    "authoring_type",
    "query_original",
    "supersedes_reason",
}
AUTHORING_TYPES = {
    "no_op",
    "colloquial",
    "ellipsis",
    "ambiguous_multi_intent",
    "explicit_multi_matter",
}
REVIEW_REASONS = {
    "approved",
    "semantic_change",
    "gt_incompatible",
    "unrecoverable_ambiguity",
    "unnatural_input",
}
AUTHOR_BY_BATCH = {**{number: "a" for number in range(1, 4)}, **{number: "b" for number in range(4, 7)}, **{number: "c" for number in range(7, 9)}}
REVIEWER_BY_BATCH = {**{number: "b" for number in (1, 2, 3, 7, 8)}, **{number: "c" for number in (4, 5, 6)}}


class QuerySftFullInputAssemblyError(RuntimeError):
    """表示全量 Query-SFT 输入未满足可进入教师候选生成的条件。"""


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
        raise QuerySftFullInputAssemblyError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise QuerySftFullInputAssemblyError(f"{description}必须是 JSON 对象")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    raise QuerySftFullInputAssemblyError(f"{description}不允许空行: {line_number}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise QuerySftFullInputAssemblyError(f"{description}第 {line_number} 条必须是对象")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftFullInputAssemblyError):
            raise
        raise QuerySftFullInputAssemblyError(f"无法读取{description}: {path}") from error
    return rows


def _verify_hash_sidecar(path: Path, description: str) -> Path:
    hash_path = path.with_suffix(".sha256")
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftFullInputAssemblyError(f"无法读取{description} SHA-256") from error
    if lines != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftFullInputAssemblyError(f"{description} SHA-256 无效")
    return hash_path


def _verify_work_package(directory: Path) -> dict[str, Any]:
    manifest_path = directory / WORK_PACKAGE_MANIFEST
    hash_path = directory / WORK_PACKAGE_HASH
    try:
        entries = {
            relative: digest
            for digest, relative in (
                line.split("  ", 1) for line in hash_path.read_text(encoding="utf-8").splitlines()
            )
        }
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise QuerySftFullInputAssemblyError("全量工作包 SHA-256 格式无效") from error
    if len(entries) != 18:
        raise QuerySftFullInputAssemblyError("全量工作包 SHA-256 条目数无效")
    for relative, digest in entries.items():
        path = directory / relative
        if not path.is_file() or _sha256_file(path) != digest:
            raise QuerySftFullInputAssemblyError(f"全量工作包身份已变化: {relative}")
    manifest = _load_json(manifest_path, "全量工作包 manifest")
    if (
        manifest.get("pipeline") != "query_sft_full_work_package_v1"
        or manifest.get("release_status") != "full_input_construction_pending"
        or manifest.get("batches", {}).get("records") != 574
        or manifest.get("batches", {}).get("count") != 8
        or manifest.get("readiness", {}).get("full_input_construction_ready") is not True
        or manifest.get("complete") is not True
    ):
        raise QuerySftFullInputAssemblyError("全量工作包状态无效")
    return {"manifest": manifest_path, "hash": hash_path, "value": manifest}


def _normalize_query(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuerySftFullInputAssemblyError(f"{description}必须为非空字符串")
    normalized = unicodedata.normalize("NFC", value.strip())
    if normalized != value or "\n" in value or "\r" in value or len(normalized) > 96:
        raise QuerySftFullInputAssemblyError(f"{description}必须为不超过 96 字符的 NFC 单行文本")
    return normalized


def _load_evaluation_digests(path: Path) -> tuple[set[str], Path]:
    hash_path = _verify_hash_sidecar(path, "评估隔离清单")
    manifest = _load_json(path, "评估隔离清单")
    digests = manifest.get("question_sha256")
    if (
        manifest.get("complete_for_formal_sft") is not True
        or not isinstance(digests, list)
        or not digests
        or len(digests) != len(set(digests))
        or any(not isinstance(item, str) or len(item) != 64 for item in digests)
    ):
        raise QuerySftFullInputAssemblyError("评估隔离清单状态无效")
    return set(digests), hash_path


def _rows_by_work_id(rows: list[dict[str, Any]], *, fields: set[str], description: str) -> dict[str, dict[str, Any]]:
    by_id = {}
    for position, row in enumerate(rows, 1):
        if set(row) != fields:
            raise QuerySftFullInputAssemblyError(f"{description}第 {position} 条字段无效")
        work_id = row.get("work_id")
        if not isinstance(work_id, str) or work_id in by_id:
            raise QuerySftFullInputAssemblyError(f"{description} work_id 无效或重复")
        by_id[work_id] = row
    return by_id


def assemble_query_sft_full_inputs(
    *,
    work_package_dir: Path = DEFAULT_OUTPUT_DIR,
    evaluation_exclusions_path: Path = DEFAULT_EVALUATION_EXCLUSIONS,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, object]:
    """只发布已独立审核的全量 Query-SFT 教师候选输入，不生成教师 target。"""

    work_package_dir = Path(work_package_dir).resolve()
    evaluation_exclusions_path = Path(evaluation_exclusions_path).resolve()
    output_path = Path(output_path).resolve()
    report_path = Path(report_path).resolve()
    if any(path.exists() for path in (output_path, output_path.with_suffix(".sha256"), report_path, report_path.with_suffix(".sha256"))):
        raise QuerySftFullInputAssemblyError("全量输入候选输出已存在")
    package = _verify_work_package(work_package_dir)
    evaluation_digests, evaluation_hash_path = _load_evaluation_digests(evaluation_exclusions_path)
    selected = []
    identities: dict[str, object] = {}
    type_counts: Counter[str] = Counter()
    replacement_count = 0
    seen_queries: set[str] = set()

    for batch in range(1, 9):
        suffix = f"batch-{batch:02d}.jsonl"
        blind_path = work_package_dir / "blind" / suffix
        draft_path = work_package_dir / "input-drafts" / f"author-{AUTHOR_BY_BATCH[batch]}-{suffix}"
        review_path = work_package_dir / "input-reviews" / f"reviewer-{REVIEWER_BY_BATCH[batch]}-{suffix}"
        blind_rows = _load_jsonl(blind_path, f"第 {batch} 批盲构造输入")
        draft_by_id = _rows_by_work_id(_load_jsonl(draft_path, f"第 {batch} 批草稿"), fields=DRAFT_FIELDS, description=f"第 {batch} 批草稿")
        review_by_id = _rows_by_work_id(_load_jsonl(review_path, f"第 {batch} 批审核账本"), fields=REVIEW_FIELDS, description=f"第 {batch} 批审核账本")
        blind_by_id = _rows_by_work_id(blind_rows, fields={"work_id", "source_id", "source_query"}, description=f"第 {batch} 批盲构造输入")
        if set(blind_by_id) != set(draft_by_id) or set(blind_by_id) != set(review_by_id):
            raise QuerySftFullInputAssemblyError(f"第 {batch} 批输入、草稿和审核账本未闭合")
        corrections = {}
        correction_review = {}
        rejected_work_ids = set()
        if batch == 4 and any(
            item.get("review_decision") == "rejected" for item in review_by_id.values()
        ):
            correction_path = work_package_dir / "input-corrections" / "reviser-a-batch-04.jsonl"
            correction_review_path = work_package_dir / "input-reviews" / "reviewer-c-batch-04-corrections.jsonl"
            corrections = _rows_by_work_id(_load_jsonl(correction_path, "第 04 批修订"), fields=CORRECTION_FIELDS, description="第 04 批修订")
            correction_review = _rows_by_work_id(_load_jsonl(correction_review_path, "第 04 批修订审核"), fields=REVIEW_FIELDS, description="第 04 批修订审核")
            if set(corrections) != set(correction_review):
                raise QuerySftFullInputAssemblyError("第 04 批修订与修订审核未闭合")
            identities["corrections"] = _identity(correction_path, records=len(corrections))
            identities["correction_review"] = _identity(correction_review_path, records=len(correction_review))
        for work_id, blind in blind_by_id.items():
            draft = draft_by_id[work_id]
            review = review_by_id[work_id]
            if (
                draft.get("source_id") != blind["source_id"]
                or draft.get("source_query") != blind["source_query"]
                or draft.get("authoring_type") not in AUTHORING_TYPES
                or review.get("source_id") != blind["source_id"]
                or review.get("query_original") != draft.get("query_original")
                or review.get("review_decision") not in {"approved", "rejected"}
                or review.get("reason_code") not in REVIEW_REASONS
                or (review["review_decision"] == "approved") != (review["reason_code"] == "approved")
            ):
                raise QuerySftFullInputAssemblyError(f"第 {batch} 批 {work_id} 草稿或审核映射无效")
            query = _normalize_query(draft.get("query_original"), f"第 {batch} 批 {work_id} query_original")
            if (draft["authoring_type"] == "no_op") != (query == blind["source_query"]):
                raise QuerySftFullInputAssemblyError(f"第 {batch} 批 {work_id} no_op 语义无效")
            final = draft
            if review["review_decision"] == "rejected":
                rejected_work_ids.add(work_id)
                correction = corrections.get(work_id)
                correction_result = correction_review.get(work_id)
                if (
                    correction is None
                    or correction_result is None
                    or correction.get("source_id") != blind["source_id"]
                    or correction.get("source_query") != blind["source_query"]
                    or correction.get("authoring_type") != "no_op"
                    or correction.get("query_original") != blind["source_query"]
                    or correction.get("supersedes_reason") != "semantic_change_fallback_to_no_op"
                    or correction_result.get("source_id") != blind["source_id"]
                    or correction_result.get("query_original") != blind["source_query"]
                    or correction_result.get("review_decision") != "approved"
                    or correction_result.get("reason_code") != "approved"
                ):
                    raise QuerySftFullInputAssemblyError(f"第 {batch} 批 {work_id} 被拒绝后未完成独立 no_op 修订")
                final = correction
                replacement_count += 1
            final_query = _normalize_query(final["query_original"], f"第 {batch} 批 {work_id} 最终 query_original")
            digest = question_digest(final_query)
            if digest in seen_queries:
                raise QuerySftFullInputAssemblyError("最终 query_original 去重失败")
            if digest in evaluation_digests:
                raise QuerySftFullInputAssemblyError("最终 query_original 命中评估隔离清单")
            seen_queries.add(digest)
            type_counts[final["authoring_type"]] += 1
            selected.append({"work_id": work_id, "source_id": blind["source_id"], "authoring_type": final["authoring_type"], "query_original": final_query})
        if corrections and set(corrections) != rejected_work_ids:
            raise QuerySftFullInputAssemblyError("修订记录必须与被拒绝草稿一一对应")
        identities[f"blind_batch_{batch:02d}"] = _identity(blind_path, records=len(blind_rows))
        identities[f"draft_batch_{batch:02d}"] = _identity(draft_path, records=len(draft_by_id))
        identities[f"review_batch_{batch:02d}"] = _identity(review_path, records=len(review_by_id))
    if len(selected) != 574:
        raise QuerySftFullInputAssemblyError("全量最终输入数量必须为 574")
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selected)
    report = {
        "pipeline": "query_sft_full_input_structural_audit_v1",
        "release_status": "full_teacher_candidate_input",
        "inputs": {
            "work_package_manifest": _identity(package["manifest"]),
            "work_package_hash_manifest": _identity(package["hash"]),
            "evaluation_exclusions": {**_identity(evaluation_exclusions_path), "hash_manifest": _identity(evaluation_hash_path)},
            **identities,
        },
        "records": {"work_package": 574, "final_input": 574, "replaced_with_independently_reviewed_no_op": replacement_count, "by_authoring_type": dict(sorted(type_counts.items()))},
        "outputs": {"input_candidate": {"path": str(output_path), "records": 574, "bytes": len(payload.encode("utf-8")), "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}},
        "validation": {"work_package_identity_bound": True, "draft_and_review_coverage_closed": True, "rejected_drafts_not_emitted": True, "corrections_independently_reviewed": True, "queries_unique_and_within_limit": True, "evaluation_question_overlap": 0, "required_gt_not_emitted": True},
        "readiness": {"full_input_semantic_gt_review_complete": True, "teacher_candidate_generation_ready": True, "frozen_retrieval_selection_complete": False, "query_sft_training_ready": False},
        "complete": True,
    }
    report_payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    try:
        output_path.write_text(payload, encoding="utf-8", newline="\n")
        output_path.with_suffix(".sha256").write_text(f"{_sha256_file(output_path)}  {output_path.name}\n", encoding="utf-8", newline="\n")
        report_path.write_text(report_payload, encoding="utf-8", newline="\n")
        report_path.with_suffix(".sha256").write_text(f"{_sha256_file(report_path)}  {report_path.name}\n", encoding="utf-8", newline="\n")
    except (OSError, UnicodeError) as error:
        for path in (output_path, output_path.with_suffix(".sha256"), report_path, report_path.with_suffix(".sha256")):
            path.unlink(missing_ok=True)
        raise QuerySftFullInputAssemblyError("无法发布全量 Query-SFT 教师候选输入") from error
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-package-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--evaluation-exclusions", type=Path, default=DEFAULT_EVALUATION_EXCLUSIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    args = parser.parse_args()
    try:
        report = assemble_query_sft_full_inputs(work_package_dir=args.work_package_dir, evaluation_exclusions_path=args.evaluation_exclusions, output_path=args.output, report_path=args.report)
    except QuerySftFullInputAssemblyError as error:
        parser.error(str(error))
    print(f"QUERY_SFT_FULL_INPUT_AUDIT_OK records={report['records']['final_input']}")


if __name__ == "__main__":
    main()
