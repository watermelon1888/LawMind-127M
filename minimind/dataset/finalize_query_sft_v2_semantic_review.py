"""核验并发布 Query-SFT v2 的独立语义质量审核账本。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rag.query.enhancement import parse_and_validate_query_enhancement

from .prepare_query_sft_v2_semantic_review import (
    DEFAULT_OUTPUT_DIR as DEFAULT_WORK_PACKAGE_DIR,
    HASH_FILENAME as WORK_HASH_FILENAME,
    MANIFEST_FILENAME as WORK_MANIFEST_FILENAME,
    QUEUE_FIELDS,
)
from .query_sft_v2_contract import (
    SEMANTIC_REVIEW_FIELDS,
    QuerySftV2ContractError,
    validate_semantic_review,
    validate_target_for_authoring_type,
)


LEDGER_FILENAME = "query-sft-v2-semantic-review-ledger.jsonl"
AUDIT_FILENAME = "query-sft-v2-semantic-quality-audit.json"
REVIEW_RUN_FILENAME = "query-sft-v2-semantic-review-run.json"
REVIEW_RUN_IDENTITY_FIELDS = (
    "reviewer_id",
    "reviewer_role",
    "review_method",
    "independence_declaration",
)


class QuerySftV2SemanticReviewFinalizationError(RuntimeError):
    """表示 v2 独立语义审核结果不闭合或不满足数据质量门槛。"""


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
        raise QuerySftV2SemanticReviewFinalizationError(f"无法读取{label}") from error
    if not isinstance(value, dict):
        raise QuerySftV2SemanticReviewFinalizationError(f"{label}必须是 JSON 对象")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    raise QuerySftV2SemanticReviewFinalizationError(f"{label}不允许空行: {number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise QuerySftV2SemanticReviewFinalizationError(f"{label}第 {number} 条必须是对象")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftV2SemanticReviewFinalizationError):
            raise
        raise QuerySftV2SemanticReviewFinalizationError(f"无法读取{label}") from error
    return rows


def _verify_work_package(directory: Path) -> Path:
    hash_path = directory / WORK_HASH_FILENAME
    try:
        entries = {
            name: digest
            for line in hash_path.read_text(encoding="utf-8").splitlines()
            for digest, name in [line.split("  ", 1)]
        }
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise QuerySftV2SemanticReviewFinalizationError("v2 语义审核工作包 SHA-256 清单无效") from error
    expected_names = {
        WORK_MANIFEST_FILENAME,
        *(f"review-queue/batch-{batch:02d}.jsonl" for batch in range(1, 9)),
    }
    if set(entries) != expected_names:
        raise QuerySftV2SemanticReviewFinalizationError("v2 语义审核工作包 SHA-256 清单未精确覆盖全部受签文件")
    for name, digest in entries.items():
        path = directory / name
        if not path.is_file() or _sha256_file(path) != digest:
            raise QuerySftV2SemanticReviewFinalizationError(f"v2 语义审核工作包身份已变化: {name}")
    manifest = _load_json(directory / WORK_MANIFEST_FILENAME, "v2 语义审核工作包 manifest")
    if (
        manifest.get("pipeline") != "query_sft_v2_semantic_review_work_package"
        or manifest.get("records", {}).get("teacher_candidates") != 1722
        or manifest.get("complete") is not True
    ):
        raise QuerySftV2SemanticReviewFinalizationError("v2 语义审核工作包状态无效")
    return hash_path


def _verify_review_run(directory: Path, *, work_hash: Path) -> tuple[Path, dict[str, object]]:
    path = directory / REVIEW_RUN_FILENAME
    sidecar = path.with_suffix(".sha256")
    try:
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftV2SemanticReviewFinalizationError("无法读取语义审核运行记录 SHA-256") from error
    if lines != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftV2SemanticReviewFinalizationError("语义审核运行记录 SHA-256 无效")
    run = _load_json(path, "语义审核运行记录")
    if (
        run.get("pipeline") != "query_sft_v2_semantic_review_run"
        or run.get("complete") is not True
        or run.get("records", {}).get("review_results") != 1722
    ):
        raise QuerySftV2SemanticReviewFinalizationError("语义审核运行记录状态无效")
    identity = run.get("reviewer_identity")
    if not isinstance(identity, dict) or tuple(identity) != REVIEW_RUN_IDENTITY_FIELDS or not all(
        isinstance(identity[name], str) and identity[name].strip() for name in REVIEW_RUN_IDENTITY_FIELDS
    ):
        raise QuerySftV2SemanticReviewFinalizationError("语义审核者身份字段无效")
    if identity["reviewer_role"] == "teacher_generator":
        raise QuerySftV2SemanticReviewFinalizationError("语义审核者不得声明为教师生成器")
    work_identity = run.get("inputs", {}).get("semantic_review_work_package_hash_manifest")
    if not isinstance(work_identity, dict) or work_identity.get("sha256") != _sha256_file(work_hash):
        raise QuerySftV2SemanticReviewFinalizationError("语义审核运行记录未绑定当前审核工作包")
    fragments = run.get("outputs", {}).get("review_result_fragments")
    if not isinstance(fragments, dict) or set(fragments) != {f"batch_{batch:02d}" for batch in range(1, 9)}:
        raise QuerySftV2SemanticReviewFinalizationError("语义审核运行记录未完整声明结果分片")
    for batch in range(1, 9):
        result_path = directory / "review-results" / f"batch-{batch:02d}.jsonl"
        fragment = fragments[f"batch_{batch:02d}"]
        expected_records = 216 if batch < 8 else 210
        if (
            not isinstance(fragment, dict)
            or fragment.get("sha256") != _sha256_file(result_path)
            or fragment.get("bytes") != result_path.stat().st_size
            or fragment.get("records") != expected_records
        ):
            raise QuerySftV2SemanticReviewFinalizationError("语义审核结果分片未与运行记录交叉核验")
    return path, run


def finalize_query_sft_v2_semantic_review(
    *, work_package_dir: Path = DEFAULT_WORK_PACKAGE_DIR, output_dir: Path | None = None
) -> dict[str, object]:
    """发布逐候选语义质量账本，并在类型职责未满足时阻断。"""

    directory = Path(work_package_dir).resolve()
    output_directory = Path(output_dir or directory).resolve()
    ledger_path = output_directory / LEDGER_FILENAME
    audit_path = output_directory / AUDIT_FILENAME
    if any(path.exists() for path in (ledger_path, ledger_path.with_suffix(".sha256"), audit_path, audit_path.with_suffix(".sha256"))):
        raise QuerySftV2SemanticReviewFinalizationError("v2 语义审核输出已存在，禁止覆盖")
    if output_directory.exists() and not output_directory.is_dir():
        raise QuerySftV2SemanticReviewFinalizationError("v2 语义审核输出路径必须是目录")
    work_hash = _verify_work_package(directory)
    review_run_path, review_run = _verify_review_run(directory, work_hash=work_hash)
    queued = {}
    order = []
    for batch in range(1, 9):
        rows = _load_jsonl(directory / "review-queue" / f"batch-{batch:02d}.jsonl", f"v2 审核队列第 {batch} 批")
        expected = 216 if batch < 8 else 210
        if len(rows) != expected:
            raise QuerySftV2SemanticReviewFinalizationError(f"v2 审核队列第 {batch} 批数量无效")
        for row in rows:
            candidate_id = row.get("candidate_id")
            if tuple(row) != QUEUE_FIELDS or not isinstance(candidate_id, str) or candidate_id in queued:
                raise QuerySftV2SemanticReviewFinalizationError("v2 审核队列字段或候选身份无效")
            queued[candidate_id] = row
            order.append(candidate_id)
    if len(queued) != 1722:
        raise QuerySftV2SemanticReviewFinalizationError("v2 审核队列必须覆盖 1722 条候选")

    decisions = {}
    result_identities = {}
    for batch in range(1, 9):
        path = directory / "review-results" / f"batch-{batch:02d}.jsonl"
        rows = _load_jsonl(path, f"v2 审核结果第 {batch} 批")
        expected_ids = {
            row["candidate_id"]
            for row in _load_jsonl(directory / "review-queue" / f"batch-{batch:02d}.jsonl", f"v2 审核队列第 {batch} 批")
        }
        if len(rows) != len(expected_ids):
            raise QuerySftV2SemanticReviewFinalizationError(f"v2 审核结果第 {batch} 批数量不闭合")
        for row in rows:
            candidate_id = row.get("candidate_id")
            if candidate_id not in expected_ids or candidate_id in decisions:
                raise QuerySftV2SemanticReviewFinalizationError("v2 审核结果候选身份无效")
            try:
                validate_semantic_review(row)
            except QuerySftV2ContractError as error:
                raise QuerySftV2SemanticReviewFinalizationError(f"v2 审核结果无效: {candidate_id}") from error
            queue = queued[candidate_id]
            if row["work_id"] != queue["work_id"]:
                raise QuerySftV2SemanticReviewFinalizationError("v2 审核结果 work_id 未与队列一致")
            if row["explicit_multi_matter_exception"] and queue["authoring_type"] != "explicit_multi_matter":
                raise QuerySftV2SemanticReviewFinalizationError("仅明确多事项记录可声明 subqueries 例外")
            if row["explicit_multi_matter_exception"] and "例外" not in row["reason"]:
                raise QuerySftV2SemanticReviewFinalizationError("明确多事项例外必须在审核理由中说明")
            try:
                validate_target_for_authoring_type(
                    query_original=queue["query_original"],
                    authoring_type=queue["authoring_type"],
                    target=queue["candidate_target"],
                    allow_explicit_multi_matter_exception=(
                        queue["authoring_type"] == "explicit_multi_matter"
                        and row["explicit_multi_matter_exception"]
                    ),
                )
                target_type_valid = True
            except QuerySftV2ContractError:
                target_type_valid = False
            if row["review_decision"] == "approved" and not target_type_valid:
                raise QuerySftV2SemanticReviewFinalizationError("通过的 v2 候选未满足 authoring_type 最低职责")
            decisions[candidate_id] = row
        result_identities[f"batch_{batch:02d}"] = _identity(path, records=len(rows))
    if set(decisions) != set(queued):
        raise QuerySftV2SemanticReviewFinalizationError("v2 审核结果未完整覆盖候选队列")

    ledger = [decisions[candidate_id] for candidate_id in order]
    approved = [row for row in ledger if row["review_decision"] == "approved"]
    counts = Counter(row["review_decision"] for row in ledger)
    approved_by_type = Counter(queued[row["candidate_id"]]["authoring_type"] for row in approved)
    exceptions = [
        row["candidate_id"]
        for row in approved
        if row["explicit_multi_matter_exception"]
    ]
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in ledger)
    report = {
        "schema_version": "1.0",
        "pipeline": "query_sft_v2_semantic_quality_audit",
        "release_status": "frozen_retrieval_selection_pending",
        "inputs": {"semantic_review_work_package_hash_manifest": _identity(work_hash), "semantic_review_run": {**_identity(review_run_path), "reviewer_identity": review_run["reviewer_identity"]}, "review_result_fragments": result_identities},
        "outputs": {"semantic_review_ledger": {"path": str(ledger_path), "bytes": len(payload.encode("utf-8")), "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(), "records": 1722}},
        "records": {"teacher_candidates": 1722, "approved": counts["approved"], "rejected": counts["rejected"], "approved_by_authoring_type": dict(sorted(approved_by_type.items())), "explicit_multi_matter_exceptions": {"count": len(exceptions), "candidate_ids": exceptions}},
        "validation": {"all_candidates_reviewed_once": True, "independent_semantic_review_complete": True, "approved_candidates_satisfy_type_contract": True, "required_gt_not_emitted": True},
        "readiness": {"semantic_quality_ready": True, "frozen_retrieval_selection_ready": False, "training_ready": False},
        "complete": True,
    }
    try:
        output_directory.mkdir(parents=True, exist_ok=False)
        ledger_path.write_text(payload, encoding="utf-8", newline="\n")
        ledger_path.with_suffix(".sha256").write_text(f"{_sha256_file(ledger_path)}  {ledger_path.name}\n", encoding="utf-8", newline="\n")
        audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        audit_path.with_suffix(".sha256").write_text(f"{_sha256_file(audit_path)}  {audit_path.name}\n", encoding="utf-8", newline="\n")
    except (OSError, UnicodeError) as error:
        raise QuerySftV2SemanticReviewFinalizationError("无法发布 v2 语义质量审核") from error
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-package-dir", type=Path, default=DEFAULT_WORK_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        report = finalize_query_sft_v2_semantic_review(
            work_package_dir=args.work_package_dir, output_dir=args.output_dir
        )
    except QuerySftV2SemanticReviewFinalizationError as error:
        parser.error(str(error))
    print(f"QUERY_SFT_V2_SEMANTIC_QUALITY_AUDIT_OK approved={report['records']['approved']}")


if __name__ == "__main__":
    main()
