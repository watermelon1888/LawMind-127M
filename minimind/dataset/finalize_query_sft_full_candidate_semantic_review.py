"""核验并发布全量 Query-SFT 候选的独立语义与 GT 审核账本。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .prepare_query_sft_full_candidate_semantic_review import (
        DEFAULT_OUTPUT_DIR as DEFAULT_WORK_PACKAGE_DIR,
        HASH_FILENAME as WORK_PACKAGE_HASH_FILENAME,
        MANIFEST_FILENAME as WORK_PACKAGE_MANIFEST_FILENAME,
        REVIEW_FIELDS,
        _identity,
        _load_json,
        _load_jsonl,
        _sha256_file,
    )
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset.prepare_query_sft_full_candidate_semantic_review import (
        DEFAULT_OUTPUT_DIR as DEFAULT_WORK_PACKAGE_DIR,
        HASH_FILENAME as WORK_PACKAGE_HASH_FILENAME,
        MANIFEST_FILENAME as WORK_PACKAGE_MANIFEST_FILENAME,
        REVIEW_FIELDS,
        _identity,
        _load_json,
        _load_jsonl,
        _sha256_file,
    )


RESULT_SOURCES = {
    1: "reviewer-a-r1-batch-01.jsonl",
    2: "reviewer-e-r1-batch-02.jsonl",
    3: "reviewer-f-r1-batch-03.jsonl",
    4: "reviewer-d-r1-batch-04.jsonl",
    5: "reviewer-a-r1-batch-05.jsonl",
    6: "reviewer-a-r1-batch-06.jsonl",
    7: "reviewer-d-r1-batch-07.jsonl",
    8: "reviewer-e-r1-batch-08.jsonl",
}
RESULT_FIELDS = {"candidate_id", "review_decision", "reason"}
LEDGER_FILENAME = "query-sft-v1-candidate-semantic-gt-review.jsonl"
AUDIT_FILENAME = "query-sft-v1-candidate-semantic-gt-review-audit.json"


class QuerySftFullCandidateReviewFinalizationError(RuntimeError):
    """表示全量候选的独立语义与 GT 审核结果不闭合或身份无效。"""


def _verify_work_package(directory: Path) -> Path:
    hash_path = directory / WORK_PACKAGE_HASH_FILENAME
    try:
        entries = {
            name: digest
            for digest, name in (line.split("  ", 1) for line in hash_path.read_text(encoding="utf-8").splitlines())
        }
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise QuerySftFullCandidateReviewFinalizationError("候选语义审核工作包 SHA-256 格式无效") from error
    if len(entries) != 9:
        raise QuerySftFullCandidateReviewFinalizationError("候选语义审核工作包 SHA-256 条目数无效")
    for name, digest in entries.items():
        path = directory / name
        if not path.is_file() or _sha256_file(path) != digest:
            raise QuerySftFullCandidateReviewFinalizationError(f"候选语义审核工作包身份已变化: {name}")
    manifest = _load_json(directory / WORK_PACKAGE_MANIFEST_FILENAME, "候选语义审核工作包 manifest")
    if (
        manifest.get("pipeline") != "query_sft_full_candidate_semantic_gt_review_work_package_v1"
        or manifest.get("complete") is not True
        or manifest.get("records", {}).get("teacher_candidates") != 1722
        or manifest.get("records", {}).get("batches") != 8
    ):
        raise QuerySftFullCandidateReviewFinalizationError("候选语义审核工作包状态无效")
    return hash_path


def finalize_query_sft_full_candidate_semantic_review(
    *, work_package_dir: Path = DEFAULT_WORK_PACKAGE_DIR, output_dir: Path | None = None
) -> dict[str, object]:
    """将八位独立审核者的逐候选结论汇总为冻结账本。"""

    directory = Path(work_package_dir).resolve()
    output_directory = Path(output_dir or directory).resolve()
    ledger_path = output_directory / LEDGER_FILENAME
    audit_path = output_directory / AUDIT_FILENAME
    if any(path.exists() for path in (ledger_path, ledger_path.with_suffix(".sha256"), audit_path, audit_path.with_suffix(".sha256"))):
        raise QuerySftFullCandidateReviewFinalizationError("候选语义审核发布输出已存在")
    work_hash = _verify_work_package(directory)
    expected: dict[str, dict[str, Any]] = {}
    order = []
    for batch in range(1, 9):
        rows = _load_jsonl(directory / "review-queue" / f"batch-{batch:02d}.jsonl", f"候选审核队列第 {batch} 批")
        expected_count = 216 if batch < 8 else 210
        if len(rows) != expected_count:
            raise QuerySftFullCandidateReviewFinalizationError(f"候选审核队列第 {batch} 批数量无效")
        for row in rows:
            candidate_id = row.get("candidate_id")
            if set(row) != REVIEW_FIELDS or not isinstance(candidate_id, str) or candidate_id in expected:
                raise QuerySftFullCandidateReviewFinalizationError("候选审核队列映射无效")
            expected[candidate_id] = row
            order.append(candidate_id)
    if len(expected) != 1722:
        raise QuerySftFullCandidateReviewFinalizationError("候选审核队列总数必须为 1722")

    decisions = {}
    identities = {}
    for batch, filename in RESULT_SOURCES.items():
        path = directory / "review-results" / filename
        rows = _load_jsonl(path, f"候选审核结果 {filename}")
        expected_ids = {
            row["candidate_id"]
            for row in _load_jsonl(directory / "review-queue" / f"batch-{batch:02d}.jsonl", f"候选审核队列第 {batch} 批")
        }
        if len(rows) != len(expected_ids):
            raise QuerySftFullCandidateReviewFinalizationError(f"候选审核结果 {filename} 数量不闭合")
        for row in rows:
            candidate_id = row.get("candidate_id")
            if (
                set(row) != RESULT_FIELDS
                or candidate_id not in expected_ids
                or candidate_id in decisions
                or row.get("review_decision") not in {"approved", "rejected"}
                or not isinstance(row.get("reason"), str)
                or not row["reason"].strip()
            ):
                raise QuerySftFullCandidateReviewFinalizationError(f"候选审核结果 {filename} 映射或结论无效")
            decisions[candidate_id] = row
        identities[filename] = _identity(path, records=len(rows))
    if set(decisions) != set(expected):
        raise QuerySftFullCandidateReviewFinalizationError("候选审核结果未完整覆盖 1722 个候选")

    ledger = [decisions[candidate_id] for candidate_id in order]
    counts = Counter(row["review_decision"] for row in ledger)
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in ledger)
    report = {
        "pipeline": "query_sft_full_candidate_semantic_gt_review_v1",
        "release_status": "candidate_text_audit_pending",
        "inputs": {"work_package_hash_manifest": _identity(work_hash), "review_result_fragments": identities},
        "outputs": {"semantic_gt_review_ledger": {"path": str(ledger_path), "records": len(ledger), "bytes": len(payload.encode("utf-8")), "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()}},
        "records": {"teacher_candidates": len(ledger), "approved": counts["approved"], "rejected": counts["rejected"]},
        "validation": {"work_package_identity_bound": True, "all_candidates_reviewed_once": True, "semantic_gt_review_completed_by_independent_reviewers": True, "required_gt_not_emitted": True},
        "readiness": {"candidate_semantic_gt_review_complete": True, "candidate_text_audit_ready": True, "frozen_retrieval_selection_ready": False, "query_sft_training_ready": False},
        "complete": True,
    }
    try:
        ledger_path.write_text(payload, encoding="utf-8", newline="\n")
        ledger_path.with_suffix(".sha256").write_text(f"{_sha256_file(ledger_path)}  {ledger_path.name}\n", encoding="utf-8", newline="\n")
        audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        audit_path.with_suffix(".sha256").write_text(f"{_sha256_file(audit_path)}  {audit_path.name}\n", encoding="utf-8", newline="\n")
    except (OSError, UnicodeError) as error:
        raise QuerySftFullCandidateReviewFinalizationError("无法发布候选语义审核账本") from error
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-package-dir", type=Path, default=DEFAULT_WORK_PACKAGE_DIR)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        report = finalize_query_sft_full_candidate_semantic_review(work_package_dir=args.work_package_dir, output_dir=args.output_dir)
    except QuerySftFullCandidateReviewFinalizationError as error:
        parser.error(str(error))
    print(f"QUERY_SFT_FULL_CANDIDATE_SEMANTIC_REVIEW_OK approved={report['records']['approved']} rejected={report['records']['rejected']}")


if __name__ == "__main__":
    main()
