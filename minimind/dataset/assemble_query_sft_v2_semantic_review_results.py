"""合并独立语义审核分片，并发布其可追溯运行记录。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .finalize_query_sft_v2_semantic_review import (
    REVIEW_RUN_FILENAME,
    REVIEW_RUN_IDENTITY_FIELDS,
    _identity,
    _sha256_file,
    _verify_work_package,
)
from .query_sft_v2_contract import QuerySftV2ContractError, validate_semantic_review


DEFAULT_WORK_PACKAGE_DIR = (
    Path(__file__).resolve().parent
    / "QUERY-POOL"
    / "full"
    / "query-sft-v2-semantic-review-work-package-r1"
)


class QuerySftV2SemanticReviewAssemblyError(RuntimeError):
    """表示独立语义审核结果无法组成完整、可审计的运行。"""


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    raise QuerySftV2SemanticReviewAssemblyError(f"{label}不允许空行: {number}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise QuerySftV2SemanticReviewAssemblyError(f"{label}第 {number} 条必须是对象")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftV2SemanticReviewAssemblyError):
            raise
        raise QuerySftV2SemanticReviewAssemblyError(f"无法读取{label}") from error
    return rows


def assemble_query_sft_v2_semantic_review_results(
    *,
    work_package_dir: Path = DEFAULT_WORK_PACKAGE_DIR,
    reviewer_id: str = "Codex multi-agent semantic review r1",
    reviewer_role: str = "independent_semantic_reviewer",
    review_method: str = "blind_candidate_review_with_type_and_required_gt",
    independence_declaration: str = "审核代理未参与其所审候选的教师生成，且仅消费已签名审核队列。",
) -> dict[str, object]:
    """验证八个独立审核分片，并写入不可覆盖的审核运行记录。"""

    directory = Path(work_package_dir).resolve()
    run_path = directory / REVIEW_RUN_FILENAME
    if run_path.exists() or run_path.with_suffix(".sha256").exists():
        raise QuerySftV2SemanticReviewAssemblyError("语义审核运行记录已存在，禁止覆盖")
    work_hash = _verify_work_package(directory)
    fragments = {}
    reviewed = set()
    for batch in range(1, 9):
        queue = {
            row["candidate_id"]: row
            for row in _load_jsonl(
                directory / "review-queue" / f"batch-{batch:02d}.jsonl",
                f"语义审核队列第 {batch} 批",
            )
        }
        result_path = directory / "review-results" / f"batch-{batch:02d}.jsonl"
        rows = _load_jsonl(result_path, f"语义审核结果第 {batch} 批")
        if len(rows) != len(queue):
            raise QuerySftV2SemanticReviewAssemblyError(f"语义审核结果第 {batch} 批数量不闭合")
        seen = set()
        for row in rows:
            candidate_id = row.get("candidate_id")
            if candidate_id not in queue or candidate_id in seen:
                raise QuerySftV2SemanticReviewAssemblyError("语义审核结果候选身份无效")
            try:
                validate_semantic_review(row)
            except QuerySftV2ContractError as error:
                raise QuerySftV2SemanticReviewAssemblyError("语义审核结果不符合冻结协议") from error
            if row["work_id"] != queue[candidate_id]["work_id"]:
                raise QuerySftV2SemanticReviewAssemblyError("语义审核结果 work_id 与队列不一致")
            seen.add(candidate_id)
            reviewed.add(candidate_id)
        if seen != set(queue):
            raise QuerySftV2SemanticReviewAssemblyError(f"语义审核结果第 {batch} 批覆盖不完整")
        fragments[f"batch_{batch:02d}"] = _identity(result_path, records=len(rows))
    if len(reviewed) != 1722:
        raise QuerySftV2SemanticReviewAssemblyError("语义审核结果未完整覆盖 1722 个候选")
    identity = {
        "reviewer_id": reviewer_id,
        "reviewer_role": reviewer_role,
        "review_method": review_method,
        "independence_declaration": independence_declaration,
    }
    if tuple(identity) != REVIEW_RUN_IDENTITY_FIELDS:
        raise AssertionError("语义审核者身份字段顺序意外变化")
    if not all(isinstance(value, str) and value.strip() for value in identity.values()):
        raise QuerySftV2SemanticReviewAssemblyError("语义审核者身份不得为空")
    run = {
        "schema_version": "1.0",
        "pipeline": "query_sft_v2_semantic_review_run",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": {"semantic_review_work_package_hash_manifest": _identity(work_hash)},
        "reviewer_identity": identity,
        "outputs": {"review_result_fragments": fragments},
        "records": {"review_results": len(reviewed), "batches": 8},
        "validation": {
            "all_candidates_reviewed_once": True,
            "candidate_queue_mapping_preserved": True,
            "reviewer_generation_role_separated": True,
        },
        "complete": True,
    }
    try:
        run_path.write_text(
            json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        run_path.with_suffix(".sha256").write_text(
            f"{_sha256_file(run_path)}  {run_path.name}\n",
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, UnicodeError) as error:
        run_path.unlink(missing_ok=True)
        run_path.with_suffix(".sha256").unlink(missing_ok=True)
        raise QuerySftV2SemanticReviewAssemblyError("无法发布语义审核运行记录") from error
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-package-dir", type=Path, default=DEFAULT_WORK_PACKAGE_DIR)
    parser.add_argument("--reviewer-id", default="Codex multi-agent semantic review r1")
    parser.add_argument("--reviewer-role", default="independent_semantic_reviewer")
    parser.add_argument("--review-method", default="blind_candidate_review_with_type_and_required_gt")
    parser.add_argument("--independence-declaration", default="审核代理未参与其所审候选的教师生成，且仅消费已签名审核队列。")
    args = parser.parse_args()
    try:
        run = assemble_query_sft_v2_semantic_review_results(
            work_package_dir=args.work_package_dir,
            reviewer_id=args.reviewer_id,
            reviewer_role=args.reviewer_role,
            review_method=args.review_method,
            independence_declaration=args.independence_declaration,
        )
    except QuerySftV2SemanticReviewAssemblyError as error:
        parser.error(str(error))
    print(f"QUERY_SFT_V2_SEMANTIC_REVIEW_ASSEMBLY_OK reviews={run['records']['review_results']}")


if __name__ == "__main__":
    main()
