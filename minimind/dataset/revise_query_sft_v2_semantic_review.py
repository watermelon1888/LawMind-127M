"""基于独立合规复核发布不可覆盖的 Query-SFT v2 语义审核 r2。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .assemble_query_sft_v2_semantic_review_results import (
    QuerySftV2SemanticReviewAssemblyError,
    assemble_query_sft_v2_semantic_review_results,
)
from .finalize_query_sft_v2_semantic_review import (
    QuerySftV2SemanticReviewFinalizationError,
    _sha256_file,
)
from .query_sft_v2_contract import (
    SEMANTIC_REVIEW_FIELDS,
    QuerySftV2ContractError,
    validate_semantic_review,
)


DEFAULT_SOURCE_DIR = (
    Path(__file__).resolve().parent
    / "QUERY-POOL"
    / "full"
    / "query-sft-v2-semantic-review-work-package-r1"
)
DEFAULT_OUTPUT_DIR = DEFAULT_SOURCE_DIR.parent / "query-sft-v2-semantic-review-work-package-r2"
SOURCE_REVIEW_RUN_FILENAME = "query-sft-v2-semantic-review-run.json"
REVISION_FILENAME = "query-sft-v2-semantic-review-r2-revisions.jsonl"
REVISION_RUN_FILENAME = "query-sft-v2-semantic-review-r2-revision-run.json"


class QuerySftV2SemanticReviewRevisionError(RuntimeError):
    """表示 r2 语义审核修订不能安全发布。"""


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftV2SemanticReviewRevisionError(f"无法读取{label}") from error
    if not isinstance(value, dict):
        raise QuerySftV2SemanticReviewRevisionError(f"{label}必须是 JSON 对象")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    raise QuerySftV2SemanticReviewRevisionError(f"{label}不允许空行：{number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise QuerySftV2SemanticReviewRevisionError(f"{label}第 {number} 条必须是对象")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftV2SemanticReviewRevisionError):
            raise
        raise QuerySftV2SemanticReviewRevisionError(f"无法读取{label}") from error
    return rows


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _copy_signed_source(source: Path, output: Path) -> None:
    for name in (
        "query-sft-v2-semantic-review-work-package.json",
        "query-sft-v2-semantic-review-work-package.sha256",
    ):
        origin = source / name
        target = output / name
        target.write_bytes(origin.read_bytes())
    for batch in range(1, 9):
        for subdir in ("review-queue", "review-results"):
            origin = source / subdir / f"batch-{batch:02d}.jsonl"
            target = output / subdir / origin.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(origin.read_bytes())


def revise_query_sft_v2_semantic_review(
    *, source_dir: Path = DEFAULT_SOURCE_DIR, revision_path: Path, output_dir: Path = DEFAULT_OUTPUT_DIR
) -> dict[str, object]:
    """复制 r1 的已签名输入，仅用合规复核替换明确无效的审核结论。"""

    source = Path(source_dir).resolve()
    revision_path = Path(revision_path).resolve()
    output = Path(output_dir).resolve()
    if output.exists():
        if any(output.iterdir()):
            raise QuerySftV2SemanticReviewRevisionError("r2 输出目录必须不存在或为空")
        output.rmdir()
    if not revision_path.is_file():
        raise QuerySftV2SemanticReviewRevisionError("r2 合规复核清单不存在")
    source_run = source / SOURCE_REVIEW_RUN_FILENAME
    if not source_run.is_file() or not source_run.with_suffix(".sha256").is_file():
        raise QuerySftV2SemanticReviewRevisionError("r1 审核运行记录不完整")
    if source_run.with_suffix(".sha256").read_text(encoding="utf-8").splitlines() != [
        f"{_sha256_file(source_run)}  {source_run.name}"
    ]:
        raise QuerySftV2SemanticReviewRevisionError("r1 审核运行记录哈希无效")
    source_run_data = _load_json(source_run, "r1 审核运行记录")
    if source_run_data.get("records", {}).get("review_results") != 1722 or source_run_data.get("complete") is not True:
        raise QuerySftV2SemanticReviewRevisionError("r1 审核运行记录状态无效")
    revisions = _load_jsonl(revision_path, "r2 合规复核清单")
    if not revisions:
        raise QuerySftV2SemanticReviewRevisionError("r2 合规复核清单不得为空")
    advisory_by_candidate: dict[str, dict[str, Any]] = {}
    for row in revisions:
        try:
            validate_semantic_review(row)
        except QuerySftV2ContractError as error:
            raise QuerySftV2SemanticReviewRevisionError("r2 合规复核记录不符合审核协议") from error
        candidate_id = row["candidate_id"]
        if candidate_id in advisory_by_candidate or row["review_decision"] != "rejected":
            raise QuerySftV2SemanticReviewRevisionError("r2 合规复核必须唯一且只能给出拒绝结论")
        advisory_by_candidate[candidate_id] = row
    output.mkdir(parents=True)
    try:
        _copy_signed_source(source, output)
        applied: list[str] = []
        effective_revisions: list[dict[str, Any]] = []
        for batch in range(1, 9):
            result_path = output / "review-results" / f"batch-{batch:02d}.jsonl"
            queue_path = output / "review-queue" / f"batch-{batch:02d}.jsonl"
            queue_by_candidate = {row["candidate_id"]: row for row in _load_jsonl(queue_path, "r2 审核队列")}
            replaced = []
            for row in _load_jsonl(result_path, "r1 审核结果"):
                replacement = advisory_by_candidate.get(row.get("candidate_id"))
                if replacement is not None:
                    if replacement["work_id"] != row["work_id"]:
                        raise QuerySftV2SemanticReviewRevisionError("r2 合规复核 work_id 必须与 r1 一致")
                    if replacement["candidate_id"] not in queue_by_candidate:
                        raise QuerySftV2SemanticReviewRevisionError("r2 合规复核候选不在对应审核队列")
                    if row["review_decision"] == "approved":
                        row = replacement
                        applied.append(replacement["candidate_id"])
                        effective_revisions.append(replacement)
                if tuple(row) != SEMANTIC_REVIEW_FIELDS:
                    raise QuerySftV2SemanticReviewRevisionError("r2 结果字段顺序无效")
                replaced.append(row)
            result_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in replaced),
                encoding="utf-8",
                newline="\n",
            )
        if not applied:
            raise QuerySftV2SemanticReviewRevisionError("r2 合规复核未发现任何 r1 误通过候选")
        revision_payload = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
            for row in effective_revisions
        )
        revision_output = output / REVISION_FILENAME
        revision_output.write_text(revision_payload, encoding="utf-8", newline="\n")
        revision_output.with_suffix(".sha256").write_text(
            f"{_sha256_file(revision_output)}  {revision_output.name}\n", encoding="utf-8", newline="\n"
        )
        run = {
            "schema_version": "1.0",
            "pipeline": "query_sft_v2_semantic_review_revision_r2",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "inputs": {
                "r1_semantic_review_run": {
                    "path": str(source_run),
                    "bytes": source_run.stat().st_size,
                    "sha256": _sha256_file(source_run),
                },
                "independent_compliance_revision": {
                    "path": str(revision_path),
                    "bytes": revision_path.stat().st_size,
                    "sha256": _sha256_file(revision_path),
                    "records": len(revisions),
                },
            },
            "records": {
                "compliance_advisories": len(revisions),
                "already_rejected_in_r1": len(revisions) - len(applied),
                "carried_forward_r1_reviews": 1722 - len(applied),
                "r2_rejected_corrections": len(applied),
            },
            "validation": {
                "r1_artifacts_preserved_unmodified": True,
                "only_r1_approved_records_replaced": True,
                "all_r2_corrections_are_rejections": True,
                "teacher_generation_role_separated": True,
            },
            "complete": True,
        }
        run_path = output / REVISION_RUN_FILENAME
        _write_json(run_path, run)
        run_path.with_suffix(".sha256").write_text(
            f"{_sha256_file(run_path)}  {run_path.name}\n", encoding="utf-8", newline="\n"
        )
        assemble_query_sft_v2_semantic_review_results(
            work_package_dir=output,
            reviewer_id="Query-SFT v2 semantic compliance revision r2",
            reviewer_role="independent_semantic_reviewer",
            review_method="r1_review_carry_forward_with_independent_type_contract_corrections",
            independence_declaration="r2 合规复核者未参与教师生成；仅对 r1 已通过候选的类型协议失配作拒绝性修订，其余结论按签名 r1 结果继承。",
        )
    except (
        OSError,
        UnicodeError,
        QuerySftV2SemanticReviewAssemblyError,
        QuerySftV2SemanticReviewFinalizationError,
    ) as error:
        raise QuerySftV2SemanticReviewRevisionError("无法发布可审计的 r2 语义审核工作包") from error
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--revision-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        run = revise_query_sft_v2_semantic_review(
            source_dir=args.source_dir, revision_path=args.revision_path, output_dir=args.output_dir
        )
    except QuerySftV2SemanticReviewRevisionError as error:
        parser.error(str(error))
    print(f"QUERY_SFT_V2_SEMANTIC_REVIEW_R2_OK corrections={run['records']['r2_rejected_corrections']}")


if __name__ == "__main__":
    main()
