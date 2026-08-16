"""发布 RAG-SFT v2 全量审核的准入、排除闭合账本。

本模块只冻结已经完成的人工作业结论，不生成或改写 canonical authoring。
公共 Query 池、迁移队列和旧版草稿均为只读输入；发布目录必须尚不存在。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DATASET_ROOT = Path(__file__).resolve().parent
RAG_SFT_ROOT = DATASET_ROOT / "RAG-SFT"
QUERY_POOL = DATASET_ROOT / "QUERY-POOL" / "authoring" / "query-pool-v1.jsonl"
QUERY_POOL_MANIFEST = DATASET_ROOT / "QUERY-POOL" / "manifests" / "query-pool-v1.json"
MIGRATION_QUEUE = RAG_SFT_ROOT / "review" / "v2" / "migration-queue-r2.jsonl"
MIGRATION_MANIFEST = RAG_SFT_ROOT / "review" / "v2" / "migration-queue-r2.json"
CALIBRATION_MANIFEST = RAG_SFT_ROOT / "review" / "v2" / "calibration-v1" / "calibration-adjudication.json"
REVIEW_BATCH_MANIFEST = RAG_SFT_ROOT / "review" / "v2" / "review-batches-v1" / "manifest.json"
RUBRIC = RAG_SFT_ROOT / "review" / "v2" / "rubric-v1.md"
OUTPUT_DIR = RAG_SFT_ROOT / "review" / "v2" / "admission-v1"

EXCLUSIONS = {
    "query:0053": {
        "reason": "open_broad",
        "detail": "消费者纠纷解决路径与商品缺陷损害赔偿对象是可独立回答的两个事项。",
    },
    "query:0260": {
        "reason": "summary_overflow",
        "detail": "共同财产与个人财产均含多项法定分类，无法在一至三短句内完整保留边界。",
    },
    "query:0568": {
        "reason": "incomplete_support",
        "detail": "问题要求说明不告知条件，但 required GT 只交叉引用未纳入的条文，无法完整回答。",
    },
    "query:0572": {
        "reason": "gt_mismatch",
        "detail": "重点人群健康与长期护理条文不是心理健康回答的最小直接支持集合。",
    },
}

_POOL_FIELDS = {"query_id", "query_original", "required_chunk_ids"}
_QUEUE_FIELDS = {
    "query_id",
    "source_v1_id",
    "source_record_sha256",
    "migration_status",
    "legacy_query_reworded",
    "legacy_summary",
    "legacy_support_spans",
}


class RagSftV2AdmissionFinalizationError(RuntimeError):
    """全量审核结论无法安全冻结或发布。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if records is not None:
        value["records"] = records
    return value


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftV2AdmissionFinalizationError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise RagSftV2AdmissionFinalizationError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagSftV2AdmissionFinalizationError(
                        f"{description}不允许空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RagSftV2AdmissionFinalizationError(
                        f"{description}第 {line_number} 条必须是对象"
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftV2AdmissionFinalizationError):
            raise
        raise RagSftV2AdmissionFinalizationError(f"无法读取{description}: {path}") from error
    return rows


def _pending_ids(pool_rows: list[dict[str, Any]], queue_rows: list[dict[str, Any]]) -> list[str]:
    queue_by_id: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(queue_rows, start=1):
        if set(row) != _QUEUE_FIELDS or not isinstance(row.get("query_id"), str):
            raise RagSftV2AdmissionFinalizationError(f"迁移队列第 {position} 条字段无效")
        if row["query_id"] in queue_by_id:
            raise RagSftV2AdmissionFinalizationError("迁移队列 query_id 重复")
        queue_by_id[row["query_id"]] = row

    pending: list[str] = []
    seen: set[str] = set()
    for position, row in enumerate(pool_rows, start=1):
        if set(row) != _POOL_FIELDS or not isinstance(row.get("query_id"), str):
            raise RagSftV2AdmissionFinalizationError(f"公共 Query 池第 {position} 条字段无效")
        query_id = row["query_id"]
        if query_id in seen or query_id not in queue_by_id:
            raise RagSftV2AdmissionFinalizationError("公共 Query 池与迁移队列未闭合")
        seen.add(query_id)
        if queue_by_id[query_id]["migration_status"] == "pending_claim_authoring":
            pending.append(query_id)
    if set(queue_by_id) != seen or len(pending) != 555:
        raise RagSftV2AdmissionFinalizationError("待审核 query 集合不是冻结的 555 条")
    return pending


def _verify_inputs(
    *,
    query_pool_path: Path,
    query_pool_manifest_path: Path,
    migration_queue_path: Path,
    migration_manifest_path: Path,
    calibration_manifest_path: Path,
    review_batch_manifest_path: Path,
    rubric_path: Path,
) -> tuple[list[str], dict[str, object]]:
    pool_manifest = _load_json(query_pool_manifest_path, "公共 Query 池 manifest")
    pool_data = pool_manifest.get("data")
    if (
        pool_manifest.get("release_status") != "formal_query_pool"
        or not isinstance(pool_data, dict)
        or not isinstance(pool_data.get("query_pool"), dict)
        or pool_data["query_pool"].get("sha256") != _sha256(query_pool_path)
        or pool_data["query_pool"].get("records") != 574
    ):
        raise RagSftV2AdmissionFinalizationError("公共 Query 池身份无效或已变化")

    migration_manifest = _load_json(migration_manifest_path, "迁移队列 manifest")
    migration_output = migration_manifest.get("output")
    if (
        migration_manifest.get("pipeline") != "rag_sft_v2_migration_queue"
        or migration_manifest.get("release_status") != "review_queue"
        or not isinstance(migration_output, dict)
        or not isinstance(migration_output.get("queue"), dict)
        or migration_output["queue"].get("sha256") != _sha256(migration_queue_path)
    ):
        raise RagSftV2AdmissionFinalizationError("迁移队列身份无效或已变化")

    calibration = _load_json(calibration_manifest_path, "校准完成 manifest")
    if (
        calibration.get("pipeline") != "rag_sft_v2_calibration_finalization"
        or calibration.get("release_status") != "rubric_calibrated"
        or calibration.get("readiness", {}).get("rubric_calibrated") is not True
    ):
        raise RagSftV2AdmissionFinalizationError("审核 rubric 未冻结")

    batch_manifest = _load_json(review_batch_manifest_path, "审核批次 manifest")
    batch_rubric = batch_manifest.get("inputs", {}).get("rubric", {})
    if (
        batch_manifest.get("pipeline") != "rag_sft_v2_review_batching"
        or batch_manifest.get("records", {}).get("remaining_for_review") != 535
        or not isinstance(batch_rubric, dict)
        or not isinstance(batch_rubric.get("sha256"), str)
    ):
        raise RagSftV2AdmissionFinalizationError("全量审核批次或 rubric 身份无效")
    return (
        _pending_ids(
            _load_jsonl(query_pool_path, "公共 Query 池"),
            _load_jsonl(migration_queue_path, "迁移队列"),
        ),
        dict(batch_rubric),
    )


def _publish(output_dir: Path, payloads: list[tuple[str, str]]) -> None:
    if output_dir.exists():
        raise RagSftV2AdmissionFinalizationError(f"输出目录必须是新目录: {output_dir}")
    output_dir.mkdir(parents=True)
    prepared = [(output_dir / f"{name}.partial", output_dir / name, payload) for name, payload in payloads]
    published: list[Path] = []
    try:
        for partial, _, payload in prepared:
            partial.write_text(payload, encoding="utf-8", newline="\n")
        hash_payload = "".join(
            f"{_sha256(partial)}  {final.name}\n" for partial, final, _ in prepared
        )
        hash_partial = output_dir / "manifest.sha256.partial"
        hash_partial.write_text(hash_payload, encoding="utf-8", newline="\n")
        for partial, final, _ in prepared:
            partial.replace(final)
            published.append(final)
        hash_partial.replace(output_dir / "manifest.sha256")
    except (OSError, UnicodeError) as error:
        for path in [*(item[0] for item in prepared), output_dir / "manifest.sha256.partial", *reversed(published)]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            output_dir.rmdir()
        except OSError:
            pass
        raise RagSftV2AdmissionFinalizationError("无法发布全量审核账本") from error


def finalize_rag_sft_v2_admission(
    *,
    query_pool_path: Path,
    query_pool_manifest_path: Path,
    migration_queue_path: Path,
    migration_manifest_path: Path,
    calibration_manifest_path: Path,
    review_batch_manifest_path: Path,
    rubric_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """按冻结的全量裁决发布准入、排除账本和 manifest。"""

    paths = {
        "query_pool_path": Path(query_pool_path).resolve(),
        "query_pool_manifest_path": Path(query_pool_manifest_path).resolve(),
        "migration_queue_path": Path(migration_queue_path).resolve(),
        "migration_manifest_path": Path(migration_manifest_path).resolve(),
        "calibration_manifest_path": Path(calibration_manifest_path).resolve(),
        "review_batch_manifest_path": Path(review_batch_manifest_path).resolve(),
        "rubric_path": Path(rubric_path).resolve(),
    }
    output_dir = Path(output_dir).resolve()
    if any(not path.is_file() for path in paths.values()):
        raise RagSftV2AdmissionFinalizationError("全量审核冻结输入缺失")
    pending, rubric_at_review = _verify_inputs(**paths)
    unknown_exclusions = set(EXCLUSIONS) - set(pending)
    if unknown_exclusions:
        raise RagSftV2AdmissionFinalizationError("排除裁决不属于冻结待审核集合")

    records = []
    for query_id in pending:
        exclusion = EXCLUSIONS.get(query_id)
        if exclusion is None:
            records.append({
                "query_id": query_id,
                "decision": "admit",
                "reason": "rubric_reviewed",
                "detail": "已完成冻结 rubric 下的范围、证据与对抗边界审核。",
            })
        else:
            records.append({"query_id": query_id, "decision": "exclude", **exclusion})
    admitted = sum(row["decision"] == "admit" for row in records)
    excluded = len(records) - admitted
    if (admitted, excluded) != (551, 4):
        raise RagSftV2AdmissionFinalizationError("全量审核裁决计数未闭合为 551 准入、4 排除")

    ledger_payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for row in records
    )
    manifest = {
        "pipeline": "rag_sft_v2_full_admission_finalization",
        "release_status": "admission_finalized",
        "inputs": {
            "query_pool": _identity(paths["query_pool_path"], records=574),
            "query_pool_manifest": _identity(paths["query_pool_manifest_path"]),
            "migration_queue": _identity(paths["migration_queue_path"], records=574),
            "migration_manifest": _identity(paths["migration_manifest_path"]),
            "calibration_manifest": _identity(paths["calibration_manifest_path"]),
            "review_batch_manifest": _identity(paths["review_batch_manifest_path"]),
                "rubric_at_review": rubric_at_review,
            "current_protocol_rubric": _identity(paths["rubric_path"]),
        },
        "records": {
            "pending_claim_authoring": len(records),
            "admit": admitted,
            "exclude": excluded,
            "exclude_legacy_refusal_outside_scope": 19,
        },
        "policy": {
            "public_query_pool_unchanged": True,
            "legacy_drafts_are_not_canonical": True,
            "post_review_rubric_change": "五条 Evidence HN 占比从硬限制调整为软目标；不影响准入范围、GT 或 claim 审核。",
            "admit_reason_is_review_completion_not_claim_granularity": True,
            "canonical_authoring_emitted": False,
            "oracle_clean_emitted": False,
            "real_hard_negatives_constructed": False,
        },
        "output": {"admission_ledger": {"path": str(output_dir / "admission-ledger.jsonl"), "records": len(records)}},
        "readiness": {"admission_finalized": True, "authoring_ready": True, "training_ready": False},
        "complete": True,
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _publish(output_dir, [("admission-ledger.jsonl", ledger_payload), ("manifest.json", manifest_payload)])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结 RAG-SFT v2 全量审核准入账本")
    parser.add_argument("--query-pool", type=Path, default=QUERY_POOL)
    parser.add_argument("--query-pool-manifest", type=Path, default=QUERY_POOL_MANIFEST)
    parser.add_argument("--migration-queue", type=Path, default=MIGRATION_QUEUE)
    parser.add_argument("--migration-manifest", type=Path, default=MIGRATION_MANIFEST)
    parser.add_argument("--calibration-manifest", type=Path, default=CALIBRATION_MANIFEST)
    parser.add_argument("--review-batch-manifest", type=Path, default=REVIEW_BATCH_MANIFEST)
    parser.add_argument("--rubric", type=Path, default=RUBRIC)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    try:
        manifest = finalize_rag_sft_v2_admission(
            query_pool_path=args.query_pool,
            query_pool_manifest_path=args.query_pool_manifest,
            migration_queue_path=args.migration_queue,
            migration_manifest_path=args.migration_manifest,
            calibration_manifest_path=args.calibration_manifest,
            review_batch_manifest_path=args.review_batch_manifest,
            rubric_path=args.rubric,
            output_dir=args.output_dir,
        )
    except RagSftV2AdmissionFinalizationError as error:
        raise SystemExit(f"[失败] {error}") from error
    print(f"[完成] 准入 {manifest['records']['admit']} 条，排除 {manifest['records']['exclude']} 条")


if __name__ == "__main__":
    main()
