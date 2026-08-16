"""从冻结的 RAG-SFT v2 迁移队列发布剩余 query 的只读审核批次。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_ROOT = Path(__file__).resolve().parent
RAG_SFT_ROOT = DATASET_ROOT / "RAG-SFT"
QUERY_POOL = DATASET_ROOT / "QUERY-POOL" / "authoring" / "query-pool-v1.jsonl"
MIGRATION_QUEUE = RAG_SFT_ROOT / "review" / "v2" / "migration-queue-r2.jsonl"
MIGRATION_MANIFEST = RAG_SFT_ROOT / "review" / "v2" / "migration-queue-r2.json"
CALIBRATION_MANIFEST = (
    RAG_SFT_ROOT / "review" / "v2" / "calibration-v1" / "calibration-adjudication.json"
)
ARTICLE_INDEX = Path(__file__).resolve().parents[2] / "rag" / "chunk" / "article_index.jsonl"
RUBRIC = RAG_SFT_ROOT / "review" / "v2" / "rubric-v1.md"
OUTPUT_DIR = RAG_SFT_ROOT / "review" / "v2" / "review-batches-v1"
BATCH_SIZE = 50

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
_ARTICLE_FIELDS = {
    "chunk_id", "law_name", "article_no", "article_no_sort_key", "content",
    "token_count", "char_count", "department", "effective_date", "hierarchy",
}


class RagSftV2ReviewBatchError(RuntimeError):
    """RAG-SFT v2 全量审核批次不能可靠绑定冻结输入。"""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)
    }
    if records is not None:
        result["records"] = records
    return result


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftV2ReviewBatchError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise RagSftV2ReviewBatchError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagSftV2ReviewBatchError(
                        f"{description}不允许空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RagSftV2ReviewBatchError(
                        f"{description}第 {line_number} 条必须是 object"
                    )
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftV2ReviewBatchError):
            raise
        raise RagSftV2ReviewBatchError(f"无法读取{description}: {path}") from error
    if not records:
        raise RagSftV2ReviewBatchError(f"{description}不能为空")
    return records


def _index_pool(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records, start=1):
        if set(record) != _POOL_FIELDS:
            raise RagSftV2ReviewBatchError(f"公共 Query 池第 {position} 条字段无效")
        query_id = record.get("query_id")
        query = record.get("query_original")
        required = record.get("required_chunk_ids")
        if (
            not isinstance(query_id, str) or not query_id.strip() or query_id in indexed
            or not isinstance(query, str) or not query.strip()
            or not isinstance(required, list) or not 1 <= len(required) <= 3
            or any(not isinstance(item, str) or not item.strip() for item in required)
            or len(required) != len(set(required))
        ):
            raise RagSftV2ReviewBatchError(f"公共 Query 池第 {position} 条无效")
        indexed[query_id] = record
    return indexed


def _index_queue(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records, start=1):
        if set(record) != _QUEUE_FIELDS:
            raise RagSftV2ReviewBatchError(f"迁移队列第 {position} 条字段无效")
        query_id = record.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip() or query_id in indexed:
            raise RagSftV2ReviewBatchError(f"迁移队列第 {position} 条 query_id 无效")
        indexed[query_id] = record
    return indexed


def _index_articles(records: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for position, record in enumerate(records, start=1):
        if set(record) != _ARTICLE_FIELDS:
            raise RagSftV2ReviewBatchError(f"法条索引第 {position} 条字段无效")
        chunk_id = record.get("chunk_id")
        law_name = record.get("law_name")
        article_no = record.get("article_no")
        content = record.get("content")
        if (
            not isinstance(chunk_id, str) or not chunk_id.strip() or chunk_id in indexed
            or not isinstance(law_name, str) or not law_name.strip()
            or not isinstance(article_no, str) or not article_no.strip()
            or not isinstance(content, str) or not content.strip()
        ):
            raise RagSftV2ReviewBatchError(f"法条索引第 {position} 条无效")
        indexed[chunk_id] = {
            "chunk_id": chunk_id,
            "law_name": law_name,
            "article_no": article_no,
            "content": content,
        }
    return indexed


def _verify_migration_identity(manifest: dict[str, Any], queue_path: Path) -> None:
    output = manifest.get("output")
    records = manifest.get("records")
    if (
        manifest.get("pipeline") != "rag_sft_v2_migration_queue"
        or manifest.get("release_status") != "review_queue"
        or not isinstance(output, dict)
        or not isinstance(output.get("queue"), dict)
        or not isinstance(records, dict)
    ):
        raise RagSftV2ReviewBatchError("迁移队列 manifest 身份无效")
    queue = output["queue"]
    if queue.get("sha256") != _sha256_file(queue_path) or queue.get("records") != records.get("total"):
        raise RagSftV2ReviewBatchError("迁移队列与 manifest 身份不一致")


def _calibration_query_ids(manifest: dict[str, Any]) -> set[str]:
    records = manifest.get("records")
    readiness = manifest.get("readiness")
    inputs = manifest.get("inputs")
    if (
        manifest.get("pipeline") != "rag_sft_v2_calibration_finalization"
        or manifest.get("release_status") != "rubric_calibrated"
        or not isinstance(records, dict) or records.get("total") != 20
        or not isinstance(readiness, dict) or readiness.get("rubric_calibrated") is not True
        or not isinstance(inputs, dict) or not isinstance(inputs.get("work_items"), dict)
    ):
        raise RagSftV2ReviewBatchError("校准冻结 manifest 身份无效")
    work_path = Path(inputs["work_items"].get("path", ""))
    if not work_path.is_file() or inputs["work_items"].get("sha256") != _sha256_file(work_path):
        raise RagSftV2ReviewBatchError("校准工作项身份已变化")
    work_rows = _load_jsonl(work_path, "校准工作项")
    query_ids = [row.get("query_id") for row in work_rows]
    if len(query_ids) != 20 or any(not isinstance(item, str) for item in query_ids) or len(set(query_ids)) != 20:
        raise RagSftV2ReviewBatchError("校准工作项 query_id 无效")
    return set(query_ids)


def _work_item(
    pool_record: dict[str, Any], queue_record: dict[str, Any], articles: dict[str, dict[str, str]]
) -> dict[str, object]:
    required = pool_record["required_chunk_ids"]
    required_articles: list[dict[str, str]] = []
    for chunk_id in required:
        article = articles.get(chunk_id)
        if article is None:
            raise RagSftV2ReviewBatchError(
                f"审核 query 缺少法条: {pool_record['query_id']}/{chunk_id}"
            )
        required_articles.append(article)
    return {
        "query_id": pool_record["query_id"],
        "query_original": pool_record["query_original"],
        "required_chunk_ids": required,
        "required_articles": required_articles,
        "legacy_source": {
            "source_v1_id": queue_record["source_v1_id"],
            "source_record_sha256": queue_record["source_record_sha256"],
            "legacy_query_reworded": queue_record["legacy_query_reworded"],
            "legacy_summary": queue_record["legacy_summary"],
            "legacy_support_spans": queue_record["legacy_support_spans"],
        },
    }


def _publish(output_dir: Path, payloads: list[tuple[str, str]]) -> None:
    if output_dir.exists():
        raise RagSftV2ReviewBatchError(f"批次输出目录必须是新目录: {output_dir}")
    output_dir.mkdir(parents=True)
    prepared = [(output_dir / f"{name}.partial", output_dir / name, payload) for name, payload in payloads]
    published: list[Path] = []
    try:
        for partial, _, payload in prepared:
            partial.write_text(payload, encoding="utf-8", newline="\n")
        hash_partial = output_dir / "manifest.sha256.partial"
        hash_partial.write_text(
            "".join(f"{_sha256_file(partial)}  {final.name}\n" for partial, final, _ in prepared),
            encoding="utf-8",
            newline="\n",
        )
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
        raise RagSftV2ReviewBatchError("无法发布 v2 全量审核批次") from error


def build_rag_sft_v2_review_batches(
    *,
    query_pool_path: Path,
    migration_queue_path: Path,
    migration_manifest_path: Path,
    calibration_manifest_path: Path,
    article_index_path: Path,
    rubric_path: Path,
    output_dir: Path,
    batch_size: int = BATCH_SIZE,
) -> dict[str, object]:
    """发布除已校准项外的 pending_claim_authoring 审核批次。"""

    if type(batch_size) is not int or batch_size <= 0:
        raise RagSftV2ReviewBatchError("batch_size 必须是正整数")
    query_pool_path = Path(query_pool_path).resolve()
    migration_queue_path = Path(migration_queue_path).resolve()
    migration_manifest_path = Path(migration_manifest_path).resolve()
    calibration_manifest_path = Path(calibration_manifest_path).resolve()
    article_index_path = Path(article_index_path).resolve()
    rubric_path = Path(rubric_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RagSftV2ReviewBatchError(f"批次输出目录必须是新目录: {output_dir}")
    if not rubric_path.is_file():
        raise RagSftV2ReviewBatchError(f"审核 rubric 不存在: {rubric_path}")

    pool_rows = _load_jsonl(query_pool_path, "公共 Query 池")
    queue_rows = _load_jsonl(migration_queue_path, "v2 迁移队列")
    article_rows = _load_jsonl(article_index_path, "法条索引")
    migration_manifest = _load_json(migration_manifest_path, "迁移队列 manifest")
    calibration_manifest = _load_json(calibration_manifest_path, "校准冻结 manifest")
    _verify_migration_identity(migration_manifest, migration_queue_path)
    calibration_ids = _calibration_query_ids(calibration_manifest)
    pool = _index_pool(pool_rows)
    queue = _index_queue(queue_rows)
    articles = _index_articles(article_rows)
    if set(pool) != set(queue):
        raise RagSftV2ReviewBatchError("公共 Query 池与迁移队列 query_id 不闭合")
    pending_ids = [
        query_id for query_id in pool
        if queue[query_id]["migration_status"] == "pending_claim_authoring"
    ]
    if not calibration_ids.issubset(pending_ids):
        raise RagSftV2ReviewBatchError("校准 query 不属于待审核回答队列")
    remaining_ids = [query_id for query_id in pending_ids if query_id not in calibration_ids]
    if len(pending_ids) != 555 or len(remaining_ids) != 535:
        raise RagSftV2ReviewBatchError("待审核回答与校准后的剩余数量不符合冻结队列")
    batches = [
        remaining_ids[offset : offset + batch_size]
        for offset in range(0, len(remaining_ids), batch_size)
    ]
    if [query_id for batch in batches for query_id in batch] != remaining_ids:
        raise RagSftV2ReviewBatchError("审核批次未保留公共池顺序")
    if len(set(remaining_ids)) != len(remaining_ids):
        raise RagSftV2ReviewBatchError("审核批次 query_id 重复")

    payloads: list[tuple[str, str]] = []
    batch_metadata: list[dict[str, object]] = []
    for index, batch_ids in enumerate(batches, start=1):
        rows = [_work_item(pool[query_id], queue[query_id], articles) for query_id in batch_ids]
        filename = f"review-batch-{index:02d}.jsonl"
        payload = "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
            for row in rows
        )
        payloads.append((filename, payload))
        batch_metadata.append({
            "batch": index,
            "filename": filename,
            "records": len(rows),
            "first_query_id": batch_ids[0],
            "last_query_id": batch_ids[-1],
            "legacy_query_reworded": sum(
                queue[query_id]["legacy_query_reworded"] for query_id in batch_ids
            ),
        })
    manifest = {
        "pipeline": "rag_sft_v2_review_batching",
        "release_status": "human_review_only",
        "inputs": {
            "query_pool": _identity(query_pool_path, records=len(pool_rows)),
            "migration_queue": _identity(migration_queue_path, records=len(queue_rows)),
            "migration_manifest": _identity(migration_manifest_path),
            "calibration_manifest": _identity(calibration_manifest_path),
            "article_index": _identity(article_index_path, records=len(article_rows)),
            "rubric": _identity(rubric_path),
        },
        "records": {
            "public_query_pool": len(pool_rows),
            "pending_claim_authoring": len(pending_ids),
            "exclude_legacy_refusal": sum(
                row["migration_status"] == "exclude_legacy_refusal" for row in queue_rows
            ),
            "reused_calibration": len(calibration_ids),
            "remaining_for_review": len(remaining_ids),
            "batches": len(batches),
            "remaining_legacy_query_reworded": sum(
                queue[query_id]["legacy_query_reworded"] for query_id in remaining_ids
            ),
        },
        "policy": {
            "batch_size": batch_size,
            "public_query_pool_order_preserved": True,
            "calibration_reused_without_rereview": True,
            "legacy_drafts_are_not_canonical": True,
            "training_authoring_emitted": False,
        },
        "batches": batch_metadata,
        "validation": {
            "calibration_and_batches_close_pending_queue": True,
            "remaining_query_ids_unique": True,
            "required_articles_complete": True,
        },
        "readiness": {"review_batches_ready": True, "authoring_ready": False, "training_ready": False},
        "complete": True,
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    payloads.append(("manifest.json", manifest_payload))
    _publish(output_dir, payloads)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="发布 RAG-SFT v2 全量审核批次")
    parser.add_argument("--query-pool", type=Path, default=QUERY_POOL)
    parser.add_argument("--migration-queue", type=Path, default=MIGRATION_QUEUE)
    parser.add_argument("--migration-manifest", type=Path, default=MIGRATION_MANIFEST)
    parser.add_argument("--calibration-manifest", type=Path, default=CALIBRATION_MANIFEST)
    parser.add_argument("--article-index", type=Path, default=ARTICLE_INDEX)
    parser.add_argument("--rubric", type=Path, default=RUBRIC)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    try:
        manifest = build_rag_sft_v2_review_batches(
            query_pool_path=args.query_pool,
            migration_queue_path=args.migration_queue,
            migration_manifest_path=args.migration_manifest,
            calibration_manifest_path=args.calibration_manifest,
            article_index_path=args.article_index,
            rubric_path=args.rubric,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
    except RagSftV2ReviewBatchError as error:
        raise SystemExit(f"[失败] {error}") from error
    print(f"[完成] 发布 {manifest['records']['remaining_for_review']} 条 v2 审核工作项")


if __name__ == "__main__":
    main()
