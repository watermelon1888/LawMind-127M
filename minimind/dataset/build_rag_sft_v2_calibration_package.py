"""从冻结的 RAG-SFT v2 审核队列发布不可覆盖的校准工作包。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DATASET_ROOT = Path(__file__).resolve().parent
RAG_SFT_ROOT = DATASET_ROOT / "RAG-SFT"
QUERY_POOL = DATASET_ROOT / "QUERY-POOL" / "authoring" / "query-pool-v1.jsonl"
MIGRATION_QUEUE = RAG_SFT_ROOT / "review" / "v2" / "migration-queue-r2.jsonl"
ARTICLE_INDEX = Path(__file__).resolve().parents[2] / "rag" / "chunk" / "article_index.jsonl"
RUBRIC = RAG_SFT_ROOT / "review" / "v2" / "rubric-v1.md"
OUTPUT_DIR = RAG_SFT_ROOT / "review" / "v2" / "calibration-v1"

CALIBRATION_QUERY_IDS = (
    "query:0001", "query:0002", "query:0004", "query:0013", "query:0037",
    "query:0060", "query:0090", "query:0107", "query:0154", "query:0213",
    "query:0267", "query:0287", "query:0294", "query:0313", "query:0335",
    "query:0378", "query:0391", "query:0401", "query:0449", "query:0554",
)

_POOL_FIELDS = {"query_id", "query_original", "required_chunk_ids"}
_QUEUE_FIELDS = {
    "query_id", "source_v1_id", "source_record_sha256", "migration_status",
    "legacy_query_reworded", "legacy_summary", "legacy_support_spans",
}
_ARTICLE_FIELDS = {
    "chunk_id", "law_name", "article_no", "article_no_sort_key", "content",
    "token_count", "char_count", "department", "effective_date", "hierarchy",
}


class RagSftV2CalibrationPackageError(RuntimeError):
    """校准工作包不能可靠绑定冻结输入。"""


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


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagSftV2CalibrationPackageError(
                        f"{description}不允许空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RagSftV2CalibrationPackageError(
                        f"{description}第 {line_number} 条必须是对象"
                    )
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftV2CalibrationPackageError):
            raise
        raise RagSftV2CalibrationPackageError(f"无法读取{description}: {path}") from error
    if not records:
        raise RagSftV2CalibrationPackageError(f"{description}不能为空")
    return records


def _index_pool(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records, start=1):
        if set(record) != _POOL_FIELDS:
            raise RagSftV2CalibrationPackageError(f"公共 Query 池第 {position} 条字段无效")
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
            raise RagSftV2CalibrationPackageError(f"公共 Query 池第 {position} 条无效")
        indexed[query_id] = record
    return indexed


def _index_queue(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records, start=1):
        if set(record) != _QUEUE_FIELDS:
            raise RagSftV2CalibrationPackageError(f"迁移队列第 {position} 条字段无效")
        query_id = record.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip() or query_id in indexed:
            raise RagSftV2CalibrationPackageError(f"迁移队列第 {position} 条 query_id 无效")
        indexed[query_id] = record
    return indexed


def _index_articles(records: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for position, record in enumerate(records, start=1):
        if set(record) != _ARTICLE_FIELDS:
            raise RagSftV2CalibrationPackageError(f"法条索引第 {position} 条字段无效")
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
            raise RagSftV2CalibrationPackageError(f"法条索引第 {position} 条无效")
        indexed[chunk_id] = {
            "chunk_id": chunk_id, "law_name": law_name,
            "article_no": article_no, "content": content,
        }
    return indexed


def _require_new_outputs(work_items: Path, manifest: Path, hash_path: Path) -> None:
    occupied = [
        str(candidate)
        for target in (work_items, manifest, hash_path)
        for candidate in (target, target.with_name(target.name + ".partial"))
        if candidate.exists()
    ]
    if occupied:
        raise RagSftV2CalibrationPackageError("目标输出已存在: " + ", ".join(occupied))


def _publish(outputs: list[tuple[Path, str]]) -> None:
    pending = [(path.with_name(path.name + ".partial"), path, payload) for path, payload in outputs]
    published: list[Path] = []
    try:
        for partial, _, payload in pending:
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(payload, encoding="utf-8", newline="\n")
        for partial, final, _ in pending:
            partial.replace(final)
            published.append(final)
    except (OSError, UnicodeError) as error:
        for path in [*(item[0] for item in pending), *reversed(published)]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise RagSftV2CalibrationPackageError("无法发布校准工作包") from error


def build_rag_sft_v2_calibration_package(
    *,
    query_pool_path: Path,
    migration_queue_path: Path,
    article_index_path: Path,
    rubric_path: Path,
    work_items_output: Path,
    manifest_output: Path,
    query_ids: tuple[str, ...] = CALIBRATION_QUERY_IDS,
) -> dict[str, object]:
    """发布仅供人工审核的校准工作项，不派生 authoring 或训练数据。"""

    query_pool_path = Path(query_pool_path).resolve()
    migration_queue_path = Path(migration_queue_path).resolve()
    article_index_path = Path(article_index_path).resolve()
    rubric_path = Path(rubric_path).resolve()
    work_items_output = Path(work_items_output).resolve()
    manifest_output = Path(manifest_output).resolve()
    hash_output = manifest_output.with_suffix(".sha256")
    if len(query_ids) != 20 or len(query_ids) != len(set(query_ids)):
        raise RagSftV2CalibrationPackageError("校准样本必须恰好包含 20 个不重复 query_id")
    _require_new_outputs(work_items_output, manifest_output, hash_output)
    if not rubric_path.is_file():
        raise RagSftV2CalibrationPackageError(f"审核 rubric 不存在: {rubric_path}")

    pool_rows = _load_jsonl(query_pool_path, "公共 Query 池")
    queue_rows = _load_jsonl(migration_queue_path, "v2 迁移队列")
    article_rows = _load_jsonl(article_index_path, "法条索引")
    pool = _index_pool(pool_rows)
    queue = _index_queue(queue_rows)
    articles = _index_articles(article_rows)
    if set(pool) != set(queue):
        raise RagSftV2CalibrationPackageError("公共 Query 池与迁移队列 query_id 不闭合")

    work_items: list[dict[str, object]] = []
    for query_id in query_ids:
        pool_record = pool.get(query_id)
        queue_record = queue.get(query_id)
        if pool_record is None or queue_record is None:
            raise RagSftV2CalibrationPackageError(f"校准 query_id 不存在: {query_id}")
        if queue_record["migration_status"] != "pending_claim_authoring":
            raise RagSftV2CalibrationPackageError(f"校准 query_id 不是待审核回答: {query_id}")
        required = pool_record["required_chunk_ids"]
        required_articles = []
        for chunk_id in required:
            article = articles.get(chunk_id)
            if article is None:
                raise RagSftV2CalibrationPackageError(f"校准 query 缺少法条: {query_id}/{chunk_id}")
            required_articles.append(article)
        work_items.append(
            {
                "query_id": query_id,
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
        )

    work_payload = "".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for item in work_items
    )
    manifest = {
        "pipeline": "rag_sft_v2_calibration_work_package",
        "release_status": "human_review_only",
        "inputs": {
            "query_pool": _identity(query_pool_path, records=len(pool_rows)),
            "migration_queue": _identity(migration_queue_path, records=len(queue_rows)),
            "article_index": _identity(article_index_path, records=len(article_rows)),
            "rubric": _identity(rubric_path),
        },
        "selection": {
            "query_ids": list(query_ids),
            "records": len(work_items),
            "policy": "覆盖一至三条 GT、改写、跨法及高风险语义形态；不构成准入裁决",
        },
        "output": {
            "work_items": {
                "path": str(work_items_output), "bytes": len(work_payload.encode("utf-8")),
                "sha256": _sha256_bytes(work_payload.encode("utf-8")), "records": len(work_items),
            }
        },
        "readiness": {"calibration_adjudicated": False, "authoring_ready": False, "training_ready": False},
        "complete": True,
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    hash_payload = f"{_sha256_bytes(manifest_payload.encode('utf-8'))}  {manifest_output.name}\n"
    _publish([
        (work_items_output, work_payload), (manifest_output, manifest_payload), (hash_output, hash_payload)
    ])
    return manifest


def main() -> None:
    """解析参数并发布 v2 校准工作包。"""

    parser = argparse.ArgumentParser(description="发布 RAG-SFT v2 校准审核工作包")
    parser.add_argument("--query-pool", type=Path, default=QUERY_POOL)
    parser.add_argument("--migration-queue", type=Path, default=MIGRATION_QUEUE)
    parser.add_argument("--article-index", type=Path, default=ARTICLE_INDEX)
    parser.add_argument("--rubric", type=Path, default=RUBRIC)
    parser.add_argument("--work-items-output", type=Path, default=OUTPUT_DIR / "work-items.jsonl")
    parser.add_argument("--manifest-output", type=Path, default=OUTPUT_DIR / "manifest.json")
    args = parser.parse_args()
    try:
        manifest = build_rag_sft_v2_calibration_package(
            query_pool_path=args.query_pool, migration_queue_path=args.migration_queue,
            article_index_path=args.article_index, rubric_path=args.rubric,
            work_items_output=args.work_items_output, manifest_output=args.manifest_output,
        )
    except RagSftV2CalibrationPackageError as error:
        raise SystemExit(f"[失败] {error}") from error
    print(f"[完成] 发布 {manifest['selection']['records']} 条 v2 校准审核工作项")


if __name__ == "__main__":
    main()
