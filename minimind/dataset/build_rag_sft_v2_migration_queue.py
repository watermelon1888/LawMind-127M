"""从只读 v1 事实建立 RAG-SFT v2 审核迁移队列。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .build_sft_evaluation_exclusions import question_digest
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset.build_sft_evaluation_exclusions import question_digest


DATASET_ROOT = Path(__file__).resolve().parent
RAG_SFT_ROOT = DATASET_ROOT / "RAG-SFT"
QUERY_POOL_ROOT = DATASET_ROOT / "QUERY-POOL"
DEFAULT_QUERY_POOL = QUERY_POOL_ROOT / "authoring" / "query-pool-v1.jsonl"
DEFAULT_QUERY_POOL_MANIFEST = QUERY_POOL_ROOT / "manifests" / "query-pool-v1.json"
DEFAULT_SOURCE_MAPPING = (
    QUERY_POOL_ROOT / "manifests" / "query-pool-v1-formal-source-mapping.jsonl"
)
DEFAULT_V1_AUTHORING = RAG_SFT_ROOT / "authoring" / "rag-sft-canonical-v2.jsonl"
DEFAULT_QUEUE_OUTPUT = RAG_SFT_ROOT / "review" / "v2" / "migration-queue-r2.jsonl"
DEFAULT_MANIFEST_OUTPUT = RAG_SFT_ROOT / "review" / "v2" / "migration-queue-r2.json"

_POOL_FIELDS = {"query_id", "query_original", "required_chunk_ids"}
_MAPPING_FIELDS = {"source_id", "query_id", "source_record_sha256", "query_sha256"}
_V1_FIELDS = {
    "id",
    "query_original",
    "evidence_source",
    "visible_chunk_ids",
    "required_chunk_ids",
    "target",
    "support_spans",
    "review_status",
    "review_notes",
}
_TARGET_FIELDS = {"summary", "refuse"}
_QUEUE_FIELDS = {
    "query_id",
    "source_v1_id",
    "source_record_sha256",
    "migration_status",
    "legacy_query_reworded",
    "legacy_summary",
    "legacy_support_spans",
}


class RagSftV2MigrationQueueError(RuntimeError):
    """表示不能安全建立 RAG-SFT v2 审核迁移队列。"""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    identity: dict[str, object] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if records is not None:
        identity["records"] = records
    return identity


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftV2MigrationQueueError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise RagSftV2MigrationQueueError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagSftV2MigrationQueueError(
                        f"{description}不允许空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RagSftV2MigrationQueueError(
                        f"{description}第 {line_number} 条必须是 object"
                    )
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftV2MigrationQueueError):
            raise
        raise RagSftV2MigrationQueueError(f"无法读取{description}: {path}") from error
    if not records:
        raise RagSftV2MigrationQueueError(f"{description}不能为空")
    return records


def _require_non_blank(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RagSftV2MigrationQueueError(f"{field} 必须是非空字符串")
    return value


def _required_chunk_ids(value: object, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= 3
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or len(value) != len(set(value))
    ):
        raise RagSftV2MigrationQueueError(f"{field} 必须为 1 至 3 条唯一 chunk_id")
    return tuple(value)


def _record_hash(record: dict[str, Any]) -> str:
    payload = json.dumps(
        record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return _sha256_bytes(payload.encode("utf-8"))


def _validate_query_pool_manifest(manifest: dict[str, Any]) -> None:
    readiness = manifest.get("readiness")
    if not isinstance(readiness, dict) or readiness.get("final_query_text_frozen") is not True:
        raise RagSftV2MigrationQueueError("公共 Query 池最终文本尚未冻结")


def _index_query_pool(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records, start=1):
        prefix = f"公共 Query 池第 {index} 条"
        if set(record) != _POOL_FIELDS:
            raise RagSftV2MigrationQueueError(f"{prefix}字段必须精确匹配 schema")
        query_id = _require_non_blank(record["query_id"], f"{prefix} query_id")
        _require_non_blank(record["query_original"], f"{prefix} query_original")
        _required_chunk_ids(record["required_chunk_ids"], f"{prefix} required_chunk_ids")
        if query_id in indexed:
            raise RagSftV2MigrationQueueError(f"公共 Query 池 query_id 重复: {query_id}")
        indexed[query_id] = record
    return indexed


def _index_mapping(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    source_ids: set[str] = set()
    for index, record in enumerate(records, start=1):
        prefix = f"公共 Query 池映射第 {index} 条"
        if set(record) != _MAPPING_FIELDS:
            raise RagSftV2MigrationQueueError(f"{prefix}字段必须精确匹配 schema")
        query_id = _require_non_blank(record["query_id"], f"{prefix} query_id")
        source_id = _require_non_blank(record["source_id"], f"{prefix} source_id")
        for field in ("source_record_sha256", "query_sha256"):
            value = _require_non_blank(record[field], f"{prefix} {field}")
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise RagSftV2MigrationQueueError(f"{prefix} {field} 无效")
        if query_id in indexed or source_id in source_ids:
            raise RagSftV2MigrationQueueError(f"{prefix} query_id 或 source_id 重复")
        indexed[query_id] = record
        source_ids.add(source_id)
    return indexed


def _index_v1_authoring(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records, start=1):
        prefix = f"v1 canonical authoring 第 {index} 条"
        if set(record) != _V1_FIELDS:
            raise RagSftV2MigrationQueueError(f"{prefix}字段必须精确匹配 schema")
        record_id = _require_non_blank(record["id"], f"{prefix} id")
        _require_non_blank(record["query_original"], f"{prefix} query_original")
        _required_chunk_ids(record["required_chunk_ids"], f"{prefix} required_chunk_ids")
        target = record["target"]
        if not isinstance(target, dict) or set(target) != _TARGET_FIELDS:
            raise RagSftV2MigrationQueueError(f"{prefix} target 字段必须精确匹配 schema")
        if not isinstance(target["summary"], str) or not isinstance(target["refuse"], bool):
            raise RagSftV2MigrationQueueError(f"{prefix} target 类型无效")
        if record["review_status"] != "approved":
            raise RagSftV2MigrationQueueError(f"{prefix}必须是 approved")
        if record_id in indexed:
            raise RagSftV2MigrationQueueError(f"v1 canonical authoring id 重复: {record_id}")
        indexed[record_id] = record
    return indexed


def _queue_record(
    pool: dict[str, Any], mapping: dict[str, Any], source: dict[str, Any]
) -> dict[str, object]:
    target = source["target"]
    query_matches = pool["query_original"] == source["query_original"]
    gt_matches = tuple(pool["required_chunk_ids"]) == tuple(source["required_chunk_ids"])
    legacy_query_reworded = not query_matches
    if target["refuse"]:
        status = "exclude_legacy_refusal"
        summary: str | None = None
        spans: list[object] = []
    elif not gt_matches:
        status = "pending_public_pool_changed"
        summary = None
        spans = []
    else:
        status = "pending_claim_authoring"
        summary = target["summary"]
        spans = source["support_spans"]
    return {
        "query_id": pool["query_id"],
        "source_v1_id": source["id"],
        "source_record_sha256": mapping["source_record_sha256"],
        "migration_status": status,
        "legacy_query_reworded": legacy_query_reworded,
        "legacy_summary": summary,
        "legacy_support_spans": spans,
    }


def _require_new_outputs(queue_output: Path, manifest_output: Path) -> None:
    targets = (queue_output, manifest_output, manifest_output.with_suffix(".sha256"))
    occupied = [
        str(path)
        for target in targets
        for path in (target, target.with_name(target.name + ".partial"))
        if path.exists()
    ]
    if occupied:
        raise RagSftV2MigrationQueueError("目标输出已存在: " + ", ".join(occupied))


def _publish(
    queue_output: Path, queue_payload: str, manifest_output: Path, manifest_payload: str
) -> None:
    hash_output = manifest_output.with_suffix(".sha256")
    hash_payload = f"{_sha256_bytes(manifest_payload.encode('utf-8'))}  {manifest_output.name}\n"
    outputs = [
        (queue_output, queue_payload),
        (manifest_output, manifest_payload),
        (hash_output, hash_payload),
    ]
    partials = [(path.with_name(path.name + ".partial"), path, payload) for path, payload in outputs]
    published: list[Path] = []
    try:
        for partial, _, payload in partials:
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(payload, encoding="utf-8", newline="\n")
        for partial, final, _ in partials:
            partial.replace(final)
            published.append(final)
    except (OSError, UnicodeError) as error:
        for path in [*(item[0] for item in partials), *reversed(published)]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise RagSftV2MigrationQueueError("无法发布 v2 审核迁移队列") from error


def build_rag_sft_v2_migration_queue(
    *,
    query_pool_path: Path,
    query_pool_manifest_path: Path,
    source_mapping_path: Path,
    v1_authoring_path: Path,
    queue_output: Path,
    manifest_output: Path,
) -> dict[str, object]:
    """建立只读 v1 到公共 Query 池的审核队列，不生成 authoring。"""

    query_pool_path = Path(query_pool_path).resolve()
    query_pool_manifest_path = Path(query_pool_manifest_path).resolve()
    source_mapping_path = Path(source_mapping_path).resolve()
    v1_authoring_path = Path(v1_authoring_path).resolve()
    queue_output = Path(queue_output).resolve()
    manifest_output = Path(manifest_output).resolve()
    _require_new_outputs(queue_output, manifest_output)

    query_pool_manifest = _load_json(query_pool_manifest_path, "公共 Query 池 manifest")
    _validate_query_pool_manifest(query_pool_manifest)
    pool = _index_query_pool(_load_jsonl(query_pool_path, "公共 Query 池"))
    mappings = _index_mapping(_load_jsonl(source_mapping_path, "公共 Query 池映射"))
    sources = _index_v1_authoring(_load_jsonl(v1_authoring_path, "v1 canonical authoring"))
    if set(pool) != set(mappings):
        raise RagSftV2MigrationQueueError("公共 Query 池与 source mapping 的 query_id 不一致")

    queue = []
    for query_id, pool_record in pool.items():
        mapping = mappings[query_id]
        source_id = mapping["source_id"]
        source = sources.get(source_id)
        if source is None:
            raise RagSftV2MigrationQueueError(f"source mapping 指向不存在的 v1 记录: {source_id}")
        if _record_hash(source) != mapping["source_record_sha256"]:
            raise RagSftV2MigrationQueueError(f"v1 记录身份已变化: {source_id}")
        if question_digest(source["query_original"]) != mapping["query_sha256"]:
            raise RagSftV2MigrationQueueError(f"v1 问题身份已变化: {source_id}")
        queue.append(_queue_record(pool_record, mapping, source))

    queue_payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for record in queue
    )
    status_counts = Counter(record["migration_status"] for record in queue)
    manifest = {
        "pipeline": "rag_sft_v2_migration_queue",
        "release_status": "review_queue",
        "inputs": {
            "query_pool": _identity(query_pool_path, records=len(pool)),
            "query_pool_manifest": _identity(query_pool_manifest_path),
            "source_mapping": _identity(source_mapping_path, records=len(mappings)),
            "v1_authoring": _identity(v1_authoring_path, records=len(sources)),
        },
        "output": {
            "queue": {
                "path": str(queue_output),
                "bytes": len(queue_payload.encode("utf-8")),
                "sha256": _sha256_bytes(queue_payload.encode("utf-8")),
                "records": len(queue),
            }
        },
        "records": {"total": len(queue), "by_migration_status": dict(sorted(status_counts.items()))},
        "migration_policy": {
            "legacy_query_reworded": (
                "required_chunk_ids 不变时保留为 pending_claim_authoring；"
                "旧摘要和 support spans 仅作草稿，必须按正式 query 重新审核"
            ),
            "required_chunk_ids_changed": "保留为 pending_public_pool_changed，不迁入 authoring",
        },
        "readiness": {"authoring_ready": False, "training_ready": False},
        "complete": True,
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _publish(queue_output, queue_payload, manifest_output, manifest_payload)
    return manifest


def main() -> None:
    """解析路径并发布 v2 审核迁移队列。"""

    parser = argparse.ArgumentParser(description="建立 RAG-SFT v2 审核迁移队列")
    parser.add_argument("--query-pool", type=Path, default=DEFAULT_QUERY_POOL)
    parser.add_argument("--query-pool-manifest", type=Path, default=DEFAULT_QUERY_POOL_MANIFEST)
    parser.add_argument("--source-mapping", type=Path, default=DEFAULT_SOURCE_MAPPING)
    parser.add_argument("--v1-authoring", type=Path, default=DEFAULT_V1_AUTHORING)
    parser.add_argument("--queue-output", type=Path, default=DEFAULT_QUEUE_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    args = parser.parse_args()
    try:
        manifest = build_rag_sft_v2_migration_queue(
            query_pool_path=args.query_pool,
            query_pool_manifest_path=args.query_pool_manifest,
            source_mapping_path=args.source_mapping,
            v1_authoring_path=args.v1_authoring,
            queue_output=args.queue_output,
            manifest_output=args.manifest_output,
        )
    except RagSftV2MigrationQueueError as error:
        raise SystemExit(f"[失败] {error}") from error
    print(f"[完成] 发布 {manifest['records']['total']} 条 v2 审核迁移队列")


if __name__ == "__main__":
    main()
