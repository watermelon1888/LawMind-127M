"""从已冻结的准入账本发布 RAG-SFT v2 canonical authoring 工作包。"""

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
ADMISSION_LEDGER = RAG_SFT_ROOT / "review" / "v2" / "admission-v1" / "admission-ledger.jsonl"
ADMISSION_MANIFEST = RAG_SFT_ROOT / "review" / "v2" / "admission-v1" / "manifest.json"
ARTICLE_INDEX = Path(__file__).resolve().parents[2] / "rag" / "chunk" / "article_index.jsonl"
OUTPUT_DIR = RAG_SFT_ROOT / "review" / "v2" / "canonical-authoring-work-v1"

_POOL_FIELDS = {"query_id", "query_original", "required_chunk_ids"}
_QUEUE_FIELDS = {
    "query_id", "source_v1_id", "source_record_sha256", "migration_status",
    "legacy_query_reworded", "legacy_summary", "legacy_support_spans",
}
_ARTICLE_FIELDS = {
    "chunk_id", "law_name", "article_no", "article_no_sort_key", "content",
    "token_count", "char_count", "department", "effective_date", "hierarchy",
}
_LEDGER_REQUIRED_FIELDS = {"query_id", "decision", "reason", "detail"}


class RagSftV2CanonicalAuthoringWorkPackageError(RuntimeError):
    """表示 canonical authoring 工作包无法可靠发布。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if records is not None:
        value["records"] = records
    return value


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftV2CanonicalAuthoringWorkPackageError(
            f"无法读取{description}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise RagSftV2CanonicalAuthoringWorkPackageError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagSftV2CanonicalAuthoringWorkPackageError(
                        f"{description}不允许空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RagSftV2CanonicalAuthoringWorkPackageError(
                        f"{description}第 {line_number} 条必须是对象"
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftV2CanonicalAuthoringWorkPackageError):
            raise
        raise RagSftV2CanonicalAuthoringWorkPackageError(
            f"无法读取{description}: {path}"
        ) from error
    if not rows:
        raise RagSftV2CanonicalAuthoringWorkPackageError(f"{description}不能为空")
    return rows


def _non_blank(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RagSftV2CanonicalAuthoringWorkPackageError(f"{field}必须是非空字符串")
    return value


def _index_pool(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(rows, start=1):
        if set(row) != _POOL_FIELDS:
            raise RagSftV2CanonicalAuthoringWorkPackageError(
                f"公共 Query 池第 {position} 条字段无效"
            )
        query_id = _non_blank(row.get("query_id"), "公共 Query 池 query_id")
        _non_blank(row.get("query_original"), "公共 Query 池 query_original")
        required = row.get("required_chunk_ids")
        if (
            query_id in indexed
            or not isinstance(required, list)
            or not 1 <= len(required) <= 3
            or any(not isinstance(item, str) or not item.strip() for item in required)
            or len(required) != len(set(required))
        ):
            raise RagSftV2CanonicalAuthoringWorkPackageError(
                f"公共 Query 池第 {position} 条 required GT 无效"
            )
        indexed[query_id] = row
    return indexed


def _index_queue(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(rows, start=1):
        if set(row) != _QUEUE_FIELDS:
            raise RagSftV2CanonicalAuthoringWorkPackageError(
                f"迁移队列第 {position} 条字段无效"
            )
        query_id = _non_blank(row.get("query_id"), "迁移队列 query_id")
        if query_id in indexed:
            raise RagSftV2CanonicalAuthoringWorkPackageError("迁移队列 query_id 重复")
        indexed[query_id] = row
    return indexed


def _index_articles(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for position, row in enumerate(rows, start=1):
        if set(row) != _ARTICLE_FIELDS:
            raise RagSftV2CanonicalAuthoringWorkPackageError(
                f"法条索引第 {position} 条字段无效"
            )
        chunk_id = _non_blank(row.get("chunk_id"), "法条索引 chunk_id")
        if chunk_id in indexed:
            raise RagSftV2CanonicalAuthoringWorkPackageError("法条索引 chunk_id 重复")
        indexed[chunk_id] = {
            "chunk_id": chunk_id,
            "law_name": _non_blank(row.get("law_name"), "法条索引 law_name"),
            "article_no": _non_blank(row.get("article_no"), "法条索引 article_no"),
            "content": _non_blank(row.get("content"), "法条索引 content"),
        }
    return indexed


def _index_ledger(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for position, row in enumerate(rows, start=1):
        if set(row) != _LEDGER_REQUIRED_FIELDS:
            raise RagSftV2CanonicalAuthoringWorkPackageError(
                f"准入账本第 {position} 条字段无效"
            )
        query_id = _non_blank(row.get("query_id"), "准入账本 query_id")
        decision = _non_blank(row.get("decision"), "准入账本 decision")
        reason = _non_blank(row.get("reason"), "准入账本 reason")
        detail = _non_blank(row.get("detail"), "准入账本 detail")
        if query_id in indexed or decision not in {"admit", "exclude"}:
            raise RagSftV2CanonicalAuthoringWorkPackageError("准入账本记录无效或重复")
        indexed[query_id] = {
            "query_id": query_id,
            "decision": decision,
            "reason": reason,
            "detail": detail,
        }
    return indexed


def _require_new_outputs(paths: list[Path]) -> None:
    occupied = [
        str(candidate)
        for path in paths
        for candidate in (path, path.with_name(path.name + ".partial"))
        if candidate.exists()
    ]
    if occupied:
        raise RagSftV2CanonicalAuthoringWorkPackageError(
            "目标输出已存在: " + ", ".join(occupied)
        )


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
        raise RagSftV2CanonicalAuthoringWorkPackageError("无法发布 canonical authoring 工作包") from error


def build_rag_sft_v2_canonical_authoring_work_package(
    *,
    query_pool_path: Path,
    migration_queue_path: Path,
    admission_ledger_path: Path,
    admission_manifest_path: Path,
    article_index_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """为准入 query 发布人工编写 canonical authoring 的唯一工作包。"""

    paths = {
        "query_pool": Path(query_pool_path).resolve(),
        "migration_queue": Path(migration_queue_path).resolve(),
        "admission_ledger": Path(admission_ledger_path).resolve(),
        "admission_manifest": Path(admission_manifest_path).resolve(),
        "article_index": Path(article_index_path).resolve(),
    }
    output_dir = Path(output_dir).resolve()
    work_items_path = output_dir / "work-items.jsonl"
    exclusions_path = output_dir / "exclusion-closure.jsonl"
    manifest_path = output_dir / "manifest.json"
    hash_path = output_dir / "manifest.sha256"
    if any(not path.is_file() for path in paths.values()):
        raise RagSftV2CanonicalAuthoringWorkPackageError("canonical authoring 工作包输入缺失")
    _require_new_outputs([work_items_path, exclusions_path, manifest_path, hash_path])

    admission_manifest = _load_json(paths["admission_manifest"], "准入 manifest")
    if (
        admission_manifest.get("release_status") != "admission_finalized"
        or admission_manifest.get("readiness", {}).get("authoring_ready") is not True
        or admission_manifest.get("policy", {}).get("canonical_authoring_emitted") is not False
    ):
        raise RagSftV2CanonicalAuthoringWorkPackageError("准入 manifest 不是可开始 authoring 的冻结状态")

    pool_rows = _load_jsonl(paths["query_pool"], "公共 Query 池")
    queue_rows = _load_jsonl(paths["migration_queue"], "迁移队列")
    ledger_rows = _load_jsonl(paths["admission_ledger"], "准入账本")
    article_rows = _load_jsonl(paths["article_index"], "法条索引")
    pool = _index_pool(pool_rows)
    queue = _index_queue(queue_rows)
    ledger = _index_ledger(ledger_rows)
    articles = _index_articles(article_rows)
    pending = {
        query_id
        for query_id, row in queue.items()
        if row["migration_status"] == "pending_claim_authoring"
    }
    if set(pool) != set(queue) or set(ledger) != pending:
        raise RagSftV2CanonicalAuthoringWorkPackageError(
            "公共 Query 池、迁移队列与准入账本未对同一待编写集合闭合"
        )

    manifest_records = admission_manifest.get("records")
    if not isinstance(manifest_records, dict):
        raise RagSftV2CanonicalAuthoringWorkPackageError("准入 manifest 缺少记录计数")

    work_items: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    for query_id in (row["query_id"] for row in pool_rows if row["query_id"] in pending):
        pool_row = pool[query_id]
        ledger_row = ledger[query_id]
        required_articles = []
        for chunk_id in pool_row["required_chunk_ids"]:
            article = articles.get(chunk_id)
            if article is None:
                raise RagSftV2CanonicalAuthoringWorkPackageError(
                    f"准入 query 缺少 required GT 正文: {query_id}/{chunk_id}"
                )
            required_articles.append(article)
        if ledger_row["decision"] == "exclude":
            exclusions.append({
                "query_id": query_id,
                "query_original": pool_row["query_original"],
                "required_chunk_ids": pool_row["required_chunk_ids"],
                "reason": ledger_row["reason"],
                "detail": ledger_row["detail"],
            })
            continue
        legacy = queue[query_id]
        work_items.append({
            "query_id": query_id,
            "query_original": pool_row["query_original"],
            "required_chunk_ids": pool_row["required_chunk_ids"],
            "required_articles": required_articles,
            "legacy_draft": {
                "source_v1_id": legacy["source_v1_id"],
                "source_record_sha256": legacy["source_record_sha256"],
                "legacy_query_reworded": legacy["legacy_query_reworded"],
                "legacy_summary": legacy["legacy_summary"],
                "legacy_support_spans": legacy["legacy_support_spans"],
            },
        })

    if len(work_items) + len(exclusions) != len(pending):
        raise RagSftV2CanonicalAuthoringWorkPackageError("准入账本统计未闭合")
    if (
        manifest_records.get("pending_claim_authoring") != len(pending)
        or manifest_records.get("admit") != len(work_items)
        or manifest_records.get("exclude") != len(exclusions)
    ):
        raise RagSftV2CanonicalAuthoringWorkPackageError("准入 manifest 与账本统计不一致")
    work_payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for row in work_items
    )
    exclusion_payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for row in exclusions
    )
    manifest = {
        "pipeline": "rag_sft_v2_canonical_authoring_work_package",
        "release_status": "human_authoring_only",
        "inputs": {
            "query_pool": _identity(paths["query_pool"], records=len(pool_rows)),
            "migration_queue": _identity(paths["migration_queue"], records=len(queue_rows)),
            "admission_ledger": _identity(paths["admission_ledger"], records=len(ledger_rows)),
            "admission_manifest": _identity(paths["admission_manifest"]),
            "article_index": _identity(paths["article_index"], records=len(article_rows)),
        },
        "records": {
            "pending_claim_authoring": len(pending),
            "admit_work_items": len(work_items),
            "exclude_closure": len(exclusions),
        },
        "policy": {
            "legacy_draft_is_reference_only": True,
            "canonical_authoring_emitted": False,
            "oracle_clean_emitted": False,
            "real_hard_negatives_constructed": False,
            "required_gt_is_read_only_from_public_query_pool": True,
        },
        "output": {
            "work_items": {"path": str(work_items_path), "records": len(work_items)},
            "exclusion_closure": {"path": str(exclusions_path), "records": len(exclusions)},
        },
        "readiness": {"authoring_work_package_ready": True, "training_ready": False},
        "complete": True,
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    hash_payload = f"{hashlib.sha256(manifest_payload.encode('utf-8')).hexdigest()}  manifest.json\n"
    _publish([
        (work_items_path, work_payload),
        (exclusions_path, exclusion_payload),
        (manifest_path, manifest_payload),
        (hash_path, hash_payload),
    ])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="发布 RAG-SFT v2 canonical authoring 工作包")
    parser.add_argument("--query-pool", type=Path, default=QUERY_POOL)
    parser.add_argument("--migration-queue", type=Path, default=MIGRATION_QUEUE)
    parser.add_argument("--admission-ledger", type=Path, default=ADMISSION_LEDGER)
    parser.add_argument("--admission-manifest", type=Path, default=ADMISSION_MANIFEST)
    parser.add_argument("--article-index", type=Path, default=ARTICLE_INDEX)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    try:
        manifest = build_rag_sft_v2_canonical_authoring_work_package(
            query_pool_path=args.query_pool,
            migration_queue_path=args.migration_queue,
            admission_ledger_path=args.admission_ledger,
            admission_manifest_path=args.admission_manifest,
            article_index_path=args.article_index,
            output_dir=args.output_dir,
        )
    except RagSftV2CanonicalAuthoringWorkPackageError as error:
        raise SystemExit(f"[失败] {error}") from error
    print(
        f"[完成] 准入工作项 {manifest['records']['admit_work_items']} 条，"
        f"排除闭合 {manifest['records']['exclude_closure']} 条"
    )


if __name__ == "__main__":
    main()
