"""把已审定的 RAG SFT v1 记录保守迁移为待人工复核的 v2 草稿。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from rag.knowledge import ArticleRepository


RAG_SFT_ROOT = Path(__file__).resolve().parent / "RAG-SFT"
DEFAULT_SOURCE = RAG_SFT_ROOT / "authoring" / "rag-sft-v1.jsonl"
DEFAULT_ARTICLE_INDEX = Path(__file__).resolve().parents[2] / "rag" / "chunk" / "article_index.jsonl"
DEFAULT_OUTPUT = RAG_SFT_ROOT / "authoring" / "rag-sft-v2.jsonl"
DEFAULT_MANIFEST = RAG_SFT_ROOT / "manifests" / "rag-sft-v2-migration.json"

_V1_FIELDS = {
    "id",
    "sample_type",
    "query",
    "ordered_chunk_ids",
    "target",
    "review_status",
    "review_notes",
}
_TARGET_FIELDS = {"summary", "cited_chunk_ids", "quotes", "refuse"}
_QUOTE_FIELDS = {"chunk_id", "text"}
_REFUSAL_SAMPLE_TYPE = "evidence_insufficient_refusal"


class RagSftV2MigrationError(RuntimeError):
    """表示 v1 authoring 不能保守迁移为 v2 草稿。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RagSftV2MigrationError(
                        f"v1 authoring JSON 无效: {path}:{line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise RagSftV2MigrationError(
                        f"v1 authoring 记录必须是对象: {path}:{line_number}"
                    )
                records.append(record)
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftV2MigrationError(f"无法读取 v1 authoring: {path}") from error
    return records


def _require_non_blank(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RagSftV2MigrationError(f"{field} 必须是非空字符串")
    return value


def _validate_v1_record(record: dict[str, object], position: int) -> None:
    prefix = f"v1 authoring 第 {position} 条"
    if set(record) != _V1_FIELDS:
        raise RagSftV2MigrationError(f"{prefix}顶层字段必须精确匹配 v1 schema")
    _require_non_blank(record["id"], f"{prefix} id")
    _require_non_blank(record["query"], f"{prefix} query")
    if record["review_status"] != "approved":
        raise RagSftV2MigrationError(f"{prefix}只允许迁移 approved 记录")
    chunk_ids = record["ordered_chunk_ids"]
    if (
        not isinstance(chunk_ids, list)
        or not chunk_ids
        or any(not isinstance(item, str) or not item.strip() for item in chunk_ids)
        or len(set(chunk_ids)) != len(chunk_ids)
    ):
        raise RagSftV2MigrationError(
            f"{prefix} ordered_chunk_ids 必须是无重复的非空字符串数组"
        )
    target = record["target"]
    if not isinstance(target, dict) or set(target) != _TARGET_FIELDS:
        raise RagSftV2MigrationError(f"{prefix} target 字段必须精确匹配 schema")
    if not isinstance(target["refuse"], bool):
        raise RagSftV2MigrationError(f"{prefix} target.refuse 必须是布尔值")
    quotes = target["quotes"]
    if not isinstance(quotes, list):
        raise RagSftV2MigrationError(f"{prefix} target.quotes 必须是对象数组")
    for quote_index, quote in enumerate(quotes):
        if not isinstance(quote, dict) or set(quote) != _QUOTE_FIELDS:
            raise RagSftV2MigrationError(
                f"{prefix} target.quotes[{quote_index}] 字段必须精确匹配 schema"
            )
        _require_non_blank(quote["chunk_id"], f"{prefix} target.quotes[{quote_index}].chunk_id")
        _require_non_blank(quote["text"], f"{prefix} target.quotes[{quote_index}].text")
        if quote["chunk_id"] not in chunk_ids:
            raise RagSftV2MigrationError(f"{prefix} quote 不属于 ordered_chunk_ids")
    is_refusal = record["sample_type"] == _REFUSAL_SAMPLE_TYPE
    if is_refusal != target["refuse"]:
        raise RagSftV2MigrationError(f"{prefix} sample_type 与 target.refuse 不一致")


def _require_new_outputs(output: Path, manifest: Path) -> None:
    targets = (output, manifest, manifest.with_suffix(".sha256"))
    occupied = [
        str(path)
        for target in targets
        for path in (target, target.with_name(target.name + ".partial"))
        if path.exists()
    ]
    if occupied:
        raise RagSftV2MigrationError("目标输出已存在: " + ", ".join(occupied))


def _v2_excerpts(record: dict[str, object], repository: ArticleRepository) -> tuple[list[dict[str, str]], str]:
    chunk_ids = record["ordered_chunk_ids"]
    target = record["target"]
    if target["refuse"]:
        excerpts = [
            {"chunk_id": chunk_id, "text": repository.get_by_chunk_id(chunk_id).content}
            for chunk_id in chunk_ids
        ]
        return excerpts, "拒答候选暂保留完整法条，必须人工选择最小充分摘录。"

    excerpts = [
        {"chunk_id": quote["chunk_id"], "text": quote["text"]}
        for quote in target["quotes"]
    ]
    pairs = [(item["chunk_id"], item["text"]) for item in excerpts]
    if len(set(pairs)) != len(pairs):
        raise RagSftV2MigrationError(
            f"{record['id']} 的 target.quotes 包含完全重复摘录，无法保守迁移"
        )
    if {item["chunk_id"] for item in excerpts} != set(chunk_ids):
        raise RagSftV2MigrationError(
            f"{record['id']} 的 target.quotes 未覆盖每条 ordered evidence，无法擅自补摘录"
        )
    for excerpt in excerpts:
        article = repository.get_by_chunk_id(excerpt["chunk_id"])
        if excerpt["text"] not in article.content:
            raise RagSftV2MigrationError(
                f"{record['id']} 的 target quote 不属于当前法条正文"
            )
    return excerpts, "可见摘录暂取 v1 target.quotes，必须人工复核其对摘要的充分性。"


def _publish(output: Path, output_payload: str, manifest: Path, manifest_payload: str) -> None:
    manifest_hash = manifest.with_suffix(".sha256")
    hash_payload = (
        f"{hashlib.sha256(manifest_payload.encode('utf-8')).hexdigest()}  "
        f"{manifest.name}\n"
    )
    partials = [
        (output.with_name(output.name + ".partial"), output, output_payload),
        (manifest.with_name(manifest.name + ".partial"), manifest, manifest_payload),
        (manifest_hash.with_name(manifest_hash.name + ".partial"), manifest_hash, hash_payload),
    ]
    published = []
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
        raise RagSftV2MigrationError("无法发布 v2 迁移草稿") from error


def migrate_rag_sft_v1_to_v2(
    *,
    source_path: Path,
    article_index_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> dict[str, object]:
    """生成全部为 draft 的 v2 authoring 与可审计迁移 manifest。"""

    source_path = Path(source_path).resolve()
    article_index_path = Path(article_index_path).resolve()
    output_path = Path(output_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    _require_new_outputs(output_path, manifest_path)
    records = _load_jsonl(source_path)
    seen_ids = set()
    for position, record in enumerate(records, start=1):
        _validate_v1_record(record, position)
        if record["id"] in seen_ids:
            raise RagSftV2MigrationError(f"v1 authoring id 重复: {record['id']}")
        seen_ids.add(record["id"])
    try:
        repository = ArticleRepository.from_jsonl(article_index_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RagSftV2MigrationError("无法加载法条索引") from error

    migrated = []
    for position, record in enumerate(records, start=1):
        try:
            excerpts, review_requirement = _v2_excerpts(record, repository)
        except (KeyError, TypeError, ValueError) as error:
            raise RagSftV2MigrationError(
                f"{record['id']} 无法构造 v2 摘录草稿: {error}"
            ) from error
        migrated.append(
            {
                "id": f"rag_sft_v2:{position:04d}",
                "sample_type": record["sample_type"],
                "query": record["query"],
                "ordered_chunk_ids": record["ordered_chunk_ids"],
                "target": record["target"],
                "evidence_excerpts": excerpts,
                "review_status": "draft",
                "review_notes": (
                    f"由 {record['id']} 保守迁移为 v2 draft。{review_requirement}"
                ),
            }
        )
    output_payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in migrated
    )
    output_identity = {
        "path": str(output_path),
        "bytes": len(output_payload.encode("utf-8")),
        "sha256": hashlib.sha256(output_payload.encode("utf-8")).hexdigest(),
    }
    sample_type_counts = Counter(record["sample_type"] for record in migrated)
    manifest = {
        "schema_version": "1.0",
        "pipeline": "rag_sft_v1_to_v2_draft",
        "inputs": {
            "v1_authoring": _file_identity(source_path),
            "article_index": _file_identity(article_index_path),
        },
        "output": {"v2_authoring": output_identity},
        "records": {
            "source": len(records),
            "v2_draft": len(migrated),
            "by_sample_type": dict(sorted(sample_type_counts.items())),
        },
        "excerpt_policy": {
            "non_refusal": "copy_v1_target_quotes",
            "refusal": "copy_canonical_full_content_for_manual_reduction",
            "all_review_status": "draft",
        },
        "complete": True,
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _publish(output_path, output_payload, manifest_path, manifest_payload)
    return manifest


def main() -> None:
    """解析路径并发布 v2 authoring 草稿。"""

    parser = argparse.ArgumentParser(description="保守迁移 RAG SFT v1 authoring 为 v2 draft")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--article-index", type=Path, default=DEFAULT_ARTICLE_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    try:
        manifest = migrate_rag_sft_v1_to_v2(
            source_path=args.source,
            article_index_path=args.article_index,
            output_path=args.output,
            manifest_path=args.manifest,
        )
    except (RagSftV2MigrationError, OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"[失败] {error}") from error
    print(f"[完成] 迁移 {manifest['records']['v2_draft']} 条 v2 draft")


if __name__ == "__main__":
    main()
