"""将已裁决的小批 RAG-SFT v2 canonical authoring 发布为不可覆盖资产。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .rag_sft_v2_contract import RagSftV2ContractError, validate_canonical_authoring


DATASET_ROOT = Path(__file__).resolve().parent
QUERY_POOL = DATASET_ROOT / "QUERY-POOL" / "authoring" / "query-pool-v1.jsonl"
ARTICLE_INDEX = Path(__file__).resolve().parents[2] / "rag" / "chunk" / "article_index.jsonl"

_POOL_FIELDS = {"query_id", "query_original", "required_chunk_ids"}
_ARTICLE_FIELDS = {
    "chunk_id", "law_name", "article_no", "article_no_sort_key", "content",
    "token_count", "char_count", "department", "effective_date", "hierarchy",
}


class RagSftV2CanonicalBatchFinalizationError(RuntimeError):
    """表示 canonical 小批无法按审核裁决安全发布。"""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)
    }
    if records is not None:
        value["records"] = records
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagSftV2CanonicalBatchFinalizationError(
                        f"{description}不允许空行: {number}"
                    )
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise RagSftV2CanonicalBatchFinalizationError(
                        f"{description}第 {number} 条必须是对象"
                    )
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftV2CanonicalBatchFinalizationError):
            raise
        raise RagSftV2CanonicalBatchFinalizationError(f"无法读取{description}: {path}") from error
    if not rows:
        raise RagSftV2CanonicalBatchFinalizationError(f"{description}不能为空")
    return rows


def _index_pool(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(rows, start=1):
        if set(row) != _POOL_FIELDS or not isinstance(row.get("query_id"), str):
            raise RagSftV2CanonicalBatchFinalizationError(f"公共 Query 池第 {number} 条无效")
        if row["query_id"] in result:
            raise RagSftV2CanonicalBatchFinalizationError("公共 Query 池 query_id 重复")
        result[row["query_id"]] = row
    return result


def _article_contents(rows: list[dict[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for number, row in enumerate(rows, start=1):
        if (
            set(row) != _ARTICLE_FIELDS
            or not isinstance(row.get("chunk_id"), str)
            or not isinstance(row.get("content"), str)
            or row["chunk_id"] in result
        ):
            raise RagSftV2CanonicalBatchFinalizationError(f"法条索引第 {number} 条无效")
        result[row["chunk_id"]] = row["content"]
    return result


def _load_adjudications(path: Path) -> dict[str, str]:
    rows = _load_jsonl(path, "主审裁决账本")
    decisions: dict[str, str] = {}
    for number, row in enumerate(rows, start=1):
        if set(row) != {"query_id", "final_decision"} or not isinstance(row.get("query_id"), str):
            raise RagSftV2CanonicalBatchFinalizationError(f"主审裁决账本第 {number} 条字段无效")
        if row["final_decision"] not in {"approved", "excluded", "escalated"}:
            raise RagSftV2CanonicalBatchFinalizationError("主审裁决类型无效")
        if row["query_id"] in decisions:
            raise RagSftV2CanonicalBatchFinalizationError("主审裁决账本 query_id 重复")
        decisions[row["query_id"]] = row["final_decision"]
    if not any(decision == "approved" for decision in decisions.values()):
        raise RagSftV2CanonicalBatchFinalizationError("主审裁决账本没有已批准记录")
    return decisions


def _publish(output_dir: Path, payloads: list[tuple[str, str]]) -> None:
    if output_dir.exists():
        raise RagSftV2CanonicalBatchFinalizationError(f"输出目录必须是新目录: {output_dir}")
    output_dir.mkdir(parents=True)
    pending = [(output_dir / f"{name}.partial", output_dir / name, payload) for name, payload in payloads]
    published: list[Path] = []
    try:
        for partial, _, payload in pending:
            partial.write_text(payload, encoding="utf-8", newline="\n")
        hash_payload = "".join(
            f"{_sha256(partial)}  {final.name}\n" for partial, final, _ in pending
        )
        hash_partial = output_dir / "manifest.sha256.partial"
        hash_partial.write_text(hash_payload, encoding="utf-8", newline="\n")
        for partial, final, _ in pending:
            partial.replace(final)
            published.append(final)
        hash_partial.replace(output_dir / "manifest.sha256")
    except (OSError, UnicodeError) as error:
        for path in [*(item[0] for item in pending), output_dir / "manifest.sha256.partial", *reversed(published)]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            output_dir.rmdir()
        except OSError:
            pass
        raise RagSftV2CanonicalBatchFinalizationError("无法发布 canonical 小批") from error


def finalize_rag_sft_v2_canonical_batch(
    *,
    candidate_paths: list[Path],
    adjudication_path: Path,
    query_pool_path: Path,
    article_index_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """校验已批准候选并发布纯四字段 canonical authoring 小批。"""

    candidates = [Path(path).resolve() for path in candidate_paths]
    adjudication_path = Path(adjudication_path).resolve()
    query_pool_path = Path(query_pool_path).resolve()
    article_index_path = Path(article_index_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not candidates or any(not path.is_file() for path in [*candidates, adjudication_path, query_pool_path, article_index_path]):
        raise RagSftV2CanonicalBatchFinalizationError("canonical 小批输入缺失")
    adjudications = _load_adjudications(adjudication_path)
    pool_rows = _load_jsonl(query_pool_path, "公共 Query 池")
    article_rows = _load_jsonl(article_index_path, "法条索引")
    pool = _index_pool(pool_rows)
    contents = _article_contents(article_rows)
    validated_by_id: dict[str, dict[str, object]] = {}
    seen: set[str] = set()
    for path in candidates:
        for number, candidate in enumerate(_load_jsonl(path, f"候选草稿 {path.name}"), start=1):
            query_id = candidate.get("query_id")
            if not isinstance(query_id, str) or query_id in seen:
                raise RagSftV2CanonicalBatchFinalizationError("候选 query_id 无效或重复")
            seen.add(query_id)
            if query_id not in pool:
                raise RagSftV2CanonicalBatchFinalizationError("候选不属于公共 Query 池")
            try:
                validated = validate_canonical_authoring(candidate, pool[query_id], contents)
            except RagSftV2ContractError as error:
                raise RagSftV2CanonicalBatchFinalizationError(
                    f"候选 {query_id} 未通过 canonical 契约"
                ) from error
            validated_by_id[query_id] = {
                field: validated[field]
                for field in ("query_id", "query_original", "claims", "summary")
            }
    if set(adjudications) != seen:
        raise RagSftV2CanonicalBatchFinalizationError("主审裁决集与候选集不闭合")
    approved_ids = {query_id for query_id, decision in adjudications.items() if decision == "approved"}
    records = [record for query_id, record in validated_by_id.items() if query_id in approved_ids]
    payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for record in records
    )
    manifest = {
        "pipeline": "rag_sft_v2_canonical_batch_finalization",
        "release_status": "canonical_batch_approved",
        "inputs": {
            "query_pool": _identity(query_pool_path, records=len(pool_rows)),
            "article_index": _identity(article_index_path, records=len(article_rows)),
            "adjudication": _identity(adjudication_path, records=len(adjudications)),
            "candidates": [_identity(path) for path in candidates],
        },
        "records": {
            "candidate_authoring": len(validated_by_id),
            "canonical_authoring": len(records),
            "excluded_or_escalated": len(validated_by_id) - len(records),
        },
        "policy": {"canonical_schema": ["query_id", "query_original", "claims", "summary"], "review_metadata_excluded": True},
        "output": {"canonical_authoring": {"path": str(output_dir / "canonical-authoring.jsonl"), "records": len(records)}},
        "readiness": {"canonical_batch_approved": True, "oracle_clean_emitted": False, "training_ready": False},
        "complete": True,
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _publish(output_dir, [("canonical-authoring.jsonl", payload), ("manifest.json", manifest_payload)])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="发布已批准的 RAG-SFT v2 canonical 小批")
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--adjudication", type=Path, required=True)
    parser.add_argument("--query-pool", type=Path, default=QUERY_POOL)
    parser.add_argument("--article-index", type=Path, default=ARTICLE_INDEX)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = finalize_rag_sft_v2_canonical_batch(
            candidate_paths=args.candidate, adjudication_path=args.adjudication,
            query_pool_path=args.query_pool, article_index_path=args.article_index,
            output_dir=args.output_dir,
        )
    except RagSftV2CanonicalBatchFinalizationError as error:
        raise SystemExit(f"[失败] {error}") from error
    print(f"[完成] 发布 {manifest['records']['canonical_authoring']} 条 canonical authoring")


if __name__ == "__main__":
    main()
