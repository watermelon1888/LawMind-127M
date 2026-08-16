"""从只读 BM25 索引生成 curated HN 的人工语义筛选队列。

队列只提供真实法条短名单，绝不把 BM25 排名、相似主题或队列中的任何候选
当作 HN 决定。每条进入后续账本的法条仍须经过独立对抗审阅、法律支持审阅和
主审裁决。本脚本不重建索引、不加载模型、不做上下文长度审核。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Iterable

from rag.retrieval.sparse import SparseSearcher

from .build_rag_sft_v2_curated_hn_reuse_review import (
    DEFAULT_BASE_RELEASE,
    DEFAULT_OUTPUT_DIR as DEFAULT_REUSE_REVIEW_ROOT,
    HASH_FILENAME,
    MANIFEST_FILENAME,
    CuratedHnReuseReviewError,
    _bound_path,
    _canonical_from_clean,
    _identity,
    _load_base_release,
    _load_json,
    _load_jsonl,
    _payload_identity,
    _sha256,
    _serialize_jsonl,
    _verify_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTICLE_INDEX = PROJECT_ROOT / "rag" / "chunk" / "article_index.jsonl"
DEFAULT_SPARSE_INDEX = PROJECT_ROOT / "rag" / "retrieval" / "artifacts" / "law_sparse.pkl"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "minimind"
    / "dataset"
    / "RAG-SFT"
    / "review"
    / "v2"
    / "curated-hn-manual-review-queue-v1-20260814"
)
QUEUE_FILENAME = "manual-review-queue.jsonl"
README_FILENAME = "README.md"


class CuratedHnManualQueueError(RuntimeError):
    """curated HN 人工筛选队列的输入身份或输出边界未闭合。"""


def _load_articles(path: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in _load_jsonl(path, "法条索引"):
        chunk_id = row.get("chunk_id")
        law_name = row.get("law_name")
        article_no = row.get("article_no")
        content = row.get("content")
        if (
            not isinstance(chunk_id, str)
            or not isinstance(law_name, str)
            or not isinstance(article_no, str)
            or not isinstance(content, str)
            or not chunk_id
            or not content.strip()
            or chunk_id in result
        ):
            raise CuratedHnManualQueueError("法条索引存在无效或重复 chunk_id")
        result[chunk_id] = {
            "chunk_id": chunk_id,
            "law_name": law_name,
            "article_no": article_no,
            "content": content,
        }
    if not result:
        raise CuratedHnManualQueueError("法条索引为空")
    return result


def _load_reuse_queries(reuse_root: Path) -> tuple[dict[str, Any], str, set[str]]:
    manifest_path = reuse_root / MANIFEST_FILENAME
    manifest_sha = _verify_manifest(manifest_path)
    manifest = _load_json(manifest_path, "curated HN 复用审阅 manifest")
    if (
        manifest.get("pipeline") != "rag_sft_v2_curated_hn_reuse_semantic_review"
        or manifest.get("release_status") != "curated_hn_reuse_semantic_review_closed"
        or manifest.get("records", {}).get("curated_hn_variants") != 98
        or manifest.get("policy", {}).get("formal_hn_materialized") is not False
        or manifest.get("policy", {}).get("training_ready") is not False
        or manifest.get("policy", {}).get("context_length_audit") != "deferred"
    ):
        raise CuratedHnManualQueueError("curated HN 复用审阅资产状态无效")
    variants_path = _bound_path(
        manifest.get("output", {}).get("curated_hn_variants"), "curated HN 复用 variants"
    )
    variants = _load_jsonl(variants_path, "curated HN 复用 variants")
    queries: set[str] = set()
    for row in variants:
        query_id = row.get("query_id")
        if (
            not isinstance(query_id, str)
            or query_id in queries
            or row.get("source") != "curated"
            or row.get("review_decision") != "approved"
        ):
            raise CuratedHnManualQueueError("curated HN 复用 variants 身份或状态无效")
        queries.add(query_id)
    if len(queries) != 98:
        raise CuratedHnManualQueueError("curated HN 复用 query 数量不闭合")
    return manifest, manifest_sha, queries


def _load_sparse_searcher(path: Path) -> SparseSearcher:
    try:
        with path.open("rb") as source:
            payload = pickle.load(source)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError) as error:
        raise CuratedHnManualQueueError(f"无法读取只读 BM25 稀疏索引: {path}") from error
    if not isinstance(payload, dict):
        raise CuratedHnManualQueueError("BM25 稀疏索引格式无效")
    metadata = payload.get("metadata")
    positions = metadata.get("position_to_chunk_id") if isinstance(metadata, dict) else None
    if not isinstance(positions, (list, tuple)) or not positions:
        raise CuratedHnManualQueueError("BM25 稀疏索引缺少 chunk_id 映射")
    try:
        return SparseSearcher(bm25=payload.get("bm25"), position_to_chunk_id=positions)
    except (TypeError, ValueError) as error:
        raise CuratedHnManualQueueError("BM25 稀疏索引无法构造只读搜索器") from error


def _partition(index: int) -> str:
    """为三名独立对抗审阅者固定分配不重叠的 query 子集。"""

    return ("a", "b", "c")[index % 3]


def prepare_rag_sft_v2_curated_hn_manual_review_queue(
    *,
    base_release: Path,
    reuse_review_root: Path,
    article_index: Path,
    sparse_index: Path,
    output_dir: Path,
    top_k: int = 8,
) -> dict[str, object]:
    """发布不可覆盖的真实法条人工筛选队列。"""

    if type(top_k) is not int or not 1 <= top_k <= 12:
        raise CuratedHnManualQueueError("人工筛选队列 top_k 必须在 1 至 12 之间")
    base_release = Path(base_release).resolve()
    reuse_review_root = Path(reuse_review_root).resolve()
    article_index = Path(article_index).resolve()
    sparse_index = Path(sparse_index).resolve()
    output_dir = Path(output_dir).resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise CuratedHnManualQueueError(f"输出目录或临时目录已存在: {output_dir}")
    if not article_index.is_file() or not sparse_index.is_file():
        raise CuratedHnManualQueueError("法条索引或 BM25 稀疏索引不存在")

    _base_manifest, base_manifest_sha, base_candidate_path, clean_by_query, existing_hn_queries = _load_base_release(base_release)
    reuse_manifest, reuse_manifest_sha, reuse_queries = _load_reuse_queries(reuse_review_root)
    if existing_hn_queries.intersection(reuse_queries):
        raise CuratedHnManualQueueError("retrieved HN 与 curated 复用 HN query 发生重叠")
    remaining_queries = sorted(set(clean_by_query) - existing_hn_queries - reuse_queries)
    if len(remaining_queries) != 381:
        raise CuratedHnManualQueueError("待人工筛选的无 HN query 数量不为 381")
    articles = _load_articles(article_index)
    searcher = _load_sparse_searcher(sparse_index)

    queue_rows: list[dict[str, object]] = []
    no_shortlist_queries: list[str] = []
    for index, query_id in enumerate(remaining_queries):
        canonical = _canonical_from_clean(clean_by_query[query_id])
        required = canonical["required_chunk_ids"]
        if any(chunk_id not in articles for chunk_id in required):
            raise CuratedHnManualQueueError(f"required GT 不存在于法条索引: {query_id}")
        try:
            ranked = searcher.search(canonical["query_original"], top_k=top_k + len(required))
        except Exception as error:
            raise CuratedHnManualQueueError(f"BM25 读取失败: {query_id}") from error
        candidates: list[dict[str, object]] = []
        for item in ranked:
            if item.chunk_id in required or item.chunk_id not in articles:
                continue
            article = articles[item.chunk_id]
            candidates.append({"rank": len(candidates) + 1, "score": item.score, **article})
            if len(candidates) == top_k:
                break
        if not candidates:
            no_shortlist_queries.append(query_id)
        queue_rows.append(
            {
                "query_id": query_id,
                "review_partition": _partition(index),
                "query_original": canonical["query_original"],
                "summary": canonical["summary"],
                "required_gt": [articles[chunk_id] for chunk_id in required],
                "candidate_articles": candidates,
                "queue_policy": "候选仅供人工语义筛选；不得因 BM25 排名或主题相近直接准入 HN。",
            }
        )

    queue_payload = _serialize_jsonl(queue_rows)
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest: dict[str, object] = {
        "pipeline": "rag_sft_v2_curated_hn_manual_review_queue",
        "release_status": "manual_semantic_review_queue_ready",
        "inputs": {
            "base_release_manifest": {
                **_identity(base_release / MANIFEST_FILENAME),
                "manifest_sha256": base_manifest_sha,
            },
            "base_training_candidate": _identity(base_candidate_path, records=619),
            "curated_reuse_manifest": {
                **_identity(reuse_review_root / MANIFEST_FILENAME),
                "manifest_sha256": reuse_manifest_sha,
            },
            "article_index": _identity(article_index, records=len(articles)),
            "read_only_sparse_index": _identity(sparse_index),
        },
        "records": {
            "base_clean": 549,
            "base_retrieved_hn": 70,
            "curated_reuse_hn": len(reuse_queries),
            "manual_review_queries": len(queue_rows),
            "candidate_articles_per_query_max": top_k,
            "queries_without_shortlist": len(no_shortlist_queries),
            "partition_queries": {
                partition: sum(1 for row in queue_rows if row["review_partition"] == partition)
                for partition in ("a", "b", "c")
            },
        },
        "policy": {
            "article_source": "rag/chunk/article_index.jsonl",
            "sparse_index": "只读短名单辅助；不构成 HN 审批或检索评估",
            "source_candidates_modified": False,
            "base_619_release_modified": False,
            "formal_hn_materialized": False,
            "context_length_audit": "deferred",
            "training_ready": False,
        },
        "validation": {
            "strict_utf8_jsonl": True,
            "base_release_identity_closed": True,
            "reuse_review_identity_closed": True,
            "all_queries_currently_without_hn": True,
            "all_shortlist_articles_from_real_article_index": True,
            "sparse_index_not_rebuilt": True,
            "no_model_or_tokenizer_loaded": True,
            "no_context_length_audit_run": True,
        },
        "output": {
            "manual_review_queue": _payload_identity(output_dir / QUEUE_FILENAME, queue_payload, records=len(queue_rows)),
        },
        "complete": True,
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    readme_payload = (
        "# RAG-SFT v2 curated HN 人工语义筛选队列\n\n"
        "本目录为 381 个尚无 HN 的 query 提供真实法条短名单，供独立对抗审阅者人工筛选。"
        "短名单来自只读 BM25 稀疏索引，不重建索引、不运行检索评估。\n\n"
        "队列中的候选不是 HN，也不代表法律支持关系。每条候选必须另行经过对抗审阅、"
        "法律支持审阅和主审裁决；直接、partial、alternative、redundant_support、uncertain "
        "及纯无关法条均不得进入 curated HN。\n\n"
        "本目录不包含训练数据或正式 HN：`formal_hn_materialized=false`、"
        "`context_length_audit=deferred`、`training_ready=false`。\n"
    )
    payloads = {
        QUEUE_FILENAME: queue_payload,
        MANIFEST_FILENAME: manifest_payload,
        README_FILENAME: readme_payload,
    }
    try:
        partial_dir.mkdir(parents=True, exist_ok=False)
        for name, payload in payloads.items():
            (partial_dir / name).write_text(payload, encoding="utf-8", newline="\n")
        hash_payload = "".join(f"{_sha256(partial_dir / name)}  {name}\n" for name in payloads)
        (partial_dir / HASH_FILENAME).write_text(hash_payload, encoding="ascii", newline="\n")
        partial_dir.replace(output_dir)
    except OSError as error:
        raise CuratedHnManualQueueError(
            "无法原子发布人工筛选队列；已保留临时目录以便审计"
        ) from error
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-release", type=Path, default=DEFAULT_BASE_RELEASE)
    parser.add_argument("--reuse-review-root", type=Path, default=DEFAULT_REUSE_REVIEW_ROOT)
    parser.add_argument("--article-index", type=Path, default=DEFAULT_ARTICLE_INDEX)
    parser.add_argument("--sparse-index", type=Path, default=DEFAULT_SPARSE_INDEX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()
    try:
        manifest = prepare_rag_sft_v2_curated_hn_manual_review_queue(
            base_release=args.base_release,
            reuse_review_root=args.reuse_review_root,
            article_index=args.article_index,
            sparse_index=args.sparse_index,
            output_dir=args.output_dir,
            top_k=args.top_k,
        )
    except (CuratedHnReuseReviewError, CuratedHnManualQueueError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    records = manifest["records"]
    print(
        "RAG_SFT_V2_CURATED_MANUAL_QUEUE_OK "
        f"queries={records['manual_review_queries']} training_ready=false"
    )


if __name__ == "__main__":
    main()
