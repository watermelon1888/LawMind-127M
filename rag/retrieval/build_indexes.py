"""从 article_index.jsonl 显式构建新 retrieval 索引产物。"""

import argparse
import json
from pathlib import Path

from rag.knowledge import ArticleRepository
from rag.retrieval.dense import build_dense_index, save_dense_index
from rag.retrieval.loader import DEFAULT_EMBEDDING_MODEL, _resolve_device
from rag.retrieval.sparse import (
    build_sparse_index,
    format_sparse_document,
    save_sparse_index,
)


RAG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARTICLE_INDEX = RAG_DIR / "chunk" / "article_index.jsonl"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


def load_index_inputs(path):
    """返回 canonical 法条和只用于 BM25 构建的层级文本。"""
    repository = ArticleRepository.from_jsonl(path)
    articles = []
    sparse_documents = []
    with Path(path).open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            article = repository.get_by_chunk_id(record["chunk_id"])
            articles.append(article)
            sparse_documents.append(
                format_sparse_document(
                    law_name=article.law_name,
                    article_no=article.article_no,
                    content=article.content,
                    hierarchy=record.get("hierarchy", {}),
                )
            )
    return tuple(articles), tuple(sparse_documents)


def build_selected_indexes(
    *,
    target,
    article_index,
    artifact_dir,
    embedding_model,
    device,
    overlap_tokens,
    batch_size,
):
    """只构建调用方明确选择的 dense、sparse 或全部索引。"""
    articles, sparse_documents = load_index_inputs(article_index)
    if target in ("sparse", "all"):
        bm25, metadata = build_sparse_index(
            sparse_documents,
            tuple(article.chunk_id for article in articles),
        )
        save_sparse_index(bm25, metadata, artifact_dir)
    if target in ("dense", "all"):
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(
            embedding_model,
            device=_resolve_device(device),
        )
        index, metadata = build_dense_index(
            articles,
            encoder,
            model_name=embedding_model,
            overlap_tokens=overlap_tokens,
            batch_size=batch_size,
        )
        save_dense_index(index, metadata, artifact_dir)


def main():
    parser = argparse.ArgumentParser(description="构建 retrieval 运行时索引")
    parser.add_argument("--target", required=True, choices=("dense", "sparse", "all"))
    parser.add_argument("--article-index", default=str(DEFAULT_ARTICLE_INDEX))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overlap-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    build_selected_indexes(
        target=args.target,
        article_index=args.article_index,
        artifact_dir=args.artifact_dir,
        embedding_model=args.embedding_model,
        device=args.device,
        overlap_tokens=args.overlap_tokens,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
