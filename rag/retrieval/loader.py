"""显式加载本地检索产物和重量级模型。"""

import json
import pickle
from pathlib import Path

from rag.knowledge import ArticleRepository
from rag.retrieval.dense import DenseSearcher
from rag.retrieval.rerank import WindowReranker
from rag.retrieval.semantic import SemanticRetrievalConfig, SemanticRetriever
from rag.retrieval.sparse import SparseSearcher


DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


def validate_loaded_artifacts(
    *,
    repository,
    dense_index,
    dense_metadata,
    sparse_metadata,
    embedding_dimension,
    bm25_corpus_size=None,
):
    """在接收请求前校验索引数量、维度和 canonical chunk_id。"""
    if not isinstance(repository, ArticleRepository):
        raise TypeError("repository 必须是 ArticleRepository")
    windows = tuple(dense_metadata.get("position_to_window", ()))
    if not windows or int(dense_index.ntotal) != len(windows):
        raise ValueError("FAISS 向量数必须与窗口元数据条数一致且非空")
    if int(dense_index.d) != int(embedding_dimension):
        raise ValueError("FAISS 向量维度与 embedding 模型不一致")

    seen_windows = set()
    for item in windows:
        try:
            chunk_id = item["chunk_id"]
            window_index = item["window_index"]
            start_token = item["start_token"]
            end_token = item["end_token"]
        except (KeyError, TypeError) as exc:
            raise ValueError("dense 窗口元数据字段不完整") from exc
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (window_index, start_token, end_token)
        ):
            raise ValueError("dense 窗口位置必须是整数")
        if window_index < 0 or start_token < 0 or end_token <= start_token:
            raise ValueError("dense 窗口位置范围无效")
        identity = (chunk_id, window_index)
        if identity in seen_windows:
            raise ValueError(f"dense 窗口身份重复: {identity!r}")
        seen_windows.add(identity)
        repository.get_by_chunk_id(chunk_id)

    chunk_ids = tuple(sparse_metadata.get("position_to_chunk_id", ()))
    if not chunk_ids or len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("BM25 chunk_id 映射必须非空且唯一")
    if bm25_corpus_size is not None and int(bm25_corpus_size) != len(chunk_ids):
        raise ValueError("BM25 文档数与 chunk_id 映射数量不一致")
    for chunk_id in chunk_ids:
        repository.get_by_chunk_id(chunk_id)


def _resolve_device(requested):
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_semantic_retriever(
    *,
    repository,
    artifact_dir,
    embedding_model=DEFAULT_EMBEDDING_MODEL,
    reranker_model=DEFAULT_RERANKER_MODEL,
    device="auto",
    config=SemanticRetrievalConfig(),
):
    """显式加载本地索引和模型，完成校验后返回 SemanticRetriever。"""
    import faiss
    from sentence_transformers import CrossEncoder, SentenceTransformer

    directory = Path(artifact_dir)
    dense_index = faiss.read_index(str(directory / "law_dense.faiss"))
    dense_metadata = json.loads(
        (directory / "law_dense_meta.json").read_text(encoding="utf-8")
    )
    with (directory / "law_sparse.pkl").open("rb") as source:
        sparse_payload = pickle.load(source)

    actual_device = _resolve_device(device)
    encoder = SentenceTransformer(embedding_model, device=actual_device)
    reranker_model_instance = CrossEncoder(
        reranker_model,
        max_length=512,
        device=actual_device,
    )
    sparse_metadata = sparse_payload["metadata"]
    bm25 = sparse_payload["bm25"]
    validate_loaded_artifacts(
        repository=repository,
        dense_index=dense_index,
        dense_metadata=dense_metadata,
        sparse_metadata=sparse_metadata,
        embedding_dimension=encoder.get_sentence_embedding_dimension(),
        bm25_corpus_size=getattr(bm25, "corpus_size", None),
    )

    return SemanticRetriever(
        dense_searcher=DenseSearcher(
            index=dense_index,
            position_to_window=dense_metadata["position_to_window"],
            encoder=encoder,
        ),
        sparse_searcher=SparseSearcher(
            bm25=bm25,
            position_to_chunk_id=sparse_metadata["position_to_chunk_id"],
        ),
        article_repository=repository,
        reranker=WindowReranker(
            model=reranker_model_instance,
            tokenizer=reranker_model_instance.tokenizer,
            max_length=512,
            overlap_tokens=config.window_overlap_tokens,
            batch_size=config.batch_size,
        ),
        config=config,
    )


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_RERANKER_MODEL",
    "load_semantic_retriever",
    "validate_loaded_artifacts",
]
