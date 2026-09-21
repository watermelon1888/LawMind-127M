"""加载并校验版本化子单元检索索引。"""

import json
import pickle
from pathlib import Path

from rag.knowledge import ArticleRepository, EvidenceUnitRepository
from rag.retrieval.evidence_units import (
    EvidenceUnitReranker,
    EvidenceUnitRetriever,
    UnitDenseSearcher,
    UnitSparseSearcher,
)
from rag.retrieval.loader import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    _resolve_device,
)
from rag.retrieval.semantic import SemanticRetrievalConfig


def validate_unit_artifacts(
    *,
    article_repository,
    unit_repository,
    dense_index,
    dense_metadata,
    sparse_metadata,
    embedding_dimension,
    bm25_corpus_size=None,
):
    """校验子索引版本、数量、维度与显式父子映射。"""
    if not isinstance(article_repository, ArticleRepository):
        raise TypeError("article_repository 必须是 ArticleRepository")
    if not isinstance(unit_repository, EvidenceUnitRepository):
        raise TypeError("unit_repository 必须是 EvidenceUnitRepository")
    expected_version = unit_repository.splitter_version
    for name, metadata in (
        ("dense", dense_metadata),
        ("sparse", sparse_metadata),
    ):
        if metadata.get("splitter_version") != expected_version:
            raise ValueError(f"{name} 索引与 sidecar 的 splitter_version 不一致")
        if int(metadata.get("unit_count", -1)) != len(unit_repository):
            raise ValueError(f"{name} 索引与 sidecar 的 unit_count 不一致")

    windows = tuple(dense_metadata.get("position_to_window", ()))
    if not windows or int(dense_index.ntotal) != len(windows):
        raise ValueError("FAISS 向量数必须与子单元窗口元数据等长且非空")
    if int(dense_index.d) != int(embedding_dimension):
        raise ValueError("FAISS 向量维度与 embedding 模型不一致")
    seen_windows = set()
    covered_unit_ids = set()
    for item in windows:
        try:
            unit_id = item["unit_id"]
            parent_chunk_id = item["parent_chunk_id"]
            window_index = item["window_index"]
            start_token = item["start_token"]
            end_token = item["end_token"]
        except (KeyError, TypeError) as exc:
            raise ValueError("dense 子单元窗口元数据字段不完整") from exc
        unit = unit_repository.get_by_unit_id(unit_id)
        if unit.parent_chunk_id != parent_chunk_id:
            raise ValueError("dense 窗口的显式父子映射不一致")
        article_repository.get_by_chunk_id(parent_chunk_id)
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (window_index, start_token, end_token)
        ) or window_index < 0 or start_token < 0 or end_token <= start_token:
            raise ValueError("dense 子单元窗口位置无效")
        identity = (unit_id, window_index)
        if identity in seen_windows:
            raise ValueError("dense 子单元窗口身份重复")
        seen_windows.add(identity)
        covered_unit_ids.add(unit_id)
    if len(covered_unit_ids) != len(unit_repository):
        raise ValueError("dense 索引没有覆盖全部 evidence unit")

    sparse_ids = tuple(sparse_metadata.get("position_to_unit_id", ()))
    if len(sparse_ids) != len(unit_repository) or len(set(sparse_ids)) != len(
        sparse_ids
    ):
        raise ValueError("BM25 unit_id 映射必须全量且唯一")
    if bm25_corpus_size is not None and int(bm25_corpus_size) != len(sparse_ids):
        raise ValueError("BM25 文档数与 unit_id 映射数量不一致")
    if set(sparse_ids) != {unit.unit_id for unit in unit_repository.iter_units()}:
        raise ValueError("BM25 unit_id 集合与 sidecar 不一致")


def load_evidence_unit_retriever(
    *,
    article_repository,
    unit_repository,
    artifact_dir,
    embedding_model=DEFAULT_EMBEDDING_MODEL,
    reranker_model=DEFAULT_RERANKER_MODEL,
    device="auto",
    config=SemanticRetrievalConfig(),
):
    """加载独立子索引和模型，返回父级去重的子单元检索器。"""
    import faiss
    from sentence_transformers import CrossEncoder, SentenceTransformer

    directory = Path(artifact_dir)
    dense_index = faiss.read_index(str(directory / "evidence_unit_dense.faiss"))
    dense_metadata = json.loads(
        (directory / "evidence_unit_dense_meta.json").read_text(encoding="utf-8")
    )
    with (directory / "evidence_unit_sparse.pkl").open("rb") as source:
        sparse_payload = pickle.load(source)
    actual_device = _resolve_device(device)
    encoder = SentenceTransformer(embedding_model, device=actual_device)
    cross_encoder = CrossEncoder(
        reranker_model,
        max_length=512,
        device=actual_device,
    )
    bm25 = sparse_payload["bm25"]
    sparse_metadata = sparse_payload["metadata"]
    validate_unit_artifacts(
        article_repository=article_repository,
        unit_repository=unit_repository,
        dense_index=dense_index,
        dense_metadata=dense_metadata,
        sparse_metadata=sparse_metadata,
        embedding_dimension=encoder.get_sentence_embedding_dimension(),
        bm25_corpus_size=getattr(bm25, "corpus_size", None),
    )
    return EvidenceUnitRetriever(
        dense_searcher=UnitDenseSearcher(
            index=dense_index,
            position_to_window=dense_metadata["position_to_window"],
            encoder=encoder,
        ),
        sparse_searcher=UnitSparseSearcher(
            bm25=bm25,
            position_to_unit_id=sparse_metadata["position_to_unit_id"],
        ),
        article_repository=article_repository,
        unit_repository=unit_repository,
        reranker=EvidenceUnitReranker(
            model=cross_encoder,
            tokenizer=cross_encoder.tokenizer,
            article_repository=article_repository,
            max_length=512,
            overlap_tokens=config.window_overlap_tokens,
            batch_size=config.batch_size,
        ),
        config=config,
        reranker_model_name=reranker_model,
    )


__all__ = ["load_evidence_unit_retriever", "validate_unit_artifacts"]
