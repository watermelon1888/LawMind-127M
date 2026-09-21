"""以原文子单元召回、按父法条聚合并执行子单元精排。"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import pickle

import numpy as np

from rag.knowledge import (
    ArticleRepository,
    EvidenceUnit,
    EvidenceUnitRepository,
    LegalArticle,
)
from rag.retrieval.dense import QUERY_INSTRUCTION
from rag.retrieval.semantic import RetrievalIntegrityError, SemanticRetrievalConfig
from rag.retrieval.sparse import build_sparse_index, format_sparse_document, tokenize
from rag.retrieval.text import build_retrieval_windows, format_article_heading


MAX_UNITS_PER_PARENT = 3
DENSE_ENCODE_BUFFER_SIZE = 2048


def _finite_score(name, value):
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} 必须是有限数值")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} 必须是有限数值")
    return numeric


@dataclass(frozen=True)
class ScoredUnit:
    """单路子索引返回的原文单元命中。"""

    unit_id: str
    score: float

    def __post_init__(self):
        if not isinstance(self.unit_id, str) or not self.unit_id.strip():
            raise ValueError("unit_id 必须是非空字符串")
        object.__setattr__(self, "score", _finite_score("score", self.score))


@dataclass(frozen=True)
class FusedUnit:
    """子单元级 RRF 结果。"""

    unit_id: str
    rank: int
    score: float


@dataclass(frozen=True)
class ParentCandidate:
    """子单元 RRF 聚合得到的唯一父法条候选。"""

    article: LegalArticle
    units: tuple[EvidenceUnit, ...]
    rank: int
    score: float

    def __post_init__(self):
        if not isinstance(self.article, LegalArticle):
            raise TypeError("article 必须是 LegalArticle")
        units = tuple(self.units)
        object.__setattr__(self, "units", units)
        if not units or any(not isinstance(unit, EvidenceUnit) for unit in units):
            raise TypeError("units 必须包含 EvidenceUnit")
        if any(unit.parent_chunk_id != self.article.chunk_id for unit in units):
            raise ValueError("候选子单元与父法条不匹配")
        if len({unit.unit_id for unit in units}) != len(units):
            raise ValueError("父候选不能包含重复 unit_id")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank <= 0:
            raise ValueError("rank 必须是正整数")
        score = _finite_score("score", self.score)
        if score <= 0:
            raise ValueError("score 必须大于零")
        object.__setattr__(self, "score", score)

    @property
    def chunk_id(self):
        return self.article.chunk_id


@dataclass(frozen=True)
class RankedParentEvidence:
    """以最佳相关子单元分数排序的父法条。"""

    article: LegalArticle
    units: tuple[EvidenceUnit, ...]
    unit_scores: tuple[float, ...]
    rrf_rank: int
    rrf_score: float
    rerank_score: float

    def __post_init__(self):
        units = tuple(self.units)
        scores = tuple(_finite_score("unit_score", item) for item in self.unit_scores)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "unit_scores", scores)
        if not isinstance(self.article, LegalArticle):
            raise TypeError("article 必须是 LegalArticle")
        if not units or len(units) != len(scores):
            raise ValueError("units 与 unit_scores 必须非空且等长")
        if any(unit.parent_chunk_id != self.article.chunk_id for unit in units):
            raise ValueError("精排子单元与父法条不匹配")
        if not isinstance(self.rrf_rank, int) or self.rrf_rank <= 0:
            raise ValueError("rrf_rank 必须是正整数")
        object.__setattr__(self, "rrf_score", _finite_score("rrf_score", self.rrf_score))
        object.__setattr__(
            self, "rerank_score", _finite_score("rerank_score", self.rerank_score)
        )

    @property
    def chunk_id(self):
        return self.article.chunk_id


def rrf_unit_fusion(result_lists, *, k=4):
    """等权融合多路 unit_id 排名，不把 unit_id 冒充 chunk_id。"""
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k 必须是正整数")
    scores = {}
    for results in result_lists:
        seen = set()
        for rank, item in enumerate(tuple(results), start=1):
            if not isinstance(item, ScoredUnit):
                raise TypeError("每路结果必须由 ScoredUnit 组成")
            if item.unit_id in seen:
                raise ValueError("单路结果不能包含重复 unit_id")
            seen.add(item.unit_id)
            scores[item.unit_id] = scores.get(item.unit_id, 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return tuple(
        FusedUnit(unit_id, rank, score)
        for rank, (unit_id, score) in enumerate(ordered, start=1)
    )


class UnitDenseSearcher:
    """搜索 FAISS 子单元窗口，并折叠到唯一 unit_id。"""

    def __init__(self, *, index, position_to_window, encoder):
        if not callable(getattr(index, "search", None)):
            raise TypeError("index 必须提供 search")
        if not callable(getattr(encoder, "encode", None)):
            raise TypeError("encoder 必须提供 encode")
        self._index = index
        self._position_to_window = tuple(position_to_window)
        self._encoder = encoder

    def search(self, query, *, top_k):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k 必须是正整数")
        total = int(self._index.ntotal)
        if total <= 0:
            return ()
        raw_k = min(total, max(top_k, top_k * 2))
        vector = self._encoder.encode(
            [QUERY_INSTRUCTION + query], normalize_embeddings=True
        )
        best = {}
        while True:
            scores, positions = self._index.search(
                np.asarray(vector, dtype=np.float32), raw_k
            )
            for raw_score, raw_position in zip(scores[0], positions[0]):
                position = int(raw_position)
                if position < 0:
                    continue
                try:
                    unit_id = self._position_to_window[position]["unit_id"]
                except (IndexError, KeyError, TypeError) as exc:
                    raise RetrievalIntegrityError("FAISS 位置缺少子单元元数据") from exc
                score = _finite_score("dense score", raw_score)
                previous = best.get(unit_id)
                if previous is None or score > previous:
                    best[unit_id] = score
            if len(best) >= top_k or raw_k == total:
                break
            raw_k = min(total, raw_k * 2)
        ordered = sorted(best.items(), key=lambda item: (-item[1], item[0]))
        return tuple(ScoredUnit(unit_id, score) for unit_id, score in ordered[:top_k])


class UnitSparseSearcher:
    """搜索子单元 BM25，并返回唯一 unit_id。"""

    def __init__(self, *, bm25, position_to_unit_id, tokenize_fn=tokenize):
        if not callable(getattr(bm25, "get_scores", None)):
            raise TypeError("bm25 必须提供 get_scores")
        self._bm25 = bm25
        self._position_to_unit_id = tuple(position_to_unit_id)
        self._tokenize = tokenize_fn

    def search(self, query, *, top_k):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        scores = tuple(self._bm25.get_scores(self._tokenize(query)))
        if len(scores) != len(self._position_to_unit_id):
            raise RetrievalIntegrityError("BM25 分数数量与 unit_id 映射不一致")
        candidates = [
            ScoredUnit(unit_id, score)
            for unit_id, raw_score in zip(self._position_to_unit_id, scores)
            for score in (_finite_score("BM25 score", raw_score),)
            if score > 0
        ]
        candidates.sort(key=lambda item: (-item.score, item.unit_id))
        return tuple(candidates[:top_k])


class EvidenceUnitReranker:
    """对原文子单元的完整可见窗口执行 Cross-Encoder 评分。"""

    def __init__(
        self,
        *,
        model,
        tokenizer,
        article_repository,
        max_length=512,
        overlap_tokens=64,
        batch_size=32,
    ):
        if not callable(getattr(model, "predict", None)):
            raise TypeError("model 必须提供 predict")
        if not isinstance(article_repository, ArticleRepository):
            raise TypeError("article_repository 必须是 ArticleRepository")
        self._model = model
        self._tokenizer = tokenizer
        self._article_repository = article_repository
        self._max_length = max_length
        self._overlap_tokens = overlap_tokens
        self._batch_size = batch_size

    def score(self, query, units):
        unit_values = tuple(units)
        if not unit_values:
            return ()
        pairs = []
        owners = []
        for owner, unit in enumerate(unit_values):
            if not isinstance(unit, EvidenceUnit):
                raise TypeError("units 必须由 EvidenceUnit 组成")
            parent = self._article_repository.get_by_chunk_id(unit.parent_chunk_id)
            projected = LegalArticle(
                chunk_id=unit.unit_id,
                law_name=parent.law_name,
                article_no=parent.article_no,
                content=unit.text,
                source_type=parent.source_type,
            )
            for window in build_retrieval_windows(
                projected,
                self._tokenizer,
                max_length=self._max_length,
                overlap_tokens=self._overlap_tokens,
                query=query,
            ):
                pairs.append((query, window.text))
                owners.append(owner)
        scores = tuple(
            self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
        )
        if len(scores) != len(pairs):
            raise RetrievalIntegrityError("子单元 reranker 分数数量无效")
        best = [-math.inf] * len(unit_values)
        for owner, score in zip(owners, scores):
            best[owner] = max(best[owner], _finite_score("rerank score", score))
        return tuple(best)


class EvidenceUnitRetriever:
    """对子单元召回后按父法条去重，并以最佳子单元执行精排。"""

    def __init__(
        self,
        *,
        dense_searcher,
        sparse_searcher,
        article_repository,
        unit_repository,
        reranker,
        config=SemanticRetrievalConfig(),
        reranker_model_name=None,
    ):
        if not isinstance(article_repository, ArticleRepository):
            raise TypeError("article_repository 必须是 ArticleRepository")
        if not isinstance(unit_repository, EvidenceUnitRepository):
            raise TypeError("unit_repository 必须是 EvidenceUnitRepository")
        for searcher in (dense_searcher, sparse_searcher):
            if not callable(getattr(searcher, "search", None)):
                raise TypeError("子索引搜索器必须提供 search")
        if not callable(getattr(reranker, "score", None)):
            raise TypeError("reranker 必须提供 score")
        self._dense_searcher = dense_searcher
        self._sparse_searcher = sparse_searcher
        self._articles = article_repository
        self._units = unit_repository
        self._reranker = reranker
        self._config = config
        self._reranker_model_name = reranker_model_name

    def audit_metadata(self):
        return {
            "retrieval_granularity": "evidence_unit",
            "splitter_version": self._units.splitter_version,
            "dense_top_k": self._config.dense_top_k,
            "bm25_top_k": self._config.sparse_top_k,
            "rrf_k": self._config.rrf_k,
            "candidate_pool": self._config.candidate_pool,
            "reranker_top_k": self._config.top_k,
            "reranker_model": self._reranker_model_name
            or type(self._reranker).__name__,
            "max_units_per_parent": MAX_UNITS_PER_PARENT,
        }

    @property
    def unit_scorer(self):
        """返回与子单元精排一致的 scorer，供证据 bundle 复用。"""
        return self._reranker

    def _parents_from_unit_results(self, result_lists):
        fused = rrf_unit_fusion(result_lists, k=self._config.rrf_k)
        grouped = {}
        for item in fused:
            try:
                unit = self._units.get_by_unit_id(item.unit_id)
                article = self._articles.get_by_chunk_id(unit.parent_chunk_id)
            except KeyError as exc:
                raise RetrievalIntegrityError("子索引包含未知父子身份") from exc
            state = grouped.setdefault(
                article.chunk_id,
                {"article": article, "rank": item.rank, "score": item.score, "units": []},
            )
            if len(state["units"]) < MAX_UNITS_PER_PARENT:
                state["units"].append(unit)
        ordered = sorted(
            grouped.values(),
            key=lambda state: (state["rank"], -state["score"], state["article"].chunk_id),
        )[: self._config.candidate_pool]
        return tuple(
            ParentCandidate(
                article=state["article"],
                units=tuple(state["units"]),
                rank=index,
                score=state["score"],
            )
            for index, state in enumerate(ordered, start=1)
        )

    def retrieve_candidates(self, query):
        dense = self._dense_searcher.search(query, top_k=self._config.dense_top_k)
        sparse = self._sparse_searcher.search(query, top_k=self._config.sparse_top_k)
        return self._parents_from_unit_results((dense, sparse))

    def retrieve_candidates_many(self, retrieval_queries):
        queries = tuple(retrieval_queries)
        if not queries or len(queries) > 6 or len(set(queries)) != len(queries):
            raise ValueError("retrieval_queries 必须包含一至六条唯一 query")
        legs = tuple(self.retrieve_candidates(query) for query in queries)
        if len(legs) == 1:
            return legs[0]
        parent_scores = {}
        parent_units = {}
        parent_articles = {}
        for leg in legs:
            for rank, candidate in enumerate(leg, start=1):
                chunk_id = candidate.chunk_id
                parent_scores[chunk_id] = parent_scores.get(chunk_id, 0.0) + 1.0 / (
                    self._config.rrf_k + rank
                )
                parent_articles[chunk_id] = candidate.article
                unit_map = parent_units.setdefault(chunk_id, {})
                for unit in candidate.units:
                    unit_map.setdefault(unit.unit_id, unit)
        ordered = sorted(parent_scores, key=lambda item: (-parent_scores[item], item))[
            : self._config.candidate_pool
        ]
        return tuple(
            ParentCandidate(
                article=parent_articles[chunk_id],
                units=tuple(parent_units[chunk_id].values())[:MAX_UNITS_PER_PARENT],
                rank=rank,
                score=parent_scores[chunk_id],
            )
            for rank, chunk_id in enumerate(ordered, start=1)
        )

    def rerank_candidates(self, query, candidates):
        candidates = tuple(candidates)
        if not candidates:
            return ()
        all_units = tuple(unit for candidate in candidates for unit in candidate.units)
        scores = tuple(self._reranker.score(query, all_units))
        if len(scores) != len(all_units):
            raise RetrievalIntegrityError("子单元精排分数数量无效")
        ranked = []
        offset = 0
        for candidate in candidates:
            unit_scores = scores[offset : offset + len(candidate.units)]
            offset += len(candidate.units)
            paired = sorted(
                zip(candidate.units, unit_scores),
                key=lambda item: (-item[1], item[0].start_char),
            )
            ranked.append(
                RankedParentEvidence(
                    article=candidate.article,
                    units=tuple(unit for unit, _ in paired),
                    unit_scores=tuple(score for _, score in paired),
                    rrf_rank=candidate.rank,
                    rrf_score=candidate.score,
                    rerank_score=paired[0][1],
                )
            )
        ranked.sort(
            key=lambda item: (-item.rerank_score, item.rrf_rank, item.chunk_id)
        )
        return tuple(ranked[: self._config.top_k])

    def search(self, query):
        return self.rerank_candidates(query, self.retrieve_candidates(query))

    def search_many(self, original_query, retrieval_queries):
        return self.rerank_candidates(
            original_query,
            self.retrieve_candidates_many(retrieval_queries),
        )


def format_unit_document(unit, article):
    """构造带父法条标题但只含子单元原文的检索文档。"""
    if not isinstance(unit, EvidenceUnit) or not isinstance(article, LegalArticle):
        raise TypeError("unit 和 article 类型无效")
    if unit.parent_chunk_id != article.chunk_id:
        raise ValueError("unit 与 article 不属于同一父法条")
    return f"{format_article_heading(article)}\n{unit.text}"


def build_unit_sparse_index(unit_repository, article_repository):
    documents = []
    unit_ids = []
    for unit in unit_repository.iter_units():
        article = article_repository.get_by_chunk_id(unit.parent_chunk_id)
        documents.append(
            format_sparse_document(
                law_name=article.law_name,
                article_no=article.article_no,
                content=unit.text,
                hierarchy={},
            )
        )
        unit_ids.append(unit.unit_id)
    bm25, metadata = build_sparse_index(documents, unit_ids)
    position_to_unit_id = metadata.pop("position_to_chunk_id")
    metadata = {
        **metadata,
        "splitter_version": unit_repository.splitter_version,
        "unit_count": len(unit_ids),
        "position_to_unit_id": position_to_unit_id,
    }
    return bm25, metadata


def build_unit_dense_index(
    unit_repository,
    article_repository,
    encoder,
    *,
    model_name,
    overlap_tokens=64,
    batch_size=32,
):
    """把子单元展开为 BGE 可见窗口并构建独立 FAISS 索引。"""
    import faiss

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")
    position_to_window = []
    tokenizer = encoder.tokenizer
    max_length = int(encoder.max_seq_length)
    vector_dimension = int(encoder.get_sentence_embedding_dimension())
    if vector_dimension <= 0:
        raise ValueError("embedding 向量维度必须是正整数")
    index = faiss.IndexFlatIP(vector_dimension)
    pending_texts = []

    def flush_pending():
        if not pending_texts:
            return
        vectors = np.asarray(
            encoder.encode(
                tuple(pending_texts),
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )
        if vectors.ndim != 2 or vectors.shape != (
            len(pending_texts),
            vector_dimension,
        ):
            raise RetrievalIntegrityError("子单元 embedding 形状无效")
        index.add(vectors)
        pending_texts.clear()

    for unit in unit_repository.iter_units():
        article = article_repository.get_by_chunk_id(unit.parent_chunk_id)
        projected = LegalArticle(
            chunk_id=unit.unit_id,
            law_name=article.law_name,
            article_no=article.article_no,
            content=unit.text,
            source_type=article.source_type,
        )
        for window in build_retrieval_windows(
            projected,
            tokenizer,
            max_length=max_length,
            overlap_tokens=overlap_tokens,
        ):
            pending_texts.append(window.text)
            position_to_window.append(
                {
                    "unit_id": unit.unit_id,
                    "parent_chunk_id": unit.parent_chunk_id,
                    "window_index": window.window_index,
                    "start_token": window.start_token,
                    "end_token": window.end_token,
                }
            )
            if len(pending_texts) >= max(batch_size, DENSE_ENCODE_BUFFER_SIZE):
                flush_pending()
    flush_pending()
    if not position_to_window:
        raise ValueError("子单元仓库不能为空")
    return index, {
        "embedding_model": model_name,
        "vector_dimension": vector_dimension,
        "splitter_version": unit_repository.splitter_version,
        "unit_count": len(unit_repository),
        "window_count": len(position_to_window),
        "position_to_window": position_to_window,
    }


def save_unit_indexes(*, bm25=None, sparse_metadata=None, dense_index=None, dense_metadata=None, artifact_dir):
    """保存调用方明确构建的子单元索引，不覆盖父法条 artifacts。"""
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if bm25 is not None:
        with (directory / "evidence_unit_sparse.pkl").open("wb") as target:
            pickle.dump({"bm25": bm25, "metadata": sparse_metadata}, target)
    if dense_index is not None:
        import faiss

        faiss.write_index(dense_index, str(directory / "evidence_unit_dense.faiss"))
        (directory / "evidence_unit_dense_meta.json").write_text(
            json.dumps(dense_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


__all__ = [
    "EvidenceUnitReranker",
    "EvidenceUnitRetriever",
    "FusedUnit",
    "MAX_UNITS_PER_PARENT",
    "ParentCandidate",
    "RankedParentEvidence",
    "ScoredUnit",
    "UnitDenseSearcher",
    "UnitSparseSearcher",
    "build_unit_dense_index",
    "build_unit_sparse_index",
    "format_unit_document",
    "rrf_unit_fusion",
    "save_unit_indexes",
]
