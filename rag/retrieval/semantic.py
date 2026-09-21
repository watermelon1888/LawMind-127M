"""把 dense、BM25、RRF 和窗口重排封装为法条级候选。"""

import math
from dataclasses import dataclass

from rag.knowledge import ArticleRepository, LegalArticle
from rag.retrieval.fusion import ScoredChunk, rrf_fusion


class RetrievalIntegrityError(RuntimeError):
    """检索产物与 canonical 法条仓库之间的契约被破坏。"""


@dataclass(frozen=True)
class SemanticRetrievalConfig:
    """开发集参数评估后采用的语义检索运行时配置。"""

    dense_top_k: int = 30
    sparse_top_k: int = 30
    rrf_k: int = 4
    candidate_pool: int = 20
    top_k: int = 5
    batch_size: int = 32
    window_overlap_tokens: int = 64

    def __post_init__(self):
        for name in (
            "dense_top_k",
            "sparse_top_k",
            "rrf_k",
            "candidate_pool",
            "top_k",
            "batch_size",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if self.top_k > self.candidate_pool:
            raise ValueError("top_k 不能大于 candidate_pool")
        if (
            not isinstance(self.window_overlap_tokens, int)
            or isinstance(self.window_overlap_tokens, bool)
            or self.window_overlap_tokens < 0
        ):
            raise ValueError("window_overlap_tokens 必须是非负整数")


@dataclass(frozen=True)
class RankedArticle:
    """保留完整法条身份和两阶段法条级排序诊断的候选。"""

    article: LegalArticle
    rrf_rank: int
    rrf_score: float
    rerank_score: float

    def __post_init__(self):
        if not isinstance(self.article, LegalArticle):
            raise TypeError("article 必须是 LegalArticle")
        if (
            not isinstance(self.rrf_rank, int)
            or isinstance(self.rrf_rank, bool)
            or self.rrf_rank <= 0
        ):
            raise ValueError("rrf_rank 必须是正整数")
        for name in ("rrf_score", "rerank_score"):
            value = getattr(self, name)
            if isinstance(value, (bool, str, bytes)):
                raise ValueError(f"{name} 必须是有限数值")
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"{name} 必须是有限数值")
            if name == "rrf_score" and numeric <= 0:
                raise ValueError("rrf_score 必须大于零")
            object.__setattr__(self, name, numeric)


class SemanticRetriever:
    """对调用方隐藏双路召回、融合、补全和窗口重排。"""

    def __init__(
        self,
        *,
        dense_searcher,
        sparse_searcher,
        article_repository,
        reranker,
        config=SemanticRetrievalConfig(),
        reranker_model_name=None,
    ):
        for name, value in (
            ("dense_searcher", dense_searcher),
            ("sparse_searcher", sparse_searcher),
        ):
            if not callable(getattr(value, "search", None)):
                raise TypeError(f"{name} 必须提供 search")
        if not isinstance(article_repository, ArticleRepository):
            raise TypeError("article_repository 必须是 ArticleRepository")
        if not callable(getattr(reranker, "score", None)):
            raise TypeError("reranker 必须提供 score")
        if not isinstance(config, SemanticRetrievalConfig):
            raise TypeError("config 必须是 SemanticRetrievalConfig")
        if reranker_model_name is not None and (
            not isinstance(reranker_model_name, str) or not reranker_model_name.strip()
        ):
            raise ValueError("reranker_model_name 必须是非空字符串或 None")
        self._dense_searcher = dense_searcher
        self._sparse_searcher = sparse_searcher
        self._article_repository = article_repository
        self._reranker = reranker
        self._config = config
        self._reranker_model_name = reranker_model_name

    def audit_metadata(self):
        """返回本次检索链使用的稳定配置，供审计界面展示。"""
        return {
            "dense_top_k": self._config.dense_top_k,
            "bm25_top_k": self._config.sparse_top_k,
            "rrf_k": self._config.rrf_k,
            "candidate_pool": self._config.candidate_pool,
            "reranker_top_k": self._config.top_k,
            "reranker_model": (
                self._reranker_model_name or type(self._reranker).__name__
            ),
        }

    @staticmethod
    def _validate_query(query):
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")

    def retrieve_candidates(self, query):
        """返回单条 query 经 dense、BM25 和第一层 RRF 得到的候选。"""
        self._validate_query(query)
        dense_results = self._dense_searcher.search(
            query, top_k=self._config.dense_top_k
        )
        sparse_results = self._sparse_searcher.search(
            query, top_k=self._config.sparse_top_k
        )
        fused = rrf_fusion(
            (dense_results, sparse_results),
            k=self._config.rrf_k,
        )[: self._config.candidate_pool]
        return tuple(fused)

    def retrieve_candidates_many(self, retrieval_queries):
        """返回多 query 经两层等权 RRF 得到的统一候选池。"""
        queries = tuple(retrieval_queries)
        if not queries:
            raise ValueError("retrieval_queries 不能为空")
        if len(queries) > 6:
            raise ValueError("retrieval_queries 最多包含六条 query")
        for query in queries:
            self._validate_query(query)
        if len(set(queries)) != len(queries):
            raise ValueError("retrieval_queries 不能包含重复 query")

        first_level = tuple(self.retrieve_candidates(query) for query in queries)
        if len(first_level) == 1:
            return first_level[0]
        return tuple(
            rrf_fusion(
                tuple(
                    tuple(ScoredChunk(item.chunk_id, item.score) for item in results)
                    for results in first_level
                ),
                k=self._config.rrf_k,
            )[: self._config.candidate_pool]
        )

    def rerank_candidates(self, query, fused):
        """使用 query 对已经融合的法条候选统一执行 Cross-Encoder 重排。"""
        self._validate_query(query)
        fused = tuple(fused)
        if not fused:
            return ()

        articles = []
        for item in fused:
            try:
                article = self._article_repository.get_by_chunk_id(item.chunk_id)
            except KeyError as exc:
                raise RetrievalIntegrityError(
                    f"检索索引引用了仓库中不存在的 chunk_id: {item.chunk_id}"
                ) from exc
            articles.append(article)
        rerank_scores = tuple(self._reranker.score(query, tuple(articles)))
        if len(rerank_scores) != len(fused):
            raise RetrievalIntegrityError("reranker 分数数量与候选数量不一致")

        ranked = []
        for article, fused_item, raw_score in zip(articles, fused, rerank_scores):
            try:
                ranked.append(
                    RankedArticle(
                        article=article,
                        rrf_rank=fused_item.rank,
                        rrf_score=fused_item.score,
                        rerank_score=raw_score,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise RetrievalIntegrityError("reranker 返回了无效法条分数") from exc
        ranked.sort(
            key=lambda item: (
                -item.rerank_score,
                item.rrf_rank,
                item.article.chunk_id,
            )
        )
        return tuple(ranked[: self._config.top_k])

    def search(self, query):
        """返回完成两路召回和 Cross-Encoder 精排的 top-5 法条候选。"""
        return self.rerank_candidates(query, self.retrieve_candidates(query))

    def search_many(self, original_query, retrieval_queries):
        """按两层等权 RRF 融合多 query，并用原始 query 统一重排。"""
        self._validate_query(original_query)
        fused = self.retrieve_candidates_many(retrieval_queries)
        return self.rerank_candidates(original_query, fused)


__all__ = [
    "RankedArticle",
    "RetrievalIntegrityError",
    "SemanticRetrievalConfig",
    "SemanticRetriever",
]
