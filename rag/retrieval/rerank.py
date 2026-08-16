"""对候选法条的所有可见窗口执行 Cross-Encoder 精排。"""

import math

from rag.knowledge import LegalArticle
from rag.retrieval.text import build_retrieval_windows


class WindowReranker:
    """以最佳窗口分数作为完整法条的 rerank 分数。"""

    def __init__(
        self,
        *,
        model,
        tokenizer,
        max_length=512,
        overlap_tokens=64,
        batch_size=32,
    ):
        if not callable(getattr(model, "predict", None)):
            raise TypeError("model 必须提供 predict")
        for name, value in (
            ("max_length", max_length),
            ("batch_size", batch_size),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if (
            not isinstance(overlap_tokens, int)
            or isinstance(overlap_tokens, bool)
            or overlap_tokens < 0
        ):
            raise ValueError("overlap_tokens 必须是非负整数")
        self._model = model
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._overlap_tokens = overlap_tokens
        self._batch_size = batch_size

    def score(self, query, articles):
        """按输入法条顺序返回逐法条最佳窗口分数。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        article_values = tuple(articles)
        if not article_values:
            return ()
        pairs = []
        owners = []
        for article_index, article in enumerate(article_values):
            if not isinstance(article, LegalArticle):
                raise TypeError("articles 必须由 LegalArticle 组成")
            windows = build_retrieval_windows(
                article,
                self._tokenizer,
                max_length=self._max_length,
                overlap_tokens=self._overlap_tokens,
                query=query,
            )
            for window in windows:
                pairs.append((query, window.text))
                owners.append(article_index)

        raw_scores = tuple(
            self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
        )
        if len(raw_scores) != len(pairs):
            raise RuntimeError("reranker 分数数量与窗口数量不一致")
        best_scores = [-math.inf] * len(article_values)
        for owner, raw_score in zip(owners, raw_scores):
            if isinstance(raw_score, (bool, str, bytes)):
                raise RuntimeError("reranker 返回了无效分数类型")
            score = float(raw_score)
            if not math.isfinite(score):
                raise RuntimeError("reranker 返回了非有限分数")
            best_scores[owner] = max(best_scores[owner], score)
        return tuple(best_scores)


__all__ = ["WindowReranker"]
