"""法条级 jieba + BM25 稀疏检索。"""

import math
import pickle
import re
from pathlib import Path

from rag.retrieval.fusion import ScoredChunk
from rag.retrieval.text import format_article_heading_fields


_CITATION_RE = re.compile(r"(第\d+条(?:之[一二三四五六七八九十]+)?)")


def tokenize(text):
    """使用 jieba 分词，并把规范条号锁定为单个 token。"""
    import jieba

    tokens = []
    for index, part in enumerate(_CITATION_RE.split(text)):
        if not part:
            continue
        if index % 2 == 1:
            tokens.append(part)
        else:
            tokens.extend(token for token in jieba.cut(part) if token.strip())
    return tokens


def format_sparse_document(*, law_name, article_no, content, hierarchy):
    """构造不重复法名的 BM25 索引文本。"""
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content 必须是非空字符串")
    if not isinstance(hierarchy, dict):
        raise TypeError("hierarchy 必须是字典")
    parts = [format_article_heading_fields(law_name, article_no)]
    for key in ("编", "分编", "章", "节"):
        value = hierarchy.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)
    parts.append(content)
    return "\n".join(parts)


class SparseSearcher:
    """把 BM25 原始分数转换为正分法条候选。"""

    def __init__(self, *, bm25, position_to_chunk_id, tokenize=tokenize):
        if not callable(getattr(bm25, "get_scores", None)):
            raise TypeError("bm25 必须提供 get_scores")
        if not callable(tokenize):
            raise TypeError("tokenize 必须可调用")
        self._bm25 = bm25
        self._position_to_chunk_id = tuple(position_to_chunk_id)
        self._tokenize = tokenize

    def search(self, query, *, top_k):
        """只返回 BM25 分数大于零的候选。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k 必须是正整数")
        scores = tuple(self._bm25.get_scores(self._tokenize(query)))
        if len(scores) != len(self._position_to_chunk_id):
            raise RuntimeError("BM25 分数数量与 chunk_id 映射数量不一致")
        candidates = []
        for chunk_id, raw_score in zip(self._position_to_chunk_id, scores):
            score = float(raw_score)
            if not math.isfinite(score):
                raise RuntimeError("BM25 返回了非有限分数")
            if score > 0:
                candidates.append(ScoredChunk(chunk_id, score))
        candidates.sort(key=lambda item: (-item.score, item.chunk_id))
        return tuple(candidates[:top_k])


def build_sparse_index(documents, chunk_ids, *, tokenize_fn=tokenize):
    """构建 BM25 模型；运行时产物只保留位置到 chunk_id 的映射。"""
    from rank_bm25 import BM25Okapi

    document_values = tuple(documents)
    chunk_id_values = tuple(chunk_ids)
    if len(document_values) != len(chunk_id_values) or not document_values:
        raise ValueError("BM25 文档和 chunk_id 必须非空且等长")
    tokenized = [tokenize_fn(document) for document in document_values]
    return BM25Okapi(tokenized), {
        "tokenizer": "jieba",
        "article_count": len(chunk_id_values),
        "position_to_chunk_id": chunk_id_values,
    }


def save_sparse_index(bm25, metadata, artifact_dir):
    """保存 BM25 状态和最小位置映射，不复制法条正文。"""
    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "law_sparse.pkl").open("wb") as target:
        pickle.dump({"bm25": bm25, "metadata": metadata}, target)


__all__ = [
    "SparseSearcher",
    "build_sparse_index",
    "format_sparse_document",
    "save_sparse_index",
    "tokenize",
]
