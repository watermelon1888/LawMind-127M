"""窗口级 BGE + FAISS 稠密检索。"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.knowledge import LegalArticle
from rag.retrieval.fusion import ScoredChunk
from rag.retrieval.text import build_retrieval_windows


QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


@dataclass(frozen=True)
class DenseHit(ScoredChunk):
    """一条唯一法条及其最佳 dense 窗口。"""

    window_index: int


class DenseSearcher:
    """把 FAISS 窗口命中折叠为按分数排序的唯一法条。"""

    def __init__(self, *, index, position_to_window, encoder):
        if not callable(getattr(index, "search", None)):
            raise TypeError("index 必须提供 search")
        if not callable(getattr(encoder, "encode", None)):
            raise TypeError("encoder 必须提供 encode")
        self._index = index
        self._position_to_window = tuple(position_to_window)
        self._encoder = encoder

    def search(self, query, *, top_k):
        """返回 top_k 条唯一法条；同法条只保留最高分窗口。"""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query 必须是非空字符串")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k 必须是正整数")
        total = int(self._index.ntotal)
        if total <= 0:
            return ()
        raw_k = min(total, max(top_k, top_k * 2))
        query_vector = self._encoder.encode(
            [QUERY_INSTRUCTION + query],
            normalize_embeddings=True,
        )

        best_by_chunk = {}
        while True:
            scores, positions = self._index.search(
                np.asarray(query_vector, dtype=np.float32), raw_k
            )
            best_by_chunk = self._collapse_hits(scores[0], positions[0])
            if len(best_by_chunk) >= top_k or raw_k == total:
                break
            raw_k = min(total, raw_k * 2)

        ordered = sorted(
            best_by_chunk.values(),
            key=lambda item: (-item.score, item.chunk_id),
        )
        return tuple(ordered[:top_k])

    def _collapse_hits(self, scores, positions):
        best_by_chunk = {}
        for raw_score, raw_position in zip(scores, positions):
            position = int(raw_position)
            if position < 0:
                continue
            if position >= len(self._position_to_window):
                raise RuntimeError("FAISS 位置没有对应的窗口元数据")
            metadata = self._position_to_window[position]
            try:
                chunk_id = metadata["chunk_id"]
                window_index = metadata["window_index"]
            except (KeyError, TypeError) as exc:
                raise RuntimeError("dense 窗口元数据格式无效") from exc
            score = float(raw_score)
            if not math.isfinite(score):
                raise RuntimeError("FAISS 返回了非有限分数")
            candidate = DenseHit(chunk_id, score, int(window_index))
            previous = best_by_chunk.get(chunk_id)
            if previous is None or (candidate.score, -candidate.window_index) > (
                previous.score,
                -previous.window_index,
            ):
                best_by_chunk[chunk_id] = candidate
        return best_by_chunk


def build_dense_index(
    articles,
    encoder,
    *,
    model_name,
    overlap_tokens=64,
    batch_size=32,
):
    """把完整法条展开为窗口向量，返回 FAISS 索引和最小元数据。"""
    import faiss

    tokenizer = encoder.tokenizer
    max_length = int(encoder.max_seq_length)
    article_values = tuple(articles)
    texts = []
    position_to_window = []
    for article in article_values:
        if not isinstance(article, LegalArticle):
            raise TypeError("articles 必须由 LegalArticle 组成")
        for window in build_retrieval_windows(
            article,
            tokenizer,
            max_length=max_length,
            overlap_tokens=overlap_tokens,
        ):
            texts.append(window.text)
            position_to_window.append(
                {
                    "chunk_id": window.chunk_id,
                    "window_index": window.window_index,
                    "start_token": window.start_token,
                    "end_token": window.end_token,
                }
            )
    vectors = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    metadata = {
        "embedding_model": model_name,
        "vector_dimension": int(vectors.shape[1]),
        "document_text": "law_name + canonical article_no + full content windows",
        "tokenizer_max_length": max_length,
        "window_overlap_tokens": overlap_tokens,
        "article_count": len(article_values),
        "window_count": len(position_to_window),
        "position_to_window": position_to_window,
    }
    return index, metadata


def save_dense_index(index, metadata, artifact_dir):
    """把 dense 索引和 JSON 元数据写入显式生成物目录。"""
    import faiss

    directory = Path(artifact_dir)
    directory.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(directory / "law_dense.faiss"))
    (directory / "law_dense_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


__all__ = [
    "DenseHit",
    "DenseSearcher",
    "QUERY_INSTRUCTION",
    "build_dense_index",
    "save_dense_index",
]
