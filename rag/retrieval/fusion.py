"""在法条级候选上执行等权 Reciprocal Rank Fusion。"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoredChunk:
    """单路检索返回的一条法条及其内部原始分数。"""

    chunk_id: str
    score: float

    def __post_init__(self):
        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise ValueError("chunk_id 必须是非空字符串")
        if isinstance(self.score, (bool, str, bytes)):
            raise ValueError("score 必须是有限数值")
        score = float(self.score)
        if not math.isfinite(score):
            raise ValueError("score 必须是有限数值")
        object.__setattr__(self, "score", score)


@dataclass(frozen=True)
class FusedChunk:
    """按完整精度 RRF 分数排序后的法条候选。"""

    chunk_id: str
    rank: int
    score: float


def rrf_fusion(result_lists, *, k=60):
    """等权融合多路法条排名，并以 chunk_id 稳定处理同分。"""
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError("k 必须是正整数")
    scores = {}
    for results in result_lists:
        seen = set()
        for rank, item in enumerate(results, start=1):
            if not isinstance(item, ScoredChunk):
                raise TypeError("每路结果必须由 ScoredChunk 组成")
            if item.chunk_id in seen:
                raise ValueError(f"单路结果包含重复 chunk_id: {item.chunk_id}")
            seen.add(item.chunk_id)
            scores[item.chunk_id] = scores.get(item.chunk_id, 0.0) + 1.0 / (k + rank)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return tuple(
        FusedChunk(chunk_id=chunk_id, rank=rank, score=score)
        for rank, (chunk_id, score) in enumerate(ordered, start=1)
    )


__all__ = ["FusedChunk", "ScoredChunk", "rrf_fusion"]
