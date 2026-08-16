"""把有序完整法条构造成可计数、可审计的模型证据包。"""

import json
from dataclasses import dataclass, field
from typing import Callable, Sequence, Tuple

from rag.core.contracts import Evidence
from rag.knowledge import LegalArticle


RAG_MAX_OUTPUT_TOKENS = 150
MAX_EVIDENCE_ITEMS = 5


class EvidencePackagingError(RuntimeError):
    """证据包无法在完整性和预算约束下安全建立。"""


def _require_non_blank(name, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")


def _is_positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True)
class EvidencePackage:
    """模型请求对应的原始问题和有序 canonical 证据。"""

    query: str
    evidence: Tuple[Evidence, ...] = field(default_factory=tuple)

    def __post_init__(self):
        _require_non_blank("query", self.query)
        evidence = tuple(self.evidence)
        object.__setattr__(self, "evidence", evidence)
        if not evidence:
            raise ValueError("EvidencePackage 必须包含至少一条 Evidence")
        if any(not isinstance(item, Evidence) for item in evidence):
            raise TypeError("evidence 中的元素必须是 Evidence")
        identities = tuple((item.law_name, item.article_no) for item in evidence)
        if len(set(identities)) != len(identities):
            raise ValueError("EvidencePackage 不能包含重复法条")

    def to_model_json(self):
        """生成字段顺序稳定的紧凑 user JSON。"""
        payload = {
            "query": self.query,
            "evidence": [
                {
                    "evidence_id": f"E{index}",
                    "law_name": item.law_name,
                    "article_no": item.article_no,
                    "excerpts": [item.content],
                }
                for index, item in enumerate(self.evidence, start=1)
            ],
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class EvidencePackager:
    """按完整 prompt 预算选择最多五条候选的最大有序前缀。"""

    def __init__(
        self,
        *,
        context_limit: int,
        max_output_tokens: int = RAG_MAX_OUTPUT_TOKENS,
        count_prompt_tokens: Callable[[EvidencePackage], int],
    ):
        if not _is_positive_int(context_limit):
            raise ValueError("context_limit 必须是正整数")
        if not _is_positive_int(max_output_tokens):
            raise ValueError("max_output_tokens 必须是正整数")
        if max_output_tokens >= context_limit:
            raise ValueError("max_output_tokens 必须小于 context_limit")
        if not callable(count_prompt_tokens):
            raise TypeError("count_prompt_tokens 必须可调用")
        self._context_limit = context_limit
        self._max_output_tokens = max_output_tokens
        self._count_prompt_tokens = count_prompt_tokens

    @property
    def max_output_tokens(self):
        return self._max_output_tokens

    @property
    def context_limit(self):
        return self._context_limit

    def build(
        self, query: str, articles: Sequence[LegalArticle]
    ) -> tuple[EvidencePackage, int]:
        """构造最大完整有序前缀，并返回证据包与 prompt token 数。"""
        _require_non_blank("query", query)
        try:
            candidates = tuple(articles)[:MAX_EVIDENCE_ITEMS]
        except TypeError as exc:
            raise TypeError("articles 必须是 LegalArticle 序列") from exc
        if not candidates:
            raise ValueError("EvidencePackager 不接收空候选")

        seen_chunk_ids = set()
        selected = []
        package = None
        used_tokens = 0
        for article in candidates:
            if not isinstance(article, LegalArticle):
                raise TypeError("articles 中的元素必须是 LegalArticle")
            if article.chunk_id in seen_chunk_ids:
                raise EvidencePackagingError(
                    f"候选中存在重复 chunk_id: {article.chunk_id}"
                )
            seen_chunk_ids.add(article.chunk_id)
            selected.append(
                Evidence(
                    law_name=article.law_name,
                    article_no=article.article_no,
                    content=article.content,
                )
            )
            tentative = EvidencePackage(query=query, evidence=tuple(selected))
            measured = self._count_prompt_tokens(tentative)
            if not _is_positive_int(measured):
                raise EvidencePackagingError(
                    "count_prompt_tokens 必须返回正整数"
                )
            if measured + self._max_output_tokens > self._context_limit:
                selected.pop()
                if package is None:
                    raise EvidencePackagingError("首候选超过上下文预算")
                break
            package = tentative
            used_tokens = measured

        if package is None:
            raise EvidencePackagingError("未能建立非空证据包")
        return package, used_tokens


__all__ = [
    "EvidencePackage",
    "EvidencePackager",
    "EvidencePackagingError",
    "MAX_EVIDENCE_ITEMS",
    "RAG_MAX_OUTPUT_TOKENS",
]
