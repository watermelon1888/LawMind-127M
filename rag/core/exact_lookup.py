"""连接 query 引用提取与 knowledge 确定性查条。"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

from rag.knowledge import ArticleRepository, LegalArticle
from rag.query import QueryDecision, QueryRoute
from rag.query.exact_reference import (
    ExactReferenceStatus,
    extract_exact_references,
)


class ExactLookupStatus(str, Enum):
    """精确查条成功或需要用户澄清。"""

    FOUND = "found"
    CLARIFICATION_REQUIRED = "clarification_required"


@dataclass(frozen=True)
class ExactLookupResolution:
    """完整命中的一至三条法条，或不携带部分结果的澄清。"""

    status: ExactLookupStatus
    articles: Tuple[LegalArticle, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not isinstance(self.status, ExactLookupStatus):
            raise TypeError("status 必须是 ExactLookupStatus")
        articles = tuple(self.articles)
        object.__setattr__(self, "articles", articles)
        if any(not isinstance(article, LegalArticle) for article in articles):
            raise TypeError("articles 中的元素必须是 LegalArticle")
        if self.status is ExactLookupStatus.FOUND:
            if not 1 <= len(articles) <= 3:
                raise ValueError("查找成功时必须携带一至三条 LegalArticle")
        elif articles:
            raise ValueError("请求澄清时不能携带部分法条")


def resolve_exact_lookup(decision, repository: ArticleRepository):
    """执行已通过前置路由的一至三条确定性查找。"""
    if not isinstance(decision, QueryDecision):
        raise TypeError("decision 必须是 QueryDecision")
    if decision.route is not QueryRoute.EXACT_LOOKUP:
        raise ValueError("只有 exact_lookup 决策可以精确查条")

    parsed = extract_exact_references(decision.query)
    if parsed.status is ExactReferenceStatus.CLARIFICATION_REQUIRED:
        return ExactLookupResolution(
            ExactLookupStatus.CLARIFICATION_REQUIRED
        )

    articles = []
    for reference in parsed.references:
        article = repository.lookup(reference.law_name, reference.article_no)
        if article is None:
            return ExactLookupResolution(
                ExactLookupStatus.CLARIFICATION_REQUIRED
            )
        articles.append(article)
    return ExactLookupResolution(
        ExactLookupStatus.FOUND,
        tuple(articles),
    )


__all__ = [
    "ExactLookupResolution",
    "ExactLookupStatus",
    "resolve_exact_lookup",
]
