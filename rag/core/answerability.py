"""检索完成后的可回答性条件判断。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

from rag.core.contracts import BusinessRoute, LegalTaskType

if TYPE_CHECKING:
    from rag.answering.evidence import EvidencePackage


_GENERIC_CASE_PATTERNS = (
    re.compile(
        r"(?:这种情况|这种行为|这个行为|这件事|上述情况|该行为)"
        r"[^。！？!?]{0,12}"
        r"(?:怎么办|如何处理|是否合法|是否违法|承担什么责任|赔偿多少)"
    ),
    re.compile(
        r"^(?:发生纠纷|出了问题|有争议|遇到纠纷)"
        r"[^。！？!?]{0,12}"
        r"(?:怎么办|如何处理|是否合法|是否违法|承担什么责任|赔偿多少)"
    ),
)


@dataclass(frozen=True)
class AnswerabilityDecision:
    """检索后是否允许进入回答模型的确定性决策。"""

    route: BusinessRoute
    reason: str

    def __post_init__(self):
        if self.route not in {BusinessRoute.ANSWER, BusinessRoute.CLARIFY}:
            raise ValueError("可回答性决策只能使用 ANSWER 或 CLARIFY")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason 必须是非空字符串")


def _has_obvious_missing_case_facts(query: str) -> bool:
    return any(pattern.search(query) for pattern in _GENERIC_CASE_PATTERNS)


def decide_answerability(
    query: str,
    task_type: Optional[LegalTaskType],
    ranked_articles: Sequence[object],
    evidence_package: Optional[EvidencePackage],
) -> AnswerabilityDecision:
    """依据任务类型、检索候选和证据包做最小可解释判断。"""
    from rag.answering.evidence import EvidencePackage

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串")
    if task_type is not None and not isinstance(task_type, LegalTaskType):
        raise TypeError("task_type 必须是 LegalTaskType 或 None")
    try:
        ranked = tuple(ranked_articles)
    except TypeError as exc:
        raise TypeError("ranked_articles 必须是可迭代对象") from exc
    if evidence_package is not None and not isinstance(
        evidence_package, EvidencePackage
    ):
        raise TypeError("evidence_package 必须是 EvidencePackage 或 None")

    if task_type is None:
        return AnswerabilityDecision(
            BusinessRoute.CLARIFY,
            "missing_task_type",
        )
    if not ranked:
        return AnswerabilityDecision(
            BusinessRoute.CLARIFY,
            "no_retrieval_candidates",
        )
    if evidence_package is None or not evidence_package.evidence:
        return AnswerabilityDecision(
            BusinessRoute.CLARIFY,
            "no_evidence_package",
        )
    if task_type is LegalTaskType.CASE_APPLICATION and _has_obvious_missing_case_facts(
        query
    ):
        return AnswerabilityDecision(
            BusinessRoute.CLARIFY,
            "missing_key_facts",
        )
    return AnswerabilityDecision(BusinessRoute.ANSWER, "evidence_sufficient")


__all__ = ["AnswerabilityDecision", "decide_answerability"]
