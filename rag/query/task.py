"""法律请求的规则查询与案情适用分类。"""

import re
from dataclasses import dataclass
from typing import Optional

from rag.core.contracts import BusinessRoute, LegalTaskType
from rag.query.external_analysis import (
    ExternalRequestDecision,
    analyze_request,
)


_RULE_CUES = (
    "规定",
    "怎么规定",
    "如何规定",
    "规定是什么",
    "有哪些规定",
    "法律规定",
    "法律依据",
    "依据",
    "法条",
    "条文",
    "构成要件",
    "适用条件",
    "适用范围",
    "法律含义",
    "定义",
)
_CASE_ACTOR_CUES = (
    "我",
    "本人",
    "他",
    "她",
    "我们",
    "对方",
    "老板",
    "房东",
    "公司",
    "单位",
    "员工",
    "商家",
)
_CASE_FACT_CUES = (
    "行为",
    "签订",
    "发生",
    "支付",
    "收到",
    "扣除",
    "拖欠",
    "不给",
    "拒绝",
    "受伤",
    "损失",
    "事故",
    "纠纷",
    "涉嫌",
    "被",
)
_CASE_ACTION_CUES = (
    "怎么办",
    "怎么处理",
    "是否合法",
    "违法吗",
    "能否",
    "可以吗",
    "责任",
    "赔偿",
    "维权",
    "起诉",
)


@dataclass(frozen=True)
class LegalTaskDecision:
    """经过分类的法律任务及其来源。"""

    task_type: Optional[LegalTaskType]
    decision_source: str
    reason: Optional[str] = None
    route: Optional[BusinessRoute] = None

    def __post_init__(self):
        if self.task_type is not None and not isinstance(
            self.task_type, LegalTaskType
        ):
            raise TypeError("task_type 必须是 LegalTaskType 或 None")
        if self.decision_source not in {"deterministic", "external"}:
            raise ValueError("decision_source 必须是 deterministic 或 external")
        if self.reason is not None and (
            not isinstance(self.reason, str) or not self.reason.strip()
        ):
            raise ValueError("reason 必须是非空字符串")
        if self.route is not None and not isinstance(self.route, BusinessRoute):
            raise TypeError("route 必须是 BusinessRoute 或 None")
        if self.route is BusinessRoute.ANSWER and self.task_type is None:
            raise ValueError("answer 路由必须携带 task_type")
        if self.route is not BusinessRoute.ANSWER and self.route is not None:
            if self.task_type is not None:
                raise ValueError("非 answer 路由不能携带 task_type")


def _has_case_signal(query):
    has_actor = any(cue in query for cue in _CASE_ACTOR_CUES)
    has_fact = any(cue in query for cue in _CASE_FACT_CUES)
    has_action = any(cue in query for cue in _CASE_ACTION_CUES)
    return (has_actor and has_action) or (has_fact and has_action)


def _has_rule_signal(query):
    return any(cue in query for cue in _RULE_CUES)


def _from_external(decision):
    if not isinstance(decision, ExternalRequestDecision):
        raise TypeError("external_decision 必须是 ExternalRequestDecision 或 None")
    if decision.route is BusinessRoute.ANSWER:
        if decision.task_type is None:
            return LegalTaskDecision(
                None,
                "external",
                reason="external_analysis_failed",
                route=BusinessRoute.CLARIFY,
            )
        return LegalTaskDecision(
            decision.task_type,
            "external",
            reason=decision.reason,
            route=BusinessRoute.ANSWER,
        )
    return LegalTaskDecision(
        None,
        "external",
        reason=decision.reason,
        route=decision.route,
    )


def classify_legal_task(
    query,
    *,
    external_llm=None,
    external_decision=None,
):
    """分类法律任务，歧义时最多调用一次外部请求分析。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串")
    compact_query = re.sub(r"\s+", "", query)

    if _has_case_signal(compact_query):
        return LegalTaskDecision(
            LegalTaskType.CASE_APPLICATION,
            "deterministic",
            reason="case_facts_and_action",
            route=BusinessRoute.ANSWER,
        )
    if _has_rule_signal(compact_query):
        return LegalTaskDecision(
            LegalTaskType.RULE_LOOKUP,
            "deterministic",
            reason="legal_rule_cue",
            route=BusinessRoute.ANSWER,
        )

    if external_decision is not None:
        return _from_external(external_decision)
    if external_llm is not None:
        return _from_external(analyze_request(query, external_llm))
    return LegalTaskDecision(
        None,
        "deterministic",
        reason="task_type_ambiguous",
    )


__all__ = ["LegalTaskDecision", "classify_legal_task"]
