"""外部请求分析协议及其严格解析逻辑。"""

import json
import re
from dataclasses import dataclass
from typing import Optional

from rag.core.contracts import BusinessRoute, LegalTaskType


EXTERNAL_ANALYSIS_SYSTEM_PROMPT = """你负责分析用户请求的业务路由，不回答法律问题。
只输出一个严格 JSON 对象，字段必须是 route、task_type、reason，不能输出 Markdown、解释或其他字段。
route 只能是 answer、clarify、general_chat。
当 route 为 answer 时，task_type 必须是 rule_lookup 或 case_application。
当 route 为 clarify 或 general_chat 时，task_type 必须为 null。
reason 必须是简短、稳定的小写 snake_case 原因码，不要写自然语言句子。
"""

EXTERNAL_ANALYSIS_SCHEMA = (
    '{"route":"answer | clarify | general_chat",'
    '"task_type":"rule_lookup | case_application | null",'
    '"reason":"stable_snake_case_reason_code"}'
)

_FIELDS = frozenset({"route", "task_type", "reason"})
_REASON_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class ExternalAnalysisProtocolError(ValueError):
    """外部请求分析结果不符合严格协议。"""


class _StrictJsonError(ValueError):
    """JSON 使用了协议禁止的写法。"""


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJsonError("JSON 对象包含重复字段")
        result[key] = value
    return result


def _reject_nonstandard_constant(value):
    raise _StrictJsonError(f"JSON 包含非标准常量: {value}")


@dataclass(frozen=True)
class ExternalRequestDecision:
    """经过协议校验的外部请求分析结果。"""

    route: BusinessRoute
    task_type: Optional[LegalTaskType]
    reason: str

    def __post_init__(self):
        if not isinstance(self.route, BusinessRoute):
            raise TypeError("route 必须是 BusinessRoute")
        if self.task_type is not None and not isinstance(
            self.task_type, LegalTaskType
        ):
            raise TypeError("task_type 必须是 LegalTaskType 或 None")
        if not isinstance(self.reason, str) or not _REASON_PATTERN.fullmatch(
            self.reason
        ):
            raise ValueError("reason 必须是小写 snake_case 原因码")
        if self.route is BusinessRoute.ANSWER and self.task_type is None:
            raise ValueError("answer 路由必须携带 task_type")
        if self.route is not BusinessRoute.ANSWER and self.task_type is not None:
            raise ValueError("非 answer 路由的 task_type 必须为 None")


def build_external_analysis_prompt(query):
    """构造外部请求分析所需的系统提示和用户消息。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串")
    return [
        {"role": "system", "content": EXTERNAL_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]


def parse_and_validate_external_analysis(raw_text):
    """严格解析外部模型返回的三字段 JSON 对象。"""
    if not isinstance(raw_text, str):
        raise TypeError("raw_text 必须是字符串")
    try:
        payload = json.loads(
            raw_text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (json.JSONDecodeError, _StrictJsonError) as exc:
        raise ExternalAnalysisProtocolError("模型输出不是严格 JSON") from exc

    if not isinstance(payload, dict) or frozenset(payload) != _FIELDS:
        raise ExternalAnalysisProtocolError(
            "顶层字段必须严格为 route、task_type、reason"
        )

    route_value = payload["route"]
    task_value = payload["task_type"]
    reason = payload["reason"]
    try:
        route = BusinessRoute(route_value)
        task_type = (
            None if task_value is None else LegalTaskType(task_value)
        )
        return ExternalRequestDecision(
            route=route,
            task_type=task_type,
            reason=reason,
        )
    except (TypeError, ValueError) as exc:
        raise ExternalAnalysisProtocolError("外部分析字段值不合法") from exc


def analyze_request(query, external_llm):
    """调用一次外部模型并返回分析结果，失败时统一降级为澄清。"""
    try:
        messages = build_external_analysis_prompt(query)
        raw_text = external_llm.generate(
            messages,
            temperature=0,
            max_tokens=128,
        )
        return parse_and_validate_external_analysis(raw_text)
    except Exception:
        return ExternalRequestDecision(
            route=BusinessRoute.CLARIFY,
            task_type=None,
            reason="external_analysis_failed",
        )


__all__ = [
    "EXTERNAL_ANALYSIS_SCHEMA",
    "EXTERNAL_ANALYSIS_SYSTEM_PROMPT",
    "ExternalAnalysisProtocolError",
    "ExternalRequestDecision",
    "analyze_request",
    "build_external_analysis_prompt",
    "parse_and_validate_external_analysis",
]
