"""检索前判断法律问题是否缺少决定性信息。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from rag.external import generate_json


QUERY_ASSESSMENT_SYSTEM_PROMPT = """你是法律 RAG 的检索前问题判断器，只判断用户问题是否必须先澄清，不回答法律问题，也不生成检索词。
选择 answer：问题可以给出一般性或条件式回答，且不需要虚构事实。事实不够详细、存在多种情形、答案可能不够全面，都不是澄清理由。“不小心打了老板违法犯罪吗”“试用期最多多久”应选择 answer。
选择 clarify：问题缺少会决定适用制度的核心对象或范围，任何直接答案都只能任意猜测某个具体制度。例如“申请材料需要提交几份”未说明申请事项，应选择 clarify，并只追问最重要的一个缺失信息。
只输出严格 JSON 对象，字段必须是 decision 和 clarification，不得输出 Markdown、解释或其他字段。
decision 只能是 answer 或 clarify。answer 时 clarification 必须为 null。clarify 时 clarification 必须是一个简短问题，只能有一句且恰好一个问号。
JSON 示例：{"decision":"answer","clarification":null} 或 {"decision":"clarify","clarification":"请说明具体申请事项？"}"""

_FIELDS = frozenset({"decision", "clarification"})
_QUESTION_MARKS = ("?", "？")
_SENTENCE_MARKS = "。；;！!"
_MAX_CLARIFICATION_CHARS = 120


class QueryAssessmentProtocolError(ValueError):
    """检索前问题判断不符合严格协议。"""


class _StrictJsonError(ValueError):
    """严格 JSON 解析失败。"""


def _object_without_duplicate_keys(pairs):
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise _StrictJsonError("JSON 对象包含重复字段")
        payload[key] = value
    return payload


@dataclass(frozen=True)
class QueryAssessment:
    """通过协议校验的检索前问题判断。"""

    decision: str
    clarification: str | None = None

    def __post_init__(self):
        if self.decision not in {"answer", "clarify"}:
            raise ValueError("decision 只能是 answer 或 clarify")
        if self.decision == "answer":
            if self.clarification is not None:
                raise ValueError("answer 的 clarification 必须为 null")
            return
        if not isinstance(self.clarification, str):
            raise ValueError("clarify 必须携带 clarification")
        question = re.sub(r"\s+", " ", self.clarification).strip()
        if not question or len(question) > _MAX_CLARIFICATION_CHARS:
            raise ValueError("clarification 必须是非空短问题")
        if sum(question.count(mark) for mark in _QUESTION_MARKS) != 1:
            raise ValueError("clarification 必须包含一个问号")
        if not question.endswith(_QUESTION_MARKS):
            raise ValueError("clarification 必须以问号结尾")
        if any(mark in question for mark in _SENTENCE_MARKS):
            raise ValueError("clarification 只能包含一句问题")
        object.__setattr__(self, "clarification", question)


def parse_and_validate_query_assessment(raw_text):
    """解析严格两字段问题判断。"""
    if not isinstance(raw_text, str):
        raise TypeError("raw_text 必须是字符串")
    try:
        payload = json.loads(
            raw_text,
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (json.JSONDecodeError, _StrictJsonError) as exc:
        raise QueryAssessmentProtocolError("问题判断不是严格 JSON") from exc
    if not isinstance(payload, dict) or set(payload) != _FIELDS:
        raise QueryAssessmentProtocolError("问题判断字段不匹配")
    try:
        return QueryAssessment(
            decision=payload["decision"],
            clarification=payload["clarification"],
        )
    except (TypeError, ValueError) as exc:
        raise QueryAssessmentProtocolError("问题判断字段不合法") from exc


def assess_query(query, external_llm):
    """调用外部模型判断检索前是否必须澄清。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串")
    raw_text = generate_json(
        external_llm,
        [
            {"role": "system", "content": QUERY_ASSESSMENT_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        max_tokens=160,
    )
    return parse_and_validate_query_assessment(raw_text)


__all__ = [
    "QUERY_ASSESSMENT_SYSTEM_PROMPT",
    "QueryAssessment",
    "QueryAssessmentProtocolError",
    "assess_query",
    "parse_and_validate_query_assessment",
]
