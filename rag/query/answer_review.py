"""外部模型对轻量法律回答的宽容审查调整协议。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from rag.external import generate_json


ANSWER_REVIEW_SYSTEM_PROMPT = """你是轻量法律模型的审查调整器，负责审查候选答案并在必要时直接给出受证据约束的调整结果。
目标是判断候选回答能否近似正确地交付，而不是追求完美、全面或律师级措辞。
选择 accept：候选回答的核心结论与实际引用证据一致并回应原问题。允许答案简短、不完整、措辞朴素、轻微重复，也允许对信息不足的问题给出有意义的一般性或条件式回答。数字、期限、适用条件、行为主体或义务主体与证据不一致属于核心错误，不能 accept。
选择 clarify：仅当用户原问题缺少会改变法律结论的决定性适用范围，现有证据只是任意一个具体制度的规定，因而无法给出具有一般意义或条件式的回答时使用。例如未说明何种许可、资质、考试或处罚制度，却询问期限、条件、金额或办理方式。不得因为答案质量问题选择 clarify，不得因为存在更多细分情形或答案不够全面而追问；“试用期最多多久”这类存在一般适用规则的问题应回答而不是澄清。
选择 adjust：仅当候选回答存在会改变核心结论的证据外内容、数字或主体错配、严重遗漏、严重重复、偏题或引用错配时使用。adjusted_summary 必须直接给出调整后的完整答案，只能依据输入 evidence，保持一至三个单行短句，不得包含法条条号、Markdown、换行或证据编号。citations 必须列出支持调整后答案所需的全部 evidence_id，按证据顺序且不得重复。轻微瑕疵不得 adjust。
只输出严格 JSON 对象，字段必须是 decision、reason、clarification、adjusted_summary、citations，不得输出 Markdown、解释或其他字段。
decision 只能是 accept、clarify 或 adjust。reason 必须是简短的小写 snake_case 原因码。
accept 时 clarification、adjusted_summary 和 citations 都必须为 null。
clarify 时 adjusted_summary 和 citations 必须为 null；clarification 只问最重要的一个缺失事实，只能写一句且恰好一个问号，不得提及候选答案、证据或审查过程。
adjust 时 clarification 必须为 null，adjusted_summary 必须是完整调整结果，citations 必须是非空证据编号数组。"""

ANSWER_REVIEW_SCHEMA = (
    '{"decision":"accept | clarify | adjust",'
    '"reason":"stable_snake_case_reason_code",'
    '"clarification":"one short question | null",'
    '"adjusted_summary":"one to three grounded sentences | null",'
    '"citations":["E1"] | null}'
)

_FIELDS = frozenset(
    {"decision", "reason", "clarification", "adjusted_summary", "citations"}
)
_NULLABLE_FIELDS = ("clarification", "adjusted_summary", "citations")
_REASON_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_EVIDENCE_ID_PATTERN = re.compile(r"^E[1-9][0-9]*$")
_QUESTION_MARKS = ("?", "？")
_CLARIFICATION_SENTENCE_MARKS = "。；;！!"
_MAX_CLARIFICATION_CHARS = 120
_MAX_ADJUSTED_SUMMARY_CHARS = 800


class AnswerReviewProtocolError(ValueError):
    """外部答案审查结果不符合严格协议。"""


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
class AnswerReviewDecision:
    """通过协议校验的候选回答审查结果。"""

    decision: str
    reason: str
    clarification: str | None = None
    adjusted_summary: str | None = None
    citations: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.decision not in {"accept", "clarify", "adjust"}:
            raise ValueError("decision 只能是 accept、clarify 或 adjust")
        if not isinstance(self.reason, str) or not _REASON_PATTERN.fullmatch(
            self.reason
        ):
            raise ValueError("reason 必须是小写 snake_case 原因码")

        if self.decision == "accept":
            if any(
                item is not None
                for item in (self.clarification, self.adjusted_summary, self.citations)
            ):
                raise ValueError(
                    "accept 不能携带 clarification、adjusted_summary 或 citations"
                )
            return

        if self.decision == "adjust":
            if self.clarification is not None:
                raise ValueError("adjust 的 clarification 必须为 null")
            if not isinstance(self.adjusted_summary, str):
                raise ValueError("adjust 必须携带 adjusted_summary")
            summary = self.adjusted_summary.strip()
            if not summary or len(summary) > _MAX_ADJUSTED_SUMMARY_CHARS:
                raise ValueError("adjusted_summary 必须是非空短答案")
            if not isinstance(self.citations, (list, tuple)):
                raise ValueError("adjust 必须携带 citations")
            citations = tuple(self.citations)
            if not citations or any(
                not isinstance(item, str) or not _EVIDENCE_ID_PATTERN.fullmatch(item)
                for item in citations
            ):
                raise ValueError("citations 必须是非空证据编号数组")
            if len(set(citations)) != len(citations):
                raise ValueError("citations 不能重复")
            object.__setattr__(self, "adjusted_summary", summary)
            object.__setattr__(self, "citations", citations)
            return

        if self.adjusted_summary is not None or self.citations is not None:
            raise ValueError("clarify 的 adjusted_summary 和 citations 必须为 null")
        if not isinstance(self.clarification, str):
            raise ValueError("clarify 必须携带 clarification")
        question = re.sub(r"\s+", " ", self.clarification).strip()
        if not question:
            raise ValueError("clarification 不能为空")
        if len(question) > _MAX_CLARIFICATION_CHARS:
            raise ValueError("clarification 过长")
        if sum(question.count(mark) for mark in _QUESTION_MARKS) != 1:
            raise ValueError("clarification 必须包含一个问号")
        if not question.endswith(_QUESTION_MARKS):
            raise ValueError("clarification 必须以问号结尾")
        if any(mark in question for mark in _CLARIFICATION_SENTENCE_MARKS):
            raise ValueError("clarification 只能包含一句问题")
        object.__setattr__(self, "clarification", question)


def build_answer_review_prompt(
    query,
    candidate_answer,
    evidence,
    *,
    candidate_citations=(),
):
    """构造包含完整证据包的答案审查调整请求。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串")
    if not isinstance(candidate_answer, str) or not candidate_answer.strip():
        raise ValueError("candidate_answer 必须是非空字符串")
    try:
        evidence_items = tuple(evidence)
    except TypeError as exc:
        raise TypeError("evidence 必须是可迭代对象") from exc
    try:
        candidate_citation_items = tuple(candidate_citations)
    except TypeError as exc:
        raise TypeError("candidate_citations 必须是可迭代对象") from exc
    if any(not isinstance(item, str) for item in candidate_citation_items):
        raise TypeError("candidate_citations 必须是字符串数组")
    payload = {
        "query": query,
        "candidate_answer": candidate_answer,
        "candidate_citations": list(candidate_citation_items),
        "evidence": [
            {
                "evidence_id": f"E{index}",
                "law_name": item.law_name,
                "article_no": item.article_no,
                "content": item.content,
            }
            for index, item in enumerate(evidence_items, start=1)
        ],
    }
    return [
        {"role": "system", "content": ANSWER_REVIEW_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def parse_and_validate_answer_review(raw_text):
    """严格解析外部答案审查 JSON。"""
    if not isinstance(raw_text, str):
        raise TypeError("raw_text 必须是字符串")
    try:
        payload = json.loads(
            raw_text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (json.JSONDecodeError, _StrictJsonError) as exc:
        raise AnswerReviewProtocolError("模型输出不是严格 JSON") from exc
    if not isinstance(payload, dict) or "decision" not in payload:
        raise AnswerReviewProtocolError(
            "答案审查必须是包含 decision 的 JSON 对象"
        )
    unexpected = frozenset(payload) - _FIELDS
    if unexpected:
        raise AnswerReviewProtocolError("答案审查包含未知字段")
    decision = payload["decision"]
    reason = payload.get("reason")
    if not isinstance(reason, str) or not _REASON_PATTERN.fullmatch(reason):
        reason = f"{decision}_review" if decision in {"accept", "clarify", "adjust"} else "invalid_review"
    for field_name in _NULLABLE_FIELDS:
        payload.setdefault(field_name, None)
    try:
        return AnswerReviewDecision(
            decision=decision,
            reason=reason,
            clarification=payload["clarification"],
            adjusted_summary=payload["adjusted_summary"],
            citations=payload["citations"],
        )
    except (TypeError, ValueError) as exc:
        raise AnswerReviewProtocolError("答案审查字段不合法") from exc


def review_answer(
    query,
    candidate_answer,
    evidence,
    external_llm,
    *,
    candidate_citations=(),
):
    """调用一次外部模型审查候选答案。"""
    if external_llm is None or not callable(
        getattr(external_llm, "generate", None)
    ):
        raise TypeError("external_llm 必须提供可调用的 generate")
    messages = build_answer_review_prompt(
        query,
        candidate_answer,
        evidence,
        candidate_citations=candidate_citations,
    )
    raw_text = generate_json(external_llm, messages, max_tokens=480)
    return parse_and_validate_answer_review(raw_text)


__all__ = [
    "ANSWER_REVIEW_SCHEMA",
    "ANSWER_REVIEW_SYSTEM_PROMPT",
    "AnswerReviewDecision",
    "AnswerReviewProtocolError",
    "build_answer_review_prompt",
    "parse_and_validate_answer_review",
    "review_answer",
]
