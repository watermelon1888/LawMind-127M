"""法律 RAG 的统一入口契约与跨模块数据结构。"""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Tuple


class BusinessRoute(str, Enum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    REFUSE = "refuse"
    GENERAL_CHAT = "general_chat"


class AnswerMode(str, Enum):
    EXACT_LOOKUP = "exact_lookup"
    RETRIEVAL = "retrieval"


@dataclass(frozen=True)
class RouteDecision:
    query: str
    route: BusinessRoute
    answer_mode: Optional[AnswerMode] = None
    reason: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.query, str):
            raise TypeError("query must be a string")
        if not isinstance(self.route, BusinessRoute):
            raise TypeError("route must be BusinessRoute")
        if self.answer_mode is not None and not isinstance(
            self.answer_mode, AnswerMode
        ):
            raise TypeError("answer_mode must be AnswerMode or None")
        if self.reason is not None:
            _require_non_blank("reason", self.reason)
        if self.route is not BusinessRoute.ANSWER:
            if self.answer_mode is not None:
                raise ValueError("non-answer routes cannot carry answer_mode")
            if self.route is BusinessRoute.REFUSE and self.reason is None:
                raise ValueError("refuse route must carry reason")
            return

        if self.answer_mode is None:
            raise ValueError("answer route must carry answer_mode")
        if self.reason is not None:
            raise ValueError("answer routes cannot carry reason")


class AnswerStatus(str, Enum):
    """一次法律问答的对外处理结果。"""

    VERIFIED_LOOKUP = "verified_lookup"
    RETRIEVED_EVIDENCE = "retrieved_evidence"
    GENERAL_CHAT = "general_chat"
    CLARIFICATION_REQUIRED = "clarification_required"
    REFUSED = "refused"
    PROCESSING_FAILED = "processing_failed"


class UnansweredReason(str, Enum):
    """系统没有返回法律回答的稳定原因。"""

    NON_LEGAL = "non_legal"
    TIME_SENSITIVE = "time_sensitive"
    UNSUPPORTED_LEGAL_SOURCE = "unsupported_legal_source"
    UNSUPPORTED_LEGAL_TASK = "unsupported_legal_task"
    NO_VERIFIABLE_EVIDENCE = "no_verifiable_evidence"
    CLARIFICATION_REQUIRED = "clarification_required"
    PROCESSING_FAILED = "processing_failed"


class QueryEnhancementStatus(str, Enum):
    """Query 增强层对本次请求的处理状态。"""

    NOT_ATTEMPTED = "not_attempted"
    APPLIED = "applied"
    FALLBACK = "fallback"


class QueryEnhancementFailureReason(str, Enum):
    """允许显式回退原始 query 的稳定原因。"""

    CALL_FAILED = "call_failed"
    TIMEOUT = "timeout"
    INVALID_OUTPUT = "invalid_output"
    ENHANCED_RETRIEVAL_FAILED = "enhanced_retrieval_failed"


_UNANSWERED_REASONS_BY_STATUS = {
    AnswerStatus.CLARIFICATION_REQUIRED: frozenset(
        {UnansweredReason.CLARIFICATION_REQUIRED}
    ),
    AnswerStatus.REFUSED: frozenset(
        {
            UnansweredReason.NON_LEGAL,
            UnansweredReason.TIME_SENSITIVE,
            UnansweredReason.UNSUPPORTED_LEGAL_SOURCE,
            UnansweredReason.UNSUPPORTED_LEGAL_TASK,
            UnansweredReason.NO_VERIFIABLE_EVIDENCE,
        }
    ),
    AnswerStatus.PROCESSING_FAILED: frozenset(
        {UnansweredReason.PROCESSING_FAILED}
    ),
}


def _require_non_blank(name, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")


@dataclass(frozen=True)
class QueryEnhancementTrace:
    """不进入回答文本的 Query 增强审计记录。"""

    status: QueryEnhancementStatus = QueryEnhancementStatus.NOT_ATTEMPTED
    failure_reason: Optional[QueryEnhancementFailureReason] = None
    retrieval_queries: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not isinstance(self.status, QueryEnhancementStatus):
            raise TypeError("status 必须是 QueryEnhancementStatus")
        if self.failure_reason is not None and not isinstance(
            self.failure_reason, QueryEnhancementFailureReason
        ):
            raise TypeError("failure_reason 必须是 QueryEnhancementFailureReason 或 None")
        queries = tuple(self.retrieval_queries)
        object.__setattr__(self, "retrieval_queries", queries)
        for query in queries:
            _require_non_blank("retrieval_queries 中的 query", query)
        if len(set(queries)) != len(queries):
            raise ValueError("retrieval_queries 不能包含重复 query")
        if len(queries) > 6:
            raise ValueError("retrieval_queries 最多包含六条 query")
        if self.status is QueryEnhancementStatus.FALLBACK:
            if self.failure_reason is None:
                raise ValueError("fallback 必须携带 failure_reason")
            if len(queries) != 1:
                raise ValueError("fallback 必须只使用原始 query")
        elif self.failure_reason is not None:
            raise ValueError("只有 fallback 可以携带 failure_reason")
        if self.status is QueryEnhancementStatus.APPLIED and not queries:
            raise ValueError("applied 必须携带实际检索 query")
        if self.status is QueryEnhancementStatus.NOT_ATTEMPTED and len(queries) > 1:
            raise ValueError("not_attempted 最多只能执行原始 query")


def _freeze_audit_value(value):
    """将审计明细保存为不可变的可序列化结构。"""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("audit details 的键必须是非空字符串")
            items.append((key, _freeze_audit_value(item)))
        return tuple(sorted(items))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_audit_value(item) for item in value)
    raise TypeError("audit details 只能包含 JSON 可序列化值")


def _thaw_audit_value(value):
    if isinstance(value, tuple):
        if value and all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in value
        ):
            return {key: _thaw_audit_value(item) for key, item in value}
        return [_thaw_audit_value(item) for item in value]
    return value


@dataclass(frozen=True)
class AuditEvent:
    """按实际执行顺序记录的单个审计事件。"""

    stage: str
    status: str
    source: Optional[str] = None
    reason: Optional[str] = None
    details: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self):
        _require_non_blank("stage", self.stage)
        _require_non_blank("status", self.status)
        if self.source is not None:
            _require_non_blank("source", self.source)
        if self.reason is not None:
            _require_non_blank("reason", self.reason)
        details = (
            self.details.items()
            if isinstance(self.details, Mapping)
            else self.details
        )
        normalized = []
        seen = set()
        for item in details:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise TypeError("audit details 必须由键值对组成")
            key, value = item
            if not isinstance(key, str) or not key.strip():
                raise ValueError("audit details 的键必须是非空字符串")
            if key in seen:
                raise ValueError("audit details 不能包含重复键")
            seen.add(key)
            normalized.append((key, _freeze_audit_value(value)))
        object.__setattr__(self, "details", tuple(sorted(normalized)))

    def to_dict(self):
        """返回适合 JSON 序列化的审计事件。"""
        return {
            "stage": self.stage,
            "status": self.status,
            "source": self.source,
            "reason": self.reason,
            "details": {
                key: _thaw_audit_value(value) for key, value in self.details
            },
        }


@dataclass(frozen=True)
class AuditTrace:
    """从请求接入到最终结果的有序审计链。"""

    trace_id: str
    events: Tuple[AuditEvent, ...] = field(default_factory=tuple)
    final_route: Optional[BusinessRoute] = None
    final_status: Optional[AnswerStatus] = None

    def __post_init__(self):
        _require_non_blank("trace_id", self.trace_id)
        events = tuple(self.events)
        if any(not isinstance(item, AuditEvent) for item in events):
            raise TypeError("events 中的元素必须是 AuditEvent")
        object.__setattr__(self, "events", events)
        if self.final_route is not None and not isinstance(
            self.final_route, BusinessRoute
        ):
            raise TypeError("final_route 必须是 BusinessRoute 或 None")
        if self.final_status is not None and not isinstance(
            self.final_status, AnswerStatus
        ):
            raise TypeError("final_status 必须是 AnswerStatus 或 None")

    def to_dict(self):
        """返回适合 JSON 序列化的完整审计链。"""
        return {
            "trace_id": self.trace_id,
            "events": [event.to_dict() for event in self.events],
            "final_route": (
                None if self.final_route is None else self.final_route.value
            ),
            "final_status": None
            if self.final_status is None
            else self.final_status.value,
        }


@dataclass(frozen=True)
class Evidence:
    """不依赖请求内编号的 canonical 完整法条证据。"""

    law_name: str
    article_no: str
    content: str

    def __post_init__(self):
        for name in ("law_name", "article_no", "content"):
            _require_non_blank(name, getattr(self, name))


@dataclass(frozen=True)
class ModelAnswer:
    """通过严格两字段协议校验后的原子法律结论。"""

    summary: str
    citations: Tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        citations = tuple(self.citations)
        object.__setattr__(self, "citations", citations)
        if not isinstance(self.summary, str):
            raise TypeError("summary 必须是字符串")
        for citation in citations:
            _require_non_blank("citations 中的证据编号", citation)
        if not self.summary.strip() or not citations:
            raise ValueError("模型回答必须携带非空归纳和引用")


@dataclass(frozen=True)
class RenderedEvidence:
    """允许向用户展示的法条投影。"""

    law_name: str
    article_no: str
    content: str

    def __post_init__(self):
        for name in ("law_name", "article_no", "content"):
            _require_non_blank(name, getattr(self, name))


_ANSWER_SCOPE = (
    "以上内容仅依据本次检索并引用的现行有效法条，"
    "不代表已经穷尽所有可能适用的法律规定。"
)


def _format_article_no(article_no):
    main, separator, suffix = article_no.partition("之")
    return f"第{main}条之{suffix}" if separator else f"第{main}条"


@dataclass(frozen=True)
class RenderedAnswer:
    """所有路径共用的公开展示结构。"""

    evidence: Tuple[RenderedEvidence, ...] = field(default_factory=tuple)
    summary: Optional[str] = None
    message: Optional[str] = None

    def __post_init__(self):
        evidence = tuple(self.evidence)
        object.__setattr__(self, "evidence", evidence)
        if any(not isinstance(item, RenderedEvidence) for item in evidence):
            raise TypeError("evidence 中的元素必须是 RenderedEvidence")
        if self.summary is not None:
            _require_non_blank("summary", self.summary)
        if self.message is not None:
            _require_non_blank("message", self.message)
        if self.message is not None and (evidence or self.summary is not None):
            raise ValueError("消息型展示不能同时携带证据或归纳")
        if self.summary is not None and not evidence:
            raise ValueError("语义归纳必须携带至少一条展示证据")
        if not evidence and self.message is None:
            raise ValueError("RenderedAnswer 不能为空")

    def to_text(self):
        """生成唯一固定中文表示。"""
        if self.message is not None:
            return f"提示：\n{self.message}"
        sections = []
        for index, item in enumerate(self.evidence, start=1):
            sections.append(
                f"{index}. 《{item.law_name}》{_format_article_no(item.article_no)}\n"
                f"   法条全文：{item.content}"
            )
        evidence_text = "\n\n".join(sections)
        if self.summary is not None:
            return (
                f"简要归纳：\n{self.summary}\n\n"
                f"法律依据：\n{evidence_text}\n\n"
                f"回答范围：\n{_ANSWER_SCOPE}"
            )
        return "法律依据：\n" + evidence_text


@dataclass(frozen=True)
class LegalRAGResult:
    """`LegalRAG.answer` 返回的完整、可审计结果。"""

    query: str
    status: AnswerStatus
    rendered_answer: RenderedAnswer
    unanswered_reason: Optional[UnansweredReason] = None
    diagnostic_code: Optional[str] = None
    evidence: Tuple[Evidence, ...] = field(default_factory=tuple)
    model_answer: Optional[ModelAnswer] = None
    candidate_answer: Optional[str] = None
    query_enhancement: QueryEnhancementTrace = field(
        default_factory=QueryEnhancementTrace
    )
    audit_trace: AuditTrace = field(default_factory=lambda: AuditTrace("untracked"))

    def __post_init__(self):
        if not isinstance(self.query, str):
            raise TypeError("query 必须是字符串")
        if not isinstance(self.status, AnswerStatus):
            raise TypeError("status 必须是 AnswerStatus")
        if not isinstance(self.rendered_answer, RenderedAnswer):
            raise TypeError("rendered_answer 必须是 RenderedAnswer")
        if self.unanswered_reason is not None and not isinstance(
            self.unanswered_reason, UnansweredReason
        ):
            raise TypeError("unanswered_reason 必须是 UnansweredReason 或 None")
        if self.diagnostic_code is not None:
            _require_non_blank("diagnostic_code", self.diagnostic_code)

        allowed_reasons = _UNANSWERED_REASONS_BY_STATUS.get(self.status)
        if allowed_reasons is None:
            if self.unanswered_reason is not None:
                raise ValueError("成功回答不能携带未作答原因")
            if self.diagnostic_code is not None:
                raise ValueError("成功回答不能携带诊断码")
        elif self.unanswered_reason is None:
            raise ValueError("未作答状态必须携带 unanswered_reason")
        elif self.unanswered_reason not in allowed_reasons:
            raise ValueError("status 与 unanswered_reason 不相容")
        if (
            self.status is not AnswerStatus.PROCESSING_FAILED
            and self.diagnostic_code is not None
        ):
            raise ValueError("只有处理失败可以携带诊断码")

        evidence = tuple(self.evidence)
        object.__setattr__(self, "evidence", evidence)
        if any(not isinstance(item, Evidence) for item in evidence):
            raise TypeError("evidence 中的元素必须是 Evidence")
        if self.model_answer is not None and not isinstance(
            self.model_answer, ModelAnswer
        ):
            raise TypeError("model_answer 必须是 ModelAnswer 或 None")
        if self.candidate_answer is not None:
            _require_non_blank("candidate_answer", self.candidate_answer)
        if not isinstance(self.query_enhancement, QueryEnhancementTrace):
            raise TypeError("query_enhancement 必须是 QueryEnhancementTrace")
        if not isinstance(self.audit_trace, AuditTrace):
            raise TypeError("audit_trace 必须是 AuditTrace")


class LegalRAG(ABC):
    """法律 RAG 对调用方暴露的统一接口。"""

    @abstractmethod
    def answer(self, query: str) -> LegalRAGResult:
        """依据现行有效法律证据回答问题，或返回受控未作答结果。"""
        raise NotImplementedError


__all__ = [
    "AuditEvent",
    "AuditTrace",
    "AnswerMode",
    "AnswerStatus",
    "BusinessRoute",
    "Evidence",
    "LegalRAG",
    "LegalRAGResult",
    "ModelAnswer",
    "QueryEnhancementFailureReason",
    "QueryEnhancementStatus",
    "QueryEnhancementTrace",
    "RenderedAnswer",
    "RenderedEvidence",
    "RouteDecision",
    "UnansweredReason",
]
