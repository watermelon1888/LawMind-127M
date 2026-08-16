"""法律 RAG 的统一入口契约与跨模块数据结构。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple


class AnswerStatus(str, Enum):
    """一次法律问答的对外处理结果。"""

    VERIFIED_LOOKUP = "verified_lookup"
    RETRIEVED_EVIDENCE = "retrieved_evidence"
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
    query_enhancement: QueryEnhancementTrace = field(
        default_factory=QueryEnhancementTrace
    )

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
        if not isinstance(self.query_enhancement, QueryEnhancementTrace):
            raise TypeError("query_enhancement 必须是 QueryEnhancementTrace")


class LegalRAG(ABC):
    """法律 RAG 对调用方暴露的统一接口。"""

    @abstractmethod
    def answer(self, query: str) -> LegalRAGResult:
        """依据现行有效法律证据回答问题，或返回受控未作答结果。"""
        raise NotImplementedError


__all__ = [
    "AnswerStatus",
    "Evidence",
    "LegalRAG",
    "LegalRAGResult",
    "ModelAnswer",
    "QueryEnhancementFailureReason",
    "QueryEnhancementStatus",
    "QueryEnhancementTrace",
    "RenderedAnswer",
    "RenderedEvidence",
    "UnansweredReason",
]
