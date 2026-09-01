"""法律 RAG 的公共契约与总编排接口。"""

from rag.core.contracts import (
    AnswerMode,
    AnswerStatus,
    BusinessRoute,
    Evidence,
    LegalRAG,
    LegalRAGResult,
    LegalTaskType,
    ModelAnswer,
    QueryEnhancementFailureReason,
    QueryEnhancementStatus,
    QueryEnhancementTrace,
    RenderedAnswer,
    RenderedEvidence,
    RouteDecision,
    UnansweredReason,
)


def __getattr__(name):
    """延迟加载总编排，避免 answering 导入契约时形成循环依赖。"""
    if name == "CurrentLawRAG":
        from rag.core.legal_rag import CurrentLawRAG

        globals()[name] = CurrentLawRAG
        return CurrentLawRAG
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AnswerMode",
    "AnswerStatus",
    "BusinessRoute",
    "CurrentLawRAG",
    "Evidence",
    "LegalRAG",
    "LegalRAGResult",
    "LegalTaskType",
    "ModelAnswer",
    "QueryEnhancementFailureReason",
    "QueryEnhancementStatus",
    "QueryEnhancementTrace",
    "RenderedAnswer",
    "RenderedEvidence",
    "RouteDecision",
    "UnansweredReason",
]
