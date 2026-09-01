"""用户问题识别、规范化与路由模块。"""

from rag.query.enhancement import (
    QUERY_ENHANCEMENT_SCHEMA,
    QUERY_ENHANCEMENT_SYSTEM_PROMPT,
    QueryEnhancement,
    QueryEnhancementProtocolError,
    build_query_enhancement_prompt,
    compile_retrieval_queries,
    parse_and_validate_query_enhancement,
)
from rag.query.router import (
    QueryDecision,
    QueryReason,
    QueryRoute,
    route_query,
)
from rag.core.contracts import (
    AnswerMode,
    BusinessRoute,
    LegalTaskType,
    RouteDecision,
)

__all__ = [
    "AnswerMode",
    "BusinessRoute",
    "LegalTaskType",
    "QUERY_ENHANCEMENT_SCHEMA",
    "QUERY_ENHANCEMENT_SYSTEM_PROMPT",
    "QueryDecision",
    "QueryEnhancement",
    "QueryEnhancementProtocolError",
    "QueryReason",
    "QueryRoute",
    "RouteDecision",
    "build_query_enhancement_prompt",
    "compile_retrieval_queries",
    "parse_and_validate_query_enhancement",
    "route_query",
]
