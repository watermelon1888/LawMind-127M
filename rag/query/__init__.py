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
from rag.query.external_analysis import (
    EXTERNAL_ANALYSIS_SCHEMA,
    EXTERNAL_ANALYSIS_SYSTEM_PROMPT,
    ExternalAnalysisProtocolError,
    ExternalRequestDecision,
    analyze_request,
    build_external_analysis_prompt,
    parse_and_validate_external_analysis,
)
from rag.query.task import LegalTaskDecision, classify_legal_task
from rag.query.clarification import (
    CLARIFICATION_SCHEMA,
    CLARIFICATION_SYSTEM_PROMPT,
    ClarificationPlan,
    ClarificationPlanningError,
    ClarificationProtocolError,
    build_clarification_prompt,
    parse_and_validate_clarification,
    plan_clarification,
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
    "EXTERNAL_ANALYSIS_SCHEMA",
    "EXTERNAL_ANALYSIS_SYSTEM_PROMPT",
    "ExternalAnalysisProtocolError",
    "ExternalRequestDecision",
    "analyze_request",
    "build_external_analysis_prompt",
    "parse_and_validate_external_analysis",
    "LegalTaskDecision",
    "classify_legal_task",
    "CLARIFICATION_SCHEMA",
    "CLARIFICATION_SYSTEM_PROMPT",
    "ClarificationPlan",
    "ClarificationPlanningError",
    "ClarificationProtocolError",
    "build_clarification_prompt",
    "parse_and_validate_clarification",
    "plan_clarification",
]
