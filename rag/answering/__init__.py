"""法律证据构包、回答协议与中文渲染。"""

from rag.answering.evidence import (
    EvidencePackage,
    EvidencePackager,
    EvidencePackagingError,
    MAX_EVIDENCE_ITEMS,
    RAG_MAX_OUTPUT_TOKENS,
)
from rag.answering.protocol import (
    ASSISTANT_SCHEMA,
    AnswerProtocolError,
    SYSTEM_PROMPT,
    build_answer_prompt,
    parse_and_validate_answer,
)
from rag.answering.render import (
    render_clarification,
    render_exact_lookup,
    render_failure,
    render_refusal,
    render_semantic_answer,
)
from rag.answering.token_count import AnswerPromptTokenCounter, PromptTokenCountError

__all__ = [
    "ASSISTANT_SCHEMA",
    "AnswerPromptTokenCounter",
    "AnswerProtocolError",
    "EvidencePackage",
    "EvidencePackager",
    "EvidencePackagingError",
    "MAX_EVIDENCE_ITEMS",
    "RAG_MAX_OUTPUT_TOKENS",
    "PromptTokenCountError",
    "SYSTEM_PROMPT",
    "build_answer_prompt",
    "parse_and_validate_answer",
    "render_clarification",
    "render_exact_lookup",
    "render_failure",
    "render_refusal",
    "render_semantic_answer",
]
