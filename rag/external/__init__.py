"""外部大模型调用 adapter。"""

from rag.external.client import OpenAICompatibleLLM
from rag.external.contracts import (
    ExternalLLM,
    ExternalLLMError,
    ExternalLLMErrorCode,
    ExternalLLMResponse,
    ExternalToolCall,
    generate_json,
)

__all__ = [
    "ExternalLLM",
    "ExternalLLMError",
    "ExternalLLMErrorCode",
    "ExternalLLMResponse",
    "ExternalToolCall",
    "OpenAICompatibleLLM",
    "generate_json",
]
