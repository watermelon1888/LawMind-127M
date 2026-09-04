"""外部大模型调用 adapter。"""

from rag.external.client import OpenAICompatibleLLM
from rag.external.contracts import (
    ExternalLLM,
    ExternalLLMError,
    ExternalLLMErrorCode,
)

__all__ = [
    "ExternalLLM",
    "ExternalLLMError",
    "ExternalLLMErrorCode",
    "OpenAICompatibleLLM",
]
