"""外部大模型调用的稳定契约。"""

from enum import Enum
from typing import Protocol, Sequence


class ExternalLLMErrorCode(str, Enum):
    """外部大模型调用失败的稳定错误码。"""

    CONFIGURATION_ERROR = "configuration_error"
    INVALID_REQUEST = "invalid_request"
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    SERVICE_ERROR = "service_error"
    INVALID_RESPONSE = "invalid_response"


class ExternalLLMError(Exception):
    """外部大模型调用的统一异常。"""

    def __init__(self, code: ExternalLLMErrorCode, message: str):
        if not isinstance(code, ExternalLLMErrorCode):
            raise TypeError("code 必须是 ExternalLLMErrorCode")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message 必须是非空字符串")
        self.code = code
        super().__init__(message)


class ExternalLLM(Protocol):
    """外部大模型 adapter 对业务层暴露的最小接口。"""

    def generate(
        self,
        messages: Sequence[dict],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """根据消息生成一段文本。"""
        ...


__all__ = ["ExternalLLM", "ExternalLLMError", "ExternalLLMErrorCode"]
