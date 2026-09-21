"""外部大模型调用的稳定契约。"""

from enum import Enum
import inspect
from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Sequence


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

    def __init__(
        self,
        code: ExternalLLMErrorCode,
        message: str,
        *,
        status_code: Optional[int] = None,
    ):
        if not isinstance(code, ExternalLLMErrorCode):
            raise TypeError("code 必须是 ExternalLLMErrorCode")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message 必须是非空字符串")
        if status_code is not None and (
            isinstance(status_code, bool)
            or not isinstance(status_code, int)
            or status_code < 100
            or status_code > 599
        ):
            raise ValueError("status_code 必须是有效 HTTP 状态码或 None")
        self.code = code
        self.status_code = status_code
        super().__init__(message)


@dataclass(frozen=True)
class ExternalToolCall:
    """外部模型返回的单个工具调用。"""

    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ExternalLLMResponse:
    """保留文本、工具调用和结束原因的外部模型响应。"""

    content: Optional[str]
    tool_calls: tuple[ExternalToolCall, ...] = ()
    finish_reason: Optional[str] = None


class ExternalLLM(Protocol):
    """外部大模型 adapter 对业务层暴露的最小接口。"""

    def generate(
        self,
        messages: Sequence[dict],
        *,
        temperature: float,
        max_tokens: int,
        response_format: Optional[Mapping[str, str]] = None,
        extra_body: Optional[Mapping[str, object]] = None,
    ) -> str:
        """根据消息生成一段文本，可选用 JSON 输出模式。"""
        ...

    def generate_with_tools(
        self,
        messages: Sequence[dict],
        *,
        tools: Sequence[Mapping[str, object]],
        temperature: float,
        max_tokens: int,
    ) -> ExternalLLMResponse:
        """调用外部模型并保留原生工具调用结果。"""
        ...


def generate_json(external_llm, messages, *, max_tokens):
    """以关闭思考模式的 JSON 输出调用 adapter，兼容旧版替身。"""
    generate = getattr(external_llm, "generate", None)
    if not callable(generate):
        raise TypeError("external_llm 必须提供可调用的 generate")
    kwargs = {"temperature": 0, "max_tokens": max_tokens}
    try:
        parameters = inspect.signature(generate).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
        supports_response_format = True
    else:
        supports_response_format = any(
            parameter.name == "response_format"
            or parameter.kind is parameter.VAR_KEYWORD
            for parameter in parameters
        )
    if supports_response_format:
        kwargs["response_format"] = {"type": "json_object"}
    supports_extra_body = False
    try:
        parameters = inspect.signature(generate).parameters.values()
    except (TypeError, ValueError):
        supports_extra_body = True
    else:
        supports_extra_body = any(
            parameter.name == "extra_body"
            or parameter.kind is parameter.VAR_KEYWORD
            for parameter in parameters
        )
    if supports_extra_body:
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return generate(messages, **kwargs)


__all__ = [
    "ExternalLLM",
    "ExternalLLMError",
    "ExternalLLMErrorCode",
    "ExternalLLMResponse",
    "ExternalToolCall",
    "generate_json",
]
