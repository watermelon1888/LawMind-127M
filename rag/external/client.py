"""OpenAI-compatible 外部大模型 adapter。"""

import math
import os
import logging
import time
from collections.abc import Mapping, Sequence

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from rag.external.contracts import ExternalLLMError, ExternalLLMErrorCode


_LOGGER = logging.getLogger(__name__)


class OpenAICompatibleLLM:
    """面向任意 OpenAI-compatible 服务的最小消息生成客户端。"""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float,
        transport=None,
    ):
        self._validate_text("base_url", base_url)
        if not isinstance(api_key, str):
            raise ExternalLLMError(
                ExternalLLMErrorCode.CONFIGURATION_ERROR,
                "api_key 必须是字符串",
            )
        self._validate_text("model", model)
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ExternalLLMError(
                ExternalLLMErrorCode.CONFIGURATION_ERROR,
                "timeout_seconds 必须是正的有限数值",
            )

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = float(timeout_seconds)
        self._transport = (
            transport
            if transport is not None
            else OpenAI(
                api_key=api_key,
                base_url=self.base_url,
                timeout=self.timeout_seconds,
            )
        )

    @classmethod
    def from_env(cls):
        """从 EXTERNAL_LLM_* 环境变量创建生产客户端。"""
        load_dotenv(override=False)
        values = {
            "base_url": os.getenv("EXTERNAL_LLM_BASE_URL"),
            "api_key": os.getenv("EXTERNAL_LLM_API_KEY"),
            "model": os.getenv("EXTERNAL_LLM_MODEL"),
            "timeout_seconds": os.getenv("EXTERNAL_LLM_TIMEOUT_SECONDS"),
        }
        missing = [
            name
            for name, value in values.items()
            if value is None or (isinstance(value, str) and not value.strip())
        ]
        if missing:
            raise ExternalLLMError(
                ExternalLLMErrorCode.CONFIGURATION_ERROR,
                "缺少外部大模型配置: " + ", ".join(missing),
            )
        try:
            timeout_seconds = float(values["timeout_seconds"])
        except (TypeError, ValueError) as exc:
            raise ExternalLLMError(
                ExternalLLMErrorCode.CONFIGURATION_ERROR,
                "EXTERNAL_LLM_TIMEOUT_SECONDS 必须是数值",
            ) from exc
        return cls(
            base_url=values["base_url"],
            api_key=values["api_key"],
            model=values["model"],
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def _validate_text(name, value):
        if not isinstance(value, str) or not value.strip():
            raise ExternalLLMError(
                ExternalLLMErrorCode.CONFIGURATION_ERROR,
                f"{name} 必须是非空字符串",
            )

    @staticmethod
    def _validate_messages(messages):
        if isinstance(messages, (str, bytes)) or not isinstance(
            messages, Sequence
        ):
            raise ExternalLLMError(
                ExternalLLMErrorCode.INVALID_REQUEST,
                "messages 必须是非空消息序列",
            )
        if not messages:
            raise ExternalLLMError(
                ExternalLLMErrorCode.INVALID_REQUEST,
                "messages 必须是非空消息序列",
            )
        allowed_roles = {"system", "user", "assistant"}
        for message in messages:
            if not isinstance(message, Mapping):
                raise ExternalLLMError(
                    ExternalLLMErrorCode.INVALID_REQUEST,
                    "每条 message 必须是对象",
                )
            role = message.get("role")
            content = message.get("content")
            if role not in allowed_roles:
                raise ExternalLLMError(
                    ExternalLLMErrorCode.INVALID_REQUEST,
                    "message.role 不合法",
                )
            if not isinstance(content, str) or not content.strip():
                raise ExternalLLMError(
                    ExternalLLMErrorCode.INVALID_REQUEST,
                    "message.content 必须是非空字符串",
                )

    @staticmethod
    def _validate_parameters(temperature, max_tokens):
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or not 0 <= temperature <= 2
        ):
            raise ExternalLLMError(
                ExternalLLMErrorCode.INVALID_REQUEST,
                "temperature 必须位于 0 到 2 之间",
            )
        if (
            isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
        ):
            raise ExternalLLMError(
                ExternalLLMErrorCode.INVALID_REQUEST,
                "max_tokens 必须是正整数",
            )

    def generate(self, messages, *, temperature, max_tokens):
        """调用一次外部服务并返回首个候选的文本内容。"""
        self._validate_messages(messages)
        self._validate_parameters(temperature, max_tokens)
        started = time.monotonic()
        try:
            response = self._transport.chat.completions.create(
                model=self.model,
                messages=[dict(message) for message in messages],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
        except (APITimeoutError, TimeoutError) as exc:
            _LOGGER.warning(
                "外部大模型调用失败 model=%s code=%s elapsed_ms=%d",
                self.model,
                ExternalLLMErrorCode.TIMEOUT.value,
                int((time.monotonic() - started) * 1000),
            )
            raise ExternalLLMError(
                ExternalLLMErrorCode.TIMEOUT,
                "外部大模型请求超时",
            ) from exc
        except (APIConnectionError, ConnectionError) as exc:
            _LOGGER.warning(
                "外部大模型调用失败 model=%s code=%s elapsed_ms=%d",
                self.model,
                ExternalLLMErrorCode.CONNECTION_ERROR.value,
                int((time.monotonic() - started) * 1000),
            )
            raise ExternalLLMError(
                ExternalLLMErrorCode.CONNECTION_ERROR,
                "无法连接外部大模型服务",
            ) from exc
        except APIError as exc:
            _LOGGER.warning(
                "外部大模型调用失败 model=%s code=%s elapsed_ms=%d",
                self.model,
                ExternalLLMErrorCode.SERVICE_ERROR.value,
                int((time.monotonic() - started) * 1000),
            )
            raise ExternalLLMError(
                ExternalLLMErrorCode.SERVICE_ERROR,
                "外部大模型服务调用失败",
            ) from exc
        except Exception as exc:
            _LOGGER.warning(
                "外部大模型调用失败 model=%s code=%s elapsed_ms=%d",
                self.model,
                ExternalLLMErrorCode.SERVICE_ERROR.value,
                int((time.monotonic() - started) * 1000),
            )
            raise ExternalLLMError(
                ExternalLLMErrorCode.SERVICE_ERROR,
                "外部大模型调用失败",
            ) from exc

        try:
            content = response.choices[0].message.content
        except (AttributeError, IndexError, KeyError, TypeError) as exc:
            _LOGGER.warning(
                "外部大模型响应无效 model=%s code=%s elapsed_ms=%d",
                self.model,
                ExternalLLMErrorCode.INVALID_RESPONSE.value,
                int((time.monotonic() - started) * 1000),
            )
            raise ExternalLLMError(
                ExternalLLMErrorCode.INVALID_RESPONSE,
                "外部大模型响应结构无效",
            ) from exc
        if not isinstance(content, str) or not content.strip():
            _LOGGER.warning(
                "外部大模型响应为空 model=%s code=%s elapsed_ms=%d",
                self.model,
                ExternalLLMErrorCode.INVALID_RESPONSE.value,
                int((time.monotonic() - started) * 1000),
            )
            raise ExternalLLMError(
                ExternalLLMErrorCode.INVALID_RESPONSE,
                "外部大模型返回了空内容",
            )
        _LOGGER.debug(
            "外部大模型调用成功 model=%s elapsed_ms=%d",
            self.model,
            int((time.monotonic() - started) * 1000),
        )
        return content


__all__ = ["OpenAICompatibleLLM"]
