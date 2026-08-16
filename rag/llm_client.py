"""
MiniMind HTTP API 客户端封装。

通过 OpenAI 兼容协议调用 MiniMind serve_openai_api 服务。
供 rag.query_understanding 及后续 Ch9/Ch11 共用。
"""
from openai import OpenAI, APIConnectionError, APIError

DEFAULT_BASE_URL = "http://localhost:8998/v1"
DEFAULT_MODEL = "minimind-local:latest"
DEFAULT_TIMEOUT = 60.0

_STARTUP_HINT = (
    "请先启动 MiniMind API 服务:\n"
    "  python -m minimind.scripts.serve_openai_api"
)


class LLMAPIError(Exception):
    """LLM API 调用失败。"""


class LLMClient:
    """MiniMind HTTP API 客户端。"""

    def __init__(self, base_url=DEFAULT_BASE_URL, api_key="sk-123",
                 model=DEFAULT_MODEL, timeout=DEFAULT_TIMEOUT):
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def chat(self, messages, temperature=0.7, max_tokens=512):
        """非流式调用，返回 assistant 文本。

        messages: [{"role": "user"|"assistant"|"system", "content": "..."}, ...]
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
                extra_body={"chat_template_kwargs": {"open_thinking": False}},
            )
            return response.choices[0].message.content or ""
        except APIConnectionError as e:
            raise LLMAPIError(f"无法连接 MiniMind API 服务。{_STARTUP_HINT}") from e
        except APIError as e:
            raise LLMAPIError(f"MiniMind API 调用失败: {e}") from e

    def health_check(self):
        """发送简短问候验证服务可达。"""
        try:
            reply = self.chat(
                [{"role": "user", "content": "你好"}],
                temperature=0.7, max_tokens=32,
            )
            return bool(reply.strip())
        except LLMAPIError:
            return False


# ============================================================
if __name__ == "__main__":
    client = LLMClient()
    print(f"MiniMind API: {client.model} @ {DEFAULT_BASE_URL}")
    if not client.health_check():
        print("连通性自检失败。")
        print(_STARTUP_HINT)
        exit(1)
    print("连通性自检通过。")

    reply = client.chat(
        [{"role": "user", "content": "用一句话解释什么是取保候审"}],
        temperature=0.7, max_tokens=128,
    )
    print(f"\nQ: 什么是取保候审？\nA: {reply}")
