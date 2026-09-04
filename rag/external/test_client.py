import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rag.external import (
    ExternalLLMError,
    ExternalLLMErrorCode,
    OpenAICompatibleLLM,
)


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class FakeTransport:
    def __init__(self, response=None, error=None):
        self.chat = SimpleNamespace(
            completions=FakeCompletions(response=response, error=error)
        )


def response_with_content(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class TestOpenAICompatibleLLM(unittest.TestCase):
    def make_client(self, transport=None):
        return OpenAICompatibleLLM(
            base_url="https://api.deepseek.com",
            api_key="",
            model="deepseek-chat",
            timeout_seconds=10,
            transport=transport or FakeTransport(response_with_content("ok")),
        )

    def test_success_returns_first_choice_and_does_not_mutate_messages(self):
        transport = FakeTransport(response_with_content("回答"))
        client = self.make_client(transport)
        messages = [{"role": "user", "content": "问题"}]

        self.assertEqual("回答", client.generate(messages, temperature=0, max_tokens=32))
        self.assertEqual(messages, [{"role": "user", "content": "问题"}])
        call = transport.chat.completions.calls[0]
        self.assertEqual("deepseek-chat", call["model"])
        self.assertEqual(messages, call["messages"])
        self.assertFalse(call["stream"])

    def test_invalid_request_is_rejected_before_transport_call(self):
        transport = FakeTransport(response_with_content("ok"))
        client = self.make_client(transport)
        cases = (
            ([], 0, 32),
            ([{"role": "tool", "content": "x"}], 0, 32),
            ([{"role": "user", "content": "x"}], -1, 32),
            ([{"role": "user", "content": "x"}], 0, 0),
        )
        for messages, temperature, max_tokens in cases:
            with self.subTest(messages=messages, temperature=temperature):
                with self.assertRaises(ExternalLLMError) as context:
                    client.generate(
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                self.assertIs(
                    ExternalLLMErrorCode.INVALID_REQUEST,
                    context.exception.code,
                )
        self.assertEqual([], transport.chat.completions.calls)

    def test_transport_errors_are_normalized_without_retry(self):
        cases = (
            (TimeoutError("timeout"), ExternalLLMErrorCode.TIMEOUT),
            (ConnectionError("connection"), ExternalLLMErrorCode.CONNECTION_ERROR),
            (RuntimeError("connection"), ExternalLLMErrorCode.SERVICE_ERROR),
        )
        for error, expected_code in cases:
            with self.subTest(error=error):
                transport = FakeTransport(error=error)
                with self.assertRaises(ExternalLLMError) as context:
                    self.make_client(transport).generate(
                        [{"role": "user", "content": "问题"}],
                        temperature=0,
                        max_tokens=32,
                    )
                self.assertIs(expected_code, context.exception.code)
                self.assertEqual(1, len(transport.chat.completions.calls))

    def test_empty_or_malformed_response_is_invalid_response(self):
        responses = (
            SimpleNamespace(choices=[]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=""))]),
            SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=None))]),
        )
        for response in responses:
            with self.subTest(response=response):
                with self.assertRaises(ExternalLLMError) as context:
                    self.make_client(FakeTransport(response)).generate(
                        [{"role": "user", "content": "问题"}],
                        temperature=0,
                        max_tokens=32,
                    )
                self.assertIs(
                    ExternalLLMErrorCode.INVALID_RESPONSE,
                    context.exception.code,
                )

    def test_from_env_requires_all_values_and_parses_timeout(self):
        values = {
            "EXTERNAL_LLM_BASE_URL": "https://api.deepseek.com",
            "EXTERNAL_LLM_API_KEY": "secret",
            "EXTERNAL_LLM_MODEL": "deepseek-chat",
            "EXTERNAL_LLM_TIMEOUT_SECONDS": "12.5",
        }
        with patch.dict(os.environ, values, clear=False):
            client = OpenAICompatibleLLM.from_env()
        self.assertEqual("https://api.deepseek.com", client.base_url)
        self.assertEqual("deepseek-chat", client.model)
        self.assertEqual(12.5, client.timeout_seconds)

        missing = dict(values)
        del missing["EXTERNAL_LLM_API_KEY"]
        with patch.dict(os.environ, missing, clear=True):
            with patch("rag.external.client.load_dotenv"):
                with self.assertRaises(ExternalLLMError) as context:
                    OpenAICompatibleLLM.from_env()
        self.assertIs(
            ExternalLLMErrorCode.CONFIGURATION_ERROR,
            context.exception.code,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
