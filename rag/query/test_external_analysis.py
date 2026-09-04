import unittest

from rag.core.contracts import BusinessRoute, LegalTaskType
from rag.query.external_analysis import (
    ExternalAnalysisProtocolError,
    ExternalRequestDecision,
    analyze_request,
    build_external_analysis_prompt,
    parse_and_validate_external_analysis,
)


class FakeExternalLLM:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def generate(self, messages, *, temperature, max_tokens):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if self.error is not None:
            raise self.error
        return self.response


class TestExternalAnalysis(unittest.TestCase):
    def test_parses_all_three_routes(self):
        cases = (
            (
                '{"route":"answer","task_type":"rule_lookup",'
                '"reason":"legal_rule_question"}',
                BusinessRoute.ANSWER,
                LegalTaskType.RULE_LOOKUP,
            ),
            (
                '{"route":"answer","task_type":"case_application",'
                '"reason":"case_facts_provided"}',
                BusinessRoute.ANSWER,
                LegalTaskType.CASE_APPLICATION,
            ),
            (
                '{"route":"clarify","task_type":null,"reason":"missing_key_facts"}',
                BusinessRoute.CLARIFY,
                None,
            ),
            (
                '{"route":"general_chat","task_type":null,"reason":"non_legal"}',
                BusinessRoute.GENERAL_CHAT,
                None,
            ),
        )
        for raw, route, task_type in cases:
            with self.subTest(raw=raw):
                decision = parse_and_validate_external_analysis(raw)
                self.assertEqual(route, decision.route)
                self.assertEqual(task_type, decision.task_type)

    def test_rejects_malformed_protocol_outputs(self):
        cases = (
            '{"route":"answer","task_type":"rule_lookup"}',
            '{"route":"answer","task_type":"rule_lookup","reason":"x","extra":1}',
            '{"route":"unknown","task_type":null,"reason":"x"}',
            '{"route":"answer","task_type":null,"reason":"x"}',
            '{"route":"clarify","task_type":"rule_lookup","reason":"x"}',
            '```json\n{"route":"clarify","task_type":null,"reason":"x"}\n```',
            '{"route":"clarify","task_type":null,"reason":"x"} trailing',
            '{"route":"clarify","task_type":null,"reason":"not stable"}',
        )
        for raw in cases:
            with self.subTest(raw=raw):
                with self.assertRaises(ExternalAnalysisProtocolError):
                    parse_and_validate_external_analysis(raw)

    def test_analyze_request_calls_once_and_uses_controlled_fallback(self):
        llm = FakeExternalLLM(
            '{"route":"general_chat","task_type":null,"reason":"non_legal"}'
        )
        decision = analyze_request("天气怎么样", llm)
        self.assertEqual(BusinessRoute.GENERAL_CHAT, decision.route)
        self.assertEqual(1, len(llm.calls))
        self.assertEqual(0, llm.calls[0]["temperature"])
        self.assertEqual(128, llm.calls[0]["max_tokens"])

        failed = analyze_request("问题", FakeExternalLLM(error=RuntimeError("x")))
        self.assertEqual(BusinessRoute.CLARIFY, failed.route)
        self.assertIsNone(failed.task_type)
        self.assertEqual("external_analysis_failed", failed.reason)

    def test_prompt_rejects_blank_query(self):
        with self.assertRaises(ValueError):
            build_external_analysis_prompt("  ")

    def test_decision_invariants(self):
        with self.assertRaises(ValueError):
            ExternalRequestDecision(
                BusinessRoute.ANSWER,
                None,
                "legal_rule_question",
            )
        with self.assertRaises(ValueError):
            ExternalRequestDecision(
                BusinessRoute.CLARIFY,
                LegalTaskType.RULE_LOOKUP,
                "missing_key_facts",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
