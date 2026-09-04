"""外部澄清规划协议测试。"""

import json
import unittest

from rag.core.contracts import Evidence, LegalTaskType
from rag.query.clarification import (
    ClarificationPlan,
    ClarificationPlanningError,
    ClarificationProtocolError,
    build_clarification_prompt,
    parse_and_validate_clarification,
    plan_clarification,
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


def valid_payload():
    return json.dumps(
        {
            "missing_information": ["发生时间"],
            "question": "请问该行为发生在什么时间？",
        },
        ensure_ascii=False,
    )


class TestClarificationProtocol(unittest.TestCase):
    def test_parses_and_normalizes_plan(self):
        plan = parse_and_validate_clarification(
            json.dumps(
                {
                    "missing_information": [" 发生时间\n"],
                    "question": " 请问该行为发生在什么时间？ ",
                },
                ensure_ascii=False,
            )
        )
        self.assertIsInstance(plan, ClarificationPlan)
        self.assertEqual(("发生时间",), plan.missing_information)
        self.assertEqual("请问该行为发生在什么时间？", plan.question)

    def test_rejects_malformed_or_unsafe_outputs(self):
        cases = (
            '{"missing_information":["主体"]}',
            '{"missing_information":[],"question":"请补充主体？"}',
            '{"missing_information":["主体","主体"],"question":"请补充主体？"}',
            '{"missing_information":["主体"],"question":"问题一？问题二？"}',
            '{"missing_information":["主体"],"question":"该行为构成违法。"}',
            '{"missing_information":["主体"],"question":"请问该行为是否合法？"}',
            '{"missing_information":["主体"],"question":"请参考《刑法》第264条。"}',
            '{"missing_information":["主体"],"question":"请补充主体？","extra":1}',
            '{"missing_information":["主体"],"question":"请补充主体？"} trailing',
            '```json\n' + valid_payload() + '\n```',
        )
        for raw in cases:
            with self.subTest(raw=raw):
                with self.assertRaises(ClarificationProtocolError):
                    parse_and_validate_clarification(raw)

    def test_prompt_contains_data_only_and_validates_inputs(self):
        evidence = (
            Evidence(
                law_name="劳动法",
                article_no="50",
                content="工资应当按月支付。",
            ),
        )
        messages = build_clarification_prompt(
            "老板拖欠工资怎么办？",
            LegalTaskType.CASE_APPLICATION,
            evidence,
        )
        self.assertEqual("system", messages[0]["role"])
        payload = json.loads(messages[1]["content"])
        self.assertEqual("case_application", payload["task_type"])
        self.assertEqual("劳动法", payload["evidence"][0]["law_name"])
        with self.assertRaises(ValueError):
            build_clarification_prompt("  ", None, ())

    def test_plan_calls_external_once_with_controlled_parameters(self):
        llm = FakeExternalLLM(valid_payload())
        plan = plan_clarification(
            "老板拖欠工资怎么办？",
            LegalTaskType.CASE_APPLICATION,
            (),
            llm,
        )
        self.assertEqual(("发生时间",), plan.missing_information)
        self.assertEqual(1, len(llm.calls))
        self.assertEqual(0, llm.calls[0]["temperature"])
        self.assertEqual(128, llm.calls[0]["max_tokens"])

    def test_external_failure_is_raised_as_planning_error(self):
        with self.assertRaises(ClarificationPlanningError):
            plan_clarification(
                "问题",
                None,
                (),
                FakeExternalLLM(error=RuntimeError("不可用")),
            )


if __name__ == "__main__":
    unittest.main()
