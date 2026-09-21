"""检索前问题充分性判断测试。"""

import unittest

from rag.query.query_assessment import (
    QueryAssessmentProtocolError,
    assess_query,
)


class _External:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, messages, *, temperature, max_tokens, **kwargs):
        self.calls.append((messages, temperature, max_tokens, kwargs))
        return next(self.responses)


class QueryAssessmentTests(unittest.TestCase):
    def test_accepts_answerable_question_and_clarifies_missing_legal_object(self):
        external = _External(
            (
                '{"decision":"answer","clarification":null}',
                '{"decision":"clarify","clarification":"请说明具体申请的是哪项许可、资质或业务？"}',
            )
        )

        answer = assess_query("不小心打了老板违法犯罪吗？", external)
        clarify = assess_query("申请材料需要提交几份？", external)

        self.assertEqual("answer", answer.decision)
        self.assertIsNone(answer.clarification)
        self.assertEqual("clarify", clarify.decision)
        self.assertEqual(
            "请说明具体申请的是哪项许可、资质或业务？",
            clarify.clarification,
        )
        system_prompt = external.calls[0][0][0]["content"]
        self.assertIn("一般性或条件式回答", system_prompt)
        self.assertIn("申请材料需要提交几份", system_prompt)
        self.assertEqual(
            {"type": "json_object"},
            external.calls[0][3]["response_format"],
        )
        self.assertEqual(
            {"thinking": {"type": "disabled"}},
            external.calls[0][3]["extra_body"],
        )

    def test_rejects_inconsistent_or_extra_fields(self):
        invalid = (
            '{"decision":"answer","clarification":"请补充信息？"}',
            '{"decision":"clarify","clarification":null}',
            '{"decision":"clarify","clarification":"请补充信息","reason":"x"}',
        )
        for raw_text in invalid:
            with self.subTest(raw_text=raw_text):
                with self.assertRaises(QueryAssessmentProtocolError):
                    assess_query("申请材料需要提交几份？", _External((raw_text,)))


if __name__ == "__main__":
    unittest.main()
