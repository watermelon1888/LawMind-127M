"""外部候选回答审查调整协议测试。"""

import json
import unittest

from rag.core import Evidence
from rag.query import (
    AnswerReviewProtocolError,
    build_answer_review_prompt,
    parse_and_validate_answer_review,
    review_answer,
)


class FakeExternalLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, messages, *, temperature, max_tokens, **kwargs):
        self.calls.append((messages, temperature, max_tokens, kwargs))
        return self.response


def make_evidence():
    return Evidence("中华人民共和国民法典", "143", "具备相应条件的民事法律行为有效。")


class TestAnswerReviewProtocol(unittest.TestCase):
    def test_accepts_accept_clarify_and_adjust_decisions(self):
        accept = parse_and_validate_answer_review(
            '{"decision":"accept","reason":"core_answer_supported",'
            '"clarification":null,"adjusted_summary":null,"citations":null}'
        )
        clarify = parse_and_validate_answer_review(
            '{"decision":"clarify","reason":"missing_decisive_fact",'
            '"clarification":"请补充合同签订的具体时间？",'
            '"adjusted_summary":null,"citations":null}'
        )
        adjust = parse_and_validate_answer_review(
            '{"decision":"adjust","reason":"unsupported_claim",'
            '"clarification":null,"adjusted_summary":"期限应以证据原文为准。",'
            '"citations":["E2","E1"]}'
        )

        self.assertEqual("accept", accept.decision)
        self.assertEqual("请补充合同签订的具体时间？", clarify.clarification)
        self.assertEqual("unsupported_claim", adjust.reason)
        self.assertEqual("期限应以证据原文为准。", adjust.adjusted_summary)
        self.assertEqual(("E2", "E1"), adjust.citations)

    def test_rejects_invalid_field_combinations(self):
        invalid = (
            '{"decision":"accept","reason":"ok","clarification":"继续说明？","adjusted_summary":null,"citations":null}',
            '{"decision":"clarify","reason":"missing_facts","clarification":null,"adjusted_summary":null,"citations":null}',
            '{"decision":"adjust","reason":"bad","clarification":"说明？","adjusted_summary":"调整。","citations":["E1"]}',
            '{"decision":"adjust","reason":"bad","clarification":null,"adjusted_summary":null,"citations":["E1"]}',
            '{"decision":"adjust","reason":"bad","clarification":null,"adjusted_summary":"调整。","citations":[]}',
            '{"decision":"adjust","reason":"bad","clarification":null,"adjusted_summary":"调整。","citations":["E1","E1"]}',
            '{"decision":"accept","reason":"ok","clarification":null,"adjusted_summary":"调整。","citations":["E1"]}',
            '{"decision":"unknown","reason":"x","clarification":null,"adjusted_summary":null,"citations":null}',
            '~~~json\n{"decision":"accept","reason":"ok","clarification":null,"adjusted_summary":null,"citations":null}\n~~~',
            '{"decision":"clarify","reason":"missing_facts",'
            '"clarification":"现有证据不相关。请补充具体事实？",'
            '"adjusted_summary":null,"citations":null}',
            '{"decision":"clarify","reason":"missing_facts",'
            '"clarification":"请补充具体事实","adjusted_summary":null,"citations":null}',
        )

        for raw_text in invalid:
            with self.subTest(raw_text=raw_text):
                with self.assertRaises(AnswerReviewProtocolError):
                    parse_and_validate_answer_review(raw_text)

    def test_normalizes_audit_reason_and_missing_nullable_fields(self):
        accepted = parse_and_validate_answer_review(
            '{"decision":"accept","reason":"not stable"}'
        )
        adjusted = parse_and_validate_answer_review(
            '{"decision":"adjust",'
            '"reason":"candidate_confuses_立项建议_with_立项申请材料",'
            '"adjusted_summary":"应以证据中的申请材料为准。",'
            '"citations":["E1"]}'
        )

        self.assertEqual("accept_review", accepted.reason)
        self.assertIsNone(accepted.clarification)
        self.assertEqual("adjust_review", adjusted.reason)
        self.assertIsNone(adjusted.clarification)

    def test_review_prompt_is_tolerant_and_exposes_full_evidence(self):
        llm = FakeExternalLLM(
            '{"decision":"accept","reason":"core_answer_supported",'
            '"clarification":null,"adjusted_summary":null,"citations":null}'
        )

        decision = review_answer(
            "民事法律行为有效需要满足什么条件？",
            "候选回答",
            (make_evidence(), Evidence("中华人民共和国民法典", "144", "补充证据。")),
            llm,
            candidate_citations=("E1",),
        )

        self.assertEqual("accept", decision.decision)
        messages, temperature, max_tokens, kwargs = llm.calls[0]
        system_prompt = messages[0]["content"]
        payload = json.loads(messages[1]["content"])
        self.assertIn("核心结论", system_prompt)
        self.assertIn("轻微重复", system_prompt)
        self.assertIn("一般性或条件式回答", system_prompt)
        self.assertIn("数字、期限、适用条件", system_prompt)
        self.assertIn("不得因为答案质量问题选择 clarify", system_prompt)
        self.assertIn("任意一个具体制度", system_prompt)
        self.assertEqual("候选回答", payload["candidate_answer"])
        self.assertEqual(["E1"], payload["candidate_citations"])
        self.assertEqual(["E1", "E2"], [item["evidence_id"] for item in payload["evidence"]])
        self.assertEqual("143", payload["evidence"][0]["article_no"])
        self.assertEqual(0, temperature)
        self.assertEqual(480, max_tokens)
        self.assertEqual({"type": "json_object"}, kwargs["response_format"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
