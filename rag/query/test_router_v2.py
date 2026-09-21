"""简化后三路入口的确定性路由测试。"""

import unittest

from rag.core import AnswerMode, BusinessRoute
from rag.query import route_query


class TestThreeRouteDecision(unittest.TestCase):
    def test_exact_reference_is_answer_exact_lookup(self):
        decision = route_query("刑法第264条是什么？")

        self.assertIs(BusinessRoute.ANSWER, decision.route)
        self.assertIs(AnswerMode.EXACT_LOOKUP, decision.answer_mode)

    def test_other_legal_or_unknown_requests_use_single_retrieval_path(self):
        for query in (
            "盗窃行为应承担什么法律责任？",
            "房东不退押金怎么办？",
            "我想咨询一下",
            "公司可以开除我吗？",
        ):
            with self.subTest(query=query):
                decision = route_query(query)
                self.assertIs(BusinessRoute.ANSWER, decision.route)
                self.assertIs(AnswerMode.RETRIEVAL, decision.answer_mode)

    def test_explicit_non_legal_question_is_general_chat(self):
        decision = route_query("今天天气怎么样？")

        self.assertIs(BusinessRoute.GENERAL_CHAT, decision.route)
        self.assertEqual("non_legal", decision.reason)

    def test_unsupported_legal_task_uses_policy_refusal_route(self):
        decision = route_query("请帮我代写一份离婚起诉状。")

        self.assertIs(BusinessRoute.REFUSE, decision.route)
        self.assertEqual("unsupported_legal_task", decision.reason)

    def test_historical_question_uses_policy_refusal_route(self):
        cases = (
            ("2020年修法前劳动法如何规定？", "time_sensitive"),
        )
        for query, reason in cases:
            with self.subTest(query=query):
                decision = route_query(query)
                self.assertIs(BusinessRoute.REFUSE, decision.route)
                self.assertEqual(reason, decision.reason)


if __name__ == "__main__":
    unittest.main(verbosity=2)
