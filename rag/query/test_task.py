import unittest

from rag.core.contracts import BusinessRoute, LegalTaskType
from rag.query.external_analysis import ExternalRequestDecision
from rag.query.task import LegalTaskDecision, classify_legal_task
from rag.query.router import route_query


class FakeExternalLLM:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate(self, messages, *, temperature, max_tokens):
        self.calls.append((messages, temperature, max_tokens))
        return self.response


class TestLegalTaskClassification(unittest.TestCase):
    def test_deterministic_rule_and_case_classification(self):
        rule = classify_legal_task("劳动法规定加班费怎么算？")
        self.assertIs(LegalTaskType.RULE_LOOKUP, rule.task_type)
        self.assertEqual("deterministic", rule.decision_source)

        case = classify_legal_task("老板三个月没给我加班费，我怎么办？")
        self.assertIs(LegalTaskType.CASE_APPLICATION, case.task_type)
        self.assertEqual("deterministic", case.decision_source)

        mixed = classify_legal_task("劳动法规定加班费怎么算？我去年也没拿到，该怎么办？")
        self.assertIs(LegalTaskType.CASE_APPLICATION, mixed.task_type)

    def test_ambiguous_request_uses_external_analysis_once(self):
        llm = FakeExternalLLM(
            '{"route":"answer","task_type":"case_application",'
            '"reason":"case_facts_provided"}'
        )
        decision = classify_legal_task("劳动争议如何处理", external_llm=llm)
        self.assertIs(LegalTaskType.CASE_APPLICATION, decision.task_type)
        self.assertEqual("external", decision.decision_source)
        self.assertEqual(1, len(llm.calls))

    def test_external_non_answer_routes_are_preserved(self):
        clarify = classify_legal_task(
            "劳动争议如何处理",
            external_decision=ExternalRequestDecision(
                BusinessRoute.CLARIFY,
                None,
                "missing_key_facts",
            ),
        )
        self.assertIs(BusinessRoute.CLARIFY, clarify.route)
        self.assertIsNone(clarify.task_type)

    def test_route_query_attaches_task_type_and_source(self):
        rule = route_query("劳动法规定加班费怎么算？")
        self.assertIs(LegalTaskType.RULE_LOOKUP, rule.task_type)
        self.assertEqual("deterministic", rule.decision_source)

        case = route_query("劳动法规定加班费，我去年未拿到，我如何维权？")
        self.assertIs(LegalTaskType.CASE_APPLICATION, case.task_type)

    def test_task_decision_rejects_invalid_combinations(self):
        with self.assertRaises(ValueError):
            LegalTaskDecision(
                None,
                "external",
                route=BusinessRoute.ANSWER,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
