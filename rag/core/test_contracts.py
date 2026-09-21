"""法律 RAG 统一契约的纯单元测试。"""

import json
import unittest
from dataclasses import FrozenInstanceError, asdict

from rag.core import (
    AnswerMode,
    AnswerStatus,
    BusinessRoute,
    Evidence,
    LegalRAG,
    LegalRAGResult,
    ModelAnswer,
    QueryEnhancementFailureReason,
    QueryEnhancementStatus,
    QueryEnhancementTrace,
    RenderedAnswer,
    RenderedEvidence,
    RouteDecision,
    UnansweredReason,
)


class TestRouteDecision(unittest.TestCase):
    def test_enum_values_are_stable(self):
        self.assertEqual(
            {"answer", "clarify", "refuse", "general_chat"},
            {route.value for route in BusinessRoute},
        )
        self.assertEqual(
            {"exact_lookup", "retrieval"},
            {mode.value for mode in AnswerMode},
        )

    def test_valid_combinations_are_constructible(self):
        exact = RouteDecision(
            "《刑法》第264条是什么？",
            BusinessRoute.ANSWER,
            AnswerMode.EXACT_LOOKUP,
        )
        retrieval = RouteDecision(
            "员工未签劳动合同怎么办？",
            BusinessRoute.ANSWER,
            AnswerMode.RETRIEVAL,
        )
        pending = RouteDecision(
            "这个行为合法吗？",
            BusinessRoute.ANSWER,
            AnswerMode.RETRIEVAL,
        )
        self.assertIs(AnswerMode.EXACT_LOOKUP, exact.answer_mode)
        self.assertIs(AnswerMode.RETRIEVAL, retrieval.answer_mode)
        self.assertIs(AnswerMode.RETRIEVAL, pending.answer_mode)

    def test_non_answer_routes_cannot_carry_answer_fields(self):
        with self.assertRaises(ValueError):
            RouteDecision(
                "闲聊",
                BusinessRoute.GENERAL_CHAT,
                AnswerMode.RETRIEVAL,
            )

    def test_refuse_route_requires_reason(self):
        with self.assertRaisesRegex(ValueError, "refuse route"):
            RouteDecision("历史法律问题", BusinessRoute.REFUSE)
        decision = RouteDecision(
            "历史法律问题",
            BusinessRoute.REFUSE,
            reason="time_sensitive",
        )
        self.assertEqual("time_sensitive", decision.reason)

    def test_answer_route_requires_mode_and_cannot_carry_reason(self):
        with self.assertRaises(ValueError):
            RouteDecision("法律问题", BusinessRoute.ANSWER)
        with self.assertRaises(ValueError):
            RouteDecision(
                "法律问题",
                BusinessRoute.ANSWER,
                AnswerMode.RETRIEVAL,
                reason="unexpected_reason",
            )

def make_evidence(**overrides):
    defaults = {
        "law_name": "中华人民共和国刑法",
        "article_no": "264",
        "content": "盗窃公私财物，数额较大的，依法承担刑事责任。",
    }
    defaults.update(overrides)
    return Evidence(**defaults)


def make_message(text="固定安全话术"):
    return RenderedAnswer(message=text)


class TestAnswerStatus(unittest.TestCase):
    def test_values_match_public_protocol(self):
        self.assertEqual(
            {
                "verified_lookup",
                "retrieved_evidence",
                "general_chat",
                "clarification_required",
                "refused",
                "processing_failed",
            },
            {status.value for status in AnswerStatus},
        )


class TestUnansweredReason(unittest.TestCase):
    def test_values_match_public_protocol(self):
        expected = {
            "non_legal",
            "time_sensitive",
            "unsupported_legal_source",
            "unsupported_legal_task",
            "no_verifiable_evidence",
            "clarification_required",
            "processing_failed",
        }
        self.assertEqual(expected, {reason.value for reason in UnansweredReason})


class TestQueryEnhancementTrace(unittest.TestCase):
    def test_applied_trace_preserves_unique_retrieval_queries(self):
        trace = QueryEnhancementTrace(
            status=QueryEnhancementStatus.APPLIED,
            retrieval_queries=("原问题", "规范化改写"),
        )

        self.assertEqual(("原问题", "规范化改写"), trace.retrieval_queries)
        self.assertIsNone(trace.failure_reason)

    def test_fallback_requires_reason_and_original_query_only(self):
        with self.assertRaisesRegex(ValueError, "failure_reason"):
            QueryEnhancementTrace(
                status=QueryEnhancementStatus.FALLBACK,
                retrieval_queries=("原问题",),
            )
        with self.assertRaisesRegex(ValueError, "原始 query"):
            QueryEnhancementTrace(
                status=QueryEnhancementStatus.FALLBACK,
                failure_reason=QueryEnhancementFailureReason.CALL_FAILED,
                retrieval_queries=("原问题", "改写"),
            )

    def test_non_fallback_cannot_carry_failure_reason(self):
        with self.assertRaisesRegex(ValueError, "只有 fallback"):
            QueryEnhancementTrace(
                failure_reason=QueryEnhancementFailureReason.INVALID_OUTPUT,
            )

    def test_trace_enforces_retrieval_leg_limits(self):
        with self.assertRaisesRegex(ValueError, "六条"):
            QueryEnhancementTrace(
                status=QueryEnhancementStatus.APPLIED,
                retrieval_queries=tuple(f"query-{index}" for index in range(7)),
            )
        with self.assertRaisesRegex(ValueError, "原始 query"):
            QueryEnhancementTrace(
                retrieval_queries=("原问题", "改写"),
            )


class TestEvidence(unittest.TestCase):
    def test_is_frozen_and_has_no_request_local_id(self):
        evidence = make_evidence()
        with self.assertRaises(FrozenInstanceError):
            evidence.content = "被修改的正文"
        self.assertNotIn("evidence_id", asdict(evidence))
        self.assertNotIn("effective_date", asdict(evidence))

    def test_rejects_blank_required_field(self):
        with self.assertRaises(ValueError):
            make_evidence(law_name=" ")


class TestModelAnswer(unittest.TestCase):
    def test_converts_citations_to_tuple(self):
        answer = ModelAnswer(
            summary="该条规定了盗窃行为的刑事责任。",
            citations=["E1"],
        )
        self.assertEqual(("E1",), answer.citations)

    def test_normal_answer_requires_summary_and_citations(self):
        with self.assertRaises(ValueError):
            ModelAnswer(summary="", citations=())
        with self.assertRaises(ValueError):
            ModelAnswer(summary="有结论", citations=())


class TestRenderedAnswer(unittest.TestCase):
    def test_semantic_answer_renders_program_owned_evidence(self):
        rendered = RenderedAnswer(
            evidence=(
                RenderedEvidence(
                    "中华人民共和国刑法",
                    "264",
                    "第二百六十四条 法条全文。",
                ),
            ),
            summary="该条规定了相应责任。",
        )
        text = rendered.to_text()
        self.assertIn("《中华人民共和国刑法》第264条", text)
        self.assertIn("法条全文：第二百六十四条 法条全文。", text)
        self.assertIn("简要归纳：\n该条规定了相应责任。", text)

    def test_message_cannot_be_mixed_with_evidence(self):
        with self.assertRaises(ValueError):
            RenderedAnswer(
                evidence=(RenderedEvidence("示例法", "1", "正文"),),
                message="提示",
            )


class TestLegalRAGResult(unittest.TestCase):
    def test_success_result_is_json_serializable(self):
        evidence = make_evidence()
        rendered = RenderedAnswer(
            evidence=(RenderedEvidence(evidence.law_name, evidence.article_no, evidence.content),)
        )
        result = LegalRAGResult(
            query="刑法第264条是什么？",
            status=AnswerStatus.VERIFIED_LOOKUP,
            evidence=[evidence],
            rendered_answer=rendered,
        )

        encoded = json.dumps(asdict(result), ensure_ascii=False)
        self.assertIn("verified_lookup", encoded)
        self.assertIn("中华人民共和国刑法", encoded)
        self.assertIsInstance(result.evidence, tuple)
        self.assertIsNone(result.unanswered_reason)
        self.assertIsNone(result.diagnostic_code)
        self.assertIsNone(result.candidate_answer)
        self.assertIs(
            QueryEnhancementStatus.NOT_ATTEMPTED,
            result.query_enhancement.status,
        )

    def test_candidate_answer_must_be_non_blank_text(self):
        result = LegalRAGResult(
            query="测试问题",
            status=AnswerStatus.PROCESSING_FAILED,
            unanswered_reason=UnansweredReason.PROCESSING_FAILED,
            diagnostic_code="output_validation_failed",
            rendered_answer=make_message(),
            candidate_answer="127M 原始输出",
        )
        self.assertEqual("127M 原始输出", result.candidate_answer)

        with self.assertRaises(ValueError):
            LegalRAGResult(
                query="测试问题",
                status=AnswerStatus.PROCESSING_FAILED,
                unanswered_reason=UnansweredReason.PROCESSING_FAILED,
                diagnostic_code="output_validation_failed",
                rendered_answer=make_message(),
                candidate_answer="   ",
            )

    def test_unanswered_status_requires_compatible_reason(self):
        compatible_pairs = (
            (AnswerStatus.CLARIFICATION_REQUIRED, UnansweredReason.CLARIFICATION_REQUIRED),
            (AnswerStatus.REFUSED, UnansweredReason.NON_LEGAL),
            (AnswerStatus.REFUSED, UnansweredReason.UNSUPPORTED_LEGAL_SOURCE),
            (AnswerStatus.REFUSED, UnansweredReason.UNSUPPORTED_LEGAL_TASK),
            (AnswerStatus.REFUSED, UnansweredReason.TIME_SENSITIVE),
            (AnswerStatus.REFUSED, UnansweredReason.NO_VERIFIABLE_EVIDENCE),
            (AnswerStatus.PROCESSING_FAILED, UnansweredReason.PROCESSING_FAILED),
        )
        for status, reason in compatible_pairs:
            with self.subTest(status=status, reason=reason):
                result = LegalRAGResult(
                    query="测试问题",
                    status=status,
                    unanswered_reason=reason,
                    diagnostic_code=(
                        "test_failure"
                        if status is AnswerStatus.PROCESSING_FAILED
                        else None
                    ),
                    rendered_answer=make_message(),
                )
                self.assertIs(reason, result.unanswered_reason)

    def test_rejects_missing_or_incompatible_reason(self):
        with self.assertRaisesRegex(ValueError, "未作答状态"):
            LegalRAGResult(
                query="测试问题",
                status=AnswerStatus.REFUSED,
                rendered_answer=make_message(),
            )
        with self.assertRaisesRegex(ValueError, "不相容"):
            LegalRAGResult(
                query="测试问题",
                status=AnswerStatus.CLARIFICATION_REQUIRED,
                unanswered_reason=UnansweredReason.NON_LEGAL,
                rendered_answer=make_message(),
            )

    def test_only_processing_failure_can_carry_diagnostic_code(self):
        with self.assertRaisesRegex(ValueError, "处理失败"):
            LegalRAGResult(
                query="测试问题",
                status=AnswerStatus.REFUSED,
                unanswered_reason=UnansweredReason.NON_LEGAL,
                diagnostic_code="internal_code",
                rendered_answer=make_message(),
            )


class TestLegalRAGInterface(unittest.TestCase):
    def test_base_class_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            LegalRAG()


if __name__ == "__main__":
    unittest.main(verbosity=2)
