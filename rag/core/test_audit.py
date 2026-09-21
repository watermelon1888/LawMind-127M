"""步骤 12 决策追踪与全链路审计测试。"""

import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from rag.core import (
    AnswerMode,
    AnswerStatus,
    AuditEvent,
    AuditTrace,
    BusinessRoute,
    CurrentLawRAG,
    LegalRAGResult,
    RenderedAnswer,
    RouteDecision,
    UnansweredReason,
)
from rag.core.test_legal_rag import (
    FakeSemanticRetriever,
    RecordingGenerator,
    RecordingExternalLLM,
    assessment_json,
    answer_json,
    make_article,
    make_packager,
    make_ranked,
    make_repository,
)
from rag.query import QueryAssessment


class TestAuditContracts(unittest.TestCase):
    def test_event_details_are_immutable_and_json_ready(self):
        event = AuditEvent(
            "route_decided",
            "succeeded",
            source="deterministic",
            details={"route": "clarify", "nested": ["a", 1]},
        )

        self.assertEqual(
            {
                "stage": "route_decided",
                "status": "succeeded",
                "source": "deterministic",
                "reason": None,
                "details": {"nested": ["a", 1], "route": "clarify"},
            },
            event.to_dict(),
        )
        with self.assertRaises(FrozenInstanceError):
            event.stage = "changed"

    def test_trace_rejects_non_events(self):
        with self.assertRaises(TypeError):
            AuditTrace("trace", events=("invalid",))


class TestCurrentLawRAGAudit(unittest.TestCase):
    def make_rag(self):
        return object.__new__(CurrentLawRAG)

    def test_clarification_path_is_closed_from_request_to_completion(self):
        rag = self.make_rag()
        decision = RouteDecision(
            "信息不完整",
            BusinessRoute.CLARIFY,
            reason="missing_key_facts",
        )

        with patch("rag.core.legal_rag.route_query", return_value=decision):
            result = rag.answer(decision.query)

        self.assertIsInstance(result, LegalRAGResult)
        self.assertEqual(BusinessRoute.CLARIFY, result.audit_trace.final_route)
        self.assertEqual(
            AnswerStatus.CLARIFICATION_REQUIRED,
            result.audit_trace.final_status,
        )
        stages = [event.stage for event in result.audit_trace.events]
        self.assertEqual(
            [
                "request_received",
                "route_decided",
                "clarification",
                "final_render",
                "completed",
            ],
            stages,
        )
        self.assertEqual("clarify", result.audit_trace.to_dict()["final_route"])

    def test_processing_failure_keeps_answer_route_in_audit(self):
        rag = self.make_rag()
        decision = RouteDecision(
            "法律问题",
            BusinessRoute.ANSWER,
            answer_mode=AnswerMode.RETRIEVAL,
        )
        failure = LegalRAGResult(
            query=decision.query,
            status=AnswerStatus.PROCESSING_FAILED,
            unanswered_reason=UnansweredReason.PROCESSING_FAILED,
            diagnostic_code="test_failure",
            rendered_answer=RenderedAnswer(message="处理失败"),
        )
        with patch("rag.core.legal_rag.route_query", return_value=decision):
            with patch(
                "rag.core.legal_rag.assess_query",
                return_value=QueryAssessment("answer"),
            ):
                with patch.object(
                    CurrentLawRAG,
                    "_answer_semantic_search",
                    return_value=failure,
                ):
                    result = rag.answer(decision.query)

        self.assertEqual(BusinessRoute.ANSWER, result.audit_trace.final_route)
        self.assertEqual("test_failure", result.diagnostic_code)

    def test_retrieval_path_records_generation_and_validation(self):
        article = make_article()
        rag = CurrentLawRAG(
            article_repository=make_repository((article,)),
            semantic_retriever=FakeSemanticRetriever((make_ranked(article),)),
            evidence_packager=make_packager(),
            generate=RecordingGenerator(answer_json("依法处理。", ["E1"])),
            external_llm=RecordingExternalLLM(assessment_json()),
        )
        decision = RouteDecision(
            "法律问题",
            BusinessRoute.ANSWER,
            AnswerMode.RETRIEVAL,
        )
        with patch("rag.core.legal_rag.route_query", return_value=decision):
            result = rag.answer(decision.query)

        self.assertIs(AnswerStatus.RETRIEVED_EVIDENCE, result.status)
        stages = [event.stage for event in result.audit_trace.events]
        for stage in (
            "query_assessment",
            "retrieval",
            "evidence_packaging",
            "answer_generation",
            "protocol_validation",
            "final_render",
            "completed",
        ):
            self.assertIn(stage, stages)


if __name__ == "__main__":
    unittest.main(verbosity=2)
