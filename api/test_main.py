"""法律 RAG FastAPI 入口测试。"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi.testclient import TestClient

from api.main import create_app
from api.server import ServerSettings, build_web_app
from rag.core import (
    AnswerStatus,
    AuditEvent,
    AuditTrace,
    BusinessRoute,
    LegalRAG,
    LegalRAGResult,
    ModelAnswer,
    RenderedAnswer,
)


class FakeLegalRAG(LegalRAG):
    """用于验证 API 序列化的最小核心替身。"""

    def __init__(self, result):
        self.result = result
        self.queries = []

    def answer(self, query: str) -> LegalRAGResult:
        self.queries.append(query)
        return self.result


def make_result():
    trace = AuditTrace(
        "trace-api-1",
        events=(
            AuditEvent(
                "route_decided",
                "succeeded",
                details={
                    "route": "answer",
                    "answer_mode": "retrieval",
                },
            ),
        ),
        final_route=BusinessRoute.ANSWER,
        final_status=AnswerStatus.RETRIEVED_EVIDENCE,
    )
    return LegalRAGResult(
        query="劳动合同问题",
        status=AnswerStatus.RETRIEVED_EVIDENCE,
        rendered_answer=RenderedAnswer(message="示例回答"),
        model_answer=ModelAnswer("示例回答", ("E1",)),
        candidate_answer="127M 候选回答",
        audit_trace=trace,
    )


class TestFastAPI入口(unittest.TestCase):
    def test_answer_returns_business_status_and_audit_trace(self):
        fake = FakeLegalRAG(make_result())
        client = TestClient(create_app(fake))

        response = client.post("/v1/answer", json={"query": "劳动合同问题"})

        self.assertEqual(200, response.status_code)
        body = response.json()
        self.assertEqual("trace-api-1", body["request_id"])
        self.assertEqual("劳动合同问题", body["query"])
        self.assertEqual("answer", body["route"])
        self.assertEqual("retrieved_evidence", body["status"])
        self.assertEqual("retrieval", body["answer_mode"])
        self.assertNotIn("task_type", body)
        self.assertEqual("示例回答", body["answer"])
        self.assertNotIn("法律依据：", body["answer"])
        self.assertNotIn("回答范围：", body["answer"])
        self.assertEqual("127M 候选回答", body["candidate_answer"])
        self.assertEqual("trace-api-1", body["audit_trace"]["trace_id"])
        self.assertEqual(["劳动合同问题"], fake.queries)

    def test_blank_query_is_rejected(self):
        client = TestClient(create_app(FakeLegalRAG(make_result())))
        response = client.post("/v1/answer", json={"query": "   "})
        self.assertEqual(422, response.status_code)

    def test_health_endpoints_do_not_call_application(self):
        fake = FakeLegalRAG(make_result())
        client = TestClient(create_app(fake))
        self.assertEqual(200, client.get("/health/live").status_code)
        self.assertEqual({"status": "ok"}, client.get("/health/live").json())
        self.assertEqual(200, client.get("/health/ready").status_code)
        self.assertEqual({"status": "ready"}, client.get("/health/ready").json())
        self.assertEqual([], fake.queries)

    def test_unconfigured_application_is_not_ready(self):
        client = TestClient(create_app())
        self.assertEqual(503, client.get("/health/ready").status_code)
        self.assertEqual(503, client.post("/v1/answer", json={"query": "问题"}).status_code)

    def test_built_frontend_can_be_served_without_shadowing_api(self):
        with TemporaryDirectory() as directory:
            frontend_dir = Path(directory)
            (frontend_dir / "index.html").write_text(
                "<h1>LawMind</h1>",
                encoding="utf-8",
            )
            client = TestClient(
                create_app(FakeLegalRAG(make_result()), frontend_dir=frontend_dir)
            )

            self.assertIn("LawMind", client.get("/").text)
            self.assertEqual(200, client.get("/health/ready").status_code)

    def test_web_server_uses_current_department_rule_assets(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "model.pth"
            tokenizer = root / "tokenizer"
            article_index = root / "articles.jsonl"
            artifacts = root / "artifacts"
            frontend = root / "frontend"
            weights.write_bytes(b"weights")
            tokenizer.mkdir()
            article_index.write_text("{}\n", encoding="utf-8")
            artifacts.mkdir()
            frontend.mkdir()
            (frontend / "index.html").write_text("LawMind", encoding="utf-8")
            settings = ServerSettings(
                weights=weights,
                weights_sha256="a" * 64,
                tokenizer_path=tokenizer,
                article_index=article_index,
                artifact_dir=artifacts,
                frontend_dir=frontend,
                device="cpu",
                enable_external_llm=True,
                host="127.0.0.1",
                port=8000,
            )
            captured = {}

            def loader(**kwargs):
                captured.update(kwargs)
                return FakeLegalRAG(make_result())

            client = TestClient(build_web_app(settings, loader=loader))

            self.assertEqual(200, client.get("/health/ready").status_code)
            self.assertIn("LawMind", client.get("/").text)
            self.assertEqual(article_index, captured["article_index"])
            self.assertEqual(artifacts, captured["artifact_dir"])
            self.assertTrue(captured["enable_external_llm"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
