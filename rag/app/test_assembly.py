"""统一应用装配入口测试。"""

import unittest
from unittest.mock import patch

from rag.app import RAGApplicationConfig, build_current_law_rag
from rag.answering import EvidenceBundlePackager, EvidencePackager
from rag.app.assembly import PRODUCTION_RETRIEVAL_CONFIG
from rag.core import CurrentLawRAG
from rag.knowledge import EvidenceUnitRepository, split_article
from rag.core.test_legal_rag import (
    FakeSemanticRetriever,
    make_article,
    make_repository,
)


class TestRAGApplicationConfig(unittest.TestCase):
    def test_production_retrieval_parameters_are_frozen(self):
        self.assertEqual(30, PRODUCTION_RETRIEVAL_CONFIG.dense_top_k)
        self.assertEqual(30, PRODUCTION_RETRIEVAL_CONFIG.sparse_top_k)
        self.assertEqual(4, PRODUCTION_RETRIEVAL_CONFIG.rrf_k)
        self.assertEqual(20, PRODUCTION_RETRIEVAL_CONFIG.candidate_pool)
        self.assertEqual(5, PRODUCTION_RETRIEVAL_CONFIG.top_k)

    def test_normalizes_paths_and_validates_budget(self):
        config = RAGApplicationConfig("laws.jsonl", "artifacts", 1024)
        self.assertEqual("laws.jsonl", str(config.article_index))
        self.assertEqual("artifacts", str(config.artifact_dir))
        with self.assertRaises(ValueError):
            RAGApplicationConfig("laws.jsonl", "artifacts", 0)

    def test_evidence_unit_mode_requires_its_versioned_directory(self):
        with self.assertRaisesRegex(ValueError, "evidence_unit_dir"):
            RAGApplicationConfig(
                "laws.jsonl",
                "artifacts",
                768,
                retrieval_mode="evidence_unit",
            )
        config = RAGApplicationConfig(
            "laws.jsonl",
            "artifacts",
            768,
            retrieval_mode="evidence_unit",
            evidence_unit_dir="unit-artifacts/v1",
        )
        self.assertEqual("unit-artifacts\\v1", str(config.evidence_unit_dir))


class TestBuildCurrentLawRAG(unittest.TestCase):
    def setUp(self):
        article = make_article()
        self.repository = make_repository((article,))
        self.retriever = FakeSemanticRetriever()
        self.config = RAGApplicationConfig("missing.jsonl", "missing-artifacts", 1024)

    def test_assembles_with_injected_runtime_dependencies(self):
        application = build_current_law_rag(
            self.config,
            generate=lambda messages, *, temperature, max_tokens: "",
            count_prompt_tokens=lambda package: 1,
            repository=self.repository,
            semantic_retriever=self.retriever,
        )
        self.assertIsInstance(application, CurrentLawRAG)
        self.assertIs(self.repository, application._article_repository)
        self.assertIs(self.retriever, application._semantic_retriever)
        self.assertIsNone(application._external_llm)
        self.assertIsInstance(application._evidence_packager, EvidencePackager)

    def test_evidence_unit_mode_switches_retriever_and_packager_together(self):
        article = self.repository.get_by_chunk_id("中华人民共和国刑法#264")
        units = split_article(article)
        unit_repository = EvidenceUnitRepository(
            {unit.unit_id: unit for unit in units},
            {article.chunk_id: units},
            units[0].splitter_version,
        )

        class UnitScorer:
            def score(self, query, values):
                return tuple(1.0 for _ in values)

        config = RAGApplicationConfig(
            "missing.jsonl",
            "parent-artifacts",
            768,
            retrieval_mode="evidence_unit",
            evidence_unit_dir="unit-artifacts/v1",
        )
        application = build_current_law_rag(
            config,
            generate=lambda messages, *, temperature, max_tokens: "",
            count_prompt_tokens=lambda package: 1,
            repository=self.repository,
            semantic_retriever=self.retriever,
            unit_repository=unit_repository,
            unit_scorer=UnitScorer(),
        )

        self.assertIs(self.retriever, application._semantic_retriever)
        self.assertIsInstance(application._evidence_packager, EvidenceBundlePackager)

    def test_can_create_external_adapter_through_config(self):
        class FakeExternal:
            def generate(self, messages, *, temperature, max_tokens):
                return ""

        external = FakeExternal()
        config = RAGApplicationConfig(
            "missing.jsonl",
            "missing-artifacts",
            1024,
            enable_external_llm=True,
        )
        with patch("rag.app.assembly.OpenAICompatibleLLM.from_env", return_value=external):
            application = build_current_law_rag(
                config,
                generate=lambda messages, *, temperature, max_tokens: "",
                count_prompt_tokens=lambda package: 1,
                repository=self.repository,
                semantic_retriever=self.retriever,
            )
        self.assertIs(external, application._external_llm)

    def test_requires_exactly_one_prompt_counter_source(self):
        with self.assertRaises(TypeError):
            build_current_law_rag(
                self.config,
                generate=lambda messages, *, temperature, max_tokens: "",
                repository=self.repository,
                semantic_retriever=self.retriever,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
