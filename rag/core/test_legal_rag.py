"""有界 Agentic RAG 总编排的垂直闭环测试。"""

import json
import unittest
from types import SimpleNamespace

from rag.answering import EvidencePackager, RAG_MAX_OUTPUT_TOKENS
from rag.core import AnswerStatus, BusinessRoute, CurrentLawRAG, QueryEnhancementStatus
from rag.external import ExternalLLMResponse, ExternalToolCall
from rag.knowledge import ArticleRepository, LegalArticle


def make_article(
    law_name="中华人民共和国刑法",
    article_no="264",
    content="第二百六十四条 盗窃公私财物，数额较大的，依法承担刑事责任。",
    source_type="legal_regulation",
):
    return LegalArticle(
        chunk_id=f"{law_name}#{article_no}",
        law_name=law_name,
        article_no=article_no,
        content=content,
        source_type=source_type,
    )


def make_repository(articles):
    articles = tuple(articles)
    formal_names = {article.law_name for article in articles}
    aliases = {name: {name} for name in formal_names}
    for name in formal_names:
        if name.startswith("中华人民共和国"):
            aliases.setdefault(name[7:], set()).add(name)
    return ArticleRepository(
        {(article.law_name, article.article_no): article for article in articles},
        aliases,
        formal_names,
    )


def make_ranked(article, rank=1, rerank_score=0.9):
    """保留旧测试 helper 名称，返回 Hybrid 候选所需的最小投影。"""
    del rank, rerank_score
    return SimpleNamespace(chunk_id=article.chunk_id)


def make_packager(max_output_tokens=RAG_MAX_OUTPUT_TOKENS):
    return EvidencePackager(
        context_limit=10000,
        max_output_tokens=max_output_tokens,
        count_prompt_tokens=lambda package: len(package.to_model_json()),
    )


def answer_json(summary="依法处理。", citations=("E1",)):
    return json.dumps(
        {"summary": summary, "citations": list(citations)},
        ensure_ascii=False,
    )


def assessment_json(decision="answer", clarification=None):
    return json.dumps(
        {
            "decision": decision,
            "clarification": clarification,
        },
        ensure_ascii=False,
    )


def tool_call(call_id, name, payload):
    return ExternalToolCall(
        call_id,
        name,
        json.dumps(payload, ensure_ascii=False),
    )


class FakeSemanticRetriever:
    def __init__(self, results=(), error=None):
        self.results = tuple(results)
        self.error = error
        self.calls = []

    def retrieve_candidates(self, query):
        self.calls.append(query)
        if self.error is not None:
            raise self.error
        return self.results

    def rerank_candidates(self, query, candidates):
        del query
        return tuple(candidates)[:5]

    def audit_metadata(self):
        return {
            "dense_top_k": 30,
            "bm25_top_k": 30,
            "rrf_k": 4,
            "candidate_pool": 20,
            "reranker_model": "BAAI/bge-reranker-base",
        }


class FakeAgenticRetriever(FakeSemanticRetriever):
    pass


class FailingRerankRetriever(FakeSemanticRetriever):
    def rerank_candidates(self, query, candidates):
        del query, candidates
        raise RuntimeError("精排失败")


class MappingAgenticRetriever(FakeSemanticRetriever):
    def __init__(self, results_by_query):
        super().__init__()
        self.results_by_query = results_by_query

    def retrieve_candidates(self, query):
        self.calls.append(query)
        return tuple(self.results_by_query.get(query, ()))


class RecordingGenerator:
    def __init__(self, response="", error=None):
        self.response = response
        self.responses = iter(response) if isinstance(response, (tuple, list)) else None
        self.error = error
        self.calls = []

    def __call__(self, messages, *, temperature, max_tokens):
        self.calls.append((messages, temperature, max_tokens))
        if self.error is not None:
            raise self.error
        if self.responses is not None:
            return next(self.responses)
        return self.response


class RecordingExternalLLM:
    def __init__(self, response="", error=None, tool_error=None):
        self.response = response
        self.responses = iter(response) if isinstance(response, (tuple, list)) else None
        self.error = error
        self.tool_error = tool_error
        self.calls = []
        self.tool_calls = []

    def generate(self, messages, *, temperature, max_tokens, **kwargs):
        self.calls.append((messages, temperature, max_tokens, kwargs))
        if self.error is not None:
            raise self.error
        if self.responses is not None:
            return next(self.responses)
        return self.response

    def generate_with_tools(self, messages, *, tools, temperature, max_tokens):
        self.tool_calls.append((messages, tools, temperature, max_tokens))
        if self.tool_error is not None:
            raise self.tool_error
        return ExternalLLMResponse(
            content=None,
            tool_calls=(
                tool_call(
                    "finalize-default",
                    "finalize_evidence",
                    {"promoted_chunk_ids": []},
                ),
            ),
            finish_reason="tool_calls",
        )


class AgenticExternalLLM(RecordingExternalLLM):
    def __init__(self, tool_responses, assessment_response=None):
        super().__init__(assessment_response or assessment_json())
        self.tool_responses = iter(tool_responses)

    def generate_with_tools(self, messages, *, tools, temperature, max_tokens):
        self.tool_calls.append((messages, tools, temperature, max_tokens))
        return next(self.tool_responses)


class CurrentLawRAGTestCase(unittest.TestCase):
    def setUp(self):
        self.article = make_article()
        self.repository = make_repository((self.article,))

    def make_rag(
        self,
        *,
        retriever=None,
        generator=None,
        external_llm=None,
        repository=None,
        evidence_packager=None,
    ):
        return CurrentLawRAG(
            article_repository=repository or self.repository,
            semantic_retriever=retriever or FakeSemanticRetriever(),
            evidence_packager=evidence_packager or make_packager(),
            generate=generator or RecordingGenerator(),
            external_llm=external_llm,
        )


class TestCurrentLawRAGSimplifiedFlow(CurrentLawRAGTestCase):
    def test_agentic_retrieval_applies_one_supplemental_query(self):
        original = "盗窃行为应承担什么责任？"
        supplemental_query = "盗窃行为刑事责任"
        added = make_article(article_no="265", content="盗窃行为依法承担相应责任。")
        retriever = MappingAgenticRetriever(
            {
                original: (make_ranked(self.article),),
                supplemental_query: (make_ranked(added),),
            }
        )
        external = AgenticExternalLLM(
            (
                ExternalLLMResponse(
                    content=None,
                    tool_calls=(
                        tool_call(
                            "search-1",
                            "search_law",
                            {"query": supplemental_query},
                        ),
                    ),
                ),
                ExternalLLMResponse(
                    content=None,
                    tool_calls=(
                        tool_call(
                            "finalize-1",
                            "finalize_evidence",
                            {"promoted_chunk_ids": [added.chunk_id]},
                        ),
                    ),
                ),
            )
        )
        rag = self.make_rag(
            retriever=retriever,
            generator=RecordingGenerator(answer_json()),
            external_llm=external,
            repository=make_repository((self.article, added)),
        )

        result = rag.answer(original)

        self.assertIs(AnswerStatus.RETRIEVED_EVIDENCE, result.status)
        self.assertIs(QueryEnhancementStatus.APPLIED, result.query_enhancement.status)
        self.assertEqual(
            (original, supplemental_query),
            result.query_enhancement.retrieval_queries,
        )
        event = next(
            item.to_dict()
            for item in result.audit_trace.events
            if item.stage == "agentic_retrieval"
        )
        self.assertEqual(
            ["search_law", "finalize_evidence"],
            [item["name"] for item in event["details"]["tool_calls"]],
        )
        self.assertEqual(
            [self.article.chunk_id, added.chunk_id],
            event["details"]["selected_chunk_ids"],
        )
        self.assertEqual(
            [
                {
                    "law_name": added.law_name,
                    "article_no": added.article_no,
                    "source_type": added.source_type,
                }
            ],
            event["details"]["added_hits"],
        )

    def test_agent_can_commit_hybrid_candidate_outside_old_top_five(self):
        articles = tuple(
            make_article(article_no=str(index), content=f"第{index}条 示例正文。")
            for index in range(1, 7)
        )
        retriever = FakeSemanticRetriever(tuple(make_ranked(item) for item in articles))
        external = AgenticExternalLLM(
            (
                ExternalLLMResponse(
                    content=None,
                    tool_calls=(
                        tool_call(
                            "finalize-1",
                            "finalize_evidence",
                            {"promoted_chunk_ids": [articles[5].chunk_id]},
                        ),
                    ),
                ),
            )
        )
        rag = self.make_rag(
            retriever=retriever,
            generator=RecordingGenerator(answer_json()),
            external_llm=external,
            repository=make_repository(articles),
        )

        result = rag.answer("示例问题如何处理？")

        self.assertIs(AnswerStatus.RETRIEVED_EVIDENCE, result.status)
        self.assertEqual(
            ("1", "6", "2", "3", "4"),
            tuple(item.article_no for item in result.evidence),
        )
        event = next(
            item.to_dict()
            for item in result.audit_trace.events
            if item.stage == "agentic_retrieval"
        )
        self.assertEqual(
            [
                {
                    "law_name": articles[5].law_name,
                    "article_no": "6",
                    "source_type": articles[5].source_type,
                }
            ],
            event["details"]["added_hits"],
        )

    def test_top_five_is_packaged_as_prefix_and_minimind_selects_citation(self):
        articles = tuple(
            make_article(article_no=str(index), content=f"第{index}条 示例正文。")
            for index in range(1, 6)
        )
        packager = EvidencePackager(
            context_limit=500,
            max_output_tokens=RAG_MAX_OUTPUT_TOKENS,
            count_prompt_tokens=lambda package: len(package.evidence) * 100,
        )
        rag = self.make_rag(
            retriever=FakeSemanticRetriever(
                tuple(make_ranked(article) for article in articles)
            ),
            generator=RecordingGenerator(answer_json("应当依法处理。", ("E2",))),
            external_llm=RecordingExternalLLM(assessment_json()),
            repository=make_repository(articles),
            evidence_packager=packager,
        )

        result = rag.answer("示例问题如何处理？")

        self.assertIs(AnswerStatus.RETRIEVED_EVIDENCE, result.status)
        self.assertEqual(("1", "2", "3"), tuple(item.article_no for item in result.evidence))
        self.assertEqual(("E2",), result.model_answer.citations)
        self.assertEqual(("2",), tuple(item.article_no for item in result.rendered_answer.evidence))
        events = {item.stage: item.to_dict() for item in result.audit_trace.events}
        self.assertEqual(5, events["agentic_retrieval"]["details"]["selected_count"])
        self.assertEqual(3, events["evidence_packaging"]["details"]["evidence_count"])
        self.assertEqual(["E2"], events["answer_generation"]["details"]["citations"])

    def test_agentic_retrieval_error_does_not_fallback(self):
        external = AgenticExternalLLM(
            (
                ExternalLLMResponse(
                    content=None,
                    tool_calls=(
                        tool_call("call-1", "unknown", {"query": "盗窃责任"}),
                    ),
                ),
            )
        )
        rag = self.make_rag(
            retriever=FakeSemanticRetriever((make_ranked(self.article),)),
            generator=RecordingGenerator(error=AssertionError("不应生成")),
            external_llm=external,
        )

        result = rag.answer("盗窃行为应承担什么责任？")

        self.assertIs(AnswerStatus.PROCESSING_FAILED, result.status)
        self.assertEqual("agentic_retrieval_failed", result.diagnostic_code)

    def test_agentic_audit_records_only_selected_new_hit(self):
        original = "盗窃行为应承担什么责任？"
        supplemental_query = "盗窃行为刑事责任"
        department_rule = make_article(
            law_name="公安机关办理刑事案件程序规定",
            article_no="1",
            content="公安机关依法办理刑事案件。",
            source_type="department_rule",
        )
        retriever = MappingAgenticRetriever(
            {
                original: (make_ranked(self.article),),
                supplemental_query: (make_ranked(department_rule),),
            }
        )
        external = AgenticExternalLLM(
            (
                ExternalLLMResponse(
                    content=None,
                    tool_calls=(
                        tool_call(
                            "search-1",
                            "search_law",
                            {"query": supplemental_query},
                        ),
                    ),
                ),
                ExternalLLMResponse(
                    content=None,
                    tool_calls=(
                        tool_call(
                            "finalize-1",
                            "finalize_evidence",
                            {"promoted_chunk_ids": [department_rule.chunk_id]},
                        ),
                    ),
                ),
            )
        )
        rag = self.make_rag(
            retriever=retriever,
            generator=RecordingGenerator(answer_json()),
            external_llm=external,
            repository=make_repository((self.article, department_rule)),
        )

        result = rag.answer(original)

        event = next(
            item.to_dict()
            for item in result.audit_trace.events
            if item.stage == "agentic_retrieval"
        )
        self.assertEqual(
            [
                {
                    "law_name": department_rule.law_name,
                    "article_no": "1",
                    "source_type": "department_rule",
                }
            ],
            event["details"]["new_hits"],
        )

    def test_exact_lookup_bypasses_generation_and_agents(self):
        generator = RecordingGenerator(error=AssertionError("不应生成"))
        external = RecordingExternalLLM(error=AssertionError("不应审查"))
        rag = self.make_rag(generator=generator, external_llm=external)

        result = rag.answer("《中华人民共和国刑法》第264条是什么？")

        self.assertIs(AnswerStatus.VERIFIED_LOOKUP, result.status)
        self.assertEqual([], generator.calls)
        self.assertEqual([], external.calls)
        self.assertEqual([], external.tool_calls)

    def test_valid_minimind_answer_is_returned_without_external_review(self):
        second_article = make_article(article_no="265", content="与本题无关的另一条法条。")
        retriever = FakeSemanticRetriever(
            (make_ranked(self.article), make_ranked(second_article))
        )
        generator = RecordingGenerator(answer_json())
        external = RecordingExternalLLM(assessment_json())
        rag = self.make_rag(
            retriever=retriever,
            generator=generator,
            external_llm=external,
            repository=make_repository((self.article, second_article)),
        )

        result = rag.answer("盗窃行为应承担什么责任？")

        self.assertIs(AnswerStatus.RETRIEVED_EVIDENCE, result.status)
        self.assertEqual(["盗窃行为应承担什么责任？"], retriever.calls)
        self.assertEqual(1, len(generator.calls))
        self.assertEqual(1, len(external.calls))
        self.assertEqual("盗窃行为应承担什么责任？", external.calls[0][0][1]["content"])
        self.assertEqual(("E1",), result.model_answer.citations)
        self.assertEqual(1, len(result.rendered_answer.evidence))
        self.assertEqual("依法处理。", result.model_answer.summary)
        self.assertIs(QueryEnhancementStatus.NOT_ATTEMPTED, result.query_enhancement.status)
        stages = [item.stage for item in result.audit_trace.events]
        self.assertIn("query_assessment", stages)
        self.assertNotIn("answer_review", stages)
        self.assertNotIn("answer_adjustment", stages)
        retrieval_event = next(
            item.to_dict()
            for item in result.audit_trace.events
            if item.stage == "retrieval"
        )
        self.assertEqual(20, retrieval_event["details"]["parameters"]["candidate_pool"])
        self.assertNotIn("reranker_model", retrieval_event["details"]["parameters"])
        self.assertEqual(2, retrieval_event["details"]["candidate_count"])

    def test_pre_retrieval_clarification_bypasses_retrieval_agent_and_generation(self):
        retriever = FakeSemanticRetriever(
            error=AssertionError("澄清问题不应进入检索")
        )
        generator = RecordingGenerator(error=AssertionError("澄清问题不应生成"))
        external = RecordingExternalLLM(
            assessment_json("clarify", "请说明具体申请事项？")
        )
        rag = self.make_rag(
            retriever=retriever,
            generator=generator,
            external_llm=external,
        )

        result = rag.answer("申请材料需要提交几份？")

        self.assertIs(AnswerStatus.CLARIFICATION_REQUIRED, result.status)
        self.assertIs(BusinessRoute.CLARIFY, result.audit_trace.final_route)
        self.assertEqual("请说明具体申请事项？", result.rendered_answer.message)
        self.assertIsNone(result.model_answer)
        self.assertEqual([], retriever.calls)
        self.assertEqual([], generator.calls)
        self.assertEqual([], external.tool_calls)
        self.assertEqual(1, len(external.calls))

    def test_query_assessment_failure_returns_processing_error_without_retrieval(self):
        retriever = FakeSemanticRetriever(error=AssertionError("判断失败后不应检索"))
        generator = RecordingGenerator(error=AssertionError("判断失败后不应生成"))
        rag = self.make_rag(
            retriever=retriever,
            generator=generator,
            external_llm=RecordingExternalLLM("not-json"),
        )

        result = rag.answer("盗窃行为应承担什么责任？")

        self.assertIs(AnswerStatus.PROCESSING_FAILED, result.status)
        self.assertEqual("query_assessment_failed", result.diagnostic_code)
        self.assertEqual([], retriever.calls)
        self.assertEqual([], generator.calls)
        assessment_event = next(
            item.to_dict()
            for item in result.audit_trace.events
            if item.stage == "query_assessment"
        )
        self.assertEqual("invalid_response", assessment_event["details"]["error_code"])
        self.assertEqual(
            "QueryAssessmentProtocolError",
            assessment_event["details"]["error_type"],
        )

    def test_missing_external_agent_is_an_error_not_a_fallback(self):
        generator = RecordingGenerator(error=AssertionError("不应生成"))
        rag = self.make_rag(
            retriever=FakeSemanticRetriever((make_ranked(self.article),)),
            generator=generator,
        )

        result = rag.answer("盗窃行为应承担什么责任？")

        self.assertIs(AnswerStatus.PROCESSING_FAILED, result.status)
        self.assertEqual("query_assessment_failed", result.diagnostic_code)
        self.assertEqual([], generator.calls)

    def test_empty_retrieval_uses_fixed_clarification_before_generation(self):
        generator = RecordingGenerator(error=AssertionError("不应生成"))
        external = RecordingExternalLLM(assessment_json())
        rag = self.make_rag(generator=generator, external_llm=external)

        result = rag.answer("一个知识库没有覆盖的问题")

        self.assertIs(AnswerStatus.CLARIFICATION_REQUIRED, result.status)
        self.assertEqual([], generator.calls)
        self.assertEqual(1, len(external.calls))
        self.assertEqual([], external.tool_calls)

    def test_semantic_retrieval_failure_is_reported_without_generation(self):
        generator = RecordingGenerator(error=AssertionError("不应生成"))
        rag = self.make_rag(
            retriever=FakeSemanticRetriever(error=RuntimeError("检索失败")),
            generator=generator,
            external_llm=RecordingExternalLLM(assessment_json()),
        )

        result = rag.answer("盗窃行为应承担什么责任？")

        self.assertIs(AnswerStatus.PROCESSING_FAILED, result.status)
        self.assertEqual("semantic_retrieval_failed", result.diagnostic_code)
        self.assertEqual([], generator.calls)

    def test_reranking_failure_is_reported_without_generation(self):
        generator = RecordingGenerator(error=AssertionError("不应生成"))
        rag = self.make_rag(
            retriever=FailingRerankRetriever((make_ranked(self.article),)),
            generator=generator,
            external_llm=RecordingExternalLLM(assessment_json()),
        )

        result = rag.answer("盗窃行为应承担什么责任？")

        self.assertIs(AnswerStatus.PROCESSING_FAILED, result.status)
        self.assertEqual("reranking_failed", result.diagnostic_code)
        self.assertEqual([], generator.calls)

    def test_evidence_packaging_failure_is_reported_without_generation(self):
        generator = RecordingGenerator(error=AssertionError("不应生成"))
        packager = EvidencePackager(
            context_limit=200,
            max_output_tokens=RAG_MAX_OUTPUT_TOKENS,
            count_prompt_tokens=lambda package: 60,
        )
        rag = self.make_rag(
            retriever=FakeSemanticRetriever((make_ranked(self.article),)),
            generator=generator,
            external_llm=RecordingExternalLLM(assessment_json()),
            evidence_packager=packager,
        )

        result = rag.answer("盗窃行为应承担什么责任？")

        self.assertIs(AnswerStatus.PROCESSING_FAILED, result.status)
        self.assertEqual("evidence_packaging_failed", result.diagnostic_code)
        self.assertEqual([], generator.calls)

    def test_generation_and_protocol_failures_remain_processing_failures(self):
        cases = (
            (RecordingGenerator(error=RuntimeError("生成失败")), "generation_failed"),
            (RecordingGenerator("not-json"), "output_validation_failed"),
            (
                RecordingGenerator(answer_json(citations=("E2",))),
                "output_validation_failed",
            ),
        )
        for generator, diagnostic_code in cases:
            with self.subTest(diagnostic_code=diagnostic_code):
                rag = self.make_rag(
                    retriever=FakeSemanticRetriever((make_ranked(self.article),)),
                    generator=generator,
                    external_llm=RecordingExternalLLM(assessment_json()),
                )

                result = rag.answer("盗窃行为应承担什么责任？")

                self.assertIs(AnswerStatus.PROCESSING_FAILED, result.status)
                self.assertEqual(diagnostic_code, result.diagnostic_code)

    def test_simple_questions_can_be_accepted_without_clarification(self):
        for query in (
            "试用期最多是多久？",
            "民事法律行为有效需要满足什么条件？",
        ):
            with self.subTest(query=query):
                rag = self.make_rag(
                    retriever=FakeSemanticRetriever((make_ranked(self.article),)),
                    generator=RecordingGenerator(answer_json()),
                    external_llm=RecordingExternalLLM(assessment_json()),
                )

                result = rag.answer(query)

                self.assertIs(AnswerStatus.RETRIEVED_EVIDENCE, result.status)


if __name__ == "__main__":
    unittest.main(verbosity=2)
