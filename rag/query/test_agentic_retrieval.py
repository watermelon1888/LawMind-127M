"""证据助理工具协议测试。"""

import json
import unittest
from types import SimpleNamespace

from rag.external import ExternalLLMResponse, ExternalToolCall
from rag.knowledge import ArticleRepository, LegalArticle, split_article
from rag.query.agentic_retrieval import (
    AgenticRetrievalError,
    run_agentic_retrieval,
    validate_supplemental_query,
)


def make_article(index):
    return LegalArticle(
        chunk_id=f"示例法#{index}",
        law_name="示例法",
        article_no=str(index),
        content=f"第{index}条 示例正文。",
    )


def make_repository(articles):
    articles = tuple(articles)
    formal_law_names = {item.law_name for item in articles}
    return ArticleRepository(
        {(item.law_name, item.article_no): item for item in articles},
        {law_name: {law_name} for law_name in formal_law_names},
        formal_law_names,
    )


def tool_call(call_id, name, payload):
    return ExternalToolCall(
        call_id,
        name,
        json.dumps(payload, ensure_ascii=False),
    )


class _Retriever:
    def __init__(self, results_by_query=None):
        self.results_by_query = results_by_query or {}
        self.calls = []

    def retrieve_candidates(self, query):
        self.calls.append(query)
        return tuple(
            SimpleNamespace(chunk_id=chunk_id)
            for chunk_id in self.results_by_query.get(query, ())
        )


class _External:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate_with_tools(self, messages, *, tools, temperature, max_tokens):
        self.calls.append((messages, tools, temperature, max_tokens))
        return next(self.responses)


class AgenticRetrievalTests(unittest.TestCase):
    def test_finalize_promotes_hybrid_candidate_behind_reranker_top_five(self):
        articles = tuple(make_article(index) for index in range(1, 7))
        external = _External(
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

        selection = run_agentic_retrieval(
            query="示例问题如何处理？",
            baseline=articles,
            reranked=articles[:5],
            retriever=_Retriever(),
            article_repository=make_repository(articles),
            external_llm=external,
        )

        self.assertEqual(
            (articles[0], articles[5], articles[1], articles[2], articles[3]),
            selection.articles,
        )
        self.assertEqual((articles[5].chunk_id,), selection.promoted_chunk_ids)
        self.assertEqual(("示例问题如何处理？",), selection.retrieval_queries)
        self.assertEqual((), selection.searches)
        tool_names = {
            item["function"]["name"] for item in external.calls[0][1]
        }
        self.assertEqual({"search_law", "finalize_evidence"}, tool_names)
        system_prompt = external.calls[0][0][0]["content"]
        self.assertIn("零至五个", system_prompt)
        self.assertIn("Cross-Encoder", system_prompt)
        self.assertIn("检查全部 Hybrid top-20", system_prompt)
        self.assertIn("先提升遗漏证据，不调用 search_law", system_prompt)
        payload = json.loads(external.calls[0][0][1]["content"])
        self.assertEqual(
            [item.chunk_id for item in articles[:5]],
            payload["reranker_top5"],
        )
        self.assertEqual(
            [1, 2, 3, 4, 5, 6],
            [item["hybrid_rank"] for item in payload["observed_candidates"]],
        )
        self.assertEqual(
            [1, 2, 3, 4, 5, None],
            [item["reranker_rank"] for item in payload["observed_candidates"]],
        )
        self.assertEqual(
            [item.chunk_id for item in articles],
            payload["allowed_promoted_chunk_ids"],
        )
        self.assertIn("明确引用某法某条", system_prompt)
        self.assertIn("必须同时包含该法名和条号", system_prompt)

    def test_search_adds_observed_candidates_before_commit(self):
        baseline = make_article(1)
        supplemental = make_article(2)
        query = "示例问题如何处理？"
        rewrite = "示例问题处理依据"
        external = _External(
            (
                ExternalLLMResponse(
                    content=None,
                    tool_calls=(
                        tool_call("search-1", "search_law", {"query": rewrite}),
                    ),
                ),
                ExternalLLMResponse(
                    content=None,
                    tool_calls=(
                        tool_call(
                            "finalize-1",
                            "finalize_evidence",
                            {"promoted_chunk_ids": [supplemental.chunk_id]},
                        ),
                    ),
                ),
            )
        )
        retriever = _Retriever({rewrite: (supplemental.chunk_id,)})

        selection = run_agentic_retrieval(
            query=query,
            baseline=(baseline,),
            reranked=(baseline,),
            retriever=retriever,
            article_repository=make_repository((baseline, supplemental)),
            external_llm=external,
        )

        self.assertEqual((baseline, supplemental), selection.articles)
        self.assertEqual(
            (supplemental.chunk_id,), selection.promoted_chunk_ids
        )
        self.assertEqual((query, rewrite), selection.retrieval_queries)
        self.assertEqual([rewrite], retriever.calls)
        self.assertEqual(rewrite, selection.searches[0].query)
        self.assertEqual((supplemental.chunk_id,), selection.searches[0].chunk_ids)
        system_prompt = external.calls[0][0][0]["content"]
        self.assertIn("补充检索后仍必须调用 finalize_evidence", system_prompt)
        tool_observation = json.loads(external.calls[1][0][-1]["content"])
        self.assertEqual(
            [baseline.chunk_id, supplemental.chunk_id],
            tool_observation["allowed_promoted_chunk_ids"],
        )

    def test_explicit_article_search_uses_repository_and_requires_promotion(self):
        law_name = "国境口岸突发公共卫生事件出入境检验检疫应急处理规定"
        baseline = LegalArticle(
            chunk_id=f"{law_name}#14",
            law_name=law_name,
            article_no="14",
            content="有本规定第二条规定情形之一的，应当在一小时内报告。",
        )
        target = LegalArticle(
            chunk_id=f"{law_name}#2",
            law_name=law_name,
            article_no="2",
            content="本规定所称突发公共卫生事件包括若干情形。",
        )
        supplemental_query = f"{law_name} 第二条 情形"
        retriever = _Retriever()
        external = _External(
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
                            {"promoted_chunk_ids": [target.chunk_id]},
                        ),
                    ),
                ),
            )
        )

        selection = run_agentic_retrieval(
            query="哪些情况属于国境口岸突发公共卫生事件，多久要上报？",
            baseline=(baseline,),
            reranked=(baseline,),
            retriever=retriever,
            article_repository=make_repository((baseline, target)),
            external_llm=external,
        )

        self.assertEqual([], retriever.calls)
        self.assertEqual((target.chunk_id,), selection.searches[0].chunk_ids)
        self.assertIn(target.chunk_id, selection.selected_chunk_ids)
        observation = json.loads(external.calls[1][0][-1]["content"])
        self.assertEqual("exact_lookup", observation["search_mode"])
        self.assertEqual(
            [target.chunk_id],
            observation["required_promoted_chunk_ids"],
        )

    def test_finalize_rejects_omitted_explicit_article_hit(self):
        law_name = "国境口岸突发公共卫生事件出入境检验检疫应急处理规定"
        baseline = LegalArticle(
            chunk_id=f"{law_name}#14",
            law_name=law_name,
            article_no="14",
            content="有本规定第二条规定情形之一的，应当在一小时内报告。",
        )
        target = LegalArticle(
            chunk_id=f"{law_name}#2",
            law_name=law_name,
            article_no="2",
            content="本规定所称突发公共卫生事件包括若干情形。",
        )
        supplemental_query = f"{law_name} 第二条 情形"
        external = _External(
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
                            {"promoted_chunk_ids": []},
                        ),
                    ),
                ),
            )
        )

        with self.assertRaisesRegex(AgenticRetrievalError, "精确查条命中"):
            run_agentic_retrieval(
                query="哪些情况属于国境口岸突发公共卫生事件，多久要上报？",
                baseline=(baseline,),
                reranked=(baseline,),
                retriever=_Retriever(),
                article_repository=make_repository((baseline, target)),
                external_llm=external,
            )

    def test_finalize_rejects_unresolved_explicit_article_miss(self):
        law_name = "国境口岸突发公共卫生事件出入境检验检疫应急处理规定"
        baseline = LegalArticle(
            chunk_id=f"{law_name}#14",
            law_name=law_name,
            article_no="14",
            content="有本规定第二条规定情形之一的，应当在一小时内报告。",
        )
        supplemental_query = f"{law_name} 第二条 情形"
        external = _External(
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
                            {"promoted_chunk_ids": [baseline.chunk_id]},
                        ),
                    ),
                ),
            )
        )
        retriever = _Retriever({supplemental_query: (baseline.chunk_id,)})

        with self.assertRaisesRegex(AgenticRetrievalError, "精确查条未命中"):
            run_agentic_retrieval(
                query="哪些情况属于国境口岸突发公共卫生事件，多久要上报？",
                baseline=(baseline,),
                reranked=(baseline,),
                retriever=retriever,
                article_repository=make_repository((baseline,)),
                external_llm=external,
            )

        self.assertEqual([], retriever.calls)
        observation = json.loads(external.calls[1][0][-1]["content"])
        self.assertEqual("exact_lookup_miss", observation["search_mode"])
        self.assertEqual(
            [f"{law_name}#第二条"],
            observation["exact_lookup_misses"],
        )

    def test_free_text_stop_is_rejected_without_commit(self):
        article = make_article(1)
        with self.assertRaises(AgenticRetrievalError):
            run_agentic_retrieval(
                query="示例问题如何处理？",
                baseline=(article,),
                reranked=(article,),
                retriever=_Retriever(),
                article_repository=make_repository((article,)),
                external_llm=_External((ExternalLLMResponse(content="STOP"),)),
            )

    def test_commit_rejects_unobserved_chunk_id(self):
        article = make_article(1)
        with self.assertRaises(AgenticRetrievalError):
            run_agentic_retrieval(
                query="示例问题如何处理？",
                baseline=(article,),
                reranked=(article,),
                retriever=_Retriever(),
                article_repository=make_repository((article,)),
                external_llm=_External(
                    (
                        ExternalLLMResponse(
                            content=None,
                            tool_calls=(
                                tool_call(
                                    "finalize-1",
                                    "finalize_evidence",
                                    {"promoted_chunk_ids": ["示例法#999"]},
                                ),
                            ),
                        ),
                    )
                ),
            )

    def test_finalize_empty_keeps_reranker_top_five(self):
        articles = tuple(make_article(index) for index in range(1, 6))
        selection = run_agentic_retrieval(
            query="示例问题如何处理？",
            baseline=articles,
            reranked=articles,
            retriever=_Retriever(),
            article_repository=make_repository(articles),
            external_llm=_External(
                (
                    ExternalLLMResponse(
                        content=None,
                        tool_calls=(
                            tool_call(
                                "finalize-1",
                                "finalize_evidence",
                                {"promoted_chunk_ids": []},
                            ),
                        ),
                    ),
                )
            ),
        )

        self.assertEqual(articles, selection.articles)
        self.assertEqual((), selection.promoted_chunk_ids)
        self.assertEqual((), selection.selected_unit_ids)

    def test_finalize_can_select_only_observed_units_of_final_parents(self):
        article = LegalArticle(
            chunk_id="示例法#1",
            law_name="示例法",
            article_no="1",
            content="处理原则如下：\n（一）先行告知；\n（二）拒不改正的，处一万元罚款。",
        )
        units = split_article(article)
        selected = units[-1]
        external = _External(
            (
                ExternalLLMResponse(
                    content=None,
                    tool_calls=(
                        tool_call(
                            "finalize-1",
                            "finalize_evidence",
                            {
                                "promoted_chunk_ids": [],
                                "selected_unit_ids": [selected.unit_id],
                            },
                        ),
                    ),
                ),
            )
        )

        selection = run_agentic_retrieval(
            query="拒不改正怎么处罚？",
            baseline=(article,),
            reranked=(article,),
            retriever=_Retriever(),
            article_repository=make_repository((article,)),
            external_llm=external,
            units_by_parent={article.chunk_id: units},
        )

        self.assertEqual((selected.unit_id,), selection.selected_unit_ids)
        payload = json.loads(external.calls[0][0][1]["content"])
        self.assertEqual(
            [unit.unit_id for unit in units],
            payload["allowed_selected_unit_ids"],
        )
        self.assertEqual(
            [unit.unit_id for unit in units],
            [
                item["unit_id"]
                for item in payload["observed_candidates"][0]["evidence_units"]
            ],
        )

    def test_finalize_rejects_unobserved_unit(self):
        article = make_article(1)
        with self.assertRaisesRegex(AgenticRetrievalError, "未观察到的 unit_id"):
            run_agentic_retrieval(
                query="示例问题如何处理？",
                baseline=(article,),
                reranked=(article,),
                retriever=_Retriever(),
                article_repository=make_repository((article,)),
                external_llm=_External(
                    (
                        ExternalLLMResponse(
                            content=None,
                            tool_calls=(
                                tool_call(
                                    "finalize-1",
                                    "finalize_evidence",
                                    {
                                        "promoted_chunk_ids": [],
                                        "selected_unit_ids": ["不存在的单元"],
                                    },
                                ),
                            ),
                        ),
                    )
                ),
            )

    def test_finalize_five_promotions_keeps_reranker_top_two_and_accepts_three_new(self):
        articles = tuple(make_article(index) for index in range(1, 11))
        promoted = (articles[9], articles[8], articles[7], articles[6], articles[5])
        selection = run_agentic_retrieval(
            query="示例问题如何处理？",
            baseline=articles,
            reranked=articles[:5],
            retriever=_Retriever(),
            article_repository=make_repository(articles),
            external_llm=_External(
                (
                    ExternalLLMResponse(
                        content=None,
                        tool_calls=(
                            tool_call(
                                "finalize-1",
                                "finalize_evidence",
                                {
                                    "promoted_chunk_ids": [
                                        item.chunk_id for item in promoted
                                    ]
                                },
                            ),
                        ),
                    ),
                )
            ),
        )

        self.assertEqual(
            (articles[0], articles[9], articles[8], articles[7], articles[1]),
            selection.articles,
        )
        self.assertEqual(
            tuple(item.chunk_id for item in promoted),
            selection.promoted_chunk_ids,
        )

    def test_third_search_is_rejected(self):
        article = make_article(1)
        responses = tuple(
            ExternalLLMResponse(
                content=None,
                tool_calls=(
                    tool_call(
                        f"search-{index}",
                        "search_law",
                        {"query": f"示例问题处理依据{index}"},
                    ),
                ),
            )
            for index in range(1, 4)
        )
        with self.assertRaises(AgenticRetrievalError):
            run_agentic_retrieval(
                query="示例问题如何处理？",
                baseline=(article,),
                reranked=(article,),
                retriever=_Retriever(),
                article_repository=make_repository((article,)),
                external_llm=_External(responses),
            )

    def test_query_boundary_rejects_historical_expansion(self):
        with self.assertRaises(AgenticRetrievalError):
            validate_supplemental_query(
                "盗窃行为应承担什么责任？",
                "盗窃行为历史版本废止前如何规定",
            )


if __name__ == "__main__":
    unittest.main()
