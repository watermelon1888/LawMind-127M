"""连接现行法律查询、检索、证据协议与中文渲染。"""

import logging

from rag.answering import (
    AnswerProtocolError,
    EvidencePackager,
    RAG_MAX_OUTPUT_TOKENS,
    build_answer_prompt,
    parse_and_validate_answer,
    render_clarification,
    render_exact_lookup,
    render_failure,
    render_refusal,
    render_semantic_answer,
)
from rag.core.contracts import (
    AnswerMode,
    AnswerStatus,
    BusinessRoute,
    Evidence,
    LegalRAG,
    LegalRAGResult,
    QueryEnhancementFailureReason,
    QueryEnhancementStatus,
    QueryEnhancementTrace,
    UnansweredReason,
)
from rag.core.exact_lookup import ExactLookupStatus, resolve_exact_lookup
from rag.core.semantic_search import resolve_semantic_search
from rag.knowledge import ArticleRepository, LegalArticle
from rag.query import (
    QueryEnhancement,
    QueryEnhancementProtocolError,
    QueryReason,
    QueryRoute,
    compile_retrieval_queries,
    parse_and_validate_query_enhancement,
    route_query,
)
from rag.retrieval import RankedArticle


_LOGGER = logging.getLogger(__name__)


def _to_evidence(article):
    if not isinstance(article, LegalArticle):
        raise TypeError("待转换对象必须是 LegalArticle")
    return Evidence(
        law_name=article.law_name,
        article_no=article.article_no,
        content=article.content,
    )


def _ranked_articles(results):
    ranked = tuple(results)
    if any(not isinstance(item, RankedArticle) for item in ranked):
        raise TypeError("语义检索结果必须由 RankedArticle 组成")
    return ranked


def _unanswered_result(
    *,
    query,
    status,
    reason,
    rendered_answer,
    evidence=(),
    query_enhancement=QueryEnhancementTrace(),
):
    return LegalRAGResult(
        query=query,
        status=status,
        unanswered_reason=reason,
        rendered_answer=rendered_answer,
        evidence=evidence,
        query_enhancement=query_enhancement,
    )


def _processing_failure(
    query,
    diagnostic_code,
    evidence=(),
    query_enhancement=QueryEnhancementTrace(),
):
    return LegalRAGResult(
        query=query,
        status=AnswerStatus.PROCESSING_FAILED,
        unanswered_reason=UnansweredReason.PROCESSING_FAILED,
        diagnostic_code=diagnostic_code,
        evidence=evidence,
        rendered_answer=render_failure(diagnostic_code),
        query_enhancement=query_enhancement,
    )


class CurrentLawRAG(LegalRAG):
    """仅依据当前知识库中现行有效法条作答的总编排器。"""

    def __init__(
        self,
        *,
        article_repository,
        semantic_retriever,
        evidence_packager,
        generate,
        query_enhancer=None,
        max_output_tokens=RAG_MAX_OUTPUT_TOKENS,
    ):
        if not isinstance(article_repository, ArticleRepository):
            raise TypeError("article_repository 必须是 ArticleRepository")
        if not callable(getattr(semantic_retriever, "search", None)):
            raise TypeError("semantic_retriever 必须提供可调用的 search")
        if not isinstance(evidence_packager, EvidencePackager):
            raise TypeError("evidence_packager 必须是 EvidencePackager")
        if not callable(generate):
            raise TypeError("generate 必须可调用")
        if query_enhancer is not None:
            if not callable(query_enhancer):
                raise TypeError("query_enhancer 必须可调用或为 None")
            if not callable(getattr(semantic_retriever, "search_many", None)):
                raise TypeError("启用 Query 增强时 semantic_retriever 必须提供 search_many")
        if (
            not isinstance(max_output_tokens, int)
            or isinstance(max_output_tokens, bool)
            or max_output_tokens <= 0
        ):
            raise ValueError("max_output_tokens 必须是正整数")
        if max_output_tokens != evidence_packager.max_output_tokens:
            raise ValueError("生成输出预算必须与证据包构造预算一致")

        self._article_repository = article_repository
        self._semantic_retriever = semantic_retriever
        self._evidence_packager = evidence_packager
        self._generate = generate
        self._query_enhancer = query_enhancer
        self._max_output_tokens = max_output_tokens

    def answer(self, query):
        """返回基于现行法证据的回答，或带审计原因的安全未作答。"""
        decision = route_query(query)
        if decision.route is BusinessRoute.CLARIFY:
            return _unanswered_result(
                query=decision.query,
                status=AnswerStatus.CLARIFICATION_REQUIRED,
                reason=UnansweredReason.CLARIFICATION_REQUIRED,
                rendered_answer=render_clarification(),
            )
        if decision.route is BusinessRoute.GENERAL_CHAT:
            return _unanswered_result(
                query=decision.query,
                status=AnswerStatus.REFUSED,
                reason=UnansweredReason.NON_LEGAL,
                rendered_answer=render_refusal(UnansweredReason.NON_LEGAL),
            )
        if decision.route is QueryRoute.REFUSE:
            reason = {
                QueryReason.NON_LEGAL: UnansweredReason.NON_LEGAL,
                QueryReason.TIME_SENSITIVE: UnansweredReason.TIME_SENSITIVE,
                QueryReason.UNSUPPORTED_LEGAL_SOURCE: (
                    UnansweredReason.UNSUPPORTED_LEGAL_SOURCE
                ),
                QueryReason.UNSUPPORTED_LEGAL_TASK: (
                    UnansweredReason.UNSUPPORTED_LEGAL_TASK
                ),
            }[decision.reason]
            return _unanswered_result(
                query=decision.query,
                status=AnswerStatus.REFUSED,
                reason=reason,
                rendered_answer=render_refusal(reason),
            )
        if (
            decision.route is BusinessRoute.ANSWER
            and decision.answer_mode is AnswerMode.EXACT_LOOKUP
        ):
            return self._answer_exact_lookup(decision)
        if (
            decision.route is BusinessRoute.ANSWER
            and decision.answer_mode is AnswerMode.RETRIEVAL
        ):
            return self._answer_semantic_search(decision)
        return _processing_failure(
            decision.query,
            "unsupported_route_decision",
        )

    def _answer_exact_lookup(self, decision):
        try:
            resolution = resolve_exact_lookup(decision, self._article_repository)
        except Exception:
            return _processing_failure(decision.query, "exact_lookup_failed")
        if resolution.status is not ExactLookupStatus.FOUND:
            return _unanswered_result(
                query=decision.query,
                status=AnswerStatus.CLARIFICATION_REQUIRED,
                reason=UnansweredReason.CLARIFICATION_REQUIRED,
                rendered_answer=render_clarification(),
            )
        evidence = tuple(_to_evidence(item) for item in resolution.articles)
        return LegalRAGResult(
            query=decision.query,
            status=AnswerStatus.VERIFIED_LOOKUP,
            evidence=evidence,
            rendered_answer=render_exact_lookup(evidence),
        )

    def _answer_semantic_search(self, decision):
        baseline_query = compile_retrieval_queries(
            decision.query,
            QueryEnhancement(rewrite=decision.query),
        )[0]
        trace = QueryEnhancementTrace(
            retrieval_queries=(baseline_query,),
        )
        if self._query_enhancer is not None:
            try:
                raw_enhancement = self._query_enhancer(decision.query)
            except TimeoutError:
                _LOGGER.warning("Query 增强调用超时", exc_info=True)
                trace = QueryEnhancementTrace(
                    status=QueryEnhancementStatus.FALLBACK,
                    failure_reason=QueryEnhancementFailureReason.TIMEOUT,
                    retrieval_queries=(baseline_query,),
                )
            except Exception:
                _LOGGER.warning("Query 增强调用失败", exc_info=True)
                trace = QueryEnhancementTrace(
                    status=QueryEnhancementStatus.FALLBACK,
                    failure_reason=QueryEnhancementFailureReason.CALL_FAILED,
                    retrieval_queries=(baseline_query,),
                )
            else:
                try:
                    enhancement = parse_and_validate_query_enhancement(
                        raw_enhancement
                    )
                    retrieval_queries = compile_retrieval_queries(
                        decision.query,
                        enhancement,
                    )
                except (QueryEnhancementProtocolError, TypeError, ValueError):
                    _LOGGER.warning("Query 增强输出协议不合法", exc_info=True)
                    trace = QueryEnhancementTrace(
                        status=QueryEnhancementStatus.FALLBACK,
                        failure_reason=(
                            QueryEnhancementFailureReason.INVALID_OUTPUT
                        ),
                        retrieval_queries=(baseline_query,),
                    )
                else:
                    trace = QueryEnhancementTrace(
                        status=QueryEnhancementStatus.APPLIED,
                        retrieval_queries=retrieval_queries,
                    )

        try:
            if trace.status is QueryEnhancementStatus.APPLIED:
                try:
                    ranked = _ranked_articles(
                        self._semantic_retriever.search_many(
                            decision.query,
                            trace.retrieval_queries,
                        )
                    )
                except Exception:
                    _LOGGER.warning("增强检索失败，回退原始 query", exc_info=True)
                    trace = QueryEnhancementTrace(
                        status=QueryEnhancementStatus.FALLBACK,
                        failure_reason=(
                            QueryEnhancementFailureReason.ENHANCED_RETRIEVAL_FAILED
                        ),
                        retrieval_queries=(baseline_query,),
                    )
                    ranked = _ranked_articles(
                        resolve_semantic_search(
                            decision,
                            self._semantic_retriever,
                            baseline_query,
                        )
                    )
            else:
                ranked = _ranked_articles(
                    resolve_semantic_search(
                        decision,
                        self._semantic_retriever,
                        baseline_query,
                    )
                )
        except Exception:
            return _processing_failure(
                decision.query,
                "semantic_retrieval_failed",
                query_enhancement=trace,
            )
        if not ranked:
            reason = UnansweredReason.NO_VERIFIABLE_EVIDENCE
            return _unanswered_result(
                query=decision.query,
                status=AnswerStatus.REFUSED,
                reason=reason,
                rendered_answer=render_refusal(reason),
                query_enhancement=trace,
            )

        try:
            package, _prompt_tokens = self._evidence_packager.build(
                decision.query, tuple(item.article for item in ranked)
            )
        except Exception:
            return _processing_failure(
                decision.query,
                "evidence_packaging_failed",
                query_enhancement=trace,
            )

        evidence = package.evidence
        try:
            raw_text = self._generate(
                build_answer_prompt(package),
                temperature=0,
                max_tokens=self._max_output_tokens,
            )
        except Exception:
            return _processing_failure(
                decision.query,
                "generation_failed",
                evidence=evidence,
                query_enhancement=trace,
            )
        try:
            model_answer = parse_and_validate_answer(package, raw_text)
        except (AnswerProtocolError, TypeError):
            return _processing_failure(
                decision.query,
                "output_validation_failed",
                evidence=evidence,
                query_enhancement=trace,
            )

        return LegalRAGResult(
            query=decision.query,
            status=AnswerStatus.RETRIEVED_EVIDENCE,
            evidence=evidence,
            model_answer=model_answer,
            rendered_answer=render_semantic_answer(package, model_answer),
            query_enhancement=trace,
        )


__all__ = ["CurrentLawRAG"]
