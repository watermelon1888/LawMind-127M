"""连接现行法律查询、检索、证据协议与中文渲染。"""

import hashlib
import uuid
from dataclasses import replace

from rag.answering import (
    AnswerProtocolError,
    EvidenceBundlePackager,
    EvidencePackager,
    RAG_MAX_OUTPUT_TOKENS,
    build_answer_prompt,
    parse_and_validate_answer,
    render_clarification,
    render_exact_lookup,
    render_failure,
    render_general_chat,
    render_refusal,
    render_semantic_answer,
)
from rag.core.contracts import (
    AuditEvent,
    AuditTrace,
    AnswerMode,
    AnswerStatus,
    BusinessRoute,
    Evidence,
    LegalRAG,
    LegalRAGResult,
    QueryEnhancementStatus,
    QueryEnhancementTrace,
    UnansweredReason,
)
from rag.core.exact_lookup import ExactLookupStatus, resolve_exact_lookup
from rag.knowledge import ArticleRepository, LegalArticle
from rag.query import (
    QueryAssessmentProtocolError,
    QueryReason,
    RouteDecision,
    assess_query,
    route_query,
    run_agentic_retrieval,
)


GENERAL_CHAT_MAX_OUTPUT_TOKENS = 512
_GENERAL_CHAT_SYSTEM_PROMPT = (
    "你是通用对话助手。请直接、简洁地回答用户问题，不要输出法律引用、法律结论或法律任务分类。"
)


def _to_evidence(article):
    if not isinstance(article, LegalArticle):
        raise TypeError("待转换对象必须是 LegalArticle")
    return Evidence(
        law_name=article.law_name,
        article_no=article.article_no,
        content=article.content,
    )


def _candidate_articles(results, repository):
    """将 Hybrid 候选按原顺序映射为 canonical 完整法条。"""
    candidates = tuple(results)
    articles = []
    seen = set()
    for item in candidates:
        chunk_id = getattr(item, "chunk_id", None)
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise TypeError("Hybrid 候选必须携带有效 chunk_id")
        if chunk_id in seen:
            raise ValueError("Hybrid 候选不能包含重复 chunk_id")
        seen.add(chunk_id)
        articles.append(repository.get_by_chunk_id(chunk_id))
    return tuple(articles)


def _reranked_articles(results, repository):
    """将 Cross-Encoder 结果映射为 canonical 完整法条。"""
    articles = []
    seen = set()
    for item in tuple(results):
        article = getattr(item, "article", None)
        if not isinstance(article, LegalArticle):
            chunk_id = getattr(item, "chunk_id", None)
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise TypeError("精排结果必须携带有效法条或 chunk_id")
            article = repository.get_by_chunk_id(chunk_id)
        if article.chunk_id in seen:
            raise ValueError("精排结果不能包含重复 chunk_id")
        seen.add(article.chunk_id)
        articles.append(article)
    return tuple(articles)


def _candidate_units_by_parent(results):
    """提取父子检索候选中已观察的显式子单元。"""
    mapping = {}
    for item in tuple(results):
        units = tuple(getattr(item, "units", ()))
        if units:
            mapping[item.chunk_id] = units
    return mapping


def _audit_hit(article):
    """投影审计界面所需的法条身份，不复制正文。"""
    if not isinstance(article, LegalArticle):
        raise TypeError("审计命中必须是 LegalArticle")
    return {
        "law_name": article.law_name,
        "article_no": article.article_no,
        "source_type": article.source_type,
    }


def _retrieval_audit_metadata(retriever):
    describe = getattr(retriever, "audit_metadata", None)
    if not callable(describe):
        return None
    metadata = describe()
    if not isinstance(metadata, dict):
        raise TypeError("semantic_retriever.audit_metadata 必须返回字典")
    return {
        key: value
        for key, value in metadata.items()
        if key not in {"reranker_model", "reranker_top_k"}
    }


def _unanswered_result(
    *,
    query,
    status,
    reason,
    rendered_answer,
    evidence=(),
    candidate_answer=None,
    query_enhancement=QueryEnhancementTrace(),
):
    return LegalRAGResult(
        query=query,
        status=status,
        unanswered_reason=reason,
        rendered_answer=rendered_answer,
        evidence=evidence,
        candidate_answer=candidate_answer,
        query_enhancement=query_enhancement,
    )


def _processing_failure(
    query,
    diagnostic_code,
    evidence=(),
    candidate_answer=None,
    query_enhancement=QueryEnhancementTrace(),
):
    return LegalRAGResult(
        query=query,
        status=AnswerStatus.PROCESSING_FAILED,
        unanswered_reason=UnansweredReason.PROCESSING_FAILED,
        diagnostic_code=diagnostic_code,
        evidence=evidence,
        candidate_answer=candidate_answer,
        rendered_answer=render_failure(diagnostic_code),
        query_enhancement=query_enhancement,
    )


class _AuditRecorder:
    """收集单次请求的有序审计事件，避免业务分支直接拼装审计结构。"""

    def __init__(self, query):
        self.trace_id = uuid.uuid4().hex
        self.events = []
        query_bytes = (
            query.encode("utf-8") if isinstance(query, str) else repr(query).encode()
        )
        self.add(
            "request_received",
            "succeeded",
            source="core",
            details={
                "query_sha256": hashlib.sha256(query_bytes).hexdigest(),
                "query_length": len(query) if isinstance(query, str) else None,
            },
        )

    def add(self, stage, status, *, source=None, reason=None, details=None):
        self.events.append(
            AuditEvent(
                stage=stage,
                status=status,
                source=source,
                reason=reason,
                details={} if details is None else details,
            )
        )

    def finish(self, route, status):
        self.add(
            "completed",
            "succeeded",
            source="core",
            details={"final_route": route.value, "final_status": status.value},
        )
        return AuditTrace(
            trace_id=self.trace_id,
            events=tuple(self.events),
            final_route=route,
            final_status=status,
        )


def _finalize_result(result, audit, route):
    if not isinstance(result, LegalRAGResult):
        return result
    audit.add(
        "final_render",
        "succeeded",
        source="answering",
        details={
            "status": result.status.value,
            "has_message": result.rendered_answer.message is not None,
            "evidence_count": len(result.rendered_answer.evidence),
        },
    )
    return replace(result, audit_trace=audit.finish(route, result.status))


def _render_fixed_clarification(audit, reason):
    """记录固定澄清并返回统一模板。"""
    audit.add(
        "clarification",
        "fallback",
        source="fallback",
        reason=reason,
    )
    return render_clarification()


class CurrentLawRAG(LegalRAG):
    """仅依据当前知识库中现行有效法条作答的总编排器。"""

    def __init__(
        self,
        *,
        article_repository,
        semantic_retriever,
        evidence_packager,
        generate,
        external_llm=None,
        max_output_tokens=RAG_MAX_OUTPUT_TOKENS,
    ):
        if not isinstance(article_repository, ArticleRepository):
            raise TypeError("article_repository 必须是 ArticleRepository")
        if not callable(getattr(semantic_retriever, "retrieve_candidates", None)):
            raise TypeError("semantic_retriever 必须提供可调用的 retrieve_candidates")
        if not callable(getattr(semantic_retriever, "rerank_candidates", None)):
            raise TypeError("semantic_retriever 必须提供可调用的 rerank_candidates")
        if not isinstance(
            evidence_packager,
            (EvidencePackager, EvidenceBundlePackager),
        ):
            raise TypeError(
                "evidence_packager 必须是 EvidencePackager 或 EvidenceBundlePackager"
            )
        if not callable(generate):
            raise TypeError("generate 必须可调用")
        if external_llm is not None and not callable(
            getattr(external_llm, "generate", None)
        ):
            raise TypeError("external_llm 必须提供可调用的 generate 或为 None")
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
        self._external_llm = external_llm
        self._max_output_tokens = max_output_tokens

    def answer(self, query):
        """确定性路由后先判断是否需要澄清，再执行受控法律回答。"""
        audit = _AuditRecorder(query)
        decision = route_query(query)
        audit.add(
            "route_decided",
            "succeeded",
            source="deterministic",
            reason=getattr(decision, "reason", None),
            details={
                "route": decision.route.value,
                "answer_mode": (
                    None if decision.answer_mode is None else decision.answer_mode.value
                ),
            },
        )
        if decision.route is BusinessRoute.REFUSE:
            reason = {
                QueryReason.TIME_SENSITIVE: UnansweredReason.TIME_SENSITIVE,
                QueryReason.UNSUPPORTED_LEGAL_SOURCE: (
                    UnansweredReason.UNSUPPORTED_LEGAL_SOURCE
                ),
                QueryReason.UNSUPPORTED_LEGAL_TASK: (
                    UnansweredReason.UNSUPPORTED_LEGAL_TASK
                ),
            }[QueryReason(decision.reason)]
            result = _unanswered_result(
                query=decision.query,
                status=AnswerStatus.REFUSED,
                reason=reason,
                rendered_answer=render_refusal(reason),
            )
            return _finalize_result(result, audit, BusinessRoute.REFUSE)
        if decision.route is BusinessRoute.CLARIFY:
            rendered_clarification = _render_fixed_clarification(
                audit, decision.reason or "input_requires_clarification"
            )
            result = _unanswered_result(
                query=decision.query,
                status=AnswerStatus.CLARIFICATION_REQUIRED,
                reason=UnansweredReason.CLARIFICATION_REQUIRED,
                rendered_answer=rendered_clarification,
            )
            return _finalize_result(result, audit, BusinessRoute.CLARIFY)
        if decision.route is BusinessRoute.GENERAL_CHAT:
            return _finalize_result(
                self._answer_general_chat(decision, audit),
                audit,
                BusinessRoute.GENERAL_CHAT,
            )
        if (
            decision.route is BusinessRoute.ANSWER
            and decision.answer_mode is AnswerMode.EXACT_LOOKUP
        ):
            result = self._answer_exact_lookup(decision, audit)
            final_route = (
                BusinessRoute.CLARIFY
                if result.status is AnswerStatus.CLARIFICATION_REQUIRED
                else BusinessRoute.ANSWER
            )
            return _finalize_result(result, audit, final_route)
        if (
            decision.route is BusinessRoute.ANSWER
            and decision.answer_mode is AnswerMode.RETRIEVAL
        ):
            external_llm = getattr(self, "_external_llm", None)
            try:
                assessment = assess_query(decision.query, external_llm)
            except Exception as error:
                error_code = getattr(error, "code", None)
                if hasattr(error_code, "value"):
                    error_code = error_code.value
                if error_code is None and isinstance(
                    error, QueryAssessmentProtocolError
                ):
                    error_code = "invalid_response"
                audit.add(
                    "query_assessment",
                    "failed",
                    source="external_llm",
                    reason="query_assessment_failed",
                    details={
                        "model": getattr(external_llm, "model", None),
                        "error_code": error_code,
                        "error_type": type(error).__name__,
                        "status_code": getattr(error, "status_code", None),
                    },
                )
                return _finalize_result(
                    _processing_failure(
                        decision.query,
                        "query_assessment_failed",
                    ),
                    audit,
                    BusinessRoute.ANSWER,
                )
            audit.add(
                "query_assessment",
                "succeeded",
                source="external_llm",
                details={"decision": assessment.decision},
            )
            if assessment.decision == "clarify":
                audit.add(
                    "clarification",
                    "succeeded",
                    source="external_llm",
                    reason="missing_decisive_information",
                    details={"question": assessment.clarification},
                )
                result = _unanswered_result(
                    query=decision.query,
                    status=AnswerStatus.CLARIFICATION_REQUIRED,
                    reason=UnansweredReason.CLARIFICATION_REQUIRED,
                    rendered_answer=render_clarification(
                        assessment.clarification
                    ),
                )
                return _finalize_result(result, audit, BusinessRoute.CLARIFY)
            result = self._answer_semantic_search(decision, audit)
            final_route = (
                BusinessRoute.CLARIFY
                if result.status is AnswerStatus.CLARIFICATION_REQUIRED
                else BusinessRoute.ANSWER
            )
            return _finalize_result(result, audit, final_route)
        return _finalize_result(_processing_failure(
            decision.query,
            "unsupported_route_decision",
        ), audit, BusinessRoute.CLARIFY)

    def _answer_general_chat(self, decision, audit):
        external_llm = getattr(self, "_external_llm", None)
        if external_llm is None:
            audit.add(
                "general_chat_generation",
                "skipped",
                source="external_llm",
                reason="general_chat_unavailable",
            )
            return _processing_failure(
                decision.query,
                "general_chat_unavailable",
            )
        try:
            raw_text = external_llm.generate(
                [
                    {"role": "system", "content": _GENERAL_CHAT_SYSTEM_PROMPT},
                    {"role": "user", "content": decision.query},
                ],
                temperature=0,
                max_tokens=GENERAL_CHAT_MAX_OUTPUT_TOKENS,
            )
            rendered_answer = render_general_chat(raw_text)
        except Exception as error:
            error_code = getattr(error, "code", None)
            if hasattr(error_code, "value"):
                error_code = error_code.value
            audit.add(
                "general_chat_generation",
                "failed",
                source="external_llm",
                reason="general_chat_failed",
                details={
                    "error_code": error_code,
                    "status_code": getattr(error, "status_code", None),
                },
            )
            return _processing_failure(
                decision.query,
                "general_chat_failed",
            )
        audit.add(
            "general_chat_generation",
            "succeeded",
            source="external_llm",
            details={
                "model": getattr(external_llm, "model", None),
                "output_length": len(raw_text),
            },
        )
        return LegalRAGResult(
            query=decision.query,
            status=AnswerStatus.GENERAL_CHAT,
            rendered_answer=rendered_answer,
        )

    def _answer_exact_lookup(self, decision, audit):
        try:
            resolution = resolve_exact_lookup(decision, self._article_repository)
        except Exception:
            audit.add(
                "exact_lookup",
                "failed",
                source="exact_lookup",
                reason="exact_lookup_failed",
            )
            return _processing_failure(decision.query, "exact_lookup_failed")
        audit.add(
            "exact_lookup",
            "succeeded",
            source="exact_lookup",
            details={
                "status": resolution.status.value,
                "article_count": len(resolution.articles),
            },
        )
        if resolution.status is not ExactLookupStatus.FOUND:
            return _unanswered_result(
                query=decision.query,
                status=AnswerStatus.CLARIFICATION_REQUIRED,
                reason=UnansweredReason.CLARIFICATION_REQUIRED,
                rendered_answer=_render_fixed_clarification(
                    audit, "exact_lookup_not_found"
                ),
            )
        evidence = tuple(_to_evidence(item) for item in resolution.articles)
        audit.add(
            "evidence_selected",
            "succeeded",
            source="exact_lookup",
            details={"article_count": len(evidence)},
        )
        return LegalRAGResult(
            query=decision.query,
            status=AnswerStatus.VERIFIED_LOOKUP,
            evidence=evidence,
            rendered_answer=render_exact_lookup(evidence),
        )

    def _answer_semantic_search(self, decision, audit):
        baseline_query = decision.query
        external_llm = getattr(self, "_external_llm", None)
        trace = QueryEnhancementTrace(
            retrieval_queries=(baseline_query,),
        )

        try:
            candidate_chunks = tuple(
                self._semantic_retriever.retrieve_candidates(baseline_query)
            )
            baseline_articles = _candidate_articles(
                candidate_chunks,
                self._article_repository,
            )
        except Exception:
            audit.add(
                "retrieval",
                "failed",
                source="semantic_retriever",
                reason="semantic_retrieval_failed",
            )
            return _processing_failure(
                decision.query,
                "semantic_retrieval_failed",
                query_enhancement=trace,
            )
        audit.add(
            "retrieval",
            "succeeded",
            source="semantic_retriever",
            details={
                "query_count": 1,
                "candidate_count": len(baseline_articles),
                "chunk_ids": tuple(item.chunk_id for item in baseline_articles),
                "parameters": _retrieval_audit_metadata(
                    self._semantic_retriever
                ),
                "hits": tuple(_audit_hit(item) for item in baseline_articles),
            },
        )
        if not baseline_articles:
            return _unanswered_result(
                query=decision.query,
                status=AnswerStatus.CLARIFICATION_REQUIRED,
                reason=UnansweredReason.CLARIFICATION_REQUIRED,
                rendered_answer=_render_fixed_clarification(
                    audit, "no_retrieval_candidates"
                ),
                query_enhancement=trace,
            )

        try:
            reranked_results = self._semantic_retriever.rerank_candidates(
                baseline_query, candidate_chunks
            )
            reranked_articles = _reranked_articles(
                reranked_results, self._article_repository
            )
            if not reranked_articles:
                raise ValueError("Cross-Encoder 未返回候选")
        except Exception:
            audit.add(
                "reranking",
                "failed",
                source="semantic_retriever",
                reason="reranking_failed",
            )
            return _processing_failure(
                decision.query,
                "reranking_failed",
                query_enhancement=trace,
            )
        audit.add(
            "reranking",
            "succeeded",
            source="semantic_retriever",
            details={
                "candidate_count": len(reranked_articles),
                "chunk_ids": tuple(item.chunk_id for item in reranked_articles),
                "hits": tuple(_audit_hit(item) for item in reranked_articles),
            },
        )

        baseline_chunk_ids = frozenset(
            item.chunk_id for item in baseline_articles
        )
        try:
            selection = run_agentic_retrieval(
                query=baseline_query,
                baseline=baseline_articles,
                reranked=reranked_articles,
                retriever=self._semantic_retriever,
                article_repository=self._article_repository,
                external_llm=external_llm,
                units_by_parent=_candidate_units_by_parent(candidate_chunks),
            )
        except Exception as error:
            error_code = getattr(error, "code", None)
            if hasattr(error_code, "value"):
                error_code = error_code.value
            audit.add(
                "agentic_retrieval",
                "failed",
                source="external_llm",
                reason="agentic_retrieval_failed",
                details={
                    "error_code": error_code,
                    "error_type": type(error).__name__,
                },
            )
            return _processing_failure(
                decision.query,
                "agentic_retrieval_failed",
                query_enhancement=trace,
            )

        selected_articles = selection.articles
        trace = QueryEnhancementTrace(
            status=(
                QueryEnhancementStatus.APPLIED
                if selection.searches
                else QueryEnhancementStatus.NOT_ATTEMPTED
            ),
            retrieval_queries=selection.retrieval_queries,
        )
        tool_calls = [
            {
                "name": "search_law",
                "query": item.query,
                "chunk_ids": item.chunk_ids,
            }
            for item in selection.searches
        ]
        tool_calls.append(
            {
                "name": "finalize_evidence",
                "promoted_chunk_ids": selection.promoted_chunk_ids,
                "selected_unit_ids": selection.selected_unit_ids,
            }
        )
        observed_chunk_ids = baseline_chunk_ids | {
            chunk_id
            for item in selection.searches
            for chunk_id in item.chunk_ids
        }
        reranked_chunk_ids = frozenset(
            item.chunk_id for item in reranked_articles
        )
        audit.add(
            "agentic_retrieval",
            "succeeded",
            source="external_llm",
            details={
                "query_count": len(selection.retrieval_queries),
                "supplemental_count": len(selection.searches),
                "observed_count": len(observed_chunk_ids),
                "selected_count": len(selected_articles),
                "promoted_count": len(selection.promoted_chunk_ids),
                "retrieval_queries": selection.retrieval_queries,
                "reranked_chunk_ids": tuple(
                    item.chunk_id for item in reranked_articles
                ),
                "promoted_chunk_ids": selection.promoted_chunk_ids,
                "selected_unit_ids": selection.selected_unit_ids,
                "selected_chunk_ids": selection.selected_chunk_ids,
                "tool_calls": tool_calls,
                "selected_hits": tuple(
                    _audit_hit(item) for item in selected_articles
                ),
                "added_hits": tuple(
                    _audit_hit(item)
                    for item in selected_articles
                    if item.chunk_id not in reranked_chunk_ids
                ),
                "new_hits": tuple(
                    _audit_hit(item)
                    for item in selected_articles
                    if item.chunk_id not in baseline_chunk_ids
                ),
            },
        )

        try:
            if isinstance(self._evidence_packager, EvidenceBundlePackager):
                packaging_result = self._evidence_packager.build(
                    decision.query,
                    selected_articles,
                    selected_unit_ids=selection.selected_unit_ids,
                )
            else:
                packaging_result = self._evidence_packager.build(
                    decision.query, selected_articles
                )
            if len(packaging_result) == 2:
                package, prompt_tokens = packaging_result
                bundles = ()
            else:
                package, prompt_tokens, bundles = packaging_result
        except Exception:
            audit.add(
                "evidence_packaging",
                "failed",
                source="evidence_packager",
                reason="evidence_packaging_failed",
            )
            return _processing_failure(
                decision.query,
                "evidence_packaging_failed",
                query_enhancement=trace,
            )

        evidence = package.display_evidence
        evidence_keys = frozenset(
            (item.law_name, item.article_no) for item in evidence
        )
        audit.add(
            "evidence_packaging",
            "succeeded",
            source="evidence_packager",
            details={
                "evidence_count": len(evidence),
                "prompt_tokens": prompt_tokens,
                "context_budget_tokens": (
                    self._evidence_packager.context_limit
                    - self._evidence_packager.max_output_tokens
                ),
                "context_limit": self._evidence_packager.context_limit,
                "reserved_output_tokens": self._evidence_packager.max_output_tokens,
                "hits": tuple(
                    _audit_hit(item)
                    for item in selected_articles
                    if (item.law_name, item.article_no) in evidence_keys
                ),
                "bundles": tuple(
                    {
                        "chunk_id": bundle.article.chunk_id,
                        "unit_ids": bundle.unit_ids,
                        "content_chars": len(bundle.content),
                    }
                    for bundle in bundles
                ),
            },
        )
        try:
            raw_text = self._generate(
                build_answer_prompt(package),
                temperature=0,
                max_tokens=self._max_output_tokens,
            )
        except Exception:
            audit.add(
                "answer_generation",
                "failed",
                source="rag_generator",
                reason="generation_failed",
            )
            return _processing_failure(
                decision.query,
                "generation_failed",
                evidence=evidence,
                query_enhancement=trace,
            )
        try:
            model_answer = parse_and_validate_answer(package, raw_text)
        except (AnswerProtocolError, TypeError):
            audit.add(
                "protocol_validation",
                "failed",
                source="answer_protocol",
                reason="output_validation_failed",
            )
            return _processing_failure(
                decision.query,
                "output_validation_failed",
                evidence=evidence,
                candidate_answer=(
                    raw_text.strip()
                    if isinstance(raw_text, str) and raw_text.strip()
                    else None
                ),
                query_enhancement=trace,
            )

        audit.add(
            "answer_generation",
            "succeeded",
            source="rag_generator",
            details={
                "attempt": 1,
                "citation_count": len(model_answer.citations),
                "citations": model_answer.citations,
            },
        )
        audit.add(
            "protocol_validation",
            "succeeded",
            source="answer_protocol",
            details={"attempt": 1},
        )
        rendered_candidate = render_semantic_answer(package, model_answer)
        candidate_answer = rendered_candidate.to_text()
        return LegalRAGResult(
            query=decision.query,
            status=AnswerStatus.RETRIEVED_EVIDENCE,
            evidence=evidence,
            model_answer=model_answer,
            candidate_answer=candidate_answer,
            rendered_answer=rendered_candidate,
            query_enhancement=trace,
        )


__all__ = ["CurrentLawRAG"]
