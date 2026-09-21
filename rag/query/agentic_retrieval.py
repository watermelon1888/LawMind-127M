"""受控的法律证据助理工具协议。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from rag.external import ExternalLLMResponse
from rag.knowledge import ArticleRepository, EvidenceUnit, LegalArticle
from rag.query.exact_reference import (
    ExactReferenceStatus,
    extract_exact_references,
)


MAX_SUPPLEMENTAL_SEARCHES = 2
MAX_SELECTED_EVIDENCE = 5
MAX_SELECTED_UNITS = 10
MAX_SELECTED_UNITS_PER_PARENT = 2
SEARCH_LAW_TOOL = {
    "type": "function",
    "function": {
        "name": "search_law",
        "description": "仅在现有候选缺少回答原问题所必需的依据时，检索原问题边界内的补充法条。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}
FINALIZE_EVIDENCE_TOOL = {
    "type": "function",
    "function": {
        "name": "finalize_evidence",
        "description": "结束证据检查，提交零至五个需要提升的已观察 chunk_id，并可选择已观察的原文子单元；空数组表示无需调整。",
        "parameters": {
            "type": "object",
            "properties": {
                "promoted_chunk_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 0,
                    "maxItems": MAX_SELECTED_EVIDENCE,
                    "uniqueItems": True,
                },
                "selected_unit_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 0,
                    "maxItems": MAX_SELECTED_UNITS,
                    "uniqueItems": True,
                },
            },
            "required": ["promoted_chunk_ids"],
            "additionalProperties": False,
        },
    },
}
EVIDENCE_ASSISTANT_TOOLS = (SEARCH_LAW_TOOL, FINALIZE_EVIDENCE_TOOL)

AGENTIC_RETRIEVAL_SYSTEM_PROMPT = """你是受控的法律证据助理，不负责回答用户问题。
你的目标是检查全部 Hybrid top-20，判断 Cross-Encoder top-5 是否遗漏了回答原问题所必需的证据，并提出有限纠偏。
每轮必须且只能调用一个工具，不得输出 STOP 或其他自由文本作为结束。
严格按以下顺序决策：
1. 逐条检查 observed_candidates，不能只看 reranker_top5。
2. 必需证据已在 Cross-Encoder top-5 时，调用 finalize_evidence([])。
3. 必需证据在 Hybrid top-20 但不在 top-5 时，先提升遗漏证据，不调用 search_law，直接通过 finalize_evidence 提交一至五个 chunk_id。
4. 只有回答所必需的证据在全部 Hybrid top-20 中都不存在时，才调用 search_law(query)。不能因为 top-5 不完整、希望增加背景或追求面面俱到而搜索。
search_law 最多调用两次；query 只能是原问题的受限改写或子问题，不得新增主体、行为、法律关系、时间、地域、行业、事实前提、历史版本或知识库外资料。若已观察正文明确引用某法某条，且该被引用条文是回答所必需但尚未观察到的证据，补查 query 必须同时包含该法名和条号。补充检索后仍必须调用 finalize_evidence；只有新增候选仍缺少另一项必需证据时才进行第二次搜索。若补查 query 含完整法名和条号，系统会优先确定性查条；观察结果中的 required_promoted_chunk_ids 必须全部提交，不能用其他条文代替。
最终必须且只能调用一次 finalize_evidence(promoted_chunk_ids, selected_unit_ids)。按优先级提交零至五个需要提升的唯一 chunk_id，不得为了凑数提交无关证据。selected_unit_ids 是可选字段；仅在 observed_candidates 展示了 evidence_units 时使用，每个最终入选父法条最多选择两个最相关 unit_id，程序会自动补齐依赖单元。每轮输入中的 allowed_promoted_chunk_ids 和 allowed_selected_unit_ids 是唯一允许提交的 ID 集合，必须逐字复制其中的值，不得自行拼接 ID。最终 top-5 由程序确定性合并，轻量模型再通过 citations 选择实际采用的证据。"""


class AgenticRetrievalError(ValueError):
    """证据助理协议或工具调用无效。"""


@dataclass(frozen=True)
class EvidenceSearch:
    """一次已执行补充检索的审计结果。"""

    query: str
    chunk_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("query 必须是非空字符串")
        chunk_ids = tuple(self.chunk_ids)
        if any(not isinstance(item, str) or not item.strip() for item in chunk_ids):
            raise ValueError("chunk_ids 必须由非空字符串组成")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("chunk_ids 不能重复")
        object.__setattr__(self, "chunk_ids", chunk_ids)


@dataclass(frozen=True)
class EvidenceSelection:
    """程序融合后的有序完整法条及证据助理检索轨迹。"""

    articles: tuple[LegalArticle, ...]
    retrieval_queries: tuple[str, ...]
    searches: tuple[EvidenceSearch, ...] = field(default_factory=tuple)
    promoted_chunk_ids: tuple[str, ...] = field(default_factory=tuple)
    selected_unit_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        articles = tuple(self.articles)
        queries = tuple(self.retrieval_queries)
        searches = tuple(self.searches)
        promoted_chunk_ids = tuple(self.promoted_chunk_ids)
        selected_unit_ids = tuple(self.selected_unit_ids)
        if not 1 <= len(articles) <= MAX_SELECTED_EVIDENCE:
            raise ValueError("articles 必须包含一至五条法条")
        if any(not isinstance(item, LegalArticle) for item in articles):
            raise TypeError("articles 必须由 LegalArticle 组成")
        if len({item.chunk_id for item in articles}) != len(articles):
            raise ValueError("articles 不能包含重复法条")
        if not queries or any(
            not isinstance(item, str) or not item.strip() for item in queries
        ):
            raise ValueError("retrieval_queries 必须包含非空 query")
        if len(set(queries)) != len(queries):
            raise ValueError("retrieval_queries 不能重复")
        if any(not isinstance(item, EvidenceSearch) for item in searches):
            raise TypeError("searches 必须由 EvidenceSearch 组成")
        if len(promoted_chunk_ids) > MAX_SELECTED_EVIDENCE or any(
            not isinstance(item, str) or not item.strip()
            for item in promoted_chunk_ids
        ):
            raise ValueError("promoted_chunk_ids 必须包含零至五个非空字符串")
        if len(set(promoted_chunk_ids)) != len(promoted_chunk_ids):
            raise ValueError("promoted_chunk_ids 不能重复")
        if len(selected_unit_ids) > MAX_SELECTED_UNITS or any(
            not isinstance(item, str) or not item.strip()
            for item in selected_unit_ids
        ):
            raise ValueError("selected_unit_ids 必须包含零至十个非空字符串")
        if len(set(selected_unit_ids)) != len(selected_unit_ids):
            raise ValueError("selected_unit_ids 不能重复")
        object.__setattr__(self, "articles", articles)
        object.__setattr__(self, "retrieval_queries", queries)
        object.__setattr__(self, "searches", searches)
        object.__setattr__(self, "promoted_chunk_ids", promoted_chunk_ids)
        object.__setattr__(self, "selected_unit_ids", selected_unit_ids)

    @property
    def selected_chunk_ids(self):
        return tuple(item.chunk_id for item in self.articles)


def _compact(value: str) -> str:
    return re.sub(r"\s+", "", value)


def _evidence_payload(articles, reranked=(), units_by_parent=None):
    units_by_parent = units_by_parent or {}
    reranker_ranks = {
        item.chunk_id: rank
        for rank, item in enumerate(tuple(reranked), start=1)
    }
    payload = []
    for hybrid_rank, item in enumerate(tuple(articles), start=1):
        record = {
            "chunk_id": item.chunk_id,
            "law_name": item.law_name,
            "article_no": item.article_no,
            "content": item.content,
            "hybrid_rank": hybrid_rank,
            "reranker_rank": reranker_ranks.get(item.chunk_id),
        }
        units = tuple(units_by_parent.get(item.chunk_id, ()))
        if units:
            record["evidence_units"] = [
                {
                    "unit_id": unit.unit_id,
                    "text": unit.text,
                    "dependency_unit_ids": list(unit.dependency_unit_ids),
                }
                for unit in units
            ]
        payload.append(record)
    return payload


def build_agent_messages(query, baseline, reranked, *, units_by_parent=None):
    """构造证据助理首轮消息。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串")
    baseline = tuple(baseline)
    reranked = tuple(reranked)
    if any(not isinstance(item, LegalArticle) for item in baseline):
        raise TypeError("baseline 必须由 LegalArticle 组成")
    if not reranked or any(not isinstance(item, LegalArticle) for item in reranked):
        raise TypeError("reranked 必须包含 LegalArticle")
    baseline_ids = {item.chunk_id for item in baseline}
    reranked_ids = tuple(item.chunk_id for item in reranked)
    if len(set(reranked_ids)) != len(reranked_ids):
        raise ValueError("reranked 不能包含重复法条")
    if any(chunk_id not in baseline_ids for chunk_id in reranked_ids):
        raise ValueError("reranked 必须来自 baseline")
    units_by_parent = _validated_unit_mapping(units_by_parent or {}, baseline_ids)
    return [
        {"role": "system", "content": AGENTIC_RETRIEVAL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "original_query": query,
                    "reranker_top5": list(reranked_ids),
                    "allowed_promoted_chunk_ids": [
                        item.chunk_id for item in baseline
                    ],
                    "allowed_selected_unit_ids": [
                        unit.unit_id
                        for article in baseline
                        for unit in units_by_parent.get(article.chunk_id, ())
                    ],
                    "observed_candidates": _evidence_payload(
                        baseline,
                        reranked,
                        units_by_parent,
                    ),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        },
    ]


def validate_supplemental_query(original_query, query):
    """校验补充 query 没有超出原问题边界。"""
    if not isinstance(query, str) or not query.strip():
        raise AgenticRetrievalError("补充 query 必须是非空字符串")
    original = _compact(original_query)
    candidate = _compact(query)
    if len(candidate) > max(160, len(original) * 2):
        raise AgenticRetrievalError("补充 query 超出原问题长度边界")
    forbidden = ("历史版本", "废止前", "失效前", "未来施行", "以前怎么", "当时怎么")
    if any(marker in candidate and marker not in original for marker in forbidden):
        raise AgenticRetrievalError("补充 query 引入了原问题未要求的时间语义")
    shared = set(original) & set(candidate)
    if len(shared) < min(4, max(2, len(original) // 10)):
        raise AgenticRetrievalError("补充 query 与原问题缺少明确语义关联")
    return query.strip()


def _parse_arguments(arguments, expected_field):
    try:
        payload = json.loads(arguments)
    except (TypeError, json.JSONDecodeError) as exc:
        raise AgenticRetrievalError("工具参数不是有效 JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {expected_field}:
        raise AgenticRetrievalError(f"工具参数必须严格包含 {expected_field}")
    return payload[expected_field]


def _parse_finalize_arguments(arguments):
    try:
        payload = json.loads(arguments)
    except (TypeError, json.JSONDecodeError) as exc:
        raise AgenticRetrievalError("工具参数不是有效 JSON") from exc
    allowed_fields = {"promoted_chunk_ids", "selected_unit_ids"}
    if (
        not isinstance(payload, dict)
        or "promoted_chunk_ids" not in payload
        or not set(payload).issubset(allowed_fields)
    ):
        raise AgenticRetrievalError("finalize_evidence 参数字段不匹配")
    chunk_ids = payload["promoted_chunk_ids"]
    if not isinstance(chunk_ids, list) or len(chunk_ids) > MAX_SELECTED_EVIDENCE:
        raise AgenticRetrievalError("finalize_evidence 必须提交零至五个 chunk_id")
    if any(not isinstance(item, str) or not item.strip() for item in chunk_ids):
        raise AgenticRetrievalError("finalize_evidence 的 chunk_id 必须是非空字符串")
    normalized = tuple(item.strip() for item in chunk_ids)
    if len(set(normalized)) != len(normalized):
        raise AgenticRetrievalError("finalize_evidence 不能提交重复 chunk_id")
    unit_ids = payload.get("selected_unit_ids", [])
    if not isinstance(unit_ids, list) or len(unit_ids) > MAX_SELECTED_UNITS:
        raise AgenticRetrievalError("finalize_evidence 最多提交十个 unit_id")
    if any(not isinstance(item, str) or not item.strip() for item in unit_ids):
        raise AgenticRetrievalError("finalize_evidence 的 unit_id 必须是非空字符串")
    normalized_units = tuple(item.strip() for item in unit_ids)
    if len(set(normalized_units)) != len(normalized_units):
        raise AgenticRetrievalError("finalize_evidence 不能提交重复 unit_id")
    return normalized, normalized_units


def _validated_unit_mapping(units_by_parent, allowed_parent_ids):
    if not isinstance(units_by_parent, dict):
        raise TypeError("units_by_parent 必须是字典")
    normalized = {}
    seen = set()
    for parent_chunk_id, units in units_by_parent.items():
        if parent_chunk_id not in allowed_parent_ids:
            raise ValueError("子单元映射包含未观察父法条")
        values = tuple(units)
        if any(not isinstance(unit, EvidenceUnit) for unit in values):
            raise TypeError("子单元映射必须包含 EvidenceUnit")
        if any(unit.parent_chunk_id != parent_chunk_id for unit in values):
            raise ValueError("子单元与父法条映射不一致")
        if any(unit.unit_id in seen for unit in values):
            raise ValueError("子单元映射不能包含重复 unit_id")
        seen.update(unit.unit_id for unit in values)
        normalized[parent_chunk_id] = values
    return normalized


def _candidate_unit_mapping(candidates):
    mapping = {}
    for candidate in tuple(candidates):
        chunk_id = getattr(candidate, "chunk_id", None)
        units = tuple(getattr(candidate, "units", ()))
        if not units:
            continue
        mapping[chunk_id] = units
    return _validated_unit_mapping(mapping, set(mapping))


def _merge_evidence(reranked, promoted):
    """保留精排前两条，用 Agent 建议补位，再按精排顺序补足五条。"""
    selected = []
    seen = set()

    def append(article):
        if article.chunk_id not in seen and len(selected) < MAX_SELECTED_EVIDENCE:
            seen.add(article.chunk_id)
            selected.append(article)

    for article in tuple(reranked)[:2]:
        append(article)
    for article in promoted:
        append(article)
    for article in reranked:
        append(article)

    selected_ids = {item.chunk_id for item in selected}
    ordered = []
    if reranked and reranked[0].chunk_id in selected_ids:
        ordered.append(reranked[0])
    ordered.extend(
        item
        for item in promoted
        if item.chunk_id in selected_ids
        and all(existing.chunk_id != item.chunk_id for existing in ordered)
    )
    ordered.extend(
        item
        for item in selected
        if all(existing.chunk_id != item.chunk_id for existing in ordered)
    )
    return tuple(ordered)


def _resolve_candidates(candidates, article_repository):
    resolved = []
    seen = set()
    for candidate in tuple(candidates):
        chunk_id = getattr(candidate, "chunk_id", None)
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise AgenticRetrievalError("检索候选缺少有效 chunk_id")
        if chunk_id in seen:
            continue
        try:
            article = article_repository.get_by_chunk_id(chunk_id)
        except (KeyError, TypeError, ValueError) as exc:
            raise AgenticRetrievalError("检索候选无法映射到 canonical 法条") from exc
        seen.add(chunk_id)
        resolved.append(article)
    return tuple(resolved)


def _exact_search(query, article_repository):
    """解析完整法名和条号并执行确定性查条。"""
    try:
        parsed = extract_exact_references(query)
    except (TypeError, ValueError):
        return None
    if parsed.status is not ExactReferenceStatus.FOUND or len(parsed.references) != 1:
        return None
    reference = parsed.references[0]
    article = article_repository.lookup(reference.law_name, reference.article_no)
    return reference, article


def _append_tool_exchange(messages, response, call, observation):
    messages.append(
        {
            "role": "assistant",
            "content": response.content or "",
            "tool_calls": [
                {
                    "id": call.call_id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": call.arguments,
                    },
                }
            ],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": call.call_id,
            "content": json.dumps(observation, ensure_ascii=False, separators=(",", ":")),
        }
    )


def run_agentic_retrieval(
    *,
    query,
    baseline,
    reranked,
    retriever,
    article_repository,
    external_llm,
    units_by_parent=None,
):
    """执行最多两次补充检索，并要求显式结束证据检查。"""
    if not isinstance(article_repository, ArticleRepository):
        raise TypeError("article_repository 必须是 ArticleRepository")
    if not callable(getattr(retriever, "retrieve_candidates", None)):
        raise TypeError("retriever 必须提供可调用的 retrieve_candidates")
    if not callable(getattr(external_llm, "generate_with_tools", None)):
        raise TypeError("external_llm 必须提供可调用的 generate_with_tools")

    baseline = tuple(baseline)
    reranked = tuple(reranked)
    units_by_parent = _validated_unit_mapping(
        units_by_parent or {},
        {item.chunk_id for item in baseline},
    )
    messages = build_agent_messages(
        query,
        baseline,
        reranked,
        units_by_parent=units_by_parent,
    )
    observed = {item.chunk_id: item for item in baseline}
    observed_units = {
        unit.unit_id: unit
        for units in units_by_parent.values()
        for unit in units
    }
    retrieval_queries = [query]
    searches = []
    required_promoted_chunk_ids = set()
    unresolved_exact_chunk_ids = set()

    while True:
        response = external_llm.generate_with_tools(
            messages,
            tools=EVIDENCE_ASSISTANT_TOOLS,
            temperature=0,
            max_tokens=256,
        )
        if not isinstance(response, ExternalLLMResponse):
            raise AgenticRetrievalError("外部模型工具响应类型无效")
        if len(response.tool_calls) != 1:
            raise AgenticRetrievalError("每轮必须且只能调用一个证据工具")
        call = response.tool_calls[0]
        if not isinstance(call.call_id, str) or not call.call_id.strip():
            raise AgenticRetrievalError("工具调用 ID 无效")

        if call.name == "finalize_evidence":
            promoted_chunk_ids, selected_unit_ids = _parse_finalize_arguments(
                call.arguments
            )
            unknown = tuple(
                chunk_id
                for chunk_id in promoted_chunk_ids
                if chunk_id not in observed
            )
            if unknown:
                raise AgenticRetrievalError("finalize_evidence 提交了未观察到的 chunk_id")
            missing_required = tuple(
                chunk_id
                for chunk_id in required_promoted_chunk_ids
                if chunk_id not in promoted_chunk_ids
            )
            if missing_required:
                raise AgenticRetrievalError(
                    "finalize_evidence 遗漏了精确查条命中证据"
                )
            if unresolved_exact_chunk_ids:
                raise AgenticRetrievalError(
                    "finalize_evidence 在精确查条未命中后提交了其他证据"
                )
            promoted = tuple(observed[chunk_id] for chunk_id in promoted_chunk_ids)
            unknown_units = tuple(
                unit_id for unit_id in selected_unit_ids if unit_id not in observed_units
            )
            if unknown_units:
                raise AgenticRetrievalError("finalize_evidence 提交了未观察到的 unit_id")
            selected_articles = _merge_evidence(reranked, promoted)
            selected_parent_ids = {item.chunk_id for item in selected_articles}
            selected_counts = {}
            for unit_id in selected_unit_ids:
                parent_chunk_id = observed_units[unit_id].parent_chunk_id
                if parent_chunk_id not in selected_parent_ids:
                    raise AgenticRetrievalError("selected unit 的父法条未进入最终 top-5")
                selected_counts[parent_chunk_id] = selected_counts.get(parent_chunk_id, 0) + 1
                if selected_counts[parent_chunk_id] > MAX_SELECTED_UNITS_PER_PARENT:
                    raise AgenticRetrievalError("每个父法条最多选择两个 unit_id")
            return EvidenceSelection(
                articles=selected_articles,
                retrieval_queries=tuple(retrieval_queries),
                searches=tuple(searches),
                promoted_chunk_ids=promoted_chunk_ids,
                selected_unit_ids=selected_unit_ids,
            )

        if call.name != "search_law":
            raise AgenticRetrievalError("证据助理调用了未知工具")
        if len(searches) >= MAX_SUPPLEMENTAL_SEARCHES:
            raise AgenticRetrievalError("search_law 调用超过两次上限")
        supplemental = validate_supplemental_query(
            query,
            _parse_arguments(call.arguments, "query"),
        )
        if supplemental in retrieval_queries:
            raise AgenticRetrievalError("补充 query 不能重复")
        exact_result = _exact_search(supplemental, article_repository)
        if exact_result is not None:
            reference, exact_article = exact_result
            candidates = ()
            articles = (exact_article,) if exact_article is not None else ()
            supplemental_units = {}
            search_mode = (
                "exact_lookup" if exact_article is not None else "exact_lookup_miss"
            )
            if exact_article is not None:
                required_promoted_chunk_ids.add(exact_article.chunk_id)
            else:
                unresolved_exact_chunk_ids.add(
                    f"{reference.law_name}#{reference.article_no}"
                )
        else:
            candidates = retriever.retrieve_candidates(supplemental)
            articles = _resolve_candidates(candidates, article_repository)
            supplemental_units = _candidate_unit_mapping(candidates)
            search_mode = "hybrid"
        for article in articles:
            observed.setdefault(article.chunk_id, article)
        for units in supplemental_units.values():
            for unit in units:
                observed_units.setdefault(unit.unit_id, unit)
        retrieval_queries.append(supplemental)
        searches.append(
            EvidenceSearch(
                query=supplemental,
                chunk_ids=tuple(item.chunk_id for item in articles),
            )
        )
        _append_tool_exchange(
            messages,
            response,
            call,
            {
                "query": supplemental,
                "search_mode": search_mode,
                "required_promoted_chunk_ids": sorted(required_promoted_chunk_ids),
                "exact_lookup_misses": sorted(unresolved_exact_chunk_ids),
                "allowed_promoted_chunk_ids": list(observed),
                "allowed_selected_unit_ids": list(observed_units),
                "observed_candidates": _evidence_payload(
                    articles,
                    units_by_parent=supplemental_units,
                ),
            },
        )


__all__ = [
    "AGENTIC_RETRIEVAL_SYSTEM_PROMPT",
    "EVIDENCE_ASSISTANT_TOOLS",
    "FINALIZE_EVIDENCE_TOOL",
    "AgenticRetrievalError",
    "EvidenceSearch",
    "EvidenceSelection",
    "MAX_SELECTED_EVIDENCE",
    "MAX_SELECTED_UNITS",
    "MAX_SUPPLEMENTAL_SEARCHES",
    "SEARCH_LAW_TOOL",
    "build_agent_messages",
    "run_agentic_retrieval",
    "validate_supplemental_query",
]
