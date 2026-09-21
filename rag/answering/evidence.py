"""把有序完整法条构造成可计数、可审计的模型证据包。"""

import json
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Sequence, Tuple

from rag.core.contracts import Evidence
from rag.knowledge import EvidenceUnitRepository, LegalArticle


RAG_MAX_OUTPUT_TOKENS = 150
MAX_EVIDENCE_ITEMS = 5


class EvidencePackagingError(RuntimeError):
    """证据包无法在完整性和预算约束下安全建立。"""


def _require_non_blank(name, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")


def _is_positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


@dataclass(frozen=True)
class EvidencePackage:
    """模型请求对应的原始问题和有序 canonical 证据。"""

    query: str
    evidence: Tuple[Evidence, ...] = field(default_factory=tuple)
    display_evidence: Tuple[Evidence, ...] = field(default_factory=tuple)

    def __post_init__(self):
        _require_non_blank("query", self.query)
        evidence = tuple(self.evidence)
        object.__setattr__(self, "evidence", evidence)
        if not evidence:
            raise ValueError("EvidencePackage 必须包含至少一条 Evidence")
        if any(not isinstance(item, Evidence) for item in evidence):
            raise TypeError("evidence 中的元素必须是 Evidence")
        identities = tuple((item.law_name, item.article_no) for item in evidence)
        if len(set(identities)) != len(identities):
            raise ValueError("EvidencePackage 不能包含重复法条")
        display_evidence = tuple(self.display_evidence) or evidence
        object.__setattr__(self, "display_evidence", display_evidence)
        if any(not isinstance(item, Evidence) for item in display_evidence):
            raise TypeError("display_evidence 中的元素必须是 Evidence")
        display_identities = tuple(
            (item.law_name, item.article_no) for item in display_evidence
        )
        if display_identities != identities:
            raise ValueError("display_evidence 必须与模型证据逐条对应")

    def to_model_json(self):
        """生成字段顺序稳定的紧凑 user JSON。"""
        payload = {
            "query": self.query,
            "evidence": [
                {
                    "evidence_id": f"E{index}",
                    "law_name": item.law_name,
                    "article_no": item.article_no,
                    "excerpts": [item.content],
                }
                for index, item in enumerate(self.evidence, start=1)
            ],
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class EvidencePackager:
    """按完整 prompt 预算选择最多五条完整候选。"""

    def __init__(
        self,
        *,
        context_limit: int,
        max_output_tokens: int = RAG_MAX_OUTPUT_TOKENS,
        count_prompt_tokens: Callable[[EvidencePackage], int],
    ):
        if not _is_positive_int(context_limit):
            raise ValueError("context_limit 必须是正整数")
        if not _is_positive_int(max_output_tokens):
            raise ValueError("max_output_tokens 必须是正整数")
        if max_output_tokens >= context_limit:
            raise ValueError("max_output_tokens 必须小于 context_limit")
        if not callable(count_prompt_tokens):
            raise TypeError("count_prompt_tokens 必须可调用")
        self._context_limit = context_limit
        self._max_output_tokens = max_output_tokens
        self._count_prompt_tokens = count_prompt_tokens

    @property
    def max_output_tokens(self):
        return self._max_output_tokens

    @property
    def context_limit(self):
        return self._context_limit

    def build(
        self, query: str, articles: Sequence[LegalArticle]
    ) -> tuple[EvidencePackage, int]:
        """按候选顺序装入可容纳的完整法条，并返回实际 token 数。"""
        _require_non_blank("query", query)
        try:
            candidates = tuple(articles)[:MAX_EVIDENCE_ITEMS]
        except TypeError as exc:
            raise TypeError("articles 必须是 LegalArticle 序列") from exc
        if not candidates:
            raise ValueError("EvidencePackager 不接收空候选")

        seen_chunk_ids = set()
        selected = []
        package = None
        used_tokens = 0
        for article in candidates:
            if not isinstance(article, LegalArticle):
                raise TypeError("articles 中的元素必须是 LegalArticle")
            if article.chunk_id in seen_chunk_ids:
                raise EvidencePackagingError(
                    f"候选中存在重复 chunk_id: {article.chunk_id}"
                )
            seen_chunk_ids.add(article.chunk_id)
            selected.append(
                Evidence(
                    law_name=article.law_name,
                    article_no=article.article_no,
                    content=article.content,
                )
            )
            tentative = EvidencePackage(query=query, evidence=tuple(selected))
            measured = self._count_prompt_tokens(tentative)
            if not _is_positive_int(measured):
                raise EvidencePackagingError(
                    "count_prompt_tokens 必须返回正整数"
                )
            if measured + self._max_output_tokens > self._context_limit:
                selected.pop()
                continue
            package = tentative
            used_tokens = measured

        if package is None:
            raise EvidencePackagingError("所有候选都超过上下文预算")
        return package, used_tokens


@dataclass(frozen=True)
class EvidenceBundle:
    """一个父法条对应的最小原文子单元闭包。"""

    article: LegalArticle
    unit_ids: Tuple[str, ...]
    content: str

    def __post_init__(self):
        if not isinstance(self.article, LegalArticle):
            raise TypeError("article 必须是 LegalArticle")
        unit_ids = tuple(self.unit_ids)
        object.__setattr__(self, "unit_ids", unit_ids)
        if not unit_ids or len(set(unit_ids)) != len(unit_ids):
            raise ValueError("unit_ids 必须非空且不能重复")
        _require_non_blank("content", self.content)


class EvidenceBundlePackager:
    """按子单元相关性和完整 prompt 预算组合父法条证据。"""

    def __init__(
        self,
        *,
        unit_repository,
        unit_scorer,
        context_limit,
        count_prompt_tokens,
        max_output_tokens=RAG_MAX_OUTPUT_TOKENS,
        max_units_per_parent=2,
    ):
        if not isinstance(unit_repository, EvidenceUnitRepository):
            raise TypeError("unit_repository 必须是 EvidenceUnitRepository")
        if not callable(getattr(unit_scorer, "score", None)):
            raise TypeError("unit_scorer 必须提供 score")
        if not _is_positive_int(context_limit):
            raise ValueError("context_limit 必须是正整数")
        if not _is_positive_int(max_output_tokens) or max_output_tokens >= context_limit:
            raise ValueError("max_output_tokens 必须是小于 context_limit 的正整数")
        if not _is_positive_int(max_units_per_parent):
            raise ValueError("max_units_per_parent 必须是正整数")
        if not callable(count_prompt_tokens):
            raise TypeError("count_prompt_tokens 必须可调用")
        self._units = unit_repository
        self._unit_scorer = unit_scorer
        self._context_limit = context_limit
        self._max_output_tokens = max_output_tokens
        self._max_units_per_parent = max_units_per_parent
        self._count_prompt_tokens = count_prompt_tokens

    @property
    def max_output_tokens(self):
        return self._max_output_tokens

    @property
    def context_limit(self):
        return self._context_limit

    def _bundle(self, query, article, selected_unit_ids=()):
        units = self._units.get_for_parent(article.chunk_id)
        by_id = {unit.unit_id: unit for unit in units}
        selected_unit_ids = tuple(selected_unit_ids)
        if selected_unit_ids:
            if len(selected_unit_ids) > self._max_units_per_parent:
                raise EvidencePackagingError("每个父法条选择的子单元超过上限")
            try:
                seeds = tuple(by_id[unit_id] for unit_id in selected_unit_ids)
            except KeyError as exc:
                raise EvidencePackagingError("选择的子单元不属于对应父法条") from exc
        else:
            scores = tuple(self._unit_scorer.score(query, units))
            if len(scores) != len(units):
                raise EvidencePackagingError("unit_scorer 分数数量与子单元数量不一致")
            ranked = sorted(
                zip(units, scores),
                key=lambda item: (-float(item[1]), item[0].start_char),
            )
            seeds = tuple(unit for unit, _ in ranked[: self._max_units_per_parent])
        selected = {}

        def add_with_dependencies(unit):
            if unit.unit_id in selected:
                return
            for dependency_id in unit.dependency_unit_ids:
                try:
                    dependency = by_id[dependency_id]
                except KeyError as exc:
                    raise EvidencePackagingError("子单元依赖闭包不完整") from exc
                add_with_dependencies(dependency)
            selected[unit.unit_id] = unit

        for seed in seeds:
            add_with_dependencies(seed)
        ordered = tuple(sorted(selected.values(), key=lambda unit: unit.start_char))
        return EvidenceBundle(
            article=article,
            unit_ids=tuple(unit.unit_id for unit in ordered),
            content="\n".join(unit.text for unit in ordered),
        )

    def build(self, query, articles, *, selected_unit_ids=()):
        """返回预算内最优父法条组合、实际 token 数与对应 bundle。"""
        _require_non_blank("query", query)
        candidates = tuple(articles)[:MAX_EVIDENCE_ITEMS]
        if not candidates:
            raise ValueError("EvidenceBundlePackager 不接收空候选")
        if any(not isinstance(article, LegalArticle) for article in candidates):
            raise TypeError("articles 中的元素必须是 LegalArticle")
        if len({article.chunk_id for article in candidates}) != len(candidates):
            raise EvidencePackagingError("候选中存在重复 chunk_id")
        selected_unit_ids = tuple(selected_unit_ids)
        selected_by_parent = {}
        candidate_ids = {article.chunk_id for article in candidates}
        for unit_id in selected_unit_ids:
            try:
                unit = self._units.get_by_unit_id(unit_id)
            except KeyError as exc:
                raise EvidencePackagingError("选择了不存在的子单元") from exc
            if unit.parent_chunk_id not in candidate_ids:
                raise EvidencePackagingError("选择的子单元父法条不在候选中")
            selected_by_parent.setdefault(unit.parent_chunk_id, []).append(unit_id)
        bundles = tuple(
            self._bundle(
                query,
                article,
                selected_by_parent.get(article.chunk_id, ()),
            )
            for article in candidates
        )
        best = None
        for size in range(1, len(bundles) + 1):
            for indexes in combinations(range(len(bundles)), size):
                selected = tuple(bundles[index] for index in indexes)
                model_evidence = tuple(
                    Evidence(
                        law_name=bundle.article.law_name,
                        article_no=bundle.article.article_no,
                        content=bundle.content,
                    )
                    for bundle in selected
                )
                display_evidence = tuple(
                    Evidence(
                        law_name=bundle.article.law_name,
                        article_no=bundle.article.article_no,
                        content=bundle.article.content,
                    )
                    for bundle in selected
                )
                package = EvidencePackage(
                    query=query,
                    evidence=model_evidence,
                    display_evidence=display_evidence,
                )
                measured = self._count_prompt_tokens(package)
                if not _is_positive_int(measured):
                    raise EvidencePackagingError(
                        "count_prompt_tokens 必须返回正整数"
                    )
                if measured + self._max_output_tokens > self._context_limit:
                    continue
                inclusion = tuple(index in indexes for index in range(len(bundles)))
                key = (inclusion, len(indexes), -measured)
                if best is None or key > best[0]:
                    best = (key, package, measured, selected)
        if best is None:
            raise EvidencePackagingError("所有子单元证据组合都超过上下文预算")
        _, package, measured, selected = best
        return package, measured, selected


__all__ = [
    "EvidencePackage",
    "EvidenceBundle",
    "EvidenceBundlePackager",
    "EvidencePackager",
    "EvidencePackagingError",
    "MAX_EVIDENCE_ITEMS",
    "RAG_MAX_OUTPUT_TOKENS",
]
