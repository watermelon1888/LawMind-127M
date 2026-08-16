"""定义 Query-SFT v2 的 authoring、语义审核与冻结检索选择约束。"""

from __future__ import annotations

from typing import Any


AUTHORING_TYPES = frozenset(
    {
        "no_op",
        "colloquial",
        "ellipsis",
        "ambiguous_multi_intent",
        "explicit_multi_matter",
    }
)
AUTHORING_FIELDS = (
    "id",
    "source_id",
    "query_original",
    "required_chunk_ids",
    "target",
)
TARGET_FIELDS = ("rewrite", "expansion_terms", "subqueries")
SEMANTIC_REVIEW_FIELDS = (
    "candidate_id",
    "work_id",
    "review_decision",
    "semantic_preserved",
    "unsupported_fact_absent",
    "ambiguity_not_resolved",
    "natural_text",
    "authoring_type_fulfilled",
    "subquery_coverage_complete",
    "explicit_multi_matter_exception",
    "reason",
)


class QuerySftV2ContractError(ValueError):
    """表示 Query-SFT v2 记录未满足冻结的数据协议。"""


def is_exact_noop(query_original: str, target: dict[str, object]) -> bool:
    """判断目标是否为不改变原始 query 的确定性 no-op。"""

    return (
        target.get("rewrite") == query_original
        and target.get("expansion_terms") == []
        and target.get("subqueries") == []
    )


def validate_target_for_authoring_type(
    *,
    query_original: str,
    authoring_type: str,
    target: dict[str, object],
    allow_explicit_multi_matter_exception: bool = False,
) -> None:
    """验证 target 是否满足对应输入类型的最低监督价值。"""

    if authoring_type not in AUTHORING_TYPES:
        raise QuerySftV2ContractError("authoring_type 无效")
    if not isinstance(query_original, str) or not query_original:
        raise QuerySftV2ContractError("query_original 无效")
    if not isinstance(target, dict) or tuple(target) != TARGET_FIELDS:
        raise QuerySftV2ContractError("target 字段或顺序无效")
    rewrite = target.get("rewrite")
    terms = target.get("expansion_terms")
    subqueries = target.get("subqueries")
    if (
        not isinstance(rewrite, str)
        or not rewrite
        or not isinstance(terms, list)
        or not isinstance(subqueries, list)
        or any(not isinstance(value, str) or not value for value in [*terms, *subqueries])
    ):
        raise QuerySftV2ContractError("target 内容无效")
    if authoring_type == "no_op" and not is_exact_noop(query_original, target):
        raise QuerySftV2ContractError("no_op 记录不得无依据增强")
    if authoring_type in {"colloquial", "ellipsis"} and rewrite == query_original:
        raise QuerySftV2ContractError("口语或省略记录必须提供等价规范化 rewrite")
    if (
        authoring_type == "explicit_multi_matter"
        and not allow_explicit_multi_matter_exception
        and not 2 <= len(subqueries) <= 3
    ):
        raise QuerySftV2ContractError("明确多事项记录必须提供 2 至 3 条 subqueries")


def validate_semantic_review(review: dict[str, object]) -> None:
    """验证独立审核账本的逐候选结论已经完整表达语义门槛。"""

    if not isinstance(review, dict) or tuple(review) != SEMANTIC_REVIEW_FIELDS:
        raise QuerySftV2ContractError("语义审核字段或顺序无效")
    if not isinstance(review.get("candidate_id"), str) or not review["candidate_id"]:
        raise QuerySftV2ContractError("候选 ID 无效")
    if not isinstance(review.get("work_id"), str) or not review["work_id"]:
        raise QuerySftV2ContractError("work_id 无效")
    if review.get("review_decision") not in {"approved", "rejected"}:
        raise QuerySftV2ContractError("语义审核结论无效")
    if not isinstance(review.get("reason"), str) or not review["reason"].strip():
        raise QuerySftV2ContractError("语义审核理由无效")
    flags = SEMANTIC_REVIEW_FIELDS[3:-1]
    if any(type(review.get(name)) is not bool for name in flags):
        raise QuerySftV2ContractError("语义审核布尔结论无效")
    if review["review_decision"] == "approved" and not all(review[name] for name in flags):
        allowed_false = {"explicit_multi_matter_exception"}
        if any(not review[name] and name not in allowed_false for name in flags):
            raise QuerySftV2ContractError("通过的语义审核不得缺少任一质量结论")
    if review["review_decision"] == "rejected" and all(review[name] for name in flags):
        raise QuerySftV2ContractError("拒绝的语义审核必须明确至少一项未通过的质量结论")


def select_non_degrading_candidate(
    *,
    baseline_score: tuple[int, float, float],
    noop_score: tuple[int, float, float],
    approved_candidates: list[tuple[str, tuple[int, float, float]]],
) -> tuple[str, str, tuple[int, float, float]]:
    """选择不劣于原始 query 与 no-op 的已审核候选，持平时允许保留增强。"""

    if noop_score < baseline_score:
        raise QuerySftV2ContractError("确定性 no-op 检索分数不得低于原始 query 基线")
    reference = max(baseline_score, noop_score)
    qualified = [
        (candidate_id, score)
        for candidate_id, score in approved_candidates
        if score >= reference
    ]
    if not qualified:
        return "noop", "noop", noop_score
    candidate_id, score = max(qualified, key=lambda item: (item[1], item[0]))
    return "teacher_candidate", candidate_id, score


__all__ = [
    "AUTHORING_FIELDS",
    "AUTHORING_TYPES",
    "SEMANTIC_REVIEW_FIELDS",
    "TARGET_FIELDS",
    "QuerySftV2ContractError",
    "is_exact_noop",
    "select_non_degrading_candidate",
    "validate_semantic_review",
    "validate_target_for_authoring_type",
]
