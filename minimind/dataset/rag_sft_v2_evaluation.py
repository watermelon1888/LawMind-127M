"""RAG-SFT v2 隐藏 claims 的训练后评估目标与人工裁决聚合。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .rag_sft_v2_contract import RagSftV2ContractError, derive_required_chunk_ids


class RagSftV2EvaluationError(ValueError):
    """隐藏 claims 评估目标或人工裁决记录无效。"""


@dataclass(frozen=True)
class RagSftV2EvaluationTarget:
    """不进入模型上下文的单 query 隐藏评估真值。"""

    query_id: str
    claims: tuple[str, ...]
    required_chunk_ids: tuple[str, ...]


def build_evaluation_target(canonical: Mapping[str, object]) -> RagSftV2EvaluationTarget:
    """从 canonical authoring 提取后续人工语义评估所需的最小隐藏真值。"""

    query_id = canonical.get("query_id")
    claims = canonical.get("claims")
    if not isinstance(query_id, str) or not query_id.strip():
        raise RagSftV2EvaluationError("canonical.query_id 必须是非空字符串")
    if not isinstance(claims, list) or not claims:
        raise RagSftV2EvaluationError("canonical.claims 必须是非空数组")
    texts: list[str] = []
    for index, claim in enumerate(claims, start=1):
        if not isinstance(claim, dict) or not isinstance(claim.get("text"), str) or not claim["text"].strip():
            raise RagSftV2EvaluationError(f"claims[{index}].text 必须是非空字符串")
        texts.append(claim["text"])
    if len(texts) != len(set(texts)):
        raise RagSftV2EvaluationError("claims.text 不能重复")
    try:
        required = derive_required_chunk_ids(claims)
    except RagSftV2ContractError as error:
        raise RagSftV2EvaluationError("canonical.claims 的支持关系无效") from error
    return RagSftV2EvaluationTarget(
        query_id=query_id,
        claims=tuple(texts),
        required_chunk_ids=required,
    )


def summarize_claim_adjudications(
    targets: Sequence[RagSftV2EvaluationTarget],
    adjudications: Sequence[Mapping[str, object]],
) -> dict[str, int | float]:
    """汇总人工或受控评估器给出的逐 claim 支持判断，不替代语义裁决本身。"""

    target_by_id = {target.query_id: target for target in targets}
    if len(target_by_id) != len(targets):
        raise RagSftV2EvaluationError("评估目标 query_id 不能重复")
    seen: set[str] = set()
    total_claims = 0
    supported_claims = 0
    complete_records = 0
    for item in adjudications:
        if not isinstance(item, dict) or set(item) != {"query_id", "supported_claim_indexes"}:
            raise RagSftV2EvaluationError("裁决记录字段必须精确匹配 schema")
        query_id = item.get("query_id")
        indexes = item.get("supported_claim_indexes")
        target = target_by_id.get(query_id)
        if target is None or query_id in seen:
            raise RagSftV2EvaluationError("裁决记录 query_id 无效或重复")
        if not isinstance(indexes, list) or any(type(index) is not int for index in indexes):
            raise RagSftV2EvaluationError("supported_claim_indexes 必须是整数数组")
        expected = set(range(1, len(target.claims) + 1))
        actual = set(indexes)
        if len(actual) != len(indexes) or not actual.issubset(expected):
            raise RagSftV2EvaluationError("supported_claim_indexes 超出隐藏 claims 范围或重复")
        seen.add(query_id)
        total_claims += len(target.claims)
        supported_claims += len(actual)
        complete_records += actual == expected
    if set(target_by_id) != seen:
        raise RagSftV2EvaluationError("裁决记录必须覆盖全部评估目标")
    return {
        "records": len(targets),
        "claims": total_claims,
        "supported_claims": supported_claims,
        "claim_coverage": supported_claims / total_claims,
        "complete_records": complete_records,
        "record_complete_answer_rate": complete_records / len(targets),
    }


__all__ = [
    "RagSftV2EvaluationError",
    "RagSftV2EvaluationTarget",
    "build_evaluation_target",
    "summarize_claim_adjudications",
]
