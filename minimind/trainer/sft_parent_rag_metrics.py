"""汇总父权重比较使用的冻结 RAG 成对证据指标。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


RAG_METRICS = (
    "protocol_valid_rate",
    "sufficient_answer_success_rate",
    "insufficient_refusal_correct_rate",
    "paired_success_rate",
    "required_citation_recall",
    "invalid_citation_rate",
    "hard_negative_citation_rate",
    "refusal_shape_compliance_rate",
)
_VARIANTS = ("sufficient", "insufficient")


def _nonempty_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} 必须是非空字符串")
    return value


def _string_ids(value: object, description: str, *, nonempty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{description} 必须是{'非空' if nonempty else ''}字符串数组")
    if not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{description} 必须只包含非空字符串")
    if len(set(value)) != len(value):
        raise ValueError(f"{description} 不能重复")
    return tuple(value)


def _normalized_record(record: object) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise ValueError("RAG 成对评估记录必须是 JSON object")
    expected = {
        "query_id",
        "variant",
        "visible_chunk_ids",
        "required_chunk_ids",
        "hard_negative_chunk_ids",
        "protocol_valid",
        "refuse",
        "summary_nonempty",
        "citation_chunk_ids",
    }
    if set(record) != expected:
        raise ValueError("RAG 成对评估记录必须使用封闭 schema")
    query_id = _nonempty_string(record["query_id"], "query_id")
    variant = record["variant"]
    if variant not in _VARIANTS:
        raise ValueError("variant 必须是 sufficient 或 insufficient")
    visible = _string_ids(record["visible_chunk_ids"], "visible_chunk_ids", nonempty=True)
    required = _string_ids(record["required_chunk_ids"], "required_chunk_ids", nonempty=True)
    hard_negative = _string_ids(
        record["hard_negative_chunk_ids"], "hard_negative_chunk_ids", nonempty=False
    )
    citations = _string_ids(
        record["citation_chunk_ids"], "citation_chunk_ids", nonempty=False
    )
    if not set(hard_negative).issubset(visible) or set(hard_negative) & set(required):
        raise ValueError("hard_negative_chunk_ids 必须是可见且非必要的证据")
    required_visible = set(required).issubset(visible)
    if variant == "sufficient" and not required_visible:
        raise ValueError("充分证据包必须包含全部 required_chunk_ids")
    if variant == "insufficient" and required_visible:
        raise ValueError("不充分证据包必须缺少至少一个 required_chunk_id")
    for name in ("protocol_valid", "refuse", "summary_nonempty"):
        if not isinstance(record[name], bool):
            raise ValueError(f"{name} 必须是布尔值")
    return {
        "query_id": query_id,
        "variant": variant,
        "visible": frozenset(visible),
        "required": frozenset(required),
        "hard_negative": frozenset(hard_negative),
        "protocol_valid": record["protocol_valid"],
        "refuse": record["refuse"],
        "summary_nonempty": record["summary_nonempty"],
        "citations": frozenset(citations),
    }


def aggregate_rag_pair_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """按 canonical query 聚合充分/不充分证据的冻结自动指标。"""

    if not records:
        raise ValueError("RAG 成对评估至少需要一条记录")
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for raw_record in records:
        record = _normalized_record(raw_record)
        query_records = grouped[record["query_id"]]
        if record["variant"] in query_records:
            raise ValueError("同一 query_id 不能存在重复 variant")
        query_records[record["variant"]] = record

    normalized = [record for variants in grouped.values() for record in variants.values()]
    sufficient = [record for record in normalized if record["variant"] == "sufficient"]
    insufficient = [record for record in normalized if record["variant"] == "insufficient"]
    if not sufficient:
        raise ValueError("RAG 成对评估至少需要充分证据记录")

    def citations_valid(record: dict[str, Any]) -> bool:
        return record["citations"].issubset(record["visible"])

    def sufficient_success(record: dict[str, Any]) -> bool:
        return (
            record["protocol_valid"]
            and not record["refuse"]
            and record["summary_nonempty"]
            and citations_valid(record)
            and record["required"].issubset(record["citations"])
            and not (record["hard_negative"] & record["citations"])
        )

    def insufficient_refusal(record: dict[str, Any]) -> bool:
        return record["protocol_valid"] and record["refuse"]

    def refusal_shape(record: dict[str, Any]) -> bool:
        return (
            insufficient_refusal(record)
            and not record["summary_nonempty"]
            and not record["citations"]
        )

    paired = [variants for variants in grouped.values() if set(variants) == set(_VARIANTS)]
    required_total = sum(len(record["required"]) for record in sufficient)
    required_found = sum(
        len(record["required"] & record["citations"]) for record in sufficient
    )
    hard_negative_eligible = [record for record in sufficient if record["hard_negative"]]

    return {
        "protocol_valid_rate": sum(record["protocol_valid"] for record in normalized)
        / len(normalized),
        "sufficient_answer_success_rate": sum(sufficient_success(record) for record in sufficient)
        / len(sufficient),
        "insufficient_refusal_correct_rate": (
            sum(insufficient_refusal(record) for record in insufficient) / len(insufficient)
            if insufficient
            else 0.0
        ),
        "paired_success_rate": (
            sum(
                sufficient_success(variants["sufficient"])
                and refusal_shape(variants["insufficient"])
                for variants in paired
            )
            / len(paired)
            if paired
            else 0.0
        ),
        "required_citation_recall": required_found / required_total,
        "invalid_citation_rate": sum(
            not citations_valid(record) for record in normalized
        )
        / len(normalized),
        "hard_negative_citation_rate": (
            sum(
                bool(record["hard_negative"] & record["citations"])
                for record in hard_negative_eligible
            )
            / len(hard_negative_eligible)
            if hard_negative_eligible
            else 0.0
        ),
        "refusal_shape_compliance_rate": (
            sum(refusal_shape(record) for record in insufficient) / len(insufficient)
            if insufficient
            else 0.0
        ),
    }
