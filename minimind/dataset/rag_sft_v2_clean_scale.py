"""固定 226 条唯一 HN、改变 Clean 覆盖量的确定性实验计划。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PIPELINE = "rag_sft_v2_clean_scale_plan_v1"
SUBSET_SEED = 42
EXPECTED_UNIQUE_COUNTS = {"clean": 549, "hard_negative": 226}
GROUP_CLEAN_COUNTS = {
    "clean_only": 549,
    "clean_113_hn_226": 113,
    "clean_226_hn_226": 226,
    "clean_452_hn_226": 452,
}
GROUP_HN_COUNTS = {
    "clean_only": 0,
    "clean_113_hn_226": 226,
    "clean_226_hn_226": 226,
    "clean_452_hn_226": 226,
}
_LENGTH_BUCKET_CHARS = 256


@dataclass(frozen=True)
class CleanScalePlan:
    """描述一个 Clean 规模实验组的冻结数据身份。"""

    group: str
    subset_seed: int
    clean_count: int
    hard_negative_count: int
    paired_clean_count: int
    indices: tuple[int, ...]
    clean_record_ids: tuple[str, ...]
    hard_negative_record_ids: tuple[str, ...]
    clean_query_ids: tuple[str, ...]
    sha256: str

    @property
    def total_records(self) -> int:
        return len(self.indices)


@dataclass(frozen=True)
class _CandidateRecord:
    index: int
    record_id: str
    query_id: str
    variant: str
    stratum: tuple[Any, ...] | None


def _stable_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _require_string_list(
    value: object, *, name: str, minimum: int, maximum: int
) -> list[str]:
    if (
        not isinstance(value, list)
        or not minimum <= len(value) <= maximum
        or len(value) != len(set(value))
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise ValueError(f"candidate {name} 无效")
    return value


def _law_names(chunk_ids: list[str]) -> tuple[str, ...]:
    values = []
    for chunk_id in chunk_ids:
        if "#" not in chunk_id:
            raise ValueError("candidate required_chunk_ids 缺少法条分隔符")
        values.append(chunk_id.rsplit("#", 1)[0])
    return tuple(sorted(set(values)))


def _record_stratum(record: dict[str, Any]) -> tuple[Any, ...]:
    required = _require_string_list(
        record.get("required_chunk_ids"),
        name="required_chunk_ids",
        minimum=1,
        maximum=3,
    )
    visible = _require_string_list(
        record.get("visible_chunk_ids"),
        name="visible_chunk_ids",
        minimum=1,
        maximum=5,
    )
    if not set(required).issubset(visible):
        raise ValueError("candidate 缺少 required evidence")
    conversations = record.get("conversations")
    if (
        not isinstance(conversations, list)
        or len(conversations) != 3
        or not isinstance(conversations[1], dict)
        or not isinstance(conversations[1].get("content"), str)
    ):
        raise ValueError("candidate conversations 无效")
    required_set = set(required)
    positions = tuple(
        index for index, chunk_id in enumerate(visible, 1) if chunk_id in required_set
    )
    return (
        _law_names(required),
        len(required),
        len(visible),
        positions,
        len(conversations[1]["content"]) // _LENGTH_BUCKET_CHARS,
    )


def _load_candidate(path: str | Path) -> tuple[list[_CandidateRecord], dict[str, int]]:
    resolved = Path(path).resolve()
    records: list[_CandidateRecord] = []
    counts = {"clean": 0, "hard_negative": 0}
    seen_record_ids: set[str] = set()
    seen_variant_queries: set[tuple[str, str]] = set()
    with resolved.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"candidate 第 {line_number} 行为空")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"candidate 第 {line_number} 行不是 object")
            record_id = value.get("id")
            query_id = value.get("query_id")
            variant = value.get("variant")
            if (
                not isinstance(record_id, str)
                or not record_id
                or record_id in seen_record_ids
                or not isinstance(query_id, str)
                or not query_id
                or variant not in counts
                or (variant, query_id) in seen_variant_queries
            ):
                raise ValueError(f"candidate 第 {line_number} 行身份无效")
            seen_record_ids.add(record_id)
            seen_variant_queries.add((variant, query_id))
            counts[variant] += 1
            records.append(
                _CandidateRecord(
                    index=len(records),
                    record_id=record_id,
                    query_id=query_id,
                    variant=variant,
                    stratum=_record_stratum(value),
                )
            )
    if counts != EXPECTED_UNIQUE_COUNTS:
        raise ValueError(
            "Clean 规模实验要求 549 clean + 226 HN，"
            f"实际为 {counts}"
        )
    return records, counts


def _select_stratified_queries(
    records_by_query: dict[str, _CandidateRecord],
    *,
    target: int,
    seed: int,
    stream: str,
) -> set[str]:
    if not 0 <= target <= len(records_by_query):
        raise ValueError("分层抽样目标数量无效")
    strata: dict[tuple[Any, ...], list[str]] = {}
    for query_id, record in records_by_query.items():
        if record.stratum is None:
            raise ValueError("candidate stratum 缺失")
        strata.setdefault(record.stratum, []).append(query_id)

    total = len(records_by_query)
    quotas: dict[tuple[Any, ...], int] = {}
    remainders: list[tuple[int, tuple[Any, ...]]] = []
    for stratum, query_ids in strata.items():
        scaled = target * len(query_ids)
        quotas[stratum] = scaled // total
        remainders.append((scaled % total, stratum))
    remaining = target - sum(quotas.values())
    ordered_remainders = sorted(
        remainders,
        key=lambda item: (
            -item[0],
            _stable_digest(
                {"seed": seed, "stratum": item[1], "stream": f"{stream}:quota"}
            ),
        ),
    )
    for _, stratum in ordered_remainders[:remaining]:
        quotas[stratum] += 1

    selected: set[str] = set()
    for stratum, query_ids in strata.items():
        ordered = sorted(
            query_ids,
            key=lambda query_id: _stable_digest(
                {
                    "seed": seed,
                    "query_id": query_id,
                    "stream": f"{stream}:within_stratum",
                }
            ),
        )
        selected.update(ordered[: quotas[stratum]])
    if len(selected) != target:
        raise ValueError(f"分层抽样数量未闭合: {len(selected)} != {target}")
    return selected


def build_clean_scale_plan(
    candidate_path: str | Path, *, group: str, subset_seed: int = SUBSET_SEED
) -> CleanScalePlan:
    """从冻结 candidate 构造嵌套 Clean 子集和完整 HN 集。"""

    if group not in GROUP_CLEAN_COUNTS:
        raise ValueError(f"未知 Clean 规模实验组: {group}")
    if (
        isinstance(subset_seed, bool)
        or not isinstance(subset_seed, int)
        or subset_seed < 0
    ):
        raise ValueError("subset_seed 必须是非负整数")
    records, _ = _load_candidate(candidate_path)
    clean_by_query = {
        record.query_id: record for record in records if record.variant == "clean"
    }
    hn_by_query = {
        record.query_id: record
        for record in records
        if record.variant == "hard_negative"
    }
    missing_clean = sorted(set(hn_by_query) - set(clean_by_query))
    if missing_clean:
        raise ValueError(f"HN 缺少对应 clean variant: {missing_clean[0]}")

    paired_queries = set(hn_by_query)
    c113_queries = _select_stratified_queries(
        hn_by_query,
        target=113,
        seed=subset_seed,
        stream="paired_half",
    )
    unpaired_clean = {
        query_id: record
        for query_id, record in clean_by_query.items()
        if query_id not in paired_queries
    }
    c452_queries = paired_queries | _select_stratified_queries(
        unpaired_clean,
        target=226,
        seed=subset_seed,
        stream="unpaired_fill",
    )
    if group == "clean_113_hn_226":
        selected_clean_queries = c113_queries
    elif group == "clean_226_hn_226":
        selected_clean_queries = paired_queries
    elif group == "clean_452_hn_226":
        selected_clean_queries = c452_queries
    else:
        selected_clean_queries = set(clean_by_query)
    selected_hn_queries = set() if group == "clean_only" else paired_queries

    clean_records = [
        record
        for record in records
        if record.variant == "clean" and record.query_id in selected_clean_queries
    ]
    hn_records = [
        record
        for record in records
        if record.variant == "hard_negative" and record.query_id in selected_hn_queries
    ]
    expected_clean = GROUP_CLEAN_COUNTS[group]
    expected_hn = GROUP_HN_COUNTS[group]
    if len(clean_records) != expected_clean or len(hn_records) != expected_hn:
        raise ValueError(
            f"{group} 数据计数未闭合: clean={len(clean_records)}, "
            f"HN={len(hn_records)}"
        )
    indices = tuple(sorted(record.index for record in clean_records + hn_records))
    clean_record_ids = tuple(record.record_id for record in clean_records)
    hn_record_ids = tuple(record.record_id for record in hn_records)
    clean_query_ids = tuple(record.query_id for record in clean_records)
    identity = {
        "pipeline": PIPELINE,
        "group": group,
        "subset_seed": subset_seed,
        "clean_record_ids": clean_record_ids,
        "hard_negative_record_ids": hn_record_ids,
        "stratification": {
            "paired_half": group == "clean_113_hn_226",
            "unpaired_fill_to_two_to_one": group == "clean_452_hn_226",
            "length_bucket_chars": _LENGTH_BUCKET_CHARS,
        },
    }
    return CleanScalePlan(
        group=group,
        subset_seed=subset_seed,
        clean_count=len(clean_records),
        hard_negative_count=len(hn_records),
        paired_clean_count=sum(
            query_id in paired_queries for query_id in selected_clean_queries
        ),
        indices=indices,
        clean_record_ids=clean_record_ids,
        hard_negative_record_ids=hn_record_ids,
        clean_query_ids=clean_query_ids,
        sha256=_stable_digest(identity),
    )


def audit_clean_scale_candidate(candidate_path: str | Path) -> dict[str, object]:
    """核对新 release 是否满足四组实验的唯一数据和配对前提。"""

    records, counts = _load_candidate(candidate_path)
    clean_queries = {
        record.query_id for record in records if record.variant == "clean"
    }
    hn_queries = {
        record.query_id
        for record in records
        if record.variant == "hard_negative"
    }
    missing_clean = sorted(hn_queries - clean_queries)
    if missing_clean:
        raise ValueError(f"HN 缺少对应 clean variant: {missing_clean[0]}")
    return {
        "unique_counts": counts,
        "paired_hn_queries": len(hn_queries),
        "all_hn_have_clean_variant": True,
        "strict_two_to_one_available": True,
        "full_clean_to_hn_ratio": counts["clean"] / counts["hard_negative"],
    }


__all__ = [
    "EXPECTED_UNIQUE_COUNTS",
    "GROUP_CLEAN_COUNTS",
    "GROUP_HN_COUNTS",
    "PIPELINE",
    "SUBSET_SEED",
    "CleanScalePlan",
    "audit_clean_scale_candidate",
    "build_clean_scale_plan",
]
