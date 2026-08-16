"""RAG-SFT v2 clean/HN 对照实验的确定性曝光采样。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np
from torch.utils.data import Sampler


EXPOSURE_GROUPS = {
    "clean_only": 0,
    "hn_low": 1,
    "hn_mid": 2,
    "hn_high": 4,
}
EXPECTED_UNIQUE_COUNTS = {"clean": 549, "hard_negative": 70}
COMMON_OPTIMIZER_STEPS = 140
SEQUENCES_PER_OPTIMIZER_STEP = 16
COMMON_EXPOSURES = COMMON_OPTIMIZER_STEPS * SEQUENCES_PER_OPTIMIZER_STEP


@dataclass(frozen=True)
class ExposurePlan:
    """描述一个实验组固定长度采样流的身份与精确曝光数。"""

    group: str
    hard_negative_multiplier: int
    clean_exposures: int
    hard_negative_exposures: int
    total_exposures: int
    indices: tuple[int, ...]
    sha256: str


def audit_clean_hn_pairing(candidate_path: str | Path) -> dict[str, object]:
    """核对 549 clean、70 HN 以及每条 HN 对应的 clean variant。"""

    path = Path(candidate_path).resolve()
    counts = {"clean": 0, "hard_negative": 0}
    clean_queries: set[str] = set()
    hn_queries: set[str] = set()
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as source:
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
                or record_id in seen_ids
                or not isinstance(query_id, str)
                or not query_id
                or variant not in counts
            ):
                raise ValueError(f"candidate 第 {line_number} 行身份或 variant 无效")
            seen_ids.add(record_id)
            counts[variant] += 1
            (clean_queries if variant == "clean" else hn_queries).add(query_id)
    if counts != EXPECTED_UNIQUE_COUNTS:
        raise ValueError(f"实验要求唯一数据计数为 549 clean + 70 HN，实际为 {counts}")
    missing_clean = sorted(hn_queries - clean_queries)
    if missing_clean:
        raise ValueError(f"HN 缺少对应 clean variant: {missing_clean[0]}")
    return {
        "unique_counts": counts,
        "paired_hn_queries": len(hn_queries),
        "all_hn_have_clean_variant": True,
    }


def _balanced_exposures(
    unique_indices: Sequence[int], target: int, *, seed: int, stream: int
) -> list[int]:
    if target == 0:
        return []
    if not unique_indices:
        raise ValueError("非零曝光目标缺少唯一样本")
    values: list[int] = []
    cycle = 0
    while len(values) < target:
        indices = np.asarray(unique_indices, dtype=np.uint32).copy()
        generator = np.random.Generator(
            np.random.PCG64(seed + stream * 1_000_003 + cycle)
        )
        generator.shuffle(indices)
        remaining = target - len(values)
        values.extend(int(index) for index in indices[:remaining])
        cycle += 1
    return values


def _target_hn_exposures(multiplier: int) -> int:
    if multiplier == 0:
        return 0
    weighted_clean = EXPECTED_UNIQUE_COUNTS["clean"]
    weighted_hn = EXPECTED_UNIQUE_COUNTS["hard_negative"] * multiplier
    return round(COMMON_EXPOSURES * weighted_hn / (weighted_clean + weighted_hn))


def build_exposure_plan(
    variant_indices: Mapping[str, Sequence[int]], *, group: str, seed: int
) -> ExposurePlan:
    """按组别构造长度固定、类别计数精确且可恢复的采样流。"""

    if group not in EXPOSURE_GROUPS:
        raise ValueError(f"未知曝光实验组: {group}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed 必须是非负整数")
    clean = tuple(variant_indices.get("clean", ()))
    hn = tuple(variant_indices.get("hard_negative", ()))
    if len(clean) != EXPECTED_UNIQUE_COUNTS["clean"] or len(hn) != EXPECTED_UNIQUE_COUNTS["hard_negative"]:
        raise ValueError("曝光实验要求 549 个 clean 索引和 70 个 HN 索引")
    if len(set(clean + hn)) != len(clean) + len(hn):
        raise ValueError("clean/HN 索引存在重复或交叉")

    multiplier = EXPOSURE_GROUPS[group]
    hn_target = _target_hn_exposures(multiplier)
    clean_target = COMMON_EXPOSURES - hn_target
    stream = _balanced_exposures(clean, clean_target, seed=seed, stream=1)
    stream.extend(_balanced_exposures(hn, hn_target, seed=seed, stream=2))
    shuffled = np.asarray(stream, dtype=np.uint32)
    np.random.Generator(np.random.PCG64(seed + multiplier * 10_000_019)).shuffle(shuffled)
    indices = tuple(int(index) for index in shuffled)
    identity = {
        "group": group,
        "seed": seed,
        "hard_negative_multiplier": multiplier,
        "clean_exposures": clean_target,
        "hard_negative_exposures": hn_target,
        "indices": indices,
    }
    digest = hashlib.sha256(
        json.dumps(identity, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()
    return ExposurePlan(
        group=group,
        hard_negative_multiplier=multiplier,
        clean_exposures=clean_target,
        hard_negative_exposures=hn_target,
        total_exposures=len(indices),
        indices=indices,
        sha256=digest,
    )


class RagSftV2ExposureSampler(Sampler[int]):
    """从确定性曝光流的指定位置继续读取。"""

    def __init__(self, plan: ExposurePlan, *, start_position: int = 0) -> None:
        if (
            isinstance(start_position, bool)
            or not isinstance(start_position, int)
            or not 0 <= start_position <= plan.total_exposures
        ):
            raise ValueError("start_position 超出曝光流范围")
        self.plan = plan
        self.start_position = start_position

    def __iter__(self) -> Iterator[int]:
        return iter(self.plan.indices[self.start_position :])

    def __len__(self) -> int:
        return self.plan.total_exposures - self.start_position


__all__ = [
    "COMMON_EXPOSURES",
    "COMMON_OPTIMIZER_STEPS",
    "EXPOSURE_GROUPS",
    "ExposurePlan",
    "RagSftV2ExposureSampler",
    "SEQUENCES_PER_OPTIMIZER_STEP",
    "audit_clean_hn_pairing",
    "build_exposure_plan",
]
