"""当前法律 RAG 检索与构包评估的纯协议和聚合逻辑。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from statistics import fmean
from random import Random
from typing import Mapping, Optional, Tuple

from rag.answering import EvidencePackagingError
from rag.core import (
    QueryEnhancementFailureReason,
    QueryEnhancementStatus,
)
from rag.query import (
    QueryEnhancement,
    QueryEnhancementProtocolError,
    compile_retrieval_queries,
    parse_and_validate_query_enhancement,
)


class EvaluationMode(str, Enum):
    """当前检索链路需要分别报告的评估模式。"""

    BASELINE_ORIGINAL = "baseline_original"
    NOOP_APPLIED = "noop_applied"
    REWRITE_ONLY = "rewrite_only"
    TERMS_ONLY = "terms_only"
    SUBQUERIES_ONLY = "subqueries_only"
    FULL_ENHANCEMENT = "full_enhancement"
    ORACLE_FORMAL = "oracle_formal"


class PackagingStatus(str, Enum):
    """离线构包阶段是否实际形成证据包。"""

    NOT_ATTEMPTED = "not_attempted"
    APPLIED = "applied"
    FAILED = "failed"


@dataclass(frozen=True)
class EvaluationCase:
    """一条只绑定原始问题与 canonical required GT 的评估样本。"""

    query_id: str
    query_original: str
    required_chunk_ids: Tuple[str, ...]

    def __post_init__(self):
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise ValueError("query_id 必须是非空字符串")
        if not isinstance(self.query_original, str) or not self.query_original.strip():
            raise ValueError("query_original 必须是非空字符串")
        required = tuple(self.required_chunk_ids)
        object.__setattr__(self, "required_chunk_ids", required)
        if not required or any(
            not isinstance(item, str) or not item.strip() for item in required
        ):
            raise ValueError("required_chunk_ids 必须是非空字符串序列")
        if len(required) != len(set(required)):
            raise ValueError("required_chunk_ids 不能重复")


@dataclass(frozen=True)
class QueryEvaluationPlan:
    """一个评估模式的增强协议状态与实际检索腿。"""

    mode: EvaluationMode
    status: QueryEnhancementStatus
    failure_reason: Optional[QueryEnhancementFailureReason]
    protocol_attempted: bool
    protocol_output_received: bool
    protocol_valid: Optional[bool]
    is_noop: bool
    planned_retrieval_queries: Tuple[str, ...]
    executed_retrieval_queries: Tuple[str, ...]


@dataclass(frozen=True)
class StageMetrics:
    """一条题目在某个有序候选阶段的 required GT 指标。"""

    matched_gt_chunk_ids: Tuple[str, ...]
    required_gt_coverage: float
    any_hit: bool
    complete_hit: bool
    first_gt_rank: Optional[int]
    reciprocal_rank: float

    def to_dict(self):
        return {
            "matched_gt_chunk_ids": list(self.matched_gt_chunk_ids),
            "required_gt_coverage": self.required_gt_coverage,
            "any_hit": self.any_hit,
            "complete_hit": self.complete_hit,
            "first_gt_rank": self.first_gt_rank,
            "reciprocal_rank": self.reciprocal_rank,
        }


@dataclass(frozen=True)
class QueryEvaluationRecord:
    """一条题目在一个评估模式下的检索与构包诊断记录。"""

    query_id: str
    query_original: str
    required_chunk_ids: Tuple[str, ...]
    plan: QueryEvaluationPlan
    candidate_pool_chunk_ids: Tuple[str, ...]
    reranked_top5_chunk_ids: Tuple[str, ...]
    packaged_chunk_ids: Tuple[str, ...]
    packaging_status: PackagingStatus
    prompt_tokens: Optional[int]
    candidate_pool_metrics: StageMetrics
    reranked_top5_metrics: StageMetrics
    package_metrics: StageMetrics
    fifth_candidate_entered: bool
    fifth_candidate_completed_gt: bool

    def to_dict(self):
        return {
            "query_id": self.query_id,
            "mode": self.plan.mode.value,
            "query_original": self.query_original,
            "required_chunk_ids": list(self.required_chunk_ids),
            "query_enhancement": {
                "status": self.plan.status.value,
                "failure_reason": (
                    self.plan.failure_reason.value if self.plan.failure_reason else None
                ),
                "protocol_attempted": self.plan.protocol_attempted,
                "protocol_output_received": self.plan.protocol_output_received,
                "protocol_valid": self.plan.protocol_valid,
                "is_noop": self.plan.is_noop,
                "planned_retrieval_queries": list(
                    self.plan.planned_retrieval_queries
                ),
                "executed_retrieval_queries": list(
                    self.plan.executed_retrieval_queries
                ),
                "executed_leg_count": len(self.plan.executed_retrieval_queries),
            },
            "retrieval": {
                "candidate_pool_chunk_ids": list(self.candidate_pool_chunk_ids),
                "reranked_top5_chunk_ids": list(self.reranked_top5_chunk_ids),
                "candidate_pool_metrics": self.candidate_pool_metrics.to_dict(),
                "reranked_top5_metrics": self.reranked_top5_metrics.to_dict(),
            },
            "packaging": {
                "status": self.packaging_status.value,
                "packaged_chunk_ids": list(self.packaged_chunk_ids),
                "packaged_count": len(self.packaged_chunk_ids),
                "prompt_tokens": self.prompt_tokens,
                "metrics": self.package_metrics.to_dict(),
                "fifth_candidate_entered": self.fifth_candidate_entered,
                "fifth_candidate_completed_gt": self.fifth_candidate_completed_gt,
            },
        }


def _baseline_queries(original_query):
    return compile_retrieval_queries(
        original_query,
        QueryEnhancement(rewrite=original_query),
    )


def _fallback_plan(
    *,
    mode,
    original_query,
    failure_reason,
    output_received,
    protocol_valid,
):
    baseline = _baseline_queries(original_query)
    return QueryEvaluationPlan(
        mode=mode,
        status=QueryEnhancementStatus.FALLBACK,
        failure_reason=failure_reason,
        protocol_attempted=True,
        protocol_output_received=output_received,
        protocol_valid=protocol_valid,
        is_noop=False,
        planned_retrieval_queries=baseline,
        executed_retrieval_queries=baseline,
    )


def build_query_evaluation_plan(
    original_query,
    mode,
    *,
    raw_enhancement=None,
    failure_reason=None,
    query_formal=None,
):
    """从已物化增强输出构造一个评估模式，不调用增强模型。"""
    if not isinstance(mode, EvaluationMode):
        raise TypeError("mode 必须是 EvaluationMode")
    baseline = _baseline_queries(original_query)

    if mode is EvaluationMode.BASELINE_ORIGINAL:
        if raw_enhancement is not None or failure_reason is not None:
            raise ValueError("原始单 query 基线不能携带增强输出或失败原因")
        return QueryEvaluationPlan(
            mode=mode,
            status=QueryEnhancementStatus.NOT_ATTEMPTED,
            failure_reason=None,
            protocol_attempted=False,
            protocol_output_received=False,
            protocol_valid=None,
            is_noop=False,
            planned_retrieval_queries=baseline,
            executed_retrieval_queries=baseline,
        )

    if mode is EvaluationMode.NOOP_APPLIED:
        if raw_enhancement is not None or failure_reason is not None:
            raise ValueError("确定性 no-op 不能携带外部增强输出或失败原因")
        enhancement = QueryEnhancement(rewrite=original_query)
        compiled = compile_retrieval_queries(original_query, enhancement)
        return QueryEvaluationPlan(
            mode=mode,
            status=QueryEnhancementStatus.APPLIED,
            failure_reason=None,
            protocol_attempted=True,
            protocol_output_received=True,
            protocol_valid=True,
            is_noop=True,
            planned_retrieval_queries=compiled,
            executed_retrieval_queries=compiled,
        )

    if mode is EvaluationMode.ORACLE_FORMAL:
        if raw_enhancement is not None or failure_reason is not None:
            raise ValueError("formal Oracle 不能携带增强模型输出或失败原因")
        enhancement = QueryEnhancement(rewrite=query_formal)
        compiled = compile_retrieval_queries(original_query, enhancement)
        return QueryEvaluationPlan(
            mode=mode,
            status=QueryEnhancementStatus.APPLIED,
            failure_reason=None,
            protocol_attempted=False,
            protocol_output_received=False,
            protocol_valid=None,
            is_noop=compiled == baseline,
            planned_retrieval_queries=compiled,
            executed_retrieval_queries=compiled,
        )

    if failure_reason is not None:
        if not isinstance(failure_reason, QueryEnhancementFailureReason):
            raise TypeError("failure_reason 必须是 QueryEnhancementFailureReason")
        if failure_reason not in {
            QueryEnhancementFailureReason.CALL_FAILED,
            QueryEnhancementFailureReason.TIMEOUT,
        }:
            raise ValueError("物化前失败只能是 call_failed 或 timeout")
        if raw_enhancement is not None:
            raise ValueError("调用失败或超时时不能同时携带增强输出")
        return _fallback_plan(
            mode=mode,
            original_query=original_query,
            failure_reason=failure_reason,
            output_received=False,
            protocol_valid=None,
        )
    if raw_enhancement is None:
        raise ValueError("增强评估模式必须携带物化输出或调用失败原因")

    try:
        parsed = parse_and_validate_query_enhancement(raw_enhancement)
    except (QueryEnhancementProtocolError, TypeError):
        return _fallback_plan(
            mode=mode,
            original_query=original_query,
            failure_reason=QueryEnhancementFailureReason.INVALID_OUTPUT,
            output_received=True,
            protocol_valid=False,
        )

    if mode is EvaluationMode.REWRITE_ONLY:
        enhancement = QueryEnhancement(rewrite=parsed.rewrite)
    elif mode is EvaluationMode.TERMS_ONLY:
        enhancement = QueryEnhancement(
            rewrite=original_query,
            expansion_terms=parsed.expansion_terms,
        )
    elif mode is EvaluationMode.SUBQUERIES_ONLY:
        enhancement = QueryEnhancement(
            rewrite=original_query,
            subqueries=parsed.subqueries,
        )
    elif mode is EvaluationMode.FULL_ENHANCEMENT:
        enhancement = parsed
    else:
        raise AssertionError(f"未处理的评估模式: {mode}")

    compiled = compile_retrieval_queries(original_query, enhancement)
    return QueryEvaluationPlan(
        mode=mode,
        status=QueryEnhancementStatus.APPLIED,
        failure_reason=None,
        protocol_attempted=True,
        protocol_output_received=True,
        protocol_valid=True,
        is_noop=compiled == baseline,
        planned_retrieval_queries=compiled,
        executed_retrieval_queries=compiled,
    )


def evaluate_stage(required_chunk_ids, ranked_chunk_ids):
    """计算单题 required GT 覆盖与首个 GT 排名。"""
    required = tuple(required_chunk_ids)
    ranking = tuple(ranked_chunk_ids)
    if not required or len(required) != len(set(required)):
        raise ValueError("required_chunk_ids 必须非空且不能重复")
    if len(ranking) != len(set(ranking)):
        raise ValueError("ranked_chunk_ids 不能重复")
    ranking_set = set(ranking)
    matched = tuple(chunk_id for chunk_id in required if chunk_id in ranking_set)
    first_gt_rank = next(
        (
            index
            for index, chunk_id in enumerate(ranking, start=1)
            if chunk_id in set(required)
        ),
        None,
    )
    return StageMetrics(
        matched_gt_chunk_ids=matched,
        required_gt_coverage=len(matched) / len(required),
        any_hit=bool(matched),
        complete_hit=len(matched) == len(required),
        first_gt_rank=first_gt_rank,
        reciprocal_rank=0.0 if first_gt_rank is None else 1.0 / first_gt_rank,
    )


def evaluate_retrieval_case(case, plan, *, retriever, packager):
    """执行一题已规划检索并记录 candidate pool、top-5 与构包结果。"""
    if not isinstance(case, EvaluationCase):
        raise TypeError("case 必须是 EvaluationCase")
    if not isinstance(plan, QueryEvaluationPlan):
        raise TypeError("plan 必须是 QueryEvaluationPlan")

    try:
        candidate_pool = tuple(
            retriever.retrieve_candidates_many(plan.executed_retrieval_queries)
        )
    except Exception:
        if plan.status is not QueryEnhancementStatus.APPLIED:
            raise
        baseline = _baseline_queries(case.query_original)
        plan = replace(
            plan,
            status=QueryEnhancementStatus.FALLBACK,
            failure_reason=QueryEnhancementFailureReason.ENHANCED_RETRIEVAL_FAILED,
            is_noop=False,
            executed_retrieval_queries=baseline,
        )
        candidate_pool = tuple(retriever.retrieve_candidates_many(baseline))

    ranked = tuple(
        retriever.rerank_candidates(case.query_original, candidate_pool)
    )[:5]
    candidate_pool_ids = tuple(item.chunk_id for item in candidate_pool)
    top5_ids = tuple(item.article.chunk_id for item in ranked)

    packaged_ids = ()
    prompt_tokens = None
    if not ranked:
        packaging_status = PackagingStatus.NOT_ATTEMPTED
    else:
        try:
            package, prompt_tokens = packager.build(
                case.query_original,
                tuple(item.article for item in ranked),
            )
        except EvidencePackagingError:
            packaging_status = PackagingStatus.FAILED
        else:
            packaging_status = PackagingStatus.APPLIED
            packaged_ids = top5_ids[: len(package.evidence)]

    required = case.required_chunk_ids
    package_metrics = evaluate_stage(required, packaged_ids)
    top4_metrics = evaluate_stage(required, packaged_ids[:4])
    fifth_entered = len(packaged_ids) == 5
    return QueryEvaluationRecord(
        query_id=case.query_id,
        query_original=case.query_original,
        required_chunk_ids=required,
        plan=plan,
        candidate_pool_chunk_ids=candidate_pool_ids,
        reranked_top5_chunk_ids=top5_ids,
        packaged_chunk_ids=packaged_ids,
        packaging_status=packaging_status,
        prompt_tokens=prompt_tokens,
        candidate_pool_metrics=evaluate_stage(required, candidate_pool_ids),
        reranked_top5_metrics=evaluate_stage(required, top5_ids),
        package_metrics=package_metrics,
        fifth_candidate_entered=fifth_entered,
        fifth_candidate_completed_gt=(
            fifth_entered
            and not top4_metrics.complete_hit
            and package_metrics.complete_hit
        ),
    )


def _aggregate_stage(records, attribute):
    metrics = [getattr(record, attribute) for record in records]
    return {
        "any_hit": fmean(metric.any_hit for metric in metrics),
        "required_gt_coverage": fmean(
            metric.required_gt_coverage for metric in metrics
        ),
        "complete_hit": fmean(metric.complete_hit for metric in metrics),
        "mrr": fmean(metric.reciprocal_rank for metric in metrics),
    }


def _paired_delta(records, baseline_by_id, attribute, metric):
    counts = {"improved": 0, "tied": 0, "degraded": 0}
    for record in records:
        baseline = baseline_by_id[record.query_id]
        candidate_value = getattr(getattr(record, attribute), metric)
        baseline_value = getattr(getattr(baseline, attribute), metric)
        if candidate_value > baseline_value:
            counts["improved"] += 1
        elif candidate_value < baseline_value:
            counts["degraded"] += 1
        else:
            counts["tied"] += 1
    return counts


def _paired_metric(
    records,
    baseline_by_id,
    attribute,
    metric,
    *,
    bootstrap_samples=2_000,
    seed=42,
):
    """汇总逐题差值，并给出可复现的 paired bootstrap 95% 区间。"""

    deltas = []
    for record in records:
        baseline = baseline_by_id[record.query_id]
        current = getattr(getattr(record, attribute), metric)
        previous = getattr(getattr(baseline, attribute), metric)
        deltas.append(float(current) - float(previous))
    counts = {
        "improved": sum(value > 0 for value in deltas),
        "tied": sum(value == 0 for value in deltas),
        "degraded": sum(value < 0 for value in deltas),
    }
    random = Random(seed)
    means = sorted(
        fmean(random.choice(deltas) for _ in deltas)
        for _ in range(bootstrap_samples)
    )
    lower = means[int(0.025 * bootstrap_samples)]
    upper = means[min(bootstrap_samples - 1, int(0.975 * bootstrap_samples))]
    return {
        **counts,
        "mean_delta": fmean(deltas),
        "confidence_interval_95": [lower, upper],
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
    }


def summarize_mode(records, *, baseline_records=()):
    """聚合一个模式，并按需给出相对原始单 query 的逐题变化。"""
    values = tuple(records)
    if not values:
        raise ValueError("records 不能为空")
    modes = {record.plan.mode for record in values}
    if len(modes) != 1:
        raise ValueError("一次只能聚合同一个评估模式")

    attempted = [record for record in values if record.plan.protocol_attempted]
    outputs = [record for record in values if record.plan.protocol_output_received]
    applied = [
        record
        for record in attempted
        if record.plan.status is QueryEnhancementStatus.APPLIED
    ]
    packaging_distribution = {
        str(count): sum(len(record.packaged_chunk_ids) == count for record in values)
        for count in range(1, 6)
    }
    fifth_eligible = [
        record for record in values if len(record.reranked_top5_chunk_ids) == 5
    ]
    summary = {
        "mode": next(iter(modes)).value,
        "question_count": len(values),
        "retrieval": {
            "candidate_pool": _aggregate_stage(values, "candidate_pool_metrics"),
            "reranked_top5": _aggregate_stage(values, "reranked_top5_metrics"),
        },
        "query_enhancement": {
            "protocol_attempted_count": len(attempted),
            "protocol_output_count": len(outputs),
            "protocol_legal_rate": (
                None
                if not outputs
                else sum(record.plan.protocol_valid is True for record in outputs)
                / len(outputs)
            ),
            "fallback_rate": (
                None
                if not attempted
                else sum(
                    record.plan.status is QueryEnhancementStatus.FALLBACK
                    for record in attempted
                )
                / len(attempted)
            ),
            "noop_rate": (
                None
                if not applied
                else sum(record.plan.is_noop for record in applied) / len(applied)
            ),
            "average_executed_leg_count": fmean(
                len(record.plan.executed_retrieval_queries) for record in values
            ),
            "fallback_reason_counts": {
                reason.value: sum(record.plan.failure_reason is reason for record in values)
                for reason in QueryEnhancementFailureReason
            },
        },
        "packaging": {
            "evidence_count_distribution": packaging_distribution,
            "required_gt": _aggregate_stage(values, "package_metrics"),
            "failed_count": sum(
                record.packaging_status is PackagingStatus.FAILED
                for record in values
            ),
            "fifth_candidate_eligible_count": len(fifth_eligible),
            "fifth_candidate_entered_count": sum(
                record.fifth_candidate_entered for record in fifth_eligible
            ),
            "fifth_candidate_entry_rate": (
                None
                if not fifth_eligible
                else sum(
                    record.fifth_candidate_entered for record in fifth_eligible
                )
                / len(fifth_eligible)
            ),
            "fifth_candidate_completed_gt_count": sum(
                record.fifth_candidate_completed_gt for record in values
            ),
        },
    }

    baselines = tuple(baseline_records)
    if baselines:
        baseline_by_id: Mapping[str, QueryEvaluationRecord] = {
            record.query_id: record for record in baselines
        }
        if set(baseline_by_id) != {record.query_id for record in values}:
            raise ValueError("baseline_records 必须与当前模式包含相同 query_id")
        summary["paired_vs_baseline"] = {
            "complete_hit": _paired_delta(
                values,
                baseline_by_id,
                "reranked_top5_metrics",
                "complete_hit",
            ),
            "required_gt_coverage": _paired_delta(
                values,
                baseline_by_id,
                "reranked_top5_metrics",
                "required_gt_coverage",
            ),
            "reciprocal_rank": _paired_delta(
                values,
                baseline_by_id,
                "reranked_top5_metrics",
                "reciprocal_rank",
            ),
            "primary_metrics": {
                "reranked_top5_complete_hit": _paired_metric(
                    values,
                    baseline_by_id,
                    "reranked_top5_metrics",
                    "complete_hit",
                ),
                "packaged_complete_hit": _paired_metric(
                    values,
                    baseline_by_id,
                    "package_metrics",
                    "complete_hit",
                ),
            },
            "packaged_required_gt_coverage": _paired_metric(
                values,
                baseline_by_id,
                "package_metrics",
                "required_gt_coverage",
            ),
        }
    return summary


__all__ = [
    "EvaluationCase",
    "EvaluationMode",
    "PackagingStatus",
    "QueryEvaluationPlan",
    "QueryEvaluationRecord",
    "StageMetrics",
    "build_query_evaluation_plan",
    "evaluate_retrieval_case",
    "evaluate_stage",
    "summarize_mode",
]
