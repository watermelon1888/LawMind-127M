"""比较 reranker 与候选 RRF 名次融合的 top-5 效果。"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from statistics import fmean

from rag.answering import AnswerPromptTokenCounter, EvidencePackager, EvidencePackagingError
from rag.eval import retrieval_baseline as baseline
from rag.eval.retrieval_chain_evaluation import evaluate_stage
from rag.knowledge import ArticleRepository
from rag.retrieval import SemanticRetrievalConfig
from rag.retrieval.loader import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    _resolve_device,
    load_semantic_retriever,
)


DEFAULT_BASELINE_RECORDS = (
    baseline.DEFAULT_OUTPUT_DIR / "records.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    baseline.RAG_DIR / "eval" / "results" / "rerank-rank-fusion-original-v1"
)
PIPELINE = "legal_rag_rerank_rank_fusion_original_v1"
SCHEMA_VERSION = "1.0"
FUSION_K = 60


@dataclass(frozen=True)
class FusionSpec:
    """一个固定的 reranker 与 RRF 名次融合权重。"""

    name: str
    reranker_weight: float

    @property
    def rrf_weight(self):
        return 1.0 - self.reranker_weight


FUSION_SPECS = (
    FusionSpec("reranker_only", 1.0),
    FusionSpec("reranker_3_rrf_1", 0.75),
    FusionSpec("equal_rank_fusion", 0.5),
    FusionSpec("reranker_1_rrf_3", 0.25),
    FusionSpec("rrf_only", 0.0),
)


def fuse_rankings(
    candidate_pool_chunk_ids,
    reranked_chunk_ids,
    *,
    reranker_weight,
    fusion_k=FUSION_K,
):
    """用固定权重融合候选 RRF 名次和 reranker 名次。"""
    pool = tuple(candidate_pool_chunk_ids)
    reranked = tuple(reranked_chunk_ids)
    if not pool or len(pool) != len(set(pool)):
        raise ValueError("candidate pool 必须非空且不能重复")
    if len(reranked) != len(set(reranked)) or set(reranked) != set(pool):
        raise ValueError("reranked_chunk_ids 必须是 candidate pool 的完整排列")
    if (
        not isinstance(reranker_weight, (int, float))
        or isinstance(reranker_weight, bool)
        or not 0.0 <= float(reranker_weight) <= 1.0
    ):
        raise ValueError("reranker_weight 必须位于 0 到 1")
    if not isinstance(fusion_k, int) or isinstance(fusion_k, bool) or fusion_k <= 0:
        raise ValueError("fusion_k 必须是正整数")

    reranker_weight = float(reranker_weight)
    rrf_weight = 1.0 - reranker_weight
    pool_ranks = {chunk_id: rank for rank, chunk_id in enumerate(pool, start=1)}
    reranker_ranks = {
        chunk_id: rank for rank, chunk_id in enumerate(reranked, start=1)
    }

    def sort_key(chunk_id):
        score = (
            reranker_weight / (fusion_k + reranker_ranks[chunk_id])
            + rrf_weight / (fusion_k + pool_ranks[chunk_id])
        )
        return (
            -score,
            reranker_ranks[chunk_id],
            pool_ranks[chunk_id],
            chunk_id,
        )

    return tuple(sorted(pool, key=sort_key))


def _package_ranking(query, ranking, *, repository, packager):
    top5 = tuple(ranking[:5])
    if not top5:
        return (), "not_attempted", None
    try:
        package, prompt_tokens = packager.build(
            query,
            tuple(repository.get_by_chunk_id(chunk_id) for chunk_id in top5),
        )
    except EvidencePackagingError:
        return (), "failed", None
    return top5[: len(package.evidence)], "applied", prompt_tokens


def _configuration_result(
    case,
    ranking,
    *,
    repository,
    packager,
):
    top5 = tuple(ranking[:5])
    packaged, packaging_status, prompt_tokens = _package_ranking(
        case.query_original,
        ranking,
        repository=repository,
        packager=packager,
    )
    top5_metrics = evaluate_stage(case.required_chunk_ids, top5)
    package_metrics = evaluate_stage(case.required_chunk_ids, packaged)
    top4_package_metrics = evaluate_stage(case.required_chunk_ids, packaged[:4])
    fifth_entered = len(packaged) == 5
    return {
        "top5_chunk_ids": list(top5),
        "top5_metrics": top5_metrics.to_dict(),
        "packaging": {
            "status": packaging_status,
            "packaged_chunk_ids": list(packaged),
            "packaged_count": len(packaged),
            "prompt_tokens": prompt_tokens,
            "metrics": package_metrics.to_dict(),
            "fifth_candidate_entered": fifth_entered,
            "fifth_candidate_completed_gt": (
                fifth_entered
                and not top4_package_metrics.complete_hit
                and package_metrics.complete_hit
            ),
        },
    }


def _load_baseline_records(path):
    records = {}
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"基线逐题记录第 {line_number} 行不是有效 JSON"
                ) from exc
            query_id = item.get("query_id")
            if not isinstance(query_id, str) or query_id in records:
                raise ValueError("基线逐题记录的 query_id 无效或重复")
            records[query_id] = item
    if len(records) != baseline.EXPECTED_CASE_COUNT:
        raise ValueError(
            f"基线逐题记录必须包含 {baseline.EXPECTED_CASE_COUNT} 条"
        )
    return records


def _assert_reproduces_baseline(query_id, configuration, baseline_record):
    expected_top5 = baseline_record["retrieval"]["reranked_top5_chunk_ids"]
    expected_packaged = baseline_record["packaging"]["packaged_chunk_ids"]
    expected_status = baseline_record["packaging"]["status"]
    if configuration["top5_chunk_ids"] != expected_top5:
        raise RuntimeError(f"{query_id} 纯 reranker top-5 未复现固定基线")
    if (
        configuration["packaging"]["packaged_chunk_ids"] != expected_packaged
        or configuration["packaging"]["status"] != expected_status
    ):
        raise RuntimeError(f"{query_id} 构包结果未复现固定基线")


def evaluate_cases(
    cases,
    *,
    retriever,
    repository,
    packager,
    baseline_records,
    progress_every=10,
):
    """执行完整 reranker 排名并生成所有融合配置的逐题结果。"""
    values = tuple(cases)
    if not values:
        raise ValueError("cases 不能为空")
    records = []
    started = time.perf_counter()
    for index, case in enumerate(values, start=1):
        pool = tuple(retriever.retrieve_candidates_many((case.query_original,)))
        reranked = tuple(retriever.rerank_candidates(case.query_original, pool))
        pool_ids = tuple(item.chunk_id for item in pool)
        reranked_ids = tuple(item.article.chunk_id for item in reranked)
        if set(pool_ids) != set(reranked_ids):
            raise RuntimeError(f"{case.query_id} reranker 未返回完整 candidate pool")

        configurations = {}
        for spec in FUSION_SPECS:
            ranking = fuse_rankings(
                pool_ids,
                reranked_ids,
                reranker_weight=spec.reranker_weight,
            )
            configurations[spec.name] = _configuration_result(
                case,
                ranking,
                repository=repository,
                packager=packager,
            )
        _assert_reproduces_baseline(
            case.query_id,
            configurations["reranker_only"],
            baseline_records[case.query_id],
        )
        records.append(
            {
                "query_id": case.query_id,
                "query_original": case.query_original,
                "required_chunk_ids": list(case.required_chunk_ids),
                "candidate_pool": [
                    {
                        "chunk_id": item.chunk_id,
                        "rrf_rank": item.rank,
                        "rrf_score": item.score,
                    }
                    for item in pool
                ],
                "reranker_full_ranking": [
                    {
                        "chunk_id": item.article.chunk_id,
                        "rerank_rank": rank,
                        "rerank_score": item.rerank_score,
                    }
                    for rank, item in enumerate(reranked, start=1)
                ],
                "configurations": configurations,
            }
        )
        if progress_every and (index % progress_every == 0 or index == len(values)):
            elapsed = time.perf_counter() - started
            print(
                f"RERANK_FUSION_PROGRESS completed={index}/{len(values)} "
                f"elapsed_seconds={elapsed:.2f}",
                flush=True,
            )
    return tuple(records), time.perf_counter() - started


def _average_metrics(records, spec_name, section, metrics_key=None):
    if section == "top5_metrics":
        values = [record["configurations"][spec_name][section] for record in records]
    else:
        values = [
            record["configurations"][spec_name][section][metrics_key]
            for record in records
        ]
    return {
        "any_hit": fmean(item["any_hit"] for item in values),
        "required_gt_coverage": fmean(
            item["required_gt_coverage"] for item in values
        ),
        "complete_hit": fmean(item["complete_hit"] for item in values),
        "mrr": fmean(item["reciprocal_rank"] for item in values),
    }


def _paired_counts(records, spec_name, path):
    counts = {"improved": 0, "tied": 0, "degraded": 0}
    for record in records:
        baseline_value = record["configurations"]["reranker_only"]
        candidate_value = record["configurations"][spec_name]
        for key in path:
            baseline_value = baseline_value[key]
            candidate_value = candidate_value[key]
        if candidate_value > baseline_value:
            counts["improved"] += 1
        elif candidate_value < baseline_value:
            counts["degraded"] += 1
        else:
            counts["tied"] += 1
    return counts


def summarize(records, *, elapsed_seconds):
    """聚合每个融合权重及其相对纯 reranker 的逐题变化。"""
    values = tuple(records)
    configurations = {}
    for spec in FUSION_SPECS:
        packaged_counts = {
            str(count): sum(
                record["configurations"][spec.name]["packaging"]["packaged_count"]
                == count
                for record in values
            )
            for count in range(0, 6)
        }
        configurations[spec.name] = {
            "reranker_weight": spec.reranker_weight,
            "rrf_weight": spec.rrf_weight,
            "top5": _average_metrics(values, spec.name, "top5_metrics"),
            "packaging": {
                "required_gt": _average_metrics(
                    values,
                    spec.name,
                    "packaging",
                    "metrics",
                ),
                "evidence_count_distribution": packaged_counts,
                "failed_count": sum(
                    record["configurations"][spec.name]["packaging"]["status"]
                    == "failed"
                    for record in values
                ),
            },
            "paired_vs_reranker": {
                "top5_complete_hit": _paired_counts(
                    values,
                    spec.name,
                    ("top5_metrics", "complete_hit"),
                ),
                "top5_required_gt_coverage": _paired_counts(
                    values,
                    spec.name,
                    ("top5_metrics", "required_gt_coverage"),
                ),
                "package_complete_hit": _paired_counts(
                    values,
                    spec.name,
                    ("packaging", "metrics", "complete_hit"),
                ),
            },
        }
    return {
        "question_count": len(values),
        "fusion_k": FUSION_K,
        "configurations": configurations,
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "average_seconds_per_question": elapsed_seconds / len(values),
        },
    }


def _percent(value):
    return f"{value * 100:.2f}%"


def _delta_text(counts):
    return f"+{counts['improved']} / -{counts['degraded']}"


def _changed_ids(records, spec_name, path, direction):
    result = []
    for record in records:
        baseline_value = record["configurations"]["reranker_only"]
        candidate_value = record["configurations"][spec_name]
        for key in path:
            baseline_value = baseline_value[key]
            candidate_value = candidate_value[key]
        changed = (
            candidate_value > baseline_value
            if direction == "improved"
            else candidate_value < baseline_value
        )
        if changed:
            result.append(record["query_id"])
    return result


def select_report_leader(summary):
    """按 top-5 complete、coverage、构包 complete 和 MRR 选择报告首位。"""
    configurations = summary["configurations"]
    return max(
        FUSION_SPECS,
        key=lambda spec: (
            configurations[spec.name]["top5"]["complete_hit"],
            configurations[spec.name]["top5"]["required_gt_coverage"],
            configurations[spec.name]["packaging"]["required_gt"][
                "complete_hit"
            ],
            configurations[spec.name]["top5"]["mrr"],
        ),
    ).name


def render_report(summary, records, *, manifest):
    """生成 rank-fusion 对比报告。"""
    configurations = summary["configurations"]
    leader = select_report_leader(summary)
    lines = [
        "# Reranker 与 RRF 名次融合评估",
        "",
        "> 本实验使用原始单 query candidate pool，不包含尚未物化的真实 Query 增强输出。",
        "> 结果只用于选择 rerank 候选方案，不自动修改生产排序。",
        "",
        "## 评估范围",
        "",
        f"- 样本：{summary['question_count']} 条 `legal_query + answer`。",
        "- candidate pool：当前 dense、BM25 和 RRF 固定参数下的 20 条候选。",
        "- reranker：使用原始 query 对完整 candidate pool 排序。",
        f"- 名次融合：`weight / ({summary['fusion_k']} + rank)`。",
        "- 所有配置统一截取 top-5，并重新执行真实 token 构包。",
        "",
        "## 汇总结果",
        "",
        "| 配置 | reranker:RRF | complete@5 | Recall@5 | any@5 | MRR@5 | top-5 完整命中 + / - | 构包 complete | 构包 + / - |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for spec in FUSION_SPECS:
        item = configurations[spec.name]
        top5 = item["top5"]
        package = item["packaging"]["required_gt"]
        paired = item["paired_vs_reranker"]
        lines.append(
            f"| `{spec.name}` | {spec.reranker_weight:.2f}:{spec.rrf_weight:.2f} | "
            f"{_percent(top5['complete_hit'])} | {_percent(top5['required_gt_coverage'])} | "
            f"{_percent(top5['any_hit'])} | {top5['mrr']:.4f} | "
            f"{_delta_text(paired['top5_complete_hit'])} | "
            f"{_percent(package['complete_hit'])} | "
            f"{_delta_text(paired['package_complete_hit'])} |"
        )

    top5_improved = _changed_ids(
        records,
        leader,
        ("top5_metrics", "complete_hit"),
        "improved",
    )
    top5_degraded = _changed_ids(
        records,
        leader,
        ("top5_metrics", "complete_hit"),
        "degraded",
    )
    package_improved = _changed_ids(
        records,
        leader,
        ("packaging", "metrics", "complete_hit"),
        "improved",
    )
    package_degraded = _changed_ids(
        records,
        leader,
        ("packaging", "metrics", "complete_hit"),
        "degraded",
    )
    lines.extend(
        [
            "",
            "## 报告首位配置",
            "",
            f"按 complete@5、Recall@5、构包 complete、MRR@5 依次排序，首位为 `{leader}`。",
            "该排序只是离线候选结论；真实 Query 增强接入后必须在增强 candidate pool 上重跑。",
            "",
            "### Top-5 完整覆盖变化",
            "",
            f"- 提升：{'、'.join(top5_improved) if top5_improved else '无'}",
            f"- 退化：{'、'.join(top5_degraded) if top5_degraded else '无'}",
            "",
            "### 构包完整覆盖变化",
            "",
            f"- 提升：{'、'.join(package_improved) if package_improved else '无'}",
            f"- 退化：{'、'.join(package_degraded) if package_degraded else '无'}",
            "",
            "## 运行身份",
            "",
            f"- embedding：`{manifest['models']['embedding']}`。",
            f"- reranker：`{manifest['models']['reranker']}`。",
            f"- 设备：`{manifest['device']}`。",
            f"- 生成时间：{manifest['generated_at']}。",
            f"- 完整 reranker 评分耗时：{summary['runtime']['elapsed_seconds']:.2f} 秒。",
            "",
            "## 复现命令",
            "",
            "```powershell",
            "conda run --no-capture-output -n minimind python -m rag.eval.rerank_fusion_evaluation",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def publish_results(output_dir, *, records, summary, manifest):
    """发布逐题记录、汇总、报告和哈希清单。"""
    directory = Path(output_dir).resolve()
    paths = {
        "records": directory / "records.jsonl",
        "summary": directory / "summary.json",
        "report": directory / "report.md",
        "manifest": directory / "manifest.json",
    }
    for path in paths.values():
        if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
            raise FileExistsError(f"输出已存在，不能覆盖: {path}")
    report = render_report(summary, records, manifest=manifest)
    output_identities = {
        "records": {
            "path": paths["records"].name,
            "records": len(records),
            "sha256": baseline._write_immutable_jsonl(paths["records"], records),
        },
        "summary": {
            "path": paths["summary"].name,
            "sha256": baseline._write_immutable_json(paths["summary"], summary),
        },
        "report": {
            "path": paths["report"].name,
            "sha256": baseline._write_immutable_text(paths["report"], report),
        },
    }
    final_manifest = dict(manifest)
    final_manifest["outputs"] = output_identities
    baseline._write_immutable_json(paths["manifest"], final_manifest)
    return final_manifest


def run_evaluation(
    *,
    eval_set,
    article_index,
    artifact_dir,
    tokenizer_path,
    baseline_records_path,
    output_dir,
    device,
    limit=None,
):
    """运行固定原始 query candidate pool 的 rank-fusion 比较。"""
    repository = ArticleRepository.from_jsonl(article_index)
    all_cases = baseline.load_baseline_cases(eval_set, repository)
    baseline_records = _load_baseline_records(baseline_records_path)
    if {case.query_id for case in all_cases} != set(baseline_records):
        raise ValueError("当前评估集与固定基线逐题记录的 query_id 不一致")
    cases = all_cases
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit 必须是正整数")
        cases = all_cases[:limit]

    tokenizer = baseline._load_tokenizer(tokenizer_path)
    packager = EvidencePackager(
        context_limit=baseline.CONTEXT_LIMIT,
        max_output_tokens=baseline.MAX_OUTPUT_TOKENS,
        count_prompt_tokens=AnswerPromptTokenCounter(tokenizer),
    )
    production_config = SemanticRetrievalConfig()
    diagnostic_config = replace(
        production_config,
        top_k=production_config.candidate_pool,
    )
    actual_device = _resolve_device(device)
    retriever = load_semantic_retriever(
        repository=repository,
        artifact_dir=artifact_dir,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        reranker_model=DEFAULT_RERANKER_MODEL,
        device=actual_device,
        config=diagnostic_config,
    )
    records, elapsed_seconds = evaluate_cases(
        cases,
        retriever=retriever,
        repository=repository,
        packager=packager,
        baseline_records=baseline_records,
    )
    summary = summarize(records, elapsed_seconds=elapsed_seconds)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    artifact_path = Path(artifact_dir).resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": PIPELINE,
        "generated_at": generated_at,
        "complete_dataset": len(cases) == len(all_cases),
        "records": len(records),
        "inputs": {
            "eval_set": baseline._file_identity(eval_set),
            "article_index": baseline._file_identity(article_index),
            "baseline_records": baseline._file_identity(baseline_records_path),
            "retrieval_artifacts": {
                filename: baseline._file_identity(artifact_path / filename)
                for filename in baseline._ARTIFACT_FILENAMES
            },
            "tokenizer": baseline._tokenizer_identity(tokenizer, tokenizer_path),
        },
        "models": {
            "embedding": DEFAULT_EMBEDDING_MODEL,
            "reranker": DEFAULT_RERANKER_MODEL,
            "query_enhancer": None,
            "answer_model": None,
        },
        "device": actual_device,
        "production_retrieval_config": asdict(production_config),
        "diagnostic_rerank_depth": diagnostic_config.top_k,
        "evaluated_top_k": production_config.top_k,
        "fusion_k": FUSION_K,
        "fusion_specs": [asdict(spec) for spec in FUSION_SPECS],
        "production_changed": False,
    }
    final_manifest = publish_results(
        output_dir,
        records=records,
        summary=summary,
        manifest=manifest,
    )
    print(
        f"RERANK_FUSION_OK records={len(records)} output={Path(output_dir).resolve()}",
        flush=True,
    )
    return final_manifest


def _parser():
    parser = argparse.ArgumentParser(description="评估 reranker 与 RRF 名次融合")
    parser.add_argument("--eval-set", default=str(baseline.DEFAULT_EVAL_SET))
    parser.add_argument("--article-index", default=str(baseline.DEFAULT_ARTICLE_INDEX))
    parser.add_argument("--artifact-dir", default=str(baseline.DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--tokenizer-path", default=str(baseline.DEFAULT_TOKENIZER_PATH))
    parser.add_argument("--baseline-records", default=str(DEFAULT_BASELINE_RECORDS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    return parser


def main():
    args = _parser().parse_args()
    try:
        run_evaluation(
            eval_set=args.eval_set,
            article_index=args.article_index,
            artifact_dir=args.artifact_dir,
            tokenizer_path=args.tokenizer_path,
            baseline_records_path=args.baseline_records,
            output_dir=args.output_dir,
            device=args.device,
            limit=args.limit,
        )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        raise SystemExit(f"[失败] {error}") from error


if __name__ == "__main__":
    main()
