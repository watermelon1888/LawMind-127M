"""复用已保存的完整排名，诊断窄权重融合与 top-5 边界保护策略。"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace

from rag.answering import AnswerPromptTokenCounter, EvidencePackager
from rag.eval import retrieval_baseline as baseline
from rag.eval.rerank_fusion_evaluation import (
    _configuration_result,
    fuse_rankings,
)
from rag.knowledge import ArticleRepository


DEFAULT_SOURCE_RECORDS = (
    baseline.RAG_DIR
    / "eval"
    / "results"
    / "rerank-rank-fusion-original-v1"
    / "records.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    baseline.RAG_DIR
    / "eval"
    / "results"
    / "rerank-boundary-policy-original-v2"
)
PIPELINE = "legal_rag_rerank_boundary_policy_original_v2"
SCHEMA_VERSION = "1.0"
MAX_CHALLENGER_RANK = 10


@dataclass(frozen=True)
class PolicySpec:
    """一个固定的离线 rerank 策略。"""

    name: str
    kind: str
    value: float | None = None


POLICY_SPECS = (
    PolicySpec("reranker_only", "baseline"),
    PolicySpec("global_fusion_0975", "global_fusion", 0.975),
    PolicySpec("global_fusion_095", "global_fusion", 0.95),
    PolicySpec("global_fusion_090", "global_fusion", 0.90),
    PolicySpec("boundary_gap_0005", "boundary", 0.005),
    PolicySpec("boundary_gap_001", "boundary", 0.01),
    PolicySpec("boundary_gap_002", "boundary", 0.02),
    PolicySpec("boundary_gap_005", "boundary", 0.05),
)


def select_boundary_ranking(
    candidate_pool_chunk_ids,
    reranked_items,
    *,
    max_score_gap,
    max_challenger_rank=MAX_CHALLENGER_RANK,
):
    """保留前四，仅在局部分差足够小时用更高 RRF 名次替换第五名。"""
    pool = tuple(candidate_pool_chunk_ids)
    ranked = tuple(reranked_items)
    reranked_ids = tuple(item[0] for item in ranked)
    if len(pool) < 5 or len(pool) != len(set(pool)):
        raise ValueError("candidate pool 必须至少有 5 条且不能重复")
    if len(reranked_ids) != len(set(reranked_ids)) or set(reranked_ids) != set(pool):
        raise ValueError("reranked_items 必须是 candidate pool 的完整排列")
    if (
        not isinstance(max_score_gap, (int, float))
        or isinstance(max_score_gap, bool)
        or float(max_score_gap) < 0.0
    ):
        raise ValueError("max_score_gap 必须是非负数")
    if (
        not isinstance(max_challenger_rank, int)
        or isinstance(max_challenger_rank, bool)
        or not 6 <= max_challenger_rank <= len(ranked)
    ):
        raise ValueError("max_challenger_rank 必须覆盖第 6 名且不超过排名长度")

    pool_ranks = {chunk_id: rank for rank, chunk_id in enumerate(pool, start=1)}
    cutoff_id, cutoff_score = ranked[4]
    eligible = []
    for rerank_rank, (chunk_id, score) in enumerate(
        ranked[5:max_challenger_rank],
        start=6,
    ):
        score_gap = float(cutoff_score) - float(score)
        if (
            score_gap <= float(max_score_gap) + 1e-12
            and pool_ranks[chunk_id] < pool_ranks[cutoff_id]
        ):
            eligible.append((pool_ranks[chunk_id], rerank_rank, chunk_id, score_gap))
    if not eligible:
        return reranked_ids, None

    pool_rank, rerank_rank, selected_id, score_gap = min(eligible)
    ranking = list(reranked_ids)
    ranking.remove(selected_id)
    ranking.insert(4, selected_id)
    return tuple(ranking), {
        "selected_chunk_id": selected_id,
        "replaced_chunk_id": cutoff_id,
        "selected_pool_rank": pool_rank,
        "selected_rerank_rank": rerank_rank,
        "score_gap": score_gap,
    }


def _load_source_records(path):
    records = []
    seen = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"源记录第 {line_number} 行不是有效 JSON") from exc
            query_id = item.get("query_id")
            if not isinstance(query_id, str) or query_id in seen:
                raise ValueError("源记录 query_id 无效或重复")
            seen.add(query_id)
            records.append(item)
    if len(records) != baseline.EXPECTED_CASE_COUNT:
        raise ValueError(
            f"源记录必须包含 {baseline.EXPECTED_CASE_COUNT} 条，实际为 {len(records)} 条"
        )
    return tuple(records)


def _source_ranking(record):
    pool_ids = tuple(item["chunk_id"] for item in record["candidate_pool"])
    ranked = tuple(
        (item["chunk_id"], float(item["rerank_score"]))
        for item in record["reranker_full_ranking"]
    )
    return pool_ids, ranked


def evaluate_records(source_records, *, repository, packager):
    """对固定完整排名执行全部离线策略与真实 token 构包。"""
    started = time.perf_counter()
    output = []
    for record in source_records:
        pool_ids, ranked = _source_ranking(record)
        reranked_ids = tuple(item[0] for item in ranked)
        configurations = {}
        decisions = {}
        for spec in POLICY_SPECS:
            decision = None
            if spec.kind == "baseline":
                ranking = reranked_ids
            elif spec.kind == "global_fusion":
                ranking = fuse_rankings(
                    pool_ids,
                    reranked_ids,
                    reranker_weight=spec.value,
                )
            elif spec.kind == "boundary":
                ranking, decision = select_boundary_ranking(
                    pool_ids,
                    ranked,
                    max_score_gap=spec.value,
                )
            else:
                raise ValueError(f"未知策略类型: {spec.kind}")
            configurations[spec.name] = _configuration_result(
                SimpleNamespace(
                    query_original=record["query_original"],
                    required_chunk_ids=tuple(record["required_chunk_ids"]),
                ),
                ranking,
                repository=repository,
                packager=packager,
            )
            decisions[spec.name] = decision

        expected = record["configurations"]["reranker_only"]
        actual = configurations["reranker_only"]
        if actual["top5_chunk_ids"] != expected["top5_chunk_ids"] or actual[
            "packaging"
        ]["packaged_chunk_ids"] != expected["packaging"]["packaged_chunk_ids"]:
            raise RuntimeError(f"{record['query_id']} 未复现第一轮固定基线")
        output.append(
            {
                "query_id": record["query_id"],
                "query_original": record["query_original"],
                "required_chunk_ids": record["required_chunk_ids"],
                "boundary": {
                    "reranker_rank5": ranked[4][0],
                    "reranker_rank5_score": ranked[4][1],
                    "decisions": decisions,
                },
                "configurations": configurations,
            }
        )
    return tuple(output), time.perf_counter() - started


def _average_metrics(records, spec_name, *, packaging=False):
    values = []
    for record in records:
        configuration = record["configurations"][spec_name]
        values.append(
            configuration["packaging"]["metrics"]
            if packaging
            else configuration["top5_metrics"]
        )
    return {
        "any_hit": fmean(item["any_hit"] for item in values),
        "required_gt_coverage": fmean(item["required_gt_coverage"] for item in values),
        "complete_hit": fmean(item["complete_hit"] for item in values),
        "mrr": fmean(item["reciprocal_rank"] for item in values),
    }


def _paired(records, spec_name, path):
    result = {"improved": [], "tied": [], "degraded": []}
    for record in records:
        baseline_value = record["configurations"]["reranker_only"]
        candidate_value = record["configurations"][spec_name]
        for key in path:
            baseline_value = baseline_value[key]
            candidate_value = candidate_value[key]
        if candidate_value > baseline_value:
            result["improved"].append(record["query_id"])
        elif candidate_value < baseline_value:
            result["degraded"].append(record["query_id"])
        else:
            result["tied"].append(record["query_id"])
    return {key: {"count": len(ids), "query_ids": ids} for key, ids in result.items()}


def summarize(records, *, elapsed_seconds):
    values = tuple(records)
    configurations = {}
    for spec in POLICY_SPECS:
        configurations[spec.name] = {
            "spec": asdict(spec),
            "changed_ranking_count": sum(
                record["configurations"][spec.name]["top5_chunk_ids"]
                != record["configurations"]["reranker_only"]["top5_chunk_ids"]
                for record in values
            ),
            "top5": _average_metrics(values, spec.name),
            "packaging": _average_metrics(values, spec.name, packaging=True),
            "paired_vs_reranker": {
                "top5_complete_hit": _paired(
                    values, spec.name, ("top5_metrics", "complete_hit")
                ),
                "top5_required_gt_coverage": _paired(
                    values, spec.name, ("top5_metrics", "required_gt_coverage")
                ),
                "package_complete_hit": _paired(
                    values,
                    spec.name,
                    ("packaging", "metrics", "complete_hit"),
                ),
            },
        }
    return {
        "question_count": len(values),
        "max_challenger_rank": MAX_CHALLENGER_RANK,
        "configurations": configurations,
        "runtime": {"elapsed_seconds": elapsed_seconds},
    }


def _percent(value):
    return f"{value * 100:.2f}%"


def _delta(item):
    return f"+{item['improved']['count']} / -{item['degraded']['count']}"


def select_leader(summary):
    """只接受 complete@5 严格提升且完整命中无净退化的策略。"""
    configurations = summary["configurations"]
    baseline_complete = configurations["reranker_only"]["top5"]["complete_hit"]
    eligible = [POLICY_SPECS[0]]
    for spec in POLICY_SPECS[1:]:
        paired = configurations[spec.name]["paired_vs_reranker"]["top5_complete_hit"]
        if (
            configurations[spec.name]["top5"]["complete_hit"] > baseline_complete
            and paired["improved"]["count"] >= paired["degraded"]["count"]
        ):
            eligible.append(spec)
    return max(
        eligible,
        key=lambda spec: (
            configurations[spec.name]["top5"]["complete_hit"],
            configurations[spec.name]["top5"]["required_gt_coverage"],
            configurations[spec.name]["packaging"]["complete_hit"],
            configurations[spec.name]["top5"]["mrr"],
            -configurations[spec.name]["changed_ranking_count"],
        ),
    ).name


def render_report(summary, *, manifest):
    configurations = summary["configurations"]
    leader = select_leader(summary)
    lines = [
        "# Rerank Top-5 边界策略诊断",
        "",
        "> 本报告复用固定的原始单 query candidate pool 与完整 reranker 排名，不重新运行模型。",
        "> 分差只用于识别同一 query 内的 top-5 局部边界，不作为线上置信度或拒答阈值。",
        "",
        "## 策略",
        "",
        "- 窄权重融合：reranker 权重 0.975、0.95、0.90，RRF 使用剩余权重。",
        f"- 边界保护：固定 reranker 前 4，只检查第 6～{summary['max_challenger_rank']} 名；候选需比第 5 名有更高 RRF 名次且分差不超过阈值。",
        "- 边界存在多个候选时，优先选择 RRF 名次更高者；每题最多替换第 5 名一次。",
        "",
        "## 结果",
        "",
        "| 配置 | 改变题数 | complete@5 | Recall@5 | MRR@5 | top-5 + / - | 构包 complete | 构包 + / - |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for spec in POLICY_SPECS:
        item = configurations[spec.name]
        top5 = item["top5"]
        paired = item["paired_vs_reranker"]
        lines.append(
            f"| `{spec.name}` | {item['changed_ranking_count']} | "
            f"{_percent(top5['complete_hit'])} | "
            f"{_percent(top5['required_gt_coverage'])} | {top5['mrr']:.4f} | "
            f"{_delta(paired['top5_complete_hit'])} | "
            f"{_percent(item['packaging']['complete_hit'])} | "
            f"{_delta(paired['package_complete_hit'])} |"
        )
    leader_item = configurations[leader]
    widest_boundary = configurations["boundary_gap_005"]["paired_vs_reranker"][
        "top5_complete_hit"
    ]
    recovered_ids = "、".join(widest_boundary["improved"]["query_ids"]) or "无"
    degraded_ids = "、".join(widest_boundary["degraded"]["query_ids"]) or "无"
    lines.extend(
        [
            "",
            "## 结论",
            "",
            f"只有 complete@5 严格提高且无净退化时才替换基线；当前部署候选仍为 `{leader}`。",
            f"其 top-5 完整命中变化为 {_delta(leader_item['paired_vs_reranker']['top5_complete_hit'])}，"
            f"构包完整命中变化为 {_delta(leader_item['paired_vs_reranker']['package_complete_hit'])}。",
            "`global_fusion_090` 只改变 top-5 内部顺序，没有挽回任何 rerank 完整命中；构包为 +1 / -1。",
            f"最宽边界策略挽回：{recovered_ids}；新增退化：{degraded_ids}。",
            "因此不修改生产排序；下一步需要独立验证的法律 hard-negative 微调或 reranker 替换实验。",
            "该开发集用于诊断和选型，不能把同集最优阈值直接视为已验证的生产参数；需要独立集或真实 Query 增强结果复验。",
            "",
            "## 运行身份",
            "",
            f"- 源完整排名：`{manifest['inputs']['source_records']['path']}`。",
            f"- 源文件 SHA-256：`{manifest['inputs']['source_records']['sha256']}`。",
            f"- 生成时间：{manifest['generated_at']}。",
            f"- 离线重排与构包耗时：{summary['runtime']['elapsed_seconds']:.2f} 秒。",
            "- 生产排序已修改：否。",
            "",
            "## 复现命令",
            "",
            "```powershell",
            "conda run --no-capture-output -n minimind python -m rag.eval.rerank_boundary_evaluation",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def publish_results(output_dir, *, records, summary, manifest):
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
    report = render_report(summary, manifest=manifest)
    outputs = {
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
    final_manifest["outputs"] = outputs
    baseline._write_immutable_json(paths["manifest"], final_manifest)
    return final_manifest


def run_evaluation(*, source_records, article_index, tokenizer_path, output_dir):
    source_path = Path(source_records).resolve()
    repository = ArticleRepository.from_jsonl(article_index)
    tokenizer = baseline._load_tokenizer(tokenizer_path)
    packager = EvidencePackager(
        context_limit=baseline.CONTEXT_LIMIT,
        max_output_tokens=baseline.MAX_OUTPUT_TOKENS,
        count_prompt_tokens=AnswerPromptTokenCounter(tokenizer),
    )
    records, elapsed_seconds = evaluate_records(
        _load_source_records(source_path),
        repository=repository,
        packager=packager,
    )
    summary = summarize(records, elapsed_seconds=elapsed_seconds)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": PIPELINE,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "records": len(records),
        "inputs": {
            "source_records": baseline._file_identity(source_path),
            "article_index": baseline._file_identity(article_index),
            "tokenizer": baseline._tokenizer_identity(tokenizer, tokenizer_path),
        },
        "policy_specs": [asdict(spec) for spec in POLICY_SPECS],
        "max_challenger_rank": MAX_CHALLENGER_RANK,
        "production_changed": False,
    }
    final_manifest = publish_results(
        output_dir,
        records=records,
        summary=summary,
        manifest=manifest,
    )
    print(
        f"RERANK_BOUNDARY_OK records={len(records)} output={Path(output_dir).resolve()}",
        flush=True,
    )
    return final_manifest


def _parser():
    parser = argparse.ArgumentParser(description="离线诊断 rerank top-5 边界策略")
    parser.add_argument("--source-records", default=str(DEFAULT_SOURCE_RECORDS))
    parser.add_argument("--article-index", default=str(baseline.DEFAULT_ARTICLE_INDEX))
    parser.add_argument("--tokenizer-path", default=str(baseline.DEFAULT_TOKENIZER_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser


def main():
    args = _parser().parse_args()
    try:
        run_evaluation(
            source_records=args.source_records,
            article_index=args.article_index,
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        raise SystemExit(f"[失败] {error}") from error


if __name__ == "__main__":
    main()
