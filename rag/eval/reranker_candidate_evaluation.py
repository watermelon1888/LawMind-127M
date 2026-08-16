"""在固定 candidate pool 上评估一个候选 Cross-Encoder reranker。"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from statistics import fmean
from types import SimpleNamespace

from rag.answering import AnswerPromptTokenCounter, EvidencePackager
from rag.eval import retrieval_baseline as baseline
from rag.eval.rerank_boundary_evaluation import _load_source_records
from rag.eval.rerank_fusion_evaluation import _configuration_result
from rag.knowledge import ArticleRepository
from rag.retrieval.loader import _resolve_device
from rag.retrieval.rerank import WindowReranker


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
    / "reranker-candidate-bce-base-original-v1"
)
DEFAULT_MODEL = "maidalun1020/bce-reranker-base_v1"
BASELINE_NAME = "BAAI/bge-reranker-base"
PIPELINE = "legal_rag_reranker_candidate_original_v1"
SCHEMA_VERSION = "1.0"
MAX_LENGTH = 512
WINDOW_OVERLAP_TOKENS = 64
MIN_NET_COMPLETE_GAIN = 3


def rank_candidate_pool(candidate_pool, scores):
    """按候选模型分数排序，并沿用生产链路的稳定平局规则。"""
    pool = tuple(candidate_pool)
    score_values = tuple(scores)
    if not pool or len(pool) != len(score_values):
        raise ValueError("candidate_pool 与 scores 必须等长且非空")
    chunk_ids = tuple(item["chunk_id"] for item in pool)
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("candidate_pool 不能包含重复 chunk_id")
    ranked = []
    for item, raw_score in zip(pool, score_values):
        if isinstance(raw_score, (bool, str, bytes)):
            raise ValueError("候选 reranker 分数类型无效")
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("候选 reranker 分数必须是有限数值")
        ranked.append((item["chunk_id"], int(item["rrf_rank"]), score))
    ranked.sort(key=lambda item: (-item[2], item[1], item[0]))
    return tuple(ranked)


def _score_record(record, *, repository, reranker, packager):
    articles = tuple(
        repository.get_by_chunk_id(item["chunk_id"])
        for item in record["candidate_pool"]
    )
    scores = reranker.score(record["query_original"], articles)
    ranked = rank_candidate_pool(record["candidate_pool"], scores)
    ranking_ids = tuple(item[0] for item in ranked)
    candidate = _configuration_result(
        SimpleNamespace(
            query_original=record["query_original"],
            required_chunk_ids=tuple(record["required_chunk_ids"]),
        ),
        ranking_ids,
        repository=repository,
        packager=packager,
    )
    return {
        "query_id": record["query_id"],
        "query_original": record["query_original"],
        "required_chunk_ids": record["required_chunk_ids"],
        "candidate_pool_chunk_ids": [
            item["chunk_id"] for item in record["candidate_pool"]
        ],
        "baseline": record["configurations"]["reranker_only"],
        "candidate_full_ranking": [
            {
                "chunk_id": chunk_id,
                "candidate_rerank_rank": rank,
                "candidate_rerank_score": score,
            }
            for rank, (chunk_id, _rrf_rank, score) in enumerate(ranked, start=1)
        ],
        "candidate": candidate,
    }


def evaluate_records(records, *, repository, reranker, packager, progress_every=10):
    """按源记录顺序执行候选 reranker 评分与真实 token 构包。"""
    values = tuple(records)
    if not values:
        raise ValueError("records 不能为空")
    started = time.perf_counter()
    output = []
    for index, record in enumerate(values, start=1):
        output.append(
            _score_record(
                record,
                repository=repository,
                reranker=reranker,
                packager=packager,
            )
        )
        if progress_every and (index % progress_every == 0 or index == len(values)):
            print(
                f"RERANKER_CANDIDATE_PROGRESS completed={index}/{len(values)} "
                f"elapsed_seconds={time.perf_counter() - started:.2f}",
                flush=True,
            )
    return tuple(output), time.perf_counter() - started


def _metrics(records, side, *, packaging=False):
    values = []
    for record in records:
        configuration = record[side]
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


def _paired(records, path):
    groups = {"improved": [], "tied": [], "degraded": []}
    for record in records:
        baseline_value = record["baseline"]
        candidate_value = record["candidate"]
        for key in path:
            baseline_value = baseline_value[key]
            candidate_value = candidate_value[key]
        if candidate_value > baseline_value:
            group = "improved"
        elif candidate_value < baseline_value:
            group = "degraded"
        else:
            group = "tied"
        groups[group].append(record["query_id"])
    return {
        name: {"count": len(query_ids), "query_ids": query_ids}
        for name, query_ids in groups.items()
    }


def evaluate_acceptance(summary):
    """按预先声明的完整命中、召回和构包门槛判断候选是否可晋级。"""
    paired = summary["paired_vs_baseline"]["top5_complete_hit"]
    net_gain = paired["improved"]["count"] - paired["degraded"]["count"]
    checks = {
        "net_complete_gain_at_least_3": net_gain >= MIN_NET_COMPLETE_GAIN,
        "recall_at5_not_lower": (
            summary["candidate"]["top5"]["required_gt_coverage"]
            >= summary["baseline"]["top5"]["required_gt_coverage"]
        ),
        "packaging_complete_not_lower": (
            summary["candidate"]["packaging"]["complete_hit"]
            >= summary["baseline"]["packaging"]["complete_hit"]
        ),
    }
    return {
        "passed": all(checks.values()),
        "net_complete_gain": net_gain,
        "checks": checks,
    }


def summarize(records, *, elapsed_seconds, cuda_memory=None):
    values = tuple(records)
    summary = {
        "question_count": len(values),
        "baseline": {
            "top5": _metrics(values, "baseline"),
            "packaging": _metrics(values, "baseline", packaging=True),
        },
        "candidate": {
            "top5": _metrics(values, "candidate"),
            "packaging": _metrics(values, "candidate", packaging=True),
        },
        "paired_vs_baseline": {
            "top5_complete_hit": _paired(
                values, ("top5_metrics", "complete_hit")
            ),
            "top5_required_gt_coverage": _paired(
                values, ("top5_metrics", "required_gt_coverage")
            ),
            "package_complete_hit": _paired(
                values, ("packaging", "metrics", "complete_hit")
            ),
        },
        "runtime": {
            "elapsed_seconds": elapsed_seconds,
            "average_seconds_per_question": elapsed_seconds / len(values),
            "cuda_memory": cuda_memory,
        },
    }
    summary["acceptance"] = evaluate_acceptance(summary)
    return summary


def _percent(value):
    return f"{value * 100:.2f}%"


def _delta(paired):
    return f"+{paired['improved']['count']} / -{paired['degraded']['count']}"


def _ids(paired, group):
    return "、".join(paired[group]["query_ids"]) or "无"


def render_report(summary, *, manifest):
    baseline_metrics = summary["baseline"]
    candidate_metrics = summary["candidate"]
    paired = summary["paired_vs_baseline"]
    acceptance = summary["acceptance"]
    memory = summary["runtime"]["cuda_memory"]
    lines = [
        "# BCE Reranker 固定候选池评估",
        "",
        "> 本实验只替换 reranker，复用原始单 query 的固定 candidate pool；不重新运行召回模型。",
        "> 候选未通过全部门槛前不修改生产排序。",
        "",
        "## 运行配置",
        "",
        f"- 基线：`{manifest['models']['baseline']}`。",
        f"- 候选：`{manifest['models']['candidate']}`。",
        f"- 设备：`{manifest['device']}`；精度：`{manifest['precision']}`；batch size：{manifest['batch_size']}。",
        f"- 输入长度：{manifest['max_length']}；窗口重叠：{manifest['window_overlap_tokens']} tokens。",
        f"- 样本：{summary['question_count']} 条。",
        "",
        "## 结果",
        "",
        "| 模型 | complete@5 | Recall@5 | any@5 | MRR@5 | 构包 complete | 构包 Recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| 当前 BGE | {_percent(baseline_metrics['top5']['complete_hit'])} | {_percent(baseline_metrics['top5']['required_gt_coverage'])} | {_percent(baseline_metrics['top5']['any_hit'])} | {baseline_metrics['top5']['mrr']:.4f} | {_percent(baseline_metrics['packaging']['complete_hit'])} | {_percent(baseline_metrics['packaging']['required_gt_coverage'])} |",
        f"| 候选 BCE | {_percent(candidate_metrics['top5']['complete_hit'])} | {_percent(candidate_metrics['top5']['required_gt_coverage'])} | {_percent(candidate_metrics['top5']['any_hit'])} | {candidate_metrics['top5']['mrr']:.4f} | {_percent(candidate_metrics['packaging']['complete_hit'])} | {_percent(candidate_metrics['packaging']['required_gt_coverage'])} |",
        "",
        f"- top-5 完整命中变化：{_delta(paired['top5_complete_hit'])}。",
        f"- 构包完整命中变化：{_delta(paired['package_complete_hit'])}。",
        f"- top-5 提升题：{_ids(paired['top5_complete_hit'], 'improved')}。",
        f"- top-5 退化题：{_ids(paired['top5_complete_hit'], 'degraded')}。",
        "",
        "## 准入判断",
        "",
        f"- 结论：{'通过' if acceptance['passed'] else '不通过'}。",
        f"- top-5 完整命中净增：{acceptance['net_complete_gain']} 题，要求至少 {MIN_NET_COMPLETE_GAIN} 题。",
        f"- Recall@5 不下降：{'是' if acceptance['checks']['recall_at5_not_lower'] else '否'}。",
        f"- 构包 complete 不下降：{'是' if acceptance['checks']['packaging_complete_not_lower'] else '否'}。",
        "- 本次结果只对原始单 query candidate pool 有效；Query 增强池物化后仍需重跑。",
        "",
        "## 资源",
        "",
        f"- 候选模型加载耗时：{manifest['runtime']['model_load_seconds']:.2f} 秒。",
        f"- 评分与构包耗时：{summary['runtime']['elapsed_seconds']:.2f} 秒。",
        f"- 平均每题：{summary['runtime']['average_seconds_per_question']:.3f} 秒。",
    ]
    if memory is not None:
        lines.extend(
            [
                f"- CUDA 当前分配：{memory['allocated_mib']:.1f} MiB。",
                f"- CUDA 峰值分配：{memory['peak_allocated_mib']:.1f} MiB。",
                f"- CUDA 峰值保留：{memory['peak_reserved_mib']:.1f} MiB。",
            ]
        )
    lines.extend(
        [
            "",
            "## 复现命令",
            "",
            "```powershell",
            "conda run --no-capture-output -n minimind python -m rag.eval.reranker_candidate_evaluation",
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
            "sha256": baseline._write_immutable_text(
                paths["report"], render_report(summary, manifest=manifest)
            ),
        },
    }
    final_manifest = dict(manifest)
    final_manifest["outputs"] = outputs
    baseline._write_immutable_json(paths["manifest"], final_manifest)
    return final_manifest


def _cuda_memory(torch, device):
    if device != "cuda":
        return None
    return {
        "allocated_mib": torch.cuda.memory_allocated() / (1024**2),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
    }


def _model_identity(cross_encoder, requested_model):
    config = cross_encoder.model.config
    return {
        "requested": requested_model,
        "resolved_name_or_path": str(config._name_or_path),
        "commit_hash": getattr(config, "_commit_hash", None),
        "architecture": type(cross_encoder.model).__name__,
        "parameter_count": sum(
            parameter.numel() for parameter in cross_encoder.model.parameters()
        ),
        "dtype": str(next(cross_encoder.model.parameters()).dtype),
    }


def run_evaluation(
    *,
    source_records,
    article_index,
    tokenizer_path,
    output_dir,
    model,
    device,
    batch_size,
    precision,
    limit=None,
):
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")
    if precision not in {"fp32", "fp16"}:
        raise ValueError("precision 必须是 fp32 或 fp16")
    actual_device = _resolve_device(device)
    if precision == "fp16" and actual_device != "cuda":
        raise ValueError("fp16 只允许在 CUDA 设备上使用")

    from sentence_transformers import CrossEncoder
    import torch

    load_started = time.perf_counter()
    cross_encoder = CrossEncoder(
        model,
        max_length=MAX_LENGTH,
        device=actual_device,
    )
    if precision == "fp16":
        cross_encoder.model.half()
    model_load_seconds = time.perf_counter() - load_started
    if actual_device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    repository = ArticleRepository.from_jsonl(article_index)
    source_values = _load_source_records(source_records)
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit 必须是正整数")
        source_values = source_values[:limit]
    tokenizer = baseline._load_tokenizer(tokenizer_path)
    packager = EvidencePackager(
        context_limit=baseline.CONTEXT_LIMIT,
        max_output_tokens=baseline.MAX_OUTPUT_TOKENS,
        count_prompt_tokens=AnswerPromptTokenCounter(tokenizer),
    )
    reranker = WindowReranker(
        model=cross_encoder,
        tokenizer=cross_encoder.tokenizer,
        max_length=MAX_LENGTH,
        overlap_tokens=WINDOW_OVERLAP_TOKENS,
        batch_size=batch_size,
    )
    records, elapsed_seconds = evaluate_records(
        source_values,
        repository=repository,
        reranker=reranker,
        packager=packager,
    )
    summary = summarize(
        records,
        elapsed_seconds=elapsed_seconds,
        cuda_memory=_cuda_memory(torch, actual_device),
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": PIPELINE,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "complete_dataset": limit is None,
        "records": len(records),
        "inputs": {
            "source_records": baseline._file_identity(source_records),
            "article_index": baseline._file_identity(article_index),
            "tokenizer": baseline._tokenizer_identity(tokenizer, tokenizer_path),
        },
        "models": {
            "baseline": BASELINE_NAME,
            "candidate": model,
            "candidate_identity": _model_identity(cross_encoder, model),
        },
        "device": actual_device,
        "precision": precision,
        "batch_size": batch_size,
        "max_length": MAX_LENGTH,
        "window_overlap_tokens": WINDOW_OVERLAP_TOKENS,
        "acceptance_policy": {
            "minimum_net_complete_gain": MIN_NET_COMPLETE_GAIN,
            "recall_at5_must_not_decrease": True,
            "packaging_complete_must_not_decrease": True,
        },
        "runtime": {"model_load_seconds": model_load_seconds},
        "production_changed": False,
    }
    result = publish_results(
        output_dir,
        records=records,
        summary=summary,
        manifest=manifest,
    )
    print(
        f"RERANKER_CANDIDATE_OK records={len(records)} "
        f"passed={summary['acceptance']['passed']} output={Path(output_dir).resolve()}",
        flush=True,
    )
    return result


def _parser():
    parser = argparse.ArgumentParser(description="评估固定 candidate pool 上的候选 reranker")
    parser.add_argument("--source-records", default=str(DEFAULT_SOURCE_RECORDS))
    parser.add_argument("--article-index", default=str(baseline.DEFAULT_ARTICLE_INDEX))
    parser.add_argument("--tokenizer-path", default=str(baseline.DEFAULT_TOKENIZER_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--limit", type=int)
    return parser


def main():
    args = _parser().parse_args()
    try:
        run_evaluation(
            source_records=args.source_records,
            article_index=args.article_index,
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
            model=args.model,
            device=args.device,
            batch_size=args.batch_size,
            precision=args.precision,
            limit=args.limit,
        )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        raise SystemExit(f"[失败] {error}") from error


if __name__ == "__main__":
    main()
