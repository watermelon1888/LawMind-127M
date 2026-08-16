"""使用物化 Query-SFT 模型输出运行真实检索、构包和成对报告。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from rag.answering import AnswerPromptTokenCounter, EvidencePackager
from rag.core import QueryEnhancementFailureReason
from rag.eval import retrieval_baseline as baseline
from rag.eval.retrieval_chain_evaluation import (
    EvaluationMode,
    build_query_evaluation_plan,
    evaluate_retrieval_case,
    summarize_mode,
)
from rag.knowledge import ArticleRepository
from rag.retrieval import SemanticRetrievalConfig
from rag.retrieval.loader import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    _resolve_device,
    load_semantic_retriever,
)
from rag.query import QUERY_ENHANCEMENT_SYSTEM_PROMPT


PIPELINE = "query_sft_real_retrieval_evaluation_v1"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "query-sft-model"
FORMAL_BASELINE_REPORT = Path(__file__).resolve().parent / "results" / "RAG评估总报告.md"
FORMAL_BASELINE = {
    "reranked_top5_complete_hit": 0.8071428571428572,
    "packaged_complete_hit": 0.7285714285714285,
}


def load_inference_release(path):
    """校验推理 manifest、固定 prompt/解码和逐题输出身份。"""

    manifest_path = Path(path).resolve()
    hash_path = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not manifest_path.is_file() or not hash_path.is_file():
        raise ValueError("推理 manifest 或相邻哈希不存在")
    fields = hash_path.read_text(encoding="utf-8").strip().split()
    if (
        len(fields) != 2
        or fields[1] != manifest_path.name
        or fields[0] != baseline._sha256_file(manifest_path)
    ):
        raise ValueError("推理 manifest 相邻哈希无效")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_prompt_sha = hashlib.sha256(
        QUERY_ENHANCEMENT_SYSTEM_PROMPT.encode("utf-8")
    ).hexdigest()
    records = manifest.get("outputs", {}).get("records")
    if (
        manifest.get("pipeline") != "query_sft_model_inference_v1"
        or manifest.get("complete") is not True
        or manifest.get("prompt", {}).get("sha256") != expected_prompt_sha
        or manifest.get("decoding")
        != {"strategy": "greedy", "do_sample": False, "max_new_tokens": 336}
        or not isinstance(records, dict)
        or not isinstance(records.get("path"), str)
        or not isinstance(records.get("sha256"), str)
    ):
        raise ValueError("推理 manifest 协议或固定身份无效")
    records_path = manifest_path.parent / records["path"]
    if (
        not records_path.is_file()
        or baseline._sha256_file(records_path) != records["sha256"]
    ):
        raise ValueError("推理逐题输出身份无效")
    return manifest, records_path


def load_model_outputs(path, *, expected_query_ids=()):
    """加载逐题原始模型输出，并校验与评估集的一一对应关系。"""

    values = {}
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"模型输出第 {line_number} 行不是有效 JSON") from error
            query_id = item.get("query_id")
            if not isinstance(query_id, str) or not query_id or query_id in values:
                raise ValueError(f"模型输出 query_id 无效或重复: {query_id!r}")
            if item.get("status") not in {"applied", "fallback"}:
                raise ValueError(f"{query_id} 的模型输出状态无效")
            raw_output = item.get("raw_output")
            if raw_output is not None and not isinstance(raw_output, str):
                raise ValueError(f"{query_id} 的 raw_output 无效")
            values[query_id] = item
    expected = set(expected_query_ids)
    if expected and set(values) != expected:
        raise ValueError("模型输出必须与评估集包含完全相同的 query_id")
    if not values:
        raise ValueError("模型输出不能为空")
    return values


def evaluate_query_model_cases(
    cases,
    model_outputs,
    *,
    retriever,
    packager,
    progress_every=10,
):
    """在同一冻结链路中成对运行原始 query 与完整增强。"""

    values = tuple(cases)
    baseline_records = []
    enhanced_records = []
    started = time.perf_counter()
    for index, case in enumerate(values, start=1):
        output = model_outputs[case.query_id]
        baseline_plan = build_query_evaluation_plan(
            case.query_original, EvaluationMode.BASELINE_ORIGINAL
        )
        if output.get("raw_output") is None:
            enhanced_plan = build_query_evaluation_plan(
                case.query_original,
                EvaluationMode.FULL_ENHANCEMENT,
                failure_reason=QueryEnhancementFailureReason.CALL_FAILED,
            )
        else:
            enhanced_plan = build_query_evaluation_plan(
                case.query_original,
                EvaluationMode.FULL_ENHANCEMENT,
                raw_enhancement=output["raw_output"],
            )
        baseline_records.append(
            evaluate_retrieval_case(
                case, baseline_plan, retriever=retriever, packager=packager
            )
        )
        enhanced_records.append(
            evaluate_retrieval_case(
                case, enhanced_plan, retriever=retriever, packager=packager
            )
        )
        if progress_every and (index % progress_every == 0 or index == len(values)):
            print(f"QUERY_EVAL_PROGRESS completed={index}/{len(values)}", flush=True)
    elapsed = time.perf_counter() - started
    summary = summarize_mode(
        enhanced_records,
        baseline_records=baseline_records,
    )
    summary["baseline"] = summarize_mode(baseline_records)
    summary["runtime"] = {
        "elapsed_seconds": elapsed,
        "average_seconds_per_question_pair": elapsed / len(values),
    }
    return tuple(baseline_records), tuple(enhanced_records), summary


def _percent(value):
    return f"{value * 100:.2f}%"


def _metric_line(name, current, baseline_value, paired):
    ci = paired["confidence_interval_95"]
    return (
        f"| {name} | {_percent(baseline_value)} | {_percent(current)} | "
        f"{paired['mean_delta']:+.4f} | "
        f"{paired['improved']}/{paired['tied']}/{paired['degraded']} | "
        f"[{ci[0]:+.4f}, {ci[1]:+.4f}] |"
    )


def render_report(summary, *, manifest):
    """生成以两项 complete hit 为主指标的 Query-SFT 报告。"""

    current_top5 = summary["retrieval"]["reranked_top5"]
    current_package = summary["packaging"]["required_gt"]
    paired_baseline_top5 = summary["baseline"]["retrieval"]["reranked_top5"]
    paired_baseline_package = summary["baseline"]["packaging"]["required_gt"]
    primary = summary["paired_vs_baseline"]["primary_metrics"]
    lines = [
        "# Query-SFT 真实检索与构包评估",
        "",
        "> 原始 query 与模型增强结果在相同检索、rerank 和 EvidencePackager 链路中成对比较。",
        "",
        "## 主指标",
        "",
        "| 指标 | 正式参考基线 | Query-SFT | 成对均值差 | 提升/持平/退化 | 95% CI |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        _metric_line(
            "rerank complete_hit@5",
            current_top5["complete_hit"],
            FORMAL_BASELINE["reranked_top5_complete_hit"],
            primary["reranked_top5_complete_hit"],
        ),
        _metric_line(
            "token 构包后 complete hit",
            current_package["complete_hit"],
            FORMAL_BASELINE["packaged_complete_hit"],
            primary["packaged_complete_hit"],
        ),
        "",
        "正式参考基线固定引用 `rag/eval/results/RAG评估总报告.md` 的 compact-v3 口径。"
        "成对均值差、题数和置信区间仍由本次运行的逐题原始 query 结果计算。",
        "",
        "## 同次成对诊断基线",
        "",
        f"- rerank complete_hit@5：{_percent(paired_baseline_top5['complete_hit'])}。",
        f"- token 构包后 complete hit：{_percent(paired_baseline_package['complete_hit'])}。",
        "",
        "## 协议与成本",
        "",
        f"- 三字段协议合法率：{_percent(summary['query_enhancement']['protocol_legal_rate'])}。",
        f"- 回退率：{_percent(summary['query_enhancement']['fallback_rate'])}。",
        f"- no-op 比例：{_percent(summary['query_enhancement']['noop_rate'])}。",
        f"- 平均实际检索腿：{summary['query_enhancement']['average_executed_leg_count']:.4f}。",
        "",
        "## 身份",
        "",
        f"- 模型输出：`{manifest['inputs']['model_outputs']['sha256']}`。",
        f"- 评估样本：{manifest['records']} 条。",
        f"- 生成时间：{manifest['generated_at']}。",
        "",
    ]
    return "\n".join(lines)


def publish_results(output_dir, *, baseline_records, enhanced_records, summary, manifest):
    """不可变发布成对逐题记录、汇总、报告和 manifest。"""

    directory = Path(output_dir).resolve()
    baseline_path = directory / "baseline-records.jsonl"
    enhanced_path = directory / "enhanced-records.jsonl"
    summary_path = directory / "summary.json"
    report_path = directory / "report.md"
    manifest_path = directory / "manifest.json"
    baseline_sha = baseline._write_immutable_jsonl(
        baseline_path, [record.to_dict() for record in baseline_records]
    )
    enhanced_sha = baseline._write_immutable_jsonl(
        enhanced_path, [record.to_dict() for record in enhanced_records]
    )
    summary_sha = baseline._write_immutable_json(summary_path, summary)
    report_sha = baseline._write_immutable_text(
        report_path, render_report(summary, manifest=manifest)
    )
    final = dict(manifest)
    final["outputs"] = {
        "baseline_records": {"path": baseline_path.name, "sha256": baseline_sha},
        "enhanced_records": {"path": enhanced_path.name, "sha256": enhanced_sha},
        "summary": {"path": summary_path.name, "sha256": summary_sha},
        "report": {"path": report_path.name, "sha256": report_sha},
    }
    baseline._write_immutable_json(manifest_path, final)
    return final


def run_evaluation(
    *,
    inference_manifest,
    eval_set,
    article_index,
    artifact_dir,
    tokenizer_path,
    output_dir,
    device,
    limit=None,
):
    """加载冻结检索链并运行 Query-SFT 真实成对评估。"""

    repository = ArticleRepository.from_jsonl(article_index)
    all_cases = baseline.load_baseline_cases(eval_set, repository)
    cases = all_cases if limit is None else all_cases[:limit]
    inference, model_outputs = load_inference_release(inference_manifest)
    outputs = load_model_outputs(
        model_outputs,
        expected_query_ids=(case.query_id for case in cases),
    )
    tokenizer = baseline._load_tokenizer(tokenizer_path)
    packager = EvidencePackager(
        context_limit=baseline.CONTEXT_LIMIT,
        max_output_tokens=baseline.MAX_OUTPUT_TOKENS,
        count_prompt_tokens=AnswerPromptTokenCounter(tokenizer),
    )
    config = SemanticRetrievalConfig()
    actual_device = _resolve_device(device)
    retriever = load_semantic_retriever(
        repository=repository,
        artifact_dir=artifact_dir,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        reranker_model=DEFAULT_RERANKER_MODEL,
        device=actual_device,
        config=config,
    )
    baseline_records, enhanced_records, summary = evaluate_query_model_cases(
        cases,
        outputs,
        retriever=retriever,
        packager=packager,
    )
    manifest = {
        "schema_version": "1.0",
        "pipeline": PIPELINE,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "complete_dataset": len(cases) == len(all_cases),
        "records": len(cases),
        "inputs": {
            "model_outputs": baseline._file_identity(model_outputs),
            "inference_manifest": baseline._file_identity(inference_manifest),
            "eval_set": baseline._file_identity(eval_set),
            "article_index": baseline._file_identity(article_index),
            "tokenizer": baseline._tokenizer_identity(tokenizer, tokenizer_path),
            "retrieval_artifacts": {
                filename: baseline._file_identity(Path(artifact_dir).resolve() / filename)
                for filename in baseline._ARTIFACT_FILENAMES
            },
            "formal_baseline_report": baseline._file_identity(FORMAL_BASELINE_REPORT),
        },
        "formal_baseline": FORMAL_BASELINE,
        "models": {
            "embedding": DEFAULT_EMBEDDING_MODEL,
            "reranker": DEFAULT_RERANKER_MODEL,
        },
        "retrieval_config": asdict(config),
        "packaging": {
            "context_limit": baseline.CONTEXT_LIMIT,
            "max_output_tokens": baseline.MAX_OUTPUT_TOKENS,
            "selection": "max_complete_ordered_prefix",
        },
        "device": actual_device,
        "query_model": inference["inputs"]["model_only_weights"],
        "complete": True,
    }
    return publish_results(
        output_dir,
        baseline_records=baseline_records,
        enhanced_records=enhanced_records,
        summary=summary,
        manifest=manifest,
    )


def _parser():
    parser = argparse.ArgumentParser(description="运行 Query-SFT 真实检索与构包评估")
    parser.add_argument("--inference-manifest", required=True)
    parser.add_argument("--eval-set", default=str(baseline.DEFAULT_EVAL_SET))
    parser.add_argument("--article-index", default=str(baseline.DEFAULT_ARTICLE_INDEX))
    parser.add_argument("--artifact-dir", default=str(baseline.DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--tokenizer-path", default=str(baseline.DEFAULT_TOKENIZER_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    return parser


def main():
    args = _parser().parse_args()
    run_evaluation(
        inference_manifest=args.inference_manifest,
        eval_set=args.eval_set,
        article_index=args.article_index,
        artifact_dir=args.artifact_dir,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        device=args.device,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
