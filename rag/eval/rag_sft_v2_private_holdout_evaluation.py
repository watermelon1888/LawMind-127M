"""一次性运行 RAG-SFT v2 私有留出集的检索、构包与回答评估。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from minimind.trainer import evaluate_rag_sft_v2 as answer_eval
from minimind.trainer import train_full_sft as model_entry
from rag.answering import (
    AnswerPromptTokenCounter,
    EvidencePackager,
    EvidencePackagingError,
)
from rag.core.exact_lookup import ExactLookupStatus, resolve_exact_lookup
from rag.eval import private_holdout
from rag.eval.retrieval_chain_evaluation import (
    EvaluationCase,
    EvaluationMode,
    PackagingStatus,
    build_query_evaluation_plan,
    evaluate_retrieval_case,
    summarize_mode,
)
from rag.knowledge import ArticleRepository
from rag.query import QueryReason, QueryRoute, route_query
from rag.retrieval import SemanticRetrievalConfig
from rag.retrieval.loader import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    _resolve_device,
    load_semantic_retriever,
)


PIPELINE = "rag_sft_v2_private_holdout_evaluation_v1"
SCHEMA_VERSION = "1.0"
PRIVATE_ASSET_NAME = "project_private_holdout_v2"
EXPECTED_RECORDS = 100
EXPECTED_ANSWERABLE = 80
EXPECTED_UNANSWERABLE = 20
EXPECTED_SEMANTIC_ANSWERABLE = 60
EXPECTED_EXACT_ANSWERABLE = 20
CONTEXT_LIMIT = 768
MAX_NEW_TOKENS = 150


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: str | Path, *, records: int | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"输入文件不存在: {resolved}")
    value: dict[str, Any] = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }
    if records is not None:
        value["records"] = records
    return value


def _load_json(path: str | Path, label: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} 不是有效 UTF-8 JSON: {resolved}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} 必须是 JSON object")
    return value


def _load_jsonl(path: str | Path, label: str) -> list[dict[str, Any]]:
    records = []
    resolved = Path(path).resolve()
    try:
        with resolved.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    raise ValueError(f"{label} 第 {line_number} 行为空")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{label} 第 {line_number} 行不是 object")
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取 {label}: {resolved}") from error
    return records


def load_verified_holdout(
    holdout_path: str | Path,
    evaluation_exclusions_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """按正式训练隔离 manifest 验证私有 v2 身份并加载记录。"""

    exclusions = _load_json(evaluation_exclusions_path, "评估隔离 manifest")
    assets = exclusions.get("assets")
    if (
        exclusions.get("complete_for_formal_sft") is not True
        or not isinstance(assets, list)
        or exclusions.get("isolation_audit", {}).get("compatible") is not True
    ):
        raise ValueError("评估隔离 manifest 未闭合")
    matches = [
        item
        for item in assets
        if isinstance(item, dict) and item.get("name") == PRIVATE_ASSET_NAME
    ]
    if len(matches) != 1:
        raise ValueError("评估隔离 manifest 未唯一绑定私有留出 v2")
    expected = matches[0]
    actual = _identity(holdout_path)
    if (
        actual["sha256"] != expected.get("sha256")
        or actual["bytes"] != expected.get("bytes")
        or expected.get("records") != EXPECTED_RECORDS
        or exclusions.get("isolation_audit", {})
        .get("input_sha256", {})
        .get("authoring")
        != actual["sha256"]
    ):
        raise ValueError("私有留出 v2 与正式训练隔离身份不一致")

    records = _load_jsonl(holdout_path, "私有留出 v2")
    seen = set()
    for line_number, record in enumerate(records, 1):
        if set(record) != private_holdout.AUTHORING_FIELDS:
            raise ValueError(f"私有留出第 {line_number} 行 schema 无效")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise ValueError(f"私有留出第 {line_number} 行身份无效")
        seen.add(record_id)
        if record.get("review_status") != "approved":
            raise ValueError(f"{record_id} 尚未 approved")
        retrieval_gt = record.get("retrieval_gt")
        if not isinstance(retrieval_gt, dict) or set(retrieval_gt) != {"required_chunk_ids"}:
            raise ValueError(f"{record_id} retrieval_gt 无效")
        required = retrieval_gt["required_chunk_ids"]
        if not isinstance(required, list) or len(required) != len(set(required)):
            raise ValueError(f"{record_id} required_chunk_ids 无效")
        if record.get("answerable") is True and not required:
            raise ValueError(f"{record_id} 可答题缺少 required GT")
        if record.get("answerable") is False and required:
            raise ValueError(f"{record_id} 不可答题不能携带 required GT")

    answerable = sum(record.get("answerable") is True for record in records)
    unanswerable = sum(record.get("answerable") is False for record in records)
    semantic_answerable = sum(
        record.get("answerable") is True and record.get("query_type") != "exact_lookup"
        for record in records
    )
    exact_answerable = sum(
        record.get("answerable") is True and record.get("query_type") == "exact_lookup"
        for record in records
    )
    if (
        len(records) != EXPECTED_RECORDS
        or answerable != EXPECTED_ANSWERABLE
        or unanswerable != EXPECTED_UNANSWERABLE
        or semantic_answerable != EXPECTED_SEMANTIC_ANSWERABLE
        or exact_answerable != EXPECTED_EXACT_ANSWERABLE
    ):
        raise ValueError("私有留出计数不闭合")
    actual["records"] = len(records)
    return records, {
        "holdout": actual,
        "evaluation_exclusions": _identity(evaluation_exclusions_path),
        "asset_name": PRIVATE_ASSET_NAME,
    }


def evaluate_routing(
    records: Sequence[Mapping[str, Any]], repository: ArticleRepository
) -> tuple[dict[str, Any], dict[str, Any]]:
    """评估全部100题的当前确定性路由和不可答门控。"""

    observations: dict[str, Any] = {}
    correct = 0
    unanswerable_correct = 0
    route_counts: Counter[str] = Counter()
    for record in records:
        decision = route_query(record["query_original"])
        route_counts[decision.route.value] += 1
        answerable = record["answerable"] is True
        query_type = record["query_type"]
        reason = record["unanswerable_reason"]
        exact_status = None
        if decision.route is QueryRoute.EXACT_LOOKUP:
            exact_status = resolve_exact_lookup(decision, repository).status.value

        if answerable and query_type == "exact_lookup":
            expected_ok = (
                decision.route is QueryRoute.EXACT_LOOKUP
                and exact_status == ExactLookupStatus.FOUND.value
            )
        elif answerable:
            expected_ok = decision.route is QueryRoute.SEMANTIC_SEARCH
        elif reason == "irrelevant":
            expected_ok = (
                decision.route is QueryRoute.REFUSE
                and decision.reason is QueryReason.NON_LEGAL
            )
        else:
            expected_ok = (
                decision.route is QueryRoute.EXACT_LOOKUP
                and exact_status == ExactLookupStatus.CLARIFICATION_REQUIRED.value
            )
        correct += expected_ok
        if not answerable:
            unanswerable_correct += expected_ok
        observations[record["id"]] = {
            "route": decision.route.value,
            "reason": decision.reason.value if decision.reason else None,
            "exact_lookup_status": exact_status,
            "expected_ok": expected_ok,
        }
    return {
        "records": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "route_counts": dict(sorted(route_counts.items())),
        "unanswerable_records": EXPECTED_UNANSWERABLE,
        "unanswerable_correct": unanswerable_correct,
        "unanswerable_gate_accuracy": unanswerable_correct / EXPECTED_UNANSWERABLE,
    }, observations


def build_semantic_cases(
    records: Sequence[Mapping[str, Any]], repository: ArticleRepository
) -> tuple[EvaluationCase, ...]:
    """构造生产语义检索适用的60道可答题。"""

    cases = []
    for record in records:
        if record["answerable"] is not True or record["query_type"] == "exact_lookup":
            continue
        required = tuple(record["retrieval_gt"]["required_chunk_ids"])
        for chunk_id in required:
            repository.get_by_chunk_id(chunk_id)
        cases.append(
            EvaluationCase(
                query_id=record["id"],
                query_original=record["query_original"],
                required_chunk_ids=required,
            )
        )
    if len(cases) != EXPECTED_SEMANTIC_ANSWERABLE:
        raise ValueError("语义检索可答题不是60道")
    return tuple(cases)


def run_retrieval(
    cases: Sequence[EvaluationCase], *, retriever: Any, packager: EvidencePackager
) -> tuple[tuple[Any, ...], float]:
    """使用原始单 query、候选20、rerank top-5和768构包运行检索。"""

    outputs = []
    started = time.perf_counter()
    for index, case in enumerate(cases, 1):
        plan = build_query_evaluation_plan(
            case.query_original, EvaluationMode.BASELINE_ORIGINAL
        )
        outputs.append(
            evaluate_retrieval_case(
                case,
                plan,
                retriever=retriever,
                packager=packager,
            )
        )
        if index % 10 == 0 or index == len(cases):
            print(f"PRIVATE_HOLDOUT_RETRIEVAL_PROGRESS {index}/{len(cases)}", flush=True)
    return tuple(outputs), time.perf_counter() - started


def retrieval_summary(records: Sequence[Any], *, elapsed_seconds: float) -> dict[str, Any]:
    """只发布用户确认的检索与构包字段。"""

    summary = summarize_mode(records)
    pool = summary["retrieval"]["candidate_pool"]
    top5 = summary["retrieval"]["reranked_top5"]
    packaged = summary["packaging"]["required_gt"]
    return {
        "records": len(records),
        "candidate_complete_hit_at_20": pool["complete_hit"],
        "rerank_complete_hit_at_5": top5["complete_hit"],
        "mrr_at_5": top5["mrr"],
        "required_gt_macro_recall_at_5": top5["required_gt_coverage"],
        "packaged_complete_hit": packaged["complete_hit"],
        "packaged_required_gt_macro_recall": packaged["required_gt_coverage"],
        "packaging_failed": summary["packaging"]["failed_count"],
        "elapsed_seconds": elapsed_seconds,
    }


def _answer_case(
    *,
    record: Mapping[str, Any],
    visible_chunk_ids: Sequence[str],
    repository: ArticleRepository,
    source: str,
    packaged_complete: bool,
    prompt_tokens: int | None,
) -> dict[str, Any]:
    evidence = []
    for index, chunk_id in enumerate(visible_chunk_ids, 1):
        article = repository.get_by_chunk_id(chunk_id)
        evidence.append(
            {
                "evidence_id": f"E{index}",
                "chunk_id": chunk_id,
                "law_name": article.law_name,
                "article_no": article.article_no,
                "content": article.content,
            }
        )
    return {
        "query_id": record["id"],
        "query_type": (
            "exact_lookup" if record["query_type"] == "exact_lookup" else "legal_query"
        ),
        "evaluation_scope": "primary_768",
        "query": record["query_original"],
        "evidence": evidence,
        "visible_chunk_ids": list(visible_chunk_ids),
        "required_chunk_ids": list(record["retrieval_gt"]["required_chunk_ids"]),
        "hard_negative_chunk_ids": [],
        "retrieval_attribution": {
            "source": source,
            "packaged_complete": packaged_complete,
            "prompt_tokens": prompt_tokens,
        },
    }


def build_answer_cases(
    holdout_records: Sequence[Mapping[str, Any]],
    retrieval_records: Sequence[Any],
    repository: ArticleRepository,
    routing: Mapping[str, Any],
    packager: EvidencePackager,
) -> tuple[list[dict[str, Any]], list[str]]:
    """构造60道真实构包题和20道精确查条诊断题。"""

    by_id = {record["id"]: record for record in holdout_records}
    cases = []
    skipped = []
    for retrieved in retrieval_records:
        source = by_id[retrieved.query_id]
        if retrieved.packaging_status is not PackagingStatus.APPLIED:
            skipped.append(retrieved.query_id)
            continue
        cases.append(
            _answer_case(
                record=source,
                visible_chunk_ids=retrieved.packaged_chunk_ids,
                repository=repository,
                source="production_semantic_retrieval",
                packaged_complete=retrieved.package_metrics.complete_hit,
                prompt_tokens=retrieved.prompt_tokens,
            )
        )
    for record in holdout_records:
        if record["answerable"] is not True or record["query_type"] != "exact_lookup":
            continue
        route = routing[record["id"]]
        if (
            route["route"] != QueryRoute.EXACT_LOOKUP.value
            or route["exact_lookup_status"] != ExactLookupStatus.FOUND.value
        ):
            skipped.append(record["id"])
            continue
        decision = route_query(record["query_original"])
        resolution = resolve_exact_lookup(decision, repository)
        try:
            package, prompt_tokens = packager.build(
                record["query_original"], resolution.articles
            )
        except EvidencePackagingError:
            skipped.append(record["id"])
            continue
        visible = [article.chunk_id for article in resolution.articles][
            : len(package.evidence)
        ]
        required = set(record["retrieval_gt"]["required_chunk_ids"])
        cases.append(
            _answer_case(
                record=record,
                visible_chunk_ids=visible,
                repository=repository,
                source="deterministic_exact_lookup_model_diagnostic",
                packaged_complete=set(visible) == required,
                prompt_tokens=prompt_tokens,
            )
        )
    return cases, sorted(skipped)


def _model_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)
    if not total:
        return {
            "records": 0,
            "protocol_valid_rate": 0.0,
            "citation_exact_set_accuracy": 0.0,
            "required_citation_recall": 0.0,
        }
    return {
        "records": total,
        "protocol_valid_rate": sum(record["protocol_valid"] for record in records)
        / total,
        "citation_exact_set_accuracy": sum(
            record["citation_metrics"]["exact_set"] for record in records
        )
        / total,
        "required_citation_recall": sum(
            record["citation_metrics"]["required_recall"] for record in records
        )
        / total,
    }


def summarize_answer_results(
    results: Sequence[Mapping[str, Any]], *, eligible_answerable: int
) -> dict[str, Any]:
    """汇总三项模型指标，并保留构包完整性分层。"""

    semantic = [
        record
        for record in results
        if record["retrieval_attribution"]["source"] == "production_semantic_retrieval"
    ]
    exact_lookup = [
        record
        for record in results
        if record["retrieval_attribution"]["source"]
        == "deterministic_exact_lookup_model_diagnostic"
    ]
    return {
        "eligible_answerable_records": eligible_answerable,
        "model_invoked_records": len(results),
        "model_input_coverage": len(results) / eligible_answerable,
        "overall": _model_metrics(results),
        "semantic_retrieval": _model_metrics(semantic),
        "packaged_complete": _model_metrics(
            [
                record
                for record in semantic
                if record["retrieval_attribution"]["packaged_complete"] is True
            ]
        ),
        "packaged_incomplete": _model_metrics(
            [
                record
                for record in semantic
                if record["retrieval_attribution"]["packaged_complete"] is False
            ]
        ),
        "exact_lookup_diagnostic": _model_metrics(exact_lookup),
    }


def redact_answer_results(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """移除问题、证据、GT、模型原文和解析文本，只保留审计指标。"""

    redacted = []
    for record in results:
        redacted.append(
            {
                "query_id": record["query_id"],
                "query_type": record["query_type"],
                "source": record["retrieval_attribution"]["source"],
                "packaged_complete": record["retrieval_attribution"][
                    "packaged_complete"
                ],
                "protocol_valid": record["protocol_valid"],
                "citation_exact_set": record["citation_metrics"]["exact_set"],
                "required_citation_recall": record["citation_metrics"][
                    "required_recall"
                ],
                "prompt_tokens": record["generation_metrics"]["prompt_tokens"],
                "generated_tokens": record["generation_metrics"]["generated_tokens"],
                "hit_max_new_tokens": record["generation_metrics"][
                    "hit_max_new_tokens"
                ],
                "protocol_error": record["error"],
            }
        )
    return redacted


def _percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def render_report(report: Mapping[str, Any]) -> str:
    """生成不包含任何私有题面、证据或答案的 Markdown 报告。"""

    retrieval = report["summary"]["retrieval_and_packaging"]
    model = report["summary"]["rag_sft_model"]
    routing = report["summary"]["routing_and_unanswerable_gate"]
    overall = model["overall"]
    lines = [
        "# RAG-SFT v2 私有留出集一次性评估报告",
        "",
        "> 本报告不包含私有题面、证据原文、GT、模型原始输出或解析答案。",
        "> 本次运行已消费私有留出集，不得据此继续选择 checkpoint、调参或逐题修复。",
        "",
        "## 运行身份",
        "",
        f"- 模型权重 SHA-256：`{report['inputs']['weights']['sha256']}`。",
        f"- 私有留出 SHA-256：`{report['inputs']['holdout']['sha256']}`。",
        "- 生产配置：原始单 query、candidate pool 20、rerank top-5、context 768、greedy、max_new_tokens 150。",
        "",
        "## Retrieval 与 Answering",
        "",
        "以下指标只以60道需要语义检索的可答题为分母；精确查条和不可答题不混入语义检索指标。",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
        f"| candidate complete_hit@20 | {_percent(retrieval['candidate_complete_hit_at_20'])} |",
        f"| rerank complete_hit@5 | {_percent(retrieval['rerank_complete_hit_at_5'])} |",
        f"| MRR@5 | {retrieval['mrr_at_5']:.4f} |",
        f"| required GT Macro Recall@5 | {_percent(retrieval['required_gt_macro_recall_at_5'])} |",
        f"| packaged complete_hit | {_percent(retrieval['packaged_complete_hit'])} |",
        f"| packaged required GT Macro Recall | {_percent(retrieval['packaged_required_gt_macro_recall'])} |",
        "",
        "## RAG-SFT 模型",
        "",
        f"- 可答题：{model['eligible_answerable_records']}；实际形成模型输入：{model['model_invoked_records']}；覆盖率：{_percent(model['model_input_coverage'])}。",
        "- 20道可答精确查条题属于模型诊断口径；当前生产系统的精确查条路径本身不调用回答模型。",
        "",
        "| 范围 | 题数 | protocol_valid_rate | citation exact-set accuracy | required citation recall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, key in (
        ("全部模型输入", "overall"),
        ("语义检索输入", "semantic_retrieval"),
        ("构包完整", "packaged_complete"),
        ("构包不完整", "packaged_incomplete"),
        ("精确查条模型诊断", "exact_lookup_diagnostic"),
    ):
        value = model[key]
        lines.append(
            f"| {label} | {value['records']} | {_percent(value['protocol_valid_rate'])} | "
            f"{_percent(value['citation_exact_set_accuracy'])} | "
            f"{_percent(value['required_citation_recall'])} |"
        )
    lines.extend(
        [
            "",
            "## 路由与不可答门控",
            "",
            f"- 全100题路由/查条符合预期：{routing['correct']}/{routing['records']}（{_percent(routing['accuracy'])}）。",
            f"- 20道不可答题正确阻止回答模型：{routing['unanswerable_correct']}/{routing['unanswerable_records']}（{_percent(routing['unanswerable_gate_accuracy'])}）。",
            "",
            "## 使用约束",
            "",
            "- `private_holdout_used=true`。",
            "- 不基于本报告调整 epoch、父权重、学习率、数据配比或逐题规则。",
            "- 留出集没有冻结 HN 身份，因此不报告 hard-negative citation rate、Clean exact-set 或 HN exact-set。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_immutable(path: Path, text: str) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"输出已存在，不能覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    digest = _sha256_file(path)
    sidecar.write_text(f"{digest}  {path.name}\n", encoding="utf-8", newline="\n")
    return digest


def publish_report(output_dir: str | Path, report: dict[str, Any]) -> dict[str, Any]:
    """不可覆盖地发布脱敏 JSON、Markdown 与 manifest。"""

    root = Path(output_dir).resolve()
    report_json = root / "report.json"
    report_md = root / "report.md"
    manifest_path = root / "manifest.json"
    for path in (report_json, report_md, manifest_path):
        if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
            raise FileExistsError(f"输出已存在，不能覆盖: {path}")
    report_sha = _write_immutable(
        report_json,
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    markdown_sha = _write_immutable(report_md, render_report(report))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": PIPELINE,
        "private_holdout_used": True,
        "raw_private_content_included": False,
        "outputs": {
            "report_json": {"path": report_json.name, "sha256": report_sha},
            "report_md": {"path": report_md.name, "sha256": markdown_sha},
        },
        "complete": True,
    }
    _write_immutable(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    return manifest


def run_evaluation(
    *,
    holdout_path: str | Path,
    evaluation_exclusions_path: str | Path,
    article_index_path: str | Path,
    artifact_dir: str | Path,
    tokenizer_path: str | Path,
    weights_path: str | Path,
    weights_sha256: str,
    output_dir: str | Path,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    """执行一次完整私有留出评估并发布脱敏结果。"""

    holdout, private_identity = load_verified_holdout(
        holdout_path, evaluation_exclusions_path
    )
    repository = ArticleRepository.from_jsonl(article_index_path)
    for record in holdout:
        for chunk_id in record["retrieval_gt"]["required_chunk_ids"]:
            repository.get_by_chunk_id(chunk_id)
    routing_summary, routing_observations = evaluate_routing(holdout, repository)

    tokenizer = model_entry._load_tokenizer(tokenizer_path)
    tokenizer_identity = answer_eval._tokenizer_identity(tokenizer, tokenizer_path)
    packager = EvidencePackager(
        context_limit=CONTEXT_LIMIT,
        max_output_tokens=MAX_NEW_TOKENS,
        count_prompt_tokens=AnswerPromptTokenCounter(tokenizer),
    )
    config = SemanticRetrievalConfig()
    actual_device = _resolve_device(device_name)
    retriever = load_semantic_retriever(
        repository=repository,
        artifact_dir=artifact_dir,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        reranker_model=DEFAULT_RERANKER_MODEL,
        device=actual_device,
        config=config,
    )
    semantic_cases = build_semantic_cases(holdout, repository)
    retrieval_records, retrieval_seconds = run_retrieval(
        semantic_cases, retriever=retriever, packager=packager
    )
    retrieval_metrics = retrieval_summary(
        retrieval_records, elapsed_seconds=retrieval_seconds
    )
    answer_cases, skipped_answerable = build_answer_cases(
        holdout,
        retrieval_records,
        repository,
        routing_observations,
        packager,
    )

    del retriever
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 CUDA 但当前无可用 GPU")
    model = model_entry.MiniMindForCausalLM(model_entry._model_config()).to(device)
    actual_weights_sha = model_entry._load_parent_weights(
        weights_path, weights_sha256, model
    )
    started = time.perf_counter()
    answer_results = answer_eval.evaluate_records(
        answer_cases,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )
    answer_seconds = time.perf_counter() - started
    answer_summary = summarize_answer_results(
        answer_results, eligible_answerable=EXPECTED_ANSWERABLE
    )
    answer_summary["elapsed_seconds"] = answer_seconds
    answer_summary["skipped_answerable_count"] = len(skipped_answerable)
    answer_summary["skipped_answerable_ids"] = skipped_answerable

    artifact_root = Path(artifact_dir).resolve()
    report = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": PIPELINE,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "private_holdout_used": True,
        "selection_or_tuning_allowed": False,
        "inputs": {
            **private_identity,
            "article_index": _identity(article_index_path),
            "retrieval_artifacts": {
                filename: _identity(artifact_root / filename)
                for filename in (
                    "law_dense.faiss",
                    "law_dense_meta.json",
                    "law_sparse.pkl",
                )
            },
            "tokenizer": tokenizer_identity,
            "weights": {
                **_identity(weights_path),
                "sha256": actual_weights_sha,
            },
        },
        "configuration": {
            "retrieval": asdict(config),
            "query": "original_single_query",
            "context_limit": CONTEXT_LIMIT,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "retry": False,
            "repair_json": False,
            "inference_rope_scaling": False,
        },
        "metric_scope": {
            "retrieval_and_packaging": "60 answerable semantic-search records",
            "rag_sft_model": "all answerable records with a valid model input",
            "unanswerable": "routing and gate only",
            "hard_negative_labels_used": False,
        },
        "summary": {
            "retrieval_and_packaging": retrieval_metrics,
            "rag_sft_model": answer_summary,
            "routing_and_unanswerable_gate": routing_summary,
        },
        "records": {
            "retrieval": [
                {
                    "query_id": record.query_id,
                    "candidate_complete_hit_at_20": record.candidate_pool_metrics.complete_hit,
                    "rerank_complete_hit_at_5": record.reranked_top5_metrics.complete_hit,
                    "mrr_at_5": record.reranked_top5_metrics.reciprocal_rank,
                    "required_gt_recall_at_5": record.reranked_top5_metrics.required_gt_coverage,
                    "packaged_complete_hit": record.package_metrics.complete_hit,
                    "packaged_required_gt_recall": record.package_metrics.required_gt_coverage,
                    "packaging_status": record.packaging_status.value,
                }
                for record in retrieval_records
            ],
            "rag_sft_model": redact_answer_results(answer_results),
            "routing": [
                {"query_id": query_id, **value}
                for query_id, value in sorted(routing_observations.items())
            ],
        },
        "privacy": {
            "raw_questions_included": False,
            "evidence_text_included": False,
            "required_chunk_ids_included": False,
            "raw_model_outputs_included": False,
            "parsed_model_answers_included": False,
        },
        "complete": True,
    }
    publish_report(output_dir, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="一次性运行 RAG-SFT v2 私有留出评估")
    parser.add_argument("--holdout", required=True)
    parser.add_argument("--evaluation-exclusions", required=True)
    parser.add_argument("--article-index", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_evaluation(
        holdout_path=args.holdout,
        evaluation_exclusions_path=args.evaluation_exclusions,
        article_index_path=args.article_index,
        artifact_dir=args.artifact_dir,
        tokenizer_path=args.tokenizer_path,
        weights_path=args.weights,
        weights_sha256=args.weights_sha256,
        output_dir=args.output_dir,
        device_name=args.device,
    )
    model = report["summary"]["rag_sft_model"]["overall"]
    print(
        "PRIVATE_HOLDOUT_EVALUATION_OK "
        f"records={report['summary']['routing_and_unanswerable_gate']['records']} "
        f"model_records={model['records']} "
        f"protocol_valid_rate={model['protocol_valid_rate']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
