"""运行原始单 query、top-5 检索与真实构包基线。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from rag.answering import (
    ASSISTANT_SCHEMA,
    SYSTEM_PROMPT,
    AnswerPromptTokenCounter,
    EvidencePackager,
    MAX_EVIDENCE_ITEMS,
)
from rag.answering import evidence as evidence_module
from rag.answering import protocol as protocol_module
from rag.answering import token_count as token_count_module
from rag.eval.retrieval_chain_evaluation import (
    EvaluationCase,
    EvaluationMode,
    PackagingStatus,
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


RAG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_SET = RAG_DIR / "eval" / "eval_set.jsonl"
DEFAULT_ARTICLE_INDEX = RAG_DIR / "chunk" / "article_index.jsonl"
DEFAULT_ARTIFACT_DIR = RAG_DIR / "retrieval" / "artifacts"
DEFAULT_TOKENIZER_PATH = RAG_DIR.parent / "minimind" / "model"
DEFAULT_OUTPUT_DIR = (
    RAG_DIR
    / "eval"
    / "results"
    / "retrieval-baseline-original-top5-two-field-compact-v3"
)

PIPELINE = "legal_rag_retrieval_baseline_original_top5_two_field_compact_v3"
SCHEMA_VERSION = "1.1"
EXPECTED_CASE_COUNT = 140
CONTEXT_LIMIT = 768
MAX_OUTPUT_TOKENS = 150
_ARTIFACT_FILENAMES = (
    "law_dense.faiss",
    "law_dense_meta.json",
    "law_sparse.pkl",
)


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path):
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"输入文件不存在: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _text_sha256(value):
    if not isinstance(value, str):
        raise TypeError("哈希文本必须是字符串")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _git_identity(project_root):
    root = Path(project_root).resolve()
    try:
        commit = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "status", "--porcelain=v1"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("无法记录 Git 运行身份") from exc
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "status_sha256": _text_sha256(status),
        "changed_path_count": len(status.splitlines()),
    }


def _answering_runtime_identity():
    return {
        "protocol": "summary_citations_two_field_v1",
        "system_prompt_sha256": _text_sha256(SYSTEM_PROMPT),
        "assistant_schema_sha256": _text_sha256(ASSISTANT_SCHEMA),
        "evidence_serialization": {
            "format": "compact_json_utf8_v1",
            "fields": ["query", "evidence"],
            "evidence_fields": [
                "evidence_id",
                "law_name",
                "article_no",
                "excerpts",
            ],
            "ensure_ascii": False,
            "separators": [",", ":"],
        },
        "prompt_token_count": {
            "apply_chat_template": {
                "tokenize": False,
                "add_generation_prompt": True,
                "tools": None,
                "open_thinking": False,
            },
            "tokenizer_encode": {
                "add_special_tokens": True,
                "truncation": False,
            },
            "assistant_prefix": "empty_think",
        },
        "source_files": {
            "evidence": _file_identity(evidence_module.__file__),
            "protocol": _file_identity(protocol_module.__file__),
            "token_count": _file_identity(token_count_module.__file__),
        },
    }


def _load_tokenizer(path):
    resolved = Path(path).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Tokenizer 目录不存在: {resolved}")
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            resolved,
            use_fast=True,
            local_files_only=True,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"无法加载固定 Tokenizer: {resolved}") from exc


def _tokenizer_identity(tokenizer, path):
    resolved = Path(path).resolve()
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("Tokenizer 缺少 chat template")
    return {
        "path": str(resolved),
        "vocab_size": len(tokenizer),
        "chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest(),
        "files": {
            filename: _file_identity(resolved / filename)
            for filename in ("tokenizer.json", "tokenizer_config.json")
        },
    }


def load_baseline_cases(eval_set_path, repository):
    """加载当前开发集全部 legal_query + answer 样本。"""
    if not isinstance(repository, ArticleRepository):
        raise TypeError("repository 必须是 ArticleRepository")
    cases = []
    seen_ids = set()
    with Path(eval_set_path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"开发集第 {line_number} 行不是有效 JSON") from exc
            if item.get("query_type") != "legal_query" or item.get(
                "expected_action"
            ) != "answer":
                continue
            query_id = item.get("id")
            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError(f"开发集第 {line_number} 行缺少 id")
            if query_id in seen_ids:
                raise ValueError(f"开发集存在重复 id: {query_id}")
            seen_ids.add(query_id)
            query = item.get("query_original")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"{query_id} 缺少 query_original")
            references = item.get("gt_articles")
            if not isinstance(references, list) or not references:
                raise ValueError(f"{query_id} 缺少 gt_articles")
            required = []
            for reference in references:
                if not isinstance(reference, dict):
                    raise ValueError(f"{query_id} 的 GT 格式无效")
                article = repository.lookup(
                    reference.get("law_name"),
                    reference.get("article_no"),
                )
                if article is None:
                    raise ValueError(f"{query_id} 的 GT 无法解析: {reference!r}")
                required.append(article.chunk_id)
            cases.append(
                EvaluationCase(
                    query_id=query_id,
                    query_original=query,
                    required_chunk_ids=tuple(required),
                )
            )
    if len(cases) != EXPECTED_CASE_COUNT:
        raise ValueError(
            f"当前基线必须包含 {EXPECTED_CASE_COUNT} 条 answer，实际为 {len(cases)} 条"
        )
    return tuple(cases)


def evaluate_cases(cases, *, retriever, packager, progress_every=10):
    """按输入顺序执行原始单 query 基线。"""
    values = tuple(cases)
    if not values:
        raise ValueError("cases 不能为空")
    records = []
    started = time.perf_counter()
    for index, case in enumerate(values, start=1):
        plan = build_query_evaluation_plan(
            case.query_original,
            EvaluationMode.BASELINE_ORIGINAL,
        )
        records.append(
            evaluate_retrieval_case(
                case,
                plan,
                retriever=retriever,
                packager=packager,
            )
        )
        if progress_every and (index % progress_every == 0 or index == len(values)):
            elapsed = time.perf_counter() - started
            print(
                f"BASELINE_PROGRESS completed={index}/{len(values)} "
                f"elapsed_seconds={elapsed:.2f}",
                flush=True,
            )
    return tuple(records), time.perf_counter() - started


def build_failure_attribution(records):
    """按最早失败阶段生成互斥的逐题归因。"""
    groups = {
        "candidate_pool_missing": [],
        "rerank_top5_loss": [],
        "packaging_budget_loss": [],
        "packaged_complete": [],
    }
    for record in records:
        if not record.candidate_pool_metrics.complete_hit:
            group = "candidate_pool_missing"
        elif not record.reranked_top5_metrics.complete_hit:
            group = "rerank_top5_loss"
        elif not record.package_metrics.complete_hit:
            group = "packaging_budget_loss"
        else:
            group = "packaged_complete"
        groups[group].append(record.query_id)
    return {
        name: {"count": len(query_ids), "query_ids": query_ids}
        for name, query_ids in groups.items()
    }


def build_summary(records, *, elapsed_seconds):
    """在共享协议指标之上补充失败归因与运行耗时。"""
    values = tuple(records)
    summary = summarize_mode(values)
    summary["failure_attribution"] = build_failure_attribution(values)
    summary["runtime"] = {
        "elapsed_seconds": elapsed_seconds,
        "average_seconds_per_question": elapsed_seconds / len(values),
    }
    summary["packaging"]["not_attempted_count"] = sum(
        record.packaging_status is PackagingStatus.NOT_ATTEMPTED
        for record in values
    )
    return summary


def _percent(value):
    if value is None:
        return "不适用"
    return f"{value * 100:.2f}%"


def _query_ids_text(attribution, name):
    query_ids = attribution[name]["query_ids"]
    return "、".join(query_ids) if query_ids else "无"


def render_report(summary, *, manifest):
    """生成当前基线的可读 Markdown 报告。"""
    retrieval = summary["retrieval"]
    packaging = summary["packaging"]
    attribution = summary["failure_attribution"]
    pool = retrieval["candidate_pool"]
    top5 = retrieval["reranked_top5"]
    packaged = packaging["required_gt"]
    distribution = packaging["evidence_count_distribution"]
    question_count = summary["question_count"]
    config = manifest["retrieval_config"]
    lines = [
        "# 原始单 query Top-5 检索与构包基线",
        "",
        "> 当前正式基线：只使用 `query_original`，Query 增强状态为 `NOT_ATTEMPTED`。",
        "> 本报告只做离线分层诊断，不转换为线上 `complete | partial | none`，也不引入阈值拒答。",
        "",
        "## 评估范围",
        "",
        f"- 样本：当前开发集 {question_count} 条 `legal_query + answer`；"
        "30 条 `exact_lookup + answer` 由确定性查条路径处理，不进入本报告。",
        "- 检索：原始单 query，经 dense、BM25、等权 RRF 和原始 query reranker。",
        "- 构包：最多考虑 rerank top-5，按真实 tokenizer 选择最大完整有序前缀。",
        f"- 预算：上下文 {manifest['packaging']['context_limit']} tokens，"
        f"输出预留 {manifest['packaging']['max_output_tokens']} tokens。",
        f"- 生成时间：{manifest['generated_at']}。",
        "",
        "## 固定配置",
        "",
        f"- dense top-k：{config['dense_top_k']}；BM25 top-k：{config['sparse_top_k']}。",
        f"- RRF k：{config['rrf_k']}；candidate pool：{config['candidate_pool']}。",
        f"- rerank top-k：{config['top_k']}；batch size：{config['batch_size']}。",
        f"- embedding：`{manifest['models']['embedding']}`。",
        f"- reranker：`{manifest['models']['reranker']}`；设备：`{manifest['device']}`。",
        "",
        "## 检索结果",
        "",
        "| 阶段 | any hit | required GT Macro Recall | complete hit | MRR |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| candidate pool | {_percent(pool['any_hit'])} | {_percent(pool['required_gt_coverage'])} | {_percent(pool['complete_hit'])} | {pool['mrr']:.4f} |",
        f"| rerank top-5 | {_percent(top5['any_hit'])} | {_percent(top5['required_gt_coverage'])} | {_percent(top5['complete_hit'])} | {top5['mrr']:.4f} |",
        "",
        "## 构包结果",
        "",
        f"- 预算后 required GT Macro Recall：{_percent(packaged['required_gt_coverage'])}。",
        f"- 预算后 complete hit：{_percent(packaged['complete_hit'])}。",
        f"- 构包失败：{packaging['failed_count']} 条；无候选未尝试：{packaging['not_attempted_count']} 条。",
        "",
        "| 入包证据数 | 题数 | 占全部题目 |",
        "| ---: | ---: | ---: |",
    ]
    for count in range(1, 6):
        value = distribution[str(count)]
        lines.append(f"| {count} | {value} | {_percent(value / question_count)} |")
    lines.extend(
        [
            "",
            f"- 存在第五条候选的题目：{packaging['fifth_candidate_eligible_count']} 条。",
            f"- 第五条实际入包：{packaging['fifth_candidate_entered_count']} 条，"
            f"进入率 {_percent(packaging['fifth_candidate_entry_rate'])}。",
            f"- 第五条首次补齐全部 required GT：{packaging['fifth_candidate_completed_gt_count']} 条。",
            "",
            "## 分层归因",
            "",
            "归因按最早失败阶段互斥统计，检索问题、构包问题和回答模型问题不混合。",
            "",
            f"- candidate pool 缺失 GT：{attribution['candidate_pool_missing']['count']} 条。",
            f"- pool 已完整但 rerank top-5 丢失 GT：{attribution['rerank_top5_loss']['count']} 条。",
            f"- top-5 已完整但 token 构包后丢失 GT：{attribution['packaging_budget_loss']['count']} 条。",
            f"- 构包后完整覆盖 GT：{attribution['packaged_complete']['count']} 条。",
            "",
            "### Candidate Pool 缺失",
            "",
            _query_ids_text(attribution, "candidate_pool_missing"),
            "",
            "### Rerank Top-5 损失",
            "",
            _query_ids_text(attribution, "rerank_top5_loss"),
            "",
            "### Token 构包损失",
            "",
            _query_ids_text(attribution, "packaging_budget_loss"),
            "",
            "### 第五条新增完整覆盖",
            "",
            "、".join(
                record["query_id"]
                for record in manifest["fifth_completed_records"]
            )
            or "无",
            "",
            "## 运行耗时",
            "",
            f"- 总耗时：{summary['runtime']['elapsed_seconds']:.2f} 秒。",
            f"- 平均每题：{summary['runtime']['average_seconds_per_question']:.2f} 秒。",
            "",
            "## 复现命令",
            "",
            "```powershell",
            "conda run --no-capture-output -n minimind python -m rag.eval.retrieval_baseline",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _write_hash(path, digest):
    hash_path = path.with_suffix(path.suffix + ".sha256")
    if hash_path.exists():
        raise FileExistsError(f"输出已存在，不能覆盖: {hash_path}")
    hash_path.write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_immutable_text(path, text):
    resolved = Path(path).resolve()
    hash_path = resolved.with_suffix(resolved.suffix + ".sha256")
    if resolved.exists() or hash_path.exists():
        raise FileExistsError(f"输出已存在，不能覆盖: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(text, encoding="utf-8", newline="\n")
    digest = _sha256_file(resolved)
    _write_hash(resolved, digest)
    return digest


def _write_immutable_json(path, payload):
    return _write_immutable_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


def _write_immutable_jsonl(path, records):
    text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in records
    )
    return _write_immutable_text(path, text)


def publish_results(output_dir, *, records, summary, manifest):
    """不可变地发布逐题记录、汇总、报告和运行清单。"""
    directory = Path(output_dir).resolve()
    records_payload = [record.to_dict() for record in records]
    records_path = directory / "records.jsonl"
    summary_path = directory / "summary.json"
    report_path = directory / "report.md"
    manifest_path = directory / "manifest.json"
    for path in (records_path, summary_path, report_path, manifest_path):
        if path.exists() or path.with_suffix(path.suffix + ".sha256").exists():
            raise FileExistsError(f"输出已存在，不能覆盖: {path}")

    report = render_report(summary, manifest=manifest)
    records_sha256 = _write_immutable_jsonl(records_path, records_payload)
    summary_sha256 = _write_immutable_json(summary_path, summary)
    report_sha256 = _write_immutable_text(report_path, report)
    final_manifest = dict(manifest)
    final_manifest["outputs"] = {
        "records": {
            "path": records_path.name,
            "records": len(records_payload),
            "sha256": records_sha256,
        },
        "summary": {"path": summary_path.name, "sha256": summary_sha256},
        "report": {"path": report_path.name, "sha256": report_sha256},
    }
    _write_immutable_json(manifest_path, final_manifest)
    return final_manifest


def run_baseline(
    *,
    eval_set,
    article_index,
    artifact_dir,
    tokenizer_path,
    output_dir,
    device,
    limit=None,
):
    """加载当前运行时并执行一次可复现的基线。"""
    repository = ArticleRepository.from_jsonl(article_index)
    all_cases = load_baseline_cases(eval_set, repository)
    if limit is not None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit 必须是正整数")
        cases = all_cases[:limit]
    else:
        cases = all_cases

    tokenizer = _load_tokenizer(tokenizer_path)
    counter = AnswerPromptTokenCounter(tokenizer)
    packager = EvidencePackager(
        context_limit=CONTEXT_LIMIT,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        count_prompt_tokens=counter,
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
    records, elapsed_seconds = evaluate_cases(
        cases,
        retriever=retriever,
        packager=packager,
    )
    summary = build_summary(records, elapsed_seconds=elapsed_seconds)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    artifact_path = Path(artifact_dir).resolve()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "pipeline": PIPELINE,
        "generated_at": generated_at,
        "mode": EvaluationMode.BASELINE_ORIGINAL.value,
        "complete_dataset": len(cases) == len(all_cases),
        "records": len(records),
        "inputs": {
            "eval_set": _file_identity(eval_set),
            "article_index": _file_identity(article_index),
            "retrieval_artifacts": {
                filename: _file_identity(artifact_path / filename)
                for filename in _ARTIFACT_FILENAMES
            },
            "tokenizer": _tokenizer_identity(tokenizer, tokenizer_path),
            "answering_runtime": _answering_runtime_identity(),
        },
        "models": {
            "embedding": DEFAULT_EMBEDDING_MODEL,
            "reranker": DEFAULT_RERANKER_MODEL,
            "query_enhancer": None,
            "answer_model": None,
        },
        "device": actual_device,
        "retrieval_config": asdict(config),
        "packaging": {
            "context_limit": CONTEXT_LIMIT,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "selection": "max_complete_ordered_prefix",
            "maximum_evidence": MAX_EVIDENCE_ITEMS,
            "truncate_content": False,
            "skip_intermediate_candidate": False,
        },
        "fifth_completed_records": [
            {"query_id": record.query_id}
            for record in records
            if record.fifth_candidate_completed_gt
        ],
        "source_control": _git_identity(RAG_DIR.parent),
    }
    final_manifest = publish_results(
        output_dir,
        records=records,
        summary=summary,
        manifest=manifest,
    )
    print(
        f"RETRIEVAL_BASELINE_OK records={len(records)} output={Path(output_dir).resolve()}",
        flush=True,
    )
    return final_manifest


def _parser():
    parser = argparse.ArgumentParser(description="运行原始单 query top-5 检索与构包基线")
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    parser.add_argument("--article-index", default=str(DEFAULT_ARTICLE_INDEX))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--tokenizer-path", default=str(DEFAULT_TOKENIZER_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    return parser


def main():
    args = _parser().parse_args()
    try:
        run_baseline(
            eval_set=args.eval_set,
            article_index=args.article_index,
            artifact_dir=args.artifact_dir,
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
            device=args.device,
            limit=args.limit,
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise SystemExit(f"[失败] {error}") from error


if __name__ == "__main__":
    main()
