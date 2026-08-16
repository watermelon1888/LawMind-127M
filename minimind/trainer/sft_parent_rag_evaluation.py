"""执行父权重比较的冻结 RAG 成对证据推理评估。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from rag.answering import AnswerPromptTokenCounter, EvidencePackage, build_answer_prompt
from rag.answering.protocol import AnswerProtocolError, parse_and_validate_answer
from rag.core import Evidence
from rag.eval import parent_pair_evaluation as pair_assets

from . import sft_parent_rag_metrics as metrics
from . import train_full_sft as training


PIPELINE = "legal_rag_parent_model_evaluation_v1"
AUDIT_PIPELINE = "legal_rag_parent_prompt_length_audit_v1"
MAX_SEQ_LEN = 768
MAX_NEW_TOKENS = 160


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_immutable_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    resolved = Path(path).resolve()
    hash_path = resolved.with_suffix(".sha256")
    if resolved.exists() or hash_path.exists():
        raise FileExistsError(f"输出已存在，不能覆盖: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    hash_path.write_text(
        f"{_sha256_file(resolved)}  {resolved.name}\n",
        encoding="utf-8",
        newline="\n",
    )


def _package(record: Mapping[str, Any]) -> EvidencePackage:
    return EvidencePackage(
        query=record["query"],
        evidence=tuple(
            Evidence(
                law_name=item["law_name"],
                article_no=item["article_no"],
                content=item["content"],
            )
            for item in record["evidence"]
        ),
    )


def _load_records(
    manifest_path: str | Path,
    records_path: str | Path,
    article_index: str | Path,
    tokenizer: Any,
    tokenizer_path: str | Path,
) -> tuple[dict[str, Any], str, list[dict[str, Any]], str]:
    manifest = pair_assets.verify_pair_evaluation(
        manifest_path,
        records_path,
        article_index,
        tokenizer=tokenizer,
        tokenizer_path=tokenizer_path,
    )
    manifest_path = Path(manifest_path).resolve()
    records_path = Path(records_path).resolve()
    return (
        manifest,
        _sha256_file(manifest_path),
        pair_assets._load_verified_jsonl(records_path, "RAG 成对评估记录")[0],
        _sha256_file(records_path),
    )


def audit_prompt_lengths(
    records: Sequence[Mapping[str, Any]], tokenizer: Any
) -> dict[str, Any]:
    """按真实生成模板审计每个证据包是否可在固定 768 长度内生成。"""

    counter = AnswerPromptTokenCounter(tokenizer)
    lengths = []
    overflow = []
    for record in records:
        prompt_tokens = counter(_package(record))
        lengths.append(prompt_tokens)
        if prompt_tokens + MAX_NEW_TOKENS > MAX_SEQ_LEN:
            overflow.append(
                {
                    "query_id": record["query_id"],
                    "variant": record["variant"],
                    "prompt_tokens": prompt_tokens,
                    "available_output_tokens": MAX_SEQ_LEN - prompt_tokens,
                }
            )
    ordered = sorted(lengths)
    return {
        "fixed_max_seq_len": MAX_SEQ_LEN,
        "max_new_tokens": MAX_NEW_TOKENS,
        "records": len(records),
        "prompt_tokens": {
            "min": ordered[0],
            "p50": ordered[(len(ordered) - 1) // 2],
            "p95": ordered[(len(ordered) * 95 + 99) // 100 - 1],
            "max": ordered[-1],
        },
        "overflow": overflow,
        "all_fit": not overflow,
    }


def _render_prompt(package: EvidencePackage, tokenizer: Any) -> str:
    prompt = tokenizer.apply_chat_template(
        build_answer_prompt(package),
        tokenize=False,
        add_generation_prompt=True,
        tools=None,
        open_thinking=False,
    )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("chat template 没有返回非空 prompt")
    return prompt


def _model_metric_record(record: Mapping[str, Any], raw_text: str) -> dict[str, Any]:
    package = _package(record)
    visible = record["visible_chunk_ids"]
    try:
        answer = parse_and_validate_answer(package, raw_text)
        citations = [visible[int(item[1:]) - 1] for item in answer.citations]
        protocol_valid = True
        refuse = answer.refuse
        summary_nonempty = bool(answer.summary)
    except (AnswerProtocolError, ValueError, IndexError):
        citations = []
        protocol_valid = False
        refuse = False
        summary_nonempty = False
    return {
        "query_id": record["query_id"],
        "variant": record["variant"],
        "visible_chunk_ids": record["visible_chunk_ids"],
        "required_chunk_ids": record["required_chunk_ids"],
        "hard_negative_chunk_ids": record["hard_negative_chunk_ids"],
        "protocol_valid": protocol_valid,
        "refuse": refuse,
        "summary_nonempty": summary_nonempty,
        "citation_chunk_ids": citations,
    }


def evaluate_records(
    records: Sequence[Mapping[str, Any]], *, model: Any, tokenizer: Any, device: torch.device
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """生成严格 JSON 并投影为不含题面和法条原文的指标记录。"""

    audit = audit_prompt_lengths(records, tokenizer)
    if not audit["all_fit"]:
        raise ValueError("存在无法为 160 个生成 token 预留空间的 RAG 证据包")
    model.eval()
    result_records = []
    with torch.inference_mode():
        for record in records:
            prompt = _render_prompt(_package(record), tokenizer)
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            input_ids = encoded["input_ids"].to(device)
            generated = model.generate(
                inputs=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=1.0,
                top_k=0,
                use_cache=True,
                do_sample=False,
            )
            raw_text = tokenizer.decode(
                generated[0, input_ids.shape[1] :], skip_special_tokens=True
            )
            result_records.append(_model_metric_record(record, raw_text))
    return metrics.aggregate_rag_pair_metrics(result_records), result_records


def run_evaluation(
    *,
    pair_manifest: str | Path,
    pair_records: str | Path,
    article_index: str | Path,
    tokenizer_path: str | Path,
    weights: str | Path,
    weights_sha256: str,
    output: str | Path,
    device_name: str,
) -> dict[str, Any]:
    """加载一个 model-only 权重并发布不可变的 RAG 成对评估报告。"""

    tokenizer = training._load_tokenizer(tokenizer_path)
    manifest, manifest_digest, records, record_digest = _load_records(
        pair_manifest, pair_records, article_index, tokenizer, tokenizer_path
    )
    audit = audit_prompt_lengths(records, tokenizer)
    if not audit["all_fit"]:
        raise ValueError("RAG 成对评估长度审计失败，禁止生成")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 CUDA 设备不可用")
    model = training.MiniMindForCausalLM(training._model_config()).to(device)
    actual_weights_sha = training._load_parent_weights(weights, weights_sha256, model)
    aggregate, results = evaluate_records(
        records, model=model, tokenizer=tokenizer, device=device
    )
    report = {
        "schema_version": "1.0",
        "pipeline": PIPELINE,
        "inputs": {
            "pair_manifest_sha256": manifest_digest,
            "pair_records_sha256": record_digest,
            "weights_sha256": actual_weights_sha,
            "tokenizer": training._tokenizer_identity(tokenizer, tokenizer_path),
        },
        "generation": {
            "fixed_max_seq_len": MAX_SEQ_LEN,
            "max_new_tokens": MAX_NEW_TOKENS,
            "sampling": "greedy",
        },
        "length_audit": audit,
        "metrics": aggregate,
        "records": results,
        "complete": True,
    }
    if manifest["records"]["eligible_model_cases"] != len(
        {item["query_id"] for item in results}
    ):
        raise ValueError("评估结果题目数量与冻结 manifest 不一致")
    _write_immutable_json(output, report)
    return report


def run_length_audit(
    *,
    pair_manifest: str | Path,
    pair_records: str | Path,
    article_index: str | Path,
    tokenizer_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """不加载模型权重，发布冻结成对资产的真实模板长度审计。"""

    tokenizer = training._load_tokenizer(tokenizer_path)
    manifest, manifest_digest, records, record_digest = _load_records(
        pair_manifest, pair_records, article_index, tokenizer, tokenizer_path
    )
    audit = audit_prompt_lengths(records, tokenizer)
    report = {
        "schema_version": "1.0",
        "pipeline": AUDIT_PIPELINE,
        "inputs": {
            "pair_manifest_sha256": manifest_digest,
            "pair_records_sha256": record_digest,
            "tokenizer": training._tokenizer_identity(tokenizer, tokenizer_path),
        },
        "length_audit": audit,
        "readiness": {
            "prompt_length_audited": True,
            "all_records_fit_768_with_160_generation_tokens": audit["all_fit"],
            "evaluation_ready": audit["all_fit"],
        },
        "complete": True,
    }
    if manifest["pair_records"]["records"] != audit["records"]:
        raise ValueError("长度审计记录数与冻结 manifest 不一致")
    _write_immutable_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="执行父权重比较的 RAG 成对评估")
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit", help="仅执行真实模板长度审计")
    evaluate_parser = commands.add_parser("evaluate", help="执行模型生成与成对指标评估")
    for command_parser in (audit_parser, evaluate_parser):
        command_parser.add_argument("--pair-manifest", required=True)
        command_parser.add_argument("--pair-records", required=True)
        command_parser.add_argument("--article-index", required=True)
        command_parser.add_argument("--tokenizer-path", required=True)
        command_parser.add_argument("--output", required=True)
    evaluate_parser.add_argument("--weights", required=True)
    evaluate_parser.add_argument("--weights-sha256", required=True)
    evaluate_parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    try:
        if args.command == "audit":
            report = run_length_audit(
                pair_manifest=args.pair_manifest,
                pair_records=args.pair_records,
                article_index=args.article_index,
                tokenizer_path=args.tokenizer_path,
                output=args.output,
            )
            print(
                "SFT_PARENT_RAG_LENGTH_AUDIT_OK "
                f"records={report['length_audit']['records']} "
                f"all_fit={str(report['length_audit']['all_fit']).lower()}"
            )
        else:
            report = run_evaluation(
                pair_manifest=args.pair_manifest, pair_records=args.pair_records,
                article_index=args.article_index, tokenizer_path=args.tokenizer_path,
                weights=args.weights, weights_sha256=args.weights_sha256,
                output=args.output, device_name=args.device,
            )
            print(f"SFT_PARENT_RAG_EVALUATION_OK records={len(report['records'])}")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        raise SystemExit(f"[失败] {error}") from error


if __name__ == "__main__":
    main()
