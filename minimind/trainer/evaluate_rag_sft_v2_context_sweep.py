"""评估同一 RAG-SFT v2 模型在四档自然重构包上下文中的表现。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from rag.eval.rag_sft_v2_context_sweep_inputs import (
    CONTEXT_LIMITS,
    EXPECTED_CELLS,
    EXPECTED_QUERIES,
    MAX_OUTPUT_TOKENS,
    PIPELINE as INPUT_PIPELINE,
    SELECTION_COUNTS,
)

from . import evaluate_rag_sft_v2 as base_eval
from . import train_full_sft as base_entry


PIPELINE = "rag_sft_v2_context_sweep_evaluation_v2"


def _portable_tokenizer_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    files = value.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("Tokenizer 身份缺少 files")
    return {
        "vocab_size": value.get("vocab_size"),
        "chat_template_sha256": value.get("chat_template_sha256"),
        "files": {
            name: {
                "bytes": item.get("bytes"),
                "sha256": item.get("sha256"),
            }
            for name, item in files.items()
            if isinstance(name, str) and isinstance(item, Mapping)
        },
    }


def _load_cases(
    manifest_path: str | Path, cases_path: str | Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest, manifest_sha256 = base_eval._load_verified_json(
        manifest_path, "上下文阶梯输入 manifest"
    )
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("pipeline") != INPUT_PIPELINE
        or manifest.get("policy", {}).get("context_limits")
        != list(CONTEXT_LIMITS)
        or manifest.get("policy", {}).get("max_output_tokens")
        != MAX_OUTPUT_TOKENS
        or manifest.get("policy", {}).get("inference_rope_scaling") is not False
        or manifest.get("records", {}).get("queries") != EXPECTED_QUERIES
        or manifest.get("records", {}).get("context_cells") != EXPECTED_CELLS
        or manifest.get("records", {}).get("selection_roles")
        != SELECTION_COUNTS
        or manifest.get("policy", {}).get("selection_uses_model_outputs")
        is not False
        or manifest.get("complete") is not True
    ):
        raise ValueError("上下文阶梯输入 manifest 身份或策略无效")
    metadata = manifest.get("output", {}).get("cases")
    if not isinstance(metadata, Mapping):
        raise ValueError("上下文阶梯输入 manifest 缺少 cases 身份")
    resolved = Path(cases_path or metadata.get("path", "")).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"上下文阶梯 cases 不存在: {resolved}")
    if (
        resolved.stat().st_size != metadata.get("bytes")
        or base_eval._sha256_file(resolved) != metadata.get("sha256")
    ):
        raise ValueError("上下文阶梯 cases 与 manifest 身份不一致")

    records = []
    seen = set()
    with resolved.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"上下文阶梯 cases 第 {line_number} 行为空")
            value = json.loads(line)
            _validate_case(value, line_number)
            identity = (value["context_limit"], value["query_id"])
            if identity in seen:
                raise ValueError("上下文阶梯 cases 单元身份重复")
            seen.add(identity)
            records.append(value)
    if len(records) != EXPECTED_CELLS:
        raise ValueError(f"上下文阶梯 cases 数量不闭合: {len(records)}")
    for context_limit in CONTEXT_LIMITS:
        current = [
            record for record in records if record["context_limit"] == context_limit
        ]
        if len(current) != EXPECTED_QUERIES:
            raise ValueError(f"{context_limit} 档评估题数量不闭合")
        role_counts = {
            role: sum(record["selection_role"] == role for record in current)
            for role in SELECTION_COUNTS
        }
        if role_counts != SELECTION_COUNTS:
            raise ValueError(f"{context_limit} 档选择分组数量不闭合")
    baseline_identity = {
        (
            record["query_id"],
            record["selection_role"],
            record["matched_query_id"],
        )
        for record in records
        if record["context_limit"] == CONTEXT_LIMITS[0]
    }
    for context_limit in CONTEXT_LIMITS[1:]:
        current_identity = {
            (
                record["query_id"],
                record["selection_role"],
                record["matched_query_id"],
            )
            for record in records
            if record["context_limit"] == context_limit
        }
        if current_identity != baseline_identity:
            raise ValueError(f"{context_limit} 档选择身份或配对漂移")
    return records, {
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": manifest_sha256,
        "cases_path": str(resolved),
        "cases_sha256": metadata["sha256"],
        "tokenizer": manifest.get("inputs", {}).get("tokenizer"),
    }


def _validate_case(record: object, line_number: int) -> None:
    expected = {
        "query_id",
        "query_type",
        "selection_role",
        "matched_query_id",
        "context_limit",
        "status",
        "failure_reason",
        "query",
        "evidence",
        "visible_chunk_ids",
        "required_chunk_ids",
        "hard_negative_chunk_ids",
        "retrieval_attribution",
    }
    if not isinstance(record, dict) or set(record) != expected:
        raise ValueError(f"上下文阶梯 cases 第 {line_number} 行 schema 无效")
    if (
        not isinstance(record.get("query_id"), str)
        or not record["query_id"]
        or record.get("query_type") != "legal_query"
        or record.get("selection_role") not in SELECTION_COUNTS
        or (
            record["selection_role"] == "retrieval_failure_negative_control"
            and record.get("matched_query_id") is not None
        )
        or (
            record["selection_role"] != "retrieval_failure_negative_control"
            and not isinstance(record.get("matched_query_id"), str)
        )
        or record.get("context_limit") not in CONTEXT_LIMITS
        or record.get("status") not in {"ready", "overbudget"}
        or not isinstance(record.get("required_chunk_ids"), list)
        or not record["required_chunk_ids"]
        or len(record["required_chunk_ids"])
        != len(set(record["required_chunk_ids"]))
        or not isinstance(record.get("retrieval_attribution"), dict)
    ):
        raise ValueError(f"上下文阶梯 cases 第 {line_number} 行身份无效")
    if record["status"] == "ready":
        base_eval._package(record)
        prompt_tokens = record["retrieval_attribution"].get("prompt_tokens")
        if (
            not isinstance(prompt_tokens, int)
            or prompt_tokens <= 0
            or prompt_tokens + MAX_OUTPUT_TOKENS > record["context_limit"]
        ):
            raise ValueError(
                f"上下文阶梯 cases 第 {line_number} 行 ready 预算无效"
            )
    elif (
        record["evidence"]
        or record["visible_chunk_ids"]
        or record["hard_negative_chunk_ids"]
        or not isinstance(record.get("failure_reason"), str)
        or not record["failure_reason"]
    ):
        raise ValueError(
            f"上下文阶梯 cases 第 {line_number} 行 overbudget 状态无效"
        )


def _input_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ready = [record for record in records if record["status"] == "ready"]
    prompts = [
        int(record["retrieval_attribution"]["prompt_tokens"])
        for record in ready
    ]
    evidence_counts = [len(record["visible_chunk_ids"]) for record in ready]
    required_coverage = []
    for record in ready:
        required = set(record["required_chunk_ids"])
        visible = set(record["visible_chunk_ids"])
        required_coverage.append(len(required & visible) / len(required))
    return {
        "records": len(records),
        "ready": len(ready),
        "overbudget": len(records) - len(ready),
        "packaged_complete": sum(
            record["retrieval_attribution"].get("packaged_complete") is True
            for record in ready
        ),
        "packaged_incomplete": sum(
            record["retrieval_attribution"].get("packaged_complete") is False
            for record in ready
        ),
        "prompt_tokens": {
            "min": min(prompts) if prompts else None,
            "max": max(prompts) if prompts else None,
            "mean": sum(prompts) / len(prompts) if prompts else None,
        },
        "evidence_items": {
            "min": min(evidence_counts) if evidence_counts else None,
            "max": max(evidence_counts) if evidence_counts else None,
            "mean": (
                sum(evidence_counts) / len(evidence_counts)
                if evidence_counts
                else None
            ),
        },
        "visible_required_recall": (
            sum(required_coverage) / len(required_coverage)
            if required_coverage
            else None
        ),
    }


def _metrics_with_generation(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values = list(records)
    summary = base_eval._summary_metrics(values)
    summary["generation"] = {
        "hit_max_new_tokens": sum(
            record.get("generation_metrics", {}).get("hit_max_new_tokens") is True
            for record in values
        ),
        "hit_max_new_tokens_rate": (
            sum(
                record.get("generation_metrics", {}).get("hit_max_new_tokens")
                is True
                for record in values
            )
            / len(values)
            if values
            else 0.0
        ),
    }
    return summary


def _retrieval_failure_summary(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    values = list(records)
    valid = sum(record["protocol_valid"] for record in values)
    return {
        "records": len(values),
        "excluded_from_answer_model_recall_and_exact_set": True,
        "protocol_valid": valid,
        "protocol_valid_rate": valid / len(values) if values else 0.0,
        "citation_precision": (
            sum(record["citation_metrics"]["precision"] for record in values)
            / len(values)
            if values
            else 0.0
        ),
        "hard_negative_citation_rate": (
            sum(
                record["citation_metrics"]["hard_negative_citations"] > 0
                for record in values
            )
            / len(values)
            if values
            else 0.0
        ),
        "hit_max_new_tokens_rate": (
            sum(
                record.get("generation_metrics", {}).get("hit_max_new_tokens")
                is True
                for record in values
            )
            / len(values)
            if values
            else 0.0
        ),
    }


def _answer_summary(
    outputs: Sequence[Mapping[str, Any]], common_query_ids: set[str]
) -> dict[str, Any]:
    values = list(outputs)
    answer_model_scope = [
        record
        for record in values
        if record["selection_role"]
        in {"rescuable_packaging", "complete_matched_control"}
    ]
    retrieval_failures = [
        record
        for record in values
        if record["selection_role"] == "retrieval_failure_negative_control"
    ]
    return {
        "system_all_ready": _metrics_with_generation(values),
        "answer_model_scope": _metrics_with_generation(answer_model_scope),
        "paired_common": _metrics_with_generation(
            [record for record in values if record["query_id"] in common_query_ids]
        ),
        "rescuable_packaging": _metrics_with_generation(
            [
                record
                for record in values
                if record["selection_role"] == "rescuable_packaging"
            ]
        ),
        "complete_matched_control": _metrics_with_generation(
            [
                record
                for record in values
                if record["selection_role"] == "complete_matched_control"
            ]
        ),
        "retrieval_failure_negative_control": _retrieval_failure_summary(
            retrieval_failures
        ),
        "packaged_complete": _metrics_with_generation(
            [
                record
                for record in values
                if record["retrieval_attribution"].get("packaged_complete") is True
            ]
        ),
        "packaged_incomplete": _metrics_with_generation(
            [
                record
                for record in values
                if record["retrieval_attribution"].get("packaged_complete") is False
            ]
        ),
    }


def _transition_summary(
    prepared: Sequence[Mapping[str, Any]], outputs: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    inputs = {
        (record["context_limit"], record["query_id"]): record
        for record in prepared
    }
    generated = {
        (record["context_limit"], record["query_id"]): record
        for record in outputs
    }
    baseline = [
        record for record in prepared if record["context_limit"] == CONTEXT_LIMITS[0]
    ]
    baseline_overbudget = {
        record["query_id"] for record in baseline if record["status"] == "overbudget"
    }
    baseline_incomplete = {
        record["query_id"]
        for record in baseline
        if record["status"] == "ready"
        and record["retrieval_attribution"].get("packaged_complete") is False
    }
    baseline_rescuable = {
        record["query_id"]
        for record in baseline
        if record["selection_role"] == "rescuable_packaging"
    }
    baseline_controls = {
        record["query_id"]
        for record in baseline
        if record["selection_role"] == "complete_matched_control"
    }
    baseline_control_exact = {
        query_id
        for query_id in baseline_controls
        if (CONTEXT_LIMITS[0], query_id) in generated
        and generated[(CONTEXT_LIMITS[0], query_id)]["citation_metrics"][
            "exact_set"
        ]
    }
    by_context = {}
    for context_limit in CONTEXT_LIMITS:
        newly_ready = {
            query_id
            for query_id in baseline_overbudget
            if inputs[(context_limit, query_id)]["status"] == "ready"
        }
        newly_complete = {
            query_id
            for query_id in baseline_incomplete
            if inputs[(context_limit, query_id)]["status"] == "ready"
            and inputs[(context_limit, query_id)]["retrieval_attribution"].get(
                "packaged_complete"
            )
            is True
        }
        rescuable_complete = newly_complete & baseline_rescuable
        control_exact = {
            query_id
            for query_id in baseline_controls
            if (context_limit, query_id) in generated
            and generated[(context_limit, query_id)]["citation_metrics"]["exact_set"]
        }
        retained_control_exact = control_exact & baseline_control_exact
        by_context[str(context_limit)] = {
            "baseline_overbudget_now_ready": len(newly_ready),
            "baseline_overbudget_now_ready_query_ids": sorted(newly_ready),
            "baseline_packaged_incomplete_now_complete": len(newly_complete),
            "baseline_packaged_incomplete_now_complete_query_ids": sorted(
                newly_complete
            ),
            "newly_complete_exact_set_success": sum(
                generated[(context_limit, query_id)]["citation_metrics"]["exact_set"]
                for query_id in newly_complete
            ),
            "rescuable_packaged_complete": len(rescuable_complete),
            "rescuable_packaged_complete_query_ids": sorted(rescuable_complete),
            "rescuable_net_exact_set_success": sum(
                generated[(context_limit, query_id)]["citation_metrics"]["exact_set"]
                for query_id in rescuable_complete
            ),
            "complete_control_exact_set_success": len(control_exact),
            "complete_control_baseline_successes_retained": len(
                retained_control_exact
            ),
            "complete_control_baseline_success_retention_rate": (
                len(retained_control_exact) / len(baseline_control_exact)
                if baseline_control_exact
                else None
            ),
        }
    return {
        "baseline_context_limit": CONTEXT_LIMITS[0],
        "baseline_overbudget": len(baseline_overbudget),
        "baseline_packaged_incomplete": len(baseline_incomplete),
        "baseline_rescuable_packaging": len(baseline_rescuable),
        "baseline_retrieval_failure_negative_control": sum(
            record["selection_role"] == "retrieval_failure_negative_control"
            for record in baseline
        ),
        "baseline_complete_matched_control": len(baseline_controls),
        "baseline_complete_control_exact_set_success": len(baseline_control_exact),
        "by_context": by_context,
    }


def run_context_sweep_evaluation(
    *,
    cases_manifest: str | Path,
    cases: str | Path | None,
    tokenizer_path: str | Path,
    weights: str | Path,
    weights_sha256: str,
    output: str | Path,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    prepared, input_identity = _load_cases(cases_manifest, cases)
    tokenizer = base_entry._load_tokenizer(tokenizer_path)
    tokenizer_identity = base_eval._tokenizer_identity(tokenizer, tokenizer_path)
    if _portable_tokenizer_identity(tokenizer_identity) != _portable_tokenizer_identity(
        input_identity["tokenizer"]
    ):
        raise ValueError("当前 Tokenizer 与上下文阶梯输入身份不一致")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 CUDA 但当前无可用 GPU")
    config = base_entry._model_config()
    if (
        getattr(config, "inference_rope_scaling", False)
        or getattr(config, "max_position_embeddings", 0) < max(CONTEXT_LIMITS)
    ):
        raise ValueError("模型 RoPE 配置不满足原始位置编码上下文阶梯实验")
    model = base_entry.MiniMindForCausalLM(config).to(device)
    actual_weights_sha256 = base_entry._load_parent_weights(
        weights, weights_sha256, model
    )

    ready_by_context = {
        context_limit: [
            record
            for record in prepared
            if record["context_limit"] == context_limit
            and record["status"] == "ready"
        ]
        for context_limit in CONTEXT_LIMITS
    }
    common_query_ids = set.intersection(
        *(
            {record["query_id"] for record in records}
            for records in ready_by_context.values()
        )
    )
    outputs = []
    runtime = {}
    for context_limit in CONTEXT_LIMITS:
        current = ready_by_context[context_limit]
        evaluation_records = [
            {**record, "evaluation_scope": f"context_{context_limit}"}
            for record in current
        ]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        generated = base_eval.evaluate_records(
            evaluation_records,
            model=model,
            tokenizer=tokenizer,
            device=device,
            context_limit_override=context_limit,
        )
        selection_by_query_id = {
            record["query_id"]: (
                record["selection_role"],
                record["matched_query_id"],
            )
            for record in current
        }
        elapsed = time.perf_counter() - started
        for record in generated:
            role, matched_query_id = selection_by_query_id[record["query_id"]]
            record["selection_role"] = role
            record["matched_query_id"] = matched_query_id
            record["context_limit"] = context_limit
        outputs.extend(generated)
        runtime[str(context_limit)] = {
            "generated_records": len(generated),
            "elapsed_seconds": elapsed,
            "records_per_second": len(generated) / elapsed if elapsed else None,
            "cuda_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None
            ),
            "cuda_peak_reserved_bytes": (
                torch.cuda.max_memory_reserved(device)
                if device.type == "cuda"
                else None
            ),
        }

    contexts = {}
    for context_limit in CONTEXT_LIMITS:
        current_inputs = [
            record
            for record in prepared
            if record["context_limit"] == context_limit
        ]
        current_outputs = [
            record for record in outputs if record["context_limit"] == context_limit
        ]
        contexts[str(context_limit)] = {
            "inputs": _input_summary(current_inputs),
            "answers": _answer_summary(current_outputs, common_query_ids),
            "runtime": runtime[str(context_limit)],
        }
    skipped = [record for record in prepared if record["status"] == "overbudget"]
    report = {
        "schema_version": "1.0",
        "pipeline": PIPELINE,
        "inputs": {
            "context_sweep": input_identity,
            "weights_sha256": actual_weights_sha256,
            "tokenizer": tokenizer_identity,
        },
        "generation": {
            "context_limits": list(CONTEXT_LIMITS),
            "max_new_tokens": MAX_OUTPUT_TOKENS,
            "do_sample": False,
            "use_cache": True,
            "inference_rope_scaling": False,
            "retry": False,
            "repair": False,
        },
        "summary": {
            "query_identities": EXPECTED_QUERIES,
            "context_cells": EXPECTED_CELLS,
            "generated_records": len(outputs),
            "skipped_overbudget_records": len(skipped),
            "paired_common_queries": len(common_query_ids),
            "paired_common_query_ids": sorted(common_query_ids),
            "contexts": contexts,
            "transitions": _transition_summary(prepared, outputs),
        },
        "records": outputs,
        "skipped": skipped,
        "complete": True,
    }
    output_path = Path(output).resolve()
    output_sidecar = output_path.with_suffix(output_path.suffix + ".sha256")
    if output_path.exists() or output_sidecar.exists():
        raise FileExistsError(f"上下文阶梯评估输出已存在: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    digest = base_eval._sha256_file(output_path)
    output_sidecar.write_text(
        f"{digest}  {output_path.name}\n", encoding="utf-8", newline="\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="评估固定 RAG-SFT v2 checkpoint 的四档自然上下文表现"
    )
    parser.add_argument("--cases-manifest", required=True)
    parser.add_argument("--cases")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run_context_sweep_evaluation(
        cases_manifest=args.cases_manifest,
        cases=args.cases,
        tokenizer_path=args.tokenizer_path,
        weights=args.weights,
        weights_sha256=args.weights_sha256,
        output=args.output,
        device_name=args.device,
    )
    print(
        "RAG_SFT_V2_CONTEXT_SWEEP_EVALUATION_OK "
        f"generated={report['summary']['generated_records']}"
    )


if __name__ == "__main__":
    main()
