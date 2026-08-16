"""从 Query-SFT model-only 权重批量生成并严格解析三字段输出。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import torch

from rag.query import (
    QUERY_ENHANCEMENT_SYSTEM_PROMPT,
    QueryEnhancementProtocolError,
    build_query_enhancement_prompt,
    parse_and_validate_query_enhancement,
)

from . import train_full_sft as base_entry


PIPELINE = "query_sft_model_inference_v1"
MAX_NEW_TOKENS = 336


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_query_inputs(path: str | Path) -> tuple[dict[str, str], ...]:
    """加载稳定 query_id 与原始问题，不接收 required GT。"""

    records = []
    seen = set()
    with Path(path).open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"推理输入第 {line_number} 行不是有效 JSON") from error
            query_id = item.get("query_id", item.get("id"))
            query = item.get("query_original")
            if query is None and isinstance(item.get("conversations"), list):
                conversations = item["conversations"]
                if len(conversations) >= 2 and conversations[1].get("role") == "user":
                    query = conversations[1].get("content")
            if (
                not isinstance(query_id, str)
                or not query_id
                or not isinstance(query, str)
                or not query.strip()
            ):
                raise ValueError(f"推理输入第 {line_number} 行缺少 query_id/query_original")
            if query_id in seen:
                raise ValueError(f"推理输入包含重复 query_id: {query_id}")
            seen.add(query_id)
            records.append({"query_id": query_id, "query_original": query})
    if not records:
        raise ValueError("推理输入不能为空")
    return tuple(records)


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("Query-SFT 模型没有参数") from error


def generate_query_records(
    inputs: Iterable[dict[str, str]],
    *,
    model: torch.nn.Module,
    tokenizer: Any,
) -> tuple[dict[str, Any], ...]:
    """使用固定贪心配置逐题生成，并保留严格解析与回退记录。"""

    values = tuple(inputs)
    if not values:
        raise ValueError("inputs 不能为空")
    device = _model_device(model)
    model.eval()
    outputs = []
    with torch.inference_mode():
        for item in values:
            messages = build_query_enhancement_prompt(item["query_original"])
            try:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                new_tokens = generated[0, input_ids.shape[1] :]
                raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            except Exception as error:
                outputs.append(
                    {
                        "query_id": item["query_id"],
                        "query_original": item["query_original"],
                        "status": "fallback",
                        "failure_reason": "generation_failed",
                        "raw_output": None,
                        "enhancement": None,
                        "diagnostic": type(error).__name__,
                    }
                )
                continue
            try:
                parsed = parse_and_validate_query_enhancement(raw_output)
            except (QueryEnhancementProtocolError, TypeError, ValueError) as error:
                outputs.append(
                    {
                        "query_id": item["query_id"],
                        "query_original": item["query_original"],
                        "status": "fallback",
                        "failure_reason": "invalid_output",
                        "raw_output": raw_output,
                        "enhancement": None,
                        "diagnostic": type(error).__name__,
                    }
                )
                continue
            outputs.append(
                {
                    "query_id": item["query_id"],
                    "query_original": item["query_original"],
                    "status": "applied",
                    "failure_reason": None,
                    "raw_output": raw_output,
                    "enhancement": {
                        "rewrite": parsed.rewrite,
                        "expansion_terms": list(parsed.expansion_terms),
                        "subqueries": list(parsed.subqueries),
                    },
                    "diagnostic": None,
                }
            )
    return tuple(outputs)


def _write_immutable(path: Path, text: str) -> str:
    hash_path = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or hash_path.exists():
        raise FileExistsError(f"推理输出已存在，不能覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    digest = _sha256_file(path)
    hash_path.write_text(f"{digest}  {path.name}\n", encoding="utf-8", newline="\n")
    return digest


def publish_inference(output_dir, *, records, manifest):
    """不可变发布逐题输出及绑定权重、prompt、Tokenizer 的 manifest。"""

    directory = Path(output_dir).resolve()
    record_values = tuple(records)
    records_path = directory / "records.jsonl"
    manifest_path = directory / "manifest.json"
    records_text = "".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for item in record_values
    )
    records_sha = _write_immutable(records_path, records_text)
    final_manifest = dict(manifest)
    final_manifest["outputs"] = {
        "records": {
            "path": records_path.name,
            "records": len(record_values),
            "sha256": records_sha,
        }
    }
    _write_immutable(
        manifest_path,
        json.dumps(final_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    return final_manifest


def run_inference(*, input_path, weights_path, tokenizer_path, output_dir, device):
    """严格加载 model-only 权重并执行一次可审计的批量推理。"""

    weights = Path(weights_path).resolve()
    weights_sha = _sha256_file(weights)
    tokenizer = base_entry._load_tokenizer(tokenizer_path)
    tokenizer_identity = base_entry._tokenizer_identity(tokenizer, tokenizer_path)
    actual_device = torch.device(device)
    if actual_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前环境没有可用 GPU")
    model = base_entry.MiniMindForCausalLM(base_entry._model_config()).to(actual_device)
    base_entry._load_parent_weights(weights, weights_sha, model)
    inputs = load_query_inputs(input_path)
    records = generate_query_records(inputs, model=model, tokenizer=tokenizer)
    manifest = {
        "schema_version": "1.0",
        "pipeline": PIPELINE,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs": {
            "queries": {
                "path": str(Path(input_path).resolve()),
                "sha256": _sha256_file(Path(input_path).resolve()),
                "records": len(inputs),
            },
            "model_only_weights": {
                "path": str(weights),
                "sha256": weights_sha,
                "load_semantics": "strict_model_state_only",
            },
            "tokenizer": tokenizer_identity,
        },
        "prompt": {
            "sha256": hashlib.sha256(
                QUERY_ENHANCEMENT_SYSTEM_PROMPT.encode("utf-8")
            ).hexdigest()
        },
        "decoding": {
            "strategy": "greedy",
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
        },
        "records": {
            "total": len(records),
            "applied": sum(item["status"] == "applied" for item in records),
            "fallback": sum(item["status"] == "fallback" for item in records),
        },
        "complete": True,
    }
    return publish_inference(output_dir, records=records, manifest=manifest)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行 Query-SFT model-only 批量推理")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--weights_path", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = _parser().parse_args()
    run_inference(
        input_path=args.input_path,
        weights_path=args.weights_path,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
