"""加载阶段 B 最终权重并执行可复现的固定前缀续写验收。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from trainer.trainer_utils import setup_seed


DEFAULT_WEIGHT_PATH = (
    "/root/autodl-tmp/minimind-work/checkpoints/stage-b/"
    "stage-b-lr-1e-3/weights/pretrain-9737362944.pth"
)
DEFAULT_TOKENIZER_PATH = "/root/autodl-tmp/minimind/model"
DEFAULT_OUTPUT_PATH = (
    "/root/autodl-tmp/minimind-work/runs/stage-b/"
    "stage-b-lr-1e-3/final-generation-9737362944.json"
)
PROMPTS = (
    "中国幅员辽阔，",
    "在计算机科学中，神经网络",
    "水在标准大气压下",
    "研究人员通过实验发现，",
    "《中华人民共和国民法典》规定，",
    "春天到了，田野里的",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="阶段 B 最终预训练权重续写验收")
    parser.add_argument("--weight_path", default=DEFAULT_WEIGHT_PATH)
    parser.add_argument("--tokenizer_path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--max_new_tokens", type=_positive_int, default=128)
    parser.add_argument("--base_seed", type=int, default=20260728)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--top_k", type=_positive_int, default=50)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_model(weight_path: Path, device: torch.device) -> MiniMindForCausalLM:
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 CUDA 设备不支持 BF16")

    state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("权重文件不是非空 model state_dict")
    if not all(isinstance(value, torch.Tensor) for value in state_dict.values()):
        raise ValueError("model state_dict 包含非 Tensor 值")
    floating_tensors = [value for value in state_dict.values() if value.is_floating_point()]
    if not floating_tensors or any(
        value.dtype != torch.bfloat16 for value in floating_tensors
    ):
        raise ValueError("最终导出权重的浮点 Tensor 必须全部为 BF16")

    model = MiniMindForCausalLM(MiniMindConfig())
    model.load_state_dict(state_dict, strict=True)
    del state_dict

    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return model.to(device=device, dtype=dtype).eval()


def _generate_samples(
    model: MiniMindForCausalLM,
    tokenizer,
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict]:
    if tokenizer.bos_token is None:
        raise ValueError("Tokenizer 缺少 bos_token")

    samples = []
    for index, prefix in enumerate(PROMPTS, start=1):
        seed = args.base_seed + index - 1
        setup_seed(seed)
        encoded = tokenizer(
            tokenizer.bos_token + prefix,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(device)
        input_length = int(encoded["input_ids"].shape[1])
        generated = model.generate(
            inputs=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=1.0,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated_ids = generated[0, input_length:].detach().cpu().tolist()
        continuation = tokenizer.decode(generated_ids, skip_special_tokens=True)
        unk_occurrences = (
            generated_ids.count(tokenizer.unk_token_id)
            if tokenizer.unk_token_id is not None
            else 0
        )
        sample = {
            "index": index,
            "seed": seed,
            "prefix": prefix,
            "continuation": continuation,
            "generated_tokens": len(generated_ids),
            "ended_with_eos": bool(
                generated_ids
                and tokenizer.eos_token_id is not None
                and generated_ids[-1] == tokenizer.eos_token_id
            ),
            "empty_continuation": not continuation.strip(),
            "contains_replacement_character": "\ufffd" in continuation,
            "contains_nul": "\x00" in continuation,
            "unk_token_occurrences": unk_occurrences,
        }
        samples.append(sample)

        print("=" * 80)
        print(f"样本 {index} | seed={seed}")
        print(f"前缀: {prefix}")
        print(f"续写: {continuation}")
        print(f"生成 tokens: {len(generated_ids)}")
        print(f"包含替换字符: {sample['contains_replacement_character']}")
        print(f"UNK 次数: {unk_occurrences}")

    return samples


def _summarize(samples: list[dict]) -> dict:
    summary = {
        "samples": len(samples),
        "empty_continuations": sum(item["empty_continuation"] for item in samples),
        "replacement_character_samples": sum(
            item["contains_replacement_character"] for item in samples
        ),
        "nul_samples": sum(item["contains_nul"] for item in samples),
        "unk_token_occurrences": sum(
            item["unk_token_occurrences"] for item in samples
        ),
    }
    summary["structural_checks_passed"] = not any(
        summary[key]
        for key in (
            "empty_continuations",
            "replacement_character_samples",
            "nul_samples",
            "unk_token_occurrences",
        )
    )
    return summary


def main() -> int:
    args = build_parser().parse_args()
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        raise ValueError("temperature 必须大于 0，top_p 必须位于 (0, 1]")

    weight_path = Path(args.weight_path).resolve()
    tokenizer_path = Path(args.tokenizer_path).resolve()
    output_path = Path(args.output_path).resolve()
    if not weight_path.is_file():
        raise FileNotFoundError(f"权重文件不存在: {weight_path}")
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Tokenizer 目录不存在: {tokenizer_path}")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )
    model = _load_model(weight_path, device)
    if len(tokenizer) != model.config.vocab_size:
        raise ValueError(
            f"Tokenizer 大小 {len(tokenizer)} 与模型词表 {model.config.vocab_size} 不一致"
        )
    samples = _generate_samples(model, tokenizer, device, args)
    summary = _summarize(samples)

    report = {
        "schema_version": "1.0",
        "weight": {
            "path": str(weight_path),
            "size_bytes": weight_path.stat().st_size,
            "sha256": _sha256(weight_path),
            "floating_dtype": "bfloat16",
        },
        "tokenizer_path": str(tokenizer_path),
        "model": {
            "vocab_size": model.config.vocab_size,
            "hidden_size": model.config.hidden_size,
            "num_hidden_layers": model.config.num_hidden_layers,
            "num_attention_heads": model.config.num_attention_heads,
            "num_key_value_heads": model.config.num_key_value_heads,
            "intermediate_size": model.config.intermediate_size,
        },
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "base_seed": args.base_seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": 1.0,
        },
        "summary": summary,
        "samples": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"验收报告: {output_path}")
    print(f"权重 SHA-256: {report['weight']['sha256']}")
    return 0 if summary["structural_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
