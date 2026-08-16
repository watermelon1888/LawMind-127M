"""交互式体验 RAG-SFT v2 model-only 权重。"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from rag.answering import EvidencePackage, build_answer_prompt, parse_and_validate_answer
from rag.answering.protocol import AnswerProtocolError
from rag.core.contracts import Evidence

from . import train_full_sft as base_entry


CONTEXT_LIMIT = 768
MAX_NEW_TOKENS = 150
MAX_EVIDENCE_ITEMS = 5
DEFAULT_TOKENIZER_PATH = Path(__file__).resolve().parents[1] / "model"


@dataclass(frozen=True)
class GenerationResult:
    """保存一次交互生成及协议校验结果。"""

    raw_output: str
    parsed_output: dict[str, Any] | None
    protocol_error: str | None
    prompt_tokens: int
    generated_tokens: int
    eos_generated: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="rag_epoch_2.pth 路径")
    parser.add_argument("--weights-sha256", required=True, help="权重的预期 SHA-256")
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
        help="冻结 Tokenizer 目录",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="推理设备，例如 cuda:0 或 cpu",
    )
    return parser


def load_runtime(
    *, weights: Path, weights_sha256: str, tokenizer_path: Path, device_name: str
) -> tuple[Any, Any, torch.device]:
    """按正式训练结构和冻结 Tokenizer 加载 model-only 权重。"""
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 CUDA 设备，但当前没有可用 GPU")

    tokenizer = base_entry._load_tokenizer(tokenizer_path)
    model = base_entry.MiniMindForCausalLM(base_entry._model_config())
    base_entry._load_parent_weights(weights, weights_sha256.lower(), model)
    if device.type == "cuda":
        model = model.half()
    model = model.eval().to(device)
    return model, tokenizer, device


def generate_answer(
    package: EvidencePackage,
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
) -> GenerationResult:
    """使用正式 prompt、768 上下文和 greedy 解码生成一次回答。"""
    prompt = tokenizer.apply_chat_template(
        build_answer_prompt(package),
        tokenize=False,
        add_generation_prompt=True,
        tools=None,
        open_thinking=False,
    )
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("chat template 未返回非空 prompt")

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=False,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    else:
        attention_mask = attention_mask.to(device)
    prompt_tokens = int(input_ids.shape[1])
    if prompt_tokens + MAX_NEW_TOKENS > CONTEXT_LIMIT:
        raise ValueError(
            f"prompt_tokens={prompt_tokens} 超过固定预算 "
            f"{CONTEXT_LIMIT}-{MAX_NEW_TOKENS}={CONTEXT_LIMIT - MAX_NEW_TOKENS}"
        )

    with torch.inference_mode():
        generated = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=True,
        )
    generated_ids = generated[0, prompt_tokens:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    generated_tokens = int(generated_ids.shape[0])
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    eos_generated = bool(
        generated_tokens
        and eos_token_id is not None
        and generated[0, -1].item() == eos_token_id
    )

    try:
        answer = parse_and_validate_answer(package, raw_output)
        parsed_output = {
            "summary": answer.summary,
            "citations": list(answer.citations),
        }
        protocol_error = None
    except (AnswerProtocolError, TypeError, ValueError) as error:
        parsed_output = None
        protocol_error = str(error)

    return GenerationResult(
        raw_output=raw_output,
        parsed_output=parsed_output,
        protocol_error=protocol_error,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        eos_generated=eos_generated,
    )


def read_package(input_fn: Callable[[str], str] = input) -> EvidencePackage | None:
    """从终端读取一条独立 query 和一至五条有序完整证据。"""
    query = input_fn("\n问题（直接回车退出）：").strip()
    if not query:
        return None
    count_text = input_fn("证据数量（1-5）：").strip()
    try:
        count = int(count_text)
    except ValueError as error:
        raise ValueError("证据数量必须是 1-5 的整数") from error
    if not 1 <= count <= MAX_EVIDENCE_ITEMS:
        raise ValueError("证据数量必须是 1-5 的整数")

    evidence = []
    for index in range(1, count + 1):
        law_name = input_fn(f"E{index} 法律名称：").strip()
        article_no = input_fn(f"E{index} 条号：").strip()
        content = input_fn(f"E{index} 完整法条内容（单行粘贴）：").strip()
        evidence.append(Evidence(law_name, article_no, content))
    return EvidencePackage(query=query, evidence=tuple(evidence))


def run_interactive(*, model: Any, tokenizer: Any, device: torch.device) -> None:
    """循环执行彼此独立的单 query RAG 请求。"""
    print("RAG-SFT v2 交互模式：每轮问题相互独立，不携带聊天历史。")
    print(f"固定配置：context={CONTEXT_LIMIT}, max_new_tokens={MAX_NEW_TOKENS}, greedy=true")
    while True:
        try:
            package = read_package()
            if package is None:
                return
            result = generate_answer(
                package, model=model, tokenizer=tokenizer, device=device
            )
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            return
        except (TypeError, ValueError) as error:
            print(f"输入或预算错误：{error}")
            continue

        print("\n模型原始输出：")
        print(result.raw_output)
        if result.parsed_output is None:
            print(f"协议校验：失败（{result.protocol_error}）")
        else:
            print("协议校验：通过")
            print(json.dumps(result.parsed_output, ensure_ascii=False, separators=(",", ":")))
        print(
            "生成统计："
            f"prompt_tokens={result.prompt_tokens}, "
            f"generated_tokens={result.generated_tokens}, "
            f"eos_generated={str(result.eos_generated).lower()}"
        )


def main() -> None:
    args = build_parser().parse_args()
    model, tokenizer, device = load_runtime(
        weights=args.weights,
        weights_sha256=args.weights_sha256,
        tokenizer_path=args.tokenizer_path,
        device_name=args.device,
    )
    run_interactive(model=model, tokenizer=tokenizer, device=device)


if __name__ == "__main__":
    main()
