"""运行本地 MiniMind 模型与现行法律 RAG 的终端问答入口。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from rag.answering import (
    AnswerPromptTokenCounter,
    EvidencePackager,
    RAG_MAX_OUTPUT_TOKENS,
)
from rag.core import CurrentLawRAG, LegalRAG
from rag.knowledge import ArticleRepository
from rag.retrieval import load_semantic_retriever
from rag.retrieval.loader import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL

from .chat_rag_sft_v2 import CONTEXT_LIMIT, DEFAULT_TOKENIZER_PATH, load_runtime


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTICLE_INDEX = PROJECT_ROOT / "rag" / "chunk" / "article_index.jsonl"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "rag" / "retrieval" / "artifacts"


def resolve_local_huggingface_model(model_id: str) -> Path:
    """把固定 Hugging Face 模型 ID 解析为本机缓存快照。"""
    hub_cache = os.environ.get("HF_HUB_CACHE")
    if hub_cache:
        hub_root = Path(hub_cache)
    else:
        hf_home = os.environ.get("HF_HOME")
        hub_root = (
            Path(hf_home) / "hub"
            if hf_home
            else Path.home() / ".cache" / "huggingface" / "hub"
        )
    repository = hub_root / f"models--{model_id.replace('/', '--')}"
    main_ref = repository / "refs" / "main"
    if not main_ref.is_file():
        raise FileNotFoundError(f"本机缺少 Hugging Face 模型缓存：{model_id}")
    revision = main_ref.read_text(encoding="utf-8").strip()
    snapshot = repository / "snapshots" / revision
    if not revision or not snapshot.is_dir():
        raise FileNotFoundError(f"本机 Hugging Face 模型缓存不完整：{model_id}")
    return snapshot


class MiniMindGenerator:
    """把本地 MiniMind 推理适配为 CurrentLawRAG 的生成接口。"""

    def __init__(self, *, model: Any, tokenizer: Any, device: torch.device):
        if not callable(getattr(model, "generate", None)):
            raise TypeError("model 必须提供可调用的 generate")
        if not callable(getattr(tokenizer, "apply_chat_template", None)):
            raise TypeError("tokenizer 必须提供 apply_chat_template")
        if not callable(tokenizer):
            raise TypeError("tokenizer 必须可调用")
        if not isinstance(device, torch.device):
            raise TypeError("device 必须是 torch.device")
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    def __call__(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """使用固定无思考模板和 greedy 解码返回模型原始文本。"""
        if temperature != 0:
            raise ValueError("RAG 回答模型只支持 temperature=0")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ValueError("max_tokens 必须是正整数")

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            tools=None,
            open_thinking=False,
        )
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("chat template 未返回非空 prompt")
        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=False,
        )
        input_ids = encoded["input_ids"].to(self._device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(self._device)
        prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens + max_tokens > CONTEXT_LIMIT:
            raise ValueError("模型输入超过固定上下文预算")

        with torch.inference_mode():
            generated = self._model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,
                use_cache=True,
            )
        generated_ids = generated[0, prompt_tokens:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True)


def load_application(
    *,
    weights: Path,
    weights_sha256: str,
    tokenizer_path: Path,
    article_index: Path,
    artifact_dir: Path,
    device_name: str,
    output_fn: Callable[[str], None] = print,
) -> LegalRAG:
    """加载一次完整本地运行时并返回统一法律 RAG 接口。"""
    output_fn("[1/3] 正在加载法律知识库……")
    repository = ArticleRepository.from_jsonl(article_index)

    output_fn("[2/3] 正在加载 dense、BM25 与 reranker……")
    embedding_model = resolve_local_huggingface_model(DEFAULT_EMBEDDING_MODEL)
    reranker_model = resolve_local_huggingface_model(DEFAULT_RERANKER_MODEL)
    retriever = load_semantic_retriever(
        repository=repository,
        artifact_dir=artifact_dir,
        embedding_model=str(embedding_model),
        reranker_model=str(reranker_model),
        device=device_name,
    )

    output_fn("[3/3] 正在加载 MiniMind 回答模型……")
    model, tokenizer, device = load_runtime(
        weights=weights,
        weights_sha256=weights_sha256,
        tokenizer_path=tokenizer_path,
        device_name=device_name,
    )
    packager = EvidencePackager(
        context_limit=CONTEXT_LIMIT,
        max_output_tokens=RAG_MAX_OUTPUT_TOKENS,
        count_prompt_tokens=AnswerPromptTokenCounter(tokenizer),
    )
    application = CurrentLawRAG(
        article_repository=repository,
        semantic_retriever=retriever,
        evidence_packager=packager,
        generate=MiniMindGenerator(model=model, tokenizer=tokenizer, device=device),
        max_output_tokens=RAG_MAX_OUTPUT_TOKENS,
    )
    output_fn("加载完成。")
    return application


def run_interactive(
    application: LegalRAG,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> None:
    """循环读取独立法律问题并展示完整 RAG 回答。"""
    output_fn("现行法律 RAG 问答：每轮问题相互独立，不携带聊天历史。")
    while True:
        try:
            query = input_fn("\n请输入法律问题（直接回车退出）：")
        except (EOFError, KeyboardInterrupt):
            output_fn("\n已退出。")
            return
        if not query.strip():
            output_fn("已退出。")
            return
        result = application.answer(query)
        output_fn("\n" + result.rendered_answer.to_text())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True, help="RAG-SFT v2 权重路径")
    parser.add_argument("--weights-sha256", required=True, help="权重的预期 SHA-256")
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=DEFAULT_TOKENIZER_PATH,
        help="冻结 Tokenizer 目录",
    )
    parser.add_argument(
        "--article-index",
        type=Path,
        default=DEFAULT_ARTICLE_INDEX,
        help="canonical 法条索引路径",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="dense 与 BM25 检索产物目录",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="模型与检索设备，例如 cuda:0 或 cpu",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    application = load_application(
        weights=args.weights,
        weights_sha256=args.weights_sha256.lower(),
        tokenizer_path=args.tokenizer_path,
        article_index=args.article_index,
        artifact_dir=args.artifact_dir,
        device_name=args.device,
    )
    run_interactive(application)


if __name__ == "__main__":
    main()
