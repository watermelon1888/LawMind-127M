"""使用固定法律语料训练项目当前唯一的 tokenizer。"""

from __future__ import annotations

import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = PROJECT_ROOT / "minimind" / "dataset" / "tokenizer_data" / "prepared" / "tokenizer_corpus.jsonl"
MODEL_DIR = PROJECT_ROOT / "minimind" / "model"
VOCAB_SIZE = 12000
SPECIAL_TOKENS_NUM = 36

CONTROL_TOKENS = [
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|audio_pad|>",
    "<tts_pad>",
    "<tts_text_bos>",
    "<tts_text_eod>",
    "<tts_text_bos_single>",
]
ADDITIONAL_TOKENS = [
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<think>",
    "</think>",
]
BUFFER_TOKENS = [
    f"<|buffer{index}|>"
    for index in range(1, SPECIAL_TOKENS_NUM - len(CONTROL_TOKENS + ADDITIONAL_TOKENS) + 1)
]
ALL_TOKENS = CONTROL_TOKENS + ADDITIONAL_TOKENS + BUFFER_TOKENS


def iter_texts(corpus_path: Path):
    """逐行读取准备好的 tokenizer 训练文本。"""
    if not corpus_path.is_file():
        raise FileNotFoundError(f"训练语料不存在，请先运行 prepare_tokenizer_corpus.py: {corpus_path}")

    with corpus_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"训练语料第 {line_number} 行不是有效 JSON") from error
            text = record.get("text") if isinstance(record, dict) else None
            if not isinstance(text, str) or not text:
                raise ValueError(f"训练语料第 {line_number} 行缺少非空 text 字段")
            yield text


def write_tokenizer_config(tokenizer: Tokenizer) -> None:
    """保留现有 Qwen3 chat template，并更新新词表的 token ID。"""
    config_path = MODEL_DIR / "tokenizer_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"缺少现有 tokenizer 配置: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["added_tokens_decoder"] = {
        str(tokenizer.token_to_id(token)): {
            "content": token,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": token in CONTROL_TOKENS,
        }
        for token in ALL_TOKENS
    }
    config["additional_special_tokens"] = [
        token for token in CONTROL_TOKENS if token != "<|endoftext|>"
    ]
    config["bos_token"] = "<|im_start|>"
    config["eos_token"] = "<|im_end|>"
    config["pad_token"] = "<|endoftext|>"
    config["unk_token"] = "<|endoftext|>"
    config["tokenizer_class"] = "PreTrainedTokenizerFast"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def mark_control_tokens_special() -> None:
    """仅将对话控制 token 标记为 special，保持原项目的处理方式。"""
    tokenizer_path = MODEL_DIR / "tokenizer.json"
    tokenizer_data = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    for token_info in tokenizer_data.get("added_tokens", []):
        token_info["special"] = token_info["content"] in CONTROL_TOKENS
    tokenizer_path.write_text(
        json.dumps(tokenizer_data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def train_tokenizer() -> None:
    """训练 12000 词表，并覆盖项目中的旧 tokenizer 文件。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=ALL_TOKENS,
    )
    tokenizer.train_from_iterator(iter_texts(CORPUS_PATH), trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.add_special_tokens(CONTROL_TOKENS)
    tokenizer.save(str(MODEL_DIR / "tokenizer.json"))
    tokenizer.model.save(str(MODEL_DIR))
    mark_control_tokens_special()
    write_tokenizer_config(tokenizer)


def verify_tokenizer() -> None:
    """验证新 tokenizer 可以加载、词表大小正确且能无损编解码。"""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    if len(tokenizer) != VOCAB_SIZE:
        raise ValueError(f"词表大小错误，期望 {VOCAB_SIZE}，实际 {len(tokenizer)}")

    samples = [
        "中华人民共和国民法典规定民事主体的人身权利、财产权利以及其他合法权益受法律保护。",
        "请解释取保候审的适用条件。",
    ]
    for text in samples:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if tokenizer.decode(token_ids, skip_special_tokens=False) != text:
            raise ValueError(f"编解码不一致: {text}")


def main() -> None:
    """按固定路径训练并验证法律 tokenizer。"""
    train_tokenizer()
    verify_tokenizer()
    print(f"Tokenizer 训练完成: {MODEL_DIR}")


if __name__ == "__main__":
    main()
