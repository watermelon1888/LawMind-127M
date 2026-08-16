"""从 Query-SFT training release 生成严格三字段 JSON labels。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, BinaryIO

import torch
from torch.utils.data import Dataset

from rag.query import (
    QUERY_ENHANCEMENT_SYSTEM_PROMPT,
    QueryEnhancementProtocolError,
    parse_and_validate_query_enhancement,
)


MAX_SEQ_LEN = 768
PIPELINE = "query_sft_training_release_v1"
LABEL_MASK_VERSION = "query_assistant_json_only_v1"
RECORD_FIELDS = {"id", "source", "conversations"}


class QuerySftDatasetError(RuntimeError):
    """Query-SFT release、记录或标签边界无效。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_verified_manifest(path: Path) -> dict[str, Any]:
    hash_path = path.with_suffix(".sha256")
    if not path.is_file() or not hash_path.is_file():
        raise QuerySftDatasetError("Query-SFT release 或相邻哈希不存在")
    fields = hash_path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != path.name or fields[0] != _sha256_file(path):
        raise QuerySftDatasetError("Query-SFT release 相邻哈希无效")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftDatasetError("无法读取 Query-SFT release") from error
    if not isinstance(value, dict):
        raise QuerySftDatasetError("Query-SFT release 必须是 JSON object")
    return value


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    except Exception as error:
        raise QuerySftDatasetError("Tokenizer 编码失败") from error
    if not isinstance(values, list) or any(type(value) is not int for value in values):
        raise QuerySftDatasetError("Tokenizer 没有返回一维 input_ids")
    return values


def _subsequence_positions(values: list[int], pattern: list[int]) -> list[int]:
    return [
        index
        for index in range(len(values) - len(pattern) + 1)
        if values[index : index + len(pattern)] == pattern
    ]


def _candidate_identity(path: Path) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _validate_release(
    manifest_path: Path,
    tokenizer: Any,
    candidate_path: Path | None,
) -> tuple[Path, dict[str, Any], list[int], list[int]]:
    manifest = _load_verified_manifest(manifest_path)
    readiness = manifest.get("readiness")
    policy = manifest.get("policy")
    prompt = manifest.get("prompt")
    tokenizer_identity = manifest.get("tokenizer")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("pipeline") != PIPELINE
        or manifest.get("release_status")
        not in {"pilot_training_candidate", "formal_training_candidate"}
        or manifest.get("complete") is not True
        or not isinstance(readiness, dict)
        or readiness.get("protocol_audited") is not True
        or readiness.get("chat_template_length_audited") is not True
        or readiness.get("dataset_label_mask_audited") is not True
        or readiness.get("training_ready") is not True
        or not isinstance(policy, dict)
        or policy.get("fixed_max_seq_len") != MAX_SEQ_LEN
        or policy.get("truncation") != "forbidden"
        or policy.get("label_mask_version") != LABEL_MASK_VERSION
    ):
        raise QuerySftDatasetError("Query-SFT release 状态或训练策略无效")
    expected_prompt_sha = hashlib.sha256(
        QUERY_ENHANCEMENT_SYSTEM_PROMPT.encode("utf-8")
    ).hexdigest()
    if not isinstance(prompt, dict) or prompt.get("sha256") != expected_prompt_sha:
        raise QuerySftDatasetError("Query-SFT system prompt 身份无效")
    chat_template = getattr(tokenizer, "chat_template", None)
    if (
        not isinstance(tokenizer_identity, dict)
        or not isinstance(chat_template, str)
        or len(tokenizer) != tokenizer_identity.get("vocab_size")
        or hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
        != tokenizer_identity.get("chat_template_sha256")
    ):
        raise QuerySftDatasetError("Tokenizer 与 Query-SFT release 身份不一致")
    candidate = manifest.get("data", {}).get("training_candidate")
    if (
        not isinstance(candidate, dict)
        or type(candidate.get("records")) is not int
        or candidate["records"] <= 0
        or type(candidate.get("bytes")) is not int
        or not isinstance(candidate.get("sha256"), str)
        or not isinstance(candidate.get("path"), str)
    ):
        raise QuerySftDatasetError("Query-SFT candidate 身份无效")
    resolved = (
        Path(candidate["path"]).resolve()
        if candidate_path is None
        else Path(candidate_path).resolve()
    )
    if not resolved.is_file():
        raise QuerySftDatasetError(f"Query-SFT candidate 不存在: {resolved}")
    actual = _candidate_identity(resolved)
    if any(candidate.get(key) != actual[key] for key in ("bytes", "sha256")):
        raise QuerySftDatasetError("Query-SFT candidate 身份已变化")
    bos_token = getattr(tokenizer, "bos_token", None)
    eos_token = getattr(tokenizer, "eos_token", None)
    if not isinstance(bos_token, str) or not isinstance(eos_token, str):
        raise QuerySftDatasetError("Tokenizer 缺少 BOS/EOS token")
    prefix_ids = _token_ids(tokenizer, f"{bos_token}assistant\n<think>\n\n</think>\n\n")
    end_ids = _token_ids(tokenizer, f"{eos_token}\n")
    if not prefix_ids or not end_ids:
        raise QuerySftDatasetError("Tokenizer 无法编码 assistant 标签边界")
    return resolved, candidate, prefix_ids, end_ids


def _index_candidate(path: Path, metadata: dict[str, Any]) -> list[tuple[int, int]]:
    offsets = []
    offset = 0
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            if not raw_line.strip():
                raise QuerySftDatasetError(f"Query-SFT candidate 含空行: {line_number}")
            offsets.append((offset, line_number))
            offset += len(raw_line)
    if len(offsets) != metadata["records"]:
        raise QuerySftDatasetError("Query-SFT candidate 记录数与 release 不一致")
    return offsets


def _validate_record(record: object, line_number: int) -> dict[str, Any]:
    if (
        not isinstance(record, dict)
        or set(record) != RECORD_FIELDS
        or not isinstance(record.get("id"), str)
        or not record["id"]
        or record.get("source") != "query_sft"
    ):
        raise QuerySftDatasetError(f"Query-SFT 记录身份无效: {line_number}")
    conversations = record.get("conversations")
    if (
        not isinstance(conversations, list)
        or len(conversations) != 3
        or [item.get("role") for item in conversations]
        != ["system", "user", "assistant"]
        or any(
            not isinstance(item, dict)
            or set(item) != {"role", "content"}
            or not isinstance(item.get("content"), str)
            or not item["content"]
            for item in conversations
        )
    ):
        raise QuerySftDatasetError(f"Query-SFT conversations 无效: {line_number}")
    if conversations[0]["content"] != QUERY_ENHANCEMENT_SYSTEM_PROMPT:
        raise QuerySftDatasetError(f"Query-SFT system prompt 无效: {line_number}")
    try:
        parse_and_validate_query_enhancement(conversations[2]["content"])
    except QueryEnhancementProtocolError as error:
        raise QuerySftDatasetError(
            f"Query-SFT assistant JSON 无效: {line_number}"
        ) from error
    return record


class QuerySftDataset(Dataset):
    """随机访问固定 Query-SFT candidate，并生成 assistant-only labels。"""

    def __init__(
        self,
        release_manifest: str | Path,
        tokenizer: Any,
        *,
        candidate_path: str | Path | None = None,
    ) -> None:
        self.release_manifest = Path(release_manifest).resolve()
        self.tokenizer = tokenizer
        resolved, metadata, self._prefix_ids, self._end_ids = _validate_release(
            self.release_manifest,
            tokenizer,
            None if candidate_path is None else Path(candidate_path),
        )
        self.candidate_path = resolved
        self._offsets = _index_candidate(resolved, metadata)
        self._handles: dict[int, BinaryIO] = {}
        self._handle_pid: int | None = None

    def __len__(self) -> int:
        return len(self._offsets)

    def _handle(self) -> BinaryIO:
        pid = os.getpid()
        if self._handle_pid != pid:
            self.close()
            self._handle_pid = pid
        if pid not in self._handles:
            self._handles[pid] = self.candidate_path.open("rb")
        return self._handles[pid]

    def _load_record(self, index: int) -> tuple[dict[str, Any], int]:
        offset, line_number = self._offsets[index]
        handle = self._handle()
        handle.seek(offset)
        raw_line = handle.readline()
        try:
            record = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise QuerySftDatasetError(
                f"Query-SFT candidate JSON 无效: {line_number}"
            ) from error
        return _validate_record(record, line_number), line_number

    def __getitem__(self, index: int):
        record, line_number = self._load_record(index)
        conversations = record["conversations"]
        try:
            prompt = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as error:
            raise QuerySftDatasetError("Tokenizer chat template 渲染失败") from error
        expected_suffix = (
            f"{self.tokenizer.bos_token}assistant\n<think>\n\n</think>\n\n"
            f"{conversations[2]['content']}{self.tokenizer.eos_token}\n"
        )
        if not prompt.endswith(expected_suffix):
            raise QuerySftDatasetError(
                f"assistant content 被模板改写或边界变化: {line_number}"
            )
        input_ids = _token_ids(self.tokenizer, prompt)
        if len(input_ids) > MAX_SEQ_LEN:
            raise QuerySftDatasetError(
                f"模板后长度超过固定 768，禁止截断: {line_number}"
            )
        positions = _subsequence_positions(input_ids, self._prefix_ids)
        if len(positions) != 1:
            raise QuerySftDatasetError(f"assistant 前缀不能唯一定位: {line_number}")
        answer_start = positions[0] + len(self._prefix_ids)
        answer_end = len(input_ids) - len(self._end_ids)
        if answer_end < answer_start or input_ids[answer_end:] != self._end_ids:
            raise QuerySftDatasetError(f"assistant 结束标记无效: {line_number}")
        original_length = len(input_ids)
        labels = [-100] * MAX_SEQ_LEN
        labels[answer_start:original_length] = input_ids[answer_start:original_length]
        input_ids.extend(
            [self.tokenizer.pad_token_id] * (MAX_SEQ_LEN - original_length)
        )
        return torch.tensor(input_ids), torch.tensor(labels)

    def close(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            handle.close()
        getattr(self, "_handles", {}).clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_handle_pid"] = None
        return state

    def __del__(self):
        self.close()


__all__ = [
    "LABEL_MASK_VERSION",
    "MAX_SEQ_LEN",
    "PIPELINE",
    "QuerySftDataset",
    "QuerySftDatasetError",
]
