"""为固定 768 的 RAG-SFT candidate 生成 JSON-only labels。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, BinaryIO

import torch
from torch.utils.data import Dataset

from rag.answering import (
    SYSTEM_PROMPT,
    AnswerProtocolError,
    EvidencePackage,
    parse_and_validate_answer,
)
from rag.core import Evidence

try:
    from . import audit_disc_law_sft as auditor
    from . import audit_rag_sft_chat_lengths as rag_length_auditor
    from . import audit_sft_chat_lengths as length_auditor
    from . import derive_rag_sft_768 as derivation
    from .sft_dataset import LABEL_MASK_VERSION
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import audit_disc_law_sft as auditor
    from dataset import audit_rag_sft_chat_lengths as rag_length_auditor
    from dataset import audit_sft_chat_lengths as length_auditor
    from dataset import derive_rag_sft_768 as derivation
    from dataset.sft_dataset import LABEL_MASK_VERSION


MAX_SEQ_LEN = 768


class RagSftDatasetError(RuntimeError):
    """固定 768 RAG-SFT Dataset 的输入或标签边界无效。"""


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftDatasetError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise RagSftDatasetError(f"{description}必须是 JSON object")
    return value


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    except Exception as error:
        raise RagSftDatasetError("Tokenizer 编码失败") from error
    if not isinstance(values, list) or any(type(value) is not int for value in values):
        raise RagSftDatasetError("Tokenizer 没有返回一维 input_ids")
    return values


def _subsequence_positions(values: list[int], pattern: list[int]) -> list[int]:
    if not pattern or len(pattern) > len(values):
        return []
    return [
        index
        for index in range(len(values) - len(pattern) + 1)
        if values[index : index + len(pattern)] == pattern
    ]


def _validate_manifest(
    manifest_path: Path,
    tokenizer: Any,
    candidate_path: Path | None = None,
) -> tuple[Path, dict[str, object], list[int], list[int]]:
    try:
        length_auditor._verify_adjacent_hash(manifest_path)
    except length_auditor.ChatLengthAuditError as error:
        raise RagSftDatasetError(str(error)) from error
    manifest = _load_json(manifest_path, "固定 768 RAG-SFT manifest")
    policy = manifest.get("policy")
    readiness = manifest.get("readiness")
    output = manifest.get("output")
    if (
        manifest.get("pipeline") != derivation.PIPELINE
        or manifest.get("complete") is not True
        or not isinstance(policy, dict)
        or policy.get("fixed_max_seq_len") != MAX_SEQ_LEN
        or policy.get("truncation") != "forbidden"
        or not isinstance(readiness, dict)
        or readiness.get("protocol_projection_ready") is not True
        or readiness.get("chat_template_length_audited") is not True
        or readiness.get("overlength_records_isolated") is not True
        or not isinstance(output, dict)
        or set(output.get("record_fields", ()))
        != rag_length_auditor.REQUIRED_RECORD_FIELDS
    ):
        raise RagSftDatasetError("固定 768 RAG-SFT manifest 状态或策略无效")

    tokenizer_report = manifest.get("tokenizer")
    chat_template = getattr(tokenizer, "chat_template", None)
    bos_token = getattr(tokenizer, "bos_token", None)
    eos_token = getattr(tokenizer, "eos_token", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if (
        not isinstance(tokenizer_report, dict)
        or not isinstance(chat_template, str)
        or not isinstance(bos_token, str)
        or not isinstance(eos_token, str)
        or type(pad_token_id) is not int
        or len(tokenizer) != tokenizer_report.get("vocab_size")
        or hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
        != tokenizer_report.get("chat_template_sha256")
    ):
        raise RagSftDatasetError("Tokenizer 与长度审计身份不一致")

    candidate = output.get("candidate")
    if (
        not isinstance(candidate, dict)
        or not isinstance(candidate.get("path"), str)
        or type(candidate.get("records")) is not int
        or type(candidate.get("bytes")) is not int
        or not isinstance(candidate.get("sha256"), str)
    ):
        raise RagSftDatasetError("固定 768 candidate 身份无效")
    resolved_candidate = (
        Path(candidate["path"]).resolve()
        if candidate_path is None
        else Path(candidate_path).resolve()
    )
    if not resolved_candidate.is_file():
        raise RagSftDatasetError(f"固定 768 candidate 不存在: {resolved_candidate}")

    prefix_ids = _token_ids(
        tokenizer,
        f"{bos_token}assistant\n<think>\n\n</think>\n\n",
    )
    end_ids = _token_ids(tokenizer, f"{eos_token}\n")
    if not prefix_ids or not end_ids:
        raise RagSftDatasetError("Tokenizer 无法编码 assistant 标签边界")
    return resolved_candidate, candidate, prefix_ids, end_ids


def _index_candidate(
    path: Path, metadata: dict[str, object]
) -> list[tuple[int, int]]:
    digest = hashlib.sha256()
    offsets: list[tuple[int, int]] = []
    offset = 0
    try:
        with path.open("rb") as source:
            for line_number, raw_line in enumerate(source, start=1):
                offsets.append((offset, line_number))
                digest.update(raw_line)
                offset += len(raw_line)
    except OSError as error:
        raise RagSftDatasetError("无法索引固定 768 candidate") from error
    if (
        len(offsets) != metadata["records"]
        or offset != metadata["bytes"]
        or digest.hexdigest() != metadata["sha256"]
    ):
        raise RagSftDatasetError("固定 768 candidate 身份已变化")
    return offsets


def _validate_record(record: object, line_number: int) -> dict[str, object]:
    if (
        not isinstance(record, dict)
        or set(record) != rag_length_auditor.REQUIRED_RECORD_FIELDS
        or not isinstance(record.get("id"), str)
        or not record["id"]
        or record.get("source") != "rag_sft"
        or record.get("evidence_source") not in rag_length_auditor.EVIDENCE_SOURCES
    ):
        raise RagSftDatasetError(f"固定 768 candidate 记录身份无效: {line_number}")
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 3:
        raise RagSftDatasetError(f"固定 768 conversations 无效: {line_number}")
    if [message.get("role") for message in conversations] != [
        "system",
        "user",
        "assistant",
    ] or any(
        not isinstance(message, dict)
        or set(message) != {"role", "content"}
        or not isinstance(message.get("content"), str)
        or not message["content"]
        for message in conversations
    ):
        raise RagSftDatasetError(f"固定 768 消息结构无效: {line_number}")
    if conversations[0]["content"] != SYSTEM_PROMPT:
        raise RagSftDatasetError(f"system prompt 身份无效: {line_number}")

    try:
        user = rag_length_auditor._load_compact_json(
            conversations[1]["content"], "RAG candidate user"
        )
        evidence_items = user.get("evidence")
        if (
            set(user) != {"query", "evidence"}
            or not isinstance(user.get("query"), str)
            or not user["query"]
            or not isinstance(evidence_items, list)
            or not 1 <= len(evidence_items) <= 4
        ):
            raise RagSftDatasetError(f"user JSON 无效: {line_number}")
        evidence = []
        for index, item in enumerate(evidence_items, start=1):
            if (
                not isinstance(item, dict)
                or set(item)
                != {"evidence_id", "law_name", "article_no", "excerpts"}
                or item.get("evidence_id") != f"E{index}"
                or not isinstance(item.get("law_name"), str)
                or not item["law_name"]
                or not isinstance(item.get("article_no"), str)
                or not item["article_no"]
                or not isinstance(item.get("excerpts"), list)
                or len(item["excerpts"]) != 1
                or not isinstance(item["excerpts"][0], str)
                or not item["excerpts"][0]
            ):
                raise RagSftDatasetError(f"evidence JSON 无效: {line_number}")
            evidence.append(
                Evidence(
                    law_name=item["law_name"],
                    article_no=item["article_no"],
                    content=item["excerpts"][0],
                )
            )
        package = EvidencePackage(query=user["query"], evidence=tuple(evidence))
        rag_length_auditor._load_compact_json(
            conversations[2]["content"], "RAG candidate assistant"
        )
        parse_and_validate_answer(package, conversations[2]["content"])
    except (rag_length_auditor.RagChatLengthAuditError, AnswerProtocolError) as error:
        raise RagSftDatasetError(
            f"固定 768 三字段协议无效: {line_number}"
        ) from error
    return record


class RagSftDataset(Dataset):
    """随机访问固定 768 RAG-SFT，并只监督 assistant JSON 与 EOS。"""

    def __init__(
        self,
        data_manifest: Path,
        tokenizer: Any,
        *,
        candidate_path: Path | None = None,
    ):
        manifest_path = Path(data_manifest).resolve()
        candidate_path, metadata, prefix_ids, end_ids = _validate_manifest(
            manifest_path,
            tokenizer,
            candidate_path,
        )
        self.data_manifest = manifest_path
        self.tokenizer = tokenizer
        self.max_seq_len = MAX_SEQ_LEN
        self.label_mask_version = LABEL_MASK_VERSION
        self._candidate_path = candidate_path
        self._prefix_ids = prefix_ids
        self._end_ids = end_ids
        self._index = _index_candidate(candidate_path, metadata)
        self._handle_value: BinaryIO | None = None
        self._handle_pid: int | None = os.getpid()

    def __len__(self) -> int:
        return len(self._index)

    def _handle(self) -> BinaryIO:
        current_pid = os.getpid()
        if self._handle_pid != current_pid:
            self.close()
            self._handle_pid = current_pid
        if self._handle_value is None or self._handle_value.closed:
            try:
                self._handle_value = self._candidate_path.open("rb")
            except OSError as error:
                raise RagSftDatasetError("无法打开固定 768 candidate") from error
        return self._handle_value

    def _load_record(self, index: int) -> tuple[dict[str, object], int]:
        if index < 0:
            index += len(self._index)
        if index < 0 or index >= len(self._index):
            raise IndexError("RAG-SFT Dataset index 越界")
        offset, line_number = self._index[index]
        handle = self._handle()
        try:
            handle.seek(offset)
            record = json.loads(handle.readline().decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RagSftDatasetError(
                f"固定 768 candidate 无法读取: {line_number}"
            ) from error
        return _validate_record(record, line_number), line_number

    def _encode_record(
        self, record: dict[str, object], line_number: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conversations = record["conversations"]
        assistant_content = conversations[-1]["content"]
        try:
            prompt = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as error:
            raise RagSftDatasetError(
                f"chat template 渲染失败: {line_number}"
            ) from error
        if not isinstance(prompt, str):
            raise RagSftDatasetError(
                f"chat template 没有返回字符串: {line_number}"
            )
        expected_tail = (
            f"{self.tokenizer.bos_token}assistant\n<think>\n\n</think>\n\n"
            f"{assistant_content}{self.tokenizer.eos_token}\n"
        )
        if not prompt.endswith(expected_tail):
            raise RagSftDatasetError(
                f"assistant JSON 被模板改写或空 think 边界变化: {line_number}"
            )

        input_ids = _token_ids(self.tokenizer, prompt)
        if len(input_ids) > MAX_SEQ_LEN:
            raise RagSftDatasetError(
                f"模板后长度超过固定 768，禁止截断: {line_number}"
            )
        prefix_positions = _subsequence_positions(input_ids, self._prefix_ids)
        if len(prefix_positions) != 1:
            raise RagSftDatasetError(
                f"assistant 前缀不能唯一定位: {line_number}"
            )
        answer_start = prefix_positions[0] + len(self._prefix_ids)
        assistant_end = len(input_ids) - len(self._end_ids)
        if assistant_end <= answer_start or input_ids[assistant_end:] != self._end_ids:
            raise RagSftDatasetError(f"assistant 结束标记无效: {line_number}")
        if _subsequence_positions(input_ids[answer_start:assistant_end], self._end_ids):
            raise RagSftDatasetError(
                f"assistant JSON 包含额外结束标记: {line_number}"
            )

        original_length = len(input_ids)
        labels = [-100] * MAX_SEQ_LEN
        labels[answer_start:original_length] = input_ids[answer_start:original_length]
        input_ids.extend(
            [self.tokenizer.pad_token_id] * (MAX_SEQ_LEN - original_length)
        )
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def __getitem__(self, index: int):
        record, line_number = self._load_record(index)
        return self._encode_record(record, line_number)

    def close(self) -> None:
        handle = getattr(self, "_handle_value", None)
        if handle is not None:
            handle.close()
        self._handle_value = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handle_value"] = None
        state["_handle_pid"] = None
        return state

    def __del__(self):
        self.close()


__all__ = [
    "LABEL_MASK_VERSION",
    "MAX_SEQ_LEN",
    "RagSftDataset",
    "RagSftDatasetError",
]
