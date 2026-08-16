"""为固定 768 的法律 SFT 数据生成确定性 input_ids 和 labels。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, BinaryIO

import torch
from torch.utils.data import Dataset

try:
    from . import audit_disc_law_sft as auditor
    from . import audit_sft_chat_lengths as length_auditor
    from . import derive_disc_law_sft_768 as derivation
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import audit_disc_law_sft as auditor
    from dataset import audit_sft_chat_lengths as length_auditor
    from dataset import derive_disc_law_sft_768 as derivation


MAX_SEQ_LEN = 768
LABEL_MASK_VERSION = "assistant_answer_only_v1"
ALLOWED_SPLITS = ("train", "validation/full", "validation/quick")
DATASET_KINDS = ("pair_qa", "triplet_qa")


class SftDatasetError(RuntimeError):
    """表示法律 SFT Dataset 的输入身份或标签边界无效。"""


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SftDatasetError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise SftDatasetError(f"{description}必须是 JSON object: {path}")
    return value


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if not isinstance(input_ids, list) or any(type(value) is not int for value in input_ids):
        raise SftDatasetError("Tokenizer 没有返回一维 input_ids")
    return input_ids


def _subsequence_positions(values: list[int], pattern: list[int]) -> list[int]:
    if not pattern or len(pattern) > len(values):
        return []
    return [
        index
        for index in range(len(values) - len(pattern) + 1)
        if values[index : index + len(pattern)] == pattern
    ]


def _validate_tokenizer(
    tokenizer: Any, audit_report: dict[str, Any]
) -> tuple[list[int], list[int]]:
    tokenizer_report = audit_report.get("tokenizer")
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
    ):
        raise SftDatasetError("Tokenizer 或长度审计中的模板身份不完整")
    template_sha256 = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
    if (
        len(tokenizer) != tokenizer_report.get("vocab_size")
        or template_sha256 != tokenizer_report.get("chat_template_sha256")
    ):
        raise SftDatasetError("Tokenizer 词表或 chat template 与长度审计不一致")

    assistant_prefix = f"{bos_token}assistant\n<think>\n\n</think>\n\n"
    assistant_end = f"{eos_token}\n"
    prefix_ids = _token_ids(tokenizer, assistant_prefix)
    end_ids = _token_ids(tokenizer, assistant_end)
    if not prefix_ids or not end_ids:
        raise SftDatasetError("Tokenizer 无法编码 assistant 标签边界")
    return prefix_ids, end_ids


def _validate_manifest(
    manifest_path: Path,
    tokenizer: Any,
    split: str,
    dataset_kinds: tuple[str, ...],
) -> tuple[dict[str, Any], list[tuple[str, Path, dict[str, object]]], list[int], list[int]]:
    try:
        length_auditor._verify_adjacent_hash(manifest_path)
    except length_auditor.ChatLengthAuditError as error:
        raise SftDatasetError(str(error)) from error
    manifest = _load_json(manifest_path, "固定 768 data manifest")
    policy = manifest.get("policy")
    readiness = manifest.get("readiness")
    validation = manifest.get("validation")
    output = manifest.get("output")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("pipeline") != derivation.PIPELINE
        or manifest.get("complete") is not True
        or not isinstance(policy, dict)
        or policy.get("fixed_max_seq_len") != MAX_SEQ_LEN
        or policy.get("truncation") != "forbidden"
        or not isinstance(readiness, dict)
        or readiness.get("standardization_complete") is not True
        or readiness.get("chat_template_length_audited") is not True
        or readiness.get("overlength_records_isolated") is not True
        or not isinstance(validation, dict)
        or validation.get("quick_is_subset_of_filtered_full") is not True
        or not isinstance(output, dict)
        or not isinstance(output.get("root"), str)
        or not isinstance(output.get("files"), dict)
    ):
        raise SftDatasetError("固定 768 data manifest 状态或策略无效")
    if set(output["files"]) != set(length_auditor.EXPECTED_FILE_KEYS):
        raise SftDatasetError("固定 768 data manifest 的六文件范围无效")
    record_fields = output.get("record_fields")
    if (
        not isinstance(record_fields, list)
        or len(record_fields) != len(length_auditor.OUTPUT_FIELDS)
        or any(not isinstance(field, str) for field in record_fields)
        or set(record_fields) != length_auditor.OUTPUT_FIELDS
    ):
        raise SftDatasetError("固定 768 data manifest 的记录字段范围无效")

    audit = manifest.get("length_audit")
    if not isinstance(audit, dict) or audit.get("fixed_max_seq_len") != MAX_SEQ_LEN:
        raise SftDatasetError("固定 768 data manifest 缺少长度审计身份")
    report_meta = audit.get("report")
    if (
        not isinstance(report_meta, dict)
        or not isinstance(report_meta.get("path"), str)
        or type(report_meta.get("bytes")) is not int
        or not isinstance(report_meta.get("sha256"), str)
    ):
        raise SftDatasetError("长度审计报告元数据无效")
    report_path = Path(report_meta["path"]).resolve()
    if (
        not report_path.is_file()
        or report_path.stat().st_size != report_meta["bytes"]
        or auditor.sha256_file(report_path) != report_meta["sha256"]
    ):
        raise SftDatasetError("长度审计报告身份已变化")
    audit_report = _load_json(report_path, "长度审计报告")
    scope = audit_report.get("scope")
    if (
        audit_report.get("schema_version") != "1.0"
        or audit_report.get("pipeline") != "legal_sft_chat_length_audit_768_v1"
        or audit_report.get("complete") is not True
        or not isinstance(scope, dict)
        or scope.get("fixed_max_seq_len") != MAX_SEQ_LEN
        or audit_report.get("input", {}).get("data_manifest", {}).get("sha256")
        != manifest.get("parent_data_manifest", {}).get("sha256")
    ):
        raise SftDatasetError("长度审计报告与派生 data manifest 不闭合")
    prefix_ids, end_ids = _validate_tokenizer(tokenizer, audit_report)

    output_root = Path(output["root"]).resolve()
    selected: list[tuple[str, Path, dict[str, object]]] = []
    for dataset_kind in dataset_kinds:
        key = f"{split}/{dataset_kind}"
        metadata = output["files"][key]
        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("path"), str)
            or type(metadata.get("records")) is not int
            or type(metadata.get("bytes")) is not int
            or not isinstance(metadata.get("sha256"), str)
        ):
            raise SftDatasetError(f"派生 JSONL 元数据无效: {key}")
        path = (output_root / metadata["path"]).resolve()
        if not auditor._path_is_within(path, output_root) or not path.is_file():
            raise SftDatasetError(f"派生 JSONL 不存在或越出输出根目录: {key}")
        selected.append((key, path, metadata))
    return manifest, selected, prefix_ids, end_ids


def _index_file(path: Path, key: str, metadata: dict[str, object]) -> list[tuple[int, int]]:
    digest = hashlib.sha256()
    offsets: list[tuple[int, int]] = []
    offset = 0
    try:
        with path.open("rb") as source:
            for line_number, raw_line in enumerate(source, 1):
                offsets.append((offset, line_number))
                digest.update(raw_line)
                offset += len(raw_line)
    except OSError as error:
        raise SftDatasetError(f"无法索引派生 JSONL: {key}") from error
    if (
        len(offsets) != metadata["records"]
        or offset != metadata["bytes"]
        or digest.hexdigest() != metadata["sha256"]
    ):
        raise SftDatasetError(f"派生 JSONL 身份已变化: {key}")
    return offsets


class SftDataset(Dataset):
    """从固定 768 manifest 随机访问法律 SFT 记录，并生成方案 B labels。"""

    def __init__(
        self,
        data_manifest: Path,
        split: str,
        tokenizer: Any,
        dataset_kind: str | None = None,
    ):
        if split not in ALLOWED_SPLITS:
            raise SftDatasetError(f"SFT split 无效: {split}")
        if dataset_kind is not None and dataset_kind not in DATASET_KINDS:
            raise SftDatasetError(f"SFT dataset_kind 无效: {dataset_kind}")
        dataset_kinds = DATASET_KINDS if dataset_kind is None else (dataset_kind,)
        manifest_path = Path(data_manifest).resolve()
        _, selected, prefix_ids, end_ids = _validate_manifest(
            manifest_path, tokenizer, split, dataset_kinds
        )

        self.data_manifest = manifest_path
        self.split = split
        self.dataset_kind = dataset_kind
        self.tokenizer = tokenizer
        self.max_seq_len = MAX_SEQ_LEN
        self.label_mask_version = LABEL_MASK_VERSION
        self._prefix_ids = prefix_ids
        self._end_ids = end_ids
        self._file_keys = [key for key, _, _ in selected]
        self._paths = [path for _, path, _ in selected]
        self._index: list[tuple[int, int, int]] = []
        for file_index, (key, path, metadata) in enumerate(selected):
            for offset, line_number in _index_file(path, key, metadata):
                self._index.append((file_index, offset, line_number))
        self._handles: dict[int, BinaryIO] = {}
        self._handle_pid: int | None = os.getpid()

    def __len__(self) -> int:
        return len(self._index)

    def _handle(self, file_index: int) -> BinaryIO:
        current_pid = os.getpid()
        if self._handle_pid != current_pid:
            self.close()
            self._handle_pid = current_pid
        handle = self._handles.get(file_index)
        if handle is None or handle.closed:
            try:
                handle = self._paths[file_index].open("rb")
            except OSError as error:
                raise SftDatasetError(
                    f"无法打开派生 JSONL: {self._file_keys[file_index]}"
                ) from error
            self._handles[file_index] = handle
        return handle

    def _load_record(self, index: int) -> tuple[dict[str, object], str, int]:
        if index < 0:
            index += len(self._index)
        if index < 0 or index >= len(self._index):
            raise IndexError("SFT Dataset index 越界")
        file_index, offset, line_number = self._index[index]
        handle = self._handle(file_index)
        try:
            handle.seek(offset)
            raw_line = handle.readline()
            parsed = json.loads(raw_line.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SftDatasetError(
                f"派生 JSONL 记录无法读取: {self._file_keys[file_index]}:{line_number}"
            ) from error
        key = self._file_keys[file_index]
        try:
            record = length_auditor._validate_record(parsed, key, line_number)
        except length_auditor.ChatLengthAuditError as error:
            raise SftDatasetError(str(error)) from error
        return record, key, line_number

    def _encode_record(
        self, record: dict[str, object], key: str, line_number: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        conversations = record["conversations"]
        assistant_content = conversations[1]["content"]
        try:
            prompt = self.tokenizer.apply_chat_template(
                conversations,
                tokenize=False,
                add_generation_prompt=False,
            )
        except (TypeError, ValueError, KeyError) as error:
            raise SftDatasetError(f"chat template 渲染失败: {key}:{line_number}") from error
        if not isinstance(prompt, str):
            raise SftDatasetError(f"chat template 没有返回字符串: {key}:{line_number}")
        expected_tail = (
            f"{self.tokenizer.bos_token}assistant\n<think>\n\n</think>\n\n"
            f"{assistant_content}{self.tokenizer.eos_token}\n"
        )
        if not prompt.endswith(expected_tail):
            raise SftDatasetError(
                f"assistant content 被模板改写或空 think 边界变化: {key}:{line_number}"
            )

        input_ids = _token_ids(self.tokenizer, prompt)
        if len(input_ids) > MAX_SEQ_LEN:
            raise SftDatasetError(
                f"模板后长度超过固定 768，禁止截断: {key}:{line_number}"
            )
        prefix_positions = _subsequence_positions(input_ids, self._prefix_ids)
        if len(prefix_positions) != 1:
            raise SftDatasetError(f"assistant 前缀不能唯一定位: {key}:{line_number}")
        answer_start = prefix_positions[0] + len(self._prefix_ids)
        assistant_end = len(input_ids) - len(self._end_ids)
        if assistant_end < answer_start or input_ids[assistant_end:] != self._end_ids:
            raise SftDatasetError(f"assistant 结束标记无效: {key}:{line_number}")
        if _subsequence_positions(input_ids[answer_start:assistant_end], self._end_ids):
            raise SftDatasetError(f"assistant 回答包含额外结束标记: {key}:{line_number}")
        if assistant_end == answer_start:
            raise SftDatasetError(f"assistant 回答没有有效 token: {key}:{line_number}")

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
        record, key, line_number = self._load_record(index)
        return self._encode_record(record, key, line_number)

    def close(self) -> None:
        handles = getattr(self, "_handles", {})
        for handle in handles.values():
            handle.close()
        handles.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_handle_pid"] = None
        return state

    def __del__(self):
        self.close()
