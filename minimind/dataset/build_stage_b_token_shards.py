"""将阶段 B clean JSONL 构建为可审计的连续 token shards。"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Sequence

import numpy as np

from .audit_pretrain_samples import AuditError, load_tokenizer, sha256_file
from .prepare_stage_b_corpus import (
    EXPECTED_VOCAB_SIZE,
    PreparationError,
    SOURCE_PRIORITY,
    tokenizer_metadata,
    validation_bucket,
)


MINIMIND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORK_ROOT = Path(os.environ.get("MINIMIND_WORK_ROOT", "/root/autodl-tmp/minimind-work"))
DEFAULT_CLEAN_MANIFEST = DEFAULT_WORK_ROOT / "manifests" / "stage-b-clean-v1.json"
DEFAULT_OUTPUT_ROOT = DEFAULT_WORK_ROOT / "data" / "shards"
DEFAULT_MANIFEST_OUTPUT = DEFAULT_WORK_ROOT / "manifests" / "stage-b-shards-v1.json"
DEFAULT_TOKENIZER_PATH = MINIMIND_ROOT / "model"

EXPECTED_BOS_TOKEN_ID = 1
EXPECTED_EOS_TOKEN_ID = 2
EXPECTED_PAD_TOKEN_ID = 0
EXPECTED_UNK_TOKEN_ID = 0
SEQUENCE_LENGTH = 768
SHARD_TOKEN_CAPACITY = 99_999_744
TOKENIZER_BATCH_SIZE = 256
TOKENIZER_CHARACTER_BUDGET = 1_000_000
PROGRESS_INTERVAL = 100_000
STREAM_NAMES = ("train", "full_validation", "quick_validation")
HASH_LINE_RE = re.compile(r"([0-9a-fA-F]{64})  (.+)")
DOCUMENT_ID_RE = re.compile(r"[0-9a-f]{64}")
PART_FILENAME_RE = re.compile(r"part-[0-9]{5}\.jsonl")


class ShardBuildError(RuntimeError):
    """表示 token shard 构建过程无法安全继续。"""


@dataclass(frozen=True)
class CleanRecord:
    """保存一条已经通过结构校验的 clean JSONL 记录。"""

    document_id: str
    text: str


@dataclass(frozen=True)
class SourceInputs:
    """保存一个来源的冻结输入路径和预期文档数。"""

    train_paths: tuple[Path, ...]
    validation_paths: tuple[Path, ...]
    train_documents: int
    full_validation_documents: int
    quick_validation_documents: int


class TokenStreamPacker:
    """把批量 token 追加到固定容量缓冲区并发布完整 sequence 分片。"""

    def __init__(
        self,
        output_root: Path,
        relative_directory: Path,
        sequence_length: int,
        shard_token_capacity: int,
        vocab_size: int,
    ) -> None:
        self.output_root = output_root
        self.directory = output_root / relative_directory
        self.sequence_length = sequence_length
        self.shard_token_capacity = shard_token_capacity
        self.vocab_size = vocab_size
        self.buffer = np.empty(shard_token_capacity, dtype=np.uint16)
        self.buffered_tokens = 0
        self.documents = 0
        self.text_tokens = 0
        self.written_tokens = 0
        self.dropped_tail_tokens = 0
        self.shards: list[dict[str, object]] = []
        self.finished = False

    def append(self, tokens: np.ndarray, documents: int, text_tokens: int) -> None:
        """追加一个已带 BOS/EOS 的批次，并在缓冲区写满时发布 shard。"""
        if self.finished:
            raise ShardBuildError("不能向已经完成的 token 流继续追加数据")
        if tokens.ndim != 1 or tokens.dtype != np.dtype(np.uint16):
            raise ShardBuildError("内部 token 批次必须是一维 uint16")
        if documents < 0 or text_tokens < 0:
            raise ShardBuildError("token 批次统计不能为负数")
        if tokens.size != text_tokens + 2 * documents:
            raise ShardBuildError("token 批次的正文与 BOS/EOS 计数不闭合")

        self.documents += documents
        self.text_tokens += text_tokens
        offset = 0
        while offset < tokens.size:
            available = self.shard_token_capacity - self.buffered_tokens
            copied = min(available, tokens.size - offset)
            self.buffer[self.buffered_tokens : self.buffered_tokens + copied] = tokens[
                offset : offset + copied
            ]
            self.buffered_tokens += copied
            offset += copied
            if self.buffered_tokens == self.shard_token_capacity:
                self._finalize_shard(self.shard_token_capacity)

    def finish(self) -> dict[str, object]:
        """写出最后的完整 sequence，并丢弃不足一个 sequence 的尾部。"""
        if self.finished:
            raise ShardBuildError("token 流不能重复完成")
        aligned_tokens = (self.buffered_tokens // self.sequence_length) * self.sequence_length
        self.dropped_tail_tokens = self.buffered_tokens - aligned_tokens
        if aligned_tokens:
            self._finalize_shard(aligned_tokens)
        else:
            self.buffered_tokens = 0
        self.finished = True

        tokens_before_tail = self.text_tokens + 2 * self.documents
        if self.written_tokens + self.dropped_tail_tokens != tokens_before_tail:
            raise ShardBuildError("token 流的 written/tail 计数不闭合")
        if self.written_tokens % self.sequence_length:
            raise ShardBuildError("token 流写入数量不能被 sequence 长度整除")
        return {
            "documents": self.documents,
            "text_tokens": self.text_tokens,
            "bos_tokens": self.documents,
            "eos_tokens": self.documents,
            "tokens_before_tail": tokens_before_tail,
            "written_tokens": self.written_tokens,
            "dropped_tail_tokens": self.dropped_tail_tokens,
            "sequence_count": self.written_tokens // self.sequence_length,
            "shard_count": len(self.shards),
            "shards": self.shards,
        }

    def _finalize_shard(self, token_count: int) -> None:
        if token_count <= 0 or token_count > self.buffered_tokens:
            raise ShardBuildError("内部 shard token 数无效")
        if token_count % self.sequence_length:
            raise ShardBuildError("shard token 数不能被 sequence 长度整除")

        self.directory.mkdir(parents=True, exist_ok=True)
        partial_path = self.directory / f"part-{len(self.shards):05d}.npy.partial"
        final_path = partial_path.with_suffix("")
        try:
            with partial_path.open("xb") as file:
                np.save(file, self.buffer[:token_count], allow_pickle=False)
            report = self._validate_partial(partial_path, token_count)
            report["sha256"] = sha256_file(partial_path)
            partial_path.replace(final_path)
        except ShardBuildError:
            raise
        except (OSError, ValueError) as error:
            raise ShardBuildError(f"无法写入或复验 token shard: {partial_path}") from error

        report["path"] = final_path.relative_to(self.output_root).as_posix()
        report["bytes"] = final_path.stat().st_size
        self.shards.append(report)
        self.written_tokens += token_count

        remaining = self.buffered_tokens - token_count
        if remaining:
            self.buffer[:remaining] = self.buffer[token_count : self.buffered_tokens]
        self.buffered_tokens = remaining

    def _validate_partial(self, path: Path, expected_tokens: int) -> dict[str, object]:
        """使用 mmap 重开临时 shard，并在发布前验证格式和 token 范围。"""
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise ShardBuildError(f"无法重新加载临时 token shard: {path}") from error
        try:
            if array.ndim != 1:
                raise ShardBuildError(f"token shard 必须是一维数组: {path}")
            if array.dtype != np.dtype(np.uint16):
                raise ShardBuildError(f"token shard dtype 必须为 uint16: {path}")
            if array.size != expected_tokens:
                raise ShardBuildError(f"token shard shape 与写入计数不一致: {path}")
            if array.size % self.sequence_length:
                raise ShardBuildError(f"token shard 不能切成完整 sequence: {path}")
            min_token_id = int(array.min())
            max_token_id = int(array.max())
            if min_token_id < 0 or max_token_id >= self.vocab_size:
                raise ShardBuildError(f"token shard 含词表范围外 ID: {path}")
            return {
                "shape": list(array.shape),
                "token_count": int(array.size),
                "sequence_count": int(array.size // self.sequence_length),
                "min_token_id": min_token_id,
                "max_token_id": max_token_id,
                "dtype": "uint16",
            }
        finally:
            mmap_handle = getattr(array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()


def _require_dict(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ShardBuildError(f"{location} 必须是 JSON object")
    return value


def _require_list(value: object, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ShardBuildError(f"{location} 必须是 JSON array")
    return value


def _require_int(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ShardBuildError(f"{location} 必须是非负整数")
    return value


def ensure_empty_outputs(output_root: Path, manifest_output: Path) -> None:
    """拒绝覆盖已有 shard、临时文件或 shard manifest。"""
    existing_shards: list[Path] = []
    if output_root.exists():
        existing_shards = sorted(
            {
                *output_root.rglob("part-*.npy"),
                *output_root.rglob("*.partial"),
            }
        )
    if existing_shards:
        raise ShardBuildError(f"输出目录已有 token shard 或临时文件: {existing_shards[0]}")

    hash_output = manifest_output.with_suffix(".sha256")
    candidates = (
        manifest_output,
        hash_output,
        manifest_output.with_name(manifest_output.name + ".partial"),
        hash_output.with_name(hash_output.name + ".partial"),
    )
    existing_manifests = [path for path in candidates if path.exists()]
    if existing_manifests:
        raise ShardBuildError(f"输出位置已有 shard manifest: {existing_manifests[0]}")


def verify_clean_manifest(clean_manifest_path: Path) -> tuple[dict[str, Any], dict[str, object]]:
    """验证 clean manifest 的配套哈希并加载严格 JSON。"""
    hash_path = clean_manifest_path.with_suffix(".sha256")
    if not clean_manifest_path.is_file():
        raise ShardBuildError(f"clean manifest 不存在: {clean_manifest_path}")
    if not hash_path.is_file():
        raise ShardBuildError(f"clean manifest 哈希文件不存在: {hash_path}")
    try:
        lines = [
            line
            for line in hash_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError) as error:
        raise ShardBuildError(f"无法读取 clean manifest 哈希文件: {hash_path}") from error
    if len(lines) != 1:
        raise ShardBuildError("clean manifest 哈希文件必须只有一条有效记录")
    match = HASH_LINE_RE.fullmatch(lines[0])
    if match is None:
        raise ShardBuildError("clean manifest 哈希文件格式无效")
    expected_digest, filename = match.groups()
    if filename != clean_manifest_path.name:
        raise ShardBuildError("clean manifest 哈希文件记录了错误的文件名")
    try:
        actual_digest = sha256_file(clean_manifest_path)
    except OSError as error:
        raise ShardBuildError(f"无法读取 clean manifest: {clean_manifest_path}") from error
    if actual_digest.lower() != expected_digest.lower():
        raise ShardBuildError("clean manifest SHA-256 不匹配")
    try:
        manifest = json.loads(clean_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ShardBuildError(f"clean manifest 不是有效 UTF-8 JSON: {clean_manifest_path}") from error
    manifest = _require_dict(manifest, "clean manifest")
    if manifest.get("schema_version") != "1.0":
        raise ShardBuildError("不支持的 clean manifest schema_version")
    return manifest, {
        "path": str(clean_manifest_path),
        "sha256": actual_digest,
        "verified_files": 0,
    }


def _resolve_reported_path(
    clean_root: Path,
    path_text: object,
    split_directory: str,
    source: str,
    expected_index: int,
) -> Path:
    if not isinstance(path_text, str):
        raise ShardBuildError(f"{source}/{split_directory} JSONL path 必须是字符串")
    relative = PurePosixPath(path_text)
    expected_parts = (split_directory, source, f"part-{expected_index:05d}.jsonl")
    if relative.is_absolute() or relative.parts != expected_parts:
        raise ShardBuildError(
            f"clean manifest JSONL 路径不符合固定目录和编号: {path_text}"
        )
    if PART_FILENAME_RE.fullmatch(relative.name) is None:
        raise ShardBuildError(f"clean manifest JSONL 文件名无效: {path_text}")
    path = clean_root.joinpath(*relative.parts).resolve()
    try:
        path.relative_to(clean_root)
    except ValueError as error:
        raise ShardBuildError(f"clean manifest JSONL 路径越出 clean root: {path_text}") from error
    return path


def _verify_file_report(
    report: object,
    clean_root: Path,
    split_directory: str,
    source: str,
    index: int,
) -> tuple[Path, int]:
    report_dict = _require_dict(report, f"{source}/{split_directory} output file")
    path = _resolve_reported_path(
        clean_root,
        report_dict.get("path"),
        split_directory,
        source,
        index,
    )
    records = _require_int(report_dict.get("records"), f"{path} records")
    expected_bytes = _require_int(report_dict.get("bytes"), f"{path} bytes")
    expected_digest = report_dict.get("sha256")
    if not isinstance(expected_digest, str) or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
        raise ShardBuildError(f"{path} SHA-256 格式无效")
    if not path.is_file():
        raise ShardBuildError(f"clean JSONL 文件不存在: {path}")
    try:
        actual_bytes = path.stat().st_size
        if actual_bytes != expected_bytes:
            raise ShardBuildError(f"clean JSONL 文件大小不匹配: {path}")
        if sha256_file(path) != expected_digest:
            raise ShardBuildError(f"clean JSONL SHA-256 不匹配: {path}")
    except ShardBuildError:
        raise
    except OSError as error:
        raise ShardBuildError(f"无法读取 clean JSONL: {path}") from error
    return path, records


def verify_clean_inputs(manifest: dict[str, Any]) -> tuple[Path, dict[str, SourceInputs], int]:
    """验证 clean JSONL 范围、大小、哈希及 manifest 记录数闭合。"""
    output = _require_dict(manifest.get("output"), "clean manifest output")
    root_text = output.get("root")
    if not isinstance(root_text, str) or not root_text:
        raise ShardBuildError("clean manifest output.root 必须是非空字符串")
    if output.get("record_fields") != ["id", "source", "text"]:
        raise ShardBuildError("clean manifest record_fields 与固定 schema 不一致")
    clean_root = Path(root_text).resolve()
    if not clean_root.is_dir():
        raise ShardBuildError(f"clean JSONL 根目录不存在: {clean_root}")

    source_reports = _require_dict(manifest.get("sources"), "clean manifest sources")
    if set(source_reports) != set(SOURCE_PRIORITY):
        raise ShardBuildError("clean manifest 来源范围必须严格为 wikipedia/fineweb/minimind")

    verified_paths: set[Path] = set()
    source_inputs: dict[str, SourceInputs] = {}
    total_expected = {"train": 0, "full_validation": 0, "quick_validation": 0}
    for source in SOURCE_PRIORITY:
        source_report = _require_dict(source_reports[source], f"sources.{source}")
        records = _require_dict(source_report.get("records"), f"sources.{source}.records")
        retained = _require_int(records.get("retained"), f"sources.{source}.records.retained")
        train_documents = _require_int(records.get("train"), f"sources.{source}.records.train")
        full_documents = _require_int(
            records.get("full_validation"),
            f"sources.{source}.records.full_validation",
        )
        quick_documents = _require_int(
            records.get("quick_validation"),
            f"sources.{source}.records.quick_validation",
        )
        if retained != train_documents + full_documents or quick_documents > full_documents:
            raise ShardBuildError(f"{source} clean manifest 记录数不闭合")

        output_files = _require_dict(
            source_report.get("output_files"),
            f"sources.{source}.output_files",
        )
        if set(output_files) != {"train", "validation"}:
            raise ShardBuildError(f"{source} output_files split 范围无效")

        split_results: dict[str, tuple[Path, ...]] = {}
        for manifest_split, directory_split, expected_documents in (
            ("train", "train", train_documents),
            ("validation", "validation", full_documents),
        ):
            reports = _require_list(
                output_files[manifest_split],
                f"sources.{source}.output_files.{manifest_split}",
            )
            paths: list[Path] = []
            reported_records = 0
            for index, report in enumerate(reports):
                path, file_records = _verify_file_report(
                    report,
                    clean_root,
                    directory_split,
                    source,
                    index,
                )
                if path in verified_paths:
                    raise ShardBuildError(f"clean manifest 重复引用 JSONL: {path}")
                verified_paths.add(path)
                paths.append(path)
                reported_records += file_records
            if reported_records != expected_documents:
                raise ShardBuildError(f"{source}/{manifest_split} clean manifest 记录数不闭合")
            split_results[manifest_split] = tuple(paths)

        source_inputs[source] = SourceInputs(
            train_paths=split_results["train"],
            validation_paths=split_results["validation"],
            train_documents=train_documents,
            full_validation_documents=full_documents,
            quick_validation_documents=quick_documents,
        )
        total_expected["train"] += train_documents
        total_expected["full_validation"] += full_documents
        total_expected["quick_validation"] += quick_documents

    partials = sorted(clean_root.rglob("*.partial"))
    if partials:
        raise ShardBuildError(f"clean JSONL 根目录含未完成临时文件: {partials[0]}")
    actual_paths = {path.resolve() for path in clean_root.rglob("part-*.jsonl")}
    if actual_paths != verified_paths:
        missing = sorted(str(path) for path in verified_paths - actual_paths)
        extra = sorted(str(path) for path in actual_paths - verified_paths)
        raise ShardBuildError(
            "clean JSONL 范围不一致，"
            f"缺失={missing[:5]}，额外={extra[:5]}"
        )

    totals = _require_dict(manifest.get("totals"), "clean manifest totals")
    total_records = _require_dict(totals.get("records"), "clean manifest totals.records")
    for key, expected in total_expected.items():
        actual = _require_int(total_records.get(key), f"totals.records.{key}")
        if actual != expected:
            raise ShardBuildError(f"clean manifest totals.records.{key} 不闭合")
    return clean_root, source_inputs, len(verified_paths)


def validate_tokenizer(
    tokenizer: Any,
    tokenizer_path: Path,
    clean_manifest: dict[str, Any],
) -> dict[str, object]:
    """验证当前 Fast Tokenizer 与 clean manifest 中的冻结指纹一致。"""
    if getattr(tokenizer, "is_fast", False) is not True:
        raise ShardBuildError("阶段 B shard 构建必须使用 Fast Tokenizer")
    try:
        current_report, _ = tokenizer_metadata(tokenizer, tokenizer_path)
    except (PreparationError, OSError) as error:
        raise ShardBuildError(str(error)) from error
    clean_report = _require_dict(clean_manifest.get("tokenizer"), "clean manifest tokenizer")
    for key in ("vocab_size", "files", "added_tokens"):
        if clean_report.get(key) != current_report.get(key):
            raise ShardBuildError(f"Tokenizer 指纹不一致: {key}")

    special_ids = {
        "bos_token_id": ("BOS", EXPECTED_BOS_TOKEN_ID),
        "eos_token_id": ("EOS", EXPECTED_EOS_TOKEN_ID),
        "pad_token_id": ("PAD", EXPECTED_PAD_TOKEN_ID),
        "unk_token_id": ("UNK", EXPECTED_UNK_TOKEN_ID),
    }
    report = {
        **current_report,
        "path": str(tokenizer_path),
        "is_fast": True,
    }
    for attribute, (label, expected) in special_ids.items():
        actual = getattr(tokenizer, attribute, None)
        if actual != expected:
            raise ShardBuildError(
                f"Tokenizer {label} token ID 应为 {expected}，实际为 {actual}"
            )
        report[attribute] = int(actual)
    return report


def iter_clean_batches(
    paths: Sequence[Path],
    source: str,
    split: str,
    batch_size: int,
    character_budget: int,
) -> Iterator[list[CleanRecord]]:
    """单次流式读取 clean JSONL，并按文档数或字符数形成编码批次。"""
    batch: list[CleanRecord] = []
    characters = 0
    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    location = f"{path}:{line_number}"
                    if not line.strip():
                        raise ShardBuildError(f"clean JSONL 含空行: {location}")
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ShardBuildError(f"clean JSONL 行不是有效 JSON: {location}") from error
                    record = _require_dict(record, location)
                    if set(record) != {"id", "source", "text"}:
                        raise ShardBuildError(f"clean JSONL 字段必须严格为 id/source/text: {location}")
                    document_id = record["id"]
                    record_source = record["source"]
                    text = record["text"]
                    if not isinstance(document_id, str) or DOCUMENT_ID_RE.fullmatch(document_id) is None:
                        raise ShardBuildError(f"clean JSONL 文档 ID 格式无效: {location}")
                    if record_source != source:
                        raise ShardBuildError(f"clean JSONL 记录 source 不一致: {location}")
                    if not isinstance(text, str) or not text:
                        raise ShardBuildError(f"clean JSONL text 必须是非空字符串: {location}")
                    bucket = validation_bucket(document_id)
                    if split == "train" and bucket < 50:
                        raise ShardBuildError(f"train 记录落入 validation bucket: {location}")
                    if split == "validation" and bucket >= 50:
                        raise ShardBuildError(f"validation 记录落入 train bucket: {location}")

                    if batch and (
                        len(batch) >= batch_size
                        or characters + len(text) > character_budget
                    ):
                        yield batch
                        batch = []
                        characters = 0
                    batch.append(CleanRecord(document_id=document_id, text=text))
                    characters += len(text)
                    if len(batch) >= batch_size or characters >= character_budget:
                        yield batch
                        batch = []
                        characters = 0
        except ShardBuildError:
            raise
        except (OSError, UnicodeDecodeError) as error:
            raise ShardBuildError(f"无法读取 clean JSONL: {path}") from error
    if batch:
        yield batch


def tokenize_texts(tokenizer: Any, records: Sequence[CleanRecord]) -> list[Sequence[int]]:
    """使用固定参数批量编码，并保持与输入记录完全相同的顺序。"""
    texts = [record.text for record in records]
    try:
        result = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = result["input_ids"]
    except Exception as error:
        raise ShardBuildError("Fast Tokenizer 批量编码失败") from error
    if not isinstance(input_ids, (list, tuple)) or len(input_ids) != len(records):
        raise ShardBuildError("Fast Tokenizer 返回的 input_ids 批次长度不一致")
    for index, token_ids in enumerate(input_ids):
        if not isinstance(token_ids, (list, tuple, np.ndarray)):
            raise ShardBuildError(f"Fast Tokenizer 第 {index} 条 input_ids 不是序列")
    return list(input_ids)


def frame_documents(
    encoded_documents: Sequence[Sequence[int]],
    selected_indexes: Sequence[int] | None = None,
) -> tuple[np.ndarray, int, int]:
    """复用编码结果，为选中文档添加 BOS/EOS 并转为 uint16。"""
    indexes: Sequence[int]
    if selected_indexes is None:
        indexes = range(len(encoded_documents))
    else:
        indexes = selected_indexes
    documents = len(indexes)
    text_tokens = sum(len(encoded_documents[index]) for index in indexes)
    framed = np.empty(text_tokens + 2 * documents, dtype=np.int64)
    offset = 0
    try:
        for index in indexes:
            token_ids = encoded_documents[index]
            length = len(token_ids)
            framed[offset] = EXPECTED_BOS_TOKEN_ID
            framed[offset + 1 : offset + 1 + length] = token_ids
            framed[offset + 1 + length] = EXPECTED_EOS_TOKEN_ID
            offset += length + 2
    except (TypeError, ValueError, OverflowError) as error:
        raise ShardBuildError("Fast Tokenizer 返回了无法转换的 token ID") from error
    if framed.size:
        min_token_id = int(framed.min())
        max_token_id = int(framed.max())
        if min_token_id < 0 or max_token_id >= EXPECTED_VOCAB_SIZE:
            raise ShardBuildError(
                "token ID 超出词表范围: "
                f"min={min_token_id}, max={max_token_id}, vocab={EXPECTED_VOCAB_SIZE}"
            )
    return framed.astype(np.uint16, copy=False), documents, text_tokens


def _print_progress(
    source: str,
    split: str,
    packer: TokenStreamPacker,
    started_at: float,
) -> None:
    elapsed = max(time.monotonic() - started_at, 1e-9)
    tokens_before_tail = packer.text_tokens + 2 * packer.documents
    print(
        f"[进度] {source}/{split}: 文档={packer.documents} "
        f"正文tokens={packer.text_tokens} shards={len(packer.shards)} "
        f"written_tokens={packer.written_tokens} tokens/s={tokens_before_tail / elapsed:.0f} "
        f"耗时={elapsed / 60:.1f}分钟",
        file=sys.stderr,
        flush=True,
    )


def process_train_stream(
    source: str,
    inputs: SourceInputs,
    tokenizer: Any,
    output_root: Path,
    sequence_length: int,
    shard_token_capacity: int,
    tokenizer_batch_size: int,
    tokenizer_character_budget: int,
) -> dict[str, object]:
    """编码并打包一个来源的训练流。"""
    packer = TokenStreamPacker(
        output_root,
        Path("train") / source,
        sequence_length,
        shard_token_capacity,
        EXPECTED_VOCAB_SIZE,
    )
    started_at = time.monotonic()
    last_progress = 0
    for records in iter_clean_batches(
        inputs.train_paths,
        source,
        "train",
        tokenizer_batch_size,
        tokenizer_character_budget,
    ):
        encoded = tokenize_texts(tokenizer, records)
        framed, documents, text_tokens = frame_documents(encoded)
        packer.append(framed, documents, text_tokens)
        if packer.documents - last_progress >= PROGRESS_INTERVAL:
            _print_progress(source, "train", packer, started_at)
            last_progress = packer.documents
    if packer.documents != inputs.train_documents:
        raise ShardBuildError(
            f"{source}/train 实际文档数 {packer.documents} "
            f"与 clean manifest {inputs.train_documents} 不一致"
        )
    return packer.finish()


def process_validation_streams(
    source: str,
    inputs: SourceInputs,
    tokenizer: Any,
    output_root: Path,
    sequence_length: int,
    shard_token_capacity: int,
    tokenizer_batch_size: int,
    tokenizer_character_budget: int,
) -> tuple[dict[str, object], dict[str, object]]:
    """单次编码 validation，并把同一结果分别送入 full 和 quick packer。"""
    full_packer = TokenStreamPacker(
        output_root,
        Path("validation") / "full" / source,
        sequence_length,
        shard_token_capacity,
        EXPECTED_VOCAB_SIZE,
    )
    quick_packer = TokenStreamPacker(
        output_root,
        Path("validation") / "quick" / source,
        sequence_length,
        shard_token_capacity,
        EXPECTED_VOCAB_SIZE,
    )
    started_at = time.monotonic()
    last_progress = 0
    for records in iter_clean_batches(
        inputs.validation_paths,
        source,
        "validation",
        tokenizer_batch_size,
        tokenizer_character_budget,
    ):
        encoded = tokenize_texts(tokenizer, records)
        full_tokens, documents, text_tokens = frame_documents(encoded)
        full_packer.append(full_tokens, documents, text_tokens)

        quick_indexes = [
            index
            for index, record in enumerate(records)
            if validation_bucket(record.document_id) < 5
        ]
        if quick_indexes:
            quick_tokens, quick_documents, quick_text_tokens = frame_documents(
                encoded,
                quick_indexes,
            )
            quick_packer.append(quick_tokens, quick_documents, quick_text_tokens)
        if full_packer.documents - last_progress >= PROGRESS_INTERVAL:
            _print_progress(source, "full_validation", full_packer, started_at)
            _print_progress(source, "quick_validation", quick_packer, started_at)
            last_progress = full_packer.documents

    if full_packer.documents != inputs.full_validation_documents:
        raise ShardBuildError(
            f"{source}/full validation 实际文档数 {full_packer.documents} "
            f"与 clean manifest {inputs.full_validation_documents} 不一致"
        )
    if quick_packer.documents != inputs.quick_validation_documents:
        raise ShardBuildError(
            f"{source}/quick validation 实际文档数 {quick_packer.documents} "
            f"与 clean manifest {inputs.quick_validation_documents} 不一致"
        )
    return full_packer.finish(), quick_packer.finish()


def aggregate_streams(
    sources: dict[str, dict[str, dict[str, object]]],
    stream_name: str,
    sequence_length: int,
) -> dict[str, int]:
    """汇总一个 split，并再次验证全来源 token 统计闭合。"""
    keys = (
        "documents",
        "text_tokens",
        "bos_tokens",
        "eos_tokens",
        "tokens_before_tail",
        "written_tokens",
        "dropped_tail_tokens",
        "sequence_count",
        "shard_count",
    )
    report = {
        key: sum(int(sources[source][stream_name][key]) for source in SOURCE_PRIORITY)
        for key in keys
    }
    if report["tokens_before_tail"] != (
        report["text_tokens"] + report["bos_tokens"] + report["eos_tokens"]
    ):
        raise ShardBuildError(f"totals.{stream_name} BOS/EOS 计数不闭合")
    if report["written_tokens"] + report["dropped_tail_tokens"] != report["tokens_before_tail"]:
        raise ShardBuildError(f"totals.{stream_name} written/tail 计数不闭合")
    if report["sequence_count"] != report["written_tokens"] // sequence_length:
        raise ShardBuildError(f"totals.{stream_name} sequence 计数不闭合")
    return report


def write_shard_manifest(
    manifest: dict[str, object],
    manifest_output: Path,
) -> tuple[Path, Path]:
    """校验临时 JSON 后原子发布 shard manifest 及其 SHA-256。"""
    hash_output = manifest_output.with_suffix(".sha256")
    manifest_partial = manifest_output.with_name(manifest_output.name + ".partial")
    hash_partial = hash_output.with_name(hash_output.name + ".partial")
    payload = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    try:
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        with manifest_partial.open("xb") as file:
            file.write(payload)
        json.loads(manifest_partial.read_text(encoding="utf-8"))
        digest = sha256_file(manifest_partial)
        with hash_partial.open("xb") as file:
            file.write(f"{digest}  {manifest_output.name}\n".encode("utf-8"))
        manifest_partial.replace(manifest_output)
        hash_partial.replace(hash_output)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise ShardBuildError(f"无法发布 shard manifest: {manifest_output}") from error
    return manifest_output, hash_output


def build_stage_b_token_shards(
    clean_manifest_path: Path,
    tokenizer_path: Path,
    output_root: Path,
    manifest_output: Path,
    tokenizer: Any | None = None,
    sequence_length: int = SEQUENCE_LENGTH,
    shard_token_capacity: int = SHARD_TOKEN_CAPACITY,
    tokenizer_batch_size: int = TOKENIZER_BATCH_SIZE,
    tokenizer_character_budget: int = TOKENIZER_CHARACTER_BUDGET,
) -> dict[str, object]:
    """验证冻结 clean 语料，并生成确定性的 uint16 NumPy token shards。"""
    clean_manifest_path = clean_manifest_path.resolve()
    tokenizer_path = tokenizer_path.resolve()
    output_root = output_root.resolve()
    manifest_output = manifest_output.resolve()
    if sequence_length <= 0:
        raise ShardBuildError("sequence_length 必须大于零")
    if shard_token_capacity <= 0 or shard_token_capacity % sequence_length:
        raise ShardBuildError("shard_token_capacity 必须是 sequence_length 的正整数倍")
    if tokenizer_batch_size <= 0 or tokenizer_character_budget <= 0:
        raise ShardBuildError("Tokenizer 批次和字符预算必须大于零")

    ensure_empty_outputs(output_root, manifest_output)
    print("[构建] 开始验证 clean manifest 和全部 JSONL 哈希", file=sys.stderr, flush=True)
    clean_manifest, clean_manifest_report = verify_clean_manifest(clean_manifest_path)
    _, source_inputs, verified_files = verify_clean_inputs(clean_manifest)
    clean_manifest_report["verified_files"] = verified_files
    print(f"[构建] clean 输入验证完成: {verified_files} 个 JSONL", file=sys.stderr, flush=True)

    if tokenizer is None:
        try:
            tokenizer = load_tokenizer(tokenizer_path)
        except AuditError as error:
            raise ShardBuildError(str(error)) from error
    tokenizer_report = validate_tokenizer(tokenizer, tokenizer_path, clean_manifest)
    print("[构建] Tokenizer 指纹和特殊 token 验证完成", file=sys.stderr, flush=True)

    source_reports: dict[str, dict[str, dict[str, object]]] = {}
    for source in SOURCE_PRIORITY:
        print(f"[构建] 开始 {source}/train", file=sys.stderr, flush=True)
        train_report = process_train_stream(
            source,
            source_inputs[source],
            tokenizer,
            output_root,
            sequence_length,
            shard_token_capacity,
            tokenizer_batch_size,
            tokenizer_character_budget,
        )
        print(f"[构建] 开始 {source}/validation", file=sys.stderr, flush=True)
        full_report, quick_report = process_validation_streams(
            source,
            source_inputs[source],
            tokenizer,
            output_root,
            sequence_length,
            shard_token_capacity,
            tokenizer_batch_size,
            tokenizer_character_budget,
        )
        source_reports[source] = {
            "train": train_report,
            "full_validation": full_report,
            "quick_validation": quick_report,
        }

    totals = {
        stream: aggregate_streams(source_reports, stream, sequence_length)
        for stream in STREAM_NAMES
    }
    manifest = {
        "schema_version": "1.0",
        "clean_manifest": clean_manifest_report,
        "tokenizer": tokenizer_report,
        "tokenization": {
            "add_special_tokens": False,
            "truncation": False,
            "padding": False,
            "document_framing": "[BOS] + text_tokens + [EOS]",
            "batch_size": tokenizer_batch_size,
            "character_budget": tokenizer_character_budget,
        },
        "packing": {
            "mode": "continuous_per_source_and_split",
            "sequence_length": sequence_length,
            "shard_token_capacity": shard_token_capacity,
            "tail_rule": "drop_tokens_shorter_than_one_sequence",
            "cross_source_packing": False,
            "quick_validation_rule": "int(id[:16], 16) % 10000 < 5",
        },
        "output": {
            "root": str(output_root),
            "format": "numpy_npy",
            "dtype": "uint16",
            "dimensions": 1,
        },
        "sources": source_reports,
        "totals": totals,
    }
    manifest_path, manifest_hash_path = write_shard_manifest(manifest, manifest_output)
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256_path": str(manifest_hash_path),
    }


def parse_args() -> argparse.Namespace:
    """解析云端正式 shard 构建所需的路径参数。"""
    parser = argparse.ArgumentParser(description="生成阶段 B token shards 和可审计 manifest")
    parser.add_argument("--clean-manifest", type=Path, default=DEFAULT_CLEAN_MANIFEST)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """运行正式 token shard 构建并打印三个 split 的最终 token 数。"""
    args = parse_args()
    try:
        result = build_stage_b_token_shards(
            clean_manifest_path=args.clean_manifest,
            tokenizer_path=args.tokenizer_path,
            output_root=args.output_root,
            manifest_output=args.manifest_output,
        )
    except ShardBuildError as error:
        print(f"[失败] {error}", file=sys.stderr)
        raise SystemExit(1) from error

    for split in STREAM_NAMES:
        report = result["totals"][split]
        print(
            f"{split}: documents={report['documents']} "
            f"written_tokens={report['written_tokens']} "
            f"sequences={report['sequence_count']} shards={report['shard_count']}"
        )
    print(f"shard manifest: {result['manifest_path']}")
    print(f"manifest 哈希: {result['manifest_sha256_path']}")


if __name__ == "__main__":
    main()
