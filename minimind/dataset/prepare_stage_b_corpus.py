"""将阶段 B 冻结的原始数据清洗为可审计的 JSONL 语料。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from .audit_pretrain_samples import (
    AuditError,
    FINEWEB_REQUIRED_FIELDS,
    WIKIPEDIA_REQUIRED_FIELDS,
    discover_input_files,
    load_t2s_converter,
    load_tokenizer,
    sha256_file,
    validate_manifest_scope,
    verify_raw_manifest,
)


MINIMIND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORK_ROOT = Path(os.environ.get("MINIMIND_WORK_ROOT", "/root/autodl-tmp/minimind-work"))
DEFAULT_DATA_ROOT = DEFAULT_WORK_ROOT / "data" / "raw" / "stage-b"
DEFAULT_RAW_MANIFEST = DEFAULT_WORK_ROOT / "manifests" / "stage-b-raw-snapshot-002.sha256"
DEFAULT_OUTPUT_ROOT = DEFAULT_WORK_ROOT / "data" / "clean"
DEFAULT_MANIFEST_OUTPUT = DEFAULT_WORK_ROOT / "manifests" / "stage-b-clean-v1.json"
DEFAULT_TOKENIZER_PATH = MINIMIND_ROOT / "model"

EXPECTED_VOCAB_SIZE = 12000
EXPECTED_ADDED_TOKEN_COUNT = 36
TARGET_FILE_BYTES = 256 * 1024 * 1024
PARQUET_BATCH_SIZE = 1024
PROGRESS_INTERVAL = 100_000
SOURCE_PRIORITY = ("wikipedia", "fineweb", "minimind")
TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)
FILTER_KEYS = (
    "json_parse_error",
    "invalid_text",
    "empty_text",
    "nul_character",
    "replacement_character",
    "reserved_added_token",
)
WHITESPACE_RE = re.compile(r"\s+")

TextConverter = Callable[[str], str]


class PreparationError(RuntimeError):
    """表示正式清洗无法继续的硬错误。"""


@dataclass(frozen=True)
class RawRecord:
    """保存来源读取器产生的一条原始记录。"""

    path: Path
    index: int
    text: object
    error_reason: str | None = None


@dataclass
class SplitStats:
    """统计保留在一个数据划分中的正文规模。"""

    records: int = 0
    characters: int = 0
    text_utf8_bytes: int = 0

    def record(self, text: str) -> None:
        self.records += 1
        self.characters += len(text)
        self.text_utf8_bytes += len(text.encode("utf-8"))

    def to_report(self) -> dict[str, int]:
        return {
            "records": self.records,
            "characters": self.characters,
            "text_utf8_bytes": self.text_utf8_bytes,
        }


@dataclass
class SourceStats:
    """保存单个来源的互斥清洗计数。"""

    name: str
    input_file_count: int
    scanned: int = 0
    retained: int = 0
    train: SplitStats = field(default_factory=SplitStats)
    full_validation: SplitStats = field(default_factory=SplitStats)
    quick_validation: SplitStats = field(default_factory=SplitStats)
    filtered: Counter[str] = field(default_factory=Counter)
    duplicate_within_source: int = 0
    duplicate_higher_priority_source: int = 0
    output_files: dict[str, list[dict[str, object]]] = field(
        default_factory=lambda: {"train": [], "validation": []}
    )
    last_progress_records: int = 0

    def record_retained(self, text: str, split: str, is_quick: bool) -> None:
        self.retained += 1
        if split == "train":
            self.train.record(text)
            return
        self.full_validation.record(text)
        if is_quick:
            self.quick_validation.record(text)

    def to_report(self) -> dict[str, object]:
        return {
            "input_file_count": self.input_file_count,
            "records": {
                "scanned": self.scanned,
                "retained": self.retained,
                "train": self.train.records,
                "full_validation": self.full_validation.records,
                "quick_validation": self.quick_validation.records,
                "filtered": {key: self.filtered[key] for key in FILTER_KEYS},
                "duplicates": {
                    "within_source": self.duplicate_within_source,
                    "higher_priority_source": self.duplicate_higher_priority_source,
                },
            },
            "size": {
                "retained": {
                    "characters": self.train.characters + self.full_validation.characters,
                    "text_utf8_bytes": (
                        self.train.text_utf8_bytes + self.full_validation.text_utf8_bytes
                    ),
                },
                "train": self.train.to_report(),
                "full_validation": self.full_validation.to_report(),
                "quick_validation": self.quick_validation.to_report(),
            },
            "output_files": self.output_files,
        }


class JsonlShardWriter:
    """按完整 JSONL 记录轮换文件，并在关闭后校验其哈希。"""

    def __init__(
        self,
        output_root: Path,
        split: str,
        source: str,
        target_file_bytes: int,
    ) -> None:
        self.output_root = output_root
        self.directory = output_root / split / source
        self.target_file_bytes = target_file_bytes
        self.file_index = 0
        self.file = None
        self.partial_path: Path | None = None
        self.current_bytes = 0
        self.current_records = 0
        self.total_bytes = 0
        self.reports: list[dict[str, object]] = []

    def write(self, record: dict[str, str]) -> None:
        line = (
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        if self.file is not None and self.current_bytes + len(line) > self.target_file_bytes:
            self._finalize_current()
        if self.file is None:
            self._open_next()
        self.file.write(line)
        self.current_bytes += len(line)
        self.current_records += 1
        self.total_bytes += len(line)

    def finish(self) -> list[dict[str, object]]:
        if self.file is not None:
            self._finalize_current()
        return self.reports

    def abort(self) -> None:
        if self.file is not None:
            self.file.close()
            self.file = None

    def _open_next(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        filename = f"part-{self.file_index:05d}.jsonl.partial"
        self.partial_path = self.directory / filename
        try:
            self.file = self.partial_path.open("xb")
        except OSError as error:
            raise PreparationError(f"无法创建清洗分片: {self.partial_path}") from error
        self.current_bytes = 0
        self.current_records = 0

    def _finalize_current(self) -> None:
        if self.file is None or self.partial_path is None:
            return
        partial_path = self.partial_path
        try:
            self.file.close()
            self.file = None
            digest = sha256_file(partial_path)
            byte_count = partial_path.stat().st_size
            if byte_count != self.current_bytes:
                raise PreparationError(f"清洗分片写入字节数不一致: {partial_path}")
            final_path = partial_path.with_suffix("")
            partial_path.replace(final_path)
        except PreparationError:
            raise
        except OSError as error:
            raise PreparationError(f"无法完成清洗分片: {partial_path}") from error

        self.reports.append(
            {
                "path": final_path.relative_to(self.output_root).as_posix(),
                "records": self.current_records,
                "bytes": byte_count,
                "sha256": digest,
            }
        )
        self.file_index += 1
        self.partial_path = None
        self.current_bytes = 0
        self.current_records = 0


def validation_bucket(document_id: str) -> int:
    """使用文档 ID 的前 64 位计算固定的万分桶。"""
    return int(document_id[:16], 16) % 10_000


def document_signature(text: str) -> bytes:
    """生成折叠内部空白后的规范化正文 SHA-256。"""
    dedup_text = WHITESPACE_RE.sub(" ", text).strip()
    return hashlib.sha256(dedup_text.encode("utf-8")).digest()


def normalize_and_filter_text(
    value: object,
    converter: TextConverter | None,
    reserved_tokens: tuple[str, ...],
    location: str,
) -> tuple[str | None, str | None]:
    """按固定顺序标准化正文，并返回唯一的首个过滤原因。"""
    if not isinstance(value, str):
        return None, "invalid_text"
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    if converter is not None:
        try:
            text = converter(text)
        except Exception as error:
            raise PreparationError(f"Wikipedia t2s 转换失败: {location}") from error
        if not isinstance(text, str):
            raise PreparationError(f"Wikipedia t2s 返回值不是字符串: {location}")
    text = unicodedata.normalize("NFC", text).strip()
    if not text:
        return None, "empty_text"
    if "\x00" in text:
        return None, "nul_character"
    if "\ufffd" in text:
        return None, "replacement_character"
    if "<" in text and any(token in text for token in reserved_tokens):
        return None, "reserved_added_token"
    return text, None


def iter_parquet_records(
    paths: list[Path],
    required_fields: tuple[str, ...],
) -> Iterator[RawRecord]:
    """验证 Parquet Schema，并只流式读取正文列。"""
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise PreparationError("缺少 pyarrow，无法读取阶段 B Parquet") from error

    for path in paths:
        print(f"[清洗] 读取 {path}", file=sys.stderr, flush=True)
        try:
            parquet_file = parquet.ParquetFile(path)
            missing_fields = sorted(set(required_fields) - set(parquet_file.schema_arrow.names))
            if missing_fields:
                raise PreparationError(f"Parquet 缺少必需字段 {missing_fields}: {path}")
            start_index = 1
            for batch in parquet_file.iter_batches(
                batch_size=PARQUET_BATCH_SIZE,
                columns=("text",),
            ):
                for offset, value in enumerate(batch.column(0).to_pylist()):
                    yield RawRecord(path=path, index=start_index + offset, text=value)
                start_index += batch.num_rows
        except PreparationError:
            raise
        except Exception as error:
            raise PreparationError(f"无法读取 Parquet: {path}") from error


def iter_minimind_records(path: Path) -> Iterator[RawRecord]:
    """逐行读取 MiniMind JSONL，并把坏 JSON 作为可过滤记录返回。"""
    print(f"[清洗] 读取 {path}", file=sys.stderr, flush=True)
    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    yield RawRecord(path, line_number, None, "json_parse_error")
                    continue
                text = record.get("text") if isinstance(record, dict) else None
                yield RawRecord(path, line_number, text)
    except UnicodeDecodeError as error:
        raise PreparationError(f"MiniMind JSONL 不是有效 UTF-8: {path}") from error
    except OSError as error:
        raise PreparationError(f"无法读取 MiniMind JSONL: {path}") from error


def source_records(source: str, paths: list[Path]) -> Iterator[RawRecord]:
    """根据来源选择固定的流式读取器。"""
    if source == "wikipedia":
        return iter_parquet_records(paths, WIKIPEDIA_REQUIRED_FIELDS)
    if source == "fineweb":
        return iter_parquet_records(paths, FINEWEB_REQUIRED_FIELDS)
    return iter_minimind_records(paths[0])


def tokenizer_metadata(
    tokenizer: Any,
    tokenizer_path: Path,
) -> tuple[dict[str, object], tuple[str, ...]]:
    """校验固定 Tokenizer，并返回可复现指纹和 added-token 字面量。"""
    if len(tokenizer) != EXPECTED_VOCAB_SIZE:
        raise PreparationError(
            f"Tokenizer 词表大小应为 {EXPECTED_VOCAB_SIZE}，实际为 {len(tokenizer)}"
        )
    decoder = getattr(tokenizer, "added_tokens_decoder", None)
    if not isinstance(decoder, dict):
        raise PreparationError("Tokenizer 缺少 added_tokens_decoder")
    added_tokens = [
        {"id": int(token_id), "content": str(token)}
        for token_id, token in sorted(decoder.items())
    ]
    reserved_tokens = tuple(item["content"] for item in added_tokens if item["content"])
    if len(reserved_tokens) != EXPECTED_ADDED_TOKEN_COUNT or len(set(reserved_tokens)) != len(
        reserved_tokens
    ):
        raise PreparationError(
            "Tokenizer added-token 数量应为 "
            f"{EXPECTED_ADDED_TOKEN_COUNT}，实际为 {len(reserved_tokens)}"
        )

    file_hashes = {}
    for filename in TOKENIZER_FILENAMES:
        path = tokenizer_path / filename
        if not path.is_file():
            raise PreparationError(f"Tokenizer 指纹文件不存在: {path}")
        file_hashes[filename] = sha256_file(path)
    return (
        {
            "path": str(tokenizer_path),
            "vocab_size": len(tokenizer),
            "files": file_hashes,
            "added_tokens": {
                "count": len(added_tokens),
                "items": added_tokens,
            },
        },
        reserved_tokens,
    )


def ensure_empty_outputs(output_root: Path, manifest_output: Path) -> None:
    """拒绝覆盖任何已有正式分片、临时分片或 manifest。"""
    existing_shards = []
    if output_root.exists():
        existing_shards = sorted(
            {
                *output_root.rglob("part-*.jsonl"),
                *output_root.rglob("*.partial"),
            }
        )
    if existing_shards:
        raise PreparationError(f"输出目录已有清洗分片: {existing_shards[0]}")
    manifest_hash_path = manifest_output.with_suffix(".sha256")
    manifest_partials = (
        manifest_output.with_name(manifest_output.name + ".partial"),
        manifest_hash_path.with_name(manifest_hash_path.name + ".partial"),
    )
    existing_manifests = [
        path
        for path in (manifest_output, manifest_hash_path, *manifest_partials)
        if path.exists()
    ]
    if existing_manifests:
        raise PreparationError(f"输出位置已有 clean manifest: {existing_manifests[0]}")


def prepare_inputs(
    data_root: Path,
    raw_manifest_path: Path,
) -> tuple[dict[str, object], dict[str, list[Path]]]:
    """复用审计模块的快照校验，并统一转换错误类型。"""
    try:
        raw_manifest, verified_paths = verify_raw_manifest(raw_manifest_path)
        source_files = discover_input_files(data_root)
        validate_manifest_scope(source_files, verified_paths)
    except AuditError as error:
        raise PreparationError(str(error)) from error
    return raw_manifest, source_files


def process_source(
    source: str,
    paths: list[Path],
    output_root: Path,
    target_file_bytes: int,
    reserved_tokens: tuple[str, ...],
    converter: TextConverter | None,
    higher_priority_signatures: set[bytes],
    started_at: float,
) -> tuple[SourceStats, set[bytes]]:
    """清洗单个来源，同时隐藏过滤、去重、划分和分片细节。"""
    stats = SourceStats(source, len(paths))
    source_signatures: set[bytes] = set()
    writers = {
        "train": JsonlShardWriter(output_root, "train", source, target_file_bytes),
        "validation": JsonlShardWriter(
            output_root,
            "validation",
            source,
            target_file_bytes,
        ),
    }
    try:
        for raw_record in source_records(source, paths):
            stats.scanned += 1
            if raw_record.error_reason is not None:
                stats.filtered[raw_record.error_reason] += 1
                print_progress(stats, raw_record.path, writers, started_at)
                continue

            location = f"{raw_record.path}:{raw_record.index}"
            text, filter_reason = normalize_and_filter_text(
                raw_record.text,
                converter,
                reserved_tokens,
                location,
            )
            if filter_reason is not None:
                stats.filtered[filter_reason] += 1
                print_progress(stats, raw_record.path, writers, started_at)
                continue

            signature = document_signature(text)
            if signature in source_signatures:
                stats.duplicate_within_source += 1
                print_progress(stats, raw_record.path, writers, started_at)
                continue
            if signature in higher_priority_signatures:
                stats.duplicate_higher_priority_source += 1
                print_progress(stats, raw_record.path, writers, started_at)
                continue

            source_signatures.add(signature)
            document_id = signature.hex()
            bucket = validation_bucket(document_id)
            split = "validation" if bucket < 50 else "train"
            is_quick = bucket < 5
            writers[split].write({"id": document_id, "source": source, "text": text})
            stats.record_retained(text, split, is_quick)
            print_progress(stats, raw_record.path, writers, started_at)

        stats.output_files = {
            split: writer.finish()
            for split, writer in writers.items()
        }
    except Exception:
        for writer in writers.values():
            writer.abort()
        raise
    if not stats.retained:
        raise PreparationError(f"{source} 清洗后没有任何保留记录")
    validate_source_stats(stats)
    return stats, source_signatures


def print_progress(
    stats: SourceStats,
    path: Path,
    writers: dict[str, JsonlShardWriter],
    started_at: float,
) -> None:
    """按固定记录间隔输出可观测的流式进度。"""
    if stats.scanned - stats.last_progress_records < PROGRESS_INTERVAL:
        return
    filtered = sum(stats.filtered.values())
    duplicates = stats.duplicate_within_source + stats.duplicate_higher_priority_source
    output_gib = sum(writer.total_bytes for writer in writers.values()) / (1024**3)
    elapsed = time.monotonic() - started_at
    print(
        f"[进度] {stats.name}: 文件={path.name} 扫描={stats.scanned} "
        f"保留={stats.retained} 过滤={filtered} 去重={duplicates} "
        f"输出={output_gib:.2f}GiB 耗时={elapsed / 60:.1f}分钟",
        file=sys.stderr,
        flush=True,
    )
    stats.last_progress_records = stats.scanned


def validate_source_stats(stats: SourceStats) -> None:
    """验证互斥计数、划分数量和输出记录数严格闭合。"""
    removed = (
        sum(stats.filtered.values())
        + stats.duplicate_within_source
        + stats.duplicate_higher_priority_source
    )
    if stats.scanned != stats.retained + removed:
        raise PreparationError(f"{stats.name} 清洗记录计数不闭合")
    if stats.retained != stats.train.records + stats.full_validation.records:
        raise PreparationError(f"{stats.name} train/validation 计数不闭合")
    if stats.quick_validation.records > stats.full_validation.records:
        raise PreparationError(f"{stats.name} quick validation 超出 full validation")
    output_train = sum(item["records"] for item in stats.output_files["train"])
    output_validation = sum(item["records"] for item in stats.output_files["validation"])
    if output_train != stats.train.records or output_validation != stats.full_validation.records:
        raise PreparationError(f"{stats.name} 输出文件记录数不闭合")


def totals_report(stats_by_source: dict[str, SourceStats]) -> dict[str, object]:
    """汇总来源统计，并保留过滤原因的互斥口径。"""
    stats = list(stats_by_source.values())
    return {
        "records": {
            "scanned": sum(item.scanned for item in stats),
            "retained": sum(item.retained for item in stats),
            "train": sum(item.train.records for item in stats),
            "full_validation": sum(item.full_validation.records for item in stats),
            "quick_validation": sum(item.quick_validation.records for item in stats),
            "filtered": {
                key: sum(item.filtered[key] for item in stats)
                for key in FILTER_KEYS
            },
            "duplicates": {
                "within_source": sum(item.duplicate_within_source for item in stats),
                "higher_priority_source": sum(
                    item.duplicate_higher_priority_source for item in stats
                ),
            },
        },
        "size": {
            "characters": sum(
                item.train.characters + item.full_validation.characters
                for item in stats
            ),
            "text_utf8_bytes": sum(
                item.train.text_utf8_bytes + item.full_validation.text_utf8_bytes
                for item in stats
            ),
            "jsonl_bytes": sum(
                file_report["bytes"]
                for item in stats
                for reports in item.output_files.values()
                for file_report in reports
            ),
        },
    }


def write_manifest(
    manifest: dict[str, object],
    manifest_output: Path,
) -> tuple[Path, Path]:
    """先校验临时 manifest，再原子发布 manifest 及其哈希。"""
    hash_output = manifest_output.with_suffix(".sha256")
    manifest_partial = manifest_output.with_name(manifest_output.name + ".partial")
    hash_partial = hash_output.with_name(hash_output.name + ".partial")
    payload = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    try:
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        manifest_partial.write_bytes(payload)
        json.loads(manifest_partial.read_text(encoding="utf-8"))
        digest = sha256_file(manifest_partial)
        hash_partial.write_bytes(f"{digest}  {manifest_output.name}\n".encode("utf-8"))
        manifest_partial.replace(manifest_output)
        hash_partial.replace(hash_output)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise PreparationError(f"无法发布 clean manifest: {manifest_output}") from error
    return manifest_output, hash_output


def prepare_stage_b_corpus(
    data_root: Path,
    tokenizer_path: Path,
    raw_manifest_path: Path,
    output_root: Path,
    manifest_output: Path,
    tokenizer: Any | None = None,
    wikipedia_converter: TextConverter | None = None,
    opencc_version: str | None = None,
    target_file_bytes: int = TARGET_FILE_BYTES,
) -> dict[str, object]:
    """验证冻结输入，并生成 clean JSONL 与确定性 manifest。"""
    data_root = data_root.resolve()
    tokenizer_path = tokenizer_path.resolve()
    raw_manifest_path = raw_manifest_path.resolve()
    output_root = output_root.resolve()
    manifest_output = manifest_output.resolve()
    if target_file_bytes <= 0:
        raise PreparationError("JSONL 分片目标字节数必须大于零")

    ensure_empty_outputs(output_root, manifest_output)
    raw_manifest, source_files = prepare_inputs(data_root, raw_manifest_path)
    if tokenizer is None:
        try:
            tokenizer = load_tokenizer(tokenizer_path)
        except AuditError as error:
            raise PreparationError(str(error)) from error
    tokenizer_report, reserved_tokens = tokenizer_metadata(tokenizer, tokenizer_path)
    if wikipedia_converter is None:
        try:
            wikipedia_converter, opencc_version = load_t2s_converter()
        except AuditError as error:
            raise PreparationError(str(error)) from error

    started_at = time.monotonic()
    higher_priority_signatures: set[bytes] = set()
    stats_by_source: dict[str, SourceStats] = {}
    for source in SOURCE_PRIORITY:
        converter = wikipedia_converter if source == "wikipedia" else None
        stats, source_signatures = process_source(
            source=source,
            paths=source_files[source],
            output_root=output_root,
            target_file_bytes=target_file_bytes,
            reserved_tokens=reserved_tokens,
            converter=converter,
            higher_priority_signatures=higher_priority_signatures,
            started_at=started_at,
        )
        stats_by_source[source] = stats
        higher_priority_signatures.update(source_signatures)

    manifest = {
        "schema_version": "1.0",
        "raw_manifest": raw_manifest,
        "tokenizer": tokenizer_report,
        "wikipedia_normalization": {
            "opencc_config": "t2s",
            "opencc_version": opencc_version or "unknown",
        },
        "cleaning": {
            "source_priority": list(SOURCE_PRIORITY),
            "normalization_order": [
                "line_endings_to_lf",
                "unicode_nfc",
                "wikipedia_opencc_t2s",
                "unicode_nfc",
                "strip_outer_whitespace",
            ],
            "filter_order": list(FILTER_KEYS),
            "filter_accounting": "first_match_exclusive",
            "retained_without_heuristic_filtering": [
                "html",
                "url",
                "low_cjk_ratio",
                "fineweb_score_variation",
            ],
            "deduplication": {
                "algorithm": "sha256",
                "canonicalization": "collapse_whitespace_to_ascii_space",
                "scope": "global",
                "near_duplicate_removal": False,
            },
        },
        "validation": {
            "bucket_expression": "int(id[:16], 16) % 10000",
            "full_validation_buckets": "0-49",
            "quick_validation_buckets": "0-4",
        },
        "output": {
            "root": str(output_root),
            "target_file_bytes": target_file_bytes,
            "encoding": "utf-8",
            "newline": "lf",
            "record_fields": ["id", "source", "text"],
        },
        "sources": {
            source: stats_by_source[source].to_report()
            for source in SOURCE_PRIORITY
        },
        "totals": totals_report(stats_by_source),
    }
    manifest_path, manifest_hash_path = write_manifest(manifest, manifest_output)
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256_path": str(manifest_hash_path),
    }


def parse_args() -> argparse.Namespace:
    """解析云端正式清洗所需的最小路径参数。"""
    parser = argparse.ArgumentParser(description="正式清洗阶段 B 通用预训练数据")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--raw-manifest", type=Path, default=DEFAULT_RAW_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    return parser.parse_args()


def main() -> None:
    """运行正式清洗并打印最终记录数和 manifest 路径。"""
    args = parse_args()
    try:
        result = prepare_stage_b_corpus(
            data_root=args.data_root,
            tokenizer_path=args.tokenizer_path,
            raw_manifest_path=args.raw_manifest,
            output_root=args.output_root,
            manifest_output=args.manifest_output,
        )
    except PreparationError as error:
        print(f"[失败] {error}", file=sys.stderr)
        raise SystemExit(1) from error

    for source, report in result["sources"].items():
        records = report["records"]
        print(
            f"{source}: 扫描={records['scanned']} 保留={records['retained']} "
            f"train={records['train']} validation={records['full_validation']}"
        )
    print(f"clean manifest: {result['manifest_path']}")
    print(f"manifest 哈希: {result['manifest_sha256_path']}")


if __name__ == "__main__":
    main()
