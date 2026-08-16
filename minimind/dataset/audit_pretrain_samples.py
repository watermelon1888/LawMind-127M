"""流式审计阶段 B 当前已下载的通用预训练数据样本。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Sequence


MINIMIND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORK_ROOT = Path(os.environ.get("MINIMIND_WORK_ROOT", "/root/autodl-tmp/minimind-work"))
DEFAULT_DATA_ROOT = DEFAULT_WORK_ROOT / "data" / "raw" / "stage-b"
DEFAULT_RAW_MANIFEST = DEFAULT_WORK_ROOT / "manifests" / "stage-b-raw-samples.sha256"
DEFAULT_OUTPUT_DIR = DEFAULT_WORK_ROOT / "manifests" / "audit"
DEFAULT_TOKENIZER_PATH = MINIMIND_ROOT / "model"

EXPECTED_VOCAB_SIZE = 12000
PARQUET_BATCH_SIZE = 1024
TOKENIZER_BATCH_SIZE = 256
TOKENIZER_CHARACTER_BUDGET = 1_000_000
PROGRESS_INTERVAL = 100_000

FINEWEB_REQUIRED_FIELDS = ("text", "score", "source")
WIKIPEDIA_REQUIRED_FIELDS = ("id", "url", "title", "text")

CJK_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")
NON_WHITESPACE_RE = re.compile(r"\S")
HTML_RE = re.compile(r"<[A-Za-z/][^>\n]{0,500}>")
URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
MANIFEST_LINE_RE = re.compile(r"^([0-9a-fA-F]{64}) [ *](.+)$")

TextConverter = Callable[[str], str]


class AuditError(RuntimeError):
    """表示审计无法继续的硬错误。"""


@dataclass
class SourceStats:
    """保存单个来源的流式统计状态，不保留完整正文。"""

    name: str
    files: list[Path]
    total_records: int = 0
    valid_records: int = 0
    parse_errors: int = 0
    empty_texts: int = 0
    required_field_missing_values: int = 0
    text_utf8_bytes: int = 0
    characters: int = 0
    tokens: int = 0
    unk_token_id_occurrences: int = 0
    cjk_characters: int = 0
    non_whitespace_characters: int = 0
    low_cjk_ratio_documents: int = 0
    normalized_exact_duplicate_records: int = 0
    character_lengths: list[int] = field(default_factory=list)
    token_lengths: list[int] = field(default_factory=list)
    signatures: set[bytes] = field(default_factory=set)
    anomalies: Counter[str] = field(default_factory=Counter)
    source_fields: Counter[str] = field(default_factory=Counter)
    source_metrics: dict[str, object] = field(default_factory=dict)
    skipped_examples: list[dict[str, object]] = field(default_factory=list)
    last_progress_records: int = 0

    def record_skipped(self, location: str, reason: str) -> None:
        """只保留少量位置和原因，避免把原始正文复制进报告。"""
        if len(self.skipped_examples) < 5:
            self.skipped_examples.append({"location": location, "reason": reason})

    def record_text(
        self,
        text: str,
        token_ids: Sequence[int],
        unk_token_id: int | None,
        special_tokens: tuple[str, ...],
    ) -> None:
        """记录一条有效审计文本的最小指标。"""
        character_length = len(text)
        token_length = len(token_ids)
        cjk_characters = len(CJK_RE.findall(text))
        non_whitespace_characters = len(NON_WHITESPACE_RE.findall(text))

        self.valid_records += 1
        self.text_utf8_bytes += len(text.encode("utf-8"))
        self.characters += character_length
        self.tokens += token_length
        self.cjk_characters += cjk_characters
        self.non_whitespace_characters += non_whitespace_characters
        self.character_lengths.append(character_length)
        self.token_lengths.append(token_length)

        if non_whitespace_characters and cjk_characters / non_whitespace_characters < 0.30:
            self.low_cjk_ratio_documents += 1
        if unk_token_id is not None:
            self.unk_token_id_occurrences += sum(token_id == unk_token_id for token_id in token_ids)

        if "\ufffd" in text:
            self.anomalies["replacement_character_documents"] += 1
        if "\x00" in text:
            self.anomalies["nul_documents"] += 1
        if special_tokens and any(token in text for token in special_tokens):
            self.anomalies["reserved_special_token_documents"] += 1
        if HTML_RE.search(text):
            self.anomalies["html_documents"] += 1
        if URL_RE.search(text):
            self.anomalies["url_documents"] += 1

        signature = normalized_text_signature(text)
        if signature in self.signatures:
            self.normalized_exact_duplicate_records += 1
        else:
            self.signatures.add(signature)

    def to_report(self) -> dict[str, object]:
        """转换为不含内部哈希集合和长度列表的 JSON 对象。"""
        return {
            "files": [str(path) for path in self.files],
            "file_count": len(self.files),
            "records": {
                "total": self.total_records,
                "valid": self.valid_records,
                "parse_errors": self.parse_errors,
                "empty_texts": self.empty_texts,
                "required_field_missing_values": self.required_field_missing_values,
                "skipped_examples": self.skipped_examples,
            },
            "size": {
                "text_utf8_bytes": self.text_utf8_bytes,
                "characters": self.characters,
                "tokens": self.tokens,
            },
            "lengths": {
                "characters": length_summary(self.character_lengths),
                "tokens": length_summary(self.token_lengths),
            },
            "tokenizer": {
                "tokens_per_character": safe_ratio(self.tokens, self.characters),
                "unk_token_id_occurrences": self.unk_token_id_occurrences,
            },
            "language": {
                "cjk_characters": self.cjk_characters,
                "non_whitespace_characters": self.non_whitespace_characters,
                "cjk_ratio": safe_ratio(self.cjk_characters, self.non_whitespace_characters),
                "documents_below_30_percent_cjk": self.low_cjk_ratio_documents,
            },
            "anomalies": anomaly_report(self.anomalies),
            "duplicates": {
                "normalized_exact_duplicate_records": self.normalized_exact_duplicate_records,
                "unique_normalized_signatures": len(self.signatures),
            },
            "source_fields": {
                **dict(sorted(self.source_fields.items())),
                **self.source_metrics,
            },
        }


def safe_ratio(numerator: int, denominator: int) -> float:
    """返回稳定的比率，空分母统一记为零。"""
    return numerator / denominator if denominator else 0.0


def nearest_rank(values: list[int], quantile: float) -> int:
    """使用最近秩定义计算离散分位数。"""
    if not values:
        return 0
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def length_summary(values: list[int]) -> dict[str, int]:
    """返回规格要求的四个长度统计。"""
    return {
        "p50": nearest_rank(values, 0.50),
        "p95": nearest_rank(values, 0.95),
        "p99": nearest_rank(values, 0.99),
        "max": max(values, default=0),
    }


def numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    """返回来源数值字段的数量和必要分位数。"""
    if not values:
        return {"count": 0, "min": None, "p50": None, "p95": None, "p99": None, "max": None}
    ordered = sorted(values)

    def at(quantile: float) -> float:
        rank = max(1, math.ceil(quantile * len(ordered)))
        return ordered[rank - 1]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": ordered[-1],
    }


def anomaly_report(counter: Counter[str]) -> dict[str, int]:
    """确保没有命中的异常项也以零写入报告。"""
    keys = (
        "replacement_character_documents",
        "nul_documents",
        "reserved_special_token_documents",
        "html_documents",
        "url_documents",
    )
    return {key: counter[key] for key in keys}


def normalized_text_signature(text: str) -> bytes:
    """生成仅用于完全重复检查的保守规范化签名。"""
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = WHITESPACE_RE.sub(" ", normalized).strip()
    return hashlib.sha256(normalized.encode("utf-8")).digest()


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_raw_manifest(manifest_path: Path) -> tuple[dict[str, object], set[Path]]:
    """校验原始样本清单中的每个文件并返回已验证路径。"""
    if not manifest_path.is_file():
        raise AuditError(f"原始样本 SHA-256 清单不存在: {manifest_path}")

    verified_paths: set[Path] = set()
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        match = MANIFEST_LINE_RE.fullmatch(line)
        if match is None:
            raise AuditError(f"SHA-256 清单第 {line_number} 行格式无效")
        expected_digest, path_text = match.groups()
        path = Path(path_text)
        if not path.is_absolute():
            path = manifest_path.parent / path
        path = path.resolve()
        if not path.is_file():
            raise AuditError(f"SHA-256 清单中的文件不存在: {path}")
        actual_digest = sha256_file(path)
        if actual_digest.lower() != expected_digest.lower():
            raise AuditError(f"SHA-256 不匹配: {path}")
        verified_paths.add(path)

    if not verified_paths:
        raise AuditError(f"SHA-256 清单中没有有效文件: {manifest_path}")
    return (
        {
            "path": str(manifest_path.resolve()),
            "sha256": sha256_file(manifest_path),
            "verified_files": len(verified_paths),
        },
        verified_paths,
    )


def discover_input_files(data_root: Path) -> dict[str, list[Path]]:
    """按约定目录发现三个来源的当前样本。"""
    sources = {
        "fineweb": sorted((data_root / "fineweb_edu_chinese_v2_1" / "4_5").glob("*.parquet")),
        "wikipedia": sorted((data_root / "wikipedia_20231101_zh").glob("*.parquet")),
        "minimind": [data_root / "minimind_pretrain" / "pretrain_t2t_mini.jsonl"],
    }
    for source, paths in sources.items():
        if not paths or any(not path.is_file() for path in paths):
            raise AuditError(f"{source} 当前样本文件不存在或不完整: {paths}")
    return sources


def validate_manifest_scope(source_files: dict[str, list[Path]], verified_paths: set[Path]) -> None:
    """确保参与审计的文件与已校验的原始样本清单完全一致。"""
    audit_paths = {path.resolve() for paths in source_files.values() for path in paths}
    if audit_paths == verified_paths:
        return
    missing = sorted(str(path) for path in audit_paths - verified_paths)
    extra = sorted(str(path) for path in verified_paths - audit_paths)
    raise AuditError(
        "审计输入与 SHA-256 清单范围不一致，"
        f"未校验输入={missing[:5]}，清单额外文件={extra[:5]}"
    )


def load_tokenizer(tokenizer_path: Path) -> Any:
    """延迟加载 Transformers，避免单元测试依赖真实模型目录。"""
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise AuditError("缺少 transformers，无法加载阶段 A Tokenizer") from error
    try:
        return AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True, local_files_only=True)
    except Exception as error:
        raise AuditError(f"无法加载 Tokenizer: {tokenizer_path}") from error


def load_t2s_converter() -> tuple[TextConverter, str]:
    """加载 OpenCC t2s 转换器并记录实现版本。"""
    try:
        from opencc import OpenCC
    except ImportError as error:
        raise AuditError(
            "缺少 OpenCC，请安装 opencc-python-reimplemented 后重试"
        ) from error
    try:
        package_version = version("opencc-python-reimplemented")
    except PackageNotFoundError:
        package_version = "unknown"
    converter = OpenCC("t2s")
    return converter.convert, package_version


def valid_text(
    stats: SourceStats,
    value: object,
    location: str,
    converter: TextConverter | None = None,
) -> str | None:
    """校验 text 字段并按来源执行审计转换。"""
    if not isinstance(value, str):
        stats.required_field_missing_values += 1
        stats.record_skipped(location, "text 字段缺失或不是字符串")
        return None
    if not value.strip():
        stats.empty_texts += 1
        stats.record_skipped(location, "text 字段为空")
        return None
    if converter is None:
        return value
    try:
        converted = converter(value)
    except Exception as error:
        raise AuditError(f"Wikipedia t2s 转换失败: {location}") from error
    if not converted.strip():
        stats.empty_texts += 1
        stats.record_skipped(location, "t2s 后文本为空")
        return None
    return converted


def tokenize_chunk(
    stats: SourceStats,
    texts: list[str],
    tokenizer: Any,
    special_tokens: tuple[str, ...],
) -> None:
    """批量 tokenize 一组文本并写入来源统计。"""
    try:
        encoded = tokenizer(
            texts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = encoded["input_ids"]
    except Exception as error:
        raise AuditError(f"{stats.name} 批量 tokenize 失败") from error
    if len(input_ids) != len(texts):
        raise AuditError(f"{stats.name} Tokenizer 返回数量与输入不一致")

    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    for text, token_ids in zip(texts, input_ids):
        stats.record_text(text, token_ids, unk_token_id, special_tokens)


def tokenize_texts(stats: SourceStats, texts: list[str], tokenizer: Any) -> None:
    """同时按文档数和字符预算切分 Tokenizer 批次。"""
    special_tokens = tuple(
        token for token in getattr(tokenizer, "all_special_tokens", ()) if isinstance(token, str) and token
    )
    pending: list[str] = []
    pending_characters = 0
    for text in texts:
        exceeds_budget = pending and pending_characters + len(text) > TOKENIZER_CHARACTER_BUDGET
        if pending and (len(pending) >= TOKENIZER_BATCH_SIZE or exceeds_budget):
            tokenize_chunk(stats, pending, tokenizer, special_tokens)
            pending = []
            pending_characters = 0
        pending.append(text)
        pending_characters += len(text)
    if pending:
        tokenize_chunk(stats, pending, tokenizer, special_tokens)


def is_missing(value: object) -> bool:
    """判断来源元数据字段是否缺失。"""
    return value is None or isinstance(value, str) and not value.strip()


def audit_fineweb(paths: list[Path], tokenizer: Any) -> SourceStats:
    """流式扫描 Fineweb-Edu 当前 Parquet 样本。"""
    stats = SourceStats("fineweb", paths)
    for key in ("score_non_numeric", "score_below_0_8", "source_empty"):
        stats.source_fields[key] = 0
    scores: list[float] = []
    for path in paths:
        print(f"[审计] Fineweb: {path}", file=sys.stderr, flush=True)
        for columns, start_index in iter_parquet_batches(path, FINEWEB_REQUIRED_FIELDS):
            texts: list[str] = []
            for offset, text_value in enumerate(columns["text"]):
                stats.total_records += 1
                location = f"{path}:{start_index + offset}"
                score = columns["score"][offset]
                source = columns["source"][offset]

                if is_missing(score):
                    stats.required_field_missing_values += 1
                if not isinstance(score, Real) or isinstance(score, bool) or not math.isfinite(float(score)):
                    stats.source_fields["score_non_numeric"] += 1
                else:
                    numeric_score = float(score)
                    scores.append(numeric_score)
                    if numeric_score < 0.8:
                        stats.source_fields["score_below_0_8"] += 1
                if is_missing(source):
                    stats.required_field_missing_values += 1
                    stats.source_fields["source_empty"] += 1

                text = valid_text(stats, text_value, location)
                if text is not None:
                    texts.append(text)
            tokenize_texts(stats, texts, tokenizer)
            print_progress(stats)
    stats.source_metrics["score_summary"] = numeric_summary(scores)
    return stats


def audit_wikipedia(paths: list[Path], tokenizer: Any, converter: TextConverter) -> SourceStats:
    """流式扫描并转简 Wikipedia 当前 Parquet 样本。"""
    stats = SourceStats("wikipedia", paths)
    for field_name in ("id", "url", "title"):
        stats.source_fields[f"{field_name}_missing"] = 0
    for path in paths:
        print(f"[审计] Wikipedia: {path}", file=sys.stderr, flush=True)
        for columns, start_index in iter_parquet_batches(path, WIKIPEDIA_REQUIRED_FIELDS):
            texts: list[str] = []
            for offset, text_value in enumerate(columns["text"]):
                stats.total_records += 1
                location = f"{path}:{start_index + offset}"
                for field_name in ("id", "url", "title"):
                    if is_missing(columns[field_name][offset]):
                        stats.required_field_missing_values += 1
                        stats.source_fields[f"{field_name}_missing"] += 1
                text = valid_text(stats, text_value, location, converter)
                if text is not None:
                    texts.append(text)
            tokenize_texts(stats, texts, tokenizer)
            print_progress(stats)
    return stats


def iter_parquet_batches(path: Path, required_fields: tuple[str, ...]):
    """读取 Parquet 并在扫描前验证列结构。"""
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise AuditError("缺少 pyarrow，无法读取 Parquet 样本") from error
    try:
        parquet_file = parquet.ParquetFile(path)
        missing_fields = sorted(set(required_fields) - set(parquet_file.schema_arrow.names))
        if missing_fields:
            raise AuditError(f"Parquet 缺少必需字段 {missing_fields}: {path}")
        start_index = 1
        for batch in parquet_file.iter_batches(batch_size=PARQUET_BATCH_SIZE, columns=required_fields):
            columns = batch.to_pydict()
            yield columns, start_index
            start_index += batch.num_rows
    except AuditError:
        raise
    except Exception as error:
        raise AuditError(f"无法读取 Parquet: {path}") from error


def audit_minimind(path: Path, tokenizer: Any) -> SourceStats:
    """逐行扫描完整的 MiniMind mini JSONL 文件。"""
    stats = SourceStats("minimind", [path])
    texts: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                stats.total_records += 1
                location = f"{path}:{line_number}"
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats.parse_errors += 1
                    stats.record_skipped(location, "JSON 解析失败")
                    continue
                if not isinstance(record, dict):
                    stats.required_field_missing_values += 1
                    stats.record_skipped(location, "记录不是 JSON 对象")
                    continue
                text = valid_text(stats, record.get("text"), location)
                if text is not None:
                    texts.append(text)
                if len(texts) >= TOKENIZER_BATCH_SIZE:
                    tokenize_texts(stats, texts, tokenizer)
                    texts = []
                print_progress(stats)
    except UnicodeDecodeError as error:
        raise AuditError(f"MiniMind JSONL 不是有效 UTF-8: {path}") from error
    except OSError as error:
        raise AuditError(f"无法读取 MiniMind JSONL: {path}") from error
    if texts:
        tokenize_texts(stats, texts, tokenizer)
    return stats


def print_progress(stats: SourceStats) -> None:
    """每处理固定数量记录输出一次进度。"""
    if stats.total_records - stats.last_progress_records >= PROGRESS_INTERVAL:
        print(f"[进度] {stats.name}: {stats.total_records} 条", file=sys.stderr, flush=True)
        stats.last_progress_records = stats.total_records


def cross_source_overlaps(stats_by_source: dict[str, SourceStats]) -> dict[str, int]:
    """计算来源唯一签名集合的两两交集，不依赖扫描顺序。"""
    pairs = (
        ("fineweb", "wikipedia"),
        ("fineweb", "minimind"),
        ("wikipedia", "minimind"),
    )
    overlaps: dict[str, int] = {}
    for left_name, right_name in pairs:
        left = stats_by_source[left_name].signatures
        right = stats_by_source[right_name].signatures
        smaller, larger = (left, right) if len(left) <= len(right) else (right, left)
        overlaps[f"{left_name}__{right_name}"] = sum(signature in larger for signature in smaller)
    return overlaps


def totals_report(stats_by_source: dict[str, SourceStats], overlaps: dict[str, int]) -> dict[str, object]:
    """汇总三个来源的基础统计，不扣除重复 token。"""
    stats = list(stats_by_source.values())
    characters = sum(item.characters for item in stats)
    tokens = sum(item.tokens for item in stats)
    cjk_characters = sum(item.cjk_characters for item in stats)
    non_whitespace_characters = sum(item.non_whitespace_characters for item in stats)
    return {
        "records": {
            "total": sum(item.total_records for item in stats),
            "valid": sum(item.valid_records for item in stats),
            "parse_errors": sum(item.parse_errors for item in stats),
            "empty_texts": sum(item.empty_texts for item in stats),
            "required_field_missing_values": sum(item.required_field_missing_values for item in stats),
        },
        "size": {
            "text_utf8_bytes": sum(item.text_utf8_bytes for item in stats),
            "characters": characters,
            "tokens": tokens,
        },
        "tokenizer": {
            "tokens_per_character": safe_ratio(tokens, characters),
            "unk_token_id_occurrences": sum(item.unk_token_id_occurrences for item in stats),
        },
        "language": {
            "cjk_characters": cjk_characters,
            "non_whitespace_characters": non_whitespace_characters,
            "cjk_ratio": safe_ratio(cjk_characters, non_whitespace_characters),
            "documents_below_30_percent_cjk": sum(item.low_cjk_ratio_documents for item in stats),
        },
        "anomalies": {
            key: sum(item.anomalies[key] for item in stats)
            for key in anomaly_report(Counter())
        },
        "duplicates": {
            "normalized_exact_duplicate_records_within_sources": sum(
                item.normalized_exact_duplicate_records for item in stats
            ),
            "cross_source_shared_signature_pairs": sum(overlaps.values()),
        },
    }


def write_report(report: dict[str, object], output_dir: Path) -> tuple[Path, Path]:
    """写出严格 JSON，重新解析后生成报告哈希。"""
    report_path = output_dir / "stage-b-sample-audit.json"
    hash_path = output_dir / "stage-b-sample-audit.sha256"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        json.loads(report_path.read_text(encoding="utf-8"))
        hash_path.write_text(f"{sha256_file(report_path)}  {report_path.name}\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise AuditError(f"审计报告无法写入或重新解析: {report_path}") from error
    return report_path, hash_path


def audit_samples(
    data_root: Path,
    tokenizer_path: Path,
    raw_manifest_path: Path,
    output_dir: Path,
    tokenizer: Any | None = None,
    wikipedia_converter: TextConverter | None = None,
    opencc_version: str | None = None,
) -> dict[str, object]:
    """执行阶段 B 当前样本的完整自动审计。"""
    data_root = data_root.resolve()
    tokenizer_path = tokenizer_path.resolve()
    raw_manifest_path = raw_manifest_path.resolve()
    output_dir = output_dir.resolve()

    raw_manifest, verified_paths = verify_raw_manifest(raw_manifest_path)
    source_files = discover_input_files(data_root)
    validate_manifest_scope(source_files, verified_paths)

    tokenizer = tokenizer or load_tokenizer(tokenizer_path)
    if len(tokenizer) != EXPECTED_VOCAB_SIZE:
        raise AuditError(f"Tokenizer 词表大小应为 {EXPECTED_VOCAB_SIZE}，实际为 {len(tokenizer)}")
    if wikipedia_converter is None:
        wikipedia_converter, opencc_version = load_t2s_converter()

    stats_by_source = {
        "fineweb": audit_fineweb(source_files["fineweb"], tokenizer),
        "wikipedia": audit_wikipedia(source_files["wikipedia"], tokenizer, wikipedia_converter),
        "minimind": audit_minimind(source_files["minimind"][0], tokenizer),
    }
    for source, stats in stats_by_source.items():
        if not stats.valid_records:
            raise AuditError(f"{source} 没有任何有效记录")

    overlaps = cross_source_overlaps(stats_by_source)
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": {
            "path": str(tokenizer_path),
            "vocab_size": len(tokenizer),
        },
        "raw_manifest": raw_manifest,
        "wikipedia_normalization": {
            "opencc_config": "t2s",
            "opencc_version": opencc_version or "unknown",
            "prior_audit_path": str(
                raw_manifest_path.parent / "wikipedia-2-traditional-audit.json"
            ),
        },
        "sources": {
            source: stats.to_report()
            for source, stats in stats_by_source.items()
        },
        "cross_source_exact_duplicates": overlaps,
        "totals": totals_report(stats_by_source, overlaps),
    }
    report_path, hash_path = write_report(report, output_dir)
    report["report_path"] = str(report_path)
    report["report_sha256_path"] = str(hash_path)
    return report


def parse_args() -> argparse.Namespace:
    """解析云端审计所需的最小路径参数。"""
    parser = argparse.ArgumentParser(description="流式审计阶段 B 当前通用预训练数据样本")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--raw-manifest", type=Path, default=DEFAULT_RAW_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    """运行审计并打印各来源的有效 token 数。"""
    args = parse_args()
    try:
        report = audit_samples(
            data_root=args.data_root,
            tokenizer_path=args.tokenizer_path,
            raw_manifest_path=args.raw_manifest,
            output_dir=args.output_dir,
        )
    except AuditError as error:
        print(f"[失败] {error}", file=sys.stderr)
        raise SystemExit(1) from error

    for source, source_report in report["sources"].items():
        print(f"{source}: {source_report['size']['tokens']} tokens")
    print(f"审计报告: {report['report_path']}")
    print(f"报告哈希: {report['report_sha256_path']}")


if __name__ == "__main__":
    main()
