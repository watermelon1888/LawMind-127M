"""在固定 768 tokens 下只读审计法律 SFT chat template 长度。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from . import audit_disc_law_sft as auditor
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import audit_disc_law_sft as auditor


DEFAULT_WORK_ROOT = Path(os.environ.get("MINIMIND_WORK_ROOT", "/root/autodl-tmp/minimind-work"))
DEFAULT_DATA_MANIFEST = (
    DEFAULT_WORK_ROOT / "manifests" / "disc-law-sft-v1-project-rag-only-provisional.json"
)
DEFAULT_TOKENIZER_PATH = auditor.DEFAULT_TOKENIZER_PATH
DEFAULT_OUTPUT_DIR = (
    DEFAULT_WORK_ROOT / "reports" / "sft" / "disc_law_sft_chat_length_audit_768_v1"
)

MAX_SEQ_LEN = 768
EXPECTED_VOCAB_SIZE = 12_000
TOKENIZER_BATCH_SIZE = 256
TOKENIZER_CHARACTER_BUDGET = 1_000_000
PROGRESS_INTERVAL = 10_000

REPORT_FILENAME = "sft-chat-length-audit-768.json"
OVERFLOW_FILENAME = "sft-chat-length-overflow-768.jsonl"
HASH_FILENAME = "sft-chat-length-audit-768.sha256"

UNIQUE_FILE_KEYS = (
    "train/pair_qa",
    "train/triplet_qa",
    "validation/full/pair_qa",
    "validation/full/triplet_qa",
)
EXPECTED_FILE_KEYS = UNIQUE_FILE_KEYS + (
    "validation/quick/pair_qa",
    "validation/quick/triplet_qa",
)
OUTPUT_FIELDS = {
    "id",
    "source",
    "source_file",
    "source_line",
    "task_type",
    "conversations",
}
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class ChatLengthAuditError(RuntimeError):
    """表示 chat template 长度审计无法安全完成。"""


@dataclass
class ScopeStats:
    """聚合一个 split 或来源的固定长度审计指标。"""

    counts: Counter[str] = field(default_factory=Counter)
    full_lengths: Counter[int] = field(default_factory=Counter)
    assistant_lengths: Counter[int] = field(default_factory=Counter)
    visible_assistant_lengths: Counter[int] = field(default_factory=Counter)
    total_full_tokens: int = 0
    total_assistant_tokens: int = 0
    total_visible_assistant_tokens: int = 0

    def record(self, metrics: dict[str, int | bool]) -> None:
        full_tokens = int(metrics["full_tokens"])
        assistant_tokens = int(metrics["candidate_assistant_label_tokens"])
        visible_tokens = int(metrics["candidate_assistant_labels_within_limit"])
        self.counts["records"] += 1
        self.counts["within_limit" if metrics["within_limit"] else "over_limit"] += 1
        self.counts[
            "assistant_complete" if metrics["assistant_complete"] else "assistant_tail_truncated"
        ] += 1
        if not visible_tokens:
            self.counts["zero_candidate_assistant_labels_at_limit"] += 1
        self.full_lengths[full_tokens] += 1
        self.assistant_lengths[assistant_tokens] += 1
        self.visible_assistant_lengths[visible_tokens] += 1
        self.total_full_tokens += full_tokens
        self.total_assistant_tokens += assistant_tokens
        self.total_visible_assistant_tokens += visible_tokens

    def to_report(self) -> dict[str, object]:
        records = self.counts["records"]
        return {
            "counts": {
                "records": records,
                "within_limit": self.counts["within_limit"],
                "over_limit": self.counts["over_limit"],
                "assistant_complete": self.counts["assistant_complete"],
                "assistant_tail_truncated": self.counts["assistant_tail_truncated"],
                "zero_candidate_assistant_labels_at_limit": self.counts[
                    "zero_candidate_assistant_labels_at_limit"
                ],
            },
            "rates": {
                "within_limit": round(auditor.safe_ratio(self.counts["within_limit"], records), 8),
                "over_limit": round(auditor.safe_ratio(self.counts["over_limit"], records), 8),
                "assistant_complete": round(
                    auditor.safe_ratio(self.counts["assistant_complete"], records), 8
                ),
            },
            "full_tokens": auditor.histogram_report(
                self.full_lengths, self.total_full_tokens
            ),
            "candidate_assistant_label_tokens": auditor.histogram_report(
                self.assistant_lengths, self.total_assistant_tokens
            ),
            "candidate_assistant_labels_within_limit": auditor.histogram_report(
                self.visible_assistant_lengths, self.total_visible_assistant_tokens
            ),
        }


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ChatLengthAuditError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise ChatLengthAuditError(f"{description}必须是 JSON object: {path}")
    return value


def _verify_adjacent_hash(path: Path) -> dict[str, object]:
    hash_path = path.with_suffix(".sha256")
    if not hash_path.is_file():
        raise ChatLengthAuditError(f"data manifest 缺少相邻 SHA-256 清单: {hash_path}")
    try:
        lines = [line for line in hash_path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeDecodeError) as error:
        raise ChatLengthAuditError(f"无法读取 data manifest SHA-256 清单: {hash_path}") from error
    if len(lines) != 1:
        raise ChatLengthAuditError("data manifest SHA-256 清单必须只有一条记录")
    parts = lines[0].split("  ", 1)
    actual = auditor.sha256_file(path)
    if len(parts) != 2 or parts[0] != actual or parts[1] != path.name:
        raise ChatLengthAuditError("data manifest SHA-256 清单与 JSON 不一致")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": actual,
        "hash_manifest": {
            "path": str(hash_path),
            "bytes": hash_path.stat().st_size,
            "sha256": auditor.sha256_file(hash_path),
        },
    }


def _stream_file_identity(path: Path) -> dict[str, int | str]:
    digest = hashlib.sha256()
    records = 0
    try:
        with path.open("rb") as source:
            for line in source:
                digest.update(line)
                records += 1
    except OSError as error:
        raise ChatLengthAuditError(f"无法读取标准化 JSONL: {path}") from error
    return {"records": records, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _validate_data_manifest(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, object], dict[str, Path]]:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise ChatLengthAuditError(f"data manifest 不存在: {manifest_path}")
    identity = _verify_adjacent_hash(manifest_path)
    manifest = _load_json(manifest_path, "data manifest")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("pipeline") != "disc_law_sft_retain_only_v1"
    ):
        raise ChatLengthAuditError("data manifest 版本或 pipeline 无效")
    readiness = manifest.get("readiness")
    if not isinstance(readiness, dict) or readiness.get("standardization_complete") is not True:
        raise ChatLengthAuditError("data manifest 没有证明标准化已经完成")
    validation = manifest.get("validation")
    if not isinstance(validation, dict) or validation.get("quick_is_subset_of_full") is not True:
        raise ChatLengthAuditError("data manifest 没有证明 quick 是 full 子集")
    output = manifest.get("output")
    if not isinstance(output, dict) or not isinstance(output.get("root"), str):
        raise ChatLengthAuditError("data manifest 缺少标准化输出根目录")
    output_root = Path(output["root"]).resolve()
    files = output.get("files")
    if not isinstance(files, dict) or set(files) != set(EXPECTED_FILE_KEYS):
        raise ChatLengthAuditError("data manifest 的标准化文件范围无效")

    resolved_paths: dict[str, Path] = {}
    verified_files: dict[str, object] = {}
    for key in EXPECTED_FILE_KEYS:
        metadata = files[key]
        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("path"), str)
            or not isinstance(metadata.get("records"), int)
            or not isinstance(metadata.get("bytes"), int)
            or not isinstance(metadata.get("sha256"), str)
            or DIGEST_RE.fullmatch(metadata["sha256"]) is None
        ):
            raise ChatLengthAuditError(f"标准化文件元数据无效: {key}")
        path = (output_root / metadata["path"]).resolve()
        if not auditor._path_is_within(path, output_root) or not path.is_file():
            raise ChatLengthAuditError(f"标准化文件不存在或越出输出根目录: {key}")
        actual = _stream_file_identity(path)
        if actual != {
            "records": metadata["records"],
            "bytes": metadata["bytes"],
            "sha256": metadata["sha256"],
        }:
            raise ChatLengthAuditError(f"标准化文件身份已变化: {key}")
        resolved_paths[key] = path
        verified_files[key] = {"path": str(path), **actual}

    expected_unique_records = manifest.get("records", {}).get("totals", {}).get("standardized")
    actual_unique_records = sum(int(files[key]["records"]) for key in UNIQUE_FILE_KEYS)
    if expected_unique_records != actual_unique_records:
        raise ChatLengthAuditError("train + full 与 standardized 记录数不闭合")
    identity["verified_output_files"] = verified_files
    identity["unique_records"] = actual_unique_records
    return manifest, identity, resolved_paths


def _tokenizer_identity(tokenizer: Any, tokenizer_path: Path) -> dict[str, object]:
    tokenizer_path = tokenizer_path.resolve()
    if len(tokenizer) != EXPECTED_VOCAB_SIZE:
        raise ChatLengthAuditError(
            f"Tokenizer 词表大小应为 {EXPECTED_VOCAB_SIZE}，实际为 {len(tokenizer)}"
        )
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ChatLengthAuditError("Tokenizer 缺少固定 chat_template")
    files: dict[str, object] = {}
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        path = tokenizer_path / filename
        if not path.is_file():
            raise ChatLengthAuditError(f"Tokenizer 身份文件不存在: {path}")
        files[filename] = {
            "bytes": path.stat().st_size,
            "sha256": auditor.sha256_file(path),
        }
    return {
        "path": str(tokenizer_path),
        "vocab_size": len(tokenizer),
        "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest(),
        "files": files,
    }


def _extract_input_ids(encoded: Any) -> list[int] | list[list[int]]:
    if isinstance(encoded, dict):
        values = encoded.get("input_ids")
    else:
        values = getattr(encoded, "input_ids", None)
    if not isinstance(values, list):
        raise ChatLengthAuditError("Tokenizer 没有返回 input_ids list")
    return values


def _encode_marker(tokenizer: Any, text: str) -> list[int]:
    try:
        values = _extract_input_ids(tokenizer(text, add_special_tokens=False))
    except Exception as error:
        raise ChatLengthAuditError("无法编码 assistant 边界标记") from error
    if not values or not all(isinstance(value, int) for value in values):
        raise ChatLengthAuditError("assistant 边界标记 token 无效")
    return values  # type: ignore[return-value]


def _find_subsequence(values: list[int], pattern: list[int], start: int = 0) -> int:
    for index in range(start, len(values) - len(pattern) + 1):
        if values[index : index + len(pattern)] == pattern:
            return index
    return -1


def analyze_tokenized_record(
    input_ids: list[int],
    assistant_start_marker: list[int],
    assistant_end_marker: list[int],
) -> dict[str, int | bool]:
    """计算固定 768 下的完整性和候选 assistant label 指标。"""

    marker_start = _find_subsequence(input_ids, assistant_start_marker)
    if marker_start < 0:
        raise ChatLengthAuditError("chat template 中没有唯一 assistant 起始标记")
    if _find_subsequence(input_ids, assistant_start_marker, marker_start + 1) >= 0:
        raise ChatLengthAuditError("chat template 中存在多个 assistant 起始标记")
    payload_start = marker_start + len(assistant_start_marker)
    marker_end = _find_subsequence(input_ids, assistant_end_marker, payload_start)
    if marker_end < 0:
        raise ChatLengthAuditError("chat template 中缺少 assistant 结束标记")
    end_exclusive = marker_end + len(assistant_end_marker)
    if end_exclusive != len(input_ids):
        raise ChatLengthAuditError("assistant 结束标记后存在未解释 token")
    assistant_tokens = end_exclusive - payload_start
    visible_tokens = max(0, min(end_exclusive, MAX_SEQ_LEN) - payload_start)
    return {
        "full_tokens": len(input_ids),
        "candidate_assistant_label_tokens": assistant_tokens,
        "candidate_assistant_labels_within_limit": visible_tokens,
        "tokens_over_limit": max(0, len(input_ids) - MAX_SEQ_LEN),
        "within_limit": len(input_ids) <= MAX_SEQ_LEN,
        "assistant_complete": end_exclusive <= MAX_SEQ_LEN,
    }


def _validate_record(record: object, key: str, line_number: int) -> dict[str, object]:
    if not isinstance(record, dict) or set(record) != OUTPUT_FIELDS:
        raise ChatLengthAuditError(f"标准化记录 schema 无效: {key}:{line_number}")
    _, dataset_kind = key.rsplit("/", 1)
    expected_task = "legal_qa" if dataset_kind == "pair_qa" else "legal_qa_with_context"
    expected_namespace = f"disc_law_sft:{dataset_kind}:"
    if (
        not isinstance(record.get("id"), str)
        or not record["id"].startswith(expected_namespace)
        or record.get("source") != "disc_law_sft"
        or not isinstance(record.get("source_file"), str)
        or not isinstance(record.get("source_line"), int)
        or record["source_line"] <= 0
        or record.get("task_type") != expected_task
    ):
        raise ChatLengthAuditError(f"标准化记录身份无效: {key}:{line_number}")
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        raise ChatLengthAuditError(f"标准化 conversations 无效: {key}:{line_number}")
    user_message, assistant_message = conversations
    if not isinstance(user_message, dict) or not isinstance(assistant_message, dict):
        raise ChatLengthAuditError(f"标准化 conversations 无效: {key}:{line_number}")
    if (
        set(user_message) != {"role", "content"}
        or set(assistant_message) != {"role", "content"}
        or user_message.get("role") != "user"
        or assistant_message.get("role") != "assistant"
        or not isinstance(user_message.get("content"), str)
        or not user_message["content"]
        or not isinstance(assistant_message.get("content"), str)
        or not assistant_message["content"]
    ):
        raise ChatLengthAuditError(f"标准化 conversations 无效: {key}:{line_number}")
    return record


def _render_prompt(tokenizer: Any, conversations: list[dict[str, str]]) -> str:
    try:
        prompt = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as error:
        raise ChatLengthAuditError("无法应用固定 chat_template") from error
    if not isinstance(prompt, str) or not prompt:
        raise ChatLengthAuditError("chat_template 返回了空文本")
    return prompt


def _encode_prompts(tokenizer: Any, prompts: list[str]) -> list[list[int]]:
    try:
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        values = _extract_input_ids(encoded)
    except Exception as error:
        raise ChatLengthAuditError("批量编码 chat template 失败") from error
    if len(values) != len(prompts) or any(
        not isinstance(item, list) or any(not isinstance(token, int) for token in item)
        for item in values
    ):
        raise ChatLengthAuditError("批量 Tokenizer 输出结构无效")
    return values  # type: ignore[return-value]


def _scope_report(scopes: dict[str, ScopeStats]) -> dict[str, object]:
    return {name: scopes[name].to_report() for name in sorted(scopes)}


def _atomic_write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    json.loads(path.read_text(encoding="utf-8"))


def audit_sft_chat_lengths(
    data_manifest: Path,
    tokenizer_path: Path,
    output_dir: Path,
    tokenizer: Any | None = None,
) -> dict[str, object]:
    """审计 train + full 唯一记录，不截断或改写任何样本。"""

    data_manifest = data_manifest.resolve()
    tokenizer_path = tokenizer_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ChatLengthAuditError(f"审计输出目录必须是新目录: {output_dir}")
    manifest, manifest_identity, input_paths = _validate_data_manifest(data_manifest)
    data_root = Path(manifest["output"]["root"]).resolve()
    if auditor._path_is_within(output_dir, data_root):
        raise ChatLengthAuditError("审计报告目录不能位于标准化数据目录内")
    tokenizer = tokenizer or auditor.load_tokenizer(tokenizer_path)
    tokenizer_report = _tokenizer_identity(tokenizer, tokenizer_path)
    bos_token = getattr(tokenizer, "bos_token", None)
    eos_token = getattr(tokenizer, "eos_token", None)
    if not isinstance(bos_token, str) or not isinstance(eos_token, str):
        raise ChatLengthAuditError("Tokenizer 缺少 assistant 边界特殊 token")
    assistant_start_marker = _encode_marker(tokenizer, f"{bos_token}assistant\n")
    assistant_end_marker = _encode_marker(tokenizer, f"{eos_token}\n")

    output_dir.mkdir(parents=True)
    report_path = output_dir / REPORT_FILENAME
    overflow_path = output_dir / OVERFLOW_FILENAME
    hash_path = output_dir / HASH_FILENAME
    report_pending = report_path.with_name(report_path.name + ".pending")
    overflow_pending = overflow_path.with_name(overflow_path.name + ".pending")
    hash_pending = hash_path.with_name(hash_path.name + ".pending")

    overall = ScopeStats()
    by_split: dict[str, ScopeStats] = {}
    by_dataset: dict[str, ScopeStats] = {}
    by_split_dataset: dict[str, ScopeStats] = {}
    scanned = 0
    overflow_records = 0
    try:
        with overflow_pending.open("x", encoding="utf-8", newline="\n") as overflow_output:
            for key in UNIQUE_FILE_KEYS:
                split, dataset_kind = key.rsplit("/", 1)
                split_stats = by_split.setdefault(split, ScopeStats())
                dataset_stats = by_dataset.setdefault(dataset_kind, ScopeStats())
                combined_stats = by_split_dataset.setdefault(key, ScopeStats())
                prompts: list[str] = []
                locators: list[dict[str, object]] = []
                character_budget = 0

                def flush() -> None:
                    nonlocal character_budget, overflow_records, scanned
                    if not prompts:
                        return
                    tokenized = _encode_prompts(tokenizer, prompts)
                    for locator, input_ids in zip(locators, tokenized):
                        metrics = analyze_tokenized_record(
                            input_ids, assistant_start_marker, assistant_end_marker
                        )
                        for scope in (overall, split_stats, dataset_stats, combined_stats):
                            scope.record(metrics)
                        risks = []
                        if not metrics["within_limit"]:
                            risks.append("full_sequence_over_768")
                        if not metrics["assistant_complete"]:
                            risks.append("assistant_tail_truncated_at_768")
                        if not metrics["candidate_assistant_labels_within_limit"]:
                            risks.append("zero_candidate_assistant_labels_at_768")
                        if risks:
                            overflow_output.write(
                                json.dumps(
                                    {
                                        **locator,
                                        "full_tokens": metrics["full_tokens"],
                                        "candidate_assistant_label_tokens": metrics[
                                            "candidate_assistant_label_tokens"
                                        ],
                                        "candidate_assistant_labels_within_limit": metrics[
                                            "candidate_assistant_labels_within_limit"
                                        ],
                                        "tokens_over_limit": metrics["tokens_over_limit"],
                                        "risks": risks,
                                    },
                                    ensure_ascii=False,
                                    separators=(",", ":"),
                                )
                                + "\n"
                            )
                            overflow_records += 1
                        scanned += 1
                        if scanned % PROGRESS_INTERVAL == 0:
                            print(f"[长度审计] 已处理 {scanned:,} 条", file=sys.stderr, flush=True)
                    prompts.clear()
                    locators.clear()
                    character_budget = 0

                with input_paths[key].open("r", encoding="utf-8") as source:
                    for line_number, line in enumerate(source, start=1):
                        try:
                            parsed = json.loads(line)
                        except json.JSONDecodeError as error:
                            raise ChatLengthAuditError(
                                f"标准化 JSONL 无法解析: {key}:{line_number}"
                            ) from error
                        record = _validate_record(parsed, key, line_number)
                        conversations = record["conversations"]
                        prompt = _render_prompt(tokenizer, conversations)  # type: ignore[arg-type]
                        prompts.append(prompt)
                        locators.append(
                            {
                                "id": record["id"],
                                "source_file": record["source_file"],
                                "source_line": record["source_line"],
                                "split": split,
                                "dataset_kind": dataset_kind,
                            }
                        )
                        character_budget += len(prompt)
                        if (
                            len(prompts) >= TOKENIZER_BATCH_SIZE
                            or character_budget >= TOKENIZER_CHARACTER_BUDGET
                        ):
                            flush()
                flush()

        expected_records = int(manifest_identity["unique_records"])
        if scanned != expected_records:
            raise ChatLengthAuditError("长度审计记录数与 data manifest 不闭合")
        totals_report = overall.to_report()
        if totals_report["counts"]["records"] != (
            totals_report["counts"]["within_limit"] + totals_report["counts"]["over_limit"]
        ):
            raise ChatLengthAuditError("固定 768 长度漏斗不闭合")
        if overflow_records != totals_report["counts"]["over_limit"]:
            raise ChatLengthAuditError("超长 locator 与汇总数量不闭合")

        report: dict[str, object] = {
            "schema_version": "1.0",
            "pipeline": "legal_sft_chat_length_audit_768_v1",
            "scope": {
                "fixed_max_seq_len": MAX_SEQ_LEN,
                "files": list(UNIQUE_FILE_KEYS),
                "quick_excluded_from_unique_totals": True,
                "audit_only": True,
                "training_dataset_written": False,
                "input_files_modified": False,
            },
            "input": {
                "data_manifest": manifest_identity,
                "release_status": manifest.get("release_status"),
            },
            "tokenizer": tokenizer_report,
            "template": {
                "render": "apply_chat_template_tokenize_false_add_generation_prompt_false",
                "tokenize": "add_special_tokens_false",
                "assistant_start_marker_ids": assistant_start_marker,
                "assistant_end_marker_ids": assistant_end_marker,
                "candidate_label_scope": (
                    "assistant 起始标记之后至 <|im_end|> 换行结束，包含固定空 think 块；"
                    "最终 Dataset label mask 仍需单独验收"
                ),
            },
            "records": {
                "totals": totals_report,
                "by_split": _scope_report(by_split),
                "by_dataset": _scope_report(by_dataset),
                "by_split_dataset": _scope_report(by_split_dataset),
            },
            "decision": {
                "max_seq_len": MAX_SEQ_LEN,
                "policy": "fixed_768_audit_only_no_truncation_no_rewrite",
                "overlength_records": "isolated_for_follow_up_decision",
            },
            "readiness": {
                "length_audit_complete": True,
                "all_records_fit_768": totals_report["counts"]["over_limit"] == 0,
                "dataset_label_mask_audited": False,
                "training_ready": False,
            },
            "outputs": {
                "report": REPORT_FILENAME,
                "overflow_locators": {
                    "path": OVERFLOW_FILENAME,
                    "records": overflow_records,
                },
                "sha256_manifest": HASH_FILENAME,
            },
            "limitations": [
                "本报告不截断、改写、排除或发布任何训练记录。",
                "候选 assistant label 范围沿用当前模板边界；正式 Dataset mask 需另行测试。",
                "本报告不判断法律正确性、语义近重复或 LawBench/STARD 污染。",
            ],
            "complete": True,
        }
        _atomic_write_json(report_pending, report)
        hash_pending.write_text(
            f"{auditor.sha256_file(report_pending)}  {REPORT_FILENAME}\n"
            f"{auditor.sha256_file(overflow_pending)}  {OVERFLOW_FILENAME}\n",
            encoding="utf-8",
            newline="\n",
        )
        report_pending.replace(report_path)
        overflow_pending.replace(overflow_path)
        hash_pending.replace(hash_path)
        return report
    except BaseException:
        raise


def main() -> None:
    """解析云端参数并执行固定 768 的只读长度审计。"""

    parser = argparse.ArgumentParser(description="固定 768 只读审计法律 SFT chat template 长度")
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_DATA_MANIFEST)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        report = audit_sft_chat_lengths(
            data_manifest=args.data_manifest,
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
        )
    except (ChatLengthAuditError, OSError, ValueError) as error:
        raise SystemExit(f"[失败] {error}") from error
    counts = report["records"]["totals"]["counts"]
    print(
        f"[完成] 固定 768 审计 {counts['records']:,} 条，"
        f"完整 {counts['within_limit']:,}，超长 {counts['over_limit']:,}，"
        f"assistant 尾部截断风险 {counts['assistant_tail_truncated']:,}"
    )
    print("[阻断] 本程序只发布审计报告，不生成训练数据，training_ready=false")
    print(f"汇总报告: {args.output_dir / REPORT_FILENAME}")
    print(f"超长定位: {args.output_dir / OVERFLOW_FILENAME}")


if __name__ == "__main__":
    main()
