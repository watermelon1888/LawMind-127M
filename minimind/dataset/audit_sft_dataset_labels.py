"""全量审计固定 768 法律 SFT Dataset 的有效 labels。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

try:
    from . import audit_disc_law_sft as auditor
    from . import audit_sft_chat_lengths as length_auditor
    from . import sft_dataset
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import audit_disc_law_sft as auditor
    from dataset import audit_sft_chat_lengths as length_auditor
    from dataset import sft_dataset


DEFAULT_WORK_ROOT = Path(
    os.environ.get("MINIMIND_WORK_ROOT", "/root/autodl-tmp/minimind-work")
)
DEFAULT_DATA_MANIFEST = (
    DEFAULT_WORK_ROOT
    / "manifests"
    / "disc-law-sft-v1-project-rag-only-provisional-768.json"
)
DEFAULT_TOKENIZER_PATH = auditor.DEFAULT_TOKENIZER_PATH
DEFAULT_OUTPUT_DIR = (
    DEFAULT_WORK_ROOT / "reports" / "sft" / "disc_law_sft_label_audit_768_v1"
)

PIPELINE = "legal_sft_dataset_label_audit_768_v1"
REPORT_FILENAME = "sft-dataset-label-audit-768.json"
HASH_FILENAME = "sft-dataset-label-audit-768.sha256"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 4
PROGRESS_INTERVAL = 10_000


class SftLabelAuditError(RuntimeError):
    """表示全量 SFT label 审计无法安全完成。"""


@dataclass
class ScopeStats:
    """聚合一个 split 或来源的有效 label 数量。"""

    records: int = 0
    total_active_labels: int = 0
    active_label_lengths: Counter[int] = field(default_factory=Counter)

    def record(self, active_labels: list[int]) -> None:
        self.records += len(active_labels)
        self.total_active_labels += sum(active_labels)
        self.active_label_lengths.update(active_labels)

    def to_report(self) -> dict[str, object]:
        return {
            "records": self.records,
            "total_active_label_tokens": self.total_active_labels,
            "active_label_tokens": auditor.histogram_report(
                self.active_label_lengths, self.total_active_labels
            ),
            "active_label_ratio_at_fixed_768": round(
                auditor.safe_ratio(
                    self.total_active_labels,
                    self.records * sft_dataset.MAX_SEQ_LEN,
                ),
                8,
            ),
        }


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SftLabelAuditError(f"无法读取固定 768 data manifest: {path}") from error
    if not isinstance(value, dict):
        raise SftLabelAuditError("固定 768 data manifest 必须是 JSON object")
    return value


def _expected_records(manifest: dict[str, Any], key: str) -> int:
    files = manifest.get("output", {}).get("files")
    metadata = files.get(key) if isinstance(files, dict) else None
    if not isinstance(metadata, dict) or type(metadata.get("records")) is not int:
        raise SftLabelAuditError(f"固定 768 data manifest 缺少记录数: {key}")
    return metadata["records"]


def _validate_batch(
    input_ids: object,
    labels: object,
    key: str,
    first_record_index: int,
) -> list[int]:
    if (
        not isinstance(input_ids, torch.Tensor)
        or not isinstance(labels, torch.Tensor)
        or input_ids.dtype != torch.long
        or labels.dtype != torch.long
        or input_ids.ndim != 2
        or labels.ndim != 2
        or input_ids.shape != labels.shape
        or input_ids.shape[1] != sft_dataset.MAX_SEQ_LEN
    ):
        raise SftLabelAuditError(f"Dataset batch shape 或 dtype 无效: {key}")

    active_mask = labels != -100
    active_counts = active_mask.sum(dim=1)
    zero_rows = torch.nonzero(active_counts == 0, as_tuple=False)
    if zero_rows.numel():
        record_index = first_record_index + int(zero_rows[0, 0])
        raise SftLabelAuditError(f"样本没有有效 assistant label: {key}:{record_index}")
    if not torch.equal(labels[active_mask], input_ids[active_mask]):
        raise SftLabelAuditError(
            f"有效 labels 与 input_ids 不一致: {key}:{first_record_index}"
        )
    return [int(value) for value in active_counts.tolist()]


def _scope_report(scopes: dict[str, ScopeStats]) -> dict[str, object]:
    return {name: scopes[name].to_report() for name in sorted(scopes)}


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    json.loads(path.read_text(encoding="utf-8"))


def audit_sft_dataset_labels(
    data_manifest: Path,
    tokenizer_path: Path,
    output_dir: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    tokenizer: Any | None = None,
) -> dict[str, object]:
    """扫描 train + full 唯一记录并发布不含原文的 label 汇总。"""

    if batch_size <= 0:
        raise SftLabelAuditError("batch_size 必须大于 0")
    if num_workers < 0:
        raise SftLabelAuditError("num_workers 不能小于 0")

    data_manifest = data_manifest.resolve()
    tokenizer_path = tokenizer_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise SftLabelAuditError(f"审计输出目录必须是新目录: {output_dir}")
    manifest = _load_manifest(data_manifest)
    output_root_value = manifest.get("output", {}).get("root")
    if not isinstance(output_root_value, str):
        raise SftLabelAuditError("固定 768 data manifest 缺少输出根目录")
    output_root = Path(output_root_value).resolve()
    if auditor._path_is_within(output_dir, output_root):
        raise SftLabelAuditError("label 审计报告目录不能位于训练数据目录内")

    tokenizer = tokenizer or auditor.load_tokenizer(tokenizer_path)
    overall = ScopeStats()
    by_split: dict[str, ScopeStats] = {}
    by_dataset: dict[str, ScopeStats] = {}
    by_split_dataset: dict[str, ScopeStats] = {}
    scanned = 0
    next_progress = PROGRESS_INTERVAL

    for key in length_auditor.UNIQUE_FILE_KEYS:
        split, dataset_kind = key.rsplit("/", 1)
        dataset = sft_dataset.SftDataset(
            data_manifest,
            split,
            tokenizer,
            dataset_kind=dataset_kind,
        )
        expected = _expected_records(manifest, key)
        if len(dataset) != expected:
            dataset.close()
            raise SftLabelAuditError(f"Dataset 记录数与 manifest 不一致: {key}")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
        key_scanned = 0
        try:
            for input_ids, labels in loader:
                active_labels = _validate_batch(input_ids, labels, key, key_scanned)
                for scope in (
                    overall,
                    by_split.setdefault(split, ScopeStats()),
                    by_dataset.setdefault(dataset_kind, ScopeStats()),
                    by_split_dataset.setdefault(key, ScopeStats()),
                ):
                    scope.record(active_labels)
                batch_records = len(active_labels)
                key_scanned += batch_records
                scanned += batch_records
                if scanned >= next_progress:
                    print(
                        f"[label 审计] 已处理 {scanned:,} 条",
                        file=sys.stderr,
                        flush=True,
                    )
                    while scanned >= next_progress:
                        next_progress += PROGRESS_INTERVAL
        finally:
            dataset.close()
        if key_scanned != expected:
            raise SftLabelAuditError(f"Dataset 扫描记录数与 manifest 不一致: {key}")

    quick_dataset = sft_dataset.SftDataset(
        data_manifest, "validation/quick", tokenizer
    )
    try:
        quick_records = len(quick_dataset)
    finally:
        quick_dataset.close()
    expected_quick_records = sum(
        _expected_records(manifest, key)
        for key in length_auditor.EXPECTED_FILE_KEYS
        if key.startswith("validation/quick/")
    )
    if quick_records != expected_quick_records:
        raise SftLabelAuditError("quick Dataset 记录数与 manifest 不一致")

    expected_unique_records = sum(
        _expected_records(manifest, key) for key in length_auditor.UNIQUE_FILE_KEYS
    )
    if scanned != expected_unique_records or overall.records != expected_unique_records:
        raise SftLabelAuditError("全量 label 审计记录数不闭合")

    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str):
        raise SftLabelAuditError("Tokenizer 缺少 chat_template")
    report: dict[str, object] = {
        "schema_version": "1.0",
        "pipeline": PIPELINE,
        "scope": {
            "fixed_max_seq_len": sft_dataset.MAX_SEQ_LEN,
            "label_mask_version": sft_dataset.LABEL_MASK_VERSION,
            "files": list(length_auditor.UNIQUE_FILE_KEYS),
            "quick_excluded_from_unique_totals": True,
            "audit_only": True,
            "training_data_written": False,
            "input_files_modified": False,
        },
        "input": {
            "data_manifest": {
                "path": str(data_manifest),
                "bytes": data_manifest.stat().st_size,
                "sha256": auditor.sha256_file(data_manifest),
                "pipeline": manifest.get("pipeline"),
                "release_status": manifest.get("release_status"),
            }
        },
        "tokenizer": {
            "path": str(tokenizer_path),
            "vocab_size": len(tokenizer),
            "chat_template_sha256": hashlib.sha256(
                chat_template.encode("utf-8")
            ).hexdigest(),
        },
        "execution": {
            "batch_size": batch_size,
            "num_workers": num_workers,
        },
        "records": {
            "totals": overall.to_report(),
            "by_split": _scope_report(by_split),
            "by_dataset": _scope_report(by_dataset),
            "by_split_dataset": _scope_report(by_split_dataset),
            "quick_identity_only_records": quick_records,
        },
        "validation": {
            "expected_unique_records": expected_unique_records,
            "scanned_unique_records": scanned,
            "zero_active_label_records": 0,
            "label_input_mismatch_records": 0,
            "invalid_shape_records": 0,
            "quick_file_identity_verified": True,
            "counts_closed": True,
        },
        "readiness": {
            "dataset_label_mask_audited": True,
            "all_unique_records_have_active_labels": True,
            "training_ready": False,
        },
        "outputs": {
            "report": REPORT_FILENAME,
            "sha256_manifest": HASH_FILENAME,
        },
        "limitations": [
            "本报告不修改、截断、改写或发布任何训练记录。",
            "quick 只验证文件身份和记录数，不重复计入唯一记录统计。",
            "本报告须由正式发布器复算后，才可解除数据层 training_ready 阻断。",
            "数据层就绪不代表 CPT 父权重或正式训练运行已经验收。",
        ],
        "complete": True,
    }

    output_dir.mkdir(parents=True)
    report_path = output_dir / REPORT_FILENAME
    hash_path = output_dir / HASH_FILENAME
    report_pending = report_path.with_name(report_path.name + ".pending")
    hash_pending = hash_path.with_name(hash_path.name + ".pending")
    _write_json(report_pending, report)
    hash_pending.write_text(
        f"{auditor.sha256_file(report_pending)}  {REPORT_FILENAME}\n",
        encoding="utf-8",
        newline="\n",
    )
    report_pending.replace(report_path)
    hash_pending.replace(hash_path)
    return report


def main() -> None:
    """解析云端参数并执行固定 768 全量 label 审计。"""

    parser = argparse.ArgumentParser(description="全量审计固定 768 法律 SFT Dataset labels")
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_DATA_MANIFEST)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    args = parser.parse_args()
    try:
        report = audit_sft_dataset_labels(
            data_manifest=args.data_manifest,
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    except (SftLabelAuditError, sft_dataset.SftDatasetError, OSError, RuntimeError) as error:
        raise SystemExit(f"[失败] {error}") from error
    totals = report["records"]["totals"]
    print(
        f"[完成] 全量 label 审计 {totals['records']:,} 条，"
        f"有效 assistant tokens {totals['total_active_label_tokens']:,}"
    )
    print("[待完成] 由正式发布器复算全部数据身份并发布 training_ready manifest")
    print(f"汇总报告: {args.output_dir / REPORT_FILENAME}")


if __name__ == "__main__":
    main()
