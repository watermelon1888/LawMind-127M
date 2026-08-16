"""全量审计固定 768 RAG-SFT Dataset 的 assistant-only labels。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

try:
    from . import audit_disc_law_sft as auditor
    from . import audit_sft_chat_lengths as length_auditor
    from . import derive_rag_sft_768 as derivation
    from . import rag_sft_dataset
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import audit_disc_law_sft as auditor
    from dataset import audit_sft_chat_lengths as length_auditor
    from dataset import derive_rag_sft_768 as derivation
    from dataset import rag_sft_dataset


RAG_SFT_ROOT = Path(__file__).resolve().parent / "RAG-SFT"
DEFAULT_DATA_MANIFEST = (
    RAG_SFT_ROOT / "manifests" / "rag-sft-canonical-v1-manifest-768.json"
)
DEFAULT_TOKENIZER_PATH = auditor.DEFAULT_TOKENIZER_PATH
DEFAULT_OUTPUT_DIR = (
    RAG_SFT_ROOT / "reports" / "rag-sft-canonical-v1-label-audit-768"
)

PIPELINE = "rag_sft_dataset_label_audit_768"
REPORT_FILENAME = "rag-sft-dataset-label-audit-768.json"
HASH_FILENAME = "rag-sft-dataset-label-audit-768.sha256"


class RagSftLabelAuditError(RuntimeError):
    """固定 768 RAG-SFT labels 无法安全完成全量审计。"""


@dataclass
class ScopeStats:
    """聚合一个 RAG-SFT 分层的有效 label 数量。"""

    records: int = 0
    total_active_labels: int = 0
    active_label_lengths: Counter[int] = field(default_factory=Counter)

    def record(self, active_labels: int) -> None:
        self.records += 1
        self.total_active_labels += active_labels
        self.active_label_lengths[active_labels] += 1

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
                    self.records * rag_sft_dataset.MAX_SEQ_LEN,
                ),
                8,
            ),
        }


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftLabelAuditError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise RagSftLabelAuditError(f"{description}必须是 JSON object")
    return value


def _scope_report(scopes: dict[str, ScopeStats]) -> dict[str, object]:
    return {name: scopes[name].to_report() for name in sorted(scopes)}


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    json.loads(path.read_text(encoding="utf-8"))


def _validate_item(
    *,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    record: dict[str, object],
    tokenizer: Any,
    end_ids: list[int],
) -> int:
    record_id = record["id"]
    if (
        not isinstance(input_ids, torch.Tensor)
        or not isinstance(labels, torch.Tensor)
        or input_ids.dtype != torch.long
        or labels.dtype != torch.long
        or input_ids.shape != (rag_sft_dataset.MAX_SEQ_LEN,)
        or labels.shape != input_ids.shape
    ):
        raise RagSftLabelAuditError(f"Dataset shape 或 dtype 无效: {record_id}")
    active_positions = torch.nonzero(labels != -100, as_tuple=False).flatten()
    if not active_positions.numel():
        raise RagSftLabelAuditError(f"样本没有有效 assistant label: {record_id}")
    first = int(active_positions[0])
    last = int(active_positions[-1])
    expected_positions = torch.arange(first, last + 1, dtype=torch.long)
    if not torch.equal(active_positions.cpu(), expected_positions):
        raise RagSftLabelAuditError(f"有效 labels 不是连续 assistant 尾部: {record_id}")
    if not torch.equal(labels[active_positions], input_ids[active_positions]):
        raise RagSftLabelAuditError(f"有效 labels 与 input_ids 不一致: {record_id}")
    if torch.any(labels[:first] != -100) or torch.any(labels[last + 1 :] != -100):
        raise RagSftLabelAuditError(f"输入或 padding 被误计入 labels: {record_id}")

    assistant_content = record["conversations"][-1]["content"]
    expected_active = rag_sft_dataset._token_ids(
        tokenizer, f"{assistant_content}{tokenizer.eos_token}\n"
    )
    active_values = labels[active_positions].tolist()
    if active_values != expected_active:
        raise RagSftLabelAuditError(
            f"有效 labels 未精确覆盖 assistant JSON 与 EOS: {record_id}"
        )
    if active_values[-len(end_ids) :] != end_ids:
        raise RagSftLabelAuditError(f"EOS 没有完整参与 labels: {record_id}")
    original_length = last + 1
    if torch.any(input_ids[original_length:] != tokenizer.pad_token_id):
        raise RagSftLabelAuditError(f"assistant EOS 后不是纯 padding: {record_id}")
    return len(active_values)


def audit_rag_sft_dataset_labels(
    *,
    data_manifest: Path,
    tokenizer_path: Path,
    output_dir: Path,
    tokenizer: Any | None = None,
) -> dict[str, object]:
    """逐条调用真实 Dataset，并验证 JSON-only label mask。"""

    data_manifest = Path(data_manifest).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RagSftLabelAuditError(f"审计输出目录必须是新目录: {output_dir}")
    try:
        manifest_identity = length_auditor._verify_adjacent_hash(data_manifest)
    except length_auditor.ChatLengthAuditError as error:
        raise RagSftLabelAuditError(str(error)) from error
    manifest = _load_json(data_manifest, "固定 768 RAG-SFT manifest")
    if manifest.get("pipeline") != derivation.PIPELINE:
        raise RagSftLabelAuditError("固定 768 RAG-SFT manifest pipeline 无效")

    tokenizer = tokenizer or auditor.load_tokenizer(tokenizer_path)
    try:
        dataset = rag_sft_dataset.RagSftDataset(data_manifest, tokenizer)
    except rag_sft_dataset.RagSftDatasetError as error:
        raise RagSftLabelAuditError(str(error)) from error
    end_ids = rag_sft_dataset._token_ids(tokenizer, f"{tokenizer.eos_token}\n")
    overall = ScopeStats()
    by_evidence_source: dict[str, ScopeStats] = {}
    by_behavior: dict[str, ScopeStats] = {}
    validation_counts: Counter[str] = Counter()
    try:
        for index in range(len(dataset)):
            record, _line_number = dataset._load_record(index)
            input_ids, labels = dataset[index]
            active_count = _validate_item(
                input_ids=input_ids,
                labels=labels,
                record=record,
                tokenizer=tokenizer,
                end_ids=end_ids,
            )
            assistant = json.loads(record["conversations"][-1]["content"])
            behavior = "refusal" if assistant["refuse"] else "answer"
            evidence_source = record["evidence_source"]
            overall.record(active_count)
            by_behavior.setdefault(behavior, ScopeStats()).record(active_count)
            by_evidence_source.setdefault(
                evidence_source, ScopeStats()
            ).record(active_count)
            validation_counts["shape_valid"] += 1
            validation_counts["input_side_masked"] += 1
            validation_counts["assistant_json_complete"] += 1
            validation_counts["eos_labeled"] += 1
            validation_counts["padding_masked"] += 1
    finally:
        dataset.close()

    expected_records = manifest.get("records", {}).get("candidate")
    if type(expected_records) is not int or overall.records != expected_records:
        raise RagSftLabelAuditError("Dataset 记录数与固定 768 manifest 不闭合")
    if any(validation_counts[name] != expected_records for name in (
        "shape_valid",
        "input_side_masked",
        "assistant_json_complete",
        "eos_labeled",
        "padding_masked",
    )):
        raise RagSftLabelAuditError("Dataset label 验证计数不闭合")

    candidate = manifest["output"]["candidate"]
    report: dict[str, object] = {
        "pipeline": PIPELINE,
        "scope": {
            "fixed_max_seq_len": rag_sft_dataset.MAX_SEQ_LEN,
            "records": expected_records,
            "label_mask_version": rag_sft_dataset.LABEL_MASK_VERSION,
            "label_scope": "assistant JSON + EOS",
            "system_user_evidence_masked": True,
            "assistant_template_prefix_masked": True,
            "fixed_empty_think_masked": True,
            "padding_masked": True,
            "audit_only": True,
            "input_files_modified": False,
        },
        "input": {
            "data_manifest": manifest_identity,
            "candidate": candidate,
        },
        "tokenizer": manifest["tokenizer"],
        "records": {
            "totals": overall.to_report(),
            "by_evidence_source": _scope_report(by_evidence_source),
            "by_behavior": _scope_report(by_behavior),
        },
        "validation": {
            "expected_records": expected_records,
            "scanned_records": overall.records,
            "zero_active_label_records": 0,
            "mismatched_label_records": 0,
            "non_contiguous_label_records": 0,
            "input_side_label_leak_records": 0,
            "incomplete_assistant_json_records": 0,
            "missing_eos_label_records": 0,
            "padding_label_leak_records": 0,
            "counts_closed": True,
        },
        "readiness": {
            "dataset_label_mask_audited": True,
            "all_records_have_complete_assistant_json_and_eos": True,
            "training_ready": False,
        },
        "limitations": [
            "本报告只验证固定 768 Dataset 的确定性 label mask。",
            "本报告不判断 summary 法律正确性或真实 retrieval 表现。",
            "评估隔离、运行时输出预算和 retrieved 数据尚未全部闭合。",
        ],
        "outputs": {
            "report": REPORT_FILENAME,
            "sha256_manifest": HASH_FILENAME,
        },
        "complete": True,
    }

    output_dir.mkdir(parents=True)
    report_path = output_dir / REPORT_FILENAME
    hash_path = output_dir / HASH_FILENAME
    report_pending = report_path.with_name(report_path.name + ".pending")
    hash_pending = hash_path.with_name(hash_path.name + ".pending")
    try:
        _write_json(report_pending, report)
        hash_pending.write_text(
            f"{auditor.sha256_file(report_pending)}  {REPORT_FILENAME}\n",
            encoding="utf-8",
            newline="\n",
        )
        report_pending.replace(report_path)
        hash_pending.replace(hash_path)
    except (OSError, UnicodeError, ValueError) as error:
        for path in (report_pending, hash_pending, report_path, hash_path):
            path.unlink(missing_ok=True)
        raise RagSftLabelAuditError("无法发布 RAG-SFT label 审计产物") from error
    return report


def main() -> None:
    """解析参数并执行固定 768 RAG-SFT label 审计。"""

    parser = argparse.ArgumentParser(
        description="审计固定 768 RAG-SFT 的 assistant-only labels"
    )
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_DATA_MANIFEST)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        report = audit_rag_sft_dataset_labels(
            data_manifest=args.data_manifest,
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
        )
    except (
        RagSftLabelAuditError,
        rag_sft_dataset.RagSftDatasetError,
        OSError,
        ValueError,
    ) as error:
        raise SystemExit(f"[失败] {error}") from error
    totals = report["records"]["totals"]
    print(
        f"[完成] 审计 {totals['records']} 条，"
        f"有效 assistant label tokens {totals['total_active_label_tokens']}"
    )
    print("[阻断] retrieved 数据和其他训练关卡尚未闭合，training_ready=false")
    print(f"报告目录: {args.output_dir}")


if __name__ == "__main__":
    main()
