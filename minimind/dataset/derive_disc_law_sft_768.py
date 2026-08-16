"""按已验收的超长 locator 派生固定 768 的法律 SFT 候选数据。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from . import audit_disc_law_sft as auditor
    from . import audit_sft_chat_lengths as length_auditor
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import audit_disc_law_sft as auditor
    from dataset import audit_sft_chat_lengths as length_auditor


DEFAULT_WORK_ROOT = Path(os.environ.get("MINIMIND_WORK_ROOT", "/root/autodl-tmp/minimind-work"))
DEFAULT_DATA_MANIFEST = length_auditor.DEFAULT_DATA_MANIFEST
DEFAULT_LENGTH_AUDIT_DIR = length_auditor.DEFAULT_OUTPUT_DIR
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_WORK_ROOT
    / "data"
    / "standardized"
    / "sft"
    / "disc_law_sft_v1_project_rag_only_provisional_768"
)
DEFAULT_MANIFEST_OUTPUT = (
    DEFAULT_WORK_ROOT
    / "manifests"
    / "disc-law-sft-v1-project-rag-only-provisional-768.json"
)

PIPELINE = "disc_law_sft_length_filtered_768_v1"
LOCATOR_FIELDS = {
    "id",
    "source_file",
    "source_line",
    "split",
    "dataset_kind",
    "full_tokens",
    "candidate_assistant_label_tokens",
    "candidate_assistant_labels_within_limit",
    "tokens_over_limit",
    "risks",
}
SOURCE_FILES = {
    "pair_qa": "DISC-Law-SFT-Pair-QA-released.jsonl",
    "triplet_qa": "DISC-Law-SFT-Triplet-QA-released.jsonl",
}


class SftLengthDerivationError(RuntimeError):
    """表示无法安全发布固定 768 的派生候选数据。"""


class BinaryJsonlWriter:
    """复制原始 JSONL 字节，并在全部校验通过后发布。"""

    def __init__(self, path: Path):
        self.path = path
        self.partial_path = path.with_name(path.name + ".partial")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.partial_path.open("xb")
        self.records = 0

    def write(self, raw_line: bytes) -> None:
        self.file.write(raw_line)
        self.records += 1

    def publish(self, output_root: Path) -> dict[str, object]:
        self.file.close()
        self.partial_path.replace(self.path)
        return {
            "path": self.path.relative_to(output_root).as_posix(),
            "records": self.records,
            "bytes": self.path.stat().st_size,
            "sha256": auditor.sha256_file(self.path),
        }

    def abort(self) -> None:
        if not self.file.closed:
            self.file.close()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SftLengthDerivationError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise SftLengthDerivationError(f"{description}必须是 JSON object: {path}")
    return value


def _load_hash_entries(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise SftLengthDerivationError(f"无法读取长度审计 SHA-256 清单: {path}") from error
    for line_number, line in enumerate(lines, 1):
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or length_auditor.DIGEST_RE.fullmatch(parts[0]) is None:
            raise SftLengthDerivationError(f"长度审计 SHA-256 清单格式无效: {path}:{line_number}")
        if parts[1] in entries:
            raise SftLengthDerivationError(f"长度审计 SHA-256 清单文件名重复: {parts[1]}")
        entries[parts[1]] = parts[0]
    return entries


def _validate_locator(value: object, line_number: int) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != LOCATOR_FIELDS:
        raise SftLengthDerivationError(f"overflow locator schema 无效: {line_number}")
    split = value.get("split")
    dataset_kind = value.get("dataset_kind")
    full_tokens = value.get("full_tokens")
    label_tokens = value.get("candidate_assistant_label_tokens")
    visible_labels = value.get("candidate_assistant_labels_within_limit")
    tokens_over = value.get("tokens_over_limit")
    source_line = value.get("source_line")
    if (
        split not in {"train", "validation/full"}
        or dataset_kind not in SOURCE_FILES
        or not isinstance(value.get("id"), str)
        or not value["id"].startswith(f"disc_law_sft:{dataset_kind}:")
        or value.get("source_file") != SOURCE_FILES[dataset_kind]
        or type(source_line) is not int
        or source_line <= 0
        or type(full_tokens) is not int
        or full_tokens <= length_auditor.MAX_SEQ_LEN
        or type(label_tokens) is not int
        or type(visible_labels) is not int
        or visible_labels < 0
        or label_tokens <= visible_labels
        or type(tokens_over) is not int
        or tokens_over != full_tokens - length_auditor.MAX_SEQ_LEN
    ):
        raise SftLengthDerivationError(f"overflow locator 身份或长度无效: {line_number}")
    expected_risks = ["full_sequence_over_768", "assistant_tail_truncated_at_768"]
    if visible_labels == 0:
        expected_risks.append("zero_candidate_assistant_labels_at_768")
    if value.get("risks") != expected_risks:
        raise SftLengthDerivationError(f"overflow locator 风险标记无效: {line_number}")
    return value


def _validate_length_audit(
    audit_dir: Path,
    parent_identity: dict[str, object],
) -> tuple[dict[str, Any], dict[str, object], dict[str, dict[str, dict[str, object]]]]:
    report_path = audit_dir / length_auditor.REPORT_FILENAME
    overflow_path = audit_dir / length_auditor.OVERFLOW_FILENAME
    hash_path = audit_dir / length_auditor.HASH_FILENAME
    for path in (report_path, overflow_path, hash_path):
        if not path.is_file():
            raise SftLengthDerivationError(f"长度审计产物不存在: {path}")

    hash_entries = _load_hash_entries(hash_path)
    expected_names = {length_auditor.REPORT_FILENAME, length_auditor.OVERFLOW_FILENAME}
    if set(hash_entries) != expected_names:
        raise SftLengthDerivationError("长度审计 SHA-256 清单文件范围无效")
    for path in (report_path, overflow_path):
        if auditor.sha256_file(path) != hash_entries[path.name]:
            raise SftLengthDerivationError(f"长度审计产物 SHA-256 不匹配: {path.name}")

    report = _load_json(report_path, "长度审计报告")
    scope = report.get("scope")
    counts = report.get("records", {}).get("totals", {}).get("counts", {})
    report_parent = report.get("input", {}).get("data_manifest", {})
    readiness = report.get("readiness")
    if (
        report.get("schema_version") != "1.0"
        or report.get("pipeline") != "legal_sft_chat_length_audit_768_v1"
        or report.get("complete") is not True
        or not isinstance(scope, dict)
        or scope.get("fixed_max_seq_len") != length_auditor.MAX_SEQ_LEN
        or scope.get("files") != list(length_auditor.UNIQUE_FILE_KEYS)
        or scope.get("quick_excluded_from_unique_totals") is not True
        or not isinstance(readiness, dict)
        or readiness.get("length_audit_complete") is not True
        or report_parent.get("sha256") != parent_identity.get("sha256")
        or report_parent.get("bytes") != parent_identity.get("bytes")
        or counts.get("records") != parent_identity.get("unique_records")
        or type(counts.get("over_limit")) is not int
        or counts["over_limit"] < 0
        or counts.get("within_limit") != counts["records"] - counts["over_limit"]
        or counts.get("assistant_tail_truncated") != counts["over_limit"]
    ):
        raise SftLengthDerivationError("长度审计报告与父 data manifest 不闭合")

    locators: dict[str, dict[str, dict[str, object]]] = {
        key: {} for key in length_auditor.UNIQUE_FILE_KEYS
    }
    seen_locations: set[tuple[str, int]] = set()
    locator_records = 0
    try:
        with overflow_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    raise SftLengthDerivationError(f"overflow locator 包含空行: {line_number}")
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SftLengthDerivationError(
                        f"overflow locator JSON 无效: {line_number}"
                    ) from error
                locator = _validate_locator(parsed, line_number)
                key = f"{locator['split']}/{locator['dataset_kind']}"
                record_id = str(locator["id"])
                location = (str(locator["source_file"]), int(locator["source_line"]))
                if record_id in locators[key]:
                    raise SftLengthDerivationError(f"overflow locator ID 重复: {record_id}")
                if location in seen_locations:
                    raise SftLengthDerivationError(
                        f"overflow locator 源定位重复: {location[0]}:{location[1]}"
                    )
                locators[key][record_id] = locator
                seen_locations.add(location)
                locator_records += 1
    except (OSError, UnicodeDecodeError) as error:
        raise SftLengthDerivationError(f"无法读取 overflow locator: {overflow_path}") from error

    output_locator = report.get("outputs", {}).get("overflow_locators", {})
    if (
        locator_records != counts["over_limit"]
        or output_locator.get("path") != length_auditor.OVERFLOW_FILENAME
        or output_locator.get("records") != locator_records
    ):
        raise SftLengthDerivationError("overflow locator 与长度审计报告数量不闭合")

    audit_identity = {
        "report": {
            "path": str(report_path),
            "bytes": report_path.stat().st_size,
            "sha256": hash_entries[report_path.name],
        },
        "overflow_locators": {
            "path": str(overflow_path),
            "records": locator_records,
            "bytes": overflow_path.stat().st_size,
            "sha256": hash_entries[overflow_path.name],
        },
        "hash_manifest": {
            "path": str(hash_path),
            "bytes": hash_path.stat().st_size,
            "sha256": auditor.sha256_file(hash_path),
        },
        "fixed_max_seq_len": length_auditor.MAX_SEQ_LEN,
    }
    return report, audit_identity, locators


def _parse_standardized_record(raw_line: bytes, key: str, line_number: int) -> dict[str, object]:
    try:
        value = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SftLengthDerivationError(f"标准化 JSONL 无法解析: {key}:{line_number}") from error
    try:
        return length_auditor._validate_record(value, key, line_number)
    except length_auditor.ChatLengthAuditError as error:
        raise SftLengthDerivationError(str(error)) from error


def _record_matches_locator(record: dict[str, object], locator: dict[str, object]) -> bool:
    return (
        record["id"] == locator["id"]
        and record["source_file"] == locator["source_file"]
        and record["source_line"] == locator["source_line"]
    )


def _write_manifest(manifest: dict[str, object], output_path: Path) -> None:
    hash_path = output_path.with_suffix(".sha256")
    partial_path = output_path.with_name(output_path.name + ".partial")
    hash_partial_path = hash_path.with_name(hash_path.name + ".partial")
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial_path.write_text(payload, encoding="utf-8", newline="\n")
        json.loads(partial_path.read_text(encoding="utf-8"))
        hash_partial_path.write_text(
            f"{auditor.sha256_file(partial_path)}  {output_path.name}\n",
            encoding="utf-8",
            newline="\n",
        )
        partial_path.replace(output_path)
        hash_partial_path.replace(hash_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SftLengthDerivationError(f"无法发布派生 data manifest: {output_path}") from error


def derive_disc_law_sft_768(
    data_manifest: Path,
    length_audit_dir: Path,
    output_root: Path,
    manifest_output: Path,
) -> dict[str, object]:
    """整条隔离固定 768 超长记录，并重建 quick 子集。"""

    data_manifest = data_manifest.resolve()
    length_audit_dir = length_audit_dir.resolve()
    output_root = output_root.resolve()
    manifest_output = manifest_output.resolve()
    if output_root.exists():
        raise SftLengthDerivationError(f"派生数据输出目录必须是新目录: {output_root}")
    manifest_paths = (
        manifest_output,
        manifest_output.with_suffix(".sha256"),
        manifest_output.with_name(manifest_output.name + ".partial"),
        manifest_output.with_suffix(".sha256").with_name(
            manifest_output.with_suffix(".sha256").name + ".partial"
        ),
    )
    occupied = [str(path) for path in manifest_paths if path.exists()]
    if occupied:
        raise SftLengthDerivationError("manifest 输出位置已被占用: " + ", ".join(occupied))

    try:
        parent_manifest, parent_identity, input_paths = length_auditor._validate_data_manifest(
            data_manifest
        )
    except length_auditor.ChatLengthAuditError as error:
        raise SftLengthDerivationError(str(error)) from error
    input_root = Path(parent_manifest["output"]["root"]).resolve()
    if auditor._path_is_within(output_root, input_root) or auditor._path_is_within(
        manifest_output, input_root
    ):
        raise SftLengthDerivationError("派生输出和 manifest 不能位于父数据目录内")

    _, audit_identity, locators = _validate_length_audit(
        length_audit_dir, parent_identity
    )
    output_paths = {
        key: output_root / f"{key}.jsonl" for key in length_auditor.EXPECTED_FILE_KEYS
    }
    output_root.mkdir(parents=True)
    writers = {key: BinaryJsonlWriter(path) for key, path in output_paths.items()}
    matched_ids: set[str] = set()
    full_records: dict[str, dict[str, bytes]] = {"pair_qa": {}, "triplet_qa": {}}
    retained_full_ids: dict[str, set[str]] = {"pair_qa": set(), "triplet_qa": set()}
    input_counts: Counter[str] = Counter()
    excluded_counts: Counter[str] = Counter()
    quick_excluded_counts: Counter[str] = Counter()
    try:
        for key in length_auditor.UNIQUE_FILE_KEYS:
            split, dataset_kind = key.rsplit("/", 1)
            with input_paths[key].open("rb") as source:
                for line_number, raw_line in enumerate(source, 1):
                    record = _parse_standardized_record(raw_line, key, line_number)
                    record_id = str(record["id"])
                    input_counts[key] += 1
                    if split == "validation/full":
                        if record_id in full_records[dataset_kind]:
                            raise SftLengthDerivationError(f"full validation ID 重复: {record_id}")
                        full_records[dataset_kind][record_id] = raw_line
                    locator = locators[key].get(record_id)
                    if locator is not None:
                        if not _record_matches_locator(record, locator):
                            raise SftLengthDerivationError(
                                f"overflow locator 与记录身份不一致: {record_id}"
                            )
                        matched_ids.add(record_id)
                        excluded_counts[key] += 1
                        continue
                    writers[key].write(raw_line)
                    if split == "validation/full":
                        retained_full_ids[dataset_kind].add(record_id)

        expected_ids = {
            record_id for by_id in locators.values() for record_id in by_id
        }
        if matched_ids != expected_ids:
            missing = sorted(expected_ids - matched_ids)
            raise SftLengthDerivationError(
                "overflow locator 未在父数据中逐条命中: " + ", ".join(missing[:5])
            )

        for key in length_auditor.EXPECTED_FILE_KEYS[4:]:
            _, _, dataset_kind = key.split("/")
            seen_quick_ids: set[str] = set()
            with input_paths[key].open("rb") as source:
                for line_number, raw_line in enumerate(source, 1):
                    record = _parse_standardized_record(raw_line, key, line_number)
                    record_id = str(record["id"])
                    input_counts[key] += 1
                    if record_id in seen_quick_ids:
                        raise SftLengthDerivationError(f"quick validation ID 重复: {record_id}")
                    seen_quick_ids.add(record_id)
                    full_raw_line = full_records[dataset_kind].get(record_id)
                    if full_raw_line is None or full_raw_line != raw_line:
                        raise SftLengthDerivationError(
                            f"quick 不是 full 的物化子集: {record_id}"
                        )
                    if record_id not in retained_full_ids[dataset_kind]:
                        quick_excluded_counts[dataset_kind] += 1
                        continue
                    writers[key].write(raw_line)

        total_locators = sum(len(by_id) for by_id in locators.values())
        if sum(excluded_counts.values()) != total_locators:
            raise SftLengthDerivationError("实际隔离数量与 overflow locator 不闭合")
        for key in length_auditor.EXPECTED_FILE_KEYS:
            expected_records = int(parent_manifest["output"]["files"][key]["records"])
            if input_counts[key] != expected_records:
                raise SftLengthDerivationError(f"输入记录数与父 manifest 不闭合: {key}")

        outputs = {
            key: writers[key].publish(output_root)
            for key in length_auditor.EXPECTED_FILE_KEYS
        }
    except BaseException:
        for writer in writers.values():
            writer.abort()
        raise

    by_dataset: dict[str, dict[str, int]] = {}
    totals: Counter[str] = Counter()
    for dataset_kind in ("pair_qa", "triplet_qa"):
        train_key = f"train/{dataset_kind}"
        full_key = f"validation/full/{dataset_kind}"
        quick_key = f"validation/quick/{dataset_kind}"
        counts = {
            "parent_standardized": input_counts[train_key] + input_counts[full_key],
            "excluded_overlength": excluded_counts[train_key] + excluded_counts[full_key],
            "standardized": writers[train_key].records + writers[full_key].records,
            "train": writers[train_key].records,
            "full_validation": writers[full_key].records,
            "quick_validation": writers[quick_key].records,
            "quick_excluded_overlength": quick_excluded_counts[dataset_kind],
        }
        by_dataset[dataset_kind] = counts
        totals.update(counts)

    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "pipeline": PIPELINE,
        "release_status": parent_manifest.get("release_status"),
        "parent_data_manifest": parent_identity,
        "length_audit": audit_identity,
        "source": parent_manifest.get("source"),
        "cleaning": parent_manifest.get("cleaning"),
        "evaluation_exclusion": parent_manifest.get("evaluation_exclusion"),
        "policy": {
            "fixed_max_seq_len": length_auditor.MAX_SEQ_LEN,
            "overlength_records": "exclude_whole_record_by_verified_locator",
            "truncation": "forbidden",
            "rewrite": "forbidden",
            "quick_rebuild": "original_quick_intersection_retained_full",
        },
        "normalization": parent_manifest.get("normalization"),
        "validation": {
            **parent_manifest.get("validation", {}),
            "quick_is_subset_of_filtered_full": True,
            "all_overflow_locators_matched_once": True,
        },
        "records": {
            "by_dataset": by_dataset,
            "totals": dict(sorted(totals.items())),
        },
        "output": {
            "root": str(output_root),
            "record_fields": parent_manifest.get("output", {}).get("record_fields"),
            "files": outputs,
        },
        "readiness": {
            "standardization_complete": True,
            "formal_evaluation_isolation_complete": parent_manifest.get("readiness", {}).get(
                "formal_evaluation_isolation_complete", False
            ),
            "chat_template_length_audited": True,
            "overlength_records_isolated": True,
            "dataset_label_mask_audited": False,
            "cpt_parent_bound": False,
            "training_ready": False,
        },
        "limitations": [
            "固定 768 candidate 仍需完成全量 Dataset assistant label 审计和正式发布复算。",
            "数据层 candidate 不绑定 CPT 父权重，也不代表正式训练运行已经验收。",
        ],
        "complete": True,
    }
    _write_manifest(manifest, manifest_output)
    return manifest


def main() -> None:
    """解析云端参数并发布固定 768 的派生候选数据。"""

    parser = argparse.ArgumentParser(description="整条隔离固定 768 超长记录并重建法律 SFT quick")
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_DATA_MANIFEST)
    parser.add_argument("--length-audit-dir", type=Path, default=DEFAULT_LENGTH_AUDIT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    args = parser.parse_args()
    try:
        manifest = derive_disc_law_sft_768(
            data_manifest=args.data_manifest,
            length_audit_dir=args.length_audit_dir,
            output_root=args.output_root,
            manifest_output=args.manifest_output,
        )
    except SftLengthDerivationError as error:
        print(f"[错误] {error}", file=sys.stderr)
        raise SystemExit(2) from error

    totals = manifest["records"]["totals"]
    print(
        f"[完成] 隔离 {totals['excluded_overlength']:,} 条，"
        f"train={totals['train']:,}，full={totals['full_validation']:,}，"
        f"quick={totals['quick_validation']:,}"
    )
    print("[待完成] 全量 Dataset label 审计和正式发布复算，当前 training_ready=false")
    print(f"data manifest: {args.manifest_output}")


if __name__ == "__main__":
    main()
