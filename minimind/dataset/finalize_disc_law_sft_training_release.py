"""验证基础法律 SFT 数据关卡并发布正式训练 manifest。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

try:
    from . import audit_sft_chat_lengths as length_auditor
    from . import sft_dataset
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import audit_sft_chat_lengths as length_auditor
    from dataset import sft_dataset


DEFAULT_WORK_ROOT = Path(
    os.environ.get("MINIMIND_WORK_ROOT", "/root/autodl-tmp/minimind-work")
)
DEFAULT_PROJECT_ROOT = Path(
    os.environ.get("MINIMIND_PROJECT_ROOT", "/root/autodl-tmp/minimind")
)
DEFAULT_FIXED_MANIFEST = (
    DEFAULT_WORK_ROOT / "manifests" / "disc-law-sft-v1-formal-768-candidate.json"
)
DEFAULT_EVALUATION_EXCLUSIONS = (
    DEFAULT_PROJECT_ROOT
    / "dataset"
    / "RAG-SFT"
    / "manifests"
    / "evaluation-exclusions-project-rag-v2.json"
)
DEFAULT_LABEL_REPORT = (
    DEFAULT_WORK_ROOT
    / "reports"
    / "sft"
    / "disc_law_sft_formal_label_audit_768_v1"
    / "sft-dataset-label-audit-768.json"
)
DEFAULT_OUTPUT = (
    DEFAULT_WORK_ROOT / "manifests" / "disc-law-sft-v1-formal-768.json"
)

PIPELINE = "disc_law_sft_length_filtered_768_v1"
PARENT_PIPELINE = "disc_law_sft_retain_only_v1"
EXCLUSIONS_PIPELINE = "legal_sft_evaluation_exclusions"
LENGTH_PIPELINE = "legal_sft_chat_length_audit_768_v1"
LABEL_PIPELINE = "legal_sft_dataset_label_audit_768_v1"


class DiscLawSftReleaseError(RuntimeError):
    """表示基础法律 SFT 正式发布条件未闭合。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiscLawSftReleaseError(message)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DiscLawSftReleaseError(f"无法读取{label}: {path}") from error
    if not isinstance(payload, dict):
        raise DiscLawSftReleaseError(f"{label}必须是 JSON 对象: {path}")
    return payload


def _verify_single_sidecar(path: Path, label: str) -> Path:
    sidecar = path.with_suffix(".sha256")
    try:
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise DiscLawSftReleaseError(f"无法读取{label} SHA-256 清单: {sidecar}") from error
    expected = f"{_sha256_file(path)}  {path.name}"
    if lines != [expected]:
        raise DiscLawSftReleaseError(f"{label} SHA-256 校验失败")
    return sidecar


def _load_verified_json(
    path: Path, label: str
) -> tuple[dict[str, Any], dict[str, object], Path]:
    path = path.resolve()
    if not path.is_file():
        raise DiscLawSftReleaseError(f"{label}不存在: {path}")
    sidecar = _verify_single_sidecar(path, label)
    return _load_json(path, label), _identity(path), sidecar


def _verify_embedded_identity(value: object, label: str) -> Path:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise DiscLawSftReleaseError(f"{label}身份无效")
    path = Path(value["path"]).resolve()
    if (
        not path.is_file()
        or value.get("bytes") != path.stat().st_size
        or value.get("sha256") != _sha256_file(path)
    ):
        raise DiscLawSftReleaseError(f"{label}身份与当前文件不一致")
    return path


def _verify_multi_file_sidecar(sidecar: Path, expected: set[Path]) -> None:
    try:
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise DiscLawSftReleaseError(f"无法读取长度审计 SHA-256 清单: {sidecar}") from error
    actual_paths: set[Path] = set()
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise DiscLawSftReleaseError("长度审计 SHA-256 清单格式无效")
        digest, filename = parts[0], parts[1].strip()
        target = (sidecar.parent / filename).resolve()
        if not target.is_file() or _sha256_file(target) != digest:
            raise DiscLawSftReleaseError(f"长度审计 SHA-256 校验失败: {target}")
        actual_paths.add(target)
    if actual_paths != {path.resolve() for path in expected}:
        raise DiscLawSftReleaseError("长度审计 SHA-256 清单文件范围无效")


def _stream_jsonl_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    records = 0
    with path.open("rb") as source:
        for line in source:
            digest.update(line)
            records += 1
    return {
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _verify_fixed_outputs(manifest: dict[str, Any]) -> tuple[int, int]:
    output = manifest.get("output")
    if not isinstance(output, dict) or not isinstance(output.get("root"), str):
        raise DiscLawSftReleaseError("固定 768 manifest 缺少输出根目录")
    files = output.get("files")
    if not isinstance(files, dict) or set(files) != set(length_auditor.EXPECTED_FILE_KEYS):
        raise DiscLawSftReleaseError("固定 768 manifest 的六文件范围无效")
    record_fields = output.get("record_fields")
    if (
        not isinstance(record_fields, list)
        or len(record_fields) != len(length_auditor.OUTPUT_FIELDS)
        or not all(isinstance(field, str) for field in record_fields)
        or set(record_fields) != length_auditor.OUTPUT_FIELDS
    ):
        raise DiscLawSftReleaseError("固定 768 manifest 的记录字段范围无效")
    output_root = Path(output["root"]).resolve()
    counts: dict[str, int] = {}
    for key in length_auditor.EXPECTED_FILE_KEYS:
        metadata = files[key]
        if not isinstance(metadata, dict) or not isinstance(metadata.get("path"), str):
            raise DiscLawSftReleaseError(f"固定 768 文件元数据无效: {key}")
        path = (output_root / metadata["path"]).resolve()
        try:
            path.relative_to(output_root)
        except ValueError as error:
            raise DiscLawSftReleaseError(f"固定 768 文件越出输出根目录: {key}") from error
        if not path.is_file():
            raise DiscLawSftReleaseError(f"固定 768 文件不存在: {key}")
        actual = _stream_jsonl_identity(path)
        expected = {
            "records": metadata.get("records"),
            "bytes": metadata.get("bytes"),
            "sha256": metadata.get("sha256"),
        }
        if actual != expected:
            raise DiscLawSftReleaseError(f"固定 768 文件身份已变化: {key}")
        counts[key] = int(actual["records"])
    unique_records = sum(counts[key] for key in length_auditor.UNIQUE_FILE_KEYS)
    quick_records = sum(
        counts[key]
        for key in length_auditor.EXPECTED_FILE_KEYS
        if key.startswith("validation/quick/")
    )
    totals = manifest.get("records", {}).get("totals", {})
    if (
        totals.get("standardized") != unique_records
        or totals.get("train")
        != sum(counts[key] for key in counts if key.startswith("train/"))
        or totals.get("full_validation")
        != sum(counts[key] for key in counts if key.startswith("validation/full/"))
        or totals.get("quick_validation") != quick_records
    ):
        raise DiscLawSftReleaseError("固定 768 记录计数未闭合")
    return unique_records, quick_records


def _validate_evaluation_exclusions(payload: dict[str, Any]) -> None:
    assets = payload.get("assets")
    asset_names = {
        asset.get("name") for asset in assets if isinstance(asset, dict)
    } if isinstance(assets, list) else set()
    audit = payload.get("isolation_audit")
    overlap_counts = audit.get("overlap_counts") if isinstance(audit, dict) else None
    _require(
        payload.get("schema_version") == "1.2"
        and payload.get("pipeline") == EXCLUSIONS_PIPELINE
        and payload.get("complete_for_formal_sft") is True
        and payload.get("raw_questions_included") is False
        and {"project_rag_eval", "project_private_holdout_v2"} <= asset_names
        and isinstance(audit, dict)
        and audit.get("compatible") is True
        and audit.get("raw_text_included") is False
        and isinstance(overlap_counts, dict)
        and bool(overlap_counts)
        and all(value == 0 for value in overlap_counts.values()),
        "正式评估排除清单关卡未闭合",
    )


def _publish(manifest: dict[str, object], output_path: Path) -> None:
    output_path = output_path.resolve()
    hash_path = output_path.with_suffix(".sha256")
    partial = output_path.with_name(output_path.name + ".partial")
    hash_partial = hash_path.with_name(hash_path.name + ".partial")
    occupied = [
        str(path) for path in (output_path, hash_path, partial, hash_partial) if path.exists()
    ]
    if occupied:
        raise DiscLawSftReleaseError("输出位置已有发布产物: " + ", ".join(occupied))
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        hash_partial.write_text(
            f"{_sha256_file(partial)}  {output_path.name}\n",
            encoding="utf-8",
            newline="\n",
        )
        partial.replace(output_path)
        hash_partial.replace(hash_path)
    except OSError as error:
        partial.unlink(missing_ok=True)
        hash_partial.unlink(missing_ok=True)
        raise DiscLawSftReleaseError("无法发布基础法律 SFT 正式 manifest") from error


def finalize_release(
    *,
    fixed_manifest: Path,
    evaluation_exclusions: Path,
    label_report: Path,
    output_path: Path,
) -> dict[str, object]:
    """复算正式数据血缘并发布可训练的固定 768 manifest。"""

    output_path = Path(output_path).resolve()
    if output_path.exists() or output_path.with_suffix(".sha256").exists():
        raise DiscLawSftReleaseError("输出位置已有发布产物")
    fixed, fixed_identity, fixed_sidecar = _load_verified_json(
        Path(fixed_manifest), "固定 768 candidate manifest"
    )
    exclusions, exclusions_identity, exclusions_sidecar = _load_verified_json(
        Path(evaluation_exclusions), "正式评估排除 manifest"
    )
    labels, labels_identity, labels_sidecar = _load_verified_json(
        Path(label_report), "label 审计报告"
    )

    fixed_readiness = fixed.get("readiness")
    _require(
        fixed.get("schema_version") == "1.0"
        and fixed.get("pipeline") == PIPELINE
        and fixed.get("release_status") == "formal_candidate"
        and fixed.get("complete") is True
        and isinstance(fixed_readiness, dict)
        and fixed_readiness.get("standardization_complete") is True
        and fixed_readiness.get("formal_evaluation_isolation_complete") is True
        and fixed_readiness.get("chat_template_length_audited") is True
        and fixed_readiness.get("overlength_records_isolated") is True,
        "固定 768 manifest 尚非正式候选",
    )
    unique_records, quick_records = _verify_fixed_outputs(fixed)

    parent_path = _verify_embedded_identity(
        fixed.get("parent_data_manifest"), "正式标准化父 manifest"
    )
    parent, parent_identity, parent_sidecar = _load_verified_json(
        parent_path, "正式标准化父 manifest"
    )
    parent_readiness = parent.get("readiness")
    _require(
        parent.get("schema_version") == "1.0"
        and parent.get("pipeline") == PARENT_PIPELINE
        and parent.get("release_status") == "formal_candidate"
        and parent.get("complete") is True
        and isinstance(parent_readiness, dict)
        and parent_readiness.get("standardization_complete") is True
        and parent_readiness.get("formal_evaluation_isolation_complete") is True,
        "标准化父 manifest 尚非正式候选",
    )
    embedded_parent = fixed["parent_data_manifest"]
    _require(
        embedded_parent.get("sha256") == parent_identity["sha256"]
        and embedded_parent.get("bytes") == parent_identity["bytes"]
        and embedded_parent.get("unique_records")
        == parent.get("records", {}).get("totals", {}).get("standardized", unique_records),
        "固定 768 manifest 与标准化父 manifest 身份不一致",
    )

    _validate_evaluation_exclusions(exclusions)
    for owner, embedded in (
        ("标准化父 manifest", parent.get("evaluation_exclusion")),
        ("固定 768 manifest", fixed.get("evaluation_exclusion")),
    ):
        _require(
            isinstance(embedded, dict)
            and embedded.get("sha256") == exclusions_identity["sha256"]
            and embedded.get("bytes") == exclusions_identity["bytes"]
            and embedded.get("complete_for_formal_sft") is True,
            f"{owner}未绑定当前正式评估排除 manifest",
        )

    length = fixed.get("length_audit")
    _require(isinstance(length, dict), "固定 768 manifest 缺少长度审计身份")
    length_report_path = _verify_embedded_identity(length.get("report"), "长度审计报告")
    overflow_path = _verify_embedded_identity(
        length.get("overflow_locators"), "超长定位清单"
    )
    length_sidecar = _verify_embedded_identity(
        length.get("hash_manifest"), "长度审计 SHA-256 清单"
    )
    _verify_multi_file_sidecar(length_sidecar, {length_report_path, overflow_path})
    length_report = _load_json(length_report_path, "长度审计报告")
    _require(
        length.get("fixed_max_seq_len") == sft_dataset.MAX_SEQ_LEN
        and length_report.get("schema_version") == "1.0"
        and length_report.get("pipeline") == LENGTH_PIPELINE
        and length_report.get("scope", {}).get("fixed_max_seq_len")
        == sft_dataset.MAX_SEQ_LEN
        and length_report.get("input", {}).get("data_manifest", {}).get("sha256")
        == parent_identity["sha256"]
        and length_report.get("readiness", {}).get("length_audit_complete") is True
        and length_report.get("complete") is True,
        "长度审计与正式标准化父 manifest 未闭合",
    )

    label_validation = labels.get("validation")
    label_readiness = labels.get("readiness")
    _require(
        labels.get("schema_version") == "1.0"
        and labels.get("pipeline") == LABEL_PIPELINE
        and labels.get("complete") is True
        and labels.get("scope", {}).get("fixed_max_seq_len")
        == sft_dataset.MAX_SEQ_LEN
        and labels.get("scope", {}).get("label_mask_version")
        == sft_dataset.LABEL_MASK_VERSION
        and labels.get("input", {}).get("data_manifest", {}).get("pipeline")
        == PIPELINE
        and labels.get("input", {}).get("data_manifest", {}).get("sha256")
        == fixed_identity["sha256"]
        and labels.get("records", {}).get("totals", {}).get("records")
        == unique_records
        and labels.get("records", {}).get("quick_identity_only_records")
        == quick_records
        and isinstance(label_validation, dict)
        and label_validation.get("expected_unique_records") == unique_records
        and label_validation.get("scanned_unique_records") == unique_records
        and label_validation.get("zero_active_label_records") == 0
        and label_validation.get("label_input_mismatch_records") == 0
        and label_validation.get("invalid_shape_records") == 0
        and label_validation.get("quick_file_identity_verified") is True
        and label_validation.get("counts_closed") is True
        and isinstance(label_readiness, dict)
        and label_readiness.get("dataset_label_mask_audited") is True
        and label_readiness.get("all_unique_records_have_active_labels") is True,
        "label 审计与固定 768 candidate 未闭合",
    )

    release = dict(fixed)
    release["release_status"] = "formal_training_candidate"
    release["release_artifacts"] = {
        "fixed_manifest": {
            "file": fixed_identity,
            "sha256_manifest": _identity(fixed_sidecar),
        },
        "parent_manifest": {
            "file": parent_identity,
            "sha256_manifest": _identity(parent_sidecar),
        },
        "evaluation_exclusions": {
            "file": exclusions_identity,
            "sha256_manifest": _identity(exclusions_sidecar),
        },
        "label_report": {
            "file": labels_identity,
            "sha256_manifest": _identity(labels_sidecar),
        },
    }
    release["tokenizer"] = labels.get("tokenizer")
    release["readiness"] = {
        "standardization_complete": True,
        "formal_evaluation_isolation_complete": True,
        "chat_template_length_audited": True,
        "overlength_records_isolated": True,
        "dataset_label_mask_audited": True,
        "training_ready": True,
    }
    release["limitations"] = [
        "数据层就绪不代表父权重、训练参数或正式训练运行已经验收。"
    ]
    release["complete"] = True
    _publish(release, output_path)
    return release


def main() -> None:
    """解析正式发布输入并输出基础法律 SFT manifest。"""

    parser = argparse.ArgumentParser(description="发布基础法律 SFT 正式训练 manifest")
    parser.add_argument("--fixed-manifest", type=Path, default=DEFAULT_FIXED_MANIFEST)
    parser.add_argument(
        "--evaluation-exclusions", type=Path, default=DEFAULT_EVALUATION_EXCLUSIONS
    )
    parser.add_argument("--label-report", type=Path, default=DEFAULT_LABEL_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        release = finalize_release(
            fixed_manifest=args.fixed_manifest,
            evaluation_exclusions=args.evaluation_exclusions,
            label_report=args.label_report,
            output_path=args.output,
        )
    except (DiscLawSftReleaseError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(release, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
