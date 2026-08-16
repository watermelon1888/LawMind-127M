"""从完整 RAG-SFT 投影确定性派生固定 768 的训练候选。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from . import audit_disc_law_sft as auditor
    from . import audit_rag_sft_chat_lengths as rag_length_auditor
    from . import audit_sft_chat_lengths as length_auditor
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import audit_disc_law_sft as auditor
    from dataset import audit_rag_sft_chat_lengths as rag_length_auditor
    from dataset import audit_sft_chat_lengths as length_auditor


RAG_SFT_ROOT = Path(__file__).resolve().parent / "RAG-SFT"
DEFAULT_DATA_MANIFEST = (
    RAG_SFT_ROOT / "manifests" / "rag-sft-canonical-v1-candidate.json"
)
DEFAULT_LENGTH_AUDIT_DIR = (
    RAG_SFT_ROOT / "reports" / "rag-sft-canonical-v1-chat-length-audit-768"
)
DEFAULT_CANDIDATE_OUTPUT = (
    RAG_SFT_ROOT / "standardized" / "768" / "rag-sft-canonical-v1-candidate.jsonl"
)
DEFAULT_MANIFEST_OUTPUT = (
    RAG_SFT_ROOT / "manifests" / "rag-sft-canonical-v1-manifest-768.json"
)

PIPELINE = "rag_sft_length_filtered_768"


class RagSftLengthDerivationError(RuntimeError):
    """固定 768 RAG-SFT 候选无法安全派生。"""


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftLengthDerivationError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise RagSftLengthDerivationError(f"{description}必须是 JSON object")
    return value


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": auditor.sha256_file(path),
    }
    if records is not None:
        result["records"] = records
    return result


def _payload_identity(
    path: Path, payload: str, *, records: int
) -> dict[str, object]:
    encoded = payload.encode("utf-8")
    return {
        "path": str(path.resolve()),
        "records": records,
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _verify_audit_hashes(audit_dir: Path) -> tuple[Path, Path, Path]:
    report_path = audit_dir / rag_length_auditor.REPORT_FILENAME
    overflow_path = audit_dir / rag_length_auditor.OVERFLOW_FILENAME
    hash_path = audit_dir / rag_length_auditor.HASH_FILENAME
    if not report_path.is_file() or not overflow_path.is_file() or not hash_path.is_file():
        raise RagSftLengthDerivationError("RAG 长度审计产物不完整")
    try:
        lines = [line for line in hash_path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftLengthDerivationError("无法读取 RAG 长度审计哈希清单") from error
    expected = [
        f"{auditor.sha256_file(report_path)}  {report_path.name}",
        f"{auditor.sha256_file(overflow_path)}  {overflow_path.name}",
    ]
    if lines != expected:
        raise RagSftLengthDerivationError("RAG 长度审计哈希清单无效")
    return report_path, overflow_path, hash_path


def _load_overflow_ids(path: Path, expected_records: int) -> set[str]:
    ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagSftLengthDerivationError(
                        f"超长定位清单不允许空行: {line_number}"
                    )
                try:
                    locator = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RagSftLengthDerivationError(
                        f"超长定位 JSON 无效: {line_number}"
                    ) from error
                record_id = locator.get("id") if isinstance(locator, dict) else None
                if not isinstance(record_id, str) or not record_id or record_id in ids:
                    raise RagSftLengthDerivationError("超长定位 ID 无效或重复")
                if locator.get("full_tokens", 0) <= length_auditor.MAX_SEQ_LEN:
                    raise RagSftLengthDerivationError("超长定位记录没有超过固定 768")
                ids.add(record_id)
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftLengthDerivationError("无法读取超长定位清单") from error
    if len(ids) != expected_records:
        raise RagSftLengthDerivationError("超长定位数量与报告不一致")
    return ids


def _filter_candidate(
    candidate_path: Path, overflow_ids: set[str], expected_records: int
) -> tuple[str, Counter[str], set[str]]:
    output_lines: list[str] = []
    source_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    matched_overflow: set[str] = set()
    try:
        with candidate_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagSftLengthDerivationError(
                        f"父 candidate 不允许空行: {line_number}"
                    )
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RagSftLengthDerivationError(
                        f"父 candidate JSON 无效: {line_number}"
                    ) from error
                record_id = record.get("id") if isinstance(record, dict) else None
                evidence_source = (
                    record.get("evidence_source") if isinstance(record, dict) else None
                )
                if (
                    not isinstance(record_id, str)
                    or not record_id
                    or record_id in seen_ids
                    or evidence_source not in rag_length_auditor.EVIDENCE_SOURCES
                ):
                    raise RagSftLengthDerivationError(
                        f"父 candidate 身份无效: {line_number}"
                    )
                seen_ids.add(record_id)
                if record_id in overflow_ids:
                    matched_overflow.add(record_id)
                    continue
                output_lines.append(line)
                source_counts[evidence_source] += 1
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftLengthDerivationError("无法读取父 candidate") from error
    if len(seen_ids) != expected_records:
        raise RagSftLengthDerivationError("父 candidate 记录数与 manifest 不一致")
    if matched_overflow != overflow_ids:
        raise RagSftLengthDerivationError("超长定位未在父 candidate 中逐条命中")
    return "".join(output_lines), source_counts, seen_ids - overflow_ids


def _require_new_outputs(candidate_output: Path, manifest_output: Path) -> None:
    targets = (
        candidate_output,
        manifest_output,
        manifest_output.with_suffix(".sha256"),
    )
    occupied = [
        str(path)
        for target in targets
        for path in (target, target.with_name(target.name + ".partial"))
        if path.exists()
    ]
    if occupied:
        raise RagSftLengthDerivationError("目标输出已存在: " + ", ".join(occupied))


def _publish(
    candidate_output: Path,
    candidate_payload: str,
    manifest_output: Path,
    manifest_payload: str,
) -> None:
    hash_output = manifest_output.with_suffix(".sha256")
    outputs = (
        (candidate_output, candidate_payload),
        (manifest_output, manifest_payload),
        (
            hash_output,
            f"{hashlib.sha256(manifest_payload.encode('utf-8')).hexdigest()}  "
            f"{manifest_output.name}\n",
        ),
    )
    partials = [
        (path.with_name(path.name + ".partial"), path, payload)
        for path, payload in outputs
    ]
    published: list[Path] = []
    try:
        for partial, _, payload in partials:
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(payload, encoding="utf-8", newline="\n")
        for partial, final, _ in partials:
            partial.replace(final)
            published.append(final)
    except (OSError, UnicodeError) as error:
        for path in [*(item[0] for item in partials), *reversed(published)]:
            path.unlink(missing_ok=True)
        raise RagSftLengthDerivationError("无法发布固定 768 RAG-SFT 产物") from error


def derive_rag_sft_768(
    *,
    data_manifest: Path,
    length_audit_dir: Path,
    candidate_output: Path,
    manifest_output: Path,
) -> dict[str, object]:
    """只排除审计明确定位的超长记录并发布新身份。"""

    data_manifest = Path(data_manifest).resolve()
    length_audit_dir = Path(length_audit_dir).resolve()
    candidate_output = Path(candidate_output).resolve()
    manifest_output = Path(manifest_output).resolve()
    _require_new_outputs(candidate_output, manifest_output)
    try:
        (
            parent_manifest,
            parent_identity,
            parent_candidate,
            parent_candidate_identity,
            _authoring_path,
            authoring_identity,
        ) = rag_length_auditor._verify_manifest(data_manifest)
    except rag_length_auditor.RagChatLengthAuditError as error:
        raise RagSftLengthDerivationError(str(error)) from error

    report_path, overflow_path, audit_hash_path = _verify_audit_hashes(
        length_audit_dir
    )
    report = _load_json(report_path, "RAG 长度审计报告")
    scope = report.get("scope")
    validation = report.get("validation")
    readiness = report.get("readiness")
    if (
        report.get("pipeline") != "rag_sft_chat_length_audit_768"
        or report.get("complete") is not True
        or not isinstance(scope, dict)
        or scope.get("fixed_max_seq_len") != length_auditor.MAX_SEQ_LEN
        or not isinstance(validation, dict)
        or validation.get("counts_closed") is not True
        or not isinstance(readiness, dict)
        or readiness.get("chat_template_length_audited") is not True
        or report.get("input", {}).get("rag_manifest", {}).get("sha256")
        != parent_identity["sha256"]
        or report.get("input", {}).get("candidate", {}).get("sha256")
        != parent_candidate_identity["sha256"]
    ):
        raise RagSftLengthDerivationError("RAG 长度审计与父 manifest 不闭合")
    overflow_meta = report.get("outputs", {}).get("overflow")
    if not isinstance(overflow_meta, dict) or type(overflow_meta.get("records")) is not int:
        raise RagSftLengthDerivationError("RAG 长度审计缺少超长记录数")
    overflow_ids = _load_overflow_ids(overflow_path, overflow_meta["records"])
    candidate_payload, source_counts, retained_ids = _filter_candidate(
        parent_candidate,
        overflow_ids,
        parent_manifest["records"]["candidate"],
    )

    candidate_identity = _payload_identity(
        candidate_output, candidate_payload, records=len(retained_ids)
    )

    isolated = parent_manifest.get("readiness", {}).get(
        "evaluation_isolation_complete"
    ) is True
    manifest: dict[str, object] = {
        "pipeline": PIPELINE,
        "release_status": parent_manifest.get("release_status"),
        "parent_data_manifest": parent_identity,
        "parent_candidate": parent_candidate_identity,
        "authoring": authoring_identity,
        "length_audit": {
            "report": _identity(report_path),
            "overflow": _identity(overflow_path, records=len(overflow_ids)),
            "sha256_manifest": _identity(audit_hash_path),
        },
        "tokenizer": report["tokenizer"],
        "policy": {
            "fixed_max_seq_len": length_auditor.MAX_SEQ_LEN,
            "truncation": "forbidden",
            "selector": "disabled",
            "overlength": "exclude_from_candidate_keep_authoring",
        },
        "records": {
            "parent_candidate": parent_manifest["records"]["candidate"],
            "excluded_overlength": len(overflow_ids),
            "candidate": len(retained_ids),
            "by_evidence_source": dict(sorted(source_counts.items())),
        },
        "output": {
            "record_fields": sorted(rag_length_auditor.REQUIRED_RECORD_FIELDS),
            "candidate": candidate_identity,
        },
        "readiness": {
            "protocol_projection_ready": True,
            "evaluation_isolation_complete": isolated,
            "chat_template_length_audited": True,
            "overlength_records_isolated": True,
            "dataset_label_mask_audited": False,
            "training_ready": False,
        },
        "complete": True,
    }
    manifest_payload = json.dumps(
        manifest, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    _publish(candidate_output, candidate_payload, manifest_output, manifest_payload)
    return manifest


def main() -> None:
    """解析路径并派生固定 768 RAG-SFT candidate。"""

    parser = argparse.ArgumentParser(description="派生固定 768 的 canonical RAG-SFT candidate")
    parser.add_argument("--data-manifest", type=Path, default=DEFAULT_DATA_MANIFEST)
    parser.add_argument(
        "--length-audit-dir", type=Path, default=DEFAULT_LENGTH_AUDIT_DIR
    )
    parser.add_argument(
        "--candidate-output", type=Path, default=DEFAULT_CANDIDATE_OUTPUT
    )
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    args = parser.parse_args()
    try:
        manifest = derive_rag_sft_768(
            data_manifest=args.data_manifest,
            length_audit_dir=args.length_audit_dir,
            candidate_output=args.candidate_output,
            manifest_output=args.manifest_output,
        )
    except RagSftLengthDerivationError as error:
        parser.error(str(error))
    records = manifest["records"]
    print(
        f"[完成] 保留 {records['candidate']} 条，"
        f"排除超长 {records['excluded_overlength']} 条"
    )
    print("[阻断] labels 尚未审计，training_ready=false")
    print(f"data manifest: {args.manifest_output}")


if __name__ == "__main__":
    main()
