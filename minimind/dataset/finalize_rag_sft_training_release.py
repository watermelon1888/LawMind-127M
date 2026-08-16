"""验证全部 RAG-SFT 数据关卡并发布最终训练候选 manifest。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


RAG_SFT_ROOT = Path(__file__).resolve().parent / "RAG-SFT"
DEFAULT_AUTHORING_MANIFEST = RAG_SFT_ROOT / "manifests" / "rag-sft-training-v1.json"
DEFAULT_CANDIDATE_MANIFEST = (
    RAG_SFT_ROOT / "manifests" / "rag-sft-training-v1-final-candidate.json"
)
DEFAULT_FIXED_MANIFEST = (
    RAG_SFT_ROOT / "manifests" / "rag-sft-training-v1-final-manifest-768.json"
)
DEFAULT_EVALUATION_EXCLUSIONS = (
    RAG_SFT_ROOT / "manifests" / "evaluation-exclusions-project-rag-v2.json"
)
DEFAULT_LENGTH_REPORT = (
    RAG_SFT_ROOT
    / "reports"
    / "rag-sft-training-v1-final-chat-length-audit-768"
    / "rag-sft-chat-length-audit-768.json"
)
DEFAULT_LABEL_REPORT = (
    RAG_SFT_ROOT
    / "reports"
    / "rag-sft-training-v1-final-label-audit-768"
    / "rag-sft-dataset-label-audit-768.json"
)
DEFAULT_RUNTIME_REPORT = (
    RAG_SFT_ROOT
    / "reports"
    / "rag-sft-training-v1-final-runtime-budget-audit-160"
    / "rag-sft-runtime-budget-audit-160.json"
)
DEFAULT_OUTPUT = RAG_SFT_ROOT / "manifests" / "rag-sft-training-v1-release.json"


class RagSftReleaseError(RuntimeError):
    """表示 RAG-SFT 最终训练发布条件未闭合。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftReleaseError(f"无法读取{label}: {path}") from error
    if not isinstance(value, dict):
        raise RagSftReleaseError(f"{label}必须是 JSON 对象: {path}")
    return value


def _identity(path: Path) -> dict[str, object]:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _verify_sidecar(path: Path) -> Path:
    sidecar = path.with_suffix(".sha256")
    try:
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftReleaseError(f"无法读取 SHA-256 清单: {sidecar}") from error
    if not lines:
        raise RagSftReleaseError(f"SHA-256 清单为空: {sidecar}")
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise RagSftReleaseError(f"SHA-256 清单格式无效: {sidecar}")
        expected, filename = parts[0], parts[1].strip()
        target = sidecar.parent / filename
        if not target.is_file() or _sha256_file(target) != expected:
            raise RagSftReleaseError(f"SHA-256 复算失败: {target}")
    return sidecar


def _verify_embedded_identity(identity: object, label: str) -> Path:
    if not isinstance(identity, dict):
        raise RagSftReleaseError(f"{label}身份无效")
    path_value = identity.get("path")
    if not isinstance(path_value, str):
        raise RagSftReleaseError(f"{label}缺少路径")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise RagSftReleaseError(f"{label}文件不存在: {path}")
    if identity.get("bytes") != path.stat().st_size or identity.get("sha256") != _sha256_file(path):
        raise RagSftReleaseError(f"{label}身份与当前文件不一致")
    return path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RagSftReleaseError(message)


def _publish(manifest: dict[str, object], output_path: Path) -> None:
    output_path = output_path.resolve()
    hash_path = output_path.with_suffix(".sha256")
    partial = output_path.with_name(output_path.name + ".partial")
    hash_partial = hash_path.with_name(hash_path.name + ".partial")
    occupied = [
        str(path) for path in (output_path, hash_path, partial, hash_partial) if path.exists()
    ]
    if occupied:
        raise RagSftReleaseError("输出位置已有发布产物: " + ", ".join(occupied))
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(payload, encoding="utf-8", newline="\n")
        digest = _sha256_file(partial)
        hash_partial.write_text(
            f"{digest}  {output_path.name}\n", encoding="utf-8", newline="\n"
        )
        partial.replace(output_path)
        hash_partial.replace(hash_path)
    except OSError as error:
        partial.unlink(missing_ok=True)
        hash_partial.unlink(missing_ok=True)
        raise RagSftReleaseError("无法发布 RAG-SFT 最终 manifest") from error


def finalize_release(
    *,
    authoring_manifest: Path,
    candidate_manifest: Path,
    fixed_manifest: Path,
    evaluation_exclusions: Path,
    length_report: Path,
    label_report: Path,
    runtime_report: Path,
    output_path: Path,
) -> dict[str, object]:
    """复算输入身份，闭合关卡并发布最终训练候选 manifest。"""

    output_path = Path(output_path).resolve()
    if any(
        path.exists()
        for path in (
            output_path,
            output_path.with_suffix(".sha256"),
            output_path.with_name(output_path.name + ".partial"),
            output_path.with_suffix(".sha256").with_name(
                output_path.with_suffix(".sha256").name + ".partial"
            ),
        )
    ):
        raise RagSftReleaseError("输出位置已有发布产物")

    paths = {
        "authoring_manifest": Path(authoring_manifest).resolve(),
        "candidate_manifest": Path(candidate_manifest).resolve(),
        "fixed_manifest": Path(fixed_manifest).resolve(),
        "evaluation_exclusions": Path(evaluation_exclusions).resolve(),
        "length_report": Path(length_report).resolve(),
        "label_report": Path(label_report).resolve(),
        "runtime_report": Path(runtime_report).resolve(),
    }
    sidecars = {name: _verify_sidecar(path) for name, path in paths.items()}
    authoring = _load_json(paths["authoring_manifest"], "authoring 汇编 manifest")
    candidate = _load_json(paths["candidate_manifest"], "完整 candidate manifest")
    fixed = _load_json(paths["fixed_manifest"], "固定 768 manifest")
    exclusions = _load_json(paths["evaluation_exclusions"], "评估排除 manifest")
    length = _load_json(paths["length_report"], "长度审计报告")
    labels = _load_json(paths["label_report"], "labels 审计报告")
    runtime = _load_json(paths["runtime_report"], "运行时预算审计报告")

    authoring_file = _verify_embedded_identity(
        authoring.get("output", {}).get("authoring") if isinstance(authoring.get("output"), dict) else None,
        "authoring",
    )
    full_candidate = _verify_embedded_identity(
        candidate.get("output", {}).get("candidate") if isinstance(candidate.get("output"), dict) else None,
        "完整 candidate",
    )
    fixed_candidate = _verify_embedded_identity(
        fixed.get("output", {}).get("candidate") if isinstance(fixed.get("output"), dict) else None,
        "固定 768 candidate",
    )

    authoring_sha = _sha256_file(authoring_file)
    full_candidate_sha = _sha256_file(full_candidate)
    fixed_candidate_sha = _sha256_file(fixed_candidate)
    _require(candidate.get("release_status") == "formal_candidate", "完整 candidate 尚非正式状态")
    _require(candidate.get("inputs", {}).get("authoring", {}).get("sha256") == authoring_sha, "candidate 未绑定当前 authoring")
    _require(fixed.get("authoring", {}).get("sha256") == authoring_sha, "固定 768 未绑定当前 authoring")
    _require(fixed.get("parent_candidate", {}).get("sha256") == full_candidate_sha, "固定 768 未绑定当前完整 candidate")
    _require(length.get("input", {}).get("candidate", {}).get("sha256") == full_candidate_sha, "长度报告未绑定当前完整 candidate")
    _require(labels.get("input", {}).get("candidate", {}).get("sha256") == fixed_candidate_sha, "labels 报告未绑定当前固定 768 candidate")
    _require(runtime.get("input", {}).get("candidate", {}).get("sha256") == full_candidate_sha, "预算报告未绑定当前完整 candidate")

    authoring_total = authoring.get("records", {}).get("total")
    full_total = candidate.get("records", {}).get("candidate")
    fixed_records = fixed.get("records", {})
    fixed_total = fixed_records.get("candidate")
    excluded_overlength = fixed_records.get("excluded_overlength")
    label_totals = labels.get("records", {}).get("totals", {})
    behavior = labels.get("records", {}).get("by_behavior", {})
    _require(authoring_total == full_total == fixed_total + excluded_overlength, "833→832 记录计数未闭合")
    _require(label_totals.get("records") == fixed_total, "labels 记录数与固定 768 candidate 不一致")

    overlap_counts = exclusions.get("isolation_audit", {}).get("overlap_counts", {})
    gates = {
        "authoring_approved": authoring.get("validation", {}).get("all_records_approved") is True,
        "model_visible_conversations_unique": authoring.get("validation", {}).get("model_visible_conversations_unique") is True,
        "evaluation_isolation_complete": exclusions.get("complete_for_formal_sft") is True
        and exclusions.get("isolation_audit", {}).get("compatible") is True
        and isinstance(overlap_counts, dict)
        and bool(overlap_counts)
        and all(value == 0 for value in overlap_counts.values()),
        "protocol_projection_ready": candidate.get("readiness", {}).get("protocol_projection_ready") is True,
        "chat_template_length_audited": fixed.get("readiness", {}).get("chat_template_length_audited") is True,
        "overlength_records_isolated": fixed.get("readiness", {}).get("overlength_records_isolated") is True,
        "dataset_label_mask_audited": labels.get("readiness", {}).get("dataset_label_mask_audited") is True
        and labels.get("validation", {}).get("counts_closed") is True
        and labels.get("validation", {}).get("zero_active_label_records") == 0
        and labels.get("validation", {}).get("input_side_label_leak_records") == 0,
        "runtime_budget_160_audited": runtime.get("readiness", {}).get("canonical_budget_160_audited") is True
        and runtime.get("scope", {}).get("primary_max_output_tokens") == 160
        and runtime.get("validation", {}).get("counts_closed") is True,
    }
    _require(all(gates.values()), "至少一个最终训练关卡未闭合")

    manifest = {
        "pipeline": "rag_sft_training_release",
        "release_status": "formal_training_candidate",
        "artifacts": {
            name: {
                "file": _identity(path),
                "sha256_manifest": _identity(sidecars[name]),
            }
            for name, path in paths.items()
        },
        "data": {
            "authoring": _identity(authoring_file),
            "full_candidate": _identity(full_candidate),
            "fixed_768_candidate": _identity(fixed_candidate),
        },
        "protocol": candidate.get("protocol"),
        "records": {
            "authoring": authoring_total,
            "full_candidate": full_total,
            "fixed_768_candidate": fixed_total,
            "excluded_overlength": excluded_overlength,
            "answers": behavior.get("answer", {}).get("records"),
            "refusals": behavior.get("refusal", {}).get("records"),
            "by_evidence_source": fixed_records.get("by_evidence_source"),
        },
        "evaluation": {
            "digest_count": exclusions.get("digest_count"),
            "assets": exclusions.get("assets"),
            "overlap_counts": overlap_counts,
            "raw_questions_included": exclusions.get("raw_questions_included"),
        },
        "readiness": {**gates, "training_ready": True},
        "complete": True,
    }
    _publish(manifest, output_path)
    return manifest


def main() -> None:
    """解析最终发布输入并输出 manifest。"""

    parser = argparse.ArgumentParser(description="发布最终 RAG-SFT 训练候选 manifest")
    parser.add_argument("--authoring-manifest", type=Path, default=DEFAULT_AUTHORING_MANIFEST)
    parser.add_argument("--candidate-manifest", type=Path, default=DEFAULT_CANDIDATE_MANIFEST)
    parser.add_argument("--fixed-manifest", type=Path, default=DEFAULT_FIXED_MANIFEST)
    parser.add_argument("--evaluation-exclusions", type=Path, default=DEFAULT_EVALUATION_EXCLUSIONS)
    parser.add_argument("--length-report", type=Path, default=DEFAULT_LENGTH_REPORT)
    parser.add_argument("--label-report", type=Path, default=DEFAULT_LABEL_REPORT)
    parser.add_argument("--runtime-report", type=Path, default=DEFAULT_RUNTIME_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        manifest = finalize_release(
            authoring_manifest=args.authoring_manifest,
            candidate_manifest=args.candidate_manifest,
            fixed_manifest=args.fixed_manifest,
            evaluation_exclusions=args.evaluation_exclusions,
            length_report=args.length_report,
            label_report=args.label_report,
            runtime_report=args.runtime_report,
            output_path=args.output,
        )
    except (RagSftReleaseError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
