"""冻结项目 RAG 开发集问题，生成法律 SFT 精确排除清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

try:
    from .audit_disc_law_sft import normalize_whitespace, sha256_file
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset.audit_disc_law_sft import normalize_whitespace, sha256_file


MINIMIND_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = MINIMIND_ROOT.parent
DEFAULT_WORK_ROOT = Path(os.environ.get("MINIMIND_WORK_ROOT", "/root/autodl-tmp/minimind-work"))
DEFAULT_PROJECT_RAG_EVAL = PROJECT_ROOT / "rag" / "eval" / "eval_set.jsonl"
DEFAULT_OUTPUT = DEFAULT_WORK_ROOT / "manifests" / "sft-evaluation-exclusions-project-rag-v1.json"

DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
QUERY_FIELDS = ("query_original", "query_formal")


class EvaluationExclusionError(RuntimeError):
    """表示评估排除清单无法安全生成。"""


def audit_private_holdout_compatibility(**kwargs):
    """延迟加载私有留出集审计器，保持仅查看 CLI 帮助时不加载项目模块。"""

    from rag.eval.private_holdout import audit_private_holdout_compatibility as audit

    return audit(**kwargs)


def question_digest(question: str) -> str:
    """对 NFC 和空白归一后的问题计算 SHA-256。"""

    normalized = normalize_whitespace(question)
    if not normalized:
        raise EvaluationExclusionError("评估问题不能为空")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _write_manifest(manifest: dict[str, object], output_path: Path) -> None:
    output_path = output_path.resolve()
    hash_path = output_path.with_suffix(".sha256")
    partial_path = output_path.with_name(output_path.name + ".partial")
    hash_partial_path = hash_path.with_name(hash_path.name + ".partial")
    occupied = [path.name for path in (output_path, hash_path, partial_path, hash_partial_path) if path.exists()]
    if occupied:
        raise EvaluationExclusionError("输出位置已有评估排除产物: " + ", ".join(occupied))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        partial_path.write_text(payload, encoding="utf-8", newline="\n")
        json.loads(partial_path.read_text(encoding="utf-8"))
        digest = sha256_file(partial_path)
        hash_partial_path.write_text(
            f"{digest}  {output_path.name}\n", encoding="utf-8", newline="\n"
        )
        partial_path.replace(output_path)
        hash_partial_path.replace(hash_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise EvaluationExclusionError(f"无法发布评估排除清单: {output_path}") from error


def build_project_rag_exclusions(
    project_rag_eval: Path,
    output_path: Path,
) -> dict[str, object]:
    """冻结项目 RAG 开发集，并标记项目私有留出集尚未冻结。"""

    project_rag_eval = project_rag_eval.resolve()
    if not project_rag_eval.is_file():
        raise EvaluationExclusionError(f"项目 RAG 评估集不存在: {project_rag_eval}")

    record_count = 0
    seen_ids: set[str] = set()
    digests: set[str] = set()
    try:
        with project_rag_eval.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise EvaluationExclusionError(
                        f"项目 RAG 评估集 JSON 无效: {project_rag_eval}:{line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise EvaluationExclusionError(
                        f"项目 RAG 评估记录不是对象: {project_rag_eval}:{line_number}"
                    )
                record_id = record.get("id")
                if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
                    raise EvaluationExclusionError(
                        f"项目 RAG 评估 ID 缺失或重复: {project_rag_eval}:{line_number}"
                    )
                seen_ids.add(record_id)
                for field_name in QUERY_FIELDS:
                    value = record.get(field_name)
                    if field_name == "query_formal" and value is None:
                        continue
                    if not isinstance(value, str) or not value.strip():
                        raise EvaluationExclusionError(
                            f"项目 RAG 评估字段 {field_name} 无效: "
                            f"{project_rag_eval}:{line_number}"
                        )
                    digests.add(question_digest(value))
                record_count += 1
    except (OSError, UnicodeDecodeError) as error:
        raise EvaluationExclusionError(f"无法读取项目 RAG 评估集: {project_rag_eval}") from error

    if not record_count or not digests:
        raise EvaluationExclusionError("项目 RAG 评估集不能为空")
    manifest: dict[str, object] = {
        "schema_version": "1.1",
        "pipeline": "legal_sft_evaluation_exclusions",
        "scope": "project_rag_development_only",
        "normalization": "unicode_nfc_trim_and_collapse_whitespace_then_sha256",
        "assets": [
            {
                "name": "project_rag_eval",
                "path": str(project_rag_eval),
                "bytes": project_rag_eval.stat().st_size,
                "sha256": sha256_file(project_rag_eval),
                "records": record_count,
                "question_fields": list(QUERY_FIELDS),
            }
        ],
        "question_sha256": sorted(digests),
        "digest_count": len(digests),
        "complete_for_formal_sft": False,
        "missing_required_assets": ["project_private_holdout"],
        "optional_assets_not_enabled": ["LawBench", "STARD"],
        "raw_questions_included": False,
    }
    _write_manifest(manifest, output_path)
    return manifest


def _collect_question_asset(
    path: Path,
    *,
    label: str,
    asset_name: str,
    allow_null_formal: bool,
) -> tuple[dict[str, object], set[str]]:
    path = path.resolve()
    if not path.is_file():
        raise EvaluationExclusionError(f"{label}不存在: {path}")
    record_count = 0
    seen_ids: set[str] = set()
    digests: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise EvaluationExclusionError(
                        f"{label} JSON 无效: {path}:{line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise EvaluationExclusionError(
                        f"{label}记录不是对象: {path}:{line_number}"
                    )
                record_id = record.get("id")
                if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
                    raise EvaluationExclusionError(
                        f"{label} ID 缺失或重复: {path}:{line_number}"
                    )
                seen_ids.add(record_id)
                for field_name in QUERY_FIELDS:
                    value = record.get(field_name)
                    if field_name == "query_formal" and value is None and allow_null_formal:
                        continue
                    if not isinstance(value, str) or not value.strip():
                        raise EvaluationExclusionError(
                            f"{label}字段 {field_name} 无效: {path}:{line_number}"
                        )
                    digests.add(question_digest(value))
                record_count += 1
    except (OSError, UnicodeDecodeError) as error:
        raise EvaluationExclusionError(f"无法读取{label}: {path}") from error
    if not record_count or not digests:
        raise EvaluationExclusionError(f"{label}不能为空")
    return (
        {
            "name": asset_name,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "records": record_count,
            "question_fields": list(QUERY_FIELDS),
        },
        digests,
    )


def build_complete_project_evaluation_exclusions(
    *,
    project_rag_eval: Path,
    private_holdout: Path,
    article_index: Path,
    rag_authoring: Path,
    output_path: Path,
) -> dict[str, object]:
    """发布开发集与私有留出集共同构成的正式训练排除清单。"""

    project_rag_eval = project_rag_eval.resolve()
    private_holdout = private_holdout.resolve()
    article_index = article_index.resolve()
    rag_authoring = rag_authoring.resolve()
    for label, path in (
        ("法条索引", article_index),
        ("RAG SFT authoring", rag_authoring),
    ):
        if not path.is_file():
            raise EvaluationExclusionError(f"{label}不存在: {path}")

    development_asset, development_digests = _collect_question_asset(
        project_rag_eval,
        label="项目 RAG 开发集",
        asset_name="project_rag_eval",
        allow_null_formal=True,
    )
    private_asset, private_digests = _collect_question_asset(
        private_holdout,
        label="项目私有留出集",
        asset_name="project_private_holdout_v2",
        allow_null_formal=False,
    )
    audit = audit_private_holdout_compatibility(
        authoring_path=private_holdout,
        article_index_path=article_index,
        development_eval_path=project_rag_eval,
        rag_authoring_path=rag_authoring,
    )
    overlap_counts = audit.get("overlap_counts")
    if audit.get("compatible") is not True or not isinstance(overlap_counts, dict) or any(
        value != 0 for value in overlap_counts.values()
    ):
        raise EvaluationExclusionError("私有留出集与开发或训练数据存在交集")
    if audit.get("raw_text_included") is not False:
        raise EvaluationExclusionError("兼容性审计报告不得包含私有原文")

    audit_inputs = audit.get("inputs")
    if not isinstance(audit_inputs, dict):
        raise EvaluationExclusionError("兼容性审计缺少输入身份")
    expected_hashes = {
        "authoring": sha256_file(private_holdout),
        "article_index": sha256_file(article_index),
        "development_eval": sha256_file(project_rag_eval),
        "rag_authoring": sha256_file(rag_authoring),
    }
    reported_hashes = {
        name: value.get("sha256") if isinstance(value, dict) else None
        for name, value in audit_inputs.items()
    }
    if reported_hashes != expected_hashes:
        raise EvaluationExclusionError("兼容性审计输入哈希与当前文件不一致")

    digests = development_digests | private_digests
    manifest: dict[str, object] = {
        "schema_version": "1.2",
        "pipeline": "legal_sft_evaluation_exclusions",
        "scope": "project_rag_development_and_private_holdout",
        "normalization": "unicode_nfc_trim_and_collapse_whitespace_then_sha256",
        "assets": [development_asset, private_asset],
        "question_sha256": sorted(digests),
        "digest_count": len(digests),
        "isolation_audit": {
            "pipeline": audit.get("pipeline"),
            "compatible": True,
            "input_sha256": expected_hashes,
            "private": audit.get("private"),
            "overlap_counts": overlap_counts,
            "raw_text_included": False,
        },
        "complete_for_formal_sft": True,
        "missing_required_assets": [],
        "optional_assets_not_enabled": ["LawBench", "STARD"],
        "raw_questions_included": False,
    }
    _write_manifest(manifest, output_path)
    return manifest


def main() -> None:
    """解析命令行并冻结项目自有评估问题。"""

    parser = argparse.ArgumentParser(description="生成法律 SFT 项目 RAG 评估问题排除清单")
    parser.add_argument(
        "--project-rag-eval",
        type=Path,
        default=DEFAULT_PROJECT_RAG_EVAL,
        help="项目自有 RAG 评估 JSONL",
    )
    parser.add_argument("--private-holdout", type=Path)
    parser.add_argument("--article-index", type=Path)
    parser.add_argument("--rag-authoring", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="新的排除清单 JSON")
    args = parser.parse_args()
    try:
        complete_inputs = (args.private_holdout, args.article_index, args.rag_authoring)
        if any(item is not None for item in complete_inputs):
            if any(item is None for item in complete_inputs):
                raise EvaluationExclusionError(
                    "完整排除清单必须同时指定 private-holdout、article-index 和 rag-authoring"
                )
            manifest = build_complete_project_evaluation_exclusions(
                project_rag_eval=args.project_rag_eval,
                private_holdout=args.private_holdout,
                article_index=args.article_index,
                rag_authoring=args.rag_authoring,
                output_path=args.output,
            )
        else:
            manifest = build_project_rag_exclusions(args.project_rag_eval, args.output)
    except (EvaluationExclusionError, OSError, ValueError) as error:
        raise SystemExit(f"[失败] {error}") from error
    print(f"[完成] 生成 {manifest['digest_count']:,} 个评估问题摘要")
    if manifest["complete_for_formal_sft"]:
        print("[完成] 开发集、私有留出集与当前 RAG SFT 隔离已闭合")
    else:
        print("[阻断] 项目私有留出集尚未冻结，本清单不能标记正式 SFT 数据就绪")
    print("[信息] LawBench/STARD 为未启用的非阻断外部诊断")
    print(f"排除清单: {args.output}")


if __name__ == "__main__":
    main()
