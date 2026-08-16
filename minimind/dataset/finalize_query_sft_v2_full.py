"""从冻结的 Query-SFT v2 Retrieval 选择发布可训练数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rag.query import QUERY_ENHANCEMENT_SYSTEM_PROMPT
from rag.query.enhancement import (
    QueryEnhancementProtocolError,
    parse_and_validate_query_enhancement,
)

from .finalize_query_sft_training_release import publish_training_release


DATASET_ROOT = Path(__file__).resolve().parent
POOL_ROOT = DATASET_ROOT / "QUERY-POOL" / "full"
DEFAULT_WORK_PACKAGE_DIR = POOL_ROOT / "query-sft-v2-work-package"
DEFAULT_TEACHER_RESULTS = (
    POOL_ROOT
    / "query-sft-v2-local-teacher-generation-r1"
    / "query-sft-v2-teacher-candidate-results.jsonl"
)
DEFAULT_TEACHER_DIR = POOL_ROOT / "query-sft-v2-local-teacher-generation-r1"
DEFAULT_SEMANTIC_DIR = POOL_ROOT / "query-sft-v2-semantic-quality-r2a"
DEFAULT_RETRIEVAL_DIR = POOL_ROOT / "query-sft-v2-retrieval-evaluation-r1"
DEFAULT_EVALUATION_MANIFEST = (
    DATASET_ROOT / "RAG-SFT" / "manifests" / "evaluation-exclusions-project-rag-v2.json"
)
DEFAULT_TOKENIZER_PATH = DATASET_ROOT.parent / "model"
DEFAULT_OUTPUT_DIR = POOL_ROOT / "query-sft-v2-training-release-r2"

INPUT_FILENAME = "query-sft-v2-input-candidate.jsonl"
NOOP_FILENAME = "query-sft-v2-deterministic-noop-candidates.jsonl"
TEACHER_FIELDS = ("candidate_id", "work_id", "raw_output")
INPUT_FIELDS = ("work_id", "source_id", "authoring_type", "query_original")
NOOP_FIELDS = ("candidate_id", "work_id", "raw_output")
AUTHORING_FILENAME = "query-sft-v2-authoring.jsonl"
CANDIDATE_FILENAME = "query-sft-v2-training-candidate.jsonl"
SELECTION_FILENAME = "query-sft-v2-formal-selection.json"
FINAL_MANIFEST_FILENAME = "query-sft-v2-formal-release.json"
HASH_FILENAME = "query-sft-v2-formal-release.sha256"
RETRIEVAL_FILENAMES = (
    "query-sft-v2-retrieval-records.jsonl",
    "query-sft-v2-retrieval-summary.json",
    "query-sft-v2-retrieval-manifest.json",
)


class QuerySftV2FullReleaseError(RuntimeError):
    """表示 v2 正式发布输入或质量门槛无效。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if records is not None:
        value["records"] = records
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftV2FullReleaseError(f"无法读取{label}") from error
    if not isinstance(value, dict):
        raise QuerySftV2FullReleaseError(f"{label}必须是 JSON 对象")
    return value


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    raise QuerySftV2FullReleaseError(f"{label}不允许空行：{number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise QuerySftV2FullReleaseError(f"{label}第 {number} 条必须是对象")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftV2FullReleaseError):
            raise
        raise QuerySftV2FullReleaseError(f"无法读取{label}") from error
    return rows


def _verify_sidecar(path: Path, label: str) -> Path:
    sidecar = path.with_suffix(".sha256")
    try:
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftV2FullReleaseError(f"无法读取{label} SHA-256") from error
    if lines != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftV2FullReleaseError(f"{label} SHA-256 无效")
    return sidecar


def _verify_retrieval(directory: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, object]]:
    hash_path = directory / "query-sft-v2-retrieval-manifest.sha256"
    try:
        entries = {
            name: digest
            for line in hash_path.read_text(encoding="utf-8").splitlines()
            for digest, name in [line.split("  ", 1)]
        }
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise QuerySftV2FullReleaseError("冻结 Retrieval 哈希清单无效") from error
    if set(entries) != set(RETRIEVAL_FILENAMES):
        raise QuerySftV2FullReleaseError("冻结 Retrieval 哈希清单范围无效")
    for name, digest in entries.items():
        path = directory / name
        if not path.is_file() or _sha256_file(path) != digest:
            raise QuerySftV2FullReleaseError(f"冻结 Retrieval 产物身份不匹配：{name}")
    summary = _read_json(directory / RETRIEVAL_FILENAMES[1], "冻结 Retrieval 汇总")
    manifest = _read_json(directory / RETRIEVAL_FILENAMES[2], "冻结 Retrieval manifest")
    records = summary.get("records")
    selections = summary.get("selections")
    expected = {
        "query_inputs": 574,
        "baseline_variants": 574,
        "noop_variants": 574,
        "approved_teacher_candidate_variants": 684,
        "total_retrieval_evaluations": 1832,
        "selected_by_kind": {"teacher_candidate": 448, "noop": 126},
    }
    if (
        manifest.get("pipeline") != "query_sft_v2_retrieval_evaluation"
        or manifest.get("complete") is not True
        or manifest.get("validation", {}).get(
            "all_queries_evaluated_with_baseline_noop_and_semantic_approved_candidates"
        ) is not True
        or manifest.get("validation", {}).get(
            "selection_uses_non_degrading_tie_allowed_v2_contract"
        ) is not True
        or not isinstance(records, dict)
        or records != expected
        or not isinstance(selections, list)
        or len(selections) != 574
    ):
        raise QuerySftV2FullReleaseError("冻结 Retrieval 结果未满足 v2 发布门槛")
    return summary, manifest, {
        "records": _identity(directory / RETRIEVAL_FILENAMES[0], records=1832),
        "summary": _identity(directory / RETRIEVAL_FILENAMES[1]),
        "manifest": _identity(directory / RETRIEVAL_FILENAMES[2]),
        "hash_manifest": _identity(hash_path),
    }


def _verify_directory_hash(
    directory: Path, hash_name: str, required_names: set[str], label: str
) -> Path:
    hash_path = directory / hash_name
    try:
        entries = {
            name: digest
            for line in hash_path.read_text(encoding="utf-8").splitlines()
            for digest, name in [line.split("  ", 1)]
        }
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise QuerySftV2FullReleaseError(f"{label}哈希清单无效") from error
    if not required_names <= set(entries):
        raise QuerySftV2FullReleaseError(f"{label}哈希清单缺少必要文件")
    for name, digest in entries.items():
        path = directory / name
        if not path.is_file() or _sha256_file(path) != digest:
            raise QuerySftV2FullReleaseError(f"{label}身份不匹配：{name}")
    return hash_path


def _canonical_target(raw_output: str, label: str) -> dict[str, object]:
    try:
        target = parse_and_validate_query_enhancement(raw_output)
    except (TypeError, QueryEnhancementProtocolError) as error:
        raise QuerySftV2FullReleaseError(f"{label}不符合 Query Enhancement 协议") from error
    return {
        "rewrite": target.rewrite,
        "expansion_terms": list(target.expansion_terms),
        "subqueries": list(target.subqueries),
    }


def _load_selected_targets(
    *,
    input_rows: list[dict[str, Any]],
    teacher_rows: list[dict[str, Any]],
    noop_rows: list[dict[str, Any]],
    approved_ids: set[str],
    selections: list[dict[str, Any]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], Counter[str]]:
    inputs: dict[str, dict[str, Any]] = {}
    for row in input_rows:
        work_id = row.get("work_id")
        if tuple(row) != INPUT_FIELDS or not isinstance(work_id, str) or work_id in inputs:
            raise QuerySftV2FullReleaseError("v2 输入候选字段或身份无效")
        inputs[work_id] = row
    if len(inputs) != 574:
        raise QuerySftV2FullReleaseError("v2 输入候选必须恰好为 574 条")
    teachers: dict[str, tuple[str, str]] = {}
    for row in teacher_rows:
        candidate_id = row.get("candidate_id")
        if (
            tuple(row) != TEACHER_FIELDS
            or not isinstance(candidate_id, str)
            or candidate_id in teachers
            or not isinstance(row.get("work_id"), str)
            or not isinstance(row.get("raw_output"), str)
        ):
            raise QuerySftV2FullReleaseError("教师候选字段或身份无效")
        teachers[candidate_id] = (row["work_id"], row["raw_output"])
    if len(teachers) != 1722:
        raise QuerySftV2FullReleaseError("教师候选必须恰好为 1,722 条")
    noops: dict[str, str] = {}
    for row in noop_rows:
        work_id = row.get("work_id")
        if (
            tuple(row) != NOOP_FIELDS
            or not isinstance(work_id, str)
            or work_id in noops
            or row.get("candidate_id") != f"{work_id}/noop"
            or not isinstance(row.get("raw_output"), str)
        ):
            raise QuerySftV2FullReleaseError("确定性 no-op 字段或身份无效")
        noops[work_id] = row["raw_output"]
    if set(noops) != set(inputs):
        raise QuerySftV2FullReleaseError("确定性 no-op 未覆盖全部 v2 输入")

    seen = set()
    selection_counts: Counter[str] = Counter()
    authoring: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    for selection in selections:
        work_id = selection.get("work_id")
        kind = selection.get("selection")
        variant_id = selection.get("selected_variant_id")
        if (
            not isinstance(work_id, str)
            or work_id not in inputs
            or work_id in seen
            or selection.get("authoring_type") != inputs[work_id]["authoring_type"]
            or not isinstance(variant_id, str)
        ):
            raise QuerySftV2FullReleaseError("冻结 Retrieval 选择与 v2 输入不一致")
        if kind == "teacher_candidate":
            teacher = teachers.get(variant_id)
            if teacher is None or teacher[0] != work_id or variant_id not in approved_ids:
                raise QuerySftV2FullReleaseError("选择的教师候选不是已批准的同题候选")
            target = _canonical_target(teacher[1], variant_id)
        elif kind == "noop":
            if variant_id != f"{work_id}/noop":
                raise QuerySftV2FullReleaseError("选择的 no-op 标识无效")
            target = _canonical_target(noops[work_id], variant_id)
        else:
            raise QuerySftV2FullReleaseError("冻结 Retrieval 选择类型无效")
        source = inputs[work_id]
        authoring.append(
            {
                "id": work_id,
                "source_id": source["source_id"],
                "query_original": source["query_original"],
                "target": target,
            }
        )
        candidates.append(
            {
                "id": work_id,
                "source": "query_sft",
                "conversations": [
                    {"role": "system", "content": QUERY_ENHANCEMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": source["query_original"]},
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            target,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            allow_nan=False,
                        ),
                    },
                ],
            }
        )
        seen.add(work_id)
        selection_counts[kind] += 1
    if seen != set(inputs) or selection_counts != Counter({"teacher_candidate": 448, "noop": 126}):
        raise QuerySftV2FullReleaseError("冻结 Retrieval 选择未精确覆盖预期的 574 条输入")
    return sorted(authoring, key=lambda row: row["id"]), sorted(candidates, key=lambda row: row["id"]), selection_counts


def _jsonl_payload(rows: list[dict[str, object]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in rows
    )


def _write_payload(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf-8", newline="\n")
    path.with_suffix(".sha256").write_text(
        f"{_sha256_file(path)}  {path.name}\n", encoding="utf-8", newline="\n"
    )


def finalize_query_sft_v2_full(
    *,
    work_package_dir: Path = DEFAULT_WORK_PACKAGE_DIR,
    teacher_results_path: Path = DEFAULT_TEACHER_RESULTS,
    semantic_dir: Path = DEFAULT_SEMANTIC_DIR,
    retrieval_dir: Path = DEFAULT_RETRIEVAL_DIR,
    evaluation_manifest_path: Path = DEFAULT_EVALUATION_MANIFEST,
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    """发布带完整证据链、长度审计和 label 审计的 Query-SFT v2。"""

    work_package_dir = Path(work_package_dir).resolve()
    teacher_results_path = Path(teacher_results_path).resolve()
    teacher_dir = teacher_results_path.parent
    semantic_dir = Path(semantic_dir).resolve()
    retrieval_dir = Path(retrieval_dir).resolve()
    evaluation_manifest_path = Path(evaluation_manifest_path).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise QuerySftV2FullReleaseError("v2 正式发布输出目录必须不存在")

    summary, retrieval_manifest, retrieval_identity = _verify_retrieval(retrieval_dir)
    input_path = work_package_dir / INPUT_FILENAME
    teacher_work_package_dir = work_package_dir / "query-sft-v2-teacher-candidate-work-package"
    noop_path = teacher_work_package_dir / NOOP_FILENAME
    ledger_path = semantic_dir / "query-sft-v2-semantic-review-ledger.jsonl"
    semantic_audit_path = semantic_dir / "query-sft-v2-semantic-quality-audit.json"
    text_audit_path = semantic_dir / "query-sft-v2-candidate-text-audit.json"
    _verify_directory_hash(
        work_package_dir,
        "query-sft-v2-work-package.sha256",
        {INPUT_FILENAME, "query-sft-v2-work-package.json"},
        "v2 工作包",
    )
    _verify_directory_hash(
        teacher_work_package_dir,
        "query-sft-v2-teacher-candidate-work-package.sha256",
        {NOOP_FILENAME, "query-sft-v2-teacher-candidate-work-package.json"},
        "v2 教师工作包",
    )
    for path, label in (
        (teacher_results_path, "教师候选结果"),
        (teacher_dir / "query-sft-v2-teacher-generation-run.json", "教师生成运行记录"),
        (teacher_dir / "query-sft-v2-teacher-candidate-protocol-audit.json", "教师候选协议审核"),
        (ledger_path, "语义审核账本"),
        (semantic_audit_path, "语义质量审核"),
        (text_audit_path, "候选文本审核"),
        (evaluation_manifest_path, "评估隔离 manifest"),
    ):
        _verify_sidecar(path, label)
    semantic_audit = _read_json(semantic_audit_path, "语义质量审核")
    text_audit = _read_json(text_audit_path, "候选文本审核")
    teacher_protocol_audit_path = teacher_dir / "query-sft-v2-teacher-candidate-protocol-audit.json"
    teacher_protocol_audit = _read_json(teacher_protocol_audit_path, "教师候选协议审核")
    if (
        teacher_protocol_audit.get("pipeline")
        != "query_sft_v2_teacher_candidate_protocol_audit"
        or teacher_protocol_audit.get("records", {}).get("teacher_candidates") != 1722
        or teacher_protocol_audit.get("validation", {}).get(
            "teacher_work_package_identity_bound"
        ) is not True
        or teacher_protocol_audit.get("validation", {}).get(
            "teacher_runtime_identity_bound"
        ) is not True
        or teacher_protocol_audit.get("validation", {}).get(
            "all_teacher_requests_returned_once"
        ) is not True
        or teacher_protocol_audit.get("validation", {}).get(
            "all_raw_outputs_strict_protocol_valid"
        ) is not True
        or teacher_protocol_audit.get("validation", {}).get(
            "required_gt_not_read_or_emitted"
        ) is not True
        or teacher_protocol_audit.get("complete") is not True
        or
        semantic_audit.get("pipeline") != "query_sft_v2_semantic_quality_audit"
        or semantic_audit.get("records", {}).get("approved") != 684
        or semantic_audit.get("validation", {}).get("independent_semantic_review_complete") is not True
        or text_audit.get("pipeline") != "query_sft_v2_candidate_text_audit"
        or text_audit.get("validation", {}).get("approved_targets_unique_across_work_ids") is not True
        or text_audit.get("validation", {}).get("approved_target_texts_excluded_from_evaluation") is not True
    ):
        raise QuerySftV2FullReleaseError("语义审核或文本隔离门槛未通过")
    ledger_rows = _read_jsonl(ledger_path, "语义审核账本")
    approved_ids = {
        row["candidate_id"]
        for row in ledger_rows
        if row.get("review_decision") == "approved" and isinstance(row.get("candidate_id"), str)
    }
    if len(ledger_rows) != 1722 or len(approved_ids) != 684:
        raise QuerySftV2FullReleaseError("语义审核账本计数无效")

    authoring, candidates, selection_counts = _load_selected_targets(
        input_rows=_read_jsonl(input_path, "v2 输入候选"),
        teacher_rows=_read_jsonl(teacher_results_path, "教师候选结果"),
        noop_rows=_read_jsonl(noop_path, "确定性 no-op"),
        approved_ids=approved_ids,
        selections=summary["selections"],
    )
    output_dir.mkdir(parents=True)
    authoring_path = output_dir / AUTHORING_FILENAME
    candidate_path = output_dir / CANDIDATE_FILENAME
    selection_path = output_dir / SELECTION_FILENAME
    _write_payload(authoring_path, _jsonl_payload(authoring))
    _write_payload(candidate_path, _jsonl_payload(candidates))
    selection_payload = {
        "schema_version": "1.0",
        "pipeline": "query_sft_v2_frozen_selection_projection",
        "inputs": {"retrieval": retrieval_identity},
        "records": {
            "query_inputs": 574,
            "authoring": 574,
            "training_candidates": 574,
            "selected_by_kind": dict(selection_counts),
        },
        "validation": {
            "only_semantic_approved_teacher_candidates_selected": True,
            "noop_fallback_is_deterministic": True,
            "required_gt_not_emitted_to_training": True,
        },
        "complete": True,
    }
    _write_payload(
        selection_path,
        json.dumps(selection_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    training_release = publish_training_release(
        candidate_path=candidate_path,
        evaluation_manifest_path=evaluation_manifest_path,
        tokenizer_path=tokenizer_path,
        output_dir=output_dir,
        release_status="formal_training_candidate",
    )
    final_path = output_dir / FINAL_MANIFEST_FILENAME
    final_payload = {
        "schema_version": "1.0",
        "pipeline": "query_sft_v2_formal_release",
        "release_status": "formal_training_ready",
        "inputs": {
            "work_package": _identity(work_package_dir / "query-sft-v2-work-package.json"),
            "work_package_hash_manifest": _identity(work_package_dir / "query-sft-v2-work-package.sha256"),
            "teacher_candidates": _identity(teacher_results_path, records=1722),
            "teacher_generation_run": _identity(
                teacher_dir / "query-sft-v2-teacher-generation-run.json"
            ),
            "teacher_candidate_protocol_audit": _identity(teacher_protocol_audit_path),
            "semantic_ledger": _identity(ledger_path, records=1722),
            "semantic_quality_audit": _identity(semantic_audit_path),
            "candidate_text_audit": _identity(text_audit_path),
            "retrieval": retrieval_identity,
            "evaluation_exclusions": _identity(evaluation_manifest_path),
        },
        "outputs": {
            "authoring": _identity(authoring_path, records=574),
            "training_candidate": _identity(candidate_path, records=574),
            "selection": _identity(selection_path),
            "training_release": _identity(output_dir / "query-sft-training-release.json"),
            "chat_length_audit": _identity(output_dir / "query-sft-chat-length-audit-768.json"),
            "label_audit": _identity(output_dir / "query-sft-label-audit-768.json"),
        },
        "records": {
            "training": 574,
            "selected_by_kind": dict(selection_counts),
            "assistant_tokens_per_epoch": training_release["records"]["assistant_tokens_per_epoch"],
        },
        "policy": {
            "fixed_max_seq_len": 768,
            "truncation": "forbidden",
            "assistant_only_labels": True,
            "semantic_approved_ties_allowed": True,
            "retrieval_non_degrading": True,
        },
        "readiness": {
            "frozen_retrieval_selection_complete": True,
            "chat_template_length_audited": True,
            "dataset_label_mask_audited": True,
            "evaluation_isolation_complete": True,
            "training_ready": True,
        },
        "complete": True,
    }
    _write_payload(final_path, json.dumps(final_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (output_dir / HASH_FILENAME).write_text(
        "".join(
            f"{_sha256_file(path)}  {path.name}\n"
            for path in (
                authoring_path,
                candidate_path,
                selection_path,
                output_dir / "query-sft-chat-length-audit-768.json",
                output_dir / "query-sft-label-audit-768.json",
                output_dir / "query-sft-training-release.json",
                final_path,
            )
        ),
        encoding="utf-8",
        newline="\n",
    )
    return final_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-package-dir", type=Path, default=DEFAULT_WORK_PACKAGE_DIR)
    parser.add_argument("--teacher-results", type=Path, default=DEFAULT_TEACHER_RESULTS)
    parser.add_argument("--semantic-dir", type=Path, default=DEFAULT_SEMANTIC_DIR)
    parser.add_argument("--retrieval-dir", type=Path, default=DEFAULT_RETRIEVAL_DIR)
    parser.add_argument("--evaluation-manifest", type=Path, default=DEFAULT_EVALUATION_MANIFEST)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        result = finalize_query_sft_v2_full(
            work_package_dir=args.work_package_dir,
            teacher_results_path=args.teacher_results,
            semantic_dir=args.semantic_dir,
            retrieval_dir=args.retrieval_dir,
            evaluation_manifest_path=args.evaluation_manifest,
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError, QuerySftV2FullReleaseError) as error:
        parser.error(str(error))
    print(
        "QUERY_SFT_V2_FORMAL_RELEASE_OK "
        f"records={result['records']['training']} "
        f"assistant_tokens={result['records']['assistant_tokens_per_epoch']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
