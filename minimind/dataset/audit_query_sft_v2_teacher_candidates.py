"""审计 Query-SFT v2 教师候选结果、运行身份与工作包覆盖关系。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rag.query.enhancement import parse_and_validate_query_enhancement

from .prepare_query_sft_v2_teacher_candidates import (
    DEFAULT_OUTPUT_DIR,
    HASH_FILENAME as WORK_HASH_FILENAME,
    MANIFEST_FILENAME as WORK_MANIFEST_FILENAME,
    NOOP_FILENAME,
)


RESULT_FILENAME = "query-sft-v2-teacher-candidate-results.jsonl"
RUNTIME_FILENAME = "query-sft-v2-teacher-generation-run.json"
AUDIT_FILENAME = "query-sft-v2-teacher-candidate-protocol-audit.json"
QUEUE_FIELDS = ("candidate_id", "work_id", "query_original")
RESULT_FIELDS = ("candidate_id", "work_id", "raw_output")
RUNTIME_IDENTITY_FIELDS = (
    "teacher_model_id",
    "teacher_endpoint_or_local_checkpoint",
    "decoding_temperature",
    "decoding_seed_or_nondeterminism_declaration",
    "max_output_tokens",
)


class QuerySftV2TeacherCandidateAuditError(RuntimeError):
    """表示教师候选结果不满足 Query-SFT v2 的可审计协议。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    identity: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if records is not None:
        identity["records"] = records
    return identity


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftV2TeacherCandidateAuditError(f"无法读取{label}") from error
    if not isinstance(value, dict):
        raise QuerySftV2TeacherCandidateAuditError(f"{label}必须是 JSON 对象")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    raise QuerySftV2TeacherCandidateAuditError(f"{label}不允许空行: {number}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise QuerySftV2TeacherCandidateAuditError(f"{label}第 {number} 条必须是对象")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftV2TeacherCandidateAuditError):
            raise
        raise QuerySftV2TeacherCandidateAuditError(f"无法读取{label}") from error
    return rows


def _verify_sidecar(path: Path, label: str) -> Path:
    sidecar = path.with_suffix(".sha256")
    try:
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftV2TeacherCandidateAuditError(f"无法读取{label} SHA-256") from error
    if lines != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftV2TeacherCandidateAuditError(f"{label} SHA-256 无效")
    return sidecar


def _verify_work_package(directory: Path) -> tuple[Path, dict[str, dict[str, str]]]:
    hash_path = directory / WORK_HASH_FILENAME
    expected_names = {
        WORK_MANIFEST_FILENAME,
        NOOP_FILENAME,
        *(f"teacher-queue/batch-{batch:02d}.jsonl" for batch in range(1, 9)),
    }
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
        entries = {}
        for line in lines:
            digest, name = line.split("  ", 1)
            if name in entries or not digest or not name:
                raise ValueError
            entries[name] = digest
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise QuerySftV2TeacherCandidateAuditError("教师工作包 SHA-256 清单无效") from error
    if set(entries) != expected_names:
        raise QuerySftV2TeacherCandidateAuditError("教师工作包 SHA-256 清单未精确覆盖全部受签文件")
    for name, digest in entries.items():
        path = directory / name
        if not path.is_file() or _sha256_file(path) != digest:
            raise QuerySftV2TeacherCandidateAuditError(f"教师工作包身份已变化: {name}")
    manifest = _load_json(directory / WORK_MANIFEST_FILENAME, "教师工作包 manifest")
    if (
        manifest.get("pipeline") != "query_sft_v2_teacher_candidate_work_package"
        or manifest.get("records", {}).get("teacher_candidate_requests") != 1722
        or manifest.get("complete") is not True
    ):
        raise QuerySftV2TeacherCandidateAuditError("教师工作包状态无效")
    queues: dict[str, dict[str, str]] = {}
    for batch in range(1, 9):
        rows = _load_jsonl(directory / f"teacher-queue/batch-{batch:02d}.jsonl", f"教师队列第 {batch} 批")
        expected_count = 216 if batch < 8 else 210
        if len(rows) != expected_count:
            raise QuerySftV2TeacherCandidateAuditError(f"教师队列第 {batch} 批数量无效")
        for row in rows:
            candidate_id = row.get("candidate_id")
            if (
                tuple(row) != QUEUE_FIELDS
                or not isinstance(candidate_id, str)
                or candidate_id in queues
                or not isinstance(row.get("work_id"), str)
                or not candidate_id.startswith(f"{row['work_id']}/teacher-")
            ):
                raise QuerySftV2TeacherCandidateAuditError("教师队列记录无效或候选身份错配")
            queues[candidate_id] = row
    if len(queues) != 1722:
        raise QuerySftV2TeacherCandidateAuditError("教师队列未完整覆盖 1722 个候选槽位")
    return hash_path, queues


def _verify_runtime(runtime: dict[str, Any], *, queue_records: dict[str, dict[str, str]], directory: Path) -> None:
    if (
        runtime.get("pipeline") != "query_sft_v2_teacher_generation_run"
        or runtime.get("complete") is not True
        or runtime.get("records", {}).get("candidate_results") != len(queue_records)
    ):
        raise QuerySftV2TeacherCandidateAuditError("教师运行记录状态无效")
    identity = runtime.get("runtime_identity")
    if not isinstance(identity, dict) or tuple(identity) != RUNTIME_IDENTITY_FIELDS:
        raise QuerySftV2TeacherCandidateAuditError("教师运行身份字段必须精确匹配协议")
    if (
        not all(isinstance(identity[name], (str, int, float)) and not isinstance(identity[name], bool) and str(identity[name]).strip() for name in RUNTIME_IDENTITY_FIELDS)
        or (isinstance(identity["max_output_tokens"], int) and identity["max_output_tokens"] <= 0)
    ):
        raise QuerySftV2TeacherCandidateAuditError("教师运行身份取值无效")
    input_identity = runtime.get("inputs", {}).get("teacher_work_package_hash_manifest")
    work_hash = directory / WORK_HASH_FILENAME
    if not isinstance(input_identity, dict) or input_identity.get("sha256") != _sha256_file(work_hash):
        raise QuerySftV2TeacherCandidateAuditError("教师运行记录未绑定当前工作包身份")


def audit_query_sft_v2_teacher_candidates(
    *,
    work_package_dir: Path = DEFAULT_OUTPUT_DIR,
    results_path: Path | None = None,
    runtime_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, object]:
    """审计一次完整的教师候选运行；教师只可消费已签名的盲队列。"""

    directory = Path(work_package_dir).resolve()
    results_path = Path(results_path or directory / RESULT_FILENAME).resolve()
    runtime_path = Path(runtime_path or directory / RUNTIME_FILENAME).resolve()
    output_path = Path(output_path or directory / AUDIT_FILENAME).resolve()
    if output_path.exists() or output_path.with_suffix(".sha256").exists():
        raise QuerySftV2TeacherCandidateAuditError("教师候选协议审计输出已存在，禁止覆盖")
    work_hash, expected = _verify_work_package(directory)
    result_hash = _verify_sidecar(results_path, "教师候选结果")
    runtime_hash = _verify_sidecar(runtime_path, "教师运行记录")
    _verify_runtime(_load_json(runtime_path, "教师运行记录"), queue_records=expected, directory=directory)
    results = _load_jsonl(results_path, "教师候选结果")
    if len(results) != len(expected):
        raise QuerySftV2TeacherCandidateAuditError("教师候选结果数量不闭合")
    seen = set()
    noop_count = 0
    non_noop_targets = set()
    for row in results:
        candidate_id = row.get("candidate_id")
        if (
            tuple(row) != RESULT_FIELDS
            or not isinstance(candidate_id, str)
            or candidate_id not in expected
            or candidate_id in seen
            or row.get("work_id") != expected[candidate_id]["work_id"]
            or not isinstance(row.get("raw_output"), str)
        ):
            raise QuerySftV2TeacherCandidateAuditError("教师候选结果字段、覆盖或身份无效")
        try:
            target = parse_and_validate_query_enhancement(row["raw_output"])
        except (TypeError, ValueError) as error:
            raise QuerySftV2TeacherCandidateAuditError(f"教师候选不符合严格 Query Enhancement 协议: {candidate_id}") from error
        canonical = json.dumps(
            {"rewrite": target.rewrite, "expansion_terms": list(target.expansion_terms), "subqueries": list(target.subqueries)},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if canonical == json.dumps({"rewrite": expected[candidate_id]["query_original"], "expansion_terms": [], "subqueries": []}, ensure_ascii=False, separators=(",", ":")):
            noop_count += 1
        else:
            non_noop_targets.add(canonical)
        seen.add(candidate_id)
    if set(expected) != seen:
        raise QuerySftV2TeacherCandidateAuditError("教师候选结果未完整覆盖所有槽位")
    report = {
        "schema_version": "1.0",
        "pipeline": "query_sft_v2_teacher_candidate_protocol_audit",
        "release_status": "independent_semantic_review_pending",
        "inputs": {
            "teacher_work_package_hash_manifest": _identity(work_hash),
            "teacher_candidate_results": {**_identity(results_path, records=len(results)), "hash_manifest": _identity(result_hash)},
            "teacher_generation_run": {**_identity(runtime_path), "hash_manifest": _identity(runtime_hash)},
        },
        "records": {
            "teacher_candidates": len(results),
            "teacher_output_noop_candidates": noop_count,
            "unique_non_noop_targets": len(non_noop_targets),
        },
        "validation": {
            "teacher_work_package_identity_bound": True,
            "teacher_runtime_identity_bound": True,
            "all_teacher_requests_returned_once": True,
            "candidate_queue_mapping_preserved": True,
            "all_raw_outputs_strict_protocol_valid": True,
            "required_gt_not_read_or_emitted": True,
        },
        "readiness": {
            "teacher_generation_protocol_audit_complete": True,
            "independent_semantic_review_ready": True,
            "frozen_retrieval_selection_ready": False,
            "training_ready": False,
        },
        "complete": True,
    }
    try:
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        output_path.with_suffix(".sha256").write_text(f"{_sha256_file(output_path)}  {output_path.name}\n", encoding="utf-8", newline="\n")
    except (OSError, UnicodeError) as error:
        output_path.unlink(missing_ok=True)
        output_path.with_suffix(".sha256").unlink(missing_ok=True)
        raise QuerySftV2TeacherCandidateAuditError("无法发布教师候选协议审计") from error
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-package-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = audit_query_sft_v2_teacher_candidates(
            work_package_dir=args.work_package_dir,
            results_path=args.results,
            runtime_path=args.runtime,
            output_path=args.output,
        )
    except QuerySftV2TeacherCandidateAuditError as error:
        parser.error(str(error))
    print(f"QUERY_SFT_V2_TEACHER_PROTOCOL_AUDIT_OK candidates={report['records']['teacher_candidates']}")


if __name__ == "__main__":
    main()

