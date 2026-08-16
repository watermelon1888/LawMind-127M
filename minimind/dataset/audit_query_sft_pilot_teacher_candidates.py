"""审计 Query-SFT pilot 教师候选生成结果的协议和运行时身份。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rag.query.enhancement import parse_and_validate_query_enhancement

try:
    from .prepare_query_sft_pilot_teacher_candidates import (
        DEFAULT_OUTPUT_DIR,
        HASH_FILENAME as WORK_PACKAGE_HASH_FILENAME,
        MANIFEST_FILENAME as WORK_PACKAGE_MANIFEST_FILENAME,
        QUEUE_FILENAME,
    )
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset.prepare_query_sft_pilot_teacher_candidates import (
        DEFAULT_OUTPUT_DIR,
        HASH_FILENAME as WORK_PACKAGE_HASH_FILENAME,
        MANIFEST_FILENAME as WORK_PACKAGE_MANIFEST_FILENAME,
        QUEUE_FILENAME,
    )


DEFAULT_RESULT_FILENAME = "query-sft-pilot-v1-teacher-candidate-results.jsonl"
DEFAULT_RUNTIME_FILENAME = "query-sft-pilot-v1-teacher-generation-run.json"
DEFAULT_AUDIT_FILENAME = "query-sft-pilot-v1-teacher-candidate-protocol-audit.json"

_QUEUE_FIELDS = {"candidate_id", "pilot_id", "query_original"}
_RESULT_FIELDS = {"candidate_id", "pilot_id", "raw_output"}
_RUNTIME_REQUIRED_FIELDS = {
    "teacher_model_id",
    "teacher_endpoint_or_local_checkpoint",
    "decoding_temperature",
    "decoding_seed_or_nondeterminism_declaration",
    "max_output_tokens",
}


class QuerySftPilotTeacherCandidateAuditError(RuntimeError):
    """Query-SFT pilot 教师候选结果不满足可审计协议。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if records is not None:
        result["records"] = records
    return result


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftPilotTeacherCandidateAuditError(
            f"无法读取{description}: {path}"
        ) from error
    if not isinstance(payload, dict):
        raise QuerySftPilotTeacherCandidateAuditError(f"{description}必须是 JSON object")
    return payload


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise QuerySftPilotTeacherCandidateAuditError(
                        f"{description}不允许空行: {line_number}"
                    )
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise QuerySftPilotTeacherCandidateAuditError(
                        f"{description}第 {line_number} 条必须是 object"
                    )
                records.append(item)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftPilotTeacherCandidateAuditError):
            raise
        raise QuerySftPilotTeacherCandidateAuditError(
            f"无法读取{description}: {path}"
        ) from error
    if not records:
        raise QuerySftPilotTeacherCandidateAuditError(f"{description}不能为空")
    return records


def _verify_adjacent_hash(path: Path, description: str) -> Path:
    hash_path = path.with_suffix(".sha256")
    try:
        actual = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftPilotTeacherCandidateAuditError(
            f"无法读取{description}相邻 SHA-256: {hash_path}"
        ) from error
    if actual != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftPilotTeacherCandidateAuditError(f"{description}相邻 SHA-256 无效")
    return hash_path


def _verify_work_package_hashes(directory: Path) -> tuple[Path, Path, Path]:
    manifest_path = directory / WORK_PACKAGE_MANIFEST_FILENAME
    queue_path = directory / QUEUE_FILENAME
    hash_path = directory / WORK_PACKAGE_HASH_FILENAME
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftPilotTeacherCandidateAuditError(
            "无法读取教师候选工作包 SHA-256"
        ) from error
    expected = {
        manifest_path.name: _sha256_file(manifest_path),
        queue_path.name: _sha256_file(queue_path),
    }
    found = {}
    for line in lines:
        digest, separator, filename = line.partition("  ")
        if not separator or filename in found:
            raise QuerySftPilotTeacherCandidateAuditError(
                "教师候选工作包 SHA-256 格式无效"
            )
        found[filename] = digest
    if any(found.get(filename) != digest for filename, digest in expected.items()):
        raise QuerySftPilotTeacherCandidateAuditError("教师候选工作包身份已变化")
    return manifest_path, queue_path, hash_path


def _validate_runtime(
    runtime: dict[str, Any], *, queue_path: Path, expected_requests: int
) -> None:
    if (
        runtime.get("pipeline") != "query_sft_pilot_teacher_generation_run_v1"
        or runtime.get("complete") is not True
        or runtime.get("records", {}).get("candidate_results") != expected_requests
    ):
        raise QuerySftPilotTeacherCandidateAuditError("教师运行记录状态无效")
    queue_identity = runtime.get("inputs", {}).get("teacher_candidate_queue")
    if not isinstance(queue_identity, dict) or (
        queue_identity.get("sha256") != _sha256_file(queue_path)
        or queue_identity.get("bytes") != queue_path.stat().st_size
        or queue_identity.get("records") != expected_requests
    ):
        raise QuerySftPilotTeacherCandidateAuditError("教师运行记录未绑定当前候选队列")
    identity = runtime.get("runtime_identity")
    if not isinstance(identity, dict) or set(identity) != _RUNTIME_REQUIRED_FIELDS:
        raise QuerySftPilotTeacherCandidateAuditError("教师运行时身份字段必须精确匹配协议")
    if (
        not isinstance(identity["teacher_model_id"], str)
        or not identity["teacher_model_id"].strip()
        or not isinstance(identity["teacher_endpoint_or_local_checkpoint"], str)
        or not identity["teacher_endpoint_or_local_checkpoint"].strip()
        or not isinstance(identity["decoding_temperature"], (float, int, str))
        or isinstance(identity["decoding_temperature"], bool)
        or (
            isinstance(identity["decoding_temperature"], str)
            and not identity["decoding_temperature"].strip()
        )
        or not isinstance(
            identity["decoding_seed_or_nondeterminism_declaration"], (str, int)
        )
        or isinstance(identity["decoding_seed_or_nondeterminism_declaration"], bool)
        or not isinstance(identity["max_output_tokens"], (int, str))
        or isinstance(identity["max_output_tokens"], bool)
        or (
            isinstance(identity["max_output_tokens"], int)
            and identity["max_output_tokens"] <= 0
        )
        or (
            isinstance(identity["max_output_tokens"], str)
            and not identity["max_output_tokens"].strip()
        )
    ):
        raise QuerySftPilotTeacherCandidateAuditError("教师运行时身份值无效")


def audit_query_sft_pilot_teacher_candidates(
    *,
    work_package_dir: Path,
    result_path: Path,
    runtime_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """审计 117 条物化教师输出，绝不读取或输出 GT。"""

    work_package_dir = Path(work_package_dir).resolve()
    result_path = Path(result_path).resolve()
    runtime_path = Path(runtime_path).resolve()
    output_path = Path(output_path).resolve()
    output_hash_path = output_path.with_suffix(".sha256")
    if output_path.exists() or output_hash_path.exists():
        raise QuerySftPilotTeacherCandidateAuditError(f"审计输出已存在: {output_path}")
    manifest_path, queue_path, work_hash_path = _verify_work_package_hashes(work_package_dir)
    result_hash_path = _verify_adjacent_hash(result_path, "教师候选结果")
    runtime_hash_path = _verify_adjacent_hash(runtime_path, "教师运行记录")
    package = _load_json(manifest_path, "教师候选工作包 manifest")
    if (
        package.get("pipeline") != "query_sft_pilot_teacher_candidate_work_package_v1"
        or package.get("complete") is not True
        or package.get("readiness", {}).get("teacher_candidate_work_package_ready")
        is not True
    ):
        raise QuerySftPilotTeacherCandidateAuditError("教师候选工作包状态无效")
    queue = _load_jsonl(queue_path, "教师候选队列")
    expected: dict[str, dict[str, str]] = {}
    for position, item in enumerate(queue, start=1):
        if set(item) != _QUEUE_FIELDS:
            raise QuerySftPilotTeacherCandidateAuditError(
                f"教师候选队列第 {position} 条字段无效"
            )
        candidate_id = item.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in expected:
            raise QuerySftPilotTeacherCandidateAuditError("教师候选队列 candidate_id 无效或重复")
        expected[candidate_id] = item
    if package.get("records", {}).get("teacher_candidate_requests") != len(expected):
        raise QuerySftPilotTeacherCandidateAuditError("教师候选工作包请求数不一致")
    _validate_runtime(
        _load_json(runtime_path, "教师运行记录"),
        queue_path=queue_path,
        expected_requests=len(expected),
    )

    results = _load_jsonl(result_path, "教师候选结果")
    parsed_noop_count = 0
    seen = set()
    for position, item in enumerate(results, start=1):
        if set(item) != _RESULT_FIELDS:
            raise QuerySftPilotTeacherCandidateAuditError(
                f"教师候选结果第 {position} 条字段无效"
            )
        candidate_id = item.get("candidate_id")
        if candidate_id not in expected or candidate_id in seen:
            raise QuerySftPilotTeacherCandidateAuditError(
                f"教师候选结果 candidate_id 不存在或重复: {candidate_id}"
            )
        if item.get("pilot_id") != expected[candidate_id]["pilot_id"]:
            raise QuerySftPilotTeacherCandidateAuditError(
                f"教师候选结果 pilot_id 与队列不一致: {candidate_id}"
            )
        raw_output = item.get("raw_output")
        if not isinstance(raw_output, str):
            raise QuerySftPilotTeacherCandidateAuditError(
                f"教师候选结果 raw_output 必须为字符串: {candidate_id}"
            )
        try:
            enhancement = parse_and_validate_query_enhancement(raw_output)
        except (TypeError, ValueError) as error:
            raise QuerySftPilotTeacherCandidateAuditError(
                f"教师候选结果不符合 Query Enhancement 协议: {candidate_id}"
            ) from error
        if (
            enhancement.rewrite == expected[candidate_id]["query_original"]
            and not enhancement.expansion_terms
            and not enhancement.subqueries
        ):
            parsed_noop_count += 1
        seen.add(candidate_id)
    if set(seen) != set(expected):
        missing = sorted(set(expected) - seen)
        raise QuerySftPilotTeacherCandidateAuditError(
            f"教师候选结果缺少请求: {missing}"
        )

    report: dict[str, object] = {
        "pipeline": "query_sft_pilot_teacher_candidate_protocol_audit_v1",
        "inputs": {
            "work_package_manifest": _identity(manifest_path),
            "work_package_hash_manifest": _identity(work_hash_path),
            "teacher_candidate_queue": _identity(queue_path, records=len(queue)),
            "teacher_candidate_results": {**_identity(result_path, records=len(results)), "hash_manifest": _identity(result_hash_path)},
            "teacher_generation_run": {**_identity(runtime_path), "hash_manifest": _identity(runtime_hash_path)},
        },
        "records": {
            "expected_teacher_candidates": len(expected),
            "audited_teacher_candidates": len(results),
            "protocol_valid_candidates": len(results),
            "teacher_output_noop_candidates": parsed_noop_count,
        },
        "validation": {
            "work_package_identity_bound": True,
            "teacher_runtime_identity_bound": True,
            "all_teacher_requests_returned_once": True,
            "candidate_queue_mapping_preserved": True,
            "all_raw_outputs_strict_protocol_valid": True,
            "required_gt_not_read_or_emitted": True,
        },
        "readiness": {
            "teacher_generation_protocol_audit_complete": True,
            "semantic_gt_candidate_review_ready": True,
            "retrieval_evaluation_ready": False,
            "query_sft_training_ready": False,
        },
        "complete": True,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        output_path.write_text(payload, encoding="utf-8", newline="\n")
        output_hash_path.write_text(
            f"{_sha256_file(output_path)}  {output_path.name}\n",
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, UnicodeError) as error:
        output_path.unlink(missing_ok=True)
        output_hash_path.unlink(missing_ok=True)
        raise QuerySftPilotTeacherCandidateAuditError("无法发布教师候选协议审计") from error
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-package-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--results", type=Path, default=DEFAULT_OUTPUT_DIR / DEFAULT_RESULT_FILENAME
    )
    parser.add_argument(
        "--runtime", type=Path, default=DEFAULT_OUTPUT_DIR / DEFAULT_RUNTIME_FILENAME
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR / DEFAULT_AUDIT_FILENAME
    )
    args = parser.parse_args()
    try:
        report = audit_query_sft_pilot_teacher_candidates(
            work_package_dir=args.work_package_dir,
            result_path=args.results,
            runtime_path=args.runtime,
            output_path=args.output,
        )
    except QuerySftPilotTeacherCandidateAuditError as error:
        parser.error(str(error))
    print(
        "[完成] 教师候选协议审计 "
        f"candidates={report['records']['protocol_valid_candidates']}"
    )
    print(f"审计报告: {args.output}")


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_AUDIT_FILENAME",
    "DEFAULT_RESULT_FILENAME",
    "DEFAULT_RUNTIME_FILENAME",
    "QuerySftPilotTeacherCandidateAuditError",
    "audit_query_sft_pilot_teacher_candidates",
]
