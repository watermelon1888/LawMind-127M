"""审计 Query-SFT pilot 实际输入的结构、去重和评估隔离。"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .build_sft_evaluation_exclusions import question_digest
    from .prepare_query_sft_pilot import (
        BLIND_QUEUE_FILENAME,
        DEFAULT_OUTPUT_DIR,
        HASH_FILENAME as WORK_PACKAGE_HASH_FILENAME,
        MANIFEST_FILENAME as WORK_PACKAGE_MANIFEST_FILENAME,
    )
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset.build_sft_evaluation_exclusions import question_digest
    from dataset.prepare_query_sft_pilot import (
        BLIND_QUEUE_FILENAME,
        DEFAULT_OUTPUT_DIR,
        HASH_FILENAME as WORK_PACKAGE_HASH_FILENAME,
        MANIFEST_FILENAME as WORK_PACKAGE_MANIFEST_FILENAME,
    )


DATASET_ROOT = Path(__file__).resolve().parent
QUERY_POOL_ROOT = DATASET_ROOT / "QUERY-POOL"
DEFAULT_SOURCE_CANDIDATE = (
    QUERY_POOL_ROOT / "authoring" / "query-pool-v1-source-candidate.jsonl"
)
DEFAULT_EVALUATION_EXCLUSIONS = (
    DATASET_ROOT / "RAG-SFT" / "manifests" / "evaluation-exclusions-project-rag-v2.json"
)
DEFAULT_DRAFT_FILENAME = "query-sft-pilot-v1-query-input-draft.jsonl"
DEFAULT_AUDIT_FILENAME = "query-sft-pilot-v1-query-input-audit.json"

_BLIND_FIELDS = {"pilot_id", "source_id", "authoring_type", "source_query"}
_DRAFT_FIELDS = {"pilot_id", "source_id", "authoring_type", "query_original"}
_SOURCE_FIELDS = {"query_id", "query_original", "required_chunk_ids"}


class QuerySftPilotInputAuditError(RuntimeError):
    """Query-SFT pilot 实际输入不能通过结构或隔离审计。"""


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
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftPilotInputAuditError(
            f"无法读取{description}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise QuerySftPilotInputAuditError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise QuerySftPilotInputAuditError(
                        f"{description}不允许空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise QuerySftPilotInputAuditError(
                        f"{description}第 {line_number} 条必须是 object"
                    )
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftPilotInputAuditError):
            raise
        raise QuerySftPilotInputAuditError(
            f"无法读取{description}: {path}"
        ) from error
    if not records:
        raise QuerySftPilotInputAuditError(f"{description}不能为空")
    return records


def _verify_single_hash(path: Path, description: str) -> Path:
    hash_path = path.with_suffix(".sha256")
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftPilotInputAuditError(
            f"无法读取{description}相邻 SHA-256: {hash_path}"
        ) from error
    if lines != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftPilotInputAuditError(f"{description}相邻 SHA-256 无效")
    return hash_path


def _verify_work_package_hashes(directory: Path) -> tuple[Path, Path, Path]:
    manifest_path = directory / WORK_PACKAGE_MANIFEST_FILENAME
    blind_path = directory / BLIND_QUEUE_FILENAME
    hash_path = directory / WORK_PACKAGE_HASH_FILENAME
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftPilotInputAuditError("无法读取 pilot 工作包 SHA-256") from error
    entries = {}
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise QuerySftPilotInputAuditError("pilot 工作包 SHA-256 格式无效")
        entries[parts[1]] = parts[0]
    for path in (blind_path, manifest_path):
        if entries.get(path.name) != _sha256_file(path):
            raise QuerySftPilotInputAuditError(f"pilot 工作包身份已变化: {path.name}")
    return manifest_path, blind_path, hash_path


def _verify_identity(
    metadata: object,
    path: Path,
    *,
    records: int,
    description: str,
) -> None:
    if not isinstance(metadata, dict):
        raise QuerySftPilotInputAuditError(f"工作包缺少{description}身份")
    if (
        metadata.get("bytes") != path.stat().st_size
        or metadata.get("sha256") != _sha256_file(path)
        or metadata.get("records") != records
    ):
        raise QuerySftPilotInputAuditError(f"{description} 身份已变化")


def _normalize_query(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuerySftPilotInputAuditError(f"{description}必须是非空字符串")
    normalized = unicodedata.normalize("NFC", value.strip())
    if normalized != value or "\n" in value or "\r" in value:
        raise QuerySftPilotInputAuditError(
            f"{description}必须是 NFC、无首尾空白的单行字符串"
        )
    if len(normalized) > 96:
        raise QuerySftPilotInputAuditError(f"{description}超过 96 个字符")
    return normalized


def _load_exclusions(path: Path) -> tuple[set[str], Path]:
    hash_path = _verify_single_hash(path, "评估排除清单")
    payload = _load_json(path, "评估排除清单")
    digests = payload.get("question_sha256")
    if (
        payload.get("complete_for_formal_sft") is not True
        or not isinstance(digests, list)
        or not digests
        or len(digests) != len(set(digests))
        or any(not isinstance(item, str) or len(item) != 64 for item in digests)
    ):
        raise QuerySftPilotInputAuditError("评估排除清单状态或 question_sha256 无效")
    return set(digests), hash_path


def _load_source_queries(path: Path, metadata: object) -> dict[str, tuple[str, str]]:
    records = _load_jsonl(path, "source candidate")
    _verify_identity(
        metadata,
        path,
        records=len(records),
        description="source candidate",
    )
    source_by_id = {}
    for position, record in enumerate(records, start=1):
        if set(record) != _SOURCE_FIELDS:
            raise QuerySftPilotInputAuditError(
                f"source candidate 第 {position} 条字段无效"
            )
        query_id = record.get("query_id")
        query = _normalize_query(
            record.get("query_original"), f"source candidate 第 {position} 条 query_original"
        )
        if not isinstance(query_id, str) or query_id in source_by_id:
            raise QuerySftPilotInputAuditError("source candidate query_id 无效或重复")
        source_by_id[query_id] = (query, question_digest(query))
    return source_by_id


def audit_query_sft_pilot_inputs(
    *,
    work_package_dir: Path,
    source_candidate_path: Path,
    draft_path: Path,
    evaluation_exclusions_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """审计已人工构造的 pilot 输入，不读取或输出 required GT。"""

    work_package_dir = Path(work_package_dir).resolve()
    source_candidate_path = Path(source_candidate_path).resolve()
    draft_path = Path(draft_path).resolve()
    evaluation_exclusions_path = Path(evaluation_exclusions_path).resolve()
    output_path = Path(output_path).resolve()
    hash_output_path = output_path.with_suffix(".sha256")
    if output_path.exists() or hash_output_path.exists():
        raise QuerySftPilotInputAuditError(f"审计输出已存在: {output_path}")

    manifest_path, blind_path, work_hash_path = _verify_work_package_hashes(
        work_package_dir
    )
    work_manifest = _load_json(manifest_path, "pilot 工作包 manifest")
    if (
        work_manifest.get("pipeline") != "query_sft_pilot_work_package_v1"
        or work_manifest.get("release_status") != "pilot_authoring_work_package"
        or work_manifest.get("complete") is not True
        or work_manifest.get("readiness", {}).get("blind_authoring_queue_ready")
        is not True
    ):
        raise QuerySftPilotInputAuditError("pilot 工作包状态无效")
    blind_records = _load_jsonl(blind_path, "盲构造队列")
    if work_manifest.get("records", {}).get("selected") != len(blind_records):
        raise QuerySftPilotInputAuditError("盲构造队列数量与工作包不一致")
    _verify_identity(
        work_manifest.get("outputs", {}).get("blind_authoring_queue"),
        blind_path,
        records=len(blind_records),
        description="盲构造队列",
    )
    source_by_id = _load_source_queries(
        source_candidate_path, work_manifest.get("inputs", {}).get("source_candidate")
    )
    exclusions, exclusions_hash_path = _load_exclusions(evaluation_exclusions_path)
    draft_hash_path = _verify_single_hash(draft_path, "实际输入 draft")
    draft_records = _load_jsonl(draft_path, "实际输入 draft")
    if len(draft_records) != len(blind_records):
        raise QuerySftPilotInputAuditError("实际输入 draft 数量与盲构造队列不一致")

    blind_by_id = {}
    for position, record in enumerate(blind_records, start=1):
        if set(record) != _BLIND_FIELDS:
            raise QuerySftPilotInputAuditError(f"盲构造队列第 {position} 条字段无效")
        pilot_id = record.get("pilot_id")
        source_id = record.get("source_id")
        if not isinstance(pilot_id, str) or pilot_id in blind_by_id:
            raise QuerySftPilotInputAuditError("盲构造队列 pilot_id 无效或重复")
        if source_id not in source_by_id:
            raise QuerySftPilotInputAuditError("盲构造队列 source_id 不在 source candidate")
        if record.get("source_query") != source_by_id[source_id][0]:
            raise QuerySftPilotInputAuditError("盲构造队列来源问题身份已变化")
        blind_by_id[pilot_id] = record

    normalized_drafts = set()
    seen_pilot_ids = set()
    no_op = 0
    modified = 0
    other_source_collisions = 0
    source_digest_owner = {
        digest: source_id for source_id, (_, digest) in source_by_id.items()
    }
    for position, record in enumerate(draft_records, start=1):
        if set(record) != _DRAFT_FIELDS:
            raise QuerySftPilotInputAuditError(f"实际输入 draft 第 {position} 条字段无效")
        pilot_id = record.get("pilot_id")
        if not isinstance(pilot_id, str) or pilot_id in seen_pilot_ids:
            raise QuerySftPilotInputAuditError("实际输入 draft pilot_id 无效或重复")
        if pilot_id not in blind_by_id:
            raise QuerySftPilotInputAuditError("实际输入 draft 与盲构造队列不一致")
        seen_pilot_ids.add(pilot_id)
        blind = blind_by_id[pilot_id]
        if (
            record.get("source_id") != blind["source_id"]
            or record.get("authoring_type") != blind["authoring_type"]
        ):
            raise QuerySftPilotInputAuditError("实际输入 draft 与盲构造队列不一致")
        query = _normalize_query(
            record.get("query_original"), f"实际输入 draft 第 {position} 条 query_original"
        )
        query_digest = question_digest(query)
        if query_digest in normalized_drafts:
            raise QuerySftPilotInputAuditError("实际输入 draft query_original 重复")
        normalized_drafts.add(query_digest)
        source_query = blind["source_query"]
        if blind["authoring_type"] == "no_op":
            if query != source_query:
                raise QuerySftPilotInputAuditError("no_op 必须原样保留 source_query")
            no_op += 1
        else:
            if query == source_query:
                raise QuerySftPilotInputAuditError(
                    "非 no_op 必须修改 source_query"
                )
            modified += 1
        source_owner = source_digest_owner.get(query_digest)
        if source_owner is not None and source_owner != blind["source_id"]:
            other_source_collisions += 1
            raise QuerySftPilotInputAuditError(
                "实际输入 draft 与其他 source candidate query 重复"
            )
        if query_digest in exclusions:
            raise QuerySftPilotInputAuditError("实际输入 draft 命中评估排除清单")
    if set(blind_by_id) != seen_pilot_ids:
        raise QuerySftPilotInputAuditError("实际输入 draft 与盲构造队列不闭合")

    report: dict[str, object] = {
        "pipeline": "query_sft_pilot_input_audit_v1",
        "inputs": {
            "work_package_manifest": _identity(manifest_path),
            "work_package_hash_manifest": _identity(work_hash_path),
            "blind_authoring_queue": _identity(blind_path, records=len(blind_records)),
            "source_candidate": _identity(
                source_candidate_path, records=len(source_by_id)
            ),
            "draft": {
                **_identity(draft_path, records=len(draft_records)),
                "hash_manifest": _identity(draft_hash_path),
            },
            "evaluation_exclusions": {
                **_identity(evaluation_exclusions_path),
                "hash_manifest": _identity(exclusions_hash_path),
            },
        },
        "records": {
            "blind_queue": len(blind_records),
            "draft": len(draft_records),
            "no_op": no_op,
            "modified": modified,
            "unique_normalized_queries": len(normalized_drafts),
            "evaluation_question_overlap": 0,
            "other_source_query_collisions": other_source_collisions,
        },
        "validation": {
            "work_package_identity_bound": True,
            "source_candidate_identity_bound": True,
            "draft_identity_bound": True,
            "draft_exactly_matches_blind_queue": True,
            "no_op_preserves_source_query": True,
            "non_no_op_changes_source_query": True,
            "queries_unique_and_within_limit": True,
            "evaluation_question_overlap": 0,
            "other_source_query_collisions": 0,
            "required_gt_not_emitted": True,
        },
        "readiness": {
            "query_input_draft_structural_ready": True,
            "semantic_and_gt_compatibility_review_complete": False,
            "teacher_candidate_generation_ready": False,
            "retrieval_evaluation_ready": False,
            "query_sft_training_ready": False,
        },
        "complete": True,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial = output_path.with_name(output_path.name + ".partial")
        hash_partial = hash_output_path.with_name(hash_output_path.name + ".partial")
        partial.write_text(payload, encoding="utf-8", newline="\n")
        hash_partial.write_text(
            f"{_sha256_file(partial)}  {output_path.name}\n",
            encoding="utf-8",
            newline="\n",
        )
        partial.replace(output_path)
        hash_partial.replace(hash_output_path)
    except (OSError, UnicodeError) as error:
        for path in (
            output_path.with_name(output_path.name + ".partial"),
            hash_output_path.with_name(hash_output_path.name + ".partial"),
            output_path,
            hash_output_path,
        ):
            path.unlink(missing_ok=True)
        raise QuerySftPilotInputAuditError("无法发布实际输入审计报告") from error
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-package-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-candidate", type=Path, default=DEFAULT_SOURCE_CANDIDATE)
    parser.add_argument(
        "--draft",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / DEFAULT_DRAFT_FILENAME,
    )
    parser.add_argument(
        "--evaluation-exclusions",
        type=Path,
        default=DEFAULT_EVALUATION_EXCLUSIONS,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / DEFAULT_AUDIT_FILENAME,
    )
    args = parser.parse_args()
    try:
        report = audit_query_sft_pilot_inputs(
            work_package_dir=args.work_package_dir,
            source_candidate_path=args.source_candidate,
            draft_path=args.draft,
            evaluation_exclusions_path=args.evaluation_exclusions,
            output_path=args.output,
        )
    except QuerySftPilotInputAuditError as error:
        parser.error(str(error))
    print(f"[完成] 审计 {report['records']['draft']} 条 Query-SFT pilot 实际输入")
    print("[待完成] 语义与 GT 兼容性审核、教师候选和 Retrieval 比较")
    print(f"审计报告: {args.output}")


if __name__ == "__main__":
    main()


__all__ = [
    "BLIND_QUEUE_FILENAME",
    "DEFAULT_AUDIT_FILENAME",
    "DEFAULT_DRAFT_FILENAME",
    "QuerySftPilotInputAuditError",
    "WORK_PACKAGE_HASH_FILENAME",
    "WORK_PACKAGE_MANIFEST_FILENAME",
    "audit_query_sft_pilot_inputs",
    "question_digest",
]
