"""从公共 query 池 source candidate 发布 Query-SFT pilot 盲构造工作包。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


DATASET_ROOT = Path(__file__).resolve().parent
QUERY_POOL_ROOT = DATASET_ROOT / "QUERY-POOL"
DEFAULT_SOURCE_MANIFEST = (
    QUERY_POOL_ROOT / "manifests" / "query-pool-v1-source-candidate.json"
)
DEFAULT_SOURCE_CANDIDATE = (
    QUERY_POOL_ROOT / "authoring" / "query-pool-v1-source-candidate.jsonl"
)
DEFAULT_SELECTION_PLAN = (
    QUERY_POOL_ROOT / "authoring" / "query-sft-pilot-v1-selection.jsonl"
)
DEFAULT_OUTPUT_DIR = QUERY_POOL_ROOT / "pilot" / "query-sft-pilot-v1-work-package"

BLIND_QUEUE_FILENAME = "query-sft-pilot-v1-blind-authoring-queue.jsonl"
AUDIT_REFERENCE_FILENAME = "query-sft-pilot-v1-audit-reference.jsonl"
MANIFEST_FILENAME = "query-sft-pilot-v1-work-package.json"
HASH_FILENAME = "query-sft-pilot-v1-work-package.sha256"

AUTHORING_TYPE_TARGETS = {
    "no_op": 10,
    "colloquial": 12,
    "ellipsis": 8,
    "ambiguous_multi_intent": 5,
    "explicit_multi_matter": 5,
}
PILOT_RECORDS = sum(AUTHORING_TYPE_TARGETS.values())

_QUERY_ID_RE = re.compile(r"query:\d{4}$")
_SOURCE_FIELDS = {"query_id", "query_original", "required_chunk_ids"}
_SELECTION_FIELDS = {
    "query_id",
    "authoring_type",
    "coverage_domain",
    "selection_status",
}


class QuerySftPilotPreparationError(RuntimeError):
    """Query-SFT pilot 工作包无法安全发布。"""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record_digest(record: dict[str, object]) -> str:
    payload = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return _sha256_bytes(payload.encode("utf-8"))


def _file_identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    identity: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if records is not None:
        identity["records"] = records
    return identity


def _payload_identity(
    path: Path, payload: str, *, records: int
) -> dict[str, object]:
    encoded = payload.encode("utf-8")
    return {
        "path": str(path.resolve()),
        "records": records,
        "bytes": len(encoded),
        "sha256": _sha256_bytes(encoded),
    }


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftPilotPreparationError(
            f"无法读取{description}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise QuerySftPilotPreparationError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise QuerySftPilotPreparationError(
                        f"{description}不允许空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise QuerySftPilotPreparationError(
                        f"{description}第 {line_number} 条必须是 object"
                    )
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftPilotPreparationError):
            raise
        raise QuerySftPilotPreparationError(
            f"无法读取{description}: {path}"
        ) from error
    if not records:
        raise QuerySftPilotPreparationError(f"{description}不能为空")
    return records


def _verify_adjacent_hash(path: Path, description: str) -> Path:
    hash_path = path.with_suffix(".sha256")
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftPilotPreparationError(
            f"无法读取{description}相邻 SHA-256: {hash_path}"
        ) from error
    expected = f"{_sha256_file(path)}  {path.name}"
    if lines != [expected]:
        raise QuerySftPilotPreparationError(f"{description}相邻 SHA-256 无效")
    return hash_path


def _verify_bound_identity(
    metadata: object,
    path: Path,
    *,
    records: int,
    description: str,
) -> None:
    if not isinstance(metadata, dict):
        raise QuerySftPilotPreparationError(f"source manifest 缺少{description}身份")
    if (
        metadata.get("bytes") != path.stat().st_size
        or metadata.get("sha256") != _sha256_file(path)
        or metadata.get("records") != records
    ):
        raise QuerySftPilotPreparationError(f"{description} 身份已变化")


def _normalized_text(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuerySftPilotPreparationError(f"{description}必须是非空字符串")
    normalized = unicodedata.normalize("NFC", value.strip())
    if normalized != value or "\n" in value or "\r" in value:
        raise QuerySftPilotPreparationError(
            f"{description}必须是 NFC、无首尾空白的单行字符串"
        )
    return normalized


def _validate_source_records(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(records, start=1):
        prefix = f"source candidate 第 {position} 条"
        if set(record) != _SOURCE_FIELDS:
            raise QuerySftPilotPreparationError(f"{prefix}字段与冻结 schema 不一致")
        query_id = record.get("query_id")
        if (
            not isinstance(query_id, str)
            or not _QUERY_ID_RE.fullmatch(query_id)
            or query_id in by_id
        ):
            raise QuerySftPilotPreparationError(f"{prefix} query_id 无效或重复")
        _normalized_text(record.get("query_original"), f"{prefix} query_original")
        required = record.get("required_chunk_ids")
        if (
            not isinstance(required, list)
            or not 1 <= len(required) <= 3
            or any(not isinstance(item, str) or not item.strip() for item in required)
            or len(set(required)) != len(required)
        ):
            raise QuerySftPilotPreparationError(
                f"{prefix} required_chunk_ids 必须包含 1 至 3 个唯一非空字符串"
            )
        by_id[query_id] = record
    expected_ids = [f"query:{position:04d}" for position in range(1, len(records) + 1)]
    if list(by_id) != expected_ids:
        raise QuerySftPilotPreparationError("source candidate query_id 必须连续有序")
    return by_id


def _load_and_bind_source(
    manifest_path: Path,
    candidate_path: Path,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest_hash_path = _verify_adjacent_hash(manifest_path, "source manifest")
    manifest = _load_json(manifest_path, "source manifest")
    readiness = manifest.get("readiness")
    if (
        manifest.get("pipeline") != "query_pool_source_candidate_v1"
        or manifest.get("release_status") != "source_candidate"
        or manifest.get("complete") is not True
        or not isinstance(readiness, dict)
        or readiness.get("interface_frozen") is not True
        or readiness.get("id_mapping_frozen") is not True
        or readiness.get("final_query_text_frozen") is not False
        or readiness.get("query_sft_training_ready") is not False
    ):
        raise QuerySftPilotPreparationError("source manifest 状态无效")
    expected_records = manifest.get("records", {}).get("candidate")
    if type(expected_records) is not int or expected_records <= 0:
        raise QuerySftPilotPreparationError("source manifest 的 candidate 数量无效")
    _verify_bound_identity(
        manifest.get("outputs", {}).get("candidate"),
        candidate_path,
        records=expected_records,
        description="source candidate",
    )
    records = _load_jsonl(candidate_path, "source candidate")
    if expected_records != len(records):
        raise QuerySftPilotPreparationError(
            "source candidate 数量与 source manifest 不一致"
        )
    return manifest, manifest_hash_path, records, _validate_source_records(records)


def _validate_selection(
    records: list[dict[str, Any]],
    source_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(records) != PILOT_RECORDS:
        raise QuerySftPilotPreparationError(
            f"选择计划必须恰好包含 {PILOT_RECORDS} 条"
        )
    selected_ids: set[str] = set()
    type_counts: Counter[str] = Counter()
    approved = True
    validated = []
    for position, record in enumerate(records, start=1):
        prefix = f"选择计划第 {position} 条"
        if set(record) != _SELECTION_FIELDS:
            raise QuerySftPilotPreparationError(f"{prefix}字段无效")
        query_id = record.get("query_id")
        if not isinstance(query_id, str) or not _QUERY_ID_RE.fullmatch(query_id):
            raise QuerySftPilotPreparationError(f"{prefix} query_id 无效")
        if query_id in selected_ids:
            raise QuerySftPilotPreparationError(f"选择计划 query_id 重复: {query_id}")
        if query_id not in source_by_id:
            raise QuerySftPilotPreparationError(
                f"{query_id} 不存在于 source candidate"
            )
        selected_ids.add(query_id)
        authoring_type = record.get("authoring_type")
        if authoring_type not in AUTHORING_TYPE_TARGETS:
            raise QuerySftPilotPreparationError(
                f"{prefix} authoring_type 不在冻结类型中"
            )
        coverage_domain = _normalized_text(
            record.get("coverage_domain"), f"{prefix} coverage_domain"
        )
        approved = approved and record.get("selection_status") == "approved"
        type_counts[authoring_type] += 1
        validated.append(
            {
                "query_id": query_id,
                "authoring_type": authoring_type,
                "coverage_domain": coverage_domain,
                "selection_status": record.get("selection_status"),
            }
        )
    if not approved:
        raise QuerySftPilotPreparationError("选择计划必须全部 approved")
    if dict(type_counts) != AUTHORING_TYPE_TARGETS:
        raise QuerySftPilotPreparationError(
            f"选择计划分层配额无效: {dict(sorted(type_counts.items()))}"
        )
    return sorted(validated, key=lambda item: item["query_id"])


def _jsonl_payload(records: list[dict[str, object]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in records
    )


def _publish(
    output_dir: Path,
    payloads: list[tuple[str, str]],
    hash_payload: str,
) -> None:
    if output_dir.exists():
        raise QuerySftPilotPreparationError(f"输出目录必须不存在: {output_dir}")
    partials: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        output_dir.mkdir(parents=True)
        for filename, payload in payloads:
            partial = output_dir / f"{filename}.partial"
            final = output_dir / filename
            partial.write_text(payload, encoding="utf-8", newline="\n")
            partials.append((partial, final))
        hash_partial = output_dir / f"{HASH_FILENAME}.partial"
        hash_partial.write_text(hash_payload, encoding="utf-8", newline="\n")
        for partial, final in partials:
            partial.replace(final)
            published.append(final)
        hash_partial.replace(output_dir / HASH_FILENAME)
    except (OSError, UnicodeError) as error:
        for path in [
            *(partial for partial, _ in partials),
            output_dir / f"{HASH_FILENAME}.partial",
            *reversed(published),
        ]:
            path.unlink(missing_ok=True)
        try:
            output_dir.rmdir()
        except OSError:
            pass
        raise QuerySftPilotPreparationError("无法发布 Query-SFT pilot 工作包") from error


def prepare_query_sft_pilot(
    *,
    source_manifest_path: Path,
    source_candidate_path: Path,
    selection_plan_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """绑定显式选择计划并发布不向构造者暴露 GT 的 40 条工作包。"""

    source_manifest_path = Path(source_manifest_path).resolve()
    source_candidate_path = Path(source_candidate_path).resolve()
    selection_plan_path = Path(selection_plan_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise QuerySftPilotPreparationError(f"输出目录必须不存在: {output_dir}")

    _, source_manifest_hash_path, source_records, source_by_id = _load_and_bind_source(
        source_manifest_path, source_candidate_path
    )
    selection_hash_path = _verify_adjacent_hash(selection_plan_path, "选择计划")
    selection_records = _load_jsonl(selection_plan_path, "选择计划")
    selection = _validate_selection(selection_records, source_by_id)

    blind_queue = []
    audit_reference = []
    gt_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    required_law_counts: Counter[str] = Counter()
    for position, plan_record in enumerate(selection, start=1):
        source = source_by_id[plan_record["query_id"]]
        pilot_id = f"query_sft_pilot:{position:04d}"
        blind_queue.append(
            {
                "pilot_id": pilot_id,
                "source_id": source["query_id"],
                "authoring_type": plan_record["authoring_type"],
                "source_query": source["query_original"],
            }
        )
        audit_reference.append(
            {
                "pilot_id": pilot_id,
                "source_id": source["query_id"],
                "authoring_type": plan_record["authoring_type"],
                "coverage_domain": plan_record["coverage_domain"],
                "source_query": source["query_original"],
                "required_chunk_ids": list(source["required_chunk_ids"]),
                "source_record_sha256": _record_digest(source),
            }
        )
        gt_counts[str(len(source["required_chunk_ids"]))] += 1
        domain_counts[plan_record["coverage_domain"]] += 1
        for chunk_id in source["required_chunk_ids"]:
            required_law_counts[chunk_id.split("#", 1)[0]] += 1
    if set(gt_counts) != {"1", "2", "3"}:
        raise QuerySftPilotPreparationError("required GT 数量必须覆盖 1、2、3")

    blind_path = output_dir / BLIND_QUEUE_FILENAME
    audit_path = output_dir / AUDIT_REFERENCE_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    blind_payload = _jsonl_payload(blind_queue)
    audit_payload = _jsonl_payload(audit_reference)
    manifest: dict[str, object] = {
        "pipeline": "query_sft_pilot_work_package_v1",
        "release_status": "pilot_authoring_work_package",
        "inputs": {
            "source_manifest": {
                **_file_identity(source_manifest_path),
                "hash_manifest": _file_identity(source_manifest_hash_path),
            },
            "source_candidate": _file_identity(
                source_candidate_path, records=len(source_records)
            ),
            "selection_plan": {
                **_file_identity(selection_plan_path, records=len(selection_records)),
                "hash_manifest": _file_identity(selection_hash_path),
            },
        },
        "policy": {
            "selection_plan_fields": [
                "query_id",
                "authoring_type",
                "coverage_domain",
                "selection_status",
            ],
            "authoring_type_targets": AUTHORING_TYPE_TARGETS,
            "pilot_order": "query_id_ascending",
            "constructor_can_read_required_gt": False,
            "coverage_domain_source": "human_approved_selection_plan",
            "required_gt_count_coverage": [1, 2, 3],
            "retrieval_dependency": "none",
        },
        "records": {
            "source_candidate": len(source_records),
            "selected": len(selection),
            "by_authoring_type": dict(
                Counter(item["authoring_type"] for item in selection)
            ),
            "by_required_gt_count": dict(sorted(gt_counts.items())),
            "by_coverage_domain": dict(sorted(domain_counts.items())),
            "by_required_law": dict(sorted(required_law_counts.items())),
        },
        "validation": {
            "source_candidate_identity_bound": True,
            "selection_plan_identity_bound": True,
            "selected_query_ids_unique_and_known": True,
            "all_selection_records_approved": True,
            "authoring_type_quotas_exact": True,
            "required_gt_count_buckets_covered": True,
            "coverage_domains_reported": True,
            "blind_queue_excludes_required_gt": True,
            "retrieval_not_invoked": True,
        },
        "outputs": {
            "blind_authoring_queue": _payload_identity(
                blind_path, blind_payload, records=len(blind_queue)
            ),
            "audit_reference": _payload_identity(
                audit_path, audit_payload, records=len(audit_reference)
            ),
            "manifest": MANIFEST_FILENAME,
            "sha256_manifest": HASH_FILENAME,
        },
        "readiness": {
            "pilot_selection_frozen": True,
            "blind_authoring_queue_ready": True,
            "query_original_authoring_complete": False,
            "teacher_candidate_generation_ready": False,
            "retrieval_evaluation_ready": False,
            "query_sft_training_ready": False,
        },
        "complete": True,
    }
    manifest_payload = json.dumps(
        manifest, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    hash_payload = "".join(
        [
            f"{_sha256_bytes(blind_payload.encode('utf-8'))}  {BLIND_QUEUE_FILENAME}\n",
            f"{_sha256_bytes(audit_payload.encode('utf-8'))}  {AUDIT_REFERENCE_FILENAME}\n",
            f"{_sha256_bytes(manifest_payload.encode('utf-8'))}  {MANIFEST_FILENAME}\n",
        ]
    )
    _publish(
        output_dir,
        [
            (BLIND_QUEUE_FILENAME, blind_payload),
            (AUDIT_REFERENCE_FILENAME, audit_payload),
            (MANIFEST_FILENAME, manifest_payload),
        ],
        hash_payload,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST
    )
    parser.add_argument(
        "--source-candidate", type=Path, default=DEFAULT_SOURCE_CANDIDATE
    )
    parser.add_argument(
        "--selection-plan", type=Path, default=DEFAULT_SELECTION_PLAN
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        manifest = prepare_query_sft_pilot(
            source_manifest_path=args.source_manifest,
            source_candidate_path=args.source_candidate,
            selection_plan_path=args.selection_plan,
            output_dir=args.output_dir,
        )
    except QuerySftPilotPreparationError as error:
        parser.error(str(error))
    print(
        f"[完成] 发布 {manifest['records']['selected']} 条 Query-SFT pilot 盲构造任务"
    )
    print("[待完成] 人工构造 query、教师候选生成与真实检索筛选")
    print(f"工作包: {args.output_dir}")


if __name__ == "__main__":
    main()


__all__ = [
    "AUDIT_REFERENCE_FILENAME",
    "AUTHORING_TYPE_TARGETS",
    "BLIND_QUEUE_FILENAME",
    "HASH_FILENAME",
    "MANIFEST_FILENAME",
    "QuerySftPilotPreparationError",
    "prepare_query_sft_pilot",
]
