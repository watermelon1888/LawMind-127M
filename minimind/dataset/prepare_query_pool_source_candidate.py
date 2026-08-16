"""从冻结 canonical query 发布公共 query 池 source candidate。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .build_sft_evaluation_exclusions import question_digest
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset.build_sft_evaluation_exclusions import question_digest


DATASET_ROOT = Path(__file__).resolve().parent
RAG_SFT_ROOT = DATASET_ROOT / "RAG-SFT"
QUERY_POOL_ROOT = DATASET_ROOT / "QUERY-POOL"
DEFAULT_CANONICAL_MANIFEST = (
    RAG_SFT_ROOT / "manifests" / "rag-sft-canonical-v2.json"
)
DEFAULT_EVALUATION_EXCLUSIONS = (
    RAG_SFT_ROOT / "manifests" / "evaluation-exclusions-project-rag-v2.json"
)
DEFAULT_CANDIDATE_OUTPUT = (
    QUERY_POOL_ROOT / "authoring" / "query-pool-v1-source-candidate.jsonl"
)
DEFAULT_MAPPING_OUTPUT = (
    QUERY_POOL_ROOT / "manifests" / "query-pool-v1-source-mapping.jsonl"
)
DEFAULT_MANIFEST_OUTPUT = (
    QUERY_POOL_ROOT / "manifests" / "query-pool-v1-source-candidate.json"
)

QUERY_MAX_CHARS = 96
REQUIRED_GT_MAX = 3
EXPECTED_PROJECT_RECORDS = 575
_SOURCE_ID_RE = re.compile(r"rag_sft:\d{4}$")
_QUERY_ID_RE = re.compile(r"query:\d{4}$")
_SHA256_RE = re.compile(r"[0-9a-f]{64}$")
_CANDIDATE_FIELDS = {"query_id", "query_original", "required_chunk_ids"}


class QueryPoolPreparationError(RuntimeError):
    """公共 query 池 source candidate 无法安全发布。"""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _record_digest(record: dict[str, object]) -> str:
    payload = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return _sha256_bytes(payload.encode("utf-8"))


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QueryPoolPreparationError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise QueryPoolPreparationError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise QueryPoolPreparationError(
                        f"{description}不允许空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise QueryPoolPreparationError(
                        f"{description}第 {line_number} 条必须是 object"
                    )
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QueryPoolPreparationError):
            raise
        raise QueryPoolPreparationError(f"无法读取{description}: {path}") from error
    if not records:
        raise QueryPoolPreparationError(f"{description}不能为空")
    return records


def _verify_adjacent_hash(path: Path, description: str) -> Path:
    hash_path = path.with_suffix(".sha256")
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QueryPoolPreparationError(
            f"无法读取{description}相邻 SHA-256"
        ) from error
    if lines != [f"{_sha256_file(path)}  {path.name}"]:
        raise QueryPoolPreparationError(f"{description}相邻 SHA-256 无效")
    return hash_path


def _bound_path(metadata: object, description: str) -> Path:
    if not isinstance(metadata, dict):
        raise QueryPoolPreparationError(f"canonical manifest 缺少{description}身份")
    path_value = metadata.get("path")
    if (
        not isinstance(path_value, str)
        or type(metadata.get("bytes")) is not int
        or not isinstance(metadata.get("sha256"), str)
    ):
        raise QueryPoolPreparationError(f"canonical manifest 的{description}身份无效")
    path = Path(path_value).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != metadata["bytes"]
        or _sha256_file(path) != metadata["sha256"]
    ):
        raise QueryPoolPreparationError(f"{description}身份已变化")
    return path


def _normalized_query(value: object, prefix: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QueryPoolPreparationError(f"{prefix} query_original 必须是非空字符串")
    normalized = unicodedata.normalize("NFC", value.strip())
    if normalized != value or "\n" in value or "\r" in value:
        raise QueryPoolPreparationError(
            f"{prefix} query_original 必须是 NFC、无首尾空白的单行字符串"
        )
    if len(normalized) > QUERY_MAX_CHARS:
        raise QueryPoolPreparationError(
            f"{prefix} query_original 超过 {QUERY_MAX_CHARS} 个字符"
        )
    return normalized


def _required_gt(value: object, prefix: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not 1 <= len(value) <= REQUIRED_GT_MAX
        or any(not isinstance(item, str) or not item.strip() for item in value)
        or len(set(value)) != len(value)
    ):
        raise QueryPoolPreparationError(
            f"{prefix} required_chunk_ids 必须包含 1 至 {REQUIRED_GT_MAX} 个唯一非空字符串"
        )
    return list(value)


def _load_exclusions(path: Path) -> tuple[set[str], dict[str, Any]]:
    payload = _load_json(path, "评估排除清单")
    digests = payload.get("question_sha256")
    if (
        payload.get("complete_for_formal_sft") is not True
        or not isinstance(digests, list)
        or not digests
        or len(digests) != len(set(digests))
        or any(not isinstance(item, str) or not _SHA256_RE.fullmatch(item) for item in digests)
    ):
        raise QueryPoolPreparationError("评估排除清单状态或 question_sha256 无效")
    return set(digests), payload


def _publish(outputs: list[tuple[Path, str]]) -> None:
    occupied = [str(path) for path, _ in outputs if path.exists()]
    if occupied:
        raise QueryPoolPreparationError("目标输出已存在: " + ", ".join(occupied))
    partials = [
        (path.with_name(path.name + ".partial"), path, payload)
        for path, payload in outputs
    ]
    published = []
    try:
        for partial, final, payload in partials:
            final.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(payload, encoding="utf-8", newline="\n")
        for partial, final, _ in partials:
            partial.replace(final)
            published.append(final)
    except (OSError, UnicodeError) as error:
        for path in [*(item[0] for item in partials), *reversed(published)]:
            path.unlink(missing_ok=True)
        raise QueryPoolPreparationError("无法发布公共 query 池 candidate") from error


def prepare_query_pool_source_candidate(
    *,
    canonical_manifest_path: Path,
    evaluation_exclusions_path: Path,
    candidate_output: Path,
    mapping_output: Path,
    manifest_output: Path,
    expected_records: int = EXPECTED_PROJECT_RECORDS,
) -> dict[str, object]:
    """绑定 575 条 canonical 来源并发布独立公共 ID 与最小三字段接口。"""

    canonical_manifest_path = Path(canonical_manifest_path).resolve()
    evaluation_exclusions_path = Path(evaluation_exclusions_path).resolve()
    candidate_output = Path(candidate_output).resolve()
    mapping_output = Path(mapping_output).resolve()
    manifest_output = Path(manifest_output).resolve()
    hash_output = manifest_output.with_suffix(".sha256")
    occupied = [
        str(path)
        for path in (candidate_output, mapping_output, manifest_output, hash_output)
        if path.exists()
    ]
    if occupied:
        raise QueryPoolPreparationError("目标输出已存在: " + ", ".join(occupied))

    canonical_hash_path = _verify_adjacent_hash(
        canonical_manifest_path, "canonical manifest"
    )
    exclusions_hash_path = _verify_adjacent_hash(
        evaluation_exclusions_path, "评估排除清单"
    )
    canonical_manifest = _load_json(canonical_manifest_path, "canonical manifest")
    if (
        canonical_manifest.get("pipeline")
        != "rag_sft_canonical_authoring_correction"
        or canonical_manifest.get("complete") is not True
        or canonical_manifest.get("validation", {}).get("all_records_approved")
        is not True
        or canonical_manifest.get("validation", {}).get(
            "article_and_support_spans_validated"
        )
        is not True
    ):
        raise QueryPoolPreparationError("canonical manifest 状态无效")
    canonical_path = _bound_path(
        canonical_manifest.get("output", {}).get("authoring"),
        "canonical authoring",
    )
    source_records = _load_jsonl(canonical_path, "canonical authoring")
    if len(source_records) != expected_records:
        raise QueryPoolPreparationError(
            f"canonical authoring 应为 {expected_records} 条，实际 {len(source_records)} 条"
        )

    exclusions, exclusion_payload = _load_exclusions(evaluation_exclusions_path)
    source_ids: set[str] = set()
    normalized_queries: dict[str, tuple[str, ...]] = {}
    candidates = []
    mappings = []
    gt_counts: Counter[str] = Counter()
    for position, source_record in enumerate(source_records, start=1):
        prefix = f"canonical authoring 第 {position} 条"
        source_id = source_record.get("id")
        if (
            not isinstance(source_id, str)
            or not _SOURCE_ID_RE.fullmatch(source_id)
            or source_id in source_ids
        ):
            raise QueryPoolPreparationError(f"{prefix} id 无效或重复")
        source_ids.add(source_id)
        if source_record.get("review_status") != "approved":
            raise QueryPoolPreparationError(f"{prefix}尚未批准")

        query = _normalized_query(source_record.get("query_original"), prefix)
        required = _required_gt(source_record.get("required_chunk_ids"), prefix)
        normalized_key = " ".join(query.split())
        gt_signature = tuple(required)
        if normalized_key in normalized_queries:
            previous = normalized_queries[normalized_key]
            if previous != gt_signature:
                raise QueryPoolPreparationError(
                    f"{prefix}与既有 query 相同但 required GT 不同"
                )
            raise QueryPoolPreparationError(f"{prefix}与既有 query 重复")
        normalized_queries[normalized_key] = gt_signature
        digest = question_digest(query)
        if digest in exclusions:
            raise QueryPoolPreparationError(f"{prefix}命中评估排除清单")

        query_id = f"query:{position:04d}"
        if not _QUERY_ID_RE.fullmatch(query_id):
            raise QueryPoolPreparationError("公共 query_id 超出四位编号范围")
        candidate = {
            "query_id": query_id,
            "query_original": query,
            "required_chunk_ids": required,
        }
        if set(candidate) != _CANDIDATE_FIELDS:
            raise QueryPoolPreparationError("公共 query 池字段与冻结 schema 不一致")
        candidates.append(candidate)
        mappings.append(
            {
                "source_id": source_id,
                "query_id": query_id,
                "source_record_sha256": _record_digest(source_record),
                "query_sha256": digest,
            }
        )
        gt_counts[str(len(required))] += 1

    candidate_payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in candidates
    )
    mapping_payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in mappings
    )
    manifest: dict[str, object] = {
        "pipeline": "query_pool_source_candidate_v1",
        "release_status": "source_candidate",
        "inputs": {
            "canonical_manifest": {
                **_identity(canonical_manifest_path),
                "hash_manifest": _identity(canonical_hash_path),
            },
            "canonical_authoring": _identity(
                canonical_path, records=len(source_records)
            ),
            "evaluation_exclusions": {
                **_identity(evaluation_exclusions_path),
                "hash_manifest": _identity(exclusions_hash_path),
                "digest_count": len(exclusions),
                "scope": exclusion_payload.get("scope"),
            },
        },
        "policy": {
            "record_fields": [
                "query_id",
                "query_original",
                "required_chunk_ids",
            ],
            "query_id_format": "query:NNNN",
            "source_order": "canonical_authoring_file_order",
            "query_max_chars": QUERY_MAX_CHARS,
            "required_gt_min": 1,
            "required_gt_max": REQUIRED_GT_MAX,
            "evaluation_isolation": "normalized_question_sha256",
        },
        "records": {
            "source": len(source_records),
            "candidate": len(candidates),
            "unique_source_ids": len(source_ids),
            "unique_query_ids": len({item["query_id"] for item in candidates}),
            "unique_normalized_queries": len(normalized_queries),
            "by_required_gt_count": dict(sorted(gt_counts.items())),
        },
        "validation": {
            "canonical_identity_bound": True,
            "all_source_records_approved": True,
            "ids_contiguous_and_unique": True,
            "queries_unique_and_within_limit": True,
            "required_gt_nonempty_and_within_limit": True,
            "evaluation_question_overlap": 0,
            "old_assets_modified": False,
        },
        "outputs": {
            "candidate": {
                "path": str(candidate_output),
                "records": len(candidates),
                "bytes": len(candidate_payload.encode("utf-8")),
                "sha256": _sha256_bytes(candidate_payload.encode("utf-8")),
            },
            "source_mapping": {
                "path": str(mapping_output),
                "records": len(mappings),
                "bytes": len(mapping_payload.encode("utf-8")),
                "sha256": _sha256_bytes(mapping_payload.encode("utf-8")),
            },
        },
        "readiness": {
            "interface_frozen": True,
            "id_mapping_frozen": True,
            "final_query_text_frozen": False,
            "query_sft_training_ready": False,
            "rag_sft_clean_materialization_ready": False,
        },
        "complete": True,
    }
    manifest_payload = json.dumps(
        manifest, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    hash_payload = (
        f"{_sha256_bytes(manifest_payload.encode('utf-8'))}  {manifest_output.name}\n"
    )
    _publish(
        [
            (candidate_output, candidate_payload),
            (mapping_output, mapping_payload),
            (manifest_output, manifest_payload),
            (hash_output, hash_payload),
        ]
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-manifest", type=Path, default=DEFAULT_CANONICAL_MANIFEST
    )
    parser.add_argument(
        "--evaluation-exclusions", type=Path, default=DEFAULT_EVALUATION_EXCLUSIONS
    )
    parser.add_argument("--candidate-output", type=Path, default=DEFAULT_CANDIDATE_OUTPUT)
    parser.add_argument("--mapping-output", type=Path, default=DEFAULT_MAPPING_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    parser.add_argument(
        "--expected-records", type=int, default=EXPECTED_PROJECT_RECORDS
    )
    args = parser.parse_args()
    try:
        manifest = prepare_query_pool_source_candidate(
            canonical_manifest_path=args.canonical_manifest,
            evaluation_exclusions_path=args.evaluation_exclusions,
            candidate_output=args.candidate_output,
            mapping_output=args.mapping_output,
            manifest_output=args.manifest_output,
            expected_records=args.expected_records,
        )
    except QueryPoolPreparationError as error:
        parser.error(str(error))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
