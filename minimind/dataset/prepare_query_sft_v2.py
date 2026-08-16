"""从正式公共 Query 池发布 Query-SFT v2 的 GT 隔离构造工作包。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .query_sft_v2_contract import AUTHORING_TYPES


DATASET_ROOT = Path(__file__).resolve().parent
QUERY_POOL_ROOT = DATASET_ROOT / "QUERY-POOL"
DEFAULT_QUERY_POOL = QUERY_POOL_ROOT / "authoring" / "query-pool-v1.jsonl"
DEFAULT_QUERY_POOL_MANIFEST = QUERY_POOL_ROOT / "manifests" / "query-pool-v1.json"
DEFAULT_TYPE_MAPPING = (
    QUERY_POOL_ROOT / "full" / "query-sft-v1-work-package" / "query-sft-v1-input-candidate-r1.jsonl"
)
DEFAULT_OUTPUT_DIR = QUERY_POOL_ROOT / "full" / "query-sft-v2-work-package"

EXPECTED_RECORDS = 574
BATCH_SIZE = 72
WORK_PACKAGE_FILENAME = "query-sft-v2-work-package.json"
HASH_FILENAME = "query-sft-v2-work-package.sha256"
INPUT_FILENAME = "query-sft-v2-input-candidate.jsonl"
BLIND_FIELDS = ("work_id", "query_original")
REFERENCE_FIELDS = (
    "work_id",
    "source_id",
    "authoring_type",
    "query_original",
    "required_chunk_ids",
)
INPUT_FIELDS = ("work_id", "source_id", "authoring_type", "query_original")


class QuerySftV2PreparationError(RuntimeError):
    """表示 Query-SFT v2 构造工作包的上游身份或隔离状态无效。"""


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


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftV2PreparationError(f"无法读取{label}: {path}") from error
    if not isinstance(value, dict):
        raise QuerySftV2PreparationError(f"{label}必须是 JSON 对象")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    raise QuerySftV2PreparationError(f"{label}不允许空行: {number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise QuerySftV2PreparationError(f"{label}第 {number} 条必须是对象")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftV2PreparationError):
            raise
        raise QuerySftV2PreparationError(f"无法读取{label}: {path}") from error
    return rows


def _verify_sidecar(path: Path, label: str) -> Path:
    sidecar = path.with_suffix(".sha256")
    try:
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftV2PreparationError(f"无法读取{label} SHA-256") from error
    if lines != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftV2PreparationError(f"{label} SHA-256 无效")
    return sidecar


def _payload(rows: list[dict[str, object]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in rows
    )


def prepare_query_sft_v2(
    *,
    query_pool_path: Path = DEFAULT_QUERY_POOL,
    query_pool_manifest_path: Path = DEFAULT_QUERY_POOL_MANIFEST,
    type_mapping_path: Path = DEFAULT_TYPE_MAPPING,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    """绑定 574 条正式公共 query，发布教师不可见 GT 的 v2 工作包。"""

    query_pool_path = Path(query_pool_path).resolve()
    query_pool_manifest_path = Path(query_pool_manifest_path).resolve()
    type_mapping_path = Path(type_mapping_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise QuerySftV2PreparationError(f"v2 输出目录必须不存在: {output_dir}")
    manifest_hash = _verify_sidecar(query_pool_manifest_path, "正式公共 Query 池 manifest")
    mapping_hash = _verify_sidecar(type_mapping_path, "历史输入类型映射")
    pool_manifest = _load_json(query_pool_manifest_path, "正式公共 Query 池 manifest")
    if (
        pool_manifest.get("pipeline") != "query_pool_v1_formal_release"
        or pool_manifest.get("release_status") != "formal_query_pool"
        or pool_manifest.get("records", {}).get("queries") != EXPECTED_RECORDS
        or pool_manifest.get("readiness", {}).get("final_query_text_frozen") is not True
        or pool_manifest.get("complete") is not True
    ):
        raise QuerySftV2PreparationError("正式公共 Query 池尚未冻结")
    declared_pool = pool_manifest.get("data", {}).get("query_pool")
    if (
        not isinstance(declared_pool, dict)
        or declared_pool.get("bytes") != query_pool_path.stat().st_size
        or declared_pool.get("sha256") != _sha256_file(query_pool_path)
        or declared_pool.get("records") != EXPECTED_RECORDS
    ):
        raise QuerySftV2PreparationError("正式公共 Query 池未与正式 manifest 交叉核验")
    pool = _load_jsonl(query_pool_path, "正式公共 Query 池")
    if len(pool) != EXPECTED_RECORDS:
        raise QuerySftV2PreparationError("正式公共 Query 池记录数必须为 574")
    sources = {}
    for position, row in enumerate(pool, 1):
        if set(row) != {"query_id", "query_original", "required_chunk_ids"}:
            raise QuerySftV2PreparationError(f"正式公共 Query 池第 {position} 条字段无效")
        query_id = row.get("query_id")
        query = row.get("query_original")
        required = row.get("required_chunk_ids")
        if (
            not isinstance(query_id, str)
            or query_id in sources
            or not isinstance(query, str)
            or not query
            or not isinstance(required, list)
            or not 1 <= len(required) <= 3
            or any(not isinstance(value, str) or not value for value in required)
        ):
            raise QuerySftV2PreparationError(f"正式公共 Query 池第 {position} 条内容无效")
        sources[query_id] = row

    types = {}
    for position, row in enumerate(_load_jsonl(type_mapping_path, "历史输入类型映射"), 1):
        if set(row) != set(INPUT_FIELDS):
            raise QuerySftV2PreparationError(f"历史输入类型映射第 {position} 条字段无效")
        source_id = row.get("source_id")
        authoring_type = row.get("authoring_type")
        if (
            not isinstance(source_id, str)
            or source_id not in sources
            or source_id in types
            or authoring_type not in AUTHORING_TYPES
            or row.get("query_original") != sources[source_id]["query_original"]
        ):
            raise QuerySftV2PreparationError(f"历史输入类型映射第 {position} 条未与正式公共池闭合")
        types[source_id] = authoring_type
    if set(types) != set(sources):
        raise QuerySftV2PreparationError("历史输入类型映射必须完整覆盖正式公共 Query 池")

    inputs = []
    blind_payloads: dict[str, str] = {}
    reference_payloads: dict[str, str] = {}
    batches = []
    ordered_pool = list(sources.values())
    for offset in range(0, EXPECTED_RECORDS, BATCH_SIZE):
        batch_number = offset // BATCH_SIZE + 1
        source_batch = ordered_pool[offset : offset + BATCH_SIZE]
        blind_rows = []
        reference_rows = []
        for index, source in enumerate(source_batch, 1):
            work_id = f"query_sft_v2:{offset + index:04d}"
            input_row = {
                "work_id": work_id,
                "source_id": source["query_id"],
                "authoring_type": types[source["query_id"]],
                "query_original": source["query_original"],
            }
            inputs.append(input_row)
            blind_rows.append({"work_id": work_id, "query_original": source["query_original"]})
            reference_rows.append({**input_row, "required_chunk_ids": source["required_chunk_ids"]})
        if any(tuple(row) != BLIND_FIELDS for row in blind_rows) or any(
            tuple(row) != REFERENCE_FIELDS for row in reference_rows
        ):
            raise AssertionError("v2 工作包字段顺序意外变化")
        blind_name = f"blind/batch-{batch_number:02d}.jsonl"
        reference_name = f"audit-reference/batch-{batch_number:02d}.jsonl"
        blind_payloads[blind_name] = _payload(blind_rows)
        reference_payloads[reference_name] = _payload(reference_rows)
        batches.append(
            {
                "batch": batch_number,
                "records": len(blind_rows),
                "blind_input": blind_name,
                "audit_reference": reference_name,
            }
        )
    if len(inputs) != EXPECTED_RECORDS or len(batches) != 8:
        raise AssertionError("v2 工作包记录数未闭合")
    package = {
        "schema_version": "1.0",
        "pipeline": "query_sft_v2_work_package",
        "release_status": "teacher_candidate_generation_pending",
        "inputs": {
            "formal_query_pool": _identity(query_pool_path, records=EXPECTED_RECORDS),
            "formal_query_pool_manifest": {**_identity(query_pool_manifest_path), "hash_manifest": _identity(manifest_hash)},
            "authoring_type_mapping": {**_identity(type_mapping_path, records=EXPECTED_RECORDS), "hash_manifest": _identity(mapping_hash)},
        },
        "records": {
            "query_inputs": EXPECTED_RECORDS,
            "by_authoring_type": dict(sorted(Counter(types.values()).items())),
            "batches": len(batches),
        },
        "generation_protocol": {
            "teacher_visible_record_fields": ["work_id", "query_original"],
            "forbidden_teacher_context": ["source_id", "required_chunk_ids", "法条正文", "法律答案", "评估题", "检索结果", "检索分数", "旧版候选", "authoring_type"],
            "candidate_slots_per_query": 3,
        },
        "review_protocol": {
            "reviewer_may_read_required_gt": True,
            "required_gt_must_not_be_emitted_to_training": True,
            "authoring_type_is_a_quality_constraint": True,
        },
        "batches": batches,
        "readiness": {
            "teacher_candidate_generation_ready": True,
            "independent_semantic_review_ready": False,
            "frozen_retrieval_selection_ready": False,
            "training_ready": False,
        },
        "complete": True,
    }
    payloads = {
        INPUT_FILENAME: _payload(inputs),
        **blind_payloads,
        **reference_payloads,
        WORK_PACKAGE_FILENAME: json.dumps(package, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    }
    try:
        output_dir.mkdir(parents=True)
        for name, payload in payloads.items():
            path = output_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8", newline="\n")
        (output_dir / HASH_FILENAME).write_text(
            "".join(
                f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()}  {name}\n"
                for name, payload in payloads.items()
            ),
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, UnicodeError) as error:
        raise QuerySftV2PreparationError("无法发布 Query-SFT v2 工作包") from error
    return package


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-pool", type=Path, default=DEFAULT_QUERY_POOL)
    parser.add_argument("--query-pool-manifest", type=Path, default=DEFAULT_QUERY_POOL_MANIFEST)
    parser.add_argument("--type-mapping", type=Path, default=DEFAULT_TYPE_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        package = prepare_query_sft_v2(
            query_pool_path=args.query_pool,
            query_pool_manifest_path=args.query_pool_manifest,
            type_mapping_path=args.type_mapping,
            output_dir=args.output_dir,
        )
    except QuerySftV2PreparationError as error:
        parser.error(str(error))
    print(f"QUERY_SFT_V2_WORK_PACKAGE_OK records={package['records']['query_inputs']}")


if __name__ == "__main__":
    main()
