"""从冻结公共 query source candidate 发布正式全量 Query-SFT 分批盲构造工作包。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DATASET_ROOT = Path(__file__).resolve().parent
QUERY_POOL_ROOT = DATASET_ROOT / "QUERY-POOL"
DEFAULT_SOURCE_MANIFEST = QUERY_POOL_ROOT / "manifests" / "query-pool-v1-source-candidate.json"
DEFAULT_SOURCE_CANDIDATE = QUERY_POOL_ROOT / "authoring" / "query-pool-v1-source-candidate.jsonl"
DEFAULT_EXCLUSIONS = QUERY_POOL_ROOT / "authoring" / "query-sft-v1-source-exclusions.jsonl"
DEFAULT_OUTPUT_DIR = QUERY_POOL_ROOT / "full" / "query-sft-v1-work-package"

EXPECTED_SOURCE_RECORDS = 575
EXPECTED_FINAL_RECORDS = 574
BATCH_SIZE = 72
EXCLUSION_FIELDS = {"source_id", "retained_source_id", "reason"}
SOURCE_FIELDS = {"query_id", "query_original", "required_chunk_ids"}
BLIND_FIELDS = {"work_id", "source_id", "source_query"}
REFERENCE_FIELDS = {"work_id", "source_id", "source_query", "required_chunk_ids"}
PLAN_FILENAME = "query-sft-v1-batch-plan.json"
MANIFEST_FILENAME = "query-sft-v1-work-package.json"
HASH_FILENAME = "query-sft-v1-work-package.sha256"


class QuerySftFullPreparationError(RuntimeError):
    """表示全量 Query-SFT 工作包的来源、排除或发布状态无效。"""


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
        raise QuerySftFullPreparationError(f"无法读取{label}: {path}") from error
    if not isinstance(value, dict):
        raise QuerySftFullPreparationError(f"{label}必须是 JSON 对象")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    raise QuerySftFullPreparationError(f"{label}不允许空行: {number}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise QuerySftFullPreparationError(f"{label}第 {number} 条必须是对象")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftFullPreparationError):
            raise
        raise QuerySftFullPreparationError(f"无法读取{label}: {path}") from error
    if not rows:
        raise QuerySftFullPreparationError(f"{label}不能为空")
    return rows


def _verify_sidecar(path: Path, label: str) -> Path:
    sidecar = path.with_suffix(".sha256")
    try:
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftFullPreparationError(f"无法读取{label} SHA-256") from error
    if lines != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftFullPreparationError(f"{label} SHA-256 无效")
    return sidecar


def _load_exclusions(path: Path) -> dict[str, str]:
    _verify_sidecar(path, "全量 source 排除清单")
    rows = _load_jsonl(path, "全量 source 排除清单")
    if len(rows) != 1 or set(rows[0]) != EXCLUSION_FIELDS:
        raise QuerySftFullPreparationError("全量 source 排除清单必须恰好为一条固定记录")
    row = rows[0]
    if (
        row.get("source_id") != "query:0044"
        or row.get("retained_source_id") != "query:0035"
        or not isinstance(row.get("reason"), str)
        or not row["reason"]
    ):
        raise QuerySftFullPreparationError("全量 source 排除清单内容无效")
    return {row["source_id"]: row["retained_source_id"]}


def _payload(rows: list[dict[str, object]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    )


def _publish(output_dir: Path, payloads: dict[str, str]) -> None:
    if output_dir.exists():
        raise QuerySftFullPreparationError(f"输出目录必须不存在: {output_dir}")
    try:
        output_dir.mkdir(parents=True)
        for name, payload in payloads.items():
            target = output_dir / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(payload, encoding="utf-8", newline="\n")
        (output_dir / HASH_FILENAME).write_text(
            "".join(
                f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()}  {name}\n"
                for name, payload in payloads.items()
            ),
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, UnicodeError) as error:
        raise QuerySftFullPreparationError("无法发布全量 Query-SFT 工作包") from error


def prepare_query_sft_full(
    *,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
    source_candidate_path: Path = DEFAULT_SOURCE_CANDIDATE,
    exclusions_path: Path = DEFAULT_EXCLUSIONS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    """绑定 575 条来源、排除批准的重复项并发布八个隔离构造批次。"""

    source_manifest_path = Path(source_manifest_path).resolve()
    source_candidate_path = Path(source_candidate_path).resolve()
    exclusions_path = Path(exclusions_path).resolve()
    output_dir = Path(output_dir).resolve()
    source_manifest_hash = _verify_sidecar(source_manifest_path, "source candidate manifest")
    source_manifest = _load_json(source_manifest_path, "source candidate manifest")
    if (
        source_manifest.get("pipeline") != "query_pool_source_candidate_v1"
        or source_manifest.get("release_status") != "source_candidate"
        or source_manifest.get("complete") is not True
        or source_manifest.get("readiness", {}).get("interface_frozen") is not True
        or source_manifest.get("readiness", {}).get("final_query_text_frozen") is not False
        or source_manifest.get("records", {}).get("candidate") != EXPECTED_SOURCE_RECORDS
    ):
        raise QuerySftFullPreparationError("source candidate manifest 状态无效")
    source_output = source_manifest.get("outputs", {}).get("candidate")
    if (
        not isinstance(source_output, dict)
        or source_output.get("bytes") != source_candidate_path.stat().st_size
        or source_output.get("sha256") != _sha256_file(source_candidate_path)
        or source_output.get("records") != EXPECTED_SOURCE_RECORDS
    ):
        raise QuerySftFullPreparationError("source candidate 身份未与 manifest 绑定")
    exclusions = _load_exclusions(exclusions_path)
    sources = _load_jsonl(source_candidate_path, "source candidate")
    if len(sources) != EXPECTED_SOURCE_RECORDS:
        raise QuerySftFullPreparationError("source candidate 记录数无效")
    retained = []
    seen_ids = set()
    seen_queries = {}
    for position, source in enumerate(sources, 1):
        if set(source) != SOURCE_FIELDS:
            raise QuerySftFullPreparationError(f"source candidate 字段无效: {position}")
        source_id = source.get("query_id")
        query = source.get("query_original")
        required = source.get("required_chunk_ids")
        if (
            not isinstance(source_id, str)
            or source_id in seen_ids
            or not isinstance(query, str)
            or not query
            or not isinstance(required, list)
            or not 1 <= len(required) <= 3
            or any(not isinstance(item, str) or not item for item in required)
        ):
            raise QuerySftFullPreparationError(f"source candidate 映射无效: {position}")
        seen_ids.add(source_id)
        if source_id in exclusions:
            continue
        normalized = " ".join(query.split())
        signature = tuple(required)
        if normalized in seen_queries:
            raise QuerySftFullPreparationError("正式全量范围仍有重复 query")
        seen_queries[normalized] = signature
        retained.append(source)
    if len(retained) != EXPECTED_FINAL_RECORDS:
        raise QuerySftFullPreparationError("批准排除后的正式全量记录数必须为 574")
    retained_ids = {row["query_id"] for row in retained}
    if exclusions["query:0044"] not in retained_ids or "query:0044" in retained_ids:
        raise QuerySftFullPreparationError("重复 query 排除与保留映射无效")

    payloads: dict[str, str] = {}
    batch_plan = []
    for offset in range(0, len(retained), BATCH_SIZE):
        batch_number = offset // BATCH_SIZE + 1
        batch = retained[offset : offset + BATCH_SIZE]
        blind_rows = [
            {
                "work_id": f"query_sft_v1:{offset + index + 1:04d}",
                "source_id": source["query_id"],
                "source_query": source["query_original"],
            }
            for index, source in enumerate(batch)
        ]
        reference_rows = [
            {
                **blind,
                "required_chunk_ids": source["required_chunk_ids"],
            }
            for blind, source in zip(blind_rows, batch, strict=True)
        ]
        if any(set(row) != BLIND_FIELDS for row in blind_rows) or any(
            set(row) != REFERENCE_FIELDS for row in reference_rows
        ):
            raise QuerySftFullPreparationError("全量批次字段范围无效")
        blind_name = f"blind/batch-{batch_number:02d}.jsonl"
        reference_name = f"audit-reference/batch-{batch_number:02d}.jsonl"
        blind_payload = _payload(blind_rows)
        reference_payload = _payload(reference_rows)
        payloads[blind_name] = blind_payload
        payloads[reference_name] = reference_payload
        batch_plan.append(
            {
                "batch": batch_number,
                "records": len(batch),
                "work_id_first": blind_rows[0]["work_id"],
                "work_id_last": blind_rows[-1]["work_id"],
                "blind_input": blind_name,
                "audit_reference": reference_name,
            }
        )
    if len(batch_plan) != 8 or sum(row["records"] for row in batch_plan) != 574:
        raise QuerySftFullPreparationError("全量批次计划未闭合")
    plan = {
        "pipeline": "query_sft_full_batch_plan_v1",
        "batch_size": BATCH_SIZE,
        "records": 574,
        "batches": batch_plan,
        "blind_generation_contract": "只读取 blind/batch-*.jsonl，不读取 GT、法条、答案、评估题或检索分数。",
        "independent_audit_contract": "仅独立审核角色读取 audit-reference/batch-*.jsonl 的 required_chunk_ids。",
        "complete": True,
    }
    plan_payload = json.dumps(plan, ensure_ascii=False, indent=2) + "\n"
    payloads[PLAN_FILENAME] = plan_payload
    manifest = {
        "pipeline": "query_sft_full_work_package_v1",
        "release_status": "full_input_construction_pending",
        "inputs": {
            "source_candidate_manifest": {**_identity(source_manifest_path), "sha256_manifest": _identity(source_manifest_hash)},
            "source_candidate": _identity(source_candidate_path, records=575),
            "source_exclusions": {**_identity(exclusions_path, records=1), "sha256_manifest": _identity(exclusions_path.with_suffix(".sha256"))},
        },
        "deduplication": {"source_records": 575, "excluded_records": 1, "retained_records": 574, "excluded_source_id": "query:0044", "retained_equivalent_source_id": "query:0035"},
        "batches": {"count": 8, "size": BATCH_SIZE, "records": 574},
        "validation": {"source_identity_bound": True, "approved_duplicate_exclusion_applied": True, "retained_queries_unique": True, "blind_and_gt_reference_views_separated": True, "batch_coverage_closed": True},
        "readiness": {"full_input_construction_ready": True, "teacher_generation_ready": True, "independent_gt_audit_ready": True, "query_sft_training_ready": False},
        "complete": True,
    }
    payloads[MANIFEST_FILENAME] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    _publish(output_dir, payloads)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--source-candidate", type=Path, default=DEFAULT_SOURCE_CANDIDATE)
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        manifest = prepare_query_sft_full(
            source_manifest_path=args.source_manifest,
            source_candidate_path=args.source_candidate,
            exclusions_path=args.exclusions,
            output_dir=args.output_dir,
        )
    except QuerySftFullPreparationError as error:
        parser.error(str(error))
    print("QUERY_SFT_FULL_WORK_PACKAGE_OK records=574 batches=8")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
