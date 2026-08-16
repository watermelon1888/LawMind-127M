"""发布绑定当前 rubric 的 RAG-SFT v2 canonical 快照。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rag.knowledge import ArticleRepository

from .rag_sft_v2_contract import RagSftV2ContractError, validate_canonical_authoring
from .rag_sft_v2_projection import RagSftV2ProjectionError, project_rag_sft_v2_record


DATASET_ROOT = Path(__file__).resolve().parent
RAG_SFT_ROOT = DATASET_ROOT / "RAG-SFT"
REVIEW_ROOT = RAG_SFT_ROOT / "review" / "v2"
DEFAULT_SOURCE_ROOT = REVIEW_ROOT / "canonical-authoring-approved-v1"
DEFAULT_CORRECTIONS = REVIEW_ROOT / "canonical-authoring-corrections-v1" / "corrections.jsonl"
DEFAULT_RUBRIC = REVIEW_ROOT / "rubric-v2.md"
DEFAULT_CALIBRATION = REVIEW_ROOT / "calibration-v2" / "calibration-adjudication.json"
DEFAULT_QUERY_POOL = DATASET_ROOT / "QUERY-POOL" / "authoring" / "query-pool-v1.jsonl"
DEFAULT_ARTICLE_INDEX = Path(__file__).resolve().parents[2] / "rag" / "chunk" / "article_index.jsonl"
DEFAULT_OUTPUT_ROOT = REVIEW_ROOT / "canonical-authoring-approved-v2"

EXPECTED_BATCHES = tuple(f"batch-{number:03d}" for number in range(1, 20))
CANONICAL_FIELDS = {"query_id", "query_original", "claims", "summary"}
POOL_FIELDS = {"query_id", "query_original", "required_chunk_ids"}
CORRECTION_FIELDS = {"query_id", "source_summary_sha256", "summary", "reason"}


class RagSftV2CanonicalSnapshotError(RuntimeError):
    """当前审核规则下的 canonical 快照无法可靠发布。"""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    identity: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if records is not None:
        identity["records"] = records
    return identity


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftV2CanonicalSnapshotError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise RagSftV2CanonicalSnapshotError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagSftV2CanonicalSnapshotError(
                        f"{description}不允许空行: {number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RagSftV2CanonicalSnapshotError(
                        f"{description}第 {number} 条必须是对象"
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftV2CanonicalSnapshotError):
            raise
        raise RagSftV2CanonicalSnapshotError(f"无法读取{description}: {path}") from error
    if not rows:
        raise RagSftV2CanonicalSnapshotError(f"{description}不能为空")
    return rows


def _verify_hash_manifest(directory: Path) -> None:
    try:
        lines = (directory / "manifest.sha256").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftV2CanonicalSnapshotError("源 canonical 批次缺少哈希清单") from error
    parsed: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or Path(parts[1]).name != parts[1]:
            raise RagSftV2CanonicalSnapshotError("源 canonical 批次哈希清单格式无效")
        parsed[parts[1]] = parts[0]
    if set(parsed) != {"canonical-authoring.jsonl", "manifest.json"}:
        raise RagSftV2CanonicalSnapshotError("源 canonical 批次哈希文件集合无效")
    if any(_sha256(directory / name) != digest for name, digest in parsed.items()):
        raise RagSftV2CanonicalSnapshotError("源 canonical 批次哈希不一致")


def _load_pool_and_contents(
    query_pool_path: Path, article_index_path: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    pool: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(query_pool_path, "公共 Query 池"):
        query_id = row.get("query_id")
        if set(row) != POOL_FIELDS or not isinstance(query_id, str) or query_id in pool:
            raise RagSftV2CanonicalSnapshotError("公共 Query 池 schema 无效")
        pool[query_id] = row
    contents: dict[str, str] = {}
    for row in _load_jsonl(article_index_path, "法条索引"):
        chunk_id, content = row.get("chunk_id"), row.get("content")
        if not isinstance(chunk_id, str) or not isinstance(content, str) or chunk_id in contents:
            raise RagSftV2CanonicalSnapshotError("法条索引 schema 无效")
        contents[chunk_id] = content
    return pool, contents


def _load_corrections(path: Path, source_records: dict[str, dict[str, Any]]) -> dict[str, str]:
    corrections: dict[str, str] = {}
    for number, row in enumerate(_load_jsonl(path, "摘要修订账本"), start=1):
        if set(row) != CORRECTION_FIELDS:
            raise RagSftV2CanonicalSnapshotError(f"摘要修订账本字段无效: {number}")
        values = {field: row.get(field) for field in CORRECTION_FIELDS}
        if any(not isinstance(value, str) or not value.strip() for value in values.values()):
            raise RagSftV2CanonicalSnapshotError(f"摘要修订账本字段为空: {number}")
        query_id = values["query_id"]
        source = source_records.get(query_id)
        if source is None or query_id in corrections:
            raise RagSftV2CanonicalSnapshotError(f"摘要修订 query_id 无效或重复: {query_id}")
        source_summary = source["summary"]
        if values["source_summary_sha256"] != hashlib.sha256(
            source_summary.encode("utf-8")
        ).hexdigest():
            raise RagSftV2CanonicalSnapshotError(f"摘要修订源哈希不匹配: {query_id}")
        if values["summary"] == source_summary:
            raise RagSftV2CanonicalSnapshotError(f"摘要修订没有改变摘要: {query_id}")
        corrections[query_id] = values["summary"]
    return corrections


def _write_batch(directory: Path, records: list[dict[str, object]], manifest: dict[str, object]) -> None:
    directory.mkdir(parents=True)
    canonical_path = directory / "canonical-authoring.jsonl"
    manifest_path = directory / "manifest.json"
    canonical_payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for record in records
    )
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    canonical_path.write_text(canonical_payload, encoding="utf-8", newline="\n")
    manifest_path.write_text(manifest_payload, encoding="utf-8", newline="\n")
    (directory / "manifest.sha256").write_text(
        f"{_sha256(canonical_path)}  {canonical_path.name}\n"
        f"{_sha256(manifest_path)}  {manifest_path.name}\n",
        encoding="utf-8",
        newline="\n",
    )


def publish_rag_sft_v2_canonical_snapshot(
    *,
    source_root: Path,
    corrections_path: Path,
    rubric_path: Path,
    calibration_path: Path,
    query_pool_path: Path,
    article_index_path: Path,
    output_root: Path,
) -> dict[str, object]:
    """将受控摘要修订固化为当前规则下的新的 canonical 身份。"""

    paths = {
        "source_root": Path(source_root).resolve(),
        "corrections": Path(corrections_path).resolve(),
        "rubric": Path(rubric_path).resolve(),
        "calibration": Path(calibration_path).resolve(),
        "query_pool": Path(query_pool_path).resolve(),
        "article_index": Path(article_index_path).resolve(),
        "output_root": Path(output_root).resolve(),
    }
    if paths["output_root"].exists():
        raise RagSftV2CanonicalSnapshotError("快照输出目录必须不存在")
    if not paths["source_root"].is_dir() or any(
        not paths[name].is_file()
        for name in ("corrections", "rubric", "calibration", "query_pool", "article_index")
    ):
        raise RagSftV2CanonicalSnapshotError("快照输入缺失")
    calibration = _load_json(paths["calibration"], "校准完成 manifest")
    rubric_identity = calibration.get("inputs", {}).get("rubric")
    if (
        calibration.get("release_status") != "rubric_calibrated"
        or calibration.get("readiness", {}).get("rubric_calibrated") is not True
        or not isinstance(rubric_identity, dict)
        or rubric_identity.get("sha256") != _sha256(paths["rubric"])
    ):
        raise RagSftV2CanonicalSnapshotError("校准完成 manifest 未绑定当前 rubric")

    pool, contents = _load_pool_and_contents(paths["query_pool"], paths["article_index"])
    source_batches = sorted(path.name for path in paths["source_root"].iterdir() if path.is_dir())
    if tuple(source_batches) != EXPECTED_BATCHES:
        raise RagSftV2CanonicalSnapshotError("源 canonical 批次集合无效")
    source_by_batch: dict[str, list[dict[str, Any]]] = {}
    source_by_id: dict[str, dict[str, Any]] = {}
    for name in EXPECTED_BATCHES:
        directory = paths["source_root"] / name
        _verify_hash_manifest(directory)
        manifest = _load_json(directory / "manifest.json", f"{name} manifest")
        rows = _load_jsonl(directory / "canonical-authoring.jsonl", f"{name} canonical")
        if manifest.get("release_status") != "canonical_batch_approved":
            raise RagSftV2CanonicalSnapshotError(f"{name} 不是已批准 canonical 批次")
        for row in rows:
            query_id = row.get("query_id")
            if set(row) != CANONICAL_FIELDS or not isinstance(query_id, str) or query_id in source_by_id:
                raise RagSftV2CanonicalSnapshotError("源 canonical schema 或 query_id 无效")
            source_by_id[query_id] = row
        source_by_batch[name] = rows
    if len(source_by_id) != 550:
        raise RagSftV2CanonicalSnapshotError("源 canonical 总数不是 550")
    corrections = _load_corrections(paths["corrections"], source_by_id)
    try:
        repository = ArticleRepository.from_jsonl(paths["article_index"])
    except Exception as error:
        raise RagSftV2CanonicalSnapshotError("无法加载法条仓库") from error

    paths["output_root"].mkdir(parents=True)
    published: list[Path] = []
    try:
        for name in EXPECTED_BATCHES:
            output_rows: list[dict[str, object]] = []
            for source in source_by_batch[name]:
                row = dict(source)
                query_id = row["query_id"]
                if query_id in corrections:
                    row["summary"] = corrections[query_id]
                try:
                    validated = validate_canonical_authoring(row, pool[query_id], contents)
                    project_rag_sft_v2_record(
                        validated,
                        {
                            chunk_id: repository.get_by_chunk_id(chunk_id)
                            for chunk_id in validated["required_chunk_ids"]
                        },
                    )
                except (RagSftV2ContractError, RagSftV2ProjectionError) as error:
                    raise RagSftV2CanonicalSnapshotError(
                        f"当前协议下 canonical 无法投影: {query_id}"
                    ) from error
                output_rows.append({field: validated[field] for field in CANONICAL_FIELDS})
            source_directory = paths["source_root"] / name
            manifest = {
                "pipeline": "rag_sft_v2_canonical_snapshot",
                "release_status": "canonical_snapshot_approved",
                "inputs": {
                    "source_batch": {
                        "canonical_authoring": _identity(
                            source_directory / "canonical-authoring.jsonl", records=len(output_rows)
                        ),
                        "manifest": _identity(source_directory / "manifest.json"),
                        "sha256_manifest": _identity(source_directory / "manifest.sha256"),
                    },
                    "canonical_summary_corrections": _identity(
                        paths["corrections"], records=len(corrections)
                    ),
                    "rubric": _identity(paths["rubric"]),
                    "calibration": _identity(paths["calibration"]),
                    "query_pool": _identity(paths["query_pool"], records=len(pool)),
                    "article_index": _identity(paths["article_index"], records=len(contents)),
                },
                "records": {
                    "canonical_authoring": len(output_rows),
                    "summary_corrections_applied": sum(
                        row["query_id"] in corrections for row in output_rows
                    ),
                },
                "policy": {
                    "canonical_schema": ["query_id", "query_original", "claims", "summary"],
                    "source_canonical_immutable": True,
                    "summary_corrections_materialized": True,
                },
                "readiness": {
                    "canonical_snapshot_approved": True,
                    "oracle_clean_emitted": False,
                    "training_ready": False,
                },
                "complete": True,
            }
            directory = paths["output_root"] / name
            _write_batch(directory, output_rows, manifest)
            published.append(directory)
    except Exception:
        for directory in reversed(published):
            for path in directory.iterdir():
                path.unlink()
            directory.rmdir()
        paths["output_root"].rmdir()
        raise
    return {
        "records": 550,
        "batches": len(EXPECTED_BATCHES),
        "summary_corrections_applied": len(corrections),
        "output_root": str(paths["output_root"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS)
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--query-pool", type=Path, default=DEFAULT_QUERY_POOL)
    parser.add_argument("--article-index", type=Path, default=DEFAULT_ARTICLE_INDEX)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    try:
        report = publish_rag_sft_v2_canonical_snapshot(
            source_root=args.source_root,
            corrections_path=args.corrections,
            rubric_path=args.rubric,
            calibration_path=args.calibration,
            query_pool_path=args.query_pool,
            article_index_path=args.article_index,
            output_root=args.output_root,
        )
    except RagSftV2CanonicalSnapshotError as error:
        raise SystemExit(f"[失败] {error}") from error
    print(
        f"[完成] 发布 {report['records']} 条 canonical，"
        f"{report['summary_corrections_applied']} 条摘要修订已物化"
    )


if __name__ == "__main__":
    main()
