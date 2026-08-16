"""对冻结 canonical RAG-SFT 应用显式内容修订并发布新身份。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rag.knowledge import ArticleRepository

try:
    from . import materialize_rag_sft_retrieved as materializer
    from . import prepare_rag_sft as preparation
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import materialize_rag_sft_retrieved as materializer
    from dataset import prepare_rag_sft as preparation


RAG_SFT_ROOT = Path(__file__).resolve().parent / "RAG-SFT"
DEFAULT_PARENT_MANIFEST = RAG_SFT_ROOT / "manifests" / "rag-sft-canonical-v1.json"
DEFAULT_CORRECTIONS = RAG_SFT_ROOT / "authoring" / "canonical-corrections-20260809.jsonl"
DEFAULT_ARTICLE_INDEX = Path(__file__).resolve().parents[2] / "rag" / "chunk" / "article_index.jsonl"
DEFAULT_AUTHORING_OUTPUT = RAG_SFT_ROOT / "authoring" / "rag-sft-canonical-v2.jsonl"
DEFAULT_MAPPING_OUTPUT = RAG_SFT_ROOT / "manifests" / "rag-sft-canonical-v2-corrections.jsonl"
DEFAULT_MANIFEST_OUTPUT = RAG_SFT_ROOT / "manifests" / "rag-sft-canonical-v2.json"

CORRECTION_FIELDS = {"id", "expected_record_sha256", "reason", "replacement"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}$")


class RagSftAuthoringCorrectionError(RuntimeError):
    """canonical 修订未绑定当前记录或无法安全发布。"""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
        raise RagSftAuthoringCorrectionError(
            f"无法读取{description}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise RagSftAuthoringCorrectionError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagSftAuthoringCorrectionError(
                        f"{description}不允许空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RagSftAuthoringCorrectionError(
                        f"{description}第 {line_number} 条必须是 object"
                    )
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftAuthoringCorrectionError):
            raise
        raise RagSftAuthoringCorrectionError(
            f"无法读取{description}: {path}"
        ) from error
    if not records:
        raise RagSftAuthoringCorrectionError(f"{description}不能为空")
    return records


def _verify_adjacent_hash(path: Path, description: str) -> Path:
    hash_path = path.with_suffix(".sha256")
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftAuthoringCorrectionError(
            f"无法读取{description}相邻 SHA-256"
        ) from error
    if lines != [f"{_sha256_file(path)}  {path.name}"]:
        raise RagSftAuthoringCorrectionError(f"{description}相邻 SHA-256 无效")
    return hash_path


def _bound_path(metadata: object, description: str) -> Path:
    if not isinstance(metadata, dict):
        raise RagSftAuthoringCorrectionError(f"父 manifest 缺少{description}身份")
    path_value = metadata.get("path")
    if (
        not isinstance(path_value, str)
        or type(metadata.get("bytes")) is not int
        or not isinstance(metadata.get("sha256"), str)
    ):
        raise RagSftAuthoringCorrectionError(f"父 manifest 的{description}身份无效")
    path = Path(path_value).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != metadata["bytes"]
        or _sha256_file(path) != metadata["sha256"]
    ):
        raise RagSftAuthoringCorrectionError(f"{description}身份已变化")
    return path


def _publish(outputs: list[tuple[Path, str]]) -> None:
    occupied = [str(path) for path, _ in outputs if path.exists()]
    if occupied:
        raise RagSftAuthoringCorrectionError("目标输出已存在: " + ", ".join(occupied))
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
        raise RagSftAuthoringCorrectionError("无法发布 canonical 修订产物") from error


def apply_rag_sft_authoring_corrections(
    *,
    parent_manifest_path: Path,
    corrections_path: Path,
    article_index_path: Path,
    authoring_output: Path,
    mapping_output: Path,
    manifest_output: Path,
) -> dict[str, object]:
    """验证逐条替换身份并发布不覆盖父资产的新 canonical。"""

    parent_manifest_path = Path(parent_manifest_path).resolve()
    corrections_path = Path(corrections_path).resolve()
    article_index_path = Path(article_index_path).resolve()
    authoring_output = Path(authoring_output).resolve()
    mapping_output = Path(mapping_output).resolve()
    manifest_output = Path(manifest_output).resolve()
    hash_output = manifest_output.with_suffix(".sha256")
    occupied = [
        str(path)
        for path in (authoring_output, mapping_output, manifest_output, hash_output)
        if path.exists()
    ]
    if occupied:
        raise RagSftAuthoringCorrectionError("目标输出已存在: " + ", ".join(occupied))

    parent_hash_path = _verify_adjacent_hash(parent_manifest_path, "父 manifest")
    corrections_hash_path = _verify_adjacent_hash(corrections_path, "修订清单")
    parent_manifest = _load_json(parent_manifest_path, "父 manifest")
    if (
        parent_manifest.get("pipeline") != "rag_sft_canonical_authoring_assembly"
        or parent_manifest.get("complete") is not True
        or parent_manifest.get("readiness", {}).get(
            "canonical_authoring_schema_validated"
        )
        is not True
    ):
        raise RagSftAuthoringCorrectionError("父 manifest 状态无效")
    parent_authoring_path = _bound_path(
        parent_manifest.get("output", {}).get("authoring"),
        "canonical authoring",
    )
    bound_index_path = _bound_path(
        parent_manifest.get("inputs", {}).get("article_index"),
        "article index",
    )
    if bound_index_path != article_index_path:
        raise RagSftAuthoringCorrectionError("传入 article index 与父 manifest 不一致")
    parent_records = _load_jsonl(parent_authoring_path, "父 canonical authoring")
    corrections = _load_jsonl(corrections_path, "修订清单")
    parent_by_id = {}
    for position, record in enumerate(parent_records, start=1):
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id or record_id in parent_by_id:
            raise RagSftAuthoringCorrectionError(
                f"父 canonical 第 {position} 条 id 无效或重复"
            )
        parent_by_id[record_id] = record
    correction_by_id = {}
    for position, correction in enumerate(corrections, start=1):
        prefix = f"修订清单第 {position} 条"
        if set(correction) != CORRECTION_FIELDS:
            raise RagSftAuthoringCorrectionError(f"{prefix}字段必须精确匹配 schema")
        record_id = correction["id"]
        if not isinstance(record_id, str) or record_id not in parent_by_id:
            raise RagSftAuthoringCorrectionError(f"{prefix}引用未知 id")
        if record_id in correction_by_id:
            raise RagSftAuthoringCorrectionError(f"{prefix} id 重复")
        if not isinstance(correction["reason"], str) or not correction["reason"].strip():
            raise RagSftAuthoringCorrectionError(f"{prefix} reason 不能为空")
        expected = correction["expected_record_sha256"]
        if not isinstance(expected, str) or not _SHA256_RE.fullmatch(expected):
            raise RagSftAuthoringCorrectionError(f"{prefix} expected_record_sha256 无效")
        if _record_digest(parent_by_id[record_id]) != expected:
            raise RagSftAuthoringCorrectionError(f"{record_id} 父记录身份已变化")
        replacement = correction["replacement"]
        if not isinstance(replacement, dict) or replacement.get("id") != record_id:
            raise RagSftAuthoringCorrectionError(f"{prefix} replacement id 无效")
        if replacement == parent_by_id[record_id]:
            raise RagSftAuthoringCorrectionError(f"{record_id} replacement 没有实际变化")
        correction_by_id[record_id] = correction

    revised_records = [
        correction_by_id.get(record["id"], {}).get("replacement", record)
        for record in parent_records
    ]
    try:
        repository = ArticleRepository.from_jsonl(article_index_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RagSftAuthoringCorrectionError("无法加载 article index") from error
    signatures = set()
    source_counts: Counter[str] = Counter()
    behavior_counts: Counter[str] = Counter()
    gt_counts: Counter[str] = Counter()
    for position, record in enumerate(revised_records, start=1):
        try:
            preparation._validate_record_shape(record, position)
            preparation._package_and_assistant(record, repository)
        except preparation.RagSftPreparationError as error:
            raise RagSftAuthoringCorrectionError(
                f"修订后 canonical 无效: {error}"
            ) from error
        if record["review_status"] != "approved":
            raise RagSftAuthoringCorrectionError(
                f"修订后 canonical 包含未批准记录: {record['id']}"
            )
        signature = materializer._signature(record)
        if signature in signatures:
            raise RagSftAuthoringCorrectionError("修订后存在重复模型可见 conversations")
        signatures.add(signature)
        source_counts[record["evidence_source"]] += 1
        behavior_counts["refusal" if record["target"]["refuse"] else "answer"] += 1
        gt_counts[str(len(record["required_chunk_ids"]))] += 1

    authoring_payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in revised_records
    )
    mapping = []
    for record_id, correction in correction_by_id.items():
        before = parent_by_id[record_id]
        after = correction["replacement"]
        mapping.append(
            {
                "id": record_id,
                "before_sha256": _record_digest(before),
                "after_sha256": _record_digest(after),
                "changed_fields": sorted(
                    key for key in before if before[key] != after[key]
                ),
                "reason": correction["reason"],
            }
        )
    mapping_payload = "".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for item in mapping
    )
    manifest: dict[str, object] = {
        "pipeline": "rag_sft_canonical_authoring_correction",
        "release_status": "provisional_canonical",
        "input": {
            "parent_manifest": {
                **_identity(parent_manifest_path),
                "hash_manifest": _identity(parent_hash_path),
            },
            "parent_authoring": _identity(
                parent_authoring_path, records=len(parent_records)
            ),
            "corrections": {
                **_identity(corrections_path, records=len(corrections)),
                "hash_manifest": _identity(corrections_hash_path),
            },
            "article_index": _identity(article_index_path),
        },
        "records": {
            "total": len(revised_records),
            "corrected": len(corrections),
            "unchanged": len(revised_records) - len(corrections),
            "by_evidence_source": dict(sorted(source_counts.items())),
            "by_behavior": dict(sorted(behavior_counts.items())),
            "by_required_gt_count": dict(sorted(gt_counts.items())),
        },
        "validation": {
            "parent_record_hashes_matched": True,
            "ids_and_order_preserved": [item["id"] for item in revised_records]
            == [item["id"] for item in parent_records],
            "all_records_approved": True,
            "article_and_support_spans_validated": True,
            "model_visible_signatures_unique": True,
            "old_assets_modified": False,
        },
        "output": {
            "authoring": {
                "path": str(authoring_output),
                "records": len(revised_records),
                "bytes": len(authoring_payload.encode("utf-8")),
                "sha256": _sha256_bytes(authoring_payload.encode("utf-8")),
            },
            "correction_mapping": {
                "path": str(mapping_output),
                "records": len(mapping),
                "bytes": len(mapping_payload.encode("utf-8")),
                "sha256": _sha256_bytes(mapping_payload.encode("utf-8")),
            },
        },
        "readiness": {
            "canonical_authoring_schema_validated": True,
            "retrieved_records_requiring_exclusion": sorted(correction_by_id),
            "evaluation_isolation_complete": False,
            "training_ready": False,
        },
        "complete": True,
    }
    manifest_payload = json.dumps(
        manifest, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    hash_payload = f"{_sha256_bytes(manifest_payload.encode('utf-8'))}  {manifest_output.name}\n"
    _publish(
        [
            (authoring_output, authoring_payload),
            (mapping_output, mapping_payload),
            (manifest_output, manifest_payload),
            (hash_output, hash_payload),
        ]
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, default=DEFAULT_PARENT_MANIFEST)
    parser.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS)
    parser.add_argument("--article-index", type=Path, default=DEFAULT_ARTICLE_INDEX)
    parser.add_argument("--authoring-output", type=Path, default=DEFAULT_AUTHORING_OUTPUT)
    parser.add_argument("--mapping-output", type=Path, default=DEFAULT_MAPPING_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    args = parser.parse_args()
    try:
        manifest = apply_rag_sft_authoring_corrections(
            parent_manifest_path=args.parent_manifest,
            corrections_path=args.corrections,
            article_index_path=args.article_index,
            authoring_output=args.authoring_output,
            mapping_output=args.mapping_output,
            manifest_output=args.manifest_output,
        )
    except RagSftAuthoringCorrectionError as error:
        parser.error(str(error))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "RagSftAuthoringCorrectionError",
    "apply_rag_sft_authoring_corrections",
]
