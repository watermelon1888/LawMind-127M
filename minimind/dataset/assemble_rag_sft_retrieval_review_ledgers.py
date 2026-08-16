"""汇编冻结 retrieved 审核队列的全量双角色 ledger，不产生裁决或训练数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

try:
    from . import build_rag_sft_retrieval_review_batches as batch_builder
    from . import build_rag_sft_retrieval_review_queue as queue_builder
    from . import validate_rag_sft_retrieval_pilot_reviews as review_validator
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import build_rag_sft_retrieval_review_batches as batch_builder
    from dataset import build_rag_sft_retrieval_review_queue as queue_builder
    from dataset import validate_rag_sft_retrieval_pilot_reviews as review_validator


RAG_SFT_ROOT = Path(__file__).resolve().parent / "RAG-SFT"
DEFAULT_REVIEW_DIR = RAG_SFT_ROOT / "retrieved" / "review-canonical-v1-20260809"
DEFAULT_BATCH_DIR = RAG_SFT_ROOT / "retrieved" / "review-batches-canonical-v1-20260809"
DEFAULT_PILOT_LEDGER_DIR = (
    RAG_SFT_ROOT / "retrieved" / "review-ledgers-canonical-v1-20260809"
)

ROLE_TO_PILOT_FILENAME = {
    "legal_support": "legal-support.jsonl",
    "adversarial_boundary": "adversarial-boundary.jsonl",
}
ROLE_TO_BATCH_PREFIX = {
    "legal_support": "legal-support",
    "adversarial_boundary": "adversarial-boundary",
}
ROLE_TO_OUTPUT_FILENAME = ROLE_TO_PILOT_FILENAME
MANIFEST_FILENAME = "rag-sft-retrieval-review-ledger-assembly.json"
HASH_FILENAME = "rag-sft-retrieval-review-ledger-assembly.sha256"
_SHA256_RE = re.compile(r"[0-9a-f]{64}$")


class RagSftRetrievalReviewLedgerAssemblyError(RuntimeError):
    """全量 retrieved 双审 ledger 无法安全闭合或发布。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if records is not None:
        result["records"] = records
    return result


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftRetrievalReviewLedgerAssemblyError(
            f"无法读取{description}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise RagSftRetrievalReviewLedgerAssemblyError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagSftRetrievalReviewLedgerAssemblyError(
                        f"{description}不允许空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RagSftRetrievalReviewLedgerAssemblyError(
                        f"{description}第 {line_number} 条必须是 object"
                    )
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftRetrievalReviewLedgerAssemblyError):
            raise
        raise RagSftRetrievalReviewLedgerAssemblyError(
            f"无法读取{description}: {path}"
        ) from error
    if not records:
        raise RagSftRetrievalReviewLedgerAssemblyError(f"{description}不能为空")
    return records


def _parse_hash_manifest(path: Path, description: str) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftRetrievalReviewLedgerAssemblyError(
            f"无法读取{description} SHA-256 清单: {path}"
        ) from error
    result = {}
    for line_number, line in enumerate(lines, start=1):
        parts = line.split("  ", 1)
        if (
            len(parts) != 2
            or not _SHA256_RE.fullmatch(parts[0])
            or not parts[1]
            or Path(parts[1]).name != parts[1]
            or parts[1] in result
        ):
            raise RagSftRetrievalReviewLedgerAssemblyError(
                f"{description} SHA-256 清单格式无效: {line_number}"
            )
        result[parts[1]] = parts[0]
    return result


def _verify_hash_manifest(
    directory: Path,
    *,
    hash_filename: str,
    expected_paths: dict[str, Path],
    description: str,
) -> Path:
    hash_path = directory / hash_filename
    parsed = _parse_hash_manifest(hash_path, description)
    if set(parsed) != set(expected_paths):
        raise RagSftRetrievalReviewLedgerAssemblyError(
            f"{description} SHA-256 文件集合无效"
        )
    for name, path in expected_paths.items():
        if not path.is_file() or _sha256(path) != parsed[name]:
            raise RagSftRetrievalReviewLedgerAssemblyError(
                f"{description}文件身份已变化: {name}"
            )
    return hash_path


def _verify_identity(metadata: object, path: Path, description: str) -> None:
    if not isinstance(metadata, dict):
        raise RagSftRetrievalReviewLedgerAssemblyError(f"{description}身份信息无效")
    if (
        metadata.get("path") != str(path.resolve())
        or metadata.get("bytes") != path.stat().st_size
        or metadata.get("sha256") != _sha256(path)
    ):
        raise RagSftRetrievalReviewLedgerAssemblyError(f"{description}身份不一致")


def _review_error(error: Exception) -> RagSftRetrievalReviewLedgerAssemblyError:
    return RagSftRetrievalReviewLedgerAssemblyError(str(error))


def _validate_queue_bundle(
    review_dir: Path,
) -> tuple[Path, dict[str, Any], Path, list[dict[str, Any]], Path, list[dict[str, Any]]]:
    review_dir = review_dir.resolve()
    pilot_path = review_dir / queue_builder.PILOT_FILENAME
    queue_path = review_dir / queue_builder.QUEUE_FILENAME
    try:
        manifest_path, manifest = review_validator._verify_review_bundle(pilot_path)
    except review_validator.RagSftRetrievalPilotReviewError as error:
        raise _review_error(error) from error
    queue = _load_jsonl(queue_path, "全量 review queue")
    pilot = _load_jsonl(pilot_path, "review pilot")
    try:
        queue_by_id = review_validator._validate_pilot(queue)
        pilot_by_id = review_validator._validate_pilot(pilot)
    except review_validator.RagSftRetrievalPilotReviewError as error:
        raise _review_error(error) from error
    records = manifest.get("records", {})
    if (
        records.get("semantic_review_queue") != len(queue)
        or records.get("pilot") != len(pilot)
    ):
        raise RagSftRetrievalReviewLedgerAssemblyError(
            "review queue 或 pilot 数量与 manifest 不一致"
        )
    if not set(pilot_by_id).issubset(queue_by_id):
        raise RagSftRetrievalReviewLedgerAssemblyError("pilot 不是全量 review queue 子集")
    for source_id, item in pilot_by_id.items():
        if queue_builder.canonical_digest(item) != queue_builder.canonical_digest(
            queue_by_id[source_id]
        ):
            raise RagSftRetrievalReviewLedgerAssemblyError(
                f"pilot 与全量 review queue 内容不一致: {source_id}"
            )
    return manifest_path, manifest, queue_path, queue, pilot_path, pilot


def _verify_batch_bundle(
    batch_dir: Path,
    *,
    review_manifest_path: Path,
    review_manifest: dict[str, Any],
    queue_path: Path,
    queue: list[dict[str, Any]],
    pilot_path: Path,
    pilot: list[dict[str, Any]],
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    batch_dir = batch_dir.resolve()
    manifest_path = batch_dir / batch_builder.MANIFEST_FILENAME
    manifest = _load_json(manifest_path, "batch bundle manifest")
    metadata = manifest.get("batches")
    if not isinstance(metadata, list) or not metadata:
        raise RagSftRetrievalReviewLedgerAssemblyError("batch bundle batches 无效")
    batch_paths = {}
    for position, entry in enumerate(metadata, start=1):
        if not isinstance(entry, dict):
            raise RagSftRetrievalReviewLedgerAssemblyError("batch bundle batch 元数据无效")
        index = entry.get("batch")
        filename = entry.get("filename")
        if (
            type(index) is not int
            or index != position
            or filename != f"review-batch-{index:02d}.jsonl"
            or type(entry.get("records")) is not int
            or entry["records"] <= 0
        ):
            raise RagSftRetrievalReviewLedgerAssemblyError("batch bundle batch 元数据不连续")
        batch_paths[filename] = batch_dir / filename
    _verify_hash_manifest(
        batch_dir,
        hash_filename=batch_builder.HASH_FILENAME,
        expected_paths={**batch_paths, batch_builder.MANIFEST_FILENAME: manifest_path},
        description="batch bundle",
    )
    if (
        manifest.get("pipeline") != "rag_sft_retrieval_review_batching"
        or manifest.get("complete") is not True
    ):
        raise RagSftRetrievalReviewLedgerAssemblyError("batch bundle manifest 状态无效")
    inputs = manifest.get("input", {})
    _verify_identity(inputs.get("review_manifest"), review_manifest_path, "batch bundle review manifest")
    _verify_identity(inputs.get("review_queue"), queue_path, "batch bundle review queue")
    _verify_identity(inputs.get("review_pilot"), pilot_path, "batch bundle review pilot")
    records = manifest.get("records", {})
    if (
        records.get("semantic_review_queue") != len(queue)
        or records.get("reused_pilot") != len(pilot)
        or records.get("batches") != len(metadata)
    ):
        raise RagSftRetrievalReviewLedgerAssemblyError("batch bundle 记录数与输入不一致")

    batches = []
    for entry in metadata:
        batch = _load_jsonl(batch_dir / entry["filename"], entry["filename"])
        if len(batch) != entry["records"]:
            raise RagSftRetrievalReviewLedgerAssemblyError(
                f"batch bundle 条数不一致: {entry['filename']}"
            )
        try:
            review_validator._validate_pilot(batch)
        except review_validator.RagSftRetrievalPilotReviewError as error:
            raise _review_error(error) from error
        batches.append(batch)
    remaining = [
        item for item in queue if item["source_id"] not in {entry["source_id"] for entry in pilot}
    ]
    flattened = [item for batch in batches for item in batch]
    if (
        records.get("remaining") != len(remaining)
        or len(flattened) != len(remaining)
        or [item["source_id"] for item in flattened]
        != [item["source_id"] for item in remaining]
    ):
        raise RagSftRetrievalReviewLedgerAssemblyError(
            "batch bundle 未按原队列顺序完整覆盖非 pilot 记录"
        )
    for expected, actual in zip(remaining, flattened):
        if queue_builder.canonical_digest(expected) != queue_builder.canonical_digest(actual):
            raise RagSftRetrievalReviewLedgerAssemblyError(
                f"batch bundle 与 review queue 内容不一致: {actual['source_id']}"
            )
    if review_manifest.get("records", {}).get("semantic_review_queue") != len(queue):
        raise RagSftRetrievalReviewLedgerAssemblyError("review manifest 队列数量不一致")
    return manifest_path, manifest, batches, flattened


def _validate_ledger(
    path: Path,
    *,
    role: str,
    expected_items: list[dict[str, Any]],
    description: str,
) -> dict[str, dict[str, Any]]:
    records = _load_jsonl(path, description)
    try:
        expected_by_id = review_validator._validate_pilot(expected_items)
    except review_validator.RagSftRetrievalPilotReviewError as error:
        raise _review_error(error) from error
    by_id = {}
    for position, record in enumerate(records, start=1):
        try:
            review_validator._validate_review(
                record,
                position=position,
                expected_role=role,
                pilot_by_id=expected_by_id,
            )
        except review_validator.RagSftRetrievalPilotReviewError as error:
            raise _review_error(error) from error
        source_id = record["source_id"]
        if source_id in by_id:
            raise RagSftRetrievalReviewLedgerAssemblyError(
                f"{description} source_id 重复: {source_id}"
            )
        by_id[source_id] = record
    if set(by_id) != set(expected_by_id):
        raise RagSftRetrievalReviewLedgerAssemblyError(
            f"{description}必须恰好覆盖对应 batch 的全部 source_id"
        )
    return by_id


def _jsonl_payload(records: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in records
    )


def _publish(
    output_dir: Path,
    ledgers: dict[str, list[dict[str, Any]]],
    manifest: dict[str, object],
) -> None:
    if output_dir.exists():
        raise RagSftRetrievalReviewLedgerAssemblyError("ledger 输出目录必须是新目录")
    output_dir.mkdir(parents=True)
    payloads = [
        (ROLE_TO_OUTPUT_FILENAME[role], _jsonl_payload(ledgers[role]))
        for role in ROLE_TO_OUTPUT_FILENAME
    ]
    payloads.append(
        (
            MANIFEST_FILENAME,
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )
    )
    partials = []
    published = []
    try:
        for filename, payload in payloads:
            partial = output_dir / f"{filename}.partial"
            final = output_dir / filename
            partial.write_text(payload, encoding="utf-8", newline="\n")
            partials.append((partial, final))
        hash_partial = output_dir / f"{HASH_FILENAME}.partial"
        hash_partial.write_text(
            "".join(
                f"{_sha256(partial)}  {final.name}\n" for partial, final in partials
            ),
            encoding="utf-8",
            newline="\n",
        )
        for partial, final in partials:
            partial.replace(final)
            published.append(final)
        hash_partial.replace(output_dir / HASH_FILENAME)
    except (OSError, UnicodeError, ValueError) as error:
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
        raise RagSftRetrievalReviewLedgerAssemblyError("无法发布全量双审 ledger") from error


def assemble_retrieval_review_ledgers(
    *,
    review_dir: Path,
    batch_dir: Path,
    pilot_ledger_dir: Path,
    batch_ledger_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """校验 pilot 与批次双审记录后，按全量队列顺序汇编两个角色的 ledger。"""

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RagSftRetrievalReviewLedgerAssemblyError("ledger 输出目录必须是新目录")
    (
        review_manifest_path,
        review_manifest,
        queue_path,
        queue,
        pilot_path,
        pilot,
    ) = _validate_queue_bundle(Path(review_dir))
    batch_manifest_path, batch_manifest, batches, flattened = _verify_batch_bundle(
        Path(batch_dir),
        review_manifest_path=review_manifest_path,
        review_manifest=review_manifest,
        queue_path=queue_path,
        queue=queue,
        pilot_path=pilot_path,
        pilot=pilot,
    )
    pilot_ledger_dir = Path(pilot_ledger_dir).resolve()
    batch_ledger_dir = Path(batch_ledger_dir).resolve()
    reviews_by_role: dict[str, dict[str, dict[str, Any]]] = {}
    input_ledgers: dict[str, object] = {"pilot": {}, "batches": {}}
    expected_batch_filenames = set()
    for role, pilot_filename in ROLE_TO_PILOT_FILENAME.items():
        pilot_ledger_path = pilot_ledger_dir / pilot_filename
        by_id = _validate_ledger(
            pilot_ledger_path,
            role=role,
            expected_items=pilot,
            description=f"{role} pilot ledger",
        )
        input_ledgers["pilot"][role] = _identity(pilot_ledger_path, records=len(by_id))
        for index, batch in enumerate(batches, start=1):
            filename = f"{ROLE_TO_BATCH_PREFIX[role]}-batch-{index:02d}.jsonl"
            expected_batch_filenames.add(filename)
            path = batch_ledger_dir / filename
            batch_by_id = _validate_ledger(
                path,
                role=role,
                expected_items=batch,
                description=f"{role} batch {index:02d} ledger",
            )
            overlap = set(by_id) & set(batch_by_id)
            if overlap:
                raise RagSftRetrievalReviewLedgerAssemblyError(
                    f"{role} pilot 与 batch ledger source_id 重复: {sorted(overlap)[0]}"
                )
            by_id.update(batch_by_id)
            input_ledgers["batches"][filename] = _identity(
                path, records=len(batch_by_id)
            )
        if set(by_id) != {item["source_id"] for item in queue}:
            raise RagSftRetrievalReviewLedgerAssemblyError(
                f"{role} pilot 与 batch ledger 未完整闭合全量 review queue"
            )
        reviews_by_role[role] = by_id
    actual_batch_filenames = {path.name for path in batch_ledger_dir.glob("*.jsonl")}
    if actual_batch_filenames != expected_batch_filenames:
        raise RagSftRetrievalReviewLedgerAssemblyError(
            "批次 ledger 目录 JSONL 文件集合与 batch bundle 不一致"
        )

    ledgers = {
        role: [reviews_by_role[role][item["source_id"]] for item in queue]
        for role in ROLE_TO_PILOT_FILENAME
    }
    if any(len(ledger) != len(queue) for ledger in ledgers.values()):
        raise RagSftRetrievalReviewLedgerAssemblyError("全量 ledger 条数未闭合")
    manifest: dict[str, object] = {
        "pipeline": "rag_sft_retrieval_review_ledger_assembly",
        "policy": {
            "review_fields_validated_by_existing_validator": True,
            "full_queue_order_preserved": True,
            "consensus_emitted": False,
            "adjudication_emitted": False,
            "training_authoring_emitted": False,
        },
        "input": {
            "review_bundle_manifest": _identity(review_manifest_path),
            "batch_bundle_manifest": _identity(batch_manifest_path),
            "review_queue": _identity(queue_path, records=len(queue)),
            "review_pilot": _identity(pilot_path, records=len(pilot)),
            "ledgers": input_ledgers,
        },
        "records": {
            "review_queue": len(queue),
            "pilot": len(pilot),
            "batch_records": len(flattened),
            "batches": len(batches),
            "full_ledger_by_role": {
                role: len(records) for role, records in ledgers.items()
            },
        },
        "validation": {
            "review_bundle_hashes_verified": True,
            "batch_bundle_hashes_verified": True,
            "pilot_and_batches_close_full_queue": True,
            "each_role_covers_each_source_once": True,
            "output_order_matches_review_queue": True,
        },
        "outputs": {
            "legal_support": ROLE_TO_OUTPUT_FILENAME["legal_support"],
            "adversarial_boundary": ROLE_TO_OUTPUT_FILENAME[
                "adversarial_boundary"
            ],
            "manifest": MANIFEST_FILENAME,
            "sha256_manifest": HASH_FILENAME,
        },
        "complete": True,
    }
    _publish(output_dir, ledgers, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument(
        "--pilot-ledger-dir", type=Path, default=DEFAULT_PILOT_LEDGER_DIR
    )
    parser.add_argument("--batch-ledger-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = assemble_retrieval_review_ledgers(
            review_dir=args.review_dir,
            batch_dir=args.batch_dir,
            pilot_ledger_dir=args.pilot_ledger_dir,
            batch_ledger_dir=args.batch_ledger_dir,
            output_dir=args.output_dir,
        )
    except RagSftRetrievalReviewLedgerAssemblyError as error:
        parser.error(str(error))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "RagSftRetrievalReviewLedgerAssemblyError",
    "assemble_retrieval_review_ledgers",
]
