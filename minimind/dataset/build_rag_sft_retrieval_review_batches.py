"""把冻结的 retrieved 语义审核队列切为可独立双审的不可覆盖批次。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from . import build_rag_sft_retrieval_review_queue as queue_builder
    from . import validate_rag_sft_retrieval_pilot_reviews as review_validator
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import build_rag_sft_retrieval_review_queue as queue_builder
    from dataset import validate_rag_sft_retrieval_pilot_reviews as review_validator


RAG_SFT_ROOT = Path(__file__).resolve().parent / "RAG-SFT"
DEFAULT_REVIEW_DIR = (
    RAG_SFT_ROOT / "retrieved" / "review-canonical-v1-20260809"
)
DEFAULT_OUTPUT_DIR = (
    RAG_SFT_ROOT / "retrieved" / "review-batches-canonical-v1-20260809"
)
BATCH_SIZE = 50
MANIFEST_FILENAME = "rag-sft-retrieval-review-batches.json"
HASH_FILENAME = "rag-sft-retrieval-review-batches.sha256"


class RagSftRetrievalReviewBatchError(RuntimeError):
    """冻结审核队列无法无损切分。"""


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


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagSftRetrievalReviewBatchError(
                        f"{description}不允许空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RagSftRetrievalReviewBatchError(
                        f"{description}第 {line_number} 条必须是 object"
                    )
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftRetrievalReviewBatchError):
            raise
        raise RagSftRetrievalReviewBatchError(
            f"无法读取{description}: {path}"
        ) from error
    return records


def _unique_by_source(
    records: list[dict[str, Any]], description: str
) -> dict[str, dict[str, Any]]:
    result = {}
    for position, record in enumerate(records, start=1):
        source_id = record.get("source_id")
        if not isinstance(source_id, str) or not source_id or source_id in result:
            raise RagSftRetrievalReviewBatchError(
                f"{description}第 {position} 条 source_id 无效或重复"
            )
        result[source_id] = record
    return result


def _jsonl_payload(records: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in records
    )


def _publish(
    output_dir: Path,
    batches: list[list[dict[str, Any]]],
    manifest: dict[str, object],
) -> None:
    if output_dir.exists():
        raise RagSftRetrievalReviewBatchError(f"批次输出目录必须是新目录: {output_dir}")
    output_dir.mkdir(parents=True)
    payloads = [
        (f"review-batch-{index:02d}.jsonl", _jsonl_payload(batch))
        for index, batch in enumerate(batches, start=1)
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
                f"{_sha256(partial)}  {final.name}\n"
                for partial, final in partials
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
        raise RagSftRetrievalReviewBatchError("无法发布 retrieved 审核批次") from error


def build_retrieval_review_batches(
    *,
    review_dir: Path,
    output_dir: Path,
    batch_size: int = BATCH_SIZE,
) -> dict[str, object]:
    """复用已审 pilot，并把其余完整队列按原顺序切成固定批次。"""

    if type(batch_size) is not int or batch_size <= 0:
        raise RagSftRetrievalReviewBatchError("batch_size 必须是正整数")
    review_dir = Path(review_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RagSftRetrievalReviewBatchError(f"批次输出目录必须是新目录: {output_dir}")
    queue_path = (review_dir / queue_builder.QUEUE_FILENAME).resolve()
    pilot_path = (review_dir / queue_builder.PILOT_FILENAME).resolve()
    try:
        review_manifest_path, review_manifest = review_validator._verify_review_bundle(
            queue_path,
            expected_filename=queue_builder.QUEUE_FILENAME,
        )
    except review_validator.RagSftRetrievalPilotReviewError as error:
        raise RagSftRetrievalReviewBatchError(str(error)) from error
    queue = _load_jsonl(queue_path, "语义审核全量队列")
    pilot = _load_jsonl(pilot_path, "已审 pilot")
    queue_by_id = _unique_by_source(queue, "语义审核全量队列")
    pilot_by_id = _unique_by_source(pilot, "已审 pilot")
    records = review_manifest.get("records", {})
    if (
        records.get("semantic_review_queue") != len(queue)
        or records.get("pilot") != len(pilot)
    ):
        raise RagSftRetrievalReviewBatchError("队列或 pilot 数量与 review manifest 不一致")
    if not set(pilot_by_id).issubset(queue_by_id):
        raise RagSftRetrievalReviewBatchError("pilot 不是全量语义队列的子集")
    for source_id, pilot_item in pilot_by_id.items():
        if queue_builder.canonical_digest(pilot_item) != queue_builder.canonical_digest(
            queue_by_id[source_id]
        ):
            raise RagSftRetrievalReviewBatchError(
                f"pilot 与全量队列内容不一致: {source_id}"
            )

    remaining = [item for item in queue if item["source_id"] not in pilot_by_id]
    batches = [
        remaining[offset : offset + batch_size]
        for offset in range(0, len(remaining), batch_size)
    ]
    flattened_ids = [item["source_id"] for batch in batches for item in batch]
    if (
        len(pilot) + len(flattened_ids) != len(queue)
        or len(flattened_ids) != len(set(flattened_ids))
        or set(flattened_ids) & set(pilot_by_id)
        or set(flattened_ids) | set(pilot_by_id) != set(queue_by_id)
    ):
        raise RagSftRetrievalReviewBatchError("pilot 与新增批次未闭合全量语义队列")
    lane_counts = Counter(item["lane"] for item in remaining)
    batch_metadata = []
    for index, batch in enumerate(batches, start=1):
        batch_metadata.append(
            {
                "batch": index,
                "filename": f"review-batch-{index:02d}.jsonl",
                "records": len(batch),
                "first_source_id": batch[0]["source_id"],
                "last_source_id": batch[-1]["source_id"],
                "by_lane": dict(
                    sorted(Counter(item["lane"] for item in batch).items())
                ),
            }
        )
    manifest: dict[str, object] = {
        "pipeline": "rag_sft_retrieval_review_batching",
        "policy": {
            "pilot_reused_without_rereview": True,
            "full_queue_order_preserved": True,
            "batch_size": batch_size,
            "training_authoring_emitted": False,
        },
        "input": {
            "review_manifest": _identity(review_manifest_path),
            "review_queue": _identity(queue_path, records=len(queue)),
            "review_pilot": _identity(pilot_path, records=len(pilot)),
        },
        "records": {
            "semantic_review_queue": len(queue),
            "reused_pilot": len(pilot),
            "remaining": len(remaining),
            "batches": len(batches),
            "remaining_by_lane": dict(sorted(lane_counts.items())),
        },
        "batches": batch_metadata,
        "validation": {
            "pilot_is_exact_queue_subset": True,
            "remaining_ids_unique": True,
            "pilot_and_batches_close_full_queue": True,
        },
        "complete": True,
    }
    _publish(output_dir, batches, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    try:
        manifest = build_retrieval_review_batches(
            review_dir=args.review_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
    except RagSftRetrievalReviewBatchError as error:
        parser.error(str(error))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "RagSftRetrievalReviewBatchError",
    "build_retrieval_review_batches",
]
