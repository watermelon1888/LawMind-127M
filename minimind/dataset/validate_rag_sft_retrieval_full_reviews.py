"""验证全量 retrieved 双审 ledger 并发布保守共识，不生成训练 authoring。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from . import assemble_rag_sft_retrieval_review_ledgers as assembler
    from . import build_rag_sft_retrieval_review_queue as queue_builder
    from . import validate_rag_sft_retrieval_pilot_reviews as review_validator
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import assemble_rag_sft_retrieval_review_ledgers as assembler
    from dataset import build_rag_sft_retrieval_review_queue as queue_builder
    from dataset import validate_rag_sft_retrieval_pilot_reviews as review_validator


RAG_SFT_ROOT = Path(__file__).resolve().parent / "RAG-SFT"
DEFAULT_REVIEW_DIR = RAG_SFT_ROOT / "retrieved" / "review-canonical-v1-20260809"
_SHA256_RE = re.compile(r"[0-9a-f]{64}$")

CONSENSUS_FILENAME = "rag-sft-retrieval-full-consensus.jsonl"
DECISIONS_FILENAME = "rag-sft-retrieval-full-decisions.jsonl"
USER_REVIEW_FILENAME = "rag-sft-retrieval-full-user-review.jsonl"
REPORT_FILENAME = "rag-sft-retrieval-full-review-report.json"
HASH_FILENAME = "rag-sft-retrieval-full-review.sha256"
ROLE_TO_LEDGER_FILENAME = {
    "legal_support": "legal-support.jsonl",
    "adversarial_boundary": "adversarial-boundary.jsonl",
}


class RagSftRetrievalFullReviewError(RuntimeError):
    """全量 retrieved 双审 ledger 无法验证、闭合或安全发布。"""


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
        raise RagSftRetrievalFullReviewError(
            f"无法读取{description}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise RagSftRetrievalFullReviewError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagSftRetrievalFullReviewError(
                        f"{description}不允许空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RagSftRetrievalFullReviewError(
                        f"{description}第 {line_number} 条必须是 object"
                    )
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftRetrievalFullReviewError):
            raise
        raise RagSftRetrievalFullReviewError(
            f"无法读取{description}: {path}"
        ) from error
    if not records:
        raise RagSftRetrievalFullReviewError(f"{description}不能为空")
    return records


def _verify_identity(metadata: object, path: Path, description: str) -> None:
    if not isinstance(metadata, dict):
        raise RagSftRetrievalFullReviewError(f"{description}身份信息无效")
    if (
        metadata.get("path") != str(path.resolve())
        or metadata.get("bytes") != path.stat().st_size
        or metadata.get("sha256") != _sha256(path)
    ):
        raise RagSftRetrievalFullReviewError(f"{description}身份不一致")


def _verify_assembly_bundle(
    ledger_dir: Path,
    *,
    review_manifest_path: Path,
    queue_path: Path,
    queue: list[dict[str, Any]],
) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    ledger_dir = ledger_dir.resolve()
    manifest_path = ledger_dir / assembler.MANIFEST_FILENAME
    manifest = _load_json(manifest_path, "ledger assembly manifest")
    ledger_paths = {
        role: ledger_dir / filename
        for role, filename in ROLE_TO_LEDGER_FILENAME.items()
    }
    expected_paths = {
        **{path.name: path for path in ledger_paths.values()},
        assembler.MANIFEST_FILENAME: manifest_path,
    }
    hash_path = ledger_dir / assembler.HASH_FILENAME
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftRetrievalFullReviewError(
            "无法读取 ledger assembly SHA-256 清单"
        ) from error
    hashes = {}
    for line_number, line in enumerate(lines, start=1):
        parts = line.split("  ", 1)
        if (
            len(parts) != 2
            or not _SHA256_RE.fullmatch(parts[0])
            or not parts[1]
            or Path(parts[1]).name != parts[1]
            or parts[1] in hashes
        ):
            raise RagSftRetrievalFullReviewError(
                f"ledger assembly SHA-256 清单格式无效: {line_number}"
            )
        hashes[parts[1]] = parts[0]
    if set(hashes) != set(expected_paths):
        raise RagSftRetrievalFullReviewError("ledger assembly SHA-256 文件集合无效")
    for name, path in expected_paths.items():
        if not path.is_file() or _sha256(path) != hashes[name]:
            raise RagSftRetrievalFullReviewError(
                f"ledger assembly 文件身份已变化: {name}"
            )
    if (
        manifest.get("pipeline") != "rag_sft_retrieval_review_ledger_assembly"
        or manifest.get("complete") is not True
    ):
        raise RagSftRetrievalFullReviewError("ledger assembly manifest 状态无效")
    inputs = manifest.get("input", {})
    _verify_identity(
        inputs.get("review_bundle_manifest"),
        review_manifest_path,
        "ledger assembly review manifest",
    )
    _verify_identity(
        inputs.get("review_queue"), queue_path, "ledger assembly review queue"
    )
    records = manifest.get("records", {})
    if (
        records.get("review_queue") != len(queue)
        or records.get("full_ledger_by_role")
        != {role: len(queue) for role in ROLE_TO_LEDGER_FILENAME}
    ):
        raise RagSftRetrievalFullReviewError("ledger assembly 记录数不闭合")
    outputs = manifest.get("outputs", {})
    if any(outputs.get(role) != path.name for role, path in ledger_paths.items()):
        raise RagSftRetrievalFullReviewError("ledger assembly 输出文件契约无效")
    return manifest_path, manifest, ledger_paths


def _validate_ledger(
    path: Path,
    *,
    role: str,
    queue: list[dict[str, Any]],
    queue_by_id: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    records = _load_jsonl(path, f"{role} full ledger")
    if len(records) != len(queue):
        raise RagSftRetrievalFullReviewError(f"{role} full ledger 条数不一致")
    by_id = {}
    for position, record in enumerate(records, start=1):
        try:
            review_validator._validate_review(
                record,
                position=position,
                expected_role=role,
                pilot_by_id=queue_by_id,
            )
        except review_validator.RagSftRetrievalPilotReviewError as error:
            raise RagSftRetrievalFullReviewError(str(error)) from error
        source_id = record["source_id"]
        if source_id in by_id:
            raise RagSftRetrievalFullReviewError(
                f"{role} full ledger source_id 重复: {source_id}"
            )
        by_id[source_id] = record
    expected_ids = [item["source_id"] for item in queue]
    if [record["source_id"] for record in records] != expected_ids:
        raise RagSftRetrievalFullReviewError(f"{role} full ledger 顺序与 review queue 不一致")
    if set(by_id) != set(queue_by_id):
        raise RagSftRetrievalFullReviewError(f"{role} full ledger 未闭合 review queue")
    return by_id


def _user_review_record(
    item: dict[str, Any],
    legal: dict[str, Any],
    adversarial: dict[str, Any],
    reason_codes: list[str],
) -> dict[str, object]:
    return {
        "source_id": item["source_id"],
        "derived_id": item["derived_id"],
        "locator_sha256": item["locator_sha256"],
        "lane": item["lane"],
        "reason_codes": reason_codes,
        "legal_support_review": legal,
        "adversarial_boundary_review": adversarial,
    }


def _exact_or_union_consensus(
    item: dict[str, Any],
    legal: dict[str, Any],
    adversarial: dict[str, Any],
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    legal_facts = review_validator._review_facts(legal)
    adversarial_facts = review_validator._review_facts(adversarial)
    reason_codes = []
    if legal_facts["target_conflict"] or adversarial_facts["target_conflict"]:
        reason_codes.append("target_conflict")
    if (
        legal_facts["alternative_complete_answer_chunk_ids"]
        or adversarial_facts["alternative_complete_answer_chunk_ids"]
    ):
        reason_codes.append("alternative_complete_answer")
    if reason_codes:
        if legal_facts != adversarial_facts:
            reason_codes.append("review_disagreement")
        return None, _user_review_record(item, legal, adversarial, reason_codes)
    if legal_facts == adversarial_facts:
        decision = review_validator._consensus_decision(item, legal_facts)
        if decision is None:
            raise RagSftRetrievalFullReviewError("完全一致事实无法派生共识")
        decision["consensus_mode"] = "exact_fact_consensus"
        return decision, None

    differing_fields = {
        field
        for field in legal_facts
        if legal_facts[field] != adversarial_facts[field]
    }
    partition_fields = {
        "supporting_extra_chunk_ids",
        "non_supporting_extra_chunk_ids",
    }
    if not differing_fields.issubset(partition_fields):
        return None, _user_review_record(
            item, legal, adversarial, ["review_disagreement"]
        )
    extras = list(item["extra_non_gt_chunk_ids"])
    support_union = set(legal_facts["supporting_extra_chunk_ids"]) | set(
        adversarial_facts["supporting_extra_chunk_ids"]
    )
    facts = {
        "supporting_extra_chunk_ids": [
            chunk_id for chunk_id in extras if chunk_id in support_union
        ],
        "non_supporting_extra_chunk_ids": [
            chunk_id for chunk_id in extras if chunk_id not in support_union
        ],
        "alternative_complete_answer_chunk_ids": [],
        "target_conflict": False,
    }
    decision = review_validator._consensus_decision(item, facts)
    if decision is None:
        raise RagSftRetrievalFullReviewError("支持性并集无法派生保守共识")
    decision["decision"]["reason"] = (
        "两角色仅对 extra 的支持性分组不同；按 supporting 并集采取保守排除。"
    )
    decision["consensus_mode"] = "conservative_support_union"
    return decision, None


def _write_jsonl(records: list[dict[str, object]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in records
    )


def _publish(
    output_dir: Path,
    *,
    consensus: list[dict[str, object]],
    user_review: list[dict[str, object]],
    report: dict[str, object],
) -> None:
    if output_dir.exists():
        raise RagSftRetrievalFullReviewError("全量共识输出目录必须是新目录")
    output_dir.mkdir(parents=True)
    payloads = (
        (CONSENSUS_FILENAME, _write_jsonl(consensus)),
        (DECISIONS_FILENAME, _write_jsonl([item["decision"] for item in consensus])),
        (USER_REVIEW_FILENAME, _write_jsonl(user_review)),
        (
            REPORT_FILENAME,
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        ),
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
        raise RagSftRetrievalFullReviewError("无法发布全量双审共识") from error


def validate_retrieval_full_reviews(
    *, review_dir: Path, ledger_dir: Path, output_dir: Path
) -> dict[str, object]:
    """验证 495 条全量双审记录，并发布共识、扁平决策和用户终裁队列。"""

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RagSftRetrievalFullReviewError("全量共识输出目录必须是新目录")
    review_dir = Path(review_dir).resolve()
    queue_path = review_dir / queue_builder.QUEUE_FILENAME
    try:
        review_manifest_path, review_manifest = review_validator._verify_review_bundle(
            queue_path, expected_filename=queue_builder.QUEUE_FILENAME
        )
        queue = review_validator._load_jsonl(queue_path, "全量 review queue")
        queue_by_id = review_validator._validate_pilot(queue)
    except review_validator.RagSftRetrievalPilotReviewError as error:
        raise RagSftRetrievalFullReviewError(str(error)) from error
    if review_manifest.get("records", {}).get("semantic_review_queue") != len(queue):
        raise RagSftRetrievalFullReviewError("review queue 数量与 manifest 不一致")
    assembly_manifest_path, assembly_manifest, ledger_paths = _verify_assembly_bundle(
        Path(ledger_dir),
        review_manifest_path=review_manifest_path,
        queue_path=queue_path,
        queue=queue,
    )
    ledgers = {
        role: _validate_ledger(
            path, role=role, queue=queue, queue_by_id=queue_by_id
        )
        for role, path in ledger_paths.items()
    }

    consensus = []
    user_review = []
    decision_counts: Counter[str] = Counter()
    mode_counts: Counter[str] = Counter()
    escalation_counts: Counter[str] = Counter()
    for item in queue:
        decision, escalation = _exact_or_union_consensus(
            item,
            ledgers["legal_support"][item["source_id"]],
            ledgers["adversarial_boundary"][item["source_id"]],
        )
        if escalation is not None:
            user_review.append(escalation)
            escalation_counts.update(escalation["reason_codes"])
            continue
        if decision is None:
            raise RagSftRetrievalFullReviewError("共识记录未闭合")
        consensus.append(decision)
        decision_counts[decision["decision"]["decision"]] += 1
        mode_counts[decision["consensus_mode"]] += 1
    if len(consensus) + len(user_review) != len(queue):
        raise RagSftRetrievalFullReviewError("共识与用户终裁队列数量不闭合")
    report: dict[str, object] = {
        "pipeline": "rag_sft_retrieval_full_review_validation",
        "policy": {
            "review_roles": sorted(ROLE_TO_LEDGER_FILENAME),
            "exact_fact_consensus": True,
            "support_partition_disagreement": "conservative_support_union",
            "target_conflict_requires_user_review": True,
            "alternative_complete_answer_requires_user_review": True,
            "deterministic_approvals_merged": False,
            "canonical_corrections_processed": False,
            "training_authoring_emitted": False,
        },
        "input": {
            "review_bundle_manifest": _identity(review_manifest_path),
            "review_queue": _identity(queue_path, records=len(queue)),
            "ledger_assembly_manifest": _identity(assembly_manifest_path),
            "full_ledgers": {
                role: _identity(path, records=len(queue))
                for role, path in ledger_paths.items()
            },
        },
        "records": {
            "review_queue": len(queue),
            "review_rows": len(queue) * len(ledgers),
            "consensus": len(consensus),
            "consensus_decisions": dict(sorted(decision_counts.items())),
            "consensus_modes": dict(sorted(mode_counts.items())),
            "user_review_required": len(user_review),
            "escalation_reasons": dict(sorted(escalation_counts.items())),
        },
        "validation": {
            "review_bundle_hashes_verified": True,
            "ledger_assembly_hashes_verified": True,
            "each_ledger_binds_every_queue_record": True,
            "both_ledgers_follow_queue_order": True,
            "consensus_and_user_review_close_queue": True,
        },
        "readiness": {
            "full_reviews_complete": True,
            "user_review_required": bool(user_review),
            "training_ready": False,
        },
        "outputs": {
            "consensus": CONSENSUS_FILENAME,
            "flat_decisions": DECISIONS_FILENAME,
            "user_review": USER_REVIEW_FILENAME,
            "report": REPORT_FILENAME,
            "sha256_manifest": HASH_FILENAME,
        },
        "complete": True,
    }
    _publish(
        output_dir, consensus=consensus, user_review=user_review, report=report
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = validate_retrieval_full_reviews(
            review_dir=args.review_dir,
            ledger_dir=args.ledger_dir,
            output_dir=args.output_dir,
        )
    except RagSftRetrievalFullReviewError as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "RagSftRetrievalFullReviewError",
    "validate_retrieval_full_reviews",
]
