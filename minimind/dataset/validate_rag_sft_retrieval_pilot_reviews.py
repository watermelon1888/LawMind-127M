"""校验 retrieved pilot 的双角色语义审核并发布共识与用户终裁队列。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from . import build_rag_sft_retrieval_review_queue as queue_builder
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import build_rag_sft_retrieval_review_queue as queue_builder


RAG_SFT_ROOT = Path(__file__).resolve().parent / "RAG-SFT"
DEFAULT_REVIEW_DIR = (
    RAG_SFT_ROOT / "retrieved" / "review-canonical-v1-20260809"
)
DEFAULT_LEGAL_LEDGER = (
    RAG_SFT_ROOT
    / "retrieved"
    / "review-ledgers-canonical-v1-20260809"
    / "legal-support.jsonl"
)
DEFAULT_ADVERSARIAL_LEDGER = (
    RAG_SFT_ROOT
    / "retrieved"
    / "review-ledgers-canonical-v1-20260809"
    / "adversarial-boundary.jsonl"
)

CONSENSUS_FILENAME = "rag-sft-retrieval-pilot-consensus.jsonl"
DECISIONS_FILENAME = "rag-sft-retrieval-pilot-decisions.jsonl"
USER_REVIEW_FILENAME = "rag-sft-retrieval-pilot-user-review.jsonl"
REPORT_FILENAME = "rag-sft-retrieval-pilot-review-report.json"
HASH_FILENAME = "rag-sft-retrieval-pilot-review.sha256"

REVIEW_FIELDS = {
    "source_id",
    "review_item_sha256",
    "reviewer_role",
    "supporting_extra_chunk_ids",
    "non_supporting_extra_chunk_ids",
    "alternative_complete_answer_chunk_ids",
    "target_conflict",
    "notes",
}
REVIEWER_ROLES = {"legal_support", "adversarial_boundary"}
ANSWER_LANES = {"answer_same_law_extra", "answer_cross_law_only"}
REFUSAL_LANES = {
    "refusal_budget_dropped_gt",
    "refusal_top5_missing_gt",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}$")


class RagSftRetrievalPilotReviewError(RuntimeError):
    """pilot 双审记录不完整、未绑定当前证据或无法闭合。"""


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
                    raise RagSftRetrievalPilotReviewError(
                        f"{description}不允许空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RagSftRetrievalPilotReviewError(
                        f"{description}第 {line_number} 条必须是 object"
                    )
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftRetrievalPilotReviewError):
            raise
        raise RagSftRetrievalPilotReviewError(
            f"无法读取{description}: {path}"
        ) from error
    if not records:
        raise RagSftRetrievalPilotReviewError(f"{description}不能为空")
    return records


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftRetrievalPilotReviewError(
            f"无法读取{description}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise RagSftRetrievalPilotReviewError(f"{description}必须是 JSON object")
    return value


def _verify_review_bundle(
    review_items_path: Path,
    *,
    expected_filename: str = queue_builder.PILOT_FILENAME,
) -> tuple[Path, dict[str, Any]]:
    directory = review_items_path.parent
    expected_paths = {
        queue_builder.DETERMINISTIC_FILENAME: directory
        / queue_builder.DETERMINISTIC_FILENAME,
        queue_builder.QUEUE_FILENAME: directory / queue_builder.QUEUE_FILENAME,
        queue_builder.PILOT_FILENAME: directory / queue_builder.PILOT_FILENAME,
        queue_builder.MANIFEST_FILENAME: directory / queue_builder.MANIFEST_FILENAME,
    }
    if expected_filename not in {
        queue_builder.PILOT_FILENAME,
        queue_builder.QUEUE_FILENAME,
    }:
        raise RagSftRetrievalPilotReviewError("review bundle 目标文件类型无效")
    if review_items_path != expected_paths[expected_filename].resolve():
        raise RagSftRetrievalPilotReviewError(
            "审核对象路径与 review bundle 契约不一致"
        )
    hash_path = directory / queue_builder.HASH_FILENAME
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftRetrievalPilotReviewError("无法读取 review bundle SHA-256 清单") from error
    parsed = {}
    for line in lines:
        parts = line.split("  ", 1)
        if (
            len(parts) != 2
            or parts[1] in parsed
            or not _SHA256_RE.fullmatch(parts[0])
        ):
            raise RagSftRetrievalPilotReviewError("review bundle SHA-256 清单格式无效")
        parsed[parts[1]] = parts[0]
    if set(parsed) != set(expected_paths):
        raise RagSftRetrievalPilotReviewError("review bundle SHA-256 文件集合无效")
    for name, path in expected_paths.items():
        if not path.is_file() or _sha256(path) != parsed[name]:
            raise RagSftRetrievalPilotReviewError(f"review bundle 文件身份已变化: {name}")
    manifest_path = expected_paths[queue_builder.MANIFEST_FILENAME]
    manifest = _load_json(manifest_path, "review bundle manifest")
    if (
        manifest.get("pipeline") != "rag_sft_retrieval_review_queue"
        or manifest.get("complete") is not True
        or manifest.get("readiness", {}).get("review_queue_ready") is not True
        or manifest.get("outputs", {}).get(
            "pilot" if expected_filename == queue_builder.PILOT_FILENAME else "review_queue"
        )
        != expected_filename
    ):
        raise RagSftRetrievalPilotReviewError("review bundle manifest 状态无效")
    return manifest_path, manifest


def _ordered_string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise RagSftRetrievalPilotReviewError(f"{field} 必须是无重复字符串数组")
    if len(value) != len(set(value)):
        raise RagSftRetrievalPilotReviewError(f"{field} 必须是无重复字符串数组")
    return value


def _validate_pilot(pilot: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id = {}
    for position, item in enumerate(pilot, start=1):
        prefix = f"pilot 第 {position} 条"
        source_id = item.get("source_id")
        if not isinstance(source_id, str) or not source_id or source_id in by_id:
            raise RagSftRetrievalPilotReviewError(f"{prefix} source_id 无效或重复")
        lane = item.get("lane")
        if lane not in ANSWER_LANES | REFUSAL_LANES:
            raise RagSftRetrievalPilotReviewError(f"{prefix} lane 无效")
        required = item.get("required_evidence")
        if not isinstance(required, list) or not required or any(
            not isinstance(entry, dict) for entry in required
        ):
            raise RagSftRetrievalPilotReviewError(f"{prefix} required_evidence 无效")
        required_ids = [entry.get("chunk_id") for entry in required]
        if any(
            not isinstance(chunk_id, str) or not chunk_id
            for chunk_id in required_ids
        ) or len(set(required_ids)) != len(required_ids):
            raise RagSftRetrievalPilotReviewError(
                f"{prefix} required_evidence chunk_id 无效或重复"
            )
        packaged = item.get("packaged_evidence")
        if not isinstance(packaged, list) or not packaged:
            raise RagSftRetrievalPilotReviewError(f"{prefix} packaged_evidence 无效")
        packaged_ids = [entry.get("chunk_id") for entry in packaged if isinstance(entry, dict)]
        if len(packaged_ids) != len(packaged) or any(
            not isinstance(chunk_id, str) or not chunk_id for chunk_id in packaged_ids
        ):
            raise RagSftRetrievalPilotReviewError(f"{prefix} packaged_evidence chunk_id 无效")
        if len(set(packaged_ids)) != len(packaged_ids):
            raise RagSftRetrievalPilotReviewError(
                f"{prefix} packaged_evidence chunk_id 重复"
            )
        extras = _ordered_string_list(
            item.get("extra_non_gt_chunk_ids"),
            f"{prefix} extra_non_gt_chunk_ids",
        )
        missing = _ordered_string_list(
            item.get("missing_required_chunk_ids"),
            f"{prefix} missing_required_chunk_ids",
        )
        expected_extras = [
            chunk_id for chunk_id in packaged_ids if chunk_id not in set(required_ids)
        ]
        expected_missing = [
            chunk_id for chunk_id in required_ids if chunk_id not in set(packaged_ids)
        ]
        if extras != expected_extras or missing != expected_missing:
            raise RagSftRetrievalPilotReviewError(
                f"{prefix} required、packaged、missing 与 extra 关系不闭合"
            )
        if lane in ANSWER_LANES and missing:
            raise RagSftRetrievalPilotReviewError(f"{prefix}回答 lane 不能缺少 required GT")
        if lane in REFUSAL_LANES and not missing:
            raise RagSftRetrievalPilotReviewError(f"{prefix}拒答 lane 必须缺少 required GT")
        if not isinstance(item.get("locator_sha256"), str) or not _SHA256_RE.fullmatch(
            item["locator_sha256"]
        ):
            raise RagSftRetrievalPilotReviewError(f"{prefix} locator_sha256 无效")
        by_id[source_id] = item
    return by_id


def _validate_review(
    review: dict[str, Any],
    *,
    position: int,
    expected_role: str,
    pilot_by_id: dict[str, dict[str, Any]],
) -> None:
    prefix = f"{expected_role} ledger 第 {position} 条"
    if set(review) != REVIEW_FIELDS:
        raise RagSftRetrievalPilotReviewError(f"{prefix}顶层字段必须精确匹配 schema")
    source_id = review["source_id"]
    if source_id not in pilot_by_id:
        raise RagSftRetrievalPilotReviewError(f"{prefix}引用了未知 source_id")
    if review["reviewer_role"] != expected_role:
        raise RagSftRetrievalPilotReviewError(f"{prefix} reviewer_role 与 ledger 不一致")
    item = pilot_by_id[source_id]
    if review["review_item_sha256"] != queue_builder.canonical_digest(item):
        raise RagSftRetrievalPilotReviewError(f"{prefix}没有绑定当前 pilot 内容哈希")
    extras = list(item["extra_non_gt_chunk_ids"])
    supporting = _ordered_string_list(
        review["supporting_extra_chunk_ids"],
        f"{prefix} supporting_extra_chunk_ids",
    )
    non_supporting = _ordered_string_list(
        review["non_supporting_extra_chunk_ids"],
        f"{prefix} non_supporting_extra_chunk_ids",
    )
    if set(supporting) & set(non_supporting) or set(supporting) | set(
        non_supporting
    ) != set(extras):
        raise RagSftRetrievalPilotReviewError(f"{prefix} extra 语义分组未闭合")
    if supporting != [item for item in extras if item in set(supporting)] or non_supporting != [
        item for item in extras if item in set(non_supporting)
    ]:
        raise RagSftRetrievalPilotReviewError(f"{prefix} extra 分组必须保留检索包顺序")

    alternatives = _ordered_string_list(
        review["alternative_complete_answer_chunk_ids"],
        f"{prefix} alternative_complete_answer_chunk_ids",
    )
    packaged_ids = [entry["chunk_id"] for entry in item["packaged_evidence"]]
    if alternatives != [item for item in packaged_ids if item in set(alternatives)]:
        raise RagSftRetrievalPilotReviewError(f"{prefix}替代答案证据必须保留检索包顺序")
    if not set(alternatives).issubset(packaged_ids):
        raise RagSftRetrievalPilotReviewError(f"{prefix}替代答案引用了模型不可见证据")
    required_ids = {entry["chunk_id"] for entry in item["required_evidence"]}
    if not (set(alternatives) - required_ids).issubset(supporting):
        raise RagSftRetrievalPilotReviewError(
            f"{prefix}替代答案中的 non-GT 证据必须归为 supporting extra"
        )
    if item["lane"] in ANSWER_LANES and alternatives:
        raise RagSftRetrievalPilotReviewError(f"{prefix}回答 lane 不接受替代答案字段")
    if not isinstance(review["target_conflict"], bool):
        raise RagSftRetrievalPilotReviewError(f"{prefix} target_conflict 必须是布尔值")
    notes = review["notes"]
    if not isinstance(notes, str) or not notes.strip() or "\n" in notes or "\r" in notes:
        raise RagSftRetrievalPilotReviewError(f"{prefix} notes 必须是非空单行字符串")


def _review_facts(review: dict[str, Any]) -> dict[str, object]:
    return {
        "supporting_extra_chunk_ids": review["supporting_extra_chunk_ids"],
        "non_supporting_extra_chunk_ids": review["non_supporting_extra_chunk_ids"],
        "alternative_complete_answer_chunk_ids": review[
            "alternative_complete_answer_chunk_ids"
        ],
        "target_conflict": review["target_conflict"],
    }


def _consensus_decision(
    item: dict[str, Any], facts: dict[str, object]
) -> dict[str, object] | None:
    lane = item["lane"]
    if facts["target_conflict"] or facts["alternative_complete_answer_chunk_ids"]:
        return None
    if lane in ANSWER_LANES:
        action = (
            "exclude_supporting_extra"
            if facts["supporting_extra_chunk_ids"]
            else "approve_answer"
        )
        reason = (
            "双审一致认定存在直接支持 target 的额外证据。"
            if action == "exclude_supporting_extra"
            else "双审一致认定 extra 均不直接支持 target，且 target 无冲突。"
        )
    else:
        action = "approve_refusal"
        reason = "双审一致认定模型可见证据不能形成替代完整答案，且 target 无冲突。"
    return {
        "source_id": item["source_id"],
        "derived_id": item["derived_id"],
        "locator_sha256": item["locator_sha256"],
        "decision": {
            "source_id": item["source_id"],
            "decision": action,
            "supporting_extra_chunk_ids": facts["supporting_extra_chunk_ids"],
            "non_supporting_extra_chunk_ids": facts["non_supporting_extra_chunk_ids"],
            "replacement_summary": None,
            "reason": reason,
        },
    }


def _write_jsonl(values: list[dict[str, object]]) -> str:
    return "".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for item in values
    )


def _publish(
    output_dir: Path,
    consensus: list[dict[str, object]],
    user_review: list[dict[str, object]],
    report: dict[str, object],
) -> None:
    if output_dir.exists():
        raise RagSftRetrievalPilotReviewError(f"pilot 审核输出目录必须是新目录: {output_dir}")
    output_dir.mkdir(parents=True)
    payloads = (
        (CONSENSUS_FILENAME, _write_jsonl(consensus)),
        (
            DECISIONS_FILENAME,
            _write_jsonl([item["decision"] for item in consensus]),
        ),
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
        raise RagSftRetrievalPilotReviewError("无法发布 pilot 双审结果") from error


def validate_retrieval_pilot_reviews(
    *,
    pilot_path: Path,
    legal_support_ledger_path: Path,
    adversarial_boundary_ledger_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """严格校验两个角色的逐条判断并确定性生成共识与升级队列。"""

    pilot_path = Path(pilot_path).resolve()
    legal_support_ledger_path = Path(legal_support_ledger_path).resolve()
    adversarial_boundary_ledger_path = Path(adversarial_boundary_ledger_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RagSftRetrievalPilotReviewError(f"pilot 审核输出目录必须是新目录: {output_dir}")
    review_manifest_path, review_manifest = _verify_review_bundle(pilot_path)
    pilot = _load_jsonl(pilot_path, "retrieved pilot")
    pilot_by_id = _validate_pilot(pilot)
    if review_manifest.get("records", {}).get("pilot") != len(pilot):
        raise RagSftRetrievalPilotReviewError("pilot 数量与 review bundle manifest 不一致")
    ledgers = {
        "legal_support": (
            legal_support_ledger_path,
            _load_jsonl(legal_support_ledger_path, "legal_support ledger"),
        ),
        "adversarial_boundary": (
            adversarial_boundary_ledger_path,
            _load_jsonl(
                adversarial_boundary_ledger_path,
                "adversarial_boundary ledger",
            ),
        ),
    }
    reviews_by_role: dict[str, dict[str, dict[str, Any]]] = {}
    for role, (_, reviews) in ledgers.items():
        by_id = {}
        for position, review in enumerate(reviews, start=1):
            _validate_review(
                review,
                position=position,
                expected_role=role,
                pilot_by_id=pilot_by_id,
            )
            source_id = review["source_id"]
            if source_id in by_id:
                raise RagSftRetrievalPilotReviewError(
                    f"{role} ledger source_id 重复: {source_id}"
                )
            by_id[source_id] = review
        if set(by_id) != set(pilot_by_id):
            raise RagSftRetrievalPilotReviewError(
                f"{role} ledger 必须逐条覆盖全部 pilot"
            )
        reviews_by_role[role] = by_id

    consensus = []
    user_review = []
    consensus_counts: Counter[str] = Counter()
    escalation_counts: Counter[str] = Counter()
    for item in pilot:
        source_id = item["source_id"]
        legal = reviews_by_role["legal_support"][source_id]
        adversarial = reviews_by_role["adversarial_boundary"][source_id]
        legal_facts = _review_facts(legal)
        adversarial_facts = _review_facts(adversarial)
        reason_codes = []
        if legal_facts != adversarial_facts:
            reason_codes.append("review_disagreement")
        else:
            if legal_facts["target_conflict"]:
                reason_codes.append("target_conflict")
            if legal_facts["alternative_complete_answer_chunk_ids"]:
                reason_codes.append("alternative_complete_answer")
        if reason_codes:
            escalation_counts.update(reason_codes)
            user_review.append(
                {
                    "source_id": source_id,
                    "derived_id": item["derived_id"],
                    "locator_sha256": item["locator_sha256"],
                    "lane": item["lane"],
                    "reason_codes": reason_codes,
                    "legal_support_review": legal,
                    "adversarial_boundary_review": adversarial,
                }
            )
            continue
        decision = _consensus_decision(item, legal_facts)
        if decision is None:
            raise RagSftRetrievalPilotReviewError(
                f"{source_id} 共识事实无法确定性生成裁决"
            )
        consensus_counts[decision["decision"]["decision"]] += 1
        consensus.append(decision)

    if len(consensus) + len(user_review) != len(pilot):
        raise RagSftRetrievalPilotReviewError("共识与用户终裁队列数量不闭合")
    report: dict[str, object] = {
        "pipeline": "rag_sft_retrieval_pilot_review_validation",
        "policy": {
            "review_roles": sorted(REVIEWER_ROLES),
            "majority_vote": False,
            "exact_fact_consensus_required": True,
            "target_conflict_requires_user_review": True,
            "alternative_complete_answer_requires_user_review": True,
            "training_authoring_emitted": False,
            "flat_decisions_match_adjudicator_schema": True,
        },
        "input": {
            "pilot": _identity(pilot_path, records=len(pilot)),
            "review_bundle_manifest": _identity(review_manifest_path),
            "legal_support_ledger": _identity(
                legal_support_ledger_path,
                records=len(ledgers["legal_support"][1]),
            ),
            "adversarial_boundary_ledger": _identity(
                adversarial_boundary_ledger_path,
                records=len(ledgers["adversarial_boundary"][1]),
            ),
        },
        "records": {
            "pilot": len(pilot),
            "review_rows": sum(len(value[1]) for value in ledgers.values()),
            "consensus": len(consensus),
            "consensus_decisions": dict(sorted(consensus_counts.items())),
            "user_review_required": len(user_review),
            "escalation_reasons": dict(sorted(escalation_counts.items())),
        },
        "validation": {
            "each_role_covers_every_pilot_record": True,
            "all_reviews_bind_current_pilot_hash": True,
            "all_extra_partitions_closed": True,
            "consensus_and_user_review_close_pilot": True,
        },
        "readiness": {
            "pilot_reviews_complete": True,
            "pilot_user_review_required": bool(user_review),
            "full_queue_review_may_start": not user_review,
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
    _publish(output_dir, consensus, user_review, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pilot",
        type=Path,
        default=DEFAULT_REVIEW_DIR / queue_builder.PILOT_FILENAME,
    )
    parser.add_argument(
        "--legal-support-ledger", type=Path, default=DEFAULT_LEGAL_LEDGER
    )
    parser.add_argument(
        "--adversarial-boundary-ledger",
        type=Path,
        default=DEFAULT_ADVERSARIAL_LEDGER,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = validate_retrieval_pilot_reviews(
            pilot_path=args.pilot,
            legal_support_ledger_path=args.legal_support_ledger,
            adversarial_boundary_ledger_path=args.adversarial_boundary_ledger,
            output_dir=args.output_dir,
        )
    except RagSftRetrievalPilotReviewError as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "REVIEW_FIELDS",
    "RagSftRetrievalPilotReviewError",
    "validate_retrieval_pilot_reviews",
]
