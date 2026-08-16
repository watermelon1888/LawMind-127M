"""把人工语义裁决发布为独立的 retrieved RAG-SFT 校准产物。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from . import materialize_rag_sft_retrieved as materializer
    from . import prepare_rag_sft as preparation
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import materialize_rag_sft_retrieved as materializer
    from dataset import prepare_rag_sft as preparation


ADJUDICATION_FILENAME = "rag-sft-retrieved-adjudication.jsonl"
CURATED_FILENAME = "rag-sft-retrieved-curated.jsonl"
REPORT_FILENAME = "rag-sft-retrieved-adjudication.json"
HASH_FILENAME = "rag-sft-retrieved-adjudication.sha256"

DECISION_FIELDS = {
    "source_id",
    "decision",
    "supporting_extra_chunk_ids",
    "non_supporting_extra_chunk_ids",
    "replacement_summary",
    "reason",
}
APPROVAL_DECISIONS = {"approve_answer", "approve_refusal"}
DETERMINISTIC_EXCLUSION_DECISIONS = {
    "exclude_duplicate",
    "exclude_incomplete_answer",
}
EXCLUSION_DECISIONS = {
    "exclude_duplicate",
    "exclude_incomplete_answer",
    "exclude_semantic_duplicate",
    "exclude_supporting_extra",
    "exclude_target_conflict",
}


class RagSftRetrievedAdjudicationError(RuntimeError):
    """retrieved 人工裁决不完整或无法安全发布。"""


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
                    raise RagSftRetrievedAdjudicationError(
                        f"{description}不允许空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RagSftRetrievedAdjudicationError(
                        f"{description}第 {line_number} 条必须是 object"
                    )
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftRetrievedAdjudicationError):
            raise
        raise RagSftRetrievedAdjudicationError(
            f"无法读取{description}: {path}"
        ) from error
    return records


def _unique_by(records, field: str, description: str) -> dict[str, dict[str, Any]]:
    result = {}
    for position, record in enumerate(records, start=1):
        value = record.get(field)
        if not isinstance(value, str) or not value or value in result:
            raise RagSftRetrievedAdjudicationError(
                f"{description}第 {position} 条 {field} 无效或重复"
            )
        result[value] = record
    return result


def _string_list(value: object, field: str) -> list[str]:
    if (
        not isinstance(value, list)
        or len(value) != len(set(value))
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise RagSftRetrievedAdjudicationError(f"{field} 必须是无重复字符串数组")
    return value


def _validate_decision(
    decision: dict[str, Any], locator: dict[str, Any]
) -> tuple[list[str], list[str]]:
    source_id = locator["source_id"]
    if set(decision) != DECISION_FIELDS:
        raise RagSftRetrievedAdjudicationError(f"{source_id} 裁决字段不符合封闭 schema")
    action = decision["decision"]
    if action not in APPROVAL_DECISIONS | EXCLUSION_DECISIONS:
        raise RagSftRetrievedAdjudicationError(f"{source_id} decision 无效")
    if not isinstance(decision["reason"], str) or not decision["reason"].strip():
        raise RagSftRetrievedAdjudicationError(f"{source_id} reason 不能为空")
    replacement = decision["replacement_summary"]
    if replacement is not None and (
        not isinstance(replacement, str)
        or not replacement.strip()
        or "\n" in replacement
        or "\r" in replacement
    ):
        raise RagSftRetrievedAdjudicationError(
            f"{source_id} replacement_summary 必须为空值或非空单行字符串"
        )
    supporting = _string_list(
        decision["supporting_extra_chunk_ids"],
        f"{source_id} supporting_extra_chunk_ids",
    )
    non_supporting = _string_list(
        decision["non_supporting_extra_chunk_ids"],
        f"{source_id} non_supporting_extra_chunk_ids",
    )
    if set(supporting) & set(non_supporting):
        raise RagSftRetrievedAdjudicationError(f"{source_id} extra 语义分组重叠")
    if action in DETERMINISTIC_EXCLUSION_DECISIONS:
        if supporting or non_supporting:
            raise RagSftRetrievedAdjudicationError(
                f"{source_id} 确定性隔离记录不得伪造 extra 语义分组"
            )
    elif set(supporting) | set(non_supporting) != set(
        locator["extra_non_gt_chunk_ids"]
    ):
        raise RagSftRetrievedAdjudicationError(f"{source_id} extra 语义分组未闭合")
    if action == "approve_answer" and supporting:
        raise RagSftRetrievedAdjudicationError(
            f"{source_id} 回答含直接支持 target 的额外证据，不能批准"
        )
    if action != "approve_answer" and replacement is not None:
        raise RagSftRetrievedAdjudicationError(
            f"{source_id} 只有 approve_answer 可提供 replacement_summary"
        )
    if action == "exclude_duplicate" and locator["disposition"] != "duplicate_existing":
        raise RagSftRetrievedAdjudicationError(f"{source_id} 不是确定性重复记录")
    if (
        action == "exclude_incomplete_answer"
        and locator["disposition"] != "needs_positive_summary"
    ):
        raise RagSftRetrievedAdjudicationError(f"{source_id} 不是缺少正向 summary 的记录")
    return supporting, non_supporting


def _build_curated_record(
    parent: dict[str, Any], locator: dict[str, Any], decision: dict[str, Any]
) -> dict[str, object]:
    action = decision["decision"]
    visible = list(locator["packaged_chunk_ids"])
    required = list(locator["required_chunk_ids"])
    required_complete = set(required).issubset(visible)
    if action == "approve_refusal":
        if required_complete:
            raise RagSftRetrievedAdjudicationError(
                f"{decision['source_id']} 拒答裁决的真实包已包含全部 required GT"
            )
        target = {"summary": "", "refuse": True}
        spans = []
    else:
        if not required_complete:
            raise RagSftRetrievedAdjudicationError(
                f"{decision['source_id']} 回答裁决的真实包缺少 required GT"
            )
        summary = decision["replacement_summary"] or parent["target"]["summary"]
        if not isinstance(summary, str) or not summary.strip():
            raise RagSftRetrievedAdjudicationError(
                f"{decision['source_id']} 回答裁决缺少已审核 summary"
            )
        target = {"summary": summary, "refuse": False}
        spans = (
            list(parent["support_spans"])
            if decision["replacement_summary"] is None
            else []
        )
    record = {
        "id": locator["derived_id"],
        "query_original": parent["query_original"],
        "evidence_source": "retrieved",
        "visible_chunk_ids": visible,
        "required_chunk_ids": required,
        "target": target,
        "support_spans": spans,
        "review_status": "approved",
        "review_notes": (
            "retrieved 全量语义审核通过；"
            f"源记录 {decision['source_id']}，裁决为 {action}。"
        ),
    }
    preparation._validate_record_shape(record, decision["source_id"])
    return record


def _canonical_revision_required(
    adjudication: list[dict[str, object]],
) -> list[str]:
    return [
        str(item["source_id"])
        for item in adjudication
        if item["decision"] == "exclude_target_conflict"
    ]


def _publish(output_dir: Path, adjudication, curated, report) -> None:
    if output_dir.exists():
        raise RagSftRetrievedAdjudicationError(f"裁决输出目录必须是新目录: {output_dir}")
    output_dir.mkdir(parents=True)
    payloads = (
        (
            ADJUDICATION_FILENAME,
            "".join(
                json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
                for item in adjudication
            ),
        ),
        (
            CURATED_FILENAME,
            "".join(
                json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n"
                for item in curated
            ),
        ),
        (REPORT_FILENAME, json.dumps(report, ensure_ascii=False, indent=2) + "\n"),
    )
    partials = []
    published = []
    try:
        for filename, payload in payloads:
            final = output_dir / filename
            partial = output_dir / f"{filename}.partial"
            partial.write_text(payload, encoding="utf-8", newline="\n")
            partials.append((partial, final))
        hashes = "".join(
            f"{_sha256(partial)}  {final.name}\n" for partial, final in partials
        )
        hash_partial = output_dir / f"{HASH_FILENAME}.partial"
        hash_partial.write_text(hashes, encoding="utf-8", newline="\n")
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
        raise RagSftRetrievedAdjudicationError("无法发布 retrieved 裁决产物") from error


def adjudicate_rag_sft_retrieved(
    *,
    canonical_authoring_path: Path,
    locator_path: Path,
    provisional_authoring_path: Path,
    decisions_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """校验逐条语义裁决并发布全量裁决与批准记录。"""

    canonical_records = _load_jsonl(canonical_authoring_path, "canonical authoring")
    locators = _load_jsonl(locator_path, "retrieved locator")
    provisional = _load_jsonl(provisional_authoring_path, "retrieved provisional")
    decisions = _load_jsonl(decisions_path, "人工裁决")
    canonical_by_id = _unique_by(canonical_records, "id", "canonical authoring")
    provisional_by_id = _unique_by(provisional, "id", "retrieved provisional")
    decision_by_id = _unique_by(decisions, "source_id", "人工裁决")
    locator_by_id = _unique_by(locators, "source_id", "retrieved locator")
    if set(decision_by_id) != set(locator_by_id):
        raise RagSftRetrievedAdjudicationError("人工裁决必须逐条覆盖全部 locator")

    curated = []
    adjudication = []
    curated_signatures = set()
    canonical_signatures = {
        materializer._signature(record) for record in canonical_records
    }
    decision_counts: Counter[str] = Counter()
    original_counts: Counter[str] = Counter()
    for locator in locators:
        source_id = locator["source_id"]
        if source_id not in canonical_by_id:
            raise RagSftRetrievedAdjudicationError(f"{source_id} 缺少 canonical 父记录")
        if locator.get("derived_id") != materializer._derived_id(source_id):
            raise RagSftRetrievedAdjudicationError(f"{source_id} derived_id 无效")
        decision = decision_by_id[source_id]
        supporting, non_supporting = _validate_decision(decision, locator)
        action = decision["decision"]
        decision_counts[action] += 1
        original_counts[locator["disposition"]] += 1
        final_record = None
        if action in APPROVAL_DECISIONS:
            final_record = _build_curated_record(
                canonical_by_id[source_id], locator, decision
            )
            signature = materializer._signature(final_record)
            if signature in canonical_signatures or signature in curated_signatures:
                raise RagSftRetrievedAdjudicationError(
                    f"{source_id} 批准后形成未裁决的重复 conversations"
                )
            curated_signatures.add(signature)
            curated.append(final_record)
        adjudication.append(
            {
                "source_id": source_id,
                "derived_id": locator["derived_id"],
                "original_disposition": locator["disposition"],
                "decision": action,
                "supporting_extra_chunk_ids": supporting,
                "non_supporting_extra_chunk_ids": non_supporting,
                "replacement_summary": decision["replacement_summary"],
                "reason": decision["reason"],
                "curated_record_emitted": final_record is not None,
            }
        )

    provisional_ids = set(provisional_by_id)
    expected_provisional_ids = {
        locator["derived_id"]
        for locator in locators
        if locator["disposition"]
        in {
            "retrieved_answer_approved",
            "retrieved_answer_draft",
            "retrieved_refusal_draft",
        }
    }
    if provisional_ids != expected_provisional_ids:
        raise RagSftRetrievedAdjudicationError("provisional 与 locator 输出身份不闭合")

    report: dict[str, object] = {
        "pipeline": "rag_sft_retrieved_semantic_adjudication",
        "scope": {
            "reviewed_records": len(locators),
            "extra_count_used_as_decision_threshold": False,
            "retrieval_order_preserved": True,
            "original_materialization_modified": False,
        },
        "frozen_rules": {
            "semantic_review_extras_partitioned": True,
            "deterministic_exclusions_require_no_semantic_partition": True,
            "answer_requires_all_required_gt": True,
            "answer_with_supporting_extra_is_excluded": True,
            "refusal_requires_missing_required_gt_and_no_alternative_complete_answer": True,
            "incomplete_positive_rewrite_is_forbidden": True,
            "target_conflict_requires_canonical_revision": True,
            "semantic_duplicates_are_not_emitted": True,
        },
        "input": {
            "canonical_authoring": _identity(
                canonical_authoring_path, records=len(canonical_records)
            ),
            "locators": _identity(locator_path, records=len(locators)),
            "provisional_authoring": _identity(
                provisional_authoring_path, records=len(provisional)
            ),
            "decisions": _identity(decisions_path, records=len(decisions)),
        },
        "records": {
            "original_dispositions": dict(sorted(original_counts.items())),
            "decisions": dict(sorted(decision_counts.items())),
            "curated": len(curated),
            "excluded": len(locators) - len(curated),
        },
        "canonical_revision_required": _canonical_revision_required(adjudication),
        "validation": {
            "all_locators_adjudicated": len(decisions) == len(locators),
            "semantic_extra_partitions_closed": True,
            "curated_ids_unique": len({item["id"] for item in curated}) == len(curated),
            "curated_signatures_unique": len(curated_signatures) == len(curated),
            "no_canonical_conversation_duplicates": True,
        },
        "readiness": {
            "calibration_review_complete": True,
            "semantic_rule_frozen": True,
            "expansion_design_may_start": True,
            "training_ready": False,
        },
        "outputs": {
            "adjudication": ADJUDICATION_FILENAME,
            "curated_authoring": CURATED_FILENAME,
            "report": REPORT_FILENAME,
            "sha256_manifest": HASH_FILENAME,
        },
        "complete": True,
    }
    _publish(output_dir.resolve(), adjudication, curated, report)
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-authoring", type=Path, required=True)
    parser.add_argument("--locators", type=Path, required=True)
    parser.add_argument("--provisional-authoring", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = adjudicate_rag_sft_retrieved(
        canonical_authoring_path=args.canonical_authoring,
        locator_path=args.locators,
        provisional_authoring_path=args.provisional_authoring,
        decisions_path=args.decisions,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
