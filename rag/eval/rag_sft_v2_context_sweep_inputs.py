"""发布 RAG-SFT v2 的固定题目上下文阶梯评估输入。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from rag.answering import (
    AnswerPromptTokenCounter,
    EvidencePackage,
    EvidencePackager,
    EvidencePackagingError,
)
from rag.core import Evidence
from rag.eval import rag_sft_v2_evaluation_inputs as source_inputs
from rag.knowledge import ArticleRepository


PIPELINE = "rag_sft_v2_context_sweep_inputs_v2"
EXPECTED_RETRIEVAL_PIPELINE = (
    "legal_rag_retrieval_baseline_original_top5_two_field_compact_v3"
)
CONTEXT_LIMITS = (768, 896, 1024, 1280)
MAX_OUTPUT_TOKENS = 150
SOURCE_QUERY_COUNTS = {"legal_query": 140, "exact_lookup": 30}
SELECTION_COUNTS = {
    "rescuable_packaging": 9,
    "retrieval_failure_negative_control": 27,
    "complete_matched_control": 9,
}
EXPECTED_QUERIES = sum(SELECTION_COUNTS.values())
EXPECTED_CELLS = EXPECTED_QUERIES * len(CONTEXT_LIMITS)


def _tokenizer_identity(tokenizer: Any, tokenizer_path: str | Path) -> dict[str, Any]:
    root = Path(tokenizer_path).resolve()
    return {
        "path": str(root),
        "vocab_size": len(tokenizer),
        "chat_template_sha256": hashlib.sha256(
            tokenizer.chat_template.encode("utf-8")
        ).hexdigest(),
        "files": {
            filename: {
                "bytes": (root / filename).stat().st_size,
                "sha256": source_inputs._sha256_file(root / filename),
            }
            for filename in ("tokenizer.json", "tokenizer_config.json")
        },
    }


def _manifest_tokenizer_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or not isinstance(value.get("files"), dict):
        raise ValueError("冻结检索 manifest 缺少 Tokenizer 身份")
    return {
        "vocab_size": value.get("vocab_size"),
        "chat_template_sha256": value.get("chat_template_sha256"),
        "files": {
            filename: {
                "bytes": item.get("bytes"),
                "sha256": item.get("sha256"),
            }
            for filename, item in value["files"].items()
            if isinstance(filename, str) and isinstance(item, dict)
        },
    }


def _actual_tokenizer_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "vocab_size": value["vocab_size"],
        "chat_template_sha256": value["chat_template_sha256"],
        "files": {
            filename: {
                "bytes": item["bytes"],
                "sha256": item["sha256"],
            }
            for filename, item in value["files"].items()
        },
    }


def _evidence_records(
    repository: ArticleRepository, visible_chunk_ids: list[str]
) -> list[dict[str, str]]:
    return [
        source_inputs._evidence_record(
            repository.get_by_chunk_id(chunk_id), index
        )
        for index, chunk_id in enumerate(visible_chunk_ids, 1)
    ]


def _legal_cell(
    *,
    case: Mapping[str, Any],
    retrieved: Mapping[str, Any],
    required: list[str],
    repository: ArticleRepository,
    packager: EvidencePackager,
    counter: AnswerPromptTokenCounter,
    context_limit: int,
    selection_role: str | None = None,
    matched_query_id: str | None = None,
) -> dict[str, Any]:
    query_id = case["id"]
    query = case["query_original"]
    retrieval = retrieved.get("retrieval")
    reranked = retrieval.get("reranked_top5_chunk_ids") if isinstance(retrieval, dict) else None
    if (
        not isinstance(reranked, list)
        or not 1 <= len(reranked) <= 5
        or len(reranked) != len(set(reranked))
    ):
        raise ValueError(f"{query_id} 缺少有效冻结 rerank top-5")
    articles = [repository.get_by_chunk_id(chunk_id) for chunk_id in reranked]
    status = "ready"
    failure_reason = None
    try:
        package, prompt_tokens = packager.build(query, articles)
        visible = [
            f"{item.law_name}#{item.article_no}" for item in package.evidence
        ]
    except EvidencePackagingError as error:
        status = "overbudget"
        failure_reason = str(error)
        first = articles[0]
        prompt_tokens = counter(
            EvidencePackage(
                query=query,
                evidence=(
                    Evidence(first.law_name, first.article_no, first.content),
                ),
            )
        )
        visible = []
    required_set = set(required)
    candidate_metrics = retrieval.get("candidate_pool_metrics", {})
    top5_metrics = retrieval.get("reranked_top5_metrics", {})
    return {
        "query_id": query_id,
        "query_type": "legal_query",
        "selection_role": selection_role,
        "matched_query_id": matched_query_id,
        "context_limit": context_limit,
        "status": status,
        "failure_reason": failure_reason,
        "query": query,
        "evidence": _evidence_records(repository, visible),
        "visible_chunk_ids": visible,
        "required_chunk_ids": required,
        "hard_negative_chunk_ids": [
            chunk_id for chunk_id in visible if chunk_id not in required_set
        ],
        "retrieval_attribution": {
            "source": "frozen_rerank_context_repackaging",
            "candidate_pool_complete": candidate_metrics.get("complete_hit"),
            "reranked_top5_complete": top5_metrics.get("complete_hit"),
            "packaged_complete": required_set.issubset(visible),
            "prompt_tokens": prompt_tokens,
            "reranked_top5_chunk_ids": reranked,
        },
    }


def _select_legal_queries(
    baseline_records: list[dict[str, Any]],
) -> tuple[list[str], dict[str, str], dict[str, str | None]]:
    incomplete = [
        record
        for record in baseline_records
        if record["status"] == "ready"
        and record["retrieval_attribution"]["packaged_complete"] is False
    ]
    rescuable = [
        record
        for record in incomplete
        if record["retrieval_attribution"]["reranked_top5_complete"] is True
    ]
    retrieval_failures = [
        record
        for record in incomplete
        if record["retrieval_attribution"]["reranked_top5_complete"] is False
    ]
    if (
        len(rescuable) != SELECTION_COUNTS["rescuable_packaging"]
        or len(retrieval_failures)
        != SELECTION_COUNTS["retrieval_failure_negative_control"]
    ):
        raise ValueError(
            "768 档 packaged_incomplete 分层不闭合: "
            f"rescuable={len(rescuable)}, retrieval_failure={len(retrieval_failures)}"
        )

    complete_candidates = [
        record
        for record in baseline_records
        if record["status"] == "ready"
        and record["retrieval_attribution"]["packaged_complete"] is True
    ]
    unused = {record["query_id"]: record for record in complete_candidates}
    matched_pairs: dict[str, str] = {}
    for target in sorted(rescuable, key=lambda record: record["query_id"]):
        target_required = len(target["required_chunk_ids"])
        target_evidence = len(target["visible_chunk_ids"])
        target_prompt = target["retrieval_attribution"]["prompt_tokens"]
        control = min(
            unused.values(),
            key=lambda record: (
                abs(len(record["required_chunk_ids"]) - target_required),
                abs(len(record["visible_chunk_ids"]) - target_evidence),
                abs(
                    record["retrieval_attribution"]["prompt_tokens"]
                    - target_prompt
                ),
                record["query_id"],
            ),
        )
        matched_pairs[target["query_id"]] = control["query_id"]
        del unused[control["query_id"]]

    roles = {
        **{record["query_id"]: "rescuable_packaging" for record in rescuable},
        **{
            record["query_id"]: "retrieval_failure_negative_control"
            for record in retrieval_failures
        },
        **{
            control_id: "complete_matched_control"
            for control_id in matched_pairs.values()
        },
    }
    matches: dict[str, str | None] = {query_id: None for query_id in roles}
    for target_id, control_id in matched_pairs.items():
        matches[target_id] = control_id
        matches[control_id] = target_id
    selected_ids = [
        record["query_id"]
        for record in baseline_records
        if record["query_id"] in roles
    ]
    if len(selected_ids) != EXPECTED_QUERIES:
        raise ValueError(f"上下文外推题目选择数量不闭合: {len(selected_ids)}")
    return selected_ids, roles, matches


def build_context_sweep_records(
    eval_cases: list[dict[str, Any]],
    retrieval_records: list[dict[str, Any]],
    repository: ArticleRepository,
    counter: AnswerPromptTokenCounter,
) -> list[dict[str, Any]]:
    """选择固定45题，并按四档上下文重新构包。"""

    retrieval_by_id = {
        record.get("query_id"): record for record in retrieval_records
    }
    if len(retrieval_by_id) != SOURCE_QUERY_COUNTS["legal_query"] or None in retrieval_by_id:
        raise ValueError("冻结检索 records 必须包含 140 个唯一 legal query")
    answer_cases = [
        case for case in eval_cases if case.get("expected_action") == "answer"
    ]
    counts = {
        query_type: sum(case.get("query_type") == query_type for case in answer_cases)
        for query_type in SOURCE_QUERY_COUNTS
    }
    if counts != SOURCE_QUERY_COUNTS:
        raise ValueError(f"固定评估题计数不闭合: {counts}")
    query_ids = [case.get("id") for case in answer_cases]
    if any(not isinstance(query_id, str) or not query_id for query_id in query_ids):
        raise ValueError("固定评估题 query_id 无效")
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("固定评估题 query_id 重复")

    case_by_id = {case["id"]: case for case in answer_cases}
    baseline_packager = EvidencePackager(
        context_limit=CONTEXT_LIMITS[0],
        max_output_tokens=MAX_OUTPUT_TOKENS,
        count_prompt_tokens=counter,
    )
    baseline_records = []
    for case in answer_cases:
        if case["query_type"] != "legal_query":
            continue
        retrieved = retrieval_by_id.get(case["id"])
        if retrieved is None:
            raise ValueError(f"{case['id']} 缺少冻结检索记录")
        baseline_records.append(
            _legal_cell(
                case=case,
                retrieved=retrieved,
                required=source_inputs._required_chunk_ids(case, repository),
                repository=repository,
                packager=baseline_packager,
                counter=counter,
                context_limit=CONTEXT_LIMITS[0],
            )
        )
    selected_ids, roles, matches = _select_legal_queries(baseline_records)

    records = []
    for context_limit in CONTEXT_LIMITS:
        packager = EvidencePackager(
            context_limit=context_limit,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            count_prompt_tokens=counter,
        )
        for query_id in selected_ids:
            case = case_by_id[query_id]
            required = source_inputs._required_chunk_ids(case, repository)
            cell = _legal_cell(
                case=case,
                retrieved=retrieval_by_id[query_id],
                required=required,
                repository=repository,
                packager=packager,
                counter=counter,
                context_limit=context_limit,
                selection_role=roles[query_id],
                matched_query_id=matches[query_id],
            )
            records.append(cell)

    if len(records) != EXPECTED_CELLS:
        raise ValueError(f"上下文阶梯单元数不闭合: {len(records)}")
    for context_limit in CONTEXT_LIMITS:
        current = [
            record for record in records if record["context_limit"] == context_limit
        ]
        if [record["query_id"] for record in current] != selected_ids:
            raise ValueError(f"{context_limit} 档评估题身份或顺序漂移")
    return records


def publish_context_sweep_inputs(
    *,
    eval_set: str | Path,
    retrieval_manifest: str | Path,
    retrieval_records: str | Path,
    article_index: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    manifest, manifest_sha256 = source_inputs._load_verified_json(
        retrieval_manifest, "冻结检索 manifest"
    )
    if manifest.get("pipeline") != EXPECTED_RETRIEVAL_PIPELINE:
        raise ValueError("冻结检索 pipeline 身份不匹配")
    packaging = manifest.get("packaging")
    if (
        not isinstance(packaging, dict)
        or packaging.get("context_limit") != 768
        or packaging.get("max_output_tokens") != MAX_OUTPUT_TOKENS
    ):
        raise ValueError("冻结检索 manifest 的原始构包预算无效")
    retrieval_path = Path(retrieval_records).resolve()
    retrieval_values = source_inputs._load_jsonl(
        retrieval_path, "冻结检索 records"
    )
    retrieval_identity = source_inputs._identity(
        retrieval_path, records=len(retrieval_values)
    )
    bound_records = manifest.get("outputs", {}).get("records")
    if (
        not isinstance(bound_records, dict)
        or bound_records.get("sha256") != retrieval_identity["sha256"]
        or bound_records.get("records") != retrieval_identity["records"]
    ):
        raise ValueError("冻结检索 records 与 manifest 身份不一致")

    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            Path(tokenizer_path).resolve(), use_fast=True, local_files_only=True
        )
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("无法加载固定 Tokenizer") from error
    tokenizer_identity = _tokenizer_identity(tokenizer, tokenizer_path)
    if _actual_tokenizer_identity(tokenizer_identity) != _manifest_tokenizer_identity(
        manifest.get("inputs", {}).get("tokenizer")
    ):
        raise ValueError("当前 Tokenizer 与冻结检索 manifest 身份不一致")

    eval_values = source_inputs._load_jsonl(eval_set, "固定评估集")
    repository = ArticleRepository.from_jsonl(article_index)
    counter = AnswerPromptTokenCounter(tokenizer)
    records = build_context_sweep_records(
        eval_values, retrieval_values, repository, counter
    )

    root = Path(output_dir).resolve()
    cases_path = root / "cases.jsonl"
    cases_text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    cases_sha256 = source_inputs._write_immutable(cases_path, cases_text)
    baseline = [
        record
        for record in records
        if record["context_limit"] == CONTEXT_LIMITS[0]
    ]
    baseline_by_id = {record["query_id"]: record for record in baseline}
    matched_pairs = []
    for record in baseline:
        if record["selection_role"] != "rescuable_packaging":
            continue
        control = baseline_by_id[record["matched_query_id"]]
        matched_pairs.append(
            {
                "rescuable_query_id": record["query_id"],
                "control_query_id": control["query_id"],
                "rescuable_features": {
                    "required_items": len(record["required_chunk_ids"]),
                    "evidence_items": len(record["visible_chunk_ids"]),
                    "prompt_tokens": record["retrieval_attribution"][
                        "prompt_tokens"
                    ],
                },
                "control_features": {
                    "required_items": len(control["required_chunk_ids"]),
                    "evidence_items": len(control["visible_chunk_ids"]),
                    "prompt_tokens": control["retrieval_attribution"][
                        "prompt_tokens"
                    ],
                },
            }
        )
    context_counts = {}
    for context_limit in CONTEXT_LIMITS:
        current = [
            record for record in records if record["context_limit"] == context_limit
        ]
        context_counts[str(context_limit)] = {
            "records": len(current),
            "ready": sum(record["status"] == "ready" for record in current),
            "overbudget": sum(
                record["status"] == "overbudget" for record in current
            ),
            "packaged_complete": sum(
                record["status"] == "ready"
                and record["retrieval_attribution"]["packaged_complete"] is True
                for record in current
            ),
            "selection_roles": {
                role: sum(record["selection_role"] == role for record in current)
                for role in SELECTION_COUNTS
            },
        }
    payload = {
        "schema_version": "1.0",
        "pipeline": PIPELINE,
        "inputs": {
            "eval_set": source_inputs._identity(eval_set),
            "retrieval_manifest": {
                **source_inputs._identity(retrieval_manifest),
                "verified_sha256": manifest_sha256,
                "pipeline": EXPECTED_RETRIEVAL_PIPELINE,
            },
            "retrieval_records": retrieval_identity,
            "article_index": source_inputs._identity(article_index),
            "tokenizer": tokenizer_identity,
        },
        "policy": {
            "context_limits": list(CONTEXT_LIMITS),
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "max_evidence_items": 5,
            "repackage_from_frozen_reranked_top5": True,
            "truncate_evidence": False,
            "inference_rope_scaling": False,
            "selection_uses_model_outputs": False,
        },
        "selection": {
            "baseline_context_limit": CONTEXT_LIMITS[0],
            "source_queries": sum(SOURCE_QUERY_COUNTS.values()),
            "source_legal_queries": SOURCE_QUERY_COUNTS["legal_query"],
            "counts": dict(SELECTION_COUNTS),
            "query_ids": {
                role: [
                    record["query_id"]
                    for record in baseline
                    if record["selection_role"] == role
                ]
                for role in SELECTION_COUNTS
            },
            "matching_priority": [
                "required_items_absolute_difference",
                "evidence_items_absolute_difference",
                "prompt_tokens_absolute_difference",
                "query_id",
            ],
            "matched_pairs": matched_pairs,
        },
        "records": {
            "queries": EXPECTED_QUERIES,
            "legal_query": EXPECTED_QUERIES,
            "exact_lookup": 0,
            "selection_roles": dict(SELECTION_COUNTS),
            "context_cells": EXPECTED_CELLS,
            "by_context": context_counts,
        },
        "output": {
            "cases": {
                "path": str(cases_path),
                "bytes": cases_path.stat().st_size,
                "sha256": cases_sha256,
                "records": len(records),
            }
        },
        "model_input_fields": ["query", "evidence"],
        "hidden_from_model": [
            "query_id",
            "query_type",
            "selection_role",
            "matched_query_id",
            "context_limit",
            "status",
            "failure_reason",
            "visible_chunk_ids",
            "required_chunk_ids",
            "hard_negative_chunk_ids",
            "retrieval_attribution",
        ],
        "complete": True,
    }
    manifest_path = root / "manifest.json"
    source_inputs._write_immutable(
        manifest_path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="发布 RAG-SFT v2 固定45题的四档上下文外推评估输入"
    )
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--retrieval-manifest", required=True)
    parser.add_argument("--retrieval-records", required=True)
    parser.add_argument("--article-index", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    report = publish_context_sweep_inputs(
        eval_set=args.eval_set,
        retrieval_manifest=args.retrieval_manifest,
        retrieval_records=args.retrieval_records,
        article_index=args.article_index,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
    )
    print(
        "RAG_SFT_V2_CONTEXT_SWEEP_INPUTS_OK "
        f"cells={report['records']['context_cells']}"
    )


if __name__ == "__main__":
    main()
