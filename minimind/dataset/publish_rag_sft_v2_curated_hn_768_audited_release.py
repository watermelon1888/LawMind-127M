"""发布经 768 长度审计后的非训练 RAG-SFT v2 curated HN 语义候选。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rag.knowledge import ArticleRepository

try:
    from . import audit_disc_law_sft as tokenizer_auditor
    from .build_rag_sft_v2_curated_hn_reuse_review import (
        _bound_path,
        _canonical_from_clean,
        _identity,
        _load_json,
        _load_jsonl,
        _payload_identity,
        _serialize_jsonl,
        _sha256,
        _verify_manifest,
    )
    from .rag_sft_v2_contract import validate_hn_variant
    from .rag_sft_v2_projection import (
        CONTEXT_LIMIT,
        MAX_OUTPUT_TOKENS,
        MAX_PROMPT_TOKENS,
        RagSftV2ProjectionError,
        audit_projected_rag_sft_v2_record,
        project_rag_sft_v2_record,
    )
except ImportError:  # 支持从项目根目录以模块方式运行。
    from minimind.dataset import audit_disc_law_sft as tokenizer_auditor
    from minimind.dataset.build_rag_sft_v2_curated_hn_reuse_review import (
        _bound_path,
        _canonical_from_clean,
        _identity,
        _load_json,
        _load_jsonl,
        _payload_identity,
        _serialize_jsonl,
        _sha256,
        _verify_manifest,
    )
    from minimind.dataset.rag_sft_v2_contract import validate_hn_variant
    from minimind.dataset.rag_sft_v2_projection import (
        CONTEXT_LIMIT,
        MAX_OUTPUT_TOKENS,
        MAX_PROMPT_TOKENS,
        RagSftV2ProjectionError,
        audit_projected_rag_sft_v2_record,
        project_rag_sft_v2_record,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEMANTIC_RELEASE = (
    PROJECT_ROOT
    / "minimind/dataset/RAG-SFT/releases/v2/"
    "rag-sft-v2-curated-hn-semantic-release-779-v1-20260814"
)
DEFAULT_AUDIT_ROOT = (
    PROJECT_ROOT
    / "minimind/dataset/RAG-SFT/reports/v2/"
    "rag-sft-v2-curated-hn-semantic-release-779-768-audit-v2-20260814"
)
DEFAULT_C6_REVIEW = (
    PROJECT_ROOT
    / "minimind/dataset/RAG-SFT/review/v2/"
    "curated-hn-manual-semantic-review-v6-20260814"
)
DEFAULT_ARTICLE_INDEX = PROJECT_ROOT / "rag/chunk/article_index.jsonl"
DEFAULT_TOKENIZER = PROJECT_ROOT / "minimind/model"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "minimind/dataset/RAG-SFT/releases/v2/"
    "rag-sft-v2-curated-hn-768-audited-release-776-v1-20260814"
)


class CuratedHn768PublishError(RuntimeError):
    """768 审计 release 输入、身份或准入状态不闭合。"""


def _json_sha256(value: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _load_semantic_release(root: Path) -> tuple[dict[str, Any], str, Path, list[dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    manifest_sha256 = _verify_manifest(manifest_path)
    manifest = _load_json(manifest_path, "语义冻结 release manifest")
    if (
        manifest.get("pipeline") != "rag_sft_v2_curated_hn_semantic_release"
        or manifest.get("release_status") != "semantic_identity_freeze_only"
        or manifest.get("records", {}).get("total") != 779
        or manifest.get("records", {}).get("clean") != 549
        or manifest.get("records", {}).get("hard_negative") != 230
        or manifest.get("policy", {}).get("training_ready") is not False
        or manifest.get("complete") is not True
    ):
        raise CuratedHn768PublishError("语义冻结 release 状态或计数无效")
    candidate_path = _bound_path(
        manifest.get("output", {}).get("semantic_freeze_candidate"), "语义冻结 candidate"
    )
    rows = _load_jsonl(candidate_path, "语义冻结 candidate")
    if len(rows) != 779:
        raise CuratedHn768PublishError("语义冻结 candidate 数量不闭合")
    return manifest, manifest_sha256, candidate_path, rows


def _load_audit(
    root: Path, semantic_candidate_path: Path
) -> tuple[dict[str, Any], str, dict[str, dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    manifest_sha256 = _verify_manifest(manifest_path)
    manifest = _load_json(manifest_path, "768 审计 manifest")
    if (
        manifest.get("pipeline") != "rag_sft_v2_curated_hn_768_audit"
        or manifest.get("policy", {}).get("training_ready") is not False
        or manifest.get("records", {}).get("total") != 779
        or manifest.get("complete") is not True
    ):
        raise CuratedHn768PublishError("768 审计状态或计数无效")
    bound_input = manifest.get("inputs", {}).get("semantic_freeze_candidate")
    actual_input = _identity(semantic_candidate_path, records=779)
    if not isinstance(bound_input, dict) or any(
        bound_input.get(key) != actual_input[key] for key in actual_input
    ):
        raise CuratedHn768PublishError("768 审计未绑定当前语义冻结 candidate")
    ledger_path = _bound_path(
        manifest.get("output", {}).get("length_audit_ledger"), "768 长度审计 ledger"
    )
    rows = _load_jsonl(ledger_path, "768 长度审计 ledger")
    indexed: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(rows, start=1):
        record_id = row.get("id")
        if not isinstance(record_id, str) or record_id in indexed:
            raise CuratedHn768PublishError(f"768 长度审计 ledger 第 {number} 条身份无效")
        if row.get("decision") not in {"keep", "exclude"}:
            raise CuratedHn768PublishError(f"768 长度审计 ledger 第 {number} 条决策无效")
        indexed[record_id] = row
    if len(indexed) != 779:
        raise CuratedHn768PublishError("768 长度审计 ledger 覆盖不闭合")
    return manifest, manifest_sha256, indexed


def _load_c6_variants(root: Path) -> tuple[dict[str, Any], str, Path, list[dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    manifest_sha256 = _verify_manifest(manifest_path)
    manifest = _load_json(manifest_path, "C6 人工双审 manifest")
    if (
        manifest.get("pipeline") != "rag_sft_v2_curated_hn_manual_semantic_review"
        or manifest.get("records", {}).get("reviewed") != 7
        or manifest.get("records", {}).get("approved") != 7
        or manifest.get("records", {}).get("legal_support", {}).get("none") != 7
        or manifest.get("policy", {}).get("formal_hn_materialized") is not False
        or manifest.get("policy", {}).get("training_ready") is not False
        or manifest.get("complete") is not True
    ):
        raise CuratedHn768PublishError("C6 人工双审资产状态或计数无效")
    variants_path = _bound_path(
        manifest.get("output", {}).get("curated-hn-variants.jsonl"), "C6 HN variants"
    )
    variants = _load_jsonl(variants_path, "C6 HN variants")
    if len(variants) != 7:
        raise CuratedHn768PublishError("C6 HN variants 数量不闭合")
    return manifest, manifest_sha256, variants_path, variants


def _tokenizer_identity(tokenizer: Any, tokenizer_path: Path) -> dict[str, Any]:
    report = tokenizer_auditor._tokenizer_report(tokenizer, tokenizer_path)
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str):
        raise CuratedHn768PublishError("Tokenizer 缺少 chat template")
    report["chat_template_sha256"] = hashlib.sha256(template.encode("utf-8")).hexdigest()
    return report


def _audit_added_record(record: object, tokenizer: Any) -> tuple[dict[str, object], dict[str, object]]:
    """审计新增 HN 的长度与 labels，并返回可发布记录和细粒度 ledger。"""

    if not hasattr(record, "record_id") or not hasattr(record, "training_conversations"):
        raise CuratedHn768PublishError("新增 HN 投影对象无效")
    try:
        audit = audit_projected_rag_sft_v2_record(record, tokenizer)
    except RagSftV2ProjectionError as error:
        return {}, {
            "id": record.record_id,
            "query_id": record.query_id,
            "decision": "exclude",
            "reason": "context_or_label_audit_failed",
            "detail": str(error),
        }
    active = [token for token in audit.labels if token != -100]
    labels_closed = (
        len(audit.input_ids) == CONTEXT_LIMIT
        and len(audit.labels) == CONTEXT_LIMIT
        and active == list(audit.input_ids[audit.prompt_tokens : audit.total_tokens])
        and all(token == -100 for token in audit.labels[: audit.prompt_tokens])
        and all(token == -100 for token in audit.labels[audit.total_tokens :])
    )
    if not labels_closed:
        return {}, {
            "id": record.record_id,
            "query_id": record.query_id,
            "decision": "exclude",
            "reason": "assistant_only_labels_not_closed",
            "detail": "新增 HN 的 assistant-only labels 与 EOS 边界不闭合",
        }
    candidate = {
        "id": record.record_id,
        "query_id": record.query_id,
        "variant": record.variant,
        "query_original": record.conversations[1]["content"],
        "visible_chunk_ids": list(record.visible_chunk_ids),
        "required_chunk_ids": list(record.required_chunk_ids),
        "citations": list(record.citations),
        "conversations": record.training_conversations(),
    }
    candidate["query_original"] = json.loads(candidate["query_original"])["query"]
    return candidate, {
        "id": record.record_id,
        "query_id": record.query_id,
        "decision": "keep",
        "reason": "within_768_and_labels_closed",
        "prompt_tokens": audit.prompt_tokens,
        "assistant_tokens": audit.assistant_label_tokens,
        "total_tokens": audit.total_tokens,
        "assistant_only_labels_correct": True,
    }


def publish(
    *,
    semantic_release: Path,
    audit_root: Path,
    c6_review: Path,
    article_index: Path,
    tokenizer_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """只保留通过审计的候选，并有条件加入 C6 的已裁决 HN。"""

    semantic_release = semantic_release.resolve()
    audit_root = audit_root.resolve()
    c6_review = c6_review.resolve()
    article_index = article_index.resolve()
    tokenizer_path = tokenizer_path.resolve()
    output_dir = output_dir.resolve()
    partial = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial.exists():
        raise CuratedHn768PublishError(f"输出目录或临时目录已存在: {output_dir}")
    if not article_index.is_file() or not tokenizer_path.is_dir():
        raise CuratedHn768PublishError("法条索引或 Tokenizer 路径不存在")

    semantic_manifest, semantic_sha256, _, semantic_rows = _load_semantic_release(semantic_release)
    semantic_candidate_path = _bound_path(
        semantic_manifest.get("output", {}).get("semantic_freeze_candidate"), "语义冻结 candidate"
    )
    audit_manifest, audit_sha256, audit_by_id = _load_audit(audit_root, semantic_candidate_path)
    c6_manifest, c6_sha256, c6_variants_path, c6_variants = _load_c6_variants(c6_review)
    tokenizer = tokenizer_auditor.load_tokenizer(tokenizer_path)
    tokenizer_report = _tokenizer_identity(tokenizer, tokenizer_path)
    expected_tokenizer = audit_manifest.get("tokenizer")
    if tokenizer_report != expected_tokenizer:
        raise CuratedHn768PublishError("当前 Tokenizer 与 768 审计的冻结身份不一致")

    base_ids = {row.get("id") for row in semantic_rows}
    if len(base_ids) != len(semantic_rows) or not all(isinstance(value, str) for value in base_ids):
        raise CuratedHn768PublishError("语义冻结 candidate record id 不唯一")
    if set(audit_by_id) != base_ids:
        raise CuratedHn768PublishError("768 审计 ledger 与语义冻结 candidate 身份不一致")
    clean_by_query: dict[str, dict[str, Any]] = {}
    base_hn_queries: set[str] = set()
    clean_excluded_queries: set[str] = set()
    for row in semantic_rows:
        query_id = row.get("query_id")
        if not isinstance(query_id, str):
            raise CuratedHn768PublishError("语义冻结 candidate query_id 无效")
        if row.get("variant") == "clean":
            if query_id in clean_by_query:
                raise CuratedHn768PublishError("语义冻结 candidate clean query 重复")
            clean_by_query[query_id] = row
            if audit_by_id[row["id"]].get("decision") != "keep":
                clean_excluded_queries.add(query_id)
        elif row.get("variant") == "hard_negative":
            if query_id in base_hn_queries:
                raise CuratedHn768PublishError("语义冻结 candidate HN query 重复")
            base_hn_queries.add(query_id)
        else:
            raise CuratedHn768PublishError("语义冻结 candidate variant 无效")
    if len(clean_by_query) != 549 or len(base_hn_queries) != 230:
        raise CuratedHn768PublishError("语义冻结 candidate clean/HN query 数不闭合")

    kept: list[dict[str, Any]] = []
    identity_ledger: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    source_counts: Counter[str] = Counter()
    retained_hn_queries: set[str] = set()
    for row in semantic_rows:
        audit_row = audit_by_id[row["id"]]
        query_id = row["query_id"]
        variant = row["variant"]
        source = audit_row.get("source")
        if source not in {"oracle_clean", "retrieved", "curated"}:
            raise CuratedHn768PublishError("768 审计 ledger 来源无效")
        decision = audit_row.get("decision")
        exclude_reason: str | None = None
        if query_id in clean_excluded_queries:
            exclude_reason = "clean_context_overlimit_group_exclusion"
        elif decision != "keep":
            exclude_reason = str(audit_row.get("reason"))
        if exclude_reason is not None:
            exclusions.append(
                {
                    "id": row["id"],
                    "query_id": query_id,
                    "variant": variant,
                    "source": source,
                    "reason": exclude_reason,
                    "audit_ledger_id": row["id"],
                }
            )
            continue
        kept.append(row)
        source_counts[source] += 1
        if variant == "hard_negative":
            retained_hn_queries.add(query_id)
        identity_ledger.append(
            {
                "id": row["id"],
                "query_id": query_id,
                "variant": variant,
                "source": source,
                "origin": "semantic_freeze_779_v1",
                "record_sha256": _json_sha256(row),
                "length_audit_ledger_id": row["id"],
            }
        )
    if clean_excluded_queries:
        raise CuratedHn768PublishError("当前输入存在 clean 超限，本发布版本必须另行处理 clean 配对排除")
    if len(kept) != 769 or len(retained_hn_queries) != 220:
        raise CuratedHn768PublishError("通过 768 审计的基线候选计数不符合预期")

    repository = ArticleRepository.from_jsonl(article_index)
    supplemental_audit: list[dict[str, object]] = []
    accepted_c6_variants: list[dict[str, Any]] = []
    accepted_c6_queries: set[str] = set()
    for variant in c6_variants:
        query_id = variant.get("query_id")
        if not isinstance(query_id, str) or query_id not in clean_by_query:
            raise CuratedHn768PublishError("C6 HN 缺少对应 clean")
        if query_id in retained_hn_queries or query_id in accepted_c6_queries:
            raise CuratedHn768PublishError("C6 HN 与已保留 HN query 冲突")
        canonical = _canonical_from_clean(clean_by_query[query_id])
        try:
            approved = validate_hn_variant(variant, canonical)
            article_by_chunk_id = {
                chunk_id: repository.get_by_chunk_id(chunk_id)
                for chunk_id in approved["visible_chunk_ids"]
            }
            projected = project_rag_sft_v2_record(
                canonical, article_by_chunk_id, hn_variant=variant
            )
        except Exception as error:
            raise CuratedHn768PublishError(f"C6 HN 语义或法条身份无效: {query_id}") from error
        candidate, audit_row = _audit_added_record(projected, tokenizer)
        supplemental_audit.append(audit_row)
        if audit_row["decision"] != "keep":
            exclusions.append(
                {
                    "id": projected.record_id,
                    "query_id": query_id,
                    "variant": "hard_negative",
                    "source": "curated",
                    "reason": audit_row["reason"],
                    "audit_ledger_id": projected.record_id,
                }
            )
            continue
        kept.append(candidate)
        accepted_c6_variants.append(variant)
        accepted_c6_queries.add(query_id)
        retained_hn_queries.add(query_id)
        source_counts["curated"] += 1
        identity_ledger.append(
            {
                "id": candidate["id"],
                "query_id": query_id,
                "variant": "hard_negative",
                "source": "curated",
                "origin": "curated_manual_review_v6",
                "record_sha256": _json_sha256(candidate),
                "length_audit_ledger_id": candidate["id"],
            }
        )

    records = Counter(row["variant"] for row in kept)
    if records["clean"] != 549 or records["hard_negative"] != len(retained_hn_queries):
        raise CuratedHn768PublishError("最终 release clean/HN 计数不闭合")
    if len({row["id"] for row in kept}) != len(kept):
        raise CuratedHn768PublishError("最终 release record id 重复")
    if len({row["query_id"] for row in kept if row["variant"] == "clean"}) != 549:
        raise CuratedHn768PublishError("最终 release clean query 身份不闭合")
    if not retained_hn_queries.issubset(clean_by_query):
        raise CuratedHn768PublishError("最终 release HN 缺少 clean 配对")

    candidate_payload = _serialize_jsonl(kept)
    identity_payload = _serialize_jsonl(identity_ledger)
    exclusions_payload = _serialize_jsonl(exclusions)
    supplemental_payload = _serialize_jsonl(supplemental_audit)
    variants_payload = _serialize_jsonl(accepted_c6_variants)
    output = {
        "semantic_freeze_candidate": _payload_identity(
            output_dir / "semantic-freeze-candidate.jsonl", candidate_payload, len(kept)
        ),
        "identity_ledger": _payload_identity(
            output_dir / "identity-ledger.jsonl", identity_payload, len(identity_ledger)
        ),
        "context_exclusions": _payload_identity(
            output_dir / "context-exclusions.jsonl", exclusions_payload, len(exclusions)
        ),
        "c6_supplemental_length_audit": _payload_identity(
            output_dir / "c6-supplemental-length-audit.jsonl",
            supplemental_payload,
            len(supplemental_audit),
        ),
        "accepted_c6_curated_hn_variants": _payload_identity(
            output_dir / "accepted-c6-curated-hn-variants.jsonl",
            variants_payload,
            len(accepted_c6_variants),
        ),
    }
    manifest: dict[str, object] = {
        "pipeline": "rag_sft_v2_curated_hn_768_audited_release",
        "release_status": "length_audited_semantic_candidate_only",
        "inputs": {
            "semantic_freeze_manifest": {
                **_identity(semantic_release / "manifest.json"),
                "manifest_sha256": semantic_sha256,
            },
            "semantic_freeze_candidate": _identity(semantic_candidate_path, records=779),
            "length_audit_manifest": {
                **_identity(audit_root / "manifest.json"),
                "manifest_sha256": audit_sha256,
            },
            "c6_review_manifest": {
                **_identity(c6_review / "manifest.json"),
                "manifest_sha256": c6_sha256,
            },
            "c6_review_variants": _identity(c6_variants_path, records=7),
            "article_index": _identity(article_index),
        },
        "tokenizer": tokenizer_report,
        "policy": {
            "context_limit": CONTEXT_LIMIT,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "truncation": "forbidden",
            "source_assets_modified": False,
            "formal_hn_materialized": False,
            "formal_training_candidate_emitted": False,
            "context_length_audit": "completed",
            "training_ready": False,
        },
        "records": {
            "total": len(kept),
            "clean": records["clean"],
            "retrieved_hn": source_counts["retrieved"],
            "curated_hn": source_counts["curated"],
            "hard_negative": records["hard_negative"],
            "unique_queries": 549,
            "paired_hn_queries": len(retained_hn_queries),
            "context_excluded_clean": 0,
            "context_excluded_hn": sum(
                item["variant"] == "hard_negative" for item in exclusions
            ),
            "c6_reviewed": len(c6_variants),
            "c6_accepted_after_768_audit": len(accepted_c6_variants),
            "c6_excluded_after_768_audit": len(c6_variants) - len(accepted_c6_variants),
        },
        "validation": {
            "strict_utf8_jsonl": True,
            "input_hashes_verified": True,
            "base_779_length_ledger_covered_once": True,
            "all_retained_base_records_passed_768_and_labels_audit": True,
            "all_c6_records_reaudited_with_frozen_tokenizer": True,
            "clean_overlimit_pair_policy_applied": True,
            "hn_overlimit_excluded_without_clean_removal": True,
            "one_hn_per_query": True,
            "all_hn_have_clean_pair": True,
            "quarantined_batch7_legacy_not_referenced": True,
        },
        "output": output,
        "complete": True,
    }
    readme = (
        "# RAG-SFT v2 curated HN 768 审计 release\n\n"
        "本目录只发布通过 768 token、assistant-only labels 与身份审计的语义候选。"
        "10 条超限 HN 已排除；第六轮已双审 HN 仅在再次通过长度审计后补入。"
        "未物化正式 HN，`formal_hn_materialized=false`，`training_ready=false`。\n"
    )
    payloads = {
        "semantic-freeze-candidate.jsonl": candidate_payload,
        "identity-ledger.jsonl": identity_payload,
        "context-exclusions.jsonl": exclusions_payload,
        "c6-supplemental-length-audit.jsonl": supplemental_payload,
        "accepted-c6-curated-hn-variants.jsonl": variants_payload,
        "manifest.json": json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        "README.md": readme,
    }
    try:
        partial.mkdir(parents=True, exist_ok=False)
        for filename, payload in payloads.items():
            (partial / filename).write_text(payload, encoding="utf-8", newline="\n")
        (partial / "manifest.sha256").write_text(
            "".join(f"{_sha256(partial / filename)}  {filename}\n" for filename in payloads),
            encoding="ascii",
            newline="\n",
        )
        partial.replace(output_dir)
    except OSError as error:
        raise CuratedHn768PublishError("无法原子发布 768 审计 release；已保留临时目录") from error
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-release", type=Path, default=DEFAULT_SEMANTIC_RELEASE)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--c6-review", type=Path, default=DEFAULT_C6_REVIEW)
    parser.add_argument("--article-index", type=Path, default=DEFAULT_ARTICLE_INDEX)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        manifest = publish(
            semantic_release=args.semantic_release,
            audit_root=args.audit_root,
            c6_review=args.c6_review,
            article_index=args.article_index,
            tokenizer_path=args.tokenizer,
            output_dir=args.output_dir,
        )
    except (CuratedHn768PublishError, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"RAG_SFT_V2_CURATED_HN_768_RELEASE_FAILED: {error}") from error
    print(
        "RAG_SFT_V2_CURATED_HN_768_RELEASE_OK "
        f"total={manifest['records']['total']} hn={manifest['records']['hard_negative']} "
        "training_ready=false"
    )


if __name__ == "__main__":
    main()
