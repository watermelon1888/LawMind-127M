"""使用真实运行时模板审计 canonical RAG-SFT 的证据预算。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rag.answering import (
    AnswerPromptTokenCounter,
    EvidencePackage,
    EvidencePackager,
    EvidencePackagingError,
    RAG_MAX_OUTPUT_TOKENS,
)
from rag.core import Evidence
from rag.knowledge import LegalArticle

try:
    from . import audit_disc_law_sft as base_auditor
    from . import audit_rag_sft_chat_lengths as rag_length_auditor
    from . import audit_sft_chat_lengths as length_auditor
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import audit_disc_law_sft as base_auditor
    from dataset import audit_rag_sft_chat_lengths as rag_length_auditor
    from dataset import audit_sft_chat_lengths as length_auditor


RAG_SFT_ROOT = Path(__file__).resolve().parent / "RAG-SFT"
DEFAULT_RAG_MANIFEST = (
    RAG_SFT_ROOT / "manifests" / "rag-sft-canonical-v1-candidate.json"
)
DEFAULT_TOKENIZER_PATH = base_auditor.DEFAULT_TOKENIZER_PATH
DEFAULT_OUTPUT_DIR = (
    RAG_SFT_ROOT / "reports" / "rag-sft-canonical-v1-runtime-budget-audit-160"
)
DEFAULT_CONTEXT_LIMIT = 768
DEFAULT_BUDGETS = (128, RAG_MAX_OUTPUT_TOKENS, 192)

REPORT_FILENAME = "rag-sft-runtime-budget-audit-160.json"
LOCATOR_FILENAME = "rag-sft-runtime-budget-locators-160.jsonl"
HASH_FILENAME = "rag-sft-runtime-budget-audit-160.sha256"


class RagRuntimeBudgetAuditError(RuntimeError):
    """RAG-SFT 运行时证据预算无法可靠审计。"""


def _distribution(values: list[int]) -> dict[str, int | float | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    ordered = sorted(values)

    def percentile(percent: int) -> int:
        index = max(0, (len(ordered) * percent + 99) // 100 - 1)
        return ordered[index]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "p99": percentile(99),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 4),
    }


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": base_auditor.sha256_file(path),
    }


def _record_package(
    record: dict[str, object], authoring: dict[str, object]
) -> tuple[EvidencePackage, tuple[LegalArticle, ...]]:
    conversations = record["conversations"]
    user = rag_length_auditor._load_compact_json(
        conversations[1]["content"], "RAG candidate user"
    )
    evidence_items = user["evidence"]
    visible_chunk_ids = authoring["visible_chunk_ids"]
    evidence = tuple(
        Evidence(
            law_name=item["law_name"],
            article_no=item["article_no"],
            content=item["excerpts"][0],
        )
        for item in evidence_items
    )
    package = EvidencePackage(query=user["query"], evidence=evidence)
    articles = tuple(
        LegalArticle(
            chunk_id=chunk_id,
            law_name=item.law_name,
            article_no=item.article_no,
            content=item.content,
        )
        for chunk_id, item in zip(visible_chunk_ids, evidence)
    )
    return package, articles


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    records = len(rows)
    counts = Counter()
    for row in rows:
        counts["first_evidence_fit"] += bool(row["first_evidence_fit"])
        counts["full_visible_package_fit"] += bool(row["full_visible_package_fit"])
        counts["all_required_gt_selected"] += bool(row["all_required_gt_selected"])
        counts["visible_evidence_dropped"] += bool(row["visible_evidence_dropped"])
    return {
        "counts": {
            "records": records,
            **{key: counts[key] for key in sorted(counts)},
        },
        "rates": {
            key: round(counts[key] / records, 8) if records else None
            for key in sorted(counts)
        },
        "selected_evidence_count": _distribution(
            [int(row["selected_evidence_count"]) for row in rows]
        ),
        "full_prompt_tokens": _distribution(
            [int(row["full_prompt_tokens"]) for row in rows]
        ),
        "full_package_margin_tokens": _distribution(
            [int(row["full_package_margin_tokens"]) for row in rows]
        ),
    }


def _write_outputs(
    output_dir: Path,
    report: dict[str, object],
    locators: list[dict[str, object]],
) -> None:
    if output_dir.exists():
        raise RagRuntimeBudgetAuditError(f"审计输出目录必须是新目录: {output_dir}")
    report_payload = json.dumps(
        report, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    locator_payload = "".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for item in locators
    )
    output_dir.mkdir(parents=True)
    report_path = output_dir / REPORT_FILENAME
    locator_path = output_dir / LOCATOR_FILENAME
    hash_path = output_dir / HASH_FILENAME
    pending = [
        (report_path.with_suffix(report_path.suffix + ".partial"), report_path, report_payload),
        (
            locator_path.with_suffix(locator_path.suffix + ".partial"),
            locator_path,
            locator_payload,
        ),
    ]
    published: list[Path] = []
    try:
        for partial, _, payload in pending:
            partial.write_text(payload, encoding="utf-8", newline="\n")
        hash_payload = (
            f"{base_auditor.sha256_file(pending[0][0])}  {REPORT_FILENAME}\n"
            f"{base_auditor.sha256_file(pending[1][0])}  {LOCATOR_FILENAME}\n"
        )
        hash_pending = hash_path.with_suffix(hash_path.suffix + ".partial")
        hash_pending.write_text(hash_payload, encoding="utf-8", newline="\n")
        for partial, final, _ in pending:
            partial.replace(final)
            published.append(final)
        hash_pending.replace(hash_path)
        published.append(hash_path)
    except (OSError, UnicodeError, ValueError) as error:
        for path in [
            *(item[0] for item in pending),
            hash_path.with_suffix(hash_path.suffix + ".partial"),
            *reversed(published),
        ]:
            path.unlink(missing_ok=True)
        try:
            output_dir.rmdir()
        except OSError:
            pass
        raise RagRuntimeBudgetAuditError("无法发布运行时预算审计产物") from error


def audit_rag_sft_runtime_budget(
    rag_manifest: Path,
    tokenizer_path: Path,
    output_dir: Path,
    *,
    context_limit: int = DEFAULT_CONTEXT_LIMIT,
    budgets: tuple[int, ...] = DEFAULT_BUDGETS,
    tokenizer: Any | None = None,
) -> dict[str, object]:
    """比较多个输出预算下 canonical 证据最大有序前缀的覆盖情况。"""

    if type(context_limit) is not int or context_limit <= 0:
        raise RagRuntimeBudgetAuditError("context_limit 必须是正整数")
    if (
        not isinstance(budgets, tuple)
        or RAG_MAX_OUTPUT_TOKENS not in budgets
        or len(budgets) != len(set(budgets))
        or any(type(value) is not int or not 0 < value < context_limit for value in budgets)
    ):
        raise RagRuntimeBudgetAuditError("budgets 必须是包含当前 160 的不重复正整数 tuple")
    budgets = tuple(sorted(budgets))
    rag_manifest = Path(rag_manifest).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RagRuntimeBudgetAuditError(f"审计输出目录必须是新目录: {output_dir}")

    try:
        (
            manifest,
            manifest_identity,
            candidate_path,
            candidate_identity,
            authoring_path,
            authoring_identity,
        ) = rag_length_auditor._verify_manifest(rag_manifest)
        authoring_by_id = rag_length_auditor._load_authoring(authoring_path)
        records = rag_length_auditor._load_candidate(
            candidate_path,
            manifest["records"]["candidate"],
            authoring_by_id,
        )
    except rag_length_auditor.RagChatLengthAuditError as error:
        raise RagRuntimeBudgetAuditError("RAG canonical 输入身份或结构无效") from error

    tokenizer = tokenizer or base_auditor.load_tokenizer(tokenizer_path)
    try:
        tokenizer_report = length_auditor._tokenizer_identity(tokenizer, tokenizer_path)
        counter = AnswerPromptTokenCounter(tokenizer)
    except Exception as error:
        raise RagRuntimeBudgetAuditError("运行时 tokenizer 身份或模板无效") from error

    rows_by_budget: dict[int, list[dict[str, object]]] = {
        budget: [] for budget in budgets
    }
    record_results: dict[str, dict[int, dict[str, object]]] = {}
    metadata_by_id: dict[str, dict[str, object]] = {}
    for record in records:
        record_id = record["id"]
        authoring = authoring_by_id[record_id]
        package, articles = _record_package(record, authoring)
        try:
            full_prompt_tokens = counter(package)
        except Exception as error:
            raise RagRuntimeBudgetAuditError(
                f"{record_id} 完整运行时 prompt 无法计数"
            ) from error
        visible_chunk_ids = tuple(authoring["visible_chunk_ids"])
        required_chunk_ids = set(authoring["required_chunk_ids"])
        metadata_by_id[record_id] = {
            "evidence_source": authoring["evidence_source"],
            "behavior": "refusal" if authoring["target"]["refuse"] else "answer",
            "visible_evidence_count": len(visible_chunk_ids),
            "required_gt_count": len(required_chunk_ids),
        }
        record_results[record_id] = {}

        for budget in budgets:
            packager = EvidencePackager(
                context_limit=context_limit,
                max_output_tokens=budget,
                count_prompt_tokens=counter,
            )
            try:
                selected_package, selected_prompt_tokens = packager.build(
                    package.query, articles
                )
                selected_count = len(selected_package.evidence)
            except EvidencePackagingError:
                selected_prompt_tokens = None
                selected_count = 0
            selected_chunk_ids = visible_chunk_ids[:selected_count]
            full_fit = selected_count == len(visible_chunk_ids)
            if full_fit != (full_prompt_tokens + budget <= context_limit):
                raise RagRuntimeBudgetAuditError(
                    f"{record_id} 的完整包适配判定不闭合: {budget}"
                )
            row = {
                "id": record_id,
                **metadata_by_id[record_id],
                "budget": budget,
                "full_prompt_tokens": full_prompt_tokens,
                "full_package_margin_tokens": context_limit
                - budget
                - full_prompt_tokens,
                "selected_prompt_tokens": selected_prompt_tokens,
                "selected_evidence_count": selected_count,
                "first_evidence_fit": selected_count > 0,
                "full_visible_package_fit": full_fit,
                "visible_evidence_dropped": selected_count < len(visible_chunk_ids),
                "all_required_gt_selected": required_chunk_ids.issubset(
                    selected_chunk_ids
                ),
                "selected_chunk_ids": list(selected_chunk_ids),
                "omitted_visible_chunk_ids": list(visible_chunk_ids[selected_count:]),
            }
            rows_by_budget[budget].append(row)
            record_results[record_id][budget] = row

    for record_id, results in record_results.items():
        selected_counts = [int(results[budget]["selected_evidence_count"]) for budget in budgets]
        if selected_counts != sorted(selected_counts, reverse=True):
            raise RagRuntimeBudgetAuditError(
                f"{record_id} 的有序前缀未随输出预算单调收缩"
            )

    primary_rows = rows_by_budget[RAG_MAX_OUTPUT_TOKENS]
    locators: list[dict[str, object]] = []
    for record_id in sorted(record_results):
        results = record_results[record_id]
        counts = {
            str(budget): int(results[budget]["selected_evidence_count"])
            for budget in budgets
        }
        primary = results[RAG_MAX_OUTPUT_TOKENS]
        risks = []
        if not primary["first_evidence_fit"]:
            risks.append("primary_first_evidence_over_budget")
        elif not primary["full_visible_package_fit"]:
            risks.append("primary_drops_visible_evidence")
        if len(set(counts.values())) > 1:
            risks.append("budget_sensitive_prefix")
        if risks:
            locators.append(
                {
                    "id": record_id,
                    **metadata_by_id[record_id],
                    "full_prompt_tokens": primary["full_prompt_tokens"],
                    "selected_evidence_count_by_budget": counts,
                    "primary_selected_chunk_ids": primary["selected_chunk_ids"],
                    "primary_omitted_visible_chunk_ids": primary[
                        "omitted_visible_chunk_ids"
                    ],
                    "risks": risks,
                }
            )

    budget_reports = {}
    for budget in budgets:
        rows = rows_by_budget[budget]
        by_source = {
            source: _summarize(
                [row for row in rows if row["evidence_source"] == source]
            )
            for source in sorted({str(row["evidence_source"]) for row in rows})
        }
        oracle_rows = [row for row in rows if row["evidence_source"] == "oracle"]
        budget_reports[str(budget)] = {
            "overall": _summarize(rows),
            "oracle": _summarize(oracle_rows),
            "by_evidence_source": by_source,
        }

    primary_counts = {
        row["id"]: int(row["selected_evidence_count"]) for row in primary_rows
    }
    comparisons = {}
    for budget in budgets:
        if budget == RAG_MAX_OUTPUT_TOKENS:
            continue
        comparison_counts = {
            row["id"]: int(row["selected_evidence_count"])
            for row in rows_by_budget[budget]
        }
        comparisons[str(budget)] = {
            "selects_more_than_160": sum(
                comparison_counts[item] > primary_counts[item]
                for item in primary_counts
            ),
            "selects_same_as_160": sum(
                comparison_counts[item] == primary_counts[item]
                for item in primary_counts
            ),
            "selects_fewer_than_160": sum(
                comparison_counts[item] < primary_counts[item]
                for item in primary_counts
            ),
        }

    report: dict[str, object] = {
        "pipeline": "rag_sft_runtime_budget_audit",
        "scope": {
            "context_limit": context_limit,
            "primary_max_output_tokens": RAG_MAX_OUTPUT_TOKENS,
            "comparison_budgets": [
                value for value in budgets if value != RAG_MAX_OUTPUT_TOKENS
            ],
            "candidate_records": len(records),
            "evidence_policy": "complete_article_maximum_ordered_prefix_no_selector",
            "audit_only": True,
            "input_files_modified": False,
        },
        "input": {
            "rag_manifest": manifest_identity,
            "candidate": candidate_identity,
            "authoring": authoring_identity,
        },
        "tokenizer": tokenizer_report,
        "template": {
            "render": "apply_chat_template_tokenize_false_add_generation_prompt_true",
            "open_thinking": False,
            "tools": None,
            "tokenize": "server_style_add_special_tokens_true_no_truncation",
            "prompt_definition": "system + user EvidencePackage + empty think assistant generation prefix",
        },
        "budgets": budget_reports,
        "comparison_to_160": comparisons,
        "validation": {
            "expected_records": manifest["records"]["candidate"],
            "scanned_records": len(records),
            "budget_rows": {
                str(budget): len(rows_by_budget[budget]) for budget in budgets
            },
            "ordered_prefix_monotonic": True,
            "locator_records": len(locators),
            "counts_closed": all(
                len(rows_by_budget[budget]) == len(records) for budget in budgets
            ),
        },
        "readiness": {
            "runtime_prompt_counter_audited": True,
            "canonical_budget_160_audited": True,
            "real_retrieval_audited": False,
            "training_ready": False,
        },
        "limitations": [
            "本报告使用 canonical authoring 的既定证据顺序，不代表真实 retrieval 排名或召回覆盖率。",
            "本报告不调用生成模型、embedding 或 reranker。",
            "160 是否作为最终长期值仍需结合真实 retrieval 包覆盖与训练后生成行为复核。",
        ],
        "outputs": {
            "report": REPORT_FILENAME,
            "locators": {"path": LOCATOR_FILENAME, "records": len(locators)},
            "sha256_manifest": HASH_FILENAME,
        },
        "complete": True,
    }
    _write_outputs(output_dir, report, locators)
    return report


def main() -> None:
    """解析参数并运行 canonical RAG-SFT 运行时预算审计。"""

    parser = argparse.ArgumentParser(
        description="审计 128/160/192 输出预算下的 canonical RAG-SFT 证据覆盖"
    )
    parser.add_argument("--rag-manifest", type=Path, default=DEFAULT_RAG_MANIFEST)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        report = audit_rag_sft_runtime_budget(
            args.rag_manifest,
            args.tokenizer_path,
            args.output_dir,
        )
    except RagRuntimeBudgetAuditError as error:
        parser.error(str(error))
    primary = report["budgets"][str(RAG_MAX_OUTPUT_TOKENS)]["overall"]["counts"]
    print(
        f"[完成] 审计 {primary['records']} 条，160-token 完整包适配 "
        f"{primary['full_visible_package_fit']} 条，首条适配 "
        f"{primary['first_evidence_fit']} 条"
    )
    print("[边界] canonical 预算审计不等于真实 retrieval 覆盖")
    print(f"报告目录: {args.output_dir}")


if __name__ == "__main__":
    main()
