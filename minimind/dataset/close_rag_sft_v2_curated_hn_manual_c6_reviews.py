"""闭合第六轮 curated HN 的独立双审资产，不生成训练数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .build_rag_sft_v2_curated_hn_reuse_review import (
        _bound_path,
        _identity,
        _load_json,
        _load_jsonl,
        _payload_identity,
        _sha256,
        _serialize_jsonl,
        _stable_visible,
        _verify_manifest,
    )
except ImportError:  # 支持从项目根目录以模块方式运行。
    from minimind.dataset.build_rag_sft_v2_curated_hn_reuse_review import (
        _bound_path,
        _identity,
        _load_json,
        _load_jsonl,
        _payload_identity,
        _sha256,
        _serialize_jsonl,
        _stable_visible,
        _verify_manifest,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUEUE_ROOT = (
    PROJECT_ROOT
    / "minimind/dataset/RAG-SFT/review/v2/"
    "curated-hn-manual-review-queue-v1-20260814"
)
DEFAULT_DRAFTS_ROOT = (
    PROJECT_ROOT
    / "minimind/dataset/RAG-SFT/review/v2/"
    "curated-hn-manual-drafts-v1-20260814"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "minimind/dataset/RAG-SFT/review/v2/"
    "curated-hn-manual-semantic-review-v6-20260814"
)


class CuratedHnManualC6Error(RuntimeError):
    """第六轮 curated HN 双审记录不满足身份或裁决要求。"""


def _key(row: object, description: str) -> tuple[str, str, int]:
    if not isinstance(row, dict):
        raise CuratedHnManualC6Error(f"{description}必须是对象")
    query_id = row.get("query_id")
    chunk_id = row.get("chunk_id")
    rank = row.get("candidate_rank")
    if not isinstance(query_id, str) or not isinstance(chunk_id, str) or type(rank) is not int:
        raise CuratedHnManualC6Error(f"{description}身份字段无效")
    return query_id, chunk_id, rank


def _unique(rows: list[dict[str, Any]], description: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    result: dict[tuple[str, str, int], dict[str, Any]] = {}
    for number, row in enumerate(rows, start=1):
        key = _key(row, f"{description}第 {number} 条")
        if key in result:
            raise CuratedHnManualC6Error(f"{description}存在重复候选: {key[0]}")
        result[key] = row
    return result


def close(*, queue_root: Path, drafts_root: Path, output_dir: Path) -> dict[str, object]:
    """只关闭已完成的 C6 对抗审阅与 B6 法律支持复审。"""

    queue_root = queue_root.resolve()
    drafts_root = drafts_root.resolve()
    output_dir = output_dir.resolve()
    partial = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial.exists():
        raise CuratedHnManualC6Error(f"输出目录或临时目录已存在: {output_dir}")

    queue_manifest_path = queue_root / "manifest.json"
    queue_manifest_sha256 = _verify_manifest(queue_manifest_path)
    queue_manifest = _load_json(queue_manifest_path, "人工候选队列 manifest")
    if (
        queue_manifest.get("pipeline") != "rag_sft_v2_curated_hn_manual_review_queue"
        or queue_manifest.get("policy", {}).get("training_ready") is not False
    ):
        raise CuratedHnManualC6Error("人工候选队列状态无效")
    queue_path = _bound_path(
        queue_manifest.get("output", {}).get("manual_review_queue"), "人工候选队列"
    )
    queue_rows = _load_jsonl(queue_path, "人工候选队列")
    queue = {row.get("query_id"): row for row in queue_rows}
    if len(queue) != 381 or len(queue) != len(queue_rows):
        raise CuratedHnManualC6Error("人工候选队列 query 身份不闭合")

    adversarial_path = drafts_root / "agent-c-6-adversarial.jsonl"
    legal_path = drafts_root / "agent-b-6-legal-c.jsonl"
    adversarial_by_key = _unique(_load_jsonl(adversarial_path, "C6 对抗草稿"), "C6 对抗草稿")
    legal_by_key = _unique(_load_jsonl(legal_path, "B6 法律支持复审"), "B6 法律支持复审")
    if not adversarial_by_key or set(adversarial_by_key) != set(legal_by_key):
        raise CuratedHnManualC6Error("C6 对抗与法律支持复审覆盖不一致")

    adversarial: list[dict[str, object]] = []
    legal: list[dict[str, object]] = []
    adjudication: list[dict[str, object]] = []
    source_ledger: list[dict[str, object]] = []
    variants: list[dict[str, object]] = []
    seen_queries: set[str] = set()
    for query_id, chunk_id, rank in sorted(adversarial_by_key):
        adversarial_row = adversarial_by_key[(query_id, chunk_id, rank)]
        legal_row = legal_by_key[(query_id, chunk_id, rank)]
        queue_row = queue.get(query_id)
        candidates = queue_row.get("candidate_articles") if isinstance(queue_row, dict) else None
        candidate = next(
            (
                item
                for item in candidates or []
                if item.get("chunk_id") == chunk_id and item.get("rank") == rank
            ),
            None,
        )
        if (
            not isinstance(queue_row, dict)
            or queue_row.get("review_partition") != "c"
            or not isinstance(candidate, dict)
            or adversarial_row.get("adversarial_label") != "hard_negative"
        ):
            raise CuratedHnManualC6Error(f"C6 候选身份或对抗审阅不闭合: {query_id}")
        legal_support = legal_row.get("legal_support")
        if legal_support not in {"direct", "partial", "alternative", "none", "uncertain"}:
            raise CuratedHnManualC6Error(f"C6 法律支持标签无效: {query_id}")
        if query_id in seen_queries:
            raise CuratedHnManualC6Error(f"C6 存在重复 query: {query_id}")
        seen_queries.add(query_id)

        variant_id = f"{query_id}:hn:curated:manual-v6"
        review_id = f"{variant_id}:{chunk_id}"
        adversarial.append(
            {
                "review_id": review_id,
                "query_id": query_id,
                "variant_id": variant_id,
                "chunk_id": chunk_id,
                "adversarial_label": "hard_negative",
                "reason": adversarial_row.get("reason"),
                "reviewer": adversarial_row.get("reviewer"),
            }
        )
        legal.append(
            {
                "review_id": review_id,
                "query_id": query_id,
                "variant_id": variant_id,
                "chunk_id": chunk_id,
                "legal_support": legal_support,
                "reason": legal_row.get("reason"),
                "reviewer": legal_row.get("reviewer"),
            }
        )
        approved = legal_support == "none"
        adjudication.append(
            {
                "review_id": review_id,
                "query_id": query_id,
                "variant_id": variant_id,
                "chunk_id": chunk_id,
                "adversarial_label": "hard_negative",
                "legal_support": legal_support,
                "decision": "approve_evidence" if approved else "exclude_evidence",
                "decision_reason": (
                    "同时满足 hard_negative 与 none，准入。"
                    if approved
                    else "法律支持不为 none，保守排除。"
                ),
                "adjudicator": "curated_manual_adjudication_v6",
            }
        )
        source_ledger.append(
            {
                "review_id": review_id,
                "query_id": query_id,
                "chunk_id": chunk_id,
                "candidate_rank": rank,
                "queue_path": str(queue_path),
                "queue_sha256": _sha256(queue_path),
                "article_source": "rag/chunk/article_index.jsonl",
                "article_law_name": candidate.get("law_name"),
                "article_no": candidate.get("article_no"),
                "adversarial_draft": adversarial_path.name,
                "legal_draft": legal_path.name,
            }
        )
        if approved:
            required = [item.get("chunk_id") for item in queue_row.get("required_gt", [])]
            if not required or any(not isinstance(item, str) for item in required):
                raise CuratedHnManualC6Error(f"C6 required GT 无效: {query_id}")
            variants.append(
                {
                    "variant_id": variant_id,
                    "query_id": query_id,
                    "visible_chunk_ids": _stable_visible(required, query_id, chunk_id),
                    "source": "curated",
                    "retrieval_identity": (
                        "curated:manual:c6:queue-sha256:"
                        f"{_sha256(queue_path)}:rank:{rank}"
                    ),
                    "non_gt_labels": [{"chunk_id": chunk_id, "label": "hard_negative"}],
                    "review_decision": "approved",
                }
            )

    payloads = {
        "curated-hn-variants.jsonl": _serialize_jsonl(variants),
        "adversarial.jsonl": _serialize_jsonl(adversarial),
        "legal-support.jsonl": _serialize_jsonl(legal),
        "adjudication.jsonl": _serialize_jsonl(adjudication),
        "source-evidence-ledger.jsonl": _serialize_jsonl(source_ledger),
    }
    manifest = {
        "pipeline": "rag_sft_v2_curated_hn_manual_semantic_review",
        "release_status": "curated_hn_manual_semantic_review_closed",
        "inputs": {
            "queue_manifest": {
                **_identity(queue_manifest_path),
                "manifest_sha256": queue_manifest_sha256,
            },
            "queue": _identity(queue_path, records=len(queue_rows)),
            "independent_drafts": [
                _identity(adversarial_path, records=len(adversarial)),
                _identity(legal_path, records=len(legal)),
            ],
        },
        "records": {
            "reviewed": len(adjudication),
            "approved": len(variants),
            "excluded": len(adjudication) - len(variants),
            "adversarial_labels": {"hard_negative": len(adversarial)},
            "legal_support": {
                label: sum(item["legal_support"] == label for item in legal)
                for label in ("direct", "partial", "alternative", "none", "uncertain")
            },
        },
        "policy": {
            "source_candidates_modified": False,
            "formal_hn_materialized": False,
            "context_length_audit": "pending",
            "training_ready": False,
        },
        "validation": {
            "strict_utf8_jsonl": True,
            "independent_double_review_covered": True,
            "admit_rule": "adversarial_label == hard_negative and legal_support == none",
            "one_query_one_variant": True,
            "queue_candidate_identity_closed": True,
        },
        "output": {
            name: _payload_identity(output_dir / name, payload, len(payload.splitlines()))
            for name, payload in payloads.items()
        },
        "complete": True,
    }
    payloads["manifest.json"] = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    payloads["README.md"] = (
        "# RAG-SFT v2 curated HN 第六轮人工双审闭合\n\n"
        "本目录闭合 C6 对抗审阅、B6 独立法律支持复审和主审裁决。"
        "仅 `hard_negative + none` 准入语义候选；未物化正式 HN，"
        "长度审核未完成，`training_ready=false`。\n"
    )
    try:
        partial.mkdir(parents=True, exist_ok=False)
        for name, payload in payloads.items():
            (partial / name).write_text(payload, encoding="utf-8", newline="\n")
        (partial / "manifest.sha256").write_text(
            "".join(f"{_sha256(partial / name)}  {name}\n" for name in payloads),
            encoding="ascii",
            newline="\n",
        )
        partial.replace(output_dir)
    except OSError as error:
        raise CuratedHnManualC6Error("无法原子发布 C6 双审闭合资产；已保留临时目录") from error
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    parser.add_argument("--drafts-root", type=Path, default=DEFAULT_DRAFTS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        manifest = close(
            queue_root=args.queue_root,
            drafts_root=args.drafts_root,
            output_dir=args.output_dir,
        )
    except (CuratedHnManualC6Error, OSError, TypeError, ValueError) as error:
        raise SystemExit(f"RAG_SFT_V2_CURATED_HN_C6_CLOSE_FAILED: {error}") from error
    print(
        "RAG_SFT_V2_CURATED_HN_C6_CLOSE_OK "
        f"approved={manifest['records']['approved']} training_ready=false"
    )


if __name__ == "__main__":
    main()
