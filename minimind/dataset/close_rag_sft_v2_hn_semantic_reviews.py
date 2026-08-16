"""闭合 RAG-SFT v2 真实检索 HN 的双审与主审账本。

本工具只读取冻结的 source batch 和独立审核记录，在新的输出目录发布
可追溯账本；不会改写源候选，也不会物化训练 HN。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


class HnSemanticReviewClosureError(RuntimeError):
    """HN 语义审核无法安全闭合。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, start=1):
                if not line.strip():
                    raise HnSemanticReviewClosureError(f"{description}存在空行: {number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise HnSemanticReviewClosureError(f"{description}第 {number} 行不是对象")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, HnSemanticReviewClosureError):
            raise
        raise HnSemanticReviewClosureError(f"无法读取{description}: {path}") from error
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    )
    path.write_text(payload, encoding="utf-8", newline="\n")


def _identity(path: Path, records: int | None = None) -> dict[str, object]:
    identity: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if records is not None:
        identity["records"] = records
    return identity


def _load_reviews(path: Path, role: str) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(path, role)
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        query_id = row.get("query_id")
        variant_id = row.get("variant_id")
        if not isinstance(query_id, str):
            raise HnSemanticReviewClosureError(f"{role}缺少 query 身份")
        if variant_id is not None and not isinstance(variant_id, str):
            raise HnSemanticReviewClosureError(f"{role} variant 身份无效")
        if query_id in indexed:
            raise HnSemanticReviewClosureError(f"{role} query 重复: {query_id}")
        indexed[query_id] = row
    return indexed


def _review_by_chunk(row: dict[str, Any], *, role: str) -> dict[str, dict[str, Any]]:
    key = "non_gt_reviews" if role == "adversarial" else "reviews"
    value = row.get(key)
    if not isinstance(value, list):
        raise HnSemanticReviewClosureError(f"{role}审核记录缺少 {key}")
    indexed: dict[str, dict[str, Any]] = {}
    for review in value:
        if not isinstance(review, dict) or not isinstance(review.get("chunk_id"), str):
            raise HnSemanticReviewClosureError(f"{role}审核证据身份无效")
        chunk_id = review["chunk_id"]
        if chunk_id in indexed:
            raise HnSemanticReviewClosureError(f"{role}审核证据重复: {chunk_id}")
        indexed[chunk_id] = review
    return indexed


def _close_batch(
    *,
    source_path: Path,
    adversarial_path: Path,
    legal_path: Path,
    output_dir: Path,
    batch: int,
) -> dict[str, object]:
    source_rows = _read_jsonl(source_path, "source batch")
    adversarial = _load_reviews(adversarial_path, "adversarial")
    legal = _load_reviews(legal_path, "legal_support")
    source_hash = _sha256(source_path)
    if len(source_rows) != len(adversarial) or len(source_rows) != len(legal):
        raise HnSemanticReviewClosureError(f"第 {batch} 批 package 数不闭合")

    evidence_rows: list[dict[str, Any]] = []
    package_rows: list[dict[str, Any]] = []
    reusable_rows: list[dict[str, Any]] = []
    adversarial_counts: Counter[str] = Counter()
    legal_counts: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    for package in source_rows:
        query_id = package.get("query_id")
        variant_id = package.get("variant_id")
        evidence = package.get("evidence")
        if not isinstance(query_id, str) or not isinstance(variant_id, str) or not isinstance(evidence, list):
            raise HnSemanticReviewClosureError(f"第 {batch} 批 source package 身份无效")
        adv_row = adversarial.get(query_id)
        legal_row = legal.get(query_id)
        if adv_row is None or legal_row is None:
            raise HnSemanticReviewClosureError(f"第 {batch} 批缺少审核 query: {query_id}")
        if (
            adv_row.get("variant_id") not in {None, variant_id}
            or legal_row.get("variant_id") not in {None, variant_id}
        ):
            raise HnSemanticReviewClosureError(f"第 {batch} 批 variant 不闭合: {query_id}")
        adv_by_chunk = _review_by_chunk(adv_row, role="adversarial")
        legal_by_chunk = _review_by_chunk(legal_row, role="legal_support")
        non_gt = [item for item in evidence if item.get("is_required_gt") is False]
        chunk_ids = [item.get("chunk_id") for item in non_gt]
        if not all(isinstance(item, str) for item in chunk_ids) or len(chunk_ids) != len(set(chunk_ids)):
            raise HnSemanticReviewClosureError(f"第 {batch} 批 source non-GT 身份无效: {query_id}")
        if set(chunk_ids) != set(adv_by_chunk) or set(chunk_ids) != set(legal_by_chunk):
            raise HnSemanticReviewClosureError(f"第 {batch} 批双审覆盖不闭合: {query_id}")

        approved_ids: list[str] = []
        package_blockers: list[str] = []
        for item in non_gt:
            chunk_id = item["chunk_id"]
            evidence_id = item.get("evidence_id")
            if not isinstance(evidence_id, str):
                raise HnSemanticReviewClosureError(f"第 {batch} 批 evidence_id 无效: {query_id}")
            adv = adv_by_chunk[chunk_id]
            support = legal_by_chunk[chunk_id]
            adv_label = adv.get("label")
            support_label = support.get("label")
            if adv_label not in {"hard_negative", "redundant_support", "irrelevant", "uncertain"}:
                raise HnSemanticReviewClosureError(f"第 {batch} 批对抗标签无效: {query_id}/{chunk_id}")
            if support_label not in {"direct", "partial", "alternative", "none", "uncertain"}:
                raise HnSemanticReviewClosureError(f"第 {batch} 批法律支持标签无效: {query_id}/{chunk_id}")
            approved = adv_label == "hard_negative" and support_label == "none"
            decision = "approve_evidence" if approved else "exclude_evidence"
            if approved:
                approved_ids.append(evidence_id)
            if adv_label == "redundant_support" or support_label != "none":
                package_blockers.append(chunk_id)
            adversarial_counts[adv_label] += 1
            legal_counts[support_label] += 1
            evidence_rows.append(
                {
                    "record_type": "evidence_adjudication",
                    "source_batch": source_path.name,
                    "source_batch_sha256": source_hash,
                    "query_id": query_id,
                    "variant_id": variant_id,
                    "evidence_id": evidence_id,
                    "chunk_id": chunk_id,
                    "is_required_gt": False,
                    "adversarial_label": adv_label,
                    "adversarial_reason": adv.get("reason", ""),
                    "legal_support": support_label,
                    "legal_support_reason": support.get("reason", ""),
                    "decision": decision,
                    "decision_reason": (
                        "对抗审核为 hard_negative 且法律支持审核为 none。"
                        if approved
                        else "未同时满足 hard_negative 与 none 的唯一准入规则。"
                    ),
                    "adjudicator": "hn_semantic_review_closure_v2",
                }
            )
        package_decision = "approve_candidate" if approved_ids and not package_blockers else "exclude_candidate"
        decisions[package_decision] += 1
        package_rows.append(
            {
                "record_type": "package_adjudication",
                "source_batch": source_path.name,
                "source_batch_sha256": source_hash,
                "query_id": query_id,
                "variant_id": variant_id,
                "decision": package_decision,
                "approved_evidence_ids": approved_ids if package_decision == "approve_candidate" else [],
                "reason": (
                    "存在准入证据且包内所有 non-GT 均无支持性或不确定性关系。"
                    if package_decision == "approve_candidate"
                    else "包内存在支持性、冗余或不确定证据，或没有准入证据；保守排除。"
                ),
            }
        )
        # 包级准入不等于可复用证据的单独再准入。本轮不把候选物化或登记为
        # 可复用 HN，避免从原始 retrieved package 反向构造训练数据。

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output_dir / "adjudication.jsonl", [*package_rows, *evidence_rows])
    _write_jsonl(output_dir / "reusable-hn-evidence.jsonl", reusable_rows)
    manifest = {
        "pipeline": "rag_sft_v2_hn_semantic_review_closure",
        "release_status": "batch_semantic_review_closed",
        "batch": batch,
        "inputs": {
            "review_batch": _identity(source_path, len(source_rows)),
            "adversarial": _identity(adversarial_path, len(adversarial)),
            "legal_support": _identity(legal_path, len(legal)),
        },
        "outputs": {
            "adjudication": _identity(output_dir / "adjudication.jsonl", len(package_rows) + len(evidence_rows)),
            "reusable_hn_evidence": _identity(output_dir / "reusable-hn-evidence.jsonl", len(reusable_rows)),
        },
        "records": {
            "packages": len(package_rows),
            "non_gt_evidence": len(evidence_rows),
            "package_decisions": dict(decisions),
            "adversarial_labels": dict(adversarial_counts),
            "legal_support_labels": dict(legal_counts),
            "reusable_hn_evidence": len(reusable_rows),
        },
        "policy": {
            "source_candidates_modified": False,
            "formal_hn_materialized": False,
            "retrieval_package_reconstruction_forbidden": True,
            "training_ready": False,
            "approve_rule": "adversarial_label == hard_negative 且 legal_support == none",
        },
        "readiness": {"training_ready": False, "semantic_review_complete": True},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    readme = (
        f"# 第 {batch} 批 HN 语义审核闭合\n\n"
        "本目录只读取冻结的真实检索 source batch 与两份独立审核。"
        "未修改 source candidates，未物化正式训练 HN，`training_ready=false`。\n\n"
        "唯一准入规则为 `adversarial_label == hard_negative` 且 `legal_support == none`。"
        "任何支持性、冗余或不确定关系均保守排除。\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    files = ["adjudication.jsonl", "reusable-hn-evidence.jsonl", "manifest.json", "README.md"]
    (output_dir / "manifest.sha256").write_text(
        "".join(f"{_sha256(output_dir / name)}  {name}\n" for name in files),
        encoding="ascii",
        newline="\n",
    )
    return manifest


def _close_label_input(
    *, source_path: Path, label_path: Path, output_dir: Path, batch: int
) -> dict[str, object]:
    """用人工确认的逐证据双审标签闭合尚无历史 JSONL 的批次。"""

    source_rows = _read_jsonl(source_path, "source batch")
    labels = [
        row for row in _read_jsonl(label_path, "双审标签输入")
        if row.get("batch") in {batch, str(batch), f"batch-{batch:02d}"}
    ]
    source_hash = _sha256(source_path)
    source_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for package in source_rows:
        query_id = package.get("query_id")
        evidence = package.get("evidence")
        if not isinstance(query_id, str) or not isinstance(evidence, list):
            raise HnSemanticReviewClosureError("source package 身份无效")
        for item in evidence:
            if item.get("is_required_gt") is False:
                evidence_id = item.get("evidence_id")
                if not isinstance(evidence_id, str):
                    raise HnSemanticReviewClosureError("source evidence_id 无效")
                source_by_key[(query_id, evidence_id)] = {"package": package, "evidence": item}
    label_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in labels:
        query_id = row.get("query_id")
        evidence_id = row.get("evidence_id")
        if not isinstance(query_id, str) or not isinstance(evidence_id, str):
            raise HnSemanticReviewClosureError("标签身份无效")
        key = (query_id, evidence_id)
        if key in label_by_key:
            raise HnSemanticReviewClosureError(f"标签证据重复: {key}")
        label_by_key[key] = row
    if set(source_by_key) != set(label_by_key):
        raise HnSemanticReviewClosureError("source 与双审标签未逐证据闭合")

    evidence_rows: list[dict[str, Any]] = []
    package_rows: list[dict[str, Any]] = []
    adversarial_counts: Counter[str] = Counter()
    legal_counts: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    for package in source_rows:
        query_id = package["query_id"]
        variant_id = package["variant_id"]
        approved_ids: list[str] = []
        blockers: list[str] = []
        for item in package["evidence"]:
            if item.get("is_required_gt") is not False:
                continue
            evidence_id = item["evidence_id"]
            label = label_by_key[(query_id, evidence_id)]
            if label.get("variant_id") != variant_id or label.get("chunk_id") != item.get("chunk_id"):
                raise HnSemanticReviewClosureError(f"标签 query/variant/evidence 不闭合: {query_id}/{evidence_id}")
            adv_label = label.get("adversarial_label")
            support_label = label.get("legal_support")
            if adv_label not in {"hard_negative", "redundant_support", "irrelevant", "uncertain"}:
                raise HnSemanticReviewClosureError("对抗标签无效")
            if support_label not in {"direct", "partial", "alternative", "none", "uncertain"}:
                raise HnSemanticReviewClosureError("法律支持标签无效")
            approved = adv_label == "hard_negative" and support_label == "none"
            if approved:
                approved_ids.append(evidence_id)
            if adv_label == "redundant_support" or support_label != "none":
                blockers.append(item["chunk_id"])
            adversarial_counts[adv_label] += 1
            legal_counts[support_label] += 1
            evidence_rows.append(
                {
                    "record_type": "evidence_adjudication",
                    "source_batch": source_path.name,
                    "source_batch_sha256": source_hash,
                    "query_id": query_id,
                    "variant_id": variant_id,
                    "evidence_id": evidence_id,
                    "chunk_id": item["chunk_id"],
                    "is_required_gt": False,
                    "adversarial_label": adv_label,
                    "adversarial_reason": label.get("adversarial_reason", ""),
                    "legal_support": support_label,
                    "legal_support_reason": label.get("legal_support_reason", ""),
                    "decision": "approve_evidence" if approved else "exclude_evidence",
                    "decision_reason": (
                        "对抗审核为 hard_negative 且法律支持审核为 none。"
                        if approved else "未同时满足 hard_negative 与 none 的唯一准入规则。"
                    ),
                    "adjudicator": "hn_semantic_review_closure_v2",
                }
            )
        decision = "approve_candidate" if approved_ids and not blockers else "exclude_candidate"
        decisions[decision] += 1
        package_rows.append(
            {
                "record_type": "package_adjudication",
                "source_batch": source_path.name,
                "source_batch_sha256": source_hash,
                "query_id": query_id,
                "variant_id": variant_id,
                "decision": decision,
                "approved_evidence_ids": approved_ids if decision == "approve_candidate" else [],
                "reason": (
                    "存在准入证据且包内所有 non-GT 均无支持性或不确定性关系。"
                    if decision == "approve_candidate"
                    else "包内存在支持性、冗余或不确定证据，或没有准入证据；保守排除。"
                ),
            }
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output_dir / "adjudication.jsonl", [*package_rows, *evidence_rows])
    _write_jsonl(output_dir / "reusable-hn-evidence.jsonl", [])
    manifest = {
        "pipeline": "rag_sft_v2_hn_semantic_review_closure",
        "release_status": "batch_semantic_review_closed",
        "batch": batch,
        "inputs": {
            "review_batch": _identity(source_path, len(source_rows)),
            "independent_double_review_labels": _identity(label_path, len(labels)),
        },
        "outputs": {
            "adjudication": _identity(output_dir / "adjudication.jsonl", len(package_rows) + len(evidence_rows)),
            "reusable_hn_evidence": _identity(output_dir / "reusable-hn-evidence.jsonl", 0),
        },
        "records": {
            "packages": len(package_rows),
            "non_gt_evidence": len(evidence_rows),
            "package_decisions": dict(decisions),
            "adversarial_labels": dict(adversarial_counts),
            "legal_support_labels": dict(legal_counts),
            "reusable_hn_evidence": 0,
        },
        "policy": {
            "source_candidates_modified": False,
            "formal_hn_materialized": False,
            "retrieval_package_reconstruction_forbidden": True,
            "training_ready": False,
            "approve_rule": "adversarial_label == hard_negative 且 legal_support == none",
        },
        "readiness": {"training_ready": False, "semantic_review_complete": True},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output_dir / "README.md").write_text(
        f"# 第 {batch} 批 HN 语义审核闭合\n\n"
        "本目录由 source batch 和独立双审标签闭合。未修改源候选，未物化正式训练 HN，`training_ready=false`。\n",
        encoding="utf-8", newline="\n",
    )
    files = ["adjudication.jsonl", "reusable-hn-evidence.jsonl", "manifest.json", "README.md"]
    (output_dir / "manifest.sha256").write_text(
        "".join(f"{_sha256(output_dir / name)}  {name}\n" for name in files), encoding="ascii", newline="\n"
    )
    return manifest


_ADVERSARIAL_LABELS = {"hard_negative", "redundant_support", "irrelevant", "uncertain"}
_LEGAL_SUPPORT_LABELS = {"direct", "partial", "alternative", "none", "uncertain"}


def _source_identity(
    source_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], str]:
    """读取并验证一个冻结 source batch 的包与非 GT 身份。"""

    packages = _read_jsonl(source_path, "source batch")
    by_query: dict[str, dict[str, Any]] = {}
    for package in packages:
        query_id = package.get("query_id")
        variant_id = package.get("variant_id")
        evidence = package.get("evidence")
        if not isinstance(query_id, str) or not isinstance(variant_id, str) or not isinstance(evidence, list):
            raise HnSemanticReviewClosureError(f"source package 身份无效: {source_path.name}")
        if query_id in by_query:
            raise HnSemanticReviewClosureError(f"source query 重复: {source_path.name}/{query_id}")
        non_gt: list[dict[str, Any]] = []
        seen_chunks: set[str] = set()
        seen_evidence: set[str] = set()
        for item in evidence:
            if item.get("is_required_gt") is not False:
                continue
            chunk_id = item.get("chunk_id")
            evidence_id = item.get("evidence_id")
            if not isinstance(chunk_id, str) or not isinstance(evidence_id, str):
                raise HnSemanticReviewClosureError(f"source non-GT 身份无效: {source_path.name}/{query_id}")
            if chunk_id in seen_chunks or evidence_id in seen_evidence:
                raise HnSemanticReviewClosureError(f"source non-GT 重复: {source_path.name}/{query_id}")
            seen_chunks.add(chunk_id)
            seen_evidence.add(evidence_id)
            non_gt.append(item)
        by_query[query_id] = {
            "query_id": query_id,
            "variant_id": variant_id,
            "non_gt": non_gt,
        }
    return packages, by_query, _sha256(source_path)


def _label_rows_for_batch(label_path: Path, batch: int) -> list[dict[str, Any]]:
    """从联合标签输入取得一个批次，保留其原始逐证据审阅结论。"""

    batch_markers: set[object] = {batch, str(batch), f"batch-{batch:02d}"}
    return [row for row in _read_jsonl(label_path, "双审标签输入") if row.get("batch") in batch_markers]


def _normalize_label_reviews(
    *, source_path: Path, label_path: Path, batch: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, object]]]:
    """将已经逐证据核验的联合标签拆成两个独立角色账本。"""

    _, source_by_query, source_hash = _source_identity(source_path)
    label_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for label in _label_rows_for_batch(label_path, batch):
        query_id = label.get("query_id")
        evidence_id = label.get("evidence_id")
        if not isinstance(query_id, str) or not isinstance(evidence_id, str):
            raise HnSemanticReviewClosureError(f"联合标签身份无效: 第 {batch} 批")
        key = (query_id, evidence_id)
        if key in label_by_key:
            raise HnSemanticReviewClosureError(f"联合标签证据重复: 第 {batch} 批/{key}")
        label_by_key[key] = label

    expected_keys = {
        (query_id, item["evidence_id"])
        for query_id, package in source_by_query.items()
        for item in package["non_gt"]
    }
    if set(label_by_key) != expected_keys:
        raise HnSemanticReviewClosureError(f"联合标签与 source non-GT 不闭合: 第 {batch} 批")

    adversarial: list[dict[str, Any]] = []
    legal: list[dict[str, Any]] = []
    for query_id, package in source_by_query.items():
        adv_reviews: list[dict[str, Any]] = []
        legal_reviews: list[dict[str, Any]] = []
        for item in package["non_gt"]:
            label = label_by_key[(query_id, item["evidence_id"])]
            if label.get("variant_id") != package["variant_id"] or label.get("chunk_id") != item["chunk_id"]:
                raise HnSemanticReviewClosureError(f"联合标签 query/variant/evidence 不闭合: {query_id}")
            adversarial_label = label.get("adversarial_label")
            legal_support = label.get("legal_support")
            if adversarial_label not in _ADVERSARIAL_LABELS or legal_support not in _LEGAL_SUPPORT_LABELS:
                raise HnSemanticReviewClosureError(f"联合标签存在无效结论: {query_id}/{item['evidence_id']}")
            adv_reviews.append(
                {
                    "evidence_id": item["evidence_id"],
                    "chunk_id": item["chunk_id"],
                    "label": adversarial_label,
                    "confusion_basis": label.get("adversarial_confusion_basis", ""),
                    "decisive_mismatch": label.get("adversarial_decisive_mismatch", ""),
                    "reason": label.get("adversarial_reason", ""),
                }
            )
            legal_reviews.append(
                {
                    "evidence_id": item["evidence_id"],
                    "chunk_id": item["chunk_id"],
                    "label": legal_support,
                    "supported_claims": label.get("supported_claims", []),
                    "reason": label.get("legal_support_reason", ""),
                }
            )
        adversarial.append(
            {
                "record_type": "package_adversarial_review",
                "source_batch": source_path.name,
                "source_batch_sha256": source_hash,
                "query_id": query_id,
                "variant_id": package["variant_id"],
                "non_gt_reviews": adv_reviews,
            }
        )
        legal.append(
            {
                "record_type": "package_legal_support_review",
                "source_batch": source_path.name,
                "source_batch_sha256": source_hash,
                "query_id": query_id,
                "variant_id": package["variant_id"],
                "reviews": legal_reviews,
            }
        )
    return adversarial, legal, [{"role": "independent_double_review_labels", **_identity(label_path, len(_read_jsonl(label_path, "双审标签输入")))}]


def _normalize_historical_reviews(
    *, source_path: Path, historical_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, object]]]:
    """将已验证的历史双审规范化，并补入冻结 source 身份字段。"""

    _, source_by_query, source_hash = _source_identity(source_path)
    adversarial_path = historical_dir / "adversarial.jsonl"
    legal_path = historical_dir / "legal-support.jsonl"
    raw_adversarial = _load_reviews(adversarial_path, "adversarial")
    raw_legal = _load_reviews(legal_path, "legal_support")
    if set(raw_adversarial) != set(source_by_query) or set(raw_legal) != set(source_by_query):
        raise HnSemanticReviewClosureError(f"历史双审 package 覆盖不闭合: {source_path.name}")

    adversarial: list[dict[str, Any]] = []
    legal: list[dict[str, Any]] = []
    for query_id, package in source_by_query.items():
        adv_raw = raw_adversarial[query_id]
        legal_raw = raw_legal[query_id]
        if adv_raw.get("variant_id") not in {None, package["variant_id"]}:
            raise HnSemanticReviewClosureError(f"历史对抗审阅 variant 不闭合: {query_id}")
        if legal_raw.get("variant_id") not in {None, package["variant_id"]}:
            raise HnSemanticReviewClosureError(f"历史法律支持审阅 variant 不闭合: {query_id}")
        adv_by_chunk = _review_by_chunk(adv_raw, role="adversarial")
        legal_by_chunk = _review_by_chunk(legal_raw, role="legal_support")
        source_chunks = [item["chunk_id"] for item in package["non_gt"]]
        if set(adv_by_chunk) != set(source_chunks) or set(legal_by_chunk) != set(source_chunks):
            raise HnSemanticReviewClosureError(f"历史双审 evidence 覆盖不闭合: {query_id}")
        adv_reviews: list[dict[str, Any]] = []
        legal_reviews: list[dict[str, Any]] = []
        for item in package["non_gt"]:
            adv = adv_by_chunk[item["chunk_id"]]
            legal_item = legal_by_chunk[item["chunk_id"]]
            if adv.get("label") not in _ADVERSARIAL_LABELS or legal_item.get("label") not in _LEGAL_SUPPORT_LABELS:
                raise HnSemanticReviewClosureError(f"历史双审标签无效: {query_id}/{item['evidence_id']}")
            adv_reviews.append(
                {
                    "evidence_id": item["evidence_id"],
                    "chunk_id": item["chunk_id"],
                    "label": adv["label"],
                    "confusion_basis": adv.get("confusion_basis", ""),
                    "decisive_mismatch": adv.get("decisive_mismatch", ""),
                    "reason": adv.get("reason", ""),
                }
            )
            legal_reviews.append(
                {
                    "evidence_id": item["evidence_id"],
                    "chunk_id": item["chunk_id"],
                    "label": legal_item["label"],
                    "supported_claims": legal_item.get("supported_claims", []),
                    "reason": legal_item.get("reason", ""),
                }
            )
        adversarial.append(
            {
                "record_type": "package_adversarial_review",
                "source_batch": source_path.name,
                "source_batch_sha256": source_hash,
                "query_id": query_id,
                "variant_id": package["variant_id"],
                "non_gt_reviews": adv_reviews,
            }
        )
        legal.append(
            {
                "record_type": "package_legal_support_review",
                "source_batch": source_path.name,
                "source_batch_sha256": source_hash,
                "query_id": query_id,
                "variant_id": package["variant_id"],
                "reviews": legal_reviews,
            }
        )
    return adversarial, legal, [
        {"role": "historical_adversarial", **_identity(adversarial_path, len(raw_adversarial))},
        {"role": "historical_legal_support", **_identity(legal_path, len(raw_legal))},
    ]


def _write_final_batch(
    *,
    source_path: Path,
    adversarial: list[dict[str, Any]],
    legal: list[dict[str, Any]],
    review_provenance: list[dict[str, object]],
    output_dir: Path,
    batch: int,
) -> dict[str, Any]:
    """写出一个拥有两份独立审核、裁决和空复用账本的最终批次目录。"""

    source_rows, source_by_query, source_hash = _source_identity(source_path)
    adv_by_query = {row.get("query_id"): row for row in adversarial}
    legal_by_query = {row.get("query_id"): row for row in legal}
    if len(adv_by_query) != len(adversarial) or len(legal_by_query) != len(legal):
        raise HnSemanticReviewClosureError(f"最终审核 package 身份重复: 第 {batch} 批")
    if set(adv_by_query) != set(source_by_query) or set(legal_by_query) != set(source_by_query):
        raise HnSemanticReviewClosureError(f"最终审核 package 覆盖不闭合: 第 {batch} 批")

    package_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    adversarial_counts: Counter[str] = Counter()
    legal_counts: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    for package in source_rows:
        query_id = package["query_id"]
        variant_id = package["variant_id"]
        expected = source_by_query[query_id]["non_gt"]
        adv_row = adv_by_query[query_id]
        legal_row = legal_by_query[query_id]
        if (
            adv_row.get("source_batch") != source_path.name
            or legal_row.get("source_batch") != source_path.name
            or adv_row.get("source_batch_sha256") != source_hash
            or legal_row.get("source_batch_sha256") != source_hash
            or adv_row.get("variant_id") != variant_id
            or legal_row.get("variant_id") != variant_id
        ):
            raise HnSemanticReviewClosureError(f"最终审核 source/query/variant 不闭合: {query_id}")
        adv_items = adv_row.get("non_gt_reviews")
        legal_items = legal_row.get("reviews")
        if not isinstance(adv_items, list) or not isinstance(legal_items, list):
            raise HnSemanticReviewClosureError(f"最终审核证据列表无效: {query_id}")
        adv_by_evidence = {item.get("evidence_id"): item for item in adv_items}
        legal_by_evidence = {item.get("evidence_id"): item for item in legal_items}
        expected_ids = [item["evidence_id"] for item in expected]
        if (
            len(adv_by_evidence) != len(adv_items)
            or len(legal_by_evidence) != len(legal_items)
            or set(adv_by_evidence) != set(expected_ids)
            or set(legal_by_evidence) != set(expected_ids)
        ):
            raise HnSemanticReviewClosureError(f"最终双审证据覆盖不闭合: {query_id}")
        approved_ids: list[str] = []
        blockers: list[str] = []
        for item in expected:
            evidence_id = item["evidence_id"]
            chunk_id = item["chunk_id"]
            adv_item = adv_by_evidence[evidence_id]
            legal_item = legal_by_evidence[evidence_id]
            if adv_item.get("chunk_id") != chunk_id or legal_item.get("chunk_id") != chunk_id:
                raise HnSemanticReviewClosureError(f"最终双审 chunk 身份不闭合: {query_id}/{evidence_id}")
            adversarial_label = adv_item.get("label")
            legal_support = legal_item.get("label")
            if adversarial_label not in _ADVERSARIAL_LABELS or legal_support not in _LEGAL_SUPPORT_LABELS:
                raise HnSemanticReviewClosureError(f"最终双审标签无效: {query_id}/{evidence_id}")
            approved = adversarial_label == "hard_negative" and legal_support == "none"
            if approved:
                approved_ids.append(evidence_id)
            if adversarial_label == "redundant_support" or legal_support != "none":
                blockers.append(chunk_id)
            adversarial_counts[adversarial_label] += 1
            legal_counts[legal_support] += 1
            evidence_rows.append(
                {
                    "record_type": "evidence_adjudication",
                    "source_batch": source_path.name,
                    "source_batch_sha256": source_hash,
                    "query_id": query_id,
                    "variant_id": variant_id,
                    "evidence_id": evidence_id,
                    "chunk_id": chunk_id,
                    "is_required_gt": False,
                    "adversarial_label": adversarial_label,
                    "adversarial_reason": adv_item.get("reason", ""),
                    "legal_support": legal_support,
                    "legal_support_reason": legal_item.get("reason", ""),
                    "decision": "approve_evidence" if approved else "exclude_evidence",
                    "decision_reason": (
                        "对抗审核为 hard_negative 且法律支持审核为 none。"
                        if approved
                        else "未同时满足 hard_negative 与 none 的唯一准入规则。"
                    ),
                    "adjudicator": "hn_semantic_review_closure_v2",
                }
            )
        package_decision = "approve_candidate" if approved_ids and not blockers else "exclude_candidate"
        decisions[package_decision] += 1
        package_rows.append(
            {
                "record_type": "package_adjudication",
                "source_batch": source_path.name,
                "source_batch_sha256": source_hash,
                "query_id": query_id,
                "variant_id": variant_id,
                "decision": package_decision,
                "approved_evidence_ids": approved_ids if package_decision == "approve_candidate" else [],
                "reason": (
                    "存在准入证据且包内所有 non-GT 均无支持性或不确定性关系。"
                    if package_decision == "approve_candidate"
                    else "包内存在支持性、冗余或不确定证据，或没有准入证据；保守排除。"
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output_dir / "adversarial.jsonl", adversarial)
    _write_jsonl(output_dir / "legal-support.jsonl", legal)
    _write_jsonl(output_dir / "adjudication.jsonl", [*package_rows, *evidence_rows])
    _write_jsonl(output_dir / "reusable-hn-evidence.jsonl", [])
    manifest: dict[str, Any] = {
        "pipeline": "rag_sft_v2_hn_semantic_review_closure",
        "release_status": "batch_semantic_review_closed",
        "batch": batch,
        "inputs": {
            "review_batch": _identity(source_path, len(source_rows)),
            "review_provenance": review_provenance,
        },
        "outputs": {
            "adversarial": _identity(output_dir / "adversarial.jsonl", len(adversarial)),
            "legal_support": _identity(output_dir / "legal-support.jsonl", len(legal)),
            "adjudication": _identity(output_dir / "adjudication.jsonl", len(package_rows) + len(evidence_rows)),
            "reusable_hn_evidence": _identity(output_dir / "reusable-hn-evidence.jsonl", 0),
        },
        "records": {
            "packages": len(package_rows),
            "non_gt_evidence": len(evidence_rows),
            "package_decisions": dict(decisions),
            "adversarial_labels": dict(adversarial_counts),
            "legal_support_labels": dict(legal_counts),
            "reusable_hn_evidence": 0,
        },
        "policy": {
            "source_candidates_modified": False,
            "formal_hn_materialized": False,
            "retrieval_package_reconstruction_forbidden": True,
            "training_ready": False,
            "approve_rule": "adversarial_label == hard_negative 且 legal_support == none",
        },
        "readiness": {"training_ready": False, "semantic_review_complete": True},
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (output_dir / "README.md").write_text(
        f"# 第 {batch} 批 HN 语义审核闭合\n\n"
        "本目录只读取冻结的真实检索 source batch 和独立双审记录。"
        "未修改 source candidates，未物化正式训练 HN，`formal_hn_materialized=false`，"
        "`training_ready=false`。\n\n"
        "`adversarial.jsonl` 与 `legal-support.jsonl` 各有每个 source package 恰好一条记录。"
        "唯一证据准入规则为 `adversarial_label == hard_negative` 且 `legal_support == none`；"
        "任何支持性、冗余或不确定关系均保守排除。`reusable-hn-evidence.jsonl` 为可解析空账本，"
        "本轮不将候选物化或登记为可复用正式 HN。\n",
        encoding="utf-8",
        newline="\n",
    )
    file_names = [
        "adversarial.jsonl",
        "legal-support.jsonl",
        "adjudication.jsonl",
        "reusable-hn-evidence.jsonl",
        "manifest.json",
        "README.md",
    ]
    (output_dir / "manifest.sha256").write_text(
        "".join(f"{_sha256(output_dir / name)}  {name}\n" for name in file_names),
        encoding="ascii",
        newline="\n",
    )
    return manifest


def _verify_sha256_manifest(directory: Path, names: list[str]) -> None:
    """校验目录内明确列出的产物哈希，拒绝不完整或多余的清单行。"""

    sha_path = directory / "manifest.sha256"
    lines = sha_path.read_text(encoding="ascii").splitlines()
    expected = set(names)
    actual: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64 or not all(c in "0123456789abcdef" for c in parts[0]):
            raise HnSemanticReviewClosureError(f"哈希清单格式无效: {sha_path}")
        if parts[1] in actual:
            raise HnSemanticReviewClosureError(f"哈希清单文件重复: {sha_path}/{parts[1]}")
        actual[parts[1]] = parts[0]
    if set(actual) != expected:
        raise HnSemanticReviewClosureError(f"哈希清单文件集合不闭合: {sha_path}")
    for name, expected_hash in actual.items():
        file_path = directory / name
        if not file_path.is_file() or _sha256(file_path) != expected_hash:
            raise HnSemanticReviewClosureError(f"哈希校验失败: {file_path}")


def _audit_final_release(review_root: Path, output_root: Path, *, verify_root_hash: bool) -> dict[str, Any]:
    """逐批重读 source 与最终账本，执行全局发布前身份、覆盖和状态审计。"""

    batch_rows: list[dict[str, Any]] = []
    adversarial_counts: Counter[str] = Counter()
    legal_counts: Counter[str] = Counter()
    package_decisions: Counter[str] = Counter()
    evidence_decisions: Counter[str] = Counter()
    uncertain_rows: list[dict[str, str]] = []
    total_packages = 0
    total_evidence = 0
    batch_files = [
        "adversarial.jsonl",
        "legal-support.jsonl",
        "adjudication.jsonl",
        "reusable-hn-evidence.jsonl",
        "manifest.json",
        "README.md",
    ]
    for batch in range(1, 20):
        source_path = review_root / f"review-batch-{batch:02d}.jsonl"
        batch_dir = output_root / f"batch-{batch:02d}"
        if not batch_dir.is_dir():
            raise HnSemanticReviewClosureError(f"最终批次目录缺失: {batch_dir}")
        _verify_sha256_manifest(batch_dir, batch_files)
        source_rows, source_by_query, source_hash = _source_identity(source_path)
        manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
        review_input = manifest.get("inputs", {}).get("review_batch", {})
        if (
            manifest.get("batch") != batch
            or review_input.get("sha256") != source_hash
            or review_input.get("records") != len(source_rows)
            or manifest.get("policy", {}).get("formal_hn_materialized") is not False
            or manifest.get("policy", {}).get("training_ready") is not False
            or manifest.get("readiness", {}).get("training_ready") is not False
        ):
            raise HnSemanticReviewClosureError(f"批次 manifest 身份或训练状态不闭合: 第 {batch} 批")
        adversarial = _read_jsonl(batch_dir / "adversarial.jsonl", "最终对抗审阅")
        legal = _read_jsonl(batch_dir / "legal-support.jsonl", "最终法律支持审阅")
        if len(adversarial) != len(source_rows) or len(legal) != len(source_rows):
            raise HnSemanticReviewClosureError(f"最终双审 package 数不闭合: 第 {batch} 批")
        adv_by_query = {row.get("query_id"): row for row in adversarial}
        legal_by_query = {row.get("query_id"): row for row in legal}
        if (
            len(adv_by_query) != len(adversarial)
            or len(legal_by_query) != len(legal)
            or set(adv_by_query) != set(source_by_query)
            or set(legal_by_query) != set(source_by_query)
        ):
            raise HnSemanticReviewClosureError(f"最终双审 package 身份不闭合: 第 {batch} 批")
        expected_evidence: dict[tuple[str, str], dict[str, Any]] = {}
        expected_packages = set(source_by_query)
        for query_id, package in source_by_query.items():
            adv_row = adv_by_query[query_id]
            legal_row = legal_by_query[query_id]
            if (
                adv_row.get("record_type") != "package_adversarial_review"
                or legal_row.get("record_type") != "package_legal_support_review"
                or adv_row.get("variant_id") != package["variant_id"]
                or legal_row.get("variant_id") != package["variant_id"]
                or adv_row.get("source_batch") != source_path.name
                or legal_row.get("source_batch") != source_path.name
                or adv_row.get("source_batch_sha256") != source_hash
                or legal_row.get("source_batch_sha256") != source_hash
            ):
                raise HnSemanticReviewClosureError(f"最终双审 source/query/variant 不闭合: 第 {batch} 批/{query_id}")
            adv_items = adv_row.get("non_gt_reviews")
            legal_items = legal_row.get("reviews")
            if not isinstance(adv_items, list) or not isinstance(legal_items, list):
                raise HnSemanticReviewClosureError(f"最终双审 evidence 列表无效: 第 {batch} 批/{query_id}")
            adv_map = {item.get("evidence_id"): item for item in adv_items}
            legal_map = {item.get("evidence_id"): item for item in legal_items}
            source_ids = [item["evidence_id"] for item in package["non_gt"]]
            if (
                len(adv_map) != len(adv_items)
                or len(legal_map) != len(legal_items)
                or set(adv_map) != set(source_ids)
                or set(legal_map) != set(source_ids)
            ):
                raise HnSemanticReviewClosureError(f"最终双审 evidence 覆盖不闭合: 第 {batch} 批/{query_id}")
            for item in package["non_gt"]:
                evidence_id = item["evidence_id"]
                adv_item = adv_map[evidence_id]
                legal_item = legal_map[evidence_id]
                if adv_item.get("chunk_id") != item["chunk_id"] or legal_item.get("chunk_id") != item["chunk_id"]:
                    raise HnSemanticReviewClosureError(f"最终双审 chunk 身份不闭合: 第 {batch} 批/{query_id}/{evidence_id}")
                adv_label = adv_item.get("label")
                support_label = legal_item.get("label")
                if adv_label not in _ADVERSARIAL_LABELS or support_label not in _LEGAL_SUPPORT_LABELS:
                    raise HnSemanticReviewClosureError(f"最终双审标签无效: 第 {batch} 批/{query_id}/{evidence_id}")
                adversarial_counts[adv_label] += 1
                legal_counts[support_label] += 1
                expected_evidence[(query_id, evidence_id)] = {
                    "variant_id": package["variant_id"],
                    "chunk_id": item["chunk_id"],
                    "adversarial_label": adv_label,
                    "legal_support": support_label,
                }
                if adv_label == "uncertain" or support_label == "uncertain":
                    uncertain_rows.append(
                        {
                            "batch": str(batch),
                            "query_id": query_id,
                            "evidence_id": evidence_id,
                            "chunk_id": item["chunk_id"],
                            "adversarial_label": str(adv_label),
                            "legal_support": str(support_label),
                        }
                    )
        adjudication = _read_jsonl(batch_dir / "adjudication.jsonl", "最终裁决")
        package_adjudications = [row for row in adjudication if row.get("record_type") == "package_adjudication"]
        evidence_adjudications = [row for row in adjudication if row.get("record_type") == "evidence_adjudication"]
        if len(package_adjudications) + len(evidence_adjudications) != len(adjudication):
            raise HnSemanticReviewClosureError(f"裁决记录类型无效: 第 {batch} 批")
        package_map = {row.get("query_id"): row for row in package_adjudications}
        evidence_map = {(row.get("query_id"), row.get("evidence_id")): row for row in evidence_adjudications}
        if (
            len(package_map) != len(package_adjudications)
            or len(evidence_map) != len(evidence_adjudications)
            or set(package_map) != expected_packages
            or set(evidence_map) != set(expected_evidence)
        ):
            raise HnSemanticReviewClosureError(f"裁决覆盖不闭合: 第 {batch} 批")
        by_query_adjudication: dict[str, list[dict[str, Any]]] = {query_id: [] for query_id in expected_packages}
        for key, expected in expected_evidence.items():
            query_id, evidence_id = key
            row = evidence_map[key]
            approved = expected["adversarial_label"] == "hard_negative" and expected["legal_support"] == "none"
            if (
                row.get("source_batch") != source_path.name
                or row.get("source_batch_sha256") != source_hash
                or row.get("variant_id") != expected["variant_id"]
                or row.get("chunk_id") != expected["chunk_id"]
                or row.get("adversarial_label") != expected["adversarial_label"]
                or row.get("legal_support") != expected["legal_support"]
                or row.get("decision") != ("approve_evidence" if approved else "exclude_evidence")
            ):
                raise HnSemanticReviewClosureError(f"裁决身份或准入规则不闭合: 第 {batch} 批/{query_id}/{evidence_id}")
            evidence_decisions[row["decision"]] += 1
            by_query_adjudication[query_id].append(row)
        for query_id, package_row in package_map.items():
            evidence_for_package = by_query_adjudication[query_id]
            approved_ids = [row["evidence_id"] for row in evidence_for_package if row["decision"] == "approve_evidence"]
            blockers = [
                row["evidence_id"]
                for row in evidence_for_package
                if row["adversarial_label"] == "redundant_support" or row["legal_support"] != "none"
            ]
            expected_decision = "approve_candidate" if approved_ids and not blockers else "exclude_candidate"
            if (
                package_row.get("source_batch") != source_path.name
                or package_row.get("source_batch_sha256") != source_hash
                or package_row.get("variant_id") != source_by_query[query_id]["variant_id"]
                or package_row.get("decision") != expected_decision
                or package_row.get("approved_evidence_ids") != (approved_ids if expected_decision == "approve_candidate" else [])
            ):
                raise HnSemanticReviewClosureError(f"包级裁决不闭合: 第 {batch} 批/{query_id}")
            package_decisions[expected_decision] += 1
        reusable = _read_jsonl(batch_dir / "reusable-hn-evidence.jsonl", "可复用 HN 账本")
        for row in reusable:
            key = (row.get("query_id"), row.get("evidence_id"))
            if key not in evidence_map or evidence_map[key].get("decision") != "approve_evidence":
                raise HnSemanticReviewClosureError(f"可复用证据无法追溯批准裁决: 第 {batch} 批")
        total_packages += len(source_rows)
        total_evidence += len(expected_evidence)
        batch_rows.append(
            {
                "batch": batch,
                "source_sha256": source_hash,
                "packages": len(source_rows),
                "non_gt_evidence": len(expected_evidence),
                "approved_packages": sum(row["decision"] == "approve_candidate" for row in package_adjudications),
                "approved_evidence": sum(row["decision"] == "approve_evidence" for row in evidence_adjudications),
                "reusable_evidence": len(reusable),
            }
        )
    if total_packages != 452 or total_evidence != 880:
        raise HnSemanticReviewClosureError(f"全局 source 数量异常: packages={total_packages}, non_gt={total_evidence}")
    root_manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    if (
        root_manifest.get("formal_hn_materialized") is not False
        or root_manifest.get("training_ready") is not False
        or root_manifest.get("records", {}).get("packages") != total_packages
        or root_manifest.get("records", {}).get("non_gt_evidence") != total_evidence
    ):
        raise HnSemanticReviewClosureError("根 manifest 训练状态或全局计数不闭合")
    if verify_root_hash:
        _verify_sha256_manifest(output_root, ["README.md", "manifest.json", "GLOBAL_AUDIT_REPORT.md"])
    return {
        "batch_rows": batch_rows,
        "packages": total_packages,
        "non_gt_evidence": total_evidence,
        "adversarial_labels": dict(adversarial_counts),
        "legal_support_labels": dict(legal_counts),
        "package_decisions": dict(package_decisions),
        "evidence_decisions": dict(evidence_decisions),
        "reusable_hn_evidence": sum(row["reusable_evidence"] for row in batch_rows),
        "uncertain_rows": uncertain_rows,
    }


def _write_global_audit_report(output_root: Path, audit: dict[str, Any]) -> None:
    """写出只陈述已由静态和逐行审计验证的全局发布前报告。"""

    adv = audit["adversarial_labels"]
    legal = audit["legal_support_labels"]
    package = audit["package_decisions"]
    evidence = audit["evidence_decisions"]
    lines = [
        "# RAG-SFT v2 真实检索 HN 语义审核全局发布前审计",
        "",
        "本报告仅覆盖 `semantic-review-closure-v2-20260813-final-v1` 内的 19 个最终批次目录。",
        "所有数字均由逐行 JSONL、SHA-256、source batch、query/variant/evidence identity 交叉核验后得出。",
        "",
        "## 结论",
        "",
        f"- 真实检索候选包：{audit['packages']}（恰好一次覆盖）。",
        f"- 非 GT evidence：{audit['non_gt_evidence']}（恰好一次双审和裁决覆盖）。",
        f"- 包级：批准 {package.get('approve_candidate', 0)}，保守排除 {package.get('exclude_candidate', 0)}。",
        f"- 证据级：批准 {evidence.get('approve_evidence', 0)}，排除 {evidence.get('exclude_evidence', 0)}。",
        f"- 可复用证据：{audit['reusable_hn_evidence']}；正式训练 HN 未物化，`training_ready=false`。",
        "",
        "## 标签统计",
        "",
        "| 维度 | 标签 | 数量 |",
        "| --- | --- | ---: |",
    ]
    for label in ["hard_negative", "redundant_support", "irrelevant", "uncertain"]:
        lines.append(f"| 对抗审核 | `{label}` | {adv.get(label, 0)} |")
    for label in ["direct", "partial", "alternative", "none", "uncertain"]:
        lines.append(f"| 法律支持 | `{label}` | {legal.get(label, 0)} |")
    lines.extend(
        [
            "",
            "## 批次身份与闭合",
            "",
            "| 批次 | source SHA-256 | 包 | 非 GT | 批准包 | 批准证据 | 可复用 | 身份/哈希闭合 |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in audit["batch_rows"]:
        lines.append(
            f"| {row['batch']} | `{row['source_sha256']}` | {row['packages']} | {row['non_gt_evidence']} | "
            f"{row['approved_packages']} | {row['approved_evidence']} | {row['reusable_evidence']} | 通过 |"
        )
    lines.extend(
        [
            "",
            "## 隔离与训练状态",
            "",
            "- 第 7 批只使用已校正工作区的身份绑定双审；错配历史文件未作为任何最终输入。",
            "- 每批和根 manifest 均为 `formal_hn_materialized=false`、`training_ready=false`。",
            "- 每个可复用账本均已逐行检查；当前均为空，未产生不可追溯的复用证据。",
            "",
            "## 高风险项",
            "",
        ]
    )
    if audit["uncertain_rows"]:
        lines.extend(["以下证据因存在不确定结论而已保守排除；如未来计划物化正式 HN，应先人工复核：", ""])
        lines.extend(["| 批次 | query | evidence | chunk | 对抗 | 法律支持 |", "| ---: | --- | --- | --- | --- | --- |"])
        for row in audit["uncertain_rows"]:
            lines.append(
                f"| {row['batch']} | `{row['query_id']}` | `{row['evidence_id']}` | `{row['chunk_id']}` | "
                f"`{row['adversarial_label']}` | `{row['legal_support']}` |"
            )
    else:
        lines.append("无仍需人工处理的未决项。")
    (output_root / "GLOBAL_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _publish_final_release(review_root: Path, output_root: Path) -> dict[str, Any]:
    """发布唯一的 19 批最终闭合目录，并在写入后重复执行全局审计。"""

    label_root = review_root / "semantic-closure-inputs"
    label_by_batch: dict[int, Path] = {
        **{batch: label_root / "batch-01-04-09-10-labels.jsonl" for batch in [1, 2, 3, 4, 9, 10]},
        **{batch: label_root / "batch-11-14-labels.jsonl" for batch in [11, 12, 13, 14]},
        **{batch: label_root / "batch-15-19-labels.jsonl" for batch in [15, 16, 17, 18, 19]},
    }
    manifests: list[dict[str, Any]] = []
    for batch in range(1, 20):
        source_path = review_root / f"review-batch-{batch:02d}.jsonl"
        if batch in label_by_batch:
            adversarial, legal, provenance = _normalize_label_reviews(
                source_path=source_path, label_path=label_by_batch[batch], batch=batch
            )
        else:
            historical_dir = review_root / ("batch-07-corrected" if batch == 7 else f"batch-{batch:02d}-reviews")
            adversarial, legal, provenance = _normalize_historical_reviews(
                source_path=source_path, historical_dir=historical_dir
            )
        manifests.append(
            _write_final_batch(
                source_path=source_path,
                adversarial=adversarial,
                legal=legal,
                review_provenance=provenance,
                output_dir=output_root / f"batch-{batch:02d}",
                batch=batch,
            )
        )
    root_manifest = {
        "pipeline": "rag_sft_v2_hn_semantic_review_closure",
        "release_status": "all_19_batches_semantic_review_closed",
        "records": {"batches": 19, "packages": 452, "non_gt_evidence": 880},
        "batch_manifests": [
            {
                "batch": item["batch"],
                "source_batch_sha256": item["inputs"]["review_batch"]["sha256"],
                "manifest": _identity(output_root / f"batch-{item['batch']:02d}" / "manifest.json"),
            }
            for item in manifests
        ],
        "identity_quarantine": {
            "batch": 7,
            "enforced": True,
            "statement": "第 7 批仅读取已校正工作区；错配历史产物未参与最终发布。",
        },
        "policy": {
            "source_candidates_modified": False,
            "formal_hn_materialized": False,
            "training_ready": False,
            "quarantined_batch7_legacy_not_referenced": True,
        },
        "formal_hn_materialized": False,
        "training_ready": False,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(root_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (output_root / "README.md").write_text(
        "# RAG-SFT v2 真实检索 HN 语义审核最终闭合\n\n"
        "本目录是 19 个真实检索 HN 候选的唯一最终语义审核入口。每个批次均包含独立对抗审阅、"
        "独立法律支持审阅、主审裁决、可解析的可复用证据账本、README、manifest 与 SHA-256。\n\n"
        "本发布只读取冻结 source batch 与已完成的双审记录；没有修改 source candidates，"
        "没有重建索引、下载模型或运行检索评估。第 7 批已执行身份隔离，只使用校正后的工作区。"
        "本目录不物化正式训练 HN，`formal_hn_materialized=false`，`training_ready=false`。\n\n"
        "见 `GLOBAL_AUDIT_REPORT.md` 获取覆盖、标签、准入、身份和哈希审计结论。\n",
        encoding="utf-8",
        newline="\n",
    )
    audit = _audit_final_release(review_root, output_root, verify_root_hash=False)
    _write_global_audit_report(output_root, audit)
    (output_root / "manifest.sha256").write_text(
        "".join(
            f"{_sha256(output_root / name)}  {name}\n"
            for name in ["README.md", "manifest.json", "GLOBAL_AUDIT_REPORT.md"]
        ),
        encoding="ascii",
        newline="\n",
    )
    final_audit = _audit_final_release(review_root, output_root, verify_root_hash=True)
    _write_global_audit_report(output_root, final_audit)
    return final_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch", type=int, action="append")
    parser.add_argument(
        "--label-input", type=Path,
        help="包含逐证据独立双审标签的 JSONL；仅支持一个批次。",
    )
    parser.add_argument(
        "--publish-final",
        action="store_true",
        help="发布全部 19 批的统一最终闭合目录并执行全局审计。",
    )
    args = parser.parse_args()
    if args.publish_final == (args.batch is not None):
        parser.error("必须且只能指定 --publish-final 或至少一个 --batch")
    if args.label_input is not None and (args.publish_final or len(args.batch) != 1):
        parser.error("--label-input 只能与一个 --batch 同时使用")
    review_root = args.review_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise SystemExit(f"[失败] 输出目录必须不存在: {output_root}")
    output_root.mkdir(parents=True)
    try:
        if args.publish_final:
            audit = _publish_final_release(review_root, output_root)
            print(
                json.dumps(
                    {
                        "batches": 19,
                        "packages": audit["packages"],
                        "non_gt_evidence": audit["non_gt_evidence"],
                        "training_ready": False,
                    },
                    ensure_ascii=False,
                )
            )
            return
        manifests = []
        for batch in args.batch:
            if args.label_input is not None:
                manifests.append(
                    _close_label_input(
                        source_path=review_root / f"review-batch-{batch:02d}.jsonl",
                        label_path=args.label_input.resolve(),
                        output_dir=output_root / f"batch-{batch:02d}",
                        batch=batch,
                    )
                )
                continue
            historical = (
                review_root / "batch-07-corrected"
                if batch == 7
                else review_root / f"batch-{batch:02d}-reviews"
            )
            manifests.append(
                _close_batch(
                    source_path=review_root / f"review-batch-{batch:02d}.jsonl",
                    adversarial_path=historical / "adversarial.jsonl",
                    legal_path=historical / "legal-support.jsonl",
                    output_dir=output_root / f"batch-{batch:02d}",
                    batch=batch,
                )
            )
        root_manifest = {
            "pipeline": "rag_sft_v2_hn_semantic_review_closure",
            "release_status": "partial_closure",
            "batches": [item["batch"] for item in manifests],
            "formal_hn_materialized": False,
            "training_ready": False,
        }
        (output_root / "manifest.json").write_text(
            json.dumps(root_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (output_root / "manifest.sha256").write_text(
            f"{_sha256(output_root / 'manifest.json')}  manifest.json\n",
            encoding="ascii",
            newline="\n",
        )
    except Exception:
        raise
        print(json.dumps({"batches": args.batch, "training_ready": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
