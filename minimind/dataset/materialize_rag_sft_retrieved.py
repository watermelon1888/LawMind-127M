"""把真实检索包确定性物化为 provisional retrieved RAG-SFT authoring。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rag.answering import (
    EvidencePackager,
    EvidencePackagingError,
    PromptTokenCountError,
    RAG_MAX_OUTPUT_TOKENS,
)
from rag.retrieval import RankedArticle

try:
    from . import audit_disc_law_sft as base_auditor
    from . import audit_rag_sft_chat_lengths as rag_auditor
    from . import audit_sft_chat_lengths as length_auditor
    from . import derive_rag_sft_768 as derivation
    from . import prepare_rag_sft as preparation
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import audit_disc_law_sft as base_auditor
    from dataset import audit_rag_sft_chat_lengths as rag_auditor
    from dataset import audit_sft_chat_lengths as length_auditor
    from dataset import derive_rag_sft_768 as derivation
    from dataset import prepare_rag_sft as preparation


AUTHORING_FILENAME = "rag-sft-retrieved-provisional.jsonl"
LOCATOR_FILENAME = "rag-sft-retrieved-materialization-locators.jsonl"
REPORT_FILENAME = "rag-sft-retrieved-materialization.json"
HASH_FILENAME = "rag-sft-retrieved-materialization.sha256"

CONTEXT_LIMIT = 768
RETRIEVED_ID_OFFSET = 5000


class RagSftRetrievedMaterializationError(RuntimeError):
    """真实检索结果无法安全物化为 RAG-SFT authoring。"""


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftRetrievedMaterializationError(
            f"无法读取{description}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise RagSftRetrievedMaterializationError(f"{description}必须是 JSON object")
    return value


def _verify_bound_file(metadata: object, description: str) -> Path:
    if not isinstance(metadata, dict):
        raise RagSftRetrievedMaterializationError(f"缺少{description}身份")
    path_value = metadata.get("path")
    expected_records = metadata.get("records")
    if (
        not isinstance(path_value, str)
        or type(metadata.get("bytes")) is not int
        or not isinstance(metadata.get("sha256"), str)
        or (expected_records is not None and type(expected_records) is not int)
    ):
        raise RagSftRetrievedMaterializationError(f"{description}身份字段无效")
    path = Path(path_value).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != metadata["bytes"]
        or base_auditor.sha256_file(path) != metadata["sha256"]
    ):
        raise RagSftRetrievedMaterializationError(f"{description}身份已变化")
    return path


def _load_inputs(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, object], list[str], dict[str, dict[str, object]]]:
    manifest_path = manifest_path.resolve()
    try:
        manifest_identity = length_auditor._verify_adjacent_hash(manifest_path)
    except length_auditor.ChatLengthAuditError as error:
        raise RagSftRetrievedMaterializationError(
            "固定 768 manifest 哈希清单无效"
        ) from error
    manifest = _load_json(manifest_path, "固定 768 RAG-SFT manifest")
    policy = manifest.get("policy")
    records = manifest.get("records")
    output = manifest.get("output")
    if (
        manifest.get("pipeline") != derivation.PIPELINE
        or manifest.get("complete") is not True
        or not isinstance(policy, dict)
        or policy.get("fixed_max_seq_len") != CONTEXT_LIMIT
        or policy.get("truncation") != "forbidden"
        or policy.get("selector") != "disabled"
        or not isinstance(records, dict)
        or type(records.get("candidate")) is not int
        or not isinstance(output, dict)
    ):
        raise RagSftRetrievedMaterializationError("固定 768 manifest 状态或策略无效")

    authoring_path = _verify_bound_file(manifest.get("authoring"), "canonical authoring")
    candidate_metadata = output.get("candidate")
    candidate_path = _verify_bound_file(candidate_metadata, "固定 768 candidate")
    authoring = rag_auditor._load_authoring(authoring_path)

    eligible_ids: list[str] = []
    seen_ids: set[str] = set()
    try:
        with candidate_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagSftRetrievedMaterializationError(
                        f"固定 768 candidate 不允许空行: {line_number}"
                    )
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RagSftRetrievedMaterializationError(
                        f"固定 768 candidate JSON 无效: {line_number}"
                    ) from error
                record_id = record.get("id") if isinstance(record, dict) else None
                if (
                    not isinstance(record, dict)
                    or set(record) != rag_auditor.REQUIRED_RECORD_FIELDS
                    or not isinstance(record_id, str)
                    or record_id in seen_ids
                    or record_id not in authoring
                    or record.get("source") != "rag_sft"
                    or record.get("evidence_source")
                    != authoring[record_id]["evidence_source"]
                    or authoring[record_id]["review_status"] != "approved"
                ):
                    raise RagSftRetrievedMaterializationError(
                        f"固定 768 candidate 身份无效: {line_number}"
                    )
                seen_ids.add(record_id)
                eligible_ids.append(record_id)
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftRetrievedMaterializationError(
            "无法读取固定 768 candidate"
        ) from error
    if (
        len(eligible_ids) != records["candidate"]
        or len(eligible_ids) != candidate_metadata.get("records")
    ):
        raise RagSftRetrievedMaterializationError("固定 768 candidate 记录数不闭合")
    return manifest, manifest_identity, eligible_ids, authoring


def _derived_id(source_id: str) -> str:
    try:
        number = int(source_id.split(":", 1)[1]) + RETRIEVED_ID_OFFSET
    except (IndexError, ValueError) as error:
        raise RagSftRetrievedMaterializationError(
            f"无法从源 ID 派生 retrieved ID: {source_id}"
        ) from error
    if number > 9999:
        raise RagSftRetrievedMaterializationError("retrieved ID 超出四位 schema")
    return f"rag_sft:{number:04d}"


def _signature(record: dict[str, object]) -> tuple[object, ...]:
    target = record["target"]
    return (
        record["query_original"],
        tuple(record["visible_chunk_ids"]),
        target["summary"],
        target["refuse"],
    )


def _retrieval_diagnostics(ranked: tuple[RankedArticle, ...]) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": item.article.chunk_id,
            "rrf_rank": item.rrf_rank,
            "rrf_score": item.rrf_score,
            "rerank_score": item.rerank_score,
        }
        for item in ranked
    ]


def _normalize_retrieval_identity(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not value:
        raise TypeError("retrieval_identity 必须是非空 JSON object")
    try:
        payload = json.dumps(
            value, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        normalized = json.loads(payload)
    except (TypeError, ValueError) as error:
        raise TypeError("retrieval_identity 必须可序列化为严格 JSON") from error
    return normalized


def _record_from_package(
    *,
    source_id: str,
    parent: dict[str, object],
    visible_chunk_ids: tuple[str, ...],
    target: dict[str, object],
    review_status: str,
    review_notes: str,
) -> dict[str, object]:
    result = {
        "id": _derived_id(source_id),
        "query_original": parent["query_original"],
        "evidence_source": "retrieved",
        "visible_chunk_ids": list(visible_chunk_ids),
        "required_chunk_ids": list(parent["required_chunk_ids"]),
        "target": target,
        "support_spans": (
            list(parent["support_spans"]) if not target["refuse"] else []
        ),
        "review_status": review_status,
        "review_notes": review_notes,
    }
    preparation._validate_record_shape(result, source_id)
    return result


def _publish(output_dir: Path, records: list[dict[str, object]], locators, report) -> None:
    if output_dir.exists():
        raise RagSftRetrievedMaterializationError(
            f"物化输出目录必须是新目录: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    authoring_payload = "".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for item in records
    )
    locator_payload = "".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for item in locators
    )
    report_payload = json.dumps(
        report, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    payloads = (
        (AUTHORING_FILENAME, authoring_payload),
        (LOCATOR_FILENAME, locator_payload),
        (REPORT_FILENAME, report_payload),
    )
    partials = []
    published = []
    try:
        for filename, payload in payloads:
            final = output_dir / filename
            partial = output_dir / f"{filename}.partial"
            partial.write_text(payload, encoding="utf-8", newline="\n")
            partials.append((partial, final))
        hash_payload = "".join(
            f"{base_auditor.sha256_file(partial)}  {final.name}\n"
            for partial, final in partials
        )
        hash_partial = output_dir / f"{HASH_FILENAME}.partial"
        hash_partial.write_text(hash_payload, encoding="utf-8", newline="\n")
        for partial, final in partials:
            partial.replace(final)
            published.append(final)
        hash_partial.replace(output_dir / HASH_FILENAME)
        published.append(output_dir / HASH_FILENAME)
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
        raise RagSftRetrievedMaterializationError(
            "无法发布 retrieved 物化产物"
        ) from error


def materialize_rag_sft_retrieved(
    *,
    manifest_768_path: Path,
    retriever,
    evidence_packager: EvidencePackager,
    retrieval_identity: dict[str, object],
    output_dir: Path,
) -> dict[str, object]:
    """逐条执行注入的真实检索链，并发布 retrieved authoring 草稿。"""

    if not callable(getattr(retriever, "search", None)):
        raise TypeError("retriever 必须提供 search")
    if not isinstance(evidence_packager, EvidencePackager):
        raise TypeError("evidence_packager 必须是 EvidencePackager")
    if (
        evidence_packager.context_limit != CONTEXT_LIMIT
        or evidence_packager.max_output_tokens != RAG_MAX_OUTPUT_TOKENS
    ):
        raise ValueError("retrieved 物化必须使用固定 768 上下文和 160 输出预算")
    retrieval_identity = _normalize_retrieval_identity(retrieval_identity)
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RagSftRetrievedMaterializationError(
            f"物化输出目录必须是新目录: {output_dir}"
        )

    manifest, manifest_identity, eligible_ids, authoring = _load_inputs(
        Path(manifest_768_path)
    )
    derived_ids = []
    for source_id in eligible_ids:
        derived_id = _derived_id(source_id)
        if derived_id in authoring:
            raise RagSftRetrievedMaterializationError(
                f"retrieved 派生 ID 与 canonical 冲突: {derived_id}"
            )
        derived_ids.append(derived_id)
    if len(set(derived_ids)) != len(derived_ids):
        raise RagSftRetrievedMaterializationError("retrieved 派生 ID 不唯一")
    existing_signatures = {_signature(record) for record in authoring.values()}
    output_signatures = set()
    outputs: list[dict[str, object]] = []
    locators: list[dict[str, object]] = []
    disposition_counts: Counter[str] = Counter()

    for source_id in eligible_ids:
        parent = authoring[source_id]
        derived_id = _derived_id(source_id)
        ranked: tuple[RankedArticle, ...] = ()
        package_ids: tuple[str, ...] = ()
        extra_ids: tuple[str, ...] = ()
        prompt_tokens = None
        diagnostic_code = None
        output_record = None
        manual_review_required = False
        try:
            ranked = tuple(retriever.search(parent["query_original"]))
            if any(not isinstance(item, RankedArticle) for item in ranked):
                raise TypeError("检索结果必须由 RankedArticle 组成")
        except Exception:
            ranked = ()
            disposition = "retrieval_failed"
            diagnostic_code = "semantic_retrieval_failed"
        else:
            if not ranked:
                disposition = "empty_retrieval"
            else:
                try:
                    package, prompt_tokens = evidence_packager.build(
                        parent["query_original"],
                        tuple(item.article for item in ranked),
                    )
                except (EvidencePackagingError, PromptTokenCountError):
                    disposition = "packaging_failed"
                    diagnostic_code = "evidence_packaging_failed"
                else:
                    package_ids = tuple(
                        item.article.chunk_id for item in ranked[: len(package.evidence)]
                    )
                    required = set(parent["required_chunk_ids"])
                    extra_ids = tuple(
                        chunk_id for chunk_id in package_ids if chunk_id not in required
                    )
                    if not required.issubset(package_ids):
                        target = {"summary": "", "refuse": True}
                        candidate = _record_from_package(
                            source_id=source_id,
                            parent=parent,
                            visible_chunk_ids=package_ids,
                            target=target,
                            review_status="draft",
                            review_notes=(
                                f"由 {source_id} 的真实检索包自动物化；"
                                "缺少必要 GT，需人工复核其他证据是否仍可共同支持回答。"
                            ),
                        )
                        signature = _signature(candidate)
                        if signature in existing_signatures:
                            disposition = "duplicate_existing"
                        elif signature in output_signatures:
                            disposition = "duplicate_retrieved"
                        else:
                            disposition = "retrieved_refusal_draft"
                            output_record = candidate
                            manual_review_required = True
                    elif not parent["target"]["summary"]:
                        disposition = "needs_positive_summary"
                        manual_review_required = True
                    else:
                        target = {
                            "summary": parent["target"]["summary"],
                            "refuse": False,
                        }
                        review_status = "draft" if extra_ids else "approved"
                        candidate = _record_from_package(
                            source_id=source_id,
                            parent=parent,
                            visible_chunk_ids=package_ids,
                            target=target,
                            review_status=review_status,
                            review_notes=(
                                f"由 {source_id} 的真实检索包自动物化；"
                                + (
                                    "含非 GT，需逐条确认其不能支持或扭曲 summary。"
                                    if extra_ids
                                    else "全部必要 GT 入包且无额外证据，沿用已审核 summary。"
                                )
                            ),
                        )
                        signature = _signature(candidate)
                        if signature in existing_signatures:
                            disposition = "duplicate_existing"
                        elif signature in output_signatures:
                            disposition = "duplicate_retrieved"
                        else:
                            disposition = (
                                "retrieved_answer_draft"
                                if extra_ids
                                else "retrieved_answer_approved"
                            )
                            output_record = candidate
                            manual_review_required = bool(extra_ids)

        if output_record is not None:
            signature = _signature(output_record)
            if signature in output_signatures:
                raise RagSftRetrievedMaterializationError(
                    f"retrieved 输出 conversations 重复: {source_id}"
                )
            output_signatures.add(signature)
            outputs.append(output_record)
        disposition_counts[disposition] += 1
        locators.append(
            {
                "source_id": source_id,
                "derived_id": derived_id,
                "source_evidence_source": parent["evidence_source"],
                "disposition": disposition,
                "manual_review_required": manual_review_required,
                "diagnostic_code": diagnostic_code,
                "required_chunk_ids": list(parent["required_chunk_ids"]),
                "retrieved_candidates": _retrieval_diagnostics(ranked),
                "packaged_chunk_ids": list(package_ids),
                "extra_non_gt_chunk_ids": list(extra_ids),
                "prompt_tokens": prompt_tokens,
            }
        )

    output_review_counts = Counter(item["review_status"] for item in outputs)
    output_behavior_counts = Counter(
        "refusal" if item["target"]["refuse"] else "answer" for item in outputs
    )
    if sum(disposition_counts.values()) != len(eligible_ids):
        raise RagSftRetrievedMaterializationError("物化 disposition 数量不闭合")
    report: dict[str, object] = {
        "pipeline": "rag_sft_retrieved_materialization",
        "scope": {
            "context_limit": CONTEXT_LIMIT,
            "max_output_tokens": RAG_MAX_OUTPUT_TOKENS,
            "eligible_records": len(eligible_ids),
            "id_mapping": f"rag_sft:NNNN + {RETRIEVED_ID_OFFSET}",
            "retrieval_order_preserved": True,
            "selector": "disabled",
            "input_files_modified": False,
        },
        "input": {
            "manifest_768": manifest_identity,
            "candidate": manifest["output"]["candidate"],
            "authoring": manifest["authoring"],
            "retrieval": retrieval_identity,
        },
        "records": {
            "dispositions": dict(sorted(disposition_counts.items())),
            "output": len(outputs),
            "output_by_review_status": dict(sorted(output_review_counts.items())),
            "output_by_behavior": dict(sorted(output_behavior_counts.items())),
            "manual_review_required": sum(
                bool(item["manual_review_required"]) for item in locators
            ),
        },
        "validation": {
            "scanned_records": len(locators),
            "counts_closed": sum(disposition_counts.values()) == len(eligible_ids),
            "output_ids_unique": len({item["id"] for item in outputs}) == len(outputs),
            "output_signatures_unique": len(output_signatures) == len(outputs),
            "no_source_id_reused": True,
        },
        "readiness": {
            "retrieved_materialization_complete": True,
            "all_manual_reviews_complete": not any(
                item["manual_review_required"] for item in locators
            ),
            "real_retrieval_executed": True,
            "training_ready": False,
        },
        "limitations": [
            "物化器不生成正向 summary；缺少已审核 summary 的充分证据记录进入 needs_positive_summary。",
            "含非 GT 的回答和缺 GT 的拒答必须人工复核法律语义充分性。",
            "空检索、检索故障和构包故障只记录诊断，不构造模型拒答。",
        ],
        "outputs": {
            "authoring": AUTHORING_FILENAME,
            "locators": LOCATOR_FILENAME,
            "report": REPORT_FILENAME,
            "sha256_manifest": HASH_FILENAME,
        },
        "complete": True,
    }
    _publish(output_dir, outputs, locators, report)
    return report


__all__ = [
    "AUTHORING_FILENAME",
    "HASH_FILENAME",
    "LOCATOR_FILENAME",
    "REPORT_FILENAME",
    "RagSftRetrievedMaterializationError",
    "materialize_rag_sft_retrieved",
]
