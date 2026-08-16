"""在固定 768 tokens 下审计 canonical RAG-SFT 的完整训练序列。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rag.answering import (
    ASSISTANT_SCHEMA,
    SYSTEM_PROMPT,
    AnswerProtocolError,
    EvidencePackage,
    parse_and_validate_answer,
)
from rag.core import Evidence

try:
    from . import audit_disc_law_sft as auditor
    from . import audit_sft_chat_lengths as length_auditor
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import audit_disc_law_sft as auditor
    from dataset import audit_sft_chat_lengths as length_auditor


RAG_SFT_ROOT = Path(__file__).resolve().parent / "RAG-SFT"
DEFAULT_RAG_MANIFEST = (
    RAG_SFT_ROOT / "manifests" / "rag-sft-canonical-v1-candidate.json"
)
DEFAULT_TOKENIZER_PATH = auditor.DEFAULT_TOKENIZER_PATH
DEFAULT_OUTPUT_DIR = (
    RAG_SFT_ROOT / "reports" / "rag-sft-canonical-v1-chat-length-audit-768"
)

REPORT_FILENAME = "rag-sft-chat-length-audit-768.json"
OVERFLOW_FILENAME = "rag-sft-chat-length-overflow-768.jsonl"
HASH_FILENAME = "rag-sft-chat-length-audit-768.sha256"

REQUIRED_RECORD_FIELDS = {"id", "source", "evidence_source", "conversations"}
EVIDENCE_SOURCES = {"oracle", "curated_insufficient", "retrieved"}


class RagChatLengthAuditError(RuntimeError):
    """canonical RAG-SFT 长度审计无法安全完成。"""


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagChatLengthAuditError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise RagChatLengthAuditError(f"{description}必须是 JSON object: {path}")
    return value


def _verify_bound_file(metadata: object, description: str) -> tuple[Path, dict[str, object]]:
    if not isinstance(metadata, dict):
        raise RagChatLengthAuditError(f"RAG data manifest 缺少{description}身份")
    path_value = metadata.get("path")
    expected_bytes = metadata.get("bytes")
    expected_sha256 = metadata.get("sha256")
    if (
        not isinstance(path_value, str)
        or type(expected_bytes) is not int
        or not isinstance(expected_sha256, str)
    ):
        raise RagChatLengthAuditError(f"{description}身份字段无效")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise RagChatLengthAuditError(f"{description}不存在: {path}")
    actual = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": auditor.sha256_file(path),
    }
    if actual["bytes"] != expected_bytes or actual["sha256"] != expected_sha256:
        raise RagChatLengthAuditError(f"{description}身份已变化")
    return path, actual


def _verify_manifest(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, object], Path, dict[str, object], Path, dict[str, object]]:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise RagChatLengthAuditError(f"RAG data manifest 不存在: {manifest_path}")
    try:
        manifest_identity = length_auditor._verify_adjacent_hash(manifest_path)
    except length_auditor.ChatLengthAuditError as error:
        raise RagChatLengthAuditError("RAG data manifest 哈希清单无效") from error
    manifest = _load_json(manifest_path, "RAG data manifest")
    if manifest.get("pipeline") != "rag_sft" or manifest.get("complete") is not True:
        raise RagChatLengthAuditError("RAG data manifest pipeline 或完成状态无效")

    protocol = manifest.get("protocol")
    expected_protocol = {
        "hash_algorithm": "sha256",
        "system_prompt_source": "rag.answering.protocol.SYSTEM_PROMPT",
        "system_prompt_sha256": "sha256:"
        + hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "assistant_schema_source": "rag.answering.protocol.ASSISTANT_SCHEMA",
        "assistant_schema_sha256": "sha256:"
        + hashlib.sha256(ASSISTANT_SCHEMA.encode("utf-8")).hexdigest(),
    }
    if protocol != expected_protocol:
        raise RagChatLengthAuditError("RAG data manifest 协议身份与当前代码不一致")

    output = manifest.get("output")
    inputs = manifest.get("inputs")
    if not isinstance(output, dict) or not isinstance(inputs, dict):
        raise RagChatLengthAuditError("RAG data manifest 输入或输出结构无效")
    candidate_path, candidate_identity = _verify_bound_file(
        output.get("candidate"), "RAG candidate"
    )
    authoring_path, authoring_identity = _verify_bound_file(
        inputs.get("authoring"), "canonical authoring"
    )
    records = manifest.get("records")
    if not isinstance(records, dict) or type(records.get("candidate")) is not int:
        raise RagChatLengthAuditError("RAG data manifest 缺少 candidate 记录数")
    return (
        manifest,
        manifest_identity,
        candidate_path,
        candidate_identity,
        authoring_path,
        authoring_identity,
    )


def _load_authoring(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagChatLengthAuditError(
                        f"canonical authoring 不允许空行: {line_number}"
                    )
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RagChatLengthAuditError(
                        f"canonical authoring JSON 无效: {line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise RagChatLengthAuditError(
                        f"canonical authoring 记录必须是对象: {line_number}"
                    )
                record_id = record.get("id")
                target = record.get("target")
                visible = record.get("visible_chunk_ids")
                required = record.get("required_chunk_ids")
                if (
                    not isinstance(record_id, str)
                    or not record_id
                    or record_id in records
                    or record.get("evidence_source") not in EVIDENCE_SOURCES
                    or not isinstance(record.get("query_original"), str)
                    or not record["query_original"]
                    or not isinstance(visible, list)
                    or not visible
                    or any(not isinstance(item, str) or not item for item in visible)
                    or not isinstance(required, list)
                    or not 1 <= len(required) <= 3
                    or any(not isinstance(item, str) or not item for item in required)
                    or not isinstance(target, dict)
                    or not isinstance(target.get("summary"), str)
                    or not isinstance(target.get("refuse"), bool)
                    or record.get("review_status") not in {"draft", "approved"}
                ):
                    raise RagChatLengthAuditError(
                        f"canonical authoring 审计字段无效: {line_number}"
                    )
                records[record_id] = record
    except (OSError, UnicodeDecodeError) as error:
        raise RagChatLengthAuditError(f"无法读取 canonical authoring: {path}") from error
    return records


def _load_compact_json(content: str, description: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
        canonical = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise RagChatLengthAuditError(f"{description}不是严格 JSON") from error
    if not isinstance(value, dict) or canonical != content:
        raise RagChatLengthAuditError(f"{description}不是唯一紧凑 JSON")
    return value


def _validate_conversations(
    record: dict[str, object], authoring: dict[str, object], line_number: int
) -> None:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 3:
        raise RagChatLengthAuditError(
            f"RAG candidate conversations 无效: {line_number}"
        )
    roles = [
        message.get("role") if isinstance(message, dict) else None
        for message in conversations
    ]
    if roles != ["system", "user", "assistant"] or any(
        not isinstance(message, dict)
        or set(message) != {"role", "content"}
        or not isinstance(message.get("content"), str)
        or not message["content"]
        for message in conversations
    ):
        raise RagChatLengthAuditError(
            f"RAG candidate 消息结构无效: {line_number}"
        )
    if conversations[0]["content"] != SYSTEM_PROMPT:
        raise RagChatLengthAuditError(
            f"RAG candidate system prompt 与当前协议不一致: {line_number}"
        )

    user = _load_compact_json(conversations[1]["content"], "RAG candidate user")
    if set(user) != {"query", "evidence"} or user.get("query") != authoring["query_original"]:
        raise RagChatLengthAuditError(
            f"RAG candidate user 与 authoring 不一致: {line_number}"
        )
    evidence_items = user.get("evidence")
    if (
        not isinstance(evidence_items, list)
        or len(evidence_items) != len(authoring["visible_chunk_ids"])
        or not 1 <= len(evidence_items) <= 4
    ):
        raise RagChatLengthAuditError(
            f"RAG candidate evidence 数量无效: {line_number}"
        )

    evidence = []
    for index, item in enumerate(evidence_items, start=1):
        if (
            not isinstance(item, dict)
            or set(item) != {"evidence_id", "law_name", "article_no", "excerpts"}
            or item.get("evidence_id") != f"E{index}"
            or not isinstance(item.get("law_name"), str)
            or not item["law_name"]
            or not isinstance(item.get("article_no"), str)
            or not item["article_no"]
            or not isinstance(item.get("excerpts"), list)
            or len(item["excerpts"]) != 1
            or not isinstance(item["excerpts"][0], str)
            or not item["excerpts"][0]
        ):
            raise RagChatLengthAuditError(
                f"RAG candidate evidence schema 无效: {line_number}"
            )
        evidence.append(
            Evidence(
                law_name=item["law_name"],
                article_no=item["article_no"],
                content=item["excerpts"][0],
            )
        )

    package = EvidencePackage(query=user["query"], evidence=tuple(evidence))
    raw_answer = conversations[2]["content"]
    _load_compact_json(raw_answer, "RAG candidate assistant")
    try:
        answer = parse_and_validate_answer(package, raw_answer)
    except (AnswerProtocolError, TypeError, ValueError) as error:
        raise RagChatLengthAuditError(
            f"RAG candidate assistant 未通过三字段协议: {line_number}"
        ) from error

    target = authoring["target"]
    required = set(authoring["required_chunk_ids"])
    expected_citations = tuple(
        f"E{index}"
        for index, chunk_id in enumerate(authoring["visible_chunk_ids"], start=1)
        if chunk_id in required
    )
    if target["refuse"]:
        expected_citations = ()
    if (
        answer.summary != target["summary"]
        or answer.refuse is not target["refuse"]
        or answer.citations != expected_citations
    ):
        raise RagChatLengthAuditError(
            f"RAG candidate assistant 与 authoring 不一致: {line_number}"
        )


def _load_candidate(
    candidate_path: Path,
    expected_records: int,
    authoring_by_id: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    approved_ids = {
        record_id
        for record_id, record in authoring_by_id.items()
        if record["review_status"] == "approved"
    }
    try:
        with candidate_path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagChatLengthAuditError(
                        f"RAG candidate 不允许空行: {line_number}"
                    )
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RagChatLengthAuditError(
                        f"RAG candidate JSON 无效: {line_number}"
                    ) from error
                if not isinstance(record, dict) or set(record) != REQUIRED_RECORD_FIELDS:
                    raise RagChatLengthAuditError(
                        f"RAG candidate schema 无效: {line_number}"
                    )
                record_id = record.get("id")
                authoring = authoring_by_id.get(record_id) if isinstance(record_id, str) else None
                if (
                    authoring is None
                    or record_id in seen_ids
                    or record.get("source") != "rag_sft"
                    or record.get("evidence_source") != authoring["evidence_source"]
                ):
                    raise RagChatLengthAuditError(
                        f"RAG candidate 身份或来源无效: {line_number}"
                    )
                _validate_conversations(record, authoring, line_number)
                seen_ids.add(record_id)
                records.append(record)
    except (OSError, UnicodeDecodeError) as error:
        raise RagChatLengthAuditError(f"无法读取 RAG candidate: {candidate_path}") from error
    if len(records) != expected_records or seen_ids != approved_ids:
        raise RagChatLengthAuditError("RAG candidate 与 manifest/approved authoring 不闭合")
    return records


def _record_metadata(authoring: dict[str, object]) -> dict[str, object]:
    visible = authoring["visible_chunk_ids"]
    required = authoring["required_chunk_ids"]
    refuse = authoring["target"]["refuse"]
    return {
        "evidence_source": authoring["evidence_source"],
        "behavior": "refusal" if refuse else "answer",
        "required_gt_count": len(required),
        "visible_evidence_count": len(visible),
        "has_hard_negative": bool(not refuse and set(visible) - set(required)),
    }


def _scope_report(scopes: dict[str, length_auditor.ScopeStats]) -> dict[str, object]:
    return {name: scopes[name].to_report() for name in sorted(scopes)}


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    json.loads(path.read_text(encoding="utf-8"))


def audit_rag_sft_chat_lengths(
    rag_manifest: Path,
    tokenizer_path: Path,
    output_dir: Path,
    tokenizer: Any | None = None,
) -> dict[str, object]:
    """审计全部 canonical candidate，不截断、改写或排除训练记录。"""

    rag_manifest = Path(rag_manifest).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RagChatLengthAuditError(f"审计输出目录必须是新目录: {output_dir}")
    (
        manifest,
        manifest_identity,
        candidate_path,
        candidate_identity,
        authoring_path,
        authoring_identity,
    ) = _verify_manifest(rag_manifest)
    if auditor._path_is_within(output_dir, candidate_path.parent):
        raise RagChatLengthAuditError("审计报告目录不能位于 candidate 目录内")
    authoring_by_id = _load_authoring(authoring_path)
    records = _load_candidate(
        candidate_path,
        manifest["records"]["candidate"],
        authoring_by_id,
    )

    tokenizer = tokenizer or auditor.load_tokenizer(tokenizer_path)
    tokenizer_report = length_auditor._tokenizer_identity(tokenizer, tokenizer_path)
    bos_token = getattr(tokenizer, "bos_token", None)
    eos_token = getattr(tokenizer, "eos_token", None)
    if not isinstance(bos_token, str) or not isinstance(eos_token, str):
        raise RagChatLengthAuditError("Tokenizer 缺少 assistant 边界特殊 token")
    assistant_start = length_auditor._encode_marker(tokenizer, f"{bos_token}assistant\n")
    assistant_end = length_auditor._encode_marker(tokenizer, f"{eos_token}\n")

    overall = length_auditor.ScopeStats()
    by_evidence_source: dict[str, length_auditor.ScopeStats] = {}
    by_behavior: dict[str, length_auditor.ScopeStats] = {}
    by_required_gt_count: dict[str, length_auditor.ScopeStats] = {}
    by_hard_negative: dict[str, length_auditor.ScopeStats] = {}
    by_visible_evidence_count: dict[str, length_auditor.ScopeStats] = {}
    metadata_counts: Counter[str] = Counter()
    overflow: list[dict[str, object]] = []
    prompts = [
        length_auditor._render_prompt(tokenizer, record["conversations"])
        for record in records
    ]
    tokenized = length_auditor._encode_prompts(tokenizer, prompts)
    for record, input_ids in zip(records, tokenized):
        metrics = length_auditor.analyze_tokenized_record(
            input_ids, assistant_start, assistant_end
        )
        metadata = _record_metadata(authoring_by_id[record["id"]])
        source = metadata["evidence_source"]
        behavior = metadata["behavior"]
        gt_count = str(metadata["required_gt_count"])
        visible_count = str(metadata["visible_evidence_count"])
        hard_negative = (
            "with_hard_negative"
            if metadata["has_hard_negative"]
            else "without_hard_negative"
        )
        for scope in (
            overall,
            by_evidence_source.setdefault(source, length_auditor.ScopeStats()),
            by_behavior.setdefault(behavior, length_auditor.ScopeStats()),
            by_required_gt_count.setdefault(gt_count, length_auditor.ScopeStats()),
            by_hard_negative.setdefault(hard_negative, length_auditor.ScopeStats()),
            by_visible_evidence_count.setdefault(
                visible_count, length_auditor.ScopeStats()
            ),
        ):
            scope.record(metrics)
        metadata_counts[f"behavior:{behavior}"] += 1
        metadata_counts[f"required_gt_count:{gt_count}"] += 1
        metadata_counts[f"hard_negative:{hard_negative}"] += 1
        if not metrics["within_limit"]:
            risks = ["full_sequence_over_768"]
            if not metrics["assistant_complete"]:
                risks.append("assistant_tail_truncated_at_768")
            if not metrics["candidate_assistant_labels_within_limit"]:
                risks.append("zero_candidate_assistant_labels_at_768")
            overflow.append(
                {
                    "id": record["id"],
                    **metadata,
                    "full_tokens": metrics["full_tokens"],
                    "tokens_over_limit": metrics["tokens_over_limit"],
                    "candidate_assistant_label_tokens": metrics[
                        "candidate_assistant_label_tokens"
                    ],
                    "candidate_assistant_labels_within_limit": metrics[
                        "candidate_assistant_labels_within_limit"
                    ],
                    "risks": risks,
                }
            )

    counts = overall.to_report()["counts"]
    if counts["records"] != len(records) or counts["over_limit"] != len(overflow):
        raise RagChatLengthAuditError("RAG 长度审计汇总与定位记录不闭合")
    all_fit = counts["over_limit"] == 0
    report: dict[str, object] = {
        "pipeline": "rag_sft_chat_length_audit_768",
        "scope": {
            "fixed_max_seq_len": length_auditor.MAX_SEQ_LEN,
            "candidate_records": len(records),
            "group_by": [
                "evidence_source",
                "behavior",
                "required_gt_count",
                "has_hard_negative",
                "visible_evidence_count",
            ],
            "audit_only": True,
            "training_dataset_written": False,
            "input_files_modified": False,
        },
        "input": {
            "rag_manifest": manifest_identity,
            "candidate": {**candidate_identity, "release_status": manifest.get("release_status")},
            "authoring": authoring_identity,
        },
        "tokenizer": tokenizer_report,
        "template": {
            "render": "apply_chat_template_tokenize_false_add_generation_prompt_false",
            "tokenize": "add_special_tokens_false",
            "assistant_start_marker_ids": assistant_start,
            "assistant_end_marker_ids": assistant_end,
            "full_sequence_definition": "system + user + assistant + chat template 边界 + EOS",
        },
        "records": {
            "totals": overall.to_report(),
            "by_evidence_source": _scope_report(by_evidence_source),
            "by_behavior": _scope_report(by_behavior),
            "by_required_gt_count": _scope_report(by_required_gt_count),
            "by_hard_negative": _scope_report(by_hard_negative),
            "by_visible_evidence_count": _scope_report(by_visible_evidence_count),
            "metadata_counts": dict(sorted(metadata_counts.items())),
        },
        "validation": {
            "expected_records": manifest["records"]["candidate"],
            "scanned_records": len(records),
            "approved_authoring_ids_closed": len(records)
            == sum(
                record["review_status"] == "approved"
                for record in authoring_by_id.values()
            ),
            "overflow_records": len(overflow),
            "counts_closed": counts["records"]
            == counts["within_limit"] + counts["over_limit"],
        },
        "decision": {
            "max_seq_len": length_auditor.MAX_SEQ_LEN,
            "policy": "fixed_768_audit_only_no_truncation_no_rewrite_no_selector",
            "overlength_records": "keep_in_authoring_exclude_from_later_candidate",
        },
        "readiness": {
            "chat_template_length_audited": True,
            "all_records_fit_768": all_fit,
            "dataset_label_mask_audited": False,
            "training_ready": False,
        },
        "outputs": {
            "report": REPORT_FILENAME,
            "overflow": {"path": OVERFLOW_FILENAME, "records": len(overflow)},
            "sha256_manifest": HASH_FILENAME,
        },
        "limitations": [
            "本报告不修改、截断、改写、排除或发布训练记录。",
            "本报告只验证长度与确定性投影，不判断 summary 的法律语义正确性。",
            "本报告不替代 assistant-only Dataset label 审计。",
            "本报告不冻结运行时 max_output_tokens，也不执行真实 retrieval。",
        ],
        "complete": True,
    }

    output_dir.mkdir(parents=True)
    report_path = output_dir / REPORT_FILENAME
    overflow_path = output_dir / OVERFLOW_FILENAME
    hash_path = output_dir / HASH_FILENAME
    report_pending = report_path.with_name(report_path.name + ".pending")
    overflow_pending = overflow_path.with_name(overflow_path.name + ".pending")
    hash_pending = hash_path.with_name(hash_path.name + ".pending")
    try:
        with overflow_pending.open("x", encoding="utf-8", newline="\n") as output:
            for locator in overflow:
                output.write(
                    json.dumps(
                        locator,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
        _write_json(report_pending, report)
        hash_pending.write_text(
            f"{auditor.sha256_file(report_pending)}  {REPORT_FILENAME}\n"
            f"{auditor.sha256_file(overflow_pending)}  {OVERFLOW_FILENAME}\n",
            encoding="utf-8",
            newline="\n",
        )
        report_pending.replace(report_path)
        overflow_pending.replace(overflow_path)
        hash_pending.replace(hash_path)
    except (OSError, UnicodeError, ValueError) as error:
        for path in (
            report_pending,
            overflow_pending,
            hash_pending,
            report_path,
            overflow_path,
            hash_path,
        ):
            path.unlink(missing_ok=True)
        raise RagChatLengthAuditError("无法发布 RAG 长度审计产物") from error
    return report


def main() -> None:
    """解析参数并执行 canonical RAG-SFT 固定 768 长度审计。"""

    parser = argparse.ArgumentParser(
        description="审计 canonical RAG-SFT 固定 768 chat template 长度"
    )
    parser.add_argument("--rag-manifest", type=Path, default=DEFAULT_RAG_MANIFEST)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        report = audit_rag_sft_chat_lengths(
            rag_manifest=args.rag_manifest,
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
        )
    except (
        RagChatLengthAuditError,
        length_auditor.ChatLengthAuditError,
        OSError,
        ValueError,
    ) as error:
        raise SystemExit(f"[失败] {error}") from error
    totals = report["records"]["totals"]["counts"]
    print(
        f"[完成] 审计 {totals['records']:,} 条，"
        f"合格 {totals['within_limit']:,} 条，超长 {totals['over_limit']:,} 条"
    )
    print("[阻断] 本程序不发布训练数据，training_ready=false")
    print(f"报告目录: {args.output_dir}")


if __name__ == "__main__":
    main()
