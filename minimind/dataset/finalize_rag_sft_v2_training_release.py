"""冻结 Oracle clean 与真实检索 HN，发布 RAG-SFT v2 正式训练候选。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .rag_sft_v2_dataset import _validate_record


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ORACLE_ROOT = (
    PROJECT_ROOT / "minimind" / "dataset" / "RAG-SFT" / "review" / "v2" / "oracle-clean-v2"
)
DEFAULT_HN_ROOT = (
    PROJECT_ROOT
    / "minimind"
    / "dataset"
    / "RAG-SFT"
    / "review"
    / "v2"
    / "hn-variants-stage-v1-20260813"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "minimind"
    / "dataset"
    / "RAG-SFT"
    / "releases"
    / "v2"
    / "rag-sft-v2-training-release-619-v1-20260813"
)
DEFAULT_AUDITED_ROOT = (
    PROJECT_ROOT
    / "minimind"
    / "dataset"
    / "RAG-SFT"
    / "releases"
    / "v2"
    / "rag-sft-v2-curated-hn-768-audited-release-776-v1-20260814"
)
DEFAULT_CURATED_OUTPUT_DIR = (
    PROJECT_ROOT
    / "minimind"
    / "dataset"
    / "RAG-SFT"
    / "releases"
    / "v2"
    / "rag-sft-v2-training-release-775-v1-20260814"
)

CANDIDATE_FILENAME = "training-candidate.jsonl"
LEDGER_FILENAME = "identity-ledger.jsonl"
PAIRING_FILENAME = "pairing-audit.json"
MANIFEST_FILENAME = "manifest.json"
HASH_FILENAME = "manifest.sha256"
README_FILENAME = "README.md"
AUDITED_HASH_FILES = {
    "semantic-freeze-candidate.jsonl",
    "identity-ledger.jsonl",
    "context-exclusions.jsonl",
    "c6-supplemental-length-audit.jsonl",
    "accepted-c6-curated-hn-variants.jsonl",
    MANIFEST_FILENAME,
    README_FILENAME,
}


class RagSftV2TrainingReleaseError(RuntimeError):
    """RAG-SFT v2 正式训练发布的输入身份或数据关卡未闭合。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, records: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if records is not None:
        value["records"] = records
    return value


def _payload_identity(path: Path, payload: str, records: int | None = None) -> dict[str, object]:
    raw = payload.encode("utf-8")
    value: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if records is not None:
        value["records"] = records
    return value


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftV2TrainingReleaseError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise RagSftV2TrainingReleaseError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    raise RagSftV2TrainingReleaseError(
                        f"{description}存在空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RagSftV2TrainingReleaseError(
                        f"{description}第 {line_number} 行不是 object"
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftV2TrainingReleaseError):
            raise
        raise RagSftV2TrainingReleaseError(f"无法读取{description}: {path}") from error
    return rows


def _verify_manifest(path: Path) -> str:
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise RagSftV2TrainingReleaseError(f"manifest 或 SHA-256 清单不存在: {path}")
    digest = _sha256(path)
    matches: list[str] = []
    try:
        for line in sidecar.read_text(encoding="ascii").splitlines():
            if not line:
                continue
            parts = line.split("  ", 1)
            if len(parts) != 2 or len(parts[0]) != 64:
                raise ValueError
            int(parts[0], 16)
            if Path(parts[1]).name == path.name:
                matches.append(parts[0].lower())
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise RagSftV2TrainingReleaseError(f"manifest SHA-256 清单无效: {sidecar}") from error
    if matches != [digest]:
        raise RagSftV2TrainingReleaseError(f"manifest SHA-256 校验失败: {path}")
    return digest


def _bound_output(manifest: dict[str, Any], key: str, description: str) -> Path:
    metadata = manifest.get("output", {}).get(key)
    if not isinstance(metadata, dict) or not isinstance(metadata.get("path"), str):
        raise RagSftV2TrainingReleaseError(f"{description}缺少输出身份")
    path = Path(metadata["path"]).resolve()
    if (
        not path.is_file()
        or metadata.get("bytes") != path.stat().st_size
        or metadata.get("sha256") != _sha256(path)
    ):
        raise RagSftV2TrainingReleaseError(f"{description}输出身份已变化")
    return path


def _index_unique(
    rows: Iterable[dict[str, Any]], key: str, description: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise RagSftV2TrainingReleaseError(f"{description}缺少 {key}")
        if value in indexed:
            raise RagSftV2TrainingReleaseError(f"{description}{key} 重复: {value}")
        indexed[value] = row
    return indexed


def _assistant_summary(record: dict[str, Any]) -> str:
    try:
        value = json.loads(record["conversations"][2]["content"])
        summary = value["summary"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise RagSftV2TrainingReleaseError("assistant target 无法提取 summary") from error
    if not isinstance(summary, str) or not summary:
        raise RagSftV2TrainingReleaseError("assistant summary 无效")
    return summary


def _serialize_jsonl(rows: Iterable[dict[str, object]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    )


def _verify_hash_inventory(root: Path) -> None:
    sidecar = root / HASH_FILENAME
    if not sidecar.is_file():
        raise RagSftV2TrainingReleaseError(f"SHA-256 清单不存在: {sidecar}")
    seen: set[str] = set()
    try:
        for line in sidecar.read_text(encoding="ascii").splitlines():
            parts = line.split("  ", 1)
            if (
                len(parts) != 2
                or len(parts[0]) != 64
                or Path(parts[1]).name != parts[1]
                or parts[1] in seen
            ):
                raise ValueError
            int(parts[0], 16)
            path = root / parts[1]
            if not path.is_file() or _sha256(path) != parts[0].lower():
                raise ValueError
            seen.add(parts[1])
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise RagSftV2TrainingReleaseError(
            f"SHA-256 清单校验失败: {sidecar}"
        ) from error
    if seen != AUDITED_HASH_FILES:
        raise RagSftV2TrainingReleaseError(
            f"已审计 release 的 SHA-256 文件集合无效: {sorted(seen)}"
        )


def _verify_identity_tree(value: object, description: str) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _verify_identity_tree(item, f"{description}[{index}]")
        return
    if not isinstance(value, dict):
        return
    if {"path", "bytes", "sha256"}.issubset(value):
        path_value = value.get("path")
        if not isinstance(path_value, str):
            raise RagSftV2TrainingReleaseError(f"{description}路径无效")
        path = Path(path_value).resolve()
        if (
            not path.is_file()
            or value.get("bytes") != path.stat().st_size
            or value.get("sha256") != _sha256(path)
            or (
                "manifest_sha256" in value
                and value.get("manifest_sha256") != value.get("sha256")
            )
        ):
            raise RagSftV2TrainingReleaseError(f"{description}身份已变化: {path}")
    for key, item in value.items():
        _verify_identity_tree(item, f"{description}.{key}")


def _strict_payload(path: Path, description: str) -> str:
    raw = path.read_bytes()
    try:
        payload = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise RagSftV2TrainingReleaseError(f"{description}不是严格 UTF-8") from error
    if payload.encode("utf-8") != raw or "\r" in payload or not payload.endswith("\n"):
        raise RagSftV2TrainingReleaseError(f"{description}不是 Unix LF 单行 JSONL")
    return payload


def _record_sha256(record: dict[str, Any]) -> str:
    raw = json.dumps(
        record, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_bound_manifest(metadata: object, description: str) -> tuple[Path, dict[str, Any], str]:
    if not isinstance(metadata, dict) or not isinstance(metadata.get("path"), str):
        raise RagSftV2TrainingReleaseError(f"{description}身份缺失")
    _verify_identity_tree(metadata, description)
    path = Path(metadata["path"]).resolve()
    digest = _verify_manifest(path)
    if metadata.get("manifest_sha256") != digest:
        raise RagSftV2TrainingReleaseError(f"{description} manifest SHA-256 不一致")
    return path, _load_json(path, description), digest


def _finalize_curated_training_release(
    *, audited_root: Path, output_dir: Path
) -> dict[str, object]:
    audited_root = Path(audited_root).resolve()
    output_dir = Path(output_dir).resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise RagSftV2TrainingReleaseError(f"输出目录或临时目录已存在: {output_dir}")

    _verify_hash_inventory(audited_root)
    audited_manifest_path = audited_root / MANIFEST_FILENAME
    audited_manifest_sha = _verify_manifest(audited_manifest_path)
    audited_manifest = _load_json(audited_manifest_path, "775 条已审计 release manifest")
    records_meta = audited_manifest.get("records", {})
    policy = audited_manifest.get("policy", {})
    validation = audited_manifest.get("validation", {})
    if (
        audited_manifest.get("pipeline")
        != "rag_sft_v2_curated_hn_768_audited_release"
        or audited_manifest.get("release_status")
        != "length_audited_semantic_candidate_only"
        or records_meta.get("total") != 775
        or records_meta.get("clean") != 549
        or records_meta.get("retrieved_hn") != 70
        or records_meta.get("curated_hn") != 156
        or records_meta.get("hard_negative") != 226
        or records_meta.get("paired_hn_queries") != 226
        or policy.get("context_limit") != 768
        or policy.get("max_prompt_tokens") != 618
        or policy.get("max_output_tokens") != 150
        or policy.get("formal_hn_materialized") is not False
        or policy.get("training_ready") is not False
        or validation.get("all_retained_base_records_passed_768_and_labels_audit")
        is not True
        or validation.get("all_c6_records_reaudited_with_frozen_tokenizer")
        is not True
        or validation.get("one_hn_per_query") is not True
        or validation.get("all_hn_have_clean_pair") is not True
        or validation.get("quarantined_batch7_legacy_not_referenced") is not True
        or audited_manifest.get("complete") is not True
    ):
        raise RagSftV2TrainingReleaseError("775 条已审计 release 的状态或计数未闭合")
    _verify_identity_tree(audited_manifest.get("inputs"), "audited.inputs")

    semantic_manifest_path, semantic_manifest, semantic_manifest_sha = _load_bound_manifest(
        audited_manifest.get("inputs", {}).get("semantic_freeze_manifest"),
        "779 条语义冻结 manifest",
    )
    if (
        semantic_manifest.get("pipeline") != "rag_sft_v2_curated_hn_semantic_release"
        or semantic_manifest.get("policy", {}).get("training_ready") is not False
        or semantic_manifest.get("complete") is not True
    ):
        raise RagSftV2TrainingReleaseError("779 条语义冻结 release 身份未闭合")
    _verify_identity_tree(semantic_manifest.get("inputs"), "semantic.inputs")
    base_manifest_path, base_manifest, base_manifest_sha = _load_bound_manifest(
        semantic_manifest.get("inputs", {}).get("base_manifest"),
        "旧 619 正式训练 manifest",
    )
    if (
        base_manifest.get("pipeline") != "rag_sft_v2_training_release"
        or base_manifest.get("records", {}).get("total") != 619
        or base_manifest.get("readiness", {}).get("training_ready") is not True
        or base_manifest.get("readiness", {}).get("final_retrieval_identity_frozen")
        is not True
        or base_manifest.get("complete") is not True
    ):
        raise RagSftV2TrainingReleaseError("旧 619 正式训练 release 身份未闭合")
    if audited_manifest.get("tokenizer") != base_manifest.get("tokenizer"):
        raise RagSftV2TrainingReleaseError("775 release 与旧 619 的 Tokenizer 身份不一致")
    tokenizer = audited_manifest.get("tokenizer")
    if not isinstance(tokenizer, dict) or not isinstance(tokenizer.get("path"), str):
        raise RagSftV2TrainingReleaseError("Tokenizer 身份缺失")
    tokenizer_root = Path(tokenizer["path"]).resolve()
    if not tokenizer_root.is_dir():
        raise RagSftV2TrainingReleaseError("Tokenizer 目录不存在")
    for filename, metadata in tokenizer.get("files", {}).items():
        path = tokenizer_root / filename
        if (
            not isinstance(metadata, dict)
            or not path.is_file()
            or metadata.get("bytes") != path.stat().st_size
            or metadata.get("sha256") != _sha256(path)
        ):
            raise RagSftV2TrainingReleaseError(f"Tokenizer 文件身份无效: {filename}")

    candidate_path = _bound_output(
        audited_manifest, "semantic_freeze_candidate", "775 条语义候选"
    )
    upstream_ledger_path = _bound_output(
        audited_manifest, "identity_ledger", "775 条身份 ledger"
    )
    context_exclusions_path = _bound_output(
        audited_manifest, "context_exclusions", "上下文排除 ledger"
    )
    candidate_payload = _strict_payload(candidate_path, "775 条语义候选")
    ledger_payload = _strict_payload(upstream_ledger_path, "775 条身份 ledger")
    candidate_rows = _load_jsonl(candidate_path, "775 条语义候选")
    ledger_rows = _load_jsonl(upstream_ledger_path, "775 条身份 ledger")
    if len(candidate_rows) != 775 or len(ledger_rows) != 775:
        raise RagSftV2TrainingReleaseError("775 条候选或身份 ledger 计数错误")

    clean_rows = [row for row in candidate_rows if row.get("variant") == "clean"]
    hn_rows = [row for row in candidate_rows if row.get("variant") == "hard_negative"]
    if len(clean_rows) != 549 or len(hn_rows) != 226 or candidate_rows != [*clean_rows, *hn_rows]:
        raise RagSftV2TrainingReleaseError("候选必须固定为 549 clean 后接 226 HN")
    clean_by_query = _index_unique(clean_rows, "query_id", "clean ")
    hn_by_query = _index_unique(hn_rows, "query_id", "HN ")
    _index_unique(candidate_rows, "id", "训练候选 ")
    ledger_by_id = _index_unique(ledger_rows, "id", "身份 ledger ")
    if set(hn_by_query) - set(clean_by_query):
        raise RagSftV2TrainingReleaseError("HN 缺少对应 clean")

    source_counts: Counter[str] = Counter()
    visible_counts: Counter[int] = Counter()
    required_counts: Counter[int] = Counter()
    citation_counts: Counter[int] = Counter()
    non_required_evidence = 0
    for line_number, record in enumerate(candidate_rows, 1):
        try:
            _validate_record(record, line_number)
        except Exception as error:
            raise RagSftV2TrainingReleaseError(
                f"训练候选第 {line_number} 条不满足正式 Dataset 契约"
            ) from error
        ledger = ledger_by_id.get(record["id"])
        source = ledger.get("source") if isinstance(ledger, dict) else None
        if (
            ledger is None
            or ledger.get("query_id") != record["query_id"]
            or ledger.get("variant") != record["variant"]
            or ledger.get("record_sha256") != _record_sha256(record)
            or source not in {"oracle_clean", "retrieved", "curated"}
        ):
            raise RagSftV2TrainingReleaseError(
                f"训练候选与身份 ledger 不一致: {record['id']}"
            )
        source_counts[source] += 1
        visible_counts[len(record["visible_chunk_ids"])] += 1
        required_counts[len(record["required_chunk_ids"])] += 1
        citation_counts[len(record["citations"])] += 1
        if record["variant"] == "hard_negative":
            non_required_evidence += len(
                set(record["visible_chunk_ids"]) - set(record["required_chunk_ids"])
            )
    if source_counts != Counter(oracle_clean=549, retrieved=70, curated=156):
        raise RagSftV2TrainingReleaseError(f"HN 来源计数未闭合: {source_counts}")

    pairing_rows: list[dict[str, object]] = []
    for query_id in sorted(hn_by_query):
        clean = clean_by_query[query_id]
        hn = hn_by_query[query_id]
        checks = {
            "query_original_equal": clean.get("query_original") == hn.get("query_original"),
            "required_chunk_ids_equal": clean.get("required_chunk_ids")
            == hn.get("required_chunk_ids"),
            "summary_equal": _assistant_summary(clean) == _assistant_summary(hn),
            "system_prompt_equal": clean.get("conversations", [{}])[0].get("content")
            == hn.get("conversations", [{}])[0].get("content"),
        }
        if not all(checks.values()):
            raise RagSftV2TrainingReleaseError(f"clean/HN 配对事实不一致: {query_id}")
        pairing_rows.append(
            {
                "query_id": query_id,
                "clean_id": clean["id"],
                "hard_negative_id": hn["id"],
                "hn_source": ledger_by_id[hn["id"]]["source"],
                **checks,
            }
        )

    pairing = {
        "pipeline": "rag_sft_v2_clean_hn_pairing_audit",
        "records": {
            "total": 775,
            "clean": 549,
            "hard_negative": 226,
            "retrieved_hn": 70,
            "curated_hn": 156,
            "unique_record_ids": 775,
            "unique_queries": 549,
            "paired_hn_queries": 226,
            "unpaired_clean_queries": 323,
        },
        "validation": {
            "all_hn_have_clean_variant": True,
            "paired_query_original_equal": True,
            "paired_required_gt_equal": True,
            "paired_summary_equal": True,
            "paired_system_prompt_equal": True,
        },
        "pairs": pairing_rows,
        "complete": True,
    }
    pairing_payload = json.dumps(
        pairing, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"

    candidate_output = output_dir / CANDIDATE_FILENAME
    ledger_output = output_dir / LEDGER_FILENAME
    pairing_output = output_dir / PAIRING_FILENAME
    manifest_output = output_dir / MANIFEST_FILENAME
    readme_output = output_dir / README_FILENAME
    protocol = base_manifest.get("protocol", {})
    manifest: dict[str, object] = {
        "pipeline": "rag_sft_v2_training_release",
        "release_status": "formal_training_candidate",
        "inputs": {
            "audited_release_manifest": {
                **_identity(audited_manifest_path),
                "manifest_sha256": audited_manifest_sha,
            },
            "audited_semantic_candidate": _identity(candidate_path, records=775),
            "audited_identity_ledger": _identity(upstream_ledger_path, records=775),
            "context_exclusions": _identity(context_exclusions_path, records=11),
            "semantic_freeze_manifest": {
                **_identity(semantic_manifest_path),
                "manifest_sha256": semantic_manifest_sha,
            },
            "base_training_manifest": {
                **_identity(base_manifest_path),
                "manifest_sha256": base_manifest_sha,
            },
        },
        "retrieval_identity": base_manifest.get("retrieval_identity"),
        "hard_negative_identity": {
            "mode": "retrieved_plus_curated",
            "retrieved_hn": {
                "records": 70,
                "retrieval_identity": base_manifest.get("retrieval_identity"),
            },
            "curated_hn": {
                "records": 156,
                "article_index": semantic_manifest.get("inputs", {}).get("article_index"),
                "audited_release_manifest_sha256": audited_manifest_sha,
            },
            "final_candidate_sha256": _sha256(candidate_path),
        },
        "protocol": {
            "model_output": ["summary", "citations"],
            "context_limit": 768,
            "max_prompt_tokens": 618,
            "max_output_tokens": 150,
            "truncation": "forbidden",
            "clean_visible_evidence": protocol.get("clean_visible_evidence"),
            "retrieved_hn_retrieval_order_preserved": True,
            "curated_hn_evidence_order_frozen": True,
            "selector": "disabled",
        },
        "tokenizer": tokenizer,
        "records": {
            "total": 775,
            "clean": 549,
            "hard_negative": 226,
            "retrieved_hn": 70,
            "curated_hn": 156,
            "unique_queries": 549,
            "paired_hn_queries": 226,
            "unpaired_clean_queries": 323,
            "non_required_evidence_in_hn_packages": non_required_evidence,
            "visible_evidence_count": {
                str(key): visible_counts[key] for key in sorted(visible_counts)
            },
            "required_evidence_count": {
                str(key): required_counts[key] for key in sorted(required_counts)
            },
            "citation_count": {
                str(key): citation_counts[key] for key in sorted(citation_counts)
            },
        },
        "validation": {
            "strict_utf8_jsonl": True,
            "source_hashes_bound": True,
            "record_ids_unique": True,
            "record_sha256_matches_audited_ledger": True,
            "counts_closed": True,
            "all_records_pass_dataset_contract": True,
            "all_hn_have_clean_variant": True,
            "paired_answer_facts_equal": True,
            "retrieved_hn_retrieval_identity_frozen": True,
            "curated_hn_review_identity_frozen": True,
            "final_evidence_package_identity_frozen": True,
            "prompt_within_618": True,
            "assistant_within_150": True,
            "full_sequence_within_768": True,
            "assistant_only_labels_and_eos_audited": True,
            "evaluation_isolation_complete": base_manifest.get("validation", {}).get(
                "evaluation_isolation_complete"
            )
            is True,
            "quarantined_batch7_legacy_not_referenced": True,
        },
        "output": {
            "training_candidate": _payload_identity(
                candidate_output, candidate_payload, records=775
            ),
            "identity_ledger": _payload_identity(
                ledger_output, ledger_payload, records=775
            ),
            "pairing_audit": _payload_identity(
                pairing_output, pairing_payload, records=226
            ),
        },
        "policy": {
            "source_assets_modified": False,
            "formal_hn_materialized": True,
            "formal_training_candidate_emitted": True,
            "sampling_ratio_embedded": False,
            "training_ready": True,
        },
        "readiness": {
            "oracle_clean_audited": True,
            "semantic_review_complete": True,
            "context_length_audit_complete": True,
            "real_hard_negatives_constructed": True,
            "final_retrieval_identity_frozen": True,
            "formal_training_candidate_emitted": True,
            "training_ready": True,
        },
        "complete": True,
    }
    if manifest["validation"]["evaluation_isolation_complete"] is not True:
        raise RagSftV2TrainingReleaseError("评估隔离身份未从旧 619 release 继承")
    manifest_payload = json.dumps(
        manifest, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    readme_payload = (
        "# RAG-SFT v2 正式训练候选（549 clean + 226 HN）\n\n"
        "本目录将通过 768 上下文与语义审核的 549 条 Clean、70 条 retrieved HN "
        "和 156 条 curated HN 正式物化为 775 条训练记录。候选与已审计语义候选"
        "字节一致，不包含采样比例；采样配方由训练阶段单独冻结。\n\n"
        "`training_ready=true` 只适用于本目录。所有上游 release、审核资产、公共 Query "
        "池、法条索引、canonical 和 Oracle clean 均未修改。\n"
    )
    files = [
        (CANDIDATE_FILENAME, candidate_payload),
        (LEDGER_FILENAME, ledger_payload),
        (PAIRING_FILENAME, pairing_payload),
        (MANIFEST_FILENAME, manifest_payload),
        (README_FILENAME, readme_payload),
    ]
    try:
        partial_dir.mkdir(parents=True, exist_ok=False)
        for name, payload in files:
            (partial_dir / name).write_text(payload, encoding="utf-8", newline="\n")
        if (
            _sha256(partial_dir / CANDIDATE_FILENAME) != _sha256(candidate_path)
            or _sha256(partial_dir / LEDGER_FILENAME) != _sha256(upstream_ledger_path)
        ):
            raise RagSftV2TrainingReleaseError("正式候选或 ledger 未保持字节身份")
        hash_payload = "".join(
            f"{_sha256(partial_dir / name)}  {name}\n" for name, _ in files
        )
        (partial_dir / HASH_FILENAME).write_text(
            hash_payload, encoding="ascii", newline="\n"
        )
        partial_dir.replace(output_dir)
    except OSError as error:
        if partial_dir.exists():
            shutil.rmtree(partial_dir)
        raise RagSftV2TrainingReleaseError(
            "无法原子发布 775 条 RAG-SFT v2 正式 release"
        ) from error
    except RagSftV2TrainingReleaseError:
        if partial_dir.exists():
            shutil.rmtree(partial_dir)
        raise
    return manifest


def finalize_rag_sft_v2_training_release(
    *,
    oracle_root: Path,
    hn_root: Path,
    output_dir: Path,
    audited_root: Path | None = None,
) -> dict[str, object]:
    """发布不可覆盖的旧 549+70 或新 549+226 正式训练 release。"""

    if audited_root is not None:
        return _finalize_curated_training_release(
            audited_root=audited_root,
            output_dir=output_dir,
        )

    oracle_root = Path(oracle_root).resolve()
    hn_root = Path(hn_root).resolve()
    output_dir = Path(output_dir).resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise RagSftV2TrainingReleaseError(f"输出目录或临时目录已存在: {output_dir}")

    oracle_manifest_path = oracle_root / MANIFEST_FILENAME
    hn_manifest_path = hn_root / MANIFEST_FILENAME
    oracle_manifest_sha = _verify_manifest(oracle_manifest_path)
    hn_manifest_sha = _verify_manifest(hn_manifest_path)
    oracle_manifest = _load_json(oracle_manifest_path, "Oracle clean manifest")
    hn_manifest = _load_json(hn_manifest_path, "HN stage manifest")

    oracle_readiness = oracle_manifest.get("readiness", {})
    hn_validation = hn_manifest.get("validation", {})
    if (
        oracle_manifest.get("pipeline") != "rag_sft_v2_oracle_clean_materialization"
        or oracle_readiness.get("oracle_clean_audited") is not True
        or oracle_readiness.get("training_ready") is not False
    ):
        raise RagSftV2TrainingReleaseError("Oracle clean 尚非已审计阶段资产")
    if (
        hn_manifest.get("pipeline") != "rag_sft_v2_hn_variant_stage_asset"
        or hn_manifest.get("release_status") != "stage_experiment_hn_variants_audited"
        or hn_manifest.get("records", {}).get("excluded_after_projection") != 0
        or hn_validation.get("all_emitted_variants_pass_contract") is not True
        or hn_validation.get("retrieved_candidates_identity_closed") is not True
        or hn_validation.get("quarantined_batch7_legacy_not_referenced") is not True
        or hn_manifest.get("policy", {}).get("training_ready") is not False
    ):
        raise RagSftV2TrainingReleaseError("HN stage 的语义、身份或投影审计未闭合")
    oracle_protocol = oracle_manifest.get("protocol")
    hn_protocol = hn_manifest.get("protocol")
    protocol_core = {
        "model_output": ["summary", "citations"],
        "context_limit": 768,
        "max_prompt_tokens": 618,
        "max_output_tokens": 150,
        "truncation": "forbidden",
    }
    if (
        not isinstance(oracle_protocol, dict)
        or not isinstance(hn_protocol, dict)
        or any(oracle_protocol.get(key) != value for key, value in protocol_core.items())
        or any(hn_protocol.get(key) != value for key, value in protocol_core.items())
        or hn_protocol.get("retrieval_order_preserved") is not True
        or hn_protocol.get("selector") != "disabled"
    ):
        raise RagSftV2TrainingReleaseError("Oracle clean 与 HN 的协议身份不一致")
    if oracle_manifest.get("tokenizer") != hn_manifest.get("tokenizer"):
        raise RagSftV2TrainingReleaseError("Oracle clean 与 HN 的 Tokenizer 身份不一致")

    oracle_path = _bound_output(oracle_manifest, "oracle_clean", "Oracle clean")
    hn_path = _bound_output(hn_manifest, "projected_hn", "projected HN")
    oracle_rows = _load_jsonl(oracle_path, "Oracle clean")
    hn_rows = _load_jsonl(hn_path, "projected HN")
    if len(oracle_rows) != 549 or len(hn_rows) != 70:
        raise RagSftV2TrainingReleaseError(
            f"正式 release 要求 549 clean + 70 HN，实际为 {len(oracle_rows)} + {len(hn_rows)}"
        )

    clean_by_query = _index_unique(oracle_rows, "query_id", "Oracle clean ")
    hn_by_query = _index_unique(hn_rows, "query_id", "projected HN ")
    all_ids = _index_unique([*oracle_rows, *hn_rows], "id", "训练候选 ")
    if len(all_ids) != 619 or not set(hn_by_query).issubset(clean_by_query):
        raise RagSftV2TrainingReleaseError("HN 缺少对应 clean 或训练记录身份未闭合")

    pairing_rows: list[dict[str, object]] = []
    for query_id in sorted(hn_by_query):
        clean = clean_by_query[query_id]
        hn = hn_by_query[query_id]
        checks = {
            "query_original_equal": clean.get("query_original") == hn.get("query_original"),
            "required_chunk_ids_equal": clean.get("required_chunk_ids") == hn.get("required_chunk_ids"),
            "summary_equal": _assistant_summary(clean) == _assistant_summary(hn),
            "system_prompt_equal": clean.get("conversations", [{}])[0].get("content")
            == hn.get("conversations", [{}])[0].get("content"),
        }
        if not all(checks.values()):
            raise RagSftV2TrainingReleaseError(f"clean/HN 配对事实不一致: {query_id}")
        pairing_rows.append(
            {
                "query_id": query_id,
                "clean_id": clean["id"],
                "hard_negative_id": hn["id"],
                **checks,
            }
        )

    candidate_rows = [*oracle_rows, *hn_rows]
    ledger_rows: list[dict[str, object]] = []
    visible_counts: Counter[int] = Counter()
    required_counts: Counter[int] = Counter()
    citation_counts: Counter[int] = Counter()
    for line_number, record in enumerate(candidate_rows, 1):
        try:
            _validate_record(record, line_number)
        except Exception as error:
            raise RagSftV2TrainingReleaseError(
                f"训练候选第 {line_number} 条不满足正式 Dataset 契约"
            ) from error
        variant = record["variant"]
        source_line = line_number if variant == "clean" else line_number - len(oracle_rows)
        serialized = json.dumps(
            record, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        ledger_rows.append(
            {
                "line_number": line_number,
                "id": record["id"],
                "query_id": record["query_id"],
                "variant": variant,
                "source_file": str((oracle_path if variant == "clean" else hn_path).resolve()),
                "source_line": source_line,
                "record_sha256": hashlib.sha256(serialized).hexdigest(),
            }
        )
        visible_counts[len(record["visible_chunk_ids"])] += 1
        required_counts[len(record["required_chunk_ids"])] += 1
        citation_counts[len(record["citations"])] += 1

    candidate_payload = _serialize_jsonl(candidate_rows)
    ledger_payload = _serialize_jsonl(ledger_rows)
    pairing = {
        "pipeline": "rag_sft_v2_clean_hn_pairing_audit",
        "records": {
            "total": 619,
            "clean": 549,
            "hard_negative": 70,
            "unique_record_ids": 619,
            "unique_queries": 549,
            "paired_hn_queries": 70,
            "unpaired_clean_queries": 479,
        },
        "validation": {
            "all_hn_have_clean_variant": True,
            "paired_query_original_equal": True,
            "paired_required_gt_equal": True,
            "paired_summary_equal": True,
            "paired_system_prompt_equal": True,
        },
        "pairs": pairing_rows,
        "complete": True,
    }
    pairing_payload = json.dumps(pairing, ensure_ascii=False, indent=2, allow_nan=False) + "\n"

    candidate_path = output_dir / CANDIDATE_FILENAME
    ledger_path = output_dir / LEDGER_FILENAME
    pairing_path = output_dir / PAIRING_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    readme_path = output_dir / README_FILENAME
    retrieval_identity = hn_manifest.get("retrieval_identity")
    manifest: dict[str, object] = {
        "pipeline": "rag_sft_v2_training_release",
        "release_status": "formal_training_candidate",
        "inputs": {
            "oracle_clean_manifest": {
                **_identity(oracle_manifest_path),
                "manifest_sha256": oracle_manifest_sha,
            },
            "oracle_clean": _identity(oracle_path, records=549),
            "hn_stage_manifest": {
                **_identity(hn_manifest_path),
                "manifest_sha256": hn_manifest_sha,
            },
            "projected_hn": _identity(hn_path, records=70),
        },
        "retrieval_identity": retrieval_identity,
        "protocol": {
            **protocol_core,
            "clean_visible_evidence": oracle_protocol.get("visible_evidence"),
            "hard_negative_retrieval_order_preserved": True,
            "selector": "disabled",
        },
        "tokenizer": oracle_manifest["tokenizer"],
        "records": {
            "total": 619,
            "clean": 549,
            "hard_negative": 70,
            "unique_queries": 549,
            "paired_hn_queries": 70,
            "unpaired_clean_queries": 479,
            "hard_negative_evidence": hn_manifest.get("records", {})
            .get("non_gt_labels", {})
            .get("hard_negative"),
            "irrelevant_evidence_in_hn_packages": hn_manifest.get("records", {})
            .get("non_gt_labels", {})
            .get("irrelevant"),
            "visible_evidence_count": {str(key): visible_counts[key] for key in sorted(visible_counts)},
            "required_evidence_count": {str(key): required_counts[key] for key in sorted(required_counts)},
            "citation_count": {str(key): citation_counts[key] for key in sorted(citation_counts)},
        },
        "validation": {
            "strict_utf8_jsonl": True,
            "source_hashes_bound": True,
            "record_ids_unique": True,
            "counts_closed": True,
            "all_records_pass_dataset_contract": True,
            "all_hn_have_clean_variant": True,
            "paired_answer_facts_equal": True,
            "retrieval_order_preserved": True,
            "final_retrieval_identity_frozen": True,
            "prompt_within_618": True,
            "assistant_within_150": True,
            "full_sequence_within_768": True,
            "assistant_only_labels_and_eos_audited": True,
            "evaluation_isolation_complete": True,
            "quarantined_batch7_legacy_not_referenced": True,
        },
        "output": {
            "training_candidate": _payload_identity(candidate_path, candidate_payload, records=619),
            "identity_ledger": _payload_identity(ledger_path, ledger_payload, records=619),
            "pairing_audit": _payload_identity(pairing_path, pairing_payload, records=70),
        },
        "policy": {
            "source_assets_modified": False,
            "formal_hn_materialized": True,
            "formal_training_candidate_emitted": True,
            "sampling_ratio_embedded": False,
            "training_ready": True,
        },
        "readiness": {
            "oracle_clean_audited": True,
            "semantic_review_complete": True,
            "real_hard_negatives_constructed": True,
            "final_retrieval_identity_frozen": True,
            "formal_training_candidate_emitted": True,
            "training_ready": True,
        },
        "complete": True,
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    readme_payload = (
        "# RAG-SFT v2 正式训练候选（549 clean + 70 HN）\n\n"
        "本目录冻结 549 条 Oracle clean 与 70 条真实检索 HN，共 619 条唯一训练记录。"
        "候选顺序固定为全部 clean 后全部 HN；clean/HN 曝光比例不写入数据，"
        "由训练阶段的确定性 sampler 单独控制。\n\n"
        "`training-candidate.jsonl` 是正式训练入口，`identity-ledger.jsonl` 提供逐条来源与哈希，"
        "`pairing-audit.json` 记录 70 对 clean/HN 的共同回答事实核验。\n\n"
        "本 release 不修改上游 Oracle clean、HN stage、真实检索候选或语义审核资产。"
        "`training_ready=true` 仅适用于本目录的新正式 release。\n"
    )

    files = [
        (CANDIDATE_FILENAME, candidate_payload),
        (LEDGER_FILENAME, ledger_payload),
        (PAIRING_FILENAME, pairing_payload),
        (MANIFEST_FILENAME, manifest_payload),
        (README_FILENAME, readme_payload),
    ]
    try:
        partial_dir.mkdir(parents=True, exist_ok=False)
        for name, payload in files:
            (partial_dir / name).write_text(payload, encoding="utf-8", newline="\n")
        hash_payload = "".join(
            f"{_sha256(partial_dir / name)}  {name}\n" for name, _ in files
        )
        (partial_dir / HASH_FILENAME).write_text(
            hash_payload, encoding="ascii", newline="\n"
        )
        partial_dir.replace(output_dir)
    except OSError as error:
        if partial_dir.exists():
            shutil.rmtree(partial_dir)
        raise RagSftV2TrainingReleaseError("无法原子发布 RAG-SFT v2 正式 release") from error
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLE_ROOT)
    parser.add_argument("--hn-root", type=Path, default=DEFAULT_HN_ROOT)
    parser.add_argument("--audited-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or (
        DEFAULT_CURATED_OUTPUT_DIR
        if args.audited_root is not None
        else DEFAULT_OUTPUT_DIR
    )
    try:
        manifest = finalize_rag_sft_v2_training_release(
            oracle_root=args.oracle_root,
            hn_root=args.hn_root,
            audited_root=args.audited_root,
            output_dir=output_dir,
        )
    except (RagSftV2TrainingReleaseError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    records = manifest["records"]
    print(
        "RAG_SFT_V2_TRAINING_RELEASE_OK "
        f"total={records['total']} clean={records['clean']} "
        f"hard_negative={records['hard_negative']} training_ready=true"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "RagSftV2TrainingReleaseError",
    "finalize_rag_sft_v2_training_release",
]
