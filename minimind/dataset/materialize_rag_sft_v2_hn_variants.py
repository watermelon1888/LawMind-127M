"""将已批准的真实检索 HN 包物化为阶段性 HN variant 与投影审计资产。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from rag.knowledge import ArticleRepository

from . import audit_disc_law_sft as tokenizer_auditor
from .rag_sft_v2_contract import (
    RagSftV2ContractError,
    validate_hn_variant,
    validate_hn_visible_evidence_distribution,
)
from .rag_sft_v2_projection import (
    CONTEXT_LIMIT,
    MAX_OUTPUT_TOKENS,
    MAX_PROMPT_TOKENS,
    RagSftV2ProjectionError,
    audit_projected_rag_sft_v2_record,
    project_rag_sft_v2_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REVIEW_ROOT = (
    PROJECT_ROOT
    / "minimind"
    / "dataset"
    / "RAG-SFT"
    / "review"
    / "v2"
    / "hn-semantic-review-v1-20260813"
)
DEFAULT_CLOSURE_ROOT = (
    DEFAULT_REVIEW_ROOT / "semantic-review-closure-v2-20260813-final-v1"
)
DEFAULT_TOKENIZER = PROJECT_ROOT / "minimind" / "model"

VARIANTS_FILENAME = "hn-variants.jsonl"
PROJECTED_FILENAME = "projected-hn.jsonl"
AUDIT_FILENAME = "projection-audit.jsonl"
EXCLUSIONS_FILENAME = "exclusions.jsonl"
MANIFEST_FILENAME = "manifest.json"
HASH_FILENAME = "manifest.sha256"
README_FILENAME = "README.md"


class RagSftV2HnVariantMaterializationError(RuntimeError):
    """HN variant 的输入身份、裁决或投影无法安全闭合。"""


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
        raise RagSftV2HnVariantMaterializationError(
            f"无法读取{description}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise RagSftV2HnVariantMaterializationError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagSftV2HnVariantMaterializationError(
                        f"{description}存在空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RagSftV2HnVariantMaterializationError(
                        f"{description}第 {line_number} 行不是对象"
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftV2HnVariantMaterializationError):
            raise
        raise RagSftV2HnVariantMaterializationError(
            f"无法读取{description}: {path}"
        ) from error
    return rows


def _verify_manifest_sidecar(path: Path) -> str:
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise RagSftV2HnVariantMaterializationError(f"manifest 或相邻 SHA-256 不存在: {path}")
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
        raise RagSftV2HnVariantMaterializationError(
            f"manifest SHA-256 清单格式无效: {sidecar}"
        ) from error
    if matches != [digest]:
        raise RagSftV2HnVariantMaterializationError(f"manifest SHA-256 校验失败: {path}")
    return digest


def _bound_path(metadata: object, description: str) -> Path:
    if not isinstance(metadata, dict):
        raise RagSftV2HnVariantMaterializationError(f"缺少{description}身份")
    path_value = metadata.get("path")
    expected_bytes = metadata.get("bytes")
    expected_sha256 = metadata.get("sha256")
    if (
        not isinstance(path_value, str)
        or type(expected_bytes) is not int
        or not isinstance(expected_sha256, str)
    ):
        raise RagSftV2HnVariantMaterializationError(f"{description}身份字段无效")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise RagSftV2HnVariantMaterializationError(f"{description}不存在: {path}")
    if path.stat().st_size != expected_bytes or _sha256(path) != expected_sha256:
        raise RagSftV2HnVariantMaterializationError(f"{description}身份已变化: {path}")
    return path


def _index_unique(rows: Iterable[dict[str, Any]], key: str, description: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise RagSftV2HnVariantMaterializationError(f"{description}缺少 {key}")
        if value in indexed:
            raise RagSftV2HnVariantMaterializationError(f"{description}身份重复: {value}")
        indexed[value] = row
    return indexed


def _canonical_from_clean(clean: dict[str, Any]) -> dict[str, object]:
    conversations = clean.get("conversations")
    if (
        not isinstance(conversations, list)
        or len(conversations) != 3
        or [item.get("role") for item in conversations if isinstance(item, dict)]
        != ["system", "user", "assistant"]
    ):
        raise RagSftV2HnVariantMaterializationError("Oracle clean conversations 身份无效")
    try:
        assistant = json.loads(conversations[2]["content"])
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise RagSftV2HnVariantMaterializationError("Oracle clean assistant JSON 无效") from error
    summary = assistant.get("summary") if isinstance(assistant, dict) else None
    query_id = clean.get("query_id")
    query_original = clean.get("query_original")
    required = clean.get("required_chunk_ids")
    if (
        not isinstance(query_id, str)
        or not isinstance(query_original, str)
        or not isinstance(summary, str)
        or not isinstance(required, list)
    ):
        raise RagSftV2HnVariantMaterializationError("Oracle clean 回答事实字段无效")
    return {
        "query_id": query_id,
        "query_original": query_original,
        "summary": summary,
        "required_chunk_ids": required,
    }


def build_approved_hn_variant(
    *,
    source_package: dict[str, Any],
    retrieved_candidate: dict[str, Any],
    package_adjudication: dict[str, Any],
    evidence_adjudications: list[dict[str, Any]],
    oracle_clean: dict[str, Any],
    retrieval_identity: str,
) -> tuple[dict[str, object], dict[str, object]]:
    """从一个身份闭合的批准包构造正式 schema 的阶段性 HN variant。"""

    query_id = source_package.get("query_id")
    variant_id = source_package.get("variant_id")
    visible = source_package.get("visible_chunk_ids")
    required = source_package.get("required_chunk_ids")
    evidence = source_package.get("evidence")
    if (
        not isinstance(query_id, str)
        or not isinstance(variant_id, str)
        or not isinstance(visible, list)
        or not isinstance(required, list)
        or not isinstance(evidence, list)
    ):
        raise RagSftV2HnVariantMaterializationError("source package 身份无效")
    canonical = _canonical_from_clean(oracle_clean)
    if (
        oracle_clean.get("query_id") != query_id
        or oracle_clean.get("query_original") != source_package.get("query_original")
        or canonical["required_chunk_ids"] != required
    ):
        raise RagSftV2HnVariantMaterializationError(f"Oracle clean 与 source 不闭合: {query_id}")
    if (
        retrieved_candidate.get("query_id") != query_id
        or retrieved_candidate.get("variant_id") != variant_id
        or retrieved_candidate.get("visible_chunk_ids") != visible
        or retrieved_candidate.get("required_chunk_ids") != required
        or retrieved_candidate.get("source") != "retrieved"
    ):
        raise RagSftV2HnVariantMaterializationError(f"原始检索 candidate 与 source 不闭合: {query_id}")
    if (
        package_adjudication.get("record_type") != "package_adjudication"
        or package_adjudication.get("query_id") != query_id
        or package_adjudication.get("variant_id") != variant_id
        or package_adjudication.get("decision") != "approve_candidate"
    ):
        raise RagSftV2HnVariantMaterializationError(f"包级批准裁决无效: {query_id}")

    evidence_by_id: dict[str, dict[str, Any]] = {}
    for item in evidence:
        evidence_id = item.get("evidence_id") if isinstance(item, dict) else None
        chunk_id = item.get("chunk_id") if isinstance(item, dict) else None
        if not isinstance(evidence_id, str) or not isinstance(chunk_id, str):
            raise RagSftV2HnVariantMaterializationError(f"source evidence 身份无效: {query_id}")
        if evidence_id in evidence_by_id:
            raise RagSftV2HnVariantMaterializationError(f"source evidence_id 重复: {query_id}")
        evidence_by_id[evidence_id] = item
    if [item["chunk_id"] for item in evidence] != visible:
        raise RagSftV2HnVariantMaterializationError(f"source evidence 顺序与 visible 不一致: {query_id}")

    adjudication_by_evidence: dict[str, dict[str, Any]] = {}
    for row in evidence_adjudications:
        evidence_id = row.get("evidence_id")
        if not isinstance(evidence_id, str) or evidence_id in adjudication_by_evidence:
            raise RagSftV2HnVariantMaterializationError(f"证据裁决身份重复或无效: {query_id}")
        adjudication_by_evidence[evidence_id] = row
    non_gt = [item for item in evidence if item.get("is_required_gt") is False]
    expected_ids = {item["evidence_id"] for item in non_gt}
    if set(adjudication_by_evidence) != expected_ids:
        raise RagSftV2HnVariantMaterializationError(f"证据裁决未覆盖全部且仅覆盖 non-GT: {query_id}")

    labels: list[dict[str, str]] = []
    approved_ids: list[str] = []
    for item in non_gt:
        row = adjudication_by_evidence[item["evidence_id"]]
        label = row.get("adversarial_label")
        legal_support = row.get("legal_support")
        expected_decision = "approve_evidence" if label == "hard_negative" and legal_support == "none" else "exclude_evidence"
        if (
            row.get("record_type") != "evidence_adjudication"
            or row.get("query_id") != query_id
            or row.get("variant_id") != variant_id
            or row.get("chunk_id") != item["chunk_id"]
            or row.get("decision") != expected_decision
        ):
            raise RagSftV2HnVariantMaterializationError(
                f"证据裁决身份或准入规则不闭合: {query_id}/{item['evidence_id']}"
            )
        if legal_support != "none" or label not in {"hard_negative", "irrelevant"}:
            raise RagSftV2HnVariantMaterializationError(
                f"批准包包含不允许进入 HN variant 的证据: {query_id}/{item['evidence_id']}"
            )
        if expected_decision == "approve_evidence":
            approved_ids.append(item["evidence_id"])
        labels.append({"chunk_id": item["chunk_id"], "label": label})
    if package_adjudication.get("approved_evidence_ids") != approved_ids:
        raise RagSftV2HnVariantMaterializationError(f"包级与证据级批准集合不闭合: {query_id}")

    variant: dict[str, object] = {
        "variant_id": variant_id,
        "query_id": query_id,
        "visible_chunk_ids": visible,
        "source": "retrieved",
        "retrieval_identity": retrieval_identity,
        "non_gt_labels": labels,
        "review_decision": "approved",
    }
    try:
        validate_hn_variant(variant, canonical)
    except RagSftV2ContractError as error:
        raise RagSftV2HnVariantMaterializationError(
            f"批准包不满足 HN variant 契约: {query_id}"
        ) from error
    return variant, canonical


def _serialize_jsonl(rows: Iterable[dict[str, object]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    )


def _distribution(counter: Counter[int]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter)}


def materialize_rag_sft_v2_hn_variants(
    *,
    review_root: Path,
    closure_root: Path,
    tokenizer_path: Path,
    output_dir: Path,
    tokenizer: Any | None = None,
) -> dict[str, object]:
    """发布阶段性 HN variant、投影和逐条预算审计，不发布训练 release。"""

    review_root = Path(review_root).resolve()
    closure_root = Path(closure_root).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RagSftV2HnVariantMaterializationError(f"输出目录必须不存在: {output_dir}")
    if not review_root.is_dir() or not closure_root.is_dir() or not tokenizer_path.is_dir():
        raise RagSftV2HnVariantMaterializationError("review、closure 或 tokenizer 输入目录不存在")

    review_manifest_path = review_root / MANIFEST_FILENAME
    closure_manifest_path = closure_root / MANIFEST_FILENAME
    _verify_manifest_sidecar(review_manifest_path)
    _verify_manifest_sidecar(closure_manifest_path)
    review_manifest = _load_json(review_manifest_path, "HN 审阅包装 manifest")
    closure_manifest = _load_json(closure_manifest_path, "HN 语义闭合 manifest")
    if (
        review_manifest.get("pipeline") != "rag_sft_v2_hn_semantic_review_packaging"
        or review_manifest.get("records", {}).get("candidates") != 452
        or review_manifest.get("readiness", {}).get("training_ready") is not False
    ):
        raise RagSftV2HnVariantMaterializationError("HN 审阅包装身份或状态无效")
    if (
        closure_manifest.get("pipeline") != "rag_sft_v2_hn_semantic_review_closure"
        or closure_manifest.get("records")
        != {"batches": 19, "packages": 452, "non_gt_evidence": 880}
        or closure_manifest.get("formal_hn_materialized") is not False
        or closure_manifest.get("training_ready") is not False
    ):
        raise RagSftV2HnVariantMaterializationError("HN 语义闭合身份或状态无效")

    review_inputs = review_manifest.get("inputs")
    if not isinstance(review_inputs, dict):
        raise RagSftV2HnVariantMaterializationError("HN 审阅包装 manifest 缺少 inputs")
    oracle_manifest_path = _bound_path(review_inputs.get("oracle_manifest"), "Oracle clean manifest")
    retrieval_report_path = _bound_path(review_inputs.get("materialization_report"), "检索物化报告")
    retrieved_candidates_path = _bound_path(review_inputs.get("hn_candidates"), "原始 HN candidates")
    article_index_path = _bound_path(review_inputs.get("article_index"), "法条索引")
    _verify_manifest_sidecar(oracle_manifest_path)
    oracle_manifest = _load_json(oracle_manifest_path, "Oracle clean manifest")
    if (
        oracle_manifest.get("pipeline") != "rag_sft_v2_oracle_clean_materialization"
        or oracle_manifest.get("release_status") != "stage_5_oracle_clean_audited"
        or oracle_manifest.get("readiness", {}).get("oracle_clean_audited") is not True
        or oracle_manifest.get("readiness", {}).get("training_ready") is not False
    ):
        raise RagSftV2HnVariantMaterializationError("Oracle clean 身份或阶段状态无效")
    oracle_path = _bound_path(oracle_manifest.get("output", {}).get("oracle_clean"), "Oracle clean")
    clean_by_query = _index_unique(_load_jsonl(oracle_path, "Oracle clean"), "query_id", "Oracle clean")

    retrieval_report = _load_json(retrieval_report_path, "检索物化报告")
    retrieval_object = retrieval_report.get("input", {}).get("retrieval_identity")
    if (
        retrieval_report.get("pipeline") != "rag_sft_v2_retrieved_materialization"
        or retrieval_report.get("release_status") != "stage_7_retrieved_candidates"
        or retrieval_report.get("records", {}).get("candidates") != 452
        or not isinstance(retrieval_object, dict)
    ):
        raise RagSftV2HnVariantMaterializationError("检索物化报告身份无效")
    retrieval_object_payload = json.dumps(
        retrieval_object, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    retrieval_identity = (
        f"retrieved:report-sha256:{_sha256(retrieval_report_path)}:"
        f"identity-sha256:{hashlib.sha256(retrieval_object_payload).hexdigest()}"
    )
    retrieved_by_query = _index_unique(
        _load_jsonl(retrieved_candidates_path, "原始 HN candidates"),
        "query_id",
        "原始 HN candidates",
    )
    if any(row.get("retrieval_identity") != retrieval_object for row in retrieved_by_query.values()):
        raise RagSftV2HnVariantMaterializationError("原始 HN candidate 检索身份不统一")

    try:
        repository = ArticleRepository.from_jsonl(article_index_path)
        tokenizer = tokenizer or tokenizer_auditor.load_tokenizer(tokenizer_path)
    except Exception as error:
        raise RagSftV2HnVariantMaterializationError("无法加载法条索引或本地 Tokenizer") from error
    tokenizer_report = tokenizer_auditor._tokenizer_report(tokenizer, tokenizer_path)
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str):
        raise RagSftV2HnVariantMaterializationError("Tokenizer 缺少 chat template")
    tokenizer_report["chat_template_sha256"] = hashlib.sha256(
        chat_template.encode("utf-8")
    ).hexdigest()
    if tokenizer_report != oracle_manifest.get("tokenizer"):
        raise RagSftV2HnVariantMaterializationError("Tokenizer 身份与 Oracle clean 不一致")

    variants: list[dict[str, object]] = []
    projected_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    exclusions: list[dict[str, object]] = []
    approved_queries: set[str] = set()
    label_counts: Counter[str] = Counter()
    evidence_counts: Counter[int] = Counter()
    hard_negative_counts: Counter[int] = Counter()
    prompt_counts: Counter[int] = Counter()
    assistant_counts: Counter[int] = Counter()
    total_counts: Counter[int] = Counter()
    gt_positions: Counter[int] = Counter()
    hn_positions: Counter[int] = Counter()
    approved_scanned = 0

    batch_metadata = closure_manifest.get("batch_manifests")
    if not isinstance(batch_metadata, list) or len(batch_metadata) != 19:
        raise RagSftV2HnVariantMaterializationError("闭合根 manifest 批次身份无效")
    batch_metadata_by_number = {item.get("batch"): item for item in batch_metadata if isinstance(item, dict)}
    if set(batch_metadata_by_number) != set(range(1, 20)):
        raise RagSftV2HnVariantMaterializationError("闭合根 manifest 批次集合不完整")

    for batch in range(1, 20):
        source_path = review_root / f"review-batch-{batch:02d}.jsonl"
        batch_dir = closure_root / f"batch-{batch:02d}"
        batch_manifest_path = batch_dir / MANIFEST_FILENAME
        _verify_manifest_sidecar(batch_manifest_path)
        expected_batch_manifest = batch_metadata_by_number[batch].get("manifest")
        if _bound_path(expected_batch_manifest, f"第 {batch} 批闭合 manifest") != batch_manifest_path.resolve():
            raise RagSftV2HnVariantMaterializationError(f"第 {batch} 批闭合 manifest 路径不一致")
        batch_manifest = _load_json(batch_manifest_path, f"第 {batch} 批闭合 manifest")
        source_metadata = batch_manifest.get("inputs", {}).get("review_batch")
        if _bound_path(source_metadata, f"第 {batch} 批 source") != source_path.resolve():
            raise RagSftV2HnVariantMaterializationError(f"第 {batch} 批 source 路径不一致")
        source_rows = _load_jsonl(source_path, f"第 {batch} 批 source")
        source_by_query = _index_unique(source_rows, "query_id", f"第 {batch} 批 source")
        adjudication_path = _bound_path(
            batch_manifest.get("outputs", {}).get("adjudication"),
            f"第 {batch} 批裁决",
        )
        adjudications = _load_jsonl(adjudication_path, f"第 {batch} 批裁决")
        packages = [row for row in adjudications if row.get("record_type") == "package_adjudication"]
        evidence_rows = [row for row in adjudications if row.get("record_type") == "evidence_adjudication"]
        if len(packages) + len(evidence_rows) != len(adjudications):
            raise RagSftV2HnVariantMaterializationError(f"第 {batch} 批裁决记录类型无效")
        package_by_query = _index_unique(packages, "query_id", f"第 {batch} 批包级裁决")
        evidence_by_query: dict[str, list[dict[str, Any]]] = {}
        for row in evidence_rows:
            query_id = row.get("query_id")
            if not isinstance(query_id, str):
                raise RagSftV2HnVariantMaterializationError(f"第 {batch} 批证据裁决缺少 query_id")
            evidence_by_query.setdefault(query_id, []).append(row)
        if set(package_by_query) != set(source_by_query):
            raise RagSftV2HnVariantMaterializationError(f"第 {batch} 批包级裁决覆盖不闭合")

        for query_id, package_row in package_by_query.items():
            if package_row.get("decision") != "approve_candidate":
                continue
            approved_scanned += 1
            if query_id in approved_queries:
                raise RagSftV2HnVariantMaterializationError(f"批准 query 跨批重复: {query_id}")
            approved_queries.add(query_id)
            if query_id not in retrieved_by_query or query_id not in clean_by_query:
                raise RagSftV2HnVariantMaterializationError(f"批准 query 缺少检索或 Oracle 身份: {query_id}")
            source_package = source_by_query[query_id]
            for source_evidence in source_package.get("evidence", []):
                if not isinstance(source_evidence, dict):
                    raise RagSftV2HnVariantMaterializationError(f"source evidence 无效: {query_id}")
                try:
                    article = repository.get_by_chunk_id(source_evidence.get("chunk_id"))
                except (KeyError, ValueError) as error:
                    raise RagSftV2HnVariantMaterializationError(f"法条索引缺少 source evidence: {query_id}") from error
                if (
                    article.law_name != source_evidence.get("law_name")
                    or article.article_no != source_evidence.get("article_no")
                    or article.content != source_evidence.get("content")
                ):
                    raise RagSftV2HnVariantMaterializationError(f"source evidence 与法条索引不一致: {query_id}")
            variant, canonical = build_approved_hn_variant(
                source_package=source_package,
                retrieved_candidate=retrieved_by_query[query_id],
                package_adjudication=package_row,
                evidence_adjudications=evidence_by_query.get(query_id, []),
                oracle_clean=clean_by_query[query_id],
                retrieval_identity=retrieval_identity,
            )
            try:
                article_by_chunk_id = {
                    chunk_id: repository.get_by_chunk_id(chunk_id)
                    for chunk_id in variant["visible_chunk_ids"]
                }
                projected = project_rag_sft_v2_record(
                    canonical, article_by_chunk_id, hn_variant=variant
                )
                audit = audit_projected_rag_sft_v2_record(projected, tokenizer)
            except (RagSftV2ContractError, RagSftV2ProjectionError) as error:
                exclusions.append(
                    {
                        "query_id": query_id,
                        "variant_id": variant["variant_id"],
                        "reason": str(error),
                    }
                )
                continue
            variants.append(variant)
            projected_rows.append(
                {
                    "id": projected.record_id,
                    "query_id": projected.query_id,
                    "variant": projected.variant,
                    "query_original": canonical["query_original"],
                    "visible_chunk_ids": list(projected.visible_chunk_ids),
                    "required_chunk_ids": list(projected.required_chunk_ids),
                    "citations": list(projected.citations),
                    "conversations": projected.training_conversations(),
                }
            )
            required_set = set(projected.required_chunk_ids)
            non_gt_labels = {item["chunk_id"]: item["label"] for item in variant["non_gt_labels"]}
            hard_negative_count = sum(label == "hard_negative" for label in non_gt_labels.values())
            first_gt = next(
                index for index, chunk_id in enumerate(projected.visible_chunk_ids, start=1)
                if chunk_id in required_set
            )
            hard_negative_position_values = [
                index
                for index, chunk_id in enumerate(projected.visible_chunk_ids, start=1)
                if non_gt_labels.get(chunk_id) == "hard_negative"
            ]
            audit_rows.append(
                {
                    "query_id": query_id,
                    "variant_id": variant["variant_id"],
                    "source_batch": package_row["source_batch"],
                    "source_batch_sha256": package_row["source_batch_sha256"],
                    "visible_evidence": len(projected.visible_chunk_ids),
                    "required_evidence": len(projected.required_chunk_ids),
                    "hard_negative_evidence": hard_negative_count,
                    "irrelevant_evidence": sum(label == "irrelevant" for label in non_gt_labels.values()),
                    "first_gt_position": first_gt,
                    "hard_negative_positions": hard_negative_position_values,
                    "hard_negative_before_first_gt": any(
                        position < first_gt for position in hard_negative_position_values
                    ),
                    "prompt_tokens": audit.prompt_tokens,
                    "assistant_label_tokens": audit.assistant_label_tokens,
                    "total_tokens": audit.total_tokens,
                    "citations": list(projected.citations),
                }
            )
            for label in non_gt_labels.values():
                label_counts[label] += 1
            evidence_counts[len(projected.visible_chunk_ids)] += 1
            hard_negative_counts[hard_negative_count] += 1
            prompt_counts[audit.prompt_tokens] += 1
            assistant_counts[audit.assistant_label_tokens] += 1
            total_counts[audit.total_tokens] += 1
            for position, chunk_id in enumerate(projected.visible_chunk_ids, start=1):
                if chunk_id in required_set:
                    gt_positions[position] += 1
                elif non_gt_labels.get(chunk_id) == "hard_negative":
                    hn_positions[position] += 1

    if approved_scanned != sum(
        int(item.get("records", {}).get("package_decisions", {}).get("approve_candidate", 0))
        for item in (
            _load_json(closure_root / f"batch-{batch:02d}" / MANIFEST_FILENAME, "批次 manifest")
            for batch in range(1, 20)
        )
    ):
        raise RagSftV2HnVariantMaterializationError("批准包计数与批次 manifest 不一致")
    if len(variants) + len(exclusions) != approved_scanned:
        raise RagSftV2HnVariantMaterializationError("批准包未全部进入 variant 或排除账本")
    if not variants or len({row["query_id"] for row in variants}) != len(variants):
        raise RagSftV2HnVariantMaterializationError("HN variant 为空或 query 身份重复")

    five_evidence = validate_hn_visible_evidence_distribution(variants)
    variants_payload = _serialize_jsonl(variants)
    projected_payload = _serialize_jsonl(projected_rows)
    audit_payload = _serialize_jsonl(audit_rows)
    exclusions_payload = _serialize_jsonl(exclusions)
    variant_path = output_dir / VARIANTS_FILENAME
    projected_path = output_dir / PROJECTED_FILENAME
    audit_path = output_dir / AUDIT_FILENAME
    exclusions_path = output_dir / EXCLUSIONS_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME

    manifest: dict[str, object] = {
        "pipeline": "rag_sft_v2_hn_variant_stage_asset",
        "release_status": "stage_experiment_hn_variants_audited",
        "inputs": {
            "oracle_clean_manifest": _identity(oracle_manifest_path),
            "review_packaging_manifest": _identity(review_manifest_path),
            "semantic_review_closure_manifest": _identity(closure_manifest_path),
            "retrieval_materialization_report": _identity(retrieval_report_path),
            "retrieved_hn_candidates": _identity(retrieved_candidates_path, len(retrieved_by_query)),
            "article_index": _identity(article_index_path),
        },
        "retrieval_identity": {
            "id": retrieval_identity,
            "report_sha256": _sha256(retrieval_report_path),
            "retrieval_object_sha256": hashlib.sha256(retrieval_object_payload).hexdigest(),
            "query_mode": retrieval_object.get("query_mode"),
            "config": retrieval_object.get("config"),
        },
        "protocol": {
            "model_output": ["summary", "citations"],
            "context_limit": CONTEXT_LIMIT,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "retrieval_order_preserved": True,
            "truncation": "forbidden",
            "selector": "disabled",
        },
        "tokenizer": tokenizer_report,
        "records": {
            "approved_packages_scanned": approved_scanned,
            "hn_variants": len(variants),
            "projected_hn": len(projected_rows),
            "excluded_after_projection": len(exclusions),
            "non_gt_labels": dict(sorted(label_counts.items())),
            "visible_evidence_count": _distribution(evidence_counts),
            "hard_negative_count": _distribution(hard_negative_counts),
            "prompt_tokens": _distribution(prompt_counts),
            "assistant_label_tokens": _distribution(assistant_counts),
            "total_tokens": _distribution(total_counts),
            "required_gt_positions": _distribution(gt_positions),
            "hard_negative_positions": _distribution(hn_positions),
            "hard_negative_before_first_gt": sum(
                row["hard_negative_before_first_gt"] is True for row in audit_rows
            ),
            "five_evidence": five_evidence,
        },
        "validation": {
            "source_batches_hash_bound": True,
            "retrieved_candidates_identity_closed": True,
            "oracle_query_and_required_gt_closed": True,
            "one_hn_variant_per_query": True,
            "all_non_gt_double_reviewed_and_adjudicated": True,
            "all_emitted_variants_pass_contract": True,
            "retrieval_order_preserved": True,
            "citations_map_all_required_gt": True,
            "prompt_within_618": all(value <= MAX_PROMPT_TOKENS for value in prompt_counts),
            "assistant_within_150": all(value <= MAX_OUTPUT_TOKENS for value in assistant_counts),
            "full_sequence_within_768": all(value <= CONTEXT_LIMIT for value in total_counts),
            "assistant_only_labels_and_eos_audited": True,
            "quarantined_batch7_legacy_not_referenced": True,
        },
        "output": {
            "hn_variants": _payload_identity(variant_path, variants_payload, len(variants)),
            "projected_hn": _payload_identity(projected_path, projected_payload, len(projected_rows)),
            "projection_audit": _payload_identity(audit_path, audit_payload, len(audit_rows)),
            "exclusions": _payload_identity(exclusions_path, exclusions_payload, len(exclusions)),
        },
        "policy": {
            "source_candidates_modified": False,
            "formal_hn_materialized": False,
            "formal_training_candidate_emitted": False,
            "training_ready": False,
        },
        "readiness": {
            "semantic_review_complete": True,
            "stage_hn_variants_materialized": True,
            "stage_hn_variants_audited": True,
            "real_hard_negatives_constructed": False,
            "final_training_release": False,
            "training_ready": False,
        },
        "limitations": [
            "本目录仅发布阶段性 HN variant 与确定性投影，不是正式训练 release。",
            "projected-hn.jsonl 只用于后续实验组装与复核，不能单独将 training_ready 改为 true。",
            "正式发布仍需与 clean 合并、冻结最终 release identity 并执行 Dataset 全量审计。",
        ],
        "complete": True,
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    readme_payload = (
        "# RAG-SFT v2 阶段性真实检索 HN variants\n\n"
        f"本目录从 {approved_scanned} 个语义审核批准包生成 {len(variants)} 条 HN variant；"
        f"另有 {len(exclusions)} 条在正式契约或 618/150/768 投影审计后被排除。\n\n"
        "`hn-variants.jsonl` 保存 HN variant 账本，`projected-hn.jsonl` 保存确定性三轮投影，"
        "`projection-audit.jsonl` 保存逐条长度、引用和位置审计，`exclusions.jsonl` 保存未发布项。\n\n"
        "本资产不修改任何源候选，不生成正式训练 release。"
        "`formal_hn_materialized=false`，`training_ready=false`。\n"
    )
    payloads = [
        (VARIANTS_FILENAME, variants_payload),
        (PROJECTED_FILENAME, projected_payload),
        (AUDIT_FILENAME, audit_payload),
        (EXCLUSIONS_FILENAME, exclusions_payload),
        (MANIFEST_FILENAME, manifest_payload),
        (README_FILENAME, readme_payload),
    ]
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, payload in payloads:
        (output_dir / filename).write_text(payload, encoding="utf-8", newline="\n")
    (output_dir / HASH_FILENAME).write_text(
        "".join(
            f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()}  {filename}\n"
            for filename, payload in payloads
        ),
        encoding="ascii",
        newline="\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--closure-root", type=Path, default=DEFAULT_CLOSURE_ROOT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = materialize_rag_sft_v2_hn_variants(
        review_root=args.review_root,
        closure_root=args.closure_root,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
    )
    records = manifest["records"]
    print(
        "RAG_SFT_V2_HN_VARIANTS_OK "
        f"approved={records['approved_packages_scanned']} "
        f"variants={records['hn_variants']} excluded={records['excluded_after_projection']} "
        "training_ready=false"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "RagSftV2HnVariantMaterializationError",
    "build_approved_hn_variant",
    "materialize_rag_sft_v2_hn_variants",
]
