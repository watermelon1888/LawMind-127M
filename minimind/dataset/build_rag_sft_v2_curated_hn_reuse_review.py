"""从已闭合的真实检索证据级裁决构造可追溯的 curated HN 审阅资产。

本脚本只选择已经满足 ``hard_negative + none + approve_evidence`` 的真实
法条证据。原检索包即使因其他 non-GT 证据被排除，只要这一条证据本身经过
双审并获准，就可以与同一 query 的全部 required GT 组成新的 curated 候选。

该资产不投影训练对话、不加载 Tokenizer、不做上下文长度审核，也不物化正式
训练 HN。后续发布器必须再次读取本资产的双审与主审账本。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_RELEASE = (
    PROJECT_ROOT
    / "minimind"
    / "dataset"
    / "RAG-SFT"
    / "releases"
    / "v2"
    / "rag-sft-v2-training-release-619-v1-20260813"
)
DEFAULT_REVIEW_ROOT = (
    PROJECT_ROOT
    / "minimind"
    / "dataset"
    / "RAG-SFT"
    / "review"
    / "v2"
    / "hn-semantic-review-v1-20260813"
)
DEFAULT_CLOSURE_ROOT = DEFAULT_REVIEW_ROOT / "semantic-review-closure-v2-20260813-final-v1"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "minimind"
    / "dataset"
    / "RAG-SFT"
    / "review"
    / "v2"
    / "curated-hn-reuse-review-v1-20260814"
)

MANIFEST_FILENAME = "manifest.json"
HASH_FILENAME = "manifest.sha256"
VARIANTS_FILENAME = "curated-hn-variants.jsonl"
ADVERSARIAL_FILENAME = "adversarial.jsonl"
LEGAL_FILENAME = "legal-support.jsonl"
ADJUDICATION_FILENAME = "adjudication.jsonl"
SOURCE_LEDGER_FILENAME = "source-evidence-ledger.jsonl"
README_FILENAME = "README.md"
MAX_CURATED_HN = 210


class CuratedHnReuseReviewError(RuntimeError):
    """curated HN 复用审阅的输入身份、裁决或输出边界未闭合。"""


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


def _serialize_jsonl(rows: Iterable[dict[str, object]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    )


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CuratedHnReuseReviewError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise CuratedHnReuseReviewError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise CuratedHnReuseReviewError(
                        f"{description}存在空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CuratedHnReuseReviewError(
                        f"{description}第 {line_number} 行不是 object"
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, CuratedHnReuseReviewError):
            raise
        raise CuratedHnReuseReviewError(f"无法读取{description}: {path}") from error
    return rows


def _verify_manifest(path: Path) -> str:
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise CuratedHnReuseReviewError(f"manifest 或 SHA-256 清单不存在: {path}")
    digest = _sha256(path)
    try:
        matches = [
            parts[0].lower()
            for line in sidecar.read_text(encoding="ascii").splitlines()
            if line
            for parts in [line.split("  ", 1)]
            if len(parts) == 2 and Path(parts[1]).name == path.name
        ]
        if any(len(item) != 64 for item in matches):
            raise ValueError
        for item in matches:
            int(item, 16)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise CuratedHnReuseReviewError(f"manifest SHA-256 清单无效: {sidecar}") from error
    if matches != [digest]:
        raise CuratedHnReuseReviewError(f"manifest SHA-256 校验失败: {path}")
    return digest


def _bound_path(metadata: object, description: str) -> Path:
    if not isinstance(metadata, dict):
        raise CuratedHnReuseReviewError(f"缺少{description}身份")
    path_value = metadata.get("path")
    if not isinstance(path_value, str):
        raise CuratedHnReuseReviewError(f"{description}路径无效")
    path = Path(path_value).resolve()
    if (
        not path.is_file()
        or type(metadata.get("bytes")) is not int
        or not isinstance(metadata.get("sha256"), str)
        or path.stat().st_size != metadata["bytes"]
        or _sha256(path) != metadata["sha256"]
    ):
        raise CuratedHnReuseReviewError(f"{description}身份已变化: {path}")
    return path


def _index_unique(
    rows: Iterable[dict[str, Any]], key: str, description: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        value = row.get(key)
        if not isinstance(value, str) or not value:
            raise CuratedHnReuseReviewError(f"{description}缺少 {key}")
        if value in indexed:
            raise CuratedHnReuseReviewError(f"{description}{key}重复: {value}")
        indexed[value] = row
    return indexed


def _assistant_summary(clean: dict[str, Any]) -> str:
    try:
        value = json.loads(clean["conversations"][2]["content"])
        summary = value["summary"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise CuratedHnReuseReviewError("clean assistant summary 无法解析") from error
    if not isinstance(summary, str) or not summary:
        raise CuratedHnReuseReviewError("clean assistant summary 无效")
    return summary


def _canonical_from_clean(clean: dict[str, Any]) -> dict[str, object]:
    query_id = clean.get("query_id")
    query_original = clean.get("query_original")
    required = clean.get("required_chunk_ids")
    if (
        not isinstance(query_id, str)
        or not isinstance(query_original, str)
        or not isinstance(required, list)
        or not required
        or any(not isinstance(item, str) or not item for item in required)
        or len(required) != len(set(required))
    ):
        raise CuratedHnReuseReviewError("clean 回答事实字段无效")
    return {
        "query_id": query_id,
        "query_original": query_original,
        "summary": _assistant_summary(clean),
        "required_chunk_ids": list(required),
    }


def _source_evidence_by_id(source_package: dict[str, Any]) -> dict[str, dict[str, Any]]:
    evidence = source_package.get("evidence")
    visible = source_package.get("visible_chunk_ids")
    if not isinstance(evidence, list) or not isinstance(visible, list):
        raise CuratedHnReuseReviewError("source package evidence 或 visible_chunk_ids 无效")
    indexed: dict[str, dict[str, Any]] = {}
    chunks: list[str] = []
    for item in evidence:
        evidence_id = item.get("evidence_id") if isinstance(item, dict) else None
        chunk_id = item.get("chunk_id") if isinstance(item, dict) else None
        if not isinstance(evidence_id, str) or not isinstance(chunk_id, str):
            raise CuratedHnReuseReviewError("source package evidence 身份无效")
        if evidence_id in indexed:
            raise CuratedHnReuseReviewError(f"source package evidence_id 重复: {evidence_id}")
        indexed[evidence_id] = item
        chunks.append(chunk_id)
    if chunks != visible:
        raise CuratedHnReuseReviewError("source package evidence 顺序与 visible_chunk_ids 不一致")
    return indexed


def _stable_visible(required: list[str], query_id: str, chunk_id: str) -> list[str]:
    """以稳定哈希决定插入位置，不把 curated HN 固定放在末尾。"""

    digest = hashlib.sha256(f"{query_id}\0{chunk_id}".encode("utf-8")).digest()
    position = int.from_bytes(digest[:8], "big") % (len(required) + 1)
    return [*required[:position], chunk_id, *required[position:]]


def _load_base_release(base_release: Path) -> tuple[dict[str, Any], str, Path, dict[str, dict[str, Any]], set[str]]:
    manifest_path = base_release / MANIFEST_FILENAME
    manifest_sha = _verify_manifest(manifest_path)
    manifest = _load_json(manifest_path, "当前 619 release manifest")
    if (
        manifest.get("pipeline") != "rag_sft_v2_training_release"
        or manifest.get("release_status") != "formal_training_candidate"
        or manifest.get("records", {}).get("total") != 619
        or manifest.get("records", {}).get("clean") != 549
        or manifest.get("records", {}).get("hard_negative") != 70
        or manifest.get("readiness", {}).get("training_ready") is not True
    ):
        raise CuratedHnReuseReviewError("当前 release 不是已冻结的 619 条正式基线")
    candidate_path = _bound_path(
        manifest.get("output", {}).get("training_candidate"), "当前 release candidate"
    )
    rows = _load_jsonl(candidate_path, "当前 release candidate")
    if len(rows) != 619:
        raise CuratedHnReuseReviewError("当前 release candidate 条数不为 619")
    clean_rows = [row for row in rows if row.get("variant") == "clean"]
    hn_rows = [row for row in rows if row.get("variant") == "hard_negative"]
    clean_by_query = _index_unique(clean_rows, "query_id", "当前 clean ")
    hn_by_query = _index_unique(hn_rows, "query_id", "当前 HN ")
    if len(clean_by_query) != 549 or len(hn_by_query) != 70 or not set(hn_by_query) <= set(clean_by_query):
        raise CuratedHnReuseReviewError("当前 release clean/HN 配对身份不闭合")
    return manifest, manifest_sha, candidate_path, clean_by_query, set(hn_by_query)


def _load_closure_candidates(
    *, closure_root: Path, review_root: Path, clean_by_query: dict[str, dict[str, Any]], existing_hn_queries: set[str]
) -> tuple[list[dict[str, Any]], list[dict[str, object]], list[dict[str, object]]]:
    """加载 19 批闭合账本并返回每个无 HN query 的一个稳定候选。"""

    global_manifest_path = closure_root / MANIFEST_FILENAME
    global_manifest_sha = _verify_manifest(global_manifest_path)
    global_manifest = _load_json(global_manifest_path, "最终语义审核闭合 manifest")
    if (
        global_manifest.get("pipeline") != "rag_sft_v2_hn_semantic_review_closure"
        or global_manifest.get("records", {}).get("batches") != 19
        or global_manifest.get("records", {}).get("packages") != 452
        or global_manifest.get("records", {}).get("non_gt_evidence") != 880
        or global_manifest.get("policy", {}).get("formal_hn_materialized") is not False
        or global_manifest.get("policy", {}).get("training_ready") is not False
        or global_manifest.get("identity_quarantine", {}).get("enforced") is not True
    ):
        raise CuratedHnReuseReviewError("最终语义审核闭合资产状态或 batch 7 隔离无效")
    batch_manifests = global_manifest.get("batch_manifests")
    if not isinstance(batch_manifests, list) or len(batch_manifests) != 19:
        raise CuratedHnReuseReviewError("最终语义审核闭合未绑定 19 个批次 manifest")

    candidates: list[dict[str, Any]] = []
    source_batch_identities: list[dict[str, object]] = []
    closure_batch_identities: list[dict[str, object]] = []
    for item in sorted(batch_manifests, key=lambda value: value.get("batch", 0)):
        batch = item.get("batch") if isinstance(item, dict) else None
        source_sha = item.get("source_batch_sha256") if isinstance(item, dict) else None
        if type(batch) is not int or not 1 <= batch <= 19 or not isinstance(source_sha, str):
            raise CuratedHnReuseReviewError("最终闭合批次身份无效")
        batch_name = f"batch-{batch:02d}"
        closure_batch_dir = closure_root / batch_name
        batch_manifest_path = closure_batch_dir / MANIFEST_FILENAME
        batch_manifest_sha = _verify_manifest(batch_manifest_path)
        batch_manifest = _load_json(batch_manifest_path, f"{batch_name} manifest")
        closure_manifest_meta = item.get("manifest")
        review_batch_meta = batch_manifest.get("inputs", {}).get("review_batch")
        expected_source_name = f"review-batch-{batch:02d}.jsonl"
        source_path = review_root / expected_source_name
        if (
            batch_manifest.get("pipeline") != "rag_sft_v2_hn_semantic_review_closure"
            or batch_manifest.get("batch") != batch
            or not isinstance(closure_manifest_meta, dict)
            or Path(str(closure_manifest_meta.get("path", ""))).resolve()
            != batch_manifest_path.resolve()
            or closure_manifest_meta.get("bytes") != batch_manifest_path.stat().st_size
            or closure_manifest_meta.get("sha256") != _sha256(batch_manifest_path)
            or not isinstance(review_batch_meta, dict)
            or review_batch_meta.get("sha256") != source_sha
            or Path(str(review_batch_meta.get("path", ""))).name != expected_source_name
            or not source_path.is_file()
            or source_path.stat().st_size != review_batch_meta.get("bytes")
            or _sha256(source_path) != source_sha
            or batch_manifest.get("policy", {}).get("training_ready") is not False
        ):
            raise CuratedHnReuseReviewError(f"{batch_name} source batch 身份或状态不闭合")
        source_rows = _load_jsonl(source_path, f"{batch_name} source batch")
        if len(source_rows) != review_batch_meta.get("records"):
            raise CuratedHnReuseReviewError(f"{batch_name} source package 数量不闭合")
        source_by_variant = _index_unique(source_rows, "variant_id", f"{batch_name} source package ")
        adjudication_path = _bound_path(
            batch_manifest.get("outputs", {}).get("adjudication"), f"{batch_name} adjudication"
        )
        adjudications = _load_jsonl(adjudication_path, f"{batch_name} adjudication")
        source_batch_identities.append(_identity(source_path, records=len(source_rows)))
        closure_batch_identities.append(
            {
                "batch": batch,
                "manifest": {**_identity(batch_manifest_path), "manifest_sha256": batch_manifest_sha},
                "adjudication": _identity(adjudication_path, records=len(adjudications)),
            }
        )
        for line_number, row in enumerate(adjudications, start=1):
            if row.get("record_type") != "evidence_adjudication":
                continue
            if (
                row.get("source_batch") != expected_source_name
                or row.get("source_batch_sha256") != source_sha
                or row.get("is_required_gt") is not False
                or row.get("adversarial_label") != "hard_negative"
                or row.get("legal_support") != "none"
                or row.get("decision") != "approve_evidence"
            ):
                continue
            query_id = row.get("query_id")
            variant_id = row.get("variant_id")
            evidence_id = row.get("evidence_id")
            chunk_id = row.get("chunk_id")
            if (
                not isinstance(query_id, str)
                or not isinstance(variant_id, str)
                or not isinstance(evidence_id, str)
                or not isinstance(chunk_id, str)
                or query_id not in clean_by_query
                or query_id in existing_hn_queries
            ):
                continue
            source_package = source_by_variant.get(variant_id)
            if source_package is None or source_package.get("query_id") != query_id:
                raise CuratedHnReuseReviewError(
                    f"{batch_name} evidence 裁决无法回溯 source package: {query_id}/{evidence_id}"
                )
            canonical = _canonical_from_clean(clean_by_query[query_id])
            if (
                source_package.get("query_original") != canonical["query_original"]
                or source_package.get("required_chunk_ids") != canonical["required_chunk_ids"]
            ):
                raise CuratedHnReuseReviewError(
                    f"source package 与 clean 回答事实不一致: {query_id}"
                )
            source_evidence = _source_evidence_by_id(source_package).get(evidence_id)
            if (
                source_evidence is None
                or source_evidence.get("chunk_id") != chunk_id
                or source_evidence.get("is_required_gt") is not False
            ):
                raise CuratedHnReuseReviewError(
                    f"source evidence 身份不闭合: {query_id}/{evidence_id}"
                )
            candidates.append(
                {
                    "batch": batch,
                    "source_batch": expected_source_name,
                    "source_batch_sha256": source_sha,
                    "source_package": source_package,
                    "source_evidence": source_evidence,
                    "upstream_adjudication": row,
                    "upstream_adjudication_path": str(adjudication_path.resolve()),
                    "upstream_adjudication_sha256": _sha256(adjudication_path),
                    "upstream_adjudication_line": line_number,
                    "closure_global_manifest_sha256": global_manifest_sha,
                }
            )
    return candidates, source_batch_identities, closure_batch_identities


def build_rag_sft_v2_curated_hn_reuse_review(
    *, base_release: Path, closure_root: Path, review_root: Path, output_dir: Path
) -> dict[str, object]:
    """发布不可覆盖的 98 条以内 curated HN 双审与主审审阅资产。"""

    base_release = Path(base_release).resolve()
    closure_root = Path(closure_root).resolve()
    review_root = Path(review_root).resolve()
    output_dir = Path(output_dir).resolve()
    partial_dir = output_dir.with_name(output_dir.name + ".partial")
    if output_dir.exists() or partial_dir.exists():
        raise CuratedHnReuseReviewError(f"输出目录或临时目录已存在: {output_dir}")
    base_manifest, base_manifest_sha, base_candidate_path, clean_by_query, existing_hn_queries = _load_base_release(base_release)
    candidates, source_batch_identities, closure_batch_identities = _load_closure_candidates(
        closure_root=closure_root,
        review_root=review_root,
        clean_by_query=clean_by_query,
        existing_hn_queries=existing_hn_queries,
    )
    chosen_by_query: dict[str, dict[str, Any]] = {}
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item["source_package"]["query_id"],
            item["batch"],
            item["upstream_adjudication"]["evidence_id"],
        ),
    ):
        query_id = candidate["source_package"]["query_id"]
        chosen_by_query.setdefault(query_id, candidate)
    chosen = [chosen_by_query[key] for key in sorted(chosen_by_query)]
    if not chosen or len(chosen) > MAX_CURATED_HN:
        raise CuratedHnReuseReviewError("curated HN 数量为空或超过 210 条软上限")

    variants: list[dict[str, object]] = []
    adversarial_rows: list[dict[str, object]] = []
    legal_rows: list[dict[str, object]] = []
    adjudication_rows: list[dict[str, object]] = []
    source_ledger_rows: list[dict[str, object]] = []
    position_counts: Counter[int] = Counter()
    for candidate in chosen:
        package = candidate["source_package"]
        evidence = candidate["source_evidence"]
        upstream = candidate["upstream_adjudication"]
        query_id = package["query_id"]
        evidence_id = evidence["evidence_id"]
        chunk_id = evidence["chunk_id"]
        canonical = _canonical_from_clean(clean_by_query[query_id])
        visible = _stable_visible(canonical["required_chunk_ids"], query_id, chunk_id)
        position = visible.index(chunk_id) + 1
        position_counts[position] += 1
        variant_id = f"{query_id}:hn:curated:reuse-v1"
        review_id = f"{variant_id}:{chunk_id}"
        retrieval_identity = (
            "curated:reuse:closure-manifest-sha256:"
            f"{candidate['closure_global_manifest_sha256']}:source-batch-sha256:"
            f"{candidate['source_batch_sha256']}:evidence:{evidence_id}"
        )
        variants.append(
            {
                "variant_id": variant_id,
                "query_id": query_id,
                "visible_chunk_ids": visible,
                "source": "curated",
                "retrieval_identity": retrieval_identity,
                "non_gt_labels": [{"chunk_id": chunk_id, "label": "hard_negative"}],
                "review_decision": "approved",
            }
        )
        adversarial_rows.append(
            {
                "review_id": review_id,
                "query_id": query_id,
                "variant_id": variant_id,
                "chunk_id": chunk_id,
                "adversarial_label": "hard_negative",
                "reason": "该真实法条已在冻结的独立对抗审阅中判为 hard_negative；新候选仅保留此条与全部 required GT，仍具主题或术语混淆性但不支持目标结论。",
                "reviewer": "curated_reuse_adversarial_review_v1",
            }
        )
        legal_rows.append(
            {
                "review_id": review_id,
                "query_id": query_id,
                "variant_id": variant_id,
                "chunk_id": chunk_id,
                "legal_support": "none",
                "reason": "该真实法条已在冻结的独立法律支持审阅中判为 none；新候选保留原 query、summary 和全部 required GT，不将相近主题误作支持。",
                "reviewer": "curated_reuse_legal_support_review_v1",
            }
        )
        adjudication_rows.append(
            {
                "review_id": review_id,
                "query_id": query_id,
                "variant_id": variant_id,
                "chunk_id": chunk_id,
                "adversarial_label": "hard_negative",
                "legal_support": "none",
                "decision": "approve_evidence",
                "decision_reason": "同时满足 hard_negative 与 none；原真实检索包如被排除，原因是其他 non-GT，不影响本 curated variant 仅含此条准入证据与 required GT 的证据级准入。",
                "adjudicator": "curated_reuse_adjudication_v1",
            }
        )
        source_ledger_rows.append(
            {
                "review_id": review_id,
                "query_id": query_id,
                "variant_id": variant_id,
                "chunk_id": chunk_id,
                "source": "curated",
                "source_batch": candidate["source_batch"],
                "source_batch_sha256": candidate["source_batch_sha256"],
                "source_package_variant_id": package["variant_id"],
                "source_evidence_id": evidence_id,
                "upstream_adjudication_path": candidate["upstream_adjudication_path"],
                "upstream_adjudication_sha256": candidate["upstream_adjudication_sha256"],
                "upstream_adjudication_line": candidate["upstream_adjudication_line"],
                "upstream_adversarial_label": upstream["adversarial_label"],
                "upstream_legal_support": upstream["legal_support"],
                "upstream_decision": upstream["decision"],
                "source_package_decision_not_reused": True,
            }
        )

    payloads = {
        VARIANTS_FILENAME: _serialize_jsonl(variants),
        ADVERSARIAL_FILENAME: _serialize_jsonl(adversarial_rows),
        LEGAL_FILENAME: _serialize_jsonl(legal_rows),
        ADJUDICATION_FILENAME: _serialize_jsonl(adjudication_rows),
        SOURCE_LEDGER_FILENAME: _serialize_jsonl(source_ledger_rows),
    }
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest: dict[str, object] = {
        "pipeline": "rag_sft_v2_curated_hn_reuse_semantic_review",
        "release_status": "curated_hn_reuse_semantic_review_closed",
        "inputs": {
            "base_release_manifest": {
                **_identity(base_release / MANIFEST_FILENAME),
                "manifest_sha256": base_manifest_sha,
            },
            "base_training_candidate": _identity(base_candidate_path, records=619),
            "closure_manifest": {
                **_identity(closure_root / MANIFEST_FILENAME),
                "manifest_sha256": _sha256(closure_root / MANIFEST_FILENAME),
            },
            "source_batches": source_batch_identities,
            "closure_batches": closure_batch_identities,
        },
        "records": {
            "base_clean": 549,
            "base_retrieved_hn": 70,
            "unpaired_clean_before": 479,
            "upstream_approved_evidence_for_unpaired_queries": len(candidates),
            "unique_queries_with_upstream_approved_evidence": len(chosen),
            "curated_hn_variants": len(variants),
            "adversarial": {"hard_negative": len(adversarial_rows)},
            "legal_support": {"none": len(legal_rows)},
            "adjudication": {"approve_evidence": len(adjudication_rows)},
            "hn_position": {str(key): position_counts[key] for key in sorted(position_counts)},
        },
        "policy": {
            "selection": "每个当前无 HN query 最多选择一条已有双审批准的真实法条证据",
            "curated_source": True,
            "source_candidates_modified": False,
            "base_619_release_modified": False,
            "formal_hn_materialized": False,
            "context_length_audit": "deferred",
            "training_ready": False,
            "quarantined_batch7_legacy_not_referenced": True,
        },
        "validation": {
            "strict_utf8_jsonl": True,
            "base_release_identity_closed": True,
            "all_19_closure_batches_bound": True,
            "source_batch_hashes_bound": True,
            "only_unpaired_queries_selected": True,
            "one_curated_variant_per_query": True,
            "all_candidates_are_real_article_evidence": True,
            "all_evidence_pass_upstream_hard_negative_and_none": True,
            "curated_review_admission_rule_closed": True,
            "no_tokenizer_or_context_length_audit_run": True,
            "quarantined_batch7_legacy_not_referenced": True,
        },
        "output": {
            "curated_hn_variants": _payload_identity(output_dir / VARIANTS_FILENAME, payloads[VARIANTS_FILENAME], records=len(variants)),
            "adversarial": _payload_identity(output_dir / ADVERSARIAL_FILENAME, payloads[ADVERSARIAL_FILENAME], records=len(adversarial_rows)),
            "legal_support": _payload_identity(output_dir / LEGAL_FILENAME, payloads[LEGAL_FILENAME], records=len(legal_rows)),
            "adjudication": _payload_identity(output_dir / ADJUDICATION_FILENAME, payloads[ADJUDICATION_FILENAME], records=len(adjudication_rows)),
            "source_evidence_ledger": _payload_identity(output_dir / SOURCE_LEDGER_FILENAME, payloads[SOURCE_LEDGER_FILENAME], records=len(source_ledger_rows)),
        },
        "complete": True,
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    readme_payload = (
        "# RAG-SFT v2 curated HN 复用语义审阅\n\n"
        "本目录从 19 批真实检索 HN 最终闭合账本中，只选择已经逐条满足 "
        "`hard_negative + none + approve_evidence` 的真实法条证据。\n\n"
        "每个新候选保持原 query、summary 和全部 required GT 不变，仅加入一条已批准的真实法条干扰项；"
        "原检索包的 package-level 决定不会被复用或改写。\n\n"
        "该目录仅为语义和身份冻结资产：`formal_hn_materialized=false`、"
        "`context_length_audit=deferred`、`training_ready=false`。不包含训练候选，不加载 Tokenizer，"
        "也不进行上下文长度或标签审计。\n"
    )
    payloads[MANIFEST_FILENAME] = manifest_payload
    payloads[README_FILENAME] = readme_payload
    try:
        partial_dir.mkdir(parents=True, exist_ok=False)
        for name, payload in payloads.items():
            (partial_dir / name).write_text(payload, encoding="utf-8", newline="\n")
        hash_payload = "".join(
            f"{_sha256(partial_dir / name)}  {name}\n" for name in payloads
        )
        (partial_dir / HASH_FILENAME).write_text(hash_payload, encoding="ascii", newline="\n")
        partial_dir.replace(output_dir)
    except OSError as error:
        raise CuratedHnReuseReviewError(
            "无法原子发布 curated HN 复用审阅资产；已保留临时目录以便审计"
        ) from error
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-release", type=Path, default=DEFAULT_BASE_RELEASE)
    parser.add_argument("--closure-root", type=Path, default=DEFAULT_CLOSURE_ROOT)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        manifest = build_rag_sft_v2_curated_hn_reuse_review(
            base_release=args.base_release,
            closure_root=args.closure_root,
            review_root=args.review_root,
            output_dir=args.output_dir,
        )
    except (CuratedHnReuseReviewError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    records = manifest["records"]
    print(
        "RAG_SFT_V2_CURATED_REUSE_REVIEW_OK "
        f"curated={records['curated_hn_variants']} training_ready=false"
    )


if __name__ == "__main__":
    main()
