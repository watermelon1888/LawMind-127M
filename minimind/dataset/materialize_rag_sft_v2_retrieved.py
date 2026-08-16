"""把 v2 Oracle clean 逐条执行真实检索并发布 HN 候选审计资产。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rag.answering import EvidencePackager, EvidencePackagingError
from rag.retrieval import RankedArticle

from . import rag_sft_v2_contract as contract


CONTEXT_LIMIT = 768
MAX_OUTPUT_TOKENS = 150
MAX_PROMPT_TOKENS = CONTEXT_LIMIT - MAX_OUTPUT_TOKENS
PIPELINE = "rag_sft_v2_retrieved_materialization"
REPORT_FILENAME = "report.json"
CANDIDATES_FILENAME = "hn-candidates.jsonl"
LOCATORS_FILENAME = "locators.jsonl"
FAILURES_FILENAME = "retrieval-failures.jsonl"
HASH_FILENAME = "sha256-manifest.txt"


class RagSftV2RetrievedMaterializationError(RuntimeError):
    """v2 真实检索物化输入或输出不满足审计边界。"""


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


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftV2RetrievedMaterializationError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise RagSftV2RetrievedMaterializationError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, start=1):
                if not line.strip():
                    raise ValueError(f"{description}不允许空行: {number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{description}第 {number} 条必须是 object")
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        if isinstance(error, RagSftV2RetrievedMaterializationError):
            raise
        raise RagSftV2RetrievedMaterializationError(f"无法读取{description}: {path}") from error
    if not rows:
        raise RagSftV2RetrievedMaterializationError(f"{description}不能为空")
    return rows


def _verify_bound_file(metadata: object, description: str) -> Path:
    if not isinstance(metadata, dict):
        raise RagSftV2RetrievedMaterializationError(f"缺少{description}身份")
    path_value = metadata.get("path")
    if not isinstance(path_value, str) or type(metadata.get("bytes")) is not int or not isinstance(metadata.get("sha256"), str):
        raise RagSftV2RetrievedMaterializationError(f"{description}身份字段无效")
    path = Path(path_value).resolve()
    if not path.is_file() or path.stat().st_size != metadata["bytes"] or _sha256(path) != metadata["sha256"]:
        raise RagSftV2RetrievedMaterializationError(f"{description}身份已变化")
    return path


def _load_clean(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest = _load_json(manifest_path, "Oracle clean manifest")
    readiness = manifest.get("readiness")
    protocol = manifest.get("protocol")
    if (
        manifest.get("pipeline") != "rag_sft_v2_oracle_clean_materialization"
        or manifest.get("release_status") != "stage_5_oracle_clean_audited"
        or not isinstance(readiness, dict)
        or readiness.get("oracle_clean_audited") is not True
        or not isinstance(protocol, dict)
        or protocol.get("context_limit") != CONTEXT_LIMIT
        or protocol.get("max_output_tokens") != MAX_OUTPUT_TOKENS
        or protocol.get("truncation") != "forbidden"
    ):
        raise RagSftV2RetrievedMaterializationError("Oracle clean 不是 768/150 v2 审计资产")
    clean_path = _verify_bound_file(manifest.get("output", {}).get("oracle_clean"), "Oracle clean")
    target_path = _verify_bound_file(manifest.get("output", {}).get("evaluation_targets"), "evaluation targets")
    clean_rows = _load_jsonl(clean_path, "Oracle clean")
    target_rows = _load_jsonl(target_path, "evaluation targets")
    if len(clean_rows) != len(target_rows):
        raise RagSftV2RetrievedMaterializationError("Oracle clean 与 evaluation targets 数量不一致")
    targets = {row.get("query_id"): row for row in target_rows}
    parents: list[dict[str, Any]] = []
    for row in clean_rows:
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or query_id in {item.get("query_id") for item in parents}:
            raise RagSftV2RetrievedMaterializationError("Oracle clean query_id 无效或重复")
        target = targets.get(query_id)
        if target is None or set(row) != {"id", "query_id", "variant", "query_original", "visible_chunk_ids", "required_chunk_ids", "citations", "conversations"}:
            raise RagSftV2RetrievedMaterializationError(f"Oracle clean 记录字段或 target 无效: {query_id}")
        required = row.get("required_chunk_ids")
        if not isinstance(required, list) or not 1 <= len(required) <= 3 or len(required) != len(set(required)):
            raise RagSftV2RetrievedMaterializationError(f"required GT 无效: {query_id}")
        if target.get("required_chunk_ids") != required or not isinstance(target.get("claims"), list):
            raise RagSftV2RetrievedMaterializationError(f"evaluation target 与 clean 不一致: {query_id}")
        parents.append(row)
    return manifest, parents, targets


def _diagnostics(ranked: tuple[RankedArticle, ...]) -> list[dict[str, object]]:
    return [
        {
            "chunk_id": item.article.chunk_id,
            "rrf_rank": item.rrf_rank,
            "rrf_score": item.rrf_score,
            "rerank_score": item.rerank_score,
        }
        for item in ranked
    ]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> str:
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return payload


def materialize_rag_sft_v2_retrieved(*, manifest_path: Path, retriever, evidence_packager: EvidencePackager, retrieval_identity: dict[str, object], output_dir: Path) -> dict[str, object]:
    """用原始 query 执行当前 top-5 检索，输出待审核 HN 候选，不发布训练数据。"""
    if not callable(getattr(retriever, "search", None)):
        raise TypeError("retriever 必须提供 search")
    if not isinstance(evidence_packager, EvidencePackager):
        raise TypeError("evidence_packager 必须是 EvidencePackager")
    if evidence_packager.context_limit != CONTEXT_LIMIT or evidence_packager.max_output_tokens != MAX_OUTPUT_TOKENS:
        raise ValueError("v2 物化必须使用 768 上下文和 150 输出预算")
    retrieval_identity = json.loads(json.dumps(retrieval_identity, ensure_ascii=False, allow_nan=False))
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RagSftV2RetrievedMaterializationError(f"物化输出目录必须是新目录: {output_dir}")
    manifest, parents, _ = _load_clean(Path(manifest_path).resolve())
    candidates: list[dict[str, object]] = []
    locators: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    dispositions: Counter[str] = Counter()
    for parent in parents:
        query_id = parent["query_id"]
        required = tuple(parent["required_chunk_ids"])
        try:
            ranked = tuple(retriever.search(parent["query_original"]))
            if any(not isinstance(item, RankedArticle) for item in ranked):
                raise TypeError("检索结果必须由 RankedArticle 组成")
            if not ranked:
                dispositions["empty_retrieval"] += 1
                failures.append({"query_id": query_id, "reason": "empty_retrieval"})
                locators.append({"query_id": query_id, "status": "retrieval_failure", "retrieved_candidates": [], "packaged_chunk_ids": []})
                continue
            package, prompt_tokens = evidence_packager.build(parent["query_original"], tuple(item.article for item in ranked))
        except EvidencePackagingError as error:
            dispositions["packaging_failed"] += 1
            failures.append({"query_id": query_id, "reason": "packaging_failed", "detail": str(error)})
            locators.append({"query_id": query_id, "status": "overbudget", "retrieved_candidates": _diagnostics(locals().get("ranked", ())), "packaged_chunk_ids": [], "prompt_tokens": None})
            continue
        except Exception as error:
            dispositions["retrieval_failed"] += 1
            failures.append({"query_id": query_id, "reason": "retrieval_failed", "detail": type(error).__name__})
            locators.append({"query_id": query_id, "status": "retrieval_failure", "retrieved_candidates": [], "packaged_chunk_ids": []})
            continue
        visible = tuple(item.article.chunk_id for item in ranked[: len(package.evidence)])
        extras = tuple(chunk_id for chunk_id in visible if chunk_id not in required)
        if not set(required).issubset(visible):
            status = "missing_required_gt"
            dispositions[status] += 1
            failures.append({"query_id": query_id, "reason": status, "required_chunk_ids": list(required), "packaged_chunk_ids": list(visible)})
        elif extras:
            status = "hn_candidate"
            dispositions[status] += 1
            candidates.append({
                "variant_id": f"{query_id}:hn:retrieved",
                "query_id": query_id,
                "visible_chunk_ids": list(visible),
                "required_chunk_ids": list(required),
                "source": "retrieved",
                "retrieval_identity": retrieval_identity,
                "retrieved_candidates": _diagnostics(ranked),
                "prompt_tokens": prompt_tokens,
                "non_gt_chunk_ids": list(extras),
                "review_decision": "pending",
            })
        else:
            status = "all_required_no_hn"
            dispositions[status] += 1
        locators.append({"query_id": query_id, "status": status, "required_chunk_ids": list(required), "retrieved_candidates": _diagnostics(ranked), "packaged_chunk_ids": list(visible), "extra_non_gt_chunk_ids": list(extras), "prompt_tokens": prompt_tokens})
    report = {
        "pipeline": PIPELINE,
        "release_status": "stage_7_retrieved_candidates",
        "policy": {"context_limit": CONTEXT_LIMIT, "max_output_tokens": MAX_OUTPUT_TOKENS, "max_prompt_tokens": MAX_PROMPT_TOKENS, "top_k": 5, "query_mode": "original_only", "retrieval_order_preserved": True, "selector": "disabled", "truncation": "forbidden"},
        "input": {"oracle_clean_manifest": _identity(Path(manifest_path).resolve()), "retrieval_identity": retrieval_identity},
        "records": {"scanned": len(parents), "candidates": len(candidates), "failures": len(failures), "dispositions": dict(sorted(dispositions.items()))},
        "readiness": {"retrieved_materialization_complete": True, "hn_semantic_review_complete": False, "final_training_release": False, "training_ready": False},
        "outputs": {"candidates": CANDIDATES_FILENAME, "locators": LOCATORS_FILENAME, "failures": FAILURES_FILENAME, "report": REPORT_FILENAME, "sha256_manifest": HASH_FILENAME},
        "complete": True,
    }
    output_dir.mkdir(parents=True)
    rows = [(CANDIDATES_FILENAME, candidates), (LOCATORS_FILENAME, locators), (FAILURES_FILENAME, failures)]
    payloads: list[tuple[str, str]] = []
    for filename, data in rows:
        payloads.append((filename, _write_jsonl(output_dir / filename, data)))
    report_payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    (output_dir / REPORT_FILENAME).write_text(report_payload, encoding="utf-8", newline="\n")
    payloads.append((REPORT_FILENAME, report_payload))
    hash_payload = "".join(f"{hashlib.sha256((output_dir / filename).read_bytes()).hexdigest()}  {filename}\n" for filename, _ in payloads)
    (output_dir / HASH_FILENAME).write_text(hash_payload, encoding="ascii", newline="\n")
    return report


__all__ = ["RagSftV2RetrievedMaterializationError", "materialize_rag_sft_v2_retrieved"]
