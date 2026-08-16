"""发布 RAG-SFT v2 的 170 条主评估与 RoPE 外推评估输入。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from rag.answering import AnswerPromptTokenCounter, EvidencePackage, EvidencePackager
from rag.core import Evidence
from rag.knowledge import ArticleRepository


PIPELINE = "rag_sft_v2_evaluation_inputs_v1"
EXPECTED_SOURCE_LEGAL_QUERY = 140
REPACKAGED_EXTRAPOLATION_QUERY_IDS = ("Q048", "Q063")
EXACT_LOOKUP_EXTRAPOLATION_QUERY_IDS = ("Q033", "Q091")
EXTRAPOLATION_QUERY_IDS = tuple(
    sorted(REPACKAGED_EXTRAPOLATION_QUERY_IDS + EXACT_LOOKUP_EXTRAPOLATION_QUERY_IDS)
)
EXPECTED_LEGAL_QUERY = EXPECTED_SOURCE_LEGAL_QUERY
EXPECTED_EXACT_LOOKUP = 30
EXPECTED_RECORDS = EXPECTED_LEGAL_QUERY + EXPECTED_EXACT_LOOKUP
PRIMARY_RECORDS = EXPECTED_RECORDS - len(EXTRAPOLATION_QUERY_IDS)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: str | Path, *, records: int | None = None) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"输入文件不存在: {resolved}")
    value: dict[str, Any] = {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }
    if records is not None:
        value["records"] = records
    return value


def _load_verified_json(path: str | Path, description: str) -> tuple[dict[str, Any], str]:
    resolved = Path(path).resolve()
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"{description} 或相邻 SHA-256 不存在: {resolved}")
    digest = _sha256_file(resolved)
    expected = f"{digest}  {resolved.name}"
    lines = [line for line in sidecar.read_text(encoding="utf-8").splitlines() if line]
    if lines != [expected]:
        raise ValueError(f"{description} SHA-256 校验失败")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} 不是有效 UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} 必须是 JSON object")
    return payload, digest


def _load_jsonl(path: str | Path, description: str) -> list[dict[str, Any]]:
    records = []
    with Path(path).resolve().open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"{description} 第 {line_number} 行为空")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{description} 第 {line_number} 行不是 JSON") from error
            if not isinstance(record, dict):
                raise ValueError(f"{description} 第 {line_number} 行不是 object")
            records.append(record)
    return records


def _write_immutable(path: Path, content: str) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"输出已存在，不能覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8", newline="\n")
    temporary.replace(path)
    digest = _sha256_file(path)
    hash_temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    hash_temporary.write_text(
        f"{digest}  {path.name}\n", encoding="utf-8", newline="\n"
    )
    hash_temporary.replace(sidecar)
    return digest


def _evidence_record(article: Any, index: int) -> dict[str, str]:
    return {
        "evidence_id": f"E{index}",
        "chunk_id": article.chunk_id,
        "law_name": article.law_name,
        "article_no": article.article_no,
        "content": article.content,
    }


def _required_chunk_ids(case: Mapping[str, Any], repository: ArticleRepository) -> list[str]:
    required = []
    for reference in case.get("gt_articles", []):
        if not isinstance(reference, dict):
            raise ValueError(f"{case.get('id')} 的 gt_articles 无效")
        article = repository.lookup(reference.get("law_name"), reference.get("article_no"))
        if article is None:
            raise ValueError(f"{case.get('id')} 的 GT 法条无法解析")
        required.append(article.chunk_id)
    if not required or len(required) != len(set(required)):
        raise ValueError(f"{case.get('id')} 的 required_chunk_ids 无效")
    return required


def build_evaluation_records(
    eval_cases: list[dict[str, Any]],
    retrieval_records: list[dict[str, Any]],
    repository: ArticleRepository,
    *,
    extrapolation_packager: EvidencePackager,
) -> list[dict[str, Any]]:
    """合并 140 条冻结构包结果与 30 条确定性精确查条结果。"""

    retrieval_by_id = {item.get("query_id"): item for item in retrieval_records}
    if len(retrieval_by_id) != EXPECTED_SOURCE_LEGAL_QUERY or None in retrieval_by_id:
        raise ValueError("冻结检索 records 必须包含 140 个唯一 query_id")
    output = []
    counts = {"legal_query": 0, "exact_lookup": 0}
    seen = set()
    for case in eval_cases:
        if case.get("expected_action") != "answer":
            continue
        query_id = case.get("id")
        query_type = case.get("query_type")
        query = case.get("query_original")
        if (
            not isinstance(query_id, str)
            or query_id in seen
            or query_type not in counts
            or not isinstance(query, str)
            or not query
        ):
            raise ValueError("开发集 answer 记录身份无效或重复")
        seen.add(query_id)
        required = _required_chunk_ids(case, repository)
        attribution: dict[str, Any]
        if query_type == "legal_query":
            retrieved = retrieval_by_id.get(query_id)
            if retrieved is None:
                raise ValueError(f"{query_id} 缺少冻结检索记录")
            packaging = retrieved.get("packaging")
            packaged_ids = packaging.get("packaged_chunk_ids") if isinstance(packaging, dict) else None
            if not isinstance(packaged_ids, list) or not packaged_ids:
                if (
                    query_id in REPACKAGED_EXTRAPOLATION_QUERY_IDS
                    and isinstance(packaging, dict)
                    and packaging.get("status") == "failed"
                    and packaged_ids == []
                ):
                    reranked = retrieved.get("retrieval", {}).get("reranked_top5_chunk_ids")
                    if not isinstance(reranked, list) or not reranked:
                        raise ValueError(f"{query_id} 缺少冻结 rerank top-5")
                    articles = [repository.get_by_chunk_id(chunk_id) for chunk_id in reranked]
                    package, prompt_tokens = extrapolation_packager.build(query, articles)
                    visible = [f"{item.law_name}#{item.article_no}" for item in package.evidence]
                    attribution = {
                        "source": "frozen_rerank_repackaged_1024",
                        "candidate_pool_complete": retrieved.get("retrieval", {})
                        .get("candidate_pool_metrics", {})
                        .get("complete_hit"),
                        "reranked_top5_complete": retrieved.get("retrieval", {})
                        .get("reranked_top5_metrics", {})
                        .get("complete_hit"),
                        "packaged_complete": set(required).issubset(visible),
                        "prompt_tokens": prompt_tokens,
                    }
                else:
                    raise ValueError(f"{query_id} 没有非空冻结 EvidencePackage")
            else:
                visible = packaged_ids
                attribution = {
                    "source": "frozen_retrieval",
                    "candidate_pool_complete": retrieved.get("retrieval", {})
                    .get("candidate_pool_metrics", {})
                    .get("complete_hit"),
                    "reranked_top5_complete": retrieved.get("retrieval", {})
                    .get("reranked_top5_metrics", {})
                    .get("complete_hit"),
                    "packaged_complete": packaging.get("metrics", {}).get("complete_hit"),
                    "prompt_tokens": packaging.get("prompt_tokens"),
                }
        else:
            visible = list(required)
            attribution = {
                "source": "deterministic_exact_lookup",
                "candidate_pool_complete": None,
                "reranked_top5_complete": None,
                "packaged_complete": True,
                "prompt_tokens": None,
            }
        if len(visible) > 5 or len(visible) != len(set(visible)):
            raise ValueError(f"{query_id} 的 visible_chunk_ids 无效")
        articles = [repository.get_by_chunk_id(chunk_id) for chunk_id in visible]
        evidence = [_evidence_record(article, index) for index, article in enumerate(articles, 1)]
        hard_negative = [chunk_id for chunk_id in visible if chunk_id not in set(required)]
        output.append(
            {
                "query_id": query_id,
                "query_type": query_type,
                "evaluation_scope": (
                    "rope_extrapolation_1024"
                    if query_id in EXTRAPOLATION_QUERY_IDS
                    else "primary_768"
                ),
                "query": query,
                "evidence": evidence,
                "visible_chunk_ids": visible,
                "required_chunk_ids": required,
                "hard_negative_chunk_ids": hard_negative,
                "retrieval_attribution": attribution,
            }
        )
        counts[query_type] += 1
    if counts != {"legal_query": EXPECTED_LEGAL_QUERY, "exact_lookup": EXPECTED_EXACT_LOOKUP}:
        raise ValueError(f"评估题型计数不闭合: {counts}")
    if len(output) != EXPECTED_RECORDS:
        raise ValueError("评估输入总数不是 170")
    return output


def publish_inputs(
    *,
    eval_set: str | Path,
    retrieval_manifest: str | Path,
    retrieval_records: str | Path,
    article_index: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    expected_retrieval_pipeline: str,
) -> dict[str, Any]:
    manifest, manifest_sha256 = _load_verified_json(retrieval_manifest, "冻结检索 manifest")
    if manifest.get("pipeline") != expected_retrieval_pipeline:
        raise ValueError("冻结检索 pipeline 与显式预期不一致")
    packaging = manifest.get("packaging")
    tokenizer = manifest.get("inputs", {}).get("tokenizer")
    if (
        not isinstance(packaging, dict)
        or packaging.get("context_limit") != 768
        or packaging.get("max_output_tokens") != 150
        or not isinstance(tokenizer, dict)
        or tokenizer.get("vocab_size") != 12000
        or not isinstance(tokenizer.get("files"), dict)
    ):
        raise ValueError("冻结检索 manifest 的长度或 Tokenizer 身份无效")
    records_path = Path(retrieval_records).resolve()
    records_identity = _identity(records_path, records=len(_load_jsonl(records_path, "冻结检索 records")))
    bound = manifest.get("outputs", {}).get("records")
    if (
        not isinstance(bound, dict)
        or bound.get("sha256") != records_identity["sha256"]
        or bound.get("records") != records_identity["records"]
    ):
        raise ValueError("冻结检索 records 与 manifest 身份不一致")
    eval_records = _load_jsonl(eval_set, "开发集")
    retrieval_values = _load_jsonl(records_path, "冻结检索 records")
    repository = ArticleRepository.from_jsonl(article_index)
    try:
        from transformers import AutoTokenizer

        tokenizer_instance = AutoTokenizer.from_pretrained(
            Path(tokenizer_path).resolve(), use_fast=True, local_files_only=True
        )
    except (OSError, TypeError, ValueError) as error:
        raise ValueError("无法加载评估输入固定 Tokenizer") from error
    tokenizer_actual = {
        "vocab_size": len(tokenizer_instance),
        "chat_template_sha256": hashlib.sha256(
            tokenizer_instance.chat_template.encode("utf-8")
        ).hexdigest(),
        "files": {
            filename: {
                "bytes": (Path(tokenizer_path).resolve() / filename).stat().st_size,
                "sha256": _sha256_file(Path(tokenizer_path).resolve() / filename),
            }
            for filename in ("tokenizer.json", "tokenizer_config.json")
        },
    }
    tokenizer_expected = {
        "vocab_size": tokenizer.get("vocab_size"),
        "chat_template_sha256": tokenizer.get("chat_template_sha256"),
        "files": {
            filename: {"bytes": item.get("bytes"), "sha256": item.get("sha256")}
            for filename, item in tokenizer.get("files", {}).items()
            if isinstance(item, dict)
        },
    }
    if tokenizer_actual != tokenizer_expected:
        raise ValueError("当前 Tokenizer 与冻结检索 manifest 身份不一致")
    counter = AnswerPromptTokenCounter(tokenizer_instance)
    extrapolation_packager = EvidencePackager(
        context_limit=1024,
        max_output_tokens=150,
        count_prompt_tokens=counter,
    )
    records = build_evaluation_records(
        eval_records,
        retrieval_values,
        repository,
        extrapolation_packager=extrapolation_packager,
    )
    for record in records:
        package = EvidencePackage(
            query=record["query"],
            evidence=tuple(
                Evidence(item["law_name"], item["article_no"], item["content"])
                for item in record["evidence"]
            ),
        )
        prompt_tokens = counter(package)
        recorded = record["retrieval_attribution"].get("prompt_tokens")
        if recorded is not None and recorded != prompt_tokens:
            raise ValueError(f"{record['query_id']} 冻结构包 prompt token 身份漂移")
        context_limit = 1024 if record["evaluation_scope"] == "rope_extrapolation_1024" else 768
        if prompt_tokens + 150 > context_limit:
            raise ValueError(
                f"{record['query_id']} 评估 EvidencePackage 超过 {context_limit}/150 预算"
            )
        record["retrieval_attribution"]["prompt_tokens"] = prompt_tokens
    root = Path(output_dir).resolve()
    cases_path = root / "cases.jsonl"
    cases_text = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        for record in records
    )
    cases_sha256 = _write_immutable(cases_path, cases_text)
    payload = {
        "schema_version": "1.0",
        "pipeline": PIPELINE,
        "inputs": {
            "eval_set": _identity(eval_set),
            "retrieval_manifest": {
                **_identity(retrieval_manifest),
                "verified_sha256": manifest_sha256,
                "pipeline": expected_retrieval_pipeline,
            },
            "retrieval_records": records_identity,
            "article_index": _identity(article_index),
            "tokenizer_path": str(Path(tokenizer_path).resolve()),
        },
        "records": {
            "total": EXPECTED_RECORDS,
            "legal_query": EXPECTED_LEGAL_QUERY,
            "exact_lookup": EXPECTED_EXACT_LOOKUP,
        },
        "evaluation_scopes": {
            "primary_768": {
                "records": PRIMARY_RECORDS,
                "legal_query": 138,
                "exact_lookup": 28,
                "used_for_selection": True,
            },
            "rope_extrapolation_1024": {
                "records": len(EXTRAPOLATION_QUERY_IDS),
                "legal_query": 2,
                "exact_lookup": 2,
                "query_ids": list(EXTRAPOLATION_QUERY_IDS),
                "used_for_selection": False,
            },
        },
        "tokenizer": tokenizer,
        "length_policy": {
            "primary_context_limit": 768,
            "extrapolation_context_limit": 1024,
            "max_output_tokens": 150,
            "inference_rope_scaling": False,
        },
        "output": {
            "cases": {
                "path": str(cases_path),
                "bytes": cases_path.stat().st_size,
                "sha256": cases_sha256,
                "records": EXPECTED_RECORDS,
            }
        },
        "model_input_fields": ["query", "evidence"],
        "hidden_from_model": [
            "query_id",
            "query_type",
            "evaluation_scope",
            "visible_chunk_ids",
            "required_chunk_ids",
            "hard_negative_chunk_ids",
            "retrieval_attribution",
        ],
        "complete": True,
    }
    manifest_path = root / "manifest.json"
    _write_immutable(
        manifest_path,
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="发布 RAG-SFT v2 的 170 条双口径回答评估输入")
    parser.add_argument("--eval-set", required=True)
    parser.add_argument("--retrieval-manifest", required=True)
    parser.add_argument("--retrieval-records", required=True)
    parser.add_argument("--article-index", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-retrieval-pipeline", required=True)
    args = parser.parse_args()
    report = publish_inputs(
        eval_set=args.eval_set,
        retrieval_manifest=args.retrieval_manifest,
        retrieval_records=args.retrieval_records,
        article_index=args.article_index,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        expected_retrieval_pipeline=args.expected_retrieval_pipeline,
    )
    print(f"RAG_SFT_V2_EVALUATION_INPUTS_OK records={report['records']['total']}")


if __name__ == "__main__":
    main()
