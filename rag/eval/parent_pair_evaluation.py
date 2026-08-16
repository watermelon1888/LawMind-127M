"""构造并校验历史 top-4 父权重成对证据评估资产。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rag.answering import (
    AnswerPromptTokenCounter,
    EvidencePackage,
    EvidencePackager,
    EvidencePackagingError,
)
from rag.core.contracts import Evidence
from rag.knowledge import ArticleRepository


RANKING_PIPELINE = "legal_rag_parent_retrieval_materialization_v1"
PAIR_PIPELINE = "legal_rag_parent_pair_evaluation_v2"
PAIR_SCHEMA_VERSION = "1.1"
CONTEXT_LIMIT = 768
MAX_OUTPUT_TOKENS = 160
MAX_PROMPT_TOKENS = CONTEXT_LIMIT - MAX_OUTPUT_TOKENS
LENGTH_SELECTION = "max_complete_ordered_prefix"
_RANKING_RECORD_FIELDS = {"query_id", "candidate_chunk_ids"}
_PAIR_RECORD_FIELDS = {
    "query_id",
    "variant",
    "query",
    "evidence",
    "visible_chunk_ids",
    "required_chunk_ids",
    "hard_negative_chunk_ids",
}
_EVIDENCE_FIELDS = {"chunk_id", "law_name", "article_no", "content"}
_RANKING_TOP_K = 20


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_path(path: Path) -> Path:
    return path.with_suffix(".sha256")


def _write_hash(path: Path, digest: str) -> None:
    hash_path = _sha_path(path)
    if hash_path.exists():
        raise FileExistsError(f"输出已存在，不能覆盖: {hash_path}")
    hash_path.write_text(f"{digest}  {path.name}\n", encoding="utf-8", newline="\n")


def _write_immutable_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    resolved = Path(path).resolve()
    if resolved.exists() or _sha_path(resolved).exists():
        raise FileExistsError(f"输出已存在，不能覆盖: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    digest = _sha256_file(resolved)
    _write_hash(resolved, digest)
    return digest


def _write_immutable_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> str:
    resolved = Path(path).resolve()
    if resolved.exists() or _sha_path(resolved).exists():
        raise FileExistsError(f"输出已存在，不能覆盖: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("x", encoding="utf-8", newline="\n") as target:
        for record in records:
            target.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                + "\n"
            )
    digest = _sha256_file(resolved)
    _write_hash(resolved, digest)
    return digest


def _load_verified_json(path: str | Path, description: str) -> tuple[dict[str, Any], str]:
    resolved = Path(path).resolve()
    hash_path = _sha_path(resolved)
    if not resolved.is_file() or not hash_path.is_file():
        raise FileNotFoundError(f"{description} 或对应 SHA-256 清单不存在: {resolved}")
    digest = _sha256_file(resolved)
    if hash_path.read_text(encoding="utf-8").splitlines() != [f"{digest}  {resolved.name}"]:
        raise ValueError(f"{description} SHA-256 校验失败")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} 不是有效 UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{description} 必须是 JSON object")
    return payload, digest


def _load_verified_jsonl(path: str | Path, description: str) -> tuple[list[dict[str, Any]], str]:
    resolved = Path(path).resolve()
    hash_path = _sha_path(resolved)
    if not resolved.is_file() or not hash_path.is_file():
        raise FileNotFoundError(f"{description} 或对应 SHA-256 清单不存在: {resolved}")
    digest = _sha256_file(resolved)
    if hash_path.read_text(encoding="utf-8").splitlines() != [f"{digest}  {resolved.name}"]:
        raise ValueError(f"{description} SHA-256 校验失败")
    records = []
    for line_number, line in enumerate(resolved.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise ValueError(f"{description} 第 {line_number} 行不能为空")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{description} 第 {line_number} 行不是有效 JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{description} 第 {line_number} 行必须是 JSON object")
        records.append(record)
    if not records:
        raise ValueError(f"{description} 不能为空")
    return records, digest


def _nonempty_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} 必须是非空字符串")
    return value


def _load_tokenizer(path: str | Path) -> Any:
    tokenizer_path = Path(path).resolve()
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Tokenizer 目录不存在: {tokenizer_path}")
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            tokenizer_path, use_fast=True, local_files_only=True
        )
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"无法加载固定 Tokenizer: {tokenizer_path}") from exc


def _tokenizer_identity(tokenizer: Any, path: str | Path) -> dict[str, Any]:
    tokenizer_path = Path(path).resolve()
    files = {}
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        file_path = tokenizer_path / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"Tokenizer 指纹文件不存在: {file_path}")
        files[filename] = {
            "bytes": file_path.stat().st_size,
            "sha256": _sha256_file(file_path),
        }
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("Tokenizer 缺少 chat template")
    return {
        "chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest(),
        "files": files,
    }


def _string_list(value: object, description: str, *, nonempty: bool) -> tuple[str, ...]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{description} 必须是{'非空' if nonempty else ''}字符串数组")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{description} 只能包含非空字符串")
    if len(value) != len(set(value)):
        raise ValueError(f"{description} 不能重复")
    return tuple(value)


def _load_model_cases(eval_set: str | Path, repository: ArticleRepository) -> tuple[dict[str, Any], ...]:
    records = []
    for line_number, line in enumerate(Path(eval_set).read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            raise ValueError(f"开发集第 {line_number} 行不能为空")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"开发集第 {line_number} 行不是有效 JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"开发集第 {line_number} 行必须是 JSON object")
        if record.get("query_type") != "legal_query" or record.get("expected_action") != "answer":
            continue
        query_id = _nonempty_string(record.get("id"), f"开发集第 {line_number} 行 id")
        query = _nonempty_string(record.get("query_original"), f"{query_id} query_original")
        raw_gt = record.get("gt_articles")
        if not isinstance(raw_gt, list) or not 1 <= len(raw_gt) <= 3:
            raise ValueError(f"{query_id} 必须有一至三条 GT")
        required = []
        for position, reference in enumerate(raw_gt, 1):
            if not isinstance(reference, dict):
                raise ValueError(f"{query_id} GT[{position}] 必须是 JSON object")
            article = repository.lookup(reference.get("law_name"), reference.get("article_no"))
            if article is None:
                raise ValueError(f"{query_id} GT[{position}] 无法解析到文章索引")
            required.append(article.chunk_id)
        if len(required) != len(set(required)):
            raise ValueError(f"{query_id} 的 GT 不能重复")
        records.append({"query_id": query_id, "query": query, "required_chunk_ids": tuple(required)})
    if not records:
        raise ValueError("开发集不含 legal_query + answer 记录")
    if len({record["query_id"] for record in records}) != len(records):
        raise ValueError("开发集模型评估 query_id 不能重复")
    return tuple(records)


def _validate_ranking_records(records: Sequence[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]]) -> dict[str, tuple[str, ...]]:
    expected_ids = {case["query_id"] for case in cases}
    rankings: dict[str, tuple[str, ...]] = {}
    for record in records:
        if set(record) != _RANKING_RECORD_FIELDS:
            raise ValueError("检索排序记录必须使用闭合 schema")
        query_id = _nonempty_string(record["query_id"], "检索排序 query_id")
        candidates = _string_list(record["candidate_chunk_ids"], f"{query_id} candidate_chunk_ids", nonempty=True)
        if len(candidates) > _RANKING_TOP_K:
            raise ValueError(f"{query_id} 的检索候选不能超过 {_RANKING_TOP_K} 条")
        if query_id in rankings:
            raise ValueError(f"检索排序包含重复 query_id: {query_id}")
        rankings[query_id] = candidates
    if set(rankings) != expected_ids:
        raise ValueError("检索排序与开发集模型评估题目不完全一致")
    return rankings


def publish_retrieval_materialization(
    *,
    cases: Sequence[Mapping[str, Any]],
    rankings: Mapping[str, Sequence[str]],
    eval_set: str | Path,
    article_index: str | Path,
    output_rankings: str | Path,
    output_manifest: str | Path,
) -> dict[str, Any]:
    """将一次真实检索的排序冻结为不可变 JSONL 和 manifest。"""

    normalized_cases = tuple(cases)
    records = [
        {
            "query_id": case.case_id,
            "candidate_chunk_ids": list(rankings[case.case_id]),
        }
        for case in normalized_cases
    ]
    normalized = _validate_ranking_records(
        records,
        [{"query_id": case.case_id} for case in normalized_cases],
    )
    ranking_path = Path(output_rankings).resolve()
    ranking_digest = _write_immutable_jsonl(ranking_path, records)
    manifest = {
        "schema_version": "1.0",
        "pipeline": RANKING_PIPELINE,
        "inputs": {
            "eval_set_sha256": _sha256_file(Path(eval_set).resolve()),
            "article_index_sha256": _sha256_file(Path(article_index).resolve()),
        },
        "retrieval": {
            "dense_top_k": 30,
            "sparse_top_k": 30,
            "rrf_k": 60,
            "rerank_candidate_pool": _RANKING_TOP_K,
        },
        "ranking_file": {
            "path": str(ranking_path),
            "sha256": ranking_digest,
            "records": len(normalized),
        },
        "complete": True,
    }
    _write_immutable_json(output_manifest, manifest)
    return manifest


def materialize_current_rankings(
    *,
    eval_set: str | Path,
    article_index: str | Path,
    artifact_dir: str | Path,
    output_rankings: str | Path,
    output_manifest: str | Path,
    device: str,
    batch_size: int,
) -> dict[str, Any]:
    """按当前冻结检索参数物化 140 条模型评估题的真实排序。"""

    from rag.eval.retrieval_parameter_report import (
        _collect_rankings,
        _load_components,
        _rerank_requirements,
        build_fusion_rankings,
        load_retrieval_cases,
    )

    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size 必须是正整数")
    repository = ArticleRepository.from_jsonl(article_index)
    cases = load_retrieval_cases(eval_set, repository)
    dense_searcher, sparse_searcher, reranker, _, _ = _load_components(
        repository=repository,
        artifact_dir=artifact_dir,
        device=device,
        batch_size=batch_size,
    )
    dense_raw, _ = _collect_rankings(dense_searcher, cases, top_k=30, label="dense")
    sparse_raw, _ = _collect_rankings(sparse_searcher, cases, top_k=30, label="BM25")
    fused = build_fusion_rankings(cases, dense_raw, sparse_raw)
    reranked, _, _ = _rerank_requirements(
        cases,
        {"parent-pair": (fused, _RANKING_TOP_K)},
        repository,
        reranker,
    )
    return publish_retrieval_materialization(
        cases=cases,
        rankings=reranked["parent-pair"],
        eval_set=eval_set,
        article_index=article_index,
        output_rankings=output_rankings,
        output_manifest=output_manifest,
    )


def _load_ranking_manifest(path: str | Path, cases: Sequence[Mapping[str, Any]], eval_set: str | Path, article_index: str | Path) -> dict[str, tuple[str, ...]]:
    manifest, _ = _load_verified_json(path, "检索物化 manifest")
    expected = {"schema_version", "pipeline", "inputs", "retrieval", "ranking_file", "complete"}
    if set(manifest) != expected or manifest["schema_version"] != "1.0" or manifest["pipeline"] != RANKING_PIPELINE or manifest["complete"] is not True:
        raise ValueError("检索物化 manifest 不符合冻结契约")
    inputs = manifest["inputs"]
    if not isinstance(inputs, dict) or inputs != {
        "eval_set_sha256": _sha256_file(Path(eval_set).resolve()),
        "article_index_sha256": _sha256_file(Path(article_index).resolve()),
    }:
        raise ValueError("检索物化 manifest 的输入身份不匹配")
    retrieval = manifest["retrieval"]
    if retrieval != {"dense_top_k": 30, "sparse_top_k": 30, "rrf_k": 60, "rerank_candidate_pool": _RANKING_TOP_K}:
        raise ValueError("检索物化 manifest 的检索契约不匹配")
    ranking_file = manifest["ranking_file"]
    if not isinstance(ranking_file, dict) or set(ranking_file) != {"path", "sha256", "records"}:
        raise ValueError("检索物化 manifest 缺少 ranking_file 身份")
    records, digest = _load_verified_jsonl(ranking_file["path"], "检索排序 JSONL")
    if ranking_file["sha256"] != digest or ranking_file["records"] != len(records):
        raise ValueError("检索物化 manifest 与排序文件不一致")
    return _validate_ranking_records(records, cases)


def _evidence_record(repository: ArticleRepository, chunk_id: str) -> dict[str, str]:
    article = repository.get_by_chunk_id(chunk_id)
    return {"chunk_id": article.chunk_id, "law_name": article.law_name, "article_no": article.article_no, "content": article.content}


def _build_pair_record(
    *,
    case: Mapping[str, Any],
    variant: str,
    visible_chunk_ids: Sequence[str],
    hard_negative_chunk_ids: Sequence[str],
    repository: ArticleRepository,
) -> dict[str, Any]:
    evidence = [_evidence_record(repository, chunk_id) for chunk_id in visible_chunk_ids]
    EvidencePackage(
        query=case["query"],
        evidence=tuple(
            Evidence(law_name=item["law_name"], article_no=item["article_no"], content=item["content"])
            for item in evidence
        ),
    )
    return {
        "query_id": case["query_id"],
        "variant": variant,
        "query": case["query"],
        "evidence": evidence,
        "visible_chunk_ids": list(visible_chunk_ids),
        "required_chunk_ids": list(case["required_chunk_ids"]),
        "hard_negative_chunk_ids": list(hard_negative_chunk_ids),
    }


def _fit_visible_prefix(
    *,
    query: str,
    visible_chunk_ids: Sequence[str],
    repository: ArticleRepository,
    packager: EvidencePackager,
) -> tuple[str, ...]:
    articles = tuple(
        repository.get_by_chunk_id(chunk_id) for chunk_id in visible_chunk_ids
    )
    package, _ = packager.build(query, articles)
    return tuple(visible_chunk_ids[: len(package.evidence)])


def _validate_pair_records(
    records: Sequence[Mapping[str, Any]],
    repository: ArticleRepository,
    count_prompt_tokens: AnswerPromptTokenCounter,
) -> tuple[int, int, int]:
    grouped: dict[str, set[str]] = {}
    sufficient = 0
    insufficient = 0
    for record in records:
        if set(record) != _PAIR_RECORD_FIELDS:
            raise ValueError("成对评估记录必须使用闭合 schema")
        query_id = _nonempty_string(record["query_id"], "query_id")
        variant = record["variant"]
        if variant not in {"sufficient", "insufficient"}:
            raise ValueError("variant 必须是 sufficient 或 insufficient")
        query = _nonempty_string(record["query"], f"{query_id} query")
        visible = _string_list(record["visible_chunk_ids"], f"{query_id} visible_chunk_ids", nonempty=True)
        required = _string_list(record["required_chunk_ids"], f"{query_id} required_chunk_ids", nonempty=True)
        hard_negative = _string_list(record["hard_negative_chunk_ids"], f"{query_id} hard_negative_chunk_ids", nonempty=False)
        evidence = record["evidence"]
        if not isinstance(evidence, list) or len(evidence) != len(visible) or len(evidence) > 5:
            raise ValueError(f"{query_id} evidence 必须是一至五条且与 visible 对齐")
        seen_evidence = []
        evidence_values = []
        for item in evidence:
            if not isinstance(item, dict) or set(item) != _EVIDENCE_FIELDS:
                raise ValueError(f"{query_id} evidence 必须使用闭合 schema")
            chunk_id = _nonempty_string(item["chunk_id"], f"{query_id} evidence chunk_id")
            article = repository.get_by_chunk_id(chunk_id)
            if item != _evidence_record(repository, article.chunk_id):
                raise ValueError(f"{query_id} evidence 与文章索引不一致")
            seen_evidence.append(chunk_id)
            evidence_values.append(Evidence(law_name=article.law_name, article_no=article.article_no, content=article.content))
        if tuple(seen_evidence) != visible:
            raise ValueError(f"{query_id} evidence 顺序必须与 visible_chunk_ids 一致")
        package = EvidencePackage(query=query, evidence=tuple(evidence_values))
        if count_prompt_tokens(package) > MAX_PROMPT_TOKENS:
            raise ValueError(f"{query_id} 证据包超过固定 768/160 token 预算")
        if not set(hard_negative).issubset(visible) or set(hard_negative) & set(required):
            raise ValueError(f"{query_id} hard_negative_chunk_ids 必须可见且非 GT")
        required_visible = set(required).issubset(visible)
        if variant == "sufficient":
            if not required_visible:
                raise ValueError(f"{query_id} 充分包必须覆盖全部 GT")
            sufficient += 1
        else:
            if required_visible:
                raise ValueError(f"{query_id} 不充分包必须缺少至少一条 GT")
            if not hard_negative:
                raise ValueError(f"{query_id} 不充分包必须保留真实检索 hard negative")
            insufficient += 1
        variants = grouped.setdefault(query_id, set())
        if variant in variants:
            raise ValueError(f"{query_id} 存在重复 variant")
        variants.add(variant)
    paired = sum(variants == {"sufficient", "insufficient"} for variants in grouped.values())
    return sufficient, insufficient, paired


def publish_pair_evaluation(
    *,
    eval_set: str | Path,
    article_index: str | Path,
    ranking_manifest: str | Path,
    evaluation_exclusions: str | Path,
    output_records: str | Path,
    output_manifest: str | Path,
    tokenizer: Any,
    tokenizer_path: str | Path,
) -> dict[str, Any]:
    """将真实排序投影为充分/不充分证据包，并发布父权重评估 manifest。"""

    repository = ArticleRepository.from_jsonl(article_index)
    cases = _load_model_cases(eval_set, repository)
    rankings = _load_ranking_manifest(ranking_manifest, cases, eval_set, article_index)
    _, exclusions_digest = _load_verified_json(evaluation_exclusions, "评估排除 manifest")
    tokenizer_identity = _tokenizer_identity(tokenizer, tokenizer_path)
    counter = AnswerPromptTokenCounter(tokenizer)
    packager = EvidencePackager(
        context_limit=CONTEXT_LIMIT,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        count_prompt_tokens=counter,
    )
    records: list[dict[str, Any]] = []
    excluded_query_ids = []
    for case in cases:
        required = case["required_chunk_ids"]
        retrieved_non_gt = tuple(chunk_id for chunk_id in rankings[case["query_id"]] if chunk_id not in required)
        if not retrieved_non_gt:
            raise ValueError(f"{case['query_id']} 缺少真实检索 hard negative")
        sufficient_candidates = tuple(required) + retrieved_non_gt[: 4 - len(required)]
        insufficient_candidates = retrieved_non_gt[:4]
        try:
            sufficient_visible = _fit_visible_prefix(
                query=case["query"], visible_chunk_ids=sufficient_candidates,
                repository=repository, packager=packager,
            )
            insufficient_visible = _fit_visible_prefix(
                query=case["query"], visible_chunk_ids=insufficient_candidates,
                repository=repository, packager=packager,
            )
        except EvidencePackagingError:
            excluded_query_ids.append(case["query_id"])
            continue
        if len(sufficient_visible) < len(required):
            excluded_query_ids.append(case["query_id"])
            continue
        records.append(
            _build_pair_record(
                case=case,
                variant="sufficient",
                visible_chunk_ids=sufficient_visible,
                hard_negative_chunk_ids=sufficient_visible[len(required) :],
                repository=repository,
            )
        )
        records.append(
            _build_pair_record(
                case=case,
                variant="insufficient",
                visible_chunk_ids=insufficient_visible,
                hard_negative_chunk_ids=insufficient_visible,
                repository=repository,
            )
        )
    sufficient, insufficient, paired = _validate_pair_records(
        records, repository, counter
    )
    record_path = Path(output_records).resolve()
    record_digest = _write_immutable_jsonl(record_path, records)
    full_system_cases = sum(1 for line in Path(eval_set).read_text(encoding="utf-8").splitlines() if line)
    manifest = {
        "schema_version": PAIR_SCHEMA_VERSION,
        "pipeline": PAIR_PIPELINE,
        "evaluation_exclusions_sha256": exclusions_digest,
        "pair_records": {
            "path": record_path.name,
            "bytes": record_path.stat().st_size,
            "sha256": record_digest,
            "records": len(records),
        },
        "records": {
            "full_system_cases": full_system_cases,
            "model_cases": len(cases),
            "eligible_model_cases": len(cases) - len(excluded_query_ids),
            "excluded_overlength_cases": len(excluded_query_ids),
            "sufficient_cases": sufficient,
            "insufficient_cases": insufficient,
            "paired_queries": paired,
        },
        "length_policy": {
            "fixed_max_seq_len": CONTEXT_LIMIT,
            "max_new_tokens": MAX_OUTPUT_TOKENS,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "selection": LENGTH_SELECTION,
            "tokenizer": tokenizer_identity,
            "excluded_query_ids": excluded_query_ids,
        },
        "complete": True,
    }
    _write_immutable_json(output_manifest, manifest)
    return manifest


def verify_pair_evaluation(
    manifest_path: str | Path,
    record_path: str | Path,
    article_index: str | Path,
    *,
    tokenizer: Any,
    tokenizer_path: str | Path,
) -> dict[str, Any]:
    """验证已发布 manifest 与隔离成对证据记录的一致性。"""

    manifest, _ = _load_verified_json(manifest_path, "RAG 成对评估 manifest")
    if set(manifest) != {"schema_version", "pipeline", "evaluation_exclusions_sha256", "pair_records", "records", "length_policy", "complete"} or manifest["schema_version"] != PAIR_SCHEMA_VERSION or manifest["pipeline"] != PAIR_PIPELINE or manifest["complete"] is not True:
        raise ValueError("RAG 成对评估 manifest 不符合冻结契约")
    if not isinstance(manifest["evaluation_exclusions_sha256"], str) or len(manifest["evaluation_exclusions_sha256"]) != 64:
        raise ValueError("RAG 成对评估 manifest 缺少排除清单哈希")
    records, record_digest = _load_verified_jsonl(record_path, "RAG 成对评估记录")
    resolved_records = Path(record_path).resolve()
    pair_records = manifest["pair_records"]
    if not isinstance(pair_records, dict) or pair_records != {
        "path": resolved_records.name,
        "bytes": resolved_records.stat().st_size,
        "sha256": record_digest,
        "records": len(records),
    }:
        raise ValueError("RAG 成对评估 manifest 未绑定当前记录文件")
    repository = ArticleRepository.from_jsonl(article_index)
    length_policy = manifest["length_policy"]
    expected_policy = {
        "fixed_max_seq_len",
        "max_new_tokens",
        "max_prompt_tokens",
        "selection",
        "tokenizer",
        "excluded_query_ids",
    }
    if not isinstance(length_policy, dict) or set(length_policy) != expected_policy:
        raise ValueError("RAG 成对评估 manifest 缺少长度策略")
    if (
        length_policy["fixed_max_seq_len"] != CONTEXT_LIMIT
        or length_policy["max_new_tokens"] != MAX_OUTPUT_TOKENS
        or length_policy["max_prompt_tokens"] != MAX_PROMPT_TOKENS
        or length_policy["selection"] != LENGTH_SELECTION
        or length_policy["tokenizer"] != _tokenizer_identity(tokenizer, tokenizer_path)
    ):
        raise ValueError("RAG 成对评估长度策略或 Tokenizer 身份无效")
    excluded = _string_list(
        length_policy["excluded_query_ids"], "长度排除 query_id", nonempty=False
    )
    counter = AnswerPromptTokenCounter(tokenizer)
    sufficient, insufficient, paired = _validate_pair_records(
        records, repository, counter
    )
    expected = {"full_system_cases", "model_cases", "eligible_model_cases", "excluded_overlength_cases", "sufficient_cases", "insufficient_cases", "paired_queries"}
    if not isinstance(manifest["records"], dict) or set(manifest["records"]) != expected:
        raise ValueError("RAG 成对评估 manifest records 不符合冻结契约")
    counts = manifest["records"]
    if counts["sufficient_cases"] != sufficient or counts["insufficient_cases"] != insufficient or counts["paired_queries"] != paired:
        raise ValueError("RAG 成对评估 manifest 计数与记录不一致")
    record_query_ids = {record["query_id"] for record in records}
    if record_query_ids & set(excluded):
        raise ValueError("长度排除问题不能出现在成对评估记录中")
    if (
        counts["eligible_model_cases"] != len(record_query_ids)
        or counts["excluded_overlength_cases"] != len(excluded)
        or counts["model_cases"] != len(record_query_ids) + len(excluded)
        or paired != len(record_query_ids)
    ):
        raise ValueError("RAG 成对评估长度排除计数无效")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="冻结父权重比较的 RAG 成对证据评估资产")
    commands = parser.add_subparsers(dest="command", required=True)
    materialize = commands.add_parser("materialize", help="物化当前检索排序")
    materialize.add_argument("--eval-set", required=True)
    materialize.add_argument("--article-index", required=True)
    materialize.add_argument("--artifact-dir", required=True)
    materialize.add_argument("--output-rankings", required=True)
    materialize.add_argument("--output-manifest", required=True)
    materialize.add_argument("--device", default="auto")
    materialize.add_argument("--batch-size", type=int, default=8)
    publish = commands.add_parser("publish", help="发布成对 EvidencePackage 评估资产")
    publish.add_argument("--eval-set", required=True)
    publish.add_argument("--article-index", required=True)
    publish.add_argument("--ranking-manifest", required=True)
    publish.add_argument("--evaluation-exclusions", required=True)
    publish.add_argument("--tokenizer-path", required=True)
    publish.add_argument("--output-records", required=True)
    publish.add_argument("--output-manifest", required=True)
    verify = commands.add_parser("verify", help="校验已发布的成对评估资产")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--records", required=True)
    verify.add_argument("--article-index", required=True)
    verify.add_argument("--tokenizer-path", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        if args.command == "materialize":
            manifest = materialize_current_rankings(
                eval_set=args.eval_set, article_index=args.article_index, artifact_dir=args.artifact_dir,
                output_rankings=args.output_rankings, output_manifest=args.output_manifest,
                device=args.device, batch_size=args.batch_size,
            )
            print(f"RAG_PARENT_RANKINGS_OK records={manifest['ranking_file']['records']}")
        elif args.command == "publish":
            tokenizer = _load_tokenizer(args.tokenizer_path)
            manifest = publish_pair_evaluation(
                eval_set=args.eval_set, article_index=args.article_index,
                ranking_manifest=args.ranking_manifest, evaluation_exclusions=args.evaluation_exclusions,
                output_records=args.output_records, output_manifest=args.output_manifest,
                tokenizer=tokenizer, tokenizer_path=args.tokenizer_path,
            )
            print(f"RAG_PARENT_PAIR_EVALUATION_OK pairs={manifest['records']['paired_queries']}")
        else:
            tokenizer = _load_tokenizer(args.tokenizer_path)
            manifest = verify_pair_evaluation(
                args.manifest, args.records, args.article_index,
                tokenizer=tokenizer, tokenizer_path=args.tokenizer_path,
            )
            print(
                "RAG_PARENT_PAIR_EVALUATION_VERIFY_OK "
                f"eligible={manifest['records']['eligible_model_cases']} "
                f"excluded={manifest['records']['excluded_overlength_cases']}"
            )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise SystemExit(f"[失败] {error}") from error


if __name__ == "__main__":
    main()
