"""项目私有留出集的 authoring 校验与不可覆盖冻结入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

from rag.eval import eval_lib
from rag.knowledge import ArticleRepository


AUTHORING_FIELDS = {
    "id",
    "query_original",
    "query_formal",
    "query_type",
    "department",
    "answerable",
    "difficulty",
    "gt_tool_calls",
    "unanswerable_reason",
    "retrieval_gt",
    "answer_gt",
    "evidence_variants",
    "review_status",
    "review_notes",
}
RETRIEVAL_FIELDS = {"required_chunk_ids"}
ANSWER_FIELDS = {"summary", "cited_chunk_ids", "quotes", "refuse"}
QUOTE_FIELDS = {"chunk_id", "text"}
VARIANT_FIELDS = {"id", "variant_type", "ordered_chunk_ids", "answer_gt"}
QUERY_TYPES = {
    "exact_lookup",
    "semantic_search",
    "scenario_advice",
    "cross_law",
    "irrelevant",
}
DIFFICULTIES = {"easy", "medium", "hard", "boundary", "unanswerable"}
UNANSWERABLE_REASONS = {"missing_article", "irrelevant"}
VARIANT_TYPES = {"irrelevant_evidence", "partial_evidence"}
REVIEW_STATUSES = {"draft", "approved"}

TARGET_TOTAL = 100
TARGET_ANSWERABLE = 80
TARGET_UNANSWERABLE = 20
TARGET_QUERY_TYPE_COUNTS = {
    "cross_law": 20,
    "exact_lookup": 30,
    "irrelevant": 10,
    "scenario_advice": 20,
    "semantic_search": 20,
}
TARGET_UNANSWERABLE_REASON_COUNTS = {"irrelevant": 10, "missing_article": 10}
TARGET_VARIANT_COUNTS = {"irrelevant_evidence": 10, "partial_evidence": 10}


class PrivateHoldoutError(RuntimeError):
    """表示私有留出集不满足 schema、隔离或冻结约束。"""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def question_digest(value: str) -> str:
    """按 NFC、去首尾和折叠空白后的问题计算 SHA-256。"""

    normalized = " ".join(unicodedata.normalize("NFC", value).split())
    if not normalized:
        raise PrivateHoldoutError("问题不能为空")
    return _sha256_bytes(normalized.encode("utf-8"))


def _json_digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _load_jsonl(path: Path, *, label: str) -> list[dict[str, object]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise PrivateHoldoutError(
                        f"{label} JSON 无效: {path}:{line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise PrivateHoldoutError(
                        f"{label} 记录必须是对象: {path}:{line_number}"
                    )
                records.append(record)
    except (OSError, UnicodeDecodeError) as error:
        raise PrivateHoldoutError(f"无法读取 {label}: {path}") from error
    return records


def _require_string(value: object, field: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        suffix = "字符串" if allow_empty else "非空字符串"
        raise PrivateHoldoutError(f"{field} 必须是{suffix}")
    return value


def _require_string_list(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise PrivateHoldoutError(f"{field} 必须是字符串数组")
    if len(set(value)) != len(value):
        raise PrivateHoldoutError(f"{field} 不能包含重复值")
    return value


def _source_record_id(record: dict[str, object], prefix: str, position: int) -> str:
    value = record.get("id")
    if isinstance(value, str) and value.strip():
        return value
    return f"{prefix}:{position:04d}"


def _load_reserved_question_references(
    development_eval_path: Path,
    rag_authoring_path: Path,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, list[dict[str, str]]]]:
    development_references: dict[str, list[dict[str, str]]] = {}
    for position, record in enumerate(
        _load_jsonl(development_eval_path, label="项目 RAG 开发集"), start=1
    ):
        source_id = _source_record_id(record, "development", position)
        for field in ("query_original", "query_formal"):
            value = record.get(field)
            if field == "query_formal" and value is None:
                continue
            digest = question_digest(_require_string(value, field))
            development_references.setdefault(digest, []).append(
                {"source_id": source_id, "source_field": field}
            )

    rag_references: dict[str, list[dict[str, str]]] = {}
    for position, record in enumerate(
        _load_jsonl(rag_authoring_path, label="RAG SFT authoring"), start=1
    ):
        source_id = _source_record_id(record, "rag_authoring", position)
        source_field = "query_original" if "query_original" in record else "query"
        digest = question_digest(
            _require_string(record.get(source_field), source_field)
        )
        rag_references.setdefault(digest, []).append(
            {"source_id": source_id, "source_field": source_field}
        )
    return development_references, rag_references


def _load_reserved_question_digests(
    development_eval_path: Path,
    rag_authoring_path: Path,
) -> set[str]:
    development, rag = _load_reserved_question_references(
        development_eval_path, rag_authoring_path
    )
    return {*development, *rag}


def _load_reserved_gold_references(
    development_eval_path: Path,
    rag_authoring_path: Path,
    repository: ArticleRepository,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    development_gold: dict[str, list[str]] = {}
    for position, record in enumerate(
        _load_jsonl(development_eval_path, label="项目 RAG 开发集"), start=1
    ):
        source_id = _source_record_id(record, "development", position)
        articles = record.get("gt_articles", [])
        if not isinstance(articles, list):
            raise PrivateHoldoutError(
                f"项目 RAG 开发集第 {position} 条 gt_articles 必须是数组"
            )
        for article in articles:
            if not isinstance(article, dict):
                raise PrivateHoldoutError(
                    f"项目 RAG 开发集第 {position} 条 gt_articles 元素必须是对象"
                )
            law_name = _require_string(article.get("law_name"), "gt_articles.law_name")
            article_no = _require_string(
                article.get("article_no"), "gt_articles.article_no"
            )
            article = repository.lookup(law_name, article_no)
            if article is None:
                raise PrivateHoldoutError(
                    f"项目 RAG 开发集第 {position} 条 gold 无法映射到当前法条索引"
                )
            references = development_gold.setdefault(article.chunk_id, [])
            if source_id not in references:
                references.append(source_id)

    rag_gold: dict[str, list[str]] = {}
    for position, record in enumerate(
        _load_jsonl(rag_authoring_path, label="RAG SFT authoring"), start=1
    ):
        source_id = _source_record_id(record, "rag_authoring", position)
        if "required_chunk_ids" in record:
            gold_chunk_ids = _require_string_list(
                record.get("required_chunk_ids"),
                f"RAG SFT authoring 第 {position} 条 required_chunk_ids",
            )
        else:
            ordered_chunk_ids = _require_string_list(
                record.get("ordered_chunk_ids", []),
                f"RAG SFT authoring 第 {position} 条 ordered_chunk_ids",
            )
            target = record.get("target", {})
            if not isinstance(target, dict):
                raise PrivateHoldoutError(
                    f"RAG SFT authoring 第 {position} 条 target 必须是对象"
                )
            cited_chunk_ids = _require_string_list(
                target.get("cited_chunk_ids", []),
                f"RAG SFT authoring 第 {position} 条 target.cited_chunk_ids",
            )
            gold_chunk_ids = list({*ordered_chunk_ids, *cited_chunk_ids})
        for chunk_id in gold_chunk_ids:
            try:
                repository.get_by_chunk_id(chunk_id)
            except (KeyError, ValueError) as error:
                raise PrivateHoldoutError(
                    f"RAG SFT authoring 第 {position} 条 gold 不在当前法条索引"
                ) from error
            rag_gold.setdefault(chunk_id, []).append(source_id)
    return development_gold, rag_gold


def _load_reserved_gold_chunk_ids(
    development_eval_path: Path,
    rag_authoring_path: Path,
    repository: ArticleRepository,
) -> tuple[set[str], set[str]]:
    development, rag = _load_reserved_gold_references(
        development_eval_path, rag_authoring_path, repository
    )
    return set(development), set(rag)


def _validate_missing_article(
    record: dict[str, object],
    articles_by_law: dict[str, list[dict[str, object]]],
    position: int,
) -> None:
    citation = eval_lib.extract_citation(record["query_original"])
    if citation is None:
        raise PrivateHoldoutError(
            f"第 {position} 条 missing_article 原始问题必须包含可解析的法名和条号"
    )
    law_name, article_no = citation
    resolved_law_name = eval_lib.resolve_law_name(articles_by_law, law_name)
    if resolved_law_name is None or not eval_lib.check_article_missing(
        articles_by_law, resolved_law_name, article_no
    ):
        raise PrivateHoldoutError(
            f"第 {position} 条 missing_article 必须引用库内法律的不存在条号"
        )


def _validate_record(
    record: dict[str, object],
    position: int,
    repository: ArticleRepository,
    articles_by_law: dict[str, list[dict[str, object]]],
) -> None:
    prefix = f"私有留出集第 {position} 条"
    if set(record) != AUTHORING_FIELDS:
        raise PrivateHoldoutError(f"{prefix}顶层字段必须精确匹配 schema")
    _require_string(record["id"], f"{prefix} id")
    _require_string(record["query_original"], f"{prefix} query_original")
    _require_string(record["query_formal"], f"{prefix} query_formal")
    if record["query_type"] not in QUERY_TYPES:
        raise PrivateHoldoutError(f"{prefix} query_type 无效")
    _require_string(record["department"], f"{prefix} department")
    if not isinstance(record["answerable"], bool):
        raise PrivateHoldoutError(f"{prefix} answerable 必须是布尔值")
    if record["difficulty"] not in DIFFICULTIES:
        raise PrivateHoldoutError(f"{prefix} difficulty 无效")
    if not isinstance(record["gt_tool_calls"], int) or record["gt_tool_calls"] < 0:
        raise PrivateHoldoutError(f"{prefix} gt_tool_calls 必须是非负整数")
    if record["review_status"] not in REVIEW_STATUSES:
        raise PrivateHoldoutError(f"{prefix} review_status 无效")
    _require_string(record["review_notes"], f"{prefix} review_notes", allow_empty=True)

    retrieval = record["retrieval_gt"]
    if not isinstance(retrieval, dict) or set(retrieval) != RETRIEVAL_FIELDS:
        raise PrivateHoldoutError(f"{prefix} retrieval_gt 字段必须精确匹配 schema")
    required_chunk_ids = _require_string_list(
        retrieval["required_chunk_ids"], f"{prefix} required_chunk_ids"
    )
    if record["answerable"] and record["query_type"] == "exact_lookup":
        citation = eval_lib.extract_citation(record["query_original"])
        if citation is None or len(required_chunk_ids) != 1:
            raise PrivateHoldoutError(
                f"{prefix}可答 exact_lookup 的原始问题必须包含唯一正式法名和条号"
            )
        article = repository.lookup(*citation)
        if article is None or article.chunk_id != required_chunk_ids[0]:
            raise PrivateHoldoutError(f"{prefix}exact_lookup 原始引用必须匹配唯一 gold 法条")
        if record["difficulty"] != "easy":
            raise PrivateHoldoutError(f"{prefix}exact_lookup 难度必须为 easy")

    answer = record["answer_gt"]
    if not isinstance(answer, dict) or set(answer) != ANSWER_FIELDS:
        raise PrivateHoldoutError(f"{prefix} answer_gt 字段必须精确匹配 schema")
    summary = _require_string(
        answer["summary"], f"{prefix} answer_gt.summary", allow_empty=True
    )
    cited_chunk_ids = _require_string_list(
        answer["cited_chunk_ids"], f"{prefix} cited_chunk_ids"
    )
    if not isinstance(answer["refuse"], bool):
        raise PrivateHoldoutError(f"{prefix} answer_gt.refuse 必须是布尔值")
    quotes = answer["quotes"]
    if not isinstance(quotes, list):
        raise PrivateHoldoutError(f"{prefix} answer_gt.quotes 必须是数组")

    article_content = {}
    for chunk_id in required_chunk_ids:
        try:
            article_content[chunk_id] = repository.get_by_chunk_id(chunk_id).content
        except (KeyError, ValueError) as error:
            raise PrivateHoldoutError(f"{prefix}不存在 chunk_id: {chunk_id}") from error
    for quote_index, quote in enumerate(quotes):
        if not isinstance(quote, dict) or set(quote) != QUOTE_FIELDS:
            raise PrivateHoldoutError(f"{prefix} quotes[{quote_index}] 字段无效")
        chunk_id = _require_string(
            quote["chunk_id"], f"{prefix} quotes[{quote_index}].chunk_id"
        )
        text = _require_string(quote["text"], f"{prefix} quotes[{quote_index}].text")
        if chunk_id not in article_content:
            raise PrivateHoldoutError(f"{prefix}摘录不属于 required_chunk_ids")
        if text not in article_content[chunk_id]:
            raise PrivateHoldoutError(f"{prefix}摘录不是法条正文的连续子串")

    reason = record["unanswerable_reason"]
    if record["answerable"]:
        if reason is not None:
            raise PrivateHoldoutError(f"{prefix}可答题不能携带 unanswerable_reason")
        if not required_chunk_ids or not summary.strip() or answer["refuse"]:
            raise PrivateHoldoutError(f"{prefix}可答题必须包含证据和非拒答答案")
        if cited_chunk_ids != required_chunk_ids or not quotes:
            raise PrivateHoldoutError(f"{prefix}可答题必须引用全部必需证据并提供摘录")
        if record["difficulty"] == "unanswerable" or record["gt_tool_calls"] < 1:
            raise PrivateHoldoutError(f"{prefix}可答题的难度或工具调用数无效")
    else:
        if reason not in UNANSWERABLE_REASONS:
            raise PrivateHoldoutError(f"{prefix}不可答原因无效")
        if required_chunk_ids or summary or cited_chunk_ids or quotes or not answer["refuse"]:
            raise PrivateHoldoutError(f"{prefix}不可答题不得夹带答案或法条")
        if record["difficulty"] != "unanswerable" or record["gt_tool_calls"] != 0:
            raise PrivateHoldoutError(f"{prefix}不可答题的难度和工具调用数必须为零")
        if reason == "irrelevant" and record["query_type"] != "irrelevant":
            raise PrivateHoldoutError(f"{prefix}非法律问题必须使用 irrelevant")
        if reason == "missing_article":
            if record["query_type"] != "exact_lookup":
                raise PrivateHoldoutError(f"{prefix}不存在条号必须使用 exact_lookup")
            _validate_missing_article(record, articles_by_law, position)

    variants = record["evidence_variants"]
    if not isinstance(variants, list):
        raise PrivateHoldoutError(f"{prefix} evidence_variants 必须是数组")
    if variants and not record["answerable"]:
        raise PrivateHoldoutError(f"{prefix}不可答题不能派生证据不足变体")
    seen_variant_ids = set()
    required_set = set(required_chunk_ids)
    for variant_index, variant in enumerate(variants):
        if not isinstance(variant, dict) or set(variant) != VARIANT_FIELDS:
            raise PrivateHoldoutError(f"{prefix} evidence_variants[{variant_index}] 字段无效")
        variant_id = _require_string(variant["id"], f"{prefix} variant.id")
        if variant_id in seen_variant_ids:
            raise PrivateHoldoutError(f"{prefix} variant.id 重复")
        seen_variant_ids.add(variant_id)
        variant_type = variant["variant_type"]
        if variant_type not in VARIANT_TYPES:
            raise PrivateHoldoutError(f"{prefix} variant_type 无效")
        chunk_ids = _require_string_list(
            variant["ordered_chunk_ids"], f"{prefix} variant ordered_chunk_ids"
        )
        if not chunk_ids:
            raise PrivateHoldoutError(f"{prefix}证据不足变体必须使用非空证据包")
        for chunk_id in chunk_ids:
            try:
                repository.get_by_chunk_id(chunk_id)
            except (KeyError, ValueError) as error:
                raise PrivateHoldoutError(f"{prefix}变体 chunk_id 不存在: {chunk_id}") from error
        overlap = required_set.intersection(chunk_ids)
        if variant_type == "irrelevant_evidence" and overlap:
            raise PrivateHoldoutError(f"{prefix}无关证据变体不能包含必需证据")
        if variant_type == "partial_evidence" and (
            not overlap or required_set.issubset(chunk_ids)
        ):
            raise PrivateHoldoutError(f"{prefix}部分证据变体必须命中但不能覆盖全部必需证据")

        variant_answer = variant["answer_gt"]
        if not isinstance(variant_answer, dict) or set(variant_answer) != ANSWER_FIELDS:
            raise PrivateHoldoutError(f"{prefix}变体 answer_gt 字段必须精确匹配 schema")
        variant_summary = _require_string(
            variant_answer["summary"], f"{prefix}变体 answer_gt.summary", allow_empty=True
        )
        variant_cited = _require_string_list(
            variant_answer["cited_chunk_ids"], f"{prefix}变体 cited_chunk_ids"
        )
        variant_quotes = variant_answer["quotes"]
        if not isinstance(variant_answer["refuse"], bool):
            raise PrivateHoldoutError(f"{prefix}变体 answer_gt.refuse 必须是布尔值")
        if not isinstance(variant_quotes, list):
            raise PrivateHoldoutError(f"{prefix}变体 answer_gt.quotes 必须是数组")

        expected_cited = [chunk_id for chunk_id in chunk_ids if chunk_id in required_set]
        if variant_type == "irrelevant_evidence":
            if variant_summary or variant_cited or variant_quotes or not variant_answer["refuse"]:
                raise PrivateHoldoutError(f"{prefix}无关证据变体必须拒答且不得携带答案或引用")
            continue
        if (
            not variant_summary.strip()
            or variant_answer["refuse"]
            or variant_cited != expected_cited
            or not variant_quotes
        ):
            raise PrivateHoldoutError(f"{prefix}部分证据变体必须只回答并引用现有必需证据")

        quoted_chunk_ids = set()
        for quote_index, quote in enumerate(variant_quotes):
            if not isinstance(quote, dict) or set(quote) != QUOTE_FIELDS:
                raise PrivateHoldoutError(f"{prefix}变体 quotes[{quote_index}] 字段无效")
            chunk_id = _require_string(
                quote["chunk_id"], f"{prefix}变体 quotes[{quote_index}].chunk_id"
            )
            text = _require_string(
                quote["text"], f"{prefix}变体 quotes[{quote_index}].text"
            )
            if chunk_id not in variant_cited:
                raise PrivateHoldoutError(f"{prefix}变体摘录不属于 cited_chunk_ids")
            try:
                content = repository.get_by_chunk_id(chunk_id).content
            except (KeyError, ValueError) as error:
                raise PrivateHoldoutError(f"{prefix}变体引用 chunk_id 不存在: {chunk_id}") from error
            if text not in content:
                raise PrivateHoldoutError(f"{prefix}变体摘录不是法条正文的连续子串")
            quoted_chunk_ids.add(chunk_id)
        if quoted_chunk_ids != set(variant_cited):
            raise PrivateHoldoutError(f"{prefix}变体每个引用法条都必须提供摘录")


def validate_private_holdout(
    *,
    authoring_path: Path,
    article_index_path: Path,
    development_eval_path: Path,
    rag_authoring_path: Path,
) -> dict[str, object]:
    """校验私有留出 authoring，并返回不含原文的统计。"""

    paths = {
        "authoring": Path(authoring_path).resolve(),
        "article_index": Path(article_index_path).resolve(),
        "development_eval": Path(development_eval_path).resolve(),
        "rag_authoring": Path(rag_authoring_path).resolve(),
    }
    records = _load_jsonl(paths["authoring"], label="私有留出 authoring")
    if not records:
        raise PrivateHoldoutError("私有留出 authoring 不能为空")
    try:
        repository = ArticleRepository.from_jsonl(paths["article_index"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PrivateHoldoutError("无法加载法条索引") from error
    _, articles_by_law, _ = eval_lib.build_indexes(
        _load_jsonl(paths["article_index"], label="法条索引")
    )
    reserved_digests = _load_reserved_question_digests(
        paths["development_eval"], paths["rag_authoring"]
    )
    development_gold, rag_gold = _load_reserved_gold_chunk_ids(
        paths["development_eval"], paths["rag_authoring"], repository
    )

    seen_ids = set()
    seen_questions = set()
    seen_required_gold = set()
    for position, record in enumerate(records, start=1):
        _validate_record(record, position, repository, articles_by_law)
        if record["id"] in seen_ids:
            raise PrivateHoldoutError(f"私有留出集 id 重复: {record['id']}")
        seen_ids.add(record["id"])
        for field in ("query_original", "query_formal"):
            digest = question_digest(record[field])
            if digest in reserved_digests:
                raise PrivateHoldoutError(f"{record['id']} 的 {field} 命中既有开发或训练问题")
            if digest in seen_questions:
                raise PrivateHoldoutError(f"{record['id']} 的问题在私有留出集中重复")
            seen_questions.add(digest)
        for chunk_id in record["retrieval_gt"]["required_chunk_ids"]:
            if chunk_id in seen_required_gold:
                raise PrivateHoldoutError(
                    f"{record['id']} 的 gold chunk_id 在私有留出集中重复"
                )
            if chunk_id in development_gold:
                raise PrivateHoldoutError(
                    f"{record['id']} 的 required gold 命中项目 RAG 开发集 gold"
                )
            if chunk_id in rag_gold:
                raise PrivateHoldoutError(
                    f"{record['id']} 的 required gold 命中 RAG SFT gold"
                )
            seen_required_gold.add(chunk_id)

    query_counts = Counter(record["query_type"] for record in records)
    status_counts = Counter(record["review_status"] for record in records)
    reason_counts = Counter(
        record["unanswerable_reason"]
        for record in records
        if record["unanswerable_reason"] is not None
    )
    variant_counts = Counter(
        variant["variant_type"]
        for record in records
        for variant in record["evidence_variants"]
    )
    return {
        "records": len(records),
        "answerable": sum(record["answerable"] for record in records),
        "unanswerable": sum(not record["answerable"] for record in records),
        "required_gold": len(seen_required_gold),
        "by_query_type": dict(sorted(query_counts.items())),
        "by_review_status": dict(sorted(status_counts.items())),
        "by_unanswerable_reason": dict(sorted(reason_counts.items())),
        "by_variant_type": dict(sorted(variant_counts.items())),
        "paths": {name: str(path) for name, path in paths.items()},
    }


def audit_private_holdout_compatibility(
    *,
    authoring_path: Path,
    article_index_path: Path,
    development_eval_path: Path,
    rag_authoring_path: Path,
) -> dict[str, object]:
    """只读汇总私有集与开发集、RAG SFT 的全部问题和 gold 交集。"""

    paths = {
        "authoring": Path(authoring_path).resolve(),
        "article_index": Path(article_index_path).resolve(),
        "development_eval": Path(development_eval_path).resolve(),
        "rag_authoring": Path(rag_authoring_path).resolve(),
    }
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    records = _load_jsonl(paths["authoring"], label="私有留出 authoring")
    if not records:
        raise PrivateHoldoutError("私有留出 authoring 不能为空")
    try:
        repository = ArticleRepository.from_jsonl(paths["article_index"])
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise PrivateHoldoutError("无法加载法条索引") from error
    _, articles_by_law, _ = eval_lib.build_indexes(
        _load_jsonl(paths["article_index"], label="法条索引")
    )

    development_questions, rag_questions = _load_reserved_question_references(
        paths["development_eval"], paths["rag_authoring"]
    )
    development_gold, rag_gold = _load_reserved_gold_references(
        paths["development_eval"], paths["rag_authoring"], repository
    )
    overlaps: dict[str, list[dict[str, str]]] = {
        "development_questions": [],
        "rag_questions": [],
        "development_gold": [],
        "rag_gold": [],
    }
    seen_ids = set()
    seen_questions = set()
    seen_required_gold = set()
    for position, record in enumerate(records, start=1):
        _validate_record(record, position, repository, articles_by_law)
        private_id = record["id"]
        if private_id in seen_ids:
            raise PrivateHoldoutError(f"私有留出集 id 重复: {private_id}")
        seen_ids.add(private_id)

        for field in ("query_original", "query_formal"):
            digest = question_digest(record[field])
            if digest in seen_questions:
                raise PrivateHoldoutError(f"{private_id} 的问题在私有留出集中重复")
            seen_questions.add(digest)
            for reference in development_questions.get(digest, []):
                overlaps["development_questions"].append(
                    {
                        "private_id": private_id,
                        "private_field": field,
                        "source_id": reference["source_id"],
                        "source_field": reference["source_field"],
                        "question_sha256": digest,
                    }
                )
            for reference in rag_questions.get(digest, []):
                overlaps["rag_questions"].append(
                    {
                        "private_id": private_id,
                        "private_field": field,
                        "source_id": reference["source_id"],
                        "source_field": reference["source_field"],
                        "question_sha256": digest,
                    }
                )

        for chunk_id in record["retrieval_gt"]["required_chunk_ids"]:
            if chunk_id in seen_required_gold:
                raise PrivateHoldoutError(
                    f"{private_id} 的 gold chunk_id 在私有留出集中重复"
                )
            seen_required_gold.add(chunk_id)
            for source_id in development_gold.get(chunk_id, []):
                overlaps["development_gold"].append(
                    {
                        "private_id": private_id,
                        "source_id": source_id,
                        "chunk_id": chunk_id,
                    }
                )
            for source_id in rag_gold.get(chunk_id, []):
                overlaps["rag_gold"].append(
                    {
                        "private_id": private_id,
                        "source_id": source_id,
                        "chunk_id": chunk_id,
                    }
                )

    current_hashes = {name: _sha256_file(path) for name, path in paths.items()}
    changed = [name for name in paths if current_hashes[name] != hashes[name]]
    if changed:
        raise PrivateHoldoutError("兼容性审计期间输入发生变化: " + ", ".join(changed))

    for items in overlaps.values():
        items.sort(key=lambda item: tuple(item[key] for key in sorted(item)))
    overlap_counts = {name: len(items) for name, items in overlaps.items()}
    return {
        "schema_version": "1.0",
        "pipeline": "private_holdout_compatibility_audit",
        "compatible": not any(overlap_counts.values()),
        "inputs": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in paths.items()
        },
        "private": {
            "records": len(records),
            "question_digests": len(seen_questions),
            "required_gold": len(seen_required_gold),
        },
        "overlap_counts": overlap_counts,
        "overlaps": overlaps,
        "raw_text_included": False,
    }


def publish_new_authoring(
    *,
    records: list[dict[str, object]],
    authoring_output: Path,
    article_index_path: Path,
    development_eval_path: Path,
    rag_authoring_path: Path,
) -> dict[str, object]:
    """校验标准输入中的记录，并原子发布一个新的 authoring 文件。"""

    authoring_output = Path(authoring_output).resolve()
    partial_output = authoring_output.with_name(authoring_output.name + ".partial")
    occupied = [str(path) for path in (authoring_output, partial_output) if path.exists()]
    if occupied:
        raise PrivateHoldoutError("authoring 输出已存在: " + ", ".join(occupied))
    payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in records
    )
    try:
        authoring_output.parent.mkdir(parents=True, exist_ok=True)
        partial_output.write_text(payload, encoding="utf-8", newline="\n")
        report = validate_private_holdout(
            authoring_path=partial_output,
            article_index_path=article_index_path,
            development_eval_path=development_eval_path,
            rag_authoring_path=rag_authoring_path,
        )
        partial_output.replace(authoring_output)
    except (OSError, UnicodeError, PrivateHoldoutError, ValueError, TypeError, KeyError):
        try:
            partial_output.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    report["paths"]["authoring"] = str(authoring_output)
    return report


def revise_authoring(
    *,
    records: list[dict[str, object]],
    authoring_path: Path,
    article_index_path: Path,
    development_eval_path: Path,
    rag_authoring_path: Path,
) -> dict[str, object]:
    """在记录 ID 及顺序不变时，校验并原子修订现有 authoring。"""

    authoring_path = Path(authoring_path).resolve()
    partial_path = authoring_path.with_name(authoring_path.name + ".partial")
    if partial_path.exists():
        raise PrivateHoldoutError(f"authoring 临时输出已存在: {partial_path}")

    existing_records = _load_jsonl(authoring_path, label="现有私有留出 authoring")
    existing_ids = [record.get("id") for record in existing_records]
    revised_ids = [record.get("id") for record in records]
    if revised_ids != existing_ids:
        raise PrivateHoldoutError("修订不得增删、替换或重排 authoring 记录")

    payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in records
    )
    try:
        partial_path.write_text(payload, encoding="utf-8", newline="\n")
        report = validate_private_holdout(
            authoring_path=partial_path,
            article_index_path=article_index_path,
            development_eval_path=development_eval_path,
            rag_authoring_path=rag_authoring_path,
        )
        partial_path.replace(authoring_path)
    except (OSError, UnicodeError, PrivateHoldoutError, ValueError, TypeError, KeyError):
        try:
            partial_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    report["paths"]["authoring"] = str(authoring_path)
    return report


def append_authoring_batch(
    *,
    records: list[dict[str, object]],
    authoring_path: Path,
    article_index_path: Path,
    development_eval_path: Path,
    rag_authoring_path: Path,
) -> dict[str, object]:
    """在已有记录全部批准后，原子追加一批 20 条 authoring 草稿。"""

    authoring_path = Path(authoring_path).resolve()
    partial_path = authoring_path.with_name(authoring_path.name + ".partial")
    if partial_path.exists():
        raise PrivateHoldoutError(f"authoring 临时输出已存在: {partial_path}")
    if len(records) != 20:
        raise PrivateHoldoutError("每批必须恰好追加 20 条记录")
    if any(record.get("review_status") != "draft" for record in records):
        raise PrivateHoldoutError("新追加批次必须全部为 draft")

    existing_records = _load_jsonl(authoring_path, label="现有私有留出 authoring")
    existing_report = validate_private_holdout(
        authoring_path=authoring_path,
        article_index_path=article_index_path,
        development_eval_path=development_eval_path,
        rag_authoring_path=rag_authoring_path,
    )
    if existing_report["by_review_status"] != {"approved": len(existing_records)}:
        raise PrivateHoldoutError("上一批记录全部 approved 后才能追加新批次")
    if len(existing_records) + len(records) > TARGET_TOTAL:
        raise PrivateHoldoutError(f"authoring 总量不得超过 {TARGET_TOTAL} 条")

    combined_records = [*existing_records, *records]
    payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in combined_records
    )
    try:
        partial_path.write_text(payload, encoding="utf-8", newline="\n")
        report = validate_private_holdout(
            authoring_path=partial_path,
            article_index_path=article_index_path,
            development_eval_path=development_eval_path,
            rag_authoring_path=rag_authoring_path,
        )
        partial_path.replace(authoring_path)
    except (OSError, UnicodeError, PrivateHoldoutError, ValueError, TypeError, KeyError):
        try:
            partial_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    report["paths"]["authoring"] = str(authoring_path)
    return report


def _require_freeze_distribution(report: dict[str, object]) -> None:
    if report["records"] != TARGET_TOTAL:
        raise PrivateHoldoutError(f"冻结要求恰好 {TARGET_TOTAL} 条记录")
    if report["by_review_status"] != {"approved": TARGET_TOTAL}:
        raise PrivateHoldoutError("冻结要求全部记录均为 approved")
    if (
        report["answerable"] != TARGET_ANSWERABLE
        or report["unanswerable"] != TARGET_UNANSWERABLE
    ):
        raise PrivateHoldoutError(
            f"冻结要求 {TARGET_ANSWERABLE} 条可答、{TARGET_UNANSWERABLE} 条不可答"
        )
    if report["by_query_type"] != TARGET_QUERY_TYPE_COUNTS:
        raise PrivateHoldoutError("冻结时 query_type 分布不符合目标")
    if report["by_unanswerable_reason"] != TARGET_UNANSWERABLE_REASON_COUNTS:
        raise PrivateHoldoutError("冻结时不可答原因分布不符合目标")
    if report["by_variant_type"] != TARGET_VARIANT_COUNTS:
        raise PrivateHoldoutError("冻结时证据不足变体分布不符合目标")


def freeze_private_holdout(
    *,
    authoring_path: Path,
    article_index_path: Path,
    development_eval_path: Path,
    rag_authoring_path: Path,
    frozen_output: Path,
    manifest_output: Path,
) -> dict[str, object]:
    """在全部关卡通过后发布不含审阅备注的冻结集及审计 manifest。"""

    report = validate_private_holdout(
        authoring_path=authoring_path,
        article_index_path=article_index_path,
        development_eval_path=development_eval_path,
        rag_authoring_path=rag_authoring_path,
    )
    _require_freeze_distribution(report)
    authoring_path = Path(authoring_path).resolve()
    frozen_output = Path(frozen_output).resolve()
    manifest_output = Path(manifest_output).resolve()
    hash_output = manifest_output.with_suffix(".sha256")
    targets = (frozen_output, manifest_output, hash_output)
    occupied = [str(path) for path in targets if path.exists()]
    if occupied:
        raise PrivateHoldoutError("冻结输出已存在: " + ", ".join(occupied))

    authoring_records = _load_jsonl(authoring_path, label="私有留出 authoring")
    frozen_records = [
        {key: value for key, value in record.items() if key not in {"review_status", "review_notes"}}
        for record in authoring_records
    ]
    frozen_payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for record in frozen_records
    )
    question_digests = sorted(
        {
            question_digest(record[field])
            for record in authoring_records
            for field in ("query_original", "query_formal")
        }
    )
    answer_digests = sorted(_json_digest(record["answer_gt"]) for record in authoring_records)
    evidence_digests = sorted(
        _json_digest(
            {
                "required_chunk_ids": record["retrieval_gt"]["required_chunk_ids"],
                "evidence_variants": record["evidence_variants"],
            }
        )
        for record in authoring_records
    )
    frozen_identity = {
        "path": str(frozen_output),
        "bytes": len(frozen_payload.encode("utf-8")),
        "sha256": _sha256_bytes(frozen_payload.encode("utf-8")),
        "records": len(frozen_records),
    }
    manifest = {
        "schema_version": "1.0",
        "pipeline": "project_private_holdout_v1",
        "inputs": {
            "authoring": {
                "path": str(authoring_path),
                "bytes": authoring_path.stat().st_size,
                "sha256": _sha256_file(authoring_path),
            },
            "article_index": {
                "path": str(Path(article_index_path).resolve()),
                "sha256": _sha256_file(Path(article_index_path).resolve()),
            },
            "development_eval": {
                "path": str(Path(development_eval_path).resolve()),
                "sha256": _sha256_file(Path(development_eval_path).resolve()),
            },
            "rag_authoring": {
                "path": str(Path(rag_authoring_path).resolve()),
                "sha256": _sha256_file(Path(rag_authoring_path).resolve()),
            },
        },
        "records": {key: value for key, value in report.items() if key != "paths"},
        "digests": {
            "question_sha256": question_digests,
            "answer_sha256": answer_digests,
            "evidence_set_sha256": evidence_digests,
        },
        "output": {"frozen": frozen_identity},
        "raw_questions_included": False,
        "frozen": True,
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    hash_payload = (
        f"{_sha256_bytes(manifest_payload.encode('utf-8'))}  {manifest_output.name}\n"
    )
    partials = [
        (frozen_output.with_name(frozen_output.name + ".partial"), frozen_output, frozen_payload),
        (manifest_output.with_name(manifest_output.name + ".partial"), manifest_output, manifest_payload),
        (hash_output.with_name(hash_output.name + ".partial"), hash_output, hash_payload),
    ]
    published = []
    try:
        for partial, final, payload in partials:
            final.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(payload, encoding="utf-8", newline="\n")
        for partial, final, _ in partials:
            partial.replace(final)
            published.append(final)
    except OSError as error:
        for path in [*(item[0] for item in partials), *reversed(published)]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise PrivateHoldoutError("无法发布私有留出集冻结产物") from error
    return manifest


def main() -> None:
    """解析私有留出集校验、兼容性审计或冻结命令。"""

    parser = argparse.ArgumentParser(description="校验、审计或冻结项目私有留出集")
    parser.add_argument("--authoring", type=Path, required=True)
    parser.add_argument("--article-index", type=Path, required=True)
    parser.add_argument("--development-eval", type=Path, required=True)
    parser.add_argument("--rag-authoring", type=Path, required=True)
    parser.add_argument("--audit-compatibility", action="store_true")
    parser.add_argument("--freeze-output", type=Path)
    parser.add_argument("--manifest-output", type=Path)
    args = parser.parse_args()
    common = {
        "authoring_path": args.authoring,
        "article_index_path": args.article_index,
        "development_eval_path": args.development_eval,
        "rag_authoring_path": args.rag_authoring,
    }
    try:
        if args.audit_compatibility:
            if args.freeze_output is not None or args.manifest_output is not None:
                raise PrivateHoldoutError("兼容性审计不能同时指定冻结输出")
            report = audit_private_holdout_compatibility(**common)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            if not report["compatible"]:
                raise SystemExit(1)
        elif args.freeze_output is None and args.manifest_output is None:
            report = validate_private_holdout(**common)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        elif args.freeze_output is not None and args.manifest_output is not None:
            manifest = freeze_private_holdout(
                **common,
                frozen_output=args.freeze_output,
                manifest_output=args.manifest_output,
            )
            print(f"[完成] 冻结 {manifest['output']['frozen']['records']} 条私有留出记录")
        else:
            raise PrivateHoldoutError("冻结输出与 manifest 输出必须同时提供")
    except (PrivateHoldoutError, OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"[失败] {error}") from error


if __name__ == "__main__":
    main()
