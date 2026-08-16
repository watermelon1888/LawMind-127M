"""校验 RAG-SFT 扩充规则卡并生成确定性覆盖报告。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from rag.answering import (
    AnswerPromptTokenCounter,
    EvidencePackage,
    build_answer_prompt,
    parse_and_validate_answer,
)
from rag.core import Evidence
from rag.knowledge import ArticleRepository
from rag.query import QueryRoute, route_query

try:
    from .build_sft_evaluation_exclusions import question_digest
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset.build_sft_evaluation_exclusions import question_digest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_SFT_ROOT = Path(__file__).resolve().parent / "RAG-SFT"
DEFAULT_RULE_CARDS = RAG_SFT_ROOT / "rule-cards" / "rule-cards.jsonl"
DEFAULT_ARTICLE_INDEX = PROJECT_ROOT / "rag" / "chunk" / "article_index.jsonl"
DEFAULT_EXISTING_AUTHORING = RAG_SFT_ROOT / "authoring" / "rag-sft.jsonl"
DEFAULT_EVALUATION_EXCLUSIONS = (
    RAG_SFT_ROOT
    / "manifests"
    / "evaluation-exclusions-project-rag-dev-current.json"
)
DEFAULT_REPORT_OUTPUT = (
    RAG_SFT_ROOT / "reports" / "rag-sft-rule-card-validation.json"
)
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "minimind" / "model"

MAX_CONTEXT_TOKENS = 768
MAX_PROMPT_TOKENS = 608
MAX_ASSISTANT_TOKENS = 160
PREFERRED_ASSISTANT_TOKENS = 128

RULE_CARD_FIELDS = {
    "candidate_id",
    "required_chunk_ids",
    "answer_elements",
    "question_focus",
    "coverage_tags",
    "risk_flags",
    "query_original",
    "target_summary",
    "support_spans",
}
COVERAGE_FIELDS = {"domain", "priority", "query_form"}
SPAN_FIELDS = {"chunk_id", "text"}

DOMAINS = {
    "civil_life",
    "labor_social_security",
    "consumer_product_food_drug",
    "administrative_public_service_traffic",
    "criminal_public_safety",
    "procedural_relief_arbitration_legal_aid",
    "personal_information_data_network",
    "special_groups_education_health_environment",
}
PRIORITIES = {"high_frequency", "confusing_multi_condition", "long_tail"}
QUERY_FORMS = {"scenario", "direct_rule", "comparison_combination"}
RISK_FLAGS = {
    "multi_gt",
    "cross_law",
    "exception",
    "deadline",
    "amount",
    "responsibility_order",
    "near_duplicate",
    "length_edge",
    "gt_minimality_uncertain",
}

_CANDIDATE_ID_RE = re.compile(r"rag_sft_candidate:\d{4}$")
_SHA256_RE = re.compile(r"[0-9a-f]{64}$")


class RagSftRuleCardError(RuntimeError):
    """规则卡不满足确定性准入约束。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rule_card_digest(card: dict[str, object]) -> str:
    """计算不受 JSON 字段顺序影响的单条规则卡内容摘要。"""

    if not isinstance(card, dict):
        raise TypeError("card 必须是对象")
    payload = json.dumps(
        card,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _load_jsonl(path: Path, description: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RagSftRuleCardError(
                        f"{description} JSON 无效: {path}:{line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise RagSftRuleCardError(
                        f"{description} 记录必须是对象: {path}:{line_number}"
                    )
                records.append(record)
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftRuleCardError(f"无法读取{description}: {path}") from error
    if not records:
        raise RagSftRuleCardError(f"{description}不能为空")
    return records


def _non_blank_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _string_list(
    value: object,
    field: str,
    *,
    maximum: int | None = None,
) -> list[str]:
    if not isinstance(value, list) or not value:
        raise RagSftRuleCardError(f"{field} 必须是非空字符串数组")
    if any(not _non_blank_string(item) for item in value):
        raise RagSftRuleCardError(f"{field} 必须是非空字符串数组")
    items = list(value)
    if len(set(items)) != len(items):
        raise RagSftRuleCardError(f"{field} 不能包含重复值")
    if maximum is not None and len(items) > maximum:
        raise RagSftRuleCardError(f"{field} 最多包含 {maximum} 项")
    return items


def _required_gt_key(chunk_ids: list[str]) -> tuple[str, ...]:
    return tuple(sorted(chunk_ids))


def _load_exclusions(path: Path) -> tuple[set[str], dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftRuleCardError(f"无法读取评估排除清单: {path}") from error
    values = payload.get("question_sha256") if isinstance(payload, dict) else None
    if (
        not isinstance(values, list)
        or not values
        or len(values) != len(set(values))
        or any(not isinstance(item, str) or not _SHA256_RE.fullmatch(item) for item in values)
    ):
        raise RagSftRuleCardError("评估排除清单 question_sha256 无效")
    return set(values), payload


def _existing_groups(
    records: list[dict[str, object]],
) -> tuple[set[str], dict[tuple[str, ...], set[str]]]:
    queries: set[str] = set()
    groups: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for position, record in enumerate(records, start=1):
        try:
            query = record["query_original"]
            required = record["required_chunk_ids"]
        except KeyError as error:
            raise RagSftRuleCardError(
                f"现有 authoring 第 {position} 条缺少分组字段"
            ) from error
        if not _non_blank_string(query) or not isinstance(required, list):
            raise RagSftRuleCardError(
                f"现有 authoring 第 {position} 条分组字段无效"
            )
        required_items = _string_list(
            required, f"现有 authoring 第 {position} 条 required_chunk_ids", maximum=3
        )
        digest = question_digest(query)
        queries.add(digest)
        groups[_required_gt_key(required_items)].add(digest)
    return queries, groups


def _validate_card(
    card: dict[str, object],
    position: int,
    repository: ArticleRepository,
) -> tuple[
    dict[str, object],
    tuple[str, ...],
    set[str],
    EvidencePackage,
    str,
]:
    prefix = f"规则卡第 {position} 条"
    if set(card) != RULE_CARD_FIELDS:
        raise RagSftRuleCardError(f"{prefix}顶层字段必须精确匹配 schema")
    candidate_id = card["candidate_id"]
    if not isinstance(candidate_id, str) or not _CANDIDATE_ID_RE.fullmatch(candidate_id):
        raise RagSftRuleCardError(
            f"{prefix} candidate_id 必须匹配 rag_sft_candidate:NNNN"
        )

    required = _string_list(
        card["required_chunk_ids"], f"{prefix} required_chunk_ids", maximum=3
    )
    answer_elements = _string_list(card["answer_elements"], f"{prefix} answer_elements")
    for field in ("question_focus", "query_original", "target_summary"):
        value = card[field]
        if not _non_blank_string(value) or "\n" in value or "\r" in value:
            raise RagSftRuleCardError(f"{prefix} {field} 必须是非空单行字符串")

    coverage = card["coverage_tags"]
    if not isinstance(coverage, dict) or set(coverage) != COVERAGE_FIELDS:
        raise RagSftRuleCardError(f"{prefix} coverage_tags 必须精确匹配 schema")
    if coverage["domain"] not in DOMAINS:
        raise RagSftRuleCardError(f"{prefix} coverage_tags.domain 无效")
    if coverage["priority"] not in PRIORITIES:
        raise RagSftRuleCardError(f"{prefix} coverage_tags.priority 无效")
    if coverage["query_form"] not in QUERY_FORMS:
        raise RagSftRuleCardError(f"{prefix} coverage_tags.query_form 无效")

    risk_flags = card["risk_flags"]
    if not isinstance(risk_flags, list) or any(
        not isinstance(item, str) or item not in RISK_FLAGS for item in risk_flags
    ):
        raise RagSftRuleCardError(f"{prefix} risk_flags 必须是闭合枚举数组")
    if len(set(risk_flags)) != len(risk_flags):
        raise RagSftRuleCardError(f"{prefix} risk_flags 不能包含重复值")

    articles = []
    try:
        for chunk_id in required:
            articles.append(repository.get_by_chunk_id(chunk_id))
    except KeyError as error:
        raise RagSftRuleCardError(f"{prefix}引用了不存在的 chunk_id") from error
    law_names = {article.law_name for article in articles}
    if (len(required) > 1) != ("multi_gt" in risk_flags):
        raise RagSftRuleCardError(f"{prefix} multi_gt 风险标记与 required 数量不一致")
    if (len(law_names) > 1) != ("cross_law" in risk_flags):
        raise RagSftRuleCardError(f"{prefix} cross_law 风险标记与法条来源不一致")

    spans = card["support_spans"]
    if not isinstance(spans, list) or not spans:
        raise RagSftRuleCardError(f"{prefix} support_spans 必须是非空对象数组")
    article_by_id = {article.chunk_id: article for article in articles}
    seen_spans: set[tuple[str, str]] = set()
    anchored: set[str] = set()
    for index, span in enumerate(spans):
        if not isinstance(span, dict) or set(span) != SPAN_FIELDS:
            raise RagSftRuleCardError(f"{prefix} support_spans[{index}] 字段无效")
        chunk_id = span["chunk_id"]
        text = span["text"]
        if not _non_blank_string(chunk_id) or not _non_blank_string(text):
            raise RagSftRuleCardError(f"{prefix} support_spans[{index}] 内容无效")
        pair = (chunk_id, text)
        if pair in seen_spans:
            raise RagSftRuleCardError(f"{prefix} support_spans 不能重复")
        seen_spans.add(pair)
        if chunk_id not in article_by_id:
            raise RagSftRuleCardError(f"{prefix} support span 不属于 required GT")
        if text not in article_by_id[chunk_id].content:
            raise RagSftRuleCardError(f"{prefix} support span 不是连续法条原文")
        anchored.add(chunk_id)
    if anchored != set(required):
        raise RagSftRuleCardError(f"{prefix}每条 required GT 都必须至少有一个 support span")

    decision = route_query(card["query_original"])
    if decision.route is not QueryRoute.SEMANTIC_SEARCH:
        raise RagSftRuleCardError(
            f"{prefix} query_original 必须路由为 semantic_search，实际为 {decision.route.value}"
        )

    package = EvidencePackage(
        query=card["query_original"],
        evidence=tuple(
            Evidence(
                law_name=article.law_name,
                article_no=article.article_no,
                content=article.content,
            )
            for article in articles
        ),
    )
    assistant = json.dumps(
        {
            "summary": card["target_summary"],
            "citations": [f"E{index}" for index in range(1, len(articles) + 1)],
            "refuse": False,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    try:
        parse_and_validate_answer(package, assistant)
    except ValueError as error:
        raise RagSftRuleCardError(f"{prefix} target_summary 未通过回答协议: {error}") from error

    normalized = {
        "candidate_id": candidate_id,
        "query_digest": question_digest(card["query_original"]),
        "required_gt_key": _required_gt_key(required),
        "coverage": dict(coverage),
        "risk_flags": tuple(risk_flags),
        "gt_count": len(required),
        "answer_element_count": len(answer_elements),
    }
    return normalized, _required_gt_key(required), law_names, package, assistant


def _token_ids(tokenizer: object, text: str) -> list[int]:
    try:
        encoded = tokenizer(text, add_special_tokens=False)
    except Exception as error:
        raise RagSftRuleCardError("Tokenizer 无法编码规则卡投影") from error
    values = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(
        encoded, "input_ids", None
    )
    if not isinstance(values, list) or any(not isinstance(item, int) for item in values):
        raise RagSftRuleCardError("Tokenizer 没有返回一维 input_ids")
    return values


def _length_metrics(
    tokenizer: object,
    package: EvidencePackage,
    assistant: str,
) -> dict[str, int]:
    eos_token = getattr(tokenizer, "eos_token", None)
    if not isinstance(eos_token, str) or not eos_token:
        raise RagSftRuleCardError("Tokenizer 缺少 eos_token")
    try:
        prompt_tokens = AnswerPromptTokenCounter(tokenizer)(package)
        full_prompt = tokenizer.apply_chat_template(
            build_answer_prompt(package)
            + [{"role": "assistant", "content": assistant}],
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as error:
        raise RagSftRuleCardError("Tokenizer 无法渲染规则卡完整对话") from error
    if not isinstance(full_prompt, str) or not full_prompt:
        raise RagSftRuleCardError("chat template 返回了空文本")
    assistant_tokens = len(_token_ids(tokenizer, f"{assistant}{eos_token}\n"))
    full_tokens = len(_token_ids(tokenizer, full_prompt))
    return {
        "prompt_tokens": prompt_tokens,
        "assistant_json_eos_tokens": assistant_tokens,
        "full_tokens": full_tokens,
    }


def _distribution(values: list[int]) -> dict[str, int | float]:
    ordered = sorted(values)

    def percentile(percent: int) -> int:
        index = max(0, (len(ordered) * percent + 99) // 100 - 1)
        return ordered[index]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(50),
        "p90": percentile(90),
        "p95": percentile(95),
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 4),
    }


def _load_tokenizer(tokenizer_path: Path) -> object:
    try:
        from .audit_disc_law_sft import load_tokenizer
    except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
        from dataset.audit_disc_law_sft import load_tokenizer
    try:
        return load_tokenizer(tokenizer_path)
    except Exception as error:
        raise RagSftRuleCardError("无法加载本地 Tokenizer") from error


def _soft_goal_report(counts: Counter[str], total: int, goals: dict[str, float]) -> dict[str, object]:
    return {
        key: {
            "count": counts[key],
            "rate": round(counts[key] / total, 8),
            "soft_target_rate": target,
        }
        for key, target in goals.items()
    }


def publish_report(path: Path, report: dict[str, object]) -> None:
    hash_path = path.with_suffix(".sha256")
    occupied = [item for item in (path, hash_path) if item.exists()]
    if occupied:
        raise RagSftRuleCardError(
            "目标输出已存在: " + ", ".join(str(item) for item in occupied)
        )
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    hash_payload = f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()}  {path.name}\n"
    pending = [
        (path.with_name(path.name + ".partial"), path, payload),
        (hash_path.with_name(hash_path.name + ".partial"), hash_path, hash_payload),
    ]
    published: list[Path] = []
    try:
        for partial, _, content in pending:
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(content, encoding="utf-8", newline="\n")
        for partial, final, _ in pending:
            partial.replace(final)
            published.append(final)
    except (OSError, UnicodeError) as error:
        for item in [*(entry[0] for entry in pending), *reversed(published)]:
            item.unlink(missing_ok=True)
        raise RagSftRuleCardError("无法发布报告") from error


def validate_rag_sft_rule_cards(
    *,
    rule_cards_path: Path,
    article_index_path: Path,
    existing_authoring_path: Path,
    evaluation_exclusions_path: Path,
    tokenizer_path: Path | None = None,
    tokenizer: object | None = None,
    report_output: Path | None = None,
) -> dict[str, object]:
    """校验扩充规则卡；可选发布不覆盖的统计报告。"""

    rule_cards_path = Path(rule_cards_path).resolve()
    article_index_path = Path(article_index_path).resolve()
    existing_authoring_path = Path(existing_authoring_path).resolve()
    evaluation_exclusions_path = Path(evaluation_exclusions_path).resolve()
    tokenizer_path = Path(tokenizer_path).resolve() if tokenizer_path is not None else None
    report_output = Path(report_output).resolve() if report_output is not None else None

    cards = _load_jsonl(rule_cards_path, "规则卡")
    existing_records = _load_jsonl(existing_authoring_path, "现有 authoring")
    exclusions, exclusion_payload = _load_exclusions(evaluation_exclusions_path)
    try:
        repository = ArticleRepository.from_jsonl(article_index_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RagSftRuleCardError("无法加载法条索引") from error

    if tokenizer is None and tokenizer_path is not None:
        tokenizer = _load_tokenizer(tokenizer_path)

    existing_queries, gt_groups = _existing_groups(existing_records)
    normalized_cards: list[dict[str, object]] = []
    length_rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    law_counts: Counter[str] = Counter()
    for position, card in enumerate(cards, start=1):
        normalized, gt_key, law_names, package, assistant = _validate_card(
            card, position, repository
        )
        candidate_id = normalized["candidate_id"]
        query_hash = normalized["query_digest"]
        if candidate_id in seen_ids:
            raise RagSftRuleCardError(f"candidate_id 重复: {candidate_id}")
        if query_hash in seen_queries:
            raise RagSftRuleCardError(f"规则卡 query_original 精确重复: {candidate_id}")
        if query_hash in existing_queries:
            raise RagSftRuleCardError(f"规则卡与现有 authoring query 重复: {candidate_id}")
        if query_hash in exclusions:
            raise RagSftRuleCardError(f"规则卡 query 命中评估排除清单: {candidate_id}")
        seen_ids.add(candidate_id)
        seen_queries.add(query_hash)
        gt_groups[gt_key].add(query_hash)
        if len(gt_groups[gt_key]) > 2:
            raise RagSftRuleCardError(
                f"同一 required GT 集合超过两个 query 组: {candidate_id}"
            )
        law_counts.update(law_names)
        if tokenizer is not None:
            length_rows.append(
                {
                    "candidate_id": candidate_id,
                    **_length_metrics(tokenizer, package, assistant),
                }
            )
        normalized_cards.append(normalized)

    length_blockers = [
        {
            **row,
            "risks": [
                risk
                for risk, failed in (
                    ("prompt_over_608", row["prompt_tokens"] > MAX_PROMPT_TOKENS),
                    (
                        "assistant_json_eos_over_160",
                        row["assistant_json_eos_tokens"] > MAX_ASSISTANT_TOKENS,
                    ),
                    ("full_sequence_over_768", row["full_tokens"] > MAX_CONTEXT_TOKENS),
                )
                if failed
            ],
        }
        for row in length_rows
        if (
            row["prompt_tokens"] > MAX_PROMPT_TOKENS
            or row["assistant_json_eos_tokens"] > MAX_ASSISTANT_TOKENS
            or row["full_tokens"] > MAX_CONTEXT_TOKENS
        )
    ]
    if length_blockers:
        details = "; ".join(
            f"{item['candidate_id']}={','.join(item['risks'])}"
            for item in length_blockers
        )
        raise RagSftRuleCardError("规则卡未通过 608/160/768 长度门禁: " + details)

    total = len(normalized_cards)
    domain_counts = Counter(item["coverage"]["domain"] for item in normalized_cards)
    priority_counts = Counter(item["coverage"]["priority"] for item in normalized_cards)
    form_counts = Counter(item["coverage"]["query_form"] for item in normalized_cards)
    gt_counts = Counter(str(item["gt_count"]) for item in normalized_cards)
    risk_counts = Counter(
        flag for item in normalized_cards for flag in item["risk_flags"]
    )
    single_law_soft_limit = math.ceil(total * 0.15)
    warnings = [
        {
            "code": "single_law_soft_limit_exceeded",
            "law_name": law_name,
            "count": count,
            "soft_limit": single_law_soft_limit,
        }
        for law_name, count in sorted(law_counts.items())
        if count > single_law_soft_limit
    ]
    warnings.extend(
        {
            "code": "assistant_above_preferred_128",
            "candidate_id": row["candidate_id"],
            "assistant_json_eos_tokens": row["assistant_json_eos_tokens"],
        }
        for row in length_rows
        if row["assistant_json_eos_tokens"] > PREFERRED_ASSISTANT_TOKENS
    )
    report: dict[str, object] = {
        "pipeline": "rag_sft_rule_card_validation",
        "inputs": {
            "rule_cards": _identity(rule_cards_path),
            "article_index": _identity(article_index_path),
            "existing_authoring": _identity(existing_authoring_path),
            "evaluation_exclusions": _identity(evaluation_exclusions_path),
            "tokenizer": (
                {"path": str(tokenizer_path), "vocab_size": len(tokenizer)}
                if tokenizer is not None and tokenizer_path is not None
                else None
            ),
        },
        "records": {
            "total": total,
            "by_domain": dict(sorted(domain_counts.items())),
            "by_priority": dict(sorted(priority_counts.items())),
            "by_query_form": dict(sorted(form_counts.items())),
            "by_required_gt_count": dict(sorted(gt_counts.items())),
            "by_risk_flag": dict(sorted(risk_counts.items())),
            "by_law": dict(sorted(law_counts.items())),
        },
        "soft_goals": {
            "priority": _soft_goal_report(
                priority_counts,
                total,
                {
                    "high_frequency": 0.70,
                    "confusing_multi_condition": 0.20,
                    "long_tail": 0.10,
                },
            ),
            "query_form": _soft_goal_report(
                form_counts,
                total,
                {
                    "scenario": 0.50,
                    "direct_rule": 0.35,
                    "comparison_combination": 0.15,
                },
            ),
            "required_gt_count": _soft_goal_report(
                gt_counts, total, {"1": 0.60, "2": 0.35, "3": 0.05}
            ),
            "single_law_rate": 0.15,
        },
        "warnings": warnings,
        "length": (
            {
                "limits": {
                    "prompt_tokens": MAX_PROMPT_TOKENS,
                    "assistant_json_eos_tokens": MAX_ASSISTANT_TOKENS,
                    "full_tokens": MAX_CONTEXT_TOKENS,
                },
                "prompt_tokens": _distribution(
                    [row["prompt_tokens"] for row in length_rows]
                ),
                "assistant_json_eos_tokens": _distribution(
                    [row["assistant_json_eos_tokens"] for row in length_rows]
                ),
                "full_tokens": _distribution(
                    [row["full_tokens"] for row in length_rows]
                ),
            }
            if length_rows
            else None
        ),
        "validation": {
            "closed_schema": True,
            "all_queries_route_to_semantic_search": True,
            "all_required_gt_exist": True,
            "all_required_gt_have_exact_support_spans": True,
            "exact_query_duplicates": 0,
            "evaluation_exact_conflicts": 0,
            "required_gt_group_limit_respected": True,
        },
        "readiness": {
            "rule_cards_deterministically_valid": True,
            "semantic_review_complete": False,
            "token_length_audit_complete": bool(length_rows),
            "evaluation_isolation_complete": exclusion_payload.get(
                "complete_for_formal_sft"
            )
            is True,
            "canonical_authoring_ready": False,
        },
        "complete": True,
    }
    if report_output is not None:
        publish_report(report_output, report)
    return report


def main() -> None:
    """解析路径并校验 RAG-SFT 扩充规则卡。"""

    parser = argparse.ArgumentParser(description="校验 RAG-SFT 扩充规则卡")
    parser.add_argument("--rule-cards", type=Path, default=DEFAULT_RULE_CARDS)
    parser.add_argument("--article-index", type=Path, default=DEFAULT_ARTICLE_INDEX)
    parser.add_argument(
        "--existing-authoring", type=Path, default=DEFAULT_EXISTING_AUTHORING
    )
    parser.add_argument(
        "--evaluation-exclusions", type=Path, default=DEFAULT_EVALUATION_EXCLUSIONS
    )
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    args = parser.parse_args()
    try:
        report = validate_rag_sft_rule_cards(
            rule_cards_path=args.rule_cards,
            article_index_path=args.article_index,
            existing_authoring_path=args.existing_authoring,
            evaluation_exclusions_path=args.evaluation_exclusions,
            tokenizer_path=args.tokenizer_path,
            report_output=args.report_output,
        )
    except RagSftRuleCardError as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
