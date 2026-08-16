"""校验 RAG-SFT 规则卡的分责审核 ledger。"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from rag.knowledge import ArticleRepository

try:
    from . import validate_rag_sft_rule_cards as cards
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import validate_rag_sft_rule_cards as cards


RAG_SFT_ROOT = Path(__file__).resolve().parent / "RAG-SFT"
DEFAULT_RULE_CARDS = cards.DEFAULT_RULE_CARDS
DEFAULT_ARTICLE_INDEX = cards.DEFAULT_ARTICLE_INDEX
DEFAULT_REVIEW_LEDGER = RAG_SFT_ROOT / "rule-cards" / "review-ledger.jsonl"
DEFAULT_REPORT_OUTPUT = RAG_SFT_ROOT / "reports" / "rag-sft-rule-card-review.json"

REVIEW_FIELDS = {
    "candidate_id",
    "rule_card_sha256",
    "revision_round",
    "reviewer_role",
    "decision",
    "risk_reasons",
    "evidence_spans",
    "notes",
}
REVIEWER_ROLES = {
    "legal_facts",
    "query_gt_sufficiency",
    "corpus_quality",
    "adversarial",
}
BASE_ROLES = {"legal_facts", "query_gt_sufficiency", "corpus_quality"}
EVIDENCE_ROLES = {"legal_facts", "query_gt_sufficiency"}
DECISIONS = {"pass", "revise", "exclude", "escalate"}
_SHA256_RE = re.compile(r"[0-9a-f]{64}$")


class RagSftRuleCardReviewError(RuntimeError):
    """审核 ledger 不能可靠绑定规则卡或违反审核流程。"""


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
                    raise RagSftRuleCardReviewError(
                        f"{description} JSON 无效: {path}:{line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise RagSftRuleCardReviewError(
                        f"{description} 记录必须是对象: {path}:{line_number}"
                    )
                records.append(record)
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftRuleCardReviewError(f"无法读取{description}: {path}") from error
    if not records:
        raise RagSftRuleCardReviewError(f"{description}不能为空")
    return records


def _validate_span(
    span: object,
    *,
    prefix: str,
    required: set[str] | None,
    repository: ArticleRepository,
) -> tuple[str, str]:
    if not isinstance(span, dict) or set(span) != cards.SPAN_FIELDS:
        raise RagSftRuleCardReviewError(f"{prefix} evidence_spans 字段无效")
    chunk_id = span["chunk_id"]
    text = span["text"]
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        raise RagSftRuleCardReviewError(f"{prefix} evidence_spans.chunk_id 无效")
    if not isinstance(text, str) or not text.strip():
        raise RagSftRuleCardReviewError(f"{prefix} evidence_spans.text 无效")
    if required is not None and chunk_id not in required:
        raise RagSftRuleCardReviewError(f"{prefix} evidence span 不属于 required GT")
    try:
        article = repository.get_by_chunk_id(chunk_id)
    except KeyError as error:
        raise RagSftRuleCardReviewError(f"{prefix} evidence span 法条不存在") from error
    if text not in article.content:
        raise RagSftRuleCardReviewError(f"{prefix} evidence span 不是连续法条原文")
    return chunk_id, text


def _validate_review_shape(
    review: dict[str, object],
    *,
    position: int,
    repository: ArticleRepository,
) -> None:
    prefix = f"审核 ledger 第 {position} 条"
    if set(review) != REVIEW_FIELDS:
        raise RagSftRuleCardReviewError(f"{prefix}顶层字段必须精确匹配 schema")
    if not isinstance(review["candidate_id"], str) or not review["candidate_id"].strip():
        raise RagSftRuleCardReviewError(f"{prefix} candidate_id 无效")
    rule_card_sha256 = review["rule_card_sha256"]
    if not isinstance(rule_card_sha256, str) or not _SHA256_RE.fullmatch(
        rule_card_sha256
    ):
        raise RagSftRuleCardReviewError(f"{prefix} rule_card_sha256 无效")
    revision_round = review["revision_round"]
    if type(revision_round) is not int or not 0 <= revision_round <= 2:
        raise RagSftRuleCardReviewError(f"{prefix} revision_round 必须是 0、1 或 2")
    role = review["reviewer_role"]
    if role not in REVIEWER_ROLES:
        raise RagSftRuleCardReviewError(f"{prefix} reviewer_role 无效")
    decision = review["decision"]
    if decision not in DECISIONS:
        raise RagSftRuleCardReviewError(f"{prefix} decision 无效")
    if revision_round == 2 and decision == "revise":
        raise RagSftRuleCardReviewError(f"{prefix}第二轮后不能继续要求实质修订")

    risk_reasons = review["risk_reasons"]
    if not isinstance(risk_reasons, list) or any(
        not isinstance(item, str) or not item.strip() for item in risk_reasons
    ):
        raise RagSftRuleCardReviewError(f"{prefix} risk_reasons 必须是字符串数组")
    if len(set(risk_reasons)) != len(risk_reasons):
        raise RagSftRuleCardReviewError(f"{prefix} risk_reasons 不能重复")
    if decision != "pass" and not risk_reasons:
        raise RagSftRuleCardReviewError(f"{prefix}非 pass 决定必须说明风险原因")

    notes = review["notes"]
    if not isinstance(notes, str) or not notes.strip() or "\n" in notes or "\r" in notes:
        raise RagSftRuleCardReviewError(f"{prefix} notes 必须是非空单行字符串")
    spans = review["evidence_spans"]
    if not isinstance(spans, list):
        raise RagSftRuleCardReviewError(f"{prefix} evidence_spans 必须是对象数组")
    seen = set()
    for span in spans:
        pair = _validate_span(
            span, prefix=prefix, required=None, repository=repository
        )
        if pair in seen:
            raise RagSftRuleCardReviewError(f"{prefix} evidence_spans 不能重复")
        seen.add(pair)


def _validate_current_review(
    review: dict[str, object],
    *,
    position: int,
    card: dict[str, object],
) -> None:
    prefix = f"审核 ledger 第 {position} 条"
    if review["candidate_id"] != card["candidate_id"]:
        raise RagSftRuleCardReviewError(f"{prefix} candidate_id 与规则卡不一致")
    if review["rule_card_sha256"] != cards.rule_card_digest(card):
        raise RagSftRuleCardReviewError(f"{prefix}没有绑定当前规则卡内容哈希")
    required = set(card["required_chunk_ids"])
    spans = review["evidence_spans"]
    if any(span["chunk_id"] not in required for span in spans):
        raise RagSftRuleCardReviewError(f"{prefix} evidence span 不属于 required GT")
    role = review["reviewer_role"]
    decision = review["decision"]
    if role in EVIDENCE_ROLES and decision == "pass" and not spans:
        raise RagSftRuleCardReviewError(f"{prefix}证据审核 pass 必须携带原文依据")


def validate_rag_sft_rule_card_reviews(
    *,
    rule_cards_path: Path,
    review_ledger_path: Path,
    article_index_path: Path,
    report_output: Path | None = None,
) -> dict[str, object]:
    """校验当前规则卡版本的审核角色、决定与证据绑定。"""

    rule_cards_path = Path(rule_cards_path).resolve()
    review_ledger_path = Path(review_ledger_path).resolve()
    article_index_path = Path(article_index_path).resolve()
    report_output = Path(report_output).resolve() if report_output is not None else None
    rule_cards = _load_jsonl(rule_cards_path, "规则卡")
    reviews = _load_jsonl(review_ledger_path, "审核 ledger")
    try:
        repository = ArticleRepository.from_jsonl(article_index_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RagSftRuleCardReviewError("无法加载法条索引") from error

    cards_by_id = {}
    for position, card in enumerate(rule_cards, start=1):
        if set(card) != cards.RULE_CARD_FIELDS:
            raise RagSftRuleCardReviewError(
                f"规则卡第 {position} 条顶层字段必须精确匹配 schema"
            )
        candidate_id = card.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in cards_by_id:
            raise RagSftRuleCardReviewError("规则卡 candidate_id 缺失或重复")
        if not isinstance(card.get("required_chunk_ids"), list) or not card[
            "required_chunk_ids"
        ]:
            raise RagSftRuleCardReviewError(f"规则卡 required_chunk_ids 无效: {candidate_id}")
        if not isinstance(card.get("risk_flags"), list):
            raise RagSftRuleCardReviewError(f"规则卡 risk_flags 无效: {candidate_id}")
        cards_by_id[candidate_id] = card

    grouped: dict[str, list[tuple[int, dict[str, object]]]] = defaultdict(list)
    seen_review_keys = set()
    for position, review in enumerate(reviews, start=1):
        _validate_review_shape(review, position=position, repository=repository)
        candidate_id = review["candidate_id"]
        if candidate_id not in cards_by_id:
            raise RagSftRuleCardReviewError(
                f"审核 ledger 第 {position} 条引用了未知 candidate_id"
            )
        revision_round = review["revision_round"]
        role = review["reviewer_role"]
        key = (candidate_id, revision_round, role)
        if key in seen_review_keys:
            raise RagSftRuleCardReviewError(f"审核角色记录重复: {candidate_id}/{role}")
        seen_review_keys.add(key)
        grouped[candidate_id].append((position, review))

    statuses = Counter()
    decision_counts = Counter()
    role_counts = Counter()
    locators = []
    for candidate_id, card in cards_by_id.items():
        entries = grouped.get(candidate_id)
        if not entries:
            raise RagSftRuleCardReviewError(f"规则卡缺少审核记录: {candidate_id}")
        current_hash = cards.rule_card_digest(card)
        current_entries = [
            (position, review)
            for position, review in entries
            if review["rule_card_sha256"] == current_hash
        ]
        if not current_entries:
            raise RagSftRuleCardReviewError(f"规则卡缺少当前内容审核: {candidate_id}")
        rounds_by_hash: dict[str, set[int]] = defaultdict(set)
        hashes_by_round: dict[int, set[str]] = defaultdict(set)
        for _, review in entries:
            rounds_by_hash[review["rule_card_sha256"]].add(review["revision_round"])
            hashes_by_round[review["revision_round"]].add(review["rule_card_sha256"])
        if any(len(rounds) != 1 for rounds in rounds_by_hash.values()):
            raise RagSftRuleCardReviewError(
                f"同一规则卡哈希不能跨修订轮次复用: {candidate_id}"
            )
        if any(len(hashes) != 1 for hashes in hashes_by_round.values()):
            raise RagSftRuleCardReviewError(
                f"同一修订轮次不能绑定多个规则卡哈希: {candidate_id}"
            )
        current_rounds = {review["revision_round"] for _, review in current_entries}
        if len(current_rounds) != 1:
            raise RagSftRuleCardReviewError(f"当前规则卡审核轮次不唯一: {candidate_id}")
        current_round = next(iter(current_rounds))
        all_rounds = set(hashes_by_round)
        if current_round != max(all_rounds):
            raise RagSftRuleCardReviewError(f"当前规则卡不是最新审核轮次: {candidate_id}")
        if all_rounds != set(range(current_round + 1)):
            raise RagSftRuleCardReviewError(f"规则卡审核轮次不连续: {candidate_id}")
        for position, review in current_entries:
            _validate_current_review(
                review,
                position=position,
                card=card,
            )
        required_roles = set(BASE_ROLES)
        if card["risk_flags"]:
            required_roles.add("adversarial")
        by_role = {review["reviewer_role"]: review for _, review in current_entries}
        missing_roles = required_roles - set(by_role)
        if missing_roles:
            raise RagSftRuleCardReviewError(
                f"规则卡缺少审核角色 {sorted(missing_roles)}: {candidate_id}"
            )
        decisions = [review["decision"] for _, review in current_entries]
        if "exclude" in decisions:
            status = "excluded"
        elif "escalate" in decisions:
            status = "user_review_required"
        elif "revise" in decisions:
            status = "revision_required"
        elif all(decision == "pass" for decision in decisions):
            status = "approved"
        else:
            raise RagSftRuleCardReviewError(f"规则卡审核状态无法闭合: {candidate_id}")
        statuses[status] += 1
        decision_counts.update(decisions)
        role_counts.update(by_role.keys())
        locators.append(
            {
                "candidate_id": candidate_id,
                "revision_round": current_round,
                "high_risk": bool(card["risk_flags"]),
                "required_roles": sorted(required_roles),
                "status": status,
            }
        )

    approved = statuses["approved"]
    pending = statuses["revision_required"] + statuses["user_review_required"]
    semantic_review_complete = pending == 0
    report: dict[str, object] = {
        "pipeline": "rag_sft_rule_card_review_validation",
        "inputs": {
            "rule_cards": cards._identity(rule_cards_path),
            "review_ledger": cards._identity(review_ledger_path),
            "article_index": cards._identity(article_index_path),
        },
        "records": {
            "rule_cards": len(rule_cards),
            "review_rows": len(reviews),
            "by_status": dict(sorted(statuses.items())),
            "by_decision": dict(sorted(decision_counts.items())),
            "by_reviewed_role": dict(sorted(role_counts.items())),
        },
        "locators": locators,
        "validation": {
            "current_reviews_bind_current_card_hash": True,
            "role_domains_do_not_use_majority_vote": True,
            "high_risk_requires_adversarial_review": True,
            "maximum_revision_round": 2,
            "evidence_passes_have_exact_source_spans": True,
        },
        "readiness": {
            "semantic_review_complete": semantic_review_complete,
            "user_review_required": statuses["user_review_required"] > 0,
            "revision_required": statuses["revision_required"] > 0,
            "approved_records": approved,
            "excluded_records": statuses["excluded"],
            "approved_records_ready_for_projection": (
                semantic_review_complete and approved > 0
            ),
        },
        "complete": True,
    }
    if report_output is not None:
        try:
            cards.publish_report(report_output, report)
        except cards.RagSftRuleCardError as error:
            raise RagSftRuleCardReviewError(str(error)) from error
    return report


def main() -> None:
    """解析路径并校验规则卡审核 ledger。"""

    parser = argparse.ArgumentParser(description="校验 RAG-SFT 规则卡分责审核 ledger")
    parser.add_argument("--rule-cards", type=Path, default=DEFAULT_RULE_CARDS)
    parser.add_argument("--review-ledger", type=Path, default=DEFAULT_REVIEW_LEDGER)
    parser.add_argument("--article-index", type=Path, default=DEFAULT_ARTICLE_INDEX)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT_OUTPUT)
    args = parser.parse_args()
    try:
        report = validate_rag_sft_rule_card_reviews(
            rule_cards_path=args.rule_cards,
            review_ledger_path=args.review_ledger,
            article_index_path=args.article_index,
            report_output=args.report_output,
        )
    except RagSftRuleCardReviewError as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
