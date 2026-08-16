"""将已审核通过的 RAG-SFT 规则卡物化为 Oracle authoring。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

try:
    from . import prepare_rag_sft as preparation
    from . import validate_rag_sft_rule_card_reviews as review_validator
    from . import validate_rag_sft_rule_cards as card_validator
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import prepare_rag_sft as preparation
    from dataset import validate_rag_sft_rule_card_reviews as review_validator
    from dataset import validate_rag_sft_rule_cards as card_validator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_SFT_ROOT = Path(__file__).resolve().parent / "RAG-SFT"
DEFAULT_RULE_CARDS = RAG_SFT_ROOT / "rule-cards" / "rule-cards.jsonl"
DEFAULT_REVIEW_LEDGER = RAG_SFT_ROOT / "rule-cards" / "review-ledger.jsonl"
DEFAULT_ARTICLE_INDEX = PROJECT_ROOT / "rag" / "chunk" / "article_index.jsonl"
DEFAULT_EXISTING_AUTHORING = RAG_SFT_ROOT / "authoring" / "rag-sft.jsonl"
DEFAULT_EVALUATION_EXCLUSIONS = (
    RAG_SFT_ROOT / "manifests" / "evaluation-exclusions-project-rag-dev-current.json"
)
DEFAULT_AUTHORING_OUTPUT = (
    RAG_SFT_ROOT / "authoring" / "rag-sft-rule-cards-oracle-v1.jsonl"
)
DEFAULT_MAPPING_OUTPUT = (
    RAG_SFT_ROOT / "manifests" / "rag-sft-rule-cards-oracle-v1-mapping.jsonl"
)
DEFAULT_MANIFEST_OUTPUT = (
    RAG_SFT_ROOT / "manifests" / "rag-sft-rule-cards-oracle-v1.json"
)

_CANDIDATE_ID_RE = re.compile(r"rag_sft_candidate:(\d{4})$")


class RagSftRuleCardMaterializationError(RuntimeError):
    """规则卡不能安全物化为独立 Oracle authoring。"""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
                    raise RagSftRuleCardMaterializationError(
                        f"{description} JSON 无效: {path}:{line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise RagSftRuleCardMaterializationError(
                        f"{description} 记录必须是对象: {path}:{line_number}"
                    )
                records.append(record)
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftRuleCardMaterializationError(
            f"无法读取{description}: {path}"
        ) from error
    if not records:
        raise RagSftRuleCardMaterializationError(f"{description}不能为空")
    return records


def _load_existing_ids(path: Path) -> set[str]:
    records = _load_jsonl(path, "现有 canonical authoring")
    seen_ids: set[str] = set()
    for position, record in enumerate(records, start=1):
        try:
            preparation._validate_record_shape(record, position)
        except preparation.RagSftPreparationError as error:
            raise RagSftRuleCardMaterializationError(
                f"现有 canonical authoring 无效: {error}"
            ) from error
        record_id = record["id"]
        if record_id in seen_ids:
            raise RagSftRuleCardMaterializationError(
                f"现有 canonical authoring ID 重复: {record_id}"
            )
        seen_ids.add(record_id)
    return seen_ids


def _authoring_id(candidate_id: str) -> str:
    match = _CANDIDATE_ID_RE.fullmatch(candidate_id)
    if match is None:
        raise RagSftRuleCardMaterializationError(
            f"无效规则卡 candidate_id: {candidate_id}"
        )
    return f"rag_sft:{1000 + int(match.group(1)):04d}"


def _build_record(card: dict[str, object]) -> dict[str, object]:
    required = list(card["required_chunk_ids"])
    return {
        "id": _authoring_id(card["candidate_id"]),
        "query_original": card["query_original"],
        "evidence_source": "oracle",
        "visible_chunk_ids": required,
        "required_chunk_ids": required,
        "target": {"summary": card["target_summary"], "refuse": False},
        "support_spans": card["support_spans"],
        "review_status": "approved",
        "review_notes": "由已通过当前轮次分责审核的规则卡确定性物化。",
    }


def _require_new_outputs(*paths: Path) -> None:
    occupied = [
        str(item)
        for path in paths
        for item in (path, path.with_name(path.name + ".partial"))
        if item.exists()
    ]
    if occupied:
        raise RagSftRuleCardMaterializationError(
            "目标输出已存在: " + ", ".join(occupied)
        )


def _identity_from_payload(path: Path, payload: str) -> dict[str, object]:
    encoded = payload.encode("utf-8")
    return {"path": str(path), "bytes": len(encoded), "sha256": _sha256_bytes(encoded)}


def _identity_from_file(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _publish(outputs: list[tuple[Path, str]]) -> None:
    pending = [
        (path.with_name(path.name + ".partial"), path, payload)
        for path, payload in outputs
    ]
    published: list[Path] = []
    try:
        for partial, _, payload in pending:
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(payload, encoding="utf-8", newline="\n")
        for partial, final, _ in pending:
            partial.replace(final)
            published.append(final)
    except (OSError, UnicodeError) as error:
        for path in [*(item[0] for item in pending), *reversed(published)]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise RagSftRuleCardMaterializationError("无法发布规则卡物化产物") from error


def materialize_rule_cards_to_oracle_authoring(
    *,
    rule_cards_path: Path,
    review_ledger_path: Path,
    article_index_path: Path,
    existing_authoring_path: Path,
    evaluation_exclusions_path: Path,
    authoring_output: Path,
    mapping_output: Path,
    manifest_output: Path,
) -> dict[str, object]:
    """只将当前审核终态为 approved 的规则卡发布为独立 Oracle authoring。"""

    rule_cards_path = Path(rule_cards_path).resolve()
    review_ledger_path = Path(review_ledger_path).resolve()
    article_index_path = Path(article_index_path).resolve()
    existing_authoring_path = Path(existing_authoring_path).resolve()
    evaluation_exclusions_path = Path(evaluation_exclusions_path).resolve()
    authoring_output = Path(authoring_output).resolve()
    mapping_output = Path(mapping_output).resolve()
    manifest_output = Path(manifest_output).resolve()
    hash_output = manifest_output.with_suffix(".sha256")
    _require_new_outputs(authoring_output, mapping_output, manifest_output, hash_output)

    try:
        card_validator.validate_rag_sft_rule_cards(
            rule_cards_path=rule_cards_path,
            article_index_path=article_index_path,
            existing_authoring_path=existing_authoring_path,
            evaluation_exclusions_path=evaluation_exclusions_path,
            tokenizer_path=None,
            report_output=None,
        )
    except card_validator.RagSftRuleCardError as error:
        raise RagSftRuleCardMaterializationError(
            f"规则卡全局校验失败: {error}"
        ) from error

    try:
        review_report = review_validator.validate_rag_sft_rule_card_reviews(
            rule_cards_path=rule_cards_path,
            review_ledger_path=review_ledger_path,
            article_index_path=article_index_path,
            report_output=None,
        )
    except review_validator.RagSftRuleCardReviewError as error:
        raise RagSftRuleCardMaterializationError(
            f"规则卡审核账本无效: {error}"
        ) from error

    pending = [
        locator["candidate_id"]
        for locator in review_report["locators"]
        if locator["status"] in {"revision_required", "user_review_required"}
    ]
    if pending:
        raise RagSftRuleCardMaterializationError(
            "存在待修订或待用户裁决的规则卡: " + ", ".join(pending)
        )

    cards = _load_jsonl(rule_cards_path, "规则卡")
    existing_ids = _load_existing_ids(existing_authoring_path)

    status_by_candidate = {
        locator["candidate_id"]: locator["status"] for locator in review_report["locators"]
    }
    records: list[dict[str, object]] = []
    mapping: list[dict[str, object]] = []
    seen_ids = set(existing_ids)
    source_counts: Counter[str] = Counter()
    for position, card in enumerate(cards, start=1):
        candidate_id = card["candidate_id"]
        status = status_by_candidate.get(candidate_id)
        if status is None:
            raise RagSftRuleCardMaterializationError(
                f"规则卡缺少审核终态: {candidate_id}"
            )
        source_counts[status] += 1
        if status != "approved":
            continue
        record = _build_record(card)
        record_id = record["id"]
        if record_id in seen_ids:
            raise RagSftRuleCardMaterializationError(
                f"物化 ID 与已有 authoring 冲突: {record_id}"
            )
        seen_ids.add(record_id)
        try:
            preparation._validate_record_shape(record, position)
        except preparation.RagSftPreparationError as error:
            raise RagSftRuleCardMaterializationError(
                f"物化 authoring schema 无效: {candidate_id}: {error}"
            ) from error
        records.append(record)
        mapping.append(
            {
                "id": record_id,
                "candidate_id": candidate_id,
                "rule_card_sha256": card_validator.rule_card_digest(card),
                "revision_round": next(
                    locator["revision_round"]
                    for locator in review_report["locators"]
                    if locator["candidate_id"] == candidate_id
                ),
            }
        )

    if not records:
        raise RagSftRuleCardMaterializationError("没有 approved 规则卡可物化")

    authoring_payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for record in records
    )
    mapping_payload = "".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for item in mapping
    )
    manifest = {
        "pipeline": "rag_sft_rule_cards_to_oracle_authoring",
        "schema_version": "1.0",
        "inputs": {
            "rule_cards": _identity_from_file(rule_cards_path),
            "review_ledger": _identity_from_file(review_ledger_path),
            "article_index": _identity_from_file(article_index_path),
            "existing_authoring": _identity_from_file(existing_authoring_path),
            "evaluation_exclusions": _identity_from_file(evaluation_exclusions_path),
        },
        "projection": {
            "evidence_source": "oracle",
            "visible_chunk_ids": "required_chunk_ids",
            "summary": "rule_card.target_summary",
            "support_spans": "rule_card.support_spans",
            "id_strategy": "rag_sft:1000_plus_candidate_numeric_suffix",
            "excluded_rule_cards_not_materialized": source_counts["excluded"],
        },
        "records": {
            "rule_cards": len(cards),
            "by_review_status": dict(sorted(source_counts.items())),
            "materialized": len(records),
            "existing_authoring_ids_preserved": len(existing_ids),
        },
        "output": {
            "authoring": _identity_from_payload(authoring_output, authoring_payload),
            "source_mapping": _identity_from_payload(mapping_output, mapping_payload),
        },
        "readiness": {
            "semantic_review_complete": True,
            "canonical_authoring_schema_validated": True,
            "evaluation_isolation_validated": True,
            "prepared_training_candidate": False,
            "training_ready": False,
        },
        "complete": True,
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    hash_payload = f"{_sha256_bytes(manifest_payload.encode('utf-8'))}  {manifest_output.name}\n"
    _publish(
        [
            (authoring_output, authoring_payload),
            (mapping_output, mapping_payload),
            (manifest_output, manifest_payload),
            (hash_output, hash_payload),
        ]
    )
    return manifest


def main() -> None:
    """解析命令行参数并发布独立规则卡 Oracle authoring。"""

    parser = argparse.ArgumentParser(description="物化已审核通过的 RAG-SFT 规则卡")
    parser.add_argument("--rule-cards", type=Path, default=DEFAULT_RULE_CARDS)
    parser.add_argument("--review-ledger", type=Path, default=DEFAULT_REVIEW_LEDGER)
    parser.add_argument("--article-index", type=Path, default=DEFAULT_ARTICLE_INDEX)
    parser.add_argument("--existing-authoring", type=Path, default=DEFAULT_EXISTING_AUTHORING)
    parser.add_argument(
        "--evaluation-exclusions", type=Path, default=DEFAULT_EVALUATION_EXCLUSIONS
    )
    parser.add_argument("--authoring-output", type=Path, default=DEFAULT_AUTHORING_OUTPUT)
    parser.add_argument("--mapping-output", type=Path, default=DEFAULT_MAPPING_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    args = parser.parse_args()
    try:
        manifest = materialize_rule_cards_to_oracle_authoring(
            rule_cards_path=args.rule_cards,
            review_ledger_path=args.review_ledger,
            article_index_path=args.article_index,
            existing_authoring_path=args.existing_authoring,
            evaluation_exclusions_path=args.evaluation_exclusions,
            authoring_output=args.authoring_output,
            mapping_output=args.mapping_output,
            manifest_output=args.manifest_output,
        )
    except RagSftRuleCardMaterializationError as error:
        parser.error(str(error))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
