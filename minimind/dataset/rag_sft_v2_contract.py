"""RAG-SFT v2 的纯数据契约与派生规则。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


ADMISSION_DECISIONS = frozenset({"admit", "exclude", "pending"})
ADMISSION_REASONS = frozenset({"focused", "bounded_composite", "rubric_reviewed"})
HN_SOURCES = frozenset({"retrieved", "curated"})
MAX_VISIBLE_EVIDENCE = 5
FIVE_EVIDENCE_HN_SOFT_TARGET = 0.05
NON_GT_LABELS = frozenset(
    {"hard_negative", "irrelevant", "redundant_support", "uncertain"}
)


class RagSftV2ContractError(ValueError):
    """RAG-SFT v2 记录不满足已冻结的数据契约。"""


def _non_blank_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RagSftV2ContractError(f"{field} 必须是非空字符串")
    return value


def _unique_strings(
    value: object,
    field: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not minimum <= len(value):
        raise RagSftV2ContractError(f"{field} 必须是字符串数组")
    if maximum is not None and len(value) > maximum:
        raise RagSftV2ContractError(f"{field} 最多包含 {maximum} 项")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise RagSftV2ContractError(f"{field} 必须是字符串数组")
    if len(value) != len(set(value)):
        raise RagSftV2ContractError(f"{field} 不能包含重复项")
    return tuple(value)


def _require_exact_fields(record: object, fields: frozenset[str], prefix: str) -> dict[str, Any]:
    if not isinstance(record, dict) or set(record) != fields:
        raise RagSftV2ContractError(f"{prefix} 字段必须精确匹配 schema")
    return record


def validate_admission_record(record: object) -> dict[str, str]:
    """验证独立 RAG-SFT 准入账本的一条记录。

    `focused` 与 `bounded_composite` 是逐条准入类型；已经冻结的
    admission-v1 全量账本使用 `rubric_reviewed` 表示其已完成同一审核，
    并额外保留不进入最小账本语义的 `detail`。两种已发布形态均可读取。
    """

    if not isinstance(record, dict) or set(record) not in {
        frozenset({"query_id", "decision", "reason"}),
        frozenset({"query_id", "decision", "reason", "detail"}),
    }:
        raise RagSftV2ContractError("准入记录字段必须匹配已支持的 schema")
    value = record
    query_id = _non_blank_string(value["query_id"], "query_id")
    decision = _non_blank_string(value["decision"], "decision")
    reason = _non_blank_string(value["reason"], "reason")
    if decision not in ADMISSION_DECISIONS:
        raise RagSftV2ContractError("decision 无效")
    if decision == "admit" and reason not in ADMISSION_REASONS:
        raise RagSftV2ContractError(
            "admit 只能使用 focused、bounded_composite 或 rubric_reviewed"
        )
    if "detail" in value:
        _non_blank_string(value["detail"], "detail")
    return {"query_id": query_id, "decision": decision, "reason": reason}


def derive_required_chunk_ids(claims: object) -> tuple[str, ...]:
    """按 claims 的首次 support 顺序推导最小充分 GT。"""

    if not isinstance(claims, list) or not claims:
        raise RagSftV2ContractError("claims 必须是非空数组")
    result: list[str] = []
    seen: set[str] = set()
    for claim_index, claim in enumerate(claims, start=1):
        if not isinstance(claim, dict):
            raise RagSftV2ContractError(f"claims[{claim_index}] 必须是对象")
        support = claim.get("support")
        if not isinstance(support, list) or not support:
            raise RagSftV2ContractError(f"claims[{claim_index}].support 必须非空")
        for item in support:
            if not isinstance(item, dict):
                raise RagSftV2ContractError(
                    f"claims[{claim_index}].support 必须是对象数组"
                )
            chunk_id = item.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id.strip():
                raise RagSftV2ContractError(
                    f"claims[{claim_index}].support.chunk_id 无效"
                )
            if chunk_id not in seen:
                seen.add(chunk_id)
                result.append(chunk_id)
    if not 1 <= len(result) <= 3:
        raise RagSftV2ContractError("derived required GT 必须为 1 至 3 条")
    return tuple(result)


def validate_canonical_authoring(
    record: object,
    query_pool_record: object,
    article_contents: Mapping[str, str],
) -> dict[str, object]:
    """验证单份 canonical authoring 与公共 Query 池、法条正文的闭合关系。"""

    value = _require_exact_fields(
        record,
        frozenset({"query_id", "query_original", "claims", "summary"}),
        "canonical authoring",
    )
    pool = _require_exact_fields(
        query_pool_record,
        frozenset({"query_id", "query_original", "required_chunk_ids"}),
        "公共 Query 池记录",
    )
    query_id = _non_blank_string(value["query_id"], "query_id")
    query_original = _non_blank_string(value["query_original"], "query_original")
    if "\n" in query_original or "\r" in query_original:
        raise RagSftV2ContractError("query_original 必须是单行字符串")
    if query_id != _non_blank_string(pool["query_id"], "公共 Query 池 query_id"):
        raise RagSftV2ContractError("canonical authoring 与公共 Query 池 query_id 不一致")
    if query_original != _non_blank_string(
        pool["query_original"], "公共 Query 池 query_original"
    ):
        raise RagSftV2ContractError("canonical authoring 与公共 Query 池 query_original 不一致")

    summary = _non_blank_string(value["summary"], "summary")
    if "\n" in summary or "\r" in summary:
        raise RagSftV2ContractError("summary 必须是单行字符串")
    claims = value["claims"]
    if not isinstance(claims, list) or not claims:
        raise RagSftV2ContractError("claims 必须是非空数组")

    claim_texts: set[str] = set()
    for claim_index, claim in enumerate(claims, start=1):
        prefix = f"claims[{claim_index}]"
        item = _require_exact_fields(
            claim,
            frozenset({"text", "query_basis", "support"}),
            prefix,
        )
        text = _non_blank_string(item["text"], f"{prefix}.text")
        if text in claim_texts:
            raise RagSftV2ContractError("claims.text 不能重复")
        claim_texts.add(text)
        query_basis = _unique_strings(item["query_basis"], f"{prefix}.query_basis")
        if any(basis not in query_original for basis in query_basis):
            raise RagSftV2ContractError(f"{prefix}.query_basis 无法在 query_original 定位")
        support = item["support"]
        if not isinstance(support, list) or not support:
            raise RagSftV2ContractError(f"{prefix}.support 必须是非空数组")
        seen_support: set[tuple[str, tuple[str, ...]]] = set()
        for support_index, support_item in enumerate(support, start=1):
            support_prefix = f"{prefix}.support[{support_index}]"
            support_value = _require_exact_fields(
                support_item,
                frozenset({"chunk_id", "spans"}),
                support_prefix,
            )
            chunk_id = _non_blank_string(
                support_value["chunk_id"], f"{support_prefix}.chunk_id"
            )
            spans = _unique_strings(
                support_value["spans"], f"{support_prefix}.spans", minimum=1
            )
            support_key = (chunk_id, spans)
            if support_key in seen_support:
                raise RagSftV2ContractError(f"{prefix}.support 不能重复")
            seen_support.add(support_key)
            content = article_contents.get(chunk_id)
            if not isinstance(content, str):
                raise RagSftV2ContractError(f"{support_prefix}.chunk_id 不存在于法条正文")
            if any(span not in content for span in spans):
                raise RagSftV2ContractError(
                    f"{support_prefix}.spans 不是连续法条原文"
                )

    derived_required = derive_required_chunk_ids(claims)
    pool_required = _unique_strings(
        pool["required_chunk_ids"], "公共 Query 池 required_chunk_ids", minimum=1, maximum=3
    )
    if derived_required != pool_required:
        raise RagSftV2ContractError("derived required GT 与公共 Query 池不一致")
    return {
        "query_id": query_id,
        "query_original": query_original,
        "claims": claims,
        "summary": summary,
        "required_chunk_ids": derived_required,
    }


def derive_clean_visible_chunk_ids(canonical: Mapping[str, object]) -> tuple[str, ...]:
    """clean 的完整 Evidence 固定等于 canonical 的 required GT。"""

    required = canonical.get("required_chunk_ids")
    if not isinstance(required, (list, tuple)):
        raise RagSftV2ContractError("canonical required_chunk_ids 必须是字符串数组")
    return _unique_strings(
        list(required), "canonical required_chunk_ids", minimum=1, maximum=3
    )


def derive_citations(
    visible_chunk_ids: Sequence[str], required_chunk_ids: Sequence[str]
) -> tuple[str, ...]:
    """按 EvidencePackage 自然顺序映射 required GT 的临时引用编号。"""

    visible = tuple(visible_chunk_ids)
    required = tuple(required_chunk_ids)
    if not 1 <= len(visible) <= MAX_VISIBLE_EVIDENCE or len(visible) != len(set(visible)):
        raise RagSftV2ContractError(
            f"visible_chunk_ids 必须包含 1 至 {MAX_VISIBLE_EVIDENCE} 条唯一 Evidence"
        )
    if not 1 <= len(required) <= 3 or len(required) != len(set(required)):
        raise RagSftV2ContractError("required_chunk_ids 必须包含 1 至 3 条唯一 GT")
    if not set(required).issubset(visible):
        raise RagSftV2ContractError("visible_chunk_ids 缺少 required GT")
    required_set = set(required)
    return tuple(
        f"E{index}"
        for index, chunk_id in enumerate(visible, start=1)
        if chunk_id in required_set
    )


def validate_hn_variant(
    record: object, canonical: Mapping[str, object]
) -> dict[str, object]:
    """验证 HN variant 对 canonical 回答事实的引用与非 GT 裁决。"""

    value = _require_exact_fields(
        record,
        frozenset(
            {
                "variant_id",
                "query_id",
                "visible_chunk_ids",
                "source",
                "retrieval_identity",
                "non_gt_labels",
                "review_decision",
            }
        ),
        "HN variant",
    )
    variant_id = _non_blank_string(value["variant_id"], "variant_id")
    query_id = _non_blank_string(value["query_id"], "HN variant query_id")
    if query_id != canonical.get("query_id"):
        raise RagSftV2ContractError("HN variant 与 canonical query_id 不一致")
    source = _non_blank_string(value["source"], "HN variant source")
    if source not in HN_SOURCES:
        raise RagSftV2ContractError("HN variant source 无效")
    retrieval_identity = _non_blank_string(
        value["retrieval_identity"], "HN variant retrieval_identity"
    )
    review_decision = _non_blank_string(
        value["review_decision"], "HN variant review_decision"
    )
    if review_decision != "approved":
        raise RagSftV2ContractError("正式 HN variant 必须已批准")
    visible = _unique_strings(
        value["visible_chunk_ids"],
        "HN variant visible_chunk_ids",
        minimum=1,
        maximum=MAX_VISIBLE_EVIDENCE,
    )
    required = derive_clean_visible_chunk_ids(canonical)
    citations = derive_citations(visible, required)

    labels = value["non_gt_labels"]
    if not isinstance(labels, list):
        raise RagSftV2ContractError("HN variant non_gt_labels 必须是数组")
    expected_non_gt = set(visible) - set(required)
    actual_labels: dict[str, str] = {}
    for index, item in enumerate(labels, start=1):
        label_item = _require_exact_fields(
            item,
            frozenset({"chunk_id", "label"}),
            f"HN variant non_gt_labels[{index}]",
        )
        chunk_id = _non_blank_string(
            label_item["chunk_id"], f"HN variant non_gt_labels[{index}].chunk_id"
        )
        label = _non_blank_string(
            label_item["label"], f"HN variant non_gt_labels[{index}].label"
        )
        if label not in NON_GT_LABELS:
            raise RagSftV2ContractError("HN variant non_gt_labels.label 无效")
        if chunk_id in actual_labels:
            raise RagSftV2ContractError("HN variant non_gt_labels.chunk_id 不能重复")
        actual_labels[chunk_id] = label
    if set(actual_labels) != expected_non_gt:
        raise RagSftV2ContractError("HN variant non_gt_labels 必须覆盖全部且仅覆盖非 GT Evidence")
    if "hard_negative" not in set(actual_labels.values()):
        raise RagSftV2ContractError("正式 HN variant 至少包含一条 hard_negative")
    forbidden = {"redundant_support", "uncertain"}
    if forbidden.intersection(actual_labels.values()):
        raise RagSftV2ContractError("正式 HN variant 不能含冗余支持或不确定 Evidence")
    if source == "curated" and any(
        label != "hard_negative" for label in actual_labels.values()
    ):
        raise RagSftV2ContractError("curated HN 的每条非 GT Evidence 必须是 hard_negative")
    return {
        "variant_id": variant_id,
        "query_id": query_id,
        "visible_chunk_ids": visible,
        "source": source,
        "retrieval_identity": retrieval_identity,
        "non_gt_labels": tuple(actual_labels.items()),
        "review_decision": review_decision,
        "citations": citations,
    }


def validate_hn_visible_evidence_distribution(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """审计正式 HN 的五条 Evidence 分布，不将软目标作为发布阻断条件。"""

    total = len(records)
    if not total:
        return {
            "records": 0,
            "five_evidence_records": 0,
            "five_evidence_ratio": None,
            "five_evidence_soft_target": FIVE_EVIDENCE_HN_SOFT_TARGET,
            "five_evidence_ratio_within_soft_target": True,
            "five_evidence_distribution_warning": None,
        }

    five_evidence_records = 0
    for index, record in enumerate(records, start=1):
        visible = record.get("visible_chunk_ids") if isinstance(record, Mapping) else None
        if not isinstance(visible, (list, tuple)):
            raise RagSftV2ContractError(
                f"HN 分布记录[{index}].visible_chunk_ids 必须是数组"
            )
        if len(visible) == MAX_VISIBLE_EVIDENCE:
            five_evidence_records += 1
    ratio = five_evidence_records / total
    within_soft_target = ratio < FIVE_EVIDENCE_HN_SOFT_TARGET
    return {
        "records": total,
        "five_evidence_records": five_evidence_records,
        "five_evidence_ratio": ratio,
        "five_evidence_soft_target": FIVE_EVIDENCE_HN_SOFT_TARGET,
        "five_evidence_ratio_within_soft_target": within_soft_target,
        "five_evidence_distribution_warning": (
            None
            if within_soft_target
            else "五条 Evidence HN 占比未低于 5% 的软目标；应在 manifest 中说明原因。"
        ),
    }
