"""把旧 RAG-SFT v1 人工事实迁移为当前 canonical authoring 草稿。"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from rag.knowledge import ArticleRepository


RAG_SFT_ROOT = Path(__file__).resolve().parent / "RAG-SFT"
DEFAULT_SOURCE = RAG_SFT_ROOT / "authoring" / "rag-sft-v1.jsonl"
DEFAULT_ARTICLE_INDEX = (
    Path(__file__).resolve().parents[2] / "rag" / "chunk" / "article_index.jsonl"
)
DEFAULT_OUTPUT = RAG_SFT_ROOT / "authoring" / "rag-sft.jsonl"

AUTHORING_FIELDS = {
    "id",
    "query_original",
    "evidence_source",
    "visible_chunk_ids",
    "required_chunk_ids",
    "target",
    "support_spans",
    "review_status",
    "review_notes",
}
TARGET_FIELDS = {"summary", "refuse"}
SUPPORT_SPAN_FIELDS = {"chunk_id", "text"}
EVIDENCE_SOURCES = {"oracle", "curated_insufficient", "retrieved"}
REVIEW_STATUSES = {"draft", "approved"}
_ID_PATTERN = re.compile(r"rag_sft:\d{4}")

_OLD_FIELDS = {
    "id",
    "sample_type",
    "query",
    "ordered_chunk_ids",
    "target",
    "review_status",
    "review_notes",
}
_OLD_TARGET_FIELDS = {"summary", "cited_chunk_ids", "quotes", "refuse"}
_OLD_QUOTE_FIELDS = {"chunk_id", "text"}
_ANSWER_TYPES = {"semantic_single_answer", "semantic_multi_answer"}
_REFUSAL_TYPE = "evidence_insufficient_refusal"
_EXACT_TYPE = "exact_answer"

# 这些 GT 只证明旧人工候选为何不足，不会进入拒答样本的模型输入。
CURATED_INSUFFICIENT_GT = {
    "rag_sft_v1:0007": (
        ("中华人民共和国刑法#264",),
        "民法典第7条仅规定诚信原则，不能支持盗窃罪刑罚幅度；完整回答需要刑法第264条。",
    ),
    "rag_sft_v1:0008": (
        ("中华人民共和国劳动合同法#10", "中华人民共和国劳动合同法#82"),
        "第10条只支持书面合同订立期限，完整回答二倍工资责任还需要第82条。",
    ),
    "rag_sft_v1:0021": (
        ("中华人民共和国民法典#1127",),
        "第705条规定租赁期限，不能支持第一顺序继承人范围；完整回答需要第1127条。",
    ),
    "rag_sft_v1:0022": (
        ("中华人民共和国劳动法#44",),
        "民法典第1032条规定隐私权，不能支持休息日加班工资；完整回答需要劳动法第44条。",
    ),
    "rag_sft_v1:0023": (
        ("中华人民共和国民法典#1077",),
        "第1076条只规定协议离婚申请，第1077条才规定撤回期和申请离婚证的期限。",
    ),
    "rag_sft_v1:0024": (
        ("中华人民共和国消费者权益保护法#24",),
        "第25条规定无理由退货，不能完整支持质量不合格退货及必要运输费；完整回答需要第24条。",
    ),
    "rag_sft_v1:0037": (
        ("中华人民共和国刑法#20",),
        "民法典第577条规定合同违约责任，不能支持防卫过当；完整回答需要刑法第20条。",
    ),
    "rag_sft_v1:0038": (
        ("中华人民共和国行政诉讼法#85",),
        "消费者权益保护法第24条与行政上诉期限无关；完整回答需要行政诉讼法第85条。",
    ),
    "rag_sft_v1:0039": (
        ("中华人民共和国民法典#563", "中华人民共和国民法典#566"),
        "第577条只给出一般违约责任，不能支持催告后解除条件及违约解除后的责任；完整回答需要第563条和第566条。",
    ),
    "rag_sft_v1:0040": (
        ("中华人民共和国民法典#1128",),
        "第1127条只规定法定继承顺序，不能支持代位继承人及份额；完整回答需要第1128条。",
    ),
    "rag_sft_v1:0053": (
        ("中华人民共和国个人信息保护法#15",),
        "第14条规定同意条件，不支持撤回对既往处理效力的影响；完整回答需要第15条。",
    ),
    "rag_sft_v1:0054": (
        ("中华人民共和国劳动合同法#46", "中华人民共和国劳动合同法#47"),
        "第47条只支持补偿计算，完整回答应支付补偿的情形还需要第46条。",
    ),
    "rag_sft_v1:0055": (
        ("中华人民共和国行政诉讼法#20",),
        "刑法第16条规定不可抗力和意外事件，不能支持不动产行政诉讼管辖；完整回答需要行政诉讼法第20条。",
    ),
    "rag_sft_v1:0056": (
        ("中华人民共和国刑法#17",),
        "消费者权益保护法第39条规定争议解决途径，不能支持未成年人刑事责任；完整回答需要刑法第17条。",
    ),
    "rag_sft_v1:0069": (
        ("中华人民共和国个人信息保护法#15",),
        "消费者权益保护法第39条与撤回同意的既往效力无关；完整回答需要个人信息保护法第15条。",
    ),
    "rag_sft_v1:0070": (
        ("中华人民共和国民法典#1086",),
        "刑法第17条规定刑事责任年龄，不能支持离婚后的探望权；完整回答需要民法典第1086条。",
    ),
    "rag_sft_v1:0071": (
        ("中华人民共和国劳动合同法#19", "中华人民共和国劳动合同法#20"),
        "第19条只支持试用期上限，完整回答试用期工资最低标准还需要第20条。",
    ),
    "rag_sft_v1:0072": (
        ("中华人民共和国消费者权益保护法#40",),
        "第39条只规定争议解决途径，不能支持商品缺陷索赔对象和追偿；完整回答需要第40条。",
    ),
    "rag_sft_v1:0079": (
        ("中华人民共和国行政处罚法#44", "中华人民共和国行政处罚法#45"),
        "第44条只支持处罚前告知，完整回答申辩复核和不得加重处罚还需要第45条。",
    ),
    "rag_sft_v1:0080": (
        ("中华人民共和国个人信息保护法#47",),
        "道路交通安全法第76条规定交通事故赔偿，不能支持撤回同意后的删除义务；完整回答需要个人信息保护法第47条。",
    ),
}


class RagSftMigrationError(RuntimeError):
    """表示旧人工事实不能安全迁移为当前 authoring。"""


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagSftMigrationError(f"JSONL 不允许空行: {path}:{line_number}")
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RagSftMigrationError(
                        f"JSON 无效: {path}:{line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise RagSftMigrationError(
                        f"记录必须是 JSON 对象: {path}:{line_number}"
                    )
                records.append(record)
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftMigrationError(f"无法读取 JSONL: {path}") from error
    return records


def _require_non_blank(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RagSftMigrationError(f"{field} 必须是非空字符串")
    return value


def _require_string_list(value: object, field: str, *, min_items: int = 1) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise RagSftMigrationError(f"{field} 必须是非空字符串数组")
    if len(value) < min_items:
        raise RagSftMigrationError(f"{field} 至少包含 {min_items} 项")
    if len(set(value)) != len(value):
        raise RagSftMigrationError(f"{field} 不能包含重复值")
    return value


def _validate_old_record(record: dict[str, object], position: int) -> None:
    prefix = f"旧 authoring 第 {position} 条"
    if set(record) != _OLD_FIELDS:
        raise RagSftMigrationError(f"{prefix}顶层字段不匹配旧 schema")
    _require_non_blank(record["id"], f"{prefix} id")
    _require_non_blank(record["query"], f"{prefix} query")
    if record["review_status"] != "approved":
        raise RagSftMigrationError(f"{prefix}只允许迁移旧 approved 记录")
    chunk_ids = _require_string_list(
        record["ordered_chunk_ids"], f"{prefix} ordered_chunk_ids"
    )
    target = record["target"]
    if not isinstance(target, dict) or set(target) != _OLD_TARGET_FIELDS:
        raise RagSftMigrationError(f"{prefix} target 字段不匹配旧 schema")
    if not isinstance(target["summary"], str) or not isinstance(target["refuse"], bool):
        raise RagSftMigrationError(f"{prefix} target 类型无效")
    cited = _require_string_list(
        target["cited_chunk_ids"],
        f"{prefix} target.cited_chunk_ids",
        min_items=0,
    )
    if not set(cited).issubset(chunk_ids):
        raise RagSftMigrationError(f"{prefix}引用了 ordered evidence 之外的法条")
    quotes = target["quotes"]
    if not isinstance(quotes, list):
        raise RagSftMigrationError(f"{prefix} target.quotes 必须是数组")
    quote_ids = set()
    for quote_index, quote in enumerate(quotes):
        if not isinstance(quote, dict) or set(quote) != _OLD_QUOTE_FIELDS:
            raise RagSftMigrationError(
                f"{prefix} target.quotes[{quote_index}] 字段无效"
            )
        _require_non_blank(quote["chunk_id"], f"{prefix} quote.chunk_id")
        _require_non_blank(quote["text"], f"{prefix} quote.text")
        if quote["chunk_id"] not in chunk_ids:
            raise RagSftMigrationError(f"{prefix} quote 不属于 ordered evidence")
        quote_ids.add(quote["chunk_id"])

    sample_type = record["sample_type"]
    if sample_type in _ANSWER_TYPES | {_EXACT_TYPE}:
        if target["refuse"] or not target["summary"].strip() or not cited or not quotes:
            raise RagSftMigrationError(f"{prefix}回答 target 形状无效")
        if quote_ids != set(cited):
            raise RagSftMigrationError(f"{prefix}回答 citation 与 quote 不一致")
    elif sample_type == _REFUSAL_TYPE:
        if target["summary"] != "" or not target["refuse"] or cited or quotes:
            raise RagSftMigrationError(f"{prefix}拒答 target 必须严格为空内容")
    else:
        raise RagSftMigrationError(f"{prefix} sample_type 无效: {sample_type}")


def validate_authoring_records(
    records: list[dict[str, object]], repository: ArticleRepository
) -> Counter:
    """校验 canonical authoring 的结构、集合关系与逐字审核片段。"""
    seen_ids = set()
    counts = Counter()
    for position, record in enumerate(records, start=1):
        prefix = f"canonical authoring 第 {position} 条"
        if not isinstance(record, dict) or set(record) != AUTHORING_FIELDS:
            raise RagSftMigrationError(f"{prefix}顶层字段必须精确匹配 schema")
        record_id = _require_non_blank(record["id"], f"{prefix} id")
        if _ID_PATTERN.fullmatch(record_id) is None:
            raise RagSftMigrationError(f"{prefix} id 格式无效")
        if record_id in seen_ids:
            raise RagSftMigrationError(f"canonical authoring id 重复: {record_id}")
        seen_ids.add(record_id)
        _require_non_blank(record["query_original"], f"{prefix} query_original")
        source = record["evidence_source"]
        if source not in EVIDENCE_SOURCES:
            raise RagSftMigrationError(f"{prefix} evidence_source 无效")
        visible = _require_string_list(
            record["visible_chunk_ids"], f"{prefix} visible_chunk_ids"
        )
        required = _require_string_list(
            record["required_chunk_ids"], f"{prefix} required_chunk_ids"
        )
        if len(required) > 3:
            raise RagSftMigrationError(f"{prefix} required_chunk_ids 不能超过 3 条")
        for chunk_id in set(visible) | set(required):
            try:
                repository.get_by_chunk_id(chunk_id)
            except (KeyError, ValueError) as error:
                raise RagSftMigrationError(
                    f"{prefix}引用了不存在的法条: {chunk_id}"
                ) from error

        target = record["target"]
        if not isinstance(target, dict) or set(target) != TARGET_FIELDS:
            raise RagSftMigrationError(f"{prefix} target 字段必须精确匹配 schema")
        if not isinstance(target["summary"], str) or not isinstance(target["refuse"], bool):
            raise RagSftMigrationError(f"{prefix} target 类型无效")
        if target["refuse"]:
            if target["summary"] != "":
                raise RagSftMigrationError(f"{prefix}拒答 summary 必须为空")
            if set(required).issubset(visible):
                raise RagSftMigrationError(f"{prefix}拒答证据不能覆盖全部 required GT")
        else:
            if not target["summary"].strip():
                raise RagSftMigrationError(f"{prefix}回答 summary 不能为空")
            if "\n" in target["summary"] or "\r" in target["summary"]:
                raise RagSftMigrationError(f"{prefix}回答 summary 不能换行")
            if not set(required).issubset(visible):
                raise RagSftMigrationError(f"{prefix}回答必须包含全部 required GT")

        if source == "oracle" and (target["refuse"] or visible != required):
            raise RagSftMigrationError(f"{prefix} oracle 必须由有序 required GT 回答")
        if source == "curated_insufficient" and not target["refuse"]:
            raise RagSftMigrationError(f"{prefix} curated_insufficient 必须拒答")
        spans = record["support_spans"]
        if not isinstance(spans, list):
            raise RagSftMigrationError(f"{prefix} support_spans 必须是数组")
        if target["refuse"] and spans:
            raise RagSftMigrationError(f"{prefix}拒答不能携带 support_spans")
        span_pairs = set()
        for span_index, span in enumerate(spans):
            if not isinstance(span, dict) or set(span) != SUPPORT_SPAN_FIELDS:
                raise RagSftMigrationError(
                    f"{prefix} support_spans[{span_index}] 字段无效"
                )
            chunk_id = _require_non_blank(
                span["chunk_id"], f"{prefix} support_span.chunk_id"
            )
            text = _require_non_blank(span["text"], f"{prefix} support_span.text")
            if chunk_id not in required:
                raise RagSftMigrationError(f"{prefix} support_span 不属于 required GT")
            pair = (chunk_id, text)
            if pair in span_pairs:
                raise RagSftMigrationError(f"{prefix} support_spans 包含重复片段")
            span_pairs.add(pair)
            if text not in repository.get_by_chunk_id(chunk_id).content:
                raise RagSftMigrationError(f"{prefix} support_span 不是法条连续原文")

        if record["review_status"] not in REVIEW_STATUSES:
            raise RagSftMigrationError(f"{prefix} review_status 无效")
        if not isinstance(record["review_notes"], str):
            raise RagSftMigrationError(f"{prefix} review_notes 必须是字符串")
        counts[(source, target["refuse"])] += 1
    return counts


def migrate_records(
    source_records: list[dict[str, object]],
    repository: ArticleRepository,
    insufficient_gt: dict[str, tuple[tuple[str, ...], str]],
) -> tuple[list[dict[str, object]], Counter]:
    """把已验证的旧记录转换为连续编号的 canonical draft。"""
    migrated = []
    old_counts = Counter()
    seen_old_ids = set()
    refusal_ids = set()
    for position, record in enumerate(source_records, start=1):
        _validate_old_record(record, position)
        old_id = record["id"]
        if old_id in seen_old_ids:
            raise RagSftMigrationError(f"旧 authoring id 重复: {old_id}")
        seen_old_ids.add(old_id)
        sample_type = record["sample_type"]
        old_counts[sample_type] += 1
        if sample_type == _EXACT_TYPE:
            continue
        if sample_type in _ANSWER_TYPES:
            target = record["target"]
            if target["refuse"] or record["ordered_chunk_ids"] != target["cited_chunk_ids"]:
                raise RagSftMigrationError(f"{old_id} 不是无干扰候选的语义回答")
            visible = list(target["cited_chunk_ids"])
            required = list(target["cited_chunk_ids"])
            support_spans = [
                {"chunk_id": quote["chunk_id"], "text": quote["text"]}
                for quote in target["quotes"]
            ]
            evidence_source = "oracle"
            summary = target["summary"]
            refuse = False
            review_notes = "需按当前三字段协议复核问题边界、最小充分GT和summary。"
        elif sample_type == _REFUSAL_TYPE:
            refusal_ids.add(old_id)
            if old_id not in insufficient_gt:
                raise RagSftMigrationError(f"{old_id} 缺少人工不足证据 GT")
            required_tuple, review_notes = insufficient_gt[old_id]
            visible = list(record["ordered_chunk_ids"])
            required = list(required_tuple)
            support_spans = []
            evidence_source = "curated_insufficient"
            summary = ""
            refuse = True
        else:
            raise RagSftMigrationError(f"{old_id} sample_type 无效: {sample_type}")
        migrated.append(
            {
                "id": f"rag_sft:{len(migrated) + 1:04d}",
                "query_original": record["query"],
                "evidence_source": evidence_source,
                "visible_chunk_ids": visible,
                "required_chunk_ids": required,
                "target": {"summary": summary, "refuse": refuse},
                "support_spans": support_spans,
                "review_status": "draft",
                "review_notes": review_notes,
            }
        )
    unused_gt = set(insufficient_gt) - refusal_ids
    if unused_gt:
        raise RagSftMigrationError(
            "人工不足证据 GT 包含旧数据中不存在的拒答 ID: "
            + ", ".join(sorted(unused_gt))
        )
    validate_authoring_records(migrated, repository)
    return migrated, old_counts


def migrate_project_authoring(
    *, source_path: Path, article_index_path: Path, output_path: Path
) -> dict[str, object]:
    """迁移项目固定的 80 条旧 v1，并原子发布 canonical authoring。"""
    source_path = Path(source_path).resolve()
    article_index_path = Path(article_index_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path == source_path:
        raise RagSftMigrationError("canonical 输出不能覆盖迁移输入")
    if output_path.exists():
        raise RagSftMigrationError(f"canonical 输出已存在，拒绝覆盖: {output_path}")
    partial = output_path.with_name(output_path.name + ".partial")
    if partial.exists():
        raise RagSftMigrationError(f"迁移临时文件已存在，拒绝覆盖: {partial}")
    records = _load_jsonl(source_path)
    try:
        repository = ArticleRepository.from_jsonl(article_index_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RagSftMigrationError("无法加载法条索引") from error
    migrated, old_counts = migrate_records(
        records, repository, CURATED_INSUFFICIENT_GT
    )
    expected_old_counts = Counter(
        {
            _EXACT_TYPE: 15,
            "semantic_single_answer": 20,
            "semantic_multi_answer": 25,
            _REFUSAL_TYPE: 20,
        }
    )
    if old_counts != expected_old_counts or len(migrated) != 65:
        raise RagSftMigrationError(
            f"旧 v1 盘点不符合固定 80→65 迁移边界: {dict(old_counts)}"
        )
    payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in migrated
    )
    partial_created = False
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with partial.open("x", encoding="utf-8", newline="\n") as target:
            partial_created = True
            target.write(payload)
        _load_jsonl(partial)
        if output_path.exists():
            raise RagSftMigrationError(
                f"canonical 输出在迁移期间出现，拒绝覆盖: {output_path}"
            )
        partial.replace(output_path)
    except (OSError, UnicodeError, RagSftMigrationError) as error:
        if partial_created:
            partial.unlink(missing_ok=True)
        raise RagSftMigrationError("无法发布 canonical authoring") from error
    return {
        "source_records": len(records),
        "canonical_drafts": len(migrated),
        "oracle_drafts": sum(
            record["evidence_source"] == "oracle" for record in migrated
        ),
        "curated_insufficient_drafts": sum(
            record["evidence_source"] == "curated_insufficient"
            for record in migrated
        ),
        "excluded_exact": old_counts[_EXACT_TYPE],
        "output": str(output_path),
    }


def main() -> None:
    """解析路径并迁移项目 canonical RAG-SFT authoring。"""
    parser = argparse.ArgumentParser(description="迁移 canonical RAG-SFT authoring")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--article-index", type=Path, default=DEFAULT_ARTICLE_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = migrate_project_authoring(
            source_path=args.source,
            article_index_path=args.article_index,
            output_path=args.output,
        )
    except (RagSftMigrationError, OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"[失败] {error}") from error
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
