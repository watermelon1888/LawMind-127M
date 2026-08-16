"""校验当前 200 条法律 RAG 开发评估集的目标契约。"""

import json
import os
import re
import sys

EVAL_SET = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_set.jsonl")
ARTICLE_INDEX = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "chunk",
    "article_index.jsonl",
)
EXPECTED_IDS = {f"Q{number:03d}" for number in range(1, 201)}
REQUIRED_FIELDS = {
    "id",
    "query_original",
    "query_formal",
    "query_type",
    "departments",
    "expected_action",
    "action_reason",
    "gt_articles",
}
VALID_QUERY_TYPES = {
    "exact_lookup",
    "legal_query",
    "irrelevant",
}
VALID_DEPARTMENTS = {
    "刑法",
    "宪法及宪法相关法",
    "民法商法",
    "生态环境法",
    "社会法",
    "经济法",
    "行政法",
    "诉讼与非诉讼程序法",
}
VALID_REASONS = {
    "answer": {None},
    "clarify": {None},
    "refuse": {
        "non_legal",
        "time_sensitive",
        "unsupported_legal_source",
        "unsupported_legal_task",
    },
}


def load_index(path):
    """加载法条索引，返回按法名和条号组织的映射。"""

    lookup = {}
    with open(path, encoding="utf-8") as source:
        for line in source:
            line = line.strip()
            if not line:
                continue
            article = json.loads(line)
            lookup[(article["law_name"], article["article_no"])] = article
    law_count = len({law_name for law_name, _ in lookup})
    print(f"索引加载: {len(lookup)} 条法条, {law_count} 部法律")
    return lookup


def _is_nonempty_string(value):
    return isinstance(value, str) and bool(value.strip())


def verify(lookup, eval_path, expected_ids=EXPECTED_IDS):
    """逐条校验闭合 schema、行为契约、GT 可解析性和部门派生结果。"""

    failures = []
    entries = []
    parse_failures = 0
    with open(eval_path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                failures.append(f"[L{line_number}] JSON 解析失败: {error}")
                parse_failures += 1
                continue
            if not isinstance(entry, dict):
                failures.append(f"[L{line_number}] 评估记录必须是 JSON 对象")
                parse_failures += 1
                continue
            entries.append(entry)
    print(f"评估集加载: {len(entries)} 条")

    pass_count = 0
    fail_count = parse_failures
    seen_ids = set()
    for entry in entries:
        qid = entry.get("id", "?")
        entry_failures = []

        actual_fields = set(entry)
        missing_fields = REQUIRED_FIELDS - actual_fields
        unexpected_fields = actual_fields - REQUIRED_FIELDS
        if missing_fields:
            entry_failures.append(f"缺少必填字段: {sorted(missing_fields)}")
        if unexpected_fields:
            entry_failures.append(f"存在 schema 外字段: {sorted(unexpected_fields)}")

        if not isinstance(qid, str) or re.fullmatch(r"Q\d{3}", qid) is None:
            entry_failures.append(f"无效 id: {qid!r}")
        else:
            if qid in seen_ids:
                entry_failures.append("重复 ID")
            seen_ids.add(qid)

        query_original = entry.get("query_original")
        query_formal = entry.get("query_formal")
        query_type = entry.get("query_type")
        expected_action = entry.get("expected_action")
        action_reason = entry.get("action_reason")
        departments = entry.get("departments")
        gt_articles = entry.get("gt_articles")

        if not _is_nonempty_string(query_original):
            entry_failures.append("query_original 必须是非空字符串")
        if not isinstance(query_type, str) or query_type not in VALID_QUERY_TYPES:
            entry_failures.append(f"无效 query_type: {query_type}")
        if not isinstance(expected_action, str) or expected_action not in VALID_REASONS:
            entry_failures.append(f"无效 expected_action: {expected_action}")
        elif not (
            action_reason is None or isinstance(action_reason, str)
        ) or action_reason not in VALID_REASONS[expected_action]:
            entry_failures.append(
                f"action_reason 与 {expected_action} 不相容: {action_reason}"
            )

        if not isinstance(departments, list):
            entry_failures.append("departments 必须是数组")
            departments = []
        elif any(
            not isinstance(department, str) or department not in VALID_DEPARTMENTS
            for department in departments
        ) or len(departments) != len(set(departments)):
            entry_failures.append(f"departments 含无效值或重复值: {departments}")

        if not isinstance(gt_articles, list):
            entry_failures.append("gt_articles 必须是数组")
            gt_articles = []

        if expected_action == "answer":
            if not _is_nonempty_string(query_formal):
                entry_failures.append("answer 记录的 query_formal 必须是非空字符串")
            if query_type == "irrelevant":
                entry_failures.append("answer 记录不能使用 irrelevant")
            if not 1 <= len(gt_articles) <= 3:
                entry_failures.append("answer 记录必须包含 1 至 3 条 gt_articles")
        elif expected_action in ("clarify", "refuse"):
            if query_formal is not None:
                entry_failures.append(
                    f"{expected_action} 记录的 query_formal 必须为 null"
                )
            if departments:
                entry_failures.append(f"{expected_action} 记录的 departments 必须为空")
            if gt_articles:
                entry_failures.append(f"{expected_action} 记录的 gt_articles 必须为空")

        if query_type == "irrelevant" and not (
            expected_action == "refuse" and action_reason == "non_legal"
        ):
            entry_failures.append("irrelevant 只能对应 refuse + non_legal")
        if expected_action == "refuse" and action_reason == "non_legal" and query_type != "irrelevant":
            entry_failures.append("refuse + non_legal 必须使用 irrelevant")
        if expected_action == "refuse" and action_reason != "non_legal" and query_type == "irrelevant":
            entry_failures.append("法律类拒答不能使用 irrelevant")

        article_keys = []
        derived_departments = []
        for position, gt_article in enumerate(gt_articles):
            if not isinstance(gt_article, dict) or set(gt_article) != {
                "law_name",
                "article_no",
            }:
                entry_failures.append(
                    f"gt_articles[{position}] 必须只含 law_name 和 article_no"
                )
                continue
            law_name = gt_article.get("law_name")
            article_no = gt_article.get("article_no")
            if not _is_nonempty_string(law_name) or not _is_nonempty_string(article_no):
                entry_failures.append(f"gt_articles[{position}] 的法名或条号无效")
                continue
            key = (law_name, article_no)
            article_keys.append(key)
            article = lookup.get(key)
            if article is None:
                entry_failures.append(
                    f"gt_articles[{position}] 无法从索引解析: {law_name}#{article_no}"
                )
                continue
            department = article.get("department")
            if department not in VALID_DEPARTMENTS:
                entry_failures.append(
                    f"索引部门无效: {law_name}#{article_no} -> {department}"
                )
            elif department not in derived_departments:
                derived_departments.append(department)

        if len(article_keys) != len(set(article_keys)):
            entry_failures.append("gt_articles 包含重复法名和条号")
        if expected_action == "answer" and departments != derived_departments:
            entry_failures.append(
                f"departments 未按 GT 派生: eval={departments}, actual={derived_departments}"
            )

        if entry_failures:
            fail_count += 1
            failures.extend(f"[{qid}] {failure}" for failure in entry_failures)
        else:
            pass_count += 1

    if expected_ids is not None and seen_ids != set(expected_ids):
        missing_ids = sorted(set(expected_ids) - seen_ids)
        unexpected_ids = sorted(seen_ids - set(expected_ids))
        failures.append(
            f"[集合] ID 空间不完整: missing={missing_ids}, unexpected={unexpected_ids}"
        )
        fail_count += 1

    print(f"\n{'=' * 60}")
    print(f"合计: {len(entries)} 条")
    print(f"通过: {pass_count} 条")
    print(f"失败记录或集合约束: {fail_count} 项")
    if failures:
        print(f"\n失败详情 ({len(failures)} 条):")
        for failure in failures:
            print(f"  {failure}")
    denominator = len(entries) or 1
    print(f"\n记录通过率: {pass_count}/{len(entries)} = {pass_count / denominator * 100:.1f}%")
    return pass_count, fail_count, failures


if __name__ == "__main__":
    if not os.path.exists(EVAL_SET):
        print(f"错误: 评估集文件不存在: {EVAL_SET}")
        sys.exit(1)
    if not os.path.exists(ARTICLE_INDEX):
        print(f"错误: 法条索引不存在: {ARTICLE_INDEX}")
        sys.exit(1)

    article_lookup = load_index(ARTICLE_INDEX)
    _, failed, _ = verify(article_lookup, EVAL_SET)
    sys.exit(0 if failed == 0 else 1)
