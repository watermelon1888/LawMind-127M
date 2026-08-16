"""审计夫子·明察裁判文书的匿名化、抽取噪声和程序性内容。"""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


SOURCE = "fuzi_mingcha"
EXPECTED_FIELDS = {"instruction", "input", "output", "type"}
PLACEHOLDERS = ("[A]", "[B]", "[C]", "[LOC]", "[DATA]")

HTML_ENTITY_PATTERN = re.compile(r"&(?:#x?a0|nbsp);", re.IGNORECASE)
HTML_FRAGMENT_PATTERN = re.compile(
    r"(?:</?[A-Za-z][^>\n]{0,200}>|/[A-Za-z][A-Za-z0-9]*>)",
    re.IGNORECASE,
)
URL_PATTERN = re.compile(r"https?://|www\.", re.IGNORECASE)
PAGE_NUMBER_PATTERN = re.compile(r"(?m)^[ \t]*-[ \t]*\d+[ \t]*-[ \t]*$")
CROSS_LINE_PAGE_NUMBER_PATTERN = re.compile(
    r"(?m)^[ \t]*-[ \t]*\r?\n[ \t]*\d+[ \t]*\r?\n[ \t]*-[ \t]*$"
)
CROSS_LINE_BIRTH_DATE_PATTERN = re.compile(
    r"(?:1\s*[89]|2\s*0)\s*\d\s*\d\s*年"
    r"\s*(?:0?\s*[1-9]|1\s*[0-2])\s*月"
    r"\s*(?:0?\s*[1-9]|[12]\s*\d|3\s*[01])\s*日\s*出生"
)
ID_PATTERN = re.compile(
    r"(?<!\d)[1-9]\d{5}(?:18|19|20)\d{2}"
    r"(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"
)
PHONE_PATTERN = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
ADDRESS_INTRO_PATTERN = re.compile(r"住址|住所地|户籍地|现住址|居住地|(?:^|[，,：:])住")
DETAILED_ADDRESS_PATTERN = re.compile(
    r"(?:"
    r"(?:住址|住所地|户籍地|现住址|居住地|(?:^|[，,：:])住)"
    r"[^\n。；]{0,100}(?:村|组|社|街道|路|巷|号|栋|幢|单元|室)"
    r"|"
    r"[\u4e00-\u9fff]{1,20}(?:路|街|巷|村|组|社)[ \t]*\d+"
    r"(?:号|组|社)[^\n。；]{0,30}(?:栋|幢|单元|室|门面)?"
    r")"
)
NATURAL_PARTY_PATTERN = re.compile(
    r"(?m)^(?:原告|被告人?|上诉人|被上诉人|申请人|被申请人|"
    r"再审申请人|被害人|经营者)(?:（[^）\n]{0,30}）)?[：:]?[ \t]*"
    r"(?P<name>[\u4e00-\u9fff·]{2,8})(?=[，,][ \t]*(?:男|女))"
)
PROCEDURE_PATTERNS = {
    "withdrawal": re.compile(
        r"提出[^。；\n]{0,20}撤诉申请|"
        r"申请[^。；\n]{0,20}撤回[^。；\n]{0,30}(?:起诉|上诉)|"
        r"准许[^。；\n]{0,30}撤回[^。；\n]{0,30}(?:起诉|上诉)"
    ),
    "preservation": re.compile(r"财产保全|诉前保全|保全申请|解除保全"),
    "jurisdiction": re.compile(r"管辖权异议|移送[^。；\n]{0,40}法院|指定管辖"),
    "dismissal": re.compile(r"驳回[^。；\n]{0,30}起诉"),
    "unpaid_fee": re.compile(
        r"(?:收到|接到)[^。；\n]{0,50}(?:缴费|缴纳)[^。；\n]{0,30}通知后"
        r"[^。；\n]{0,30}(?:逾期)?未(?:缴纳|交)[^。；\n]{0,30}"
        r"(?:案件受理费|上诉费)|本案按[^。；\n]{0,50}自动撤回上诉处理"
    ),
    "administrative_enforcement": re.compile(
        r"(?s)(?=.*(?:申请执行人|被执行人|行政非诉))"
        r"(?=.*准予[^。；\n]{0,30}强制执行)"
    ),
}


def _normalized_lines(text: str) -> list[str]:
    return [
        " ".join(line.split())
        for line in html.unescape(text).splitlines()
        if line.split()
    ]


def has_internal_full_duplicate(text: str) -> bool:
    """判断单条记录是否由两份完整相同正文拼接而成。"""
    lines = _normalized_lines(text)
    if len(lines) >= 4 and len(lines) % 2 == 0:
        midpoint = len(lines) // 2
        if lines[:midpoint] == lines[midpoint:]:
            return True

    normalized = " ".join(lines)
    if len(normalized) < 200:
        return False
    midpoint = len(normalized) // 2
    for split in range(max(100, midpoint - 2), min(len(normalized) - 100, midpoint + 2) + 1):
        if normalized[:split].strip() == normalized[split:].strip():
            return True
    return False


def has_unmasked_natural_party(text: str) -> bool:
    """判断文书头部是否保留了明确的自然人当事人姓名。"""
    for match in NATURAL_PARTY_PATTERN.finditer(text):
        if not any(marker in match.group("name") for marker in ("某", "×", "*", "＊")):
            return True
    return False


def has_unknown_control_character(text: str) -> bool:
    """判断正文是否含除常规空白之外的控制、私用或代理字符。"""
    return any(
        char not in "\n\r\t" and unicodedata.category(char) in {"Cc", "Cf", "Co", "Cs"}
        for char in text
    )


def analyze_text(text: str) -> dict[str, object]:
    """返回单条正文的无原文审计信号。"""
    unescaped = html.unescape(text)
    nonempty_lines = [line.strip() for line in unescaped.splitlines() if line.strip()]
    single_character_lines = sum(
        bool(re.fullmatch(r"[\u4e00-\u9fff\d]", line)) for line in nonempty_lines
    )
    return {
        "placeholder_occurrences": {
            placeholder: text.count(placeholder) for placeholder in PLACEHOLDERS
        },
        "html_entity": bool(HTML_ENTITY_PATTERN.search(text)),
        "html_fragment": bool(HTML_FRAGMENT_PATTERN.search(unescaped)),
        "url": bool(URL_PATTERN.search(unescaped)),
        "page_number": bool(PAGE_NUMBER_PATTERN.search(unescaped)),
        "cross_line_page_number": bool(
            CROSS_LINE_PAGE_NUMBER_PATTERN.search(unescaped)
        ),
        "fragmented_lines": single_character_lines >= 6,
        "cross_line_birth_date": bool(
            CROSS_LINE_BIRTH_DATE_PATTERN.search(unescaped)
        ),
        "id_number": bool(ID_PATTERN.search(unescaped)),
        "mobile_number": bool(PHONE_PATTERN.search(unescaped)),
        "address_intro": bool(ADDRESS_INTRO_PATTERN.search(unescaped)),
        "detailed_address": bool(DETAILED_ADDRESS_PATTERN.search(unescaped)),
        "unmasked_natural_party": has_unmasked_natural_party(unescaped),
        "replacement_character": "\ufffd" in unescaped,
        "unknown_control_character": has_unknown_control_character(unescaped),
        "internal_full_duplicate": has_internal_full_duplicate(text),
        "procedures": {
            name: bool(pattern.search(unescaped))
            for name, pattern in PROCEDURE_PATTERNS.items()
        },
    }


def audit_file(input_path: Path, report_path: Path) -> dict[str, object]:
    """全量扫描 JSONL，并写入不含正文和个人信息的汇总报告。"""
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise ValueError(f"缺少输入文件: {input_path}")

    counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    placeholder_records: Counter[str] = Counter()
    placeholder_occurrences: Counter[str] = Counter()
    procedure_counts: Counter[str] = Counter()
    first_failures: list[dict[str, object]] = []
    body_characters = 0
    stable_count_names = (
        "input_records",
        "parsed_records",
        "failed_records",
        "field_anomaly_records",
        "non_string_instruction_records",
        "empty_instruction_records",
        "nonempty_input_records",
        "nonempty_output_records",
        "audited_records",
        "html_entity_records",
        "html_fragment_records",
        "url_records",
        "page_number_records",
        "cross_line_page_number_records",
        "fragmented_lines_records",
        "cross_line_birth_date_records",
        "id_number_records",
        "mobile_number_records",
        "address_intro_records",
        "detailed_address_records",
        "unmasked_natural_party_records",
        "replacement_character_records",
        "unknown_control_character_records",
        "internal_full_duplicate_records",
    )
    for name in stable_count_names:
        counts[name] = 0

    with input_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            counts["input_records"] += 1
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError("记录不是 JSON 对象")
            except (json.JSONDecodeError, ValueError) as error:
                counts["failed_records"] += 1
                if len(first_failures) < 20:
                    first_failures.append({"line": line_number, "reason": str(error)})
                continue

            counts["parsed_records"] += 1
            if set(record) != EXPECTED_FIELDS:
                counts["field_anomaly_records"] += 1

            instruction = record.get("instruction")
            if not isinstance(instruction, str):
                counts["non_string_instruction_records"] += 1
                continue

            if isinstance(record.get("input"), str) and record["input"].strip():
                counts["nonempty_input_records"] += 1
            if isinstance(record.get("output"), str) and record["output"].strip():
                counts["nonempty_output_records"] += 1

            text = instruction.strip()
            if not text:
                counts["empty_instruction_records"] += 1
                continue

            counts["audited_records"] += 1
            type_counts[str(record.get("type", ""))] += 1
            body_characters += len(text)
            signals = analyze_text(text)

            for placeholder, occurrences in signals["placeholder_occurrences"].items():
                placeholder_occurrences[placeholder] += occurrences
                if occurrences:
                    placeholder_records[placeholder] += 1

            for signal in (
                "html_entity",
                "html_fragment",
                "url",
                "page_number",
                "cross_line_page_number",
                "fragmented_lines",
                "cross_line_birth_date",
                "id_number",
                "mobile_number",
                "address_intro",
                "detailed_address",
                "unmasked_natural_party",
                "replacement_character",
                "unknown_control_character",
                "internal_full_duplicate",
            ):
                if signals[signal]:
                    counts[f"{signal}_records"] += 1

            for name, matched in signals["procedures"].items():
                if matched:
                    procedure_counts[name] += 1

    audited_records = counts["audited_records"]
    report = {
        "source": SOURCE,
        "input_file": input_path.name,
        "input_bytes": input_path.stat().st_size,
        **dict(counts),
        "body_characters": body_characters,
        "mean_body_characters": (
            round(body_characters / audited_records, 2) if audited_records else 0
        ),
        "type_counts": dict(sorted(type_counts.items())),
        "placeholder_counts": {
            placeholder: {
                "records": placeholder_records[placeholder],
                "occurrences": placeholder_occurrences[placeholder],
            }
            for placeholder in PLACEHOLDERS
        },
        "procedure_counts": {
            name: procedure_counts[name] for name in PROCEDURE_PATTERNS
        },
        "first_failures": first_failures,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(report_path)
    return report


def main() -> None:
    """解析参数并执行裁判文书增强审计。"""
    parser = argparse.ArgumentParser(description="审计夫子·明察裁判文书质量")
    parser.add_argument("--input", type=Path, required=True, help="原始裁判文书 JSONL")
    parser.add_argument("--report", type=Path, required=True, help="审计报告 JSON")
    args = parser.parse_args()
    report = audit_file(args.input, args.report)
    print(
        f"[完成] 输入 {report['input_records']:,} 条，"
        f"审计 {report.get('audited_records', 0):,} 条，"
        f"失败 {report.get('failed_records', 0):,} 条，"
        f"内部整篇重复 {report.get('internal_full_duplicate_records', 0):,} 条"
    )


if __name__ == "__main__":
    main()
