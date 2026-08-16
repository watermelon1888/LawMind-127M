"""审计夫子·明察法规预训练数据的模板剥离可行性。"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


SOURCE = "fuzi_mingcha"
BAOFA_CODE_MARKER = "【法宝引证码】"
BAOFA_METADATA_BOUNDARY_CODES = ("CLI.DL.", "CLI.AR.")
BODY_CONTROL_MARKERS = frozenset(
    {"收起", "引用本法", "全部", "未修改", "新增法条", "实质性修改", "非实质性修改"}
)
FOOTER_MARKERS = frozenset({"复制全文", "复制链接", "下载PDF", "下载WORD"})
RESIDUAL_TEMPLATE_MARKERS = (
    BAOFA_CODE_MARKER,
    "微信扫码阅读",
    "正文右侧法宝联想",
    "法宝产品资讯",
)
METADATA_FIELDS = {
    "制定机关": "issuing_authority",
    "发文字号": "document_number",
    "公布日期": "publication_date",
    "施行日期": "effective_date",
    "时效性": "validity",
    "效力位阶": "authority_level",
    "法规类别": "categories",
    "类别": "categories",
    "截止日期": "feedback_deadline",
}


def _normalize_lines(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", html.unescape(text))
    return [" ".join(line.split()) for line in normalized.splitlines() if line.split()]


def _extract_metadata(lines: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in lines:
        for label, field in METADATA_FIELDS.items():
            match = re.match(rf"^{re.escape(label)}\s*[：:]\s*(.+)$", line)
            if not match:
                continue
            value = match.group(1).strip()
            if field == "issuing_authority":
                value = re.sub(r"\s*机构沿革\s*$", "", value)
            if value:
                metadata[field] = value
            break
    return metadata


def _strip_plain_prefix(title: str, body_lines: list[str]) -> list[str]:
    if not body_lines:
        return body_lines
    first = body_lines[0]
    if first == title:
        return body_lines[1:]
    if first.startswith(title):
        remainder = first[len(title) :].strip()
        return ([remainder] if remainder else []) + body_lines[1:]
    return body_lines


def transform_record(record: object, line_number: int, shard_name: str) -> dict[str, str]:
    """将一条原始记录转换为仅含正文与可验证元数据的审计记录。"""
    if not isinstance(record, dict):
        raise ValueError("记录不是 JSON 对象")
    instruction = record.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction 缺失或为空")
    if record.get("input") not in {None, ""} or record.get("output") not in {None, ""}:
        raise ValueError("input/output 不是空字符串")

    lines = _normalize_lines(instruction)
    if len(lines) < 2:
        raise ValueError("规范化后不足两行")
    title = lines[0]
    is_baofa = any(BAOFA_CODE_MARKER in line for line in lines)
    metadata: dict[str, str] = {}

    if is_baofa:
        code_index = next(
            index for index, line in enumerate(lines) if BAOFA_CODE_MARKER in line
        )
        footer_indexes = [
            index
            for index, line in enumerate(lines)
            if index > code_index and line in FOOTER_MARKERS
        ]
        if not footer_indexes:
            raise ValueError("法宝模板缺少正文结束锚点")
        footer_index = footer_indexes[0]
        control_indexes = [
            index
            for index, line in enumerate(
                lines[code_index + 1 : footer_index], code_index + 1
            )
            if line in BODY_CONTROL_MARKERS
        ]
        if control_indexes:
            body_start = control_indexes[-1] + 1
            body_lines = _strip_plain_prefix(title, lines[body_start:footer_index])
        elif any(code in lines[code_index] for code in BAOFA_METADATA_BOUNDARY_CODES):
            metadata_indexes = [
                index
                for index, line in enumerate(
                    lines[code_index + 1 : footer_index], code_index + 1
                )
                if any(
                    re.match(rf"^{re.escape(label)}\s*[：:]", line)
                    for label in METADATA_FIELDS
                )
            ]
            if not metadata_indexes:
                raise ValueError("法宝征求意见模板缺少元数据边界")
            body_start = metadata_indexes[-1] + 1
            body_lines = lines[body_start:footer_index]
        else:
            title_indexes = [
                index
                for index, line in enumerate(
                    lines[code_index + 1 : footer_index], code_index + 1
                )
                if line == title
            ]
            if not title_indexes:
                raise ValueError("法宝模板缺少正文开始锚点")
            body_start = title_indexes[-1] + 1
            body_lines = lines[body_start:footer_index]
        metadata = _extract_metadata(lines[code_index + 1 : body_start])
    else:
        body_lines = _strip_plain_prefix(title, lines[1:])

    if not body_lines:
        raise ValueError("模板剥离后正文为空")
    text = "\n".join(body_lines).strip()
    if not text:
        raise ValueError("模板剥离后正文为空")
    if any(marker in text for marker in RESIDUAL_TEMPLATE_MARKERS):
        raise ValueError("正文仍包含网页模板标记")

    transformed = {
        "id": f"fuzi_mingcha_{shard_name}_{line_number:06d}",
        "source": SOURCE,
        "title": title,
        "text": text,
    }
    transformed.update(metadata)
    return transformed


def audit_file(input_path: Path, report_path: Path) -> dict[str, object]:
    """全量验证模板剥离，并只写入不含正文的汇总报告。"""
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise ValueError(f"缺少输入文件: {input_path}")

    counts: Counter[str] = Counter()
    failure_reasons: Counter[str] = Counter()
    metadata_counts: Counter[str] = Counter()
    first_failures: list[dict[str, object]] = []
    body_hashes: set[bytes] = set()
    body_characters = 0
    min_body_characters: int | None = None
    max_body_characters = 0
    shard_name = input_path.stem

    with input_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            counts["input_records"] += 1
            try:
                record = json.loads(line)
                transformed = transform_record(record, line_number, shard_name)
            except (json.JSONDecodeError, ValueError) as error:
                reason = str(error)
                counts["failed_records"] += 1
                failure_reasons[reason] += 1
                if len(first_failures) < 20:
                    first_failures.append({"line": line_number, "reason": reason})
                continue

            counts["transformed_records"] += 1
            raw_instruction = record["instruction"]
            if BAOFA_CODE_MARKER in raw_instruction:
                counts["baofa_template_records"] += 1
            else:
                counts["plain_records"] += 1
            for field in set(METADATA_FIELDS.values()):
                if field in transformed:
                    metadata_counts[field] += 1

            text = transformed["text"]
            body_characters += len(text)
            min_body_characters = (
                len(text)
                if min_body_characters is None
                else min(min_body_characters, len(text))
            )
            max_body_characters = max(max_body_characters, len(text))
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            if digest in body_hashes:
                counts["exact_duplicate_records"] += 1
            else:
                body_hashes.add(digest)

    transformed_records = counts["transformed_records"]
    report = {
        "source": SOURCE,
        "input_file": input_path.name,
        "input_bytes": input_path.stat().st_size,
        **dict(counts),
        "success_rate": (
            round(transformed_records / counts["input_records"], 6)
            if counts["input_records"]
            else 0
        ),
        "metadata_counts": dict(sorted(metadata_counts.items())),
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "first_failures": first_failures,
        "body_characters": body_characters,
        "mean_body_characters": (
            round(body_characters / transformed_records, 2) if transformed_records else 0
        ),
        "min_body_characters": min_body_characters or 0,
        "max_body_characters": max_body_characters,
        "complete": counts["failed_records"] == 0,
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
    """解析参数并执行夫子·明察法规模板审计。"""
    parser = argparse.ArgumentParser(description="审计夫子·明察法规模板剥离可行性")
    parser.add_argument("--input", type=Path, required=True, help="原始法规 JSONL")
    parser.add_argument("--report", type=Path, required=True, help="审计报告 JSON")
    args = parser.parse_args()
    report = audit_file(args.input, args.report)
    print(
        f"[完成] 输入 {report['input_records']:,} 条，"
        f"成功 {report['transformed_records']:,} 条，"
        f"失败 {report['failed_records']:,} 条，"
        f"精确重复 {report.get('exact_duplicate_records', 0):,} 条"
    )


if __name__ == "__main__":
    main()
