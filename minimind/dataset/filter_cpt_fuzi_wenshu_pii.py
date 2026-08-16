"""从裁判文书 CPT 标准化语料中排除未匿名化自然人当事人。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from .audit_cpt_fuzi_wenshu import has_unmasked_natural_party
from .audit_cpt_fuzi_wenshu_corpus import _valid_record


PROGRESS_INTERVAL = 100_000
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class PiiFilterError(ValueError):
    """表示裁判文书 PII 定向过滤无法安全完成。"""


def _partial_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".partial")


def filter_corpus(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    expected_input_sha256: str,
    expected_records: int,
    expected_body_characters: int,
) -> dict[str, object]:
    """保持原始记录字节与顺序，只排除未匿名化自然人当事人记录。"""
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    report_path = report_path.resolve()
    if len({input_path, output_path, report_path}) != 3:
        raise PiiFilterError("输入、输出和报告必须使用三个不同路径")
    if not input_path.is_file():
        raise PiiFilterError(f"输入语料不存在: {input_path}")
    if SHA256_PATTERN.fullmatch(expected_input_sha256) is None:
        raise PiiFilterError("expected_input_sha256 格式无效")
    if expected_records <= 0 or expected_body_characters <= 0:
        raise PiiFilterError("预期记录数和正文字符数必须大于零")

    output_partial = _partial_path(output_path)
    report_partial = _partial_path(report_path)
    for path in (output_path, report_path, output_partial, report_partial):
        if path.exists():
            raise PiiFilterError(f"输出路径已存在: {path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter(
        input_records=0,
        input_body_characters=0,
        excluded_records=0,
        excluded_body_characters=0,
        output_records=0,
        output_body_characters=0,
    )
    excluded_by_source_file: Counter[str] = Counter()
    first_excluded_ids: list[str] = []
    input_digest = hashlib.sha256()
    output_digest = hashlib.sha256()

    try:
        with input_path.open("rb") as source, output_partial.open("xb") as output:
            for line_number, raw_line in enumerate(source, start=1):
                counts["input_records"] += 1
                input_digest.update(raw_line)
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise PiiFilterError(
                        f"输入语料不是有效 UTF-8 JSONL: {input_path}:{line_number}"
                    ) from error
                if not _valid_record(record):
                    raise PiiFilterError(
                        f"输入语料 schema 无效: {input_path}:{line_number}"
                    )

                text = record["text"]
                text_characters = len(text)
                counts["input_body_characters"] += text_characters
                if has_unmasked_natural_party(text):
                    counts["excluded_records"] += 1
                    counts["excluded_body_characters"] += text_characters
                    excluded_by_source_file[str(record["source_file"])] += 1
                    if len(first_excluded_ids) < 20:
                        first_excluded_ids.append(str(record["id"]))
                else:
                    output.write(raw_line)
                    output_digest.update(raw_line)
                    counts["output_records"] += 1
                    counts["output_body_characters"] += text_characters

                if counts["input_records"] % PROGRESS_INTERVAL == 0:
                    print(
                        f"[PII 定向过滤] 已扫描 {counts['input_records']:,} 条，"
                        f"排除 {counts['excluded_records']:,} 条",
                        flush=True,
                    )

        actual_input_sha256 = input_digest.hexdigest()
        if actual_input_sha256 != expected_input_sha256:
            raise PiiFilterError("输入语料 SHA-256 与预期不一致")
        if counts["input_records"] != expected_records:
            raise PiiFilterError("输入语料记录数与预期不一致")
        if counts["input_body_characters"] != expected_body_characters:
            raise PiiFilterError("输入语料正文字符数与预期不一致")

        funnel_closed = (
            counts["input_records"]
            == counts["excluded_records"] + counts["output_records"]
            and counts["input_body_characters"]
            == counts["excluded_body_characters"]
            + counts["output_body_characters"]
        )
        if not funnel_closed:
            raise PiiFilterError("PII 定向过滤漏斗未闭合")

        report = {
            "schema_version": "1.0",
            "pipeline": "cpt_fuzi_wenshu_pii_filter",
            "rule": "has_unmasked_natural_party",
            "input": {
                "path": str(input_path),
                "bytes": input_path.stat().st_size,
                "sha256": actual_input_sha256,
                "records": counts["input_records"],
                "body_characters": counts["input_body_characters"],
            },
            "excluded": {
                "records": counts["excluded_records"],
                "body_characters": counts["excluded_body_characters"],
                "by_source_file": dict(sorted(excluded_by_source_file.items())),
                "first_ids": first_excluded_ids,
            },
            "output": {
                "path": str(output_path),
                "bytes": output_partial.stat().st_size,
                "sha256": output_digest.hexdigest(),
                "records": counts["output_records"],
                "body_characters": counts["output_body_characters"],
            },
            "funnel_closed": True,
            "complete": True,
        }
        report_partial.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        output_partial.replace(output_path)
        try:
            report_partial.replace(report_path)
        except OSError:
            output_path.unlink(missing_ok=True)
            raise
        return report
    except (Exception, KeyboardInterrupt):
        output_partial.unlink(missing_ok=True)
        report_partial.unlink(missing_ok=True)
        raise


def main() -> None:
    """解析参数并执行裁判文书 PII 定向过滤。"""
    parser = argparse.ArgumentParser(description="定向过滤裁判文书 CPT 中的自然人姓名")
    parser.add_argument("--input", type=Path, required=True, help="v2 标准化 JSONL")
    parser.add_argument("--output", type=Path, required=True, help="v3 标准化 JSONL")
    parser.add_argument("--report", type=Path, required=True, help="过滤闭合报告")
    parser.add_argument(
        "--expected-input-sha256", required=True, help="v2 语料 SHA-256"
    )
    parser.add_argument(
        "--expected-records", type=int, required=True, help="v2 语料记录数"
    )
    parser.add_argument(
        "--expected-body-characters", type=int, required=True, help="v2 正文字符数"
    )
    args = parser.parse_args()
    report = filter_corpus(
        args.input,
        args.output,
        args.report,
        args.expected_input_sha256,
        args.expected_records,
        args.expected_body_characters,
    )
    print(
        f"[完成] 输入 {report['input']['records']:,} 条，"
        f"排除 {report['excluded']['records']:,} 条，"
        f"输出 {report['output']['records']:,} 条"
    )
    print(f"输出 SHA-256: {report['output']['sha256']}")
    print(f"报告路径: {args.report}")


if __name__ == "__main__":
    main()
