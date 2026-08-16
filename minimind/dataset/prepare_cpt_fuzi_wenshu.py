"""清洗、标准化并去重夫子·明察裁判文书 CPT 语料。"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sqlite3
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .audit_cpt_fuzi_wenshu import (
    CROSS_LINE_BIRTH_DATE_PATTERN,
    CROSS_LINE_PAGE_NUMBER_PATTERN,
    DETAILED_ADDRESS_PATTERN,
    EXPECTED_FIELDS,
    HTML_FRAGMENT_PATTERN,
    ID_PATTERN,
    PAGE_NUMBER_PATTERN,
    PHONE_PATTERN,
    PROCEDURE_PATTERNS,
    URL_PATTERN,
    has_unknown_control_character,
    has_unmasked_natural_party,
)


SOURCE = "fuzi_mingcha"
ALLOWED_CASE_TYPES = frozenset(
    {"民事案件", "刑事案件", "行政案件", "强制清算与破产案件"}
)
EXCLUDED_SOURCE_FILES = frozenset({"2018.10.jsonl"})
MIN_TEXT_CHARACTERS = 200
SOURCE_FILE_PATTERN = re.compile(
    r"^(?P<year>\d{4})\.(?P<month>\d{2})(?:_\d+)?\.jsonl$"
)
LEGAL_STRUCTURE_PATTERN = re.compile(
    r"本院认为|本院经审理认为|经审理查明|判决如下|裁定如下|决定如下|"
    r"调解如下|判决主文|裁判结果"
)
FILTER_REASONS = (
    "excluded_anonymization_shard",
    "json_error",
    "schema_anomaly",
    "empty_text",
    "disallowed_case_type",
    "procedural_content",
    "personal_information",
    "text_artifact",
    "fragmented_lines",
    "too_short",
    "missing_legal_structure",
)
STANDARDIZED_FIELDS = {
    "id",
    "source",
    "source_file",
    "year",
    "month",
    "case_type",
    "text",
}
ADMINISTRATIVE_ENFORCEMENT_MARKERS = ("申请执行人", "被执行人", "行政非诉")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _normalized_lines(text: str) -> list[str]:
    lines = []
    for raw_line in html.unescape(text).splitlines():
        line = " ".join(raw_line.split())
        if not line or PAGE_NUMBER_PATTERN.fullmatch(line):
            continue
        lines.append(line)
    return lines


def clean_text(text: str) -> tuple[str, dict[str, bool]]:
    """执行不会改变法律含义的正文清理，并折叠单条内部整篇重复。"""
    lines = _normalized_lines(text)
    internal_duplicate = False
    if len(lines) >= 4 and len(lines) % 2 == 0:
        midpoint = len(lines) // 2
        if lines[:midpoint] == lines[midpoint:]:
            lines = lines[:midpoint]
            internal_duplicate = True

    cleaned = "\n".join(lines)
    if not internal_duplicate and len(cleaned) >= MIN_TEXT_CHARACTERS * 2:
        midpoint = len(cleaned) // 2
        for split in range(midpoint - 2, midpoint + 3):
            if cleaned[:split].strip() == cleaned[split:].strip():
                cleaned = cleaned[:split].strip()
                internal_duplicate = True
                break

    return cleaned, {
        "html_unescaped": html.unescape(text) != text,
        "page_number_removed": bool(PAGE_NUMBER_PATTERN.search(html.unescape(text))),
        "internal_duplicate_repaired": internal_duplicate,
    }


def _source_date(source_file: str) -> tuple[int, int]:
    match = SOURCE_FILE_PATTERN.fullmatch(source_file)
    if match is None:
        raise ValueError(f"裁判文书分片文件名不符合年月规则: {source_file}")
    return int(match.group("year")), int(match.group("month"))


def _empty_funnel() -> dict[str, Counter[str]]:
    return {
        reason: Counter(records=0, characters=0) for reason in FILTER_REASONS
    }


def _record_exclusion(
    funnel: dict[str, Counter[str]], reason: str, characters: int
) -> None:
    funnel[reason]["records"] += 1
    funnel[reason]["characters"] += characters


def _valid_schema(record: object) -> bool:
    if not isinstance(record, dict) or set(record) != EXPECTED_FIELDS:
        return False
    return all(isinstance(record.get(field), str) for field in EXPECTED_FIELDS) and not (
        record["input"].strip() or record["output"].strip()
    )


def first_exclusion_reason(text: str) -> str | None:
    """按清洗漏斗顺序返回首个排除原因，避免执行无关的完整审计。"""
    unescaped = html.unescape(text)
    for name, pattern in PROCEDURE_PATTERNS.items():
        if name == "administrative_enforcement" and not any(
            marker in unescaped for marker in ADMINISTRATIVE_ENFORCEMENT_MARKERS
        ):
            continue
        if pattern.search(unescaped):
            return "procedural_content"
    if (
        CROSS_LINE_BIRTH_DATE_PATTERN.search(unescaped)
        or ID_PATTERN.search(unescaped)
        or PHONE_PATTERN.search(unescaped)
        or DETAILED_ADDRESS_PATTERN.search(unescaped)
        or has_unmasked_natural_party(unescaped)
    ):
        return "personal_information"
    if (
        "\ufffd" in unescaped
        or HTML_FRAGMENT_PATTERN.search(unescaped)
        or URL_PATTERN.search(unescaped)
        or has_unknown_control_character(unescaped)
        or CROSS_LINE_PAGE_NUMBER_PATTERN.search(unescaped)
    ):
        return "text_artifact"

    nonempty_lines = [line.strip() for line in unescaped.splitlines() if line.strip()]
    single_character_lines = sum(
        bool(re.fullmatch(r"[\u4e00-\u9fff\d]", line)) for line in nonempty_lines
    )
    if single_character_lines >= 6:
        return "fragmented_lines"
    return None


def clean_shard(input_path: Path, output_path: Path, report_path: Path) -> dict[str, object]:
    """流式清洗一个原始分片，并写出不含原文的漏斗报告。"""
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise ValueError(f"缺少裁判文书分片: {input_path}")
    year, month = _source_date(input_path.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")

    funnel = _empty_funnel()
    counts: Counter[str] = Counter(
        input_records=0,
        input_body_characters=0,
        output_records=0,
        output_characters=0,
        html_unescaped_records=0,
        page_number_removed_records=0,
        internal_duplicate_repaired_records=0,
    )
    case_type_counts: Counter[str] = Counter()
    excluded_case_type_counts: Counter[str] = Counter()

    with input_path.open("r", encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        for line_number, line in enumerate(source, start=1):
            counts["input_records"] += 1

            if input_path.name in EXCLUDED_SOURCE_FILES:
                try:
                    excluded_record = json.loads(line)
                except json.JSONDecodeError:
                    excluded_record = None
                excluded_text = (
                    excluded_record.get("instruction")
                    if isinstance(excluded_record, dict)
                    else None
                )
                body_characters = (
                    len(excluded_text.strip())
                    if isinstance(excluded_text, str)
                    else 0
                )
                counts["input_body_characters"] += body_characters
                _record_exclusion(
                    funnel, "excluded_anonymization_shard", body_characters
                )
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                _record_exclusion(funnel, "json_error", 0)
                continue
            if not _valid_schema(record):
                anomalous_text = (
                    record.get("instruction") if isinstance(record, dict) else None
                )
                body_characters = (
                    len(anomalous_text.strip())
                    if isinstance(anomalous_text, str)
                    else 0
                )
                counts["input_body_characters"] += body_characters
                _record_exclusion(funnel, "schema_anomaly", body_characters)
                continue

            text = record["instruction"].strip()
            counts["input_body_characters"] += len(text)
            if not text:
                _record_exclusion(funnel, "empty_text", 0)
                continue

            case_type = record["type"].strip()
            case_type_counts[case_type] += 1
            if case_type not in ALLOWED_CASE_TYPES:
                excluded_case_type_counts[case_type] += 1
                _record_exclusion(funnel, "disallowed_case_type", len(text))
                continue

            exclusion_reason = first_exclusion_reason(text)
            if exclusion_reason is not None:
                _record_exclusion(funnel, exclusion_reason, len(text))
                continue

            cleaned, cleaning = clean_text(text)
            if len(cleaned) < MIN_TEXT_CHARACTERS:
                _record_exclusion(funnel, "too_short", len(cleaned))
                continue
            if LEGAL_STRUCTURE_PATTERN.search(cleaned) is None:
                _record_exclusion(funnel, "missing_legal_structure", len(cleaned))
                continue

            document_id = f"{SOURCE}:{input_path.name}:{line_number}"
            standardized = {
                "id": document_id,
                "source": SOURCE,
                "source_file": input_path.name,
                "year": year,
                "month": month,
                "case_type": case_type,
                "text": cleaned,
            }
            output.write(json.dumps(standardized, ensure_ascii=False) + "\n")
            counts["output_records"] += 1
            counts["output_characters"] += len(cleaned)
            for name, changed in cleaning.items():
                if changed:
                    counts[f"{name}_records"] += 1

    temporary.replace(output_path)
    excluded_records = sum(
        funnel[reason]["records"] for reason in FILTER_REASONS
    )
    if excluded_records + counts["output_records"] != counts["input_records"]:
        raise RuntimeError(f"分片清洗漏斗未闭合: {input_path.name}")

    report = {
        "source": SOURCE,
        "input_file": input_path.name,
        "input_bytes": input_path.stat().st_size,
        **dict(counts),
        "filter_funnel": {
            reason: dict(funnel[reason]) for reason in FILTER_REASONS
        },
        "case_type_counts": dict(sorted(case_type_counts.items())),
        "excluded_case_type_counts": dict(sorted(excluded_case_type_counts.items())),
        "output_file": output_path.name,
        "funnel_closed": True,
        "complete": True,
    }
    _write_json(report_path, report)
    return report


def _clean_worker(arguments: tuple[Path, Path, Path]) -> dict[str, object]:
    return clean_shard(*arguments)


def _shard_report_path(report_dir: Path, input_path: Path) -> Path:
    return report_dir / f"fuzi-wenshu-clean-{input_path.stem.replace('.', '-')}.json"


def validate_completed_shard(
    input_path: Path,
    output_path: Path,
    report_path: Path,
    *,
    reused: bool,
) -> dict[str, object]:
    """复验已完成分片、报告和原始输入的一致性。"""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取分片报告: {report_path}") from error
    if not isinstance(report, dict):
        raise ValueError(f"分片报告不是 JSON 对象: {report_path}")

    expected_metadata = {
        "source": SOURCE,
        "input_file": input_path.name,
        "input_bytes": input_path.stat().st_size,
        "output_file": output_path.name,
        "funnel_closed": True,
        "complete": True,
    }
    for name, expected in expected_metadata.items():
        if report.get(name) != expected:
            raise ValueError(
                f"分片报告字段不一致: {input_path.name} 的 {name}"
            )

    integer_fields = (
        "input_records",
        "input_body_characters",
        "output_records",
        "output_characters",
    )
    if any(
        not isinstance(report.get(name), int) or report[name] < 0
        for name in integer_fields
    ):
        raise ValueError(f"分片报告计数字段无效: {input_path.name}")
    funnel = report.get("filter_funnel")
    if not isinstance(funnel, dict) or set(funnel) != set(FILTER_REASONS):
        raise ValueError(f"分片报告漏斗字段不完整: {input_path.name}")
    excluded_records = 0
    for reason in FILTER_REASONS:
        item = funnel[reason]
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("records"), int)
            or not isinstance(item.get("characters"), int)
            or item["records"] < 0
            or item["characters"] < 0
        ):
            raise ValueError(f"分片报告漏斗计数无效: {input_path.name}")
        excluded_records += item["records"]
    if excluded_records + report["output_records"] != report["input_records"]:
        raise ValueError(f"分片报告漏斗未闭合: {input_path.name}")

    year, month = _source_date(input_path.name)
    digest = hashlib.sha256()
    output_records = 0
    output_characters = 0
    try:
        with output_path.open("rb") as source:
            for output_line_number, raw_line in enumerate(source, start=1):
                digest.update(raw_line)
                try:
                    record = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(
                        f"输出 JSONL 无法解析: {output_path}:{output_line_number}"
                    ) from error
                if not isinstance(record, dict) or set(record) != STANDARDIZED_FIELDS:
                    raise ValueError(
                        f"输出记录字段不符合约定: {output_path}:{output_line_number}"
                    )
                if (
                    not all(
                        isinstance(record[name], str)
                        for name in ("id", "source", "source_file", "case_type", "text")
                    )
                    or not isinstance(record["year"], int)
                    or not isinstance(record["month"], int)
                    or record["source"] != SOURCE
                    or record["source_file"] != input_path.name
                    or record["year"] != year
                    or record["month"] != month
                    or not record["id"].startswith(f"{SOURCE}:{input_path.name}:")
                ):
                    raise ValueError(
                        f"输出记录元数据不一致: {output_path}:{output_line_number}"
                    )
                output_records += 1
                output_characters += len(record["text"])
    except OSError as error:
        raise ValueError(f"无法读取正式分片输出: {output_path}") from error

    if output_records != report["output_records"]:
        raise ValueError(f"输出记录数与报告不一致: {input_path.name}")
    if output_characters != report["output_characters"]:
        raise ValueError(f"输出字符数与报告不一致: {input_path.name}")
    return {
        **report,
        "reused": reused,
        "output_bytes": output_path.stat().st_size,
        "output_sha256": digest.hexdigest(),
    }


def clean_shards(
    input_paths: list[Path],
    work_dir: Path,
    report_dir: Path,
    workers: int,
    resume_completed: bool = False,
) -> list[dict[str, object]]:
    """按分片并行清洗；返回值始终按来源文件名排序。"""
    if workers <= 0:
        raise ValueError("workers 必须大于 0")
    ordered_paths = sorted((path.resolve() for path in input_paths), key=lambda p: p.name)
    if not ordered_paths:
        raise ValueError("没有找到裁判文书 JSONL 分片")
    if len({path.name for path in ordered_paths}) != len(ordered_paths):
        raise ValueError("裁判文书分片包含重复文件名")

    arguments = [
        (
            path,
            work_dir / path.name,
            _shard_report_path(report_dir, path),
        )
        for path in ordered_paths
    ]
    reports_by_name: dict[str, dict[str, object]] = {}
    pending_arguments = []
    for input_path, output_path, report_path in arguments:
        if resume_completed:
            output_exists = output_path.is_file()
            report_exists = report_path.is_file()
            if output_exists != report_exists:
                raise ValueError(
                    f"恢复时正式输出与分片报告必须同时存在: {input_path.name}"
                )
            if output_exists:
                reports_by_name[input_path.name] = validate_completed_shard(
                    input_path, output_path, report_path, reused=True
                )
                continue
        pending_arguments.append((input_path, output_path, report_path))

    if workers == 1:
        for item in pending_arguments:
            _clean_worker(item)
    elif pending_arguments:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            list(executor.map(_clean_worker, pending_arguments))

    for input_path, output_path, report_path in pending_arguments:
        if resume_completed:
            reports_by_name[input_path.name] = validate_completed_shard(
                input_path, output_path, report_path, reused=False
            )
        else:
            reports_by_name[input_path.name] = json.loads(
                report_path.read_text(encoding="utf-8")
            )
    return [reports_by_name[path.name] for path in ordered_paths]


def merge_deduplicated_shards(
    shard_paths: list[Path], output_path: Path, database_path: Path
) -> dict[str, object]:
    """按固定分片与行顺序合并，并使用磁盘索引执行全局精确去重。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    counts: Counter[str] = Counter(
        candidate_records=0,
        candidate_characters=0,
        output_records=0,
        output_characters=0,
        duplicate_records=0,
        duplicate_characters=0,
    )
    duplicates_by_source: Counter[str] = Counter()

    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            "CREATE TABLE seen (digest BLOB PRIMARY KEY) WITHOUT ROWID"
        )
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for shard_path in sorted(shard_paths, key=lambda path: path.name):
                with shard_path.open("r", encoding="utf-8") as source:
                    for line in source:
                        record = json.loads(line)
                        text = record["text"]
                        counts["candidate_records"] += 1
                        counts["candidate_characters"] += len(text)
                        digest = hashlib.sha256(text.encode("utf-8")).digest()
                        cursor = connection.execute(
                            "INSERT OR IGNORE INTO seen(digest) VALUES (?)",
                            (digest,),
                        )
                        if cursor.rowcount == 0:
                            counts["duplicate_records"] += 1
                            counts["duplicate_characters"] += len(text)
                            duplicates_by_source[record["source_file"]] += 1
                            continue
                        output.write(json.dumps(record, ensure_ascii=False) + "\n")
                        counts["output_records"] += 1
                        counts["output_characters"] += len(text)
                connection.commit()
        temporary.replace(output_path)
    finally:
        connection.close()

    return {
        **dict(counts),
        "duplicate_records_by_source_file": dict(sorted(duplicates_by_source.items())),
        "deduplication": "SHA-256 of standardized text",
    }


def prepare_corpus(
    input_dir: Path,
    work_dir: Path,
    output_path: Path,
    report_dir: Path,
    workers: int,
    resume_completed: bool = False,
) -> dict[str, object]:
    """完成分片清洗、全局去重并写出汇总报告。"""
    input_dir = input_dir.resolve()
    input_paths = sorted(input_dir.glob("*.jsonl"))
    shard_reports = clean_shards(
        input_paths, work_dir, report_dir, workers, resume_completed
    )
    cleaned_paths = [work_dir / report["output_file"] for report in shard_reports]
    merge_report = merge_deduplicated_shards(
        cleaned_paths, output_path, work_dir / "dedup-index.sqlite3"
    )
    expected_candidates = sum(item["output_records"] for item in shard_reports)
    if merge_report["candidate_records"] != expected_candidates:
        raise RuntimeError("分片候选记录数与合并输入不一致")
    if (
        merge_report["output_records"] + merge_report["duplicate_records"]
        != merge_report["candidate_records"]
    ):
        raise RuntimeError("全局去重漏斗未闭合")

    aggregate_funnel = _empty_funnel()
    for report in shard_reports:
        for reason in FILTER_REASONS:
            aggregate_funnel[reason].update(report["filter_funnel"][reason])

    report = {
        "source": SOURCE,
        "input_directory": str(input_dir),
        "input_files": [path.name for path in input_paths],
        "workers": workers,
        "resume_completed": resume_completed,
        "allowed_case_types": sorted(ALLOWED_CASE_TYPES),
        "excluded_source_files": sorted(EXCLUDED_SOURCE_FILES),
        "minimum_text_characters": MIN_TEXT_CHARACTERS,
        "input_records": sum(item["input_records"] for item in shard_reports),
        "input_body_characters": sum(
            item["input_body_characters"] for item in shard_reports
        ),
        "filter_funnel": {
            reason: dict(aggregate_funnel[reason]) for reason in FILTER_REASONS
        },
        **merge_report,
        "output_file": str(output_path.resolve()),
        "funnel_closed": True,
        "complete": True,
    }
    if resume_completed:
        report["shards"] = [
            {
                "input_file": item["input_file"],
                "output_file": item["output_file"],
                "reused": item["reused"],
                "output_bytes": item["output_bytes"],
                "output_sha256": item["output_sha256"],
            }
            for item in shard_reports
        ]
    _write_json(report_dir / "fuzi-wenshu-cleaning-funnel.json", report)
    return report


def main() -> None:
    """解析参数并生成裁判文书 CPT 来源标准化语料。"""
    parser = argparse.ArgumentParser(description="生成夫子·明察裁判文书 CPT 语料")
    parser.add_argument("--input-dir", type=Path, required=True, help="原始文书分片目录")
    parser.add_argument("--work-dir", type=Path, required=True, help="分片候选和去重索引目录")
    parser.add_argument("--output", type=Path, required=True, help="最终标准化 JSONL")
    parser.add_argument("--report-dir", type=Path, required=True, help="漏斗报告目录")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="分片并行进程数，默认保留一个 CPU 核",
    )
    parser.add_argument(
        "--resume-completed",
        action="store_true",
        help="严格复验并复用已完成的正式分片与报告",
    )
    args = parser.parse_args()
    report = prepare_corpus(
        args.input_dir,
        args.work_dir,
        args.output,
        args.report_dir,
        args.workers,
        args.resume_completed,
    )
    print(
        f"[完成] 输入 {report['input_records']:,} 条，"
        f"候选 {report['candidate_records']:,} 条，"
        f"去重后 {report['output_records']:,} 条，"
        f"跨分片精确重复 {report['duplicate_records']:,} 条"
    )
    print(f"报告路径: {args.report_dir / 'fuzi-wenshu-cleaning-funnel.json'}")


if __name__ == "__main__":
    main()
