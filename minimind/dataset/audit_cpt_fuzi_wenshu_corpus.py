"""审计标准化裁判文书 CPT 语料，并生成确定性分层样本。"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path

from .audit_cpt_fuzi_wenshu import PLACEHOLDERS, PROCEDURE_PATTERNS, analyze_text
from .prepare_cpt_fuzi_wenshu import (
    ALLOWED_CASE_TYPES,
    LEGAL_STRUCTURE_PATTERN,
    MIN_TEXT_CHARACTERS,
    SOURCE,
    SOURCE_FILE_PATTERN,
)


EXPECTED_FIELDS = {
    "id",
    "source",
    "source_file",
    "year",
    "month",
    "case_type",
    "text",
}
QUALITY_SIGNALS = (
    "missing_legal_structure",
    "procedural_content",
    "personal_information",
    "replacement_character",
    "html_fragment",
    "url",
    "unknown_control_character",
    "fragmented_lines",
    "html_entity",
    "page_number",
    "cross_line_page_number",
    "internal_full_duplicate",
)
FULL_RECORD_SIGNALS = (
    "invalid_source",
    "disallowed_case_type",
    "source_date_mismatch",
    "too_short",
)
PROGRESS_INTERVAL = 100_000


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def allocate_stratified_samples(
    stratum_counts: Counter[tuple[int, str]], sample_size: int
) -> dict[tuple[int, str], int]:
    """每层至少一条，其余名额按层内记录数确定性分配。"""
    if sample_size <= 0:
        raise ValueError("sample_size 必须大于 0")
    strata = sorted(key for key, count in stratum_counts.items() if count > 0)
    if not strata:
        raise ValueError("标准化语料中没有可抽样记录")
    target_size = min(sample_size, sum(stratum_counts.values()))
    if target_size < len(strata):
        raise ValueError(
            f"样本数 {target_size} 小于非空分层数 {len(strata)}，无法每层至少抽一条"
        )

    allocation = {key: 1 for key in strata}
    remaining = target_size - len(strata)
    capacities = {key: stratum_counts[key] - 1 for key in strata}
    while remaining:
        total_capacity = sum(capacities.values())
        if total_capacity < remaining:
            raise RuntimeError("分层样本容量不足")
        ideals = {
            key: remaining * capacities[key] / total_capacity
            for key in strata
            if capacities[key] > 0
        }
        added = 0
        for key, ideal in ideals.items():
            count = min(capacities[key], int(ideal))
            allocation[key] += count
            capacities[key] -= count
            remaining -= count
            added += count
        if not remaining:
            break
        ranked = sorted(
            (key for key in strata if capacities[key] > 0),
            key=lambda key: (-(ideals.get(key, 0) % 1), key),
        )
        if not ranked:
            raise RuntimeError("分层样本余数无法分配")
        for key in ranked:
            allocation[key] += 1
            capacities[key] -= 1
            remaining -= 1
            added += 1
            if not remaining:
                break
        if added == 0:
            raise RuntimeError("分层样本分配没有进展")
    return allocation


def _valid_record(record: object) -> bool:
    if not isinstance(record, dict) or set(record) != EXPECTED_FIELDS:
        return False
    string_fields = ("id", "source", "source_file", "case_type", "text")
    return all(isinstance(record.get(name), str) for name in string_fields) and all(
        isinstance(record.get(name), int) and not isinstance(record.get(name), bool)
        for name in ("year", "month")
    )


def _source_date_matches(record: dict[str, object]) -> bool:
    match = SOURCE_FILE_PATTERN.fullmatch(str(record["source_file"]))
    return bool(
        match
        and int(match.group("year")) == record["year"]
        and int(match.group("month")) == record["month"]
    )


def _sample_score(seed: int, document_id: str, line_number: int) -> int:
    value = f"{seed}\0{document_id}\0{line_number}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _update_reservoir(
    reservoir: list[tuple[int, str, dict[str, object]]],
    record: dict[str, object],
    score: int,
    capacity: int,
) -> None:
    entry = (-score, str(record["id"]), record)
    if len(reservoir) < capacity:
        heapq.heappush(reservoir, entry)
    elif entry > reservoir[0]:
        heapq.heapreplace(reservoir, entry)


def _audit_quality_sample(
    records: list[dict[str, object]],
) -> dict[str, object]:
    """仅对确定性分层样本运行昂贵质量规则。"""
    quality_counts: Counter[str] = Counter({name: 0 for name in QUALITY_SIGNALS})
    procedure_counts: Counter[str] = Counter(
        {name: 0 for name in PROCEDURE_PATTERNS}
    )
    personal_information_counts: Counter[str] = Counter(
        {
            name: 0
            for name in (
                "cross_line_birth_date",
                "id_number",
                "mobile_number",
                "detailed_address",
                "unmasked_natural_party",
            )
        }
    )
    placeholder_records: Counter[str] = Counter()
    placeholder_occurrences: Counter[str] = Counter()

    for record in records:
        text = str(record["text"])
        signals = analyze_text(text)
        if LEGAL_STRUCTURE_PATTERN.search(text) is None:
            quality_counts["missing_legal_structure"] += 1
        for name, matched in signals["procedures"].items():
            if matched:
                procedure_counts[name] += 1
        if any(signals["procedures"].values()):
            quality_counts["procedural_content"] += 1
        for name in personal_information_counts:
            if signals[name]:
                personal_information_counts[name] += 1
        if any(signals[name] for name in personal_information_counts):
            quality_counts["personal_information"] += 1
        for name in (
            "replacement_character",
            "html_fragment",
            "url",
            "unknown_control_character",
            "fragmented_lines",
            "html_entity",
            "page_number",
            "cross_line_page_number",
            "internal_full_duplicate",
        ):
            if signals[name]:
                quality_counts[name] += 1
        for placeholder, occurrences in signals["placeholder_occurrences"].items():
            placeholder_occurrences[placeholder] += occurrences
            if occurrences:
                placeholder_records[placeholder] += 1

    return {
        "scope": "deterministic_stratified_sample",
        "checked_records": len(records),
        "violation_counts": dict(quality_counts),
        "procedure_violation_counts": dict(procedure_counts),
        "personal_information_violation_counts": dict(
            personal_information_counts
        ),
        "placeholder_counts": {
            placeholder: {
                "records": placeholder_records[placeholder],
                "occurrences": placeholder_occurrences[placeholder],
            }
            for placeholder in PLACEHOLDERS
        },
        "passed": not any(quality_counts.values()),
    }


def audit_corpus(
    input_path: Path,
    report_path: Path,
    sample_path: Path,
    sample_size: int = 200,
    seed: int = 20_260_730,
    expected_records: int | None = None,
    expected_body_characters: int | None = None,
) -> dict[str, object]:
    """全量审计结构与完整性，并抽样审计昂贵质量规则。"""
    input_path = input_path.resolve()
    if not input_path.is_file():
        raise ValueError(f"缺少标准化裁判文书语料: {input_path}")
    if sample_size <= 0:
        raise ValueError("sample_size 必须大于 0")
    if expected_records is not None and expected_records <= 0:
        raise ValueError("expected_records 必须大于 0")
    if expected_body_characters is not None and expected_body_characters <= 0:
        raise ValueError("expected_body_characters 必须大于 0")
    resolved_outputs = {report_path.resolve(), sample_path.resolve()}
    if input_path in resolved_outputs or len(resolved_outputs) != 2:
        raise ValueError("输入、报告和样本必须使用三个不同路径")

    counts: Counter[str] = Counter(
        input_records=0,
        parsed_records=0,
        json_error_records=0,
        unicode_error_records=0,
        schema_anomaly_records=0,
        body_characters=0,
    )
    full_record_counts: Counter[str] = Counter(
        {name: 0 for name in FULL_RECORD_SIGNALS}
    )
    year_counts: Counter[int] = Counter()
    case_type_counts: Counter[str] = Counter()
    stratum_counts: Counter[tuple[int, str]] = Counter()
    stratum_characters: Counter[tuple[int, str]] = Counter()
    reservoirs: dict[
        tuple[int, str], list[tuple[int, str, dict[str, object]]]
    ] = defaultdict(list)
    input_digest = hashlib.sha256()
    min_characters: int | None = None
    max_characters = 0

    with input_path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            counts["input_records"] += 1
            input_digest.update(raw_line)
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                counts["unicode_error_records"] += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                counts["json_error_records"] += 1
                continue
            counts["parsed_records"] += 1
            if not _valid_record(record):
                counts["schema_anomaly_records"] += 1
                continue

            text = record["text"]
            year = record["year"]
            case_type = record["case_type"]
            stratum = (year, case_type)
            text_characters = len(text)
            counts["body_characters"] += text_characters
            min_characters = (
                text_characters
                if min_characters is None
                else min(min_characters, text_characters)
            )
            max_characters = max(max_characters, text_characters)
            year_counts[year] += 1
            case_type_counts[case_type] += 1
            stratum_counts[stratum] += 1
            stratum_characters[stratum] += text_characters

            if record["source"] != SOURCE:
                full_record_counts["invalid_source"] += 1
            if case_type not in ALLOWED_CASE_TYPES:
                full_record_counts["disallowed_case_type"] += 1
            if not _source_date_matches(record):
                full_record_counts["source_date_mismatch"] += 1
            if text_characters < MIN_TEXT_CHARACTERS:
                full_record_counts["too_short"] += 1

            score = _sample_score(seed, str(record["id"]), line_number)
            _update_reservoir(reservoirs[stratum], record, score, sample_size)
            if counts["input_records"] % PROGRESS_INTERVAL == 0:
                print(
                    f"[轻量审计] 已扫描 {counts['input_records']:,} 条，"
                    f"有效 {sum(stratum_counts.values()):,} 条",
                    flush=True,
                )

    valid_records = sum(stratum_counts.values())
    if valid_records == 0:
        raise ValueError("标准化裁判文书语料没有有效记录")
    allocation = allocate_stratified_samples(stratum_counts, sample_size)
    samples: list[tuple[int, str, int, dict[str, object]]] = []
    for (year, case_type), requested in allocation.items():
        selected = sorted(
            reservoirs[(year, case_type)],
            key=lambda item: (-item[0], item[1]),
        )[:requested]
        samples.extend((year, case_type, -score, record) for score, _, record in selected)
    samples.sort(key=lambda item: (item[0], item[1], item[2], item[3]["id"]))
    sample_records = [item[3] for item in samples]
    if len(sample_records) != sum(allocation.values()):
        raise RuntimeError("分层样本输出数与分配数不一致")
    quality_sample = _audit_quality_sample(sample_records)

    distribution = []
    for (year, case_type), records in sorted(stratum_counts.items()):
        characters = stratum_characters[(year, case_type)]
        distribution.append(
            {
                "year": year,
                "case_type": case_type,
                "records": records,
                "characters": characters,
                "mean_characters": round(characters / records, 2),
                "sample_records": allocation[(year, case_type)],
            }
        )

    full_integrity_checks = {
        "all_lines_parsed": counts["parsed_records"] == counts["input_records"],
        "schema_valid": counts["schema_anomaly_records"] == 0,
        "record_count_matches": (
            expected_records is None or valid_records == expected_records
        ),
        "body_characters_match": (
            expected_body_characters is None
            or counts["body_characters"] == expected_body_characters
        ),
        "source_valid": full_record_counts["invalid_source"] == 0,
        "case_type_valid": full_record_counts["disallowed_case_type"] == 0,
        "source_date_matches": full_record_counts["source_date_mismatch"] == 0,
        "minimum_length_valid": full_record_counts["too_short"] == 0,
    }
    passed = all(full_integrity_checks.values()) and quality_sample["passed"] is True
    report = {
        "audit_mode": "full_structure_sampled_quality_v1",
        "source": SOURCE,
        "input_file": input_path.name,
        "input_bytes": input_path.stat().st_size,
        "input_sha256": input_digest.hexdigest(),
        **dict(counts),
        "valid_records": valid_records,
        "mean_body_characters": round(counts["body_characters"] / valid_records, 2),
        "min_body_characters": min_characters,
        "max_body_characters": max_characters,
        "year_counts": {str(key): year_counts[key] for key in sorted(year_counts)},
        "case_type_counts": dict(sorted(case_type_counts.items())),
        "year_case_type_distribution": distribution,
        "expected": {
            "records": expected_records,
            "body_characters": expected_body_characters,
        },
        "full_record_violation_counts": dict(full_record_counts),
        "full_integrity_checks": full_integrity_checks,
        "quality_sample": quality_sample,
        "sample": {
            "seed": seed,
            "requested_records": sample_size,
            "output_records": len(sample_records),
            "output_file": sample_path.name,
        },
        "passed": passed,
        "complete": True,
    }
    _write_jsonl(sample_path, sample_records)
    _write_json(report_path, report)
    return report


def main() -> None:
    """解析参数并审计最终裁判文书 CPT 语料。"""
    parser = argparse.ArgumentParser(description="审计最终裁判文书 CPT 语料")
    parser.add_argument("--input", type=Path, required=True, help="最终标准化 JSONL")
    parser.add_argument("--report", type=Path, required=True, help="审计报告 JSON")
    parser.add_argument("--sample", type=Path, required=True, help="人工审阅样本 JSONL")
    parser.add_argument("--sample-size", type=int, default=200, help="分层样本总数")
    parser.add_argument("--seed", type=int, default=20_260_730, help="确定性抽样 seed")
    parser.add_argument(
        "--expected-records", type=int, required=True, help="清洗漏斗确认的最终记录数"
    )
    parser.add_argument(
        "--expected-body-characters",
        type=int,
        required=True,
        help="清洗漏斗确认的最终正文字符数",
    )
    args = parser.parse_args()
    report = audit_corpus(
        args.input,
        args.report,
        args.sample,
        args.sample_size,
        args.seed,
        args.expected_records,
        args.expected_body_characters,
    )
    print(
        f"[完成] 审计 {report['valid_records']:,} 条，"
        f"抽样 {report['sample']['output_records']:,} 条，"
        f"passed={report['passed']}"
    )
    print(f"报告路径: {args.report}")
    print(f"样本路径: {args.sample}")


if __name__ == "__main__":
    main()
