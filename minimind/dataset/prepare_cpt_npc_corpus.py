"""生成国家法律法规数据库的 CPT 来源标准化语料。"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Iterable

from minimind.dataset.audit_npc_legal_data import classify_token_count, load_tokenizer
from minimind.dataset.import_npc_legal_archives import (
    EXPECTED_DOCUMENTS,
    EXPECTED_QUERY_RECORDS,
)
from minimind.dataset.prepare_tokenizer_corpus import extract_docx_paragraphs


SOURCE = "npc_flk"
SOURCE_URL = "https://flk.npc.gov.cn/search"
EXPECTED_DUPLICATE_RECORDS = 5
OUTPUT_DIRECTORY = "cpt-standardized"
CORPUS_FILENAME = "cpt-npc-flk.jsonl"
DUPLICATES_FILENAME = "cpt-npc-flk-duplicates.jsonl"
REPORT_FILENAME = "cpt-npc-flk-report.json"

TITLE_SUFFIX_PATTERN = re.compile(r"_(?:[0-9]{8})?$")
DECIMAL_MARKER_PATTERN = re.compile(r"(?<=\d)\ue010(?=\d)")
LAYOUT_PRIVATE_USE = frozenset(
    {
        "\ue003",
        "\ue004",
        "\ue005",
        "\ue008",
        "\ue009",
        "\ue5f9",
        "\ue73c",
    }
)
KNOWN_CHARACTER_REPAIRS = {
    "\ue073": "拐",
    "\ue584": "。",
    "\ue723": "：",
}
REPLACEMENT_CHARACTER_SEQUENCE = "本行政区\ufffd\ufffd\ufffd发展规划"


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise ValueError(f"缺少 JSONL 文件: {path}")
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"JSONL 第 {line_number} 行无法解析: {path}") from error
        if not isinstance(record, dict):
            raise ValueError(f"JSONL 第 {line_number} 行不是对象: {path}")
        records.append(record)
    return records


def _records_by_id(
    records: Iterable[dict[str, object]], id_field: str, description: str
) -> dict[str, dict[str, object]]:
    records_by_id: dict[str, dict[str, object]] = {}
    for record in records:
        document_id = record.get(id_field)
        if not isinstance(document_id, str) or not document_id:
            raise ValueError(f"{description}包含缺失或无效的文档 ID")
        if document_id in records_by_id:
            raise ValueError(f"{description}包含重复文档 ID: {document_id}")
        records_by_id[document_id] = record
    return records_by_id


def _resolve_relative(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("文档清单包含缺失或无效的相对路径")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"文档清单路径越界: {value}") from error
    if not path.is_file():
        raise ValueError(f"文档文件不存在: {path}")
    return path


def _document_path(
    root: Path,
    inventory_record: dict[str, object],
    repairs: dict[str, dict[str, object]],
) -> Path:
    document_id = str(inventory_record["document_id"])
    repair = repairs.get(document_id)
    if repair is not None:
        return _resolve_relative(root, repair.get("repaired_path"))

    extension = inventory_record.get("original_extension")
    if extension == ".doc":
        return _resolve_relative(root, inventory_record.get("converted_path"))
    if extension not in {".docx", ".docm"}:
        raise ValueError(f"不支持的文档格式: {extension}")
    return _resolve_relative(root, inventory_record.get("original_path"))


def extract_title(source_member_path: object) -> str:
    """从官方导出文件名中移除扩展名和固定导出后缀。"""
    if not isinstance(source_member_path, str) or not source_member_path:
        raise ValueError("文档清单缺少来源成员路径")
    filename = PurePosixPath(source_member_path.replace("\\", "/")).name
    stem = Path(filename).stem
    title = TITLE_SUFFIX_PATTERN.sub("", stem)
    if not title or title == stem:
        raise ValueError(f"来源文件名不符合标题规则: {source_member_path}")
    title = unicodedata.normalize("NFC", title)
    _reject_unknown_characters(title, "法规标题")
    return title


def _reject_unknown_characters(text: str, description: str) -> None:
    invalid = Counter(
        char
        for char in text
        if char == "\ufffd"
        or unicodedata.category(char) in {"Cc", "Cf", "Co", "Cs"}
        and char != "\n"
    )
    if not invalid:
        return
    details = ", ".join(
        f"U+{ord(char):04X}×{count}" for char, count in sorted(invalid.items())
    )
    raise ValueError(f"{description}包含未知异常字符: {details}")


def standardize_text(text: str) -> tuple[str, Counter[str]]:
    """按已审定的保守规则规范化一部法规正文。"""
    stats: Counter[str] = Counter()
    normalized = unicodedata.normalize("NFC", text)
    if normalized != text:
        stats["nfc_changed_documents"] = 1

    replacement_sequences = normalized.count(REPLACEMENT_CHARACTER_SEQUENCE)
    if replacement_sequences:
        normalized = normalized.replace(
            REPLACEMENT_CHARACTER_SEQUENCE, "本行政区域发展规划"
        )
        stats["repaired_replacement_character_sequences"] = replacement_sequences
        stats["repaired_replacement_characters"] = replacement_sequences * 3

    decimal_markers = len(DECIMAL_MARKER_PATTERN.findall(normalized))
    if decimal_markers:
        normalized = DECIMAL_MARKER_PATTERN.sub(".", normalized)
        stats["replaced_decimal_markers"] = decimal_markers

    for old, new in KNOWN_CHARACTER_REPAIRS.items():
        count = normalized.count(old)
        if count:
            normalized = normalized.replace(old, new)
            stats[f"replaced_u_{ord(old):04x}"] = count

    layout_count = sum(normalized.count(char) for char in LAYOUT_PRIVATE_USE)
    if layout_count:
        normalized = normalized.translate(
            {ord(char): None for char in LAYOUT_PRIVATE_USE}
        )
        stats["removed_layout_private_use"] = layout_count

    for char, key in (
        ("\u200b", "removed_zero_width_space"),
        ("\u200c", "removed_zero_width_non_joiner"),
        ("\u007f", "removed_delete_control"),
    ):
        count = normalized.count(char)
        if count:
            normalized = normalized.replace(char, "")
            stats[key] = count

    lines = [" ".join(line.split()) for line in normalized.split("\n")]
    normalized = "\n".join(line for line in lines if line)
    if not normalized:
        raise ValueError("法规正文在标准化后为空")
    _reject_unknown_characters(normalized, "法规正文")
    return normalized, stats


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def prepare_corpus(
    input_root: Path,
    tokenizer: object,
    expected_documents: int = EXPECTED_DOCUMENTS,
    expected_duplicate_records: int = EXPECTED_DUPLICATE_RECORDS,
    official_result_count: int = EXPECTED_QUERY_RECORDS,
) -> dict[str, object]:
    """生成去重后的来源标准化语料、重复关系和汇总报告。"""
    if len(tokenizer) != 12_000:
        raise ValueError(f"Tokenizer 词表大小应为 12000，实际为 {len(tokenizer)}")

    input_root = input_root.resolve()
    inventory = _read_jsonl(input_root / "document-inventory.jsonl")
    inventory_by_id = _records_by_id(inventory, "document_id", "文档清单")
    repairs = _records_by_id(
        _read_jsonl(input_root / "document-repair-manifest.jsonl"),
        "document_id",
        "修复清单",
    )
    if len(inventory_by_id) != expected_documents:
        raise ValueError(
            f"文档数应为 {expected_documents}，实际为 {len(inventory_by_id)}"
        )
    if not set(repairs).issubset(inventory_by_id):
        unknown_ids = sorted(set(repairs) - set(inventory_by_id))
        raise ValueError(f"修复清单包含未知文档 ID: {unknown_ids}")

    output_records: list[dict[str, object]] = []
    duplicate_records: list[dict[str, str]] = []
    first_id_by_text: dict[str, str] = {}
    cleaning_counts: Counter[str] = Counter()
    title_body_prefix_matches = 0
    output_characters = 0
    output_tokens = 0

    for document_id in sorted(inventory_by_id):
        inventory_record = inventory_by_id[document_id]
        document_path = _document_path(input_root, inventory_record, repairs)
        raw_text = "\n".join(extract_docx_paragraphs(document_path))
        if not raw_text:
            raise ValueError(f"法规正文为空: {document_id}")
        try:
            text, document_stats = standardize_text(raw_text)
        except ValueError as error:
            raise ValueError(f"{document_id}: {error}") from error
        cleaning_counts.update(document_stats)
        title = extract_title(inventory_record.get("source_member_path"))
        if text.startswith(title):
            title_body_prefix_matches += 1

        duplicate_of = first_id_by_text.get(text)
        if duplicate_of is not None:
            duplicate_records.append({"id": document_id, "duplicate_of": duplicate_of})
            continue

        first_id_by_text[text] = document_id
        output_records.append(
            {"id": document_id, "source": SOURCE, "title": title, "text": text}
        )
        output_characters += len(text)
        output_tokens += len(tokenizer.encode(text, add_special_tokens=False))

    if len(duplicate_records) != expected_duplicate_records:
        raise ValueError(
            "精确重复记录数与审定结果不一致: "
            f"期望 {expected_duplicate_records}，实际 {len(duplicate_records)}"
        )

    output_root = input_root / OUTPUT_DIRECTORY
    output_root.mkdir(parents=True, exist_ok=True)
    corpus_path = output_root / CORPUS_FILENAME
    duplicates_path = output_root / DUPLICATES_FILENAME
    report_path = output_root / REPORT_FILENAME
    duplicate_groups = len({record["duplicate_of"] for record in duplicate_records})
    report = {
        "source": SOURCE,
        "source_url": SOURCE_URL,
        "official_result_count": official_result_count,
        "exported_document_count": len(inventory_by_id),
        "unexported_record_count": official_result_count - len(inventory_by_id),
        "repaired_document_count": len(repairs),
        "output_document_count": len(output_records),
        "duplicate_record_count": len(duplicate_records),
        "duplicate_group_count": duplicate_groups,
        "title_body_prefix_matches": title_body_prefix_matches,
        "title_body_prefix_mismatches": len(inventory_by_id) - title_body_prefix_matches,
        "unicode_normalization": "NFC",
        "cleaning_counts": dict(sorted(cleaning_counts.items())),
        "output_characters": output_characters,
        "output_tokens": output_tokens,
        "size_conclusion": classify_token_count(output_tokens),
        "output_files": [CORPUS_FILENAME, DUPLICATES_FILENAME, REPORT_FILENAME],
        "complete": True,
    }

    _write_jsonl(corpus_path, output_records)
    _write_jsonl(duplicates_path, duplicate_records)
    _write_json(report_path, report)
    return report


def main() -> None:
    """解析命令行参数并生成 CPT 来源标准化语料。"""
    parser = argparse.ArgumentParser(
        description="生成国家法律法规数据库的 CPT 来源标准化语料"
    )
    parser.add_argument("--input-root", type=Path, required=True, help="离线导入输出根目录")
    parser.add_argument(
        "--tokenizer-path", type=Path, required=True, help="当前 12000 词表 Tokenizer"
    )
    args = parser.parse_args()

    report = prepare_corpus(args.input_root, load_tokenizer(args.tokenizer_path))
    print(
        f"[完成] 输入 {report['exported_document_count']:,} 份，"
        f"输出 {report['output_document_count']:,} 条，"
        f"精确重复 {report['duplicate_record_count']:,} 条"
    )
    print(
        f"  tokens: {report['output_tokens']:,}，"
        f"complete={report['complete']}"
    )


if __name__ == "__main__":
    main()
