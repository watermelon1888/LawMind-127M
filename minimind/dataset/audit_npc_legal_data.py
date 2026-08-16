"""审计国家法律法规数据库离线文档的正文与 token 规模。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from minimind.dataset.import_npc_legal_archives import (
    EXPECTED_DOCUMENTS,
    EXPECTED_DOCUMENT_COUNTS,
)
from minimind.dataset.prepare_tokenizer_corpus import extract_docx_paragraphs


EXPECTED_ARCHIVES = len(EXPECTED_DOCUMENT_COUNTS)


def extract_document_text(path: Path) -> str:
    """复用现有 OpenXML 解析器，并按原顺序合并非空正文段落。"""
    return "\n".join(extract_docx_paragraphs(path))


def classify_token_count(tokens: int) -> str:
    """按 CPT 最低和理想 token 规模返回唯一结论。"""
    if tokens < 500_000_000:
        return "<500M"
    if tokens < 1_000_000_000:
        return "500M-<1B"
    return ">=1B"


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"缺少 JSON 文件: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"JSON 文件无法解析: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON 文件不是对象: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise ValueError(f"缺少 JSONL 文件: {path}")
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"JSONL 第 {line_number} 行无法解析: {path}") from error
        if not isinstance(record, dict):
            raise ValueError(f"JSONL 第 {line_number} 行不是对象: {path}")
        records.append(record)
    return records


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_relative(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("文档清单包含缺失或无效的相对路径")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"文档清单路径越界: {value}") from error
    return path


def _conversion_records(path: Path) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for record in _read_jsonl(path):
        document_id = record.get("document_id")
        if not isinstance(document_id, str) or not document_id:
            raise ValueError("转换清单包含缺失或无效的文档 ID")
        if document_id in records:
            raise ValueError(f"转换清单包含重复文档 ID: {document_id}")
        records[document_id] = record
    return records


def _extraction_path(
    root: Path,
    inventory_record: dict[str, object],
    conversions: dict[str, dict[str, object]],
) -> Path:
    extension = inventory_record.get("original_extension")
    if extension != ".doc":
        if extension not in {".docx", ".docm"}:
            raise ValueError(f"不支持的文档格式: {extension}")
        return _resolve_relative(root, inventory_record.get("original_path"))

    document_id = str(inventory_record["document_id"])
    conversion = conversions.get(document_id)
    if conversion is None or conversion.get("status") not in {"converted", "skipped"}:
        raise ValueError("缺少成功的旧 DOC 转换记录")
    if conversion.get("original_sha256") != inventory_record.get("sha256"):
        raise ValueError("旧 DOC 转换记录的原文件哈希不一致")
    if conversion.get("converted_path") != inventory_record.get("converted_path"):
        raise ValueError("旧 DOC 转换记录的输出路径不一致")
    converted_path = _resolve_relative(root, conversion.get("converted_path"))
    expected_hash = conversion.get("converted_sha256")
    if not isinstance(expected_hash, str) or _sha256_file(converted_path) != expected_hash:
        raise ValueError("旧 DOC 转换结果哈希不一致")
    return converted_path


def load_tokenizer(path: Path):
    """加载项目当前 Tokenizer，并验证固定词表大小。"""
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("缺少 transformers，无法加载 Tokenizer") from error

    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    if len(tokenizer) != 12_000:
        raise ValueError(f"Tokenizer 词表大小应为 12000，实际为 {len(tokenizer)}")
    return tokenizer


def audit_dataset(
    input_root: Path,
    tokenizer: object,
    target_archives: int = EXPECTED_ARCHIVES,
    target_documents: int = EXPECTED_DOCUMENTS,
) -> dict[str, object]:
    """复核导入与转换清单，提取全部正文并统计总 tokens。"""
    if len(tokenizer) != 12_000:
        raise ValueError(f"Tokenizer 词表大小应为 12000，实际为 {len(tokenizer)}")

    input_root = input_root.resolve()
    manifest = _read_json(input_root / "manual-download-manifest.json")
    inventory = _read_jsonl(input_root / "document-inventory.jsonl")
    conversions = _conversion_records(input_root / "conversion-manifest.jsonl")

    identifiers = [record.get("document_id") for record in inventory]
    if any(not isinstance(document_id, str) or not document_id for document_id in identifiers):
        raise ValueError("文档清单包含缺失或无效的文档 ID")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("文档清单包含重复文档 ID")

    import_failures: list[str] = []
    extraction_failures: list[str] = []
    imported_documents = 0
    extracted_documents = 0
    characters = 0
    all_records_tokens = 0

    extracted_path = input_root / "extracted-text.jsonl"
    with extracted_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in inventory:
            document_id = str(record["document_id"])
            try:
                original_path = _resolve_relative(input_root, record.get("original_path"))
                expected_hash = record.get("sha256")
                if not isinstance(expected_hash, str) or _sha256_file(original_path) != expected_hash:
                    raise ValueError("原始文档哈希不一致")
            except (OSError, ValueError):
                import_failures.append(document_id)
                continue
            imported_documents += 1

            try:
                document_path = _extraction_path(input_root, record, conversions)
                text = extract_document_text(document_path)
                if not text:
                    raise ValueError("正文为空")
            except (KeyError, OSError, ValueError):
                extraction_failures.append(document_id)
                continue

            token_count = len(tokenizer.encode(text, add_special_tokens=False))
            extracted = {
                "document_id": document_id,
                "archive_order": record.get("archive_order"),
                "source_archive": record.get("source_archive"),
                "source_member_path": record.get("source_member_path"),
                "original_extension": record.get("original_extension"),
                "text": text,
            }
            output.write(json.dumps(extracted, ensure_ascii=False) + "\n")
            extracted_documents += 1
            characters += len(text)
            all_records_tokens += token_count

    archive_count = manifest.get("archive_count")
    manifest_documents = manifest.get("document_count")
    complete = (
        archive_count == target_archives
        and manifest_documents == target_documents
        and len(inventory) == target_documents
        and imported_documents == target_documents
        and extracted_documents == target_documents
        and not import_failures
        and not extraction_failures
    )
    report = {
        "target_archives": target_archives,
        "archive_count": archive_count,
        "target_documents": target_documents,
        "inventory_documents": len(inventory),
        "imported_documents": imported_documents,
        "import_failures": import_failures,
        "extracted_documents": extracted_documents,
        "extraction_failures": extraction_failures,
        "characters": characters,
        "all_records_tokens": all_records_tokens,
        "complete": complete,
        "size_conclusion": classify_token_count(all_records_tokens) if complete else None,
    }
    (input_root / "audit-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return report


def main() -> None:
    """解析命令行参数并生成法律数据量审计报告。"""
    parser = argparse.ArgumentParser(description="审计国家法律法规数据库离线文档数据量")
    parser.add_argument("--input-root", type=Path, required=True, help="离线导入输出根目录")
    parser.add_argument("--tokenizer-path", type=Path, required=True, help="当前 12000 词表 Tokenizer")
    args = parser.parse_args()

    report = audit_dataset(args.input_root, load_tokenizer(args.tokenizer_path))
    print(
        f"[完成] 导入 {report['imported_documents']:,}/{report['target_documents']:,}，"
        f"提取 {report['extracted_documents']:,}/{report['target_documents']:,}"
    )
    print(
        f"  tokens: {report['all_records_tokens']:,}，"
        f"结论: {report['size_conclusion']}，complete={report['complete']}"
    )


if __name__ == "__main__":
    main()
