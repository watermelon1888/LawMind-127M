"""准备一次性法律 tokenizer 训练语料。"""

from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path
from xml.etree import ElementTree


SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "tokenizer_data" / "raw"
DEFAULT_LEGAL_DIR = RAW_DIR / "legal"
DEFAULT_LAW_SFT_DIR = RAW_DIR / "law_sft"
DEFAULT_GENERAL_FILE = RAW_DIR / "general" / "pretrain_t2t_mini.jsonl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "tokenizer_data" / "prepared"
LAW_SFT_FILENAMES = (
    "DISC-Law-SFT-Pair-QA-released.jsonl",
    "DISC-Law-SFT-Triplet-QA-released.jsonl",
)
SEED = 42
DOCX_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def utf8_size(text: str) -> int:
    """返回文本编码为 UTF-8 后占用的字节数。"""
    return len(text.encode("utf-8"))


def iter_legal_docx(legal_dir: Path):
    """返回法律源目录中非隐藏位置的 DOCX 文件。"""
    for path in sorted(legal_dir.rglob("*.docx")):
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(legal_dir).parts):
            yield path


def extract_docx_paragraphs(path: Path) -> list[str]:
    """从 DOCX 正文提取全部非空段落。"""
    try:
        with zipfile.ZipFile(path) as archive:
            document_xml = archive.read("word/document.xml")
        root = ElementTree.fromstring(document_xml)
    except (KeyError, ElementTree.ParseError, zipfile.BadZipFile) as error:
        raise ValueError(f"无法读取 DOCX 文件: {path}") from error

    paragraphs = []
    for paragraph in root.iter(f"{DOCX_NAMESPACE}p"):
        text = "".join(node.text or "" for node in paragraph.iter(f"{DOCX_NAMESPACE}t"))
        text = " ".join(text.split())
        if text:
            paragraphs.append(text)
    return paragraphs


def read_legal_docx_paragraphs(legal_dir: Path) -> list[str]:
    """读取法律源中的全部非空正文段落。"""
    if not legal_dir.is_dir():
        raise ValueError(f"法律 DOCX 目录不存在: {legal_dir}")

    paragraphs: list[str] = []
    for path in iter_legal_docx(legal_dir):
        paragraphs.extend(extract_docx_paragraphs(path))
    if not paragraphs:
        raise ValueError(f"法律 DOCX 中没有有效段落: {legal_dir}")
    return paragraphs


def read_law_sft_texts(law_sft_dir: Path) -> tuple[list[str], int]:
    """读取指定法律问答文件中的完整 input/output 记录。"""
    if not law_sft_dir.is_dir():
        raise ValueError(f"法律 SFT 目录不存在: {law_sft_dir}")

    texts: list[str] = []
    skipped = 0
    for filename in LAW_SFT_FILENAMES:
        path = law_sft_dir / filename
        if not path.is_file():
            raise ValueError(f"缺少法律 SFT 文件: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            input_text = record.get("input") if isinstance(record, dict) else None
            output_text = record.get("output") if isinstance(record, dict) else None
            if not isinstance(input_text, str) or not isinstance(output_text, str):
                skipped += 1
                continue
            input_text = input_text.strip()
            output_text = output_text.strip()
            if not input_text or not output_text:
                skipped += 1
                continue
            texts.append(f"{input_text}\n{output_text}")
    if not texts:
        raise ValueError(f"法律 SFT 中没有有效问答: {law_sft_dir}")
    return texts, skipped


def read_general_texts(general_file: Path) -> tuple[list[str], int]:
    """读取通用中文 JSONL 中的 text 字段。"""
    if not general_file.is_file():
        raise ValueError(f"通用中文 JSONL 不存在: {general_file}")

    texts: list[str] = []
    skipped = 0
    for line in general_file.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        text = record.get("text") if isinstance(record, dict) else None
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            continue
        texts.append(text.strip())
    if not texts:
        raise ValueError(f"通用中文 JSONL 中没有有效 text 字段: {general_file}")
    return texts, skipped


def sample_to_budget(texts: list[str], budget: int, seed: int) -> list[str]:
    """以固定种子选择完整记录，直到达到字节预算。"""
    shuffled = list(texts)
    random.Random(seed).shuffle(shuffled)

    selected: list[str] = []
    selected_bytes = 0
    for text in shuffled:
        selected.append(text)
        selected_bytes += utf8_size(text)
        if selected_bytes >= budget:
            return selected
    raise ValueError(f"数据字节数不足，目标为 {budget} 字节，实际仅有 {selected_bytes} 字节")


def prepare_corpus(
    legal_dir: Path = DEFAULT_LEGAL_DIR,
    law_sft_dir: Path = DEFAULT_LAW_SFT_DIR,
    general_file: Path = DEFAULT_GENERAL_FILE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int = SEED,
) -> dict[str, object]:
    """生成 60/20/20 的一次性 tokenizer 训练语料。"""
    legal_texts = read_legal_docx_paragraphs(legal_dir)
    legal_bytes = sum(utf8_size(text) for text in legal_texts)
    budget = legal_bytes // 3
    law_sft_texts, law_sft_skipped = read_law_sft_texts(law_sft_dir)
    general_texts, general_skipped = read_general_texts(general_file)
    selected_law_sft = sample_to_budget(law_sft_texts, budget, seed)
    selected_general = sample_to_budget(general_texts, budget, seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "tokenizer_corpus.jsonl"
    with corpus_path.open("w", encoding="utf-8", newline="\n") as file:
        for text in [*legal_texts, *selected_law_sft, *selected_general]:
            file.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    categories = {
        "法律全文": legal_texts,
        "法律 SFT": selected_law_sft,
        "通用中文": selected_general,
    }
    total_bytes = sum(sum(utf8_size(text) for text in texts) for texts in categories.values())
    return {
        "corpus_path": corpus_path,
        "categories": categories,
        "skipped_records": law_sft_skipped + general_skipped,
        "total_bytes": total_bytes,
    }


def print_statistics(result: dict[str, object]) -> None:
    """打印三类语料的记录数、字节数和实际占比。"""
    categories = result["categories"]
    total_bytes = result["total_bytes"]
    for name, texts in categories.items():
        byte_count = sum(utf8_size(text) for text in texts)
        ratio = byte_count / total_bytes if total_bytes else 0
        print(f"{name}: {len(texts)} 条，{byte_count} 字节，占比 {ratio:.2%}")
    print(f"跳过记录: {result['skipped_records']} 条")
    print(f"训练语料: {result['corpus_path']}")


def main() -> None:
    """使用固定路径生成 tokenizer 训练语料。"""
    print_statistics(prepare_corpus())


if __name__ == "__main__":
    main()
