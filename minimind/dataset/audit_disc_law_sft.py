"""只读审计四个 DISC-Law-SFT 原始 JSONL 文件。"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
import re
import sqlite3
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


MINIMIND_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORK_ROOT = Path(os.environ.get("MINIMIND_WORK_ROOT", "/root/autodl-tmp/minimind-work"))
DEFAULT_INPUT_DIR = MINIMIND_ROOT / "dataset" / "tokenizer_data" / "raw" / "law_sft"
DEFAULT_TOKENIZER_PATH = MINIMIND_ROOT / "model"
DEFAULT_NPC_CORPUS = (
    DEFAULT_WORK_ROOT
    / "data"
    / "standardized"
    / "cpt"
    / "npc_effective_20260727"
    / "cpt-npc-flk.jsonl"
)
DEFAULT_OUTPUT_DIR = DEFAULT_WORK_ROOT / "reports" / "sft" / "disc_law_sft_audit"

EXPECTED_VOCAB_SIZE = 12_000
TOKENIZER_BATCH_SIZE = 256
TOKENIZER_CHARACTER_BUDGET = 1_000_000
PROGRESS_INTERVAL = 50_000

REPORT_FILENAME = "disc-law-sft-audit.json"
DUPLICATE_FILENAME = "disc-law-sft-duplicate-groups.jsonl"
ANSWER_VARIANT_FILENAME = "disc-law-sft-answer-variants.jsonl"
REVIEW_FILENAME = "disc-law-sft-review-candidates.jsonl"
HASH_FILENAME = "disc-law-sft-audit.sha256"

UPSTREAM_REPOSITORY = "ShengbinYue/DISC-Law-SFT"
UPSTREAM_REVISION = "fb12cf02809a85724f7e36977529b1d4b5f9f920"


@dataclass(frozen=True)
class FileSpec:
    """描述一个固定 DISC 原始文件的已验证结构。"""

    filename: str
    dataset_kind: str
    required_fields: tuple[str, ...]
    has_reference: bool
    is_qa: bool
    upstream_bytes: int
    upstream_sha256: str
    expected_task_prefixes: tuple[str, ...] = ()


FILE_SPECS = (
    FileSpec(
        filename="DISC-Law-SFT-Pair-QA-released.jsonl",
        dataset_kind="pair_qa",
        required_fields=("id", "input", "output"),
        has_reference=False,
        is_qa=True,
        upstream_bytes=94_042_837,
        upstream_sha256="0a12835997a2e8028b0c8073aecf840c93f0c5c294953471380275f43bb95280",
        expected_task_prefixes=("legal_question_answering",),
    ),
    FileSpec(
        filename="DISC-Law-SFT-Triplet-QA-released.jsonl",
        dataset_kind="triplet_qa",
        required_fields=("id", "input", "output", "reference"),
        has_reference=True,
        is_qa=True,
        upstream_bytes=85_520_456,
        upstream_sha256="df3d2d7c75ae07e5be78c6214805c47b896874e58b4fffabe97fd0470aa3ef79",
        expected_task_prefixes=("legal_question_answering",),
    ),
    FileSpec(
        filename="DISC-Law-SFT-Pair.jsonl",
        dataset_kind="pair",
        required_fields=("id", "input", "output"),
        has_reference=False,
        is_qa=False,
        upstream_bytes=346_801_238,
        upstream_sha256="d181a4baeca0b384a508dcba1e8c0888c57471ed1272e83c7e859c4464950aae",
    ),
    FileSpec(
        filename="DISC-Law-SFT-Triplet-released.jsonl",
        dataset_kind="triplet",
        required_fields=("id", "input", "output", "reference"),
        has_reference=True,
        is_qa=False,
        upstream_bytes=52_814_582,
        upstream_sha256="b368306f80a3f71e632298a8e7eda75a1bf029f8358dcb8af31fb912270890c2",
        expected_task_prefixes=("judgement_predit",),
    ),
)

TASK_TYPE_MAP = {
    "legal_question_answering": "legal_qa",
    "jud_doc_sum": "legal_document_summary",
    "jud_read_compre": "legal_reading_comprehension",
    "leg_case_cls": "legal_case_classification",
    "leg_ele_extra": "legal_information_extraction",
    "leg_eve_detec": "legal_event_detection",
    "exam": "judicial_exam",
    "sent_pred": "judgement_prediction",
    "sim_case_match": "legal_case_matching",
    "judgement_predit": "judgement_prediction",
}

CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
CHINESE_UNITS = {"十": 10, "百": 100, "千": 1000}

DUPLICATE_KINDS = (
    "id",
    "exact_input",
    "normalized_input",
    "exact_prompt",
    "normalized_prompt",
    "exact_full_record",
    "normalized_full_record",
    "qa_question",
)
ANSWER_VARIANT_SCOPES = ("id_payload", "task_prompt", "qa_question")

WHITESPACE_RE = re.compile(r"\s+")
TASK_SUFFIX_RE = re.compile(r"^(.*?)[_-]\d+$")
ARTICLE_MARKER_RE = re.compile(r"(?m)^\s*第\s*([〇零一二三四五六七八九十百千两\d]+)\s*条")
CITATION_RE = re.compile(
    r"《(?P<law>[^》\r\n]{1,80})》\s*第\s*"
    r"(?P<article>[〇零一二三四五六七八九十百千两\d]+)\s*条"
)
HTML_RE = re.compile(r"(?:</?[A-Za-z][^>\n]{0,200}>|/[A-Za-z][A-Za-z0-9]*>)", re.IGNORECASE)
URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?!\w)")
ID_CANDIDATE_RE = re.compile(r"(?<!\d)[1-9]\d{16}[\dXx](?!\d)")
MASKED_ID_RE = re.compile(r"(?<![\dXx*×])[1-9]\d{5}[\dXx*×]{9,13}(?![\dXx*×])")
MOBILE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
CONTACT_MOBILE_RE = re.compile(
    r"(?:手机|电话|联系电话|联系方式)[^\d\n]{0,8}(1[3-9]\d{9})"
)
DETAILED_ADDRESS_RE = re.compile(
    r"(?:"
    r"(?:住址|住所地|户籍地|现住址|居住地|(?:^|[，,：:])住)"
    r"[^\n。；]{0,100}(?:村|组|社|街道|路|巷|号|栋|幢|单元|室)"
    r"|"
    r"[\u4e00-\u9fff]{1,20}(?:路|街|巷|村|组|社)[ \t]*\d+"
    r"(?:号|组|社)[^\n。；]{0,30}(?:栋|幢|单元|室|门面)?"
    r")"
)
NATURAL_PARTY_RE = re.compile(
    r"(?m)^(?:原告|被告人?|上诉人|被上诉人|申请人|被申请人|"
    r"再审申请人|被害人|法定代表人|委托代理人)(?:（[^）\n]{0,30}）)?"
    r"[：:]?[ \t]*(?P<name>[\u4e00-\u9fff·]{2,8})"
)
GENERIC_ANSWER_RE = re.compile(
    r"建议.{0,20}咨询.{0,10}(?:律师|专业人士)|具体情况具体分析|"
    r"需(?:要)?结合具体情况|仅供参考"
)
FOLLOW_UP_ANSWER_RE = re.compile(r"(?:请问|能否|是否可以).{0,40}[？?]$|建议补充.{0,30}(?:材料|信息)")
REFUSAL_ANSWER_RE = re.compile(r"(?:无法|不能|不便).{0,12}(?:回答|判断|确定)|抱歉.{0,20}无法")

ID_WEIGHTS = (7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2)
ID_CHECK_CODES = "10X98765432"


class AuditError(RuntimeError):
    """表示审计无法继续的硬错误。"""


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    """返回稳定比率，空分母统一记为零。"""

    return numerator / denominator if denominator else 0.0


def normalize_whitespace(value: str) -> str:
    """仅执行 NFC、首尾清理和连续空白折叠。"""

    return WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", value)).strip()


def stable_signature(value: object) -> bytes:
    """对保留字段边界的 JSON 值生成 SHA-256。"""

    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).digest()


def sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_prefix(record_id: str) -> str:
    """从稳定 ID 中提取已知任务前缀，异常 ID 不回显原值。"""

    normalized = record_id.strip()
    if not normalized:
        return "<empty>"
    match = TASK_SUFFIX_RE.fullmatch(normalized)
    prefix = match.group(1) if match else normalized
    return prefix if prefix in TASK_TYPE_MAP else "<unknown>"


def is_valid_id_card(value: str) -> bool:
    """校验 18 位居民身份证号的日期和校验位。"""

    if not ID_CANDIDATE_RE.fullmatch(value):
        return False
    try:
        datetime.strptime(value[6:14], "%Y%m%d")
    except ValueError:
        return False
    checksum = sum(int(digit) * weight for digit, weight in zip(value[:17], ID_WEIGHTS))
    return ID_CHECK_CODES[checksum % 11] == value[-1].upper()


def has_unknown_control_character(text: str) -> bool:
    """检查除常规空白外的控制、私用和代理字符。"""

    return any(
        char not in "\n\r\t" and unicodedata.category(char) in {"Cc", "Cf", "Co", "Cs"}
        for char in text
    )


def matched_text_coverage_ratio(text: str, values: Sequence[str]) -> float:
    """计算多个精确匹配文本在目标字符串中的区间并集覆盖率。"""

    if not text:
        return 0.0
    intervals: list[tuple[int, int]] = []
    for value in set(values):
        if not value:
            continue
        start = 0
        while True:
            position = text.find(value, start)
            if position < 0:
                break
            intervals.append((position, position + len(value)))
            start = position + len(value)
    if not intervals:
        return 0.0
    intervals.sort()
    covered = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        covered += current_end - current_start
        current_start, current_end = start, end
    covered += current_end - current_start
    return safe_ratio(covered, len(text))


def normalize_article_number(value: str) -> str:
    """把常见中文或阿拉伯条号归一为十进制字符串。"""

    compact = normalize_whitespace(value).replace(" ", "")
    if compact.isdigit():
        number = int(compact)
        return str(number) if number > 0 else compact
    if any(char.isdigit() for char in compact):
        return compact
    total = 0
    current = 0
    for char in compact:
        if char in CHINESE_DIGITS:
            current = CHINESE_DIGITS[char]
            continue
        unit = CHINESE_UNITS.get(char)
        if unit is None:
            return compact
        total += (current or 1) * unit
        current = 0
    total += current
    return str(total) if total > 0 else compact


def extract_citations(text: str) -> list[tuple[str, str]]:
    """抽取显式的“《法名》第X条”引用。"""

    return [
        (
            normalize_whitespace(match.group("law")),
            normalize_article_number(match.group("article")),
        )
        for match in CITATION_RE.finditer(text)
    ]


def _histogram_quantile(histogram: Counter[int], quantile: float) -> int:
    total = sum(histogram.values())
    if not total:
        return 0
    target = max(1, math.ceil(total * quantile))
    seen = 0
    for value in sorted(histogram):
        seen += histogram[value]
        if seen >= target:
            return value
    return max(histogram)


def histogram_report(histogram: Counter[int], total_value: int) -> dict[str, int | float]:
    """从长度直方图生成固定分位数。"""

    count = sum(histogram.values())
    return {
        "count": count,
        "min": min(histogram, default=0),
        "p50": _histogram_quantile(histogram, 0.50),
        "p90": _histogram_quantile(histogram, 0.90),
        "p95": _histogram_quantile(histogram, 0.95),
        "p99": _histogram_quantile(histogram, 0.99),
        "max": max(histogram, default=0),
        "mean": round(safe_ratio(total_value, count), 4),
    }


@dataclass
class LengthStats:
    """保存一个逻辑字段的字符和 token 长度直方图。"""

    character_lengths: Counter[int] = field(default_factory=Counter)
    token_lengths: Counter[int] = field(default_factory=Counter)
    total_characters: int = 0
    total_tokens: int = 0

    def record_characters(self, text: str) -> None:
        length = len(text)
        self.character_lengths[length] += 1
        self.total_characters += length

    def record_tokens(self, length: int) -> None:
        self.token_lengths[length] += 1
        self.total_tokens += length

    def to_report(self) -> dict[str, object]:
        return {
            "characters": histogram_report(self.character_lengths, self.total_characters),
            "tokens": histogram_report(self.token_lengths, self.total_tokens),
            "tokens_per_character": round(
                safe_ratio(self.total_tokens, self.total_characters), 6
            ),
        }


class TokenBatcher:
    """按条数和字符预算批量调用当前 Tokenizer。"""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.pending: list[tuple[str, tuple[LengthStats, ...]]] = []
        self.pending_characters = 0

    def add(self, text: str, *targets: LengthStats) -> None:
        for target in targets:
            target.record_characters(text)
        exceeds_budget = (
            self.pending
            and self.pending_characters + len(text) > TOKENIZER_CHARACTER_BUDGET
        )
        if self.pending and (len(self.pending) >= TOKENIZER_BATCH_SIZE or exceeds_budget):
            self.flush()
        self.pending.append((text, targets))
        self.pending_characters += len(text)
        if len(text) >= TOKENIZER_CHARACTER_BUDGET:
            self.flush()

    def flush(self) -> None:
        if not self.pending:
            return
        texts = [item[0] for item in self.pending]
        try:
            encoded = self.tokenizer(
                texts,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"]
        except Exception as error:
            raise AuditError("批量计算 SFT 字段 token 长度失败") from error
        if len(encoded) != len(self.pending):
            raise AuditError("Tokenizer 返回数量与待审计字段数量不一致")
        for token_ids, (_, targets) in zip(encoded, self.pending):
            for target in targets:
                target.record_tokens(len(token_ids))
        self.pending.clear()
        self.pending_characters = 0


class ReviewSampler:
    """为每种风险保留确定性、无原文的小型复核样本。"""

    def __init__(self, capacity: int, seed: int):
        if capacity <= 0:
            raise ValueError("每种风险的复核样本数必须大于 0")
        self.capacity = capacity
        self.seed = seed
        self.heaps: dict[str, list[tuple[int, str, dict[str, object]]]] = defaultdict(list)
        self.seen: set[tuple[str, str]] = set()

    def add(
        self,
        signal: str,
        source_file: str,
        line_number: int,
        record_id: str,
        task: str,
        field_name: str,
        _value: str | None = None,
    ) -> None:
        record_id_hash = hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:16]
        locator = f"{source_file}:{line_number}:{record_id_hash}:{field_name}"
        candidate_key = (signal, locator)
        if candidate_key in self.seen:
            return
        self.seen.add(candidate_key)
        score = int.from_bytes(
            hashlib.sha256(f"{self.seed}\0{signal}\0{locator}".encode("utf-8")).digest()[:8],
            "big",
        )
        item: dict[str, object] = {
            "signal": signal,
            "source_file": source_file,
            "line_number": line_number,
            "record_id_sha256_prefix": record_id_hash,
            "task_prefix": task,
            "field": field_name,
        }
        entry = (-score, locator, item)
        heap = self.heaps[signal]
        if len(heap) < self.capacity:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)

    def records(self) -> list[dict[str, object]]:
        output: list[dict[str, object]] = []
        for signal in sorted(self.heaps):
            entries = sorted(self.heaps[signal], key=lambda item: (-item[0], item[1]))
            output.extend(entry[2] for entry in entries)
        return output


class FingerprintStore:
    """使用临时 SQLite 保存全量去重和答案变体指纹。"""

    def __init__(self, path: Path, file_names: Sequence[str]):
        self.file_names = list(file_names)
        self.connection = sqlite3.connect(path)
        try:
            self.connection.execute("PRAGMA journal_mode=OFF")
            self.connection.execute("PRAGMA synchronous=OFF")
            self.connection.execute("PRAGMA temp_store=FILE")
            self.connection.executescript(
                """
                CREATE TABLE fingerprints (
                    kind TEXT NOT NULL,
                    signature BLOB NOT NULL,
                    file_index INTEGER NOT NULL,
                    count INTEGER NOT NULL,
                    first_line INTEGER NOT NULL,
                    first_id_hash TEXT NOT NULL,
                    PRIMARY KEY (kind, signature, file_index)
                ) WITHOUT ROWID;
                CREATE TABLE answer_variants (
                    scope TEXT NOT NULL,
                    prompt_signature BLOB NOT NULL,
                    answer_signature BLOB NOT NULL,
                    file_index INTEGER NOT NULL,
                    count INTEGER NOT NULL,
                    first_line INTEGER NOT NULL,
                    first_id_hash TEXT NOT NULL,
                    PRIMARY KEY (scope, prompt_signature, answer_signature, file_index)
                ) WITHOUT ROWID;
                """
            )
        except BaseException:
            self.connection.close()
            raise
        self.fingerprint_buffer: list[tuple[object, ...]] = []
        self.variant_buffer: list[tuple[object, ...]] = []

    def add_fingerprint(
        self,
        kind: str,
        signature: bytes,
        file_index: int,
        line_number: int,
        id_hash: str,
    ) -> None:
        self.fingerprint_buffer.append((kind, signature, file_index, 1, line_number, id_hash))
        if len(self.fingerprint_buffer) >= 10_000:
            self.flush()

    def add_variant(
        self,
        scope: str,
        prompt_signature: bytes,
        answer_signature: bytes,
        file_index: int,
        line_number: int,
        id_hash: str,
    ) -> None:
        self.variant_buffer.append(
            (scope, prompt_signature, answer_signature, file_index, 1, line_number, id_hash)
        )
        if len(self.variant_buffer) >= 10_000:
            self.flush()

    def flush(self) -> None:
        if self.fingerprint_buffer:
            self.connection.executemany(
                """
                INSERT INTO fingerprints
                    (kind, signature, file_index, count, first_line, first_id_hash)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(kind, signature, file_index)
                DO UPDATE SET count = count + 1
                """,
                self.fingerprint_buffer,
            )
            self.fingerprint_buffer.clear()
        if self.variant_buffer:
            self.connection.executemany(
                """
                INSERT INTO answer_variants
                    (scope, prompt_signature, answer_signature, file_index,
                     count, first_line, first_id_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope, prompt_signature, answer_signature, file_index)
                DO UPDATE SET count = count + 1
                """,
                self.variant_buffer,
            )
            self.variant_buffer.clear()
        self.connection.commit()

    def duplicate_summary(self) -> dict[str, object]:
        self.flush()
        within: dict[str, dict[str, object]] = {}
        cross: dict[str, dict[str, int]] = {}
        for kind in DUPLICATE_KINDS:
            per_file = {
                name: {
                    "unique_keys": 0,
                    "duplicate_groups": 0,
                    "duplicate_records": 0,
                    "max_group_size": 0,
                }
                for name in self.file_names
            }
            rows = self.connection.execute(
                """
                SELECT file_index,
                       COUNT(*) AS unique_keys,
                       SUM(CASE WHEN count > 1 THEN 1 ELSE 0 END) AS duplicate_groups,
                       SUM(CASE WHEN count > 1 THEN count - 1 ELSE 0 END) AS duplicate_records,
                       MAX(count) AS max_group_size
                FROM fingerprints
                WHERE kind = ?
                GROUP BY file_index
                """,
                (kind,),
            )
            for file_index, unique_keys, groups, records, maximum in rows:
                per_file[self.file_names[file_index]] = {
                    "unique_keys": unique_keys or 0,
                    "duplicate_groups": groups or 0,
                    "duplicate_records": records or 0,
                    "max_group_size": maximum or 0,
                }
            within[kind] = per_file

            pair_counts: dict[str, int] = {}
            rows = self.connection.execute(
                """
                SELECT left_side.file_index, right_side.file_index, COUNT(*)
                FROM fingerprints AS left_side
                JOIN fingerprints AS right_side
                  ON left_side.kind = right_side.kind
                 AND left_side.signature = right_side.signature
                 AND left_side.file_index < right_side.file_index
                WHERE left_side.kind = ?
                GROUP BY left_side.file_index, right_side.file_index
                """,
                (kind,),
            )
            for left_index, right_index, count in rows:
                key = f"{self.file_names[left_index]}__{self.file_names[right_index]}"
                pair_counts[key] = count
            cross[kind] = pair_counts
        return {"within_files": within, "cross_file_shared_signature_groups": cross}

    def answer_variant_summary(self) -> dict[str, object]:
        self.flush()
        report: dict[str, object] = {}
        for scope in ANSWER_VARIANT_SCOPES:
            row = self.connection.execute(
                """
                SELECT COUNT(*), COALESCE(SUM(total_records), 0), COALESCE(MAX(variants), 0)
                FROM (
                    SELECT prompt_signature,
                           COUNT(DISTINCT answer_signature) AS variants,
                           SUM(count) AS total_records
                    FROM answer_variants
                    WHERE scope = ?
                    GROUP BY prompt_signature
                    HAVING COUNT(DISTINCT answer_signature) > 1
                )
                """,
                (scope,),
            ).fetchone()
            report[scope] = {
                "variant_groups": row[0] or 0,
                "records_in_variant_groups": row[1] or 0,
                "max_variants": row[2] or 0,
            }
        return report

    def write_duplicate_groups(self, path: Path) -> int:
        self.flush()
        query = """
            SELECT item.kind, hex(item.signature), item.file_index, item.count,
                   item.first_line, item.first_id_hash
            FROM fingerprints AS item
            JOIN (
                SELECT kind, signature
                FROM fingerprints
                GROUP BY kind, signature
                HAVING SUM(count) > 1
            ) AS duplicate
              ON item.kind = duplicate.kind AND item.signature = duplicate.signature
            ORDER BY item.kind, item.signature, item.file_index
        """
        return self._write_grouped_rows(path, query, ("kind", "signature_sha256"))

    def write_answer_variants(self, path: Path) -> int:
        self.flush()
        query = """
            SELECT item.scope, hex(item.prompt_signature), hex(item.answer_signature),
                   item.file_index, item.count, item.first_line, item.first_id_hash
            FROM answer_variants AS item
            JOIN (
                SELECT scope, prompt_signature
                FROM answer_variants
                GROUP BY scope, prompt_signature
                HAVING COUNT(DISTINCT answer_signature) > 1
            ) AS variant
              ON item.scope = variant.scope
             AND item.prompt_signature = variant.prompt_signature
            ORDER BY item.scope, item.prompt_signature, item.answer_signature, item.file_index
        """
        temporary = path.with_suffix(path.suffix + ".tmp")
        count = 0
        current_key: tuple[str, str] | None = None
        current: dict[str, object] | None = None
        current_variant_hash: str | None = None
        current_variant: dict[str, object] | None = None
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for scope, prompt_hash, answer_hash, file_index, records, line, id_hash in self.connection.execute(query):
                key = (scope, prompt_hash.lower())
                if key != current_key:
                    if current is not None:
                        output.write(json.dumps(current, ensure_ascii=False) + "\n")
                        count += 1
                    current_key = key
                    current = {
                        "scope": scope,
                        "prompt_signature_sha256": prompt_hash.lower(),
                        "variants": [],
                    }
                    current_variant_hash = None
                    current_variant = None
                assert current is not None
                normalized_answer_hash = answer_hash.lower()
                if normalized_answer_hash != current_variant_hash:
                    current_variant_hash = normalized_answer_hash
                    current_variant = {
                        "variant_signature_sha256": normalized_answer_hash,
                        "members": [],
                    }
                    current["variants"].append(current_variant)
                assert current_variant is not None
                current_variant["members"].append(
                    {
                        "source_file": self.file_names[file_index],
                        "records": records,
                        "first_line": line,
                        "first_id_sha256_prefix": id_hash,
                    }
                )
            if current is not None:
                output.write(json.dumps(current, ensure_ascii=False) + "\n")
                count += 1
        temporary.replace(path)
        return count

    def _write_grouped_rows(self, path: Path, query: str, key_names: tuple[str, str]) -> int:
        temporary = path.with_suffix(path.suffix + ".tmp")
        count = 0
        current_key: tuple[str, str] | None = None
        current: dict[str, object] | None = None
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            for kind, signature, file_index, records, line, id_hash in self.connection.execute(query):
                key = (kind, signature.lower())
                if key != current_key:
                    if current is not None:
                        output.write(json.dumps(current, ensure_ascii=False) + "\n")
                        count += 1
                    current_key = key
                    current = {key_names[0]: kind, key_names[1]: signature.lower(), "members": []}
                assert current is not None
                current["members"].append(
                    {
                        "source_file": self.file_names[file_index],
                        "records": records,
                        "first_line": line,
                        "first_id_sha256_prefix": id_hash,
                    }
                )
            if current is not None:
                output.write(json.dumps(current, ensure_ascii=False) + "\n")
                count += 1
        temporary.replace(path)
        return count

    def close(self, commit: bool = True) -> None:
        try:
            if commit:
                self.flush()
            else:
                self.fingerprint_buffer.clear()
                self.variant_buffer.clear()
                self.connection.rollback()
        finally:
            self.connection.close()


@dataclass
class NpcIndex:
    """保存 NPC 现行法规标题别名和条号集合。"""

    articles_by_title: dict[str, set[str]]
    report: dict[str, object]

    def locate(self, law_name: str, article: str) -> str:
        normalized_law = normalize_whitespace(law_name).replace(" ", "")
        normalized_article = normalize_article_number(article)
        articles = self.articles_by_title.get(normalized_law)
        if articles is None:
            return "law_not_found_in_snapshot"
        if normalized_article not in articles:
            return "article_not_found_in_snapshot"
        return "article_found_in_snapshot"


def _title_aliases(title: str) -> set[str]:
    normalized = normalize_whitespace(title).replace(" ", "")
    aliases = {normalized}
    if normalized.startswith("中华人民共和国"):
        aliases.add(normalized[len("中华人民共和国") :])
    without_version = re.sub(r"（[^）]{1,40}）$", "", normalized)
    aliases.add(without_version)
    if without_version.startswith("中华人民共和国"):
        aliases.add(without_version[len("中华人民共和国") :])
    return {alias for alias in aliases if alias}


def load_npc_index(path: Path | None) -> NpcIndex | None:
    """加载可选 NPC 语料；缺失时由主报告明确记录未执行。"""

    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    articles_by_title: dict[str, set[str]] = defaultdict(set)
    counts = Counter(physical_lines=0, parsed_records=0, valid_records=0, invalid_records=0)
    with path.open("rb") as source:
        for raw_line in source:
            digest.update(raw_line)
            counts["physical_lines"] += 1
            try:
                record = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                counts["invalid_records"] += 1
                continue
            counts["parsed_records"] += 1
            if not isinstance(record, dict):
                counts["invalid_records"] += 1
                continue
            title = record.get("title")
            text = record.get("text")
            if not isinstance(title, str) or not title.strip() or not isinstance(text, str):
                counts["invalid_records"] += 1
                continue
            counts["valid_records"] += 1
            articles = {
                normalize_article_number(match.group(1))
                for match in ARTICLE_MARKER_RE.finditer(text)
            }
            for alias in _title_aliases(title):
                articles_by_title[alias].update(articles)
    report = {
        "status": "loaded",
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
        **dict(counts),
        "title_aliases": len(articles_by_title),
        "title_alias_article_pairs": sum(
            len(articles) for articles in articles_by_title.values()
        ),
    }
    return NpcIndex(dict(articles_by_title), report)


@dataclass
class FileStats:
    """保存单个 DISC 文件的聚合审计状态。"""

    spec: FileSpec
    path: Path
    counts: Counter[str] = field(default_factory=Counter)
    schema_counts: Counter[str] = field(default_factory=Counter)
    missing_fields: Counter[str] = field(default_factory=Counter)
    unexpected_fields: Counter[str] = field(default_factory=Counter)
    task_prefix_counts: Counter[str] = field(default_factory=Counter)
    task_type_counts: Counter[str] = field(default_factory=Counter)
    signal_counts: Counter[str] = field(default_factory=Counter)
    signal_field_counts: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    citation_counts: Counter[str] = field(default_factory=Counter)
    npc_status_counts: Counter[str] = field(default_factory=Counter)
    reference_counts: Counter[str] = field(default_factory=Counter)
    reference_list_lengths: Counter[int] = field(default_factory=Counter)
    reference_coverage_permyriad: Counter[int] = field(default_factory=Counter)
    lengths: dict[str, LengthStats] = field(default_factory=lambda: defaultdict(LengthStats))
    first_errors: list[dict[str, object]] = field(default_factory=list)
    input_bytes: int = 0
    input_sha256: str = ""

    def record_error(self, line_number: int, reason: str) -> None:
        if len(self.first_errors) < 20:
            self.first_errors.append({"line_number": line_number, "reason": reason})

    def record_signals(self, field_name: str, signals: dict[str, str | None]) -> set[str]:
        names = set(signals)
        for signal in names:
            self.signal_field_counts[field_name][signal] += 1
        return names

    def to_report(self, duplicate_stats: dict[str, object]) -> dict[str, object]:
        upstream = {
            "repository": UPSTREAM_REPOSITORY,
            "revision": UPSTREAM_REVISION,
            "expected_bytes": self.spec.upstream_bytes,
            "expected_sha256": self.spec.upstream_sha256,
            "matches_expected_bytes": self.input_bytes == self.spec.upstream_bytes,
            "matches_expected_sha256": self.input_sha256 == self.spec.upstream_sha256,
        }
        coverage_record_count = sum(self.reference_coverage_permyriad.values())
        coverage_total = sum(
            value * count for value, count in self.reference_coverage_permyriad.items()
        )
        return {
            "dataset_kind": self.spec.dataset_kind,
            "path": str(self.path),
            "input_bytes": self.input_bytes,
            "input_sha256": self.input_sha256,
            "upstream_snapshot": upstream,
            "records": {
                "physical_lines": self.counts["physical_lines"],
                "blank_lines": self.counts["blank_lines"],
                "utf8_errors": self.counts["utf8_errors"],
                "json_errors": self.counts["json_errors"],
                "json_values_parsed": self.counts["json_values_parsed"],
                "non_object_records": self.counts["non_object_records"],
                "parsed_objects": self.counts["parsed_objects"],
                "schema_anomaly_records": self.counts["schema_anomaly_records"],
                "core_valid_records": self.counts["core_valid_records"],
                "empty_id_records": self.counts["empty_id_records"],
                "empty_input_records": self.counts["empty_input_records"],
                "empty_output_records": self.counts["empty_output_records"],
                "json_parse_rate_nonblank": round(
                    safe_ratio(
                        self.counts["json_values_parsed"],
                        self.counts["physical_lines"] - self.counts["blank_lines"],
                    ),
                    8,
                ),
                "object_record_rate_nonblank": round(
                    safe_ratio(
                        self.counts["parsed_objects"],
                        self.counts["physical_lines"] - self.counts["blank_lines"],
                    ),
                    8,
                ),
                "first_errors": self.first_errors,
            },
            "schemas": dict(sorted(self.schema_counts.items())),
            "missing_fields": dict(sorted(self.missing_fields.items())),
            "unexpected_fields": dict(sorted(self.unexpected_fields.items())),
            "task_prefix_counts": dict(self.task_prefix_counts.most_common()),
            "task_type_counts": dict(self.task_type_counts.most_common()),
            "lengths": {name: self.lengths[name].to_report() for name in sorted(self.lengths)},
            "quality_signal_records": dict(sorted(self.signal_counts.items())),
            "quality_signal_records_by_field": {
                field_name: dict(sorted(counter.items()))
                for field_name, counter in sorted(self.signal_field_counts.items())
            },
            "citations": {
                **dict(sorted(self.citation_counts.items())),
                "npc_output_unique_citation_pair_checks_by_status": dict(
                    sorted(self.npc_status_counts.items())
                ),
            },
            "references": {
                **dict(sorted(self.reference_counts.items())),
                "list_length": histogram_report(
                    self.reference_list_lengths,
                    sum(length * count for length, count in self.reference_list_lengths.items()),
                ),
                "matched_reference_coverage_records": coverage_record_count,
                "mean_matched_reference_character_coverage_ratio_in_input": (
                    round(safe_ratio(coverage_total, coverage_record_count) / 10_000, 6)
                    if coverage_record_count
                    else None
                ),
            },
            "duplicates": duplicate_stats,
        }


def _field_signals(text: str, special_tokens: tuple[str, ...]) -> dict[str, str | None]:
    signals: dict[str, str | None] = {}
    valid_ids = [value for value in ID_CANDIDATE_RE.findall(text) if is_valid_id_card(value)]
    if valid_ids:
        signals["valid_id_number"] = valid_ids[0]
    masked_id = next(
        (
            match
            for match in MASKED_ID_RE.finditer(text)
            if any(char in "*×" for char in match.group(0))
            or any(char in "Xx" for char in match.group(0)[6:-1])
        ),
        None,
    )
    if masked_id:
        signals["masked_id_number"] = masked_id.group(0)
    contact_mobile = CONTACT_MOBILE_RE.search(text)
    mobile = MOBILE_RE.search(text)
    if contact_mobile:
        signals["contact_mobile_number"] = contact_mobile.group(1)
    elif mobile:
        signals["mobile_number_candidate"] = mobile.group(0)
    email = EMAIL_RE.search(text)
    if email:
        signals["email_address"] = email.group(0)
    address = DETAILED_ADDRESS_RE.search(text)
    if address:
        signals["detailed_address_candidate"] = address.group(0)
    party = NATURAL_PARTY_RE.search(text)
    if party and not any(marker in party.group("name") for marker in ("某", "×", "*", "＊")):
        signals["unmasked_natural_party_candidate"] = party.group("name")
    if "\ufffd" in text:
        signals["replacement_character"] = None
    if "\x00" in text:
        signals["nul_character"] = None
    if has_unknown_control_character(text):
        signals["unknown_control_character"] = None
    if HTML_RE.search(text):
        signals["html_fragment"] = None
    if URL_RE.search(text):
        signals["url"] = None
    if any(token in text for token in special_tokens):
        signals["reserved_special_token"] = None
    return signals


def _answer_signals(text: str) -> dict[str, str | None]:
    signals: dict[str, str | None] = {}
    if len(text.strip()) < 20:
        signals["answer_under_20_characters"] = None
    if GENERIC_ANSWER_RE.search(text):
        signals["generic_answer_candidate"] = None
    if FOLLOW_UP_ANSWER_RE.search(text.strip()):
        signals["follow_up_answer_candidate"] = None
    if REFUSAL_ANSWER_RE.search(text):
        signals["refusal_answer_candidate"] = None
    return signals


def _record_review_signals(
    stats: FileStats,
    sampler: ReviewSampler,
    line_number: int,
    record_id: str,
    task: str,
    field_name: str,
    signals: dict[str, str | None],
) -> set[str]:
    names = stats.record_signals(field_name, signals)
    for signal, value in signals.items():
        sampler.add(
            signal,
            stats.spec.filename,
            line_number,
            record_id,
            task,
            field_name,
            value,
        )
    return names


def _validate_record_schema(
    stats: FileStats,
    record: dict[str, object],
    line_number: int,
) -> tuple[str, str, str, list[str]] | None:
    actual_fields = set(record)
    expected_fields = set(stats.spec.required_fields)
    stats.schema_counts[",".join(sorted(actual_fields))] += 1
    missing = sorted(expected_fields - actual_fields)
    extra = sorted(actual_fields - expected_fields)
    schema_anomaly = bool(missing or extra)
    if missing or extra:
        stats.missing_fields.update(missing)
        stats.unexpected_fields.update(extra)
        stats.record_error(line_number, f"字段集合异常，缺失={missing}，额外={extra}")

    invalid = False
    for field_name in ("id", "input", "output"):
        if not isinstance(record.get(field_name), str):
            schema_anomaly = True
            invalid = True
            stats.record_error(line_number, f"{field_name} 缺失或不是字符串")

    references: list[str] = []
    if stats.spec.has_reference:
        value = record.get("reference")
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            schema_anomaly = True
            invalid = True
            stats.record_error(line_number, "reference 必须是字符串列表")
        else:
            references = list(value)

    if schema_anomaly:
        stats.counts["schema_anomaly_records"] += 1
    if invalid:
        return None

    stats.counts["core_valid_records"] += 1
    return str(record["id"]), str(record["input"]), str(record["output"]), references


def _record_reference_metrics(
    stats: FileStats,
    sampler: ReviewSampler,
    line_number: int,
    record_id: str,
    task: str,
    input_text: str,
    output_text: str,
    references: list[str],
) -> str | None:
    stats.reference_counts["records_with_reference_field"] += 1
    stats.reference_list_lengths[len(references)] += 1
    if not references:
        stats.reference_counts["empty_reference_lists"] += 1
        sampler.add(
            "empty_reference_list",
            stats.spec.filename,
            line_number,
            record_id,
            task,
            "reference",
        )
    empty_items = sum(not item.strip() for item in references)
    if empty_items:
        stats.reference_counts["records_with_empty_reference_items"] += 1
        stats.reference_counts["empty_reference_items"] += empty_items
        sampler.add(
            "empty_reference_item",
            stats.spec.filename,
            line_number,
            record_id,
            task,
            "reference",
        )

    exact_extras = len(references) - len(set(references))
    normalized_references = [normalize_whitespace(item) for item in references]
    normalized_extras = len(normalized_references) - len(set(normalized_references))
    if exact_extras:
        stats.reference_counts["records_with_exact_duplicate_items"] += 1
        stats.reference_counts["exact_duplicate_extra_items"] += exact_extras
        sampler.add(
            "exact_duplicate_reference_items",
            stats.spec.filename,
            line_number,
            record_id,
            task,
            "reference",
        )
    if normalized_extras:
        stats.reference_counts["records_with_normalized_duplicate_items"] += 1
        stats.reference_counts["normalized_duplicate_extra_items"] += normalized_extras
        sampler.add(
            "normalized_duplicate_reference_items",
            stats.spec.filename,
            line_number,
            record_id,
            task,
            "reference",
        )

    question: str | None = None
    if stats.spec.dataset_kind == "triplet_qa":
        marker_count = input_text.count("<问题>")
        stats.reference_counts[f"question_marker_count_{marker_count}"] += 1
        if marker_count == 1:
            prefix, question = input_text.split("<问题>", 1)
            if question.strip():
                stats.reference_counts["nonempty_questions_after_marker"] += 1
            else:
                stats.reference_counts["empty_questions_after_marker"] += 1
                sampler.add(
                    "empty_question_after_marker",
                    stats.spec.filename,
                    line_number,
                    record_id,
                    task,
                    "input",
                )
        else:
            prefix = input_text
            sampler.add(
                "question_marker_anomaly",
                stats.spec.filename,
                line_number,
                record_id,
                task,
                "input",
            )

        positions = [prefix.find(reference) for reference in references]
        present = [position >= 0 for position in positions]
        if references and all(present):
            stats.reference_counts["all_reference_items_found_in_input"] += 1
            if positions == sorted(positions):
                stats.reference_counts["reference_order_matches_input"] += 1
            else:
                stats.reference_counts["reference_order_differs_from_input"] += 1
        elif any(present):
            stats.reference_counts["partial_reference_items_found_in_input"] += 1
        elif references:
            stats.reference_counts["no_reference_items_found_in_input"] += 1

        repeated_occurrences = sum(max(0, input_text.count(reference) - 1) for reference in references if reference)
        if repeated_occurrences:
            stats.reference_counts["records_with_repeated_reference_text_in_input"] += 1
            stats.reference_counts["repeated_reference_text_occurrences"] += repeated_occurrences
            sampler.add(
                "repeated_reference_text_in_input",
                stats.spec.filename,
                line_number,
                record_id,
                task,
                "input",
            )
        coverage = round(matched_text_coverage_ratio(prefix, references) * 10_000)
        stats.reference_coverage_permyriad[coverage] += 1
    else:
        found = sum(bool(reference and reference in input_text) for reference in references)
        if found:
            stats.reference_counts["records_with_reference_text_already_in_input"] += 1
            stats.reference_counts["reference_items_already_in_input"] += found

    exact_output_overlap = sum(bool(reference and reference in output_text) for reference in references)
    if exact_output_overlap:
        stats.reference_counts["records_with_full_reference_in_output"] += 1
        stats.reference_counts["full_reference_items_in_output"] += exact_output_overlap
    return question


def _record_citation_metrics(
    stats: FileStats,
    sampler: ReviewSampler,
    npc_index: NpcIndex | None,
    line_number: int,
    record_id: str,
    task: str,
    input_text: str,
    output_text: str,
    references: list[str],
) -> None:
    input_citations = extract_citations(input_text)
    output_citations = extract_citations(output_text)
    reference_citations = extract_citations("\n".join(references))
    for field_name, citations in (
        ("input", input_citations),
        ("output", output_citations),
        ("reference", reference_citations),
    ):
        if citations:
            stats.citation_counts[f"{field_name}_records_with_citations"] += 1
            stats.citation_counts[f"{field_name}_citation_occurrences"] += len(citations)

    output_set = set(output_citations)
    input_set = set(input_citations)
    reference_set = set(reference_citations)
    if not output_set:
        stats.citation_counts["output_records_without_explicit_citations"] += 1
        return

    input_matches = len(output_set & input_set)
    if input_matches == len(output_set):
        stats.citation_counts["output_citations_all_found_in_input"] += 1
    elif input_matches:
        stats.citation_counts["output_citations_partly_found_in_input"] += 1
    else:
        stats.citation_counts["output_citations_none_found_in_input"] += 1

    if references:
        reference_matches = len(output_set & reference_set)
        if reference_matches == len(output_set):
            stats.citation_counts["output_citations_all_found_in_reference"] += 1
        elif reference_matches:
            stats.citation_counts["output_citations_partly_found_in_reference"] += 1
        else:
            stats.citation_counts["output_citations_none_found_in_reference"] += 1
            sampler.add(
                "output_citation_not_found_in_reference",
                stats.spec.filename,
                line_number,
                record_id,
                task,
                "output",
                json.dumps(sorted(output_set), ensure_ascii=False),
            )

    if npc_index is not None:
        for law_name, article in output_set:
            status = npc_index.locate(law_name, article)
            stats.npc_status_counts[status] += 1
            if status != "article_found_in_snapshot":
                sampler.add(
                    status,
                    stats.spec.filename,
                    line_number,
                    record_id,
                    task,
                    "output",
                    f"{law_name}\0{article}",
                )


def _add_fingerprints(
    store: FingerprintStore,
    file_index: int,
    line_number: int,
    record_id: str,
    task: str,
    input_text: str,
    output_text: str,
    references: list[str],
    question: str | None,
) -> None:
    id_hash = hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:16]
    if record_id.strip():
        id_signature = stable_signature(record_id)
        store.add_fingerprint("id", id_signature, file_index, line_number, id_hash)
    if not input_text.strip():
        return

    normalized_input = normalize_whitespace(input_text)
    normalized_output = normalize_whitespace(output_text)
    normalized_references = [normalize_whitespace(item) for item in references]
    exact_prompt = [references, input_text]
    normalized_prompt = [normalized_references, normalized_input]
    exact_full = [references, input_text, output_text]
    normalized_full = [normalized_references, normalized_input, normalized_output]

    fingerprints = {
        "exact_input": stable_signature(input_text),
        "normalized_input": stable_signature(normalized_input),
        "exact_prompt": stable_signature(exact_prompt),
        "normalized_prompt": stable_signature(normalized_prompt),
        "exact_full_record": stable_signature(exact_full),
        "normalized_full_record": stable_signature(normalized_full),
    }
    if question is not None and question.strip():
        fingerprints["qa_question"] = stable_signature(normalize_whitespace(question))
    for kind, signature in fingerprints.items():
        store.add_fingerprint(kind, signature, file_index, line_number, id_hash)

    exact_full_signature = fingerprints["exact_full_record"]
    normalized_output_signature = stable_signature(normalized_output)
    if record_id.strip():
        store.add_variant(
            "id_payload",
            stable_signature(record_id),
            exact_full_signature,
            file_index,
            line_number,
            id_hash,
        )
    store.add_variant(
        "task_prompt",
        stable_signature([task, normalized_prompt]),
        normalized_output_signature,
        file_index,
        line_number,
        id_hash,
    )
    if question is not None and question.strip():
        store.add_variant(
            "qa_question",
            stable_signature(normalize_whitespace(question)),
            normalized_output_signature,
            file_index,
            line_number,
            id_hash,
        )


def _audit_valid_record(
    stats: FileStats,
    file_index: int,
    line_number: int,
    values: tuple[str, str, str, list[str]],
    tokenizer_batcher: TokenBatcher,
    store: FingerprintStore,
    sampler: ReviewSampler,
    npc_index: NpcIndex | None,
    special_tokens: tuple[str, ...],
) -> None:
    record_id, input_text, output_text, references = values
    task = task_prefix(record_id)
    stats.task_prefix_counts[task] += 1
    mapped_task = TASK_TYPE_MAP.get(task, f"unmapped:{task}")
    stats.task_type_counts[mapped_task] += 1
    if stats.spec.expected_task_prefixes and task not in stats.spec.expected_task_prefixes:
        stats.signal_counts["unexpected_task_prefix"] += 1
        sampler.add(
            "unexpected_task_prefix",
            stats.spec.filename,
            line_number,
            record_id,
            task,
            "id",
        )

    if not record_id.strip():
        stats.counts["empty_id_records"] += 1
    if not input_text.strip():
        stats.counts["empty_input_records"] += 1
    if not output_text.strip():
        stats.counts["empty_output_records"] += 1

    record_signals: set[str] = set()
    for field_name, text in (("input", input_text), ("output", output_text)):
        signals = _field_signals(text, special_tokens)
        if field_name == "output":
            signals.update(_answer_signals(text))
        record_signals.update(
            _record_review_signals(
                stats,
                sampler,
                line_number,
                record_id,
                task,
                field_name,
                signals,
            )
        )
    reference_signals: dict[str, str | None] = {}
    for reference in references:
        for signal, value in _field_signals(reference, special_tokens).items():
            reference_signals.setdefault(signal, value)
    if reference_signals:
        record_signals.update(
            _record_review_signals(
                stats,
                sampler,
                line_number,
                record_id,
                task,
                "reference",
                reference_signals,
            )
        )
    stats.signal_counts.update(record_signals)

    question: str | None = input_text if stats.spec.dataset_kind == "pair_qa" else None
    if stats.spec.has_reference:
        question = _record_reference_metrics(
            stats,
            sampler,
            line_number,
            record_id,
            task,
            input_text,
            output_text,
            references,
        )

    input_targets = [stats.lengths["input"]]
    if stats.spec.dataset_kind == "pair_qa":
        input_targets.append(stats.lengths["question"])
    tokenizer_batcher.add(input_text, *input_targets)
    tokenizer_batcher.add(output_text, stats.lengths["output"])
    if question is not None and stats.spec.dataset_kind == "triplet_qa":
        tokenizer_batcher.add(question, stats.lengths["question"])
    if references:
        context = "\n".join(references)
        tokenizer_batcher.add(context, stats.lengths["reference_total"])
        for reference in references:
            tokenizer_batcher.add(reference, stats.lengths["reference_item"])

    _record_citation_metrics(
        stats,
        sampler,
        npc_index,
        line_number,
        record_id,
        task,
        input_text,
        output_text,
        references,
    )
    _add_fingerprints(
        store,
        file_index,
        line_number,
        record_id,
        task,
        input_text,
        output_text,
        references,
        question,
    )


def audit_file(
    path: Path,
    spec: FileSpec,
    file_index: int,
    tokenizer_batcher: TokenBatcher,
    store: FingerprintStore,
    sampler: ReviewSampler,
    npc_index: NpcIndex | None,
    special_tokens: tuple[str, ...],
) -> FileStats:
    """二进制逐行扫描单个文件，同步计算原始 SHA-256。"""

    stats = FileStats(spec=spec, path=path.resolve())
    digest = hashlib.sha256()
    print(f"[审计] {path}", file=sys.stderr, flush=True)
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, start=1):
            digest.update(raw_line)
            stats.counts["physical_lines"] += 1
            if not raw_line.strip():
                stats.counts["blank_lines"] += 1
                continue
            try:
                decoded = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                stats.counts["utf8_errors"] += 1
                stats.record_error(line_number, "UTF-8 解码失败")
                sampler.add(
                    "utf8_error",
                    spec.filename,
                    line_number,
                    f"line-{line_number}",
                    "<unknown>",
                    "record",
                )
                continue
            try:
                record = json.loads(decoded)
            except json.JSONDecodeError:
                stats.counts["json_errors"] += 1
                stats.record_error(line_number, "JSON 解析失败")
                sampler.add(
                    "json_error",
                    spec.filename,
                    line_number,
                    f"line-{line_number}",
                    "<unknown>",
                    "record",
                )
                continue
            stats.counts["json_values_parsed"] += 1
            if not isinstance(record, dict):
                stats.counts["non_object_records"] += 1
                stats.record_error(line_number, "顶层 JSON 不是对象")
                sampler.add(
                    "non_object_record",
                    spec.filename,
                    line_number,
                    f"line-{line_number}",
                    "<unknown>",
                    "record",
                )
                continue
            stats.counts["parsed_objects"] += 1
            values = _validate_record_schema(stats, record, line_number)
            if values is None:
                sampler.add(
                    "schema_error",
                    spec.filename,
                    line_number,
                    str(record.get("id", f"line-{line_number}")),
                    "<unknown>",
                    "record",
                )
                continue
            _audit_valid_record(
                stats,
                file_index,
                line_number,
                values,
                tokenizer_batcher,
                store,
                sampler,
                npc_index,
                special_tokens,
            )
            if stats.counts["physical_lines"] % PROGRESS_INTERVAL == 0:
                print(
                    f"[进度] {spec.filename}: {stats.counts['physical_lines']:,} 行",
                    file=sys.stderr,
                    flush=True,
                )
    tokenizer_batcher.flush()
    store.flush()
    stats.input_bytes = path.stat().st_size
    stats.input_sha256 = digest.hexdigest()
    return stats


def load_tokenizer(path: Path) -> Any:
    """只从本地加载当前 Tokenizer。"""

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise AuditError("缺少 transformers，无法加载当前 Tokenizer") from error
    try:
        return AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True)
    except Exception as error:
        raise AuditError(f"无法加载 Tokenizer: {path}") from error


def _atomic_write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    json.loads(temporary.read_text(encoding="utf-8"))
    temporary.replace(path)


def _atomic_write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> int:
    temporary = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    temporary.replace(path)
    return count


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _tokenizer_report(tokenizer: Any, tokenizer_path: Path) -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        path = tokenizer_path / filename
        if path.is_file():
            files[filename] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"path": str(tokenizer_path), "vocab_size": len(tokenizer), "files": files}


def audit_dataset(
    input_dir: Path,
    tokenizer_path: Path,
    npc_corpus_path: Path | None,
    output_dir: Path,
    review_samples_per_signal: int = 20,
    seed: int = 20_260_730,
    tokenizer: Any | None = None,
) -> dict[str, object]:
    """执行四文件全量只读审计并发布不含原文的报告。"""

    input_dir = input_dir.resolve()
    tokenizer_path = tokenizer_path.resolve()
    output_dir = output_dir.resolve()
    npc_corpus_path = npc_corpus_path.resolve() if npc_corpus_path is not None else None
    if not input_dir.is_dir():
        raise AuditError(f"缺少 DISC 原始目录: {input_dir}")
    if _path_is_within(output_dir, input_dir):
        raise AuditError("报告目录不能位于 DISC 原始目录内")
    if review_samples_per_signal <= 0:
        raise AuditError("每种风险的复核样本数必须大于 0")
    output_dir.mkdir(parents=True, exist_ok=True)

    present_specs = [spec for spec in FILE_SPECS if (input_dir / spec.filename).is_file()]
    missing_files = [spec.filename for spec in FILE_SPECS if spec not in present_specs]
    if not present_specs:
        raise AuditError("四个 DISC 原始文件均不存在")

    tokenizer = tokenizer or load_tokenizer(tokenizer_path)
    if len(tokenizer) != EXPECTED_VOCAB_SIZE:
        raise AuditError(
            f"Tokenizer 词表大小应为 {EXPECTED_VOCAB_SIZE}，实际为 {len(tokenizer)}"
        )
    special_tokens = tuple(
        token
        for token in getattr(tokenizer, "all_special_tokens", ())
        if isinstance(token, str) and token
    )
    npc_index = load_npc_index(npc_corpus_path)
    npc_report: dict[str, object]
    if npc_index is None:
        npc_report = {
            "status": "missing_or_not_configured",
            "path": str(npc_corpus_path) if npc_corpus_path is not None else None,
        }
    else:
        npc_report = npc_index.report

    sampler = ReviewSampler(review_samples_per_signal, seed)
    tokenizer_batcher = TokenBatcher(tokenizer)
    file_names = [spec.filename for spec in FILE_SPECS]
    stats_by_file: dict[str, FileStats] = {}
    duplicate_path = output_dir / DUPLICATE_FILENAME
    answer_variant_path = output_dir / ANSWER_VARIANT_FILENAME
    review_path = output_dir / REVIEW_FILENAME
    report_path = output_dir / REPORT_FILENAME
    hash_path = output_dir / HASH_FILENAME
    final_paths = (report_path, duplicate_path, answer_variant_path, review_path, hash_path)
    pending_paths = {
        path: path.with_name(path.name + ".pending")
        for path in final_paths
    }
    occupied_paths = [path for path in (*final_paths, *pending_paths.values()) if path.exists()]
    if occupied_paths:
        raise AuditError(
            "报告目录已包含审计产物或待发布文件，请为本次运行使用新的 --output-dir: "
            + ", ".join(path.name for path in occupied_paths)
        )

    duplicate_pending_path = pending_paths[duplicate_path]
    answer_variant_pending_path = pending_paths[answer_variant_path]
    review_pending_path = pending_paths[review_path]
    report_pending_path = pending_paths[report_path]
    hash_pending_path = pending_paths[hash_path]

    with tempfile.TemporaryDirectory(prefix=".disc-audit-", dir=output_dir) as temp_dir:
        store = FingerprintStore(Path(temp_dir) / "fingerprints.sqlite3", file_names)
        try:
            for file_index, spec in enumerate(FILE_SPECS):
                path = input_dir / spec.filename
                if not path.is_file():
                    continue
                stats_by_file[spec.filename] = audit_file(
                    path,
                    spec,
                    file_index,
                    tokenizer_batcher,
                    store,
                    sampler,
                    npc_index,
                    special_tokens,
                )
            duplicate_summary = store.duplicate_summary()
            answer_variant_summary = store.answer_variant_summary()
            duplicate_group_count = store.write_duplicate_groups(duplicate_pending_path)
            answer_variant_group_count = store.write_answer_variants(
                answer_variant_pending_path
            )
        except BaseException:
            store.close(commit=False)
            raise
        else:
            store.close()

    review_records = sampler.records()
    review_record_count = _atomic_write_jsonl(review_pending_path, review_records)
    files_report = {}
    for filename, stats in stats_by_file.items():
        per_file_duplicates = {
            kind: duplicate_summary["within_files"][kind][filename]
            for kind in DUPLICATE_KINDS
        }
        files_report[filename] = stats.to_report(per_file_duplicates)

    totals = Counter()
    for stats in stats_by_file.values():
        totals.update(stats.counts)
    all_upstream_snapshots_verified = not missing_files and all(
        stats.input_bytes == stats.spec.upstream_bytes
        and stats.input_sha256 == stats.spec.upstream_sha256
        for stats in stats_by_file.values()
    )
    all_records_expected_schema = not any(
        totals[name]
        for name in (
            "utf8_errors",
            "json_errors",
            "non_object_records",
            "schema_anomaly_records",
        )
    )
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "input_dir": str(input_dir),
            "expected_files": [spec.filename for spec in FILE_SPECS],
            "present_files": list(stats_by_file),
            "missing_files": missing_files,
            "run_completed": True,
            "scan_complete": not missing_files,
            "scan_complete_definition": "仅表示四个预期文件均完成逐行扫描，不代表质量通过",
            "quality_gate_evaluated": False,
            "raw_inputs_modified": False,
        },
        "verification": {
            "all_expected_files_scanned": not missing_files,
            "all_upstream_snapshots_verified": all_upstream_snapshots_verified,
            "all_records_match_expected_core_schema": all_records_expected_schema,
            "npc_grounding_performed": npc_index is not None,
        },
        "tokenizer": _tokenizer_report(tokenizer, tokenizer_path),
        "npc_effective_corpus": npc_report,
        "files": files_report,
        "cross_file_duplicates": duplicate_summary["cross_file_shared_signature_groups"],
        "answer_variants": answer_variant_summary,
        "totals": {
            "physical_lines": totals["physical_lines"],
            "blank_lines": totals["blank_lines"],
            "json_values_parsed": totals["json_values_parsed"],
            "parsed_objects": totals["parsed_objects"],
            "core_valid_records": totals["core_valid_records"],
            "utf8_errors": totals["utf8_errors"],
            "json_errors": totals["json_errors"],
            "schema_anomaly_records": totals["schema_anomaly_records"],
        },
        "outputs": {
            "summary": REPORT_FILENAME,
            "duplicate_groups": {
                "file": DUPLICATE_FILENAME,
                "records": duplicate_group_count,
            },
            "answer_variants": {
                "file": ANSWER_VARIANT_FILENAME,
                "records": answer_variant_group_count,
            },
            "review_candidates": {
                "file": REVIEW_FILENAME,
                "records": review_record_count,
                "max_records_per_signal": review_samples_per_signal,
                "seed": seed,
            },
            "sha256_manifest": HASH_FILENAME,
        },
        "metric_definitions": {
            "duplicate_records": "每个重复组内除首条外的记录数，不是两两组合数",
            "normalized": "仅执行 Unicode NFC、首尾清理和连续空白折叠",
            "answer_variant_scopes": {
                "id_payload": "同一非空 ID 对应多个不同内容 payload，属于完整性冲突候选",
                "task_prompt": "同一任务和空白归一 prompt 对应多个空白归一答案",
                "qa_question": "同一空白归一法律问题对应多个空白归一答案",
            },
            "npc_output_unique_citation_pair_checks_by_status": (
                "按每条记录去重后的 output 法名与条号组合计数"
            ),
        },
        "limitations": [
            "PII 正则命中仅表示复核候选，不等同于自动确认个人信息泄露。",
            "NPC 快照未定位不等于法条伪造，可能涉及简称、历史版本或快照范围外规范。",
            "同题多答案只称为答案变体，不自动判定答案冲突或错误。",
            "本报告不判断法律结论正确性、法律时效适用或回答是否忠实使用上下文。",
            "本报告不包含语义近重复和评估资产污染检查；需在评估清单固定后单独执行。",
        ],
    }
    _atomic_write_json(report_pending_path, report)

    manifest_lines = [
        f"{sha256_file(pending_paths[path])}  {path.name}\n"
        for path in (report_path, duplicate_path, answer_variant_path, review_path)
    ]
    temporary_hash = hash_pending_path.with_suffix(hash_pending_path.suffix + ".tmp")
    temporary_hash.write_text("".join(manifest_lines), encoding="utf-8", newline="\n")
    temporary_hash.replace(hash_pending_path)
    for path in (report_path, duplicate_path, answer_variant_path, review_path, hash_path):
        pending_paths[path].replace(path)
    return report


def parse_args() -> argparse.Namespace:
    """解析云端只读审计参数。"""

    parser = argparse.ArgumentParser(description="只读审计四个 DISC-Law-SFT 原始 JSONL")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="四个 DISC JSONL 所在目录")
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH, help="当前 12000 词表 Tokenizer")
    parser.add_argument("--npc-corpus", type=Path, default=DEFAULT_NPC_CORPUS, help="NPC 现行法规标准化 JSONL")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="审计报告输出目录")
    parser.add_argument("--review-samples-per-signal", type=int, default=20, help="每种风险保留的确定性复核定位数")
    parser.add_argument("--seed", type=int, default=20_260_730, help="复核候选确定性抽样 seed")
    return parser.parse_args()


def main() -> None:
    """运行审计并打印报告位置。"""

    args = parse_args()
    try:
        report = audit_dataset(
            input_dir=args.input_dir,
            tokenizer_path=args.tokenizer_path,
            npc_corpus_path=args.npc_corpus,
            output_dir=args.output_dir,
            review_samples_per_signal=args.review_samples_per_signal,
            seed=args.seed,
        )
    except (AuditError, OSError, ValueError, sqlite3.Error) as error:
        print(f"[失败] {error}", file=sys.stderr)
        raise SystemExit(1) from error

    completion_label = "完成" if report["scope"]["scan_complete"] else "部分完成"
    print(
        f"[{completion_label}] 有效记录 {report['totals']['core_valid_records']:,}，"
        f"缺失文件 {len(report['scope']['missing_files'])} 个"
    )
    print(f"汇总报告: {args.output_dir / REPORT_FILENAME}")
    print(f"哈希清单: {args.output_dir / HASH_FILENAME}")


if __name__ == "__main__":
    main()
