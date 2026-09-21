"""现行有效法条索引的确定性访问。"""
import json
import re
from dataclasses import dataclass
from pathlib import Path


_CN_DIGITS = {
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
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}
_INT_TO_CN_DIGIT = "零一二三四五六七八九"
_CN_NORMALIZE_TABLE = str.maketrans({"〇": "零", "两": "二"})
_ARTICLE_NO_RE = re.compile(
    r"^第?(?P<main>[零〇一二三四五六七八九十百千两\d]+)条?"
    r"(?:之(?P<suffix>[零〇一二三四五六七八九十百千两\d]+))?$"
)
_VERSION_SUFFIX_RE = re.compile(r"（[^）]+）$")
_COUNTRY_PREFIX = "中华人民共和国"
_REQUIRED_RECORD_FIELDS = (
    "chunk_id",
    "law_name",
    "article_no",
    "content",
)
_SOURCE_TYPES = frozenset({"legal_regulation", "department_rule"})


class IndexIntegrityError(ValueError):
    """法条索引结构或唯一性约束被破坏。"""


@dataclass(frozen=True)
class LegalArticle:
    """knowledge 向调用方返回的完整法条。"""

    chunk_id: str
    law_name: str
    article_no: str
    content: str
    source_type: str = "legal_regulation"

    def __post_init__(self):
        if self.source_type not in _SOURCE_TYPES:
            raise ValueError("source_type 必须是 legal_regulation 或 department_rule")


class ArticleRepository:
    """把 JSONL 法条索引封装为确定性查找接口。"""

    def __init__(self, articles_by_key, law_names_by_alias, formal_law_names):
        self._articles_by_key = articles_by_key
        self._articles_by_chunk_id = {
            article.chunk_id: article for article in articles_by_key.values()
        }
        self._law_names_by_alias = law_names_by_alias
        self._formal_law_names = formal_law_names

    @classmethod
    def from_jsonl(cls, path):
        """加载 JSONL 索引并建立法名、条号二元查找表。"""
        articles_by_key = {}
        formal_law_names = set()
        with Path(path).open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise IndexIntegrityError(
                        f"第 {line_number} 行不是有效 JSON"
                    ) from exc
                if not isinstance(record, dict):
                    raise IndexIntegrityError(
                        f"第 {line_number} 行必须是 JSON 对象"
                    )
                for field in _REQUIRED_RECORD_FIELDS:
                    value = record.get(field)
                    if not isinstance(value, str) or not value.strip():
                        raise IndexIntegrityError(
                            f"第 {line_number} 行的 {field} 必须是非空字符串"
                        )

                try:
                    canonical_article_no = _normalize_article_no(
                        record["article_no"]
                    )
                except ValueError as exc:
                    raise IndexIntegrityError(
                        f"第 {line_number} 行的 article_no 格式无效"
                    ) from exc
                if canonical_article_no != record["article_no"]:
                    raise IndexIntegrityError(
                        f"第 {line_number} 行的 article_no 不是规范格式"
                    )

                expected_chunk_id = (
                    f"{record['law_name']}#{record['article_no']}"
                )
                if record["chunk_id"] != expected_chunk_id:
                    raise IndexIntegrityError(
                        f"第 {line_number} 行的 chunk_id 与法名、条号不一致"
                    )
                content = record["content"]
                source_type = record.get("source_type", "legal_regulation")
                if source_type not in _SOURCE_TYPES:
                    raise IndexIntegrityError(
                        f"第 {line_number} 行的 source_type 无效"
                    )
                article = LegalArticle(
                    chunk_id=record["chunk_id"],
                    law_name=record["law_name"],
                    article_no=record["article_no"],
                    content=content,
                    source_type=source_type,
                )
                key = (article.law_name, article.article_no)
                if key in articles_by_key:
                    raise IndexIntegrityError(
                        f"第 {line_number} 行存在重复法条键: {key!r}"
                    )
                articles_by_key[key] = article
                formal_law_names.add(article.law_name)

        if not articles_by_key:
            raise IndexIntegrityError("法条索引不能为空")

        law_names_by_alias = {}
        for formal_name in formal_law_names:
            for alias in _derive_law_aliases(formal_name):
                law_names_by_alias.setdefault(alias, set()).add(formal_name)
        return cls(articles_by_key, law_names_by_alias, formal_law_names)

    def lookup(self, law_name, article_no):
        """按正式法名和规范条号查找唯一法条。"""
        normalized_law_name = _normalize_law_name(law_name)
        if normalized_law_name in self._formal_law_names:
            candidates = {normalized_law_name}
        else:
            candidates = self._law_names_by_alias.get(normalized_law_name)
        if not candidates:
            return None
        if len(candidates) > 1:
            return None
        formal_name = next(iter(candidates))
        try:
            normalized_article_no = _normalize_article_no(article_no)
        except ValueError:
            return None
        return self._articles_by_key.get(
            (formal_name, normalized_article_no)
        )

    def get_by_chunk_id(self, chunk_id):
        """按内部唯一标识返回同一个法条对象；缺失时抛出 KeyError。"""
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("chunk_id 必须是非空字符串")
        try:
            return self._articles_by_chunk_id[chunk_id]
        except KeyError as exc:
            raise KeyError(f"法条仓库中不存在 chunk_id: {chunk_id}") from exc


def _normalize_law_name(law_name):
    compact = re.sub(r"\s+", "", law_name)
    if compact.startswith("《") and compact.endswith("》"):
        compact = compact[1:-1]
    return compact


def _derive_law_aliases(formal_name):
    aliases = {_normalize_law_name(formal_name)}
    without_version = _VERSION_SUFFIX_RE.sub("", formal_name)
    aliases.add(_normalize_law_name(without_version))
    for name in tuple(aliases):
        if name.startswith(_COUNTRY_PREFIX):
            aliases.add(name[len(_COUNTRY_PREFIX) :])
    return aliases


def _normalize_article_no(article_no):
    compact = re.sub(r"\s+", "", article_no)
    match = _ARTICLE_NO_RE.fullmatch(compact)
    if not match:
        raise ValueError("article_no 格式无效")

    main = str(_parse_number(match.group("main")))
    suffix = match.group("suffix")
    if suffix is None:
        return main
    return f"{main}之{_format_chinese_number(_parse_number(suffix))}"


def _parse_number(text):
    is_arabic = text.isdigit()
    if is_arabic:
        value = int(text)
    else:
        if any(char.isdigit() for char in text):
            raise ValueError("条号不能混用中文数字和阿拉伯数字")
        value = 0
        current = 0
        for char in text:
            if char in _CN_DIGITS:
                current = _CN_DIGITS[char]
                continue
            unit = _CN_UNITS.get(char)
            if unit is None:
                raise ValueError("条号包含不支持的数字字符")
            value += (current or 1) * unit
            current = 0
        value += current
    if value <= 0:
        raise ValueError("条号必须大于 0")
    if not is_arabic and text.translate(_CN_NORMALIZE_TABLE) != _format_chinese_integer(
        value
    ):
        raise ValueError("条号中的中文数字不是规范写法")
    return value


def _format_chinese_integer(value):
    if not 0 < value < 10000:
        raise ValueError("中文条号必须小于一万")

    parts = []
    started = False
    zero_pending = False
    for divisor, unit in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
        digit = (value // divisor) % 10
        remaining = value % divisor
        if digit == 0:
            if started and remaining:
                zero_pending = True
            continue
        if zero_pending:
            parts.append("零")
            zero_pending = False
        if not (divisor == 10 and digit == 1 and not started):
            parts.append(_INT_TO_CN_DIGIT[digit])
        parts.append(unit)
        started = True
    return "".join(parts)


def _format_chinese_number(value):
    if value < 100:
        return _format_chinese_integer(value)
    raise ValueError("条号之几的后缀不能超过九十九")


__all__ = [
    "ArticleRepository",
    "IndexIntegrityError",
    "LegalArticle",
]
