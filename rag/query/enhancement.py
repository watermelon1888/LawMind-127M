"""Query 增强模型协议与多 query 检索腿编译。"""

import json
import re
import unicodedata
from dataclasses import dataclass, field


QUERY_ENHANCEMENT_SYSTEM_PROMPT = """你是法律检索查询增强器。只输出一个紧凑JSON：{"rewrite":"规范化等价改写","expansion_terms":[],"subqueries":[]}，字段顺序固定。
rewrite必须保留原问题全部事实、关系、时间、金额、请求事项和歧义；不得回答、补充事实或擅自选择可能意图，可以与原问题相同。
expansion_terms为0至4个简短检索术语或同义表达，不得包含完整问句、答案、法名、条号或隐藏证据信息。
subqueries仅将原问题明确包含的多个事项拆为0至3个完整问题，不得枚举歧义猜测。
无需增强时两个数组为空。不得输出Markdown、解释或思考过程。"""

QUERY_ENHANCEMENT_SCHEMA = (
    '{"rewrite":"非空的规范化等价改写",'
    '"expansion_terms":["最多四个简短法律术语"],'
    '"subqueries":["最多三个完整的原子法律问题"]}'
)

_FIELDS = ("rewrite", "expansion_terms", "subqueries")
_MAX_REWRITE_CHARS = 112
_MAX_EXPANSION_TERM_CHARS = 16
_MAX_EXPANSION_TERMS = 4
_MAX_SUBQUERY_CHARS = 80
_MAX_SUBQUERIES = 3


class QueryEnhancementProtocolError(ValueError):
    """增强模型输出不符合严格三字段协议。"""


class _StrictJsonError(ValueError):
    """标准解析器默认放行了协议禁止的 JSON 写法。"""


def _object_without_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _StrictJsonError("JSON 对象包含重复字段")
        result[key] = value
    return result


def _reject_nonstandard_constant(value):
    raise _StrictJsonError(f"JSON 包含非标准常量: {value}")


def _normalize_text(value):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()


def _normalize_string_array(name, values, *, maximum, maximum_chars):
    if not isinstance(values, list) and not isinstance(values, tuple):
        raise TypeError(f"{name} 必须是字符串数组")
    if len(values) > maximum:
        raise ValueError(f"{name} 最多包含 {maximum} 项")
    normalized = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"{name} 必须是字符串数组")
        item = _normalize_text(value)
        if not item:
            raise ValueError(f"{name} 不能包含空字符串")
        if len(item) > maximum_chars:
            raise ValueError(f"{name} 单项最多包含 {maximum_chars} 个字符")
        if item not in seen:
            seen.add(item)
            normalized.append(item)
    return tuple(normalized)


@dataclass(frozen=True)
class QueryEnhancement:
    """通过协议校验并完成最小规范化的 Query 增强结果。"""

    rewrite: str
    expansion_terms: tuple[str, ...] = field(default_factory=tuple)
    subqueries: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not isinstance(self.rewrite, str):
            raise TypeError("rewrite 必须是字符串")
        rewrite = _normalize_text(self.rewrite)
        if not rewrite:
            raise ValueError("rewrite 必须是非空字符串")
        if len(rewrite) > _MAX_REWRITE_CHARS:
            raise ValueError(f"rewrite 最多包含 {_MAX_REWRITE_CHARS} 个字符")
        object.__setattr__(self, "rewrite", rewrite)
        object.__setattr__(
            self,
            "expansion_terms",
            _normalize_string_array(
                "expansion_terms",
                self.expansion_terms,
                maximum=_MAX_EXPANSION_TERMS,
                maximum_chars=_MAX_EXPANSION_TERM_CHARS,
            ),
        )
        object.__setattr__(
            self,
            "subqueries",
            _normalize_string_array(
                "subqueries",
                self.subqueries,
                maximum=_MAX_SUBQUERIES,
                maximum_chars=_MAX_SUBQUERY_CHARS,
            ),
        )


def build_query_enhancement_prompt(query):
    """构造只包含原始问题的 Query 增强模型消息。"""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 必须是非空字符串")
    return [
        {"role": "system", "content": QUERY_ENHANCEMENT_SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]


def parse_and_validate_query_enhancement(raw_text):
    """严格解析三字段 JSON，任一字段非法时拒绝整份输出。"""
    if not isinstance(raw_text, str):
        raise TypeError("raw_text 必须是字符串")
    try:
        payload = json.loads(
            raw_text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonstandard_constant,
        )
    except (json.JSONDecodeError, _StrictJsonError) as exc:
        raise QueryEnhancementProtocolError("模型输出不是严格 JSON") from exc
    if not isinstance(payload, dict) or tuple(payload) != _FIELDS:
        raise QueryEnhancementProtocolError("顶层字段及顺序必须精确匹配三字段协议")
    try:
        return QueryEnhancement(
            rewrite=payload["rewrite"],
            expansion_terms=payload["expansion_terms"],
            subqueries=payload["subqueries"],
        )
    except (TypeError, ValueError) as exc:
        raise QueryEnhancementProtocolError("Query 增强字段不合法") from exc


def compile_retrieval_queries(original_query, enhancement):
    """按固定规则编译并稳定去重最多六条检索 query。"""
    if not isinstance(original_query, str):
        raise TypeError("original_query 必须是字符串")
    original = _normalize_text(original_query)
    if not original:
        raise ValueError("original_query 必须是非空字符串")
    if not isinstance(enhancement, QueryEnhancement):
        raise TypeError("enhancement 必须是 QueryEnhancement")

    candidates = [original, enhancement.rewrite]
    if enhancement.expansion_terms:
        candidates.append(
            " ".join((enhancement.rewrite, *enhancement.expansion_terms))
        )
    candidates.extend(enhancement.subqueries)

    compiled = []
    seen = set()
    for candidate in candidates:
        normalized = _normalize_text(candidate)
        if normalized not in seen:
            seen.add(normalized)
            compiled.append(normalized)
    return tuple(compiled)


__all__ = [
    "QUERY_ENHANCEMENT_SCHEMA",
    "QUERY_ENHANCEMENT_SYSTEM_PROMPT",
    "QueryEnhancement",
    "QueryEnhancementProtocolError",
    "build_query_enhancement_prompt",
    "compile_retrieval_queries",
    "parse_and_validate_query_enhancement",
]
