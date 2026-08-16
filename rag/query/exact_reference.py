"""从原始问题中确定性提取一至三条法条引用。"""
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple


MAX_EXACT_REFERENCES = 3

_NUMBER_PATTERN = r"[零〇一二三四五六七八九十百千万两\d]+"
_ARTICLE_SUFFIX_PATTERN = rf"(?:\s*之\s*{_NUMBER_PATTERN})?"
_ARTICLE_REFERENCE_PATTERN = (
    rf"第\s*{_NUMBER_PATTERN}\s*条{_ARTICLE_SUFFIX_PATTERN}"
)
_ARTICLE_REFERENCE_RE = re.compile(_ARTICLE_REFERENCE_PATTERN)
_REFERENCE_SEPARATOR_PATTERN = r"(?:、|，|,|以及|或者|和|及|与|或)"
_SHARED_ARTICLE_ENDING_RE = re.compile(
    rf"第\s*(?P<numbers>{_NUMBER_PATTERN}"
    rf"(?:\s*{_REFERENCE_SEPARATOR_PATTERN}\s*(?:第\s*)?{_NUMBER_PATTERN})+)"
    rf"\s*条(?P<suffix>{_ARTICLE_SUFFIX_PATTERN})"
)
_ABBREVIATED_ARTICLE_RE = re.compile(
    rf"(?P<full>{_ARTICLE_REFERENCE_PATTERN})"
    rf"(?P<separator>\s*{_REFERENCE_SEPARATOR_PATTERN}\s*)"
    rf"(?P<number>{_NUMBER_PATTERN})\s*条"
    rf"(?P<suffix>{_ARTICLE_SUFFIX_PATTERN})"
)
_BOOK_TITLE_RE = re.compile(r"《(?P<law_name>[^《》]+)》")
_LAW_NAME_ENDING_PATTERN = r"(?:法典|法|条例|规定|决定|办法|规则|通则)"
_BARE_LAW_NAME_RE = re.compile(
    rf"^(?P<law_name>[\u4e00-\u9fffA-Za-z0-9·]+?"
    rf"{_LAW_NAME_ENDING_PATTERN}(?:（[^（）]+）)?)(?:的)?$"
)
_MULTIPLE_BARE_LAW_NAMES_RE = re.compile(
    rf"(?:{_LAW_NAME_ENDING_PATTERN}|》){_REFERENCE_SEPARATOR_PATTERN}"
    rf"(?:《)?[^《》]*{_LAW_NAME_ENDING_PATTERN}(?:》)?(?:的)?$"
)
_ENACTMENT_CONTEXT_RE = re.compile(
    r"^(?:在)?(?:19|20)\d{2}年(?:通过|颁布|公布|施行|生效)的"
)
_LEADING_SEPARATOR_RE = re.compile(
    rf"^(?:{_REFERENCE_SEPARATOR_PATTERN}|[。；;：:])*"
)
_BARE_QUERY_PREFIXES = (
    "请告诉我",
    "请解释一下",
    "请说明一下",
    "请对比一下",
    "请比较一下",
    "帮我查一下",
    "我想了解",
    "请解释",
    "请说明",
    "请对比",
    "请比较",
    "帮我查",
    "请问",
    "根据",
    "按照",
    "对比",
    "比较",
)


class ExactReferenceStatus(str, Enum):
    """精确引用解析是否可以进入知识库查找。"""

    FOUND = "found"
    CLARIFICATION_REQUIRED = "clarification_required"


@dataclass(frozen=True)
class ExactReference:
    """一条尚未经过知识库真实性核验的法名与条号。"""

    law_name: str
    article_no: str

    def __post_init__(self):
        if not isinstance(self.law_name, str) or not self.law_name.strip():
            raise ValueError("law_name 必须是非空字符串")
        if not isinstance(self.article_no, str) or not self.article_no.strip():
            raise ValueError("article_no 必须是非空字符串")


@dataclass(frozen=True)
class ExactReferenceResult:
    """一至三条完整引用，或不携带部分结果的受控澄清。"""

    status: ExactReferenceStatus
    references: Tuple[ExactReference, ...] = field(default_factory=tuple)

    def __post_init__(self):
        if not isinstance(self.status, ExactReferenceStatus):
            raise TypeError("status 必须是 ExactReferenceStatus")
        references = tuple(self.references)
        object.__setattr__(self, "references", references)
        if any(not isinstance(item, ExactReference) for item in references):
            raise TypeError("references 中的元素必须是 ExactReference")
        if self.status is ExactReferenceStatus.FOUND:
            if not 1 <= len(references) <= MAX_EXACT_REFERENCES:
                raise ValueError("解析成功时必须包含一至三条引用")
        elif references:
            raise ValueError("请求澄清时不能携带部分引用")


def _expand_article_sequences(query):
    def expand_shared(match):
        numbers = re.findall(_NUMBER_PATTERN, match.group("numbers"))
        suffix = match.group("suffix") or ""
        references = []
        for index, number in enumerate(numbers):
            current_suffix = suffix if index == len(numbers) - 1 else ""
            references.append(f"第{number}条{current_suffix}")
        return "、".join(references)

    expanded = _SHARED_ARTICLE_ENDING_RE.sub(expand_shared, query)
    while True:
        expanded, count = _ABBREVIATED_ARTICLE_RE.subn(
            lambda match: (
                match.group("full")
                + match.group("separator")
                + f"第{match.group('number')}条{match.group('suffix') or ''}"
            ),
            expanded,
        )
        if count == 0:
            return expanded


def count_article_references(query):
    """统计单条及并列写法中的法条引用数量。"""
    if not isinstance(query, str):
        raise TypeError("query 必须是字符串")
    compact_query = re.sub(r"\s+", "", query)
    expanded_query = _expand_article_sequences(compact_query)
    return len(tuple(_ARTICLE_REFERENCE_RE.finditer(expanded_query)))


def contains_article_reference(query):
    """判断问题是否包含至少一条法条引用。"""
    return count_article_references(query) > 0


def extract_exact_references(query):
    """提取一至三条完整引用；不完整时不返回部分结果。"""
    if not isinstance(query, str):
        raise TypeError("query 必须是字符串")
    if not query.strip():
        raise ValueError("query 必须包含法条引用")

    compact_query = re.sub(r"\s+", "", query)
    expanded_query = _expand_article_sequences(compact_query)
    article_matches = tuple(_ARTICLE_REFERENCE_RE.finditer(expanded_query))
    if not article_matches:
        raise ValueError("query 中没有可识别的法条引用")
    if len(article_matches) > MAX_EXACT_REFERENCES:
        return ExactReferenceResult(
            ExactReferenceStatus.CLARIFICATION_REQUIRED
        )

    references = []
    inherited_law_name = None
    previous_end = 0
    for article_match in article_matches:
        law_segment = expanded_query[previous_end : article_match.start()]
        if _contains_multiple_law_names(law_segment):
            return ExactReferenceResult(
                ExactReferenceStatus.CLARIFICATION_REQUIRED
            )
        law_name = _extract_law_name(law_segment)
        if law_name is None:
            law_name = inherited_law_name
        if law_name is None:
            return ExactReferenceResult(
                ExactReferenceStatus.CLARIFICATION_REQUIRED
            )
        references.append(
            ExactReference(
                law_name=law_name,
                article_no=article_match.group(0),
            )
        )
        inherited_law_name = law_name
        previous_end = article_match.end()

    return ExactReferenceResult(
        ExactReferenceStatus.FOUND,
        tuple(references),
    )


def _extract_law_name(segment):
    book_titles = tuple(_BOOK_TITLE_RE.finditer(segment))
    if book_titles:
        return book_titles[-1].group("law_name").strip()

    compact_segment = _LEADING_SEPARATOR_RE.sub("", segment)
    compact_segment = _strip_supported_bare_context(compact_segment)
    bare_law_name = _BARE_LAW_NAME_RE.fullmatch(compact_segment)
    if bare_law_name is None:
        return None
    return bare_law_name.group("law_name")


def _contains_multiple_law_names(segment):
    book_titles = tuple(_BOOK_TITLE_RE.finditer(segment))
    if len(book_titles) > 1:
        return True
    return bool(_MULTIPLE_BARE_LAW_NAMES_RE.search(segment))


def _strip_supported_bare_context(compact_prefix):
    for prefix in _BARE_QUERY_PREFIXES:
        if compact_prefix.startswith(prefix):
            compact_prefix = compact_prefix[len(prefix) :].lstrip("：:,，")
            break
    enactment_context = _ENACTMENT_CONTEXT_RE.match(compact_prefix)
    if enactment_context is not None:
        compact_prefix = compact_prefix[enactment_context.end() :]
    return compact_prefix


__all__ = [
    "ExactReference",
    "ExactReferenceResult",
    "ExactReferenceStatus",
    "count_article_references",
    "contains_article_reference",
    "extract_exact_references",
]
