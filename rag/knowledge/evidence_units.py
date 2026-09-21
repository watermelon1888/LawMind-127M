"""从 canonical 法条确定性生成可审计的原文证据子单元。"""

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Tuple

from rag.knowledge.repository import LegalArticle
from rag.knowledge.repository import ArticleRepository


SPLITTER_VERSION = "v1"
SCHEMA_VERSION = "evidence-unit-v1"
MIN_SENTENCE_SPLIT_CHARS = 160
MIN_NESTED_LIST_SPLIT_CHARS = 240
UNIT_TYPES = frozenset(
    {"lead", "paragraph", "list_item", "sentence", "tail", "full"}
)

_VERSION_RE = re.compile(r"^[a-z0-9.-]+$")
_LIST_MARKER_RE = re.compile(
    r"^[ \t]*(?:[（(](?P<cn_paren>[一二三四五六七八九十百零〇两]+)[）)]"
    r"|(?P<cn_bare>[一二三四五六七八九十百零〇两]+)、"
    r"|(?P<num>\d+)(?:、|\.(?!\d)))"
)
_INLINE_LIST_MARKER_RE = re.compile(
    r"(?:[（(](?P<cn_paren>[一二三四五六七八九十百零〇两]+)[）)]"
    r"|(?P<cn_bare>[一二三四五六七八九十百零〇两]+)、"
    r"|(?P<num>\d+)(?:、|\.(?!\d)))"
)
_SENTENCE_END_RE = re.compile(r"[。！？][”’》）)]*")
_NESTED_NUMERIC_MARKER_RE = re.compile(
    r"(?<!\d)(?P<num>\d{1,2})[、.]"
)
_DEPENDENT_PREFIX_RE = re.compile(
    r"^(?:但(?:是)?|然而|其中|同时|并且|此外|上述|该|其|"
    r"前款|前两款|前述|前项|对前款|有前款|依照前款|本条前款|本条第)"
)
_PREVIOUS_REFERENCE_RE = re.compile(
    r"(?:前款|前两款|前述|前项|依照前款|有前款|犯前款罪)"
)
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
_CN_UNITS = {"十": 10, "百": 100}


def _require_non_blank(name, value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")


def make_unit_id(parent_chunk_id, splitter_version, start_char, end_char):
    """根据父 ID、splitter 版本和原文区间生成稳定子单元 ID。"""
    _require_non_blank("parent_chunk_id", parent_chunk_id)
    _require_non_blank("splitter_version", splitter_version)
    if not _VERSION_RE.fullmatch(splitter_version):
        raise ValueError("splitter_version 只能包含 ASCII 小写字母、数字、点和短横线")
    if (
        not isinstance(start_char, int)
        or isinstance(start_char, bool)
        or start_char < 0
    ):
        raise ValueError("start_char 必须是非负整数")
    if (
        not isinstance(end_char, int)
        or isinstance(end_char, bool)
        or end_char <= start_char
    ):
        raise ValueError("end_char 必须是大于 start_char 的整数")
    return (
        f"{parent_chunk_id}::span-{splitter_version}@"
        f"{start_char:06d}-{end_char:06d}"
    )


@dataclass(frozen=True)
class EvidenceUnit:
    """canonical 父法条中的一个连续、可复原原文区间。"""

    unit_id: str
    parent_chunk_id: str
    unit_index: int
    unit_type: str
    text: str
    start_char: int
    end_char: int
    dependency_unit_ids: Tuple[str, ...] = field(default_factory=tuple)
    splitter_version: str = SPLITTER_VERSION

    def __post_init__(self):
        for name in ("unit_id", "parent_chunk_id", "unit_type", "text"):
            _require_non_blank(name, getattr(self, name))
        if self.unit_type not in UNIT_TYPES:
            raise ValueError(f"unit_type 不受支持: {self.unit_type}")
        if (
            not isinstance(self.unit_index, int)
            or isinstance(self.unit_index, bool)
            or self.unit_index < 0
        ):
            raise ValueError("unit_index 必须是非负整数")
        expected = make_unit_id(
            self.parent_chunk_id,
            self.splitter_version,
            self.start_char,
            self.end_char,
        )
        if self.unit_id != expected:
            raise ValueError("unit_id 与父 ID、版本和原文区间不一致")
        dependencies = tuple(self.dependency_unit_ids)
        object.__setattr__(self, "dependency_unit_ids", dependencies)
        if len(set(dependencies)) != len(dependencies):
            raise ValueError("dependency_unit_ids 不能重复")
        if self.unit_id in dependencies:
            raise ValueError("子单元不能依赖自身")
        for dependency in dependencies:
            _require_non_blank("dependency_unit_ids 中的 ID", dependency)

    def to_dict(self):
        """返回字段顺序稳定的 JSON 可序列化记录。"""
        return {
            "unit_id": self.unit_id,
            "parent_chunk_id": self.parent_chunk_id,
            "unit_index": self.unit_index,
            "unit_type": self.unit_type,
            "text": self.text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "dependency_unit_ids": list(self.dependency_unit_ids),
            "splitter_version": self.splitter_version,
        }


class EvidenceUnitRepository:
    """加载版本化 sidecar，并提供显式父子映射与稳定查找。"""

    def __init__(self, units_by_id, units_by_parent, splitter_version):
        self._units_by_id = units_by_id
        self._units_by_parent = units_by_parent
        self._splitter_version = splitter_version

    @classmethod
    def from_jsonl(
        cls,
        path,
        *,
        article_repository,
        splitter_version=SPLITTER_VERSION,
    ):
        """加载并逐父法条校验 sidecar，不从 unit_id 反解父 ID。"""
        if not isinstance(article_repository, ArticleRepository):
            raise TypeError("article_repository 必须是 ArticleRepository")
        units_by_id = {}
        units_by_parent = {}
        current_parent_id = None
        current_units = []
        closed_parents = set()

        def close_group():
            nonlocal current_parent_id, current_units
            if current_parent_id is None:
                return
            article = article_repository.get_by_chunk_id(current_parent_id)
            validate_article_units(
                article,
                current_units,
                splitter_version=splitter_version,
            )
            units_by_parent[current_parent_id] = tuple(current_units)
            closed_parents.add(current_parent_id)
            current_parent_id = None
            current_units = []

        with Path(path).open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    unit = EvidenceUnit(
                        unit_id=record["unit_id"],
                        parent_chunk_id=record["parent_chunk_id"],
                        unit_index=record["unit_index"],
                        unit_type=record["unit_type"],
                        text=record["text"],
                        start_char=record["start_char"],
                        end_char=record["end_char"],
                        dependency_unit_ids=tuple(record["dependency_unit_ids"]),
                        splitter_version=record["splitter_version"],
                    )
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"sidecar 第 {line_number} 行无效") from exc
                if unit.splitter_version != splitter_version:
                    raise ValueError("sidecar 包含不匹配的 splitter_version")
                if current_parent_id is None:
                    if unit.parent_chunk_id in closed_parents:
                        raise ValueError("同一父法条的子单元必须连续存储")
                    current_parent_id = unit.parent_chunk_id
                elif unit.parent_chunk_id != current_parent_id:
                    close_group()
                    if unit.parent_chunk_id in closed_parents:
                        raise ValueError("同一父法条的子单元必须连续存储")
                    current_parent_id = unit.parent_chunk_id
                if unit.unit_id in units_by_id:
                    raise ValueError(f"sidecar 包含重复 unit_id: {unit.unit_id}")
                current_units.append(unit)
                units_by_id[unit.unit_id] = unit
        close_group()
        if not units_by_id:
            raise ValueError("证据子单元 sidecar 不能为空")
        return cls(units_by_id, units_by_parent, splitter_version)

    @property
    def splitter_version(self):
        return self._splitter_version

    def __len__(self):
        return len(self._units_by_id)

    def iter_units(self):
        """按 sidecar 的稳定写入顺序迭代全部子单元。"""
        return iter(self._units_by_id.values())

    def get_by_unit_id(self, unit_id):
        _require_non_blank("unit_id", unit_id)
        try:
            return self._units_by_id[unit_id]
        except KeyError as exc:
            raise KeyError(f"子单元仓库中不存在 unit_id: {unit_id}") from exc

    def get_for_parent(self, parent_chunk_id):
        _require_non_blank("parent_chunk_id", parent_chunk_id)
        try:
            return self._units_by_parent[parent_chunk_id]
        except KeyError as exc:
            raise KeyError(
                f"子单元仓库中不存在父法条: {parent_chunk_id}"
            ) from exc


@dataclass(frozen=True)
class _DraftSpan:
    start: int
    end: int
    unit_type: str
    dependency_indexes: Tuple[int, ...] = field(default_factory=tuple)


def _trim_span(content, start, end):
    while start < end and content[start].isspace():
        start += 1
    while end > start and content[end - 1].isspace():
        end -= 1
    return start, end


def _paragraph_spans(content):
    raw_spans = []
    start = 0
    for match in re.finditer(r"\r?\n+", content):
        trimmed = _trim_span(content, start, match.start())
        if trimmed[0] < trimmed[1]:
            raw_spans.append(trimmed)
        start = match.end()
    trimmed = _trim_span(content, start, len(content))
    if trimmed[0] < trimmed[1]:
        raw_spans.append(trimmed)

    spans = []
    for span in raw_spans:
        if not spans:
            spans.append(span)
            continue
        previous_start, previous_end = spans[-1]
        previous_text = content[previous_start:previous_end].rstrip()
        current_text = content[span[0] : span[1]].lstrip()
        previous_is_complete = previous_text.endswith(
            ("。", "！", "？", "；", ";", "：", ":")
        )
        current_is_list_item = _LIST_MARKER_RE.match(current_text) is not None
        current_starts_with_closer = current_text.startswith(
            ("，", "。", "；", "：", "、", "）", ")", "》", "”", "’")
        )
        if (
            (not previous_is_complete and not current_is_list_item)
            or current_starts_with_closer
        ):
            spans[-1] = (previous_start, span[1])
        else:
            spans.append(span)
    return tuple(spans)


def _parse_marker_number(match):
    if match.group("num") is not None:
        return int(match.group("num"))
    text = match.group("cn_paren") or match.group("cn_bare")
    value = 0
    current = 0
    for char in text:
        if char in _CN_DIGITS:
            current = _CN_DIGITS[char]
            continue
        unit = _CN_UNITS.get(char)
        if unit is None:
            return None
        value += (current or 1) * unit
        current = 0
    return value + current


def _ordered_list_markers(text):
    matches = tuple(_INLINE_LIST_MARKER_RE.finditer(text))
    if len(matches) < 2:
        return ()
    numbers = tuple(_parse_marker_number(match) for match in matches)
    if numbers[0] != 1:
        return ()
    if any(number is None for number in numbers):
        return ()
    if numbers != tuple(range(1, len(numbers) + 1)):
        return ()
    return matches


def _paragraph_list_markers(content, paragraphs):
    """返回同一编号体系且从一开始连续的段首列表项。"""
    candidates = {"cn_paren": [], "cn_bare": [], "num": []}
    for paragraph_index, (start, end) in enumerate(paragraphs):
        match = _LIST_MARKER_RE.match(content[start:end])
        if match is None:
            continue
        style = (
            "cn_paren"
            if match.group("cn_paren") is not None
            else "cn_bare"
            if match.group("cn_bare") is not None
            else "num"
        )
        candidates[style].append(
            (paragraph_index, _parse_marker_number(match))
        )
    valid = []
    for style, values in candidates.items():
        numbers = tuple(number for _, number in values)
        if len(values) >= 2 and numbers == tuple(range(1, len(values) + 1)):
            valid.append((values[0][0], style, values))
    if not valid:
        return ()
    _, _, selected = min(valid, key=lambda item: item[0])
    return tuple(index for index, _ in selected)


def _sentence_spans(content, start, end):
    text = content[start:end]
    boundaries = [match.end() for match in _SENTENCE_END_RE.finditer(text)]
    if not boundaries or boundaries[-1] != len(text):
        boundaries.append(len(text))
    spans = []
    local_start = 0
    for boundary in boundaries:
        span = _trim_span(content, start + local_start, start + boundary)
        local_start = boundary
        if span[0] < span[1]:
            if spans and _DEPENDENT_PREFIX_RE.match(content[span[0] : span[1]]):
                previous_start, _ = spans.pop()
                spans.append((previous_start, span[1]))
            else:
                spans.append(span)
    return tuple(spans)


def _split_paragraph(content, start, end, *, paragraph_count):
    text = content[start:end]
    markers = _ordered_list_markers(text)
    if markers:
        spans = []
        first_marker = start + markers[0].start()
        lead = _trim_span(content, start, first_marker)
        if lead[0] < lead[1]:
            spans.append((lead[0], lead[1], "lead"))
        for index, marker in enumerate(markers):
            item_start = start + marker.start()
            item_end = (
                start + markers[index + 1].start()
                if index + 1 < len(markers)
                else end
            )
            item = _trim_span(content, item_start, item_end)
            if item[0] < item[1]:
                spans.append((item[0], item[1], "list_item"))
        return tuple(spans)

    if _LIST_MARKER_RE.match(text):
        return ((start, end, "list_item"),)
    if paragraph_count == 1 and len(text) < MIN_SENTENCE_SPLIT_CHARS:
        return ((start, end, "full"),)
    sentence_spans = _sentence_spans(content, start, end)
    if len(sentence_spans) >= 2:
        return tuple((left, right, "sentence") for left, right in sentence_spans)
    return ((start, end, "paragraph" if paragraph_count > 1 else "full"),)


def _expand_nested_lists(content, raw_spans):
    """把超长顶层列表项中的连续阿拉伯数字子列表拆成依赖单元。"""
    expanded = []
    for start, end, unit_type in raw_spans:
        text = content[start:end]
        outer_marker = _LIST_MARKER_RE.match(text)
        if unit_type != "list_item" or len(text) < MIN_NESTED_LIST_SPLIT_CHARS:
            expanded.append((start, end, unit_type, None))
            continue
        search_start = 0 if outer_marker is None else outer_marker.end()
        markers = tuple(_NESTED_NUMERIC_MARKER_RE.finditer(text, search_start))
        numbers = tuple(int(marker.group("num")) for marker in markers)
        if len(markers) < 2 or numbers != tuple(range(1, len(markers) + 1)):
            expanded.append((start, end, unit_type, None))
            continue
        prefix_end = start + markers[0].start()
        prefix = _trim_span(content, start, prefix_end)
        if prefix[0] >= prefix[1] or len(content[prefix[0] : prefix[1]]) > 120:
            expanded.append((start, end, unit_type, None))
            continue
        parent_index = len(expanded)
        expanded.append((prefix[0], prefix[1], "list_item", None))
        for marker_index, marker in enumerate(markers):
            item_start = start + marker.start()
            item_end = (
                start + markers[marker_index + 1].start()
                if marker_index + 1 < len(markers)
                else end
            )
            item = _trim_span(content, item_start, item_end)
            if item[0] < item[1]:
                expanded.append(
                    (item[0], item[1], "list_item", parent_index)
                )
    return tuple(expanded)


def _draft_units(content, *, enforce_safety=True):
    paragraphs = _paragraph_spans(content)
    if not paragraphs:
        return ()
    raw = []
    paragraph_markers = _paragraph_list_markers(content, paragraphs)
    if paragraph_markers:
        first_marker = paragraph_markers[0]
        for paragraph_index in range(first_marker):
            start, end = paragraphs[paragraph_index]
            split = _split_paragraph(
                content,
                start,
                end,
                paragraph_count=len(paragraphs),
            )
            if paragraph_index == first_marker - 1:
                split = tuple(split)
                last_left, last_right, _ = split[-1]
                split = split[:-1] + ((last_left, last_right, "lead"),)
            raw.extend(split)

        consumed_through = first_marker
        for marker_position, paragraph_index in enumerate(paragraph_markers):
            next_marker = (
                paragraph_markers[marker_position + 1]
                if marker_position + 1 < len(paragraph_markers)
                else len(paragraphs)
            )
            item_end_index = paragraph_index
            if marker_position + 1 < len(paragraph_markers):
                item_end_index = next_marker - 1
            else:
                initial_text = content[
                    paragraphs[paragraph_index][0] : paragraphs[paragraph_index][1]
                ]
                nested_numbers = tuple(
                    int(match.group("num"))
                    for match in _NESTED_NUMERIC_MARKER_RE.finditer(initial_text)
                )
                if nested_numbers and nested_numbers[0] == 1:
                    item_end_index = next_marker - 1
                else:
                    while item_end_index + 1 < next_marker:
                        _, current_end = paragraphs[item_end_index]
                        if content[
                            paragraphs[paragraph_index][0] : current_end
                        ].rstrip().endswith(("。", "！", "？")):
                            break
                        item_end_index += 1
            item_start = paragraphs[paragraph_index][0]
            item_end = paragraphs[item_end_index][1]
            raw.append((item_start, item_end, "list_item"))
            consumed_through = item_end_index + 1

        for paragraph_index in range(consumed_through, len(paragraphs)):
            start, end = paragraphs[paragraph_index]
            raw.extend(
                _split_paragraph(
                    content,
                    start,
                    end,
                    paragraph_count=len(paragraphs),
                )
            )
    else:
        for start, end in paragraphs:
            raw.extend(
                _split_paragraph(
                    content,
                    start,
                    end,
                    paragraph_count=len(paragraphs),
                )
            )
    for index, (start, end, unit_type) in enumerate(raw):
        if unit_type != "list_item" or index == 0:
            continue
        marker = _LIST_MARKER_RE.match(content[start:end])
        if marker is None or _parse_marker_number(marker) != 1:
            continue
        previous_start, previous_end, previous_type = raw[index - 1]
        if previous_type not in {"lead", "list_item"}:
            raw[index - 1] = (
                previous_start,
                previous_end,
                "lead",
            )
    raw = _expand_nested_lists(content, raw)
    if len(raw) == 1:
        start, end, _, _ = raw[0]
        return (_DraftSpan(start, end, "full"),)

    drafts = []
    active_lead = None
    previous_was_list = False
    first_unit_index = 0
    for start, end, unit_type, contextual_dependency in raw:
        text = content[start:end]
        dependencies = (
            () if contextual_dependency is None else (contextual_dependency,)
        )
        if unit_type == "lead" or (
            unit_type != "list_item" and text.rstrip().endswith(("：", ":"))
        ):
            unit_type = "lead"
            if dependencies:
                pass
            elif text.startswith("本条第") and drafts:
                dependencies = (first_unit_index,)
            elif _PREVIOUS_REFERENCE_RE.search(text) and drafts:
                dependencies = (len(drafts) - 1,)
            elif _DEPENDENT_PREFIX_RE.match(text) and drafts:
                dependencies = (len(drafts) - 1,)
            active_lead = len(drafts)
            previous_was_list = False
        elif unit_type == "list_item":
            dependency_values = list(dependencies)
            if active_lead is not None:
                dependency_values.append(active_lead)
            if _PREVIOUS_REFERENCE_RE.search(text) and drafts:
                dependency_values.append(len(drafts) - 1)
            dependencies = tuple(dict.fromkeys(dependency_values))
            previous_was_list = True
        else:
            if previous_was_list and contextual_dependency is None:
                unit_type = "tail"
            if not dependencies and _PREVIOUS_REFERENCE_RE.search(text) and drafts:
                dependencies = (
                    active_lead if active_lead is not None else len(drafts) - 1,
                )
            elif not dependencies and _DEPENDENT_PREFIX_RE.match(text) and drafts:
                dependencies = (len(drafts) - 1,)
            elif not dependencies and text.startswith("本条第") and drafts:
                dependencies = (first_unit_index,)
            elif (
                not dependencies
                and active_lead is not None
                and not previous_was_list
            ):
                unit_type = "list_item"
                dependencies = (active_lead,)
                active_lead = None
            if previous_was_list:
                active_lead = None
            previous_was_list = False
        drafts.append(_DraftSpan(start, end, unit_type, dependencies))
    drafts = tuple(drafts)
    if not enforce_safety:
        return drafts
    if _fallback_reason(content, drafts) is not None:
        return (_DraftSpan(0, len(content), "full"),)
    return drafts


def _fallback_reason(content, drafts):
    dependent_indexes = {
        dependency
        for draft in drafts
        for dependency in draft.dependency_indexes
    }
    for index, draft in enumerate(drafts):
        text = content[draft.start : draft.end].strip()
        if text.startswith(("，", "。", "；", "：", "、", "）", ")", "》", "”", "’")):
            return f"bad_start:{index}"
        if index < len(drafts) - 1 and text.endswith(
            ("，", "、", "（", "(", "《", "“", "‘")
        ):
            return f"mid_clause_end:{index}"
        marker = _LIST_MARKER_RE.fullmatch(text.rstrip("，,；;。、"))
        if marker is not None:
            return f"marker_only:{index}"
        if draft.unit_type == "list_item" and not draft.dependency_indexes:
            return f"orphan_list_item:{index}"
        if draft.unit_type == "lead" and index not in dependent_indexes:
            return f"orphan_lead:{index}"
    return None


def split_article(article, *, splitter_version=SPLITTER_VERSION):
    """将一条法条拆成确定性、连续原文子单元。"""
    if not isinstance(article, LegalArticle):
        raise TypeError("article 必须是 LegalArticle")
    _require_non_blank("splitter_version", splitter_version)
    if not _VERSION_RE.fullmatch(splitter_version):
        raise ValueError("splitter_version 格式无效")
    drafts = _draft_units(article.content)
    if not drafts:
        raise ValueError("法条正文不能生成空的证据子单元")

    unit_ids = tuple(
        make_unit_id(article.chunk_id, splitter_version, draft.start, draft.end)
        for draft in drafts
    )
    units = []
    for index, (draft, unit_id) in enumerate(zip(drafts, unit_ids)):
        units.append(
            EvidenceUnit(
                unit_id=unit_id,
                parent_chunk_id=article.chunk_id,
                unit_index=index,
                unit_type=draft.unit_type,
                text=article.content[draft.start : draft.end],
                start_char=draft.start,
                end_char=draft.end,
                dependency_unit_ids=tuple(
                    unit_ids[dependency_index]
                    for dependency_index in draft.dependency_indexes
                ),
                splitter_version=splitter_version,
            )
        )
    validate_article_units(article, units, splitter_version=splitter_version)
    return tuple(units)


def validate_article_units(article, units, *, splitter_version=SPLITTER_VERSION):
    """校验一条父法条的子单元身份、区间覆盖和依赖无环。"""
    if not isinstance(article, LegalArticle):
        raise TypeError("article 必须是 LegalArticle")
    values = tuple(units)
    if not values:
        raise ValueError("每个父法条必须至少包含一个子单元")
    ids = {unit.unit_id for unit in values}
    if len(ids) != len(values):
        raise ValueError("同一父法条内存在重复 unit_id")
    previous_end = 0
    covered = [False] * len(article.content)
    for index, unit in enumerate(values):
        if not isinstance(unit, EvidenceUnit):
            raise TypeError("units 必须由 EvidenceUnit 组成")
        if unit.parent_chunk_id != article.chunk_id:
            raise ValueError("子单元 parent_chunk_id 与父法条不一致")
        if unit.splitter_version != splitter_version:
            raise ValueError("子单元 splitter_version 不一致")
        if unit.unit_index != index:
            raise ValueError("unit_index 必须从 0 连续递增")
        if unit.start_char < previous_end:
            raise ValueError("同一父法条的子单元区间不能重叠")
        if unit.end_char > len(article.content):
            raise ValueError("子单元区间超出父法条正文")
        if article.content[unit.start_char : unit.end_char] != unit.text:
            raise ValueError("子单元文本不能由父法条区间精确复原")
        if not unit.text.strip():
            raise ValueError("子单元文本不能为空白")
        for position in range(unit.start_char, unit.end_char):
            covered[position] = True
        previous_end = unit.end_char
        for dependency in unit.dependency_unit_ids:
            if dependency not in ids:
                raise ValueError("子单元依赖必须指向同一父法条中的已知单元")

    uncovered = [
        index
        for index, char in enumerate(article.content)
        if not char.isspace() and not covered[index]
    ]
    if uncovered:
        raise ValueError("子单元没有覆盖父法条全部非空白字符")

    dependencies = {
        unit.unit_id: tuple(unit.dependency_unit_ids) for unit in values
    }
    visiting = set()
    visited = set()

    def visit(unit_id):
        if unit_id in visiting:
            raise ValueError("子单元依赖不能形成环")
        if unit_id in visited:
            return
        visiting.add(unit_id)
        for dependency in dependencies[unit_id]:
            visit(dependency)
        visiting.remove(unit_id)
        visited.add(unit_id)

    for unit_id in dependencies:
        visit(unit_id)


__all__ = [
    "EvidenceUnit",
    "EvidenceUnitRepository",
    "MIN_NESTED_LIST_SPLIT_CHARS",
    "MIN_SENTENCE_SPLIT_CHARS",
    "SCHEMA_VERSION",
    "SPLITTER_VERSION",
    "UNIT_TYPES",
    "make_unit_id",
    "split_article",
    "validate_article_units",
]
