"""构造模型可完整接收的法条检索文本窗口。"""

from dataclasses import dataclass
import re

from rag.knowledge import LegalArticle


@dataclass(frozen=True)
class RetrievalWindow:
    """一条 canonical 法条在单个检索模型中的临时窗口。"""

    chunk_id: str
    window_index: int
    start_token: int
    end_token: int
    content: str
    text: str


def format_article_no(article_no):
    """把 canonical 条号转换为中文引用中的规范条号。"""
    main, separator, suffix = article_no.partition("之")
    return f"第{main}条之{suffix}" if separator else f"第{main}条"


def format_article_heading(article):
    """返回包含正式法名和规范条号的检索标题。"""
    if not isinstance(article, LegalArticle):
        raise TypeError("article 必须是 LegalArticle")
    return format_article_heading_fields(article.law_name, article.article_no)


def format_article_heading_fields(law_name, article_no):
    """按字段构造检索标题，供离线索引构建使用。"""
    if not isinstance(law_name, str) or not law_name.strip():
        raise ValueError("law_name 必须是非空字符串")
    if not isinstance(article_no, str) or not article_no.strip():
        raise ValueError("article_no 必须是非空字符串")
    return f"《{law_name}》{format_article_no(article_no)}"


def _encode_without_special_tokens(tokenizer, text):
    try:
        return tuple(
            tokenizer.encode(
                text,
                add_special_tokens=False,
                verbose=False,
            )
        )
    except TypeError:
        return tuple(tokenizer.encode(text, add_special_tokens=False))


def _encode_with_offsets(tokenizer, text):
    """编码正文并在 fast tokenizer 可用时保留原文字符偏移。"""
    if callable(tokenizer):
        try:
            encoded = tokenizer(
                text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                verbose=False,
            )
            token_ids = tuple(encoded["input_ids"])
            offsets = tuple(tuple(item) for item in encoded["offset_mapping"])
            if len(token_ids) == len(offsets):
                return token_ids, offsets
        except (KeyError, TypeError, ValueError, NotImplementedError):
            pass
    return _encode_without_special_tokens(tokenizer, text), None


def _model_input_length(tokenizer, *, query, document):
    if query is None:
        return len(tokenizer.encode(document, add_special_tokens=True))
    return len(
        tokenizer.encode(
            query,
            text_pair=document,
            add_special_tokens=True,
        )
    )


def _semantic_boundaries(tokenizer, token_slice):
    """返回 token 片段内可安全切分的句末或段落边界。"""
    decoded = tokenizer.decode(
        token_slice,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    character_boundaries = [
        match.end()
        for match in re.finditer(r"(?:[。！？；]|\n)\s*", decoded)
    ]
    positions = []
    if callable(tokenizer):
        try:
            encoded = tokenizer(
                decoded,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            offsets = encoded["offset_mapping"]
        except (KeyError, TypeError, ValueError, NotImplementedError):
            offsets = None
        if offsets is not None:
            for boundary in character_boundaries:
                position = sum(
                    1 for start, end in offsets if end and end <= boundary
                )
                if 0 < position < len(token_slice):
                    positions.append(position)
            return tuple(sorted(set(positions)))
    for boundary in character_boundaries:
        prefix = decoded[:boundary].rstrip()
        if prefix:
            position = len(_encode_without_special_tokens(tokenizer, prefix))
            if 0 < position < len(token_slice):
                positions.append(position)
    return tuple(sorted(set(positions)))


def _semantic_end(tokenizer, content_tokens, start, end):
    """优先在窗口后半段的语义边界结束，避免产生过短窗口。"""
    if end >= len(content_tokens):
        return end
    token_slice = content_tokens[start:end]
    boundaries = _semantic_boundaries(tokenizer, token_slice)
    minimum_progress = max(1, len(token_slice) // 2)
    candidates = [position for position in boundaries if position >= minimum_progress]
    return start + candidates[-1] if candidates else end


def _semantic_overlap_start(tokenizer, content_tokens, start, end, overlap_tokens):
    """让下一窗口从完整语义单元起点开始，必要时才退化为 token 重叠。"""
    if overlap_tokens == 0:
        return end
    fallback = end - overlap_tokens
    if fallback <= start:
        return fallback
    token_slice = content_tokens[start:end]
    boundaries = _semantic_boundaries(tokenizer, token_slice)
    candidates = [start + position for position in boundaries if start + position <= fallback]
    if candidates:
        return candidates[-1]
    decoded = tokenizer.decode(
        token_slice,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).rstrip()
    if decoded.endswith(("。", "！", "？", "；", "\n")):
        return end
    return fallback


def build_retrieval_windows(
    article,
    tokenizer,
    *,
    max_length,
    overlap_tokens,
    query=None,
):
    """按模型 tokenizer 完整覆盖正文，并为固定输入保留 token 空间。"""
    if not isinstance(article, LegalArticle):
        raise TypeError("article 必须是 LegalArticle")
    if not isinstance(max_length, int) or isinstance(max_length, bool) or max_length <= 0:
        raise ValueError("max_length 必须是正整数")
    if (
        not isinstance(overlap_tokens, int)
        or isinstance(overlap_tokens, bool)
        or overlap_tokens < 0
    ):
        raise ValueError("overlap_tokens 必须是非负整数")
    if query is not None and (not isinstance(query, str) or not query.strip()):
        raise ValueError("query 必须是非空字符串或 None")

    heading = format_article_heading(article) + "\n"
    heading_tokens = _encode_without_special_tokens(tokenizer, heading)
    query_tokens = () if query is None else _encode_without_special_tokens(tokenizer, query)
    special_tokens = tokenizer.num_special_tokens_to_add(pair=query is not None)
    content_capacity = (
        max_length - len(heading_tokens) - len(query_tokens) - special_tokens
    )
    if content_capacity <= 0:
        raise ValueError("固定检索输入已经占满模型上下文")
    if overlap_tokens >= content_capacity:
        raise ValueError("overlap_tokens 必须小于正文窗口容量")

    content_tokens, content_offsets = _encode_with_offsets(tokenizer, article.content)
    if not content_tokens:
        raise ValueError("法条正文经过 tokenizer 后不能为空")
    windows = []
    start = 0
    window_index = 0
    while start < len(content_tokens):
        end = min(start + content_capacity, len(content_tokens))
        semantic_end = _semantic_end(tokenizer, content_tokens, start, end)
        if semantic_end > start and semantic_end < end:
            end = semantic_end
        while True:
            token_slice = content_tokens[start:end]
            if content_offsets is None:
                window_content = tokenizer.decode(
                    token_slice,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            else:
                char_start = content_offsets[start][0]
                char_end = content_offsets[end - 1][1]
                window_content = article.content[char_start:char_end]
            window_text = heading + window_content
            if _model_input_length(
                tokenizer, query=query, document=window_text
            ) <= max_length:
                break
            end -= 1
            if end <= start:
                raise ValueError("模型 tokenizer 无法容纳非空正文窗口")
        windows.append(
            RetrievalWindow(
                chunk_id=article.chunk_id,
                window_index=window_index,
                start_token=start,
                end_token=end,
                content=window_content,
                text=window_text,
            )
        )
        if end == len(content_tokens):
            break
        next_start = _semantic_overlap_start(
            tokenizer,
            content_tokens,
            start,
            end,
            overlap_tokens,
        )
        if next_start <= start:
            raise ValueError("overlap_tokens 使正文窗口无法向前移动")
        start = next_start
        window_index += 1
    return tuple(windows)


__all__ = [
    "RetrievalWindow",
    "build_retrieval_windows",
    "format_article_heading",
    "format_article_heading_fields",
    "format_article_no",
]
