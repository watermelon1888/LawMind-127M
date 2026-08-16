"""构造模型可完整接收的法条检索文本窗口。"""

from dataclasses import dataclass

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
    return tuple(tokenizer.encode(text, add_special_tokens=False))


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

    content_tokens = _encode_without_special_tokens(tokenizer, article.content)
    if not content_tokens:
        raise ValueError("法条正文经过 tokenizer 后不能为空")
    windows = []
    start = 0
    window_index = 0
    while start < len(content_tokens):
        end = min(start + content_capacity, len(content_tokens))
        while True:
            token_slice = content_tokens[start:end]
            window_content = tokenizer.decode(
                token_slice,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
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
        next_start = end - overlap_tokens
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
