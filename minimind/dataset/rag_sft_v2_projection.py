"""将已审核的 RAG-SFT v2 记录投影为训练对话，并审计固定上下文预算。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rag.answering import (
    AnswerProtocolError,
    EvidencePackage,
    RAG_MAX_OUTPUT_TOKENS,
    SYSTEM_PROMPT,
    build_answer_prompt,
    parse_and_validate_answer,
)
from rag.core import Evidence
from rag.knowledge import LegalArticle

from .rag_sft_v2_contract import (
    RagSftV2ContractError,
    MAX_VISIBLE_EVIDENCE,
    derive_citations,
    derive_clean_visible_chunk_ids,
    validate_hn_variant,
)


CONTEXT_LIMIT = 768
MAX_OUTPUT_TOKENS = RAG_MAX_OUTPUT_TOKENS
MAX_PROMPT_TOKENS = CONTEXT_LIMIT - MAX_OUTPUT_TOKENS


class RagSftV2ProjectionError(ValueError):
    """RAG-SFT v2 的对话投影或固定预算审计失败。"""


@dataclass(frozen=True)
class ProjectedRagSftV2Record:
    """不含隐藏 claims 的单条训练对话及其程序侧审计元数据。"""

    record_id: str
    query_id: str
    variant: str
    visible_chunk_ids: tuple[str, ...]
    required_chunk_ids: tuple[str, ...]
    citations: tuple[str, ...]
    conversations: tuple[dict[str, str], ...]

    def training_conversations(self) -> list[dict[str, str]]:
        """返回可供后续物化器写入的纯三轮对话，不暴露隐藏审计字段。"""

        return [dict(message) for message in self.conversations]


@dataclass(frozen=True)
class ProjectedRagSftV2Audit:
    """真实 chat template 下的单条固定长度与标签审计结果。"""

    prompt_tokens: int
    assistant_label_tokens: int
    total_tokens: int
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


def _non_blank_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RagSftV2ProjectionError(f"{field} 必须是非空字符串")
    return value


def _unique_chunk_ids(value: Sequence[str], field: str, maximum: int) -> tuple[str, ...]:
    values = tuple(value)
    if not 1 <= len(values) <= maximum or len(values) != len(set(values)):
        raise RagSftV2ProjectionError(f"{field} 必须包含一至{maximum}条不重复法条")
    if any(not isinstance(item, str) or not item.strip() for item in values):
        raise RagSftV2ProjectionError(f"{field} 必须是非空字符串数组")
    return values


def _article_evidence(
    article_by_chunk_id: Mapping[str, LegalArticle], visible_chunk_ids: Sequence[str]
) -> tuple[Evidence, ...]:
    evidence: list[Evidence] = []
    for chunk_id in visible_chunk_ids:
        article = article_by_chunk_id.get(chunk_id)
        if not isinstance(article, LegalArticle) or article.chunk_id != chunk_id:
            raise RagSftV2ProjectionError(f"缺少完整法条: {chunk_id}")
        evidence.append(
            Evidence(
                law_name=article.law_name,
                article_no=article.article_no,
                content=article.content,
            )
        )
    return tuple(evidence)


def project_rag_sft_v2_record(
    canonical: Mapping[str, object],
    article_by_chunk_id: Mapping[str, LegalArticle],
    *,
    hn_variant: Mapping[str, object] | None = None,
) -> ProjectedRagSftV2Record:
    """从已通过语义审核的 canonical 或正式 HN variant 确定性生成三轮对话。"""

    query_id = _non_blank_string(canonical.get("query_id"), "canonical.query_id")
    query = _non_blank_string(canonical.get("query_original"), "canonical.query_original")
    summary = _non_blank_string(canonical.get("summary"), "canonical.summary")
    required_chunk_ids = _unique_chunk_ids(
        derive_clean_visible_chunk_ids(canonical), "required_chunk_ids", 3
    )

    if hn_variant is None:
        variant = "clean"
        record_id = f"{query_id}:clean"
        visible_chunk_ids = required_chunk_ids
    else:
        try:
            approved_hn = validate_hn_variant(hn_variant, canonical)
        except RagSftV2ContractError as error:
            raise RagSftV2ProjectionError("HN variant 不满足 v2 契约") from error
        variant = "hard_negative"
        record_id = approved_hn["variant_id"]
        visible_chunk_ids = _unique_chunk_ids(
            approved_hn["visible_chunk_ids"],
            "HN visible_chunk_ids",
            MAX_VISIBLE_EVIDENCE,
        )

    citations = derive_citations(visible_chunk_ids, required_chunk_ids)
    package = EvidencePackage(
        query=query,
        evidence=_article_evidence(article_by_chunk_id, visible_chunk_ids),
    )
    assistant = json.dumps(
        {"summary": summary, "citations": list(citations)},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    try:
        parse_and_validate_answer(package, assistant)
    except AnswerProtocolError as error:
        raise RagSftV2ProjectionError("canonical summary 不满足两字段回答协议") from error
    messages = build_answer_prompt(package)
    conversations = tuple(
        [*messages, {"role": "assistant", "content": assistant}]
    )
    if [message["role"] for message in conversations] != ["system", "user", "assistant"]:
        raise RagSftV2ProjectionError("投影后的 conversations 必须为 system/user/assistant")
    if conversations[0]["content"] != SYSTEM_PROMPT:
        raise RagSftV2ProjectionError("投影后的 system prompt 身份不一致")
    return ProjectedRagSftV2Record(
        record_id=record_id,
        query_id=query_id,
        variant=variant,
        visible_chunk_ids=visible_chunk_ids,
        required_chunk_ids=required_chunk_ids,
        citations=citations,
        conversations=conversations,
    )


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        values = encoded.get("input_ids") if isinstance(encoded, dict) else encoded.input_ids
    except Exception as error:
        raise RagSftV2ProjectionError("tokenizer 编码失败") from error
    if not isinstance(values, list) or any(type(value) is not int for value in values):
        raise RagSftV2ProjectionError("tokenizer 未返回整数 input_ids")
    return values


def _render(tokenizer: Any, conversations: Sequence[Mapping[str, str]], *, generation: bool) -> str:
    if not callable(getattr(tokenizer, "apply_chat_template", None)):
        raise RagSftV2ProjectionError("tokenizer 缺少 apply_chat_template")
    try:
        rendered = tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=generation,
            tools=None,
            open_thinking=False,
        )
    except Exception as error:
        raise RagSftV2ProjectionError("chat template 渲染失败") from error
    if not isinstance(rendered, str) or not rendered:
        raise RagSftV2ProjectionError("chat template 未返回非空字符串")
    return rendered


def audit_projected_rag_sft_v2_record(
    record: ProjectedRagSftV2Record, tokenizer: Any
) -> ProjectedRagSftV2Audit:
    """以实际运行 chat template 检查 618/150/768 预算和 assistant-only labels。"""

    if not isinstance(record, ProjectedRagSftV2Record):
        raise TypeError("record 必须是 ProjectedRagSftV2Record")
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if type(pad_token_id) is not int:
        raise RagSftV2ProjectionError("tokenizer 缺少整数 pad_token_id")

    prompt = _render(tokenizer, record.conversations[:2], generation=True)
    full = _render(tokenizer, record.conversations, generation=False)
    prompt_ids = _token_ids(tokenizer, prompt)
    full_ids = _token_ids(tokenizer, full)
    if not full_ids[: len(prompt_ids)] == prompt_ids:
        raise RagSftV2ProjectionError("训练模板没有保留运行时 prompt 前缀")
    assistant_ids = full_ids[len(prompt_ids) :]
    if not assistant_ids:
        raise RagSftV2ProjectionError("assistant-only labels 为空")
    if len(prompt_ids) > MAX_PROMPT_TOKENS:
        raise RagSftV2ProjectionError(
            f"运行时 prompt 超过 {MAX_PROMPT_TOKENS} token 预算"
        )
    if len(assistant_ids) > MAX_OUTPUT_TOKENS:
        raise RagSftV2ProjectionError(
            f"assistant JSON 超过 {MAX_OUTPUT_TOKENS} token 输出预算"
        )
    if len(full_ids) > CONTEXT_LIMIT:
        raise RagSftV2ProjectionError("训练 conversations 超过 768 token，禁止截断")

    input_ids = [*full_ids, *([pad_token_id] * (CONTEXT_LIMIT - len(full_ids)))]
    labels = [-100] * CONTEXT_LIMIT
    labels[len(prompt_ids) : len(full_ids)] = assistant_ids
    return ProjectedRagSftV2Audit(
        prompt_tokens=len(prompt_ids),
        assistant_label_tokens=len(assistant_ids),
        total_tokens=len(full_ids),
        input_ids=tuple(input_ids),
        labels=tuple(labels),
    )


__all__ = [
    "CONTEXT_LIMIT",
    "MAX_OUTPUT_TOKENS",
    "MAX_PROMPT_TOKENS",
    "ProjectedRagSftV2Audit",
    "ProjectedRagSftV2Record",
    "RagSftV2ProjectionError",
    "audit_projected_rag_sft_v2_record",
    "project_rag_sft_v2_record",
]
