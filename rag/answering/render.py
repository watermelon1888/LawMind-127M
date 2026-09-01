"""把可信内部对象投影为稳定、可审计的中文回答。"""

from rag.answering.evidence import EvidencePackage
from rag.core.contracts import (
    Evidence,
    ModelAnswer,
    RenderedAnswer,
    RenderedEvidence,
    UnansweredReason,
)


_CLARIFICATION_MESSAGE = (
    "现有信息不足以安全确定需要回答的具体法律问题。"
    "请补充或收窄问题，只保留争议主体、具体行为和关键事实；"
    "如需查询指定法条，请提供法律名称和条号。"
)

_REFUSAL_MESSAGES = {
    UnansweredReason.NON_LEGAL: "该问题不属于当前法律问答范围。",
    UnansweredReason.TIME_SENSITIVE: (
        "当前知识库仅支持现行有效法律，不能可靠回答历史版本内容或过去事实应适用哪一版法律。"
        "为避免用现行法替代当时规定，本次暂不作答。"
    ),
    UnansweredReason.UNSUPPORTED_LEGAL_SOURCE: (
        "当前知识库未收录所指定的法律规范或案例来源。"
        "为避免引用无法核验的依据，本次暂不作答。"
    ),
    UnansweredReason.UNSUPPORTED_LEGAL_TASK: (
        "当前系统仅提供现行法条的展示和简短归纳，不代写法律文书、"
        "不预测案件结果，也不提供规避监管的方案。"
    ),
    UnansweredReason.NO_VERIFIABLE_EVIDENCE: (
        "当前未检索到可供核验的现行有效法条候选。"
        "为避免编造法条或条号，本次暂不作答。"
    ),
}

_FAILURE_STAGE_MESSAGES = {
    "exact_lookup_failed": "本次精确条文查询未能完成。",
    "semantic_retrieval_failed": "本次法律条文检索未能完成。",
    "evidence_packaging_failed": "本次法律证据构包未能完成。",
    "generation_failed": "本次法律回答模型生成未能完成。",
    "output_validation_failed": "本次法律回答未通过输出协议校验。",
}

_UNKNOWN_FAILURE_MESSAGE = "本次法律证据处理未能完整完成。"
_FAILURE_MESSAGE_SUFFIX = "为避免依据不完整时作出结论，本次暂不作答。"


def _render_evidence(evidence):
    if not isinstance(evidence, Evidence):
        raise TypeError("待展示证据必须是 Evidence")
    return RenderedEvidence(
        law_name=evidence.law_name,
        article_no=evidence.article_no,
        content=evidence.content,
    )


def render_exact_lookup(evidence):
    """展示程序精确定位的一至三条完整法条。"""
    evidence = tuple(evidence)
    if not 1 <= len(evidence) <= 3:
        raise ValueError("精确查条必须包含一至三条 Evidence")
    return RenderedAnswer(evidence=tuple(_render_evidence(item) for item in evidence))


def render_semantic_answer(package, answer):
    """按规范化 citations 展示模型实际采用的完整法条。"""
    if not isinstance(package, EvidencePackage):
        raise TypeError("package 必须是 EvidencePackage")
    if not isinstance(answer, ModelAnswer):
        raise TypeError("answer 必须是 ModelAnswer")
    cited_positions = {int(citation[1:]) - 1 for citation in answer.citations}
    selected = tuple(
        _render_evidence(item)
        for index, item in enumerate(package.evidence)
        if index in cited_positions
    )
    return RenderedAnswer(evidence=selected, summary=answer.summary)


def render_clarification():
    """返回所有澄清路径共用的固定话术。"""
    return RenderedAnswer(message=_CLARIFICATION_MESSAGE)


def render_refusal(reason):
    """按稳定未作答原因返回固定拒答话术。"""
    if not isinstance(reason, UnansweredReason):
        raise TypeError("reason 必须是 UnansweredReason")
    try:
        message = _REFUSAL_MESSAGES[reason]
    except KeyError as exc:
        raise ValueError("该原因不属于策略拒答") from exc
    return RenderedAnswer(message=message)


def render_failure(diagnostic_code):
    """按处理失败阶段返回安全话术和稳定诊断标识。"""
    if not isinstance(diagnostic_code, str) or not diagnostic_code.strip():
        raise TypeError("diagnostic_code 必须是非空字符串")
    stage_message = _FAILURE_STAGE_MESSAGES.get(
        diagnostic_code, _UNKNOWN_FAILURE_MESSAGE
    )
    return RenderedAnswer(
        message=(
            f"{stage_message}{_FAILURE_MESSAGE_SUFFIX}\n"
            f"诊断标识：{diagnostic_code}"
        )
    )


__all__ = [
    "render_clarification",
    "render_exact_lookup",
    "render_failure",
    "render_refusal",
    "render_semantic_answer",
]
