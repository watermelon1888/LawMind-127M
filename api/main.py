"""法律 RAG 的最小 FastAPI 入口。

API 层只负责输入校验、调用已装配的 ``LegalRAG`` 和结果序列化，
不在这里加载模型、索引或复制核心路由逻辑。
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag.core import AnswerStatus, LegalRAG, LegalRAGResult


class AnswerRequest(BaseModel):
    """统一问答请求。"""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """拒绝只包含空白字符的问题，并保留用户原始文本。"""
        if not value.strip():
            raise ValueError("query 不能只包含空白字符")
        return value


def _answer_mode(result: LegalRAGResult) -> Optional[str]:
    """从安全审计事件中提取回答模式。"""
    answer_mode = None
    for event in result.audit_trace.events:
        if event.stage != "route_decided":
            continue
        details = event.to_dict()["details"]
        if details.get("answer_mode") is not None:
            answer_mode = details["answer_mode"]
    return answer_mode


def _display_answer(result: LegalRAGResult) -> str:
    """检索回答只交付已校验归纳，完整法条由结构化 evidence 承载。"""
    if (
        result.status is AnswerStatus.RETRIEVED_EVIDENCE
        and result.model_answer is not None
    ):
        return result.model_answer.summary
    return result.rendered_answer.to_text()


def _serialize_result(result: LegalRAGResult) -> dict[str, Any]:
    """将核心结果投影为稳定的 HTTP JSON 结构。"""
    if not isinstance(result, LegalRAGResult):
        raise TypeError("LegalRAG.answer 必须返回 LegalRAGResult")

    route = result.audit_trace.final_route
    if route is None:
        raise ValueError("LegalRAGResult.audit_trace 缺少 final_route")
    answer_mode = _answer_mode(result)
    return {
        "request_id": result.audit_trace.trace_id,
        "query": result.query,
        "route": route.value,
        "status": result.status.value,
        "answer_mode": answer_mode,
        "answer": _display_answer(result),
        "candidate_answer": result.candidate_answer,
        "evidence": [
            {
                "law_name": item.law_name,
                "article_no": item.article_no,
                "content": item.content,
            }
            for item in result.evidence
        ],
        "unanswered_reason": (
            None
            if result.unanswered_reason is None
            else result.unanswered_reason.value
        ),
        "diagnostic_code": result.diagnostic_code,
        "audit_trace": result.audit_trace.to_dict(),
    }


def create_app(
    application: Optional[LegalRAG] = None,
    *,
    frontend_dir: Optional[Path] = None,
) -> FastAPI:
    """创建 FastAPI 应用，可注入步骤 13 装配好的法律 RAG 实例。"""
    if application is not None and not callable(getattr(application, "answer", None)):
        raise TypeError("application 必须提供可调用的 answer")

    app = FastAPI(title="法律 RAG API", version="1.0")
    app.state.application = application
    app.state.answer_lock = Lock()

    @app.post("/v1/answer")
    def answer(request: AnswerRequest) -> dict[str, Any]:
        """处理单次独立法律问答请求。"""
        current = app.state.application
        if current is None:
            raise HTTPException(status_code=503, detail="法律 RAG 尚未就绪")
        with app.state.answer_lock:
            result = current.answer(request.query)
        return _serialize_result(result)

    @app.get("/health/live")
    def live() -> dict[str, str]:
        """返回进程存活状态，不触发模型调用。"""
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready() -> dict[str, str]:
        """返回应用依赖是否已注入，不触发真实问答。"""
        if app.state.application is None:
            raise HTTPException(status_code=503, detail="法律 RAG 尚未装配")
        return {"status": "ready"}

    static_directory = (
        Path(frontend_dir)
        if frontend_dir is not None
        else Path(__file__).resolve().parents[1] / "frontend" / "dist"
    )
    if static_directory.is_dir():
        app.mount(
            "/",
            StaticFiles(directory=static_directory, html=True),
            name="frontend",
        )

    return app


# 提供可直接被 uvicorn 导入的入口；生产启动时应通过 create_app 注入实例。
app = create_app()


__all__ = ["AnswerRequest", "app", "create_app"]
