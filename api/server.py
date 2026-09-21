"""装配本地法律 RAG，并以 FastAPI 同时提供 API 与 React 页面。"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import uvicorn

from api.main import create_app
from minimind.trainer.chat_current_law_rag import load_application


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _environment_flag(name: str, default: bool) -> bool:
    """读取简单布尔环境变量。"""
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} 必须是 true/false 或 1/0")


def _sha256_file(path: Path) -> str:
    """计算本地模型权重摘要，供现有严格加载器校验。"""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ServerSettings:
    """Web 展示服务所需的本地运行时路径和启动参数。"""

    weights: Path
    tokenizer_path: Path
    article_index: Path
    artifact_dir: Path
    frontend_dir: Path
    device: str
    enable_external_llm: bool
    host: str
    port: int
    retrieval_mode: str = "article"
    evidence_unit_dir: Path | None = None
    weights_sha256: str | None = None

    @classmethod
    def from_env(cls) -> "ServerSettings":
        """读取环境覆盖，并为当前项目布局提供可直接运行的默认值。"""
        return cls(
            weights=Path(
                os.getenv(
                    "LAWMIND_WEIGHTS",
                    PROJECT_ROOT / "minimind" / "checkpoints" / "rag_epoch_2.pth",
                )
            ),
            tokenizer_path=Path(
                os.getenv(
                    "LAWMIND_TOKENIZER_PATH",
                    PROJECT_ROOT / "minimind" / "model",
                )
            ),
            article_index=Path(
                os.getenv(
                    "LAWMIND_ARTICLE_INDEX",
                    PROJECT_ROOT
                    / "rag"
                    / "chunk"
                    / "article_index_with_department_rules.jsonl",
                )
            ),
            artifact_dir=Path(
                os.getenv(
                    "LAWMIND_ARTIFACT_DIR",
                    PROJECT_ROOT / "rag" / "retrieval" / "artifacts_department_rules",
                )
            ),
            frontend_dir=Path(
                os.getenv(
                    "LAWMIND_FRONTEND_DIR",
                    PROJECT_ROOT / "frontend" / "dist",
                )
            ),
            device=os.getenv(
                "LAWMIND_DEVICE",
                "cuda:0" if torch.cuda.is_available() else "cpu",
            ),
            enable_external_llm=_environment_flag(
                "LAWMIND_ENABLE_EXTERNAL_LLM",
                True,
            ),
            retrieval_mode=os.getenv("LAWMIND_RETRIEVAL_MODE", "article"),
            evidence_unit_dir=Path(
                os.getenv(
                    "LAWMIND_EVIDENCE_UNIT_DIR",
                    PROJECT_ROOT
                    / "rag"
                    / "retrieval"
                    / "evidence_unit_artifacts"
                    / "v1",
                )
            ),
            host=os.getenv("LAWMIND_HOST", "127.0.0.1"),
            port=int(os.getenv("LAWMIND_PORT", "8000")),
            weights_sha256=os.getenv("LAWMIND_WEIGHTS_SHA256"),
        )

    def validate(self) -> None:
        """在加载大模型前报告缺失的本地资源。"""
        for label, path in (
            ("模型权重", self.weights),
            ("Tokenizer", self.tokenizer_path),
            ("法条索引", self.article_index),
            ("检索产物", self.artifact_dir),
            ("前端构建产物", self.frontend_dir),
        ):
            if not path.exists():
                raise FileNotFoundError(f"缺少{label}：{path}")
        if not 1 <= self.port <= 65535:
            raise ValueError("LAWMIND_PORT 必须位于 1 到 65535 之间")
        if self.retrieval_mode not in {"article", "evidence_unit"}:
            raise ValueError(
                "LAWMIND_RETRIEVAL_MODE 必须是 article 或 evidence_unit"
            )
        if self.retrieval_mode == "evidence_unit":
            if self.evidence_unit_dir is None:
                raise ValueError(
                    "evidence_unit 模式必须提供 LAWMIND_EVIDENCE_UNIT_DIR"
                )
            for label, path in (
                ("子单元 sidecar", self.evidence_unit_dir / "evidence_units.jsonl"),
                ("子单元索引目录", self.evidence_unit_dir / "indexes"),
            ):
                if not path.exists():
                    raise FileNotFoundError(f"缺少{label}：{path}")


def build_web_app(settings: ServerSettings, *, loader=load_application):
    """装配一次共享 RAG 实例并创建同源 Web 应用。"""
    if not isinstance(settings, ServerSettings):
        raise TypeError("settings 必须是 ServerSettings")
    settings.validate()
    weights_sha256 = (
        settings.weights_sha256.strip().lower()
        if settings.weights_sha256
        else _sha256_file(settings.weights)
    )
    application = loader(
        weights=settings.weights,
        weights_sha256=weights_sha256,
        tokenizer_path=settings.tokenizer_path,
        article_index=settings.article_index,
        artifact_dir=settings.artifact_dir,
        device_name=settings.device,
        enable_external_llm=settings.enable_external_llm,
        retrieval_mode=settings.retrieval_mode,
        evidence_unit_dir=settings.evidence_unit_dir,
    )
    return create_app(application, frontend_dir=settings.frontend_dir)


def main() -> None:
    """启动完整 LawMind Web 展示服务。"""
    settings = ServerSettings.from_env()
    app = build_web_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()


__all__ = ["ServerSettings", "build_web_app", "main"]
