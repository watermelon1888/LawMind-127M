"""集中装配法律 RAG 应用及其运行时依赖。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from rag.answering import (
    AnswerPromptTokenCounter,
    EvidenceBundlePackager,
    EvidencePackager,
    RAG_MAX_OUTPUT_TOKENS,
)
from rag.core import CurrentLawRAG, LegalRAG
from rag.external import OpenAICompatibleLLM
from rag.knowledge import ArticleRepository, EvidenceUnitRepository
from rag.retrieval import (
    SemanticRetrievalConfig,
    load_evidence_unit_retriever,
    load_semantic_retriever,
)
from rag.retrieval.loader import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL


PRODUCTION_RETRIEVAL_CONFIG = SemanticRetrievalConfig(
    dense_top_k=30,
    sparse_top_k=30,
    rrf_k=4,
    candidate_pool=20,
    top_k=5,
)
RETRIEVAL_MODES = frozenset({"article", "evidence_unit"})


@dataclass(frozen=True)
class RAGApplicationConfig:
    """统一应用装配所需的路径、模型和预算配置。"""

    article_index: Path
    artifact_dir: Path
    context_limit: int
    max_output_tokens: int = RAG_MAX_OUTPUT_TOKENS
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    reranker_model: str = DEFAULT_RERANKER_MODEL
    device: str = "auto"
    enable_external_llm: bool = False
    retrieval_mode: str = "article"
    evidence_unit_dir: Optional[Path] = None

    def __post_init__(self):
        object.__setattr__(self, "article_index", Path(self.article_index))
        object.__setattr__(self, "artifact_dir", Path(self.artifact_dir))
        if self.evidence_unit_dir is not None:
            object.__setattr__(
                self,
                "evidence_unit_dir",
                Path(self.evidence_unit_dir),
            )
        for name in ("context_limit", "max_output_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        for name in ("embedding_model", "reranker_model", "device"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} 必须是非空字符串")
        if not isinstance(self.enable_external_llm, bool):
            raise TypeError("enable_external_llm 必须是布尔值")
        if self.retrieval_mode not in RETRIEVAL_MODES:
            raise ValueError("retrieval_mode 必须是 article 或 evidence_unit")
        if self.retrieval_mode == "evidence_unit" and self.evidence_unit_dir is None:
            raise ValueError("evidence_unit 模式必须提供 evidence_unit_dir")


def build_current_law_rag(
    config: RAGApplicationConfig,
    *,
    generate: Callable[..., str],
    tokenizer: Optional[Any] = None,
    count_prompt_tokens: Optional[Callable[..., int]] = None,
    external_llm=None,
    repository: Optional[ArticleRepository] = None,
    semantic_retriever=None,
    unit_repository: Optional[EvidenceUnitRepository] = None,
    unit_scorer=None,
) -> LegalRAG:
    """创建可直接调用的 `CurrentLawRAG`，并集中管理依赖装配。"""
    if not isinstance(config, RAGApplicationConfig):
        raise TypeError("config 必须是 RAGApplicationConfig")
    if not callable(generate):
        raise TypeError("generate 必须是可调用对象")
    if tokenizer is not None and count_prompt_tokens is not None:
        raise ValueError("tokenizer 与 count_prompt_tokens 只能提供一个")
    if tokenizer is not None:
        count_prompt_tokens = AnswerPromptTokenCounter(tokenizer)
    if not callable(count_prompt_tokens):
        raise TypeError("必须提供 tokenizer 或可调用的 count_prompt_tokens")

    if repository is None:
        repository = ArticleRepository.from_jsonl(config.article_index)
    elif not isinstance(repository, ArticleRepository):
        raise TypeError("repository 必须是 ArticleRepository")

    if config.retrieval_mode == "article":
        if unit_repository is not None or unit_scorer is not None:
            raise ValueError("article 模式不能注入子单元依赖")
        if semantic_retriever is None:
            semantic_retriever = load_semantic_retriever(
                repository=repository,
                artifact_dir=config.artifact_dir,
                embedding_model=config.embedding_model,
                reranker_model=config.reranker_model,
                device=config.device,
                config=PRODUCTION_RETRIEVAL_CONFIG,
            )
    else:
        if unit_repository is None:
            unit_repository = EvidenceUnitRepository.from_jsonl(
                config.evidence_unit_dir / "evidence_units.jsonl",
                article_repository=repository,
            )
        elif not isinstance(unit_repository, EvidenceUnitRepository):
            raise TypeError("unit_repository 必须是 EvidenceUnitRepository")
        if semantic_retriever is None:
            semantic_retriever = load_evidence_unit_retriever(
                article_repository=repository,
                unit_repository=unit_repository,
                artifact_dir=config.evidence_unit_dir / "indexes",
                embedding_model=config.embedding_model,
                reranker_model=config.reranker_model,
                device=config.device,
                config=PRODUCTION_RETRIEVAL_CONFIG,
            )
        if unit_scorer is None:
            unit_scorer = getattr(semantic_retriever, "unit_scorer", None)
        if not callable(getattr(unit_scorer, "score", None)):
            raise TypeError("evidence_unit 检索器必须提供 unit_scorer")

    if external_llm is None and config.enable_external_llm:
        external_llm = OpenAICompatibleLLM.from_env()

    if config.retrieval_mode == "article":
        packager = EvidencePackager(
            context_limit=config.context_limit,
            max_output_tokens=config.max_output_tokens,
            count_prompt_tokens=count_prompt_tokens,
        )
    else:
        packager = EvidenceBundlePackager(
            unit_repository=unit_repository,
            unit_scorer=unit_scorer,
            context_limit=config.context_limit,
            max_output_tokens=config.max_output_tokens,
            count_prompt_tokens=count_prompt_tokens,
        )
    return CurrentLawRAG(
        article_repository=repository,
        semantic_retriever=semantic_retriever,
        evidence_packager=packager,
        generate=generate,
        external_llm=external_llm,
        max_output_tokens=config.max_output_tokens,
    )


__all__ = ["RAGApplicationConfig", "build_current_law_rag"]
