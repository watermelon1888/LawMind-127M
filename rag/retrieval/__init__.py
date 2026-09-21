"""候选法条召回与排序模块。"""

from rag.retrieval.semantic import (
    RankedArticle,
    RetrievalIntegrityError,
    SemanticRetrievalConfig,
    SemanticRetriever,
)


def load_semantic_retriever(*args, **kwargs):
    """延迟导入显式组装函数，保持包导入轻量。"""
    from rag.retrieval.loader import load_semantic_retriever as load

    return load(*args, **kwargs)


def load_evidence_unit_retriever(*args, **kwargs):
    """延迟加载父子证据检索器及其重量级模型。"""
    from rag.retrieval.evidence_unit_loader import (
        load_evidence_unit_retriever as load,
    )

    return load(*args, **kwargs)

__all__ = [
    "RankedArticle",
    "RetrievalIntegrityError",
    "SemanticRetrievalConfig",
    "SemanticRetriever",
    "load_semantic_retriever",
    "load_evidence_unit_retriever",
]
