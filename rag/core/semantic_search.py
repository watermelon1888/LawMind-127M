"""语义检索分支的路由守卫。"""
from rag.query import QueryDecision, QueryRoute


def resolve_semantic_search(decision, retriever, retrieval_query=None):
    """只把 semantic_search 决策交给单 query retrieval。"""
    if not isinstance(decision, QueryDecision):
        raise TypeError("decision 必须是 QueryDecision")
    if decision.route is not QueryRoute.SEMANTIC_SEARCH:
        raise ValueError("只有 semantic_search 决策可以语义检索")
    query = decision.query if retrieval_query is None else retrieval_query
    if not isinstance(query, str) or not query.strip():
        raise ValueError("retrieval_query 必须是非空字符串")
    return retriever.search(query)


__all__ = ["resolve_semantic_search"]
