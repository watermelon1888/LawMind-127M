"""语义检索分支的路由守卫。"""
from rag.query import AnswerMode, BusinessRoute, RouteDecision


def resolve_semantic_search(decision, retriever, retrieval_query=None):
    """只把 semantic_search 决策交给单 query retrieval。"""
    if not isinstance(decision, RouteDecision):
        raise TypeError("decision must be RouteDecision")
    if not (
        decision.route is BusinessRoute.ANSWER
        and decision.answer_mode is AnswerMode.RETRIEVAL
    ):
        raise ValueError("only retrieval decisions can use semantic search")
    query = decision.query if retrieval_query is None else retrieval_query
    if not isinstance(query, str) or not query.strip():
        raise ValueError("retrieval_query 必须是非空字符串")
    return retriever.search(query)


__all__ = ["resolve_semantic_search"]
