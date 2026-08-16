"""现行有效法律知识库的确定性访问模块。"""

from rag.knowledge.repository import (
    ArticleRepository,
    IndexIntegrityError,
    LegalArticle,
)

__all__ = [
    "ArticleRepository",
    "IndexIntegrityError",
    "LegalArticle",
]
