"""现行有效法律知识库的确定性访问模块。"""

from rag.knowledge.repository import (
    ArticleRepository,
    IndexIntegrityError,
    LegalArticle,
)
from rag.knowledge.evidence_units import (
    EvidenceUnit,
    EvidenceUnitRepository,
    SPLITTER_VERSION,
    make_unit_id,
    split_article,
    validate_article_units,
)

__all__ = [
    "ArticleRepository",
    "EvidenceUnit",
    "EvidenceUnitRepository",
    "IndexIntegrityError",
    "LegalArticle",
    "SPLITTER_VERSION",
    "make_unit_id",
    "split_article",
    "validate_article_units",
]
