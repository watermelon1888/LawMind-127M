"""统一应用装配入口。"""

from rag.app.assembly import (
    RAGApplicationConfig,
    build_current_law_rag,
)

__all__ = ["RAGApplicationConfig", "build_current_law_rag"]
