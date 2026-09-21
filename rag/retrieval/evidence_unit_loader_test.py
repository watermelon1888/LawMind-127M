"""子单元索引启动前一致性校验测试。"""

import json

import pytest

from rag.knowledge import ArticleRepository, EvidenceUnitRepository
from rag.knowledge.evidence_units import split_article
from rag.retrieval.evidence_unit_loader import validate_unit_artifacts


class _Index:
    ntotal = 1
    d = 3


def _inputs(tmp_path):
    record = {
        "chunk_id": "示例法#1",
        "law_name": "示例法",
        "article_no": "1",
        "content": "完整法条。",
    }
    article_path = tmp_path / "articles.jsonl"
    article_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    articles = ArticleRepository.from_jsonl(article_path)
    unit = split_article(articles.get_by_chunk_id("示例法#1"))[0]
    unit_path = tmp_path / "units.jsonl"
    unit_path.write_text(
        json.dumps(unit.to_dict(), ensure_ascii=False) + "\n", encoding="utf-8"
    )
    units = EvidenceUnitRepository.from_jsonl(unit_path, article_repository=articles)
    dense = {
        "splitter_version": "v1",
        "unit_count": 1,
        "position_to_window": [
            {
                "unit_id": unit.unit_id,
                "parent_chunk_id": unit.parent_chunk_id,
                "window_index": 0,
                "start_token": 0,
                "end_token": 3,
            }
        ],
    }
    sparse = {
        "splitter_version": "v1",
        "unit_count": 1,
        "position_to_unit_id": [unit.unit_id],
    }
    return articles, units, dense, sparse


def test_validate_unit_artifacts_accepts_matching_versions_and_mappings(tmp_path):
    articles, units, dense, sparse = _inputs(tmp_path)

    validate_unit_artifacts(
        article_repository=articles,
        unit_repository=units,
        dense_index=_Index(),
        dense_metadata=dense,
        sparse_metadata=sparse,
        embedding_dimension=3,
        bm25_corpus_size=1,
    )


def test_validate_unit_artifacts_rejects_version_and_parent_mismatch(tmp_path):
    articles, units, dense, sparse = _inputs(tmp_path)
    wrong_version = {**dense, "splitter_version": "v2"}
    with pytest.raises(ValueError, match="splitter_version"):
        validate_unit_artifacts(
            article_repository=articles,
            unit_repository=units,
            dense_index=_Index(),
            dense_metadata=wrong_version,
            sparse_metadata=sparse,
            embedding_dimension=3,
        )

    wrong_parent = {
        **dense,
        "position_to_window": [
            {**dense["position_to_window"][0], "parent_chunk_id": "其他法#1"}
        ],
    }
    with pytest.raises(ValueError, match="父子映射"):
        validate_unit_artifacts(
            article_repository=articles,
            unit_repository=units,
            dense_index=_Index(),
            dense_metadata=wrong_parent,
            sparse_metadata=sparse,
            embedding_dimension=3,
        )
