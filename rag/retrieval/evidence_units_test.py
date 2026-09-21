"""子单元 Hybrid、父级聚合和精排测试。"""

import json

import numpy as np
import rag.retrieval.evidence_units as evidence_units_module

from rag.knowledge import ArticleRepository, EvidenceUnitRepository
from rag.knowledge.evidence_units import split_article
from rag.retrieval.evidence_units import (
    EvidenceUnitRetriever,
    ScoredUnit,
    build_unit_dense_index,
    rrf_unit_fusion,
)
from rag.retrieval.semantic import SemanticRetrievalConfig
from rag.retrieval.test_text import CharacterTokenizer


def _repositories(tmp_path):
    records = [
        {
            "chunk_id": "甲办法#1",
            "law_name": "甲办法",
            "article_no": "1",
            "content": "要求如下：\n（一）甲条件；\n（二）乙条件；\n（三）丙条件。",
        },
        {
            "chunk_id": "乙办法#2",
            "law_name": "乙办法",
            "article_no": "2",
            "content": "乙法条完整规定。",
        },
        {
            "chunk_id": "丙办法#3",
            "law_name": "丙办法",
            "article_no": "3",
            "content": "丙法条完整规定。",
        },
    ]
    article_path = tmp_path / "articles.jsonl"
    article_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    articles = ArticleRepository.from_jsonl(article_path)
    unit_path = tmp_path / "units.jsonl"
    with unit_path.open("w", encoding="utf-8") as target:
        for record in records:
            article = articles.get_by_chunk_id(record["chunk_id"])
            for unit in split_article(article):
                target.write(json.dumps(unit.to_dict(), ensure_ascii=False) + "\n")
    units = EvidenceUnitRepository.from_jsonl(
        unit_path,
        article_repository=articles,
    )
    return articles, units


class _Searcher:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query

    def search(self, query, *, top_k):
        return tuple(self.results_by_query.get(query, ()))[:top_k]


class _Reranker:
    def __init__(self, score_by_text):
        self.score_by_text = score_by_text

    def score(self, query, units):
        return tuple(self.score_by_text.get(unit.text, 0.0) for unit in units)


class _Encoder:
    tokenizer = CharacterTokenizer()
    max_seq_length = 64

    def __init__(self):
        self.batch_sizes = []

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(
        self,
        texts,
        *,
        batch_size,
        show_progress_bar,
        normalize_embeddings,
    ):
        self.batch_sizes.append(len(texts))
        assert batch_size == 2
        assert show_progress_bar is False
        assert normalize_embeddings is True
        return np.asarray(
            [[float(len(text)), 1.0, 0.5] for text in texts],
            dtype=np.float32,
        )


def test_unit_rrf_never_uses_parent_chunk_identity():
    fused = rrf_unit_fusion(
        (
            (ScoredUnit("父法#1::span-v1@000000-000010", 0.9),),
            (ScoredUnit("父法#1::span-v1@000020-000030", 2.0),),
        ),
        k=4,
    )

    assert [item.unit_id for item in fused] == [
        "父法#1::span-v1@000000-000010",
        "父法#1::span-v1@000020-000030",
    ]


def test_parent_aggregation_uses_best_unit_without_count_bonus(tmp_path):
    articles, units = _repositories(tmp_path)
    alpha_units = units.get_for_parent("甲办法#1")
    beta = units.get_for_parent("乙办法#2")[0]
    dense = _Searcher(
        {
            "问题": (
                ScoredUnit(alpha_units[1].unit_id, 0.9),
                ScoredUnit(alpha_units[2].unit_id, 0.8),
                ScoredUnit(beta.unit_id, 0.7),
            )
        }
    )
    sparse = _Searcher({"问题": (ScoredUnit(beta.unit_id, 4.0),)})
    retriever = EvidenceUnitRetriever(
        dense_searcher=dense,
        sparse_searcher=sparse,
        article_repository=articles,
        unit_repository=units,
        reranker=_Reranker({}),
        config=SemanticRetrievalConfig(candidate_pool=2, top_k=2),
    )

    candidates = retriever.retrieve_candidates("问题")

    assert [item.chunk_id for item in candidates] == ["乙办法#2", "甲办法#1"]
    assert len(candidates[1].units) == 2
    assert candidates[1].score == 1 / 5


def test_reranker_uses_best_unit_and_returns_unique_parent_top_k(tmp_path):
    articles, units = _repositories(tmp_path)
    alpha_units = units.get_for_parent("甲办法#1")
    beta = units.get_for_parent("乙办法#2")[0]
    dense = _Searcher(
        {
            "问题": (
                ScoredUnit(alpha_units[1].unit_id, 0.9),
                ScoredUnit(beta.unit_id, 0.8),
                ScoredUnit(alpha_units[2].unit_id, 0.7),
            )
        }
    )
    retriever = EvidenceUnitRetriever(
        dense_searcher=dense,
        sparse_searcher=_Searcher({}),
        article_repository=articles,
        unit_repository=units,
        reranker=_Reranker(
            {
                alpha_units[1].text: 0.2,
                alpha_units[2].text: 0.95,
                beta.text: 0.7,
            }
        ),
        config=SemanticRetrievalConfig(candidate_pool=2, top_k=2),
    )

    ranked = retriever.search("问题")

    assert [item.chunk_id for item in ranked] == ["甲办法#1", "乙办法#2"]
    assert ranked[0].rerank_score == 0.95
    assert ranked[0].units[0].text == alpha_units[2].text


def test_multi_query_fusion_preserves_parent_units(tmp_path):
    articles, units = _repositories(tmp_path)
    alpha = units.get_for_parent("甲办法#1")[1]
    beta = units.get_for_parent("乙办法#2")[0]
    retriever = EvidenceUnitRetriever(
        dense_searcher=_Searcher(
            {
                "原问题": (ScoredUnit(alpha.unit_id, 0.9),),
                "补充问题": (ScoredUnit(beta.unit_id, 0.9),),
            }
        ),
        sparse_searcher=_Searcher({}),
        article_repository=articles,
        unit_repository=units,
        reranker=_Reranker({alpha.text: 0.5, beta.text: 0.4}),
        config=SemanticRetrievalConfig(candidate_pool=2, top_k=2),
    )

    candidates = retriever.retrieve_candidates_many(("原问题", "补充问题"))

    assert [item.chunk_id for item in candidates] == ["乙办法#2", "甲办法#1"]
    assert candidates[0].units == (beta,)


def test_dense_index_encodes_incrementally_and_preserves_window_mapping(
    tmp_path, monkeypatch
):
    articles, units = _repositories(tmp_path)
    encoder = _Encoder()
    monkeypatch.setattr(evidence_units_module, "DENSE_ENCODE_BUFFER_SIZE", 2)

    index, metadata = build_unit_dense_index(
        units,
        articles,
        encoder,
        model_name="test-encoder",
        overlap_tokens=4,
        batch_size=2,
    )

    assert index.ntotal == metadata["window_count"]
    assert metadata["unit_count"] == len(units)
    assert metadata["vector_dimension"] == 3
    assert len(encoder.batch_sizes) > 1
    assert all(size <= 2 for size in encoder.batch_sizes)
    assert [item["unit_id"] for item in metadata["position_to_window"]] == [
        unit.unit_id for unit in units.iter_units()
    ]
