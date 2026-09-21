"""证据子单元仓库的加载与父子映射测试。"""

import json

import pytest

from rag.knowledge import ArticleRepository, EvidenceUnitRepository, LegalArticle
from rag.knowledge.evidence_units import split_article


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _repositories(tmp_path):
    records = [
        {
            "chunk_id": "示例办法#1",
            "law_name": "示例办法",
            "article_no": "1",
            "content": "办理要求如下：\n（一）登记；\n（二）审查。",
        },
        {
            "chunk_id": "示例办法#2",
            "law_name": "示例办法",
            "article_no": "2",
            "content": "申请人应当提交真实材料。",
        },
    ]
    article_path = tmp_path / "articles.jsonl"
    _write_jsonl(article_path, records)
    article_repository = ArticleRepository.from_jsonl(article_path)
    units = []
    for record in records:
        article = article_repository.get_by_chunk_id(record["chunk_id"])
        units.extend(unit.to_dict() for unit in split_article(article))
    unit_path = tmp_path / "units.jsonl"
    _write_jsonl(unit_path, units)
    return article_repository, unit_path, units


def test_repository_loads_exact_parent_mapping(tmp_path):
    article_repository, unit_path, records = _repositories(tmp_path)

    repository = EvidenceUnitRepository.from_jsonl(
        unit_path,
        article_repository=article_repository,
    )

    assert len(repository) == len(records)
    first = repository.get_for_parent("示例办法#1")
    assert [unit.unit_index for unit in first] == [0, 1, 2]
    assert repository.get_by_unit_id(first[1].unit_id) is first[1]
    assert tuple(repository.iter_units())[0] is first[0]


def test_repository_rejects_version_mismatch_and_noncontiguous_parent(tmp_path):
    article_repository, unit_path, records = _repositories(tmp_path)
    records[0]["splitter_version"] = "v2"
    records[0]["unit_id"] = records[0]["unit_id"].replace("span-v1", "span-v2")
    mismatch = tmp_path / "mismatch.jsonl"
    _write_jsonl(mismatch, records)
    with pytest.raises(ValueError, match="splitter_version"):
        EvidenceUnitRepository.from_jsonl(
            mismatch,
            article_repository=article_repository,
        )

    noncontiguous = tmp_path / "noncontiguous.jsonl"
    records[0]["splitter_version"] = "v1"
    records[0]["unit_id"] = records[0]["unit_id"].replace("span-v2", "span-v1")
    first_parent = [item for item in records if item["parent_chunk_id"] == "示例办法#1"]
    second_parent = [item for item in records if item["parent_chunk_id"] == "示例办法#2"]
    _write_jsonl(
        noncontiguous,
        [*first_parent, *second_parent, first_parent[0]],
    )
    with pytest.raises(ValueError, match="连续存储"):
        EvidenceUnitRepository.from_jsonl(
            noncontiguous,
            article_repository=article_repository,
        )
