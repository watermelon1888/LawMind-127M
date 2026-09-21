"""从已验收 evidence unit sidecar 构建独立 Dense/BM25 索引。"""

import argparse
from pathlib import Path

from rag.knowledge import ArticleRepository, EvidenceUnitRepository
from rag.retrieval.evidence_units import (
    build_unit_dense_index,
    build_unit_sparse_index,
    save_unit_indexes,
)
from rag.retrieval.loader import DEFAULT_EMBEDDING_MODEL, _resolve_device


RAG_DIR = Path(__file__).resolve().parents[1]
DEFAULT_ARTICLE_INDEX = RAG_DIR / "chunk" / "article_index_with_department_rules.jsonl"
DEFAULT_UNIT_DIR = Path(__file__).resolve().parent / "evidence_unit_artifacts" / "v1"
DEFAULT_UNIT_SIDECAR = DEFAULT_UNIT_DIR / "evidence_units.jsonl"
DEFAULT_INDEX_DIR = DEFAULT_UNIT_DIR / "indexes"


def build_selected_unit_indexes(
    *,
    target,
    article_index,
    unit_sidecar,
    artifact_dir,
    embedding_model,
    device,
    overlap_tokens,
    batch_size,
):
    """构建调用方指定的子单元索引，并拒绝覆盖已有产物。"""
    directory = Path(artifact_dir)
    selected_paths = []
    if target in ("sparse", "all"):
        selected_paths.append(directory / "evidence_unit_sparse.pkl")
    if target in ("dense", "all"):
        selected_paths.extend(
            (
                directory / "evidence_unit_dense.faiss",
                directory / "evidence_unit_dense_meta.json",
            )
        )
    existing = [path for path in selected_paths if path.exists()]
    if existing:
        raise FileExistsError(f"拒绝覆盖已有子索引产物: {existing[0]}")

    articles = ArticleRepository.from_jsonl(article_index)
    units = EvidenceUnitRepository.from_jsonl(
        unit_sidecar,
        article_repository=articles,
    )
    if target in ("sparse", "all"):
        bm25, metadata = build_unit_sparse_index(units, articles)
        save_unit_indexes(
            bm25=bm25,
            sparse_metadata=metadata,
            artifact_dir=directory,
        )
    if target in ("dense", "all"):
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(
            embedding_model,
            device=_resolve_device(device),
        )
        index, metadata = build_unit_dense_index(
            units,
            articles,
            encoder,
            model_name=embedding_model,
            overlap_tokens=overlap_tokens,
            batch_size=batch_size,
        )
        save_unit_indexes(
            dense_index=index,
            dense_metadata=metadata,
            artifact_dir=directory,
        )


def main():
    parser = argparse.ArgumentParser(description="构建 evidence unit 检索索引")
    parser.add_argument("--target", required=True, choices=("dense", "sparse", "all"))
    parser.add_argument("--article-index", default=str(DEFAULT_ARTICLE_INDEX))
    parser.add_argument("--unit-sidecar", default=str(DEFAULT_UNIT_SIDECAR))
    parser.add_argument("--artifact-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overlap-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    build_selected_unit_indexes(
        target=args.target,
        article_index=args.article_index,
        unit_sidecar=args.unit_sidecar,
        artifact_dir=args.artifact_dir,
        embedding_model=args.embedding_model,
        device=args.device,
        overlap_tokens=args.overlap_tokens,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
