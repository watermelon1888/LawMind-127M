"""使用当前原始 query 检索链物化 RAG-SFT v2 HN 候选。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rag.answering import AnswerPromptTokenCounter, EvidencePackager
from rag.knowledge import ArticleRepository
from rag.retrieval import SemanticRetrievalConfig, load_semantic_retriever

from . import audit_disc_law_sft as tokenizer_auditor
from .materialize_rag_sft_v2_retrieved import (
    MAX_OUTPUT_TOKENS,
    materialize_rag_sft_v2_retrieved,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path(__file__).resolve().parent
RAG_SFT_ROOT = DATASET_ROOT / "RAG-SFT"
DEFAULT_MANIFEST = RAG_SFT_ROOT / "review" / "v2" / "oracle-clean-v2" / "manifest.json"
DEFAULT_ARTICLE_INDEX = PROJECT_ROOT / "rag" / "chunk" / "article_index.jsonl"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "rag" / "retrieval" / "artifacts"
DEFAULT_TOKENIZER = PROJECT_ROOT / "minimind" / "model"


def _file_identity(path: Path) -> dict[str, object]:
    from .audit_disc_law_sft import sha256_file

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _model_identity(path: Path) -> dict[str, object]:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(path)
    names = ("config.json", "tokenizer_config.json", "model.safetensors", "pytorch_model.bin")
    files = {name: _file_identity(path / name) for name in names if (path / name).is_file()}
    if "config.json" not in files or not ({"model.safetensors", "pytorch_model.bin"} & set(files)):
        raise ValueError(f"模型目录缺少配置或权重: {path}")
    return {"path": str(path), "snapshot": path.name, "files": files}


def run_rag_sft_v2_retrieved(
    *,
    manifest_path: Path,
    article_index_path: Path,
    artifact_dir: Path,
    tokenizer_path: Path,
    embedding_model_path: Path,
    reranker_model_path: Path,
    output_dir: Path,
    device: str = "auto",
) -> dict[str, object]:
    """加载固定本地身份并执行一次不可覆盖的 v2 候选物化。"""
    article_index_path = Path(article_index_path).resolve()
    artifact_dir = Path(artifact_dir).resolve()
    embedding_model_path = Path(embedding_model_path).resolve()
    reranker_model_path = Path(reranker_model_path).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    config = SemanticRetrievalConfig()
    repository = ArticleRepository.from_jsonl(article_index_path)
    tokenizer = tokenizer_auditor.load_tokenizer(tokenizer_path)
    packager = EvidencePackager(
        context_limit=768,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        count_prompt_tokens=AnswerPromptTokenCounter(tokenizer),
    )
    retriever = load_semantic_retriever(
        repository=repository,
        artifact_dir=artifact_dir,
        embedding_model=str(embedding_model_path),
        reranker_model=str(reranker_model_path),
        device=device,
        config=config,
    )
    retrieval_identity = {
        "article_index": _file_identity(article_index_path),
        "artifacts": {name: _file_identity(artifact_dir / name) for name in ("law_dense.faiss", "law_dense_meta.json", "law_sparse.pkl")},
        "embedding_model": _model_identity(embedding_model_path),
        "reranker_model": _model_identity(reranker_model_path),
        "config": asdict(config),
        "query_mode": "original_only",
        "requested_device": device,
    }
    return materialize_rag_sft_v2_retrieved(
        manifest_path=Path(manifest_path),
        retriever=retriever,
        evidence_packager=packager,
        retrieval_identity=retrieval_identity,
        output_dir=Path(output_dir),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--article-index", type=Path, default=DEFAULT_ARTICLE_INDEX)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--embedding-model", type=Path, required=True)
    parser.add_argument("--reranker-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    report = run_rag_sft_v2_retrieved(
        manifest_path=args.manifest,
        article_index_path=args.article_index,
        artifact_dir=args.artifact_dir,
        tokenizer_path=args.tokenizer,
        embedding_model_path=args.embedding_model,
        reranker_model_path=args.reranker_model,
        output_dir=args.output_dir,
        device=args.device,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["run_rag_sft_v2_retrieved"]
