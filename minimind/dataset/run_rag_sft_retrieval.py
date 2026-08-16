"""使用本地检索产物和模型执行 canonical RAG-SFT 真实物化。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from rag.answering import (
    AnswerPromptTokenCounter,
    EvidencePackager,
    RAG_MAX_OUTPUT_TOKENS,
)
from rag.knowledge import ArticleRepository
from rag.retrieval import SemanticRetrievalConfig, load_semantic_retriever

try:
    from . import audit_disc_law_sft as base_auditor
    from . import materialize_rag_sft_retrieved as materializer
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import audit_disc_law_sft as base_auditor
    from dataset import materialize_rag_sft_retrieved as materializer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_SFT_ROOT = Path(__file__).resolve().parent / "RAG-SFT"
DEFAULT_MANIFEST_768 = (
    RAG_SFT_ROOT / "manifests" / "rag-sft-canonical-v1-manifest-768.json"
)
DEFAULT_ARTICLE_INDEX = PROJECT_ROOT / "rag" / "chunk" / "article_index.jsonl"
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "rag" / "retrieval" / "artifacts"
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "minimind" / "model"
DEFAULT_OUTPUT_DIR = (
    RAG_SFT_ROOT / "retrieved" / "materialization-canonical-v1-20260809"
)
RETRIEVAL_ARTIFACT_FILENAMES = (
    "law_dense.faiss",
    "law_dense_meta.json",
    "law_sparse.pkl",
)
MODEL_IDENTITY_FILENAMES = (
    "config.json",
    "tokenizer_config.json",
    "model.safetensors",
    "pytorch_model.bin",
)


class RagSftRetrievalRunError(RuntimeError):
    """真实 retrieval 运行前置条件不满足。"""


def _file_identity(path: Path) -> dict[str, object]:
    path = path.resolve()
    if not path.is_file():
        raise RagSftRetrievalRunError(f"缺少本地文件: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": base_auditor.sha256_file(path),
    }


def _model_identity(path: Path) -> dict[str, object]:
    path = path.resolve()
    if not path.is_dir():
        raise RagSftRetrievalRunError(f"模型路径必须是本地目录: {path}")
    files = {
        name: _file_identity(path / name)
        for name in MODEL_IDENTITY_FILENAMES
        if (path / name).is_file()
    }
    if "config.json" not in files or not (
        {"model.safetensors", "pytorch_model.bin"} & set(files)
    ):
        raise RagSftRetrievalRunError(f"本地模型目录缺少配置或权重: {path}")
    return {
        "path": str(path),
        "snapshot": path.name,
        "files": files,
    }


def run_rag_sft_retrieval(
    *,
    manifest_768_path: Path,
    article_index_path: Path,
    artifact_dir: Path,
    tokenizer_path: Path,
    embedding_model_path: Path,
    reranker_model_path: Path,
    output_dir: Path,
    device: str = "auto",
) -> dict[str, object]:
    """显式加载本地资源并执行一次不可覆盖的真实 retrieval 物化。"""

    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RagSftRetrievalRunError(f"物化输出目录必须是新目录: {output_dir}")
    article_index_path = Path(article_index_path).resolve()
    article_index_identity = _file_identity(article_index_path)
    artifact_dir = Path(artifact_dir).resolve()
    artifact_identity = {
        name: _file_identity(artifact_dir / name)
        for name in RETRIEVAL_ARTIFACT_FILENAMES
    }
    embedding_identity = _model_identity(Path(embedding_model_path))
    reranker_identity = _model_identity(Path(reranker_model_path))
    config = SemanticRetrievalConfig()

    repository = ArticleRepository.from_jsonl(article_index_path)
    tokenizer = base_auditor.load_tokenizer(Path(tokenizer_path).resolve())
    packager = EvidencePackager(
        context_limit=materializer.CONTEXT_LIMIT,
        max_output_tokens=RAG_MAX_OUTPUT_TOKENS,
        count_prompt_tokens=AnswerPromptTokenCounter(tokenizer),
    )
    retriever = load_semantic_retriever(
        repository=repository,
        artifact_dir=artifact_dir,
        embedding_model=embedding_identity["path"],
        reranker_model=reranker_identity["path"],
        device=device,
        config=config,
    )
    retrieval_identity = {
        "article_index": article_index_identity,
        "artifacts": artifact_identity,
        "embedding_model": embedding_identity,
        "reranker_model": reranker_identity,
        "config": asdict(config),
        "requested_device": device,
    }
    return materializer.materialize_rag_sft_retrieved(
        manifest_768_path=manifest_768_path,
        retriever=retriever,
        evidence_packager=packager,
        retrieval_identity=retrieval_identity,
        output_dir=output_dir,
    )


def main() -> None:
    """解析本地资源路径并执行 canonical-v1 真实 retrieval。"""

    parser = argparse.ArgumentParser(
        description="使用本地模型和现有索引物化 canonical RAG-SFT retrieved 样本"
    )
    parser.add_argument("--manifest-768", type=Path, default=DEFAULT_MANIFEST_768)
    parser.add_argument("--article-index", type=Path, default=DEFAULT_ARTICLE_INDEX)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--embedding-model-path", type=Path, required=True)
    parser.add_argument("--reranker-model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    try:
        report = run_rag_sft_retrieval(
            manifest_768_path=args.manifest_768,
            article_index_path=args.article_index,
            artifact_dir=args.artifact_dir,
            tokenizer_path=args.tokenizer_path,
            embedding_model_path=args.embedding_model_path,
            reranker_model_path=args.reranker_model_path,
            output_dir=args.output_dir,
            device=args.device,
        )
    except (RagSftRetrievalRunError, materializer.RagSftRetrievedMaterializationError) as error:
        parser.error(str(error))
    print(
        f"[完成] 扫描 {report['validation']['scanned_records']} 条，"
        f"输出 {report['records']['output']} 条，"
        f"待人工复核 {report['records']['manual_review_required']} 条"
    )
    print(f"产物目录: {args.output_dir}")


if __name__ == "__main__":
    main()


__all__ = [
    "RagSftRetrievalRunError",
    "run_rag_sft_retrieval",
]
