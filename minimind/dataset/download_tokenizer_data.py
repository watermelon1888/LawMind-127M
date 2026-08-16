"""校验手工上传的法律 DOCX，并下载 tokenizer 所需的外部原始数据。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable


DISC_LAW_SFT_REPO = "ShengbinYue/DISC-Law-SFT"
MINIMIND_DATASET_REPO = "jingyaogong/minimind_dataset"
MINIMIND_PRETRAIN_FILE = "pretrain_t2t_mini.jsonl"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "tokenizer_data" / "raw"

SnapshotDownloader = Callable[..., str]


def find_legal_documents(legal_dir: Path) -> list[Path]:
    """返回手工上传的法律 DOCX 文件，并验证部门法目录有效性。"""
    if not legal_dir.is_dir():
        raise ValueError(f"法律 DOCX 目录不存在: {legal_dir}")

    documents = sorted(path for path in legal_dir.rglob("*.docx") if path.is_file())
    if not documents:
        raise ValueError(f"法律 DOCX 目录不包含 DOCX 文件: {legal_dir}")
    return documents


def _snapshot_download(**kwargs: object) -> str:
    """延迟加载 huggingface_hub，确保校验和测试保持离线。"""
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "缺少 huggingface_hub，请先安装 transformers 的依赖或执行 "
            "pip install huggingface_hub"
        ) from error
    return snapshot_download(**kwargs)


def _has_downloaded_files(destination: Path) -> bool:
    """判断目标目录中是否已有已下载文件。"""
    return destination.is_dir() and any(path.is_file() for path in destination.rglob("*"))


def download_dataset_snapshot(
    repo_id: str,
    destination: Path,
    snapshot_downloader: SnapshotDownloader,
    allow_patterns: Iterable[str] | None = None,
) -> bool:
    """下载数据集快照；已有完整下载时直接复用。"""
    if _has_downloaded_files(destination):
        print(f"[跳过] 已存在: {destination}")
        return False

    destination.mkdir(parents=True, exist_ok=True)
    arguments: dict[str, object] = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "local_dir": str(destination),
    }
    if allow_patterns is not None:
        arguments["allow_patterns"] = list(allow_patterns)

    print(f"[下载] {repo_id} -> {destination}")
    snapshot_downloader(**arguments)
    if not _has_downloaded_files(destination):
        raise RuntimeError(f"下载后未发现文件: {destination}")
    return True


def prepare_raw_data(
    output_dir: Path,
    snapshot_downloader: SnapshotDownloader = _snapshot_download,
    legal_dir: Path | None = None,
    download_law_sft: bool = True,
    download_general: bool = True,
) -> dict[str, object]:
    """校验法律 DOCX，并下载或复用法律 SFT 与通用中文原始数据。"""
    output_dir = output_dir.resolve()
    legal_dir = (legal_dir or output_dir / "legal" / "docx" / "Chinese-Laws").resolve()
    legal_docx_count = len(find_legal_documents(legal_dir))

    result: dict[str, object] = {
        "legal_dir": legal_dir,
        "legal_docx_count": legal_docx_count,
        "law_sft_dir": output_dir / "law_sft",
        "general_dir": output_dir / "general",
    }

    if download_law_sft:
        result["law_sft_downloaded"] = download_dataset_snapshot(
            DISC_LAW_SFT_REPO,
            result["law_sft_dir"],
            snapshot_downloader,
        )
    if download_general:
        result["general_downloaded"] = download_dataset_snapshot(
            MINIMIND_DATASET_REPO,
            result["general_dir"],
            snapshot_downloader,
            allow_patterns=[MINIMIND_PRETRAIN_FILE],
        )
    return result


def main() -> None:
    """解析命令行参数并准备 tokenizer 原始数据目录。"""
    parser = argparse.ArgumentParser(description="校验法律 DOCX 并准备 tokenizer 外部原始数据")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--legal-dir", type=Path, default=None, help="手工上传并解压后的 Chinese-Laws DOCX 目录")
    parser.add_argument("--skip-law-sft", action="store_true", help="不下载 DISC-Law-SFT")
    parser.add_argument("--skip-general", action="store_true", help="不下载 MiniMind 预训练数据")
    args = parser.parse_args()

    result = prepare_raw_data(
        output_dir=args.output_dir,
        legal_dir=args.legal_dir,
        download_law_sft=not args.skip_law_sft,
        download_general=not args.skip_general,
    )
    print(f"[完成] 法律 DOCX: {result['legal_docx_count']} 部")
    print(f"  法律数据: {result['legal_dir']}")
    print(f"  法律 SFT: {result['law_sft_dir']}")
    print(f"  通用文本: {result['general_dir']}")


if __name__ == "__main__":
    main()
