"""执行父权重比较的通用与基础法律 loss 评估。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import torch

try:
    from ..dataset.pretrain_dataset import PretrainShardCatalog
except ImportError:
    from dataset.pretrain_dataset import PretrainShardCatalog

from . import sft_parent_evaluation as parent_evaluation
from . import train_full_sft as training
from .pretrain_runtime import evaluate_pretrain_by_source
from .sft_runtime import evaluate_sft_by_source


PIPELINE = "legal_sft_parent_loss_evaluation_v1"
SUITE_TO_SPLIT = {
    "general": "full_validation",
    "legal_quick": "validation/quick",
    "legal_full": "validation/full",
}


def _evaluate_with_oom_fallback(
    evaluator: Callable[[int], dict[str, Any]],
    *,
    preferred_batch_size: int,
    device: torch.device,
) -> tuple[dict[str, Any], int]:
    try:
        return evaluator(preferred_batch_size), preferred_batch_size
    except torch.OutOfMemoryError:
        if device.type != "cuda" or preferred_batch_size <= 32:
            raise
        torch.cuda.empty_cache()
        return evaluator(32), 32


def _load_model(
    weights: str | Path, weights_sha256: str, device: torch.device
) -> torch.nn.Module:
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 CUDA 设备不可用")
    model = training.MiniMindForCausalLM(training._model_config()).to(device)
    training._load_parent_weights(weights, weights_sha256, model)
    return model


def evaluate_loss_suites(
    *,
    suites: tuple[str, ...],
    general_manifest: str | Path,
    legal_manifest: str | Path,
    tokenizer_path: str | Path,
    weights: str | Path,
    weights_sha256: str,
    output: str | Path,
    device_name: str,
    eval_batch_size: int = 64,
    eval_num_workers: int = 4,
) -> dict[str, Any]:
    """加载一次 model-only 权重并发布所需 loss suites。"""

    if not suites or len(set(suites)) != len(suites):
        raise ValueError("loss suites 不能为空或重复")
    unknown = set(suites).difference(SUITE_TO_SPLIT)
    if unknown:
        raise ValueError(f"未知 loss suite: {sorted(unknown)[0]}")
    if eval_batch_size <= 0 or eval_num_workers < 0:
        raise ValueError("评估 batch_size 必须为正，num_workers 不能为负")

    device = torch.device(device_name)
    tokenizer = training._load_tokenizer(tokenizer_path)
    tokenizer_identity = training._tokenizer_identity(tokenizer, tokenizer_path)
    model = _load_model(weights, weights_sha256, device)
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None
    reports: dict[str, Any] = {}

    general_catalog = None
    if "general" in suites:
        general_catalog = PretrainShardCatalog(general_manifest)
        count_dataset = general_catalog.create_dataset("full_validation")
        general_sample_count = len(count_dataset)
        count_dataset.close()

        def evaluate_general(batch_size: int) -> dict[str, Any]:
            return evaluate_pretrain_by_source(
                model,
                general_catalog,
                split="full_validation",
                batch_size=batch_size,
                device=device,
                num_workers=eval_num_workers,
                autocast_dtype=autocast_dtype,
            )

        report, actual_batch_size = _evaluate_with_oom_fallback(
            evaluate_general,
            preferred_batch_size=eval_batch_size,
            device=device,
        )
        reports["general"] = {
            "sample_count": general_sample_count,
            "metrics": {
                "loss": report["overall"]["loss"],
                "perplexity": report["overall"]["perplexity"],
            },
            "eval_batch_size": actual_batch_size,
            "details": report,
        }

    for suite in suites:
        if suite == "general":
            continue
        split = SUITE_TO_SPLIT[suite]

        def dataset_factory(selected_split: str, dataset_kind: str):
            return training.SftDataset(
                legal_manifest,
                selected_split,
                tokenizer,
                dataset_kind=dataset_kind,
            )

        def evaluate_legal(batch_size: int) -> dict[str, Any]:
            return evaluate_sft_by_source(
                model,
                dataset_factory,
                split=split,
                batch_size=batch_size,
                device=device,
                num_workers=eval_num_workers,
                autocast_dtype=autocast_dtype,
            )

        report, actual_batch_size = _evaluate_with_oom_fallback(
            evaluate_legal,
            preferred_batch_size=eval_batch_size,
            device=device,
        )
        reports[suite] = {
            "sample_count": report["overall"]["sequences"],
            "metrics": {
                "loss": report["overall"]["loss"],
                "perplexity": report["overall"]["perplexity"],
            },
            "eval_batch_size": actual_batch_size,
            "details": report,
        }

    payload = {
        "schema_version": "1.0",
        "pipeline": PIPELINE,
        "inputs": {
            "weights": {
                "path": str(Path(weights).resolve()),
                "sha256": weights_sha256,
            },
            "general_manifest": str(Path(general_manifest).resolve()),
            "legal_manifest": str(Path(legal_manifest).resolve()),
            "tokenizer": tokenizer_identity,
        },
        "suites": {suite: reports[suite] for suite in suites},
        "complete": True,
    }
    parent_evaluation._write_immutable_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="执行父权重比较的 loss 评估")
    parser.add_argument("--suite", action="append", required=True, choices=SUITE_TO_SPLIT)
    parser.add_argument("--general-manifest", required=True)
    parser.add_argument("--legal-manifest", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--eval-num-workers", type=int, default=4)
    args = parser.parse_args()
    try:
        report = evaluate_loss_suites(
            suites=tuple(args.suite),
            general_manifest=args.general_manifest,
            legal_manifest=args.legal_manifest,
            tokenizer_path=args.tokenizer_path,
            weights=args.weights,
            weights_sha256=args.weights_sha256,
            output=args.output,
            device_name=args.device,
            eval_batch_size=args.eval_batch_size,
            eval_num_workers=args.eval_num_workers,
        )
        print(f"SFT_PARENT_LOSS_EVALUATION_OK suites={len(report['suites'])}")
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        raise SystemExit(f"[失败] {error}") from error


if __name__ == "__main__":
    main()
