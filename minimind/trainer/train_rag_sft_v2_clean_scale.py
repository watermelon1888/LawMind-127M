"""固定 226 条唯一 HN、改变 Clean 覆盖量的四轮训练入口。"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset

if __package__ in (None, ""):
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from ..dataset.pretrain_dataset import DeterministicPretrainSampler
    from ..dataset.rag_sft_v2_clean_scale import (
        GROUP_CLEAN_COUNTS,
        SUBSET_SEED,
        CleanScalePlan,
        build_clean_scale_plan,
    )
    from ..dataset.rag_sft_v2_dataset import LABEL_MASK_VERSION, RagSftV2Dataset
except ImportError:
    from dataset.pretrain_dataset import DeterministicPretrainSampler
    from dataset.rag_sft_v2_clean_scale import (
        GROUP_CLEAN_COUNTS,
        SUBSET_SEED,
        CleanScalePlan,
        build_clean_scale_plan,
    )
    from dataset.rag_sft_v2_dataset import LABEL_MASK_VERSION, RagSftV2Dataset

from . import train_full_sft as base_entry
from . import train_rag_sft_v2 as base_training
from .pretrain_runtime import token_learning_rate
from .sft_runtime import (
    JsonlMetricLogger,
    SftProgress,
    advance_sft_position,
    build_sft_optimizer,
    group_micro_batches,
    load_sft_checkpoint,
    run_sft_optimizer_step,
    save_sft_checkpoint,
)


PIPELINE = "rag_sft_v2_clean_scale_training_v1"
EPOCHS = 4
EXPORTED_EPOCHS = (2, 3, 4)
MAX_SMOKE_OPTIMIZER_STEPS = 10
OPTIMIZER = base_training.OPTIMIZER
MiniMindForCausalLM = base_entry.MiniMindForCausalLM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MiniMind RAG-SFT v2 Clean规模实验训练")
    for name in (
        "run_name",
        "run_dir",
        "checkpoint_dir",
        "manifest",
        "candidate_path",
        "tokenizer_path",
        "parent_weights",
        "parent_sha256",
        "group",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--peak_lr", type=float, default=1.2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--floor_ratio", type=float, default=0.1)
    parser.add_argument("--micro_batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float32"), default="bfloat16"
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_optimizer_steps", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identities() -> dict[str, dict[str, Any]]:
    project_root = Path(__file__).resolve().parents[2]
    paths = {
        "training": Path(__file__).resolve(),
        "clean_scale_plan": project_root
        / "minimind"
        / "dataset"
        / "rag_sft_v2_clean_scale.py",
        "dataset": project_root
        / "minimind"
        / "dataset"
        / "rag_sft_v2_dataset.py",
        "sft_runtime": project_root / "minimind" / "trainer" / "sft_runtime.py",
        "answering_protocol": project_root / "rag" / "answering" / "protocol.py",
        "answering_evidence": project_root / "rag" / "answering" / "evidence.py",
    }
    identities = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Clean规模实验运行时源码不存在: {path}")
        identities[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return identities


def optimizer_updates_per_epoch(
    records: int, *, micro_batch_size: int, accumulation_steps: int
) -> int:
    """按保留尾部不足组的现有训练语义计算每轮 update 数。"""

    if records <= 0 or micro_batch_size <= 0 or accumulation_steps <= 0:
        raise ValueError("records、micro batch 和 accumulation 必须为正数")
    micro_batches = math.ceil(records / micro_batch_size)
    return math.ceil(micro_batches / accumulation_steps)


def _validate_args(args: argparse.Namespace) -> None:
    if args.group not in GROUP_CLEAN_COUNTS:
        raise ValueError(f"未知 Clean 规模实验组: {args.group}")
    if (
        args.peak_lr <= 0
        or args.grad_clip <= 0
        or not 0 < args.warmup_ratio < 1
        or not 0 < args.floor_ratio <= 1
    ):
        raise ValueError("学习率、warmup_ratio、floor_ratio 和 grad_clip 参数无效")
    if (
        args.micro_batch_size <= 0
        or args.accumulation_steps <= 0
        or args.num_workers < 0
        or args.seed < 0
    ):
        raise ValueError("batch、worker 和 seed 参数无效")
    if (
        args.smoke_optimizer_steps is not None
        and not 1 <= args.smoke_optimizer_steps <= MAX_SMOKE_OPTIMIZER_STEPS
    ):
        raise ValueError("smoke_optimizer_steps 必须位于 1 至 10")
    if args.resume and args.smoke_optimizer_steps is not None:
        raise ValueError("smoke 不支持恢复")
    if len(args.parent_sha256) != 64:
        raise ValueError("parent_sha256 必须是 SHA-256")
    try:
        int(args.parent_sha256, 16)
    except ValueError as error:
        raise ValueError("parent_sha256 必须是 SHA-256") from error
    if int(os.environ.get("RANK", -1)) != -1:
        raise ValueError("Clean规模实验只支持单卡，不支持 DDP")


def _plan_identity(plan: CleanScalePlan) -> dict[str, Any]:
    return {
        "group": plan.group,
        "subset_seed": plan.subset_seed,
        "clean_count": plan.clean_count,
        "hard_negative_count": plan.hard_negative_count,
        "paired_clean_count": plan.paired_clean_count,
        "total_records": plan.total_records,
        "sha256": plan.sha256,
    }


def _verify_epoch_weight(path: Path) -> str:
    return base_training._verify_epoch_weight(path)


def _export_epoch_weights(path: Path, model: torch.nn.Module) -> str:
    return base_training._export_epoch_weights(path, model)


def _reconcile_epoch_weights(
    weights_dir: Path, progress: SftProgress, model: torch.nn.Module
) -> None:
    """只允许导出 epoch 2、3、4，并在恢复时核对既有边界。"""

    forbidden = weights_dir / "rag_epoch_1.pth"
    if forbidden.exists() or forbidden.with_suffix(".sha256").exists():
        raise ValueError("Clean规模实验禁止导出 rag_epoch_1.pth")
    if progress.next_sequence_position != 0 or progress.epoch > EPOCHS:
        return
    for epoch in EXPORTED_EPOCHS:
        if epoch > progress.epoch:
            continue
        path = weights_dir / f"rag_epoch_{epoch}.pth"
        if path.is_file() or path.with_suffix(".sha256").is_file():
            _verify_epoch_weight(path)
        elif epoch == progress.epoch:
            _export_epoch_weights(path, model)
        else:
            raise FileNotFoundError(f"缺少已完成 epoch {epoch} 的权重")


def run_training(args: argparse.Namespace) -> SftProgress:
    _validate_args(args)
    manifest_path = Path(args.manifest).resolve()
    manifest, manifest_sha = base_training._load_manifest(
        manifest_path, allow_stage_experiment=False
    )
    if (
        manifest.get("pipeline") != "rag_sft_v2_training_release"
        or manifest.get("readiness", {}).get("training_ready") is not True
    ):
        raise ValueError("Clean规模实验要求 training-ready v2 正式 release")
    run_dir = Path(args.run_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    latest = checkpoint_dir / "resume" / "latest.pt"
    weights_dir = checkpoint_dir / "weights"
    run_manifest_path = run_dir / "run-manifest.json"
    metrics_path = run_dir / "metrics.jsonl"
    if not args.resume and (run_dir.exists() or checkpoint_dir.exists()):
        raise FileExistsError("Clean规模实验 run 或 checkpoint 目录已存在")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 CUDA 但当前无可用 GPU")
    if (
        args.dtype == "bfloat16"
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("当前 CUDA 不支持 BF16")
    parent_path = Path(args.parent_weights).resolve()
    if (
        not parent_path.is_file()
        or _sha256_file(parent_path) != args.parent_sha256.lower()
    ):
        raise ValueError("父权重文件 SHA-256 校验失败")

    tokenizer = base_entry._load_tokenizer(args.tokenizer_path)
    dataset = RagSftV2Dataset(
        manifest_path,
        tokenizer,
        candidate_path=args.candidate_path,
        tokenizer_path=args.tokenizer_path,
    )
    try:
        plan = build_clean_scale_plan(
            dataset.candidate_path,
            group=args.group,
            subset_seed=SUBSET_SEED,
        )
        if any(index < 0 or index >= len(dataset) for index in plan.indices):
            raise ValueError("Clean规模实验计划索引越界")
        training_dataset = Subset(dataset, plan.indices)
        updates_per_epoch = optimizer_updates_per_epoch(
            len(training_dataset),
            micro_batch_size=args.micro_batch_size,
            accumulation_steps=args.accumulation_steps,
        )
        total_optimizer_steps = updates_per_epoch * EPOCHS
        warmup_steps = max(1, int(total_optimizer_steps * args.warmup_ratio))
        source_identities = _source_identities()
        controls = {
            "stop_optimizer_steps": (
                args.smoke_optimizer_steps or total_optimizer_steps
            ),
            "smoke_optimizer_steps": args.smoke_optimizer_steps,
        }
        invariants = {
            "pipeline": PIPELINE,
            "manifest_sha256": manifest_sha,
            "candidate_sha256": _sha256_file(dataset.candidate_path),
            "tokenizer": dataset.tokenizer_identity,
            "parent_sha256": args.parent_sha256.lower(),
            "source_identities": source_identities,
            "plan": _plan_identity(plan),
            "sequence_length": 768,
            "label_mask_version": LABEL_MASK_VERSION,
            "epochs": EPOCHS,
            "exported_epochs": list(EXPORTED_EPOCHS),
            "seed": args.seed,
            "optimizer": OPTIMIZER,
            "peak_lr": args.peak_lr,
            "warmup_optimizer_steps": warmup_steps,
            "optimizer_steps_per_epoch": updates_per_epoch,
            "schedule_optimizer_steps": total_optimizer_steps,
            "floor_ratio": args.floor_ratio,
            "micro_batch_size": args.micro_batch_size,
            "accumulation_steps": args.accumulation_steps,
            "grad_clip": args.grad_clip,
            "dtype": args.dtype,
            "num_workers": args.num_workers,
        }

        base_entry.setup_seed(args.seed)
        model = MiniMindForCausalLM(base_entry._model_config()).to(device)
        base_entry.get_model_params(model, base_entry._model_config())
        if not args.resume:
            base_entry._load_parent_weights(
                parent_path, args.parent_sha256, model
            )
        optimizer = build_sft_optimizer(
            model, peak_lr=args.peak_lr, device_type=device.type
        )
        progress = SftProgress()
        if args.resume:
            saved_manifest, _ = base_entry._load_verified_json(
                run_manifest_path, "Clean规模实验 run manifest"
            )
            if saved_manifest.get("training_invariants") != invariants:
                raise ValueError("恢复时训练不变量不一致")
            progress = load_sft_checkpoint(
                latest,
                model=model,
                optimizer=optimizer,
                expected_invariants=invariants,
                controls=controls,
            ).progress
            if args.smoke_optimizer_steps is None:
                _reconcile_epoch_weights(weights_dir, progress, model)
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            base_entry._write_immutable_run_manifest(
                run_manifest_path,
                {
                    "schema_version": "1.0",
                    "pipeline": PIPELINE,
                    "run_name": args.run_name,
                    "data": {
                        "manifest_sha256": manifest_sha,
                        "candidate_sha256": _sha256_file(dataset.candidate_path),
                    },
                    "parent_model": {
                        "path": str(parent_path),
                        "sha256": args.parent_sha256.lower(),
                    },
                    "environment": {
                        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                        "TOKENIZERS_PARALLELISM": os.environ.get(
                            "TOKENIZERS_PARALLELISM"
                        ),
                    },
                    "training_invariants": invariants,
                },
            )

        logger = JsonlMetricLogger(metrics_path)
        stop_steps = args.smoke_optimizer_steps or total_optimizer_steps
        while progress.epoch < EPOCHS and progress.optimizer_step < stop_steps:
            sampler = DeterministicPretrainSampler(
                training_dataset,
                seed=args.seed,
                epoch=progress.epoch,
                start_position=progress.next_sequence_position,
            )
            loader = DataLoader(
                training_dataset,
                batch_size=args.micro_batch_size,
                sampler=sampler,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.num_workers > 0,
            )
            for groups in group_micro_batches(loader, args.accumulation_steps):
                if progress.optimizer_step >= stop_steps:
                    break
                lr = token_learning_rate(
                    progress.optimizer_step + 1,
                    args.peak_lr,
                    warmup_steps,
                    total_optimizer_steps,
                    args.floor_ratio,
                )
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                result = run_sft_optimizer_step(
                    model,
                    optimizer,
                    groups,
                    device=device,
                    learning_rate=lr,
                    grad_clip=args.grad_clip,
                    autocast_dtype=(
                        torch.bfloat16 if args.dtype == "bfloat16" else None
                    ),
                )
                elapsed = time.perf_counter() - started
                epoch, position = advance_sft_position(
                    progress.epoch,
                    progress.next_sequence_position,
                    result.sequence_count,
                    len(training_dataset),
                )
                progress = SftProgress(
                    epoch=epoch,
                    next_sequence_position=position,
                    completed_sequences=(
                        progress.completed_sequences + result.sequence_count
                    ),
                    completed_assistant_tokens=(
                        progress.completed_assistant_tokens
                        + result.assistant_tokens
                    ),
                    optimizer_step=progress.optimizer_step + 1,
                )
                logger.log(
                    {
                        "type": "train",
                        "group": args.group,
                        "optimizer_step": progress.optimizer_step,
                        "epoch": progress.epoch,
                        "completed_sequences": progress.completed_sequences,
                        "completed_assistant_tokens": (
                            progress.completed_assistant_tokens
                        ),
                        "assistant_tokens": result.assistant_tokens,
                        "loss": result.loss,
                        "learning_rate": lr,
                        "grad_norm": result.grad_norm,
                        "elapsed_seconds": elapsed,
                        "assistant_tokens_per_second": (
                            result.assistant_tokens / elapsed
                        ),
                        "cuda_peak_memory_bytes": (
                            torch.cuda.max_memory_allocated(device)
                            if device.type == "cuda"
                            else None
                        ),
                    }
                )
                if position == 0 or progress.optimizer_step >= stop_steps:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    save_sft_checkpoint(
                        latest,
                        model=model,
                        optimizer=optimizer,
                        progress=progress,
                        invariants=invariants,
                        controls=controls,
                    )
                    if position == 0 and args.smoke_optimizer_steps is None:
                        _reconcile_epoch_weights(weights_dir, progress, model)
            del loader

        if args.smoke_optimizer_steps is None and (
            progress.epoch != EPOCHS
            or progress.next_sequence_position != 0
            or progress.optimizer_step != total_optimizer_steps
            or progress.completed_sequences != len(training_dataset) * EPOCHS
        ):
            raise RuntimeError("Clean规模实验未完成四个真实 epoch")
        if args.smoke_optimizer_steps is not None:
            load_sft_checkpoint(
                latest,
                model=model,
                optimizer=optimizer,
                expected_invariants=invariants,
                controls=controls,
            )
        logger.log(
            {
                "type": "run_complete",
                "group": args.group,
                "epoch": progress.epoch,
                "next_sequence_position": progress.next_sequence_position,
                "completed_sequences": progress.completed_sequences,
                "completed_assistant_tokens": progress.completed_assistant_tokens,
                "optimizer_step": progress.optimizer_step,
                "weights_exported_epochs": (
                    list(EXPORTED_EPOCHS)
                    if args.smoke_optimizer_steps is None
                    else []
                ),
            }
        )
        return progress
    finally:
        dataset.close()


def main() -> None:
    progress = run_training(build_parser().parse_args())
    print(
        "RAG_SFT_V2_CLEAN_SCALE_TRAINING_OK "
        f"epoch={progress.epoch} optimizer_step={progress.optimizer_step}"
    )


if __name__ == "__main__":
    main()
