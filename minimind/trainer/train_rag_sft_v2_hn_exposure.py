"""cpt_750m RAG-SFT v2 clean/HN 曝光率对照训练入口。"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

try:
    from ..dataset.rag_sft_v2_dataset import LABEL_MASK_VERSION, RagSftV2Dataset
    from ..dataset.rag_sft_v2_exposure import (
        COMMON_EXPOSURES,
        COMMON_OPTIMIZER_STEPS,
        EXPOSURE_GROUPS,
        RagSftV2ExposureSampler,
        audit_clean_hn_pairing,
        build_exposure_plan,
    )
except ImportError:
    from dataset.rag_sft_v2_dataset import LABEL_MASK_VERSION, RagSftV2Dataset
    from dataset.rag_sft_v2_exposure import (
        COMMON_EXPOSURES,
        COMMON_OPTIMIZER_STEPS,
        EXPOSURE_GROUPS,
        RagSftV2ExposureSampler,
        audit_clean_hn_pairing,
        build_exposure_plan,
    )
from . import train_full_sft as base_entry
from . import train_rag_sft_v2 as stage_f_training
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


PIPELINE = "rag_sft_v2_cpt750m_hn_exposure_v1"
CHECKPOINT_STEPS = (35, 70, 105, 140)
MAX_SMOKE_OPTIMIZER_STEPS = 10
OPTIMIZER = stage_f_training.OPTIMIZER
MiniMindForCausalLM = base_entry.MiniMindForCausalLM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="cpt_750m clean/HN 曝光率对照训练")
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
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke_optimizer_steps", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.group not in EXPOSURE_GROUPS:
        raise ValueError(f"未知实验组: {args.group}")
    if args.micro_batch_size * args.accumulation_steps != 16:
        raise ValueError("固定 140-step 实验要求 micro batch × accumulation = 16")
    if args.peak_lr <= 0 or args.grad_clip <= 0:
        raise ValueError("学习率和 grad_clip 必须为正")
    if not 0 < args.warmup_ratio < 1 or not 0 < args.floor_ratio <= 1:
        raise ValueError("warmup_ratio 或 floor_ratio 无效")
    if args.num_workers < 0 or args.seed < 0:
        raise ValueError("num_workers 和 seed 不能为负")
    if args.smoke_optimizer_steps is not None and not 1 <= args.smoke_optimizer_steps <= MAX_SMOKE_OPTIMIZER_STEPS:
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
        raise ValueError("HN 曝光率实验只支持单卡")


def _export_step_weights(path: Path, model: torch.nn.Module) -> str:
    return stage_f_training._export_epoch_weights(path, model)


def _verify_step_weight(path: Path) -> str:
    return stage_f_training._verify_epoch_weight(path)


def _reconcile_step_weights(
    weights_dir: Path, progress: SftProgress, model: torch.nn.Module
) -> None:
    """根据恢复点验证历史导出，并只补发当前 step 边界。"""

    for step in CHECKPOINT_STEPS:
        if step > progress.optimizer_step:
            break
        path = weights_dir / f"rag_step_{step}.pth"
        if path.is_file() or path.with_suffix(".sha256").is_file():
            _verify_step_weight(path)
        elif step == progress.optimizer_step:
            _export_step_weights(path, model)
        else:
            raise FileNotFoundError(f"缺少已完成 step {step} 的权重")


def _source_identities() -> dict[str, dict[str, object]]:
    project_root = Path(__file__).resolve().parents[2]
    paths = {
        "training": Path(__file__).resolve(),
        "exposure_sampler": project_root / "minimind" / "dataset" / "rag_sft_v2_exposure.py",
        "dataset": project_root / "minimind" / "dataset" / "rag_sft_v2_dataset.py",
        "sft_runtime": project_root / "minimind" / "trainer" / "sft_runtime.py",
    }
    return {
        name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": stage_f_training._sha256_file(path),
        }
        for name, path in paths.items()
    }


def run_training(args: argparse.Namespace) -> SftProgress:
    _validate_args(args)
    manifest_path = Path(args.manifest).resolve()
    manifest, manifest_sha = stage_f_training._load_manifest(
        manifest_path, allow_stage_experiment=False
    )
    if (
        manifest.get("pipeline") != "rag_sft_v2_training_release"
        or manifest.get("readiness", {}).get("training_ready") is not True
    ):
        raise ValueError("HN 曝光率正式实验要求 training-ready v2 release")

    run_dir = Path(args.run_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    latest = checkpoint_dir / "resume" / "latest.pt"
    weights_dir = checkpoint_dir / "weights"
    run_manifest_path = run_dir / "run-manifest.json"
    metrics_path = run_dir / "metrics.jsonl"
    if not args.resume and (run_dir.exists() or checkpoint_dir.exists()):
        raise FileExistsError("实验 run 或 checkpoint 目录已存在")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 CUDA 但当前无可用 GPU")
    if args.dtype == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 CUDA 不支持 BF16")
    parent_path = Path(args.parent_weights).resolve()
    if not parent_path.is_file() or stage_f_training._sha256_file(parent_path) != args.parent_sha256.lower():
        raise ValueError("cpt_750m 父权重 SHA-256 校验失败")

    tokenizer = base_entry._load_tokenizer(args.tokenizer_path)
    dataset = RagSftV2Dataset(
        manifest_path,
        tokenizer,
        candidate_path=args.candidate_path,
        tokenizer_path=args.tokenizer_path,
    )
    try:
        pairing = audit_clean_hn_pairing(dataset.candidate_path)
        plan = build_exposure_plan(dataset.variant_indices, group=args.group, seed=args.seed)
        stop_steps = args.smoke_optimizer_steps or COMMON_OPTIMIZER_STEPS
        warmup_steps = max(1, round(COMMON_OPTIMIZER_STEPS * args.warmup_ratio))
        controls = {"stop_optimizer_steps": stop_steps}
        invariants = {
            "pipeline": PIPELINE,
            "manifest_sha256": manifest_sha,
            "candidate_sha256": stage_f_training._sha256_file(dataset.candidate_path),
            "tokenizer": dataset.tokenizer_identity,
            "parent_role": "cpt_750m",
            "parent_sha256": args.parent_sha256.lower(),
            "source_identities": _source_identities(),
            "sequence_length": 768,
            "label_mask_version": LABEL_MASK_VERSION,
            "group": args.group,
            "hard_negative_multiplier": plan.hard_negative_multiplier,
            "unique_counts": pairing["unique_counts"],
            "paired_hn_queries": pairing["paired_hn_queries"],
            "exposure_plan_sha256": plan.sha256,
            "clean_exposures": plan.clean_exposures,
            "hard_negative_exposures": plan.hard_negative_exposures,
            "total_exposures": plan.total_exposures,
            "optimizer_steps": COMMON_OPTIMIZER_STEPS,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "seed": args.seed,
            "optimizer": OPTIMIZER,
            "peak_lr": args.peak_lr,
            "warmup_optimizer_steps": warmup_steps,
            "schedule_optimizer_steps": COMMON_OPTIMIZER_STEPS,
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
            base_entry._load_parent_weights(parent_path, args.parent_sha256, model)
        optimizer = build_sft_optimizer(model, peak_lr=args.peak_lr, device_type=device.type)
        progress = SftProgress()
        if args.resume:
            saved_manifest, _ = base_entry._load_verified_json(
                run_manifest_path, "HN 曝光率实验 run manifest"
            )
            if saved_manifest.get("training_invariants") != invariants:
                raise ValueError("恢复时 HN 曝光率实验不变量不一致")
            progress = load_sft_checkpoint(
                latest,
                model=model,
                optimizer=optimizer,
                expected_invariants=invariants,
                controls=controls,
            ).progress
            if args.smoke_optimizer_steps is None:
                _reconcile_step_weights(weights_dir, progress, model)
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
                        "candidate_path": str(dataset.candidate_path),
                    },
                    "parent_model": {
                        "role": "cpt_750m",
                        "path": str(parent_path),
                        "sha256": args.parent_sha256.lower(),
                    },
                    "training_invariants": invariants,
                },
            )

        logger = JsonlMetricLogger(metrics_path)
        sampler = RagSftV2ExposureSampler(
            plan, start_position=progress.completed_sequences
        )
        loader = DataLoader(
            dataset,
            batch_size=args.micro_batch_size,
            sampler=sampler,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=args.num_workers > 0,
        )
        try:
            for groups in group_micro_batches(loader, args.accumulation_steps):
                if progress.optimizer_step >= stop_steps:
                    break
                lr = token_learning_rate(
                    progress.optimizer_step + 1,
                    args.peak_lr,
                    warmup_steps,
                    COMMON_OPTIMIZER_STEPS,
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
                    autocast_dtype=torch.bfloat16 if args.dtype == "bfloat16" else None,
                )
                elapsed = time.perf_counter() - started
                completed_sequences = progress.completed_sequences + result.sequence_count
                epoch, position = advance_sft_position(
                    0, 0, completed_sequences, COMMON_EXPOSURES
                )
                progress = SftProgress(
                    epoch=epoch,
                    next_sequence_position=position,
                    completed_sequences=completed_sequences,
                    completed_assistant_tokens=progress.completed_assistant_tokens + result.assistant_tokens,
                    optimizer_step=progress.optimizer_step + 1,
                )
                logger.log(
                    {
                        "type": "train",
                        "group": args.group,
                        "optimizer_step": progress.optimizer_step,
                        "completed_sequences": progress.completed_sequences,
                        "completed_assistant_tokens": progress.completed_assistant_tokens,
                        "assistant_tokens": result.assistant_tokens,
                        "loss": result.loss,
                        "learning_rate": lr,
                        "grad_norm": result.grad_norm,
                        "elapsed_seconds": elapsed,
                        "assistant_tokens_per_second": result.assistant_tokens / elapsed,
                        "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                    }
                )
                if progress.optimizer_step in CHECKPOINT_STEPS or progress.optimizer_step == stop_steps:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    save_sft_checkpoint(
                        latest,
                        model=model,
                        optimizer=optimizer,
                        progress=progress,
                        invariants=invariants,
                        controls=controls,
                    )
                    if args.smoke_optimizer_steps is None:
                        _reconcile_step_weights(weights_dir, progress, model)
        finally:
            del loader

        if progress.optimizer_step != stop_steps:
            raise RuntimeError("HN 曝光率实验未到达固定 optimizer-step 终点")
        if args.smoke_optimizer_steps is None and (
            progress.completed_sequences != COMMON_EXPOSURES
            or progress.epoch != 1
            or progress.next_sequence_position != 0
        ):
            raise RuntimeError("HN 曝光率实验曝光流未精确闭合")
        logger.log(
            {
                "type": "run_complete",
                "group": args.group,
                "optimizer_step": progress.optimizer_step,
                "completed_sequences": progress.completed_sequences,
                "completed_assistant_tokens": progress.completed_assistant_tokens,
                "weights_exported": args.smoke_optimizer_steps is None,
            }
        )
        return progress
    finally:
        dataset.close()


def main() -> None:
    progress = run_training(build_parser().parse_args())
    print(
        "RAG_SFT_V2_HN_EXPOSURE_OK "
        f"optimizer_step={progress.optimizer_step} sequences={progress.completed_sequences}"
    )


if __name__ == "__main__":
    main()
