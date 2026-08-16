"""从预打包 token shards 训练阶段 B 的 127M Dense MiniMind。"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from ..dataset.pretrain_dataset import (
        DeterministicPretrainSampler,
        PretrainShardCatalog,
    )
    from ..model.model_minimind import MiniMindConfig, MiniMindForCausalLM
except ImportError:
    from dataset.pretrain_dataset import (
        DeterministicPretrainSampler,
        PretrainShardCatalog,
    )
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

from .pretrain_runtime import (
    JsonlMetricLogger,
    PretrainCheckpointError,
    PretrainProgress,
    advance_pretrain_position,
    build_pretrain_optimizer,
    crossed_token_events,
    evaluate_pretrain_by_source,
    export_pretrain_weights,
    group_micro_batches,
    load_pretrain_checkpoint,
    run_pretrain_optimizer_step,
    save_pretrain_checkpoint,
    token_learning_rate,
)
from .trainer_utils import Logger, get_model_params, setup_seed


DEFAULT_WORK_ROOT = "/root/autodl-tmp/minimind-work"
DEFAULT_SHARD_MANIFEST = (
    "/root/autodl-tmp/minimind-work/manifests/stage-b-shards-v1.json"
)
EXPECTED_SEQUENCE_LENGTH = 768
EFFECTIVE_BATCH_SEQUENCES = 256
RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("不能是负数")
    return parsed


def _token_thresholds(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    try:
        thresholds = tuple(sorted({int(item) for item in value.split(",")}))
    except ValueError as error:
        raise argparse.ArgumentTypeError("token 阈值必须是逗号分隔的整数") from error
    if any(threshold <= 0 for threshold in thresholds):
        raise argparse.ArgumentTypeError("token 阈值必须为正数")
    return thresholds


def build_parser() -> argparse.ArgumentParser:
    """创建阶段 B 正式预训练 CLI。"""
    parser = argparse.ArgumentParser(description="MiniMind 阶段 B 通用预训练")
    parser.add_argument("--run_name", required=True, help="实验名称和本地目录名")
    parser.add_argument(
        "--work_root",
        default=os.environ.get("MINIMIND_WORK_ROOT", DEFAULT_WORK_ROOT),
        help="minimind-work 根目录",
    )
    parser.add_argument(
        "--shard_manifest",
        default=DEFAULT_SHARD_MANIFEST,
        help="stage-b token shard manifest",
    )
    parser.add_argument("--resume", action="store_true", help="严格恢复 latest.pt")

    parser.add_argument("--peak_lr", type=float, default=5e-4)
    parser.add_argument("--stop_tokens", type=_positive_int, default=4_868_681_472)
    parser.add_argument("--warmup_tokens", type=_positive_int, default=50_000_000)
    parser.add_argument("--schedule_tokens", type=_positive_int, default=4_868_681_472)
    parser.add_argument("--floor_ratio", type=float, default=0.1)
    parser.add_argument("--micro_batch_size", type=_positive_int, default=16)
    parser.add_argument("--accumulation_steps", type=_positive_int, default=16)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--log_interval_tokens", type=_positive_int, default=10_000_000)
    parser.add_argument(
        "--quick_interval_tokens",
        type=_non_negative_int,
        default=100_000_000,
    )
    parser.add_argument(
        "--full_interval_tokens",
        type=_non_negative_int,
        default=500_000_000,
    )
    parser.add_argument("--fixed_quick_tokens", type=_token_thresholds, default=())
    parser.add_argument("--fixed_full_tokens", type=_token_thresholds, default=())
    parser.add_argument("--eval_batch_size", type=_positive_int, default=64)

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--num_workers", type=_non_negative_int, default=8)
    parser.add_argument("--eval_num_workers", type=_non_negative_int, default=4)
    parser.add_argument("--seed", type=_non_negative_int, default=42)
    parser.add_argument("--use_compile", action="store_true")

    parser.add_argument(
        "--swanlab",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用或禁用 SwanLab；本地 JSONL 始终启用",
    )
    parser.add_argument("--swanlab_project", default="minimind+rag")
    parser.add_argument("--swanlab_workspace", default="Bigwatermelon")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if RUN_NAME_RE.fullmatch(args.run_name) is None:
        raise ValueError("run_name 只能包含字母、数字、点、下划线和连字符")
    if args.micro_batch_size * args.accumulation_steps != EFFECTIVE_BATCH_SEQUENCES:
        raise ValueError("micro_batch_size × accumulation_steps 必须等于 256")
    if args.peak_lr <= 0 or args.grad_clip <= 0:
        raise ValueError("peak_lr 和 grad_clip 必须大于 0")
    if not 0 < args.floor_ratio <= 1:
        raise ValueError("floor_ratio 必须位于 (0, 1] 区间")
    if args.schedule_tokens <= args.warmup_tokens:
        raise ValueError("schedule_tokens 必须大于 warmup_tokens")
    if int(os.environ.get("RANK", -1)) != -1:
        raise RuntimeError("阶段 B 当前只支持单卡训练，不能使用 torchrun/DDP")


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    work_root = Path(args.work_root).resolve()
    run_dir = work_root / "runs" / "stage-b" / args.run_name
    checkpoint_dir = work_root / "checkpoints" / "stage-b" / args.run_name
    return {
        "run_dir": run_dir,
        "metrics": run_dir / "metrics.jsonl",
        "checkpoint_dir": checkpoint_dir,
        "resume": checkpoint_dir / "resume" / "latest.pt",
        "weights": checkpoint_dir / "weights",
    }


def _model_architecture(config: MiniMindConfig) -> dict[str, Any]:
    return {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "intermediate_size": config.intermediate_size,
        "max_position_embeddings": config.max_position_embeddings,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "tie_word_embeddings": config.tie_word_embeddings,
        "use_moe": config.use_moe,
    }


def _training_invariants(
    args: argparse.Namespace,
    catalog: PretrainShardCatalog,
    config: MiniMindConfig,
    train_sequence_count: int,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    return {
        "shard_manifest_sha256": catalog.manifest_sha256,
        "sequence_length": catalog.sequence_length,
        "train_sequence_count": train_sequence_count,
        "model": _model_architecture(config),
        "sampler_seed": args.seed,
        "micro_batch_size": args.micro_batch_size,
        "accumulation_steps": args.accumulation_steps,
        "effective_batch_sequences": EFFECTIVE_BATCH_SEQUENCES,
        "dtype": args.dtype,
        "peak_lr": args.peak_lr,
        "warmup_tokens": args.warmup_tokens,
        "schedule_tokens": args.schedule_tokens,
        "floor_ratio": args.floor_ratio,
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.95],
            "weight_decay": 0.1,
            "eps": 1e-8,
            "fused": bool(optimizer.defaults.get("fused", False)),
        },
        "grad_clip": args.grad_clip,
        "use_compile": args.use_compile,
        "num_workers": args.num_workers,
    }


def _training_controls(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "stop_tokens": args.stop_tokens,
        "log_interval_tokens": args.log_interval_tokens,
        "quick_interval_tokens": args.quick_interval_tokens,
        "full_interval_tokens": args.full_interval_tokens,
        "fixed_quick_tokens": list(args.fixed_quick_tokens),
        "fixed_full_tokens": list(args.fixed_full_tokens),
        "eval_batch_size": args.eval_batch_size,
        "eval_num_workers": args.eval_num_workers,
    }


def _validate_resume_progress(
    progress: PretrainProgress,
    *,
    sequence_count: int,
    sequence_length: int,
) -> None:
    expected_sequences = progress.epoch * sequence_count + progress.next_sequence_position
    if progress.next_sequence_position >= sequence_count and progress.next_sequence_position != 0:
        raise PretrainCheckpointError("恢复点 sequence 位置超过当前 epoch")
    if progress.completed_sequences != expected_sequences:
        raise PretrainCheckpointError("恢复点累计 sequence 与 epoch/position 不闭合")
    if progress.completed_tokens != progress.completed_sequences * sequence_length:
        raise PretrainCheckpointError("恢复点累计 token 与 sequence 数不闭合")


def _init_swanlab(
    args: argparse.Namespace,
    run_dir: Path,
    config: dict[str, Any],
    resume_run_id: str | None,
) -> tuple[object | None, object | None]:
    if not args.swanlab:
        return None, None
    try:
        import swanlab

        run = swanlab.init(
            project=args.swanlab_project,
            workspace=args.swanlab_workspace,
            experiment_name=args.run_name,
            config=config,
            logdir=str(run_dir / "swanlog"),
            id=resume_run_id,
            resume="must" if resume_run_id else None,
        )
        return swanlab, run
    except Exception as error:
        warnings.warn(
            f"SwanLab 初始化失败，将只写本地 JSONL: {error}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None, None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _crossed_interval(previous_tokens: int, completed_tokens: int, interval: int) -> bool:
    return completed_tokens // interval > previous_tokens // interval


def _validation_record(
    report: dict[str, Any],
    *,
    split: str,
    progress: PretrainProgress,
    eval_batch_size: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "type": "validation",
        "timestamp": _utc_now(),
        "split": split,
        "optimizer_step": progress.optimizer_step,
        "completed_tokens": progress.completed_tokens,
        "eval_batch_size": eval_batch_size,
        "validation/overall/loss": report["overall"]["loss"],
        "validation/overall/perplexity": report["overall"]["perplexity"],
        "validation/overall/prediction_tokens": report["overall"][
            "prediction_tokens"
        ],
    }
    for source, source_report in report["sources"].items():
        prefix = f"validation/{source}"
        record[f"{prefix}/loss"] = source_report["loss"]
        record[f"{prefix}/perplexity"] = source_report["perplexity"]
        record[f"{prefix}/prediction_tokens"] = source_report["prediction_tokens"]
    return record


def _evaluate_with_oom_fallback(
    model: torch.nn.Module,
    catalog: PretrainShardCatalog,
    *,
    split: str,
    args: argparse.Namespace,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> tuple[dict[str, Any], int]:
    batch_size = args.eval_batch_size
    try:
        report = evaluate_pretrain_by_source(
            model,
            catalog,
            split=split,
            batch_size=batch_size,
            device=device,
            num_workers=args.eval_num_workers,
            autocast_dtype=autocast_dtype,
        )
        return report, batch_size
    except torch.OutOfMemoryError:
        if device.type != "cuda" or batch_size <= 32:
            raise
        Logger(f"验证 batch_size={batch_size} OOM，清理显存后回退到 32")
        torch.cuda.empty_cache()
        report = evaluate_pretrain_by_source(
            model,
            catalog,
            split=split,
            batch_size=32,
            device=device,
            num_workers=args.eval_num_workers,
            autocast_dtype=autocast_dtype,
        )
        return report, 32


def _current_swanlab_run_id(run: object | None, fallback: str | None) -> str | None:
    run_id = getattr(run, "id", None) if run is not None else None
    return run_id if isinstance(run_id, str) and run_id else fallback


def run_training(args: argparse.Namespace) -> PretrainProgress:
    """执行单卡阶段 B 训练，并返回最后一个 optimizer 边界进度。"""
    _validate_args(args)
    paths = _resolve_paths(args)
    resume_path = paths["resume"]
    if args.resume and not resume_path.is_file():
        raise FileNotFoundError(f"--resume 指定的恢复点不存在: {resume_path}")
    if not args.resume and (resume_path.exists() or paths["metrics"].exists()):
        raise FileExistsError("新 run 已存在指标或恢复点；请更换 run_name 或使用 --resume")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前环境没有可用 GPU")
    if args.dtype == "bfloat16" and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 CUDA 设备不支持 BF16")
    autocast_dtype = torch.bfloat16 if args.dtype == "bfloat16" else None

    setup_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    catalog = PretrainShardCatalog(args.shard_manifest)
    if catalog.sequence_length != EXPECTED_SEQUENCE_LENGTH:
        raise ValueError(
            f"阶段 B sequence_length 必须是 {EXPECTED_SEQUENCE_LENGTH}，"
            f"实际为 {catalog.sequence_length}"
        )
    train_dataset = catalog.create_dataset("train")
    if len(train_dataset) == 0:
        raise ValueError("train split 不能为空")

    model_config = MiniMindConfig(
        vocab_size=12_000,
        hidden_size=768,
        num_hidden_layers=16,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=2432,
        use_moe=False,
    )
    model = MiniMindForCausalLM(model_config).to(device)
    get_model_params(model, model_config)
    optimizer = build_pretrain_optimizer(
        model,
        peak_lr=args.peak_lr,
        device_type=device.type,
    )
    invariants = _training_invariants(
        args,
        catalog,
        model_config,
        len(train_dataset),
        optimizer,
    )
    controls = _training_controls(args)

    progress = PretrainProgress()
    resume_run_id = None
    if args.resume:
        loaded = load_pretrain_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            expected_invariants=invariants,
            controls=controls,
        )
        progress = loaded.progress
        resume_run_id = loaded.swanlab_run_id
        _validate_resume_progress(
            progress,
            sequence_count=len(train_dataset),
            sequence_length=catalog.sequence_length,
        )
        Logger(
            f"恢复成功: tokens={progress.completed_tokens:,}, "
            f"epoch={progress.epoch}, position={progress.next_sequence_position:,}"
        )

    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
    tracker, swanlab_run = _init_swanlab(
        args,
        paths["run_dir"],
        {**invariants, **controls},
        resume_run_id,
    )
    active_run_id = _current_swanlab_run_id(swanlab_run, resume_run_id)
    metric_logger = JsonlMetricLogger(paths["metrics"], tracker=tracker)
    metric_logger.log(
        {
            "type": "run_start" if not args.resume else "run_resume",
            "timestamp": _utc_now(),
            "optimizer_step": progress.optimizer_step,
            "completed_tokens": progress.completed_tokens,
            "run_name": args.run_name,
        }
    )

    if args.use_compile:
        model = torch.compile(model)
        Logger("torch.compile 已启用")

    fixed_quick_tokens = tuple(
        sorted({args.warmup_tokens, *args.fixed_quick_tokens})
    )
    fixed_full_tokens = tuple(
        sorted({args.schedule_tokens, *args.fixed_full_tokens})
    )
    last_saved_tokens = progress.completed_tokens if args.resume else -1
    training_started = time.perf_counter()
    try:
        while progress.completed_tokens < args.stop_tokens:
            sampler = DeterministicPretrainSampler(
                train_dataset,
                seed=args.seed,
                epoch=progress.epoch,
                start_position=progress.next_sequence_position,
            )
            loader = DataLoader(
                train_dataset,
                batch_size=args.micro_batch_size,
                sampler=sampler,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.num_workers > 0,
            )
            reached_stop = False
            for micro_batches in group_micro_batches(loader, args.accumulation_steps):
                previous_tokens = progress.completed_tokens
                group_sequences = sum(batch[0].shape[0] for batch in micro_batches)
                projected_tokens = (
                    previous_tokens + group_sequences * catalog.sequence_length
                )
                learning_rate = token_learning_rate(
                    projected_tokens,
                    args.peak_lr,
                    args.warmup_tokens,
                    args.schedule_tokens,
                    args.floor_ratio,
                )
                step_started = time.perf_counter()
                step_metrics = run_pretrain_optimizer_step(
                    model,
                    optimizer,
                    micro_batches,
                    device=device,
                    learning_rate=learning_rate,
                    grad_clip=args.grad_clip,
                    autocast_dtype=autocast_dtype,
                )
                epoch, next_position = advance_pretrain_position(
                    progress.epoch,
                    progress.next_sequence_position,
                    step_metrics.sequence_count,
                    len(train_dataset),
                )
                progress = PretrainProgress(
                    epoch=epoch,
                    next_sequence_position=next_position,
                    completed_sequences=(
                        progress.completed_sequences + step_metrics.sequence_count
                    ),
                    completed_tokens=previous_tokens + step_metrics.input_tokens,
                    optimizer_step=progress.optimizer_step + 1,
                )

                should_log = progress.optimizer_step == 1 or _crossed_interval(
                    previous_tokens,
                    progress.completed_tokens,
                    args.log_interval_tokens,
                )
                if should_log:
                    step_seconds = time.perf_counter() - step_started
                    elapsed_seconds = time.perf_counter() - training_started
                    record = {
                        "type": "train",
                        "timestamp": _utc_now(),
                        "optimizer_step": progress.optimizer_step,
                        "epoch": progress.epoch,
                        "next_sequence_position": progress.next_sequence_position,
                        "completed_sequences": progress.completed_sequences,
                        "completed_tokens": progress.completed_tokens,
                        "loss": step_metrics.loss,
                        "logits_loss": step_metrics.logits_loss,
                        "aux_loss": step_metrics.aux_loss,
                        "learning_rate": learning_rate,
                        "grad_norm": step_metrics.grad_norm,
                        "step_tokens": step_metrics.input_tokens,
                        "step_tokens_per_second": step_metrics.input_tokens / step_seconds,
                        "elapsed_seconds": elapsed_seconds,
                    }
                    if device.type == "cuda":
                        record["cuda/max_memory_allocated_bytes"] = (
                            torch.cuda.max_memory_allocated(device)
                        )
                        record["cuda/max_memory_reserved_bytes"] = (
                            torch.cuda.max_memory_reserved(device)
                        )
                    metric_logger.log(record)
                    Logger(
                        f"step={progress.optimizer_step:,} "
                        f"tokens={progress.completed_tokens:,} "
                        f"loss={step_metrics.loss:.4f} lr={learning_rate:.3e} "
                        f"tokens/s={record['step_tokens_per_second']:,.0f}"
                    )

                actions = crossed_token_events(
                    previous_tokens,
                    progress.completed_tokens,
                    quick_interval_tokens=args.quick_interval_tokens,
                    full_interval_tokens=args.full_interval_tokens,
                    fixed_quick_tokens=fixed_quick_tokens,
                    fixed_full_tokens=fixed_full_tokens,
                )
                if actions.quick_validation or actions.full_validation:
                    split = (
                        "full_validation"
                        if actions.full_validation
                        else "quick_validation"
                    )
                    report, actual_eval_batch_size = _evaluate_with_oom_fallback(
                        model,
                        catalog,
                        split=split,
                        args=args,
                        device=device,
                        autocast_dtype=autocast_dtype,
                    )
                    metric_logger.log(
                        _validation_record(
                            report,
                            split=split,
                            progress=progress,
                            eval_batch_size=actual_eval_batch_size,
                        )
                    )
                    Logger(
                        f"{split}: tokens={progress.completed_tokens:,} "
                        f"loss={report['overall']['loss']:.4f} "
                        f"ppl={report['overall']['perplexity']:.2f}"
                    )

                if actions.save_resume:
                    save_pretrain_checkpoint(
                        resume_path,
                        model=model,
                        optimizer=optimizer,
                        progress=progress,
                        invariants=invariants,
                        controls=controls,
                        swanlab_run_id=active_run_id,
                    )
                    last_saved_tokens = progress.completed_tokens
                if actions.export_weights:
                    weight_path = (
                        paths["weights"]
                        / f"pretrain-{progress.completed_tokens}.pth"
                    )
                    export_pretrain_weights(weight_path, model)

                if progress.completed_tokens >= args.stop_tokens:
                    reached_stop = True
                    break
            del loader
            if reached_stop:
                break

        if last_saved_tokens != progress.completed_tokens:
            save_pretrain_checkpoint(
                resume_path,
                model=model,
                optimizer=optimizer,
                progress=progress,
                invariants=invariants,
                controls=controls,
                swanlab_run_id=active_run_id,
            )
        metric_logger.log(
            {
                "type": "run_complete",
                "timestamp": _utc_now(),
                "optimizer_step": progress.optimizer_step,
                "completed_tokens": progress.completed_tokens,
                "epoch": progress.epoch,
                "next_sequence_position": progress.next_sequence_position,
            }
        )
        return progress
    finally:
        train_dataset.close()
        if tracker is not None:
            try:
                tracker.finish()
            except Exception as error:
                warnings.warn(
                    f"SwanLab finish 失败: {error}",
                    RuntimeWarning,
                    stacklevel=2,
                )


def main() -> None:
    args = build_parser().parse_args()
    progress = run_training(args)
    Logger(
        f"训练结束: tokens={progress.completed_tokens:,}, "
        f"optimizer_steps={progress.optimizer_step:,}"
    )


if __name__ == "__main__":
    main()
