"""从基础法律 SFT model-only 权重启动独立 Query-SFT 阶段。"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from ..dataset.pretrain_dataset import DeterministicPretrainSampler
    from ..dataset.query_sft_dataset import (
        LABEL_MASK_VERSION,
        MAX_SEQ_LEN,
        PIPELINE as RELEASE_PIPELINE,
        QuerySftDataset,
    )
except ImportError:
    from dataset.pretrain_dataset import DeterministicPretrainSampler
    from dataset.query_sft_dataset import (
        LABEL_MASK_VERSION,
        MAX_SEQ_LEN,
        PIPELINE as RELEASE_PIPELINE,
        QuerySftDataset,
    )

from . import train_full_sft as base_entry
from .pretrain_runtime import group_micro_batches
from .sft_runtime import (
    JsonlMetricLogger,
    SftProgress,
    advance_sft_position,
    assistant_token_learning_rate,
    build_sft_optimizer,
    export_sft_weights,
    load_sft_checkpoint,
    run_sft_optimizer_step,
    save_sft_checkpoint,
)


PIPELINE = "query_enhancement_sft"
MAX_SMOKE_OPTIMIZER_STEPS = 10
MODEL_ARCHITECTURE = base_entry.MODEL_ARCHITECTURE
MiniMindForCausalLM = base_entry.MiniMindForCausalLM
Logger = base_entry.Logger
get_model_params = base_entry.get_model_params
setup_seed = base_entry.setup_seed


def build_parser() -> argparse.ArgumentParser:
    """创建 Query-SFT 单卡正式训练与受限冒烟 CLI。"""

    parser = argparse.ArgumentParser(description="MiniMind 独立 Query-SFT")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--release_manifest", required=True)
    parser.add_argument("--candidate_path", required=True)
    parser.add_argument("--evaluation_manifest", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--parent_weights", required=True)
    parser.add_argument("--parent_sha256", type=base_entry._sha256_value, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--pilot_pipeline",
        action="store_true",
        help="允许 pilot release 跑通训练、导出和后续推理，但不得参与父权重排名",
    )
    parser.add_argument(
        "--smoke_optimizer_steps",
        type=base_entry.pretrain_entry._positive_int,
        help="设置后进入不导出权重的 1-10 step 工程冒烟模式",
    )
    parser.add_argument("--peak_lr", type=float, required=True)
    parser.add_argument(
        "--stop_assistant_tokens",
        type=base_entry.pretrain_entry._positive_int,
        required=True,
    )
    parser.add_argument(
        "--warmup_assistant_tokens",
        type=base_entry.pretrain_entry._positive_int,
        required=True,
    )
    parser.add_argument(
        "--schedule_assistant_tokens",
        type=base_entry.pretrain_entry._positive_int,
        required=True,
    )
    parser.add_argument("--floor_ratio", type=float, default=0.1)
    parser.add_argument(
        "--micro_batch_size",
        type=base_entry.pretrain_entry._positive_int,
        required=True,
    )
    parser.add_argument(
        "--accumulation_steps",
        type=base_entry.pretrain_entry._positive_int,
        required=True,
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--log_interval_tokens",
        type=base_entry.pretrain_entry._positive_int,
        default=20_000,
    )
    parser.add_argument(
        "--checkpoint_interval_tokens",
        type=base_entry.pretrain_entry._non_negative_int,
        default=0,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--num_workers", type=base_entry.pretrain_entry._non_negative_int, default=4
    )
    parser.add_argument("--seed", type=base_entry.pretrain_entry._non_negative_int, default=42)
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--swanlab", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--swanlab_project", default="minimind+query")
    parser.add_argument("--swanlab_workspace", default="Bigwatermelon")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if base_entry.RUN_NAME_RE.fullmatch(args.run_name) is None:
        raise ValueError("run_name 只能包含字母、数字、点、下划线和连字符")
    if args.peak_lr <= 0 or args.grad_clip <= 0:
        raise ValueError("peak_lr 和 grad_clip 必须大于 0")
    if not 0 < args.floor_ratio <= 1:
        raise ValueError("floor_ratio 必须位于 (0, 1] 区间")
    if args.schedule_assistant_tokens <= args.warmup_assistant_tokens:
        raise ValueError("schedule_assistant_tokens 必须大于 warmup_assistant_tokens")
    if Path(args.run_dir).resolve() == Path(args.checkpoint_dir).resolve():
        raise ValueError("run_dir 和 checkpoint_dir 必须分离")
    if args.smoke_optimizer_steps is not None:
        if args.resume:
            raise ValueError("Query-SFT smoke 不支持恢复")
        if args.smoke_optimizer_steps > MAX_SMOKE_OPTIMIZER_STEPS:
            raise ValueError("Query-SFT smoke 最多允许 10 个 optimizer steps")
    if args.pilot_pipeline and args.smoke_optimizer_steps is not None:
        raise ValueError("--pilot_pipeline 与 --smoke_optimizer_steps 不能同时使用")
    if int(os.environ.get("RANK", -1)) != -1:
        raise RuntimeError("Query-SFT 当前只支持单卡，不能使用 torchrun/DDP")


def _file_identity(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"输入文件不存在: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": base_entry._sha256_file(resolved),
    }


def _identity_matches(expected: object, actual: dict[str, object]) -> bool:
    return isinstance(expected, dict) and all(
        expected.get(key) == actual[key] for key in ("bytes", "sha256")
    )


def _load_query_training_inputs(
    release_manifest_path: str | Path,
    candidate_path: str | Path,
    evaluation_manifest_path: str | Path,
    *,
    smoke: bool,
    pilot_pipeline: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """复算 Query release、candidate 与评估隔离身份。"""

    release_path = Path(release_manifest_path).resolve()
    candidate = Path(candidate_path).resolve()
    evaluation_path = Path(evaluation_manifest_path).resolve()
    release, release_sha = base_entry._load_verified_json(
        release_path, "Query-SFT release manifest"
    )
    evaluation, evaluation_sha = base_entry._load_verified_json(
        evaluation_path, "评估排除 manifest"
    )
    candidate_identity = _file_identity(candidate)
    readiness = release.get("readiness")
    release_status = release.get("release_status")
    if (
        release.get("schema_version") != "1.0"
        or release.get("pipeline") != RELEASE_PIPELINE
        or release.get("complete") is not True
        or not isinstance(readiness, dict)
        or readiness.get("training_ready") is not True
        or release_status
        not in {"pilot_training_candidate", "formal_training_candidate"}
    ):
        raise ValueError("Query-SFT release 尚未训练就绪")
    if release_status == "pilot_training_candidate" and not (
        smoke or pilot_pipeline
    ):
        raise ValueError("pilot release 只能用于 Query-SFT smoke 或 pilot_pipeline")
    if pilot_pipeline and release_status != "pilot_training_candidate":
        raise ValueError("pilot_pipeline 必须使用 pilot training release")
    release_candidate = release.get("data", {}).get("training_candidate")
    if not _identity_matches(release_candidate, candidate_identity):
        raise ValueError("Query-SFT candidate 身份不一致")
    records = release_candidate.get("records") if isinstance(release_candidate, dict) else None
    if type(records) is not int or records <= 0:
        raise ValueError("Query-SFT candidate 记录数无效")
    evaluation_identity = _file_identity(evaluation_path)
    if not _identity_matches(release.get("evaluation_exclusions"), evaluation_identity):
        raise ValueError("Query-SFT release 与评估排除身份不一致")
    if (
        evaluation.get("schema_version") != "1.2"
        or evaluation.get("pipeline") != "legal_sft_evaluation_exclusions"
        or evaluation.get("complete_for_formal_sft") is not True
    ):
        raise ValueError("Query-SFT 评估隔离 manifest 尚未正式就绪")
    return release, {
        "release_manifest_path": str(release_path),
        "release_manifest_sha256": release_sha,
        "release_status": release_status,
        "candidate_path": str(candidate),
        "candidate_sha256": candidate_identity["sha256"],
        "candidate_bytes": candidate_identity["bytes"],
        "candidate_records": records,
        "evaluation_manifest_path": str(evaluation_path),
        "evaluation_manifest_sha256": evaluation_sha,
        "formal_training_ready": release_status == "formal_training_candidate",
    }


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    run_dir = Path(args.run_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    return {
        "run_dir": run_dir,
        "metrics": run_dir / "metrics.jsonl",
        "run_manifest": run_dir / "run-manifest.json",
        "checkpoint_dir": checkpoint_dir,
        "resume": checkpoint_dir / "resume" / "latest.pt",
        "weights": checkpoint_dir / "weights",
    }


def _training_invariants(args, *, identity, tokenizer_identity, optimizer):
    mode = (
        "smoke"
        if args.smoke_optimizer_steps is not None
        else "pilot_pipeline"
        if args.pilot_pipeline
        else "train"
    )
    return {
        "pipeline": PIPELINE,
        "mode": mode,
        "release_manifest_sha256": identity["release_manifest_sha256"],
        "candidate_sha256": identity["candidate_sha256"],
        "evaluation_manifest_sha256": identity["evaluation_manifest_sha256"],
        "parent_weights_sha256": args.parent_sha256,
        "parent_stage": "base_legal_sft_model_only",
        "model": MODEL_ARCHITECTURE,
        "tokenizer": tokenizer_identity,
        "sequence_length": MAX_SEQ_LEN,
        "label_mask_version": LABEL_MASK_VERSION,
        "train_sequence_count": identity["candidate_records"],
        "sampler_seed": args.seed,
        "micro_batch_size": args.micro_batch_size,
        "accumulation_steps": args.accumulation_steps,
        "dtype": args.dtype,
        "peak_lr": args.peak_lr,
        "warmup_assistant_tokens": args.warmup_assistant_tokens,
        "schedule_assistant_tokens": args.schedule_assistant_tokens,
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
    }


def _controls(args):
    return {
        "stop_assistant_tokens": args.stop_assistant_tokens,
        "log_interval_tokens": args.log_interval_tokens,
        "checkpoint_interval_tokens": args.checkpoint_interval_tokens,
        "smoke_optimizer_steps": args.smoke_optimizer_steps,
    }


def _group_assistant_tokens(micro_batches) -> int:
    return sum(int((labels[:, 1:] != -100).sum().item()) for _, labels in micro_batches)


def _run_manifest_payload(args, *, identity, tokenizer_identity, invariants, controls):
    return {
        "schema_version": "1.0",
        "pipeline": PIPELINE,
        "run_name": args.run_name,
        "created_at": base_entry.pretrain_entry._utc_now(),
        "experimental_scope": (
            "bounded_smoke_unbound_to_parent_ranking"
            if args.smoke_optimizer_steps is not None
            else "pilot_pipeline_unbound_to_parent_ranking"
            if args.pilot_pipeline
            else None
        ),
        "parent_model": {
            "stage": "base_legal_sft_model_only",
            "path": str(Path(args.parent_weights).resolve()),
            "sha256": args.parent_sha256,
            "load_semantics": "strict_model_state_only",
        },
        "data": identity,
        "tokenizer": tokenizer_identity,
        "training_invariants": invariants,
        "initial_controls": controls,
        "arguments": base_entry._jsonable_arguments(args),
    }


def run_training(args: argparse.Namespace) -> SftProgress:
    """执行 Query-SFT 正式训练或最多 10-step 的工程冒烟。"""

    _validate_args(args)
    paths = _resolve_paths(args)
    if args.resume:
        if not paths["resume"].is_file():
            raise FileNotFoundError(f"--resume 指定的恢复点不存在: {paths['resume']}")
    elif paths["run_dir"].exists() or paths["checkpoint_dir"].exists():
        raise FileExistsError("新 Query-SFT run 目录已存在；请更换目录或使用 --resume")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前环境没有可用 GPU")
    if args.dtype == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 CUDA 设备不支持 BF16")
    autocast_dtype = torch.bfloat16 if args.dtype == "bfloat16" else None
    _, identity = _load_query_training_inputs(
        args.release_manifest,
        args.candidate_path,
        args.evaluation_manifest,
        smoke=args.smoke_optimizer_steps is not None,
        pilot_pipeline=args.pilot_pipeline,
    )
    tokenizer = base_entry._load_tokenizer(args.tokenizer_path)
    tokenizer_record = base_entry._tokenizer_identity(tokenizer, args.tokenizer_path)
    train_dataset = QuerySftDataset(
        args.release_manifest,
        tokenizer,
        candidate_path=args.candidate_path,
    )
    if len(train_dataset) != identity["candidate_records"]:
        train_dataset.close()
        raise ValueError("Query-SFT Dataset 记录数与 release 不一致")
    tracker = None
    try:
        setup_seed(args.seed)
        torch.set_float32_matmul_precision("high")
        model_config = base_entry._model_config()
        model = MiniMindForCausalLM(model_config).to(device)
        get_model_params(model, model_config)
        if not args.resume:
            base_entry._load_parent_weights(args.parent_weights, args.parent_sha256, model)
        optimizer = build_sft_optimizer(model, peak_lr=args.peak_lr, device_type=device.type)
        invariants = _training_invariants(
            args,
            identity=identity,
            tokenizer_identity=tokenizer_record,
            optimizer=optimizer,
        )
        controls = _controls(args)
        progress = SftProgress()
        resume_run_id = None
        if args.resume:
            run_manifest, run_manifest_sha256 = base_entry._load_verified_run_manifest(
                paths["run_manifest"]
            )
            if run_manifest.get("run_name") != args.run_name:
                raise ValueError("Query-SFT run manifest 的 run_name 与当前命令不一致")
            if run_manifest.get("training_invariants") != invariants:
                raise ValueError("Query-SFT run manifest 训练不变量与当前配置不一致")
            loaded = load_sft_checkpoint(
                paths["resume"],
                model=model,
                optimizer=optimizer,
                expected_invariants=invariants,
                controls=controls,
            )
            progress = loaded.progress
            resume_run_id = loaded.swanlab_run_id
            if progress.next_sequence_position > len(train_dataset):
                raise ValueError("Query-SFT 恢复点数据位置超过当前 Dataset")
        else:
            run_manifest_sha256 = base_entry._write_immutable_run_manifest(
                paths["run_manifest"],
                _run_manifest_payload(
                    args,
                    identity=identity,
                    tokenizer_identity=tokenizer_record,
                    invariants=invariants,
                    controls=controls,
                ),
            )
        paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
        tracker, swanlab_run = base_entry.pretrain_entry._init_swanlab(
            args, paths["run_dir"], {**invariants, **controls}, resume_run_id
        )
        active_run_id = base_entry.pretrain_entry._current_swanlab_run_id(
            swanlab_run, resume_run_id
        )
        metric_logger = JsonlMetricLogger(paths["metrics"], tracker=tracker)
        metric_logger.log(
            {
                "type": "run_resume" if args.resume else "run_start",
                "timestamp": base_entry.pretrain_entry._utc_now(),
                "run_name": args.run_name,
                "completed_assistant_tokens": progress.completed_assistant_tokens,
                "run_manifest_sha256": run_manifest_sha256,
                "controls": controls,
            }
        )
        if args.use_compile:
            model = torch.compile(model)
        last_saved_tokens = progress.completed_assistant_tokens if args.resume else -1
        started = time.perf_counter()
        stop_reason = "assistant_token_budget_reached"
        stop_before_next_group = False
        while progress.completed_assistant_tokens < args.stop_assistant_tokens:
            if args.smoke_optimizer_steps is not None and progress.optimizer_step >= args.smoke_optimizer_steps:
                stop_reason = "smoke_optimizer_step_limit_reached"
                break
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
            for micro_batches in group_micro_batches(loader, args.accumulation_steps):
                previous_tokens = progress.completed_assistant_tokens
                projected_tokens = previous_tokens + _group_assistant_tokens(micro_batches)
                if projected_tokens > args.stop_assistant_tokens:
                    stop_reason = "next_optimizer_step_would_exceed_budget"
                    stop_before_next_group = True
                    break
                learning_rate = assistant_token_learning_rate(
                    projected_tokens,
                    args.peak_lr,
                    args.warmup_assistant_tokens,
                    args.schedule_assistant_tokens,
                    args.floor_ratio,
                )
                step_started = time.perf_counter()
                metrics = run_sft_optimizer_step(
                    model,
                    optimizer,
                    micro_batches,
                    device=device,
                    learning_rate=learning_rate,
                    grad_clip=args.grad_clip,
                    autocast_dtype=autocast_dtype,
                )
                epoch, next_position = advance_sft_position(
                    progress.epoch,
                    progress.next_sequence_position,
                    metrics.sequence_count,
                    len(train_dataset),
                )
                progress = SftProgress(
                    epoch=epoch,
                    next_sequence_position=next_position,
                    completed_sequences=progress.completed_sequences + metrics.sequence_count,
                    completed_assistant_tokens=previous_tokens + metrics.assistant_tokens,
                    optimizer_step=progress.optimizer_step + 1,
                )
                if progress.optimizer_step == 1 or base_entry.pretrain_entry._crossed_interval(
                    previous_tokens,
                    progress.completed_assistant_tokens,
                    args.log_interval_tokens,
                ):
                    elapsed = time.perf_counter() - step_started
                    metric_logger.log(
                        {
                            "type": "train",
                            "timestamp": base_entry.pretrain_entry._utc_now(),
                            "optimizer_step": progress.optimizer_step,
                            "epoch": progress.epoch,
                            "next_sequence_position": progress.next_sequence_position,
                            "completed_sequences": progress.completed_sequences,
                            "completed_assistant_tokens": progress.completed_assistant_tokens,
                            "loss": metrics.loss,
                            "learning_rate": learning_rate,
                            "grad_norm": metrics.grad_norm,
                            "step_input_tokens": metrics.input_tokens,
                            "step_assistant_tokens": metrics.assistant_tokens,
                            "assistant_tokens_per_second": metrics.assistant_tokens / elapsed,
                            "elapsed_seconds": time.perf_counter() - started,
                        }
                    )
                if args.checkpoint_interval_tokens > 0 and base_entry.pretrain_entry._crossed_interval(
                    previous_tokens,
                    progress.completed_assistant_tokens,
                    args.checkpoint_interval_tokens,
                ):
                    save_sft_checkpoint(
                        paths["resume"],
                        model=model,
                        optimizer=optimizer,
                        progress=progress,
                        invariants=invariants,
                        controls=controls,
                        swanlab_run_id=active_run_id,
                    )
                    last_saved_tokens = progress.completed_assistant_tokens
                if args.smoke_optimizer_steps is not None and progress.optimizer_step >= args.smoke_optimizer_steps:
                    stop_reason = "smoke_optimizer_step_limit_reached"
                    break
            del loader
            if stop_before_next_group or stop_reason == "smoke_optimizer_step_limit_reached":
                break
        if progress.optimizer_step == 0 and stop_before_next_group:
            raise ValueError("stop_assistant_tokens 小于首个 optimizer step 的监督量")
        if last_saved_tokens != progress.completed_assistant_tokens:
            save_sft_checkpoint(
                paths["resume"],
                model=model,
                optimizer=optimizer,
                progress=progress,
                invariants=invariants,
                controls=controls,
                swanlab_run_id=active_run_id,
            )
        if args.smoke_optimizer_steps is not None:
            loaded = load_sft_checkpoint(
                paths["resume"],
                model=model,
                optimizer=optimizer,
                expected_invariants=invariants,
                controls=controls,
            )
            if loaded.progress != progress:
                raise RuntimeError("Query-SFT smoke checkpoint 严格恢复后的进度不一致")
        else:
            if progress.completed_assistant_tokens != args.stop_assistant_tokens:
                raise RuntimeError("Query-SFT 未精确到达共同 assistant-token 终点")
            prefix = "query-sft-pilot" if args.pilot_pipeline else "query-sft"
            weight_path = paths["weights"] / f"{prefix}-{progress.completed_assistant_tokens}.pth"
            if not weight_path.exists():
                export_sft_weights(weight_path, model)
        metric_logger.log(
            {
                "type": "run_complete",
                "timestamp": base_entry.pretrain_entry._utc_now(),
                "mode": (
                    "smoke"
                    if args.smoke_optimizer_steps is not None
                    else "pilot_pipeline"
                    if args.pilot_pipeline
                    else "train"
                ),
                "optimizer_step": progress.optimizer_step,
                "completed_assistant_tokens": progress.completed_assistant_tokens,
                "completed_sequences": progress.completed_sequences,
                "epoch": progress.epoch,
                "next_sequence_position": progress.next_sequence_position,
                "stop_reason": stop_reason,
                "weights_exported": args.smoke_optimizer_steps is None,
            }
        )
        return progress
    finally:
        train_dataset.close()
        if tracker is not None:
            try:
                tracker.finish()
            except Exception as error:
                warnings.warn(f"SwanLab finish 失败: {error}", RuntimeWarning, stacklevel=2)


def main() -> None:
    args = build_parser().parse_args()
    progress = run_training(args)
    mode = (
        "smoke"
        if args.smoke_optimizer_steps is not None
        else "pilot_pipeline"
        if args.pilot_pipeline
        else "train"
    )
    Logger(
        f"Query-SFT {mode} 结束: assistant_tokens={progress.completed_assistant_tokens:,}, "
        f"optimizer_steps={progress.optimizer_step:,}"
    )


if __name__ == "__main__":
    main()
