"""RAG-SFT v2 独立训练入口：五条 lineage 共用配置并保存 E2–E4 权重。"""

from __future__ import annotations

import argparse
import hashlib
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
    from ..dataset.rag_sft_v2_dataset import (
        LABEL_MASK_VERSION,
        RagSftV2Dataset,
        validate_manifest_sha256,
    )
except ImportError:
    from dataset.pretrain_dataset import DeterministicPretrainSampler
    from dataset.rag_sft_v2_dataset import (
        LABEL_MASK_VERSION,
        RagSftV2Dataset,
        validate_manifest_sha256,
    )

from . import train_full_sft as base_entry
from .sft_runtime import (
    JsonlMetricLogger,
    SftProgress,
    advance_sft_position,
    assistant_token_learning_rate,
    build_sft_optimizer,
    export_sft_weights,
    group_micro_batches,
    load_sft_checkpoint,
    run_sft_optimizer_step,
    save_sft_checkpoint,
)


PIPELINE = "rag_sft_v2_training"
EPOCHS = 4
EXPORTED_EPOCHS = (2, 3, 4)
MAX_SMOKE_OPTIMIZER_STEPS = 30
MODEL_ARCHITECTURE = base_entry.MODEL_ARCHITECTURE
MiniMindForCausalLM = base_entry.MiniMindForCausalLM
OPTIMIZER = {
    "name": "AdamW",
    "betas": [0.9, 0.95],
    "weight_decay": 0.1,
    "eps": 1e-8,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MiniMind RAG-SFT v2 训练")
    for name in ("run_name", "run_dir", "checkpoint_dir", "manifest", "tokenizer_path", "parent_weights", "parent_sha256"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--candidate_path")
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
    parser.add_argument("--allow_stage_experiment", action="store_true")
    parser.add_argument(
        "--swanlab", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--swanlab_project", default="minimind+rag")
    parser.add_argument("--swanlab_workspace", default="Bigwatermelon")
    parser.add_argument("--resume", action="store_true")
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_identities() -> dict[str, dict[str, Any]]:
    project_root = Path(__file__).resolve().parents[2]
    paths = {
        "training": Path(__file__).resolve(),
        "dataset": project_root / "minimind" / "dataset" / "rag_sft_v2_dataset.py",
        "sft_runtime": project_root / "minimind" / "trainer" / "sft_runtime.py",
        "answering_protocol": project_root / "rag" / "answering" / "protocol.py",
        "answering_evidence": project_root / "rag" / "answering" / "evidence.py",
    }
    identities = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"RAG-SFT v2 运行时源码不存在: {path}")
        identities[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    return identities


def _validate_args(args: argparse.Namespace) -> None:
    if args.peak_lr <= 0 or args.grad_clip <= 0 or not 0 < args.warmup_ratio < 1 or not 0 < args.floor_ratio <= 1:
        raise ValueError("学习率、warmup_ratio、floor_ratio 和 grad_clip 参数无效")
    if args.micro_batch_size <= 0 or args.accumulation_steps <= 0 or args.num_workers < 0 or args.seed < 0:
        raise ValueError("batch、worker 和 seed 参数无效")
    if args.smoke_optimizer_steps is not None and not 1 <= args.smoke_optimizer_steps <= MAX_SMOKE_OPTIMIZER_STEPS:
        raise ValueError("smoke_optimizer_steps 必须位于 1 至 30")
    if args.resume and args.smoke_optimizer_steps is not None:
        raise ValueError("smoke 不支持恢复")
    if len(args.parent_sha256) != 64:
        raise ValueError("parent_sha256 必须是 SHA-256")
    try:
        int(args.parent_sha256, 16)
    except ValueError as error:
        raise ValueError("parent_sha256 必须是 SHA-256") from error
    if int(os.environ.get("RANK", -1)) != -1:
        raise ValueError("RAG-SFT v2 只支持单卡，不支持 DDP")


def _load_manifest(path: Path, *, allow_stage_experiment: bool) -> tuple[dict[str, Any], str]:
    digest = validate_manifest_sha256(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("无法读取 RAG-SFT v2 manifest") from error
    if not isinstance(payload, dict) or payload.get("pipeline") not in {
        "rag_sft_v2_oracle_clean_materialization",
        "rag_sft_v2_training_release",
    }:
        raise ValueError("不是 RAG-SFT v2 manifest")
    readiness = payload.get("readiness")
    if not isinstance(readiness, dict) or (
        payload.get("pipeline") == "rag_sft_v2_oracle_clean_materialization"
        and readiness.get("oracle_clean_audited") is not True
    ):
        raise ValueError("RAG-SFT v2 manifest 尚未完成审计")
    if readiness.get("training_ready") is not True and not allow_stage_experiment:
        raise ValueError("RAG-SFT v2 manifest training_ready=false，不能用于正式训练")
    return payload, digest


def _group_tokens(groups: tuple[tuple[torch.Tensor, torch.Tensor], ...]) -> int:
    return sum(int((labels[:, 1:] != -100).sum().item()) for _, labels in groups)


def _export_epoch_weights(path: Path, model: torch.nn.Module) -> str:
    hash_path = path.with_suffix(".sha256")
    if path.exists() or hash_path.exists():
        raise FileExistsError(f"epoch 权重或 SHA-256 已存在: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    export_sft_weights(path, model)
    digest = _sha256_file(path)
    temporary = hash_path.with_suffix(hash_path.suffix + ".tmp")
    temporary.write_text(
        f"{digest}  {path.name}\n", encoding="utf-8", newline="\n"
    )
    temporary.replace(hash_path)
    return digest


def _verify_epoch_weight(path: Path) -> str:
    hash_path = path.with_suffix(".sha256")
    if not path.is_file() or not hash_path.is_file():
        raise FileNotFoundError(f"epoch 权重或 SHA-256 不完整: {path}")
    digest = _sha256_file(path)
    if hash_path.read_text(encoding="utf-8").splitlines() != [
        f"{digest}  {path.name}"
    ]:
        raise ValueError(f"epoch 权重 SHA-256 校验失败: {path}")
    return digest


def _reconcile_epoch_weights(
    weights_dir: Path, progress: SftProgress, model: torch.nn.Module
) -> None:
    """只允许导出 E2–E4，并在恢复时核对既有边界。"""

    forbidden = weights_dir / "rag_epoch_1.pth"
    if forbidden.exists() or forbidden.with_suffix(".sha256").exists():
        raise ValueError("RAG-SFT v2 正式训练禁止导出 rag_epoch_1.pth")
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
    manifest, manifest_sha = _load_manifest(
        manifest_path, allow_stage_experiment=args.allow_stage_experiment
    )
    if args.smoke_optimizer_steps is None and (
        manifest.get("pipeline") != "rag_sft_v2_training_release"
        or manifest.get("readiness", {}).get("training_ready") is not True
    ):
        raise ValueError("正式训练要求 v2 training release 且 training_ready=true")
    run_dir = Path(args.run_dir).resolve()
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    latest = checkpoint_dir / "resume" / "latest.pt"
    weights_dir = checkpoint_dir / "weights"
    run_manifest_path = run_dir / "run-manifest.json"
    metrics_path = run_dir / "metrics.jsonl"
    if not args.resume and (run_dir.exists() or checkpoint_dir.exists()):
        raise FileExistsError("RAG-SFT v2 run 或 checkpoint 目录已存在")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 CUDA 但当前无可用 GPU")
    if args.dtype == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 CUDA 不支持 BF16")
    parent_path = Path(args.parent_weights).resolve()
    if not parent_path.is_file() or _sha256_file(parent_path) != args.parent_sha256.lower():
        raise ValueError("父权重文件 SHA-256 校验失败")
    tokenizer = base_entry._load_tokenizer(args.tokenizer_path)
    dataset = RagSftV2Dataset(
        manifest_path,
        tokenizer,
        candidate_path=args.candidate_path,
        tokenizer_path=args.tokenizer_path,
    )
    total_tokens = sum(int((dataset[index][1][1:] != -100).sum().item()) for index in range(len(dataset)))
    schedule_tokens = total_tokens * EPOCHS
    warmup_tokens = max(1, int(schedule_tokens * args.warmup_ratio))
    source_identities = _source_identities()
    controls = {"stop_assistant_tokens": schedule_tokens, "smoke_optimizer_steps": args.smoke_optimizer_steps}
    invariants = {"pipeline": PIPELINE, "manifest_sha256": manifest_sha, "candidate_sha256": _sha256_file(dataset.candidate_path), "tokenizer": dataset.tokenizer_identity, "parent_sha256": args.parent_sha256.lower(), "source_identities": source_identities, "sequence_length": 768, "label_mask_version": LABEL_MASK_VERSION, "epochs": EPOCHS, "exported_epochs": list(EXPORTED_EPOCHS), "sampler": {"name": "DeterministicPretrainSampler", "mode": "without_replacement_full_candidate", "records_per_epoch": len(dataset)}, "seed": args.seed, "optimizer": OPTIMIZER, "peak_lr": args.peak_lr, "warmup_assistant_tokens": warmup_tokens, "schedule_assistant_tokens": schedule_tokens, "floor_ratio": args.floor_ratio, "micro_batch_size": args.micro_batch_size, "accumulation_steps": args.accumulation_steps, "grad_clip": args.grad_clip, "dtype": args.dtype, "num_workers": args.num_workers}
    tracker = None
    try:
        base_entry.setup_seed(args.seed)
        model = MiniMindForCausalLM(base_entry._model_config()).to(device)
        base_entry.get_model_params(model, base_entry._model_config())
        if not args.resume:
            base_entry._load_parent_weights(parent_path, args.parent_sha256, model)
        optimizer = build_sft_optimizer(model, peak_lr=args.peak_lr, device_type=device.type)
        progress = SftProgress()
        resume_run_id = None
        if args.resume:
            saved_manifest, _ = base_entry._load_verified_json(run_manifest_path, "RAG-SFT v2 run manifest")
            if saved_manifest.get("run_name") != args.run_name:
                raise ValueError("RAG-SFT v2 run manifest 的 run_name 与当前命令不一致")
            if saved_manifest.get("training_invariants") != invariants:
                raise ValueError("恢复时训练不变量不一致")
            loaded = load_sft_checkpoint(
                latest,
                model=model,
                optimizer=optimizer,
                expected_invariants=invariants,
                controls=controls,
            )
            progress = loaded.progress
            resume_run_id = loaded.swanlab_run_id
            if args.smoke_optimizer_steps is None:
                _reconcile_epoch_weights(weights_dir, progress, model)
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            base_entry._write_immutable_run_manifest(
                run_manifest_path,
                {"schema_version": "1.0", "pipeline": PIPELINE, "run_name": args.run_name, "data": {"manifest_sha256": manifest_sha}, "parent_model": {"path": str(Path(args.parent_weights).resolve()), "sha256": args.parent_sha256}, "environment": {"OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"), "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM")}, "training_invariants": invariants},
            )
        tracker, swanlab_run = base_entry.pretrain_entry._init_swanlab(
            args,
            run_dir,
            {**invariants, **controls},
            resume_run_id,
        )
        active_run_id = base_entry.pretrain_entry._current_swanlab_run_id(
            swanlab_run, resume_run_id
        )
        logger = JsonlMetricLogger(metrics_path, tracker=tracker)
        while progress.completed_assistant_tokens < schedule_tokens:
            sampler = DeterministicPretrainSampler(dataset, seed=args.seed, epoch=progress.epoch, start_position=progress.next_sequence_position)
            loader = DataLoader(dataset, batch_size=args.micro_batch_size, sampler=sampler, shuffle=False, drop_last=False, num_workers=args.num_workers, pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)
            for groups in group_micro_batches(loader, args.accumulation_steps):
                if args.smoke_optimizer_steps is not None and progress.optimizer_step >= args.smoke_optimizer_steps:
                    break
                previous = progress.completed_assistant_tokens
                step_tokens = _group_tokens(groups)
                lr = assistant_token_learning_rate(previous + step_tokens, args.peak_lr, warmup_tokens, schedule_tokens, args.floor_ratio)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                result = run_sft_optimizer_step(model, optimizer, groups, device=device, learning_rate=lr, grad_clip=args.grad_clip, autocast_dtype=torch.bfloat16 if args.dtype == "bfloat16" else None)
                elapsed = time.perf_counter() - started
                epoch, position = advance_sft_position(progress.epoch, progress.next_sequence_position, result.sequence_count, len(dataset))
                progress = SftProgress(epoch=epoch, next_sequence_position=position, completed_sequences=progress.completed_sequences + result.sequence_count, completed_assistant_tokens=previous + result.assistant_tokens, optimizer_step=progress.optimizer_step + 1)
                grad_clip_scale = min(1.0, args.grad_clip / (result.grad_norm + 1e-6))
                logger.log({"type": "train", "optimizer_step": progress.optimizer_step, "epoch": progress.epoch, "completed_assistant_tokens": progress.completed_assistant_tokens, "assistant_tokens": result.assistant_tokens, "loss": result.loss, "learning_rate": lr, "grad_norm": result.grad_norm, "grad_clip_scale": grad_clip_scale, "grad_was_clipped": result.grad_norm > args.grad_clip, "elapsed_seconds": elapsed, "assistant_tokens_per_second": result.assistant_tokens / elapsed, "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None})
                if position == 0 or (args.smoke_optimizer_steps is not None and progress.optimizer_step >= args.smoke_optimizer_steps):
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    save_sft_checkpoint(
                        latest,
                        model=model,
                        optimizer=optimizer,
                        progress=progress,
                        invariants=invariants,
                        controls=controls,
                        swanlab_run_id=active_run_id,
                    )
                    if position == 0 and args.smoke_optimizer_steps is None:
                        _reconcile_epoch_weights(weights_dir, progress, model)
                if args.smoke_optimizer_steps is not None and progress.optimizer_step >= args.smoke_optimizer_steps:
                    break
            del loader
            if args.smoke_optimizer_steps is not None and progress.optimizer_step >= args.smoke_optimizer_steps:
                break
        if args.smoke_optimizer_steps is None and progress.completed_assistant_tokens != schedule_tokens:
            raise RuntimeError("RAG-SFT v2 未完成共同 token 终点")
        if args.smoke_optimizer_steps is not None:
            load_sft_checkpoint(latest, model=model, optimizer=optimizer, expected_invariants=invariants, controls=controls)
        logger.log({"type": "run_complete", "epoch": progress.epoch, "next_sequence_position": progress.next_sequence_position, "completed_sequences": progress.completed_sequences, "completed_assistant_tokens": progress.completed_assistant_tokens, "optimizer_step": progress.optimizer_step, "weights_exported": args.smoke_optimizer_steps is None, "weights_exported_epochs": list(EXPORTED_EPOCHS) if args.smoke_optimizer_steps is None else []})
        return progress
    finally:
        dataset.close()
        if tracker is not None:
            try:
                tracker.finish()
            except Exception as error:
                warnings.warn(
                    f"SwanLab finish 失败: {error}", RuntimeWarning, stacklevel=2
                )


def main() -> None:
    progress = run_training(build_parser().parse_args())
    print(f"RAG_SFT_V2_TRAINING_OK epoch={progress.epoch} assistant_tokens={progress.completed_assistant_tokens}")


if __name__ == "__main__":
    main()
