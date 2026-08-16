"""法律 SFT 训练循环使用的可测试运行时能力。"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from .pretrain_runtime import (
    JsonlMetricLogger,
    TokenEventActions,
    build_pretrain_optimizer,
    crossed_token_events,
    group_micro_batches,
    token_learning_rate,
)
from .trainer_utils import (
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
    unwrap_model,
)


RUNTIME_VERSION = "legal_sft_runtime_v1"
DATASET_KINDS = ("pair_qa", "triplet_qa")
VALIDATION_SPLITS = ("validation/quick", "validation/full")
DatasetFactory = Callable[[str, str], Dataset]


class SftCheckpointError(RuntimeError):
    """表示 SFT 恢复点缺失、损坏或与当前训练身份冲突。"""


@dataclass(frozen=True)
class SftProgress:
    """保存 optimizer 边界处可精确恢复的 SFT 进度。"""

    epoch: int = 0
    next_sequence_position: int = 0
    completed_sequences: int = 0
    completed_assistant_tokens: int = 0
    optimizer_step: int = 0
    micro_step: int = 0

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values.values()
        ):
            raise ValueError("SFT 进度字段必须是非负整数")
        if self.micro_step != 0:
            raise ValueError("SFT 恢复点只能保存在梯度累积边界")


@dataclass(frozen=True)
class LoadedSftCheckpoint:
    """返回恢复后的 SFT 进度和外部实验标识。"""

    progress: SftProgress
    swanlab_run_id: str | None
    saved_controls: dict[str, Any]


@dataclass(frozen=True)
class SftOptimizerStepMetrics:
    """汇总一次 SFT optimizer update 的实际监督量与 loss。"""

    sequence_count: int
    input_tokens: int
    assistant_tokens: int
    loss: float
    logits_loss: float
    aux_loss: float
    grad_norm: float


def advance_sft_position(
    epoch: int,
    next_sequence_position: int,
    consumed_sequences: int,
    epoch_sequence_count: int,
) -> tuple[int, int]:
    """推进实际消费位置，并在完整 epoch 后规范化到下一轮起点。"""

    values = (epoch, next_sequence_position, consumed_sequences, epoch_sequence_count)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("epoch 和 sequence 位置必须是整数")
    if epoch < 0 or next_sequence_position < 0 or consumed_sequences <= 0:
        raise ValueError("epoch/position 不能为负，consumed_sequences 必须为正")
    if epoch_sequence_count <= 0:
        raise ValueError("epoch_sequence_count 必须为正数")
    updated_position = next_sequence_position + consumed_sequences
    if updated_position > epoch_sequence_count:
        raise ValueError("消费 sequence 后的位置超过当前 epoch")
    if updated_position == epoch_sequence_count:
        return epoch + 1, 0
    return epoch, updated_position


def assistant_token_learning_rate(
    completed_assistant_tokens: int,
    peak_lr: float,
    warmup_assistant_tokens: int,
    schedule_assistant_tokens: int,
    floor_ratio: float = 0.1,
) -> float:
    """根据累计有效 assistant tokens 计算当前 optimizer update 的学习率。"""

    return token_learning_rate(
        completed_assistant_tokens,
        peak_lr,
        warmup_assistant_tokens,
        schedule_assistant_tokens,
        floor_ratio,
    )


def crossed_assistant_token_events(
    previous_assistant_tokens: int,
    completed_assistant_tokens: int,
    *,
    quick_interval_tokens: int,
    full_interval_tokens: int,
    fixed_quick_tokens: tuple[int, ...] = (),
    fixed_full_tokens: tuple[int, ...] = (),
) -> TokenEventActions:
    """返回本次 update 跨过的 assistant-token 验证与保存事件。"""

    return crossed_token_events(
        previous_assistant_tokens,
        completed_assistant_tokens,
        quick_interval_tokens=quick_interval_tokens,
        full_interval_tokens=full_interval_tokens,
        fixed_quick_tokens=fixed_quick_tokens,
        fixed_full_tokens=fixed_full_tokens,
    )


def build_sft_optimizer(
    model: nn.Module,
    *,
    peak_lr: float,
    device_type: str,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> optim.AdamW:
    """沿用阶段 B/C 已验证的 AdamW 参数分组。"""

    return build_pretrain_optimizer(
        model,
        peak_lr=peak_lr,
        device_type=device_type,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )


def _batch_active_label_counts(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    if input_ids.ndim != 2 or labels.shape != input_ids.shape:
        raise ValueError("input_ids/labels 必须是形状相同的二维张量")
    if input_ids.shape[0] == 0 or input_ids.shape[1] < 2:
        raise ValueError("SFT batch 不能为空且 sequence length 至少为 2")
    active_counts = (labels[:, 1:] != -100).sum(dim=1)
    if torch.any(active_counts == 0):
        raise ValueError("SFT batch 包含零有效 assistant label 的样本")
    return active_counts


def run_sft_optimizer_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    micro_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    learning_rate: float,
    grad_clip: float,
    autocast_dtype: torch.dtype | None = None,
) -> SftOptimizerStepMetrics:
    """按整组有效 assistant tokens 归一化一次 optimizer update。"""

    if not micro_batches:
        raise ValueError("optimizer step 至少需要一个 micro batch")
    if learning_rate < 0 or grad_clip <= 0:
        raise ValueError("learning_rate 不能为负，grad_clip 必须为正")

    sequence_length: int | None = None
    sequence_count = 0
    batch_active_tokens: list[int] = []
    for input_ids, labels in micro_batches:
        active_counts = _batch_active_label_counts(input_ids, labels)
        if sequence_length is None:
            sequence_length = input_ids.shape[1]
        elif input_ids.shape[1] != sequence_length:
            raise ValueError("同一 optimizer step 的 sequence 长度必须一致")
        sequence_count += input_ids.shape[0]
        batch_active_tokens.append(int(active_counts.sum().item()))
    assistant_tokens = sum(batch_active_tokens)
    if sequence_count <= 0 or sequence_length is None or assistant_tokens <= 0:
        raise ValueError("optimizer step 不能包含空监督量")

    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    weighted_logits_loss = 0.0
    weighted_aux_loss = 0.0
    for (input_ids, labels), active_tokens in zip(
        micro_batches, batch_active_tokens
    ):
        input_ids = input_ids.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if device.type == "cuda" and autocast_dtype is not None
            else nullcontext()
        )
        with autocast_context:
            output = model(input_ids, labels=labels)
            logits_loss = getattr(output, "loss", None)
            if not isinstance(logits_loss, torch.Tensor) or logits_loss.numel() != 1:
                optimizer.zero_grad(set_to_none=True)
                raise ValueError("模型必须返回标量 loss")
            aux_loss = getattr(output, "aux_loss", None)
            if aux_loss is None:
                aux_loss = logits_loss.new_zeros(())
            if not isinstance(aux_loss, torch.Tensor) or aux_loss.numel() != 1:
                optimizer.zero_grad(set_to_none=True)
                raise ValueError("模型必须返回标量 aux_loss")
            combined_loss = logits_loss + aux_loss
            scaled_loss = combined_loss * (active_tokens / assistant_tokens)
        if not torch.isfinite(combined_loss.detach()):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("SFT 训练 loss 出现 NaN 或 Inf")
        scaled_loss.backward()
        weighted_logits_loss += float(logits_loss.detach()) * active_tokens
        weighted_aux_loss += float(aux_loss.detach()) * active_tokens

    try:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), grad_clip, error_if_nonfinite=True
        )
    except RuntimeError as error:
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("SFT 梯度出现 NaN 或 Inf") from error
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    logits_loss_value = weighted_logits_loss / assistant_tokens
    aux_loss_value = weighted_aux_loss / assistant_tokens
    return SftOptimizerStepMetrics(
        sequence_count=sequence_count,
        input_tokens=sequence_count * sequence_length,
        assistant_tokens=assistant_tokens,
        loss=logits_loss_value + aux_loss_value,
        logits_loss=logits_loss_value,
        aux_loss=aux_loss_value,
        grad_norm=float(grad_norm),
    )


def evaluate_sft_by_source(
    model: nn.Module,
    dataset_factory: DatasetFactory,
    *,
    split: str,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """固定顺序验证 Pair-QA/Triplet-QA，并按有效 labels 汇总 loss。"""

    if split not in VALIDATION_SPLITS:
        raise ValueError(f"SFT 验证 split 无效: {split}")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size 必须为正，num_workers 不能为负")

    was_training = model.training
    model.eval()
    source_reports: dict[str, dict[str, float | int]] = {}
    overall_negative_log_likelihood = 0.0
    overall_assistant_tokens = 0
    overall_sequences = 0
    try:
        with torch.inference_mode():
            for dataset_kind in DATASET_KINDS:
                dataset = dataset_factory(split, dataset_kind)
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=num_workers,
                    pin_memory=device.type == "cuda",
                    persistent_workers=num_workers > 0,
                )
                source_negative_log_likelihood = 0.0
                source_assistant_tokens = 0
                source_sequences = 0
                try:
                    for input_ids, labels in loader:
                        active_counts = _batch_active_label_counts(input_ids, labels)
                        active_tokens = int(active_counts.sum().item())
                        source_sequences += input_ids.shape[0]
                        input_ids = input_ids.to(
                            device, non_blocking=device.type == "cuda"
                        )
                        labels = labels.to(
                            device, non_blocking=device.type == "cuda"
                        )
                        autocast_context = (
                            torch.autocast(
                                device_type="cuda", dtype=autocast_dtype
                            )
                            if device.type == "cuda" and autocast_dtype is not None
                            else nullcontext()
                        )
                        with autocast_context:
                            output = model(input_ids, labels=labels)
                        output_loss = getattr(output, "loss", None)
                        if (
                            not isinstance(output_loss, torch.Tensor)
                            or output_loss.numel() != 1
                        ):
                            raise ValueError("模型必须返回标量 validation loss")
                        loss = float(output_loss.detach())
                        if not math.isfinite(loss):
                            raise FloatingPointError(
                                "SFT validation loss 出现 NaN 或 Inf"
                            )
                        source_negative_log_likelihood += loss * active_tokens
                        source_assistant_tokens += active_tokens
                finally:
                    close = getattr(dataset, "close", None)
                    if callable(close):
                        close()
                if source_assistant_tokens == 0:
                    raise ValueError(f"验证来源 {dataset_kind} 没有有效 assistant labels")
                source_loss = (
                    source_negative_log_likelihood / source_assistant_tokens
                )
                source_reports[dataset_kind] = {
                    "loss": source_loss,
                    "perplexity": math.exp(min(source_loss, 80.0)),
                    "assistant_tokens": source_assistant_tokens,
                    "sequences": source_sequences,
                }
                overall_negative_log_likelihood += source_negative_log_likelihood
                overall_assistant_tokens += source_assistant_tokens
                overall_sequences += source_sequences
    finally:
        if was_training:
            model.train()

    overall_loss = overall_negative_log_likelihood / overall_assistant_tokens
    return {
        "split": split,
        "sources": source_reports,
        "overall": {
            "loss": overall_loss,
            "perplexity": math.exp(min(overall_loss, 80.0)),
            "assistant_tokens": overall_assistant_tokens,
            "sequences": overall_sequences,
        },
    }


def save_sft_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    progress: SftProgress,
    invariants: Mapping[str, Any],
    controls: Mapping[str, Any],
    swanlab_run_id: str | None = None,
) -> None:
    """原子保存正式 SFT 恢复点，模型参数保持原始训练精度。"""

    _validate_sft_stop_controls(controls, progress, ValueError)
    raw_model = unwrap_model(model)
    model_state = {
        name: value.detach().cpu() for name, value in raw_model.state_dict().items()
    }
    payload = {
        "schema_version": "1.0",
        "runtime": RUNTIME_VERSION,
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "progress": asdict(progress),
        "invariants": dict(invariants),
        "controls": dict(controls),
        "rng_state": capture_rng_state(),
        "swanlab_run_id": swanlab_run_id,
    }
    atomic_torch_save(payload, path)


def load_sft_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    expected_invariants: Mapping[str, Any],
    controls: Mapping[str, Any],
) -> LoadedSftCheckpoint:
    """严格校验训练身份后恢复模型、optimizer、进度和随机状态。"""

    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise SftCheckpointError(f"SFT 恢复点不存在: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise SftCheckpointError("无法读取 SFT 恢复点") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "1.0"
        or payload.get("runtime") != RUNTIME_VERSION
    ):
        raise SftCheckpointError("不支持的 SFT 恢复点格式")
    if payload.get("invariants") != dict(expected_invariants):
        raise SftCheckpointError("SFT 恢复点训练不变量与当前配置不一致")

    saved_controls = payload.get("controls")
    if not isinstance(saved_controls, dict):
        raise SftCheckpointError("SFT 恢复点缺少可变控制参数")
    for name in ("stop_assistant_tokens", "stop_optimizer_steps"):
        saved_value = saved_controls.get(name)
        if saved_value is None:
            continue
        current_value = controls.get(name)
        if (
            isinstance(saved_value, bool)
            or not isinstance(saved_value, int)
            or saved_value <= 0
            or isinstance(current_value, bool)
            or not isinstance(current_value, int)
            or current_value < saved_value
        ):
            raise SftCheckpointError(f"恢复时 {name} 只能保持或提高")

    try:
        progress = SftProgress(**payload["progress"])
    except (KeyError, TypeError, ValueError) as error:
        raise SftCheckpointError("SFT 恢复点进度无效") from error
    _validate_sft_stop_controls(saved_controls, progress, SftCheckpointError)
    try:
        unwrap_model(model).load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng_state(payload["rng_state"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise SftCheckpointError("SFT 恢复点内容不完整或无法恢复") from error
    swanlab_run_id = payload.get("swanlab_run_id")
    if swanlab_run_id is not None and not isinstance(swanlab_run_id, str):
        raise SftCheckpointError("SFT 恢复点 SwanLab run ID 无效")
    return LoadedSftCheckpoint(
        progress=progress,
        swanlab_run_id=swanlab_run_id,
        saved_controls=saved_controls,
    )


def _validate_sft_stop_controls(
    controls: Mapping[str, Any],
    progress: SftProgress,
    error_type: type[Exception],
) -> None:
    """校验至少一种 SFT 停止预算覆盖当前进度。"""

    limits = {
        "stop_assistant_tokens": progress.completed_assistant_tokens,
        "stop_optimizer_steps": progress.optimizer_step,
    }
    present = False
    for name, completed in limits.items():
        value = controls.get(name)
        if value is None:
            continue
        present = True
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value < completed
        ):
            raise error_type(f"{name} 必须覆盖当前 SFT 训练进度")
    if not present:
        raise error_type("SFT checkpoint 缺少停止预算")


def export_sft_weights(path: str | Path, model: nn.Module) -> None:
    """原子导出不含 optimizer 的 BF16 SFT 模型权重。"""

    exported_state = {}
    for name, value in unwrap_model(model).state_dict().items():
        value = value.detach().cpu()
        exported_state[name] = (
            value.to(torch.bfloat16) if value.is_floating_point() else value
        )
    atomic_torch_save(exported_state, path)
