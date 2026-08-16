"""通用预训练训练循环使用的可测试基础能力。"""

from __future__ import annotations

import json
import math
import warnings
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, TypeVar

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .trainer_utils import (
    atomic_torch_save,
    capture_rng_state,
    restore_rng_state,
    unwrap_model,
)


T = TypeVar("T")


class PretrainCheckpointError(RuntimeError):
    """表示阶段 B 恢复点缺失、损坏或与当前训练配置冲突。"""


class JsonlMetricLogger:
    """先持久化本地 JSONL，再尽力上传同一条 SwanLab 指标。"""

    def __init__(self, path: str | Path, tracker: object | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tracker = tracker

    def log(self, record: Mapping[str, Any]) -> None:
        serialized = json.dumps(
            dict(record),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        with self.path.open("a", encoding="utf-8", newline="\n") as file:
            file.write(serialized + "\n")
            file.flush()
        if self.tracker is None:
            return
        try:
            step = record.get("optimizer_step")
            numeric_record = {
                key: value
                for key, value in record.items()
                if not isinstance(value, bool) and isinstance(value, (int, float))
            }
            self.tracker.log(numeric_record, step=step)
        except Exception as error:
            warnings.warn(
                f"SwanLab 指标上传失败，本地 JSONL 已保存: {error}",
                RuntimeWarning,
                stacklevel=2,
            )


@dataclass(frozen=True)
class PretrainProgress:
    """保存 optimizer 边界处可精确恢复的训练进度。"""

    epoch: int = 0
    next_sequence_position: int = 0
    completed_sequences: int = 0
    completed_tokens: int = 0
    optimizer_step: int = 0
    micro_step: int = 0

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values.values()):
            raise ValueError("训练进度字段必须是非负整数")
        if self.micro_step != 0:
            raise ValueError("正式恢复点只能保存在梯度累积边界")


@dataclass(frozen=True)
class LoadedPretrainCheckpoint:
    """返回恢复后的进度和外部实验标识。"""

    progress: PretrainProgress
    swanlab_run_id: str | None
    saved_controls: dict[str, Any]


def advance_pretrain_position(
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


def save_pretrain_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    progress: PretrainProgress,
    invariants: Mapping[str, Any],
    controls: Mapping[str, Any],
    swanlab_run_id: str | None = None,
) -> None:
    """原子保存正式预训练恢复点，模型参数保持原始训练精度。"""
    raw_model = unwrap_model(model)
    model_state = {
        name: value.detach().cpu() for name, value in raw_model.state_dict().items()
    }
    payload = {
        "schema_version": "1.0",
        "model": model_state,
        "optimizer": optimizer.state_dict(),
        "progress": asdict(progress),
        "invariants": dict(invariants),
        "controls": dict(controls),
        "rng_state": capture_rng_state(),
        "swanlab_run_id": swanlab_run_id,
    }
    atomic_torch_save(payload, path)


def export_pretrain_weights(path: str | Path, model: nn.Module) -> None:
    """原子导出不含 optimizer 的 BF16 模型权重。"""
    exported_state = {}
    for name, value in unwrap_model(model).state_dict().items():
        value = value.detach().cpu()
        exported_state[name] = value.to(torch.bfloat16) if value.is_floating_point() else value
    atomic_torch_save(exported_state, path)


def load_pretrain_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: optim.Optimizer,
    expected_invariants: Mapping[str, Any],
    controls: Mapping[str, Any],
) -> LoadedPretrainCheckpoint:
    """严格校验配置后恢复模型、optimizer、进度和随机状态。"""
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise PretrainCheckpointError(f"恢复点不存在: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise PretrainCheckpointError("无法读取预训练恢复点") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise PretrainCheckpointError("不支持的预训练恢复点格式")
    if payload.get("invariants") != dict(expected_invariants):
        raise PretrainCheckpointError("恢复点训练不变量与当前配置不一致")

    saved_controls = payload.get("controls")
    if not isinstance(saved_controls, dict):
        raise PretrainCheckpointError("恢复点缺少可变控制参数")
    saved_stop_tokens = saved_controls.get("stop_tokens")
    current_stop_tokens = controls.get("stop_tokens")
    if (
        isinstance(saved_stop_tokens, bool)
        or not isinstance(saved_stop_tokens, int)
        or isinstance(current_stop_tokens, bool)
        or not isinstance(current_stop_tokens, int)
        or current_stop_tokens < saved_stop_tokens
    ):
        raise PretrainCheckpointError("恢复时 stop_tokens 只能保持或提高")

    try:
        progress = PretrainProgress(**payload["progress"])
        unwrap_model(model).load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng_state(payload["rng_state"])
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        raise PretrainCheckpointError("预训练恢复点内容不完整或无法恢复") from error
    swanlab_run_id = payload.get("swanlab_run_id")
    if swanlab_run_id is not None and not isinstance(swanlab_run_id, str):
        raise PretrainCheckpointError("恢复点 SwanLab run ID 无效")
    return LoadedPretrainCheckpoint(
        progress=progress,
        swanlab_run_id=swanlab_run_id,
        saved_controls=saved_controls,
    )


@dataclass(frozen=True)
class TokenEventActions:
    """描述一个 optimizer 边界需要执行的验证和保存动作。"""

    quick_validation: bool = False
    full_validation: bool = False
    save_resume: bool = False
    export_weights: bool = False


@dataclass(frozen=True)
class OptimizerStepMetrics:
    """汇总一次 optimizer update 实际消费的数据和 loss。"""

    sequence_count: int
    input_tokens: int
    loss: float
    logits_loss: float
    aux_loss: float
    grad_norm: float


def group_micro_batches(
    batches: Iterable[T],
    accumulation_steps: int,
) -> Iterator[tuple[T, ...]]:
    """把 micro batch 组成 optimizer group，并保留 epoch 尾部不足组。"""
    if accumulation_steps <= 0:
        raise ValueError("accumulation_steps 必须为正数")
    group: list[T] = []
    for batch in batches:
        group.append(batch)
        if len(group) == accumulation_steps:
            yield tuple(group)
            group.clear()
    if group:
        yield tuple(group)


def evaluate_pretrain_by_source(
    model: nn.Module,
    catalog: object,
    *,
    split: str,
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
    autocast_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    """固定顺序遍历验证集，并按预测 token 数汇总来源与总体指标。"""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size 必须为正，num_workers 不能为负")
    source_names = getattr(catalog, "source_names", None)
    if not isinstance(source_names, tuple) or not source_names:
        raise ValueError("catalog 必须提供非空 source_names")

    was_training = model.training
    model.eval()
    source_reports: dict[str, dict[str, float | int]] = {}
    overall_negative_log_likelihood = 0.0
    overall_prediction_tokens = 0
    try:
        with torch.inference_mode():
            for source in source_names:
                dataset = catalog.create_dataset(split, sources=(source,))
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
                source_prediction_tokens = 0
                for input_ids, labels in loader:
                    input_ids = input_ids.to(
                        device,
                        non_blocking=device.type == "cuda",
                    )
                    labels = labels.to(device, non_blocking=device.type == "cuda")
                    autocast_context = (
                        torch.autocast(device_type="cuda", dtype=autocast_dtype)
                        if device.type == "cuda" and autocast_dtype is not None
                        else nullcontext()
                    )
                    with autocast_context:
                        output = model(input_ids, labels=labels)
                    prediction_tokens = int((labels[:, 1:] != -100).sum().item())
                    loss = float(output.loss.detach())
                    if not math.isfinite(loss):
                        raise FloatingPointError("验证 loss 出现 NaN 或 Inf")
                    source_negative_log_likelihood += loss * prediction_tokens
                    source_prediction_tokens += prediction_tokens
                if source_prediction_tokens == 0:
                    raise ValueError(f"验证来源 {source} 没有可预测 token")
                source_loss = (
                    source_negative_log_likelihood / source_prediction_tokens
                )
                source_reports[source] = {
                    "loss": source_loss,
                    "perplexity": math.exp(min(source_loss, 80.0)),
                    "prediction_tokens": source_prediction_tokens,
                }
                overall_negative_log_likelihood += source_negative_log_likelihood
                overall_prediction_tokens += source_prediction_tokens
    finally:
        if was_training:
            model.train()

    overall_loss = overall_negative_log_likelihood / overall_prediction_tokens
    return {
        "sources": source_reports,
        "overall": {
            "loss": overall_loss,
            "perplexity": math.exp(min(overall_loss, 80.0)),
            "prediction_tokens": overall_prediction_tokens,
        },
    }


def run_pretrain_optimizer_step(
    model: nn.Module,
    optimizer: optim.Optimizer,
    micro_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
    learning_rate: float,
    grad_clip: float,
    autocast_dtype: torch.dtype | None = None,
) -> OptimizerStepMetrics:
    """执行一个完整或尾部部分 accumulation group。"""
    if not micro_batches:
        raise ValueError("optimizer step 至少需要一个 micro batch")
    if learning_rate < 0 or grad_clip <= 0:
        raise ValueError("learning_rate 不能为负，grad_clip 必须为正")

    sequence_length: int | None = None
    sequence_count = 0
    for input_ids, labels in micro_batches:
        if input_ids.ndim != 2 or labels.shape != input_ids.shape:
            raise ValueError("input_ids/labels 必须是形状相同的二维张量")
        if sequence_length is None:
            sequence_length = input_ids.shape[1]
        elif input_ids.shape[1] != sequence_length:
            raise ValueError("同一 optimizer step 的 sequence 长度必须一致")
        sequence_count += input_ids.shape[0]
    if sequence_count <= 0 or sequence_length is None:
        raise ValueError("optimizer step 不能包含空 batch")

    for parameter_group in optimizer.param_groups:
        parameter_group["lr"] = learning_rate
    optimizer.zero_grad(set_to_none=True)
    weighted_logits_loss = 0.0
    weighted_aux_loss = 0.0
    for input_ids, labels in micro_batches:
        batch_size = input_ids.shape[0]
        input_ids = input_ids.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if device.type == "cuda" and autocast_dtype is not None
            else nullcontext()
        )
        with autocast_context:
            output = model(input_ids, labels=labels)
            logits_loss = output.loss
            aux_loss = getattr(output, "aux_loss", None)
            if aux_loss is None:
                aux_loss = logits_loss.new_zeros(())
            combined_loss = logits_loss + aux_loss
            scaled_loss = combined_loss * (batch_size / sequence_count)
        if not torch.isfinite(combined_loss.detach()):
            optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("训练 loss 出现 NaN 或 Inf")
        scaled_loss.backward()
        weighted_logits_loss += float(logits_loss.detach()) * batch_size
        weighted_aux_loss += float(aux_loss.detach()) * batch_size

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    logits_loss_value = weighted_logits_loss / sequence_count
    aux_loss_value = weighted_aux_loss / sequence_count
    return OptimizerStepMetrics(
        sequence_count=sequence_count,
        input_tokens=sequence_count * sequence_length,
        loss=logits_loss_value + aux_loss_value,
        logits_loss=logits_loss_value,
        aux_loss=aux_loss_value,
        grad_norm=float(grad_norm),
    )


def build_pretrain_optimizer(
    model: nn.Module,
    *,
    peak_lr: float,
    device_type: str,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> optim.AdamW:
    """按参数维度创建 AdamW，并在 CUDA 上优先尝试 fused 实现。"""
    decay_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.ndim >= 2
    ]
    no_decay_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.ndim < 2
    ]
    if not decay_parameters and not no_decay_parameters:
        raise ValueError("模型没有可训练参数")

    parameter_groups = [
        {"params": decay_parameters, "weight_decay": weight_decay},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ]
    optimizer_kwargs = {
        "lr": peak_lr,
        "betas": betas,
        "eps": eps,
    }
    if device_type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        return optim.AdamW(parameter_groups, **optimizer_kwargs)
    except (TypeError, RuntimeError):
        if "fused" not in optimizer_kwargs:
            raise
        optimizer_kwargs.pop("fused")
        return optim.AdamW(parameter_groups, **optimizer_kwargs)


def crossed_token_events(
    previous_tokens: int,
    completed_tokens: int,
    *,
    quick_interval_tokens: int,
    full_interval_tokens: int,
    fixed_quick_tokens: tuple[int, ...] = (),
    fixed_full_tokens: tuple[int, ...] = (),
) -> TokenEventActions:
    """返回本次 optimizer update 跨过的 token 事件。"""
    if previous_tokens < 0 or completed_tokens < previous_tokens:
        raise ValueError("累计 token 区间无效")
    if quick_interval_tokens < 0 or full_interval_tokens < 0:
        raise ValueError("事件间隔不能为负数")
    fixed_tokens = fixed_quick_tokens + fixed_full_tokens
    if any(token <= 0 for token in fixed_tokens):
        raise ValueError("固定事件 token 必须为正数")

    def crossed_interval(interval: int) -> bool:
        return interval > 0 and (
            completed_tokens // interval > previous_tokens // interval
        )

    def crossed_fixed(thresholds: tuple[int, ...]) -> bool:
        return any(
            previous_tokens < threshold <= completed_tokens
            for threshold in thresholds
        )

    full_validation = crossed_interval(full_interval_tokens) or crossed_fixed(
        fixed_full_tokens
    )
    quick_validation = not full_validation and (
        crossed_interval(quick_interval_tokens) or crossed_fixed(fixed_quick_tokens)
    )
    should_save = quick_validation or full_validation
    return TokenEventActions(
        quick_validation=quick_validation,
        full_validation=full_validation,
        save_resume=should_save,
        export_weights=full_validation,
    )


def token_learning_rate(
    completed_tokens: int,
    peak_lr: float,
    warmup_tokens: int,
    schedule_tokens: int,
    floor_ratio: float = 0.1,
) -> float:
    """根据累计输入 token 数计算当前 optimizer update 的学习率。"""
    if completed_tokens < 0:
        raise ValueError("completed_tokens 不能为负数")
    if peak_lr <= 0:
        raise ValueError("peak_lr 必须大于 0")
    if warmup_tokens <= 0 or schedule_tokens <= warmup_tokens:
        raise ValueError("schedule_tokens 必须大于 warmup_tokens，且两者均为正数")
    if not 0 < floor_ratio <= 1:
        raise ValueError("floor_ratio 必须位于 (0, 1] 区间")

    if completed_tokens <= warmup_tokens:
        return peak_lr * completed_tokens / warmup_tokens
    if completed_tokens >= schedule_tokens:
        return peak_lr * floor_ratio

    decay_progress = (completed_tokens - warmup_tokens) / (
        schedule_tokens - warmup_tokens
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return peak_lr * (floor_ratio + (1.0 - floor_ratio) * cosine)
