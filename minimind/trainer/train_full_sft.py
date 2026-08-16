"""从明确的 CPT 父权重启动法律全参数 SFT。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from ..dataset.pretrain_dataset import DeterministicPretrainSampler
    from ..dataset.sft_dataset import LABEL_MASK_VERSION, SftDataset
    from ..model.model_minimind import MiniMindConfig, MiniMindForCausalLM
except ImportError:
    from dataset.pretrain_dataset import DeterministicPretrainSampler
    from dataset.sft_dataset import LABEL_MASK_VERSION, SftDataset
    from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

from . import train_pretrain as pretrain_entry
from .pretrain_runtime import group_micro_batches
from .sft_runtime import (
    JsonlMetricLogger,
    SftProgress,
    advance_sft_position,
    assistant_token_learning_rate,
    build_sft_optimizer,
    crossed_assistant_token_events,
    evaluate_sft_by_source,
    export_sft_weights,
    load_sft_checkpoint,
    run_sft_optimizer_step,
    save_sft_checkpoint,
)
from .trainer_utils import Logger, get_model_params, setup_seed


EXPECTED_SEQUENCE_LENGTH = 768
DATA_TIER = "all-retained"
RUN_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
MODEL_ARCHITECTURE = {
    "vocab_size": 12_000,
    "hidden_size": 768,
    "num_hidden_layers": 16,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "intermediate_size": 2432,
    "use_moe": False,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_value(value: str) -> str:
    parsed = value.lower()
    if len(parsed) != 64 or any(character not in "0123456789abcdef" for character in parsed):
        raise argparse.ArgumentTypeError("必须是 64 位十六进制 SHA-256")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """创建法律全参数 SFT 的单卡 CLI。"""
    parser = argparse.ArgumentParser(description="MiniMind 法律全参数 SFT")
    parser.add_argument("--run_name", required=True, help="实验名称")
    parser.add_argument("--run_dir", required=True, help="本地指标和 run manifest 目录")
    parser.add_argument("--checkpoint_dir", required=True, help="恢复点和权重目录")
    parser.add_argument("--data_manifest", required=True, help="固定 768 的 SFT data manifest")
    parser.add_argument("--evaluation_manifest", required=True, help="评估排除 manifest")
    parser.add_argument("--tokenizer_path", required=True, help="固定 Tokenizer 目录")
    parser.add_argument("--parent_weights", required=True, help="CPT BF16 model-only 父权重")
    parser.add_argument("--parent_sha256", type=_sha256_value, required=True)
    parser.add_argument("--resume", action="store_true", help="严格恢复 latest.pt")
    parser.add_argument(
        "--allow_provisional_data",
        action="store_true",
        help="显式允许仅用于接入或冒烟的 provisional 数据",
    )

    parser.add_argument("--peak_lr", type=float, required=True)
    parser.add_argument(
        "--stop_assistant_tokens", type=pretrain_entry._positive_int, required=True
    )
    parser.add_argument(
        "--warmup_assistant_tokens", type=pretrain_entry._positive_int, required=True
    )
    parser.add_argument(
        "--schedule_assistant_tokens", type=pretrain_entry._positive_int, required=True
    )
    parser.add_argument("--floor_ratio", type=float, default=0.1)
    parser.add_argument(
        "--micro_batch_size", type=pretrain_entry._positive_int, required=True
    )
    parser.add_argument(
        "--accumulation_steps", type=pretrain_entry._positive_int, required=True
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument(
        "--log_interval_tokens", type=pretrain_entry._positive_int, default=100_000
    )
    parser.add_argument(
        "--quick_interval_tokens",
        type=pretrain_entry._non_negative_int,
        default=0,
    )
    parser.add_argument(
        "--full_interval_tokens",
        type=pretrain_entry._non_negative_int,
        default=0,
    )
    parser.add_argument(
        "--fixed_quick_tokens", type=pretrain_entry._token_thresholds, default=()
    )
    parser.add_argument(
        "--fixed_full_tokens", type=pretrain_entry._token_thresholds, default=()
    )
    parser.add_argument(
        "--eval_batch_size", type=pretrain_entry._positive_int, required=True
    )

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument(
        "--num_workers", type=pretrain_entry._non_negative_int, default=8
    )
    parser.add_argument(
        "--eval_num_workers", type=pretrain_entry._non_negative_int, default=4
    )
    parser.add_argument("--seed", type=pretrain_entry._non_negative_int, default=42)
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
    if args.peak_lr <= 0 or args.grad_clip <= 0:
        raise ValueError("peak_lr 和 grad_clip 必须大于 0")
    if not 0 < args.floor_ratio <= 1:
        raise ValueError("floor_ratio 必须位于 (0, 1] 区间")
    if args.schedule_assistant_tokens <= args.warmup_assistant_tokens:
        raise ValueError("schedule_assistant_tokens 必须大于 warmup_assistant_tokens")
    if Path(args.run_dir).resolve() == Path(args.checkpoint_dir).resolve():
        raise ValueError("run_dir 和 checkpoint_dir 必须分离")
    if int(os.environ.get("RANK", -1)) != -1:
        raise RuntimeError("法律 SFT 当前只支持单卡，不能使用 torchrun/DDP")


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


def _load_verified_json(path: Path, description: str) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    hash_path = path.with_suffix(".sha256")
    if not path.is_file() or not hash_path.is_file():
        raise FileNotFoundError(f"{description} 或相邻 SHA-256 清单不存在: {path}")
    digest = _sha256_file(path)
    expected_line = f"{digest}  {path.name}"
    lines = [line for line in hash_path.read_text(encoding="utf-8").splitlines() if line]
    if lines != [expected_line]:
        raise ValueError(f"{description} SHA-256 校验失败")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} 不是有效 UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} 必须是 JSON object")
    return payload, digest


def _load_training_manifests(
    data_manifest_path: str | Path,
    evaluation_manifest_path: str | Path,
    *,
    allow_provisional_data: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """验证 all-retained 数据与评估排除身份。"""
    data_path = Path(data_manifest_path).resolve()
    evaluation_path = Path(evaluation_manifest_path).resolve()
    data_manifest, data_sha256 = _load_verified_json(data_path, "SFT data manifest")
    evaluation_manifest, evaluation_sha256 = _load_verified_json(
        evaluation_path, "评估排除 manifest"
    )
    if (
        data_manifest.get("schema_version") != "1.0"
        or data_manifest.get("pipeline") != "disc_law_sft_length_filtered_768_v1"
        or data_manifest.get("complete") is not True
    ):
        raise ValueError("SFT data manifest 版本、pipeline 或完成状态无效")
    policy = data_manifest.get("policy")
    if not isinstance(policy, dict) or policy.get("fixed_max_seq_len") != EXPECTED_SEQUENCE_LENGTH:
        raise ValueError("SFT data manifest 未固定 max_seq_len=768")
    evaluation_schema = evaluation_manifest.get("schema_version")
    if evaluation_schema not in {"1.0", "1.1", "1.2"} or evaluation_manifest.get(
        "pipeline"
    ) != "legal_sft_evaluation_exclusions":
        raise ValueError("评估排除 manifest 版本或 pipeline 无效")
    evaluation_identity = data_manifest.get("evaluation_exclusion")
    if (
        not isinstance(evaluation_identity, dict)
        or evaluation_identity.get("sha256") != evaluation_sha256
    ):
        raise ValueError("SFT data manifest 绑定的评估排除 manifest 身份不一致")
    readiness = data_manifest.get("readiness")
    formal_training_ready = bool(
        isinstance(readiness, dict)
        and readiness.get("training_ready") is True
        and data_manifest.get("release_status") == "formal_training_candidate"
        and evaluation_schema == "1.2"
        and evaluation_manifest.get("complete_for_formal_sft") is True
    )
    if not formal_training_ready and not allow_provisional_data:
        raise ValueError(
            "当前数据仍是 provisional；接入或冒烟必须显式使用 "
            "--allow_provisional_data，正式训练仍被阻断"
        )
    identity = {
        "data_manifest_path": str(data_path),
        "data_manifest_sha256": data_sha256,
        "evaluation_manifest_path": str(evaluation_path),
        "evaluation_manifest_sha256": evaluation_sha256,
        "data_tier": DATA_TIER,
        "formal_training_ready": formal_training_ready,
    }
    return data_manifest, evaluation_manifest, identity


def _load_tokenizer(path: str | Path) -> Any:
    tokenizer_path = Path(path).resolve()
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Tokenizer 目录不存在: {tokenizer_path}")
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            tokenizer_path, use_fast=True, local_files_only=True
        )
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"无法加载固定 Tokenizer: {tokenizer_path}") from error


def _tokenizer_identity(tokenizer: Any, path: str | Path) -> dict[str, Any]:
    tokenizer_path = Path(path).resolve()
    files = {}
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        file_path = tokenizer_path / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"Tokenizer 指纹文件不存在: {file_path}")
        files[filename] = {
            "bytes": file_path.stat().st_size,
            "sha256": _sha256_file(file_path),
        }
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("Tokenizer 缺少 chat template")
    return {
        "path": str(tokenizer_path),
        "vocab_size": len(tokenizer),
        "chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest(),
        "files": files,
    }


def _model_config() -> MiniMindConfig:
    return MiniMindConfig(**MODEL_ARCHITECTURE)


def _load_parent_weights(path: str | Path, expected_sha256: str, model: torch.nn.Module) -> str:
    parent_path = Path(path).resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(f"CPT 父权重不存在: {parent_path}")
    actual_sha256 = _sha256_file(parent_path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "CPT 父权重 SHA-256 不匹配: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    try:
        state = torch.load(parent_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise ValueError("无法读取 CPT model-only 父权重") from error
    if not isinstance(state, dict) or not state:
        raise ValueError("CPT model-only 父权重必须是非空 state_dict")
    try:
        model.load_state_dict(state, strict=True)
    except (TypeError, RuntimeError) as error:
        raise ValueError("CPT 父权重与法律 SFT 模型结构不兼容") from error
    return actual_sha256


def _training_invariants(
    args: argparse.Namespace,
    *,
    manifest_identity: dict[str, Any],
    tokenizer_identity: dict[str, Any],
    train_sequence_count: int,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    return {
        "pipeline": "legal_full_sft",
        "data_manifest_sha256": manifest_identity["data_manifest_sha256"],
        "evaluation_manifest_sha256": manifest_identity[
            "evaluation_manifest_sha256"
        ],
        "data_tier": DATA_TIER,
        "parent_weights_sha256": args.parent_sha256,
        "model": MODEL_ARCHITECTURE,
        "tokenizer": tokenizer_identity,
        "sequence_length": EXPECTED_SEQUENCE_LENGTH,
        "label_mask_version": LABEL_MASK_VERSION,
        "train_sequence_count": train_sequence_count,
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


def _training_controls(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "stop_assistant_tokens": args.stop_assistant_tokens,
        "log_interval_tokens": args.log_interval_tokens,
        "quick_interval_tokens": args.quick_interval_tokens,
        "full_interval_tokens": args.full_interval_tokens,
        "fixed_quick_tokens": sorted(
            {args.warmup_assistant_tokens, *args.fixed_quick_tokens}
        ),
        "fixed_full_tokens": sorted(
            {
                args.stop_assistant_tokens,
                args.schedule_assistant_tokens,
                *args.fixed_full_tokens,
            }
        ),
        "eval_batch_size": args.eval_batch_size,
        "eval_num_workers": args.eval_num_workers,
    }


def _jsonable_arguments(args: argparse.Namespace) -> dict[str, Any]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in vars(args).items()
    }


def _run_manifest_payload(
    args: argparse.Namespace,
    *,
    manifest_identity: dict[str, Any],
    tokenizer_identity: dict[str, Any],
    invariants: dict[str, Any],
    controls: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "pipeline": "legal_full_sft",
        "run_name": args.run_name,
        "created_at": pretrain_entry._utc_now(),
        "experimental_scope": (
            "all-retained provisional integration"
            if not manifest_identity["formal_training_ready"]
            else None
        ),
        "parent_model": {
            "path": str(Path(args.parent_weights).resolve()),
            "sha256": args.parent_sha256,
            "load_semantics": "strict_model_state_only",
        },
        "data": {
            "manifest": manifest_identity["data_manifest_path"],
            "manifest_sha256": manifest_identity["data_manifest_sha256"],
            "evaluation_manifest": manifest_identity["evaluation_manifest_path"],
            "evaluation_manifest_sha256": manifest_identity[
                "evaluation_manifest_sha256"
            ],
            "tier": DATA_TIER,
            "formal_training_ready": manifest_identity["formal_training_ready"],
        },
        "tokenizer": tokenizer_identity,
        "training_invariants": invariants,
        "initial_controls": controls,
        "arguments": _jsonable_arguments(args),
    }


def _write_immutable_run_manifest(path: Path, payload: dict[str, Any]) -> str:
    hash_path = path.with_suffix(".sha256")
    if path.exists() or hash_path.exists():
        raise FileExistsError("SFT run manifest 已存在，不能覆盖")
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8", newline="\n")
    temporary.replace(path)
    digest = _sha256_file(path)
    hash_temporary = hash_path.with_suffix(hash_path.suffix + ".tmp")
    hash_temporary.write_text(
        f"{digest}  {path.name}\n", encoding="utf-8", newline="\n"
    )
    hash_temporary.replace(hash_path)
    return digest


def _load_verified_run_manifest(path: Path) -> tuple[dict[str, Any], str]:
    return _load_verified_json(path, "SFT run manifest")


def _validation_record(
    report: dict[str, Any], progress: SftProgress, eval_batch_size: int
) -> dict[str, Any]:
    split = report["split"]
    prefix = f"validation/{split}"
    record: dict[str, Any] = {
        "type": "validation",
        "timestamp": pretrain_entry._utc_now(),
        "split": split,
        "optimizer_step": progress.optimizer_step,
        "completed_assistant_tokens": progress.completed_assistant_tokens,
        "eval_batch_size": eval_batch_size,
        f"{prefix}/overall/loss": report["overall"]["loss"],
        f"{prefix}/overall/perplexity": report["overall"]["perplexity"],
        f"{prefix}/overall/assistant_tokens": report["overall"][
            "assistant_tokens"
        ],
        f"{prefix}/overall/sequences": report["overall"]["sequences"],
    }
    for source, source_report in report["sources"].items():
        source_prefix = f"{prefix}/{source}"
        for key in ("loss", "perplexity", "assistant_tokens", "sequences"):
            record[f"{source_prefix}/{key}"] = source_report[key]
    return record


def _group_assistant_tokens(
    micro_batches: tuple[tuple[torch.Tensor, torch.Tensor], ...]
) -> int:
    return sum(int((labels[:, 1:] != -100).sum().item()) for _, labels in micro_batches)


def _export_new_weights(path: Path, model: torch.nn.Module) -> None:
    if path.exists():
        raise FileExistsError(f"SFT BF16 权重已存在，不能覆盖: {path}")
    export_sft_weights(path, model)


def run_training(args: argparse.Namespace) -> SftProgress:
    """执行单卡 all-retained 法律 SFT，并返回最终 optimizer 边界进度。"""
    _validate_args(args)
    paths = _resolve_paths(args)
    if args.resume:
        if not paths["resume"].is_file():
            raise FileNotFoundError(f"--resume 指定的恢复点不存在: {paths['resume']}")
    elif paths["run_dir"].exists() or paths["checkpoint_dir"].exists():
        raise FileExistsError("新 SFT run 目录已存在；请更换目录或使用 --resume")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定了 CUDA，但当前环境没有可用 GPU")
    if args.dtype == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 CUDA 设备不支持 BF16")
    autocast_dtype = torch.bfloat16 if args.dtype == "bfloat16" else None

    _, _, manifest_identity = _load_training_manifests(
        args.data_manifest,
        args.evaluation_manifest,
        allow_provisional_data=args.allow_provisional_data,
    )
    tokenizer = _load_tokenizer(args.tokenizer_path)
    tokenizer_record = _tokenizer_identity(tokenizer, args.tokenizer_path)
    train_dataset = SftDataset(args.data_manifest, "train", tokenizer)
    if len(train_dataset) == 0:
        train_dataset.close()
        raise ValueError("SFT train 数据集不能为空")

    tracker = None
    try:
        setup_seed(args.seed)
        torch.set_float32_matmul_precision("high")
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        model_config = _model_config()
        model = MiniMindForCausalLM(model_config).to(device)
        get_model_params(model, model_config)
        if not args.resume:
            _load_parent_weights(args.parent_weights, args.parent_sha256, model)
            Logger(f"CPT 父权重严格加载成功: {args.parent_sha256}")
        optimizer = build_sft_optimizer(
            model, peak_lr=args.peak_lr, device_type=device.type
        )
        invariants = _training_invariants(
            args,
            manifest_identity=manifest_identity,
            tokenizer_identity=tokenizer_record,
            train_sequence_count=len(train_dataset),
            optimizer=optimizer,
        )
        controls = _training_controls(args)
        progress = SftProgress()
        resume_run_id = None
        if args.resume:
            run_manifest, run_manifest_sha256 = _load_verified_run_manifest(
                paths["run_manifest"]
            )
            if run_manifest.get("run_name") != args.run_name:
                raise ValueError("SFT run manifest 的 run_name 与当前命令不一致")
            if run_manifest.get("training_invariants") != invariants:
                raise ValueError("SFT run manifest 训练不变量与当前配置不一致")
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
                raise ValueError("SFT 恢复点数据位置超过当前 train 数据集")
        else:
            run_manifest_sha256 = _write_immutable_run_manifest(
                paths["run_manifest"],
                _run_manifest_payload(
                    args,
                    manifest_identity=manifest_identity,
                    tokenizer_identity=tokenizer_record,
                    invariants=invariants,
                    controls=controls,
                ),
            )

        paths["checkpoint_dir"].mkdir(parents=True, exist_ok=True)
        tracker, swanlab_run = pretrain_entry._init_swanlab(
            args,
            paths["run_dir"],
            {**invariants, **controls},
            resume_run_id,
        )
        active_run_id = pretrain_entry._current_swanlab_run_id(
            swanlab_run, resume_run_id
        )
        metric_logger = JsonlMetricLogger(paths["metrics"], tracker=tracker)
        metric_logger.log(
            {
                "type": "run_resume" if args.resume else "run_start",
                "timestamp": pretrain_entry._utc_now(),
                "run_name": args.run_name,
                "optimizer_step": progress.optimizer_step,
                "completed_assistant_tokens": progress.completed_assistant_tokens,
                "run_manifest_sha256": run_manifest_sha256,
                "controls": controls,
            }
        )

        def dataset_factory(split: str, dataset_kind: str):
            return SftDataset(
                args.data_manifest,
                split,
                tokenizer,
                dataset_kind=dataset_kind,
            )

        def evaluate(split: str) -> None:
            report = evaluate_sft_by_source(
                model,
                dataset_factory,
                split=split,
                batch_size=args.eval_batch_size,
                device=device,
                num_workers=args.eval_num_workers,
                autocast_dtype=autocast_dtype,
            )
            metric_logger.log(_validation_record(report, progress, args.eval_batch_size))

        if not args.resume:
            evaluate("validation/quick")
            evaluate("validation/full")
        if args.use_compile:
            model = torch.compile(model)
            Logger("torch.compile 已启用")

        fixed_quick_tokens = tuple(controls["fixed_quick_tokens"])
        fixed_full_tokens = tuple(controls["fixed_full_tokens"])
        last_saved_tokens = progress.completed_assistant_tokens if args.resume else -1
        current_weight_path = (
            paths["weights"] / f"legal-sft-{progress.completed_assistant_tokens}.pth"
        )
        last_exported_tokens = (
            progress.completed_assistant_tokens
            if args.resume and current_weight_path.is_file()
            else -1
        )
        last_full_validation_tokens = 0 if not args.resume else -1
        training_started = time.perf_counter()
        stop_reason = "assistant_token_budget_reached"
        stop_before_next_group = False
        while progress.completed_assistant_tokens < args.stop_assistant_tokens:
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
                step_metrics = run_sft_optimizer_step(
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
                    step_metrics.sequence_count,
                    len(train_dataset),
                )
                progress = SftProgress(
                    epoch=epoch,
                    next_sequence_position=next_position,
                    completed_sequences=(
                        progress.completed_sequences + step_metrics.sequence_count
                    ),
                    completed_assistant_tokens=(
                        previous_tokens + step_metrics.assistant_tokens
                    ),
                    optimizer_step=progress.optimizer_step + 1,
                )
                should_log = progress.optimizer_step == 1 or pretrain_entry._crossed_interval(
                    previous_tokens,
                    progress.completed_assistant_tokens,
                    args.log_interval_tokens,
                )
                if should_log:
                    step_seconds = time.perf_counter() - step_started
                    record = {
                        "type": "train",
                        "timestamp": pretrain_entry._utc_now(),
                        "optimizer_step": progress.optimizer_step,
                        "epoch": progress.epoch,
                        "next_sequence_position": progress.next_sequence_position,
                        "completed_sequences": progress.completed_sequences,
                        "completed_assistant_tokens": progress.completed_assistant_tokens,
                        "loss": step_metrics.loss,
                        "logits_loss": step_metrics.logits_loss,
                        "aux_loss": step_metrics.aux_loss,
                        "learning_rate": learning_rate,
                        "grad_norm": step_metrics.grad_norm,
                        "step_input_tokens": step_metrics.input_tokens,
                        "step_assistant_tokens": step_metrics.assistant_tokens,
                        "assistant_tokens_per_second": (
                            step_metrics.assistant_tokens / step_seconds
                        ),
                        "elapsed_seconds": time.perf_counter() - training_started,
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
                        f"assistant_tokens={progress.completed_assistant_tokens:,} "
                        f"loss={step_metrics.loss:.4f} lr={learning_rate:.3e}"
                    )

                actions = crossed_assistant_token_events(
                    previous_tokens,
                    progress.completed_assistant_tokens,
                    quick_interval_tokens=args.quick_interval_tokens,
                    full_interval_tokens=args.full_interval_tokens,
                    fixed_quick_tokens=fixed_quick_tokens,
                    fixed_full_tokens=fixed_full_tokens,
                )
                if actions.full_validation:
                    evaluate("validation/full")
                    last_full_validation_tokens = progress.completed_assistant_tokens
                elif actions.quick_validation:
                    evaluate("validation/quick")
                if actions.save_resume:
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
                if actions.export_weights:
                    _export_new_weights(
                        paths["weights"]
                        / f"legal-sft-{progress.completed_assistant_tokens}.pth",
                        model,
                    )
                    last_exported_tokens = progress.completed_assistant_tokens
            del loader
            if stop_before_next_group:
                break

        if progress.optimizer_step == 0 and stop_before_next_group:
            raise ValueError("stop_assistant_tokens 小于首个 optimizer step 的监督量")
        if last_full_validation_tokens != progress.completed_assistant_tokens:
            evaluate("validation/full")
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
        if last_exported_tokens != progress.completed_assistant_tokens:
            _export_new_weights(
                paths["weights"] / f"legal-sft-{progress.completed_assistant_tokens}.pth",
                model,
            )
        metric_logger.log(
            {
                "type": "run_complete",
                "timestamp": pretrain_entry._utc_now(),
                "optimizer_step": progress.optimizer_step,
                "completed_assistant_tokens": progress.completed_assistant_tokens,
                "completed_sequences": progress.completed_sequences,
                "epoch": progress.epoch,
                "next_sequence_position": progress.next_sequence_position,
                "stop_reason": stop_reason,
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
                    f"SwanLab finish 失败: {error}", RuntimeWarning, stacklevel=2
                )


def main() -> None:
    args = build_parser().parse_args()
    progress = run_training(args)
    Logger(
        f"法律 SFT 结束: assistant_tokens={progress.completed_assistant_tokens:,}, "
        f"optimizer_steps={progress.optimizer_step:,}"
    )


if __name__ == "__main__":
    main()
