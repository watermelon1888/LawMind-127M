"""从阶段 B 最终模型初始化并执行法律持续训练（CPT）。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

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
from . import train_pretrain as pretrain_entry
from .pretrain_runtime import (
    JsonlMetricLogger,
    PretrainProgress,
    advance_pretrain_position,
    build_pretrain_optimizer,
    crossed_token_events,
    export_pretrain_weights,
    group_micro_batches,
    load_pretrain_checkpoint,
    run_pretrain_optimizer_step,
    save_pretrain_checkpoint,
    token_learning_rate,
)
from .trainer_utils import Logger, get_model_params, setup_seed


DEFAULT_WORK_ROOT = "/root/autodl-tmp/minimind-work"
DEFAULT_CPT_SHARD_MANIFEST = (
    "/root/autodl-tmp/minimind-work/manifests/cpt-shards-v1.json"
)
DEFAULT_GENERAL_SHARD_MANIFEST = (
    "/root/autodl-tmp/minimind-work/manifests/stage-b-shards-v1.json"
)
DEFAULT_PARENT_WEIGHTS = (
    "/root/autodl-tmp/minimind-work/checkpoints/stage-b/"
    "stage-b-lr-1e-3/weights/pretrain-9737362944.pth"
)
DEFAULT_PARENT_SHA256 = (
    "1e8e0c27f0a36e2e1fff7de81890baf0304eaa7032f9739492a4d6001f61a8ba"
)
PARENT_COMPLETED_TOKENS = 9_737_362_944
EXPECTED_CPT_SOURCES = ("npc_flk", "fuzi_mingcha")
EXPECTED_SEQUENCE_LENGTH = 768
EFFECTIVE_BATCH_SEQUENCES = 256


class CptTrainDataset(ConcatDataset):
    """按来源曝光次数连接 Dataset，并确保每个底层 Dataset 只关闭一次。"""

    def __init__(
        self,
        virtual_datasets: list[Dataset],
        owned_datasets: list[Dataset],
    ) -> None:
        super().__init__(virtual_datasets)
        self.owned_datasets = tuple(owned_datasets)

    def close(self) -> None:
        """关闭所有来源 Dataset 持有的 mmap。"""
        for dataset in self.owned_datasets:
            close = getattr(dataset, "close", None)
            if callable(close):
                close()


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
    """创建法律 CPT 训练 CLI。"""
    parser = argparse.ArgumentParser(description="MiniMind 法律持续训练（CPT）")
    parser.add_argument("--run_name", required=True, help="实验名称和本地目录名")
    parser.add_argument(
        "--work_root",
        default=os.environ.get("MINIMIND_WORK_ROOT", DEFAULT_WORK_ROOT),
        help="minimind-work 根目录",
    )
    parser.add_argument(
        "--cpt_shard_manifest",
        default=DEFAULT_CPT_SHARD_MANIFEST,
        help="CPT token shard manifest",
    )
    parser.add_argument(
        "--general_shard_manifest",
        default=DEFAULT_GENERAL_SHARD_MANIFEST,
        help="阶段 B 通用 full validation shard manifest",
    )
    parser.add_argument(
        "--parent_weights",
        default=DEFAULT_PARENT_WEIGHTS,
        help="阶段 B 最终 BF16 model-only 权重",
    )
    parser.add_argument(
        "--parent_sha256",
        type=_sha256_value,
        default=DEFAULT_PARENT_SHA256,
        help="阶段 B 父权重 SHA-256",
    )
    parser.add_argument(
        "--parent_completed_tokens",
        type=pretrain_entry._positive_int,
        default=PARENT_COMPLETED_TOKENS,
        help="阶段 B 父模型最终累计 tokens",
    )
    parser.add_argument("--resume", action="store_true", help="严格恢复 CPT latest.pt")

    parser.add_argument("--peak_lr", type=float, required=True)
    parser.add_argument("--stop_tokens", type=pretrain_entry._positive_int, required=True)
    parser.add_argument("--warmup_tokens", type=pretrain_entry._positive_int, required=True)
    parser.add_argument("--schedule_tokens", type=pretrain_entry._positive_int, required=True)
    parser.add_argument("--npc_repeat", type=pretrain_entry._positive_int, required=True)
    parser.add_argument("--floor_ratio", type=float, default=0.1)
    parser.add_argument("--micro_batch_size", type=pretrain_entry._positive_int, default=16)
    parser.add_argument("--accumulation_steps", type=pretrain_entry._positive_int, default=16)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument(
        "--log_interval_tokens", type=pretrain_entry._positive_int, default=10_000_000
    )
    parser.add_argument(
        "--quick_interval_tokens",
        type=pretrain_entry._non_negative_int,
        default=100_000_000,
    )
    parser.add_argument(
        "--full_interval_tokens",
        type=pretrain_entry._non_negative_int,
        default=500_000_000,
    )
    parser.add_argument(
        "--fixed_quick_tokens", type=pretrain_entry._token_thresholds, default=()
    )
    parser.add_argument(
        "--fixed_full_tokens", type=pretrain_entry._token_thresholds, default=()
    )
    parser.add_argument("--eval_batch_size", type=pretrain_entry._positive_int, default=64)

    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--num_workers", type=pretrain_entry._non_negative_int, default=8)
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
    if pretrain_entry.RUN_NAME_RE.fullmatch(args.run_name) is None:
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
        raise RuntimeError("CPT 当前只支持单卡训练，不能使用 torchrun/DDP")


def _resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    work_root = Path(args.work_root).resolve()
    run_dir = work_root / "runs" / "cpt" / args.run_name
    checkpoint_dir = work_root / "checkpoints" / "cpt" / args.run_name
    return {
        "run_dir": run_dir,
        "metrics": run_dir / "metrics.jsonl",
        "run_manifest": run_dir / "run-manifest.json",
        "checkpoint_dir": checkpoint_dir,
        "resume": checkpoint_dir / "resume" / "latest.pt",
        "weights": checkpoint_dir / "weights",
    }


def _build_train_dataset(
    catalog: PretrainShardCatalog,
    npc_repeat: int,
) -> tuple[CptTrainDataset, dict[str, dict[str, int]]]:
    if catalog.source_names != EXPECTED_CPT_SOURCES:
        raise ValueError(
            f"CPT manifest 来源必须严格为 {EXPECTED_CPT_SOURCES}，"
            f"实际为 {catalog.source_names}"
        )
    virtual_datasets: list[Dataset] = []
    owned_datasets: list[Dataset] = []
    exposure: dict[str, dict[str, int]] = {}
    for source in catalog.source_names:
        dataset = catalog.create_dataset("train", sources=(source,))
        if len(dataset) == 0:
            raise ValueError(f"CPT train 来源不能为空: {source}")
        repeat = npc_repeat if source == "npc_flk" else 1
        owned_datasets.append(dataset)
        virtual_datasets.extend([dataset] * repeat)
        unique_sequences = len(dataset)
        exposure[source] = {
            "unique_sequences": unique_sequences,
            "trainable_unique_input_tokens": unique_sequences
            * catalog.sequence_length,
            "repeat": repeat,
            "exposed_sequences_per_epoch": unique_sequences * repeat,
            "exposed_input_tokens_per_epoch": unique_sequences
            * repeat
            * catalog.sequence_length,
        }
    return CptTrainDataset(virtual_datasets, owned_datasets), exposure


def _load_parent_weights(
    path: Path,
    expected_sha256: str,
    model: torch.nn.Module,
) -> str:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"阶段 B 父权重不存在: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "阶段 B 父权重 SHA-256 不匹配: "
            f"expected={expected_sha256}, actual={actual_sha256}"
        )
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, EOFError) as error:
        raise ValueError("无法读取阶段 B model-only 权重") from error
    if not isinstance(state, dict) or not state:
        raise ValueError("阶段 B model-only 权重必须是非空 state_dict")
    if any(
        not isinstance(name, str) or not torch.is_tensor(value)
        for name, value in state.items()
    ):
        raise ValueError("阶段 B model-only 权重包含非法 state_dict 项")
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ValueError("阶段 B 父权重与 CPT 模型结构不兼容") from error
    return actual_sha256


def _load_cpt_manifest(catalog: PretrainShardCatalog) -> dict[str, Any]:
    try:
        manifest = json.loads(catalog.manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("无法读取已验证的 CPT shard manifest") from error
    if not isinstance(manifest, dict):
        raise ValueError("CPT shard manifest 必须是 JSON object")
    return manifest


def _load_tokenizer_record(manifest: dict[str, Any]) -> dict[str, Any]:
    tokenizer = manifest.get("tokenizer") if isinstance(manifest, dict) else None
    if not isinstance(tokenizer, dict) or not tokenizer:
        raise ValueError("CPT shard manifest 缺少 Tokenizer 指纹")
    return tokenizer


def _complete_source_exposure(
    manifest: dict[str, Any],
    exposure: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or tuple(sources) != EXPECTED_CPT_SOURCES:
        raise ValueError("CPT shard manifest 来源统计缺失或顺序不一致")
    completed: dict[str, dict[str, int]] = {}
    for source in EXPECTED_CPT_SOURCES:
        source_report = sources.get(source)
        train = source_report.get("train") if isinstance(source_report, dict) else None
        if not isinstance(train, dict):
            raise ValueError(f"CPT shard manifest 缺少 {source}/train 统计")
        names = (
            "text_tokens",
            "tokens_before_tail",
            "written_tokens",
            "dropped_tail_tokens",
            "sequence_count",
        )
        values = {name: train.get(name) for name in names}
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in values.values()
        ):
            raise ValueError(f"CPT shard manifest 的 {source}/train token 统计无效")
        base = exposure[source]
        if values["sequence_count"] != base["unique_sequences"]:
            raise ValueError(f"{source}/train sequence_count 与 Dataset 不一致")
        if values["written_tokens"] != base["trainable_unique_input_tokens"]:
            raise ValueError(f"{source}/train written_tokens 与 Dataset 不一致")
        if (
            values["written_tokens"] + values["dropped_tail_tokens"]
            != values["tokens_before_tail"]
        ):
            raise ValueError(f"{source}/train written/tail token 统计不闭合")
        completed[source] = {
            "unique_sequences": base["unique_sequences"],
            "unique_text_tokens": values["text_tokens"],
            "unique_framed_tokens": values["tokens_before_tail"],
            "trainable_unique_input_tokens": values["written_tokens"],
            "dropped_tail_tokens": values["dropped_tail_tokens"],
            "repeat": base["repeat"],
            "exposed_sequences_per_epoch": base["exposed_sequences_per_epoch"],
            "exposed_input_tokens_per_epoch": base[
                "exposed_input_tokens_per_epoch"
            ],
        }
    return completed


def _training_invariants(
    args: argparse.Namespace,
    cpt_catalog: PretrainShardCatalog,
    general_catalog: PretrainShardCatalog,
    model_config: MiniMindConfig,
    train_sequence_count: int,
    source_exposure: dict[str, dict[str, int]],
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    return {
        "pipeline": "cpt",
        "cpt_shard_manifest_sha256": cpt_catalog.manifest_sha256,
        "general_shard_manifest_sha256": general_catalog.manifest_sha256,
        "parent_weights_sha256": args.parent_sha256,
        "parent_completed_tokens": args.parent_completed_tokens,
        "sequence_length": cpt_catalog.sequence_length,
        "train_sequence_count": train_sequence_count,
        "source_exposure": source_exposure,
        "model": pretrain_entry._model_architecture(model_config),
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
        "fixed_quick_tokens": sorted(
            {args.warmup_tokens, *args.fixed_quick_tokens}
        ),
        "fixed_full_tokens": sorted(
            {args.stop_tokens, args.schedule_tokens, *args.fixed_full_tokens}
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
    cpt_catalog: PretrainShardCatalog,
    general_catalog: PretrainShardCatalog,
    tokenizer: dict[str, Any],
    invariants: dict[str, Any],
    controls: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "pipeline": "cpt",
        "run_name": args.run_name,
        "created_at": pretrain_entry._utc_now(),
        "parent_model": {
            "path": str(Path(args.parent_weights).resolve()),
            "sha256": args.parent_sha256,
            "completed_tokens": args.parent_completed_tokens,
            "load_semantics": "strict_model_state_only",
        },
        "data": {
            "cpt_shard_manifest": str(cpt_catalog.manifest_path),
            "cpt_shard_manifest_sha256": cpt_catalog.manifest_sha256,
            "general_validation_manifest": str(general_catalog.manifest_path),
            "general_validation_manifest_sha256": general_catalog.manifest_sha256,
            "unique_text_tokens": sum(
                item["unique_text_tokens"]
                for item in invariants["source_exposure"].values()
            ),
            "unique_framed_tokens": sum(
                item["unique_framed_tokens"]
                for item in invariants["source_exposure"].values()
            ),
            "trainable_unique_input_tokens": sum(
                item["trainable_unique_input_tokens"]
                for item in invariants["source_exposure"].values()
            ),
            "exposed_input_tokens_per_epoch": sum(
                item["exposed_input_tokens_per_epoch"]
                for item in invariants["source_exposure"].values()
            ),
            "source_exposure": invariants["source_exposure"],
        },
        "tokenizer": tokenizer,
        "training_invariants": invariants,
        "initial_controls": controls,
        "arguments": _jsonable_arguments(args),
    }


def _write_immutable_run_manifest(path: Path, payload: dict[str, Any]) -> str:
    hash_path = path.with_suffix(".sha256")
    if path.exists() or hash_path.exists():
        raise FileExistsError("CPT run manifest 已存在，不能覆盖")
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
    hash_path = path.with_suffix(".sha256")
    if not path.is_file() or not hash_path.is_file():
        raise FileNotFoundError("CPT 严格恢复缺少 run manifest 或其哈希文件")
    lines = [line for line in hash_path.read_text(encoding="utf-8").splitlines() if line]
    actual = _sha256_file(path)
    if lines != [f"{actual}  {path.name}"]:
        raise ValueError("CPT run manifest SHA-256 校验失败")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("CPT run manifest 不是有效 UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("CPT run manifest 必须是 JSON object")
    return payload, actual


def _validation_record(
    report: dict[str, Any],
    *,
    domain: str,
    split: str,
    progress: PretrainProgress,
    eval_batch_size: int,
) -> dict[str, Any]:
    prefix = f"validation/{domain}"
    record: dict[str, Any] = {
        "type": "validation",
        "timestamp": pretrain_entry._utc_now(),
        "domain": domain,
        "split": split,
        "optimizer_step": progress.optimizer_step,
        "completed_tokens": progress.completed_tokens,
        "eval_batch_size": eval_batch_size,
        f"{prefix}/overall/loss": report["overall"]["loss"],
        f"{prefix}/overall/perplexity": report["overall"]["perplexity"],
        f"{prefix}/overall/prediction_tokens": report["overall"][
            "prediction_tokens"
        ],
    }
    for source, source_report in report["sources"].items():
        source_prefix = f"{prefix}/{source}"
        record[f"{source_prefix}/loss"] = source_report["loss"]
        record[f"{source_prefix}/perplexity"] = source_report["perplexity"]
        record[f"{source_prefix}/prediction_tokens"] = source_report[
            "prediction_tokens"
        ]
    return record


def _evaluate(
    model: torch.nn.Module,
    catalog: PretrainShardCatalog,
    *,
    domain: str,
    split: str,
    args: argparse.Namespace,
    progress: PretrainProgress,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> dict[str, Any]:
    report, actual_batch_size = pretrain_entry._evaluate_with_oom_fallback(
        model,
        catalog,
        split=split,
        args=args,
        device=device,
        autocast_dtype=autocast_dtype,
    )
    Logger(
        f"{domain}/{split}: tokens={progress.completed_tokens:,} "
        f"loss={report['overall']['loss']:.4f} "
        f"ppl={report['overall']['perplexity']:.2f}"
    )
    return _validation_record(
        report,
        domain=domain,
        split=split,
        progress=progress,
        eval_batch_size=actual_batch_size,
    )


def run_training(args: argparse.Namespace) -> PretrainProgress:
    """执行单卡法律 CPT，并返回最后一个 optimizer 边界进度。"""
    _validate_args(args)
    paths = _resolve_paths(args)
    if args.resume:
        if not paths["resume"].is_file():
            raise FileNotFoundError(f"--resume 指定的恢复点不存在: {paths['resume']}")
    elif paths["run_dir"].exists() or paths["checkpoint_dir"].exists():
        raise FileExistsError("新 CPT run 目录已存在；请更换 run_name 或使用 --resume")

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

    cpt_catalog = PretrainShardCatalog(args.cpt_shard_manifest)
    general_catalog = PretrainShardCatalog(args.general_shard_manifest)
    if cpt_catalog.sequence_length != EXPECTED_SEQUENCE_LENGTH:
        raise ValueError(
            f"CPT sequence_length 必须是 {EXPECTED_SEQUENCE_LENGTH}，"
            f"实际为 {cpt_catalog.sequence_length}"
        )
    if general_catalog.sequence_length != EXPECTED_SEQUENCE_LENGTH:
        raise ValueError(
            f"通用验证 sequence_length 必须是 {EXPECTED_SEQUENCE_LENGTH}，"
            f"实际为 {general_catalog.sequence_length}"
        )
    train_dataset, source_exposure = _build_train_dataset(cpt_catalog, args.npc_repeat)
    tracker = None
    try:
        cpt_manifest = _load_cpt_manifest(cpt_catalog)
        source_exposure = _complete_source_exposure(cpt_manifest, source_exposure)
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
        if not args.resume:
            _load_parent_weights(
                Path(args.parent_weights), args.parent_sha256, model
            )
            Logger(f"阶段 B 父权重严格加载成功: {args.parent_sha256}")

        optimizer = build_pretrain_optimizer(
            model,
            peak_lr=args.peak_lr,
            device_type=device.type,
        )
        invariants = _training_invariants(
            args,
            cpt_catalog,
            general_catalog,
            model_config,
            len(train_dataset),
            source_exposure,
            optimizer,
        )
        controls = _training_controls(args)
        tokenizer_record = _load_tokenizer_record(cpt_manifest)

        progress = PretrainProgress()
        resume_run_id = None
        baseline_records: list[dict[str, Any]] = []
        if args.resume:
            manifest, run_manifest_sha256 = _load_verified_run_manifest(
                paths["run_manifest"]
            )
            if manifest.get("run_name") != args.run_name:
                raise ValueError("CPT run manifest 的 run_name 与当前命令不一致")
            if manifest.get("training_invariants") != invariants:
                raise ValueError("CPT run manifest 训练不变量与当前配置不一致")
            loaded = load_pretrain_checkpoint(
                paths["resume"],
                model=model,
                optimizer=optimizer,
                expected_invariants=invariants,
                controls=controls,
            )
            progress = loaded.progress
            resume_run_id = loaded.swanlab_run_id
            pretrain_entry._validate_resume_progress(
                progress,
                sequence_count=len(train_dataset),
                sequence_length=cpt_catalog.sequence_length,
            )
            Logger(
                f"CPT 恢复成功: tokens={progress.completed_tokens:,}, "
                f"epoch={progress.epoch}, position={progress.next_sequence_position:,}"
            )
        else:
            baseline_records = [
                _evaluate(
                    model,
                    cpt_catalog,
                    domain="legal",
                    split="quick_validation",
                    args=args,
                    progress=progress,
                    device=device,
                    autocast_dtype=autocast_dtype,
                ),
                _evaluate(
                    model,
                    cpt_catalog,
                    domain="legal",
                    split="full_validation",
                    args=args,
                    progress=progress,
                    device=device,
                    autocast_dtype=autocast_dtype,
                ),
                _evaluate(
                    model,
                    general_catalog,
                    domain="general",
                    split="full_validation",
                    args=args,
                    progress=progress,
                    device=device,
                    autocast_dtype=autocast_dtype,
                ),
            ]
            run_manifest_sha256 = _write_immutable_run_manifest(
                paths["run_manifest"],
                _run_manifest_payload(
                    args,
                    cpt_catalog=cpt_catalog,
                    general_catalog=general_catalog,
                    tokenizer=tokenizer_record,
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
                "optimizer_step": progress.optimizer_step,
                "completed_tokens": progress.completed_tokens,
                "run_name": args.run_name,
                "run_manifest_sha256": run_manifest_sha256,
                "controls": controls,
            }
        )
        for record in baseline_records:
            metric_logger.log(record)

        if args.use_compile:
            model = torch.compile(model)
            Logger("torch.compile 已启用")

        fixed_quick_tokens = tuple(controls["fixed_quick_tokens"])
        fixed_full_tokens = tuple(controls["fixed_full_tokens"])
        last_saved_tokens = progress.completed_tokens if args.resume else -1
        training_started = time.perf_counter()
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
                    previous_tokens
                    + group_sequences * cpt_catalog.sequence_length
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

                should_log = progress.optimizer_step == 1 or pretrain_entry._crossed_interval(
                    previous_tokens,
                    progress.completed_tokens,
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
                        "completed_tokens": progress.completed_tokens,
                        "loss": step_metrics.loss,
                        "logits_loss": step_metrics.logits_loss,
                        "aux_loss": step_metrics.aux_loss,
                        "learning_rate": learning_rate,
                        "grad_norm": step_metrics.grad_norm,
                        "step_tokens": step_metrics.input_tokens,
                        "step_tokens_per_second": step_metrics.input_tokens
                        / step_seconds,
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
                if actions.full_validation:
                    metric_logger.log(
                        _evaluate(
                            model,
                            cpt_catalog,
                            domain="legal",
                            split="full_validation",
                            args=args,
                            progress=progress,
                            device=device,
                            autocast_dtype=autocast_dtype,
                        )
                    )
                    metric_logger.log(
                        _evaluate(
                            model,
                            general_catalog,
                            domain="general",
                            split="full_validation",
                            args=args,
                            progress=progress,
                            device=device,
                            autocast_dtype=autocast_dtype,
                        )
                    )
                elif actions.quick_validation:
                    metric_logger.log(
                        _evaluate(
                            model,
                            cpt_catalog,
                            domain="legal",
                            split="quick_validation",
                            args=args,
                            progress=progress,
                            device=device,
                            autocast_dtype=autocast_dtype,
                        )
                    )

                if actions.save_resume:
                    save_pretrain_checkpoint(
                        paths["resume"],
                        model=model,
                        optimizer=optimizer,
                        progress=progress,
                        invariants=invariants,
                        controls=controls,
                        swanlab_run_id=active_run_id,
                    )
                    last_saved_tokens = progress.completed_tokens
                if actions.export_weights:
                    export_pretrain_weights(
                        paths["weights"] / f"cpt-{progress.completed_tokens}.pth",
                        model,
                    )
                if progress.completed_tokens >= args.stop_tokens:
                    reached_stop = True
                    break
            del loader
            if reached_stop:
                break

        if last_saved_tokens != progress.completed_tokens:
            save_pretrain_checkpoint(
                paths["resume"],
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
                "timestamp": pretrain_entry._utc_now(),
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
        f"CPT 训练结束: tokens={progress.completed_tokens:,}, "
        f"optimizer_steps={progress.optimizer_step:,}"
    )


if __name__ == "__main__":
    main()
