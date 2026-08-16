"""逐项执行基础法律 SFT 的共同 LR 校准计划。"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import sft_base_generation_evaluation as generation
from . import sft_base_lr_calibration as calibration
from . import sft_parent_loss_evaluation as loss_evaluation
from . import train_full_sft as training


CommandExecutor = Callable[[Sequence[str], Path], None]


def _subprocess_executor(command: Sequence[str], cwd: Path) -> None:
    subprocess.run(list(command), cwd=cwd, check=True)


def _load_plan(path: str | Path) -> tuple[dict[str, Any], str]:
    plan, digest = calibration._load_verified(path, "基础法律 LR 校准 plan")
    if (
        plan.get("schema_version") != "1.0"
        or plan.get("pipeline") != calibration.PLAN_PIPELINE
        or plan.get("complete") is not True
    ):
        raise ValueError("基础法律 LR 校准 plan 无效")
    spec, spec_digest = calibration._load_verified(
        plan.get("spec", {}).get("path", ""), "基础法律 LR 校准 spec"
    )
    if (
        spec.get("pipeline") != calibration.SPEC_PIPELINE
        or spec_digest != plan.get("spec", {}).get("sha256")
    ):
        raise ValueError("LR 校准 plan 与 spec 身份不一致")
    for name, identity in plan["inputs"].items():
        calibration._verify_identity(identity, f"校准输入 {name}")
    tokenizer = training._load_tokenizer(plan["tokenizer"]["path"])
    if training._tokenizer_identity(tokenizer, plan["tokenizer"]["path"]) != plan[
        "tokenizer"
    ]:
        raise ValueError("LR 校准 Tokenizer 身份发生漂移")
    if (
        plan.get("formal_training_policy", {}).get(
            "must_restart_from_original_cpt_model_only"
        )
        is not True
        or plan.get("formal_training_policy", {}).get(
            "calibration_checkpoint_continuation_forbidden"
        )
        is not True
    ):
        raise ValueError("LR 校准 plan 缺少正式训练重启约束")
    swanlab = plan.get("tracking", {}).get("swanlab", {})
    if (
        swanlab.get("enabled") is not True
        or not isinstance(swanlab.get("project"), str)
        or not swanlab["project"]
        or not isinstance(swanlab.get("workspace"), str)
        or not swanlab["workspace"]
        or swanlab.get("experiment_name_source") != "trial_id"
        or swanlab.get("resume") != "same_trial_run_id"
        or swanlab.get("failure_policy") != "warn_and_continue_local_jsonl"
        or swanlab.get("local_jsonl_always_enabled") is not True
    ):
        raise ValueError("LR 校准 plan 的 SwanLab 跟踪配置无效")
    return plan, digest


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"训练 metrics 无法解析: {path}:{line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"训练 metrics 记录必须是 object: {path}:{line_number}")
            records.append(record)
    return records


def _completed_training_tokens(metrics_path: Path, target: int) -> int | None:
    records = _read_metrics(metrics_path)
    starts = [
        index
        for index, record in enumerate(records)
        if record.get("type") in {"run_start", "run_resume"}
        and record.get("controls", {}).get("stop_assistant_tokens") == target
    ]
    if not starts:
        return None
    completions = [
        record
        for record in records[starts[-1] + 1 :]
        if record.get("type") == "run_complete"
    ]
    if not completions:
        return None
    completed = completions[-1].get("completed_assistant_tokens")
    if type(completed) is not int or completed <= 0 or completed > target:
        raise ValueError("训练完成记录的 assistant-token 进度无效")
    return completed


def _training_stability(metrics_path: Path) -> dict[str, Any]:
    records = _read_metrics(metrics_path)
    train_records = [record for record in records if record.get("type") == "train"]
    if not train_records:
        raise ValueError("校准训练 metrics 缺少 train 记录")
    for record in train_records:
        for key in ("loss", "logits_loss", "aux_loss", "grad_norm", "learning_rate"):
            value = record.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"校准训练指标缺少有限数值: {key}")
            if not math.isfinite(float(value)):
                raise ValueError(f"校准训练出现非有限指标: {key}")
    return {
        "train_metric_records": len(train_records),
        "loss_first": train_records[0]["loss"],
        "loss_last": train_records[-1]["loss"],
        "loss_min": min(record["loss"] for record in train_records),
        "loss_max": max(record["loss"] for record in train_records),
        "grad_norm_max": max(record["grad_norm"] for record in train_records),
        "learning_rate_max": max(
            record["learning_rate"] for record in train_records
        ),
        "nonfinite_metrics": False,
    }


def _training_command(
    plan: Mapping[str, Any], trial: Mapping[str, Any], resume: bool
) -> list[str]:
    root = Path(plan["output_root"])
    config = plan["training"]
    swanlab = plan["tracking"]["swanlab"]
    quick_points = sorted(set(plan["evaluation"]["training_quick_points"]))
    command = [
        sys.executable,
        "-m",
        "trainer.train_full_sft",
        "--run_name",
        trial["trial_id"],
        "--run_dir",
        str(root / trial["run_directory"]),
        "--checkpoint_dir",
        str(root / trial["checkpoint_directory"]),
        "--data_manifest",
        plan["inputs"]["data_manifest"]["path"],
        "--evaluation_manifest",
        plan["inputs"]["evaluation_manifest"]["path"],
        "--tokenizer_path",
        plan["tokenizer"]["path"],
        "--parent_weights",
        trial["parent_weights"]["path"],
        "--parent_sha256",
        trial["parent_weights"]["sha256"],
        "--peak_lr",
        str(trial["peak_lr"]),
        "--stop_assistant_tokens",
        str(config["calibration_stop_assistant_tokens"]),
        "--warmup_assistant_tokens",
        str(config["warmup_assistant_tokens"]),
        "--schedule_assistant_tokens",
        str(config["schedule_assistant_tokens"]),
        "--floor_ratio",
        str(config["floor_ratio"]),
        "--micro_batch_size",
        str(config["micro_batch_size"]),
        "--accumulation_steps",
        str(config["accumulation_steps"]),
        "--grad_clip",
        str(config["grad_clip"]),
        "--fixed_quick_tokens",
        ",".join(str(item) for item in quick_points),
        "--eval_batch_size",
        "64",
        "--device",
        "cuda:0",
        "--dtype",
        config["dtype"],
        "--num_workers",
        "8",
        "--eval_num_workers",
        "4",
        "--seed",
        str(trial["seed"]),
        "--swanlab",
        "--swanlab_project",
        swanlab["project"],
        "--swanlab_workspace",
        swanlab["workspace"],
    ]
    if resume:
        command.append("--resume")
    return command


def _loss_command(
    plan: Mapping[str, Any], weights: Mapping[str, Any], output: Path
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "trainer.sft_parent_loss_evaluation",
        "--suite",
        "general",
        "--suite",
        "legal_full",
        "--general-manifest",
        plan["inputs"]["general_validation"]["path"],
        "--legal-manifest",
        plan["inputs"]["data_manifest"]["path"],
        "--tokenizer-path",
        plan["tokenizer"]["path"],
        "--weights",
        weights["path"],
        "--weights-sha256",
        weights["sha256"],
        "--output",
        str(output),
        "--device",
        "cuda:0",
        "--eval-batch-size",
        "64",
        "--eval-num-workers",
        "4",
    ]


def _generation_command(
    plan: Mapping[str, Any], weights: Mapping[str, Any], output: Path
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "trainer.sft_base_generation_evaluation",
        "evaluate",
        "--generation-manifest",
        plan["inputs"]["generation_manifest"]["path"],
        "--tokenizer-path",
        plan["tokenizer"]["path"],
        "--weights",
        weights["path"],
        "--weights-sha256",
        weights["sha256"],
        "--output",
        str(output),
        "--device",
        "cuda:0",
    ]


def _load_reports(
    loss_path: Path, generation_path: Path, weights_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    loss, _ = calibration._load_verified(loss_path, "校准 loss 报告")
    generated, _ = calibration._load_verified(
        generation_path, "校准固定生成报告"
    )
    if (
        loss.get("pipeline") != loss_evaluation.PIPELINE
        or set(loss.get("suites", {})) != {"general", "legal_full"}
        or loss.get("inputs", {}).get("weights", {}).get("sha256")
        != weights_sha256
    ):
        raise ValueError("校准 loss 报告身份无效")
    if (
        generated.get("pipeline") != generation.REPORT_PIPELINE
        or generated.get("inputs", {}).get("weights", {}).get("sha256")
        != weights_sha256
        or generated.get("complete") is not True
    ):
        raise ValueError("校准固定生成报告身份无效")
    return loss, generated


def _run_evaluations(
    *,
    plan: Mapping[str, Any],
    weights: Mapping[str, Any],
    loss_path: Path,
    generation_path: Path,
    project_root: Path,
    executor: CommandExecutor,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not loss_path.exists() and not loss_path.with_suffix(".sha256").exists():
        executor(_loss_command(plan, weights, loss_path), project_root)
    if (
        not generation_path.exists()
        and not generation_path.with_suffix(".sha256").exists()
    ):
        executor(_generation_command(plan, weights, generation_path), project_root)
    return _load_reports(loss_path, generation_path, weights["sha256"])


def _result_metrics(
    loss: Mapping[str, Any], generated: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "general": loss["suites"]["general"]["metrics"],
        "legal_full": loss["suites"]["legal_full"]["metrics"],
        "fixed_generation": generated["metrics"],
    }


def _run_baseline(
    *,
    plan: Mapping[str, Any],
    plan_digest: str,
    item: Mapping[str, Any],
    project_root: Path,
    executor: CommandExecutor,
) -> dict[str, Any]:
    root = Path(plan["output_root"])
    result_path = root / item["result_path"]
    artifact_root = result_path.parent / "artifacts" / item["candidate_role"]
    loss_path = artifact_root / "loss.json"
    generation_path = artifact_root / "generation.json"
    weights = item["weights"]
    calibration._verify_identity(weights, "baseline 原始父权重")
    loss, generated = _run_evaluations(
        plan=plan,
        weights=weights,
        loss_path=loss_path,
        generation_path=generation_path,
        project_root=project_root,
        executor=executor,
    )
    result = {
        "schema_version": "1.0",
        "pipeline": calibration.BASELINE_PIPELINE,
        "plan_sha256": plan_digest,
        "observation_id": item["observation_id"],
        "candidate_role": item["candidate_role"],
        "weights": weights,
        "metrics": _result_metrics(loss, generated),
        "private_holdout_evaluated": False,
        "complete": True,
    }
    calibration._write_immutable_json(result_path, result)
    return result


def _run_trial(
    *,
    plan: Mapping[str, Any],
    plan_digest: str,
    trial: Mapping[str, Any],
    project_root: Path,
    executor: CommandExecutor,
) -> dict[str, Any]:
    if (
        trial.get("purpose") != "lr_calibration_only"
        or trial.get("eligible_for_formal_continuation") is not False
        or trial.get("resume_scope") != "same_trial_only"
    ):
        raise ValueError("trial 缺少校准专用和禁止正式续训约束")
    calibration._verify_identity(trial["parent_weights"], "trial 原始 CPT 父权重")
    root = Path(plan["output_root"])
    run_dir = root / trial["run_directory"]
    checkpoint_dir = root / trial["checkpoint_directory"]
    metrics_path = run_dir / "metrics.jsonl"
    target = plan["training"]["calibration_stop_assistant_tokens"]
    completed = _completed_training_tokens(metrics_path, target)
    if completed is None:
        resume_path = checkpoint_dir / "resume" / "latest.pt"
        resume = resume_path.is_file()
        if not resume and (run_dir.exists() or checkpoint_dir.exists()):
            raise RuntimeError("校准 trial 有残留目录但没有安全恢复点，禁止覆盖")
        executor(_training_command(plan, trial, resume), project_root)
        completed = _completed_training_tokens(metrics_path, target)
    if completed is None:
        raise RuntimeError("校准训练命令结束但没有有效 run_complete 记录")

    weight_path = checkpoint_dir / "weights" / f"legal-sft-{completed}.pth"
    weights = calibration._identity(weight_path)
    loss_path = root / trial["loss_report_path"]
    generation_path = root / trial["generation_report_path"]
    loss, generated = _run_evaluations(
        plan=plan,
        weights=weights,
        loss_path=loss_path,
        generation_path=generation_path,
        project_root=project_root,
        executor=executor,
    )
    result = {
        "schema_version": "1.0",
        "pipeline": calibration.TRIAL_PIPELINE,
        "plan_sha256": plan_digest,
        "trial_id": trial["trial_id"],
        "candidate_role": trial["candidate_role"],
        "parent_weights": trial["parent_weights"],
        "peak_lr": trial["peak_lr"],
        "seed": trial["seed"],
        "requested_stop_assistant_tokens": target,
        "completed_assistant_tokens": completed,
        "training_stability": _training_stability(metrics_path),
        "metrics": _result_metrics(loss, generated),
        "calibration_weights": weights,
        "purpose": "lr_calibration_only",
        "eligible_for_formal_continuation": False,
        "formal_training_restart_required": True,
        "private_holdout_evaluated": False,
        "complete": True,
    }
    calibration._write_immutable_json(root / trial["result_path"], result)
    return result


def run_next(
    *,
    plan_path: str | Path,
    project_root: str | Path,
    executor: CommandExecutor = _subprocess_executor,
) -> dict[str, Any]:
    """执行下一个缺失 baseline 或 trial，已完成项只验证不覆盖。"""

    plan, plan_digest = _load_plan(plan_path)
    root = Path(plan["output_root"])
    tasks = [
        *(('baseline', item) for item in plan["baselines"]),
        *(('trial', item) for item in plan["trials"]),
    ]
    missing_seen = False
    completed = []
    next_task: tuple[str, Mapping[str, Any]] | None = None
    for kind, item in tasks:
        result_path = root / item["result_path"]
        present = result_path.exists() or result_path.with_suffix(".sha256").exists()
        identifier = item.get("observation_id", item.get("trial_id"))
        if present:
            if missing_seen:
                raise ValueError("LR 校准结果存在越序观察项")
            pipeline = (
                calibration.BASELINE_PIPELINE
                if kind == "baseline"
                else calibration.TRIAL_PIPELINE
            )
            result, _ = calibration._load_result(result_path, pipeline)
            if result.get("plan_sha256") != plan_digest:
                raise ValueError("LR 校准结果 plan 身份不一致")
            completed.append(identifier)
        elif next_task is None:
            missing_seen = True
            next_task = (kind, item)

    if next_task is None:
        return {"complete": True, "observation": None, "completed": completed}
    kind, item = next_task
    project = Path(project_root).resolve()
    if not project.is_dir():
        raise FileNotFoundError(f"MiniMind 项目目录不存在: {project}")
    if kind == "baseline":
        result = _run_baseline(
            plan=plan,
            plan_digest=plan_digest,
            item=item,
            project_root=project,
            executor=executor,
        )
        identifier = result["observation_id"]
    else:
        result = _run_trial(
            plan=plan,
            plan_digest=plan_digest,
            trial=item,
            project_root=project,
            executor=executor,
        )
        identifier = result["trial_id"]
    return {
        "complete": len(completed) + 1 == len(tasks),
        "observation": identifier,
        "completed": [*completed, identifier],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="执行基础法律 SFT 共同 LR 校准")
    parser.add_argument("--plan", required=True)
    parser.add_argument(
        "--project-root", default="/root/autodl-tmp/minimind"
    )
    args = parser.parse_args()
    try:
        status = run_next(plan_path=args.plan, project_root=args.project_root)
        print(
            "SFT_BASE_LR_CALIBRATION_NEXT_OK "
            f"observation={status['observation']} "
            f"complete={str(status['complete']).lower()}"
        )
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        raise SystemExit(f"[失败] {error}") from error


if __name__ == "__main__":
    main()
