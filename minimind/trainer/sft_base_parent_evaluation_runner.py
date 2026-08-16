"""逐候选执行正式基础法律 SFT 父权重比较。"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import sft_base_generation_evaluation as generation
from . import sft_base_lr_calibration as calibration
from . import sft_base_lr_calibration_runner as calibration_runner
from . import sft_base_parent_evaluation as formal
from . import sft_parent_loss_evaluation as loss_evaluation
from . import train_full_sft as training


CommandExecutor = Callable[[Sequence[str], Path], None]
WEIGHT_NAME_RE = re.compile(r"^legal-sft-(\d+)\.pth$")


def _subprocess_executor(command: Sequence[str], cwd: Path) -> None:
    subprocess.run(list(command), cwd=cwd, check=True)


def _load_plan(path: str | Path) -> tuple[dict[str, Any], str]:
    plan, digest = formal._load_verified(path, "正式基础法律父权重 plan")
    if (
        plan.get("schema_version") != "1.0"
        or plan.get("pipeline") != formal.PLAN_PIPELINE
        or plan.get("complete") is not True
        or plan.get("private_holdout_used") is not False
    ):
        raise ValueError("正式基础法律父权重 plan 无效")
    spec_identity = plan.get("spec")
    if not isinstance(spec_identity, dict):
        raise ValueError("正式 plan 缺少 spec 身份")
    spec, spec_digest = formal._load_verified(
        spec_identity.get("path", ""), "正式基础法律父权重 spec"
    )
    if (
        spec_digest != spec_identity.get("sha256")
        or spec.get("pipeline") != formal.SPEC_PIPELINE
        or spec.get("complete") is not True
    ):
        raise ValueError("正式 plan 与 spec 身份不一致")
    decision_identity = plan.get("calibration", {}).get("decision")
    if not isinstance(decision_identity, dict):
        raise ValueError("正式 plan 缺少共同 LR decision 身份")
    decision, decision_digest = formal._load_verified(
        decision_identity.get("path", ""), "基础法律共同 LR decision"
    )
    if (
        decision_digest != decision_identity.get("sha256")
        or decision.get("pipeline") != calibration.DECISION_PIPELINE
        or decision.get("decision", {}).get("selected_peak_lr")
        != plan.get("training", {}).get("peak_lr")
        or decision.get("complete") is not True
    ):
        raise ValueError("正式 plan 与共同 LR decision 身份不一致")
    for name, identity in plan.get("inputs", {}).items():
        calibration._verify_identity(identity, f"正式比较输入 {name}")
    tokenizer_identity = plan.get("tokenizer")
    if not isinstance(tokenizer_identity, dict):
        raise ValueError("正式 plan 缺少 Tokenizer 身份")
    tokenizer = training._load_tokenizer(tokenizer_identity.get("path", ""))
    if training._tokenizer_identity(tokenizer, tokenizer_identity["path"]) != tokenizer_identity:
        raise ValueError("正式比较 Tokenizer 身份发生漂移")
    policy = plan.get("formal_training_policy", {})
    if (
        policy.get("continuous_single_run_per_candidate") is not True
        or policy.get("must_start_from_original_cpt_model_only") is not True
        or policy.get("same_candidate_resume_only") is not True
        or policy.get("intermediate_model_only_continuation_forbidden") is not True
        or policy.get("calibration_checkpoint_continuation_forbidden") is not True
        or policy.get("all_candidates_must_reach_base_100") is not True
    ):
        raise ValueError("正式 plan 缺少连续训练或原始父权重约束")
    if [item.get("candidate_role") for item in plan.get("candidates", [])] != list(
        formal.CANDIDATE_ROLES
    ):
        raise ValueError("正式 plan 必须按冻结顺序包含五个候选")
    for item in plan["candidates"]:
        calibration._verify_identity(item["parent_weights"], "正式原始 CPT 父权重")
        if (
            item.get("resume_scope") != "same_candidate_formal_run_only"
            or item.get("eligible_for_formal_base_100") is not True
        ):
            raise ValueError("正式候选缺少恢复或 base_100 资格约束")
    return plan, digest


def _training_command(
    plan: Mapping[str, Any], candidate: Mapping[str, Any], resume: bool
) -> list[str]:
    root = Path(plan["output_root"])
    config = plan["training"]
    thresholds = config["observation_thresholds"]
    swanlab = plan["tracking"]["swanlab"]
    command = [
        sys.executable,
        "-m",
        "trainer.train_full_sft",
        "--run_name",
        candidate["candidate_id"],
        "--run_dir",
        str(root / candidate["run_directory"]),
        "--checkpoint_dir",
        str(root / candidate["checkpoint_directory"]),
        "--data_manifest",
        plan["inputs"]["data_manifest"]["path"],
        "--evaluation_manifest",
        plan["inputs"]["evaluation_manifest"]["path"],
        "--tokenizer_path",
        plan["tokenizer"]["path"],
        "--parent_weights",
        candidate["parent_weights"]["path"],
        "--parent_sha256",
        candidate["parent_weights"]["sha256"],
        "--peak_lr",
        str(config["peak_lr"]),
        "--stop_assistant_tokens",
        str(config["stop_assistant_tokens"]),
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
        "--fixed_full_tokens",
        f"{thresholds['base_25']},{thresholds['base_50']}",
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
        str(config["seed"]),
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
    plan: Mapping[str, Any], weights: Mapping[str, Any], suites: Sequence[str], output: Path
) -> list[str]:
    command = [sys.executable, "-m", "trainer.sft_parent_loss_evaluation"]
    for suite in suites:
        command.extend(("--suite", suite))
    command.extend(
        (
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
        )
    )
    return command


def _generation_command(
    plan: Mapping[str, Any], weights: Mapping[str, Any], output: Path
) -> list[str]:
    return calibration_runner._generation_command(plan, weights, output)


def _load_loss_report(
    path: Path, suites: Sequence[str], weights_sha256: str
) -> dict[str, Any]:
    report, _ = formal._load_verified(path, "正式观察点 loss 报告")
    if (
        report.get("pipeline") != loss_evaluation.PIPELINE
        or list(report.get("suites", {})) != list(suites)
        or report.get("inputs", {}).get("weights", {}).get("sha256")
        != weights_sha256
        or report.get("complete") is not True
    ):
        raise ValueError("正式观察点 loss 报告身份无效")
    return report


def _load_generation_report(path: Path, weights_sha256: str) -> dict[str, Any]:
    report, _ = formal._load_verified(path, "正式观察点固定生成报告")
    if (
        report.get("pipeline") != generation.REPORT_PIPELINE
        or report.get("inputs", {}).get("weights", {}).get("sha256")
        != weights_sha256
        or report.get("complete") is not True
    ):
        raise ValueError("正式观察点固定生成报告身份无效")
    return report


def _run_observation(
    *,
    plan: Mapping[str, Any],
    plan_digest: str,
    candidate: Mapping[str, Any],
    definition: Mapping[str, Any],
    weights: Mapping[str, Any],
    actual_assistant_tokens: int,
    project_root: Path,
    executor: CommandExecutor,
) -> dict[str, Any]:
    root = Path(plan["output_root"])
    result_path = root / definition["result_path"]
    if result_path.exists() or result_path.with_suffix(".sha256").exists():
        result, _ = formal._load_verified(result_path, "正式观察点结果")
        if (
            result.get("pipeline") != formal.OBSERVATION_PIPELINE
            or result.get("plan_sha256") != plan_digest
            or result.get("candidate_id") != candidate["candidate_id"]
            or result.get("phase") != definition["phase"]
            or result.get("weights", {}).get("sha256") != weights["sha256"]
            or result.get("complete") is not True
        ):
            raise ValueError("已存在正式观察点结果身份无效")
        return result

    artifact_root = result_path.parent / "artifacts" / definition["phase"]
    loss_path = artifact_root / "loss.json"
    generation_path = artifact_root / "generation.json"
    suites = definition["loss_suites"]
    if not loss_path.exists() and not loss_path.with_suffix(".sha256").exists():
        executor(_loss_command(plan, weights, suites, loss_path), project_root)
    loss = _load_loss_report(loss_path, suites, weights["sha256"])
    generated = None
    if definition["fixed_generation"]:
        if (
            not generation_path.exists()
            and not generation_path.with_suffix(".sha256").exists()
        ):
            executor(_generation_command(plan, weights, generation_path), project_root)
        generated = _load_generation_report(generation_path, weights["sha256"])

    metrics = {
        suite: loss["suites"][suite]["metrics"] for suite in suites
    }
    if generated is not None:
        metrics["fixed_generation"] = generated["metrics"]
    result = {
        "schema_version": "1.0",
        "pipeline": formal.OBSERVATION_PIPELINE,
        "plan_sha256": plan_digest,
        "candidate_id": candidate["candidate_id"],
        "candidate_role": candidate["candidate_role"],
        "phase": definition["phase"],
        "requested_assistant_tokens": definition["requested_assistant_tokens"],
        "actual_assistant_tokens": actual_assistant_tokens,
        "weights": weights,
        "metrics": metrics,
        "fixed_generation_evaluated": generated is not None,
        "private_holdout_evaluated": False,
        "complete": True,
    }
    calibration._write_immutable_json(result_path, result)
    return result


def _completed_formal_training(
    metrics_path: Path, config: Mapping[str, Any]
) -> dict[str, int] | None:
    records = calibration_runner._read_metrics(metrics_path)
    completions = [record for record in records if record.get("type") == "run_complete"]
    if not completions:
        return None
    completed = completions[-1]
    expected_tokens = config["stop_assistant_tokens"]
    expected_sequences = config["train_records"]
    if (
        completed.get("completed_assistant_tokens") != expected_tokens
        or completed.get("completed_sequences") != expected_sequences
        or completed.get("epoch") != 1
        or completed.get("next_sequence_position") != 0
    ):
        raise ValueError("正式训练没有精确闭合完整单遍 train")
    return {
        "completed_assistant_tokens": expected_tokens,
        "completed_sequences": expected_sequences,
        "epoch": 1,
        "next_sequence_position": 0,
    }


def _discover_observation_weights(
    checkpoint_dir: Path, thresholds: Mapping[str, int]
) -> dict[str, dict[str, Any]]:
    weights_root = checkpoint_dir / "weights"
    found = []
    if weights_root.is_dir():
        for path in weights_root.iterdir():
            match = WEIGHT_NAME_RE.fullmatch(path.name)
            if match and path.is_file():
                found.append((int(match.group(1)), path))
    found.sort()
    selected: dict[str, dict[str, Any]] = {}
    previous_actual = 0
    for phase in ("base_25", "base_50", "base_100"):
        requested = thresholds[phase]
        eligible = [item for item in found if item[0] >= requested and item[0] > previous_actual]
        if not eligible:
            raise FileNotFoundError(f"缺少正式观察权重: {phase}")
        actual, path = eligible[0]
        if phase == "base_100" and actual != requested:
            raise ValueError("base_100 权重没有精确闭合完整单遍")
        selected[phase] = {
            **calibration._identity(path),
            "requested_assistant_tokens": requested,
            "actual_assistant_tokens": actual,
        }
        previous_actual = actual
    return selected


def _run_candidate(
    *,
    plan: Mapping[str, Any],
    plan_digest: str,
    candidate: Mapping[str, Any],
    project_root: Path,
    executor: CommandExecutor,
) -> dict[str, Any]:
    if (
        candidate.get("resume_scope") != "same_candidate_formal_run_only"
        or candidate.get("eligible_for_formal_base_100") is not True
    ):
        raise ValueError("正式候选恢复或 base_100 约束无效")
    parent = candidate["parent_weights"]
    calibration._verify_identity(parent, "正式原始 CPT 父权重")
    definitions = {item["phase"]: item for item in candidate["observations"]}
    observations = [
        _run_observation(
            plan=plan,
            plan_digest=plan_digest,
            candidate=candidate,
            definition=definitions["parent"],
            weights=parent,
            actual_assistant_tokens=0,
            project_root=project_root,
            executor=executor,
        )
    ]

    root = Path(plan["output_root"])
    run_dir = root / candidate["run_directory"]
    checkpoint_dir = root / candidate["checkpoint_directory"]
    metrics_path = run_dir / "metrics.jsonl"
    completion = _completed_formal_training(metrics_path, plan["training"])
    if completion is None:
        resume_path = checkpoint_dir / "resume" / "latest.pt"
        resume = resume_path.is_file()
        if not resume and (run_dir.exists() or checkpoint_dir.exists()):
            raise RuntimeError("正式候选有残留目录但没有同候选安全恢复点")
        executor(_training_command(plan, candidate, resume), project_root)
        completion = _completed_formal_training(metrics_path, plan["training"])
    if completion is None:
        raise RuntimeError("正式训练命令结束但没有完整单遍 run_complete")

    milestone_weights = _discover_observation_weights(
        checkpoint_dir, plan["training"]["observation_thresholds"]
    )
    for phase in ("base_25", "base_50", "base_100"):
        weights = milestone_weights[phase]
        observations.append(
            _run_observation(
                plan=plan,
                plan_digest=plan_digest,
                candidate=candidate,
                definition=definitions[phase],
                weights=weights,
                actual_assistant_tokens=weights["actual_assistant_tokens"],
                project_root=project_root,
                executor=executor,
            )
        )

    result = {
        "schema_version": "1.0",
        "pipeline": formal.CANDIDATE_PIPELINE,
        "plan_sha256": plan_digest,
        "candidate_id": candidate["candidate_id"],
        "candidate_role": candidate["candidate_role"],
        "parent_weights": parent,
        "training": {
            "peak_lr": plan["training"]["peak_lr"],
            **completion,
            "stability": calibration_runner._training_stability(metrics_path),
            "continuous_single_run": True,
        },
        "observations": observations,
        "base_100_weights": milestone_weights["base_100"],
        "formal_base_100_eligible": True,
        "calibration_checkpoint_used": False,
        "private_holdout_evaluated": False,
        "complete": True,
    }
    calibration._write_immutable_json(root / candidate["result_path"], result)
    return result


def run_next(
    *,
    plan_path: str | Path,
    project_root: str | Path,
    executor: CommandExecutor = _subprocess_executor,
) -> dict[str, Any]:
    """执行下一个未完成正式候选，已完成候选只验证不覆盖。"""

    plan, plan_digest = _load_plan(plan_path)
    root = Path(plan["output_root"])
    missing_seen = False
    completed = []
    next_candidate = None
    for candidate in plan["candidates"]:
        result_path = root / candidate["result_path"]
        present = result_path.exists() or result_path.with_suffix(".sha256").exists()
        if present:
            if missing_seen:
                raise ValueError("正式父权重结果存在越序候选")
            result, _ = formal._load_verified(result_path, "正式候选结果")
            if (
                result.get("pipeline") != formal.CANDIDATE_PIPELINE
                or result.get("plan_sha256") != plan_digest
                or result.get("candidate_id") != candidate["candidate_id"]
                or result.get("complete") is not True
            ):
                raise ValueError("已完成正式候选结果身份无效")
            completed.append(candidate["candidate_id"])
        elif next_candidate is None:
            missing_seen = True
            next_candidate = candidate

    if next_candidate is None:
        return {"complete": True, "candidate": None, "completed": completed}
    project = Path(project_root).resolve()
    if not project.is_dir():
        raise FileNotFoundError(f"MiniMind 项目目录不存在: {project}")
    result = _run_candidate(
        plan=plan,
        plan_digest=plan_digest,
        candidate=next_candidate,
        project_root=project,
        executor=executor,
    )
    completed.append(result["candidate_id"])
    return {
        "complete": len(completed) == len(plan["candidates"]),
        "candidate": result["candidate_id"],
        "completed": completed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="执行正式基础法律 SFT 五候选比较")
    parser.add_argument("--plan", required=True)
    parser.add_argument("--project-root", default="/root/autodl-tmp/minimind")
    args = parser.parse_args()
    try:
        status = run_next(plan_path=args.plan, project_root=args.project_root)
        print(
            "SFT_BASE_PARENT_FORMAL_NEXT_OK "
            f"candidate={status['candidate']} "
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
