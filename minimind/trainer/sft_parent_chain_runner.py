"""按 v4 plan 逐观察点执行一条父权重连续训练链。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ == "trainer":
    repository_root = str(Path(__file__).resolve().parents[2])
    if repository_root not in sys.path:
        sys.path.append(repository_root)

from . import sft_parent_evaluation as parent_evaluation
from . import sft_parent_loss_evaluation as loss_evaluation
from . import sft_parent_rag_evaluation as rag_evaluation
from . import train_rag_sft as rag_training


PIPELINE = "legal_sft_parent_chain_execution_v1"
MODEL_ARTIFACT_PIPELINE = "legal_sft_parent_model_observation_v1"
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CommandExecutor = Callable[[Sequence[str], Path], None]


def _file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"文件不存在: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": parent_evaluation._sha256_file(resolved),
    }


def _verify_plan_runtime_assets(plan: Mapping[str, Any]) -> None:
    """重新核对 plan 直接引用的文件、Tokenizer 和通用验证 shard。"""

    verified: dict[Path, str] = {}
    for name, identity in plan["inputs"].items():
        path = Path(identity["path"]).resolve()
        digest = verified.get(path)
        if digest is None:
            digest = _file_identity(path)["sha256"]
            verified[path] = digest
        if digest != identity["sha256"]:
            raise ValueError(f"plan 共享输入发生漂移: {name}")

    general_path = Path(plan["inputs"]["general_validation"]["path"])
    general_payload, general_digest = parent_evaluation._load_verified_json(
        general_path, "通用验证 manifest"
    )
    general_identity = parent_evaluation._general_validation_identity(
        general_path.resolve(), general_payload, general_digest
    )
    if general_identity != plan["inputs"]["general_validation"]:
        raise ValueError("通用验证身份与 plan 不一致")

    tokenizer = parent_evaluation._tokenizer_identity(plan["tokenizer"]["path"])
    if tokenizer != plan["tokenizer"]:
        raise ValueError("Tokenizer 身份与 plan 不一致")


def _select_chain(plan: Mapping[str, Any], chain_id: str) -> dict[str, Any]:
    matches = [chain for chain in plan["chains"] if chain.get("chain_id") == chain_id]
    if len(matches) != 1:
        raise ValueError(f"plan 中不存在唯一连续链: {chain_id}")
    return matches[0]


def prepare_execution(
    *,
    plan_path: str | Path,
    chain_id: str,
    output_root: str | Path,
    rag_fixed_manifest: str | Path,
    rag_candidate: str | Path,
    pair_records: str | Path,
    article_index: str | Path,
    project_root: str | Path = DEFAULT_PROJECT_ROOT,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """复验本机资产并发布一条连续链的不可变运行绑定。"""

    plan, plan_digest = parent_evaluation._load_plan(plan_path)
    _verify_plan_runtime_assets(plan)
    chain = _select_chain(plan, chain_id)
    candidate = next(
        item for item in plan["candidates"] if item["role"] == chain["candidate_role"]
    )
    if _file_identity(candidate["path"])["sha256"] != candidate["sha256"]:
        raise ValueError("当前连续链父权重身份发生漂移")

    evaluation_manifest = plan["inputs"]["evaluation_exclusions"]["path"]
    release_manifest = plan["inputs"]["rag_sft_data"]["path"]
    _, _, rag_identity = rag_training._load_rag_training_inputs(
        release_manifest,
        rag_fixed_manifest,
        rag_candidate,
        evaluation_manifest,
    )
    pair_manifest = plan["inputs"]["rag_pair_evaluation"]["path"]
    tokenizer_path = plan["tokenizer"]["path"]
    tokenizer = rag_training.base_entry._load_tokenizer(tokenizer_path)
    _, _, pair_payload, _ = rag_evaluation._load_records(
        pair_manifest,
        pair_records,
        article_index,
        tokenizer,
        tokenizer_path,
    )
    if len(pair_payload) != plan["inputs"]["rag_pair_evaluation"]["records"][
        "eligible_model_cases"
    ] * 2:
        raise ValueError("RAG 成对评估记录数与 plan 不一致")

    project_path = Path(project_root).resolve()
    if not (project_path / "trainer").is_dir():
        raise FileNotFoundError(f"MiniMind 项目根目录无效: {project_path}")
    root = Path(output_root).resolve()
    manifest_path = root / "chains" / chain_id / "chain-execution-manifest.json"
    payload = {
        "schema_version": "1.0",
        "pipeline": PIPELINE,
        "plan": {
            "path": str(Path(plan_path).resolve()),
            "sha256": plan_digest,
            "shared_invariants_sha256": plan["shared_invariants_sha256"],
        },
        "chain": chain,
        "candidate": candidate,
        "bindings": {
            "rag_release_manifest": plan["inputs"]["rag_sft_data"],
            "rag_fixed_manifest": _file_identity(rag_fixed_manifest),
            "rag_candidate": _file_identity(rag_candidate),
            "evaluation_manifest": plan["inputs"]["evaluation_exclusions"],
            "pair_manifest": plan["inputs"]["rag_pair_evaluation"],
            "pair_records": _file_identity(pair_records),
            "article_index": _file_identity(article_index),
        },
        "runtime": {
            "project_root": str(project_path),
            "python_executable": str(Path(sys.executable).resolve()),
            "device": device,
            "train_num_workers": 8,
            "eval_num_workers": 4,
            "eval_batch_size": 64,
            "eval_oom_fallback_batch_size": 32,
            "swanlab": False,
            "omp_num_threads": 8,
        },
        "output_root": str(root),
        "complete": True,
    }
    if rag_identity["fixed_manifest_sha256"] != payload["bindings"][
        "rag_fixed_manifest"
    ]["sha256"] or rag_identity["candidate_sha256"] != payload["bindings"][
        "rag_candidate"
    ]["sha256"]:
        raise ValueError("RAG-SFT 运行绑定与正式 release 身份不一致")
    parent_evaluation._write_immutable_json(manifest_path, payload)
    return payload


def _load_execution(path: str | Path) -> tuple[dict[str, Any], str]:
    payload, digest = parent_evaluation._load_verified_json(path, "连续链运行绑定")
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("pipeline") != PIPELINE
        or payload.get("complete") is not True
    ):
        raise ValueError("连续链运行绑定版本或状态无效")
    plan, plan_digest = parent_evaluation._load_plan(payload["plan"]["path"])
    if (
        plan_digest != payload["plan"]["sha256"]
        or plan["shared_invariants_sha256"]
        != payload["plan"]["shared_invariants_sha256"]
        or _select_chain(plan, payload["chain"]["chain_id"]) != payload["chain"]
    ):
        raise ValueError("连续链运行绑定与 v4 plan 身份不一致")
    return payload, digest


def _subprocess_executor(command: Sequence[str], cwd: Path) -> None:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "8"
    repository_root = str(cwd.resolve().parent)
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        repository_root
        if not current_pythonpath
        else os.pathsep.join((repository_root, current_pythonpath))
    )
    subprocess.run(list(command), cwd=cwd, env=environment, check=True)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError
            records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"训练指标 JSONL 无效: {path}") from error
    return records


def _completed_training_tokens(metrics_path: Path, target: int) -> int | None:
    records = _read_jsonl(metrics_path)
    starts = [
        index
        for index, record in enumerate(records)
        if record.get("type") in {"run_start", "run_resume"}
    ]
    if not starts:
        return None
    start_index = starts[-1]
    controls = records[start_index].get("controls")
    if not isinstance(controls, dict) or controls.get("stop_assistant_tokens") != target:
        return None
    completions = [
        record
        for record in records[start_index + 1 :]
        if record.get("type") == "run_complete"
    ]
    if not completions:
        return None
    completed = completions[-1].get("completed_assistant_tokens")
    if isinstance(completed, bool) or not isinstance(completed, int):
        raise ValueError("训练完成记录缺少有效 assistant-token 进度")
    if completed <= 0 or completed > target:
        raise ValueError("训练完成进度超出当前观察点预算")
    return completed


def _training_paths(root: Path, chain_id: str, stage: str) -> dict[str, Path]:
    base = root / "chains" / chain_id / "training" / stage
    return {
        "run": base / "run",
        "checkpoints": base / "checkpoints",
        "metrics": base / "run" / "metrics.jsonl",
        "resume": base / "checkpoints" / "resume" / "latest.pt",
        "weights": base / "checkpoints" / "weights",
    }


def _base_training_command(
    execution: Mapping[str, Any], plan: Mapping[str, Any], target: int, resume: bool
) -> list[str]:
    chain = execution["chain"]
    paths = _training_paths(
        Path(execution["output_root"]), chain["chain_id"], "base_sft"
    )
    config = plan["training"]["base_sft"]
    command = [
        execution["runtime"]["python_executable"],
        "-m",
        "trainer.train_full_sft",
        "--run_name",
        f"{chain['chain_id']}-base-sft",
        "--run_dir",
        str(paths["run"]),
        "--checkpoint_dir",
        str(paths["checkpoints"]),
        "--data_manifest",
        plan["inputs"]["base_sft_data"]["path"],
        "--evaluation_manifest",
        plan["inputs"]["evaluation_exclusions"]["path"],
        "--tokenizer_path",
        plan["tokenizer"]["path"],
        "--parent_weights",
        execution["candidate"]["path"],
        "--parent_sha256",
        execution["candidate"]["sha256"],
        "--peak_lr",
        str(config["peak_lr"]),
        "--stop_assistant_tokens",
        str(target),
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
        "250000,500000",
        "--fixed_full_tokens",
        "1000000",
        "--eval_batch_size",
        str(execution["runtime"]["eval_batch_size"]),
        "--device",
        execution["runtime"]["device"],
        "--dtype",
        config["dtype"],
        "--num_workers",
        str(execution["runtime"]["train_num_workers"]),
        "--eval_num_workers",
        str(execution["runtime"]["eval_num_workers"]),
        "--seed",
        str(chain["seed"]),
        "--no-swanlab",
    ]
    if resume:
        command.append("--resume")
    return command


def _rag_training_command(
    execution: Mapping[str, Any], plan: Mapping[str, Any], target: int, resume: bool
) -> list[str]:
    chain = execution["chain"]
    paths = _training_paths(
        Path(execution["output_root"]), chain["chain_id"], "rag_sft"
    )
    config = plan["training"]["rag_sft"]
    base_model = _load_model_artifact(execution, plan, "base_1m")
    command = [
        execution["runtime"]["python_executable"],
        "-m",
        "trainer.train_rag_sft",
        "--run_name",
        f"{chain['chain_id']}-rag-sft",
        "--run_dir",
        str(paths["run"]),
        "--checkpoint_dir",
        str(paths["checkpoints"]),
        "--release_manifest",
        execution["bindings"]["rag_release_manifest"]["path"],
        "--fixed_manifest",
        execution["bindings"]["rag_fixed_manifest"]["path"],
        "--candidate_path",
        execution["bindings"]["rag_candidate"]["path"],
        "--evaluation_manifest",
        execution["bindings"]["evaluation_manifest"]["path"],
        "--tokenizer_path",
        plan["tokenizer"]["path"],
        "--parent_weights",
        base_model["weight"]["path"],
        "--parent_sha256",
        base_model["weight"]["sha256"],
        "--peak_lr",
        str(config["peak_lr"]),
        "--stop_assistant_tokens",
        str(target),
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
        "--checkpoint_interval_tokens",
        str(config["assistant_tokens_per_epoch"]),
        "--device",
        execution["runtime"]["device"],
        "--dtype",
        config["dtype"],
        "--num_workers",
        str(execution["runtime"]["train_num_workers"]),
        "--seed",
        str(chain["seed"]),
        "--no-swanlab",
    ]
    if resume:
        command.append("--resume")
    return command


def _model_artifact_path(execution: Mapping[str, Any], phase: str) -> Path:
    return (
        Path(execution["output_root"])
        / "chains"
        / execution["chain"]["chain_id"]
        / "artifacts"
        / f"{phase}-model.json"
    )


def _load_model_artifact(
    execution: Mapping[str, Any], plan: Mapping[str, Any], phase: str
) -> dict[str, Any]:
    path = _model_artifact_path(execution, phase)
    payload, _ = parent_evaluation._load_verified_json(path, "观察点模型产物")
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("pipeline") != MODEL_ARTIFACT_PIPELINE
        or payload.get("plan_sha256") != execution["plan"]["sha256"]
        or payload.get("chain_id") != execution["chain"]["chain_id"]
        or payload.get("phase") != phase
        or payload.get("complete") is not True
    ):
        raise ValueError("观察点模型产物身份无效")
    weight = payload.get("weight")
    if not isinstance(weight, dict) or _file_identity(weight["path"]) != weight:
        raise ValueError("观察点 model-only 权重身份发生漂移")
    expected = next(
        item for item in execution["chain"]["observations"] if item["phase"] == phase
    )
    if payload["requested_stop_assistant_tokens"] != expected[
        "stop_assistant_tokens"
    ]:
        raise ValueError("观察点模型产物训练预算与 plan 不一致")
    return payload


def _publish_model_artifact(
    execution: Mapping[str, Any],
    plan: Mapping[str, Any],
    phase: str,
    executor: CommandExecutor,
) -> dict[str, Any]:
    artifact_path = _model_artifact_path(execution, phase)
    if artifact_path.exists() or artifact_path.with_suffix(".sha256").exists():
        return _load_model_artifact(execution, plan, phase)
    observation = next(
        item for item in execution["chain"]["observations"] if item["phase"] == phase
    )
    target = observation["stop_assistant_tokens"]
    if phase == "baseline":
        completed = 0
        weight = _file_identity(execution["candidate"]["path"])
    else:
        stage = "base_sft" if phase.startswith("base_") else "rag_sft"
        paths = _training_paths(
            Path(execution["output_root"]), execution["chain"]["chain_id"], stage
        )
        completed = _completed_training_tokens(paths["metrics"], target)
        if completed is None:
            resume = paths["resume"].is_file()
            if not resume and (paths["run"].exists() or paths["checkpoints"].exists()):
                raise RuntimeError(
                    f"{phase} 存在不完整目录但没有安全恢复点，禁止覆盖"
                )
            command = (
                _base_training_command(execution, plan, target, resume)
                if stage == "base_sft"
                else _rag_training_command(execution, plan, target, resume)
            )
            executor(command, Path(execution["runtime"]["project_root"]))
            completed = _completed_training_tokens(paths["metrics"], target)
        if completed is None:
            raise RuntimeError(f"{phase} 训练命令结束但没有匹配的完成记录")
        prefix = "legal-sft" if stage == "base_sft" else "rag-sft"
        source_weight = paths["weights"] / f"{prefix}-{completed}.pth"
        if not source_weight.is_file():
            raise FileNotFoundError(f"{phase} 未导出预期 model-only 权重: {source_weight}")
        planned_relative = observation["model_only_path"]
        if not isinstance(planned_relative, str):
            raise ValueError(f"{phase} plan 缺少 model_only_path")
        planned_weight = Path(execution["output_root"]) / planned_relative
        planned_weight.parent.mkdir(parents=True, exist_ok=True)
        source_identity = _file_identity(source_weight)
        if planned_weight.exists():
            if _file_identity(planned_weight)["sha256"] != source_identity["sha256"]:
                raise ValueError(f"{phase} 计划权重路径已被其他内容占用")
        else:
            os.link(source_weight, planned_weight)
        weight = _file_identity(planned_weight)
    payload = {
        "schema_version": "1.0",
        "pipeline": MODEL_ARTIFACT_PIPELINE,
        "plan_sha256": execution["plan"]["sha256"],
        "chain_id": execution["chain"]["chain_id"],
        "phase": phase,
        "requested_stop_assistant_tokens": target,
        "completed_assistant_tokens": completed,
        "weight": weight,
        "complete": True,
    }
    parent_evaluation._write_immutable_json(artifact_path, payload)
    return payload


def _load_loss_report(
    path: Path, expected_suites: Sequence[str], weights_sha256: str
) -> dict[str, Any]:
    payload, _ = parent_evaluation._load_verified_json(path, "loss 评估报告")
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("pipeline") != loss_evaluation.PIPELINE
        or payload.get("complete") is not True
        or payload.get("inputs", {}).get("weights", {}).get("sha256")
        != weights_sha256
        or set(payload.get("suites", {})) != set(expected_suites)
    ):
        raise ValueError("loss 评估报告身份或 suites 无效")
    return payload


def _evaluate_loss(
    execution: Mapping[str, Any],
    plan: Mapping[str, Any],
    phase: str,
    model_artifact: Mapping[str, Any],
    executor: CommandExecutor,
) -> dict[str, Any]:
    suites = [
        suite
        for suite in plan["evaluation"]["phase_suites"][phase]
        if suite in parent_evaluation.LOSS_SUITES
    ]
    output = (
        Path(execution["output_root"])
        / "chains"
        / execution["chain"]["chain_id"]
        / "artifacts"
        / f"{phase}-loss.json"
    )
    if output.exists() or output.with_suffix(".sha256").exists():
        return _load_loss_report(output, suites, model_artifact["weight"]["sha256"])
    command = [
        execution["runtime"]["python_executable"],
        "-m",
        "trainer.sft_parent_loss_evaluation",
    ]
    for suite in suites:
        command.extend(["--suite", suite])
    command.extend(
        [
            "--general-manifest",
            plan["inputs"]["general_validation"]["path"],
            "--legal-manifest",
            plan["inputs"]["base_sft_data"]["path"],
            "--tokenizer-path",
            plan["tokenizer"]["path"],
            "--weights",
            model_artifact["weight"]["path"],
            "--weights-sha256",
            model_artifact["weight"]["sha256"],
            "--output",
            str(output),
            "--device",
            execution["runtime"]["device"],
            "--eval-batch-size",
            str(execution["runtime"]["eval_batch_size"]),
            "--eval-num-workers",
            str(execution["runtime"]["eval_num_workers"]),
        ]
    )
    executor(command, Path(execution["runtime"]["project_root"]))
    return _load_loss_report(output, suites, model_artifact["weight"]["sha256"])


def _load_rag_report(path: Path, weights_sha256: str) -> dict[str, Any]:
    payload, _ = parent_evaluation._load_verified_json(path, "RAG 评估报告")
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("pipeline") != rag_evaluation.PIPELINE
        or payload.get("complete") is not True
        or payload.get("inputs", {}).get("weights_sha256") != weights_sha256
        or set(payload.get("metrics", {}))
        != set(parent_evaluation.RAG_METRICS)
    ):
        raise ValueError("RAG 评估报告身份或指标 schema 无效")
    return payload


def _evaluate_rag(
    execution: Mapping[str, Any],
    plan: Mapping[str, Any],
    phase: str,
    model_artifact: Mapping[str, Any],
    executor: CommandExecutor,
) -> dict[str, Any] | None:
    suites = plan["evaluation"]["phase_suites"][phase]
    if not any(suite in parent_evaluation.RAG_SUITES for suite in suites):
        return None
    output = (
        Path(execution["output_root"])
        / "chains"
        / execution["chain"]["chain_id"]
        / "artifacts"
        / f"{phase}-rag.json"
    )
    if output.exists() or output.with_suffix(".sha256").exists():
        return _load_rag_report(output, model_artifact["weight"]["sha256"])
    command = [
        execution["runtime"]["python_executable"],
        "-m",
        "trainer.sft_parent_rag_evaluation",
        "evaluate",
        "--pair-manifest",
        execution["bindings"]["pair_manifest"]["path"],
        "--pair-records",
        execution["bindings"]["pair_records"]["path"],
        "--article-index",
        execution["bindings"]["article_index"]["path"],
        "--tokenizer-path",
        plan["tokenizer"]["path"],
        "--weights",
        model_artifact["weight"]["path"],
        "--weights-sha256",
        model_artifact["weight"]["sha256"],
        "--output",
        str(output),
        "--device",
        execution["runtime"]["device"],
    ]
    executor(command, Path(execution["runtime"]["project_root"]))
    return _load_rag_report(output, model_artifact["weight"]["sha256"])


def _result_path(execution: Mapping[str, Any], observation: Mapping[str, Any]) -> Path:
    return Path(execution["output_root"]) / observation["result_path"]


def _completed_phases(
    execution: Mapping[str, Any], plan: Mapping[str, Any]
) -> list[str]:
    completed = []
    missing_seen = False
    for observation in execution["chain"]["observations"]:
        path = _result_path(execution, observation)
        present = path.exists() or path.with_suffix(".sha256").exists()
        if present:
            if missing_seen:
                raise ValueError("连续链存在越序 trial 结果")
            parent_evaluation.validate_trial_result(
                plan, execution["plan"]["sha256"], path
            )
            completed.append(observation["phase"])
        else:
            missing_seen = True
    return completed


def run_next(
    execution_manifest: str | Path,
    *,
    executor: CommandExecutor = _subprocess_executor,
) -> dict[str, Any]:
    """执行一条连续链的下一个未完成观察点。"""

    execution, _ = _load_execution(execution_manifest)
    plan, _ = parent_evaluation._load_plan(execution["plan"]["path"])
    completed = _completed_phases(execution, plan)
    if len(completed) == len(execution["chain"]["observations"]):
        return {"complete": True, "completed_phases": completed, "phase": None}
    observation = execution["chain"]["observations"][len(completed)]
    phase = observation["phase"]
    model_artifact = _publish_model_artifact(
        execution, plan, phase, executor
    )
    loss_report = _evaluate_loss(
        execution, plan, phase, model_artifact, executor
    )
    rag_report = _evaluate_rag(
        execution, plan, phase, model_artifact, executor
    )
    suites = {}
    for suite in plan["evaluation"]["phase_suites"][phase]:
        if suite in parent_evaluation.LOSS_SUITES:
            report = loss_report["suites"][suite]
            suites[suite] = {
                "sample_count": report["sample_count"],
                "metrics": report["metrics"],
            }
        else:
            if rag_report is None:
                raise RuntimeError(f"{phase} 缺少 RAG 评估报告")
            suites[suite] = {
                "sample_count": plan["inputs"]["rag_pair_evaluation"]["records"][
                    "paired_queries"
                ],
                "metrics": rag_report["metrics"],
            }
    trial = next(
        item
        for item in plan["trials"]
        if item["trial_id"] == observation["trial_id"]
    )
    result = {
        "schema_version": "1.0",
        "pipeline": parent_evaluation.TRIAL_RESULT_PIPELINE,
        "trial_id": trial["trial_id"],
        "plan_sha256": execution["plan"]["sha256"],
        "shared_invariants_sha256": plan["shared_invariants_sha256"],
        "candidate_role": trial["candidate_role"],
        "parent_sha256": trial["parent_sha256"],
        "seed": trial["seed"],
        "phase": trial["phase"],
        "provenance": {"kind": "native"},
        "completed": True,
        "suites": suites,
    }
    result_path = _result_path(execution, observation)
    parent_evaluation._write_immutable_json(result_path, result)
    parent_evaluation.validate_trial_result(
        plan, execution["plan"]["sha256"], result_path
    )
    return {
        "complete": len(completed) + 1 == len(execution["chain"]["observations"]),
        "completed_phases": [*completed, phase],
        "phase": phase,
        "result_path": str(result_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="执行父权重比较的一条连续训练链")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="发布云端运行绑定")
    prepare.add_argument("--plan", required=True)
    prepare.add_argument("--chain-id", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--rag-fixed-manifest", required=True)
    prepare.add_argument("--rag-candidate", required=True)
    prepare.add_argument("--pair-records", required=True)
    prepare.add_argument("--article-index", required=True)
    prepare.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    prepare.add_argument("--device", default="cuda:0")
    next_parser = commands.add_parser("next", help="执行下一个未完成观察点")
    next_parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            execution = prepare_execution(
                plan_path=args.plan,
                chain_id=args.chain_id,
                output_root=args.output_root,
                rag_fixed_manifest=args.rag_fixed_manifest,
                rag_candidate=args.rag_candidate,
                pair_records=args.pair_records,
                article_index=args.article_index,
                project_root=args.project_root,
                device=args.device,
            )
            print(
                "SFT_PARENT_CHAIN_PREPARE_OK "
                f"chain={execution['chain']['chain_id']} phases=6"
            )
        else:
            status = run_next(args.manifest)
            print(
                "SFT_PARENT_CHAIN_NEXT_OK "
                f"phase={status['phase']} complete={str(status['complete']).lower()}"
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
