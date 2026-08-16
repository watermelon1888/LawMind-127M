"""编排固定 226 条 HN、改变 Clean 覆盖量的四组训练与评估。"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from ..dataset.rag_sft_v2_clean_scale import (
        GROUP_CLEAN_COUNTS,
        GROUP_HN_COUNTS,
        SUBSET_SEED,
        audit_clean_scale_candidate,
        build_clean_scale_plan,
    )
except ImportError:
    from dataset.rag_sft_v2_clean_scale import (
        GROUP_CLEAN_COUNTS,
        GROUP_HN_COUNTS,
        SUBSET_SEED,
        audit_clean_scale_candidate,
        build_clean_scale_plan,
    )
from . import rag_sft_v2_stage_f_runner as stage_f
from .train_rag_sft_v2_clean_scale import (
    EPOCHS,
    EXPORTED_EPOCHS,
    optimizer_updates_per_epoch,
)


PIPELINE = "rag_sft_v2_cpt750m_clean_scale_execution_v1"
AGGREGATE_PIPELINE = "rag_sft_v2_cpt750m_clean_scale_aggregate_v1"
GROUPS = tuple(GROUP_CLEAN_COUNTS)
EVALUATION_RECORDS = 170
PRIMARY_EVALUATION_RECORDS = 166
EVALUATION_COUNTS = {"total": 170, "legal_query": 140, "exact_lookup": 30}
EXTRAPOLATION_QUERY_IDS = ["Q033", "Q048", "Q063", "Q091"]
EXPECTED_MODELS = 1 + len(GROUPS) * len(EXPORTED_EPOCHS)
EXPECTED_ANSWERS = EXPECTED_MODELS * EVALUATION_RECORDS


def _plan_summary(plan: Any, *, micro_batch_size: int, accumulation_steps: int) -> dict[str, Any]:
    updates = optimizer_updates_per_epoch(
        plan.total_records,
        micro_batch_size=micro_batch_size,
        accumulation_steps=accumulation_steps,
    )
    return {
        "group": plan.group,
        "subset_seed": plan.subset_seed,
        "clean_count": plan.clean_count,
        "hard_negative_count": plan.hard_negative_count,
        "paired_clean_count": plan.paired_clean_count,
        "records_per_epoch": plan.total_records,
        "sequence_exposures": plan.total_records * EPOCHS,
        "optimizer_steps_per_epoch": updates,
        "optimizer_steps": updates * EPOCHS,
        "plan_sha256": plan.sha256,
    }


def prepare_execution(
    *,
    parent_weights: str | Path,
    parent_sha256: str,
    data_manifest: str | Path,
    candidate_path: str | Path,
    tokenizer_path: str | Path,
    evaluation_manifest: str | Path,
    evaluation_cases: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """验证未来 775 条正式 release 并发布不可变本地/云端执行绑定。"""

    parent_identity = stage_f._identity(parent_weights)
    if parent_identity["sha256"] != parent_sha256.lower():
        raise ValueError("cpt_750m 父权重 SHA-256 校验失败")
    parent = {
        "role": "cpt_750m",
        "path": parent_identity["path"],
        "bytes": parent_identity["bytes"],
        "sha256": parent_identity["sha256"],
    }
    data, data_sha256 = stage_f._load_data_manifest(data_manifest)
    readiness = data.get("readiness", {})
    if (
        data.get("pipeline") != "rag_sft_v2_training_release"
        or readiness.get("training_ready") is not True
        or readiness.get("real_hard_negatives_constructed") is not True
        or readiness.get("final_retrieval_identity_frozen") is not True
        or data.get("complete") is not True
    ):
        raise ValueError("Clean规模实验只接受 training-ready v2 release")
    candidate = stage_f._identity(candidate_path)
    metadata = data.get("output", {}).get("training_candidate")
    if not isinstance(metadata, dict) or any(
        candidate[field] != metadata.get(field) for field in ("bytes", "sha256")
    ):
        raise ValueError("candidate 与 data manifest 身份不一致")
    pairing = audit_clean_scale_candidate(candidate["path"])

    evaluation, evaluation_sha256 = stage_f._load_verified_json(
        evaluation_manifest, "回答评估输入 manifest"
    )
    if (
        evaluation.get("pipeline") != "rag_sft_v2_evaluation_inputs_v1"
        or evaluation.get("records") != EVALUATION_COUNTS
        or evaluation.get("evaluation_scopes", {})
        .get("rope_extrapolation_1024", {})
        .get("query_ids")
        != EXTRAPOLATION_QUERY_IDS
        or evaluation.get("complete") is not True
    ):
        raise ValueError("评估输入未闭合 170 题双口径身份")
    cases = stage_f._identity(evaluation_cases)
    cases_metadata = evaluation.get("output", {}).get("cases")
    if not isinstance(cases_metadata, dict) or any(
        cases[field] != cases_metadata.get(field) for field in ("bytes", "sha256")
    ):
        raise ValueError("evaluation cases 与 manifest 身份不一致")

    tokenizer_root = Path(tokenizer_path).resolve()
    expected_tokenizer = data.get("tokenizer")
    if not tokenizer_root.is_dir() or not isinstance(expected_tokenizer, dict):
        raise ValueError("Tokenizer 目录或 manifest 身份无效")
    tokenizer_files = {}
    for filename, expected in expected_tokenizer.get("files", {}).items():
        actual = stage_f._identity(tokenizer_root / filename)
        if any(
            actual[field] != expected.get(field) for field in ("bytes", "sha256")
        ):
            raise ValueError(f"Tokenizer 文件身份不一致: {filename}")
        tokenizer_files[filename] = actual

    micro_batch_size = 8
    accumulation_steps = 2
    seed = 42
    plans = {
        group: _plan_summary(
            build_clean_scale_plan(
                candidate["path"], group=group, subset_seed=SUBSET_SEED
            ),
            micro_batch_size=micro_batch_size,
            accumulation_steps=accumulation_steps,
        )
        for group in GROUPS
    }
    root = Path(output_root).resolve()
    models = [
        {
            "model_id": "cpt_750m/parent",
            "group": "parent",
            "phase": "parent",
            "weights": parent,
        }
    ]
    for group in GROUPS:
        for epoch in EXPORTED_EPOCHS:
            path = (
                root
                / "groups"
                / f"{group}-seed-{seed}"
                / "checkpoints"
                / "weights"
                / f"rag_epoch_{epoch}.pth"
            )
            models.append(
                {
                    "model_id": f"{group}/rag_epoch_{epoch}",
                    "group": group,
                    "phase": f"rag_epoch_{epoch}",
                    "weights_path": str(path),
                }
            )
    payload = {
        "schema_version": "1.0",
        "pipeline": PIPELINE,
        "inputs": {
            "parent": parent,
            "data_manifest": {
                "path": str(Path(data_manifest).resolve()),
                "sha256": data_sha256,
            },
            "candidate": candidate,
            "pairing_audit": pairing,
            "tokenizer": {"path": str(tokenizer_root), "files": tokenizer_files},
            "evaluation_manifest": {
                "path": str(Path(evaluation_manifest).resolve()),
                "sha256": evaluation_sha256,
            },
            "evaluation_cases": cases,
        },
        "training": {
            "groups": list(GROUPS),
            "group_clean_counts": GROUP_CLEAN_COUNTS,
            "group_hard_negative_counts": GROUP_HN_COUNTS,
            "plans": plans,
            "epochs": EPOCHS,
            "exported_epochs": list(EXPORTED_EPOCHS),
            "seed": seed,
            "followup_seeds": [43, 44],
            "peak_lr": 1.2e-5,
            "warmup_ratio": 0.1,
            "floor_ratio": 0.1,
            "micro_batch_size": micro_batch_size,
            "accumulation_steps": accumulation_steps,
            "grad_clip": 1.0,
            "dtype": "bfloat16",
            "num_workers": 4,
            "device": "cuda:0",
        },
        "evaluation_models": models,
        "expected_answers": EXPECTED_ANSWERS,
        "selection_rule": {
            "status": "deferred_until_results",
            "primary_endpoint": "同一 epoch 的 166 题 primary_768 配对指标",
            "clean_guardrail": "相对新 clean-only 的下降阈值待结果后审核",
            "followup": "seed 42 后，最好的两种混合配方追加 seed 43、44",
        },
        "output_root": str(root),
        "complete": True,
    }
    stage_f._write_immutable_json(root / "clean-scale-execution.json", payload)
    return payload


def _load_execution(path: str | Path) -> dict[str, Any]:
    payload, _ = stage_f._load_verified_json(path, "Clean规模实验执行绑定")
    training = payload.get("training", {})
    if (
        payload.get("pipeline") != PIPELINE
        or training.get("groups") != list(GROUPS)
        or training.get("epochs") != EPOCHS
        or training.get("exported_epochs") != list(EXPORTED_EPOCHS)
        or len(payload.get("evaluation_models", [])) != EXPECTED_MODELS
        or payload.get("expected_answers") != EXPECTED_ANSWERS
        or payload.get("complete") is not True
    ):
        raise ValueError("Clean规模实验执行绑定版本、epoch 或计数无效")
    return payload


def training_commands(
    execution: Mapping[str, Any], *, resume: bool = False
) -> list[list[str]]:
    root = Path(execution["output_root"])
    inputs = execution["inputs"]
    config = execution["training"]
    commands = []
    for group in config["groups"]:
        group_root = root / "groups" / f"{group}-seed-{config['seed']}"
        command = [
            sys.executable,
            "-m",
            "trainer.train_rag_sft_v2_clean_scale",
            "--run_name",
            f"cpt_750m-{group}-seed-{config['seed']}",
            "--run_dir",
            str(group_root / "run"),
            "--checkpoint_dir",
            str(group_root / "checkpoints"),
            "--manifest",
            inputs["data_manifest"]["path"],
            "--candidate_path",
            inputs["candidate"]["path"],
            "--tokenizer_path",
            inputs["tokenizer"]["path"],
            "--parent_weights",
            inputs["parent"]["path"],
            "--parent_sha256",
            inputs["parent"]["sha256"],
            "--group",
            group,
            "--peak_lr",
            str(config["peak_lr"]),
            "--warmup_ratio",
            str(config["warmup_ratio"]),
            "--floor_ratio",
            str(config["floor_ratio"]),
            "--micro_batch_size",
            str(config["micro_batch_size"]),
            "--accumulation_steps",
            str(config["accumulation_steps"]),
            "--grad_clip",
            str(config["grad_clip"]),
            "--device",
            config["device"],
            "--dtype",
            config["dtype"],
            "--num_workers",
            str(config["num_workers"]),
            "--seed",
            str(config["seed"]),
        ]
        if resume:
            command.append("--resume")
        commands.append(command)
    return commands


def pending_training_commands(execution: Mapping[str, Any]) -> list[list[str]]:
    fresh = training_commands(execution)
    resumed = training_commands(execution, resume=True)
    root = Path(execution["output_root"])
    config = execution["training"]
    commands = []
    for index, group in enumerate(config["groups"]):
        group_root = root / "groups" / f"{group}-seed-{config['seed']}"
        metrics = stage_f._read_metrics(group_root / "run" / "metrics.jsonl")
        completions = [item for item in metrics if item.get("type") == "run_complete"]
        plan = config["plans"][group]
        weights = [
            group_root
            / "checkpoints"
            / "weights"
            / f"rag_epoch_{epoch}.pth"
            for epoch in EXPORTED_EPOCHS
        ]
        forbidden = (
            group_root / "checkpoints" / "weights" / "rag_epoch_1.pth"
        )
        if forbidden.exists() or forbidden.with_suffix(".sha256").exists():
            raise ValueError(f"{group} 非法导出了 rag_epoch_1.pth")
        complete_weights = all(
            path.is_file() and path.with_suffix(".sha256").is_file()
            for path in weights
        )
        if completions:
            last = completions[-1]
            if (
                last.get("epoch") != EPOCHS
                or last.get("completed_sequences")
                != plan["sequence_exposures"]
                or last.get("optimizer_step") != plan["optimizer_steps"]
                or last.get("weights_exported_epochs")
                != list(EXPORTED_EPOCHS)
                or not complete_weights
            ):
                raise ValueError(f"{group} 完成记录与后三个 epoch 权重不闭合")
            for path in weights:
                digest = stage_f._sha256_file(path)
                if path.with_suffix(".sha256").read_text(
                    encoding="utf-8"
                ).splitlines() != [f"{digest}  {path.name}"]:
                    raise ValueError(f"{group} epoch 权重 SHA-256 无效: {path.name}")
            continue
        latest = group_root / "checkpoints" / "resume" / "latest.pt"
        if latest.is_file():
            commands.append(resumed[index])
        elif (group_root / "run").exists() or (group_root / "checkpoints").exists():
            raise RuntimeError(f"{group} 有残留目录但没有安全恢复点")
        else:
            commands.append(fresh[index])
    return commands


def _model_weight_digest(model: Mapping[str, Any]) -> str:
    if model["phase"] == "parent":
        path = Path(model["weights"]["path"])
        expected = model["weights"]["sha256"]
        if not path.is_file() or stage_f._sha256_file(path) != expected:
            raise ValueError(f"待评估父权重 SHA-256 校验失败: {path}")
        return expected
    path = Path(model["weights_path"])
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"待评估 epoch 权重或 SHA-256 不存在: {path}")
    digest = stage_f._sha256_file(path)
    if sidecar.read_text(encoding="utf-8").splitlines() != [
        f"{digest}  {path.name}"
    ]:
        raise ValueError(f"待评估 epoch 权重 SHA-256 无效: {path}")
    return digest


def _validate_evaluation_report(
    report: Mapping[str, Any],
    *,
    model: Mapping[str, Any],
    execution: Mapping[str, Any],
    expected_weights_sha256: str,
) -> list[dict[str, Any]]:
    summary = report.get("summary", {})
    inputs = report.get("inputs", {})
    evaluation = inputs.get("evaluation", {})
    if (
        report.get("pipeline") != "rag_sft_v2_answer_model_evaluation"
        or summary.get("records") != PRIMARY_EVALUATION_RECORDS
        or summary.get("total_generated_records") != EVALUATION_RECORDS
        or inputs.get("weights_sha256") != expected_weights_sha256
        or evaluation.get("manifest_sha256")
        != execution["inputs"]["evaluation_manifest"]["sha256"]
        or evaluation.get("cases_sha256")
        != execution["inputs"]["evaluation_cases"]["sha256"]
        or report.get("complete") is not True
    ):
        raise ValueError(f"单模型评估报告身份未闭合: {model['model_id']}")
    records = report.get("records")
    if not isinstance(records, list) or len(records) != EVALUATION_RECORDS:
        raise ValueError(f"单模型逐题记录未闭合: {model['model_id']}")
    primary = [
        item for item in records if item.get("evaluation_scope") == "primary_768"
    ]
    extrapolation = [
        item
        for item in records
        if item.get("evaluation_scope") == "rope_extrapolation_1024"
    ]
    if (
        len(primary) != PRIMARY_EVALUATION_RECORDS
        or [item.get("query_id") for item in extrapolation]
        != EXTRAPOLATION_QUERY_IDS
        or len({item.get("query_id") for item in records}) != EVALUATION_RECORDS
    ):
        raise ValueError(f"单模型评估范围未闭合: {model['model_id']}")
    return records


def evaluation_commands(execution: Mapping[str, Any]) -> list[list[str]]:
    root = Path(execution["output_root"])
    inputs = execution["inputs"]
    commands = []
    for model in execution["evaluation_models"]:
        path = Path(
            model["weights"]["path"]
            if model["phase"] == "parent"
            else model["weights_path"]
        )
        digest = _model_weight_digest(model)
        output = (
            root
            / "evaluation"
            / "models"
            / model["group"]
            / model["phase"]
            / "report.json"
        )
        commands.append(
            [
                sys.executable,
                "-m",
                "trainer.evaluate_rag_sft_v2",
                "--cases_manifest",
                inputs["evaluation_manifest"]["path"],
                "--cases",
                inputs["evaluation_cases"]["path"],
                "--tokenizer_path",
                inputs["tokenizer"]["path"],
                "--weights",
                str(path),
                "--weights_sha256",
                digest,
                "--output",
                str(output),
                "--device",
                execution["training"]["device"],
            ]
        )
    return commands


def pending_evaluation_commands(execution: Mapping[str, Any]) -> list[list[str]]:
    """跳过身份、范围和计数均已闭合的单模型评估报告。"""

    commands = evaluation_commands(execution)
    pending = []
    for model, command in zip(execution["evaluation_models"], commands):
        output = Path(command[command.index("--output") + 1])
        sidecar = output.with_suffix(output.suffix + ".sha256")
        if not output.exists() and not sidecar.exists():
            pending.append(command)
            continue
        report, _ = stage_f._load_verified_json(output, "单模型评估报告")
        _validate_evaluation_report(
            report,
            model=model,
            execution=execution,
            expected_weights_sha256=command[
                command.index("--weights_sha256") + 1
            ],
        )
    return pending


def aggregate_evaluations(execution: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(execution["output_root"])
    results = []
    records_by_model: dict[str, list[dict[str, Any]]] = {}
    for model in execution["evaluation_models"]:
        path = (
            root
            / "evaluation"
            / "models"
            / model["group"]
            / model["phase"]
            / "report.json"
        )
        report, digest = stage_f._load_verified_json(path, "单模型评估报告")
        records = _validate_evaluation_report(
            report,
            model=model,
            execution=execution,
            expected_weights_sha256=_model_weight_digest(model),
        )
        records_by_model[model["model_id"]] = records
        results.append(
            {
                "model_id": model["model_id"],
                "report": {"path": str(path), "sha256": digest},
                "summary": report["summary"],
            }
        )

    paired_metrics = {
        "protocol_valid_rate": ("protocol_valid", "maximize"),
        "required_recall": ("citation_metrics.required_recall", "maximize"),
        "citation_precision": ("citation_metrics.precision", "maximize"),
        "citation_exact_set": ("citation_metrics.exact_set", "maximize"),
        "hard_negative_citation": (
            "citation_metrics.hard_negative_citations",
            "minimize",
        ),
        "over_citation": ("citation_metrics.over_citations", "minimize"),
    }

    def record_metric(record: Mapping[str, Any], path: str) -> float:
        value: Any = record
        for part in path.split("."):
            value = value[part]
        if path.endswith("hard_negative_citations") or path.endswith(
            "over_citations"
        ):
            return float(value > 0)
        return float(value)

    def bootstrap(
        model_id: str, metric_name: str, deltas: list[float]
    ) -> dict[str, Any]:
        rng = random.Random(f"42:{model_id}:{metric_name}:clean-scale")
        estimates = []
        for _ in range(2000):
            estimates.append(
                sum(deltas[rng.randrange(len(deltas))] for _ in deltas)
                / len(deltas)
            )
        estimates.sort()
        return {
            "delta": sum(deltas) / len(deltas),
            "ci95": [estimates[49], estimates[1949]],
            "improved": sum(value > 0 for value in deltas),
            "tied": sum(value == 0 for value in deltas),
            "regressed": sum(value < 0 for value in deltas),
            "bootstrap_samples": 2000,
            "seed": 42,
        }

    paired_comparisons = []
    mixture_groups = [group for group in GROUPS if group != "clean_only"]
    for group in mixture_groups:
        for epoch in EXPORTED_EPOCHS:
            model_id = f"{group}/rag_epoch_{epoch}"
            baseline_id = f"clean_only/rag_epoch_{epoch}"
            child_records = [
                item
                for item in records_by_model[model_id]
                if item.get("evaluation_scope") == "primary_768"
            ]
            baseline_records = [
                item
                for item in records_by_model[baseline_id]
                if item.get("evaluation_scope") == "primary_768"
            ]
            if [item["query_id"] for item in child_records] != [
                item["query_id"] for item in baseline_records
            ]:
                raise ValueError(f"{model_id} 与同 epoch clean-only 逐题顺序不一致")
            metrics = {}
            for metric_name, (path, direction) in paired_metrics.items():
                raw = [
                    record_metric(child, path) - record_metric(baseline, path)
                    for child, baseline in zip(child_records, baseline_records)
                ]
                oriented = raw if direction == "maximize" else [-value for value in raw]
                comparison = bootstrap(model_id, metric_name, oriented)
                comparison["direction"] = direction
                comparison["raw_model_minus_clean_only_delta"] = sum(raw) / len(raw)
                metrics[metric_name] = comparison
            paired_comparisons.append(
                {
                    "model_id": model_id,
                    "clean_only_model_id": baseline_id,
                    "records": PRIMARY_EVALUATION_RECORDS,
                    "metrics": metrics,
                }
            )
    payload = {
        "schema_version": "1.0",
        "pipeline": AGGREGATE_PIPELINE,
        "models": len(results),
        "answers": sum(
            item["summary"]["total_generated_records"] for item in results
        ),
        "results": results,
        "paired_clean_only_comparisons": paired_comparisons,
        "selection_rule": execution["selection_rule"],
        "selection_applied": False,
        "private_holdout_used": False,
        "complete": len(results) == EXPECTED_MODELS,
    }
    if payload["answers"] != EXPECTED_ANSWERS:
        raise ValueError(
            f"Clean规模实验评估回答数没有闭合 {EXPECTED_ANSWERS}"
        )
    stage_f._write_immutable_json(
        root / "evaluation" / "aggregate" / "summary.json", payload
    )
    return payload


def _print_or_run(
    commands: Sequence[Sequence[str]], project_root: str | Path, execute: bool
) -> None:
    if execute:
        stage_f.run_commands(commands, project_root)
    else:
        for command in commands:
            print(subprocess.list2cmdline(command))


def main() -> None:
    parser = argparse.ArgumentParser(description="cpt_750m Clean规模实验编排")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    for name in (
        "parent_weights",
        "parent_sha256",
        "data_manifest",
        "candidate_path",
        "tokenizer_path",
        "evaluation_manifest",
        "evaluation_cases",
        "output_root",
    ):
        prepare.add_argument(f"--{name}", required=True)
    for name in ("train", "evaluate", "aggregate"):
        child = subparsers.add_parser(name)
        child.add_argument("--execution", required=True)
        if name != "aggregate":
            child.add_argument("--project-root", default="/root/autodl-tmp/minimind")
            child.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        names = (
            "parent_weights",
            "parent_sha256",
            "data_manifest",
            "candidate_path",
            "tokenizer_path",
            "evaluation_manifest",
            "evaluation_cases",
            "output_root",
        )
        payload = prepare_execution(
            **{name: getattr(args, name) for name in names}
        )
        print(
            "RAG_SFT_V2_CLEAN_SCALE_PREPARED "
            f"models={len(payload['evaluation_models'])}"
        )
        return
    execution = _load_execution(args.execution)
    if args.command == "aggregate":
        payload = aggregate_evaluations(execution)
        print(
            "RAG_SFT_V2_CLEAN_SCALE_AGGREGATE_OK "
            f"models={payload['models']} answers={payload['answers']}"
        )
        return
    commands = (
        pending_training_commands(execution)
        if args.command == "train"
        else pending_evaluation_commands(execution)
    )
    _print_or_run(commands, args.project_root, args.execute)


if __name__ == "__main__":
    main()
