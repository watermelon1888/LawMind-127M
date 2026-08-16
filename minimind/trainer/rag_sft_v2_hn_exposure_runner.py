"""编排 cpt_750m clean/HN 曝光率对照训练与统一评估。"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from ..dataset.rag_sft_v2_exposure import EXPOSURE_GROUPS, audit_clean_hn_pairing
except ImportError:
    from dataset.rag_sft_v2_exposure import EXPOSURE_GROUPS, audit_clean_hn_pairing
from . import rag_sft_v2_stage_f_runner as stage_f


PIPELINE = "rag_sft_v2_cpt750m_hn_exposure_execution_v1"
CHECKPOINT_STEPS = (35, 70, 105, 140)
GROUPS = tuple(EXPOSURE_GROUPS)
EVALUATION_RECORDS = 170
PRIMARY_EVALUATION_RECORDS = 166
EVALUATION_COUNTS = {"total": 170, "legal_query": 140, "exact_lookup": 30}
EXTRAPOLATION_QUERY_IDS = ["Q033", "Q048", "Q063", "Q091"]


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
        raise ValueError("HN 曝光率实验只接受 training-ready v2 release")
    candidate = stage_f._identity(candidate_path)
    metadata = data.get("output", {}).get("training_candidate")
    if not isinstance(metadata, dict) or any(
        candidate[field] != metadata.get(field) for field in ("bytes", "sha256")
    ):
        raise ValueError("candidate 与 data manifest 身份不一致")
    pairing = audit_clean_hn_pairing(candidate["path"])

    evaluation, evaluation_sha256 = stage_f._load_verified_json(
        evaluation_manifest, "回答评估输入 manifest"
    )
    if (
        evaluation.get("pipeline") != "rag_sft_v2_evaluation_inputs_v1"
        or evaluation.get("records") != EVALUATION_COUNTS
        or evaluation.get("evaluation_scopes", {})
        .get("rope_extrapolation_1024", {})
        .get("query_ids") != EXTRAPOLATION_QUERY_IDS
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
        if any(actual[field] != expected.get(field) for field in ("bytes", "sha256")):
            raise ValueError(f"Tokenizer 文件身份不一致: {filename}")
        tokenizer_files[filename] = actual

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
        for step in CHECKPOINT_STEPS:
            path = root / "groups" / f"{group}-seed-42" / "checkpoints" / "weights" / f"rag_step_{step}.pth"
            models.append(
                {
                    "model_id": f"{group}/rag_step_{step}",
                    "group": group,
                    "phase": f"rag_step_{step}",
                    "weights_path": str(path),
                }
            )
    payload = {
        "schema_version": "1.0",
        "pipeline": PIPELINE,
        "inputs": {
            "parent": parent,
            "data_manifest": {"path": str(Path(data_manifest).resolve()), "sha256": data_sha256},
            "candidate": candidate,
            "pairing_audit": pairing,
            "tokenizer": {"path": str(tokenizer_root), "files": tokenizer_files},
            "evaluation_manifest": {"path": str(Path(evaluation_manifest).resolve()), "sha256": evaluation_sha256},
            "evaluation_cases": cases,
        },
        "training": {
            "groups": list(GROUPS),
            "hn_multipliers": EXPOSURE_GROUPS,
            "optimizer_steps": 140,
            "checkpoint_steps": list(CHECKPOINT_STEPS),
            "seed": 42,
            "followup_seeds": [43, 44],
            "peak_lr": 1.2e-5,
            "warmup_ratio": 0.1,
            "floor_ratio": 0.1,
            "micro_batch_size": 8,
            "accumulation_steps": 2,
            "grad_clip": 1.0,
            "dtype": "bfloat16",
            "num_workers": 4,
            "device": "cuda:0",
        },
        "evaluation_models": models,
        "expected_answers": 17 * EVALUATION_RECORDS,
        "selection_rule": {
            "status": "deferred_until_results",
            "clean_guardrail": "相对 clean-only 最好下降不超过 1–2 个百分点，最终硬阈值待结果后审核",
            "tie_break": "HN-Mid 与 HN-High 接近时选择 HN-Mid",
            "followup": "首轮 seed 42 后，最好的两档追加 seed 43、44",
        },
        "output_root": str(root),
        "complete": True,
    }
    stage_f._write_immutable_json(root / "hn-exposure-execution.json", payload)
    return payload


def _load_execution(path: str | Path) -> dict[str, Any]:
    payload, _ = stage_f._load_verified_json(path, "HN 曝光率执行绑定")
    if (
        payload.get("pipeline") != PIPELINE
        or payload.get("training", {}).get("optimizer_steps") != 140
        or payload.get("training", {}).get("checkpoint_steps") != list(CHECKPOINT_STEPS)
        or len(payload.get("evaluation_models", [])) != 17
        or payload.get("expected_answers") != 2890
        or payload.get("complete") is not True
    ):
        raise ValueError("HN 曝光率执行绑定版本、步数或计数无效")
    return payload


def training_commands(execution: Mapping[str, Any], *, resume: bool = False) -> list[list[str]]:
    root = Path(execution["output_root"])
    inputs = execution["inputs"]
    config = execution["training"]
    commands = []
    for group in config["groups"]:
        group_root = root / "groups" / f"{group}-seed-{config['seed']}"
        command = [
            sys.executable,
            "-m",
            "trainer.train_rag_sft_v2_hn_exposure",
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
    commands = []
    for index, group in enumerate(execution["training"]["groups"]):
        group_root = root / "groups" / f"{group}-seed-{execution['training']['seed']}"
        metrics = stage_f._read_metrics(group_root / "run" / "metrics.jsonl")
        completions = [item for item in metrics if item.get("type") == "run_complete"]
        weights = [group_root / "checkpoints" / "weights" / f"rag_step_{step}.pth" for step in CHECKPOINT_STEPS]
        complete_weights = all(path.is_file() and path.with_suffix(".sha256").is_file() for path in weights)
        if completions:
            last = completions[-1]
            if last.get("optimizer_step") != 140 or last.get("completed_sequences") != 2240 or last.get("weights_exported") is not True or not complete_weights:
                raise ValueError(f"{group} 完成记录与四个 step 权重不闭合")
            for path in weights:
                digest = stage_f._sha256_file(path)
                if path.with_suffix(".sha256").read_text(encoding="utf-8").splitlines() != [f"{digest}  {path.name}"]:
                    raise ValueError(f"{group} step 权重 SHA-256 无效: {path.name}")
            continue
        latest = group_root / "checkpoints" / "resume" / "latest.pt"
        if latest.is_file():
            commands.append(resumed[index])
        elif (group_root / "run").exists() or (group_root / "checkpoints").exists():
            raise RuntimeError(f"{group} 有残留目录但没有安全恢复点")
        else:
            commands.append(fresh[index])
    return commands


def evaluation_commands(execution: Mapping[str, Any]) -> list[list[str]]:
    root = Path(execution["output_root"])
    inputs = execution["inputs"]
    commands = []
    for model in execution["evaluation_models"]:
        if model["phase"] == "parent":
            path = Path(model["weights"]["path"])
            digest = model["weights"]["sha256"]
        else:
            path = Path(model["weights_path"])
            sidecar = path.with_suffix(".sha256")
            if not path.is_file() or not sidecar.is_file():
                raise FileNotFoundError(f"待评估 step 权重或 SHA-256 不存在: {path}")
            digest = stage_f._sha256_file(path)
            if sidecar.read_text(encoding="utf-8").splitlines() != [f"{digest}  {path.name}"]:
                raise ValueError(f"待评估 step 权重 SHA-256 无效: {path}")
        output = root / "evaluation" / "models" / model["group"] / model["phase"] / "report.json"
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


def aggregate_evaluations(execution: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(execution["output_root"])
    results = []
    records_by_model: dict[str, list[dict[str, Any]]] = {}
    for model in execution["evaluation_models"]:
        path = root / "evaluation" / "models" / model["group"] / model["phase"] / "report.json"
        report, digest = stage_f._load_verified_json(path, "单模型评估报告")
        if (
            report.get("summary", {}).get("records") != PRIMARY_EVALUATION_RECORDS
            or report.get("summary", {}).get("total_generated_records") != EVALUATION_RECORDS
            or report.get("complete") is not True
        ):
            raise ValueError(f"单模型评估报告未闭合: {model['model_id']}")
        records = report.get("records")
        if not isinstance(records, list) or len(records) != EVALUATION_RECORDS:
            raise ValueError(f"单模型逐题记录未闭合: {model['model_id']}")
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
        "hard_negative_citation": ("citation_metrics.hard_negative_citations", "minimize"),
        "over_citation": ("citation_metrics.over_citations", "minimize"),
    }

    def record_metric(record: Mapping[str, Any], path: str) -> float:
        value: Any = record
        for part in path.split("."):
            value = value[part]
        if path.endswith("hard_negative_citations") or path.endswith("over_citations"):
            return float(value > 0)
        return float(value)

    def bootstrap(model_id: str, metric_name: str, deltas: list[float]) -> dict[str, Any]:
        rng = random.Random(f"42:{model_id}:{metric_name}:clean-only")
        estimates = []
        for _ in range(2000):
            estimates.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas))
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
    for group in ("hn_low", "hn_mid", "hn_high"):
        for step in CHECKPOINT_STEPS:
            model_id = f"{group}/rag_step_{step}"
            baseline_id = f"clean_only/rag_step_{step}"
            child_records = records_by_model[model_id]
            baseline_records = records_by_model[baseline_id]
            child_records = [
                item for item in child_records if item.get("evaluation_scope") == "primary_768"
            ]
            baseline_records = [
                item for item in baseline_records if item.get("evaluation_scope") == "primary_768"
            ]
            if [item["query_id"] for item in child_records] != [item["query_id"] for item in baseline_records]:
                raise ValueError(f"{model_id} 与同 step clean-only 逐题顺序不一致")
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
        "pipeline": "rag_sft_v2_cpt750m_hn_exposure_aggregate_v1",
        "models": len(results),
        "answers": sum(item["summary"]["total_generated_records"] for item in results),
        "results": results,
        "paired_clean_only_comparisons": paired_comparisons,
        "selection_rule": execution["selection_rule"],
        "selection_applied": False,
        "private_holdout_used": False,
        "complete": len(results) == 17,
    }
    if payload["answers"] != 2890:
        raise ValueError("HN 曝光率评估回答数没有闭合 2,890")
    stage_f._write_immutable_json(root / "evaluation" / "aggregate" / "summary.json", payload)
    return payload


def _print_or_run(commands: Sequence[Sequence[str]], project_root: str | Path, execute: bool) -> None:
    if execute:
        stage_f.run_commands(commands, project_root)
    else:
        for command in commands:
            print(subprocess.list2cmdline(command))


def main() -> None:
    parser = argparse.ArgumentParser(description="cpt_750m HN 曝光率实验编排")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    for name in ("parent_weights", "parent_sha256", "data_manifest", "candidate_path", "tokenizer_path", "evaluation_manifest", "evaluation_cases", "output_root"):
        prepare.add_argument(f"--{name}", required=True)
    for name in ("train", "evaluate", "aggregate"):
        child = subparsers.add_parser(name)
        child.add_argument("--execution", required=True)
        if name != "aggregate":
            child.add_argument("--project-root", default="/root/autodl-tmp/minimind")
            child.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        payload = prepare_execution(**{name: getattr(args, name) for name in ("parent_weights", "parent_sha256", "data_manifest", "candidate_path", "tokenizer_path", "evaluation_manifest", "evaluation_cases", "output_root")})
        print(f"RAG_SFT_V2_HN_EXPOSURE_PREPARED models={len(payload['evaluation_models'])}")
        return
    execution = _load_execution(args.execution)
    if args.command == "aggregate":
        payload = aggregate_evaluations(execution)
        print(f"RAG_SFT_V2_HN_EXPOSURE_AGGREGATE_OK models={payload['models']} answers={payload['answers']}")
        return
    commands = pending_training_commands(execution) if args.command == "train" else evaluation_commands(execution)
    _print_or_run(commands, args.project_root, args.execute)


if __name__ == "__main__":
    main()
