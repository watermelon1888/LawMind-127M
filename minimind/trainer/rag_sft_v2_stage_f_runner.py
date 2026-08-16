"""发布并执行阶段 F 的五路训练与 20 模型统一评估计划。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


PIPELINE = "rag_sft_v2_stage_f_execution_v2"
ROLES = ("stage_b_control", "cpt_250m", "cpt_750m", "cpt_2b", "cpt_final")
EXPORTED_EPOCHS = (2, 3, 4)
SMOKE_OPTIMIZER_STEPS = 30
EVALUATION_RECORDS = 170
PRIMARY_EVALUATION_RECORDS = 166
ROPE_EXTRAPOLATION_RECORDS = 4
EXPECTED_RAG_MODELS = len(ROLES) * len(EXPORTED_EPOCHS)
EXPECTED_MODELS = len(ROLES) + EXPECTED_RAG_MODELS
EXPECTED_ANSWERS = EXPECTED_MODELS * EVALUATION_RECORDS
EXPECTED_RELEASE_RECORDS = {
    "total": 775,
    "clean": 549,
    "hard_negative": 226,
    "retrieved_hn": 70,
    "curated_hn": 156,
    "unique_queries": 549,
    "paired_hn_queries": 226,
}
CommandExecutor = Callable[[Sequence[str], Path], None]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"文件不存在: {resolved}")
    return {"path": str(resolved), "bytes": resolved.stat().st_size, "sha256": _sha256_file(resolved)}


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"输出已存在，不能覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
    digest = _sha256_file(path)
    hash_temporary = sidecar.with_suffix(sidecar.suffix + ".tmp")
    hash_temporary.write_text(f"{digest}  {path.name}\n", encoding="utf-8", newline="\n")
    hash_temporary.replace(sidecar)
    return digest


def _load_verified_json(path: str | Path, description: str) -> tuple[dict[str, Any], str]:
    resolved = Path(path).resolve()
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"{description} 或相邻 SHA-256 不存在: {resolved}")
    digest = _sha256_file(resolved)
    lines = [line for line in sidecar.read_text(encoding="utf-8").splitlines() if line]
    if lines != [f"{digest}  {resolved.name}"]:
        raise ValueError(f"{description} SHA-256 校验失败")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{description} 必须是 JSON object")
    return payload, digest


def _load_data_manifest(path: str | Path) -> tuple[dict[str, Any], str]:
    """验证 RAG-SFT v2 的多文件 manifest.sha256 清单。"""

    resolved = Path(path).resolve()
    sidecar = resolved.with_suffix(".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"RAG-SFT v2 data manifest 或清单不存在: {resolved}")
    digest = _sha256_file(resolved)
    matches = []
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError("RAG-SFT v2 data manifest SHA-256 清单格式无效")
        if Path(parts[1]).name == resolved.name:
            matches.append((parts[0].lower(), parts[1]))
    if matches != [(digest, resolved.name)]:
        raise ValueError("RAG-SFT v2 data manifest SHA-256 校验失败")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RAG-SFT v2 data manifest 必须是 JSON object")
    return payload, digest


def _load_parent_assets(path: str | Path) -> list[dict[str, str]]:
    payload = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(candidates, list) or [item.get("role") for item in candidates] != list(ROLES):
        raise ValueError("父权重清单必须按冻结顺序包含五个 role")
    values = []
    for item in candidates:
        parent = Path(item.get("path", "")).resolve()
        digest = item.get("sha256")
        if not parent.is_file() or _sha256_file(parent) != digest:
            raise ValueError(f"父权重身份校验失败: {item.get('role')}")
        values.append({"role": item["role"], "path": str(parent), "sha256": digest})
    return values


def _validate_release_records(data: Mapping[str, Any]) -> None:
    """阶段 F 只接受冻结的 549 Clean + 226 HN 正式身份。"""

    records = data.get("records")
    if not isinstance(records, dict) or any(
        records.get(name) != expected
        for name, expected in EXPECTED_RELEASE_RECORDS.items()
    ):
        raise ValueError("阶段 F 正式 release 记录数必须闭合为 549 Clean + 226 HN")
    output = data.get("output", {}).get("training_candidate")
    if not isinstance(output, dict) or output.get("records") != 775:
        raise ValueError("阶段 F training candidate 必须精确包含 775 条记录")
    if data.get("policy", {}).get("sampling_ratio_embedded") is not False:
        raise ValueError("阶段 F 正式 release 不得内嵌采样比例")


def prepare_execution(
    *,
    parent_assets: str | Path,
    data_manifest: str | Path,
    candidate_path: str | Path,
    tokenizer_path: str | Path,
    evaluation_manifest: str | Path,
    evaluation_cases: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """复核所有冻结输入并发布不可变阶段 F 执行绑定。"""

    parents = _load_parent_assets(parent_assets)
    data, data_sha256 = _load_data_manifest(data_manifest)
    readiness = data.get("readiness", {})
    if (
        data.get("pipeline") != "rag_sft_v2_training_release"
        or readiness.get("training_ready") is not True
        or readiness.get("real_hard_negatives_constructed") is not True
        or readiness.get("final_retrieval_identity_frozen") is not True
        or data.get("complete") is not True
    ):
        raise ValueError("阶段 F 只接受 training-ready v2 正式 release")
    _validate_release_records(data)
    evaluation, evaluation_sha256 = _load_verified_json(evaluation_manifest, "回答评估输入 manifest")
    if (
        evaluation.get("pipeline") != "rag_sft_v2_evaluation_inputs_v1"
        or evaluation.get("records") != {"total": 170, "legal_query": 140, "exact_lookup": 30}
        or evaluation.get("complete") is not True
    ):
        raise ValueError("回答评估输入 manifest 未闭合 170 题")
    candidate = _identity(candidate_path)
    output_metadata = data.get("output", {}).get("training_candidate")
    if not isinstance(output_metadata, dict) or any(candidate[field] != output_metadata.get(field) for field in ("bytes", "sha256")):
        raise ValueError("candidate 与 data manifest 身份不一致")
    cases = _identity(evaluation_cases)
    cases_metadata = evaluation.get("output", {}).get("cases")
    if not isinstance(cases_metadata, dict) or any(cases[field] != cases_metadata.get(field) for field in ("bytes", "sha256")):
        raise ValueError("evaluation cases 与 manifest 身份不一致")
    tokenizer_root = Path(tokenizer_path).resolve()
    expected_tokenizer = data.get("tokenizer")
    if not tokenizer_root.is_dir() or not isinstance(expected_tokenizer, dict):
        raise ValueError("Tokenizer 目录或 manifest 身份无效")
    tokenizer_files = {}
    for filename, expected in expected_tokenizer.get("files", {}).items():
        actual = _identity(tokenizer_root / filename)
        if any(actual[field] != expected.get(field) for field in ("bytes", "sha256")):
            raise ValueError(f"Tokenizer 文件身份不一致: {filename}")
        tokenizer_files[filename] = actual
    root = Path(output_root).resolve()
    models = []
    for parent in parents:
        candidate_root = root / "candidates" / f"{parent['role']}-seed-42"
        models.append({"model_id": f"{parent['role']}/parent", "role": parent["role"], "phase": "parent", "weights": parent})
        for epoch in EXPORTED_EPOCHS:
            path = candidate_root / "checkpoints" / "weights" / f"rag_epoch_{epoch}.pth"
            models.append({"model_id": f"{parent['role']}/rag_epoch_{epoch}", "role": parent["role"], "phase": f"rag_epoch_{epoch}", "weights_path": str(path)})
    payload = {
        "schema_version": "1.0",
        "pipeline": PIPELINE,
        "inputs": {
            "parent_assets": _identity(parent_assets),
            "data_manifest": {"path": str(Path(data_manifest).resolve()), "sha256": data_sha256},
            "candidate": candidate,
            "tokenizer": {"path": str(tokenizer_root), "files": tokenizer_files},
            "evaluation_manifest": {"path": str(Path(evaluation_manifest).resolve()), "sha256": evaluation_sha256},
            "evaluation_cases": cases,
        },
        "training": {
            "epochs": 4,
            "exported_epochs": list(EXPORTED_EPOCHS),
            "seed": 42,
            "peak_lr": 1.2e-5,
            "warmup_ratio": 0.1,
            "floor_ratio": 0.1,
            "micro_batch_size": 8,
            "accumulation_steps": 2,
            "grad_clip": 1.0,
            "dtype": "bfloat16",
            "num_workers": 4,
            "device": "cuda:0",
            "smoke_optimizer_steps": SMOKE_OPTIMIZER_STEPS,
            "swanlab": {
                "project": "minimind+rag",
                "workspace": "Bigwatermelon",
            },
            "sampling": {
                "mode": "natural_full_candidate",
                "records_per_epoch": 775,
                "clean": 549,
                "hard_negative": 226,
            },
            "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "weight_decay": 0.1, "eps": 1e-8},
            "environment": {"OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"},
        },
        "candidates": parents,
        "evaluation_models": models,
        "expected_answers": EXPECTED_ANSWERS,
        "output_root": str(root),
        "complete": True,
    }
    _write_immutable_json(root / "stage-f-execution.json", payload)
    return payload


def _load_execution(path: str | Path) -> dict[str, Any]:
    payload, _ = _load_verified_json(path, "阶段 F 执行绑定")
    if (
        payload.get("schema_version") != "1.0"
        or payload.get("pipeline") != PIPELINE
        or [item.get("role") for item in payload.get("candidates", [])] != list(ROLES)
        or len(payload.get("evaluation_models", [])) != EXPECTED_MODELS
        or payload.get("expected_answers") != EXPECTED_ANSWERS
        or payload.get("training", {}).get("exported_epochs") != list(EXPORTED_EPOCHS)
        or payload.get("training", {}).get("smoke_optimizer_steps") != SMOKE_OPTIMIZER_STEPS
        or payload.get("complete") is not True
    ):
        raise ValueError("阶段 F 执行绑定版本、顺序或计数无效")
    return payload


def training_commands(execution: Mapping[str, Any], *, resume: bool = False) -> list[list[str]]:
    root = Path(execution["output_root"])
    inputs = execution["inputs"]
    config = execution["training"]
    commands = []
    for candidate in execution["candidates"]:
        candidate_root = root / "candidates" / f"{candidate['role']}-seed-42"
        run_name = f"rag-sft-stage-f-v2-{candidate['role']}-seed-42"
        command = [
            sys.executable, "-m", "trainer.train_rag_sft_v2",
            "--run_name", run_name,
            "--run_dir", str(candidate_root / "run"),
            "--checkpoint_dir", str(candidate_root / "checkpoints"),
            "--manifest", inputs["data_manifest"]["path"],
            "--candidate_path", inputs["candidate"]["path"],
            "--tokenizer_path", inputs["tokenizer"]["path"],
            "--parent_weights", candidate["path"],
            "--parent_sha256", candidate["sha256"],
            "--peak_lr", str(config["peak_lr"]),
            "--warmup_ratio", str(config["warmup_ratio"]),
            "--floor_ratio", str(config["floor_ratio"]),
            "--micro_batch_size", str(config["micro_batch_size"]),
            "--accumulation_steps", str(config["accumulation_steps"]),
            "--grad_clip", str(config["grad_clip"]),
            "--device", config["device"],
            "--dtype", config["dtype"],
            "--num_workers", str(config["num_workers"]),
            "--seed", str(config["seed"]),
            "--swanlab",
            "--swanlab_project", config["swanlab"]["project"],
            "--swanlab_workspace", config["swanlab"]["workspace"],
        ]
        if resume:
            command.append("--resume")
        commands.append(command)
    return commands


def smoke_commands(execution: Mapping[str, Any]) -> list[list[str]]:
    """为五个父权重分别生成覆盖峰值 LR 的 30-step 隔离 smoke。"""

    inputs = execution["inputs"]
    config = execution["training"]
    commands = []
    for candidate in execution["candidates"]:
        role = candidate["role"]
        root = Path(execution["output_root"]) / "smoke" / f"{role}-seed-42"
        commands.append([
            sys.executable, "-m", "trainer.train_rag_sft_v2",
            "--run_name", f"{role}-seed-42-smoke",
            "--run_dir", str(root / "run"),
            "--checkpoint_dir", str(root / "checkpoints"),
            "--manifest", inputs["data_manifest"]["path"],
            "--candidate_path", inputs["candidate"]["path"],
            "--tokenizer_path", inputs["tokenizer"]["path"],
            "--parent_weights", candidate["path"],
            "--parent_sha256", candidate["sha256"],
            "--peak_lr", str(config["peak_lr"]),
            "--warmup_ratio", str(config["warmup_ratio"]),
            "--floor_ratio", str(config["floor_ratio"]),
            "--micro_batch_size", str(config["micro_batch_size"]),
            "--accumulation_steps", str(config["accumulation_steps"]),
            "--grad_clip", str(config["grad_clip"]),
            "--device", config["device"],
            "--dtype", config["dtype"],
            "--num_workers", str(config["num_workers"]),
            "--seed", str(config["seed"]),
            "--smoke_optimizer_steps", str(config["smoke_optimizer_steps"]),
            "--no-swanlab",
        ])
    return commands


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    values = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"训练 metrics 第 {line_number} 行为空")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"训练 metrics 第 {line_number} 行不是 object")
            values.append(value)
    return values


def pending_training_commands(execution: Mapping[str, Any]) -> list[list[str]]:
    """按 lineage 状态生成新训或同身份恢复命令，完整 lineage 直接跳过。"""

    fresh = training_commands(execution, resume=False)
    resumed = training_commands(execution, resume=True)
    root = Path(execution["output_root"])
    commands = []
    for index, candidate in enumerate(execution["candidates"]):
        candidate_root = root / "candidates" / f"{candidate['role']}-seed-42"
        run_dir = candidate_root / "run"
        checkpoint_dir = candidate_root / "checkpoints"
        completions = [
            item for item in _read_metrics(run_dir / "metrics.jsonl")
            if item.get("type") == "run_complete"
        ]
        forbidden = checkpoint_dir / "weights" / "rag_epoch_1.pth"
        if forbidden.exists() or forbidden.with_suffix(".sha256").exists():
            raise ValueError(f"{candidate['role']} 禁止存在 rag_epoch_1.pth")
        weights = [
            checkpoint_dir / "weights" / f"rag_epoch_{epoch}.pth"
            for epoch in EXPORTED_EPOCHS
        ]
        complete_weights = all(
            path.is_file() and path.with_suffix(".sha256").is_file()
            for path in weights
        )
        if completions:
            last = completions[-1]
            if (
                last.get("epoch") != 4
                or last.get("next_sequence_position") != 0
                or last.get("weights_exported_epochs") != list(EXPORTED_EPOCHS)
                or not complete_weights
            ):
                raise ValueError(f"{candidate['role']} 的完成记录与 E2–E4 权重不闭合")
            for path in weights:
                digest = _sha256_file(path)
                if path.with_suffix(".sha256").read_text(encoding="utf-8").splitlines() != [f"{digest}  {path.name}"]:
                    raise ValueError(f"{candidate['role']} 的 epoch 权重 SHA-256 无效: {path.name}")
            continue
        latest = checkpoint_dir / "resume" / "latest.pt"
        if latest.is_file():
            commands.append(resumed[index])
        elif run_dir.exists() or checkpoint_dir.exists():
            raise RuntimeError(f"{candidate['role']} 有残留目录但没有安全恢复点")
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
                raise FileNotFoundError(f"待评估 epoch 权重或 SHA-256 不存在: {path}")
            digest = _sha256_file(path)
            if sidecar.read_text(encoding="utf-8").splitlines() != [f"{digest}  {path.name}"]:
                raise ValueError(f"待评估 epoch 权重 SHA-256 校验失败: {path}")
        output = root / "evaluation" / "models" / model["role"] / model["phase"] / "report.json"
        commands.append([
            sys.executable, "-m", "trainer.evaluate_rag_sft_v2",
            "--cases_manifest", inputs["evaluation_manifest"]["path"],
            "--cases", inputs["evaluation_cases"]["path"],
            "--tokenizer_path", inputs["tokenizer"]["path"],
            "--weights", str(path),
            "--weights_sha256", digest,
            "--output", str(output),
            "--device", execution["training"]["device"],
        ])
    return commands


def pending_evaluation_commands(execution: Mapping[str, Any]) -> list[list[str]]:
    """跳过已通过身份与 170 题计数校验的单模型报告。"""

    commands = evaluation_commands(execution)
    pending = []
    for command in commands:
        output = Path(command[command.index("--output") + 1])
        sidecar = output.with_suffix(output.suffix + ".sha256")
        if not output.exists() and not sidecar.exists():
            pending.append(command)
            continue
        report, _ = _load_verified_json(output, "单模型评估报告")
        summary = report.get("summary", {})
        if (
            report.get("pipeline") != "rag_sft_v2_answer_model_evaluation"
            or summary.get("records") != PRIMARY_EVALUATION_RECORDS
            or summary.get("total_generated_records") != EVALUATION_RECORDS
            or summary.get("rope_extrapolation", {}).get("records")
            != ROPE_EXTRAPOLATION_RECORDS
            or report.get("complete") is not True
        ):
            raise ValueError(f"已存在单模型评估报告无效: {output}")
    return pending


def aggregate_evaluations(execution: Mapping[str, Any]) -> dict[str, Any]:
    """在 20 份报告全部闭合后发布结构化自动指标汇总。"""

    root = Path(execution["output_root"])
    results = []
    records_by_model: dict[str, list[dict[str, Any]]] = {}
    for model in execution["evaluation_models"]:
        path = root / "evaluation" / "models" / model["role"] / model["phase"] / "report.json"
        report, digest = _load_verified_json(path, "单模型评估报告")
        summary = report.get("summary", {})
        if (
            report.get("pipeline") != "rag_sft_v2_answer_model_evaluation"
            or summary.get("records") != PRIMARY_EVALUATION_RECORDS
            or summary.get("total_generated_records") != EVALUATION_RECORDS
            or summary.get("rope_extrapolation", {}).get("records")
            != ROPE_EXTRAPOLATION_RECORDS
            or report.get("complete") is not True
        ):
            raise ValueError(f"单模型评估报告未闭合: {model['model_id']}")
        records = report.get("records")
        if not isinstance(records, list) or len(records) != EVALUATION_RECORDS:
            raise ValueError(f"单模型逐题记录未闭合: {model['model_id']}")
        records_by_model[model["model_id"]] = records
        results.append({"model_id": model["model_id"], "report": {"path": str(path), "sha256": digest}, "summary": report["summary"]})
    rag_results = [item for item in results if "/rag_epoch_" in item["model_id"]]
    dimensions = {
        "maximize": [
            "protocol_valid_rate",
            "citation.required_recall",
            "citation.precision",
            "citation.exact_set_accuracy",
        ],
        "minimize": [
            "citation.hard_negative_citation_rate",
            "citation.over_citation_rate",
        ],
    }

    def metric(item: Mapping[str, Any], path: str) -> float:
        value: Any = item["summary"]
        for part in path.split("."):
            value = value[part]
        return float(value)

    def dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
        non_worse = all(metric(left, name) >= metric(right, name) for name in dimensions["maximize"])
        non_worse = non_worse and all(metric(left, name) <= metric(right, name) for name in dimensions["minimize"])
        strictly_better = any(metric(left, name) > metric(right, name) for name in dimensions["maximize"])
        strictly_better = strictly_better or any(metric(left, name) < metric(right, name) for name in dimensions["minimize"])
        return non_worse and strictly_better

    non_dominated = [
        item["model_id"]
        for item in rag_results
        if not any(dominates(other, item) for other in rag_results if other is not item)
    ]
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

    def paired_bootstrap(model_id: str, metric_name: str, deltas: list[float]) -> dict[str, Any]:
        mean = sum(deltas) / len(deltas)
        rng = random.Random(f"42:{model_id}:{metric_name}")
        estimates = []
        for _ in range(2000):
            estimates.append(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas))
        estimates.sort()
        return {
            "delta": mean,
            "ci95": [estimates[49], estimates[1949]],
            "improved": sum(value > 0 for value in deltas),
            "tied": sum(value == 0 for value in deltas),
            "regressed": sum(value < 0 for value in deltas),
            "bootstrap_samples": 2000,
            "seed": 42,
        }

    paired_comparisons = []
    for item in rag_results:
        model_id = item["model_id"]
        role = model_id.split("/", 1)[0]
        parent_id = f"{role}/parent"
        child_records = records_by_model[model_id]
        parent_records = records_by_model[parent_id]
        if [record["query_id"] for record in child_records] != [record["query_id"] for record in parent_records]:
            raise ValueError(f"{model_id} 与父模型逐题顺序不一致")
        metrics = {}
        for metric_name, (path, direction) in paired_metrics.items():
            raw_deltas = [
                record_metric(child, path) - record_metric(parent, path)
                for child, parent in zip(child_records, parent_records)
            ]
            oriented = raw_deltas if direction == "maximize" else [-value for value in raw_deltas]
            comparison = paired_bootstrap(model_id, metric_name, oriented)
            comparison["direction"] = direction
            comparison["raw_child_minus_parent_delta"] = sum(raw_deltas) / len(raw_deltas)
            metrics[metric_name] = comparison
        paired_comparisons.append({"model_id": model_id, "parent_model_id": parent_id, "records": EVALUATION_RECORDS, "metrics": metrics})
    payload = {
        "schema_version": "1.0",
        "pipeline": "rag_sft_v2_stage_f_evaluation_aggregate_v2",
        "models": len(results),
        "answers": sum(
            item["summary"]["total_generated_records"] for item in results
        ),
        "results": results,
        "paired_parent_comparisons": paired_comparisons,
        "semantic_review_filter": {
            "method": "automatic_metric_pareto_front",
            "scope": "15_rag_checkpoints_only",
            "dimensions": dimensions,
            "non_dominated_model_ids": non_dominated,
            "final_selection_rule": None,
        },
        "selection_applied": False,
        "private_holdout_used": False,
        "complete": len(results) == EXPECTED_MODELS,
    }
    if payload["answers"] != EXPECTED_ANSWERS:
        raise ValueError("阶段 F 自动评估回答数没有闭合 3,400")
    _write_immutable_json(root / "evaluation" / "aggregate" / "summary.json", payload)
    return payload


def _subprocess_executor(command: Sequence[str], cwd: Path) -> None:
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    python_paths = [str(cwd.parent), str(cwd)]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    subprocess.run(list(command), cwd=cwd, check=True, env=environment)


def run_commands(commands: Sequence[Sequence[str]], project_root: str | Path, executor: CommandExecutor = _subprocess_executor) -> None:
    project = Path(project_root).resolve()
    if not (project / "trainer").is_dir():
        raise FileNotFoundError(f"MiniMind 项目根目录无效: {project}")
    for command in commands:
        executor(command, project)


def main() -> None:
    parser = argparse.ArgumentParser(description="阶段 F 五路训练与 20 模型评估编排")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    for name in ("parent_assets", "data_manifest", "candidate_path", "tokenizer_path", "evaluation_manifest", "evaluation_cases", "output_root"):
        prepare.add_argument(f"--{name}", required=True)
    for name in ("smoke", "train", "evaluate", "aggregate"):
        child = subparsers.add_parser(name)
        child.add_argument("--execution", required=True)
        if name != "aggregate":
            child.add_argument("--project-root", default="/root/autodl-tmp/minimind")
            child.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        payload = prepare_execution(**{name: getattr(args, name) for name in ("parent_assets", "data_manifest", "candidate_path", "tokenizer_path", "evaluation_manifest", "evaluation_cases", "output_root")})
        print(f"RAG_SFT_V2_STAGE_F_PREPARED models={len(payload['evaluation_models'])}")
        return
    execution = _load_execution(args.execution)
    if args.command == "aggregate":
        payload = aggregate_evaluations(execution)
        print(f"RAG_SFT_V2_STAGE_F_AGGREGATE_OK models={payload['models']} answers={payload['answers']}")
        return
    if args.command == "smoke":
        commands = smoke_commands(execution)
    elif args.command == "train":
        commands = pending_training_commands(execution)
    else:
        commands = pending_evaluation_commands(execution)
    if args.execute:
        run_commands(commands, args.project_root)
    else:
        for command in commands:
            print(subprocess.list2cmdline(command))


if __name__ == "__main__":
    main()
