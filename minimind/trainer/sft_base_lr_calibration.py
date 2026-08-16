"""发布和汇总基础法律 SFT 的共同 peak LR 校准计划。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import sft_base_generation_evaluation as generation
from . import train_full_sft as training
try:
    from ..dataset.pretrain_dataset import PretrainShardCatalog
except ImportError:
    from dataset.pretrain_dataset import PretrainShardCatalog


SPEC_PIPELINE = "legal_sft_base_lr_calibration_spec_v1"
PLAN_PIPELINE = "legal_sft_base_lr_calibration_plan_v1"
BASELINE_PIPELINE = "legal_sft_base_lr_calibration_baseline_v1"
TRIAL_PIPELINE = "legal_sft_base_lr_calibration_trial_v1"
SUMMARY_PIPELINE = "legal_sft_base_lr_calibration_summary_v1"
DECISION_PIPELINE = "legal_sft_base_lr_calibration_decision_v1"
CALIBRATION_ROLES = ("stage_b_control", "cpt_final")
PARENT_CPT_INPUT_TOKENS = {
    "stage_b_control": 0,
    "cpt_final": 2_447_821_056,
}
PEAK_LRS = (8e-6, 1e-5, 1.2e-5)
SEED = 42
CALIBRATION_PERCENT = 5
WARMUP_ASSISTANT_TOKENS = 100_000
FLOOR_RATIO = 0.1
MICRO_BATCH_SIZE = 8
ACCUMULATION_STEPS = 2
GRAD_CLIP = 1.0
DTYPE = "bfloat16"
SWANLAB_PROJECT = "minimind+rag"
SWANLAB_WORKSPACE = "Bigwatermelon"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"文件不存在: {resolved}")
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _verify_identity(value: object, description: str) -> Path:
    if not isinstance(value, dict) or not isinstance(value.get("path"), str):
        raise ValueError(f"{description}身份无效")
    path = Path(value["path"]).resolve()
    if (
        not path.is_file()
        or value.get("bytes") != path.stat().st_size
        or value.get("sha256") != _sha256_file(path)
    ):
        raise ValueError(f"{description}身份发生漂移")
    return path


def _write_immutable_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    resolved = Path(path).resolve()
    hash_path = resolved.with_suffix(".sha256")
    if resolved.exists() or hash_path.exists():
        raise FileExistsError(f"输出已存在，不能覆盖: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    digest = _sha256_file(resolved)
    hash_path.write_text(
        f"{digest}  {resolved.name}\n", encoding="utf-8", newline="\n"
    )
    return digest


def _load_verified(path: str | Path, description: str) -> tuple[dict[str, Any], str]:
    return training._load_verified_json(Path(path).resolve(), description)


def _embedded_label_report(data_manifest: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    artifacts = data_manifest.get("release_artifacts")
    label_artifact = artifacts.get("label_report") if isinstance(artifacts, dict) else None
    file_identity = label_artifact.get("file") if isinstance(label_artifact, dict) else None
    path = _verify_identity(file_identity, "正式 label 审计报告")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("无法读取正式 label 审计报告") from error
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "1.0"
        or payload.get("pipeline") != "legal_sft_dataset_label_audit_768_v1"
        or payload.get("complete") is not True
    ):
        raise ValueError("正式 label 审计报告状态无效")
    return path, payload


def _training_budget(label_report: Mapping[str, Any]) -> tuple[int, int, int]:
    records = label_report.get("records")
    by_split = records.get("by_split") if isinstance(records, dict) else None
    train = by_split.get("train") if isinstance(by_split, dict) else None
    if not isinstance(train, dict):
        raise ValueError("label 审计报告缺少 train 分层")
    train_records = train.get("records")
    train_tokens = train.get("total_active_label_tokens")
    if (
        type(train_records) is not int
        or train_records <= 0
        or type(train_tokens) is not int
        or train_tokens <= WARMUP_ASSISTANT_TOKENS
    ):
        raise ValueError("正式 train 记录或 assistant-token 统计无效")
    calibration_tokens = train_tokens * CALIBRATION_PERCENT // 100
    if calibration_tokens <= WARMUP_ASSISTANT_TOKENS:
        raise ValueError("5% 校准预算必须覆盖固定 100K warmup")
    return train_records, train_tokens, calibration_tokens


def _candidate(role: str, path: str | Path) -> dict[str, Any]:
    identity = _identity(path)
    identity.update(
        {
            "role": role,
            "cpt_input_tokens": PARENT_CPT_INPUT_TOKENS[role],
            "load_semantics": "strict_original_parent_model_state_only",
        }
    )
    return identity


def _general_validation_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    payload, digest = _load_verified(resolved, "Stage B 通用 full validation manifest")
    packing = payload.get("packing")
    if (
        payload.get("schema_version") != "1.0"
        or not isinstance(packing, dict)
        or packing.get("sequence_length") != 768
    ):
        raise ValueError("通用验证 manifest 必须固定 sequence_length=768")
    catalog = PretrainShardCatalog(resolved)
    dataset = catalog.create_dataset("full_validation")
    try:
        sequences = len(dataset)
    finally:
        dataset.close()
    identity = _identity(resolved)
    identity.update({"split": "full_validation", "sequences": sequences})
    if identity["sha256"] != digest:
        raise ValueError("通用验证 manifest 摘要漂移")
    return identity


def _generation_identity(
    path: str | Path, tokenizer: Any, tokenizer_path: str | Path
) -> dict[str, Any]:
    manifest, digest, records = generation._load_generation_records(
        path, tokenizer, tokenizer_path
    )
    identity = _identity(path)
    if identity["sha256"] != digest:
        raise ValueError("固定生成集 manifest 摘要漂移")
    identity.update(
        {
            "records": len(records),
            "selection": manifest["selection"],
            "generation": manifest["generation"],
        }
    )
    return identity


def create_spec(
    *,
    stage_b_control: str | Path,
    cpt_final: str | Path,
    data_manifest: str | Path,
    evaluation_manifest: str | Path,
    general_validation: str | Path,
    generation_manifest: str | Path,
    tokenizer_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """复算正式资产并发布两端三档共同 LR 校准 spec。"""

    data, evaluation, manifest_identity = training._load_training_manifests(
        Path(data_manifest),
        Path(evaluation_manifest),
        allow_provisional_data=False,
    )
    if not manifest_identity["formal_training_ready"]:
        raise ValueError("基础法律数据尚未正式 training-ready")
    label_path, label_report = _embedded_label_report(data)
    train_records, train_tokens, calibration_tokens = _training_budget(label_report)
    tokenizer = training._load_tokenizer(tokenizer_path)
    tokenizer_identity = training._tokenizer_identity(tokenizer, tokenizer_path)
    candidates = [
        _candidate("stage_b_control", stage_b_control),
        _candidate("cpt_final", cpt_final),
    ]
    if candidates[0]["sha256"] == candidates[1]["sha256"]:
        raise ValueError("Stage B 与 CPT final 不能使用相同父权重")

    spec = {
        "schema_version": "1.0",
        "pipeline": SPEC_PIPELINE,
        "created_at": _utc_now(),
        "scope": {
            "task": "base_legal_sft_parent_lr_calibration",
            "query_sft_in_scope": False,
            "rag_sft_in_scope": False,
            "private_holdout_used": False,
        },
        "candidates": candidates,
        "inputs": {
            "data_manifest": _identity(data_manifest),
            "evaluation_manifest": _identity(evaluation_manifest),
            "label_report": _identity(label_path),
            "general_validation": _general_validation_identity(general_validation),
            "generation_manifest": _generation_identity(
                generation_manifest, tokenizer, tokenizer_path
            ),
        },
        "tokenizer": tokenizer_identity,
        "training": {
            "seed": SEED,
            "peak_lrs": list(PEAK_LRS),
            "train_records": train_records,
            "full_epoch_assistant_tokens": train_tokens,
            "calibration_percent": CALIBRATION_PERCENT,
            "calibration_stop_assistant_tokens": calibration_tokens,
            "warmup_assistant_tokens": WARMUP_ASSISTANT_TOKENS,
            "schedule_assistant_tokens": train_tokens,
            "floor_ratio": FLOOR_RATIO,
            "micro_batch_size": MICRO_BATCH_SIZE,
            "accumulation_steps": ACCUMULATION_STEPS,
            "grad_clip": GRAD_CLIP,
            "dtype": DTYPE,
            "sequence_length": 768,
            "label_mask_version": "assistant_answer_only_v1",
        },
        "evaluation": {
            "baseline_suites": ["general", "legal_full", "fixed_generation"],
            "trial_suites": ["general", "legal_full", "fixed_generation"],
            "training_quick_points": [
                WARMUP_ASSISTANT_TOKENS,
                calibration_tokens // 2,
            ],
            "automatic_winner": None,
            "fallback_peak_lr": 1e-5,
        },
        "tracking": {
            "swanlab": {
                "enabled": True,
                "project": SWANLAB_PROJECT,
                "workspace": SWANLAB_WORKSPACE,
                "experiment_name_source": "trial_id",
                "resume": "same_trial_run_id",
                "failure_policy": "warn_and_continue_local_jsonl",
                "local_jsonl_always_enabled": True,
            }
        },
        "formal_training_policy": {
            "allowed_parent_roles": [
                "stage_b_control",
                "cpt_250m",
                "cpt_750m",
                "cpt_2b",
                "cpt_final",
            ],
            "must_restart_from_original_cpt_model_only": True,
            "calibration_checkpoint_continuation_forbidden": True,
        },
        "complete": True,
    }
    _write_immutable_json(output, spec)
    return spec


def _lr_slug(value: float) -> str:
    slugs = {8e-6: "8e-6", 1e-5: "1e-5", 1.2e-5: "1p2e-5"}
    try:
        return slugs[value]
    except KeyError as error:
        raise ValueError(f"未知校准 LR: {value}") from error


def prepare_plan(
    *, spec_path: str | Path, output_root: str | Path, output: str | Path
) -> dict[str, Any]:
    """把校准 spec 编译为两个 baseline 和六个独立 trial。"""

    spec, spec_digest = _load_verified(spec_path, "基础法律 LR 校准 spec")
    if spec.get("pipeline") != SPEC_PIPELINE or spec.get("complete") is not True:
        raise ValueError("基础法律 LR 校准 spec 无效")
    candidates = spec.get("candidates")
    if (
        not isinstance(candidates, list)
        or [item.get("role") for item in candidates] != list(CALIBRATION_ROLES)
    ):
        raise ValueError("校准 spec 必须按 Stage B、CPT final 排列")

    root = Path(output_root).resolve()
    baselines = []
    trials = []
    for candidate in candidates:
        role = candidate["role"]
        baselines.append(
            {
                "observation_id": f"{role}-parent",
                "candidate_role": role,
                "weights": candidate,
                "result_path": f"baselines/{role}.json",
            }
        )
        for peak_lr in PEAK_LRS:
            trial_id = f"{role}-lr-{_lr_slug(peak_lr)}-seed-{SEED}"
            trials.append(
                {
                    "trial_id": trial_id,
                    "candidate_role": role,
                    "parent_weights": candidate,
                    "peak_lr": peak_lr,
                    "seed": SEED,
                    "run_directory": f"trials/{trial_id}/training/run",
                    "checkpoint_directory": (
                        f"trials/{trial_id}/training/checkpoints"
                    ),
                    "loss_report_path": f"trials/{trial_id}/loss.json",
                    "generation_report_path": (
                        f"trials/{trial_id}/generation.json"
                    ),
                    "result_path": f"trials/{trial_id}/result.json",
                    "purpose": "lr_calibration_only",
                    "eligible_for_formal_continuation": False,
                    "resume_scope": "same_trial_only",
                }
            )

    plan = {
        "schema_version": "1.0",
        "pipeline": PLAN_PIPELINE,
        "created_at": _utc_now(),
        "spec": {
            "path": str(Path(spec_path).resolve()),
            "sha256": spec_digest,
        },
        "output_root": str(root),
        "inputs": spec["inputs"],
        "tokenizer": spec["tokenizer"],
        "training": spec["training"],
        "evaluation": spec["evaluation"],
        "tracking": spec["tracking"],
        "formal_training_policy": spec["formal_training_policy"],
        "baselines": baselines,
        "trials": trials,
        "execution_order": [
            *[item["observation_id"] for item in baselines],
            *[item["trial_id"] for item in trials],
        ],
        "complete": True,
    }
    _write_immutable_json(output, plan)
    return plan


def _load_result(path: Path, pipeline: str) -> tuple[dict[str, Any], str]:
    payload, digest = _load_verified(path, "LR 校准结果")
    if payload.get("pipeline") != pipeline or payload.get("complete") is not True:
        raise ValueError(f"LR 校准结果状态无效: {path}")
    return payload, digest


def summarize(
    *, plan_path: str | Path, run_root: str | Path, output: str | Path
) -> dict[str, Any]:
    """要求 2 个 baseline 和 6 个 trial 完整后发布无自动赢家汇总。"""

    plan, plan_digest = _load_verified(plan_path, "基础法律 LR 校准 plan")
    if plan.get("pipeline") != PLAN_PIPELINE or plan.get("complete") is not True:
        raise ValueError("基础法律 LR 校准 plan 无效")
    root = Path(run_root).resolve()
    if str(root) != plan.get("output_root"):
        raise ValueError("LR 校准 run root 与 plan 不一致")

    baseline_results = []
    for item in plan["baselines"]:
        path = root / item["result_path"]
        result, digest = _load_result(path, BASELINE_PIPELINE)
        if (
            result.get("observation_id") != item["observation_id"]
            or result.get("plan_sha256") != plan_digest
        ):
            raise ValueError("baseline 结果身份不一致")
        baseline_results.append({"path": str(path), "sha256": digest, **result})

    trial_results = []
    for item in plan["trials"]:
        path = root / item["result_path"]
        result, digest = _load_result(path, TRIAL_PIPELINE)
        if (
            result.get("trial_id") != item["trial_id"]
            or result.get("plan_sha256") != plan_digest
            or result.get("purpose") != "lr_calibration_only"
            or result.get("eligible_for_formal_continuation") is not False
        ):
            raise ValueError("trial 结果身份或禁止续训标记无效")
        trial_results.append({"path": str(path), "sha256": digest, **result})

    grouped = []
    for peak_lr in PEAK_LRS:
        members = [item for item in trial_results if item["peak_lr"] == peak_lr]
        if [item["candidate_role"] for item in members] != list(CALIBRATION_ROLES):
            raise ValueError("每档 LR 必须同时覆盖 Stage B 与 CPT final")
        grouped.append({"peak_lr": peak_lr, "trials": members})

    report = {
        "schema_version": "1.0",
        "pipeline": SUMMARY_PIPELINE,
        "created_at": _utc_now(),
        "plan": {
            "path": str(Path(plan_path).resolve()),
            "sha256": plan_digest,
        },
        "coverage": {
            "expected_baselines": 2,
            "completed_baselines": len(baseline_results),
            "expected_trials": 6,
            "completed_trials": len(trial_results),
        },
        "baselines": baseline_results,
        "learning_rates": grouped,
        "decision": {
            "automatic_winner": None,
            "requires_human_generation_review": True,
            "fallback_if_indistinguishable": plan["evaluation"]["fallback_peak_lr"],
            "formal_training_must_restart_from_original_cpt": True,
        },
        "private_holdout_evaluated": False,
        "complete": True,
    }
    _write_immutable_json(output, report)
    return report


def publish_decision(
    *, summary_path: str | Path, selected_peak_lr: float, output: str | Path
) -> dict[str, Any]:
    """把人工批准的共同 peak LR 与完整校准证据绑定。"""

    summary, summary_digest = _load_verified(
        summary_path, "基础法律 LR 校准 summary"
    )
    if (
        summary.get("schema_version") != "1.0"
        or summary.get("pipeline") != SUMMARY_PIPELINE
        or summary.get("coverage")
        != {
            "expected_baselines": 2,
            "completed_baselines": 2,
            "expected_trials": 6,
            "completed_trials": 6,
        }
        or summary.get("complete") is not True
        or summary.get("private_holdout_evaluated") is not False
    ):
        raise ValueError("LR 校准 summary 尚未完整闭合")
    if selected_peak_lr not in PEAK_LRS:
        raise ValueError("选择的 peak LR 不属于已校准候选")
    groups = summary.get("learning_rates")
    if not isinstance(groups, list):
        raise ValueError("LR 校准 summary 缺少三档结果")
    selected = [item for item in groups if item.get("peak_lr") == selected_peak_lr]
    if len(selected) != 1 or len(selected[0].get("trials", [])) != 2:
        raise ValueError("选择的 peak LR 没有同时覆盖两个校准端点")
    for trial in selected[0]["trials"]:
        stability = trial.get("training_stability")
        if (
            not isinstance(stability, dict)
            or stability.get("nonfinite_metrics") is not False
            or trial.get("eligible_for_formal_continuation") is not False
            or trial.get("complete") is not True
        ):
            raise ValueError("选择的 peak LR 包含无效或可续训校准 trial")

    decision = {
        "schema_version": "1.0",
        "pipeline": DECISION_PIPELINE,
        "created_at": _utc_now(),
        "calibration_summary": {
            "path": str(Path(summary_path).resolve()),
            "sha256": summary_digest,
        },
        "decision": {
            "status": "approved",
            "selected_peak_lr": selected_peak_lr,
            "selection_scope": "current_three_lr_candidates_and_two_parent_endpoints",
            "evidence_dimensions": [
                "training_stability",
                "legal_full_loss",
                "general_full_loss",
                "fixed_generation_structure",
            ],
            "global_optimum_claimed": False,
        },
        "formal_training_policy": {
            "calibration_weights_eligible": False,
            "must_restart_from_original_cpt_model_only": True,
            "shared_peak_lr_across_all_five_candidates": True,
        },
        "private_holdout_used": False,
        "complete": True,
    }
    _write_immutable_json(output, decision)
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基础法律 SFT 共同 LR 校准计划")
    commands = parser.add_subparsers(dest="command", required=True)
    spec = commands.add_parser("spec", help="发布校准 spec")
    spec.add_argument("--stage-b-control", required=True)
    spec.add_argument("--cpt-final", required=True)
    spec.add_argument("--data-manifest", required=True)
    spec.add_argument("--evaluation-manifest", required=True)
    spec.add_argument("--general-validation", required=True)
    spec.add_argument("--generation-manifest", required=True)
    spec.add_argument("--tokenizer-path", required=True)
    spec.add_argument("--output", required=True)

    plan = commands.add_parser("prepare", help="发布六 trial 校准 plan")
    plan.add_argument("--spec", required=True)
    plan.add_argument("--output-root", required=True)
    plan.add_argument("--output", required=True)

    summary = commands.add_parser("summarize", help="汇总完整校准结果")
    summary.add_argument("--plan", required=True)
    summary.add_argument("--run-root", required=True)
    summary.add_argument("--output", required=True)

    decision = commands.add_parser("decide", help="发布人工批准的共同 LR 决策")
    decision.add_argument("--summary", required=True)
    decision.add_argument("--selected-peak-lr", type=float, required=True)
    decision.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "spec":
            result = create_spec(
                stage_b_control=args.stage_b_control,
                cpt_final=args.cpt_final,
                data_manifest=args.data_manifest,
                evaluation_manifest=args.evaluation_manifest,
                general_validation=args.general_validation,
                generation_manifest=args.generation_manifest,
                tokenizer_path=args.tokenizer_path,
                output=args.output,
            )
            print(
                "SFT_BASE_LR_CALIBRATION_SPEC_OK "
                f"train_tokens={result['training']['full_epoch_assistant_tokens']} "
                f"calibration_tokens={result['training']['calibration_stop_assistant_tokens']}"
            )
        elif args.command == "prepare":
            result = prepare_plan(
                spec_path=args.spec, output_root=args.output_root, output=args.output
            )
            print(
                "SFT_BASE_LR_CALIBRATION_PLAN_OK "
                f"baselines={len(result['baselines'])} trials={len(result['trials'])}"
            )
        elif args.command == "summarize":
            result = summarize(
                plan_path=args.plan, run_root=args.run_root, output=args.output
            )
            print(
                "SFT_BASE_LR_CALIBRATION_SUMMARY_OK "
                f"trials={result['coverage']['completed_trials']}"
            )
        else:
            result = publish_decision(
                summary_path=args.summary,
                selected_peak_lr=args.selected_peak_lr,
                output=args.output,
            )
            print(
                "SFT_BASE_LR_CALIBRATION_DECISION_OK "
                f"peak_lr={result['decision']['selected_peak_lr']}"
            )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        raise SystemExit(f"[失败] {error}") from error


if __name__ == "__main__":
    main()
