"""发布和汇总正式基础法律 SFT 五候选父权重比较。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import sft_base_lr_calibration as calibration
from . import train_full_sft as training


SPEC_PIPELINE = "legal_sft_base_parent_formal_spec_v1"
PLAN_PIPELINE = "legal_sft_base_parent_formal_plan_v1"
OBSERVATION_PIPELINE = "legal_sft_base_parent_formal_observation_v1"
CANDIDATE_PIPELINE = "legal_sft_base_parent_formal_candidate_v1"
SUMMARY_PIPELINE = "legal_sft_base_parent_formal_summary_v1"
CANDIDATE_ROLES = (
    "stage_b_control",
    "cpt_250m",
    "cpt_750m",
    "cpt_2b",
    "cpt_final",
)
PARENT_CPT_INPUT_TOKENS = {
    "stage_b_control": 0,
    "cpt_250m": 250_085_376,
    "cpt_750m": 750_059_520,
    "cpt_2b": 2_000_093_184,
    "cpt_final": 2_447_821_056,
}
OBSERVATION_PHASES = ("parent", "base_25", "base_50", "base_100")


def _load_verified(path: str | Path, description: str) -> tuple[dict[str, Any], str]:
    return calibration._load_verified(path, description)


def _candidate(role: str, path: str | Path) -> dict[str, Any]:
    identity = calibration._identity(path)
    identity.update(
        {
            "role": role,
            "cpt_input_tokens": PARENT_CPT_INPUT_TOKENS[role],
            "load_semantics": "strict_original_parent_model_state_only",
        }
    )
    return identity


def _load_calibration_chain(
    decision_path: str | Path,
) -> tuple[dict[str, Any], str, dict[str, Any], str, dict[str, Any], str]:
    decision, decision_digest = _load_verified(
        decision_path, "基础法律共同 LR decision"
    )
    policy = decision.get("formal_training_policy")
    if (
        decision.get("schema_version") != "1.0"
        or decision.get("pipeline") != calibration.DECISION_PIPELINE
        or decision.get("decision", {}).get("status") != "approved"
        or decision.get("decision", {}).get("global_optimum_claimed") is not False
        or not isinstance(policy, dict)
        or policy.get("calibration_weights_eligible") is not False
        or policy.get("must_restart_from_original_cpt_model_only") is not True
        or policy.get("shared_peak_lr_across_all_five_candidates") is not True
        or decision.get("private_holdout_used") is not False
        or decision.get("complete") is not True
    ):
        raise ValueError("共同 LR decision 状态或正式重启约束无效")

    summary_identity = decision.get("calibration_summary")
    if not isinstance(summary_identity, dict):
        raise ValueError("共同 LR decision 缺少 calibration summary 身份")
    summary, summary_digest = _load_verified(
        summary_identity.get("path", ""), "基础法律 LR 校准 summary"
    )
    if (
        summary_digest != summary_identity.get("sha256")
        or summary.get("pipeline") != calibration.SUMMARY_PIPELINE
        or summary.get("complete") is not True
    ):
        raise ValueError("共同 LR decision 与 calibration summary 身份不一致")

    plan_identity = summary.get("plan")
    if not isinstance(plan_identity, dict):
        raise ValueError("LR 校准 summary 缺少 plan 身份")
    calibration_plan, calibration_plan_digest = _load_verified(
        plan_identity.get("path", ""), "基础法律 LR 校准 plan"
    )
    if (
        calibration_plan_digest != plan_identity.get("sha256")
        or calibration_plan.get("pipeline") != calibration.PLAN_PIPELINE
        or calibration_plan.get("complete") is not True
    ):
        raise ValueError("LR 校准 summary 与 plan 身份不一致")
    return (
        decision,
        decision_digest,
        summary,
        summary_digest,
        calibration_plan,
        calibration_plan_digest,
    )


def _verify_calibration_inputs(plan: Mapping[str, Any]) -> None:
    for name, identity in plan.get("inputs", {}).items():
        calibration._verify_identity(identity, f"正式比较输入 {name}")
    tokenizer_identity = plan.get("tokenizer")
    if not isinstance(tokenizer_identity, dict):
        raise ValueError("LR 校准 plan 缺少 Tokenizer 身份")
    tokenizer = training._load_tokenizer(tokenizer_identity.get("path", ""))
    if training._tokenizer_identity(tokenizer, tokenizer_identity["path"]) != tokenizer_identity:
        raise ValueError("正式比较 Tokenizer 身份发生漂移")


def create_spec(
    *,
    decision_path: str | Path,
    stage_b_control: str | Path,
    cpt_250m: str | Path,
    cpt_750m: str | Path,
    cpt_2b: str | Path,
    cpt_final: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """从共同 LR 决策发布五个原始 CPT 父权重的正式 spec。"""

    (
        decision,
        decision_digest,
        _summary,
        summary_digest,
        calibration_plan,
        calibration_plan_digest,
    ) = _load_calibration_chain(decision_path)
    _verify_calibration_inputs(calibration_plan)
    selected_peak_lr = decision["decision"]["selected_peak_lr"]
    if selected_peak_lr not in calibration.PEAK_LRS:
        raise ValueError("共同 LR decision 的选择不属于原校准三档")

    candidates = [
        _candidate("stage_b_control", stage_b_control),
        _candidate("cpt_250m", cpt_250m),
        _candidate("cpt_750m", cpt_750m),
        _candidate("cpt_2b", cpt_2b),
        _candidate("cpt_final", cpt_final),
    ]
    calibration_candidates = {
        item["candidate_role"]: item["weights"]
        for item in calibration_plan.get("baselines", [])
    }
    for candidate in (candidates[0], candidates[-1]):
        calibrated = calibration_candidates.get(candidate["role"])
        if (
            not isinstance(calibrated, dict)
            or calibrated.get("sha256") != candidate["sha256"]
            or calibrated.get("bytes") != candidate["bytes"]
        ):
            raise ValueError("正式端点父权重与共同 LR 校准时不一致")
    if len({item["sha256"] for item in candidates}) != len(candidates):
        raise ValueError("五个正式父权重必须具有不同身份")

    calibrated_training = calibration_plan["training"]
    full_tokens = calibrated_training["full_epoch_assistant_tokens"]
    base_25 = full_tokens * 25 // 100
    base_50 = full_tokens * 50 // 100
    if not 0 < base_25 < base_50 < full_tokens:
        raise ValueError("正式单遍观察点无效")
    training_config = {
        **calibrated_training,
        "peak_lr": selected_peak_lr,
        "stop_assistant_tokens": full_tokens,
        "schedule_assistant_tokens": full_tokens,
        "observation_thresholds": {
            "base_25": base_25,
            "base_50": base_50,
            "base_100": full_tokens,
        },
    }
    training_config.pop("peak_lrs", None)
    training_config.pop("calibration_percent", None)
    training_config.pop("calibration_stop_assistant_tokens", None)

    spec = {
        "schema_version": "1.0",
        "pipeline": SPEC_PIPELINE,
        "created_at": calibration._utc_now(),
        "scope": {
            "task": "formal_base_legal_sft_parent_evaluation",
            "query_sft_in_scope": False,
            "rag_sft_in_scope": False,
            "private_holdout_used": False,
        },
        "calibration": {
            "decision": {
                "path": str(Path(decision_path).resolve()),
                "sha256": decision_digest,
            },
            "summary_sha256": summary_digest,
            "plan_sha256": calibration_plan_digest,
        },
        "candidates": candidates,
        "inputs": calibration_plan["inputs"],
        "tokenizer": calibration_plan["tokenizer"],
        "training": training_config,
        "observations": {
            "parent": {
                "requested_assistant_tokens": 0,
                "loss_suites": ["general", "legal_full"],
                "fixed_generation": True,
            },
            "base_25": {
                "requested_assistant_tokens": base_25,
                "loss_suites": ["general", "legal_quick"],
                "fixed_generation": False,
            },
            "base_50": {
                "requested_assistant_tokens": base_50,
                "loss_suites": ["general", "legal_full"],
                "fixed_generation": True,
            },
            "base_100": {
                "requested_assistant_tokens": full_tokens,
                "loss_suites": ["general", "legal_full"],
                "fixed_generation": True,
            },
        },
        "tracking": {
            "swanlab": {
                **calibration_plan["tracking"]["swanlab"],
                "experiment_name_source": "candidate_run_id",
            }
        },
        "formal_training_policy": {
            "seed": calibration.SEED,
            "continuous_single_run_per_candidate": True,
            "must_start_from_original_cpt_model_only": True,
            "same_candidate_resume_only": True,
            "intermediate_model_only_continuation_forbidden": True,
            "calibration_checkpoint_continuation_forbidden": True,
            "all_candidates_must_reach_base_100": True,
        },
        "complete": True,
    }
    calibration._write_immutable_json(output, spec)
    return spec


def prepare_plan(
    *, spec_path: str | Path, output_root: str | Path, output: str | Path
) -> dict[str, Any]:
    """把正式 spec 编译为五条独立、连续的完整单遍训练链。"""

    spec, spec_digest = _load_verified(spec_path, "正式基础法律父权重 spec")
    if (
        spec.get("schema_version") != "1.0"
        or spec.get("pipeline") != SPEC_PIPELINE
        or spec.get("complete") is not True
        or [item.get("role") for item in spec.get("candidates", [])]
        != list(CANDIDATE_ROLES)
    ):
        raise ValueError("正式基础法律父权重 spec 无效")
    root = Path(output_root).resolve()
    candidates = []
    for parent in spec["candidates"]:
        role = parent["role"]
        candidate_id = f"{role}-seed-{spec['training']['seed']}"
        observations = []
        for phase in OBSERVATION_PHASES:
            definition = spec["observations"][phase]
            observations.append(
                {
                    "phase": phase,
                    **definition,
                    "result_path": (
                        f"candidates/{candidate_id}/observations/{phase}.json"
                    ),
                }
            )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "candidate_role": role,
                "parent_weights": parent,
                "run_directory": f"candidates/{candidate_id}/training/run",
                "checkpoint_directory": (
                    f"candidates/{candidate_id}/training/checkpoints"
                ),
                "observations": observations,
                "result_path": f"candidates/{candidate_id}/result.json",
                "resume_scope": "same_candidate_formal_run_only",
                "eligible_for_formal_base_100": True,
            }
        )

    plan = {
        "schema_version": "1.0",
        "pipeline": PLAN_PIPELINE,
        "created_at": calibration._utc_now(),
        "spec": {
            "path": str(Path(spec_path).resolve()),
            "sha256": spec_digest,
        },
        "output_root": str(root),
        "calibration": spec["calibration"],
        "inputs": spec["inputs"],
        "tokenizer": spec["tokenizer"],
        "training": spec["training"],
        "tracking": spec["tracking"],
        "formal_training_policy": spec["formal_training_policy"],
        "candidates": candidates,
        "execution_order": [item["candidate_id"] for item in candidates],
        "private_holdout_used": False,
        "complete": True,
    }
    calibration._write_immutable_json(output, plan)
    return plan


def summarize(
    *, plan_path: str | Path, run_root: str | Path, output: str | Path
) -> dict[str, Any]:
    """汇总五个完整候选，不自动选择父权重赢家。"""

    plan, plan_digest = _load_verified(plan_path, "正式基础法律父权重 plan")
    root = Path(run_root).resolve()
    if (
        plan.get("pipeline") != PLAN_PIPELINE
        or plan.get("complete") is not True
        or plan.get("output_root") != str(root)
    ):
        raise ValueError("正式基础法律父权重 plan 或 run root 无效")
    results = []
    for item in plan["candidates"]:
        path = root / item["result_path"]
        result, digest = _load_verified(path, "正式基础法律候选结果")
        if (
            result.get("pipeline") != CANDIDATE_PIPELINE
            or result.get("plan_sha256") != plan_digest
            or result.get("candidate_id") != item["candidate_id"]
            or result.get("candidate_role") != item["candidate_role"]
            or result.get("formal_base_100_eligible") is not True
            or result.get("private_holdout_evaluated") is not False
            or result.get("complete") is not True
            or [entry.get("phase") for entry in result.get("observations", [])]
            != list(OBSERVATION_PHASES)
        ):
            raise ValueError("正式基础法律候选结果身份或观察点无效")
        results.append({"path": str(path), "sha256": digest, **result})

    report = {
        "schema_version": "1.0",
        "pipeline": SUMMARY_PIPELINE,
        "created_at": calibration._utc_now(),
        "plan": {
            "path": str(Path(plan_path).resolve()),
            "sha256": plan_digest,
        },
        "coverage": {
            "expected_candidates": len(CANDIDATE_ROLES),
            "completed_candidates": len(results),
            "expected_observations": len(CANDIDATE_ROLES)
            * len(OBSERVATION_PHASES),
            "completed_observations": sum(
                len(item["observations"]) for item in results
            ),
        },
        "candidates": results,
        "decision": {
            "automatic_winner": None,
            "requires_anonymous_generation_review": True,
            "requires_paired_analysis": True,
            "seed_43_top_two_pending": True,
        },
        "private_holdout_evaluated": False,
        "complete": True,
    }
    calibration._write_immutable_json(output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="正式基础法律 SFT 五候选父权重比较")
    commands = parser.add_subparsers(dest="command", required=True)
    spec = commands.add_parser("spec", help="发布正式五候选 spec")
    spec.add_argument("--decision", required=True)
    spec.add_argument("--stage-b-control", required=True)
    spec.add_argument("--cpt-250m", required=True)
    spec.add_argument("--cpt-750m", required=True)
    spec.add_argument("--cpt-2b", required=True)
    spec.add_argument("--cpt-final", required=True)
    spec.add_argument("--output", required=True)

    plan = commands.add_parser("prepare", help="发布正式五候选 plan")
    plan.add_argument("--spec", required=True)
    plan.add_argument("--output-root", required=True)
    plan.add_argument("--output", required=True)

    summary = commands.add_parser("summarize", help="汇总五候选完整结果")
    summary.add_argument("--plan", required=True)
    summary.add_argument("--run-root", required=True)
    summary.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "spec":
            result = create_spec(
                decision_path=args.decision,
                stage_b_control=args.stage_b_control,
                cpt_250m=args.cpt_250m,
                cpt_750m=args.cpt_750m,
                cpt_2b=args.cpt_2b,
                cpt_final=args.cpt_final,
                output=args.output,
            )
            print(
                "SFT_BASE_PARENT_FORMAL_SPEC_OK "
                f"candidates={len(result['candidates'])} "
                f"peak_lr={result['training']['peak_lr']}"
            )
        elif args.command == "prepare":
            result = prepare_plan(
                spec_path=args.spec,
                output_root=args.output_root,
                output=args.output,
            )
            print(
                "SFT_BASE_PARENT_FORMAL_PLAN_OK "
                f"candidates={len(result['candidates'])}"
            )
        else:
            result = summarize(
                plan_path=args.plan,
                run_root=args.run_root,
                output=args.output,
            )
            print(
                "SFT_BASE_PARENT_FORMAL_SUMMARY_OK "
                f"candidates={result['coverage']['completed_candidates']}"
            )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        raise SystemExit(f"[失败] {error}") from error


if __name__ == "__main__":
    main()
