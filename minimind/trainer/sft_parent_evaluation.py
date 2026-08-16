"""发布法律 SFT 父权重比较计划并汇总无原文 trial 指标。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .sft_parent_rag_metrics import RAG_METRICS


SPEC_PIPELINE = "legal_sft_parent_evaluation_spec_v4"
PLAN_PIPELINE = "legal_sft_parent_evaluation_plan_v4"
TRIAL_RESULT_PIPELINE = "legal_sft_parent_trial_result_v4"
SUMMARY_PIPELINE = "legal_sft_parent_evaluation_summary_v4"
MIGRATION_PIPELINE = "legal_sft_parent_result_migration_v3_to_v4"
LEGACY_PLAN_PIPELINE = "legal_sft_parent_evaluation_plan_v3"
LEGACY_TRIAL_RESULT_PIPELINE = "legal_sft_parent_trial_result_v3"
LEGACY_PARENT_ROLES = (
    "stage_b_control",
    "cpt_250m",
    "cpt_750m",
    "cpt_2b",
)
REQUIRED_PARENT_ROLES = (
    *LEGACY_PARENT_ROLES,
    "cpt_final",
)
PARENT_CPT_INPUT_TOKENS = {
    "stage_b_control": 0,
    "cpt_250m": 250_085_376,
    "cpt_750m": 750_059_520,
    "cpt_2b": 2_000_093_184,
    "cpt_final": 2_447_821_056,
}
REQUIRED_INPUTS = (
    "base_sft_data",
    "general_validation",
    "evaluation_exclusions",
    "rag_sft_data",
    "rag_development",
    "rag_holdout",
    "rag_pair_evaluation",
)
FIRST_ROUND_PHASES = (
    "baseline",
    "base_250k",
    "base_500k",
    "base_1m",
    "rag_epoch_1",
    "rag_epoch_2",
)
PHASE_SUITES = {
    "baseline": ("general", "legal_full", "rag_development_diagnostic"),
    "base_250k": ("general", "legal_quick"),
    "base_500k": ("general", "legal_quick"),
    "base_1m": ("general", "legal_full"),
    "rag_epoch_1": ("general", "legal_full", "rag_development"),
    "rag_epoch_2": ("general", "legal_full", "rag_development"),
}
LOSS_SUITES = ("general", "legal_quick", "legal_full")
RAG_SUITES = ("rag_development_diagnostic", "rag_development")
LOSS_METRICS = ("loss", "perplexity")
BASE_PHASE_TOKEN_BUDGETS = {
    "base_250k": 250_000,
    "base_500k": 500_000,
    "base_1m": 1_000_000,
}
RAG_PHASE_TOKEN_BUDGETS = {
    "rag_epoch_1": 67_316,
    "rag_epoch_2": 134_632,
}
_EVALUATION_EXCLUSIONS_PIPELINE = "legal_sft_evaluation_exclusions"
_RAG_SFT_RELEASE_PIPELINE = "rag_sft_training_release"
_RAG_PAIR_EVALUATION_PIPELINE = "legal_rag_parent_pair_evaluation_v2"
_GENERAL_VALIDATION_PIPELINE = "stage_b_token_shards_v1"
_GENERAL_VALIDATION_SOURCES = ("wikipedia", "fineweb", "minimind")
_EVALUATION_ASSET_NAMES = {
    "rag_development": "project_rag_eval",
    "rag_holdout": "project_private_holdout_v2",
}
_METRIC_NAME_RE = re.compile(r"[a-z][a-z0-9_]*(?:/[a-z][a-z0-9_]*)*")
_SAFE_NAME_RE = re.compile(r"[a-z][a-z0-9_.-]*")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _expect_fields(
    payload: dict[str, Any], expected: set[str], description: str
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{description} 必须使用封闭 schema")


def _load_verified_json(path: str | Path, description: str) -> tuple[dict[str, Any], str]:
    resolved = Path(path).resolve()
    hash_path = resolved.with_suffix(".sha256")
    if not resolved.is_file() or not hash_path.is_file():
        raise FileNotFoundError(f"{description} 或相邻 SHA-256 清单不存在: {resolved}")
    digest = _sha256_file(resolved)
    expected = f"{digest}  {resolved.name}"
    lines = [
        line
        for line in hash_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if lines != [expected]:
        raise ValueError(f"{description} SHA-256 校验失败")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} 不是有效 UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} 必须是 JSON object")
    return payload, digest


def _write_immutable_json(path: str | Path, payload: dict[str, Any]) -> str:
    resolved = Path(path).resolve()
    hash_path = resolved.with_suffix(".sha256")
    if resolved.exists() or hash_path.exists():
        raise FileExistsError(f"输出已存在，不能覆盖: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(serialized, encoding="utf-8", newline="\n")
    temporary.replace(resolved)
    digest = _sha256_file(resolved)
    hash_temporary = hash_path.with_suffix(hash_path.suffix + ".tmp")
    hash_temporary.write_text(
        f"{digest}  {resolved.name}\n", encoding="utf-8", newline="\n"
    )
    hash_temporary.replace(hash_path)
    return digest


def _positive_int(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{description} 必须是正整数")
    return value


def _non_negative_int(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{description} 必须是非负整数")
    return value


def _finite_number(value: object, description: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{description} 必须是有限数值")
    parsed = float(value)
    if not math.isfinite(parsed) or (positive and parsed <= 0):
        raise ValueError(f"{description} 必须是有限数值")
    return parsed


def _validate_spec(spec: dict[str, Any]) -> None:
    _expect_fields(
        spec,
        {
            "schema_version",
            "pipeline",
            "round",
            "candidates",
            "inputs",
            "tokenizer_path",
            "seeds",
            "phases",
            "training",
            "evaluation",
        },
        "父权重评估 spec",
    )
    if spec["schema_version"] != "1.0" or spec["pipeline"] != SPEC_PIPELINE:
        raise ValueError("父权重评估 spec 版本或 pipeline 无效")
    if spec["round"] != "first_round":
        raise ValueError("当前入口只发布第一轮父权重比较")
    if spec["phases"] != list(FIRST_ROUND_PHASES):
        raise ValueError("第一轮必须覆盖六个固定观察点")
    seeds = spec["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds 必须是非空数组")
    parsed_seeds = [_non_negative_int(seed, "seed") for seed in seeds]
    if len(set(parsed_seeds)) != len(parsed_seeds):
        raise ValueError("seeds 不能重复")
    if parsed_seeds != [42]:
        raise ValueError("第一轮必须固定使用 seed 42")


def _candidate_identities(candidates: object) -> list[dict[str, Any]]:
    if not isinstance(candidates, list):
        raise ValueError("candidates 必须是数组")
    by_role: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("父权重候选必须是 JSON object")
        _expect_fields(
            candidate, {"role", "cpt_input_tokens", "path", "sha256"}, "父权重候选"
        )
        role = candidate["role"]
        if not isinstance(role, str) or role in by_role:
            raise ValueError("父权重候选必须覆盖五个固定角色且不能重复")
        path = Path(candidate["path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"父权重不存在: {path}")
        actual = _sha256_file(path)
        if candidate["sha256"] != actual:
            raise ValueError(f"父权重 SHA-256 不匹配: {role}")
        expected_tokens = PARENT_CPT_INPUT_TOKENS.get(role)
        if candidate["cpt_input_tokens"] != expected_tokens:
            raise ValueError(f"父权重 CPT token 位置无效: {role}")
        by_role[role] = {
            "role": role,
            "cpt_input_tokens": expected_tokens,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": actual,
        }
    if set(by_role) != set(REQUIRED_PARENT_ROLES):
        raise ValueError("父权重候选必须覆盖五个固定角色")
    return [by_role[role] for role in REQUIRED_PARENT_ROLES]


def _manifest_identity(path: Path, digest: str, pipeline: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "pipeline": pipeline,
    }


def _general_validation_identity(
    path: Path, payload: dict[str, Any], digest: str
) -> dict[str, Any]:
    """校验并冻结 Stage B full validation 的实际 shard 身份。"""

    if payload.get("schema_version") != "1.0":
        raise ValueError("通用验证 shard manifest schema_version 无效")
    packing = payload.get("packing")
    if not isinstance(packing, dict) or packing.get("sequence_length") != 768:
        raise ValueError("通用验证必须固定使用 sequence_length=768")
    output = payload.get("output")
    if (
        not isinstance(output, dict)
        or not isinstance(output.get("root"), str)
        or not output["root"]
    ):
        raise ValueError("通用验证 shard manifest 缺少 output.root")
    output_root = Path(output["root"]).resolve()
    sources = payload.get("sources")
    if not isinstance(sources, dict) or tuple(sources) != _GENERAL_VALIDATION_SOURCES:
        raise ValueError("通用验证必须按冻结顺序覆盖 wikipedia、fineweb、minimind")

    source_identities: dict[str, Any] = {}
    total_sequences = 0
    total_tokens = 0
    total_shards = 0
    for source in _GENERAL_VALIDATION_SOURCES:
        source_report = sources[source]
        if not isinstance(source_report, dict):
            raise ValueError(f"通用验证来源 {source} 必须是 JSON object")
        stream = source_report.get("full_validation")
        if not isinstance(stream, dict):
            raise ValueError(f"通用验证来源 {source} 缺少 full_validation")
        sequence_count = _positive_int(
            stream.get("sequence_count"), f"{source} full_validation sequence_count"
        )
        written_tokens = _positive_int(
            stream.get("written_tokens"), f"{source} full_validation written_tokens"
        )
        shard_count = _positive_int(
            stream.get("shard_count"), f"{source} full_validation shard_count"
        )
        shards = stream.get("shards")
        if not isinstance(shards, list) or len(shards) != shard_count:
            raise ValueError(f"通用验证来源 {source} shard_count 不闭合")

        shard_identities = []
        source_sequences = 0
        source_tokens = 0
        for index, shard in enumerate(shards):
            location = f"{source} full_validation shards[{index}]"
            if not isinstance(shard, dict):
                raise ValueError(f"{location} 必须是 JSON object")
            relative_text = shard.get("path")
            if not isinstance(relative_text, str):
                raise ValueError(f"{location}.path 必须是字符串")
            relative_path = PurePosixPath(relative_text)
            expected_prefix = ("validation", "full", source)
            if (
                relative_path.is_absolute()
                or ".." in relative_path.parts
                or relative_path.parts[:3] != expected_prefix
                or relative_path.suffix != ".npy"
            ):
                raise ValueError(f"{location}.path 不属于冻结 full validation")
            shard_path = (output_root / Path(*relative_path.parts)).resolve()
            if not shard_path.is_file():
                raise FileNotFoundError(f"通用验证 shard 不存在: {shard_path}")
            expected_bytes = _positive_int(shard.get("bytes"), f"{location}.bytes")
            expected_digest = shard.get("sha256")
            if (
                not isinstance(expected_digest, str)
                or _SHA256_RE.fullmatch(expected_digest) is None
            ):
                raise ValueError(f"{location}.sha256 无效")
            token_count = _positive_int(
                shard.get("token_count"), f"{location}.token_count"
            )
            shard_sequences = _positive_int(
                shard.get("sequence_count"), f"{location}.sequence_count"
            )
            if (
                shard.get("dtype") != "uint16"
                or shard.get("shape") != [token_count]
                or token_count % 768
                or shard_sequences != token_count // 768
            ):
                raise ValueError(f"{location} 格式或 sequence 计数无效")
            if shard_path.stat().st_size != expected_bytes:
                raise ValueError(f"通用验证 shard 大小漂移: {shard_path}")
            if _sha256_file(shard_path) != expected_digest:
                raise ValueError(f"通用验证 shard SHA-256 漂移: {shard_path}")
            shard_identities.append(
                {
                    "path": relative_text,
                    "bytes": expected_bytes,
                    "sha256": expected_digest,
                    "token_count": token_count,
                    "sequence_count": shard_sequences,
                }
            )
            source_tokens += token_count
            source_sequences += shard_sequences
        if source_tokens != written_tokens or source_sequences != sequence_count:
            raise ValueError(f"通用验证来源 {source} token 或 sequence 计数不闭合")
        source_identities[source] = {
            "sequence_count": sequence_count,
            "written_tokens": written_tokens,
            "shards": shard_identities,
        }
        total_sequences += sequence_count
        total_tokens += written_tokens
        total_shards += shard_count

    totals = payload.get("totals")
    full_totals = totals.get("full_validation") if isinstance(totals, dict) else None
    if not isinstance(full_totals, dict) or any(
        full_totals.get(key) != expected
        for key, expected in (
            ("sequence_count", total_sequences),
            ("written_tokens", total_tokens),
            ("shard_count", total_shards),
        )
    ):
        raise ValueError("通用验证 totals.full_validation 计数不闭合")
    identity = _manifest_identity(path, digest, _GENERAL_VALIDATION_PIPELINE)
    identity.update(
        {
            "split": "full_validation",
            "sequence_length": 768,
            "sources": source_identities,
            "totals": {
                "sequence_count": total_sequences,
                "written_tokens": total_tokens,
                "shard_count": total_shards,
            },
        }
    )
    return identity


def _evaluation_asset_identity(
    payload: dict[str, Any], role: str
) -> dict[str, Any]:
    if (
        payload.get("schema_version") != "1.2"
        or payload.get("pipeline") != _EVALUATION_EXCLUSIONS_PIPELINE
        or payload.get("complete_for_formal_sft") is not True
    ):
        raise ValueError("正式评估排除 manifest 版本或完成状态无效")
    assets = payload.get("assets")
    if not isinstance(assets, list):
        raise ValueError("正式评估排除 manifest 缺少 assets")
    expected_name = _EVALUATION_ASSET_NAMES[role]
    matches = [asset for asset in assets if isinstance(asset, dict) and asset.get("name") == expected_name]
    if len(matches) != 1:
        raise ValueError(f"正式评估排除 manifest 缺少 {expected_name} 资产")
    asset = matches[0]
    _expect_fields(
        asset,
        {"name", "path", "bytes", "sha256", "records", "question_fields"},
        f"评估资产 {expected_name}",
    )
    if (
        not isinstance(asset["path"], str)
        or not asset["path"]
        or not isinstance(asset["bytes"], int)
        or asset["bytes"] <= 0
        or not isinstance(asset["sha256"], str)
        or not _SHA256_RE.fullmatch(asset["sha256"])
        or not isinstance(asset["records"], int)
        or asset["records"] <= 0
        or not isinstance(asset["question_fields"], list)
        or not asset["question_fields"]
        or not all(isinstance(field, str) and field for field in asset["question_fields"])
    ):
        raise ValueError(f"评估资产 {expected_name} 身份无效")
    return {
        "name": asset["name"],
        "path": asset["path"],
        "bytes": asset["bytes"],
        "sha256": asset["sha256"],
        "records": asset["records"],
        "question_fields": asset["question_fields"],
    }


def _input_identities(inputs: object) -> dict[str, dict[str, Any]]:
    if not isinstance(inputs, dict) or set(inputs) != set(REQUIRED_INPUTS):
        raise ValueError("共享输入必须覆盖七个固定角色")
    identities: dict[str, dict[str, Any]] = {}
    loaded: dict[str, dict[str, Any]] = {}
    digests: dict[str, str] = {}
    resolved_paths: dict[str, Path] = {}
    for name in REQUIRED_INPUTS:
        definition = inputs[name]
        if not isinstance(definition, dict):
            raise ValueError(f"共享输入 {name} 必须是 JSON object")
        expected_fields = {"path", "pipeline"}
        if name in _EVALUATION_ASSET_NAMES:
            expected_fields.add("asset_name")
        if name == "general_validation":
            expected_fields.add("split")
        _expect_fields(definition, expected_fields, f"共享输入 {name}")
        expected_pipeline = definition["pipeline"]
        if not isinstance(expected_pipeline, str) or not _SAFE_NAME_RE.fullmatch(
            expected_pipeline
        ):
            raise ValueError(f"共享输入 {name} pipeline 无效")
        path = Path(definition["path"]).resolve()
        payload, digest = _load_verified_json(path, f"共享输入 {name}")
        if name == "general_validation":
            if (
                expected_pipeline != _GENERAL_VALIDATION_PIPELINE
                or definition["split"] != "full_validation"
            ):
                raise ValueError("通用验证输入类型或 split 无效")
        elif payload.get("pipeline") != expected_pipeline:
            raise ValueError(f"共享输入 {name} pipeline 无效")
        loaded[name] = payload
        digests[name] = digest
        resolved_paths[name] = path
        identities[name] = _manifest_identity(path, digest, expected_pipeline)
    identities["general_validation"] = _general_validation_identity(
        resolved_paths["general_validation"],
        loaded["general_validation"],
        digests["general_validation"],
    )
    base_readiness = loaded["base_sft_data"].get("readiness")
    rag_readiness = loaded["rag_sft_data"].get("readiness")
    if (
        loaded["base_sft_data"].get("schema_version") != "1.0"
        or loaded["base_sft_data"].get("complete") is not True
        or not isinstance(base_readiness, dict)
        or base_readiness.get("training_ready") is not True
    ):
        raise ValueError("基础法律 SFT 数据尚未正式就绪")
    if (
        loaded["rag_sft_data"].get("pipeline") != _RAG_SFT_RELEASE_PIPELINE
        or loaded["rag_sft_data"].get("release_status") != "formal_training_candidate"
        or loaded["rag_sft_data"].get("complete") is not True
        or not isinstance(rag_readiness, dict)
        or rag_readiness.get("training_ready") is not True
    ):
        raise ValueError("RAG SFT 数据尚未正式就绪")
    exclusion_path = resolved_paths["evaluation_exclusions"]
    if any(
        resolved_paths[name] != exclusion_path
        for name in ("rag_development", "rag_holdout")
    ):
        raise ValueError("项目 RAG 开发集与私有留出集必须绑定同一份评估排除 manifest")
    for role in ("rag_development", "rag_holdout"):
        if loaded[role] != loaded["evaluation_exclusions"]:
            raise ValueError("项目 RAG 评估资产必须来自正式评估排除 manifest")
        if inputs[role]["asset_name"] != _EVALUATION_ASSET_NAMES[role]:
            raise ValueError(f"项目 RAG {role} asset_name 无效")
        identities[role]["asset"] = _evaluation_asset_identity(loaded[role], role)
    _evaluation_asset_identity(loaded["evaluation_exclusions"], "rag_development")
    _evaluation_asset_identity(loaded["evaluation_exclusions"], "rag_holdout")
    pair_evaluation = loaded["rag_pair_evaluation"]
    if (
        pair_evaluation.get("schema_version") != "1.1"
        or pair_evaluation.get("pipeline") != _RAG_PAIR_EVALUATION_PIPELINE
        or pair_evaluation.get("complete") is not True
        or pair_evaluation.get("evaluation_exclusions_sha256")
        != digests["evaluation_exclusions"]
    ):
        raise ValueError("RAG 成对评估 manifest 尚未正式就绪")
    records = pair_evaluation.get("records")
    if not isinstance(records, dict):
        raise ValueError("RAG 成对评估 manifest 缺少 records")
    _expect_fields(
        records,
        {
            "full_system_cases",
            "model_cases",
            "eligible_model_cases",
            "excluded_overlength_cases",
            "sufficient_cases",
            "insufficient_cases",
            "paired_queries",
        },
        "RAG 成对评估 records",
    )
    if records["full_system_cases"] != 200 or records["model_cases"] != 140:
        raise ValueError("RAG 成对评估记录数与冻结开发集不一致")
    eligible = _positive_int(records["eligible_model_cases"], "可评估模型问题数")
    excluded = _non_negative_int(
        records["excluded_overlength_cases"], "长度排除问题数"
    )
    sufficient = _positive_int(records["sufficient_cases"], "充分证据样本数")
    insufficient = _non_negative_int(records["insufficient_cases"], "不充分证据样本数")
    pairs = _non_negative_int(records["paired_queries"], "成对问题数")
    if (
        eligible + excluded != records["model_cases"]
        or sufficient != eligible
        or insufficient != eligible
        or pairs != eligible
    ):
        raise ValueError("RAG 成对评估长度排除或成对计数无效")
    length_policy = pair_evaluation.get("length_policy")
    if not isinstance(length_policy, dict):
        raise ValueError("RAG 成对评估 manifest 缺少长度策略")
    _expect_fields(
        length_policy,
        {
            "fixed_max_seq_len",
            "max_new_tokens",
            "max_prompt_tokens",
            "selection",
            "tokenizer",
            "excluded_query_ids",
        },
        "RAG 成对评估长度策略",
    )
    excluded_ids = length_policy["excluded_query_ids"]
    if (
        length_policy["fixed_max_seq_len"] != 768
        or length_policy["max_new_tokens"] != 160
        or length_policy["max_prompt_tokens"] != 608
        or length_policy["selection"] != "max_complete_ordered_prefix"
        or not isinstance(excluded_ids, list)
        or len(excluded_ids) != excluded
        or len(set(excluded_ids)) != len(excluded_ids)
        or any(not isinstance(item, str) or not item for item in excluded_ids)
    ):
        raise ValueError("RAG 成对评估长度策略无效")
    pair_records = pair_evaluation.get("pair_records")
    if not isinstance(pair_records, dict) or set(pair_records) != {
        "path",
        "bytes",
        "sha256",
        "records",
    }:
        raise ValueError("RAG 成对评估 manifest 缺少记录文件身份")
    if (
        not isinstance(pair_records["path"], str)
        or not pair_records["path"]
        or Path(pair_records["path"]).name != pair_records["path"]
        or not isinstance(pair_records["bytes"], int)
        or isinstance(pair_records["bytes"], bool)
        or pair_records["bytes"] <= 0
        or not isinstance(pair_records["sha256"], str)
        or _SHA256_RE.fullmatch(pair_records["sha256"]) is None
        or pair_records["records"] != sufficient + insufficient
    ):
        raise ValueError("RAG 成对评估记录文件身份无效")
    identities["rag_pair_evaluation"]["records"] = records
    identities["rag_pair_evaluation"]["length_policy"] = length_policy
    return identities


def _tokenizer_identity(path: object) -> dict[str, Any]:
    tokenizer_path = Path(path).resolve()
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Tokenizer 目录不存在: {tokenizer_path}")
    files = {}
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        file_path = tokenizer_path / filename
        if not file_path.is_file():
            raise FileNotFoundError(f"Tokenizer 文件不存在: {file_path}")
        files[filename] = {
            "bytes": file_path.stat().st_size,
            "sha256": _sha256_file(file_path),
        }
    try:
        config = json.loads(
            (tokenizer_path / "tokenizer_config.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("tokenizer_config.json 无效") from error
    chat_template = config.get("chat_template") if isinstance(config, dict) else None
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError("Tokenizer 缺少 chat template")
    return {
        "path": str(tokenizer_path),
        "chat_template_sha256": hashlib.sha256(
            chat_template.encode("utf-8")
        ).hexdigest(),
        "files": files,
    }


def _validate_pair_tokenizer(
    inputs: Mapping[str, Mapping[str, Any]], tokenizer: Mapping[str, Any]
) -> None:
    pair_tokenizer = inputs["rag_pair_evaluation"]["length_policy"]["tokenizer"]
    expected = {
        "chat_template_sha256": tokenizer["chat_template_sha256"],
        "files": tokenizer["files"],
    }
    if pair_tokenizer != expected:
        raise ValueError("RAG 成对评估与父权重比较 Tokenizer 身份不一致")


def _validate_training(training: object) -> dict[str, Any]:
    if not isinstance(training, dict):
        raise ValueError("training 必须是 JSON object")
    _expect_fields(training, {"base_sft", "rag_sft"}, "training")
    config = training["base_sft"]
    if not isinstance(config, dict):
        raise ValueError("base_sft 训练不变量必须是 JSON object")
    expected = {
        "assistant_token_budget",
        "sample_order_seed",
        "peak_lr",
        "warmup_assistant_tokens",
        "schedule_assistant_tokens",
        "floor_ratio",
        "micro_batch_size",
        "accumulation_steps",
        "grad_clip",
        "dtype",
        "sequence_length",
        "label_mask_version",
    }
    _expect_fields(config, expected, "base_sft 训练不变量")
    budget = _positive_int(
        config["assistant_token_budget"], "assistant_token_budget"
    )
    _non_negative_int(config["sample_order_seed"], "sample_order_seed")
    _finite_number(config["peak_lr"], "peak_lr", positive=True)
    _positive_int(config["warmup_assistant_tokens"], "warmup_assistant_tokens")
    schedule = _positive_int(
        config["schedule_assistant_tokens"], "schedule_assistant_tokens"
    )
    if config["warmup_assistant_tokens"] > schedule:
        raise ValueError("warmup_assistant_tokens 不能超过 schedule")
    if budget > schedule:
        raise ValueError("assistant_token_budget 不能超过 schedule")
    floor = _finite_number(config["floor_ratio"], "floor_ratio", positive=True)
    if floor > 1:
        raise ValueError("floor_ratio 不能超过 1")
    _positive_int(config["micro_batch_size"], "micro_batch_size")
    _positive_int(config["accumulation_steps"], "accumulation_steps")
    _finite_number(config["grad_clip"], "grad_clip", positive=True)
    if config["dtype"] != "bfloat16":
        raise ValueError("父权重比较固定使用 bfloat16")
    if config["sequence_length"] != 768:
        raise ValueError("父权重比较固定 sequence_length=768")
    if config["label_mask_version"] != "assistant_answer_only_v1":
        raise ValueError("父权重比较 label mask 版本无效")
    if (
        budget != 1_000_000
        or config["sample_order_seed"] != 42
        or config["peak_lr"] != 1e-5
        or config["warmup_assistant_tokens"] != 100_000
        or schedule != 1_000_000
    ):
        raise ValueError("基础法律 SFT 训练预算或学习率偏离冻结比较口径")

    rag = training["rag_sft"]
    if not isinstance(rag, dict):
        raise ValueError("rag_sft 训练不变量必须是 JSON object")
    rag_expected = {
        "records_per_epoch",
        "epochs",
        "assistant_tokens_per_epoch",
        "assistant_token_budget",
        "sample_order_seed",
        "peak_lr",
        "warmup_assistant_tokens",
        "schedule_assistant_tokens",
        "floor_ratio",
        "micro_batch_size",
        "accumulation_steps",
        "grad_clip",
        "dtype",
        "sequence_length",
        "label_mask_version",
    }
    _expect_fields(rag, rag_expected, "rag_sft 训练不变量")
    records = _positive_int(rag["records_per_epoch"], "records_per_epoch")
    epochs = _positive_int(rag["epochs"], "epochs")
    tokens_per_epoch = _positive_int(
        rag["assistant_tokens_per_epoch"], "assistant_tokens_per_epoch"
    )
    rag_budget = _positive_int(rag["assistant_token_budget"], "rag assistant_token_budget")
    if rag_budget != epochs * tokens_per_epoch:
        raise ValueError("RAG-SFT assistant-token 预算必须覆盖完整 epoch")
    _non_negative_int(rag["sample_order_seed"], "RAG sample_order_seed")
    _finite_number(rag["peak_lr"], "RAG peak_lr", positive=True)
    rag_warmup = _positive_int(
        rag["warmup_assistant_tokens"], "RAG warmup_assistant_tokens"
    )
    rag_schedule = _positive_int(
        rag["schedule_assistant_tokens"], "RAG schedule_assistant_tokens"
    )
    if rag_warmup > rag_schedule or rag_budget > rag_schedule:
        raise ValueError("RAG-SFT warmup、schedule 或预算无效")
    rag_floor = _finite_number(rag["floor_ratio"], "RAG floor_ratio", positive=True)
    if rag_floor > 1:
        raise ValueError("RAG floor_ratio 不能超过 1")
    _positive_int(rag["micro_batch_size"], "RAG micro_batch_size")
    _positive_int(rag["accumulation_steps"], "RAG accumulation_steps")
    _finite_number(rag["grad_clip"], "RAG grad_clip", positive=True)
    if rag["dtype"] != "bfloat16" or rag["sequence_length"] != 768:
        raise ValueError("RAG-SFT 精度或 sequence_length 无效")
    if rag["label_mask_version"] != "assistant_answer_only_v1":
        raise ValueError("RAG-SFT label mask 版本无效")
    if (
        records != 832
        or epochs != 2
        or tokens_per_epoch != 67_316
        or rag_budget != 134_632
        or rag["sample_order_seed"] != 42
        or rag["peak_lr"] != 5e-6
        or rag_warmup != 13_463
        or rag_schedule != 134_632
        or rag_floor != 0.1
        or rag["micro_batch_size"] != config["micro_batch_size"]
        or rag["accumulation_steps"] != config["accumulation_steps"]
        or rag["grad_clip"] != config["grad_clip"]
    ):
        raise ValueError("RAG-SFT 训练预算或学习率偏离冻结比较口径")
    return training


def _validate_evaluation(evaluation: object) -> dict[str, Any]:
    if not isinstance(evaluation, dict):
        raise ValueError("evaluation 必须是 JSON object")
    _expect_fields(
        evaluation,
        {
            "phase_suites",
            "metric_schema_version",
            "decoding",
            "bootstrap_resamples",
            "bootstrap_confidence",
        },
        "evaluation",
    )
    phase_suites = evaluation["phase_suites"]
    if not isinstance(phase_suites, dict) or phase_suites != {
        phase: list(suites) for phase, suites in PHASE_SUITES.items()
    }:
        raise ValueError("evaluation 必须为六个观察点使用固定 suites")
    metric_version = evaluation["metric_schema_version"]
    if metric_version != "legal_parent_metrics_v2":
        raise ValueError("metric_schema_version 无效")
    if evaluation["bootstrap_resamples"] != 10_000:
        raise ValueError("paired bootstrap 必须固定为 10,000 次")
    if _finite_number(evaluation["bootstrap_confidence"], "bootstrap_confidence") != 0.95:
        raise ValueError("bootstrap 置信水平必须固定为 0.95")
    decoding = evaluation["decoding"]
    if not isinstance(decoding, dict):
        raise ValueError("decoding 必须是 JSON object")
    _expect_fields(
        decoding,
        {"max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty"},
        "decoding",
    )
    _positive_int(decoding["max_new_tokens"], "max_new_tokens")
    temperature = _finite_number(decoding["temperature"], "temperature")
    if temperature < 0:
        raise ValueError("temperature 不能为负")
    top_p = _finite_number(decoding["top_p"], "top_p", positive=True)
    if top_p > 1:
        raise ValueError("top_p 不能超过 1")
    _non_negative_int(decoding["top_k"], "top_k")
    _finite_number(
        decoding["repetition_penalty"], "repetition_penalty", positive=True
    )
    return evaluation


def _resolve_base_sft_manifest(
    base_sft_manifest: str | Path | None,
    manifest_root: str | Path | None,
) -> Path:
    if base_sft_manifest is not None:
        return Path(base_sft_manifest).resolve()
    if manifest_root is None:
        raise ValueError("必须提供正式基础法律 SFT manifest 或 manifest 根目录")
    root = Path(manifest_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"manifest 根目录不存在: {root}")
    candidates = []
    for path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("schema_version") == "1.0"
            and payload.get("pipeline") == "disc_law_sft_length_filtered_768_v1"
            and payload.get("complete") is True
            and isinstance(payload.get("readiness"), dict)
            and payload["readiness"].get("training_ready") is True
        ):
            _load_verified_json(path, "正式基础法律 SFT manifest")
            candidates.append(path.resolve())
    if len(candidates) != 1:
        rendered = ", ".join(str(path) for path in candidates) or "无"
        raise ValueError(
            "manifest 根目录必须唯一包含一份正式基础法律 SFT manifest，"
            f"当前候选: {rendered}"
        )
    return candidates[0]


def create_spec(
    *,
    candidate_paths: dict[str, str | Path],
    general_validation: str | Path,
    evaluation_exclusions: str | Path,
    rag_sft_release: str | Path,
    rag_pair_evaluation: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
    base_sft_manifest: str | Path | None = None,
    manifest_root: str | Path | None = None,
) -> dict[str, Any]:
    """从正式资产生成并发布不可变第一轮父权重比较 spec。"""

    if set(candidate_paths) != set(REQUIRED_PARENT_ROLES):
        raise ValueError("父权重路径必须覆盖五个固定角色")
    candidates = []
    for role in REQUIRED_PARENT_ROLES:
        path = Path(candidate_paths[role]).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"父权重不存在: {path}")
        candidates.append(
            {
                "role": role,
                "cpt_input_tokens": PARENT_CPT_INPUT_TOKENS[role],
                "path": str(path),
                "sha256": _sha256_file(path),
            }
        )
    base_path = _resolve_base_sft_manifest(base_sft_manifest, manifest_root)
    base_payload, _ = _load_verified_json(base_path, "正式基础法律 SFT manifest")
    general_validation_path = Path(general_validation).resolve()
    exclusions_path = Path(evaluation_exclusions).resolve()
    rag_release_path = Path(rag_sft_release).resolve()
    rag_pair_path = Path(rag_pair_evaluation).resolve()
    spec = {
        "schema_version": "1.0",
        "pipeline": SPEC_PIPELINE,
        "round": "first_round",
        "candidates": candidates,
        "inputs": {
            "base_sft_data": {
                "path": str(base_path),
                "pipeline": base_payload.get("pipeline"),
            },
            "general_validation": {
                "path": str(general_validation_path),
                "pipeline": _GENERAL_VALIDATION_PIPELINE,
                "split": "full_validation",
            },
            "evaluation_exclusions": {
                "path": str(exclusions_path),
                "pipeline": _EVALUATION_EXCLUSIONS_PIPELINE,
            },
            "rag_sft_data": {
                "path": str(rag_release_path),
                "pipeline": _RAG_SFT_RELEASE_PIPELINE,
            },
            "rag_development": {
                "path": str(exclusions_path),
                "pipeline": _EVALUATION_EXCLUSIONS_PIPELINE,
                "asset_name": _EVALUATION_ASSET_NAMES["rag_development"],
            },
            "rag_holdout": {
                "path": str(exclusions_path),
                "pipeline": _EVALUATION_EXCLUSIONS_PIPELINE,
                "asset_name": _EVALUATION_ASSET_NAMES["rag_holdout"],
            },
            "rag_pair_evaluation": {
                "path": str(rag_pair_path),
                "pipeline": _RAG_PAIR_EVALUATION_PIPELINE,
            },
        },
        "tokenizer_path": str(Path(tokenizer_path).resolve()),
        "seeds": [42],
        "phases": list(FIRST_ROUND_PHASES),
        "training": {
            "base_sft": {
                "assistant_token_budget": 1_000_000,
                "sample_order_seed": 42,
                "peak_lr": 1e-5,
                "warmup_assistant_tokens": 100_000,
                "schedule_assistant_tokens": 1_000_000,
                "floor_ratio": 0.1,
                "micro_batch_size": 8,
                "accumulation_steps": 2,
                "grad_clip": 1.0,
                "dtype": "bfloat16",
                "sequence_length": 768,
                "label_mask_version": "assistant_answer_only_v1",
            },
            "rag_sft": {
                "records_per_epoch": 832,
                "epochs": 2,
                "assistant_tokens_per_epoch": 67_316,
                "assistant_token_budget": 134_632,
                "sample_order_seed": 42,
                "peak_lr": 5e-6,
                "warmup_assistant_tokens": 13_463,
                "schedule_assistant_tokens": 134_632,
                "floor_ratio": 0.1,
                "micro_batch_size": 8,
                "accumulation_steps": 2,
                "grad_clip": 1.0,
                "dtype": "bfloat16",
                "sequence_length": 768,
                "label_mask_version": "assistant_answer_only_v1",
            }
        },
        "evaluation": {
            "phase_suites": {
                phase: list(suites) for phase, suites in PHASE_SUITES.items()
            },
            "metric_schema_version": "legal_parent_metrics_v2",
            "decoding": {
                "max_new_tokens": 160,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "repetition_penalty": 1.0,
            },
            "bootstrap_resamples": 10_000,
            "bootstrap_confidence": 0.95,
        },
    }
    _validate_spec(spec)
    _candidate_identities(spec["candidates"])
    inputs = _input_identities(spec["inputs"])
    tokenizer = _tokenizer_identity(spec["tokenizer_path"])
    _validate_pair_tokenizer(inputs, tokenizer)
    _validate_training(spec["training"])
    _validate_evaluation(spec["evaluation"])
    _write_immutable_json(output_path, spec)
    return spec


def _trial_execution(
    candidate_role: str, parent_sha256: str, seed: int, phase: str
) -> dict[str, Any]:
    """把观察点编译为训练边界明确的执行契约。"""

    if phase == "baseline":
        return {
            "stage": "baseline",
            "model_source": {
                "kind": "parent_model_only",
                "parent_sha256": parent_sha256,
            },
            "stop_assistant_tokens": 0,
            "model_only_export_required": False,
            "resume_scope": "not_applicable",
        }
    if phase in BASE_PHASE_TOKEN_BUDGETS:
        return {
            "stage": "base_sft",
            "model_source": {
                "kind": "parent_model_only",
                "parent_sha256": parent_sha256,
            },
            "stop_assistant_tokens": BASE_PHASE_TOKEN_BUDGETS[phase],
            "model_only_export_required": True,
            "resume_scope": "same_candidate_same_seed_base_sft_only",
        }
    if phase in RAG_PHASE_TOKEN_BUDGETS:
        return {
            "stage": "rag_sft",
            "model_source": {
                "kind": "base_1m_model_only",
                "trial_id": f"{candidate_role}-seed-{seed}-base_1m",
            },
            "stop_assistant_tokens": RAG_PHASE_TOKEN_BUDGETS[phase],
            "model_only_export_required": True,
            "resume_scope": "same_candidate_same_seed_rag_sft_only",
        }
    raise ValueError(f"未知父权重比较观察点: {phase}")


def _compile_continuous_chains(
    candidates: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    trials: Sequence[Mapping[str, Any]],
    evaluation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """把观察结果编译为每个候选一条连续训练链。"""

    trial_by_id = {trial["trial_id"]: trial for trial in trials}
    chains = []
    for candidate in candidates:
        for seed in seeds:
            chain_id = f"{candidate['role']}-seed-{seed}"
            observations = []
            for phase in FIRST_ROUND_PHASES:
                trial_id = f"{chain_id}-{phase}"
                trial = trial_by_id[trial_id]
                observations.append(
                    {
                        "trial_id": trial_id,
                        "phase": phase,
                        "suites": evaluation["phase_suites"][phase],
                        "stop_assistant_tokens": trial["execution"][
                            "stop_assistant_tokens"
                        ],
                        "model_only_export_required": trial["execution"][
                            "model_only_export_required"
                        ],
                        "resume_scope": trial["execution"]["resume_scope"],
                        "result_path": f"chains/{chain_id}/results/{phase}.json",
                        "model_only_path": (
                            None
                            if phase == "baseline"
                            else f"chains/{chain_id}/weights/{phase}.pth"
                        ),
                    }
                )
            chains.append(
                {
                    "chain_id": chain_id,
                    "candidate_role": candidate["role"],
                    "parent_sha256": candidate["sha256"],
                    "seed": seed,
                    "stages": [
                        {
                            "stage": "baseline",
                            "phases": ["baseline"],
                            "model_source": trial_by_id[
                                f"{chain_id}-baseline"
                            ]["execution"]["model_source"],
                            "resume_scope": "not_applicable",
                            "resume_checkpoint": None,
                        },
                        {
                            "stage": "base_sft",
                            "phases": ["base_250k", "base_500k", "base_1m"],
                            "model_source": trial_by_id[
                                f"{chain_id}-base_250k"
                            ]["execution"]["model_source"],
                            "resume_scope": "same_candidate_same_seed_base_sft_only",
                            "resume_checkpoint": (
                                f"chains/{chain_id}/resume/base-sft.pt"
                            ),
                        },
                        {
                            "stage": "rag_sft",
                            "phases": ["rag_epoch_1", "rag_epoch_2"],
                            "model_source": trial_by_id[
                                f"{chain_id}-rag_epoch_1"
                            ]["execution"]["model_source"],
                            "resume_scope": "same_candidate_same_seed_rag_sft_only",
                            "resume_checkpoint": (
                                f"chains/{chain_id}/resume/rag-sft.pt"
                            ),
                        },
                    ],
                    "observations": observations,
                }
            )
    return chains


def prepare_plan(spec_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """校验五候选与共享输入后发布不可变第一轮评估计划。"""

    spec, spec_digest = _load_verified_json(spec_path, "父权重评估 spec")
    _validate_spec(spec)
    candidates = _candidate_identities(spec["candidates"])
    inputs = _input_identities(spec["inputs"])
    tokenizer = _tokenizer_identity(spec["tokenizer_path"])
    _validate_pair_tokenizer(inputs, tokenizer)
    training = _validate_training(spec["training"])
    evaluation = _validate_evaluation(spec["evaluation"])
    shared = {
        "inputs": inputs,
        "tokenizer": tokenizer,
        "phases": spec["phases"],
        "training": training,
        "evaluation": evaluation,
    }
    shared_digest = _canonical_sha256(shared)
    trials = []
    for candidate in candidates:
        for seed in spec["seeds"]:
            for phase in spec["phases"]:
                trials.append(
                    {
                        "trial_id": f"{candidate['role']}-seed-{seed}-{phase}",
                        "candidate_role": candidate["role"],
                        "parent_sha256": candidate["sha256"],
                        "seed": seed,
                        "phase": phase,
                        "execution": _trial_execution(
                            candidate["role"], candidate["sha256"], seed, phase
                        ),
                    }
                )
    chains = _compile_continuous_chains(
        candidates, spec["seeds"], trials, evaluation
    )
    plan = {
        "schema_version": "1.0",
        "pipeline": PLAN_PIPELINE,
        "created_at": _utc_now(),
        "round": "first_round",
        "formal_parent_selected": False,
        "spec": {
            "path": str(Path(spec_path).resolve()),
            "sha256": spec_digest,
        },
        "candidates": candidates,
        "inputs": inputs,
        "tokenizer": tokenizer,
        "seeds": spec["seeds"],
        "phases": spec["phases"],
        "training": training,
        "evaluation": evaluation,
        "shared_invariants_sha256": shared_digest,
        "chains": chains,
        "trials": trials,
        "complete": True,
    }
    _write_immutable_json(output_path, plan)
    return plan


def _load_plan(path: str | Path) -> tuple[dict[str, Any], str]:
    plan, digest = _load_verified_json(path, "父权重评估 plan")
    if (
        plan.get("schema_version") != "1.0"
        or plan.get("pipeline") != PLAN_PIPELINE
        or plan.get("complete") is not True
        or plan.get("formal_parent_selected") is not False
        or plan.get("round") != "first_round"
        or plan.get("phases") != list(FIRST_ROUND_PHASES)
    ):
        raise ValueError("父权重评估 plan 版本或状态无效")
    try:
        expected_chains = _compile_continuous_chains(
            plan["candidates"], plan["seeds"], plan["trials"], plan["evaluation"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("父权重评估 plan 连续训练链无效") from error
    if plan.get("chains") != expected_chains:
        raise ValueError("父权重评估 plan 连续训练链无效")
    return plan, digest


def _load_legacy_v3_plan(path: str | Path) -> tuple[dict[str, Any], str]:
    """只为一次性迁移读取并复验冻结的 v3 第一轮计划。"""

    plan, digest = _load_verified_json(path, "v3 父权重评估 plan")
    if (
        plan.get("schema_version") != "1.0"
        or plan.get("pipeline") != LEGACY_PLAN_PIPELINE
        or plan.get("complete") is not True
        or plan.get("formal_parent_selected") is not False
        or plan.get("round") != "first_round"
        or plan.get("seeds") != [42]
        or plan.get("phases") != list(FIRST_ROUND_PHASES)
    ):
        raise ValueError("v3 父权重评估 plan 版本或状态无效")
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or [
        item.get("role") for item in candidates if isinstance(item, dict)
    ] != list(LEGACY_PARENT_ROLES):
        raise ValueError("v3 父权重评估 plan 候选身份无效")
    for candidate in candidates:
        role = candidate["role"]
        if (
            candidate.get("cpt_input_tokens") != PARENT_CPT_INPUT_TOKENS[role]
            or not isinstance(candidate.get("sha256"), str)
            or not _SHA256_RE.fullmatch(candidate["sha256"])
        ):
            raise ValueError(f"v3 父权重候选身份无效: {role}")
    shared = {
        "inputs": plan.get("inputs"),
        "tokenizer": plan.get("tokenizer"),
        "phases": plan.get("phases"),
        "training": plan.get("training"),
        "evaluation": plan.get("evaluation"),
    }
    if plan.get("shared_invariants_sha256") != _canonical_sha256(shared):
        raise ValueError("v3 父权重评估 plan 共享不变量身份无效")
    try:
        expected_chains = _compile_continuous_chains(
            candidates, plan["seeds"], plan["trials"], plan["evaluation"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("v3 父权重评估 plan 连续训练链无效") from error
    if plan.get("chains") != expected_chains or len(plan["trials"]) != 24:
        raise ValueError("v3 父权重评估 plan trial 覆盖无效")
    return plan, digest


def _validate_result_provenance(provenance: object) -> None:
    if not isinstance(provenance, dict):
        raise ValueError("trial provenance 必须是 JSON object")
    kind = provenance.get("kind")
    if kind == "native":
        _expect_fields(provenance, {"kind"}, "原生 trial provenance")
        return
    if kind == "migrated_v3":
        _expect_fields(
            provenance,
            {"kind", "source_plan_sha256", "source_result_sha256"},
            "迁移 trial provenance",
        )
        for name in ("source_plan_sha256", "source_result_sha256"):
            value = provenance[name]
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                raise ValueError(f"迁移 trial provenance/{name} 无效")
        return
    raise ValueError("trial provenance kind 无效")


def _validate_trial_result_for_pipeline(
    plan: dict[str, Any],
    plan_digest: str,
    result_path: str | Path,
    *,
    result_pipeline: str,
    provenance_required: bool,
) -> tuple[dict[str, Any], str]:
    result, result_digest = _load_verified_json(result_path, "父权重 trial 结果")
    fields = {
        "schema_version",
        "pipeline",
        "trial_id",
        "plan_sha256",
        "shared_invariants_sha256",
        "candidate_role",
        "parent_sha256",
        "seed",
        "phase",
        "completed",
        "suites",
    }
    if provenance_required:
        fields.add("provenance")
    _expect_fields(
        result,
        fields,
        "父权重 trial 结果",
    )
    if (
        result["schema_version"] != "1.0"
        or result["pipeline"] != result_pipeline
        or result["completed"] is not True
    ):
        raise ValueError("父权重 trial 结果版本或完成状态无效")
    if provenance_required:
        _validate_result_provenance(result["provenance"])
    if result["plan_sha256"] != plan_digest:
        raise ValueError("trial 绑定的评估 plan 身份不一致")
    if result["shared_invariants_sha256"] != plan["shared_invariants_sha256"]:
        raise ValueError("trial 共享不变量身份不一致")
    trials = {trial["trial_id"]: trial for trial in plan["trials"]}
    expected = trials.get(result["trial_id"])
    if expected is None or any(
        result[key] != expected[key]
        for key in ("candidate_role", "parent_sha256", "seed", "phase")
    ):
        raise ValueError("trial 身份不属于评估 plan")
    suites = result["suites"]
    expected_suites = plan["evaluation"]["phase_suites"].get(result["phase"])
    if not isinstance(expected_suites, list):
        raise ValueError("trial phase 没有固定评估 suites")
    if not isinstance(suites, dict) or set(suites) != set(expected_suites):
        raise ValueError("trial suites 必须覆盖当前观察点的固定集合")
    for suite_name in expected_suites:
        suite = suites[suite_name]
        if not isinstance(suite, dict):
            raise ValueError(f"{suite_name} 指标必须是 JSON object")
        _expect_fields(suite, {"sample_count", "metrics"}, f"{suite_name} 指标")
        _positive_int(suite["sample_count"], f"{suite_name} sample_count")
        metrics = suite["metrics"]
        expected_metrics = (
            LOSS_METRICS if suite_name in LOSS_SUITES else RAG_METRICS
        )
        if not isinstance(metrics, dict) or set(metrics) != set(expected_metrics):
            raise ValueError(f"{suite_name} metrics 必须使用冻结 schema")
        for name, value in metrics.items():
            if not isinstance(name, str) or not _METRIC_NAME_RE.fullmatch(name):
                raise ValueError(f"{suite_name} metric 名称无效")
            parsed = _finite_number(value, f"{suite_name}/{name}")
            if suite_name in RAG_SUITES and not 0 <= parsed <= 1:
                raise ValueError(f"{suite_name}/{name} 必须位于 [0, 1]")
            if suite_name in LOSS_SUITES and parsed <= 0:
                raise ValueError(f"{suite_name}/{name} 必须为正数")
    return result, result_digest


def validate_trial_result(
    plan: dict[str, Any], plan_digest: str, result_path: str | Path
) -> dict[str, Any]:
    """校验单个 v4 trial 的身份、来源、封闭指标 schema 和有限数值。"""

    result, _ = _validate_trial_result_for_pipeline(
        plan,
        plan_digest,
        result_path,
        result_pipeline=TRIAL_RESULT_PIPELINE,
        provenance_required=True,
    )
    return result


def _result_paths_by_trial(
    plan: Mapping[str, Any], run_root: Path
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for chain in plan["chains"]:
        for observation in chain["observations"]:
            trial_id = observation["trial_id"]
            if trial_id in paths:
                raise ValueError(f"重复 trial 结果路径: {trial_id}")
            paths[trial_id] = run_root / observation["result_path"]
    return paths


def migrate_v3_results(
    *,
    source_plan_path: str | Path,
    target_plan_path: str | Path,
    source_run_root: str | Path,
    target_run_root: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """把已完成 v3 trial 审计重绑定到共享不变量相同的 v4 plan。"""

    source_plan, source_plan_digest = _load_legacy_v3_plan(source_plan_path)
    target_plan, target_plan_digest = _load_plan(target_plan_path)
    if (
        source_plan["shared_invariants_sha256"]
        != target_plan["shared_invariants_sha256"]
    ):
        raise ValueError("v3 与 v4 plan 的共享不变量不一致，禁止迁移")
    target_candidates = {
        item["role"]: item for item in target_plan["candidates"]
    }
    for source_candidate in source_plan["candidates"]:
        target_candidate = target_candidates.get(source_candidate["role"])
        if target_candidate is None or any(
            source_candidate[key] != target_candidate[key]
            for key in ("cpt_input_tokens", "sha256")
        ):
            raise ValueError(
                f"v3 与 v4 父权重身份不一致: {source_candidate['role']}"
            )

    source_root = Path(source_run_root).resolve()
    target_root = Path(target_run_root).resolve()
    source_paths = _result_paths_by_trial(source_plan, source_root)
    target_paths = _result_paths_by_trial(target_plan, target_root)
    target_trials = {trial["trial_id"]: trial for trial in target_plan["trials"]}
    prepared: list[tuple[Path, Path, dict[str, Any], str]] = []
    for chain in source_plan["chains"]:
        missing_seen = False
        for observation in chain["observations"]:
            trial_id = observation["trial_id"]
            source_path = source_paths[trial_id]
            present = source_path.exists() or source_path.with_suffix(".sha256").exists()
            if not present:
                missing_seen = True
                continue
            if missing_seen:
                raise ValueError(f"v3 连续链存在越序 trial 结果: {trial_id}")
            source_result, source_result_digest = _validate_trial_result_for_pipeline(
                source_plan,
                source_plan_digest,
                source_path,
                result_pipeline=LEGACY_TRIAL_RESULT_PIPELINE,
                provenance_required=False,
            )
            target_trial = target_trials.get(trial_id)
            if target_trial is None or any(
                source_result[key] != target_trial[key]
                for key in ("candidate_role", "parent_sha256", "seed", "phase")
            ):
                raise ValueError(f"v3 trial 不属于目标 v4 plan: {trial_id}")
            migrated = dict(source_result)
            migrated.update(
                {
                    "pipeline": TRIAL_RESULT_PIPELINE,
                    "plan_sha256": target_plan_digest,
                    "provenance": {
                        "kind": "migrated_v3",
                        "source_plan_sha256": source_plan_digest,
                        "source_result_sha256": source_result_digest,
                    },
                }
            )
            prepared.append(
                (
                    source_path,
                    target_paths[trial_id],
                    migrated,
                    source_result_digest,
                )
            )
    if not prepared:
        raise ValueError("v3 运行根目录没有可迁移的连续 trial 结果")

    migration_path = Path(output_path).resolve()
    destinations = [migration_path, *(item[1] for item in prepared)]
    for destination in destinations:
        if destination.exists() or destination.with_suffix(".sha256").exists():
            raise FileExistsError(f"输出已存在，不能覆盖: {destination}")

    migrated_results = []
    for source_path, target_path, payload, source_digest in prepared:
        _write_immutable_json(target_path, payload)
        validate_trial_result(target_plan, target_plan_digest, target_path)
        migrated_results.append(
            {
                "trial_id": payload["trial_id"],
                "source_path": str(source_path),
                "source_sha256": source_digest,
                "target_path": str(target_path),
                "target_sha256": _sha256_file(target_path),
            }
        )
    report = {
        "schema_version": "1.0",
        "pipeline": MIGRATION_PIPELINE,
        "created_at": _utc_now(),
        "source_plan": {
            "path": str(Path(source_plan_path).resolve()),
            "sha256": source_plan_digest,
        },
        "target_plan": {
            "path": str(Path(target_plan_path).resolve()),
            "sha256": target_plan_digest,
        },
        "source_run_root": str(source_root),
        "target_run_root": str(target_root),
        "migrated_results": len(migrated_results),
        "results": migrated_results,
        "complete": True,
    }
    _write_immutable_json(migration_path, report)
    return report


def _aggregate_results(
    plan: dict[str, Any], results: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        role: {phase: [] for phase in plan["phases"]}
        for role in REQUIRED_PARENT_ROLES
    }
    for result in results:
        grouped[result["candidate_role"]][result["phase"]].append(result)
    candidates = {
        candidate["role"]: {"parent_sha256": candidate["sha256"], "phases": {}}
        for candidate in plan["candidates"]
    }
    for role in REQUIRED_PARENT_ROLES:
        for phase in plan["phases"]:
            phase_results = sorted(grouped[role][phase], key=lambda item: item["seed"])
            suite_reports = {}
            for suite_name in plan["evaluation"]["phase_suites"][phase]:
                sample_counts = {item["suites"][suite_name]["sample_count"] for item in phase_results}
                metric_sets = {
                    tuple(sorted(item["suites"][suite_name]["metrics"]))
                    for item in phase_results
                }
                if len(sample_counts) != 1 or len(metric_sets) != 1:
                    raise ValueError(f"{suite_name} 的样本数或 metric 集合在 seeds 间不一致")
                phase_sample_counts = {
                    result["suites"][suite_name]["sample_count"]
                    for result in results
                    if result["phase"] == phase
                }
                phase_metric_sets = {
                    tuple(sorted(result["suites"][suite_name]["metrics"]))
                    for result in results
                    if result["phase"] == phase
                }
                if len(phase_sample_counts) != 1 or len(phase_metric_sets) != 1:
                    raise ValueError(
                        f"{phase}/{suite_name} 的样本数或 metric 集合在候选间不一致"
                    )
                metrics = {}
                for metric_name in next(iter(metric_sets)):
                    by_seed = [
                        {
                            "seed": item["seed"],
                            "value": float(
                                item["suites"][suite_name]["metrics"][metric_name]
                            ),
                        }
                        for item in phase_results
                    ]
                    values = [item["value"] for item in by_seed]
                    metrics[metric_name] = {
                        "mean": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "by_seed": by_seed,
                    }
                suite_reports[suite_name] = {
                    "sample_count": next(iter(sample_counts)),
                    "metrics": metrics,
                }
            candidates[role]["phases"][phase] = {
                "seeds": [item["seed"] for item in phase_results],
                "suites": suite_reports,
            }
    return candidates


def summarize_results(
    plan_path: str | Path,
    result_paths: Sequence[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """要求完整 trial 覆盖并发布不自动选赢家的指标汇总。"""

    plan, plan_digest = _load_plan(plan_path)
    if len({str(Path(path).resolve()) for path in result_paths}) != len(result_paths):
        raise ValueError("trial 结果路径不能重复")
    results = [
        validate_trial_result(plan, plan_digest, path) for path in result_paths
    ]
    expected_ids = {trial["trial_id"] for trial in plan["trials"]}
    result_ids = [result["trial_id"] for result in results]
    if len(set(result_ids)) != len(result_ids) or set(result_ids) != expected_ids:
        raise ValueError("trial 覆盖不完整或存在重复")
    report = {
        "schema_version": "1.0",
        "pipeline": SUMMARY_PIPELINE,
        "created_at": _utc_now(),
        "plan": {
            "path": str(Path(plan_path).resolve()),
            "sha256": plan_digest,
            "shared_invariants_sha256": plan["shared_invariants_sha256"],
        },
        "coverage": {
            "expected_trials": len(expected_ids),
            "completed_trials": len(results),
        },
        "results": [
            {
                "trial_id": result["trial_id"],
                "path": str(Path(path).resolve()),
                "sha256": _sha256_file(Path(path).resolve()),
            }
            for path, result in sorted(
                zip(result_paths, results), key=lambda item: item[1]["trial_id"]
            )
        ],
        "candidates": _aggregate_results(plan, results),
        "selection": {
            "automatic_winner": None,
            "requires_human_decision": True,
            "decision_phase": "rag_epoch_2",
            "decision_suites": plan["evaluation"]["phase_suites"]["rag_epoch_2"],
            "private_holdout_evaluated": False,
        },
        "complete": True,
    }
    _write_immutable_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    """创建 spec、计划发布和结果汇总 CLI。"""

    parser = argparse.ArgumentParser(description="法律 SFT 父权重评估基础设施")
    subparsers = parser.add_subparsers(dest="command", required=True)
    spec = subparsers.add_parser("spec", help="从正式资产发布第一轮父权重比较 spec")
    for role in REQUIRED_PARENT_ROLES:
        spec.add_argument(f"--{role}", required=True)
    spec.add_argument("--base_sft_manifest")
    spec.add_argument("--manifest_root")
    spec.add_argument("--general_validation", required=True)
    spec.add_argument("--evaluation_exclusions", required=True)
    spec.add_argument("--rag_sft_release", required=True)
    spec.add_argument("--rag_pair_evaluation", required=True)
    spec.add_argument("--tokenizer_path", required=True)
    spec.add_argument("--output", required=True)
    prepare = subparsers.add_parser("prepare", help="发布不可变第一轮评估计划")
    prepare.add_argument("--spec", required=True)
    prepare.add_argument("--output", required=True)
    summarize = subparsers.add_parser("summarize", help="汇总完整 trial 指标")
    summarize.add_argument("--plan", required=True)
    summarize.add_argument("--result", action="append", required=True)
    summarize.add_argument("--output", required=True)
    migrate = subparsers.add_parser(
        "migrate-v3", help="把共享不变量一致的已完成 v3 trial 审计迁移到 v4"
    )
    migrate.add_argument("--source-plan", required=True)
    migrate.add_argument("--target-plan", required=True)
    migrate.add_argument("--source-run-root", required=True)
    migrate.add_argument("--target-run-root", required=True)
    migrate.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "spec":
            spec = create_spec(
                candidate_paths={
                    role: getattr(args, role) for role in REQUIRED_PARENT_ROLES
                },
                base_sft_manifest=args.base_sft_manifest,
                manifest_root=args.manifest_root,
                general_validation=args.general_validation,
                evaluation_exclusions=args.evaluation_exclusions,
                rag_sft_release=args.rag_sft_release,
                rag_pair_evaluation=args.rag_pair_evaluation,
                tokenizer_path=args.tokenizer_path,
                output_path=args.output,
            )
            print(
                "PARENT_EVALUATION_SPEC_OK "
                f"candidates={len(spec['candidates'])} seeds={len(spec['seeds'])}"
            )
        elif args.command == "prepare":
            plan = prepare_plan(args.spec, args.output)
            print(f"PARENT_EVALUATION_PLAN_OK trials={len(plan['trials'])}")
        elif args.command == "summarize":
            report = summarize_results(args.plan, args.result, args.output)
            print(
                "PARENT_EVALUATION_SUMMARY_OK "
                f"trials={report['coverage']['completed_trials']}"
            )
        else:
            report = migrate_v3_results(
                source_plan_path=args.source_plan,
                target_plan_path=args.target_plan,
                source_run_root=args.source_run_root,
                target_run_root=args.target_run_root,
                output_path=args.output,
            )
            print(
                "PARENT_EVALUATION_V3_MIGRATION_OK "
                f"results={report['migrated_results']}"
            )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise SystemExit(f"[失败] {error}") from error


if __name__ == "__main__":
    main()
