"""构建并评估基础法律 SFT 的固定生成开发集。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

try:
    from ..dataset import audit_sft_chat_lengths as length_auditor
except ImportError:
    from dataset import audit_sft_chat_lengths as length_auditor

from . import train_full_sft as training


SET_PIPELINE = "legal_sft_base_generation_set_v1"
REPORT_PIPELINE = "legal_sft_base_generation_evaluation_v1"
MAX_SEQ_LEN = 768
MAX_NEW_TOKENS = 256
DEFAULT_PAIR_RECORDS = 48
DEFAULT_TRIPLET_RECORDS = 16
DATASET_TARGETS = {
    "pair_qa": DEFAULT_PAIR_RECORDS,
    "triplet_qa": DEFAULT_TRIPLET_RECORDS,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _write_immutable_json(path: str | Path, payload: Mapping[str, Any]) -> str:
    resolved = Path(path).resolve()
    hash_path = resolved.with_suffix(".sha256")
    if resolved.exists() or hash_path.exists():
        raise FileExistsError(f"输出已存在，不能覆盖: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    resolved.write_text(serialized, encoding="utf-8", newline="\n")
    digest = _sha256_file(resolved)
    hash_path.write_text(
        f"{digest}  {resolved.name}\n", encoding="utf-8", newline="\n"
    )
    return digest


def _write_immutable_jsonl(
    path: str | Path, records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    resolved = Path(path).resolve()
    hash_path = resolved.with_suffix(".sha256")
    if resolved.exists() or hash_path.exists():
        raise FileExistsError(f"输出已存在，不能覆盖: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8", newline="\n") as target:
        for record in records:
            target.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            )
    digest = _sha256_file(resolved)
    hash_path.write_text(
        f"{digest}  {resolved.name}\n", encoding="utf-8", newline="\n"
    )
    return {
        "path": str(resolved),
        "records": len(records),
        "bytes": resolved.stat().st_size,
        "sha256": digest,
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
        raise ValueError(f"{description}身份已变化")
    return path


def _load_json(path: str | Path, description: str) -> dict[str, Any]:
    resolved = Path(path).resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取{description}: {resolved}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description}必须是 JSON object")
    return payload


def _normalize_prompt(value: str) -> str:
    return " ".join(value.split()).casefold()


def _render_prompt(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> str:
    try:
        prompt = tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError("固定生成问题无法应用 chat template") from error
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("chat template 没有生成有效 prompt")
    return prompt


def _prompt_tokens(tokenizer: Any, prompt: str) -> int:
    encoded = tokenizer(prompt, add_special_tokens=True)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if not isinstance(input_ids, list) or any(type(item) is not int for item in input_ids):
        raise ValueError("Tokenizer 没有返回一维 input_ids")
    return len(input_ids)


def _source_path(data_manifest: Mapping[str, Any], dataset_kind: str) -> Path:
    output = data_manifest.get("output")
    files = output.get("files") if isinstance(output, dict) else None
    metadata = (
        files.get(f"validation/full/{dataset_kind}")
        if isinstance(files, dict)
        else None
    )
    if (
        not isinstance(output, dict)
        or not isinstance(output.get("root"), str)
        or not isinstance(metadata, dict)
        or not isinstance(metadata.get("path"), str)
    ):
        raise ValueError(f"正式 data manifest 缺少 full/{dataset_kind}")
    root = Path(output["root"]).resolve()
    path = (root / metadata["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"full/{dataset_kind} 路径越出数据根目录") from error
    if (
        not path.is_file()
        or metadata.get("bytes") != path.stat().st_size
        or metadata.get("sha256") != _sha256_file(path)
    ):
        raise ValueError(f"full/{dataset_kind} 文件身份已变化")
    return path


def _read_candidates(
    path: Path, dataset_kind: str, tokenizer: Any
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                parsed = json.loads(line)
                record = length_auditor._validate_record(
                    parsed, f"validation/full/{dataset_kind}", line_number
                )
            except (json.JSONDecodeError, length_auditor.ChatLengthAuditError) as error:
                raise ValueError(
                    f"固定生成候选记录无效: {path}:{line_number}"
                ) from error
            user = record["conversations"][0]["content"]
            answer = record["conversations"][1]["content"]
            prompt_key = _normalize_prompt(user)
            if prompt_key in candidates:
                continue
            messages = [{"role": "user", "content": user}]
            prompt = _render_prompt(tokenizer, messages)
            prompt_tokens = _prompt_tokens(tokenizer, prompt)
            if prompt_tokens + MAX_NEW_TOKENS > MAX_SEQ_LEN:
                continue
            candidates[prompt_key] = {
                "record_id": record["id"],
                "dataset_kind": dataset_kind,
                "messages": messages,
                "reference_answer": answer,
                "prompt_tokens": prompt_tokens,
            }
    return sorted(
        candidates.values(),
        key=lambda item: hashlib.sha256(
            f"{dataset_kind}\0{item['record_id']}".encode("utf-8")
        ).hexdigest(),
    )


def build_generation_set(
    *,
    data_manifest: str | Path,
    tokenizer_path: str | Path,
    output_dir: str | Path,
    targets: Mapping[str, int] = DATASET_TARGETS,
) -> dict[str, Any]:
    """从 full validation 确定性发布固定生成开发子集。"""

    manifest, manifest_digest = training._load_verified_json(
        Path(data_manifest).resolve(), "基础法律正式 data manifest"
    )
    readiness = manifest.get("readiness")
    if (
        manifest.get("pipeline") != "disc_law_sft_length_filtered_768_v1"
        or manifest.get("release_status") != "formal_training_candidate"
        or manifest.get("complete") is not True
        or not isinstance(readiness, dict)
        or readiness.get("training_ready") is not True
    ):
        raise ValueError("固定生成集只能从正式 training-ready 数据发布")
    if set(targets) != set(DATASET_TARGETS) or any(
        type(value) is not int or value <= 0 for value in targets.values()
    ):
        raise ValueError("固定生成集分层目标无效")

    tokenizer = training._load_tokenizer(tokenizer_path)
    tokenizer_identity = training._tokenizer_identity(tokenizer, tokenizer_path)
    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for dataset_kind in DATASET_TARGETS:
        candidates = _read_candidates(
            _source_path(manifest, dataset_kind), dataset_kind, tokenizer
        )
        target = targets[dataset_kind]
        if len(candidates) < target:
            raise ValueError(
                f"full/{dataset_kind} 可用固定生成问题不足: {len(candidates)} < {target}"
            )
        selected.extend(candidates[:target])
        counts[dataset_kind] = target

    root = Path(output_dir).resolve()
    if root.exists():
        raise FileExistsError(f"固定生成集输出目录已存在: {root}")
    records_identity = _write_immutable_jsonl(root / "records.jsonl", selected)
    payload = {
        "schema_version": "1.0",
        "pipeline": SET_PIPELINE,
        "input": {
            "data_manifest": {
                "path": str(Path(data_manifest).resolve()),
                "sha256": manifest_digest,
            }
        },
        "tokenizer": tokenizer_identity,
        "selection": {
            "split": "validation/full",
            "method": "unique_prompt_sha256_order_v1",
            "counts": counts,
            "total": len(selected),
        },
        "generation": {
            "fixed_max_seq_len": MAX_SEQ_LEN,
            "max_new_tokens": MAX_NEW_TOKENS,
            "sampling": "greedy",
        },
        "records": records_identity,
        "private_holdout_used": False,
        "complete": True,
    }
    _write_immutable_json(root / "manifest.json", payload)
    return payload


def _load_generation_records(
    manifest_path: str | Path, tokenizer: Any, tokenizer_path: str | Path
) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    manifest, digest = training._load_verified_json(
        Path(manifest_path).resolve(), "固定法律生成集 manifest"
    )
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("pipeline") != SET_PIPELINE
        or manifest.get("complete") is not True
        or manifest.get("private_holdout_used") is not False
        or manifest.get("tokenizer")
        != training._tokenizer_identity(tokenizer, tokenizer_path)
    ):
        raise ValueError("固定法律生成集 manifest 身份或状态无效")
    records_path = _verify_identity(manifest.get("records"), "固定生成 records")
    records: list[dict[str, Any]] = []
    with records_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"固定生成 records 无法解析: {line_number}") from error
            if (
                not isinstance(record, dict)
                or set(record)
                != {
                    "record_id",
                    "dataset_kind",
                    "messages",
                    "reference_answer",
                    "prompt_tokens",
                }
                or record["dataset_kind"] not in DATASET_TARGETS
                or not isinstance(record["messages"], list)
                or len(record["messages"]) != 1
                or not isinstance(record["reference_answer"], str)
                or not record["reference_answer"]
                or type(record["prompt_tokens"]) is not int
                or record["prompt_tokens"] + MAX_NEW_TOKENS > MAX_SEQ_LEN
            ):
                raise ValueError(f"固定生成 record schema 无效: {line_number}")
            records.append(record)
    if len(records) != manifest.get("records", {}).get("records"):
        raise ValueError("固定生成 records 数量与 manifest 不一致")
    return manifest, digest, records


def evaluate_generation(
    *,
    generation_manifest: str | Path,
    tokenizer_path: str | Path,
    weights: str | Path,
    weights_sha256: str,
    output: str | Path,
    device_name: str,
) -> dict[str, Any]:
    """使用固定 greedy 配置保存逐题原始生成与结构诊断。"""

    tokenizer = training._load_tokenizer(tokenizer_path)
    manifest, manifest_digest, records = _load_generation_records(
        generation_manifest, tokenizer, tokenizer_path
    )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 CUDA 设备不可用")
    model = training.MiniMindForCausalLM(training._model_config()).to(device)
    actual_weights_sha256 = training._load_parent_weights(
        weights, weights_sha256, model
    )
    model.eval()

    outputs: list[dict[str, Any]] = []
    with torch.inference_mode():
        for record in records:
            prompt = _render_prompt(tokenizer, record["messages"])
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
            input_ids = encoded["input_ids"].to(device)
            if input_ids.shape[1] != record["prompt_tokens"]:
                raise ValueError("固定生成 prompt token 数发生漂移")
            generated = model.generate(
                inputs=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=1.0,
                top_k=0,
                use_cache=True,
                do_sample=False,
            )
            output_ids = generated[0, input_ids.shape[1] :]
            raw_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            generated_tokens = int(output_ids.numel())
            outputs.append(
                {
                    **record,
                    "raw_output": raw_text,
                    "generated_tokens": generated_tokens,
                    "nonempty_output": bool(raw_text.strip()),
                    "hit_max_new_tokens": generated_tokens == MAX_NEW_TOKENS,
                }
            )

    nonempty = sum(record["nonempty_output"] for record in outputs)
    hit_max = sum(record["hit_max_new_tokens"] for record in outputs)
    report = {
        "schema_version": "1.0",
        "pipeline": REPORT_PIPELINE,
        "inputs": {
            "generation_manifest": {
                "path": str(Path(generation_manifest).resolve()),
                "sha256": manifest_digest,
            },
            "weights": {
                "path": str(Path(weights).resolve()),
                "sha256": actual_weights_sha256,
            },
            "tokenizer": manifest["tokenizer"],
        },
        "generation": manifest["generation"],
        "metrics": {
            "records": len(outputs),
            "nonempty_output_rate": nonempty / len(outputs),
            "hit_max_new_tokens_rate": hit_max / len(outputs),
        },
        "records": outputs,
        "review": {
            "required": True,
            "method": "anonymous_legal_error_categories",
            "automatic_quality_winner": None,
        },
        "complete": True,
    }
    _write_immutable_json(output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基础法律固定生成开发集与评估")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="发布固定生成开发子集")
    prepare.add_argument("--data-manifest", required=True)
    prepare.add_argument("--tokenizer-path", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--pair-records", type=int, default=DEFAULT_PAIR_RECORDS)
    prepare.add_argument(
        "--triplet-records", type=int, default=DEFAULT_TRIPLET_RECORDS
    )

    evaluate = commands.add_parser("evaluate", help="执行固定 greedy 生成评估")
    evaluate.add_argument("--generation-manifest", required=True)
    evaluate.add_argument("--tokenizer-path", required=True)
    evaluate.add_argument("--weights", required=True)
    evaluate.add_argument("--weights-sha256", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "prepare":
            result = build_generation_set(
                data_manifest=args.data_manifest,
                tokenizer_path=args.tokenizer_path,
                output_dir=args.output_dir,
                targets={
                    "pair_qa": args.pair_records,
                    "triplet_qa": args.triplet_records,
                },
            )
            print(
                "SFT_BASE_GENERATION_SET_OK "
                f"records={result['selection']['total']}"
            )
        else:
            result = evaluate_generation(
                generation_manifest=args.generation_manifest,
                tokenizer_path=args.tokenizer_path,
                weights=args.weights,
                weights_sha256=args.weights_sha256,
                output=args.output,
                device_name=args.device,
            )
            print(
                "SFT_BASE_GENERATION_EVALUATION_OK "
                f"records={result['metrics']['records']}"
            )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        raise SystemExit(f"[失败] {error}") from error


if __name__ == "__main__":
    main()
