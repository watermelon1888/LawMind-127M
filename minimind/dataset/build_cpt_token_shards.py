"""验证冻结的 CPT clean 语料并构建可训练 token shards。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path, PurePosixPath
from typing import Any

from .audit_pretrain_samples import load_tokenizer, sha256_file
from .build_stage_b_token_shards import (
    SEQUENCE_LENGTH,
    SHARD_TOKEN_CAPACITY,
    TOKENIZER_BATCH_SIZE,
    TOKENIZER_CHARACTER_BUDGET,
    STREAM_NAMES,
    ShardBuildError,
    SourceInputs,
    ensure_empty_outputs,
    process_train_stream,
    process_validation_streams,
    validate_tokenizer,
    verify_clean_manifest,
    write_shard_manifest,
)
from .prepare_cpt_corpus import SOURCE_PRIORITY


PART_FILENAME_RE = re.compile(r"part-[0-9]{5}\.jsonl")


def _require_dict(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ShardBuildError(f"{location} 必须是 JSON object")
    return value


def _require_list(value: object, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ShardBuildError(f"{location} 必须是 JSON array")
    return value


def _require_int(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ShardBuildError(f"{location} 必须是非负整数")
    return value


def _verify_file_report(
    report: object,
    clean_root: Path,
    split: str,
    source: str,
    index: int,
) -> tuple[Path, int]:
    report = _require_dict(report, f"{source}/{split} output file")
    path_text = report.get("path")
    if not isinstance(path_text, str):
        raise ShardBuildError(f"{source}/{split} JSONL path 必须是字符串")
    relative = PurePosixPath(path_text)
    expected_parts = (split, source, f"part-{index:05d}.jsonl")
    if relative.is_absolute() or relative.parts != expected_parts:
        raise ShardBuildError(f"CPT clean JSONL 路径不符合固定目录和编号: {path_text}")
    if PART_FILENAME_RE.fullmatch(relative.name) is None:
        raise ShardBuildError(f"CPT clean JSONL 文件名无效: {path_text}")
    path = clean_root.joinpath(*relative.parts).resolve()
    try:
        path.relative_to(clean_root)
    except ValueError as error:
        raise ShardBuildError(f"CPT clean JSONL 路径越出输出根目录: {path_text}") from error

    records = _require_int(report.get("records"), f"{path} records")
    expected_bytes = _require_int(report.get("bytes"), f"{path} bytes")
    expected_digest = report.get("sha256")
    if (
        not isinstance(expected_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
    ):
        raise ShardBuildError(f"{path} SHA-256 格式无效")
    if not path.is_file():
        raise ShardBuildError(f"CPT clean JSONL 不存在: {path}")
    if path.stat().st_size != expected_bytes:
        raise ShardBuildError(f"CPT clean JSONL 文件大小不匹配: {path}")
    if sha256_file(path) != expected_digest:
        raise ShardBuildError(f"CPT clean JSONL SHA-256 不匹配: {path}")
    return path, records


def verify_cpt_clean_inputs(
    manifest: dict[str, Any],
) -> tuple[Path, dict[str, SourceInputs], int]:
    """验证 CPT clean JSONL 的范围、身份和记录数闭合。"""
    if manifest.get("pipeline") != "cpt":
        raise ShardBuildError("clean manifest pipeline 必须为 cpt")
    if manifest.get("source_priority") != list(SOURCE_PRIORITY):
        raise ShardBuildError("CPT clean manifest 来源优先级不一致")
    exclusion = _require_dict(
        manifest.get("evaluation_exclusion"), "clean manifest evaluation_exclusion"
    )
    if exclusion.get("complete") is not True:
        raise ShardBuildError("评估资产排除尚未完成")

    output = _require_dict(manifest.get("output"), "clean manifest output")
    root_text = output.get("root")
    if not isinstance(root_text, str) or not root_text:
        raise ShardBuildError("clean manifest output.root 必须是非空字符串")
    if output.get("record_fields") != ["id", "source", "text"]:
        raise ShardBuildError("CPT clean manifest record_fields 不一致")
    clean_root = Path(root_text).resolve()
    if not clean_root.is_dir():
        raise ShardBuildError(f"CPT clean JSONL 根目录不存在: {clean_root}")

    source_reports = _require_dict(manifest.get("sources"), "clean manifest sources")
    if tuple(source_reports) != SOURCE_PRIORITY:
        raise ShardBuildError("CPT clean manifest 来源顺序或范围不一致")
    verified_paths: set[Path] = set()
    source_inputs: dict[str, SourceInputs] = {}
    totals = {"train": 0, "full_validation": 0, "quick_validation": 0}
    for source in SOURCE_PRIORITY:
        source_report = _require_dict(source_reports[source], f"sources.{source}")
        records = _require_dict(source_report.get("records"), f"sources.{source}.records")
        retained = _require_int(records.get("retained"), f"{source}.records.retained")
        train = _require_int(records.get("train"), f"{source}.records.train")
        full = _require_int(
            records.get("full_validation"), f"{source}.records.full_validation"
        )
        quick = _require_int(
            records.get("quick_validation"), f"{source}.records.quick_validation"
        )
        if retained != train + full or not train or not full or not quick or quick > full:
            raise ShardBuildError(f"{source} CPT clean 记录数不闭合或 split 为空")

        output_files = _require_dict(
            source_report.get("output_files"), f"sources.{source}.output_files"
        )
        if set(output_files) != {"train", "validation"}:
            raise ShardBuildError(f"{source} CPT clean split 范围无效")
        split_paths: dict[str, tuple[Path, ...]] = {}
        for split, expected_records in (("train", train), ("validation", full)):
            reports = _require_list(output_files[split], f"{source}.output_files.{split}")
            paths: list[Path] = []
            reported_records = 0
            for index, report in enumerate(reports):
                path, file_records = _verify_file_report(
                    report, clean_root, split, source, index
                )
                if path in verified_paths:
                    raise ShardBuildError(f"CPT clean manifest 重复引用 JSONL: {path}")
                verified_paths.add(path)
                paths.append(path)
                reported_records += file_records
            if reported_records != expected_records:
                raise ShardBuildError(f"{source}/{split} CPT clean 文件记录数不闭合")
            split_paths[split] = tuple(paths)
        source_inputs[source] = SourceInputs(
            train_paths=split_paths["train"],
            validation_paths=split_paths["validation"],
            train_documents=train,
            full_validation_documents=full,
            quick_validation_documents=quick,
        )
        totals["train"] += train
        totals["full_validation"] += full
        totals["quick_validation"] += quick

    partials = sorted(clean_root.rglob("*.partial"))
    if partials:
        raise ShardBuildError(f"CPT clean 根目录含未完成临时文件: {partials[0]}")
    actual_paths = {path.resolve() for path in clean_root.rglob("part-*.jsonl")}
    if actual_paths != verified_paths:
        raise ShardBuildError("CPT clean JSONL 文件范围与 manifest 不一致")
    manifest_totals = _require_dict(manifest.get("totals"), "clean manifest totals")
    total_records = _require_dict(manifest_totals.get("records"), "totals.records")
    for key, expected in totals.items():
        if _require_int(total_records.get(key), f"totals.records.{key}") != expected:
            raise ShardBuildError(f"CPT clean totals.records.{key} 不闭合")
    return clean_root, source_inputs, len(verified_paths)


def aggregate_streams(
    sources: dict[str, dict[str, dict[str, object]]],
    stream_name: str,
    sequence_length: int,
) -> dict[str, int]:
    """汇总 CPT 来源的一个 token 流并验证计数闭合。"""
    keys = (
        "documents",
        "text_tokens",
        "bos_tokens",
        "eos_tokens",
        "tokens_before_tail",
        "written_tokens",
        "dropped_tail_tokens",
        "sequence_count",
        "shard_count",
    )
    report = {
        key: sum(int(sources[source][stream_name][key]) for source in SOURCE_PRIORITY)
        for key in keys
    }
    if report["tokens_before_tail"] != (
        report["text_tokens"] + report["bos_tokens"] + report["eos_tokens"]
    ):
        raise ShardBuildError(f"CPT totals.{stream_name} BOS/EOS 计数不闭合")
    if (
        report["written_tokens"] + report["dropped_tail_tokens"]
        != report["tokens_before_tail"]
    ):
        raise ShardBuildError(f"CPT totals.{stream_name} written/tail 计数不闭合")
    if report["sequence_count"] != report["written_tokens"] // sequence_length:
        raise ShardBuildError(f"CPT totals.{stream_name} sequence 计数不闭合")
    return report


def build_cpt_token_shards(
    clean_manifest_path: Path,
    tokenizer_path: Path,
    output_root: Path,
    manifest_output: Path,
    tokenizer: Any | None = None,
    sequence_length: int = SEQUENCE_LENGTH,
    shard_token_capacity: int = SHARD_TOKEN_CAPACITY,
    tokenizer_batch_size: int = TOKENIZER_BATCH_SIZE,
    tokenizer_character_budget: int = TOKENIZER_CHARACTER_BUDGET,
) -> dict[str, object]:
    """生成 CPT 的确定性 uint16 token shards。"""
    clean_manifest_path = clean_manifest_path.resolve()
    tokenizer_path = tokenizer_path.resolve()
    output_root = output_root.resolve()
    manifest_output = manifest_output.resolve()
    if sequence_length <= 0:
        raise ShardBuildError("sequence_length 必须大于零")
    if shard_token_capacity <= 0 or shard_token_capacity % sequence_length:
        raise ShardBuildError("shard_token_capacity 必须是 sequence_length 的正整数倍")
    if tokenizer_batch_size <= 0 or tokenizer_character_budget <= 0:
        raise ShardBuildError("Tokenizer 批次和字符预算必须大于零")
    ensure_empty_outputs(output_root, manifest_output)

    clean_manifest, clean_report = verify_clean_manifest(clean_manifest_path)
    _, source_inputs, verified_files = verify_cpt_clean_inputs(clean_manifest)
    clean_report["verified_files"] = verified_files
    if tokenizer is None:
        tokenizer = load_tokenizer(tokenizer_path)
    tokenizer_report = validate_tokenizer(tokenizer, tokenizer_path, clean_manifest)

    source_reports: dict[str, dict[str, dict[str, object]]] = {}
    for source in SOURCE_PRIORITY:
        print(f"[CPT shards] 开始 {source}/train", flush=True)
        train = process_train_stream(
            source,
            source_inputs[source],
            tokenizer,
            output_root,
            sequence_length,
            shard_token_capacity,
            tokenizer_batch_size,
            tokenizer_character_budget,
        )
        print(f"[CPT shards] 开始 {source}/validation", flush=True)
        full, quick = process_validation_streams(
            source,
            source_inputs[source],
            tokenizer,
            output_root,
            sequence_length,
            shard_token_capacity,
            tokenizer_batch_size,
            tokenizer_character_budget,
        )
        source_reports[source] = {
            "train": train,
            "full_validation": full,
            "quick_validation": quick,
        }
    totals = {
        stream: aggregate_streams(source_reports, stream, sequence_length)
        for stream in STREAM_NAMES
    }
    manifest = {
        "schema_version": "1.0",
        "pipeline": "cpt",
        "clean_manifest": clean_report,
        "tokenizer": tokenizer_report,
        "tokenization": {
            "add_special_tokens": False,
            "truncation": False,
            "padding": False,
            "document_framing": "[BOS] + text_tokens + [EOS]",
            "batch_size": tokenizer_batch_size,
            "character_budget": tokenizer_character_budget,
        },
        "packing": {
            "mode": "continuous_per_source_and_split",
            "sequence_length": sequence_length,
            "shard_token_capacity": shard_token_capacity,
            "tail_rule": "drop_tokens_shorter_than_one_sequence",
            "cross_source_packing": False,
            "quick_validation_rule": "int(id[:16], 16) % 10000 < 5",
        },
        "output": {
            "root": str(output_root),
            "format": "numpy_npy",
            "dtype": "uint16",
            "dimensions": 1,
        },
        "source_priority": list(SOURCE_PRIORITY),
        "sources": source_reports,
        "totals": totals,
    }
    manifest_path, hash_path = write_shard_manifest(manifest, manifest_output)
    return {
        **manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256_path": str(hash_path),
    }


def main() -> None:
    """解析参数并生成 CPT token shards。"""
    parser = argparse.ArgumentParser(description="生成 CPT token shards 与 manifest")
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    args = parser.parse_args()
    result = build_cpt_token_shards(
        args.clean_manifest,
        args.tokenizer_path,
        args.output_root,
        args.manifest_output,
    )
    print(
        f"[完成] train={result['totals']['train']['written_tokens']:,} tokens，"
        f"full={result['totals']['full_validation']['written_tokens']:,} tokens，"
        f"quick={result['totals']['quick_validation']['written_tokens']:,} tokens"
    )
    print(f"manifest: {args.manifest_output}")


if __name__ == "__main__":
    main()
