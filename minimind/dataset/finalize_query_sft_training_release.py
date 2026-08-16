"""复算并不可变发布 Query-SFT training-ready release。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rag.query import QUERY_ENHANCEMENT_SYSTEM_PROMPT

from .query_sft_dataset import (
    LABEL_MASK_VERSION,
    MAX_SEQ_LEN,
    PIPELINE,
    QuerySftDatasetError,
    _subsequence_positions,
    _token_ids,
    _validate_record,
)


LENGTH_AUDIT_FILENAME = "query-sft-chat-length-audit-768.json"
LABEL_AUDIT_FILENAME = "query-sft-label-audit-768.json"
RELEASE_FILENAME = "query-sft-training-release.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    value = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if records is not None:
        value["records"] = records
    return value


def _verify_adjacent_json(path: Path, description: str) -> dict[str, Any]:
    hash_path = path.with_suffix(".sha256")
    if not path.is_file() or not hash_path.is_file():
        raise ValueError(f"{description}或相邻哈希不存在")
    fields = hash_path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 2 or fields[1] != path.name or fields[0] != _sha256_file(path):
        raise ValueError(f"{description}相邻哈希无效")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description}必须是 JSON object")
    return value


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> str:
    hash_path = path.with_suffix(".sha256")
    if path.exists() or hash_path.exists():
        raise FileExistsError(f"发布产物已存在，不能覆盖: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    digest = _sha256_file(path)
    hash_path.write_text(f"{digest}  {path.name}\n", encoding="utf-8", newline="\n")
    return digest


def audit_candidate(candidate_path: str | Path, tokenizer: Any):
    """全量复算协议、真实模板长度和 assistant-only labels。"""

    candidate = Path(candidate_path).resolve()
    bos_token = getattr(tokenizer, "bos_token", None)
    eos_token = getattr(tokenizer, "eos_token", None)
    chat_template = getattr(tokenizer, "chat_template", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if (
        not isinstance(bos_token, str)
        or not isinstance(eos_token, str)
        or not isinstance(chat_template, str)
        or type(pad_token_id) is not int
    ):
        raise ValueError("Tokenizer 缺少 Query-SFT 审计所需身份")
    prefix_ids = _token_ids(
        tokenizer, f"{bos_token}assistant\n<think>\n\n</think>\n\n"
    )
    end_ids = _token_ids(tokenizer, f"{eos_token}\n")
    records = 0
    active_tokens = []
    prompt_tokens = []
    seen_ids = set()
    with candidate.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                raise ValueError(f"candidate 含空行: {line_number}")
            try:
                record = _validate_record(json.loads(line), line_number)
            except (json.JSONDecodeError, QuerySftDatasetError) as error:
                raise ValueError(f"candidate 第 {line_number} 行无效") from error
            if record["id"] in seen_ids:
                raise ValueError(f"candidate 包含重复 id: {record['id']}")
            seen_ids.add(record["id"])
            rendered = tokenizer.apply_chat_template(
                record["conversations"],
                tokenize=False,
                add_generation_prompt=False,
            )
            input_ids = _token_ids(tokenizer, rendered)
            if len(input_ids) > MAX_SEQ_LEN:
                raise ValueError(f"candidate 第 {line_number} 行超过固定 768")
            positions = _subsequence_positions(input_ids, prefix_ids)
            if len(positions) != 1:
                raise ValueError(f"candidate 第 {line_number} 行 assistant 边界无效")
            answer_start = positions[0] + len(prefix_ids)
            if input_ids[-len(end_ids) :] != end_ids:
                raise ValueError(f"candidate 第 {line_number} 行 EOS 边界无效")
            records += 1
            prompt_tokens.append(len(input_ids))
            active_tokens.append(len(input_ids) - answer_start)
    if records == 0:
        raise ValueError("candidate 不能为空")
    return {
        "records": records,
        "prompt_tokens": {
            "min": min(prompt_tokens),
            "max": max(prompt_tokens),
            "mean": sum(prompt_tokens) / records,
        },
        "assistant_tokens": {
            "per_epoch": sum(active_tokens),
            "min_per_record": min(active_tokens),
            "max_per_record": max(active_tokens),
            "mean_per_record": sum(active_tokens) / records,
        },
    }


def publish_training_release(
    *,
    candidate_path,
    evaluation_manifest_path,
    tokenizer_path,
    output_dir,
    release_status,
    tokenizer=None,
):
    """发布可由 Query Dataset 和训练入口直接消费的 release。"""

    if release_status not in {"pilot_training_candidate", "formal_training_candidate"}:
        raise ValueError("release_status 无效")
    candidate = Path(candidate_path).resolve()
    evaluation_path = Path(evaluation_manifest_path).resolve()
    evaluation = _verify_adjacent_json(evaluation_path, "评估排除 manifest")
    if (
        evaluation.get("schema_version") != "1.2"
        or evaluation.get("pipeline") != "legal_sft_evaluation_exclusions"
        or evaluation.get("complete_for_formal_sft") is not True
    ):
        raise ValueError("评估排除 manifest 尚未正式就绪")
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            Path(tokenizer_path).resolve(), use_fast=True, local_files_only=True
        )
    audit = audit_candidate(candidate, tokenizer)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    prompt_sha = hashlib.sha256(
        QUERY_ENHANCEMENT_SYSTEM_PROMPT.encode("utf-8")
    ).hexdigest()
    tokenizer_identity = {
        "path": str(Path(tokenizer_path).resolve()),
        "vocab_size": len(tokenizer),
        "chat_template_sha256": hashlib.sha256(
            tokenizer.chat_template.encode("utf-8")
        ).hexdigest(),
    }
    length_audit = {
        "schema_version": "1.0",
        "pipeline": "query_sft_chat_length_audit_768_v1",
        "generated_at": generated_at,
        "candidate": _file_identity(candidate, records=audit["records"]),
        "tokenizer": tokenizer_identity,
        "records": audit["records"],
        "prompt_tokens": audit["prompt_tokens"],
        "fixed_max_seq_len": MAX_SEQ_LEN,
        "all_records_fit": True,
        "complete": True,
    }
    label_audit = {
        "schema_version": "1.0",
        "pipeline": "query_sft_label_audit_768_v1",
        "generated_at": generated_at,
        "candidate": _file_identity(candidate, records=audit["records"]),
        "label_mask_version": LABEL_MASK_VERSION,
        "label_scope": "assistant JSON + EOS",
        "system_user_and_padding_masked": True,
        "records": audit["records"],
        "assistant_tokens": audit["assistant_tokens"],
        "complete": True,
    }
    directory = Path(output_dir).resolve()
    length_path = directory / LENGTH_AUDIT_FILENAME
    label_path = directory / LABEL_AUDIT_FILENAME
    release_path = directory / RELEASE_FILENAME
    _write_immutable_json(length_path, length_audit)
    _write_immutable_json(label_path, label_audit)
    release = {
        "schema_version": "1.0",
        "pipeline": PIPELINE,
        "release_status": release_status,
        "generated_at": generated_at,
        "scope": {"pilot_only": release_status == "pilot_training_candidate"},
        "policy": {
            "fixed_max_seq_len": MAX_SEQ_LEN,
            "truncation": "forbidden",
            "label_mask_version": LABEL_MASK_VERSION,
        },
        "prompt": {
            "text": QUERY_ENHANCEMENT_SYSTEM_PROMPT,
            "sha256": prompt_sha,
        },
        "tokenizer": tokenizer_identity,
        "data": {
            "training_candidate": _file_identity(
                candidate, records=audit["records"]
            )
        },
        "evaluation_exclusions": _file_identity(evaluation_path),
        "audits": {
            "length": _file_identity(length_path),
            "labels": _file_identity(label_path),
        },
        "records": {
            "training": audit["records"],
            "assistant_tokens_per_epoch": audit["assistant_tokens"]["per_epoch"],
        },
        "readiness": {
            "protocol_audited": True,
            "chat_template_length_audited": True,
            "dataset_label_mask_audited": True,
            "evaluation_exclusion_audited": True,
            "training_ready": True,
        },
        "complete": True,
    }
    _write_immutable_json(release_path, release)
    return release


def _parser():
    parser = argparse.ArgumentParser(description="发布 Query-SFT training-ready release")
    parser.add_argument("--candidate-path", required=True)
    parser.add_argument("--evaluation-manifest-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--release-status",
        choices=("pilot_training_candidate", "formal_training_candidate"),
        required=True,
    )
    return parser


def main():
    args = _parser().parse_args()
    release = publish_training_release(
        candidate_path=args.candidate_path,
        evaluation_manifest_path=args.evaluation_manifest_path,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        release_status=args.release_status,
    )
    print(
        "QUERY_SFT_RELEASE_OK "
        f"records={release['records']['training']} "
        f"assistant_tokens={release['records']['assistant_tokens_per_epoch']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
