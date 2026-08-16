"""发布经真实 Retrieval 筛选后的 Query-SFT pilot 训练候选。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rag.query.enhancement import (
    QUERY_ENHANCEMENT_SYSTEM_PROMPT,
    QueryEnhancementProtocolError,
    parse_and_validate_query_enhancement,
)

try:
    from . import audit_disc_law_sft as tokenizer_loader
except ImportError:  # 支持在 MiniMind 根目录以 dataset 包运行。
    from dataset import audit_disc_law_sft as tokenizer_loader


DATASET_ROOT = Path(__file__).resolve().parent
QUERY_POOL_ROOT = DATASET_ROOT / "QUERY-POOL"
DEFAULT_INPUT_DIR = (
    QUERY_POOL_ROOT / "pilot" / "query-sft-pilot-v1-work-package"
)
DEFAULT_TEACHER_DIR = DEFAULT_INPUT_DIR / "query-sft-pilot-v1-teacher-candidate-work-package"
DEFAULT_RETRIEVAL_DIR = DEFAULT_TEACHER_DIR / "query-sft-pilot-v1-retrieval-evaluation"
DEFAULT_TOKENIZER_PATH = DATASET_ROOT.parent / "model"
DEFAULT_OUTPUT_DIR = QUERY_POOL_ROOT / "pilot" / "query-sft-pilot-v1-training-release"

MAX_SEQ_LEN = 768
AUTHORING_FILENAME = "query-sft-pilot-v1-authoring.jsonl"
CANDIDATE_FILENAME = "query-sft-pilot-v1-training-candidate.jsonl"
LENGTH_REPORT_FILENAME = "query-sft-pilot-v1-chat-length-audit-768.json"
OVERFLOW_FILENAME = "query-sft-pilot-v1-chat-length-overflow-768.jsonl"
LABEL_REPORT_FILENAME = "query-sft-pilot-v1-label-audit-768.json"
MANIFEST_FILENAME = "query-sft-pilot-v1-pilot-manifest.json"
HASH_FILENAME = "query-sft-pilot-v1-pilot.sha256"

_DRAFT_FIELDS = {"pilot_id", "source_id", "authoring_type", "query_original"}
_REFERENCE_FIELDS = {
    "pilot_id",
    "source_id",
    "authoring_type",
    "coverage_domain",
    "source_query",
    "required_chunk_ids",
    "source_record_sha256",
}
_RESULT_FIELDS = {"candidate_id", "pilot_id", "raw_output"}
_LEDGER_FIELDS = {"candidate_id", "pilot_id", "review_decision", "reason"}
_NOOP_FIELDS = {"candidate_id", "pilot_id", "raw_output"}
_TARGET_FIELDS = ("rewrite", "expansion_terms", "subqueries")


class QuerySftPilotReleaseError(RuntimeError):
    """表示 Query-SFT pilot 的输入、审计或发布状态无效。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if records is not None:
        value["records"] = records
    return value


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftPilotReleaseError(f"无法读取{label}: {path}") from error
    if not isinstance(value, dict):
        raise QuerySftPilotReleaseError(f"{label}必须是 JSON 对象")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    raise QuerySftPilotReleaseError(f"{label}不允许空行: {number}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise QuerySftPilotReleaseError(f"{label}第 {number} 条必须是对象")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftPilotReleaseError):
            raise
        raise QuerySftPilotReleaseError(f"无法读取{label}: {path}") from error
    if not rows:
        raise QuerySftPilotReleaseError(f"{label}不能为空")
    return rows


def _verify_hash(path: Path, label: str) -> Path:
    sidecar = path.with_suffix(".sha256")
    try:
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftPilotReleaseError(f"无法读取{label} SHA-256: {sidecar}") from error
    if lines != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftPilotReleaseError(f"{label} SHA-256 无效")
    return sidecar


def _verify_retrieval_hashes(retrieval_dir: Path) -> dict[str, Path]:
    hash_path = retrieval_dir / "query-sft-pilot-v1-retrieval.sha256"
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftPilotReleaseError("无法读取 Retrieval SHA-256 清单") from error
    expected_names = {
        "query-sft-pilot-v1-retrieval-records.jsonl",
        "query-sft-pilot-v1-retrieval-summary.json",
        "query-sft-pilot-v1-retrieval-manifest.json",
    }
    found: dict[str, Path] = {}
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise QuerySftPilotReleaseError("Retrieval SHA-256 清单格式无效")
        expected, name = parts[0], parts[1].strip()
        target = retrieval_dir / name
        if name not in expected_names or not target.is_file() or _sha256_file(target) != expected:
            raise QuerySftPilotReleaseError("Retrieval SHA-256 清单与当前文件不一致")
        found[name] = target
    if set(found) != expected_names:
        raise QuerySftPilotReleaseError("Retrieval SHA-256 清单范围无效")
    found[hash_path.name] = hash_path
    return found


def _canonical_target(raw_output: str, label: str) -> tuple[dict[str, object], str]:
    try:
        enhancement = parse_and_validate_query_enhancement(raw_output)
    except (TypeError, QueryEnhancementProtocolError) as error:
        raise QuerySftPilotReleaseError(f"{label}不符合严格 Query Enhancement 协议") from error
    target: dict[str, object] = {
        "rewrite": enhancement.rewrite,
        "expansion_terms": list(enhancement.expansion_terms),
        "subqueries": list(enhancement.subqueries),
    }
    if (
        len(enhancement.rewrite) > 112
        or any(len(item) > 16 for item in enhancement.expansion_terms)
        or any(len(item) > 80 for item in enhancement.subqueries)
    ):
        raise QuerySftPilotReleaseError(f"{label}超出 Query-SFT 字段长度上限")
    return target, json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def _load_selected_records(
    *,
    input_dir: Path,
    teacher_dir: Path,
    retrieval_dir: Path,
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    draft_path = input_dir / "query-sft-pilot-v1-query-input-draft.jsonl"
    reference_path = input_dir / "query-sft-pilot-v1-audit-reference.jsonl"
    results_path = teacher_dir / "query-sft-pilot-v1-teacher-candidate-results.jsonl"
    noop_path = teacher_dir / "query-sft-pilot-v1-deterministic-noop-candidates.jsonl"
    ledger_path = teacher_dir / "query-sft-pilot-v1-candidate-semantic-review" / "query-sft-pilot-v1-candidate-semantic-gt-review.jsonl"
    text_audit_path = teacher_dir / "query-sft-pilot-v1-candidate-text-audit.json"

    input_hashes = {
        "draft": _verify_hash(draft_path, "最终输入草稿"),
        "results": _verify_hash(results_path, "教师候选结果"),
        "noop": _verify_hash(noop_path, "确定性 no-op"),
        "ledger": _verify_hash(ledger_path, "教师候选语义审计账本"),
        "text_audit": _verify_hash(text_audit_path, "教师候选文本审计"),
    }
    retrieval_paths = _verify_retrieval_hashes(retrieval_dir)
    text_audit = _load_json(text_audit_path, "教师候选文本审计")
    if (
        text_audit.get("pipeline") != "query_sft_pilot_candidate_text_audit_v1"
        or text_audit.get("complete") is not True
        or text_audit.get("records", {}).get("teacher_candidates_approved") != 114
        or text_audit.get("records", {}).get("evaluation_text_overlap") != 0
    ):
        raise QuerySftPilotReleaseError("教师候选文本审计状态无效")

    drafts = _load_jsonl(draft_path, "最终输入草稿")
    references = _load_jsonl(reference_path, "GT 审核引用")
    if len(drafts) != 40 or len(references) != 40:
        raise QuerySftPilotReleaseError("pilot 输入必须恰好为 40 条")
    approved: dict[str, dict[str, object]] = {}
    for position, (draft, reference) in enumerate(zip(drafts, references, strict=True), 1):
        if set(draft) != _DRAFT_FIELDS or set(reference) != _REFERENCE_FIELDS:
            raise QuerySftPilotReleaseError(f"pilot 输入字段无效: {position}")
        pilot_id = draft.get("pilot_id")
        if (
            not isinstance(pilot_id, str)
            or pilot_id != reference.get("pilot_id")
            or draft.get("source_id") != reference.get("source_id")
            or draft.get("authoring_type") != reference.get("authoring_type")
            or not isinstance(draft.get("query_original"), str)
            or not draft["query_original"].strip()
            or not isinstance(reference.get("required_chunk_ids"), list)
            or not 1 <= len(reference["required_chunk_ids"]) <= 3
            or any(not isinstance(value, str) or not value for value in reference["required_chunk_ids"])
        ):
            raise QuerySftPilotReleaseError(f"pilot 输入映射无效: {position}")
        if pilot_id == "query_sft_pilot:0017":
            continue
        approved[pilot_id] = {
            "source_id": draft["source_id"],
            "authoring_type": draft["authoring_type"],
            "query_original": draft["query_original"],
            "required_chunk_ids": reference["required_chunk_ids"],
        }
    if len(approved) != 39 or "query_sft_pilot:0017" in approved:
        raise QuerySftPilotReleaseError("输入审核通过记录必须恰好为 39 条且排除 0017")

    decisions: dict[str, str] = {}
    for row in _load_jsonl(ledger_path, "教师候选语义审计账本"):
        if set(row) != _LEDGER_FIELDS or not isinstance(row.get("candidate_id"), str):
            raise QuerySftPilotReleaseError("教师候选语义审计账本字段无效")
        if row["candidate_id"] in decisions or row.get("review_decision") not in {"approved", "rejected"}:
            raise QuerySftPilotReleaseError("教师候选语义审计账本决定无效")
        decisions[row["candidate_id"]] = row["review_decision"]
    if len(decisions) != 117 or Counter(decisions.values()) != {"approved": 114, "rejected": 3}:
        raise QuerySftPilotReleaseError("教师候选语义审计计数无效")

    teacher_targets: dict[str, tuple[str, str]] = {}
    for row in _load_jsonl(results_path, "教师候选结果"):
        if set(row) != _RESULT_FIELDS or not isinstance(row.get("candidate_id"), str):
            raise QuerySftPilotReleaseError("教师候选结果字段无效")
        candidate_id = row["candidate_id"]
        if candidate_id in teacher_targets or candidate_id not in decisions:
            raise QuerySftPilotReleaseError("教师候选结果身份无效")
        if decisions[candidate_id] == "approved":
            if row.get("pilot_id") not in approved or not isinstance(row.get("raw_output"), str):
                raise QuerySftPilotReleaseError("已通过教师候选映射无效")
            _, target_json = _canonical_target(row["raw_output"], candidate_id)
            teacher_targets[candidate_id] = (row["pilot_id"], target_json)
    if len(teacher_targets) != 114:
        raise QuerySftPilotReleaseError("已通过教师候选必须恰好为 114 条")

    noop_targets: dict[str, str] = {}
    for row in _load_jsonl(noop_path, "确定性 no-op"):
        if set(row) != _NOOP_FIELDS or not isinstance(row.get("pilot_id"), str) or not isinstance(row.get("raw_output"), str):
            raise QuerySftPilotReleaseError("确定性 no-op 字段无效")
        pilot_id = row["pilot_id"]
        if pilot_id not in approved or pilot_id in noop_targets:
            raise QuerySftPilotReleaseError("确定性 no-op 映射无效")
        _, noop_targets[pilot_id] = _canonical_target(row["raw_output"], row.get("candidate_id", pilot_id))
    if len(noop_targets) != 39:
        raise QuerySftPilotReleaseError("确定性 no-op 必须恰好为 39 条")

    retrieval_manifest = _load_json(
        retrieval_paths["query-sft-pilot-v1-retrieval-manifest.json"], "Retrieval manifest"
    )
    summary = _load_json(
        retrieval_paths["query-sft-pilot-v1-retrieval-summary.json"], "Retrieval summary"
    )
    expected_inputs = retrieval_manifest.get("inputs_frozen_before_run", {})
    for key, path in {
        "draft": draft_path,
        "teacher_results": results_path,
        "noop": noop_path,
        "semantic_ledger": ledger_path,
        "candidate_text_audit": text_audit_path,
    }.items():
        identity = expected_inputs.get(key) if isinstance(expected_inputs, dict) else None
        if not isinstance(identity, dict) or identity.get("bytes") != path.stat().st_size or identity.get("sha256") != _sha256_file(path):
            raise QuerySftPilotReleaseError(f"Retrieval 未绑定当前输入: {key}")
    if (
        retrieval_manifest.get("pipeline") != "query_sft_pilot_retrieval_evaluation_v1"
        or retrieval_manifest.get("complete") is not True
        or retrieval_manifest.get("validation", {}).get("retrieval_assets_unchanged_after_run") is not True
        or retrieval_manifest.get("validation", {}).get("baseline_noop_and_all_approved_candidates_compared") is not True
    ):
        raise QuerySftPilotReleaseError("Retrieval manifest 状态无效")
    selections = summary.get("selections")
    counts = summary.get("records", {}).get("selected_by_kind")
    if not isinstance(selections, list) or len(selections) != 39 or counts != {"teacher_candidate": 23, "noop": 16}:
        raise QuerySftPilotReleaseError("Retrieval 选择汇总计数无效")

    final_records: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    selection_kinds: Counter[str] = Counter()
    for selection in selections:
        if not isinstance(selection, dict):
            raise QuerySftPilotReleaseError("Retrieval 选择记录必须是对象")
        pilot_id = selection.get("pilot_id")
        selected_id = selection.get("selected_variant_id")
        kind = selection.get("selection")
        if not isinstance(pilot_id, str) or pilot_id not in approved or pilot_id in selected_ids:
            raise QuerySftPilotReleaseError("Retrieval 选择的 pilot 无效或重复")
        if kind == "teacher_candidate":
            candidate = teacher_targets.get(selected_id)
            if candidate is None or candidate[0] != pilot_id:
                raise QuerySftPilotReleaseError("Retrieval 选择了未通过或错配的教师候选")
            target_json = candidate[1]
        elif kind == "noop":
            if selected_id != f"{pilot_id}/noop":
                raise QuerySftPilotReleaseError("Retrieval no-op 选择标识无效")
            target_json = noop_targets[pilot_id]
        else:
            raise QuerySftPilotReleaseError("Retrieval 选择类型无效")
        target = json.loads(target_json)
        if tuple(target) != _TARGET_FIELDS:
            raise QuerySftPilotReleaseError("最终 target 字段顺序无效")
        selected_ids.add(pilot_id)
        selection_kinds[kind] += 1
        source = approved[pilot_id]
        final_records.append(
            {
                "id": pilot_id,
                "source_id": source["source_id"],
                "query_original": source["query_original"],
                "required_chunk_ids": source["required_chunk_ids"],
                "target": target,
            }
        )
    if selected_ids != set(approved) or dict(selection_kinds) != {"teacher_candidate": 23, "noop": 16}:
        raise QuerySftPilotReleaseError("最终选择未闭合")
    final_records.sort(key=lambda row: row["id"])
    inputs = {
        "draft": _identity(draft_path, records=40),
        "draft_hash": _identity(input_hashes["draft"]),
        "audit_reference": _identity(reference_path, records=40),
        "teacher_results": _identity(results_path, records=117),
        "teacher_results_hash": _identity(input_hashes["results"]),
        "noop": _identity(noop_path, records=39),
        "noop_hash": _identity(input_hashes["noop"]),
        "semantic_ledger": _identity(ledger_path, records=117),
        "semantic_ledger_hash": _identity(input_hashes["ledger"]),
        "candidate_text_audit": _identity(text_audit_path),
        "candidate_text_audit_hash": _identity(input_hashes["text_audit"]),
        "retrieval_manifest": _identity(retrieval_paths["query-sft-pilot-v1-retrieval-manifest.json"]),
        "retrieval_summary": _identity(retrieval_paths["query-sft-pilot-v1-retrieval-summary.json"]),
        "retrieval_records": _identity(retrieval_paths["query-sft-pilot-v1-retrieval-records.jsonl"], records=192),
        "retrieval_hash": _identity(retrieval_paths["query-sft-pilot-v1-retrieval.sha256"]),
    }
    return final_records, inputs


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=False)
    values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    if not isinstance(values, list) or any(type(value) is not int for value in values):
        raise QuerySftPilotReleaseError("Tokenizer 没有返回整数 input_ids")
    return values


def _subsequence_positions(values: list[int], pattern: list[int]) -> list[int]:
    return [
        index
        for index in range(len(values) - len(pattern) + 1)
        if values[index : index + len(pattern)] == pattern
    ] if pattern else []


def _audit_training_projection(
    records: list[dict[str, object]], tokenizer: Any
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object], list[dict[str, object]]]:
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not isinstance(getattr(tokenizer, "bos_token", None), str) or not isinstance(getattr(tokenizer, "eos_token", None), str) or type(getattr(tokenizer, "pad_token_id", None)) is not int:
        raise QuerySftPilotReleaseError("Tokenizer 缺少 Query-SFT 所需 chat template 身份")
    assistant_prefix = f"{tokenizer.bos_token}assistant\n<think>\n\n</think>\n\n"
    assistant_end = f"{tokenizer.eos_token}\n"
    prefix_ids = _token_ids(tokenizer, assistant_prefix)
    end_ids = _token_ids(tokenizer, assistant_end)
    candidates: list[dict[str, object]] = []
    overflow: list[dict[str, object]] = []
    active_lengths: list[int] = []
    max_prompt_tokens = 0
    for record in records:
        target_json = json.dumps(record["target"], ensure_ascii=False, separators=(",", ":"))
        conversations = [
            {"role": "system", "content": QUERY_ENHANCEMENT_SYSTEM_PROMPT},
            {"role": "user", "content": record["query_original"]},
            {"role": "assistant", "content": target_json},
        ]
        try:
            prompt = tokenizer.apply_chat_template(
                conversations, tokenize=False, add_generation_prompt=False
            )
        except (TypeError, ValueError, KeyError) as error:
            raise QuerySftPilotReleaseError(f"chat template 渲染失败: {record['id']}") from error
        if not isinstance(prompt, str):
            raise QuerySftPilotReleaseError(f"chat template 未返回字符串: {record['id']}")
        expected_tail = f"{assistant_prefix}{target_json}{assistant_end}"
        if not prompt.endswith(expected_tail):
            raise QuerySftPilotReleaseError(f"assistant 模板边界变化: {record['id']}")
        ids = _token_ids(tokenizer, prompt)
        max_prompt_tokens = max(max_prompt_tokens, len(ids))
        if len(ids) > MAX_SEQ_LEN:
            overflow.append({"id": record["id"], "tokens": len(ids)})
            continue
        positions = _subsequence_positions(ids, prefix_ids)
        if len(positions) != 1:
            raise QuerySftPilotReleaseError(f"assistant 前缀不能唯一定位: {record['id']}")
        answer_start = positions[0] + len(prefix_ids)
        answer_end = len(ids) - len(end_ids)
        if answer_end <= answer_start or ids[answer_end:] != end_ids:
            raise QuerySftPilotReleaseError(f"assistant EOS 边界无效: {record['id']}")
        labels = [-100] * MAX_SEQ_LEN
        labels[answer_start : len(ids)] = ids[answer_start:]
        input_ids = ids + [tokenizer.pad_token_id] * (MAX_SEQ_LEN - len(ids))
        active = [index for index, value in enumerate(labels) if value != -100]
        if (
            not active
            or labels[active[0] : active[-1] + 1] != input_ids[active[0] : active[-1] + 1]
            or any(value != -100 for value in labels[: active[0]])
            or any(value != -100 for value in labels[active[-1] + 1 :])
            or labels[active[-len(end_ids)] : active[-1] + 1] != end_ids
            or any(value != tokenizer.pad_token_id for value in input_ids[len(ids) :])
        ):
            raise QuerySftPilotReleaseError(f"assistant-only labels 无效: {record['id']}")
        active_lengths.append(len(active))
        candidates.append(
            {
                "id": record["id"],
                "source": "query_sft",
                "conversations": conversations,
            }
        )
    if overflow:
        raise QuerySftPilotReleaseError(
            "存在超出固定 768 的 pilot 记录: " + ", ".join(item["id"] for item in overflow)
        )
    if len(candidates) != 39 or len(active_lengths) != 39:
        raise QuerySftPilotReleaseError("训练投影记录数未闭合")
    length_report = {
        "pipeline": "query_sft_pilot_chat_length_audit_768_v1",
        "scope": {"fixed_max_seq_len": MAX_SEQ_LEN, "records": 39, "audit_only": True},
        "tokenizer": {"vocab_size": len(tokenizer), "chat_template_sha256": hashlib.sha256(chat_template.encode("utf-8")).hexdigest()},
        "template": {"render": "apply_chat_template_tokenize_false_add_generation_prompt_false", "assistant_start_marker_ids": prefix_ids, "assistant_end_marker_ids": end_ids},
        "records": {"within_limit": 39, "over_limit": 0, "max_prompt_tokens": max_prompt_tokens},
        "readiness": {"chat_template_length_audited": True, "all_records_fit_768": True, "training_ready": False},
        "complete": True,
    }
    label_report = {
        "pipeline": "query_sft_pilot_label_audit_768_v1",
        "scope": {"fixed_max_seq_len": MAX_SEQ_LEN, "records": 39, "label_scope": "assistant JSON + EOS", "system_user_and_padding_masked": True, "audit_only": True},
        "records": {"total_active_label_tokens": sum(active_lengths), "active_label_tokens": {"min": min(active_lengths), "max": max(active_lengths), "mean": round(sum(active_lengths) / len(active_lengths), 4)}},
        "validation": {"expected_records": 39, "scanned_records": 39, "zero_active_label_records": 0, "input_side_label_leak_records": 0, "padding_label_leak_records": 0, "eos_labeled_records": 39, "counts_closed": True},
        "readiness": {"dataset_label_mask_audited": True, "all_records_have_active_assistant_labels": True, "training_ready": False},
        "complete": True,
    }
    return candidates, length_report, label_report, overflow


def _jsonl_payload(records: list[dict[str, object]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records)


def _publish(output_dir: Path, payloads: dict[str, str]) -> None:
    if output_dir.exists():
        raise QuerySftPilotReleaseError(f"发布目录必须不存在: {output_dir}")
    for name, payload in payloads.items():
        try:
            if name.endswith(".json"):
                json.loads(payload)
            elif payload:
                for line in payload.splitlines():
                    json.loads(line)
        except json.JSONDecodeError as error:
            raise QuerySftPilotReleaseError(f"待发布 JSON 载荷无效: {name}") from error
    try:
        output_dir.mkdir(parents=True)
        for name, payload in payloads.items():
            (output_dir / name).write_text(payload, encoding="utf-8", newline="\n")
        hash_payload = "".join(
            f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()}  {name}\n"
            for name, payload in payloads.items()
        )
        (output_dir / HASH_FILENAME).write_text(hash_payload, encoding="utf-8", newline="\n")
    except (OSError, UnicodeError) as error:
        raise QuerySftPilotReleaseError("无法发布 Query-SFT pilot 产物") from error


def finalize_query_sft_pilot(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    teacher_dir: Path = DEFAULT_TEACHER_DIR,
    retrieval_dir: Path = DEFAULT_RETRIEVAL_DIR,
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tokenizer: Any | None = None,
) -> dict[str, object]:
    """闭合候选来源、训练投影、长度和 labels，并发布仅限 pilot 的 manifest。"""

    input_dir = Path(input_dir).resolve()
    teacher_dir = Path(teacher_dir).resolve()
    retrieval_dir = Path(retrieval_dir).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    output_dir = Path(output_dir).resolve()
    records, inputs = _load_selected_records(
        input_dir=input_dir, teacher_dir=teacher_dir, retrieval_dir=retrieval_dir
    )
    tokenizer = tokenizer or tokenizer_loader.load_tokenizer(tokenizer_path)
    candidates, length_report, label_report, overflow = _audit_training_projection(records, tokenizer)
    if overflow:
        raise QuerySftPilotReleaseError("超长记录未隔离，禁止发布 pilot")
    authoring_payload = _jsonl_payload(records)
    candidate_payload = _jsonl_payload(candidates)
    length_payload = json.dumps(length_report, ensure_ascii=False, indent=2) + "\n"
    label_payload = json.dumps(label_report, ensure_ascii=False, indent=2) + "\n"
    overflow_payload = ""
    manifest = {
        "pipeline": "query_sft_pilot_training_release_v1",
        "release_status": "pilot_training_candidate",
        "scope": {"pilot_only": True, "formal_full_query_sft_published": False, "training_started": False},
        "inputs": inputs,
        "outputs": {
            "authoring": {"file": AUTHORING_FILENAME, "records": 39, "sha256": hashlib.sha256(authoring_payload.encode("utf-8")).hexdigest()},
            "training_candidate": {"file": CANDIDATE_FILENAME, "records": 39, "sha256": hashlib.sha256(candidate_payload.encode("utf-8")).hexdigest()},
        },
        "records": {"input_total": 40, "input_approved": 39, "input_rejected": 1, "teacher_candidates_total": 117, "teacher_candidates_approved": 114, "teacher_candidates_rejected": 3, "selected_teacher_candidates": 23, "selected_noop": 16, "final_authoring": 39},
        "retrieval": {"identity": _load_json(retrieval_dir / "query-sft-pilot-v1-retrieval-manifest.json", "Retrieval manifest")["retrieval_identity"], "selection_policy": _load_json(retrieval_dir / "query-sft-pilot-v1-retrieval-summary.json", "Retrieval summary")["selection_policy"]},
        "readiness": {"input_semantic_gt_review_complete": True, "teacher_protocol_semantic_text_audit_complete": True, "frozen_retrieval_selection_complete": True, "chat_template_length_audited": True, "dataset_label_mask_audited": True, "pilot_training_compatible": True, "formal_full_query_sft_training_ready": False},
        "limitations": ["本发布只覆盖 40 条 pilot 的 39 条通过记录。", "正式全量 Query-SFT、RAG-SFT clean 发布与模型训练均不属于本产物。"],
        "complete": True,
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    _publish(output_dir, {
        AUTHORING_FILENAME: authoring_payload,
        CANDIDATE_FILENAME: candidate_payload,
        LENGTH_REPORT_FILENAME: length_payload,
        OVERFLOW_FILENAME: overflow_payload,
        LABEL_REPORT_FILENAME: label_payload,
        MANIFEST_FILENAME: manifest_payload,
    })
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--teacher-dir", type=Path, default=DEFAULT_TEACHER_DIR)
    parser.add_argument("--retrieval-dir", type=Path, default=DEFAULT_RETRIEVAL_DIR)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        manifest = finalize_query_sft_pilot(
            input_dir=args.input_dir,
            teacher_dir=args.teacher_dir,
            retrieval_dir=args.retrieval_dir,
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
        )
    except QuerySftPilotReleaseError as error:
        parser.error(str(error))
    print("[完成] Query-SFT pilot 发布完成: authoring=39, teacher=23, noop=16")
    print(f"manifest: {args.output_dir / MANIFEST_FILENAME}")
    print(f"pilot_training_compatible={manifest['readiness']['pilot_training_compatible']}")


if __name__ == "__main__":
    main()
