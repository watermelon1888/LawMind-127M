"""为唯一 RAG-SFT 候选发布匿名、可审计的回答语义评审工作包。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping


PIPELINE = "rag_sft_v2_answer_semantic_review_work_package_v1"
EXPECTED_MODEL_ID = "cpt_250m/rag_epoch_2"
EXPECTED_PRIMARY = 166
EXPECTED_PRIMARY_VALID = 158
EXPECTED_PRIMARY_INVALID = 8
EXPECTED_PRIMARY_CLEAN_VALID = 34
EXPECTED_PRIMARY_HN_VALID = 124
EXPECTED_PRIMARY_ANSWERABLE_VALID = 123
EXPECTED_PRIMARY_INCOMPLETE_VALID = 35
EXPECTED_ROPE = 4
EXPECTED_ROPE_VALID = 3
EXPECTED_ROPE_INVALID = 1
REVIEW_SEED = 42
PRIMARY_BATCH_SIZES = (20, 20, 20, 20, 20, 20, 20, 18)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = (
    PROJECT_ROOT
    / "rag"
    / "eval"
    / "results"
    / "rag-sft-stage-f-775-five-parent-v2-20260814"
)
DEFAULT_EXECUTION_PATH = RESULT_ROOT / "stage-f-execution.json"
DEFAULT_EVALUATION_MANIFEST_PATH = (
    PROJECT_ROOT
    / "rag"
    / "eval"
    / "results"
    / "rag-sft-v2-evaluation-inputs-compact-v3-dual-context-v1"
    / "manifest.json"
)
DEFAULT_CASES_PATH = DEFAULT_EVALUATION_MANIFEST_PATH.with_name("cases.jsonl")
DEFAULT_REPORT_PATH = (
    RESULT_ROOT
    / "evaluation"
    / "models"
    / "cpt_250m"
    / "rag_epoch_2"
    / "report.json"
)
DEFAULT_AGGREGATE_SUMMARY_PATH = RESULT_ROOT / "evaluation" / "aggregate" / "summary.json"
DEFAULT_RUBRIC_PATH = Path(__file__).with_name("rag_sft_v2_semantic_review_rubric.md")
DEFAULT_OUTPUT_DIR = (
    RESULT_ROOT
    / "semantic-review"
    / "cpt-250m-rag-epoch-2-v1"
)

CASE_FIELDS = (
    "query_id",
    "query_type",
    "evaluation_scope",
    "query",
    "evidence",
    "visible_chunk_ids",
    "required_chunk_ids",
    "hard_negative_chunk_ids",
    "retrieval_attribution",
)
REPORT_RECORD_FIELDS = (
    "query_id",
    "query_type",
    "evaluation_scope",
    "visible_chunk_ids",
    "required_chunk_ids",
    "hard_negative_chunk_ids",
    "retrieval_attribution",
    "raw_output",
    "generation_metrics",
    "protocol_valid",
    "parsed",
    "cited_chunk_ids",
    "citation_metrics",
    "error",
)
REVIEWER_VISIBLE_FIELDS = ("review_id", "query", "evidence", "assistant")
AUDIT_REFERENCE_FIELDS = (
    "review_id",
    "query_id",
    "query_type",
    "evaluation_scope",
    "visible_chunk_ids",
    "required_chunk_ids",
    "hard_negative_chunk_ids",
    "retrieval_attribution",
    "raw_output",
    "generation_metrics",
    "citation_metrics",
)
REVIEW_RESULT_FIELDS = (
    "review_id",
    "review_decision",
    "answerability",
    "atomic_claims",
    "necessary_matters",
    "citation_findings",
    "query_responsive",
    "legal_boundaries_preserved",
    "unsupported_fact_absent",
    "error_attribution",
    "error_tags",
    "reason",
)
FORBIDDEN_BLIND_KEYS = {
    "query_id",
    "query_type",
    "evaluation_scope",
    "chunk_id",
    "visible_chunk_ids",
    "required_chunk_ids",
    "hard_negative_chunk_ids",
    "retrieval_attribution",
    "raw_output",
    "generation_metrics",
    "citation_metrics",
    "model_id",
    "weights_sha256",
}


class RagSftV2SemanticReviewPreparationError(RuntimeError):
    """表示回答语义评审工作包无法安全发布。"""


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


def _verify_sidecar(path: Path, label: str) -> Path:
    resolved = path.resolve()
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    try:
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftV2SemanticReviewPreparationError(
            f"无法读取{label}相邻 SHA-256"
        ) from error
    if lines != [f"{_sha256_file(resolved)}  {resolved.name}"]:
        raise RagSftV2SemanticReviewPreparationError(f"{label} SHA-256 校验失败")
    return sidecar


def _load_json(path: Path, label: str, *, verify: bool = True) -> dict[str, Any]:
    resolved = path.resolve()
    if verify:
        _verify_sidecar(resolved, label)
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftV2SemanticReviewPreparationError(f"无法读取{label}") from error
    if not isinstance(value, dict):
        raise RagSftV2SemanticReviewPreparationError(f"{label}必须是 JSON 对象")
    return value


def _load_jsonl(path: Path, label: str, *, verify: bool = True) -> list[dict[str, Any]]:
    resolved = path.resolve()
    if verify:
        _verify_sidecar(resolved, label)
    rows = []
    try:
        with resolved.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    raise RagSftV2SemanticReviewPreparationError(
                        f"{label}不允许空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RagSftV2SemanticReviewPreparationError(
                        f"{label}第 {line_number} 条必须是对象"
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftV2SemanticReviewPreparationError):
            raise
        raise RagSftV2SemanticReviewPreparationError(f"无法读取{label}") from error
    return rows


def _jsonl_payload(rows: list[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in rows
    )


def _blind_evidence(case: Mapping[str, Any]) -> list[dict[str, object]]:
    output = []
    for index, item in enumerate(case["evidence"], 1):
        if (
            not isinstance(item, dict)
            or item.get("evidence_id") != f"E{index}"
            or not all(
                isinstance(item.get(name), str) and item[name].strip()
                for name in ("chunk_id", "law_name", "article_no", "content")
            )
        ):
            raise RagSftV2SemanticReviewPreparationError("评估 Evidence 结构无效")
        output.append(
            {
                "evidence_id": item["evidence_id"],
                "law_name": item["law_name"],
                "article_no": item["article_no"],
                "excerpts": [item["content"]],
            }
        )
    if [item["chunk_id"] for item in case["evidence"]] != case["visible_chunk_ids"]:
        raise RagSftV2SemanticReviewPreparationError("Evidence 与 visible_chunk_ids 不一致")
    return output


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(FORBIDDEN_BLIND_KEYS.intersection(value)) or any(
            _contains_forbidden_key(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def _validate_inputs(
    *,
    execution: Mapping[str, Any],
    aggregate_summary: Mapping[str, Any],
    evaluation_manifest: Mapping[str, Any],
    cases_path: Path,
    cases: list[dict[str, Any]],
    report: Mapping[str, Any],
    report_path: Path,
    model_id: str,
) -> str:
    if (
        execution.get("pipeline") != "rag_sft_v2_stage_f_execution_v2"
        or execution.get("complete") is not True
        or execution.get("expected_answers") != 3400
    ):
        raise RagSftV2SemanticReviewPreparationError("阶段 F execution 状态无效")
    models = execution.get("evaluation_models")
    matched = (
        [item for item in models if item.get("model_id") == model_id]
        if isinstance(models, list)
        else []
    )
    if model_id != EXPECTED_MODEL_ID or len(matched) != 1:
        raise RagSftV2SemanticReviewPreparationError("语义评审模型身份无效")
    if (
        aggregate_summary.get("pipeline")
        != "rag_sft_v2_stage_f_evaluation_aggregate_v2"
        or aggregate_summary.get("complete") is not True
        or aggregate_summary.get("models") != 20
        or aggregate_summary.get("answers") != 3400
        or aggregate_summary.get("selection_applied") is not False
        or aggregate_summary.get("private_holdout_used") is not False
    ):
        raise RagSftV2SemanticReviewPreparationError("阶段 F 聚合报告状态无效")
    aggregate_results = aggregate_summary.get("results")
    aggregate_match = (
        [item for item in aggregate_results if item.get("model_id") == model_id]
        if isinstance(aggregate_results, list)
        else []
    )
    if (
        len(aggregate_match) != 1
        or aggregate_match[0].get("report", {}).get("sha256")
        != _sha256_file(report_path)
    ):
        raise RagSftV2SemanticReviewPreparationError("候选报告未与阶段 F 聚合身份闭合")
    if (
        evaluation_manifest.get("pipeline") != "rag_sft_v2_evaluation_inputs_v1"
        or evaluation_manifest.get("complete") is not True
        or evaluation_manifest.get("records", {}).get("total") != 170
    ):
        raise RagSftV2SemanticReviewPreparationError("评估输入 manifest 状态无效")
    bound_cases = evaluation_manifest.get("output", {}).get("cases")
    if (
        not isinstance(bound_cases, dict)
        or bound_cases.get("sha256") != _sha256_file(cases_path)
        or bound_cases.get("records") != 170
        or len(cases) != 170
    ):
        raise RagSftV2SemanticReviewPreparationError("评估 cases 未与 manifest 闭合")
    if (
        report.get("pipeline") != "rag_sft_v2_answer_model_evaluation"
        or report.get("complete") is not True
        or report.get("summary", {}).get("records") != EXPECTED_PRIMARY
        or report.get("summary", {}).get("total_generated_records") != 170
    ):
        raise RagSftV2SemanticReviewPreparationError("候选模型评估报告状态无效")
    weights_sha256 = report.get("inputs", {}).get("weights_sha256")
    if not isinstance(weights_sha256, str) or len(weights_sha256) != 64:
        raise RagSftV2SemanticReviewPreparationError("候选权重 SHA-256 身份无效")
    generation = report.get("generation")
    if (
        not isinstance(generation, dict)
        or generation.get("max_new_tokens") != 150
        or generation.get("do_sample") is not False
        or generation.get("primary_max_seq_len") != 768
        or generation.get("extrapolation_max_seq_len") != 1024
        or generation.get("retry") is not False
        or generation.get("repair") is not False
    ):
        raise RagSftV2SemanticReviewPreparationError("候选评估解码配置无效")
    return weights_sha256


def _validate_report_binding(
    *,
    report: Mapping[str, Any],
    evaluation_manifest_path: Path,
    cases_path: Path,
) -> None:
    bound = report.get("inputs", {}).get("evaluation")
    if (
        not isinstance(bound, dict)
        or bound.get("manifest_sha256") != _sha256_file(evaluation_manifest_path)
        or bound.get("cases_sha256") != _sha256_file(cases_path)
    ):
        raise RagSftV2SemanticReviewPreparationError("模型报告未绑定当前评估输入")


def _record_pair(
    case: Mapping[str, Any], report_record: Mapping[str, Any]
) -> tuple[dict[str, object], dict[str, object]]:

    parsed = report_record.get("parsed")
    if (
        report_record.get("protocol_valid") is not True
        or not isinstance(parsed, dict)
        or tuple(parsed) != ("summary", "citations")
        or not isinstance(parsed["summary"], str)
        or not parsed["summary"].strip()
        or not isinstance(parsed["citations"], list)
        or not parsed["citations"]
    ):
        raise RagSftV2SemanticReviewPreparationError("协议合法记录的 parsed 结构无效")
    evidence = _blind_evidence(case)
    evidence_ids = {item["evidence_id"] for item in evidence}
    if (
        any(not isinstance(item, str) for item in parsed["citations"])
        or not set(parsed["citations"]).issubset(evidence_ids)
    ):
        raise RagSftV2SemanticReviewPreparationError("协议合法记录包含无效 citation")
    blind = {
        "review_id": "",
        "query": case["query"],
        "evidence": evidence,
        "assistant": {
            "summary": parsed["summary"],
            "citations": parsed["citations"],
        },
    }
    audit = {
        "review_id": "",
        "query_id": case["query_id"],
        "query_type": case["query_type"],
        "evaluation_scope": case["evaluation_scope"],
        "visible_chunk_ids": case["visible_chunk_ids"],
        "required_chunk_ids": case["required_chunk_ids"],
        "hard_negative_chunk_ids": case["hard_negative_chunk_ids"],
        "retrieval_attribution": case["retrieval_attribution"],
        "raw_output": report_record["raw_output"],
        "generation_metrics": report_record["generation_metrics"],
        "citation_metrics": report_record["citation_metrics"],
    }
    return blind, audit


def _validate_case_report_identity(
    case: Mapping[str, Any], report_record: Mapping[str, Any]
) -> None:
    if tuple(case) != CASE_FIELDS or tuple(report_record) != REPORT_RECORD_FIELDS:
        raise RagSftV2SemanticReviewPreparationError("评估 case 或模型记录字段无效")
    for name in (
        "query_id",
        "query_type",
        "evaluation_scope",
        "visible_chunk_ids",
        "required_chunk_ids",
        "hard_negative_chunk_ids",
        "retrieval_attribution",
    ):
        if case[name] != report_record[name]:
            raise RagSftV2SemanticReviewPreparationError(
                f"模型记录未与评估 case 对齐: {case.get('query_id')}"
            )


def _invalid_record(case: Mapping[str, Any], report_record: Mapping[str, Any]) -> dict[str, object]:
    if report_record.get("protocol_valid") is not False or report_record.get("parsed") is not None:
        raise RagSftV2SemanticReviewPreparationError("协议失败记录状态无效")
    return {
        "query_id": case["query_id"],
        "query_type": case["query_type"],
        "evaluation_scope": case["evaluation_scope"],
        "query": case["query"],
        "evidence": _blind_evidence(case),
        "raw_output": report_record["raw_output"],
        "generation_metrics": report_record["generation_metrics"],
        "error": report_record["error"],
        "retrieval_attribution": case["retrieval_attribution"],
        "required_chunk_ids": case["required_chunk_ids"],
        "hard_negative_chunk_ids": case["hard_negative_chunk_ids"],
    }


def _readme() -> str:
    return f"""# RAG-SFT v2 回答语义评审工作包

本工作包只评审 `{EXPECTED_MODEL_ID}`。评审者只能接收 `rubric.md` 与 `blind/` 下的文件；不得接收 `audit-reference/`、`automatic/`、manifest 或既有自动指标。

主评估共有 {EXPECTED_PRIMARY} 题，其中 {EXPECTED_PRIMARY_VALID} 条协议合法输出进入 8 批盲审，{EXPECTED_PRIMARY_INVALID} 条协议失败由程序归档。RoPE 外推题位于 `diagnostic/`，只作诊断，不参与模型门禁。

审核结果必须按 manifest 中的 `reviewer_output_fields` 输出单行 UTF-8 JSONL。不得修改队列中的 `review_id`，不得读取其他批次的审核结论。
"""


def _review_instructions() -> str:
    return """# 主审输出说明

每个输入 batch 对应一个同名结果文件。每行必须是单个 JSON 对象，字段顺序固定如下：

```json
{"review_id":"保持队列值","review_decision":"pass|fail|escalate","answerability":"sufficient|insufficient|uncertain","atomic_claims":[{"claim":"原子主张","support":"fully_supported|partially_supported|unsupported","supporting_evidence_ids":["E1"]}],"necessary_matters":[{"matter":"必要事项","covered":true}],"citation_findings":[{"evidence_id":"E1","supports_any_claim":true,"necessary_for_summary":true}],"query_responsive":true,"legal_boundaries_preserved":true,"unsupported_fact_absent":true,"error_attribution":"none|retrieval_blocked|model_semantic_failure|both|uncertain","error_tags":[],"reason":"简短决定性理由"}
```

`citation_findings` 必须按模型 citations 的原顺序逐项覆盖。`supporting_evidence_ids` 只能使用模型已经引用的 Evidence。通过项必须满足 rubric 的全部通过条件，且 `error_tags` 为空。

完成 8 个结果分片后，使用 `python -m rag.eval.finalize_rag_sft_v2_semantic_review seal-run` 生成不可变 `review-run.json` 与相邻 SHA-256；不要手工填写文件哈希。
"""


def prepare_rag_sft_v2_semantic_review(
    *,
    execution_path: Path = DEFAULT_EXECUTION_PATH,
    aggregate_summary_path: Path = DEFAULT_AGGREGATE_SUMMARY_PATH,
    evaluation_manifest_path: Path = DEFAULT_EVALUATION_MANIFEST_PATH,
    cases_path: Path = DEFAULT_CASES_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    rubric_path: Path = DEFAULT_RUBRIC_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_id: str = EXPECTED_MODEL_ID,
) -> dict[str, object]:
    """发布唯一候选的主评估盲审队列、隐藏引用和诊断附件。"""

    execution_path = Path(execution_path).resolve()
    aggregate_summary_path = Path(aggregate_summary_path).resolve()
    evaluation_manifest_path = Path(evaluation_manifest_path).resolve()
    cases_path = Path(cases_path).resolve()
    report_path = Path(report_path).resolve()
    rubric_path = Path(rubric_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RagSftV2SemanticReviewPreparationError("语义评审输出目录必须不存在")
    execution = _load_json(execution_path, "阶段 F execution")
    aggregate_summary = _load_json(aggregate_summary_path, "阶段 F 聚合报告")
    evaluation_manifest = _load_json(evaluation_manifest_path, "评估输入 manifest")
    cases = _load_jsonl(cases_path, "评估 cases")
    report = _load_json(report_path, "候选模型评估报告")
    try:
        rubric = rubric_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftV2SemanticReviewPreparationError("无法读取语义评审规则") from error
    if not rubric.endswith("\n"):
        raise RagSftV2SemanticReviewPreparationError("语义评审规则必须以换行结束")
    weights_sha256 = _validate_inputs(
        execution=execution,
        aggregate_summary=aggregate_summary,
        evaluation_manifest=evaluation_manifest,
        cases_path=cases_path,
        cases=cases,
        report=report,
        report_path=report_path,
        model_id=model_id,
    )
    _validate_report_binding(
        report=report,
        evaluation_manifest_path=evaluation_manifest_path,
        cases_path=cases_path,
    )
    report_records = report.get("records")
    if not isinstance(report_records, list) or len(report_records) != len(cases):
        raise RagSftV2SemanticReviewPreparationError("模型逐题记录数量无效")

    primary_pairs = []
    primary_invalid = []
    rope_pairs = []
    rope_invalid = []
    seen = set()
    for case, record in zip(cases, report_records):
        _validate_case_report_identity(case, record)
        identity = (case.get("query_id"), case.get("evaluation_scope"))
        if identity in seen:
            raise RagSftV2SemanticReviewPreparationError("评估记录身份重复")
        seen.add(identity)
        if record.get("protocol_valid") is True:
            pair = _record_pair(case, record)
            if case["evaluation_scope"] == "primary_768":
                primary_pairs.append(pair)
            elif case["evaluation_scope"] == "rope_extrapolation_1024":
                rope_pairs.append(pair)
            else:
                raise RagSftV2SemanticReviewPreparationError("未知评估 scope")
        else:
            invalid = _invalid_record(case, record)
            if case["evaluation_scope"] == "primary_768":
                primary_invalid.append(invalid)
            elif case["evaluation_scope"] == "rope_extrapolation_1024":
                rope_invalid.append(invalid)
            else:
                raise RagSftV2SemanticReviewPreparationError("未知评估 scope")

    clean_valid = sum(not audit["hard_negative_chunk_ids"] for _, audit in primary_pairs)
    hn_valid = len(primary_pairs) - clean_valid
    complete_valid = sum(
        audit["query_type"] == "exact_lookup"
        or audit["retrieval_attribution"].get("packaged_complete") is True
        for _, audit in primary_pairs
    )
    incomplete_valid = len(primary_pairs) - complete_valid
    actual_counts = (
        len(primary_pairs),
        len(primary_invalid),
        clean_valid,
        hn_valid,
        complete_valid,
        incomplete_valid,
        len(rope_pairs),
        len(rope_invalid),
    )
    expected_counts = (
        EXPECTED_PRIMARY_VALID,
        EXPECTED_PRIMARY_INVALID,
        EXPECTED_PRIMARY_CLEAN_VALID,
        EXPECTED_PRIMARY_HN_VALID,
        EXPECTED_PRIMARY_ANSWERABLE_VALID,
        EXPECTED_PRIMARY_INCOMPLETE_VALID,
        EXPECTED_ROPE_VALID,
        EXPECTED_ROPE_INVALID,
    )
    if actual_counts != expected_counts:
        raise RagSftV2SemanticReviewPreparationError(
            f"候选语义评审分层计数不闭合: {actual_counts}"
        )

    primary_pairs.sort(key=lambda item: item[1]["query_id"])
    random.Random(REVIEW_SEED).shuffle(primary_pairs)
    for index, (blind, audit) in enumerate(primary_pairs, 1):
        review_id = f"RSV2-P-{index:03d}"
        blind["review_id"] = review_id
        audit["review_id"] = review_id
        if tuple(blind) != REVIEWER_VISIBLE_FIELDS or _contains_forbidden_key(blind):
            raise RagSftV2SemanticReviewPreparationError("盲审队列发生隐藏字段泄漏")
        if tuple(audit) != AUDIT_REFERENCE_FIELDS:
            raise RagSftV2SemanticReviewPreparationError("隐藏审核引用字段无效")
    rope_pairs.sort(key=lambda item: item[1]["query_id"])
    for index, (blind, audit) in enumerate(rope_pairs, 1):
        review_id = f"RSV2-R-{index:03d}"
        blind["review_id"] = review_id
        audit["review_id"] = review_id
        if tuple(blind) != REVIEWER_VISIBLE_FIELDS or _contains_forbidden_key(blind):
            raise RagSftV2SemanticReviewPreparationError("RoPE 盲审队列发生隐藏字段泄漏")

    payloads: dict[str, str] = {
        "README.md": _readme(),
        "rubric.md": rubric,
        "blind/review-instructions.md": _review_instructions(),
        "automatic/protocol-invalid-primary.jsonl": _jsonl_payload(primary_invalid),
        "diagnostic/rope-review-queue.jsonl": _jsonl_payload(
            [blind for blind, _ in rope_pairs]
        ),
        "diagnostic/rope-audit-reference.jsonl": _jsonl_payload(
            [audit for _, audit in rope_pairs]
        ),
        "diagnostic/rope-protocol-invalid.jsonl": _jsonl_payload(rope_invalid),
    }
    batches = []
    offset = 0
    for batch, size in enumerate(PRIMARY_BATCH_SIZES, 1):
        values = primary_pairs[offset : offset + size]
        offset += size
        queue_name = f"blind/review-queue/batch-{batch:02d}.jsonl"
        audit_name = f"audit-reference/batch-{batch:02d}.jsonl"
        payloads[queue_name] = _jsonl_payload([blind for blind, _ in values])
        payloads[audit_name] = _jsonl_payload([audit for _, audit in values])
        batches.append(
            {
                "batch": batch,
                "records": size,
                "review_queue": queue_name,
                "audit_reference": audit_name,
            }
        )
    if offset != EXPECTED_PRIMARY_VALID:
        raise RagSftV2SemanticReviewPreparationError("盲审 batch 计数无效")

    manifest = {
        "schema_version": "1.0",
        "pipeline": PIPELINE,
        "release_status": "semantic_review_pending",
        "inputs": {
            "stage_f_execution": _identity(execution_path),
            "aggregate_summary": _identity(aggregate_summary_path),
            "evaluation_manifest": _identity(evaluation_manifest_path),
            "evaluation_cases": _identity(cases_path, records=170),
            "model_report": _identity(report_path, records=170),
            "rubric_source": _identity(rubric_path),
            "preparation_source": _identity(Path(__file__).resolve()),
        },
        "candidate": {
            "model_id": model_id,
            "weights_sha256": weights_sha256,
            "selection_scope": "single_semantic_review_candidate",
        },
        "review_protocol": {
            "seed": REVIEW_SEED,
            "reviewer_visible_fields": list(REVIEWER_VISIBLE_FIELDS),
            "reviewer_output_fields": list(REVIEW_RESULT_FIELDS),
            "forbidden_blind_keys": sorted(FORBIDDEN_BLIND_KEYS),
            "primary_review": "all_protocol_valid_primary_records",
            "secondary_review": "all_fail_or_escalate_plus_deterministic_20_percent_pass",
            "disagreement": "adjudication_required",
        },
        "records": {
            "generated": 170,
            "primary": EXPECTED_PRIMARY,
            "primary_protocol_valid": EXPECTED_PRIMARY_VALID,
            "primary_protocol_invalid": EXPECTED_PRIMARY_INVALID,
            "primary_clean_valid": EXPECTED_PRIMARY_CLEAN_VALID,
            "primary_hn_valid": EXPECTED_PRIMARY_HN_VALID,
            "primary_answerable_valid": EXPECTED_PRIMARY_ANSWERABLE_VALID,
            "primary_packaged_incomplete_valid": EXPECTED_PRIMARY_INCOMPLETE_VALID,
            "rope": EXPECTED_ROPE,
            "rope_protocol_valid": EXPECTED_ROPE_VALID,
            "rope_protocol_invalid": EXPECTED_ROPE_INVALID,
            "batches": len(PRIMARY_BATCH_SIZES),
        },
        "batches": batches,
        "selection_policy": {
            "primary_768_used_for_gate": True,
            "rope_extrapolation_used_for_gate": False,
            "packaged_incomplete_used_for_answer_completeness": False,
            "protocol_invalid_requires_manual_review": False,
            "automatic_model_acceptance": False,
        },
        "readiness": {
            "primary_semantic_review_ready": True,
            "secondary_review_ready": False,
            "private_holdout_used": False,
        },
        "complete": True,
    }
    payloads["manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    try:
        output_dir.mkdir(parents=True)
        for name, payload in payloads.items():
            path = output_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8", newline="\n")
        (output_dir / "manifest.sha256").write_text(
            "".join(
                f"{hashlib.sha256(payloads[name].encode('utf-8')).hexdigest()}  {name}\n"
                for name in sorted(payloads)
            ),
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, UnicodeError) as error:
        raise RagSftV2SemanticReviewPreparationError("无法发布语义评审工作包") from error
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, default=DEFAULT_EXECUTION_PATH)
    parser.add_argument(
        "--aggregate-summary", type=Path, default=DEFAULT_AGGREGATE_SUMMARY_PATH
    )
    parser.add_argument(
        "--evaluation-manifest", type=Path, default=DEFAULT_EVALUATION_MANIFEST_PATH
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=EXPECTED_MODEL_ID)
    args = parser.parse_args()
    try:
        manifest = prepare_rag_sft_v2_semantic_review(
            execution_path=args.execution,
            aggregate_summary_path=args.aggregate_summary,
            evaluation_manifest_path=args.evaluation_manifest,
            cases_path=args.cases,
            report_path=args.report,
            rubric_path=args.rubric,
            output_dir=args.output_dir,
            model_id=args.model_id,
        )
    except RagSftV2SemanticReviewPreparationError as error:
        parser.error(str(error))
    print(
        "RAG_SFT_V2_SEMANTIC_REVIEW_PACKAGE_OK "
        f"records={manifest['records']['primary_protocol_valid']}"
    )


if __name__ == "__main__":
    main()
