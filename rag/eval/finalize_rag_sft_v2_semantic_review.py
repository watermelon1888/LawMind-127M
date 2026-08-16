"""校验 RAG-SFT v2 主审结果，生成二审/裁决队列并发布语义评审账本。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from rag.eval.prepare_rag_sft_v2_semantic_review import (
    AUDIT_REFERENCE_FIELDS,
    DEFAULT_OUTPUT_DIR as DEFAULT_WORK_PACKAGE_DIR,
    EXPECTED_PRIMARY_ANSWERABLE_VALID,
    EXPECTED_PRIMARY_INCOMPLETE_VALID,
    EXPECTED_PRIMARY_INVALID,
    EXPECTED_PRIMARY_VALID,
    PIPELINE as WORK_PACKAGE_PIPELINE,
    PRIMARY_BATCH_SIZES,
    REVIEWER_VISIBLE_FIELDS,
    REVIEW_RESULT_FIELDS,
)


RUN_PIPELINE = "rag_sft_v2_answer_semantic_review_run_v1"
SECONDARY_PIPELINE = "rag_sft_v2_answer_semantic_secondary_work_package_v1"
ADJUDICATION_PIPELINE = "rag_sft_v2_answer_semantic_adjudication_work_package_v1"
FINAL_PIPELINE = "rag_sft_v2_answer_semantic_review_audit_v1"
SECONDARY_SEED = 43
PASS_AUDIT_RATIO = 0.2

REVIEWER_IDENTITY_FIELDS = (
    "reviewer_id",
    "reviewer_role",
    "review_method",
    "independence_declaration",
)
ATOMIC_CLAIM_FIELDS = ("claim", "support", "supporting_evidence_ids")
NECESSARY_MATTER_FIELDS = ("matter", "covered")
CITATION_FINDING_FIELDS = (
    "evidence_id",
    "supports_any_claim",
    "necessary_for_summary",
)
DECISIONS = {"pass", "fail", "escalate"}
ANSWERABILITY = {"sufficient", "insufficient", "uncertain"}
CLAIM_SUPPORT = {"fully_supported", "partially_supported", "unsupported"}
ERROR_ATTRIBUTIONS = {
    "none",
    "retrieval_blocked",
    "model_semantic_failure",
    "both",
    "uncertain",
}
ERROR_TAGS = {
    "unsupported_claim",
    "partially_supported_claim",
    "missing_necessary_matter",
    "unnecessary_citation",
    "insufficient_citation",
    "unresponsive_answer",
    "condition_or_exception_lost",
    "unsupported_fact",
    "evidence_insufficient",
    "manual_review_required",
    "other",
}


class RagSftV2SemanticReviewFinalizationError(RuntimeError):
    """表示语义评审结果、复核范围或裁决链未闭合。"""


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
        raise RagSftV2SemanticReviewFinalizationError(f"无法读取{label}") from error
    if not isinstance(value, dict):
        raise RagSftV2SemanticReviewFinalizationError(f"{label}必须是 JSON 对象")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    raise RagSftV2SemanticReviewFinalizationError(
                        f"{label}不允许空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RagSftV2SemanticReviewFinalizationError(
                        f"{label}第 {line_number} 条必须是对象"
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftV2SemanticReviewFinalizationError):
            raise
        raise RagSftV2SemanticReviewFinalizationError(f"无法读取{label}") from error
    return rows


def _jsonl_payload(rows: list[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in rows
    )


def _verify_hash_manifest(directory: Path, expected_names: set[str], label: str) -> Path:
    hash_path = directory / "manifest.sha256"
    try:
        entries = {}
        for line in hash_path.read_text(encoding="utf-8").splitlines():
            digest, name = line.split("  ", 1)
            if name in entries:
                raise ValueError("重复文件")
            entries[name] = digest
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise RagSftV2SemanticReviewFinalizationError(
            f"{label} SHA-256 清单无效"
        ) from error
    if set(entries) != expected_names:
        raise RagSftV2SemanticReviewFinalizationError(
            f"{label} SHA-256 清单未精确覆盖受签文件"
        )
    for name, digest in entries.items():
        path = directory / name
        if not path.is_file() or _sha256_file(path) != digest:
            raise RagSftV2SemanticReviewFinalizationError(f"{label}身份已变化: {name}")
    return hash_path


def _primary_expected_names() -> set[str]:
    return {
        "README.md",
        "rubric.md",
        "blind/review-instructions.md",
        "manifest.json",
        "automatic/protocol-invalid-primary.jsonl",
        "diagnostic/rope-review-queue.jsonl",
        "diagnostic/rope-audit-reference.jsonl",
        "diagnostic/rope-protocol-invalid.jsonl",
        *(f"blind/review-queue/batch-{batch:02d}.jsonl" for batch in range(1, 9)),
        *(f"audit-reference/batch-{batch:02d}.jsonl" for batch in range(1, 9)),
    }


def _verify_primary_package(directory: Path) -> tuple[Path, dict[str, Any]]:
    hash_path = _verify_hash_manifest(
        directory, _primary_expected_names(), "主语义评审工作包"
    )
    manifest = _load_json(directory / "manifest.json", "主语义评审 manifest")
    records = manifest.get("records", {})
    if (
        manifest.get("pipeline") != WORK_PACKAGE_PIPELINE
        or manifest.get("complete") is not True
        or records.get("primary_protocol_valid") != EXPECTED_PRIMARY_VALID
        or records.get("primary_protocol_invalid") != EXPECTED_PRIMARY_INVALID
        or records.get("primary_answerable_valid")
        != EXPECTED_PRIMARY_ANSWERABLE_VALID
        or records.get("primary_packaged_incomplete_valid")
        != EXPECTED_PRIMARY_INCOMPLETE_VALID
    ):
        raise RagSftV2SemanticReviewFinalizationError("主语义评审工作包状态无效")
    return hash_path, manifest


def _load_primary_records(
    directory: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[int, list[str]]]:
    queues = {}
    audits = {}
    batches = {}
    for batch, expected in enumerate(PRIMARY_BATCH_SIZES, 1):
        queue_rows = _load_jsonl(
            directory / "blind" / "review-queue" / f"batch-{batch:02d}.jsonl",
            f"主审队列第 {batch} 批",
        )
        audit_rows = _load_jsonl(
            directory / "audit-reference" / f"batch-{batch:02d}.jsonl",
            f"主审隐藏引用第 {batch} 批",
        )
        if len(queue_rows) != expected or len(audit_rows) != expected:
            raise RagSftV2SemanticReviewFinalizationError("主审 batch 数量无效")
        batch_ids = []
        for queue, audit in zip(queue_rows, audit_rows):
            review_id = queue.get("review_id")
            if (
                tuple(queue) != REVIEWER_VISIBLE_FIELDS
                or tuple(audit) != AUDIT_REFERENCE_FIELDS
                or not isinstance(review_id, str)
                or audit.get("review_id") != review_id
                or review_id in queues
            ):
                raise RagSftV2SemanticReviewFinalizationError("主审队列身份或字段无效")
            queues[review_id] = queue
            audits[review_id] = audit
            batch_ids.append(review_id)
        batches[batch] = batch_ids
    if len(queues) != EXPECTED_PRIMARY_VALID:
        raise RagSftV2SemanticReviewFinalizationError("主审队列未覆盖 158 条记录")
    return queues, audits, batches


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _validate_review_result(
    row: Mapping[str, Any], queue: Mapping[str, Any]
) -> dict[str, Any]:
    review_id = queue["review_id"]
    if tuple(row) != REVIEW_RESULT_FIELDS or row.get("review_id") != review_id:
        raise RagSftV2SemanticReviewFinalizationError(
            f"审核结果字段或 review_id 无效: {review_id}"
        )
    decision = row.get("review_decision")
    answerability = row.get("answerability")
    attribution = row.get("error_attribution")
    if decision not in DECISIONS or answerability not in ANSWERABILITY:
        raise RagSftV2SemanticReviewFinalizationError(f"审核枚举无效: {review_id}")
    if attribution not in ERROR_ATTRIBUTIONS:
        raise RagSftV2SemanticReviewFinalizationError(f"错误归因无效: {review_id}")
    citations = queue.get("assistant", {}).get("citations")
    if not isinstance(citations, list) or not citations:
        raise RagSftV2SemanticReviewFinalizationError(f"队列 citation 无效: {review_id}")

    claims = row.get("atomic_claims")
    if not isinstance(claims, list) or not claims:
        raise RagSftV2SemanticReviewFinalizationError(f"原子主张不能为空: {review_id}")
    for claim in claims:
        if (
            not isinstance(claim, dict)
            or tuple(claim) != ATOMIC_CLAIM_FIELDS
            or not isinstance(claim.get("claim"), str)
            or not claim["claim"].strip()
            or claim.get("support") not in CLAIM_SUPPORT
            or not isinstance(claim.get("supporting_evidence_ids"), list)
            or len(claim["supporting_evidence_ids"])
            != len(set(claim["supporting_evidence_ids"]))
            or not set(claim["supporting_evidence_ids"]).issubset(set(citations))
        ):
            raise RagSftV2SemanticReviewFinalizationError(f"原子主张结构无效: {review_id}")
        support = claim["support"]
        supporting = claim["supporting_evidence_ids"]
        if (support == "unsupported" and supporting) or (
            support != "unsupported" and not supporting
        ):
            raise RagSftV2SemanticReviewFinalizationError(
                f"原子主张支持状态与证据不一致: {review_id}"
            )

    matters = row.get("necessary_matters")
    if not isinstance(matters, list) or (
        answerability == "sufficient" and not matters
    ):
        raise RagSftV2SemanticReviewFinalizationError(f"必要事项结构无效: {review_id}")
    for matter in matters:
        if (
            not isinstance(matter, dict)
            or tuple(matter) != NECESSARY_MATTER_FIELDS
            or not isinstance(matter.get("matter"), str)
            or not matter["matter"].strip()
            or not _is_bool(matter.get("covered"))
        ):
            raise RagSftV2SemanticReviewFinalizationError(f"必要事项无效: {review_id}")

    findings = row.get("citation_findings")
    if not isinstance(findings, list) or [
        item.get("evidence_id") for item in findings if isinstance(item, dict)
    ] != citations:
        raise RagSftV2SemanticReviewFinalizationError(f"引用审核未精确覆盖 citations: {review_id}")
    for finding in findings:
        if (
            not isinstance(finding, dict)
            or tuple(finding) != CITATION_FINDING_FIELDS
            or not _is_bool(finding.get("supports_any_claim"))
            or not _is_bool(finding.get("necessary_for_summary"))
            or finding["necessary_for_summary"]
            and not finding["supports_any_claim"]
        ):
            raise RagSftV2SemanticReviewFinalizationError(f"引用审核结构无效: {review_id}")

    for name in (
        "query_responsive",
        "legal_boundaries_preserved",
        "unsupported_fact_absent",
    ):
        if not _is_bool(row.get(name)):
            raise RagSftV2SemanticReviewFinalizationError(f"审核布尔字段无效: {review_id}")
    tags = row.get("error_tags")
    reason = row.get("reason")
    if (
        not isinstance(tags, list)
        or len(tags) != len(set(tags))
        or not set(tags).issubset(ERROR_TAGS)
        or not isinstance(reason, str)
        or not reason.strip()
    ):
        raise RagSftV2SemanticReviewFinalizationError(f"审核标签或理由无效: {review_id}")

    pass_conditions = (
        answerability == "sufficient"
        and all(claim["support"] == "fully_supported" for claim in claims)
        and all(matter["covered"] for matter in matters)
        and all(
            finding["supports_any_claim"] and finding["necessary_for_summary"]
            for finding in findings
        )
        and row["query_responsive"]
        and row["legal_boundaries_preserved"]
        and row["unsupported_fact_absent"]
        and attribution == "none"
        and not tags
    )
    if (decision == "pass") != pass_conditions:
        raise RagSftV2SemanticReviewFinalizationError(f"pass 决策与结构化标签不一致: {review_id}")
    if decision == "fail" and (attribution == "none" or not tags):
        raise RagSftV2SemanticReviewFinalizationError(f"fail 决策缺少错误归因: {review_id}")
    if decision == "escalate" and not (
        answerability == "uncertain"
        or attribution == "uncertain"
        or "manual_review_required" in tags
    ):
        raise RagSftV2SemanticReviewFinalizationError(f"escalate 决策缺少升级依据: {review_id}")
    return dict(row)


def _verify_review_run(
    directory: Path,
    *,
    review_stage: str,
    package_hash: Path,
    fragments: Mapping[str, tuple[Path, list[str]]],
    queues: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, object]]:
    run_path = directory / "review-run.json"
    sidecar = run_path.with_suffix(".json.sha256")
    try:
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftV2SemanticReviewFinalizationError("无法读取审核 run SHA-256") from error
    if lines != [f"{_sha256_file(run_path)}  {run_path.name}"]:
        raise RagSftV2SemanticReviewFinalizationError("审核 run SHA-256 无效")
    run = _load_json(run_path, "审核 run")
    identity = run.get("reviewer_identity")
    if (
        run.get("pipeline") != RUN_PIPELINE
        or run.get("review_stage") != review_stage
        or run.get("complete") is not True
        or not isinstance(identity, dict)
        or tuple(identity) != REVIEWER_IDENTITY_FIELDS
        or not all(
            isinstance(identity.get(name), str) and identity[name].strip()
            for name in REVIEWER_IDENTITY_FIELDS
        )
    ):
        raise RagSftV2SemanticReviewFinalizationError("审核 run 状态或审核者身份无效")
    package_identity = run.get("inputs", {}).get("review_package_hash_manifest")
    if (
        not isinstance(package_identity, dict)
        or package_identity.get("sha256") != _sha256_file(package_hash)
    ):
        raise RagSftV2SemanticReviewFinalizationError("审核 run 未绑定当前工作包")
    declared = run.get("outputs", {}).get("review_result_fragments")
    if not isinstance(declared, dict) or set(declared) != set(fragments):
        raise RagSftV2SemanticReviewFinalizationError("审核 run 结果分片声明无效")

    results = {}
    identities = {}
    total = 0
    for name, (path, expected_ids) in fragments.items():
        rows = _load_jsonl(path, f"审核结果 {name}")
        if len(rows) != len(expected_ids):
            raise RagSftV2SemanticReviewFinalizationError(f"审核结果数量不闭合: {name}")
        declaration = declared[name]
        if (
            not isinstance(declaration, dict)
            or declaration.get("sha256") != _sha256_file(path)
            or declaration.get("bytes") != path.stat().st_size
            or declaration.get("records") != len(rows)
        ):
            raise RagSftV2SemanticReviewFinalizationError(f"审核分片身份无效: {name}")
        if [row.get("review_id") for row in rows] != expected_ids:
            raise RagSftV2SemanticReviewFinalizationError(f"审核结果顺序或身份无效: {name}")
        for row in rows:
            review_id = row["review_id"]
            if review_id in results:
                raise RagSftV2SemanticReviewFinalizationError("审核结果 review_id 重复")
            results[review_id] = _validate_review_result(row, queues[review_id])
        identities[name] = _identity(path, records=len(rows))
        total += len(rows)
    if run.get("records", {}).get("review_results") != total:
        raise RagSftV2SemanticReviewFinalizationError("审核 run 总记录数无效")
    return run, results, {"run": _identity(run_path), "fragments": identities}


def _primary_run(
    work_package_dir: Path,
    primary_results_dir: Path,
    work_hash: Path,
    queues: Mapping[str, Mapping[str, Any]],
    batches: Mapping[int, list[str]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, object]]:
    fragments = {
        f"batch_{batch:02d}": (
            primary_results_dir / "review-results" / f"batch-{batch:02d}.jsonl",
            ids,
        )
        for batch, ids in batches.items()
    }
    return _verify_review_run(
        primary_results_dir,
        review_stage="primary",
        package_hash=work_hash,
        fragments=fragments,
        queues=queues,
    )


def _secondary_ids(primary: Mapping[str, Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    mandatory = sorted(
        review_id
        for review_id, row in primary.items()
        if row["review_decision"] != "pass"
    )
    passed = [
        review_id
        for review_id, row in primary.items()
        if row["review_decision"] == "pass"
    ]
    passed.sort(
        key=lambda review_id: hashlib.sha256(
            f"42:{review_id}".encode("utf-8")
        ).hexdigest()
    )
    sampled = passed[: math.ceil(len(passed) * PASS_AUDIT_RATIO)]
    return mandatory, sampled


def _write_package(output_dir: Path, payloads: Mapping[str, str], label: str) -> Path:
    if output_dir.exists():
        raise RagSftV2SemanticReviewFinalizationError(f"{label}输出目录必须不存在")
    try:
        output_dir.mkdir(parents=True)
        for name, payload in payloads.items():
            path = output_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8", newline="\n")
        hash_path = output_dir / "manifest.sha256"
        hash_path.write_text(
            "".join(
                f"{hashlib.sha256(payloads[name].encode('utf-8')).hexdigest()}  {name}\n"
                for name in sorted(payloads)
            ),
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, UnicodeError) as error:
        raise RagSftV2SemanticReviewFinalizationError(f"无法发布{label}") from error
    return hash_path


def prepare_secondary_review(
    *,
    work_package_dir: Path = DEFAULT_WORK_PACKAGE_DIR,
    primary_results_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """根据主审结果发布全部失败/升级项和 20% 通过项的二审队列。"""

    work_package_dir = Path(work_package_dir).resolve()
    primary_results_dir = Path(primary_results_dir).resolve()
    output_dir = Path(output_dir).resolve()
    work_hash, work_manifest = _verify_primary_package(work_package_dir)
    queues, audits, batches = _load_primary_records(work_package_dir)
    primary_run, primary, primary_identity = _primary_run(
        work_package_dir,
        primary_results_dir,
        work_hash,
        queues,
        batches,
    )
    mandatory, sampled = _secondary_ids(primary)
    selected = mandatory + sampled
    random.Random(SECONDARY_SEED).shuffle(selected)
    review_rows = [queues[review_id] for review_id in selected]
    audit_rows = [audits[review_id] for review_id in selected]
    rubric = (work_package_dir / "rubric.md").read_text(encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "pipeline": SECONDARY_PIPELINE,
        "release_status": "secondary_review_pending",
        "inputs": {
            "finalization_source": _identity(Path(__file__).resolve()),
            "primary_work_package_hash_manifest": _identity(work_hash),
            "primary_review_run": {
                **primary_identity["run"],
                "reviewer_identity": primary_run["reviewer_identity"],
            },
            "primary_review_result_fragments": primary_identity["fragments"],
        },
        "review_protocol": {
            "seed": SECONDARY_SEED,
            "mandatory": "all_primary_fail_or_escalate",
            "pass_audit_ratio": PASS_AUDIT_RATIO,
            "reviewer_visible_fields": list(REVIEWER_VISIBLE_FIELDS),
            "reviewer_output_fields": list(REVIEW_RESULT_FIELDS),
        },
        "records": {
            "primary_reviewed": EXPECTED_PRIMARY_VALID,
            "mandatory": len(mandatory),
            "sampled_primary_pass": len(sampled),
            "secondary_queue": len(selected),
        },
        "candidate": work_manifest["candidate"],
        "complete": True,
    }
    payloads = {
        "README.md": (
            "# RAG-SFT v2 二审工作包\n\n"
            "二审者只读取 `rubric.md` 与 `review-queue.jsonl`，不得读取主审结果、"
            "隐藏引用或自动指标。输出字段与主审完全一致。\n"
        ),
        "rubric.md": rubric,
        "review-queue.jsonl": _jsonl_payload(review_rows),
        "audit-reference.jsonl": _jsonl_payload(audit_rows),
        "manifest.json": json.dumps(
            manifest, ensure_ascii=False, indent=2, allow_nan=False
        )
        + "\n",
    }
    _write_package(output_dir, payloads, "二审工作包")
    return manifest


def _verify_secondary_package(
    directory: Path,
    *,
    work_hash: Path,
    primary_run_path: Path,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    expected = {
        "README.md",
        "rubric.md",
        "review-queue.jsonl",
        "audit-reference.jsonl",
        "manifest.json",
    }
    hash_path = _verify_hash_manifest(directory, expected, "二审工作包")
    manifest = _load_json(directory / "manifest.json", "二审 manifest")
    if (
        manifest.get("pipeline") != SECONDARY_PIPELINE
        or manifest.get("complete") is not True
        or manifest.get("inputs", {})
        .get("primary_work_package_hash_manifest", {})
        .get("sha256")
        != _sha256_file(work_hash)
        or manifest.get("inputs", {}).get("primary_review_run", {}).get("sha256")
        != _sha256_file(primary_run_path)
    ):
        raise RagSftV2SemanticReviewFinalizationError("二审工作包状态无效")
    rows = _load_jsonl(directory / "review-queue.jsonl", "二审队列")
    if len(rows) != manifest.get("records", {}).get("secondary_queue"):
        raise RagSftV2SemanticReviewFinalizationError("二审队列计数无效")
    return hash_path, manifest, rows


def _single_run(
    results_dir: Path,
    *,
    review_stage: str,
    package_hash: Path,
    queue_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[str, object]]:
    queues = {row["review_id"]: row for row in queue_rows}
    ids = [row["review_id"] for row in queue_rows]
    return _verify_review_run(
        results_dir,
        review_stage=review_stage,
        package_hash=package_hash,
        fragments={"results": (results_dir / "review-results.jsonl", ids)},
        queues=queues,
    )


def seal_review_run(
    *,
    review_package_dir: Path,
    results_dir: Path,
    review_stage: str,
    reviewer_id: str,
    reviewer_role: str,
    review_method: str,
    independence_declaration: str,
) -> dict[str, object]:
    """校验现有结果分片并生成受签的审核 run 身份。"""

    review_package_dir = Path(review_package_dir).resolve()
    results_dir = Path(results_dir).resolve()
    run_path = results_dir / "review-run.json"
    sidecar = run_path.with_suffix(".json.sha256")
    if run_path.exists() or sidecar.exists():
        raise RagSftV2SemanticReviewFinalizationError("审核 run 已存在，禁止覆盖")
    identity_values = (
        reviewer_id,
        reviewer_role,
        review_method,
        independence_declaration,
    )
    if any(not isinstance(value, str) or not value.strip() for value in identity_values):
        raise RagSftV2SemanticReviewFinalizationError("审核者身份字段不能为空")

    manifest = _load_json(review_package_dir / "manifest.json", "审核工作包 manifest")
    pipeline = manifest.get("pipeline")
    fragments: dict[str, tuple[Path, list[str]]]
    queues: dict[str, dict[str, Any]]
    if review_stage == "primary" and pipeline == WORK_PACKAGE_PIPELINE:
        package_hash, _ = _verify_primary_package(review_package_dir)
        queues, _, batches = _load_primary_records(review_package_dir)
        fragments = {
            f"batch_{batch:02d}": (
                results_dir / "review-results" / f"batch-{batch:02d}.jsonl",
                ids,
            )
            for batch, ids in batches.items()
        }
    elif review_stage == "secondary" and pipeline == SECONDARY_PIPELINE:
        package_hash = _verify_hash_manifest(
            review_package_dir,
            {
                "README.md",
                "rubric.md",
                "review-queue.jsonl",
                "audit-reference.jsonl",
                "manifest.json",
            },
            "二审工作包",
        )
        queue_rows = _load_jsonl(review_package_dir / "review-queue.jsonl", "二审队列")
        queues = {row["review_id"]: row for row in queue_rows}
        fragments = {
            "results": (
                results_dir / "review-results.jsonl",
                [row["review_id"] for row in queue_rows],
            )
        }
    elif review_stage == "adjudication" and pipeline == ADJUDICATION_PIPELINE:
        package_hash = _verify_hash_manifest(
            review_package_dir,
            {"README.md", "rubric.md", "review-queue.jsonl", "manifest.json"},
            "裁决工作包",
        )
        queue_rows = _load_jsonl(review_package_dir / "review-queue.jsonl", "裁决队列")
        queues = {row["review_id"]: row["case"] for row in queue_rows}
        fragments = {
            "results": (
                results_dir / "review-results.jsonl",
                [row["review_id"] for row in queue_rows],
            )
        }
    else:
        raise RagSftV2SemanticReviewFinalizationError("审核阶段与工作包 pipeline 不匹配")

    outputs = {}
    total = 0
    for name, (path, expected_ids) in fragments.items():
        rows = _load_jsonl(path, f"待封存审核结果 {name}")
        if [row.get("review_id") for row in rows] != expected_ids:
            raise RagSftV2SemanticReviewFinalizationError("待封存审核结果身份或顺序无效")
        for row in rows:
            _validate_review_result(row, queues[row["review_id"]])
        outputs[name] = _identity(path, records=len(rows))
        total += len(rows)
    run = {
        "schema_version": "1.0",
        "pipeline": RUN_PIPELINE,
        "review_stage": review_stage,
        "reviewer_identity": {
            "reviewer_id": reviewer_id,
            "reviewer_role": reviewer_role,
            "review_method": review_method,
            "independence_declaration": independence_declaration,
        },
        "inputs": {
            "review_package_hash_manifest": _identity(package_hash),
            "sealing_source": _identity(Path(__file__).resolve()),
        },
        "outputs": {"review_result_fragments": outputs},
        "records": {"review_results": total},
        "complete": True,
    }
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        run_path.write_text(
            json.dumps(run, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        sidecar.write_text(
            f"{_sha256_file(run_path)}  {run_path.name}\n",
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, UnicodeError) as error:
        raise RagSftV2SemanticReviewFinalizationError("无法封存审核 run") from error
    return run


def _material_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    support_counts = Counter(item["support"] for item in row["atomic_claims"])
    matter_counts = Counter(item["covered"] for item in row["necessary_matters"])
    citations = tuple(
        (item["supports_any_claim"], item["necessary_for_summary"])
        for item in row["citation_findings"]
    )
    return (
        row["review_decision"],
        row["answerability"],
        tuple(sorted(support_counts.items())),
        tuple(sorted(matter_counts.items())),
        citations,
        row["query_responsive"],
        row["legal_boundaries_preserved"],
        row["unsupported_fact_absent"],
        row["error_attribution"],
        tuple(sorted(row["error_tags"])),
    )


def prepare_adjudication_review(
    *,
    work_package_dir: Path = DEFAULT_WORK_PACKAGE_DIR,
    primary_results_dir: Path,
    secondary_package_dir: Path,
    secondary_results_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """只为主审与二审存在实质分歧的记录发布匿名裁决队列。"""

    work_package_dir = Path(work_package_dir).resolve()
    primary_results_dir = Path(primary_results_dir).resolve()
    secondary_package_dir = Path(secondary_package_dir).resolve()
    secondary_results_dir = Path(secondary_results_dir).resolve()
    output_dir = Path(output_dir).resolve()
    work_hash, work_manifest = _verify_primary_package(work_package_dir)
    queues, _, batches = _load_primary_records(work_package_dir)
    primary_run, primary, primary_identity = _primary_run(
        work_package_dir,
        primary_results_dir,
        work_hash,
        queues,
        batches,
    )
    secondary_hash, _, secondary_queue = _verify_secondary_package(
        secondary_package_dir,
        work_hash=work_hash,
        primary_run_path=Path(primary_identity["run"]["path"]),
    )
    secondary_run, secondary, secondary_identity = _single_run(
        secondary_results_dir,
        review_stage="secondary",
        package_hash=secondary_hash,
        queue_rows=secondary_queue,
    )
    if (
        secondary_run["reviewer_identity"]["reviewer_id"]
        == primary_run["reviewer_identity"]["reviewer_id"]
    ):
        raise RagSftV2SemanticReviewFinalizationError("主审与二审必须由不同审核者完成")
    disagreements = sorted(
        review_id
        for review_id, row in secondary.items()
        if _material_signature(primary[review_id]) != _material_signature(row)
    )
    queue_rows = [
        {
            "review_id": review_id,
            "case": queues[review_id],
            "review_a": primary[review_id],
            "review_b": secondary[review_id],
        }
        for review_id in disagreements
    ]
    manifest = {
        "schema_version": "1.0",
        "pipeline": ADJUDICATION_PIPELINE,
        "release_status": "adjudication_pending",
        "inputs": {
            "finalization_source": _identity(Path(__file__).resolve()),
            "primary_work_package_hash_manifest": _identity(work_hash),
            "primary_review_run": primary_identity["run"],
            "secondary_work_package_hash_manifest": _identity(secondary_hash),
            "secondary_review_run": secondary_identity["run"],
        },
        "candidate": work_manifest["candidate"],
        "records": {
            "secondary_reviewed": len(secondary),
            "material_disagreements": len(disagreements),
        },
        "review_protocol": {
            "queue_fields": ["review_id", "case", "review_a", "review_b"],
            "reviewer_output_fields": list(REVIEW_RESULT_FIELDS),
            "automatic_metrics_visible": False,
        },
        "complete": True,
    }
    payloads = {
        "README.md": (
            "# RAG-SFT v2 裁决工作包\n\n"
            "裁决者根据匿名题目、EvidencePackage 和两份结构化审核作出最终审核记录。"
            "不得读取自动指标、required GT 或 HN 标签。\n"
        ),
        "rubric.md": (work_package_dir / "rubric.md").read_text(encoding="utf-8"),
        "review-queue.jsonl": _jsonl_payload(queue_rows),
        "manifest.json": json.dumps(
            manifest, ensure_ascii=False, indent=2, allow_nan=False
        )
        + "\n",
    }
    _write_package(output_dir, payloads, "裁决工作包")
    return manifest


def _verify_adjudication_package(
    directory: Path,
    *,
    work_hash: Path,
    expected_ids: list[str],
) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    expected = {"README.md", "rubric.md", "review-queue.jsonl", "manifest.json"}
    hash_path = _verify_hash_manifest(directory, expected, "裁决工作包")
    manifest = _load_json(directory / "manifest.json", "裁决 manifest")
    rows = _load_jsonl(directory / "review-queue.jsonl", "裁决队列")
    if (
        manifest.get("pipeline") != ADJUDICATION_PIPELINE
        or manifest.get("complete") is not True
        or manifest.get("inputs", {})
        .get("primary_work_package_hash_manifest", {})
        .get("sha256")
        != _sha256_file(work_hash)
        or [row.get("review_id") for row in rows] != expected_ids
    ):
        raise RagSftV2SemanticReviewFinalizationError("裁决工作包状态或范围无效")
    return hash_path, manifest, rows


def _metric_summary(
    review_ids: list[str], final: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    rows = [final[review_id] for review_id in review_ids]
    decisions = Counter(row["review_decision"] for row in rows)
    attributions = Counter(row["error_attribution"] for row in rows)
    claims = [claim for row in rows for claim in row["atomic_claims"]]
    matters = [row for row in rows if row["answerability"] == "sufficient"]
    findings = [finding for row in rows for finding in row["citation_findings"]]

    def rate(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    complete = sum(
        bool(row["necessary_matters"])
        and all(item["covered"] for item in row["necessary_matters"])
        for row in matters
    )
    return {
        "records": len(rows),
        "decisions": dict(sorted(decisions.items())),
        "error_attribution": dict(sorted(attributions.items())),
        "atomic_claim_support": {
            "fully_supported": sum(
                claim["support"] == "fully_supported" for claim in claims
            ),
            "claims": len(claims),
            "rate": rate(
                sum(claim["support"] == "fully_supported" for claim in claims),
                len(claims),
            ),
        },
        "answer_completeness": {
            "complete": complete,
            "answerability_sufficient": len(matters),
            "rate": rate(complete, len(matters)),
        },
        "citation_semantic_precision": {
            "supporting": sum(item["supports_any_claim"] for item in findings),
            "citations": len(findings),
            "rate": rate(
                sum(item["supports_any_claim"] for item in findings), len(findings)
            ),
        },
        "citation_minimality": {
            "necessary": sum(item["necessary_for_summary"] for item in findings),
            "citations": len(findings),
            "rate": rate(
                sum(item["necessary_for_summary"] for item in findings), len(findings)
            ),
        },
        "query_responsive_rate": rate(
            sum(row["query_responsive"] for row in rows), len(rows)
        ),
        "legal_boundaries_preserved_rate": rate(
            sum(row["legal_boundaries_preserved"] for row in rows), len(rows)
        ),
        "unsupported_fact_absent_rate": rate(
            sum(row["unsupported_fact_absent"] for row in rows), len(rows)
        ),
        "semantic_pass_rate": rate(decisions["pass"], len(rows)),
    }


def finalize_semantic_review(
    *,
    work_package_dir: Path = DEFAULT_WORK_PACKAGE_DIR,
    primary_results_dir: Path,
    secondary_package_dir: Path,
    secondary_results_dir: Path,
    adjudication_package_dir: Path,
    adjudication_results_dir: Path,
    output_dir: Path,
) -> dict[str, object]:
    """发布 158 条最终语义账本和分层汇总，不自动接受模型。"""

    work_package_dir = Path(work_package_dir).resolve()
    primary_results_dir = Path(primary_results_dir).resolve()
    secondary_package_dir = Path(secondary_package_dir).resolve()
    secondary_results_dir = Path(secondary_results_dir).resolve()
    adjudication_package_dir = Path(adjudication_package_dir).resolve()
    adjudication_results_dir = Path(adjudication_results_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise RagSftV2SemanticReviewFinalizationError("最终语义评审输出目录必须不存在")
    work_hash, work_manifest = _verify_primary_package(work_package_dir)
    queues, audits, batches = _load_primary_records(work_package_dir)
    primary_run, primary, primary_identity = _primary_run(
        work_package_dir,
        primary_results_dir,
        work_hash,
        queues,
        batches,
    )
    secondary_hash, secondary_manifest, secondary_queue = _verify_secondary_package(
        secondary_package_dir,
        work_hash=work_hash,
        primary_run_path=Path(primary_identity["run"]["path"]),
    )
    secondary_run, secondary, secondary_identity = _single_run(
        secondary_results_dir,
        review_stage="secondary",
        package_hash=secondary_hash,
        queue_rows=secondary_queue,
    )
    if (
        primary_run["reviewer_identity"]["reviewer_id"]
        == secondary_run["reviewer_identity"]["reviewer_id"]
    ):
        raise RagSftV2SemanticReviewFinalizationError("主审与二审审核者不得相同")
    mandatory, sampled = _secondary_ids(primary)
    if set(secondary) != set(mandatory + sampled):
        raise RagSftV2SemanticReviewFinalizationError("二审范围未按固定规则闭合")
    disagreements = sorted(
        review_id
        for review_id, row in secondary.items()
        if _material_signature(primary[review_id]) != _material_signature(row)
    )
    adjudication_hash, adjudication_manifest, adjudication_queue = (
        _verify_adjudication_package(
            adjudication_package_dir,
            work_hash=work_hash,
            expected_ids=disagreements,
        )
    )
    adjudication_queues = {row["review_id"]: row["case"] for row in adjudication_queue}
    adjudication_run, adjudication, adjudication_identity = _verify_review_run(
        adjudication_results_dir,
        review_stage="adjudication",
        package_hash=adjudication_hash,
        fragments={
            "results": (
                adjudication_results_dir / "review-results.jsonl",
                disagreements,
            )
        },
        queues=adjudication_queues,
    )
    reviewer_ids = {
        primary_run["reviewer_identity"]["reviewer_id"],
        secondary_run["reviewer_identity"]["reviewer_id"],
        adjudication_run["reviewer_identity"]["reviewer_id"],
    }
    if len(reviewer_ids) != 3:
        raise RagSftV2SemanticReviewFinalizationError("主审、二审和裁决者必须相互独立")
    final = dict(primary)
    final.update(adjudication)
    if len(final) != EXPECTED_PRIMARY_VALID:
        raise RagSftV2SemanticReviewFinalizationError("最终语义账本未覆盖 158 条记录")

    order = sorted(final)
    ledger = []
    for review_id in order:
        ledger.append(
            {
                "review_id": review_id,
                "query_id": audits[review_id]["query_id"],
                "query_type": audits[review_id]["query_type"],
                "packaged_complete": audits[review_id]["retrieval_attribution"].get(
                    "packaged_complete"
                ),
                "has_hard_negative": bool(audits[review_id]["hard_negative_chunk_ids"]),
                "primary_review": primary[review_id],
                "secondary_review": secondary.get(review_id),
                "adjudication_review": adjudication.get(review_id),
                "final_review": final[review_id],
            }
        )
    gate_ids = [
        review_id
        for review_id, audit in audits.items()
        if audit["query_type"] == "exact_lookup"
        or audit["retrieval_attribution"].get("packaged_complete") is True
    ]
    incomplete_ids = [review_id for review_id in audits if review_id not in set(gate_ids)]
    clean_ids = [
        review_id
        for review_id, audit in audits.items()
        if not audit["hard_negative_chunk_ids"]
    ]
    complete_hn_ids = [
        review_id
        for review_id in gate_ids
        if audits[review_id]["hard_negative_chunk_ids"]
    ]
    exact_ids = [
        review_id
        for review_id, audit in audits.items()
        if audit["query_type"] == "exact_lookup"
    ]
    if len(gate_ids) != EXPECTED_PRIMARY_ANSWERABLE_VALID or len(incomplete_ids) != EXPECTED_PRIMARY_INCOMPLETE_VALID:
        raise RagSftV2SemanticReviewFinalizationError("最终语义评审分层计数无效")

    ledger_payload = _jsonl_payload(ledger)
    summary = {
        "schema_version": "1.0",
        "pipeline": FINAL_PIPELINE,
        "release_status": "user_selection_pending",
        "inputs": {
            "finalization_source": _identity(Path(__file__).resolve()),
            "primary_work_package_hash_manifest": _identity(work_hash),
            "primary_review": primary_identity,
            "secondary_work_package_hash_manifest": _identity(secondary_hash),
            "secondary_manifest": secondary_manifest,
            "secondary_review": secondary_identity,
            "adjudication_work_package_hash_manifest": _identity(adjudication_hash),
            "adjudication_manifest": adjudication_manifest,
            "adjudication_review": adjudication_identity,
        },
        "candidate": work_manifest["candidate"],
        "records": {
            "primary_protocol_valid": EXPECTED_PRIMARY_VALID,
            "primary_protocol_invalid": EXPECTED_PRIMARY_INVALID,
            "secondary_reviewed": len(secondary),
            "secondary_mandatory": len(mandatory),
            "secondary_sampled_pass": len(sampled),
            "material_disagreements": len(disagreements),
            "adjudicated": len(adjudication),
        },
        "metrics": {
            "selection_gate_eligible": _metric_summary(gate_ids, final),
            "clean": _metric_summary(clean_ids, final),
            "complete_hard_negative": _metric_summary(complete_hn_ids, final),
            "packaged_incomplete": _metric_summary(incomplete_ids, final),
            "exact_lookup": _metric_summary(exact_ids, final),
        },
        "outputs": {
            "semantic_review_ledger": {
                "path": str(output_dir / "semantic-review-ledger.jsonl"),
                "bytes": len(ledger_payload.encode("utf-8")),
                "sha256": hashlib.sha256(ledger_payload.encode("utf-8")).hexdigest(),
                "records": EXPECTED_PRIMARY_VALID,
            }
        },
        "selection_applied": False,
        "private_holdout_used": False,
        "automatic_model_acceptance": False,
        "complete": True,
    }
    summary_payload = json.dumps(
        summary, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    try:
        output_dir.mkdir(parents=True)
        ledger_path = output_dir / "semantic-review-ledger.jsonl"
        summary_path = output_dir / "semantic-review-summary.json"
        ledger_path.write_text(ledger_payload, encoding="utf-8", newline="\n")
        summary_path.write_text(summary_payload, encoding="utf-8", newline="\n")
        ledger_path.with_suffix(".jsonl.sha256").write_text(
            f"{_sha256_file(ledger_path)}  {ledger_path.name}\n",
            encoding="utf-8",
            newline="\n",
        )
        summary_path.with_suffix(".json.sha256").write_text(
            f"{_sha256_file(summary_path)}  {summary_path.name}\n",
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, UnicodeError) as error:
        raise RagSftV2SemanticReviewFinalizationError("无法发布最终语义评审账本") from error
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    seal = subparsers.add_parser("seal-run")
    seal.add_argument("--review-package-dir", type=Path, required=True)
    seal.add_argument("--results-dir", type=Path, required=True)
    seal.add_argument(
        "--review-stage", choices=("primary", "secondary", "adjudication"), required=True
    )
    seal.add_argument("--reviewer-id", required=True)
    seal.add_argument("--reviewer-role", required=True)
    seal.add_argument("--review-method", required=True)
    seal.add_argument("--independence-declaration", required=True)

    secondary = subparsers.add_parser("prepare-secondary")
    secondary.add_argument("--work-package-dir", type=Path, default=DEFAULT_WORK_PACKAGE_DIR)
    secondary.add_argument("--primary-results-dir", type=Path, required=True)
    secondary.add_argument("--output-dir", type=Path, required=True)

    adjudication = subparsers.add_parser("prepare-adjudication")
    adjudication.add_argument("--work-package-dir", type=Path, default=DEFAULT_WORK_PACKAGE_DIR)
    adjudication.add_argument("--primary-results-dir", type=Path, required=True)
    adjudication.add_argument("--secondary-package-dir", type=Path, required=True)
    adjudication.add_argument("--secondary-results-dir", type=Path, required=True)
    adjudication.add_argument("--output-dir", type=Path, required=True)

    final = subparsers.add_parser("finalize")
    final.add_argument("--work-package-dir", type=Path, default=DEFAULT_WORK_PACKAGE_DIR)
    final.add_argument("--primary-results-dir", type=Path, required=True)
    final.add_argument("--secondary-package-dir", type=Path, required=True)
    final.add_argument("--secondary-results-dir", type=Path, required=True)
    final.add_argument("--adjudication-package-dir", type=Path, required=True)
    final.add_argument("--adjudication-results-dir", type=Path, required=True)
    final.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    try:
        if args.command == "seal-run":
            result = seal_review_run(
                review_package_dir=args.review_package_dir,
                results_dir=args.results_dir,
                review_stage=args.review_stage,
                reviewer_id=args.reviewer_id,
                reviewer_role=args.reviewer_role,
                review_method=args.review_method,
                independence_declaration=args.independence_declaration,
            )
            print(
                "RAG_SFT_V2_SEMANTIC_REVIEW_RUN_OK "
                f"records={result['records']['review_results']}"
            )
        elif args.command == "prepare-secondary":
            result = prepare_secondary_review(
                work_package_dir=args.work_package_dir,
                primary_results_dir=args.primary_results_dir,
                output_dir=args.output_dir,
            )
            print(
                "RAG_SFT_V2_SECONDARY_REVIEW_PACKAGE_OK "
                f"records={result['records']['secondary_queue']}"
            )
        elif args.command == "prepare-adjudication":
            result = prepare_adjudication_review(
                work_package_dir=args.work_package_dir,
                primary_results_dir=args.primary_results_dir,
                secondary_package_dir=args.secondary_package_dir,
                secondary_results_dir=args.secondary_results_dir,
                output_dir=args.output_dir,
            )
            print(
                "RAG_SFT_V2_ADJUDICATION_PACKAGE_OK "
                f"records={result['records']['material_disagreements']}"
            )
        else:
            result = finalize_semantic_review(
                work_package_dir=args.work_package_dir,
                primary_results_dir=args.primary_results_dir,
                secondary_package_dir=args.secondary_package_dir,
                secondary_results_dir=args.secondary_results_dir,
                adjudication_package_dir=args.adjudication_package_dir,
                adjudication_results_dir=args.adjudication_results_dir,
                output_dir=args.output_dir,
            )
            print(
                "RAG_SFT_V2_SEMANTIC_REVIEW_AUDIT_OK "
                f"records={result['records']['primary_protocol_valid']}"
            )
    except RagSftV2SemanticReviewFinalizationError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
