"""从完整冻结 Retrieval 结果发布正式全量 Query-SFT 与公共 Query 池。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from rag.query import QUERY_ENHANCEMENT_SYSTEM_PROMPT
from rag.query.enhancement import QueryEnhancementProtocolError, parse_and_validate_query_enhancement

try:
    from . import audit_disc_law_sft as tokenizer_loader
    from .evaluate_query_sft_full_retrieval import (
        DEFAULT_ARTICLE_INDEX,
        DEFAULT_ARTIFACT_DIR,
        DEFAULT_INPUT_WORK_PACKAGE_DIR,
        DEFAULT_OUTPUT_DIR as DEFAULT_RETRIEVAL_DIR,
        DEFAULT_SEMANTIC_REVIEW_DIR,
        DEFAULT_TEACHER_WORK_PACKAGE_DIR,
        DEFAULT_TEXT_AUDIT,
        HASH_FILENAME as RETRIEVAL_HASH_FILENAME,
        MANIFEST_FILENAME as RETRIEVAL_MANIFEST_FILENAME,
        RESULTS_FILENAME as RETRIEVAL_RESULTS_FILENAME,
        SUMMARY_FILENAME as RETRIEVAL_SUMMARY_FILENAME,
        _load_cases,
        _load_json,
        _load_jsonl,
        _load_variants,
        _prepare_inputs,
        _sha256_file,
        _verify_published_payloads,
    )
    from .finalize_query_sft_training_release import publish_training_release
except ImportError:
    import audit_disc_law_sft as tokenizer_loader
    from evaluate_query_sft_full_retrieval import (
        DEFAULT_ARTICLE_INDEX,
        DEFAULT_ARTIFACT_DIR,
        DEFAULT_INPUT_WORK_PACKAGE_DIR,
        DEFAULT_OUTPUT_DIR as DEFAULT_RETRIEVAL_DIR,
        DEFAULT_SEMANTIC_REVIEW_DIR,
        DEFAULT_TEACHER_WORK_PACKAGE_DIR,
        DEFAULT_TEXT_AUDIT,
        HASH_FILENAME as RETRIEVAL_HASH_FILENAME,
        MANIFEST_FILENAME as RETRIEVAL_MANIFEST_FILENAME,
        RESULTS_FILENAME as RETRIEVAL_RESULTS_FILENAME,
        SUMMARY_FILENAME as RETRIEVAL_SUMMARY_FILENAME,
        _load_cases,
        _load_json,
        _load_jsonl,
        _load_variants,
        _prepare_inputs,
        _sha256_file,
        _verify_published_payloads,
    )
    from finalize_query_sft_training_release import publish_training_release


DATASET_ROOT = Path(__file__).resolve().parent
QUERY_POOL_ROOT = DATASET_ROOT / "QUERY-POOL"
DEFAULT_TOKENIZER_PATH = DATASET_ROOT.parent / "model"
DEFAULT_EVALUATION_MANIFEST = (
    DATASET_ROOT / "RAG-SFT" / "manifests" / "evaluation-exclusions-project-rag-v2.json"
)
DEFAULT_QUERY_POOL_PATH = QUERY_POOL_ROOT / "authoring" / "query-pool-v1.jsonl"
DEFAULT_QUERY_POOL_MANIFEST = QUERY_POOL_ROOT / "manifests" / "query-pool-v1.json"
DEFAULT_SOURCE_MAPPING = QUERY_POOL_ROOT / "manifests" / "query-pool-v1-source-mapping.jsonl"
DEFAULT_FORMAL_SOURCE_MAPPING = QUERY_POOL_ROOT / "manifests" / "query-pool-v1-formal-source-mapping.jsonl"
DEFAULT_OUTPUT_DIR = QUERY_POOL_ROOT / "full" / "query-sft-v1-training-release"

AUTHORING_FILENAME = "query-sft-v1-authoring.jsonl"
CANDIDATE_FILENAME = "query-sft-v1-training-candidate.jsonl"
SELECTION_MANIFEST_FILENAME = "query-sft-v1-formal-selection.json"
HASH_FILENAME = "query-sft-v1-formal-selection.sha256"
FORMAL_MANIFEST_FILENAME = "query-sft-v1-formal-release.json"
_TARGET_FIELDS = ("rewrite", "expansion_terms", "subqueries")


class QuerySftFullReleaseError(RuntimeError):
    """表示正式全量 Query-SFT 发布的证据链或输出状态无效。"""


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if records is not None:
        value["records"] = records
    return value


def _sidecar(path: Path, label: str) -> Path:
    sidecar = path.with_suffix(".sha256")
    try:
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftFullReleaseError(f"无法读取{label} SHA-256") from error
    if lines != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftFullReleaseError(f"{label} SHA-256 无效")
    return sidecar


def _json_payload(rows: list[dict[str, object]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for row in rows
    )


def _canonical_target(raw_output: str, label: str) -> dict[str, object]:
    try:
        enhancement = parse_and_validate_query_enhancement(raw_output)
    except (TypeError, QueryEnhancementProtocolError) as error:
        raise QuerySftFullReleaseError(f"{label} 不符合严格 Query Enhancement 协议") from error
    target = {
        "rewrite": enhancement.rewrite,
        "expansion_terms": list(enhancement.expansion_terms),
        "subqueries": list(enhancement.subqueries),
    }
    if tuple(target) != _TARGET_FIELDS:
        raise AssertionError("Query Enhancement target 字段顺序意外变化")
    return target


def _load_final_retrieval(
    *,
    retrieval_dir: Path,
    frozen_inputs: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    final_dir = retrieval_dir / "final"
    _verify_published_payloads(
        final_dir,
        (RETRIEVAL_RESULTS_FILENAME, RETRIEVAL_SUMMARY_FILENAME, RETRIEVAL_MANIFEST_FILENAME),
    )
    manifest = _load_json(final_dir / RETRIEVAL_MANIFEST_FILENAME, "正式 Retrieval manifest")
    summary = _load_json(final_dir / RETRIEVAL_SUMMARY_FILENAME, "正式 Retrieval summary")
    records = _load_jsonl(final_dir / RETRIEVAL_RESULTS_FILENAME, "正式 Retrieval 记录")
    expected = {
        "query_inputs": 574,
        "baseline_variants": 574,
        "noop_variants": 574,
        "approved_teacher_candidate_variants": 1720,
        "total_retrieval_evaluations": 2868,
    }
    actual = summary.get("records")
    if (
        manifest.get("pipeline") != "query_sft_full_retrieval_evaluation_v1"
        or manifest.get("complete") is not True
        or manifest.get("inputs_frozen_before_run") != frozen_inputs
        or manifest.get("validation", {}).get("input_identities_frozen_before_run") is not True
        or manifest.get("validation", {}).get("retrieval_assets_unchanged_after_run") is not True
        or manifest.get("validation", {}).get("baseline_noop_and_all_approved_candidates_compared") is not True
        or not isinstance(actual, dict)
        or any(actual.get(key) != value for key, value in expected.items())
        or not isinstance(summary.get("selections"), list)
        or len(summary["selections"]) != 574
        or len(records) != 2868
    ):
        raise QuerySftFullReleaseError("正式 Retrieval 结果未完整闭合")
    return records, summary, manifest


def _build_final_records(
    *,
    input_work_package_dir: Path,
    teacher_work_package_dir: Path,
    semantic_review_dir: Path,
    text_audit_path: Path,
    article_index: Path,
    artifact_dir: Path,
    retrieval_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    cases, authoring_types, noops, candidates, frozen_inputs = _prepare_inputs(
        input_work_package_dir=input_work_package_dir,
        teacher_work_package_dir=teacher_work_package_dir,
        semantic_review_dir=semantic_review_dir,
        text_audit_path=text_audit_path,
        article_index=article_index,
        artifact_dir=artifact_dir,
    )
    _, summary, retrieval_manifest = _load_final_retrieval(
        retrieval_dir=retrieval_dir,
        frozen_inputs=frozen_inputs,
    )
    input_rows = _load_jsonl(
        Path(input_work_package_dir) / "query-sft-v1-input-candidate-r1.jsonl",
        "冻结正式 Query-SFT 输入",
    )
    input_by_work = {}
    for row in input_rows:
        work_id = row.get("work_id")
        if (
            set(row) != {"work_id", "source_id", "authoring_type", "query_original"}
            or not isinstance(work_id, str)
            or work_id in input_by_work
        ):
            raise QuerySftFullReleaseError("冻结正式 Query-SFT 输入映射无效")
        input_by_work[work_id] = row
    source_by_work = {
        case.query_id: {
            "query_id": input_by_work[case.query_id]["source_id"],
            "query_original": case.query_original,
            "required_chunk_ids": list(case.required_chunk_ids),
        }
        for case in cases
    }
    if set(input_by_work) != set(source_by_work) or any(
        input_by_work[work_id]["query_original"] != source["query_original"]
        or input_by_work[work_id]["authoring_type"] != authoring_types[work_id]
        for work_id, source in source_by_work.items()
    ):
        raise QuerySftFullReleaseError("冻结正式 Query-SFT 输入未与 Retrieval case 闭合")
    selected_work_ids: set[str] = set()
    selection_counts: Counter[str] = Counter()
    authoring: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    query_pool: list[dict[str, object]] = []
    for selection in summary["selections"]:
        if not isinstance(selection, dict):
            raise QuerySftFullReleaseError("Retrieval 选择记录必须是对象")
        work_id = selection.get("work_id")
        variant_id = selection.get("selected_variant_id")
        kind = selection.get("selection")
        if (
            not isinstance(work_id, str)
            or work_id not in source_by_work
            or work_id in selected_work_ids
            or selection.get("authoring_type") != authoring_types[work_id]
            or not isinstance(variant_id, str)
        ):
            raise QuerySftFullReleaseError("Retrieval 选择与冻结输入映射无效")
        if kind == "teacher_candidate":
            candidate = candidates.get(variant_id)
            if candidate is None or candidate[0] != work_id:
                raise QuerySftFullReleaseError("Retrieval 选择了未审核或错配的教师候选")
            target = _canonical_target(candidate[1], variant_id)
        elif kind == "noop":
            if variant_id != f"{work_id}/noop":
                raise QuerySftFullReleaseError("Retrieval no-op 选择标识无效")
            target = _canonical_target(noops[work_id], variant_id)
        else:
            raise QuerySftFullReleaseError("Retrieval 选择类型无效")
        source = source_by_work[work_id]
        authoring.append(
            {
                "id": work_id,
                "source_id": source["query_id"],
                "query_original": source["query_original"],
                "target": target,
            }
        )
        target_json = json.dumps(target, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        candidate_rows.append(
            {
                "id": work_id,
                "source": "query_sft",
                "conversations": [
                    {"role": "system", "content": QUERY_ENHANCEMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": source["query_original"]},
                    {"role": "assistant", "content": target_json},
                ],
            }
        )
        query_pool.append(source)
        selected_work_ids.add(work_id)
        selection_counts[kind] += 1
    if selected_work_ids != set(source_by_work) or sum(selection_counts.values()) != 574:
        raise QuerySftFullReleaseError("Retrieval 选择未覆盖全部 574 条正式输入")
    authoring.sort(key=lambda row: row["id"])
    candidate_rows.sort(key=lambda row: row["id"])
    query_pool.sort(key=lambda row: row["query_id"])
    if len({row["query_id"] for row in query_pool}) != 574:
        raise QuerySftFullReleaseError("正式公共 Query 池 query_id 不唯一")
    evidence = {
        "frozen_inputs": frozen_inputs,
        "retrieval_manifest": _identity(retrieval_dir / "final" / RETRIEVAL_MANIFEST_FILENAME),
        "retrieval_summary": _identity(retrieval_dir / "final" / RETRIEVAL_SUMMARY_FILENAME),
        "retrieval_records": _identity(retrieval_dir / "final" / RETRIEVAL_RESULTS_FILENAME, records=2868),
        "retrieval_hash_manifest": _identity(retrieval_dir / "final" / RETRIEVAL_HASH_FILENAME),
        "selection_counts": dict(selection_counts),
        "retrieval_identity": retrieval_manifest.get("retrieval_identity"),
        "selection_policy": summary.get("selection_policy"),
    }
    return query_pool, authoring, candidate_rows, evidence


def _publish_pool(path: Path, manifest_path: Path, pool: list[dict[str, object]], evidence: dict[str, object]) -> None:
    if any(item.exists() for item in (path, path.with_suffix(".sha256"), manifest_path, manifest_path.with_suffix(".sha256"))):
        raise QuerySftFullReleaseError("正式公共 Query 池输出已存在，禁止覆盖")
    pool_payload = _json_payload(pool)
    manifest = {
        "schema_version": "1.0",
        "pipeline": "query_pool_v1_formal_release",
        "release_status": "formal_query_pool",
        "records": {"queries": 574, "required_chunk_ids_per_query": "1..3"},
        "data": {"query_pool": {"path": str(path.resolve()), "bytes": len(pool_payload.encode("utf-8")), "sha256": hashlib.sha256(pool_payload.encode("utf-8")).hexdigest(), "records": 574}},
        "retrieval_selection": evidence,
        "readiness": {"final_query_text_frozen": True, "query_sft_training_candidate_ready": True, "rag_sft_authoring_ready": True},
        "complete": True,
    }
    published: list[Path] = []
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(pool_payload, encoding="utf-8", newline="\n")
        published.append(path)
        path.with_suffix(".sha256").write_text(f"{_sha256_file(path)}  {path.name}\n", encoding="utf-8", newline="\n")
        published.append(path.with_suffix(".sha256"))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        published.append(manifest_path)
        manifest_path.with_suffix(".sha256").write_text(f"{_sha256_file(manifest_path)}  {manifest_path.name}\n", encoding="utf-8", newline="\n")
        published.append(manifest_path.with_suffix(".sha256"))
    except (OSError, UnicodeError) as error:
        for target in reversed(published):
            target.unlink(missing_ok=True)
        raise QuerySftFullReleaseError("无法发布正式公共 Query 池") from error


def _publish_formal_source_mapping(
    *, source_mapping_path: Path, output_path: Path, query_pool: list[dict[str, object]]
) -> None:
    """投影历史映射到 574 条正式公共池，保留原始 575 条文件不变。"""
    if output_path.exists() or output_path.with_suffix(".sha256").exists():
        raise QuerySftFullReleaseError("正式公共 Query 池 mapping 输出已存在，禁止覆盖")
    rows = _load_jsonl(source_mapping_path, "历史公共 Query 池 mapping")
    expected_ids = {row["query_id"] for row in query_pool}
    projected: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for row in rows:
        if set(row) != {"source_id", "query_id", "source_record_sha256", "query_sha256"}:
            raise QuerySftFullReleaseError("历史公共 Query 池 mapping 字段无效")
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or query_id in seen_ids:
            raise QuerySftFullReleaseError("历史公共 Query 池 mapping query_id 无效")
        seen_ids.add(query_id)
        if query_id in expected_ids:
            projected.append(row)
    if {row["query_id"] for row in projected} != expected_ids or len(projected) != 574:
        raise QuerySftFullReleaseError("正式公共 Query 池 mapping 未与 574 条正式池闭合")
    payload = _json_payload(projected)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8", newline="\n")
        output_path.with_suffix(".sha256").write_text(
            f"{_sha256_file(output_path)}  {output_path.name}\n",
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, UnicodeError) as error:
        output_path.unlink(missing_ok=True)
        output_path.with_suffix(".sha256").unlink(missing_ok=True)
        raise QuerySftFullReleaseError("无法发布正式公共 Query 池 mapping") from error


def _publish_formal_manifest(
    *,
    output_dir: Path,
    query_pool_path: Path,
    query_pool_manifest_path: Path,
    formal_source_mapping_path: Path,
    release: dict[str, object],
) -> Path:
    """将 Retrieval 选择、训练审计和公共池身份收束为唯一最终验收入口。"""
    path = output_dir / FORMAL_MANIFEST_FILENAME
    if path.exists() or path.with_suffix(".sha256").exists():
        raise QuerySftFullReleaseError("正式全量 Query-SFT 总 manifest 已存在，禁止覆盖")
    selection_path = output_dir / SELECTION_MANIFEST_FILENAME
    release_path = output_dir / "query-sft-training-release.json"
    length_path = output_dir / "query-sft-chat-length-audit-768.json"
    label_path = output_dir / "query-sft-label-audit-768.json"
    required_paths = (
        selection_path,
        release_path,
        length_path,
        label_path,
        query_pool_path,
        query_pool_manifest_path,
        formal_source_mapping_path,
    )
    for target in required_paths:
        if target == selection_path:
            continue
        _sidecar(target, f"正式发布输入 {target.name}")
    selection_hash = output_dir / HASH_FILENAME
    try:
        selection_lines = selection_hash.read_text(encoding="utf-8").splitlines()
        selection_entries = {
            name: digest
            for digest, name in (line.split("  ", 1) for line in selection_lines)
        }
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise QuerySftFullReleaseError("正式选择 manifest SHA-256 清单无效") from error
    if (
        selection_entries.get(SELECTION_MANIFEST_FILENAME) != _sha256_file(selection_path)
        or selection_entries.get(AUTHORING_FILENAME) is None
        or selection_entries.get(CANDIDATE_FILENAME) is None
    ):
        raise QuerySftFullReleaseError("正式选择 manifest SHA-256 校验失败")
    if (
        release.get("readiness", {}).get("training_ready") is not True
        or release.get("records", {}).get("training") != 574
    ):
        raise QuerySftFullReleaseError("训练 release 未通过正式全量训练准入")
    payload = {
        "schema_version": "1.0",
        "pipeline": "query_sft_full_formal_release_v1",
        "release_status": "formal_training_ready",
        "inputs": {
            "retrieval_selection": _identity(selection_path),
            "training_release": _identity(release_path),
            "chat_length_audit": _identity(length_path),
            "label_audit": _identity(label_path),
            "formal_query_pool": _identity(query_pool_path, records=574),
            "formal_query_pool_manifest": _identity(query_pool_manifest_path),
            "formal_source_mapping": _identity(formal_source_mapping_path, records=574),
        },
        "records": {
            "query_pool": 574,
            "query_sft_authoring": 574,
            "query_sft_training_candidate": 574,
            "assistant_tokens_per_epoch": release["records"]["assistant_tokens_per_epoch"],
        },
        "readiness": {
            "blind_authoring_complete": True,
            "independent_candidate_review_complete": True,
            "frozen_retrieval_selection_complete": True,
            "public_query_pool_final_text_frozen": True,
            "chat_template_length_audited": True,
            "assistant_only_label_audited": True,
            "training_ready": True,
        },
        "complete": True,
    }
    try:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        path.with_suffix(".sha256").write_text(
            f"{_sha256_file(path)}  {path.name}\n", encoding="utf-8", newline="\n"
        )
    except (OSError, UnicodeError) as error:
        path.unlink(missing_ok=True)
        path.with_suffix(".sha256").unlink(missing_ok=True)
        raise QuerySftFullReleaseError("无法发布正式全量 Query-SFT 总 manifest") from error
    return path


def finalize_query_sft_full(
    *,
    input_work_package_dir: Path = DEFAULT_INPUT_WORK_PACKAGE_DIR,
    teacher_work_package_dir: Path = DEFAULT_TEACHER_WORK_PACKAGE_DIR,
    semantic_review_dir: Path = DEFAULT_SEMANTIC_REVIEW_DIR,
    text_audit_path: Path = DEFAULT_TEXT_AUDIT,
    article_index: Path = DEFAULT_ARTICLE_INDEX,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    retrieval_dir: Path = DEFAULT_RETRIEVAL_DIR,
    evaluation_manifest_path: Path = DEFAULT_EVALUATION_MANIFEST,
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    query_pool_path: Path = DEFAULT_QUERY_POOL_PATH,
    query_pool_manifest_path: Path = DEFAULT_QUERY_POOL_MANIFEST,
    source_mapping_path: Path = DEFAULT_SOURCE_MAPPING,
    formal_source_mapping_path: Path = DEFAULT_FORMAL_SOURCE_MAPPING,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tokenizer: Any | None = None,
) -> dict[str, object]:
    """发布唯一正式全量 Query-SFT，且在最终审计前不标记 training_ready。"""
    input_work_package_dir = Path(input_work_package_dir).resolve()
    teacher_work_package_dir = Path(teacher_work_package_dir).resolve()
    semantic_review_dir = Path(semantic_review_dir).resolve()
    text_audit_path = Path(text_audit_path).resolve()
    article_index = Path(article_index).resolve()
    artifact_dir = Path(artifact_dir).resolve()
    retrieval_dir = Path(retrieval_dir).resolve()
    evaluation_manifest_path = Path(evaluation_manifest_path).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    query_pool_path = Path(query_pool_path).resolve()
    query_pool_manifest_path = Path(query_pool_manifest_path).resolve()
    source_mapping_path = Path(source_mapping_path).resolve()
    formal_source_mapping_path = Path(formal_source_mapping_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise QuerySftFullReleaseError("正式全量 Query-SFT 输出目录已存在，禁止覆盖")
    pool_targets = (
        query_pool_path,
        query_pool_path.with_suffix(".sha256"),
        query_pool_manifest_path,
        query_pool_manifest_path.with_suffix(".sha256"),
        formal_source_mapping_path,
        formal_source_mapping_path.with_suffix(".sha256"),
    )
    if any(path.exists() for path in pool_targets):
        raise QuerySftFullReleaseError("正式公共 Query 池输出已存在，禁止覆盖")
    _sidecar(evaluation_manifest_path, "评估隔离 manifest")
    query_pool, authoring, candidate_rows, evidence = _build_final_records(
        input_work_package_dir=input_work_package_dir,
        teacher_work_package_dir=teacher_work_package_dir,
        semantic_review_dir=semantic_review_dir,
        text_audit_path=text_audit_path,
        article_index=article_index,
        artifact_dir=artifact_dir,
        retrieval_dir=retrieval_dir,
    )
    authoring_payload = _json_payload(authoring)
    candidate_payload = _json_payload(candidate_rows)
    output_dir.mkdir(parents=True)
    pool_published = False
    try:
        (output_dir / AUTHORING_FILENAME).write_text(authoring_payload, encoding="utf-8", newline="\n")
        (output_dir / CANDIDATE_FILENAME).write_text(candidate_payload, encoding="utf-8", newline="\n")
        selection_manifest = {
            "schema_version": "1.0",
            "pipeline": "query_sft_full_selection_v1",
            "release_status": "formal_training_projection_pending_audit",
            "inputs": evidence,
            "outputs": {
                "authoring": {"path": str((output_dir / AUTHORING_FILENAME).resolve()), "records": 574, "bytes": len(authoring_payload.encode("utf-8")), "sha256": hashlib.sha256(authoring_payload.encode("utf-8")).hexdigest()},
                "training_candidate": {"path": str((output_dir / CANDIDATE_FILENAME).resolve()), "records": 574, "bytes": len(candidate_payload.encode("utf-8")), "sha256": hashlib.sha256(candidate_payload.encode("utf-8")).hexdigest()},
            },
            "validation": {"all_final_queries_selected_once": True, "required_gt_not_emitted_to_training_candidate": True, "only_approved_teacher_candidates_or_deterministic_noop_selected": True},
            "readiness": {"frozen_retrieval_selection_complete": True, "training_ready": False},
            "complete": True,
        }
        selection_path = output_dir / SELECTION_MANIFEST_FILENAME
        selection_path.write_text(json.dumps(selection_manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        (output_dir / HASH_FILENAME).write_text(
            "".join(
                f"{_sha256_file(output_dir / name)}  {name}\n"
                for name in (AUTHORING_FILENAME, CANDIDATE_FILENAME, SELECTION_MANIFEST_FILENAME)
            ),
            encoding="utf-8",
            newline="\n",
        )
        _publish_pool(query_pool_path, query_pool_manifest_path, query_pool, evidence)
        pool_published = True
        _publish_formal_source_mapping(
            source_mapping_path=source_mapping_path,
            output_path=formal_source_mapping_path,
            query_pool=query_pool,
        )
        release = publish_training_release(
            candidate_path=output_dir / CANDIDATE_FILENAME,
            evaluation_manifest_path=evaluation_manifest_path,
            tokenizer_path=tokenizer_path,
            output_dir=output_dir,
            release_status="formal_training_candidate",
            tokenizer=tokenizer or tokenizer_loader.load_tokenizer(tokenizer_path),
        )
        _publish_formal_manifest(
            output_dir=output_dir,
            query_pool_path=query_pool_path,
            query_pool_manifest_path=query_pool_manifest_path,
            formal_source_mapping_path=formal_source_mapping_path,
            release=release,
        )
    except Exception:
        # 只清理由本函数刚创建且尚未形成有效正式发布的产物。
        for path in (
            output_dir / "query-sft-chat-length-audit-768.json",
            output_dir / "query-sft-chat-length-audit-768.sha256",
            output_dir / "query-sft-label-audit-768.json",
            output_dir / "query-sft-label-audit-768.sha256",
            output_dir / "query-sft-training-release.json",
            output_dir / "query-sft-training-release.sha256",
            output_dir / FORMAL_MANIFEST_FILENAME,
            output_dir / "query-sft-v1-formal-release.sha256",
            output_dir / HASH_FILENAME,
            output_dir / SELECTION_MANIFEST_FILENAME,
            output_dir / CANDIDATE_FILENAME,
            output_dir / AUTHORING_FILENAME,
        ):
            path.unlink(missing_ok=True)
        try:
            output_dir.rmdir()
        except OSError:
            pass
        if pool_published:
            for path in pool_targets:
                path.unlink(missing_ok=True)
        raise
    return release


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-dir", type=Path, default=DEFAULT_RETRIEVAL_DIR)
    parser.add_argument("--evaluation-manifest", type=Path, default=DEFAULT_EVALUATION_MANIFEST)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--query-pool", type=Path, default=DEFAULT_QUERY_POOL_PATH)
    parser.add_argument("--query-pool-manifest", type=Path, default=DEFAULT_QUERY_POOL_MANIFEST)
    parser.add_argument("--source-mapping", type=Path, default=DEFAULT_SOURCE_MAPPING)
    parser.add_argument("--formal-source-mapping", type=Path, default=DEFAULT_FORMAL_SOURCE_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        release = finalize_query_sft_full(
            retrieval_dir=args.retrieval_dir,
            evaluation_manifest_path=args.evaluation_manifest,
            tokenizer_path=args.tokenizer_path,
            query_pool_path=args.query_pool,
            query_pool_manifest_path=args.query_pool_manifest,
            source_mapping_path=args.source_mapping,
            formal_source_mapping_path=args.formal_source_mapping,
            output_dir=args.output_dir,
        )
    except (OSError, ValueError, TypeError, KeyError, QuerySftFullReleaseError) as error:
        parser.error(str(error))
    print(
        "QUERY_SFT_FULL_RELEASE_OK "
        f"records={release['records']['training']} "
        f"assistant_tokens={release['records']['assistant_tokens_per_epoch']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
