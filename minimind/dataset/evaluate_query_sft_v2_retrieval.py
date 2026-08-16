"""按冻结 Retrieval 资产分片评估 Query-SFT v2 的非退化候选选择。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from rag.answering import AnswerPromptTokenCounter, EvidencePackager
from rag.eval.retrieval_baseline import (
    CONTEXT_LIMIT,
    MAX_OUTPUT_TOKENS,
    _file_identity,
    _load_tokenizer,
    _tokenizer_identity,
)
from rag.eval.retrieval_chain_evaluation import (
    EvaluationCase,
    EvaluationMode,
    build_query_evaluation_plan,
    evaluate_retrieval_case,
)
from rag.knowledge import ArticleRepository
from rag.retrieval import SemanticRetrievalConfig, load_semantic_retriever
from rag.retrieval.loader import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    _resolve_device,
)

from .audit_query_sft_v2_candidate_texts import REPORT_FILENAME as TEXT_AUDIT_FILENAME
from .audit_query_sft_v2_teacher_candidates import RESULT_FILENAME
from .finalize_query_sft_v2_semantic_review import LEDGER_FILENAME
from .query_sft_v2_contract import QuerySftV2ContractError, select_non_degrading_candidate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path(__file__).resolve().parent
RAG_ROOT = PROJECT_ROOT / "rag"
DEFAULT_INPUT_WORK_PACKAGE_DIR = DATASET_ROOT / "QUERY-POOL" / "full" / "query-sft-v2-work-package"
DEFAULT_TEACHER_RESULTS = (
    DATASET_ROOT
    / "QUERY-POOL"
    / "full"
    / "query-sft-v2-local-teacher-generation-r1"
    / RESULT_FILENAME
)
DEFAULT_NOOP_PATH = (
    DATASET_ROOT
    / "QUERY-POOL"
    / "full"
    / "query-sft-v2-work-package"
    / "query-sft-v2-teacher-candidate-work-package"
    / "query-sft-v2-deterministic-noop-candidates.jsonl"
)
DEFAULT_NOOP_WORK_PACKAGE_HASH = DEFAULT_NOOP_PATH.parent / "query-sft-v2-teacher-candidate-work-package.sha256"
DEFAULT_SEMANTIC_REVIEW_DIR = (
    DATASET_ROOT / "QUERY-POOL" / "full" / "query-sft-v2-semantic-quality-r2a"
)
DEFAULT_TEXT_AUDIT = DEFAULT_SEMANTIC_REVIEW_DIR / TEXT_AUDIT_FILENAME
DEFAULT_ARTICLE_INDEX = RAG_ROOT / "chunk" / "article_index.jsonl"
DEFAULT_ARTIFACT_DIR = RAG_ROOT / "retrieval" / "artifacts"
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "minimind" / "model"
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "QUERY-POOL" / "full" / "query-sft-v2-retrieval-evaluation-r1"
INPUT_FILENAME = "query-sft-v2-input-candidate.jsonl"
PARTS_DIRECTORY_NAME = "parts"
PART_RECORDS_FILENAME = "query-sft-v2-retrieval-part-records.jsonl"
PART_MANIFEST_FILENAME = "query-sft-v2-retrieval-part-manifest.json"
RESULTS_FILENAME = "query-sft-v2-retrieval-records.jsonl"
SUMMARY_FILENAME = "query-sft-v2-retrieval-summary.json"
MANIFEST_FILENAME = "query-sft-v2-retrieval-manifest.json"
HASH_FILENAME = "query-sft-v2-retrieval-manifest.sha256"
DEFAULT_BATCH_SIZE = 10
INPUT_FIELDS = ("work_id", "source_id", "authoring_type", "query_original")
REFERENCE_FIELDS = ("work_id", "source_id", "authoring_type", "query_original", "required_chunk_ids")
RESULT_FIELDS = ("candidate_id", "work_id", "raw_output")
SEMANTIC_REVIEW_FIELDS = (
    "candidate_id",
    "work_id",
    "review_decision",
    "semantic_preserved",
    "unsupported_fact_absent",
    "ambiguity_not_resolved",
    "natural_text",
    "authoring_type_fulfilled",
    "subquery_coverage_complete",
    "explicit_multi_matter_exception",
    "reason",
)
ARTIFACT_FILENAMES = ("law_dense.faiss", "law_dense_meta.json", "law_sparse.pkl")


class QuerySftV2RetrievalEvaluationError(RuntimeError):
    """表示 v2 Retrieval 评估输入、分片或聚合不满足冻结协议。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    value: dict[str, object] = _file_identity(path)
    if records is not None:
        value["records"] = records
    return value


def _verify_sidecar(path: Path, label: str) -> Path:
    sidecar = path.with_suffix(".sha256")
    try:
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftV2RetrievalEvaluationError(f"无法读取{label} SHA-256") from error
    if lines != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftV2RetrievalEvaluationError(f"{label} SHA-256 无效")
    return sidecar


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftV2RetrievalEvaluationError(f"无法读取{label}") from error
    if not isinstance(value, dict):
        raise QuerySftV2RetrievalEvaluationError(f"{label}必须是 JSON 对象")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    raise QuerySftV2RetrievalEvaluationError(f"{label}不允许空行：{number}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise QuerySftV2RetrievalEvaluationError(f"{label}第 {number} 条必须是对象")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftV2RetrievalEvaluationError):
            raise
        raise QuerySftV2RetrievalEvaluationError(f"无法读取{label}") from error
    return rows


def _score(record: Any) -> tuple[int, float, float]:
    metrics = record.package_metrics if hasattr(record, "package_metrics") else record["packaging"]["metrics"]
    return (
        int(metrics["complete_hit"]),
        float(metrics["required_gt_coverage"]),
        float(metrics["reciprocal_rank"]),
    )


def _score_dict(record: dict[str, object]) -> tuple[int, float, float]:
    return _score(record["record"])


def _load_cases(input_dir: Path) -> tuple[tuple[EvaluationCase, ...], dict[str, str], dict[str, object]]:
    package_hash = input_dir / "query-sft-v2-work-package.sha256"
    package_manifest = input_dir / "query-sft-v2-work-package.json"
    try:
        signed_entries = {
            name: digest
            for line in package_hash.read_text(encoding="utf-8").splitlines()
            for digest, name in [line.split("  ", 1)]
        }
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise QuerySftV2RetrievalEvaluationError("v2 正式输入工作包 SHA-256 清单无效") from error
    required_signed_paths = {
        INPUT_FILENAME,
        "query-sft-v2-work-package.json",
        *(f"audit-reference/batch-{batch:02d}.jsonl" for batch in range(1, 9)),
    }
    if not required_signed_paths <= set(signed_entries):
        raise QuerySftV2RetrievalEvaluationError("v2 正式输入工作包缺少必要签名文件")
    for name in required_signed_paths:
        path = input_dir / name
        if not path.is_file() or signed_entries[name] != _sha256_file(path):
            raise QuerySftV2RetrievalEvaluationError(f"v2 正式输入工作包身份已变化：{name}")
    package = _load_json(package_manifest, "v2 正式输入工作包 manifest")
    if (
        package.get("pipeline") != "query_sft_v2_work_package"
        or package.get("records", {}).get("query_inputs") != 574
        or package.get("complete") is not True
    ):
        raise QuerySftV2RetrievalEvaluationError("v2 正式输入工作包状态无效")
    input_path = input_dir / INPUT_FILENAME
    inputs: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(input_path, "v2 正式输入"):
        work_id = row.get("work_id")
        if tuple(row) != INPUT_FIELDS or not isinstance(work_id, str) or work_id in inputs:
            raise QuerySftV2RetrievalEvaluationError("v2 正式输入映射无效")
        inputs[work_id] = row
    references: dict[str, dict[str, Any]] = {}
    for batch in range(1, 9):
        for row in _load_jsonl(input_dir / "audit-reference" / f"batch-{batch:02d}.jsonl", "v2 GT 审核引用"):
            work_id = row.get("work_id")
            required = row.get("required_chunk_ids")
            if (
                tuple(row) != REFERENCE_FIELDS
                or not isinstance(work_id, str)
                or work_id not in inputs
                or work_id in references
                or row.get("source_id") != inputs[work_id]["source_id"]
                or row.get("authoring_type") != inputs[work_id]["authoring_type"]
                or row.get("query_original") != inputs[work_id]["query_original"]
                or not isinstance(required, list)
                or not required
                or any(not isinstance(value, str) or not value for value in required)
            ):
                raise QuerySftV2RetrievalEvaluationError("v2 GT 审核引用映射无效")
            references[work_id] = row
    if len(inputs) != 574 or set(inputs) != set(references):
        raise QuerySftV2RetrievalEvaluationError("v2 正式输入与 GT 审核引用必须闭合为 574 条")
    cases = tuple(
        EvaluationCase(
            query_id=work_id,
            query_original=inputs[work_id]["query_original"],
            required_chunk_ids=tuple(references[work_id]["required_chunk_ids"]),
        )
        for work_id in inputs
    )
    return cases, {work_id: row["authoring_type"] for work_id, row in inputs.items()}, {
        "input": input_path,
        "work_package_manifest": package_manifest,
        "work_package_hash_manifest": package_hash,
        "audit_reference_dir": input_dir / "audit-reference",
    }


def _verify_work_package_entry(path: Path, *, manifest_path: Path) -> Path:
    try:
        entries = {
            name: digest
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            for digest, name in [line.split("  ", 1)]
        }
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise QuerySftV2RetrievalEvaluationError("v2 no-op 工作包 SHA-256 清单无效") from error
    if entries.get(path.name) != _sha256_file(path):
        raise QuerySftV2RetrievalEvaluationError("v2 no-op 未通过工作包 SHA-256 校验")
    return manifest_path


def _load_variants(
    *, teacher_results: Path, noop_path: Path, noop_work_package_hash: Path, review_dir: Path
) -> tuple[dict[str, str], dict[str, tuple[str, str]], dict[str, object]]:
    ledger_path = review_dir / LEDGER_FILENAME
    semantic_audit = review_dir / "query-sft-v2-semantic-quality-audit.json"
    text_audit = review_dir / TEXT_AUDIT_FILENAME
    results_hash = _verify_sidecar(teacher_results, "v2 教师候选结果")
    noop_hash = _verify_work_package_entry(noop_path, manifest_path=noop_work_package_hash)
    ledger_hash = _verify_sidecar(ledger_path, "v2 语义审核账本")
    semantic_hash = _verify_sidecar(semantic_audit, "v2 语义质量报告")
    text_hash = _verify_sidecar(text_audit, "v2 候选文本审计")
    semantic = _load_json(semantic_audit, "v2 语义质量报告")
    text = _load_json(text_audit, "v2 候选文本审计")
    if (
        semantic.get("pipeline") != "query_sft_v2_semantic_quality_audit"
        or semantic.get("readiness", {}).get("semantic_quality_ready") is not True
        or text.get("pipeline") != "query_sft_v2_candidate_text_audit"
        or text.get("records", {}).get("approved") != semantic.get("records", {}).get("approved")
        or text.get("records", {}).get("evaluation_text_overlap") != 0
    ):
        raise QuerySftV2RetrievalEvaluationError("v2 语义质量或文本隔离审计状态无效")
    decisions: dict[str, str] = {}
    for row in _load_jsonl(ledger_path, "v2 语义审核账本"):
        candidate_id = row.get("candidate_id")
        if (
            tuple(row) != SEMANTIC_REVIEW_FIELDS
            or not isinstance(candidate_id, str)
            or candidate_id in decisions
            or row.get("review_decision") not in {"approved", "rejected"}
        ):
            raise QuerySftV2RetrievalEvaluationError("v2 语义审核账本映射无效")
        decisions[candidate_id] = row["review_decision"]
    candidates: dict[str, tuple[str, str]] = {}
    for row in _load_jsonl(teacher_results, "v2 教师候选结果"):
        candidate_id = row.get("candidate_id")
        work_id = row.get("work_id")
        if (
            tuple(row) != RESULT_FIELDS
            or not isinstance(candidate_id, str)
            or not isinstance(work_id, str)
            or candidate_id not in decisions
        ):
            raise QuerySftV2RetrievalEvaluationError("v2 教师候选映射无效")
        if decisions[candidate_id] == "approved":
            candidates[candidate_id] = (work_id, row["raw_output"])
    noops: dict[str, str] = {}
    for row in _load_jsonl(noop_path, "v2 确定性 no-op"):
        candidate_id = row.get("candidate_id")
        work_id = row.get("work_id")
        if (
            tuple(row) != RESULT_FIELDS
            or not isinstance(candidate_id, str)
            or not isinstance(work_id, str)
            or work_id in noops
            or candidate_id != f"{work_id}/noop"
        ):
            raise QuerySftV2RetrievalEvaluationError("v2 确定性 no-op 映射无效")
        noops[work_id] = row["raw_output"]
    if len(decisions) != 1722 or len(noops) != 574 or not candidates:
        raise QuerySftV2RetrievalEvaluationError("v2 候选、审核或 no-op 数量无效")
    return noops, candidates, {
        "teacher_results": {**_identity(teacher_results, records=1722), "hash_manifest": _identity(results_hash)},
        "deterministic_noop": {**_identity(noop_path, records=574), "work_package_hash_manifest": _identity(noop_hash)},
        "semantic_ledger": {**_identity(ledger_path, records=1722), "hash_manifest": _identity(ledger_hash)},
        "semantic_quality_audit": {**_identity(semantic_audit), "hash_manifest": _identity(semantic_hash)},
        "candidate_text_audit": {**_identity(text_audit), "hash_manifest": _identity(text_hash)},
    }


def _prepare_inputs(
    *,
    input_work_package_dir: Path,
    teacher_results: Path,
    noop_path: Path,
    noop_work_package_hash: Path,
    semantic_review_dir: Path,
    article_index: Path,
    artifact_dir: Path,
) -> tuple[tuple[EvaluationCase, ...], dict[str, str], dict[str, str], dict[str, tuple[str, str]], dict[str, object]]:
    input_dir = Path(input_work_package_dir).resolve()
    article_index = Path(article_index).resolve()
    artifact_dir = Path(artifact_dir).resolve()
    cases, authoring_types, input_identity = _load_cases(input_dir)
    noops, candidates, variant_identity = _load_variants(
        teacher_results=Path(teacher_results).resolve(),
        noop_path=Path(noop_path).resolve(),
        noop_work_package_hash=Path(noop_work_package_hash).resolve(),
        review_dir=Path(semantic_review_dir).resolve(),
    )
    if set(noops) != set(authoring_types) or {work_id for work_id, _ in candidates.values()} - set(authoring_types):
        raise QuerySftV2RetrievalEvaluationError("v2 候选与正式输入 work_id 未闭合")
    frozen = {
        "input_candidate": _identity(input_identity["input"], records=574),
        "input_work_package": {
            "manifest": _identity(input_identity["work_package_manifest"]),
            "hash_manifest": _identity(input_identity["work_package_hash_manifest"]),
        },
        "audit_reference_directory": str(Path(input_identity["audit_reference_dir"]).resolve()),
        "article_index": _identity(article_index),
        "retrieval_artifacts": {name: _identity(artifact_dir / name) for name in ARTIFACT_FILENAMES},
        **variant_identity,
    }
    return cases, authoring_types, noops, candidates, frozen


def _part_dir(output_dir: Path, batch_index: int) -> Path:
    return output_dir / PARTS_DIRECTORY_NAME / f"batch-{batch_index:03d}"


def _select_batch(cases: tuple[EvaluationCase, ...], *, batch_index: int, batch_size: int) -> tuple[EvaluationCase, ...]:
    if batch_index < 0 or batch_size <= 0:
        raise QuerySftV2RetrievalEvaluationError("batch_index 必须非负且 batch_size 必须为正数")
    selected = cases[batch_index * batch_size : (batch_index + 1) * batch_size]
    if not selected:
        raise QuerySftV2RetrievalEvaluationError("请求的 Retrieval 分片超出 574 条输入范围")
    return selected


def _evaluate_variants(
    *, cases: tuple[EvaluationCase, ...], noops: dict[str, str], candidates: dict[str, tuple[str, str]], retriever: Any, packager: EvidencePackager
) -> list[dict[str, object]]:
    by_work: dict[str, list[tuple[str, str]]] = {case.query_id: [] for case in cases}
    for candidate_id, (work_id, raw_output) in candidates.items():
        if work_id in by_work:
            by_work[work_id].append((candidate_id, raw_output))
    rows = []
    for position, case in enumerate(cases, 1):
        variants: list[tuple[str, str, str | None, EvaluationMode]] = [
            ("baseline", f"{case.query_id}/baseline", None, EvaluationMode.BASELINE_ORIGINAL),
            ("noop", f"{case.query_id}/noop", noops[case.query_id], EvaluationMode.NOOP_APPLIED),
        ]
        variants.extend(("teacher_candidate", candidate_id, raw_output, EvaluationMode.FULL_ENHANCEMENT) for candidate_id, raw_output in by_work[case.query_id])
        for kind, variant_id, raw_output, mode in variants:
            plan = (
                build_query_evaluation_plan(case.query_original, mode, raw_enhancement=raw_output)
                if mode is EvaluationMode.FULL_ENHANCEMENT
                else build_query_evaluation_plan(case.query_original, mode)
            )
            record = evaluate_retrieval_case(case, plan, retriever=retriever, packager=packager)
            rows.append({"variant_kind": kind, "variant_id": variant_id, "work_id": case.query_id, "record": record.to_dict()})
        print(f"QUERY_SFT_V2_RETRIEVAL_PROGRESS completed={position}/{len(cases)}", flush=True)
    return rows


def run_query_sft_v2_retrieval_batch(
    *,
    batch_index: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    input_work_package_dir: Path = DEFAULT_INPUT_WORK_PACKAGE_DIR,
    teacher_results: Path = DEFAULT_TEACHER_RESULTS,
    noop_path: Path = DEFAULT_NOOP_PATH,
    noop_work_package_hash: Path = DEFAULT_NOOP_WORK_PACKAGE_HASH,
    semantic_review_dir: Path = DEFAULT_SEMANTIC_REVIEW_DIR,
    article_index: Path = DEFAULT_ARTICLE_INDEX,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device: str = "auto",
) -> dict[str, object]:
    """运行一个不可覆盖的 work_id 分片；同题全部变体固定在同一分片。"""

    output_dir = Path(output_dir).resolve()
    part_dir = _part_dir(output_dir, batch_index)
    if part_dir.exists():
        raise QuerySftV2RetrievalEvaluationError("Retrieval 分片输出已存在，禁止覆盖")
    cases, authoring_types, noops, candidates, frozen = _prepare_inputs(
        input_work_package_dir=input_work_package_dir,
        teacher_results=teacher_results,
        noop_path=noop_path,
        noop_work_package_hash=noop_work_package_hash,
        semantic_review_dir=semantic_review_dir,
        article_index=article_index,
        artifact_dir=artifact_dir,
    )
    selected = _select_batch(cases, batch_index=batch_index, batch_size=batch_size)
    actual_device = _resolve_device(device)
    tokenizer = _load_tokenizer(Path(tokenizer_path).resolve())
    repository = ArticleRepository.from_jsonl(Path(article_index).resolve())
    retriever = load_semantic_retriever(
        repository=repository,
        artifact_dir=Path(artifact_dir).resolve(),
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        reranker_model=DEFAULT_RERANKER_MODEL,
        device=actual_device,
        config=SemanticRetrievalConfig(),
    )
    packager = EvidencePackager(
        context_limit=CONTEXT_LIMIT,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        count_prompt_tokens=AnswerPromptTokenCounter(tokenizer),
    )
    rows = _evaluate_variants(cases=selected, noops=noops, candidates=candidates, retriever=retriever, packager=packager)
    after = {
        "article_index": _identity(Path(article_index).resolve()),
        "retrieval_artifacts": {name: _identity(Path(artifact_dir).resolve() / name) for name in ARTIFACT_FILENAMES},
    }
    if after != {"article_index": frozen["article_index"], "retrieval_artifacts": frozen["retrieval_artifacts"]}:
        raise QuerySftV2RetrievalEvaluationError("Retrieval 资产在分片运行期间发生变化")
    manifest = {
        "schema_version": "1.0",
        "pipeline": "query_sft_v2_retrieval_evaluation_part",
        "batch_index": batch_index,
        "batch_size": batch_size,
        "work_ids": [case.query_id for case in selected],
        "inputs_frozen_before_run": frozen,
        "retrieval_identity": {
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "reranker_model": DEFAULT_RERANKER_MODEL,
            "device": actual_device,
            "config": SemanticRetrievalConfig().__dict__,
            "tokenizer": _tokenizer_identity(tokenizer, Path(tokenizer_path).resolve()),
        },
        "records": {"query_inputs": len(selected), "retrieval_evaluations": len(rows)},
        "validation": {"retrieval_assets_unchanged_after_run": True, "query_sft_training_ready": False},
        "complete": True,
    }
    try:
        part_dir.mkdir(parents=True)
        (part_dir / PART_RECORDS_FILENAME).write_text(
            "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in rows),
            encoding="utf-8",
            newline="\n",
        )
        (part_dir / PART_MANIFEST_FILENAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n"
        )
        for path in (part_dir / PART_RECORDS_FILENAME, part_dir / PART_MANIFEST_FILENAME):
            path.with_suffix(".sha256").write_text(f"{_sha256_file(path)}  {path.name}\n", encoding="utf-8", newline="\n")
    except (OSError, UnicodeError) as error:
        raise QuerySftV2RetrievalEvaluationError("无法发布 v2 Retrieval 分片") from error
    return manifest


def _load_part(part_dir: Path, frozen: dict[str, object]) -> list[dict[str, Any]]:
    manifest_path = part_dir / PART_MANIFEST_FILENAME
    records_path = part_dir / PART_RECORDS_FILENAME
    _verify_sidecar(manifest_path, "Retrieval 分片 manifest")
    _verify_sidecar(records_path, "Retrieval 分片记录")
    manifest = _load_json(manifest_path, "Retrieval 分片 manifest")
    if manifest.get("pipeline") != "query_sft_v2_retrieval_evaluation_part" or manifest.get("inputs_frozen_before_run") != frozen or manifest.get("complete") is not True:
        raise QuerySftV2RetrievalEvaluationError("Retrieval 分片身份或状态无效")
    return _load_jsonl(records_path, "Retrieval 分片记录")


def finalize_query_sft_v2_retrieval(
    *,
    input_work_package_dir: Path = DEFAULT_INPUT_WORK_PACKAGE_DIR,
    teacher_results: Path = DEFAULT_TEACHER_RESULTS,
    noop_path: Path = DEFAULT_NOOP_PATH,
    noop_work_package_hash: Path = DEFAULT_NOOP_WORK_PACKAGE_HASH,
    semantic_review_dir: Path = DEFAULT_SEMANTIC_REVIEW_DIR,
    article_index: Path = DEFAULT_ARTICLE_INDEX,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    """聚合所有分片，并用 v2 持平可选的非退化协议冻结每题候选。"""

    output_dir = Path(output_dir).resolve()
    results_path = output_dir / RESULTS_FILENAME
    summary_path = output_dir / SUMMARY_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    if any(path.exists() for path in (results_path, summary_path, manifest_path)):
        raise QuerySftV2RetrievalEvaluationError("最终 Retrieval 输出已存在，禁止覆盖")
    cases, authoring_types, noops, candidates, frozen = _prepare_inputs(
        input_work_package_dir=input_work_package_dir,
        teacher_results=teacher_results,
        noop_path=noop_path,
        noop_work_package_hash=noop_work_package_hash,
        semantic_review_dir=semantic_review_dir,
        article_index=article_index,
        artifact_dir=artifact_dir,
    )
    all_rows: list[dict[str, Any]] = []
    seen_work_ids: set[str] = set()
    for part_dir in sorted((output_dir / PARTS_DIRECTORY_NAME).glob("batch-*")):
        rows = _load_part(part_dir, frozen)
        work_ids = {row.get("work_id") for row in rows}
        if not work_ids or not work_ids <= set(authoring_types) or seen_work_ids & work_ids:
            raise QuerySftV2RetrievalEvaluationError("Retrieval 分片 work_id 覆盖无效或重叠")
        seen_work_ids.update(work_ids)
        all_rows.extend(rows)
    if seen_work_ids != set(authoring_types):
        raise QuerySftV2RetrievalEvaluationError("Retrieval 分片未完整覆盖 574 条正式输入")
    variants: dict[str, dict[str, dict[str, Any]]] = {case.query_id: {} for case in cases}
    for row in all_rows:
        kind, variant_id, work_id = row.get("variant_kind"), row.get("variant_id"), row.get("work_id")
        if kind not in {"baseline", "noop", "teacher_candidate"} or not isinstance(variant_id, str) or work_id not in variants or variant_id in variants[work_id]:
            raise QuerySftV2RetrievalEvaluationError("Retrieval 记录身份无效")
        variants[work_id][variant_id] = row
    selections = []
    selection_counts = {"teacher_candidate": 0, "noop": 0}
    for work_id, type_name in authoring_types.items():
        group = variants[work_id]
        baseline = group.get(f"{work_id}/baseline")
        noop = group.get(f"{work_id}/noop")
        if baseline is None or noop is None:
            raise QuerySftV2RetrievalEvaluationError("每题必须包含 baseline 与确定性 no-op")
        approved = [(candidate_id, _score_dict(record)) for candidate_id, record in group.items() if candidate_id in candidates]
        try:
            kind, winner_id, winner_score = select_non_degrading_candidate(
                baseline_score=_score_dict(baseline), noop_score=_score_dict(noop), approved_candidates=approved
            )
        except QuerySftV2ContractError as error:
            raise QuerySftV2RetrievalEvaluationError(f"Retrieval 非退化选择失败：{work_id}") from error
        selection_counts[kind] += 1
        selections.append({
            "work_id": work_id,
            "authoring_type": type_name,
            "selected_variant_id": winner_id if kind == "teacher_candidate" else f"{work_id}/noop",
            "selection": kind,
            "baseline_score": _score_dict(baseline),
            "noop_score": _score_dict(noop),
            "selected_score": winner_score,
            "approved_teacher_candidates_compared": len(approved),
        })
    summary = {
        "schema_version": "1.0",
        "pipeline": "query_sft_v2_retrieval_selection",
        "records": {
            "query_inputs": len(selections),
            "baseline_variants": len(selections),
            "noop_variants": len(selections),
            "approved_teacher_candidate_variants": sum(row["approved_teacher_candidates_compared"] for row in selections),
            "total_retrieval_evaluations": len(all_rows),
            "selected_by_kind": selection_counts,
        },
        "selection_policy": {
            "primary_metrics": ["complete_hit@5", "required_gt_coverage@5", "mrr@5"],
            "candidate_must_not_degrade_baseline_or_noop": True,
            "semantic_approved_ties_allowed": True,
            "deterministic_noop_must_not_degrade_baseline": True,
        },
        "selections": selections,
        "complete": True,
    }
    manifest = {
        "schema_version": "1.0",
        "pipeline": "query_sft_v2_retrieval_evaluation",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs_frozen_before_run": frozen,
        "records": summary["records"],
        "selection_policy": summary["selection_policy"],
        "validation": {
            "all_queries_evaluated_with_baseline_noop_and_semantic_approved_candidates": True,
            "selection_uses_non_degrading_tie_allowed_v2_contract": True,
            "query_sft_training_ready": False,
        },
        "complete": True,
    }
    try:
        # 分片先创建输出根目录；聚合仅允许在尚未发布最终文件时写入。
        output_dir.mkdir(parents=True, exist_ok=True)
        if any(path.exists() for path in (results_path, summary_path, manifest_path, output_dir / HASH_FILENAME)):
            raise QuerySftV2RetrievalEvaluationError("Retrieval 聚合结果已存在，禁止覆盖")
        results_path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in all_rows), encoding="utf-8", newline="\n")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
        hash_path = output_dir / HASH_FILENAME
        hash_path.write_text("".join(f"{_sha256_file(path)}  {path.name}\n" for path in (results_path, summary_path, manifest_path)), encoding="utf-8", newline="\n")
    except (OSError, UnicodeError) as error:
        raise QuerySftV2RetrievalEvaluationError("无法发布 v2 Retrieval 聚合结果") from error
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-work-package-dir", type=Path, default=DEFAULT_INPUT_WORK_PACKAGE_DIR)
    parser.add_argument("--teacher-results", type=Path, default=DEFAULT_TEACHER_RESULTS)
    parser.add_argument("--noop-path", type=Path, default=DEFAULT_NOOP_PATH)
    parser.add_argument("--noop-work-package-hash", type=Path, default=DEFAULT_NOOP_WORK_PACKAGE_HASH)
    parser.add_argument("--semantic-review-dir", type=Path, default=DEFAULT_SEMANTIC_REVIEW_DIR)
    parser.add_argument("--article-index", type=Path, default=DEFAULT_ARTICLE_INDEX)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--batch-index", type=int)
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    print("QUERY_SFT_V2_RETRIEVAL_ENTRYPOINT_READY", flush=True)
    try:
        if args.finalize:
            result = finalize_query_sft_v2_retrieval(
                input_work_package_dir=args.input_work_package_dir,
                teacher_results=args.teacher_results,
                noop_path=args.noop_path,
                noop_work_package_hash=args.noop_work_package_hash,
                semantic_review_dir=args.semantic_review_dir,
                article_index=args.article_index,
                artifact_dir=args.artifact_dir,
                output_dir=args.output_dir,
            )
            print(f"QUERY_SFT_V2_RETRIEVAL_FINALIZE_OK selections={result['records']['query_inputs']}")
        else:
            result = run_query_sft_v2_retrieval_batch(
                batch_index=args.batch_index,
                batch_size=args.batch_size,
                input_work_package_dir=args.input_work_package_dir,
                teacher_results=args.teacher_results,
                noop_path=args.noop_path,
                noop_work_package_hash=args.noop_work_package_hash,
                semantic_review_dir=args.semantic_review_dir,
                article_index=args.article_index,
                artifact_dir=args.artifact_dir,
                tokenizer_path=args.tokenizer_path,
                output_dir=args.output_dir,
                device=args.device,
            )
            print(f"QUERY_SFT_V2_RETRIEVAL_BATCH_OK queries={result['records']['query_inputs']}")
    except QuerySftV2RetrievalEvaluationError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
