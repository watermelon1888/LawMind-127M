"""在冻结的当前 Retrieval 链路中比较全量 Query-SFT 候选。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from rag.answering import AnswerPromptTokenCounter, EvidencePackager
from rag.eval.retrieval_baseline import CONTEXT_LIMIT, MAX_OUTPUT_TOKENS, _file_identity, _load_tokenizer, _tokenizer_identity
from rag.eval.retrieval_chain_evaluation import EvaluationCase, EvaluationMode, build_query_evaluation_plan, evaluate_retrieval_case
from rag.knowledge import ArticleRepository
from rag.retrieval import SemanticRetrievalConfig, load_semantic_retriever
from rag.retrieval.loader import DEFAULT_EMBEDDING_MODEL, DEFAULT_RERANKER_MODEL, _resolve_device

try:
    from .audit_query_sft_full_candidate_texts import REPORT_FILENAME as TEXT_AUDIT_FILENAME
    from .audit_query_sft_full_teacher_candidates import DEFAULT_OUTPUT_DIR as DEFAULT_TEACHER_WORK_PACKAGE_DIR, RESULT_FILENAME, _verify_work_package
    from .finalize_query_sft_full_candidate_semantic_review import LEDGER_FILENAME
    from .prepare_query_sft_full_candidate_semantic_review import DEFAULT_INPUT_WORK_PACKAGE_DIR
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset.audit_query_sft_full_candidate_texts import REPORT_FILENAME as TEXT_AUDIT_FILENAME
    from dataset.audit_query_sft_full_teacher_candidates import DEFAULT_OUTPUT_DIR as DEFAULT_TEACHER_WORK_PACKAGE_DIR, RESULT_FILENAME, _verify_work_package
    from dataset.finalize_query_sft_full_candidate_semantic_review import LEDGER_FILENAME
    from dataset.prepare_query_sft_full_candidate_semantic_review import DEFAULT_INPUT_WORK_PACKAGE_DIR


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_ROOT = PROJECT_ROOT / "rag"
DEFAULT_SEMANTIC_REVIEW_DIR = DEFAULT_TEACHER_WORK_PACKAGE_DIR / "query-sft-v1-candidate-semantic-review-work-package"
DEFAULT_TEXT_AUDIT = DEFAULT_SEMANTIC_REVIEW_DIR / TEXT_AUDIT_FILENAME
DEFAULT_ARTICLE_INDEX = RAG_ROOT / "chunk" / "article_index.jsonl"
DEFAULT_ARTIFACT_DIR = RAG_ROOT / "retrieval" / "artifacts"
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "minimind" / "model"
DEFAULT_OUTPUT_DIR = DEFAULT_TEACHER_WORK_PACKAGE_DIR / "query-sft-v1-retrieval-evaluation"
INPUT_FILENAME = "query-sft-v1-input-candidate-r1.jsonl"
NOOP_FILENAME = "query-sft-v1-deterministic-noop-candidates.jsonl"
RESULTS_FILENAME = "query-sft-v1-retrieval-records.jsonl"
SUMMARY_FILENAME = "query-sft-v1-retrieval-summary.json"
MANIFEST_FILENAME = "query-sft-v1-retrieval-manifest.json"
HASH_FILENAME = "query-sft-v1-retrieval.sha256"
PARTS_DIRECTORY_NAME = "parts"
FINAL_DIRECTORY_NAME = "final"
PART_RECORDS_FILENAME = "query-sft-v1-retrieval-part-records.jsonl"
PART_MANIFEST_FILENAME = "query-sft-v1-retrieval-part-manifest.json"
DEFAULT_BATCH_SIZE = 25
INPUT_FIELDS = {"work_id", "source_id", "authoring_type", "query_original"}
REFERENCE_FIELDS = {"work_id", "source_id", "source_query", "required_chunk_ids"}
RESULT_FIELDS = {"candidate_id", "work_id", "raw_output"}
LEDGER_FIELDS = {"candidate_id", "review_decision", "reason"}
_ARTIFACT_FILENAMES = ("law_dense.faiss", "law_dense_meta.json", "law_sparse.pkl")


class QuerySftFullRetrievalEvaluationError(RuntimeError):
    """表示全量 Query-SFT 冻结 Retrieval 比较的输入或运行时状态无效。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_hash(path: Path, label: str) -> Path:
    hash_path = path.with_suffix(".sha256")
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftFullRetrievalEvaluationError(f"无法读取{label} SHA-256: {hash_path}") from error
    if lines != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftFullRetrievalEvaluationError(f"{label} SHA-256 无效")
    return hash_path


def _score(record) -> tuple[int, float, float]:
    metrics = record.reranked_top5_metrics
    return (int(metrics.complete_hit), metrics.required_gt_coverage, metrics.reciprocal_rank)


def _publish(output_dir: Path, payloads: list[tuple[str, str]]) -> None:
    if output_dir.exists():
        raise QuerySftFullRetrievalEvaluationError(f"输出目录必须不存在: {output_dir}")
    try:
        output_dir.mkdir(parents=True)
        for name, payload in payloads:
            (output_dir / name).write_text(payload, encoding="utf-8", newline="\n")
        (output_dir / HASH_FILENAME).write_text(
            "".join(f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()}  {name}\n" for name, payload in payloads),
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, UnicodeError) as error:
        raise QuerySftFullRetrievalEvaluationError("无法发布 Retrieval 比较结果") from error


def _verify_published_payloads(package_dir: Path, filenames: tuple[str, ...]) -> None:
    """校验一个已发布分片或最终包中的全部内容哈希。"""
    hash_path = package_dir / HASH_FILENAME
    try:
        expected = {
            name: digest
            for line in hash_path.read_text(encoding="utf-8").splitlines()
            for digest, name in [line.split("  ", 1)]
        }
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise QuerySftFullRetrievalEvaluationError("分片 SHA-256 清单无效") from error
    if set(expected) != set(filenames):
        raise QuerySftFullRetrievalEvaluationError("分片 SHA-256 清单文件集合无效")
    for name in filenames:
        path = package_dir / name
        if not path.is_file() or expected[name] != _sha256_file(path):
            raise QuerySftFullRetrievalEvaluationError("分片 SHA-256 校验失败")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftFullRetrievalEvaluationError(f"无法读取{label}") from error
    if not isinstance(value, dict):
        raise QuerySftFullRetrievalEvaluationError(f"{label}必须是 JSON 对象")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                if not line.strip():
                    raise QuerySftFullRetrievalEvaluationError(f"{label}不允许空行: {number}")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise QuerySftFullRetrievalEvaluationError(f"{label}第 {number} 条必须是对象")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftFullRetrievalEvaluationError):
            raise
        raise QuerySftFullRetrievalEvaluationError(f"无法读取{label}") from error
    return rows


def _load_cases(input_directory: Path) -> tuple[tuple[EvaluationCase, ...], dict[str, str], dict[str, object]]:
    input_path = input_directory / INPUT_FILENAME
    input_hash = _verify_hash(input_path, "正式输入候选")
    inputs = {}
    for row in _load_jsonl(input_path, "正式输入候选"):
        work_id = row.get("work_id")
        if set(row) != INPUT_FIELDS or not isinstance(work_id, str) or work_id in inputs:
            raise QuerySftFullRetrievalEvaluationError("正式输入候选映射无效")
        inputs[work_id] = row
    references = {}
    for batch in range(1, 9):
        reference_path = input_directory / "audit-reference" / f"batch-{batch:02d}.jsonl"
        for row in _load_jsonl(reference_path, f"GT 审核引用第 {batch} 批"):
            work_id = row.get("work_id")
            required = row.get("required_chunk_ids")
            if (
                set(row) != REFERENCE_FIELDS
                or work_id not in inputs
                or work_id in references
                or row.get("source_id") != inputs[work_id]["source_id"]
                or not isinstance(required, list)
                or not required
                or any(not isinstance(value, str) or not value for value in required)
            ):
                raise QuerySftFullRetrievalEvaluationError("GT 审核引用映射无效")
            references[work_id] = row
    if len(inputs) != 574 or set(inputs) != set(references):
        raise QuerySftFullRetrievalEvaluationError("正式输入与 GT 审核引用未闭合为 574 条")
    cases = tuple(
        EvaluationCase(query_id=work_id, query_original=inputs[work_id]["query_original"], required_chunk_ids=tuple(references[work_id]["required_chunk_ids"]))
        for work_id in inputs
    )
    types = {work_id: row["authoring_type"] for work_id, row in inputs.items()}
    return cases, types, {"input": input_path, "input_hash": input_hash, "audit_reference_dir": input_directory / "audit-reference"}


def _load_variants(teacher_directory: Path, review_directory: Path) -> tuple[dict[str, str], dict[str, tuple[str, str]], dict[str, Path]]:
    result_path = teacher_directory / RESULT_FILENAME
    noop_path = teacher_directory / NOOP_FILENAME
    ledger_path = review_directory / LEDGER_FILENAME
    hashes = {
        "results": _verify_hash(result_path, "教师候选结果"),
        "noop_work_package": _verify_work_package(teacher_directory),
        "ledger": _verify_hash(ledger_path, "候选语义审核账本"),
    }
    decisions = {}
    for row in _load_jsonl(ledger_path, "候选语义审核账本"):
        candidate_id = row.get("candidate_id")
        if set(row) != LEDGER_FIELDS or not isinstance(candidate_id, str) or candidate_id in decisions or row.get("review_decision") not in {"approved", "rejected"}:
            raise QuerySftFullRetrievalEvaluationError("候选语义审核账本映射无效")
        decisions[candidate_id] = row["review_decision"]
    candidates = {}
    for row in _load_jsonl(result_path, "教师候选结果"):
        candidate_id = row.get("candidate_id")
        work_id = row.get("work_id")
        if set(row) != RESULT_FIELDS or not isinstance(candidate_id, str) or candidate_id in candidates or not isinstance(work_id, str) or candidate_id not in decisions or not isinstance(row.get("raw_output"), str):
            raise QuerySftFullRetrievalEvaluationError("教师候选结果映射无效")
        if decisions[candidate_id] == "approved":
            candidates[candidate_id] = (work_id, row["raw_output"])
    noops = {}
    for row in _load_jsonl(noop_path, "确定性 no-op"):
        candidate_id = row.get("candidate_id")
        work_id = row.get("work_id")
        if set(row) != RESULT_FIELDS or not isinstance(candidate_id, str) or not isinstance(work_id, str) or work_id in noops or not isinstance(row.get("raw_output"), str):
            raise QuerySftFullRetrievalEvaluationError("确定性 no-op 映射无效")
        noops[work_id] = row["raw_output"]
    if len(decisions) != 1722 or len(candidates) != 1720 or len(noops) != 574:
        raise QuerySftFullRetrievalEvaluationError("候选批准数或确定性 no-op 数量无效")
    return noops, candidates, {
        "results": result_path,
        "noop": noop_path,
        "ledger": ledger_path,
        "results_hash": hashes["results"],
        "noop_work_package_hash": hashes["noop_work_package"],
        "ledger_hash": hashes["ledger"],
    }


def _evaluate(cases, noops, candidates, retriever, packager) -> tuple[list[dict[str, object]], dict[str, Any]]:
    rows = []
    baseline_by_work = {}
    noop_by_work = {}
    candidate_by_work: dict[str, list[tuple[str, Any]]] = {case.query_id: [] for case in cases}
    case_by_work = {case.query_id: case for case in cases}

    def append(kind: str, variant_id: str, case: EvaluationCase, raw_output: str | None, mode: EvaluationMode):
        plan = build_query_evaluation_plan(case.query_original, mode, raw_enhancement=raw_output) if mode is EvaluationMode.FULL_ENHANCEMENT else build_query_evaluation_plan(case.query_original, mode)
        record = evaluate_retrieval_case(case, plan, retriever=retriever, packager=packager)
        rows.append({"variant_kind": kind, "variant_id": variant_id, "work_id": case.query_id, "record": record.to_dict()})
        return record

    for index, case in enumerate(cases, 1):
        baseline_by_work[case.query_id] = append("baseline", f"{case.query_id}/baseline", case, None, EvaluationMode.BASELINE_ORIGINAL)
        noop_by_work[case.query_id] = append("noop", f"{case.query_id}/noop", case, noops[case.query_id], EvaluationMode.NOOP_APPLIED)
        if index % 5 == 0 or index == len(cases):
            print(f"QUERY_SFT_FULL_BASELINE_PROGRESS completed={index}/{len(cases)}", flush=True)
    for index, (candidate_id, (work_id, raw_output)) in enumerate(candidates.items(), 1):
        record = append("teacher_candidate", candidate_id, case_by_work[work_id], raw_output, EvaluationMode.FULL_ENHANCEMENT)
        candidate_by_work[work_id].append((candidate_id, record))
        if index % 25 == 0 or index == len(candidates):
            print(f"QUERY_SFT_FULL_CANDIDATE_PROGRESS completed={index}/{len(candidates)}", flush=True)
    return rows, {"baseline": baseline_by_work, "noop": noop_by_work, "candidates": candidate_by_work}


def _summary(evaluated: dict[str, Any], authoring_types: dict[str, str]) -> dict[str, object]:
    selections = []
    counters = {"teacher_candidate": 0, "noop": 0}
    for work_id, baseline in evaluated["baseline"].items():
        reference = max(_score(baseline), _score(evaluated["noop"][work_id]))
        qualified = [(candidate_id, record) for candidate_id, record in evaluated["candidates"][work_id] if _score(record) >= reference and (_score(record) > reference or authoring_types[work_id] in {"colloquial", "ellipsis"})]
        if qualified:
            winner_id, winner = max(qualified, key=lambda value: (_score(value[1]), value[0]))
            kind = "teacher_candidate"
        else:
            winner_id, winner, kind = f"{work_id}/noop", evaluated["noop"][work_id], "noop"
        counters[kind] += 1
        selections.append({"work_id": work_id, "authoring_type": authoring_types[work_id], "selected_variant_id": winner_id, "selection": kind, "baseline_top5": _score(baseline), "noop_top5": _score(evaluated["noop"][work_id]), "selected_top5": _score(winner), "approved_teacher_candidates_compared": len(evaluated["candidates"][work_id])})
    return {"records": {"query_inputs": len(selections), "baseline_variants": len(evaluated["baseline"]), "noop_variants": len(evaluated["noop"]), "approved_teacher_candidate_variants": sum(len(rows) for rows in evaluated["candidates"].values()), "total_retrieval_evaluations": len(evaluated["baseline"]) + len(evaluated["noop"]) + sum(len(rows) for rows in evaluated["candidates"].values()), "selected_by_kind": counters}, "selection_policy": {"primary_metrics": ["complete_hit@5", "required_gt_coverage@5", "mrr@5"], "candidate_must_not_degrade_baseline_or_noop": True, "strict_improvement_required_except_authoring_types": ["colloquial", "ellipsis"], "ties_allowed_for_exceptions": True}, "selections": selections}


def _prepare_inputs(*, input_work_package_dir: Path, teacher_work_package_dir: Path, semantic_review_dir: Path, text_audit_path: Path, article_index: Path, artifact_dir: Path):
    """校验冻结输入，并返回分片和最终聚合共同使用的不可变上下文。"""
    input_directory = Path(input_work_package_dir).resolve()
    teacher_directory = Path(teacher_work_package_dir).resolve()
    review_directory = Path(semantic_review_dir).resolve()
    text_audit = Path(text_audit_path).resolve()
    article_index = Path(article_index).resolve()
    artifact_dir = Path(artifact_dir).resolve()
    text_audit_hash = _verify_hash(text_audit, "候选文本审计")
    text_audit_value = _load_json(text_audit, "候选文本审计")
    if (
        text_audit_value.get("pipeline") != "query_sft_full_candidate_text_audit_v1"
        or text_audit_value.get("complete") is not True
        or text_audit_value.get("records", {}).get("approved") != 1720
        or text_audit_value.get("records", {}).get("evaluation_text_overlap") != 0
    ):
        raise QuerySftFullRetrievalEvaluationError("候选文本审计状态无效")
    cases, authoring_types, case_inputs = _load_cases(input_directory)
    noops, candidates, variant_inputs = _load_variants(teacher_directory, review_directory)
    if set(noops) != set(authoring_types) or set(work_id for work_id, _ in candidates.values()) - set(authoring_types):
        raise QuerySftFullRetrievalEvaluationError("候选与正式输入 work_id 未闭合")
    frozen_inputs = {
        "article_index": _file_identity(article_index),
        "retrieval_artifacts": {name: _file_identity(artifact_dir / name) for name in _ARTIFACT_FILENAMES},
        "candidate_text_audit": {**_file_identity(text_audit), "hash_manifest": _file_identity(text_audit_hash)},
        "input_candidate": {**_file_identity(case_inputs["input"]), "hash_manifest": _file_identity(case_inputs["input_hash"])},
        "teacher_results": {**_file_identity(variant_inputs["results"]), "hash_manifest": _file_identity(variant_inputs["results_hash"])},
        "deterministic_noop": {**_file_identity(variant_inputs["noop"]), "work_package_hash_manifest": _file_identity(variant_inputs["noop_work_package_hash"])},
        "semantic_ledger": {**_file_identity(variant_inputs["ledger"]), "hash_manifest": _file_identity(variant_inputs["ledger_hash"])},
    }
    return cases, authoring_types, noops, candidates, frozen_inputs


def _part_directory(output_dir: Path, batch_index: int) -> Path:
    return output_dir / PARTS_DIRECTORY_NAME / f"batch-{batch_index:03d}"


def _select_batch(cases: tuple[EvaluationCase, ...], *, batch_index: int, batch_size: int) -> tuple[EvaluationCase, ...]:
    if batch_index < 0 or batch_size <= 0:
        raise QuerySftFullRetrievalEvaluationError("batch_index 必须非负且 batch_size 必须为正数")
    start = batch_index * batch_size
    selected = cases[start : start + batch_size]
    if not selected:
        raise QuerySftFullRetrievalEvaluationError("batch_index 超出正式 Query-SFT 范围")
    return selected


def run_query_sft_full_retrieval_batch(
    *,
    batch_index: int,
    batch_size: int = DEFAULT_BATCH_SIZE,
    input_work_package_dir: Path = DEFAULT_INPUT_WORK_PACKAGE_DIR,
    teacher_work_package_dir: Path = DEFAULT_TEACHER_WORK_PACKAGE_DIR,
    semantic_review_dir: Path = DEFAULT_SEMANTIC_REVIEW_DIR,
    text_audit_path: Path = DEFAULT_TEXT_AUDIT,
    article_index: Path = DEFAULT_ARTICLE_INDEX,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    tokenizer_path: Path = DEFAULT_TOKENIZER_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device: str = "auto",
) -> dict[str, object]:
    """执行一个不可覆盖的 work_id 分片；同题的全部变体始终处于同一分片。"""
    output_directory = Path(output_dir).resolve()
    cases, _, noops, candidates, frozen_inputs = _prepare_inputs(
        input_work_package_dir=input_work_package_dir,
        teacher_work_package_dir=teacher_work_package_dir,
        semantic_review_dir=semantic_review_dir,
        text_audit_path=text_audit_path,
        article_index=article_index,
        artifact_dir=artifact_dir,
    )
    selected_cases = _select_batch(cases, batch_index=batch_index, batch_size=batch_size)
    selected_ids = {case.query_id for case in selected_cases}
    selected_candidates = {candidate_id: value for candidate_id, value in candidates.items() if value[0] in selected_ids}
    total_batches = math.ceil(len(cases) / batch_size)
    part_dir = _part_directory(output_directory, batch_index)
    if part_dir.exists():
        _verify_published_payloads(part_dir, (PART_RECORDS_FILENAME, PART_MANIFEST_FILENAME))
        manifest = _load_json(part_dir / PART_MANIFEST_FILENAME, "已发布 Retrieval 分片")
        if manifest.get("batch_index") != batch_index or manifest.get("batch_size") != batch_size or manifest.get("frozen_inputs") != frozen_inputs:
            raise QuerySftFullRetrievalEvaluationError("已发布 Retrieval 分片与当前冻结输入不一致")
        print(f"QUERY_SFT_FULL_RETRIEVAL_BATCH_REUSED index={batch_index}", flush=True)
        return manifest
    print(f"QUERY_SFT_FULL_RETRIEVAL_BATCH_PRECHECK_OK index={batch_index} queries={len(selected_cases)} candidates={len(selected_candidates)}", flush=True)
    tokenizer = _load_tokenizer(Path(tokenizer_path).resolve())
    packager = EvidencePackager(context_limit=CONTEXT_LIMIT, max_output_tokens=MAX_OUTPUT_TOKENS, count_prompt_tokens=AnswerPromptTokenCounter(tokenizer))
    actual_device = _resolve_device(device)
    repository = ArticleRepository.from_jsonl(Path(article_index).resolve())
    retriever = load_semantic_retriever(repository=repository, artifact_dir=Path(artifact_dir).resolve(), embedding_model=DEFAULT_EMBEDDING_MODEL, reranker_model=DEFAULT_RERANKER_MODEL, device=actual_device, config=SemanticRetrievalConfig())
    rows, _ = _evaluate(selected_cases, noops, selected_candidates, retriever, packager)
    after = {"article_index": _file_identity(Path(article_index).resolve()), "retrieval_artifacts": {name: _file_identity(Path(artifact_dir).resolve() / name) for name in _ARTIFACT_FILENAMES}}
    if after != {"article_index": frozen_inputs["article_index"], "retrieval_artifacts": frozen_inputs["retrieval_artifacts"]}:
        raise QuerySftFullRetrievalEvaluationError("Retrieval 资产在分片运行期间发生变化")
    manifest = {
        "pipeline": "query_sft_full_retrieval_evaluation_part_v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "batch_index": batch_index,
        "batch_size": batch_size,
        "total_batches": total_batches,
        "work_ids": [case.query_id for case in selected_cases],
        "records": {"query_inputs": len(selected_cases), "approved_teacher_candidate_variants": len(selected_candidates), "total_retrieval_evaluations": len(rows)},
        "frozen_inputs": frozen_inputs,
        "retrieval_identity": {"embedding_model": DEFAULT_EMBEDDING_MODEL, "reranker_model": DEFAULT_RERANKER_MODEL, "device": actual_device, "config": asdict(SemanticRetrievalConfig()), "tokenizer": _tokenizer_identity(tokenizer, Path(tokenizer_path).resolve())},
        "complete": True,
    }
    records_payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _publish(part_dir, [(PART_RECORDS_FILENAME, records_payload), (PART_MANIFEST_FILENAME, manifest_payload)])
    print(f"QUERY_SFT_FULL_RETRIEVAL_BATCH_OK index={batch_index} evaluations={len(rows)}", flush=True)
    return manifest


def _score_dict(record: dict[str, object]) -> tuple[int, float, float]:
    try:
        metrics = record["retrieval"]["reranked_top5_metrics"]
        return (
            int(bool(metrics["complete_hit"])),
            float(metrics["required_gt_coverage"]),
            float(metrics["reciprocal_rank"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise QuerySftFullRetrievalEvaluationError("Retrieval 分片记录指标无效") from error


def _summarize_published_rows(rows: list[dict[str, object]], authoring_types: dict[str, str]) -> dict[str, object]:
    """由全部已验证分片重建唯一选择，不重新运行检索。"""
    baseline, noops = {}, {}
    candidates: dict[str, list[tuple[str, dict[str, object]]]] = {work_id: [] for work_id in authoring_types}
    for row in rows:
        try:
            kind, variant_id, work_id, record = row["variant_kind"], row["variant_id"], row["work_id"], row["record"]
        except (KeyError, TypeError) as error:
            raise QuerySftFullRetrievalEvaluationError("Retrieval 分片记录结构无效") from error
        if work_id not in authoring_types or not isinstance(variant_id, str) or not isinstance(record, dict):
            raise QuerySftFullRetrievalEvaluationError("Retrieval 分片记录 work_id 无效")
        if kind == "baseline":
            if work_id in baseline:
                raise QuerySftFullRetrievalEvaluationError("baseline 分片记录重复")
            baseline[work_id] = record
        elif kind == "noop":
            if work_id in noops:
                raise QuerySftFullRetrievalEvaluationError("no-op 分片记录重复")
            noops[work_id] = record
        elif kind == "teacher_candidate":
            candidates[work_id].append((variant_id, record))
        else:
            raise QuerySftFullRetrievalEvaluationError("Retrieval 分片记录变体类型无效")
    if set(baseline) != set(authoring_types) or set(noops) != set(authoring_types):
        raise QuerySftFullRetrievalEvaluationError("缺少 baseline 或 no-op 分片记录")
    selections, counters = [], {"teacher_candidate": 0, "noop": 0}
    for work_id in authoring_types:
        reference = max(_score_dict(baseline[work_id]), _score_dict(noops[work_id]))
        qualified = [
            (candidate_id, record)
            for candidate_id, record in candidates[work_id]
            if _score_dict(record) >= reference
            and (_score_dict(record) > reference or authoring_types[work_id] in {"colloquial", "ellipsis"})
        ]
        if qualified:
            winner_id, winner = max(qualified, key=lambda value: (_score_dict(value[1]), value[0]))
            selection = "teacher_candidate"
        else:
            winner_id, winner, selection = f"{work_id}/noop", noops[work_id], "noop"
        counters[selection] += 1
        selections.append({
            "work_id": work_id,
            "authoring_type": authoring_types[work_id],
            "selected_variant_id": winner_id,
            "selection": selection,
            "baseline_top5": _score_dict(baseline[work_id]),
            "noop_top5": _score_dict(noops[work_id]),
            "selected_top5": _score_dict(winner),
            "approved_teacher_candidates_compared": len(candidates[work_id]),
        })
    return {
        "records": {
            "query_inputs": len(selections),
            "baseline_variants": len(baseline),
            "noop_variants": len(noops),
            "approved_teacher_candidate_variants": sum(len(values) for values in candidates.values()),
            "total_retrieval_evaluations": len(rows),
            "selected_by_kind": counters,
        },
        "selection_policy": {
            "primary_metrics": ["complete_hit@5", "required_gt_coverage@5", "mrr@5"],
            "candidate_must_not_degrade_baseline_or_noop": True,
            "strict_improvement_required_except_authoring_types": ["colloquial", "ellipsis"],
            "ties_allowed_for_exceptions": True,
        },
        "selections": selections,
    }


def finalize_query_sft_full_retrieval_evaluation(
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    input_work_package_dir: Path = DEFAULT_INPUT_WORK_PACKAGE_DIR,
    teacher_work_package_dir: Path = DEFAULT_TEACHER_WORK_PACKAGE_DIR,
    semantic_review_dir: Path = DEFAULT_SEMANTIC_REVIEW_DIR,
    text_audit_path: Path = DEFAULT_TEXT_AUDIT,
    article_index: Path = DEFAULT_ARTICLE_INDEX,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, object]:
    """只在所有固定分片齐全且身份一致时，聚合为最终 Retrieval 结果。"""
    output_directory = Path(output_dir).resolve()
    cases, authoring_types, _, _, frozen_inputs = _prepare_inputs(
        input_work_package_dir=input_work_package_dir,
        teacher_work_package_dir=teacher_work_package_dir,
        semantic_review_dir=semantic_review_dir,
        text_audit_path=text_audit_path,
        article_index=article_index,
        artifact_dir=artifact_dir,
    )
    total_batches = math.ceil(len(cases) / batch_size)
    rows, part_identities = [], []
    expected_work_ids = set()
    retrieval_identity = None
    for batch_index in range(total_batches):
        part_dir = _part_directory(output_directory, batch_index)
        _verify_published_payloads(part_dir, (PART_RECORDS_FILENAME, PART_MANIFEST_FILENAME))
        part = _load_json(part_dir / PART_MANIFEST_FILENAME, "Retrieval 分片 manifest")
        expected_batch = _select_batch(cases, batch_index=batch_index, batch_size=batch_size)
        if (
            part.get("pipeline") != "query_sft_full_retrieval_evaluation_part_v1"
            or part.get("batch_index") != batch_index
            or part.get("batch_size") != batch_size
            or part.get("total_batches") != total_batches
            or part.get("work_ids") != [case.query_id for case in expected_batch]
            or part.get("frozen_inputs") != frozen_inputs
            or part.get("complete") is not True
        ):
            raise QuerySftFullRetrievalEvaluationError("Retrieval 分片 manifest 与冻结计划不一致")
        if retrieval_identity is None:
            retrieval_identity = part.get("retrieval_identity")
        elif part.get("retrieval_identity") != retrieval_identity:
            raise QuerySftFullRetrievalEvaluationError("Retrieval 分片使用的检索器身份不一致")
        expected_work_ids.update(part["work_ids"])
        part_identities.append({"batch_index": batch_index, "records": _file_identity(part_dir / PART_RECORDS_FILENAME), "manifest": _file_identity(part_dir / PART_MANIFEST_FILENAME), "hash_manifest": _file_identity(part_dir / HASH_FILENAME)})
        rows.extend(_load_jsonl(part_dir / PART_RECORDS_FILENAME, "Retrieval 分片记录"))
    if expected_work_ids != {case.query_id for case in cases}:
        raise QuerySftFullRetrievalEvaluationError("Retrieval 分片 work_id 未完整覆盖")
    summary = _summarize_published_rows(rows, authoring_types)
    expected_records = summary["records"]
    if (
        expected_records["query_inputs"] != 574
        or expected_records["baseline_variants"] != 574
        or expected_records["noop_variants"] != 574
        or expected_records["approved_teacher_candidate_variants"] != 1720
        or expected_records["total_retrieval_evaluations"] != 2868
        or sum(expected_records["selected_by_kind"].values()) != 574
    ):
        raise QuerySftFullRetrievalEvaluationError("Retrieval 分片聚合记录数未闭合")
    manifest = {
        "pipeline": "query_sft_full_retrieval_evaluation_v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs_frozen_before_run": frozen_inputs,
        "partitions": {"batch_size": batch_size, "total_batches": total_batches, "parts": part_identities},
        "retrieval_identity": retrieval_identity,
        "records": summary["records"],
        "selection_policy": summary["selection_policy"],
        "validation": {"input_identities_frozen_before_run": True, "retrieval_assets_unchanged_after_run": True, "baseline_noop_and_all_approved_candidates_compared": True, "query_sft_training_ready": False},
        "complete": True,
    }
    final_directory = output_directory / FINAL_DIRECTORY_NAME
    payloads = [
        (RESULTS_FILENAME, "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)),
        (SUMMARY_FILENAME, json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
        (MANIFEST_FILENAME, json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
    ]
    _publish(final_directory, payloads)
    print(f"QUERY_SFT_FULL_RETRIEVAL_FINALIZE_OK evaluations={summary['records']['total_retrieval_evaluations']}", flush=True)
    return manifest


def run_query_sft_full_retrieval_evaluation(*, input_work_package_dir: Path = DEFAULT_INPUT_WORK_PACKAGE_DIR, teacher_work_package_dir: Path = DEFAULT_TEACHER_WORK_PACKAGE_DIR, semantic_review_dir: Path = DEFAULT_SEMANTIC_REVIEW_DIR, text_audit_path: Path = DEFAULT_TEXT_AUDIT, article_index: Path = DEFAULT_ARTICLE_INDEX, artifact_dir: Path = DEFAULT_ARTIFACT_DIR, tokenizer_path: Path = DEFAULT_TOKENIZER_PATH, output_dir: Path = DEFAULT_OUTPUT_DIR, device: str = "auto") -> dict[str, object]:
    """执行不可覆盖的全量冻结 Retrieval 比较并发布其身份清单。"""

    input_directory, teacher_directory, review_directory = Path(input_work_package_dir).resolve(), Path(teacher_work_package_dir).resolve(), Path(semantic_review_dir).resolve()
    text_audit, article_index, artifact_dir, tokenizer_path, output_directory = Path(text_audit_path).resolve(), Path(article_index).resolve(), Path(artifact_dir).resolve(), Path(tokenizer_path).resolve(), Path(output_dir).resolve()
    text_audit_hash = _verify_hash(text_audit, "候选文本审计")
    text_audit_value = _load_json(text_audit, "候选文本审计")
    if text_audit_value.get("pipeline") != "query_sft_full_candidate_text_audit_v1" or text_audit_value.get("complete") is not True or text_audit_value.get("records", {}).get("approved") != 1720 or text_audit_value.get("records", {}).get("evaluation_text_overlap") != 0:
        raise QuerySftFullRetrievalEvaluationError("候选文本审计状态无效")
    cases, authoring_types, case_inputs = _load_cases(input_directory)
    noops, candidates, variant_inputs = _load_variants(teacher_directory, review_directory)
    if set(noops) != set(authoring_types) or set(work_id for work_id, _ in candidates.values()) - set(authoring_types):
        raise QuerySftFullRetrievalEvaluationError("候选与正式输入 work_id 未闭合")
    config = SemanticRetrievalConfig()
    frozen_inputs = {"article_index": _file_identity(article_index), "retrieval_artifacts": {name: _file_identity(artifact_dir / name) for name in _ARTIFACT_FILENAMES}, "candidate_text_audit": {**_file_identity(text_audit), "hash_manifest": _file_identity(text_audit_hash)}, "input_candidate": {**_file_identity(case_inputs["input"]), "hash_manifest": _file_identity(case_inputs["input_hash"])}, "teacher_results": {**_file_identity(variant_inputs["results"]), "hash_manifest": _file_identity(variant_inputs["results_hash"])}, "deterministic_noop": {**_file_identity(variant_inputs["noop"]), "work_package_hash_manifest": _file_identity(variant_inputs["noop_work_package_hash"])}, "semantic_ledger": {**_file_identity(variant_inputs["ledger"]), "hash_manifest": _file_identity(variant_inputs["ledger_hash"])}}
    print(
        f"QUERY_SFT_FULL_RETRIEVAL_PRECHECK_OK queries={len(cases)} candidates={len(candidates)}",
        flush=True,
    )
    tokenizer = _load_tokenizer(tokenizer_path)
    packager = EvidencePackager(context_limit=CONTEXT_LIMIT, max_output_tokens=MAX_OUTPUT_TOKENS, count_prompt_tokens=AnswerPromptTokenCounter(tokenizer))
    actual_device = _resolve_device(device)
    print(f"QUERY_SFT_FULL_RETRIEVER_LOADING device={actual_device}", flush=True)
    repository = ArticleRepository.from_jsonl(article_index)
    retriever = load_semantic_retriever(repository=repository, artifact_dir=artifact_dir, embedding_model=DEFAULT_EMBEDDING_MODEL, reranker_model=DEFAULT_RERANKER_MODEL, device=actual_device, config=config)
    print("QUERY_SFT_FULL_RETRIEVER_READY", flush=True)
    rows, evaluated = _evaluate(cases, noops, candidates, retriever, packager)
    summary = _summary(evaluated, authoring_types)
    after = {"article_index": _file_identity(article_index), "retrieval_artifacts": {name: _file_identity(artifact_dir / name) for name in _ARTIFACT_FILENAMES}}
    if after != {"article_index": frozen_inputs["article_index"], "retrieval_artifacts": frozen_inputs["retrieval_artifacts"]}:
        raise QuerySftFullRetrievalEvaluationError("Retrieval 资产在运行期间发生变化")
    manifest = {"pipeline": "query_sft_full_retrieval_evaluation_v1", "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"), "inputs_frozen_before_run": frozen_inputs, "retrieval_identity": {"embedding_model": DEFAULT_EMBEDDING_MODEL, "reranker_model": DEFAULT_RERANKER_MODEL, "device": actual_device, "config": asdict(config), "query_compiler": "rag.query.enhancement.compile_retrieval_queries", "multi_query_fusion": "two_level_equal_weight_rrf", "reranker_query": "query_original", "tokenizer": _tokenizer_identity(tokenizer, tokenizer_path)}, "records": summary["records"], "selection_policy": summary["selection_policy"], "validation": {"input_identities_frozen_before_run": True, "retrieval_assets_unchanged_after_run": True, "baseline_noop_and_all_approved_candidates_compared": True, "query_sft_training_ready": False}, "complete": True}
    payloads = [(RESULTS_FILENAME, "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)), (SUMMARY_FILENAME, json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"), (MANIFEST_FILENAME, json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")]
    _publish(output_directory, payloads)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-work-package-dir", type=Path, default=DEFAULT_INPUT_WORK_PACKAGE_DIR)
    parser.add_argument("--teacher-work-package-dir", type=Path, default=DEFAULT_TEACHER_WORK_PACKAGE_DIR)
    parser.add_argument("--semantic-review-dir", type=Path, default=DEFAULT_SEMANTIC_REVIEW_DIR)
    parser.add_argument("--text-audit", type=Path, default=DEFAULT_TEXT_AUDIT)
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
    print("QUERY_SFT_FULL_RETRIEVAL_ENTRYPOINT_READY", flush=True)
    try:
        common = {
            "batch_size": args.batch_size,
            "input_work_package_dir": args.input_work_package_dir,
            "teacher_work_package_dir": args.teacher_work_package_dir,
            "semantic_review_dir": args.semantic_review_dir,
            "text_audit_path": args.text_audit,
            "article_index": args.article_index,
            "artifact_dir": args.artifact_dir,
            "output_dir": args.output_dir,
        }
        if args.finalize:
            manifest = finalize_query_sft_full_retrieval_evaluation(**common)
        else:
            manifest = run_query_sft_full_retrieval_batch(
                **common,
                batch_index=args.batch_index,
                tokenizer_path=args.tokenizer_path,
                device=args.device,
            )
    except (OSError, ValueError, TypeError, KeyError, QuerySftFullRetrievalEvaluationError) as error:
        parser.error(str(error))
    print(f"QUERY_SFT_FULL_RETRIEVAL_EVALUATION_OK evaluations={manifest['records']['total_retrieval_evaluations']}", flush=True)


if __name__ == "__main__":
    main()
