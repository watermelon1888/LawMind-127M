"""在冻结的当前 Retrieval 链路上比较 Query-SFT pilot 候选。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
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

try:
    from .prepare_query_sft_pilot import (
        AUDIT_REFERENCE_FILENAME,
        DEFAULT_OUTPUT_DIR as DEFAULT_INPUT_WORK_PACKAGE_DIR,
    )
    from .prepare_query_sft_pilot_teacher_candidates import (
        DEFAULT_OUTPUT_DIR as DEFAULT_TEACHER_WORK_PACKAGE_DIR,
        NOOP_FILENAME,
    )
    from .prepare_query_sft_pilot_candidate_semantic_review import LEDGER_FILENAME
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset.prepare_query_sft_pilot import (
        AUDIT_REFERENCE_FILENAME,
        DEFAULT_OUTPUT_DIR as DEFAULT_INPUT_WORK_PACKAGE_DIR,
    )
    from dataset.prepare_query_sft_pilot_teacher_candidates import (
        DEFAULT_OUTPUT_DIR as DEFAULT_TEACHER_WORK_PACKAGE_DIR,
        NOOP_FILENAME,
    )
    from dataset.prepare_query_sft_pilot_candidate_semantic_review import (
        LEDGER_FILENAME,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_ROOT = PROJECT_ROOT / "rag"
DEFAULT_ARTICLE_INDEX = RAG_ROOT / "chunk" / "article_index.jsonl"
DEFAULT_ARTIFACT_DIR = RAG_ROOT / "retrieval" / "artifacts"
DEFAULT_TOKENIZER_PATH = PROJECT_ROOT / "minimind" / "model"
DEFAULT_RESULTS = (
    DEFAULT_TEACHER_WORK_PACKAGE_DIR
    / "query-sft-pilot-v1-teacher-candidate-results.jsonl"
)
DEFAULT_SEMANTIC_REVIEW_DIR = (
    DEFAULT_TEACHER_WORK_PACKAGE_DIR / "query-sft-pilot-v1-candidate-semantic-review"
)
DEFAULT_TEXT_AUDIT = (
    DEFAULT_TEACHER_WORK_PACKAGE_DIR / "query-sft-pilot-v1-candidate-text-audit.json"
)
DEFAULT_OUTPUT_DIR = (
    DEFAULT_TEACHER_WORK_PACKAGE_DIR / "query-sft-pilot-v1-retrieval-evaluation"
)
RESULTS_FILENAME = "query-sft-pilot-v1-retrieval-records.jsonl"
SUMMARY_FILENAME = "query-sft-pilot-v1-retrieval-summary.json"
MANIFEST_FILENAME = "query-sft-pilot-v1-retrieval-manifest.json"
HASH_FILENAME = "query-sft-pilot-v1-retrieval.sha256"

_REFERENCE_FIELDS = {
    "pilot_id",
    "source_id",
    "authoring_type",
    "coverage_domain",
    "source_query",
    "required_chunk_ids",
    "source_record_sha256",
}
_DRAFT_FIELDS = {"pilot_id", "source_id", "authoring_type", "query_original"}
_RESULT_FIELDS = {"candidate_id", "pilot_id", "raw_output"}
_LEDGER_FIELDS = {"candidate_id", "pilot_id", "review_decision", "reason"}
_NOOP_FIELDS = {"candidate_id", "pilot_id", "raw_output"}
_ARTIFACT_FILENAMES = ("law_dense.faiss", "law_dense_meta.json", "law_sparse.pkl")


class QuerySftPilotRetrievalEvaluationError(RuntimeError):
    """Query-SFT pilot Retrieval 比较的输入、运行时或发布状态无效。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    rows = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, start=1):
                if not line.strip():
                    raise QuerySftPilotRetrievalEvaluationError(
                        f"{description}不允许空行: {number}"
                    )
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise QuerySftPilotRetrievalEvaluationError(
                        f"{description}第 {number} 条必须是 object"
                    )
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, QuerySftPilotRetrievalEvaluationError):
            raise
        raise QuerySftPilotRetrievalEvaluationError(
            f"无法读取{description}: {path}"
        ) from error
    if not rows:
        raise QuerySftPilotRetrievalEvaluationError(f"{description}不能为空")
    return rows


def _verify_hash(path: Path, description: str) -> Path:
    hash_path = path.with_suffix(".sha256")
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftPilotRetrievalEvaluationError(
            f"无法读取{description} SHA-256: {hash_path}"
        ) from error
    if lines != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftPilotRetrievalEvaluationError(f"{description} SHA-256 无效")
    return hash_path


def _load_cases(
    *, input_work_package_dir: Path, draft_path: Path
) -> tuple[tuple[EvaluationCase, ...], dict[str, str], dict[str, str], dict[str, Path]]:
    reference_path = input_work_package_dir / AUDIT_REFERENCE_FILENAME
    draft_hash_path = _verify_hash(draft_path, "实际输入草稿")
    references = _load_jsonl(reference_path, "GT 审核引用")
    drafts = _load_jsonl(draft_path, "实际输入草稿")
    if len(references) != 40 or len(drafts) != 40:
        raise QuerySftPilotRetrievalEvaluationError("pilot 输入数量必须为 40")
    cases = []
    original_by_pilot = {}
    type_by_pilot = {}
    for position, (reference, draft) in enumerate(zip(references, drafts, strict=True), 1):
        if set(reference) != _REFERENCE_FIELDS or set(draft) != _DRAFT_FIELDS:
            raise QuerySftPilotRetrievalEvaluationError(f"第 {position} 条 pilot 字段无效")
        pilot_id = reference.get("pilot_id")
        if (
            not isinstance(pilot_id, str)
            or pilot_id != draft.get("pilot_id")
            or reference.get("source_id") != draft.get("source_id")
            or reference.get("authoring_type") != draft.get("authoring_type")
            or not isinstance(draft.get("query_original"), str)
            or not isinstance(reference.get("required_chunk_ids"), list)
            or not reference["required_chunk_ids"]
        ):
            raise QuerySftPilotRetrievalEvaluationError(f"第 {position} 条 pilot 映射无效")
        if pilot_id == "query_sft_pilot:0017":
            continue
        cases.append(
            EvaluationCase(
                query_id=pilot_id,
                query_original=draft["query_original"],
                required_chunk_ids=tuple(reference["required_chunk_ids"]),
            )
        )
        original_by_pilot[pilot_id] = draft["query_original"]
        type_by_pilot[pilot_id] = draft["authoring_type"]
    if len(cases) != 39 or len(original_by_pilot) != 39:
        raise QuerySftPilotRetrievalEvaluationError("必须只评估 39 条已通过输入审核的 pilot")
    return tuple(cases), original_by_pilot, type_by_pilot, {
        "audit_reference": reference_path,
        "draft": draft_path,
        "draft_hash": draft_hash_path,
    }


def _load_variants(
    *,
    result_path: Path,
    noop_path: Path,
    semantic_ledger_path: Path,
    allowed_pilots: set[str],
) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    results = _load_jsonl(result_path, "教师候选结果")
    ledger = _load_jsonl(semantic_ledger_path, "候选语义审核账本")
    decisions = {}
    for row in ledger:
        if set(row) != _LEDGER_FIELDS or not isinstance(row.get("candidate_id"), str):
            raise QuerySftPilotRetrievalEvaluationError("候选语义审核账本字段无效")
        if row["candidate_id"] in decisions:
            raise QuerySftPilotRetrievalEvaluationError("候选语义审核账本 candidate_id 重复")
        decisions[row["candidate_id"]] = row.get("review_decision")
    candidates = {}
    for row in results:
        if (
            set(row) != _RESULT_FIELDS
            or not isinstance(row.get("candidate_id"), str)
            or not isinstance(row.get("pilot_id"), str)
            or not isinstance(row.get("raw_output"), str)
            or row["candidate_id"] not in decisions
        ):
            raise QuerySftPilotRetrievalEvaluationError("教师候选结果字段或映射无效")
        if row["candidate_id"] in candidates:
            raise QuerySftPilotRetrievalEvaluationError("教师候选 candidate_id 重复")
        if row["pilot_id"] in allowed_pilots and decisions[row["candidate_id"]] == "approved":
            candidates[row["candidate_id"]] = (row["pilot_id"], row["raw_output"])
    if len(candidates) != 114:
        raise QuerySftPilotRetrievalEvaluationError("通过候选数量必须为 114")
    noops = {}
    for row in _load_jsonl(noop_path, "确定性 no-op"):
        if (
            set(row) != _NOOP_FIELDS
            or not isinstance(row.get("candidate_id"), str)
            or not isinstance(row.get("pilot_id"), str)
            or not isinstance(row.get("raw_output"), str)
            or row["pilot_id"] not in allowed_pilots
        ):
            raise QuerySftPilotRetrievalEvaluationError("确定性 no-op 字段或映射无效")
        noops[row["pilot_id"]] = row["raw_output"]
    if len(noops) != 39:
        raise QuerySftPilotRetrievalEvaluationError("确定性 no-op 数量必须为 39")
    return noops, candidates


def _score(record) -> tuple[int, float, float]:
    metrics = record.reranked_top5_metrics
    return (int(metrics.complete_hit), metrics.required_gt_coverage, metrics.reciprocal_rank)


def _evaluate(
    cases: tuple[EvaluationCase, ...],
    *,
    noops: dict[str, str],
    candidates: dict[str, tuple[str, str]],
    retriever,
    packager,
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    rows = []
    baseline_by_pilot = {}
    noop_by_pilot = {}
    candidates_by_pilot: dict[str, list[tuple[str, Any]]] = {case.query_id: [] for case in cases}
    started = time.perf_counter()

    def append(kind: str, variant_id: str, case: EvaluationCase, raw_output: str | None, mode):
        plan = (
            build_query_evaluation_plan(
                case.query_original, mode, raw_enhancement=raw_output
            )
            if mode is EvaluationMode.FULL_ENHANCEMENT
            else build_query_evaluation_plan(case.query_original, mode)
        )
        record = evaluate_retrieval_case(case, plan, retriever=retriever, packager=packager)
        rows.append(
            {
                "variant_kind": kind,
                "variant_id": variant_id,
                "pilot_id": case.query_id,
                "record": record.to_dict(),
            }
        )
        return record

    for index, case in enumerate(cases, 1):
        baseline_by_pilot[case.query_id] = append(
            "baseline", f"{case.query_id}/baseline", case, None, EvaluationMode.BASELINE_ORIGINAL
        )
        noop_by_pilot[case.query_id] = append(
            "noop", f"{case.query_id}/noop", case, noops[case.query_id], EvaluationMode.NOOP_APPLIED
        )
        if index % 10 == 0 or index == len(cases):
            print(f"QUERY_SFT_BASELINE_PROGRESS completed={index}/{len(cases)}", flush=True)
    for index, (candidate_id, (pilot_id, raw_output)) in enumerate(candidates.items(), 1):
        case = next(case for case in cases if case.query_id == pilot_id)
        record = append(
            "teacher_candidate", candidate_id, case, raw_output, EvaluationMode.FULL_ENHANCEMENT
        )
        candidates_by_pilot[pilot_id].append((candidate_id, record))
        if index % 10 == 0 or index == len(candidates):
            print(f"QUERY_SFT_CANDIDATE_PROGRESS completed={index}/{len(candidates)}", flush=True)
    return rows, {
        "baseline_by_pilot": baseline_by_pilot,
        "noop_by_pilot": noop_by_pilot,
        "candidates_by_pilot": candidates_by_pilot,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _summary(evaluated: dict[str, Any], authoring_types: dict[str, str]) -> dict[str, object]:
    selections = []
    counters = {"teacher_candidate": 0, "noop": 0}
    for pilot_id, baseline in evaluated["baseline_by_pilot"].items():
        reference = max(_score(baseline), _score(evaluated["noop_by_pilot"][pilot_id]))
        candidates = evaluated["candidates_by_pilot"][pilot_id]
        qualified = [
            (candidate_id, record)
            for candidate_id, record in candidates
            if _score(record) >= reference
            and (
                _score(record) > reference
                or authoring_types[pilot_id] in {"colloquial", "ellipsis"}
            )
        ]
        if qualified:
            winner_id, winner = max(qualified, key=lambda item: (_score(item[1]), item[0]))
            selection = "teacher_candidate"
        else:
            winner_id, winner = f"{pilot_id}/noop", evaluated["noop_by_pilot"][pilot_id]
            selection = "noop"
        counters[selection] += 1
        selections.append(
            {
                "pilot_id": pilot_id,
                "authoring_type": authoring_types[pilot_id],
                "selected_variant_id": winner_id,
                "selection": selection,
                "baseline_top5": _score(baseline),
                "noop_top5": _score(evaluated["noop_by_pilot"][pilot_id]),
                "selected_top5": _score(winner),
                "approved_teacher_candidates_compared": len(candidates),
            }
        )
    return {
        "records": {
            "pilot_inputs": len(selections),
            "baseline_variants": len(evaluated["baseline_by_pilot"]),
            "noop_variants": len(evaluated["noop_by_pilot"]),
            "approved_teacher_candidate_variants": sum(
                len(value) for value in evaluated["candidates_by_pilot"].values()
            ),
            "total_retrieval_evaluations": (
                len(evaluated["baseline_by_pilot"])
                + len(evaluated["noop_by_pilot"])
                + sum(len(value) for value in evaluated["candidates_by_pilot"].values())
            ),
            "selected_by_kind": counters,
        },
        "selection_policy": {
            "primary_metrics": ["complete_hit@5", "required_gt_coverage@5", "mrr@5"],
            "candidate_must_not_degrade_baseline_or_noop": True,
            "strict_improvement_required_except_authoring_types": ["colloquial", "ellipsis"],
            "ties_allowed_for_exceptions": True,
        },
        "selections": selections,
        "runtime": {"elapsed_seconds": evaluated["elapsed_seconds"]},
    }


def _publish(output_dir: Path, payloads: list[tuple[str, str]]) -> None:
    if output_dir.exists():
        raise QuerySftPilotRetrievalEvaluationError(f"输出目录必须不存在: {output_dir}")
    try:
        output_dir.mkdir(parents=True)
        for name, payload in payloads:
            (output_dir / name).write_text(payload, encoding="utf-8", newline="\n")
        (output_dir / HASH_FILENAME).write_text(
            "".join(
                f"{hashlib.sha256(payload.encode('utf-8')).hexdigest()}  {name}\n"
                for name, payload in payloads
            ),
            encoding="utf-8",
            newline="\n",
        )
    except (OSError, UnicodeError) as error:
        raise QuerySftPilotRetrievalEvaluationError("无法发布 Retrieval 比较结果") from error


def run_query_sft_pilot_retrieval_evaluation(
    *,
    input_work_package_dir: Path,
    teacher_work_package_dir: Path,
    semantic_review_dir: Path,
    text_audit_path: Path,
    article_index: Path,
    artifact_dir: Path,
    tokenizer_path: Path,
    output_dir: Path,
    device: str,
) -> dict[str, object]:
    """执行不可覆盖的真实 Retrieval 比较。"""

    input_work_package_dir = Path(input_work_package_dir).resolve()
    teacher_work_package_dir = Path(teacher_work_package_dir).resolve()
    semantic_review_dir = Path(semantic_review_dir).resolve()
    text_audit_path = Path(text_audit_path).resolve()
    article_index = Path(article_index).resolve()
    artifact_dir = Path(artifact_dir).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    output_dir = Path(output_dir).resolve()
    text_audit_hash_path = _verify_hash(text_audit_path, "候选文本审计")
    text_audit = json.loads(text_audit_path.read_text(encoding="utf-8"))
    if (
        text_audit.get("pipeline") != "query_sft_pilot_candidate_text_audit_v1"
        or text_audit.get("complete") is not True
        or text_audit.get("records", {}).get("teacher_candidates_approved") != 114
        or text_audit.get("records", {}).get("evaluation_text_overlap") != 0
    ):
        raise QuerySftPilotRetrievalEvaluationError("候选文本审计状态无效")
    draft_path = input_work_package_dir / "query-sft-pilot-v1-query-input-draft.jsonl"
    cases, originals, authoring_types, case_inputs = _load_cases(
        input_work_package_dir=input_work_package_dir, draft_path=draft_path
    )
    result_path = teacher_work_package_dir / "query-sft-pilot-v1-teacher-candidate-results.jsonl"
    noop_path = teacher_work_package_dir / NOOP_FILENAME
    ledger_path = semantic_review_dir / LEDGER_FILENAME
    input_hashes = {
        "teacher_results_hash": _verify_hash(result_path, "教师候选结果"),
        "noop_hash": _verify_hash(noop_path, "确定性 no-op"),
        "semantic_ledger_hash": _verify_hash(ledger_path, "候选语义审核账本"),
    }
    noops, candidates = _load_variants(
        result_path=result_path,
        noop_path=noop_path,
        semantic_ledger_path=ledger_path,
        allowed_pilots=set(originals),
    )
    config = SemanticRetrievalConfig()
    frozen_inputs = {
        "article_index": _file_identity(article_index),
        "retrieval_artifacts": {
            name: _file_identity(artifact_dir / name) for name in _ARTIFACT_FILENAMES
        },
        "tokenizer_path": str(tokenizer_path),
        "candidate_text_audit": {
            **_file_identity(text_audit_path),
            "hash_manifest": _file_identity(text_audit_hash_path),
        },
        "draft": {
            **_file_identity(case_inputs["draft"]),
            "hash_manifest": _file_identity(case_inputs["draft_hash"]),
        },
        "audit_reference": _file_identity(case_inputs["audit_reference"]),
        "teacher_results": {
            **_file_identity(result_path),
            "hash_manifest": _file_identity(input_hashes["teacher_results_hash"]),
        },
        "noop": {
            **_file_identity(noop_path),
            "hash_manifest": _file_identity(input_hashes["noop_hash"]),
        },
        "semantic_ledger": {
            **_file_identity(ledger_path),
            "hash_manifest": _file_identity(input_hashes["semantic_ledger_hash"]),
        },
    }
    tokenizer = _load_tokenizer(tokenizer_path)
    packager = EvidencePackager(
        context_limit=CONTEXT_LIMIT,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        count_prompt_tokens=AnswerPromptTokenCounter(tokenizer),
    )
    actual_device = _resolve_device(device)
    repository = ArticleRepository.from_jsonl(article_index)
    retriever = load_semantic_retriever(
        repository=repository,
        artifact_dir=artifact_dir,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        reranker_model=DEFAULT_RERANKER_MODEL,
        device=actual_device,
        config=config,
    )
    rows, evaluated = _evaluate(
        cases, noops=noops, candidates=candidates, retriever=retriever, packager=packager
    )
    summary = _summary(evaluated, authoring_types)
    final_inputs = {
        "article_index": _file_identity(article_index),
        "retrieval_artifacts": {
            name: _file_identity(artifact_dir / name) for name in _ARTIFACT_FILENAMES
        },
    }
    if final_inputs != {
        "article_index": frozen_inputs["article_index"],
        "retrieval_artifacts": frozen_inputs["retrieval_artifacts"],
    }:
        raise QuerySftPilotRetrievalEvaluationError("Retrieval 资产在运行期间发生变化")
    manifest = {
        "pipeline": "query_sft_pilot_retrieval_evaluation_v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "inputs_frozen_before_run": frozen_inputs,
        "retrieval_identity": {
            "embedding_model": DEFAULT_EMBEDDING_MODEL,
            "reranker_model": DEFAULT_RERANKER_MODEL,
            "device": actual_device,
            "config": asdict(config),
            "query_compiler": "rag.query.enhancement.compile_retrieval_queries",
            "multi_query_fusion": "two_level_equal_weight_rrf",
            "reranker_query": "query_original",
            "packaging": {
                "context_limit": CONTEXT_LIMIT,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "maximum_evidence": 5,
                "selection": "max_complete_ordered_prefix",
            },
            "tokenizer": _tokenizer_identity(tokenizer, tokenizer_path),
        },
        "records": summary["records"],
        "selection_policy": summary["selection_policy"],
        "validation": {
            "input_identities_frozen_before_run": True,
            "retrieval_assets_unchanged_after_run": True,
            "baseline_noop_and_all_approved_candidates_compared": True,
            "query_sft_training_ready": False,
        },
        "complete": True,
    }
    records_payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for row in rows
    )
    summary_payload = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _publish(
        output_dir,
        [
            (RESULTS_FILENAME, records_payload),
            (SUMMARY_FILENAME, summary_payload),
            (MANIFEST_FILENAME, manifest_payload),
        ],
    )
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
    args = parser.parse_args()
    try:
        manifest = run_query_sft_pilot_retrieval_evaluation(
            input_work_package_dir=args.input_work_package_dir,
            teacher_work_package_dir=args.teacher_work_package_dir,
            semantic_review_dir=args.semantic_review_dir,
            text_audit_path=args.text_audit,
            article_index=args.article_index,
            artifact_dir=args.artifact_dir,
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
            device=args.device,
        )
    except (OSError, ValueError, TypeError, KeyError, QuerySftPilotRetrievalEvaluationError) as error:
        parser.error(str(error))
    print(
        "QUERY_SFT_RETRIEVAL_EVALUATION_OK "
        f"evaluations={manifest['records']['total_retrieval_evaluations']}",
        flush=True,
    )
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()


__all__ = [
    "QuerySftPilotRetrievalEvaluationError",
    "run_query_sft_pilot_retrieval_evaluation",
]
