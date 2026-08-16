"""在冻结 Retrieval 链路上复核经人工修订的三个 Query-SFT pilot target。"""

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
    EvaluationMode,
    build_query_evaluation_plan,
    evaluate_retrieval_case,
)
from rag.knowledge import ArticleRepository
from rag.query.enhancement import parse_and_validate_query_enhancement
from rag.retrieval import SemanticRetrievalConfig, load_semantic_retriever
from rag.retrieval.loader import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    _resolve_device,
)

try:
    from . import evaluate_query_sft_pilot_retrieval as parent
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import evaluate_query_sft_pilot_retrieval as parent


DEFAULT_CORRECTIONS = (
    parent.DEFAULT_TEACHER_WORK_PACKAGE_DIR
    / "query-sft-pilot-v1-human-corrections.jsonl"
)
DEFAULT_PARENT_DIR = parent.DEFAULT_OUTPUT_DIR
DEFAULT_OUTPUT_DIR = (
    parent.DEFAULT_TEACHER_WORK_PACKAGE_DIR
    / "query-sft-pilot-v1-retrieval-correction-evaluation"
)
DEFAULT_ARTICLE_INDEX = parent.DEFAULT_ARTICLE_INDEX
DEFAULT_ARTIFACT_DIR = parent.DEFAULT_ARTIFACT_DIR
DEFAULT_TOKENIZER_PATH = parent.DEFAULT_TOKENIZER_PATH

RECORDS_FILENAME = "query-sft-pilot-v1-correction-retrieval-records.jsonl"
SUMMARY_FILENAME = "query-sft-pilot-v1-correction-retrieval-summary.json"
MANIFEST_FILENAME = "query-sft-pilot-v1-correction-retrieval-manifest.json"
HASH_FILENAME = "query-sft-pilot-v1-correction-retrieval.sha256"
_CORRECTION_FIELDS = {"candidate_id", "pilot_id", "raw_output", "reason"}
_EXPECTED_PILOTS = {
    "query_sft_pilot:0005",
    "query_sft_pilot:0024",
    "query_sft_pilot:0026",
    "query_sft_pilot:0022",
    "query_sft_pilot:0040",
}


class QuerySftPilotCorrectionEvaluationError(RuntimeError):
    """表示 pilot 人工修订候选或其 Retrieval 比较无效。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if records is not None:
        result["records"] = records
    return result


def _verify_hash(path: Path, label: str) -> Path:
    sidecar = path.with_suffix(".sha256")
    try:
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftPilotCorrectionEvaluationError(
            f"无法读取{label} SHA-256: {sidecar}"
        ) from error
    if lines != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftPilotCorrectionEvaluationError(f"{label} SHA-256 无效")
    return sidecar


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftPilotCorrectionEvaluationError(f"无法读取{label}: {path}") from error
    if not isinstance(value, dict):
        raise QuerySftPilotCorrectionEvaluationError(f"{label}必须是 JSON 对象")
    return value


def _load_corrections(path: Path) -> dict[str, dict[str, str]]:
    _verify_hash(path, "人工修订候选")
    corrections: dict[str, dict[str, str]] = {}
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, 1):
                row = json.loads(line)
                if set(row) != _CORRECTION_FIELDS:
                    raise QuerySftPilotCorrectionEvaluationError(
                        f"人工修订候选字段无效: {number}"
                    )
                if any(not isinstance(row.get(field), str) or not row[field] for field in _CORRECTION_FIELDS):
                    raise QuerySftPilotCorrectionEvaluationError(
                        f"人工修订候选值无效: {number}"
                    )
                pilot_id = row["pilot_id"]
                if pilot_id in corrections:
                    raise QuerySftPilotCorrectionEvaluationError("人工修订 pilot_id 重复")
                if row["candidate_id"] != f"{pilot_id}/human-correction-1":
                    raise QuerySftPilotCorrectionEvaluationError("人工修订 candidate_id 与 pilot_id 不匹配")
                parse_and_validate_query_enhancement(row["raw_output"])
                corrections[pilot_id] = row
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        if isinstance(error, QuerySftPilotCorrectionEvaluationError):
            raise
        raise QuerySftPilotCorrectionEvaluationError("人工修订候选协议无效") from error
    if set(corrections) != _EXPECTED_PILOTS:
        raise QuerySftPilotCorrectionEvaluationError("人工修订 pilot 范围与已批准修订清单不一致")
    return corrections


def _verify_parent_hashes(parent_dir: Path) -> dict[str, Path]:
    hash_path = parent_dir / parent.HASH_FILENAME
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftPilotCorrectionEvaluationError("无法读取父 Retrieval SHA-256 清单") from error
    names = {parent.RESULTS_FILENAME, parent.SUMMARY_FILENAME, parent.MANIFEST_FILENAME}
    found: dict[str, Path] = {}
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise QuerySftPilotCorrectionEvaluationError("父 Retrieval SHA-256 清单格式无效")
        digest, name = parts[0], parts[1].strip()
        target = parent_dir / name
        if name not in names or not target.is_file() or _sha256_file(target) != digest:
            raise QuerySftPilotCorrectionEvaluationError("父 Retrieval SHA-256 清单不匹配")
        found[name] = target
    if set(found) != names:
        raise QuerySftPilotCorrectionEvaluationError("父 Retrieval SHA-256 清单范围无效")
    found[hash_path.name] = hash_path
    return found


def _score_from_summary(value: object, label: str) -> tuple[int, float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or type(value[0]) is not int
        or not all(isinstance(item, (int, float)) for item in value[1:])
    ):
        raise QuerySftPilotCorrectionEvaluationError(f"父 Retrieval {label} 指标无效")
    return (value[0], float(value[1]), float(value[2]))


def _select(
    *,
    pilot_id: str,
    authoring_type: str,
    correction_id: str,
    correction_score: tuple[int, float, float],
    baseline_score: tuple[int, float, float],
    noop_score: tuple[int, float, float],
) -> dict[str, object]:
    reference = max(baseline_score, noop_score)
    qualifies = correction_score >= reference and (
        correction_score > reference or authoring_type in {"colloquial", "ellipsis"}
    )
    return {
        "pilot_id": pilot_id,
        "authoring_type": authoring_type,
        "selected_variant_id": correction_id if qualifies else f"{pilot_id}/noop",
        "selection": "human_correction" if qualifies else "noop",
        "baseline_top5": baseline_score,
        "noop_top5": noop_score,
        "correction_top5": correction_score,
        "selected_top5": correction_score if qualifies else noop_score,
    }


def _publish(output_dir: Path, payloads: list[tuple[str, str]]) -> None:
    if output_dir.exists():
        raise QuerySftPilotCorrectionEvaluationError(f"输出目录必须不存在: {output_dir}")
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
        raise QuerySftPilotCorrectionEvaluationError("无法发布人工修订 Retrieval 比较") from error


def evaluate_query_sft_pilot_corrections(
    *,
    corrections_path: Path,
    parent_dir: Path,
    input_work_package_dir: Path,
    article_index: Path,
    artifact_dir: Path,
    tokenizer_path: Path,
    output_dir: Path,
    device: str,
) -> dict[str, object]:
    """只评估三条人工修订候选，并按父比较的门槛决定 correction 或 no-op。"""

    corrections_path = Path(corrections_path).resolve()
    parent_dir = Path(parent_dir).resolve()
    input_work_package_dir = Path(input_work_package_dir).resolve()
    article_index = Path(article_index).resolve()
    artifact_dir = Path(artifact_dir).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    output_dir = Path(output_dir).resolve()
    corrections = _load_corrections(corrections_path)
    parent_paths = _verify_parent_hashes(parent_dir)
    parent_manifest = _load_json(parent_paths[parent.MANIFEST_FILENAME], "父 Retrieval manifest")
    parent_summary = _load_json(parent_paths[parent.SUMMARY_FILENAME], "父 Retrieval summary")
    if (
        parent_manifest.get("pipeline") != "query_sft_pilot_retrieval_evaluation_v1"
        or parent_manifest.get("complete") is not True
        or parent_manifest.get("validation", {}).get("retrieval_assets_unchanged_after_run") is not True
        or parent_summary.get("selection_policy", {}).get("primary_metrics")
        != ["complete_hit@5", "required_gt_coverage@5", "mrr@5"]
    ):
        raise QuerySftPilotCorrectionEvaluationError("父 Retrieval 发布状态无效")
    frozen = parent_manifest.get("inputs_frozen_before_run")
    current_assets = {
        "article_index": _file_identity(article_index),
        "retrieval_artifacts": {
            name: _file_identity(artifact_dir / name) for name in parent._ARTIFACT_FILENAMES
        },
    }
    if not isinstance(frozen, dict) or current_assets != {
        "article_index": frozen.get("article_index"),
        "retrieval_artifacts": frozen.get("retrieval_artifacts"),
    }:
        raise QuerySftPilotCorrectionEvaluationError("当前 Retrieval 资产已偏离父比较冻结身份")
    cases, _originals, authoring_types, _case_inputs = parent._load_cases(
        input_work_package_dir=input_work_package_dir,
        draft_path=input_work_package_dir / "query-sft-pilot-v1-query-input-draft.jsonl",
    )
    case_by_id = {case.query_id: case for case in cases}
    parent_selection = {
        row["pilot_id"]: row
        for row in parent_summary.get("selections", [])
        if isinstance(row, dict) and isinstance(row.get("pilot_id"), str)
    }
    if set(corrections) - set(parent_selection) or set(corrections) - set(case_by_id):
        raise QuerySftPilotCorrectionEvaluationError("人工修订未被父比较覆盖")

    tokenizer = _load_tokenizer(tokenizer_path)
    config = SemanticRetrievalConfig()
    actual_device = _resolve_device(device)
    current_identity = {
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
    }
    if current_identity != parent_manifest.get("retrieval_identity"):
        raise QuerySftPilotCorrectionEvaluationError("当前 Retrieval 身份已偏离父比较冻结配置")
    packager = EvidencePackager(
        context_limit=CONTEXT_LIMIT,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        count_prompt_tokens=AnswerPromptTokenCounter(tokenizer),
    )
    retriever = load_semantic_retriever(
        repository=ArticleRepository.from_jsonl(article_index),
        artifact_dir=artifact_dir,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        reranker_model=DEFAULT_RERANKER_MODEL,
        device=actual_device,
        config=config,
    )
    started = time.perf_counter()
    records = []
    selections = []
    for pilot_id in sorted(corrections):
        correction = corrections[pilot_id]
        plan = build_query_evaluation_plan(
            case_by_id[pilot_id].query_original,
            EvaluationMode.FULL_ENHANCEMENT,
            raw_enhancement=correction["raw_output"],
        )
        record = evaluate_retrieval_case(
            case_by_id[pilot_id], plan, retriever=retriever, packager=packager
        )
        score = parent._score(record)
        parent_row = parent_selection[pilot_id]
        selection = _select(
            pilot_id=pilot_id,
            authoring_type=authoring_types[pilot_id],
            correction_id=correction["candidate_id"],
            correction_score=score,
            baseline_score=_score_from_summary(parent_row.get("baseline_top5"), "baseline"),
            noop_score=_score_from_summary(parent_row.get("noop_top5"), "no-op"),
        )
        records.append({"pilot_id": pilot_id, "candidate_id": correction["candidate_id"], "record": record.to_dict()})
        selections.append(selection)
    final_assets = {
        "article_index": _file_identity(article_index),
        "retrieval_artifacts": {
            name: _file_identity(artifact_dir / name) for name in parent._ARTIFACT_FILENAMES
        },
    }
    if final_assets != current_assets:
        raise QuerySftPilotCorrectionEvaluationError("Retrieval 资产在人工修订比较期间发生变化")
    counts = {kind: sum(row["selection"] == kind for row in selections) for kind in ("human_correction", "noop")}
    summary = {
        "pipeline": "query_sft_pilot_correction_retrieval_evaluation_v1",
        "parent_selection_policy": parent_summary["selection_policy"],
        "records": {"human_corrections": 5, "retrieval_evaluations": 5, "selected_by_kind": counts},
        "selections": selections,
        "runtime": {"elapsed_seconds": time.perf_counter() - started},
        "complete": True,
    }
    manifest = {
        "pipeline": "query_sft_pilot_correction_retrieval_evaluation_v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "parent_retrieval": {
            "manifest": _identity(parent_paths[parent.MANIFEST_FILENAME]),
            "summary": _identity(parent_paths[parent.SUMMARY_FILENAME]),
            "records": _identity(parent_paths[parent.RESULTS_FILENAME], records=192),
            "sha256_manifest": _identity(parent_paths[parent.HASH_FILENAME]),
        },
        "human_corrections": {"file": _identity(corrections_path, records=5), "sha256_manifest": _identity(corrections_path.with_suffix(".sha256"))},
        "retrieval_identity": current_identity,
        "validation": {"parent_assets_matched_before_run": True, "parent_retrieval_identity_matched_before_run": True, "retrieval_assets_unchanged_after_run": True, "all_five_human_corrections_evaluated": True, "formal_full_query_sft_training_ready": False},
        "complete": True,
    }
    payloads = [
        (RECORDS_FILENAME, "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in records)),
        (SUMMARY_FILENAME, json.dumps(summary, ensure_ascii=False, indent=2) + "\n"),
        (MANIFEST_FILENAME, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"),
    ]
    _publish(output_dir, payloads)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS)
    parser.add_argument("--parent-dir", type=Path, default=DEFAULT_PARENT_DIR)
    parser.add_argument("--input-work-package-dir", type=Path, default=parent.DEFAULT_INPUT_WORK_PACKAGE_DIR)
    parser.add_argument("--article-index", type=Path, default=DEFAULT_ARTICLE_INDEX)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    try:
        manifest = evaluate_query_sft_pilot_corrections(
            corrections_path=args.corrections,
            parent_dir=args.parent_dir,
            input_work_package_dir=args.input_work_package_dir,
            article_index=args.article_index,
            artifact_dir=args.artifact_dir,
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
            device=args.device,
        )
    except (OSError, ValueError, TypeError, KeyError, QuerySftPilotCorrectionEvaluationError) as error:
        parser.error(str(error))
    print("QUERY_SFT_PILOT_CORRECTION_RETRIEVAL_OK evaluations=5")
    print(f"输出目录: {args.output_dir}")


if __name__ == "__main__":
    main()
