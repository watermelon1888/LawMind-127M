"""依据人工修订 Retrieval 比较发布 superseding Query-SFT pilot revision。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from . import finalize_query_sft_pilot as base
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import finalize_query_sft_pilot as base


DEFAULT_CORRECTIONS = (
    base.DEFAULT_TEACHER_DIR / "query-sft-pilot-v1-human-corrections.jsonl"
)
DEFAULT_CORRECTION_EVALUATION_DIR = (
    base.DEFAULT_TEACHER_DIR / "query-sft-pilot-v1-retrieval-correction-evaluation"
)
DEFAULT_PARENT_RELEASE_DIR = base.DEFAULT_OUTPUT_DIR
DEFAULT_OUTPUT_DIR = (
    base.QUERY_POOL_ROOT / "pilot" / "query-sft-pilot-v1-training-release-r1"
)
_CORRECTION_FIELDS = {"candidate_id", "pilot_id", "raw_output", "reason"}


class QuerySftPilotRevisionError(RuntimeError):
    """表示 Query-SFT pilot revision 的输入或发布关卡无效。"""


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


def _verify_single_hash(path: Path, label: str) -> Path:
    sidecar = path.with_suffix(".sha256")
    try:
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftPilotRevisionError(f"无法读取{label} SHA-256") from error
    if lines != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftPilotRevisionError(f"{label} SHA-256 无效")
    return sidecar


def _verify_multi_hash(directory: Path, hash_name: str, names: set[str]) -> dict[str, Path]:
    hash_path = directory / hash_name
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftPilotRevisionError("无法读取修订 Retrieval SHA-256 清单") from error
    found: dict[str, Path] = {}
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise QuerySftPilotRevisionError("修订 Retrieval SHA-256 清单格式无效")
        digest, name = parts[0], parts[1].strip()
        path = directory / name
        if name not in names or not path.is_file() or _sha256_file(path) != digest:
            raise QuerySftPilotRevisionError("修订 Retrieval SHA-256 清单不匹配")
        found[name] = path
    if set(found) != names:
        raise QuerySftPilotRevisionError("修订 Retrieval SHA-256 清单范围无效")
    found[hash_name] = hash_path
    return found


def _load_corrections(path: Path) -> dict[str, tuple[str, dict[str, object]]]:
    _verify_single_hash(path, "人工修订候选")
    result: dict[str, tuple[str, dict[str, object]]] = {}
    try:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            row = json.loads(line)
            if set(row) != _CORRECTION_FIELDS:
                raise QuerySftPilotRevisionError(f"人工修订候选字段无效: {number}")
            candidate_id = row.get("candidate_id")
            pilot_id = row.get("pilot_id")
            raw_output = row.get("raw_output")
            if (
                not isinstance(candidate_id, str)
                or not isinstance(pilot_id, str)
                or candidate_id != f"{pilot_id}/human-correction-1"
                or not isinstance(raw_output, str)
                or pilot_id in result
            ):
                raise QuerySftPilotRevisionError("人工修订候选身份无效")
            target, _target_json = base._canonical_target(raw_output, candidate_id)
            result[pilot_id] = (candidate_id, target)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftPilotRevisionError("人工修订候选无法解析") from error
    if len(result) != 5:
        raise QuerySftPilotRevisionError("人工修订候选必须恰好为 5 条")
    return result


def _load_noops() -> dict[str, dict[str, object]]:
    path = base.DEFAULT_TEACHER_DIR / "query-sft-pilot-v1-deterministic-noop-candidates.jsonl"
    base._verify_hash(path, "确定性 no-op")
    result: dict[str, dict[str, object]] = {}
    for row in base._load_jsonl(path, "确定性 no-op"):
        if set(row) != base._NOOP_FIELDS or not isinstance(row.get("pilot_id"), str):
            raise QuerySftPilotRevisionError("确定性 no-op 字段无效")
        target, _target_json = base._canonical_target(row.get("raw_output"), row.get("candidate_id", "no-op"))
        result[row["pilot_id"]] = target
    return result


def _apply_corrections(
    records: list[dict[str, object]],
    corrections: dict[str, tuple[str, dict[str, object]]],
    selections: list[dict[str, object]],
) -> tuple[list[dict[str, object]], int, int]:
    by_id = {row["id"]: row for row in records}
    if len(by_id) != 39:
        raise QuerySftPilotRevisionError("父 pilot authoring 记录数无效")
    noops = _load_noops()
    selected_ids = set()
    human_count = 0
    noop_count = 0
    for selection in selections:
        pilot_id = selection.get("pilot_id")
        kind = selection.get("selection")
        selected_variant = selection.get("selected_variant_id")
        if (
            not isinstance(pilot_id, str)
            or pilot_id not in corrections
            or pilot_id not in by_id
            or pilot_id in selected_ids
        ):
            raise QuerySftPilotRevisionError("人工修订选择的 pilot 无效或重复")
        if kind == "human_correction":
            candidate_id, target = corrections[pilot_id]
            if selected_variant != candidate_id:
                raise QuerySftPilotRevisionError("人工修订选择的 candidate_id 无效")
            by_id[pilot_id]["target"] = target
            human_count += 1
        elif kind == "noop":
            if selected_variant != f"{pilot_id}/noop" or pilot_id not in noops:
                raise QuerySftPilotRevisionError("人工修订 no-op 选择无效")
            by_id[pilot_id]["target"] = noops[pilot_id]
            noop_count += 1
        else:
            raise QuerySftPilotRevisionError("人工修订选择类型无效")
        selected_ids.add(pilot_id)
    if selected_ids != set(corrections):
        raise QuerySftPilotRevisionError("人工修订选择未覆盖全部候选")
    return sorted(by_id.values(), key=lambda row: row["id"]), human_count, noop_count


def finalize_query_sft_pilot_revision(
    *,
    corrections_path: Path = DEFAULT_CORRECTIONS,
    correction_evaluation_dir: Path = DEFAULT_CORRECTION_EVALUATION_DIR,
    parent_release_dir: Path = DEFAULT_PARENT_RELEASE_DIR,
    tokenizer_path: Path = base.DEFAULT_TOKENIZER_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    tokenizer: Any | None = None,
) -> dict[str, object]:
    """发布使用人工修订 Retrieval 选择的 pilot revision-1。"""

    corrections_path = Path(corrections_path).resolve()
    correction_evaluation_dir = Path(correction_evaluation_dir).resolve()
    parent_release_dir = Path(parent_release_dir).resolve()
    tokenizer_path = Path(tokenizer_path).resolve()
    output_dir = Path(output_dir).resolve()
    correction_paths = _verify_multi_hash(
        correction_evaluation_dir,
        "query-sft-pilot-v1-correction-retrieval.sha256",
        {
            "query-sft-pilot-v1-correction-retrieval-records.jsonl",
            "query-sft-pilot-v1-correction-retrieval-summary.json",
            "query-sft-pilot-v1-correction-retrieval-manifest.json",
        },
    )
    correction_manifest = base._load_json(
        correction_paths["query-sft-pilot-v1-correction-retrieval-manifest.json"],
        "人工修订 Retrieval manifest",
    )
    correction_summary = base._load_json(
        correction_paths["query-sft-pilot-v1-correction-retrieval-summary.json"],
        "人工修订 Retrieval summary",
    )
    corrections = _load_corrections(corrections_path)
    if (
        correction_manifest.get("pipeline")
        != "query_sft_pilot_correction_retrieval_evaluation_v1"
        or correction_manifest.get("complete") is not True
        or correction_manifest.get("human_corrections", {}).get("file", {}).get("sha256")
        != _sha256_file(corrections_path)
        or correction_manifest.get("validation", {}).get("parent_retrieval_identity_matched_before_run")
        is not True
        or correction_manifest.get("validation", {}).get("retrieval_assets_unchanged_after_run")
        is not True
        or correction_summary.get("records", {}).get("human_corrections") != 5
        or correction_summary.get("records", {}).get("retrieval_evaluations") != 5
        or correction_summary.get("complete") is not True
    ):
        raise QuerySftPilotRevisionError("人工修订 Retrieval 发布状态无效")
    selections = correction_summary.get("selections")
    if not isinstance(selections, list) or len(selections) != 5:
        raise QuerySftPilotRevisionError("人工修订 Retrieval 选择数量无效")
    parent_manifest_path = parent_release_dir / base.MANIFEST_FILENAME
    _verify_multi_hash(parent_release_dir, base.HASH_FILENAME, {
        base.AUTHORING_FILENAME,
        base.CANDIDATE_FILENAME,
        base.LENGTH_REPORT_FILENAME,
        base.OVERFLOW_FILENAME,
        base.LABEL_REPORT_FILENAME,
        base.MANIFEST_FILENAME,
    })
    parent_manifest = base._load_json(parent_manifest_path, "父 pilot manifest")
    if (
        parent_manifest.get("pipeline") != "query_sft_pilot_training_release_v1"
        or parent_manifest.get("release_status") != "pilot_training_candidate"
        or parent_manifest.get("complete") is not True
    ):
        raise QuerySftPilotRevisionError("父 pilot 发布状态无效")
    records, parent_inputs = base._load_selected_records(
        input_dir=base.DEFAULT_INPUT_DIR,
        teacher_dir=base.DEFAULT_TEACHER_DIR,
        retrieval_dir=base.DEFAULT_RETRIEVAL_DIR,
    )
    revised_records, human_count, noop_count = _apply_corrections(
        records, corrections, selections
    )
    tokenizer = tokenizer or base.tokenizer_loader.load_tokenizer(tokenizer_path)
    candidates, length_report, label_report, overflow = base._audit_training_projection(
        revised_records, tokenizer
    )
    if overflow:
        raise QuerySftPilotRevisionError("revision 存在超长记录，禁止发布")
    authoring_payload = base._jsonl_payload(revised_records)
    candidate_payload = base._jsonl_payload(candidates)
    length_payload = json.dumps(length_report, ensure_ascii=False, indent=2) + "\n"
    label_payload = json.dumps(label_report, ensure_ascii=False, indent=2) + "\n"
    manifest = {
        "pipeline": "query_sft_pilot_training_release_v1",
        "release_status": "pilot_training_candidate",
        "release_revision": 1,
        "supersedes": _identity(parent_manifest_path),
        "scope": {"pilot_only": True, "formal_full_query_sft_published": False, "training_started": False},
        "inputs": {
            **parent_inputs,
            "human_corrections": _identity(corrections_path, records=5),
            "human_corrections_hash": _identity(corrections_path.with_suffix(".sha256")),
            "correction_retrieval_manifest": _identity(correction_paths["query-sft-pilot-v1-correction-retrieval-manifest.json"]),
            "correction_retrieval_summary": _identity(correction_paths["query-sft-pilot-v1-correction-retrieval-summary.json"]),
            "correction_retrieval_records": _identity(correction_paths["query-sft-pilot-v1-correction-retrieval-records.jsonl"], records=5),
            "correction_retrieval_hash": _identity(correction_paths["query-sft-pilot-v1-correction-retrieval.sha256"]),
        },
        "outputs": {
            "authoring": {"file": base.AUTHORING_FILENAME, "records": 39, "sha256": hashlib.sha256(authoring_payload.encode("utf-8")).hexdigest()},
            "training_candidate": {"file": base.CANDIDATE_FILENAME, "records": 39, "sha256": hashlib.sha256(candidate_payload.encode("utf-8")).hexdigest()},
        },
        "records": {"input_total": 40, "input_approved": 39, "input_rejected": 1, "retained_parent_targets": 34, "human_correction_targets": human_count, "noop_after_correction_evaluation": noop_count, "final_authoring": 39},
        "readiness": {"input_semantic_gt_review_complete": True, "teacher_protocol_semantic_text_audit_complete": True, "frozen_retrieval_selection_complete": True, "human_correction_retrieval_complete": True, "chat_template_length_audited": True, "dataset_label_mask_audited": True, "pilot_training_compatible": True, "formal_full_query_sft_training_ready": False},
        "limitations": ["本发布仅 supersede 39 条 Query-SFT pilot，不构成正式全量 Query-SFT。", "全量构造仍须独立完成输入构造、候选生成、独立审核、检索筛选和评估隔离。"],
        "complete": True,
    }
    payloads = {
        base.AUTHORING_FILENAME: authoring_payload,
        base.CANDIDATE_FILENAME: candidate_payload,
        base.LENGTH_REPORT_FILENAME: length_payload,
        base.OVERFLOW_FILENAME: "",
        base.LABEL_REPORT_FILENAME: label_payload,
        base.MANIFEST_FILENAME: json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    }
    base._publish(output_dir, payloads)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrections", type=Path, default=DEFAULT_CORRECTIONS)
    parser.add_argument("--correction-evaluation-dir", type=Path, default=DEFAULT_CORRECTION_EVALUATION_DIR)
    parser.add_argument("--parent-release-dir", type=Path, default=DEFAULT_PARENT_RELEASE_DIR)
    parser.add_argument("--tokenizer-path", type=Path, default=base.DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        manifest = finalize_query_sft_pilot_revision(
            corrections_path=args.corrections,
            correction_evaluation_dir=args.correction_evaluation_dir,
            parent_release_dir=args.parent_release_dir,
            tokenizer_path=args.tokenizer_path,
            output_dir=args.output_dir,
        )
    except (OSError, TypeError, ValueError, KeyError, QuerySftPilotRevisionError) as error:
        parser.error(str(error))
    print("QUERY_SFT_PILOT_REVISION_OK records=39")
    print(f"输出目录: {args.output_dir}")
    print(f"human_correction_targets={manifest['records']['human_correction_targets']}")


if __name__ == "__main__":
    main()
