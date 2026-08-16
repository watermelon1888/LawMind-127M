"""验证 RAG-SFT v2 校准裁决，并发布不可覆盖的校准完成 manifest。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent / "RAG-SFT" / "review" / "v2"
DEFAULT_PACKAGE_MANIFEST = ROOT / "calibration-v1" / "manifest.json"
DEFAULT_ADJUDICATION = ROOT / "calibration-v1" / "adjudication.jsonl"
DEFAULT_SUMMARY = ROOT / "calibration-v1" / "calibration-summary.md"
DEFAULT_OUTPUT = ROOT / "calibration-v1" / "calibration-adjudication.json"

_ADJUDICATION_FIELDS = {"query_id", "decision", "reason", "high_attention"}
_ADMISSION_REASONS = {"focused", "bounded_composite"}


class RagSftV2CalibrationFinalizationError(RuntimeError):
    """校准裁决未能与工作包和冻结规则闭合。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftV2CalibrationFinalizationError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise RagSftV2CalibrationFinalizationError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagSftV2CalibrationFinalizationError(
                        f"{description}不允许空行: {line_number}"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RagSftV2CalibrationFinalizationError(
                        f"{description}第 {line_number} 条必须是 object"
                    )
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftV2CalibrationFinalizationError):
            raise
        raise RagSftV2CalibrationFinalizationError(f"无法读取{description}: {path}") from error
    return records


def _require_new_outputs(output: Path, hash_output: Path) -> None:
    occupied = [
        str(candidate)
        for target in (output, hash_output)
        for candidate in (target, target.with_name(target.name + ".partial"))
        if candidate.exists()
    ]
    if occupied:
        raise RagSftV2CalibrationFinalizationError("目标输出已存在: " + ", ".join(occupied))


def finalize_rag_sft_v2_calibration(
    *, package_manifest_path: Path, adjudication_path: Path, summary_path: Path, output_path: Path
) -> dict[str, object]:
    """验证校准准入裁决，并以新 manifest 冻结审核尺度。"""

    package_manifest_path = Path(package_manifest_path).resolve()
    adjudication_path = Path(adjudication_path).resolve()
    summary_path = Path(summary_path).resolve()
    output_path = Path(output_path).resolve()
    hash_output = output_path.with_suffix(".sha256")
    _require_new_outputs(output_path, hash_output)
    package = _load_json(package_manifest_path, "校准工作包 manifest")
    expected_package_fields = {"pipeline", "release_status", "inputs", "selection", "output", "readiness", "complete"}
    if (
        set(package) != expected_package_fields
        or package.get("pipeline") != "rag_sft_v2_calibration_work_package"
        or package.get("release_status") != "human_review_only"
        or package.get("complete") is not True
    ):
        raise RagSftV2CalibrationFinalizationError("校准工作包 manifest 身份无效")
    output = package.get("output")
    selection = package.get("selection")
    inputs = package.get("inputs")
    if not isinstance(output, dict) or not isinstance(selection, dict) or not isinstance(inputs, dict):
        raise RagSftV2CalibrationFinalizationError("校准工作包 manifest 缺少输入或输出")
    work = output.get("work_items")
    rubric = inputs.get("rubric")
    if not isinstance(work, dict) or not isinstance(rubric, dict):
        raise RagSftV2CalibrationFinalizationError("校准工作包未绑定工作项或 rubric")
    work_path = Path(work.get("path", "")).resolve()
    rubric_path = Path(rubric.get("path", "")).resolve()
    if not work_path.is_file() or not rubric_path.is_file():
        raise RagSftV2CalibrationFinalizationError("校准工作项或 rubric 不存在")
    if work.get("sha256") != _sha256(work_path) or rubric.get("sha256") != _sha256(rubric_path):
        raise RagSftV2CalibrationFinalizationError("校准工作项或 rubric 身份已变化")
    work_rows = _load_jsonl(work_path, "校准工作项")
    selected = selection.get("query_ids")
    if not isinstance(selected, list) or len(selected) != 20 or len(selected) != len(set(selected)):
        raise RagSftV2CalibrationFinalizationError("校准选择必须为 20 个唯一 query_id")
    work_ids = [row.get("query_id") for row in work_rows]
    if work_ids != selected or len(work_rows) != selection.get("records"):
        raise RagSftV2CalibrationFinalizationError("校准工作项与选择清单不闭合")

    rows = _load_jsonl(adjudication_path, "校准裁决")
    if len(rows) != len(selected):
        raise RagSftV2CalibrationFinalizationError("校准裁决条数与工作项不一致")
    by_id: dict[str, dict[str, Any]] = {}
    for position, row in enumerate(rows, start=1):
        if set(row) != _ADJUDICATION_FIELDS:
            raise RagSftV2CalibrationFinalizationError(f"校准裁决第 {position} 条字段无效")
        query_id = row.get("query_id")
        if not isinstance(query_id, str) or query_id not in selected or query_id in by_id:
            raise RagSftV2CalibrationFinalizationError(f"校准裁决第 {position} 条 query_id 无效")
        if row.get("decision") != "admit" or row.get("reason") not in _ADMISSION_REASONS:
            raise RagSftV2CalibrationFinalizationError("校准裁决必须全部为有效 admit")
        attention = row.get("high_attention")
        if not isinstance(attention, list) or any(
            not isinstance(item, str) or not item.strip() for item in attention
        ):
            raise RagSftV2CalibrationFinalizationError("high_attention 必须是字符串数组")
        by_id[query_id] = row
    if set(by_id) != set(selected):
        raise RagSftV2CalibrationFinalizationError("校准裁决未覆盖全部工作项")
    if not summary_path.is_file():
        raise RagSftV2CalibrationFinalizationError("校准摘要不存在")

    focused = sum(row["reason"] == "focused" for row in rows)
    manifest = {
        "pipeline": "rag_sft_v2_calibration_finalization",
        "release_status": "rubric_calibrated",
        "inputs": {
            "work_package_manifest": _identity(package_manifest_path),
            "work_items": _identity(work_path),
            "rubric": _identity(rubric_path),
            "adjudication": _identity(adjudication_path),
            "summary": _identity(summary_path),
        },
        "records": {"total": len(rows), "focused": focused, "bounded_composite": len(rows) - focused},
        "readiness": {
            "rubric_calibrated": True,
            "all_calibration_records_admitted": True,
            "authoring_ready": False,
            "training_ready": False,
        },
        "complete": True,
    }
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        partial = output_path.with_name(output_path.name + ".partial")
        partial.write_text(payload, encoding="utf-8", newline="\n")
        partial.replace(output_path)
        hash_output.write_text(f"{digest}  {output_path.name}\n", encoding="utf-8", newline="\n")
    except OSError as error:
        raise RagSftV2CalibrationFinalizationError("无法发布校准完成 manifest") from error
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结 RAG-SFT v2 校准裁决")
    parser.add_argument("--package-manifest", type=Path, default=DEFAULT_PACKAGE_MANIFEST)
    parser.add_argument("--adjudication", type=Path, default=DEFAULT_ADJUDICATION)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        manifest = finalize_rag_sft_v2_calibration(
            package_manifest_path=args.package_manifest, adjudication_path=args.adjudication,
            summary_path=args.summary, output_path=args.output,
        )
    except RagSftV2CalibrationFinalizationError as error:
        raise SystemExit(f"[失败] {error}") from error
    print(f"[完成] 校准审核冻结 {manifest['records']['total']} 条")


if __name__ == "__main__":
    main()
