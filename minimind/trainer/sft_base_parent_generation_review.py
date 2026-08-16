"""发布并汇总基础法律 SFT 五候选的匿名配对生成评审。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import sft_base_generation_evaluation as generation
from . import sft_base_parent_evaluation as formal
from . import train_full_sft as training


MANIFEST_PIPELINE = "legal_sft_base_parent_generation_review_manifest_v1"
REVEAL_PIPELINE = "legal_sft_base_parent_generation_review_reveal_v1"
SUMMARY_PIPELINE = "legal_sft_base_parent_generation_review_summary_v1"
REVIEW_PHASES = ("parent", "base_100")
ANONYMOUS_LABELS = ("A", "B", "C", "D", "E")
ENUM_FIELDS = {
    "conclusion_correctness": ("correct", "partial", "wrong", "unjudgeable"),
    "necessary_coverage": ("complete", "partial", "missing", "unjudgeable"),
    "instruction_following": ("pass", "partial", "fail", "unjudgeable"),
    "pair_delta": ("improved", "tied", "regressed", "unjudgeable"),
}
BOOLEAN_FIELDS = (
    "legal_basis_error",
    "unsupported_claim",
    "repetition_or_truncation",
)


def _load_verified(path: str | Path, description: str) -> tuple[dict[str, Any], str]:
    return training._load_verified_json(Path(path).resolve(), description)


def _observation(candidate: Mapping[str, Any], phase: str) -> Mapping[str, Any]:
    matches = [
        item for item in candidate.get("observations", []) if item.get("phase") == phase
    ]
    if len(matches) != 1:
        raise ValueError(f"候选缺少唯一观察点: {phase}")
    return matches[0]


def _generation_reports(
    run_root: Path, candidate: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    artifact_root = (
        run_root
        / "candidates"
        / candidate["candidate_id"]
        / "observations"
        / "artifacts"
    )
    available: dict[str, dict[str, Any]] = {}
    for path in artifact_root.rglob("generation.json"):
        report, digest = _load_verified(path, "固定生成观察报告")
        weights_sha256 = report.get("inputs", {}).get("weights", {}).get("sha256")
        if (
            report.get("pipeline") != generation.REPORT_PIPELINE
            or not isinstance(weights_sha256, str)
            or report.get("complete") is not True
        ):
            raise ValueError("固定生成观察报告身份无效")
        if weights_sha256 in available:
            raise ValueError("同一权重存在重复固定生成报告")
        available[weights_sha256] = {
            "path": str(path.resolve()),
            "sha256": digest,
            "report": report,
        }

    selected = {}
    for phase in REVIEW_PHASES:
        weights_sha256 = _observation(candidate, phase).get("weights", {}).get("sha256")
        if weights_sha256 not in available:
            raise ValueError(f"候选缺少固定生成报告: {phase}")
        selected[phase] = available[weights_sha256]
    return selected


def _records_by_id(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    records = report.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("固定生成报告缺少逐题记录")
    indexed = {}
    for record in records:
        record_id = record.get("record_id") if isinstance(record, dict) else None
        if not isinstance(record_id, str) or not record_id or record_id in indexed:
            raise ValueError("固定生成逐题 record_id 无效或重复")
        indexed[record_id] = record
    return indexed


def _public_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "raw_output": record["raw_output"],
        "generated_tokens": record["generated_tokens"],
        "nonempty_output": record["nonempty_output"],
        "hit_max_new_tokens": record["hit_max_new_tokens"],
    }


def _candidate_order(
    summary_digest: str, record_id: str, candidates: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    def key(candidate: Mapping[str, Any]) -> str:
        material = (
            f"{summary_digest}\0{record_id}\0{candidate['candidate_role']}"
        ).encode("utf-8")
        return hashlib.sha256(material).hexdigest()

    return sorted(candidates, key=key)


def prepare_review(
    *, formal_summary: str | Path, run_root: str | Path, output_dir: str | Path
) -> dict[str, Any]:
    """发布匿名配对 packet、空白 ledger 和独立 reveal。"""

    summary, summary_digest = _load_verified(formal_summary, "正式五候选 summary")
    candidates = summary.get("candidates")
    if (
        summary.get("pipeline") != formal.SUMMARY_PIPELINE
        or summary.get("coverage")
        != {
            "expected_candidates": 5,
            "completed_candidates": 5,
            "expected_observations": 20,
            "completed_observations": 20,
        }
        or not isinstance(candidates, list)
        or [item.get("candidate_role") for item in candidates]
        != list(formal.CANDIDATE_ROLES)
        or summary.get("private_holdout_evaluated") is not False
        or summary.get("complete") is not True
    ):
        raise ValueError("正式五候选 summary 状态无效")

    root = Path(run_root).resolve()
    reports = {
        candidate["candidate_role"]: _generation_reports(root, candidate)
        for candidate in candidates
    }
    canonical_role = formal.CANDIDATE_ROLES[0]
    canonical = _records_by_id(reports[canonical_role]["parent"]["report"])
    if len(canonical) != 64:
        raise ValueError("匿名评审必须固定为 64 题")

    indexed: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = {}
    for role in formal.CANDIDATE_ROLES:
        indexed[role] = {}
        for phase in REVIEW_PHASES:
            current = _records_by_id(reports[role][phase]["report"])
            if set(current) != set(canonical):
                raise ValueError("候选固定生成题目集合不一致")
            indexed[role][phase] = current
            for record_id, record in current.items():
                reference = canonical[record_id]
                if (
                    record.get("dataset_kind") != reference.get("dataset_kind")
                    or record.get("messages") != reference.get("messages")
                    or record.get("reference_answer")
                    != reference.get("reference_answer")
                ):
                    raise ValueError("候选固定生成题面或参考答案不一致")

    packet = []
    ledger = []
    mappings = []
    for question_index, record_id in enumerate(canonical, 1):
        reference = canonical[record_id]
        ordered = _candidate_order(summary_digest, record_id, candidates)
        for label, candidate in zip(ANONYMOUS_LABELS, ordered, strict=True):
            role = candidate["candidate_role"]
            review_item_id = f"Q{question_index:03d}-{label}"
            packet.append(
                {
                    "review_item_id": review_item_id,
                    "question_index": question_index,
                    "anonymous_label": label,
                    "record_id": record_id,
                    "dataset_kind": reference["dataset_kind"],
                    "messages": reference["messages"],
                    "reference_answer": reference["reference_answer"],
                    "parent": _public_record(indexed[role]["parent"][record_id]),
                    "base_100": _public_record(indexed[role]["base_100"][record_id]),
                }
            )
            ledger.append(
                {
                    "review_item_id": review_item_id,
                    **{field: None for field in ENUM_FIELDS},
                    **{field: None for field in BOOLEAN_FIELDS},
                    "notes": "",
                    "complete": False,
                }
            )
            mappings.append(
                {
                    "review_item_id": review_item_id,
                    "record_id": record_id,
                    "anonymous_label": label,
                    "candidate_role": role,
                }
            )

    output_root = Path(output_dir).resolve()
    if output_root.exists():
        raise FileExistsError(f"匿名评审输出目录已存在: {output_root}")
    packet_identity = generation._write_immutable_jsonl(
        output_root / "blind-review-packet.jsonl", packet
    )
    ledger_identity = generation._write_immutable_jsonl(
        output_root / "blind-review-ledger-template.jsonl", ledger
    )
    reveal = {
        "schema_version": "1.0",
        "pipeline": REVEAL_PIPELINE,
        "formal_summary_sha256": summary_digest,
        "mapping_method": "per_record_sha256_order_v1",
        "mappings": mappings,
        "candidates": [
            {
                "candidate_role": candidate["candidate_role"],
                "parent_weights": candidate["parent_weights"],
                "base_100_weights": candidate["base_100_weights"],
                "training": candidate["training"],
                "parent_metrics": _observation(candidate, "parent")["metrics"],
                "base_100_metrics": _observation(candidate, "base_100")["metrics"],
            }
            for candidate in candidates
        ],
        "complete": True,
    }
    reveal_path = output_root / "blind-review-reveal.json"
    generation._write_immutable_json(reveal_path, reveal)
    manifest = {
        "schema_version": "1.0",
        "pipeline": MANIFEST_PIPELINE,
        "formal_summary": {
            "path": str(Path(formal_summary).resolve()),
            "sha256": summary_digest,
        },
        "source_reports": {
            role: {
                phase: {
                    "path": value["path"],
                    "sha256": value["sha256"],
                }
                for phase, value in role_reports.items()
            }
            for role, role_reports in reports.items()
        },
        "review": {
            "questions": len(canonical),
            "candidates_per_question": len(ANONYMOUS_LABELS),
            "items": len(packet),
            "phases": list(REVIEW_PHASES),
            "enum_fields": {key: list(value) for key, value in ENUM_FIELDS.items()},
            "boolean_fields": list(BOOLEAN_FIELDS),
            "requires_full_coverage_before_reveal": True,
            "automatic_winner": None,
        },
        "packet": packet_identity,
        "ledger_template": ledger_identity,
        "reveal": generation._identity(reveal_path),
        "private_holdout_used": False,
        "complete": True,
    }
    generation._write_immutable_json(output_root / "manifest.json", manifest)
    return manifest


def _read_verified_jsonl(path: str | Path, description: str) -> list[dict[str, Any]]:
    resolved = Path(path).resolve()
    sidecar = resolved.with_suffix(".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise ValueError(f"{description}或 sidecar 不存在")
    expected = sidecar.read_text(encoding="utf-8").strip().split()[0]
    if expected != generation._sha256_file(resolved):
        raise ValueError(f"{description}哈希不一致")
    records = []
    with resolved.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{description}无法解析: {line_number}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{description}记录必须是 object")
            records.append(record)
    return records


def summarize_review(
    *, review_manifest: str | Path, completed_ledger: str | Path, output: str | Path
) -> dict[str, Any]:
    """在全量 ledger 完成后揭盲并按候选聚合错误类别。"""

    manifest, manifest_digest = _load_verified(review_manifest, "匿名评审 manifest")
    if (
        manifest.get("pipeline") != MANIFEST_PIPELINE
        or manifest.get("review", {}).get("items") != 320
        or manifest.get("private_holdout_used") is not False
        or manifest.get("complete") is not True
    ):
        raise ValueError("匿名评审 manifest 状态无效")
    packet_path = generation._verify_identity(manifest.get("packet"), "匿名评审 packet")
    reveal_path = generation._verify_identity(manifest.get("reveal"), "匿名评审 reveal")
    packet = _read_verified_jsonl(packet_path, "匿名评审 packet")
    ledger = _read_verified_jsonl(completed_ledger, "已完成匿名评审 ledger")
    reveal, _ = _load_verified(reveal_path, "匿名评审 reveal")

    expected_ids = [item["review_item_id"] for item in packet]
    ledger_by_id = {}
    for item in ledger:
        review_item_id = item.get("review_item_id")
        if not isinstance(review_item_id, str) or review_item_id in ledger_by_id:
            raise ValueError("匿名评审 ledger ID 无效或重复")
        if item.get("complete") is not True:
            raise ValueError("匿名评审 ledger 尚未全量完成")
        for field, allowed in ENUM_FIELDS.items():
            if item.get(field) not in allowed:
                raise ValueError(f"匿名评审字段无效: {field}")
        for field in BOOLEAN_FIELDS:
            if type(item.get(field)) is not bool:
                raise ValueError(f"匿名评审布尔字段无效: {field}")
        if not isinstance(item.get("notes"), str):
            raise ValueError("匿名评审 notes 必须是字符串")
        ledger_by_id[review_item_id] = item
    if set(ledger_by_id) != set(expected_ids) or len(ledger_by_id) != 320:
        raise ValueError("匿名评审 ledger 覆盖不完整")

    mapping = {
        item["review_item_id"]: item["candidate_role"]
        for item in reveal.get("mappings", [])
    }
    if set(mapping) != set(expected_ids):
        raise ValueError("匿名评审 reveal 映射不完整")
    aggregates = []
    for candidate in reveal["candidates"]:
        role = candidate["candidate_role"]
        reviews = [
            ledger_by_id[item_id] for item_id in expected_ids if mapping[item_id] == role
        ]
        aggregates.append(
            {
                "candidate_role": role,
                "reviewed_items": len(reviews),
                "categories": {
                    field: dict(Counter(item[field] for item in reviews))
                    for field in ENUM_FIELDS
                },
                "error_counts": {
                    field: sum(item[field] for item in reviews)
                    for field in BOOLEAN_FIELDS
                },
                "parent_metrics": candidate["parent_metrics"],
                "base_100_metrics": candidate["base_100_metrics"],
                "base_100_weights": candidate["base_100_weights"],
            }
        )
    report = {
        "schema_version": "1.0",
        "pipeline": SUMMARY_PIPELINE,
        "review_manifest": {
            "path": str(Path(review_manifest).resolve()),
            "sha256": manifest_digest,
        },
        "completed_ledger": generation._identity(completed_ledger),
        "coverage": {"questions": 64, "candidates": 5, "reviewed_items": 320},
        "candidates": aggregates,
        "decision": {
            "automatic_winner": None,
            "requires_paired_interpretation": True,
            "seed_43_top_two_pending": True,
        },
        "private_holdout_used": False,
        "complete": True,
    }
    generation._write_immutable_json(output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基础法律 SFT 五候选匿名生成评审")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="发布匿名配对评审包")
    prepare.add_argument("--formal-summary", required=True)
    prepare.add_argument("--run-root", required=True)
    prepare.add_argument("--output-dir", required=True)
    summary = commands.add_parser("summarize", help="汇总已完成匿名 ledger")
    summary.add_argument("--review-manifest", required=True)
    summary.add_argument("--completed-ledger", required=True)
    summary.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "prepare":
            result = prepare_review(
                formal_summary=args.formal_summary,
                run_root=args.run_root,
                output_dir=args.output_dir,
            )
            print(
                "SFT_BASE_PARENT_GENERATION_REVIEW_PREPARE_OK "
                f"items={result['review']['items']}"
            )
        else:
            result = summarize_review(
                review_manifest=args.review_manifest,
                completed_ledger=args.completed_ledger,
                output=args.output,
            )
            print(
                "SFT_BASE_PARENT_GENERATION_REVIEW_SUMMARY_OK "
                f"items={result['coverage']['reviewed_items']}"
            )
    except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
        raise SystemExit(f"[失败] {error}") from error


if __name__ == "__main__":
    main()
