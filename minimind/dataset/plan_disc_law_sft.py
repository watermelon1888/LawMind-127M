"""为 DISC-Law-SFT 法律问答生成只读清洗 dry-run 计划。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

try:
    from . import audit_disc_law_sft as auditor
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import audit_disc_law_sft as auditor


DEFAULT_WORK_ROOT = Path(os.environ.get("MINIMIND_WORK_ROOT", "/root/autodl-tmp/minimind-work"))
DEFAULT_OUTPUT_DIR = DEFAULT_WORK_ROOT / "reports" / "sft" / "disc_law_sft_cleaning_plan"

REPORT_FILENAME = "disc-law-sft-cleaning-plan.json"
MANIFEST_FILENAME = "disc-law-sft-cleaning-manifest.jsonl"
HASH_FILENAME = "disc-law-sft-cleaning-plan.sha256"

QA_SPECS = tuple(spec for spec in auditor.FILE_SPECS if spec.dataset_kind in {"pair_qa", "triplet_qa"})
PII_SIGNALS = {
    "valid_id_number",
    "contact_mobile_number",
    "mobile_number_candidate",
    "email_address",
    "detailed_address_candidate",
    "unmasked_natural_party_candidate",
}
TEXT_CORRUPTION_SIGNALS = {
    "replacement_character",
    "nul_character",
    "unknown_control_character",
    "reserved_special_token",
}
ANSWER_EXCLUSION_SIGNALS = {
    "answer_under_20_characters",
    "generic_answer_candidate",
    "follow_up_answer_candidate",
    "refusal_answer_candidate",
}


class PlanningError(RuntimeError):
    """表示 dry-run 无法安全完成。"""


@dataclass
class Candidate:
    """仅保存定位、指纹和规则，不保留原始问答文本。"""

    source_file: str
    dataset_kind: str
    line_number: int
    record_id_hash: str | None
    normalized_input_signature: bytes | None = None
    normalized_output_signature: bytes | None = None
    exclude_reasons: set[str] = field(default_factory=set)
    review_reasons: set[str] = field(default_factory=set)

    @property
    def disposition(self) -> str:
        if self.exclude_reasons:
            return "exclude"
        if self.review_reasons:
            return "review"
        return "retain"

    @property
    def all_reasons(self) -> list[str]:
        return sorted(self.exclude_reasons | self.review_reasons)


def _id_hash(record_id: str) -> str:
    return hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:16]


def _signature(value: str) -> bytes:
    return auditor.stable_signature(auditor.normalize_whitespace(value))


def _record_text_rules(
    candidate: Candidate,
    input_text: str,
    output_text: str,
    references: list[str],
    special_tokens: tuple[str, ...],
) -> None:
    signals: set[str] = set()
    for text in (input_text, output_text, *references):
        signals.update(auditor._field_signals(text, special_tokens))
    signals.update(auditor._answer_signals(output_text))
    candidate.exclude_reasons.update(PII_SIGNALS & signals)
    candidate.exclude_reasons.update(TEXT_CORRUPTION_SIGNALS & signals)
    candidate.exclude_reasons.update(ANSWER_EXCLUSION_SIGNALS & signals)
    if "html_fragment" in signals:
        candidate.review_reasons.add("html_fragment")
    if "url" in signals:
        candidate.review_reasons.add("url")


def _triplet_rules(
    candidate: Candidate,
    input_text: str,
    output_text: str,
    references: list[str],
    npc_index: auditor.NpcIndex | None,
) -> str | None:
    marker_count = input_text.count("<问题>")
    question: str | None = None
    if marker_count != 1:
        candidate.exclude_reasons.add("question_marker_anomaly")
    else:
        _, question = input_text.split("<问题>", 1)
        if not question.strip():
            candidate.exclude_reasons.add("empty_question_after_marker")

    if not references or any(not value.strip() for value in references):
        candidate.exclude_reasons.add("empty_reference")
    elif not all(reference in input_text for reference in references):
        candidate.exclude_reasons.add("reference_not_fully_embedded_in_input")

    output_citations = set(auditor.extract_citations(output_text))
    reference_citations = set(auditor.extract_citations("\n".join(references)))
    if not output_citations:
        candidate.review_reasons.add("output_without_explicit_citation")
    else:
        matches = len(output_citations & reference_citations)
        if matches == len(output_citations):
            pass
        elif matches:
            candidate.review_reasons.add("output_citations_partly_found_in_reference")
        else:
            candidate.review_reasons.add("output_citations_none_found_in_reference")

        if npc_index is not None:
            for law_name, article in output_citations:
                status = npc_index.locate(law_name, article)
                if status != "article_found_in_snapshot":
                    candidate.review_reasons.add(status)
    return question


def _parse_candidate(
    raw_line: bytes,
    spec: auditor.FileSpec,
    line_number: int,
    special_tokens: tuple[str, ...],
    npc_index: auditor.NpcIndex | None,
) -> Candidate:
    base = Candidate(spec.filename, spec.dataset_kind, line_number, None)
    try:
        text = raw_line.decode("utf-8")
    except UnicodeDecodeError:
        base.exclude_reasons.add("utf8_error")
        return base
    if not text.strip():
        base.exclude_reasons.add("blank_line")
        return base
    try:
        record = json.loads(text)
    except json.JSONDecodeError:
        base.exclude_reasons.add("json_error")
        return base
    if not isinstance(record, dict):
        base.exclude_reasons.add("non_object_record")
        return base

    expected = set(spec.required_fields)
    if set(record) != expected:
        base.exclude_reasons.add("schema_field_anomaly")
    if any(not isinstance(record.get(name), str) for name in ("id", "input", "output")):
        base.exclude_reasons.add("invalid_core_field_type")
        return base
    references: list[str] = []
    if spec.has_reference:
        value = record.get("reference")
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            base.exclude_reasons.add("invalid_reference_type")
            return base
        references = value

    record_id = str(record["id"])
    input_text = str(record["input"])
    output_text = str(record["output"])
    base.record_id_hash = _id_hash(record_id)
    if not record_id.strip():
        base.exclude_reasons.add("empty_id")
    if not input_text.strip():
        base.exclude_reasons.add("empty_input")
    if not output_text.strip():
        base.exclude_reasons.add("empty_output")
    if auditor.task_prefix(record_id) != "legal_question_answering":
        base.exclude_reasons.add("unexpected_task_prefix")

    _record_text_rules(base, input_text, output_text, references, special_tokens)
    question = input_text
    if spec.dataset_kind == "triplet_qa":
        question = _triplet_rules(base, input_text, output_text, references, npc_index)
    if question is not None and question.strip():
        base.normalized_input_signature = _signature(question)
    if output_text.strip():
        base.normalized_output_signature = _signature(output_text)
    return base


def _apply_duplicate_rules(candidates: list[Candidate]) -> None:
    by_dataset_and_input: dict[tuple[str, bytes], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        if candidate.normalized_input_signature is not None:
            by_dataset_and_input[(candidate.dataset_kind, candidate.normalized_input_signature)].append(index)

    for members in by_dataset_and_input.values():
        if len(members) < 2:
            continue
        answers = {
            candidates[index].normalized_output_signature
            for index in members
            if candidates[index].normalized_output_signature is not None
        }
        if len(answers) > 1:
            for index in members:
                candidates[index].review_reasons.add("same_question_answer_variant")
            continue
        for index in members[1:]:
            candidates[index].exclude_reasons.add("normalized_full_record_duplicate")


def _manifest_record(candidate: Candidate) -> dict[str, object]:
    return {
        "source_file": candidate.source_file,
        "source_line": candidate.line_number,
        "record_id_sha256_prefix": candidate.record_id_hash,
        "id_namespace": f"disc_law_sft:{candidate.dataset_kind}",
        "disposition": candidate.disposition,
        "reasons": candidate.all_reasons,
    }


def _write_json(path: Path, value: object) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    pending.replace(path)


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    pending = path.with_suffix(path.suffix + ".pending")
    with pending.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    pending.replace(path)


def build_cleaning_plan(
    input_dir: Path,
    output_dir: Path,
    npc_corpus_path: Path | None = None,
    special_tokens: tuple[str, ...] = ("<|im_start|>", "<|im_end|>"),
) -> dict[str, object]:
    """扫描两个 QA 文件并发布不含问答原文的 dry-run 计划。"""

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        raise PlanningError(f"缺少 DISC 原始目录: {input_dir}")
    if auditor._path_is_within(output_dir, input_dir):
        raise PlanningError("报告目录不能位于 DISC 原始目录内")
    missing = [spec.filename for spec in QA_SPECS if not (input_dir / spec.filename).is_file()]
    if missing:
        raise PlanningError("缺少 QA 原始文件: " + ", ".join(missing))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [output_dir / name for name in (REPORT_FILENAME, MANIFEST_FILENAME, HASH_FILENAME)]
    pending_paths = [path.with_suffix(path.suffix + ".pending") for path in output_paths]
    occupied = [path.name for path in (*output_paths, *pending_paths) if path.exists()]
    if occupied:
        raise PlanningError("输出目录已包含 dry-run 产物，请使用新目录: " + ", ".join(occupied))

    npc_index = auditor.load_npc_index(npc_corpus_path.resolve() if npc_corpus_path else None)
    candidates: list[Candidate] = []
    input_before: dict[str, dict[str, object]] = {}
    for spec in QA_SPECS:
        path = input_dir / spec.filename
        input_sha256 = auditor.sha256_file(path)
        input_before[spec.filename] = {
            "bytes": path.stat().st_size,
            "sha256": input_sha256,
            "upstream_verified": (
                path.stat().st_size == spec.upstream_bytes
                and input_sha256 == spec.upstream_sha256
            ),
        }
        print(f"[dry-run] {path}", file=sys.stderr, flush=True)
        with path.open("rb") as source:
            for line_number, raw_line in enumerate(source, start=1):
                candidates.append(
                    _parse_candidate(raw_line, spec, line_number, special_tokens, npc_index)
                )

    _apply_duplicate_rules(candidates)
    disposition_counts = Counter(candidate.disposition for candidate in candidates)
    rule_hits = Counter(reason for candidate in candidates for reason in candidate.all_reasons)
    combinations = Counter(",".join(candidate.all_reasons) or "<none>" for candidate in candidates)
    by_dataset: dict[str, Counter[str]] = defaultdict(Counter)
    for candidate in candidates:
        by_dataset[candidate.dataset_kind][candidate.disposition] += 1

    input_after = {
        spec.filename: {
            "bytes": (input_dir / spec.filename).stat().st_size,
            "sha256": auditor.sha256_file(input_dir / spec.filename),
        }
        for spec in QA_SPECS
    }
    inputs_unchanged = all(
        input_before[name]["bytes"] == input_after[name]["bytes"]
        and input_before[name]["sha256"] == input_after[name]["sha256"]
        for name in input_before
    )
    report: dict[str, object] = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "input_dir": str(input_dir),
            "files": [spec.filename for spec in QA_SPECS],
            "dry_run": True,
            "training_dataset_written": False,
            "raw_inputs_modified": not inputs_unchanged,
        },
        "policy": {
            "retain": "通过自动规则，可进入下一阶段标准化候选集",
            "review": "存在引用定位、答案变体或轻度噪声，仅进入人工复核桶",
            "exclude": "存在结构错误、PII、低价值回答、文本损坏或重复记录",
            "precedence": ["exclude", "review", "retain"],
            "id_namespaces": {
                "pair_qa": "disc_law_sft:pair_qa:<原始ID>",
                "triplet_qa": "disc_law_sft:triplet_qa:<原始ID>",
            },
            "triplet_input": "后续标准化必须直接使用原始 input，不得按 reference 重建或再次拼接",
        },
        "inputs": input_before,
        "totals": {
            "records": len(candidates),
            "retain": disposition_counts["retain"],
            "review": disposition_counts["review"],
            "exclude": disposition_counts["exclude"],
        },
        "by_dataset": {
            name: {
                "records": sum(counts.values()),
                "retain": counts["retain"],
                "review": counts["review"],
                "exclude": counts["exclude"],
            }
            for name, counts in sorted(by_dataset.items())
        },
        "independent_rule_hits": dict(sorted(rule_hits.items())),
        "rule_intersections": dict(sorted(combinations.items(), key=lambda item: (-item[1], item[0]))),
        "outputs": {
            "report": REPORT_FILENAME,
            "manifest": MANIFEST_FILENAME,
            "sha256_manifest": HASH_FILENAME,
        },
        "limitations": [
            "正则 PII 命中是保守自动排除规则，发布训练集前仍应抽样复核误杀率。",
            "NPC 未定位只进入复核桶，不代表法条错误。",
            "同题多答案只进入复核桶，不自动选择或判定正确答案。",
            "本计划不检查语义近重复、评估污染、法律结论正确性和法律时效。",
            "最终数据量必须通过固定验证集上的分档 SFT 学习曲线决定。",
        ],
    }

    report_path = output_dir / REPORT_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    hash_path = output_dir / HASH_FILENAME
    _write_jsonl(manifest_path, (_manifest_record(candidate) for candidate in candidates))
    _write_json(report_path, report)
    hash_pending_path = hash_path.with_suffix(hash_path.suffix + ".pending")
    hash_pending_path.write_text(
        f"{auditor.sha256_file(report_path)}  {REPORT_FILENAME}\n"
        f"{auditor.sha256_file(manifest_path)}  {MANIFEST_FILENAME}\n",
        encoding="utf-8",
        newline="\n",
    )
    hash_pending_path.replace(hash_path)
    return report


def parse_args() -> argparse.Namespace:
    """解析云端 dry-run 参数。"""

    parser = argparse.ArgumentParser(description="生成 DISC-Law-SFT QA 只读清洗 dry-run 计划")
    parser.add_argument("--input-dir", type=Path, default=auditor.DEFAULT_INPUT_DIR, help="DISC 原始 JSONL 目录")
    parser.add_argument("--npc-corpus", type=Path, default=auditor.DEFAULT_NPC_CORPUS, help="NPC 现行法规标准化 JSONL")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="新的 dry-run 输出目录")
    return parser.parse_args()


def main() -> None:
    """执行 dry-run 并打印预计去向。"""

    args = parse_args()
    try:
        report = build_cleaning_plan(args.input_dir, args.output_dir, args.npc_corpus)
    except (PlanningError, auditor.AuditError, OSError, ValueError) as error:
        print(f"[失败] {error}", file=sys.stderr)
        raise SystemExit(1) from error
    totals = report["totals"]
    print(
        f"[完成] 保留 {totals['retain']:,}，复核 {totals['review']:,}，排除 {totals['exclude']:,}"
    )
    print(f"汇总报告: {args.output_dir / REPORT_FILENAME}")
    print(f"逐条 manifest: {args.output_dir / MANIFEST_FILENAME}")


if __name__ == "__main__":
    main()
