"""将 DISC-Law-SFT retain 记录冻结为可追溯的法律 SFT 候选数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

try:
    from . import audit_disc_law_sft as auditor
    from . import plan_disc_law_sft as planner
    from .build_sft_evaluation_exclusions import question_digest
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import audit_disc_law_sft as auditor
    from dataset import plan_disc_law_sft as planner
    from dataset.build_sft_evaluation_exclusions import question_digest


DEFAULT_WORK_ROOT = Path(os.environ.get("MINIMIND_WORK_ROOT", "/root/autodl-tmp/minimind-work"))
DEFAULT_CLEANING_DIR = DEFAULT_WORK_ROOT / "reports" / "sft" / "disc_law_sft_cleaning_plan_v1"
DEFAULT_EVALUATION_EXCLUSIONS = (
    DEFAULT_WORK_ROOT / "manifests" / "sft-evaluation-exclusions-project-rag-v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_WORK_ROOT / "data" / "standardized" / "sft" / "disc_law_sft_v1"
)
DEFAULT_MANIFEST_OUTPUT = DEFAULT_WORK_ROOT / "manifests" / "disc-law-sft-v1.json"

QA_SPECS = tuple(spec for spec in auditor.FILE_SPECS if spec.dataset_kind in {"pair_qa", "triplet_qa"})
MANIFEST_FIELDS = {
    "source_file",
    "source_line",
    "record_id_sha256_prefix",
    "id_namespace",
    "disposition",
    "reasons",
}
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
OUTPUT_FIELDS = (
    "id",
    "source",
    "source_file",
    "source_line",
    "task_type",
    "conversations",
)


class SftPreparationError(RuntimeError):
    """表示法律 SFT 候选数据无法安全发布。"""


class JsonlWriter:
    """先写 partial，关闭后再发布单个标准化 JSONL。"""

    def __init__(self, path: Path):
        self.path = path
        self.partial_path = path.with_name(path.name + ".partial")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.partial_path.open("xb")
        self.records = 0

    def write(self, record: dict[str, object]) -> None:
        payload = (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        self.file.write(payload)
        self.records += 1

    def publish(self, output_root: Path) -> dict[str, object]:
        self.file.close()
        self.partial_path.replace(self.path)
        return {
            "path": self.path.relative_to(output_root).as_posix(),
            "records": self.records,
            "bytes": self.path.stat().st_size,
            "sha256": auditor.sha256_file(self.path),
        }

    def abort(self) -> None:
        if not self.file.closed:
            self.file.close()


def normalize_content(value: str) -> str:
    """统一换行、执行 NFC 并清理首尾空白，不改变内部内容。"""

    return unicodedata.normalize("NFC", value.replace("\r\n", "\n").replace("\r", "\n")).strip()


def validation_bucket(question_sha256: str) -> int:
    """使用问题摘要的前 64 位生成确定性分桶。"""

    if DIGEST_RE.fullmatch(question_sha256) is None:
        raise SftPreparationError("问题 SHA-256 格式无效")
    return int(question_sha256[:16], 16) % 10_000


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SftPreparationError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise SftPreparationError(f"{description}必须是 JSON object: {path}")
    return value


def _load_hash_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split("  ", 1)
            if len(parts) != 2 or DIGEST_RE.fullmatch(parts[0]) is None or not parts[1]:
                raise SftPreparationError(f"清洗哈希清单格式无效: {path}:{line_number}")
            if parts[1] in entries:
                raise SftPreparationError(f"清洗哈希清单文件名重复: {path}:{line_number}")
            entries[parts[1]] = parts[0]
    except (OSError, UnicodeDecodeError) as error:
        raise SftPreparationError(f"无法读取清洗哈希清单: {path}") from error
    return entries


def _validate_cleaning_outputs(
    report_path: Path,
    decisions_path: Path,
    hashes_path: Path,
) -> tuple[dict[str, Any], dict[str, object]]:
    report_path = report_path.resolve()
    decisions_path = decisions_path.resolve()
    hashes_path = hashes_path.resolve()
    for path, description in (
        (report_path, "dry-run 汇总报告"),
        (decisions_path, "dry-run 逐条 manifest"),
        (hashes_path, "dry-run SHA-256 清单"),
    ):
        if not path.is_file():
            raise SftPreparationError(f"{description}不存在: {path}")
    entries = _load_hash_manifest(hashes_path)
    expected = {
        report_path.name: auditor.sha256_file(report_path),
        decisions_path.name: auditor.sha256_file(decisions_path),
    }
    if entries != expected:
        raise SftPreparationError("dry-run SHA-256 清单与报告文件不一致")
    report = _load_json(report_path, "dry-run 汇总报告")
    scope = report.get("scope")
    if not isinstance(scope, dict) or scope.get("dry_run") is not True:
        raise SftPreparationError("dry-run 汇总报告 scope 无效")
    if scope.get("training_dataset_written") is not False or scope.get("raw_inputs_modified") is not False:
        raise SftPreparationError("dry-run 汇总报告没有证明原始数据只读")
    if report.get("schema_version") != "1.0" or report.get("policy", {}).get("precedence") != [
        "exclude",
        "review",
        "retain",
    ]:
        raise SftPreparationError("dry-run 汇总报告版本或策略不受支持")
    return report, {
        "report": {"path": str(report_path), "sha256": expected[report_path.name]},
        "decisions": {
            "path": str(decisions_path),
            "bytes": decisions_path.stat().st_size,
            "sha256": expected[decisions_path.name],
        },
        "hash_manifest": {"path": str(hashes_path), "sha256": auditor.sha256_file(hashes_path)},
    }


def _load_evaluation_exclusions(path: Path) -> tuple[set[str], dict[str, object]]:
    path = path.resolve()
    if not path.is_file():
        raise SftPreparationError(f"评估排除清单不存在: {path}")
    sidecar = path.with_suffix(".sha256")
    if not sidecar.is_file():
        raise SftPreparationError(f"评估排除清单缺少相邻 SHA-256 清单: {sidecar}")
    expected_sidecar = f"{auditor.sha256_file(path)}  {path.name}"
    try:
        sidecar_lines = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise SftPreparationError(f"评估排除 SHA-256 清单无法读取: {sidecar}") from error
    if sidecar_lines != [expected_sidecar]:
        raise SftPreparationError("评估排除 SHA-256 清单校验失败")
    manifest = _load_json(path, "评估排除清单")
    schema_version = manifest.get("schema_version")
    if schema_version not in {"1.0", "1.1", "1.2"} or manifest.get("pipeline") != (
        "legal_sft_evaluation_exclusions"
    ):
        raise SftPreparationError("评估排除清单版本或 pipeline 无效")
    if manifest.get("normalization") != (
        "unicode_nfc_trim_and_collapse_whitespace_then_sha256"
    ):
        raise SftPreparationError("评估排除清单归一规则不一致")
    values = manifest.get("question_sha256")
    if not isinstance(values, list) or not values:
        raise SftPreparationError("评估排除清单问题摘要不能为空")
    if any(not isinstance(value, str) or DIGEST_RE.fullmatch(value) is None for value in values):
        raise SftPreparationError("评估排除清单包含无效问题摘要")
    digests = set(values)
    if len(digests) != len(values) or manifest.get("digest_count") != len(digests):
        raise SftPreparationError("评估排除清单问题摘要重复或计数不一致")
    assets = manifest.get("assets")
    if not isinstance(assets, list) or not assets:
        raise SftPreparationError("评估排除清单没有冻结资产")
    for asset in assets:
        if (
            not isinstance(asset, dict)
            or not isinstance(asset.get("name"), str)
            or not asset["name"]
            or not isinstance(asset.get("path"), str)
            or not asset["path"]
            or not isinstance(asset.get("bytes"), int)
            or asset["bytes"] < 0
            or not isinstance(asset.get("records"), int)
            or asset["records"] <= 0
            or not isinstance(asset.get("sha256"), str)
            or DIGEST_RE.fullmatch(asset["sha256"]) is None
        ):
            raise SftPreparationError("评估排除清单包含无效冻结资产")
    if manifest.get("raw_questions_included") is not False:
        raise SftPreparationError("评估排除清单不得包含原始问题")
    complete_for_formal_sft = manifest.get("complete_for_formal_sft")
    missing_required_assets = manifest.get("missing_required_assets")
    if not isinstance(complete_for_formal_sft, bool) or not isinstance(
        missing_required_assets, list
    ):
        raise SftPreparationError("评估排除清单完成状态无效")
    if any(not isinstance(value, str) or not value for value in missing_required_assets):
        raise SftPreparationError("评估排除清单缺失资产名称无效")
    if complete_for_formal_sft == bool(missing_required_assets):
        raise SftPreparationError("评估排除清单完成状态与缺失资产不一致")
    isolation_audit = manifest.get("isolation_audit")
    if complete_for_formal_sft:
        asset_names = {str(asset["name"]) for asset in assets}
        overlap_counts = (
            isolation_audit.get("overlap_counts")
            if isinstance(isolation_audit, dict)
            else None
        )
        if (
            schema_version != "1.2"
            or not {"project_rag_eval", "project_private_holdout_v2"} <= asset_names
            or not isinstance(isolation_audit, dict)
            or isolation_audit.get("compatible") is not True
            or isolation_audit.get("raw_text_included") is not False
            or not isinstance(overlap_counts, dict)
            or not overlap_counts
            or any(value != 0 for value in overlap_counts.values())
        ):
            raise SftPreparationError("正式评估排除清单的私有留出兼容性审计未闭合")
    return digests, {
        "schema_version": schema_version,
        "scope": manifest.get("scope"),
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": auditor.sha256_file(path),
        "sha256_manifest": {
            "path": str(sidecar),
            "bytes": sidecar.stat().st_size,
            "sha256": auditor.sha256_file(sidecar),
        },
        "digest_count": len(digests),
        "assets": assets,
        "isolation_audit": isolation_audit,
        "complete_for_formal_sft": complete_for_formal_sft,
        "missing_required_assets": missing_required_assets,
    }


def _decision_records(path: Path) -> Iterator[dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SftPreparationError(
                        f"dry-run 逐条 manifest JSON 无效: {path}:{line_number}"
                    ) from error
                if not isinstance(record, dict) or set(record) != MANIFEST_FIELDS:
                    raise SftPreparationError(
                        f"dry-run 逐条 manifest schema 无效: {path}:{line_number}"
                    )
                if (
                    not isinstance(record["source_file"], str)
                    or not isinstance(record["source_line"], int)
                    or record["source_line"] <= 0
                    or not isinstance(record["record_id_sha256_prefix"], str)
                    or re.fullmatch(r"[0-9a-f]{16}", record["record_id_sha256_prefix"])
                    is None
                    or not isinstance(record["id_namespace"], str)
                    or not isinstance(record["disposition"], str)
                    or not isinstance(record["reasons"], list)
                    or any(not isinstance(reason, str) for reason in record["reasons"])
                ):
                    raise SftPreparationError(
                        f"dry-run 逐条 manifest 字段类型无效: {path}:{line_number}"
                    )
                yield record
    except (OSError, UnicodeDecodeError) as error:
        raise SftPreparationError(f"无法读取 dry-run 逐条 manifest: {path}") from error


def _parse_raw_record(
    raw_line: bytes,
    spec: auditor.FileSpec,
    line_number: int,
) -> tuple[str, str, str, list[str]]:
    try:
        record = json.loads(raw_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SftPreparationError(f"原始记录无法解析: {spec.filename}:{line_number}") from error
    if not isinstance(record, dict) or set(record) != set(spec.required_fields):
        raise SftPreparationError(f"原始记录 schema 已变化: {spec.filename}:{line_number}")
    if any(not isinstance(record.get(field), str) for field in ("id", "input", "output")):
        raise SftPreparationError(f"原始记录核心字段无效: {spec.filename}:{line_number}")
    references: list[str] = []
    if spec.has_reference:
        value = record.get("reference")
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise SftPreparationError(f"原始记录 reference 无效: {spec.filename}:{line_number}")
        references = value
    return str(record["id"]), str(record["input"]), str(record["output"]), references


def _question_for_record(dataset_kind: str, input_text: str) -> str:
    if dataset_kind == "pair_qa":
        return input_text
    if input_text.count("<问题>") != 1:
        raise SftPreparationError("retain Triplet-QA 的 <问题> 标记异常")
    question = input_text.split("<问题>", 1)[1]
    if not question.strip():
        raise SftPreparationError("retain Triplet-QA 的问题为空")
    return question


def _write_manifest(manifest: dict[str, object], output_path: Path) -> None:
    hash_path = output_path.with_suffix(".sha256")
    partial_path = output_path.with_name(output_path.name + ".partial")
    hash_partial_path = hash_path.with_name(hash_path.name + ".partial")
    occupied = [path.name for path in (output_path, hash_path, partial_path, hash_partial_path) if path.exists()]
    if occupied:
        raise SftPreparationError("manifest 输出位置已被占用: " + ", ".join(occupied))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        partial_path.write_text(payload, encoding="utf-8", newline="\n")
        json.loads(partial_path.read_text(encoding="utf-8"))
        digest = auditor.sha256_file(partial_path)
        hash_partial_path.write_text(
            f"{digest}  {output_path.name}\n", encoding="utf-8", newline="\n"
        )
        partial_path.replace(output_path)
        hash_partial_path.replace(hash_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise SftPreparationError(f"无法发布 SFT data manifest: {output_path}") from error


def prepare_disc_law_sft(
    input_dir: Path,
    cleaning_report: Path,
    cleaning_decisions: Path,
    cleaning_hashes: Path,
    evaluation_exclusions: Path,
    output_root: Path,
    manifest_output: Path,
) -> dict[str, object]:
    """发布 retain-only 候选数据；评估资产不完整时保持训练阻断。"""

    input_dir = input_dir.resolve()
    output_root = output_root.resolve()
    manifest_output = manifest_output.resolve()
    if not input_dir.is_dir():
        raise SftPreparationError(f"DISC 原始目录不存在: {input_dir}")
    if output_root.exists():
        raise SftPreparationError(f"标准化输出目录必须是新目录: {output_root}")
    if auditor._path_is_within(output_root, input_dir) or auditor._path_is_within(
        manifest_output, input_dir
    ):
        raise SftPreparationError("标准化输出和 manifest 不能位于原始目录内")
    manifest_hash_output = manifest_output.with_suffix(".sha256")
    manifest_partial_output = manifest_output.with_name(manifest_output.name + ".partial")
    manifest_hash_partial_output = manifest_hash_output.with_name(
        manifest_hash_output.name + ".partial"
    )
    occupied_manifest_paths = [
        path.name
        for path in (
            manifest_output,
            manifest_hash_output,
            manifest_partial_output,
            manifest_hash_partial_output,
        )
        if path.exists()
    ]
    if occupied_manifest_paths:
        raise SftPreparationError(
            "manifest 输出位置已被占用: " + ", ".join(occupied_manifest_paths)
        )

    report, cleaning_identity = _validate_cleaning_outputs(
        cleaning_report.resolve(), cleaning_decisions.resolve(), cleaning_hashes.resolve()
    )
    exclusion_digests, exclusion_report = _load_evaluation_exclusions(evaluation_exclusions)
    expected_files = [spec.filename for spec in QA_SPECS]
    if report.get("scope", {}).get("files") != expected_files:
        raise SftPreparationError("dry-run 汇总报告文件范围不一致")

    for spec in QA_SPECS:
        raw_path = input_dir / spec.filename
        input_report = report.get("inputs", {}).get(spec.filename)
        if not raw_path.is_file() or not isinstance(input_report, dict):
            raise SftPreparationError(f"原始文件或 dry-run 输入身份缺失: {spec.filename}")
        if raw_path.stat().st_size != input_report.get("bytes") or auditor.sha256_file(
            raw_path
        ) != input_report.get("sha256"):
            raise SftPreparationError(f"原始文件已变化: {spec.filename}")

    output_paths = {
        (split, spec.dataset_kind): output_root / split / f"{spec.dataset_kind}.jsonl"
        for split in ("train", "validation/full", "validation/quick")
        for spec in QA_SPECS
    }
    output_root.mkdir(parents=True)
    writers = {key: JsonlWriter(path) for key, path in output_paths.items()}
    decisions = _decision_records(cleaning_decisions.resolve())
    stats = {spec.dataset_kind: Counter() for spec in QA_SPECS}
    seen_ids: set[str] = set()
    seen_full_records: set[bytes] = set()
    try:
        for spec in QA_SPECS:
            path = input_dir / spec.filename
            with path.open("rb") as source:
                for line_number, raw_line in enumerate(source, start=1):
                    try:
                        decision = next(decisions)
                    except StopIteration as error:
                        raise SftPreparationError("dry-run 逐条 manifest 提前结束") from error
                    if decision["source_file"] != spec.filename or decision["source_line"] != line_number:
                        raise SftPreparationError(
                            f"dry-run 定位顺序不一致: {spec.filename}:{line_number}"
                        )
                    record_id, input_text, output_text, _ = _parse_raw_record(
                        raw_line, spec, line_number
                    )
                    expected_id_hash = hashlib.sha256(record_id.encode("utf-8")).hexdigest()[:16]
                    expected_namespace = f"disc_law_sft:{spec.dataset_kind}"
                    if (
                        decision["record_id_sha256_prefix"] != expected_id_hash
                        or decision["id_namespace"] != expected_namespace
                    ):
                        raise SftPreparationError(
                            f"dry-run ID 身份不一致: {spec.filename}:{line_number}"
                        )
                    disposition = decision["disposition"]
                    if disposition not in {"retain", "review", "exclude"}:
                        raise SftPreparationError(
                            f"dry-run disposition 无效: {spec.filename}:{line_number}"
                        )
                    counts = stats[spec.dataset_kind]
                    counts["scanned"] += 1
                    counts[f"plan_{disposition}"] += 1
                    if disposition != "retain":
                        continue
                    if decision["reasons"] != []:
                        raise SftPreparationError(
                            f"retain 记录仍包含原因: {spec.filename}:{line_number}"
                        )

                    normalized_input = normalize_content(input_text)
                    normalized_output = normalize_content(output_text)
                    question = _question_for_record(spec.dataset_kind, normalized_input)
                    question_sha256 = question_digest(question)
                    if question_sha256 in exclusion_digests:
                        counts["evaluation_excluded"] += 1
                        continue
                    full_signature = auditor.stable_signature(
                        [
                            auditor.normalize_whitespace(normalized_input),
                            auditor.normalize_whitespace(normalized_output),
                        ]
                    )
                    if full_signature in seen_full_records:
                        counts["standardization_duplicate"] += 1
                        continue
                    seen_full_records.add(full_signature)

                    namespaced_id = f"{expected_namespace}:{record_id}"
                    if namespaced_id in seen_ids:
                        raise SftPreparationError(f"命名空间 ID 重复: {namespaced_id}")
                    seen_ids.add(namespaced_id)
                    task_type = (
                        "legal_qa_with_context" if spec.dataset_kind == "triplet_qa" else "legal_qa"
                    )
                    output_record: dict[str, object] = {
                        "id": namespaced_id,
                        "source": "disc_law_sft",
                        "source_file": spec.filename,
                        "source_line": line_number,
                        "task_type": task_type,
                        "conversations": [
                            {"role": "user", "content": normalized_input},
                            {"role": "assistant", "content": normalized_output},
                        ],
                    }
                    if tuple(output_record) != OUTPUT_FIELDS:
                        raise AssertionError("标准化输出字段顺序异常")
                    bucket = validation_bucket(question_sha256)
                    if bucket < 500:
                        writers[("validation/full", spec.dataset_kind)].write(output_record)
                        counts["full_validation"] += 1
                        if bucket < 50:
                            writers[("validation/quick", spec.dataset_kind)].write(output_record)
                            counts["quick_validation"] += 1
                    else:
                        writers[("train", spec.dataset_kind)].write(output_record)
                        counts["train"] += 1
                    counts["standardized"] += 1
                    counts["input_characters"] += len(normalized_input)
                    counts["output_characters"] += len(normalized_output)
        try:
            extra = next(decisions)
        except StopIteration:
            extra = None
        if extra is not None:
            raise SftPreparationError("dry-run 逐条 manifest 含额外记录")

        for spec in QA_SPECS:
            counts = stats[spec.dataset_kind]
            expected = report.get("by_dataset", {}).get(spec.dataset_kind, {})
            if (
                counts["scanned"] != expected.get("records")
                or counts["plan_retain"] != expected.get("retain")
                or counts["plan_review"] != expected.get("review")
                or counts["plan_exclude"] != expected.get("exclude")
            ):
                raise SftPreparationError(f"dry-run 汇总与逐条决定不闭合: {spec.dataset_kind}")
            if counts["scanned"] != (
                counts["plan_retain"] + counts["plan_review"] + counts["plan_exclude"]
            ):
                raise SftPreparationError(f"原始准入漏斗不闭合: {spec.dataset_kind}")
            if counts["plan_retain"] != (
                counts["evaluation_excluded"]
                + counts["standardization_duplicate"]
                + counts["standardized"]
            ):
                raise SftPreparationError(f"retain 标准化漏斗不闭合: {spec.dataset_kind}")
            if counts["standardized"] != counts["train"] + counts["full_validation"]:
                raise SftPreparationError(f"标准化 split 不闭合: {spec.dataset_kind}")

        outputs = {
            f"{split}/{dataset_kind}": writers[(split, dataset_kind)].publish(output_root)
            for split in ("train", "validation/full", "validation/quick")
            for dataset_kind in ("pair_qa", "triplet_qa")
        }
    except BaseException:
        for writer in writers.values():
            writer.abort()
        raise

    total_counts = Counter()
    for counts in stats.values():
        total_counts.update(counts)
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "pipeline": "disc_law_sft_retain_only_v1",
        "release_status": (
            "formal_candidate"
            if exclusion_report["complete_for_formal_sft"]
            else "provisional_candidate"
        ),
        "source": {
            "repository": auditor.UPSTREAM_REPOSITORY,
            "revision": auditor.UPSTREAM_REVISION,
            "input_dir": str(input_dir),
            "files": report["inputs"],
        },
        "cleaning": cleaning_identity,
        "evaluation_exclusion": exclusion_report,
        "policy": {
            "accepted_disposition": "retain",
            "review_disposition": "isolated",
            "exclude_disposition": "excluded",
            "id_namespaces": report["policy"]["id_namespaces"],
            "triplet_input": "原始 input 仅执行换行、NFC 和首尾清理，不按 reference 重建",
        },
        "normalization": {
            "content": "normalize_newlines_then_unicode_nfc_then_strip",
            "question_signature": (
                "unicode_nfc_trim_and_collapse_whitespace_then_sha256"
            ),
            "duplicate_signature": "normalized_input_and_output_sha256",
        },
        "validation": {
            "group": "normalized_question_across_pair_and_triplet",
            "bucket_expression": "int(question_sha256[:16], 16) % 10000",
            "full_validation_buckets": "0-499",
            "quick_validation_buckets": "0-49",
            "quick_is_subset_of_full": True,
        },
        "records": {
            "by_dataset": {
                dataset_kind: dict(sorted(counts.items()))
                for dataset_kind, counts in sorted(stats.items())
            },
            "totals": dict(sorted(total_counts.items())),
        },
        "output": {
            "root": str(output_root),
            "record_fields": list(OUTPUT_FIELDS),
            "files": outputs,
        },
        "readiness": {
            "standardization_complete": True,
            "formal_evaluation_isolation_complete": exclusion_report[
                "complete_for_formal_sft"
            ],
            "chat_template_length_audited": False,
            "cpt_parent_bound": False,
            "training_ready": False,
        },
        "limitations": [
            "review 桶没有进入第一版候选数据。",
            "尚未执行语义近重复和法律结论正确性审计。",
            "尚未应用 chat template 统计完整和 assistant token 长度。",
            "CPT 最终父权重尚未绑定，不得启动正式 SFT。",
        ],
        "complete": True,
    }
    _write_manifest(manifest, manifest_output)
    return manifest


def main() -> None:
    """解析云端参数并生成 retain-only SFT 候选数据。"""

    parser = argparse.ArgumentParser(description="生成 DISC-Law-SFT retain-only 标准化候选数据")
    parser.add_argument("--input-dir", type=Path, default=auditor.DEFAULT_INPUT_DIR, help="DISC 原始目录")
    parser.add_argument(
        "--cleaning-report",
        type=Path,
        default=DEFAULT_CLEANING_DIR / planner.REPORT_FILENAME,
        help="dry-run 汇总报告",
    )
    parser.add_argument(
        "--cleaning-decisions",
        type=Path,
        default=DEFAULT_CLEANING_DIR / planner.MANIFEST_FILENAME,
        help="dry-run 逐条 manifest",
    )
    parser.add_argument(
        "--cleaning-hashes",
        type=Path,
        default=DEFAULT_CLEANING_DIR / planner.HASH_FILENAME,
        help="dry-run SHA-256 清单",
    )
    parser.add_argument(
        "--evaluation-exclusions",
        type=Path,
        default=DEFAULT_EVALUATION_EXCLUSIONS,
        help="非空的评估问题排除 JSON",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="新的标准化输出目录")
    parser.add_argument(
        "--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT, help="新的 data manifest"
    )
    args = parser.parse_args()
    try:
        manifest = prepare_disc_law_sft(
            args.input_dir,
            args.cleaning_report,
            args.cleaning_decisions,
            args.cleaning_hashes,
            args.evaluation_exclusions,
            args.output_root,
            args.manifest_output,
        )
    except (SftPreparationError, OSError, ValueError) as error:
        raise SystemExit(f"[失败] {error}") from error
    totals = manifest["records"]["totals"]
    print(
        f"[完成] retain {totals['plan_retain']:,} 条，标准化 {totals['standardized']:,} 条，"
        f"train={totals['train']:,}，full={totals['full_validation']:,}，"
        f"quick={totals['quick_validation']:,}"
    )
    if not manifest["readiness"]["formal_evaluation_isolation_complete"]:
        print("[阻断] 外部评估隔离未完成，本批数据仅为 provisional candidate")
    print(f"data manifest: {args.manifest_output}")


if __name__ == "__main__":
    main()
