"""审计 curated HN 语义冻结候选的 768 token 与标签边界。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from . import audit_disc_law_sft as tokenizer_auditor
    from .rag_sft_v2_dataset import _validate_record
    from .rag_sft_v2_projection import (
        CONTEXT_LIMIT,
        MAX_OUTPUT_TOKENS,
        MAX_PROMPT_TOKENS,
        ProjectedRagSftV2Record,
        RagSftV2ProjectionError,
        audit_projected_rag_sft_v2_record,
    )
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import audit_disc_law_sft as tokenizer_auditor
    from dataset.rag_sft_v2_dataset import _validate_record
    from dataset.rag_sft_v2_projection import (
        CONTEXT_LIMIT,
        MAX_OUTPUT_TOKENS,
        MAX_PROMPT_TOKENS,
        ProjectedRagSftV2Record,
        RagSftV2ProjectionError,
        audit_projected_rag_sft_v2_record,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RELEASE_DIR = (
    PROJECT_ROOT
    / "minimind/dataset/RAG-SFT/releases/v2/"
    "rag-sft-v2-curated-hn-semantic-release-779-v1-20260814"
)
DEFAULT_TOKENIZER = PROJECT_ROOT / "minimind/model"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "minimind/dataset/RAG-SFT/reports/v2/"
    "rag-sft-v2-curated-hn-semantic-release-779-768-audit-v2-20260814"
)

MANIFEST_FILENAME = "manifest.json"
HASH_FILENAME = "manifest.sha256"
LEDGER_FILENAME = "length-audit-ledger.jsonl"
REPORT_FILENAME = "report.json"
README_FILENAME = "README.md"


class CuratedHn768AuditError(RuntimeError):
    """curated HN 768 审计的输入或输出不满足身份约束。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if records is not None:
        result["records"] = records
    return result


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CuratedHn768AuditError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise CuratedHn768AuditError(f"{description}必须是 JSON object")
    return value


def _read_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("rb") as source:
            for number, raw in enumerate(source, start=1):
                if not raw or raw in {b"\n", b"\r\n"}:
                    raise CuratedHn768AuditError(f"{description}不允许空行: {number}")
                if raw.startswith(b"\xef\xbb\xbf"):
                    raise CuratedHn768AuditError(f"{description}不允许 UTF-8 BOM: {number}")
                try:
                    value = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise CuratedHn768AuditError(
                        f"{description}第 {number} 条不是严格 UTF-8 JSON"
                    ) from error
                if not isinstance(value, dict):
                    raise CuratedHn768AuditError(
                        f"{description}第 {number} 条必须是 object"
                    )
                rows.append(value)
    except OSError as error:
        raise CuratedHn768AuditError(f"无法读取{description}: {path}") from error
    if not rows:
        raise CuratedHn768AuditError(f"{description}不能为空")
    return rows


def _verify_release_manifest(release_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = release_dir / MANIFEST_FILENAME
    sidecar_path = release_dir / HASH_FILENAME
    if not manifest_path.is_file() or not sidecar_path.is_file():
        raise CuratedHn768AuditError("输入 release 缺少 manifest 或哈希清单")
    try:
        lines = sidecar_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise CuratedHn768AuditError("无法读取输入 release 哈希清单") from error
    listed: dict[str, str] = {}
    for line in lines:
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise CuratedHn768AuditError("输入 release 哈希清单格式无效")
        try:
            int(parts[0], 16)
        except ValueError as error:
            raise CuratedHn768AuditError("输入 release 哈希清单格式无效") from error
        filename = parts[1]
        if Path(filename).name != filename or filename in listed:
            raise CuratedHn768AuditError("输入 release 哈希清单文件名无效或重复")
        listed[filename] = parts[0].lower()
    expected = {
        "semantic-freeze-candidate.jsonl",
        "curated-hn-variants.jsonl",
        "identity-ledger.jsonl",
        MANIFEST_FILENAME,
        README_FILENAME,
    }
    if set(listed) != expected:
        raise CuratedHn768AuditError("输入 release 哈希清单未覆盖预期文件")
    for filename, expected_sha256 in listed.items():
        path = release_dir / filename
        if not path.is_file() or _sha256(path) != expected_sha256:
            raise CuratedHn768AuditError(f"输入 release 文件哈希不匹配: {filename}")
    manifest = _read_json(manifest_path, "输入 release manifest")
    if (
        manifest.get("pipeline") != "rag_sft_v2_curated_hn_semantic_release"
        or manifest.get("release_status") != "semantic_identity_freeze_only"
        or manifest.get("complete") is not True
    ):
        raise CuratedHn768AuditError("输入 release 状态不是预期的语义冻结版本")
    policy = manifest.get("policy")
    if not isinstance(policy, dict) or policy.get("training_ready") is not False:
        raise CuratedHn768AuditError("输入 release 训练状态无效")
    return manifest, _identity(manifest_path)


def _bound_output_path(
    manifest: dict[str, Any], key: str, release_dir: Path
) -> Path:
    output = manifest.get("output")
    if not isinstance(output, dict) or not isinstance(output.get(key), dict):
        raise CuratedHn768AuditError(f"输入 manifest 缺少 output.{key}")
    metadata = output[key]
    raw_path = metadata.get("path")
    if not isinstance(raw_path, str):
        raise CuratedHn768AuditError(f"输入 manifest output.{key}.path 无效")
    path = Path(raw_path).resolve()
    if path.parent != release_dir or not path.is_file():
        raise CuratedHn768AuditError(f"输入 manifest output.{key} 路径越界或不存在")
    actual = _identity(path)
    if (
        metadata.get("bytes") != actual["bytes"]
        or metadata.get("sha256") != actual["sha256"]
    ):
        raise CuratedHn768AuditError(f"输入 manifest output.{key} 身份不匹配")
    return path


def _bound_input_manifest(manifest: dict[str, Any]) -> tuple[Path, dict[str, Any], dict[str, object]]:
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or not isinstance(inputs.get("base_manifest"), dict):
        raise CuratedHn768AuditError("输入 release 缺少基线 manifest 身份")
    metadata = inputs["base_manifest"]
    raw_path = metadata.get("path")
    if not isinstance(raw_path, str):
        raise CuratedHn768AuditError("输入 release 基线 manifest 路径无效")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise CuratedHn768AuditError("输入 release 基线 manifest 不存在")
    identity = _identity(path)
    if (
        metadata.get("bytes") != identity["bytes"]
        or metadata.get("sha256") != identity["sha256"]
        or metadata.get("manifest_sha256") != identity["sha256"]
    ):
        raise CuratedHn768AuditError("输入 release 基线 manifest 身份不匹配")
    base_manifest = _read_json(path, "基线 manifest")
    if base_manifest.get("pipeline") != "rag_sft_v2_training_release":
        raise CuratedHn768AuditError("输入 release 基线 manifest pipeline 无效")
    return path, base_manifest, identity


def _verify_tokenizer(tokenizer: Any, tokenizer_path: Path, expected: object) -> dict[str, Any]:
    if not isinstance(expected, dict):
        raise CuratedHn768AuditError("输入 release 缺少 Tokenizer 身份")
    actual = tokenizer_auditor._tokenizer_report(tokenizer, tokenizer_path)
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str):
        raise CuratedHn768AuditError("Tokenizer 缺少 chat template")
    actual["chat_template_sha256"] = hashlib.sha256(template.encode("utf-8")).hexdigest()
    comparison = {
        "path": str(tokenizer_path.resolve()),
        "vocab_size": actual.get("vocab_size"),
        "files": actual.get("files"),
        "chat_template_sha256": actual.get("chat_template_sha256"),
    }
    if comparison != expected:
        raise CuratedHn768AuditError("本地 Tokenizer 或 chat template 与冻结身份不一致")
    return comparison


def _source_by_record_id(
    candidate_rows: list[dict[str, Any]], ledger_rows: list[dict[str, Any]]
) -> dict[str, str]:
    curated_ids: set[str] = set()
    curated_queries: set[str] = set()
    for number, row in enumerate(ledger_rows, start=1):
        if set(row) != {"id", "query_id", "variant", "source", "variant_id", "record_sha256"}:
            raise CuratedHn768AuditError(f"身份 ledger 第 {number} 条字段无效")
        if row.get("variant") != "hard_negative" or row.get("source") != "curated":
            raise CuratedHn768AuditError(f"身份 ledger 第 {number} 条来源无效")
        record_id = row.get("id")
        query_id = row.get("query_id")
        if not isinstance(record_id, str) or not isinstance(query_id, str):
            raise CuratedHn768AuditError(f"身份 ledger 第 {number} 条身份无效")
        if record_id in curated_ids or query_id in curated_queries:
            raise CuratedHn768AuditError("身份 ledger 存在重复 curated HN")
        curated_ids.add(record_id)
        curated_queries.add(query_id)
    result: dict[str, str] = {}
    for row in candidate_rows:
        record_id = row["id"]
        if row["variant"] == "clean":
            result[record_id] = "oracle_clean"
        elif record_id in curated_ids:
            result[record_id] = "curated"
        else:
            result[record_id] = "retrieved"
    seen_candidate_ids = {row["id"] for row in candidate_rows}
    if not curated_ids.issubset(seen_candidate_ids):
        raise CuratedHn768AuditError("身份 ledger 含不存在的 curated HN")
    return result


def _error_reason(error: Exception) -> str:
    message = str(error)
    if "prompt 超过" in message:
        return "prompt_exceeds_618"
    if "assistant JSON 超过" in message:
        return "assistant_exceeds_150"
    if "超过 768" in message:
        return "full_sequence_exceeds_768"
    if "prompt 前缀" in message:
        return "chat_template_prefix_mismatch"
    return "audit_error"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                + "\n"
            )


def _write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")


def audit(*, release_dir: Path, tokenizer_path: Path, output_dir: Path) -> dict[str, object]:
    """逐条审计，不截断、不重写、不物化正式训练 HN。"""

    release_dir = release_dir.resolve()
    tokenizer_path = tokenizer_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise CuratedHn768AuditError(f"审计输出目录必须是新目录: {output_dir}")
    manifest, manifest_identity = _verify_release_manifest(release_dir)
    _, base_manifest, base_manifest_identity = _bound_input_manifest(manifest)
    candidate_path = _bound_output_path(manifest, "semantic_freeze_candidate", release_dir)
    ledger_path = _bound_output_path(manifest, "identity_ledger", release_dir)
    candidate_rows = _read_jsonl(candidate_path, "语义冻结 candidate")
    ledger_rows = _read_jsonl(ledger_path, "语义冻结 identity ledger")
    records = manifest.get("records")
    if not isinstance(records, dict) or len(candidate_rows) != records.get("total"):
        raise CuratedHn768AuditError("语义冻结 candidate 数量与 manifest 不闭合")
    normalized_rows: list[dict[str, Any]] = []
    record_ids: set[str] = set()
    for number, row in enumerate(candidate_rows, start=1):
        normalized = _validate_record(row, number)
        if normalized["id"] in record_ids:
            raise CuratedHn768AuditError("语义冻结 candidate 存在重复 record id")
        record_ids.add(normalized["id"])
        normalized_rows.append(normalized)
    source_by_id = _source_by_record_id(normalized_rows, ledger_rows)
    tokenizer = tokenizer_auditor.load_tokenizer(tokenizer_path)
    tokenizer_identity = _verify_tokenizer(tokenizer, tokenizer_path, base_manifest.get("tokenizer"))

    audit_rows: list[dict[str, object]] = []
    by_variant: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    passed_by_variant: Counter[str] = Counter()
    passed_by_source: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    prompt_histogram: Counter[int] = Counter()
    assistant_histogram: Counter[int] = Counter()
    total_histogram: Counter[int] = Counter()
    for row in normalized_rows:
        record_id = row["id"]
        variant = row["variant"]
        source = source_by_id[record_id]
        by_variant[variant] += 1
        by_source[source] += 1
        projected = ProjectedRagSftV2Record(
            record_id=record_id,
            query_id=row["query_id"],
            variant=variant,
            visible_chunk_ids=tuple(row["visible_chunk_ids"]),
            required_chunk_ids=tuple(row["required_chunk_ids"]),
            citations=tuple(row["citations"]),
            conversations=tuple(row["conversations"]),
        )
        base: dict[str, object] = {
            "id": record_id,
            "query_id": row["query_id"],
            "variant": variant,
            "source": source,
            "visible_evidence_count": len(row["visible_chunk_ids"]),
            "required_gt_count": len(row["required_chunk_ids"]),
        }
        try:
            result = audit_projected_rag_sft_v2_record(projected, tokenizer)
            active = [token for token in result.labels if token != -100]
            labels_correct = (
                len(result.input_ids) == CONTEXT_LIMIT
                and len(result.labels) == CONTEXT_LIMIT
                and len(active) == result.assistant_label_tokens
                and active == list(result.input_ids[result.prompt_tokens : result.total_tokens])
                and all(token == -100 for token in result.labels[: result.prompt_tokens])
                and all(token == -100 for token in result.labels[result.total_tokens :])
            )
            if not labels_correct:
                raise CuratedHn768AuditError("assistant-only labels 与 EOS 边界不闭合")
            base.update(
                {
                    "prompt_tokens": result.prompt_tokens,
                    "assistant_tokens": result.assistant_label_tokens,
                    "total_tokens": result.total_tokens,
                    "assistant_only_labels_correct": True,
                    "decision": "keep",
                    "reason": "within_768_and_labels_closed",
                }
            )
            passed_by_variant[variant] += 1
            passed_by_source[source] += 1
            prompt_histogram[result.prompt_tokens] += 1
            assistant_histogram[result.assistant_label_tokens] += 1
            total_histogram[result.total_tokens] += 1
        except (RagSftV2ProjectionError, CuratedHn768AuditError) as error:
            reason = _error_reason(error)
            base.update(
                {
                    "prompt_tokens": None,
                    "assistant_tokens": None,
                    "total_tokens": None,
                    "assistant_only_labels_correct": False,
                    "decision": "exclude",
                    "reason": reason,
                    "detail": str(error),
                }
            )
            reason_counts[reason] += 1
        audit_rows.append(base)

    if len(audit_rows) != len(normalized_rows):
        raise CuratedHn768AuditError("长度 ledger 与 candidate 覆盖不闭合")
    count_by_variant = dict(sorted(by_variant.items()))
    count_by_source = dict(sorted(by_source.items()))
    keep_by_variant = dict(sorted(passed_by_variant.items()))
    keep_by_source = dict(sorted(passed_by_source.items()))
    report: dict[str, object] = {
        "pipeline": "rag_sft_v2_curated_hn_768_audit",
        "release_status": "length_and_label_audit_only",
        "inputs": {
            "semantic_freeze_manifest": manifest_identity,
            "base_training_manifest": base_manifest_identity,
            "semantic_freeze_candidate": _identity(candidate_path, records=len(normalized_rows)),
            "semantic_freeze_identity_ledger": _identity(ledger_path, records=len(ledger_rows)),
        },
        "tokenizer": tokenizer_identity,
        "policy": {
            "context_limit": CONTEXT_LIMIT,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "truncation": "forbidden",
            "clean_overlimit_policy": "exclude_clean_and_all_hn_for_same_query_in_next_release",
            "hn_overlimit_policy": "keep_clean_exclude_hn_and_seek_audited_replacement",
            "formal_hn_materialized": False,
            "training_ready": False,
        },
        "records": {
            "total": len(audit_rows),
            "by_variant": count_by_variant,
            "by_source": count_by_source,
            "keep": len(audit_rows) - sum(reason_counts.values()),
            "exclude": sum(reason_counts.values()),
            "keep_by_variant": keep_by_variant,
            "keep_by_source": keep_by_source,
            "exclude_reasons": dict(sorted(reason_counts.items())),
            "prompt_token_histogram": {str(key): prompt_histogram[key] for key in sorted(prompt_histogram)},
            "assistant_token_histogram": {str(key): assistant_histogram[key] for key in sorted(assistant_histogram)},
            "total_token_histogram": {str(key): total_histogram[key] for key in sorted(total_histogram)},
        },
        "validation": {
            "strict_utf8_jsonl": True,
            "input_manifest_hashes_verified": True,
            "input_record_ids_unique": True,
            "all_779_records_audited_once": len(audit_rows) == 779,
            "all_160_curated_identity_ledger_rows_bound": len(ledger_rows) == 160,
            "assistant_only_labels_checked_for_each_kept_record": True,
            "source_assets_modified": False,
        },
        "outputs": {
            "length_audit_ledger": LEDGER_FILENAME,
            "report": REPORT_FILENAME,
            "sha256_manifest": HASH_FILENAME,
        },
        "complete": True,
    }
    output_dir.mkdir(parents=True)
    _write_jsonl(output_dir / LEDGER_FILENAME, audit_rows)
    _write_json(output_dir / REPORT_FILENAME, report)
    (output_dir / README_FILENAME).write_text(
        "# RAG-SFT v2 curated HN 768 审计\n\n"
        "本目录只记录 779 条语义冻结候选的长度与 assistant-only labels 审计；"
        "不截断、不改写、不物化正式 HN，`training_ready=false`。\n",
        encoding="utf-8",
        newline="\n",
    )
    audit_manifest = {
        "pipeline": report["pipeline"],
        "release_status": report["release_status"],
        "inputs": report["inputs"],
        "tokenizer": report["tokenizer"],
        "policy": report["policy"],
        "records": report["records"],
        "validation": report["validation"],
        "output": {
            "length_audit_ledger": _identity(
                output_dir / LEDGER_FILENAME, records=len(audit_rows)
            ),
            "report": _identity(output_dir / REPORT_FILENAME),
        },
        "complete": True,
    }
    _write_json(output_dir / MANIFEST_FILENAME, audit_manifest)
    (output_dir / HASH_FILENAME).write_text(
        "".join(
            f"{_sha256(output_dir / filename)}  {filename}\n"
            for filename in (
                LEDGER_FILENAME,
                REPORT_FILENAME,
                README_FILENAME,
                MANIFEST_FILENAME,
            )
        ),
        encoding="ascii",
        newline="\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="审计 curated HN 语义冻结 release 的 768 token 预算")
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        report = audit(
            release_dir=args.release_dir,
            tokenizer_path=args.tokenizer,
            output_dir=args.output_dir,
        )
    except CuratedHn768AuditError as error:
        raise SystemExit(f"RAG_SFT_V2_CURATED_HN_768_AUDIT_FAILED: {error}") from error
    print(
        "RAG_SFT_V2_CURATED_HN_768_AUDIT_OK "
        f"total={report['records']['total']} keep={report['records']['keep']} "
        f"exclude={report['records']['exclude']} training_ready=false"
    )


if __name__ == "__main__":
    main()
