"""冻结 CPT 来源语料，完成跨来源去重和确定性数据划分。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .audit_pretrain_samples import load_tokenizer, sha256_file
from .prepare_stage_b_corpus import (
    JsonlShardWriter,
    PreparationError,
    ensure_empty_outputs,
    tokenizer_metadata,
)


SOURCE_PRIORITY = ("npc_flk", "fuzi_mingcha")
FULL_VALIDATION_BUCKETS = 50
QUICK_VALIDATION_BUCKETS = 5
BUCKET_COUNT = 10_000
TARGET_FILE_BYTES = 256 * 1024 * 1024
PROGRESS_INTERVAL = 100_000
WHITESPACE_RE = re.compile(r"\s+")
DIGEST_RE = re.compile(r"[0-9a-f]{64}")
SOURCE_FIELDS = {
    "npc_flk": {"id", "source", "title", "text"},
    "fuzi_mingcha": {
        "id",
        "source",
        "source_file",
        "year",
        "month",
        "case_type",
        "text",
    },
}


class CptPreparationError(PreparationError):
    """表示 CPT 冻结语料无法安全生成。"""


def document_signature(text: str) -> bytes:
    """计算折叠内部空白后的正文 SHA-256。"""
    normalized = WHITESPACE_RE.sub(" ", text).strip()
    return hashlib.sha256(normalized.encode("utf-8")).digest()


def validation_bucket(document_id: str) -> int:
    """将正文摘要稳定映射到万分桶。"""
    return int(document_id[:16], 16) % BUCKET_COUNT


def _write_manifest(manifest: dict[str, object], output_path: Path) -> None:
    """原子发布 clean manifest 及其 SHA-256。"""
    hash_path = output_path.with_suffix(".sha256")
    manifest_partial = output_path.with_name(output_path.name + ".partial")
    hash_partial = hash_path.with_name(hash_path.name + ".partial")
    payload = (
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_partial.open("xb") as file:
            file.write(payload)
        json.loads(manifest_partial.read_text(encoding="utf-8"))
        digest = sha256_file(manifest_partial)
        with hash_partial.open("xb") as file:
            file.write(f"{digest}  {output_path.name}\n".encode("utf-8"))
        manifest_partial.replace(output_path)
        hash_partial.replace(hash_path)
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise CptPreparationError(f"无法发布 CPT clean manifest: {output_path}") from error


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CptPreparationError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise CptPreparationError(f"{description}必须是 JSON object: {path}")
    return value


def _load_exclusion_digests(
    path: Path | None,
) -> tuple[set[bytes], dict[str, object]]:
    """加载评估资产的正文摘要清单并冻结其文件身份。"""
    if path is None:
        return set(), {
            "enabled": False,
            "path": None,
            "bytes": 0,
            "sha256": None,
            "digest_count": 0,
            "normalization": "collapse_whitespace_then_sha256",
            "complete": True,
        }
    path = path.resolve()
    if not path.is_file():
        raise CptPreparationError(f"评估排除摘要文件不存在: {path}")
    digests: set[bytes] = set()
    try:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            value = line.strip().lower()
            if not value:
                continue
            if DIGEST_RE.fullmatch(value) is None:
                raise CptPreparationError(
                    f"评估排除摘要格式无效: {path}:{line_number}"
                )
            digests.add(bytes.fromhex(value))
    except (OSError, UnicodeDecodeError) as error:
        raise CptPreparationError(f"无法读取评估排除摘要文件: {path}") from error
    if not digests:
        raise CptPreparationError("评估排除摘要文件不能为空")
    return digests, {
        "enabled": True,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "digest_count": len(digests),
        "normalization": "collapse_whitespace_then_sha256",
        "complete": True,
    }


def _validate_wenshu_audit(corpus_path: Path, audit_path: Path) -> dict[str, Any]:
    audit = _load_json(audit_path.resolve(), "裁判文书最终审计报告")
    if audit.get("complete") is not True or audit.get("passed") is not True:
        raise CptPreparationError("裁判文书最终审计报告尚未通过")
    if audit.get("audit_mode") != "full_structure_sampled_quality_v1":
        raise CptPreparationError("裁判文书最终审计模式不受支持")
    if audit.get("input_file") != corpus_path.name:
        raise CptPreparationError("裁判文书审计报告记录了错误的输入文件名")
    if audit.get("input_bytes") != corpus_path.stat().st_size:
        raise CptPreparationError("裁判文书审计报告记录的文件大小不一致")
    actual_digest = sha256_file(corpus_path)
    if audit.get("input_sha256") != actual_digest:
        raise CptPreparationError("裁判文书审计报告记录的 SHA-256 不一致")
    valid_records = audit.get("valid_records")
    if isinstance(valid_records, bool) or not isinstance(valid_records, int) or valid_records <= 0:
        raise CptPreparationError("裁判文书审计报告的 valid_records 无效")
    return {
        "path": str(audit_path.resolve()),
        "sha256": sha256_file(audit_path.resolve()),
        "input_sha256": actual_digest,
        "valid_records": valid_records,
        "audit_mode": "full_structure_sampled_quality_v1",
        "passed": True,
        "complete": True,
    }


def _validate_record(record: object, source: str, location: str) -> tuple[str, str, str]:
    if not isinstance(record, dict) or set(record) != SOURCE_FIELDS[source]:
        raise CptPreparationError(f"来源标准化记录字段无效: {location}")
    if record.get("source") != source:
        raise CptPreparationError(f"来源标准化记录 source 不一致: {location}")
    origin_id = record.get("id")
    text = record.get("text")
    if not isinstance(origin_id, str) or not origin_id:
        raise CptPreparationError(f"来源标准化记录 id 无效: {location}")
    if not isinstance(text, str) or not text.strip():
        raise CptPreparationError(f"来源标准化记录 text 无效: {location}")
    if source == "fuzi_mingcha":
        year = record.get("year")
        case_type = record.get("case_type")
        if isinstance(year, bool) or not isinstance(year, int):
            raise CptPreparationError(f"裁判文书 year 无效: {location}")
        if not isinstance(case_type, str) or not case_type:
            raise CptPreparationError(f"裁判文书 case_type 无效: {location}")
        stratum = f"{year}/{case_type}"
    else:
        stratum = source
    return origin_id, text, stratum


def _empty_split_counts() -> Counter[str]:
    return Counter(records=0, characters=0, text_utf8_bytes=0)


def _record_size(counts: Counter[str], text: str) -> None:
    counts["records"] += 1
    counts["characters"] += len(text)
    counts["text_utf8_bytes"] += len(text.encode("utf-8"))


def _process_source(
    source: str,
    input_path: Path,
    input_sha256: str,
    output_root: Path,
    target_file_bytes: int,
    connection: sqlite3.Connection,
    exclusion_digests: set[bytes],
) -> dict[str, object]:
    """冻结一个来源，并返回闭合的互斥漏斗和分层统计。"""
    writers = {
        "train": JsonlShardWriter(output_root, "train", source, target_file_bytes),
        "validation": JsonlShardWriter(
            output_root, "validation", source, target_file_bytes
        ),
    }
    records = Counter(
        scanned=0, retained=0, train=0, full_validation=0, quick_validation=0
    )
    filtered = Counter(
        evaluation_duplicate=0,
        duplicate_within_source=0,
        duplicate_higher_priority=0,
    )
    sizes = {
        "train": _empty_split_counts(),
        "full_validation": _empty_split_counts(),
        "quick_validation": _empty_split_counts(),
    }
    strata: dict[str, Counter[str]] = defaultdict(
        lambda: Counter(retained=0, train=0, full_validation=0, quick_validation=0)
    )
    try:
        with input_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                records["scanned"] += 1
                location = f"{input_path}:{line_number}"
                if not line.strip():
                    raise CptPreparationError(f"来源标准化 JSONL 含空行: {location}")
                try:
                    raw_record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise CptPreparationError(
                        f"来源标准化 JSONL 行不是有效 JSON: {location}"
                    ) from error
                _, text, stratum = _validate_record(raw_record, source, location)
                signature = document_signature(text)
                if signature in exclusion_digests:
                    filtered["evaluation_duplicate"] += 1
                    continue
                cursor = connection.execute(
                    "INSERT OR IGNORE INTO seen(digest, source) VALUES (?, ?)",
                    (signature, source),
                )
                if cursor.rowcount == 0:
                    retained_source = connection.execute(
                        "SELECT source FROM seen WHERE digest = ?", (signature,)
                    ).fetchone()[0]
                    reason = (
                        "duplicate_within_source"
                        if retained_source == source
                        else "duplicate_higher_priority"
                    )
                    filtered[reason] += 1
                    continue

                document_id = signature.hex()
                bucket = validation_bucket(document_id)
                split = "validation" if bucket < FULL_VALIDATION_BUCKETS else "train"
                is_quick = bucket < QUICK_VALIDATION_BUCKETS
                writers[split].write(
                    {"id": document_id, "source": source, "text": text}
                )
                records["retained"] += 1
                strata[stratum]["retained"] += 1
                if split == "train":
                    records["train"] += 1
                    strata[stratum]["train"] += 1
                    _record_size(sizes["train"], text)
                else:
                    records["full_validation"] += 1
                    strata[stratum]["full_validation"] += 1
                    _record_size(sizes["full_validation"], text)
                    if is_quick:
                        records["quick_validation"] += 1
                        strata[stratum]["quick_validation"] += 1
                        _record_size(sizes["quick_validation"], text)
                if records["scanned"] % PROGRESS_INTERVAL == 0:
                    print(
                        f"[CPT clean] {source}: 扫描 {records['scanned']:,}，"
                        f"保留 {records['retained']:,}",
                        flush=True,
                    )
                if records["scanned"] % 100_000 == 0:
                    connection.commit()
        connection.commit()
        output_files = {
            split: writer.finish() for split, writer in writers.items()
        }
    except Exception:
        for writer in writers.values():
            writer.abort()
        raise

    excluded = sum(filtered.values())
    if records["scanned"] != records["retained"] + excluded:
        raise CptPreparationError(f"{source} 清洗漏斗未闭合")
    if records["retained"] != records["train"] + records["full_validation"]:
        raise CptPreparationError(f"{source} 数据划分未闭合")
    if not records["train"] or not records["full_validation"] or not records["quick_validation"]:
        raise CptPreparationError(f"{source} 的 train/full/quick validation 必须均非空")
    return {
        "input": {
            "path": str(input_path),
            "bytes": input_path.stat().st_size,
            "sha256": input_sha256,
        },
        "records": {**dict(records), "filtered": dict(filtered)},
        "size": {name: dict(counts) for name, counts in sizes.items()},
        "strata": {name: dict(counts) for name, counts in sorted(strata.items())},
        "output_files": output_files,
    }


def prepare_cpt_corpus(
    npc_corpus: Path,
    wenshu_corpus: Path,
    wenshu_audit: Path,
    evaluation_exclusions: Path | None,
    tokenizer_path: Path,
    output_root: Path,
    work_dir: Path,
    manifest_output: Path,
    tokenizer: Any | None = None,
    target_file_bytes: int = TARGET_FILE_BYTES,
) -> dict[str, object]:
    """生成 CPT clean JSONL、分层统计和冻结 manifest。"""
    paths = [npc_corpus, wenshu_corpus, tokenizer_path, output_root, work_dir, manifest_output]
    npc_corpus, wenshu_corpus, tokenizer_path, output_root, work_dir, manifest_output = (
        path.resolve() for path in paths
    )
    if target_file_bytes <= 0:
        raise CptPreparationError("target_file_bytes 必须大于零")
    if not npc_corpus.is_file() or not wenshu_corpus.is_file():
        raise CptPreparationError("缺少 NPC 或裁判文书来源标准化语料")
    ensure_empty_outputs(output_root, manifest_output)
    database_path = work_dir / "cpt-clean-dedup.sqlite3"
    if database_path.exists():
        raise CptPreparationError(f"CPT clean 工作数据库已存在: {database_path}")

    exclusion_digests, exclusion_report = _load_exclusion_digests(
        evaluation_exclusions
    )
    audit_report = _validate_wenshu_audit(wenshu_corpus, wenshu_audit)
    npc_sha256 = sha256_file(npc_corpus)
    if tokenizer is None:
        tokenizer = load_tokenizer(tokenizer_path)
    tokenizer_report, _ = tokenizer_metadata(tokenizer, tokenizer_path)

    work_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            "CREATE TABLE seen (digest BLOB PRIMARY KEY, source TEXT NOT NULL) WITHOUT ROWID"
        )
        source_reports = {
            "npc_flk": _process_source(
                "npc_flk",
                npc_corpus,
                npc_sha256,
                output_root,
                target_file_bytes,
                connection,
                exclusion_digests,
            ),
            "fuzi_mingcha": _process_source(
                "fuzi_mingcha",
                wenshu_corpus,
                str(audit_report["input_sha256"]),
                output_root,
                target_file_bytes,
                connection,
                exclusion_digests,
            ),
        }
    finally:
        connection.close()

    totals = Counter()
    for report in source_reports.values():
        for key in ("scanned", "retained", "train", "full_validation", "quick_validation"):
            totals[key] += int(report["records"][key])
    manifest = {
        "schema_version": "1.0",
        "pipeline": "cpt",
        "source_priority": list(SOURCE_PRIORITY),
        "wenshu_audit": audit_report,
        "evaluation_exclusion": exclusion_report,
        "deduplication": {
            "normalization": "collapse_whitespace",
            "digest": "sha256",
            "priority_rule": "first_source_wins",
            "database": str(database_path),
        },
        "validation": {
            "bucket_expression": "int(id[:16], 16) % 10000",
            "full_validation_buckets": "0-49",
            "quick_validation_buckets": "0-4",
            "quick_is_subset_of_full": True,
            "strata": "source; fuzi_mingcha additionally reports year/case_type",
        },
        "tokenizer": tokenizer_report,
        "output": {
            "root": str(output_root),
            "record_fields": ["id", "source", "text"],
        },
        "sources": source_reports,
        "totals": {"records": dict(totals)},
        "complete": True,
    }
    _write_manifest(manifest, manifest_output)
    return manifest


def main() -> None:
    """解析参数并生成 CPT clean manifest。"""
    parser = argparse.ArgumentParser(description="生成 CPT 冻结 clean 语料与 manifest")
    parser.add_argument("--npc-corpus", type=Path, required=True, help="NPC 标准化 JSONL")
    parser.add_argument(
        "--wenshu-corpus", type=Path, required=True, help="裁判文书 v2 标准化 JSONL"
    )
    parser.add_argument(
        "--wenshu-audit", type=Path, required=True, help="裁判文书 v2 最终审计报告"
    )
    parser.add_argument(
        "--evaluation-exclusions",
        type=Path,
        help="可选的评估资产正文 SHA-256 排除清单；不传表示当前未启用",
    )
    parser.add_argument("--tokenizer-path", type=Path, required=True, help="当前 Tokenizer")
    parser.add_argument("--output-root", type=Path, required=True, help="CPT clean 输出目录")
    parser.add_argument("--work-dir", type=Path, required=True, help="去重数据库工作目录")
    parser.add_argument("--manifest-output", type=Path, required=True, help="clean manifest")
    args = parser.parse_args()
    report = prepare_cpt_corpus(
        args.npc_corpus,
        args.wenshu_corpus,
        args.wenshu_audit,
        args.evaluation_exclusions,
        args.tokenizer_path,
        args.output_root,
        args.work_dir,
        args.manifest_output,
    )
    totals = report["totals"]["records"]
    print(
        f"[完成] 扫描 {totals['scanned']:,} 条，保留 {totals['retained']:,} 条，"
        f"train={totals['train']:,}，full={totals['full_validation']:,}，"
        f"quick={totals['quick_validation']:,}"
    )
    print(f"manifest: {args.manifest_output}")


if __name__ == "__main__":
    main()
