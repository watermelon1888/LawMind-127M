"""将已批准的 RAG-SFT v2 canonical authoring 物化为阶段性 Oracle clean。"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from rag.eval.private_holdout import audit_private_holdout_compatibility
from rag.knowledge import ArticleRepository

from . import audit_disc_law_sft as tokenizer_auditor
from .build_sft_evaluation_exclusions import question_digest
from .rag_sft_v2_contract import RagSftV2ContractError, validate_canonical_authoring
from .rag_sft_v2_evaluation import build_evaluation_target
from .rag_sft_v2_projection import (
    CONTEXT_LIMIT,
    MAX_OUTPUT_TOKENS,
    MAX_PROMPT_TOKENS,
    RagSftV2ProjectionError,
    audit_projected_rag_sft_v2_record,
    project_rag_sft_v2_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = Path(__file__).resolve().parent
RAG_SFT_ROOT = DATASET_ROOT / "RAG-SFT"
DEFAULT_APPROVED_ROOT = (
    RAG_SFT_ROOT / "review" / "v2" / "canonical-authoring-approved-v1"
)
DEFAULT_QUERY_POOL = DATASET_ROOT / "QUERY-POOL" / "authoring" / "query-pool-v1.jsonl"
DEFAULT_ARTICLE_INDEX = PROJECT_ROOT / "rag" / "chunk" / "article_index.jsonl"
DEFAULT_EVALUATION_EXCLUSIONS = (
    RAG_SFT_ROOT / "manifests" / "evaluation-exclusions-project-rag-v2.json"
)
DEFAULT_DEVELOPMENT_EVAL = PROJECT_ROOT / "rag" / "eval" / "eval_set.jsonl"
DEFAULT_PRIVATE_HOLDOUT = (
    PROJECT_ROOT.parent / "private-eval" / "authoring" / "private-holdout-v2.jsonl"
)
DEFAULT_TOKENIZER = PROJECT_ROOT / "minimind" / "model"
DEFAULT_OUTPUT_DIR = RAG_SFT_ROOT / "review" / "v2" / "oracle-clean-v1"
DEFAULT_CORRECTIONS = (
    RAG_SFT_ROOT
    / "review"
    / "v2"
    / "canonical-authoring-corrections-v1"
    / "corrections.jsonl"
)

ORACLE_CLEAN_FILENAME = "oracle-clean.jsonl"
EVALUATION_TARGETS_FILENAME = "evaluation-targets.jsonl"
OVERBUDGET_FILENAME = "overbudget-exclusions.jsonl"
ISOLATION_FILENAME = "evaluation-isolation.json"
MANIFEST_FILENAME = "manifest.json"
HASH_FILENAME = "manifest.sha256"
CANONICAL_FIELDS = {"query_id", "query_original", "claims", "summary"}
POOL_FIELDS = {"query_id", "query_original", "required_chunk_ids"}
EXPECTED_BATCHES = tuple(f"batch-{number:03d}" for number in range(1, 20))
CORRECTION_FIELDS = {"query_id", "source_summary_sha256", "summary", "reason"}


class RagSftV2OracleCleanError(RuntimeError):
    """Oracle clean 的输入身份、投影或发布状态不满足阶段要求。"""


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


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftV2OracleCleanError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise RagSftV2OracleCleanError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(path: Path, description: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagSftV2OracleCleanError(f"{description}不允许空行: {number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise RagSftV2OracleCleanError(
                        f"{description}第 {number} 条必须是对象"
                    )
                rows.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftV2OracleCleanError):
            raise
        raise RagSftV2OracleCleanError(f"无法读取{description}: {path}") from error
    if not rows:
        raise RagSftV2OracleCleanError(f"{description}不能为空")
    return rows


def _verify_hash_manifest(directory: Path) -> None:
    hash_path = directory / HASH_FILENAME
    expected_paths = {
        "canonical-authoring.jsonl",
        MANIFEST_FILENAME,
    }
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftV2OracleCleanError("canonical 批次缺少哈希清单") from error
    parsed: dict[str, str] = {}
    for number, line in enumerate(lines, start=1):
        parts = line.split("  ", 1)
        if (
            len(parts) != 2
            or len(parts[0]) != 64
            or any(character not in "0123456789abcdef" for character in parts[0])
            or Path(parts[1]).name != parts[1]
            or parts[1] in parsed
        ):
            raise RagSftV2OracleCleanError(f"canonical 批次哈希格式无效: {number}")
        parsed[parts[1]] = parts[0]
    if set(parsed) != expected_paths:
        raise RagSftV2OracleCleanError("canonical 批次哈希文件集合无效")
    for name, expected in parsed.items():
        path = directory / name
        if not path.is_file() or _sha256(path) != expected:
            raise RagSftV2OracleCleanError(f"canonical 批次哈希不一致: {name}")


def _load_pool(path: Path) -> dict[str, dict[str, Any]]:
    rows = _load_jsonl(path, "公共 Query 池")
    result: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(rows, start=1):
        if set(row) != POOL_FIELDS or not isinstance(row.get("query_id"), str):
            raise RagSftV2OracleCleanError(f"公共 Query 池字段无效: {number}")
        if row["query_id"] in result:
            raise RagSftV2OracleCleanError("公共 Query 池 query_id 重复")
        result[row["query_id"]] = row
    return result


def _load_article_contents(path: Path) -> dict[str, str]:
    rows = _load_jsonl(path, "法条索引")
    result: dict[str, str] = {}
    for number, row in enumerate(rows, start=1):
        chunk_id, content = row.get("chunk_id"), row.get("content")
        if not isinstance(chunk_id, str) or not isinstance(content, str) or chunk_id in result:
            raise RagSftV2OracleCleanError(f"法条索引记录无效: {number}")
        result[chunk_id] = content
    return result


def _load_canonical_corrections(
    path: Path, canonical_records: Iterable[dict[str, object]]
) -> dict[str, dict[str, str]]:
    """加载只允许替换运行时协议冲突摘要的不可变修订账本。"""

    rows = _load_jsonl(path, "canonical 摘要修订账本")
    canonical_by_id = {
        record["query_id"]: record
        for record in canonical_records
        if isinstance(record.get("query_id"), str)
    }
    corrections: dict[str, dict[str, str]] = {}
    for number, row in enumerate(rows, start=1):
        if set(row) != CORRECTION_FIELDS:
            raise RagSftV2OracleCleanError(f"修订账本字段无效: {number}")
        values: dict[str, str] = {}
        for field in CORRECTION_FIELDS:
            value = row[field]
            if not isinstance(value, str) or not value.strip():
                raise RagSftV2OracleCleanError(f"修订账本字段必须是非空字符串: {number}.{field}")
            values[field] = value
        query_id = values["query_id"]
        canonical = canonical_by_id.get(query_id)
        if canonical is None or query_id in corrections:
            raise RagSftV2OracleCleanError(f"修订账本 query_id 无效或重复: {query_id}")
        source_summary = canonical.get("summary")
        if not isinstance(source_summary, str):
            raise RagSftV2OracleCleanError(f"canonical summary 无效: {query_id}")
        actual_hash = hashlib.sha256(source_summary.encode("utf-8")).hexdigest()
        if values["source_summary_sha256"] != actual_hash:
            raise RagSftV2OracleCleanError(f"修订账本源摘要哈希不匹配: {query_id}")
        if values["summary"] == source_summary:
            raise RagSftV2OracleCleanError(f"修订账本未实际修改摘要: {query_id}")
        corrections[query_id] = values
    return corrections


def _project_canonical_with_correction(
    canonical: dict[str, object],
    article_by_chunk_id: dict[str, object],
    correction: dict[str, str] | None,
):
    """先证明原摘要确有协议冲突，再投影仅替换摘要后的副本。"""

    if correction is None:
        return project_rag_sft_v2_record(canonical, article_by_chunk_id)
    try:
        project_rag_sft_v2_record(canonical, article_by_chunk_id)
    except RagSftV2ProjectionError as error:
        if str(error) != "canonical summary 不满足两字段回答协议":
            raise RagSftV2OracleCleanError(
                f"修订账本不能用于非协议冲突: {canonical['query_id']}"
            ) from error
    else:
        raise RagSftV2OracleCleanError(
            f"修订账本不能修改原本可投影摘要: {canonical['query_id']}"
        )
    corrected = dict(canonical)
    corrected["summary"] = correction["summary"]
    return project_rag_sft_v2_record(corrected, article_by_chunk_id)


def _load_canonical_records(
    approved_root: Path,
    pool: dict[str, dict[str, Any]],
    contents: dict[str, str],
    *,
    query_pool_path: Path,
    article_index_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    approved_root = approved_root.resolve()
    if not approved_root.is_dir():
        raise RagSftV2OracleCleanError("canonical 批次目录不存在")
    directories = sorted(path.name for path in approved_root.iterdir() if path.is_dir())
    if tuple(directories) != EXPECTED_BATCHES:
        raise RagSftV2OracleCleanError("canonical 批次目录集合不符合已冻结范围")
    expected_pool = _identity(query_pool_path, records=len(pool))
    expected_index = _identity(article_index_path, records=len(contents))
    records: list[dict[str, object]] = []
    batch_identities: list[dict[str, object]] = []
    seen: set[str] = set()
    for name in EXPECTED_BATCHES:
        directory = approved_root / name
        _verify_hash_manifest(directory)
        manifest_path = directory / MANIFEST_FILENAME
        canonical_path = directory / "canonical-authoring.jsonl"
        manifest = _load_json(manifest_path, f"{name} manifest")
        manifest_records = manifest.get("records")
        readiness = manifest.get("readiness")
        inputs = manifest.get("inputs")
        is_v1_batch = (
            manifest.get("pipeline") == "rag_sft_v2_canonical_batch_finalization"
            and manifest.get("release_status") == "canonical_batch_approved"
            and readiness.get("canonical_batch_approved") is True
        )
        is_snapshot = (
            manifest.get("pipeline") == "rag_sft_v2_canonical_snapshot"
            and manifest.get("release_status") == "canonical_snapshot_approved"
            and readiness.get("canonical_snapshot_approved") is True
        )
        if (
            not (is_v1_batch or is_snapshot)
            or manifest.get("complete") is not True
            or not isinstance(manifest_records, dict)
            or not isinstance(readiness, dict)
            or not isinstance(inputs, dict)
            or readiness.get("training_ready") is not False
            or readiness.get("oracle_clean_emitted") is not False
            or inputs.get("query_pool") != expected_pool
            or inputs.get("article_index") != expected_index
        ):
            raise RagSftV2OracleCleanError(f"{name} manifest 身份或状态无效")
        batch_rows = _load_jsonl(canonical_path, f"{name} canonical authoring")
        if manifest_records.get("canonical_authoring") != len(batch_rows):
            raise RagSftV2OracleCleanError(f"{name} canonical 计数不闭合")
        for row in batch_rows:
            query_id = row.get("query_id")
            if (
                set(row) != CANONICAL_FIELDS
                or not isinstance(query_id, str)
                or query_id in seen
                or query_id not in pool
            ):
                raise RagSftV2OracleCleanError(f"{name} canonical query_id 无效或重复")
            try:
                validated = validate_canonical_authoring(row, pool[query_id], contents)
            except RagSftV2ContractError as error:
                raise RagSftV2OracleCleanError(
                    f"{name} canonical 未通过契约: {query_id}"
                ) from error
            records.append(validated)
            seen.add(query_id)
        batch_identities.append(
            {
                "batch": name,
                "manifest": _identity(manifest_path),
                "sha256_manifest": _identity(directory / HASH_FILENAME),
                "canonical_authoring": _identity(canonical_path, records=len(batch_rows)),
            }
        )
    if len(records) != 550:
        raise RagSftV2OracleCleanError("canonical authoring 总数不是 550")
    return records, batch_identities


def _verify_formal_exclusions(path: Path) -> dict[str, Any]:
    manifest = _load_json(path, "评估隔离清单")
    hash_path = path.with_suffix(".sha256")
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftV2OracleCleanError("评估隔离清单哈希缺失") from error
    if lines != [f"{_sha256(path)}  {path.name}"]:
        raise RagSftV2OracleCleanError("评估隔离清单哈希无效")
    if (
        manifest.get("pipeline") != "legal_sft_evaluation_exclusions"
        or manifest.get("complete_for_formal_sft") is not True
        or manifest.get("raw_questions_included") is not False
        or not isinstance(manifest.get("question_sha256"), list)
        or not all(isinstance(value, str) and len(value) == 64 for value in manifest["question_sha256"])
    ):
        raise RagSftV2OracleCleanError("评估隔离清单不是正式私有留出集版本")
    return manifest


def _serialize_jsonl(records: Iterable[dict[str, object]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for record in records
    )


def _identity_from_payload(path: Path, payload: str, *, records: int) -> dict[str, object]:
    encoded = payload.encode("utf-8")
    return {
        "path": str(path.resolve()),
        "bytes": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "records": records,
    }


def _publish(output_dir: Path, payloads: list[tuple[str, str]]) -> None:
    if output_dir.exists():
        raise RagSftV2OracleCleanError(f"输出目录必须是新目录: {output_dir}")
    output_dir.mkdir(parents=True)
    pending = [(output_dir / f"{name}.partial", output_dir / name, payload) for name, payload in payloads]
    published: list[Path] = []
    try:
        for partial, _, payload in pending:
            partial.write_text(payload, encoding="utf-8", newline="\n")
        hash_payload = "".join(
            f"{_sha256(partial)}  {final.name}\n" for partial, final, _ in pending
        )
        hash_partial = output_dir / f"{HASH_FILENAME}.partial"
        hash_partial.write_text(hash_payload, encoding="utf-8", newline="\n")
        for partial, final, _ in pending:
            partial.replace(final)
            published.append(final)
        hash_partial.replace(output_dir / HASH_FILENAME)
    except (OSError, UnicodeError) as error:
        for path in [*(item[0] for item in pending), output_dir / f"{HASH_FILENAME}.partial", *reversed(published)]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            output_dir.rmdir()
        except OSError:
            pass
        raise RagSftV2OracleCleanError("无法发布 Oracle clean 阶段资产") from error


def materialize_rag_sft_v2_oracle_clean(
    *,
    approved_root: Path,
    query_pool_path: Path,
    article_index_path: Path,
    evaluation_exclusions_path: Path,
    development_eval_path: Path,
    private_holdout_path: Path,
    tokenizer_path: Path,
    corrections_path: Path | None,
    output_dir: Path,
) -> dict[str, object]:
    """完整投影 canonical clean，并发布阶段性审计资产而非训练 release。"""

    paths = {
        "approved_root": Path(approved_root).resolve(),
        "query_pool": Path(query_pool_path).resolve(),
        "article_index": Path(article_index_path).resolve(),
        "evaluation_exclusions": Path(evaluation_exclusions_path).resolve(),
        "development_eval": Path(development_eval_path).resolve(),
        "private_holdout": Path(private_holdout_path).resolve(),
        "tokenizer": Path(tokenizer_path).resolve(),
        "output_dir": Path(output_dir).resolve(),
    }
    if corrections_path is not None:
        paths["corrections"] = Path(corrections_path).resolve()
    for name in (
        "query_pool",
        "article_index",
        "evaluation_exclusions",
        "development_eval",
        "private_holdout",
    ):
        if not paths[name].is_file():
            raise RagSftV2OracleCleanError(f"输入文件不存在: {name}")
    if not paths["tokenizer"].is_dir():
        raise RagSftV2OracleCleanError("本地 tokenizer 目录不存在")
    if "corrections" in paths and not paths["corrections"].is_file():
        raise RagSftV2OracleCleanError("输入文件不存在: corrections")

    pool = _load_pool(paths["query_pool"])
    contents = _load_article_contents(paths["article_index"])
    canonical_records, batches = _load_canonical_records(
        paths["approved_root"],
        pool,
        contents,
        query_pool_path=paths["query_pool"],
        article_index_path=paths["article_index"],
    )
    corrections = (
        _load_canonical_corrections(paths["corrections"], canonical_records)
        if "corrections" in paths
        else {}
    )
    exclusions = _verify_formal_exclusions(paths["evaluation_exclusions"])
    exclusion_digests = set(exclusions["question_sha256"])
    conflicts = [
        record["query_id"]
        for record in canonical_records
        if question_digest(record["query_original"]) in exclusion_digests
    ]
    if conflicts:
        raise RagSftV2OracleCleanError("canonical query 命中开发集或私有留出集隔离清单")

    try:
        repository = ArticleRepository.from_jsonl(paths["article_index"])
        tokenizer = tokenizer_auditor.load_tokenizer(paths["tokenizer"])
    except Exception as error:
        raise RagSftV2OracleCleanError("无法加载 Oracle clean 投影依赖") from error

    clean_rows: list[dict[str, object]] = []
    targets: list[dict[str, object]] = []
    overbudget: list[dict[str, object]] = []
    prompt_counts: Counter[int] = Counter()
    assistant_counts: Counter[int] = Counter()
    total_counts: Counter[int] = Counter()
    required_counts: Counter[int] = Counter()
    claims_counts: Counter[int] = Counter()
    corrected_query_ids: list[str] = []
    for canonical in canonical_records:
        query_id = canonical["query_id"]
        try:
            article_by_chunk_id = {
                chunk_id: repository.get_by_chunk_id(chunk_id)
                for chunk_id in canonical["required_chunk_ids"]
            }
            projected = _project_canonical_with_correction(
                canonical,
                article_by_chunk_id,
                corrections.get(query_id),
            )
            audit = audit_projected_rag_sft_v2_record(projected, tokenizer)
        except RagSftV2ProjectionError as error:
            message = str(error)
            if "超过" not in message:
                raise RagSftV2OracleCleanError(
                    f"canonical {query_id} 无法完成协议投影"
                ) from error
            overbudget.append({"query_id": query_id, "reason": message})
            continue
        if query_id in corrections:
            corrected_query_ids.append(query_id)
        target = build_evaluation_target(canonical)
        clean_rows.append(
            {
                "id": projected.record_id,
                "query_id": projected.query_id,
                "variant": projected.variant,
                "query_original": canonical["query_original"],
                "visible_chunk_ids": list(projected.visible_chunk_ids),
                "required_chunk_ids": list(projected.required_chunk_ids),
                "citations": list(projected.citations),
                "conversations": projected.training_conversations(),
            }
        )
        targets.append(
            {
                "query_id": target.query_id,
                "claims": list(target.claims),
                "required_chunk_ids": list(target.required_chunk_ids),
            }
        )
        prompt_counts[audit.prompt_tokens] += 1
        assistant_counts[audit.assistant_label_tokens] += 1
        total_counts[audit.total_tokens] += 1
        required_counts[len(projected.required_chunk_ids)] += 1
        claims_counts[len(target.claims)] += 1
    if not clean_rows:
        raise RagSftV2OracleCleanError("没有任何 canonical 通过 Oracle clean 审计")

    clean_payload = _serialize_jsonl(clean_rows)
    target_payload = _serialize_jsonl(targets)
    overbudget_payload = _serialize_jsonl(overbudget)
    oracle_path = paths["output_dir"] / ORACLE_CLEAN_FILENAME
    targets_path = paths["output_dir"] / EVALUATION_TARGETS_FILENAME
    overbudget_path = paths["output_dir"] / OVERBUDGET_FILENAME
    isolation_path = paths["output_dir"] / ISOLATION_FILENAME
    manifest_path = paths["output_dir"] / MANIFEST_FILENAME

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", suffix=".jsonl", delete=False
        ) as temporary:
            temporary.write(clean_payload)
            temporary_path = Path(temporary.name)
        private_audit = audit_private_holdout_compatibility(
            authoring_path=paths["private_holdout"],
            article_index_path=paths["article_index"],
            development_eval_path=paths["development_eval"],
            rag_authoring_path=temporary_path,
        )
    except Exception as error:
        raise RagSftV2OracleCleanError("私有留出集兼容性审计失败") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    overlap_counts = private_audit.get("overlap_counts")
    if (
        private_audit.get("compatible") is not True
        or private_audit.get("raw_text_included") is not False
        or not isinstance(overlap_counts, dict)
        or any(value != 0 for value in overlap_counts.values())
    ):
        raise RagSftV2OracleCleanError("Oracle clean 与私有留出集兼容性未闭合")
    isolation = {
        "pipeline": "rag_sft_v2_oracle_clean_evaluation_isolation",
        "formal_exclusions": _identity(paths["evaluation_exclusions"]),
        "canonical_query_conflicts": 0,
        "private_holdout_compatibility": {
            "pipeline": private_audit.get("pipeline"),
            "compatible": True,
            "inputs": {
                "authoring": _identity(paths["private_holdout"]),
                "article_index": _identity(paths["article_index"]),
                "development_eval": _identity(paths["development_eval"]),
                "rag_authoring": _identity_from_payload(
                    oracle_path, clean_payload, records=len(clean_rows)
                ),
            },
            "private": private_audit.get("private"),
            "overlap_counts": overlap_counts,
            "raw_text_included": False,
        },
        "raw_private_text_included": False,
        "complete": True,
    }
    isolation_payload = json.dumps(isolation, ensure_ascii=False, indent=2, allow_nan=False) + "\n"

    tokenizer_report = tokenizer_auditor._tokenizer_report(tokenizer, paths["tokenizer"])
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str):
        raise RagSftV2OracleCleanError("tokenizer 缺少 chat template")
    tokenizer_report["chat_template_sha256"] = hashlib.sha256(
        chat_template.encode("utf-8")
    ).hexdigest()
    def distribution(counter: Counter[int]) -> dict[str, int]:
        return {str(key): counter[key] for key in sorted(counter)}

    manifest: dict[str, object] = {
        "pipeline": "rag_sft_v2_oracle_clean_materialization",
        "release_status": "stage_5_oracle_clean_audited",
        "inputs": {
            "canonical_batches": batches,
            "query_pool": _identity(paths["query_pool"], records=len(pool)),
            "article_index": _identity(paths["article_index"], records=len(contents)),
            "evaluation_exclusions": _identity(paths["evaluation_exclusions"]),
            "development_eval": _identity(paths["development_eval"]),
            "private_holdout": _identity(paths["private_holdout"]),
            **(
                {
                    "canonical_summary_corrections": _identity(
                        paths["corrections"], records=len(corrections)
                    )
                }
                if "corrections" in paths
                else {}
            ),
        },
        "protocol": {
            "model_output": ["summary", "citations"],
            "visible_evidence": "required_gt_only_in_stable_first_support_order",
            "hard_negatives_constructed": False,
            "context_limit": CONTEXT_LIMIT,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "truncation": "forbidden",
        },
        "tokenizer": tokenizer_report,
        "records": {
            "canonical_authoring": len(canonical_records),
            "oracle_clean": len(clean_rows),
            "overbudget_excluded": len(overbudget),
            "canonical_summary_corrections": len(corrections),
            "hard_negative": 0,
            "required_gt_count": distribution(required_counts),
            "claims_count": distribution(claims_counts),
            "prompt_tokens": distribution(prompt_counts),
            "assistant_label_tokens": distribution(assistant_counts),
            "total_tokens": distribution(total_counts),
        },
        "validation": {
            "canonical_records_closed": len(canonical_records) == len(clean_rows) + len(overbudget),
            "clean_query_ids_unique": len({row["query_id"] for row in clean_rows}) == len(clean_rows),
            "clean_visible_equals_required": all(
                row["visible_chunk_ids"] == row["required_chunk_ids"] for row in clean_rows
            ),
            "strict_two_field_protocol_projected": True,
            "citations_map_all_required_gt": True,
            "prompt_within_618": all(value <= MAX_PROMPT_TOKENS for value in prompt_counts),
            "assistant_within_150": all(value <= MAX_OUTPUT_TOKENS for value in assistant_counts),
            "full_sequence_within_768": all(value <= CONTEXT_LIMIT for value in total_counts),
            "assistant_only_labels_and_eos_audited": True,
            "evaluation_isolation_complete": True,
            "canonical_summary_corrections_applied": sorted(corrected_query_ids)
            == sorted(corrections),
        },
        "output": {
            "oracle_clean": _identity_from_payload(oracle_path, clean_payload, records=len(clean_rows)),
            "evaluation_targets": _identity_from_payload(targets_path, target_payload, records=len(targets)),
            "overbudget_exclusions": _identity_from_payload(overbudget_path, overbudget_payload, records=len(overbudget)),
            "evaluation_isolation": _identity_from_payload(isolation_path, isolation_payload, records=1),
        },
        "readiness": {
            "canonical_authoring_complete": True,
            "oracle_clean_materialized": True,
            "oracle_clean_audited": True,
            "real_hard_negatives_constructed": False,
            "final_retrieval_identity_frozen": False,
            "training_ready": False,
        },
        "limitations": [
            "本阶段不构造真实 hard-negative，不发布最终 v2 training release。",
            "Oracle clean 是充分证据条件下的阶段性审计资产，不能替代真实检索表现。",
            "overbudget 记录保留在排除账本，不截断法条、不启用 selector、不改变证据顺序。",
        ],
        "complete": True,
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _publish(
        paths["output_dir"],
        [
            (ORACLE_CLEAN_FILENAME, clean_payload),
            (EVALUATION_TARGETS_FILENAME, target_payload),
            (OVERBUDGET_FILENAME, overbudget_payload),
            (ISOLATION_FILENAME, isolation_payload),
            (MANIFEST_FILENAME, manifest_payload),
        ],
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-root", type=Path, default=DEFAULT_APPROVED_ROOT)
    parser.add_argument("--query-pool", type=Path, default=DEFAULT_QUERY_POOL)
    parser.add_argument("--article-index", type=Path, default=DEFAULT_ARTICLE_INDEX)
    parser.add_argument("--evaluation-exclusions", type=Path, default=DEFAULT_EVALUATION_EXCLUSIONS)
    parser.add_argument("--development-eval", type=Path, default=DEFAULT_DEVELOPMENT_EVAL)
    parser.add_argument("--private-holdout", type=Path, default=DEFAULT_PRIVATE_HOLDOUT)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--corrections", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        manifest = materialize_rag_sft_v2_oracle_clean(
            approved_root=args.approved_root,
            query_pool_path=args.query_pool,
            article_index_path=args.article_index,
            evaluation_exclusions_path=args.evaluation_exclusions,
            development_eval_path=args.development_eval,
            private_holdout_path=args.private_holdout,
            tokenizer_path=args.tokenizer,
            corrections_path=args.corrections,
            output_dir=args.output_dir,
        )
    except RagSftV2OracleCleanError as error:
        raise SystemExit(f"[失败] {error}") from error
    records = manifest["records"]
    print(
        f"[完成] canonical {records['canonical_authoring']} 条，"
        f"Oracle clean {records['oracle_clean']} 条，"
        f"超预算排除 {records['overbudget_excluded']} 条"
    )
    print("[阻断] 未构造真实 HN，training_ready=false")


if __name__ == "__main__":
    main()
