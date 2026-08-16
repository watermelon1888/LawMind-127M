"""把 canonical v2 与已裁决 retrieved 样本汇编为最终 RAG-SFT authoring。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rag.answering import ASSISTANT_SCHEMA, SYSTEM_PROMPT
from rag.knowledge import ArticleRepository

try:
    from . import adjudicate_rag_sft_retrieved as adjudicator
    from . import assemble_rag_sft_authoring as canonical_assembler
    from . import prepare_rag_sft as preparation
    from .build_sft_evaluation_exclusions import question_digest
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import adjudicate_rag_sft_retrieved as adjudicator
    from dataset import assemble_rag_sft_authoring as canonical_assembler
    from dataset import prepare_rag_sft as preparation
    from dataset.build_sft_evaluation_exclusions import question_digest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_SFT_ROOT = Path(__file__).resolve().parent / "RAG-SFT"
DEFAULT_CANONICAL_MANIFEST = RAG_SFT_ROOT / "manifests" / "rag-sft-canonical-v2.json"
DEFAULT_ADJUDICATION_DIR = (
    RAG_SFT_ROOT / "retrieved" / "adjudication-canonical-v2-20260809"
)
DEFAULT_ARTICLE_INDEX = PROJECT_ROOT / "rag" / "chunk" / "article_index.jsonl"
DEFAULT_EVALUATION_EXCLUSIONS = (
    RAG_SFT_ROOT / "manifests" / "evaluation-exclusions-project-rag-dev-current.json"
)
DEFAULT_AUTHORING_OUTPUT = RAG_SFT_ROOT / "authoring" / "rag-sft-training-v1.jsonl"
DEFAULT_MAPPING_OUTPUT = RAG_SFT_ROOT / "manifests" / "rag-sft-training-v1-mapping.jsonl"
DEFAULT_MANIFEST_OUTPUT = RAG_SFT_ROOT / "manifests" / "rag-sft-training-v1.json"

_SHA256_RE = re.compile(r"[0-9a-f]{64}$")
_EXPECTED_ADJUDICATION_FILES = {
    adjudicator.ADJUDICATION_FILENAME,
    adjudicator.CURATED_FILENAME,
    adjudicator.REPORT_FILENAME,
}


class RagSftTrainingAuthoringAssemblyError(RuntimeError):
    """最终训练 authoring 的输入身份、内容或输出无法安全闭合。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if records is not None:
        value["records"] = records
    return value


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftTrainingAuthoringAssemblyError(
            f"无法读取{description}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise RagSftTrainingAuthoringAssemblyError(f"{description}必须是 JSON object")
    return value


def _verify_identity(
    metadata: object, path: Path, description: str, *, records: int | None = None
) -> None:
    if not isinstance(metadata, dict):
        raise RagSftTrainingAuthoringAssemblyError(f"{description}身份无效")
    expected = _identity(path, records=records)
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RagSftTrainingAuthoringAssemblyError(f"{description}身份不一致")


def _verify_adjacent_manifest_hash(path: Path, description: str) -> Path:
    hash_path = path.with_suffix(".sha256")
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftTrainingAuthoringAssemblyError(
            f"无法读取{description} SHA-256"
        ) from error
    if lines != [f"{_sha256(path)}  {path.name}"]:
        raise RagSftTrainingAuthoringAssemblyError(f"{description} SHA-256 无效")
    return hash_path


def _canonical_source(manifest_path: Path) -> tuple[Path, dict[str, object]]:
    manifest_path = manifest_path.resolve()
    hash_path = _verify_adjacent_manifest_hash(manifest_path, "canonical v2 manifest")
    manifest = _load_json(manifest_path, "canonical v2 manifest")
    output = manifest.get("output", {})
    records = manifest.get("records", {})
    if (
        manifest.get("pipeline") != "rag_sft_canonical_authoring_correction"
        or manifest.get("complete") is not True
        or not isinstance(records, dict)
        or type(records.get("total")) is not int
        or records["total"] <= 0
        or not isinstance(output, dict)
        or not isinstance(output.get("authoring"), dict)
        or not isinstance(output["authoring"].get("path"), str)
    ):
        raise RagSftTrainingAuthoringAssemblyError("canonical v2 manifest 状态无效")
    authoring_path = Path(output["authoring"]["path"]).resolve()
    _verify_identity(
        output["authoring"], authoring_path, "canonical v2 authoring", records=records["total"]
    )
    return authoring_path, {
        "records": records["total"],
        "manifest": _identity(manifest_path),
        "hash_manifest": _identity(hash_path),
    }


def _adjudication_source(
    directory: Path, canonical_path: Path
) -> tuple[Path, dict[str, object]]:
    directory = directory.resolve()
    report_path = directory / adjudicator.REPORT_FILENAME
    curated_path = directory / adjudicator.CURATED_FILENAME
    adjudication_path = directory / adjudicator.ADJUDICATION_FILENAME
    paths = {
        report_path.name: report_path,
        curated_path.name: curated_path,
        adjudication_path.name: adjudication_path,
    }
    hash_path = directory / adjudicator.HASH_FILENAME
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftTrainingAuthoringAssemblyError(
            "无法读取 retrieved adjudication SHA-256"
        ) from error
    parsed = {}
    for line_number, line in enumerate(lines, start=1):
        parts = line.split("  ", 1)
        if (
            len(parts) != 2
            or not _SHA256_RE.fullmatch(parts[0])
            or Path(parts[1]).name != parts[1]
            or parts[1] in parsed
        ):
            raise RagSftTrainingAuthoringAssemblyError(
                f"retrieved adjudication SHA-256 格式无效: {line_number}"
            )
        parsed[parts[1]] = parts[0]
    if set(parsed) != _EXPECTED_ADJUDICATION_FILES:
        raise RagSftTrainingAuthoringAssemblyError(
            "retrieved adjudication SHA-256 文件集合无效"
        )
    for name, path in paths.items():
        if not path.is_file() or parsed[name] != _sha256(path):
            raise RagSftTrainingAuthoringAssemblyError(
                f"retrieved adjudication 文件身份已变化: {name}"
            )
    report = _load_json(report_path, "retrieved adjudication report")
    records = report.get("records", {})
    validation = report.get("validation", {})
    if (
        report.get("pipeline") != "rag_sft_retrieved_semantic_adjudication"
        or report.get("complete") is not True
        or not isinstance(records, dict)
        or type(records.get("curated")) is not int
        or records["curated"] <= 0
        or not isinstance(validation, dict)
        or any(
            validation.get(field) is not True
            for field in (
                "all_locators_adjudicated",
                "curated_ids_unique",
                "curated_signatures_unique",
                "no_canonical_conversation_duplicates",
            )
        )
    ):
        raise RagSftTrainingAuthoringAssemblyError("retrieved adjudication 状态无效")
    _verify_identity(
        report.get("input", {}).get("canonical_authoring"),
        canonical_path,
        "adjudication canonical authoring",
    )
    return curated_path, {
        "report": _identity(report_path),
        "hash_manifest": _identity(hash_path),
        "adjudication": _identity(
            adjudication_path, records=records.get("curated", 0) + records.get("excluded", 0)
        ),
        "curated_authoring": _identity(curated_path, records=records["curated"]),
    }


def assemble_rag_sft_training_authoring(
    *,
    canonical_manifest_path: Path,
    adjudication_dir: Path,
    article_index_path: Path,
    evaluation_exclusions_path: Path,
    authoring_output: Path,
    mapping_output: Path,
    manifest_output: Path,
) -> dict[str, object]:
    """验证两个已批准来源，按 ID 排序发布最终 authoring 与来源映射。"""

    article_index_path = Path(article_index_path).resolve()
    evaluation_exclusions_path = Path(evaluation_exclusions_path).resolve()
    authoring_output = Path(authoring_output).resolve()
    mapping_output = Path(mapping_output).resolve()
    manifest_output = Path(manifest_output).resolve()
    hash_output = manifest_output.with_suffix(".sha256")
    canonical_assembler._require_new_outputs(
        authoring_output, mapping_output, manifest_output, hash_output
    )
    canonical_path, canonical_identity = _canonical_source(Path(canonical_manifest_path))
    retrieved_path, adjudication_identity = _adjudication_source(
        Path(adjudication_dir), canonical_path
    )
    canonical_rows = canonical_assembler._load_jsonl(canonical_path, "canonical v2 authoring")
    retrieved_rows = canonical_assembler._load_jsonl(retrieved_path, "retrieved curated authoring")
    if len(canonical_rows) != canonical_identity["records"]:
        raise RagSftTrainingAuthoringAssemblyError("canonical v2 authoring 条数不闭合")
    if len(retrieved_rows) != adjudication_identity["curated_authoring"]["records"]:
        raise RagSftTrainingAuthoringAssemblyError("retrieved curated authoring 条数不闭合")
    try:
        exclusions, exclusion_payload = preparation._load_exclusions(
            evaluation_exclusions_path
        )
        repository = ArticleRepository.from_jsonl(article_index_path)
    except (
        preparation.RagSftPreparationError,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
    ) as error:
        raise RagSftTrainingAuthoringAssemblyError("无法加载最终汇编校验依赖") from error

    entries = []
    seen_ids = set()
    conversation_by_digest = {}
    source_counts: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    behavior_counts: Counter[str] = Counter()
    for source_name, rows in (
        ("canonical_v2", canonical_rows),
        ("retrieved_curated", retrieved_rows),
    ):
        for line_number, record in rows:
            try:
                preparation._validate_record_shape(record, line_number)
                package, assistant = preparation._package_and_assistant(record, repository)
            except preparation.RagSftPreparationError as error:
                raise RagSftTrainingAuthoringAssemblyError(
                    f"{source_name} 记录无效: {error}"
                ) from error
            record_id = record["id"]
            if record["review_status"] != "approved":
                raise RagSftTrainingAuthoringAssemblyError(
                    f"最终来源包含未批准记录: {record_id}"
                )
            if record_id in seen_ids:
                raise RagSftTrainingAuthoringAssemblyError(f"最终来源 ID 冲突: {record_id}")
            if question_digest(record["query_original"]) in exclusions:
                raise RagSftTrainingAuthoringAssemblyError(
                    f"最终来源 query 命中评估排除清单: {record_id}"
                )
            conversation_hash = canonical_assembler._conversation_digest(package, assistant)
            duplicate_id = conversation_by_digest.get(conversation_hash)
            if duplicate_id is not None:
                raise RagSftTrainingAuthoringAssemblyError(
                    f"模型可见 conversations 完全重复: {duplicate_id}, {record_id}"
                )
            seen_ids.add(record_id)
            conversation_by_digest[conversation_hash] = record_id
            source_counts[source_name] += 1
            evidence_counts[record["evidence_source"]] += 1
            behavior_counts["refusal" if record["target"]["refuse"] else "answer"] += 1
            entries.append(
                (
                    record,
                    {
                        "id": record_id,
                        "source_asset": source_name,
                        "source_line": line_number,
                        "source_record_sha256": canonical_assembler._record_digest(record),
                    },
                )
            )
    entries.sort(key=lambda item: canonical_assembler._id_sort_key(item[0]["id"]))
    records = [record for record, _ in entries]
    mapping = [metadata for _, metadata in entries]
    authoring_payload = "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in records
    )
    mapping_payload = "".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for item in mapping
    )
    formal_isolation = exclusion_payload.get("complete_for_formal_sft") is True
    manifest: dict[str, object] = {
        "pipeline": "rag_sft_training_authoring_assembly",
        "release_status": (
            "formal_training_authoring" if formal_isolation else "provisional_training_authoring"
        ),
        "input": {
            "canonical_v2": {
                **canonical_identity,
                "authoring": _identity(canonical_path, records=len(canonical_rows)),
            },
            "retrieved_adjudication": adjudication_identity,
            "article_index": _identity(article_index_path),
            "evaluation_exclusions": {
                **_identity(evaluation_exclusions_path),
                "digest_count": len(exclusions),
                "complete_for_formal_sft": formal_isolation,
            },
        },
        "protocol": {
            "system_prompt_sha256": "sha256:"
            + hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
            "assistant_schema_sha256": "sha256:"
            + hashlib.sha256(ASSISTANT_SCHEMA.encode("utf-8")).hexdigest(),
        },
        "records": {
            "total": len(records),
            "by_source_asset": dict(sorted(source_counts.items())),
            "by_evidence_source": dict(sorted(evidence_counts.items())),
            "by_behavior": dict(sorted(behavior_counts.items())),
        },
        "validation": {
            "all_records_approved": True,
            "ids_unique": True,
            "article_and_support_spans_validated": True,
            "model_visible_conversations_unique": True,
            "development_evaluation_question_conflicts": 0,
        },
        "output": {
            "authoring": canonical_assembler._identity_from_payload(
                authoring_output, authoring_payload, records=len(records)
            ),
            "source_mapping": canonical_assembler._identity_from_payload(
                mapping_output, mapping_payload, records=len(mapping)
            ),
        },
        "readiness": {
            "authoring_ready": True,
            "formal_evaluation_isolation_complete": formal_isolation,
            "training_ready": False,
        },
        "complete": True,
    }
    manifest_payload = json.dumps(
        manifest, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    hash_payload = f"{hashlib.sha256(manifest_payload.encode('utf-8')).hexdigest()}  {manifest_output.name}\n"
    canonical_assembler._publish(
        [
            (authoring_output, authoring_payload),
            (mapping_output, mapping_payload),
            (manifest_output, manifest_payload),
            (hash_output, hash_payload),
        ]
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-manifest", type=Path, default=DEFAULT_CANONICAL_MANIFEST)
    parser.add_argument("--adjudication-dir", type=Path, default=DEFAULT_ADJUDICATION_DIR)
    parser.add_argument("--article-index", type=Path, default=DEFAULT_ARTICLE_INDEX)
    parser.add_argument(
        "--evaluation-exclusions", type=Path, default=DEFAULT_EVALUATION_EXCLUSIONS
    )
    parser.add_argument("--authoring-output", type=Path, default=DEFAULT_AUTHORING_OUTPUT)
    parser.add_argument("--mapping-output", type=Path, default=DEFAULT_MAPPING_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    args = parser.parse_args()
    try:
        manifest = assemble_rag_sft_training_authoring(
            canonical_manifest_path=args.canonical_manifest,
            adjudication_dir=args.adjudication_dir,
            article_index_path=args.article_index,
            evaluation_exclusions_path=args.evaluation_exclusions,
            authoring_output=args.authoring_output,
            mapping_output=args.mapping_output,
            manifest_output=args.manifest_output,
        )
    except RagSftTrainingAuthoringAssemblyError as error:
        parser.error(str(error))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


__all__ = [
    "RagSftTrainingAuthoringAssemblyError",
    "assemble_rag_sft_training_authoring",
]
