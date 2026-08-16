"""把已批准的 RAG-SFT 来源汇编为统一 canonical authoring。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from rag.answering import build_answer_prompt
from rag.knowledge import ArticleRepository

try:
    from . import prepare_rag_sft as preparation
    from .build_sft_evaluation_exclusions import question_digest
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset import prepare_rag_sft as preparation
    from dataset.build_sft_evaluation_exclusions import question_digest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_SFT_ROOT = Path(__file__).resolve().parent / "RAG-SFT"
DEFAULT_SEED_AUTHORING = RAG_SFT_ROOT / "authoring" / "rag-sft.jsonl"
DEFAULT_RULE_CARD_MANIFEST = (
    RAG_SFT_ROOT / "manifests" / "rag-sft-rule-cards-oracle-v1.json"
)
DEFAULT_ARTICLE_INDEX = PROJECT_ROOT / "rag" / "chunk" / "article_index.jsonl"
DEFAULT_EVALUATION_EXCLUSIONS = (
    RAG_SFT_ROOT / "manifests" / "evaluation-exclusions-project-rag-dev-current.json"
)
DEFAULT_AUTHORING_OUTPUT = RAG_SFT_ROOT / "authoring" / "rag-sft-canonical-v1.jsonl"
DEFAULT_MAPPING_OUTPUT = (
    RAG_SFT_ROOT / "manifests" / "rag-sft-canonical-v1-mapping.jsonl"
)
DEFAULT_MANIFEST_OUTPUT = RAG_SFT_ROOT / "manifests" / "rag-sft-canonical-v1.json"

_ID_RE = re.compile(r"rag_sft:(\d{4})$")
_UPSTREAM_MAPPING_FIELDS = {
    "id",
    "candidate_id",
    "rule_card_sha256",
    "revision_round",
}


class RagSftAuthoringAssemblyError(RuntimeError):
    """RAG-SFT 来源不能安全汇编为统一 canonical authoring。"""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity_from_file(path: Path, *, records: int | None = None) -> dict[str, object]:
    identity: dict[str, object] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if records is not None:
        identity["records"] = records
    return identity


def _identity_from_payload(
    path: Path, payload: str, *, records: int
) -> dict[str, object]:
    encoded = payload.encode("utf-8")
    return {
        "path": str(path),
        "records": records,
        "bytes": len(encoded),
        "sha256": _sha256_bytes(encoded),
    }


def _load_json(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftAuthoringAssemblyError(
            f"无法读取{description}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise RagSftAuthoringAssemblyError(f"{description}必须是 JSON object")
    return value


def _load_jsonl(
    path: Path, description: str
) -> list[tuple[int, dict[str, object]]]:
    records: list[tuple[int, dict[str, object]]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RagSftAuthoringAssemblyError(
                        f"{description} JSON 无效: {path}:{line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise RagSftAuthoringAssemblyError(
                        f"{description}记录必须是对象: {path}:{line_number}"
                    )
                records.append((line_number, record))
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftAuthoringAssemblyError(f"无法读取{description}: {path}") from error
    if not records:
        raise RagSftAuthoringAssemblyError(f"{description}不能为空")
    return records


def _verify_adjacent_hash(path: Path, description: str) -> dict[str, object]:
    hash_path = path.with_suffix(".sha256")
    try:
        lines = [
            line
            for line in hash_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftAuthoringAssemblyError(
            f"无法读取{description}相邻 SHA-256: {hash_path}"
        ) from error
    expected = f"{_sha256_file(path)}  {path.name}"
    if lines != [expected]:
        raise RagSftAuthoringAssemblyError(f"{description}相邻 SHA-256 无效")
    return {
        **_identity_from_file(path),
        "hash_manifest": _identity_from_file(hash_path),
    }


def _verify_bound_file(metadata: object, description: str) -> Path:
    if not isinstance(metadata, dict):
        raise RagSftAuthoringAssemblyError(f"规则卡 manifest 缺少{description}身份")
    path_value = metadata.get("path")
    expected_bytes = metadata.get("bytes")
    expected_sha256 = metadata.get("sha256")
    if (
        not isinstance(path_value, str)
        or type(expected_bytes) is not int
        or not isinstance(expected_sha256, str)
    ):
        raise RagSftAuthoringAssemblyError(f"规则卡 manifest 的{description}身份无效")
    path = Path(path_value).resolve()
    if not path.is_file():
        raise RagSftAuthoringAssemblyError(f"{description}不存在: {path}")
    if path.stat().st_size != expected_bytes or _sha256_file(path) != expected_sha256:
        raise RagSftAuthoringAssemblyError(f"{description}身份已变化")
    return path


def _load_rule_card_source(
    manifest_path: Path,
) -> tuple[Path, Path, dict[str, object]]:
    manifest_identity = _verify_adjacent_hash(manifest_path, "规则卡物化 manifest")
    manifest = _load_json(manifest_path, "规则卡物化 manifest")
    readiness = manifest.get("readiness")
    output = manifest.get("output")
    records = manifest.get("records")
    if (
        manifest.get("pipeline") != "rag_sft_rule_cards_to_oracle_authoring"
        or manifest.get("complete") is not True
        or not isinstance(readiness, dict)
        or readiness.get("semantic_review_complete") is not True
        or readiness.get("canonical_authoring_schema_validated") is not True
        or readiness.get("evaluation_isolation_validated") is not True
        or not isinstance(output, dict)
        or not isinstance(records, dict)
        or type(records.get("materialized")) is not int
        or records["materialized"] <= 0
    ):
        raise RagSftAuthoringAssemblyError("规则卡物化 manifest 状态无效")
    authoring_path = _verify_bound_file(output.get("authoring"), "规则卡 Oracle authoring")
    mapping_path = _verify_bound_file(output.get("source_mapping"), "规则卡来源映射")
    return authoring_path, mapping_path, manifest_identity


def _record_digest(record: dict[str, object]) -> str:
    payload = json.dumps(
        record,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return _sha256_bytes(payload.encode("utf-8"))


def _conversation_digest(package, assistant: str) -> str:
    conversations = build_answer_prompt(package) + [
        {"role": "assistant", "content": assistant}
    ]
    payload = json.dumps(
        conversations,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return _sha256_bytes(payload.encode("utf-8"))


def _id_sort_key(record_id: str) -> int:
    match = _ID_RE.fullmatch(record_id)
    if match is None:
        raise RagSftAuthoringAssemblyError(f"无效 canonical ID: {record_id}")
    return int(match.group(1))


def _verify_upstream_mapping(
    mapping_path: Path,
    oracle_records: list[tuple[int, dict[str, object]]],
) -> None:
    mapping_rows = _load_jsonl(mapping_path, "规则卡来源映射")
    mapped_ids: set[str] = set()
    for line_number, row in mapping_rows:
        if set(row) != _UPSTREAM_MAPPING_FIELDS:
            raise RagSftAuthoringAssemblyError(
                f"规则卡来源映射 schema 无效: {mapping_path}:{line_number}"
            )
        record_id = row.get("id")
        if not isinstance(record_id, str) or record_id in mapped_ids:
            raise RagSftAuthoringAssemblyError("规则卡来源映射 ID 无效或重复")
        mapped_ids.add(record_id)
    oracle_ids: set[str] = set()
    for _, record in oracle_records:
        record_id = record.get("id")
        if not isinstance(record_id, str) or record_id in oracle_ids:
            raise RagSftAuthoringAssemblyError(
                "规则卡 Oracle authoring ID 无效或重复"
            )
        oracle_ids.add(record_id)
    if mapped_ids != oracle_ids:
        raise RagSftAuthoringAssemblyError(
            "规则卡来源映射未与 Oracle authoring ID 闭合"
        )


def _require_new_outputs(*paths: Path) -> None:
    occupied = [
        str(item)
        for path in paths
        for item in (path, path.with_name(path.name + ".partial"))
        if item.exists()
    ]
    if occupied:
        raise RagSftAuthoringAssemblyError(
            "目标输出已存在: " + ", ".join(occupied)
        )


def _publish(outputs: list[tuple[Path, str]]) -> None:
    pending = [
        (path.with_name(path.name + ".partial"), path, payload)
        for path, payload in outputs
    ]
    published: list[Path] = []
    try:
        for partial, _, payload in pending:
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(payload, encoding="utf-8", newline="\n")
        for partial, final, _ in pending:
            partial.replace(final)
            published.append(final)
    except (OSError, UnicodeError) as error:
        for path in [*(item[0] for item in pending), *reversed(published)]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise RagSftAuthoringAssemblyError("无法发布统一 canonical authoring") from error


def assemble_rag_sft_authoring(
    *,
    seed_authoring_path: Path,
    rule_card_manifest_path: Path,
    article_index_path: Path,
    evaluation_exclusions_path: Path,
    authoring_output: Path,
    mapping_output: Path,
    manifest_output: Path,
) -> dict[str, object]:
    """汇编已批准 seed 与规则卡 Oracle，并发布单一 canonical authoring。"""

    seed_authoring_path = Path(seed_authoring_path).resolve()
    rule_card_manifest_path = Path(rule_card_manifest_path).resolve()
    article_index_path = Path(article_index_path).resolve()
    evaluation_exclusions_path = Path(evaluation_exclusions_path).resolve()
    authoring_output = Path(authoring_output).resolve()
    mapping_output = Path(mapping_output).resolve()
    manifest_output = Path(manifest_output).resolve()
    hash_output = manifest_output.with_suffix(".sha256")
    _require_new_outputs(authoring_output, mapping_output, manifest_output, hash_output)

    oracle_path, upstream_mapping_path, upstream_manifest_identity = (
        _load_rule_card_source(rule_card_manifest_path)
    )
    seed_records = _load_jsonl(seed_authoring_path, "curated seed authoring")
    oracle_records = _load_jsonl(oracle_path, "规则卡 Oracle authoring")
    _verify_upstream_mapping(upstream_mapping_path, oracle_records)
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
        raise RagSftAuthoringAssemblyError("无法加载 canonical 校验依赖") from error

    source_rows = (
        ("curated_seed", seed_records),
        ("rule_cards_oracle_v1", oracle_records),
    )
    entries: list[tuple[dict[str, object], dict[str, object]]] = []
    seen_ids: set[str] = set()
    conversation_by_digest: dict[str, str] = {}
    source_counts: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    behavior_counts: Counter[str] = Counter()
    gt_counts: Counter[str] = Counter()
    for source_name, rows in source_rows:
        for line_number, record in rows:
            try:
                preparation._validate_record_shape(record, line_number)
            except preparation.RagSftPreparationError as error:
                raise RagSftAuthoringAssemblyError(
                    f"{source_name} authoring 无效: {error}"
                ) from error
            record_id = record["id"]
            if record["review_status"] != "approved":
                raise RagSftAuthoringAssemblyError(
                    f"canonical 来源包含未批准记录: {record_id}"
                )
            if record_id in seen_ids:
                raise RagSftAuthoringAssemblyError(
                    f"canonical 来源 ID 冲突: {record_id}"
                )
            if question_digest(record["query_original"]) in exclusions:
                raise RagSftAuthoringAssemblyError(
                    f"canonical 来源 query 命中评估排除清单: {record_id}"
                )
            try:
                package, assistant = preparation._package_and_assistant(
                    record, repository
                )
            except preparation.RagSftPreparationError as error:
                raise RagSftAuthoringAssemblyError(
                    f"canonical 来源证据无效: {record_id}: {error}"
                ) from error
            conversation_hash = _conversation_digest(package, assistant)
            duplicate_id = conversation_by_digest.get(conversation_hash)
            if duplicate_id is not None:
                raise RagSftAuthoringAssemblyError(
                    f"模型可见 conversations 完全重复: {duplicate_id}, {record_id}"
                )
            seen_ids.add(record_id)
            conversation_by_digest[conversation_hash] = record_id
            source_counts[source_name] += 1
            evidence_counts[record["evidence_source"]] += 1
            behavior_counts["refusal" if record["target"]["refuse"] else "answer"] += 1
            gt_counts[str(len(record["required_chunk_ids"]))] += 1
            entries.append(
                (
                    record,
                    {
                        "id": record_id,
                        "source_asset": source_name,
                        "source_line": line_number,
                        "source_record_sha256": _record_digest(record),
                    },
                )
            )

    entries.sort(key=lambda item: _id_sort_key(item[0]["id"]))
    records = [record for record, _ in entries]
    mapping = [source for _, source in entries]
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
    manifest = {
        "pipeline": "rag_sft_canonical_authoring_assembly",
        "release_status": (
            "formal_canonical" if formal_isolation else "provisional_canonical"
        ),
        "inputs": {
            "curated_seed": _identity_from_file(
                seed_authoring_path, records=len(seed_records)
            ),
            "rule_cards_oracle": _identity_from_file(
                oracle_path, records=len(oracle_records)
            ),
            "rule_cards_oracle_mapping": _identity_from_file(
                upstream_mapping_path, records=len(oracle_records)
            ),
            "rule_cards_oracle_manifest": upstream_manifest_identity,
            "article_index": _identity_from_file(article_index_path),
            "evaluation_exclusions": {
                **_identity_from_file(evaluation_exclusions_path),
                "digest_count": len(exclusions),
                "complete_for_formal_sft": formal_isolation,
            },
        },
        "assembly": {
            "ordering": "numeric_rag_sft_id",
            "ids_renumbered": False,
            "source_records_mutated": False,
            "same_query_with_different_model_input_allowed": True,
            "exact_model_visible_conversation_duplicates": 0,
        },
        "records": {
            "total": len(records),
            "by_source_asset": dict(sorted(source_counts.items())),
            "by_evidence_source": dict(sorted(evidence_counts.items())),
            "by_behavior": dict(sorted(behavior_counts.items())),
            "by_required_gt_count": dict(sorted(gt_counts.items())),
        },
        "output": {
            "authoring": _identity_from_payload(
                authoring_output, authoring_payload, records=len(records)
            ),
            "source_mapping": _identity_from_payload(
                mapping_output, mapping_payload, records=len(mapping)
            ),
        },
        "readiness": {
            "all_source_records_approved": True,
            "canonical_authoring_schema_validated": True,
            "article_and_support_spans_validated": True,
            "development_evaluation_exact_conflicts": 0,
            "formal_evaluation_isolation_complete": formal_isolation,
            "prepared_training_candidate": False,
            "training_ready": False,
        },
        "complete": True,
    }
    manifest_payload = json.dumps(
        manifest, ensure_ascii=False, indent=2, allow_nan=False
    ) + "\n"
    hash_payload = (
        f"{_sha256_bytes(manifest_payload.encode('utf-8'))}  {manifest_output.name}\n"
    )
    _publish(
        [
            (authoring_output, authoring_payload),
            (mapping_output, mapping_payload),
            (manifest_output, manifest_payload),
            (hash_output, hash_payload),
        ]
    )
    return manifest


def main() -> None:
    """解析命令行参数并发布统一 RAG-SFT canonical authoring。"""

    parser = argparse.ArgumentParser(description="汇编统一 RAG-SFT canonical authoring")
    parser.add_argument("--seed-authoring", type=Path, default=DEFAULT_SEED_AUTHORING)
    parser.add_argument(
        "--rule-card-manifest", type=Path, default=DEFAULT_RULE_CARD_MANIFEST
    )
    parser.add_argument("--article-index", type=Path, default=DEFAULT_ARTICLE_INDEX)
    parser.add_argument(
        "--evaluation-exclusions", type=Path, default=DEFAULT_EVALUATION_EXCLUSIONS
    )
    parser.add_argument("--authoring-output", type=Path, default=DEFAULT_AUTHORING_OUTPUT)
    parser.add_argument("--mapping-output", type=Path, default=DEFAULT_MAPPING_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    args = parser.parse_args()
    try:
        manifest = assemble_rag_sft_authoring(
            seed_authoring_path=args.seed_authoring,
            rule_card_manifest_path=args.rule_card_manifest,
            article_index_path=args.article_index,
            evaluation_exclusions_path=args.evaluation_exclusions,
            authoring_output=args.authoring_output,
            mapping_output=args.mapping_output,
            manifest_output=args.manifest_output,
        )
    except RagSftAuthoringAssemblyError as error:
        parser.error(str(error))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
