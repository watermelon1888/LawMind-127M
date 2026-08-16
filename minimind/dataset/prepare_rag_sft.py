"""把 canonical RAG-SFT authoring 投影为 MiniMind 三消息训练候选。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from rag.answering import (
    ASSISTANT_SCHEMA,
    SYSTEM_PROMPT,
    AnswerProtocolError,
    EvidencePackage,
    build_answer_prompt,
    parse_and_validate_answer,
)
from rag.core import Evidence
from rag.knowledge import ArticleRepository

try:
    from .build_sft_evaluation_exclusions import question_digest
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset.build_sft_evaluation_exclusions import question_digest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAG_SFT_ROOT = Path(__file__).resolve().parent / "RAG-SFT"
DEFAULT_AUTHORING = RAG_SFT_ROOT / "authoring" / "rag-sft-canonical-v1.jsonl"
DEFAULT_ARTICLE_INDEX = PROJECT_ROOT / "rag" / "chunk" / "article_index.jsonl"
DEFAULT_EVALUATION_EXCLUSIONS = (
    RAG_SFT_ROOT
    / "manifests"
    / "evaluation-exclusions-project-rag-dev-current.json"
)
DEFAULT_CANDIDATE_OUTPUT = (
    RAG_SFT_ROOT / "standardized" / "rag-sft-canonical-v1-candidate.jsonl"
)
DEFAULT_MANIFEST_OUTPUT = (
    RAG_SFT_ROOT / "manifests" / "rag-sft-canonical-v1-candidate.json"
)

_AUTHORING_FIELDS = {
    "id",
    "query_original",
    "evidence_source",
    "visible_chunk_ids",
    "required_chunk_ids",
    "target",
    "support_spans",
    "review_status",
    "review_notes",
}
_TARGET_FIELDS = {"summary", "refuse"}
_SPAN_FIELDS = {"chunk_id", "text"}
_EVIDENCE_SOURCES = {"oracle", "curated_insufficient", "retrieved"}
_REVIEW_STATUSES = {"draft", "approved"}
_ID_RE = re.compile(r"rag_sft:\d{4}$")


class RagSftPreparationError(RuntimeError):
    """authoring 不能安全投影或训练资产不能安全发布。"""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise RagSftPreparationError(
                        f"authoring JSON 无效: {path}:{line_number}"
                    ) from error
                if not isinstance(record, dict):
                    raise RagSftPreparationError(
                        f"authoring 记录必须是对象: {path}:{line_number}"
                    )
                records.append(record)
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftPreparationError(f"无法读取 authoring: {path}") from error
    return records


def _is_non_blank_string(value):
    return isinstance(value, str) and bool(value.strip())


def _string_list(value, field, *, maximum=None):
    if not isinstance(value, list) or not value:
        raise RagSftPreparationError(f"{field} 必须是非空字符串数组")
    if any(not _is_non_blank_string(item) for item in value):
        raise RagSftPreparationError(f"{field} 必须是非空字符串数组")
    if len(set(value)) != len(value):
        raise RagSftPreparationError(f"{field} 不能包含重复值")
    if maximum is not None and len(value) > maximum:
        raise RagSftPreparationError(f"{field} 最多包含 {maximum} 条")
    return value


def _validate_record_shape(record, position):
    prefix = f"authoring 第 {position} 条"
    if set(record) != _AUTHORING_FIELDS:
        raise RagSftPreparationError(f"{prefix}顶层字段必须精确匹配 schema")
    if not isinstance(record["id"], str) or not _ID_RE.fullmatch(record["id"]):
        raise RagSftPreparationError(f"{prefix} id 必须匹配 rag_sft:NNNN")
    if not _is_non_blank_string(record["query_original"]):
        raise RagSftPreparationError(f"{prefix} query_original 必须是非空字符串")
    if record["evidence_source"] not in _EVIDENCE_SOURCES:
        raise RagSftPreparationError(f"{prefix} evidence_source 无效")
    visible = _string_list(record["visible_chunk_ids"], f"{prefix} visible_chunk_ids", maximum=4)
    required = _string_list(record["required_chunk_ids"], f"{prefix} required_chunk_ids", maximum=3)
    target = record["target"]
    if not isinstance(target, dict) or set(target) != _TARGET_FIELDS:
        raise RagSftPreparationError(f"{prefix} target 字段必须精确匹配 schema")
    if not isinstance(target["summary"], str) or not isinstance(target["refuse"], bool):
        raise RagSftPreparationError(f"{prefix} target 类型无效")
    if target["refuse"]:
        if target["summary"] != "":
            raise RagSftPreparationError(f"{prefix}拒答 summary 必须为空")
    elif not target["summary"].strip() or "\n" in target["summary"] or "\r" in target["summary"]:
        raise RagSftPreparationError(f"{prefix}正常 summary 必须是非空单行字符串")
    if record["review_status"] not in _REVIEW_STATUSES:
        raise RagSftPreparationError(f"{prefix} review_status 无效")
    if not isinstance(record["review_notes"], str):
        raise RagSftPreparationError(f"{prefix} review_notes 必须是字符串")
    spans = record["support_spans"]
    if not isinstance(spans, list):
        raise RagSftPreparationError(f"{prefix} support_spans 必须是对象数组")
    pairs = set()
    for index, span in enumerate(spans):
        if not isinstance(span, dict) or set(span) != _SPAN_FIELDS:
            raise RagSftPreparationError(f"{prefix} support_spans[{index}] 字段无效")
        if not _is_non_blank_string(span["chunk_id"]) or not _is_non_blank_string(span["text"]):
            raise RagSftPreparationError(f"{prefix} support_spans[{index}] 内容无效")
        pair = (span["chunk_id"], span["text"])
        if pair in pairs:
            raise RagSftPreparationError(f"{prefix} support_spans 不能重复")
        pairs.add(pair)
    if target["refuse"] and spans:
        raise RagSftPreparationError(f"{prefix}拒答样本不能携带 support_spans")

    required_set = set(required)
    visible_set = set(visible)
    source = record["evidence_source"]
    if source == "oracle":
        if target["refuse"] or visible != required:
            raise RagSftPreparationError(f"{prefix} Oracle 必须回答且 visible 等于 required")
    elif source == "curated_insufficient":
        if not target["refuse"] or required_set.issubset(visible_set):
            raise RagSftPreparationError(f"{prefix} curated_insufficient 必须缺少 required GT")
    elif target["refuse"]:
        if required_set.issubset(visible_set):
            raise RagSftPreparationError(f"{prefix} retrieved 拒答必须缺少 required GT")
    else:
        if not required_set.issubset(visible_set):
            raise RagSftPreparationError(f"{prefix} retrieved 回答必须包含全部 required GT")


def _load_exclusions(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftPreparationError(f"无法读取评估排除清单: {path}") from error
    values = payload.get("question_sha256") if isinstance(payload, dict) else None
    if not isinstance(values, list) or not values or len(values) != len(set(values)):
        raise RagSftPreparationError("评估排除清单 question_sha256 无效")
    if any(not isinstance(item, str) or not re.fullmatch(r"[0-9a-f]{64}", item) for item in values):
        raise RagSftPreparationError("评估排除清单包含无效 SHA-256")
    return set(values), payload


def _package_and_assistant(record, repository):
    all_chunk_ids = set(record["visible_chunk_ids"]) | set(record["required_chunk_ids"])
    articles = {}
    try:
        for chunk_id in all_chunk_ids:
            articles[chunk_id] = repository.get_by_chunk_id(chunk_id)
    except KeyError as error:
        raise RagSftPreparationError(f"{record['id']} 引用了不存在的 chunk_id") from error
    for span in record["support_spans"]:
        if span["chunk_id"] not in set(record["required_chunk_ids"]):
            raise RagSftPreparationError(f"{record['id']} support span 不属于 required GT")
        if span["text"] not in articles[span["chunk_id"]].content:
            raise RagSftPreparationError(f"{record['id']} support span 不是连续法条原文")

    package = EvidencePackage(
        query=record["query_original"],
        evidence=tuple(
            Evidence(
                law_name=articles[chunk_id].law_name,
                article_no=articles[chunk_id].article_no,
                content=articles[chunk_id].content,
            )
            for chunk_id in record["visible_chunk_ids"]
        ),
    )
    required = set(record["required_chunk_ids"])
    citations = (
        []
        if record["target"]["refuse"]
        else [
            f"E{index}"
            for index, chunk_id in enumerate(record["visible_chunk_ids"], start=1)
            if chunk_id in required
        ]
    )
    assistant = json.dumps(
        {
            "summary": record["target"]["summary"],
            "citations": citations,
            "refuse": record["target"]["refuse"],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    try:
        parsed = parse_and_validate_answer(package, assistant)
    except AnswerProtocolError as error:
        raise RagSftPreparationError(f"{record['id']} 未通过回答协议: {error}") from error
    if parsed.citations != tuple(citations):
        raise RagSftPreparationError(f"{record['id']} citations 未按 visible 顺序生成")
    return package, assistant


def _require_new_outputs(candidate_output, manifest_output):
    targets = (candidate_output, manifest_output, manifest_output.with_suffix(".sha256"))
    occupied = [
        str(item)
        for target in targets
        for item in (target, target.with_name(target.name + ".partial"))
        if item.exists()
    ]
    if occupied:
        raise RagSftPreparationError("目标输出已存在: " + ", ".join(occupied))


def _identity(path, payload):
    encoded = payload.encode("utf-8")
    return {"path": str(path), "bytes": len(encoded), "sha256": _sha256_bytes(encoded)}


def _publish(candidate_output, candidate_payload, manifest_output, manifest_payload):
    hash_output = manifest_output.with_suffix(".sha256")
    hash_payload = f"{_sha256_bytes(manifest_payload.encode('utf-8'))}  {manifest_output.name}\n"
    outputs = [(manifest_output, manifest_payload), (hash_output, hash_payload)]
    if candidate_payload is not None:
        outputs.insert(0, (candidate_output, candidate_payload))
    partials = [(path.with_name(path.name + ".partial"), path, data) for path, data in outputs]
    published = []
    try:
        for partial, _, data in partials:
            partial.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(data, encoding="utf-8", newline="\n")
        for partial, final, _ in partials:
            partial.replace(final)
            published.append(final)
    except (OSError, UnicodeError) as error:
        for path in [*(item[0] for item in partials), *reversed(published)]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise RagSftPreparationError("无法发布 RAG SFT 产物") from error


def prepare_rag_sft(*, authoring_path, article_index_path, evaluation_exclusions_path, candidate_output, manifest_output):
    """校验 canonical authoring 并发布确定性训练投影。"""
    authoring_path = Path(authoring_path).resolve()
    article_index_path = Path(article_index_path).resolve()
    evaluation_exclusions_path = Path(evaluation_exclusions_path).resolve()
    candidate_output = Path(candidate_output).resolve()
    manifest_output = Path(manifest_output).resolve()
    _require_new_outputs(candidate_output, manifest_output)
    records = _load_jsonl(authoring_path)
    seen_ids = set()
    for index, record in enumerate(records, start=1):
        _validate_record_shape(record, index)
        if record["id"] in seen_ids:
            raise RagSftPreparationError(f"authoring id 重复: {record['id']}")
        seen_ids.add(record["id"])

    exclusions, exclusion_payload = _load_exclusions(evaluation_exclusions_path)
    for record in records:
        if question_digest(record["query_original"]) in exclusions:
            raise RagSftPreparationError(f"{record['id']} 的 query 命中评估排除清单")
    try:
        repository = ArticleRepository.from_jsonl(article_index_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise RagSftPreparationError("无法加载法条索引") from error

    projected = []
    source_counts = Counter(record["evidence_source"] for record in records)
    review_counts = Counter(record["review_status"] for record in records)
    for record in records:
        package, assistant = _package_and_assistant(record, repository)
        if record["review_status"] == "approved":
            projected.append(
                {
                    "id": record["id"],
                    "source": "rag_sft",
                    "evidence_source": record["evidence_source"],
                    "conversations": build_answer_prompt(package)
                    + [{"role": "assistant", "content": assistant}],
                }
            )

    candidate_payload = "".join(
        json.dumps(item, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for item in projected
    )
    projection_ready = bool(projected) and review_counts["draft"] == 0
    isolated = exclusion_payload.get("complete_for_formal_sft") is True
    manifest = {
        "pipeline": "rag_sft",
        "release_status": "formal_candidate" if projection_ready and isolated else (
            "provisional_candidate" if projected else "review_pending"
        ),
        "inputs": {
            "authoring": {"path": str(authoring_path), "bytes": authoring_path.stat().st_size, "sha256": _sha256_file(authoring_path)},
            "article_index": {"path": str(article_index_path), "bytes": article_index_path.stat().st_size, "sha256": _sha256_file(article_index_path)},
            "evaluation_exclusions": {"path": str(evaluation_exclusions_path), "bytes": evaluation_exclusions_path.stat().st_size, "sha256": _sha256_file(evaluation_exclusions_path), "digest_count": len(exclusions)},
        },
        "protocol": {
            "hash_algorithm": "sha256",
            "system_prompt_source": "rag.answering.protocol.SYSTEM_PROMPT",
            "system_prompt_sha256": "sha256:" + _sha256_bytes(SYSTEM_PROMPT.encode("utf-8")),
            "assistant_schema_source": "rag.answering.protocol.ASSISTANT_SCHEMA",
            "assistant_schema_sha256": "sha256:" + _sha256_bytes(ASSISTANT_SCHEMA.encode("utf-8")),
        },
        "records": {
            "total": len(records),
            "draft": review_counts["draft"],
            "approved": review_counts["approved"],
            "candidate": len(projected),
            "by_evidence_source": dict(sorted(source_counts.items())),
        },
        "output": {"candidate": _identity(candidate_output, candidate_payload) if projected else None},
        "readiness": {
            "protocol_projection_ready": projection_ready,
            "evaluation_isolation_complete": isolated,
            "training_ready": projection_ready and isolated,
        },
        "complete": True,
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _publish(candidate_output, candidate_payload if projected else None, manifest_output, manifest_payload)
    return manifest


def main():
    """解析路径并生成 canonical RAG-SFT 候选与 manifest。"""
    parser = argparse.ArgumentParser(description="生成证据约束的 canonical RAG-SFT 候选数据")
    parser.add_argument("--authoring", type=Path, default=DEFAULT_AUTHORING)
    parser.add_argument("--article-index", type=Path, default=DEFAULT_ARTICLE_INDEX)
    parser.add_argument("--evaluation-exclusions", type=Path, default=DEFAULT_EVALUATION_EXCLUSIONS)
    parser.add_argument("--candidate-output", type=Path, default=DEFAULT_CANDIDATE_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST_OUTPUT)
    args = parser.parse_args()
    try:
        manifest = prepare_rag_sft(
            authoring_path=args.authoring,
            article_index_path=args.article_index,
            evaluation_exclusions_path=args.evaluation_exclusions,
            candidate_output=args.candidate_output,
            manifest_output=args.manifest_output,
        )
    except RagSftPreparationError as error:
        parser.error(str(error))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
