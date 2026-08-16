"""加载 RAG-SFT v2 对话并生成固定 768 的 assistant-only labels。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, BinaryIO

import torch
from torch.utils.data import Dataset

from rag.answering import SYSTEM_PROMPT, AnswerProtocolError, EvidencePackage, parse_and_validate_answer
from rag.core import Evidence

MAX_SEQ_LEN = 768
MAX_PROMPT_TOKENS = 618
MAX_OUTPUT_TOKENS = 150
LABEL_MASK_VERSION = "assistant_json_and_eos_only_v2"
PIPELINE = "rag_sft_v2_oracle_clean_materialization"
FORMAL_PIPELINE = "rag_sft_v2_training_release"
OUTPUT_KEYS = {
    PIPELINE: "oracle_clean",
    FORMAL_PIPELINE: "training_candidate",
}


class RagSftV2DatasetError(RuntimeError):
    """RAG-SFT v2 数据身份、协议或标签无效。"""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _token_ids(tokenizer: Any, text: str) -> list[int]:
    try:
        encoded = tokenizer(text, add_special_tokens=True, truncation=False)
        values = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    except Exception as error:
        raise RagSftV2DatasetError("Tokenizer 编码失败") from error
    if not isinstance(values, list) or any(type(value) is not int for value in values):
        raise RagSftV2DatasetError("Tokenizer 未返回整数 input_ids")
    return values


def _render(tokenizer: Any, conversations: list[dict[str, str]], *, generation: bool) -> str:
    template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(template):
        raise RagSftV2DatasetError("Tokenizer 缺少 apply_chat_template")
    try:
        rendered = template(
            conversations,
            tokenize=False,
            add_generation_prompt=generation,
            tools=None,
            open_thinking=False,
        )
    except Exception as error:
        raise RagSftV2DatasetError("chat template 渲染失败") from error
    if not isinstance(rendered, str) or not rendered:
        raise RagSftV2DatasetError("chat template 未返回非空文本")
    return rendered


def _identity(path: Path) -> dict[str, object]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def validate_manifest_sha256(manifest_path: str | Path) -> str:
    """验证 manifest.sha256 中唯一的 manifest 自身条目。"""

    path = Path(manifest_path).resolve()
    sidecar = path.with_suffix(".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise RagSftV2DatasetError(f"v2 manifest 或 SHA-256 清单不存在: {path}")
    digest = _sha256_file(path)
    try:
        lines = sidecar.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise RagSftV2DatasetError("无法读取 v2 manifest SHA-256 清单") from error
    matches = []
    for line in lines:
        if not line:
            continue
        parts = line.split("  ", 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise RagSftV2DatasetError("v2 manifest SHA-256 清单格式无效")
        try:
            int(parts[0], 16)
        except ValueError as error:
            raise RagSftV2DatasetError("v2 manifest SHA-256 清单格式无效") from error
        if Path(parts[1]).name == path.name:
            matches.append((parts[0].lower(), parts[1]))
    if matches != [(digest, path.name)]:
        raise RagSftV2DatasetError("v2 manifest SHA-256 校验失败")
    return digest


def _manifest_file(
    manifest: dict[str, Any], key: str, candidate_path: Path | None
) -> tuple[Path, dict[str, Any]]:
    output = manifest.get("output")
    if not isinstance(output, dict):
        raise RagSftV2DatasetError("v2 manifest 缺少 output")
    value = output.get(key)
    if not isinstance(value, dict):
        raise RagSftV2DatasetError(f"v2 manifest 缺少 output.{key}")
    path_value = value.get("path")
    if not isinstance(path_value, str):
        raise RagSftV2DatasetError(f"v2 manifest output.{key}.path 无效")
    path = (candidate_path or Path(path_value)).resolve()
    if not path.is_file():
        raise RagSftV2DatasetError(f"v2 数据文件不存在: {path}")
    actual = _identity(path)
    if any(value.get(field) != actual[field] for field in ("bytes", "sha256")):
        raise RagSftV2DatasetError(f"v2 数据文件身份已变化: {path}")
    return path, value


def _validate_manifest(manifest_path: Path, manifest: dict[str, Any], candidate_path: Path | None) -> tuple[Path, int]:
    pipeline = manifest.get("pipeline")
    if pipeline not in {PIPELINE, FORMAL_PIPELINE}:
        raise RagSftV2DatasetError("不是受支持的 RAG-SFT v2 manifest")
    readiness = manifest.get("readiness")
    if not isinstance(readiness, dict) or (
        pipeline == PIPELINE and readiness.get("oracle_clean_audited") is not True
    ):
        raise RagSftV2DatasetError("v2 manifest 尚未完成数据审计")
    if pipeline == FORMAL_PIPELINE and (
        readiness.get("training_ready") is not True
        or readiness.get("real_hard_negatives_constructed") is not True
        or readiness.get("final_retrieval_identity_frozen") is not True
        or manifest.get("complete") is not True
    ):
        raise RagSftV2DatasetError("v2 正式 release 的 HN、检索身份或完成状态未闭合")
    protocol = manifest.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("model_output") != ["summary", "citations"]:
        raise RagSftV2DatasetError("v2 manifest 不是两字段协议")
    if protocol.get("context_limit") != MAX_SEQ_LEN or protocol.get("max_prompt_tokens") != MAX_PROMPT_TOKENS or protocol.get("max_output_tokens") != MAX_OUTPUT_TOKENS:
        raise RagSftV2DatasetError("v2 manifest 长度预算无效")
    output_key = OUTPUT_KEYS[pipeline]
    path, metadata = _manifest_file(manifest, output_key, candidate_path)
    records = metadata.get("records")
    if type(records) is not int or records <= 0:
        raise RagSftV2DatasetError("v2 candidate 记录数无效")
    return path, records


def _validate_tokenizer_identity(
    tokenizer: Any, tokenizer_path: str | Path | None, report: object
) -> dict[str, Any]:
    if not isinstance(report, dict) or len(tokenizer) != report.get("vocab_size"):
        raise RagSftV2DatasetError("Tokenizer 词表身份不一致")
    template = getattr(tokenizer, "chat_template", None)
    if (
        not isinstance(template, str)
        or hashlib.sha256(template.encode("utf-8")).hexdigest()
        != report.get("chat_template_sha256")
    ):
        raise RagSftV2DatasetError("Tokenizer chat template 身份不一致")
    files = report.get("files")
    if not isinstance(files, dict) or not files:
        raise RagSftV2DatasetError("v2 manifest 缺少 Tokenizer 文件身份")
    raw_path = tokenizer_path or getattr(tokenizer, "name_or_path", None)
    if not isinstance(raw_path, (str, Path)):
        raise RagSftV2DatasetError("无法确定 Tokenizer 目录")
    root = Path(raw_path).resolve()
    actual_files = {}
    for filename, expected in files.items():
        if not isinstance(filename, str) or Path(filename).name != filename or not isinstance(expected, dict):
            raise RagSftV2DatasetError("v2 manifest Tokenizer 文件身份无效")
        path = root / filename
        if not path.is_file():
            raise RagSftV2DatasetError(f"Tokenizer 指纹文件不存在: {path}")
        actual = {"bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        if actual != {"bytes": expected.get("bytes"), "sha256": expected.get("sha256")}:
            raise RagSftV2DatasetError(f"Tokenizer 文件身份不一致: {filename}")
        actual_files[filename] = actual
    return {
        "path": str(root),
        "vocab_size": len(tokenizer),
        "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
        "files": actual_files,
    }


def _validate_record(record: object, line_number: int) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise RagSftV2DatasetError(f"v2 candidate 第 {line_number} 条不是 object")
    expected = {"id", "query_id", "variant", "query_original", "visible_chunk_ids", "required_chunk_ids", "citations", "conversations"}
    variant = record.get("variant")
    if set(record) != expected or variant not in {"clean", "hard_negative"}:
        raise RagSftV2DatasetError(f"v2 candidate 第 {line_number} 条字段或 variant 无效")
    if not all(isinstance(record.get(key), str) and record[key] for key in ("id", "query_id", "query_original")):
        raise RagSftV2DatasetError(f"v2 candidate 第 {line_number} 条身份无效")
    for key, minimum, maximum in (("visible_chunk_ids", 1, 5), ("required_chunk_ids", 1, 3), ("citations", 1, 3)):
        value = record.get(key)
        if not isinstance(value, list) or not minimum <= len(value) <= maximum or len(value) != len(set(value)) or any(not isinstance(item, str) or not item for item in value):
            raise RagSftV2DatasetError(f"v2 candidate 第 {line_number} 条 {key} 无效")
    visible = record["visible_chunk_ids"]
    required = record["required_chunk_ids"]
    if not set(required).issubset(visible):
        raise RagSftV2DatasetError(f"v2 candidate 第 {line_number} 条缺少 required evidence")
    if variant == "clean" and visible != required:
        raise RagSftV2DatasetError(f"v2 candidate 第 {line_number} 条 clean evidence 不一致")
    expected_citations = [f"E{index + 1}" for index, chunk_id in enumerate(visible) if chunk_id in set(required)]
    if record["citations"] != expected_citations:
        raise RagSftV2DatasetError(f"v2 candidate 第 {line_number} 条 citations 映射无效")
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 3 or [item.get("role") for item in conversations if isinstance(item, dict)] != ["system", "user", "assistant"]:
        raise RagSftV2DatasetError(f"v2 candidate 第 {line_number} 条 conversations 无效")
    if any(not isinstance(item, dict) or set(item) != {"role", "content"} or not isinstance(item["content"], str) or not item["content"] for item in conversations):
        raise RagSftV2DatasetError(f"v2 candidate 第 {line_number} 条消息无效")
    if conversations[0]["content"] != SYSTEM_PROMPT:
        raise RagSftV2DatasetError(f"v2 candidate 第 {line_number} 条 system prompt 不一致")
    try:
        user = json.loads(conversations[1]["content"])
        assistant = json.loads(conversations[2]["content"])
    except json.JSONDecodeError as error:
        raise RagSftV2DatasetError(f"v2 candidate 第 {line_number} 条 JSON 无效") from error
    if not isinstance(user, dict) or set(user) != {"query", "evidence"} or user.get("query") != record["query_original"]:
        raise RagSftV2DatasetError(f"v2 candidate 第 {line_number} 条 user JSON 无效")
    if not isinstance(assistant, dict) or set(assistant) != {"summary", "citations"} or assistant.get("citations") != record["citations"]:
        raise RagSftV2DatasetError(f"v2 candidate 第 {line_number} 条 assistant JSON 无效")
    evidence = user.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != len(record["visible_chunk_ids"]):
        raise RagSftV2DatasetError(f"v2 candidate 第 {line_number} 条 evidence 无效")
    for index, item in enumerate(evidence, 1):
        if (
            not isinstance(item, dict)
            or set(item) != {"evidence_id", "law_name", "article_no", "excerpts"}
            or item.get("evidence_id") != f"E{index}"
            or not isinstance(item.get("law_name"), str)
            or not item["law_name"]
            or not isinstance(item.get("article_no"), str)
            or not item["article_no"]
            or not isinstance(item.get("excerpts"), list)
            or len(item["excerpts"]) != 1
            or not isinstance(item["excerpts"][0], str)
            or not item["excerpts"][0]
        ):
            raise RagSftV2DatasetError(f"v2 candidate 第 {line_number} 条 evidence item 无效")
    package = EvidencePackage(
        query=record["query_original"],
        evidence=tuple(Evidence(item["law_name"], item["article_no"], item["excerpts"][0]) for item in evidence),
    )
    try:
        parse_and_validate_answer(package, conversations[2]["content"])
    except (AnswerProtocolError, KeyError, IndexError, TypeError) as error:
        raise RagSftV2DatasetError(f"v2 candidate 第 {line_number} 条协议无效") from error
    return record


class RagSftV2Dataset(Dataset):
    """直接读取 v2 candidate，返回固定 768 的 input_ids 与 labels。"""

    def __init__(
        self,
        manifest_path: str | Path,
        tokenizer: Any,
        *,
        candidate_path: str | Path | None = None,
        tokenizer_path: str | Path | None = None,
    ):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest_sha256 = validate_manifest_sha256(self.manifest_path)
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RagSftV2DatasetError("无法读取 v2 manifest") from error
        if not isinstance(manifest, dict):
            raise RagSftV2DatasetError("v2 manifest 必须是 object")
        self.candidate_path, expected_records = _validate_manifest(self.manifest_path, manifest, Path(candidate_path) if candidate_path else None)
        self._output_key = OUTPUT_KEYS[manifest["pipeline"]]
        self.tokenizer = tokenizer
        self.pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if type(self.pad_token_id) is not int:
            raise RagSftV2DatasetError("Tokenizer 缺少整数 pad_token_id")
        self.tokenizer_identity = _validate_tokenizer_identity(
            tokenizer, tokenizer_path, manifest.get("tokenizer")
        )
        self._index: list[tuple[int, int]] = []
        self.variant_counts = {"clean": 0, "hard_negative": 0}
        variant_indices: dict[str, list[int]] = {
            "clean": [],
            "hard_negative": [],
        }
        digest = hashlib.sha256()
        offset = 0
        with self.candidate_path.open("rb") as source:
            for line_number, raw in enumerate(source, 1):
                if not raw.strip():
                    raise RagSftV2DatasetError(f"v2 candidate 第 {line_number} 条为空")
                try:
                    indexed_record = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise RagSftV2DatasetError(
                        f"v2 candidate 第 {line_number} 条不是 UTF-8 JSON"
                    ) from error
                variant = indexed_record.get("variant") if isinstance(indexed_record, dict) else None
                if variant not in self.variant_counts:
                    raise RagSftV2DatasetError(
                        f"v2 candidate 第 {line_number} 条 variant 无效"
                    )
                self.variant_counts[variant] += 1
                variant_indices[variant].append(len(self._index))
                self._index.append((offset, line_number))
                digest.update(raw)
                offset += len(raw)
        if len(self._index) != expected_records or offset != self.candidate_path.stat().st_size or digest.hexdigest() != manifest["output"][self._output_key]["sha256"]:
            raise RagSftV2DatasetError("v2 candidate 身份已变化")
        if manifest["pipeline"] == FORMAL_PIPELINE and self.variant_counts["hard_negative"] == 0:
            raise RagSftV2DatasetError("v2 正式 candidate 未实际包含 hard_negative")
        self.variant_indices = {
            name: tuple(indices) for name, indices in variant_indices.items()
        }
        self._handle: BinaryIO | None = None
        self._pid = os.getpid()

    def __len__(self) -> int:
        return len(self._index)

    def _get_handle(self) -> BinaryIO:
        if self._pid != os.getpid():
            self.close()
            self._pid = os.getpid()
        if self._handle is None or self._handle.closed:
            self._handle = self.candidate_path.open("rb")
        return self._handle

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("RAG-SFT v2 Dataset index 越界")
        offset, line_number = self._index[index]
        handle = self._get_handle()
        handle.seek(offset)
        record = _validate_record(json.loads(handle.readline().decode("utf-8")), line_number)
        conversations = record["conversations"]
        prompt_ids = _token_ids(self.tokenizer, _render(self.tokenizer, conversations[:2], generation=True))
        full_ids = _token_ids(self.tokenizer, _render(self.tokenizer, conversations, generation=False))
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RagSftV2DatasetError(f"第 {line_number} 条 chat template 未保留 prompt 前缀")
        assistant_ids = full_ids[len(prompt_ids) :]
        if len(prompt_ids) > MAX_PROMPT_TOKENS or len(assistant_ids) > MAX_OUTPUT_TOKENS or len(full_ids) > MAX_SEQ_LEN or not assistant_ids:
            raise RagSftV2DatasetError(f"第 {line_number} 条超过 v2 运行时预算")
        input_ids = full_ids + [self.pad_token_id] * (MAX_SEQ_LEN - len(full_ids))
        labels = [-100] * MAX_SEQ_LEN
        labels[len(prompt_ids) : len(full_ids)] = assistant_ids
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


__all__ = [
    "LABEL_MASK_VERSION",
    "MAX_OUTPUT_TOKENS",
    "MAX_PROMPT_TOKENS",
    "MAX_SEQ_LEN",
    "RagSftV2Dataset",
    "RagSftV2DatasetError",
    "validate_manifest_sha256",
]
