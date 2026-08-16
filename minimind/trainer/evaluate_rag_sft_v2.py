"""RAG-SFT v2 两字段回答模型的固定 EvidencePackage 推理评估。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from rag.answering import EvidencePackage, build_answer_prompt, parse_and_validate_answer
from rag.answering.protocol import AnswerProtocolError
from rag.core import Evidence

from . import train_full_sft as base_entry


PIPELINE = "rag_sft_v2_answer_model_evaluation"
INPUT_PIPELINE = "rag_sft_v2_evaluation_inputs_v1"
MAX_NEW_TOKENS = 150
PRIMARY_MAX_SEQ_LEN = 768
EXTRAPOLATION_MAX_SEQ_LEN = 1024
EXPECTED_RECORDS = {"total": 170, "legal_query": 140, "exact_lookup": 30}
EXPECTED_EXTRAPOLATION_QUERY_IDS = ["Q033", "Q048", "Q063", "Q091"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _package(record: Mapping[str, Any]) -> EvidencePackage:
    query = record.get("query")
    evidence = record.get("evidence")
    if not isinstance(query, str) or not query or not isinstance(evidence, list) or not evidence:
        raise ValueError("评估记录 query/evidence 无效")
    values = []
    for index, item in enumerate(evidence, 1):
        if (
            not isinstance(item, dict)
            or set(item) != {"evidence_id", "chunk_id", "law_name", "article_no", "content"}
            or item.get("evidence_id") != f"E{index}"
            or item.get("chunk_id") != f"{item.get('law_name')}#{item.get('article_no')}"
        ):
            raise ValueError("评估记录 evidence 字段无效")
        values.append(Evidence(item["law_name"], item["article_no"], item["content"]))
    return EvidencePackage(query=query, evidence=tuple(values))


def _render_prompt(tokenizer: Any, package: EvidencePackage) -> str:
    rendered = tokenizer.apply_chat_template(
        build_answer_prompt(package),
        tokenize=False,
        add_generation_prompt=True,
        tools=None,
        open_thinking=False,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("chat template 未返回非空 prompt")
    return rendered


def _load_verified_json(path: str | Path, description: str) -> tuple[dict[str, Any], str]:
    resolved = Path(path).resolve()
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not resolved.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"{description} 或相邻 SHA-256 不存在: {resolved}")
    digest = _sha256_file(resolved)
    lines = [line for line in sidecar.read_text(encoding="utf-8").splitlines() if line]
    if lines != [f"{digest}  {resolved.name}"]:
        raise ValueError(f"{description} SHA-256 校验失败")
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} 不是有效 UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{description} 必须是 JSON object")
    return payload, digest


def _validate_case(record: dict[str, Any], line_number: int) -> None:
    expected = {
        "query_id",
        "query_type",
        "evaluation_scope",
        "query",
        "evidence",
        "visible_chunk_ids",
        "required_chunk_ids",
        "hard_negative_chunk_ids",
        "retrieval_attribution",
    }
    if (
        set(record) != expected
        or record.get("query_type") not in {"legal_query", "exact_lookup"}
        or record.get("evaluation_scope") not in {"primary_768", "rope_extrapolation_1024"}
    ):
        raise ValueError(f"评估 records 第 {line_number} 行 schema 无效")
    if not isinstance(record.get("query_id"), str) or not record["query_id"]:
        raise ValueError(f"评估 records 第 {line_number} 行身份无效")
    for key in ("visible_chunk_ids", "required_chunk_ids", "hard_negative_chunk_ids"):
        value = record.get(key)
        if not isinstance(value, list) or len(value) != len(set(value)) or any(not isinstance(item, str) or not item for item in value):
            raise ValueError(f"评估 records 第 {line_number} 行 {key} 无效")
    visible = record["visible_chunk_ids"]
    required = record["required_chunk_ids"]
    hard_negative = record["hard_negative_chunk_ids"]
    evidence_ids = [item.get("chunk_id") for item in record.get("evidence", []) if isinstance(item, dict)]
    if (
        not 1 <= len(visible) <= 5
        or not required
        or evidence_ids != visible
        or hard_negative != [chunk_id for chunk_id in visible if chunk_id not in set(required)]
        or not isinstance(record.get("retrieval_attribution"), dict)
    ):
        raise ValueError(f"评估 records 第 {line_number} 行证据身份无效")
    _package(record)


def _tokenizer_identity(tokenizer: Any, path: str | Path) -> dict[str, Any]:
    actual = base_entry._tokenizer_identity(tokenizer, path)
    return {
        "vocab_size": actual["vocab_size"],
        "chat_template_sha256": actual["chat_template_sha256"],
        "files": {
            name: {"bytes": item["bytes"], "sha256": item["sha256"]}
            for name, item in actual["files"].items()
        },
    }


def _manifest_tokenizer_identity(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("评估输入 manifest 缺少 Tokenizer 身份")
    files = value.get("files")
    if not isinstance(files, dict):
        raise ValueError("评估输入 manifest Tokenizer 文件身份无效")
    return {
        "vocab_size": value.get("vocab_size"),
        "chat_template_sha256": value.get("chat_template_sha256"),
        "files": {
            name: {"bytes": item.get("bytes"), "sha256": item.get("sha256")}
            for name, item in files.items()
            if isinstance(name, str) and isinstance(item, dict)
        },
    }


def _load_cases(
    manifest_path: str | Path, cases_path: str | Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest, manifest_sha256 = _load_verified_json(manifest_path, "评估输入 manifest")
    if (
        manifest.get("schema_version") != "1.0"
        or manifest.get("pipeline") != INPUT_PIPELINE
        or manifest.get("records") != EXPECTED_RECORDS
        or manifest.get("evaluation_scopes", {})
        .get("rope_extrapolation_1024", {})
        .get("query_ids")
        != EXPECTED_EXTRAPOLATION_QUERY_IDS
        or manifest.get("complete") is not True
    ):
        raise ValueError("评估输入 manifest 版本、计数或状态无效")
    metadata = manifest.get("output", {}).get("cases")
    if not isinstance(metadata, dict):
        raise ValueError("评估输入 manifest 缺少 cases 身份")
    case_path = Path(cases_path or metadata.get("path", "")).resolve()
    if not case_path.is_file():
        raise FileNotFoundError(f"评估 cases 不存在: {case_path}")
    if case_path.stat().st_size != metadata.get("bytes") or _sha256_file(case_path) != metadata.get("sha256"):
        raise ValueError("评估 cases 与 manifest 身份不一致")
    records = []
    seen = set()
    with case_path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                raise ValueError(f"评估 records 第 {line_number} 行为空")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"评估 records 第 {line_number} 行不是 object")
            _validate_case(value, line_number)
            if value["query_id"] in seen:
                raise ValueError("评估 records query_id 重复")
            seen.add(value["query_id"])
            records.append(value)
    counts = {
        kind: sum(record["query_type"] == kind for record in records)
        for kind in ("legal_query", "exact_lookup")
    }
    if counts != {"legal_query": 140, "exact_lookup": 30}:
        raise ValueError(f"评估 records 题型计数不闭合: {counts}")
    extrapolation_ids = [
        record["query_id"]
        for record in records
        if record["evaluation_scope"] == "rope_extrapolation_1024"
    ]
    if extrapolation_ids != EXPECTED_EXTRAPOLATION_QUERY_IDS:
        raise ValueError("RoPE 外推题身份或顺序不闭合")
    return records, {
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": manifest_sha256,
        "cases_path": str(case_path),
        "cases_sha256": metadata["sha256"],
        "tokenizer": _manifest_tokenizer_identity(manifest.get("tokenizer")),
    }


def evaluate_records(
    records: Sequence[Mapping[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    context_limit_override: int | None = None,
) -> list[dict[str, Any]]:
    model.eval()
    outputs = []
    with torch.inference_mode():
        for record in records:
            package = _package(record)
            prompt = _render_prompt(tokenizer, package)
            encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=False)
            input_ids = encoded["input_ids"].to(device)
            max_seq_len = context_limit_override or (
                EXTRAPOLATION_MAX_SEQ_LEN
                if record["evaluation_scope"] == "rope_extrapolation_1024"
                else PRIMARY_MAX_SEQ_LEN
            )
            if input_ids.shape[1] + MAX_NEW_TOKENS > max_seq_len:
                raise ValueError(f"{record['query_id']} prompt 超过 {max_seq_len}/150 预算")
            generated = model.generate(
                inputs=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )
            raw_text = tokenizer.decode(generated[0, input_ids.shape[1] :], skip_special_tokens=True)
            generated_tokens = int(generated.shape[1] - input_ids.shape[1])
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            eos_generated = bool(
                generated_tokens
                and eos_token_id is not None
                and generated[0, -1].item() == eos_token_id
            )
            parsed = None
            error = None
            try:
                answer = parse_and_validate_answer(package, raw_text)
                parsed = {"summary": answer.summary, "citations": list(answer.citations)}
            except (AnswerProtocolError, TypeError, ValueError) as exc:
                error = str(exc)
            cited_chunks = []
            if parsed is not None:
                for citation in parsed["citations"]:
                    cited_chunks.append(record["visible_chunk_ids"][int(citation[1:]) - 1])
            required = set(record["required_chunk_ids"])
            hard_negative = set(record["hard_negative_chunk_ids"])
            cited = set(cited_chunks)
            outputs.append({
                "query_id": record["query_id"],
                "query_type": record["query_type"],
                "evaluation_scope": record["evaluation_scope"],
                "visible_chunk_ids": record["visible_chunk_ids"],
                "required_chunk_ids": record["required_chunk_ids"],
                "hard_negative_chunk_ids": record["hard_negative_chunk_ids"],
                "retrieval_attribution": record["retrieval_attribution"],
                "raw_output": raw_text,
                "generation_metrics": {
                    "prompt_tokens": int(input_ids.shape[1]),
                    "generated_tokens": generated_tokens,
                    "hit_max_new_tokens": generated_tokens == MAX_NEW_TOKENS and not eos_generated,
                    "eos_generated": eos_generated,
                    "context_limit": max_seq_len,
                },
                "protocol_valid": parsed is not None,
                "parsed": parsed,
                "cited_chunk_ids": cited_chunks,
                "citation_metrics": {
                    "required_recall": len(cited & required) / len(required) if parsed is not None else 0.0,
                    "precision": len(cited & required) / len(cited) if cited else 0.0,
                    "exact_set": cited == required if parsed is not None else False,
                    "hard_negative_citations": len(cited & hard_negative),
                    "over_citations": len(cited - required),
                },
                "error": error,
            })
    return outputs


def _summary_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(records)
    valid = sum(bool(record["protocol_valid"]) for record in records)
    citation = [record["citation_metrics"] for record in records]
    hard_negative_cases = [
        record for record in records if record["hard_negative_chunk_ids"]
    ]
    return {"records": total, "protocol_valid": valid, "protocol_valid_rate": valid / total if total else 0.0, "invalid": total - valid, "citation": {"required_recall": sum(item["required_recall"] for item in citation) / total if total else 0.0, "precision": sum(item["precision"] for item in citation) / total if total else 0.0, "exact_set_accuracy": sum(item["exact_set"] for item in citation) / total if total else 0.0, "hard_negative_citations": sum(item["hard_negative_citations"] for item in citation), "hard_negative_citation_rate": sum(record["citation_metrics"]["hard_negative_citations"] > 0 for record in hard_negative_cases) / len(hard_negative_cases) if hard_negative_cases else 0.0, "over_citations": sum(item["over_citations"] for item in citation), "over_citation_rate": sum(item["over_citations"] > 0 for item in citation) / total if total else 0.0}}


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(records)
    primary = [record for record in values if record["evaluation_scope"] == "primary_768"]
    extrapolation = [
        record for record in values if record["evaluation_scope"] == "rope_extrapolation_1024"
    ]
    summary = _summary_metrics(primary)
    summary["total_generated_records"] = len(values)
    summary["rope_extrapolation"] = _summary_metrics(extrapolation)
    summary["strata"] = {
        "legal_query": _summary_metrics(
            [record for record in primary if record["query_type"] == "legal_query"]
        ),
        "exact_lookup": _summary_metrics(
            [record for record in primary if record["query_type"] == "exact_lookup"]
        ),
        "packaged_complete": _summary_metrics(
            [
                record
                for record in primary
                if record["query_type"] == "legal_query"
                and record["retrieval_attribution"].get("packaged_complete") is True
            ]
        ),
        "packaged_incomplete": _summary_metrics(
            [
                record
                for record in primary
                if record["query_type"] == "legal_query"
                and record["retrieval_attribution"].get("packaged_complete") is False
            ]
        ),
        "clean_evidence": _summary_metrics(
            [record for record in primary if not record["hard_negative_chunk_ids"]]
        ),
        "hard_negative_evidence": _summary_metrics(
            [record for record in primary if record["hard_negative_chunk_ids"]]
        ),
    }
    return summary


def run_evaluation(*, cases_manifest: str | Path, cases: str | Path | None, tokenizer_path: str | Path, weights: str | Path, weights_sha256: str, output: str | Path, device_name: str = "cuda:0") -> dict[str, Any]:
    records, records_identity = _load_cases(cases_manifest, cases)
    tokenizer = base_entry._load_tokenizer(tokenizer_path)
    tokenizer_identity = _tokenizer_identity(tokenizer, tokenizer_path)
    if tokenizer_identity != records_identity["tokenizer"]:
        raise ValueError("当前 Tokenizer 与评估输入 manifest 身份不一致")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("指定 CUDA 但当前无可用 GPU")
    model = base_entry.MiniMindForCausalLM(base_entry._model_config()).to(device)
    actual_weights_sha256 = base_entry._load_parent_weights(weights, weights_sha256, model)
    results = evaluate_records(records, model=model, tokenizer=tokenizer, device=device)
    source_paths = {
        "evaluation": Path(__file__).resolve(),
        "protocol": Path(__file__).resolve().parents[2] / "rag" / "answering" / "protocol.py",
        "evidence": Path(__file__).resolve().parents[2] / "rag" / "answering" / "evidence.py",
    }
    source_identities = {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}
        for name, path in source_paths.items()
    }
    report = {"schema_version": "1.0", "pipeline": PIPELINE, "inputs": {"evaluation": records_identity, "weights_sha256": actual_weights_sha256, "tokenizer": tokenizer_identity, "source_identities": source_identities}, "generation": {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False, "primary_max_seq_len": PRIMARY_MAX_SEQ_LEN, "extrapolation_max_seq_len": EXTRAPOLATION_MAX_SEQ_LEN, "inference_rope_scaling": False, "retry": False, "repair": False}, "summary": summarize(results), "records": results, "complete": True}
    output_path = Path(output).resolve()
    output_sidecar = output_path.with_suffix(output_path.suffix + ".sha256")
    if output_path.exists() or output_sidecar.exists():
        raise FileExistsError(f"评估输出已存在: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    digest = _sha256_file(output_path)
    output_sidecar.write_text(
        f"{digest}  {output_path.name}\n", encoding="utf-8", newline="\n"
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG-SFT v2 两字段回答模型评估")
    parser.add_argument("--cases_manifest", required=True)
    parser.add_argument("--cases")
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--weights_sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_evaluation(cases_manifest=args.cases_manifest, cases=args.cases, tokenizer_path=args.tokenizer_path, weights=args.weights, weights_sha256=args.weights_sha256, output=args.output, device_name=args.device)
    print(f"RAG_SFT_V2_EVALUATION_OK records={report['summary']['records']} protocol_valid_rate={report['summary']['protocol_valid_rate']:.6f}")


if __name__ == "__main__":
    main()
