"""为通过输入审核的 Query-SFT pilot 发布教师候选生成工作包。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from rag.query.enhancement import (
        QUERY_ENHANCEMENT_SCHEMA,
        QUERY_ENHANCEMENT_SYSTEM_PROMPT,
    )
except ImportError:  # 支持从 MiniMind 根目录执行模块。
    from ..rag.query.enhancement import (
        QUERY_ENHANCEMENT_SCHEMA,
        QUERY_ENHANCEMENT_SYSTEM_PROMPT,
    )

try:
    from .finalize_query_sft_pilot_input_review import (
        DEFAULT_OUTPUT_PATH as DEFAULT_INPUT_REVIEW_PATH,
    )
except ImportError:  # 支持从 MiniMind 根目录以 dataset 包运行。
    from dataset.finalize_query_sft_pilot_input_review import (
        DEFAULT_OUTPUT_PATH as DEFAULT_INPUT_REVIEW_PATH,
    )


TEACHER_CANDIDATE_COUNT = 3
DEFAULT_OUTPUT_DIR = (
    DEFAULT_INPUT_REVIEW_PATH.parent / "query-sft-pilot-v1-teacher-candidate-work-package"
)
QUEUE_FILENAME = "query-sft-pilot-v1-teacher-candidate-queue.jsonl"
NOOP_FILENAME = "query-sft-pilot-v1-deterministic-noop-candidates.jsonl"
MANIFEST_FILENAME = "query-sft-pilot-v1-teacher-candidate-work-package.json"
HASH_FILENAME = "query-sft-pilot-v1-teacher-candidate-work-package.sha256"

_INPUT_FIELDS = {"pilot_id", "query_original"}
_QUEUE_FIELDS = {"candidate_id", "pilot_id", "query_original"}
_NOOP_FIELDS = {"candidate_id", "pilot_id", "raw_output"}


class QuerySftPilotTeacherPreparationError(RuntimeError):
    """Query-SFT pilot 教师候选工作包无法安全发布。"""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    result: dict[str, object] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if records is not None:
        result["records"] = records
    return result


def _payload_identity(path: Path, payload: str, *, records: int) -> dict[str, object]:
    encoded = payload.encode("utf-8")
    return {"path": str(path.resolve()), "bytes": len(encoded), "sha256": _sha256_bytes(encoded), "records": records}


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QuerySftPilotTeacherPreparationError(
            f"无法读取{description}: {path}"
        ) from error
    if not isinstance(value, dict):
        raise QuerySftPilotTeacherPreparationError(f"{description}必须是 JSON object")
    return value


def _verify_adjacent_hash(path: Path, description: str) -> Path:
    hash_path = path.with_suffix(".sha256")
    try:
        lines = hash_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise QuerySftPilotTeacherPreparationError(
            f"无法读取{description}相邻 SHA-256: {hash_path}"
        ) from error
    if lines != [f"{_sha256_file(path)}  {path.name}"]:
        raise QuerySftPilotTeacherPreparationError(f"{description}相邻 SHA-256 无效")
    return hash_path


def _validate_input_review(manifest: dict[str, Any]) -> list[dict[str, str]]:
    if (
        manifest.get("pipeline") != "query_sft_pilot_input_review_v1"
        or manifest.get("release_status") != "pilot_teacher_candidate_input"
        or manifest.get("complete") is not True
        or manifest.get("readiness", {}).get("teacher_candidate_work_package_ready")
        is not True
        or manifest.get("validation", {}).get("teacher_candidate_inputs_exclude_gt")
        is not True
    ):
        raise QuerySftPilotTeacherPreparationError("输入审核 manifest 状态无效")
    inputs = manifest.get("records", {}).get("teacher_candidate_inputs")
    approved = manifest.get("records", {}).get("approved")
    rejected = manifest.get("records", {}).get("rejected")
    if (
        type(approved) is not int
        or type(rejected) is not int
        or approved <= 0
        or rejected <= 0
        or not isinstance(inputs, list)
        or len(inputs) != approved
    ):
        raise QuerySftPilotTeacherPreparationError("输入审核 manifest 候选数量无效")
    result = []
    seen = set()
    for position, item in enumerate(inputs, start=1):
        if not isinstance(item, dict) or set(item) != _INPUT_FIELDS:
            raise QuerySftPilotTeacherPreparationError(
                f"输入审核 manifest 第 {position} 条教师输入字段无效"
            )
        pilot_id = item.get("pilot_id")
        query_original = item.get("query_original")
        if (
            not isinstance(pilot_id, str)
            or not pilot_id
            or pilot_id in seen
            or not isinstance(query_original, str)
            or not query_original.strip()
        ):
            raise QuerySftPilotTeacherPreparationError(
                f"输入审核 manifest 第 {position} 条教师输入无效或重复"
            )
        seen.add(pilot_id)
        result.append({"pilot_id": pilot_id, "query_original": query_original})
    return result


def _jsonl_payload(records: list[dict[str, str]]) -> str:
    return "".join(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        + "\n"
        for record in records
    )


def _publish(output_dir: Path, payloads: list[tuple[str, str]]) -> None:
    if output_dir.exists():
        raise QuerySftPilotTeacherPreparationError(f"输出目录必须不存在: {output_dir}")
    published: list[Path] = []
    partials: list[tuple[Path, Path]] = []
    try:
        output_dir.mkdir(parents=True)
        for filename, payload in payloads:
            partial = output_dir / f"{filename}.partial"
            final = output_dir / filename
            partial.write_text(payload, encoding="utf-8", newline="\n")
            partials.append((partial, final))
        hash_lines = [
            f"{_sha256_bytes(payload.encode('utf-8'))}  {filename}\n"
            for filename, payload in payloads
        ]
        hash_partial = output_dir / f"{HASH_FILENAME}.partial"
        hash_partial.write_text("".join(hash_lines), encoding="utf-8", newline="\n")
        for partial, final in partials:
            partial.replace(final)
            published.append(final)
        hash_partial.replace(output_dir / HASH_FILENAME)
    except (OSError, UnicodeError) as error:
        for path in [
            *(partial for partial, _ in partials),
            output_dir / f"{HASH_FILENAME}.partial",
            *reversed(published),
        ]:
            path.unlink(missing_ok=True)
        try:
            output_dir.rmdir()
        except OSError:
            pass
        raise QuerySftPilotTeacherPreparationError("无法发布教师候选工作包") from error


def prepare_query_sft_pilot_teacher_candidates(
    *, input_review_path: Path, output_dir: Path
) -> dict[str, object]:
    """创建每题三次独立教师调用及一个确定性 no-op 的可审计队列。"""

    input_review_path = Path(input_review_path).resolve()
    output_dir = Path(output_dir).resolve()
    input_review_hash_path = _verify_adjacent_hash(input_review_path, "输入审核 manifest")
    inputs = _validate_input_review(_load_json(input_review_path, "输入审核 manifest"))
    queue: list[dict[str, str]] = []
    noops: list[dict[str, str]] = []
    for item in inputs:
        pilot_id = item["pilot_id"]
        query_original = item["query_original"]
        for slot in range(1, TEACHER_CANDIDATE_COUNT + 1):
            queue.append(
                {
                    "candidate_id": f"{pilot_id}/teacher-{slot}",
                    "pilot_id": pilot_id,
                    "query_original": query_original,
                }
            )
        noops.append(
            {
                "candidate_id": f"{pilot_id}/noop",
                "pilot_id": pilot_id,
                "raw_output": json.dumps(
                    {
                        "rewrite": query_original,
                        "expansion_terms": [],
                        "subqueries": [],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                ),
            }
        )
    if len({item["candidate_id"] for item in queue}) != len(queue):
        raise AssertionError("教师 candidate_id 意外重复")

    queue_path = output_dir / QUEUE_FILENAME
    noop_path = output_dir / NOOP_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME
    queue_payload = _jsonl_payload(queue)
    noop_payload = _jsonl_payload(noops)
    manifest: dict[str, object] = {
        "pipeline": "query_sft_pilot_teacher_candidate_work_package_v1",
        "release_status": "pilot_teacher_candidate_work_package",
        "inputs": {
            "input_review_manifest": {
                **_identity(input_review_path),
                "hash_manifest": _identity(input_review_hash_path),
            }
        },
        "generation_protocol": {
            "teacher_visible_record_fields": ["pilot_id", "query_original"],
            "teacher_prompt_builder": "rag.query.enhancement.build_query_enhancement_prompt",
            "system_prompt_sha256": _sha256_bytes(
                QUERY_ENHANCEMENT_SYSTEM_PROMPT.encode("utf-8")
            ),
            "output_schema": QUERY_ENHANCEMENT_SCHEMA,
            "teacher_candidates_per_input": TEACHER_CANDIDATE_COUNT,
            "candidate_slots_are_independent": True,
            "second_round_context_forbidden": [
                "required_gt",
                "canonical_answer",
                "law_text",
                "evaluation_question",
                "retrieval_score",
                "other_candidate_output",
            ],
            "runtime_identity_required_before_execution": [
                "teacher_model_id",
                "teacher_endpoint_or_local_checkpoint",
                "decoding_temperature",
                "decoding_seed_or_nondeterminism_declaration",
                "max_output_tokens",
            ],
        },
        "records": {
            "approved_inputs": len(inputs),
            "teacher_candidate_requests": len(queue),
            "deterministic_noop_candidates": len(noops),
            "candidate_options_per_input": TEACHER_CANDIDATE_COUNT + 1,
        },
        "validation": {
            "input_review_manifest_identity_bound": True,
            "teacher_queue_contains_only_original_query": True,
            "three_independent_teacher_slots_per_input": True,
            "deterministic_noop_has_strict_protocol_shape": True,
            "teacher_prompt_excludes_gt_and_retrieval_feedback": True,
        },
        "outputs": {
            "teacher_candidate_queue": _payload_identity(
                queue_path, queue_payload, records=len(queue)
            ),
            "deterministic_noop_candidates": _payload_identity(
                noop_path, noop_payload, records=len(noops)
            ),
            "manifest": MANIFEST_FILENAME,
            "sha256_manifest": HASH_FILENAME,
        },
        "readiness": {
            "teacher_candidate_work_package_ready": True,
            "teacher_runtime_identity_configured": False,
            "teacher_generation_completed": False,
            "candidate_protocol_audit_completed": False,
            "retrieval_evaluation_ready": False,
            "query_sft_training_ready": False,
        },
        "complete": True,
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _publish(
        output_dir,
        [
            (QUEUE_FILENAME, queue_payload),
            (NOOP_FILENAME, noop_payload),
            (MANIFEST_FILENAME, manifest_payload),
        ],
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-review", type=Path, default=DEFAULT_INPUT_REVIEW_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        manifest = prepare_query_sft_pilot_teacher_candidates(
            input_review_path=args.input_review, output_dir=args.output_dir
        )
    except QuerySftPilotTeacherPreparationError as error:
        parser.error(str(error))
    print(
        "[完成] 教师候选工作包发布 "
        f"requests={manifest['records']['teacher_candidate_requests']} "
        f"noop={manifest['records']['deterministic_noop_candidates']}"
    )
    print("[待完成] 配置真实教师运行时后执行三候选生成")
    print(f"工作包: {args.output_dir}")


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "HASH_FILENAME",
    "MANIFEST_FILENAME",
    "NOOP_FILENAME",
    "QUEUE_FILENAME",
    "QuerySftPilotTeacherPreparationError",
    "TEACHER_CANDIDATE_COUNT",
    "prepare_query_sft_pilot_teacher_candidates",
]
