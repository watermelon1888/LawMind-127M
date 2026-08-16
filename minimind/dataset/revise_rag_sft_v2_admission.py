"""基于冻结准入账本发布单条人工裁决后的不可覆盖修订版本。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


DATASET_ROOT = Path(__file__).resolve().parent
ADMISSION_V1_DIR = DATASET_ROOT / "RAG-SFT" / "review" / "v2" / "admission-v1"
DEFAULT_LEDGER = ADMISSION_V1_DIR / "admission-ledger.jsonl"
DEFAULT_MANIFEST = ADMISSION_V1_DIR / "manifest.json"
DEFAULT_OUTPUT_DIR = DATASET_ROOT / "RAG-SFT" / "review" / "v2" / "admission-v2"

REVISION = {
    "query_id": "query:0003",
    "from_decision": "admit",
    "decision": "exclude",
    "reason": "incomplete_support",
    "detail": (
        "问题笼统询问未订立书面劳动合同的责任，但冻结 required GT 仅覆盖"
        "超过一个月不满一年的二倍工资，无法完整覆盖满一年后的法定后果。"
    ),
}
_LEDGER_FIELDS = {"query_id", "decision", "reason", "detail"}


class RagSftV2AdmissionRevisionError(RuntimeError):
    """表示准入裁决修订无法安全发布。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(path: Path, *, records: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "path": str(path), "bytes": path.stat().st_size, "sha256": _sha256(path)
    }
    if records is not None:
        value["records"] = records
    return value


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RagSftV2AdmissionRevisionError(f"无法读取{description}: {path}") from error
    if not isinstance(value, dict):
        raise RagSftV2AdmissionRevisionError(f"{description}必须是 JSON object")
    return value


def _load_ledger(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, start=1):
                if not line.strip():
                    raise RagSftV2AdmissionRevisionError(f"准入账本不允许空行: {number}")
                row = json.loads(line)
                if (
                    not isinstance(row, dict)
                    or set(row) != _LEDGER_FIELDS
                    or any(not isinstance(row.get(field), str) or not row[field].strip() for field in _LEDGER_FIELDS)
                ):
                    raise RagSftV2AdmissionRevisionError(f"准入账本第 {number} 条字段无效")
                rows.append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        if isinstance(error, RagSftV2AdmissionRevisionError):
            raise
        raise RagSftV2AdmissionRevisionError(f"无法读取准入账本: {path}") from error
    return rows


def _publish(output_dir: Path, payloads: list[tuple[str, str]]) -> None:
    if output_dir.exists():
        raise RagSftV2AdmissionRevisionError(f"输出目录必须是新目录: {output_dir}")
    output_dir.mkdir(parents=True)
    pending = [(output_dir / f"{name}.partial", output_dir / name, payload) for name, payload in payloads]
    published: list[Path] = []
    try:
        for partial, _, payload in pending:
            partial.write_text(payload, encoding="utf-8", newline="\n")
        sha_payload = "".join(
            f"{_sha256(partial)}  {final.name}\n" for partial, final, _ in pending
        )
        sha_partial = output_dir / "manifest.sha256.partial"
        sha_partial.write_text(sha_payload, encoding="utf-8", newline="\n")
        for partial, final, _ in pending:
            partial.replace(final)
            published.append(final)
        sha_partial.replace(output_dir / "manifest.sha256")
    except (OSError, UnicodeError) as error:
        for path in [*(item[0] for item in pending), output_dir / "manifest.sha256.partial", *reversed(published)]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            output_dir.rmdir()
        except OSError:
            pass
        raise RagSftV2AdmissionRevisionError("无法发布准入修订账本") from error


def revise_rag_sft_v2_admission(
    *, admission_ledger_path: Path, admission_manifest_path: Path, output_dir: Path
) -> dict[str, object]:
    """将已裁决的单条 GT 不充分记录从准入账本中排除。"""

    ledger_path = Path(admission_ledger_path).resolve()
    manifest_path = Path(admission_manifest_path).resolve()
    output_dir = Path(output_dir).resolve()
    if not ledger_path.is_file() or not manifest_path.is_file():
        raise RagSftV2AdmissionRevisionError("准入修订输入缺失")
    previous_manifest = _load_json(manifest_path, "上一版准入 manifest")
    if (
        previous_manifest.get("release_status") != "admission_finalized"
        or previous_manifest.get("records", {}).get("admit") != 551
        or previous_manifest.get("records", {}).get("exclude") != 4
        or previous_manifest.get("output", {}).get("admission_ledger", {}).get("sha256") not in {None, _sha256(ledger_path)}
    ):
        raise RagSftV2AdmissionRevisionError("上一版准入账本身份或计数无效")
    rows = _load_ledger(ledger_path)
    if len(rows) != 555 or len({row["query_id"] for row in rows}) != 555:
        raise RagSftV2AdmissionRevisionError("上一版准入账本未闭合为 555 条唯一记录")
    target = [row for row in rows if row["query_id"] == REVISION["query_id"]]
    if len(target) != 1 or target[0]["decision"] != REVISION["from_decision"]:
        raise RagSftV2AdmissionRevisionError("修订目标不处于预期的准入状态")
    revised_rows: list[dict[str, str]] = []
    for row in rows:
        if row["query_id"] == REVISION["query_id"]:
            revised_rows.append({
                "query_id": row["query_id"],
                "decision": REVISION["decision"],
                "reason": REVISION["reason"],
                "detail": REVISION["detail"],
            })
        else:
            revised_rows.append(row)
    admit = sum(row["decision"] == "admit" for row in revised_rows)
    exclude = sum(row["decision"] == "exclude" for row in revised_rows)
    if (admit, exclude) != (550, 5):
        raise RagSftV2AdmissionRevisionError("修订后准入账本未闭合为 550 准入、5 排除")
    ledger_payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
        for row in revised_rows
    )
    manifest = {
        "pipeline": "rag_sft_v2_admission_revision",
        "release_status": "admission_finalized",
        "supersedes": _identity(manifest_path),
        "inputs": {"admission_ledger_v1": _identity(ledger_path, records=len(rows))},
        "revision": REVISION,
        "records": {"pending_claim_authoring": 555, "admit": admit, "exclude": exclude},
        "policy": {
            "public_query_pool_unchanged": True,
            "legacy_drafts_are_not_canonical": True,
            "canonical_authoring_emitted": False,
            "oracle_clean_emitted": False,
            "real_hard_negatives_constructed": False,
        },
        "output": {"admission_ledger": {"path": str(output_dir / "admission-ledger.jsonl"), "records": len(revised_rows)}},
        "readiness": {"admission_finalized": True, "authoring_ready": True, "training_ready": False},
        "complete": True,
    }
    manifest_payload = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    _publish(output_dir, [("admission-ledger.jsonl", ledger_payload), ("manifest.json", manifest_payload)])
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="发布 RAG-SFT v2 准入账本修订版本")
    parser.add_argument("--admission-ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--admission-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        manifest = revise_rag_sft_v2_admission(
            admission_ledger_path=args.admission_ledger,
            admission_manifest_path=args.admission_manifest,
            output_dir=args.output_dir,
        )
    except RagSftV2AdmissionRevisionError as error:
        raise SystemExit(f"[失败] {error}") from error
    print(f"[完成] 准入 {manifest['records']['admit']} 条，排除 {manifest['records']['exclude']} 条")


if __name__ == "__main__":
    main()
