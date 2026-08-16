"""离线导入国家法律法规数据库的手动下载归档。"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable


EXPECTED_QUERY_RECORDS = 17_337
RECOVERED_ARCHIVE_ORDERS = frozenset({23, 95, 129, 154, 165})
DOCUMENT_COUNT_OVERRIDES = {
    11: 99,
    17: 98,
    21: 99,
    22: 98,
    31: 99,
    40: 98,
    50: 99,
    55: 99,
    66: 99,
    71: 99,
    75: 99,
    76: 97,
    78: 99,
    93: 99,
    95: 99,
    105: 99,
    110: 99,
    122: 98,
    129: 98,
    133: 99,
    134: 99,
    138: 99,
    151: 99,
    162: 99,
    174: 37,
}
EXPECTED_DOCUMENT_COUNTS = tuple(
    DOCUMENT_COUNT_OVERRIDES.get(order, 100) for order in range(1, 175)
)
EXPECTED_DOCUMENTS = sum(EXPECTED_DOCUMENT_COUNTS)
ALLOWED_EXTENSIONS = frozenset({".doc", ".docx", ".docm"})
RECOVERED_DIRECTORY_PATTERN = re.compile(r"^(\d{3})_")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_member_path(name: str) -> str:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or normalized.startswith("/")
        or re.match(r"^[A-Za-z]:", normalized)
        or ".." in path.parts
    ):
        raise ValueError(f"归档包含越界成员路径: {name}")
    return path.as_posix()


def _validate_member_names(names: Iterable[str]) -> list[str]:
    normalized = [_normalized_member_path(name) for name in names]
    folded = [name.casefold() for name in normalized]
    if len(set(folded)) != len(folded):
        raise ValueError("归档包含重复成员路径")
    for name in normalized:
        extension = Path(name).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise ValueError(f"归档包含不支持的文件格式: {name}")
    return sorted(normalized, key=lambda value: (value.casefold(), value))


def _find_recovered_directories(
    recovered_root: Path,
    expected_orders: set[int] | frozenset[int],
) -> dict[int, Path]:
    if not recovered_root.is_dir():
        raise ValueError(f"恢复目录不存在: {recovered_root}")
    directories: dict[int, Path] = {}
    for path in recovered_root.iterdir():
        if not path.is_dir():
            continue
        match = RECOVERED_DIRECTORY_PATTERN.match(path.name)
        if match is None:
            continue
        order = int(match.group(1))
        if order in directories:
            raise ValueError(f"第 {order} 包存在多个恢复目录")
        directories[order] = path
    if set(directories) != set(expected_orders):
        raise ValueError(
            "恢复目录顺序号不匹配: "
            f"期望 {sorted(expected_orders)}，实际 {sorted(directories)}"
        )
    return directories


def _write_document(target: Path, data: bytes) -> tuple[int, str]:
    digest = _sha256_bytes(data)
    if target.exists():
        if _sha256_file(target) != digest:
            raise ValueError(f"已存在文件与本次导入内容不一致: {target}")
    else:
        target.write_bytes(data)
    return len(data), digest


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _inventory_record(
    *,
    document_id: str,
    archive_order: int,
    archive_name: str,
    archive_sha256: str,
    member_path: str,
    input_mode: str,
    data: bytes,
    output_root: Path,
) -> dict[str, object]:
    extension = Path(member_path).suffix.lower()
    relative_path = Path("documents") / "original" / f"{document_id}{extension}"
    size, digest = _write_document(output_root / relative_path, data)
    converted_path = (
        (Path("documents") / "converted" / f"{document_id}.docx").as_posix()
        if extension == ".doc"
        else None
    )
    return {
        "document_id": document_id,
        "archive_order": archive_order,
        "source_archive": archive_name,
        "source_archive_sha256": archive_sha256,
        "source_member_path": member_path,
        "input_mode": input_mode,
        "original_extension": extension,
        "original_path": relative_path.as_posix(),
        "converted_path": converted_path,
        "file_size": size,
        "sha256": digest,
    }


def import_archives(
    archive_root: Path,
    recovered_root: Path,
    output_root: Path,
    expected_document_counts: tuple[int, ...] = EXPECTED_DOCUMENT_COUNTS,
    recovered_archive_orders: set[int] | frozenset[int] = RECOVERED_ARCHIVE_ORDERS,
) -> dict[str, object]:
    """导入标准 ZIP 与指定恢复目录，并生成归档和逐文档清单。"""
    archive_root = archive_root.resolve()
    recovered_root = recovered_root.resolve()
    output_root = output_root.resolve()
    if not archive_root.is_dir():
        raise ValueError(f"归档目录不存在: {archive_root}")

    archives = sorted(archive_root.glob("*.zip"), key=lambda path: path.name)
    if len(archives) != len(expected_document_counts):
        raise ValueError(
            f"归档数量应为 {len(expected_document_counts)}，实际为 {len(archives)}"
        )
    recovered_directories = _find_recovered_directories(
        recovered_root, recovered_archive_orders
    )

    original_dir = output_root / "documents" / "original"
    converted_dir = output_root / "documents" / "converted"
    original_dir.mkdir(parents=True, exist_ok=True)
    converted_dir.mkdir(parents=True, exist_ok=True)

    archive_records: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []
    for archive_order, (archive_path, expected_count) in enumerate(
        zip(archives, expected_document_counts), start=1
    ):
        archive_sha256 = _sha256_file(archive_path)
        input_mode = "recovered_directory" if archive_order in recovered_archive_orders else "zip"
        members: list[tuple[str, bytes]] = []
        recovery_directory: str | None = None
        recovery_archive: Path | None = None
        recovery_archive_sha256: str | None = None
        source_archive_name = archive_path.name
        source_archive_sha256 = archive_sha256
        if input_mode == "recovered_directory":
            recovered_dir = recovered_directories[archive_order]
            recovery_directory = recovered_dir.name
            recovery_archive = recovered_root / f"{recovered_dir.name}.zip"
            if not recovery_archive.is_file():
                raise ValueError(f"缺少恢复目录对应的复核 ZIP: {recovery_archive}")
            recovery_archive_sha256 = _sha256_file(recovery_archive)
            source_archive_name = recovery_archive.name
            source_archive_sha256 = recovery_archive_sha256
            files = [path for path in recovered_dir.rglob("*") if path.is_file()]
            relative_paths = _validate_member_names(
                path.relative_to(recovered_dir).as_posix() for path in files
            )
            files_by_name = {
                path.relative_to(recovered_dir).as_posix(): path for path in files
            }
            members = [(name, files_by_name[name].read_bytes()) for name in relative_paths]
        else:
            try:
                with zipfile.ZipFile(archive_path) as archive:
                    infos = [info for info in archive.infolist() if not info.is_dir()]
                    if any(info.flag_bits & 0x1 for info in infos):
                        raise ValueError(f"归档包含加密成员: {archive_path.name}")
                    normalized_names = _validate_member_names(info.filename for info in infos)
                    infos_by_name = {
                        _normalized_member_path(info.filename): info for info in infos
                    }
                    members = [
                        (name, archive.read(infos_by_name[name])) for name in normalized_names
                    ]
            except (OSError, zipfile.BadZipFile) as error:
                raise ValueError(f"无法读取归档: {archive_path}") from error

        if len(members) != expected_count:
            raise ValueError(
                f"第 {archive_order} 包文档数应为 {expected_count}，实际为 {len(members)}"
            )

        for member_index, (member_path, data) in enumerate(members, start=1):
            document_id = f"npc-{archive_order:03d}-{member_index:03d}"
            inventory.append(
                _inventory_record(
                    document_id=document_id,
                    archive_order=archive_order,
                    archive_name=source_archive_name,
                    archive_sha256=source_archive_sha256,
                    member_path=member_path,
                    input_mode=input_mode,
                    data=data,
                    output_root=output_root,
                )
            )

        archive_records.append(
            {
                "archive_order": archive_order,
                "archive_name": archive_path.name,
                "file_size": archive_path.stat().st_size,
                "sha256": archive_sha256,
                "expected_documents": expected_count,
                "imported_documents": len(members),
                "input_mode": input_mode,
                "recovery_directory": recovery_directory,
                "recovery_archive": recovery_archive.name if recovery_archive else None,
                "recovery_archive_size": recovery_archive.stat().st_size if recovery_archive else None,
                "recovery_archive_sha256": recovery_archive_sha256,
            }
        )

    manifest = {
        "source": "https://flk.npc.gov.cn/search",
        "query_record_count": EXPECTED_QUERY_RECORDS,
        "archive_count": len(archives),
        "document_count": len(inventory),
        "records_without_exported_document": EXPECTED_QUERY_RECORDS - len(inventory),
        "recovered_archive_orders": sorted(recovered_archive_orders),
        "archives": archive_records,
    }
    _write_json(output_root / "manual-download-manifest.json", manifest)
    _write_jsonl(output_root / "document-inventory.jsonl", inventory)
    return manifest


def main() -> None:
    """解析命令行参数并执行离线归档导入。"""
    parser = argparse.ArgumentParser(description="导入国家法律法规数据库手动下载归档")
    parser.add_argument("--archive-root", type=Path, required=True, help="174 个原始 ZIP 所在目录")
    parser.add_argument("--recovered-root", type=Path, required=True, help="5 个手工恢复目录所在目录")
    parser.add_argument("--output-root", type=Path, required=True, help="离线导入输出目录")
    args = parser.parse_args()

    result = import_archives(args.archive_root, args.recovered_root, args.output_root)
    print(
        f"[完成] 归档 {result['archive_count']:,} 个，"
        f"文档 {result['document_count']:,} 个，"
        f"未导出记录 {result['records_without_exported_document']:,} 条"
    )
    print(f"  输出目录: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
