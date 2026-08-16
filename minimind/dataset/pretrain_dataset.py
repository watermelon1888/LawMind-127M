"""从预先打包的 token shards 读取预训练 sequence。"""

from __future__ import annotations

import bisect
import hashlib
import json
import operator
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Sequence, Sized

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


HASH_LINE_RE = re.compile(r"([0-9a-fA-F]{64})  (.+)")
UINT32_SEQUENCE_LIMIT = 1 << 32


class PretrainDataError(RuntimeError):
    """表示预训练 shard 元数据或文件无法安全使用。"""


@dataclass(frozen=True)
class ShardDescriptor:
    """保存一个 token shard 的读取位置和 sequence 计数。"""

    path: Path
    token_count: int
    sequence_count: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_dict(value: object, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PretrainDataError(f"{location} 必须是 JSON object")
    return value


def _require_positive_int(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PretrainDataError(f"{location} 必须是正整数")
    return value


class PretrainShardCatalog:
    """验证 shard manifest，并创建指定 split 的 Dataset 视图。"""

    def __init__(self, manifest_path: Path | str) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        manifest = self._load_verified_manifest()

        packing = _require_dict(manifest.get("packing"), "manifest packing")
        self.sequence_length = _require_positive_int(
            packing.get("sequence_length"),
            "manifest packing.sequence_length",
        )

        output = _require_dict(manifest.get("output"), "manifest output")
        root = output.get("root")
        if not isinstance(root, str) or not root:
            raise PretrainDataError("manifest output.root 必须是非空字符串")
        self.output_root = Path(root).resolve()

        sources = _require_dict(manifest.get("sources"), "manifest sources")
        if not sources:
            raise PretrainDataError("manifest sources 不能为空")
        self._source_names = tuple(sources)
        self._streams = self._load_streams(sources)

    def _load_verified_manifest(self) -> dict[str, Any]:
        hash_path = self.manifest_path.with_suffix(".sha256")
        if not self.manifest_path.is_file():
            raise PretrainDataError(f"shard manifest 不存在: {self.manifest_path}")
        if not hash_path.is_file():
            raise PretrainDataError(f"shard manifest 哈希文件不存在: {hash_path}")
        try:
            lines = [
                line
                for line in hash_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            actual_digest = _sha256_file(self.manifest_path)
        except (OSError, UnicodeDecodeError) as error:
            raise PretrainDataError("无法读取 shard manifest 或其哈希文件") from error
        if len(lines) != 1:
            raise PretrainDataError("shard manifest 哈希文件必须只有一条有效记录")
        match = HASH_LINE_RE.fullmatch(lines[0])
        if match is None:
            raise PretrainDataError("shard manifest 哈希文件格式无效")
        expected_digest, filename = match.groups()
        if filename != self.manifest_path.name:
            raise PretrainDataError("shard manifest 哈希文件记录了错误的文件名")
        if actual_digest.lower() != expected_digest.lower():
            raise PretrainDataError("shard manifest SHA-256 不匹配")
        self.manifest_sha256 = actual_digest.lower()
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PretrainDataError("shard manifest 不是有效 UTF-8 JSON") from error
        manifest = _require_dict(manifest, "shard manifest")
        if manifest.get("schema_version") != "1.0":
            raise PretrainDataError("不支持的 shard manifest schema_version")
        return manifest

    @property
    def source_names(self) -> tuple[str, ...]:
        """按 shard manifest 顺序返回全部数据来源。"""
        return self._source_names

    def _load_streams(
        self,
        sources: dict[str, Any],
    ) -> dict[str, dict[str, tuple[ShardDescriptor, ...]]]:
        streams: dict[str, dict[str, tuple[ShardDescriptor, ...]]] = {}
        seen_paths: set[Path] = set()
        for source, source_value in sources.items():
            if not isinstance(source, str) or not source:
                raise PretrainDataError("manifest source 名称必须是非空字符串")
            source_report = _require_dict(source_value, f"manifest sources.{source}")
            streams[source] = {}
            for split, split_value in source_report.items():
                split_report = _require_dict(
                    split_value,
                    f"manifest sources.{source}.{split}",
                )
                shard_values = split_report.get("shards")
                if not isinstance(shard_values, list):
                    raise PretrainDataError(
                        f"manifest sources.{source}.{split}.shards 必须是 JSON array"
                    )
                descriptors = tuple(
                    self._load_shard(source, split, index, value, seen_paths)
                    for index, value in enumerate(shard_values)
                )
                expected_sequences = _require_positive_int(
                    split_report.get("sequence_count"),
                    f"manifest sources.{source}.{split}.sequence_count",
                )
                if sum(shard.sequence_count for shard in descriptors) != expected_sequences:
                    raise PretrainDataError(f"{source}/{split} shard sequence 计数不闭合")
                streams[source][split] = descriptors
        return streams

    def _load_shard(
        self,
        source: str,
        split: str,
        index: int,
        value: object,
        seen_paths: set[Path],
    ) -> ShardDescriptor:
        location = f"manifest sources.{source}.{split}.shards[{index}]"
        shard = _require_dict(value, location)
        path_text = shard.get("path")
        if not isinstance(path_text, str):
            raise PretrainDataError(f"{location}.path 必须是字符串")
        relative_path = PurePosixPath(path_text)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative_path.suffix != ".npy"
        ):
            raise PretrainDataError(f"{location}.path 必须是安全的相对 .npy 路径")
        path = (self.output_root / Path(*relative_path.parts)).resolve()
        if path in seen_paths:
            raise PretrainDataError(f"shard manifest 重复列出文件: {path}")
        seen_paths.add(path)

        expected_bytes = _require_positive_int(shard.get("bytes"), f"{location}.bytes")
        token_count = _require_positive_int(
            shard.get("token_count"),
            f"{location}.token_count",
        )
        sequence_count = _require_positive_int(
            shard.get("sequence_count"),
            f"{location}.sequence_count",
        )
        if token_count % self.sequence_length:
            raise PretrainDataError(f"{location}.token_count 不能切成完整 sequence")
        if sequence_count != token_count // self.sequence_length:
            raise PretrainDataError(f"{location}.sequence_count 与 token_count 不一致")
        if not path.is_file():
            raise PretrainDataError(f"token shard 不存在: {path}")
        try:
            actual_bytes = path.stat().st_size
        except OSError as error:
            raise PretrainDataError(f"无法读取 token shard 大小: {path}") from error
        if actual_bytes != expected_bytes:
            raise PretrainDataError(f"token shard 文件大小不匹配: {path}")
        self._validate_shard_header(path, token_count)
        return ShardDescriptor(
            path=path,
            token_count=token_count,
            sequence_count=sequence_count,
        )

    @staticmethod
    def _validate_shard_header(path: Path, token_count: int) -> None:
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as error:
            raise PretrainDataError(f"无法加载 token shard header: {path}") from error
        try:
            if array.ndim != 1:
                raise PretrainDataError(f"token shard 必须是一维数组: {path}")
            if array.dtype != np.dtype(np.uint16):
                raise PretrainDataError(f"token shard dtype 必须为 uint16: {path}")
            if array.size != token_count:
                raise PretrainDataError(f"token shard header 与 manifest token 数不一致: {path}")
        finally:
            mmap_handle = getattr(array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()

    def create_dataset(
        self,
        split: str,
        sources: Sequence[str] | None = None,
    ) -> "PretrainDataset":
        """创建一个按 manifest 来源顺序连接的 split 视图。"""
        selected = set(self._source_names if sources is None else sources)
        if not selected:
            raise PretrainDataError("Dataset 至少需要一个数据来源")
        unknown = selected.difference(self._source_names)
        if unknown:
            raise PretrainDataError(f"manifest 中不存在数据来源: {sorted(unknown)[0]}")

        shards: list[ShardDescriptor] = []
        for source in self._source_names:
            if source not in selected:
                continue
            source_streams = self._streams[source]
            if split not in source_streams:
                raise PretrainDataError(f"数据来源 {source} 不包含 split: {split}")
            shards.extend(source_streams[split])
        return PretrainDataset(shards, self.sequence_length)


class PretrainDataset(Dataset):
    """通过累计 sequence 前缀和按需 mmap 连续 token shards。"""

    def __init__(
        self,
        shards: Sequence[ShardDescriptor],
        sequence_length: int,
    ) -> None:
        self.shards = tuple(shards)
        self.sequence_length = sequence_length
        total = 0
        cumulative_sequences: list[int] = []
        for shard in self.shards:
            total += shard.sequence_count
            cumulative_sequences.append(total)
        self._cumulative_sequences = tuple(cumulative_sequences)
        self._sequence_count = total
        self._mmap_cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return self._sequence_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            normalized_index = operator.index(index)
        except TypeError as error:
            raise TypeError("sequence 索引必须是整数") from error
        if normalized_index < 0 or normalized_index >= self._sequence_count:
            raise IndexError("sequence 索引超出 Dataset 范围")

        shard_index = bisect.bisect_right(
            self._cumulative_sequences,
            normalized_index,
        )
        previous_total = (
            0 if shard_index == 0 else self._cumulative_sequences[shard_index - 1]
        )
        local_sequence_index = normalized_index - previous_total
        shard = self.shards[shard_index]
        array = self._mmap_cache.get(shard.path)
        if array is None:
            array = np.load(shard.path, mmap_mode="r", allow_pickle=False)
            self._mmap_cache[shard.path] = array

        token_offset = local_sequence_index * self.sequence_length
        tokens = np.array(
            array[token_offset : token_offset + self.sequence_length],
            dtype=np.int64,
            copy=True,
        )
        input_ids = torch.from_numpy(tokens)
        return input_ids, input_ids.clone()

    def close(self) -> None:
        """关闭当前进程内已打开的 mmap。"""
        for array in self._mmap_cache.values():
            mmap_handle = getattr(array, "_mmap", None)
            if mmap_handle is not None:
                mmap_handle.close()
        self._mmap_cache.clear()

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_mmap_cache"] = {}
        return state

    def __del__(self) -> None:
        self.close()


class DeterministicPretrainSampler(Sampler[int]):
    """使用 seed 和 epoch 重建无放回的全局 sequence 排列。"""

    def __init__(
        self,
        data_source: Sized,
        seed: int = 42,
        epoch: int = 0,
        start_position: int = 0,
    ) -> None:
        sequence_count = len(data_source)
        if sequence_count >= UINT32_SEQUENCE_LIMIT:
            raise ValueError("sequence 总数必须小于 2^32")
        values = (
            ("seed", seed),
            ("epoch", epoch),
            ("start_position", start_position),
        )
        for name, value in values:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} 必须是非负整数")
        if start_position > sequence_count:
            raise ValueError("start_position 不能超过 sequence 总数")

        self.sequence_count = sequence_count
        self.seed = seed
        self.epoch = epoch
        self.start_position = start_position

    def __iter__(self) -> Iterator[int]:
        indices = np.arange(self.sequence_count, dtype=np.uint32)
        generator = np.random.Generator(np.random.PCG64(self.seed + self.epoch))
        generator.shuffle(indices)
        return (int(index) for index in indices[self.start_position :])

    def __len__(self) -> int:
        return self.sequence_count - self.start_position
