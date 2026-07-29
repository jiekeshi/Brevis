#!/usr/bin/env python3
"""Fast, read-only size estimator for the pinned DFloat11 LLM format.

The official DFloat11 converter encodes one Huffman stream for the embedding,
one for each decoder layer (seven linear weights concatenated), and one for the
LM head.  Computing the compressed payload only requires the BF16 exponent
histogram of each group; the order of the values is irrelevant except for a
single four-byte ``output_positions`` boundary entry per group.

This estimator reads safetensors through NumPy memory maps.  It never loads a
Transformers model, allocates a GPU tensor, or writes to the source checkpoint.
It intentionally targets the format at DFloat11 commit
457733886ce6ebc6d8dda1621fad1ffa2661e028 (version 0.5.0).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import heapq
import json
import math
import os
import struct
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


PINNED_DFLOAT11_COMMIT = "457733886ce6ebc6d8dda1621fad1ffa2661e028"
DFLOAT11_FORMAT_VERSION = "0.5.0"
PINNED_DAHUFFMAN_VERSION = "0.4.2"
PINNED_NUMPY_VERSION = "2.5.1"
PINNED_SAFETENSORS_VERSION = "0.8.0"
BYTES_PER_THREAD = 8
THREADS_PER_BLOCK = 512
BITS_PER_THREAD = 8 * BYTES_PER_THREAD
BYTES_PER_BLOCK = THREADS_PER_BLOCK * BYTES_PER_THREAD
BITS_PER_BLOCK = 8 * BYTES_PER_BLOCK
GAP_BITS_PER_THREAD = 5
DEFAULT_ANCILLARY_UNCERTAINTY_BYTES = 64 * 1024

LLM_LINEAR_PATHS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

# Safetensors serializes tensor payloads by decreasing dtype alignment/width,
# then by tensor name.  These widths cover the current safetensors dtype set.
DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


class EstimationError(RuntimeError):
    """The checkpoint cannot be represented by the frozen estimator."""


@dataclasses.dataclass(frozen=True)
class TensorLocation:
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int
    path: Path
    absolute_data_offset: int

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


@dataclasses.dataclass(frozen=True)
class TensorDescriptor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    nbytes: int


@dataclasses.dataclass(frozen=True)
class CompressionGroup:
    module_name: str
    output_filename: str
    target_names: tuple[str, ...]
    residual_names: tuple[str, ...]


class _EndOfFileSymbol:
    """Match dahuffman 0.4.2's tie ordering for its private EOF symbol."""

    def __repr__(self) -> str:
        return "_EOF"

    def __lt__(self, other: object) -> bool:
        return True

    def __gt__(self, other: object) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return other.__class__ == self.__class__

    def __hash__(self) -> int:
        return hash(self.__class__)


_EOF = _EndOfFileSymbol()


def _compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_safetensors_header(path: Path) -> tuple[int, dict[str, Any]]:
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise EstimationError(f"truncated safetensors length: {path}")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > path.stat().st_size - 8:
            raise EstimationError(f"invalid safetensors header length: {path}")
        raw_header = handle.read(header_length)
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EstimationError(f"invalid safetensors JSON header: {path}") from exc
    if not isinstance(header, dict):
        raise EstimationError(f"safetensors header is not an object: {path}")
    return 8 + header_length, header


def load_tensor_catalog(
    model_dir: Path,
) -> tuple[dict[str, TensorLocation], tuple[Path, ...]]:
    """Read tensor locations and the benchmark source shard set."""

    index_path = model_dir / "model.safetensors.index.json"
    single_path = model_dir / "model.safetensors"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            weight_map = index["weight_map"]
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            raise EstimationError(f"invalid safetensors index: {index_path}") from exc
        if not isinstance(weight_map, dict) or not weight_map:
            raise EstimationError(f"empty safetensors weight_map: {index_path}")
        mapped_files = {
            str(name): model_dir / str(filename)
            for name, filename in weight_map.items()
        }
        source_files = tuple(sorted(set(mapped_files.values())))
    elif single_path.is_file():
        mapped_files = {}
        source_files = (single_path,)
    else:
        raise EstimationError(
            f"missing model.safetensors or model.safetensors.index.json in {model_dir}"
        )

    catalog: dict[str, TensorLocation] = {}
    for shard in source_files:
        if not shard.is_file():
            raise EstimationError(f"missing safetensors shard: {shard}")
        data_base, header = _read_safetensors_header(shard)
        for name, raw_descriptor in header.items():
            if name == "__metadata__":
                continue
            if mapped_files and mapped_files.get(name) != shard:
                continue
            if not isinstance(raw_descriptor, dict):
                raise EstimationError(f"invalid tensor descriptor {name!r} in {shard}")
            try:
                dtype = str(raw_descriptor["dtype"])
                shape = tuple(int(value) for value in raw_descriptor["shape"])
                start, end = (
                    int(value) for value in raw_descriptor["data_offsets"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise EstimationError(
                    f"invalid tensor descriptor {name!r} in {shard}"
                ) from exc
            if min(shape, default=0) < 0 or start < 0 or end < start:
                raise EstimationError(
                    f"invalid tensor shape or offsets for {name!r} in {shard}"
                )
            if data_base + end > shard.stat().st_size:
                raise EstimationError(f"tensor {name!r} exceeds {shard}")
            if name in catalog:
                raise EstimationError(f"duplicate tensor {name!r}")
            location = TensorLocation(
                name=name,
                dtype=dtype,
                shape=shape,
                nbytes=end - start,
                path=shard,
                absolute_data_offset=data_base + start,
            )
            item_bytes = DTYPE_BYTES.get(dtype)
            if item_bytes is not None and location.numel * item_bytes != location.nbytes:
                raise EstimationError(
                    f"shape/byte mismatch for {name!r}: "
                    f"{location.numel} * {item_bytes} != {location.nbytes}"
                )
            catalog[name] = location

    if mapped_files:
        missing = sorted(set(mapped_files) - set(catalog))
        if missing:
            preview = ", ".join(missing[:3])
            raise EstimationError(
                f"{len(missing)} indexed tensors are missing from shard headers: {preview}"
            )
    if not catalog:
        raise EstimationError(f"no tensors found in {model_dir}")
    return catalog, source_files


def build_llm_groups(
    catalog: Mapping[str, TensorLocation],
    config: Mapping[str, Any],
) -> tuple[tuple[CompressionGroup, ...], tuple[str, ...]]:
    """Apply the exact adapter pattern used by specialized_baselines.py."""

    try:
        num_layers = int(config["num_hidden_layers"])
    except (KeyError, TypeError, ValueError) as exc:
        raise EstimationError("config.json has no valid num_hidden_layers") from exc
    if num_layers < 1:
        raise EstimationError("num_hidden_layers must be positive")

    definitions: list[tuple[str, tuple[str, ...]]] = [
        ("model.embed_tokens", ("model.embed_tokens.weight",)),
    ]
    for layer in range(num_layers):
        module = f"model.layers.{layer}"
        definitions.append(
            (
                module,
                tuple(f"{module}.{suffix}.weight" for suffix in LLM_LINEAR_PATHS),
            )
        )
    definitions.append(("lm_head", ("lm_head.weight",)))

    groups: list[CompressionGroup] = []
    claimed: set[str] = set()
    for module_name, targets in definitions:
        missing = [name for name in targets if name not in catalog]
        if missing:
            raise EstimationError(
                f"DFloat11 target tensor(s) missing for {module_name}: {missing}"
            )
        non_bf16 = [name for name in targets if catalog[name].dtype != "BF16"]
        if non_bf16:
            raise EstimationError(
                f"DFloat11 requires BF16 target tensors: {non_bf16}"
            )
        prefix = f"{module_name}."
        members = {name for name in catalog if name.startswith(prefix)}
        residuals = tuple(sorted(members - set(targets)))
        claimed.update(members)
        groups.append(
            CompressionGroup(
                module_name=module_name,
                output_filename=f"{module_name.replace('.', '_')}.safetensors",
                target_names=targets,
                residual_names=residuals,
            )
        )
    root_residuals = tuple(sorted(set(catalog) - claimed))
    return tuple(groups), root_residuals


def scan_bf16_exponent_histogram(
    tensor: TensorLocation,
    chunk_elements: int,
) -> np.ndarray:
    """Count BF16 exponents with bounded memory and no tensor materialization."""

    if tensor.dtype != "BF16":
        raise EstimationError(f"{tensor.name} is {tensor.dtype}, not BF16")
    if chunk_elements < 1:
        raise ValueError("chunk_elements must be positive")
    mapped = np.memmap(
        tensor.path,
        mode="r",
        dtype="<u2",
        offset=tensor.absolute_data_offset,
        shape=(tensor.numel,),
    )
    counts = np.zeros(256, dtype=np.uint64)
    try:
        for start in range(0, tensor.numel, chunk_elements):
            raw = np.asarray(mapped[start : start + chunk_elements])
            exponents = np.bitwise_and(np.right_shift(raw, 7), 0xFF)
            counts += np.bincount(exponents, minlength=256).astype(
                np.uint64,
                copy=False,
            )
    finally:
        del mapped
    if int(counts.sum()) != tensor.numel:
        raise EstimationError(f"histogram count mismatch for {tensor.name}")
    return counts


def scan_group_histograms(
    groups: Sequence[CompressionGroup],
    catalog: Mapping[str, TensorLocation],
    *,
    jobs: int,
    chunk_elements: int,
    progress: bool,
) -> dict[str, np.ndarray]:
    if jobs < 1:
        raise ValueError("jobs must be positive")
    target_to_group = {
        target: group.module_name
        for group in groups
        for target in group.target_names
    }
    totals = {
        group.module_name: np.zeros(256, dtype=np.uint64) for group in groups
    }

    def finish(name: str, counts: np.ndarray) -> None:
        totals[target_to_group[name]] += counts
        if progress:
            print(
                f"[dfloat11-estimate] scanned {name} "
                f"({catalog[name].numel:,} BF16 values)",
                file=sys.stderr,
                flush=True,
            )

    if jobs == 1:
        for name in target_to_group:
            finish(
                name,
                scan_bf16_exponent_histogram(catalog[name], chunk_elements),
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(
                    scan_bf16_exponent_histogram,
                    catalog[name],
                    chunk_elements,
                ): name
                for name in target_to_group
            }
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                finish(name, future.result())
    return totals


def checkpoint_fingerprint(source_files: Sequence[Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in source_files
    ]


def load_histogram_cache(
    path: Path,
    fingerprint: list[dict[str, Any]],
    groups: Sequence[CompressionGroup],
) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("schema_version") != 1
            or value.get("dfloat11_commit") != PINNED_DFLOAT11_COMMIT
            or value.get("source_files") != fingerprint
        ):
            return None
        raw_histograms = value["histograms"]
        expected_targets = {
            group.module_name: list(group.target_names) for group in groups
        }
        if value.get("target_names") != expected_targets:
            return None
        result: dict[str, np.ndarray] = {}
        for group in groups:
            raw = raw_histograms[group.module_name]
            if not isinstance(raw, list) or len(raw) != 256:
                return None
            counts = np.asarray(raw, dtype=np.uint64)
            result[group.module_name] = counts
        return result
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def save_histogram_cache(
    path: Path,
    fingerprint: list[dict[str, Any]],
    groups: Sequence[CompressionGroup],
    histograms: Mapping[str, np.ndarray],
) -> None:
    value = {
        "schema_version": 1,
        "dfloat11_commit": PINNED_DFLOAT11_COMMIT,
        "source_files": fingerprint,
        "target_names": {
            group.module_name: list(group.target_names) for group in groups
        },
        "histograms": {
            group.module_name: [
                int(value) for value in histograms[group.module_name]
            ]
            for group in groups
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _huffman_table(counter: Mapping[int, int]) -> dict[object, tuple[int, int]]:
    """Reproduce dahuffman 0.4.2 ``HuffmanCodec.from_frequencies``."""

    if not counter:
        raise EstimationError("cannot build a Huffman table from no symbols")
    heap: list[tuple[int, list[tuple[object, tuple[int, int]]]]] = [
        (int(frequency), [(int(symbol), (0, 0))])
        for symbol, frequency in counter.items()
    ]
    heap.append((1, [(_EOF, (0, 0))]))
    heapq.heapify(heap)
    while len(heap) > 1:
        first = heapq.heappop(heap)
        second = heapq.heappop(heap)
        merged = (
            first[0] + second[0],
            [(symbol, (bits + 1, value)) for symbol, (bits, value) in first[1]]
            + [
                (symbol, (bits + 1, (1 << bits) + value))
                for symbol, (bits, value) in second[1]
            ],
        )
        heapq.heappush(heap, merged)
    return dict(heapq.heappop(heap)[1])


def official_32bit_huffman(
    histogram: Sequence[int] | np.ndarray,
) -> tuple[dict[object, tuple[int, int]], tuple[int, ...]]:
    """Reproduce DFloat11's get_32bit_codec, including NumPy tie behavior."""

    if len(histogram) != 256:
        raise ValueError("BF16 exponent histogram must contain 256 bins")
    counter = {
        exponent: int(histogram[exponent])
        for exponent in range(256)
        if int(histogram[exponent]) > 0
    }
    if not counter:
        raise EstimationError("BF16 exponent histogram is empty")
    table = _huffman_table(counter)
    frequencies = np.asarray(list(counter.values()), dtype=np.int64)
    symbols = np.asarray(list(counter), dtype=np.int64)
    adjusted_counter = counter
    min_k = 2
    while max(bits for bits, _ in table.values()) > 32:
        if min_k >= len(frequencies):
            raise EstimationError("DFloat11 could not construct a <=32-bit code")
        minimum_indices = np.argpartition(frequencies, min_k)[:min_k]
        minimum_symbols = symbols[minimum_indices]
        adjusted_counter = counter.copy()
        for symbol in minimum_symbols:
            adjusted_counter[int(symbol)] = 1
        table = _huffman_table(adjusted_counter)
        min_k += 1
    adjusted = tuple(
        symbol
        for symbol in counter
        if adjusted_counter[symbol] != counter[symbol]
    )
    return table, adjusted


def official_lut_rows(table: Mapping[object, tuple[int, int]]) -> int:
    """Return the first dimension produced by DFloat11 ``get_luts``."""

    prefixes = {""}
    for symbol, (bits, value) in table.items():
        if not isinstance(symbol, int):
            continue
        binary = bin(value)[2:].rjust(bits, "0")
        prefix_bits = ((bits - 1) // 8) * 8
        prefixes.add(binary[:prefix_bits])
    # get_luts appends one 256-byte code-length row.
    return len(prefixes) + 1


def _descriptor(location: TensorLocation) -> TensorDescriptor:
    return TensorDescriptor(
        name=location.name,
        dtype=location.dtype,
        shape=location.shape,
        nbytes=location.nbytes,
    )


def safetensors_file_layout(
    descriptors: Sequence[TensorDescriptor],
    *,
    metadata: Mapping[str, str] | None = None,
) -> dict[str, int]:
    """Compute the byte-exact layout used by safetensors for these descriptors."""

    unknown = sorted({item.dtype for item in descriptors} - set(DTYPE_BYTES))
    if unknown:
        raise EstimationError(f"unsupported safetensors dtype(s): {unknown}")
    ordered = sorted(
        descriptors,
        key=lambda item: (-DTYPE_BYTES[item.dtype], item.name),
    )
    header: dict[str, Any] = {}
    if metadata is not None:
        header["__metadata__"] = dict(metadata)
    offset = 0
    for item in ordered:
        end = offset + item.nbytes
        header[item.name] = {
            "dtype": item.dtype,
            "shape": list(item.shape),
            "data_offsets": [offset, end],
        }
        offset = end
    raw_header_bytes = len(_compact_json_bytes(header))
    padded_header_bytes = (raw_header_bytes + 7) // 8 * 8
    return {
        "payload_bytes": offset,
        "raw_header_bytes": raw_header_bytes,
        "padded_header_bytes": padded_header_bytes,
        "prefix_bytes": 8,
        "file_bytes": 8 + padded_header_bytes + offset,
    }


def estimate_group(
    group: CompressionGroup,
    histogram: np.ndarray,
    catalog: Mapping[str, TensorLocation],
) -> dict[str, Any]:
    numel = sum(catalog[name].numel for name in group.target_names)
    if int(histogram.sum()) != numel:
        raise EstimationError(
            f"histogram size for {group.module_name} is {int(histogram.sum())}, "
            f"expected {numel}"
        )
    table, adjusted_symbols = official_32bit_huffman(histogram)
    code_lengths = np.zeros(256, dtype=np.uint8)
    for symbol, (bits, _) in table.items():
        if isinstance(symbol, int):
            code_lengths[symbol] = bits
    encoded_bits = sum(
        int(histogram[exponent]) * int(code_lengths[exponent])
        for exponent in range(256)
    )
    encoded_bytes = (encoded_bits + 7) // 8
    blocks = (encoded_bytes + BYTES_PER_BLOCK - 1) // BYTES_PER_BLOCK
    gaps_bytes = (
        blocks * THREADS_PER_BLOCK * GAP_BITS_PER_THREAD + 7
    ) // 8
    # An exponent code can straddle the last 32,768-bit block boundary.
    # Histogram-only estimation cannot know whether another element starts in
    # that last block.  This changes one uint32 entry (four bytes) at most.
    output_entries_lower = max(2, blocks)
    output_entries_estimate = blocks + 1
    output_bytes_lower = 4 * output_entries_lower
    output_bytes_estimate = 4 * output_entries_estimate
    split_bytes = max(0, len(group.target_names) - 1) * 8
    lut_rows = official_lut_rows(table)
    lut_bytes = lut_rows * 256
    residuals = [_descriptor(catalog[name]) for name in group.residual_names]

    def descriptors(output_position_bytes: int) -> list[TensorDescriptor]:
        base = group.module_name
        return [
            *residuals,
            TensorDescriptor(
                f"{base}.luts",
                "U8",
                (lut_rows, 256),
                lut_bytes,
            ),
            TensorDescriptor(
                f"{base}.encoded_exponent",
                "U8",
                (encoded_bytes,),
                encoded_bytes,
            ),
            TensorDescriptor(
                f"{base}.sign_mantissa",
                "U8",
                (numel,),
                numel,
            ),
            TensorDescriptor(
                f"{base}.output_positions",
                "U8",
                (output_position_bytes,),
                output_position_bytes,
            ),
            TensorDescriptor(
                f"{base}.gaps",
                "U8",
                (gaps_bytes,),
                gaps_bytes,
            ),
            TensorDescriptor(
                f"{base}.split_positions",
                "I64",
                (max(0, len(group.target_names) - 1),),
                split_bytes,
            ),
        ]

    lower_layout = safetensors_file_layout(descriptors(output_bytes_lower))
    estimate_layout = safetensors_file_layout(
        descriptors(output_bytes_estimate)
    )
    residual_bytes = sum(item.nbytes for item in residuals)
    return {
        "module": group.module_name,
        "output_filename": group.output_filename,
        "target_tensors": list(group.target_names),
        "target_numel": numel,
        "target_source_bytes": 2 * numel,
        "nonzero_exponents": int(np.count_nonzero(histogram)),
        "max_code_length_bits": max(bits for bits, _ in table.values()),
        "code_lengths_by_exponent": [
            int(code_lengths[index]) for index in range(256)
        ],
        "frequency_adjusted_exponents": list(adjusted_symbols),
        "components": {
            "encoded_exponent_bits": encoded_bits,
            "encoded_exponent_bytes": encoded_bytes,
            "sign_mantissa_bytes": numel,
            "output_positions_bytes": output_bytes_estimate,
            "output_positions_bytes_lower": output_bytes_lower,
            "output_positions_bytes_upper": output_bytes_estimate,
            "gaps_bytes": gaps_bytes,
            "split_positions_bytes": split_bytes,
            "luts_bytes": lut_bytes,
            "residual_tensor_bytes": residual_bytes,
            "safetensors_prefix_and_header_bytes": (
                estimate_layout["file_bytes"]
                - estimate_layout["payload_bytes"]
            ),
        },
        "encoded_blocks": blocks,
        "lut_rows": lut_rows,
        "estimated_file_bytes": estimate_layout["file_bytes"],
        "file_bytes_lower": lower_layout["file_bytes"],
        "file_bytes_upper": estimate_layout["file_bytes"],
    }


def dfloat11_config_payload() -> dict[str, Any]:
    return {
        "version": DFLOAT11_FORMAT_VERSION,
        "threads_per_block": [THREADS_PER_BLOCK],
        "bytes_per_thread": BYTES_PER_THREAD,
        "pattern_dict": {
            r"model\.embed_tokens": [],
            r"model\.layers\.\d+": list(LLM_LINEAR_PATHS),
            "lm_head": [],
        },
    }


def estimate_ancillary_files(
    model_dir: Path,
    uncertainty_bytes: int,
) -> dict[str, Any]:
    """Estimate config/generation JSON written by Transformers save_pretrained."""

    if uncertainty_bytes < 0:
        raise ValueError("ancillary uncertainty must be non-negative")
    config_path = model_dir / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EstimationError(f"invalid model config: {config_path}") from exc
    if not isinstance(config, dict):
        raise EstimationError(f"model config is not an object: {config_path}")
    estimated_config = dict(config)
    estimated_config["dfloat11_config"] = dfloat11_config_payload()
    config_bytes = len(
        (json.dumps(estimated_config, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    )
    generation_path = model_dir / "generation_config.json"
    generation_bytes = (
        generation_path.stat().st_size if generation_path.is_file() else 0
    )
    estimate = config_bytes + generation_bytes
    return {
        "estimated_bytes": estimate,
        "lower_bytes": max(0, estimate - uncertainty_bytes),
        "upper_bytes": estimate + uncertainty_bytes,
        "uncertainty_bytes": uncertainty_bytes,
        "files": {
            "config.json": config_bytes,
            "generation_config.json": generation_bytes,
        },
        "note": (
            "Transformers may normalize defaults/version fields while saving JSON; "
            "the interval is an explicit non-structural allowance."
        ),
    }


def verify_upstream_commit(upstream: Path) -> None:
    try:
        actual = subprocess.run(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EstimationError(f"cannot inspect DFloat11 checkout: {upstream}") from exc
    if actual != PINNED_DFLOAT11_COMMIT:
        raise EstimationError(
            f"DFloat11 commit mismatch: {actual}; expected "
            f"{PINNED_DFLOAT11_COMMIT}"
        )


def estimate_model(
    model_dir: Path,
    histograms: Mapping[str, np.ndarray],
    *,
    ancillary_uncertainty_bytes: int = DEFAULT_ANCILLARY_UNCERTAINTY_BYTES,
    actual_output_bytes: int | None = None,
) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EstimationError(f"invalid model config: {config_path}") from exc
    catalog, source_files = load_tensor_catalog(model_dir)
    groups, root_names = build_llm_groups(catalog, config)
    missing_histograms = [
        group.module_name
        for group in groups
        if group.module_name not in histograms
    ]
    if missing_histograms:
        raise EstimationError(f"missing histograms for {missing_histograms}")
    group_results = [
        estimate_group(group, histograms[group.module_name], catalog)
        for group in groups
    ]

    root_descriptors = [_descriptor(catalog[name]) for name in root_names]
    root_layout = safetensors_file_layout(
        root_descriptors,
        metadata={"format": "pt"},
    )
    structural_estimate = (
        sum(item["estimated_file_bytes"] for item in group_results)
        + root_layout["file_bytes"]
    )
    structural_lower = (
        sum(item["file_bytes_lower"] for item in group_results)
        + root_layout["file_bytes"]
    )
    structural_upper = (
        sum(item["file_bytes_upper"] for item in group_results)
        + root_layout["file_bytes"]
    )
    component_names = (
        "encoded_exponent_bytes",
        "sign_mantissa_bytes",
        "output_positions_bytes",
        "output_positions_bytes_lower",
        "output_positions_bytes_upper",
        "gaps_bytes",
        "split_positions_bytes",
        "luts_bytes",
        "residual_tensor_bytes",
        "safetensors_prefix_and_header_bytes",
    )
    component_totals = {
        name: sum(item["components"][name] for item in group_results)
        for name in component_names
    }
    component_totals["residual_tensor_bytes"] += root_layout["payload_bytes"]
    component_totals["safetensors_prefix_and_header_bytes"] += (
        root_layout["file_bytes"] - root_layout["payload_bytes"]
    )
    ancillary = estimate_ancillary_files(
        model_dir,
        ancillary_uncertainty_bytes,
    )
    output_estimate = structural_estimate + ancillary["estimated_bytes"]
    output_lower = structural_lower + ancillary["lower_bytes"]
    output_upper = structural_upper + ancillary["upper_bytes"]
    source_checkpoint_bytes = sum(path.stat().st_size for path in source_files)
    source_tensor_payload_bytes = sum(item.nbytes for item in catalog.values())
    estimated_ratio = source_checkpoint_bytes / output_estimate
    result: dict[str, Any] = {
        "schema_version": 1,
        "method": "dfloat11-size-estimate",
        "dfloat11_commit": PINNED_DFLOAT11_COMMIT,
        "dfloat11_format_version": DFLOAT11_FORMAT_VERSION,
        "frozen_runtime": {
            "dahuffman": PINNED_DAHUFFMAN_VERSION,
            "numpy": PINNED_NUMPY_VERSION,
            "safetensors": PINNED_SAFETENSORS_VERSION,
        },
        "estimator_runtime": {
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "frozen_numpy_runtime_match": np.__version__ == PINNED_NUMPY_VERSION,
        "model_dir": str(model_dir.resolve()),
        "estimation_mode": "exact_group_exponent_histograms",
        "source": {
            "checkpoint_shard_bytes": source_checkpoint_bytes,
            "tensor_payload_bytes": source_tensor_payload_bytes,
            "shards": [
                {"path": str(path), "bytes": path.stat().st_size}
                for path in source_files
            ],
        },
        "groups": group_results,
        "root_model_safetensors": {
            "output_filename": "model.safetensors",
            "residual_tensors": list(root_names),
            **root_layout,
        },
        "component_totals": component_totals,
        "estimated_structural_safetensors_bytes": structural_estimate,
        "structural_safetensors_bytes_lower": structural_lower,
        "structural_safetensors_bytes_upper": structural_upper,
        "ancillary_files": ancillary,
        "estimated_output_bytes": output_estimate,
        "output_bytes_lower": output_lower,
        "output_bytes_upper": output_upper,
        "estimated_ratio": estimated_ratio,
        "ratio_lower": source_checkpoint_bytes / output_upper,
        "ratio_upper": source_checkpoint_bytes / output_lower,
        "uncertainty": {
            "statistical": (
                "none: all target BF16 exponent histograms are counted exactly"
            ),
            "structural": (
                "at most one four-byte output_positions entry per compression "
                "group; safetensors payload ordering/header padding is modeled "
                "byte-exactly for the pinned serializer"
            ),
            "runtime": (
                "The <=32-bit frequency adjustment uses NumPy argpartition tie "
                "behavior; use the frozen NumPy 2.5.1 runtime for formal values."
            ),
            "non_structural": ancillary["note"],
            "do_not_claim_measured": True,
        },
    }
    if actual_output_bytes is not None:
        if actual_output_bytes <= 0:
            raise ValueError("actual_output_bytes must be positive")
        residual = actual_output_bytes - output_estimate
        result["validation"] = {
            "actual_output_bytes": actual_output_bytes,
            "estimate_residual_bytes": residual,
            "absolute_relative_error": abs(residual) / actual_output_bytes,
            "actual_within_reported_interval": (
                output_lower <= actual_output_bytes <= output_upper
            ),
            "actual_ratio": source_checkpoint_bytes / actual_output_bytes,
            "note": (
                "This validates/calibrates the estimator on this checkpoint; "
                "the measured value is not silently transferred to another model."
            ),
        }
    return result


def _load_or_scan_histograms(
    model_dir: Path,
    *,
    cache_path: Path | None,
    jobs: int,
    chunk_mib: int,
    progress: bool,
) -> tuple[dict[str, np.ndarray], str]:
    config_path = model_dir / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EstimationError(f"invalid model config: {config_path}") from exc
    catalog, source_files = load_tensor_catalog(model_dir)
    groups, _ = build_llm_groups(catalog, config)
    fingerprint = checkpoint_fingerprint(source_files)
    if cache_path is not None:
        cached = load_histogram_cache(cache_path, fingerprint, groups)
        if cached is not None:
            return cached, "cache"
    chunk_elements = chunk_mib * 1024 * 1024 // 2
    histograms = scan_group_histograms(
        groups,
        catalog,
        jobs=jobs,
        chunk_elements=chunk_elements,
        progress=progress,
    )
    if cache_path is not None:
        save_histogram_cache(
            cache_path,
            fingerprint,
            groups,
            histograms,
        )
    return histograms, "scan"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate pinned DFloat11 output size by vectorized BF16 exponent "
            "histograms; the checkpoint is opened read-only."
        )
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument(
        "--upstream",
        type=Path,
        default=Path("/opt/DFloat11"),
        help="official checkout whose HEAD must match the frozen commit",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="parallel read/count jobs (default: min(4, CPU count))",
    )
    parser.add_argument(
        "--chunk-mib",
        type=int,
        default=64,
        help="maximum BF16 input MiB per counting chunk and job",
    )
    parser.add_argument(
        "--histogram-cache",
        type=Path,
        help="optional estimator cache outside the checkpoint",
    )
    parser.add_argument(
        "--ancillary-uncertainty-bytes",
        type=int,
        default=DEFAULT_ANCILLARY_UNCERTAINTY_BYTES,
    )
    parser.add_argument(
        "--actual-output-bytes",
        type=int,
        help="validate/calibrate against a completed converter output",
    )
    parser.add_argument("--output", type=Path, help="also write the JSON result")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.jobs < 1:
            raise EstimationError("--jobs must be positive")
        if args.chunk_mib < 1:
            raise EstimationError("--chunk-mib must be positive")
        if args.ancillary_uncertainty_bytes < 0:
            raise EstimationError(
                "--ancillary-uncertainty-bytes must be non-negative"
            )
        model_dir = args.model_dir.expanduser().resolve()
        verify_upstream_commit(args.upstream.expanduser().resolve())
        histograms, histogram_source = _load_or_scan_histograms(
            model_dir,
            cache_path=(
                args.histogram_cache.expanduser().resolve()
                if args.histogram_cache is not None
                else None
            ),
            jobs=args.jobs,
            chunk_mib=args.chunk_mib,
            progress=not args.no_progress,
        )
        result = estimate_model(
            model_dir,
            histograms,
            ancillary_uncertainty_bytes=args.ancillary_uncertainty_bytes,
            actual_output_bytes=args.actual_output_bytes,
        )
        result["histogram_source"] = histogram_source
        encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            output = args.output.expanduser().resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(encoded, encoding="utf-8")
        sys.stdout.write(encoded)
        return 0
    except (EstimationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
