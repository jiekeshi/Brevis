#!/usr/bin/env python3
"""Memory-bounded empirical information references for BF16 checkpoints.

This tool scans physical BF16 words directly from safetensors through NumPy
memory maps.  It reports empirical zero-order entropies and an explicitly
idealized ``8 + H(exponent)`` reference.  These are iid references for the
observed checkpoint, not universal lower bounds for compressors that exploit
order, repetition, tensor structure, or side information.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import dataclasses
import json
import math
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from analyze_brevis_attribution import classify_tensor_role


SCHEMA_VERSION = 1
DEFAULT_CHUNK_MIB = 64
ROLE_ORDER = ("embedding", "attention", "mlp", "norm", "other")
MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_DIMENSIONS = 1024
MAX_U64 = (1 << 64) - 1
MAX_JOBS = 64
MAX_CHUNK_MIB = 1024
METHOD_FIELDS = (
    "method",
    "input_kind",
    "input_value",
    "source_file_bytes_basis",
    "whole_output_bytes",
    "whole_output_bytes_source",
    "whole_output_bytes_is_exact_integer_input",
    "compression_ratio_source_over_output",
    "amortized_whole_output_bpw_per_analyzed_bf16_weight",
    "signed_amortized_bpw_difference_from_empirical_bf16_symbol_h0",
    "signed_amortized_bpw_difference_from_idealized_8_plus_iid_exponent_h0",
    "signed_amortized_bpw_difference_from_finite_adjacent_exponent_reference",
)

# Physical widths from the safetensors dtype vocabulary.  Only BF16 payloads
# are scanned, but validating skipped tensor lengths prevents accidentally
# analyzing a malformed or partial shard.
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


class InformationAnalysisError(RuntimeError):
    """The requested checkpoint cannot be analyzed safely."""


@dataclasses.dataclass(frozen=True)
class TensorLocation:
    name: str
    dtype: str
    shape: tuple[int, ...]
    numel: int
    nbytes: int
    shard: Path
    absolute_data_offset: int


@dataclasses.dataclass(frozen=True)
class Discovery:
    shards: tuple[Path, ...]
    provenance: dict[str, Any]
    index_assignments: tuple[tuple[str, Path], ...] | None = None


@dataclasses.dataclass
class TensorScan:
    location: TensorLocation
    symbol_counts: np.ndarray
    exponent_transition_counts: np.ndarray | None
    first_exponent: int | None

    @property
    def exponent_counts(self) -> np.ndarray:
        # BF16 word index = sign * 32768 + exponent * 128 + mantissa.
        return self.symbol_counts.reshape(2, 256, 128).sum(
            axis=(0, 2),
            dtype=np.uint64,
        )


@dataclasses.dataclass
class Aggregate:
    tensor_count: int = 0
    nonempty_tensor_count: int = 0
    parameters: int = 0
    symbol_counts: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(65_536, dtype=np.uint64)
    )
    exponent_transition_counts: np.ndarray | None = None
    first_exponent_counts: np.ndarray = dataclasses.field(
        default_factory=lambda: np.zeros(256, dtype=np.uint64)
    )

    def add(self, scan: TensorScan) -> None:
        self.tensor_count += 1
        self.parameters += scan.location.numel
        self.symbol_counts += scan.symbol_counts
        if scan.location.numel:
            self.nonempty_tensor_count += 1
        if scan.first_exponent is not None:
            self.first_exponent_counts[scan.first_exponent] += 1
        if scan.exponent_transition_counts is not None:
            if self.exponent_transition_counts is None:
                self.exponent_transition_counts = np.zeros(
                    (256, 256),
                    dtype=np.uint64,
                )
            self.exponent_transition_counts += scan.exponent_transition_counts


@dataclasses.dataclass(frozen=True)
class MethodInput:
    method: str
    kind: str
    value: float | int


def _checked_numel(shape: Sequence[int], label: str) -> int:
    if any(value < 0 or value > MAX_U64 for value in shape):
        raise InformationAnalysisError(f"{label}: invalid tensor dimension")
    if any(value == 0 for value in shape):
        return 0
    result = 1
    for value in shape:
        result *= value
        if result > MAX_U64:
            raise InformationAnalysisError(f"{label}: tensor shape exceeds u64")
    return result


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise InformationAnalysisError(f"{label}: cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise InformationAnalysisError(f"{label}: JSON root is not an object: {path}")
    return value


def _safe_manifest_path(root: Path, value: str, manifest: Path) -> Path:
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise InformationAnalysisError(
            f"{manifest}: weight path escapes checkpoint: {value!r}"
        ) from exc
    return candidate


def _discover_from_manifest(manifest_path: Path) -> Discovery:
    manifest = _load_json_object(manifest_path, "manifest")
    weights = manifest.get("weights")
    if not isinstance(weights, list) or not weights:
        raise InformationAnalysisError(f"{manifest_path}: weights is not a list")
    root = manifest_path.parent
    shards: list[Path] = []
    declared: list[dict[str, Any]] = []
    for item in weights:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise InformationAnalysisError(f"{manifest_path}: invalid weight entry")
        path = _safe_manifest_path(root, item["path"], manifest_path)
        if path.suffix != ".safetensors":
            continue
        if not path.is_file():
            raise InformationAnalysisError(f"{manifest_path}: missing weight {path}")
        size = path.stat().st_size
        declared_size = item.get("size")
        if declared_size is not None and (
            not isinstance(declared_size, int)
            or isinstance(declared_size, bool)
            or declared_size != size
        ):
            raise InformationAnalysisError(
                f"{manifest_path}: declared size mismatch for {path.name}"
            )
        shards.append(path.resolve())
        declared.append(
            {
                "path": item["path"],
                "size": size,
                "manifest_declared_sha256": item.get("sha256"),
            }
        )
    shards = list(dict.fromkeys(shards))
    if not shards:
        raise InformationAnalysisError(
            f"{manifest_path}: no safetensors weights in manifest"
        )
    return Discovery(
        tuple(shards),
        {
            "mode": "download-manifest",
            "manifest_path": str(manifest_path.resolve()),
            "name": manifest.get("name"),
            "repo_id": manifest.get("repo_id"),
            "revision": manifest.get("revision"),
            "manifest_reported_sha256_verified": manifest.get("sha256_verified"),
            "sha256_recomputed_by_this_analyzer": False,
            "declared_weights": declared,
        },
    )


def _discover_from_index(index_path: Path) -> Discovery:
    index = _load_json_object(index_path, "safetensors index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise InformationAnalysisError(f"{index_path}: invalid weight_map")
    root = index_path.parent
    shards: list[Path] = []
    assignments: list[tuple[str, Path]] = []
    for tensor, filename in weight_map.items():
        if (
            not isinstance(tensor, str)
            or not tensor
            or tensor == "__metadata__"
            or not isinstance(filename, str)
        ):
            raise InformationAnalysisError(f"{index_path}: invalid weight_map entry")
        path = _safe_manifest_path(root, filename, index_path)
        if path.suffix != ".safetensors" or not path.is_file():
            raise InformationAnalysisError(
                f"{index_path}: missing safetensors shard {path}"
            )
        resolved = path.resolve()
        shards.append(resolved)
        assignments.append((tensor, resolved))
    return Discovery(
        tuple(sorted(set(shards))),
        {
            "mode": "safetensors-index",
            "index_path": str(index_path.resolve()),
            "indexed_tensor_count": len(weight_map),
        },
        tuple(sorted(assignments, key=lambda item: item[0])),
    )


def discover_checkpoint(path: Path) -> Discovery:
    path = path.expanduser()
    if path.is_file():
        if path.name == "download-manifest.json":
            return _discover_from_manifest(path)
        if path.name.endswith(".safetensors.index.json"):
            return _discover_from_index(path)
        if path.suffix == ".safetensors":
            return Discovery(
                (path.resolve(),),
                {"mode": "explicit-safetensors"},
            )
        raise InformationAnalysisError(
            f"{path}: expected safetensors, index JSON, or download-manifest.json"
        )
    if not path.is_dir():
        raise InformationAnalysisError(f"{path}: checkpoint path does not exist")

    manifest = path / "download-manifest.json"
    if manifest.is_file():
        return _discover_from_manifest(manifest)
    index = path / "model.safetensors.index.json"
    if index.is_file():
        return _discover_from_index(index)
    single = path / "model.safetensors"
    if single.is_file():
        return Discovery(
            (single.resolve(),),
            {"mode": "single-model-safetensors"},
        )
    shards = tuple(sorted(item.resolve() for item in path.rglob("*.safetensors")))
    if not shards:
        raise InformationAnalysisError(f"{path}: no safetensors files found")
    return Discovery(
        shards,
        {
            "mode": "recursive-fallback",
            "warning": (
                "No manifest or root model index was found; every recursively "
                "discovered safetensors file was included."
            ),
        },
    )


def _read_header(path: Path) -> tuple[int, dict[str, Any]]:
    file_bytes = path.stat().st_size
    with path.open("rb") as handle:
        raw_length = handle.read(8)
        if len(raw_length) != 8:
            raise InformationAnalysisError(f"{path}: truncated safetensors length")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > MAX_HEADER_BYTES or header_length > file_bytes - 8:
            raise InformationAnalysisError(
                f"{path}: invalid safetensors header length {header_length}"
            )
        raw_header = handle.read(header_length)
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise InformationAnalysisError(
            f"{path}: invalid safetensors JSON header"
        ) from exc
    if not isinstance(header, dict):
        raise InformationAnalysisError(f"{path}: header root is not an object")
    return 8 + header_length, header


def load_tensor_locations(
    discovery: Discovery,
) -> tuple[list[TensorLocation], dict[str, Any]]:
    locations: list[TensorLocation] = []
    dtype_tensors: Counter[str] = Counter()
    dtype_parameters: Counter[str] = Counter()
    dtype_payload_bytes: Counter[str] = Counter()
    indexed_names_by_shard: dict[Path, set[str]] | None = None
    if discovery.index_assignments is not None:
        indexed_names_by_shard = {shard: set() for shard in discovery.shards}
        for tensor_name, shard in discovery.index_assignments:
            indexed_names_by_shard[shard].add(tensor_name)
    for shard in discovery.shards:
        data_base, header = _read_header(shard)
        metadata: list[tuple[int, int, int, TensorLocation]] = []
        actual_tensor_names: set[str] = set()
        source_order = 0
        for name, descriptor in header.items():
            if name == "__metadata__":
                if (
                    not isinstance(descriptor, dict)
                    or any(
                        not isinstance(key, str) or not isinstance(value, str)
                        for key, value in descriptor.items()
                    )
                ):
                    raise InformationAnalysisError(
                        f"{shard}: __metadata__ must map strings to strings"
                    )
                continue
            if not isinstance(name, str) or not isinstance(descriptor, dict):
                raise InformationAnalysisError(f"{shard}: invalid tensor entry")
            dtype = descriptor.get("dtype")
            shape = descriptor.get("shape")
            offsets = descriptor.get("data_offsets")
            if not isinstance(dtype, str) or dtype not in DTYPE_BYTES:
                raise InformationAnalysisError(
                    f"{shard}/{name}: unsupported dtype {dtype!r}"
                )
            if (
                not isinstance(shape, list)
                or len(shape) > MAX_DIMENSIONS
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                    or value > MAX_U64
                    for value in shape
                )
            ):
                raise InformationAnalysisError(f"{shard}/{name}: invalid shape")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(
                    not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                    or value > MAX_U64
                    for value in offsets
                )
                or offsets[1] < offsets[0]
            ):
                raise InformationAnalysisError(
                    f"{shard}/{name}: invalid data_offsets"
                )
            numel = _checked_numel(shape, f"{shard}/{name}")
            expected_bytes = numel * DTYPE_BYTES[dtype]
            if expected_bytes > MAX_U64 or offsets[1] - offsets[0] != expected_bytes:
                raise InformationAnalysisError(
                    f"{shard}/{name}: shape and byte length disagree"
                )
            location = TensorLocation(
                name=name,
                dtype=dtype,
                shape=tuple(shape),
                numel=numel,
                nbytes=expected_bytes,
                shard=shard,
                absolute_data_offset=data_base + offsets[0],
            )
            actual_tensor_names.add(name)
            metadata.append((offsets[0], offsets[1], source_order, location))
            dtype_tensors[dtype] += 1
            dtype_parameters[dtype] += numel
            dtype_payload_bytes[dtype] += expected_bytes
            source_order += 1

        metadata.sort(key=lambda item: (item[0], item[1], item[2]))
        cursor = 0
        for begin, end, _order, location in metadata:
            if begin != cursor:
                raise InformationAnalysisError(
                    f"{shard}: non-contiguous data before {location.name}"
                )
            cursor = end
            locations.append(location)
        if data_base + cursor != shard.stat().st_size:
            raise InformationAnalysisError(
                f"{shard}: header data length does not match file size"
            )
        if indexed_names_by_shard is not None:
            expected_tensor_names = indexed_names_by_shard[shard]
            if actual_tensor_names != expected_tensor_names:
                missing = sorted(expected_tensor_names - actual_tensor_names)
                extra = sorted(actual_tensor_names - expected_tensor_names)
                raise InformationAnalysisError(
                    f"{shard}: safetensors index mismatch; "
                    f"missing={missing!r}, extra={extra!r}"
                )
    locations.sort(key=lambda item: (str(item.shard), item.absolute_data_offset))
    bf16 = [item for item in locations if item.dtype == "BF16"]
    if not bf16:
        raise InformationAnalysisError("checkpoint contains no BF16 tensors")
    if not any(item.numel for item in bf16):
        raise InformationAnalysisError(
            "checkpoint contains no non-empty BF16 parameters"
        )
    inventory = {
        "all_tensor_count": len(locations),
        "bf16_tensor_count": len(bf16),
        "bf16_parameters": sum(item.numel for item in bf16),
        "bf16_payload_bytes": sum(item.nbytes for item in bf16),
        "dtype_tensor_counts": dict(sorted(dtype_tensors.items())),
        "dtype_parameter_counts": dict(sorted(dtype_parameters.items())),
        "dtype_payload_bytes": dict(sorted(dtype_payload_bytes.items())),
        "skipped_non_bf16_tensor_count": len(locations) - len(bf16),
        "skipped_non_bf16_payload_bytes": sum(
            item.nbytes for item in locations if item.dtype != "BF16"
        ),
    }
    return bf16, inventory


def scan_tensor(
    location: TensorLocation,
    *,
    chunk_elements: int,
    adjacent_exponent_order1: bool,
) -> TensorScan:
    if location.dtype != "BF16":
        raise InformationAnalysisError(f"{location.name}: expected BF16")
    if chunk_elements < 1:
        raise ValueError("chunk_elements must be positive")
    symbols = np.zeros(65_536, dtype=np.uint64)
    transitions = (
        np.zeros((256, 256), dtype=np.uint64)
        if adjacent_exponent_order1
        else None
    )
    if location.numel == 0:
        return TensorScan(location, symbols, transitions, None)
    mapped = np.memmap(
        location.shard,
        mode="r",
        dtype="<u2",
        offset=location.absolute_data_offset,
        shape=(location.numel,),
    )
    previous_exponent: int | None = None
    first_exponent: int | None = None
    try:
        for start in range(0, location.numel, chunk_elements):
            raw = np.asarray(mapped[start : start + chunk_elements])
            symbols += np.bincount(raw, minlength=65_536).astype(
                np.uint64,
                copy=False,
            )
            if transitions is None or raw.size == 0:
                continue
            exponents = np.bitwise_and(
                np.right_shift(raw, 7),
                0xFF,
            ).astype(np.uint8, copy=False)
            if first_exponent is None:
                first_exponent = int(exponents[0])
            if previous_exponent is not None:
                transitions[previous_exponent, int(exponents[0])] += 1
            if exponents.size > 1:
                pair_codes = (
                    np.left_shift(
                        exponents[:-1].astype(np.uint16, copy=False),
                        8,
                    )
                    | exponents[1:].astype(np.uint16, copy=False)
                )
                transitions += np.bincount(
                    pair_codes,
                    minlength=65_536,
                ).reshape(256, 256).astype(np.uint64, copy=False)
            previous_exponent = int(exponents[-1])
    finally:
        del mapped
    if int(symbols.sum()) != location.numel:
        raise InformationAnalysisError(
            f"{location.shard}/{location.name}: histogram count mismatch"
        )
    if transitions is not None and int(transitions.sum()) != max(
        0, location.numel - 1
    ):
        raise InformationAnalysisError(
            f"{location.shard}/{location.name}: transition count mismatch"
        )
    return TensorScan(location, symbols, transitions, first_exponent)


def empirical_entropy(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total == 0:
        return 0.0
    nonzero = counts[counts != 0].astype(np.float64)
    entropy = math.log2(total) - float(
        np.sum(nonzero * np.log2(nonzero), dtype=np.float64)
    ) / total
    return max(0.0, entropy)


def empirical_conditional_entropy(transitions: np.ndarray) -> float | None:
    total = int(transitions.sum())
    if total == 0:
        return None
    rows = transitions.sum(axis=1, dtype=np.uint64)
    nonzero_rows = rows[rows != 0].astype(np.float64)
    nonzero_cells = transitions[transitions != 0].astype(np.float64)
    entropy = (
        float(
            np.sum(nonzero_rows * np.log2(nonzero_rows), dtype=np.float64)
        )
        - float(
            np.sum(nonzero_cells * np.log2(nonzero_cells), dtype=np.float64)
        )
    ) / total
    return max(0.0, entropy)


def _scan_all(
    locations: Sequence[TensorLocation],
    *,
    jobs: int,
    chunk_elements: int,
    adjacent_exponent_order1: bool,
    progress: bool,
) -> Iterable[TensorScan]:
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if jobs == 1:
        for index, location in enumerate(locations, 1):
            result = scan_tensor(
                location,
                chunk_elements=chunk_elements,
                adjacent_exponent_order1=adjacent_exponent_order1,
            )
            if progress:
                print(
                    f"[bf16-information] {index}/{len(locations)} "
                    f"{location.name} ({location.numel:,} weights)",
                    file=sys.stderr,
                    flush=True,
                )
            yield result
        return

    # Keep at most ``jobs`` completed histogram arrays live.  Submitting every
    # tensor at once can otherwise retain one 512 KiB result per future.
    iterator = iter(locations)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        pending: dict[concurrent.futures.Future[TensorScan], TensorLocation] = {}

        def submit_one() -> bool:
            try:
                location = next(iterator)
            except StopIteration:
                return False
            future = executor.submit(
                scan_tensor,
                location,
                chunk_elements=chunk_elements,
                adjacent_exponent_order1=adjacent_exponent_order1,
            )
            pending[future] = location
            return True

        for _ in range(min(jobs, len(locations))):
            submit_one()
        while pending:
            done, _ = concurrent.futures.wait(
                pending,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                location = pending.pop(future)
                result = future.result()
                completed += 1
                if progress:
                    print(
                        f"[bf16-information] {completed}/{len(locations)} "
                        f"{location.name} ({location.numel:,} weights)",
                        file=sys.stderr,
                        flush=True,
                    )
                yield result
                submit_one()


def _aggregate_row(
    name: str,
    aggregate: Aggregate,
    total_parameters: int,
) -> dict[str, Any]:
    exponent_counts = aggregate.symbol_counts.reshape(2, 256, 128).sum(
        axis=(0, 2),
        dtype=np.uint64,
    )
    symbol_h0 = empirical_entropy(aggregate.symbol_counts)
    exponent_h0 = empirical_entropy(exponent_counts)
    conditional = (
        empirical_conditional_entropy(aggregate.exponent_transition_counts)
        if aggregate.exponent_transition_counts is not None
        else None
    )
    transition_count = (
        int(aggregate.exponent_transition_counts.sum())
        if aggregate.exponent_transition_counts is not None
        else None
    )
    if aggregate.exponent_transition_counts is not None:
        first_count = int(aggregate.first_exponent_counts.sum())
        if first_count != aggregate.nonempty_tensor_count:
            raise InformationAnalysisError(
                f"{name}: first-exponent count does not match non-empty tensors"
            )
        if transition_count + first_count != aggregate.parameters:
            raise InformationAnalysisError(
                f"{name}: exponent sequence accounting mismatch"
            )
    first_h0 = (
        empirical_entropy(aggregate.first_exponent_counts)
        if aggregate.exponent_transition_counts is not None
        and aggregate.nonempty_tensor_count
        else None
    )
    finite_exponent_bpw = None
    if (
        transition_count is not None
        and first_h0 is not None
        and aggregate.parameters
    ):
        finite_exponent_bpw = (
            aggregate.nonempty_tensor_count * first_h0
            + transition_count * (conditional if conditional is not None else 0.0)
        ) / aggregate.parameters
    return {
        "group": name,
        "tensor_count": aggregate.tensor_count,
        "nonempty_tensor_count": aggregate.nonempty_tensor_count,
        "parameters": aggregate.parameters,
        "parameter_share_percent": (
            100.0 * aggregate.parameters / total_parameters
            if total_parameters
            else 0.0
        ),
        "bf16_payload_bytes": 2 * aggregate.parameters,
        "nominal_bits_per_weight": 16.0,
        "distinct_bf16_symbols": int(np.count_nonzero(aggregate.symbol_counts)),
        "empirical_bf16_symbol_h0_bits_per_weight": symbol_h0,
        "distinct_exponents": int(np.count_nonzero(exponent_counts)),
        "empirical_exponent_h0_bits_per_exponent": exponent_h0,
        "idealized_raw_sign_mantissa_plus_iid_exponent_bpw": 8.0 + exponent_h0,
        "adjacent_exponent_transition_count": transition_count,
        "empirical_first_exponent_h0_bits_per_first_exponent": first_h0,
        "empirical_adjacent_exponent_h1_conditional_bits_per_exponent": conditional,
        "finite_sequence_adjacent_exponent_reference_bits_per_weight": (
            finite_exponent_bpw
        ),
        "idealized_raw_sign_mantissa_plus_finite_adjacent_exponent_reference_bpw": (
            8.0 + finite_exponent_bpw
            if finite_exponent_bpw is not None
            else None
        ),
        "exponent_histogram": [int(value) for value in exponent_counts],
    }


def _tensor_row(scan: TensorScan) -> dict[str, Any]:
    exponent_counts = scan.exponent_counts
    conditional = (
        empirical_conditional_entropy(scan.exponent_transition_counts)
        if scan.exponent_transition_counts is not None
        else None
    )
    transition_count = (
        int(scan.exponent_transition_counts.sum())
        if scan.exponent_transition_counts is not None
        else None
    )
    finite_exponent_bpw = (
        transition_count * (conditional if conditional is not None else 0.0)
        / scan.location.numel
        if transition_count is not None and scan.location.numel
        else None
    )
    return {
        "shard": str(scan.location.shard.resolve()),
        "tensor_name": scan.location.name,
        "role": classify_tensor_role(scan.location.name),
        "shape": "x".join(str(value) for value in scan.location.shape),
        "parameters": scan.location.numel,
        "bf16_payload_bytes": scan.location.nbytes,
        "distinct_bf16_symbols": int(np.count_nonzero(scan.symbol_counts)),
        "empirical_bf16_symbol_h0_bits_per_weight": empirical_entropy(
            scan.symbol_counts
        ),
        "distinct_exponents": int(np.count_nonzero(exponent_counts)),
        "empirical_exponent_h0_bits_per_exponent": empirical_entropy(
            exponent_counts
        ),
        "empirical_adjacent_exponent_h1_conditional_bits_per_exponent": conditional,
        "finite_sequence_adjacent_exponent_reference_bits_per_weight": (
            finite_exponent_bpw
        ),
    }


def parse_method_inputs(
    ratios: Sequence[str],
    sizes: Sequence[str],
) -> list[MethodInput]:
    result: list[MethodInput] = []
    seen: set[str] = set()
    for kind, values in (("ratio", ratios), ("size_bytes", sizes)):
        for raw in values:
            if "=" not in raw:
                raise InformationAnalysisError(
                    f"{kind} method input must be METHOD=VALUE: {raw!r}"
                )
            method, raw_value = raw.split("=", 1)
            method = method.strip()
            if not method or method in seen:
                raise InformationAnalysisError(
                    f"empty or duplicate method name: {method!r}"
                )
            try:
                if kind == "ratio":
                    value: float | int = float(raw_value)
                    if not math.isfinite(value) or value <= 0:
                        raise ValueError
                else:
                    value = int(raw_value)
                    if value <= 0 or value > MAX_U64:
                        raise ValueError
            except ValueError as exc:
                raise InformationAnalysisError(
                    f"invalid {kind} for {method}: {raw_value!r}"
                ) from exc
            seen.add(method)
            result.append(MethodInput(method, kind, value))
    return result


def method_rows(
    inputs: Sequence[MethodInput],
    *,
    source_file_bytes: int,
    parameters: int,
    overall: dict[str, Any],
) -> list[dict[str, Any]]:
    if source_file_bytes <= 0 or parameters <= 0:
        raise InformationAnalysisError(
            "method comparison requires positive source bytes and BF16 parameters"
        )
    rows: list[dict[str, Any]] = []
    symbol_h0 = overall["empirical_bf16_symbol_h0_bits_per_weight"]
    exponent_reference = overall[
        "idealized_raw_sign_mantissa_plus_iid_exponent_bpw"
    ]
    adjacent_reference = overall[
        "idealized_raw_sign_mantissa_plus_finite_adjacent_exponent_reference_bpw"
    ]
    for item in inputs:
        try:
            if item.kind == "ratio":
                ratio = float(item.value)
                whole_output_bytes: float | int = source_file_bytes / ratio
                output_source = "ratio-implied"
                exact_size = False
            elif item.kind == "size_bytes":
                whole_output_bytes = int(item.value)
                ratio = source_file_bytes / whole_output_bytes
                output_source = "explicit-size"
                exact_size = True
            else:
                raise InformationAnalysisError(
                    f"unsupported method input kind: {item.kind!r}"
                )
            amortized_bpw = 8.0 * (whole_output_bytes / parameters)
        except (OverflowError, ZeroDivisionError) as exc:
            raise InformationAnalysisError(
                f"derived comparison value is out of range for {item.method}"
            ) from exc
        derived_values = (float(whole_output_bytes), ratio, amortized_bpw)
        if any(not math.isfinite(value) or value <= 0 for value in derived_values):
            raise InformationAnalysisError(
                f"derived comparison value is out of range for {item.method}"
            )
        rows.append(
            {
                "method": item.method,
                "input_kind": item.kind,
                "input_value": item.value,
                "source_file_bytes_basis": source_file_bytes,
                "whole_output_bytes": whole_output_bytes,
                "whole_output_bytes_source": output_source,
                "whole_output_bytes_is_exact_integer_input": exact_size,
                "compression_ratio_source_over_output": ratio,
                "amortized_whole_output_bpw_per_analyzed_bf16_weight": (
                    amortized_bpw
                ),
                "signed_amortized_bpw_difference_from_empirical_bf16_symbol_h0": (
                    amortized_bpw - symbol_h0
                ),
                "signed_amortized_bpw_difference_from_idealized_8_plus_iid_exponent_h0": (
                    amortized_bpw - exponent_reference
                ),
                "signed_amortized_bpw_difference_from_finite_adjacent_exponent_reference": (
                    amortized_bpw - adjacent_reference
                    if adjacent_reference is not None
                    else None
                ),
            }
        )
    return rows


def _shard_fingerprints(
    shards: Sequence[Path],
) -> dict[Path, tuple[int, int, int, int]]:
    fingerprints: dict[Path, tuple[int, int, int, int]] = {}
    for path in shards:
        try:
            stat = path.stat()
        except OSError as exc:
            raise InformationAnalysisError(
                f"cannot stat checkpoint shard {path}: {exc}"
            ) from exc
        fingerprints[path] = (
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_dev,
            stat.st_ino,
        )
    return fingerprints


def analyze(
    checkpoint: Path,
    *,
    jobs: int = 1,
    chunk_mib: int = DEFAULT_CHUNK_MIB,
    adjacent_exponent_order1: bool = False,
    method_inputs: Sequence[MethodInput] = (),
    progress: bool = False,
) -> dict[str, Any]:
    if chunk_mib < 1 or chunk_mib > MAX_CHUNK_MIB:
        raise InformationAnalysisError(
            f"chunk_mib must be in [1, {MAX_CHUNK_MIB}]"
        )
    if jobs < 1 or jobs > MAX_JOBS:
        raise InformationAnalysisError(f"jobs must be in [1, {MAX_JOBS}]")
    discovery = discover_checkpoint(checkpoint)
    shard_stats_before = _shard_fingerprints(discovery.shards)
    locations, inventory = load_tensor_locations(discovery)
    chunk_elements = max(1, chunk_mib * 1024**2 // 2)

    overall = Aggregate(
        exponent_transition_counts=(
            np.zeros((256, 256), dtype=np.uint64)
            if adjacent_exponent_order1
            else None
        )
    )
    roles = {
        role: Aggregate(
            exponent_transition_counts=(
                np.zeros((256, 256), dtype=np.uint64)
                if adjacent_exponent_order1
                else None
            )
        )
        for role in ROLE_ORDER
    }
    tensor_rows: list[dict[str, Any]] = []
    for scan in _scan_all(
        locations,
        jobs=jobs,
        chunk_elements=chunk_elements,
        adjacent_exponent_order1=adjacent_exponent_order1,
        progress=progress,
    ):
        overall.add(scan)
        roles[classify_tensor_role(scan.location.name)].add(scan)
        tensor_rows.append(_tensor_row(scan))

    if overall.parameters != inventory["bf16_parameters"]:
        raise InformationAnalysisError("aggregate BF16 parameter count drift")
    groups = [_aggregate_row("overall", overall, overall.parameters)]
    groups.extend(
        _aggregate_row(role, roles[role], overall.parameters)
        for role in ROLE_ORDER
        if roles[role].parameters
    )
    tensor_rows.sort(key=lambda row: (row["shard"], row["tensor_name"]))
    shard_stats_after = _shard_fingerprints(discovery.shards)
    if shard_stats_after != shard_stats_before:
        raise InformationAnalysisError(
            "checkpoint shard metadata changed during analysis; discard the scan"
        )
    source_file_bytes = sum(
        fingerprint[0] for fingerprint in shard_stats_before.values()
    )
    methods = method_rows(
        method_inputs,
        source_file_bytes=source_file_bytes,
        parameters=overall.parameters,
        overall=groups[0],
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis": {
            "dtype": "BF16",
            "nominal_bits_per_weight": 16,
            "chunk_mib_per_worker": chunk_mib,
            "jobs": jobs,
            "adjacent_exponent_order1_enabled": adjacent_exponent_order1,
            "adjacency_scope": (
                "physical row-major order within each tensor; no transitions "
                "are introduced across tensor boundaries"
                if adjacent_exponent_order1
                else None
            ),
            "reference_scope": (
                "Empirical iid H0 and the optional finite within-tensor adjacent "
                "exponent reference are descriptive references for this checkpoint. "
                "They are not absolute lower bounds for arbitrary structure-aware "
                "coding."
            ),
            "memory_scope": (
                "chunk_mib controls input words per worker, not total RSS; NumPy "
                "histogram and adjacent-transition temporaries make peak memory a "
                "multiple of chunk_mib times jobs"
            ),
        },
        "checkpoint": {
            "input": str(checkpoint.resolve()),
            "source_file_bytes": source_file_bytes,
            "shards": [
                {
                    "path": str(path.resolve()),
                    "size": shard_stats_before[path][0],
                    "mtime_ns": shard_stats_before[path][1],
                }
                for path in discovery.shards
            ],
            "discovery": discovery.provenance,
            "inventory": inventory,
        },
        "groups": groups,
        "tensors": tensor_rows,
        "methods": methods,
    }


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.12g}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    empty_fieldnames: Sequence[str] = (),
) -> None:
    if not rows and not empty_fieldnames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(rows[0]) if rows else list(empty_fieldnames),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _fmt_float(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def build_markdown(report: dict[str, Any]) -> str:
    inventory = report["checkpoint"]["inventory"]
    groups = report["groups"]
    methods = report["methods"]
    lines = [
        "# BF16 empirical information references",
        "",
        "> **Interpretation warning:** empirical H0 is an iid coding reference for "
        "the observed checkpoint. It is **not** an absolute information-theoretic "
        "lower bound for arbitrary compressors that exploit order, repetition, "
        "tensor roles, program structure, or side information. Signed gaps may "
        "therefore be negative and must not be called impossible savings.",
        "",
        f"- Source files: {len(report['checkpoint']['shards'])}; bytes: "
        f"{_fmt_int(report['checkpoint']['source_file_bytes'])}.",
        f"- BF16 tensors: {_fmt_int(inventory['bf16_tensor_count'])}; parameters: "
        f"{_fmt_int(inventory['bf16_parameters'])}; payload bytes: "
        f"{_fmt_int(inventory['bf16_payload_bytes'])}.",
        f"- Skipped non-BF16 tensors: "
        f"{_fmt_int(inventory['skipped_non_bf16_tensor_count'])}; bytes: "
        f"{_fmt_int(inventory['skipped_non_bf16_payload_bytes'])}.",
        "",
        "The `8 + Hexp` column assumes all sign and mantissa bits cost exactly "
        "eight raw bits per weight, exponent symbols use an ideal iid entropy "
        "code, and model/framing/alignment costs are zero.",
        "When adjacent statistics are enabled, the finite-sequence exponent "
        "reference is `[K H0(E_first) + P H(E_i|E_{i-1})] / N`, where K is "
        "the number of non-empty tensors, P is the number of within-tensor "
        "transitions, and N is the BF16 parameter count.",
        "",
        "## Overall and role groups",
        "",
        "| group | tensors (non-empty) | parameters | share | nominal BPW | BF16-symbol H0 | exponent H0 | idealized 8+Hexp | first exponent H0 | adjacent exponent H1 | idealized 8+finite-adj |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in groups:
        lines.append(
            f"| {row['group']} | {_fmt_int(row['tensor_count'])} "
            f"({_fmt_int(row['nonempty_tensor_count'])}) | "
            f"{_fmt_int(row['parameters'])} | "
            f"{row['parameter_share_percent']:.3f}% | 16.0000 | "
            f"{_fmt_float(row['empirical_bf16_symbol_h0_bits_per_weight'])} | "
            f"{_fmt_float(row['empirical_exponent_h0_bits_per_exponent'])} | "
            f"{_fmt_float(row['idealized_raw_sign_mantissa_plus_iid_exponent_bpw'])} | "
            f"{_fmt_float(row['empirical_first_exponent_h0_bits_per_first_exponent'])} | "
            f"{_fmt_float(row['empirical_adjacent_exponent_h1_conditional_bits_per_exponent'])} | "
            f"{_fmt_float(row['idealized_raw_sign_mantissa_plus_finite_adjacent_exponent_reference_bpw'])} |"
        )
    if methods:
        lines.extend(
            [
                "",
                "## Supplied compressor results",
                "",
                "Amortized whole-output BPW divides the supplied whole compressed "
                "checkpoint size (including framing and any non-BF16 content) by "
                "the number of analyzed BF16 parameters. A size input is exact; "
                "a ratio input only implies a possibly fractional output size from "
                "the discovered whole safetensors source-file bytes. Prefer exact "
                "`--method-size` inputs for publication.",
                "",
                "| method | input | output-size basis | ratio | whole output bytes | amortized BPW | BPW−symbol H0 | BPW−(8+Hexp) | BPW−finite-adj |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in methods:
            lines.append(
                f"| {row['method']} | {row['input_kind']}="
                f"{row['input_value']} | "
                f"{row['whole_output_bytes_source']} | "
                f"{row['compression_ratio_source_over_output']:.6f}× | "
                f"{row['whole_output_bytes']:,.3f} | "
                f"{row['amortized_whole_output_bpw_per_analyzed_bf16_weight']:.4f} | "
                f"{row['signed_amortized_bpw_difference_from_empirical_bf16_symbol_h0']:.4f} | "
                f"{row['signed_amortized_bpw_difference_from_idealized_8_plus_iid_exponent_h0']:.4f} | "
                f"{_fmt_float(row['signed_amortized_bpw_difference_from_finite_adjacent_exponent_reference'])} |"
            )
    lines.extend(
        [
            "",
            "## Method notes",
            "",
            "- BF16-symbol H0 uses the empirical distribution of complete physical "
            "16-bit words and ignores their order.",
            "- Exponent H0 uses the empirical 8-bit BF16 exponent distribution and "
            "also ignores order.",
            "- Optional adjacent H1 is `H(E_i | E_{i-1})` in flattened physical "
            "order within each tensor. Tensor boundaries are never joined. The "
            "reported finite reference separately accounts for each tensor's first "
            "exponent before normalizing by all BF16 parameters.",
            "- No finite-block coding, codebook, metadata, alignment, random-access, "
            "or decoder cost is included in an entropy reference.",
            "- Compressor differences are signed descriptive gaps to these references, "
            "not proofs of remaining universally achievable compression.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "information-analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    group_rows = []
    for row in report["groups"]:
        group_rows.append(
            {
                key: value
                for key, value in row.items()
                if key != "exponent_histogram"
            }
        )
    write_csv(output_dir / "groups.csv", group_rows)
    write_csv(output_dir / "tensors.csv", report["tensors"])
    write_csv(
        output_dir / "methods.csv",
        report["methods"],
        empty_fieldnames=METHOD_FIELDS,
    )
    (output_dir / "report.md").write_text(
        build_markdown(report),
        encoding="utf-8",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute memory-bounded empirical BF16 symbol/exponent entropy "
            "references from safetensors."
        )
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help=(
            "checkpoint directory, download-manifest.json, safetensors index, "
            "or one .safetensors file"
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--chunk-mib", type=int, default=DEFAULT_CHUNK_MIB)
    parser.add_argument(
        "--adjacent-exponent-order1",
        action="store_true",
        help="also count within-tensor adjacent exponent transitions",
    )
    parser.add_argument(
        "--method-ratio",
        action="append",
        default=[],
        metavar="METHOD=RATIO",
        help="repeatable source/output ratio for amortized-BPW comparison",
    )
    parser.add_argument(
        "--method-size",
        action="append",
        default=[],
        metavar="METHOD=BYTES",
        help="repeatable exact whole-output bytes for amortized-BPW comparison",
    )
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        methods = parse_method_inputs(args.method_ratio, args.method_size)
        report = analyze(
            args.checkpoint,
            jobs=args.jobs,
            chunk_mib=args.chunk_mib,
            adjacent_exponent_order1=args.adjacent_exponent_order1,
            method_inputs=methods,
            progress=args.progress,
        )
        write_outputs(report, args.output_dir)
    except (InformationAnalysisError, OSError, ValueError) as exc:
        raise SystemExit(f"BF16 information analysis failed: {exc}") from exc
    overall = report["groups"][0]
    print(
        f"analyzed {overall['parameters']} BF16 weights: "
        f"H0(symbol)={overall['empirical_bf16_symbol_h0_bits_per_weight']:.6f} "
        f"bits/weight, H0(exponent)="
        f"{overall['empirical_exponent_h0_bits_per_exponent']:.6f}; "
        f"wrote {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
