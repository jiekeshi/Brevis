#!/usr/bin/env python3
"""Independently verify tensor bit patterns in specialized native outputs."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from safetensors import safe_open


DECODER_SOURCE = Path(__file__).with_name("dfloat11_cpu_decoder.c")
DFLOAT11_AUXILIARY_SUFFIXES = (
    ".luts",
    ".encoded_exponent",
    ".sign_mantissa",
    ".output_positions",
    ".gaps",
    ".split_positions",
)
ECF8_AUXILIARY_SUFFIXES = (
    ".luts",
    ".encoded",
    ".packed_other_4bits",
    ".output_positions",
    ".gaps",
    ".split_positions",
)
SUPPORTED_DFLOAT11_FORMATS = frozenset({"0.5.0"})
SUPPORTED_ECF8_FORMATS = frozenset({"0.2.0"})
DFLOAT11_REFERENCE_COMMIT = "457733886ce6ebc6d8dda1621fad1ffa2661e028"
ECF8_REFERENCE_COMMIT = "9cbf3d5cf77d6db8cf6f29df1fe6d52bc88fa01e"
DECODER_ERRORS = {
    1: "invalid decoder argument",
    2: "truncated Huffman bitstream",
    3: "invalid Huffman LUT pointer",
    4: "invalid Huffman code length",
}
FileState = tuple[int, int, int, int, int]


class ValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class TensorRef:
    path: Path
    name: str
    dtype: str
    shape: tuple[int, ...]


@dataclass
class ValidationStats:
    source_tensors: int = 0
    decoded_groups: int = 0
    decoded_tensors: int = 0
    direct_tensors: int = 0
    compared_bytes: int = 0
    ignored_output_tensors: tuple[str, ...] = ()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_state(path: Path) -> FileState:
    state = path.stat()
    return (
        state.st_dev,
        state.st_ino,
        state.st_size,
        state.st_mtime_ns,
        state.st_ctime_ns,
    )


def capture_file_states(
    paths: Iterable[Path],
) -> dict[Path, FileState]:
    try:
        return {path: file_state(path) for path in paths}
    except OSError as exc:
        raise ValidationError(f"cannot stat validation input: {exc}") from exc


def require_unchanged_file_states(
    initial_states: dict[Path, FileState] | None,
) -> None:
    if initial_states is None:
        return
    for path, initial_state in initial_states.items():
        try:
            current_state = file_state(path)
        except OSError as exc:
            raise ValidationError(
                f"cannot re-stat validation input {path}: {exc}"
            ) from exc
        if current_state != initial_state:
            raise ValidationError(f"file changed during validation: {path}")


def fingerprint_files(
    paths: Iterable[Path],
    initial_states: dict[Path, FileState] | None = None,
) -> list[dict[str, Any]]:
    fingerprints = []
    for path in paths:
        try:
            before = file_state(path)
            digest = sha256(path)
            after = file_state(path)
        except OSError as exc:
            raise ValidationError(f"cannot fingerprint {path}: {exc}") from exc
        if before != after or (
            initial_states is not None
            and before != initial_states.get(path)
        ):
            raise ValidationError(f"file changed during validation: {path}")
        fingerprints.append(
            {
                "path": str(path),
                "size": before[2],
                "sha256": digest,
            }
        )
    return fingerprints


def load_source_manifest(source: Path) -> tuple[Path | None, dict[str, Any]]:
    if not source.is_dir():
        return None, {}
    manifest_path = source / "download-manifest.json"
    if not manifest_path.is_file():
        return None, {}
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"invalid source manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise ValidationError(f"invalid source manifest: {manifest_path}")
    return manifest_path, manifest


def source_weight_paths(source: Path) -> tuple[Path, ...]:
    source = source.expanduser().resolve()
    if source.is_file():
        return (source,)
    if not source.is_dir():
        raise ValidationError(f"source does not exist: {source}")
    manifest_path, manifest = load_source_manifest(source)
    if manifest_path is not None:
        try:
            entries = manifest["weights"]
            paths = tuple(source / item["path"] for item in entries)
        except (KeyError, TypeError) as exc:
            raise ValidationError(
                f"invalid source manifest: {manifest_path}"
            ) from exc
    else:
        paths = tuple(sorted(source.glob("*.safetensors")))
    if not paths:
        raise ValidationError(f"source has no safetensors weights: {source}")
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValidationError(f"missing source weight(s): {missing}")
    return paths


def verify_source_manifest_fingerprints(
    source: Path,
    source_paths: tuple[Path, ...],
    fingerprints: list[dict[str, Any]],
    manifest_initial_states: dict[Path, FileState] | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    manifest_path, manifest = load_source_manifest(source)
    if manifest_path is None:
        return manifest, None
    try:
        entries = manifest["weights"]
    except KeyError as exc:
        raise ValidationError(
            f"invalid source manifest: {manifest_path}"
        ) from exc
    if not isinstance(entries, list) or len(entries) != len(source_paths):
        raise ValidationError(
            f"source manifest weight list changed: {manifest_path}"
        )
    for path, fingerprint, entry in zip(source_paths, fingerprints, entries):
        expected_size = entry.get("size") if isinstance(entry, dict) else None
        expected_sha256 = entry.get("sha256") if isinstance(entry, dict) else None
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
            or not isinstance(expected_sha256, str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256)
        ):
            raise ValidationError(
                f"source manifest lacks valid size/SHA256 for {path.name}"
            )
        if fingerprint["size"] != expected_size:
            raise ValidationError(
                f"{path}: size {fingerprint['size']} does not match "
                f"manifest size {expected_size}"
            )
        if fingerprint["sha256"] != expected_sha256.lower():
            raise ValidationError(
                f"{path}: SHA256 does not match download manifest"
            )
    manifest_fingerprint = fingerprint_files(
        (manifest_path,),
        manifest_initial_states,
    )[0]
    return manifest, manifest_fingerprint


def converted_weight_paths(converted: Path) -> tuple[Path, ...]:
    converted = converted.expanduser().resolve()
    if converted.is_file():
        return (converted,)
    if not converted.is_dir():
        raise ValidationError(f"converted output does not exist: {converted}")
    paths = tuple(sorted(converted.rglob("*.safetensors")))
    if not paths:
        raise ValidationError(
            f"converted output has no safetensors files: {converted}"
        )
    return paths


def tensor_index(
    paths: Iterable[Path],
    label: str,
) -> dict[str, TensorRef]:
    index: dict[str, TensorRef] = {}
    for path in paths:
        try:
            with safe_open(path, framework="pt", device="cpu") as source:
                for name in source.keys():
                    if name in index:
                        raise ValidationError(
                            f"{label} tensor {name!r} appears in both "
                            f"{index[name].path} and {path}"
                        )
                    tensor_slice = source.get_slice(name)
                    index[name] = TensorRef(
                        path,
                        name,
                        tensor_slice.get_dtype(),
                        tuple(tensor_slice.get_shape()),
                    )
        except ValidationError:
            raise
        except Exception as exc:
            raise ValidationError(
                f"cannot index {label} safetensors file {path}: {exc}"
            ) from exc
    return index


def load_tensor(ref: TensorRef) -> torch.Tensor:
    try:
        with safe_open(ref.path, framework="pt", device="cpu") as source:
            return source.get_tensor(ref.name)
    except Exception as exc:
        raise ValidationError(
            f"cannot load tensor {ref.name!r} from {ref.path}: {exc}"
        ) from exc


def compile_decoder(build_dir: Path | None = None) -> tuple[Path, str]:
    if not DECODER_SOURCE.is_file():
        raise ValidationError(f"missing decoder source: {DECODER_SOURCE}")
    compiler = shutil.which(os.environ.get("CC", "cc"))
    if compiler is None:
        raise ValidationError("a C compiler is required (set CC or install cc)")
    source_hash = sha256(DECODER_SOURCE)
    cache = (
        build_dir.expanduser().resolve()
        if build_dir is not None
        else Path(tempfile.gettempdir())
        / f"brevis-specialized-validator-{os.getuid()}"
    )
    try:
        cache.mkdir(mode=0o700, parents=True, exist_ok=True)
        cache_stat = cache.lstat()
    except OSError as exc:
        raise ValidationError(
            f"cannot prepare decoder build directory {cache}: {exc}"
        ) from exc
    if (
        stat.S_ISLNK(cache_stat.st_mode)
        or not stat.S_ISDIR(cache_stat.st_mode)
        or cache_stat.st_uid != os.getuid()
        or cache_stat.st_mode & 0o077
    ):
        raise ValidationError(
            f"decoder build directory must be owned by the current user "
            f"and mode 0700: {cache}"
        )
    try:
        invocation_dir = Path(
            tempfile.mkdtemp(prefix="build-", dir=cache)
        )
    except OSError as exc:
        raise ValidationError(
            f"cannot create private decoder build directory in {cache}: {exc}"
        ) from exc
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    library = invocation_dir / f"specialized-decoder{suffix}"
    command = [compiler, "-O3", "-std=c11", "-Wall", "-Wextra", "-Werror"]
    command.extend(
        ["-dynamiclib"]
        if sys.platform == "darwin"
        else ["-shared", "-fPIC"]
    )
    try:
        subprocess.run(
            [*command, str(DECODER_SOURCE), "-o", str(library)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        shutil.rmtree(invocation_dir, ignore_errors=True)
        output = (
            exc.stdout.strip()
            if isinstance(exc, subprocess.CalledProcessError)
            and isinstance(exc.stdout, str)
            else str(exc)
        )
        raise ValidationError(f"cannot compile independent decoder: {output}") from exc
    return library, source_hash


class DFloat11CPUDecoder:
    def __init__(self, library: Path):
        try:
            self.library = ctypes.CDLL(str(library))
        except OSError as exc:
            raise ValidationError(
                f"cannot load independent decoder {library}: {exc}"
            ) from exc
        self.decode_function = self.library.brevis_dfloat11_decode
        uint8_pointer = ctypes.POINTER(ctypes.c_uint8)
        uint16_pointer = ctypes.POINTER(ctypes.c_uint16)
        self.decode_function.argtypes = [
            uint8_pointer,
            ctypes.c_size_t,
            uint8_pointer,
            ctypes.c_size_t,
            uint8_pointer,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            uint16_pointer,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.decode_function.restype = ctypes.c_int

    def decode(
        self,
        luts: np.ndarray,
        codes: np.ndarray,
        sign_mantissa: np.ndarray,
        bit_position: int,
    ) -> tuple[np.ndarray, int]:
        arrays = (luts, codes, sign_mantissa)
        if any(array.dtype != np.uint8 or not array.flags.c_contiguous for array in arrays):
            raise ValidationError(
                "DFloat11 decoder inputs must be contiguous uint8 arrays"
            )
        output = np.empty(sign_mantissa.size, dtype=np.uint16)
        position = ctypes.c_uint64(bit_position)
        failed_element = ctypes.c_size_t(0)
        result = self.decode_function(
            luts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            luts.shape[0],
            codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            codes.size,
            sign_mantissa.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            sign_mantissa.size,
            ctypes.byref(position),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.byref(failed_element),
        )
        if result:
            reason = DECODER_ERRORS.get(result, f"decoder error {result}")
            raise ValidationError(
                f"{reason} at decoded element {failed_element.value}"
            )
        return output, position.value


class ECF8CPUDecoder:
    def __init__(self, library: Path):
        try:
            self.library = ctypes.CDLL(str(library))
        except OSError as exc:
            raise ValidationError(
                f"cannot load independent decoder {library}: {exc}"
            ) from exc
        self.decode_function = self.library.brevis_ecf8_decode
        uint8_pointer = ctypes.POINTER(ctypes.c_uint8)
        self.decode_function.argtypes = [
            uint8_pointer,
            ctypes.c_size_t,
            uint8_pointer,
            ctypes.c_size_t,
            uint8_pointer,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
            uint8_pointer,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.decode_function.restype = ctypes.c_int

    def decode(
        self,
        luts: np.ndarray,
        codes: np.ndarray,
        packed_other_4bits: np.ndarray,
        start_element: int,
        n_elements: int,
        bit_position: int,
    ) -> tuple[np.ndarray, int]:
        arrays = (luts, codes, packed_other_4bits)
        if any(array.dtype != np.uint8 or not array.flags.c_contiguous for array in arrays):
            raise ValidationError("ECF8 decoder inputs must be contiguous uint8 arrays")
        if start_element < 0 or n_elements < 0:
            raise ValidationError("ECF8 decoder element range must be non-negative")
        output = np.empty(n_elements, dtype=np.uint8)
        position = ctypes.c_uint64(bit_position)
        failed_element = ctypes.c_size_t(0)
        result = self.decode_function(
            luts.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            luts.shape[0],
            codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            codes.size,
            packed_other_4bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            packed_other_4bits.size,
            start_element,
            n_elements,
            ctypes.byref(position),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.byref(failed_element),
        )
        if result:
            reason = DECODER_ERRORS.get(result, f"decoder error {result}")
            raise ValidationError(
                f"{reason} at decoded element "
                f"{start_element + failed_element.value}"
            )
        return output, position.value


def numpy_uint8(tensor: torch.Tensor, label: str) -> np.ndarray:
    if tensor.dtype != torch.uint8:
        raise ValidationError(f"{label} must be uint8, got {tensor.dtype}")
    if tensor.ndim != 1:
        raise ValidationError(f"{label} must be 1-D, got shape {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValidationError(f"{label} must be contiguous")
    return tensor.numpy()


def first_difference(left: np.ndarray, right: np.ndarray) -> int | None:
    different = left != right
    return int(np.argmax(different)) if np.any(different) else None


def compare_direct_tensor(
    source_ref: TensorRef,
    output_ref: TensorRef,
    chunk_bytes: int,
) -> int:
    if source_ref.dtype != output_ref.dtype:
        raise ValidationError(
            f"{source_ref.name}: dtype changed from {source_ref.dtype} "
            f"to {output_ref.dtype}"
        )
    if source_ref.shape != output_ref.shape:
        raise ValidationError(
            f"{source_ref.name}: shape changed from {source_ref.shape} "
            f"to {output_ref.shape}"
        )
    source_tensor = load_tensor(source_ref)
    output_tensor = load_tensor(output_ref)
    source_bytes = source_tensor.view(torch.uint8).reshape(-1).numpy()
    output_bytes = output_tensor.view(torch.uint8).reshape(-1).numpy()
    if source_bytes.size != output_bytes.size:
        raise ValidationError(
            f"{source_ref.name}: raw byte length changed from "
            f"{source_bytes.size} to {output_bytes.size}"
        )
    for start in range(0, source_bytes.size, chunk_bytes):
        end = min(start + chunk_bytes, source_bytes.size)
        offset = first_difference(source_bytes[start:end], output_bytes[start:end])
        if offset is not None:
            absolute = start + offset
            raise ValidationError(
                f"{source_ref.name}: bit pattern differs at byte {absolute} "
                f"(source=0x{source_bytes[absolute]:02x}, "
                f"output=0x{output_bytes[absolute]:02x})"
            )
    return source_bytes.size


def dfloat11_config(converted: Path) -> dict[str, Any]:
    if not converted.is_dir():
        raise ValidationError(
            "DFloat11 directory output with config.json is required"
        )
    config_path = converted / "config.json"
    try:
        config = json.loads(config_path.read_text())
        dfloat_config = config["dfloat11_config"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValidationError(
            f"missing or invalid DFloat11 config: {config_path}"
        ) from exc
    if not isinstance(dfloat_config, dict) or not isinstance(
        dfloat_config.get("pattern_dict"), dict
    ):
        raise ValidationError("invalid dfloat11_config.pattern_dict")
    if dfloat_config.get("version") not in SUPPORTED_DFLOAT11_FORMATS:
        raise ValidationError(
            "unsupported DFloat11 format version "
            f"{dfloat_config.get('version')!r}; expected one of "
            f"{sorted(SUPPORTED_DFLOAT11_FORMATS)}"
        )
    return dfloat_config


def ecf8_config(converted: Path) -> dict[str, Any]:
    if not converted.is_dir():
        raise ValidationError("ECF8 directory output with config.json is required")
    config_path = converted / "config.json"
    pattern_path = converted / "pattern_dict.json"
    try:
        config = json.loads(config_path.read_text())
        dfloat_config = config["dfloat_config"]
        pattern_config = json.loads(pattern_path.read_text())
        fp8_patterns = pattern_config["fp8"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValidationError(
            f"missing or invalid ECF8 config: {config_path} / {pattern_path}"
        ) from exc
    if not isinstance(dfloat_config, dict) or not isinstance(fp8_patterns, dict):
        raise ValidationError("invalid ECF8 dfloat_config or pattern_dict.fp8")
    if dfloat_config.get("version") not in SUPPORTED_ECF8_FORMATS:
        raise ValidationError(
            "unsupported ECF8 format version "
            f"{dfloat_config.get('version')!r}; expected one of "
            f"{sorted(SUPPORTED_ECF8_FORMATS)}"
        )
    result = dict(dfloat_config)
    result["pattern_dict"] = fp8_patterns
    return result


def group_weight_names(
    base: str,
    pattern_dict: dict[str, Any],
) -> tuple[str, ...]:
    matches = [
        attributes
        for pattern, attributes in pattern_dict.items()
        if re.fullmatch(pattern, base)
    ]
    if len(matches) != 1:
        raise ValidationError(
            f"compressed group {base!r} matches {len(matches)} patterns"
        )
    attributes = matches[0]
    if not isinstance(attributes, (list, tuple)) or any(
        not isinstance(item, str) for item in attributes
    ):
        raise ValidationError(f"invalid pattern attributes for {base!r}")
    if attributes:
        weight_names = tuple(
            f"{base}.{attribute}.weight" if attribute else f"{base}.weight"
            for attribute in attributes
        )
    else:
        weight_names = (f"{base}.weight",)
    if len(weight_names) != len(set(weight_names)):
        raise ValidationError(f"duplicate pattern attributes for {base!r}")
    return weight_names


def required_group_refs(
    output_index: dict[str, TensorRef],
    base: str,
    suffixes: tuple[str, ...] = DFLOAT11_AUXILIARY_SUFFIXES,
) -> dict[str, TensorRef]:
    refs = {}
    for suffix in suffixes:
        name = base + suffix
        if name not in output_index:
            raise ValidationError(
                f"compressed group {base!r} is missing {name!r}"
            )
        refs[suffix] = output_index[name]
    return refs


def positive_config_int(config: dict[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValidationError(f"invalid positive integer config field {key!r}")
    return value


def require_frozen_cuda_geometry(
    method: str,
    config: dict[str, Any],
) -> None:
    bytes_per_thread = positive_config_int(config, "bytes_per_thread")
    threads_per_block = positive_config_int(config, "threads_per_block")
    if (bytes_per_thread, threads_per_block) != (8, 512):
        raise ValidationError(
            f"{method} validator targets bytes_per_thread=8 and "
            f"threads_per_block=512, got {bytes_per_thread} and "
            f"{threads_per_block}"
        )


def validate_parallel_metadata_structure(
    base: str,
    refs: dict[str, TensorRef],
    config: dict[str, Any],
    *,
    encoded_bytes: int,
    n_elements: int,
    output_position_bits: int,
    gap_bits: int,
) -> None:
    bytes_per_thread = positive_config_int(config, "bytes_per_thread")
    threads_per_block = positive_config_int(config, "threads_per_block")
    bytes_per_block = bytes_per_thread * threads_per_block
    n_blocks = (encoded_bytes + bytes_per_block - 1) // bytes_per_block
    if n_blocks < 1:
        raise ValidationError(f"{base}: encoded Huffman stream is empty")

    output_positions = numpy_uint8(
        load_tensor(refs[".output_positions"]),
        f"{base}.output_positions",
    )
    item_bytes = output_position_bits // 8
    if output_position_bits not in (32, 64) or output_positions.size % item_bytes:
        raise ValidationError(
            f"{base}.output_positions has invalid byte length "
            f"{output_positions.size}"
        )
    position_dtype = np.dtype(
        "<u4" if output_position_bits == 32 else "<u8"
    )
    positions = output_positions.view(position_dtype)
    if positions.size != n_blocks + 1:
        raise ValidationError(
            f"{base}.output_positions has {positions.size} entries; "
            f"expected {n_blocks + 1}"
        )
    if (
        int(positions[0]) != 0
        or int(positions[-1]) != n_elements
        or np.any(positions[1:] < positions[:-1])
        or np.any(positions > n_elements)
    ):
        raise ValidationError(
            f"{base}.output_positions is not a monotone [0, {n_elements}] "
            "block boundary table"
        )

    gaps = numpy_uint8(load_tensor(refs[".gaps"]), f"{base}.gaps")
    expected_gap_bytes = (
        n_blocks * threads_per_block * gap_bits + 7
    ) // 8
    if gaps.size != expected_gap_bytes:
        raise ValidationError(
            f"{base}.gaps has {gaps.size} bytes; expected "
            f"{expected_gap_bytes}"
        )

    used_threads = (encoded_bytes + bytes_per_thread - 1) // bytes_per_thread
    unused_threads = n_blocks * threads_per_block - used_threads
    if unused_threads:
        trailing_bits = np.unpackbits(
            gaps[-((threads_per_block * gap_bits + 7) // 8):],
            bitorder="big",
        )
        trailing_values = trailing_bits[
            :threads_per_block * gap_bits
        ].reshape(-1, gap_bits)
        if np.any(trailing_values[-unused_threads:]):
            raise ValidationError(
                f"{base}.gaps has nonzero padding for unused CUDA threads"
            )


def validate_logical_stream_length(
    base: str,
    encoded_bytes: int,
    bit_position: int,
) -> None:
    expected_bytes = (bit_position + 7) // 8
    if encoded_bytes != expected_bytes:
        raise ValidationError(
            f"{base}: decoded Huffman stream consumes {bit_position} bits "
            f"({expected_bytes} bytes), but encoded payload has "
            f"{encoded_bytes} bytes"
        )


def compare_decoded_group(
    decoder: DFloat11CPUDecoder,
    base: str,
    weight_names: tuple[str, ...],
    source_index: dict[str, TensorRef],
    output_index: dict[str, TensorRef],
    config: dict[str, Any],
    chunk_elements: int,
) -> tuple[int, int]:
    refs = required_group_refs(output_index, base)
    missing = [name for name in weight_names if name not in source_index]
    if missing:
        raise ValidationError(
            f"compressed group {base!r} refers to missing source tensor(s): "
            f"{missing}"
        )
    luts_tensor = load_tensor(refs[".luts"])
    if (
        luts_tensor.dtype != torch.uint8
        or luts_tensor.ndim != 2
        or luts_tensor.shape[1] != 256
        or luts_tensor.shape[0] < 2
    ):
        raise ValidationError(
            f"{base}.luts must have uint8 shape [rows>=2, 256]"
        )
    luts = luts_tensor.numpy()
    codes = numpy_uint8(
        load_tensor(refs[".encoded_exponent"]),
        f"{base}.encoded_exponent",
    )
    sign_mantissa = numpy_uint8(
        load_tensor(refs[".sign_mantissa"]),
        f"{base}.sign_mantissa",
    )
    if not codes.size:
        raise ValidationError(f"{base}.encoded_exponent is empty")
    split_tensor = load_tensor(refs[".split_positions"])
    if split_tensor.dtype != torch.int64 or split_tensor.ndim != 1:
        raise ValidationError(f"{base}.split_positions must be 1-D int64")
    split_positions = split_tensor.numpy().tolist()

    source_sizes = []
    for name in weight_names:
        source_ref = source_index[name]
        if source_ref.dtype != "BF16":
            raise ValidationError(
                f"{name}: compressed source must be BF16, got {source_ref.dtype}"
            )
        source_sizes.append(int(np.prod(source_ref.shape, dtype=np.int64)))
    expected_splits = np.cumsum(source_sizes, dtype=np.int64)[:-1].tolist()
    if split_positions != expected_splits:
        raise ValidationError(
            f"{base}.split_positions={split_positions} does not match "
            f"source boundaries {expected_splits}"
        )
    if sign_mantissa.size != sum(source_sizes):
        raise ValidationError(
            f"{base}.sign_mantissa has {sign_mantissa.size} elements; "
            f"source weights have {sum(source_sizes)}"
        )
    validate_parallel_metadata_structure(
        base,
        refs,
        config,
        encoded_bytes=codes.size,
        n_elements=sum(source_sizes),
        output_position_bits=32,
        gap_bits=5,
    )

    bit_position = 0
    element_position = 0
    compared_bytes = 0
    for name, expected_elements in zip(weight_names, source_sizes):
        source_tensor = load_tensor(source_index[name])
        if not source_tensor.is_contiguous():
            raise ValidationError(f"{name}: source tensor must be contiguous")
        source_bits = (
            source_tensor.reshape(-1)
            .view(torch.int16)
            .numpy()
            .view(np.uint16)
        )
        if source_bits.size != expected_elements:
            raise ValidationError(f"{name}: source tensor element count changed")
        for start in range(0, expected_elements, chunk_elements):
            count = min(chunk_elements, expected_elements - start)
            decoded, bit_position = decoder.decode(
                luts,
                codes,
                sign_mantissa[
                    element_position + start:
                    element_position + start + count
                ],
                bit_position,
            )
            offset = first_difference(
                source_bits[start:start + count],
                decoded,
            )
            if offset is not None:
                absolute = start + offset
                raise ValidationError(
                    f"{name}: BF16 bit pattern differs at element {absolute} "
                    f"(source=0x{source_bits[absolute]:04x}, "
                    f"decoded=0x{decoded[offset]:04x})"
                )
        element_position += expected_elements
        compared_bytes += expected_elements * 2
    validate_logical_stream_length(base, codes.size, bit_position)
    return len(weight_names), compared_bytes


def compare_ecf8_group(
    decoder: ECF8CPUDecoder,
    base: str,
    weight_names: tuple[str, ...],
    source_index: dict[str, TensorRef],
    output_index: dict[str, TensorRef],
    config: dict[str, Any],
    chunk_elements: int,
) -> tuple[int, int]:
    refs = required_group_refs(output_index, base, ECF8_AUXILIARY_SUFFIXES)
    missing = [name for name in weight_names if name not in source_index]
    if missing:
        raise ValidationError(
            f"compressed group {base!r} refers to missing source tensor(s): "
            f"{missing}"
        )
    luts_tensor = load_tensor(refs[".luts"])
    if (
        luts_tensor.dtype != torch.uint8
        or luts_tensor.ndim != 2
        or luts_tensor.shape[1] != 256
        or luts_tensor.shape[0] < 2
    ):
        raise ValidationError(f"{base}.luts must have uint8 shape [rows>=2, 256]")
    luts = luts_tensor.numpy()
    codes = numpy_uint8(load_tensor(refs[".encoded"]), f"{base}.encoded")
    packed_other_4bits = numpy_uint8(
        load_tensor(refs[".packed_other_4bits"]),
        f"{base}.packed_other_4bits",
    )
    if not codes.size:
        raise ValidationError(f"{base}.encoded is empty")
    split_tensor = load_tensor(refs[".split_positions"])
    if split_tensor.dtype != torch.int64 or split_tensor.ndim != 1:
        raise ValidationError(f"{base}.split_positions must be 1-D int64")
    split_positions = split_tensor.numpy().tolist()

    source_sizes = []
    for name in weight_names:
        source_ref = source_index[name]
        if source_ref.dtype != "F8_E4M3":
            raise ValidationError(
                f"{name}: compressed source must be F8_E4M3, "
                f"got {source_ref.dtype}"
            )
        source_sizes.append(int(np.prod(source_ref.shape, dtype=np.int64)))
    total_elements = sum(source_sizes)
    expected_splits = np.cumsum(source_sizes, dtype=np.int64)[:-1].tolist()
    if split_positions != expected_splits:
        raise ValidationError(
            f"{base}.split_positions={split_positions} does not match "
            f"source boundaries {expected_splits}"
        )
    if packed_other_4bits.size * 2 != total_elements:
        raise ValidationError(
            f"{base}.packed_other_4bits represents "
            f"{packed_other_4bits.size * 2} elements; source weights have "
            f"{total_elements}"
        )
    validate_parallel_metadata_structure(
        base,
        refs,
        config,
        encoded_bytes=codes.size,
        n_elements=total_elements,
        output_position_bits=64,
        gap_bits=4,
    )

    bit_position = 0
    element_position = 0
    compared_bytes = 0
    for name, expected_elements in zip(weight_names, source_sizes):
        source_tensor = load_tensor(source_index[name])
        if not source_tensor.is_contiguous():
            raise ValidationError(f"{name}: source tensor must be contiguous")
        source_bits = source_tensor.reshape(-1).view(torch.uint8).numpy()
        if source_bits.size != expected_elements:
            raise ValidationError(f"{name}: source tensor element count changed")
        for start in range(0, expected_elements, chunk_elements):
            count = min(chunk_elements, expected_elements - start)
            decoded, bit_position = decoder.decode(
                luts,
                codes,
                packed_other_4bits,
                element_position + start,
                count,
                bit_position,
            )
            offset = first_difference(source_bits[start:start + count], decoded)
            if offset is not None:
                absolute = start + offset
                raise ValidationError(
                    f"{name}: FP8 E4M3 bit pattern differs at element "
                    f"{absolute} (source=0x{source_bits[absolute]:02x}, "
                    f"decoded=0x{decoded[offset]:02x})"
                )
        element_position += expected_elements
        compared_bytes += expected_elements
    validate_logical_stream_length(base, codes.size, bit_position)
    return len(weight_names), compared_bytes


def validate_dfloat11(
    source: Path,
    converted: Path,
    *,
    chunk_mib: int = 64,
    build_dir: Path | None = None,
) -> tuple[ValidationStats, dict[str, Any]]:
    if chunk_mib < 1:
        raise ValidationError("chunk_mib must be positive")
    source = source.expanduser().resolve()
    converted = converted.expanduser().resolve()
    config = dfloat11_config(converted)
    require_frozen_cuda_geometry("DFloat11", config)
    source_paths = source_weight_paths(source)
    output_paths = converted_weight_paths(converted)
    source_states = capture_file_states(source_paths)
    output_states = capture_file_states(output_paths)
    source_manifest_path, _ = load_source_manifest(source)
    source_manifest_states = None
    if source_manifest_path is not None:
        source_manifest_states = capture_file_states((source_manifest_path,))
    converted_metadata_paths = (converted / "config.json",)
    converted_metadata_states = capture_file_states(converted_metadata_paths)
    source_index = tensor_index(source_paths, "source")
    output_index = tensor_index(output_paths, "converted")
    if not source_index:
        raise ValidationError("source checkpoint has no tensors")
    library, decoder_source_sha256 = compile_decoder(build_dir)
    decoder = DFloat11CPUDecoder(library)
    stats = ValidationStats(source_tensors=len(source_index))
    reconstructed: set[str] = set()
    auxiliary: set[str] = set()
    group_bases = sorted(
        name[: -len(".encoded_exponent")]
        for name in output_index
        if name.endswith(".encoded_exponent")
    )
    if not group_bases:
        raise ValidationError("converted output has no DFloat11 groups")
    chunk_bytes = chunk_mib * 1024 * 1024
    chunk_elements = max(1, chunk_bytes // 2)
    for base in group_bases:
        weight_names = group_weight_names(base, config["pattern_dict"])
        duplicates = reconstructed.intersection(weight_names)
        if duplicates:
            raise ValidationError(
                f"source tensor(s) reconstructed more than once: "
                f"{sorted(duplicates)}"
            )
        stored_duplicates = set(weight_names).intersection(output_index)
        if stored_duplicates:
            raise ValidationError(
                f"compressed source tensor(s) are also stored directly: "
                f"{sorted(stored_duplicates)}"
            )
        decoded_tensors, compared_bytes = compare_decoded_group(
            decoder,
            base,
            weight_names,
            source_index,
            output_index,
            config,
            chunk_elements,
        )
        reconstructed.update(weight_names)
        auxiliary.update(base + suffix for suffix in DFLOAT11_AUXILIARY_SUFFIXES)
        stats.decoded_groups += 1
        stats.decoded_tensors += decoded_tensors
        stats.compared_bytes += compared_bytes

    for name, source_ref in source_index.items():
        if name in reconstructed:
            continue
        output_ref = output_index.get(name)
        if output_ref is None:
            raise ValidationError(
                f"source tensor {name!r} is neither directly stored nor "
                "reconstructed by a DFloat11 group"
            )
        stats.compared_bytes += compare_direct_tensor(
            source_ref,
            output_ref,
            chunk_bytes,
        )
        stats.direct_tensors += 1

    ignored = sorted(set(output_index) - set(source_index) - auxiliary)
    if ignored:
        raise ValidationError(
            f"converted output contains unclassified tensor(s): {ignored}"
        )
    if converted_weight_paths(converted) != output_paths:
        raise ValidationError("converted safetensors file set changed during validation")
    source_fingerprints = fingerprint_files(source_paths, source_states)
    manifest, manifest_fingerprint = verify_source_manifest_fingerprints(
        source,
        source_paths,
        source_fingerprints,
        source_manifest_states,
    )
    converted_fingerprints = fingerprint_files(output_paths, output_states)
    converted_metadata_fingerprints = fingerprint_files(
        converted_metadata_paths,
        converted_metadata_states,
    )
    require_unchanged_file_states(source_states)
    require_unchanged_file_states(output_states)
    require_unchanged_file_states(source_manifest_states)
    require_unchanged_file_states(converted_metadata_states)
    if converted_weight_paths(converted) != output_paths:
        raise ValidationError("converted safetensors file set changed during validation")
    stats.ignored_output_tensors = ()
    provenance = {
        "source_files": source_fingerprints,
        "source_manifest": manifest_fingerprint,
        "converted_files": converted_fingerprints,
        "converted_metadata_files": converted_metadata_fingerprints,
        "source_repo_id": manifest.get("repo_id"),
        "source_revision": manifest.get("revision"),
        "source_manifest_declared_sha256_verified": manifest.get(
            "sha256_verified"
        ),
        "decoder_library": str(library),
        "decoder_library_sha256": sha256(library),
        "decoder_source_sha256": decoder_source_sha256,
        "validator_sha256": sha256(Path(__file__)),
        "dfloat11_format_version": config.get("version"),
        "format_reference_commit": DFLOAT11_REFERENCE_COMMIT,
    }
    return stats, provenance


def validate_ecf8(
    source: Path,
    converted: Path,
    *,
    chunk_mib: int = 64,
    build_dir: Path | None = None,
) -> tuple[ValidationStats, dict[str, Any]]:
    if chunk_mib < 1:
        raise ValidationError("chunk_mib must be positive")
    source = source.expanduser().resolve()
    converted = converted.expanduser().resolve()
    config = ecf8_config(converted)
    require_frozen_cuda_geometry("ECF8", config)
    source_paths = source_weight_paths(source)
    output_paths = converted_weight_paths(converted)
    source_states = capture_file_states(source_paths)
    output_states = capture_file_states(output_paths)
    source_manifest_path, _ = load_source_manifest(source)
    source_manifest_states = None
    if source_manifest_path is not None:
        source_manifest_states = capture_file_states((source_manifest_path,))
    converted_metadata_paths = (
        converted / "config.json",
        converted / "pattern_dict.json",
    )
    converted_metadata_states = capture_file_states(converted_metadata_paths)
    source_index = tensor_index(source_paths, "source")
    output_index = tensor_index(output_paths, "converted")
    if not source_index:
        raise ValidationError("source checkpoint has no tensors")
    library, decoder_source_sha256 = compile_decoder(build_dir)
    decoder = ECF8CPUDecoder(library)
    stats = ValidationStats(source_tensors=len(source_index))
    reconstructed: set[str] = set()
    auxiliary: set[str] = set()
    group_bases = sorted(
        name[: -len(".encoded")]
        for name in output_index
        if name.endswith(".encoded")
    )
    if not group_bases:
        raise ValidationError("converted output has no ECF8 groups")
    chunk_bytes = chunk_mib * 1024 * 1024
    chunk_elements = max(1, chunk_bytes)
    for base in group_bases:
        weight_names = group_weight_names(base, config["pattern_dict"])
        duplicates = reconstructed.intersection(weight_names)
        if duplicates:
            raise ValidationError(
                f"source tensor(s) reconstructed more than once: "
                f"{sorted(duplicates)}"
            )
        stored_duplicates = set(weight_names).intersection(output_index)
        if stored_duplicates:
            raise ValidationError(
                f"compressed source tensor(s) are also stored directly: "
                f"{sorted(stored_duplicates)}"
            )
        decoded_tensors, compared_bytes = compare_ecf8_group(
            decoder,
            base,
            weight_names,
            source_index,
            output_index,
            config,
            chunk_elements,
        )
        reconstructed.update(weight_names)
        auxiliary.update(base + suffix for suffix in ECF8_AUXILIARY_SUFFIXES)
        stats.decoded_groups += 1
        stats.decoded_tensors += decoded_tensors
        stats.compared_bytes += compared_bytes

    for name, source_ref in source_index.items():
        if name in reconstructed:
            continue
        output_ref = output_index.get(name)
        if output_ref is None:
            raise ValidationError(
                f"source tensor {name!r} is neither directly stored nor "
                "reconstructed by an ECF8 group"
            )
        stats.compared_bytes += compare_direct_tensor(
            source_ref,
            output_ref,
            chunk_bytes,
        )
        stats.direct_tensors += 1

    ignored = sorted(set(output_index) - set(source_index) - auxiliary)
    if ignored:
        raise ValidationError(
            f"converted output contains unclassified tensor(s): {ignored}"
        )
    if converted_weight_paths(converted) != output_paths:
        raise ValidationError("converted safetensors file set changed during validation")
    source_fingerprints = fingerprint_files(source_paths, source_states)
    manifest, manifest_fingerprint = verify_source_manifest_fingerprints(
        source,
        source_paths,
        source_fingerprints,
        source_manifest_states,
    )
    converted_fingerprints = fingerprint_files(output_paths, output_states)
    converted_metadata_fingerprints = fingerprint_files(
        converted_metadata_paths,
        converted_metadata_states,
    )
    require_unchanged_file_states(source_states)
    require_unchanged_file_states(output_states)
    require_unchanged_file_states(source_manifest_states)
    require_unchanged_file_states(converted_metadata_states)
    if converted_weight_paths(converted) != output_paths:
        raise ValidationError("converted safetensors file set changed during validation")
    stats.ignored_output_tensors = ()
    provenance = {
        "source_files": source_fingerprints,
        "source_manifest": manifest_fingerprint,
        "converted_files": converted_fingerprints,
        "converted_metadata_files": converted_metadata_fingerprints,
        "source_repo_id": manifest.get("repo_id"),
        "source_revision": manifest.get("revision"),
        "source_manifest_declared_sha256_verified": manifest.get(
            "sha256_verified"
        ),
        "decoder_library": str(library),
        "decoder_library_sha256": sha256(library),
        "decoder_source_sha256": decoder_source_sha256,
        "validator_sha256": sha256(Path(__file__)),
        "ecf8_format_version": config.get("version"),
        "format_reference_commit": ECF8_REFERENCE_COMMIT,
    }
    return stats, provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Independently CPU-decode DFloat11 or ECF8 exponent streams and "
            "compare every source checkpoint tensor by raw bit pattern."
        )
    )
    parser.add_argument("method", choices=("dfloat11", "ecf8"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--converted", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--chunk-mib", type=int, default=64)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def write_report(path: Path, report: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    started_at = utc_now()
    started = time.perf_counter()
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": args.method,
        "validation": "independent_cpu_huffman_decode_and_bit_pattern_compare",
        "exactness_scope": "manifest_declared_tensor_bit_patterns",
        "official_decoder_used": False,
        "parallel_metadata_validation": "structural_only",
        "parallel_metadata_checks": (
            "shape_length_endpoints_monotonicity_and_unused_thread_padding"
        ),
        "source": str(args.source.expanduser().resolve()),
        "converted": str(args.converted.expanduser().resolve()),
        "started_at": started_at,
        "validator_sha256": sha256(Path(__file__)),
        "decoder_source_sha256": sha256(DECODER_SOURCE),
    }
    try:
        validate = (
            validate_dfloat11
            if args.method == "dfloat11"
            else validate_ecf8
        )
        stats, provenance = validate(
            args.source,
            args.converted,
            chunk_mib=args.chunk_mib,
            build_dir=args.build_dir,
        )
    except ValidationError as exc:
        report.update(
            {
                "status": "failed",
                "exact": False,
                "error": str(exc),
                "finished_at": utc_now(),
                "wall_seconds": time.perf_counter() - started,
            }
        )
        exit_code = 1
    else:
        report.update(
            {
                "status": "ok",
                "exact": True,
                "stats": asdict(stats),
                "provenance": provenance,
                "finished_at": utc_now(),
                "wall_seconds": time.perf_counter() - started,
            }
        )
        exit_code = 0
    if args.report:
        write_report(args.report, report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    elif exit_code:
        print(
            f"{args.method.upper()} bit-pattern validation FAILED: "
            f"{report['error']}"
        )
    else:
        stats = report["stats"]
        print(
            f"{args.method.upper()} bit-pattern validation PASS: "
            f"{stats['source_tensors']} tensors, "
            f"{stats['decoded_groups']} compressed groups, "
            f"{stats['compared_bytes']} bytes compared"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
