#!/usr/bin/env python3
"""Read-only tensor/operator attribution for BRTA v3 / BRPG v2 archives.

The analyzer deliberately does not execute programs or decode literal payloads.
It uses the independently framed tensor records to measure exact per-tensor
archive bytes, and parses program/literal framing to attribute exact occupied
wire bytes.  Per-operator *causal savings* are not recoverable from an archive:
that would require counterfactual programs that are not persisted.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import struct
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence


BRTA_MAGIC = b"BRTA"
BRTA_VERSION = 3
BRPG_MAGIC = b"BRPG"
BRPG_VERSION = 2
RECORD_TAG = 1
CHECKSUM_BYTES = 8
PROGRAM_HEADER_BYTES = len(BRPG_MAGIC) + 1

MAX_TENSORS = 1_000_000
MAX_PREFIX_BYTES = 64 * 1024 * 1024
MAX_NAME_BYTES = 1024 * 1024
MAX_DIMENSIONS = 1024
MAX_NODES = 1_000_000
MAX_DEPTH = 256
MAX_U64 = (1 << 64) - 1


@dataclass(frozen=True)
class Dtype:
    name: str
    element_bytes: int
    bits: int
    float_fields: tuple[int, int] | None = None


DTYPES: dict[int, Dtype] = {
    0x01: Dtype("F16", 2, 16, (5, 10)),
    0x02: Dtype("BF16", 2, 16, (8, 7)),
    0x03: Dtype("F32", 4, 32, (8, 23)),
    0x04: Dtype("U8", 1, 8),
    0x05: Dtype("U16", 2, 16),
    0x06: Dtype("U32", 4, 32),
    0x07: Dtype("I8", 1, 8),
    0x08: Dtype("I16", 2, 16),
    0x09: Dtype("I32", 4, 32),
    0x0A: Dtype("F8_E4M3", 1, 8, (4, 3)),
    0x0B: Dtype("F8_E5M2", 1, 8, (5, 2)),
}
DTYPE_BY_NAME = {dtype.name: dtype for dtype in DTYPES.values()}

NODE_NAMES = {
    0x01: "literal",
    0x02: "constant",
    0x03: "concat",
    0x04: "repeat",
    0x05: "map",
    0x06: "scan",
    0x07: "merge",
}
MAP_NAMES = {
    0x01: "xor",
    0x02: "add_mod",
    0x03: "zigzag",
    0x04: "gray",
    0x05: "rotate_left",
    0x06: "bit_reverse",
}
SCAN_NAMES = {0x01: "xor", 0x02: "add_mod"}
MERGE_NAMES = {
    0x01: "fields",
    0x02: "float_fields",
    0x03: "bit_planes",
    0x04: "byte_planes",
}
LITERAL_CODECS = {0: "raw", 1: "bitpack", 2: "huffman", 3: "rans"}


class AttributionError(RuntimeError):
    """A malformed or unsupported attribution input."""


class Cursor:
    """A bounded random-access cursor over one already-open file."""

    def __init__(self, handle: Any, start: int, end: int, label: str):
        if start < 0 or end < start:
            raise AttributionError(f"{label}: invalid cursor bounds")
        self.handle = handle
        self.pos = start
        self.end = end
        self.label = label

    @property
    def remaining(self) -> int:
        return self.end - self.pos

    def read(self, count: int) -> bytes:
        if count < 0 or count > self.remaining:
            raise AttributionError(
                f"{self.label}: truncated at byte {self.pos}, need {count}, "
                f"have {self.remaining}"
            )
        self.handle.seek(self.pos)
        data = self.handle.read(count)
        if len(data) != count:
            raise AttributionError(f"{self.label}: short read at byte {self.pos}")
        self.pos += count
        return data

    def skip(self, count: int) -> None:
        if count < 0 or count > self.remaining:
            raise AttributionError(
                f"{self.label}: truncated skip at byte {self.pos}, need {count}, "
                f"have {self.remaining}"
            )
        self.pos += count

    def subcursor(self, count: int, label: str) -> "Cursor":
        if count < 0 or count > self.remaining:
            raise AttributionError(
                f"{self.label}: truncated {label}, need {count}, "
                f"have {self.remaining}"
            )
        child = Cursor(self.handle, self.pos, self.pos + count, label)
        self.pos += count
        return child

    def u8(self) -> int:
        return self.read(1)[0]

    def u32le(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64le(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def uleb128(self) -> int:
        value = 0
        for index in range(10):
            byte = self.u8()
            payload = byte & 0x7F
            if index == 9 and payload > 1:
                raise AttributionError(f"{self.label}: ULEB128 integer overflow")
            value |= payload << (index * 7)
            if byte & 0x80 == 0:
                if index and value < (1 << (index * 7)):
                    raise AttributionError(f"{self.label}: overlong ULEB128")
                return value
        raise AttributionError(f"{self.label}: ULEB128 integer overflow")

    def finish(self) -> None:
        if self.pos != self.end:
            raise AttributionError(
                f"{self.label}: {self.end - self.pos} trailing bytes"
            )


def checked_product(values: Sequence[int], label: str) -> int:
    if any(value < 0 for value in values):
        raise AttributionError(f"{label}: negative dimension")
    if any(value == 0 for value in values):
        return 0
    result = 1
    for value in values:
        result *= value
        if result > MAX_U64:
            raise AttributionError(f"{label}: shape product exceeds u64")
    return result


def storage_bytes(bits: int) -> int:
    if not 1 <= bits <= 32:
        raise AttributionError(f"invalid word width {bits}")
    if bits <= 8:
        return 1
    if bits <= 16:
        return 2
    return 4


def tensor_bytes(shape: Sequence[int], dtype: Dtype, label: str) -> int:
    result = checked_product(shape, label) * dtype.element_bytes
    if result > MAX_U64:
        raise AttributionError(f"{label}: tensor byte length exceeds u64")
    return result


@dataclass
class LiteralCodecStat:
    count: int = 0
    body_bytes: int = 0
    semantic_storage_bytes: int = 0
    payload_bytes: int = 0

    def add(
        self,
        *,
        body_bytes: int,
        semantic_storage_bytes: int,
        payload_bytes: int,
    ) -> None:
        self.count += 1
        self.body_bytes += body_bytes
        self.semantic_storage_bytes += semantic_storage_bytes
        self.payload_bytes += payload_bytes


@dataclass
class ProgramStats:
    node_count: int = 0
    max_depth: int = 0
    node_counts: Counter[str] = field(default_factory=Counter)
    operator_exclusive_bytes: Counter[str] = field(default_factory=Counter)
    literal_codecs: dict[str, LiteralCodecStat] = field(
        default_factory=lambda: defaultdict(LiteralCodecStat)
    )


@dataclass(frozen=True)
class NodeResult:
    bits: int
    length: int
    span_bytes: int
    operator: str


def _read_u32_uleb(cursor: Cursor, label: str) -> int:
    value = cursor.uleb128()
    if value > 0xFFFF_FFFF:
        raise AttributionError(f"{cursor.label}: {label} exceeds u32")
    return value


def _read_u8_uleb(cursor: Cursor, label: str) -> int:
    value = cursor.uleb128()
    if value > 0xFF:
        raise AttributionError(f"{cursor.label}: {label} exceeds u8")
    return value


def parse_literal_body(
    body: Cursor,
    bits: int,
    count: int,
) -> tuple[str, int]:
    if body.remaining < 1:
        raise AttributionError(f"{body.label}: empty literal body")
    codec_id = body.u8()
    codec = LITERAL_CODECS.get(codec_id)
    if codec is None:
        raise AttributionError(f"{body.label}: unknown literal codec {codec_id}")

    semantic_bytes = count * storage_bytes(bits)
    if codec == "raw":
        payload_bytes = semantic_bytes
        if body.remaining != payload_bytes:
            raise AttributionError(
                f"{body.label}: raw payload is {body.remaining} bytes, "
                f"expected {payload_bytes}"
            )
        body.skip(payload_bytes)
    elif codec == "bitpack":
        width = body.u8()
        if width < 1 or width > bits:
            raise AttributionError(f"{body.label}: invalid bitpack width {width}")
        payload_bytes = (count * width + 7) // 8
        if body.remaining != payload_bytes:
            raise AttributionError(
                f"{body.label}: bitpack payload is {body.remaining} bytes, "
                f"expected {payload_bytes}"
            )
        body.skip(payload_bytes)
    elif codec == "huffman":
        entries = body.u32le()
        table_bytes = entries * 5
        if table_bytes + 8 > body.remaining:
            raise AttributionError(f"{body.label}: truncated Huffman table")
        body.skip(table_bytes)
        payload_bits = body.u64le()
        payload_bytes = (payload_bits + 7) // 8
        if body.remaining != payload_bytes:
            raise AttributionError(
                f"{body.label}: Huffman payload is {body.remaining} bytes, "
                f"expected {payload_bytes}"
            )
        body.skip(payload_bytes)
    else:
        entries = body.u32le()
        table_bytes = entries * 8
        if table_bytes + 8 > body.remaining:
            raise AttributionError(f"{body.label}: truncated rANS table")
        body.skip(table_bytes)
        payload_bytes = body.u64le()
        if body.remaining != payload_bytes:
            raise AttributionError(
                f"{body.label}: rANS payload is {body.remaining} bytes, "
                f"expected {payload_bytes}"
            )
        body.skip(payload_bytes)
    body.finish()
    return codec, payload_bytes


def _validate_map_parameter(operation: str, parameter: int | None, bits: int) -> None:
    if operation in {"xor", "add_mod"}:
        assert parameter is not None
        mask = (1 << bits) - 1
        if parameter & ~mask:
            raise AttributionError(
                f"map.{operation}: parameter {parameter} does not fit {bits} bits"
            )
    elif operation == "rotate_left":
        assert parameter is not None
        if parameter >= bits:
            raise AttributionError(
                f"map.rotate_left: amount {parameter} is not below width {bits}"
            )


def _merge_type(
    operation: str,
    parameter: int | Dtype | None,
    children: Sequence[NodeResult],
) -> tuple[int, int]:
    if len(children) < 2 or len(children) > 32:
        raise AttributionError(f"merge.{operation}: invalid arity {len(children)}")
    length = children[0].length
    if length == 0 or any(child.length != length for child in children[1:]):
        raise AttributionError(f"merge.{operation}: child lengths disagree or are zero")

    if operation == "float_fields":
        assert isinstance(parameter, Dtype)
        if parameter.float_fields is None:
            raise AttributionError("merge.float_fields: dtype is not floating point")
        exponent, mantissa = parameter.float_fields
        expected = (1, exponent, mantissa)
        if len(children) != 3 or tuple(child.bits for child in children) != expected:
            raise AttributionError(
                "merge.float_fields: children do not match sign/exponent/mantissa"
            )
        return parameter.bits, length
    if operation == "bit_planes":
        if any(child.bits != 1 for child in children):
            raise AttributionError("merge.bit_planes: every child must be one bit")
        return len(children), length
    if operation == "byte_planes":
        if len(children) > 4:
            raise AttributionError("merge.byte_planes: more than four children")
        if any(child.bits != 8 for child in children[:-1]):
            raise AttributionError("merge.byte_planes: non-final children must be bytes")
        if not 1 <= children[-1].bits <= 8:
            raise AttributionError("merge.byte_planes: invalid final field width")
    elif operation == "fields":
        assert isinstance(parameter, int)
        if (
            len(children) != 2
            or parameter == 0
            or children[0].bits != parameter
        ):
            raise AttributionError("merge.fields: invalid low field")

    total_bits = sum(child.bits for child in children)
    if not 1 <= total_bits <= 32:
        raise AttributionError(f"merge.{operation}: invalid result width {total_bits}")
    return total_bits, length


def parse_node(cursor: Cursor, stats: ProgramStats, depth: int) -> NodeResult:
    if depth > MAX_DEPTH:
        raise AttributionError(f"{cursor.label}: program depth exceeds {MAX_DEPTH}")
    if stats.node_count >= MAX_NODES:
        raise AttributionError(f"{cursor.label}: program has more than {MAX_NODES} nodes")
    stats.node_count += 1
    stats.max_depth = max(stats.max_depth, depth)

    start = cursor.pos
    node_id = cursor.u8()
    kind = NODE_NAMES.get(node_id)
    if kind is None:
        raise AttributionError(f"{cursor.label}: unknown node id {node_id}")
    children: list[NodeResult] = []

    if kind == "literal":
        bits = cursor.u8()
        if not 1 <= bits <= 32:
            raise AttributionError(f"{cursor.label}: invalid literal width {bits}")
        length = cursor.uleb128()
        body_len = cursor.uleb128()
        body = cursor.subcursor(body_len, f"{cursor.label}/literal-body")
        codec, payload_bytes = parse_literal_body(body, bits, length)
        semantic_bytes = length * storage_bytes(bits)
        stats.literal_codecs[codec].add(
            body_bytes=body_len,
            semantic_storage_bytes=semantic_bytes,
            payload_bytes=payload_bytes,
        )
        operator = "literal"
    elif kind == "constant":
        bits = cursor.u8()
        length = cursor.uleb128()
        word = _read_u32_uleb(cursor, "constant word")
        if not 1 <= bits <= 32 or length == 0 or word >= (1 << bits):
            raise AttributionError(f"{cursor.label}: invalid constant")
        operator = "constant"
    elif kind == "concat":
        child_count = cursor.uleb128()
        if (
            child_count < 2
            or child_count > MAX_NODES - stats.node_count
            or child_count > cursor.remaining
        ):
            raise AttributionError(f"{cursor.label}: invalid concat arity {child_count}")
        children = [parse_node(cursor, stats, depth + 1) for _ in range(child_count)]
        bits = children[0].bits
        if children[0].length == 0 or any(
            child.bits != bits or child.length == 0 for child in children[1:]
        ):
            raise AttributionError(f"{cursor.label}: invalid concat child types")
        length = sum(child.length for child in children)
        operator = "concat"
    elif kind == "repeat":
        times = _read_u32_uleb(cursor, "repeat count")
        child = parse_node(cursor, stats, depth + 1)
        children = [child]
        if times < 2 or child.length == 0:
            raise AttributionError(f"{cursor.label}: invalid repeat")
        bits, length = child.bits, child.length * times
        operator = "repeat"
    elif kind == "map":
        operation_id = cursor.u8()
        operation = MAP_NAMES.get(operation_id)
        if operation is None:
            raise AttributionError(f"{cursor.label}: unknown map operation {operation_id}")
        parameter: int | None = None
        if operation in {"xor", "add_mod"}:
            parameter = _read_u32_uleb(cursor, f"map.{operation} parameter")
        elif operation == "rotate_left":
            parameter = _read_u8_uleb(cursor, "map.rotate_left amount")
        child = parse_node(cursor, stats, depth + 1)
        children = [child]
        if child.length == 0:
            raise AttributionError(f"{cursor.label}: map child is empty")
        _validate_map_parameter(operation, parameter, child.bits)
        bits, length = child.bits, child.length
        operator = f"map.{operation}"
    elif kind == "scan":
        operation_id = cursor.u8()
        operation = SCAN_NAMES.get(operation_id)
        if operation is None:
            raise AttributionError(f"{cursor.label}: unknown scan operation {operation_id}")
        initial = _read_u32_uleb(cursor, "scan initial")
        child = parse_node(cursor, stats, depth + 1)
        children = [child]
        if child.length == 0 or initial >= (1 << child.bits):
            raise AttributionError(f"{cursor.label}: invalid scan")
        bits, length = child.bits, child.length + 1
        operator = f"scan.{operation}"
    else:
        operation_id = cursor.u8()
        operation = MERGE_NAMES.get(operation_id)
        if operation is None:
            raise AttributionError(
                f"{cursor.label}: unknown merge operation {operation_id}"
            )
        parameter: int | Dtype | None = None
        if operation == "fields":
            parameter = _read_u8_uleb(cursor, "merge.fields low width")
        elif operation == "float_fields":
            dtype_id = cursor.u8()
            parameter = DTYPES.get(dtype_id)
            if parameter is None:
                raise AttributionError(
                    f"{cursor.label}: unknown merge dtype {dtype_id}"
                )
        child_count = cursor.uleb128()
        if (
            child_count > MAX_NODES - stats.node_count
            or child_count > cursor.remaining
        ):
            raise AttributionError(f"{cursor.label}: merge arity exceeds node limit")
        children = [parse_node(cursor, stats, depth + 1) for _ in range(child_count)]
        bits, length = _merge_type(operation, parameter, children)
        suffix = (
            f".{parameter.name.lower()}"
            if operation == "float_fields" and isinstance(parameter, Dtype)
            else ""
        )
        operator = f"merge.{operation}{suffix}"

    span = cursor.pos - start
    child_bytes = sum(child.span_bytes for child in children)
    exclusive = span - child_bytes
    if exclusive < 1:
        raise AttributionError(f"{cursor.label}: invalid node span")
    stats.node_counts[operator] += 1
    stats.operator_exclusive_bytes[operator] += exclusive
    return NodeResult(bits, length, span, operator)


def parse_program(program: Cursor) -> tuple[NodeResult, ProgramStats]:
    start = program.pos
    if program.read(len(BRPG_MAGIC)) != BRPG_MAGIC:
        raise AttributionError(f"{program.label}: bad BRPG magic")
    version = program.u8()
    if version != BRPG_VERSION:
        raise AttributionError(
            f"{program.label}: unsupported BRPG version {version}"
        )
    stats = ProgramStats()
    root = parse_node(program, stats, 1)
    program.finish()
    exclusive_total = sum(stats.operator_exclusive_bytes.values())
    total_bytes = program.end - start
    if (
        root.span_bytes + PROGRAM_HEADER_BYTES != total_bytes
        or exclusive_total != root.span_bytes
    ):
        raise AttributionError(f"{program.label}: program byte attribution drift")
    return root, stats


@dataclass(frozen=True)
class SafetensorMeta:
    name: str
    dtype: Dtype
    shape: tuple[int, ...]
    begin: int
    end: int

    @property
    def source_tensor_bytes(self) -> int:
        return self.end - self.begin


@dataclass(frozen=True)
class SourceDescriptor:
    path: Path
    file_bytes: int
    prefix_bytes: bytes
    prefix_sha256: str
    tensors: tuple[SafetensorMeta, ...]
    data_bytes: int


def parse_safetensors_prefix(prefix: bytes, label: str) -> tuple[SafetensorMeta, ...]:
    if len(prefix) < 8:
        raise AttributionError(f"{label}: safetensors prefix is shorter than 8 bytes")
    header_len = struct.unpack("<Q", prefix[:8])[0]
    if header_len + 8 != len(prefix):
        raise AttributionError(f"{label}: safetensors prefix length disagrees")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AttributionError(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    def bounded_json_integer(text: str) -> int:
        digits = text[1:] if text.startswith("-") else text
        if len(digits) > 20:
            raise ValueError("JSON integer exceeds 20 decimal digits")
        return int(text)

    try:
        header = json.loads(
            prefix[8:],
            object_pairs_hook=reject_duplicate_keys,
            parse_int=bounded_json_integer,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise AttributionError(f"{label}: malformed safetensors JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise AttributionError(f"{label}: safetensors header is not an object")

    tensors: list[SafetensorMeta] = []
    for name, raw in header.items():
        if name == "__metadata__":
            if not isinstance(raw, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in raw.items()
            ):
                raise AttributionError(f"{label}: invalid __metadata__")
            continue
        try:
            encoded_name = name.encode("utf-8")
        except (AttributeError, UnicodeEncodeError) as exc:
            raise AttributionError(f"{label}: invalid tensor name") from exc
        if len(encoded_name) > MAX_NAME_BYTES or not isinstance(raw, dict):
            raise AttributionError(f"{label}: invalid tensor metadata entry")
        dtype_name = raw.get("dtype")
        dtype = DTYPE_BY_NAME.get(dtype_name) if isinstance(dtype_name, str) else None
        shape = raw.get("shape")
        offsets = raw.get("data_offsets")
        if dtype is None:
            raise AttributionError(
                f"{label}: unsupported dtype {raw.get('dtype')!r} for {name}"
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
            raise AttributionError(f"{label}: invalid shape for {name}")
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
            raise AttributionError(f"{label}: invalid data_offsets for {name}")
        expected = tensor_bytes(shape, dtype, f"{label}/{name}")
        if offsets[1] - offsets[0] != expected:
            raise AttributionError(
                f"{label}: data length for {name} is {offsets[1] - offsets[0]}, "
                f"expected {expected}"
            )
        tensors.append(
            SafetensorMeta(
                name,
                dtype,
                tuple(shape),
                offsets[0],
                offsets[1],
            )
        )
    # Zig's format reader breaks equal-offset/equal-end ties by JSON source
    # order. Python's sort is stable, so omitting the name tie-break reproduces
    # that behavior for multiple zero-sized tensors at the same offset.
    tensors.sort(key=lambda item: (item.begin, item.end))
    if len(tensors) > MAX_TENSORS:
        raise AttributionError(f"{label}: too many tensors")
    cursor = 0
    for tensor in tensors:
        if tensor.begin != cursor:
            raise AttributionError(
                f"{label}: tensor data is not contiguous before {tensor.name}"
            )
        cursor = tensor.end
    return tuple(tensors)


def read_source_descriptor(path: Path) -> SourceDescriptor:
    file_bytes = path.stat().st_size
    with path.open("rb") as source:
        prefix_length_bytes = source.read(8)
        if len(prefix_length_bytes) != 8:
            raise AttributionError(f"{path}: safetensors file is too short")
        header_len = struct.unpack("<Q", prefix_length_bytes)[0]
        prefix_len = header_len + 8
        if prefix_len > MAX_PREFIX_BYTES:
            raise AttributionError(
                f"{path}: prefix {prefix_len} exceeds {MAX_PREFIX_BYTES}"
            )
        header = source.read(header_len)
        if len(header) != header_len:
            raise AttributionError(f"{path}: truncated safetensors header")
    prefix = prefix_length_bytes + header
    tensors = parse_safetensors_prefix(prefix, str(path))
    data_bytes = tensors[-1].end if tensors else 0
    if prefix_len + data_bytes != file_bytes:
        raise AttributionError(
            f"{path}: file size is {file_bytes}, metadata implies "
            f"{prefix_len + data_bytes}"
        )
    return SourceDescriptor(
        path.resolve(),
        file_bytes,
        prefix,
        hashlib.sha256(prefix).hexdigest(),
        tensors,
        data_bytes,
    )


@dataclass
class TensorRow:
    archive_path: str
    source_path: str | None
    tensor_index: int
    tensor_name: str
    role: str
    dtype: str
    shape: str
    source_tensor_bytes: int
    archive_record_bytes: int
    program_bytes: int
    record_framing_bytes: int
    saved_bytes_vs_record: int
    source_over_record_ratio: float
    record_percent_of_source: float | None
    program_selection: str
    root_operator: str
    node_count: int
    max_depth: int
    literal_count: int
    literal_semantic_storage_bytes: int
    literal_body_bytes: int
    operator_counts: dict[str, int]
    operator_exclusive_bytes: dict[str, int]
    literal_codec_counts: dict[str, int]
    literal_codec_body_bytes: dict[str, int]
    literal_codec_semantic_bytes: dict[str, int]
    literal_codec_payload_bytes: dict[str, int]
    stored_xxh3_64_le_bytes_hex: str

    def csv_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key in (
            "operator_counts",
            "operator_exclusive_bytes",
            "literal_codec_counts",
            "literal_codec_body_bytes",
            "literal_codec_semantic_bytes",
            "literal_codec_payload_bytes",
        ):
            result[key] = json.dumps(result[key], sort_keys=True, separators=(",", ":"))
        return result


@dataclass
class ArchiveSummary:
    archive_path: str
    source_path: str | None
    source_prefix_matched: bool | None
    tensor_count: int
    source_file_bytes: int
    source_prefix_bytes: int
    source_payload_bytes: int
    archive_file_bytes: int
    archive_header_bytes: int
    archive_record_bytes: int
    embedded_prefix_sha256: str
    source_over_archive_ratio: float
    archive_percent_of_source: float
    saved_bytes: int


_EMBEDDING = re.compile(
    r"(?:^|[._/])(?:embed(?:ding|dings)?|embed_tokens|word_embeddings|"
    r"position_embeddings|token_embedding|wte|wpe)(?:[._/]|$)",
    re.IGNORECASE,
)
_ATTENTION = re.compile(
    r"(?:self_attn|cross_attn|attention|(?:^|[._/])attn(?:[._/]|$)|"
    r"(?:^|[._/])(?:q|k|v|o)_proj(?:[._/]|$)|query_key_value)",
    re.IGNORECASE,
)
_MLP = re.compile(
    r"(?:[._/](?:mlp|ffn|feed_forward|feedforward)[._/]|"
    r"[._/](?:block_sparse_moe|experts?)[._/]|"
    r"(?:^|[._/])(?:gate|up|down)_proj(?:[._/]|$)|"
    r"(?:^|[._/])(?:fc[12]|w[123])(?:[._/]|$))",
    re.IGNORECASE,
)
_NORM = re.compile(
    r"(?:layer_?norm|layernorm|rms_?norm|(?:^|[._/])norm(?:[._/]|$)|"
    r"(?:^|[._/])ln_(?:1|2|f)(?:[._/]|$))",
    re.IGNORECASE,
)


def classify_tensor_role(name: str) -> str:
    if _EMBEDDING.search(name):
        return "embedding"
    # Normalization names often describe their position, for example
    # ``post_attention_layernorm`` or ``self_attn_layer_norm``.  Give the
    # semantic normalization suffix priority over those positional tokens.
    if _NORM.search(name):
        return "norm"
    if _ATTENTION.search(name):
        return "attention"
    if _MLP.search(name):
        return "mlp"
    return "other"


def _program_class(selection: str, root_operator: str) -> str:
    if selection == "literal_fallback":
        return "literal_fallback"
    return f"synthesized:{root_operator}"


def parse_archive(
    path: Path,
    source_by_prefix: dict[str, list[SourceDescriptor]],
    sources_supplied: bool,
) -> tuple[ArchiveSummary, list[TensorRow]]:
    archive_path = path.resolve()
    archive_bytes = archive_path.stat().st_size
    with archive_path.open("rb") as handle:
        archive = Cursor(handle, 0, archive_bytes, str(archive_path))
        if archive.read(len(BRTA_MAGIC)) != BRTA_MAGIC:
            raise AttributionError(f"{archive_path}: bad BRTA magic")
        version = archive.u8()
        if version != BRTA_VERSION:
            raise AttributionError(
                f"{archive_path}: unsupported BRTA version {version}"
            )
        tensor_count = archive.uleb128()
        if tensor_count > MAX_TENSORS:
            raise AttributionError(f"{archive_path}: too many tensors")
        prefix_len = archive.uleb128()
        if prefix_len > MAX_PREFIX_BYTES:
            raise AttributionError(f"{archive_path}: safetensors prefix is too large")
        prefix = archive.read(prefix_len)
        metadata = parse_safetensors_prefix(prefix, f"{archive_path}/embedded-prefix")
        if len(metadata) != tensor_count:
            raise AttributionError(
                f"{archive_path}: BRTA declares {tensor_count} tensors but embedded "
                f"metadata has {len(metadata)}"
            )
        prefix_sha = hashlib.sha256(prefix).hexdigest()
        candidates = source_by_prefix.get(prefix_sha, [])
        source: SourceDescriptor | None = None
        for candidate in candidates:
            if candidate.prefix_bytes == prefix:
                if source is not None:
                    raise AttributionError(
                        f"{archive_path}: embedded prefix matches multiple source files: "
                        f"{source.path}, {candidate.path}"
                    )
                source = candidate
        if sources_supplied and source is None:
            raise AttributionError(
                f"{archive_path}: embedded safetensors prefix matches no supplied source"
            )
        if source is not None and source.tensors != metadata:
            raise AttributionError(f"{archive_path}: matched source metadata drifted")

        header_bytes = archive.pos
        rows: list[TensorRow] = []
        records_total = 0
        for index, expected in enumerate(metadata):
            frame_start = archive.pos
            body_len = archive.uleb128()
            body = archive.subcursor(body_len, f"{archive_path}/record-{index}")
            if body.u8() != RECORD_TAG:
                raise AttributionError(f"{body.label}: unknown record tag")
            name_len = body.uleb128()
            if name_len > MAX_NAME_BYTES:
                raise AttributionError(f"{body.label}: tensor name is too large")
            try:
                name = body.read(name_len).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise AttributionError(f"{body.label}: tensor name is not UTF-8") from exc
            dtype_id = body.u8()
            dtype = DTYPES.get(dtype_id)
            if dtype is None:
                raise AttributionError(f"{body.label}: unknown dtype {dtype_id}")
            dimensions = body.uleb128()
            if dimensions > MAX_DIMENSIONS:
                raise AttributionError(f"{body.label}: too many dimensions")
            shape = tuple(body.uleb128() for _ in range(dimensions))
            source_tensor_bytes = tensor_bytes(shape, dtype, body.label)
            program_len = body.uleb128()
            if program_len > body.remaining - CHECKSUM_BYTES:
                raise AttributionError(f"{body.label}: program overlaps checksum")
            program = body.subcursor(program_len, f"{body.label}/program")
            root, stats = parse_program(program)
            stored_checksum = body.read(CHECKSUM_BYTES).hex()
            body.finish()

            if (
                name != expected.name
                or dtype != expected.dtype
                or shape != expected.shape
                or source_tensor_bytes != expected.source_tensor_bytes
            ):
                raise AttributionError(
                    f"{body.label}: record metadata does not match embedded prefix"
                )
            elements = checked_product(shape, body.label)
            if root.bits != dtype.bits or root.length != elements:
                raise AttributionError(
                    f"{body.label}: program type {root.bits}b[{root.length}] does "
                    f"not match {dtype.name}{shape}"
                )

            frame_bytes = archive.pos - frame_start
            records_total += frame_bytes
            literal_counts = {
                codec: item.count
                for codec, item in sorted(stats.literal_codecs.items())
            }
            literal_body = {
                codec: item.body_bytes
                for codec, item in sorted(stats.literal_codecs.items())
            }
            literal_semantic = {
                codec: item.semantic_storage_bytes
                for codec, item in sorted(stats.literal_codecs.items())
            }
            literal_payload = {
                codec: item.payload_bytes
                for codec, item in sorted(stats.literal_codecs.items())
            }
            literal_count = sum(literal_counts.values())
            selection = (
                "literal_fallback" if root.operator == "literal" else "synthesized"
            )
            rows.append(
                TensorRow(
                    archive_path=str(archive_path),
                    source_path=str(source.path) if source else None,
                    tensor_index=index,
                    tensor_name=name,
                    role=classify_tensor_role(name),
                    dtype=dtype.name,
                    shape="x".join(str(value) for value in shape),
                    source_tensor_bytes=source_tensor_bytes,
                    archive_record_bytes=frame_bytes,
                    program_bytes=program_len,
                    record_framing_bytes=frame_bytes - program_len,
                    saved_bytes_vs_record=source_tensor_bytes - frame_bytes,
                    source_over_record_ratio=(
                        source_tensor_bytes / frame_bytes
                        if frame_bytes
                        else math.inf
                    ),
                    record_percent_of_source=(
                        100.0 * frame_bytes / source_tensor_bytes
                        if source_tensor_bytes
                        else None
                    ),
                    program_selection=selection,
                    root_operator=root.operator,
                    node_count=stats.node_count,
                    max_depth=stats.max_depth,
                    literal_count=literal_count,
                    literal_semantic_storage_bytes=sum(literal_semantic.values()),
                    literal_body_bytes=sum(literal_body.values()),
                    operator_counts=dict(sorted(stats.node_counts.items())),
                    operator_exclusive_bytes=dict(
                        sorted(stats.operator_exclusive_bytes.items())
                    ),
                    literal_codec_counts=literal_counts,
                    literal_codec_body_bytes=literal_body,
                    literal_codec_semantic_bytes=literal_semantic,
                    literal_codec_payload_bytes=literal_payload,
                    stored_xxh3_64_le_bytes_hex=stored_checksum,
                )
            )
        archive.finish()

    if records_total != archive_bytes - header_bytes:
        raise AttributionError(f"{archive_path}: archive byte accounting drift")
    source_payload = sum(item.source_tensor_bytes for item in metadata)
    inferred_source_file = prefix_len + source_payload
    if source is not None and source.file_bytes != inferred_source_file:
        raise AttributionError(f"{archive_path}: source file length disagrees")
    return (
        ArchiveSummary(
            archive_path=str(archive_path),
            source_path=str(source.path) if source else None,
            source_prefix_matched=True if source else None,
            tensor_count=tensor_count,
            source_file_bytes=inferred_source_file,
            source_prefix_bytes=prefix_len,
            source_payload_bytes=source_payload,
            archive_file_bytes=archive_bytes,
            archive_header_bytes=header_bytes,
            archive_record_bytes=records_total,
            embedded_prefix_sha256=prefix_sha,
            source_over_archive_ratio=(
                inferred_source_file / archive_bytes if archive_bytes else math.inf
            ),
            archive_percent_of_source=(
                100.0 * archive_bytes / inferred_source_file
                if inferred_source_file
                else math.inf
            ),
            saved_bytes=inferred_source_file - archive_bytes,
        ),
        rows,
    )


def _manifest_sources(directory: Path) -> list[Path] | None:
    manifest_path = directory / "download-manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AttributionError(f"{manifest_path}: cannot read manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise AttributionError(f"{manifest_path}: manifest root is not an object")
    weights = manifest.get("weights")
    if not isinstance(weights, list):
        raise AttributionError(f"{manifest_path}: weights is not a list")
    result: list[Path] = []
    for item in weights:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise AttributionError(f"{manifest_path}: invalid weight entry")
        candidate = (directory / item["path"]).resolve()
        try:
            candidate.relative_to(directory.resolve())
        except ValueError as exc:
            raise AttributionError(f"{manifest_path}: weight escapes directory") from exc
        if candidate.suffix == ".safetensors":
            result.append(candidate)
    return result


def discover_files(
    inputs: Sequence[Path],
    *,
    kind: str,
) -> list[Path]:
    suffix = ".brv" if kind == "archive" else ".safetensors"
    discovered: list[Path] = []
    for raw in inputs:
        path = raw.expanduser()
        if path.is_file():
            if path.suffix != suffix:
                raise AttributionError(f"{path}: expected a {suffix} file")
            discovered.append(path.resolve())
            continue
        if not path.is_dir():
            raise AttributionError(f"{path}: path does not exist")
        if kind == "source":
            manifest_paths = _manifest_sources(path)
            candidates = (
                manifest_paths
                if manifest_paths is not None
                else sorted(path.rglob(f"*{suffix}"))
            )
        else:
            candidates = sorted(path.rglob(f"*{suffix}"))
        discovered.extend(candidate.resolve() for candidate in candidates)
    unique = list(dict.fromkeys(discovered))
    if not unique:
        raise AttributionError(f"no {kind} {suffix} files discovered")
    for path in unique:
        if not path.is_file():
            raise AttributionError(f"{path}: discovered file is missing")
    return unique


def load_sources(
    source_paths: Sequence[Path],
) -> tuple[list[SourceDescriptor], dict[str, list[SourceDescriptor]]]:
    descriptors = [read_source_descriptor(path) for path in source_paths]
    by_prefix: dict[str, list[SourceDescriptor]] = defaultdict(list)
    for descriptor in descriptors:
        by_prefix[descriptor.prefix_sha256].append(descriptor)
    return descriptors, by_prefix


def aggregate_rows(
    rows: Sequence[TensorRow],
    key,
) -> list[dict[str, Any]]:
    groups: dict[str, list[TensorRow]] = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    result: list[dict[str, Any]] = []
    for group in sorted(groups):
        members = groups[group]
        source = sum(row.source_tensor_bytes for row in members)
        output = sum(row.archive_record_bytes for row in members)
        fallback = [row for row in members if row.program_selection == "literal_fallback"]
        result.append(
            {
                "group": group,
                "tensor_count": len(members),
                "source_tensor_bytes": source,
                "archive_record_bytes": output,
                "saved_bytes_vs_record": source - output,
                "source_over_record_ratio": source / output if output else math.inf,
                "record_percent_of_source": (
                    100.0 * output / source if source else None
                ),
                "program_bytes": sum(row.program_bytes for row in members),
                "record_framing_bytes": sum(
                    row.record_framing_bytes for row in members
                ),
                "literal_fallback_tensors": len(fallback),
                "literal_fallback_source_bytes": sum(
                    row.source_tensor_bytes for row in fallback
                ),
                "literal_fallback_record_bytes": sum(
                    row.archive_record_bytes for row in fallback
                ),
            }
        )
    return result


def operator_rows(rows: Sequence[TensorRow]) -> list[dict[str, Any]]:
    operators = sorted(
        {
            operator
            for row in rows
            for operator in row.operator_counts
        }
    )
    total_program = sum(row.program_bytes for row in rows)
    result = []
    for operator in operators:
        containing = [row for row in rows if operator in row.operator_counts]
        roots = [row for row in rows if row.root_operator == operator]
        exclusive = sum(
            row.operator_exclusive_bytes.get(operator, 0) for row in rows
        )
        root_source = sum(row.source_tensor_bytes for row in roots)
        root_output = sum(row.archive_record_bytes for row in roots)
        result.append(
            {
                "operator": operator,
                "node_count": sum(row.operator_counts.get(operator, 0) for row in rows),
                "tensors_containing": len(containing),
                "exclusive_program_bytes": exclusive,
                "exclusive_share_of_all_program_bytes_percent": (
                    100.0 * exclusive / total_program if total_program else 0.0
                ),
                "root_selected_tensors": len(roots),
                "root_selected_source_bytes": root_source,
                "root_selected_record_bytes": root_output,
                "root_selected_source_over_record_ratio": (
                    root_source / root_output if root_output else None
                ),
            }
        )
    return result


def literal_codec_rows(rows: Sequence[TensorRow]) -> list[dict[str, Any]]:
    codecs = sorted(
        {
            codec
            for row in rows
            for codec in row.literal_codec_counts
        }
    )
    total_body = sum(row.literal_body_bytes for row in rows)
    result = []
    for codec in codecs:
        body = sum(row.literal_codec_body_bytes.get(codec, 0) for row in rows)
        result.append(
            {
                "codec": codec,
                "literal_count": sum(
                    row.literal_codec_counts.get(codec, 0) for row in rows
                ),
                "tensors_containing": sum(
                    codec in row.literal_codec_counts for row in rows
                ),
                "body_bytes": body,
                "body_share_percent": (
                    100.0 * body / total_body if total_body else 0.0
                ),
                "semantic_storage_bytes": sum(
                    row.literal_codec_semantic_bytes.get(codec, 0) for row in rows
                ),
                "payload_bytes": sum(
                    row.literal_codec_payload_bytes.get(codec, 0) for row in rows
                ),
            }
        )
    return result


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.12g}"
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    return value


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _fmt_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    amount = float(value)
    for unit in units:
        if abs(amount) < 1024.0 or unit == units[-1]:
            return f"{amount:.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def _fmt_ratio(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}×"


def _markdown_group_table(rows: Sequence[dict[str, Any]]) -> list[str]:
    lines = [
        "| group | tensors | source payload | archive records | ratio | saved | root-Lit |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {group} | {tensors} | {source} | {output} | {ratio} | "
            "{saved} | {fallback} |".format(
                group=row["group"],
                tensors=row["tensor_count"],
                source=_fmt_bytes(row["source_tensor_bytes"]),
                output=_fmt_bytes(row["archive_record_bytes"]),
                ratio=_fmt_ratio(row["source_over_record_ratio"]),
                saved=_fmt_bytes(row["saved_bytes_vs_record"]),
                fallback=row["literal_fallback_tensors"],
            )
        )
    return lines


def build_report(
    archives: Sequence[ArchiveSummary],
    rows: Sequence[TensorRow],
    grouped: dict[str, list[dict[str, Any]]],
    operators: Sequence[dict[str, Any]],
    codecs: Sequence[dict[str, Any]],
    sources_supplied: bool,
) -> str:
    source_bytes = sum(item.source_file_bytes for item in archives)
    archive_bytes = sum(item.archive_file_bytes for item in archives)
    ratio = source_bytes / archive_bytes if archive_bytes else math.inf
    source_payload = sum(row.source_tensor_bytes for row in rows)
    record_bytes = sum(row.archive_record_bytes for row in rows)
    source_headers = sum(item.source_prefix_bytes for item in archives)
    archive_headers = sum(item.archive_header_bytes for item in archives)
    fallback = [row for row in rows if row.program_selection == "literal_fallback"]
    lines = [
        "# Brevis tensor/operator attribution",
        "",
        "## Exact whole-archive accounting",
        "",
        f"- Archives: {len(archives)}; tensors: {len(rows)}.",
        f"- Source: {_fmt_bytes(source_bytes)}; archive: {_fmt_bytes(archive_bytes)}; "
        f"ratio: {_fmt_ratio(ratio)}; saved: {_fmt_bytes(source_bytes - archive_bytes)}.",
        f"- Tensor payload: {_fmt_bytes(source_payload)}; tensor records: "
        f"{_fmt_bytes(record_bytes)}; ratio: "
        f"{_fmt_ratio(source_payload / record_bytes if record_bytes else math.inf)}.",
        f"- Source prefixes: {_fmt_bytes(source_headers)}; BRTA headers including "
        f"those prefixes: {_fmt_bytes(archive_headers)}; outer framing overhead: "
        f"{_fmt_bytes(archive_headers - source_headers)}.",
        f"- Root `Lit` fallback: {len(fallback)}/{len(rows)} tensors, "
        f"{_fmt_bytes(sum(row.source_tensor_bytes for row in fallback))} source bytes.",
        (
            "- Every embedded safetensors prefix matched one supplied source file exactly."
            if sources_supplied
            else "- No source files were supplied; source sizes and tensor metadata are "
            "derived from the exact safetensors prefixes embedded in BRTA."
        ),
        "",
        "Role and dtype rows compare exact tensor payload bytes with exact complete "
        "record bytes. They exclude the one global BRTA header; whole-archive accounting "
        "above includes it.",
        "",
        "## By tensor role",
        "",
        *_markdown_group_table(grouped["role"]),
        "",
        "## By dtype",
        "",
        *_markdown_group_table(grouped["dtype"]),
        "",
        "## By fallback versus synthesized selection",
        "",
        *_markdown_group_table(grouped["selection"]),
        "",
        "## By selected root program",
        "",
        *_markdown_group_table(grouped["program"]),
        "",
        "## Program wire composition",
        "",
        "| operator | nodes | tensors | exclusive wire bytes | share of all program bytes | root selections |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in operators:
        lines.append(
            f"| {row['operator']} | {row['node_count']} | "
            f"{row['tensors_containing']} | "
            f"{_fmt_bytes(row['exclusive_program_bytes'])} | "
            f"{row['exclusive_share_of_all_program_bytes_percent']:.3f}% | "
            f"{row['root_selected_tensors']} |"
        )
    lines.extend(
        [
            "",
            "Exclusive wire bytes are the bytes syntactically owned by a node "
            "(its tag/parameters and, for `Lit`, its encoded body), excluding child "
            "subtrees. They sum with the five-byte BRPG header per tensor to exact "
            "program bytes. They are storage composition, not causal savings.",
            "",
            "## Literal physical codecs",
            "",
            "| codec | literals | tensors | body bytes | body share | semantic leaf storage | payload bytes |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in codecs:
        lines.append(
            f"| {row['codec']} | {row['literal_count']} | "
            f"{row['tensors_containing']} | {_fmt_bytes(row['body_bytes'])} | "
            f"{row['body_share_percent']:.3f}% | "
            f"{_fmt_bytes(row['semantic_storage_bytes'])} | "
            f"{_fmt_bytes(row['payload_bytes'])} |"
        )
    lines.extend(
        [
            "",
            "## Scope and non-identifiable quantities",
            "",
            "- Exact here: BRTA/BRPG versions and framing, embedded safetensors "
            "metadata, record name/dtype/shape binding, static stream bits/length, "
            "tensor source bytes, complete record/program/framing bytes, root fallback "
            "versus synthesized selection, node counts, node-exclusive wire bytes, and "
            "literal codec/body framing sizes.",
            "- Not performed here: program execution; decoder output/literal/execution-"
            "work resource-limit validation; complete entropy table, payload, or padding "
            "validation; XXH3 checksum verification; or comparison of tensor payload "
            "bytes with the source. Run `brevis verify ARCHIVE SOURCE` independently "
            "for those guarantees.",
            "- Not identifiable from BRTA alone: bytes saved *by an individual operator*. "
            "The archive persists only the winning program, not valid counterfactual "
            "program costs or the search trace. `operators.csv` therefore reports exact "
            "occupied wire bytes and root-program cohort results, never invented "
            "per-operator savings.",
            "- `literal_semantic_storage_bytes` can exceed tensor source bytes for "
            "decompositions such as bit planes. It describes leaf streams and must not "
            "be presented as an alternate model size.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze(
    archive_paths: Sequence[Path],
    source_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    sources, source_by_prefix = load_sources(source_paths)
    archives: list[ArchiveSummary] = []
    tensors: list[TensorRow] = []
    for path in archive_paths:
        summary, rows = parse_archive(path, source_by_prefix, bool(source_paths))
        archives.append(summary)
        tensors.extend(rows)

    grouped = {
        "role": aggregate_rows(tensors, lambda row: row.role),
        "dtype": aggregate_rows(tensors, lambda row: row.dtype),
        "selection": aggregate_rows(tensors, lambda row: row.program_selection),
        "program": aggregate_rows(
            tensors,
            lambda row: _program_class(row.program_selection, row.root_operator),
        ),
        "role_program": aggregate_rows(
            tensors,
            lambda row: (
                f"{row.role}/{_program_class(row.program_selection, row.root_operator)}"
            ),
        ),
    }
    operators = operator_rows(tensors)
    codecs = literal_codec_rows(tensors)
    source_bytes = sum(item.source_file_bytes for item in archives)
    archive_bytes = sum(item.archive_file_bytes for item in archives)
    return {
        "schema_version": 1,
        "format": {"archive": "BRTA v3", "program": "BRPG v2"},
        "validation_scope": {
            "source_prefix_match": bool(source_paths),
            "embedded_metadata_cross_binding": True,
            "static_stream_type_check": True,
            "literal_body_framing_check": True,
            "literal_payload_and_table_validation": False,
            "program_execution": False,
            "checksum_verification": False,
            "source_tensor_content_verification": False,
        },
        "totals": {
            "archive_count": len(archives),
            "tensor_count": len(tensors),
            "source_file_bytes": source_bytes,
            "archive_file_bytes": archive_bytes,
            "saved_bytes": source_bytes - archive_bytes,
            "source_over_archive_ratio": (
                source_bytes / archive_bytes if archive_bytes else math.inf
            ),
            "source_payload_bytes": sum(
                row.source_tensor_bytes for row in tensors
            ),
            "archive_record_bytes": sum(
                row.archive_record_bytes for row in tensors
            ),
            "source_prefix_bytes": sum(
                item.source_prefix_bytes for item in archives
            ),
            "archive_header_bytes": sum(
                item.archive_header_bytes for item in archives
            ),
        },
        "archives": [asdict(item) for item in archives],
        "tensors": [asdict(item) for item in tensors],
        "groups": grouped,
        "operators": operators,
        "literal_codecs": codecs,
        "_objects": {
            "archive_summaries": archives,
            "tensor_rows": tensors,
            "source_descriptors": sources,
        },
    }


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    objects = report["_objects"]
    archives: list[ArchiveSummary] = objects["archive_summaries"]
    tensors: list[TensorRow] = objects["tensor_rows"]
    serializable = {key: value for key, value in report.items() if key != "_objects"}
    (output_dir / "attribution.json").write_text(
        json.dumps(serializable, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(
        output_dir / "archives.csv",
        [asdict(item) for item in archives],
    )
    write_csv(
        output_dir / "tensors.csv",
        [item.csv_dict() for item in tensors],
    )
    for name, rows in report["groups"].items():
        write_csv(output_dir / f"by-{name.replace('_', '-')}.csv", rows)
    write_csv(output_dir / "operators.csv", report["operators"])
    write_csv(output_dir / "literal-codecs.csv", report["literal_codecs"])
    markdown = build_report(
        archives,
        tensors,
        report["groups"],
        report["operators"],
        report["literal_codecs"],
        bool(report["validation_scope"]["source_prefix_match"]),
    )
    (output_dir / "report.md").write_text(markdown, encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read BRTA v3 archives without decoding tensor payloads and emit exact "
            "tensor/role/dtype/program wire attribution."
        )
    )
    parser.add_argument(
        "--archives",
        nargs="+",
        required=True,
        type=Path,
        help="BRTA .brv file(s) or directories recursively containing them",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        type=Path,
        default=(),
        help=(
            "original .safetensors file(s) or checkpoint directories; a "
            "download-manifest.json is honored when present"
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        archives = discover_files(args.archives, kind="archive")
        sources = (
            discover_files(args.sources, kind="source") if args.sources else []
        )
        report = analyze(archives, sources)
        write_outputs(report, args.output_dir)
    except (AttributionError, OSError) as exc:
        raise SystemExit(f"attribution failed: {exc}") from exc
    totals = report["totals"]
    print(
        f"analyzed {totals['tensor_count']} tensors from "
        f"{totals['archive_count']} archives: "
        f"{totals['source_over_archive_ratio']:.6f}x; "
        f"wrote {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
