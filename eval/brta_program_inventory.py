#!/usr/bin/env python3
"""Inventory BRTA-v1 program structure without executing its programs.

Run ``brevis verify ARCHIVE`` before treating an inventory as evidence. This
tool checks versioned BRTA/BRPG framing and inventories ASTs, but it does not
decode literal bodies, execute programs, or verify tensor checksums.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import pathlib
import sys
from typing import BinaryIO, Iterable


SCHEMA_ID = "brevis.brta-program-inventory"
SCHEMA_VERSION = 1
BRTA_MAGIC = b"BRTA"
BRTA_VERSION = 1
BRPG_MAGIC = b"BRPG"
BRPG_VERSION = 1
RECORD_TAG = 1
CHECKSUM_BYTES = 32

MAX_TENSORS = 1_000_000
MAX_PREFIX_BYTES = 64 * 1024 * 1024
MAX_RECORD_BYTES = 16 * 1024 * 1024 * 1024
MAX_NAME_BYTES = 1024 * 1024
MAX_DIMENSIONS = 1024
MAX_PROGRAM_BYTES = 16 * 1024 * 1024 * 1024
MAX_NODES = 1_000_000
MAX_DEPTH = 256
MIN_FRAMED_RECORD_BYTES = 38

DTYPES = {
    0x01: "f16",
    0x02: "bf16",
    0x03: "f32",
    0x04: "u8",
    0x05: "u16",
    0x06: "u32",
    0x07: "i8",
    0x08: "i16",
    0x09: "i32",
    0x0A: "f8_e4m3",
    0x0B: "f8_e5m2",
}
MAP_OPS = {
    0x01: ("map.xor", "u32"),
    0x02: ("map.add_mod", "u32"),
    0x03: ("map.zigzag", None),
    0x04: ("map.gray", None),
    0x05: ("map.rotate_left", "u8"),
    0x06: ("map.bit_reverse", None),
}
SCAN_OPS = {
    0x01: "scan.xor",
    0x02: "scan.add_mod",
}
MERGE_OPS = {
    0x01: ("merge.fields", "u8"),
    0x02: ("merge.float_fields", "dtype"),
    0x03: ("merge.bit_planes", None),
    0x04: ("merge.byte_planes", None),
}


class InventoryError(ValueError):
    """The archive is not a structurally valid BRTA-v1/BRPG-v1 stream."""


class _Reader:
    def __init__(self, stream: BinaryIO, end: int, context: str):
        self.stream = stream
        self.end = end
        self.context = context

    @property
    def position(self) -> int:
        return self.stream.tell()

    @property
    def remaining(self) -> int:
        return self.end - self.position

    def take(self, count: int) -> bytes:
        if count < 0 or count > self.remaining:
            raise InventoryError(
                f"truncated {self.context} at byte {self.position}: "
                f"need {count}, have {max(self.remaining, 0)}"
            )
        data = self.stream.read(count)
        if len(data) != count:
            raise InventoryError(
                f"truncated {self.context} at byte {self.position - len(data)}"
            )
        return data

    def skip(self, count: int) -> None:
        if count < 0 or count > self.remaining:
            raise InventoryError(
                f"truncated {self.context} at byte {self.position}: "
                f"need {count}, have {max(self.remaining, 0)}"
            )
        target = self.position + count
        self.stream.seek(count, os.SEEK_CUR)
        if self.position != target:
            raise InventoryError(f"truncated {self.context} at byte {self.position}")

    def byte(self) -> int:
        return self.take(1)[0]

    def uleb128(self) -> int:
        value = 0
        for index in range(10):
            byte = self.byte()
            payload = byte & 0x7F
            if index == 9 and payload > 1:
                raise InventoryError(
                    f"ULEB128 integer overflow in {self.context} "
                    f"at byte {self.position - 1}"
                )
            value |= payload << (index * 7)
            if byte & 0x80 == 0:
                if _uleb128_size(value) != index + 1:
                    raise InventoryError(
                        f"overlong ULEB128 in {self.context} "
                        f"at byte {self.position - index - 1}"
                    )
                return value
        raise InventoryError(
            f"ULEB128 integer overflow in {self.context} at byte {self.position - 10}"
        )


def _uleb128_size(value: int) -> int:
    size = 1
    while value >= 0x80:
        value >>= 7
        size += 1
    return size


def _bounded(value: int, maximum: int, what: str, reader: _Reader) -> int:
    if value > maximum:
        raise InventoryError(
            f"{what} {value} exceeds {maximum} in {reader.context} "
            f"at byte {reader.position}"
        )
    return value


def _read_u32(reader: _Reader, what: str) -> int:
    return _bounded(reader.uleb128(), 0xFFFFFFFF, what, reader)


def _read_u8_uleb(reader: _Reader, what: str) -> int:
    return _bounded(reader.uleb128(), 0xFF, what, reader)


def _read_dtype(reader: _Reader, what: str) -> str:
    wire_id = reader.byte()
    try:
        return DTYPES[wire_id]
    except KeyError as exc:
        raise InventoryError(
            f"unknown {what} tag 0x{wire_id:02x} in {reader.context} "
            f"at byte {reader.position - 1}"
        ) from exc


@dataclasses.dataclass(frozen=True)
class _Node:
    kind: str
    node_count: int
    ast_depth: int
    semantic_op_count: int
    semantic_op_depth: int
    semantic_preorder: tuple[str, ...]
    node_kinds: collections.Counter[str]
    root_to_terminal: collections.Counter[tuple[str, ...]]


class _ProgramParser:
    def __init__(self, reader: _Reader):
        self.reader = reader
        self.nodes = 0

    def parse(self) -> _Node:
        if self.reader.take(len(BRPG_MAGIC)) != BRPG_MAGIC:
            raise InventoryError(
                f"bad BRPG magic in {self.reader.context} "
                f"at byte {self.reader.position - len(BRPG_MAGIC)}"
            )
        version = self.reader.byte()
        if version != BRPG_VERSION:
            raise InventoryError(
                f"unsupported BRPG version {version} in {self.reader.context}"
            )
        root = self._node(1)
        if self.reader.remaining != 0:
            raise InventoryError(
                f"trailing bytes in {self.reader.context}: {self.reader.remaining}"
            )
        return root

    def _claim_node(self, depth: int) -> None:
        if depth > MAX_DEPTH:
            raise InventoryError(
                f"AST depth exceeds {MAX_DEPTH} in {self.reader.context}"
            )
        if self.nodes >= MAX_NODES:
            raise InventoryError(
                f"node count exceeds {MAX_NODES} in {self.reader.context}"
            )
        self.nodes += 1

    def _children(self, count: int, depth: int) -> list[_Node]:
        if count > MAX_NODES - self.nodes:
            raise InventoryError(
                f"child count exceeds remaining node limit in {self.reader.context}"
            )
        if count > self.reader.remaining:
            raise InventoryError(
                f"truncated child list in {self.reader.context} "
                f"at byte {self.reader.position}"
            )
        return [self._node(depth + 1) for _ in range(count)]

    def _node(self, depth: int) -> _Node:
        self._claim_node(depth)
        node_tag = self.reader.byte()

        if node_tag == 0x01:
            bits = self.reader.byte()
            if not 1 <= bits <= 32:
                raise InventoryError(
                    f"invalid literal width {bits} in {self.reader.context}"
                )
            self.reader.uleb128()
            body_len = self.reader.uleb128()
            self.reader.skip(body_len)
            return _terminal("literal")

        if node_tag == 0x02:
            bits = self.reader.byte()
            count = self.reader.uleb128()
            word = _read_u32(self.reader, "constant word")
            if not 1 <= bits <= 32:
                raise InventoryError(
                    f"invalid constant width {bits} in {self.reader.context}"
                )
            if count == 0:
                raise InventoryError(
                    f"invalid zero-length constant in {self.reader.context}"
                )
            mask = 0xFFFFFFFF if bits == 32 else (1 << bits) - 1
            if word & ~mask:
                raise InventoryError(
                    f"invalid constant word for width {bits} "
                    f"in {self.reader.context}"
                )
            return _terminal("constant")

        if node_tag == 0x03:
            child_count = self.reader.uleb128()
            if child_count < 2:
                raise InventoryError(
                    f"invalid concat arity {child_count} in {self.reader.context}"
                )
            return _operator("concat", self._children(child_count, depth))

        if node_tag == 0x04:
            times = _read_u32(self.reader, "repeat count")
            if times < 2:
                raise InventoryError(
                    f"invalid repeat count {times} in {self.reader.context}"
                )
            return _operator("repeat", self._children(1, depth))

        if node_tag == 0x05:
            op_tag = self.reader.byte()
            try:
                kind, parameter = MAP_OPS[op_tag]
            except KeyError as exc:
                raise InventoryError(
                    f"unknown map tag 0x{op_tag:02x} in {self.reader.context} "
                    f"at byte {self.reader.position - 1}"
                ) from exc
            if parameter == "u32":
                _read_u32(self.reader, f"{kind} parameter")
            elif parameter == "u8":
                _read_u8_uleb(self.reader, f"{kind} parameter")
            return _operator(kind, self._children(1, depth))

        if node_tag == 0x06:
            op_tag = self.reader.byte()
            try:
                kind = SCAN_OPS[op_tag]
            except KeyError as exc:
                raise InventoryError(
                    f"unknown scan tag 0x{op_tag:02x} in {self.reader.context} "
                    f"at byte {self.reader.position - 1}"
                ) from exc
            _read_u32(self.reader, f"{kind} initial value")
            return _operator(kind, self._children(1, depth))

        if node_tag == 0x07:
            op_tag = self.reader.byte()
            try:
                kind, parameter = MERGE_OPS[op_tag]
            except KeyError as exc:
                raise InventoryError(
                    f"unknown merge tag 0x{op_tag:02x} in {self.reader.context} "
                    f"at byte {self.reader.position - 1}"
                ) from exc
            if parameter == "u8":
                low_bits = _read_u8_uleb(self.reader, f"{kind} parameter")
                if not 1 <= low_bits <= 32:
                    raise InventoryError(
                        f"invalid {kind} parameter {low_bits} "
                        f"in {self.reader.context}"
                    )
            elif parameter == "dtype":
                dtype = _read_dtype(self.reader, "program dtype")
                if dtype not in {"f16", "bf16", "f32", "f8_e4m3", "f8_e5m2"}:
                    raise InventoryError(
                        f"invalid {kind} dtype {dtype} in {self.reader.context}"
                    )
            child_count = self.reader.uleb128()
            minimum, maximum = {
                "merge.fields": (2, 2),
                "merge.float_fields": (3, 3),
                "merge.bit_planes": (2, 32),
                "merge.byte_planes": (2, 4),
            }[kind]
            if not minimum <= child_count <= maximum:
                raise InventoryError(
                    f"invalid {kind} arity {child_count}; expected "
                    f"{minimum}..{maximum} in {self.reader.context}"
                )
            return _operator(kind, self._children(child_count, depth))

        raise InventoryError(
            f"unknown node tag 0x{node_tag:02x} in {self.reader.context} "
            f"at byte {self.reader.position - 1}"
        )


def _terminal(kind: str) -> _Node:
    return _Node(
        kind=kind,
        node_count=1,
        ast_depth=1,
        semantic_op_count=0,
        semantic_op_depth=0,
        semantic_preorder=(),
        node_kinds=collections.Counter((kind,)),
        root_to_terminal=collections.Counter({(kind,): 1}),
    )


def _operator(kind: str, children: Iterable[_Node]) -> _Node:
    child_nodes = tuple(children)
    node_kinds: collections.Counter[str] = collections.Counter((kind,))
    root_to_terminal: collections.Counter[tuple[str, ...]] = collections.Counter()
    preorder: list[str] = [kind]
    for child in child_nodes:
        node_kinds.update(child.node_kinds)
        preorder.extend(child.semantic_preorder)
        root_to_terminal.update(
            {(kind, *sequence): count for sequence, count in child.root_to_terminal.items()}
        )
    if not root_to_terminal:
        root_to_terminal[(kind,)] = 1
    return _Node(
        kind=kind,
        node_count=1 + sum(child.node_count for child in child_nodes),
        ast_depth=1 + max((child.ast_depth for child in child_nodes), default=0),
        semantic_op_count=1 + sum(
            child.semantic_op_count for child in child_nodes
        ),
        semantic_op_depth=1 + max(
            (child.semantic_op_depth for child in child_nodes), default=0
        ),
        semantic_preorder=tuple(preorder),
        node_kinds=node_kinds,
        root_to_terminal=root_to_terminal,
    )


def _sequence_text(sequence: tuple[str, ...]) -> str:
    return " > ".join(sequence) if sequence else "none"


def _string_counter(counter: collections.Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _integer_counter(counter: collections.Counter[int]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter)}


def _sequence_counter(
    counter: collections.Counter[tuple[str, ...]],
) -> dict[str, int]:
    rendered = collections.Counter(
        {_sequence_text(sequence): count for sequence, count in counter.items()}
    )
    return _string_counter(rendered)


def _display_name(name: bytes) -> tuple[str, str]:
    try:
        return name.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        return name.hex(), "hex"


def _parse_tensor(reader: _Reader, index: int) -> dict[str, object]:
    body_len = _bounded(
        reader.uleb128(), MAX_RECORD_BYTES, "record body length", reader
    )
    if body_len > reader.remaining:
        raise InventoryError(
            f"truncated record {index} body at byte {reader.position}: "
            f"need {body_len}, have {reader.remaining}"
        )
    body_end = reader.position + body_len
    body = _Reader(reader.stream, body_end, f"record {index}")

    tag = body.byte()
    if tag != RECORD_TAG:
        raise InventoryError(
            f"unknown record tag 0x{tag:02x} in record {index} "
            f"at byte {body.position - 1}"
        )
    name_len = _bounded(body.uleb128(), MAX_NAME_BYTES, "tensor name length", body)
    name_bytes = body.take(name_len)
    name, name_encoding = _display_name(name_bytes)
    dtype = _read_dtype(body, "record dtype")
    dimension_count = _bounded(
        body.uleb128(), MAX_DIMENSIONS, "dimension count", body
    )
    if dimension_count > body.remaining:
        raise InventoryError(
            f"truncated shape in record {index} at byte {body.position}"
        )
    shape = [body.uleb128() for _ in range(dimension_count)]

    program_bytes = _bounded(
        body.uleb128(), MAX_PROGRAM_BYTES, "program length", body
    )
    if program_bytes > body.remaining:
        raise InventoryError(
            f"truncated program in record {index} at byte {body.position}: "
            f"need {program_bytes}, have {body.remaining}"
        )
    program_end = body.position + program_bytes
    program = _ProgramParser(
        _Reader(body.stream, program_end, f"record {index} BRPG program")
    ).parse()
    body.take(CHECKSUM_BYTES)
    if body.remaining != 0:
        raise InventoryError(
            f"trailing record data in record {index}: {body.remaining} bytes"
        )

    return {
        "index": index,
        "name": name,
        "name_encoding": name_encoding,
        "dtype": dtype,
        "shape": shape,
        "program_bytes": program_bytes,
        "root_kind": program.kind,
        "node_count": program.node_count,
        "ast_depth": program.ast_depth,
        "semantic_op_count": program.semantic_op_count,
        "semantic_op_depth": program.semantic_op_depth,
        "semantic_operators_preorder": list(program.semantic_preorder),
        "node_kinds": _string_counter(program.node_kinds),
        "root_to_terminal_sequences": _sequence_counter(
            program.root_to_terminal
        ),
    }


def inventory_stream(
    stream: BinaryIO,
    size_bytes: int,
    source: str = "<stream>",
) -> dict[str, object]:
    """Parse one seekable BRTA-v1 stream and return a JSON-ready inventory."""
    if size_bytes < 0:
        raise ValueError("size_bytes must be non-negative")
    stream.seek(0)
    archive = _Reader(stream, size_bytes, "BRTA archive")
    if archive.take(len(BRTA_MAGIC)) != BRTA_MAGIC:
        raise InventoryError("bad BRTA magic")
    version = archive.byte()
    if version != BRTA_VERSION:
        raise InventoryError(f"unsupported BRTA version {version}")

    tensor_count = _bounded(
        archive.uleb128(), MAX_TENSORS, "tensor count", archive
    )
    prefix_bytes = _bounded(
        archive.uleb128(), MAX_PREFIX_BYTES, "safetensors prefix length", archive
    )
    if prefix_bytes < 8:
        raise InventoryError("invalid safetensors prefix: fewer than 8 bytes")
    declared_json_bytes = int.from_bytes(archive.take(8), "little")
    if declared_json_bytes != prefix_bytes - 8:
        raise InventoryError(
            "invalid safetensors prefix: embedded JSON length does not match framing"
        )
    archive.skip(prefix_bytes - 8)
    if tensor_count > archive.remaining // MIN_FRAMED_RECORD_BYTES:
        raise InventoryError(
            f"truncated BRTA archive: {tensor_count} records cannot fit in "
            f"{archive.remaining} bytes"
        )

    tensors = [_parse_tensor(archive, index) for index in range(tensor_count)]
    if archive.remaining != 0:
        raise InventoryError(
            f"trailing bytes after declared records: {archive.remaining}"
        )

    root_kinds: collections.Counter[str] = collections.Counter()
    node_kinds: collections.Counter[str] = collections.Counter()
    operators: collections.Counter[str] = collections.Counter()
    node_counts: collections.Counter[int] = collections.Counter()
    ast_depths: collections.Counter[int] = collections.Counter()
    semantic_counts: collections.Counter[int] = collections.Counter()
    semantic_depths: collections.Counter[int] = collections.Counter()
    preorder_combinations: collections.Counter[tuple[str, ...]] = (
        collections.Counter()
    )
    path_sequences: collections.Counter[tuple[str, ...]] = collections.Counter()
    program_lengths: list[int] = []

    literal_only = 0
    zero_semantic_ops = 0
    single_semantic_op = 0
    at_least_two_semantic_ops = 0
    for tensor in tensors:
        root_kinds[tensor["root_kind"]] += 1
        node_counts[tensor["node_count"]] += 1
        ast_depths[tensor["ast_depth"]] += 1
        semantic_count = tensor["semantic_op_count"]
        semantic_counts[semantic_count] += 1
        semantic_depths[tensor["semantic_op_depth"]] += 1
        program_lengths.append(tensor["program_bytes"])
        node_kinds.update(tensor["node_kinds"])
        preorder = tuple(tensor["semantic_operators_preorder"])
        preorder_combinations[preorder] += 1
        operators.update(preorder)
        for sequence, count in tensor["root_to_terminal_sequences"].items():
            path_sequences[tuple(sequence.split(" > "))] += count

        if tensor["root_kind"] == "literal" and tensor["node_count"] == 1:
            literal_only += 1
        if semantic_count == 0:
            zero_semantic_ops += 1
        elif semantic_count == 1:
            single_semantic_op += 1
        else:
            at_least_two_semantic_ops += 1

    return {
        "schema": {"id": SCHEMA_ID, "version": SCHEMA_VERSION},
        "evidence_scope": {
            "structural_only": True,
            "semantic_verification_required": True,
            "verification_command": ["brevis", "verify", source],
            "not_checked": [
                "literal codec canonicality",
                "cross-node widths, lengths, and tensor type agreement",
                "program execution",
                "tensor checksums",
                "safetensors metadata agreement",
            ],
        },
        "metric_definitions": {
            "ast_depth": "root depth is one",
            "semantic_op": (
                "concat, repeat, map.*, scan.*, or merge.*; literal and "
                "constant are terminals"
            ),
            "semantic_op_depth": (
                "maximum semantic operators on any root-to-terminal path"
            ),
            "semantic_preorder_combinations": (
                "one semantic-operator preorder sequence per tensor"
            ),
            "root_to_terminal_sequences": (
                "semantic operators followed by the terminal; counts include "
                "branch multiplicity"
            ),
        },
        "archive": {
            "path": source,
            "size_bytes": size_bytes,
            "format": "BRTA-v1",
            "tensor_count": tensor_count,
            "safetensors_prefix_bytes": prefix_bytes,
        },
        "summary": {
            "tensor_count": tensor_count,
            "program_bytes": {
                "total": sum(program_lengths),
                "minimum": min(program_lengths, default=0),
                "maximum": max(program_lengths, default=0),
            },
            "root_kinds": _string_counter(root_kinds),
            "node_kinds": _string_counter(node_kinds),
            "semantic_operator_counts": _string_counter(operators),
            "node_count_total": sum(tensor["node_count"] for tensor in tensors),
            "node_count_distribution": _integer_counter(node_counts),
            "ast_depth_distribution": _integer_counter(ast_depths),
            "semantic_op_count_total": sum(
                tensor["semantic_op_count"] for tensor in tensors
            ),
            "semantic_op_count_distribution": _integer_counter(semantic_counts),
            "semantic_op_depth_distribution": _integer_counter(semantic_depths),
            "classifications": {
                "literal_only": literal_only,
                "zero_semantic_ops": zero_semantic_ops,
                "single_semantic_op": single_semantic_op,
                "at_least_two_semantic_ops": at_least_two_semantic_ops,
            },
            "semantic_preorder_combinations": _sequence_counter(
                preorder_combinations
            ),
            "root_to_terminal_sequences": _sequence_counter(path_sequences),
        },
        "tensors": tensors,
    }


def inventory_archive(path: pathlib.Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    size_bytes = path.stat().st_size
    with path.open("rb") as stream:
        return inventory_stream(stream, size_bytes, str(path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory BRTA-v1/BRPG-v1 AST structure. Run `brevis verify` "
            "separately before relying on the result."
        )
    )
    parser.add_argument("archive", type=pathlib.Path)
    parser.add_argument(
        "--compact",
        action="store_true",
        help="emit compact JSON instead of indented JSON",
    )
    arguments = parser.parse_args(argv)
    try:
        result = inventory_archive(arguments.archive)
    except (InventoryError, OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    json.dump(
        result,
        sys.stdout,
        ensure_ascii=False,
        indent=None if arguments.compact else 2,
        sort_keys=True,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
