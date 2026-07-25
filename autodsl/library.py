"""The macro-library model shared by the miner, the proposer, and the gates.

A macro is a named subtree of *existing* reversible operators whose leaves are
holes or terminals. It carries no data, so it cannot smuggle payload into the
grammar, and the engine expands it into primitives before serializing, so an
archive never depends on the library that produced it.

This module deliberately does not re-declare the DSL. Operator names, arities,
and which operators may appear in a body all come from `brevis config`, so the
engine stays the single source of truth. Legality guards (widths, depth, float
roots) are *not* mirrored either: whether a body fits a particular stream is a
question only the engine can answer, and it answers it by declining to apply
the macro.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from typing import Any, Iterable

SCHEMA = "brevis.macro-library.v1"
HOLE = {"op": "hole"}
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,47}$")


class LibraryError(ValueError):
    """A macro or library that the engine would reject, caught earlier."""


@dataclasses.dataclass(frozen=True)
class Operator:
    name: str
    terminal: bool
    arity: int | None
    usable_in_macro_body: bool


@dataclasses.dataclass(frozen=True)
class OperatorTable:
    """The engine's operator set, read from `brevis config`."""

    by_name: dict[str, Operator]
    max_macros: int
    max_body_nodes: int

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "OperatorTable":
        if config.get("macro_library_schema") != SCHEMA:
            raise LibraryError(
                f"engine speaks {config.get('macro_library_schema')!r}, "
                f"this tool speaks {SCHEMA!r}"
            )
        limits = config["macro_library_limits"]
        return cls(
            by_name={
                o["name"]: Operator(
                    name=o["name"],
                    terminal=o["terminal"],
                    arity=o["arity"],
                    usable_in_macro_body=o["usable_in_macro_body"],
                )
                for o in config["operators"]
            },
            max_macros=limits["max_macros"],
            max_body_nodes=limits["max_body_nodes"],
        )

    def body_operators(self) -> list[str]:
        return sorted(o.name for o in self.by_name.values() if o.usable_in_macro_body)


# ==================== bodies ====================


def is_hole(body: Any) -> bool:
    return isinstance(body, dict) and body.get("op") == "hole"


def node(op: str, children: Iterable[Any] = (), params: int | str = "auto") -> dict:
    return {"op": op, "params": params, "children": list(children)}


def render(body: Any) -> str:
    """`op(child,child)`, matching how the engine renders a program."""
    if is_hole(body):
        return "?"
    children = body.get("children") or []
    if not children:
        return body["op"]
    return f"{body['op']}({','.join(render(c) for c in children)})"


def walk(body: Any):
    """Every operator node in the body, pre-order. Holes are not yielded."""
    if is_hole(body):
        return
    yield body
    for child in body.get("children") or []:
        yield from walk(child)


@dataclasses.dataclass(frozen=True)
class BodyStats:
    nodes: int
    holes: int
    terminals: int
    transform_depth: int
    operators: frozenset[str]


def body_stats(body: Any, table: OperatorTable) -> BodyStats:
    if is_hole(body):
        return BodyStats(0, 1, 0, 0, frozenset())
    op = table.by_name[body["op"]]
    if op.terminal:
        return BodyStats(1, 0, 1, 0, frozenset({op.name}))
    subs = [body_stats(c, table) for c in body.get("children") or []]
    return BodyStats(
        nodes=1 + sum(s.nodes for s in subs),
        holes=sum(s.holes for s in subs),
        terminals=sum(s.terminals for s in subs),
        transform_depth=1 + max((s.transform_depth for s in subs), default=0),
        operators=frozenset({op.name}).union(*[s.operators for s in subs]) if subs
        else frozenset({op.name}),
    )


def validate_body(body: Any, table: OperatorTable, *, path: str = "body") -> None:
    if not isinstance(body, dict):
        raise LibraryError(f"{path}: expected an object, got {type(body).__name__}")
    op_name = body.get("op")
    if not isinstance(op_name, str):
        raise LibraryError(f"{path}: missing string field 'op'")
    if op_name == "hole":
        if body.get("children"):
            raise LibraryError(f"{path}: a hole takes no children")
        return
    op = table.by_name.get(op_name)
    if op is None:
        raise LibraryError(
            f"{path}: unknown operator {op_name!r}; "
            f"known: {', '.join(table.body_operators())}"
        )
    if not op.usable_in_macro_body:
        raise LibraryError(
            f"{path}: {op_name!r} has width-dependent arity and cannot appear "
            "in a macro body"
        )
    params = body.get("params", "auto")
    if params != "auto" and not (isinstance(params, int) and 0 <= params <= 0xFFFFFFFF):
        raise LibraryError(f"{path}: params must be \"auto\" or a u32, got {params!r}")
    children = body.get("children") or []
    if not isinstance(children, list):
        raise LibraryError(f"{path}: 'children' must be a list")
    if len(children) != op.arity:
        raise LibraryError(
            f"{path}: {op_name!r} takes {op.arity} children, got {len(children)}"
        )
    for i, child in enumerate(children):
        validate_body(child, table, path=f"{path}.children[{i}]")


# ==================== macros ====================


@dataclasses.dataclass(frozen=True)
class Macro:
    name: str
    body: dict
    why: str = ""
    origin: str = "unknown"

    def to_json(self) -> dict:
        out = {"name": self.name, "body": self.body}
        if self.why:
            out["why"] = self.why
        if self.origin:
            out["origin"] = self.origin
        return out

    @classmethod
    def from_json(cls, raw: dict) -> "Macro":
        return cls(
            name=raw["name"],
            body=raw["body"],
            why=raw.get("why", ""),
            origin=raw.get("origin", "unknown"),
        )

    def shape(self) -> str:
        return render(self.body)


def validate_macro(macro: Macro, table: OperatorTable) -> BodyStats:
    if not NAME_RE.match(macro.name):
        raise LibraryError(
            f"macro name {macro.name!r} must be lower_snake_case, <=48 chars"
        )
    validate_body(macro.body, table)
    if is_hole(macro.body):
        raise LibraryError(f"{macro.name}: a bare hole is the identity")
    stats = body_stats(macro.body, table)
    if stats.nodes > table.max_body_nodes:
        raise LibraryError(
            f"{macro.name}: {stats.nodes} body nodes exceeds the engine limit "
            f"of {table.max_body_nodes}"
        )
    if stats.holes == 0 and stats.terminals == 0:
        raise LibraryError(f"{macro.name}: no holes and no terminals; unfinishable")
    return stats


# ==================== libraries ====================


@dataclasses.dataclass(frozen=True)
class Library:
    macros: tuple[Macro, ...] = ()
    provenance: dict = dataclasses.field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "schema": SCHEMA,
            "provenance": self.provenance,
            "macros": [m.to_json() for m in self.macros],
        }

    @classmethod
    def from_json(cls, raw: dict) -> "Library":
        if raw.get("schema") != SCHEMA:
            raise LibraryError(f"expected schema {SCHEMA!r}, got {raw.get('schema')!r}")
        return cls(
            macros=tuple(Macro.from_json(m) for m in raw.get("macros", [])),
            provenance=raw.get("provenance", {}),
        )

    def canonical_bytes(self) -> bytes:
        """Stable serialization for hashing: only what the engine reads.

        Rationale text and provenance are excluded so that re-wording a
        justification does not look like a different library.
        """
        payload = {
            "schema": SCHEMA,
            "macros": [
                {"name": m.name, "body": _canonical_body(m.body)} for m in self.macros
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def names(self) -> set[str]:
        return {m.name for m in self.macros}

    def shapes(self) -> set[str]:
        return {m.shape() for m in self.macros}

    def with_macro(self, macro: Macro) -> "Library":
        return dataclasses.replace(self, macros=self.macros + (macro,))

    def without(self, name: str) -> "Library":
        return dataclasses.replace(
            self, macros=tuple(m for m in self.macros if m.name != name)
        )

    def validate(self, table: OperatorTable) -> None:
        if len(self.macros) > table.max_macros:
            raise LibraryError(
                f"{len(self.macros)} macros exceeds the engine limit of "
                f"{table.max_macros}"
            )
        seen_names: set[str] = set()
        seen_shapes: dict[str, str] = {}
        for macro in self.macros:
            validate_macro(macro, table)
            if macro.name in seen_names:
                raise LibraryError(f"duplicate macro name {macro.name!r}")
            seen_names.add(macro.name)
            shape = macro.shape()
            if shape in seen_shapes:
                raise LibraryError(
                    f"{macro.name!r} has the same body as {seen_shapes[shape]!r}: {shape}"
                )
            seen_shapes[shape] = macro.name

    def write(self, path) -> None:
        path.write_text(json.dumps(self.to_json(), indent=2, sort_keys=False) + "\n")

    @classmethod
    def read(cls, path) -> "Library":
        return cls.from_json(json.loads(path.read_text()))


def _canonical_body(body: Any) -> dict:
    if is_hole(body):
        return {"op": "hole"}
    return {
        "op": body["op"],
        "params": body.get("params", "auto"),
        "children": [_canonical_body(c) for c in body.get("children") or []],
    }


EMPTY = Library()
