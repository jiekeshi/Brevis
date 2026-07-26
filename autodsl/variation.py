"""Ways to make a new candidate library out of the ones already measured.

A library, not a macro, is the unit that gets a fitness. That is the whole
reason this module exists: macros interact. A macro that loses on its own can
win beside another, because what a library really spends is search budget and
two macros can cover different tensors. Greedy one-at-a-time acceptance cannot
see that — measured directly in this repository, a hand-written four-macro set
beat the baseline on both develop models while most of its members are refused
individually.

Everything here is a pure function of (library, table, rng), so the search loop
stays deterministic given a seed and every operator is testable without the
engine.
"""

from __future__ import annotations

import random
from typing import Callable

from library import HOLE, Library, Macro, OperatorTable, body_stats, render, validate_macro
from library import LibraryError

# A body deeper than this is unreachable at the depth budgets in use, so
# generating one only wastes an evaluation.
MAX_GENERATED_DEPTH = 3
MAX_GENERATED_NODES = 6
# Above this a library costs more branching than any member can repay.
MAX_LIBRARY_SIZE = 8


def _transforms(table: OperatorTable) -> list[str]:
    return sorted(
        name for name, op in table.by_name.items()
        if op.usable_in_macro_body and not op.terminal
    )


def _terminals(table: OperatorTable) -> list[str]:
    return sorted(
        name for name, op in table.by_name.items()
        if op.usable_in_macro_body and op.terminal
    )


# ==================== generation ====================


def random_body(
    table: OperatorTable,
    rng: random.Random,
    *,
    depth: int = 0,
    max_depth: int = 2,
) -> dict:
    """A uniformly-sampled legal body. The control arm's whole strategy.

    Legal here means structurally legal — right arity, no width-dependent
    operator. Whether it ever *fires* is a question only the engine answers,
    and a random body usually does not. That cost is the point of the arm.
    """
    transforms = _transforms(table)
    op = rng.choice(transforms)
    arity = table.by_name[op].arity or 1
    children = []
    for _ in range(arity):
        if depth + 1 >= max_depth:
            children.append(_random_leaf(table, rng))
        else:
            roll = rng.random()
            if roll < 0.5:
                children.append(dict(HOLE))
            elif roll < 0.75:
                children.append(_random_leaf(table, rng))
            else:
                children.append(
                    random_body(table, rng, depth=depth + 1, max_depth=max_depth)
                )
    return {"op": op, "params": "auto", "children": children}


def _random_leaf(table: OperatorTable, rng: random.Random) -> dict:
    if rng.random() < 0.5:
        return dict(HOLE)
    return {"op": rng.choice(_terminals(table)), "params": "auto", "children": []}


def random_macro(
    table: OperatorTable, rng: random.Random, taken: set[str]
) -> Macro | None:
    """A random macro, or None if the draw was structurally unusable."""
    body = random_body(table, rng)
    stats = body_stats(body, table)
    if stats.nodes < 2 or stats.nodes > MAX_GENERATED_NODES:
        return None
    if stats.transform_depth > MAX_GENERATED_DEPTH:
        return None
    if stats.holes == 0 and stats.terminals == 0:
        return None
    macro = Macro(
        name=_fresh_name("random", taken),
        body=body,
        why="Uniformly sampled from the legal body grammar (control arm).",
        origin="random",
    )
    try:
        validate_macro(macro, table)
    except LibraryError:
        return None
    return macro


def _fresh_name(stem: str, taken: set[str]) -> str:
    for index in range(1, 1000):
        candidate = f"{stem}_{index}"
        if candidate not in taken:
            return candidate
    raise ValueError(f"cannot name a macro under stem {stem!r}")


# ==================== mutation ====================


def _leaf_positions(body: dict, path: tuple = ()) -> list[tuple]:
    if body.get("op") == "hole" or not (body.get("children") or []):
        return [path]
    out = []
    for index, child in enumerate(body["children"]):
        out.extend(_leaf_positions(child, path + (index,)))
    return out


def _at(body: dict, path: tuple) -> dict:
    for index in path:
        body = body["children"][index]
    return body


def _replaced(body: dict, path: tuple, new: dict) -> dict:
    if not path:
        return new
    children = list(body["children"])
    children[path[0]] = _replaced(children[path[0]], path[1:], new)
    return {**body, "children": children}


def pin_a_hole(body: dict, table: OperatorTable, rng: random.Random) -> dict | None:
    """Close an open hole with a terminal, so the macro finishes a program."""
    holes = [p for p in _leaf_positions(body) if _at(body, p).get("op") == "hole"]
    if not holes:
        return None
    terminal = {"op": rng.choice(_terminals(table)), "params": "auto", "children": []}
    return _replaced(body, rng.choice(holes), terminal)


def open_a_leaf(body: dict, table: OperatorTable, rng: random.Random) -> dict | None:
    """Turn a pinned terminal back into a hole, generalising the macro."""
    pinned = [p for p in _leaf_positions(body) if _at(body, p).get("op") != "hole"]
    if not pinned:
        return None
    return _replaced(body, rng.choice(pinned), dict(HOLE))


def swap_an_operator(body: dict, table: OperatorTable, rng: random.Random) -> dict | None:
    """Replace one operator with another of the same arity."""
    nodes = _all_positions(body)
    rng.shuffle(nodes)
    for path in nodes:
        node = _at(body, path)
        if node.get("op") == "hole":
            continue
        arity = len(node.get("children") or [])
        alternatives = [
            name for name, op in table.by_name.items()
            if op.usable_in_macro_body and (op.arity or 0) == arity and name != node["op"]
        ]
        if not alternatives:
            continue
        return _replaced(body, path, {**node, "op": rng.choice(sorted(alternatives))})
    return None


def wrap_in_a_transform(
    body: dict, table: OperatorTable, rng: random.Random
) -> dict | None:
    """Put a unary transform above the body, deepening it by one layer."""
    unary = [n for n in _transforms(table) if (table.by_name[n].arity or 0) == 1]
    if not unary:
        return None
    return {"op": rng.choice(unary), "params": "auto", "children": [body]}


def _all_positions(body: dict, path: tuple = ()) -> list[tuple]:
    out = [path]
    for index, child in enumerate(body.get("children") or []):
        out.extend(_all_positions(child, path + (index,)))
    return out


BODY_MUTATIONS: tuple[Callable, ...] = (
    pin_a_hole, open_a_leaf, swap_an_operator, wrap_in_a_transform,
)


def mutate(
    library: Library, table: OperatorTable, rng: random.Random
) -> Library | None:
    """One edit to one member, or a member dropped outright.

    Dropping is a mutation and not an afterthought: it is how the search
    discovers that a macro admitted earlier is now dead weight beside a better
    one, which greedy acceptance can never undo.
    """
    if not library.macros:
        return None
    if len(library.macros) > 1 and rng.random() < 0.2:
        victim = rng.choice(library.macros)
        return library.without(victim.name)

    target = rng.choice(library.macros)
    for mutation in rng.sample(BODY_MUTATIONS, len(BODY_MUTATIONS)):
        body = mutation(target.body, table, rng)
        if body is None:
            continue
        stats = body_stats(body, table)
        if stats.nodes < 2 or stats.nodes > MAX_GENERATED_NODES:
            continue
        if stats.transform_depth > MAX_GENERATED_DEPTH:
            continue
        candidate = Macro(
            name=_fresh_name(f"{mutation.__name__}", library.names()),
            body=body,
            why=f"{mutation.__name__} applied to {target.name}.",
            origin="mutated",
        )
        try:
            validate_macro(candidate, table)
        except LibraryError:
            continue
        grown = library.without(target.name).with_macro(candidate)
        if render(body) in library.without(target.name).shapes():
            continue
        return grown
    return None


def crossover(
    first: Library, second: Library, rng: random.Random
) -> Library | None:
    """A library built from members of both parents.

    The point is joint evaluation: this is how a pair that only works together
    gets measured together, without either half having to survive on its own.
    """
    pool: list[Macro] = []
    seen: set[str] = set()
    for macro in list(first.macros) + list(second.macros):
        shape = macro.shape()
        if shape in seen:
            continue
        seen.add(shape)
        pool.append(macro)
    if len(pool) < 2:
        return None

    size = rng.randint(1, min(len(pool), MAX_LIBRARY_SIZE))
    chosen = rng.sample(pool, size)
    child = Library(macros=tuple(_uniquely_named(chosen)))
    if child.sha256() in (first.sha256(), second.sha256()):
        return None
    return child


def _uniquely_named(macros: list[Macro]) -> list[Macro]:
    taken: set[str] = set()
    out = []
    for macro in macros:
        name = macro.name if macro.name not in taken else _fresh_name(macro.name, taken)
        taken.add(name)
        out.append(macro if name == macro.name else Macro(
            name=name, body=macro.body, why=macro.why, origin=macro.origin))
    return out


def add(library: Library, macro: Macro) -> Library | None:
    """Extend a library, refusing a duplicate shape or an oversized result."""
    if len(library.macros) >= MAX_LIBRARY_SIZE:
        return None
    if macro.shape() in library.shapes():
        return None
    if macro.name in library.names():
        macro = Macro(name=_fresh_name(macro.name, library.names()),
                      body=macro.body, why=macro.why, origin=macro.origin)
    return library.with_macro(macro)
