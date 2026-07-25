"""Turn measured programs into macro candidates, and measured weakness into
evidence a proposer can act on.

Mining is the deterministic half of the loop and runs with no model in sight.
It answers "which subtree did the search keep paying to rediscover?", which is
the classic library-learning question, using bytes rather than counts as the
weight: a shape that governs 287 MB matters more than one that governs 0.2 MB.
"""

from __future__ import annotations

import collections
import dataclasses
from typing import Any, Iterator

from library import HOLE, Library, Macro, OperatorTable, body_stats, render

# A macro that expands to one operator saves no expansion, so it is not a macro.
MIN_BODY_NODES = 2
# Guards the generalization enumeration, which is exponential in body nodes.
MAX_TREE_NODES_TO_GENERALIZE = 8


@dataclasses.dataclass(frozen=True)
class Candidate:
    body: dict
    bytes_governed: int
    blocks: int
    node_count: int
    hole_count: int
    transform_depth: int

    @property
    def shape(self) -> str:
        return render(self.body)


def _tree_size(tree: dict) -> int:
    return 1 + sum(_tree_size(c) for c in tree.get("children") or [])


def _strip(tree: dict) -> dict:
    """A report's program tree reduced to what a macro body may contain.

    Params are dropped to `auto`: a mined body should re-derive them from the
    stream it actually meets, not freeze the one it was mined from.
    """
    return {
        "op": tree["op"],
        "params": "auto",
        "children": [_strip(c) for c in tree.get("children") or []],
    }


def generalizations(tree: dict, table: OperatorTable) -> Iterator[dict]:
    """Every way to replace subtrees of `tree` with holes, root always kept.

    The fully-holed variants are the reusable skeletons; the fully-kept variant
    is the complete program, which is worth having as a macro of its own since
    it lets one expansion finish a program outright.
    """
    op = table.by_name.get(tree["op"])
    if op is None or not op.usable_in_macro_body:
        return
    children = tree.get("children") or []
    if not children:
        yield {"op": tree["op"], "params": "auto", "children": []}
        return

    per_child: list[list[dict]] = []
    for child in children:
        options: list[dict] = [dict(HOLE)]
        options.extend(generalizations(child, table))
        per_child.append(options)

    def product(index: int) -> Iterator[list[dict]]:
        if index == len(per_child):
            yield []
            return
        for tail in product(index + 1):
            for option in per_child[index]:
                yield [option] + tail

    for combination in product(0):
        yield {"op": tree["op"], "params": "auto", "children": combination}


def mine(report: dict, table: OperatorTable, library: Library) -> list[Candidate]:
    """Rank macro candidates by the archive bytes their shape already governs."""
    weight: dict[str, int] = collections.Counter()
    blocks: dict[str, int] = collections.Counter()
    bodies: dict[str, dict] = {}

    for block in report["blocks"]:
        tree = block.get("program_tree")
        if tree is None or _tree_size(tree) > MAX_TREE_NODES_TO_GENERALIZE:
            continue
        governed = block["encoded_bytes_without_frame_headers"]
        for body in generalizations(_strip(tree), table):
            stats = body_stats(body, table)
            if stats.nodes < MIN_BODY_NODES or stats.nodes > table.max_body_nodes:
                continue
            if stats.holes == 0 and stats.terminals == 0:
                continue
            shape = render(body)
            weight[shape] += governed
            blocks[shape] += 1
            bodies.setdefault(shape, body)

    known = library.shapes()
    out: list[Candidate] = []
    for shape, governed in weight.items():
        if shape in known:
            continue
        stats = body_stats(bodies[shape], table)
        out.append(
            Candidate(
                body=bodies[shape],
                bytes_governed=governed,
                blocks=blocks[shape],
                node_count=stats.nodes,
                hole_count=stats.holes,
                transform_depth=stats.transform_depth,
            )
        )
    out.sort(key=lambda c: (-c.bytes_governed, c.node_count, c.shape))
    return out


def family(body: Any) -> str:
    """The candidate's transform skeleton, with every terminal holed.

    `generalizations` emits one candidate per way of pinning terminals, so a
    single winning tree yields a whole family of near-identical shapes. Ranked
    by bytes they arrive as a block and crowd out every other structure, which
    is a poor use of a bounded number of gate runs.
    """
    children = body.get("children") or []
    if body.get("op") == "hole" or not children:
        return "?"
    return f"{body['op']}({','.join(family(c) for c in children)})"


def diversify(candidates: list[Candidate], per_family: int = 1) -> list[Candidate]:
    """Keep the highest-earning `per_family` candidates from each skeleton."""
    seen: dict[str, int] = {}
    out: list[Candidate] = []
    for candidate in candidates:
        key = family(candidate.body)
        if seen.get(key, 0) >= per_family:
            continue
        seen[key] = seen.get(key, 0) + 1
        out.append(candidate)
    return out


def to_macro(candidate: Candidate, name: str, origin: str) -> Macro:
    return Macro(
        name=name,
        body=candidate.body,
        why=(
            f"Mined from {candidate.blocks} blocks governing "
            f"{candidate.bytes_governed / 1e6:.1f} MB of encoded output."
        ),
        origin=origin,
    )


def suggest_name(shape: str, taken: set[str]) -> str:
    base = (
        shape.replace("(", "_").replace(")", "").replace(",", "_").replace("?", "h")
    ).strip("_")[:44]
    base = base or "macro"
    if base not in taken:
        return base
    for i in range(2, 100):
        candidate = f"{base}_{i}"[:48]
        if candidate not in taken:
            return candidate
    raise ValueError(f"cannot name {shape!r}")


# ==================== evidence for a proposer ====================


def evidence(report: dict, table: OperatorTable) -> dict:
    """A compact, factual picture of where this model's bytes are going.

    This is what a proposer is shown. It is measurement only — no
    interpretation, no suggestion of what to try — so that a proposal can be
    credited to the model rather than to the prompt.
    """
    blocks = report["blocks"]
    by_program: dict[str, dict] = {}
    for block in blocks:
        entry = by_program.setdefault(
            block["program"], {"blocks": 0, "encoded_bytes": 0, "raw_bytes": 0}
        )
        entry["blocks"] += 1
        entry["encoded_bytes"] += block["encoded_bytes_without_frame_headers"]
        entry["raw_bytes"] += block["raw_bytes"]

    programs = sorted(by_program.items(), key=lambda kv: -kv[1]["encoded_bytes"])
    for _, entry in programs:
        entry["ratio"] = round(entry["raw_bytes"] / max(entry["encoded_bytes"], 1), 4)

    by_dtype: dict[str, dict] = {}
    tensors = report["tensors"]
    for tensor in tensors:
        entry = by_dtype.setdefault(
            tensor["dtype"], {"tensors": 0, "raw_bytes": 0, "encoded_bytes": 0}
        )
        entry["tensors"] += 1
        entry["raw_bytes"] += tensor["raw_bytes"]
        entry["encoded_bytes"] += tensor["encoded_bytes_without_frame_headers"]
    for entry in by_dtype.values():
        entry["ratio"] = round(entry["raw_bytes"] / max(entry["encoded_bytes"], 1), 4)

    budget = report["search"]["max_expansions"]
    saturated = sum(1 for t in tensors if t.get("search_expansions") == budget)

    worst = sorted(
        (t for t in tensors if t["raw_bytes"] > 0),
        key=lambda t: t["raw_bytes"] / max(t["encoded_bytes_without_frame_headers"], 1),
    )[:10]

    terminals = collections.Counter()
    for block in blocks:
        for op in _terminal_ops(block.get("program_tree") or {}):
            terminals[op] += 1

    return {
        "model": report["input"],
        "raw_bytes": report["raw_bytes"],
        "archive_bytes": report["projected_archive_bytes"],
        "ratio": round(report["raw_bytes"] / max(report["projected_archive_bytes"], 1), 4),
        "bytecode_share_of_archive": round(
            sum(b["program_bytecode_bytes"] for b in blocks)
            / max(report["projected_archive_bytes"], 1),
            6,
        ),
        "search_budget_saturated_tensors": f"{saturated}/{len(tensors)}",
        "distinct_block_programs": len(by_program),
        "programs_by_encoded_bytes": [
            {"program": p, **e} for p, e in programs[:15]
        ],
        "by_dtype": by_dtype,
        "terminal_node_counts": dict(terminals),
        "worst_compressing_tensors": [
            {
                "name": t["name"],
                "dtype": t["dtype"],
                "shape": t["shape"],
                "raw_bytes": t["raw_bytes"],
                "ratio": round(
                    t["raw_bytes"] / max(t["encoded_bytes_without_frame_headers"], 1), 4
                ),
                "program": t["program"],
            }
            for t in worst
        ],
    }


def _terminal_ops(tree: dict) -> Iterator[str]:
    if not tree:
        return
    if tree.get("terminal"):
        yield tree["op"]
        return
    for child in tree.get("children") or []:
        yield from _terminal_ops(child)


def _unused(value: Any) -> None:  # pragma: no cover - keeps linters quiet
    return None
