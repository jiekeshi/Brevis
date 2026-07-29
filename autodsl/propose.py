"""Ask a model for new macros, and parse what comes back.

The proposer is the creative step and the only non-deterministic one. It is
given measurements and the DSL, and nothing else: no hints about which
operators to combine, no examples of "good" macros beyond what the current
library already contains. Whatever it returns is a *candidate* — `verify.py`
and `evaluate.py` decide.

The operator list in the prompt is generated from `brevis config`, so a new
operator in the engine appears here without anyone remembering to update a
string. `SEMANTICS` is the one hand-written table, because what an operator
*means* is not in the config dump; `test_propose.py` fails if the engine grows
an operator this table does not describe.
"""

from __future__ import annotations

import dataclasses
import json
import re
from typing import Any

from library import Library, Macro, OperatorTable, render, validate_macro

# One line per operator. `src/ops.zig` is authoritative for behaviour.
SEMANTICS = {
    "raw": "terminal: store the stream verbatim. Always legal; the fallback.",
    "bitpack": "terminal: pack each element into the minimum fixed bit width.",
    "huffman": "terminal: canonical Huffman over the element alphabet. Legal only up to 16-bit elements.",
    "rans": "terminal: range asymmetric numeral systems. Legal only up to 16-bit elements.",
    "xor_const": "x -> x XOR c, with c the modal value. Alphabet permutation: entropy unchanged.",
    "add_const_mod": "x -> x + c mod 2^k, c chosen to send the mode to zero. Alphabet permutation.",
    "xor_prev": "x[i] -> x[i] XOR x[i-1]. Uses the neighbour, so it can lower entropy.",
    "diff_mod": "x[i] -> x[i] - x[i-1] mod 2^k. The delta transform.",
    "zigzag": "map signed-magnitude ordering to unsigned so small negatives become small. Alphabet permutation.",
    "gray": "binary -> Gray code, so neighbouring values differ in one bit. Alphabet permutation.",
    "rotate_bits": "rotate each element left by k/2 bits. Alphabet permutation.",
    "bit_reverse": "reverse the bit order within each element. Alphabet permutation.",
    "split_field": "split each element into a high field and the remaining low bits -> 2 streams. On a float width it splits at the mantissa boundary.",
    "topk_codebook": "replace the 15 most frequent symbols by indices -> (indices, escapes).",
    "rle": "run-length encode -> (values, run lengths).",
    "deinterleave": "split by position modulo a period -> 2 streams.",
    "split_float": "split an IEEE-style float into (sign, exponent, mantissa). Legal only at the root of a floating tensor.",
    "bit_plane": "one stream per bit position. Width-dependent arity: not usable in a macro body.",
    "byte_plane": "one stream per byte position. Width-dependent arity: not usable in a macro body.",
}

SYSTEM = """\
You extend the operator grammar of Brevis, a lossless tensor compressor that \
synthesizes one reversible program per tensor by A* search.

You propose MACROS. A macro is a named subtree of existing operators whose \
leaves are holes (the search continues there) or terminals (the branch is \
closed). A macro contains no data and invents no operator. The engine expands \
it into primitives before anything is written, so a macro can never change \
what is representable or what an archive looks like.

What a macro changes is REACHABILITY UNDER A FIXED BUDGET. The search pops a \
bounded number of partial programs per tensor. Applying a macro costs one pop \
and delivers its whole subtree. A useful macro is therefore a structure that \
is worth reaching but expensive to assemble one operator at a time.

Judge your own proposals against that. A macro is worthless if:
  - it is one operator plus a hole, which is just a primitive production;
  - the search already reaches it easily and it appears in the measurements;
  - its operators cannot legally meet the widths involved.

Reply with a single fenced ```json block and nothing else, of the form:
{"macros": [{"name": "lower_snake_case", "why": "one or two sentences", \
"body": <body>}]}

where <body> is {"op": "hole"} or \
{"op": "<operator>", "params": "auto", "children": [<body>, ...]} with exactly \
the operator's arity of children. Use "params": "auto" unless you have a \
specific reason; auto re-derives the parameter from the stream the node \
actually meets, which is what lets one macro work across dtypes.
"""

JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class ProposalError(ValueError):
    pass


@dataclasses.dataclass(frozen=True)
class Proposal:
    macros: tuple[Macro, ...]
    raw_reply: str
    prompt: str


def operator_reference(table: OperatorTable) -> str:
    lines = []
    for name in sorted(table.by_name):
        op = table.by_name[name]
        kind = "terminal" if op.terminal else f"transform, {op.arity} children"
        if not op.usable_in_macro_body:
            kind = "NOT USABLE IN A MACRO BODY (width-dependent arity)"
        lines.append(f"  {name:16s} [{kind}] {SEMANTICS.get(name, '(undocumented)')}")
    return "\n".join(lines)


def build_prompt(
    evidence: dict,
    library: Library,
    table: OperatorTable,
    *,
    budget: dict,
    mined: list | None = None,
    rejected: list[dict] | None = None,
    wanted: int = 4,
    history: list[dict] | None = None,
    marginals: dict | None = None,
) -> str:
    current = (
        "\n".join(
            f"  {m.name:28s} {render(m.body)}"
            + (f"   -- {m.why}" if m.why else "")
            for m in library.macros
        )
        or "  (empty)"
    )
    mined_text = (
        "\n".join(
            f"  {c.shape:44s} {c.bytes_governed / 1e6:8.1f} MB over {c.blocks} blocks"
            for c in (mined or [])[:12]
        )
        or "  (none)"
    )
    rejected_text = (
        "\n".join(
            f"  {r['shape']:44s} rejected: {r['reason']}" for r in (rejected or [])[:15]
        )
        or "  (none yet)"
    )
    history_text = (
        "\n".join(
            f"  {'+' if row['objective_delta_vs_best'] > 0 else ''}"
            f"{row['objective_delta_vs_best']:>10d} bytes vs the best so far  "
            f"{'' if row['feasible'] else '[INFEASIBLE: ' + row['note'] + '] '}"
            f"{' + '.join(row['macros']) or '(empty library)'}"
            for row in (history or [])[:10]
        )
        or "  (nothing measured yet)"
    )

    marginal_text = (
        "\n".join(
            f"  {name:28s} {value:>+12d} bytes when removed from the current library"
            for name, value in sorted((marginals or {}).items(),
                                      key=lambda kv: kv[1])
        )
        or "  (not measured yet)"
    )

    return f"""\
OPERATORS
{operator_reference(table)}

SEARCH BUDGET (identical for every measurement; a macro must pay within it)
{json.dumps(budget, indent=2)}

MEASURED BEHAVIOUR OF THE CURRENT SYSTEM
{json.dumps(evidence, indent=2)}

CURRENT LIBRARY
{current}

WHAT EACH MEMBER IS WORTH, measured by removing it and re-running. A negative
number means the member pays for itself; a number near zero means it is being
carried by the others and could be replaced.
{marginal_text}

SHAPES THE SEARCH ALREADY FINDS ON ITS OWN, by encoded bytes they govern.
Proposing one of these is usually pointless: the search reaches them anyway.
{mined_text}

LIBRARIES ALREADY MEASURED THIS RUN, best first. A library is a *set* of
macros measured together, so a shape that appears in a losing set is not
thereby a losing shape.
{history_text}

ALREADY TRIED AND REJECTED — do not propose these again
{rejected_text}

Propose up to {wanted} macros that are not in the library, are not in the
rejected list, and are not simply a shape the search already finds. Prefer
structures that need several operators stacked, because those are the ones a
bounded search misses. Explain each in one or two sentences grounded in the
measurements above.
"""


def parse_reply(reply: str, table: OperatorTable, library: Library) -> list[Macro]:
    """Extract macros from a reply, rejecting anything malformed loudly."""
    match = JSON_BLOCK.search(reply)
    text = match.group(1) if match else reply
    try:
        blob = json.loads(text.strip())
    except json.JSONDecodeError as exc:
        raise ProposalError(f"reply is not JSON: {exc}\n{reply[:600]}") from exc
    if not isinstance(blob, dict) or "macros" not in blob:
        raise ProposalError(f"reply has no 'macros' key: {text[:400]}")

    out: list[Macro] = []
    taken = library.names()
    for raw in blob["macros"]:
        if not isinstance(raw, dict):
            raise ProposalError(f"macro entry is not an object: {raw!r}")
        macro = Macro(
            name=str(raw.get("name", "")),
            body=_normalize(raw.get("body")),
            why=str(raw.get("why", "")).strip(),
            origin="proposed",
        )
        validate_macro(macro, table)  # raises LibraryError with a precise message
        if macro.name in taken:
            raise ProposalError(f"macro name {macro.name!r} is already taken")
        taken.add(macro.name)
        out.append(macro)
    return out


def _normalize(body: Any) -> dict:
    """Fill in the fields a model is likely to omit, and reject the rest."""
    if not isinstance(body, dict):
        raise ProposalError(f"body must be an object, got {body!r}")
    if body.get("op") == "hole":
        return {"op": "hole"}
    return {
        "op": body.get("op"),
        "params": body.get("params", "auto"),
        "children": [_normalize(c) for c in body.get("children") or []],
    }


def propose(
    llm,
    evidence: dict,
    library: Library,
    table: OperatorTable,
    *,
    budget: dict,
    mined: list | None = None,
    rejected: list[dict] | None = None,
    wanted: int = 4,
    history: list[dict] | None = None,
    marginals: dict | None = None,
) -> Proposal:
    prompt = build_prompt(
        evidence, library, table, budget=budget, mined=mined,
        rejected=rejected, wanted=wanted, history=history, marginals=marginals,
    )
    reply = llm.complete(SYSTEM, prompt)
    return Proposal(
        macros=tuple(parse_reply(reply, table, library)),
        raw_reply=reply,
        prompt=prompt,
    )
