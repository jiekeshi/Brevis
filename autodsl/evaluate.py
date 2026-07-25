"""The acceptance gate: does a macro make the corpus smaller?

The objective
-------------
The natural MDL statement for a self-extending grammar is

    minimize  |Serialize(L)| + sum_i |Serialize(P_i | L)|

Applied to Brevis literally, both of those terms are noise. Program bytecode is
about 0.13% of an archive, and because the engine expands macros before
serializing, a library contributes exactly zero bytes to an archive. Optimizing
either term would be optimizing a rounding error.

What a library actually spends is *search budget*: every macro is offered at
every hole, so a bigger library explores fewer distinct structures within a
fixed expansion budget. So the objective this module enforces is

    minimize  sum_i archive_bytes(tensor_i | L) + |Serialize(L)|
    subject to  bit-exact decode, and planning time within MAX_PLANNING_SLOWDOWN

`|Serialize(L)|` is charged even though it is negligible here (a library is
expanded away before serialization, so it costs an archive nothing). Charging
it anyway means the arithmetic stays honest if a library ever ships inside an
archive, and means a reported gain is a gain *after* the rule library is paid
for rather than before.

Success is defined on the holdout set, not this one. Acceptance into the
library is a learning step driven by develop; whether the library is worth
anything is `verdict()`, which asks only whether unseen checkpoints got
smaller.

Why a macro can lose
--------------------
A macro cannot make any single program worse — it builds programs the primitive
grammar could also build. It can still make the *corpus* worse, by consuming
expansions that would have found something better. That is a real cost, it is
what this gate measures, and it is why "the macro looks sensible" is not
evidence.

The split
---------
Proposals are written against develop-set evidence, so accepting on the develop
set alone would reward memorizing it. Acceptance therefore requires a develop
gain *and* the absence of a holdout regression. With a handful of cached
models this is a weak split and the ledger records it as such.
"""

from __future__ import annotations

import dataclasses
import pathlib

import engine
from library import Library

# A gain smaller than this is real but not worth the permanent branching cost.
# Expressed as a fraction of the corpus archive, so it scales with the corpus.
ACCEPT_MARGIN = 1e-5
# A macro may cost the holdout set a little if it earns more on develop, but
# not more than it earns.
REGRESSION_TOLERANCE = 1e-5
# Every macro is offered at every hole, so planning slows as the library grows.
# A library that buys bytes with unbounded planning time has not paid for
# itself; this is the "acceptable running cost" half of the objective.
MAX_PLANNING_SLOWDOWN = 1.25


def objective(corpus: "CorpusResult", library_bytes: int) -> int:
    """Archive bytes with the rule library charged on top."""
    return corpus.archive_bytes + library_bytes


@dataclasses.dataclass(frozen=True)
class Split:
    """Which models steer the loop and which only witness it."""

    develop: tuple[pathlib.Path, ...]
    holdout: tuple[pathlib.Path, ...]

    def all(self) -> tuple[pathlib.Path, ...]:
        return self.develop + self.holdout

    def describe(self) -> dict:
        return {
            "develop": [engine.label(p) for p in self.develop],
            "holdout": [engine.label(p) for p in self.holdout],
            "caveat": (
                "a handful of cached checkpoints is a weak split; a claim about "
                "generalization needs more models than this"
            ),
        }


@dataclasses.dataclass(frozen=True)
class CorpusResult:
    archive_bytes: int
    raw_bytes: int
    per_model: dict[str, int]
    planning_wall_ms: int

    @property
    def ratio(self) -> float:
        return self.raw_bytes / self.archive_bytes if self.archive_bytes else 0.0


def measure_corpus(
    models: tuple[pathlib.Path, ...],
    budget: engine.Budget,
    macros: pathlib.Path | None,
) -> CorpusResult:
    total = raw = wall = 0
    per_model: dict[str, int] = {}
    for model in models:
        measurement = engine.bench(model, budget, macros)
        per_model[engine.label(model)] = measurement.archive_bytes
        total += measurement.archive_bytes
        raw += measurement.raw_bytes
        wall += measurement.planning_wall_ms
    return CorpusResult(total, raw, per_model, wall)


@dataclasses.dataclass(frozen=True)
class Decision:
    accept: bool
    reason: str
    develop_delta: int
    holdout_delta: int
    develop_before: int
    develop_after: int
    holdout_before: int
    holdout_after: int
    planning_wall_delta_ms: int
    planning_slowdown: float
    library_serialized_bytes: int
    library_share_of_archive: float

    def describe(self) -> dict:
        return dataclasses.asdict(self)


def evaluate(
    library: Library,
    candidate: Library,
    split: Split,
    budget: engine.Budget,
    workdir: pathlib.Path,
    *,
    baseline: tuple[CorpusResult, CorpusResult] | None = None,
) -> tuple[Decision, tuple[CorpusResult, CorpusResult]]:
    """Compare `candidate` against `library` on both halves of the split.

    Returns the decision and the *candidate's* corpus results, so an accepted
    candidate's numbers become the next round's baseline without re-measuring.
    """
    base_path = workdir / "baseline.json"
    cand_path = workdir / "candidate.json"
    library.write(base_path)
    candidate.write(cand_path)

    if baseline is None:
        baseline = (
            measure_corpus(split.develop, budget, base_path if library.macros else None),
            measure_corpus(split.holdout, budget, base_path if library.macros else None),
        )
    develop_before, holdout_before = baseline

    develop_after = measure_corpus(split.develop, budget, cand_path)
    holdout_after = measure_corpus(split.holdout, budget, cand_path)

    base_bytes = len(library.canonical_bytes())
    serialized = len(candidate.canonical_bytes())

    develop_delta = (objective(develop_after, serialized)
                     - objective(develop_before, base_bytes))
    holdout_delta = (objective(holdout_after, serialized)
                     - objective(holdout_before, base_bytes))
    required = -int(ACCEPT_MARGIN * max(develop_before.archive_bytes, 1))
    allowed = int(REGRESSION_TOLERANCE * max(holdout_before.archive_bytes, 1))

    total_archive = develop_after.archive_bytes + holdout_after.archive_bytes
    planning_before = develop_before.planning_wall_ms + holdout_before.planning_wall_ms
    planning_after = develop_after.planning_wall_ms + holdout_after.planning_wall_ms
    slowdown = planning_after / max(planning_before, 1)

    if develop_delta > required:
        movement = (
            f"grew by {develop_delta}" if develop_delta > 0
            else f"shrank by only {-develop_delta}"
        )
        accept, reason = False, (
            f"develop {movement} bytes; acceptance needs a fall of at least "
            f"{-required}"
        )
    elif holdout_delta > allowed:
        accept, reason = False, (
            f"holdout regressed {holdout_delta} bytes, tolerance {allowed}"
        )
    elif slowdown > MAX_PLANNING_SLOWDOWN:
        accept, reason = False, (
            f"planning slowed {slowdown:.2f}x, limit {MAX_PLANNING_SLOWDOWN}x"
        )
    else:
        accept, reason = True, (
            f"develop shrank by {-develop_delta} bytes, "
            f"holdout {holdout_delta:+d} bytes"
        )

    decision = Decision(
        accept=accept,
        reason=reason,
        develop_delta=develop_delta,
        holdout_delta=holdout_delta,
        develop_before=develop_before.archive_bytes,
        develop_after=develop_after.archive_bytes,
        holdout_before=holdout_before.archive_bytes,
        holdout_after=holdout_after.archive_bytes,
        planning_wall_delta_ms=planning_after - planning_before,
        planning_slowdown=slowdown,
        library_serialized_bytes=serialized,
        library_share_of_archive=serialized / max(total_archive, 1),
    )
    return decision, (develop_after, holdout_after)


# ==================== the success question ====================


@dataclasses.dataclass(frozen=True)
class Verdict:
    """Did the learned library earn its keep on checkpoints it never saw?

    This is the whole point of the exercise, stated as one boolean. Acceptance
    decisions are made against the develop set and can be wrong; this asks the
    only question that settles it.
    """

    success: bool
    reason: str
    holdout_models: list[str]
    holdout_before: int
    holdout_after: int
    holdout_delta: int
    library_serialized_bytes: int
    planning_slowdown: float
    bit_exact: bool

    def describe(self) -> dict:
        return dataclasses.asdict(self)


def verdict(
    library: Library,
    split: Split,
    budget: engine.Budget,
    workdir: pathlib.Path,
    *,
    bit_exact: bool,
) -> Verdict:
    """Measure the holdout set with and without the library and judge."""
    path = workdir / "verdict-library.json"
    library.write(path)

    before = measure_corpus(split.holdout, budget, None)
    after = measure_corpus(split.holdout, budget, path if library.macros else None)
    serialized = len(library.canonical_bytes())
    delta = objective(after, serialized) - objective(before, 0)
    slowdown = after.planning_wall_ms / max(before.planning_wall_ms, 1)

    if not library.macros:
        success, reason = False, "the library is empty; nothing was learned"
    elif not bit_exact:
        success, reason = False, "bit-exactness was not confirmed"
    elif delta >= 0:
        success, reason = False, (
            f"unseen checkpoints did not shrink: {delta:+d} bytes once the "
            f"{serialized}-byte library is charged"
        )
    elif slowdown > MAX_PLANNING_SLOWDOWN:
        success, reason = False, (
            f"unseen checkpoints shrank by {-delta} bytes but planning slowed "
            f"{slowdown:.2f}x, above the {MAX_PLANNING_SLOWDOWN}x limit"
        )
    else:
        success, reason = True, (
            f"unseen checkpoints shrank by {-delta} bytes after charging the "
            f"{serialized}-byte library, at {slowdown:.2f}x planning cost"
        )

    return Verdict(
        success=success,
        reason=reason,
        holdout_models=[engine.label(p) for p in split.holdout],
        holdout_before=before.archive_bytes,
        holdout_after=after.archive_bytes,
        holdout_delta=delta,
        library_serialized_bytes=serialized,
        planning_slowdown=slowdown,
        bit_exact=bit_exact,
    )
