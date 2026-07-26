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
Three tiers, because two is not enough to keep an honest claim:

  develop     mined for candidates, shown to the proposer, and the set a gain
              must appear on. Thoroughly contaminated by construction.
  validation  a veto during acceptance: a candidate that helps develop but
              hurts here is refused. Seen once per candidate, so it steers the
              library and is *not* a test set.
  test        never read by mining, by the proposer, or by any acceptance
              decision. `verdict()` is the only thing that touches it, and
              every touch is appended to a test log so repeated measurement is
              visible rather than hidden.

An earlier version of this file had only develop and holdout, and used the
holdout as the acceptance veto — which makes it validation, not test. The
numbers from that arrangement are validation numbers.
"""

from __future__ import annotations

import dataclasses
import json
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
# A corpus total hides a model being badly hurt: on the locked test set the
# library was worth -7.36 MB overall while making one Qwen shard 4.43 MB
# larger. Nobody compressing that shard cares about the corpus. Cap the damage
# any single measured file may take, as a fraction of its own archive.
MAX_MODEL_REGRESSION = 2e-3


def objective(corpus: "CorpusResult", library_bytes: int) -> int:
    """Archive bytes with the rule library charged on top."""
    return corpus.archive_bytes + library_bytes


def worst_regression(
    before: dict[str, int], after: dict[str, int]
) -> tuple[str, int, float] | None:
    """The single measured file hurt most, if any exceeds the per-model limit.

    Returns (name, bytes grown, fraction of its own archive), or None.
    """
    worst = None
    for name, base in before.items():
        if name not in after or base <= 0:
            continue
        grew = after[name] - base
        share = grew / base
        if share > MAX_MODEL_REGRESSION and (worst is None or share > worst[2]):
            worst = (name, grew, share)
    return worst


@dataclasses.dataclass(frozen=True)
class Split:
    """Which models steer the loop, which veto it, and which only witness it."""

    develop: tuple[pathlib.Path, ...]
    validation: tuple[pathlib.Path, ...]
    test: tuple[pathlib.Path, ...] = ()

    def all(self) -> tuple[pathlib.Path, ...]:
        return self.develop + self.validation + self.test

    def describe(self) -> dict:
        return {
            "develop": [engine.label(p) for p in self.develop],
            "validation": [engine.label(p) for p in self.validation],
            "test": [engine.label(p) for p in self.test],
            "roles": {
                "develop": "mined, shown to the proposer, and required to improve",
                "validation": "veto during acceptance; steers the library, so not a test set",
                "test": "read only by verdict(); never by mining, proposing, or acceptance",
            },
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
            measure_corpus(split.validation, budget,
                           base_path if library.macros else None),
        )
    develop_before, holdout_before = baseline

    develop_after = measure_corpus(split.develop, budget, cand_path)
    holdout_after = measure_corpus(split.validation, budget, cand_path)

    base_bytes = len(library.canonical_bytes())
    serialized = len(candidate.canonical_bytes())

    develop_delta = (objective(develop_after, serialized)
                     - objective(develop_before, base_bytes))
    holdout_delta = (objective(holdout_after, serialized)
                     - objective(holdout_before, base_bytes))
    required = -int(ACCEPT_MARGIN * max(develop_before.archive_bytes, 1))
    allowed = int(REGRESSION_TOLERANCE * max(holdout_before.archive_bytes, 1))

    total_archive = develop_after.archive_bytes + holdout_after.archive_bytes
    worst_model = worst_regression(
        {**develop_before.per_model, **holdout_before.per_model},
        {**develop_after.per_model, **holdout_after.per_model},
    )
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
            f"validation regressed {holdout_delta} bytes, tolerance {allowed}"
        )
    elif slowdown > MAX_PLANNING_SLOWDOWN:
        accept, reason = False, (
            f"planning slowed {slowdown:.2f}x, limit {MAX_PLANNING_SLOWDOWN}x"
        )
    elif worst_model is not None:
        accept, reason = False, (
            f"{worst_model[0]} grew {worst_model[1]} bytes "
            f"({worst_model[2]:.2%} of itself), above the "
            f"{MAX_MODEL_REGRESSION:.2%} per-model limit"
        )
    else:
        accept, reason = True, (
            f"develop shrank by {-develop_delta} bytes, "
            f"validation {holdout_delta:+d} bytes"
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
    tier: str
    models: list[str]
    before: int
    after: int
    delta: int
    library_serialized_bytes: int
    planning_slowdown: float
    bit_exact: bool
    prior_measurements_of_this_tier: int
    # Per model, so "the corpus shrank" can be checked against "how many
    # models shrank" and "what was the worst regression".
    per_model_delta: dict = dataclasses.field(default_factory=dict)
    models_improved: int = 0
    models_regressed: int = 0
    worst_model_regression: int = 0

    def describe(self) -> dict:
        return dataclasses.asdict(self)


def verdict(
    library: Library,
    split: Split,
    budget: engine.Budget,
    workdir: pathlib.Path,
    *,
    bit_exact: bool,
    tier: str = "test",
    test_log: pathlib.Path | None = None,
) -> Verdict:
    """Measure the locked tier with and without the library and judge.

    Every measurement is appended to `test_log`. Reading a held-out set more
    than once is multiple testing; recording it makes that visible instead of
    letting a later run quietly become the reported one.
    """
    path = workdir / "verdict-library.json"
    library.write(path)
    models = getattr(split, tier)
    if not models:
        return Verdict(
            success=False, reason=f"the {tier} tier is empty; nothing to measure on",
            tier=tier, models=[], before=0, after=0, delta=0,
            library_serialized_bytes=len(library.canonical_bytes()),
            planning_slowdown=1.0, bit_exact=bit_exact,
            prior_measurements_of_this_tier=0,
        )

    prior_touches = 0
    if test_log is not None and test_log.exists():
        prior_touches = sum(
            1 for line in test_log.read_text().splitlines()
            if line.strip() and json.loads(line).get("tier") == tier
        )

    before = measure_corpus(models, budget, None)
    after = measure_corpus(models, budget, path if library.macros else None)
    serialized = len(library.canonical_bytes())
    delta = objective(after, serialized) - objective(before, 0)
    slowdown = after.planning_wall_ms / max(before.planning_wall_ms, 1)

    if not library.macros:
        success, reason = False, "the library is empty; nothing was learned"
    elif not bit_exact:
        success, reason = False, "bit-exactness was not confirmed"
    elif delta >= 0:
        success, reason = False, (
            f"the {tier} tier did not shrink: {delta:+d} bytes once the "
            f"{serialized}-byte library is charged"
        )
    elif slowdown > MAX_PLANNING_SLOWDOWN:
        success, reason = False, (
            f"the {tier} tier shrank by {-delta} bytes but planning slowed "
            f"{slowdown:.2f}x, above the {MAX_PLANNING_SLOWDOWN}x limit"
        )
    else:
        success, reason = True, (
            f"the {tier} tier shrank by {-delta} bytes after charging the "
            f"{serialized}-byte library, at {slowdown:.2f}x planning cost"
        )

    per_model = {
        name: after.per_model[name] - base
        for name, base in before.per_model.items()
        if name in after.per_model
    }
    result = Verdict(
        success=success,
        reason=reason,
        tier=tier,
        per_model_delta=per_model,
        models_improved=sum(1 for d in per_model.values() if d < 0),
        models_regressed=sum(1 for d in per_model.values() if d > 0),
        worst_model_regression=max(per_model.values(), default=0),
        models=[engine.label(p) for p in models],
        before=before.archive_bytes,
        after=after.archive_bytes,
        delta=delta,
        library_serialized_bytes=serialized,
        planning_slowdown=slowdown,
        bit_exact=bit_exact,
        prior_measurements_of_this_tier=prior_touches,
    )
    if test_log is not None:
        test_log.parent.mkdir(parents=True, exist_ok=True)
        with test_log.open("a") as handle:
            handle.write(json.dumps(
                {**result.describe(), "library_sha256": library.sha256(),
                 "budget": budget.describe()}, sort_keys=True) + "\n")
    return result
