#!/usr/bin/env python3
"""Population search over macro libraries, and the baselines it must beat.

A library, not a macro, carries a fitness here. Greedy one-at-a-time
acceptance can only ever find macros that pay in isolation, and macros
interact: two can cover different tensors, or one can make another's
expansions redundant. Searching over libraries is what makes a *combination*
measurable.

Fair comparison is the point, so every arm spends the same currency: one
**evaluation** is one fitness measurement of one library over the develop
tier. Cache hits cost nothing and every arm shares the cache, so a method that
revisits a library is not punished for it. When the budget is gone the arm
stops, whatever it was doing.

Fitness is develop-only, which is what a training objective should be, and
also what makes the budget affordable. The validation tier is applied once per
arm, to its final library, as a veto — an arm that wins on develop by hurting
validation does not get to report the win.

    search    population, mutation, crossover, LLM proposals, mining, random
    greedy    LLM proposals accepted one at a time (what this used to be)
    one_shot  a single LLM call; its whole batch is the library, take it or not
    mining    mined shapes accepted one at a time, no model involved
    random    uniformly sampled legal macros accepted one at a time

Only the develop and validation tiers are read. The test tier is `verdict`'s
alone.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import backend  # noqa: E402
import engine  # noqa: E402
import evaluate  # noqa: E402
import loop  # noqa: E402
import mine as mine_mod  # noqa: E402
import propose as propose_mod  # noqa: E402
import variation  # noqa: E402
import verify as verify_mod  # noqa: E402
from ledger import Ledger  # noqa: E402
from library import Library, OperatorTable  # noqa: E402

# Rounds without a new best before an arm concludes it has converged.
DEFAULT_PATIENCE = 3
DEFAULT_POPULATION = 4
DEFAULT_EVALUATIONS = 40
DEFAULT_SEED = 0x5EEDB10C


@dataclasses.dataclass(frozen=True)
class Fitness:
    """What one library is worth, and whether it is allowed to count."""

    objective: int          # develop archive bytes + |Serialize(L)|
    develop_bytes: int
    planning_ms: int
    feasible: bool
    reason: str
    per_model: dict
    # The worst per-model relative gain, negative when every model improved.
    # Under `minimax` this, not `objective`, decides which library is better.
    worst_relative_gain: float = 0.0
    mode: str = "sum"

    def score(self) -> tuple:
        """Ordering key, smaller is better.

        Under `minimax` the worst model leads and the tier total breaks ties.
        The tie-break is not cosmetic: the worst model is often untouched by
        every candidate, and without a second term the search silently
        degenerates into "prefer the smallest library" and picks among equals
        arbitrarily.
        """
        if self.mode == "minimax":
            return (self.worst_relative_gain, self.objective)
        return (self.objective,)

    def better_than(self, other: "Fitness | None") -> bool:
        if other is None:
            return self.feasible
        if self.feasible != other.feasible:
            return self.feasible
        return self.score() < other.score()


@dataclasses.dataclass
class Scored:
    library: Library
    fitness: Fitness
    origin: str
    round_found: int

    @property
    def key(self) -> str:
        return self.library.sha256()


class BudgetExhausted(Exception):
    """The arm has spent its evaluations. Not an error — a stopping condition."""


def marginals(library: Library, cache: dict) -> dict[str, float]:
    """What each member is worth, from libraries already measured.

    `fitness(L) - fitness(L without m)` is the marginal value of `m` inside
    `L`, and it is free whenever both have been measured. Leave-one-out
    libraries are generated as candidates every round precisely so that this
    is usually the case: dropping a member is both a credit-assignment probe
    and a real candidate, since a macro admitted early can become dead weight.

    Negative means the member is paying for itself.
    """
    full = cache.get(library.sha256())
    if full is None:
        return {}
    out: dict[str, float] = {}
    for macro in library.macros:
        without = cache.get(library.without(macro.name).sha256())
        if without is not None:
            out[macro.name] = full.objective - without.objective
    return out


class Evaluator:
    """Measures libraries, counts what that cost, and refuses to overspend.

    The cache is shared across arms within a run so that a library measured by
    one arm is free for another. That helps every arm equally and keeps the
    budget denominated in *distinct* libraries measured, which is the quantity
    a reader cares about.
    """

    def __init__(
        self,
        split: evaluate.Split,
        budget: engine.Budget,
        workdir: pathlib.Path,
        cap: int,
        cache: dict | None = None,
        mode: str = "sum",
    ) -> None:
        self.mode = mode
        self.split = split
        self.budget = budget
        self.workdir = workdir
        self.cap = cap
        self.cache: dict[str, Fitness] = cache if cache is not None else {}
        self.spent = 0
        self.baseline: Fitness | None = None
        self.trace: list[dict] = []

    def validate(self, library: Library) -> evaluate.CorpusResult:
        """The end-of-arm veto. Off-budget: it is not search effort, and every
        arm pays it exactly once, so it cannot advantage any of them."""
        path = self.workdir / f"val-{library.sha256()[:16]}.json"
        library.write(path)
        try:
            return evaluate.measure_corpus(
                self.split.validation, self.budget,
                path if library.macros else None)
        finally:
            path.unlink(missing_ok=True)

    @property
    def remaining(self) -> int:
        return max(0, self.cap - self.spent)

    def measure(self, library: Library, *, free: bool = False) -> Fitness:
        key = library.sha256()
        if key in self.cache:
            return self.cache[key]
        if not free and self.remaining == 0:
            raise BudgetExhausted(f"spent all {self.cap} evaluations")

        path = self.workdir / f"lib-{key[:16]}.json"
        self.workdir.mkdir(parents=True, exist_ok=True)
        library.write(path)
        macros = path if library.macros else None

        develop = evaluate.measure_corpus(self.split.develop, self.budget, macros)
        path.unlink(missing_ok=True)
        if not free:
            self.spent += 1

        serialized = len(library.canonical_bytes())
        per_model = dict(develop.per_model)
        planning = develop.planning_wall_ms

        feasible, reason = True, "feasible"
        if self.baseline is not None:
            slowdown = planning / max(self.baseline.planning_ms, 1)
            if slowdown > evaluate.MAX_PLANNING_SLOWDOWN:
                feasible, reason = False, f"planning {slowdown:.2f}x"
            else:
                worst = evaluate.worst_regression(self.baseline.per_model, per_model)
                if worst is not None:
                    feasible, reason = False, (
                        f"{worst[0]} +{worst[1]}B ({worst[2]:.2%})")

        worst_gain = 0.0
        if self.baseline is not None:
            shares = [
                (per_model[name] - base) / max(base, 1)
                for name, base in self.baseline.per_model.items()
                if name in per_model
            ]
            worst_gain = max(shares) if shares else 0.0

        fitness = Fitness(
            objective=develop.archive_bytes + serialized,
            develop_bytes=develop.archive_bytes,
            planning_ms=planning,
            feasible=feasible,
            reason=reason,
            per_model=per_model,
            worst_relative_gain=worst_gain,
            mode=self.mode,
        )
        self.cache[key] = fitness
        self.trace.append({
            "sha256": key, "macros": [m.shape() for m in library.macros],
            "objective": fitness.objective, "feasible": feasible, "reason": reason,
            "evaluations_spent": self.spent,
        })
        return fitness


# ==================== arms ====================


@dataclasses.dataclass
class ArmResult:
    name: str
    library: Library
    fitness: Fitness
    evaluations: int
    rounds: int
    stopped_because: str
    # (worst relative gain, tier total) per round. Under minimax the total can
    # *rise* while the score falls, because a library whose worst model fares
    # better is the better library even if it costs bytes overall. Recording
    # only the total would make a real improvement look like a regression.
    best_by_round: list
    trace: list[dict]

    def describe(self) -> dict:
        return {
            "arm": self.name,
            "macros": [m.shape() for m in self.library.macros],
            "origins": [m.origin for m in self.library.macros],
            "library_sha256": self.library.sha256(),
            "objective": self.fitness.objective,
            "develop_bytes": self.fitness.develop_bytes,
            "evaluations": self.evaluations,
            "rounds": self.rounds,
            "stopped_because": self.stopped_because,
            "best_by_round": self.best_by_round,
        }


def _accept_one_at_a_time(
    name: str,
    candidates,
    evaluator: Evaluator,
    baseline: Fitness,
    *,
    patience: int,
) -> ArmResult:
    """The greedy family: offer macros in order, keep each that improves.

    `candidates` is a callable taking the current library and returning macros
    to try next, so the mining, random, and LLM-greedy arms differ only in it.
    """
    library = Library()
    best = baseline
    best_by_round: list[int] = []
    rounds = 0
    idle = 0
    stopped = "candidates exhausted"

    while True:
        rounds += 1
        try:
            batch = candidates(library)
        except BudgetExhausted as exc:
            stopped = str(exc)
            break
        if not batch:
            break
        improved = False
        for macro in batch:
            grown = variation.add(library, macro)
            if grown is None:
                continue
            try:
                fitness = evaluator.measure(grown)
            except BudgetExhausted as exc:
                stopped = str(exc)
                best_by_round.append(list(best.score()))
                return ArmResult(name, library, best, evaluator.spent, rounds,
                                 stopped, best_by_round, evaluator.trace)
            if fitness.better_than(best):
                library, best, improved = grown, fitness, True
        best_by_round.append(list(best.score()))
        idle = 0 if improved else idle + 1
        if idle >= patience:
            stopped = f"no improvement for {patience} rounds"
            break
        if evaluator.remaining == 0:
            stopped = "budget exhausted"
            break
    return ArmResult(name, library, best, evaluator.spent, rounds, stopped,
                     best_by_round, evaluator.trace)


def arm_mining(evaluator, baseline, table, report, rng, patience) -> ArmResult:
    pool = mine_mod.diversify(
        mine_mod.mine(report, table, Library()), per_family=1)
    queue = list(pool)

    def next_batch(library: Library):
        if not queue:
            return []
        candidate = queue.pop(0)
        return [mine_mod.to_macro(
            candidate,
            mine_mod.suggest_name(candidate.shape, library.names()),
            "mined")]

    return _accept_one_at_a_time("mining", next_batch, evaluator, baseline,
                                 patience=patience)


def arm_random(evaluator, baseline, table, rng, patience) -> ArmResult:
    def next_batch(library: Library):
        for _ in range(20):     # a draw can be structurally unusable
            macro = variation.random_macro(table, rng, library.names())
            if macro is not None:
                return [macro]
        return []

    return _accept_one_at_a_time("random", next_batch, evaluator, baseline,
                                 patience=patience)


def arm_greedy(evaluator, baseline, table, llm, context, patience) -> ArmResult:
    def next_batch(library: Library):
        proposal = _ask(llm, context, library, table, wanted=3)
        return list(proposal.macros)

    return _accept_one_at_a_time("greedy", next_batch, evaluator, baseline,
                                 patience=patience)


def arm_one_shot(evaluator, baseline, table, llm, context) -> ArmResult:
    """One call, one library, taken whole or not at all.

    The control for "was the gain a single lucky hit?". If one shot matches a
    multi-round search, the search adds nothing.
    """
    proposal = _ask(llm, context, Library(), table, wanted=4)
    library = Library()
    for macro in proposal.macros:
        grown = variation.add(library, macro)
        if grown is not None:
            library = grown
    if not library.macros:
        return ArmResult("one_shot", Library(), baseline, evaluator.spent, 1,
                         "no usable macros proposed", [list(baseline.score())],
                         evaluator.trace)
    try:
        fitness = evaluator.measure(library)
    except BudgetExhausted as exc:
        return ArmResult("one_shot", Library(), baseline, evaluator.spent, 1,
                         str(exc), [list(baseline.score())], evaluator.trace)
    if not fitness.better_than(baseline):
        why = ("did not beat the baseline" if fitness.feasible
               else f"infeasible: {fitness.reason}")
        return ArmResult("one_shot", Library(), baseline, evaluator.spent, 1,
                         f"batch {why} ({fitness.objective - baseline.objective:+d} "
                         f"bytes)", [list(baseline.score())], evaluator.trace)
    return ArmResult("one_shot", library, fitness, evaluator.spent, 1,
                     "single shot", [list(fitness.score())], evaluator.trace)


def arm_search(
    evaluator: Evaluator,
    baseline: Fitness,
    table: OperatorTable,
    report: dict,
    rng: random.Random,
    llm,
    context: dict,
    *,
    population_size: int,
    patience: int,
    propose_every: int,
    per_round: int = 6,
) -> ArmResult:
    """Keep several libraries alive, recombine them, stop when they stop paying."""
    population: list[Scored] = [Scored(Library(), baseline, "empty", 0)]
    mined = mine_mod.diversify(mine_mod.mine(report, table, Library()), per_family=1)
    best = population[0]
    best_by_round: list[int] = []
    rounds = 0
    idle = 0
    stopped = "budget exhausted"

    while evaluator.remaining > 0:
        rounds += 1
        proposals: list[tuple[Library, str]] = []

        # Leave-one-out on the incumbent. Each is a candidate in its own right
        # — an early acceptance can become dead weight — and measuring them is
        # what makes `marginals` able to say which member is carrying the
        # library.
        for macro in best.library.macros:
            proposals.append((best.library.without(macro.name), "drop"))
        for scored in population[:2]:
            child = variation.mutate(scored.library, table, rng)
            if child is not None:
                proposals.append((child, "mutation"))
        for first in population:
            for second in population:
                if first.key >= second.key:
                    continue
                child = variation.crossover(first.library, second.library, rng)
                if child is not None:
                    proposals.append((child, "crossover"))
                    break
            else:
                continue
            break
        if mined:
            candidate = mined.pop(0)
            macro = mine_mod.to_macro(
                candidate, mine_mod.suggest_name(candidate.shape, best.library.names()),
                "mined")
            grown = variation.add(best.library, macro)
            if grown is not None:
                proposals.append((grown, "mined"))
        # Retry: a single unlucky draw must not be able to starve the round,
        # which would end the whole search at "no new candidates".
        for _ in range(20):
            macro = variation.random_macro(table, rng, best.library.names())
            if macro is None:
                continue
            grown = variation.add(best.library, macro)
            if grown is not None and grown.sha256() != best.key:
                proposals.append((grown, "random"))
                break
        if llm is not None and (rounds - 1) % propose_every == 0:
            for proposed in _ask(llm, context, best.library, table, wanted=3,
                                 history=_history(evaluator, best),
                                 marginals=marginals(best.library,
                                                     evaluator.cache)).macros:
                grown = variation.add(best.library, proposed)
                if grown is not None:
                    proposals.append((grown, "proposed"))

        fresh = []
        seen = {scored.key for scored in population}
        for library, origin in proposals:
            if library.sha256() in seen:
                continue
            seen.add(library.sha256())
            fresh.append((library, origin))
        # Bound the round so the budget buys rounds rather than one enormous
        # sweep: refinement needs feedback cycles, and an unbounded round left
        # only two or three of them inside a 30-evaluation budget.
        fresh = fresh[:per_round]
        if not fresh:
            stopped = "no new candidates could be generated"
            break

        for library, origin in fresh:
            try:
                fitness = evaluator.measure(library)
            except BudgetExhausted as exc:
                stopped = str(exc)
                fresh = []
                break
            population.append(Scored(library, fitness, origin, rounds))

        # Selection: feasible first, then objective, then the smaller library —
        # a tie broken toward fewer macros keeps branching cost down.
        population.sort(key=lambda s: (not s.fitness.feasible, s.fitness.score(),
                                       len(s.library.macros)))
        # Diversity: two members with the same macro shapes explore the same
        # neighbourhood, and a population of near-duplicates stops being a
        # population at all.
        deduped: list[Scored] = []
        shapes_seen: set[frozenset] = set()
        for scored in population:
            shapes = frozenset(scored.library.shapes())
            if shapes in shapes_seen:
                continue
            shapes_seen.add(shapes)
            deduped.append(scored)
        population = deduped[:population_size]
        if population[0].fitness.better_than(best.fitness):
            best, idle = population[0], 0
        else:
            idle += 1
        best_by_round.append(list(best.fitness.score()))

        if idle >= patience:
            stopped = f"no improvement for {patience} rounds"
            break

    return ArmResult("search", best.library, best.fitness, evaluator.spent, rounds,
                     stopped, best_by_round, evaluator.trace)


def _ask(llm, context: dict, library: Library, table: OperatorTable, *,
         wanted: int, history: list | None = None,
         marginals: dict | None = None):
    """Ask for macros, treating a malformed reply as an empty round.

    A model is untrusted input and a single bad reply must not be able to end
    a run: one `"body": null` aborted a whole five-arm arena before this. The
    failure is recorded on the proposal so an arm can report how often its
    model was unusable.
    """
    try:
        return propose_mod.propose(
            llm, context["evidence"], library, table,
            budget=context["budget"], mined=context["mined"],
            rejected=context["rejected"], wanted=wanted, history=history,
            marginals=marginals,
        )
    except Exception as exc:            # noqa: BLE001 - any bad reply, not just ours
        context.setdefault("proposal_failures", []).append(
            f"{type(exc).__name__}: {exc}"[:400])
        return propose_mod.Proposal(macros=(), raw_reply="", prompt="")


def _history(evaluator: Evaluator, best: Scored) -> list[dict]:
    """What the search has already measured, best first, for the proposer."""
    rows = sorted(evaluator.trace, key=lambda row: row["objective"])[:10]
    return [
        {"macros": row["macros"],
         "objective_delta_vs_best": row["objective"] - best.fitness.objective,
         "feasible": row["feasible"], "note": row["reason"]}
        for row in rows
    ]


# ==================== driver ====================


def run(args) -> int:
    table = OperatorTable.from_config(engine.config())
    budget = loop.budget_from(args)
    split = loop.split_from(args)
    workdir = loop.RUNS / args.run
    workdir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(workdir / "ledger.jsonl")
    out = workdir / "arms.jsonl"

    report = engine.bench(
        loop.proposer_evidence_source(split), budget, None).report
    context = {
        "evidence": mine_mod.evidence(report, table),
        "budget": budget.describe(),
        "mined": mine_mod.mine(report, table, Library()),
        "rejected": ledger.rejected(evaluate.VERSION),
    }

    shared_cache: dict[str, Fitness] = {}
    probe = Evaluator(split, budget, workdir, cap=args.evaluations,
                      cache=shared_cache, mode=args.fitness)
    baseline = probe.measure(Library(), free=True)
    baseline = dataclasses.replace(baseline, feasible=True, reason="baseline",
                                   worst_relative_gain=0.0, mode=args.fitness)
    baseline_validation = probe.validate(Library())
    print(f"baseline objective {baseline.objective} "
          f"(develop {baseline.develop_bytes}, validation "
          f"{baseline_validation.archive_bytes}, planning {baseline.planning_ms} ms)\n")
    ledger.append("arena_baseline", objective=baseline.objective,
                  budget=budget.describe(), split=split.describe(),
                  evaluations_per_arm=args.evaluations, seed=args.seed,
                  evaluator=evaluate.VERSION)

    results: list[ArmResult] = []
    for name in args.arms:
        rng = random.Random(args.seed)          # every arm gets the same stream
        evaluator = Evaluator(split, budget, workdir, cap=args.evaluations,
                              cache=shared_cache, mode=args.fitness)
        evaluator.baseline = baseline
        llm = None
        if name in ("search", "greedy", "one_shot") and not args.no_llm:
            llm = (backend.ScriptedBackend(replies=[args.dry_run_reply] * 20)
                   if args.dry_run_reply else backend.Backend(model=args.model_name))

        context["proposal_failures"] = []
        started = time.monotonic()
        if name == "search":
            result = arm_search(evaluator, baseline, table, report, rng, llm, context,
                                population_size=args.population,
                                patience=args.patience,
                                propose_every=args.propose_every,
                                per_round=args.per_round)
        elif name == "greedy":
            result = arm_greedy(evaluator, baseline, table, llm, context, args.patience)
        elif name == "one_shot":
            result = arm_one_shot(evaluator, baseline, table, llm, context)
        elif name == "mining":
            result = arm_mining(evaluator, baseline, table, report, rng, args.patience)
        elif name == "random":
            result = arm_random(evaluator, baseline, table, rng, args.patience)
        else:
            raise SystemExit(f"unknown arm {name!r}")

        validation = evaluator.validate(result.library)
        validation_delta = (validation.archive_bytes
                            - baseline_validation.archive_bytes)
        vetoed = validation_delta > int(
            evaluate.REGRESSION_TOLERANCE * baseline_validation.archive_bytes)
        if vetoed:
            print(f"  {name}: validation vetoed the result "
                  f"({validation_delta:+d} bytes); reporting the empty library")
            result = dataclasses.replace(
                result, library=Library(), fitness=baseline,
                stopped_because=f"{result.stopped_because}; validation veto")
            validation = baseline_validation
            validation_delta = 0
        record = {
            **result.describe(),
            "validation_bytes": validation.archive_bytes,
            "validation_delta_vs_baseline": validation_delta,
            "validation_vetoed": vetoed,
            "objective_delta_vs_baseline": result.fitness.objective - baseline.objective,
            "wall_s": round(time.monotonic() - started, 1),
            "seed": args.seed, "evaluator": evaluate.VERSION,
            "fitness_mode": args.fitness,
            "proposal_failures": list(context.get("proposal_failures", [])),
            "worst_relative_gain": round(result.fitness.worst_relative_gain, 6),
        }
        with out.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        ledger.append("arm", **record)
        results.append(result)

        library_path = workdir / f"library-{name}.json"
        loop.save_library(result.library, library_path, budget, split, args.run)
        print(f"{name:9s} {record['objective_delta_vs_baseline']:>+12d}  "
              f"{result.evaluations:>3d} evals  {result.rounds:>2d} rounds  "
              f"{len(result.library.macros)} macros  "
              f"[{result.stopped_because}]", flush=True)

    print(f"\nwrote {out}")
    return 0


def judge(args) -> int:
    """Measure every arm's final library on the frozen test tier, once each.

    The arena ranks arms on develop. That ranking is a hypothesis: the search
    arm won develop by 23x while *losing* validation to greedy, so develop
    order is not transfer order. This is the measurement that settles it, and
    it is the only place the test tier is read. Every read is appended to
    `runs/test-log.jsonl`.
    """
    budget = loop.budget_from(args)
    split = loop.split_from(args)
    workdir = loop.RUNS / args.run
    ledger = Ledger(workdir / "ledger.jsonl")
    rows = [json.loads(line)
            for line in (workdir / "arms.jsonl").read_text().splitlines()
            if line.strip()]

    print(f"test tier: {len(split.test)} files")
    before = evaluate.measure_corpus(split.test, budget, None)
    print(f"baseline {before.archive_bytes} bytes\n")

    verdicts = []
    for row in rows:
        name = row["arm"]
        path = workdir / f"library-{name}.json"
        library = Library.read(path)
        if not library.macros:
            print(f"{name:9s} empty library, nothing to measure")
            verdicts.append({"arm": name, "delta": 0, "bit_exact": True,
                             "macros": 0})
            continue

        check = verify_mod.roundtrip(library, list(split.test), budget, workdir)
        after = evaluate.measure_corpus(split.test, budget, path)
        serialized = len(library.canonical_bytes())
        delta = (after.archive_bytes + serialized) - before.archive_bytes
        slowdown = after.planning_wall_ms / max(before.planning_wall_ms, 1)
        improved = sum(1 for k, v in before.per_model.items()
                       if k in after.per_model and after.per_model[k] < v)
        worst = max((after.per_model[k] - v for k, v in before.per_model.items()
                     if k in after.per_model), default=0)
        verdict = {
            "arm": name, "delta": delta, "bit_exact": check.ok,
            "bit_exact_detail": check.reason, "macros": len(library.macros),
            "test_before": before.archive_bytes, "test_after": after.archive_bytes,
            "library_bytes": serialized, "planning_slowdown": round(slowdown, 3),
            "files_improved": improved, "files": len(before.per_model),
            "worst_file_regression": worst,
            "library_sha256": library.sha256(),
        }
        verdicts.append(verdict)
        print(f"{name:9s} {delta:>+12d} bytes  {improved}/{len(before.per_model)} files "
              f"better  worst +{worst}  planning {slowdown:.2f}x  "
              f"bit-exact={check.ok}", flush=True)
        ledger.append("arena_test", evaluator=evaluate.VERSION, **verdict)
        with (loop.RUNS / "test-log.jsonl").open("a") as handle:
            handle.write(json.dumps({"tier": "test", "source": "search.judge",
                                     **verdict}, sort_keys=True) + "\n")

    ranked = sorted(verdicts, key=lambda v: v["delta"])
    print("\nranking on the frozen test tier:")
    for entry in ranked:
        print(f"  {entry['arm']:9s} {entry['delta']:>+12d}")
    winner = ranked[0]
    print(f"\nbest arm: {winner['arm']} ({winner['delta']:+d} bytes)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", default="arena")
    parser.add_argument("--library", type=pathlib.Path,
                        default=loop.LIBRARIES / "learned.json")
    parser.add_argument("--develop", nargs="+", default=loop.DEVELOP)
    parser.add_argument("--validation", nargs="+", default=loop.VALIDATION)
    parser.add_argument("--test", nargs="+", default=loop.TEST)
    parser.add_argument("--arms", type=lambda v: [s for s in v.split(",") if s],
                        default=["mining", "random", "one_shot", "greedy", "search"])
    parser.add_argument("--evaluations", type=int, default=DEFAULT_EVALUATIONS,
                        help="identical evaluation budget for every arm")
    parser.add_argument("--population", type=int, default=DEFAULT_POPULATION)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    # Every round: greedy asks its model once per round, so anything larger
    # starves the search arm of proposals relative to the baseline it is being
    # compared against.
    parser.add_argument("--propose-every", type=int, default=1)
    parser.add_argument("--per-round", type=int, default=6,
                        help="candidates measured per round; smaller buys more "
                             "feedback cycles from the same budget")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fitness", choices=("sum", "minimax"), default="minimax",
                        help="how the develop tier is summarised; `sum` overfits")
    parser.add_argument("--model-name", default=backend.DEFAULT_MODEL)
    parser.add_argument("--dry-run-reply", default=None)
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-expansions", type=int, default=256)
    parser.add_argument("--max-nodes", type=int, default=12)
    parser.add_argument("--rerank-candidates", type=int, default=64)
    parser.add_argument("--rerank-blocks", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--judge", action="store_true",
                        help="skip the arena; measure existing arm libraries on "
                             "the frozen test tier")
    args = parser.parse_args(argv)
    return judge(args) if args.judge else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
