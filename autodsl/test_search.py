"""Tests for search.py.

The claim this module exists to support is "a multi-round search beats
one-shot, greedy, mining and random at the same compute budget". That claim is
worthless if the budget is not actually the same, so most of these tests are
about accounting: what costs an evaluation, what does not, and that no arm can
exceed its cap.

The engine is stubbed throughout. Whether a macro really helps is a question
for the arena; whether the arena counts honestly is a question for here.
"""

from __future__ import annotations

import pathlib
import random
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import engine  # noqa: E402
import evaluate  # noqa: E402
import library as lib  # noqa: E402
import search  # noqa: E402
import variation  # noqa: E402
from test_library import table, zigzag_split  # noqa: E402

DEVELOP = (pathlib.Path("/probe/d.safetensors"),)
VALIDATION = (pathlib.Path("/probe/v.safetensors"),)
SPLIT = evaluate.Split(develop=DEVELOP, validation=VALIDATION, test=())
BUDGET = engine.Budget()
BASE = 1_000_000


def measurement(archive_bytes: int, planning: int = 100) -> engine.Measurement:
    return engine.Measurement(
        model="probe", archive_bytes=archive_bytes, raw_bytes=BASE * 2,
        bytecode_bytes=100, payload_bytes=archive_bytes - 100,
        planning_wall_ms=planning, report={},
    )


class _Engine:
    """A stub whose archive size falls with the number of macros in play."""

    def __init__(self, per_macro: int = 10_000, planning: int = 100):
        self.per_macro = per_macro
        self.planning = planning
        self.calls = 0
        self.macro_count = 0

    def bench(self, model, budget, macros=None, **kwargs):
        self.calls += 1
        return measurement(BASE - self.per_macro * self.macro_count, self.planning)


class _Evaluator(search.Evaluator):
    """Counts real macros so the stub can respond to library size."""

    def __init__(self, stub, **kwargs):
        super().__init__(SPLIT, BUDGET, kwargs.pop("workdir"), kwargs.pop("cap"),
                         cache=kwargs.pop("cache", None))
        self.stub = stub

    def measure(self, library, *, free=False):
        self.stub.macro_count = len(library.macros)
        return super().measure(library, free=free)


class FitnessTests(unittest.TestCase):
    def fitness(self, objective, feasible=True):
        return search.Fitness(objective, objective, 0, feasible, "", {})

    def test_a_smaller_objective_wins(self):
        self.assertTrue(self.fitness(10).better_than(self.fitness(20)))
        self.assertFalse(self.fitness(20).better_than(self.fitness(10)))

    def test_feasibility_beats_a_smaller_objective(self):
        """An infeasible library is not a better library, however small."""
        self.assertFalse(self.fitness(1, feasible=False).better_than(self.fitness(999)))
        self.assertTrue(self.fitness(999).better_than(self.fitness(1, feasible=False)))

    def test_anything_feasible_beats_nothing(self):
        self.assertTrue(self.fitness(10).better_than(None))
        self.assertFalse(self.fitness(10, feasible=False).better_than(None))

    def test_an_equal_objective_does_not_displace_the_incumbent(self):
        self.assertFalse(self.fitness(10).better_than(self.fitness(10)))


class EvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = pathlib.Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.stub = _Engine()
        patcher = mock.patch.object(engine, "bench", self.stub.bench)
        patcher.start()
        self.addCleanup(patcher.stop)

    def evaluator(self, cap=5, cache=None):
        return _Evaluator(self.stub, workdir=self.workdir, cap=cap, cache=cache)

    def macro_library(self, n: int) -> lib.Library:
        library = lib.EMPTY
        for index in range(n):
            library = library.with_macro(lib.Macro(
                f"m{index}",
                lib.node("zigzag" if index % 2 else "gray",
                         [lib.node("split_field", [lib.HOLE, lib.HOLE])
                          if index < 2 else lib.HOLE])))
        return library

    def test_a_measurement_costs_one_evaluation(self):
        evaluator = self.evaluator()
        evaluator.measure(self.macro_library(1))
        self.assertEqual(1, evaluator.spent)
        self.assertEqual(4, evaluator.remaining)

    def test_a_cache_hit_is_free(self):
        evaluator = self.evaluator()
        library = self.macro_library(1)
        evaluator.measure(library)
        evaluator.measure(library)
        self.assertEqual(1, evaluator.spent)

    def test_the_cache_is_shared_so_arms_are_not_charged_twice(self):
        shared: dict = {}
        first = self.evaluator(cache=shared)
        first.measure(self.macro_library(1))
        second = self.evaluator(cache=shared)
        second.measure(self.macro_library(1))
        self.assertEqual(0, second.spent)

    def test_the_cap_is_enforced(self):
        evaluator = self.evaluator(cap=2)
        evaluator.measure(self.macro_library(1))
        evaluator.measure(self.macro_library(2))
        with self.assertRaises(search.BudgetExhausted):
            evaluator.measure(self.macro_library(3))

    def test_a_free_measurement_does_not_consume_the_cap(self):
        """The baseline is not part of any arm's search effort."""
        evaluator = self.evaluator(cap=1)
        evaluator.measure(lib.EMPTY, free=True)
        self.assertEqual(0, evaluator.spent)
        evaluator.measure(self.macro_library(1))
        self.assertEqual(1, evaluator.spent)

    def test_the_library_is_charged_into_the_objective(self):
        evaluator = self.evaluator()
        library = self.macro_library(1)
        fitness = evaluator.measure(library)
        self.assertEqual(fitness.develop_bytes + len(library.canonical_bytes()),
                         fitness.objective)

    def test_slow_planning_makes_a_library_infeasible(self):
        evaluator = self.evaluator()
        evaluator.baseline = search.Fitness(BASE, BASE, 100, True, "", {})
        self.stub.planning = int(100 * evaluate.MAX_PLANNING_SLOWDOWN) + 50
        self.assertFalse(evaluator.measure(self.macro_library(1)).feasible)

    def test_wrecking_one_model_makes_a_library_infeasible(self):
        evaluator = self.evaluator()
        evaluator.baseline = search.Fitness(
            BASE, BASE, 100, True, "", {"d.safetensors": 1_000})
        self.assertFalse(evaluator.measure(self.macro_library(1)).feasible)

    def test_no_temporary_library_files_are_left_behind(self):
        evaluator = self.evaluator()
        evaluator.measure(self.macro_library(1))
        self.assertEqual([], list(self.workdir.glob("lib-*.json")))


class ArmTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = pathlib.Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.stub = _Engine()
        patcher = mock.patch.object(engine, "bench", self.stub.bench)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.baseline = search.Fitness(BASE, BASE, 200, True, "baseline",
                                       {"d.safetensors": BASE})

    def evaluator(self, cap):
        evaluator = _Evaluator(self.stub, workdir=self.workdir, cap=cap)
        evaluator.baseline = self.baseline
        return evaluator

    def test_a_greedy_arm_stops_when_nothing_improves(self):
        """Patience, not an exhausted budget, should end a converged arm."""
        self.stub.per_macro = 0          # nothing ever helps
        evaluator = self.evaluator(cap=100)
        result = search.arm_random(evaluator, self.baseline, table(),
                                   random.Random(1), patience=2)
        self.assertIn("no improvement", result.stopped_because)
        self.assertLess(result.evaluations, 100)

    def test_no_arm_can_exceed_its_budget(self):
        for cap in (1, 3, 7):
            evaluator = self.evaluator(cap=cap)
            result = search.arm_random(evaluator, self.baseline, table(),
                                       random.Random(2), patience=50)
            self.assertLessEqual(result.evaluations, cap, f"cap {cap}")

    def test_the_population_arm_also_respects_the_cap(self):
        evaluator = self.evaluator(cap=6)
        result = search.arm_search(
            evaluator, self.baseline, table(), {"blocks": [], "tensors": []},
            random.Random(3), None, {}, population_size=3, patience=50,
            propose_every=99)
        self.assertLessEqual(result.evaluations, 6)

    def test_the_population_arm_keeps_a_population_and_improves(self):
        evaluator = self.evaluator(cap=30)
        result = search.arm_search(
            evaluator, self.baseline, table(), {"blocks": [], "tensors": []},
            random.Random(4), None, {}, population_size=4, patience=4,
            propose_every=99)
        self.assertLess(result.fitness.objective, self.baseline.objective)
        self.assertGreater(result.rounds, 1, "a single round is not a search")

    def test_the_population_arm_reports_its_trajectory(self):
        """`best_by_round` is what distinguishes a search from a lucky hit."""
        evaluator = self.evaluator(cap=30)
        result = search.arm_search(
            evaluator, self.baseline, table(), {"blocks": [], "tensors": []},
            random.Random(5), None, {}, population_size=4, patience=4,
            propose_every=99)
        self.assertEqual(result.rounds, len(result.best_by_round))
        self.assertEqual(sorted(result.best_by_round, reverse=True),
                         result.best_by_round, "the best must never get worse")

    def test_an_arm_that_finds_nothing_returns_the_empty_library(self):
        self.stub.per_macro = -50_000     # every macro hurts
        evaluator = self.evaluator(cap=10)
        result = search.arm_random(evaluator, self.baseline, table(),
                                   random.Random(6), patience=2)
        self.assertEqual(0, len(result.library.macros))
        self.assertEqual(self.baseline.objective, result.fitness.objective)

    def test_a_mining_arm_runs_without_a_model(self):
        report = {"blocks": [], "tensors": []}
        evaluator = self.evaluator(cap=5)
        result = search.arm_mining(evaluator, self.baseline, table(), report,
                                   random.Random(7), patience=2)
        self.assertEqual("mining", result.name)


if __name__ == "__main__":
    unittest.main()
