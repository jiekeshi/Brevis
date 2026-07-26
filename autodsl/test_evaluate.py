"""Tests for evaluate.py.

The gate is the only thing standing between a plausible-sounding macro and a
permanent branching cost, so its arithmetic is pinned here: a develop gain is
required, a holdout regression is refused, and a macro that changes nothing is
not "harmless".
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import engine  # noqa: E402
import evaluate  # noqa: E402
import library as lib  # noqa: E402
from test_library import zigzag_split  # noqa: E402

DEVELOP = (pathlib.Path("/probe/develop.safetensors"),)
VALIDATION = (pathlib.Path("/probe/validation.safetensors"),)
TEST = (pathlib.Path("/probe/test.safetensors"),)
SPLIT = evaluate.Split(develop=DEVELOP, validation=VALIDATION, test=TEST)
BUDGET = engine.Budget()

BASE = 100_000_000


def measurement(archive_bytes: int) -> engine.Measurement:
    return engine.Measurement(
        model="probe", archive_bytes=archive_bytes, raw_bytes=BASE * 2,
        bytecode_bytes=1000, payload_bytes=archive_bytes - 1000,
        planning_wall_ms=10, report={},
    )


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = pathlib.Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.library = lib.EMPTY
        self.candidate = self.library.with_macro(
            lib.Macro("zigzag_split", zigzag_split()))

    def decide(self, develop_after: int, holdout_after: int, planning_after: int = 10):
        baseline = (
            evaluate.CorpusResult(BASE, BASE * 2, {"d": BASE}, 10),
            evaluate.CorpusResult(BASE, BASE * 2, {"h": BASE}, 10),
        )
        sizes = iter([develop_after, holdout_after])

        def fake_bench(model, budget, macros=None, **kwargs):
            found = measurement(next(sizes))
            return dataclasses.replace(found, planning_wall_ms=planning_after)

        with mock.patch.object(engine, "bench", fake_bench):
            decision, _ = evaluate.evaluate(
                self.library, self.candidate, SPLIT, BUDGET, self.workdir,
                baseline=baseline,
            )
        return decision

    def test_a_clear_develop_gain_with_a_flat_holdout_is_accepted(self):
        decision = self.decide(BASE - 50_000, BASE)
        self.assertTrue(decision.accept, decision.reason)
        self.assertLess(decision.develop_delta, 0)

    def test_no_change_at_all_is_rejected(self):
        decision = self.decide(BASE, BASE)
        self.assertFalse(decision.accept)
        self.assertIn("needs a fall of at least", decision.reason)

    def test_a_gain_below_the_margin_is_rejected(self):
        margin = int(evaluate.ACCEPT_MARGIN * BASE)
        decision = self.decide(BASE - margin // 2, BASE)
        self.assertFalse(decision.accept)

    def test_a_gain_at_the_margin_is_accepted_once_the_library_is_paid_for(self):
        """The margin is measured after overhead, so a bare-margin gain is not
        enough — it must also cover the rules it needed."""
        margin = int(evaluate.ACCEPT_MARGIN * BASE)
        overhead = len(self.candidate.canonical_bytes()) - len(
            self.library.canonical_bytes())
        self.assertFalse(self.decide(BASE - margin, BASE).accept)
        self.assertTrue(self.decide(BASE - margin - overhead, BASE).accept)

    def test_a_develop_gain_paid_for_by_a_validation_regression_is_rejected(self):
        decision = self.decide(BASE - 50_000, BASE + 50_000)
        self.assertFalse(decision.accept)
        self.assertIn("validation regressed", decision.reason)

    def test_a_validation_gain_alone_does_not_accept(self):
        decision = self.decide(BASE, BASE - 50_000)
        self.assertFalse(decision.accept)

    def test_the_library_is_charged_against_the_gain(self):
        """A gain must survive paying for the rules that produced it."""
        overhead = len(self.candidate.canonical_bytes()) - len(
            self.library.canonical_bytes())
        self.assertGreater(overhead, 0)
        decision = self.decide(BASE - 50_000, BASE)
        self.assertEqual(-50_000 + overhead, decision.develop_delta)
        self.assertEqual(overhead, decision.holdout_delta)

    def test_a_gain_smaller_than_the_library_it_needs_is_rejected(self):
        overhead = len(self.candidate.canonical_bytes())
        decision = self.decide(BASE - overhead // 2, BASE)
        self.assertFalse(decision.accept)

    def test_a_macro_that_wrecks_one_model_is_rejected_despite_a_corpus_gain(self):
        """The locked-test run shrank the corpus by 7.36 MB while growing one
        Qwen shard by 4.43 MB. A corpus-only gate would have allowed that."""
        pair = (pathlib.Path("/probe/small.safetensors"),
                pathlib.Path("/probe/large.safetensors"))
        split = evaluate.Split(develop=pair, validation=VALIDATION, test=TEST)
        baseline = (
            evaluate.CorpusResult(
                100_000_000, 200_000_000,
                {"small.safetensors": 1_000_000, "large.safetensors": 99_000_000},
                20),   # two models, so the baseline wall is two benches
            evaluate.CorpusResult(BASE, BASE * 2, {"validation.safetensors": BASE}, 10),
        )
        # The corpus falls by 5 MB while `small` grows 10%.
        sizes = iter([1_100_000, 93_900_000, BASE])

        def fake_bench(model, budget, macros=None, **kwargs):
            return measurement(next(sizes))

        with mock.patch.object(engine, "bench", fake_bench):
            decision, _ = evaluate.evaluate(
                self.library, self.candidate, split, BUDGET, self.workdir,
                baseline=baseline,
            )
        self.assertLess(decision.develop_delta, 0, "the corpus did fall")
        self.assertFalse(decision.accept)
        self.assertIn("per-model limit", decision.reason)
        self.assertIn("small.safetensors", decision.reason)

    def test_a_macro_that_slows_planning_past_the_limit_is_rejected(self):
        slow = int(10 * evaluate.MAX_PLANNING_SLOWDOWN) + 5
        decision = self.decide(BASE - 5_000_000, BASE, planning_after=slow)
        self.assertFalse(decision.accept)
        self.assertIn("planning slowed", decision.reason)

    def test_planning_within_the_limit_still_accepts(self):
        decision = self.decide(BASE - 5_000_000, BASE, planning_after=11)
        self.assertTrue(decision.accept, decision.reason)
        self.assertAlmostEqual(1.1, decision.planning_slowdown, places=6)


class CorpusTests(unittest.TestCase):
    def test_corpus_totals_sum_every_model(self):
        sizes = iter([10, 32])

        def fake_bench(model, budget, macros=None, **kwargs):
            return measurement(next(sizes))

        with mock.patch.object(engine, "bench", fake_bench):
            result = evaluate.measure_corpus(DEVELOP + VALIDATION, BUDGET, None)
        self.assertEqual(42, result.archive_bytes)
        self.assertEqual({"develop.safetensors": 10, "validation.safetensors": 32},
                         result.per_model)


class WorstRegressionTests(unittest.TestCase):
    """A corpus total hides a single model being badly hurt."""

    def test_a_model_within_the_limit_is_not_flagged(self):
        limit = evaluate.MAX_MODEL_REGRESSION
        self.assertIsNone(evaluate.worst_regression(
            {"a": 1_000_000}, {"a": 1_000_000 + int(limit * 1_000_000 / 2)}))

    def test_a_model_past_the_limit_is_named_with_its_share(self):
        found = evaluate.worst_regression({"a": 1_000_000}, {"a": 1_010_000})
        self.assertIsNotNone(found)
        self.assertEqual(("a", 10_000), found[:2])
        self.assertAlmostEqual(0.01, found[2])

    def test_the_worst_offender_wins_when_several_regress(self):
        found = evaluate.worst_regression(
            {"a": 1_000_000, "b": 1_000_000}, {"a": 1_005_000, "b": 1_050_000})
        self.assertEqual("b", found[0])

    def test_improvements_are_never_flagged(self):
        self.assertIsNone(evaluate.worst_regression({"a": 100}, {"a": 50}))

    def test_a_model_absent_from_the_after_set_is_skipped(self):
        self.assertIsNone(evaluate.worst_regression({"a": 100}, {}))


class VerdictTests(unittest.TestCase):
    """The success criterion: unseen checkpoints must actually get smaller,
    bit-exactly, once the rule library is charged."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = pathlib.Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.library = lib.EMPTY.with_macro(lib.Macro("zigzag_split", zigzag_split()))

    def judge(self, before: int, after: int, *, bit_exact=True, library=None,
              planning=(10, 10), tier="test"):
        sizes = iter([before, after])
        walls = iter(planning)

        def fake_bench(model, budget, macros=None, **kwargs):
            found = measurement(next(sizes))
            return dataclasses.replace(found, planning_wall_ms=next(walls))

        with mock.patch.object(engine, "bench", fake_bench):
            return evaluate.verdict(
                self.library if library is None else library,
                SPLIT, BUDGET, self.workdir, bit_exact=bit_exact, tier=tier,
                test_log=self.workdir / "test-log.jsonl",
            )

    def test_a_real_gain_on_unseen_checkpoints_is_success(self):
        result = self.judge(BASE, BASE - 50_000)
        self.assertTrue(result.success, result.reason)
        self.assertEqual("test", result.tier)

    def test_every_touch_of_the_test_tier_is_logged_and_counted(self):
        """Reading a locked set twice is multiple testing; make it visible."""
        first = self.judge(BASE, BASE - 50_000)
        self.assertEqual(0, first.prior_measurements_of_this_tier)
        second = self.judge(BASE, BASE - 50_000)
        self.assertEqual(1, second.prior_measurements_of_this_tier)
        logged = (self.workdir / "test-log.jsonl").read_text().splitlines()
        self.assertEqual(2, len(logged))

    def test_an_empty_tier_is_reported_rather_than_silently_passing(self):
        empty = evaluate.Split(develop=DEVELOP, validation=VALIDATION, test=())
        with mock.patch.object(engine, "bench") as bench:
            result = evaluate.verdict(self.library, empty, BUDGET, self.workdir,
                                      bit_exact=True)
        self.assertFalse(result.success)
        self.assertIn("empty", result.reason)
        bench.assert_not_called()

    def test_no_gain_on_unseen_checkpoints_is_failure(self):
        result = self.judge(BASE, BASE)
        self.assertFalse(result.success)
        self.assertIn("did not shrink", result.reason)

    def test_a_gain_smaller_than_the_library_is_failure(self):
        overhead = len(self.library.canonical_bytes())
        result = self.judge(BASE, BASE - overhead // 2)
        self.assertFalse(result.success)

    def test_an_unverified_library_is_failure_however_small_the_archive(self):
        result = self.judge(BASE, BASE - 50_000_000, bit_exact=False)
        self.assertFalse(result.success)
        self.assertIn("bit-exactness", result.reason)

    def test_an_empty_library_is_failure_not_a_tie(self):
        result = self.judge(BASE, BASE, library=lib.EMPTY)
        self.assertFalse(result.success)
        self.assertIn("nothing was learned", result.reason)

    def test_a_gain_bought_with_unacceptable_planning_time_is_failure(self):
        result = self.judge(BASE, BASE - 50_000, planning=(10, 100))
        self.assertFalse(result.success)
        self.assertIn("planning slowed", result.reason)


class LabelTests(unittest.TestCase):
    def test_a_cached_checkpoint_is_named_by_its_repository(self):
        """Every cache entry is `<repo>/<revision>/model.safetensors`, so the
        file name alone identifies nothing."""
        self.assertEqual(
            "google-bert__bert-base-uncased/model.safetensors",
            engine.label(pathlib.Path(
                "/c/google-bert__bert-base-uncased/86b5e09/model.safetensors")),
        )

    def test_a_shallow_path_falls_back_to_the_file_name(self):
        self.assertEqual("m.safetensors", engine.label(pathlib.Path("/m.safetensors")))


class SplitTests(unittest.TestCase):
    def test_split_names_the_role_of_every_tier(self):
        roles = SPLIT.describe()["roles"]
        self.assertIn("not a test set", roles["validation"])
        self.assertIn("never by mining", roles["test"])
        self.assertEqual(DEVELOP + VALIDATION + TEST, SPLIT.all())

    def test_acceptance_never_reads_the_test_tier(self):
        """The veto is the validation tier; test must stay untouched."""
        seen = []

        def fake_bench(model, budget, macros=None, **kwargs):
            seen.append(model)
            return measurement(BASE)

        with mock.patch.object(engine, "bench", fake_bench), \
             tempfile.TemporaryDirectory() as tmp:
            evaluate.evaluate(
                lib.EMPTY, lib.EMPTY.with_macro(lib.Macro("m", zigzag_split())),
                SPLIT, BUDGET, pathlib.Path(tmp),
            )
        self.assertNotIn(TEST[0], seen)
        self.assertIn(VALIDATION[0], seen)


if __name__ == "__main__":
    unittest.main()
