"""Tests for verify.py.

The gates run in a fixed order for a reason: the cheap structural check must
refuse before anything spawns a process, and the bit-exactness check must be
the last word. These tests pin that order and pin that a failure at any stage
stops the rest.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import engine  # noqa: E402
import library as lib  # noqa: E402
import verify  # noqa: E402
from test_library import table, zigzag_split  # noqa: E402

MODELS = [pathlib.Path("/probe/a.safetensors")]
BUDGET = engine.Budget()


def macro(name="zigzag_split", body=None) -> lib.Macro:
    return lib.Macro(name, body or zigzag_split())


def report_with(selected: dict[str, int]) -> dict:
    tensors = []
    for name, count in selected.items():
        tensors.extend([{"macros_used": [name]}] * count)
    return {"tensors": tensors or [{"macros_used": []}]}


class StructuralTests(unittest.TestCase):
    def test_a_duplicate_name_is_refused(self):
        library = lib.Library(macros=(macro(),))
        verdict = verify.structural(macro(), table(), library)
        self.assertFalse(verdict.ok)
        self.assertIn("already used", verdict.reason)

    def test_a_duplicate_body_under_a_new_name_is_refused(self):
        library = lib.Library(macros=(macro("first"),))
        verdict = verify.structural(macro("second"), table(), library)
        self.assertFalse(verdict.ok)
        self.assertIn("already in library", verdict.reason)

    def test_a_malformed_body_is_refused_with_the_engine_s_reason(self):
        bad = macro("bad", lib.node("split_field", [lib.HOLE]))
        verdict = verify.structural(bad, table(), lib.EMPTY)
        self.assertFalse(verdict.ok)
        self.assertIn("takes 2 children", verdict.reason)


class FiresTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = pathlib.Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def _fires(self, selected):
        def fake_bench(model, budget, macros=None, **kwargs):
            return engine.Measurement("m", 1, 2, 0, 1, 1, report_with(selected))

        with mock.patch.object(engine, "bench", fake_bench):
            return verify.fires(
                lib.Library(macros=(macro(),)), macro(), MODELS, BUDGET, self.workdir
            )

    def test_a_macro_nothing_selects_is_refused(self):
        verdict = self._fires({})
        self.assertFalse(verdict.ok)
        self.assertIn("dilute the budget", verdict.reason)

    def test_a_macro_some_tensor_selects_passes(self):
        self.assertTrue(self._fires({"zigzag_split": 3}).ok)

    def test_selection_by_a_different_macro_does_not_count(self):
        self.assertFalse(self._fires({"other_macro": 9}).ok)

    def test_macros_selected_counts_tensors_per_name(self):
        counts = verify.macros_selected(report_with({"a": 2, "b": 1}))
        self.assertEqual({"a": 2, "b": 1}, counts)


class OrderingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.workdir = pathlib.Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_a_structural_failure_never_starts_a_process(self):
        bad = macro("bad", lib.node("split_field", [lib.HOLE]))
        with mock.patch.object(engine, "config") as config, \
             mock.patch.object(engine, "bench") as bench, \
             mock.patch.object(engine, "compress_and_verify") as roundtrip:
            verdict = verify.check(bad, lib.EMPTY, table(), MODELS, BUDGET,
                                   workdir=self.workdir)
        self.assertFalse(verdict.ok)
        config.assert_not_called()
        bench.assert_not_called()
        roundtrip.assert_not_called()

    def test_an_inert_macro_is_refused_before_a_real_archive_is_written(self):
        with mock.patch.object(engine, "config", return_value={"macro_count": 1}), \
             mock.patch.object(
                 engine, "bench",
                 return_value=engine.Measurement("m", 1, 2, 0, 1, 1, report_with({}))), \
             mock.patch.object(engine, "compress_and_verify") as roundtrip:
            verdict = verify.check(macro(), lib.EMPTY, table(), MODELS, BUDGET,
                                   workdir=self.workdir)
        self.assertFalse(verdict.ok)
        roundtrip.assert_not_called()

    def test_a_failed_roundtrip_refuses_even_when_everything_else_passed(self):
        with mock.patch.object(engine, "config", return_value={"macro_count": 1}), \
             mock.patch.object(
                 engine, "bench",
                 return_value=engine.Measurement(
                     "m", 1, 2, 0, 1, 1, report_with({"zigzag_split": 4}))), \
             mock.patch.object(engine, "compress_and_verify",
                               side_effect=engine.EngineError("decoded 3/4 tensors")):
            verdict = verify.check(macro(), lib.EMPTY, table(), MODELS, BUDGET,
                                   workdir=self.workdir)
        self.assertFalse(verdict.ok)
        self.assertIn("roundtrip failed", verdict.reason)

    def test_all_gates_passing_accepts(self):
        with mock.patch.object(engine, "config", return_value={"macro_count": 1}), \
             mock.patch.object(
                 engine, "bench",
                 return_value=engine.Measurement(
                     "m", 1, 2, 0, 1, 1, report_with({"zigzag_split": 4}))), \
             mock.patch.object(engine, "compress_and_verify", return_value=1234):
            verdict = verify.check(macro(), lib.EMPTY, table(), MODELS, BUDGET,
                                   workdir=self.workdir)
        self.assertTrue(verdict.ok, verdict.reason)

    def test_an_engine_that_drops_a_macro_is_refused(self):
        with mock.patch.object(engine, "config", return_value={"macro_count": 0}), \
             mock.patch.object(engine, "bench") as bench:
            verdict = verify.check(macro(), lib.EMPTY, table(), MODELS, BUDGET,
                                   workdir=self.workdir)
        self.assertFalse(verdict.ok)
        self.assertIn("loaded 0 of 1", verdict.reason)
        bench.assert_not_called()


if __name__ == "__main__":
    unittest.main()
