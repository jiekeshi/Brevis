"""Tests for loop.py.

What is pinned here is containment: the proposer must never be shown a
checkpoint that a generalization claim will later rest on. Everything else in
this module is orchestration and is covered by the tests of the parts it calls.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import evaluate  # noqa: E402
import loop  # noqa: E402

DEVELOP = (pathlib.Path("/probe/d1.safetensors"), pathlib.Path("/probe/d2.safetensors"))
VALIDATION = (pathlib.Path("/probe/v1.safetensors"),)
TEST = (pathlib.Path("/probe/t1.safetensors"), pathlib.Path("/probe/t2.safetensors"))
SPLIT = evaluate.Split(develop=DEVELOP, validation=VALIDATION, test=TEST)


class ContainmentTests(unittest.TestCase):
    def test_the_proposer_is_shown_a_develop_checkpoint(self):
        self.assertIn(loop.proposer_evidence_source(SPLIT), DEVELOP)

    def test_the_proposer_is_never_shown_validation_or_test(self):
        source = loop.proposer_evidence_source(SPLIT)
        self.assertNotIn(source, VALIDATION)
        self.assertNotIn(source, TEST)

    def test_containment_holds_whatever_the_tiers_contain(self):
        """Reordering or resizing the tiers must not leak one into the prompt."""
        for develop in (DEVELOP, DEVELOP[::-1], DEVELOP[:1]):
            split = evaluate.Split(develop=develop, validation=VALIDATION, test=TEST)
            self.assertIn(loop.proposer_evidence_source(split), develop)


class TierTests(unittest.TestCase):
    def test_the_three_default_tiers_are_disjoint(self):
        tiers = [set(loop.DEVELOP), set(loop.VALIDATION), set(loop.TEST)]
        for i, first in enumerate(tiers):
            for second in tiers[i + 1:]:
                self.assertEqual(set(), first & second)

    def test_absent_test_checkpoints_are_skipped_rather_than_fatal(self):
        """A claim then rests on fewer models, which the verdict records."""
        self.assertEqual([], loop.find_models(["no-such-model"], required=False))
        with self.assertRaises(SystemExit):
            loop.find_models(["no-such-model"])


if __name__ == "__main__":
    unittest.main()
