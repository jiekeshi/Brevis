"""Tests for ledger.py.

The ledger is the loop's evidence. Two properties matter: it is append-only,
and its `rejected` view is what stops the proposer being asked to rediscover
the same dead end.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ledger import Ledger  # noqa: E402


class LedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmp.name) / "nested" / "ledger.jsonl"
        self.addCleanup(self.tmp.cleanup)
        self.ledger = Ledger(self.path)

    def test_a_fresh_ledger_is_empty_and_creates_its_directory(self):
        self.assertTrue(self.path.parent.exists())
        self.assertEqual(0, self.ledger.count())
        self.assertEqual([], list(self.ledger.entries()))

    def test_entries_are_appended_with_increasing_sequence_numbers(self):
        self.ledger.append("proposed", name="a")
        self.ledger.append("accepted", name="a")
        self.assertEqual([0, 1], [e["sequence"] for e in self.ledger.entries()])

    def test_appending_never_rewrites_earlier_lines(self):
        self.ledger.append("proposed", name="a")
        first = self.path.read_text()
        self.ledger.append("rejected", name="b", shape="x", reason="r")
        self.assertTrue(self.path.read_text().startswith(first))

    def test_rejected_returns_one_entry_per_shape_oldest_first(self):
        self.ledger.append("rejected", name="a", shape="gray(?)", reason="inert")
        self.ledger.append("rejected", name="b", shape="rle(?,?)", reason="no gain")
        self.ledger.append("rejected", name="c", shape="gray(?)", reason="inert again")
        self.assertEqual(
            [{"shape": "gray(?)", "reason": "inert"},
             {"shape": "rle(?,?)", "reason": "no gain"}],
            self.ledger.rejected(),
        )

    def test_accepted_names_reflect_later_drops(self):
        self.ledger.append("accepted", name="a")
        self.ledger.append("accepted", name="b")
        self.ledger.append("dropped", name="a", reason="superseded")
        self.assertEqual(["b"], self.ledger.accepted_names())

    def test_summary_counts_by_kind(self):
        self.ledger.append("proposed", name="a")
        self.ledger.append("rejected", name="a", shape="s", reason="r")
        self.ledger.append("rejected", name="b", shape="t", reason="r")
        self.assertEqual({"proposed": 1, "rejected": 2}, self.ledger.summary())

    def test_a_blank_line_does_not_break_reading(self):
        self.ledger.append("proposed", name="a")
        with self.path.open("a") as handle:
            handle.write("\n")
        self.assertEqual(1, self.ledger.count())

    def test_entries_are_valid_json_with_sorted_keys(self):
        self.ledger.append("accepted", name="z", shape="s", detail={"b": 1, "a": 2})
        line = self.path.read_text().strip()
        self.assertEqual(json.loads(line)["name"], "z")
        self.assertEqual(line, json.dumps(json.loads(line), sort_keys=True))


if __name__ == "__main__":
    unittest.main()
