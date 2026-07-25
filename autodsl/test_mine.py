"""Tests for mine.py.

Mining decides what the loop even considers, so the properties that matter are
that it weights by bytes rather than by count, that it never emits a body the
engine would reject, and that it does not re-offer what is already known.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import library as lib  # noqa: E402
import mine  # noqa: E402
from test_library import table, zigzag_split  # noqa: E402


def tree(op, children=(), terminal=False):
    return {"op": op, "params_u32": 0, "terminal": terminal,
            "children": [dict(c) for c in children]}


ZIGZAG_SPLIT = tree("zigzag", [tree("split_field", [
    tree("rans", terminal=True), tree("raw", terminal=True)])])
GRAY_SPLIT = tree("gray", [tree("split_field", [
    tree("rans", terminal=True), tree("raw", terminal=True)])])


def block(program_tree, encoded, raw=1000, program="p"):
    return {
        "program_tree": program_tree,
        "encoded_bytes_without_frame_headers": encoded,
        "raw_bytes": raw,
        "program": program,
        "program_bytecode_bytes": 20,
    }


def report(blocks, tensors=None):
    return {
        "input": "probe.safetensors",
        "raw_bytes": sum(b["raw_bytes"] for b in blocks),
        "projected_archive_bytes": sum(
            b["encoded_bytes_without_frame_headers"] for b in blocks) or 1,
        "blocks": blocks,
        "tensors": tensors if tensors is not None else [],
        # Mirrors the engine: `search_options_applied` is a bool flag and
        # the resolved configuration lives under `search`.
        "search_options_applied": True,
        "search": {"max_expansions": 256},
    }


class GeneralizationTests(unittest.TestCase):
    def test_root_is_always_kept_and_leaves_may_become_holes(self):
        shapes = {lib.render(b) for b in mine.generalizations(ZIGZAG_SPLIT, table())}
        self.assertIn("zigzag(split_field(?,?))", shapes)
        self.assertIn("zigzag(split_field(rans,raw))", shapes)
        self.assertIn("zigzag(?)", shapes)
        for shape in shapes:
            self.assertTrue(shape.startswith("zigzag"), shape)

    def test_params_are_generalized_to_auto(self):
        for body in mine.generalizations(ZIGZAG_SPLIT, table()):
            for node in lib.walk(body):
                self.assertEqual("auto", node["params"])

    def test_every_generalization_validates(self):
        for body in mine.generalizations(ZIGZAG_SPLIT, table()):
            lib.validate_body(body, table())

    def test_a_width_dependent_root_yields_nothing(self):
        planes = tree("bit_plane", [tree("raw", terminal=True)])
        self.assertEqual([], list(mine.generalizations(planes, table())))


class MineTests(unittest.TestCase):
    def test_candidates_rank_by_bytes_not_by_block_count(self):
        blocks = [block(GRAY_SPLIT, 10) for _ in range(50)]
        blocks += [block(ZIGZAG_SPLIT, 5_000)]
        candidates = mine.mine(report(blocks), table(), lib.EMPTY)
        self.assertTrue(candidates[0].shape.startswith("zigzag"),
                        f"got {candidates[0].shape}")

    def test_single_operator_bodies_are_not_offered(self):
        candidates = mine.mine(report([block(ZIGZAG_SPLIT, 100)]), table(), lib.EMPTY)
        self.assertTrue(all(c.node_count >= mine.MIN_BODY_NODES for c in candidates))
        self.assertNotIn("zigzag(?)", {c.shape for c in candidates})

    def test_shapes_already_in_the_library_are_skipped(self):
        library = lib.Library(macros=(lib.Macro("zigzag_split", zigzag_split()),))
        shapes = {c.shape for c in mine.mine(
            report([block(ZIGZAG_SPLIT, 100)]), table(), library)}
        self.assertNotIn("zigzag(split_field(?,?))", shapes)

    def test_stats_on_a_candidate_match_the_library_model(self):
        candidate = next(
            c for c in mine.mine(report([block(ZIGZAG_SPLIT, 100)]), table(), lib.EMPTY)
            if c.shape == "zigzag(split_field(?,?))"
        )
        self.assertEqual((2, 2, 2), (candidate.node_count, candidate.hole_count,
                                     candidate.transform_depth))

    def test_a_block_without_a_program_tree_is_skipped(self):
        blocks = [{"encoded_bytes_without_frame_headers": 1, "raw_bytes": 1,
                   "program": "raw", "program_bytecode_bytes": 1}]
        self.assertEqual([], mine.mine(report(blocks), table(), lib.EMPTY))


class DiversityTests(unittest.TestCase):
    """`generalizations` emits one candidate per way of pinning terminals, so
    a single winning tree can fill every slot the gate has time to run."""

    def test_family_holes_every_terminal(self):
        self.assertEqual(
            "zigzag(split_field(?,?))",
            mine.family(lib.node("zigzag", [lib.node("split_field", [
                lib.node("rans"), lib.node("raw")])])),
        )

    def test_pinning_terminals_does_not_change_the_family(self):
        skeleton = lib.node("zigzag", [lib.node("split_field", [lib.HOLE, lib.HOLE])])
        pinned = lib.node("zigzag", [lib.node("split_field", [
            lib.node("rans"), lib.HOLE])])
        self.assertEqual(mine.family(skeleton), mine.family(pinned))

    def test_a_different_root_is_a_different_family(self):
        self.assertNotEqual(
            mine.family(lib.node("zigzag", [lib.HOLE])),
            mine.family(lib.node("gray", [lib.HOLE])),
        )

    def test_diversify_keeps_the_highest_earner_per_family(self):
        candidates = mine.mine(
            report([block(ZIGZAG_SPLIT, 5000), block(GRAY_SPLIT, 10)]),
            table(), lib.EMPTY)
        picked = mine.diversify(candidates, per_family=1)
        self.assertEqual(len(picked), len({mine.family(c.body) for c in picked}))
        self.assertEqual(candidates[0].shape, picked[0].shape)

    def test_diversify_can_keep_more_than_one_per_family(self):
        candidates = mine.mine(report([block(ZIGZAG_SPLIT, 5000)]), table(), lib.EMPTY)
        self.assertGreater(len(mine.diversify(candidates, per_family=2)),
                           len(mine.diversify(candidates, per_family=1)))


class NameTests(unittest.TestCase):
    def test_names_derive_from_the_shape_and_avoid_collisions(self):
        first = mine.suggest_name("zigzag(split_field(?,?))", set())
        self.assertEqual("zigzag_split_field_h_h", first)
        self.assertNotEqual(first, mine.suggest_name("zigzag(split_field(?,?))",
                                                     {first}))

    def test_generated_names_pass_library_validation(self):
        name = mine.suggest_name("zigzag(split_field(?,?))", set())
        lib.validate_macro(lib.Macro(name, zigzag_split()), table())


class EvidenceTests(unittest.TestCase):
    def test_evidence_reports_budget_saturation_and_bytecode_share(self):
        tensors = [
            {"name": "a", "dtype": "BF16", "shape": [4], "raw_bytes": 1000,
             "encoded_bytes_without_frame_headers": 800, "program": "p",
             "search_expansions": 256},
            {"name": "b", "dtype": "F32", "shape": [4], "raw_bytes": 1000,
             "encoded_bytes_without_frame_headers": 500, "program": "q",
             "search_expansions": 12},
        ]
        blob = mine.evidence(report([block(ZIGZAG_SPLIT, 1300)], tensors), table())
        self.assertEqual("1/2", blob["search_budget_saturated_tensors"])
        self.assertIn("BF16", blob["by_dtype"])
        self.assertGreater(blob["bytecode_share_of_archive"], 0)
        self.assertEqual("a", blob["worst_compressing_tensors"][0]["name"])

    def test_terminal_counts_come_from_the_program_trees(self):
        blob = mine.evidence(report([block(ZIGZAG_SPLIT, 10)]), table())
        self.assertEqual({"rans": 1, "raw": 1}, blob["terminal_node_counts"])


if __name__ == "__main__":
    unittest.main()
