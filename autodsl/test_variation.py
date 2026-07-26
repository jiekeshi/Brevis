"""Tests for variation.py.

Every operator here feeds a candidate straight into an expensive evaluation, so
what matters is that it never emits something the engine would refuse, never
emits a duplicate of what it was given, and is reproducible from a seed.
"""

from __future__ import annotations

import pathlib
import random
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import library as lib  # noqa: E402
import variation  # noqa: E402
from test_library import table, zigzag_split  # noqa: E402


def rng(seed: int = 7) -> random.Random:
    return random.Random(seed)


def library_of(*macros: lib.Macro) -> lib.Library:
    return lib.Library(macros=macros)


class GenerationTests(unittest.TestCase):
    def test_every_random_body_is_one_the_engine_would_accept(self):
        stream = rng()
        for _ in range(300):
            lib.validate_body(variation.random_body(table(), stream), table())

    def test_random_bodies_never_use_a_width_dependent_operator(self):
        stream = rng()
        for _ in range(300):
            for node in lib.walk(variation.random_body(table(), stream)):
                self.assertTrue(table().by_name[node["op"]].usable_in_macro_body)

    def test_random_macros_respect_the_size_and_depth_caps(self):
        stream = rng()
        made = 0
        for _ in range(300):
            macro = variation.random_macro(table(), stream, set())
            if macro is None:
                continue
            made += 1
            stats = lib.body_stats(macro.body, table())
            self.assertGreaterEqual(stats.nodes, 2)
            self.assertLessEqual(stats.nodes, variation.MAX_GENERATED_NODES)
            self.assertLessEqual(stats.transform_depth, variation.MAX_GENERATED_DEPTH)
        self.assertGreater(made, 0, "the sampler produced nothing usable at all")

    def test_generation_is_reproducible_from_a_seed(self):
        first = [variation.random_body(table(), rng(11)) for _ in range(3)]
        second = [variation.random_body(table(), rng(11)) for _ in range(3)]
        self.assertEqual(first, second)

    def test_a_generated_name_does_not_collide(self):
        macro = variation.random_macro(table(), rng(), {"random_1", "random_2"})
        if macro is not None:
            self.assertNotIn(macro.name, {"random_1", "random_2"})


class BodyMutationTests(unittest.TestCase):
    def test_pinning_a_hole_closes_one_branch(self):
        body = variation.pin_a_hole(zigzag_split(), table(), rng())
        before = lib.body_stats(zigzag_split(), table())
        after = lib.body_stats(body, table())
        self.assertEqual(before.holes - 1, after.holes)
        self.assertEqual(before.terminals + 1, after.terminals)

    def test_pinning_fails_cleanly_when_there_is_no_hole(self):
        closed = lib.node("zigzag", [lib.node("rans")])
        self.assertIsNone(variation.pin_a_hole(closed, table(), rng()))

    def test_opening_a_leaf_generalises_it(self):
        closed = lib.node("zigzag", [lib.node("rans")])
        body = variation.open_a_leaf(closed, table(), rng())
        self.assertEqual("zigzag(?)", lib.render(body))

    def test_opening_fails_cleanly_when_every_leaf_is_a_hole(self):
        self.assertIsNone(variation.open_a_leaf(zigzag_split(), table(), rng()))

    def test_swapping_keeps_the_arity(self):
        body = variation.swap_an_operator(zigzag_split(), table(), rng())
        self.assertIsNotNone(body)
        lib.validate_body(body, table())
        self.assertNotEqual(lib.render(zigzag_split()), lib.render(body))

    def test_wrapping_adds_exactly_one_transform_layer(self):
        body = variation.wrap_in_a_transform(zigzag_split(), table(), rng())
        before = lib.body_stats(zigzag_split(), table())
        after = lib.body_stats(body, table())
        self.assertEqual(before.transform_depth + 1, after.transform_depth)
        lib.validate_body(body, table())


class MutateTests(unittest.TestCase):
    def setUp(self):
        self.library = library_of(lib.Macro("base", zigzag_split()))

    def test_mutating_an_empty_library_is_not_an_error(self):
        self.assertIsNone(variation.mutate(lib.EMPTY, table(), rng()))

    def test_a_mutation_validates_and_differs_from_its_parent(self):
        produced = 0
        for seed in range(40):
            child = variation.mutate(self.library, table(), rng(seed))
            if child is None:
                continue
            produced += 1
            child.validate(table())
            self.assertNotEqual(self.library.sha256(), child.sha256())
        self.assertGreater(produced, 0)

    def test_dropping_a_member_is_reachable(self):
        pair = self.library.with_macro(
            lib.Macro("second", lib.node("gray", [lib.HOLE])))
        sizes = set()
        for seed in range(60):
            child = variation.mutate(pair, table(), rng(seed))
            if child is not None:
                sizes.add(len(child.macros))
        self.assertIn(1, sizes, "a member is never dropped, so a bad member is stuck")

    def test_a_mutation_never_duplicates_a_sibling_shape(self):
        pair = library_of(lib.Macro("a", zigzag_split()),
                          lib.Macro("b", lib.node("gray", [lib.HOLE])))
        for seed in range(60):
            child = variation.mutate(pair, table(), rng(seed))
            if child is None:
                continue
            self.assertEqual(len(child.shapes()), len(child.macros))


class CrossoverTests(unittest.TestCase):
    def setUp(self):
        self.first = library_of(lib.Macro("a", zigzag_split()))
        self.second = library_of(lib.Macro("b", lib.node("gray", [lib.HOLE])))

    def test_a_child_draws_only_from_its_parents(self):
        allowed = self.first.shapes() | self.second.shapes()
        for seed in range(40):
            child = variation.crossover(self.first, self.second, rng(seed))
            if child is None:
                continue
            self.assertTrue(child.shapes() <= allowed)
            child.validate(table())

    def test_the_union_of_both_parents_is_reachable(self):
        """The whole point: a pair that only works together must be measurable."""
        seen = set()
        for seed in range(60):
            child = variation.crossover(self.first, self.second, rng(seed))
            if child is not None:
                seen.add(frozenset(child.shapes()))
        self.assertIn(frozenset(self.first.shapes() | self.second.shapes()), seen)

    def test_a_child_identical_to_a_parent_is_refused(self):
        for seed in range(60):
            child = variation.crossover(self.first, self.second, rng(seed))
            if child is None:
                continue
            self.assertNotIn(child.sha256(),
                             {self.first.sha256(), self.second.sha256()})

    def test_identical_parents_yield_nothing(self):
        self.assertIsNone(variation.crossover(self.first, self.first, rng()))

    def test_names_are_made_unique_when_parents_collide(self):
        clash = library_of(lib.Macro("a", lib.node("gray", [lib.HOLE])))
        for seed in range(40):
            child = variation.crossover(self.first, clash, rng(seed))
            if child is not None:
                child.validate(table())


class AddTests(unittest.TestCase):
    def test_a_duplicate_shape_is_refused(self):
        base = library_of(lib.Macro("a", zigzag_split()))
        self.assertIsNone(variation.add(base, lib.Macro("b", zigzag_split())))

    def test_a_full_library_refuses_to_grow(self):
        base = lib.EMPTY
        for index in range(variation.MAX_LIBRARY_SIZE):
            base = base.with_macro(lib.Macro(
                f"m{index}", lib.node("zigzag", [lib.node("split_field", [
                    lib.HOLE, lib.HOLE if index % 2 else lib.node("rans")])])))
        self.assertIsNone(variation.add(base, lib.Macro("x", lib.node("gray", [lib.HOLE]))))

    def test_a_colliding_name_is_renamed_rather_than_refused(self):
        base = library_of(lib.Macro("a", zigzag_split()))
        grown = variation.add(base, lib.Macro("a", lib.node("gray", [lib.HOLE])))
        self.assertIsNotNone(grown)
        self.assertEqual(2, len(grown.macros))
        grown.validate(table())


if __name__ == "__main__":
    unittest.main()
