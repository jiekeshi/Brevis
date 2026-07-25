"""Tests for library.py.

The engine is the authority on what a valid macro is; these tests pin the
behaviours this module adds on top — identity, deduplication, and rejecting a
malformed body before it ever reaches the engine.
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import library as lib  # noqa: E402

CONFIG = {
    "macro_library_schema": lib.SCHEMA,
    "macro_library_limits": {"max_macros": 4, "max_body_nodes": 5},
    "operators": [
        {"name": "raw", "terminal": True, "arity": 0, "usable_in_macro_body": True},
        {"name": "rans", "terminal": True, "arity": 0, "usable_in_macro_body": True},
        {"name": "zigzag", "terminal": False, "arity": 1, "usable_in_macro_body": True},
        {"name": "split_field", "terminal": False, "arity": 2,
         "usable_in_macro_body": True},
        {"name": "diff_mod", "terminal": False, "arity": 1,
         "usable_in_macro_body": True},
        {"name": "gray", "terminal": False, "arity": 1, "usable_in_macro_body": True},
        {"name": "bit_plane", "terminal": False, "arity": None,
         "usable_in_macro_body": False},
    ],
}


def table() -> lib.OperatorTable:
    return lib.OperatorTable.from_config(CONFIG)


def zigzag_split() -> dict:
    return lib.node("zigzag", [lib.node("split_field", [lib.HOLE, lib.HOLE])])


class TableTests(unittest.TestCase):
    def test_rejects_an_engine_speaking_another_schema(self):
        config = dict(CONFIG, macro_library_schema="brevis.macro-library.v99")
        with self.assertRaises(lib.LibraryError):
            lib.OperatorTable.from_config(config)

    def test_body_operators_excludes_width_dependent_arity(self):
        self.assertNotIn("bit_plane", table().body_operators())
        self.assertIn("split_field", table().body_operators())


class BodyTests(unittest.TestCase):
    def test_render_matches_engine_program_notation(self):
        self.assertEqual("zigzag(split_field(?,?))", lib.render(zigzag_split()))

    def test_stats_count_nodes_holes_and_transform_depth(self):
        stats = lib.body_stats(zigzag_split(), table())
        self.assertEqual((2, 2, 0, 2), (stats.nodes, stats.holes,
                                        stats.terminals, stats.transform_depth))

    def test_terminals_end_a_branch(self):
        body = lib.node("zigzag", [lib.node("rans")])
        stats = lib.body_stats(body, table())
        self.assertEqual((2, 0, 1, 1), (stats.nodes, stats.holes,
                                        stats.terminals, stats.transform_depth))

    def test_wrong_child_count_is_rejected(self):
        with self.assertRaisesRegex(lib.LibraryError, "takes 2 children, got 1"):
            lib.validate_body(lib.node("split_field", [lib.HOLE]), table())

    def test_width_dependent_operator_cannot_appear_in_a_body(self):
        with self.assertRaisesRegex(lib.LibraryError, "width-dependent arity"):
            lib.validate_body(lib.node("bit_plane", []), table())

    def test_unknown_operator_names_the_alternatives(self):
        with self.assertRaisesRegex(lib.LibraryError, "unknown operator 'transpose'"):
            lib.validate_body(lib.node("transpose", []), table())

    def test_params_must_be_auto_or_a_u32(self):
        with self.assertRaises(lib.LibraryError):
            lib.validate_body(lib.node("zigzag", [lib.HOLE], params=-1), table())
        with self.assertRaises(lib.LibraryError):
            lib.validate_body(lib.node("zigzag", [lib.HOLE], params="mode"), table())
        lib.validate_body(lib.node("zigzag", [lib.HOLE], params=7), table())


class MacroTests(unittest.TestCase):
    def test_name_must_be_lower_snake_case(self):
        for bad in ("Zigzag", "zig zag", "9lives", "", "x" * 60):
            with self.assertRaises(lib.LibraryError):
                lib.validate_macro(lib.Macro(bad, zigzag_split()), table())
        lib.validate_macro(lib.Macro("zigzag_split", zigzag_split()), table())

    def test_a_bare_hole_is_rejected(self):
        with self.assertRaisesRegex(lib.LibraryError, "identity"):
            lib.validate_macro(lib.Macro("nothing", dict(lib.HOLE)), table())

    def test_body_larger_than_the_engine_allows_is_rejected(self):
        body = lib.HOLE
        for _ in range(table().max_body_nodes + 1):
            body = lib.node("zigzag", [body])
        with self.assertRaisesRegex(lib.LibraryError, "exceeds the engine limit"):
            lib.validate_macro(lib.Macro("too_deep", body), table())


class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.library = lib.Library(macros=(lib.Macro("zigzag_split", zigzag_split()),))

    def test_duplicate_names_are_rejected(self):
        doubled = self.library.with_macro(lib.Macro("zigzag_split", lib.node("zigzag", [lib.HOLE])))
        with self.assertRaisesRegex(lib.LibraryError, "duplicate macro name"):
            doubled.validate(table())

    def test_two_names_for_the_same_body_are_rejected(self):
        doubled = self.library.with_macro(lib.Macro("other_name", zigzag_split()))
        with self.assertRaisesRegex(lib.LibraryError, "same body"):
            doubled.validate(table())

    def test_more_macros_than_the_engine_accepts_is_rejected(self):
        grown = self.library
        for i in range(5):
            grown = grown.with_macro(
                lib.Macro(f"m{i}", lib.node("zigzag", [lib.node("split_field", [
                    lib.HOLE, lib.node("rans")])] if i % 2 else [lib.HOLE])))
        with self.assertRaises(lib.LibraryError):
            grown.validate(table())

    def test_hash_ignores_rationale_but_not_structure(self):
        reworded = lib.Library(
            macros=(lib.Macro("zigzag_split", zigzag_split(), why="a new explanation",
                              origin="proposed"),))
        self.assertEqual(self.library.sha256(), reworded.sha256())

        restructured = lib.Library(
            macros=(lib.Macro("zigzag_split", lib.node("zigzag", [lib.HOLE])),))
        self.assertNotEqual(self.library.sha256(), restructured.sha256())

    def test_hash_is_insensitive_to_an_omitted_default_params_field(self):
        explicit = lib.Library(macros=(lib.Macro(
            "zigzag_split",
            {"op": "zigzag", "params": "auto", "children": [
                {"op": "split_field", "params": "auto",
                 "children": [dict(lib.HOLE), dict(lib.HOLE)]}]}),))
        terse = lib.Library(macros=(lib.Macro(
            "zigzag_split",
            {"op": "zigzag", "children": [
                {"op": "split_field", "children": [dict(lib.HOLE), dict(lib.HOLE)]}]}),))
        self.assertEqual(explicit.sha256(), terse.sha256())

    def test_round_trips_through_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "lib.json"
            self.library.write(path)
            self.assertEqual(self.library.sha256(), lib.Library.read(path).sha256())

    def test_reading_a_foreign_schema_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "lib.json"
            path.write_text('{"schema": "something.else", "macros": []}')
            with self.assertRaises(lib.LibraryError):
                lib.Library.read(path)

    def test_without_removes_only_the_named_macro(self):
        grown = self.library.with_macro(lib.Macro("second", lib.node("zigzag", [lib.HOLE])))
        self.assertEqual({"zigzag_split"}, grown.without("second").names())


if __name__ == "__main__":
    unittest.main()
