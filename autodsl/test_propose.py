"""Tests for propose.py.

A model's reply is untrusted input. These tests pin that every way it can be
wrong produces a precise error rather than a macro the engine will later choke
on — and that the prompt keeps telling the truth as the engine changes.
"""

from __future__ import annotations

import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import library as lib  # noqa: E402
import propose  # noqa: E402
from backend import ScriptedBackend  # noqa: E402
from test_library import CONFIG, table, zigzag_split  # noqa: E402

EVIDENCE = {"model": "probe.safetensors", "ratio": 1.28, "raw_bytes": 10}
BUDGET = {"max_depth": 2, "max_expansions": 256}


def reply(macros) -> str:
    return "```json\n" + json.dumps({"macros": macros}) + "\n```"


GOOD = reply([{
    "name": "delta_split",
    "why": "because the exponent field barely changes between neighbours",
    "body": {"op": "diff_mod", "children": [
        {"op": "split_field", "children": [{"op": "hole"}, {"op": "hole"}]}]},
}])


class PromptTests(unittest.TestCase):
    def test_every_engine_operator_has_documented_semantics(self):
        """Guards against the engine growing an operator the prompt lies about."""
        import engine

        try:
            live = engine.config()
        except Exception as exc:  # noqa: BLE001 - binary may not be built
            self.skipTest(f"brevis binary unavailable: {exc}")
        missing = [
            op["name"] for op in live["operators"] if op["name"] not in propose.SEMANTICS
        ]
        self.assertEqual([], missing, "propose.SEMANTICS is missing operators")

    def test_prompt_names_operators_and_marks_the_unusable_ones(self):
        text = propose.operator_reference(table())
        self.assertIn("split_field", text)
        self.assertIn("NOT USABLE IN A MACRO BODY", text)

    def test_prompt_carries_evidence_library_and_rejections(self):
        library = lib.Library(macros=(lib.Macro("zigzag_split", zigzag_split()),))
        prompt = propose.build_prompt(
            EVIDENCE, library, table(), budget=BUDGET,
            rejected=[{"shape": "gray(?)", "reason": "never selected"}],
        )
        self.assertIn("probe.safetensors", prompt)
        self.assertIn("zigzag(split_field(?,?))", prompt)
        self.assertIn("gray(?)", prompt)
        self.assertIn("never selected", prompt)

    def test_member_values_are_shown_so_a_carried_macro_can_be_replaced(self):
        library = lib.Library(macros=(lib.Macro("zigzag_split", zigzag_split()),))
        prompt = propose.build_prompt(
            EVIDENCE, library, table(), budget=BUDGET,
            marginals={"zigzag_split": -4096},
        )
        self.assertIn("zigzag_split", prompt)
        self.assertIn("-4096", prompt)
        self.assertIn("pays for itself", prompt)

    def test_unmeasured_member_values_say_so_rather_than_showing_zero(self):
        prompt = propose.build_prompt(EVIDENCE, lib.EMPTY, table(), budget=BUDGET)
        self.assertIn("not measured yet", prompt)

    def test_mined_shapes_are_shown_so_they_are_not_reproposed(self):
        class Fake:
            shape = "xor_const(split_field(?,?))"
            bytes_governed = 75_200_000
            blocks = 482

        prompt = propose.build_prompt(
            EVIDENCE, lib.EMPTY, table(), budget=BUDGET, mined=[Fake()]
        )
        self.assertIn("xor_const(split_field(?,?))", prompt)


class ParseTests(unittest.TestCase):
    def test_accepts_a_well_formed_reply(self):
        macros = propose.parse_reply(GOOD, table(), lib.EMPTY)
        self.assertEqual(["delta_split"], [m.name for m in macros])
        self.assertEqual("diff_mod(split_field(?,?))", macros[0].shape())
        self.assertEqual("proposed", macros[0].origin)

    def test_accepts_an_unfenced_reply(self):
        macros = propose.parse_reply(GOOD.replace("```json", "").replace("```", ""),
                                     table(), lib.EMPTY)
        self.assertEqual(1, len(macros))

    def test_prose_around_the_block_is_ignored(self):
        macros = propose.parse_reply(
            "Here is my reasoning.\n" + GOOD + "\nHope that helps!",
            table(), lib.EMPTY)
        self.assertEqual(1, len(macros))

    def test_missing_params_defaults_to_auto(self):
        macros = propose.parse_reply(GOOD, table(), lib.EMPTY)
        self.assertEqual("auto", macros[0].body["params"])

    def test_non_json_is_rejected_with_the_text(self):
        with self.assertRaisesRegex(propose.ProposalError, "not JSON"):
            propose.parse_reply("I think you should try xor_prev.", table(), lib.EMPTY)

    def test_missing_macros_key_is_rejected(self):
        with self.assertRaisesRegex(propose.ProposalError, "no 'macros' key"):
            propose.parse_reply('```json\n{"suggestions": []}\n```', table(), lib.EMPTY)

    def test_a_hallucinated_operator_is_rejected(self):
        bad = reply([{"name": "fancy", "body": {"op": "wavelet", "children": []}}])
        with self.assertRaisesRegex(lib.LibraryError, "unknown operator"):
            propose.parse_reply(bad, table(), lib.EMPTY)

    def test_wrong_arity_is_rejected(self):
        bad = reply([{"name": "wrong", "body": {
            "op": "split_field", "children": [{"op": "hole"}]}}])
        with self.assertRaisesRegex(lib.LibraryError, "takes 2 children"):
            propose.parse_reply(bad, table(), lib.EMPTY)

    def test_a_width_dependent_operator_is_rejected(self):
        bad = reply([{"name": "planes", "body": {
            "op": "zigzag", "children": [{"op": "bit_plane", "children": []}]}}])
        with self.assertRaisesRegex(lib.LibraryError, "width-dependent"):
            propose.parse_reply(bad, table(), lib.EMPTY)

    def test_a_name_already_in_the_library_is_rejected(self):
        library = lib.Library(macros=(lib.Macro("delta_split", zigzag_split()),))
        with self.assertRaisesRegex(propose.ProposalError, "already taken"):
            propose.parse_reply(GOOD, table(), library)

    def test_two_proposals_sharing_a_name_are_rejected(self):
        twice = reply([
            {"name": "same", "body": {"op": "zigzag", "children": [{"op": "hole"}]}},
            {"name": "same", "body": {"op": "gray", "children": [{"op": "hole"}]}},
        ])
        with self.assertRaisesRegex(propose.ProposalError, "already taken"):
            propose.parse_reply(twice, table(), lib.EMPTY)


class ProposeTests(unittest.TestCase):
    def test_propose_records_what_was_sent_and_returned(self):
        llm = ScriptedBackend(replies=[GOOD])
        proposal = propose.propose(llm, EVIDENCE, lib.EMPTY, table(), budget=BUDGET)
        self.assertEqual(1, len(proposal.macros))
        self.assertEqual(GOOD, proposal.raw_reply)
        self.assertIn("probe.safetensors", proposal.prompt)
        system, user = llm.calls[0]
        self.assertIn("MACROS", system)
        self.assertEqual(proposal.prompt, user)


if __name__ == "__main__":
    unittest.main()
