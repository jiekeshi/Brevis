import hashlib
import json
import pathlib
import unittest

import brevis_benchmarking


ROOT = pathlib.Path(__file__).resolve().parent.parent
CAMPAIGNS = ROOT / "eval" / "experiments-v1.json"


class ExperimentCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = json.loads(CAMPAIGNS.read_text(encoding="utf-8"))
        cls.models_path = ROOT / cls.payload["model_manifest"]["path"]
        cls.models = json.loads(cls.models_path.read_text(encoding="utf-8"))

    def test_manifest_and_calibration_defaults_are_immutable(self):
        observed_manifest_sha256 = hashlib.sha256(self.models_path.read_bytes()).hexdigest()
        self.assertEqual(
            self.payload["model_manifest"]["sha256"], observed_manifest_sha256,
        )
        self.assertEqual(
            brevis_benchmarking.DEFAULT_CALIBRATION_SEED,
            self.payload["common"]["calibration_seed"],
        )
        self.assertEqual("preregistered_configuration_not_results", self.payload["status"])
        self.assertNotIn("results", self.payload)

    def test_campaigns_have_unique_ids_and_known_model_tags(self):
        campaigns = self.payload["campaigns"]
        identifiers = [campaign["id"] for campaign in campaigns]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        known_tags = {model["tag"] for model in self.models}
        for campaign in campaigns:
            with self.subTest(campaign=campaign["id"]):
                self.assertTrue(campaign["model_tags"])
                self.assertFalse(set(campaign["model_tags"]) - known_tags)
                self.assertGreaterEqual(campaign["warmups"], 0)
                self.assertGreaterEqual(campaign["repetitions"], 1)
                self.assertTrue(all(job >= 1 for job in campaign["jobs"]))

    def test_primary_repetition_counts_and_core_methods_are_fixed(self):
        by_id = {campaign["id"]: campaign for campaign in self.payload["campaigns"]}
        self.assertEqual(
            list(brevis_benchmarking.CORE_CONFIGURATION_IDS),
            by_id["small-core-system-v1"]["configurations"],
        )
        self.assertEqual(6, by_id["small-core-system-v1"]["repetitions"])
        self.assertEqual(6, by_id["medium-core-system-v1"]["repetitions"])
        self.assertEqual(4, by_id["large-ultra-core-system-v1"]["repetitions"])
        self.assertEqual(3, by_id["cross-scale-parallel-v1"]["repetitions"])
        self.assertEqual([1, 2, 4, 8, 16, 32], by_id["cross-scale-parallel-v1"]["jobs"])

    def test_binary_defaults_are_explicitly_frozen(self):
        common = self.payload["common"]
        self.assertEqual(
            {
                "max_expansions": 256,
                "max_realizations": 32,
                "max_realizations_cli_exposed": False,
                "max_nodes": 12,
                "max_depth": 2,
                "sample_elems": 4096,
                "rerank_candidates": 8,
                "rerank_blocks": 4,
            },
            common["search_defaults"],
        )
        self.assertEqual(262144, common["archive_defaults"]["target_block_bytes"])
        self.assertTrue(common["archive_defaults"]["raw_fallback_enabled"])
        self.assertEqual(3600, common["timeout_seconds_per_process_default"])

    def test_ablation_inventory_is_complete_and_raw_is_never_disabled(self):
        inventory = self.payload["operator_inventory"]
        expected_transforms = {
            "xor_const", "add_const_mod", "xor_prev", "diff_mod", "zigzag",
            "gray", "rotate_bits", "bit_reverse", "split_field", "topk_codebook",
            "rle", "deinterleave", "split_float", "bit_plane", "byte_plane",
        }
        self.assertEqual(expected_transforms, set(inventory["reversible_operators"]))
        self.assertEqual({"bitpack", "huffman", "rans"}, set(inventory["terminal_codecs"]))
        self.assertEqual("raw", inventory["mandatory_raw_fallback"])

        by_id = {campaign["id"]: campaign for campaign in self.payload["campaigns"]}
        self.assertEqual(
            expected_transforms,
            set(by_id["small-reversible-operator-leave-one-out-v1"]
                ["one_disabled_operator_per_configuration"]),
        )
        disabled_terminals = set(
            by_id["small-terminal-codec-leave-one-out-v1"]
            ["one_disabled_terminal_per_configuration"]
        )
        self.assertEqual({"bitpack", "huffman", "rans"}, disabled_terminals)
        self.assertNotIn("raw", disabled_terminals)

    def test_budget_depth_and_reranking_defaults_are_included(self):
        by_id = {campaign["id"]: campaign for campaign in self.payload["campaigns"]}
        budgets = by_id["small-search-budget-v1"]["independent_sweep"]
        depths = by_id["small-search-depth-v1"]["independent_sweep"]
        reranking = by_id["small-reranking-v1"]
        self.assertIn(256, budgets["max_expansions"])
        self.assertEqual(2, budgets["held_constant"]["max_depth"])
        self.assertIn(2, depths["max_depth"])
        self.assertEqual(256, depths["held_constant"]["max_expansions"])
        self.assertIn(0, reranking["candidate_sweep"]["rerank_candidates"])
        self.assertIn(8, reranking["candidate_sweep"]["rerank_candidates"])
        self.assertIn(0, reranking["block_sweep"]["rerank_blocks"])
        self.assertIn(4, reranking["block_sweep"]["rerank_blocks"])


if __name__ == "__main__":
    unittest.main()
