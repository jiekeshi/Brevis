import hashlib
import json
import pathlib
import re
import unittest


EVAL_DIR = pathlib.Path(__file__).resolve().parent
PROTOCOL_PATH = EVAL_DIR / "specialized-baselines.json"
MODEL_MANIFEST_PATH = EVAL_DIR / "models-tiered.json"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class SpecializedBaselineProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        cls.models = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.by_id = {item["id"]: item for item in cls.protocol["baselines"]}

    def test_document_is_a_preregistration_not_results(self):
        self.assertEqual(
            self.protocol["schema"], "brevis.specialized-baseline-protocol"
        )
        self.assertEqual(self.protocol["schema_version"], 1)
        self.assertEqual(self.protocol["protocol_status"], "preregistered_not_run")
        self.assertFalse(self.protocol["contains_experimental_results"])
        for baseline in self.protocol["baselines"]:
            self.assertTrue(
                baseline["protocol_status"].startswith("not_")
                or baseline["protocol_status"] == "not_a_baseline"
            )

    def test_model_manifest_binding(self):
        binding = self.protocol["model_manifest"]
        self.assertEqual(binding["path"], "eval/models-tiered.json")
        digest = hashlib.sha256(MODEL_MANIFEST_PATH.read_bytes()).hexdigest()
        self.assertEqual(binding["sha256"], digest)
        self.assertRegex(digest, HEX64)

    def test_baseline_ids_are_unique_and_complete(self):
        ids = [item["id"] for item in self.protocol["baselines"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            set(ids),
            {
                "zipnn-0.5.4-strict",
                "dfloat11-0.5.0-tinyllama-strict",
                "fpcompress-1.0.3-cpu",
                "spdp-1.1",
                "fpzip-1.3.0-full-precision",
                "adaptivefc-lc-1.2-ga",
                "openzl-0.2.0-serial-fixed",
                "openzl-0.2.0-serial-ace-input-local",
                "fpcrush-1.0-audit-only",
                "weight-compression-audit-only",
            },
        )

    def test_pinned_git_and_artifact_hashes_are_well_formed(self):
        for baseline in self.protocol["baselines"]:
            commit = baseline["source"].get("git_commit")
            if commit is not None:
                self.assertRegex(commit, HEX40, baseline["id"])

        zipnn = self.by_id["zipnn-0.5.4-strict"]
        self.assertRegex(zipnn["source"]["sdist"]["sha256"], HEX64)
        dfloat11 = self.by_id["dfloat11-0.5.0-tinyllama-strict"]
        self.assertRegex(dfloat11["source"]["decode_ptx"]["sha256"], HEX64)
        self.assertRegex(dfloat11["source"]["decode_cu"]["sha256"], HEX64)
        spdp = self.by_id["spdp-1.1"]
        self.assertRegex(spdp["source"]["source_file"]["sha256"], HEX64)

    def test_audited_source_identities_are_frozen(self):
        expected_commits = {
            "zipnn-0.5.4-strict": "6009704271394f0497dd6352f72aedc6b947bfc0",
            "dfloat11-0.5.0-tinyllama-strict":
                "457733886ce6ebc6d8dda1621fad1ffa2661e028",
            "fpcompress-1.0.3-cpu": "97f037249bd28682bfd83462ea1129e14df36d81",
            "fpzip-1.3.0-full-precision":
                "4a539c06d98b1c029b08324a086d4b75689a2b72",
            "adaptivefc-lc-1.2-ga": "0553cd874ceabd7189653dd5d28958c68256bf3b",
            "openzl-0.2.0-serial-fixed":
                "3dceb64867840201fb8f57a29d179995f700c9b8",
            "openzl-0.2.0-serial-ace-input-local":
                "3dceb64867840201fb8f57a29d179995f700c9b8",
            "weight-compression-audit-only":
                "b9510aaac657b04e11d2f0d0d51a9b94af159590",
        }
        for baseline_id, commit in expected_commits.items():
            self.assertEqual(self.by_id[baseline_id]["source"]["git_commit"], commit)
        self.assertEqual(
            self.by_id["zipnn-0.5.4-strict"]["source"]["sdist"]["sha256"],
            "abf45dbf41838f83aa2601dfb59c856b557b1df9c07cc243d93098ed889ce650",
        )
        self.assertEqual(
            self.by_id["dfloat11-0.5.0-tinyllama-strict"]
            ["source"]["decode_ptx"]["sha256"],
            "cc9b78f6a14c9118abf46a733390f14ef66a9c2eb4db04ded9b74183a02665cf",
        )
        self.assertEqual(
            self.by_id["dfloat11-0.5.0-tinyllama-strict"]
            ["source"]["decode_cu"]["sha256"],
            "9b9a3c7c491bcdd4a85949da4c337ac1ce7f9de627eea3d18cd335878d186950",
        )
        self.assertEqual(
            self.by_id["spdp-1.1"]["source"]["source_file"]["sha256"],
            "c0b6caa9bca5671b7bf72d4d6be763f3231edbbe4c53e1f3a4a8bcbb8a863116",
        )
        openzl = self.by_id["openzl-0.2.0-serial-fixed"]["source"]
        self.assertEqual(
            openzl["release_archive"]["sha256_observed"],
            "2ad14ed9af63d4a70cb05df5d5629871d052371ad017cf5559dc76c41ae3865f",
        )
        expected_submodules = {
            "googletest": (
                "56efe3983185e3f37e43415d1afa97e3860f187f",
                "20f265f7346b6a542a49ea6fa0866f82e00841f2a7c2bc41d51014e662041cf9",
            ),
            "lz4": (
                "ebb370ca83af193212df4dcbadcc5d87bc0de2f0",
                "eb1a93e934d4fd29df6e2061ba0bf447568561764d758f4a5662c0e29370ffa9",
            ),
            "xgboost": (
                "ccb511768e13d1670c10be07dea89d0edca138f3",
                "8c1d333cc4a644cebd93e72f6371329a81a713b74cfd235fb4ebaad9ad28d659",
            ),
            "zstd": (
                "f8745da6ff1ad1e7bab384bd1f9d742439278e99",
                "4b0bd1f0cfb25e61b9103c35f27395530ff5b4c0d2513a00fd745849e85ea52c",
            ),
        }
        self.assertEqual(set(expected_submodules), set(openzl["submodules"]))
        for name, (commit, archive_hash) in expected_submodules.items():
            self.assertEqual(openzl["submodules"][name]["git_commit"], commit)
            self.assertEqual(
                openzl["submodules"][name]["archive_sha256_observed"], archive_hash,
            )

    def test_scope_and_dtype_guards(self):
        model_tags = {item["tag"] for item in self.models}
        zipnn_classes = set(
            self.by_id["zipnn-0.5.4-strict"]["domain_match"][
                "classification_by_model"
            ]
        )
        spdp_classes = set(
            self.by_id["spdp-1.1"]["domain_match"]["classification_by_model"]
        )
        self.assertEqual(zipnn_classes, model_tags)
        self.assertEqual(spdp_classes, model_tags)
        self.assertEqual(
            self.by_id["zipnn-0.5.4-strict"]["domain_match"][
                "native_tensor_dtypes"
            ],
            ["F32", "F16", "BF16"],
        )
        self.assertEqual(
            self.by_id["dfloat11-0.5.0-tinyllama-strict"]["domain_match"][
                "eligible_model_tags"
            ],
            ["medium-tinyllama-bf16"],
        )
        f32_tags = ["small-bert-f32", "small-vit-f32"]
        for baseline_id in (
            "fpcompress-1.0.3-cpu",
            "fpzip-1.3.0-full-precision",
            "adaptivefc-lc-1.2-ga",
        ):
            self.assertEqual(
                self.by_id[baseline_id]["domain_match"]["eligible_model_tags"],
                f32_tags,
            )

    def test_fixed_profiles(self):
        fpcompress = self.by_id["fpcompress-1.0.3-cpu"]
        self.assertEqual(
            {profile["id"]: profile["pipeline"] for profile in fpcompress["profiles"]},
            {
                "SPspeed": "DIFFMS_4 HCLOG_4",
                "SPratio": "DIFFMS_4 BIT_4 RZE_1",
            },
        )
        spdp = self.by_id["spdp-1.1"]
        self.assertEqual([profile["level"] for profile in spdp["profiles"]], [0, 5, 9])
        fpzip = self.by_id["fpzip-1.3.0-full-precision"]["profile"]
        self.assertEqual(fpzip["precision_bits"], 32)
        for geometry in fpzip["input_geometry_by_model"].values():
            self.assertEqual(geometry["nx"] * 4, geometry["input_bytes"])

    def test_adaptivefc_preregistered_search_and_seed_limit(self):
        adaptive = self.by_id["adaptivefc-lc-1.2-ga"]
        config = adaptive["ga_configuration"]
        self.assertEqual(config["stages"], 5)
        self.assertEqual(config["generations"], 140)
        self.assertEqual(config["population"], 20)
        self.assertEqual(config["mutation_rate"], 0.8)
        self.assertEqual(config["elitism_cutoff"], 0.1)
        self.assertEqual(config["selection_method"], "tournament")
        self.assertEqual(config["crossover_method"], "masked")
        self.assertEqual(config["pilot_seed"], 0)
        self.assertEqual(config["expanded_seeds"], list(range(9)))
        self.assertEqual(
            config["replication_label"],
            "reproduction_inspired_not_exact_seed_replication",
        )

    def test_openzl_fixed_and_transductive_profiles_are_not_conflated(self):
        fixed = self.by_id["openzl-0.2.0-serial-fixed"]
        self.assertEqual(16777216, fixed["profile"]["chunk_bytes"])
        self.assertIn("--strict", fixed["profile"]["compress_argv"])
        self.assertIn("--store-on-expansion", fixed["profile"]["compress_argv"])
        self.assertEqual(
            "whole_file_dtype_agnostic_byte_stream",
            fixed["domain_match"]["classification"],
        )

        trained = self.by_id["openzl-0.2.0-serial-ace-input-local"]
        self.assertEqual(4194304, trained["sample_manifest"]["block_bytes"])
        self.assertEqual(16, trained["sample_manifest"]["windows"])
        self.assertEqual(1, trained["training"]["threads"])
        self.assertEqual(60, trained["training"]["max_time_seconds_per_training_step"])
        self.assertEqual(3, trained["stochastic_protocol"]["independent_training_runs"])
        self.assertIn("median", trained["stochastic_protocol"]["selection_rule"])
        self.assertIn("not", trained["domain_match"]["interpretation"])

    def test_fpcrush_is_an_explicit_nonbaseline(self):
        fpcrush = self.by_id["fpcrush-1.0-audit-only"]
        self.assertEqual("not_a_baseline", fpcrush["protocol_status"])
        self.assertIsNone(fpcrush["source"]["git_commit"])
        self.assertIsNone(fpcrush["source"]["public_source_artifact"])
        self.assertEqual(5, fpcrush["published_protocol"]["stages"])
        self.assertEqual(16, fpcrush["published_protocol"]["ga_generations"])
        self.assertIsNone(fpcrush["published_protocol"]["published_random_seeds"])

    def test_whole_file_and_failure_contracts_are_explicit(self):
        accounting = self.protocol["global_rules"]["strict_whole_file_accounting"]
        self.assertIn("fresh process", accounting["fresh_decode_rule"])
        self.assertIn("SHA-256", accounting["correctness"])
        statuses = set(self.protocol["global_rules"]["allowed_runtime_statuses"])
        self.assertTrue(
            {
                "success",
                "failed_build",
                "failed_gpu_gate",
                "failed_roundtrip",
                "timeout",
                "oom",
                "skipped_resource",
                "not_applicable",
            }.issubset(statuses)
        )
        self.assertEqual(
            self.by_id["weight-compression-audit-only"]["baseline_decision"],
            "excluded_not_executable_baseline",
        )


if __name__ == "__main__":
    unittest.main()
