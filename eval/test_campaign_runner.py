import copy
import hashlib
import io
import json
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

import campaign_runner as runner


ROOT = pathlib.Path(__file__).resolve().parent.parent


class CampaignRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = runner.plan_campaigns()

    def tasks(self, campaign_id, model_tag=None):
        return [
            task for task in self.plan["tasks"]
            if task["campaign"]["id"] == campaign_id
            and (model_tag is None or task["model"]["tag"] == model_tag)
        ]

    def fixture_plan(
        self,
        root,
        *,
        stage=1,
        kind="brevis_system",
        configurations=None,
        jobs=(1,),
        staging_inside_repository=False,
    ):
        root = pathlib.Path(root)
        repository = root / "repository"
        eval_directory = repository / "eval"
        cache = root / "cache"
        staging = (
            repository / "unignored-results"
            if staging_inside_repository else root / "staging"
        )
        eval_directory.mkdir(parents=True)
        revision = "a" * 40
        source = cache / "test__repo" / revision / "model.safetensors"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"fixture")
        manifest = [{
            "tag": "fixture-model",
            "repo": "test/repo",
            "revision": revision,
            "scope": "complete_weight_variant",
            "selected_bytes": source.stat().st_size,
            "files": [{
                "file": source.name,
                "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }],
        }]
        manifest_path = eval_directory / "models.json"
        manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        if configurations is None:
            configurations = (
                ("fixed", "uniform", "phog")
                if kind == "brevis_parallel_scaling"
                else ("raw-terminal", "fixed", "uniform", "phog")
            )
        campaign = {
            "id": "fixture-campaign",
            "stage": stage,
            "kind": kind,
            "model_tags": ["fixture-model"],
            "jobs": list(jobs),
            "warmups": 0,
            "repetitions": 1,
        }
        if kind in {"brevis_system", "brevis_parallel_scaling"}:
            campaign["configurations"] = list(configurations)
        if kind in runner.UNSUPPORTED_KINDS:
            campaign["registry"] = [{"id": "coordination-only"}]
        matrix = {
            "schema": {"id": "brevis.experiment-campaigns", "version": 1},
            "status": "preregistered_configuration_not_results",
            "model_manifest": {
                "path": "eval/models.json",
                "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            },
            "common": {
                "binary_build": "ReleaseFast",
                "schedule_seed": 1,
                "calibration_seed": runner.system.DEFAULT_CALIBRATION_SEED,
                "calibration_tensors": 2,
                "search_defaults": {
                    "max_expansions": 4,
                    "max_realizations": runner.system.FROZEN_MAX_REALIZATIONS,
                    "max_realizations_cli_exposed": False,
                    "max_nodes": 3,
                    "max_depth": 1,
                    "sample_elems": 8,
                    "rerank_candidates": 2,
                    "rerank_blocks": 1,
                },
                "archive_defaults": {
                    "raw_fallback_enabled": True,
                    "target_block_bytes": runner.system.FROZEN_TARGET_BLOCK_BYTES,
                },
                "timeout_seconds_per_process_default": 10,
                "disk_reserve_fraction": 0.1,
            },
            "operator_inventory": {
                "mandatory_raw_fallback": "raw",
                "terminal_codecs": ["bitpack"],
                "reversible_operators": ["xor_prev"],
            },
            "campaigns": [campaign],
        }
        experiments = eval_directory / "experiments.json"
        experiments.write_text(json.dumps(matrix) + "\n", encoding="utf-8")
        plan = runner.plan_campaigns(
            experiments,
            repository_root=repository,
            cache_root=cache,
            staging_root=staging,
        )
        return plan, source, experiments, manifest_path

    @staticmethod
    def benchmark_result(task, *, formal=True, reasons=()):
        implementation = task["implementation_binding"]
        return {
            "status": "complete",
            "success": True,
            "provenance": {
                "git": {"commit": implementation["git_commit"]},
                "harness": {
                    "sha256": implementation["brevis_benchmarking_sha256"],
                },
            },
            "input_metadata": {
                "campaign_task_semantic_sha256": task["task_semantic_sha256"],
            },
            "run_classification": {
                "formal_eligible": formal,
                "formal_ineligibility_reasons": list(reasons),
            },
        }

    def test_plan_is_deterministic_manifest_frozen_and_task_unique(self):
        second = runner.plan_campaigns()
        self.assertEqual(157, self.plan["task_count"])
        self.assertEqual(154, self.plan["supported_task_count"])
        self.assertEqual(3, self.plan["unsupported_task_count"])
        self.assertEqual(
            [task["task_semantic_sha256"] for task in self.plan["tasks"]],
            [task["task_semantic_sha256"] for task in second["tasks"]],
        )
        manifest = ROOT / "eval" / "models-tiered.json"
        observed = hashlib.sha256(manifest.read_bytes()).hexdigest()
        self.assertTrue(self.plan["model_manifest"]["verified"])
        self.assertEqual(observed, self.plan["model_manifest"]["actual_sha256"])
        hashes = [task["task_semantic_sha256"] for task in self.plan["tasks"]]
        self.assertEqual(len(hashes), len(set(hashes)))
        for task in self.plan["tasks"]:
            with self.subTest(task=task["task_semantic_sha256"]):
                semantics = task["task_semantics"]
                self.assertEqual(runner.TASK_SEMANTIC_SPEC_ID, semantics["spec_id"])
                self.assertEqual(task["task_semantic_sha256"], runner._semantic_sha256(semantics))
                self.assertEqual(task["input"]["expected_size_bytes"], semantics["input"]["bytes"])
                self.assertEqual(task["input"]["expected_sha256"], semantics["input"]["sha256"])
                self.assertEqual(task["brevis_specs"], semantics["brevis_specs"])
                self.assertEqual(
                    task["comparison_group"]["id"], semantics["comparison_group_id"],
                )
                policy = semantics["executor"]["exclusive_execution_policy"]
                self.assertEqual(runner.EXCLUSIVE_EXECUTION_POLICY_ID, policy["policy_id"])
                self.assertTrue(policy["required"])
                self.assertEqual("exclusive_nonblocking", policy["acquisition"])
                self.assertIn("advisory", policy["limitations"])

    def test_comparison_groups_preserve_required_pairing(self):
        core = self.tasks("small-core-system-v1", "small-bert-f32")
        self.assertEqual(1, len(core))
        self.assertEqual(
            ["raw-terminal", "fixed", "uniform", "phog"],
            [spec["id"] for spec in core[0]["brevis_specs"]],
        )
        self.assertTrue(core[0]["comparison_group"]["schedule"]["within_group_balanced"])

        scaling = self.tasks("cross-scale-parallel-v1", "small-bert-f32")
        self.assertEqual(6, len(scaling))
        self.assertEqual({1, 2, 4, 8, 16, 32}, {
            task["configuration"]["jobs"] for task in scaling
        })
        for task in scaling:
            self.assertEqual(
                ["fixed", "uniform", "phog"],
                [spec["id"] for spec in task["brevis_specs"]],
            )
            self.assertIsNotNone(task["machine_binding"])
            self.assertFalse(task["comparison_group"]["schedule"]["outer_axis_balanced"])

        expected_counts = {
            "small-search-budget-v1": 6,
            "small-search-depth-v1": 4,
            "small-budget-depth-grid-v1": 9,
            "small-reranking-v1": 9,
        }
        for campaign_id, expected_count in expected_counts.items():
            tasks = self.tasks(campaign_id, "small-bert-f32")
            self.assertEqual(expected_count, len(tasks))
            for task in tasks:
                self.assertEqual(
                    {"uniform", "phog"},
                    {spec["prior_policy"] == "canonical" and "phog" or "uniform"
                     for spec in task["brevis_specs"]},
                )
                self.assertEqual(2, len(task["brevis_specs"]))
                self.assertFalse(task["comparison_group"]["schedule"]["outer_axis_balanced"])

        calibration = self.tasks("small-phog-calibration-size-v1", "small-bert-f32")
        self.assertEqual(5, len(calibration))
        self.assertEqual({25, 50, 100, 200, 400}, {
            task["configuration"]["calibration_tensors"] for task in calibration
        })
        for task in calibration:
            self.assertEqual(["phog"], [spec["id"] for spec in task["brevis_specs"]])
            schedule = task["comparison_group"]["schedule"]
            self.assertFalse(schedule["within_group_balanced"])
            self.assertFalse(schedule["outer_axis_balanced"])
            self.assertIn("outer-unpaired", schedule["limitation"])

    def test_operator_and_terminal_leave_one_out_share_one_group(self):
        operator_task, = self.tasks(
            "small-reversible-operator-leave-one-out-v1", "small-bert-f32",
        )
        terminal_task, = self.tasks(
            "small-terminal-codec-leave-one-out-v1", "small-bert-f32",
        )
        self.assertEqual(16, len(operator_task["brevis_specs"]))
        self.assertEqual(4, len(terminal_task["brevis_specs"]))
        self.assertEqual(15, len(operator_task["dimension"]["disabled_operators"]))
        self.assertEqual(
            {"bitpack", "huffman", "rans"},
            set(terminal_task["dimension"]["disabled_terminals"]),
        )
        for task in (operator_task, terminal_task):
            self.assertTrue(task["comparison_group"]["schedule"]["within_group_balanced"])
            for spec in task["brevis_specs"]:
                args = spec["search_args"]
                disabled = [
                    args[index + 1] for index, value in enumerate(args[:-1])
                    if value == "--disable-op"
                ]
                self.assertNotIn("raw", disabled)

    def test_specs_bind_full_effective_config_and_guidance(self):
        expected_keys = list(runner.system.EXPECTED_EFFECTIVE_CONFIG_KEYS)
        for task in self.plan["tasks"]:
            canonical_args = {
                tuple(spec["search_args"])
                for spec in task["brevis_specs"]
                if spec["prior_policy"] == "canonical"
            }
            self.assertLessEqual(len(canonical_args), 1)
            templates = {
                item["configuration_id"]: item["commands"]
                for item in (task["executor"]["brevis_command_templates"] or [])
            }
            for spec in task["brevis_specs"]:
                with self.subTest(task=task["task_semantic_sha256"], spec=spec["id"]):
                    expected = spec["expected_effective_config"]
                    self.assertEqual(expected_keys, list(expected))
                    self.assertEqual(32, expected["max_realizations"])
                    self.assertEqual("single_stream_search_only", expected["max_realizations_scope"])
                    self.assertFalse(expected["tensor_search_uses_max_realizations"])
                    self.assertEqual(262144, expected["target_block_bytes"])
                    if spec["plan"] == "search":
                        requested, disabled = runner.system._parse_search_arguments(
                            tuple(spec["search_args"]),
                        )
                        self.assertEqual(set(dict(runner.SEARCH_OPTION_ORDER)), set(requested))
                        for key, value in requested.items():
                            self.assertEqual(value, expected[key])
                        self.assertFalse(set(disabled) & set(expected["enabled_ops"]))
                    else:
                        self.assertEqual("fixed", spec["id"])
                        self.assertEqual([], spec["search_args"])
                    if spec["id"].startswith("phog"):
                        self.assertEqual("canonical", spec["prior_policy"])
                    if spec["id"].startswith("uniform") or spec["id"] == "raw-terminal":
                        self.assertEqual("none", spec["prior_policy"])
                    commands = templates[spec["id"]]
                    self.assertEqual(spec["search_args"], commands["configuration_probe"][2:])
                    if spec["prior_policy"] == "canonical":
                        self.assertIsNotNone(commands["calibration"])
                        self.assertIn("--prior", commands["compression"])
                    else:
                        self.assertIsNone(commands["calibration"])

    def test_raw_terminal_is_a_semantic_contract_not_just_a_name(self):
        core, = self.tasks("small-core-system-v1", "small-bert-f32")
        raw = next(spec for spec in core["brevis_specs"] if spec["id"] == "raw-terminal")
        self.assertTrue(raw["require_raw_only"])
        self.assertEqual(["raw"], raw["expected_effective_config"]["enabled_ops"])
        args = raw["search_args"]
        disabled = {
            args[index + 1] for index, value in enumerate(args[:-1])
            if value == "--disable-op"
        }
        matrix = json.loads((ROOT / "eval" / "experiments-v1.json").read_text())
        inventory = matrix["operator_inventory"]
        self.assertEqual(
            set(inventory["terminal_codecs"]) | set(inventory["reversible_operators"]),
            disabled,
        )
        self.assertNotIn("raw", disabled)

    def test_generic_campaign_is_one_coordination_placeholder_per_model_file(self):
        tasks = self.tasks("small-generic-codecs-v1")
        self.assertEqual(3, len(tasks))
        for task in tasks:
            self.assertFalse(task["executor"]["supported"])
            self.assertEqual("unsupported_kind", task["executor"]["status_code"])
            self.assertEqual([], task["brevis_specs"])
            self.assertIn(
                "not one row per baseline", task["dimension"]["placeholder_semantics"],
            )
        task = tasks[0]
        with self.assertRaisesRegex(runner.CampaignError, "unsupported"):
            runner.execute_one(self.plan, task["task_semantic_sha256"])

    def test_execute_one_regenerates_task_and_preserves_formal_gates(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, _, _, _ = self.fixture_plan(temporary)
            task, = plan["tasks"]
            result = self.benchmark_result(task)
            with mock.patch.object(
                runner.system, "benchmark_file", return_value=result,
            ) as benchmark:
                receipt = runner.execute_one(plan, task["task_semantic_sha256"])
            self.assertTrue(receipt["result_success"])
            self.assertTrue(receipt["formal_eligible"])
            _, kwargs = benchmark.call_args
            self.assertEqual(
                ["raw-terminal", "fixed", "uniform", "phog"],
                [spec.identifier for spec in kwargs["specs"]],
            )
            self.assertTrue(kwargs["require_declared_integrity"])
            self.assertTrue(kwargs["require_clean_git"])
            self.assertTrue(kwargs["build_binary"])
            self.assertFalse(kwargs["force_checkpoint"])
            self.assertEqual((), kwargs["additional_formal_ineligibility_reasons"])
            self.assertEqual(
                task["comparison_group"]["id"],
                kwargs["input_metadata"]["campaign_comparison_group_id"],
            )
            with self.assertRaisesRegex(runner.CampaignError, "found 0"):
                runner.execute_one(plan, "0" * 64)

    def test_execute_rejects_forged_task_and_semantic_spec(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, _, _, _ = self.fixture_plan(temporary)
            forged = copy.deepcopy(plan)
            task = forged["tasks"][0]
            task["dimension"] = {"forged": True}
            task["task_semantics"]["dimension"] = {"forged": True}
            task["task_semantic_sha256"] = runner._semantic_sha256(task["task_semantics"])
            with self.assertRaisesRegex(runner.CampaignError, "not generated"):
                runner.execute_one(forged, task["task_semantic_sha256"])

            wrong_spec = copy.deepcopy(plan)
            task = wrong_spec["tasks"][0]
            task["task_semantics"]["spec_id"] = "unknown.task-semantics"
            task["task_semantic_sha256"] = runner._semantic_sha256(task["task_semantics"])
            with self.assertRaisesRegex(runner.CampaignError, "semantic specification"):
                runner.execute_one(wrong_spec, task["task_semantic_sha256"])

    def test_execute_rejects_commit_runner_and_harness_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, _, _, _ = self.fixture_plan(temporary)
            task, = plan["tasks"]
            original = task["implementation_binding"]
            for key, replacement in (
                ("git_commit", "b" * 40),
                ("campaign_runner_sha256", "1" * 64),
                ("brevis_benchmarking_sha256", "2" * 64),
            ):
                drifted = {**original, key: replacement}
                with self.subTest(key=key), mock.patch.object(
                    runner, "_implementation_binding", return_value=drifted,
                ):
                    with self.assertRaisesRegex(runner.CampaignError, "not generated"):
                        runner.execute_one(plan, task["task_semantic_sha256"])

    def test_execute_rejects_result_provenance_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, _, _, _ = self.fixture_plan(temporary)
            task, = plan["tasks"]
            for path, replacement, message in (
                (("provenance", "git", "commit"), "b" * 40, "result commit"),
                (("provenance", "harness", "sha256"), "4" * 64, "harness hash"),
            ):
                result = self.benchmark_result(task)
                result[path[0]][path[1]][path[2]] = replacement
                with self.subTest(path=path), mock.patch.object(
                    runner.system, "benchmark_file", return_value=result,
                ):
                    with self.assertRaisesRegex(runner.CampaignError, message):
                        runner.execute_one(plan, task["task_semantic_sha256"])

    def test_scaling_task_binds_machine_and_rejects_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, _, _, _ = self.fixture_plan(
                temporary, stage=2, kind="brevis_parallel_scaling",
            )
            task, = plan["tasks"]
            self.assertIsNotNone(task["machine_binding"])
            drifted = copy.deepcopy(task["machine_binding"])
            drifted["sha256"] = "3" * 64
            with mock.patch.object(runner, "_machine_binding", return_value=drifted):
                with self.assertRaisesRegex(runner.CampaignError, "not generated"):
                    runner.execute_one(plan, task["task_semantic_sha256"])

    def test_advisory_lock_is_nonblocking_and_covers_benchmark_call(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixed_lock = root / "machine-wide.lock"
            with mock.patch.object(runner, "DEFAULT_EXECUTION_LOCK", fixed_lock):
                first, _, _, _ = self.fixture_plan(root / "worktree-a")
                second, _, _, _ = self.fixture_plan(root / "worktree-b")
                first_task, = first["tasks"]
                second_task, = second["tasks"]
                self.assertNotEqual(
                    first["planner_configuration"]["cache_root"],
                    second["planner_configuration"]["cache_root"],
                )
                first_policy = first_task["executor"]["exclusive_execution_policy"]
                second_policy = second_task["executor"]["exclusive_execution_policy"]
                self.assertEqual(str(fixed_lock), first_policy["lock_path"])
                self.assertEqual(first_policy["lock_path"], second_policy["lock_path"])
                with runner._exclusive_execution_lock(fixed_lock):
                    with mock.patch.object(runner.system, "benchmark_file") as benchmark:
                        with self.assertRaisesRegex(runner.CampaignError, "advisory lock"):
                            runner.execute_one(
                                second, second_task["task_semantic_sha256"],
                            )
                        benchmark.assert_not_called()

    def test_stage_gate_rejects_by_default_and_override_is_nonformal(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, _, _, _ = self.fixture_plan(temporary, stage=2)
            task, = plan["tasks"]
            with mock.patch.object(runner.system, "benchmark_file") as benchmark:
                with self.assertRaisesRegex(runner.CampaignError, "stage/resource gate"):
                    runner.execute_one(plan, task["task_semantic_sha256"])
                benchmark.assert_not_called()

            reason = "campaign_stage_gate_not_proven"
            result = self.benchmark_result(task, formal=False, reasons=(reason,))
            with mock.patch.object(
                runner.system, "benchmark_file", return_value=result,
            ) as benchmark:
                receipt = runner.execute_one(
                    plan, task["task_semantic_sha256"], allow_unmet_stage_gate=True,
                )
            self.assertFalse(receipt["formal_eligible"])
            self.assertEqual(
                (reason,),
                benchmark.call_args.kwargs["additional_formal_ineligibility_reasons"],
            )

    def test_no_overwrite_and_nonignored_plan_output_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, _, _, _ = self.fixture_plan(temporary)
            task, = plan["tasks"]
            output = pathlib.Path(task["suggested_output_path"])
            output.parent.mkdir(parents=True)
            output.write_text("existing", encoding="utf-8")
            with mock.patch.object(runner.system, "benchmark_file") as benchmark:
                with self.assertRaisesRegex(runner.CampaignError, "overwrite"):
                    runner.execute_one(plan, task["task_semantic_sha256"])
                benchmark.assert_not_called()

        unsafe = ROOT / "eval" / "campaign-plan-unignored-test.json"
        self.assertFalse(runner._plan_output_safe_during_execution(unsafe, ROOT))
        self.assertTrue(runner._plan_output_safe_during_execution(
            ROOT / "eval" / "cache" / "campaign-plan-test.json", ROOT,
        ))
        with mock.patch("sys.stderr", new_callable=io.StringIO) as stderr:
            return_code = runner.main([
                "--plan-output", str(unsafe), "--execute-one", "0" * 64,
            ])
        self.assertEqual(2, return_code)
        self.assertIn("would dirty", stderr.getvalue())
        self.assertFalse(unsafe.exists())

    def test_unignored_custom_staging_root_is_rejected_before_benchmark(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, _, _, _ = self.fixture_plan(
                temporary, staging_inside_repository=True,
            )
            repository = pathlib.Path(
                plan["planner_configuration"]["repository_root"],
            )
            subprocess.run(
                ["git", "init", "-q", str(repository)], check=True,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            task, = plan["tasks"]
            self.assertTrue(
                pathlib.Path(task["suggested_output_path"]).is_relative_to(repository),
            )
            with mock.patch.object(runner.system, "benchmark_file") as benchmark:
                with self.assertRaisesRegex(runner.CampaignError, "would dirty"):
                    runner.execute_one(plan, task["task_semantic_sha256"])
                benchmark.assert_not_called()

    def test_manifest_config_and_unknown_kind_mutations_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, _, experiments, manifest_path = self.fixture_plan(temporary)
            self.assertEqual(1, plan["task_count"])
            original_manifest = manifest_path.read_text(encoding="utf-8")
            manifest_path.write_text(original_manifest + " ", encoding="utf-8")
            with self.assertRaisesRegex(runner.CampaignError, "SHA-256 mismatch"):
                runner.plan_campaigns(
                    experiments,
                    repository_root=plan["planner_configuration"]["repository_root"],
                )

            manifest_path.write_text(original_manifest, encoding="utf-8")
            matrix = json.loads(experiments.read_text(encoding="utf-8"))
            matrix["common"]["search_defaults"]["max_realizations"] = 2
            experiments.write_text(json.dumps(matrix) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(runner.CampaignError, "max_realizations"):
                runner.plan_campaigns(
                    experiments,
                    repository_root=plan["planner_configuration"]["repository_root"],
                )

            matrix["common"]["search_defaults"]["max_realizations"] = 32
            matrix["campaigns"][0]["kind"] = "unknown_future_kind"
            experiments.write_text(json.dumps(matrix) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(runner.CampaignError, "unknown campaign kind"):
                runner.plan_campaigns(
                    experiments,
                    repository_root=plan["planner_configuration"]["repository_root"],
                )

            matrix["schema"]["version"] = 2
            experiments.write_text(json.dumps(matrix) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(runner.CampaignError, "schema version 1"):
                runner.plan_campaigns(
                    experiments,
                    repository_root=plan["planner_configuration"]["repository_root"],
                )


if __name__ == "__main__":
    unittest.main()
