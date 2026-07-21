import copy
import hashlib
import io
import json
import os
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
            "warmups": runner.GENERIC_WARMUPS if kind == "generic_codecs" else 0,
            "repetitions": runner.GENERIC_REPETITIONS if kind == "generic_codecs" else 1,
        }
        if kind in {"brevis_system", "brevis_parallel_scaling"}:
            campaign["configurations"] = list(configurations)
        if kind == "generic_codecs":
            campaign["registry"] = runner.GENERIC_REGISTRY_ID
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
                "schedule_seed": (
                    runner.GENERIC_SCHEDULE_SEED if kind == "generic_codecs" else 1
                ),
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
                "disk_reserve_fraction": (
                    runner.GENERIC_DISK_RESERVE_FRACTION
                    if kind == "generic_codecs" else 0.1
                ),
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

    @staticmethod
    def clean_git(task):
        return {
            "repository": str(ROOT),
            "commit": task["implementation_binding"]["git_commit"],
            "dirty": False,
            "status_porcelain": "",
        }

    @staticmethod
    def generic_raw_result(task, source, kwargs):
        specs = tuple(kwargs["specs"])
        source = pathlib.Path(source).resolve()
        source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest_path = pathlib.Path(kwargs["manifest_path"])
        measured_orders = runner.common._balanced_measured_orders(
            specs, kwargs["repetitions"], kwargs["schedule_seed"],
        )
        execution_schedule = []
        execution_order = 0
        for warmup_index in range(kwargs["warmups"]):
            rotation = warmup_index % len(specs)
            order = (*specs[rotation:], *specs[:rotation])
            for order_within_repetition, spec in enumerate(order):
                execution_schedule.append({
                    "execution_order": execution_order,
                    "phase": "warmup",
                    "repetition": warmup_index,
                    "order_within_repetition": order_within_repetition,
                    "method": spec.identifier,
                })
                execution_order += 1
        for measured_index, order in enumerate(measured_orders):
            for order_within_repetition, spec in enumerate(order):
                execution_schedule.append({
                    "execution_order": execution_order,
                    "phase": "measured",
                    "repetition": measured_index,
                    "order_within_repetition": order_within_repetition,
                    "method": spec.identifier,
                })
                execution_order += 1
        schedule_lookup = {
            (entry["phase"], entry["repetition"], entry["method"]): entry
            for entry in execution_schedule
        }

        def iteration(identifier, phase, index, archive_sha256):
            schedule = schedule_lookup[(phase, index, identifier)]
            return {
                "phase": phase,
                "index": index,
                "execution_order": schedule["execution_order"],
                "order_within_repetition": schedule["order_within_repetition"],
                "status_code": "ok",
                "success": True,
                "failure": None,
                "bit_exact": True,
                "compression": {
                    "status_code": "ok", "exit_code": 0,
                    "timed_out": False, "error": None,
                },
                "decompression": {
                    "status_code": "ok", "exit_code": 0,
                    "timed_out": False, "error": None,
                },
                "verification": {
                    "attempted": True,
                    "bit_exact": True,
                    "source_sha256": source_sha256,
                    "restored_sha256": source_sha256,
                    "restored_size_bytes": source.stat().st_size,
                    "error": None,
                },
                "compressed_size_bytes": 4,
                "archive_storage": {"logical_size_bytes": 4},
                "archive_sha256": archive_sha256,
                "archive_consistency": {
                    "status_code": (
                        "not_applicable_warmup" if phase == "warmup"
                        else "reference" if index == 0 else "consistent"
                    ),
                    "matches_reference": None if phase == "warmup" else True,
                },
            }

        methods = []
        for method_index, spec in enumerate(specs):
            archive_sha256 = f"{method_index + 1:064x}"
            executable_sha256 = f"{method_index + 101:064x}"
            methods.append({
                "id": spec.identifier,
                "spec": spec.to_dict(),
                "available": True,
                "resolved_executable": f"/mock/{spec.executable}",
                "executable_realpath": f"/mock/{spec.executable}",
                "executable_sha256": executable_sha256,
                "executable_sha256_after_runs": executable_sha256,
                "executable_unchanged_during_benchmark": True,
                "version": "fixture 1.0",
                "version_probe": {
                    "status_code": "ok", "exit_code": 0,
                    "timed_out": False, "error": None,
                },
                "warmups": [iteration(spec.identifier, "warmup", 0, archive_sha256)],
                "runs": [
                    iteration(spec.identifier, "measured", index, archive_sha256)
                    for index in range(runner.GENERIC_REPETITIONS)
                ],
                "failure": None,
                "status_code": "ok",
                "measured_archive_consistency": {
                    "status_code": "consistent",
                    "reference_size_bytes": 4,
                    "reference_sha256": archive_sha256,
                    "consistent_runs": runner.GENERIC_REPETITIONS,
                    "inconsistent_runs": 0,
                },
            })
        return {
            "schema": {
                "id": runner.common.SCHEMA_ID,
                "version": runner.common.SCHEMA_VERSION,
            },
            "status": "complete",
            "source": {
                "path": str(source),
                "size_bytes": source.stat().st_size,
                "sha256": source_sha256,
            },
            "integrity": {
                "model_metadata_declared": True,
                "source": {
                    "expected_size_bytes": source.stat().st_size,
                    "actual_size_bytes": source.stat().st_size,
                    "expected_sha256": source_sha256,
                    "actual_sha256": source_sha256,
                    "verified": True,
                },
                "manifest": {
                    "path": str(manifest_path),
                    "expected_sha256": kwargs["expected_manifest_sha256"],
                    "actual_sha256": kwargs["expected_manifest_sha256"],
                    "verified": True,
                },
                "manifest_binding": {
                    "tag": task["model"]["tag"],
                    "repo": task["model"]["repo"],
                    "revision": task["model"]["revision"],
                    "shard": task["input"]["file"],
                    "bytes": source.stat().st_size,
                    "sha256": source_sha256,
                    "verified": True,
                },
            },
            "input_metadata": dict(kwargs["input_metadata"]),
            "configuration": {
                "warmups": kwargs["warmups"],
                "repetitions": kwargs["repetitions"],
                "timeout_seconds": kwargs["timeout_seconds"],
                "version_probe_timeout_seconds": kwargs["version_probe_timeout_seconds"],
                "schedule_seed": kwargs["schedule_seed"],
                "schedule_balance": "paired",
                "keep_artifacts": kwargs["keep_artifacts"],
                "checkpoint_path": str(kwargs["checkpoint_path"]),
                "execution_schedule": execution_schedule,
                "measured_orders": [
                    {
                        "repetition": index,
                        "pair": index // 2,
                        "direction": "forward" if index % 2 == 0 else "reverse",
                        "methods": [spec.identifier for spec in order],
                    }
                    for index, order in enumerate(measured_orders)
                ],
            },
            "provenance": {
                "git": CampaignRunnerTests.clean_git(task),
                "benchmark_script": {
                    "path": str(pathlib.Path(runner.common.__file__).resolve()),
                    "sha256": task["implementation_binding"][
                        "generic_benchmarking_sha256"
                    ],
                },
            },
            "environment": {
                "cpu": {"logical_cpu_count": 1},
                "memory": {"total_bytes": 1},
                "filesystems": {
                    "source": {"path": str(source)},
                    "work": {"path": str(kwargs["work_dir"])},
                    "checkpoint": {"path": str(pathlib.Path(kwargs["checkpoint_path"]).parent)},
                },
                "codec_environment_variables": {
                    name: None for name in runner.common.CODEC_ENVIRONMENT_VARIABLES
                },
            },
            "methods": methods,
        }

    @classmethod
    def generic_benchmark_effect(cls, task, *, mutate=None, return_different=False):
        def effect(source, **kwargs):
            raw = cls.generic_raw_result(task, source, kwargs)
            if mutate is not None:
                mutate(raw)
            runner.common.write_json(kwargs["checkpoint_path"], raw, force=False)
            if return_different:
                returned = copy.deepcopy(raw)
                returned["status"] = "returned_object_must_not_be_trusted"
                return returned
            return raw

        return effect

    def test_plan_is_deterministic_manifest_frozen_and_task_unique(self):
        second = runner.plan_campaigns()
        self.assertEqual(157, self.plan["task_count"])
        self.assertEqual(157, self.plan["supported_task_count"])
        self.assertEqual(0, self.plan["unsupported_task_count"])
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
                self.assertEqual(task["baseline_specs"], semantics["baseline_specs"])
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

    def test_generic_campaign_is_one_exact_19_method_group_per_model_file(self):
        tasks = self.tasks("small-generic-codecs-v1")
        self.assertEqual(3, len(tasks))
        expected = [spec.to_dict() for spec in runner.common.BASELINE_SPECS]
        for task in tasks:
            self.assertTrue(task["executor"]["supported"])
            self.assertEqual("ready", task["executor"]["status_code"])
            self.assertEqual([], task["brevis_specs"])
            self.assertEqual(expected, task["baseline_specs"])
            self.assertEqual(expected, task["executor"]["generic_registry"])
            self.assertEqual(19, len(task["dimension"]["method_ids"]))
            self.assertEqual(1, task["configuration"]["warmups"])
            self.assertEqual(6, task["configuration"]["repetitions"])
            self.assertEqual(2701, task["configuration"]["schedule_seed"])
            self.assertEqual(
                "generic-codec-comparison", task["comparison_group"]["id"],
            )

    def test_generic_campaign_rejects_irrelevant_calibration_expansion(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, _, experiments, _ = self.fixture_plan(
                temporary, kind="generic_codecs",
            )
            matrix = json.loads(experiments.read_text(encoding="utf-8"))
            default_calibration = matrix["common"]["calibration_tensors"]
            matrix["campaigns"][0]["calibration_tensors"] = [
                default_calibration, default_calibration + 1,
            ]
            experiments.write_text(json.dumps(matrix) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(runner.CampaignError, "calibration axis"):
                runner.plan_campaigns(
                    experiments,
                    repository_root=plan["planner_configuration"]["repository_root"],
                    cache_root=plan["planner_configuration"]["cache_root"],
                    staging_root=plan["planner_configuration"]["staging_root"],
                )

    def test_generic_execute_uses_exact_contract_and_persisted_raw_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, source, _, _ = self.fixture_plan(
                temporary, kind="generic_codecs",
            )
            task, = plan["tasks"]
            effect = self.generic_benchmark_effect(task, return_different=True)
            with mock.patch.object(
                runner.common, "_git_provenance", return_value=self.clean_git(task),
            ), mock.patch.object(
                runner.common, "benchmark_file", side_effect=effect,
            ) as benchmark:
                receipt = runner.execute_one(plan, task["task_semantic_sha256"])

            self.assertTrue(receipt["result_success"])
            self.assertTrue(receipt["formal_eligible"])
            self.assertTrue(receipt["attempt_contract_valid"])
            self.assertEqual(task["task_semantics"], receipt["task_semantics"])
            benchmark.assert_called_once()
            args, kwargs = benchmark.call_args
            self.assertEqual((source,), args)
            self.assertEqual(
                [spec.to_dict() for spec in runner.common.BASELINE_SPECS],
                [spec.to_dict() for spec in kwargs["specs"]],
            )
            self.assertEqual(19, len(kwargs["specs"]))
            self.assertEqual(1, kwargs["warmups"])
            self.assertEqual(6, kwargs["repetitions"])
            self.assertEqual(2701, kwargs["schedule_seed"])
            self.assertFalse(kwargs["keep_artifacts"])
            self.assertFalse(kwargs["force_checkpoint"])
            self.assertEqual(
                task["task_semantic_sha256"],
                kwargs["input_metadata"]["campaign_task_semantic_sha256"],
            )
            self.assertEqual(
                task["implementation_binding"]["campaign_runner_sha256"],
                kwargs["input_metadata"]["campaign_runner_sha256"],
            )
            raw_path = pathlib.Path(task["suggested_output_path"])
            receipt_path = pathlib.Path(task["suggested_receipt_path"])
            self.assertTrue(raw_path.is_file())
            self.assertTrue(receipt_path.is_file())
            persisted_raw = json.loads(raw_path.read_text(encoding="utf-8"))
            persisted_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual("complete", persisted_raw["status"])
            self.assertEqual(receipt, persisted_receipt)
            self.assertEqual(
                hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                receipt["raw_result"]["sha256"],
            )
            self.assertEqual(raw_path.stat().st_size, receipt["raw_result"]["size_bytes"])

    def test_generic_preflight_rejects_environment_dirty_tree_and_disk_pressure(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, _, _, _ = self.fixture_plan(
                pathlib.Path(temporary) / "environment", kind="generic_codecs",
            )
            task, = plan["tasks"]
            with mock.patch.dict(os.environ, {"XZ_OPT": ""}, clear=False), mock.patch.object(
                runner.common, "benchmark_file",
            ) as benchmark:
                with self.assertRaisesRegex(runner.CampaignError, "truly unset"):
                    runner.execute_one(plan, task["task_semantic_sha256"])
                benchmark.assert_not_called()

        with tempfile.TemporaryDirectory() as temporary:
            plan, _, _, _ = self.fixture_plan(
                pathlib.Path(temporary) / "dirty", kind="generic_codecs",
            )
            task, = plan["tasks"]
            dirty = {**self.clean_git(task), "dirty": True, "status_porcelain": " M file"}
            with mock.patch.object(
                runner.common, "_git_provenance", return_value=dirty,
            ), mock.patch.object(runner.common, "benchmark_file") as benchmark:
                with self.assertRaisesRegex(runner.CampaignError, "clean Git"):
                    runner.execute_one(plan, task["task_semantic_sha256"])
                benchmark.assert_not_called()

        with tempfile.TemporaryDirectory() as temporary:
            plan, _, _, _ = self.fixture_plan(
                pathlib.Path(temporary) / "disk", kind="generic_codecs",
            )
            task, = plan["tasks"]
            failed_gate = {
                "spec_id": "brevis.generic-disk-gate.v1",
                "reserve_fraction": 0.3,
                "passes": False,
            }
            with mock.patch.object(
                runner.common, "_git_provenance", return_value=self.clean_git(task),
            ), mock.patch.object(
                runner, "_generic_disk_gate", return_value=failed_gate,
            ), mock.patch.object(runner.common, "benchmark_file") as benchmark:
                with self.assertRaisesRegex(runner.CampaignError, "disk-reserve gate"):
                    runner.execute_one(plan, task["task_semantic_sha256"])
                benchmark.assert_not_called()

    def test_generic_method_and_provenance_failures_preserve_raw_and_receipt(self):
        def fail_method(raw):
            raw["methods"][3]["available"] = False
            raw["methods"][3]["status_code"] = "executable_missing"
            raw["methods"][3]["failure"] = "fixture missing executable"

        def drift_provenance(raw):
            raw["provenance"]["git"]["commit"] = "b" * 40

        def drift_provenance_paths(raw):
            raw["configuration"]["checkpoint_path"] = "/missing/checkpoint.json"
            raw["integrity"]["manifest"]["path"] = "/missing/manifest.json"
            raw["provenance"]["benchmark_script"]["path"] = "/missing/benchmarking.py"

        for label, mutate, expected_code, expected_result, expected_attempt in (
            ("method", fail_method, "method_validation_failed", False, False),
            ("provenance", drift_provenance, "result_git_provenance", True, False),
            ("paths", drift_provenance_paths, "configuration", True, False),
        ):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                plan, _, _, _ = self.fixture_plan(
                    temporary, kind="generic_codecs",
                )
                task, = plan["tasks"]
                with mock.patch.object(
                    runner.common, "_git_provenance", return_value=self.clean_git(task),
                ), mock.patch.object(
                    runner.common,
                    "benchmark_file",
                    side_effect=self.generic_benchmark_effect(task, mutate=mutate),
                ):
                    receipt = runner.execute_one(plan, task["task_semantic_sha256"])
                self.assertEqual(expected_result, receipt["result_success"])
                self.assertEqual(expected_attempt, receipt["attempt_contract_valid"])
                self.assertFalse(receipt["formal_eligible"])
                self.assertIn(
                    expected_code,
                    {item["code"] for item in receipt["postvalidation"]["failures"]},
                )
                raw = json.loads(pathlib.Path(
                    task["suggested_output_path"],
                ).read_text(encoding="utf-8"))
                self.assertEqual(19, len(raw["methods"]))
                self.assertTrue(pathlib.Path(task["suggested_receipt_path"]).is_file())

    def test_generic_missing_codec_is_valid_failed_attempt_without_schedule_rows(self):
        def remove_codec_before_schedule(raw):
            missing = raw["methods"][4]
            missing_id = missing["id"]
            missing.update({
                "available": False,
                "resolved_executable": None,
                "executable_realpath": None,
                "executable_sha256": None,
                "executable_sha256_after_runs": None,
                "executable_unchanged_during_benchmark": None,
                "version": None,
                "version_probe": None,
                "warmups": [],
                "runs": [],
                "failure": "executable not found on PATH: fixture",
                "status_code": "executable_missing",
                "measured_archive_consistency": {
                    "status_code": "not_checked",
                    "reference_size_bytes": None,
                    "reference_sha256": None,
                    "consistent_runs": 0,
                    "inconsistent_runs": 0,
                },
            })
            schedule = [
                entry for entry in raw["configuration"]["execution_schedule"]
                if entry["method"] != missing_id
            ]
            for execution_order, entry in enumerate(schedule):
                entry["execution_order"] = execution_order
            raw["configuration"]["execution_schedule"] = schedule
            lookup = {
                (entry["phase"], entry["repetition"], entry["method"]): entry
                for entry in schedule
            }
            for method in raw["methods"]:
                for iteration in (*method["warmups"], *method["runs"]):
                    entry = lookup[(iteration["phase"], iteration["index"], method["id"])]
                    iteration["execution_order"] = entry["execution_order"]

        with tempfile.TemporaryDirectory() as temporary:
            plan, _, _, _ = self.fixture_plan(temporary, kind="generic_codecs")
            task, = plan["tasks"]
            with mock.patch.object(
                runner.common, "_git_provenance", return_value=self.clean_git(task),
            ), mock.patch.object(
                runner.common,
                "benchmark_file",
                side_effect=self.generic_benchmark_effect(
                    task, mutate=remove_codec_before_schedule,
                ),
            ):
                receipt = runner.execute_one(plan, task["task_semantic_sha256"])
            self.assertTrue(receipt["attempt_contract_valid"])
            self.assertFalse(receipt["result_success"])
            self.assertFalse(receipt["formal_eligible"])
            failure_codes = {
                failure["code"] for failure in receipt["postvalidation"]["failures"]
            }
            self.assertEqual({"method_pre_run_failure"}, failure_codes)

    def test_generic_post_run_disk_reserve_failure_is_nonformal(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, _, _, _ = self.fixture_plan(temporary, kind="generic_codecs")
            task, = plan["tasks"]

            def failed_post_gate(preflight, output):
                del output
                return {
                    **preflight,
                    "post_run_free_bytes": preflight["reserved_bytes"] - 1,
                    "post_run_reserved_bytes": preflight["reserved_bytes"],
                    "post_run_passes": False,
                    "post_run_error": None,
                }

            with mock.patch.object(
                runner.common, "_git_provenance", return_value=self.clean_git(task),
            ), mock.patch.object(
                runner, "_generic_post_run_disk_gate", side_effect=failed_post_gate,
            ), mock.patch.object(
                runner.common,
                "benchmark_file",
                side_effect=self.generic_benchmark_effect(task),
            ):
                receipt = runner.execute_one(plan, task["task_semantic_sha256"])
            self.assertFalse(receipt["formal_eligible"])
            self.assertTrue(receipt["result_success"])
            self.assertTrue(receipt["attempt_contract_valid"])
            self.assertFalse(receipt["disk_gate"]["post_run_passes"])
            self.assertIn(
                "post_run_disk_reserve",
                {failure["code"] for failure in receipt["postvalidation"]["failures"]},
            )

    def test_generic_schedule_and_task_spec_drift_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, _, _, _ = self.fixture_plan(
                pathlib.Path(temporary) / "schedule", kind="generic_codecs",
            )
            task, = plan["tasks"]

            def drift_schedule(raw):
                raw["configuration"]["measured_orders"][1]["methods"] = list(
                    raw["configuration"]["measured_orders"][0]["methods"]
                )

            with mock.patch.object(
                runner.common, "_git_provenance", return_value=self.clean_git(task),
            ), mock.patch.object(
                runner.common,
                "benchmark_file",
                side_effect=self.generic_benchmark_effect(task, mutate=drift_schedule),
            ):
                receipt = runner.execute_one(plan, task["task_semantic_sha256"])
            self.assertFalse(receipt["formal_eligible"])
            self.assertTrue(receipt["result_success"])
            self.assertFalse(receipt["attempt_contract_valid"])
            self.assertIn(
                "balanced_measured_orders",
                {item["code"] for item in receipt["postvalidation"]["failures"]},
            )

        with tempfile.TemporaryDirectory() as temporary:
            plan, _, _, _ = self.fixture_plan(
                pathlib.Path(temporary) / "seed", kind="generic_codecs",
            )
            task, = plan["tasks"]

            def preserve_pairing_but_drift_seed(raw):
                orders = raw["configuration"]["measured_orders"]
                orders[0]["methods"][0], orders[0]["methods"][1] = (
                    orders[0]["methods"][1], orders[0]["methods"][0],
                )
                orders[1]["methods"] = list(reversed(orders[0]["methods"]))
                method_ids = [method["id"] for method in raw["methods"]]
                schedule = []
                execution_order = 0
                for order_within, identifier in enumerate(method_ids):
                    schedule.append({
                        "execution_order": execution_order,
                        "phase": "warmup",
                        "repetition": 0,
                        "order_within_repetition": order_within,
                        "method": identifier,
                    })
                    execution_order += 1
                for repetition, order in enumerate(orders):
                    for order_within, identifier in enumerate(order["methods"]):
                        schedule.append({
                            "execution_order": execution_order,
                            "phase": "measured",
                            "repetition": repetition,
                            "order_within_repetition": order_within,
                            "method": identifier,
                        })
                        execution_order += 1
                raw["configuration"]["execution_schedule"] = schedule
                lookup = {
                    (entry["phase"], entry["repetition"], entry["method"]): entry
                    for entry in schedule
                }
                for method in raw["methods"]:
                    for iteration in (*method["warmups"], *method["runs"]):
                        entry = lookup[
                            (iteration["phase"], iteration["index"], method["id"])
                        ]
                        iteration["execution_order"] = entry["execution_order"]
                        iteration["order_within_repetition"] = entry[
                            "order_within_repetition"
                        ]

            with mock.patch.object(
                runner.common, "_git_provenance", return_value=self.clean_git(task),
            ), mock.patch.object(
                runner.common,
                "benchmark_file",
                side_effect=self.generic_benchmark_effect(
                    task, mutate=preserve_pairing_but_drift_seed,
                ),
            ):
                receipt = runner.execute_one(plan, task["task_semantic_sha256"])
            self.assertFalse(receipt["formal_eligible"])
            self.assertFalse(receipt["attempt_contract_valid"])
            self.assertIn(
                "balanced_measured_orders",
                {item["code"] for item in receipt["postvalidation"]["failures"]},
            )

        with tempfile.TemporaryDirectory() as temporary:
            plan, _, _, _ = self.fixture_plan(
                pathlib.Path(temporary) / "spec", kind="generic_codecs",
            )
            forged = copy.deepcopy(plan)
            task = forged["tasks"][0]
            task["task_semantics"]["baseline_specs"][0]["compress_args"].append("--drift")
            task["baseline_specs"] = copy.deepcopy(task["task_semantics"]["baseline_specs"])
            task["executor"]["generic_registry"] = copy.deepcopy(task["baseline_specs"])
            task["task_semantic_sha256"] = runner._semantic_sha256(task["task_semantics"])
            with mock.patch.object(runner.common, "benchmark_file") as benchmark:
                with self.assertRaisesRegex(runner.CampaignError, "not generated"):
                    runner.execute_one(forged, task["task_semantic_sha256"])
                benchmark.assert_not_called()

    def test_generic_repeated_measured_slot_and_post_run_environment_fail(self):
        def repeat_first_measured_row(raw):
            first = copy.deepcopy(raw["methods"][0]["runs"][0])
            raw["methods"][0]["runs"] = [
                copy.deepcopy(first) for _ in range(runner.GENERIC_REPETITIONS)
            ]

        with tempfile.TemporaryDirectory() as temporary:
            plan, _, _, _ = self.fixture_plan(
                pathlib.Path(temporary) / "slots", kind="generic_codecs",
            )
            task, = plan["tasks"]
            with mock.patch.object(
                runner.common, "_git_provenance", return_value=self.clean_git(task),
            ), mock.patch.object(
                runner.common,
                "benchmark_file",
                side_effect=self.generic_benchmark_effect(
                    task, mutate=repeat_first_measured_row,
                ),
            ):
                receipt = runner.execute_one(plan, task["task_semantic_sha256"])
            self.assertFalse(receipt["formal_eligible"])
            first_outcome = receipt["method_outcomes"][0]
            self.assertFalse(first_outcome["all_iterations_bit_exact"])
            self.assertFalse(receipt["result_success"])
            self.assertFalse(receipt["attempt_contract_valid"])

        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ, {}, clear=False,
        ):
            os.environ.pop("XZ_OPT", None)
            plan, _, _, _ = self.fixture_plan(
                pathlib.Path(temporary) / "environment", kind="generic_codecs",
            )
            task, = plan["tasks"]
            base_effect = self.generic_benchmark_effect(task)

            def change_environment_after_raw(source, **kwargs):
                raw = base_effect(source, **kwargs)
                os.environ["XZ_OPT"] = "-9"
                return raw

            with mock.patch.object(
                runner.common, "_git_provenance", return_value=self.clean_git(task),
            ), mock.patch.object(
                runner.common, "benchmark_file", side_effect=change_environment_after_raw,
            ):
                receipt = runner.execute_one(plan, task["task_semantic_sha256"])
            self.assertFalse(receipt["formal_eligible"])
            self.assertTrue(receipt["result_success"])
            self.assertTrue(receipt["attempt_contract_valid"])
            self.assertIn(
                "codec_environment_changed_during_run",
                {failure["code"] for failure in receipt["postvalidation"]["failures"]},
            )

    def test_generic_harness_exception_preserves_truthful_raw_and_failed_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, _, _, _ = self.fixture_plan(temporary, kind="generic_codecs")
            task, = plan["tasks"]
            with mock.patch.object(
                runner.common, "_git_provenance", return_value=self.clean_git(task),
            ), mock.patch.object(
                runner.common, "benchmark_file", side_effect=RuntimeError("fixture crash"),
            ):
                receipt = runner.execute_one(plan, task["task_semantic_sha256"])
            self.assertFalse(receipt["formal_eligible"])
            self.assertEqual("RuntimeError", receipt["benchmark_exception"]["type"])
            raw_path = pathlib.Path(task["suggested_output_path"])
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            self.assertEqual("brevis.generic-campaign-harness-failure", raw["schema"]["id"])
            self.assertEqual(task["task_semantic_sha256"], raw["campaign_task_semantic_sha256"])
            self.assertTrue(pathlib.Path(task["suggested_receipt_path"]).is_file())

    def test_generic_interrupt_writes_evidence_then_reraises(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, _, _, _ = self.fixture_plan(temporary, kind="generic_codecs")
            task, = plan["tasks"]
            with mock.patch.object(
                runner.common, "_git_provenance", return_value=self.clean_git(task),
            ), mock.patch.object(
                runner.common, "benchmark_file", side_effect=KeyboardInterrupt(),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    runner.execute_one(plan, task["task_semantic_sha256"])
            receipt = json.loads(pathlib.Path(
                task["suggested_receipt_path"],
            ).read_text(encoding="utf-8"))
            self.assertFalse(receipt["formal_eligible"])
            self.assertEqual("KeyboardInterrupt", receipt["benchmark_exception"]["type"])
            self.assertTrue(pathlib.Path(task["suggested_output_path"]).is_file())

    def test_generic_postvalidation_interrupt_writes_receipt_then_reraises(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan, _, _, _ = self.fixture_plan(temporary, kind="generic_codecs")
            task, = plan["tasks"]
            with mock.patch.object(
                runner.common, "_git_provenance", return_value=self.clean_git(task),
            ), mock.patch.object(
                runner.common,
                "benchmark_file",
                side_effect=self.generic_benchmark_effect(task),
            ), mock.patch.object(
                runner, "_validate_generic_result", side_effect=KeyboardInterrupt(),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    runner.execute_one(plan, task["task_semantic_sha256"])
            receipt = json.loads(pathlib.Path(
                task["suggested_receipt_path"],
            ).read_text(encoding="utf-8"))
            self.assertFalse(receipt["formal_eligible"])
            self.assertIn(
                "postvalidation_exception",
                {failure["code"] for failure in receipt["postvalidation"]["failures"]},
            )

    def test_generic_post_run_disk_gate_recomputes_reserve_from_current_capacity(self):
        preflight = {
            "reserve_fraction": 0.3,
            "reserved_bytes": 300,
            "passes": True,
        }
        usage = mock.Mock(total=2000, free=400)
        with mock.patch.object(runner.shutil, "disk_usage", return_value=usage):
            gate = runner._generic_post_run_disk_gate(preflight, pathlib.Path("/tmp/out"))
        self.assertEqual(600, gate["post_run_reserved_bytes"])
        self.assertFalse(gate["post_run_passes"])

    def test_generic_raw_and_receipt_no_overwrite_and_shared_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            fixed_lock = root / "fixed-machine.lock"
            with mock.patch.object(runner, "DEFAULT_EXECUTION_LOCK", fixed_lock):
                plan, _, _, _ = self.fixture_plan(root / "locked", kind="generic_codecs")
                task, = plan["tasks"]
                with runner._exclusive_execution_lock(fixed_lock), mock.patch.object(
                    runner.common, "_git_provenance", return_value=self.clean_git(task),
                ), mock.patch.object(runner.common, "benchmark_file") as benchmark:
                    with self.assertRaisesRegex(runner.CampaignError, "advisory lock"):
                        runner.execute_one(plan, task["task_semantic_sha256"])
                    benchmark.assert_not_called()

        for existing in ("raw", "receipt", "raw_symlink", "receipt_symlink"):
            with self.subTest(existing=existing), tempfile.TemporaryDirectory() as temporary:
                plan, _, _, _ = self.fixture_plan(temporary, kind="generic_codecs")
                task, = plan["tasks"]
                target = pathlib.Path(
                    task["suggested_output_path"]
                    if existing.startswith("raw") else task["suggested_receipt_path"]
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                if existing.endswith("symlink"):
                    target.symlink_to(target.parent / "missing-target")
                else:
                    target.write_text("existing\n", encoding="utf-8")
                with mock.patch.object(runner.common, "benchmark_file") as benchmark:
                    with self.assertRaisesRegex(runner.CampaignError, "overwrite"):
                        runner.execute_one(plan, task["task_semantic_sha256"])
                    benchmark.assert_not_called()

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
                ("generic_benchmarking_sha256", "3" * 64),
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

        with tempfile.TemporaryDirectory() as temporary:
            plan, _, _, _ = self.fixture_plan(
                temporary, stage=2, kind="generic_codecs",
            )
            task, = plan["tasks"]
            with mock.patch.object(
                runner.common, "_git_provenance", return_value=self.clean_git(task),
            ), mock.patch.object(
                runner.common,
                "benchmark_file",
                side_effect=self.generic_benchmark_effect(task),
            ):
                receipt = runner.execute_one(
                    plan, task["task_semantic_sha256"], allow_unmet_stage_gate=True,
                )
            self.assertTrue(receipt["result_success"])
            self.assertTrue(receipt["attempt_contract_valid"])
            self.assertFalse(receipt["formal_eligible"])
            self.assertEqual([reason], receipt["stage_override_reasons"])

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
