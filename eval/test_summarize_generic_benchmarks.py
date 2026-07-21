import concurrent.futures
import hashlib
import json
import pathlib
import tempfile
import threading
import unittest

import benchmarking
import summarize_generic_benchmarks as summary


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _operation(
    output_bytes: int, wall_ns: int, rss_bytes: int, command: list[str],
) -> dict:
    return {
        "status_code": "ok",
        "started_at_utc": "2026-07-21T00:00:00Z",
        "ended_at_utc": "2026-07-21T00:00:01Z",
        "command": command,
        "wall_time_ns": wall_ns,
        "timeout_seconds": 3600.0,
        "timed_out": False,
        "direct_child_max_rss_bytes": rss_bytes,
        "direct_child_user_cpu_time_ns": wall_ns // 2,
        "direct_child_system_cpu_time_ns": wall_ns // 4,
        "exit_code": 0,
        "term_signal": None,
        "process_group_isolated": True,
        "started_new_session": True,
        "process_group_id": 123,
        "wait_strategy": "pidfd_wait4",
        "wait_strategy_detail": "fixture",
        "stdout": {"text": "", "base64": "", "size_bytes": 0},
        "stderr": {"text": "", "base64": "", "size_bytes": 0},
        "error": None,
        "output_size_bytes": output_bytes,
        "output_storage": {
            "logical_size_bytes": output_bytes,
            "st_blocks_512b": 1,
            "allocated_size_bytes": 512,
        },
    }


def _iteration(
    *, phase: str, repetition: int, execution_order: int, order_within: int,
    source_bytes: int, source_sha: str, archive_bytes: int, archive_sha: str,
    wall_index: int, spec: dict,
) -> dict:
    compression_wall = wall_index * 1_000_000_000
    decompression_wall = wall_index * 500_000_000
    executable = f"/usr/bin/{spec['executable']}"
    substitutions = {"input": "/tmp/fixture/input.bin", "output": "/tmp/fixture/output.bin"}
    compression_command = [
        executable,
        *(argument.format_map(substitutions) for argument in spec["compress_args"]),
    ]
    decompression_command = [
        executable,
        *(argument.format_map(substitutions) for argument in spec["decompress_args"]),
    ]
    if phase == "warmup":
        consistency = {"status_code": "not_applicable_warmup", "matches_reference": None}
    else:
        consistency = {
            "status_code": "reference" if repetition == 0 else "consistent",
            "matches_reference": True,
            "reference_size_bytes": archive_bytes,
            "reference_sha256": archive_sha,
        }
    return {
        "status_code": "ok",
        "started_at_utc": "2026-07-21T00:00:00Z",
        "ended_at_utc": "2026-07-21T00:00:01Z",
        "phase": phase,
        "index": repetition,
        "execution_order": execution_order,
        "order_within_repetition": order_within,
        "compression": _operation(
            archive_bytes, compression_wall, wall_index * 1024, compression_command,
        ),
        "compressed_size_bytes": archive_bytes,
        "archive_storage": {
            "logical_size_bytes": archive_bytes,
            "st_blocks_512b": 1,
            "allocated_size_bytes": 512,
        },
        "archive_sha256": archive_sha,
        "archive_hash_error": None,
        "archive_consistency": consistency,
        "decompression": _operation(
            source_bytes, decompression_wall, wall_index * 2048, decompression_command,
        ),
        "verification": {
            "attempted": True,
            "bit_exact": True,
            "source_sha256": source_sha,
            "restored_sha256": source_sha,
            "restored_size_bytes": source_bytes,
            "restored_storage": {
                "logical_size_bytes": source_bytes,
                "st_blocks_512b": 1,
                "allocated_size_bytes": 512,
            },
            "error": None,
        },
        "bit_exact": True,
        "success": True,
        "failure": None,
        "staging": {
            "source": "independent_copy", "archive": "independent_copy",
            "excluded_from_timing": True,
        },
        "artifact_directory": "/tmp/removed-fixture",
    }


class PairFixture:
    def __init__(
        self, root: pathlib.Path, *, tag: str = "model-a", source_bytes: int = 100_000,
        timing_bias: int = 0, shard: str = "model.safetensors",
        archive_fraction: float | None = None,
    ):
        self.root = root
        self.tag = tag
        self.source_bytes = source_bytes
        self.timing_bias = timing_bias
        self.shard = shard
        self.archive_fraction = archive_fraction
        self.raw_path = root / f"{tag}.json"
        self.receipt_path = root / f"{tag}.validation.json"
        self.registry = [spec.to_dict() for spec in benchmarking.BASELINE_SPECS]
        self.identifiers = [spec.identifier for spec in benchmarking.BASELINE_SPECS]
        self.source_sha = _sha(f"source:{tag}:{shard}:{source_bytes}")
        self.manifest_sha = _sha("shared formal model manifest")
        self.matrix_sha = _sha("matrix")
        self.implementation = {
            "git_commit": "a" * 40,
            "campaign_runner_sha256": "b" * 64,
            "brevis_benchmarking_sha256": "c" * 64,
            "generic_benchmarking_sha256": "d" * 64,
        }
        self.preflight_gate = self._disk_gate()
        self.post_gate = {
            **self.preflight_gate,
            "post_run_filesystem_total_bytes": self.preflight_gate["filesystem_total_bytes"],
            "post_run_free_bytes": 8_000_000_000,
            "post_run_reserved_bytes": self.preflight_gate["reserved_bytes"],
            "post_run_passes": True,
            "post_run_error": None,
        }
        self.task = self._task()
        self.task_sha = summary._canonical_sha256(self.task)
        self.raw = self._raw()
        self.receipt = self._receipt()
        self.publish()

    def _disk_gate(self) -> dict:
        total = 10_000_000_000
        free = 9_000_000_000
        fixed = 512 << 20
        per_iteration = 128 << 10
        estimate = self.source_bytes * 5 + fixed + 19 * 7 * per_iteration
        reserved = 3_000_000_000
        return {
            "spec_id": "brevis.generic-disk-gate.v1",
            "path": str(self.root),
            "filesystem_total_bytes": total,
            "filesystem_free_bytes": free,
            "reserve_fraction": 0.30,
            "reserved_bytes": reserved,
            "source_bytes": self.source_bytes,
            "source_multiplier": 5,
            "fixed_bytes": fixed,
            "checkpoint_bytes_per_iteration": per_iteration,
            "method_count": 19,
            "iterations_per_method": 7,
            "estimated_work_and_checkpoint_bytes": estimate,
            "projected_free_bytes": free - estimate,
            "passes": True,
            "limitation": "fixture shared-filesystem estimate",
        }

    def _task(self) -> dict:
        return {
            "spec_id": summary.TASK_SPEC_ID,
            "experiment_matrix_sha256": self.matrix_sha,
            "model_manifest_sha256": self.manifest_sha,
            "implementation_binding": self.implementation,
            "machine_binding": None,
            "campaign_id": "small-generic-codecs-v1",
            "campaign_kind": "generic_codecs",
            "campaign_stage": 1,
            "campaign_resource_gate": None,
            "model": {
                "tag": self.tag,
                "repo": f"fixture/{self.tag}",
                "revision": "1" * 40,
                "scope": "full_checkpoint",
            },
            "input": {
                "file": self.shard,
                "bytes": self.source_bytes,
                "sha256": self.source_sha,
            },
            "comparison_group_id": "generic-codec-comparison",
            "dimension": {
                "registry": summary.REGISTRY_ID,
                "method_ids": self.identifiers,
                "comparison_semantics": (
                    "all 19 pinned registry entries share one balanced harness invocation"
                ),
            },
            "comparison_schedule": {
                "within_group_balanced": True,
                "outer_axis_balanced": True,
                "status_code": "within_group_balanced_outer_axis_balanced",
                "limitation": None,
            },
            "brevis_specs": [],
            "baseline_specs": self.registry,
            "jobs": 1,
            "calibration_tensors": 200,
            "warmups": 1,
            "repetitions": 6,
            "timeout_seconds_per_process": 3600.0,
            "version_probe_timeout_seconds": 5.0,
            "schedule_seed": 2701,
            "disk_reserve_fraction": 0.30,
            "executor": {
                "kind": "generic_baseline_harness",
                "supported": True,
                "status_code": "ready",
                "implementation": "benchmarking.benchmark_file",
                "formal_gates": {
                    "require_declared_integrity": True,
                    "require_clean_git": True,
                    "require_codec_environment_unset": True,
                    "require_exact_registry": True,
                    "require_bit_exact_all_iterations": True,
                    "require_stable_measured_archives": True,
                    "require_unchanged_executables": True,
                    "force_checkpoint": False,
                },
                "exclusive_execution_policy": {
                    "policy_id": "brevis.machine-advisory-execution-lock.v1",
                    "required": True,
                    "backend": "fcntl.flock",
                    "acquisition": "exclusive_nonblocking",
                    "lock_path": "/tmp/brevis-machine-benchmark.lock",
                    "scope": "fixture process namespace",
                    "limitations": "fixture advisory lock",
                },
            },
        }

    def _orders_and_schedule(self) -> tuple[list[dict], list[dict], dict]:
        orders = []
        schedule = []
        positions = {}
        execution_order = 0
        for order_within, identifier in enumerate(self.identifiers):
            schedule.append({
                "execution_order": execution_order,
                "phase": "warmup",
                "repetition": 0,
                "order_within_repetition": order_within,
                "method": identifier,
            })
            positions[("warmup", 0, identifier)] = (execution_order, order_within)
            execution_order += 1
        measured_orders = benchmarking._balanced_measured_orders(
            benchmarking.BASELINE_SPECS, 6, 2701,
        )
        for repetition, spec_order in enumerate(measured_orders):
            methods = [spec.identifier for spec in spec_order]
            orders.append({
                "repetition": repetition,
                "pair": repetition // 2,
                "direction": "forward" if repetition % 2 == 0 else "reverse",
                "methods": list(methods),
            })
            for order_within, identifier in enumerate(methods):
                schedule.append({
                    "execution_order": execution_order,
                    "phase": "measured",
                    "repetition": repetition,
                    "order_within_repetition": order_within,
                    "method": identifier,
                })
                positions[("measured", repetition, identifier)] = (
                    execution_order, order_within,
                )
                execution_order += 1
        return orders, schedule, positions

    def _archive_identity(self, identifier: str, index: int) -> tuple[int, str]:
        if identifier == "raw/copy":
            size = self.source_bytes
        elif self.archive_fraction is not None:
            size = int(self.source_bytes * self.archive_fraction)
        elif identifier in {"bzip2/default", "bzip2/ratio"}:
            size = self.source_bytes - 5_000
        else:
            size = self.source_bytes - (index + 1) * 100
        independent = (
            "bzip2/level-9"
            if identifier in {"bzip2/default", "bzip2/ratio"} else identifier
        )
        return size, _sha(f"archive:{self.source_sha}:{independent}")

    def _method(self, spec: dict, method_index: int, positions: dict) -> dict:
        identifier = spec["id"]
        archive_bytes, archive_sha = self._archive_identity(identifier, method_index)
        warmup_order, warmup_within = positions[("warmup", 0, identifier)]
        warmup = _iteration(
            phase="warmup", repetition=0, execution_order=warmup_order,
            order_within=warmup_within, source_bytes=self.source_bytes,
            source_sha=self.source_sha, archive_bytes=archive_bytes,
            archive_sha=archive_sha, wall_index=100 + method_index, spec=spec,
        )
        runs = []
        for repetition in range(6):
            execution_order, order_within = positions[("measured", repetition, identifier)]
            runs.append(_iteration(
                phase="measured", repetition=repetition,
                execution_order=execution_order, order_within=order_within,
                source_bytes=self.source_bytes, source_sha=self.source_sha,
                archive_bytes=archive_bytes, archive_sha=archive_sha,
                wall_index=self.timing_bias + method_index + repetition + 1,
                spec=spec,
            ))
        executable_sha = _sha(f"executable:{spec['method']}")
        return {
            "id": identifier,
            "spec": spec,
            "available": True,
            "resolved_executable": f"/usr/bin/{spec['executable']}",
            "executable_realpath": f"/usr/bin/{spec['executable']}",
            "executable_sha256": executable_sha,
            "executable_sha256_after_runs": executable_sha,
            "executable_unchanged_during_benchmark": True,
            "version": f"{spec['executable']} fixture 1.0",
            "version_probe": {
                "status_code": "ok", "exit_code": 0, "timed_out": False,
                "error": None, "timeout_seconds": 5.0,
                "command": [f"/usr/bin/{spec['executable']}", *spec["version_args"]],
            },
            "warmups": [warmup],
            "runs": runs,
            "failure": None,
            "status_code": "ok",
            "measured_archive_consistency": {
                "status_code": "consistent",
                "reference_size_bytes": archive_bytes,
                "reference_sha256": archive_sha,
                "consistent_runs": 6,
                "inconsistent_runs": 0,
            },
        }

    def _raw(self) -> dict:
        orders, execution_schedule, positions = self._orders_and_schedule()
        methods = [
            self._method(spec, index, positions) for index, spec in enumerate(self.registry)
        ]
        manifest_path = str(self.root / "manifest.json")
        return {
            "schema": summary.INPUT_SCHEMA,
            "status": "complete",
            "started_at_utc": "2026-07-21T00:00:00Z",
            "ended_at_utc": "2026-07-21T01:00:00Z",
            "source": {
                "path": str(self.root / self.shard),
                "size_bytes": self.source_bytes,
                "sha256": self.source_sha,
            },
            "integrity": {
                "model_metadata_declared": True,
                "source": {
                    "expected_size_bytes": self.source_bytes,
                    "actual_size_bytes": self.source_bytes,
                    "expected_sha256": self.source_sha,
                    "actual_sha256": self.source_sha,
                    "verified": True,
                },
                "manifest": {
                    "path": manifest_path,
                    "size_bytes": 100,
                    "expected_sha256": self.manifest_sha,
                    "actual_sha256": self.manifest_sha,
                    "verified": True,
                },
                "manifest_binding": {
                    "tag": self.tag,
                    "repo": f"fixture/{self.tag}",
                    "revision": "1" * 40,
                    "shard": self.shard,
                    "bytes": self.source_bytes,
                    "sha256": self.source_sha,
                    "entry_sha256": _sha(f"entry:{self.tag}"),
                    "verified": True,
                },
            },
            "input_metadata": {
                "model_tag": self.tag,
                "model_repo": f"fixture/{self.tag}",
                "model_revision": "1" * 40,
                "manifest": manifest_path,
                "shard": self.shard,
                "campaign_id": "small-generic-codecs-v1",
                "campaign_comparison_group_id": "generic-codec-comparison",
                "campaign_task_semantic_sha256": self.task_sha,
                "campaign_experiment_matrix_sha256": self.matrix_sha,
                "campaign_runner_sha256": self.implementation["campaign_runner_sha256"],
                "generic_benchmarking_sha256": self.implementation[
                    "generic_benchmarking_sha256"
                ],
                "generic_registry_sha256": summary._canonical_sha256(self.registry),
                "campaign_disk_gate": self.preflight_gate,
            },
            "configuration": {
                "warmups": 1,
                "repetitions": 6,
                "timeout_seconds": 3600.0,
                "version_probe_timeout_seconds": 5.0,
                "termination_grace_seconds": benchmarking.PROCESS_TERMINATION_GRACE_SECONDS,
                "schedule_seed": 2701,
                "measured_orders": orders,
                "schedule_balance": "paired",
                "keep_artifacts": False,
                "artifact_root": None,
                "checkpoint_path": str(self.raw_path),
                "cache_policy": (
                    "best-effort; warmups precede measurements; raw uses "
                    "--reflink=never and --sparse=never"
                ),
                "io_policy": "regular files; decoders use --no-sparse and zstd uses --no-asyncio",
                "timing_scope": (
                    "after stdout/stderr log open, immediately before process spawn, through process reap"
                ),
                "verification_scope": (
                    "complete post-timing byte scan on every successful decompression"
                ),
                "scheduling_policy": "serial execution; paired forward/reverse order",
                "short_process_timing_limitations": "fixture",
                "execution_schedule": execution_schedule,
            },
            "provenance": {
                "git": {"commit": self.implementation["git_commit"], "dirty": False},
                "benchmark_script": {
                    "path": "/fixture/benchmarking.py",
                    "sha256": self.implementation["generic_benchmarking_sha256"],
                },
            },
            "environment": {
                "system": "Linux", "release": "fixture", "machine": "x86_64",
                "processor": "fixture", "hostname": "fixture", "python": "3.12",
                "clock": "time.perf_counter_ns", "rss_source": "direct child",
                "cpu": {
                    "model": "fixture CPU",
                    "logical_cores_host": 8,
                    "physical_cores_host": 4,
                    "affinity_logical_cores": 8,
                    "quota": {
                        "source": "/sys/fs/cgroup/cpu.max",
                        "quota_microseconds": 800000,
                        "period_microseconds": 100000,
                        "quota_cores": 8.0,
                    },
                },
                "memory": {
                    "total_bytes": 16_000_000_000,
                    "available_bytes_at_start": 8_000_000_000,
                    "cgroup_limit_bytes": 12_000_000_000,
                    "cgroup_limit_source": "/sys/fs/cgroup/memory.max",
                },
                "filesystems": {
                    "source": {"free_bytes": 1}, "work": {"free_bytes": 1},
                    "checkpoint": {"free_bytes": 1},
                },
                "codec_environment_variables": {
                    name: None for name in benchmarking.CODEC_ENVIRONMENT_VARIABLES
                },
            },
            "methods": methods,
        }

    def _outcomes(self) -> list[dict]:
        return [{
            "id": identifier,
            "spec_matches": True,
            "available": True,
            "status_ok": True,
            "tool_provenance": True,
            "executable_unchanged": True,
            "warmup_count": 1,
            "measured_count": 6,
            "all_iterations_bit_exact": True,
            "measured_archive_consistent": True,
            "valid": True,
        } for identifier in self.identifiers]

    def _receipt(self) -> dict:
        outcomes = self._outcomes()
        scalar_checks = {
            key: True for key in (
                "top_schema", "top_status", "task_and_input_metadata", "configuration",
                "balanced_measured_orders", "execution_schedule", "method_order",
                "source_provenance", "integrity", "result_git_provenance",
                "result_script_provenance", "environment_provenance",
                "codec_environment_unset", "source_unchanged_after_run",
                "manifest_unchanged_after_run", "matrix_unchanged_after_run",
                "clean_bound_git_after_run", "scripts_unchanged_after_run",
            )
        }
        return {
            "schema": summary.RECEIPT_SCHEMA,
            "status": "complete",
            "task_semantic_sha256": self.task_sha,
            "task_semantics": self.task,
            "raw_result": {},
            "receipt_path": str(self.receipt_path),
            "attempt_contract_valid": True,
            "result_success": True,
            "formal_eligible": True,
            "postvalidation": {
                "status_code": "passed",
                "checks": {
                    **scalar_checks,
                    "methods": outcomes,
                    "disk_gate": self.preflight_gate,
                    "post_run_disk_gate": self.post_gate,
                    "post_run_codec_environment": {
                        name: {"present": False, "value": None}
                        for name in benchmarking.CODEC_ENVIRONMENT_VARIABLES
                    },
                },
                "failures": [],
            },
            "method_outcomes": outcomes,
            "disk_gate": self.post_gate,
            "implementation_binding": self.implementation,
            "generic_registry": {
                "id": summary.REGISTRY_ID,
                "count": 19,
                "sha256": summary._canonical_sha256(self.registry),
            },
            "exclusive_lock_path": "/tmp/brevis-machine-benchmark.lock",
            "stage_override_reasons": [],
            "benchmark_exception": None,
        }

    def publish(self) -> None:
        self.raw_path.write_text(json.dumps(self.raw, sort_keys=True) + "\n", encoding="utf-8")
        raw_payload = self.raw_path.read_bytes()
        self.receipt["raw_result"] = {
            "path": str(self.raw_path),
            "present": True,
            "size_bytes": len(raw_payload),
            "sha256": hashlib.sha256(raw_payload).hexdigest(),
            "schema": summary.INPUT_SCHEMA,
            "status": "complete",
            "preserved_without_selective_omission": True,
        }
        self.receipt_path.write_text(
            json.dumps(self.receipt, sort_keys=True) + "\n", encoding="utf-8",
        )

    def mutate_raw(self, mutator) -> None:
        mutator(self.raw)
        self.publish()


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def rebind_task(fixture: PairFixture) -> None:
        fixture.task_sha = summary._canonical_sha256(fixture.task)
        fixture.raw["input_metadata"]["campaign_task_semantic_sha256"] = fixture.task_sha
        fixture.receipt["task_semantic_sha256"] = fixture.task_sha
        fixture.receipt["task_semantics"] = fixture.task
        fixture.publish()

    def test_valid_pair_has_traceable_statistics_and_alias_metadata(self):
        fixture = PairFixture(self.root)
        result = summary.aggregate_pairs([(fixture.raw_path, fixture.receipt_path)])
        self.assertEqual(summary.OUTPUT_SCHEMA, result["schema"])
        self.assertEqual(19, result["scope"]["registry_row_count"])
        self.assertEqual(18, result["scope"]["independent_configuration_count"])
        self.assertEqual(64, len(result["aggregate_input_fingerprint_sha256"]))
        raw = result["inputs"][0]["methods"][0]
        self.assertEqual(6, raw["metrics"]["archive_bytes"]["n"])
        self.assertEqual(float(fixture.source_bytes), raw["metrics"]["archive_bytes"]["mean"])
        self.assertEqual(1.0, raw["metrics"]["compression_ratio_raw_over_archive"]["mean"])
        self.assertEqual(0.0, raw["metrics"]["saving_fraction"]["mean"])
        self.assertAlmostEqual(
            3.5, raw["metrics"]["compression_wall_time_seconds"]["mean"], places=12,
        )
        self.assertAlmostEqual(
            3.5 ** 0.5,
            raw["metrics"]["compression_wall_time_seconds"]["sample_stddev"],
            places=12,
        )
        bzip_ratio = next(row for row in result["registry"]["rows"] if row["id"] == "bzip2/ratio")
        self.assertEqual("bzip2/default", bzip_ratio["alias_of"])
        self.assertEqual("bzip2/level-9", bzip_ratio["independent_observation_id"])

    def test_model_equal_and_raw_byte_weighted_views_are_distinct(self):
        first_root = self.root / "first"
        second_root = self.root / "second"
        first_root.mkdir()
        second_root.mkdir()
        first = PairFixture(first_root, tag="model-a", source_bytes=100_000, timing_bias=0)
        second = PairFixture(second_root, tag="model-b", source_bytes=300_000, timing_bias=10)
        result = summary.aggregate_pairs([
            (second.raw_path, second.receipt_path),
            (first.raw_path, first.receipt_path),
        ])
        raw = next(row for row in result["cross_model"]["methods"] if row["id"] == "raw/copy")
        model_equal = raw["views"]["model_equal"]["performance"]["point_estimates"]
        global_serial = raw["views"]["raw_byte_weighted"]["performance"]
        expected_model_equal_throughput = ((100_000 / 3.5) + (300_000 / 13.5)) / 2
        self.assertAlmostEqual(
            expected_model_equal_throughput,
            model_equal["compression_throughput_raw_bytes_per_second"],
        )
        self.assertAlmostEqual(
            400_000 / 17.0,
            global_serial["compression"]["throughput_raw_bytes_per_second"],
        )
        self.assertEqual(400_000, raw["pooled_storage"]["raw_bytes"])
        self.assertEqual(400_000, raw["pooled_storage"]["archive_bytes"])
        self.assertEqual(1.0, raw["pooled_storage"]["compression_ratio_raw_over_archive"])

    def test_model_equal_storage_pools_shards_before_averaging_models(self):
        fixture_specs = (
            ("a1", "model-a", "shard-1.safetensors", 10_000, 0.50, 0),
            ("a2", "model-a", "shard-2.safetensors", 90_000, 0.90, 10),
            ("b1", "model-b", "shard-1.safetensors", 20_000, 0.25, 20),
            ("b2", "model-b", "shard-2.safetensors", 80_000, 1.00, 30),
        )
        fixtures = []
        for directory, tag, shard, raw_bytes, archive_fraction, timing_bias in fixture_specs:
            root = self.root / directory
            root.mkdir()
            fixtures.append(PairFixture(
                root, tag=tag, shard=shard, source_bytes=raw_bytes,
                archive_fraction=archive_fraction, timing_bias=timing_bias,
            ))
        result = summary.aggregate_pairs([
            (fixture.raw_path, fixture.receipt_path) for fixture in fixtures
        ])
        gzip = next(
            row for row in result["cross_model"]["methods"] if row["id"] == "gzip/speed"
        )
        storage = gzip["views"]["model_equal"]["storage"]
        estimates = storage["point_estimates"]
        expected_ratio = ((100_000 / 86_000) + (100_000 / 85_000)) / 2
        expected_saving = (0.14 + 0.15) / 2
        self.assertAlmostEqual(expected_ratio, estimates["compression_ratio_raw_over_archive"])
        self.assertAlmostEqual(expected_saving, estimates["saving_fraction"])
        self.assertNotAlmostEqual(1.4, estimates["compression_ratio_raw_over_archive"])
        self.assertEqual(2, len(storage["model_points"]))
        pooled = gzip["views"]["raw_byte_weighted"]["storage"]["pooled_storage"]
        self.assertEqual(200_000, pooled["raw_bytes"])
        self.assertEqual(171_000, pooled["archive_bytes"])
        self.assertAlmostEqual(200_000 / 171_000, pooled["compression_ratio_raw_over_archive"])

        model_performance = gzip["views"]["model_equal"]["performance"]
        performance_estimates = model_performance["point_estimates"]
        self.assertAlmostEqual(
            ((100_000 / 19.0) + (100_000 / 59.0)) / 2,
            performance_estimates["compression_throughput_raw_bytes_per_second"],
        )
        self.assertAlmostEqual(
            ((100_000 / 9.5) + (100_000 / 29.5)) / 2,
            performance_estimates["decompression_throughput_raw_bytes_per_second"],
        )
        self.assertEqual(27 * 1024, performance_estimates["compression_peak_rss_bytes"])
        self.assertEqual(27 * 2048, performance_estimates["decompression_peak_rss_bytes"])
        global_performance = gzip["views"]["raw_byte_weighted"]["performance"]
        self.assertAlmostEqual(
            200_000 / 78.0,
            global_performance["compression"]["throughput_raw_bytes_per_second"],
        )
        self.assertAlmostEqual(
            200_000 / 39.0,
            global_performance["decompression"]["throughput_raw_bytes_per_second"],
        )
        self.assertEqual(37 * 1024, global_performance["compression"]["peak_rss_bytes"])
        self.assertEqual(37 * 2048, global_performance["decompression"]["peak_rss_bytes"])
        model_a = next(row for row in result["per_model"] if row["model"]["tag"] == "model-a")
        model_a_gzip = next(row for row in model_a["methods"] if row["id"] == "gzip/speed")
        serial = model_a_gzip["serial_complete_model_performance"]
        self.assertEqual(19.0, serial["compression"]["expected_wall_seconds"])
        self.assertIsNone(serial["dispersion"]["combined_sample_stddev"])
        self.assertIn("not complete-model", model_a_gzip["descriptive_shard_mean_views"]["warning"])

    def test_receipt_raw_bytes_sha_and_path_are_required(self):
        fixture = PairFixture(self.root)
        receipt = json.loads(fixture.receipt_path.read_text())
        receipt["raw_result"]["sha256"] = "0" * 64
        fixture.receipt_path.write_text(json.dumps(receipt) + "\n")
        with self.assertRaisesRegex(summary.SummaryError, "exact raw bytes"):
            summary.aggregate_pairs([(fixture.raw_path, fixture.receipt_path)])

    def test_iteration_tamper_is_detected_even_with_refreshed_receipt_hash(self):
        fixture = PairFixture(self.root)

        def mutate(raw):
            raw["methods"][0]["runs"][2]["bit_exact"] = False

        fixture.mutate_raw(mutate)
        with self.assertRaisesRegex(summary.SummaryError, "successful measured"):
            summary.aggregate_pairs([(fixture.raw_path, fixture.receipt_path)])

    def test_task_hash_and_configuration_drift_fail_closed(self):
        fixture = PairFixture(self.root)
        receipt = json.loads(fixture.receipt_path.read_text())
        receipt["task_semantics"]["campaign_stage"] = 2
        fixture.receipt_path.write_text(json.dumps(receipt) + "\n")
        with self.assertRaisesRegex(summary.SummaryError, "do not hash"):
            summary.aggregate_pairs([(fixture.raw_path, fixture.receipt_path)])

        other_root = self.root / "config"
        other_root.mkdir()
        other = PairFixture(other_root, tag="model-config")
        other.mutate_raw(lambda raw: raw["configuration"].__setitem__("schedule_seed", 7))
        with self.assertRaisesRegex(summary.SummaryError, "configuration drifted"):
            summary.aggregate_pairs([(other.raw_path, other.receipt_path)])

    def test_seed_bound_schedule_rejects_another_balanced_pairing(self):
        fixture = PairFixture(self.root)

        def mutate(raw):
            identifiers = fixture.identifiers
            raw["configuration"]["measured_orders"] = [{
                "repetition": repetition,
                "pair": repetition // 2,
                "direction": "forward" if repetition % 2 == 0 else "reverse",
                "methods": identifiers if repetition % 2 == 0 else identifiers[::-1],
            } for repetition in range(6)]

        fixture.mutate_raw(mutate)
        with self.assertRaisesRegex(summary.SummaryError, "exact seed-2701"):
            summary.aggregate_pairs([(fixture.raw_path, fixture.receipt_path)])

    def test_formal_gate_lock_and_disk_paths_are_bound(self):
        gate_root = self.root / "gate"
        gate_root.mkdir()
        gate_fixture = PairFixture(gate_root, tag="model-gate")
        gate_fixture.task["executor"]["formal_gates"]["require_clean_git"] = False
        self.rebind_task(gate_fixture)
        with self.assertRaisesRegex(summary.SummaryError, "formal gates drifted"):
            summary.aggregate_pairs([(gate_fixture.raw_path, gate_fixture.receipt_path)])

        lock_root = self.root / "lock"
        lock_root.mkdir()
        lock_fixture = PairFixture(lock_root, tag="model-lock")
        lock_fixture.receipt["exclusive_lock_path"] = "/tmp/not-the-campaign-lock"
        lock_fixture.receipt_path.write_text(json.dumps(lock_fixture.receipt) + "\n")
        with self.assertRaisesRegex(summary.SummaryError, "lock path disagrees"):
            summary.aggregate_pairs([(lock_fixture.raw_path, lock_fixture.receipt_path)])

        disk_root = self.root / "disk"
        disk_root.mkdir()
        disk_fixture = PairFixture(disk_root, tag="model-disk")
        disk_fixture.preflight_gate["path"] = "/tmp/unbound-work-directory"
        disk_fixture.post_gate["path"] = "/tmp/unbound-work-directory"
        disk_fixture.publish()
        with self.assertRaisesRegex(summary.SummaryError, "disk_gate.path"):
            summary.aggregate_pairs([(disk_fixture.raw_path, disk_fixture.receipt_path)])

    def test_missing_method_and_source_identity_tamper_fail_closed(self):
        fixture = PairFixture(self.root)
        fixture.mutate_raw(lambda raw: raw["methods"].pop())
        with self.assertRaisesRegex(summary.SummaryError, "exactly 19"):
            summary.aggregate_pairs([(fixture.raw_path, fixture.receipt_path)])

        other_root = self.root / "source"
        other_root.mkdir()
        other = PairFixture(other_root, tag="model-source")
        other.mutate_raw(lambda raw: raw["source"].__setitem__("sha256", "0" * 64))
        with self.assertRaisesRegex(summary.SummaryError, "raw source disagrees"):
            summary.aggregate_pairs([(other.raw_path, other.receipt_path)])

    def test_alias_archive_drift_is_not_counted_as_independent_result(self):
        fixture = PairFixture(self.root)

        def mutate(raw):
            method = next(row for row in raw["methods"] if row["id"] == "bzip2/ratio")
            new_size = method["runs"][0]["compressed_size_bytes"] - 1
            new_sha = _sha("tampered equivalent spelling")
            for run in method["runs"]:
                run["compressed_size_bytes"] = new_size
                run["archive_storage"]["logical_size_bytes"] = new_size
                run["archive_sha256"] = new_sha
                run["archive_consistency"]["reference_size_bytes"] = new_size
                run["archive_consistency"]["reference_sha256"] = new_sha
                run["compression"]["output_size_bytes"] = new_size
                run["compression"]["output_storage"]["logical_size_bytes"] = new_size
            method["measured_archive_consistency"]["reference_size_bytes"] = new_size
            method["measured_archive_consistency"]["reference_sha256"] = new_sha

        fixture.mutate_raw(mutate)
        with self.assertRaisesRegex(summary.SummaryError, "did not produce identical"):
            summary.aggregate_pairs([(fixture.raw_path, fixture.receipt_path)])

    def test_post_run_gate_and_codec_environment_are_enforced(self):
        fixture = PairFixture(self.root)
        receipt = json.loads(fixture.receipt_path.read_text())
        receipt["disk_gate"]["post_run_passes"] = False
        fixture.receipt_path.write_text(json.dumps(receipt) + "\n")
        with self.assertRaisesRegex(summary.SummaryError, "post-run disk gate"):
            summary.aggregate_pairs([(fixture.raw_path, fixture.receipt_path)])

        other_root = self.root / "environment"
        other_root.mkdir()
        other = PairFixture(other_root, tag="model-env")
        receipt = json.loads(other.receipt_path.read_text())
        receipt["postvalidation"]["checks"]["post_run_codec_environment"]["XZ_OPT"] = {
            "present": True, "value": "-9",
        }
        other.receipt_path.write_text(json.dumps(receipt) + "\n")
        with self.assertRaisesRegex(summary.SummaryError, "contains XZ_OPT"):
            summary.aggregate_pairs([(other.raw_path, other.receipt_path)])

    def test_duplicate_observation_and_identity_conflict_are_rejected(self):
        first_root = self.root / "a"
        second_root = self.root / "b"
        first_root.mkdir()
        second_root.mkdir()
        first = PairFixture(first_root, tag="same")
        second = PairFixture(second_root, tag="same")
        # Paths differ, but the model revision, shard, and source identity are identical.
        with self.assertRaisesRegex(summary.SummaryError, "source was supplied more than once"):
            summary.aggregate_pairs([
                (first.raw_path, first.receipt_path),
                (second.raw_path, second.receipt_path),
            ])

        conflict_a_root = self.root / "conflict-a"
        conflict_b_root = self.root / "conflict-b"
        conflict_a_root.mkdir()
        conflict_b_root.mkdir()
        conflict_a = PairFixture(conflict_a_root, tag="conflict", source_bytes=100_000)
        conflict_b = PairFixture(conflict_b_root, tag="conflict", source_bytes=120_000)
        with self.assertRaisesRegex(summary.SummaryError, "conflicting SHA-256"):
            summary.aggregate_pairs([
                (conflict_a.raw_path, conflict_a.receipt_path),
                (conflict_b.raw_path, conflict_b.receipt_path),
            ])

    def test_heterogeneous_host_and_toolchain_inputs_are_rejected(self):
        first_root = self.root / "homogeneous-a"
        second_root = self.root / "heterogeneous-host"
        first_root.mkdir()
        second_root.mkdir()
        first = PairFixture(first_root, tag="model-homogeneous-a")
        second = PairFixture(second_root, tag="model-heterogeneous-host")
        second.mutate_raw(lambda raw: raw["environment"].__setitem__("hostname", "other-host"))
        with self.assertRaisesRegex(summary.SummaryError, "environment_fingerprint"):
            summary.aggregate_pairs([
                (first.raw_path, first.receipt_path),
                (second.raw_path, second.receipt_path),
            ])

        tool_root = self.root / "heterogeneous-tool"
        tool_root.mkdir()
        tool = PairFixture(tool_root, tag="model-heterogeneous-tool")
        tool.mutate_raw(lambda raw: raw["methods"][0].__setitem__("version", "cp other 2.0"))
        with self.assertRaisesRegex(summary.SummaryError, "toolchain_fingerprint"):
            summary.aggregate_pairs([
                (first.raw_path, first.receipt_path),
                (tool.raw_path, tool.receipt_path),
            ])

    def test_atomic_no_overwrite_survives_concurrent_writers(self):
        destination = self.root / "aggregate.json"
        barrier = threading.Barrier(2)

        def publish(marker: bytes):
            barrier.wait()
            try:
                summary._atomic_write(destination, marker)
            except summary.SummaryError as exc:
                return exc.code
            return "ok"

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(publish, (b"first\n", b"second\n")))
        self.assertEqual(["ok", "output_exists"], sorted(outcomes))
        self.assertIn(destination.read_bytes(), {b"first\n", b"second\n"})
        with self.assertRaisesRegex(summary.SummaryError, "refusing to overwrite"):
            summary._atomic_write(destination, b"third\n")

    def test_duplicate_json_keys_are_rejected(self):
        fixture = PairFixture(self.root)
        payload = fixture.raw_path.read_text()
        fixture.raw_path.write_text('{"schema":null,"schema":null}\n')
        receipt = json.loads(fixture.receipt_path.read_text())
        raw_bytes = fixture.raw_path.read_bytes()
        receipt["raw_result"]["size_bytes"] = len(raw_bytes)
        receipt["raw_result"]["sha256"] = hashlib.sha256(raw_bytes).hexdigest()
        fixture.receipt_path.write_text(json.dumps(receipt) + "\n")
        with self.assertRaisesRegex(summary.SummaryError, "duplicate JSON key"):
            summary.aggregate_pairs([(fixture.raw_path, fixture.receipt_path)])
        self.assertTrue(payload)


if __name__ == "__main__":
    unittest.main()
