import concurrent.futures
import json
import os
import pathlib
import shutil
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import benchmarking


class RegistryTests(unittest.TestCase):
    def test_registry_has_raw_standard_profiles_and_unique_identifiers(self):
        self.assertEqual(20, len(benchmarking.BASELINE_SPECS))
        identifiers = [spec.identifier for spec in benchmarking.BASELINE_SPECS]
        self.assertEqual(len(identifiers), len(set(identifiers)))
        self.assertEqual({"raw/copy"}, {
            spec.identifier for spec in benchmarking.BASELINE_SPECS if spec.method == "raw"
        })
        for method in ("gzip", "bzip2", "xz", "zstd", "lz4", "brotli"):
            with self.subTest(method=method):
                specs = [spec for spec in benchmarking.BASELINE_SPECS if spec.method == method]
                expected_profiles = set(benchmarking.PROFILES)
                if method == "zstd":
                    expected_profiles.add("ultra")
                self.assertEqual(expected_profiles, {spec.profile for spec in specs})
                for spec in specs:
                    self.assertTrue(spec.version_args)
                    self.assertTrue(spec.compress_args)
                    self.assertTrue(spec.decompress_args)
                    self.assertIsNotNone(spec.thread_policy.requested_compression_threads)
                    self.assertIn("oriented", spec.notes.lower())
                    json.dumps(spec.to_dict())

    def test_all_20_registry_argv_are_fully_pinned(self):
        expected = {
            "raw/copy": (
                ("-c", benchmarking.RAW_COPY_PROGRAM, "{input}", "{output}"),
                ("-c", benchmarking.RAW_COPY_PROGRAM, "{input}", "{output}"),
            ),
            "gzip/speed": (
                ("-n", "-k", "-f", "-1", "{input}"),
                ("-d", "-k", "-f", "{input}"),
            ),
            "gzip/default": (
                ("-n", "-k", "-f", "-6", "{input}"),
                ("-d", "-k", "-f", "{input}"),
            ),
            "gzip/ratio": (
                ("-n", "-k", "-f", "-9", "{input}"),
                ("-d", "-k", "-f", "{input}"),
            ),
            "bzip2/speed": (
                ("-k", "-f", "-1", "{input}"),
                ("-d", "-k", "-f", "{input}"),
            ),
            "bzip2/default": (
                ("-k", "-f", "-9", "{input}"),
                ("-d", "-k", "-f", "{input}"),
            ),
            "bzip2/ratio": (
                ("-k", "-f", "--best", "{input}"),
                ("-d", "-k", "-f", "{input}"),
            ),
            "xz/speed": (
                ("-k", "-f", "-T1", "-1", "{input}"),
                ("-d", "-k", "-f", "--no-sparse", "{input}"),
            ),
            "xz/default": (
                ("-k", "-f", "-T1", "-6", "{input}"),
                ("-d", "-k", "-f", "--no-sparse", "{input}"),
            ),
            "xz/ratio": (
                ("-k", "-f", "-T1", "-9e", "{input}"),
                ("-d", "-k", "-f", "--no-sparse", "{input}"),
            ),
            "zstd/speed": (
                (
                    "-q", "-f", "--single-thread", "--no-asyncio", "-1",
                    "{input}", "-o", "{output}",
                ),
                (
                    "-d", "-q", "-f", "--no-asyncio", "--no-sparse",
                    "{input}", "-o", "{output}",
                ),
            ),
            "zstd/default": (
                (
                    "-q", "-f", "--single-thread", "--no-asyncio", "-3",
                    "{input}", "-o", "{output}",
                ),
                (
                    "-d", "-q", "-f", "--no-asyncio", "--no-sparse",
                    "{input}", "-o", "{output}",
                ),
            ),
            "zstd/ratio": (
                (
                    "-q", "-f", "--single-thread", "--no-asyncio", "-19",
                    "{input}", "-o", "{output}",
                ),
                (
                    "-d", "-q", "-f", "--no-asyncio", "--no-sparse",
                    "{input}", "-o", "{output}",
                ),
            ),
            "zstd/ultra": (
                (
                    "-q", "-f", "--single-thread", "--no-asyncio", "--ultra", "-22",
                    "{input}", "-o", "{output}",
                ),
                (
                    "-d", "-q", "-f", "--no-asyncio", "--no-sparse",
                    "{input}", "-o", "{output}",
                ),
            ),
            "lz4/speed": (
                ("-q", "-f", "--fast=5", "{input}", "{output}"),
                ("-d", "-q", "-f", "--no-sparse", "{input}", "{output}"),
            ),
            "lz4/default": (
                ("-q", "-f", "-1", "{input}", "{output}"),
                ("-d", "-q", "-f", "--no-sparse", "{input}", "{output}"),
            ),
            "lz4/ratio": (
                ("-q", "-f", "-12", "{input}", "{output}"),
                ("-d", "-q", "-f", "--no-sparse", "{input}", "{output}"),
            ),
            "brotli/speed": (
                ("-f", "-q", "1", "-o", "{output}", "{input}"),
                ("-d", "-f", "-o", "{output}", "{input}"),
            ),
            "brotli/default": (
                ("-f", "-q", "11", "-o", "{output}", "{input}"),
                ("-d", "-f", "-o", "{output}", "{input}"),
            ),
            "brotli/ratio": (
                ("-f", "-q", "11", "-w", "24", "-o", "{output}", "{input}"),
                ("-d", "-f", "-o", "{output}", "{input}"),
            ),
        }
        actual = {
            spec.identifier: (spec.compress_args, spec.decompress_args)
            for spec in benchmarking.BASELINE_SPECS
        }
        self.assertEqual(expected, actual)

    def test_registry_documents_equivalent_and_serialized_profiles(self):
        raw = benchmarking.SPEC_BY_ID["raw/copy"]
        for arguments in (raw.compress_args, raw.decompress_args):
            self.assertEqual("-c", arguments[0])
            self.assertEqual(benchmarking.RAW_COPY_PROGRAM, arguments[1])
        self.assertIn("regular-file", raw.notes)
        self.assertIn("1 MiB", raw.notes)

        bzip_default = benchmarking.SPEC_BY_ID["bzip2/default"]
        bzip_ratio = benchmarking.SPEC_BY_ID["bzip2/ratio"]
        self.assertIn("defaults to its ratio-oriented", bzip_default.notes)
        self.assertIn("equivalent to -9", bzip_ratio.notes)

        zstd_default = benchmarking.SPEC_BY_ID["zstd/default"]
        self.assertIn("default compression level 3", zstd_default.notes)
        self.assertIn("serial, synchronous-I/O policy", zstd_default.notes)
        for spec in (
            benchmarking.SPEC_BY_ID["zstd/speed"],
            zstd_default,
            benchmarking.SPEC_BY_ID["zstd/ratio"],
            benchmarking.SPEC_BY_ID["zstd/ultra"],
        ):
            self.assertIn("--single-thread", spec.compress_args)
            self.assertIn("--no-asyncio", spec.compress_args)
            self.assertIn("--no-asyncio", spec.decompress_args)
            self.assertIn("--no-sparse", spec.decompress_args)
            self.assertEqual(
                ("--single-thread", "--no-asyncio"),
                spec.thread_policy.cli_args,
            )
        zstd_ultra = benchmarking.SPEC_BY_ID["zstd/ultra"]
        self.assertIn("ceiling-oriented", zstd_ultra.notes.lower())
        self.assertIn("ultra level 22", zstd_ultra.notes)

    def test_raw_copy_uses_a_portable_versioned_streaming_backend(self):
        raw = benchmarking.SPEC_BY_ID["raw/copy"]
        self.assertEqual(
            pathlib.Path(sys.executable).resolve(),
            pathlib.Path(raw.executable).resolve(),
        )
        self.assertEqual(("--version",), raw.version_args)
        for arguments in (raw.compress_args, raw.decompress_args):
            self.assertEqual("-c", arguments[0])
            self.assertNotIn("--reflink=never", arguments)
            self.assertNotIn("--sparse=never", arguments)
            self.assertEqual(("{input}", "{output}"), arguments[-2:])

    def test_select_specs_rejects_unknown_and_empty_selections(self):
        with self.assertRaisesRegex(ValueError, "unknown baseline"):
            benchmarking.select_specs(["missing/default"])
        with self.assertRaisesRegex(ValueError, "at least one"):
            benchmarking.select_specs([])


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.source = self.root / "weights.bin"
        # Repeated floating-point-like byte fields plus all byte values exercise
        # compressible and incompressible portions without a large fixture.
        self.source.write_bytes((b"\x00\x00\x80\x3f" * 4096) + bytes(range(256)) * 16)
        self.model_tag = "fixture"
        self.model_repo = "test/model"
        self.model_revision = "0" * 40
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps([{
            "tag": self.model_tag,
            "repo": self.model_repo,
            "revision": self.model_revision,
            "files": [{
                "file": self.source.name,
                "bytes": self.source.stat().st_size,
                "sha256": benchmarking._sha256_file(self.source),
            }],
        }]) + "\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def _python_copy_spec(self, method):
        copy_code = (
            "import pathlib,sys; pathlib.Path(sys.argv[2]).write_bytes("
            "pathlib.Path(sys.argv[1]).read_bytes())"
        )
        return benchmarking.BaselineSpec(
            method=method,
            profile="default",
            executable=sys.executable,
            version_args=("--version",),
            compress_args=("-c", copy_code, "{input}", "{output}"),
            decompress_args=("-c", copy_code, "{input}", "{output}"),
            thread_policy=benchmarking.SERIAL_CLI,
        )

    def test_durable_output_fsync_is_inside_wall_clock(self):
        output = self.root / "durable.bin"
        events = []
        clock_values = iter((100, 175))

        def clock():
            events.append("clock")
            return next(clock_values)

        def fsync_file(path):
            events.append(f"fsync:{path}")

        with (
            mock.patch.object(benchmarking.time, "perf_counter_ns", side_effect=clock),
            mock.patch.object(benchmarking, "_fsync_file", side_effect=fsync_file),
        ):
            record = benchmarking._run_process(
                [
                    sys.executable,
                    "-c",
                    "import pathlib,sys; pathlib.Path(sys.argv[1]).write_bytes(b'x')",
                    str(output),
                ],
                self.root,
                "durable-output",
                None,
                durable_output=output,
            )

        self.assertEqual(["clock", f"fsync:{output}", "clock"], events)
        self.assertEqual(75, record["wall_time_ns"])
        self.assertEqual("ok", record["status_code"])
        self.assertEqual(
            "file_fsync_after_successful_exit_before_wall_clock_stop",
            record["durability"]["policy"],
        )
        self.assertEqual("ok", record["durability"]["status_code"])
        self.assertTrue(record["durability"]["attempted"])
        self.assertTrue(record["durability"]["succeeded"])
        self.assertIsNone(record["durability"]["error"])
        self.assertTrue(record["durability"]["included_in_wall_time"])

    def test_missing_durable_output_is_a_structured_failure(self):
        missing = self.root / "missing.bin"
        record = benchmarking._run_process(
            [sys.executable, "-c", "raise SystemExit(0)"],
            self.root,
            "missing-durable-output",
            None,
            durable_output=missing,
        )

        self.assertEqual(0, record["exit_code"])
        self.assertEqual("durability_failed", record["status_code"])
        self.assertEqual("failed", record["durability"]["status_code"])
        self.assertTrue(record["durability"]["attempted"])
        self.assertFalse(record["durability"]["succeeded"])
        self.assertIn("FileNotFoundError", record["durability"]["error"])

    def test_nonzero_process_does_not_attempt_output_durability(self):
        missing = self.root / "not-written.bin"
        with mock.patch.object(benchmarking, "_fsync_file") as fsync_file:
            record = benchmarking._run_process(
                [sys.executable, "-c", "raise SystemExit(7)"],
                self.root,
                "failed-before-durability",
                None,
                durable_output=missing,
            )

        fsync_file.assert_not_called()
        self.assertEqual("nonzero_exit", record["status_code"])
        self.assertEqual(
            "not_attempted_process_failed",
            record["durability"]["status_code"],
        )
        self.assertFalse(record["durability"]["attempted"])
        self.assertIsNone(record["durability"]["error"])

    def test_durability_error_fails_the_benchmark_iteration(self):
        copy_spec = self._python_copy_spec("durability-fixture")
        with mock.patch.object(
            benchmarking,
            "_fsync_file",
            side_effect=OSError("fixture durability failure"),
        ):
            result = benchmarking.benchmark_file(
                self.source,
                warmups=0,
                repetitions=1,
                specs=(copy_spec,),
                work_dir=self.root / "work",
            )

        run = result["methods"][0]["runs"][0]
        self.assertFalse(run["success"])
        self.assertEqual("compression_durability_failed", run["status_code"])
        self.assertIn("fixture durability failure", run["failure"])
        self.assertEqual("failed", run["compression"]["durability"]["status_code"])
        self.assertIsNone(run["decompression"])

    def test_generic_iteration_fsyncs_archive_and_restored_output(self):
        real_fsync = benchmarking._fsync_file
        with mock.patch.object(
            benchmarking,
            "_fsync_file",
            wraps=real_fsync,
        ) as fsync_file:
            result = benchmarking.benchmark_file(
                self.source,
                warmups=0,
                repetitions=1,
                specs=(self._python_copy_spec("durable-copy-fixture"),),
                work_dir=self.root / "work",
                keep_artifacts=True,
            )

        try:
            run = result["methods"][0]["runs"][0]
            run_dir = pathlib.Path(run["artifact_directory"])
            self.assertTrue(run["success"], run["failure"])
            self.assertEqual("ok", run["compression"]["durability"]["status_code"])
            self.assertEqual("ok", run["decompression"]["durability"]["status_code"])
            self.assertEqual(
                [
                    run_dir / "compress" / "archive.bin",
                    run_dir / "decompress" / "restored.bin",
                ],
                [call.args[0] for call in fsync_file.call_args_list],
            )
            self.assertIn("file fsync", result["configuration"]["io_policy"])
            self.assertIn("output file fsync", result["configuration"]["timing_scope"])
        finally:
            shutil.rmtree(result["configuration"]["artifact_root"])

    def test_restored_output_durability_error_fails_the_iteration(self):
        real_fsync = benchmarking._fsync_file

        def fail_restored(path):
            if path.name == "restored.bin":
                raise OSError("fixture restored durability failure")
            real_fsync(path)

        with mock.patch.object(
            benchmarking,
            "_fsync_file",
            side_effect=fail_restored,
        ):
            result = benchmarking.benchmark_file(
                self.source,
                warmups=0,
                repetitions=1,
                specs=(self._python_copy_spec("restored-durability-fixture"),),
                work_dir=self.root / "work",
            )

        run = result["methods"][0]["runs"][0]
        self.assertFalse(run["success"])
        self.assertEqual("decompression_durability_failed", run["status_code"])
        self.assertEqual("ok", run["compression"]["durability"]["status_code"])
        self.assertEqual("failed", run["decompression"]["durability"]["status_code"])
        self.assertIn("fixture restored durability failure", run["failure"])
        self.assertIsNone(run["verification"])

    def test_default_process_call_does_not_request_output_durability(self):
        with mock.patch.object(benchmarking, "_fsync_file") as fsync_file:
            record = benchmarking._run_process(
                [sys.executable, "--version"],
                self.root,
                "ordinary-probe",
                None,
            )

        fsync_file.assert_not_called()
        self.assertEqual("ok", record["status_code"])
        self.assertEqual("none", record["durability"]["policy"])
        self.assertEqual("not_requested", record["durability"]["status_code"])
        self.assertFalse(record["durability"]["attempted"])
        self.assertFalse(record["durability"]["included_in_wall_time"])

    def test_raw_records_resources_hashes_storage_and_provenance(self):
        result = benchmarking.benchmark_file(
            self.source,
            warmups=1,
            repetitions=2,
            specs=benchmarking.select_specs(["raw/copy"]),
            work_dir=self.root / "work",
            input_metadata={
                "model_tag": self.model_tag,
                "model_repo": self.model_repo,
                "model_revision": self.model_revision,
                "shard": self.source.name,
                "manifest": str(self.manifest),
            },
            evidence_policy={
                "run_class": "engineering",
                "paper_eligible": False,
            },
            expected_source_size_bytes=self.source.stat().st_size,
            expected_source_sha256=benchmarking._sha256_file(self.source),
            manifest_path=self.manifest,
            expected_manifest_sha256=benchmarking._sha256_file(self.manifest),
        )

        self.assertEqual(
            {"id": benchmarking.SCHEMA_ID, "version": benchmarking.SCHEMA_VERSION},
            result["schema"],
        )
        self.assertEqual("complete", result["status"])
        self.assertLessEqual(result["started_at_utc"], result["ended_at_utc"])
        self.assertEqual(self.source.stat().st_size, result["source"]["size_bytes"])
        self.assertEqual(64, len(result["source"]["sha256"]))
        self.assertEqual(self.model_tag, result["input_metadata"]["model_tag"])
        self.assertEqual("engineering", result["evidence_policy"]["run_class"])
        self.assertFalse(result["evidence_policy"]["paper_eligible"])
        self.assertTrue(result["integrity"]["source"]["verified"])
        self.assertTrue(result["integrity"]["manifest"]["verified"])
        self.assertTrue(result["integrity"]["manifest_binding"]["verified"])
        self.assertIn("immediately before process spawn", result["configuration"]["timing_scope"])
        self.assertIn("forward/reverse", result["configuration"]["scheduling_policy"])
        self.assertIn("explicit 1 MiB reads", result["configuration"]["cache_policy"])
        self.assertIn("Zstandard", result["configuration"]["io_policy"])
        self.assertIn("--no-sparse", result["configuration"]["io_policy"])
        self.assertIn("--no-asyncio", result["configuration"]["io_policy"])
        self.assertIn("file fsync", result["configuration"]["io_policy"])
        self.assertIn(
            "source and staging files are not fsynced",
            result["configuration"]["io_policy"],
        )
        self.assertIn("output file fsync", result["configuration"]["timing_scope"])
        self.assertEqual(3, len(result["configuration"]["execution_schedule"]))
        self.assertIn("commit", result["provenance"]["git"])
        self.assertIn("dirty", result["provenance"]["git"])
        self.assertEqual(64, len(result["provenance"]["benchmark_script"]["sha256"]))
        self.assertIn("logical_cores_host", result["environment"]["cpu"])
        self.assertIn("quota_cores", result["environment"]["cpu"]["quota"])
        self.assertGreater(result["environment"]["memory"]["total_bytes"], 0)
        self.assertGreater(result["environment"]["filesystems"]["source"]["free_bytes"], 0)
        self.assertGreater(result["environment"]["filesystems"]["work"]["free_bytes"], 0)
        self.assertIsNone(result["environment"]["filesystems"]["checkpoint"])
        self.assertEqual(
            set(benchmarking.CODEC_ENVIRONMENT_VARIABLES),
            set(result["environment"]["codec_environment_variables"]),
        )

        method = result["methods"][0]
        self.assertTrue(method["available"])
        self.assertEqual("ok", method["status_code"])
        self.assertTrue(pathlib.Path(method["executable_realpath"]).is_file())
        self.assertEqual(64, len(method["executable_sha256"]))
        self.assertEqual(method["executable_sha256"],
                         method["executable_sha256_after_runs"])
        self.assertTrue(method["executable_unchanged_during_benchmark"])
        self.assertIsNotNone(method["version"])
        self.assertEqual(1, len(method["warmups"]))
        self.assertEqual(2, len(method["runs"]))
        for run in (*method["warmups"], *method["runs"]):
            self.assertTrue(run["success"], run["failure"])
            self.assertEqual("ok", run["status_code"])
            self.assertLessEqual(run["started_at_utc"], run["ended_at_utc"])
            self.assertTrue(run["bit_exact"])
            self.assertEqual(self.source.stat().st_size, run["compressed_size_bytes"])
            self.assertEqual(64, len(run["archive_sha256"]))
            self.assertEqual(self.source.stat().st_size,
                             run["archive_storage"]["logical_size_bytes"])
            self.assertGreater(run["compression"]["wall_time_ns"], 0)
            self.assertGreater(run["decompression"]["wall_time_ns"], 0)
            self.assertGreater(run["compression"]["direct_child_max_rss_bytes"], 0)
            self.assertEqual("ok", run["compression"]["status_code"])
            self.assertLessEqual(
                run["compression"]["started_at_utc"], run["compression"]["ended_at_utc"],
            )
            self.assertIn(run["compression"]["wait_strategy"],
                          ("pidfd_selector_wait4", "wait4_wnohang_1ms_poll", "popen_wait"))
            self.assertEqual(0, run["compression"]["exit_code"])
            self.assertEqual(0, run["decompression"]["exit_code"])
            self.assertFalse(run["compression"]["timed_out"])
            self.assertEqual(self.source.stat().st_size,
                             run["compression"]["output_size_bytes"])
            self.assertEqual(self.source.stat().st_size,
                             run["decompression"]["output_size_bytes"])
            self.assertEqual(self.source.stat().st_size,
                             run["verification"]["restored_storage"]["logical_size_bytes"])
            self.assertGreater(
                run["verification"]["restored_storage"]["allocated_size_bytes"], 0,
            )
            self.assertEqual(result["source"]["sha256"],
                             run["verification"]["restored_sha256"])
            self.assertTrue(run["staging"]["excluded_from_timing"])
        self.assertIsNone(result["configuration"]["artifact_root"])
        json.dumps(result)

    def test_all_20_registry_configurations_round_trip_when_installed(self):
        result = benchmarking.benchmark_file(
            self.source,
            warmups=0,
            repetitions=2,
            specs=benchmarking.BASELINE_SPECS,
            work_dir=self.root / "work",
            timeout_seconds=30,
        )

        self.assertEqual(20, len(result["methods"]))
        available = 0
        methods_by_id = {method["id"]: method for method in result["methods"]}
        for method in result["methods"]:
            with self.subTest(method=method["id"]):
                if not method["available"]:
                    self.assertIn("not found", method["failure"])
                    self.assertFalse(method["runs"])
                    continue
                available += 1
                self.assertIsNone(method["failure"])
                self.assertEqual("consistent",
                                 method["measured_archive_consistency"]["status_code"])
                self.assertEqual(2, len(method["runs"]))
                for run in method["runs"]:
                    self.assertTrue(run["success"], run["failure"])
                    self.assertTrue(run["verification"]["bit_exact"])
                    self.assertEqual(64, len(run["archive_sha256"]))
                    self.assertEqual(0, run["compression"]["exit_code"])
                    self.assertEqual(0, run["decompression"]["exit_code"])
        self.assertGreaterEqual(available, 1)  # raw/copy is expected on supported hosts.
        bzip_default = methods_by_id["bzip2/default"]
        bzip_ratio = methods_by_id["bzip2/ratio"]
        if bzip_default["available"] and bzip_ratio["available"]:
            self.assertEqual(
                bzip_default["measured_archive_consistency"]["reference_size_bytes"],
                bzip_ratio["measured_archive_consistency"]["reference_size_bytes"],
            )
            self.assertEqual(
                bzip_default["measured_archive_consistency"]["reference_sha256"],
                bzip_ratio["measured_archive_consistency"]["reference_sha256"],
            )

    def test_zstd_ultra_round_trips_fixture_when_installed(self):
        result = benchmarking.benchmark_file(
            self.source,
            warmups=0,
            repetitions=1,
            specs=benchmarking.select_specs(["zstd/ultra"]),
            work_dir=self.root / "work",
            timeout_seconds=30,
        )

        method = result["methods"][0]
        if not method["available"]:
            self.assertIn("not found", method["failure"])
            return
        self.assertIsNone(method["failure"])
        self.assertEqual("zstd/ultra", method["id"])
        self.assertEqual(1, len(method["runs"]))
        run = method["runs"][0]
        self.assertTrue(run["success"], run["failure"])
        self.assertTrue(run["verification"]["bit_exact"])
        self.assertEqual(
            result["source"]["sha256"],
            run["verification"]["restored_sha256"],
        )

    def test_default_measured_schedule_is_seeded_and_forward_reverse_balanced(self):
        specs = tuple(
            benchmarking.BaselineSpec(
                method=f"copy-{index}",
                profile="default",
                executable=sys.executable,
                version_args=("--version",),
                compress_args=(
                    "-c", benchmarking.RAW_COPY_PROGRAM, "{input}", "{output}",
                ),
                decompress_args=(
                    "-c", benchmarking.RAW_COPY_PROGRAM, "{input}", "{output}",
                ),
                thread_policy=benchmarking.RAW_COPY,
            )
            for index in range(3)
        )
        result = benchmarking.benchmark_file(
            self.source, warmups=0, specs=specs,
            work_dir=self.root / "work",
        )
        self.assertEqual(6, result["configuration"]["repetitions"])
        self.assertEqual(benchmarking.DEFAULT_SCHEDULE_SEED,
                         result["configuration"]["schedule_seed"])
        self.assertEqual("paired", result["configuration"]["schedule_balance"])
        declared_orders = result["configuration"]["measured_orders"]
        self.assertEqual(6, len(declared_orders))
        for pair_start in range(0, 6, 2):
            self.assertEqual("forward", declared_orders[pair_start]["direction"])
            self.assertEqual("reverse", declared_orders[pair_start + 1]["direction"])
            self.assertEqual(
                declared_orders[pair_start]["methods"][::-1],
                declared_orders[pair_start + 1]["methods"],
            )

        schedule = result["configuration"]["execution_schedule"]
        for repetition, declared in enumerate(declared_orders):
            actual = [
                entry["method"] for entry in schedule
                if entry["phase"] == "measured" and entry["repetition"] == repetition
            ]
            self.assertEqual(declared["methods"], actual)
        self.assertEqual(list(range(18)), [entry["execution_order"] for entry in schedule])

    def test_missing_tool_and_subprocess_failure_are_serialized(self):
        missing = benchmarking.BaselineSpec(
            method="missing",
            profile="default",
            executable="brevis-tool-that-does-not-exist",
            version_args=("--version",),
            compress_args=("{input}", "{output}"),
            decompress_args=("{input}", "{output}"),
            thread_policy=benchmarking.SERIAL_CLI,
        )
        failing = benchmarking.BaselineSpec(
            method="failure-fixture",
            profile="default",
            executable=sys.executable,
            version_args=("--version",),
            compress_args=(
                "-c",
                "import sys; print('diagnostic'); print('intentional failure', file=sys.stderr); sys.exit(7)",
                "{input}",
                "{output}",
            ),
            decompress_args=("-c", "raise SystemExit(99)", "{input}", "{output}"),
            thread_policy=benchmarking.SERIAL_CLI,
        )
        result = benchmarking.benchmark_file(
            self.source,
            warmups=0,
            repetitions=1,
            specs=(missing, failing),
            work_dir=self.root / "work",
        )

        missing_result, failure_result = result["methods"]
        self.assertFalse(missing_result["available"])
        self.assertIn("not found", missing_result["failure"])
        run = failure_result["runs"][0]
        self.assertFalse(run["success"])
        self.assertEqual(7, run["compression"]["exit_code"])
        self.assertEqual("diagnostic\n", run["compression"]["stdout"]["text"])
        self.assertEqual("intentional failure\n", run["compression"]["stderr"]["text"])
        self.assertIsNone(run["decompression"])
        self.assertIn("status 7", run["failure"])
        json.dumps(result)

    def test_version_probe_exit_code_is_checked(self):
        bad_probe = benchmarking.BaselineSpec(
            method="bad-probe",
            profile="default",
            executable=sys.executable,
            version_args=("-c", "raise SystemExit(12)"),
            compress_args=("-c", "raise SystemExit(0)", "{input}", "{output}"),
            decompress_args=("-c", "raise SystemExit(0)", "{input}", "{output}"),
            thread_policy=benchmarking.SERIAL_CLI,
        )
        result = benchmarking.benchmark_file(
            self.source, warmups=0, repetitions=1, specs=(bad_probe,),
            work_dir=self.root / "work",
        )
        method = result["methods"][0]
        self.assertEqual(12, method["version_probe"]["exit_code"])
        self.assertIsNone(method["version"])
        self.assertIn("status 12", method["failure"])
        self.assertFalse(method["runs"])

    def test_bzip2_version_comes_from_stderr_not_binary_stdout(self):
        spec = benchmarking.SPEC_BY_ID["bzip2/default"]
        probe = {
            "stdout": benchmarking._captured_bytes(b"BZh9\x17rE8P\x90"),
            "stderr": benchmarking._captured_bytes(
                b"bzip2, a block-sorting file compressor.  Version 1.0.8, 13-Jul-2019.\n"
            ),
        }
        version = benchmarking._extract_tool_version(spec, probe)
        self.assertEqual(
            "bzip2, a block-sorting file compressor.  Version 1.0.8, 13-Jul-2019.",
            version,
        )

    def test_timeout_is_uniformly_recorded_as_failure(self):
        sleeper = benchmarking.BaselineSpec(
            method="sleeper",
            profile="default",
            executable=sys.executable,
            version_args=("--version",),
            compress_args=("-c", "import time; time.sleep(1)", "{input}", "{output}"),
            decompress_args=("-c", "import time; time.sleep(1)", "{input}", "{output}"),
            thread_policy=benchmarking.SERIAL_CLI,
        )
        result = benchmarking.benchmark_file(
            self.source, warmups=0, repetitions=1, specs=(sleeper,),
            work_dir=self.root / "work", timeout_seconds=0.1,
        )
        run = result["methods"][0]["runs"][0]
        self.assertTrue(run["compression"]["timed_out"])
        self.assertEqual(0.1, run["compression"]["timeout_seconds"])
        self.assertIn("timed out", run["failure"])
        self.assertFalse(run["success"])

    def test_staging_uses_independent_inodes_and_contains_destructive_codecs(self):
        source_before = self.source.read_bytes()
        destructive = benchmarking.BaselineSpec(
            method="destructive-fixture",
            profile="default",
            executable=sys.executable,
            version_args=("--version",),
            compress_args=(
                "-c",
                (
                    "import pathlib,sys; p=pathlib.Path(sys.argv[1]); d=p.read_bytes(); "
                    "p.write_bytes(b'corrupted staged source'); "
                    "pathlib.Path(sys.argv[2]).write_bytes(d)"
                ),
                "{input}", "{output}",
            ),
            decompress_args=(
                "-c",
                (
                    "import pathlib,sys; p=pathlib.Path(sys.argv[1]); d=p.read_bytes(); "
                    "p.write_bytes(b'corrupted staged archive'); "
                    "pathlib.Path(sys.argv[2]).write_bytes(d)"
                ),
                "{input}", "{output}",
            ),
            thread_policy=benchmarking.SERIAL_CLI,
        )
        result = benchmarking.benchmark_file(
            self.source, warmups=0, repetitions=2, specs=(destructive,),
            work_dir=self.root / "work", keep_artifacts=True,
        )
        try:
            self.assertEqual(source_before, self.source.read_bytes())
            for run in result["methods"][0]["runs"]:
                self.assertTrue(run["success"], run["failure"])
                self.assertEqual("independent_copy", run["staging"]["source"])
                self.assertEqual("independent_copy", run["staging"]["archive"])
                run_dir = pathlib.Path(run["artifact_directory"])
                staged_source = run_dir / "compress" / "input.bin"
                archive = run_dir / "compress" / "archive.bin"
                staged_archive = run_dir / "decompress" / "archive.bin"
                self.assertFalse(os.path.samefile(self.source, staged_source))
                self.assertFalse(os.path.samefile(archive, staged_archive))
                self.assertEqual(source_before, archive.read_bytes())
                self.assertEqual(b"corrupted staged archive", staged_archive.read_bytes())
        finally:
            shutil.rmtree(result["configuration"]["artifact_root"])

    def test_timeout_kills_descendant_process_group(self):
        child_started = self.root / "child-started"
        escaped = self.root / "escaped"
        child_code = (
            "import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text('started'); "
            "time.sleep(0.45); pathlib.Path(sys.argv[2]).write_text('escaped')"
        )
        parent_code = (
            "import subprocess,sys,time; "
            "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],sys.argv[3]]); "
            "time.sleep(5)"
        )
        tree = benchmarking.BaselineSpec(
            method="process-tree-fixture",
            profile="default",
            executable=sys.executable,
            version_args=("--version",),
            compress_args=(
                "-c", parent_code, child_code, str(child_started), str(escaped),
                "{input}", "{output}",
            ),
            decompress_args=("-c", "raise SystemExit(0)", "{input}", "{output}"),
            thread_policy=benchmarking.SERIAL_CLI,
        )
        result = benchmarking.benchmark_file(
            self.source, warmups=0, repetitions=1, specs=(tree,),
            work_dir=self.root / "work", timeout_seconds=0.2,
        )
        run = result["methods"][0]["runs"][0]
        self.assertEqual("compression_timed_out", run["status_code"])
        self.assertTrue(run["compression"]["process_group_isolated"])
        self.assertTrue(run["compression"]["started_new_session"])
        self.assertTrue(child_started.exists(), "fixture descendant did not start")
        time.sleep(0.5)
        self.assertFalse(escaped.exists(), "descendant survived the operation timeout")

    def test_keyboard_interrupt_kills_descendant_process_group_and_reaps(self):
        child_started = self.root / "interrupt-child-started"
        escaped = self.root / "interrupt-escaped"
        child_code = (
            "import pathlib,sys,time; pathlib.Path(sys.argv[1]).write_text('started'); "
            "time.sleep(0.45); pathlib.Path(sys.argv[2]).write_text('escaped')"
        )
        parent_code = (
            "import subprocess,sys,time; "
            "subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2],sys.argv[3]]); "
            "time.sleep(5)"
        )

        def interrupt_after_child_starts(process, timeout_seconds):
            del process, timeout_seconds
            time.sleep(0.2)
            raise KeyboardInterrupt

        with mock.patch.object(
            benchmarking, "_wait_direct_child", side_effect=interrupt_after_child_starts,
        ):
            with self.assertRaises(KeyboardInterrupt):
                benchmarking._run_process(
                    [sys.executable, "-c", parent_code, child_code,
                     str(child_started), str(escaped)],
                    self.root, "keyboard-interrupt", 10,
                )
        self.assertTrue(child_started.exists(), "fixture descendant did not start")
        time.sleep(0.5)
        self.assertFalse(escaped.exists(), "descendant survived KeyboardInterrupt cleanup")

    def test_model_metadata_requires_and_verifies_source_and_manifest_integrity(self):
        metadata = {
            "model_tag": self.model_tag,
            "model_repo": self.model_repo,
            "model_revision": self.model_revision,
            "shard": self.source.name,
            "manifest": str(self.manifest),
        }
        specs = benchmarking.select_specs(["raw/copy"])
        with self.assertRaisesRegex(ValueError, "requires expected source"):
            benchmarking.benchmark_file(
                self.source, warmups=0, repetitions=1, specs=specs,
                input_metadata=metadata, manifest_path=self.manifest,
                expected_manifest_sha256=benchmarking._sha256_file(self.manifest),
            )
        with self.assertRaisesRegex(ValueError, "source size mismatch"):
            benchmarking.benchmark_file(
                self.source, warmups=0, repetitions=1, specs=specs,
                input_metadata=metadata,
                expected_source_size_bytes=self.source.stat().st_size + 1,
                expected_source_sha256=benchmarking._sha256_file(self.source),
                manifest_path=self.manifest,
                expected_manifest_sha256=benchmarking._sha256_file(self.manifest),
            )
        with self.assertRaisesRegex(ValueError, "source SHA-256 mismatch"):
            benchmarking.benchmark_file(
                self.source, warmups=0, repetitions=1, specs=specs,
                input_metadata=metadata,
                expected_source_size_bytes=self.source.stat().st_size,
                expected_source_sha256="0" * 64,
                manifest_path=self.manifest,
                expected_manifest_sha256=benchmarking._sha256_file(self.manifest),
            )
        with self.assertRaisesRegex(ValueError, "manifest SHA-256 mismatch"):
            benchmarking.benchmark_file(
                self.source, warmups=0, repetitions=1, specs=specs,
                input_metadata=metadata,
                expected_source_size_bytes=self.source.stat().st_size,
                expected_source_sha256=benchmarking._sha256_file(self.source),
                manifest_path=self.manifest,
                expected_manifest_sha256="0" * 64,
            )

        result = benchmarking.benchmark_file(
            self.source, warmups=0, repetitions=1, specs=specs,
            work_dir=self.root / "integrity-work", input_metadata=metadata,
            expected_source_size_bytes=self.source.stat().st_size,
            expected_source_sha256=benchmarking._sha256_file(self.source),
            manifest_path=self.manifest,
            expected_manifest_sha256=benchmarking._sha256_file(self.manifest),
        )
        self.assertTrue(result["integrity"]["model_metadata_declared"])
        self.assertTrue(result["integrity"]["source"]["verified"])
        self.assertTrue(result["integrity"]["manifest"]["verified"])
        self.assertTrue(result["integrity"]["manifest_binding"]["verified"])

        mismatched = {**metadata, "model_repo": "other/model"}
        with self.assertRaisesRegex(ValueError, "does not match manifest"):
            benchmarking.benchmark_file(
                self.source, warmups=0, repetitions=1, specs=specs,
                input_metadata=mismatched,
                expected_source_size_bytes=self.source.stat().st_size,
                expected_source_sha256=benchmarking._sha256_file(self.source),
                manifest_path=self.manifest,
                expected_manifest_sha256=benchmarking._sha256_file(self.manifest),
            )

    def test_nondeterministic_measured_archives_are_structured_failures(self):
        nondeterministic = benchmarking.BaselineSpec(
            method="nondeterministic-fixture",
            profile="default",
            executable=sys.executable,
            version_args=("--version",),
            compress_args=(
                "-c",
                (
                    "import os,pathlib,sys; pathlib.Path(sys.argv[2]).write_bytes("
                    "os.urandom(16)+pathlib.Path(sys.argv[1]).read_bytes())"
                ),
                "{input}", "{output}",
            ),
            decompress_args=(
                "-c",
                (
                    "import pathlib,sys; pathlib.Path(sys.argv[2]).write_bytes("
                    "pathlib.Path(sys.argv[1]).read_bytes()[16:])"
                ),
                "{input}", "{output}",
            ),
            thread_policy=benchmarking.SERIAL_CLI,
        )
        result = benchmarking.benchmark_file(
            self.source, warmups=0, repetitions=2, specs=(nondeterministic,),
            work_dir=self.root / "work",
        )
        method = result["methods"][0]
        self.assertEqual("archive_inconsistent", method["status_code"])
        self.assertEqual("archive_inconsistent",
                         method["measured_archive_consistency"]["status_code"])
        self.assertEqual(1, method["measured_archive_consistency"]["inconsistent_runs"])
        self.assertTrue(method["runs"][0]["success"])
        self.assertTrue(method["runs"][1]["bit_exact"])
        self.assertFalse(method["runs"][1]["success"])
        self.assertEqual("archive_inconsistent", method["runs"][1]["status_code"])
        self.assertIn("not reproducible", method["runs"][1]["failure"])

    def test_version_probe_has_an_independent_short_timeout(self):
        slow_probe = benchmarking.BaselineSpec(
            method="slow-probe",
            profile="default",
            executable=sys.executable,
            version_args=("-c", "import time; time.sleep(1)"),
            compress_args=("-c", "raise SystemExit(0)", "{input}", "{output}"),
            decompress_args=("-c", "raise SystemExit(0)", "{input}", "{output}"),
            thread_policy=benchmarking.SERIAL_CLI,
        )
        result = benchmarking.benchmark_file(
            self.source, warmups=0, repetitions=1, specs=(slow_probe,),
            work_dir=self.root / "work", timeout_seconds=10,
            version_probe_timeout_seconds=0.1,
        )
        method = result["methods"][0]
        self.assertEqual("version_probe_failed", method["status_code"])
        self.assertTrue(method["version_probe"]["timed_out"])
        self.assertFalse(method["runs"])

    def test_keep_artifacts_retains_disk_outputs_logs_and_hashable_archive(self):
        result = benchmarking.benchmark_file(
            self.source,
            warmups=0,
            repetitions=1,
            specs=benchmarking.select_specs(["raw/copy"]),
            work_dir=self.root / "work",
            keep_artifacts=True,
        )
        artifact_root = pathlib.Path(result["configuration"]["artifact_root"])
        run = result["methods"][0]["runs"][0]
        run_dir = pathlib.Path(run["artifact_directory"])
        archive = run_dir / "compress" / "archive.bin"
        self.assertTrue(artifact_root.is_dir())
        self.assertTrue(archive.is_file())
        self.assertEqual(run["archive_sha256"], benchmarking._sha256_file(archive))
        self.assertTrue((run_dir / "decompress" / "restored.bin").is_file())
        self.assertTrue((run_dir / "compression.stdout").is_file())
        shutil.rmtree(artifact_root)

    def test_checkpoint_is_atomic_after_each_iteration(self):
        output = self.root / "checkpoint.json"
        original = benchmarking.write_json
        with mock.patch.object(benchmarking, "write_json", wraps=original) as writer:
            result = benchmarking.benchmark_file(
                self.source, warmups=0, repetitions=2,
                specs=benchmarking.select_specs(["raw/copy"]),
                work_dir=self.root / "work", checkpoint_path=output,
            )
        self.assertEqual(3, writer.call_count)  # two iterations plus final completion
        self.assertEqual("complete", json.loads(output.read_text())["status"])
        self.assertEqual("complete", result["status"])
        self.assertGreater(
            result["environment"]["filesystems"]["checkpoint"]["free_bytes"], 0,
        )
        first_force = writer.call_args_list[0].kwargs["force"]
        later_forces = [call.kwargs["force"] for call in writer.call_args_list[1:]]
        self.assertFalse(first_force)
        self.assertTrue(all(later_forces))

    def test_write_json_refuses_existing_output_and_force_replaces_it(self):
        output = self.root / "nested" / "results.json"
        benchmarking.write_json(output, {"value": 1})
        with self.assertRaises(FileExistsError):
            benchmarking.write_json(output, {"value": 2})
        self.assertEqual(1, json.loads(output.read_text())["value"])
        benchmarking.write_json(output, {"value": 3}, force=True)
        self.assertEqual(3, json.loads(output.read_text())["value"])
        self.assertFalse(list(output.parent.glob(f".{output.name}.*.tmp")))
        self.assertFalse(output.with_name(f".{output.name}.lock").exists())

    def test_concurrent_writes_use_unique_temps_and_publish_complete_json(self):
        output = self.root / "concurrent.json"
        barrier = threading.Barrier(2)
        created: list[str] = []
        created_lock = threading.Lock()
        real_mkstemp = tempfile.mkstemp

        def recording_mkstemp(*args, **kwargs):
            descriptor, name = real_mkstemp(*args, **kwargs)
            with created_lock:
                created.append(name)
            return descriptor, name

        def writer(value):
            barrier.wait()
            benchmarking.write_json(output, {"value": value, "padding": "x" * 10000}, force=True)

        with mock.patch.object(benchmarking.tempfile, "mkstemp", side_effect=recording_mkstemp):
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                list(executor.map(writer, (1, 2)))

        self.assertEqual(2, len(created))
        self.assertEqual(2, len(set(created)))
        self.assertIn(json.loads(output.read_text())["value"], (1, 2))
        self.assertFalse(list(self.root.glob(f".{output.name}.*.tmp")))

    def test_concurrent_no_force_writers_have_one_winner(self):
        output = self.root / "exclusive.json"
        barrier = threading.Barrier(2)

        def writer(value):
            barrier.wait()
            try:
                benchmarking.write_json(output, {"value": value})
                return "written"
            except FileExistsError:
                return "exists"

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            outcomes = list(executor.map(writer, (1, 2)))
        self.assertCountEqual(["written", "exists"], outcomes)
        self.assertIn(json.loads(output.read_text())["value"], (1, 2))
        self.assertFalse(output.with_name(f".{output.name}.lock").exists())

    def test_cli_writes_metadata_and_refuses_output_conflicts(self):
        output = self.root / "cli-results.json"
        source_sha256 = benchmarking._sha256_file(self.source)
        manifest_sha256 = benchmarking._sha256_file(self.manifest)
        base_args = [
            str(self.source),
            "--output", str(output),
            "--warmups", "0",
            "--repetitions", "1",
            "--work-dir", str(self.root / "work"),
            "--method", "raw/copy",
            "--model-tag", self.model_tag,
            "--model-repo", self.model_repo,
            "--model-revision", self.model_revision,
            "--manifest", str(self.manifest),
            "--shard", self.source.name,
            "--expected-source-size", str(self.source.stat().st_size),
            "--expected-source-sha256", source_sha256,
            "--expected-manifest-sha256", manifest_sha256,
        ]
        self.assertEqual(0, benchmarking.main(base_args))
        document = json.loads(output.read_text())
        self.assertEqual(["raw/copy"], [method["id"] for method in document["methods"]])
        self.assertTrue(document["methods"][0]["runs"][0]["bit_exact"])
        self.assertEqual(self.model_repo, document["input_metadata"]["model_repo"])
        self.assertEqual(self.source.name, document["input_metadata"]["shard"])
        self.assertEqual(manifest_sha256, document["integrity"]["manifest"]["actual_sha256"])
        self.assertTrue(document["integrity"]["manifest_binding"]["verified"])

        self.assertEqual(2, benchmarking.main(base_args))
        self.assertEqual(0, benchmarking.main([*base_args, "--force"]))

        source_before = self.source.read_bytes()
        conflict_args = [
            str(self.source), "--output", str(self.source),
            "--warmups", "0", "--repetitions", "1", "--method", "raw/copy", "--force",
        ]
        self.assertEqual(2, benchmarking.main(conflict_args))
        self.assertEqual(source_before, self.source.read_bytes())

    def test_cli_missing_tool_is_nonzero_unless_explicitly_allowed(self):
        first = self.root / "missing.json"
        second = self.root / "allowed.json"
        common = [
            str(self.source), "--warmups", "0", "--repetitions", "1",
            "--method", "raw/copy",
        ]
        with mock.patch.object(benchmarking.shutil, "which", return_value=None):
            self.assertEqual(1, benchmarking.main([*common, "--output", str(first)]))
            self.assertEqual(0, benchmarking.main([
                *common, "--output", str(second), "--allow-missing",
            ]))
        self.assertFalse(json.loads(first.read_text())["methods"][0]["available"])
        self.assertFalse(json.loads(second.read_text())["methods"][0]["available"])

    def test_argument_validation(self):
        spec = benchmarking.select_specs(["raw/copy"])
        cases = (
            {"warmups": -1, "repetitions": 1, "message": "warmups"},
            {"warmups": 0, "repetitions": 0, "message": "repetitions"},
            {"warmups": 0, "repetitions": 1, "timeout_seconds": 0, "message": "timeout"},
            {
                "warmups": 0, "repetitions": 1,
                "timeout_seconds": float("nan"), "message": "finite",
            },
            {
                "warmups": 0, "repetitions": 1,
                "timeout_seconds": float("inf"), "message": "finite",
            },
            {
                "warmups": 0, "repetitions": 1,
                "version_probe_timeout_seconds": float("nan"), "message": "finite",
            },
            {
                "warmups": 0, "repetitions": 1,
                "version_probe_timeout_seconds": float("-inf"), "message": "finite",
            },
        )
        for case in cases:
            with self.subTest(case=case):
                arguments = dict(case)
                message = arguments.pop("message")
                with self.assertRaisesRegex(ValueError, message):
                    benchmarking.benchmark_file(self.source, specs=spec, **arguments)

        with self.assertRaisesRegex(ValueError, "JSON serializable"):
            benchmarking.benchmark_file(
                self.source, warmups=0, repetitions=1, specs=spec,
                input_metadata={"invalid": object()},
            )


if __name__ == "__main__":
    unittest.main()
