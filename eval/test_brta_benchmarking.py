import hashlib
import json
import pathlib
import stat
import tempfile
import textwrap
import unittest
from unittest import mock

import brta_benchmarking as brta


FAKE_BREVIS = r"""#!/usr/bin/env python3
import json
import os
import pathlib
import sys


def option(name, default):
    try:
        return int(sys.argv[sys.argv.index(name) + 1])
    except ValueError:
        return default


command = sys.argv[1]
if command == "config":
    workers = option("--workers", 32)
    if os.environ.get("FAKE_CONFIG_WORKER_DELTA"):
        workers += int(os.environ["FAKE_CONFIG_WORKER_DELTA"])
    print(json.dumps({
        "synthesis_unit": "complete_tensor",
        "objective": "canonical_program_bytes",
        "literal_fallback": True,
        "phog_role": "queue_order_only",
        "archive": "BRTA-v1",
        "workers": workers,
        "max_expansions": option("--max-expansions", 512),
        "max_nodes": option("--max-nodes", 64),
        "max_decomposition_bytes": option("--max-tensor-bytes", 536870912),
        "max_depth": option("--max-depth", 4),
        "seed_float_fields": bool(option("--seed-float-fields", 1)),
        "max_repeat_period": option("--max-repeat-period", 32),
        "max_concat_splits": option("--max-concat-splits", 3),
        "max_map_constants": option("--max-map-constants", 2),
        "max_rotations": option("--max-rotations", 3),
        "max_field_splits": option("--max-field-splits", 3),
        "max_total_bytes": option("--max-total-bytes", 17179869184),
        "max_tensor_bytes": option("--max-tensor-bytes", 536870912),
        "max_prefix_bytes": option("--max-prefix-bytes", 67108864),
    }))
elif command == "calibrate":
    source = pathlib.Path(sys.argv[2])
    prior = pathlib.Path(sys.argv[3])
    prior.write_bytes(b"BRGP" + source.read_bytes()[:16])
    print("calibrated")
elif command == "compress":
    source = pathlib.Path(sys.argv[2])
    archive = pathlib.Path(sys.argv[3])
    if os.environ.get("FAKE_FAIL") == "compress":
        raise SystemExit(7)
    archive.write_bytes(b"BRTA\x01" + source.read_bytes())
    print("compressed")
elif command == "decompress":
    archive = pathlib.Path(sys.argv[2])
    restored = pathlib.Path(sys.argv[3])
    if os.environ.get("FAKE_FAIL") == "decompress":
        raise SystemExit(8)
    payload = archive.read_bytes()[5:]
    if os.environ.get("FAKE_CORRUPT"):
        payload += b"corrupt"
    restored.write_bytes(payload)
    print("decompressed")
else:
    raise SystemExit(9)
"""


class BrtaBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.source = self.root / "model.safetensors"
        self.source.write_bytes(
            b'{"weight":{"dtype":"U8","shape":[4096],"data_offsets":[0,4096]}}'
            + b" " * 8
            + bytes(range(256)) * 16
        )
        self.binary = self.root / "brevis"
        self.binary.write_text(textwrap.dedent(FAKE_BREVIS), encoding="utf-8")
        self.binary.chmod(self.binary.stat().st_mode | stat.S_IXUSR)

    def tearDown(self):
        self.temporary.cleanup()

    def run_benchmark(self, **overrides):
        arguments = {
            "binary": self.binary,
            "workers": (1, 4),
            "warmups": 1,
            "repetitions": 2,
            "options": {
                "max_expansions": 0,
                "max_nodes": 8,
                "max_depth": 2,
                "seed_float_fields": 0,
                "max_tensor_bytes": 1 << 20,
            },
            "work_dir": self.root / "work",
            "timeout_seconds": 5,
            "expected_input_size_bytes": self.source.stat().st_size,
            "expected_input_sha256": brta.generic._sha256_file(self.source),
            "input_label": "fixture",
        }
        arguments.update(overrides)
        return brta.benchmark_file(self.source, **arguments)

    def test_repeated_multiworker_run_records_exact_commands_hashes_and_times(self):
        result = self.run_benchmark()

        self.assertEqual(
            {"id": brta.SCHEMA_ID, "version": brta.SCHEMA_VERSION},
            result["schema"],
        )
        self.assertEqual("complete", result["status"])
        self.assertEqual("engineering", result["evidence_policy"]["run_class"])
        self.assertFalse(result["evidence_policy"]["paper_metrics_eligible"])
        self.assertTrue(result["source"]["declared_identity_verified"])
        self.assertTrue(result["source"]["unchanged_during_benchmark"])
        self.assertEqual([1, 4], result["configuration"]["workers"])
        self.assertIn("inside the measured child process", result["configuration"]["io_policy"])
        self.assertEqual(2, result["configuration"]["measured_repetitions_per_worker"])
        self.assertEqual(3, len(result["configuration"]["execution_schedule"]))
        self.assertIsNone(result["configuration"]["artifact_root"])
        self.assertFalse(result["configuration"]["artifacts_retained"])
        self.assertEqual("consistent", result["measured_archive_consistency"]["status"])
        self.assertEqual(
            4, result["measured_archive_consistency"]["successful_exact_runs"],
        )
        self.assertEqual(1, len(result["measured_archive_consistency"]["distinct_archives"]))
        self.assertEqual(40, len(result["provenance"]["git"]["commit"]))
        self.assertTrue(result["provenance"]["git"]["dirty"])
        self.assertTrue(result["provenance"]["git"]["unchanged_during_benchmark"])
        self.assertTrue(result["provenance"]["harness"]["unchanged_during_benchmark"])
        self.assertTrue(
            result["provenance"]["process_helper"]["unchanged_during_benchmark"],
        )
        self.assertEqual(64, len(result["provenance"]["binary"]["sha256"]))
        self.assertTrue(result["provenance"]["binary"]["unchanged_during_benchmark"])

        for worker_result in result["worker_results"]:
            workers = worker_result["workers"]
            effective = worker_result["config_probe"]["effective"]
            self.assertEqual(workers, effective["workers"])
            self.assertEqual(0, effective["max_expansions"])
            self.assertEqual(8, effective["max_nodes"])
            self.assertEqual(2, effective["max_depth"])
            self.assertFalse(effective["seed_float_fields"])
            self.assertEqual(1, len(worker_result["warmups"]))
            self.assertEqual(2, len(worker_result["runs"]))
            self.assertEqual(2, len(worker_result["timing_observations_ns"]["compression_wall"]))
            self.assertEqual(2, len(worker_result["timing_observations_ns"]["decompression_wall"]))
            for run in (*worker_result["warmups"], *worker_result["runs"]):
                self.assertTrue(run["success"], run["failure"])
                self.assertTrue(run["bit_exact"])
                self.assertGreater(run["compression"]["wall_time_ns"], 0)
                self.assertGreater(run["decompression"]["wall_time_ns"], 0)
                self.assertEqual(self.source.stat().st_size + 5,
                                 run["archive"]["size_bytes"])
                self.assertEqual(64, len(run["archive"]["sha256"]))
                self.assertTrue(run["archive"]["unchanged_during_decode"])
                self.assertEqual(
                    result["source"]["sha256"],
                    run["verification"]["restored_sha256"],
                )
                self.assertIn("--max-expansions", run["commands"]["compress"])
                self.assertIn("--workers", run["commands"]["compress"])
                self.assertNotIn("--max-expansions", run["commands"]["decompress"])
                self.assertIn("--max-tensor-bytes", run["commands"]["decompress"])
        json.dumps(result)

    def test_corrupt_decompression_is_retained_as_an_engineering_failure(self):
        with mock.patch.dict("os.environ", {"FAKE_CORRUPT": "1"}):
            result = self.run_benchmark(workers=(1,), warmups=0, repetitions=1)

        self.assertEqual("complete_with_failures", result["status"])
        run = result["worker_results"][0]["runs"][0]
        self.assertFalse(run["success"])
        self.assertFalse(run["bit_exact"])
        self.assertIn("differ", run["failure"])
        self.assertEqual("incomplete", result["measured_archive_consistency"]["status"])
        self.assertFalse(result["evidence_policy"]["paper_metrics_eligible"])

    def test_calibrated_prior_is_hashed_and_passed_to_every_compression(self):
        result = self.run_benchmark(
            workers=(2,),
            warmups=0,
            repetitions=1,
            calibrate_tensors=7,
        )

        self.assertEqual("complete", result["status"])
        self.assertEqual("calibrated_once", result["configuration"]["prior_policy"])
        self.assertEqual(7, result["calibration"]["max_tensors"])
        self.assertEqual(64, len(result["calibration"]["prior"]["sha256"]))
        self.assertTrue(result["prior"]["unchanged_during_uses"])
        command = result["worker_results"][0]["runs"][0]["commands"]["compress"]
        self.assertIn("--prior", command)

    def test_effective_config_mismatch_stops_before_measurement(self):
        with mock.patch.dict("os.environ", {"FAKE_CONFIG_WORKER_DELTA": "1"}):
            with self.assertRaisesRegex(brta.BenchmarkError, "effective config 'workers'"):
                self.run_benchmark(workers=(1,), warmups=0, repetitions=1)

    def test_optional_generic_track_uses_the_existing_pinned_registry(self):
        fake_generic = {
            "status": "complete",
            "methods": [{"available": True, "failure": None}],
        }
        with mock.patch.object(
            brta.generic, "benchmark_file", return_value=fake_generic,
        ) as benchmark_generic:
            result = self.run_benchmark(
                workers=(1,),
                warmups=0,
                repetitions=1,
                generic_methods=("zstd/default",),
            )

        self.assertEqual("complete", result["status"])
        self.assertEqual(
            "embedded engineering comparison only",
            result["generic_baselines"]["evidence_policy"],
        )
        selected = benchmark_generic.call_args.kwargs["specs"]
        self.assertEqual(("zstd/default",), tuple(spec.identifier for spec in selected))
        self.assertEqual(1, benchmark_generic.call_args.kwargs["repetitions"])
        self.assertEqual(
            "engineering",
            benchmark_generic.call_args.kwargs["evidence_policy"]["run_class"],
        )

    def test_input_identity_options_must_be_paired_and_match(self):
        with self.assertRaisesRegex(ValueError, "provided together"):
            self.run_benchmark(expected_input_sha256=None)
        with self.assertRaisesRegex(ValueError, "input size mismatch"):
            self.run_benchmark(expected_input_size_bytes=1)
        with self.assertRaisesRegex(ValueError, "unknown Brevis option"):
            self.run_benchmark(options={"removed_mode": 1})
        with self.assertRaisesRegex(ValueError, "seed_float_fields must be 0 or 1"):
            self.run_benchmark(options={"seed_float_fields": 2})

    def test_cli_writes_result_atomically_and_refuses_implicit_overwrite(self):
        output = self.root / "result.json"
        arguments = [
            str(self.source),
            "--output", str(output),
            "--binary", str(self.binary),
            "--workers", "1",
            "--warmups", "0",
            "--repetitions", "1",
            "--max-expansions", "0",
            "--expected-input-size", str(self.source.stat().st_size),
            "--expected-input-sha256", hashlib.sha256(
                self.source.read_bytes()
            ).hexdigest(),
        ]
        self.assertEqual(0, brta.main(arguments))
        document = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual("complete", document["status"])
        self.assertEqual(2, brta.main(arguments))
        self.assertEqual(0, brta.main([*arguments, "--force"]))


if __name__ == "__main__":
    unittest.main()
