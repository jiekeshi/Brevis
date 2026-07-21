import hashlib
import json
import os
import pathlib
import stat
import tempfile
import textwrap
import unittest
from unittest import mock

import benchmarking
import brevis_benchmarking as system


FAKE_BREVIS = r'''#!/usr/bin/env python3
import hashlib
import json
import os
import pathlib
import shutil
import struct
import sys
import time

OPS = ["raw", "bitpack", "huffman", "rans", "xor_prev", "diff_mod"]


def option(name, default=None):
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except ValueError:
        return default


def disabled():
    return [sys.argv[index + 1] for index, value in enumerate(sys.argv[:-1])
            if value == "--disable-op"]


def config():
    values = {
        "transform_layers": int(option("--max-depth", 2)),
        "max_depth": int(option("--max-depth", 2)),
        "max_nodes": int(option("--max-nodes", 12)),
        "max_expansions": int(option("--max-expansions", 256)),
        "sample_elems": int(option("--sample-elems", 4096)),
        "rerank_candidates": int(option("--rerank-candidates", 8)),
        "rerank_blocks": int(option("--rerank-blocks", 4)),
    }
    values["enabled_ops"] = [operator for operator in OPS if operator not in disabled()]
    values["disabled_ops"] = [operator for operator in OPS if operator in disabled()]
    values["enabled_ops_mask"] = 1
    return values


command = sys.argv[1]
sleep_command = os.environ.get("FAKE_BREVIS_SLEEP")
if sleep_command == command:
    time.sleep(2)
fail_command = os.environ.get("FAKE_BREVIS_FAIL")
if fail_command == command:
    print(f"intentional {command} failure", file=sys.stderr)
    raise SystemExit(7)

if command == "config":
    print(json.dumps(config()))
elif command == "calibrate":
    source = pathlib.Path(sys.argv[2])
    prior = pathlib.Path(sys.argv[3])
    prior_suffix = prior.parent.name.encode() if os.environ.get("FAKE_BREVIS_PRIOR_BY_PATH") else b""
    prior.write_bytes(b"prior:" + hashlib.sha256(source.read_bytes()).digest() + prior_suffix)
    print(json.dumps({
        "schema": 1,
        "kind": "brevis.calibration-report",
        "input": {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
        "output_prior": {
            "path": str(prior),
            "sha256": hashlib.sha256(prior.read_bytes()).hexdigest(),
            "nonempty": True,
            "context_counts_by_backoff_level": [3, 2, 1],
        },
        "configuration": {
            "max_tensors": (
                int(option("--tensors", 200))
                + int(os.environ.get("FAKE_BREVIS_CALIBRATION_TENSOR_DELTA", "0"))
            ),
            "seed": 0x5EED_B10C,
            "requested_threads": int(option("--jobs", 1)),
            "threads_used": min(int(option("--jobs", 1)), 2),
            "search": config(),
        },
        "observed": {
            "available_tensors": 2,
            "sampled_tensors": 2,
            "training_wall_ms": 3,
        },
    }))
elif command == "bench":
    source = pathlib.Path(sys.argv[2])
    source_size = source.stat().st_size
    plan = option("--plan", "search")
    prior_name = option("--prior")
    prior = pathlib.Path(prior_name) if prior_name else None
    schema = int(os.environ.get("FAKE_BREVIS_BENCH_SCHEMA", "4"))
    bytecode = b"PROGRAM" if os.environ.get("FAKE_BREVIS_PROGRAM_MISMATCH") else b"program"
    bench_search = config()
    bench_search["max_expansions"] += int(
        os.environ.get("FAKE_BREVIS_BENCH_SEARCH_DELTA", "0")
    )
    bytecode_sha256 = hashlib.sha256(bytecode).hexdigest()
    sequence = hashlib.sha256()
    sequence.update(b"brevis.program-bytecode-sequence.v1\x00")
    sequence.update(struct.pack("<Q", 1))
    sequence.update(struct.pack("<I", len(bytecode)))
    sequence.update(bytes.fromhex(bytecode_sha256))
    print(json.dumps({
        "schema": schema,
        "kind": "brevis.bench-report",
        "input": str(source),
        "input_size_bytes": source_size,
        "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "mode": "fixed" if plan == "fixed" else "phog" if prior else "uniform",
        "prior": {
            "supplied": prior is not None,
            "loaded": prior is not None,
            "applied": prior is not None,
            "guidance_active": prior is not None,
            "path": str(prior) if prior else None,
            "sha256": hashlib.sha256(prior.read_bytes()).hexdigest() if prior else None,
            "nonempty": prior is not None,
            "context_counts_by_backoff_level": [3, 2, 1] if prior else [0, 0, 0],
        },
        "threads": int(option("--jobs", 1)),
        "requested_threads": int(option("--jobs", 1)),
        "planning_workers_used": 1,
        "encoding_workers_used": 1,
        "search_options_applied": plan == "search",
        "search": bench_search,
        "planning_wall_ms": 2,
        "encoding_wall_ms": 1,
        "raw_bytes": source_size,
        "tensor_data_bytes": source_size,
        "safetensors_prefix_bytes": 0,
        "encoded_bytes_without_frame_headers": source_size + len(bytecode),
        "block_frame_bytes_excluding_container_header_footer": source_size + len(bytecode) + 12,
        "container_header_bytes": 8,
        "container_footer_bytes": 12,
        "projected_archive_bytes": (
            source_size + len(bytecode) + 32
            + int(os.environ.get("FAKE_BREVIS_PROJECTION_DELTA", "0"))
        ),
        "program_bytecode_evidence": {
            "version": 1,
            "hash": "sha256",
            "sequence_spec_id": "brevis.program-bytecode-sequence.v1",
            "block_count": 1,
            "sequence_sha256": sequence.hexdigest(),
        },
        "tensors": [{
            "index": 0,
            "name": "weight",
            "dtype": "U8",
            "shape": [source_size],
            "numel": source_size,
            "raw_bytes": source_size,
            "file_data_start_byte": 0,
            "file_data_end_byte_exclusive": source_size,
            "encoded_bytes_without_frame_headers": source_size + len(bytecode),
            "block_frame_bytes_excluding_container_header_footer": source_size + len(bytecode) + 12,
            "block_start": 0,
            "block_count": 1,
            "raw_root_blocks": 1,
            "planned_raw_root_blocks": 1,
            "fallback_raw_root_blocks": 0,
            "search_expansions": 1,
            "candidates_realized": 1 if plan == "search" else None,
            "candidates_reranked": 1 if plan == "search" else None,
            "probe_blocks_used": 1 if plan == "search" else None,
            "selected_sample_rank_zero_based": 0 if plan == "search" else None,
            "program": "raw",
            "root_operator": "raw",
            "program_tree": {"op": "raw", "params_u32": [], "terminal": True, "children": []},
            "program_nodes": 1,
            "program_depth": 1,
            "program_node_depth": 1,
            "program_transform_depth": 0,
            "terminal_count": 1,
        }],
        "blocks": [{
            "index": 0,
            "tensor_index": 0,
            "element_offset": 0,
            "element_count": source_size,
            "raw_bytes": source_size,
            "encoded_bytes_without_frame_headers": source_size + len(bytecode),
            "program_bytecode_bytes": len(bytecode),
            "program_bytecode_sha256": bytecode_sha256,
            "packed_terminal_payload_bytes": source_size,
            "frame_header_bytes": len(bytecode) + 12,
            "framed_bytes": source_size + len(bytecode) + 12,
            "raw_classification": "planned_raw",
            "program": "raw",
            "root_operator": "raw",
            "program_tree": {"op": "raw", "params_u32": [], "terminal": True, "children": []},
            "program_nodes": 1,
            "program_depth": 1,
            "program_node_depth": 1,
            "program_transform_depth": 0,
            "terminal_count": 1,
        }],
        "future_schema_field": {
            "preserved": True,
            "semantic_nonce": (
                os.getpid()
                if os.environ.get("FAKE_BREVIS_SEMANTIC_NONDETERMINISTIC")
                else 0
            ),
        },
    }))
elif command == "compress":
    source = pathlib.Path(sys.argv[2])
    archive = pathlib.Path(sys.argv[3])
    payload = source.read_bytes()
    bytecode = b"program"
    index_offset = 8 + 4 + len(bytecode) + 8 + len(payload)
    archive.write_bytes(
        b"BRV\x03\x06\x00\x00\x00"
        + struct.pack("<I", len(bytecode)) + bytecode
        + struct.pack("<Q", len(payload)) + payload
        + b"BRVF" + struct.pack("<Q", index_offset)
    )
    print("compressed")
elif command == "decompress":
    archive = pathlib.Path(sys.argv[2])
    restored = pathlib.Path(sys.argv[3])
    encoded = archive.read_bytes()
    program_length = struct.unpack_from("<I", encoded, 8)[0]
    payload_length_offset = 12 + program_length
    payload_length = struct.unpack_from("<Q", encoded, payload_length_offset)[0]
    payload_offset = payload_length_offset + 8
    restored.write_bytes(encoded[payload_offset:payload_offset + payload_length])
    print("decompressed")
else:
    raise SystemExit(9)
'''


class BrevisSystemBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        self.source = self.root / "weights.safetensors"
        self.source.write_bytes((b"\x00\x00\x80\x3f" * 4096) + bytes(range(256)))
        self.binary = self.root / "fake-brevis"
        self.binary.write_text(textwrap.dedent(FAKE_BREVIS), encoding="utf-8")
        self.binary.chmod(self.binary.stat().st_mode | stat.S_IXUSR)
        self.tag = "fixture"
        self.repo = "test/model"
        self.revision = "a" * 40
        self.manifest = self.root / "models.json"
        self.manifest.write_text(json.dumps([{
            "tag": self.tag,
            "repo": self.repo,
            "revision": self.revision,
            "files": [{
                "file": self.source.name,
                "bytes": self.source.stat().st_size,
                "sha256": benchmarking._sha256_file(self.source),
            }],
        }]) + "\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def arguments(self, **overrides):
        arguments = {
            "binary": self.binary,
            "warmups": 0,
            "repetitions": 2,
            "jobs": 2,
            "calibration_tensors": 7,
            "work_dir": self.root / "work",
            "timeout_seconds": 5,
            "input_metadata": {
                "model_tag": self.tag,
                "model_repo": self.repo,
                "model_revision": self.revision,
                "manifest": str(self.manifest),
                "shard": self.source.name,
            },
            "expected_source_size_bytes": self.source.stat().st_size,
            "expected_source_sha256": benchmarking._sha256_file(self.source),
            "manifest_path": self.manifest,
            "expected_manifest_sha256": benchmarking._sha256_file(self.manifest),
            "require_clean_git": False,
            "build_binary": False,
            "disk_reserve_fraction": 0,
            "temp_multiplier": 1,
            "temp_fixed_bytes": 0,
        }
        arguments.update(overrides)
        return arguments

    def test_registry_materializes_auditable_raw_and_rejects_inert_fixed_options(self):
        specs = system.core_specs(["raw", "bitpack", "huffman", "xor_prev"])
        raw, fixed, uniform, phog = specs
        self.assertEqual(system.CORE_CONFIGURATION_IDS,
                         tuple(spec.identifier for spec in specs))
        self.assertEqual((
            "--disable-op", "bitpack", "--disable-op", "huffman",
            "--disable-op", "xor_prev",
        ), raw.search_args)
        self.assertNotIn("raw", raw.search_args)
        self.assertEqual("fixed", fixed.plan)
        self.assertEqual("none", uniform.prior_policy)
        self.assertEqual("canonical", phog.prior_policy)
        with self.assertRaisesRegex(ValueError, "ignores search"):
            system.BrevisSpec("bad-fixed", "fixed", "none", ("--max-depth", "1"))

    def test_program_sequence_digest_has_fixed_vectors(self):
        blocks = [b"abc", b"de"]
        self.assertEqual(
            "cd77a0457217e6956d17f467f495d6ab8eaf31399de9a04d86eafe28a75c5af4",
            system._program_sequence_sha256(
                [len(block) for block in blocks],
                [hashlib.sha256(block).hexdigest() for block in blocks],
            ),
        )
        self.assertEqual(
            "518dd014553ca077945bf5c0144e2091c78f07a1405093d2bc31507c06b2d09c",
            system._program_sequence_sha256([], []),
        )

    def test_bench_detail_fingerprint_excludes_only_timing_and_local_paths(self):
        report = {
            "schema": 4,
            "kind": "brevis.bench-report",
            "input": "/local/a/model.safetensors",
            "planning_wall_ms": 3,
            "encoding_wall_ms": 4,
            "prior": {"path": "/local/a/prior.bin", "sha256": "a" * 64},
            "search": {"max_depth": 2},
            "tensors": [{"name": "weight", "program": "raw"}],
            "blocks": [{"program_bytecode_sha256": "b" * 64}],
            "future": {"must_remain_semantic": 7},
        }
        changed_observations = json.loads(json.dumps(report))
        changed_observations.update({
            "input": "/different/mount/model.safetensors",
            "planning_wall_ms": 300,
            "encoding_wall_ms": 400,
        })
        changed_observations["prior"]["path"] = "/different/mount/prior.bin"
        self.assertEqual(
            system._bench_detail_semantic_sha256(report),
            system._bench_detail_semantic_sha256(changed_observations),
        )

        changed_semantics = json.loads(json.dumps(changed_observations))
        changed_semantics["future"]["must_remain_semantic"] = 8
        self.assertNotEqual(
            system._bench_detail_semantic_sha256(report),
            system._bench_detail_semantic_sha256(changed_semantics),
        )

    def test_complete_core_run_is_integrity_bound_paired_exact_and_checkpointed(self):
        output = self.root / "result.json"
        result = system.benchmark_file(
            self.source,
            checkpoint_path=output,
            **self.arguments(warmups=1),
        )

        self.assertEqual(
            {"id": system.SCHEMA_ID, "version": system.SCHEMA_VERSION},
            result["schema"],
        )
        self.assertEqual("complete", result["status"])
        self.assertTrue(result["success"], result["failure"])
        self.assertEqual("pilot", result["run_classification"]["run_class"])
        self.assertFalse(result["run_classification"]["formal_eligible"])
        self.assertEqual(
            {"clean_git_gate_disabled", "releasefast_build_not_performed_by_harness"},
            set(result["run_classification"]["formal_ineligibility_reasons"]),
        )
        self.assertTrue(result["integrity"]["source"]["verified"])
        self.assertTrue(result["integrity"]["manifest_binding"]["verified"])
        self.assertFalse(result["provenance"]["git"]["dirty"] is None)
        self.assertEqual("not_verified_by_harness", result["provenance"]["build"]["mode"])
        self.assertTrue(result["provenance"]["binary"]["unchanged_during_benchmark"])
        self.assertEqual("consistent",
                         result["calibration"]["measured_prior_consistency"]["status_code"])
        self.assertTrue(result["calibration"]["measured_prior_consistency"]
                        ["context_counts_consistent"])
        self.assertEqual(64, len(result["calibration"]["canonical_prior"]["sha256"]))
        self.assertIsNone(result["calibration"]["canonical_prior"]["path"])
        self.assertFalse(result["calibration"]["canonical_prior"]["retained"])
        self.assertTrue(all(run["prior"]["path"] is None
                            and not run["prior"]["retained"]
                            for run in result["calibration"]["runs"]))

        configurations = {record["id"]: record for record in result["configurations"]}
        self.assertEqual(set(system.CORE_CONFIGURATION_IDS), set(configurations))
        self.assertEqual(["raw"], configurations["raw-terminal"]
                         ["effective_search_configuration"]["enabled_ops"])
        for identifier, configuration in configurations.items():
            with self.subTest(configuration=identifier):
                self.assertEqual("ok", configuration["status_code"])
                self.assertEqual(1, len(configuration["warmups"]))
                self.assertEqual(2, len(configuration["runs"]))
                self.assertEqual("consistent",
                                 configuration["measured_archive_consistency"]["status_code"])
                detail_consistency = configuration["bench_report_detail_consistency"]
                self.assertEqual("consistent", detail_consistency["status_code"])
                self.assertEqual("warmup", detail_consistency["canonical_reference"]["phase"])
                self.assertEqual(3, detail_consistency["matching_reports_including_canonical"])
                self.assertEqual(0, detail_consistency["mismatching_reports"])
                for run in (*configuration["warmups"], *configuration["runs"]):
                    self.assertTrue(run["success"], run["failure"])
                    self.assertTrue(run["verification"]["bit_exact"])
                    self.assertGreater(run["compression"]["wall_time_ns"], 0)
                    self.assertGreater(run["decompression"]["direct_child_max_rss_bytes"], 0)
                    self.assertEqual(4, run["bench_report"]["schema"])
                    self.assertTrue(run["bench_report"]["future_schema_field"]["preserved"])
                    self.assertTrue(run["bench_process"]["stdout"]
                                    ["embedded_as_parsed_json"])
                    self.assertEqual("consistent", run
                                     ["bench_archive_projection_consistency"]["status_code"])
                    self.assertEqual("consistent", run
                                     ["bench_archive_program_consistency"]["status_code"])
                    self.assertTrue(run["bench_archive_program_consistency"]["matches"])
                    self.assertEqual(
                        run["bench_report"]["program_bytecode_evidence"]["sequence_sha256"],
                        run["archive"]["program_bytecode_evidence"]["sequence_sha256"],
                    )
                    materialized = system._materialize_bench_report_detail(
                        configuration, run,
                    )
                    self.assertEqual(1, len(materialized["tensors"]))
                    self.assertEqual(1, len(materialized["blocks"]))
                    self.assertEqual(
                        run["bench_report_detail"]["semantic_sha256"],
                        system._bench_detail_semantic_sha256(materialized),
                    )
                    self.assertTrue(run["archive"]["unchanged_during_decode"])
                    self.assertIsNone(run["artifact_directory"])
                canonical = configuration["warmups"][0]
                self.assertEqual("inline_canonical",
                                 canonical["bench_report_detail"]["status_code"])
                self.assertIn("tensors", canonical["bench_report"])
                self.assertIn("blocks", canonical["bench_report"])
                for run in configuration["runs"]:
                    self.assertEqual("canonical_reference",
                                     run["bench_report_detail"]["status_code"])
                    self.assertEqual(
                        set(canonical["bench_report"]) - {"tensors", "blocks"},
                        set(run["bench_report"]),
                    )
                    self.assertNotIn("tensors", run["bench_report"])
                    self.assertNotIn("blocks", run["bench_report"])
                    self.assertEqual(2, run["bench_report"]["planning_wall_ms"])
                    self.assertEqual(1, run["bench_report"]["encoding_wall_ms"])

        schedule = result["configuration"]["execution_schedule"]
        self.assertTrue(schedule)
        self.assertTrue(all(entry["status"] == "completed" for entry in schedule))
        measured = [
            entry for entry in schedule
            if entry["phase"] == "measured" and entry["stage"] == "archive_pipeline"
        ]
        first = [entry["configuration"] for entry in measured if entry["repetition"] == 0]
        second = [entry["configuration"] for entry in measured if entry["repetition"] == 1]
        self.assertEqual(first[::-1], second)
        checkpoint_document = json.loads(output.read_text())
        self.assertEqual("complete", checkpoint_document["status"])
        checkpoint_configuration = checkpoint_document["configurations"][0]
        system._materialize_bench_report_detail(
            checkpoint_configuration, checkpoint_configuration["runs"][0],
        )
        self.assertIsNone(result["configuration"]["artifact_root"])
        self.assertTrue(result["cleanup"]["artifact_root"]["success"])
        self.assertIsNotNone(result["disk_gate"]["checkpoint"])
        self.assertFalse([path for path in (self.root / "work").glob("brevis-system-*")])

    def test_first_measured_report_is_canonical_when_warmups_are_disabled(self):
        result = system.benchmark_file(
            self.source,
            **self.arguments(
                configuration_ids=("uniform",), warmups=0, repetitions=2,
            ),
        )
        self.assertTrue(result["success"], result["failure"])
        configuration = result["configurations"][0]
        consistency = configuration["bench_report_detail_consistency"]
        self.assertEqual("measured", consistency["canonical_reference"]["phase"])
        self.assertEqual(0, consistency["canonical_reference"]["run_index"])
        first, second = configuration["runs"]
        self.assertEqual("inline_canonical", first["bench_report_detail"]["status_code"])
        self.assertIn("tensors", first["bench_report"])
        self.assertEqual("canonical_reference", second["bench_report_detail"]["status_code"])
        self.assertNotIn("tensors", second["bench_report"])
        self.assertEqual(
            first["bench_report_detail"]["semantic_sha256"],
            second["bench_report_detail"]["semantic_sha256"],
        )
        self.assertEqual(
            first["bench_report"]["tensors"],
            system._materialize_bench_report_detail(configuration, second)["tensors"],
        )

    def test_semantic_mismatch_retains_full_detail_and_fails_closed(self):
        with mock.patch.dict(
            os.environ, {"FAKE_BREVIS_SEMANTIC_NONDETERMINISTIC": "1"},
        ):
            result = system.benchmark_file(
                self.source,
                **self.arguments(
                    configuration_ids=("uniform",), warmups=1, repetitions=1,
                ),
            )
        self.assertFalse(result["success"])
        configuration = result["configurations"][0]
        canonical = configuration["warmups"][0]
        mismatch = configuration["runs"][0]
        self.assertEqual(
            "semantic_mismatch",
            configuration["bench_report_detail_consistency"]["status_code"],
        )
        self.assertEqual(1, configuration["bench_report_detail_consistency"]
                         ["mismatching_reports"])
        self.assertEqual("inline_canonical",
                         canonical["bench_report_detail"]["status_code"])
        self.assertEqual("semantic_mismatch_retained",
                         mismatch["bench_report_detail"]["status_code"])
        self.assertFalse(mismatch["bench_report_detail"]["matches_canonical"])
        self.assertIn("tensors", mismatch["bench_report"])
        self.assertIn("blocks", mismatch["bench_report"])
        self.assertFalse(mismatch["success"])
        self.assertFalse(mismatch["diagnostic_success"])
        self.assertIn("semantics differ", mismatch["failure"])

    def test_failed_bench_report_never_becomes_canonical(self):
        with mock.patch.dict(os.environ, {"FAKE_BREVIS_FAIL": "bench"}):
            result = system.benchmark_file(
                self.source,
                **self.arguments(
                    configuration_ids=("uniform",), warmups=1, repetitions=1,
                ),
            )
        self.assertFalse(result["success"])
        configuration = result["configurations"][0]
        consistency = configuration["bench_report_detail_consistency"]
        self.assertEqual("no_successful_canonical_report", consistency["status_code"])
        self.assertIsNone(consistency["canonical_reference"])
        self.assertEqual(2, consistency["reports_unavailable"])
        for run in (*configuration["warmups"], *configuration["runs"]):
            self.assertIsNone(run["bench_report"])
            self.assertEqual("report_unavailable",
                             run["bench_report_detail"]["status_code"])
            self.assertIsNone(run["bench_report_detail"]["semantic_sha256"])
            self.assertFalse(run["bench_report_detail"]["eligible_for_dsl_analysis"])

    def test_unknown_and_old_bench_schemas_are_structured_failures(self):
        with mock.patch.dict(os.environ, {"FAKE_BREVIS_BENCH_SCHEMA": "7"}):
            future = system.benchmark_file(
                self.source,
                **self.arguments(configuration_ids=("uniform",), repetitions=1),
            )
        self.assertFalse(future["success"])
        future_run = future["configurations"][0]["runs"][0]
        self.assertEqual(7, future_run["bench_report"]["schema"])
        self.assertIn("supports bench schema 4 exactly", future_run["failure"])
        self.assertEqual(
            "failed_or_incomplete_report_retained",
            future_run["bench_report_detail"]["status_code"],
        )
        self.assertIn("tensors", future_run["bench_report"])
        self.assertIsNotNone(future_run["bench_report_detail"]["semantic_sha256"])

        with mock.patch.dict(os.environ, {"FAKE_BREVIS_BENCH_SCHEMA": "1"}):
            old = system.benchmark_file(
                self.source,
                **self.arguments(configuration_ids=("uniform",), repetitions=1),
            )
        run = old["configurations"][0]["runs"][0]
        self.assertFalse(old["success"])
        self.assertFalse(run["success"])
        self.assertIn("schema", run["failure"])
        # The archive path still ran and was verified, preserving partial evidence.
        self.assertTrue(run["verification"]["bit_exact"])

    def test_timeout_is_retained_without_losing_valid_archive_evidence(self):
        with mock.patch.dict(os.environ, {"FAKE_BREVIS_SLEEP": "bench"}):
            result = system.benchmark_file(
                self.source,
                **self.arguments(
                    configuration_ids=("uniform",), repetitions=1,
                    timeout_seconds=0.05,
                ),
            )
        run = result["configurations"][0]["runs"][0]
        self.assertFalse(result["success"])
        self.assertTrue(run["bench_process"]["timed_out"])
        self.assertIn("timed out", run["failure"])
        self.assertTrue(run["verification"]["bit_exact"])

    def test_schema_four_archive_projection_mismatch_is_a_failure(self):
        with mock.patch.dict(os.environ, {"FAKE_BREVIS_PROJECTION_DELTA": "1"}):
            result = system.benchmark_file(
                self.source,
                **self.arguments(configuration_ids=("uniform",), repetitions=1),
            )
        run = result["configurations"][0]["runs"][0]
        self.assertFalse(result["success"])
        self.assertEqual(
            "projection_mismatch",
            run["bench_archive_projection_consistency"]["status_code"],
        )
        self.assertIn("projected_archive_bytes", run["failure"])

    def test_same_size_program_mismatch_is_a_failure(self):
        with mock.patch.dict(os.environ, {"FAKE_BREVIS_PROGRAM_MISMATCH": "1"}):
            result = system.benchmark_file(
                self.source,
                **self.arguments(configuration_ids=("uniform",), repetitions=1),
            )
        run = result["configurations"][0]["runs"][0]
        self.assertFalse(result["success"])
        self.assertEqual(
            "consistent", run["bench_archive_projection_consistency"]["status_code"],
        )
        self.assertEqual(
            "program_mismatch", run["bench_archive_program_consistency"]["status_code"],
        )
        self.assertEqual([0], run["bench_archive_program_consistency"]["mismatch_block_indices"])
        self.assertIn("program bytecode", run["failure"])

    def test_effective_calibration_and_bench_configs_are_enforced(self):
        with mock.patch.dict(
            os.environ, {"FAKE_BREVIS_CALIBRATION_TENSOR_DELTA": "1"},
        ):
            calibration = system.benchmark_file(
                self.source,
                **self.arguments(configuration_ids=("phog",), repetitions=1),
            )
        self.assertFalse(calibration["success"])
        self.assertIn("max_tensors", calibration["calibration"]["runs"][0]["failure"])
        self.assertIsNotNone(calibration["failure"])

        with mock.patch.dict(os.environ, {"FAKE_BREVIS_BENCH_SEARCH_DELTA": "1"}):
            bench = system.benchmark_file(
                self.source,
                **self.arguments(configuration_ids=("uniform",), repetitions=1),
            )
        run = bench["configurations"][0]["runs"][0]
        self.assertFalse(bench["success"])
        self.assertIn("search configuration", run["failure"])

    def test_replicated_prior_inconsistency_fails_closed(self):
        with mock.patch.dict(os.environ, {"FAKE_BREVIS_PRIOR_BY_PATH": "1"}):
            result = system.benchmark_file(
                self.source,
                **self.arguments(configuration_ids=("phog",), repetitions=2),
            )
        self.assertFalse(result["success"])
        self.assertEqual(
            "prior_replication_inconsistent", result["calibration"]["status_code"],
        )
        self.assertFalse(
            result["calibration"]["measured_prior_consistency"]["byte_consistent"]
        )
        self.assertIn("semantic equivalence is not assumed", result["calibration"]["failure"])
        self.assertIsNotNone(result["failure"])

    def test_first_checkpoint_contains_complete_pending_schedule(self):
        output = self.root / "scheduled.json"
        snapshots = []
        original_write = benchmarking.write_json

        def capture(path, payload, *, force=False):
            snapshots.append(json.loads(json.dumps(payload)))
            return original_write(path, payload, force=force)

        with mock.patch.object(system.common, "write_json", side_effect=capture):
            result = system.benchmark_file(
                self.source,
                checkpoint_path=output,
                **self.arguments(
                    configuration_ids=("uniform", "phog"), warmups=1, repetitions=2,
                ),
            )
        self.assertTrue(result["success"], result["failure"])
        first = snapshots[0]
        schedule = first["configuration"]["execution_schedule"]
        self.assertEqual(12, len(schedule))
        self.assertTrue(all(entry["status"] == "pending" for entry in schedule))
        calibration_schedule = first["calibration"]["execution_schedule"]
        self.assertEqual(3, len(calibration_schedule))
        self.assertTrue(all(entry["status"] == "pending" for entry in calibration_schedule))
        compact_references_checked = 0
        for snapshot in snapshots:
            for configuration in snapshot["configurations"]:
                for run in (*configuration["warmups"], *configuration["runs"]):
                    detail = run.get("bench_report_detail")
                    if (
                        isinstance(detail, dict)
                        and detail.get("status_code") == "canonical_reference"
                    ):
                        system._materialize_bench_report_detail(configuration, run)
                        compact_references_checked += 1
        self.assertGreater(compact_references_checked, 0)

    def test_cleanup_failure_is_visible_and_invalidates_the_run(self):
        original_cleanup = system._cleanup_directory
        injected = False

        def fail_once(path, label):
            nonlocal injected
            actual = original_cleanup(path, label)
            if not injected and label.endswith("-archive"):
                injected = True
                return {**actual, "success": False, "error": "injected cleanup failure"}
            return actual

        with mock.patch.object(system, "_cleanup_directory", side_effect=fail_once):
            result = system.benchmark_file(
                self.source,
                **self.arguments(configuration_ids=("uniform",), repetitions=1),
            )
        self.assertFalse(result["success"])
        self.assertIn("cleanup failed", result["failure"])
        run = result["configurations"][0]["runs"][0]
        self.assertFalse(run["cleanup"]["success"])
        self.assertIn("injected cleanup failure", run["failure"])

    def test_compression_failure_is_structured_and_skips_decode(self):
        with mock.patch.dict(os.environ, {"FAKE_BREVIS_FAIL": "compress"}):
            result = system.benchmark_file(
                self.source,
                **self.arguments(configuration_ids=("uniform",), repetitions=1),
            )
        run = result["configurations"][0]["runs"][0]
        self.assertEqual(7, run["compression"]["exit_code"])
        self.assertIsNone(run["archive"])
        self.assertIsNone(run["decompression"])
        self.assertFalse(run["success"])
        self.assertIn("status 7", run["failure"])
        self.assertIsNotNone(result["failure"])

    def test_integrity_clean_tree_and_disk_gates_fail_before_measurement(self):
        with self.assertRaisesRegex(ValueError, "source SHA-256 mismatch"):
            system.benchmark_file(
                self.source,
                **self.arguments(expected_source_sha256="0" * 64),
            )

        with mock.patch.object(system.common, "_git_provenance", return_value={
            "repository": str(system.ROOT),
            "commit": "b" * 40,
            "dirty": True,
            "status_porcelain": " M src/main.zig",
        }):
            with self.assertRaisesRegex(system.BenchmarkError, "clean Git tree"):
                system.benchmark_file(
                    self.source,
                    **self.arguments(require_clean_git=True),
                )

        output = self.root / "skipped.json"
        skipped = system.benchmark_file(
            self.source,
            checkpoint_path=output,
            **self.arguments(temp_multiplier=1e20),
        )
        self.assertEqual("skipped_resource", skipped["status"])
        self.assertFalse(skipped["success"])
        self.assertFalse(skipped["disk_gate"]["passes"])
        self.assertFalse(skipped["configurations"])
        self.assertEqual("skipped_resource", json.loads(output.read_text())["status"])

        with self.assertRaisesRegex(ValueError, "unrepresentable disk estimate"):
            system.benchmark_file(
                self.source,
                **self.arguments(temp_multiplier=1e308),
            )

    def test_cli_preserves_manifest_identity_and_refuses_to_clobber(self):
        output = self.root / "cli.json"
        arguments = [
            str(self.source), "--output", str(output),
            "--binary", str(self.binary), "--skip-build", "--allow-dirty",
            "--warmups", "0", "--repetitions", "1", "--jobs", "2",
            "--configuration", "uniform", "--work-dir", str(self.root / "cli-work"),
            "--disk-reserve-fraction", "0", "--temp-multiplier", "1",
            "--model-tag", self.tag, "--model-repo", self.repo,
            "--model-revision", self.revision, "--manifest", str(self.manifest),
            "--shard", self.source.name,
            "--expected-source-size", str(self.source.stat().st_size),
            "--expected-source-sha256", benchmarking._sha256_file(self.source),
            "--expected-manifest-sha256", benchmarking._sha256_file(self.manifest),
        ]
        self.assertEqual(0, system.main(arguments))
        document = json.loads(output.read_text())
        self.assertEqual(self.tag, document["integrity"]["manifest_binding"]["tag"])
        self.assertEqual(["uniform"], document["configuration"]["requested_configuration_ids"])
        self.assertEqual(2, system.main(arguments))
        self.assertEqual(0, system.main([*arguments, "--force"]))


if __name__ == "__main__":
    unittest.main()
