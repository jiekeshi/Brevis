#!/usr/bin/env python3

from __future__ import annotations

import copy
import hashlib
import json
import pathlib
import tempfile
import unittest

import summarize_system_benchmarks as summary
import test_analyze_generated_dsl as dsl_fixture


GIT_COMMIT = "1" * 40
MANIFEST_SHA256 = "2" * 64
BINARY_SHA256 = "3" * 64
HARNESS_SHA256 = "4" * 64
HELPER_SHA256 = "5" * 64
DIFF_SHA256 = "6" * 64


def process(wall_time_ns: int, output_size_bytes: int | None = None) -> dict:
    record = {
        "status_code": "ok",
        "wall_time_ns": wall_time_ns,
        "timed_out": False,
        "error": None,
        "exit_code": 0,
        "direct_child_max_rss_bytes": 4096 + wall_time_ns,
    }
    if output_size_bytes is not None:
        record["output_size_bytes"] = output_size_bytes
    return record


def formal_document(*, pilot: bool = False, config_id: str = "uniform") -> dict:
    document = dsl_fixture.system_document(pilot=pilot, config_id=config_id)
    source_size = document["source"]["size_bytes"]
    source_sha = document["source"]["sha256"]
    configuration = document["configurations"][0]
    archive_size = configuration["runs"][0]["archive"]["storage"]["logical_size_bytes"]
    document["configuration"] = {
        "requested_configuration_ids": [configuration["id"]],
        "warmups": 1,
        "repetitions": 2,
        "jobs": 1,
        "timing_scope": "external full-process wall time",
        "scheduling_policy": "archive pipelines precede diagnostic replays",
    }
    document["input_metadata"] = {
        "model_tag": "fixture-f32",
        "model_repo": "example/model",
        "model_revision": "revision-1",
        "shard": "model.safetensors",
    }
    document["integrity"] = {
        "source": {
            "expected_size_bytes": source_size,
            "actual_size_bytes": source_size,
            "expected_sha256": source_sha,
            "actual_sha256": source_sha,
            "verified": True,
        },
        "manifest": {
            "path": "/immutable/model-manifest.json",
            "size_bytes": 123,
            "expected_sha256": MANIFEST_SHA256,
            "actual_sha256": MANIFEST_SHA256,
            "verified": True,
        },
        "manifest_binding": {
            "tag": "fixture-f32",
            "repo": "example/model",
            "revision": "revision-1",
            "shard": "model.safetensors",
            "bytes": source_size,
            "sha256": source_sha,
            "entry_sha256": "7" * 64,
            "verified": True,
        },
    }
    document["provenance"] = {
        "git": {"commit": GIT_COMMIT, "dirty": False},
        "git_tracked_start": {
            "commit": GIT_COMMIT,
            "tracked_diff_sha256": DIFF_SHA256,
            "tracked_dirty": False,
        },
        "build": {"requested": True, "mode": "ReleaseFast", "success": True},
        "binary": {
            "executable": True,
            "sha256_before_runs": BINARY_SHA256,
            "sha256_after_runs": BINARY_SHA256,
            "unchanged_during_benchmark": True,
        },
        "harness": {"path": "/code/brevis_benchmarking.py", "sha256": HARNESS_SHA256},
        "generic_helper": {"path": "/code/benchmarking.py", "sha256": HELPER_SHA256},
        "end_checks": {
            "performed": True,
            "success": True,
            "failures": [],
            "checks": {
                "source": {
                    "expected_sha256": source_sha,
                    "actual_sha256": source_sha,
                    "unchanged": True,
                    "error": None,
                },
                "manifest": {
                    "expected_sha256": MANIFEST_SHA256,
                    "actual_sha256": MANIFEST_SHA256,
                    "unchanged": True,
                    "error": None,
                },
                "harness": {
                    "expected_sha256": HARNESS_SHA256,
                    "actual_sha256": HARNESS_SHA256,
                    "unchanged": True,
                    "error": None,
                },
                "generic_helper": {
                    "expected_sha256": HELPER_SHA256,
                    "actual_sha256": HELPER_SHA256,
                    "unchanged": True,
                    "error": None,
                },
                "git_tracked_state": {
                    "same_commit": True,
                    "same_tracked_diff": True,
                    "unchanged": True,
                },
            },
        },
    }
    document["environment"] = {
        "system": "Linux",
        "machine": "x86_64",
        "cpu": {"logical_cpus": 8},
    }
    document["calibration"] = {
        "required": config_id == "phog",
        "status_code": "ok" if config_id == "phog" else "not_required",
        "success": True,
        "failure": None,
        "warmups": [],
        "runs": [],
    }
    if config_id == "phog":
        prior_sha = "d" * 64
        document["calibration"].update({
            "measured_prior_consistency": {
                "status_code": "consistent",
                "reference_sha256": prior_sha,
                "distinct_sha256": [prior_sha],
                "byte_consistent": True,
                "distinct_context_counts_by_backoff_level": [[3, 2, 1]],
                "context_counts_consistent": True,
            },
            "canonical_prior": {
                "sha256": prior_sha,
                "sha256_after_uses": prior_sha,
                "unchanged_during_benchmark": True,
            },
        })
        for phase, count, destination in (
            ("warmup", 1, document["calibration"]["warmups"]),
            ("measured", 2, document["calibration"]["runs"]),
        ):
            for index in range(count):
                destination.append({
                    "status_code": "ok",
                    "phase": phase,
                    "index": index,
                    "process": process(1_000 + index * 100),
                    "prior": {"sha256": prior_sha},
                    "success": True,
                    "failure": None,
                })

    for phase, records in (("warmup", configuration["warmups"]),
                           ("measured", configuration["runs"])):
        for position, run in enumerate(records):
            compression_ns = 90 if phase == "warmup" else 100 + position * 100
            decompression_ns = 180 if phase == "warmup" else 200 + position * 100
            run.update({
                "status_code": "ok",
                "archive_execution_order": position,
                "diagnostic_execution_order": 10 + position,
                "compression": process(compression_ns, archive_size),
                "decompression": process(decompression_ns, source_size),
                "bench_process": process(50 + position * 10),
                "timing_relationship": summary.DIAGNOSTIC_TIMING_RELATIONSHIP,
            })
    return document


def document_sha(document: dict) -> str:
    raw = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


class SystemSummaryTests(unittest.TestCase):
    def test_valid_formal_result_preserves_trials_and_auditable_statistics(self):
        document = formal_document()
        digest = document_sha(document)
        shard = summary.summarize_document(
            "fixture.json", document, result_file_sha256=digest,
        )

        self.assertTrue(shard["paper_metrics_eligible"])
        self.assertEqual(digest, shard["result_file"]["sha256"])
        self.assertEqual(MANIFEST_SHA256, shard["manifest"]["sha256"])
        self.assertEqual(GIT_COMMIT, shard["implementation"]["git_commit"])
        configuration = shard["configurations"][0]
        self.assertEqual(2, len(configuration["measured_trials"]))
        self.assertEqual(150.0, configuration["compression"]["wall_time_ns"]["mean"])
        self.assertAlmostEqual(
            70.71067811865476,
            configuration["compression"]["wall_time_ns"]["sample_sd"],
        )
        self.assertFalse(configuration["search_diagnostic_replay"]["compression_time_component"])
        self.assertEqual(
            "/configurations/0/runs/0/compression",
            configuration["measured_trials"][0]["compression"]["trace"]["json_pointer"],
        )
        self.assertFalse(shard["aggregation"]["model_level_aggregation_performed"])

    def test_pilot_is_rejected_by_default_and_explicitly_ineligible_when_allowed(self):
        document = formal_document(pilot=True)
        with self.assertRaises(summary.SummaryError) as caught:
            summary.summarize_document(
                "pilot.json", document, result_file_sha256=document_sha(document),
            )
        self.assertEqual("pilot_excluded", caught.exception.code)

        shard = summary.summarize_document(
            "pilot.json", document, result_file_sha256=document_sha(document),
            allow_pilot=True,
        )
        self.assertFalse(shard["paper_metrics_eligible"])
        self.assertTrue(all(
            not config["paper_metrics_eligible"] for config in shard["configurations"]
        ))

    def test_calibration_cost_is_shard_level_and_not_compression_time(self):
        document = formal_document(config_id="phog")
        shard = summary.summarize_document(
            "phog.json", document, result_file_sha256=document_sha(document),
        )
        calibration = shard["calibration"]
        self.assertTrue(calibration["required"])
        self.assertEqual(2, calibration["external_wall_time_ns"]["n"])
        self.assertFalse(calibration["compression_time_component"])
        configuration = shard["configurations"][0]
        self.assertTrue(configuration["calibration_reference"]["required_for_configuration"])
        self.assertFalse(
            configuration["calibration_reference"]["cost_included_in_compression_time"]
        )

    def test_failed_document_is_rejected(self):
        document = formal_document()
        document.update({"status": "complete", "success": False, "failure": "fixture failure"})
        with self.assertRaises(summary.SummaryError) as caught:
            summary.summarize_document(
                "failed.json", document, result_file_sha256=document_sha(document),
            )
        self.assertEqual("unsuccessful_system_benchmark", caught.exception.code)

    def test_incomplete_repetitions_are_rejected(self):
        document = formal_document()
        document["configurations"][0]["runs"].pop()
        with self.assertRaises(summary.SummaryError):
            summary.summarize_document(
                "incomplete.json", document, result_file_sha256=document_sha(document),
            )

    def test_archive_size_drift_is_rejected(self):
        document = formal_document()
        run = document["configurations"][0]["runs"][1]
        run["archive"]["storage"]["logical_size_bytes"] += 1
        run["compression"]["output_size_bytes"] += 1
        with self.assertRaises(summary.SummaryError):
            summary.summarize_document(
                "drift.json", document, result_file_sha256=document_sha(document),
            )

    def test_missing_timing_field_is_rejected(self):
        document = formal_document()
        del document["configurations"][0]["runs"][0]["compression"]["wall_time_ns"]
        with self.assertRaises(summary.SummaryError) as caught:
            summary.summarize_document(
                "missing.json", document, result_file_sha256=document_sha(document),
            )
        self.assertEqual("missing_or_invalid_field", caught.exception.code)

    def test_multiple_files_remain_separate_shards_with_exact_file_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            paths = []
            expected_hashes = []
            for index in range(2):
                document = copy.deepcopy(formal_document())
                document["input_metadata"]["model_tag"] = f"fixture-{index}"
                document["integrity"]["manifest_binding"]["tag"] = f"fixture-{index}"
                raw = (json.dumps(document, indent=index + 1, sort_keys=True) + "\n").encode()
                path = root / f"result-{index}.json"
                path.write_bytes(raw)
                paths.append(path)
                expected_hashes.append(hashlib.sha256(raw).hexdigest())

            report = summary.summarize_files(paths)
            self.assertEqual(2, report["input_file_count"])
            self.assertFalse(report["cross_file_aggregation_performed"])
            self.assertEqual(
                expected_hashes,
                [shard["result_file"]["sha256"] for shard in report["shards"]],
            )
            self.assertTrue(all(
                len(shard["configurations"]) == 1 for shard in report["shards"]
            ))


if __name__ == "__main__":
    unittest.main()
