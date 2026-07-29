import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_throughput_paper_table as paper_table


class ThroughputPaperTableTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_environment(
        self,
        directory,
        *,
        experiment_id,
        methods,
        shard_jobs,
        zipnn_jobs=16,
    ):
        resources = {}
        for method in methods:
            jobs = zipnn_jobs if method == "zipnn" else shard_jobs
            processes_per_command = 2 if method == "libdeflate-1" else 1
            resources[f"practical/{method}"] = {
                "method": method,
                "profile": "practical",
                "execution_model": (
                    "one_worker_brevis_process_pool_across_shards"
                    if method == "brevis"
                    else "one_thread_process_pool_across_shards"
                ),
                "declared_cpu_slots": jobs,
                "shard_jobs": jobs,
                "processes_per_command": processes_per_command,
                "cpu_slots_per_command": 1,
                "codec_workers": 1,
                "expected_threads_per_process": 1,
                "cpu_affinity": f"0-{jobs - 1}",
                "environment": {},
                "runtime_controls": {},
                "note": "fixture",
            }
        environment = {
            "experiment_id": experiment_id,
            "protocol": {
                "methods": list(methods),
                "profiles": ["practical"],
                "cache_mode": "hot",
                "repetitions": 1,
            },
            "resource_matrix": resources,
            "checkpoints": [
                {
                    "name": "llama-fixture",
                    "files": [
                        {"relative_path": f"shard-{index:02d}"}
                        for index in range(30)
                    ],
                }
            ],
        }
        directory.mkdir(parents=True)
        (directory / "throughput-environment.json").write_text(
            json.dumps(environment)
        )

    def phase_records(
        self,
        *,
        experiment_id,
        method,
        strict,
        compress_seconds=2.0,
        decompress_seconds=1.0,
    ):
        identity = {
            "experiment_id": experiment_id,
            "checkpoint": "llama-fixture",
            "profile": "practical",
            "method": method,
            "sample_kind": "measured",
            "sample_index": 0,
            "attempt_id": f"attempt-{method}",
            "pair_id": f"pair-{method}",
        }
        records = [
            {
                **identity,
                "operation": "pair",
                "status": "started",
            }
        ]
        for operation, seconds, output_bytes in (
            ("compress", compress_seconds, 1_000_000_000),
            ("decompress", decompress_seconds, 2_000_000_000),
        ):
            record = {
                **identity,
                "operation": operation,
                "status": "ok",
                "logical_uncompressed_bytes": 2_000_000_000,
                "output_bytes": output_bytes,
                "phase_makespan_seconds": seconds,
                "round_trip_exact": True if strict else None,
                "exactness_scope": (
                    "file_byte_exact" if strict else "not_checked"
                ),
            }
            if not strict:
                record["round_trip_check_performed"] = False
            records.append(record)
        return records

    def append_records(self, directory, records):
        raw = directory / "raw"
        raw.mkdir(exist_ok=True)
        path = raw / "throughput-runs.jsonl"
        with path.open("a", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record) + "\n")

    def test_builds_formats_marks_missing_and_is_rerunnable(self):
        brevis = self.root / "brevis"
        zstd = self.root / "zstd"
        fast = self.root / "fast"
        output = self.root / "paper"
        self.write_environment(
            brevis,
            experiment_id="brevis-experiment",
            methods=("brevis",),
            shard_jobs=32,
        )
        self.write_environment(
            zstd,
            experiment_id="zstd-experiment",
            methods=("zstd-9",),
            shard_jobs=32,
        )
        self.write_environment(
            fast,
            experiment_id="fast-experiment",
            methods=("zipnn", "lz4-hc-9", "libdeflate-1", "snappy"),
            shard_jobs=32,
        )
        self.append_records(
            brevis,
            self.phase_records(
                experiment_id="brevis-experiment",
                method="brevis",
                strict=True,
            ),
        )
        self.append_records(
            fast,
            self.phase_records(
                experiment_id="fast-experiment",
                method="zipnn",
                strict=False,
            ),
        )
        inputs = (
            paper_table.InputSpec(
                "brevis_strict",
                brevis,
                ("brevis",),
                True,
            ),
            paper_table.InputSpec(
                "zstd_strict",
                zstd,
                ("zstd-9",),
                True,
            ),
            paper_table.InputSpec(
                "fast_baselines",
                fast,
                ("zipnn", "lz4-hc-9", "libdeflate-1", "snappy"),
                False,
            ),
        )

        rows = paper_table.build_paper_table(inputs, output)

        self.assertEqual(6, len(rows))
        self.assertEqual(2, sum(row["status"] == "complete" for row in rows))
        brevis_row = next(row for row in rows if row["method"] == "brevis")
        self.assertEqual("legacy_strict_log_with_exact_true", brevis_row[
            "round_trip_provenance"
        ])
        self.assertEqual("exact", brevis_row["round_trip_check_status"])
        self.assertEqual(1.0, brevis_row["compress_gb_per_second"])
        self.assertAlmostEqual(
            1_000_000_000 / 1024**3,
            brevis_row["compress_gib_per_second"],
        )
        self.assertEqual(2.0, brevis_row["compression_ratio_x"])
        self.assertIn("peak-throughput estimate", brevis_row[
            "measurement_label"
        ])

        zipnn_row = next(row for row in rows if row["method"] == "zipnn")
        self.assertEqual("not_checked", zipnn_row["round_trip_check_status"])
        self.assertFalse(zipnn_row["round_trip_check_performed"])
        self.assertEqual(16, zipnn_row["effective_shard_jobs"])
        self.assertEqual(
            "16 shard processes × 1 codec thread",
            zipnn_row["process_thread_config"],
        )
        zstd_row = next(row for row in rows if row["method"] == "zstd-9")
        self.assertEqual("missing", zstd_row["status"])
        self.assertIn("not started", zstd_row["missing_reason"])
        self.assertIsNone(zstd_row["compress_seconds"])

        csv_path = output / "throughput-paper-table.csv"
        markdown = (output / "throughput-paper-table.md").read_text()
        latex = (output / "throughput-paper-table.tex").read_text()
        self.assertTrue(csv_path.is_file())
        self.assertIn("## Missing or incomplete", markdown)
        self.assertIn("Zstd-9: repetition 0: not started", markdown)
        self.assertIn("not_checked", markdown)
        self.assertIn(r"$\times$", latex)
        self.assertIn("single-run peak-throughput estimate", latex)

        self.append_records(
            zstd,
            self.phase_records(
                experiment_id="zstd-experiment",
                method="zstd-9",
                strict=True,
                compress_seconds=4.0,
                decompress_seconds=2.0,
            ),
        )
        rerun_rows = paper_table.build_paper_table(inputs, output)
        rerun_zstd = next(
            row for row in rerun_rows if row["method"] == "zstd-9"
        )
        self.assertEqual("complete", rerun_zstd["status"])
        self.assertEqual(4.0, rerun_zstd["compress_seconds"])
        with csv_path.open(newline="", encoding="utf-8") as input_file:
            csv_rows = list(csv.DictReader(input_file))
        self.assertEqual(6, len(csv_rows))
        self.assertEqual(
            3,
            sum(row["status"] == "complete" for row in csv_rows),
        )


if __name__ == "__main__":
    unittest.main()
