import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_verified_results as analysis


def summary(*, complete: bool = True) -> dict:
    methods = ["brevis", "zstd-9", "zipnn"]
    outputs = {
        ("bert-fp32", "brevis"): 50,
        ("bert-fp32", "zstd-9"): 100,
        ("bert-fp32", "zipnn"): 40,
        ("whisper-large-v3-f16", "brevis"): 80,
        ("whisper-large-v3-f16", "zstd-9"): 100,
        ("whisper-large-v3-f16", "zipnn"): 80,
    }
    cells = []
    for (checkpoint, method), output_bytes in outputs.items():
        cells.append(
            {
                "checkpoint": checkpoint,
                "method": method,
                "exactness": "tensor-exact" if method == "zipnn" else "byte-exact",
                "status": "complete",
                "source_bytes": 200,
                "output_bytes": output_bytes,
                "compression_ratio": 200 / output_bytes,
            }
        )
    return {
        "schema_version": 1,
        "all_complete": complete,
        "methods": methods,
        "cells": cells,
    }


class AnalyzeVerifiedResultsTests(unittest.TestCase):
    def write_summary(self, root: Path, payload: dict) -> Path:
        path = root / "verified-ratios.json"
        path.write_text(json.dumps(payload))
        return path

    def test_pairwise_metrics_and_counterexamples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = analysis.analyze(
                self.write_summary(root, summary()),
                root / "analysis",
                bootstrap_samples=500,
                bootstrap_seed=1234,
            )

            overall = {
                item["baseline"]: item for item in report["overall"]
            }
            zstd = overall["zstd-9"]
            self.assertEqual((2, 0, 0), (zstd["wins"], zstd["ties"], zstd["losses"]))
            self.assertAlmostEqual(math.sqrt(2.0 * 1.25), zstd["geomean_compression_advantage"])
            self.assertAlmostEqual(0.35, zstd["macro_archive_saving_fraction"])
            self.assertAlmostEqual(0.35, zstd["pooled_archive_saving_fraction"])
            self.assertAlmostEqual(
                400 / 130,
                zstd["brevis_total_byte_compression_ratio"],
            )
            self.assertAlmostEqual(
                2.0,
                zstd["baseline_total_byte_compression_ratio"],
            )
            self.assertAlmostEqual(
                200 / 130,
                zstd["total_byte_compression_advantage"],
            )
            self.assertLessEqual(
                zstd["geomean_compression_advantage_ci_low"],
                zstd["geomean_compression_advantage"],
            )
            self.assertGreaterEqual(
                zstd["geomean_compression_advantage_ci_high"],
                zstd["geomean_compression_advantage"],
            )

            zipnn = overall["zipnn"]
            self.assertEqual((0, 1, 1), (zipnn["wins"], zipnn["ties"], zipnn["losses"]))
            self.assertEqual(2, len(report["counterexamples"]))
            self.assertEqual(1234, report["bootstrap"]["seed"])
            self.assertEqual(500, report["bootstrap"]["samples"])
            self.assertTrue((root / "analysis" / "paper-analysis.md").is_file())
            self.assertTrue((root / "analysis" / "pairwise-by-domain.csv").is_file())
            latex = (root / "analysis" / "paper-tables.tex").read_text()
            self.assertIn(r"\times", latex)
            self.assertIn(r"95\% CI", latex)
            self.assertIn("seed 1234", latex)

    def test_bootstrap_is_reproducible_and_checkpoint_resampled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_summary(Path(directory), summary())
            cells, methods, _ = analysis.load_cells(path)
            rows = analysis.make_pairwise_rows(cells, methods)
            zstd_rows = [row for row in rows if row.baseline == "zstd-9"]

            first = analysis.bootstrap_intervals(
                zstd_rows,
                samples=301,
                seed=99,
                group="all",
                baseline="zstd-9",
            )
            second = analysis.bootstrap_intervals(
                zstd_rows,
                samples=301,
                seed=99,
                group="all",
                baseline="zstd-9",
            )

            self.assertEqual(first, second)
            self.assertEqual(
                {
                    "win_rate",
                    "geomean_compression_advantage",
                    "total_byte_compression_advantage",
                    "macro_archive_saving_fraction",
                    "pooled_archive_saving_fraction",
                },
                set(first),
            )

    def test_single_model_bootstrap_interval_collapses_to_point(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_summary(Path(directory), summary())
            cells, methods, _ = analysis.load_cells(path)
            row = next(
                row
                for row in analysis.make_pairwise_rows(cells, methods)
                if row.checkpoint == "bert-fp32" and row.baseline == "zstd-9"
            )
            intervals = analysis.bootstrap_intervals(
                [row],
                samples=17,
                seed=1,
                group="FP32",
                baseline="zstd-9",
            )

            for low, high in intervals.values():
                self.assertEqual(low, high)

    def test_incomplete_summary_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_summary(Path(directory), summary(complete=False))
            with self.assertRaisesRegex(analysis.AnalysisError, "incomplete"):
                analysis.load_cells(path)

    def test_inconsistent_ratio_is_rejected(self):
        payload = summary()
        payload["cells"][0]["compression_ratio"] = 123.0
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_summary(Path(directory), payload)
            with self.assertRaisesRegex(analysis.AnalysisError, "inconsistent"):
                analysis.load_cells(path)

    def test_unknown_taxonomy_is_rejected(self):
        payload = summary()
        for cell in payload["cells"]:
            if cell["checkpoint"] == "bert-fp32":
                cell["checkpoint"] = "unknown-model"
        with tempfile.TemporaryDirectory() as directory:
            path = self.write_summary(Path(directory), payload)
            with self.assertRaisesRegex(analysis.AnalysisError, "taxonomy"):
                analysis.load_cells(path)


if __name__ == "__main__":
    unittest.main()
