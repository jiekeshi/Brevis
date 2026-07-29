import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_extra_paper_analysis as extra


class ExtraPaperAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw" / "runs.jsonl"
        self.raw.parent.mkdir(parents=True)
        self.attribution = self.root / "attribution"
        self.information = self.root / "information"
        self.output = self.root / "paper"
        self.records = []
        self.sequence = 0

    def tearDown(self):
        self.temp.cleanup()

    def add_run(
        self,
        *,
        stage,
        variant,
        budget,
        workers,
        output_bytes,
        wall_seconds,
        phog=True,
        astar=True,
        verified=True,
    ):
        self.sequence += 1
        run_id = f"run-{self.sequence}"
        attempt_id = f"attempt-{self.sequence}"
        compression = {
            "operation": "compress",
            "status": "ok",
            "stage": stage,
            "variant": variant,
            "max_expansions": budget,
            "workers": workers,
            "phog": phog,
            "astar_heuristic": astar,
            "calibration_tensors": 32,
            "checkpoint": "llama-fixture-shard1",
            "shard": "model-00001-of-00004.safetensors",
            "cache": "hot",
            "source_bytes": 2_000,
            "output_bytes": output_bytes,
            "wall_seconds": wall_seconds,
            "peak_rss_bytes": workers * 1024**3,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "finished_at": f"2026-01-01T00:00:{self.sequence:02d}Z",
        }
        self.records.append(compression)
        if verified:
            self.records.append(
                {
                    "operation": "verify",
                    "status": "ok",
                    "exact": True,
                    "verified_attempts": [[run_id, attempt_id]],
                }
            )
        return compression

    def write_raw(self, *, partial_tail=""):
        with self.raw.open("w", encoding="utf-8") as output:
            for record in self.records:
                output.write(json.dumps(record) + "\n")
            output.write(partial_tail)

    def write_attribution(self):
        self.attribution.mkdir()
        data = {
            "archives": [
                {
                    "archive_path": (
                        "/tmp/budget-32/model-00001."
                        "budget-32.brv"
                    )
                }
            ],
            "groups": {
                "role": [
                    {
                        "group": "attention",
                        "tensor_count": 2,
                        "source_tensor_bytes": 1_000,
                        "archive_record_bytes": 600,
                        "saved_bytes_vs_record": 400,
                        "source_over_record_ratio": 5 / 3,
                        "record_percent_of_source": 60.0,
                        "program_bytes": 590,
                        "record_framing_bytes": 10,
                        "literal_fallback_tensors": 0,
                    },
                    {
                        "group": "norm",
                        "tensor_count": 1,
                        "source_tensor_bytes": 100,
                        "archive_record_bytes": 40,
                        "saved_bytes_vs_record": 60,
                        "source_over_record_ratio": 2.5,
                        "record_percent_of_source": 40.0,
                        "program_bytes": 35,
                        "record_framing_bytes": 5,
                        "literal_fallback_tensors": 1,
                    },
                ]
            },
            "totals": {"archive_count": 1},
            "validation_scope": {
                "source_prefix_match": True,
                "program_execution": False,
            },
        }
        (self.attribution / "attribution.json").write_text(
            json.dumps(data),
            encoding="utf-8",
        )

    def write_information(self):
        self.information.mkdir()
        group_common = {
            "tensor_count": 3,
            "parameters": 1_000,
            "parameter_share_percent": 100.0,
            "nominal_bits_per_weight": 16.0,
            "distinct_bf16_symbols": 100,
            "empirical_bf16_symbol_h0_bits_per_weight": 10.5,
            "distinct_exponents": 20,
            "empirical_exponent_h0_bits_per_exponent": 2.6,
            "idealized_raw_sign_mantissa_plus_iid_exponent_bpw": 10.6,
            "empirical_adjacent_exponent_h1_conditional_bits_per_exponent": (
                2.55
            ),
            "idealized_raw_sign_mantissa_plus_finite_adjacent_exponent_reference_bpw": (
                10.55
            ),
        }
        data = {
            "checkpoint": {
                "discovery": {
                    "manifest_reported_sha256_verified": True,
                }
            },
            "groups": [
                {"group": "overall", **group_common},
                {
                    "group": "attention",
                    **{
                        **group_common,
                        "tensor_count": 2,
                        "parameters": 800,
                        "parameter_share_percent": 80.0,
                    },
                },
            ],
            "methods": [
                {
                    "method": "Brevis",
                    "source_file_bytes_basis": 2_100,
                    "whole_output_bytes": 1_300,
                    "compression_ratio_source_over_output": 2_100 / 1_300,
                    "amortized_whole_output_bpw_per_analyzed_bf16_weight": (
                        10.4
                    ),
                    "signed_amortized_bpw_difference_from_empirical_bf16_symbol_h0": (
                        -0.1
                    ),
                    "signed_amortized_bpw_difference_from_idealized_8_plus_iid_exponent_h0": (
                        -0.2
                    ),
                    "signed_amortized_bpw_difference_from_finite_adjacent_exponent_reference": (
                        -0.15
                    ),
                    "input_kind": "size_bytes",
                    "whole_output_bytes_is_exact_integer_input": True,
                },
                {
                    "method": "ZipNN",
                    "source_file_bytes_basis": 2_100,
                    "whole_output_bytes": 1_400,
                    "compression_ratio_source_over_output": 1.5,
                    "amortized_whole_output_bpw_per_analyzed_bf16_weight": (
                        11.2
                    ),
                    "signed_amortized_bpw_difference_from_empirical_bf16_symbol_h0": (
                        0.7
                    ),
                    "signed_amortized_bpw_difference_from_idealized_8_plus_iid_exponent_h0": (
                        0.6
                    ),
                    "signed_amortized_bpw_difference_from_finite_adjacent_exponent_reference": (
                        0.65
                    ),
                    "input_kind": "size_bytes",
                    "whole_output_bytes_is_exact_integer_input": True,
                },
            ],
        }
        (self.information / "information-analysis.json").write_text(
            json.dumps(data),
            encoding="utf-8",
        )

    def populate_complete_benchmarks(self):
        for budget in extra.PARETO_BUDGETS:
            self.add_run(
                stage="pareto",
                variant=f"budget-{budget}",
                budget=budget,
                workers=32,
                output_bytes=1_000 - budget,
                wall_seconds=budget + 1,
                phog=budget != 0,
            )
        for workers in extra.WORKER_COUNTS:
            self.add_run(
                stage="workers",
                variant=f"workers-{workers}",
                budget=1,
                workers=workers,
                output_bytes=990,
                wall_seconds=100 / workers,
            )
        for budget in (8, 32):
            for index, variant in enumerate(extra.ABLATION_VARIANTS):
                self.add_run(
                    stage="ablation",
                    variant=variant,
                    budget=budget,
                    workers=32,
                    output_bytes=950 + index * 10,
                    wall_seconds=20 - index,
                    phog="no-phog" not in variant,
                    astar="no-astar" not in variant,
                )

    def test_builds_all_formats_and_computes_requested_metrics(self):
        self.populate_complete_benchmarks()
        self.write_raw()
        self.write_attribution()
        self.write_information()

        tables = extra.build_extra_analysis(
            raw_path=self.raw,
            attribution_dir=self.attribution,
            information_dir=self.information,
            output_dir=self.output,
        )

        self.assertEqual(10, len(tables["pareto"]))
        budget32 = next(
            row for row in tables["pareto"] if row["budget"] == 32
        )
        self.assertAlmostEqual(2_000 / 968, budget32[
            "compression_ratio_x"
        ])
        self.assertEqual(32, budget32["saved_bytes_vs_budget0"])
        self.assertAlmostEqual(
            2_000 / 33 / 1024**2,
            budget32["throughput_mib_per_second"],
        )
        worker4 = next(
            row for row in tables["workers"] if row["workers"] == 4
        )
        self.assertEqual(4.0, worker4["speedup_vs_1_worker"])
        self.assertEqual(100.0, worker4["parallel_efficiency_percent"])

        self.assertEqual(8, len(tables["ablation"]))
        selected = [
            row for row in tables["ablation"] if row["selected_for_paper"]
        ]
        self.assertEqual(4, len(selected))
        self.assertEqual({32}, {
            row["selected_budget_for_paper"] for row in selected
        })
        no_astar = next(
            row for row in selected if row["variant"] == "no-astar"
        )
        self.assertEqual(10, no_astar["extra_output_bytes_vs_full"])
        self.assertAlmostEqual(20 / 19, no_astar["speedup_vs_full"])

        self.assertEqual(2, len(tables["attribution"]))
        self.assertEqual(
            "representative single-shard archive",
            tables["attribution"][0]["scope"],
        )
        self.assertEqual(2, len(tables["entropy"]))
        self.assertEqual(2, len(tables["methods"]))
        self.assertEqual(
            "full-model BF16 checkpoint scan",
            tables["entropy"][0]["scope"],
        )

        expected_files = {
            "pareto-budget.csv",
            "worker-scaling.csv",
            "ablation.csv",
            "attribution-roles.csv",
            "information-entropy.csv",
            "information-method-bpw.csv",
            "extra-experiments.md",
            "extra-experiments.tex",
        }
        self.assertEqual(
            expected_files,
            {path.name for path in self.output.iterdir()},
        )
        markdown = (self.output / "extra-experiments.md").read_text()
        latex = (self.output / "extra-experiments.tex").read_text()
        self.assertIn("representative Llama-3.1-8B BF16 shard", markdown)
        self.assertIn("full-model BF16 scan", markdown)
        self.assertIn("n=1", markdown)
        self.assertIn("2.000000×", markdown)
        self.assertIn(r"$\times$", latex)
        self.assertIn("budget 32", markdown)
        with (self.output / "ablation.csv").open(
            newline="",
            encoding="utf-8",
        ) as input_file:
            csv_rows = list(csv.DictReader(input_file))
        self.assertEqual(8, len(csv_rows))
        self.assertEqual(
            4,
            sum(row["selected_for_paper"] == "True" for row in csv_rows),
        )

    def test_ablation_prefers_32_only_after_its_set_is_complete(self):
        for variant in extra.ABLATION_VARIANTS:
            self.add_run(
                stage="ablation",
                variant=variant,
                budget=8,
                workers=32,
                output_bytes=1_000,
                wall_seconds=10,
            )
        self.add_run(
            stage="ablation",
            variant="full",
            budget=32,
            workers=32,
            output_bytes=990,
            wall_seconds=20,
        )
        verified = extra.exact_verified_compressions(self.records)
        rows, selected_budget = extra.build_ablation_rows(verified)
        self.assertEqual(8, selected_budget)
        self.assertEqual(
            {8},
            {
                row["search_budget"]
                for row in rows
                if row["selected_for_paper"]
            },
        )

        for variant in extra.ABLATION_VARIANTS[1:]:
            self.add_run(
                stage="ablation",
                variant=variant,
                budget=32,
                workers=32,
                output_bytes=1_000,
                wall_seconds=18,
            )
        verified = extra.exact_verified_compressions(self.records)
        rows, selected_budget = extra.build_ablation_rows(verified)
        self.assertEqual(32, selected_budget)
        self.assertEqual(
            {32},
            {
                row["search_budget"]
                for row in rows
                if row["selected_for_paper"]
            },
        )

    def test_unverified_attempt_is_excluded_and_partial_tail_is_tolerated(self):
        self.add_run(
            stage="pareto",
            variant="budget-0",
            budget=0,
            workers=32,
            output_bytes=1_000,
            wall_seconds=1,
        )
        self.add_run(
            stage="pareto",
            variant="budget-1",
            budget=1,
            workers=32,
            output_bytes=900,
            wall_seconds=2,
            verified=False,
        )
        self.write_raw(partial_tail='{"operation":"compress"')
        records = extra.read_jsonl_snapshot(self.raw)
        verified = extra.exact_verified_compressions(records)
        rows = extra.build_pareto_rows(verified)

        budget0 = next(row for row in rows if row["budget"] == 0)
        budget1 = next(row for row in rows if row["budget"] == 1)
        self.assertEqual("complete", budget0["status"])
        self.assertTrue(budget0["exact_verified"])
        self.assertEqual("missing", budget1["status"])
        self.assertEqual(0, budget1["n"])

    def test_invalid_nonfinal_jsonl_record_is_rejected(self):
        self.raw.write_text(
            '{"valid": true}\nnot-json\n{"also": "valid"}\n',
            encoding="utf-8",
        )
        with self.assertRaises(extra.ExtraAnalysisError):
            extra.read_jsonl_snapshot(self.raw)


if __name__ == "__main__":
    unittest.main()
