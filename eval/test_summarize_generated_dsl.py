#!/usr/bin/env python3

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

import analyze_generated_dsl as analyzer
import summarize_generated_dsl as aggregate
import test_analyze_generated_dsl as fixture


def make_analysis(directory: pathlib.Path, *, config_id: str = "uniform") -> None:
    bundle = analyzer.analyze_documents([
        (f"fixture-{config_id}.json", fixture.system_document(config_id=config_id)),
    ])
    if bundle.errors:
        raise AssertionError(bundle.errors)
    analyzer.write_outputs(bundle, directory)


def rewrite_logical_file(
    directory: pathlib.Path, logical_name: str, records: list[dict],
    *, update_summary: bool = False,
) -> None:
    payload = analyzer.render_jsonl(records).encode("utf-8")
    path = directory / logical_name
    path.write_bytes(payload)
    compressed = directory / f"{logical_name}.zst"
    if compressed.exists():
        compressed.unlink()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"][logical_name] = {
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if update_summary:
        manifest["summary"][aggregate.SUMMARY_COUNT_FIELDS[logical_name]] = len(records)
        if logical_name == "reports.jsonl":
            manifest["summary"]["paper_metrics_eligible_reports"] = sum(
                record.get("paper_metrics_eligible") is True for record in records
            )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def load_jsonl(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


class GeneratedDslAggregateTests(unittest.TestCase):
    def test_plain_single_directory_reports_weighted_program_and_search_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            analysis_dir = pathlib.Path(temporary) / "dsl"
            make_analysis(analysis_dir)
            result = aggregate.aggregate_directories([analysis_dir])

        self.assertEqual(aggregate.OUTPUT_SCHEMA, result["schema"])
        self.assertEqual(
            {"reports": 1, "tensors": 1, "blocks": 1, "nodes": 6, "errors": 0},
            result["traceability"]["record_counts"],
        )
        realized = result["pooled_program_observations"]["realized_block_programs"]
        node_histogram = realized["length_and_depth"]["node_count"]["program_count_weight"]
        self.assertEqual(3.0, node_histogram["weighted_mean"])
        self.assertEqual([{"value": 3, "weight": 1.0, "fraction": 1.0}], node_histogram["histogram"])
        bytecode = realized["serialized_program_bytecode_length"]
        self.assertEqual("available_for_realized_block_programs", bytecode["status"])
        self.assertEqual(
            [{"value": 10, "weight": 1.0, "fraction": 1.0}],
            bytecode["distribution"]["program_count_weight"]["histogram"],
        )
        self.assertEqual(10.0, bytecode["accounting"]["program_bytecode_bytes_total"])
        self.assertAlmostEqual(
            10.0 / 46.0, bytecode["accounting"]["fraction_of_framed_bytes"],
        )
        tensor_plan_bytecode = result["pooled_program_observations"][
            "tensor_plan_programs"
        ]["serialized_program_bytecode_length"]
        self.assertEqual("not_applicable_to_tensor_plan_templates", tensor_plan_bytecode["status"])
        self.assertIsNone(tensor_plan_bytecode["distribution"])
        structure = realized["structure_frequency"]["entries"]
        self.assertEqual("split_field(raw,huffman)", structure[0]["structure_signature"])
        terminals = {
            row["terminal_codec"]
            for row in realized["terminal_codec_presence_frequency"]["entries"]
        }
        self.assertEqual({"raw", "huffman"}, terminals)
        relationship = result["pooled_program_observations"]["search_cost_relationships"]["search_expansions"]
        self.assertEqual("insufficient_observations", relationship["status"])
        self.assertEqual(1, relationship["observed_tensor_records"])
        self.assertEqual(
            "unsupported_by_generated_dsl_analysis_schema_1",
            result["unsupported_or_unavailable"]["planning_wall_time"]["status"],
        )
        dtype_rows = [
            row for row in result["stratified_preferences"]
            if row["dimension"] == "dtype" and row["value"] == "F32"
        ]
        self.assertEqual(1, len(dtype_rows))
        self.assertEqual(8.0, dtype_rows[0]["tensor_observations"]["raw_bytes"])

    def test_formal_result_directory_resolves_dsl_and_hashes_adjacent_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            result_dir = pathlib.Path(temporary) / "model.safetensors"
            make_analysis(result_dir / "dsl")
            expected = {}
            for name, payload in (
                ("result.json", b"{}\n"),
                ("summary.json", b'{"summary":true}\n'),
                ("artifact-manifest.json", b'{"artifacts":[]}\n'),
            ):
                (result_dir / name).write_bytes(payload)
                expected[name] = hashlib.sha256(payload).hexdigest()
            output = aggregate.aggregate_directories([result_dir])

        trace = output["traceability"]["inputs"][0]
        self.assertEqual(str(result_dir.resolve()), trace["resolved_result_directory"])
        self.assertEqual(
            expected,
            {
                name: metadata["sha256"]
                for name, metadata in trace["adjacent_formal_result_files"].items()
            },
        )

    def test_zstd_files_are_streamed_and_verified_against_logical_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            analysis_dir = pathlib.Path(temporary) / "dsl"
            make_analysis(analysis_dir)
            for logical_name in ("tensors.jsonl", "blocks.jsonl", "nodes.jsonl"):
                plain = analysis_dir / logical_name
                compressed = analysis_dir / f"{logical_name}.zst"
                subprocess.run(
                    ["zstd", "-q", "-f", str(plain), "-o", str(compressed)],
                    check=True,
                )
                plain.unlink()
            result = aggregate.aggregate_directories([analysis_dir])

        files = result["traceability"]["inputs"][0]["logical_files"]
        for logical_name in ("tensors.jsonl", "blocks.jsonl", "nodes.jsonl"):
            self.assertEqual("zstd", files[logical_name]["storage"])
            self.assertTrue(files[logical_name]["identity_verified_against_manifest"])
            self.assertNotEqual(
                files[logical_name]["stored_sha256"], files[logical_name]["logical_sha256"],
            )

    def test_multiple_directories_keep_search_inapplicability_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            uniform = root / "uniform"
            fixed = root / "fixed"
            make_analysis(uniform, config_id="uniform")
            make_analysis(fixed, config_id="fixed")
            result = aggregate.aggregate_directories([uniform, fixed])

        self.assertEqual(2, result["scope"]["input_count"])
        self.assertEqual(2, result["traceability"]["record_counts"]["reports"])
        candidates = result["pooled_program_observations"]["search_cost_relationships"]["candidates_realized"]
        self.assertEqual(1, candidates["observed_tensor_records"])
        self.assertEqual(1, candidates["inapplicable_tensor_records"])
        self.assertEqual(0, candidates["missing_required_tensor_records"])

    def test_duplicate_report_across_distinct_directories_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            first = root / "first"
            second = root / "second"
            make_analysis(first)
            shutil.copytree(first, second)
            with self.assertRaises(aggregate.AggregateError) as caught:
                aggregate.aggregate_directories([first, second])
        self.assertEqual("duplicate_report_id", caught.exception.code)

    def test_logical_hash_mismatch_is_rejected_after_stream_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary) / "dsl"
            make_analysis(directory)
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["files"]["tensors.jsonl"]["sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            with self.assertRaises(aggregate.AggregateError) as caught:
                aggregate.aggregate_directories([directory])
        self.assertEqual("logical_file_identity_mismatch", caught.exception.code)

    def test_manifest_record_count_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary) / "dsl"
            make_analysis(directory)
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["summary"]["tensor_records"] += 1
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            with self.assertRaises(aggregate.AggregateError) as caught:
                aggregate.aggregate_directories([directory])
        self.assertEqual("manifest_count_mismatch", caught.exception.code)

    def test_any_analysis_exclusion_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary) / "dsl"
            make_analysis(directory)
            error_record = {
                "analysis_schema": analyzer._analysis_header("error")["analysis_schema"],
                "record_type": "error",
                "error_code": "fixture_exclusion",
            }
            rewrite_logical_file(
                directory, "errors.jsonl", [error_record], update_summary=True,
            )
            with self.assertRaises(aggregate.AggregateError) as caught:
                aggregate.aggregate_directories([directory])
        self.assertEqual("analysis_exclusions_present", caught.exception.code)

    def test_unsupported_analysis_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary) / "dsl"
            make_analysis(directory)
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_bytes())
            manifest["analysis_schema"]["version"] = 2
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            with self.assertRaises(aggregate.AggregateError) as caught:
                aggregate.aggregate_directories([directory])
        self.assertEqual("unsupported_analysis_schema", caught.exception.code)

    def test_missing_applicable_search_counter_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary) / "dsl"
            make_analysis(directory, config_id="uniform")
            tensors = load_jsonl(directory / "tensors.jsonl")
            tensors[0]["search"]["candidates_realized"] = None
            rewrite_logical_file(directory, "tensors.jsonl", tensors)
            with self.assertRaises(aggregate.AggregateError) as caught:
                aggregate.aggregate_directories([directory])
        self.assertEqual("missing_or_invalid_field", caught.exception.code)

    def test_node_owner_disagreement_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = pathlib.Path(temporary) / "dsl"
            make_analysis(directory)
            nodes = load_jsonl(directory / "nodes.jsonl")
            nodes[0]["op"] = "raw" if nodes[0]["op"] != "raw" else "huffman"
            rewrite_logical_file(directory, "nodes.jsonl", nodes)
            with self.assertRaises(aggregate.AggregateError) as caught:
                aggregate.aggregate_directories([directory])
        self.assertEqual("node_owner_mismatch", caught.exception.code)

    def test_relationship_correlations_have_explicit_weighting(self):
        relation = aggregate.RelationshipAccumulator()
        relation.observe(1.0, 0.1, 10)
        relation.observe(2.0, 0.5, 100)
        relation.observe(3.0, 0.9, 1000)
        relation.mark_inapplicable()
        rendered = relation.render()
        self.assertEqual("computed", rendered["status"])
        self.assertAlmostEqual(1.0, rendered["correlations"]["pearson_tensor_count_weighted"])
        self.assertAlmostEqual(1.0, rendered["correlations"]["spearman_tensor_count_weighted"])
        self.assertEqual(1, rendered["inapplicable_tensor_records"])
        self.assertEqual(0, rendered["missing_required_tensor_records"])

    def test_cli_preserves_existing_output_unless_force_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            directory = root / "dsl"
            output = root / "aggregate.json"
            make_analysis(directory)
            output.write_text("user data\n")
            with contextlib.redirect_stderr(io.StringIO()):
                status = aggregate.main([str(directory), "--output", str(output)])
            self.assertEqual(2, status)
            self.assertEqual("user data\n", output.read_text())
            status = aggregate.main([
                str(directory), "--output", str(output), "--force",
            ])
            self.assertEqual(0, status)
            self.assertEqual(aggregate.OUTPUT_SCHEMA, json.loads(output.read_bytes())["schema"])

    def test_concurrent_no_force_writers_publish_exactly_one_complete_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "aggregate.json"
            payloads = [f'{{"writer":{index}}}\n'.encode() for index in range(8)]
            barrier = threading.Barrier(len(payloads))
            successes: list[int] = []
            failures: list[str] = []
            unexpected: list[BaseException] = []
            lock = threading.Lock()

            def writer(index: int) -> None:
                barrier.wait()
                try:
                    aggregate._atomic_write(output, payloads[index], force=False)
                except aggregate.AggregateError as exc:
                    with lock:
                        failures.append(exc.code)
                except BaseException as exc:  # pragma: no cover - diagnostic capture
                    with lock:
                        unexpected.append(exc)
                else:
                    with lock:
                        successes.append(index)

            threads = [threading.Thread(target=writer, args=(index,)) for index in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual([], unexpected)
            self.assertEqual(1, len(successes))
            self.assertEqual(["output_exists"] * 7, sorted(failures))
            self.assertEqual(payloads[successes[0]], output.read_bytes())

    def test_cli_help_documents_input_and_output_contract(self):
        script = pathlib.Path(aggregate.__file__).resolve()
        process = subprocess.run(
            [sys.executable, str(script), "--help"], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        self.assertEqual(0, process.returncode)
        self.assertIn("Inputs are never modified", process.stdout)
        self.assertIn("formal result directories", process.stdout)
        self.assertEqual("", process.stderr)


if __name__ == "__main__":
    unittest.main()
