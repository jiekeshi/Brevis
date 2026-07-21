import copy
import json
import pathlib
import tempfile
import unittest

import analyze_generated_dsl as dsl


SOURCE_SHA256 = "a" * 64
ARCHIVE_SHA256 = "b" * 64
BYTECODE_SHA256 = "c" * 64


def terminal(op="raw", params=0):
    return {"op": op, "params_u32": params, "terminal": True, "children": []}


def split_program():
    return {
        "op": "split_field",
        "params_u32": 16,
        "terminal": False,
        "children": [terminal("raw"), terminal("huffman")],
    }


def program_fields(tree):
    info = dsl.inspect_program_tree(tree)
    return {
        "program": info.rendered_program,
        "root_operator": tree["op"],
        "program_tree": tree,
        "program_nodes": info.node_count,
        "program_depth": info.node_depth,
        "program_node_depth": info.node_depth,
        "program_transform_depth": info.transform_depth,
        "terminal_count": info.terminal_count,
    }


def bench_report(tree=None, *, config_id="uniform"):
    tree = copy.deepcopy(tree if tree is not None else split_program())
    raw = 8
    bytecode = 10
    payload = 24
    encoded = bytecode + payload
    framed = encoded + 12
    header = 8
    footer = 12
    projected = header + framed + footer
    sequence = dsl._program_sequence_sha256([bytecode], [BYTECODE_SHA256])
    is_raw = tree["op"] == "raw"
    mode = "fixed" if config_id == "fixed" else "phog" if config_id == "phog" else "uniform"
    guided = config_id == "phog"
    tensor = {
        "index": 0,
        "name": "model.layers.0.self_attn.q_proj.weight",
        "dtype": "F32",
        "shape": [2],
        "numel": 2,
        "raw_bytes": raw,
        "file_data_start_byte": 0,
        "file_data_end_byte_exclusive": raw,
        "encoded_bytes_without_frame_headers": encoded,
        "block_frame_bytes_excluding_container_header_footer": framed,
        "block_start": 0,
        "block_count": 1,
        "raw_root_blocks": int(is_raw),
        "planned_raw_root_blocks": int(is_raw),
        "fallback_raw_root_blocks": 0,
        "search_expansions": 7,
        "candidates_realized": None if config_id == "fixed" else 3,
        "candidates_reranked": None if config_id == "fixed" else 2,
        "probe_blocks_used": None if config_id == "fixed" else 1,
        "selected_sample_rank_zero_based": None if config_id == "fixed" else 0,
        **program_fields(copy.deepcopy(tree)),
    }
    block = {
        "index": 0,
        "tensor_index": 0,
        "element_offset": 0,
        "element_count": 2,
        "raw_bytes": raw,
        "encoded_bytes_without_frame_headers": encoded,
        "program_bytecode_bytes": bytecode,
        "program_bytecode_sha256": BYTECODE_SHA256,
        "packed_terminal_payload_bytes": payload,
        "frame_header_bytes": bytecode + 12,
        "framed_bytes": framed,
        "raw_classification": "planned_raw" if is_raw else None,
        **program_fields(copy.deepcopy(tree)),
    }
    enabled_ops = ["raw"] if config_id == "raw-terminal" else [
        "raw", "huffman", "split_field",
    ]
    return {
        "schema": 4,
        "kind": "brevis.bench-report",
        "input": "/immutable/model.safetensors",
        "input_size_bytes": raw,
        "input_sha256": SOURCE_SHA256,
        "timing_scope": "test fixture",
        "cache_preconditioning": "test fixture",
        "mode": mode,
        "prior": {
            "supplied": guided,
            "loaded": guided,
            "applied": guided,
            "guidance_active": guided,
            "path": "/tmp/prior.bin" if guided else None,
            "sha256": "d" * 64 if guided else None,
            "nonempty": guided,
            "context_counts_by_backoff_level": [3, 2, 1] if guided else [0, 0, 0],
        },
        "threads": 4,
        "requested_threads": 4,
        "planning_workers_used": 1,
        "encoding_workers_used": 1,
        "target_block_bytes": 262144,
        "search_options_applied": config_id != "fixed",
        "search": {
            "enabled_ops": enabled_ops,
            "max_expansions": 256,
            "max_depth": 2,
        },
        "planning_wall_ms": 11,
        "encoding_wall_ms": 3,
        "raw_bytes": raw,
        "tensor_data_bytes": raw,
        "safetensors_prefix_bytes": 0,
        "encoded_bytes_without_frame_headers": encoded,
        "block_frame_bytes_excluding_container_header_footer": framed,
        "container_header_bytes": header,
        "container_footer_bytes": footer,
        "projected_archive_bytes": projected,
        "program_bytecode_evidence": {
            "version": 1,
            "hash": "sha256",
            "sequence_spec_id": dsl.PROGRAM_SEQUENCE_SPEC_ID,
            "block_count": 1,
            "sequence_sha256": sequence,
        },
        "tensors": [tensor],
        "blocks": [block],
    }


def measured_run(index, report):
    archive_size = report["projected_archive_bytes"]
    sequence = report["program_bytecode_evidence"]["sequence_sha256"]
    return {
        "status_code": "ok",
        "phase": "measured",
        "index": index,
        "archive_pipeline_success": True,
        "diagnostic_success": True,
        "success": True,
        "failure": None,
        "archive": {
            "path": None,
            "storage": {"logical_size_bytes": archive_size, "allocated_size_bytes": 4096},
            "sha256": ARCHIVE_SHA256,
            "sha256_after_decode": ARCHIVE_SHA256,
            "unchanged_during_decode": True,
            "program_bytecode_evidence": {
                "version": 1,
                "hash": "sha256",
                "sequence_spec_id": dsl.PROGRAM_SEQUENCE_SPEC_ID,
                "archive_size_bytes": archive_size,
                "block_count": 1,
                "bytecode_lengths": [10],
                "sha256_by_block": [BYTECODE_SHA256],
                "sequence_sha256": sequence,
            },
        },
        "verification": {
            "attempted": True,
            "bit_exact": True,
            "source_sha256": SOURCE_SHA256,
            "restored_sha256": SOURCE_SHA256,
            "restored_size_bytes": 8,
            "error": None,
        },
        "bench_archive_projection_consistency": {
            "status_code": "consistent",
            "projected_archive_bytes": archive_size,
            "actual_archive_bytes": archive_size,
            "matches_actual_archive": True,
        },
        "bench_archive_program_consistency": {
            "status_code": "consistent",
            "matches": True,
            "mismatch_block_indices": [],
            "report_sequence_sha256": sequence,
            "archive_sequence_sha256": sequence,
        },
        "archive_consistency": {
            "status_code": "reference" if index == 0 else "consistent",
            "matches_reference": True,
            "reference_size_bytes": archive_size,
            "reference_sha256": ARCHIVE_SHA256,
        },
        "bench_report": report,
    }


def system_document(*, config_id="uniform", pilot=False, reports=None, include_warmup=True):
    reports = reports or [bench_report(config_id=config_id), bench_report(config_id=config_id)]
    reports = [copy.deepcopy(report) for report in reports]
    prior_policy = "canonical" if config_id == "phog" else "none"
    plan = "fixed" if config_id == "fixed" else "search"
    measured = [measured_run(index, report) for index, report in enumerate(reports)]
    warmups = []
    if include_warmup:
        warmup_report = copy.deepcopy(reports[0])
        warmup_report["planning_wall_ms"] = 101
        warmup = measured_run(0, warmup_report)
        warmup["phase"] = "warmup"
        warmup["archive_consistency"] = {
            "status_code": "not_applicable_warmup",
            "matches_reference": None,
        }
        warmups.append(warmup)
        canonical = warmup
    else:
        canonical = measured[0]
    canonical_report = canonical["bench_report"]
    semantic_sha = dsl._bench_detail_semantic_sha256(canonical_report)
    canonical_reference = {
        "scope": "same_checkpoint_configuration",
        "configuration_id": config_id,
        "phase": canonical["phase"],
        "run_index": canonical["index"],
        "report_field": "bench_report",
        "detail_fields": ["tensors", "blocks"],
        "semantic_sha256": semantic_sha,
        "resolution": "fixture checkpoint-local resolution",
    }
    technical = [*warmups, *measured]
    for run in technical:
        is_canonical = run is canonical
        run["bench_report_detail"] = {
            "status_code": "inline_canonical" if is_canonical else "canonical_reference",
            "fingerprint_spec_id": dsl.BENCH_DETAIL_FINGERPRINT_SPEC_ID,
            "hash": "sha256",
            "semantic_sha256": semantic_sha,
            "excluded_json_paths": list(dsl.BENCH_DETAIL_FINGERPRINT_EXCLUDED_JSON_PATHS),
            "detail_fields": ["tensors", "blocks"],
            "detail_fields_inline": is_canonical,
            "canonical_reference": canonical_reference,
            "matches_canonical": True,
            "eligible_for_dsl_analysis": is_canonical,
            "storage": "inline_canonical" if is_canonical else "checkpoint_local_canonical_reference",
        }
        if not is_canonical:
            del run["bench_report"]["tensors"]
            del run["bench_report"]["blocks"]
    return {
        "schema": {"id": dsl.SYSTEM_SCHEMA_ID, "version": 2},
        "status": "complete",
        "success": True,
        "failure": None,
        "run_classification": {
            "run_class": "pilot" if pilot else "formal",
            "formal_eligible": not pilot,
            "formal_ineligibility_reasons": ["test"] if pilot else [],
        },
        "source": {
            "path": "/immutable/model.safetensors",
            "size_bytes": 8,
            "sha256": SOURCE_SHA256,
        },
        "configurations": [{
            "id": config_id,
            "spec": {
                "id": config_id,
                "plan": plan,
                "prior_policy": prior_policy,
                "search_args": [],
                "notes": "fixture",
                "require_raw_only": config_id == "raw-terminal",
            },
            "effective_search_configuration": canonical_report["search"],
            "status_code": "ok",
            "failure": None,
            "warmups": warmups,
            "runs": measured,
            "measured_archive_consistency": {
                "status_code": "consistent",
                "reference_size_bytes": reports[0]["projected_archive_bytes"],
                "reference_sha256": ARCHIVE_SHA256,
                "consistent_runs": len(reports),
                "inconsistent_runs": 0,
            },
            "bench_report_detail_consistency": {
                "status_code": "consistent",
                "fingerprint_spec_id": dsl.BENCH_DETAIL_FINGERPRINT_SPEC_ID,
                "hash": "sha256",
                "canonical_json_encoding": "utf-8 sorted-key compact JSON; finite values only",
                "excluded_json_paths": list(dsl.BENCH_DETAIL_FINGERPRINT_EXCLUDED_JSON_PATHS),
                "detail_fields": ["tensors", "blocks"],
                "canonical_reference": canonical_reference,
                "successful_reports": len(technical),
                "matching_reports_including_canonical": len(technical),
                "mismatching_reports": 0,
                "failed_or_incomplete_reports_retained": 0,
                "reports_unavailable": 0,
            },
        }],
    }


class ProgramTreeTests(unittest.TestCase):
    def test_ordered_tree_walker_and_signatures(self):
        info = dsl.inspect_program_tree(split_program())
        self.assertEqual(("split_field", "raw", "huffman"), info.operators_preorder)
        self.assertEqual(("raw", "huffman"), info.terminal_codecs_preorder)
        self.assertEqual("split_field(raw,huffman)", info.structure_signature)
        self.assertEqual("split_field[16](raw[0],huffman[0])", info.parameter_signature)
        self.assertEqual([[], [0], [1]], [node["path"] for node in info.nodes])
        self.assertEqual([0, 1, 2], [node["preorder_index"] for node in info.nodes])
        self.assertEqual((3, 2, 1, 2), (
            info.node_count, info.node_depth, info.transform_depth, info.terminal_count,
        ))

    def test_unknown_operator_and_non_u32_parameter_fail_closed(self):
        with self.assertRaisesRegex(dsl.AnalysisError, "unsupported operator"):
            dsl.inspect_program_tree({
                "op": "future_op", "params_u32": 0, "terminal": False,
                "children": [terminal()],
            })
        invalid = terminal()
        invalid["params_u32"] = []
        with self.assertRaisesRegex(dsl.AnalysisError, "one u32"):
            dsl.inspect_program_tree(invalid)


class RoleRulesTests(unittest.TestCase):
    def test_role_axes_are_orthogonal_and_sha_bound(self):
        rules = dsl.load_role_rules()
        role = dsl.classify_tensor_role(
            "Model/Layers/0/Self_Attn/Q_Proj/Weight", rules,
        )
        self.assertEqual("attention", role["labels"]["component_role"])
        self.assertEqual("query", role["labels"]["projection_role"])
        self.assertEqual("weight", role["labels"]["parameter_kind"])
        self.assertEqual("shared", role["labels"]["routing_scope"])
        self.assertRegex(role["taxonomy_file_sha256"], r"^[0-9a-f]{64}$")
        router = dsl.classify_tensor_role("model.mlp.router.weight", rules)
        self.assertEqual("feed_forward", router["labels"]["component_role"])
        self.assertEqual("router", router["labels"]["routing_scope"])


class AnalysisTests(unittest.TestCase):
    def test_technical_repetitions_are_semantically_deduplicated(self):
        second = bench_report()
        second["planning_wall_ms"] = 999
        document = system_document(reports=[bench_report(), second])
        result = dsl.analyze_documents([("fixture", document)])
        self.assertFalse(result.errors)
        self.assertEqual(1, len(result.reports))
        report = result.reports[0]
        self.assertEqual([0, 1], report["technical_repetition_indices"])
        self.assertEqual(3, report["technical_repetition_count"])
        self.assertEqual(1, report["warmup_repetition_count"])
        self.assertEqual(2, report["measured_repetition_count"])
        self.assertEqual({"phase": "warmup", "index": 0}, report["canonical_detail_source"])
        self.assertFalse(report["technical_repetitions_are_independent_tensor_samples"])
        self.assertEqual(1.0, report["semantic_variant_weight"])
        self.assertEqual(1, len(result.tensors))
        self.assertEqual(1, len(result.blocks))
        self.assertEqual(6, len(result.nodes))

    def test_first_measured_report_can_be_the_single_canonical_source(self):
        result = dsl.analyze_documents([
            ("fixture", system_document(include_warmup=False)),
        ])
        self.assertFalse(result.errors)
        self.assertEqual(1, len(result.reports))
        self.assertEqual(
            {"phase": "measured", "index": 0},
            result.reports[0]["canonical_detail_source"],
        )
        self.assertEqual(1, len(result.tensors))
        self.assertEqual(1, len(result.blocks))

    def test_terminal_presence_is_not_payload_attribution(self):
        result = dsl.analyze_documents([("fixture", system_document())])
        block = result.blocks[0]
        self.assertEqual(["raw", "huffman"], block["program"]["terminal_codecs_present"])
        self.assertEqual(24, block["accounting"]["packed_terminal_payload_bytes_all_terminals"])
        terminal_nodes = [node for node in result.nodes if node["is_terminal"]]
        self.assertTrue(terminal_nodes)
        for node in terminal_nodes:
            attribution = node["terminal_payload_attribution"]
            self.assertIsNone(attribution["terminal_specific_payload_bytes"])
            self.assertIsNone(attribution["terminal_specific_payload_share"])
            self.assertFalse(attribution["terminal_presence_is_payload_share"])

    def test_raw_terminal_is_identified_by_outer_configuration(self):
        raw = terminal("raw")
        result = dsl.analyze_documents([
            ("fixture", system_document(config_id="raw-terminal", reports=[
                bench_report(raw, config_id="raw-terminal"),
            ])),
        ])
        self.assertFalse(result.errors)
        report = result.reports[0]
        self.assertEqual("raw-terminal", report["outer_configuration_id"])
        self.assertEqual("uniform", report["bench_mode"])
        self.assertEqual(
            "explicit_raw_only_contract_confirmed_by_effective_enabled_ops",
            report["raw_terminal_identity"],
        )

    def test_legacy_schema_two_raw_only_evidence_is_semantic_not_identifier_based(self):
        raw = terminal("raw")
        document = system_document(config_id="custom-raw-control", reports=[
            bench_report(raw, config_id="raw-terminal"),
        ])
        configuration = document["configurations"][0]
        configuration["spec"].pop("require_raw_only")
        result = dsl.analyze_documents([("legacy-schema2", document)])
        self.assertFalse(result.errors)
        self.assertEqual("custom-raw-control", result.reports[0]["outer_configuration_id"])
        self.assertEqual(
            "legacy_schema2_effective_enabled_ops_is_raw_only_evidence",
            result.reports[0]["raw_terminal_identity"],
        )

    def test_pilot_requires_opt_in_and_never_becomes_paper_eligible(self):
        document = system_document(pilot=True)
        rejected = dsl.analyze_documents([("pilot", document)])
        self.assertFalse(rejected.reports)
        self.assertEqual("pilot_excluded", rejected.errors[0]["error_code"])
        accepted = dsl.analyze_documents([("pilot", document)], allow_pilot=True)
        self.assertFalse(accepted.errors)
        self.assertEqual("pilot", accepted.reports[0]["run_class"])
        self.assertFalse(accepted.reports[0]["paper_metrics_eligible"])
        self.assertFalse(accepted.tensors[0]["paper_metrics_eligible"])

    def test_future_schemas_and_bad_evidence_are_excluded(self):
        future_system = system_document()
        future_system["schema"]["version"] = 1
        result = dsl.analyze_documents([("future-system", future_system)])
        self.assertFalse(result.reports)
        self.assertEqual("unsupported_system_schema", result.errors[0]["error_code"])

        future_bench = bench_report()
        future_bench["schema"] = 5
        result = dsl.analyze_documents([
            ("future-bench", system_document(reports=[future_bench])),
        ])
        self.assertFalse(result.reports)
        self.assertIn("unsupported_bench_schema", {error["error_code"] for error in result.errors})
        self.assertIn(
            "configuration_excluded_invalid_evidence",
            {error["error_code"] for error in result.errors},
        )

        mismatch = system_document()
        mismatch["configurations"][0]["runs"][0][
            "bench_archive_projection_consistency"
        ]["matches_actual_archive"] = False
        result = dsl.analyze_documents([("mismatch", mismatch)])
        self.assertFalse(result.reports)
        self.assertIn("projection_mismatch", {error["error_code"] for error in result.errors})

    def test_dangling_reference_hash_mismatch_and_duplicate_inline_are_excluded(self):
        dangling = system_document()
        configuration = dangling["configurations"][0]
        reference = configuration["bench_report_detail_consistency"]["canonical_reference"]
        reference["run_index"] = 999
        for run in [*configuration["warmups"], *configuration["runs"]]:
            run["bench_report_detail"]["canonical_reference"] = reference
        result = dsl.analyze_documents([("dangling", dangling)])
        self.assertFalse(result.reports)
        self.assertIn(
            "dangling_canonical_reference",
            {error["error_code"] for error in result.errors},
        )

        bad_hash = system_document()
        bad_hash["configurations"][0]["runs"][0]["bench_report_detail"][
            "semantic_sha256"
        ] = "e" * 64
        result = dsl.analyze_documents([("bad-hash", bad_hash)])
        self.assertFalse(result.reports)
        self.assertIn(
            "semantic_fingerprint_mismatch",
            {error["error_code"] for error in result.errors},
        )

        duplicate = system_document()
        configuration = duplicate["configurations"][0]
        canonical_report = configuration["warmups"][0]["bench_report"]
        extra = configuration["runs"][0]
        extra["bench_report"]["tensors"] = copy.deepcopy(canonical_report["tensors"])
        extra["bench_report"]["blocks"] = copy.deepcopy(canonical_report["blocks"])
        result = dsl.analyze_documents([("duplicate", duplicate)])
        self.assertFalse(result.reports)
        self.assertIn(
            "duplicate_canonical_detail",
            {error["error_code"] for error in result.errors},
        )

    def test_role_and_weight_fields_are_emitted(self):
        result = dsl.analyze_documents([("fixture", system_document())])
        tensor = result.tensors[0]
        self.assertEqual("attention", tensor["role"]["labels"]["component_role"])
        self.assertEqual("query", tensor["role"]["labels"]["projection_role"])
        self.assertEqual(8.0, tensor["weights"]["raw_byte_weight"])
        block = result.blocks[0]
        self.assertEqual(1.0, block["weights"]["tensor_equal_weight"])
        realized_root = next(
            node for node in result.nodes
            if node["tree_scope"] == "realized_block" and node["preorder_index"] == 0
        )
        self.assertEqual(8.0, realized_root["weights"]["operator_presence_raw_byte_weight"])

    def test_jsonl_and_manifest_are_canonical_and_hash_bound(self):
        result = dsl.analyze_documents([("fixture", system_document())])
        payload = dsl.render_jsonl(result.reports)
        self.assertTrue(payload.endswith("\n"))
        self.assertEqual(result.reports[0], json.loads(payload))
        with tempfile.TemporaryDirectory() as temporary:
            output = pathlib.Path(temporary) / "analysis"
            dsl.write_outputs(result, output)
            for name in ("reports.jsonl", "tensors.jsonl", "blocks.jsonl", "nodes.jsonl", "errors.jsonl"):
                self.assertTrue((output / name).is_file())
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(1, manifest["summary"]["accepted_semantic_reports"])
            self.assertEqual(
                result.role_rules["file_sha256"], manifest["role_rules"]["file_sha256"],
            )
            with self.assertRaises(FileExistsError):
                dsl.write_outputs(result, output)


if __name__ == "__main__":
    unittest.main()
