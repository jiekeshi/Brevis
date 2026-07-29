from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_bf16_information as analysis


def words(values: list[int]) -> bytes:
    return np.asarray(values, dtype="<u2").tobytes()


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
) -> None:
    header: dict[str, object] = {"__metadata__": {"format": "pt"}}
    offset = 0
    payloads = []
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
        payloads.append(payload)
    raw = json.dumps(header, separators=(",", ":")).encode()
    padded = raw + b" " * ((-len(raw)) % 8)
    path.write_bytes(
        struct.pack("<Q", len(padded)) + padded + b"".join(payloads)
    )


class BF16InformationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_entropy_goldens(self):
        counts = np.zeros(8, dtype=np.uint64)
        counts[0] = 2
        counts[1] = 2
        self.assertAlmostEqual(1.0, analysis.empirical_entropy(counts))

        transitions = np.zeros((256, 256), dtype=np.uint64)
        transitions[0, 0] = 1
        transitions[0, 1] = 1
        transitions[1, 1] = 1
        self.assertAlmostEqual(
            2.0 / 3.0,
            analysis.empirical_conditional_entropy(transitions),
        )
        self.assertIsNone(
            analysis.empirical_conditional_entropy(
                np.zeros((256, 256), dtype=np.uint64)
            )
        )

    def test_chunk_boundary_preserves_adjacent_exponent_pairs(self):
        path = self.root / "sequence.safetensors"
        # Exponents [0, 0, 1, 1]; chunk_elements=2 forces the 0->1 pair to
        # cross a chunk boundary.
        write_safetensors(
            path,
            {"weight": ("BF16", (4,), words([0x0000, 0x0001, 0x0080, 0x0081]))},
        )
        discovery = analysis.discover_checkpoint(path)
        locations, _ = analysis.load_tensor_locations(discovery)
        result = analysis.scan_tensor(
            locations[0],
            chunk_elements=2,
            adjacent_exponent_order1=True,
        )
        self.assertEqual(4, int(result.symbol_counts.sum()))
        self.assertEqual(3, int(result.exponent_transition_counts.sum()))
        self.assertEqual(1, int(result.exponent_transition_counts[0, 0]))
        self.assertEqual(1, int(result.exponent_transition_counts[0, 1]))
        self.assertEqual(1, int(result.exponent_transition_counts[1, 1]))
        self.assertAlmostEqual(
            2.0 / 3.0,
            analysis.empirical_conditional_entropy(
                result.exponent_transition_counts
            ),
        )
        report = analysis.analyze(
            path,
            adjacent_exponent_order1=True,
            chunk_mib=1,
        )
        overall = report["groups"][0]
        self.assertAlmostEqual(
            0.5,
            overall[
                "finite_sequence_adjacent_exponent_reference_bits_per_weight"
            ],
        )
        self.assertAlmostEqual(
            8.5,
            overall[
                "idealized_raw_sign_mantissa_plus_finite_adjacent_exponent_reference_bpw"
            ],
        )

    def test_manifest_role_groups_methods_and_outputs(self):
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        shard = checkpoint / "model-00001-of-00001.safetensors"
        tensors = {
            "model.embed_tokens.weight": (
                "BF16",
                (4,),
                words([0x0000, 0x0000, 0x0080, 0x0080]),
            ),
            "model.layers.0.self_attn.q_proj.weight": (
                "BF16",
                (4,),
                words([0x0000, 0x0080, 0x0100, 0x0180]),
            ),
            "model.layers.0.mlp.down_proj.weight": (
                "BF16",
                (4,),
                words([0x3F80] * 4),
            ),
            "model.layers.0.post_attention_layernorm.weight": (
                "BF16",
                (2,),
                words([0x3F80, 0x3F80]),
            ),
            "lm_head.weight": (
                "BF16",
                (2,),
                words([0x0000, 0x8000]),
            ),
            "ignored.scale": ("U8", (3,), b"\x01\x02\x03"),
        }
        write_safetensors(shard, tensors)
        ignored = checkpoint / "duplicate-not-in-manifest.safetensors"
        write_safetensors(
            ignored,
            {"ignored.weight": ("BF16", (2,), words([0, 1]))},
        )
        manifest = {
            "name": "synthetic",
            "repo_id": "local/test",
            "revision": "fixed",
            "sha256_verified": True,
            "weights": [
                {
                    "path": shard.name,
                    "size": shard.stat().st_size,
                    "sha256": "synthetic",
                }
            ],
        }
        (checkpoint / "download-manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )

        methods = analysis.parse_method_inputs(
            ["Brevis=2.0"],
            ["DFloat11=100"],
        )
        report = analysis.analyze(
            checkpoint,
            jobs=2,
            chunk_mib=1,
            adjacent_exponent_order1=True,
            method_inputs=methods,
        )
        inventory = report["checkpoint"]["inventory"]
        self.assertEqual(5, inventory["bf16_tensor_count"])
        self.assertEqual(16, inventory["bf16_parameters"])
        self.assertEqual(1, inventory["skipped_non_bf16_tensor_count"])
        self.assertEqual([str(shard.resolve())], [
            row["path"] for row in report["checkpoint"]["shards"]
        ])

        groups = {row["group"]: row for row in report["groups"]}
        self.assertEqual(
            {"overall", "embedding", "attention", "mlp", "norm", "other"},
            set(groups),
        )
        self.assertEqual(4, groups["embedding"]["parameters"])
        self.assertEqual(4, groups["attention"]["parameters"])
        self.assertEqual(2, groups["norm"]["parameters"])
        self.assertAlmostEqual(
            1.0,
            groups["embedding"][
                "empirical_bf16_symbol_h0_bits_per_weight"
            ],
        )
        self.assertAlmostEqual(
            1.0,
            groups["embedding"][
                "empirical_exponent_h0_bits_per_exponent"
            ],
        )
        self.assertAlmostEqual(
            9.0,
            groups["embedding"][
                "idealized_raw_sign_mantissa_plus_iid_exponent_bpw"
            ],
        )
        self.assertEqual(
            sum(row["parameters"] for name, row in groups.items() if name != "overall"),
            groups["overall"]["parameters"],
        )
        self.assertEqual(5, groups["overall"]["nonempty_tensor_count"])

        method_rows = {row["method"]: row for row in report["methods"]}
        source_bytes = shard.stat().st_size
        expected_brevis_bpw = 8.0 * (source_bytes / 2.0) / 16
        self.assertAlmostEqual(
            expected_brevis_bpw,
            method_rows["Brevis"][
                "amortized_whole_output_bpw_per_analyzed_bf16_weight"
            ],
        )
        self.assertAlmostEqual(
            source_bytes / 100,
            method_rows["DFloat11"][
                "compression_ratio_source_over_output"
            ],
        )
        self.assertTrue(
            method_rows["DFloat11"][
                "whole_output_bytes_is_exact_integer_input"
            ]
        )
        self.assertEqual(
            "ratio-implied",
            method_rows["Brevis"]["whole_output_bytes_source"],
        )
        self.assertEqual(
            "explicit-size",
            method_rows["DFloat11"]["whole_output_bytes_source"],
        )

        output = self.root / "output"
        analysis.write_outputs(report, output)
        self.assertEqual(
            {
                "groups.csv",
                "information-analysis.json",
                "methods.csv",
                "report.md",
                "tensors.csv",
            },
            {path.name for path in output.iterdir()},
        )
        reloaded = json.loads(
            (output / "information-analysis.json").read_text()
        )
        self.assertEqual(16, reloaded["groups"][0]["parameters"])
        markdown = (output / "report.md").read_text()
        self.assertIn("not** an absolute information-theoretic", markdown)
        self.assertIn("Signed gaps may", markdown)

    def test_tensor_boundaries_do_not_create_transitions(self):
        path = self.root / "singletons.safetensors"
        write_safetensors(
            path,
            {
                "model.embed_tokens.weight": ("BF16", (1,), words([0])),
                "lm_head.weight": ("BF16", (1,), words([0x0080])),
            },
        )
        report = analysis.analyze(
            path,
            adjacent_exponent_order1=True,
            chunk_mib=1,
        )
        overall = report["groups"][0]
        self.assertEqual(0, overall["adjacent_exponent_transition_count"])
        self.assertIsNone(
            overall[
                "empirical_adjacent_exponent_h1_conditional_bits_per_exponent"
            ]
        )
        self.assertAlmostEqual(
            9.0,
            overall[
                "idealized_raw_sign_mantissa_plus_finite_adjacent_exponent_reference_bpw"
            ],
        )
        no_adjacency = analysis.analyze(path, chunk_mib=1)
        self.assertEqual(
            2,
            no_adjacency["groups"][0]["nonempty_tensor_count"],
        )

    def test_finite_adjacent_reference_handles_mixed_tensor_lengths(self):
        path = self.root / "mixed-lengths.safetensors"
        write_safetensors(
            path,
            {
                "model.embed_tokens.weight": (
                    "BF16",
                    (2,),
                    words([0x0000, 0x0001]),
                ),
                "lm_head.weight": ("BF16", (1,), words([0x0080])),
            },
        )
        report = analysis.analyze(
            path,
            adjacent_exponent_order1=True,
            chunk_mib=1,
            method_inputs=analysis.parse_method_inputs(["Brevis=2"], []),
        )
        overall = report["groups"][0]
        self.assertEqual(2, overall["nonempty_tensor_count"])
        self.assertEqual(1, overall["adjacent_exponent_transition_count"])
        self.assertAlmostEqual(
            2.0 / 3.0,
            overall[
                "finite_sequence_adjacent_exponent_reference_bits_per_weight"
            ],
        )
        self.assertAlmostEqual(
            8.0 + 2.0 / 3.0,
            overall[
                "idealized_raw_sign_mantissa_plus_finite_adjacent_exponent_reference_bpw"
            ],
        )
        self.assertIn(
            "signed_amortized_bpw_difference_from_finite_adjacent_exponent_reference",
            report["methods"][0],
        )
        output = self.root / "mixed-output"
        analysis.write_outputs(report, output)
        self.assertIn("finite-adj", (output / "report.md").read_text())

    def test_index_requires_exact_tensor_to_shard_mapping(self):
        checkpoint = self.root / "indexed"
        checkpoint.mkdir()
        shard = checkpoint / "model-00001-of-00001.safetensors"
        write_safetensors(
            shard,
            {
                "first.weight": ("BF16", (2,), words([0, 1])),
                "extra.weight": ("BF16", (3,), words([2, 3, 4])),
            },
        )
        index_path = checkpoint / "model.safetensors.index.json"
        index_path.write_text(
            json.dumps({"weight_map": {"first.weight": shard.name}}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            analysis.InformationAnalysisError,
            "index mismatch",
        ):
            analysis.analyze(index_path, chunk_mib=1)

        index_path.write_text(
            json.dumps(
                {
                    "weight_map": {
                        "first.weight": shard.name,
                        "extra.weight": shard.name,
                    }
                }
            ),
            encoding="utf-8",
        )
        report = analysis.analyze(index_path, chunk_mib=1)
        self.assertEqual(5, report["groups"][0]["parameters"])

    def test_method_input_validation(self):
        with self.assertRaisesRegex(
            analysis.InformationAnalysisError,
            "duplicate",
        ):
            analysis.parse_method_inputs(
                ["Brevis=1.5"],
                ["Brevis=100"],
            )
        with self.assertRaisesRegex(
            analysis.InformationAnalysisError,
            "METHOD=VALUE",
        ):
            analysis.parse_method_inputs(["missing-separator"], [])
        for invalid in ("0", "-1", "nan", "inf"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(analysis.InformationAnalysisError):
                    analysis.parse_method_inputs([f"x={invalid}"], [])
        with self.assertRaises(analysis.InformationAnalysisError):
            analysis.parse_method_inputs([], [f"x={analysis.MAX_U64 + 1}"])
        extreme = analysis.parse_method_inputs(["x=5e-324"], [])
        with self.assertRaisesRegex(
            analysis.InformationAnalysisError,
            "out of range",
        ):
            analysis.method_rows(
                extreme,
                source_file_bytes=1,
                parameters=1,
                overall={
                    "empirical_bf16_symbol_h0_bits_per_weight": 1.0,
                    "idealized_raw_sign_mantissa_plus_iid_exponent_bpw": 9.0,
                    "idealized_raw_sign_mantissa_plus_finite_adjacent_exponent_reference_bpw": None,
                },
            )


if __name__ == "__main__":
    unittest.main()
