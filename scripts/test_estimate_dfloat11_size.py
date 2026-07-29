#!/usr/bin/env python3
"""Synthetic tests for the pinned DFloat11 size estimator."""

from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import estimate_dfloat11_size as estimator


def write_safetensors(
    path: Path,
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    descriptors = [
        estimator.TensorDescriptor(name, dtype, shape, len(payload))
        for name, (dtype, shape, payload) in tensors.items()
    ]
    ordered = sorted(
        descriptors,
        key=lambda item: (
            -estimator.DTYPE_BYTES[item.dtype],
            item.name,
        ),
    )
    header: dict[str, object] = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    offset = 0
    payloads: list[bytes] = []
    for descriptor in ordered:
        payload = tensors[descriptor.name][2]
        header[descriptor.name] = {
            "dtype": descriptor.dtype,
            "shape": list(descriptor.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
        payloads.append(payload)
    raw = json.dumps(header, separators=(",", ":")).encode()
    padded = raw + b" " * ((-len(raw)) % 8)
    path.write_bytes(struct.pack("<Q", len(padded)) + padded + b"".join(payloads))


def bf16_payload(exponents: list[int]) -> bytes:
    values = np.asarray(
        [
            ((index & 1) << 15) | ((exponent & 0xFF) << 7) | (index & 0x7F)
            for index, exponent in enumerate(exponents)
        ],
        dtype="<u2",
    )
    return values.tobytes()


class DFloat11EstimatorTests(unittest.TestCase):
    def test_uniform_huffman_matches_official_golden(self) -> None:
        histogram = np.zeros(256, dtype=np.uint64)
        histogram[:8] = 10
        table, adjusted = estimator.official_32bit_huffman(histogram)
        integer_table = {
            symbol: code
            for symbol, code in table.items()
            if isinstance(symbol, int)
        }
        self.assertEqual((), adjusted)
        self.assertEqual(
            {
                0: (4, 15),
                1: (3, 0),
                2: (3, 1),
                3: (3, 2),
                4: (3, 3),
                5: (3, 4),
                6: (3, 5),
                7: (3, 6),
            },
            integer_table,
        )
        self.assertEqual(2, estimator.official_lut_rows(table))

    def test_32bit_limit_matches_official_fibonacci_golden(self) -> None:
        fibonacci = [1, 1]
        for _ in range(2, 40):
            fibonacci.append(fibonacci[-1] + fibonacci[-2])
        histogram = np.zeros(256, dtype=np.uint64)
        histogram[:40] = fibonacci
        table, adjusted = estimator.official_32bit_huffman(histogram)
        lengths = {
            symbol: bits
            for symbol, (bits, _) in table.items()
            if isinstance(symbol, int)
        }
        self.assertEqual(tuple(range(2, 12)), adjusted)
        self.assertEqual(32, max(bits for bits, _ in table.values()))
        self.assertEqual(
            [32] * 9 + [31] * 3 + list(range(28, 0, -1)),
            [lengths[index] for index in range(40)],
        )
        self.assertEqual(5, estimator.official_lut_rows(table))

    def test_safetensors_layout_is_byte_exact(self) -> None:
        descriptors = [
            estimator.TensorDescriptor("z_u8", "U8", (3,), 3),
            estimator.TensorDescriptor("b_bf16", "BF16", (3,), 6),
            estimator.TensorDescriptor("a_u8", "U8", (4,), 4),
            estimator.TensorDescriptor("x_i64", "I64", (1,), 8),
        ]
        layout = estimator.safetensors_file_layout(
            descriptors,
            metadata={"format": "pt"},
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny.safetensors"
            write_safetensors(
                path,
                {
                    item.name: (item.dtype, item.shape, b"\0" * item.nbytes)
                    for item in descriptors
                },
                metadata={"format": "pt"},
            )
            self.assertEqual(path.stat().st_size, layout["file_bytes"])

    def test_group_component_accounting_matches_official_golden(self) -> None:
        histogram = np.zeros(256, dtype=np.uint64)
        histogram[:8] = 10
        location = estimator.TensorLocation(
            name="model.embed_tokens.weight",
            dtype="BF16",
            shape=(80,),
            nbytes=160,
            path=Path("/not-read"),
            absolute_data_offset=0,
        )
        group = estimator.CompressionGroup(
            module_name="model.embed_tokens",
            output_filename="model_embed_tokens.safetensors",
            target_names=(location.name,),
            residual_names=(),
        )
        result = estimator.estimate_group(
            group,
            histogram,
            {location.name: location},
        )
        components = result["components"]
        # Official table: exponent 0 uses four bits and 1..7 use three bits.
        self.assertEqual(250, components["encoded_exponent_bits"])
        self.assertEqual(32, components["encoded_exponent_bytes"])
        self.assertEqual(80, components["sign_mantissa_bytes"])
        self.assertEqual(8, components["output_positions_bytes"])
        self.assertEqual(320, components["gaps_bytes"])
        self.assertEqual(0, components["split_positions_bytes"])
        self.assertEqual(512, components["luts_bytes"])

    def test_vectorized_histogram_reads_only_tensor_slice(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tiny.safetensors"
            exponents = [0, 1, 1, 127, 128, 128, 128, 255]
            write_safetensors(
                path,
                {
                    "prefix": ("U8", (5,), b"abcde"),
                    "weight": (
                        "BF16",
                        (len(exponents),),
                        bf16_payload(exponents),
                    ),
                },
            )
            data_base, header = estimator._read_safetensors_header(path)
            start, end = header["weight"]["data_offsets"]
            location = estimator.TensorLocation(
                "weight",
                "BF16",
                (len(exponents),),
                end - start,
                path,
                data_base + start,
            )
            counts = estimator.scan_bf16_exponent_histogram(
                location,
                chunk_elements=3,
            )
            self.assertEqual(len(exponents), int(counts.sum()))
            self.assertEqual(1, int(counts[0]))
            self.assertEqual(2, int(counts[1]))
            self.assertEqual(1, int(counts[127]))
            self.assertEqual(3, int(counts[128]))
            self.assertEqual(1, int(counts[255]))

    def test_synthetic_llm_end_to_end_and_actual_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = Path(temporary)
            config = {
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 4,
                "num_attention_heads": 2,
                "num_hidden_layers": 1,
                "tie_word_embeddings": False,
                "torch_dtype": "bfloat16",
                "vocab_size": 8,
            }
            (model_dir / "config.json").write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            (model_dir / "generation_config.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            exponent_cycle = list(range(8))
            tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {
                "model.embed_tokens.weight": (
                    "BF16",
                    (8, 4),
                    bf16_payload(exponent_cycle * 4),
                ),
                "model.layers.0.input_layernorm.weight": (
                    "BF16",
                    (4,),
                    bf16_payload([1, 2, 3, 4]),
                ),
                "model.layers.0.post_attention_layernorm.weight": (
                    "BF16",
                    (4,),
                    bf16_payload([4, 3, 2, 1]),
                ),
                "model.norm.weight": (
                    "BF16",
                    (4,),
                    bf16_payload([1, 1, 1, 1]),
                ),
                "lm_head.weight": (
                    "BF16",
                    (8, 4),
                    bf16_payload(exponent_cycle * 4),
                ),
            }
            for suffix in estimator.LLM_LINEAR_PATHS:
                name = f"model.layers.0.{suffix}.weight"
                tensors[name] = (
                    "BF16",
                    (4, 4),
                    bf16_payload(exponent_cycle * 2),
                )
            shard = model_dir / "model.safetensors"
            write_safetensors(shard, tensors, metadata={"format": "pt"})
            catalog, _ = estimator.load_tensor_catalog(model_dir)
            groups, root = estimator.build_llm_groups(catalog, config)
            self.assertEqual(("model.norm.weight",), root)
            histograms = estimator.scan_group_histograms(
                groups,
                catalog,
                jobs=2,
                chunk_elements=5,
                progress=False,
            )
            first = estimator.estimate_model(
                model_dir,
                histograms,
                ancillary_uncertainty_bytes=4096,
            )
            self.assertEqual(3, len(first["groups"]))
            self.assertEqual(
                shard.stat().st_size,
                first["source"]["checkpoint_shard_bytes"],
            )
            self.assertLess(
                first["output_bytes_lower"],
                first["output_bytes_upper"],
            )
            self.assertEqual(
                first["estimated_structural_safetensors_bytes"],
                sum(
                    first["component_totals"][name]
                    for name in (
                        "encoded_exponent_bytes",
                        "sign_mantissa_bytes",
                        "output_positions_bytes",
                        "gaps_bytes",
                        "split_positions_bytes",
                        "luts_bytes",
                        "residual_tensor_bytes",
                        "safetensors_prefix_and_header_bytes",
                    )
                ),
            )
            actual = first["estimated_output_bytes"] + 7
            validated = estimator.estimate_model(
                model_dir,
                histograms,
                ancillary_uncertainty_bytes=4096,
                actual_output_bytes=actual,
            )
            self.assertEqual(7, validated["validation"]["estimate_residual_bytes"])
            self.assertTrue(
                validated["validation"]["actual_within_reported_interval"]
            )


if __name__ == "__main__":
    unittest.main()
