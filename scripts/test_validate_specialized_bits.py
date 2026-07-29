import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
import validate_specialized_bits as validator


CODE_BY_EXPONENT = {
    120: "0",
    121: "10",
    122: "110000000",
    123: "110000001",
    124: "111",
}
ECF8_CODE_BY_EXPONENT = {
    2: "0",
    3: "10",
    4: "110000000",
    5: "110000001",
    6: "111",
}


def pack_codes(
    exponents: np.ndarray,
    code_by_exponent: dict[int, str] = CODE_BY_EXPONENT,
) -> np.ndarray:
    encoded = "".join(code_by_exponent[int(value)] for value in exponents)
    encoded += "0" * ((8 - len(encoded) % 8) % 8)
    bits = np.fromiter((int(bit) for bit in encoded), dtype=np.uint8)
    return np.packbits(bits, bitorder="big")


def make_luts(
    code_by_exponent: dict[int, str] = CODE_BY_EXPONENT,
) -> np.ndarray:
    luts = np.zeros((3, 256), dtype=np.uint8)
    symbols = tuple(code_by_exponent)
    luts[0, :128] = symbols[0]
    luts[0, 128:192] = symbols[1]
    luts[0, 192] = 255
    luts[0, 224:] = symbols[4]
    luts[1, :128] = symbols[2]
    luts[1, 128:] = symbols[3]
    for exponent, code in code_by_exponent.items():
        luts[-1, exponent] = len(code)
    return luts


def write_fixture(root: Path) -> tuple[Path, Path]:
    source = root / "source"
    converted = root / "converted"
    source.mkdir()
    converted.mkdir()
    exponents = np.resize(
        np.array([120, 121, 122, 123, 124], dtype=np.uint16),
        200,
    )
    positions = np.arange(exponents.size, dtype=np.uint16)
    signs = (positions % 3 == 0).astype(np.uint16)
    mantissas = (positions * 13) & 0x7f
    bit_patterns = (
        (signs << 15) | (exponents << 7) | mantissas
    ).astype(np.uint16)
    weight = (
        torch.from_numpy(bit_patterns.view(np.int16).copy())
        .view(torch.bfloat16)
        .reshape(20, 10)
    )
    direct = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf")],
        dtype=torch.float32,
    )
    source_weights = source / "model.safetensors"
    save_file(
        {
            "block.a.weight": weight[:7].clone(),
            "block.b.weight": weight[7:].clone(),
            "model.norm.weight": direct,
        },
        source_weights,
    )
    (source / "download-manifest.json").write_text(
        json.dumps(
            {
                "name": "tiny-dfloat11",
                "repo_id": "fixture/tiny",
                "revision": "a" * 40,
                "sha256_verified": True,
                "weights": [
                    {
                        "path": source_weights.name,
                        "size": source_weights.stat().st_size,
                        "sha256": validator.sha256(source_weights),
                    }
                ],
            }
        )
    )

    sign_mantissa = (
        ((bit_patterns >> 8) & 0x80) | (bit_patterns & 0x7f)
    ).astype(np.uint8)
    output_positions = np.array(
        [0, bit_patterns.size],
        dtype=np.uint32,
    ).view(np.uint8)
    save_file(
        {
            "block.luts": torch.from_numpy(make_luts()),
            "block.encoded_exponent": torch.from_numpy(
                pack_codes(exponents)
            ),
            "block.sign_mantissa": torch.from_numpy(sign_mantissa),
            "block.output_positions": torch.from_numpy(
                output_positions.copy()
            ),
            "block.gaps": torch.zeros(320, dtype=torch.uint8),
            "block.split_positions": torch.tensor([70], dtype=torch.int64),
        },
        converted / "linear.safetensors",
    )
    save_file(
        {"model.norm.weight": direct.clone()},
        converted / "model.safetensors",
    )
    (converted / "config.json").write_text(
        json.dumps(
            {
                "dfloat11_config": {
                    "version": "0.5.0",
                    "threads_per_block": [512],
                    "bytes_per_thread": 8,
                    "pattern_dict": {"block": ["a", "b"]},
                }
            }
        )
    )
    return source, converted


def write_ecf8_fixture(root: Path) -> tuple[Path, Path]:
    source = root / "ecf8-source"
    converted = root / "ecf8-converted"
    source.mkdir()
    converted.mkdir()
    exponents = np.resize(
        np.array(tuple(ECF8_CODE_BY_EXPONENT), dtype=np.uint8),
        200,
    )
    positions = np.arange(exponents.size, dtype=np.uint8)
    signs = (positions % 3 == 0).astype(np.uint8) << 7
    mantissas = (positions * 5) & 0x07
    bit_patterns = (
        signs | (exponents << 3) | mantissas
    ).astype(np.uint8)
    weight = (
        torch.from_numpy(bit_patterns.copy())
        .view(torch.float8_e4m3fn)
        .reshape(20, 10)
    )
    direct = torch.tensor([1.0, -0.0, 2.0, -3.0], dtype=torch.bfloat16)
    source_weights = source / "model.safetensors"
    save_file(
        {
            "block.a.weight": weight.reshape(-1)[:71].clone(),
            "block.b.weight": weight.reshape(-1)[71:].clone(),
            "block.a.weight_scale_inv": direct,
        },
        source_weights,
    )
    (source / "download-manifest.json").write_text(
        json.dumps(
            {
                "name": "tiny-ecf8",
                "repo_id": "fixture/tiny-fp8",
                "revision": "b" * 40,
                "sha256_verified": True,
                "weights": [
                    {
                        "path": source_weights.name,
                        "size": source_weights.stat().st_size,
                        "sha256": validator.sha256(source_weights),
                    }
                ],
            }
        )
    )

    other_4bits = (
        ((bit_patterns >> 4) & 0x08) | (bit_patterns & 0x07)
    ).astype(np.uint8)
    packed_other_4bits = (
        (other_4bits[::2] << 4) | other_4bits[1::2]
    ).astype(np.uint8)
    output_positions = np.array(
        [0, bit_patterns.size],
        dtype=np.uint64,
    ).view(np.uint8)
    save_file(
        {
            "block.luts": torch.from_numpy(
                make_luts(ECF8_CODE_BY_EXPONENT)
            ),
            "block.encoded": torch.from_numpy(
                pack_codes(exponents, ECF8_CODE_BY_EXPONENT)
            ),
            "block.packed_other_4bits": torch.from_numpy(
                packed_other_4bits
            ),
            "block.output_positions": torch.from_numpy(
                output_positions.copy()
            ),
            "block.gaps": torch.zeros(256, dtype=torch.uint8),
            "block.split_positions": torch.tensor([71], dtype=torch.int64),
        },
        converted / "linear.safetensors",
    )
    save_file(
        {"block.a.weight_scale_inv": direct.clone()},
        converted / "model.safetensors",
    )
    (converted / "config.json").write_text(
        json.dumps(
            {
                "dfloat_config": {
                    "version": "0.2.0",
                    "threads_per_block": 512,
                    "bytes_per_thread": 8,
                }
            }
        )
    )
    (converted / "pattern_dict.json").write_text(
        json.dumps({"fp8": {"block": ["a", "b"]}, "bf16": {}})
    )
    return source, converted


class SpecializedBitValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source, self.converted = write_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def validate(self):
        return validator.validate_dfloat11(
            self.source,
            self.converted,
            chunk_mib=1,
            build_dir=self.root / "build",
        )

    def test_independent_decoder_matches_all_tensor_bit_patterns(self):
        stats, provenance = self.validate()

        self.assertEqual(3, stats.source_tensors)
        self.assertEqual(1, stats.decoded_groups)
        self.assertEqual(2, stats.decoded_tensors)
        self.assertEqual(1, stats.direct_tensors)
        self.assertEqual(200 * 2 + 4 * 4, stats.compared_bytes)
        self.assertEqual(validator.sha256(validator.DECODER_SOURCE), provenance[
            "decoder_source_sha256"
        ])
        self.assertEqual(
            validator.sha256(self.source / "model.safetensors"),
            provenance["source_files"][0]["sha256"],
        )
        self.assertEqual(
            validator.sha256(self.converted / "linear.safetensors"),
            next(
                item["sha256"]
                for item in provenance["converted_files"]
                if item["path"].endswith("linear.safetensors")
            ),
        )

    def test_compressed_tensor_bit_corruption_is_detected(self):
        path = self.converted / "linear.safetensors"
        tensors = load_file(path)
        tensors["block.sign_mantissa"][17] ^= 1
        save_file(tensors, path)

        with self.assertRaisesRegex(
            validator.ValidationError,
            "block.a.weight.*element 17",
        ):
            self.validate()

    def test_direct_tensor_bit_corruption_is_detected(self):
        path = self.converted / "model.safetensors"
        tensors = load_file(path)
        raw = tensors["model.norm.weight"].view(torch.uint8)
        raw[0] ^= 1
        save_file(tensors, path)

        with self.assertRaisesRegex(
            validator.ValidationError,
            "model.norm.weight.*byte 0",
        ):
            self.validate()

    def test_truncated_huffman_stream_is_detected(self):
        path = self.converted / "linear.safetensors"
        tensors = load_file(path)
        tensors["block.encoded_exponent"] = tensors[
            "block.encoded_exponent"
        ][:3]
        save_file(tensors, path)

        with self.assertRaisesRegex(
            validator.ValidationError,
            "Huffman|code length",
        ):
            self.validate()

    def test_manifest_sha256_mismatch_is_detected(self):
        path = self.source / "download-manifest.json"
        manifest = json.loads(path.read_text())
        manifest["weights"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(manifest))

        with self.assertRaisesRegex(
            validator.ValidationError,
            "SHA256",
        ):
            self.validate()

    def test_unclassified_output_tensor_is_rejected(self):
        path = self.converted / "model.safetensors"
        tensors = load_file(path)
        tensors["unexpected.tensor"] = torch.tensor([1], dtype=torch.uint8)
        save_file(tensors, path)

        with self.assertRaisesRegex(
            validator.ValidationError,
            "unclassified tensor",
        ):
            self.validate()

    def test_cli_writes_machine_readable_success_report(self):
        report = self.root / "report.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(validator.__file__)),
                "dfloat11",
                "--source",
                str(self.source),
                "--converted",
                str(self.converted),
                "--build-dir",
                str(self.root / "build"),
                "--report",
                str(report),
                "--json",
            ],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )

        rendered = json.loads(completed.stdout)
        persisted = json.loads(report.read_text())
        self.assertTrue(rendered["exact"])
        self.assertEqual("ok", persisted["status"])
        self.assertEqual(
            "manifest_declared_tensor_bit_patterns",
            persisted["exactness_scope"],
        )
        self.assertEqual(
            "structural_only",
            persisted["parallel_metadata_validation"],
        )


class ECF8BitValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source, self.converted = write_ecf8_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def validate(self):
        return validator.validate_ecf8(
            self.source,
            self.converted,
            chunk_mib=1,
            build_dir=self.root / "build",
        )

    def test_single_module_empty_attribute_maps_to_weight(self):
        self.assertEqual(
            ("layer.weight",),
            validator.group_weight_names("layer", {"layer": [""]}),
        )

    def test_independent_decoder_matches_fp8_and_direct_tensor_bits(self):
        stats, provenance = self.validate()

        self.assertEqual(3, stats.source_tensors)
        self.assertEqual(1, stats.decoded_groups)
        self.assertEqual(2, stats.decoded_tensors)
        self.assertEqual(1, stats.direct_tensors)
        self.assertEqual(200 + 4 * 2, stats.compared_bytes)
        self.assertEqual("0.2.0", provenance["ecf8_format_version"])

    def test_packed_fp8_bit_corruption_is_detected(self):
        path = self.converted / "linear.safetensors"
        tensors = load_file(path)
        tensors["block.packed_other_4bits"][8] ^= 1
        save_file(tensors, path)

        with self.assertRaisesRegex(
            validator.ValidationError,
            "block.a.weight.*element 17",
        ):
            self.validate()

    def test_invalid_cuda_boundary_metadata_is_detected(self):
        path = self.converted / "linear.safetensors"
        tensors = load_file(path)
        output_positions = tensors["block.output_positions"].view(
            torch.uint64
        )
        output_positions[-1] = 199
        save_file(tensors, path)

        with self.assertRaisesRegex(
            validator.ValidationError,
            "output_positions",
        ):
            self.validate()

    def test_cli_dispatches_ecf8_and_writes_success_report(self):
        report = self.root / "ecf8-report.json"
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(validator.__file__)),
                "ecf8",
                "--source",
                str(self.source),
                "--converted",
                str(self.converted),
                "--build-dir",
                str(self.root / "build"),
                "--report",
                str(report),
                "--json",
            ],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )

        rendered = json.loads(completed.stdout)
        persisted = json.loads(report.read_text())
        self.assertTrue(rendered["exact"])
        self.assertEqual("ecf8", persisted["method"])


if __name__ == "__main__":
    unittest.main()
