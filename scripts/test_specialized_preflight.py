import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import preflight_specialized_baselines as preflight
import run_benchmarks as bench
from benchmark_corpus import CHECKPOINT_BY_NAME


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "configs" / "specialized-baselines.example.json"


def write_safetensors(path: Path, tensors: list[tuple[str, str, bytes]]) -> None:
    offset = 0
    payload = bytearray()
    header = {}
    for name, dtype, data in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": [len(data)],
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        payload.extend(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def write_model_fixture(
    root: Path,
    name: str,
    tensors: list[tuple[str, str, bytes]],
    config: dict,
) -> tuple[Path, dict]:
    directory = root / name
    directory.mkdir()
    weight = directory / "model.safetensors"
    write_safetensors(weight, tensors)
    (directory / "model.safetensors.index.json").write_text("{}")
    (directory / "config.json").write_text(json.dumps(config))
    frozen = CHECKPOINT_BY_NAME[name]
    manifest = {
        "name": name,
        "repo_id": frozen.repo_id,
        "revision": frozen.revision,
        "index_file": "model.safetensors.index.json",
        "source_bytes": weight.stat().st_size,
        "sha256_verified": True,
        "weights": [
            {
                "path": weight.name,
                "size": weight.stat().st_size,
                "sha256": "0" * 64,
            }
        ],
    }
    (directory / "download-manifest.json").write_text(json.dumps(manifest))
    return directory, manifest


class SpecializedPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_example_freezes_commits_workers_and_conversion_semantics(self):
        config = bench.load_specialized_config(EXAMPLE)

        self.assertEqual(1, config["dfloat11"]["workers"])
        self.assertEqual(16, config["ecf8"]["workers"])
        for method, settings in config.items():
            self.assertEqual(
                bench.SPECIALIZED_COMMITS[method],
                settings["expected_commit"],
            )
            protocol = settings["protocol"]
            self.assertEqual(1, protocol["schema_version"])
            self.assertEqual(
                "native_checkpoint_conversion",
                protocol["kind"],
            )
            self.assertEqual("tensor_value_exact", protocol["exactness_scope"])
            self.assertEqual(
                "tensor_bit_exact",
                protocol["publication_exactness_target"],
            )
            self.assertFalse(protocol["original_checkpoint_byte_exact"])
            self.assertFalse(
                protocol["independent_bitwise_validation_performed"]
            )
            self.assertIn("--upstream", settings["compress_command"])

    def test_config_cannot_claim_original_checkpoint_byte_exactness(self):
        config = json.loads(EXAMPLE.read_text())
        config.pop("ecf8")
        config["dfloat11"]["protocol"][
            "original_checkpoint_byte_exact"
        ] = True
        path = self.root / "invalid.json"
        path.write_text(json.dumps(config))

        with self.assertRaisesRegex(
            bench.BenchmarkError,
            "original_checkpoint_byte_exact",
        ):
            bench.load_specialized_config(path)

    def test_inline_validation_requires_official_validation_flag(self):
        config = json.loads(EXAMPLE.read_text())
        config.pop("ecf8")
        config["dfloat11"]["compress_command"].remove("--validate-cuda")
        path = self.root / "missing-validation.json"
        path.write_text(json.dumps(config))

        with self.assertRaisesRegex(bench.BenchmarkError, "--validate-cuda"):
            bench.load_specialized_config(path)

    def test_harness_preflight_rejects_wrong_official_commit(self):
        config = bench.load_specialized_config(EXAMPLE)
        args = SimpleNamespace(
            methods=["dfloat11"],
            allow_missing=True,
        )

        with (
            patch.object(
                bench,
                "method_versions",
                return_value={"dfloat11": "f" * 40},
            ),
            self.assertRaisesRegex(
                bench.BenchmarkError,
                "specialized commit mismatch",
            ),
        ):
            bench.preflight(args, config)

    def test_specialized_identity_binds_exactness_scope_and_evidence(self):
        source = self.root / "model.safetensors"
        source.write_bytes(b"fixture")
        checkpoint = bench.Checkpoint("fixture", self.root, (source,))
        protocol = json.loads(EXAMPLE.read_text())["dfloat11"]["protocol"]

        identity = bench.specialized_identity(
            checkpoint,
            "dfloat11",
            1,
            {"method_version": bench.SPECIALIZED_COMMITS["dfloat11"]},
            protocol,
        )

        self.assertEqual(
            "native_checkpoint_conversion",
            identity["conversion_protocol"],
        )
        self.assertEqual(1, identity["conversion_protocol_schema_version"])
        self.assertEqual("tensor_value_exact", identity["exactness_scope"])
        self.assertEqual(
            "tensor_bit_exact",
            identity["publication_exactness_target"],
        )
        self.assertFalse(identity["original_checkpoint_byte_exact"])
        self.assertFalse(
            identity["independent_bitwise_validation_performed"]
        )

    def test_dfloat11_rejects_fp16_source_metadata(self):
        directory, manifest = write_model_fixture(
            self.root,
            "llama-3.1-8b-bf16",
            [("model.embed_tokens.weight", "F16", b"\0\0")],
            {"architectures": ["LlamaForCausalLM"]},
        )
        findings = []

        preflight.check_model(
            "dfloat11",
            "llama-3.1-8b-bf16",
            directory,
            manifest,
            findings,
        )

        self.assertIn(
            "unsupported_source_dtype",
            {item.code for item in findings if item.severity == "ERROR"},
        )

    def test_ecf8_requires_tokenizer_assets(self):
        directory, manifest = write_model_fixture(
            self.root,
            "qwen3-32b-fp8",
            [
                ("model.layers.0.weight", "F8_E4M3", b"\0"),
                ("model.norm.weight", "BF16", b"\0\0"),
            ],
            {
                "architectures": ["Qwen3ForCausalLM"],
                "quantization_config": {
                    "quant_method": "fp8",
                    "fmt": "e4m3",
                },
            },
        )
        findings = []

        preflight.check_model(
            "ecf8",
            "qwen3-32b-fp8",
            directory,
            manifest,
            findings,
        )

        self.assertIn(
            "missing_tokenizer",
            {item.code for item in findings if item.severity == "ERROR"},
        )

    def test_publication_mode_rejects_official_validation_alone(self):
        config = bench.load_specialized_config(EXAMPLE)

        findings = preflight.audit(
            config,
            self.root,
            ["ecf8"],
            skip_runtime=True,
            publication_ready=True,
        )

        publication = [
            item
            for item in findings
            if item.code == "independent_bitwise_validation_missing"
        ]
        self.assertEqual(1, len(publication))
        self.assertEqual("ERROR", publication[0].severity)


if __name__ == "__main__":
    unittest.main()
