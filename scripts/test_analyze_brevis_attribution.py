import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("analyze_brevis_attribution.py")
SPEC = importlib.util.spec_from_file_location("brevis_attribution", SCRIPT)
assert SPEC and SPEC.loader
analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


def uleb(value: int) -> bytes:
    output = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            byte |= 0x80
        output.append(byte)
        if not value:
            return bytes(output)


def safetensors_prefix(entries: list[tuple[str, str, list[int], int]]) -> bytes:
    header = {}
    offset = 0
    for name, dtype, shape, size in entries:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<Q", len(encoded)) + encoded


def literal_program(bits: int, count: int, body: bytes) -> bytes:
    return b"BRPG\x02" + b"\x01" + bytes([bits]) + uleb(count) + uleb(
        len(body)
    ) + body


def constant_program(bits: int, count: int, word: int) -> bytes:
    return (
        b"BRPG\x02"
        + b"\x02"
        + bytes([bits])
        + uleb(count)
        + uleb(word)
    )


def map_xor_program(bits: int, count: int, parameter: int, body: bytes) -> bytes:
    child = b"\x01" + bytes([bits]) + uleb(count) + uleb(len(body)) + body
    return b"BRPG\x02" + b"\x05\x01" + uleb(parameter) + child


def record(name: str, dtype_id: int, shape: list[int], program: bytes) -> bytes:
    encoded_name = name.encode()
    body = (
        b"\x01"
        + uleb(len(encoded_name))
        + encoded_name
        + bytes([dtype_id])
        + uleb(len(shape))
        + b"".join(uleb(value) for value in shape)
        + uleb(len(program))
        + program
        + b"\x00" * 8
    )
    return uleb(len(body)) + body


def fixture_bytes() -> tuple[bytes, bytes, list[bytes]]:
    entries = [
        ("model.embed_tokens.weight", "BF16", [2], 4),
        ("model.layers.0.self_attn.q_proj.weight", "BF16", [2], 4),
        ("model.layers.0.mlp.down_proj.weight", "U8", [8], 8),
    ]
    prefix = safetensors_prefix(entries)
    programs = [
        literal_program(16, 2, b"\x00" + b"\x01\x02\x03\x04"),
        constant_program(16, 2, 0),
        map_xor_program(8, 8, 1, b"\x01\x01\x55"),
    ]
    records = [
        record(entries[0][0], 0x02, [2], programs[0]),
        record(entries[1][0], 0x02, [2], programs[1]),
        record(entries[2][0], 0x04, [8], programs[2]),
    ]
    archive = (
        b"BRTA\x03"
        + uleb(len(entries))
        + uleb(len(prefix))
        + prefix
        + b"".join(records)
    )
    source = prefix + bytes(range(16))
    return archive, source, programs


class AttributionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        archive, source, programs = fixture_bytes()
        self.archive = self.root / "model.brv"
        self.source = self.root / "model.safetensors"
        self.archive.write_bytes(archive)
        self.source.write_bytes(source)
        self.programs = programs

    def tearDown(self):
        self.temporary.cleanup()

    def test_exact_record_accounting_and_grouping(self):
        report = analysis.analyze([self.archive], [self.source])
        rows = report["_objects"]["tensor_rows"]
        summary = report["_objects"]["archive_summaries"][0]

        self.assertEqual(3, len(rows))
        self.assertEqual(
            ["embedding", "attention", "mlp"],
            [row.role for row in rows],
        )
        self.assertEqual(
            ["literal_fallback", "synthesized", "synthesized"],
            [row.program_selection for row in rows],
        )
        self.assertEqual(
            ["literal", "constant", "map.xor"],
            [row.root_operator for row in rows],
        )
        self.assertEqual([4, 4, 8], [row.source_tensor_bytes for row in rows])
        self.assertEqual(
            self.archive.stat().st_size,
            summary.archive_header_bytes
            + sum(row.archive_record_bytes for row in rows),
        )
        self.assertEqual(
            self.source.stat().st_size,
            summary.source_file_bytes,
        )
        for row, program in zip(rows, self.programs):
            self.assertEqual(len(program), row.program_bytes)
            self.assertEqual(
                row.program_bytes,
                analysis.PROGRAM_HEADER_BYTES
                + sum(row.operator_exclusive_bytes.values()),
            )

        role_groups = {
            row["group"]: row for row in report["groups"]["role"]
        }
        self.assertEqual(1, role_groups["embedding"]["literal_fallback_tensors"])
        self.assertEqual(0, role_groups["attention"]["literal_fallback_tensors"])
        program_groups = {
            row["group"]: row for row in report["groups"]["program"]
        }
        self.assertIn("literal_fallback", program_groups)
        self.assertIn("synthesized:constant", program_groups)
        self.assertIn("synthesized:map.xor", program_groups)

    def test_literal_codecs_and_operator_bytes_are_measured_not_savings(self):
        report = analysis.analyze([self.archive], [self.source])
        codecs = {row["codec"]: row for row in report["literal_codecs"]}
        self.assertEqual(1, codecs["raw"]["literal_count"])
        self.assertEqual(5, codecs["raw"]["body_bytes"])
        self.assertEqual(4, codecs["raw"]["semantic_storage_bytes"])
        self.assertEqual(1, codecs["bitpack"]["literal_count"])
        self.assertEqual(3, codecs["bitpack"]["body_bytes"])
        self.assertEqual(8, codecs["bitpack"]["semantic_storage_bytes"])
        operators = {row["operator"]: row for row in report["operators"]}
        self.assertEqual(1, operators["map.xor"]["root_selected_tensors"])
        self.assertEqual(0, operators["literal"]["exclusive_program_bytes"] < 0)
        self.assertNotIn("operator_saved_bytes", operators["literal"])

    def test_source_prefix_must_match_when_sources_are_supplied(self):
        other_prefix = safetensors_prefix(
            [("different.weight", "BF16", [2], 4)]
        )
        other = self.root / "other.safetensors"
        other.write_bytes(other_prefix + b"\x00" * 4)
        with self.assertRaisesRegex(
            analysis.AttributionError,
            "matches no supplied source",
        ):
            analysis.analyze([self.archive], [other])

    def test_outputs_are_machine_readable_and_explain_limits(self):
        report = analysis.analyze([self.archive], [self.source])
        output = self.root / "analysis"
        analysis.write_outputs(report, output)
        expected = {
            "archives.csv",
            "attribution.json",
            "by-dtype.csv",
            "by-program.csv",
            "by-role-program.csv",
            "by-role.csv",
            "by-selection.csv",
            "literal-codecs.csv",
            "operators.csv",
            "report.md",
            "tensors.csv",
        }
        self.assertEqual(expected, {path.name for path in output.iterdir()})
        payload = json.loads((output / "attribution.json").read_text())
        self.assertEqual(3, payload["totals"]["tensor_count"])
        self.assertTrue(
            payload["validation_scope"]["literal_body_framing_check"]
        )
        self.assertFalse(
            payload["validation_scope"]["literal_payload_and_table_validation"]
        )
        markdown = (output / "report.md").read_text()
        self.assertIn("not causal savings", markdown)
        self.assertIn("brevis verify ARCHIVE SOURCE", markdown)

    def test_static_program_type_mismatch_is_rejected(self):
        archive, source, _ = fixture_bytes()
        # Narrow the second tensor's constant root without changing its framing.
        location = archive.index(b"BRPG\x02\x02\x10\x02")
        corrupted = bytearray(archive)
        corrupted[location + 6] = 8
        bad = self.root / "bad.brv"
        bad.write_bytes(corrupted)
        with self.assertRaisesRegex(analysis.AttributionError, "program type"):
            analysis.analyze([bad])

    def test_tensor_role_taxonomy(self):
        cases = {
            "transformer.wte.weight": "embedding",
            "model.layers.2.self_attn.o_proj.weight": "attention",
            "model.layers.2.mlp.gate_proj.weight": "mlp",
            "model.layers.2.input_layernorm.weight": "norm",
            "model.layers.2.post_attention_layernorm.weight": "norm",
            "model.layers.2.self_attn_layer_norm.weight": "norm",
            "model.layers.2.mlp_layernorm.weight": "norm",
            "lm_head.weight": "other",
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                self.assertEqual(expected, analysis.classify_tensor_role(name))

    def test_zero_sized_tensor_order_follows_json_source_order(self):
        entries = [
            ("z_empty", "U8", [0], 0),
            ("a_empty", "U8", [0], 0),
            ("payload", "U8", [1], 1),
        ]
        prefix = safetensors_prefix(entries)
        programs = [
            literal_program(8, 0, b"\x00"),
            literal_program(8, 0, b"\x00"),
            constant_program(8, 1, 7),
        ]
        frames = [
            record(name, 0x04, shape, program)
            for (name, _dtype, shape, _size), program in zip(entries, programs)
        ]
        archive = (
            b"BRTA\x03"
            + uleb(3)
            + uleb(len(prefix))
            + prefix
            + b"".join(frames)
        )
        path = self.root / "empty-order.brv"
        path.write_bytes(archive)
        report = analysis.analyze([path])
        self.assertEqual(
            ["z_empty", "a_empty", "payload"],
            [
                row.tensor_name
                for row in report["_objects"]["tensor_rows"]
            ],
        )

    def test_pathological_json_errors_are_wrapped(self):
        huge_integer_json = (
            b'{"x":{"dtype":"U8","shape":['
            + b"9" * 5000
            + b'],"data_offsets":[0,0]}}'
        )
        huge_prefix = struct.pack("<Q", len(huge_integer_json)) + huge_integer_json
        with self.assertRaisesRegex(
            analysis.AttributionError,
            "malformed safetensors JSON",
        ):
            analysis.parse_safetensors_prefix(huge_prefix, "huge")

        surrogate_json = (
            b'{"\\ud800":{"dtype":"U8","shape":[0],"data_offsets":[0,0]}}'
        )
        surrogate_prefix = struct.pack("<Q", len(surrogate_json)) + surrogate_json
        with self.assertRaisesRegex(analysis.AttributionError, "invalid tensor name"):
            analysis.parse_safetensors_prefix(surrogate_prefix, "surrogate")


if __name__ == "__main__":
    unittest.main()
