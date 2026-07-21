import json
import pathlib
import struct
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parent.parent
BREVIS = ROOT / "zig-out" / "bin" / "brevis"


def write_fixture(path: pathlib.Path) -> None:
    tiny = struct.pack("<f", 3.1415927)
    weight = b"".join(struct.pack("<f", 1.0 + (index % 8) / 1024) for index in range(131072))
    data = tiny + weight
    header = json.dumps({
        "empty": {
            "dtype": "F32",
            "shape": [0],
            "data_offsets": [0, 0],
        },
        "tiny": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, len(tiny)],
        },
        "weight": {
            "dtype": "F32",
            "shape": [2048, 64],
            "data_offsets": [len(tiny), len(data)],
        },
    }, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)


class BenchReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(
            ("zig", "build", "-Doptimize=ReleaseFast"), cwd=ROOT, check=True,
            stdout=subprocess.DEVNULL,
        )

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.source = pathlib.Path(self.temporary.name) / "fixture.safetensors"
        write_fixture(self.source)

    def tearDown(self):
        self.temporary.cleanup()

    def bench(self, *args: str) -> dict:
        output = subprocess.check_output((
            BREVIS, "bench", self.source, *args, "--jobs", "1", "--format", "json",
        ), text=True)
        return json.loads(output)

    def assert_size_accounting(self, report: dict) -> None:
        self.assertEqual(
            sum(block["framed_bytes"] for block in report["blocks"]),
            report["block_frame_bytes_excluding_container_header_footer"],
        )
        for block in report["blocks"]:
            self.assertEqual(
                block["program_bytecode_bytes"] + block["packed_terminal_payload_bytes"],
                block["encoded_bytes_without_frame_headers"],
            )
            self.assertEqual(
                block["encoded_bytes_without_frame_headers"] + 12,
                block["framed_bytes"],
            )
            self.assertEqual(
                block["frame_header_bytes"] + block["packed_terminal_payload_bytes"],
                block["framed_bytes"],
            )

    def compress_and_parse_frames(self) -> tuple[bytes, list[tuple[int, int]]]:
        archive = pathlib.Path(self.temporary.name) / "fixture.brv"
        subprocess.run((
            BREVIS, "compress", self.source, archive, "--plan", "fixed", "--jobs", "1",
        ), check=True, stdout=subprocess.DEVNULL)
        encoded = archive.read_bytes()
        self.assertEqual(b"BRV\x03\x06\x00\x00\x00", encoded[:8])
        self.assertEqual(b"BRVF", encoded[-12:-8])
        index_offset = struct.unpack_from("<Q", encoded, len(encoded) - 8)[0]
        frames = []
        position = 8
        while position < index_offset:
            bytecode_bytes = struct.unpack_from("<I", encoded, position)[0]
            position += 4 + bytecode_bytes
            packed_payload_bytes = struct.unpack_from("<Q", encoded, position)[0]
            position += 8 + packed_payload_bytes
            frames.append((bytecode_bytes, packed_payload_bytes))
        self.assertEqual(index_offset, position)
        return encoded, frames

    def assert_tree_metrics(self, tree: dict, expected: dict) -> None:
        children = tree["children"]
        nodes = 1 + sum(self.tree_metrics(child)[0] for child in children)
        node_depth = 1 + max((self.tree_metrics(child)[1] for child in children), default=0)
        transform_depth = (
            0 if tree["terminal"] else
            1 + max((self.tree_metrics(child)[2] for child in children), default=0)
        )
        terminals = int(tree["terminal"]) + sum(
            self.tree_metrics(child)[3] for child in children
        )
        self.assertEqual(expected["program_nodes"], nodes)
        self.assertEqual(expected["program_node_depth"], node_depth)
        self.assertEqual(expected["program_depth"], node_depth)
        self.assertEqual(expected["program_transform_depth"], transform_depth)
        self.assertEqual(expected["terminal_count"], terminals)

    def tree_metrics(self, tree: dict) -> tuple[int, int, int, int]:
        child_metrics = [self.tree_metrics(child) for child in tree["children"]]
        return (
            1 + sum(metric[0] for metric in child_metrics),
            1 + max((metric[1] for metric in child_metrics), default=0),
            0 if tree["terminal"] else 1 + max(
                (metric[2] for metric in child_metrics), default=0,
            ),
            int(tree["terminal"]) + sum(metric[3] for metric in child_metrics),
        )

    def test_schema_three_exposes_structured_programs_offsets_and_framing(self):
        report = self.bench("--plan", "fixed")
        self.assertEqual(3, report["schema"])
        self.assertEqual("brevis.bench-report", report["kind"])
        self.assertEqual(self.source.stat().st_size, report["input_size_bytes"])
        source_bytes = self.source.read_bytes()
        header_len = struct.unpack_from("<Q", source_bytes)[0]
        data_start = 8 + header_len
        self.assertEqual(data_start, report["safetensors_prefix_bytes"])
        self.assertEqual(report["raw_bytes"], report["tensor_data_bytes"])
        self.assertEqual(len(source_bytes), data_start + report["tensor_data_bytes"])
        self.assert_size_accounting(report)

        tensors = {tensor["name"]: tensor for tensor in report["tensors"]}
        tensor = tensors["weight"]
        self.assertEqual("split_field", tensor["root_operator"])
        self.assertEqual("split_field", tensor["program_tree"]["op"])
        self.assertIsInstance(tensor["program_tree"]["params_u32"], int)
        self.assertEqual(data_start, tensors["empty"]["file_data_start_byte"])
        self.assertEqual(data_start, tensors["tiny"]["file_data_start_byte"])
        self.assertEqual(data_start + 4, tensors["weight"]["file_data_start_byte"])
        for value in tensors.values():
            self.assertGreaterEqual(value["file_data_start_byte"], data_start)
            self.assertLessEqual(value["file_data_end_byte_exclusive"], len(source_bytes))
            self.assertEqual(
                value["raw_bytes"],
                value["file_data_end_byte_exclusive"] - value["file_data_start_byte"],
            )
            self.assertEqual(
                value["raw_root_blocks"],
                value["planned_raw_root_blocks"] + value["fallback_raw_root_blocks"],
            )
            if value["program_tree"] is not None:
                self.assert_tree_metrics(value["program_tree"], value)
        for key in (
            "program_nodes", "program_depth", "program_node_depth",
            "program_transform_depth", "terminal_count",
        ):
            self.assertIn(key, tensors["empty"])
            self.assertIsNone(tensors["empty"][key])

        self.assertEqual(
            report["encoded_bytes_without_frame_headers"],
            sum(value["encoded_bytes_without_frame_headers"] for value in tensors.values()),
        )
        self.assertEqual(
            report["block_frame_bytes_excluding_container_header_footer"],
            sum(value["block_frame_bytes_excluding_container_header_footer"] for value in tensors.values()),
        )
        covered = []
        for value in report["tensors"]:
            covered.extend(range(value["block_start"], value["block_start"] + value["block_count"]))
            for index in range(value["block_start"], value["block_start"] + value["block_count"]):
                self.assertEqual(value["index"], report["blocks"][index]["tensor_index"])
        self.assertEqual(list(range(len(report["blocks"]))), covered)
        self.assertEqual(report["raw_bytes"], sum(block["raw_bytes"] for block in report["blocks"]))
        self.assertEqual(
            report["encoded_bytes_without_frame_headers"],
            sum(block["encoded_bytes_without_frame_headers"] for block in report["blocks"]),
        )
        self.assertEqual(
            report["block_frame_bytes_excluding_container_header_footer"],
            sum(block["framed_bytes"] for block in report["blocks"]),
        )

        encoded, frames = self.compress_and_parse_frames()
        self.assertEqual(len(report["blocks"]), len(frames))
        for block, (bytecode_bytes, packed_payload_bytes) in zip(report["blocks"], frames):
            self.assertEqual(bytecode_bytes, block["program_bytecode_bytes"])
            self.assertEqual(packed_payload_bytes, block["packed_terminal_payload_bytes"])
        frame_bytes = sum(bytecode + payload + 12 for bytecode, payload in frames)
        self.assertEqual(report["block_frame_bytes_excluding_container_header_footer"], frame_bytes)
        self.assertEqual(len(encoded), report["projected_archive_bytes"])
        self.assertEqual(8, report["container_header_bytes"])
        self.assertEqual(
            len(encoded),
            report["container_header_bytes"] + frame_bytes + report["container_footer_bytes"],
        )

        self.assertGreater(tensors["tiny"]["fallback_raw_root_blocks"], 0)
        self.assertGreater(tensor["block_count"], 1)
        self.assertTrue(any(block["root_operator"] != "raw" for block in report["blocks"]))
        for block in report["blocks"]:
            self.assertIn("params_u32", block["program_tree"])
            self.assertIn(block["raw_classification"], (None, "fallback_raw"))
            self.assert_tree_metrics(block["program_tree"], block)

    def test_raw_only_search_distinguishes_planned_raw_from_fallback(self):
        config = json.loads(subprocess.check_output((BREVIS, "config"), text=True))
        disable = [
            argument
            for op in config["enabled_ops"] if op != "raw"
            for argument in ("--disable-op", op)
        ]
        report = self.bench("--plan", "search", *disable)
        self.assert_size_accounting(report)
        self.assertEqual(["raw"], report["search"]["enabled_ops"])
        for tensor in report["tensors"]:
            if tensor["block_count"] == 0:
                self.assertIsNone(tensor["root_operator"])
                continue
            self.assertEqual("raw", tensor["root_operator"])
            self.assertEqual(0, tensor["fallback_raw_root_blocks"])
            self.assertEqual(tensor["block_count"], tensor["planned_raw_root_blocks"])
        for block in report["blocks"]:
            self.assertEqual("planned_raw", block["raw_classification"])


if __name__ == "__main__":
    unittest.main()
