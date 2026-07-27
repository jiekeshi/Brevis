import io
import json
import struct
import unittest

import brta_program_inventory as inventory


def uleb(value):
    encoded = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        encoded.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(encoded)


def literal(bits=8, count=4, body=b"raw"):
    return b"\x01" + bytes((bits,)) + uleb(count) + uleb(len(body)) + body


def constant(bits=8, count=4, word=0):
    return b"\x02" + bytes((bits,)) + uleb(count) + uleb(word)


def scan_xor(child):
    return b"\x06\x01" + uleb(0) + child


def map_add(child):
    return b"\x05\x02" + uleb(1) + child


def merge_float(children):
    return b"\x07\x02\x03" + uleb(len(children)) + b"".join(children)


def program(root):
    return b"BRPG\x01" + root


def frame(name, bytecode, dtype=3, shape=(1,), body_suffix=b"", record_tag=1):
    body = (
        bytes((record_tag,))
        + uleb(len(name))
        + name
        + bytes((dtype,))
        + uleb(len(shape))
        + b"".join(uleb(dimension) for dimension in shape)
        + uleb(len(bytecode))
        + bytecode
        + bytes(32)
        + body_suffix
    )
    return uleb(len(body)) + body


def archive(frames, suffix=b"", version=1):
    prefix = struct.pack("<Q", 2) + b"{}"
    return (
        b"BRTA"
        + bytes((version,))
        + uleb(len(frames))
        + uleb(len(prefix))
        + prefix
        + b"".join(frames)
        + suffix
    )


class BrtaProgramInventoryTests(unittest.TestCase):
    def parse(self, payload):
        result = inventory.inventory_stream(
            io.BytesIO(payload), len(payload), "fixture.brta"
        )
        json.dumps(result)
        return result

    def test_inventories_terminal_single_and_composed_programs(self):
        literal_program = program(literal())
        single_program = program(scan_xor(literal()))
        composed_program = program(
            merge_float((
                map_add(scan_xor(literal())),
                literal(bits=8, count=4, body=b"x"),
                constant(),
            ))
        )
        constant_program = program(constant())
        payload = archive([
            frame(b"literal", literal_program),
            frame(b"single", single_program),
            frame(b"composed", composed_program),
            frame(b"constant", constant_program),
        ])

        result = self.parse(payload)
        summary = result["summary"]

        self.assertEqual(4, result["archive"]["tensor_count"])
        self.assertEqual(4, summary["tensor_count"])
        self.assertEqual(
            sum(map(len, (
                literal_program,
                single_program,
                composed_program,
                constant_program,
            ))),
            summary["program_bytes"]["total"],
        )
        self.assertEqual(
            {
                "literal": 1,
                "scan.xor": 1,
                "merge.float_fields": 1,
                "constant": 1,
            },
            summary["root_kinds"],
        )
        self.assertEqual(
            {
                "literal_only": 1,
                "zero_semantic_ops": 2,
                "single_semantic_op": 1,
                "at_least_two_semantic_ops": 1,
            },
            summary["classifications"],
        )
        self.assertEqual(
            {
                "map.add_mod": 1,
                "merge.float_fields": 1,
                "scan.xor": 2,
            },
            summary["semantic_operator_counts"],
        )
        self.assertEqual(
            1,
            summary["semantic_preorder_combinations"][
                "merge.float_fields > map.add_mod > scan.xor"
            ],
        )
        self.assertEqual(1, summary["semantic_preorder_combinations"]["scan.xor"])
        self.assertEqual(2, summary["semantic_preorder_combinations"]["none"])

        composed = result["tensors"][2]
        self.assertEqual("merge.float_fields", composed["root_kind"])
        self.assertEqual(6, composed["node_count"])
        self.assertEqual(4, composed["ast_depth"])
        self.assertEqual(3, composed["semantic_op_count"])
        self.assertEqual(3, composed["semantic_op_depth"])
        self.assertEqual(
            ["merge.float_fields", "map.add_mod", "scan.xor"],
            composed["semantic_operators_preorder"],
        )
        self.assertEqual(
            {
                "merge.float_fields > constant": 1,
                "merge.float_fields > literal": 1,
                (
                    "merge.float_fields > map.add_mod > scan.xor > literal"
                ): 1,
            },
            composed["root_to_terminal_sequences"],
        )
        self.assertTrue(result["evidence_scope"]["structural_only"])
        self.assertTrue(
            result["evidence_scope"]["semantic_verification_required"]
        )

    def test_rejects_truncated_and_trailing_archive_bytes(self):
        payload = archive([frame(b"x", program(literal()))])
        with self.assertRaisesRegex(inventory.InventoryError, "truncated"):
            self.parse(payload[:-1])
        with self.assertRaisesRegex(inventory.InventoryError, "trailing bytes"):
            self.parse(payload + b"x")

    def test_parses_every_node_and_parameter_shape(self):
        roots = {
            "concat": b"\x03" + uleb(2) + literal() + constant(),
            "repeat": b"\x04" + uleb(3) + literal(),
            "map.xor": b"\x05\x01" + uleb(7) + literal(),
            "map.gray": b"\x05\x04" + literal(),
            "map.rotate_left": b"\x05\x05" + uleb(7) + literal(),
            "scan.add_mod": b"\x06\x02" + uleb(0) + literal(),
            "merge.fields": (
                b"\x07\x01" + uleb(4) + uleb(2) + literal() + literal()
            ),
            "merge.bit_planes": (
                b"\x07\x03" + uleb(2) + literal() + literal()
            ),
            "merge.byte_planes": (
                b"\x07\x04" + uleb(2) + literal() + literal()
            ),
        }
        payload = archive([
            frame(kind.encode(), program(root))
            for kind, root in roots.items()
        ])
        result = self.parse(payload)
        self.assertEqual(
            list(roots),
            [tensor["root_kind"] for tensor in result["tensors"]],
        )

    def test_rejects_unknown_record_node_and_operation_tags(self):
        cases = {
            "record": (
                archive([frame(b"x", program(literal()), record_tag=0xFF)]),
                "unknown record tag",
            ),
            "node": (
                archive([frame(b"x", program(b"\xff"))]),
                "unknown node tag",
            ),
            "map": (
                archive([frame(b"x", program(b"\x05\xff"))]),
                "unknown map tag",
            ),
            "scan": (
                archive([frame(b"x", program(b"\x06\xff"))]),
                "unknown scan tag",
            ),
            "merge": (
                archive([frame(b"x", program(b"\x07\xff"))]),
                "unknown merge tag",
            ),
            "record dtype": (
                archive([frame(b"x", program(literal()), dtype=0xFF)]),
                "unknown record dtype tag",
            ),
            "program dtype": (
                archive([
                    frame(b"x", program(b"\x07\x02\xff\x00")),
                ]),
                "unknown program dtype tag",
            ),
        }
        for label, (payload, message) in cases.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(inventory.InventoryError, message):
                    self.parse(payload)

    def test_rejects_invalid_terminal_parameters_and_operator_arities(self):
        lit = literal()
        invalid_roots = {
            "constant width": b"\x02\x00" + uleb(1) + uleb(0),
            "constant width high": b"\x02\x21" + uleb(1) + uleb(0),
            "constant count": b"\x02\x08" + uleb(0) + uleb(0),
            "constant word": b"\x02\x01" + uleb(1) + uleb(2),
            "concat": b"\x03" + uleb(1) + lit,
            "repeat": b"\x04" + uleb(1) + lit,
            "fields parameter": (
                b"\x07\x01" + uleb(0) + uleb(2) + lit + lit
            ),
            "fields arity": b"\x07\x01" + uleb(4) + uleb(1) + lit,
            "fields arity high": (
                b"\x07\x01" + uleb(4) + uleb(3) + lit * 3
            ),
            "float dtype": (
                b"\x07\x02\x04" + uleb(3) + lit + lit + lit
            ),
            "float arity": b"\x07\x02\x03" + uleb(2) + lit + lit,
            "float arity high": b"\x07\x02\x03" + uleb(4) + lit * 4,
            "bit plane arity": b"\x07\x03" + uleb(1) + lit,
            "bit plane arity high": b"\x07\x03" + uleb(33) + lit * 33,
            "byte plane arity": b"\x07\x04" + uleb(1) + lit,
            "byte plane arity high": b"\x07\x04" + uleb(5) + lit * 5,
        }
        for label, root in invalid_roots.items():
            with self.subTest(label=label):
                with self.assertRaisesRegex(inventory.InventoryError, "invalid"):
                    self.parse(archive([frame(b"x", program(root))]))

    def test_rejects_program_and_record_trailing_data(self):
        with self.assertRaisesRegex(inventory.InventoryError, "trailing bytes"):
            self.parse(archive([
                frame(b"x", program(literal()) + b"x"),
            ]))
        with self.assertRaisesRegex(inventory.InventoryError, "trailing record data"):
            self.parse(archive([
                frame(b"x", program(literal()), body_suffix=b"x"),
            ]))

    def test_rejects_noncanonical_uleb_and_unsupported_versions(self):
        prefix = struct.pack("<Q", 2) + b"{}"
        overlong_count = b"BRTA\x01\x80\x00" + uleb(len(prefix)) + prefix
        with self.assertRaisesRegex(inventory.InventoryError, "overlong ULEB128"):
            self.parse(overlong_count)
        with self.assertRaisesRegex(inventory.InventoryError, "bad BRTA magic"):
            self.parse(b"XXXX" + archive([])[4:])
        with self.assertRaisesRegex(
            inventory.InventoryError, "unsupported BRTA version"
        ):
            self.parse(archive([], version=2))
        bad_program_magic = b"XXXX\x01" + literal()
        with self.assertRaisesRegex(inventory.InventoryError, "bad BRPG magic"):
            self.parse(archive([frame(b"x", bad_program_magic)]))
        bad_program = b"BRPG\x02" + literal()
        with self.assertRaisesRegex(
            inventory.InventoryError, "unsupported BRPG version"
        ):
            self.parse(archive([frame(b"x", bad_program)]))


if __name__ == "__main__":
    unittest.main()
