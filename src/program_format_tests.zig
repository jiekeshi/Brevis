const std = @import("std");
const dsl = @import("dsl.zig");
const interpreter = @import("interpreter.zig");
const format = @import("program_format.zig");

const Allocator = std.mem.Allocator;

const all_kinds_golden = [_]u8{
    'B',  'R',  'P',  'G',  0x01,
    0x03, 0x02, 0x04, 0x02, 0x05,
    0x01, 0x01, 0x01, 0x08, 0x02,
    0x03, 0x00, 0x01, 0x02, 0x06,
    0x02, 0x00, 0x07, 0x01, 0x04,
    0x02, 0x02, 0x04, 0x01, 0x01,
    0x01, 0x04, 0x01, 0x02, 0x00,
    0x02,
};

test "golden format covers all seven semantic node kinds" {
    const alloc = std.testing.allocator;
    var program = try makeAllKindsProgram(alloc);
    defer program.deinit(alloc);

    const bytes = try format.serialize(alloc, program);
    defer alloc.free(bytes);

    try std.testing.expectEqualSlices(u8, &all_kinds_golden, bytes);
    try std.testing.expectEqual(
        bytes.len,
        try format.serializedSize(alloc, program),
    );

    var decoded = try format.deserialize(alloc, bytes, .{});
    defer decoded.deinit(alloc);
    const canonical = try format.serialize(alloc, decoded);
    defer alloc.free(canonical);
    try std.testing.expectEqualSlices(u8, bytes, canonical);

    try expectSameOutput(alloc, program, decoded);
}

test "roundtrip and exact size cover every map and scan opcode" {
    const alloc = std.testing.allocator;
    const map_ops = [_]dsl.MapOp{
        .{ .xor = 0x5a },
        .{ .add_mod = 7 },
        .zigzag,
        .gray,
        .{ .rotate_left = 3 },
        .bit_reverse,
    };
    for (map_ops) |operation| {
        const literal = try dsl.Program.literal(alloc, 8, &.{ 1, 2, 0xff });
        var program = try dsl.Program.map(alloc, operation, literal);
        defer program.deinit(alloc);
        try expectCanonicalRoundtrip(alloc, program);
    }

    const scan_ops = [_]dsl.ScanOp{ .xor, .add_mod };
    for (scan_ops) |operation| {
        const updates = try dsl.Program.literal(alloc, 8, &.{ 1, 2, 3 });
        var program = try dsl.Program.scan(alloc, operation, 5, updates);
        defer program.deinit(alloc);
        try expectCanonicalRoundtrip(alloc, program);
    }
}

test "roundtrip and exact size cover every merge opcode" {
    const alloc = std.testing.allocator;

    {
        var low = try dsl.Program.literal(alloc, 4, &.{ 1, 2 });
        defer low.deinit(alloc);
        var high = try dsl.Program.literal(alloc, 4, &.{ 3, 4 });
        defer high.deinit(alloc);
        var program = try dsl.Program.merge(
            alloc,
            .{ .fields = 4 },
            &.{ low, high },
        );
        defer program.deinit(alloc);
        try expectCanonicalRoundtrip(alloc, program);
    }

    {
        var signs = try dsl.Program.literal(alloc, 1, &.{ 0, 1 });
        defer signs.deinit(alloc);
        var exponents = try dsl.Program.literal(alloc, 8, &.{ 0x7f, 0x80 });
        defer exponents.deinit(alloc);
        var mantissas = try dsl.Program.literal(alloc, 23, &.{ 0, 1 });
        defer mantissas.deinit(alloc);
        var program = try dsl.Program.merge(
            alloc,
            .{ .float_fields = .f32 },
            &.{ signs, exponents, mantissas },
        );
        defer program.deinit(alloc);
        try expectCanonicalRoundtrip(alloc, program);
    }

    {
        var bit0 = try dsl.Program.literal(alloc, 1, &.{ 0, 1 });
        defer bit0.deinit(alloc);
        var bit1 = try dsl.Program.literal(alloc, 1, &.{ 1, 0 });
        defer bit1.deinit(alloc);
        var program = try dsl.Program.merge(
            alloc,
            .bit_planes,
            &.{ bit0, bit1 },
        );
        defer program.deinit(alloc);
        try expectCanonicalRoundtrip(alloc, program);
    }

    {
        var low_byte = try dsl.Program.literal(alloc, 8, &.{ 0xab, 0xcd });
        defer low_byte.deinit(alloc);
        var high_nibble = try dsl.Program.literal(alloc, 4, &.{ 1, 2 });
        defer high_nibble.deinit(alloc);
        var program = try dsl.Program.merge(
            alloc,
            .byte_planes,
            &.{ low_byte, high_nibble },
        );
        defer program.deinit(alloc);
        try expectCanonicalRoundtrip(alloc, program);
    }
}

test "literal raw words use canonical storage widths and ULEB128 lengths" {
    const alloc = std.testing.allocator;
    var words = [_]u32{0} ** 128;
    words[0] = 0x1234_5678;
    words[127] = 0xffff_ffff;

    var program = try dsl.Program.literal(alloc, 32, &words);
    defer program.deinit(alloc);
    const bytes = try format.serialize(alloc, program);
    defer alloc.free(bytes);

    try std.testing.expectEqual(@as(u8, 0x80), bytes[7]);
    try std.testing.expectEqual(@as(u8, 0x01), bytes[8]);
    try std.testing.expectEqual(
        bytes.len,
        try format.serializedSize(alloc, program),
    );
    try expectCanonicalRoundtrip(alloc, program);
}

test "empty Lit round trips and valid non-minimal literal bodies decode" {
    const alloc = std.testing.allocator;
    var empty = try dsl.Program.literal(alloc, 8, &.{});
    defer empty.deinit(alloc);
    try expectCanonicalRoundtrip(alloc, empty);

    const non_minimal_raw = [_]u8{
        'B',  'R',  'P',  'G',  0x01,
        0x01, 0x08, 0x03, 0x04, 0x00,
        0x00, 0x00, 0x00,
    };
    var decoded = try format.deserialize(alloc, &non_minimal_raw, .{});
    defer decoded.deinit(alloc);
    const literal = decoded.kind.literal;
    try std.testing.expectEqual(@as(usize, 3), literal.count);
    try std.testing.expect(literal.isUniform());
}

test "decoder rejects trailing overlong unknown and truncated encodings" {
    const alloc = std.testing.allocator;

    const trailing = all_kinds_golden ++ [_]u8{0};
    try std.testing.expectError(
        error.TrailingBytes,
        format.deserialize(alloc, &trailing, .{}),
    );

    const overlong = [_]u8{
        'B',  'R',  'P',  'G',  0x01,
        0x01, 0x08, 0x81, 0x00, 0x00,
    };
    try std.testing.expectError(
        error.OverlongUleb128,
        format.deserialize(alloc, &overlong, .{}),
    );

    const unknown_node = [_]u8{ 'B', 'R', 'P', 'G', 0x01, 0xff };
    try std.testing.expectError(
        error.UnknownNode,
        format.deserialize(alloc, &unknown_node, .{}),
    );

    const unknown_map = [_]u8{ 'B', 'R', 'P', 'G', 0x01, 0x05, 0xff };
    try std.testing.expectError(
        error.UnknownMapOp,
        format.deserialize(alloc, &unknown_map, .{}),
    );

    const unknown_scan = [_]u8{ 'B', 'R', 'P', 'G', 0x01, 0x06, 0xff };
    try std.testing.expectError(
        error.UnknownScanOp,
        format.deserialize(alloc, &unknown_scan, .{}),
    );

    const unknown_merge = [_]u8{ 'B', 'R', 'P', 'G', 0x01, 0x07, 0xff };
    try std.testing.expectError(
        error.UnknownMergeOp,
        format.deserialize(alloc, &unknown_merge, .{}),
    );

    const unknown_dtype = [_]u8{
        'B',  'R',  'P',  'G', 0x01,
        0x07, 0x02, 0xff,
    };
    try std.testing.expectError(
        error.UnknownDtype,
        format.deserialize(alloc, &unknown_dtype, .{}),
    );

    for (0..all_kinds_golden.len) |prefix_len|
        try std.testing.expectError(
            error.Truncated,
            format.deserialize(
                alloc,
                all_kinds_golden[0..prefix_len],
                .{},
            ),
        );
}

test "decoder validates header integers and literal bodies" {
    const alloc = std.testing.allocator;

    const bad_magic = [_]u8{ 'N', 'O', 'P', 'E', 0x01, 0x01 };
    try std.testing.expectError(
        error.BadMagic,
        format.deserialize(alloc, &bad_magic, .{}),
    );

    const bad_version = [_]u8{ 'B', 'R', 'P', 'G', 0x02, 0x01 };
    try std.testing.expectError(
        error.UnsupportedVersion,
        format.deserialize(alloc, &bad_version, .{}),
    );

    const overflow = [_]u8{
        'B',  'R',  'P',  'G',  0x01,
        0x02, 0x08, 0xff, 0xff, 0xff,
        0xff, 0xff, 0xff, 0xff, 0xff,
        0xff, 0x02,
    };
    try std.testing.expectError(
        error.IntegerOverflow,
        format.deserialize(alloc, &overflow, .{}),
    );

    const high_padding_bits = [_]u8{
        'B',  'R',  'P',  'G',  0x01,
        0x01, 0x03, 0x01, 0x02, 0x00,
        0x08,
    };
    try std.testing.expectError(
        error.InvalidLiteral,
        format.deserialize(alloc, &high_padding_bits, .{}),
    );
}

test "decoder calls typeOf and rejects an ill-typed complete tree" {
    const alloc = std.testing.allocator;
    const mismatched_concat = [_]u8{
        'B',  'R',  'P',  'G',  0x01,
        0x03, 0x02, 0x01, 0x08, 0x01,
        0x02, 0x00, 0x01, 0x01, 0x10,
        0x01, 0x03, 0x00, 0x02, 0x00,
    };
    try std.testing.expectError(
        error.InvalidProgram,
        format.deserialize(alloc, &mismatched_concat, .{}),
    );
}

test "decoder enforces node depth output and literal byte limits" {
    const alloc = std.testing.allocator;

    try std.testing.expectError(
        error.NodeLimitExceeded,
        format.deserialize(
            alloc,
            &all_kinds_golden,
            .{ .max_nodes = 7 },
        ),
    );
    try std.testing.expectError(
        error.DepthLimitExceeded,
        format.deserialize(
            alloc,
            &all_kinds_golden,
            .{ .max_depth = 3 },
        ),
    );
    try std.testing.expectError(
        error.OutputLimitExceeded,
        format.deserialize(
            alloc,
            &all_kinds_golden,
            .{ .max_output_bytes = 5 },
        ),
    );
    try std.testing.expectError(
        error.LiteralLimitExceeded,
        format.deserialize(
            alloc,
            &all_kinds_golden,
            .{ .max_literal_bytes = 2 },
        ),
    );
}

test "decoder defaults are finite and cumulative transform work is bounded" {
    const alloc = std.testing.allocator;
    const defaults = format.DecodeLimits{};
    try std.testing.expect(
        defaults.max_output_bytes < std.math.maxInt(usize),
    );
    try std.testing.expect(
        defaults.max_literal_bytes < std.math.maxInt(usize),
    );
    try std.testing.expect(
        defaults.max_execution_work_bytes < std.math.maxInt(usize),
    );

    const leaf = try dsl.Program.constant(8, 4, 0);
    var first_map = try dsl.Program.map(alloc, .gray, leaf);
    errdefer first_map.deinit(alloc);
    var second_map = try dsl.Program.map(alloc, .bit_reverse, first_map);
    defer second_map.deinit(alloc);

    try std.testing.expectEqual(
        @as(usize, 12),
        try second_map.executionWorkBytes(),
    );
    const bytes = try format.serialize(alloc, second_map);
    defer alloc.free(bytes);
    try std.testing.expectError(
        error.ExecutionWorkLimitExceeded,
        format.deserialize(
            alloc,
            bytes,
            .{
                .max_output_bytes = 4,
                .max_literal_bytes = 0,
                .max_execution_work_bytes = 11,
            },
        ),
    );
}

test "literal output limit precedes constant-size rANS decoding allocation" {
    // A one-symbol rANS stream has a four-byte payload no matter how large its
    // semantic count is. With the check in the wrong place this tiny program
    // attempts count-proportional allocations before reporting the limit.
    const million_zero_bytes = [_]u8{
        'B', 'R', 'P', 'G', 0x01,
        0x01, 0x08, // Lit<u8>
        0xc0, 0x84, 0x3d, // count = 1,000,000
        0x19, // literal body length = 25
        0x03, // rANS
        0x01, 0x00, 0x00, 0x00, // one table entry
        0x00, 0x00, 0x00, 0x00, // symbol 0
        0x00, 0x40, 0x00, 0x00, // frequency = RANS_PROB_SCALE
        0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // payload bytes
        0x00, 0x80, 0x00, 0x00, // final state = RANS_L
    };
    var no_allocation_storage: [1]u8 = undefined;
    var fixed = std.heap.FixedBufferAllocator.init(&no_allocation_storage);

    try std.testing.expectError(
        error.OutputLimitExceeded,
        format.deserialize(
            fixed.allocator(),
            &million_zero_bytes,
            .{ .max_output_bytes = 16 },
        ),
    );
}

test "literal storage arithmetic overflow is an output-limit failure" {
    const overflowing_literal = [_]u8{
        'B', 'R', 'P', 'G', 0x01,
        0x01, 0x20, // Lit<u32>
        0xff, 0xff, 0xff, 0xff, 0xff, // count = max u64
        0xff, 0xff, 0xff, 0xff, 0x01,
    };
    var no_allocation_storage: [1]u8 = undefined;
    var fixed = std.heap.FixedBufferAllocator.init(&no_allocation_storage);

    try std.testing.expectError(
        error.OutputLimitExceeded,
        format.deserialize(
            fixed.allocator(),
            &overflowing_literal,
            .{ .max_output_bytes = 16 },
        ),
    );
}

test "empty literal satisfies a zero output-byte limit" {
    const alloc = std.testing.allocator;
    var empty = try dsl.Program.literal(alloc, 32, &.{});
    defer empty.deinit(alloc);
    const bytes = try format.serialize(alloc, empty);
    defer alloc.free(bytes);

    var decoded = try format.deserialize(
        alloc,
        bytes,
        .{
            .max_output_bytes = 0,
            .max_literal_bytes = 0,
        },
    );
    defer decoded.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 0), (try decoded.typeOf()).len);
}

test "deterministic mutations fail safely or remain canonical" {
    const alloc = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x4252_5047_4655_5a5a);
    const random = prng.random();

    for (0..600) |iteration| {
        var storage: [all_kinds_golden.len + 8]u8 = @splat(0);
        @memcpy(storage[0..all_kinds_golden.len], &all_kinds_golden);
        var length = all_kinds_golden.len;

        if (iteration % 3 == 0) {
            length = random.uintLessThan(
                usize,
                all_kinds_golden.len + 9,
            );
            if (length > all_kinds_golden.len)
                random.bytes(storage[all_kinds_golden.len..length]);
        } else {
            const mutations = 1 + random.uintLessThan(usize, 3);
            for (0..mutations) |_| {
                const position = 4 + random.uintLessThan(
                    usize,
                    all_kinds_golden.len - 4,
                );
                storage[position] ^= @as(u8, 1) <<
                    @intCast(random.uintLessThan(u8, 8));
            }
        }

        if (format.deserialize(
            alloc,
            storage[0..length],
            .{
                .max_nodes = 64,
                .max_depth = 16,
                .max_output_bytes = 1024 * 1024,
                .max_literal_bytes = 1024 * 1024,
            },
        )) |decoded_value| {
            var decoded = decoded_value;
            defer decoded.deinit(alloc);
            const canonical = try format.serialize(alloc, decoded);
            defer alloc.free(canonical);
            try std.testing.expectEqualSlices(
                u8,
                storage[0..length],
                canonical,
            );
        } else |_| {}
    }
}

test "random recursive valid programs roundtrip canonically and execute identically" {
    const alloc = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5245_4355_5253_4956);
    const random = prng.random();
    const widths = [_]u8{ 4, 8, 16, 32 };

    for (0..112) |iteration| {
        const root_kind: u8 = @intCast(iteration % 7);
        const bits = widths[iteration % widths.len];
        const len: usize = if (root_kind == 3)
            2 * (1 + random.uintLessThan(usize, 4))
        else
            2 + random.uintLessThan(usize, 7);
        var program = try randomValidProgram(
            alloc,
            random,
            3,
            bits,
            len,
            root_kind,
        );
        defer program.deinit(alloc);
        try expectCanonicalRoundtrip(alloc, program);
    }
}

fn randomValidProgram(
    alloc: Allocator,
    random: std.Random,
    depth: usize,
    bits: u8,
    len: usize,
    forced_kind: ?u8,
) !dsl.Program {
    std.debug.assert(bits > 0 and bits <= 32);
    std.debug.assert(len > 0);
    const kind: u8 = forced_kind orelse if (depth == 0)
        random.uintLessThan(u8, 2)
    else
        random.uintLessThan(u8, 7);
    const mask = if (bits == 32)
        std.math.maxInt(u32)
    else
        (@as(u32, 1) << @intCast(bits)) - 1;

    return switch (kind) {
        0 => blk: {
            const words = try alloc.alloc(u32, len);
            defer alloc.free(words);
            for (words) |*word| word.* = random.int(u32) & mask;
            break :blk dsl.Program.literal(alloc, bits, words);
        },
        1 => dsl.Program.constant(
            bits,
            len,
            random.int(u32) & mask,
        ),
        2 => blk: {
            if (len < 2)
                break :blk randomValidProgram(
                    alloc,
                    random,
                    depth -| 1,
                    bits,
                    len,
                    4,
                );
            const split = 1 + random.uintLessThan(usize, len - 1);
            var left = try randomValidProgram(
                alloc,
                random,
                depth -| 1,
                bits,
                split,
                null,
            );
            defer left.deinit(alloc);
            var right = try randomValidProgram(
                alloc,
                random,
                depth -| 1,
                bits,
                len - split,
                null,
            );
            defer right.deinit(alloc);
            break :blk dsl.Program.concat(alloc, &.{ left, right });
        },
        3 => blk: {
            if (len % 2 != 0)
                break :blk randomValidProgram(
                    alloc,
                    random,
                    depth -| 1,
                    bits,
                    len,
                    4,
                );
            var child = try randomValidProgram(
                alloc,
                random,
                depth -| 1,
                bits,
                len / 2,
                null,
            );
            errdefer child.deinit(alloc);
            break :blk dsl.Program.repeat(alloc, 2, child);
        },
        4 => blk: {
            var child = try randomValidProgram(
                alloc,
                random,
                depth -| 1,
                bits,
                len,
                null,
            );
            errdefer child.deinit(alloc);
            const operation: dsl.MapOp = switch (random.uintLessThan(u8, 6)) {
                0 => .{ .xor = random.int(u32) & mask },
                1 => .{ .add_mod = random.int(u32) & mask },
                2 => .zigzag,
                3 => .gray,
                4 => .{
                    .rotate_left = random.uintLessThan(u8, bits),
                },
                else => .bit_reverse,
            };
            break :blk dsl.Program.map(alloc, operation, child);
        },
        5 => blk: {
            if (len < 2)
                break :blk randomValidProgram(
                    alloc,
                    random,
                    depth -| 1,
                    bits,
                    len,
                    4,
                );
            var updates = try randomValidProgram(
                alloc,
                random,
                depth -| 1,
                bits,
                len - 1,
                null,
            );
            errdefer updates.deinit(alloc);
            const operation: dsl.ScanOp = if (random.boolean())
                .xor
            else
                .add_mod;
            break :blk dsl.Program.scan(
                alloc,
                operation,
                random.int(u32) & mask,
                updates,
            );
        },
        6 => blk: {
            if (bits < 2)
                break :blk randomValidProgram(
                    alloc,
                    random,
                    depth -| 1,
                    bits,
                    len,
                    4,
                );
            const low_bits: u8 = 1 + random.uintLessThan(
                u8,
                bits - 1,
            );
            var low = try randomValidProgram(
                alloc,
                random,
                depth -| 1,
                low_bits,
                len,
                null,
            );
            defer low.deinit(alloc);
            var high = try randomValidProgram(
                alloc,
                random,
                depth -| 1,
                bits - low_bits,
                len,
                null,
            );
            defer high.deinit(alloc);
            break :blk dsl.Program.merge(
                alloc,
                .{ .fields = low_bits },
                &.{ low, high },
            );
        },
        else => unreachable,
    };
}

fn makeAllKindsProgram(alloc: Allocator) !dsl.Program {
    var repeated = try makeRepeatedMap(alloc);
    defer repeated.deinit(alloc);
    var scanned = try makeScannedMerge(alloc);
    defer scanned.deinit(alloc);
    return dsl.Program.concat(alloc, &.{ repeated, scanned });
}

fn makeRepeatedMap(alloc: Allocator) !dsl.Program {
    var mapped = try makeMap(alloc);
    errdefer mapped.deinit(alloc);
    return dsl.Program.repeat(alloc, 2, mapped);
}

fn makeMap(alloc: Allocator) !dsl.Program {
    var literal = try dsl.Program.literal(alloc, 8, &.{ 1, 2 });
    errdefer literal.deinit(alloc);
    return dsl.Program.map(alloc, .{ .xor = 1 }, literal);
}

fn makeScannedMerge(alloc: Allocator) !dsl.Program {
    var merged = try makeFieldMerge(alloc);
    errdefer merged.deinit(alloc);
    return dsl.Program.scan(alloc, .add_mod, 0, merged);
}

fn makeFieldMerge(alloc: Allocator) !dsl.Program {
    var low = try dsl.Program.constant(4, 1, 1);
    defer low.deinit(alloc);
    var high = try dsl.Program.literal(alloc, 4, &.{2});
    defer high.deinit(alloc);
    return dsl.Program.merge(
        alloc,
        .{ .fields = 4 },
        &.{ low, high },
    );
}

fn expectCanonicalRoundtrip(alloc: Allocator, program: dsl.Program) !void {
    const bytes = try format.serialize(alloc, program);
    defer alloc.free(bytes);
    try std.testing.expectEqual(
        bytes.len,
        try format.serializedSize(alloc, program),
    );

    var decoded = try format.deserialize(alloc, bytes, .{});
    defer decoded.deinit(alloc);
    const canonical = try format.serialize(alloc, decoded);
    defer alloc.free(canonical);
    try std.testing.expectEqualSlices(u8, bytes, canonical);
    try expectSameOutput(alloc, program, decoded);
}

fn expectSameOutput(
    alloc: Allocator,
    expected_program: dsl.Program,
    actual_program: dsl.Program,
) !void {
    var expected = try interpreter.execute(alloc, expected_program);
    defer expected.deinit(alloc);
    var actual = try interpreter.execute(alloc, actual_program);
    defer actual.deinit(alloc);

    try std.testing.expectEqual(expected.bits_per_elem, actual.bits_per_elem);
    try std.testing.expectEqual(expected.count, actual.count);
    try std.testing.expectEqualSlices(u8, expected.data, actual.data);
}
