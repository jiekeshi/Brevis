//! Acceptance tests derived directly from the paper's semantic DSL.

const std = @import("std");
const dsl = @import("dsl.zig");
const decomposition = @import("decomposition.zig");
const interpreter = @import("interpreter.zig");
const semantics = @import("semantics.zig");
const types = @import("types.zig");

fn streamFromWords(
    alloc: std.mem.Allocator,
    bits: u8,
    words: []const u32,
) !types.Stream {
    var stream = try types.Stream.init(alloc, words.len, bits);
    for (words, 0..) |word, i| stream.setU32(i, word);
    return stream;
}

fn literalFromStream(alloc: std.mem.Allocator, stream: types.Stream) !dsl.Program {
    const words = try alloc.alloc(u32, stream.count);
    defer alloc.free(words);
    for (words, 0..) |*word, i| word.* = stream.getU32(i);
    return dsl.Program.literal(alloc, stream.bits_per_elem, words);
}

fn expectStreamsEqual(expected: types.Stream, actual: types.Stream) !void {
    try std.testing.expectEqual(expected.bits_per_elem, actual.bits_per_elem);
    try std.testing.expectEqual(expected.count, actual.count);
    for (0..expected.count) |i|
        try std.testing.expectEqual(expected.getU32(i), actual.getU32(i));
}

test "paper example: Repeat emits the exact FP32 physical words" {
    const alloc = std.testing.allocator;
    const period_words = [_]u32{ 0x3f80_0000, 0xbf80_0000 };

    const period = try dsl.Program.literal(alloc, 32, &period_words);
    var repeated = try dsl.Program.repeat(alloc, 3, period);
    defer repeated.deinit(alloc);

    var output = try interpreter.execute(alloc, repeated);
    defer output.deinit(alloc);

    try std.testing.expectEqual(@as(u8, 32), output.bits_per_elem);
    try std.testing.expectEqual(@as(usize, 6), output.count);
    for (0..3) |i| {
        try std.testing.expectEqual(@as(u32, 0x3f80_0000), output.getU32(i * 2));
        try std.testing.expectEqual(@as(u32, 0xbf80_0000), output.getU32(i * 2 + 1));
    }
}

test "zero-sized tensor shape short-circuits otherwise overflowing dimensions" {
    const alloc = std.testing.allocator;
    var root = try dsl.Program.literal(alloc, 8, &.{});
    var root_owned = true;
    defer if (root_owned) root.deinit(alloc);

    var tensor = try dsl.TensorProgram.init(
        alloc,
        .u8,
        &.{ std.math.maxInt(u64), 2, 0 },
        root,
    );
    root_owned = false;
    defer tensor.deinit(alloc);
    const tensor_type = try tensor.validate();
    try std.testing.expectEqual(@as(usize, 0), tensor_type.elements);
}

test "Const and Concat generate adjacent typed regions" {
    const alloc = std.testing.allocator;

    var prefix = try dsl.Program.constant(8, 3, 7);
    defer prefix.deinit(alloc);
    var suffix = try dsl.Program.literal(alloc, 8, &.{ 1, 2 });
    defer suffix.deinit(alloc);

    var combined = try dsl.Program.concat(alloc, &.{ prefix, suffix });
    defer combined.deinit(alloc);
    var output = try interpreter.execute(alloc, combined);
    defer output.deinit(alloc);

    const expected = [_]u32{ 7, 7, 7, 1, 2 };
    try std.testing.expectEqual(expected.len, output.count);
    for (expected, 0..) |word, i|
        try std.testing.expectEqual(word, output.getU32(i));
}

test "Map applies concrete width-preserving bijections pointwise" {
    const alloc = std.testing.allocator;
    const input = [_]u32{ 0x00, 0x01, 0x7f, 0x80, 0xff };
    const cases = [_]struct {
        op: dsl.MapOp,
        expected: [input.len]u32,
    }{
        .{ .op = .{ .xor = 0x5a }, .expected = .{ 0x5a, 0x5b, 0x25, 0xda, 0xa5 } },
        .{ .op = .{ .add_mod = 7 }, .expected = .{ 0x07, 0x08, 0x86, 0x87, 0x06 } },
        .{ .op = .zigzag, .expected = .{ 0x00, 0x02, 0xfe, 0xff, 0x01 } },
        .{ .op = .gray, .expected = .{ 0x00, 0x01, 0x40, 0xc0, 0x80 } },
        .{ .op = .{ .rotate_left = 3 }, .expected = .{ 0x00, 0x08, 0xfb, 0x04, 0xff } },
        .{ .op = .bit_reverse, .expected = .{ 0x00, 0x80, 0xfe, 0x01, 0xff } },
    };

    for (cases) |case| {
        const literal = try dsl.Program.literal(alloc, 8, &input);
        var mapped = try dsl.Program.map(alloc, case.op, literal);
        defer mapped.deinit(alloc);
        var output = try interpreter.execute(alloc, mapped);
        defer output.deinit(alloc);
        for (case.expected, 0..) |word, i|
            try std.testing.expectEqual(word, output.getU32(i));
    }
}

test "Scan stores the initial word and consumes exactly n-1 updates" {
    const alloc = std.testing.allocator;
    const updates = [_]u32{ 3, 6, 1 };
    const cases = [_]struct {
        op: dsl.ScanOp,
        expected: [4]u32,
    }{
        .{ .op = .xor, .expected = .{ 5, 6, 0, 1 } },
        .{ .op = .add_mod, .expected = .{ 5, 8, 14, 15 } },
    };

    for (cases) |case| {
        const child = try dsl.Program.literal(alloc, 8, &updates);
        var scanned = try dsl.Program.scan(alloc, case.op, 5, child);
        defer scanned.deinit(alloc);
        const output_type = try scanned.typeOf();
        try std.testing.expectEqual(@as(usize, updates.len + 1), output_type.len);

        var output = try interpreter.execute(alloc, scanned);
        defer output.deinit(alloc);
        for (case.expected, 0..) |word, i|
            try std.testing.expectEqual(word, output.getU32(i));
    }
}

test "Merge recomposes FP32 fields without numerical conversion" {
    const alloc = std.testing.allocator;
    var signs = try dsl.Program.literal(alloc, 1, &.{ 0, 1 });
    defer signs.deinit(alloc);
    var exponents = try dsl.Program.literal(alloc, 8, &.{ 0x7f, 0x7f });
    defer exponents.deinit(alloc);
    var mantissas = try dsl.Program.literal(alloc, 23, &.{ 0, 0 });
    defer mantissas.deinit(alloc);

    var merged = try dsl.Program.merge(
        alloc,
        .{ .float_fields = .f32 },
        &.{ signs, exponents, mantissas },
    );
    defer merged.deinit(alloc);
    const output_type = try merged.typeOf();
    try std.testing.expectEqual(@as(u8, 32), output_type.bits);
    try std.testing.expectEqual(@as(usize, 2), output_type.len);

    var output = try interpreter.execute(alloc, merged);
    defer output.deinit(alloc);
    try std.testing.expectEqual(@as(u32, 0x3f80_0000), output.getU32(0));
    try std.testing.expectEqual(@as(u32, 0xbf80_0000), output.getU32(1));
}

test "target-directed Repeat accepts only exact copies" {
    const alloc = std.testing.allocator;
    var repeated_target = try streamFromWords(alloc, 8, &.{ 1, 2, 1, 2, 1, 2 });
    defer repeated_target.deinit(alloc);

    var period = (try decomposition.repeat(alloc, repeated_target, 3)).?;
    defer period.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 2), period.count);
    try std.testing.expectEqual(@as(u32, 1), period.getU32(0));
    try std.testing.expectEqual(@as(u32, 2), period.getU32(1));

    var non_repeating = try streamFromWords(alloc, 8, &.{ 1, 2, 1, 3 });
    defer non_repeating.deinit(alloc);
    try std.testing.expect((try decomposition.repeat(alloc, non_repeating, 2)) == null);
}

test "Map Scan and Merge decompositions satisfy the reconstruction contract" {
    const alloc = std.testing.allocator;

    var map_target = try streamFromWords(alloc, 8, &.{ 0x5a, 0x5b, 0x25 });
    defer map_target.deinit(alloc);
    var map_child_target = try decomposition.map(alloc, map_target, .{ .xor = 0x5a });
    defer map_child_target.deinit(alloc);
    const map_child = try literalFromStream(alloc, map_child_target);
    var map_program = try dsl.Program.map(alloc, .{ .xor = 0x5a }, map_child);
    defer map_program.deinit(alloc);
    var map_output = try interpreter.execute(alloc, map_program);
    defer map_output.deinit(alloc);
    try expectStreamsEqual(map_target, map_output);

    var scan_target = try streamFromWords(alloc, 8, &.{ 5, 6, 0, 1 });
    defer scan_target.deinit(alloc);
    var scan_parts = (try decomposition.scan(alloc, scan_target, .xor)).?;
    defer scan_parts.deinit(alloc);
    const scan_child = try literalFromStream(alloc, scan_parts.updates);
    var scan_program = try dsl.Program.scan(alloc, .xor, scan_parts.initial, scan_child);
    defer scan_program.deinit(alloc);
    var scan_output = try interpreter.execute(alloc, scan_program);
    defer scan_output.deinit(alloc);
    try expectStreamsEqual(scan_target, scan_output);

    var merge_target = try streamFromWords(
        alloc,
        32,
        &.{ 0x3f80_0000, 0xbf80_0000 },
    );
    defer merge_target.deinit(alloc);
    const merge_targets = try decomposition.merge(
        alloc,
        merge_target,
        .{ .float_fields = .f32 },
    );
    defer {
        for (merge_targets) |*target| target.deinit(alloc);
        alloc.free(merge_targets);
    }
    const merge_children = try alloc.alloc(dsl.Program, merge_targets.len);
    defer {
        for (merge_children) |*child| child.deinit(alloc);
        alloc.free(merge_children);
    }
    for (merge_targets, 0..) |target, i|
        merge_children[i] = try literalFromStream(alloc, target);
    var merge_program = try dsl.Program.merge(
        alloc,
        .{ .float_fields = .f32 },
        merge_children,
    );
    defer merge_program.deinit(alloc);
    var merge_output = try interpreter.execute(alloc, merge_program);
    defer merge_output.deinit(alloc);
    try expectStreamsEqual(merge_target, merge_output);
}

test "prepared map and scan bulk paths match scalar semantics" {
    const alloc = std.testing.allocator;
    var random = std.Random.DefaultPrng.init(0x62d5_970d_8b9c_31f4);
    const widths = [_]u8{ 1, 7, 8, 9, 16, 23, 32 };
    const counts = [_]usize{ 2, 3, 15, 16, 17, 33 };

    for (widths) |bits| {
        const mask = if (bits == 32)
            std.math.maxInt(u32)
        else
            (@as(u32, 1) << @intCast(bits)) - 1;
        const operations = [_]dsl.MapOp{
            .{ .xor = mask / 3 },
            .{ .add_mod = mask / 5 },
            .zigzag,
            .gray,
            .{ .rotate_left = bits / 2 },
            .bit_reverse,
        };

        for (counts) |count| {
            const byte_len = count * types.roundUpToPow2(bits) / 8;
            const source_storage = try alloc.alloc(u8, byte_len + 1);
            defer alloc.free(source_storage);
            var source = types.Stream{
                .data = source_storage[1..],
                .count = count,
                .bits_per_elem = bits,
                .owns_data = false,
            };
            for (0..count) |i| source.setU32(i, random.random().int(u32) & mask);

            for (operations) |operation| {
                const prepared = try semantics.PreparedMap.init(operation, bits);
                const mapped_storage = try alloc.alloc(u8, byte_len + 1);
                defer alloc.free(mapped_storage);
                var mapped = types.Stream{
                    .data = mapped_storage[1..],
                    .count = count,
                    .bits_per_elem = bits,
                    .owns_data = false,
                };
                @memcpy(mapped.data, source.data);
                prepared.forwardInPlace(&mapped, 0, count);
                for (0..count) |i|
                    try std.testing.expectEqual(
                        prepared.forwardWord(source.getU32(i)),
                        mapped.getU32(i),
                    );

                const inverse_storage = try alloc.alloc(u8, byte_len + 1);
                defer alloc.free(inverse_storage);
                var inverse = types.Stream{
                    .data = inverse_storage[1..],
                    .count = count,
                    .bits_per_elem = bits,
                    .owns_data = false,
                };
                prepared.inverseInto(mapped, &inverse);
                try expectStreamsEqual(source, inverse);
            }

            const update_len = (count - 1) * source.elemBytes();
            for ([_]dsl.ScanOp{ .xor, .add_mod }) |operation| {
                const prepared = try semantics.PreparedScan.init(operation, bits);
                const update_storage = try alloc.alloc(u8, update_len + 1);
                defer alloc.free(update_storage);
                var updates = types.Stream{
                    .data = update_storage[1..],
                    .count = count - 1,
                    .bits_per_elem = bits,
                    .owns_data = false,
                };
                prepared.updatesInto(source, &updates);
                for (0..updates.count) |i|
                    try std.testing.expectEqual(
                        prepared.updateWord(
                            source.getU32(i),
                            source.getU32(i + 1),
                        ),
                        updates.getU32(i),
                    );
            }
        }
    }
}

test "a TensorProgram binds dtype and shape to one complete generator" {
    const alloc = std.testing.allocator;
    const root = try dsl.Program.literal(alloc, 32, &.{ 0x0000_0000, 0x8000_0000 });
    var tensor_program = try dsl.TensorProgram.init(alloc, .f32, &.{2}, root);
    defer tensor_program.deinit(alloc);

    const tensor_type = try tensor_program.validate();
    try std.testing.expectEqual(types.Dtype.f32, tensor_type.dtype);
    try std.testing.expectEqual(@as(usize, 2), tensor_type.elements);
    var output = try interpreter.executeTensor(alloc, tensor_program);
    defer output.deinit(alloc);
    try std.testing.expectEqual(@as(u32, 0x0000_0000), output.getU32(0));
    try std.testing.expectEqual(@as(u32, 0x8000_0000), output.getU32(1));

    var wrong_root = try dsl.Program.constant(32, 1, 0);
    defer wrong_root.deinit(alloc);
    try std.testing.expectError(
        error.TensorLengthMismatch,
        dsl.TensorProgram.init(alloc, .f32, &.{2}, wrong_root),
    );
}

test "empty tensors use an exact empty Lit extension" {
    const alloc = std.testing.allocator;
    const root = try dsl.Program.literal(alloc, 16, &.{});
    var tensor_program = try dsl.TensorProgram.init(
        alloc,
        .f16,
        &.{ 2, 0, 7 },
        root,
    );
    defer tensor_program.deinit(alloc);

    const tensor_type = try tensor_program.validate();
    try std.testing.expectEqual(@as(usize, 0), tensor_type.elements);
    var output = try interpreter.executeTensor(alloc, tensor_program);
    defer output.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 0), output.count);
    try std.testing.expectEqual(@as(usize, 0), output.data.len);
}

test "empty Lit is the unique zero-length production" {
    const alloc = std.testing.allocator;
    var first = try dsl.Program.literal(alloc, 8, &.{});
    defer first.deinit(alloc);
    var second = try dsl.Program.literal(alloc, 8, &.{});
    defer second.deinit(alloc);

    try std.testing.expectError(
        error.InvalidLength,
        dsl.Program.concat(alloc, &.{ first, second }),
    );

    var nonempty = try dsl.Program.literal(alloc, 8, &.{1});
    defer nonempty.deinit(alloc);
    try std.testing.expectError(
        error.InvalidLength,
        dsl.Program.concat(alloc, &.{ first, nonempty }),
    );
    try std.testing.expectError(
        error.InvalidLength,
        dsl.Program.concat(alloc, &.{ nonempty, second }),
    );

    var repeat_child = try first.clone(alloc);
    defer repeat_child.deinit(alloc);
    try std.testing.expectError(
        error.InvalidLength,
        dsl.Program.repeat(alloc, 2, repeat_child),
    );

    var map_child = try first.clone(alloc);
    defer map_child.deinit(alloc);
    try std.testing.expectError(
        error.InvalidLength,
        dsl.Program.map(alloc, .gray, map_child),
    );

    try std.testing.expectError(
        error.InvalidLength,
        dsl.Program.merge(
            alloc,
            .{ .fields = 8 },
            &.{ first, second },
        ),
    );
}

test "malformed Lit storage and high bits are rejected" {
    var extra_storage = [_]u8{ 1, 2 };
    const extra_program: dsl.Program = .{ .kind = .{ .literal = .{
        .data = &extra_storage,
        .count = 1,
        .bits_per_elem = 8,
        .owns_data = false,
    } } };
    try std.testing.expectError(
        error.InvalidLiteralValue,
        extra_program.typeOf(),
    );

    var high_bits = [_]u8{8};
    const high_bits_program: dsl.Program = .{ .kind = .{ .literal = .{
        .data = &high_bits,
        .count = 1,
        .bits_per_elem = 3,
        .owns_data = false,
    } } };
    try std.testing.expectError(
        error.InvalidLiteralValue,
        high_bits_program.typeOf(),
    );
}
