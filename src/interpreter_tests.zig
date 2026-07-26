const std = @import("std");
const dsl = @import("dsl.zig");
const interpreter = @import("interpreter.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const Program = dsl.Program;
const Stream = types.Stream;

fn expectStreamsEqual(expected: Stream, actual: Stream) !void {
    try std.testing.expectEqual(expected.bits_per_elem, actual.bits_per_elem);
    try std.testing.expectEqual(expected.count, actual.count);
    for (0..expected.count) |i|
        try std.testing.expectEqual(expected.getU32(i), actual.getU32(i));
}

fn expectExecuteIntoMatches(alloc: Allocator, program: Program) !void {
    var allocated = try interpreter.execute(alloc, program);
    defer allocated.deinit(alloc);

    var caller_owned = try Stream.init(
        alloc,
        allocated.count,
        allocated.bits_per_elem,
    );
    defer caller_owned.deinit(alloc);
    @memset(caller_owned.data, 0xa5);

    try interpreter.executeInto(alloc, program, caller_owned);
    try expectStreamsEqual(allocated, caller_owned);
}

test "executeInto matches execute for every semantic DSL node" {
    const alloc = std.testing.allocator;

    {
        var program = try Program.literal(alloc, 8, &.{ 1, 2, 255 });
        defer program.deinit(alloc);
        try expectExecuteIntoMatches(alloc, program);
    }
    {
        var program = try Program.constant(16, 4, 0xbeef);
        defer program.deinit(alloc);
        try expectExecuteIntoMatches(alloc, program);
    }
    {
        var left = try Program.literal(alloc, 8, &.{ 1, 2 });
        defer left.deinit(alloc);
        var right = try Program.constant(8, 3, 9);
        defer right.deinit(alloc);
        var program = try Program.concat(alloc, &.{ left, right });
        defer program.deinit(alloc);
        try expectExecuteIntoMatches(alloc, program);
    }
    {
        const child = try Program.literal(alloc, 16, &.{ 0x1234, 0xabcd });
        var program = try Program.repeat(alloc, 3, child);
        defer program.deinit(alloc);
        try expectExecuteIntoMatches(alloc, program);
    }
    {
        const child = try Program.literal(alloc, 8, &.{ 0, 1, 0x7f, 0xff });
        var program = try Program.map(alloc, .bit_reverse, child);
        defer program.deinit(alloc);
        try expectExecuteIntoMatches(alloc, program);
    }
    {
        const child = try Program.literal(alloc, 8, &.{ 3, 6, 1 });
        var program = try Program.scan(alloc, .xor, 5, child);
        defer program.deinit(alloc);
        try expectExecuteIntoMatches(alloc, program);
    }
    {
        var low = try Program.literal(alloc, 4, &.{ 0xa, 0x5 });
        defer low.deinit(alloc);
        var high = try Program.literal(alloc, 4, &.{ 0x3, 0xc });
        defer high.deinit(alloc);
        var program = try Program.merge(
            alloc,
            .{ .fields = 4 },
            &.{ low, high },
        );
        defer program.deinit(alloc);
        try expectExecuteIntoMatches(alloc, program);
    }
}

test "executeInto rejects output type count and storage mismatches" {
    const alloc = std.testing.allocator;
    var program = try Program.constant(8, 2, 7);
    defer program.deinit(alloc);

    var wrong_type = try Stream.init(alloc, 2, 16);
    defer wrong_type.deinit(alloc);
    try std.testing.expectError(
        error.OutputTypeMismatch,
        interpreter.executeInto(alloc, program, wrong_type),
    );

    var wrong_count = try Stream.init(alloc, 3, 8);
    defer wrong_count.deinit(alloc);
    try std.testing.expectError(
        error.OutputLengthMismatch,
        interpreter.executeInto(alloc, program, wrong_count),
    );

    var storage = [_]u8{0};
    const too_small = Stream{
        .data = &storage,
        .count = 2,
        .bits_per_elem = 8,
        .owns_data = false,
    };
    try std.testing.expectError(
        error.OutputBufferTooSmall,
        interpreter.executeInto(alloc, program, too_small),
    );
}

test "executeInto emits the paper FP32 Repeat example bit exactly" {
    const alloc = std.testing.allocator;
    const period = try Program.literal(
        alloc,
        32,
        &.{ 0x3f80_0000, 0xbf80_0000 },
    );
    var program = try Program.repeat(alloc, 3, period);
    defer program.deinit(alloc);

    var output = try Stream.init(alloc, 6, 32);
    defer output.deinit(alloc);
    try interpreter.executeInto(alloc, program, output);

    for (0..3) |repetition| {
        try std.testing.expectEqual(
            @as(u32, 0x3f80_0000),
            output.getU32(repetition * 2),
        );
        try std.testing.expectEqual(
            @as(u32, 0xbf80_0000),
            output.getU32(repetition * 2 + 1),
        );
    }
}

test "Repeat expands a tiny period to a large output exactly" {
    const alloc = std.testing.allocator;
    const period = try Program.literal(alloc, 8, &.{0x5a});
    var program = try Program.repeat(alloc, 1_000_000, period);
    defer program.deinit(alloc);

    var output = try interpreter.execute(alloc, program);
    defer output.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 1_000_000), output.count);
    for (output.data) |byte|
        try std.testing.expectEqual(@as(u8, 0x5a), byte);
}

test "executeInto Scan consumes n minus one updates and writes n words" {
    const alloc = std.testing.allocator;
    const updates = try Program.literal(alloc, 8, &.{ 3, 6, 1 });
    var program = try Program.scan(alloc, .xor, 5, updates);
    defer program.deinit(alloc);

    var output = try Stream.init(alloc, 4, 8);
    defer output.deinit(alloc);
    try interpreter.executeInto(alloc, program, output);

    const expected = [_]u32{ 5, 6, 0, 1 };
    for (expected, 0..) |word, i|
        try std.testing.expectEqual(word, output.getU32(i));
}

test "executeInto packs every Merge form at exact physical bit positions" {
    const alloc = std.testing.allocator;

    {
        var low = try Program.literal(alloc, 4, &.{0xa});
        defer low.deinit(alloc);
        var high = try Program.literal(alloc, 4, &.{0x3});
        defer high.deinit(alloc);
        var program = try Program.merge(
            alloc,
            .{ .fields = 4 },
            &.{ low, high },
        );
        defer program.deinit(alloc);
        var output = try Stream.init(alloc, 1, 8);
        defer output.deinit(alloc);
        try interpreter.executeInto(alloc, program, output);
        try std.testing.expectEqual(@as(u32, 0x3a), output.getU32(0));
    }
    {
        var bit0 = try Program.literal(alloc, 1, &.{1});
        defer bit0.deinit(alloc);
        var bit1 = try Program.literal(alloc, 1, &.{0});
        defer bit1.deinit(alloc);
        var bit2 = try Program.literal(alloc, 1, &.{1});
        defer bit2.deinit(alloc);
        var program = try Program.merge(
            alloc,
            .bit_planes,
            &.{ bit0, bit1, bit2 },
        );
        defer program.deinit(alloc);
        var output = try Stream.init(alloc, 1, 3);
        defer output.deinit(alloc);
        try interpreter.executeInto(alloc, program, output);
        try std.testing.expectEqual(@as(u32, 0b101), output.getU32(0));
    }
    {
        var low = try Program.literal(alloc, 8, &.{0x12});
        defer low.deinit(alloc);
        var middle = try Program.literal(alloc, 8, &.{0x34});
        defer middle.deinit(alloc);
        var high = try Program.literal(alloc, 4, &.{0x5});
        defer high.deinit(alloc);
        var program = try Program.merge(
            alloc,
            .byte_planes,
            &.{ low, middle, high },
        );
        defer program.deinit(alloc);
        var output = try Stream.init(alloc, 1, 20);
        defer output.deinit(alloc);
        try interpreter.executeInto(alloc, program, output);
        try std.testing.expectEqual(@as(u32, 0x5_3412), output.getU32(0));
    }
    {
        var sign = try Program.literal(alloc, 1, &.{1});
        defer sign.deinit(alloc);
        var exponent = try Program.literal(alloc, 8, &.{0x7f});
        defer exponent.deinit(alloc);
        var mantissa = try Program.literal(alloc, 23, &.{0});
        defer mantissa.deinit(alloc);
        var program = try Program.merge(
            alloc,
            .{ .float_fields = .f32 },
            &.{ sign, exponent, mantissa },
        );
        defer program.deinit(alloc);
        var output = try Stream.init(alloc, 1, 32);
        defer output.deinit(alloc);
        try interpreter.executeInto(alloc, program, output);
        try std.testing.expectEqual(
            @as(u32, 0xbf80_0000),
            output.getU32(0),
        );
    }
}

test "executeTensorInto validates binding and writes caller storage" {
    const alloc = std.testing.allocator;
    const root = try Program.literal(
        alloc,
        32,
        &.{ 0x0000_0000, 0x8000_0000 },
    );
    var tensor_program = try dsl.TensorProgram.init(
        alloc,
        .f32,
        &.{2},
        root,
    );
    defer tensor_program.deinit(alloc);

    var output = try Stream.init(alloc, 2, 32);
    defer output.deinit(alloc);
    try interpreter.executeTensorInto(alloc, tensor_program, output);
    try std.testing.expectEqual(
        @as(u32, 0x0000_0000),
        output.getU32(0),
    );
    try std.testing.expectEqual(
        @as(u32, 0x8000_0000),
        output.getU32(1),
    );
}

test "non-Merge executeInto paths require no scratch allocation" {
    const alloc = std.testing.allocator;

    var left = try Program.literal(alloc, 8, &.{ 1, 2 });
    defer left.deinit(alloc);
    var right = try Program.constant(8, 2, 3);
    defer right.deinit(alloc);
    var concat = try Program.concat(alloc, &.{ left, right });
    defer concat.deinit(alloc);
    var mapped = try Program.map(
        alloc,
        .{ .xor = 0xff },
        try concat.clone(alloc),
    );
    defer mapped.deinit(alloc);
    var repeated = try Program.repeat(alloc, 2, try mapped.clone(alloc));
    defer repeated.deinit(alloc);
    var scanned = try Program.scan(
        alloc,
        .add_mod,
        1,
        try repeated.clone(alloc),
    );
    defer scanned.deinit(alloc);

    var output = try Stream.init(alloc, 9, 8);
    defer output.deinit(alloc);

    var no_memory: [0]u8 = .{};
    var fixed = std.heap.FixedBufferAllocator.init(&no_memory);
    try interpreter.executeInto(fixed.allocator(), scanned, output);
}
