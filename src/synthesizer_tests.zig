//! Algorithm-1 acceptance tests for the semantic program synthesizer.

const std = @import("std");
const dsl = @import("dsl.zig");
const grammar_prior = @import("grammar_prior.zig");
const interpreter = @import("interpreter.zig");
const program_format = @import("program_format.zig");
const synthesizer = @import("synthesizer.zig");
const types = @import("types.zig");

fn streamFromWords(
    alloc: std.mem.Allocator,
    bits: u8,
    words: []const u32,
) !types.Stream {
    var stream = try types.Stream.init(alloc, words.len, bits);
    for (words, 0..) |word, index| stream.setU32(index, word);
    return stream;
}

fn expectStreamsEqual(expected: types.Stream, actual: types.Stream) !void {
    try std.testing.expectEqual(expected.bits_per_elem, actual.bits_per_elem);
    try std.testing.expectEqual(expected.count, actual.count);
    for (0..expected.count) |index|
        try std.testing.expectEqual(expected.getU32(index), actual.getU32(index));
}

test "zero expansion budget returns the exact full literal incumbent" {
    const alloc = std.testing.allocator;
    var target = try streamFromWords(alloc, 8, &.{ 4, 3, 2, 1 });
    defer target.deinit(alloc);

    var result = try synthesizer.synthesize(
        alloc,
        target,
        .u8,
        .{ .max_expansions = 0 },
    );
    defer result.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 0), result.expanded);
    try std.testing.expectEqual(
        synthesizer.SearchStatus.budget_exhausted,
        result.status,
    );
    try std.testing.expect(result.used_literal_fallback);
    try std.testing.expectEqual(
        result.serialized_bytes,
        try program_format.serializedSize(alloc, result.program),
    );
    var output = try interpreter.execute(alloc, result.program);
    defer output.deinit(alloc);
    try expectStreamsEqual(target, output);
}

test "bounded A-star selects the paper Repeat program by canonical bytes" {
    const alloc = std.testing.allocator;
    var target = try streamFromWords(
        alloc,
        32,
        &.{
            0x3f80_0000,
            0xbf80_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x3f80_0000,
            0xbf80_0000,
        },
    );
    defer target.deinit(alloc);

    var result = try synthesizer.synthesize(
        alloc,
        target,
        .f32,
        .{
            .max_expansions = 64,
            .max_nodes = 8,
            .grammar_options = .{ .max_depth = 1 },
        },
    );
    defer result.deinit(alloc);

    switch (result.program.kind) {
        .repeat => |times| try std.testing.expectEqual(@as(u32, 3), times),
        else => return error.TestExpectedEqual,
    }
    try std.testing.expect(!result.used_literal_fallback);
    var literal = try dsl.Program.literalFromStream(alloc, target);
    defer literal.deinit(alloc);
    try std.testing.expect(
        result.serialized_bytes <
            try program_format.serializedSize(alloc, literal),
    );
    var output = try interpreter.execute(alloc, result.program);
    defer output.deinit(alloc);
    try expectStreamsEqual(target, output);
}

test "search continues after its first completion and selects Const by bytes" {
    const alloc = std.testing.allocator;
    var target = try streamFromWords(
        alloc,
        8,
        &.{ 7, 7, 7, 7, 7, 7, 7, 7 },
    );
    defer target.deinit(alloc);

    var result = try synthesizer.synthesize(
        alloc,
        target,
        .u8,
        .{
            .max_expansions = 32,
            .max_nodes = 4,
            .grammar_options = .{ .max_depth = 0 },
        },
    );
    defer result.deinit(alloc);

    try std.testing.expect(switch (result.program.kind) {
        .constant => true,
        else => false,
    });
    try std.testing.expect(result.completed_candidates >= 1);
    try std.testing.expectEqual(
        synthesizer.SearchStatus.proven_optimal,
        result.status,
    );
    var output = try interpreter.execute(alloc, result.program);
    defer output.deinit(alloc);
    try expectStreamsEqual(target, output);
}

test "empty target remains a typed exact literal and dtype mismatches fail" {
    const alloc = std.testing.allocator;
    var target = try types.Stream.init(alloc, 0, 16);
    defer target.deinit(alloc);

    var result = try synthesizer.synthesize(
        alloc,
        target,
        .f16,
        .{ .max_expansions = 8 },
    );
    defer result.deinit(alloc);
    try std.testing.expect(switch (result.program.kind) {
        .literal => true,
        else => false,
    });
    var output = try interpreter.execute(alloc, result.program);
    defer output.deinit(alloc);
    try expectStreamsEqual(target, output);

    try std.testing.expectError(
        error.TensorWidthMismatch,
        synthesizer.synthesize(alloc, target, .f32, .{}),
    );
}

test "PHOG changes bounded exploration order but exact bytes still select" {
    const alloc = std.testing.allocator;
    var target = try streamFromWords(
        alloc,
        32,
        &.{
            0x3f80_0000,
            0xbf80_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x3f80_0000,
            0xbf80_0000,
        },
    );
    defer target.deinit(alloc);

    const period = try dsl.Program.literal(
        alloc,
        32,
        &.{ 0x3f80_0000, 0xbf80_0000 },
    );
    var training_program = try dsl.Program.repeat(alloc, 3, period);
    defer training_program.deinit(alloc);

    var counts = grammar_prior.Counts.init();
    defer counts.deinit(alloc);
    try counts.observeProgram(
        alloc,
        training_program,
        target,
        .f32,
        100,
    );
    var learned = try counts.toPrior(alloc, .{
        .learned_numerator = 19,
        .learned_denominator = 20,
    });
    defer learned.deinit(alloc);

    const bounded_options: synthesizer.Options = .{
        .max_expansions = 3,
        .max_nodes = 4,
        .grammar_options = .{
            .max_depth = 1,
            .max_repeat_period = 8,
            .max_concat_splits = 0,
            .max_map_constants = 0,
            .max_rotations = 0,
            .max_field_splits = 0,
        },
    };
    var uniform = try synthesizer.synthesize(
        alloc,
        target,
        .f32,
        bounded_options,
    );
    defer uniform.deinit(alloc);

    var guided_options = bounded_options;
    guided_options.rule_model = &learned;
    var guided = try synthesizer.synthesize(
        alloc,
        target,
        .f32,
        guided_options,
    );
    defer guided.deinit(alloc);

    try std.testing.expect(switch (uniform.program.kind) {
        .literal => true,
        else => false,
    });
    try std.testing.expect(switch (guided.program.kind) {
        .repeat => true,
        else => false,
    });
    try std.testing.expect(guided.serialized_bytes < uniform.serialized_bytes);

    var guided_output = try interpreter.execute(alloc, guided.program);
    defer guided_output.deinit(alloc);
    try expectStreamsEqual(target, guided_output);

    var exhaustive_options = bounded_options;
    exhaustive_options.max_expansions = 128;
    var uniform_exhaustive = try synthesizer.synthesize(
        alloc,
        target,
        .f32,
        exhaustive_options,
    );
    defer uniform_exhaustive.deinit(alloc);
    exhaustive_options.rule_model = &learned;
    var guided_exhaustive = try synthesizer.synthesize(
        alloc,
        target,
        .f32,
        exhaustive_options,
    );
    defer guided_exhaustive.deinit(alloc);
    try std.testing.expectEqual(
        uniform_exhaustive.serialized_bytes,
        guided_exhaustive.serialized_bytes,
    );
}

test "decomposition budget prunes amplifying targets but keeps Lit complete" {
    const alloc = std.testing.allocator;
    var target = try types.Stream.init(alloc, 16, 32);
    defer target.deinit(alloc);
    for (0..target.count) |index|
        target.setU32(index, @intCast(index));

    var result = try synthesizer.synthesize(
        alloc,
        target,
        .u32,
        .{
            .max_expansions = 16,
            .max_decomposition_bytes = target.data.len,
            .grammar_options = .{
                .max_repeat_period = 0,
                .max_concat_splits = 0,
                .max_map_constants = 0,
                .max_rotations = 0,
                .max_field_splits = 0,
            },
        },
    );
    defer result.deinit(alloc);
    var decoded = try interpreter.execute(alloc, result.program);
    defer decoded.deinit(alloc);
    try expectStreamsEqual(target, decoded);
}
