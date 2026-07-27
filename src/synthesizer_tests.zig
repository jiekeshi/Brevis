//! Algorithm-1 acceptance tests for the semantic program synthesizer.

const std = @import("std");
const dsl = @import("dsl.zig");
const grammar = @import("grammar.zig");
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

fn expectLiteralOwnership(program: dsl.Program, owns_data: bool) !void {
    switch (program.kind) {
        .literal => |literal| try std.testing.expectEqual(
            owns_data,
            literal.owns_data,
        ),
        else => for (program.children) |child|
            try expectLiteralOwnership(child, owns_data),
    }
}

test "uniform scoring skips PHOG target contexts" {
    const alloc = std.testing.allocator;
    var target = try streamFromWords(
        alloc,
        8,
        &.{ 3, 7, 2, 9, 1, 8, 4, 6 },
    );
    defer target.deinit(alloc);

    const options: synthesizer.Options = .{
        .max_expansions = 8,
        .seed_float_fields = false,
    };
    synthesizer.testing.resetContextBuildCount();
    var uniform = try synthesizer.synthesize(alloc, target, .u8, options);
    defer uniform.deinit(alloc);
    try std.testing.expectEqual(
        @as(usize, 0),
        synthesizer.testing.contextBuildCount(),
    );

    const empty_prior = grammar_prior.Prior.empty;
    var guided_options = options;
    guided_options.rule_model = &empty_prior;
    synthesizer.testing.resetContextBuildCount();
    var guided = try synthesizer.synthesize(
        alloc,
        target,
        .u8,
        guided_options,
    );
    defer guided.deinit(alloc);

    try std.testing.expect(guided.expanded > 0);
    try std.testing.expectEqual(
        guided.expanded,
        synthesizer.testing.contextBuildCount(),
    );
    try std.testing.expectEqual(uniform.expanded, guided.expanded);
    try std.testing.expectEqual(
        uniform.completed_candidates,
        guided.completed_candidates,
    );
    try std.testing.expectEqual(uniform.status, guided.status);
    try std.testing.expectEqual(
        uniform.used_literal_fallback,
        guided.used_literal_fallback,
    );
    try std.testing.expectEqualSlices(
        u8,
        uniform.serialized_program,
        guided.serialized_program,
    );
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
    const serialized = try program_format.serialize(alloc, result.program);
    defer alloc.free(serialized);
    try std.testing.expectEqualSlices(
        u8,
        serialized,
        result.serialized_program,
    );
    var output = try interpreter.execute(alloc, result.program);
    defer output.deinit(alloc);
    try expectStreamsEqual(target, output);
}

test "owning and borrowing literal fallbacks have explicit lifetimes" {
    const alloc = std.testing.allocator;
    var target = try streamFromWords(alloc, 8, &.{ 4, 3, 2, 1 });
    var target_owned = true;
    defer if (target_owned) target.deinit(alloc);

    var owned = try synthesizer.synthesize(
        alloc,
        target,
        .u8,
        .{ .max_expansions = 0 },
    );
    defer owned.deinit(alloc);
    var borrowed = try synthesizer.synthesizeBorrowingTarget(
        alloc,
        target,
        .u8,
        .{ .max_expansions = 0 },
    );
    var borrowed_owned = true;
    defer if (borrowed_owned) borrowed.deinit(alloc);

    const owned_literal = switch (owned.program.kind) {
        .literal => |literal| literal,
        else => return error.TestExpectedEqual,
    };
    const borrowed_literal = switch (borrowed.program.kind) {
        .literal => |literal| literal,
        else => return error.TestExpectedEqual,
    };
    try std.testing.expect(owned_literal.owns_data);
    try std.testing.expect(!borrowed_literal.owns_data);
    try std.testing.expect(owned_literal.data.ptr != target.data.ptr);
    try std.testing.expectEqual(target.data.ptr, borrowed_literal.data.ptr);

    const owned_bytes = try program_format.serialize(alloc, owned.program);
    defer alloc.free(owned_bytes);
    const borrowed_bytes = try program_format.serialize(
        alloc,
        borrowed.program,
    );
    defer alloc.free(borrowed_bytes);
    try std.testing.expectEqualSlices(u8, owned_bytes, borrowed_bytes);

    borrowed.deinit(alloc);
    borrowed_owned = false;
    try std.testing.expectEqual(@as(u32, 4), target.getU32(0));
    target.deinit(alloc);
    target_owned = false;

    const owned_after_target = try program_format.serialize(
        alloc,
        owned.program,
    );
    defer alloc.free(owned_after_target);
    try std.testing.expectEqualSlices(u8, owned_bytes, owned_after_target);
}

test "borrowing synthesis keeps structured winners fully owned" {
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
    var target_owned = true;
    defer if (target_owned) target.deinit(alloc);

    const options: synthesizer.Options = .{
        .max_expansions = 64,
        .max_nodes = 8,
        .grammar_options = .{ .max_depth = 1 },
    };
    var owned = try synthesizer.synthesize(alloc, target, .f32, options);
    defer owned.deinit(alloc);
    var borrowed = try synthesizer.synthesizeBorrowingTarget(
        alloc,
        target,
        .f32,
        options,
    );
    defer borrowed.deinit(alloc);

    try std.testing.expect(switch (borrowed.program.kind) {
        .repeat => true,
        else => false,
    });
    try expectLiteralOwnership(borrowed.program, true);
    const owned_bytes = try program_format.serialize(alloc, owned.program);
    defer alloc.free(owned_bytes);
    const borrowed_bytes = try program_format.serialize(
        alloc,
        borrowed.program,
    );
    defer alloc.free(borrowed_bytes);
    try std.testing.expectEqualSlices(u8, owned_bytes, borrowed_bytes);

    target.deinit(alloc);
    target_owned = false;
    var output = try interpreter.execute(alloc, borrowed.program);
    defer output.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 6), output.count);
    try std.testing.expectEqual(@as(u32, 0x3f80_0000), output.getU32(0));
    try std.testing.expectEqual(@as(u32, 0xbf80_0000), output.getU32(5));
}

test "the final permitted expansion still contributes complete candidates" {
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
            .max_expansions = 1,
            .max_nodes = 4,
            .grammar_options = .{ .max_depth = 0 },
        },
    );
    defer result.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 1), result.expanded);
    try std.testing.expect(result.completed_candidates >= 1);
    try std.testing.expect(switch (result.program.kind) {
        .constant => true,
        else => false,
    });
}

test "bounded search seeds the existing shallow float-fields program" {
    const alloc = std.testing.allocator;
    var target = try types.Stream.init(alloc, 256, 32);
    defer target.deinit(alloc);
    for (0..target.count) |index| {
        const sign: u32 = @intCast((index & 1) << 31);
        const mantissa: u32 = @intCast(index * 7919);
        target.setU32(index, sign | 0x3f80_0000 | mantissa);
    }

    const options: synthesizer.Options = .{
        .max_expansions = 1,
        .max_nodes = 4,
        .grammar_options = .{
            .max_depth = 1,
            .max_repeat_period = 0,
            .max_concat_splits = 0,
            .max_map_constants = 0,
            .max_rotations = 0,
            .max_field_splits = 0,
        },
    };
    var seeded = try synthesizer.synthesize(alloc, target, .f32, options);
    defer seeded.deinit(alloc);

    var unseeded_options = options;
    unseeded_options.seed_float_fields = false;
    var unseeded = try synthesizer.synthesize(
        alloc,
        target,
        .f32,
        unseeded_options,
    );
    defer unseeded.deinit(alloc);

    try std.testing.expect(switch (seeded.program.kind) {
        .merge => |operation| switch (operation) {
            .float_fields => true,
            else => false,
        },
        else => false,
    });
    try std.testing.expect(switch (unseeded.program.kind) {
        .literal => true,
        else => false,
    });
    try std.testing.expect(seeded.serialized_bytes < unseeded.serialized_bytes);
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
    var target = try types.Stream.init(alloc, 64, 32);
    defer target.deinit(alloc);
    for (0..target.count) |index| {
        const sign: u32 = @intCast((index & 1) << 31);
        const mantissa: u32 = @intCast(index * 7919);
        target.setU32(index, sign | 0x3f80_0000 | mantissa);
    }

    var child_targets = try grammar.childTargets(
        alloc,
        .{ .merge_float_fields = .f32 },
        target,
        .f32,
    );
    defer child_targets.deinit(alloc);
    const children = try alloc.alloc(dsl.Program, child_targets.streams.len);
    var initialized: usize = 0;
    errdefer {
        for (children[0..initialized]) |*child| child.deinit(alloc);
        alloc.free(children);
    }
    for (child_targets.streams, children) |child_target, *child| {
        child.* = try dsl.Program.literalFromStream(alloc, child_target);
        initialized += 1;
    }
    var training_program = try dsl.Program.mergeOwned(
        .{ .float_fields = .f32 },
        children,
    );
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
        .max_expansions = 6,
        .max_nodes = 8,
        .seed_float_fields = false,
        .grammar_options = .{
            .max_depth = 1,
            .max_repeat_period = 0,
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
        .merge => |operation| switch (operation) {
            .float_fields => true,
            else => false,
        },
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
