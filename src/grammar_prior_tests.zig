//! Contract tests for the PHOG-inspired prior over the paper DSL grammar.

const std = @import("std");
const dsl = @import("dsl.zig");
const grammar = @import("grammar.zig");
const prior_mod = @import("grammar_prior.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;

fn streamFromWords(
    alloc: Allocator,
    bits: u8,
    words: []const u32,
) !types.Stream {
    var stream = try types.Stream.init(alloc, words.len, bits);
    for (words, 0..) |word, index| stream.setU32(index, word);
    return stream;
}

test "the default prior uses the fully learned smoothed distribution" {
    const config: prior_mod.Config = .{};
    try std.testing.expectEqual(@as(u32, 1), config.learned_numerator);
    try std.testing.expectEqual(@as(u32, 1), config.learned_denominator);
    try std.testing.expectEqual(@as(u64, 1), config.smoothing);
}

test "an untrained prior is strictly uniform over the admitted productions" {
    const alloc = std.testing.allocator;
    var target = try streamFromWords(alloc, 8, &.{ 1, 2, 1, 2 });
    defer target.deinit(alloc);

    const context = prior_mod.Context.fromTarget(
        target,
        .u8,
        null,
        0,
        0,
    );
    const admitted = [_]grammar.ProductionId{
        .literal,
        .repeat,
        .concat,
        .map_xor,
    };
    var costs: [admitted.len]prior_mod.Cost = undefined;
    try prior_mod.Prior.empty.scoreSet(context, &admitted, &costs);

    const expected = prior_mod.uniformCost(admitted.len);
    try std.testing.expectEqual(@as(prior_mod.Cost, 2048), expected);
    for (costs) |cost| try std.testing.expectEqual(expected, cost);
}

test "observations alter costs and cannot make an unadmitted production legal" {
    const alloc = std.testing.allocator;
    var target = try streamFromWords(alloc, 8, &.{ 3, 7, 3, 7 });
    defer target.deinit(alloc);
    const context = prior_mod.Context.fromTarget(
        target,
        .u8,
        null,
        0,
        0,
    );

    var counts = prior_mod.Counts.init();
    defer counts.deinit(alloc);
    try counts.observe(alloc, context, .repeat, 100);
    try counts.observe(alloc, context, .literal, 1);
    var learned = try counts.toPrior(alloc, .{});
    defer learned.deinit(alloc);

    const admitted = [_]grammar.ProductionId{ .literal, .repeat, .concat };
    var costs: [admitted.len]prior_mod.Cost = undefined;
    try learned.scoreSet(context, &admitted, &costs);
    try std.testing.expect(costs[1] < costs[0]);
    try std.testing.expect(costs[1] < costs[2]);

    try std.testing.expectEqual(
        @as(?prior_mod.Cost, null),
        try learned.expansionCost(context, .constant, &admitted),
    );
}

test "relaxed contextual bounds stay below concrete PHOG costs" {
    const alloc = std.testing.allocator;
    var target = try streamFromWords(alloc, 8, &.{ 3, 7, 3, 7 });
    defer target.deinit(alloc);
    const context = prior_mod.Context.fromTarget(
        target,
        .u8,
        null,
        0,
        0,
    );

    var counts = prior_mod.Counts.init();
    defer counts.deinit(alloc);
    try counts.observe(alloc, context, .repeat, 100);
    try counts.observe(alloc, context, .literal, 2);
    var learned = try counts.toPrior(alloc, .{
        .learned_numerator = 19,
        .learned_denominator = 20,
    });
    defer learned.deinit(alloc);

    const admitted = [_]grammar.ProductionId{
        .literal,
        .repeat,
        .concat,
        .map_xor,
        .map_gray,
        .scan_xor,
    };
    var concrete: [admitted.len]prior_mod.Cost = undefined;
    try learned.scoreSet(context, &admitted, &concrete);

    var relaxed: [prior_mod.PRODUCTION_COUNT]prior_mod.Cost = undefined;
    try learned.contextualCostLowerBounds(admitted.len, &relaxed);
    for (admitted, concrete) |production, actual| {
        const index = for (prior_mod.PRODUCTIONS, 0..) |known, candidate| {
            if (known == production) break candidate;
        } else unreachable;
        try std.testing.expect(relaxed[index] <= actual);
    }

    const repeat_index = for (prior_mod.PRODUCTIONS, 0..) |known, index| {
        if (known == .repeat) break index;
    } else unreachable;
    const concat_index = for (prior_mod.PRODUCTIONS, 0..) |known, index| {
        if (known == .concat) break index;
    } else unreachable;
    try std.testing.expect(relaxed[repeat_index] < relaxed[concat_index]);
}

test "the prior backs off from data features and then to tree position" {
    const alloc = std.testing.allocator;
    var target = try streamFromWords(alloc, 16, &.{ 0, 1, 0, 1 });
    defer target.deinit(alloc);
    const trained = prior_mod.Context.fromTarget(
        target,
        .u16,
        .concat,
        1,
        2,
    );

    var counts = prior_mod.Counts.init();
    defer counts.deinit(alloc);
    try counts.observe(alloc, trained, .repeat, 200);
    var learned = try counts.toPrior(alloc, .{
        .learned_numerator = 19,
        .learned_denominator = 20,
    });
    defer learned.deinit(alloc);

    const admitted = [_]grammar.ProductionId{ .literal, .repeat };

    var feature_backoff = trained;
    feature_backoff.zero_bucket = (trained.zero_bucket + 1) % 4;
    var costs: [2]prior_mod.Cost = undefined;
    try learned.scoreSet(feature_backoff, &admitted, &costs);
    try std.testing.expect(costs[1] < costs[0]);

    var structural_backoff = feature_backoff;
    structural_backoff.depth_bucket =
        (trained.depth_bucket + 1) % (prior_mod.MAX_DEPTH_BUCKET + 1);
    structural_backoff.dtype = .f32;
    structural_backoff.target_bits = 32;
    try learned.scoreSet(structural_backoff, &admitted, &costs);
    try std.testing.expect(costs[1] < costs[0]);

    var no_match = structural_backoff;
    no_match.child_slot = 2;
    try learned.scoreSet(no_match, &admitted, &costs);
    try std.testing.expectEqual(costs[0], costs[1]);
}

test "context includes deterministic normalized difference entropy" {
    const alloc = std.testing.allocator;
    var linear = try streamFromWords(
        alloc,
        8,
        &.{ 1, 2, 3, 4, 5, 6, 7, 8 },
    );
    defer linear.deinit(alloc);
    var varied = try streamFromWords(
        alloc,
        8,
        &.{ 1, 2, 4, 8, 16, 31, 63, 127 },
    );
    defer varied.deinit(alloc);

    const linear_context = prior_mod.Context.fromTarget(
        linear,
        .u8,
        null,
        0,
        0,
    );
    const varied_context = prior_mod.Context.fromTarget(
        varied,
        .u8,
        null,
        0,
        0,
    );
    try std.testing.expectEqual(
        @as(u8, 0),
        linear_context.difference_entropy_bucket,
    );
    try std.testing.expect(
        varied_context.difference_entropy_bucket >
            linear_context.difference_entropy_bucket,
    );
}

test "canonical prior bytes are insertion-order independent and round trip" {
    const alloc = std.testing.allocator;
    var target = try streamFromWords(alloc, 8, &.{ 9, 8, 9, 8 });
    defer target.deinit(alloc);
    const first = prior_mod.Context.fromTarget(target, .u8, null, 0, 0);
    var second = first;
    second.zero_bucket = (second.zero_bucket + 1) % 4;

    var counts_a = prior_mod.Counts.init();
    defer counts_a.deinit(alloc);
    try counts_a.observe(alloc, first, .repeat, 7);
    try counts_a.observe(alloc, second, .map_xor, 3);

    var counts_b = prior_mod.Counts.init();
    defer counts_b.deinit(alloc);
    try counts_b.observe(alloc, second, .map_xor, 3);
    try counts_b.observe(alloc, first, .repeat, 7);

    var prior_a = try counts_a.toPrior(alloc, .{});
    defer prior_a.deinit(alloc);
    var prior_b = try counts_b.toPrior(alloc, .{});
    defer prior_b.deinit(alloc);
    const bytes_a = try prior_a.serialize(alloc);
    defer alloc.free(bytes_a);
    const bytes_b = try prior_b.serialize(alloc);
    defer alloc.free(bytes_b);
    try std.testing.expectEqualSlices(u8, bytes_a, bytes_b);

    var restored = try prior_mod.Prior.deserialize(alloc, bytes_a);
    defer restored.deinit(alloc);
    const round_trip = try restored.serialize(alloc);
    defer alloc.free(round_trip);
    try std.testing.expectEqualSlices(u8, bytes_a, round_trip);

    var original_bounds: [prior_mod.PRODUCTION_COUNT]prior_mod.Cost =
        undefined;
    var restored_bounds: [prior_mod.PRODUCTION_COUNT]prior_mod.Cost =
        undefined;
    try prior_a.contextualCostLowerBounds(5, &original_bounds);
    try restored.contextualCostLowerBounds(5, &restored_bounds);
    try std.testing.expectEqual(original_bounds, restored_bounds);
}

test "prior decoder rejects trailing data duplicate and invalid stable ids" {
    const alloc = std.testing.allocator;
    var target = try streamFromWords(alloc, 8, &.{ 1, 2, 1, 2 });
    defer target.deinit(alloc);
    const context = prior_mod.Context.fromTarget(target, .u8, null, 0, 0);

    var counts = prior_mod.Counts.init();
    defer counts.deinit(alloc);
    try counts.observe(alloc, context, .literal, 1);
    try counts.observe(alloc, context, .repeat, 1);
    var model = try counts.toPrior(alloc, .{});
    defer model.deinit(alloc);
    const canonical = try model.serialize(alloc);
    defer alloc.free(canonical);

    const trailing = try alloc.alloc(u8, canonical.len + 1);
    defer alloc.free(trailing);
    @memcpy(trailing[0..canonical.len], canonical);
    trailing[canonical.len] = 0;
    try std.testing.expectError(
        error.TrailingData,
        prior_mod.Prior.deserialize(alloc, trailing),
    );

    // The first level-zero row begins at byte 30. Its fixed-width key is eleven
    // bytes, followed by an entry count and sorted (production-id, count)
    // records. Make the second production id duplicate the first one.
    var duplicate = try alloc.dupe(u8, canonical);
    defer alloc.free(duplicate);
    const first_id_offset: usize = 42;
    const second_id_offset = first_id_offset + 10;
    duplicate[second_id_offset] = duplicate[first_id_offset];
    duplicate[second_id_offset + 1] = duplicate[first_id_offset + 1];
    try std.testing.expectError(
        error.DuplicateProduction,
        prior_mod.Prior.deserialize(alloc, duplicate),
    );

    var invalid = try alloc.dupe(u8, canonical);
    defer alloc.free(invalid);
    std.mem.writeInt(u16, invalid[first_id_offset..][0..2], 0xfffe, .little);
    try std.testing.expectError(
        error.InvalidProductionId,
        prior_mod.Prior.deserialize(alloc, invalid),
    );

    var other_context = context;
    other_context.zero_bucket = (other_context.zero_bucket + 1) % 4;
    var context_counts = prior_mod.Counts.init();
    defer context_counts.deinit(alloc);
    try context_counts.observe(alloc, context, .literal, 1);
    try context_counts.observe(alloc, other_context, .literal, 1);
    var context_model = try context_counts.toPrior(alloc, .{});
    defer context_model.deinit(alloc);
    const context_bytes = try context_model.serialize(alloc);
    defer alloc.free(context_bytes);

    // Each of these two level-zero rows has an eleven-byte key, one-byte entry
    // count, and one ten-byte entry.
    var duplicate_context = try alloc.dupe(u8, context_bytes);
    defer alloc.free(duplicate_context);
    @memcpy(duplicate_context[52..63], duplicate_context[30..41]);
    try std.testing.expectError(
        error.DuplicateContext,
        prior_mod.Prior.deserialize(alloc, duplicate_context),
    );
}

test "an exact program can be recursively observed through decomposition" {
    const alloc = std.testing.allocator;
    var target = try streamFromWords(alloc, 32, &.{
        0x3f80_0000,
        0xbf80_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x3f80_0000,
        0xbf80_0000,
    });
    defer target.deinit(alloc);

    const period_words = [_]u32{ 0x3f80_0000, 0xbf80_0000 };
    const period = try dsl.Program.literal(alloc, 32, &period_words);
    var program = try dsl.Program.repeat(alloc, 3, period);
    defer program.deinit(alloc);

    var counts = prior_mod.Counts.init();
    defer counts.deinit(alloc);
    try counts.observeProgram(alloc, program, target, .f32, 1);
    var learned = try counts.toPrior(alloc, .{
        .learned_numerator = 19,
        .learned_denominator = 20,
    });
    defer learned.deinit(alloc);

    const root = prior_mod.Context.fromTarget(target, .f32, null, 0, 0);
    const root_legal = [_]grammar.ProductionId{ .literal, .repeat };
    var root_costs: [2]prior_mod.Cost = undefined;
    try learned.scoreSet(root, &root_legal, &root_costs);
    try std.testing.expect(root_costs[1] < root_costs[0]);

    var child_targets = try grammar.childTargets(
        alloc,
        .{ .repeat = 3 },
        target,
        .f32,
    );
    defer child_targets.deinit(alloc);
    const child = prior_mod.Context.fromTarget(
        child_targets.streams[0],
        .f32,
        .repeat,
        0,
        1,
    );
    const child_legal = [_]grammar.ProductionId{ .literal, .map_xor };
    var child_costs: [2]prior_mod.Cost = undefined;
    try learned.scoreSet(child, &child_legal, &child_costs);
    try std.testing.expect(child_costs[0] < child_costs[1]);
}
