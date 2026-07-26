//! Contract tests for the finite, target-directed semantic grammar registry.

const std = @import("std");
const dsl = @import("dsl.zig");
const grammar = @import("grammar.zig");
const interpreter = @import("interpreter.zig");
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

fn literalFromStream(alloc: std.mem.Allocator, stream: types.Stream) !dsl.Program {
    const words = try alloc.alloc(u32, stream.count);
    defer alloc.free(words);
    for (words, 0..) |*word, index| word.* = stream.getU32(index);
    return dsl.Program.literal(alloc, stream.bits_per_elem, words);
}

fn programFromChoice(
    alloc: std.mem.Allocator,
    choice: grammar.Choice,
    target: types.Stream,
    children: []const types.Stream,
) !dsl.Program {
    return switch (choice) {
        .literal => literalFromStream(alloc, target),
        .constant => |word| dsl.Program.constant(
            target.bits_per_elem,
            target.count,
            word,
        ),
        .repeat => |times| blk: {
            try std.testing.expectEqual(@as(usize, 1), children.len);
            const child = try literalFromStream(alloc, children[0]);
            errdefer {
                var owned = child;
                owned.deinit(alloc);
            }
            break :blk dsl.Program.repeat(alloc, times, child);
        },
        .concat => blk: {
            try std.testing.expectEqual(@as(usize, 2), children.len);
            var child_programs: [2]dsl.Program = undefined;
            var initialized: usize = 0;
            defer for (child_programs[0..initialized]) |*child| child.deinit(alloc);
            for (children, 0..) |child_target, index| {
                child_programs[index] = try literalFromStream(alloc, child_target);
                initialized += 1;
            }
            break :blk dsl.Program.concat(alloc, &child_programs);
        },
        .map_xor,
        .map_add_mod,
        .map_zigzag,
        .map_gray,
        .map_rotate_left,
        .map_bit_reverse,
        => blk: {
            try std.testing.expectEqual(@as(usize, 1), children.len);
            const child = try literalFromStream(alloc, children[0]);
            errdefer {
                var owned = child;
                owned.deinit(alloc);
            }
            break :blk dsl.Program.map(alloc, choice.mapOperation().?, child);
        },
        .scan_xor, .scan_add_mod => |initial| blk: {
            try std.testing.expectEqual(@as(usize, 1), children.len);
            const child = try literalFromStream(alloc, children[0]);
            errdefer {
                var owned = child;
                owned.deinit(alloc);
            }
            break :blk dsl.Program.scan(
                alloc,
                choice.scanOperation().?,
                initial,
                child,
            );
        },
        .merge_fields,
        .merge_float_fields,
        .merge_bit_planes,
        .merge_byte_planes,
        => blk: {
            const child_programs = try alloc.alloc(dsl.Program, children.len);
            var initialized: usize = 0;
            defer {
                for (child_programs[0..initialized]) |*child| child.deinit(alloc);
                alloc.free(child_programs);
            }
            for (children, 0..) |child_target, index| {
                child_programs[index] = try literalFromStream(alloc, child_target);
                initialized += 1;
            }
            break :blk dsl.Program.merge(
                alloc,
                choice.mergeOperation().?,
                child_programs,
            );
        },
    };
}

fn expectStreamsEqual(expected: types.Stream, actual: types.Stream) !void {
    try std.testing.expectEqual(expected.bits_per_elem, actual.bits_per_elem);
    try std.testing.expectEqual(expected.count, actual.count);
    for (0..expected.count) |index|
        try std.testing.expectEqual(expected.getU32(index), actual.getU32(index));
}

fn expectChoiceReconstructs(
    alloc: std.mem.Allocator,
    choice: grammar.Choice,
    target: types.Stream,
    dtype: types.Dtype,
) !void {
    var children = try grammar.childTargets(alloc, choice, target, dtype);
    defer children.deinit(alloc);
    var program = try programFromChoice(alloc, choice, target, children.streams);
    defer program.deinit(alloc);
    var reconstructed = try interpreter.execute(alloc, program);
    defer reconstructed.deinit(alloc);
    try expectStreamsEqual(target, reconstructed);
}

fn countProduction(choices: []const grammar.Choice, production: grammar.ProductionId) usize {
    var count: usize = 0;
    for (choices) |choice| count += @intFromBool(choice.id() == production);
    return count;
}

test "grammar uses stable production ids and always proposes literal" {
    try std.testing.expectEqual(@as(u16, 0), @intFromEnum(grammar.ProductionId.literal));
    try std.testing.expectEqual(@as(u16, 10), @intFromEnum(grammar.ProductionId.map_xor));
    try std.testing.expectEqual(@as(u16, 33), @intFromEnum(grammar.ProductionId.merge_byte_planes));

    const alloc = std.testing.allocator;
    var target = try streamFromWords(alloc, 8, &.{ 1, 2, 3, 4 });
    defer target.deinit(alloc);

    const choices = try grammar.propose(alloc, target, .u8, 7, .{ .max_depth = 7 });
    defer alloc.free(choices);
    try std.testing.expectEqual(@as(usize, 1), choices.len);
    try std.testing.expectEqual(grammar.ProductionId.literal, choices[0].id());
    const legal = grammar.legal(target, .u8, 7, .{ .max_depth = 7 });
    try std.testing.expectEqualSlices(
        grammar.ProductionId,
        &.{.literal},
        legal.slice(),
    );
    try std.testing.expect(grammar.isLegal(
        .literal,
        target,
        .u8,
        7,
        .{ .max_depth = 7 },
    ));
}

test "grammar rejects storage that is not exactly the declared target type" {
    const alloc = std.testing.allocator;
    const oversized = types.Stream{
        .data = @constCast(&[_]u8{ 1, 2 }),
        .count = 1,
        .bits_per_elem = 8,
        .owns_data = false,
    };
    try std.testing.expectEqual(
        @as(usize, 0),
        grammar.legal(oversized, .u8, 0, .{}).len,
    );
    try std.testing.expectError(
        error.InvalidLiteralValue,
        grammar.propose(alloc, oversized, .u8, 0, .{}),
    );
}

test "every proposed semantic expansion reconstructs its target exactly" {
    const alloc = std.testing.allocator;
    var repeated = try streamFromWords(
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
    defer repeated.deinit(alloc);

    const choices = try grammar.propose(alloc, repeated, .f32, 0, .{});
    defer alloc.free(choices);
    var seen: [34]bool = @splat(false);
    var previous_id: u16 = 0;
    for (choices, 0..) |choice, index| {
        const id_value: u16 = @intFromEnum(choice.id());
        if (index > 0) try std.testing.expect(previous_id <= id_value);
        previous_id = id_value;
        seen[id_value] = true;
        try std.testing.expect(grammar.isLegal(
            choice.id(),
            repeated,
            .f32,
            0,
            .{},
        ));
        try expectChoiceReconstructs(alloc, choice, repeated, .f32);
    }

    const required = [_]grammar.ProductionId{
        .literal,
        .repeat,
        .concat,
        .map_xor,
        .map_add_mod,
        .map_zigzag,
        .map_gray,
        .map_rotate_left,
        .map_bit_reverse,
        .scan_xor,
        .scan_add_mod,
        .merge_fields,
        .merge_float_fields,
        .merge_bit_planes,
        .merge_byte_planes,
    };
    for (required) |production|
        try std.testing.expect(seen[@intFromEnum(production)]);

    try std.testing.expectEqual(@as(usize, 1), countProduction(choices, .repeat));
    for (choices) |choice| switch (choice) {
        .repeat => |times| {
            try std.testing.expectEqual(@as(u32, 3), times);
            var children = try grammar.childTargets(alloc, choice, repeated, .f32);
            defer children.deinit(alloc);
            try std.testing.expectEqual(@as(usize, 1), children.streams.len);
            try std.testing.expectEqual(@as(usize, 2), children.streams[0].count);
        },
        else => {},
    };

    var constant = try streamFromWords(alloc, 8, &.{ 7, 7, 7, 7 });
    defer constant.deinit(alloc);
    const constant_choices = try grammar.propose(alloc, constant, .u8, 0, .{});
    defer alloc.free(constant_choices);
    try std.testing.expectEqual(@as(usize, 1), countProduction(
        constant_choices,
        .constant,
    ));
    for (constant_choices) |choice|
        try expectChoiceReconstructs(alloc, choice, constant, .u8);
}

test "proposal order is deterministic and every parameter family is hard capped" {
    const alloc = std.testing.allocator;
    var target = try types.Stream.init(alloc, 4096, 16);
    defer target.deinit(alloc);
    for (0..target.count) |index|
        target.setU32(index, @intCast((index *% 4051) & 0xffff));

    const unlimited: grammar.Options = .{
        .max_repeat_period = std.math.maxInt(usize),
        .max_concat_splits = std.math.maxInt(usize),
        .max_map_constants = std.math.maxInt(usize),
        .max_rotations = std.math.maxInt(usize),
        .max_field_splits = std.math.maxInt(usize),
    };
    const first = try grammar.propose(alloc, target, .u16, 0, unlimited);
    defer alloc.free(first);
    const second = try grammar.propose(alloc, target, .u16, 0, unlimited);
    defer alloc.free(second);

    try std.testing.expect(first.len <= grammar.MAX_PROPOSALS);
    try std.testing.expectEqual(first.len, second.len);
    for (first, second) |left, right|
        try std.testing.expect(std.meta.eql(left, right));

    try std.testing.expectEqual(
        grammar.HARD_MAX_CONCAT_SPLITS,
        countProduction(first, .concat),
    );
    try std.testing.expect(
        countProduction(first, .map_xor) <= grammar.HARD_MAX_MAP_CONSTANTS,
    );
    try std.testing.expect(
        countProduction(first, .map_add_mod) <= grammar.HARD_MAX_MAP_CONSTANTS,
    );
    try std.testing.expect(
        countProduction(first, .map_rotate_left) <= grammar.HARD_MAX_ROTATIONS,
    );
    try std.testing.expect(
        countProduction(first, .merge_fields) <= grammar.HARD_MAX_FIELD_SPLITS,
    );
}

test "deterministic randomized proposals always reconstruct exact targets" {
    const alloc = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5041_5045_5244_534c);
    const random = prng.random();

    for ([_]u8{ 1, 3, 7, 8, 12, 16, 23, 32 }) |bits| {
        const mask = if (bits == 32)
            std.math.maxInt(u32)
        else
            (@as(u32, 1) << @intCast(bits)) - 1;
        for (0..6) |case_index| {
            const count: usize = 2 + case_index * 2;
            var target = try types.Stream.init(alloc, count, bits);
            defer target.deinit(alloc);
            for (0..count) |index| {
                const word = if (case_index == 0)
                    @as(u32, 3) & mask
                else if (case_index == 1)
                    @as(u32, @intCast(index % 2)) & mask
                else
                    random.int(u32) & mask;
                target.setU32(index, word);
            }

            const choices = try grammar.propose(
                alloc,
                target,
                .u32,
                0,
                .{
                    .max_depth = 2,
                    .max_repeat_period = 8,
                    .max_concat_splits = 2,
                    .max_map_constants = 2,
                    .max_rotations = 2,
                    .max_field_splits = 2,
                },
            );
            defer alloc.free(choices);
            for (choices) |choice|
                try expectChoiceReconstructs(
                    alloc,
                    choice,
                    target,
                    .u32,
                );
        }
    }
}

test "child target storage predicts bit-plane amplification without allocating" {
    const alloc = std.testing.allocator;
    var target = try types.Stream.init(alloc, 32, 32);
    defer target.deinit(alloc);

    try std.testing.expectEqual(
        @as(usize, 32 * 32),
        try grammar.childTargetStorageBytes(
            .merge_bit_planes,
            target,
            .u32,
        ),
    );
    try std.testing.expectEqual(
        target.data.len,
        try grammar.childTargetStorageBytes(
            .merge_byte_planes,
            target,
            .u32,
        ),
    );

    const field_estimate = try grammar.childTargetStorageBytes(
        .{ .merge_fields = 8 },
        target,
        .u32,
    );
    var field_children = try grammar.childTargets(
        alloc,
        .{ .merge_fields = 8 },
        target,
        .u32,
    );
    defer field_children.deinit(alloc);
    var actual_field_storage: usize = 0;
    for (field_children.streams) |child|
        actual_field_storage += child.data.len;
    try std.testing.expectEqual(actual_field_storage, field_estimate);
    try std.testing.expectEqual(
        @as(usize, target.count * (1 + 4)),
        field_estimate,
    );
}
