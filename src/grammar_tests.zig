//! Contract tests for the finite, target-directed semantic grammar registry.

const std = @import("std");
const dsl = @import("dsl.zig");
const grammar = @import("grammar.zig");
const interpreter = @import("interpreter.zig");
const program_format = @import("program_format.zig");
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

fn expectLegalMatchesProposals(
    alloc: std.mem.Allocator,
    target: types.Stream,
    dtype: types.Dtype,
    hole_depth: u8,
    options: grammar.Options,
) !void {
    const choices = try grammar.propose(
        alloc,
        target,
        dtype,
        hole_depth,
        options,
    );
    defer alloc.free(choices);
    const proposed = grammar.families(choices);
    const legal = grammar.legal(target, dtype, hole_depth, options);
    try std.testing.expectEqualSlices(
        grammar.ProductionId,
        legal.slice(),
        proposed.slice(),
    );
    for (std.enums.values(grammar.ProductionId)) |production|
        try std.testing.expectEqual(
            countProduction(choices, production) > 0,
            grammar.isLegal(
                production,
                target,
                dtype,
                hole_depth,
                options,
            ),
        );
}

fn expectCanonicalMergeAlias(
    alloc: std.mem.Allocator,
    target: types.Stream,
    dtype: types.Dtype,
    alias: grammar.Choice,
    canonical: grammar.Choice,
) !void {
    var alias_children = try grammar.childTargets(
        alloc,
        alias,
        target,
        dtype,
    );
    defer alias_children.deinit(alloc);
    var canonical_children = try grammar.childTargets(
        alloc,
        canonical,
        target,
        dtype,
    );
    defer canonical_children.deinit(alloc);
    try std.testing.expectEqual(
        alias_children.streams.len,
        canonical_children.streams.len,
    );
    for (alias_children.streams, canonical_children.streams) |left, right|
        try expectStreamsEqual(left, right);

    var alias_program = try programFromChoice(
        alloc,
        alias,
        target,
        alias_children.streams,
    );
    defer alias_program.deinit(alloc);
    var canonical_program = try programFromChoice(
        alloc,
        canonical,
        target,
        canonical_children.streams,
    );
    defer canonical_program.deinit(alloc);
    const alias_bytes = try program_format.serializedSize(
        alloc,
        alias_program,
    );
    const canonical_bytes = try program_format.serializedSize(
        alloc,
        canonical_program,
    );
    try std.testing.expectEqual(canonical_bytes + 1, alias_bytes);
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
    const fast_choices = try grammar.proposeKnownValid(
        alloc,
        repeated,
        .f32,
        0,
        .{},
    );
    defer alloc.free(fast_choices);
    try std.testing.expectEqual(choices.len, fast_choices.len);
    for (choices, fast_choices) |choice, fast_choice|
        try std.testing.expect(std.meta.eql(choice, fast_choice));
    const proposed_families = grammar.families(choices);
    const legal_families = grammar.legal(repeated, .f32, 0, .{});
    try std.testing.expectEqualSlices(
        grammar.ProductionId,
        legal_families.slice(),
        proposed_families.slice(),
    );
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

        const checked_storage = try grammar.childTargetStorageBytes(
            choice,
            repeated,
            .f32,
        );
        try std.testing.expectEqual(
            checked_storage,
            try grammar.childTargetStorageBytesForProposal(
                choice,
                repeated,
                .f32,
            ),
        );
        var checked = try grammar.childTargets(
            alloc,
            choice,
            repeated,
            .f32,
        );
        defer checked.deinit(alloc);
        var fast = try grammar.childTargetsForProposal(
            alloc,
            choice,
            repeated,
            .f32,
        );
        defer fast.deinit(alloc);
        try std.testing.expectEqual(checked.streams.len, fast.streams.len);
        const shapes = grammar.childShapesForProposal(choice, repeated);
        try std.testing.expectEqual(fast.streams.len, shapes.len);
        for (checked.streams, fast.streams, shapes.slice()) |expected, actual, shape| {
            try expectStreamsEqual(expected, actual);
            try std.testing.expectEqual(actual.bits_per_elem, shape.bits);
            try std.testing.expectEqual(actual.count, shape.count);
        }
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

test "FP8 field proposals include the sign boundary without widening search" {
    const alloc = std.testing.allocator;
    var target = try types.Stream.init(alloc, 16, 8);
    defer target.deinit(alloc);

    const fp8 = try grammar.propose(alloc, target, .f8_e4m3, 0, .{});
    defer alloc.free(fp8);
    const integer = try grammar.propose(alloc, target, .u8, 0, .{});
    defer alloc.free(integer);

    try std.testing.expectEqual(
        countProduction(integer, .merge_fields),
        countProduction(fp8, .merge_fields),
    );
    var fp8_splits: [3]u8 = undefined;
    var split_count: usize = 0;
    for (fp8) |choice| switch (choice) {
        .merge_fields => |low_bits| {
            fp8_splits[split_count] = low_bits;
            split_count += 1;
        },
        else => {},
    };
    try std.testing.expectEqualSlices(u8, &.{ 2, 4, 7 }, &fp8_splits);
}

test "normal form removes semantic identities and duplicate one-bit families" {
    const alloc = std.testing.allocator;
    var target = try streamFromWords(alloc, 1, &.{ 0, 1, 0, 1 });
    defer target.deinit(alloc);

    const choices = try grammar.propose(alloc, target, .u8, 0, .{});
    defer alloc.free(choices);
    try std.testing.expectEqual(@as(usize, 1), countProduction(
        choices,
        .map_xor,
    ));
    inline for (.{
        grammar.ProductionId.map_add_mod,
        grammar.ProductionId.map_zigzag,
        grammar.ProductionId.map_gray,
        grammar.ProductionId.map_rotate_left,
        grammar.ProductionId.map_bit_reverse,
        grammar.ProductionId.scan_add_mod,
    }) |production|
        try std.testing.expectEqual(
            @as(usize, 0),
            countProduction(choices, production),
        );
    try std.testing.expectEqual(@as(usize, 1), countProduction(
        choices,
        .scan_xor,
    ));
    for (choices) |choice| switch (choice) {
        .map_xor => |parameter| try std.testing.expectEqual(
            @as(u32, 1),
            parameter,
        ),
        else => {},
    };
    try expectLegalMatchesProposals(alloc, target, .u8, 0, .{});

    const identities = [_]grammar.Choice{
        .{ .map_xor = 0 },
        .{ .map_add_mod = 0 },
        .map_zigzag,
        .map_gray,
        .{ .map_rotate_left = 0 },
        .map_bit_reverse,
    };
    for (identities) |identity| {
        var children = try grammar.childTargets(
            alloc,
            identity,
            target,
            .u8,
        );
        defer children.deinit(alloc);
        try std.testing.expectEqual(@as(usize, 1), children.streams.len);
        try expectStreamsEqual(target, children.streams[0]);
    }

    var xor_map = try grammar.childTargets(
        alloc,
        .{ .map_xor = 1 },
        target,
        .u8,
    );
    defer xor_map.deinit(alloc);
    var add_map = try grammar.childTargets(
        alloc,
        .{ .map_add_mod = 1 },
        target,
        .u8,
    );
    defer add_map.deinit(alloc);
    try expectStreamsEqual(xor_map.streams[0], add_map.streams[0]);

    var xor_scan = try grammar.childTargets(
        alloc,
        .{ .scan_xor = 0 },
        target,
        .u8,
    );
    defer xor_scan.deinit(alloc);
    var add_scan = try grammar.childTargets(
        alloc,
        .{ .scan_add_mod = 0 },
        target,
        .u8,
    );
    defer add_scan.deinit(alloc);
    try expectStreamsEqual(xor_scan.streams[0], add_scan.streams[0]);
}

test "normal form omits zero map parameters and dominated uniform repeats" {
    const alloc = std.testing.allocator;
    var mixed = try streamFromWords(alloc, 8, &.{ 0, 3, 0, 5 });
    defer mixed.deinit(alloc);
    const mixed_choices = try grammar.propose(alloc, mixed, .u8, 0, .{});
    defer alloc.free(mixed_choices);
    for (mixed_choices) |choice| switch (choice) {
        .map_xor, .map_add_mod => |parameter| try std.testing.expect(parameter != 0),
        else => {},
    };
    try std.testing.expect(countProduction(mixed_choices, .map_xor) > 0);
    try std.testing.expect(countProduction(mixed_choices, .map_add_mod) > 0);
    try expectLegalMatchesProposals(alloc, mixed, .u8, 0, .{});

    var uniform = try streamFromWords(alloc, 8, &.{ 0, 0, 0, 0 });
    defer uniform.deinit(alloc);
    const uniform_choices = try grammar.propose(alloc, uniform, .u8, 0, .{});
    defer alloc.free(uniform_choices);
    try std.testing.expectEqual(@as(usize, 1), countProduction(
        uniform_choices,
        .constant,
    ));
    inline for (.{
        grammar.ProductionId.repeat,
        grammar.ProductionId.map_xor,
        grammar.ProductionId.map_add_mod,
    }) |production|
        try std.testing.expectEqual(
            @as(usize, 0),
            countProduction(uniform_choices, production),
        );
    try std.testing.expect(!grammar.isLegal(.repeat, uniform, .u8, 0, .{}));
    try expectLegalMatchesProposals(alloc, uniform, .u8, 0, .{});
}

test "normal form keeps parameter-free plane aliases and every distinct field split" {
    const alloc = std.testing.allocator;
    const byte_cases = [_]struct {
        bits: u8,
        requested: usize,
    }{
        .{ .bits = 9, .requested = 8 },
        .{ .bits = 10, .requested = 8 },
        .{ .bits = 11, .requested = 3 },
        .{ .bits = 12, .requested = 8 },
        .{ .bits = 13, .requested = 5 },
        .{ .bits = 14, .requested = 4 },
        .{ .bits = 15, .requested = 6 },
        .{ .bits = 16, .requested = 3 },
    };
    for (byte_cases) |case| {
        var target = try streamFromWords(
            alloc,
            case.bits,
            &.{ 1, 2, 3, 4 },
        );
        defer target.deinit(alloc);
        const options: grammar.Options = .{
            .max_field_splits = case.requested,
        };
        const choices = try grammar.propose(
            alloc,
            target,
            .u16,
            0,
            options,
        );
        defer alloc.free(choices);
        var saw_split_eight = false;
        for (choices) |choice| switch (choice) {
            .merge_fields => |low_bits| saw_split_eight =
                saw_split_eight or low_bits == 8,
            else => {},
        };
        try std.testing.expect(!saw_split_eight);
        try std.testing.expectEqual(
            case.requested - 1,
            countProduction(choices, .merge_fields),
        );
        try std.testing.expectEqual(@as(usize, 1), countProduction(
            choices,
            .merge_byte_planes,
        ));
        try std.testing.expect(grammar.isLegal(
            .merge_fields,
            target,
            .u16,
            0,
            options,
        ));
        try expectLegalMatchesProposals(
            alloc,
            target,
            .u16,
            0,
            options,
        );
        try expectCanonicalMergeAlias(
            alloc,
            target,
            .u16,
            .{ .merge_fields = 8 },
            .merge_byte_planes,
        );
    }

    var two_bit = try streamFromWords(alloc, 2, &.{ 0, 1, 2, 3 });
    defer two_bit.deinit(alloc);
    const two_bit_choices = try grammar.propose(
        alloc,
        two_bit,
        .u8,
        0,
        .{},
    );
    defer alloc.free(two_bit_choices);
    try std.testing.expectEqual(@as(usize, 0), countProduction(
        two_bit_choices,
        .merge_fields,
    ));
    try std.testing.expectEqual(@as(usize, 1), countProduction(
        two_bit_choices,
        .merge_bit_planes,
    ));
    try std.testing.expect(!grammar.isLegal(
        .merge_fields,
        two_bit,
        .u8,
        0,
        .{},
    ));
    try expectLegalMatchesProposals(alloc, two_bit, .u8, 0, .{});
    try expectCanonicalMergeAlias(
        alloc,
        two_bit,
        .u8,
        .{ .merge_fields = 1 },
        .merge_bit_planes,
    );

    var sixteen_bit = try streamFromWords(
        alloc,
        16,
        &.{ 0x0102, 0x3456, 0x789a, 0xbcde },
    );
    defer sixteen_bit.deinit(alloc);
    const no_fields: grammar.Options = .{ .max_field_splits = 0 };
    const no_field_choices = try grammar.propose(
        alloc,
        sixteen_bit,
        .u16,
        0,
        no_fields,
    );
    defer alloc.free(no_field_choices);
    try std.testing.expectEqual(@as(usize, 0), countProduction(
        no_field_choices,
        .merge_fields,
    ));
    try std.testing.expectEqual(@as(usize, 1), countProduction(
        no_field_choices,
        .merge_byte_planes,
    ));
    try expectLegalMatchesProposals(
        alloc,
        sixteen_bit,
        .u16,
        0,
        no_fields,
    );

    const only_alias: grammar.Options = .{ .max_field_splits = 1 };
    const canonical_only = try grammar.propose(
        alloc,
        sixteen_bit,
        .u16,
        0,
        only_alias,
    );
    defer alloc.free(canonical_only);
    try std.testing.expectEqual(@as(usize, 0), countProduction(
        canonical_only,
        .merge_fields,
    ));
    try std.testing.expectEqual(@as(usize, 1), countProduction(
        canonical_only,
        .merge_byte_planes,
    ));
    try expectLegalMatchesProposals(
        alloc,
        sixteen_bit,
        .u16,
        0,
        only_alias,
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

            const options: grammar.Options = .{
                .max_depth = 2,
                .max_repeat_period = 8,
                .max_concat_splits = 2,
                .max_map_constants = 2,
                .max_rotations = 2,
                .max_field_splits = 2,
            };
            const choices = try grammar.propose(
                alloc,
                target,
                .u32,
                0,
                options,
            );
            defer alloc.free(choices);
            try expectLegalMatchesProposals(
                alloc,
                target,
                .u32,
                0,
                options,
            );
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
