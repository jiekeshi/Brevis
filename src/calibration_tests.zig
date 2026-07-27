//! Contract tests for deterministic, input-local PHOG calibration.

const std = @import("std");
const calibration = @import("calibration.zig");
const dsl = @import("dsl.zig");
const grammar_prior = @import("grammar_prior.zig");
const safetensors = @import("safetensors.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const ProductionId = @TypeOf(grammar_prior.PRODUCTIONS[0]);

fn tensor(
    name: []const u8,
    data: []u8,
    shape: []const u64,
    dtype: types.Dtype,
) safetensors.Tensor {
    return .{
        .name = name,
        .view = .{
            .data = data,
            .shape = shape,
            .dtype = dtype,
            .owns_data = false,
            .owns_shape = false,
        },
    };
}

fn stream(view: types.TensorView) !types.Stream {
    return .{
        .data = @constCast(view.data),
        .count = try view.numelChecked(),
        .bits_per_elem = view.dtype.bitWidth(),
        .owns_data = false,
    };
}

fn expectRootPreferred(
    prior: *const grammar_prior.Prior,
    target: types.Stream,
    dtype: types.Dtype,
    preferred: ProductionId,
) !void {
    const context = grammar_prior.Context.fromTarget(
        target,
        dtype,
        null,
        0,
        0,
    );
    const admitted = [_]ProductionId{ .literal, preferred };
    var costs: [admitted.len]grammar_prior.Cost = undefined;
    try prior.scoreSet(context, &admitted, &costs);
    try std.testing.expect(costs[1] < costs[0]);
}

test "training Repeat and Const tensors lowers their contextual root costs" {
    const alloc = std.testing.allocator;
    var repeat_data = [_]u8{
        0x00, 0x00, 0x80, 0x3f,
        0x00, 0x00, 0x80, 0xbf,
        0x00, 0x00, 0x80, 0x3f,
        0x00, 0x00, 0x80, 0xbf,
        0x00, 0x00, 0x80, 0x3f,
        0x00, 0x00, 0x80, 0xbf,
    };
    var constant_data = [_]u8{ 7, 7, 7, 7, 7, 7, 7, 7 };
    const repeat_shape = [_]u64{6};
    const constant_shape = [_]u64{constant_data.len};
    const tensors = [_]safetensors.Tensor{
        tensor("repeat", &repeat_data, &repeat_shape, .f32),
        tensor("constant", &constant_data, &constant_shape, .u8),
    };

    var result = try calibration.train(alloc, &tensors, .{
        .max_tensors = tensors.len,
        .synthesis = .{
            .max_expansions = 96,
            .max_nodes = 8,
            .grammar_options = .{ .max_depth = 1 },
        },
        .prior_config = .{
            .learned_numerator = 19,
            .learned_denominator = 20,
        },
    });
    defer result.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 2), result.observed_tensors);
    try std.testing.expect(result.expanded > 0);
    try expectRootPreferred(
        &result.prior,
        try stream(tensors[0].view),
        .f32,
        .repeat,
    );
    try expectRootPreferred(
        &result.prior,
        try stream(tensors[1].view),
        .u8,
        .constant,
    );
}

test "training is canonical and ignores any caller supplied rule model" {
    const alloc = std.testing.allocator;
    var data = [_]u8{
        0x00, 0x00, 0x80, 0x3f,
        0x00, 0x00, 0x80, 0xbf,
        0x00, 0x00, 0x80, 0x3f,
        0x00, 0x00, 0x80, 0xbf,
        0x00, 0x00, 0x80, 0x3f,
        0x00, 0x00, 0x80, 0xbf,
    };
    const shape = [_]u64{6};
    const tensors = [_]safetensors.Tensor{
        tensor("weights", &data, &shape, .f32),
    };
    const target = try stream(tensors[0].view);

    const period = try dsl.Program.literal(
        alloc,
        32,
        &.{ 0x3f80_0000, 0xbf80_0000 },
    );
    var repeat_program = try dsl.Program.repeat(alloc, 3, period);
    defer repeat_program.deinit(alloc);
    var hostile_counts = grammar_prior.Counts.init();
    defer hostile_counts.deinit(alloc);
    try hostile_counts.observeProgram(
        alloc,
        repeat_program,
        target,
        .f32,
        1_000,
    );
    var supplied_prior = try hostile_counts.toPrior(alloc, .{
        .learned_numerator = 19,
        .learned_denominator = 20,
    });
    defer supplied_prior.deinit(alloc);

    const base_synthesis = @import("synthesizer.zig").Options{
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
    var uniform = try calibration.train(alloc, &tensors, .{
        .synthesis = base_synthesis,
    });
    defer uniform.deinit(alloc);

    var caller_guided = base_synthesis;
    caller_guided.rule_model = &supplied_prior;
    var ignored = try calibration.train(alloc, &tensors, .{
        .synthesis = caller_guided,
    });
    defer ignored.deinit(alloc);

    const uniform_bytes = try uniform.serialize(alloc);
    defer alloc.free(uniform_bytes);
    const ignored_bytes = try ignored.serialize(alloc);
    defer alloc.free(ignored_bytes);
    try std.testing.expectEqualSlices(u8, uniform_bytes, ignored_bytes);
    try std.testing.expectEqual(uniform.expanded, ignored.expanded);
    try std.testing.expectEqual(
        uniform.completed_candidates,
        ignored.completed_candidates,
    );
    try std.testing.expectEqualStrings("BRGP", uniform_bytes[0..4]);
    try std.testing.expect(std.mem.indexOf(u8, uniform_bytes, &data) == null);
}

test "parallel training matches serial prior and search statistics" {
    const alloc = std.testing.allocator;
    var repeat_data = [_]u8{ 1, 2, 1, 2, 1, 2, 1, 2 };
    var constant_data = [_]u8{ 9, 9, 9, 9, 9, 9 };
    var literal_data = [_]u8{ 8, 3, 5, 1, 7 };
    const repeat_shape = [_]u64{repeat_data.len};
    const constant_shape = [_]u64{constant_data.len};
    const literal_shape = [_]u64{literal_data.len};
    const tensors = [_]safetensors.Tensor{
        tensor("repeat", &repeat_data, &repeat_shape, .u8),
        tensor("constant", &constant_data, &constant_shape, .u8),
        tensor("literal", &literal_data, &literal_shape, .u8),
    };
    const options: calibration.Options = .{
        .max_tensors = tensors.len,
        .synthesis = .{
            .max_expansions = 64,
            .max_nodes = 8,
            .grammar_options = .{ .max_depth = 1 },
        },
    };

    var serial = try calibration.train(alloc, &tensors, options);
    defer serial.deinit(alloc);
    var parallel = try calibration.trainParallel(
        alloc,
        std.testing.io,
        &tensors,
        options,
        2,
    );
    defer parallel.deinit(alloc);
    const serial_bytes = try serial.serialize(alloc);
    defer alloc.free(serial_bytes);
    const parallel_bytes = try parallel.serialize(alloc);
    defer alloc.free(parallel_bytes);

    try std.testing.expectEqualSlices(u8, serial_bytes, parallel_bytes);
    try std.testing.expectEqual(serial.observed_tensors, parallel.observed_tensors);
    try std.testing.expectEqual(serial.expanded, parallel.expanded);
    try std.testing.expectEqual(
        serial.completed_candidates,
        parallel.completed_candidates,
    );
    try std.testing.expectEqual(
        serial.budget_exhausted_tensors,
        parallel.budget_exhausted_tensors,
    );
    try std.testing.expectEqual(
        serial.literal_fallback_tensors,
        parallel.literal_fallback_tensors,
    );
}

test "max_tensors covers dtype and physical-size strata deterministically" {
    const alloc = std.testing.allocator;
    var first_data = [_]u8{ 0, 0, 0, 0 };
    var second_data = [_]u8{ 5, 5 };
    var middle_data = [_]u8{ 9, 8, 7 };
    var fourth_data = [_]u8{ 2, 3, 4, 5 };
    var last_data = [_]u8{ 1, 2, 1, 2, 1, 2 };
    const shape4 = [_]u64{4};
    const shape2 = [_]u64{2};
    const shape3 = [_]u64{3};
    const shape6 = [_]u64{6};
    const all = [_]safetensors.Tensor{
        tensor("first", &first_data, &shape4, .u8),
        tensor("second", &second_data, &shape2, .u8),
        tensor("middle", &middle_data, &shape3, .u8),
        tensor("fourth", &fourth_data, &shape4, .u8),
        tensor("last", &last_data, &shape6, .u8),
    };
    // The two U8 size strata are 2-3 bytes and 4-7 bytes. Their first
    // representatives are `second` and `first`, respectively.
    const stratum_representatives = [_]safetensors.Tensor{
        all[1],
        all[0],
    };
    const options: calibration.Options = .{
        .max_tensors = 2,
        .synthesis = .{
            .max_expansions = 64,
            .max_nodes = 8,
            .grammar_options = .{ .max_depth = 1 },
        },
    };

    var sampled = try calibration.train(alloc, &all, options);
    defer sampled.deinit(alloc);
    var explicit = try calibration.train(
        alloc,
        &stratum_representatives,
        options,
    );
    defer explicit.deinit(alloc);
    const sampled_bytes = try sampled.serialize(alloc);
    defer alloc.free(sampled_bytes);
    const explicit_bytes = try explicit.serialize(alloc);
    defer alloc.free(explicit_bytes);

    try std.testing.expectEqual(@as(usize, 2), sampled.observed_tensors);
    try std.testing.expectEqualSlices(u8, explicit_bytes, sampled_bytes);
    try std.testing.expectEqual(explicit.expanded, sampled.expanded);
}

test "stratified calibration covers distinct dtypes before repeats" {
    const alloc = std.testing.allocator;
    var ignored_u8 = [_]u8{ 1, 2 };
    var first_f32 = [_]u8{ 0, 0, 0, 0 };
    var first_u8 = [_]u8{7};
    var ignored_f32 = [_]u8{ 1, 0, 0, 0 };
    const shape2 = [_]u64{2};
    const shape1 = [_]u64{1};
    const all = [_]safetensors.Tensor{
        tensor("ignored-u8", &ignored_u8, &shape2, .u8),
        tensor("first-f32", &first_f32, &shape1, .f32),
        tensor("first-u8", &first_u8, &shape1, .u8),
        tensor("ignored-f32", &ignored_f32, &shape1, .f32),
    };
    const representatives = [_]safetensors.Tensor{
        all[1],
        all[2],
    };
    const options: calibration.Options = .{
        .max_tensors = 2,
        .synthesis = .{
            .max_expansions = 16,
            .max_nodes = 4,
            .grammar_options = .{ .max_depth = 1 },
        },
    };

    var sampled = try calibration.train(alloc, &all, options);
    defer sampled.deinit(alloc);
    var explicit = try calibration.train(
        alloc,
        &representatives,
        options,
    );
    defer explicit.deinit(alloc);
    const sampled_bytes = try sampled.serialize(alloc);
    defer alloc.free(sampled_bytes);
    const explicit_bytes = try explicit.serialize(alloc);
    defer alloc.free(explicit_bytes);
    try std.testing.expectEqualSlices(u8, explicit_bytes, sampled_bytes);
}

test "empty inputs max zero and zero-length tensors are supported" {
    const alloc = std.testing.allocator;
    var none = try calibration.train(alloc, &.{}, .{});
    defer none.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 0), none.observed_tensors);
    try std.testing.expectEqual(@as(usize, 0), none.expanded);
    try std.testing.expect(none.prior.isEmpty());

    var data = [_]u8{ 1, 2, 3, 4 };
    const nonempty_shape = [_]u64{4};
    const nonempty = [_]safetensors.Tensor{
        tensor("ignored", &data, &nonempty_shape, .u8),
    };
    var capped = try calibration.train(alloc, &nonempty, .{
        .max_tensors = 0,
    });
    defer capped.deinit(alloc);
    const none_bytes = try none.serialize(alloc);
    defer alloc.free(none_bytes);
    const capped_bytes = try capped.serialize(alloc);
    defer alloc.free(capped_bytes);
    try std.testing.expectEqualSlices(u8, none_bytes, capped_bytes);

    var empty_data: [0]u8 = .{};
    const empty_shape = [_]u64{0};
    const empty_tensor = [_]safetensors.Tensor{
        tensor("empty", &empty_data, &empty_shape, .u8),
    };
    var observed_empty = try calibration.train(alloc, &empty_tensor, .{
        .synthesis = .{ .max_expansions = 8 },
    });
    defer observed_empty.deinit(alloc);
    try std.testing.expectEqual(
        @as(usize, 1),
        observed_empty.observed_tensors,
    );
    try std.testing.expect(!observed_empty.prior.isEmpty());
}
