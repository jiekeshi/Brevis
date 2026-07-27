const std = @import("std");
const decomposition = @import("decomposition.zig");
const dsl = @import("dsl.zig");
const interpreter = @import("interpreter.zig");
const types = @import("types.zig");

fn wordMask(bits: u8) u32 {
    return if (bits == 32)
        std.math.maxInt(u32)
    else
        (@as(u32, 1) << @intCast(bits)) - 1;
}

fn randomStream(
    alloc: std.mem.Allocator,
    random: std.Random,
    count: usize,
    bits: u8,
) !types.Stream {
    var stream = try types.Stream.initUninitialized(alloc, count, bits);
    for (0..count) |i| stream.setU32(i, random.int(u32) & wordMask(bits));
    return stream;
}

fn freeChildren(alloc: std.mem.Allocator, children: []types.Stream) void {
    for (children) |*child| child.deinit(alloc);
    alloc.free(children);
}

fn expectMergeRoundTrip(
    alloc: std.mem.Allocator,
    target: types.Stream,
    operation: dsl.MergeOp,
) !void {
    const streams = try decomposition.merge(alloc, target, operation);
    defer freeChildren(alloc, streams);
    const programs = try alloc.alloc(dsl.Program, streams.len);
    var initialized: usize = 0;
    while (initialized < streams.len) : (initialized += 1) {
        programs[initialized] = dsl.Program.literalFromStream(
            alloc,
            streams[initialized],
        ) catch |err| {
            for (programs[0..initialized]) |*program| program.deinit(alloc);
            alloc.free(programs);
            return err;
        };
    }
    var program = dsl.Program.mergeOwned(operation, programs) catch |err| {
        for (programs) |*child| child.deinit(alloc);
        alloc.free(programs);
        return err;
    };
    defer program.deinit(alloc);

    var output = try types.Stream.initUninitialized(
        alloc,
        target.count,
        target.bits_per_elem,
    );
    defer output.deinit(alloc);
    @memset(output.data, 0xa5);
    try interpreter.executeInto(alloc, program, output);
    try std.testing.expect(target.eql(output));
}

test "merge float fields matches scalar extraction across vector tails" {
    const alloc = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x1b5e_2f96_6050_ab13);
    const random = prng.random();
    const dtypes = [_]types.Dtype{ .f8_e4m3, .f8_e5m2, .f16, .bf16, .f32 };
    const counts = [_]usize{ 1, 15, 16, 17, 31, 32, 33 };

    for (dtypes) |dtype| {
        const fields = dtype.floatFields().?;
        for (counts) |count| {
            var target = try randomStream(alloc, random, count, fields.total);
            defer target.deinit(alloc);
            const children = try decomposition.merge(
                alloc,
                target,
                .{ .float_fields = dtype },
            );
            defer freeChildren(alloc, children);

            for (0..count) |i| {
                const word = target.getU32(i);
                try std.testing.expectEqual(
                    word >> @intCast(fields.total - 1),
                    children[0].getU32(i),
                );
                try std.testing.expectEqual(
                    (word >> @intCast(fields.mant)) & wordMask(fields.exp),
                    children[1].getU32(i),
                );
                try std.testing.expectEqual(
                    word & wordMask(fields.mant),
                    children[2].getU32(i),
                );
            }
        }
    }
}

test "merge fields and planes match scalar extraction" {
    const alloc = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xd76b_054f_b528_7093);
    const random = prng.random();
    const widths = [_]u8{ 2, 7, 8, 9, 15, 16, 17, 23, 24, 31, 32 };
    const counts = [_]usize{ 1, 15, 16, 17, 33 };

    for (widths) |bits| {
        for (counts) |count| {
            var target = try types.Stream.initUninitialized(alloc, count, bits);
            for (0..count) |i| target.setU32(i, random.int(u32));
            defer target.deinit(alloc);

            const split_points = [_]u8{ 1, bits / 2, bits - 1 };
            for (split_points) |low_bits| {
                const children = try decomposition.merge(
                    alloc,
                    target,
                    .{ .fields = low_bits },
                );
                defer freeChildren(alloc, children);
                for (0..count) |i| {
                    const word = target.getU32(i);
                    try std.testing.expectEqual(
                        word & wordMask(low_bits),
                        children[0].getU32(i),
                    );
                    try std.testing.expectEqual(
                        (word >> @intCast(low_bits)) &
                            wordMask(bits - low_bits),
                        children[1].getU32(i),
                    );
                }
            }

            for ([_]dsl.MergeOp{ .bit_planes, .byte_planes }) |operation| {
                const step: u8 = if (operation == .bit_planes) 1 else 8;
                const children = try decomposition.merge(alloc, target, operation);
                defer freeChildren(alloc, children);
                for (children, 0..) |child, plane| {
                    for (0..count) |i| {
                        try std.testing.expectEqual(
                            (target.getU32(i) >> @intCast(plane * step)) &
                                wordMask(child.bits_per_elem),
                            child.getU32(i),
                        );
                    }
                }
            }
        }
    }
}

test "vectorized merge decomposition and execution round trip every layout" {
    const alloc = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x22cf_1a78_f9d2_369b);
    const random = prng.random();

    for ([_]types.Dtype{ .f8_e4m3, .f8_e5m2, .f16, .bf16, .f32 }) |dtype| {
        var target = try randomStream(
            alloc,
            random,
            33,
            dtype.bitWidth(),
        );
        defer target.deinit(alloc);
        try expectMergeRoundTrip(
            alloc,
            target,
            .{ .float_fields = dtype },
        );
    }

    for ([_]u8{ 7, 8, 9, 16, 17, 23, 32 }) |bits| {
        var target = try randomStream(alloc, random, 33, bits);
        defer target.deinit(alloc);
        try expectMergeRoundTrip(
            alloc,
            target,
            .{ .fields = bits / 2 },
        );
        try expectMergeRoundTrip(alloc, target, .bit_planes);
        if (bits > 8)
            try expectMergeRoundTrip(alloc, target, .byte_planes);
    }
}
