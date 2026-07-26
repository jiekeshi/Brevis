const std = @import("std");
const brevis = @import("brevis.zig");

test "public seams compose into an exact canonical round trip" {
    const alloc = std.testing.allocator;
    var target = try brevis.types.Stream.init(alloc, 6, 32);
    defer target.deinit(alloc);
    for (0..3) |index| {
        target.setU32(index * 2, 0x3f80_0000);
        target.setU32(index * 2 + 1, 0xbf80_0000);
    }

    var synthesis = try brevis.synthesize(
        alloc,
        target,
        .f32,
        .{
            .max_expansions = 64,
            .max_nodes = 8,
            .grammar_options = .{ .max_depth = 1 },
        },
    );
    defer synthesis.deinit(alloc);

    const bytes = try brevis.writeProgram(alloc, synthesis.program);
    defer alloc.free(bytes);
    try std.testing.expectEqual(
        bytes.len,
        try brevis.serializedProgramSize(alloc, synthesis.program),
    );

    var restored = try brevis.readProgram(alloc, bytes, .{});
    defer restored.deinit(alloc);
    var output = try brevis.execute(alloc, restored);
    defer output.deinit(alloc);
    try std.testing.expectEqual(target.bits_per_elem, output.bits_per_elem);
    try std.testing.expectEqual(target.count, output.count);
    try std.testing.expectEqualSlices(u8, target.data, output.data);
}
