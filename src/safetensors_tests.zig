const std = @import("std");
const safetensors = @import("safetensors.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;

fn buildFile(alloc: Allocator, header: []const u8, data: []const u8) ![]u8 {
    const total = try std.math.add(usize, 8, header.len);
    const file_len = try std.math.add(usize, total, data.len);
    const bytes = try alloc.alloc(u8, file_len);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8 .. 8 + header.len], header);
    @memcpy(bytes[8 + header.len ..], data);
    return bytes;
}

fn expectLoadError(
    expected: anyerror,
    header: []const u8,
    data: []const u8,
) !void {
    const alloc = std.testing.allocator;
    const bytes = try buildFile(alloc, header, data);
    var input_owned = true;
    defer if (input_owned) alloc.free(bytes);

    if (safetensors.loadFromBytes(alloc, bytes)) |loaded_value| {
        input_owned = false;
        var loaded = loaded_value;
        loaded.deinit(alloc);
        return error.ExpectedSafetensorsError;
    } else |actual| {
        try std.testing.expectEqual(expected, actual);
    }
}

fn loadForAllocationFailureCheck(alloc: Allocator) !void {
    const header =
        \\{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U16","shape":[1],"data_offsets":[2,4]}}
    ;
    const bytes = try buildFile(alloc, header, &.{ 1, 2, 3, 4 });
    var input_owned = true;
    defer if (input_owned) alloc.free(bytes);
    var loaded = try safetensors.loadFromBytes(alloc, bytes);
    input_owned = false;
    defer loaded.deinit(alloc);
}

test "safetensors valid multi-tensor metadata sorting and zero elements" {
    const alloc = std.testing.allocator;
    const header =
        \\{
        \\  "late":{"dtype":"U16","shape":[2],"data_offsets":[4,8]},
        \\  "__metadata__":{"source":"paper","version":"1"},
        \\  "empty":{"dtype":"F32","shape":[0,999],"data_offsets":[0,0]},
        \\  "early":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}
        \\}
    ;
    const bytes = try buildFile(alloc, header, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    const prefix_len = 8 + header.len;

    var metadata = try safetensors.parsePrefix(alloc, bytes[0..prefix_len]);
    defer metadata.deinit(alloc);
    try std.testing.expectEqual(@as(u64, 8), metadata.data_len);
    try std.testing.expectEqual(@as(usize, 3), metadata.tensors.len);
    try std.testing.expectEqualStrings("empty", metadata.tensors[0].name);
    try std.testing.expectEqualStrings("early", metadata.tensors[1].name);
    try std.testing.expectEqualStrings("late", metadata.tensors[2].name);
    try std.testing.expectEqual(@as(usize, 0), try metadata.tensors[0].expectedByteLen());

    var loaded = try safetensors.loadFromBytes(alloc, bytes);
    defer loaded.deinit(alloc);
    try std.testing.expectEqual(@intFromPtr(bytes.ptr), @intFromPtr(loaded.bytes.ptr));
    try std.testing.expectEqualStrings("empty", loaded.tensors[0].name);
    try std.testing.expectEqualStrings("early", loaded.tensors[1].name);
    try std.testing.expectEqualStrings("late", loaded.tensors[2].name);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, loaded.tensors[1].view.data);
    try std.testing.expectEqualSlices(u8, &.{ 5, 6, 7, 8 }, loaded.tensors[2].view.data);
}

test "safetensors buildHeader JSON-escapes names and roundtrips them" {
    const alloc = std.testing.allocator;
    const special_name = "quote\" slash\\ newline\n snow雪";
    const metas = [_]safetensors.TensorMeta{
        .{ .name = special_name, .dtype = .u8, .shape = &.{1}, .byte_len = 1 },
    };
    const header = try safetensors.buildHeader(alloc, &metas);
    defer alloc.free(header);
    try std.testing.expect(std.mem.indexOf(u8, header, "\\\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, header, "\\\\") != null);
    try std.testing.expect(std.mem.indexOf(u8, header, "\\u000a") != null);

    const bytes = try buildFile(alloc, header, &.{0xa5});
    var loaded = try safetensors.loadFromBytes(alloc, bytes);
    defer loaded.deinit(alloc);
    try std.testing.expectEqualStrings(special_name, loaded.tensors[0].name);
    try std.testing.expectEqual(@as(u8, 0xa5), loaded.tensors[0].view.data[0]);
}

test "safetensors rejects malformed JSON field types and negative values without leaks" {
    try expectLoadError(
        error.MalformedSafetensors,
        \\{"x":7}
    ,
        &.{},
    );
    try expectLoadError(
        error.MalformedSafetensors,
        \\{"__metadata__":[]}
    ,
        &.{},
    );
    try expectLoadError(
        error.MalformedSafetensors,
        \\{"__metadata__":{"bad":7}}
    ,
        &.{},
    );
    try expectLoadError(
        error.MalformedSafetensors,
        \\{"x":{"dtype":7,"shape":[1],"data_offsets":[0,1]}}
    ,
        &.{0},
    );
    try expectLoadError(
        error.MalformedSafetensors,
        \\{"x":{"dtype":"U8","shape":[-1],"data_offsets":[0,0]}}
    ,
        &.{},
    );
    try expectLoadError(
        error.MalformedSafetensors,
        \\{"x":{"dtype":"U8","shape":[1],"data_offsets":["0",1]}}
    ,
        &.{0},
    );
    try expectLoadError(
        error.InvalidDataOffsets,
        \\{"x":{"dtype":"U8","shape":[0],"data_offsets":[1,0]}}
    ,
        &.{},
    );
}

test "safetensors rejects shape overflow and range length mismatch" {
    try expectLoadError(
        error.ShapeOverflow,
        \\{"x":{"dtype":"U32","shape":[9223372036854775807,9223372036854775807],"data_offsets":[0,0]}}
    ,
        &.{},
    );
    try expectLoadError(
        error.ShapeDataMismatch,
        \\{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,7]}}
    ,
        &.{ 0, 0, 0, 0, 0, 0, 0 },
    );
    try expectLoadError(
        error.SafetensorsDataOverflow,
        \\{"x":{"dtype":"U8","shape":[2],"data_offsets":[0,2]}}
    ,
        &.{0},
    );
}

test "safetensors rejects overlap holes and unclaimed trailing data" {
    try expectLoadError(
        error.NonContiguousTensorData,
        \\{"a":{"dtype":"U8","shape":[2],"data_offsets":[0,2]},"b":{"dtype":"U8","shape":[1],"data_offsets":[1,2]}}
    ,
        &.{ 1, 2 },
    );
    try expectLoadError(
        error.NonContiguousTensorData,
        \\{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}}
    ,
        &.{ 1, 0, 2 },
    );
    try expectLoadError(
        error.UnclaimedTensorData,
        \\{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}
    ,
        &.{ 1, 2 },
    );
}

test "safetensors checks header arithmetic and exact prefix length" {
    const alloc = std.testing.allocator;
    var too_short: [7]u8 = @splat(0);
    try std.testing.expectError(
        error.SafetensorsTooShort,
        safetensors.loadFromBytes(alloc, &too_short),
    );

    var overflow: [8]u8 = undefined;
    std.mem.writeInt(u64, &overflow, std.math.maxInt(u64), .little);
    try std.testing.expectError(
        error.SafetensorsHeaderOverflow,
        safetensors.loadFromBytes(alloc, &overflow),
    );

    const header = "{}";
    const bytes = try buildFile(alloc, header, &.{0});
    defer alloc.free(bytes);
    try std.testing.expectError(
        error.InvalidSafetensorsPrefix,
        safetensors.parsePrefix(alloc, bytes),
    );
}

test "safetensors parser enforces resource limits before owned metadata" {
    const alloc = std.testing.allocator;
    const header =
        \\{"long-name":{"dtype":"U8","shape":[1,1],"data_offsets":[0,1]}}
    ;
    const bytes = try buildFile(alloc, header, &.{0});
    defer alloc.free(bytes);
    const prefix = bytes[0 .. 8 + header.len];

    try std.testing.expectError(
        error.SafetensorsFileLimitExceeded,
        safetensors.loadFromBytesWithLimits(
            alloc,
            bytes,
            .{ .max_file_bytes = bytes.len - 1 },
        ),
    );
    try std.testing.expectError(
        error.SafetensorsHeaderLimitExceeded,
        safetensors.parsePrefixWithLimits(
            alloc,
            prefix,
            .{ .max_header_bytes = header.len - 1 },
        ),
    );
    try std.testing.expectError(
        error.NameLimitExceeded,
        safetensors.parsePrefixWithLimits(
            alloc,
            prefix,
            .{ .max_name_bytes = 4 },
        ),
    );
    try std.testing.expectError(
        error.DimensionLimitExceeded,
        safetensors.parsePrefixWithLimits(
            alloc,
            prefix,
            .{ .max_dimensions = 1 },
        ),
    );
    try std.testing.expectError(
        error.TensorDataLimitExceeded,
        safetensors.parsePrefixWithLimits(
            alloc,
            prefix,
            .{ .max_tensor_bytes = 0 },
        ),
    );
    try std.testing.expectError(
        error.TensorLimitExceeded,
        safetensors.parsePrefixWithLimits(
            alloc,
            prefix,
            .{ .max_tensors = 0 },
        ),
    );
}

test "safetensors parser releases every allocation on injected OOM" {
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        loadForAllocationFailureCheck,
        .{},
    );
}

test "safetensors buildHeader validates metadata and offset accumulation" {
    const alloc = std.testing.allocator;
    try std.testing.expectError(
        error.ShapeDataMismatch,
        safetensors.buildHeader(alloc, &.{
            .{ .name = "x", .dtype = .u16, .shape = &.{2}, .byte_len = 3 },
        }),
    );
    try std.testing.expectError(
        error.DuplicateTensorName,
        safetensors.buildHeader(alloc, &.{
            .{ .name = "x", .dtype = .u8, .shape = &.{1}, .byte_len = 1 },
            .{ .name = "x", .dtype = .u8, .shape = &.{1}, .byte_len = 1 },
        }),
    );
    try std.testing.expectError(
        error.ReservedTensorName,
        safetensors.buildHeader(alloc, &.{
            .{ .name = "__metadata__", .dtype = .u8, .shape = &.{0}, .byte_len = 0 },
        }),
    );

    const invalid_utf8 = [_]u8{0xff};
    try std.testing.expectError(
        error.InvalidTensorName,
        safetensors.buildHeader(alloc, &.{
            .{ .name = &invalid_utf8, .dtype = .u8, .shape = &.{0}, .byte_len = 0 },
        }),
    );

    const huge: usize = @intCast(std.math.maxInt(i64));
    try std.testing.expectError(
        error.OffsetOverflow,
        safetensors.buildHeader(alloc, &.{
            .{ .name = "a", .dtype = .u8, .shape = &.{huge}, .byte_len = huge },
            .{ .name = "b", .dtype = .u8, .shape = &.{huge}, .byte_len = huge },
            .{ .name = "c", .dtype = .u8, .shape = &.{huge}, .byte_len = huge },
        }),
    );
}

test "TensorView numelChecked rejects overflow and accepts zero dimensions" {
    const overflow = types.TensorView{
        .data = &.{},
        .shape = &.{ std.math.maxInt(u64), 2 },
        .dtype = .u8,
    };
    try std.testing.expectError(error.ShapeOverflow, overflow.numelChecked());

    const empty = types.TensorView{
        .data = &.{},
        .shape = &.{ 99, 0, std.math.maxInt(u64) },
        .dtype = .f32,
    };
    try std.testing.expectEqual(@as(usize, 0), try empty.numelChecked());
}

test "loadFromPath releases malformed mmap or fallback buffers" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(
        alloc,
        ".zig-cache/tmp/{s}/malformed.safetensors",
        .{&tmp.sub_path},
    );
    defer alloc.free(path);

    const bytes = try buildFile(
        alloc,
        \\{"x":{"dtype":"U8","shape":[2],"data_offsets":[0,1]}}
    ,
        &.{0},
    );
    defer alloc.free(bytes);

    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    {
        defer file.close(io);
        var buffer: [128]u8 = undefined;
        var writer = file.writer(io, &buffer);
        try writer.interface.writeAll(bytes);
        try writer.interface.flush();
    }
    try std.testing.expectError(
        error.ShapeDataMismatch,
        safetensors.loadFromPath(alloc, io, path),
    );
}

test "saveToPath preserves an existing file when validation fails" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    var invalid_data = [_]u8{0};
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(
        alloc,
        ".zig-cache/tmp/{s}/atomic.safetensors",
        .{&tmp.sub_path},
    );
    defer alloc.free(path);

    const original = "existing-valid-target";
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    {
        defer file.close(io);
        var buffer: [64]u8 = undefined;
        var writer = file.writer(io, &buffer);
        try writer.interface.writeAll(original);
        try writer.interface.flush();
    }

    try std.testing.expectError(
        error.ShapeDataMismatch,
        safetensors.saveToPath(alloc, io, path, &.{
            .{
                .name = "bad",
                .view = .{
                    .data = &invalid_data,
                    .shape = &.{2},
                    .dtype = .u8,
                },
            },
        }),
    );
    const retained = try std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        alloc,
        .limited(1024),
    );
    defer alloc.free(retained);
    try std.testing.expectEqualStrings(original, retained);
}

test "ordinary deinit releases a successful loadFromPath value" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(
        alloc,
        ".zig-cache/tmp/{s}/valid.safetensors",
        .{&tmp.sub_path},
    );
    defer alloc.free(path);

    var data = [_]u8{ 1, 2, 3, 4 };
    const shape = [_]u64{data.len};
    try safetensors.saveToPath(alloc, io, path, &.{
        .{
            .name = "weights",
            .view = .{
                .data = &data,
                .shape = &shape,
                .dtype = .u8,
            },
        },
    });
    var loaded = try safetensors.loadFromPath(alloc, io, path);
    defer loaded.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 1), loaded.tensors.len);
    try std.testing.expectEqualSlices(
        u8,
        &data,
        loaded.tensors[0].view.data,
    );
}
