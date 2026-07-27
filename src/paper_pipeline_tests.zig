const std = @import("std");
const dsl = @import("dsl.zig");
const pipeline = @import("paper_pipeline.zig");
const tensor_archive = @import("tensor_archive.zig");

const multi_header =
    \\{"repeat":{"dtype":"F32","shape":[6],"data_offsets":[0,24]},"constant":{"dtype":"U16","shape":[4],"data_offsets":[24,32]},"empty":{"dtype":"U8","shape":[0],"data_offsets":[32,32]}}
;
const single_header =
    \\{"x":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}
;
const tiny_header =
    \\{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}
;

fn makeSafetensors(
    alloc: std.mem.Allocator,
    header: []const u8,
    data: []const u8,
) ![]u8 {
    const total = std.math.add(usize, 8 + header.len, data.len) catch
        return error.IntegerOverflow;
    const output = try alloc.alloc(u8, total);
    std.mem.writeInt(u64, output[0..8], header.len, .little);
    @memcpy(output[8..][0..header.len], header);
    @memcpy(output[8 + header.len ..], data);
    return output;
}

fn writeFile(io: std.Io, path: []const u8, bytes: []const u8) !void {
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buffer: [4096]u8 = undefined;
    var writer = file.writer(io, &buffer);
    try writer.interface.writeAll(bytes);
    try writer.flush();
}

fn readFile(
    alloc: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        alloc,
        .unlimited,
    );
}

test "multiple whole tensors round trip byte-for-byte with one record each" {
    const alloc = std.testing.allocator;
    const data = [_]u8{
        0x00, 0x00, 0x80, 0x3f,
        0x00, 0x00, 0x80, 0xbf,
        0x00, 0x00, 0x80, 0x3f,
        0x00, 0x00, 0x80, 0xbf,
        0x00, 0x00, 0x80, 0x3f,
        0x00, 0x00, 0x80, 0xbf,
        0x34, 0x12, 0x34, 0x12,
        0x34, 0x12, 0x34, 0x12,
    };
    const source = try makeSafetensors(alloc, multi_header, &data);
    defer alloc.free(source);

    var compressed = try pipeline.compressBytes(
        alloc,
        source,
        .{
            .synthesis = .{
                .max_expansions = 96,
                .max_nodes = 8,
                .grammar_options = .{ .max_depth = 1 },
            },
        },
    );
    defer compressed.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 3), compressed.tensors.len);
    try std.testing.expectEqualStrings("repeat", compressed.tensors[0].name);
    try std.testing.expectEqualStrings("constant", compressed.tensors[1].name);
    try std.testing.expectEqualStrings("empty", compressed.tensors[2].name);

    const header = try tensor_archive.parseHeader(
        compressed.archive_bytes,
        .{},
    );
    try std.testing.expectEqual(@as(usize, 3), header.tensor_count);
    try std.testing.expectEqualSlices(
        u8,
        source[0 .. 8 + multi_header.len],
        header.safetensors_prefix,
    );

    var parsed = try tensor_archive.parseStructural(
        alloc,
        compressed.archive_bytes,
        .{},
    );
    defer parsed.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 3), parsed.records.len);
    try std.testing.expect(switch (parsed.records[0].tensor_program.root.kind) {
        .repeat => true,
        else => false,
    });
    try std.testing.expect(switch (parsed.records[1].tensor_program.root.kind) {
        .constant => true,
        else => false,
    });
    try std.testing.expect(switch (parsed.records[2].tensor_program.root.kind) {
        .literal => |literal| literal.count == 0,
        else => false,
    });

    const restored = try pipeline.decompressBytes(
        alloc,
        compressed.archive_bytes,
        .{},
    );
    defer alloc.free(restored);
    try std.testing.expectEqualSlices(u8, source, restored);
}

test "zero synthesis budget stores one exact Lit for the complete tensor" {
    const alloc = std.testing.allocator;
    const source = try makeSafetensors(
        alloc,
        single_header,
        &.{ 4, 3, 2, 1 },
    );
    defer alloc.free(source);

    var compressed = try pipeline.compressBytes(
        alloc,
        source,
        .{ .synthesis = .{ .max_expansions = 0 } },
    );
    defer compressed.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 1), compressed.tensors.len);
    try std.testing.expect(compressed.tensors[0].used_literal_fallback);
    try std.testing.expectEqual(@as(usize, 0), compressed.tensors[0].expanded);

    var parsed = try tensor_archive.parseStructural(
        alloc,
        compressed.archive_bytes,
        .{},
    );
    defer parsed.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 1), parsed.records.len);
    try std.testing.expect(switch (parsed.records[0].tensor_program.root.kind) {
        .literal => |literal| literal.count == 4,
        else => false,
    });

    const restored = try pipeline.decompressBytes(
        alloc,
        compressed.archive_bytes,
        .{},
    );
    defer alloc.free(restored);
    try std.testing.expectEqualSlices(u8, source, restored);
}

test "decompression rejects checksum corruption and trailing bytes" {
    const alloc = std.testing.allocator;
    const source = try makeSafetensors(
        alloc,
        single_header,
        &.{ 1, 2, 3, 4 },
    );
    defer alloc.free(source);
    var compressed = try pipeline.compressBytes(
        alloc,
        source,
        .{ .synthesis = .{ .max_expansions = 0 } },
    );
    defer compressed.deinit(alloc);

    const corrupted = try alloc.dupe(u8, compressed.archive_bytes);
    defer alloc.free(corrupted);
    corrupted[corrupted.len - 1] ^= 0x80;
    try std.testing.expectError(
        error.ChecksumMismatch,
        pipeline.decompressBytes(alloc, corrupted, .{}),
    );

    const trailing = try alloc.alloc(
        u8,
        compressed.archive_bytes.len + 1,
    );
    defer alloc.free(trailing);
    @memcpy(
        trailing[0..compressed.archive_bytes.len],
        compressed.archive_bytes,
    );
    trailing[trailing.len - 1] = 0;
    try std.testing.expectError(
        error.TrailingBytes,
        pipeline.decompressBytes(alloc, trailing, .{}),
    );
}

test "decompression rejects record metadata that disagrees with prefix" {
    const alloc = std.testing.allocator;
    const prefix_source = try makeSafetensors(
        alloc,
        single_header,
        &.{ 9, 8, 7, 6 },
    );
    defer alloc.free(prefix_source);
    const prefix = prefix_source[0 .. 8 + single_header.len];

    var root = try dsl.Program.literal(alloc, 8, &.{ 9, 8, 7, 6 });
    var root_owned = true;
    defer if (root_owned) root.deinit(alloc);
    var tensor_program = try dsl.TensorProgram.init(
        alloc,
        .u8,
        &.{4},
        root,
    );
    root_owned = false;
    defer tensor_program.deinit(alloc);

    const mismatched = try tensor_archive.build(alloc, prefix, &.{
        .{ .name = "not-x", .tensor_program = tensor_program },
    });
    defer alloc.free(mismatched);

    try std.testing.expectError(
        error.MetadataMismatch,
        pipeline.decompressBytes(alloc, mismatched, .{}),
    );
}

test "prefix byte length bounds a mismatched record before execution" {
    const alloc = std.testing.allocator;
    const prefix_source = try makeSafetensors(
        alloc,
        tiny_header,
        &.{9},
    );
    defer alloc.free(prefix_source);
    const prefix = prefix_source[0 .. 8 + tiny_header.len];

    var root = try dsl.Program.literal(alloc, 8, &.{ 1, 2, 3, 4 });
    var root_owned = true;
    defer if (root_owned) root.deinit(alloc);
    var tensor_program = try dsl.TensorProgram.init(
        alloc,
        .u8,
        &.{4},
        root,
    );
    root_owned = false;
    defer tensor_program.deinit(alloc);
    const mismatched = try tensor_archive.build(alloc, prefix, &.{
        .{ .name = "x", .tensor_program = tensor_program },
    });
    defer alloc.free(mismatched);

    try std.testing.expectError(
        error.OutputLimitExceeded,
        pipeline.decompressBytes(alloc, mismatched, .{}),
    );
}

test "pipeline enforces source, archive, and aggregate output limits" {
    const alloc = std.testing.allocator;
    const source = try makeSafetensors(
        alloc,
        single_header,
        &.{ 1, 2, 3, 4 },
    );
    defer alloc.free(source);

    try std.testing.expectError(
        error.SourceLimitExceeded,
        pipeline.compressBytes(
            alloc,
            source,
            .{ .max_source_bytes = source.len - 1 },
        ),
    );
    try std.testing.expectError(
        error.ArchiveLimitExceeded,
        pipeline.compressBytes(
            alloc,
            source,
            .{ .max_archive_bytes = 0 },
        ),
    );

    var compressed = try pipeline.compressBytes(
        alloc,
        source,
        .{ .synthesis = .{ .max_expansions = 0 } },
    );
    defer compressed.deinit(alloc);

    try std.testing.expectError(
        error.ArchiveLimitExceeded,
        pipeline.decompressBytes(
            alloc,
            compressed.archive_bytes,
            .{ .max_archive_bytes = compressed.archive_bytes.len - 1 },
        ),
    );
    try std.testing.expectError(
        error.OutputLimitExceeded,
        pipeline.decompressBytes(
            alloc,
            compressed.archive_bytes,
            .{ .max_output_bytes = source.len - 1 },
        ),
    );
}

test "file APIs atomically compress decompress and verify complete tensors" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const source_path = try std.fmt.allocPrint(
        alloc,
        ".zig-cache/tmp/{s}/source.safetensors",
        .{&tmp.sub_path},
    );
    defer alloc.free(source_path);
    const archive_path = try std.fmt.allocPrint(
        alloc,
        ".zig-cache/tmp/{s}/model.brta",
        .{&tmp.sub_path},
    );
    defer alloc.free(archive_path);
    const restored_path = try std.fmt.allocPrint(
        alloc,
        ".zig-cache/tmp/{s}/restored.safetensors",
        .{&tmp.sub_path},
    );
    defer alloc.free(restored_path);
    const corrupt_path = try std.fmt.allocPrint(
        alloc,
        ".zig-cache/tmp/{s}/corrupt.brta",
        .{&tmp.sub_path},
    );
    defer alloc.free(corrupt_path);
    const failed_output_path = try std.fmt.allocPrint(
        alloc,
        ".zig-cache/tmp/{s}/must-remain.safetensors",
        .{&tmp.sub_path},
    );
    defer alloc.free(failed_output_path);
    const wrong_source_path = try std.fmt.allocPrint(
        alloc,
        ".zig-cache/tmp/{s}/wrong.safetensors",
        .{&tmp.sub_path},
    );
    defer alloc.free(wrong_source_path);

    const data = [_]u8{
        0x00, 0x00, 0x80, 0x3f,
        0x00, 0x00, 0x80, 0xbf,
        0x00, 0x00, 0x80, 0x3f,
        0x00, 0x00, 0x80, 0xbf,
        0x00, 0x00, 0x80, 0x3f,
        0x00, 0x00, 0x80, 0xbf,
        0x34, 0x12, 0x34, 0x12,
        0x34, 0x12, 0x34, 0x12,
    };
    const source = try makeSafetensors(alloc, multi_header, &data);
    defer alloc.free(source);
    try writeFile(io, source_path, source);
    try writeFile(io, archive_path, "stale archive");
    var serial = try pipeline.compressBytes(
        alloc,
        source,
        .{ .synthesis = .{ .max_expansions = 0 } },
    );
    defer serial.deinit(alloc);

    var compressed = try pipeline.compressFile(
        alloc,
        io,
        source_path,
        archive_path,
        .{
            .synthesis = .{ .max_expansions = 0 },
            .workers = 2,
        },
    );
    defer compressed.deinit(alloc);
    try std.testing.expectEqual(source.len, compressed.source_bytes);
    try std.testing.expectEqual(@as(usize, 3), compressed.tensors.len);

    const archive_bytes = try readFile(alloc, io, archive_path);
    defer alloc.free(archive_bytes);
    try std.testing.expectEqual(
        archive_bytes.len,
        compressed.archive_bytes,
    );
    try std.testing.expectEqualSlices(
        u8,
        &tensor_archive.MAGIC,
        archive_bytes[0..tensor_archive.MAGIC.len],
    );
    try std.testing.expectEqualSlices(
        u8,
        serial.archive_bytes,
        archive_bytes,
    );

    try std.testing.expectError(
        error.ArchiveLimitExceeded,
        pipeline.compressFile(
            alloc,
            io,
            source_path,
            archive_path,
            .{
                .synthesis = .{ .max_expansions = 0 },
                .max_archive_bytes = 1,
            },
        ),
    );
    const preserved_archive = try readFile(alloc, io, archive_path);
    defer alloc.free(preserved_archive);
    try std.testing.expectEqualSlices(
        u8,
        archive_bytes,
        preserved_archive,
    );

    const verified = try pipeline.verifyFile(
        alloc,
        io,
        archive_path,
        source_path,
        .{ .workers = 2 },
    );
    try std.testing.expectEqual(archive_bytes.len, verified.archive_bytes);
    try std.testing.expectEqual(source.len, verified.output_bytes);
    try std.testing.expectEqual(@as(usize, 3), verified.tensor_count);

    try writeFile(io, restored_path, "stale output");
    const decompressed = try pipeline.decompressFile(
        alloc,
        io,
        archive_path,
        restored_path,
        .{ .workers = 2 },
    );
    try std.testing.expectEqual(source.len, decompressed.output_bytes);
    try std.testing.expectEqual(@as(usize, 3), decompressed.tensor_count);
    const restored = try readFile(alloc, io, restored_path);
    defer alloc.free(restored);
    try std.testing.expectEqualSlices(u8, source, restored);

    const wrong_source = try alloc.dupe(u8, source);
    defer alloc.free(wrong_source);
    wrong_source[wrong_source.len - 1] ^= 1;
    try writeFile(io, wrong_source_path, wrong_source);
    try std.testing.expectError(
        error.SourceMismatch,
        pipeline.verifyFile(
            alloc,
            io,
            archive_path,
            wrong_source_path,
            .{},
        ),
    );

    const parsed_header = try tensor_archive.parseHeader(archive_bytes, .{});
    var truncated_at = parsed_header.next_offset;
    _ = try tensor_archive.nextTensorRecordFrame(
        archive_bytes,
        &truncated_at,
        .{},
    );
    try writeFile(io, corrupt_path, archive_bytes[0..truncated_at]);
    try writeFile(io, failed_output_path, "keep me");
    try std.testing.expectError(
        error.Truncated,
        pipeline.decompressFile(
            alloc,
            io,
            corrupt_path,
            failed_output_path,
            .{ .workers = 2 },
        ),
    );
    const preserved_truncated = try readFile(
        alloc,
        io,
        failed_output_path,
    );
    defer alloc.free(preserved_truncated);
    try std.testing.expectEqualSlices(
        u8,
        "keep me",
        preserved_truncated,
    );

    const corrupted = try alloc.dupe(u8, archive_bytes);
    defer alloc.free(corrupted);
    corrupted[corrupted.len - 1] ^= 0x80;
    try writeFile(io, corrupt_path, corrupted);
    try writeFile(io, failed_output_path, "keep me");
    try std.testing.expectError(
        error.ChecksumMismatch,
        pipeline.decompressFile(
            alloc,
            io,
            corrupt_path,
            failed_output_path,
            .{},
        ),
    );
    const preserved = try readFile(alloc, io, failed_output_path);
    defer alloc.free(preserved);
    try std.testing.expectEqualSlices(u8, "keep me", preserved);
}
