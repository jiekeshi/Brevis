const std = @import("std");
const dsl = @import("dsl.zig");
const grammar_prior = @import("grammar_prior.zig");
const calibration = @import("calibration.zig");
const checkpoint = @import("checkpoint.zig");
const safetensors = @import("safetensors.zig");
const synthesizer = @import("synthesizer.zig");
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

    var compressed = try checkpoint.compressBytes(
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

    const restored = try checkpoint.decompressBytes(
        alloc,
        compressed.archive_bytes,
        .{},
    );
    defer alloc.free(restored);
    try std.testing.expectEqualSlices(u8, source, restored);
}

test "compression learns a checkpoint-local prior unless one is supplied" {
    const alloc = std.testing.allocator;
    try std.testing.expectEqual(
        calibration.DEFAULT_TENSORS,
        (checkpoint.CompressOptions{}).max_calibration_tensors,
    );
    const header =
        \\{"first":{"dtype":"U8","shape":[8],"data_offsets":[0,8]},"second":{"dtype":"U8","shape":[8],"data_offsets":[8,16]}}
    ;
    const source = try makeSafetensors(
        alloc,
        header,
        &.{ 7, 7, 7, 7, 7, 7, 7, 7, 1, 2, 1, 2, 1, 2, 1, 2 },
    );
    defer alloc.free(source);
    const synthesis = synthesizer.Options{
        .max_expansions = 16,
        .max_nodes = 8,
        .grammar_options = .{ .max_depth = 1 },
    };

    const calibration_source = try alloc.dupe(u8, source);
    var loaded = try safetensors.loadFromBytes(alloc, calibration_source);
    defer loaded.deinit(alloc);
    var trained = try calibration.train(alloc, loaded.tensors, .{
        .max_tensors = 1,
        .synthesis = synthesis,
    });
    defer trained.deinit(alloc);
    try std.testing.expect(!trained.prior.isEmpty());

    var automatic = try checkpoint.compressBytes(alloc, source, .{
        .synthesis = synthesis,
        .max_calibration_tensors = 1,
    });
    defer automatic.deinit(alloc);

    var guided_synthesis = synthesis;
    guided_synthesis.rule_model = &trained.prior;
    var explicit = try checkpoint.compressBytes(alloc, source, .{
        .synthesis = guided_synthesis,
        .max_calibration_tensors = 0,
    });
    defer explicit.deinit(alloc);
    try std.testing.expectEqualSlices(
        u8,
        explicit.archive_bytes,
        automatic.archive_bytes,
    );
    for (explicit.tensors, automatic.tensors) |expected, actual| {
        try std.testing.expectEqual(expected.expanded, actual.expanded);
        try std.testing.expectEqual(
            expected.completed_candidates,
            actual.completed_candidates,
        );
    }

    var invalid_prior: grammar_prior.Prior = .{
        .config = .{ .learned_denominator = 0 },
    };
    defer invalid_prior.deinit(alloc);
    var invalid_synthesis = synthesis;
    invalid_synthesis.rule_model = &invalid_prior;
    try std.testing.expectError(
        error.InvalidConfig,
        checkpoint.compressBytes(alloc, source, .{
            .synthesis = invalid_synthesis,
            .max_calibration_tensors = 0,
        }),
    );
}

test "one expansion uses a learned PHOG completion instead of the float seed" {
    const alloc = std.testing.allocator;
    const header =
        \\{"weights":{"dtype":"F32","shape":[256],"data_offsets":[0,1024]}}
    ;
    var data: [1024]u8 = undefined;
    for (0..256) |index| {
        const word: u32 = if (index & 1 == 0) 0x3f80_0000 else 0xbf80_0000;
        std.mem.writeInt(u32, data[index * 4 ..][0..4], word, .little);
    }
    const source = try makeSafetensors(alloc, header, &data);
    defer alloc.free(source);

    const requested = synthesizer.Options{
        .max_expansions = 1,
        .max_nodes = 8,
        .grammar_options = .{
            .max_depth = 1,
            .max_repeat_period = 8,
            .max_concat_splits = 0,
            .max_map_constants = 0,
            .max_rotations = 0,
            .max_field_splits = 0,
        },
    };
    var automatic = try checkpoint.compressBytes(alloc, source, .{
        .synthesis = requested,
        .max_calibration_tensors = 1,
    });
    defer automatic.deinit(alloc);

    const calibration_source = try alloc.dupe(u8, source);
    var loaded = try safetensors.loadFromBytes(alloc, calibration_source);
    defer loaded.deinit(alloc);
    var teacher_options = requested;
    teacher_options.max_expansions = 6;
    teacher_options.seed_float_fields = true;
    var trained = try calibration.train(alloc, loaded.tensors, .{
        .max_tensors = 1,
        .synthesis = teacher_options,
    });
    defer trained.deinit(alloc);

    var guided_options = requested;
    guided_options.seed_float_fields = false;
    guided_options.rule_model = &trained.prior;
    var explicit = try checkpoint.compressBytes(alloc, source, .{
        .synthesis = guided_options,
        .max_calibration_tensors = 0,
    });
    defer explicit.deinit(alloc);

    try std.testing.expectEqualSlices(
        u8,
        explicit.archive_bytes,
        automatic.archive_bytes,
    );
    var parsed = try tensor_archive.parseStructural(
        alloc,
        automatic.archive_bytes,
        .{},
    );
    defer parsed.deinit(alloc);
    try std.testing.expect(switch (parsed.records[0].tensor_program.root.kind) {
        .repeat => true,
        else => false,
    });
}

test "one-expansion PHOG learns float fields from a seeded teacher" {
    const alloc = std.testing.allocator;
    const header =
        \\{"weights":{"dtype":"F32","shape":[256],"data_offsets":[0,1024]}}
    ;
    var data: [1024]u8 = undefined;
    for (0..256) |index| {
        const sign: u32 = @intCast((index & 1) << 31);
        const mantissa: u32 = @intCast(index * 7919);
        std.mem.writeInt(
            u32,
            data[index * 4 ..][0..4],
            sign | 0x3f80_0000 | mantissa,
            .little,
        );
    }
    const source = try makeSafetensors(alloc, header, &data);
    defer alloc.free(source);

    var compressed = try checkpoint.compressBytes(alloc, source, .{
        .synthesis = .{
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
        },
        .max_calibration_tensors = 1,
    });
    defer compressed.deinit(alloc);

    var parsed = try tensor_archive.parseStructural(
        alloc,
        compressed.archive_bytes,
        .{},
    );
    defer parsed.deinit(alloc);
    try std.testing.expect(switch (parsed.records[0].tensor_program.root.kind) {
        .merge => |operation| switch (operation) {
            .float_fields => true,
            else => false,
        },
        else => false,
    });
    try std.testing.expectEqual(@as(usize, 1), compressed.tensors[0].expanded);
    try std.testing.expectEqual(
        @as(usize, 2),
        compressed.tensors[0].completed_candidates,
    );
}

test "zero synthesis budget stores one exact Lit for the complete tensor" {
    const alloc = std.testing.allocator;
    const source = try makeSafetensors(
        alloc,
        single_header,
        &.{ 4, 3, 2, 1 },
    );
    defer alloc.free(source);

    var compressed = try checkpoint.compressBytes(
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

    const restored = try checkpoint.decompressBytes(
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
    var compressed = try checkpoint.compressBytes(
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
        checkpoint.decompressBytes(alloc, corrupted, .{}),
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
        checkpoint.decompressBytes(alloc, trailing, .{}),
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
        checkpoint.decompressBytes(alloc, mismatched, .{}),
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
        checkpoint.decompressBytes(alloc, mismatched, .{}),
    );
}

test "checkpoint enforces source, archive, and aggregate output limits" {
    const alloc = std.testing.allocator;
    const source = try makeSafetensors(
        alloc,
        single_header,
        &.{ 1, 2, 3, 4 },
    );
    defer alloc.free(source);

    try std.testing.expectError(
        error.SourceLimitExceeded,
        checkpoint.compressBytes(
            alloc,
            source,
            .{ .max_source_bytes = source.len - 1 },
        ),
    );
    try std.testing.expectError(
        error.ArchiveLimitExceeded,
        checkpoint.compressBytes(
            alloc,
            source,
            .{ .max_archive_bytes = 0 },
        ),
    );

    var compressed = try checkpoint.compressBytes(
        alloc,
        source,
        .{ .synthesis = .{ .max_expansions = 0 } },
    );
    defer compressed.deinit(alloc);

    try std.testing.expectError(
        error.ArchiveLimitExceeded,
        checkpoint.decompressBytes(
            alloc,
            compressed.archive_bytes,
            .{ .max_archive_bytes = compressed.archive_bytes.len - 1 },
        ),
    );
    try std.testing.expectError(
        error.OutputLimitExceeded,
        checkpoint.decompressBytes(
            alloc,
            compressed.archive_bytes,
            .{ .max_output_bytes = source.len - 1 },
        ),
    );
}

test "file APIs compress decompress and verify complete tensors" {
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
        ".zig-cache/tmp/{s}/model.brv",
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
        ".zig-cache/tmp/{s}/corrupt.brv",
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
    const source_dot_alias = try std.fmt.allocPrint(
        alloc,
        ".zig-cache/tmp/{s}/./source.safetensors",
        .{&tmp.sub_path},
    );
    defer alloc.free(source_dot_alias);
    const source_hardlink = try std.fmt.allocPrint(
        alloc,
        ".zig-cache/tmp/{s}/source-hardlink.safetensors",
        .{&tmp.sub_path},
    );
    defer alloc.free(source_hardlink);
    const archive_hardlink = try std.fmt.allocPrint(
        alloc,
        ".zig-cache/tmp/{s}/archive-hardlink.brv",
        .{&tmp.sub_path},
    );
    defer alloc.free(archive_hardlink);

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
    try std.testing.expectError(
        error.InputOutputPathConflict,
        checkpoint.compressFile(
            alloc,
            io,
            source_path,
            source_path,
            .{},
        ),
    );
    const preserved_source = try readFile(alloc, io, source_path);
    defer alloc.free(preserved_source);
    try std.testing.expectEqualSlices(u8, source, preserved_source);
    try std.testing.expectError(
        error.InputOutputPathConflict,
        checkpoint.compressFile(
            alloc,
            io,
            source_path,
            source_dot_alias,
            .{},
        ),
    );
    try std.Io.Dir.hardLink(
        .cwd(),
        source_path,
        .cwd(),
        source_hardlink,
        io,
        .{},
    );
    try std.testing.expectError(
        error.InputOutputPathConflict,
        checkpoint.compressFile(
            alloc,
            io,
            source_path,
            source_hardlink,
            .{},
        ),
    );

    try writeFile(io, archive_path, "stale archive");
    var serial = try checkpoint.compressBytes(
        alloc,
        source,
        .{ .synthesis = .{ .max_expansions = 1 } },
    );
    defer serial.deinit(alloc);

    var compressed = try checkpoint.compressFile(
        alloc,
        io,
        source_path,
        archive_path,
        .{
            .synthesis = .{ .max_expansions = 1 },
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
        error.InputOutputPathConflict,
        checkpoint.decompressFile(
            alloc,
            io,
            archive_path,
            archive_path,
            .{},
        ),
    );
    const preserved_archive = try readFile(alloc, io, archive_path);
    defer alloc.free(preserved_archive);
    try std.testing.expectEqualSlices(
        u8,
        archive_bytes,
        preserved_archive,
    );
    try std.Io.Dir.hardLink(
        .cwd(),
        archive_path,
        .cwd(),
        archive_hardlink,
        io,
        .{},
    );
    try std.testing.expectError(
        error.InputOutputPathConflict,
        checkpoint.decompressFile(
            alloc,
            io,
            archive_path,
            archive_hardlink,
            .{},
        ),
    );
    const preserved_hardlink = try readFile(alloc, io, archive_hardlink);
    defer alloc.free(preserved_hardlink);
    try std.testing.expectEqualSlices(
        u8,
        archive_bytes,
        preserved_hardlink,
    );

    const verified = try checkpoint.verifyFile(
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
    const decompressed = try checkpoint.decompressFile(
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
        checkpoint.verifyFile(
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
    try std.testing.expectError(
        error.Truncated,
        checkpoint.decompressFile(
            alloc,
            io,
            corrupt_path,
            failed_output_path,
            .{ .workers = 2 },
        ),
    );

    const corrupted = try alloc.dupe(u8, archive_bytes);
    defer alloc.free(corrupted);
    corrupted[corrupted.len - 1] ^= 0x80;
    try writeFile(io, corrupt_path, corrupted);
    try std.testing.expectError(
        error.ChecksumMismatch,
        checkpoint.decompressFile(
            alloc,
            io,
            corrupt_path,
            failed_output_path,
            .{},
        ),
    );
}
