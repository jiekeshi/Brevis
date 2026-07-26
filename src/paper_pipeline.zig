//! In-memory and atomic-file end-to-end pipeline for the paper implementation.
//!
//! The unit of synthesis and archival is exactly one complete safetensors
//! tensor. This module intentionally has no block, template, sampling, or
//! re-ranking layer.

const std = @import("std");
const dsl = @import("dsl.zig");
const safetensors = @import("safetensors.zig");
const synthesizer = @import("synthesizer.zig");
const tensor_archive = @import("tensor_archive.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const CompressOptions = struct {
    synthesis: synthesizer.Options = .{},
    max_source_bytes: usize = types.defaultLargeByteLimit,
    max_prefix_bytes: usize = 64 * 1024 * 1024,
    max_tensors: usize = 1_000_000,
    max_name_bytes: usize = 1024 * 1024,
    max_dimensions: usize = 1024,
    max_tensor_bytes: usize = types.defaultLargeByteLimit,
    max_archive_bytes: usize = types.defaultLargeByteLimit,
};

pub const DecompressLimits = struct {
    archive: tensor_archive.DecodeLimits = .{},
    max_archive_bytes: usize = types.defaultLargeByteLimit,
    max_output_bytes: usize = types.defaultLargeByteLimit,
};

pub const TensorStat = struct {
    name: []u8,
    source_bytes: usize,
    archive_record_bytes: usize,
    serialized_program_bytes: usize,
    expanded: usize,
    completed_candidates: usize,
    status: synthesizer.SearchStatus,
    used_literal_fallback: bool,

    pub fn deinit(self: *TensorStat, alloc: Allocator) void {
        alloc.free(self.name);
        self.name = &.{};
    }
};

pub const CompressSummary = struct {
    source_bytes: usize,
    archive_bytes: usize,
    tensors: []TensorStat,

    pub fn deinit(self: *CompressSummary, alloc: Allocator) void {
        deinitTensorStats(alloc, self.tensors);
        self.tensors = &.{};
    }
};

pub const CompressResult = struct {
    archive_bytes: []u8,
    source_bytes: usize,
    tensors: []TensorStat,

    pub fn deinit(self: *CompressResult, alloc: Allocator) void {
        deinitTensorStats(alloc, self.tensors);
        alloc.free(self.archive_bytes);
        self.tensors = &.{};
        self.archive_bytes = &.{};
    }
};

pub const DecodeSummary = struct {
    archive_bytes: usize,
    output_bytes: usize,
    tensor_count: usize,
};

pub const Options = CompressOptions;
pub const Result = CompressResult;

/// Parse a complete safetensors value and synthesize one self-contained
/// `TensorProgram` for every complete physical-word stream.
pub fn compressBytes(
    alloc: Allocator,
    source: []const u8,
    options: CompressOptions,
) !CompressResult {
    if (source.len > options.max_source_bytes)
        return error.SourceLimitExceeded;

    const prefix_len = try safetensorsPrefixLength(source);
    if (prefix_len > options.max_prefix_bytes)
        return error.PrefixLimitExceeded;
    const source_copy = try alloc.dupe(u8, source);
    var source_copy_owned = true;
    errdefer if (source_copy_owned) alloc.free(source_copy);
    var loaded = try safetensors.loadFromBytesWithLimits(
        alloc,
        source_copy,
        safetensorsLimitsForCompression(options),
    );
    source_copy_owned = false;
    defer loaded.deinit(alloc);

    var encoded: std.ArrayList(u8) = .empty;
    errdefer encoded.deinit(alloc);
    var sink = OutputSink.memory(
        alloc,
        &encoded,
        options.max_archive_bytes,
        .archive,
    );
    var summary = try compressLoaded(
        alloc,
        &loaded,
        source[0..prefix_len],
        source.len,
        &sink,
        options,
    );
    errdefer summary.deinit(alloc);
    const archive_bytes = try encoded.toOwnedSlice(alloc);

    return .{
        .archive_bytes = archive_bytes,
        .source_bytes = summary.source_bytes,
        .tensors = summary.tensors,
    };
}

/// Mmap a safetensors source when available, synthesize and write one complete
/// tensor frame at a time, then atomically replace `archive_path`.
pub fn compressFile(
    alloc: Allocator,
    io: Io,
    source_path: []const u8,
    archive_path: []const u8,
    options: CompressOptions,
) !CompressSummary {
    var source_file = try loadFileBytes(
        alloc,
        io,
        source_path,
        options.max_source_bytes,
        .source,
    );
    var source_file_owned = true;
    defer if (source_file_owned) source_file.deinit(alloc, io);

    const prefix_len = try safetensorsPrefixLength(source_file.bytes);
    if (prefix_len > options.max_prefix_bytes)
        return error.PrefixLimitExceeded;
    var loaded = try safetensors.loadFromBytesWithLimits(
        alloc,
        source_file.bytes,
        safetensorsLimitsForCompression(options),
    );
    loaded.mmap = source_file.mmap;
    loaded.mmap_io = io;
    source_file.release();
    source_file_owned = false;
    defer loaded.deinit(alloc);

    var atomic = try std.Io.Dir.cwd().createFileAtomic(
        io,
        archive_path,
        .{ .replace = true },
    );
    defer atomic.deinit(io);
    var file_buffer: [64 * 1024]u8 = undefined;
    var file_writer = atomic.file.writer(io, &file_buffer);
    var sink = OutputSink.file(
        alloc,
        &file_writer,
        options.max_archive_bytes,
        .archive,
    );

    var summary = try compressLoaded(
        alloc,
        &loaded,
        loaded.bytes[0..prefix_len],
        loaded.bytes.len,
        &sink,
        options,
    );
    errdefer summary.deinit(alloc);
    try file_writer.flush();
    try atomic.file.sync(io);
    try atomic.replace(io);
    return summary;
}

/// Decode records in archive order, bind each one to the exact prefix metadata,
/// verify its program checksum, and reconstruct the original prefix plus
/// physical tensor bytes.
pub fn decompressBytes(
    alloc: Allocator,
    encoded: []const u8,
    limits: DecompressLimits,
) ![]u8 {
    var output: std.ArrayList(u8) = .empty;
    errdefer output.deinit(alloc);
    var sink = OutputSink.memory(
        alloc,
        &output,
        limits.max_output_bytes,
        .output,
    );
    _ = try decodeArchiveToSink(
        alloc,
        encoded,
        limits,
        &sink,
    );
    return output.toOwnedSlice(alloc);
}

/// Decode one verified tensor at a time into an atomic output file.
pub fn decompressFile(
    alloc: Allocator,
    io: Io,
    archive_path: []const u8,
    output_path: []const u8,
    limits: DecompressLimits,
) !DecodeSummary {
    var archive_file = try loadFileBytes(
        alloc,
        io,
        archive_path,
        limits.max_archive_bytes,
        .archive,
    );
    defer archive_file.deinit(alloc, io);

    var atomic = try std.Io.Dir.cwd().createFileAtomic(
        io,
        output_path,
        .{ .replace = true },
    );
    defer atomic.deinit(io);
    var file_buffer: [64 * 1024]u8 = undefined;
    var file_writer = atomic.file.writer(io, &file_buffer);
    var sink = OutputSink.file(
        alloc,
        &file_writer,
        limits.max_output_bytes,
        .output,
    );

    const summary = try decodeArchiveToSink(
        alloc,
        archive_file.bytes,
        limits,
        &sink,
    );
    try file_writer.flush();
    try atomic.file.sync(io);
    try atomic.replace(io);
    return summary;
}

/// Verify an archive against an expected safetensors file without writing an
/// output file. Both inputs use mmap with a checked-size allocation fallback.
pub fn verifyFile(
    alloc: Allocator,
    io: Io,
    archive_path: []const u8,
    expected_source_path: []const u8,
    limits: DecompressLimits,
) !DecodeSummary {
    var archive_file = try loadFileBytes(
        alloc,
        io,
        archive_path,
        limits.max_archive_bytes,
        .archive,
    );
    defer archive_file.deinit(alloc, io);
    var expected_file = try loadFileBytes(
        alloc,
        io,
        expected_source_path,
        limits.max_output_bytes,
        .output,
    );
    defer expected_file.deinit(alloc, io);

    var sink = OutputSink.compare(
        alloc,
        expected_file.bytes,
        limits.max_output_bytes,
    );
    return decodeArchiveToSink(
        alloc,
        archive_file.bytes,
        limits,
        &sink,
    );
}

fn compressLoaded(
    alloc: Allocator,
    loaded: *const safetensors.Loaded,
    safetensors_prefix: []const u8,
    source_bytes: usize,
    sink: *OutputSink,
    options: CompressOptions,
) !CompressSummary {
    if (sink.written != 0) return error.NonEmptyOutputSink;
    if (safetensors_prefix.len > options.max_prefix_bytes)
        return error.PrefixLimitExceeded;
    if (loaded.tensors.len > options.max_tensors)
        return error.TensorLimitExceeded;
    try validatePhysicalLayout(
        loaded.bytes,
        safetensors_prefix.len,
        loaded.tensors,
    );

    const stats = try alloc.alloc(TensorStat, loaded.tensors.len);
    var initialized_stats: usize = 0;
    errdefer {
        for (stats[0..initialized_stats]) |*stat| stat.deinit(alloc);
        alloc.free(stats);
    }

    const header = try tensor_archive.encodeHeader(
        alloc,
        safetensors_prefix,
        loaded.tensors.len,
    );
    defer alloc.free(header);
    try sink.writeAll(header);

    for (loaded.tensors) |tensor| {
        const elements = try tensor.view.numelChecked();
        const expected_bytes = std.math.mul(
            usize,
            elements,
            tensor.view.dtype.elemSize(),
        ) catch return error.IntegerOverflow;
        if (expected_bytes != tensor.view.data.len)
            return error.TensorByteLengthMismatch;
        if (expected_bytes > options.max_tensor_bytes)
            return error.TensorLimitExceeded;

        const target = types.Stream{
            // Grammar/synthesis only read target streams. The explicit cast
            // stays inside this adapter because public safetensors views are
            // correctly read-only, including when backed by a read-only mmap.
            .data = @constCast(tensor.view.data),
            .count = elements,
            .bits_per_elem = tensor.view.dtype.bitWidth(),
            .owns_data = false,
        };

        // This is deliberately the sole synthesis call in the per-tensor
        // path: `target` is the complete physical tensor stream.
        var synthesis = try synthesizer.synthesize(
            alloc,
            target,
            tensor.view.dtype,
            options.synthesis,
        );
        var root_owned = true;
        defer if (root_owned) synthesis.program.deinit(alloc);

        var tensor_program = try dsl.TensorProgram.init(
            alloc,
            tensor.view.dtype,
            tensor.view.shape,
            synthesis.program,
        );
        root_owned = false;
        defer tensor_program.deinit(alloc);

        const frame = try tensor_archive.encodeTensorRecord(
            alloc,
            tensor.name,
            tensor_program,
        );
        defer alloc.free(frame);
        try sink.writeAll(frame);

        stats[initialized_stats] = .{
            .name = try alloc.dupe(u8, tensor.name),
            .source_bytes = expected_bytes,
            .archive_record_bytes = frame.len,
            .serialized_program_bytes = synthesis.serialized_bytes,
            .expanded = synthesis.expanded,
            .completed_candidates = synthesis.completed_candidates,
            .status = synthesis.status,
            .used_literal_fallback = synthesis.used_literal_fallback,
        };
        initialized_stats += 1;
    }

    return .{
        .source_bytes = source_bytes,
        .archive_bytes = sink.written,
        .tensors = stats,
    };
}

fn decodeArchiveToSink(
    alloc: Allocator,
    encoded: []const u8,
    limits: DecompressLimits,
    sink: *OutputSink,
) !DecodeSummary {
    if (sink.written != 0) return error.NonEmptyOutputSink;
    if (encoded.len > limits.max_archive_bytes)
        return error.ArchiveLimitExceeded;

    const header = try tensor_archive.parseHeader(encoded, limits.archive);
    var metadata = try safetensors.parsePrefixWithLimits(
        alloc,
        header.safetensors_prefix,
        .{
            .max_file_bytes = limits.archive.max_prefix_bytes,
            .max_header_bytes = limits.archive.max_prefix_bytes,
            .max_tensors = limits.archive.max_tensors,
            .max_name_bytes = limits.archive.max_name_bytes,
            .max_dimensions = limits.archive.max_dimensions,
            .max_tensor_bytes = limits.archive.max_tensor_output_bytes,
        },
    );
    defer metadata.deinit(alloc);
    if (metadata.tensors.len != header.tensor_count)
        return error.MetadataMismatch;

    const data_len = std.math.cast(usize, metadata.data_len) orelse
        return error.IntegerOverflow;
    const expected_output_len = std.math.add(
        usize,
        header.safetensors_prefix.len,
        data_len,
    ) catch return error.IntegerOverflow;
    if (expected_output_len > limits.max_output_bytes)
        return error.OutputLimitExceeded;
    try sink.ensureTotalCapacity(expected_output_len);
    try sink.writeAll(header.safetensors_prefix);

    var position = header.next_offset;
    for (metadata.tensors) |expected| {
        if (expected.name.len > limits.archive.max_name_bytes)
            return error.NameLimitExceeded;
        if (expected.shape.len > limits.archive.max_dimensions)
            return error.DimensionLimitExceeded;
        const expected_record_bytes = try expected.expectedByteLen();
        if (expected_record_bytes > limits.archive.max_tensor_output_bytes)
            return error.OutputLimitExceeded;
        // Bind resource use to the trusted prefix metadata before executing a
        // record. A mismatched record must not allocate according to its own
        // attacker-controlled shape and only then fail the metadata check.
        var record_limits = limits.archive;
        record_limits.max_tensor_output_bytes = @min(
            record_limits.max_tensor_output_bytes,
            expected_record_bytes,
        );
        record_limits.program.max_output_bytes = @min(
            record_limits.program.max_output_bytes,
            expected_record_bytes,
        );
        var verified = try tensor_archive.nextVerifiedTensorRecord(
            alloc,
            encoded,
            &position,
            record_limits,
        );
        defer verified.deinit(alloc);
        const record = verified.record;

        if (!std.mem.eql(u8, record.name, expected.name) or
            record.tensor_program.dtype != expected.dtype or
            !std.mem.eql(u64, record.tensor_program.shape, expected.shape))
        {
            return error.MetadataMismatch;
        }

        const tensor_type = try record.tensor_program.validate();
        const record_bytes = std.math.mul(
            usize,
            tensor_type.elements,
            tensor_type.dtype.elemSize(),
        ) catch return error.IntegerOverflow;
        if (record_bytes != expected_record_bytes)
            return error.MetadataMismatch;
        if (verified.decoded.data.len != record_bytes)
            return error.TensorByteLengthMismatch;
        try sink.writeAll(verified.decoded.data);
    }
    if (position != encoded.len) return error.TrailingBytes;
    if (sink.written != expected_output_len)
        return error.MetadataMismatch;
    try sink.finish();

    return .{
        .archive_bytes = encoded.len,
        .output_bytes = sink.written,
        .tensor_count = metadata.tensors.len,
    };
}

fn deinitTensorStats(alloc: Allocator, tensors: []TensorStat) void {
    for (tensors) |*tensor| tensor.deinit(alloc);
    alloc.free(tensors);
}

fn safetensorsLimitsForCompression(options: CompressOptions) safetensors.Limits {
    return .{
        .max_file_bytes = options.max_source_bytes,
        .max_header_bytes = options.max_prefix_bytes,
        .max_tensors = options.max_tensors,
        .max_name_bytes = options.max_name_bytes,
        .max_dimensions = options.max_dimensions,
        .max_tensor_bytes = options.max_tensor_bytes,
    };
}

const LimitKind = enum {
    source,
    archive,
    output,
};

const OutputSink = struct {
    const CompareState = struct {
        expected: []const u8,
        position: usize = 0,
    };

    const Target = union(enum) {
        memory: *std.ArrayList(u8),
        file: *std.Io.File.Writer,
        compare: CompareState,
    };

    alloc: Allocator,
    target: Target,
    limit: usize,
    limit_kind: LimitKind,
    written: usize = 0,

    fn memory(
        alloc: Allocator,
        output: *std.ArrayList(u8),
        limit: usize,
        limit_kind: LimitKind,
    ) OutputSink {
        return .{
            .alloc = alloc,
            .target = .{ .memory = output },
            .limit = limit,
            .limit_kind = limit_kind,
        };
    }

    fn file(
        alloc: Allocator,
        output: *std.Io.File.Writer,
        limit: usize,
        limit_kind: LimitKind,
    ) OutputSink {
        return .{
            .alloc = alloc,
            .target = .{ .file = output },
            .limit = limit,
            .limit_kind = limit_kind,
        };
    }

    fn compare(
        alloc: Allocator,
        expected: []const u8,
        limit: usize,
    ) OutputSink {
        return .{
            .alloc = alloc,
            .target = .{ .compare = .{ .expected = expected } },
            .limit = limit,
            .limit_kind = .output,
        };
    }

    fn ensureTotalCapacity(self: *OutputSink, total: usize) !void {
        try enforceLimit(total, self.limit, self.limit_kind);
        switch (self.target) {
            .memory => |output| try output.ensureTotalCapacity(
                self.alloc,
                total,
            ),
            .file, .compare => {},
        }
    }

    fn writeAll(self: *OutputSink, bytes: []const u8) !void {
        const new_total = std.math.add(
            usize,
            self.written,
            bytes.len,
        ) catch return error.IntegerOverflow;
        try enforceLimit(new_total, self.limit, self.limit_kind);

        switch (self.target) {
            .memory => |output| try output.appendSlice(self.alloc, bytes),
            .file => |output| output.interface.writeAll(bytes) catch {
                if (output.err) |err| return err;
                return error.WriteFailed;
            },
            .compare => |*comparison| {
                if (comparison.position > comparison.expected.len or
                    bytes.len > comparison.expected.len - comparison.position)
                {
                    return error.SourceMismatch;
                }
                const expected = comparison.expected[comparison.position..][0..bytes.len];
                if (!std.mem.eql(u8, bytes, expected))
                    return error.SourceMismatch;
                comparison.position += bytes.len;
            },
        }
        self.written = new_total;
    }

    fn finish(self: *OutputSink) !void {
        switch (self.target) {
            .compare => |comparison| {
                if (comparison.position != comparison.expected.len)
                    return error.SourceMismatch;
            },
            .memory, .file => {},
        }
    }
};

const FileBytes = struct {
    bytes: []u8,
    mmap: ?std.Io.File.MemoryMap = null,

    fn release(self: *FileBytes) void {
        self.bytes = &.{};
        self.mmap = null;
    }

    fn deinit(self: *FileBytes, alloc: Allocator, io: Io) void {
        if (self.mmap) |*mapping| {
            mapping.destroy(io);
        } else {
            alloc.free(self.bytes);
        }
        self.release();
    }
};

fn loadFileBytes(
    alloc: Allocator,
    io: Io,
    path: []const u8,
    limit: usize,
    limit_kind: LimitKind,
) !FileBytes {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const initial_stat = try file.stat(io);
    const file_len = std.math.cast(usize, initial_stat.size) orelse
        return error.IntegerOverflow;
    try enforceLimit(file_len, limit, limit_kind);

    if (std.Io.File.MemoryMap.create(io, file, .{
        .len = file_len,
        .protection = .{ .read = true, .write = false },
        .populate = false,
    })) |mapping_value| {
        var mapping = mapping_value;
        errdefer mapping.destroy(io);
        const final_stat = try file.stat(io);
        if (final_stat.size != initial_stat.size)
            return error.FileChangedDuringRead;
        return .{ .bytes = mapping.memory, .mmap = mapping };
    } else |_| {
        const bytes = try alloc.alloc(u8, file_len);
        errdefer alloc.free(bytes);
        var read_buffer: [64 * 1024]u8 = undefined;
        var reader = file.reader(io, &read_buffer);
        try reader.interface.readSliceAll(bytes);
        const final_stat = try file.stat(io);
        if (final_stat.size != initial_stat.size)
            return error.FileChangedDuringRead;
        return .{ .bytes = bytes };
    }
}

fn enforceLimit(
    value: usize,
    limit: usize,
    kind: LimitKind,
) !void {
    if (value <= limit) return;
    return switch (kind) {
        .source => error.SourceLimitExceeded,
        .archive => error.ArchiveLimitExceeded,
        .output => error.OutputLimitExceeded,
    };
}

fn safetensorsPrefixLength(bytes: []const u8) !usize {
    if (bytes.len < 8) return error.SafetensorsTooShort;
    const header_len = std.math.cast(
        usize,
        std.mem.readInt(u64, bytes[0..8], .little),
    ) orelse return error.IntegerOverflow;
    const prefix_len = std.math.add(usize, 8, header_len) catch
        return error.IntegerOverflow;
    if (prefix_len > bytes.len) return error.SafetensorsHeaderOverflow;
    return prefix_len;
}

fn validatePhysicalLayout(
    bytes: []const u8,
    prefix_len: usize,
    tensors: []const safetensors.Tensor,
) !void {
    if (prefix_len > bytes.len) return error.SafetensorsHeaderOverflow;
    const base = @intFromPtr(bytes.ptr);
    var expected_offset = prefix_len;

    for (tensors) |tensor| {
        const pointer = @intFromPtr(tensor.view.data.ptr);
        if (pointer < base) return error.NonContiguousTensorData;
        const offset = pointer - base;
        if (offset != expected_offset) return error.NonContiguousTensorData;
        if (tensor.view.data.len > bytes.len - expected_offset)
            return error.SafetensorsDataOverflow;
        expected_offset = std.math.add(
            usize,
            expected_offset,
            tensor.view.data.len,
        ) catch return error.IntegerOverflow;
    }
    if (expected_offset != bytes.len)
        return error.NonContiguousTensorData;
}
