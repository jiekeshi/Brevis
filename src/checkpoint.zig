//! Safetensors checkpoint compression and decompression.
//!
//! The unit of synthesis and archival is exactly one complete safetensors
//! tensor. This module intentionally has no block, template, sampling, or
//! re-ranking layer.

const std = @import("std");
const builtin = @import("builtin");
const dsl = @import("dsl.zig");
const calibration = @import("calibration.zig");
const safetensors = @import("safetensors.zig");
const synthesizer = @import("synthesizer.zig");
const tensor_archive = @import("tensor_archive.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const DEFAULT_WORKERS: usize = 32;
const ONE_EXPANSION_TEACHER_TENSORS: usize = 4;
const ONE_EXPANSION_TEACHER_BUDGET: usize = 6;

pub const CompressOptions = struct {
    synthesis: synthesizer.Options = .{ .seed_float_fields = false },
    max_calibration_tensors: usize = calibration.DEFAULT_TENSORS,
    workers: usize = DEFAULT_WORKERS,
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
    workers: usize = DEFAULT_WORKERS,
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
        null,
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

/// Mmap a safetensors source when available, then synthesize and write one
/// complete tensor frame at a time.
pub fn compressFile(
    alloc: Allocator,
    io: Io,
    source_path: []const u8,
    archive_path: []const u8,
    options: CompressOptions,
) !CompressSummary {
    if (std.mem.eql(u8, source_path, archive_path))
        return error.InputOutputPathConflict;

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

    const archive_file = try createDistinctOutputFile(
        io,
        archive_path,
        source_file.identity,
    );
    defer archive_file.close(io);
    var file_buffer: [64 * 1024]u8 = undefined;
    var file_writer = archive_file.writer(io, &file_buffer);
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
        io,
        options,
    );
    errdefer summary.deinit(alloc);
    try file_writer.flush();
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
        null,
    );
    return output.toOwnedSlice(alloc);
}

/// Decode one verified tensor at a time into an output file.
pub fn decompressFile(
    alloc: Allocator,
    io: Io,
    archive_path: []const u8,
    output_path: []const u8,
    limits: DecompressLimits,
) !DecodeSummary {
    if (std.mem.eql(u8, archive_path, output_path))
        return error.InputOutputPathConflict;

    var archive_file = try loadFileBytes(
        alloc,
        io,
        archive_path,
        limits.max_archive_bytes,
        .archive,
    );
    defer archive_file.deinit(alloc, io);

    const output_file = try createDistinctOutputFile(
        io,
        output_path,
        archive_file.identity,
    );
    defer output_file.close(io);
    var file_buffer: [64 * 1024]u8 = undefined;
    var file_writer = output_file.writer(io, &file_buffer);
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
        io,
    );
    try file_writer.flush();
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
        io,
    );
}

fn compressLoaded(
    alloc: Allocator,
    loaded: *const safetensors.Loaded,
    safetensors_prefix: []const u8,
    source_bytes: usize,
    sink: *OutputSink,
    io: ?Io,
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

    var learned_prior: ?calibration.Result = null;
    defer if (learned_prior) |*result| result.deinit(alloc);
    var compression_options = options;
    if (compression_options.synthesis.rule_model == null and
        compression_options.synthesis.max_expansions != 0 and
        options.max_calibration_tensors != 0)
    {
        var teacher = options.synthesis;
        var calibration_tensors = options.max_calibration_tensors;
        if (teacher.max_expansions == 1) {
            teacher.max_expansions = ONE_EXPANSION_TEACHER_BUDGET;
            teacher.seed_float_fields = true;
            compression_options.synthesis.seed_float_fields = false;
            calibration_tensors = @min(
                calibration_tensors,
                ONE_EXPANSION_TEACHER_TENSORS,
            );
        }
        const calibration_options = calibration.Options{
            .max_tensors = calibration_tensors,
            .synthesis = teacher,
        };
        learned_prior = if (io) |threaded_io|
            try calibration.trainParallel(
                alloc,
                threaded_io,
                loaded.tensors,
                calibration_options,
                options.workers,
            )
        else
            try calibration.train(
                alloc,
                loaded.tensors,
                calibration_options,
            );
        compression_options.synthesis.rule_model = &learned_prior.?.prior;
    }

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

    if (io != null and options.workers > 1 and loaded.tensors.len > 1) {
        try compressTensorsParallel(
            alloc,
            io.?,
            loaded.tensors,
            stats,
            &initialized_stats,
            sink,
            compression_options,
        );
    } else {
        for (loaded.tensors) |tensor| {
            var encoded = try encodeTensor(
                alloc,
                tensor,
                compression_options,
            );
            defer encoded.deinit(alloc);
            stats[initialized_stats] = try finishEncodedTensor(
                alloc,
                tensor.name,
                encoded,
                sink,
            );
            initialized_stats += 1;
        }
    }

    return .{
        .source_bytes = source_bytes,
        .archive_bytes = sink.written,
        .tensors = stats,
    };
}

const EncodedTensor = struct {
    record: tensor_archive.PreparedTensorRecord,
    bytecode: []u8,
    source_bytes: usize,
    serialized_program_bytes: usize,
    expanded: usize,
    completed_candidates: usize,
    status: synthesizer.SearchStatus,
    used_literal_fallback: bool,

    fn deinit(self: *EncodedTensor, alloc: Allocator) void {
        self.record.deinit(alloc);
        alloc.free(self.bytecode);
        self.bytecode = &.{};
    }
};

const EncodeOutcome = union(enum) {
    success: EncodedTensor,
    failure: anyerror,

    fn deinit(self: *EncodeOutcome, alloc: Allocator) void {
        switch (self.*) {
            .success => |*encoded| encoded.deinit(alloc),
            .failure => {},
        }
    }
};

const IndexedEncodeOutcome = struct {
    index: usize,
    outcome: EncodeOutcome,
};

const EncodeCompletion = union(enum) {
    done: IndexedEncodeOutcome,
};

fn encodeTensor(
    alloc: Allocator,
    tensor: safetensors.Tensor,
    options: CompressOptions,
) !EncodedTensor {
    const elements = try tensor.view.numelChecked();
    const source_bytes = std.math.mul(
        usize,
        elements,
        tensor.view.dtype.elemSize(),
    ) catch return error.IntegerOverflow;
    if (source_bytes != tensor.view.data.len)
        return error.TensorByteLengthMismatch;
    if (source_bytes > options.max_tensor_bytes)
        return error.TensorLimitExceeded;

    const target = types.Stream{
        .data = @constCast(tensor.view.data),
        .count = elements,
        .bits_per_elem = tensor.view.dtype.bitWidth(),
        .owns_data = false,
    };
    var synthesis = try synthesizer.synthesizeBorrowingTarget(
        alloc,
        target,
        tensor.view.dtype,
        options.synthesis,
    );
    var bytecode_owned = true;
    defer if (bytecode_owned) alloc.free(synthesis.serialized_program);
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

    const record = try tensor_archive.prepareTensorRecordForSource(
        alloc,
        tensor.name,
        tensor_program,
        synthesis.serialized_program,
        tensor.view.data,
    );
    bytecode_owned = false;
    return .{
        .record = record,
        .bytecode = synthesis.serialized_program,
        .source_bytes = source_bytes,
        .serialized_program_bytes = synthesis.serialized_bytes,
        .expanded = synthesis.expanded,
        .completed_candidates = synthesis.completed_candidates,
        .status = synthesis.status,
        .used_literal_fallback = synthesis.used_literal_fallback,
    };
}

fn encodeTensorTask(
    tensor: safetensors.Tensor,
    options: CompressOptions,
) EncodeOutcome {
    const worker_alloc = std.heap.smp_allocator;
    return .{ .success = encodeTensor(
        worker_alloc,
        tensor,
        options,
    ) catch |err| return .{ .failure = err } };
}

fn encodeIndexedTensorTask(
    index: usize,
    tensor: safetensors.Tensor,
    options: CompressOptions,
) IndexedEncodeOutcome {
    return .{
        .index = index,
        .outcome = encodeTensorTask(tensor, options),
    };
}

fn finishEncodedTensor(
    alloc: Allocator,
    name: []const u8,
    encoded: EncodedTensor,
    sink: *OutputSink,
) !TensorStat {
    try sink.writeAll(encoded.record.prefix);
    try sink.writeAll(encoded.record.bytecode);
    try sink.writeAll(&encoded.record.checksum);
    return .{
        .name = try alloc.dupe(u8, name),
        .source_bytes = encoded.source_bytes,
        .archive_record_bytes = encoded.record.encodedLen(),
        .serialized_program_bytes = encoded.serialized_program_bytes,
        .expanded = encoded.expanded,
        .completed_candidates = encoded.completed_candidates,
        .status = encoded.status,
        .used_literal_fallback = encoded.used_literal_fallback,
    };
}

fn compressTensorsParallel(
    alloc: Allocator,
    io: Io,
    tensors: []const safetensors.Tensor,
    stats: []TensorStat,
    initialized_stats: *usize,
    sink: *OutputSink,
    options: CompressOptions,
) !void {
    const workers = @min(options.workers, tensors.len);
    const completion_buffer = try alloc.alloc(EncodeCompletion, workers);
    defer alloc.free(completion_buffer);
    var tasks = Io.Select(EncodeCompletion).init(io, completion_buffer);
    defer while (tasks.cancel()) |completion| {
        var outcome = completion.done.outcome;
        outcome.deinit(std.heap.smp_allocator);
    };

    const pending = try alloc.alloc(?EncodeOutcome, tensors.len);
    defer alloc.free(pending);
    @memset(pending, null);
    defer for (pending) |entry| {
        if (entry) |value| {
            var outcome = value;
            outcome.deinit(std.heap.smp_allocator);
        }
    };

    const lookahead = std.math.mul(
        usize,
        workers,
        2,
    ) catch std.math.maxInt(usize);
    var next_to_schedule: usize = 0;
    var next_to_emit: usize = 0;
    var running: usize = 0;

    while (next_to_emit < tensors.len) {
        while (running < workers and
            next_to_schedule < tensors.len and
            next_to_schedule - next_to_emit < lookahead)
        {
            tasks.async(
                .done,
                encodeIndexedTensorTask,
                .{
                    next_to_schedule,
                    tensors[next_to_schedule],
                    options,
                },
            );
            next_to_schedule += 1;
            running += 1;
        }

        const completed = (try tasks.await()).done;
        running -= 1;
        std.debug.assert(pending[completed.index] == null);
        pending[completed.index] = completed.outcome;

        while (next_to_emit < tensors.len and
            pending[next_to_emit] != null)
        {
            var outcome = pending[next_to_emit].?;
            pending[next_to_emit] = null;
            defer outcome.deinit(std.heap.smp_allocator);
            switch (outcome) {
                .success => |encoded| {
                    stats[initialized_stats.*] = try finishEncodedTensor(
                        alloc,
                        tensors[next_to_emit].name,
                        encoded,
                        sink,
                    );
                    initialized_stats.* += 1;
                },
                .failure => |err| return err,
            }
            next_to_emit += 1;
        }
    }
}

fn decodeArchiveToSink(
    alloc: Allocator,
    encoded: []const u8,
    limits: DecompressLimits,
    sink: *OutputSink,
    io: ?Io,
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
    if (io != null and limits.workers > 1 and metadata.tensors.len > 1) {
        try decodeRecordsParallel(
            alloc,
            io.?,
            encoded,
            &position,
            metadata.tensors,
            limits,
            sink,
        );
    } else {
        for (metadata.tensors) |*expected| {
            const frame = try tensor_archive.nextTensorRecordFrame(
                encoded,
                &position,
                limits.archive,
            );
            var verified = try decodeExpectedRecord(
                alloc,
                frame,
                expected,
                limits.archive,
            );
            defer verified.deinit(alloc);
            try sink.writeAll(verified.decoded.data);
        }
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

const DecodeOutcome = union(enum) {
    success: tensor_archive.VerifiedTensorRecord,
    failure: anyerror,

    fn deinit(self: *DecodeOutcome, alloc: Allocator) void {
        switch (self.*) {
            .success => |*verified| verified.deinit(alloc),
            .failure => {},
        }
    }
};

const IndexedDecodeOutcome = struct {
    index: usize,
    outcome: DecodeOutcome,
};

const DecodeCompletion = union(enum) {
    done: IndexedDecodeOutcome,
};

fn decodeExpectedRecord(
    alloc: Allocator,
    frame: []const u8,
    expected: *const safetensors.HeaderTensor,
    limits: tensor_archive.DecodeLimits,
) !tensor_archive.VerifiedTensorRecord {
    if (expected.name.len > limits.max_name_bytes)
        return error.NameLimitExceeded;
    if (expected.shape.len > limits.max_dimensions)
        return error.DimensionLimitExceeded;
    const expected_bytes = try expected.expectedByteLen();
    if (expected_bytes > limits.max_tensor_output_bytes)
        return error.OutputLimitExceeded;

    var record_limits = limits;
    record_limits.max_tensor_output_bytes = @min(
        record_limits.max_tensor_output_bytes,
        expected_bytes,
    );
    record_limits.program.max_output_bytes = @min(
        record_limits.program.max_output_bytes,
        expected_bytes,
    );
    var verified = try tensor_archive.decodeVerifiedTensorRecord(
        alloc,
        frame,
        record_limits,
    );
    errdefer verified.deinit(alloc);
    const record = verified.record;
    if (!std.mem.eql(u8, record.name, expected.name) or
        record.tensor_program.dtype != expected.dtype or
        !std.mem.eql(u64, record.tensor_program.shape, expected.shape))
    {
        return error.MetadataMismatch;
    }
    if (verified.decoded.data.len != expected_bytes)
        return error.TensorByteLengthMismatch;
    return verified;
}

fn decodeRecordTask(
    frame: []const u8,
    expected: *const safetensors.HeaderTensor,
    limits: tensor_archive.DecodeLimits,
) DecodeOutcome {
    const worker_alloc = std.heap.smp_allocator;
    return .{ .success = decodeExpectedRecord(
        worker_alloc,
        frame,
        expected,
        limits,
    ) catch |err| return .{ .failure = err } };
}

fn decodeIndexedRecordTask(
    index: usize,
    frame: []const u8,
    expected: *const safetensors.HeaderTensor,
    limits: tensor_archive.DecodeLimits,
) IndexedDecodeOutcome {
    return .{
        .index = index,
        .outcome = decodeRecordTask(frame, expected, limits),
    };
}

fn decodeRecordsParallel(
    alloc: Allocator,
    io: Io,
    encoded: []const u8,
    position: *usize,
    expected_tensors: []const safetensors.HeaderTensor,
    limits: DecompressLimits,
    sink: *OutputSink,
) !void {
    const frames = try alloc.alloc([]const u8, expected_tensors.len);
    defer alloc.free(frames);
    for (frames) |*frame| {
        frame.* = try tensor_archive.nextTensorRecordFrame(
            encoded,
            position,
            limits.archive,
        );
    }

    const workers = @min(limits.workers, expected_tensors.len);
    const completion_buffer = try alloc.alloc(DecodeCompletion, workers);
    defer alloc.free(completion_buffer);
    var tasks = Io.Select(DecodeCompletion).init(io, completion_buffer);
    defer while (tasks.cancel()) |completion| {
        var outcome = completion.done.outcome;
        outcome.deinit(std.heap.smp_allocator);
    };

    const pending = try alloc.alloc(?DecodeOutcome, expected_tensors.len);
    defer alloc.free(pending);
    @memset(pending, null);
    defer for (pending) |entry| {
        if (entry) |value| {
            var outcome = value;
            outcome.deinit(std.heap.smp_allocator);
        }
    };

    const lookahead = std.math.mul(
        usize,
        workers,
        2,
    ) catch std.math.maxInt(usize);
    var next_to_schedule: usize = 0;
    var next_to_emit: usize = 0;
    var running: usize = 0;

    while (next_to_emit < expected_tensors.len) {
        while (running < workers and
            next_to_schedule < expected_tensors.len and
            next_to_schedule - next_to_emit < lookahead)
        {
            tasks.async(
                .done,
                decodeIndexedRecordTask,
                .{
                    next_to_schedule,
                    frames[next_to_schedule],
                    &expected_tensors[next_to_schedule],
                    limits.archive,
                },
            );
            next_to_schedule += 1;
            running += 1;
        }

        const completed = (try tasks.await()).done;
        running -= 1;
        std.debug.assert(pending[completed.index] == null);
        pending[completed.index] = completed.outcome;

        while (next_to_emit < expected_tensors.len and
            pending[next_to_emit] != null)
        {
            var outcome = pending[next_to_emit].?;
            pending[next_to_emit] = null;
            defer outcome.deinit(std.heap.smp_allocator);
            switch (outcome) {
                .success => |verified| try sink.writeAll(
                    verified.decoded.data,
                ),
                .failure => |err| return err,
            }
            next_to_emit += 1;
        }
    }
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
    identity: FileIdentity,

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
    const identity: FileIdentity = .{
        .device = try fileDevice(file),
        .inode = initial_stat.inode,
    };
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
        return .{
            .bytes = mapping.memory,
            .mmap = mapping,
            .identity = identity,
        };
    } else |_| {
        const bytes = try alloc.alloc(u8, file_len);
        errdefer alloc.free(bytes);
        var read_buffer: [64 * 1024]u8 = undefined;
        var reader = file.reader(io, &read_buffer);
        try reader.interface.readSliceAll(bytes);
        const final_stat = try file.stat(io);
        if (final_stat.size != initial_stat.size)
            return error.FileChangedDuringRead;
        return .{ .bytes = bytes, .identity = identity };
    }
}

const FileIdentity = struct {
    device: ?u64,
    inode: std.Io.File.INode,

    fn eql(a: FileIdentity, b: FileIdentity) bool {
        if (a.inode != b.inode) return false;
        if (a.device) |device|
            if (b.device) |other| return device == other;
        return true;
    }
};

fn createDistinctOutputFile(
    io: Io,
    path: []const u8,
    input_identity: FileIdentity,
) !std.Io.File {
    const file = try std.Io.Dir.cwd().createFile(
        io,
        path,
        .{ .truncate = false },
    );
    errdefer file.close(io);
    const stat = try file.stat(io);
    const output_identity: FileIdentity = .{
        .device = try fileDevice(file),
        .inode = stat.inode,
    };
    if (input_identity.eql(output_identity))
        return error.InputOutputPathConflict;
    try file.setLength(io, 0);
    return file;
}

fn fileDevice(file: std.Io.File) !?u64 {
    return switch (comptime builtin.os.tag) {
        .linux => linuxFileDevice(file),
        .windows => windowsFileDevice(file),
        .wasi => wasiFileDevice(file),
        else => posixFileDevice(file),
    };
}

fn linuxFileDevice(file: std.Io.File) !?u64 {
    const linux = std.os.linux;
    var stat = std.mem.zeroes(linux.Statx);
    while (true) switch (linux.errno(linux.statx(
        file.handle,
        "",
        linux.AT.EMPTY_PATH,
        linux.STATX.BASIC_STATS,
        &stat,
    ))) {
        .SUCCESS => return @as(u64, stat.dev_major) << 32 | stat.dev_minor,
        .INTR => continue,
        else => |err| return std.posix.unexpectedErrno(err),
    };
}

fn windowsFileDevice(file: std.Io.File) !?u64 {
    const windows = std.os.windows;
    var io_status: windows.IO_STATUS_BLOCK = undefined;
    var volume: windows.FILE.FS_VOLUME_INFORMATION = undefined;
    switch (windows.ntdll.NtQueryVolumeInformationFile(
        file.handle,
        &io_status,
        &volume,
        @sizeOf(windows.FILE.FS_VOLUME_INFORMATION),
        .Volume,
    )) {
        .SUCCESS, .BUFFER_OVERFLOW => return volume.VolumeSerialNumber,
        else => |status| return windows.unexpectedStatus(status),
    }
}

fn wasiFileDevice(file: std.Io.File) !?u64 {
    const wasi = std.os.wasi;
    var stat: wasi.filestat_t = undefined;
    while (true) switch (wasi.fd_filestat_get(file.handle, &stat)) {
        .SUCCESS => return stat.dev,
        .INTR => continue,
        else => return error.Unexpected,
    };
}

fn posixFileDevice(file: std.Io.File) !?u64 {
    if (comptime std.posix.Stat == void) return null;
    var stat = std.mem.zeroes(std.posix.Stat);
    while (true) switch (std.posix.errno(std.posix.system.fstat(
        file.handle,
        &stat,
    ))) {
        .SUCCESS => return @intCast(stat.dev),
        .INTR => continue,
        else => |err| return std.posix.unexpectedErrno(err),
    };
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
