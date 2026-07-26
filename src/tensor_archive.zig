//! Versioned whole-tensor archive for the paper implementation.
//!
//! Each frame owns exactly one semantic `TensorProgram`. There are no block
//! records, shared plans, back-references, or decoder-side learned state.

const std = @import("std");
const dsl = @import("dsl.zig");
const interpreter = @import("interpreter.zig");
const program_format = @import("program_format.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;

pub const MAGIC = [_]u8{ 'B', 'R', 'T', 'A' };
pub const VERSION: u8 = 1;
const RECORD_TAG: u8 = 1;
const CHECKSUM_BYTES = std.crypto.hash.sha2.Sha256.digest_length;
const MIN_RECORD_BODY_BYTES = 1 + 1 + 1 + 1 + 1 + CHECKSUM_BYTES;
const MIN_FRAMED_RECORD_BYTES = 1 + MIN_RECORD_BODY_BYTES;

/// Stable archive IDs. These deliberately do not reuse the in-memory enum
/// ordinals or the program format's private conversion functions.
pub const DtypeWireId = enum(u8) {
    f16 = 0x01,
    bf16 = 0x02,
    f32 = 0x03,
    u8 = 0x04,
    u16 = 0x05,
    u32 = 0x06,
    i8 = 0x07,
    i16 = 0x08,
    i32 = 0x09,
    f8_e4m3 = 0x0a,
    f8_e5m2 = 0x0b,
};

pub const DecodeLimits = struct {
    max_tensors: usize = 1_000_000,
    max_prefix_bytes: usize = 64 * 1024 * 1024,
    max_record_bytes: usize = types.defaultLargeByteLimit,
    max_name_bytes: usize = 1024 * 1024,
    max_dimensions: usize = 1024,
    max_program_bytes: usize = types.defaultLargeByteLimit,
    max_tensor_output_bytes: usize = types.defaultLargeByteLimit,
    program: program_format.DecodeLimits = .{
        .max_nodes = 1_000_000,
        .max_depth = 256,
        .max_output_bytes = types.defaultTensorByteLimit,
        .max_literal_bytes = types.defaultTensorByteLimit,
        .max_execution_work_bytes = types.defaultExecutionByteLimit,
    },
};

pub const ArchiveError = Allocator.Error ||
    program_format.DecodeError ||
    interpreter.ExecuteError ||
    dsl.TensorValidationError ||
    error{
        BadMagic,
        UnsupportedVersion,
        UnknownRecord,
        UnknownDtype,
        InvalidSafetensorsPrefix,
        Truncated,
        TrailingBytes,
        TrailingRecordData,
        OverlongUleb128,
        IntegerOverflow,
        TensorLimitExceeded,
        PrefixLimitExceeded,
        RecordLimitExceeded,
        NameLimitExceeded,
        DimensionLimitExceeded,
        ProgramLimitExceeded,
        OutputLimitExceeded,
        ChecksumMismatch,
    };

/// A non-owning header view. The prefix points into the byte slice passed to
/// `parseHeader`.
pub const Header = struct {
    safetensors_prefix: []const u8,
    tensor_count: usize,
    next_offset: usize,
};

/// An independently executable, allocator-owned whole-tensor record.
pub const TensorRecord = struct {
    name: []u8,
    tensor_program: dsl.TensorProgram,
    checksum: [CHECKSUM_BYTES]u8,

    pub fn deinit(self: *TensorRecord, alloc: Allocator) void {
        self.tensor_program.deinit(alloc);
        alloc.free(self.name);
        self.name = &.{};
    }
};

/// One independently decoded record together with the physical stream whose
/// checksum has already been verified. This lets streaming consumers avoid
/// executing a valid program a second time while preserving the strict
/// behavior of the existing record APIs.
pub const VerifiedTensorRecord = struct {
    record: TensorRecord,
    decoded: types.Stream,

    pub fn deinit(self: *VerifiedTensorRecord, alloc: Allocator) void {
        self.decoded.deinit(alloc);
        self.record.deinit(alloc);
    }
};

pub const TensorInput = struct {
    name: []const u8,
    tensor_program: dsl.TensorProgram,
};

/// Fully owned convenience result. It remains valid after the encoded archive
/// byte slice is released.
pub const Parsed = struct {
    safetensors_prefix: []u8,
    records: []TensorRecord,

    pub fn deinit(self: *Parsed, alloc: Allocator) void {
        for (self.records) |*record| record.deinit(alloc);
        alloc.free(self.records);
        alloc.free(self.safetensors_prefix);
        self.records = &.{};
        self.safetensors_prefix = &.{};
    }
};

/// Encode the streamable archive prefix. `safetensors_prefix` is the original
/// eight-byte JSON length followed by exactly that many JSON bytes.
pub fn encodeHeader(
    alloc: Allocator,
    safetensors_prefix: []const u8,
    tensor_count: usize,
) ArchiveError![]u8 {
    try validateSafetensorsPrefix(safetensors_prefix);

    var output: std.ArrayList(u8) = .empty;
    errdefer output.deinit(alloc);
    var emitter = Emitter{ .allocator = alloc, .output = &output };
    try emitter.writeAll(&MAGIC);
    try emitter.writeByte(VERSION);
    try emitter.writeUleb128(try usizeToU64(tensor_count));
    try emitter.writeUleb128(try usizeToU64(safetensors_prefix.len));
    try emitter.writeAll(safetensors_prefix);
    return output.toOwnedSlice(alloc);
}

/// Encode one complete length-delimited tensor frame. The checksum covers the
/// exact decoded physical bytes, not the program bytecode.
pub fn encodeTensorRecord(
    alloc: Allocator,
    name: []const u8,
    tensor_program: dsl.TensorProgram,
) ![]u8 {
    _ = try tensor_program.validate();

    const bytecode = try program_format.serialize(alloc, tensor_program.root);
    defer alloc.free(bytecode);

    var decoded = try interpreter.executeTensor(alloc, tensor_program);
    defer decoded.deinit(alloc);
    const checksum = sha256(decoded.data);

    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(alloc);
    var body_emitter = Emitter{ .allocator = alloc, .output = &body };
    try body_emitter.writeByte(RECORD_TAG);
    try body_emitter.writeUleb128(try usizeToU64(name.len));
    try body_emitter.writeAll(name);
    try body_emitter.writeByte(@intFromEnum(dtypeToWire(tensor_program.dtype)));
    try body_emitter.writeUleb128(try usizeToU64(tensor_program.shape.len));
    for (tensor_program.shape) |dimension|
        try body_emitter.writeUleb128(dimension);
    try body_emitter.writeUleb128(try usizeToU64(bytecode.len));
    try body_emitter.writeAll(bytecode);
    try body_emitter.writeAll(&checksum);

    var frame: std.ArrayList(u8) = .empty;
    errdefer frame.deinit(alloc);
    var frame_emitter = Emitter{ .allocator = alloc, .output = &frame };
    try frame_emitter.writeUleb128(try usizeToU64(body.items.len));
    try frame_emitter.writeAll(body.items);
    return frame.toOwnedSlice(alloc);
}

/// Convenience builder over the two streaming encoding operations.
pub fn build(
    alloc: Allocator,
    safetensors_prefix: []const u8,
    tensors: []const TensorInput,
) ![]u8 {
    const header = try encodeHeader(alloc, safetensors_prefix, tensors.len);
    defer alloc.free(header);

    var output: std.ArrayList(u8) = .empty;
    errdefer output.deinit(alloc);
    try output.appendSlice(alloc, header);
    for (tensors) |tensor| {
        const frame = try encodeTensorRecord(
            alloc,
            tensor.name,
            tensor.tensor_program,
        );
        defer alloc.free(frame);
        try output.appendSlice(alloc, frame);
    }
    return output.toOwnedSlice(alloc);
}

pub fn parseHeader(bytes: []const u8, limits: DecodeLimits) ArchiveError!Header {
    var reader = Reader{ .bytes = bytes };
    const magic = try reader.take(MAGIC.len);
    if (!std.mem.eql(u8, magic, &MAGIC)) return error.BadMagic;
    if (try reader.readByte() != VERSION) return error.UnsupportedVersion;

    const tensor_count = try reader.readUsize();
    if (tensor_count > limits.max_tensors) return error.TensorLimitExceeded;
    const prefix_len = try reader.readUsize();
    if (prefix_len > limits.max_prefix_bytes) return error.PrefixLimitExceeded;
    const prefix = try reader.take(prefix_len);
    try validateSafetensorsPrefix(prefix);

    // Do not allocate a record array when the remaining archive could not
    // possibly contain the declared number of minimal frames.
    if (tensor_count > (bytes.len - reader.pos) / MIN_FRAMED_RECORD_BYTES)
        return error.Truncated;

    return .{
        .safetensors_prefix = prefix,
        .tensor_count = tensor_count,
        .next_offset = reader.pos,
    };
}

/// Decode one frame at `position`, advancing only after a successful decode.
pub fn nextTensorRecord(
    alloc: Allocator,
    bytes: []const u8,
    position: *usize,
    limits: DecodeLimits,
) ArchiveError!TensorRecord {
    var verified = try nextVerifiedTensorRecord(
        alloc,
        bytes,
        position,
        limits,
    );
    verified.decoded.deinit(alloc);
    return verified.record;
}

/// Decode and checksum-verify one frame at `position`, returning the already
/// executed physical stream and advancing only after complete success.
pub fn nextVerifiedTensorRecord(
    alloc: Allocator,
    bytes: []const u8,
    position: *usize,
    limits: DecodeLimits,
) ArchiveError!VerifiedTensorRecord {
    if (position.* > bytes.len) return error.Truncated;
    var reader = Reader{ .bytes = bytes, .pos = position.* };
    const body_len = try reader.readUsize();
    if (body_len > limits.max_record_bytes) return error.RecordLimitExceeded;
    const body = try reader.take(body_len);
    var verified = try decodeRecordBodyVerified(alloc, body, limits);
    errdefer verified.deinit(alloc);
    position.* = reader.pos;
    return verified;
}

/// Decode an isolated, length-delimited record frame.
pub fn decodeTensorRecord(
    alloc: Allocator,
    frame: []const u8,
    limits: DecodeLimits,
) ArchiveError!TensorRecord {
    var position: usize = 0;
    var record = try nextTensorRecord(alloc, frame, &position, limits);
    errdefer record.deinit(alloc);
    if (position != frame.len) return error.TrailingBytes;
    return record;
}

/// Structurally parse an entire archive and reject bytes after the declared
/// records. This low-level framing API verifies record checksums but does not
/// bind their names/dtypes/shapes to the embedded safetensors JSON. Use
/// `paper_pipeline.decompress*` for a complete archive read.
pub fn parseStructural(
    alloc: Allocator,
    bytes: []const u8,
    limits: DecodeLimits,
) ArchiveError!Parsed {
    const header = try parseHeader(bytes, limits);
    const prefix = try alloc.dupe(u8, header.safetensors_prefix);
    errdefer alloc.free(prefix);

    const records = try alloc.alloc(TensorRecord, header.tensor_count);
    var initialized: usize = 0;
    errdefer {
        for (records[0..initialized]) |*record| record.deinit(alloc);
        alloc.free(records);
    }

    var position = header.next_offset;
    for (records) |*record| {
        record.* = try nextTensorRecord(alloc, bytes, &position, limits);
        initialized += 1;
    }
    if (position != bytes.len) return error.TrailingBytes;

    return .{
        .safetensors_prefix = prefix,
        .records = records,
    };
}

/// Execute a record and verify the digest over its exact physical output.
pub fn executeVerified(
    alloc: Allocator,
    record: TensorRecord,
) ArchiveError!types.Stream {
    var decoded = try interpreter.executeTensor(alloc, record.tensor_program);
    errdefer decoded.deinit(alloc);
    if (!std.mem.eql(u8, &sha256(decoded.data), &record.checksum))
        return error.ChecksumMismatch;
    return decoded;
}

fn decodeRecordBodyVerified(
    alloc: Allocator,
    body: []const u8,
    limits: DecodeLimits,
) ArchiveError!VerifiedTensorRecord {
    var reader = Reader{ .bytes = body };
    if (try reader.readByte() != RECORD_TAG) return error.UnknownRecord;

    const name_len = try reader.readUsize();
    if (name_len > limits.max_name_bytes) return error.NameLimitExceeded;
    const name = try reader.take(name_len);

    const dtype_wire = std.enums.fromInt(DtypeWireId, try reader.readByte()) orelse
        return error.UnknownDtype;
    const dtype = wireToDtype(dtype_wire);

    const dimensions = try reader.readUsize();
    if (dimensions > limits.max_dimensions)
        return error.DimensionLimitExceeded;
    if (dimensions > body.len - reader.pos) return error.Truncated;

    const shape = try alloc.alloc(u64, dimensions);
    defer alloc.free(shape);
    for (shape) |*dimension| {
        dimension.* = try reader.readUleb128();
    }
    var elements: usize = 1;
    if (std.mem.indexOfScalar(u64, shape, 0) != null) {
        elements = 0;
    } else {
        for (shape) |dimension| {
            const dimension_usize = std.math.cast(usize, dimension) orelse
                return error.IntegerOverflow;
            elements = std.math.mul(usize, elements, dimension_usize) catch
                return error.IntegerOverflow;
        }
    }
    const output_bytes = std.math.mul(usize, elements, dtype.elemSize()) catch
        return error.IntegerOverflow;
    if (output_bytes > limits.max_tensor_output_bytes)
        return error.OutputLimitExceeded;

    const program_len = try reader.readUsize();
    if (program_len > limits.max_program_bytes)
        return error.ProgramLimitExceeded;
    const bytecode = try reader.take(program_len);
    const checksum_bytes = try reader.take(CHECKSUM_BYTES);
    if (reader.pos != body.len) return error.TrailingRecordData;

    var program_limits = limits.program;
    program_limits.max_output_bytes = @min(
        program_limits.max_output_bytes,
        output_bytes,
    );
    // A valid width-b tree can lower at worst to b one-byte bit planes, so
    // the total physical storage of all literal leaves is at most 8x the
    // final canonical storage (32 one-byte planes versus four-byte words).
    // This also bounds allocations made while parsing an ultimately ill-typed
    // tree, before TensorProgram can compare its root with the record dtype.
    const record_literal_bytes = std.math.mul(
        usize,
        output_bytes,
        8,
    ) catch std.math.maxInt(usize);
    program_limits.max_literal_bytes = @min(
        program_limits.max_literal_bytes,
        record_literal_bytes,
    );
    var root = try program_format.deserialize(alloc, bytecode, program_limits);
    var root_owned = true;
    defer if (root_owned) root.deinit(alloc);

    var tensor_program = try dsl.TensorProgram.init(alloc, dtype, shape, root);
    root_owned = false;
    var tensor_owned = true;
    defer if (tensor_owned) tensor_program.deinit(alloc);

    const owned_name = try alloc.dupe(u8, name);
    var name_owned = true;
    defer if (name_owned) alloc.free(owned_name);

    var checksum: [CHECKSUM_BYTES]u8 = undefined;
    @memcpy(&checksum, checksum_bytes);
    var record = TensorRecord{
        .name = owned_name,
        .tensor_program = tensor_program,
        .checksum = checksum,
    };
    tensor_owned = false;
    name_owned = false;
    errdefer record.deinit(alloc);

    var decoded = try executeVerified(alloc, record);
    errdefer decoded.deinit(alloc);
    if (decoded.data.len != output_bytes) return error.TensorLengthMismatch;
    return .{ .record = record, .decoded = decoded };
}

fn validateSafetensorsPrefix(prefix: []const u8) ArchiveError!void {
    if (prefix.len < 8) return error.InvalidSafetensorsPrefix;
    const declared_u64 = std.mem.readInt(u64, prefix[0..8], .little);
    const declared = std.math.cast(usize, declared_u64) orelse
        return error.InvalidSafetensorsPrefix;
    if (declared != prefix.len - 8) return error.InvalidSafetensorsPrefix;
}

fn sha256(bytes: []const u8) [CHECKSUM_BYTES]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(bytes);
    var digest: [CHECKSUM_BYTES]u8 = undefined;
    hash.final(&digest);
    return digest;
}

fn dtypeToWire(dtype: types.Dtype) DtypeWireId {
    return switch (dtype) {
        .f16 => .f16,
        .bf16 => .bf16,
        .f32 => .f32,
        .u8 => .u8,
        .u16 => .u16,
        .u32 => .u32,
        .i8 => .i8,
        .i16 => .i16,
        .i32 => .i32,
        .f8_e4m3 => .f8_e4m3,
        .f8_e5m2 => .f8_e5m2,
    };
}

fn wireToDtype(wire: DtypeWireId) types.Dtype {
    return switch (wire) {
        .f16 => .f16,
        .bf16 => .bf16,
        .f32 => .f32,
        .u8 => .u8,
        .u16 => .u16,
        .u32 => .u32,
        .i8 => .i8,
        .i16 => .i16,
        .i32 => .i32,
        .f8_e4m3 => .f8_e4m3,
        .f8_e5m2 => .f8_e5m2,
    };
}

fn usizeToU64(value: usize) ArchiveError!u64 {
    return std.math.cast(u64, value) orelse error.IntegerOverflow;
}

const Emitter = struct {
    allocator: Allocator,
    output: *std.ArrayList(u8),

    fn writeByte(self: *Emitter, byte: u8) Allocator.Error!void {
        try self.output.append(self.allocator, byte);
    }

    fn writeAll(self: *Emitter, bytes: []const u8) Allocator.Error!void {
        try self.output.appendSlice(self.allocator, bytes);
    }

    fn writeUleb128(self: *Emitter, value: u64) Allocator.Error!void {
        var remaining = value;
        while (true) {
            var byte: u8 = @truncate(remaining & 0x7f);
            remaining >>= 7;
            if (remaining != 0) byte |= 0x80;
            try self.writeByte(byte);
            if (remaining == 0) return;
        }
    }
};

const Reader = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn take(self: *Reader, count: usize) ArchiveError![]const u8 {
        if (self.pos > self.bytes.len or count > self.bytes.len - self.pos)
            return error.Truncated;
        defer self.pos += count;
        return self.bytes[self.pos..][0..count];
    }

    fn readByte(self: *Reader) ArchiveError!u8 {
        return (try self.take(1))[0];
    }

    fn readUsize(self: *Reader) ArchiveError!usize {
        return std.math.cast(usize, try self.readUleb128()) orelse
            error.IntegerOverflow;
    }

    fn readUleb128(self: *Reader) ArchiveError!u64 {
        var value: u64 = 0;
        for (0..10) |index| {
            const byte = try self.readByte();
            const payload: u64 = byte & 0x7f;
            if (index == 9 and payload > 1)
                return error.IntegerOverflow;
            const shift: u6 = @intCast(index * 7);
            value |= payload << shift;
            if (byte & 0x80 == 0) {
                if (uleb128Size(value) != index + 1)
                    return error.OverlongUleb128;
                return value;
            }
        }
        return error.IntegerOverflow;
    }
};

fn uleb128Size(value: u64) usize {
    var remaining = value;
    var size: usize = 1;
    while (remaining >= 0x80) : (size += 1)
        remaining >>= 7;
    return size;
}
