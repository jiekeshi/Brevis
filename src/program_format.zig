//! Canonical wire format for the paper's semantic program tree.
//!
//! Literal values remain semantic `Lit` nodes in the program tree. At the
//! physical boundary, each literal independently chooses the smallest complete
//! raw/bitpack/Huffman/rANS body. This lowering never changes DSL semantics.

const std = @import("std");
const dsl = @import("dsl.zig");
const literal_encoding = @import("literal_encoding.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;

/// These values are part of the persistent format. Never derive them from the
/// in-memory union tag order.
pub const NodeWireId = enum(u8) {
    literal = 0x01,
    constant = 0x02,
    concat = 0x03,
    repeat = 0x04,
    map = 0x05,
    scan = 0x06,
    merge = 0x07,
};

pub const MapWireId = enum(u8) {
    xor = 0x01,
    add_mod = 0x02,
    zigzag = 0x03,
    gray = 0x04,
    rotate_left = 0x05,
    bit_reverse = 0x06,
};

pub const ScanWireId = enum(u8) {
    xor = 0x01,
    add_mod = 0x02,
};

pub const MergeWireId = enum(u8) {
    fields = 0x01,
    float_fields = 0x02,
    bit_planes = 0x03,
    byte_planes = 0x04,
};

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
    max_nodes: usize = 1_000_000,
    /// Root depth is one.
    max_depth: usize = 256,
    /// Physical output bytes, using the DSL stream's canonical storage width.
    max_output_bytes: usize = types.defaultTensorByteLimit,
    /// Sum of all decoded literal storage bytes in the encoded tree.
    max_literal_bytes: usize = types.defaultTensorByteLimit,
    /// Sum of physical bytes touched by semantic node execution. This bounds
    /// compact, deeply nested programs that repeatedly transform a large
    /// output without increasing its final byte size.
    max_execution_work_bytes: usize = types.defaultExecutionByteLimit,
};

pub const DecodeError = Allocator.Error || error{
    BadMagic,
    UnsupportedVersion,
    Truncated,
    TrailingBytes,
    OverlongUleb128,
    IntegerOverflow,
    UnknownNode,
    UnknownMapOp,
    UnknownScanOp,
    UnknownMergeOp,
    UnknownDtype,
    InvalidLiteral,
    InvalidProgram,
    NodeLimitExceeded,
    DepthLimitExceeded,
    OutputLimitExceeded,
    LiteralLimitExceeded,
    ExecutionWorkLimitExceeded,
};

/// Serialize one validated semantic program into its canonical version-1
/// representation. The returned bytes are allocator-owned.
pub fn serialize(alloc: Allocator, program: dsl.Program) ![]u8 {
    _ = try program.typeOf();

    var output: std.ArrayList(u8) = .empty;
    errdefer output.deinit(alloc);

    var emitter = Emitter{
        .allocator = alloc,
        .output = &output,
    };
    try emitFile(&emitter, program);
    return output.toOwnedSlice(alloc);
}

/// A program's exact size together with the literal codec analyses that
/// produced it, so `emitPrepared` can write the bytes without repeating the
/// selection work.
pub const Prepared = struct {
    encodings: std.ArrayList(literal_encoding.PreparedEncoding),
    size: usize,

    pub fn deinit(self: *Prepared, alloc: Allocator) void {
        for (self.encodings.items) |*encoding| encoding.deinit(alloc);
        self.encodings.deinit(alloc);
        self.* = undefined;
    }
};

/// Size `program` while retaining its literal analyses. A null result means the
/// program provably reaches `limit`, exactly as `serializedSizeAtMost` reports.
pub fn prepareAtMost(
    alloc: Allocator,
    program: dsl.Program,
    limit: usize,
) !?Prepared {
    _ = try program.typeOf();

    var encodings: std.ArrayList(literal_encoding.PreparedEncoding) = .empty;
    errdefer {
        for (encodings.items) |*encoding| encoding.deinit(alloc);
        encodings.deinit(alloc);
    }
    var emitter = Emitter{
        .allocator = alloc,
        .limit = limit,
        .collect = &encodings,
    };
    emitFile(&emitter, program) catch |err| switch (err) {
        error.SizeLimitReached => {
            for (encodings.items) |*encoding| encoding.deinit(alloc);
            encodings.deinit(alloc);
            return null;
        },
        else => return err,
    };
    return .{ .encodings = encodings, .size = emitter.count };
}

/// Serialize using analyses captured by `prepareAtMost` for the same program.
pub fn emitPrepared(
    alloc: Allocator,
    program: dsl.Program,
    prepared: Prepared,
) ![]u8 {
    var output: std.ArrayList(u8) = .empty;
    errdefer output.deinit(alloc);
    try output.ensureTotalCapacity(alloc, prepared.size);

    var emitter = Emitter{
        .allocator = alloc,
        .output = &output,
        .cached = prepared.encodings.items,
    };
    try emitFile(&emitter, program);
    if (emitter.count != prepared.size) return error.LiteralSizeMismatch;
    return output.toOwnedSlice(alloc);
}

/// Exact canonical byte length. This uses the same emitter and literal codec
/// selection as `serialize`; there is no parallel cost model to drift from
/// the persisted bytes.
pub fn serializedSize(alloc: Allocator, program: dsl.Program) !usize {
    _ = try program.typeOf();

    var emitter = Emitter{ .allocator = alloc };
    try emitFile(&emitter, program);
    return emitter.count;
}

/// Exact canonical byte length when it is below `limit`, otherwise null.
/// Emission stops as soon as the running length or an admissible literal
/// lower bound reaches `limit`, so candidates that cannot beat the incumbent
/// never pay for a full entropy-coded measurement.
pub fn serializedSizeAtMost(
    alloc: Allocator,
    program: dsl.Program,
    limit: usize,
) !?usize {
    _ = try program.typeOf();

    var emitter = Emitter{
        .allocator = alloc,
        .limit = limit,
    };
    emitFile(&emitter, program) catch |err| switch (err) {
        error.SizeLimitReached => return null,
        else => return err,
    };
    return emitter.count;
}

/// Decode exactly one version-1 program. The whole tree is parsed first, then
/// its semantic type is checked through `Program.typeOf`.
pub fn deserialize(
    alloc: Allocator,
    bytes: []const u8,
    limits: DecodeLimits,
) DecodeError!dsl.Program {
    var reader = Reader{ .bytes = bytes };

    var state = DecodeState{ .limits = limits };
    var program = try readNode(alloc, &reader, &state, 1);
    errdefer program.deinit(alloc);

    if (reader.pos != bytes.len) return error.TrailingBytes;

    const output_type = program.typeOf() catch |validation_error| switch (validation_error) {
        error.LengthOverflow => return error.OutputLimitExceeded,
        else => return error.InvalidProgram,
    };
    const output_bytes = std.math.mul(
        usize,
        output_type.len,
        storageBytes(output_type.bits),
    ) catch return error.OutputLimitExceeded;
    if (output_bytes > limits.max_output_bytes)
        return error.OutputLimitExceeded;
    const execution_bytes = program.executionWorkBytes() catch
        return error.ExecutionWorkLimitExceeded;
    if (execution_bytes > limits.max_execution_work_bytes)
        return error.ExecutionWorkLimitExceeded;

    return program;
}

const Emitter = struct {
    allocator: Allocator,
    output: ?*std.ArrayList(u8) = null,
    count: usize = 0,
    limit: usize = literal_encoding.NO_LIMIT,
    /// Receives each literal's codec analysis so a later emission can reuse it.
    collect: ?*std.ArrayList(literal_encoding.PreparedEncoding) = null,
    /// Analyses captured by an earlier sizing pass, in emission order.
    cached: []const literal_encoding.PreparedEncoding = &.{},
    cursor: usize = 0,

    fn advance(self: *Emitter, next: usize) !void {
        if (next >= self.limit) return error.SizeLimitReached;
        self.count = next;
    }

    /// Bytes a literal body may occupy before the enclosing program is known
    /// to reach `limit`. One byte is reserved for the body-length prefix that
    /// always follows.
    fn literalBudget(self: Emitter) usize {
        if (self.limit == literal_encoding.NO_LIMIT)
            return literal_encoding.NO_LIMIT;
        if (self.limit <= self.count + 1) return 0;
        return self.limit - self.count - 1;
    }

    fn writeByte(self: *Emitter, byte: u8) !void {
        const next = std.math.add(usize, self.count, 1) catch
            return error.LengthOverflow;
        try self.advance(next);
        if (self.output) |output|
            try output.append(self.allocator, byte);
    }

    fn writeAll(self: *Emitter, bytes: []const u8) !void {
        const next = std.math.add(usize, self.count, bytes.len) catch
            return error.LengthOverflow;
        try self.advance(next);
        if (self.output) |output|
            try output.appendSlice(self.allocator, bytes);
    }

    fn writeLiteral(
        self: *Emitter,
        encoding: literal_encoding.PreparedEncoding,
    ) !void {
        const next = std.math.add(
            usize,
            self.count,
            encoding.wireSize(),
        ) catch return error.LengthOverflow;
        try self.advance(next);
        if (self.output) |output|
            try encoding.emitBody(self.allocator, output);
    }

    fn writeUleb128(self: *Emitter, value: u64) !void {
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

fn emitFile(emitter: *Emitter, program: dsl.Program) !void {
    try emitNode(emitter, program);
}

fn emitLiteral(emitter: *Emitter, literal: types.Stream) !void {
    try emitter.writeByte(@intFromEnum(NodeWireId.literal));
    try emitter.writeByte(literal.bits_per_elem);
    try emitter.writeUleb128(try usizeToU64(literal.count));

    // Selecting a literal's codec means histogramming the stream and sizing
    // every applicable body. Sizing a program and then emitting it would pay
    // that twice for the same literal, so a sizing pass can hand its analyses
    // to the emission pass.
    if (emitter.cursor < emitter.cached.len) {
        const encoding = emitter.cached[emitter.cursor];
        emitter.cursor += 1;
        try emitter.writeUleb128(try usizeToU64(encoding.wireSize()));
        try emitter.writeLiteral(encoding);
        return;
    }

    var encoding = (try literal_encoding.prepareBestWithin(
        emitter.allocator,
        literal,
        emitter.literalBudget(),
    )) orelse return error.SizeLimitReached;
    var owned = true;
    defer if (owned) encoding.deinit(emitter.allocator);
    if (emitter.collect) |list| {
        try list.append(emitter.allocator, encoding);
        owned = false;
    }
    try emitter.writeUleb128(try usizeToU64(encoding.wireSize()));
    try emitter.writeLiteral(encoding);
}

fn emitNode(emitter: *Emitter, program: dsl.Program) !void {
    switch (program.kind) {
        .literal => |literal| try emitLiteral(emitter, literal),
        .constant => |constant_value| {
            try emitter.writeByte(@intFromEnum(NodeWireId.constant));
            try emitter.writeByte(constant_value.bits);
            try emitter.writeUleb128(try usizeToU64(constant_value.len));
            try emitter.writeUleb128(constant_value.word);
        },
        .concat => {
            try emitter.writeByte(@intFromEnum(NodeWireId.concat));
            try emitter.writeUleb128(try usizeToU64(program.children.len));
            for (program.children) |child|
                try emitNode(emitter, child);
        },
        .repeat => |times| {
            try emitter.writeByte(@intFromEnum(NodeWireId.repeat));
            try emitter.writeUleb128(times);
            try emitNode(emitter, program.children[0]);
        },
        .map => |operation| {
            try emitter.writeByte(@intFromEnum(NodeWireId.map));
            switch (operation) {
                .xor => |constant| {
                    try emitter.writeByte(@intFromEnum(MapWireId.xor));
                    try emitter.writeUleb128(constant);
                },
                .add_mod => |constant| {
                    try emitter.writeByte(@intFromEnum(MapWireId.add_mod));
                    try emitter.writeUleb128(constant);
                },
                .zigzag => try emitter.writeByte(@intFromEnum(MapWireId.zigzag)),
                .gray => try emitter.writeByte(@intFromEnum(MapWireId.gray)),
                .rotate_left => |amount| {
                    try emitter.writeByte(@intFromEnum(MapWireId.rotate_left));
                    try emitter.writeUleb128(amount);
                },
                .bit_reverse => try emitter.writeByte(@intFromEnum(MapWireId.bit_reverse)),
            }
            try emitNode(emitter, program.children[0]);
        },
        .scan => |scan_value| {
            try emitter.writeByte(@intFromEnum(NodeWireId.scan));
            try emitter.writeByte(switch (scan_value.operation) {
                .xor => @intFromEnum(ScanWireId.xor),
                .add_mod => @intFromEnum(ScanWireId.add_mod),
            });
            try emitter.writeUleb128(scan_value.initial);
            try emitNode(emitter, program.children[0]);
        },
        .merge => |operation| {
            try emitter.writeByte(@intFromEnum(NodeWireId.merge));
            switch (operation) {
                .fields => |low_bits| {
                    try emitter.writeByte(@intFromEnum(MergeWireId.fields));
                    try emitter.writeUleb128(low_bits);
                },
                .float_fields => |dtype| {
                    try emitter.writeByte(@intFromEnum(MergeWireId.float_fields));
                    try emitter.writeByte(@intFromEnum(dtypeToWire(dtype)));
                },
                .bit_planes => try emitter.writeByte(@intFromEnum(MergeWireId.bit_planes)),
                .byte_planes => try emitter.writeByte(@intFromEnum(MergeWireId.byte_planes)),
            }
            try emitter.writeUleb128(try usizeToU64(program.children.len));
            for (program.children) |child|
                try emitNode(emitter, child);
        },
    }
}

fn usizeToU64(value: usize) dsl.ValidationError!u64 {
    return std.math.cast(u64, value) orelse error.LengthOverflow;
}

fn storageBytes(bits: u8) usize {
    return types.roundUpToPow2(bits) / 8;
}

const Reader = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn take(self: *Reader, count: usize) DecodeError![]const u8 {
        if (count > self.bytes.len - self.pos) return error.Truncated;
        defer self.pos += count;
        return self.bytes[self.pos..][0..count];
    }

    fn readByte(self: *Reader) DecodeError!u8 {
        return (try self.take(1))[0];
    }

    fn readUleb128(self: *Reader) DecodeError!u64 {
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

const DecodeState = struct {
    limits: DecodeLimits,
    nodes: usize = 0,
    literal_bytes: usize = 0,

    fn claimNode(self: *DecodeState, depth: usize) DecodeError!void {
        if (depth > self.limits.max_depth)
            return error.DepthLimitExceeded;
        if (self.nodes >= self.limits.max_nodes)
            return error.NodeLimitExceeded;
        self.nodes += 1;
    }

    fn claimLiteralBytes(self: *DecodeState, count: usize) DecodeError!void {
        if (count > self.limits.max_literal_bytes - self.literal_bytes)
            return error.LiteralLimitExceeded;
        self.literal_bytes += count;
    }
};

fn readNode(
    alloc: Allocator,
    reader: *Reader,
    state: *DecodeState,
    depth: usize,
) DecodeError!dsl.Program {
    try state.claimNode(depth);
    const node_id = std.enums.fromInt(NodeWireId, try reader.readByte()) orelse
        return error.UnknownNode;

    return switch (node_id) {
        .literal => readLiteral(alloc, reader, state),
        .constant => readConstant(reader),
        .concat => blk: {
            const child_count = try readUsize(reader);
            const children = try readChildren(
                alloc,
                reader,
                state,
                depth,
                child_count,
            );
            break :blk .{ .kind = .concat, .children = children };
        },
        .repeat => blk: {
            const times = try readU32(reader);
            const children = try readChildren(alloc, reader, state, depth, 1);
            break :blk .{
                .kind = .{ .repeat = times },
                .children = children,
            };
        },
        .map => blk: {
            const operation = try readMapOp(reader);
            const children = try readChildren(alloc, reader, state, depth, 1);
            break :blk .{
                .kind = .{ .map = operation },
                .children = children,
            };
        },
        .scan => blk: {
            const operation = try readScanOp(reader);
            const initial = try readU32(reader);
            const children = try readChildren(alloc, reader, state, depth, 1);
            break :blk .{
                .kind = .{ .scan = .{
                    .operation = operation,
                    .initial = initial,
                } },
                .children = children,
            };
        },
        .merge => blk: {
            const operation = try readMergeOp(reader);
            const child_count = try readUsize(reader);
            const children = try readChildren(
                alloc,
                reader,
                state,
                depth,
                child_count,
            );
            break :blk .{
                .kind = .{ .merge = operation },
                .children = children,
            };
        },
    };
}

fn readLiteral(
    alloc: Allocator,
    reader: *Reader,
    state: *DecodeState,
) DecodeError!dsl.Program {
    const bits = try reader.readByte();
    if (bits == 0 or bits > 32) return error.InvalidProgram;
    const count = try readUsize(reader);
    const byte_count = std.math.mul(usize, count, storageBytes(bits)) catch
        return error.OutputLimitExceeded;
    // Literal codecs such as single-symbol rANS can describe arbitrarily
    // large semantic streams with a constant-size body. Enforce the semantic
    // output bound before reading or decoding that body, and therefore before
    // any allocation proportional to its declared count.
    if (byte_count > state.limits.max_output_bytes)
        return error.OutputLimitExceeded;
    try state.claimLiteralBytes(byte_count);
    const body_len = try readUsize(reader);
    const body = try reader.take(body_len);

    var stream = literal_encoding.decodeBody(
        alloc,
        bits,
        count,
        body,
    ) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidLiteral,
    };
    errdefer stream.deinit(alloc);

    return .{ .kind = .{ .literal = stream } };
}

fn readConstant(reader: *Reader) DecodeError!dsl.Program {
    return .{ .kind = .{ .constant = .{
        .bits = try reader.readByte(),
        .len = try readUsize(reader),
        .word = try readU32(reader),
    } } };
}

fn readChildren(
    alloc: Allocator,
    reader: *Reader,
    state: *DecodeState,
    parent_depth: usize,
    count: usize,
) DecodeError![]dsl.Program {
    if (count > state.limits.max_nodes - state.nodes)
        return error.NodeLimitExceeded;
    // Every child needs at least its one-byte node id. Reject impossible
    // counts before allocating an attacker-controlled slice.
    if (count > reader.bytes.len - reader.pos)
        return error.Truncated;

    const children = try alloc.alloc(dsl.Program, count);
    var initialized: usize = 0;
    errdefer {
        for (children[0..initialized]) |*child|
            child.deinit(alloc);
        alloc.free(children);
    }
    for (children) |*child| {
        child.* = try readNode(alloc, reader, state, parent_depth + 1);
        initialized += 1;
    }
    return children;
}

fn readMapOp(reader: *Reader) DecodeError!dsl.MapOp {
    const wire_id = std.enums.fromInt(MapWireId, try reader.readByte()) orelse
        return error.UnknownMapOp;
    return switch (wire_id) {
        .xor => .{ .xor = try readU32(reader) },
        .add_mod => .{ .add_mod = try readU32(reader) },
        .zigzag => .zigzag,
        .gray => .gray,
        .rotate_left => .{ .rotate_left = try readU8Uleb(reader) },
        .bit_reverse => .bit_reverse,
    };
}

fn readScanOp(reader: *Reader) DecodeError!dsl.ScanOp {
    const wire_id = std.enums.fromInt(ScanWireId, try reader.readByte()) orelse
        return error.UnknownScanOp;
    return switch (wire_id) {
        .xor => .xor,
        .add_mod => .add_mod,
    };
}

fn readMergeOp(reader: *Reader) DecodeError!dsl.MergeOp {
    const wire_id = std.enums.fromInt(MergeWireId, try reader.readByte()) orelse
        return error.UnknownMergeOp;
    return switch (wire_id) {
        .fields => .{ .fields = try readU8Uleb(reader) },
        .float_fields => .{ .float_fields = try readDtype(reader) },
        .bit_planes => .bit_planes,
        .byte_planes => .byte_planes,
    };
}

fn readDtype(reader: *Reader) DecodeError!types.Dtype {
    const wire_id = std.enums.fromInt(DtypeWireId, try reader.readByte()) orelse
        return error.UnknownDtype;
    return switch (wire_id) {
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

fn readUsize(reader: *Reader) DecodeError!usize {
    return std.math.cast(usize, try reader.readUleb128()) orelse
        error.IntegerOverflow;
}

fn readU32(reader: *Reader) DecodeError!u32 {
    return std.math.cast(u32, try reader.readUleb128()) orelse
        error.IntegerOverflow;
}

fn readU8Uleb(reader: *Reader) DecodeError!u8 {
    return std.math.cast(u8, try reader.readUleb128()) orelse
        error.IntegerOverflow;
}
