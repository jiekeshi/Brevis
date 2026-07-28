//! Semantic tensor-generating programs from the paper.
//!
//! This tree contains only generation semantics. Literal compression, wire
//! framing, search state, and archive ownership live in separate modules.

const std = @import("std");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const Stream = types.Stream;

pub const StreamType = struct {
    bits: u8,
    len: usize,
};

pub const MapOp = union(enum) {
    xor: u32,
    add_mod: u32,
    zigzag,
    gray,
    rotate_left: u8,
    bit_reverse,

    pub fn validate(self: MapOp, bits: u8) ValidationError!void {
        if (bits == 0 or bits > 32) return error.InvalidWordWidth;
        const mask = if (bits == 32)
            std.math.maxInt(u32)
        else
            (@as(u32, 1) << @intCast(bits)) - 1;
        switch (self) {
            .xor, .add_mod => |parameter| {
                if (parameter & ~mask != 0) return error.InvalidParameter;
            },
            .rotate_left => |amount| {
                if (amount >= bits) return error.InvalidParameter;
            },
            .zigzag, .gray, .bit_reverse => {},
        }
    }
};

pub const ScanOp = enum {
    xor,
    add_mod,
};

pub const MergeOp = union(enum) {
    /// Pack children from least-significant to most-significant field.
    fields: u8,
    /// Children are sign, exponent, and mantissa, in that order.
    float_fields: types.Dtype,
    /// Children are one-bit planes from least to most significant.
    bit_planes,
    /// Children are byte-sized fields from least to most significant; the
    /// final field may be narrower than eight bits.
    byte_planes,
};

pub const ValidationError = error{
    InvalidWordWidth,
    InvalidLength,
    InvalidLiteralValue,
    InvalidArity,
    TypeMismatch,
    InvalidParameter,
    InvalidRepeatCount,
    LengthOverflow,
};

pub const Kind = union(enum) {
    literal: Stream,
    constant: struct {
        bits: u8,
        len: usize,
        word: u32,
    },
    concat,
    repeat: u32,
    map: MapOp,
    scan: struct {
        operation: ScanOp,
        initial: u32,
    },
    merge: MergeOp,
};

pub const Program = struct {
    kind: Kind,
    children: []Program = &.{},

    /// Construct a semantic `Lit`. Physical words are copied into program
    /// ownership so the program is self-contained.
    pub fn literal(alloc: Allocator, bits: u8, words: []const u32) (Allocator.Error || ValidationError)!Program {
        if (bits == 0 or bits > 32) return error.InvalidWordWidth;

        var stream = try Stream.initUninitialized(alloc, words.len, bits);
        errdefer stream.deinit(alloc);
        const mask = stream.mask();
        for (words, 0..) |word, i| {
            if (word & ~mask != 0) return error.InvalidLiteralValue;
            stream.setU32(i, word);
        }
        return .{ .kind = .{ .literal = stream } };
    }

    /// Copy an already materialized target stream into a semantic `Lit`.
    pub fn literalFromStream(alloc: Allocator, source: Stream) (Allocator.Error || ValidationError)!Program {
        if (source.bits_per_elem == 0 or source.bits_per_elem > 32)
            return error.InvalidWordWidth;
        const required = std.math.mul(usize, source.count, source.elemBytes()) catch
            return error.LengthOverflow;
        if (source.data.len != required) return error.InvalidLiteralValue;
        if (source.bits_per_elem != types.roundUpToPow2(source.bits_per_elem)) {
            const mask = source.mask();
            for (0..source.count) |index|
                if (source.getU32(index) & ~mask != 0)
                    return error.InvalidLiteralValue;
        }
        return .{ .kind = .{ .literal = try source.dupe(alloc) } };
    }

    pub fn constant(bits: u8, len: usize, word: u32) ValidationError!Program {
        if (bits == 0 or bits > 32) return error.InvalidWordWidth;
        if (len == 0) return error.InvalidLength;
        const mask = if (bits == 32)
            std.math.maxInt(u32)
        else
            (@as(u32, 1) << @intCast(bits)) - 1;
        if (word & ~mask != 0) return error.InvalidLiteralValue;
        return .{ .kind = .{ .constant = .{
            .bits = bits,
            .len = len,
            .word = word,
        } }, .children = &.{} };
    }

    /// Construct `Concat(children...)` by cloning its children. This keeps
    /// ownership local to each program tree and makes caller cleanup uniform.
    pub fn concat(alloc: Allocator, children: []const Program) (Allocator.Error || ValidationError)!Program {
        if (children.len < 2) return error.InvalidArity;
        const first_type = try children[0].typeOf();
        if (first_type.len == 0) return error.InvalidLength;
        var total = first_type.len;
        for (children[1..]) |child| {
            const child_type = try child.typeOf();
            if (child_type.bits != first_type.bits) return error.TypeMismatch;
            if (child_type.len == 0) return error.InvalidLength;
            total = std.math.add(usize, total, child_type.len) catch
                return error.LengthOverflow;
        }
        const owned = try alloc.alloc(Program, children.len);
        var initialized: usize = 0;
        errdefer {
            for (owned[0..initialized]) |*child| child.deinit(alloc);
            alloc.free(owned);
        }
        for (children, 0..) |child, i| {
            owned[i] = try child.clone(alloc);
            initialized += 1;
        }
        return concatOwned(owned) catch |err| {
            for (owned) |*child| child.deinit(alloc);
            alloc.free(owned);
            return err;
        };
    }

    /// Adopt an allocator-owned child slice as a `Concat`. Ownership remains
    /// with the caller on validation failure and moves to the result on
    /// success.
    pub fn concatOwned(children: []Program) ValidationError!Program {
        if (children.len < 2) return error.InvalidArity;
        const first_type = try children[0].typeOf();
        if (first_type.len == 0) return error.InvalidLength;
        var total = first_type.len;
        for (children[1..]) |child| {
            const child_type = try child.typeOf();
            if (child_type.bits != first_type.bits) return error.TypeMismatch;
            if (child_type.len == 0) return error.InvalidLength;
            total = std.math.add(usize, total, child_type.len) catch
                return error.LengthOverflow;
        }
        return .{ .kind = .concat, .children = children };
    }

    /// Construct `Repeat[times](child)`. The child is moved into the returned
    /// program on success; callers must not deinitialize it afterward.
    pub fn repeat(alloc: Allocator, times: u32, child: Program) (Allocator.Error || ValidationError)!Program {
        if (times < 2) return error.InvalidRepeatCount;
        const child_type = try child.typeOf();
        if (child_type.len == 0) return error.InvalidLength;
        _ = std.math.mul(usize, child_type.len, times) catch return error.LengthOverflow;

        const children = try alloc.alloc(Program, 1);
        children[0] = child;
        return .{
            .kind = .{ .repeat = times },
            .children = children,
        };
    }

    /// Construct a width-preserving pointwise map. The child is moved into the
    /// returned program on success.
    pub fn map(alloc: Allocator, operation: MapOp, child: Program) (Allocator.Error || ValidationError)!Program {
        const child_type = try child.typeOf();
        if (child_type.len == 0) return error.InvalidLength;
        try operation.validate(child_type.bits);
        const children = try alloc.alloc(Program, 1);
        children[0] = child;
        return .{
            .kind = .{ .map = operation },
            .children = children,
        };
    }

    /// Construct the paper's inclusive scan. `initial` is the first emitted
    /// word; the child is moved and generates the remaining `n - 1` updates.
    pub fn scan(
        alloc: Allocator,
        operation: ScanOp,
        initial: u32,
        child: Program,
    ) (Allocator.Error || ValidationError)!Program {
        const child_type = try child.typeOf();
        if (child_type.len == 0) return error.InvalidLength;
        const mask = if (child_type.bits == 32)
            std.math.maxInt(u32)
        else
            (@as(u32, 1) << @intCast(child_type.bits)) - 1;
        if (initial & ~mask != 0) return error.InvalidParameter;
        _ = std.math.add(usize, child_type.len, 1) catch return error.LengthOverflow;

        const children = try alloc.alloc(Program, 1);
        children[0] = child;
        return .{
            .kind = .{ .scan = .{
                .operation = operation,
                .initial = initial,
            } },
            .children = children,
        };
    }

    /// Construct a pointwise merge by cloning the supplied child programs.
    pub fn merge(
        alloc: Allocator,
        operation: MergeOp,
        children: []const Program,
    ) (Allocator.Error || ValidationError)!Program {
        _ = try mergeOutputType(operation, children);
        const owned = try alloc.alloc(Program, children.len);
        var initialized: usize = 0;
        errdefer {
            for (owned[0..initialized]) |*child| child.deinit(alloc);
            alloc.free(owned);
        }
        for (children, 0..) |child, i| {
            owned[i] = try child.clone(alloc);
            initialized += 1;
        }
        return mergeOwned(operation, owned) catch |err| {
            for (owned) |*child| child.deinit(alloc);
            alloc.free(owned);
            return err;
        };
    }

    /// Adopt an allocator-owned child slice as a `Merge`.
    pub fn mergeOwned(operation: MergeOp, children: []Program) ValidationError!Program {
        _ = try mergeOutputType(operation, children);
        return .{
            .kind = .{ .merge = operation },
            .children = children,
        };
    }

    pub fn clone(self: Program, alloc: Allocator) Allocator.Error!Program {
        const kind: Kind = switch (self.kind) {
            .literal => |stream| .{ .literal = try stream.dupe(alloc) },
            .constant => |constant_value| .{ .constant = constant_value },
            .concat => .concat,
            .repeat => |times| .{ .repeat = times },
            .map => |operation| .{ .map = operation },
            .scan => |scan_value| .{ .scan = scan_value },
            .merge => |operation| .{ .merge = operation },
        };
        errdefer switch (kind) {
            .literal => |stream| alloc.free(stream.data),
            else => {},
        };

        var children: []Program = &.{};
        if (self.children.len > 0) children = try alloc.alloc(Program, self.children.len);
        var initialized: usize = 0;
        errdefer {
            for (children[0..initialized]) |*child| child.deinit(alloc);
            if (children.len > 0) alloc.free(children);
        }
        for (self.children, 0..) |child, i| {
            children[i] = try child.clone(alloc);
            initialized += 1;
        }
        return .{ .kind = kind, .children = children };
    }

    pub fn deinit(self: *Program, alloc: Allocator) void {
        for (self.children) |*child| child.deinit(alloc);
        if (self.children.len > 0) alloc.free(self.children);
        self.children = &.{};

        switch (self.kind) {
            .literal => |*stream| stream.deinit(alloc),
            .constant, .concat, .repeat, .map, .scan, .merge => {},
        }
    }

    pub fn typeOf(self: Program) ValidationError!StreamType {
        return (try analyzeProgram(self)).stream_type;
    }

    /// Conservative physical-byte work performed by the exact interpreter.
    /// This is used to reject compact programs whose expansion size is legal
    /// but whose nested transforms would require unreasonable repeated passes.
    pub fn executionWorkBytes(self: Program) ValidationError!usize {
        return (try analyzeProgram(self)).execution_bytes;
    }

    pub fn countNodes(self: Program) usize {
        var count: usize = 1;
        for (self.children) |child|
            count = std.math.add(usize, count, child.countNodes()) catch
                return std.math.maxInt(usize);
        return count;
    }

    /// Tree height in edges; a leaf has depth zero.
    pub fn depth(self: Program) usize {
        var child_depth: usize = 0;
        for (self.children) |child|
            child_depth = @max(child_depth, child.depth() + 1);
        return child_depth;
    }
};

const ProgramAnalysis = struct {
    stream_type: StreamType,
    execution_bytes: usize,
};

fn analyzeProgram(program: Program) ValidationError!ProgramAnalysis {
    return switch (program.kind) {
        .literal => |stream| blk: {
            if (program.children.len != 0) return error.InvalidArity;
            if (stream.bits_per_elem == 0 or stream.bits_per_elem > 32)
                return error.InvalidWordWidth;
            const required = std.math.mul(
                usize,
                stream.count,
                stream.elemBytes(),
            ) catch return error.LengthOverflow;
            if (stream.data.len != required) return error.InvalidLiteralValue;
            if (!stream.valuesFitWidth()) return error.InvalidLiteralValue;
            break :blk .{
                .stream_type = .{
                    .bits = stream.bits_per_elem,
                    .len = stream.count,
                },
                .execution_bytes = required,
            };
        },
        .constant => |constant_value| blk: {
            if (program.children.len != 0) return error.InvalidArity;
            if (constant_value.bits == 0 or constant_value.bits > 32)
                return error.InvalidWordWidth;
            if (constant_value.len == 0) return error.InvalidLength;
            const mask = if (constant_value.bits == 32)
                std.math.maxInt(u32)
            else
                (@as(u32, 1) << @intCast(constant_value.bits)) - 1;
            if (constant_value.word & ~mask != 0)
                return error.InvalidLiteralValue;
            break :blk .{
                .stream_type = .{
                    .bits = constant_value.bits,
                    .len = constant_value.len,
                },
                .execution_bytes = try streamBytes(.{
                    .bits = constant_value.bits,
                    .len = constant_value.len,
                }),
            };
        },
        .concat => blk: {
            if (program.children.len < 2) return error.InvalidArity;
            const first = try analyzeProgram(program.children[0]);
            if (first.stream_type.len == 0) return error.InvalidLength;
            var len = first.stream_type.len;
            var execution_bytes = first.execution_bytes;
            for (program.children[1..]) |child_program| {
                const child = try analyzeProgram(child_program);
                if (child.stream_type.bits != first.stream_type.bits)
                    return error.TypeMismatch;
                if (child.stream_type.len == 0) return error.InvalidLength;
                len = std.math.add(
                    usize,
                    len,
                    child.stream_type.len,
                ) catch return error.LengthOverflow;
                execution_bytes = std.math.add(
                    usize,
                    execution_bytes,
                    child.execution_bytes,
                ) catch return error.LengthOverflow;
            }
            break :blk .{
                .stream_type = .{ .bits = first.stream_type.bits, .len = len },
                .execution_bytes = execution_bytes,
            };
        },
        .repeat => |times| blk: {
            if (times < 2) return error.InvalidRepeatCount;
            if (program.children.len != 1) return error.InvalidArity;
            const child = try analyzeProgram(program.children[0]);
            if (child.stream_type.len == 0) return error.InvalidLength;
            const len = std.math.mul(
                usize,
                child.stream_type.len,
                times,
            ) catch return error.LengthOverflow;
            const copied_elements = len - child.stream_type.len;
            const copied_bytes = std.math.mul(
                usize,
                copied_elements,
                storageBytes(child.stream_type.bits),
            ) catch return error.LengthOverflow;
            break :blk .{
                .stream_type = .{ .bits = child.stream_type.bits, .len = len },
                .execution_bytes = std.math.add(
                    usize,
                    child.execution_bytes,
                    copied_bytes,
                ) catch return error.LengthOverflow,
            };
        },
        .map => |operation| blk: {
            if (program.children.len != 1) return error.InvalidArity;
            const child = try analyzeProgram(program.children[0]);
            if (child.stream_type.len == 0) return error.InvalidLength;
            try operation.validate(child.stream_type.bits);
            break :blk .{
                .stream_type = child.stream_type,
                .execution_bytes = std.math.add(
                    usize,
                    child.execution_bytes,
                    try streamBytes(child.stream_type),
                ) catch return error.LengthOverflow,
            };
        },
        .scan => |scan_value| blk: {
            if (program.children.len != 1) return error.InvalidArity;
            const child = try analyzeProgram(program.children[0]);
            if (child.stream_type.len == 0) return error.InvalidLength;
            const mask = if (child.stream_type.bits == 32)
                std.math.maxInt(u32)
            else
                (@as(u32, 1) << @intCast(child.stream_type.bits)) - 1;
            if (scan_value.initial & ~mask != 0)
                return error.InvalidParameter;
            const output_type = StreamType{
                .bits = child.stream_type.bits,
                .len = std.math.add(
                    usize,
                    child.stream_type.len,
                    1,
                ) catch return error.LengthOverflow,
            };
            break :blk .{
                .stream_type = output_type,
                .execution_bytes = std.math.add(
                    usize,
                    child.execution_bytes,
                    try streamBytes(output_type),
                ) catch return error.LengthOverflow,
            };
        },
        .merge => |operation| blk: {
            if (program.children.len < 2 or program.children.len > 32)
                return error.InvalidArity;
            var child_types: [32]StreamType = undefined;
            var execution_bytes: usize = 0;
            for (program.children, 0..) |child_program, index| {
                const child = try analyzeProgram(child_program);
                child_types[index] = child.stream_type;
                execution_bytes = std.math.add(
                    usize,
                    execution_bytes,
                    child.execution_bytes,
                ) catch return error.LengthOverflow;
            }
            const output_type = try mergeOutputTypeFromTypes(
                operation,
                child_types[0..program.children.len],
            );
            const output_passes = std.math.add(
                usize,
                program.children.len,
                1,
            ) catch return error.LengthOverflow;
            const merge_bytes = std.math.mul(
                usize,
                try streamBytes(output_type),
                output_passes,
            ) catch return error.LengthOverflow;
            break :blk .{
                .stream_type = output_type,
                .execution_bytes = std.math.add(
                    usize,
                    execution_bytes,
                    merge_bytes,
                ) catch return error.LengthOverflow,
            };
        },
    };
}

fn storageBytes(bits: u8) usize {
    return types.roundUpToPow2(bits) / 8;
}

fn streamBytes(stream_type: StreamType) ValidationError!usize {
    return std.math.mul(
        usize,
        stream_type.len,
        storageBytes(stream_type.bits),
    ) catch error.LengthOverflow;
}

pub const TensorType = struct {
    dtype: types.Dtype,
    elements: usize,
};

pub const TensorValidationError = ValidationError || error{
    ShapeOverflow,
    TensorWidthMismatch,
    TensorLengthMismatch,
};

/// One self-contained generator bound to the dtype and shape of exactly one
/// tensor. The program and copied shape are owned by this value.
pub const TensorProgram = struct {
    dtype: types.Dtype,
    shape: []u64,
    root: Program,

    /// `root` is moved into the result on success and remains caller-owned on
    /// error.
    pub fn init(
        alloc: Allocator,
        dtype: types.Dtype,
        shape: []const u64,
        root: Program,
    ) (Allocator.Error || TensorValidationError)!TensorProgram {
        _ = try validateParts(dtype, shape, root);
        return .{
            .dtype = dtype,
            .shape = try alloc.dupe(u64, shape),
            .root = root,
        };
    }

    pub fn deinit(self: *TensorProgram, alloc: Allocator) void {
        self.root.deinit(alloc);
        alloc.free(self.shape);
        self.shape = &.{};
    }

    pub fn validate(self: TensorProgram) TensorValidationError!TensorType {
        return validateParts(self.dtype, self.shape, self.root);
    }

    pub fn clone(self: TensorProgram, alloc: Allocator) Allocator.Error!TensorProgram {
        const shape = try alloc.dupe(u64, self.shape);
        errdefer alloc.free(shape);
        return .{
            .dtype = self.dtype,
            .shape = shape,
            .root = try self.root.clone(alloc),
        };
    }
};

fn validateParts(
    dtype: types.Dtype,
    shape: []const u64,
    root: Program,
) TensorValidationError!TensorType {
    for (shape) |dimension| {
        if (dimension == 0) {
            const output_type = try root.typeOf();
            if (output_type.bits != dtype.bitWidth())
                return error.TensorWidthMismatch;
            if (output_type.len != 0)
                return error.TensorLengthMismatch;
            return .{ .dtype = dtype, .elements = 0 };
        }
    }
    var elements: usize = 1;
    for (shape) |dimension| {
        const dimension_usize = std.math.cast(usize, dimension) orelse
            return error.ShapeOverflow;
        elements = std.math.mul(usize, elements, dimension_usize) catch
            return error.ShapeOverflow;
    }
    const output_type = try root.typeOf();
    if (output_type.bits != dtype.bitWidth()) return error.TensorWidthMismatch;
    if (output_type.len != elements) return error.TensorLengthMismatch;
    return .{ .dtype = dtype, .elements = elements };
}

fn mergeOutputType(operation: MergeOp, children: []const Program) ValidationError!StreamType {
    if (children.len < 2) return error.InvalidArity;

    var child_types: [32]StreamType = undefined;
    if (children.len > child_types.len) return error.InvalidArity;
    for (children, 0..) |child, i| child_types[i] = try child.typeOf();
    return mergeOutputTypeFromTypes(operation, child_types[0..children.len]);
}

fn mergeOutputTypeFromTypes(
    operation: MergeOp,
    child_types: []const StreamType,
) ValidationError!StreamType {
    if (child_types.len < 2 or child_types.len > 32)
        return error.InvalidArity;
    const len = child_types[0].len;
    if (len == 0) return error.InvalidLength;
    for (child_types[1..]) |child_type|
        if (child_type.len != len) return error.TypeMismatch;

    switch (operation) {
        .float_fields => |dtype| {
            const fields = dtype.floatFields() orelse return error.InvalidParameter;
            if (child_types.len != 3 or
                child_types[0].bits != 1 or
                child_types[1].bits != fields.exp or
                child_types[2].bits != fields.mant)
                return error.TypeMismatch;
            return .{ .bits = fields.total, .len = len };
        },
        .bit_planes => {
            for (child_types) |child_type|
                if (child_type.bits != 1) return error.TypeMismatch;
            return .{ .bits = @intCast(child_types.len), .len = len };
        },
        .byte_planes => {
            if (child_types.len > 4) return error.InvalidArity;
            for (child_types[0 .. child_types.len - 1]) |child_type|
                if (child_type.bits != 8) return error.TypeMismatch;
            const final_bits = child_types[child_types.len - 1].bits;
            if (final_bits == 0 or final_bits > 8) return error.TypeMismatch;
        },
        .fields => |low_bits| {
            if (child_types.len != 2 or low_bits == 0 or
                child_types[0].bits != low_bits)
                return error.TypeMismatch;
        },
    }

    var bits: usize = 0;
    for (child_types) |child_type| {
        bits = std.math.add(usize, bits, child_type.bits) catch
            return error.LengthOverflow;
    }
    if (bits == 0 or bits > 32) return error.InvalidWordWidth;
    return .{ .bits = @intCast(bits), .len = len };
}
