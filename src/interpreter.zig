//! Exact interpreter for checked semantic DSL programs.
//!
//! `executeInto` is the primary execution seam. It writes into caller-owned
//! storage and materializes no intermediate stream for Lit, Const, Concat,
//! Repeat, Map, or Scan. Merge evaluates one child at a time, so its peak
//! scratch space is the largest child rather than the sum of all children.

const std = @import("std");
const dsl = @import("dsl.zig");
const semantics = @import("semantics.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const Stream = types.Stream;

pub const OutputError = error{
    OutputTypeMismatch,
    OutputLengthMismatch,
    OutputBufferTooSmall,
};

pub const ExecuteError = Allocator.Error || dsl.ValidationError || OutputError;
pub const TensorExecuteError = ExecuteError || dsl.TensorValidationError;

/// Allocate and execute a program. Prefer `executeInto` when the caller
/// already owns the final tensor buffer.
pub fn execute(alloc: Allocator, program: dsl.Program) ExecuteError!Stream {
    const output_type = try program.typeOf();
    var output = try Stream.init(alloc, output_type.len, output_type.bits);
    errdefer output.deinit(alloc);

    try executeValidatedInto(alloc, program, output, output_type);
    return output;
}

/// Execute exactly into a caller-provided stream.
///
/// The logical element width and count must exactly match the program type.
/// The byte slice may be larger than the logical stream, but it must contain
/// enough storage for every element. Only the logical stream region is
/// modified.
pub fn executeInto(
    alloc: Allocator,
    program: dsl.Program,
    output: Stream,
) ExecuteError!void {
    const output_type = try program.typeOf();
    try executeValidatedInto(alloc, program, output, output_type);
}

pub fn executeTensor(
    alloc: Allocator,
    tensor_program: dsl.TensorProgram,
) TensorExecuteError!Stream {
    const tensor_type = try tensor_program.validate();
    var output = try Stream.init(
        alloc,
        tensor_type.elements,
        tensor_type.dtype.bitWidth(),
    );
    errdefer output.deinit(alloc);

    const program_type = dsl.StreamType{
        .bits = tensor_type.dtype.bitWidth(),
        .len = tensor_type.elements,
    };
    try executeValidatedInto(alloc, tensor_program.root, output, program_type);
    return output;
}

/// Validate the tensor binding and execute into caller-owned storage.
pub fn executeTensorInto(
    alloc: Allocator,
    tensor_program: dsl.TensorProgram,
    output: Stream,
) TensorExecuteError!void {
    const tensor_type = try tensor_program.validate();
    const program_type = dsl.StreamType{
        .bits = tensor_type.dtype.bitWidth(),
        .len = tensor_type.elements,
    };
    try executeValidatedInto(alloc, tensor_program.root, output, program_type);
}

fn executeValidatedInto(
    alloc: Allocator,
    program: dsl.Program,
    output_value: Stream,
    output_type: dsl.StreamType,
) ExecuteError!void {
    try validateOutput(output_value, output_type);

    var output = output_value;
    const written = try writeInto(alloc, program, &output, 0);
    if (written != output.count) return error.OutputLengthMismatch;
}

fn validateOutput(output: Stream, expected: dsl.StreamType) ExecuteError!void {
    if (output.bits_per_elem != expected.bits)
        return error.OutputTypeMismatch;
    if (output.count != expected.len)
        return error.OutputLengthMismatch;

    const required = std.math.mul(usize, output.count, output.elemBytes()) catch
        return error.LengthOverflow;
    if (output.data.len < required)
        return error.OutputBufferTooSmall;
}

fn writeInto(
    alloc: Allocator,
    program: dsl.Program,
    output: *Stream,
    offset: usize,
) ExecuteError!usize {
    return switch (program.kind) {
        .literal => |literal| blk: {
            if (literal.bits_per_elem != output.bits_per_elem)
                return error.OutputTypeMismatch;
            try validateRegion(output.*, offset, literal.count);

            const mask = wordMask(literal.bits_per_elem);
            for (0..literal.count) |i| {
                const word = literal.getU32(i);
                if (word & ~mask != 0) return error.InvalidLiteralValue;
                output.setU32(offset + i, word);
            }
            break :blk literal.count;
        },
        .constant => |constant_value| blk: {
            if (constant_value.bits != output.bits_per_elem)
                return error.OutputTypeMismatch;
            try validateRegion(output.*, offset, constant_value.len);
            for (0..constant_value.len) |i|
                output.setU32(offset + i, constant_value.word);
            break :blk constant_value.len;
        },
        .concat => blk: {
            if (program.children.len < 2) return error.InvalidArity;
            var written: usize = 0;
            for (program.children) |child| {
                const child_offset = std.math.add(usize, offset, written) catch
                    return error.LengthOverflow;
                const child_written = try writeInto(
                    alloc,
                    child,
                    output,
                    child_offset,
                );
                written = std.math.add(usize, written, child_written) catch
                    return error.LengthOverflow;
            }
            break :blk written;
        },
        .repeat => |times| blk: {
            if (program.children.len != 1 or times < 2)
                return error.InvalidRepeatCount;

            const child_type = try program.children[0].typeOf();
            if (child_type.bits != output.bits_per_elem)
                return error.OutputTypeMismatch;
            const total = std.math.mul(
                usize,
                child_type.len,
                @as(usize, @intCast(times)),
            ) catch return error.LengthOverflow;
            try validateRegion(output.*, offset, total);

            const child_written = try writeInto(
                alloc,
                program.children[0],
                output,
                offset,
            );
            if (child_written != child_type.len)
                return error.OutputLengthMismatch;

            // The DSL is pure: evaluate one period once, then duplicate its
            // physical words directly into the remaining output regions.
            const elem_bytes = output.elemBytes();
            const source_byte = std.math.mul(usize, offset, elem_bytes) catch
                return error.LengthOverflow;
            const period_bytes = std.math.mul(
                usize,
                child_written,
                elem_bytes,
            ) catch return error.LengthOverflow;
            const total_bytes = std.math.mul(
                usize,
                total,
                elem_bytes,
            ) catch return error.LengthOverflow;
            var filled_bytes = period_bytes;
            while (filled_bytes < total_bytes) {
                // The already-emitted prefix is itself a repetition of the
                // period, so doubling it preserves semantics while avoiding
                // one tiny copy call per repetition for short periods.
                const copy_bytes = @min(
                    filled_bytes,
                    total_bytes - filled_bytes,
                );
                const source =
                    output.data[source_byte .. source_byte + copy_bytes];
                const destination_start = source_byte + filled_bytes;
                const destination =
                    output.data[destination_start .. destination_start + copy_bytes];
                std.mem.copyForwards(u8, destination, source);
                filled_bytes += copy_bytes;
            }
            break :blk total;
        },
        .map => |operation| blk: {
            if (program.children.len != 1) return error.InvalidArity;
            const child_written = try writeInto(
                alloc,
                program.children[0],
                output,
                offset,
            );
            try validateRegion(output.*, offset, child_written);
            for (0..child_written) |i| {
                const output_index = offset + i;
                output.setU32(output_index, try semantics.mapForward(
                    operation,
                    output.bits_per_elem,
                    output.getU32(output_index),
                ));
            }
            break :blk child_written;
        },
        .scan => |scan_value| blk: {
            if (program.children.len != 1) return error.InvalidArity;
            const updates_offset = std.math.add(usize, offset, 1) catch
                return error.LengthOverflow;
            const updates_written = try writeInto(
                alloc,
                program.children[0],
                output,
                updates_offset,
            );
            const count = std.math.add(usize, updates_written, 1) catch
                return error.LengthOverflow;
            try validateRegion(output.*, offset, count);

            var previous = scan_value.initial;
            output.setU32(offset, previous);
            for (0..updates_written) |i| {
                const update_index = updates_offset + i;
                previous = try semantics.scanForward(
                    scan_value.operation,
                    output.bits_per_elem,
                    previous,
                    output.getU32(update_index),
                );
                output.setU32(update_index, previous);
            }
            break :blk count;
        },
        .merge => |operation| blk: {
            if (program.children.len < 2) return error.InvalidArity;

            const merge_type = try program.typeOf();
            if (merge_type.bits != output.bits_per_elem)
                return error.OutputTypeMismatch;
            try validateRegion(output.*, offset, merge_type.len);
            for (0..merge_type.len) |i| output.setU32(offset + i, 0);

            // Materialize and combine one child at a time. This retains exact
            // physical-word semantics while bounding scratch memory by the
            // largest child stream.
            for (program.children, 0..) |child_program, child_index| {
                const child_type = try child_program.typeOf();
                if (child_type.len != merge_type.len)
                    return error.TypeMismatch;

                var child = try Stream.init(
                    alloc,
                    child_type.len,
                    child_type.bits,
                );
                defer child.deinit(alloc);
                try executeValidatedInto(
                    alloc,
                    child_program,
                    child,
                    child_type,
                );

                const shift = try mergeChildShift(
                    operation,
                    child_index,
                    program.children.len,
                );
                for (0..merge_type.len) |i| {
                    const shifted = child.getU32(i) << @intCast(shift);
                    const output_index = offset + i;
                    output.setU32(
                        output_index,
                        output.getU32(output_index) | shifted,
                    );
                }
            }
            break :blk merge_type.len;
        },
    };
}

fn validateRegion(output: Stream, offset: usize, count: usize) ExecuteError!void {
    if (offset > output.count or count > output.count - offset)
        return error.OutputLengthMismatch;
}

fn mergeChildShift(
    operation: dsl.MergeOp,
    child_index: usize,
    child_count: usize,
) dsl.ValidationError!u8 {
    return switch (operation) {
        .float_fields => |dtype| blk: {
            const fields = dtype.floatFields() orelse
                return error.InvalidParameter;
            if (child_count != 3) return error.InvalidArity;
            break :blk switch (child_index) {
                0 => fields.mant + fields.exp,
                1 => fields.mant,
                2 => 0,
                else => return error.InvalidArity,
            };
        },
        .fields => |low_bits| switch (child_index) {
            0 => 0,
            1 => low_bits,
            else => return error.InvalidArity,
        },
        .bit_planes => std.math.cast(u8, child_index) orelse
            return error.InvalidArity,
        .byte_planes => std.math.cast(u8, child_index * 8) orelse
            return error.InvalidArity,
    };
}

fn wordMask(bits: u8) u32 {
    return if (bits == 32)
        std.math.maxInt(u32)
    else
        (@as(u32, 1) << @intCast(bits)) - 1;
}
