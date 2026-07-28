//! Target-directed partial decompositions for semantic DSL productions.
//!
//! A successful function in this module must satisfy the paper's contract:
//! executing the parent production over the returned child targets recreates
//! the supplied target exactly.

const std = @import("std");
const builtin = @import("builtin");
const dsl = @import("dsl.zig");
const semantics = @import("semantics.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const Stream = types.Stream;

/// Return the repeated period for `Repeat[times]`, or null when the target is
/// not exactly `times` copies of a non-empty stream.
pub fn repeat(
    alloc: Allocator,
    target: Stream,
    times: u32,
) (Allocator.Error || dsl.ValidationError)!?Stream {
    if (times < 2 or target.count == 0 or target.count % times != 0) return null;
    const period_len = target.count / times;
    if (period_len == 0) return null;
    if (!target.hasPeriod(period_len)) return null;

    const period = try Stream.initUninitialized(
        alloc,
        period_len,
        target.bits_per_elem,
    );
    @memcpy(period.data, target.data[0..period.data.len]);
    return period;
}

/// Apply a pointwise map's exact inverse to derive its child target.
pub fn map(
    alloc: Allocator,
    target: Stream,
    operation: dsl.MapOp,
) (Allocator.Error || dsl.ValidationError)!Stream {
    const prepared = try semantics.PreparedMap.init(
        operation,
        target.bits_per_elem,
    );
    var child = try Stream.initUninitialized(
        alloc,
        target.count,
        target.bits_per_elem,
    );
    errdefer child.deinit(alloc);
    prepared.inverseInto(target, &child);
    return child;
}

pub const ScanParts = struct {
    initial: u32,
    updates: Stream,

    pub fn deinit(self: *ScanParts, alloc: Allocator) void {
        self.updates.deinit(alloc);
    }
};

/// Derive the explicit initial word and `n - 1` update stream for a scan.
pub fn scan(
    alloc: Allocator,
    target: Stream,
    operation: dsl.ScanOp,
) (Allocator.Error || dsl.ValidationError)!?ScanParts {
    const prepared = try semantics.PreparedScan.init(
        operation,
        target.bits_per_elem,
    );
    if (target.count < 2) return null;

    var updates = try Stream.initUninitialized(
        alloc,
        target.count - 1,
        target.bits_per_elem,
    );
    errdefer updates.deinit(alloc);
    prepared.updatesInto(target, &updates);
    return .{
        .initial = target.getU32(0),
        .updates = updates,
    };
}

/// Split each target word into the pointwise child streams required by a
/// concrete merge operation.
pub fn merge(
    alloc: Allocator,
    target: Stream,
    operation: dsl.MergeOp,
) (Allocator.Error || dsl.ValidationError)![]Stream {
    if (target.bits_per_elem == 0 or target.bits_per_elem > 32)
        return error.InvalidWordWidth;
    if (target.count == 0) return error.InvalidLength;

    var widths: [32]u8 = undefined;
    const count: usize = switch (operation) {
        .fields => |low_bits| blk: {
            if (low_bits == 0 or low_bits >= target.bits_per_elem)
                return error.InvalidParameter;
            widths[0] = low_bits;
            widths[1] = target.bits_per_elem - low_bits;
            break :blk 2;
        },
        .float_fields => |dtype| blk: {
            const fields = dtype.floatFields() orelse return error.InvalidParameter;
            if (fields.total != target.bits_per_elem) return error.TypeMismatch;
            widths[0] = 1;
            widths[1] = fields.exp;
            widths[2] = fields.mant;
            break :blk 3;
        },
        .bit_planes => blk: {
            for (0..target.bits_per_elem) |i| widths[i] = 1;
            break :blk target.bits_per_elem;
        },
        .byte_planes => blk: {
            const n: usize = (@as(usize, target.bits_per_elem) + 7) / 8;
            for (0..n - 1) |i| widths[i] = 8;
            widths[n - 1] = target.bits_per_elem - @as(u8, @intCast((n - 1) * 8));
            break :blk n;
        },
    };

    const children = try alloc.alloc(Stream, count);
    var initialized: usize = 0;
    errdefer {
        for (children[0..initialized]) |*child| child.deinit(alloc);
        alloc.free(children);
    }
    for (children, widths[0..count]) |*child, bits| {
        child.* = try Stream.initUninitialized(alloc, target.count, bits);
        initialized += 1;
    }

    if (builtin.cpu.arch.endian() == .little) {
        switch (operation) {
            .float_fields => |dtype| {
                splitFloatFields(target, children, dtype.floatFields().?);
            },
            .fields => |low_bits| {
                splitFields(target, children, low_bits);
            },
            .bit_planes => {
                splitPlanes(target, children, 1);
            },
            .byte_planes => {
                splitPlanes(target, children, 8);
            },
        }
        return children;
    }

    switch (operation) {
        .float_fields => |dtype| {
            const fields = dtype.floatFields().?;
            const mantissa_mask = mask(fields.mant);
            const exponent_mask = mask(fields.exp);
            for (0..target.count) |i| {
                const word = target.getU32(i);
                children[0].setU32(i, word >> @intCast(fields.total - 1));
                children[1].setU32(i, (word >> @intCast(fields.mant)) & exponent_mask);
                children[2].setU32(i, word & mantissa_mask);
            }
        },
        .fields, .bit_planes, .byte_planes => {
            for (0..target.count) |i| {
                const word = target.getU32(i);
                var shift: u8 = 0;
                for (children) |*child| {
                    child.setU32(i, (word >> @intCast(shift)) & mask(child.bits_per_elem));
                    shift += child.bits_per_elem;
                }
            }
        },
    }
    return children;
}

fn splitFloatFields(
    target: Stream,
    children: []Stream,
    fields: types.Dtype.FloatFields,
) void {
    switch (target.elemBytes()) {
        1 => splitFloatFieldsTyped(u8, u8, target, children, fields),
        2 => switch (children[2].elemBytes()) {
            1 => splitFloatFieldsTyped(u16, u8, target, children, fields),
            else => splitFloatFieldsTyped(u16, u16, target, children, fields),
        },
        else => splitFloatFieldsTyped(u32, u32, target, children, fields),
    }
}

fn splitFloatFieldsTyped(
    comptime Source: type,
    comptime Mantissa: type,
    target: Stream,
    children: []Stream,
    fields: types.Dtype.FloatFields,
) void {
    const lanes = std.simd.suggestVectorLength(u8) orelse 16;
    const SourceVector = @Vector(lanes, Source);
    const ShiftVector = @Vector(lanes, std.math.Log2Int(Source));
    const exponent_mask: SourceVector = @splat(@intCast(mask(fields.exp)));
    const mantissa_mask: SourceVector = @splat(@intCast(mask(fields.mant)));
    const sign_shift: ShiftVector = @splat(@intCast(fields.total - 1));
    const exponent_shift: ShiftVector = @splat(@intCast(fields.mant));
    var i: usize = 0;

    while (i + lanes <= target.count) : (i += lanes) {
        const words = loadVector(Source, lanes, target.data, i);
        storeVector(Source, u8, lanes, words >> sign_shift, children[0].data, i);
        storeVector(
            Source,
            u8,
            lanes,
            (words >> exponent_shift) & exponent_mask,
            children[1].data,
            i,
        );
        storeVector(
            Source,
            Mantissa,
            lanes,
            words & mantissa_mask,
            children[2].data,
            i,
        );
    }

    while (i < target.count) : (i += 1) {
        const word = target.getU32(i);
        children[0].setU32(i, word >> @intCast(fields.total - 1));
        children[1].setU32(
            i,
            (word >> @intCast(fields.mant)) & mask(fields.exp),
        );
        children[2].setU32(i, word & mask(fields.mant));
    }
}

fn splitFields(target: Stream, children: []Stream, low_bits: u8) void {
    switch (target.elemBytes()) {
        1 => splitFieldsSource(u8, target, children, low_bits),
        2 => splitFieldsSource(u16, target, children, low_bits),
        else => splitFieldsSource(u32, target, children, low_bits),
    }
}

fn splitFieldsSource(
    comptime Source: type,
    target: Stream,
    children: []Stream,
    low_bits: u8,
) void {
    switch (children[0].elemBytes()) {
        1 => switch (children[1].elemBytes()) {
            1 => splitFieldsTyped(Source, u8, u8, target, children, low_bits),
            2 => splitFieldsTyped(Source, u8, u16, target, children, low_bits),
            else => splitFieldsTyped(Source, u8, u32, target, children, low_bits),
        },
        2 => switch (children[1].elemBytes()) {
            1 => splitFieldsTyped(Source, u16, u8, target, children, low_bits),
            2 => splitFieldsTyped(Source, u16, u16, target, children, low_bits),
            else => splitFieldsTyped(Source, u16, u32, target, children, low_bits),
        },
        else => switch (children[1].elemBytes()) {
            1 => splitFieldsTyped(Source, u32, u8, target, children, low_bits),
            2 => splitFieldsTyped(Source, u32, u16, target, children, low_bits),
            else => splitFieldsTyped(Source, u32, u32, target, children, low_bits),
        },
    }
}

fn splitFieldsTyped(
    comptime Source: type,
    comptime Low: type,
    comptime High: type,
    target: Stream,
    children: []Stream,
    low_bits: u8,
) void {
    const lanes = std.simd.suggestVectorLength(u8) orelse 16;
    const SourceVector = @Vector(lanes, Source);
    const ShiftVector = @Vector(lanes, std.math.Log2Int(Source));
    const low_mask: SourceVector = @splat(@intCast(mask(low_bits)));
    const high_mask: SourceVector = @splat(@intCast(mask(
        target.bits_per_elem - low_bits,
    )));
    const shift: ShiftVector = @splat(@intCast(low_bits));
    var i: usize = 0;

    while (i + lanes <= target.count) : (i += lanes) {
        const words = loadVector(Source, lanes, target.data, i);
        storeVector(Source, Low, lanes, words & low_mask, children[0].data, i);
        storeVector(
            Source,
            High,
            lanes,
            (words >> shift) & high_mask,
            children[1].data,
            i,
        );
    }

    while (i < target.count) : (i += 1) {
        const word = target.getU32(i);
        children[0].setU32(i, word & mask(low_bits));
        children[1].setU32(
            i,
            (word >> @intCast(low_bits)) &
                mask(target.bits_per_elem - low_bits),
        );
    }
}

fn splitPlanes(target: Stream, children: []Stream, comptime step: u8) void {
    if (target.bits_per_elem == 8 and children.len == 1 and step == 8) {
        @memcpy(children[0].data, target.data);
        return;
    }
    switch (target.elemBytes()) {
        1 => splitPlanesTyped(u8, target, children, step),
        2 => splitPlanesTyped(u16, target, children, step),
        else => splitPlanesTyped(u32, target, children, step),
    }
}

fn splitPlanesTyped(
    comptime Source: type,
    target: Stream,
    children: []Stream,
    comptime step: u8,
) void {
    const lanes = std.simd.suggestVectorLength(u8) orelse 16;
    const SourceVector = @Vector(lanes, Source);
    const ShiftVector = @Vector(lanes, std.math.Log2Int(Source));
    var i: usize = 0;

    while (i + lanes <= target.count) : (i += lanes) {
        const words = loadVector(Source, lanes, target.data, i);
        for (children, 0..) |child, plane| {
            const shift: ShiftVector = @splat(@intCast(plane * step));
            const value_mask: SourceVector = @splat(@intCast(mask(
                child.bits_per_elem,
            )));
            storeVector(
                Source,
                u8,
                lanes,
                (words >> shift) & value_mask,
                child.data,
                i,
            );
        }
    }

    while (i < target.count) : (i += 1) {
        const word = target.getU32(i);
        for (children, 0..) |*child, plane| {
            child.setU32(
                i,
                (word >> @intCast(plane * step)) & mask(child.bits_per_elem),
            );
        }
    }
}

fn loadVector(
    comptime T: type,
    comptime lanes: comptime_int,
    data: []const u8,
    element_index: usize,
) @Vector(lanes, T) {
    const Vector = @Vector(lanes, T);
    const offset = element_index * @sizeOf(T);
    return std.mem.bytesToValue(
        Vector,
        data[offset..][0..@sizeOf(Vector)],
    );
}

fn storeVector(
    comptime Source: type,
    comptime Dest: type,
    comptime lanes: comptime_int,
    values: @Vector(lanes, Source),
    data: []u8,
    element_index: usize,
) void {
    const Vector = @Vector(lanes, Dest);
    var narrowed: Vector = if (Source == Dest) values else @truncate(values);
    const offset = element_index * @sizeOf(Dest);
    @memcpy(
        data[offset..][0..@sizeOf(Vector)],
        std.mem.asBytes(&narrowed),
    );
}

fn mask(bits: u8) u32 {
    return if (bits == 32)
        std.math.maxInt(u32)
    else
        (@as(u32, 1) << @intCast(bits)) - 1;
}
