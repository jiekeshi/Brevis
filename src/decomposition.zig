//! Target-directed partial decompositions for semantic DSL productions.
//!
//! A successful function in this module must satisfy the paper's contract:
//! executing the parent production over the returned child targets recreates
//! the supplied target exactly.

const std = @import("std");
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

    for (period_len..target.count) |i|
        if (target.getU32(i) != target.getU32(i % period_len)) return null;

    var period = try Stream.init(alloc, period_len, target.bits_per_elem);
    for (0..period_len) |i| period.setU32(i, target.getU32(i));
    return period;
}

/// Apply a pointwise map's exact inverse to derive its child target.
pub fn map(
    alloc: Allocator,
    target: Stream,
    operation: dsl.MapOp,
) (Allocator.Error || dsl.ValidationError)!Stream {
    try operation.validate(target.bits_per_elem);
    var child = try Stream.init(alloc, target.count, target.bits_per_elem);
    errdefer child.deinit(alloc);
    for (0..target.count) |i|
        child.setU32(i, try semantics.mapInverse(
            operation,
            target.bits_per_elem,
            target.getU32(i),
        ));
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
    if (target.bits_per_elem == 0 or target.bits_per_elem > 32)
        return error.InvalidWordWidth;
    if (target.count < 2) return null;

    var updates = try Stream.init(alloc, target.count - 1, target.bits_per_elem);
    errdefer updates.deinit(alloc);
    var previous = target.getU32(0);
    for (1..target.count) |i| {
        const next = target.getU32(i);
        updates.setU32(i - 1, try semantics.scanUpdate(
            operation,
            target.bits_per_elem,
            previous,
            next,
        ));
        previous = next;
    }
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
        child.* = try Stream.init(alloc, target.count, bits);
        initialized += 1;
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

fn mask(bits: u8) u32 {
    return if (bits == 32)
        std.math.maxInt(u32)
    else
        (@as(u32, 1) << @intCast(bits)) - 1;
}
