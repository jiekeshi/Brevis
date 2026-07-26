//! Concrete bijections used by the semantic DSL and target decomposition.

const std = @import("std");
const dsl = @import("dsl.zig");
const types = @import("types.zig");

pub fn mapForward(operation: dsl.MapOp, bits: u8, word: u32) dsl.ValidationError!u32 {
    try operation.validate(bits);
    const mask = wordMask(bits);
    return switch (operation) {
        .xor => |constant| (word ^ constant) & mask,
        .add_mod => |constant| (word +% constant) & mask,
        .zigzag => ((word << 1) ^ (0 -% (word >> @intCast(bits - 1)))) & mask,
        .gray => (word ^ (word >> 1)) & mask,
        .rotate_left => |amount| rotate(word, bits, amount, false),
        .bit_reverse => reverseBits(word, bits),
    };
}

pub fn mapInverse(operation: dsl.MapOp, bits: u8, word: u32) dsl.ValidationError!u32 {
    try operation.validate(bits);
    const mask = wordMask(bits);
    return switch (operation) {
        .xor => |constant| (word ^ constant) & mask,
        .add_mod => |constant| (word -% constant) & mask,
        .zigzag => ((word >> 1) ^ (0 -% (word & 1))) & mask,
        .gray => inverseGray(word, bits),
        .rotate_left => |amount| rotate(word, bits, amount, true),
        .bit_reverse => reverseBits(word, bits),
    };
}

pub fn scanForward(operation: dsl.ScanOp, bits: u8, previous: u32, update: u32) dsl.ValidationError!u32 {
    if (bits == 0 or bits > 32) return error.InvalidWordWidth;
    const mask = wordMask(bits);
    return switch (operation) {
        .xor => (previous ^ update) & mask,
        .add_mod => (previous +% update) & mask,
    };
}

pub fn scanUpdate(operation: dsl.ScanOp, bits: u8, previous: u32, next: u32) dsl.ValidationError!u32 {
    if (bits == 0 or bits > 32) return error.InvalidWordWidth;
    const mask = wordMask(bits);
    return switch (operation) {
        .xor => (previous ^ next) & mask,
        .add_mod => (next -% previous) & mask,
    };
}

pub fn mergeWord(
    operation: dsl.MergeOp,
    children: []const types.Stream,
    index: usize,
) dsl.ValidationError!u32 {
    if (children.len < 2) return error.InvalidArity;
    return switch (operation) {
        .float_fields => |dtype| blk: {
            const fields = dtype.floatFields() orelse return error.InvalidParameter;
            if (children.len != 3 or
                children[0].bits_per_elem != 1 or
                children[1].bits_per_elem != fields.exp or
                children[2].bits_per_elem != fields.mant)
                return error.TypeMismatch;
            const sign = children[0].getU32(index);
            const exponent = children[1].getU32(index);
            const mantissa = children[2].getU32(index);
            break :blk mantissa |
                (exponent << @intCast(fields.mant)) |
                (sign << @intCast(fields.mant + fields.exp));
        },
        .fields, .bit_planes, .byte_planes => blk: {
            var output: u32 = 0;
            var shift: u8 = 0;
            for (children) |child| {
                if (child.bits_per_elem == 0 or child.bits_per_elem > 32 - shift)
                    return error.InvalidWordWidth;
                output |= (child.getU32(index) & wordMask(child.bits_per_elem)) <<
                    @intCast(shift);
                shift += child.bits_per_elem;
            }
            break :blk output;
        },
    };
}

fn wordMask(bits: u8) u32 {
    return if (bits == 32)
        std.math.maxInt(u32)
    else
        (@as(u32, 1) << @intCast(bits)) - 1;
}

fn rotate(word: u32, bits: u8, amount: u8, inverse: bool) u32 {
    const mask = wordMask(bits);
    if (amount == 0) return word & mask;
    const effective: u8 = if (inverse) bits - amount else amount;
    return ((word << @intCast(effective)) |
        ((word & mask) >> @intCast(bits - effective))) & mask;
}

fn reverseBits(word: u32, bits: u8) u32 {
    var result: u32 = 0;
    for (0..bits) |bit|
        result |= ((word >> @intCast(bit)) & 1) << @intCast(bits - 1 - bit);
    return result;
}

fn inverseGray(word: u32, bits: u8) u32 {
    var result = word;
    var shift: u8 = 1;
    while (shift < bits) : (shift *= 2)
        result ^= result >> @intCast(shift);
    return result & wordMask(bits);
}
