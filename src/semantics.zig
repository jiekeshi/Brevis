//! Concrete bijections used by the semantic DSL and target decomposition.

const std = @import("std");
const builtin = @import("builtin");
const dsl = @import("dsl.zig");
const types = @import("types.zig");

const Stream = types.Stream;
const Direction = enum { forward, inverse };

pub const PreparedMap = struct {
    operation: dsl.MapOp,
    bits: u8,
    mask: u32,

    pub fn init(operation: dsl.MapOp, bits: u8) dsl.ValidationError!PreparedMap {
        try operation.validate(bits);
        return .{ .operation = operation, .bits = bits, .mask = wordMask(bits) };
    }

    pub inline fn forwardWord(self: PreparedMap, word: u32) u32 {
        return switch (self.operation) {
            .xor => |constant| (word ^ constant) & self.mask,
            .add_mod => |constant| (word +% constant) & self.mask,
            .zigzag => ((word << 1) ^
                (0 -% (word >> @intCast(self.bits - 1)))) & self.mask,
            .gray => (word ^ (word >> 1)) & self.mask,
            .rotate_left => |amount| rotate(word, self.bits, amount, false),
            .bit_reverse => reverseBits(word, self.bits),
        };
    }

    pub inline fn inverseWord(self: PreparedMap, word: u32) u32 {
        return switch (self.operation) {
            .xor => |constant| (word ^ constant) & self.mask,
            .add_mod => |constant| (word -% constant) & self.mask,
            .zigzag => ((word >> 1) ^ (0 -% (word & 1))) & self.mask,
            .gray => inverseGray(word, self.bits),
            .rotate_left => |amount| rotate(word, self.bits, amount, true),
            .bit_reverse => reverseBits(word, self.bits),
        };
    }

    pub fn forwardInPlace(
        self: PreparedMap,
        stream: *Stream,
        offset: usize,
        count: usize,
    ) void {
        std.debug.assert(stream.bits_per_elem == self.bits);
        std.debug.assert(offset <= stream.count and count <= stream.count - offset);
        const stride = stream.elemBytes();
        const start = offset * stride;
        var region = Stream{
            .data = stream.data[start..][0 .. count * stride],
            .count = count,
            .bits_per_elem = self.bits,
            .owns_data = false,
        };
        mapInto(self, region, &region, .forward);
    }

    pub fn inverseInto(
        self: PreparedMap,
        input: Stream,
        output: *Stream,
    ) void {
        std.debug.assert(input.bits_per_elem == self.bits);
        std.debug.assert(output.bits_per_elem == self.bits);
        std.debug.assert(input.count == output.count);
        mapInto(self, input, output, .inverse);
    }
};

pub const PreparedScan = struct {
    operation: dsl.ScanOp,
    bits: u8,
    mask: u32,

    pub fn init(operation: dsl.ScanOp, bits: u8) dsl.ValidationError!PreparedScan {
        if (bits == 0 or bits > 32) return error.InvalidWordWidth;
        return .{ .operation = operation, .bits = bits, .mask = wordMask(bits) };
    }

    pub inline fn forwardWord(
        self: PreparedScan,
        previous: u32,
        update: u32,
    ) u32 {
        return switch (self.operation) {
            .xor => (previous ^ update) & self.mask,
            .add_mod => (previous +% update) & self.mask,
        };
    }

    pub inline fn updateWord(
        self: PreparedScan,
        previous: u32,
        next: u32,
    ) u32 {
        return switch (self.operation) {
            .xor => (previous ^ next) & self.mask,
            .add_mod => (next -% previous) & self.mask,
        };
    }

    pub fn updatesInto(
        self: PreparedScan,
        target: Stream,
        updates: *Stream,
    ) void {
        std.debug.assert(target.bits_per_elem == self.bits);
        std.debug.assert(updates.bits_per_elem == self.bits);
        std.debug.assert(target.count > 0 and updates.count == target.count - 1);

        if (builtin.cpu.arch.endian() == .little) {
            switch (target.elemBytes()) {
                1 => scanUpdatesVector(u8, self, target, updates),
                2 => scanUpdatesVector(u16, self, target, updates),
                else => scanUpdatesVector(u32, self, target, updates),
            }
        } else {
            for (0..updates.count) |i|
                updates.setU32(i, self.updateWord(
                    target.getU32(i),
                    target.getU32(i + 1),
                ));
        }
    }
};

fn mapInto(
    prepared: PreparedMap,
    input: Stream,
    output: *Stream,
    comptime direction: Direction,
) void {
    const byte_len = input.count * input.elemBytes();
    if (builtin.cpu.arch.endian() == .little) {
        const input_bytes = input.data[0..byte_len];
        const output_bytes = output.data[0..byte_len];
        switch (input.elemBytes()) {
            1 => mapVector(u8, prepared, input_bytes, output_bytes, direction),
            2 => mapVector(u16, prepared, input_bytes, output_bytes, direction),
            else => mapVector(u32, prepared, input_bytes, output_bytes, direction),
        }
    } else {
        for (0..input.count) |i| {
            const word = input.getU32(i);
            output.setU32(i, switch (direction) {
                .forward => prepared.forwardWord(word),
                .inverse => prepared.inverseWord(word),
            });
        }
    }
}

fn mapVector(
    comptime T: type,
    prepared: PreparedMap,
    input: []const u8,
    output: []u8,
    comptime direction: Direction,
) void {
    const vector_len = std.simd.suggestVectorLength(T) orelse 1;
    const Vector = @Vector(vector_len, T);
    const vector_bytes = @sizeOf(Vector);
    var byte_index: usize = 0;

    if (comptime vector_len > 1) {
        while (byte_index + vector_bytes <= input.len) : (byte_index += vector_bytes) {
            const words = std.mem.bytesToValue(
                Vector,
                input[byte_index..][0..vector_bytes],
            );
            var transformed = mapVectorWords(
                T,
                vector_len,
                prepared,
                words,
                direction,
            );
            @memcpy(
                output[byte_index..][0..vector_bytes],
                std.mem.asBytes(&transformed),
            );
        }
    }

    const stride = @sizeOf(T);
    while (byte_index < input.len) : (byte_index += stride) {
        const word: u32 = std.mem.readInt(
            T,
            input[byte_index..][0..stride],
            .little,
        );
        const transformed = switch (direction) {
            .forward => prepared.forwardWord(word),
            .inverse => prepared.inverseWord(word),
        };
        std.mem.writeInt(
            T,
            output[byte_index..][0..stride],
            @truncate(transformed),
            .little,
        );
    }
}

fn mapVectorWords(
    comptime T: type,
    comptime len: comptime_int,
    prepared: PreparedMap,
    words: @Vector(len, T),
    comptime direction: Direction,
) @Vector(len, T) {
    const Vector = @Vector(len, T);
    const ShiftVector = @Vector(len, std.math.Log2Int(T));
    const mask: Vector = @splat(@truncate(prepared.mask));
    const zero: Vector = @splat(0);
    const one: Vector = @splat(1);
    const shift_one: ShiftVector = @splat(1);

    return switch (prepared.operation) {
        .xor => |constant| (words ^ @as(Vector, @splat(@truncate(constant)))) & mask,
        .add_mod => |constant| blk: {
            const value: Vector = @splat(@truncate(constant));
            break :blk (if (direction == .forward)
                words +% value
            else
                words -% value) & mask;
        },
        .zigzag => if (direction == .forward)
            ((words << shift_one) ^
                (zero -% (words >> @as(ShiftVector, @splat(
                    @intCast(prepared.bits - 1),
                ))))) & mask
        else
            ((words >> shift_one) ^ (zero -% (words & one))) & mask,
        .gray => if (direction == .forward)
            (words ^ (words >> shift_one)) & mask
        else
            inverseGrayVector(T, len, words, prepared.bits, mask),
        .rotate_left => |amount| rotateVector(
            T,
            len,
            words,
            prepared.bits,
            amount,
            direction == .inverse,
            mask,
        ),
        .bit_reverse => reverseVector(T, len, words, prepared.bits, mask),
    };
}

fn scanUpdatesVector(
    comptime T: type,
    prepared: PreparedScan,
    target: Stream,
    updates: *Stream,
) void {
    const vector_len = std.simd.suggestVectorLength(T) orelse 1;
    const Vector = @Vector(vector_len, T);
    const vector_bytes = @sizeOf(Vector);
    const stride = @sizeOf(T);
    const byte_len = updates.count * stride;
    const previous_bytes = target.data[0..byte_len];
    const next_bytes = target.data[stride..][0..byte_len];
    var byte_index: usize = 0;

    if (comptime vector_len > 1) {
        const mask: Vector = @splat(@truncate(prepared.mask));
        while (byte_index + vector_bytes <= byte_len) : (byte_index += vector_bytes) {
            const previous = std.mem.bytesToValue(
                Vector,
                previous_bytes[byte_index..][0..vector_bytes],
            );
            const next = std.mem.bytesToValue(
                Vector,
                next_bytes[byte_index..][0..vector_bytes],
            );
            var result = switch (prepared.operation) {
                .xor => (previous ^ next) & mask,
                .add_mod => (next -% previous) & mask,
            };
            @memcpy(
                updates.data[byte_index..][0..vector_bytes],
                std.mem.asBytes(&result),
            );
        }
    }

    while (byte_index < byte_len) : (byte_index += stride) {
        const previous: u32 = std.mem.readInt(
            T,
            previous_bytes[byte_index..][0..stride],
            .little,
        );
        const next: u32 = std.mem.readInt(
            T,
            next_bytes[byte_index..][0..stride],
            .little,
        );
        std.mem.writeInt(
            T,
            updates.data[byte_index..][0..stride],
            @truncate(prepared.updateWord(previous, next)),
            .little,
        );
    }
}

fn rotateVector(
    comptime T: type,
    comptime len: comptime_int,
    words: @Vector(len, T),
    bits: u8,
    amount: u8,
    inverse: bool,
    mask: @Vector(len, T),
) @Vector(len, T) {
    if (amount == 0) return words & mask;
    const ShiftVector = @Vector(len, std.math.Log2Int(T));
    const effective: u8 = if (inverse) bits - amount else amount;
    const left: ShiftVector = @splat(@intCast(effective));
    const right: ShiftVector = @splat(@intCast(bits - effective));
    return ((words << left) | ((words & mask) >> right)) & mask;
}

fn reverseVector(
    comptime T: type,
    comptime len: comptime_int,
    words: @Vector(len, T),
    bits: u8,
    mask: @Vector(len, T),
) @Vector(len, T) {
    const ShiftVector = @Vector(len, std.math.Log2Int(T));
    var result = @bitReverse(words);
    const unused: u8 = @intCast(@bitSizeOf(T) - bits);
    if (unused != 0)
        result >>= @as(ShiftVector, @splat(@intCast(unused)));
    return result & mask;
}

fn inverseGrayVector(
    comptime T: type,
    comptime len: comptime_int,
    words: @Vector(len, T),
    bits: u8,
    mask: @Vector(len, T),
) @Vector(len, T) {
    const ShiftVector = @Vector(len, std.math.Log2Int(T));
    var result = words;
    var shift: u8 = 1;
    while (shift < bits) : (shift *= 2)
        result ^= result >> @as(ShiftVector, @splat(@intCast(shift)));
    return result & mask;
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
    return @bitReverse(word) >> @intCast(32 - bits);
}

fn inverseGray(word: u32, bits: u8) u32 {
    var result = word;
    var shift: u8 = 1;
    while (shift < bits) : (shift *= 2)
        result ^= result >> @intCast(shift);
    return result & wordMask(bits);
}
