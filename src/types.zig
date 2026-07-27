const std = @import("std");
pub const Allocator = std.mem.Allocator;

/// The paper implementation uses a 16 GiB defensive ceiling on 64-bit hosts.
/// Saturating it to the addressable range keeps the same APIs buildable on
/// narrower targets, where a 16 GiB `usize` cannot be represented.
pub const defaultLargeByteLimit: usize =
    if (@bitSizeOf(usize) >= 64)
        16 * 1024 * 1024 * 1024
    else
        std.math.maxInt(usize);

/// Safe defaults for a single materialized tensor and for the cumulative
/// interpreter passes represented by one decoded program.
pub const defaultTensorByteLimit: usize = 512 * 1024 * 1024;
pub const defaultExecutionByteLimit: usize =
    if (@bitSizeOf(usize) >= 64)
        4 * 1024 * 1024 * 1024
    else
        std.math.maxInt(usize);

pub const Dtype = enum(u8) {
    f16 = 0,
    bf16 = 1,
    f32 = 2,
    u8 = 3,
    u16 = 4,
    u32 = 5,
    i8 = 6,
    i16 = 7,
    i32 = 8,
    f8_e4m3 = 9,
    f8_e5m2 = 10,

    pub fn elemSize(self: Dtype) usize {
        return switch (self) {
            .u8, .i8, .f8_e4m3, .f8_e5m2 => 1,
            .f16, .bf16, .u16, .i16 => 2,
            .f32, .u32, .i32 => 4,
        };
    }

    pub fn bitWidth(self: Dtype) u8 {
        return @intCast(self.elemSize() * 8);
    }

    pub fn isFloat(self: Dtype) bool {
        return self.floatFields() != null;
    }

    pub const FloatFields = struct { exp: u8, mant: u8, total: u8 };

    pub fn floatFields(self: Dtype) ?FloatFields {
        return switch (self) {
            .f16 => .{ .exp = 5, .mant = 10, .total = 16 },
            .bf16 => .{ .exp = 8, .mant = 7, .total = 16 },
            .f32 => .{ .exp = 8, .mant = 23, .total = 32 },
            .f8_e4m3 => .{ .exp = 4, .mant = 3, .total = 8 },
            .f8_e5m2 => .{ .exp = 5, .mant = 2, .total = 8 },
            else => null,
        };
    }

    pub fn name(self: Dtype) []const u8 {
        return switch (self) {
            .f16 => "F16",
            .bf16 => "BF16",
            .f32 => "F32",
            .u8 => "U8",
            .u16 => "U16",
            .u32 => "U32",
            .i8 => "I8",
            .i16 => "I16",
            .i32 => "I32",
            .f8_e4m3 => "F8_E4M3",
            .f8_e5m2 => "F8_E5M2",
        };
    }

    pub fn fromName(s: []const u8) ?Dtype {
        if (std.mem.eql(u8, s, "F16")) return .f16;
        if (std.mem.eql(u8, s, "BF16")) return .bf16;
        if (std.mem.eql(u8, s, "F32")) return .f32;
        if (std.mem.eql(u8, s, "U8")) return .u8;
        if (std.mem.eql(u8, s, "U16")) return .u16;
        if (std.mem.eql(u8, s, "U32")) return .u32;
        if (std.mem.eql(u8, s, "I8")) return .i8;
        if (std.mem.eql(u8, s, "I16")) return .i16;
        if (std.mem.eql(u8, s, "I32")) return .i32;
        if (std.mem.eql(u8, s, "F8_E4M3")) return .f8_e4m3;
        if (std.mem.eql(u8, s, "F8_E5M2")) return .f8_e5m2;
        return null;
    }
};

pub inline fn roundUpToPow2(bits: u8) u8 {
    if (bits <= 8) return 8;
    if (bits <= 16) return 16;
    return 32;
}

/// A 1-D buffer of `count` elements, each `bits_per_elem` wide. Elements are
/// stored at the smallest power-of-two byte width that fits; bit-level packing
/// is the terminal coder's job.
pub const Stream = struct {
    data: []u8,
    count: usize,
    bits_per_elem: u8,
    owns_data: bool = true,

    pub inline fn elemBytes(self: Stream) usize {
        return roundUpToPow2(self.bits_per_elem) / 8;
    }

    pub fn init(alloc: Allocator, count: usize, bits_per_elem: u8) !Stream {
        const stream = try initUninitialized(alloc, count, bits_per_elem);
        @memset(stream.data, 0);
        return stream;
    }

    pub fn initUninitialized(
        alloc: Allocator,
        count: usize,
        bits_per_elem: u8,
    ) !Stream {
        if (bits_per_elem == 0 or bits_per_elem > 32)
            return error.InvalidWordWidth;
        const w = roundUpToPow2(bits_per_elem) / 8;
        const byte_count = std.math.mul(usize, count, w) catch
            return error.LengthOverflow;
        const buf = try alloc.alloc(u8, byte_count);
        return .{ .data = buf, .count = count, .bits_per_elem = bits_per_elem };
    }

    pub fn deinit(self: *Stream, alloc: Allocator) void {
        if (self.owns_data) alloc.free(self.data);
        self.data = &.{};
    }

    pub inline fn getU32(self: Stream, i: usize) u32 {
        return switch (self.elemBytes()) {
            1 => self.data[i],
            2 => std.mem.readInt(u16, self.data[i * 2 ..][0..2], .little),
            else => std.mem.readInt(u32, self.data[i * 4 ..][0..4], .little),
        };
    }

    pub inline fn setU32(self: *Stream, i: usize, v: u32) void {
        switch (self.elemBytes()) {
            1 => self.data[i] = @truncate(v),
            2 => std.mem.writeInt(u16, self.data[i * 2 ..][0..2], @truncate(v), .little),
            else => std.mem.writeInt(u32, self.data[i * 4 ..][0..4], v, .little),
        }
    }

    pub inline fn mask(self: Stream) u32 {
        if (self.bits_per_elem >= 32) return 0xFFFF_FFFF;
        return (@as(u32, 1) << @intCast(self.bits_per_elem)) - 1;
    }

    pub fn dupe(self: Stream, alloc: Allocator) !Stream {
        return .{
            .data = try alloc.dupe(u8, self.data),
            .count = self.count,
            .bits_per_elem = self.bits_per_elem,
        };
    }
};

/// A typed n-dim view onto bytes. `shape` is row-major.
pub const TensorView = struct {
    /// Borrowed tensor bytes are read-only. This is important for views backed
    /// by a read-only memory map: the public type must not promise writes that
    /// the operating system will fault.
    data: []const u8,
    shape: []const u64,
    dtype: Dtype,
    owns_data: bool = false,
    owns_shape: bool = false,

    /// Return the number of logical elements, rejecting dimensions that do
    /// not fit the host address space or whose product overflows `usize`.
    pub fn numelChecked(self: TensorView) !usize {
        for (self.shape) |d| {
            if (d == 0) return 0;
        }
        var n: usize = 1;
        for (self.shape) |d| {
            const dim = std.math.cast(usize, d) orelse
                return error.ShapeOverflow;
            n = std.math.mul(usize, n, dim) catch
                return error.ShapeOverflow;
        }
        return n;
    }

    pub fn nbytes(self: TensorView) usize {
        return self.data.len;
    }

    pub fn deinit(self: *TensorView, alloc: Allocator) void {
        if (self.owns_data) alloc.free(@constCast(self.data));
        if (self.owns_shape) alloc.free(self.shape);
    }

    pub fn equalsBytes(self: TensorView, other: TensorView) bool {
        if (self.dtype != other.dtype) return false;
        if (self.shape.len != other.shape.len) return false;
        for (self.shape, other.shape) |a, b| if (a != b) return false;
        return std.mem.eql(u8, self.data, other.data);
    }
};
