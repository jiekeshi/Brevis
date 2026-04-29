//! Core data types: dtypes, streams, tensor views.
//!
//! A Stream is a 1-D byte buffer with a logical "bits per element". This is
//! how data flows between ops in the DSL. We always store the raw bytes;
//! `bits_per_elem` is metadata for entropy coders and bit-pack helpers.
//!
//! A TensorView wraps a typed multi-dimensional buffer. We deliberately don't
//! own the data here — ops that produce new tensors return owned buffers and
//! the caller frees them.

const std = @import("std");

pub const Allocator = std.mem.Allocator;

pub const Dtype = enum(u8) {
    f16 = 0,
    bf16 = 1,
    f32 = 2,
    u8 = 3,
    u16 = 4,
    u32 = 5,

    pub fn elemSize(self: Dtype) usize {
        return switch (self) {
            .f16, .bf16, .u16 => 2,
            .f32, .u32 => 4,
            .u8 => 1,
        };
    }

    pub fn isFloat16Like(self: Dtype) bool {
        return self == .f16 or self == .bf16;
    }

    pub fn name(self: Dtype) []const u8 {
        return switch (self) {
            .f16 => "F16",
            .bf16 => "BF16",
            .f32 => "F32",
            .u8 => "U8",
            .u16 => "U16",
            .u32 => "U32",
        };
    }

    pub fn fromName(s: []const u8) ?Dtype {
        if (std.mem.eql(u8, s, "F16")) return .f16;
        if (std.mem.eql(u8, s, "BF16")) return .bf16;
        if (std.mem.eql(u8, s, "F32")) return .f32;
        if (std.mem.eql(u8, s, "U8")) return .u8;
        if (std.mem.eql(u8, s, "U16")) return .u16;
        if (std.mem.eql(u8, s, "U32")) return .u32;
        return null;
    }
};

/// A 1-D buffer of `count` elements. Each element occupies `bits_per_elem`
/// logical bits but is *physically* stored in the smallest power-of-two byte
/// width that fits — so an exponent stream with 5 bits/elem still uses 1 byte
/// per element in `data`. Compactness is the entropy coder's job.
pub const Stream = struct {
    data: []u8, // owned; caller frees with the stream's allocator
    count: usize, // number of logical elements
    bits_per_elem: u8, // 1, 2, .., 32
    owns_data: bool = true,

    pub fn elemBytes(self: Stream) usize {
        return @divExact(roundUpToPow2(self.bits_per_elem), 8);
    }

    pub fn nbytes(self: Stream) usize {
        return self.data.len;
    }

    pub fn deinit(self: *Stream, alloc: Allocator) void {
        if (self.owns_data) alloc.free(self.data);
        self.data = &.{};
    }

    pub fn cloneAlloc(self: Stream, alloc: Allocator) !Stream {
        const buf = try alloc.alloc(u8, self.data.len);
        @memcpy(buf, self.data);
        return .{ .data = buf, .count = self.count, .bits_per_elem = self.bits_per_elem };
    }

    /// Read element i as a u32 (zero-extended).
    pub fn getU32(self: Stream, i: usize) u32 {
        const bpe = roundUpToPow2(self.bits_per_elem);
        return switch (bpe) {
            8 => @intCast(self.data[i]),
            16 => @intCast(std.mem.readInt(u16, self.data[i * 2 ..][0..2], .little)),
            32 => std.mem.readInt(u32, self.data[i * 4 ..][0..4], .little),
            else => unreachable,
        };
    }

    pub fn setU32(self: Stream, i: usize, v: u32) void {
        const bpe = roundUpToPow2(self.bits_per_elem);
        switch (bpe) {
            8 => self.data[i] = @intCast(v & 0xFF),
            16 => std.mem.writeInt(u16, self.data[i * 2 ..][0..2], @intCast(v & 0xFFFF), .little),
            32 => std.mem.writeInt(u32, self.data[i * 4 ..][0..4], v, .little),
            else => unreachable,
        }
    }
};

/// A typed n-dim view onto bytes. `shape` is little-endian-row-major.
pub const TensorView = struct {
    data: []u8, // raw bytes; len = prod(shape) * dtype.elemSize()
    shape: []const u64,
    dtype: Dtype,
    owns_data: bool = false,
    owns_shape: bool = false,

    pub fn numel(self: TensorView) u64 {
        var n: u64 = 1;
        for (self.shape) |d| n *= d;
        return n;
    }

    pub fn nbytes(self: TensorView) usize {
        return self.data.len;
    }

    pub fn deinit(self: *TensorView, alloc: Allocator) void {
        if (self.owns_data) alloc.free(self.data);
        if (self.owns_shape) alloc.free(self.shape);
    }

    pub fn equalsBytes(self: TensorView, other: TensorView) bool {
        if (self.dtype != other.dtype) return false;
        if (self.shape.len != other.shape.len) return false;
        for (self.shape, other.shape) |a, b| if (a != b) return false;
        return std.mem.eql(u8, self.data, other.data);
    }
};

pub fn roundUpToPow2(bits: u8) u8 {
    if (bits == 0) return 0;
    if (bits <= 8) return 8;
    if (bits <= 16) return 16;
    if (bits <= 32) return 32;
    @panic("bits_per_elem > 32 unsupported");
}
