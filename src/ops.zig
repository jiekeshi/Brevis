//! Reversible DSL ops. Each op has a `Forward` and an `Inverse` half; the
//! "inverse" reconstructs the input from the output + side_info.
//!
//! These functions are intentionally side-effect-free and don't know about
//! the program tree — that's program.zig's job.

const std = @import("std");
const types = @import("types.zig");
const codec = @import("codec.zig");

const Allocator = types.Allocator;
const Stream = types.Stream;
const TensorView = types.TensorView;
const Dtype = types.Dtype;

// ---------- split_float (fp16/bf16 -> sign/exp/mant streams) ----------
pub const SplitFloatInfo = struct {
    dtype: Dtype,
    exp_bits: u8,
    mant_bits: u8,
    ndim: u8,
    shape: [8]u64,
};

pub fn splitFloatForward(alloc: Allocator, t: TensorView) !struct {
    sign: Stream,
    exp: Stream,
    mant: Stream,
    info: SplitFloatInfo,
} {
    if (!t.dtype.isFloat16Like()) return error.NotFloat16;

    const exp_bits: u8 = if (t.dtype == .f16) 5 else 8;
    const mant_bits: u8 = if (t.dtype == .f16) 10 else 7;

    const n = t.numel();
    // Stream storage uses the smallest pow-of-2 byte width that fits the
    // logical bit width. Mantissa is 7 bits (bf16) or 10 bits (fp16) — bf16
    // packs into 1 byte/elem, fp16 needs 2 bytes/elem.
    const sign_bytes_per: usize = 1; // 1 bit -> 1 byte
    const exp_bytes_per: usize = @divExact(types.roundUpToPow2(exp_bits), 8);
    const mant_bytes_per: usize = @divExact(types.roundUpToPow2(mant_bits), 8);

    const sign_buf = try alloc.alloc(u8, @intCast(n * sign_bytes_per));
    const exp_buf = try alloc.alloc(u8, @intCast(n * exp_bytes_per));
    const mant_buf = try alloc.alloc(u8, @intCast(n * mant_bytes_per));

    const exp_mask: u16 = @intCast((@as(u32, 1) << @intCast(exp_bits)) - 1);
    const mant_mask: u16 = @intCast((@as(u32, 1) << @intCast(mant_bits)) - 1);

    const sign_stream: types.Stream = .{ .data = sign_buf, .count = @intCast(n), .bits_per_elem = 1 };
    const exp_stream: types.Stream = .{ .data = exp_buf, .count = @intCast(n), .bits_per_elem = exp_bits };
    const mant_stream: types.Stream = .{ .data = mant_buf, .count = @intCast(n), .bits_per_elem = mant_bits };

    // Vectorized fast path for the common case: bf16 (mant_bytes=1, exp_bytes=1)
    // and fp16 with mant_bytes=2, exp_bytes=1. Uses Zig @Vector(8, u16) which
    // lowers to ARM NEON (16-byte SIMD) or x86 SSE2/AVX2 — 8 elements per
    // iteration vs 1 in the scalar path.
    var i: usize = 0;
    if (exp_bytes_per == 1) {
        const V8 = @Vector(8, u16);
        const V8u8 = @Vector(8, u8);
        const V8u4 = @Vector(8, u4);
        const mant_v: V8 = @splat(mant_mask);
        const exp_v: V8 = @splat(exp_mask);
        const mant_shift: V8u4 = @splat(@intCast(mant_bits));
        const sign_shift: V8u4 = @splat(15);
        const block_n: usize = n - (n % 8);
        if (mant_bytes_per == 1) {
            // bf16 path: mantissa packs into one byte/elem.
            while (i < block_n) : (i += 8) {
                const src_ptr: *const [16]u8 = @ptrCast(t.data[i * 2 ..][0..16]);
                const raw: V8 = @bitCast(src_ptr.*);
                const sign_v: V8u8 = @truncate(raw >> sign_shift);
                const exp_out: V8u8 = @truncate((raw >> mant_shift) & exp_v);
                const mant_out: V8u8 = @truncate(raw & mant_v);
                @as(*[8]u8, @ptrCast(sign_buf[i..][0..8])).* = sign_v;
                @as(*[8]u8, @ptrCast(exp_buf[i..][0..8])).* = exp_out;
                @as(*[8]u8, @ptrCast(mant_buf[i..][0..8])).* = mant_out;
            }
        } else {
            // fp16 path: mantissa needs 2 bytes/elem (10 bits).
            while (i < block_n) : (i += 8) {
                const src_ptr: *const [16]u8 = @ptrCast(t.data[i * 2 ..][0..16]);
                const raw: V8 = @bitCast(src_ptr.*);
                const sign_v: V8u8 = @truncate(raw >> sign_shift);
                const exp_out: V8u8 = @truncate((raw >> mant_shift) & exp_v);
                const mant_out: V8 = raw & mant_v;
                @as(*[8]u8, @ptrCast(sign_buf[i..][0..8])).* = sign_v;
                @as(*[8]u8, @ptrCast(exp_buf[i..][0..8])).* = exp_out;
                @as(*[16]u8, @ptrCast(mant_buf[i * 2 ..][0..16])).* = @bitCast(mant_out);
            }
        }
    }
    // Tail (and full path for other widths) — scalar.
    while (i < n) : (i += 1) {
        const raw: u16 = std.mem.readInt(u16, t.data[i * 2 ..][0..2], .little);
        sign_stream.setU32(i, @intCast(raw >> 15));
        exp_stream.setU32(i, @intCast((raw >> @intCast(mant_bits)) & exp_mask));
        mant_stream.setU32(i, @intCast(raw & mant_mask));
    }

    var info: SplitFloatInfo = .{
        .dtype = t.dtype,
        .exp_bits = exp_bits,
        .mant_bits = mant_bits,
        .ndim = @intCast(t.shape.len),
        .shape = .{0} ** 8,
    };
    for (t.shape, 0..) |d, k| info.shape[k] = d;

    return .{
        .sign = sign_stream,
        .exp = exp_stream,
        .mant = mant_stream,
        .info = info,
    };
}

pub fn splitFloatInverse(
    alloc: Allocator,
    sign: Stream,
    exp: Stream,
    mant: Stream,
    info: SplitFloatInfo,
) !TensorView {
    var n: u64 = 1;
    for (0..info.ndim) |k| n *= info.shape[k];
    const out = try alloc.alloc(u8, @intCast(n * 2));

    var i: usize = 0;
    while (i < n) : (i += 1) {
        const s: u16 = @intCast(sign.getU32(i) & 1);
        const e: u16 = @intCast(exp.getU32(i) & ((@as(u32, 1) << @intCast(info.exp_bits)) - 1));
        const m: u16 = @intCast(mant.getU32(i) & ((@as(u32, 1) << @intCast(info.mant_bits)) - 1));
        const raw: u16 = (s << 15) | (e << @intCast(info.mant_bits)) | m;
        std.mem.writeInt(u16, out[i * 2 ..][0..2], raw, .little);
    }

    const shape_buf = try alloc.alloc(u64, info.ndim);
    for (0..info.ndim) |k| shape_buf[k] = info.shape[k];

    return .{
        .data = out,
        .shape = shape_buf,
        .dtype = info.dtype,
        .owns_data = true,
        .owns_shape = true,
    };
}

// ---------- bitplane_split (multi-bit stream -> N 1-bit planes) ----------
pub const BitplaneInfo = struct {
    n_planes: u8,
    count: usize,
};

pub fn bitplaneSplitForward(alloc: Allocator, s: Stream) !struct {
    planes: []Stream,
    info: BitplaneInfo,
} {
    const n_planes: u8 = s.bits_per_elem;
    const planes = try alloc.alloc(Stream, n_planes);
    for (planes, 0..) |*pl, p| {
        const buf = try alloc.alloc(u8, s.count);
        pl.* = .{ .data = buf, .count = s.count, .bits_per_elem = 1 };
        var i: usize = 0;
        while (i < s.count) : (i += 1) {
            const v = s.getU32(i);
            buf[i] = @intCast((v >> @intCast(p)) & 1);
        }
    }
    return .{ .planes = planes, .info = .{ .n_planes = n_planes, .count = s.count } };
}

pub fn bitplaneSplitInverse(
    alloc: Allocator,
    planes: []const Stream,
    info: BitplaneInfo,
    out_bits_per_elem: u8,
) !Stream {
    const elem_bytes: usize = switch (types.roundUpToPow2(out_bits_per_elem)) {
        8 => 1,
        16 => 2,
        32 => 4,
        else => unreachable,
    };
    const buf = try alloc.alloc(u8, info.count * elem_bytes);
    const s: Stream = .{ .data = buf, .count = info.count, .bits_per_elem = out_bits_per_elem };
    var i: usize = 0;
    while (i < info.count) : (i += 1) {
        var v: u32 = 0;
        for (planes, 0..) |pl, p| {
            const bit: u32 = pl.getU32(i) & 1;
            v |= bit << @intCast(p);
        }
        s.setU32(i, v);
    }
    return s;
}

// ---------- delta_encode (stream -> stream of differences) ----------
pub const DeltaInfo = struct {
    first: u32,
    count: usize,
    bits_per_elem: u8,
};

pub fn deltaEncodeForward(alloc: Allocator, s: Stream) !struct { out: Stream, info: DeltaInfo } {
    if (s.count == 0) {
        const buf = try alloc.alloc(u8, 0);
        return .{
            .out = .{ .data = buf, .count = 0, .bits_per_elem = s.bits_per_elem },
            .info = .{ .first = 0, .count = 0, .bits_per_elem = s.bits_per_elem },
        };
    }
    const elem_bytes: usize = switch (types.roundUpToPow2(s.bits_per_elem)) {
        8 => 1,
        16 => 2,
        32 => 4,
        else => unreachable,
    };
    const out_count = s.count - 1;
    const buf = try alloc.alloc(u8, out_count * elem_bytes);
    const out: Stream = .{ .data = buf, .count = out_count, .bits_per_elem = s.bits_per_elem };

    const modulus: u64 = @as(u64, 1) << @intCast(s.bits_per_elem);
    var prev: u32 = s.getU32(0);
    var i: usize = 1;
    while (i < s.count) : (i += 1) {
        const cur = s.getU32(i);
        const d: u64 = (@as(u64, cur) + modulus - @as(u64, prev)) % modulus;
        out.setU32(i - 1, @intCast(d));
        prev = cur;
    }

    return .{
        .out = out,
        .info = .{ .first = s.getU32(0), .count = s.count, .bits_per_elem = s.bits_per_elem },
    };
}

pub fn deltaEncodeInverse(alloc: Allocator, deltas: Stream, info: DeltaInfo) !Stream {
    const elem_bytes: usize = switch (types.roundUpToPow2(info.bits_per_elem)) {
        8 => 1,
        16 => 2,
        32 => 4,
        else => unreachable,
    };
    const buf = try alloc.alloc(u8, info.count * elem_bytes);
    const s: Stream = .{ .data = buf, .count = info.count, .bits_per_elem = info.bits_per_elem };
    if (info.count == 0) return s;

    const modulus: u64 = @as(u64, 1) << @intCast(info.bits_per_elem);
    s.setU32(0, info.first);
    var prev: u32 = info.first;
    var i: usize = 0;
    while (i < deltas.count) : (i += 1) {
        const d = deltas.getU32(i);
        const cur: u64 = (@as(u64, prev) + @as(u64, d)) % modulus;
        s.setU32(i + 1, @intCast(cur));
        prev = @intCast(cur);
    }
    return s;
}

// ---------- tensor_xor (cross-tensor reference: encode W ^ base) ----------
//
// XOR of bit patterns is exactly reversible and produces low-entropy output
// when two tensors are similar (e.g. fine-tuned-from-base, or per-layer
// pairs). The "residual" tensor uses the same dtype.

pub const TensorXorInfo = struct {
    dtype: Dtype,
    ndim: u8,
    shape: [8]u64,
};

pub fn tensorXorForward(alloc: Allocator, t: TensorView, base: TensorView) !struct {
    residual: TensorView,
    info: TensorXorInfo,
} {
    if (t.dtype != base.dtype) return error.DtypeMismatch;
    if (t.shape.len != base.shape.len) return error.ShapeMismatch;
    for (t.shape, base.shape) |a, b| if (a != b) return error.ShapeMismatch;

    const buf = try alloc.alloc(u8, t.data.len);
    for (t.data, base.data, 0..) |a, b, i| buf[i] = a ^ b;

    const shape_buf = try alloc.alloc(u64, t.shape.len);
    for (t.shape, 0..) |d, k| shape_buf[k] = d;

    var info: TensorXorInfo = .{
        .dtype = t.dtype,
        .ndim = @intCast(t.shape.len),
        .shape = .{0} ** 8,
    };
    for (t.shape, 0..) |d, k| info.shape[k] = d;

    return .{
        .residual = .{
            .data = buf,
            .shape = shape_buf,
            .dtype = t.dtype,
            .owns_data = true,
            .owns_shape = true,
        },
        .info = info,
    };
}

pub fn tensorXorInverse(alloc: Allocator, residual: TensorView, base: TensorView, info: TensorXorInfo) !TensorView {
    _ = info;
    const buf = try alloc.alloc(u8, residual.data.len);
    for (residual.data, base.data, 0..) |a, b, i| buf[i] = a ^ b;
    const shape_buf = try alloc.alloc(u64, residual.shape.len);
    for (residual.shape, 0..) |d, k| shape_buf[k] = d;
    return .{
        .data = buf,
        .shape = shape_buf,
        .dtype = residual.dtype,
        .owns_data = true,
        .owns_shape = true,
    };
}

// ---------- raw (terminal: stream -> bytes verbatim) ----------
pub const RawStreamInfo = struct {
    count: usize,
    bits_per_elem: u8,
};

pub fn rawForward(alloc: Allocator, s: Stream) !struct { bytes: []u8, info: RawStreamInfo } {
    const buf = try alloc.alloc(u8, s.data.len);
    @memcpy(buf, s.data);
    return .{
        .bytes = buf,
        .info = .{ .count = s.count, .bits_per_elem = s.bits_per_elem },
    };
}

pub fn rawInverse(alloc: Allocator, bytes: []const u8, info: RawStreamInfo) !Stream {
    const buf = try alloc.alloc(u8, bytes.len);
    @memcpy(buf, bytes);
    return .{ .data = buf, .count = info.count, .bits_per_elem = info.bits_per_elem };
}

// ---------- tensor_raw (terminal: tensor -> bytes verbatim) ----------
pub const TensorRawInfo = struct {
    dtype: Dtype,
    ndim: u8,
    shape: [8]u64,
};

pub fn tensorRawForward(alloc: Allocator, t: TensorView) !struct { bytes: []u8, info: TensorRawInfo } {
    const buf = try alloc.alloc(u8, t.data.len);
    @memcpy(buf, t.data);
    var info: TensorRawInfo = .{
        .dtype = t.dtype,
        .ndim = @intCast(t.shape.len),
        .shape = .{0} ** 8,
    };
    for (t.shape, 0..) |d, k| info.shape[k] = d;
    return .{ .bytes = buf, .info = info };
}

pub fn tensorRawInverse(alloc: Allocator, bytes: []const u8, info: TensorRawInfo) !TensorView {
    const buf = try alloc.alloc(u8, bytes.len);
    @memcpy(buf, bytes);
    const shape_buf = try alloc.alloc(u64, info.ndim);
    for (0..info.ndim) |k| shape_buf[k] = info.shape[k];
    return .{
        .data = buf,
        .shape = shape_buf,
        .dtype = info.dtype,
        .owns_data = true,
        .owns_shape = true,
    };
}
