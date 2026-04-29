//! Low-level reversible primitive library + execution VM.
//!
//! Each primitive is by-construction reversible:
//!   * `xor_const(c, k)`: s'[i] = s[i] ⊕ c          (self-inverse)
//!   * `add_const_mod(c, k)`: s'[i] = (s[i] + c) mod 2^k
//!   * `rotate_bits(r, k)`: s'[i] = rotl_k(s[i], r)
//!   * `bit_permute_pair(i, j, k)`: swap bit i and bit j of each elem
//!   * `xor_prev`: s'[i] = s[i] ⊕ s[i-1] (s'[0] = s[0])  ←→ `prefix_xor`
//!   * `diff_mod(k)`: s'[i] = (s[i] − s[i-1]) mod 2^k    ←→ `cumsum_mod(k)`
//!   * `split_field(start, n_bits, k)`: u_k stream → (u_n, u_{k-n}) streams
//!   * terminals: `huffman`, `rans`, `raw`
//!
//! Notes
//! =====
//! * The "k" parameter is the source stream's logical bits-per-elem. The op
//!   behaves modulo 2^k.
//! * `split_field` is the only op that fans out to multiple streams; it is the
//!   primitive on top of which split_float / bitplane_split can emerge.
//! * `xor_prev` / `prefix_xor` are inverse pairs and both included so the
//!   trainer can reach optimum without inferring inverses.

const std = @import("std");
const types = @import("types.zig");
const codec = @import("codec.zig");

const Allocator = types.Allocator;
const Stream = types.Stream;

pub const OpKind = enum(u8) {
    // Element-wise 1→1 (single stream → single stream)
    xor_const = 0,
    add_const_mod = 1,
    rotate_bits = 2,
    bit_swap_pair = 3,
    bit_reverse = 4, // reverse all k bits within each element
    negate_mod = 5, // x → (−x) mod 2^k
    mul_const_odd_mod = 6, // x → (c·x) mod 2^k, c odd → invertible
    xor_with_shift = 7, // x → x ⊕ (x >> s) ; inverse = same with iterated shifts
    gray_code = 8, // x → x ⊕ (x >> 1)
    inv_gray_code = 9, // inverse of gray_code

    // Cross-element 1→1 (uses prev element)
    xor_prev = 10,
    prefix_xor = 11,
    diff_mod = 12,
    cumsum_mod = 13,

    // Stream restructuring (1 → 2)
    split_field = 14,

    // Terminals
    huffman = 16,
    rans = 17,
    raw = 18,

    pub fn isTerminal(self: OpKind) bool {
        return switch (self) {
            .huffman, .rans, .raw => true,
            else => false,
        };
    }

    /// How many output streams this op produces (1 unless split_field, which is 2).
    pub fn arity(self: OpKind) u8 {
        return switch (self) {
            .split_field => 2,
            .huffman, .rans, .raw => 0, // terminals don't produce more streams
            else => 1,
        };
    }
};

pub const Params = packed struct {
    /// Generic 32-bit parameter slot. Each op interprets it differently.
    /// xor_const: low bits = c
    /// add_const_mod: low bits = c
    /// rotate_bits: low 5 bits = r, next 8 bits = k
    /// bit_swap_pair: low 5 = i, next 5 = j, next 8 = k
    /// diff_mod / cumsum_mod: low 8 = k
    /// split_field: low 8 = start, next 8 = n_bits, next 8 = k
    raw: u32 = 0,
};

pub const LowOp = struct {
    kind: OpKind,
    params: Params = .{},
};

// =================== forward / inverse on a single Stream ===================

/// Apply `op` to `input`, producing 1 or 2 output streams (split_field). The
/// caller owns the returned slices.
pub fn forward(alloc: Allocator, op: LowOp, input: Stream) !ForwardResult {
    return switch (op.kind) {
        .xor_const => .{ .one = try fwdXorConst(alloc, op.params, input) },
        .add_const_mod => .{ .one = try fwdAddConstMod(alloc, op.params, input) },
        .rotate_bits => .{ .one = try fwdRotateBits(alloc, op.params, input) },
        .bit_swap_pair => .{ .one = try fwdBitSwapPair(alloc, op.params, input) },
        .bit_reverse => .{ .one = try fwdBitReverse(alloc, input) },
        .negate_mod => .{ .one = try fwdNegateMod(alloc, input) },
        .mul_const_odd_mod => .{ .one = try fwdMulConstOddMod(alloc, op.params, input) },
        .xor_with_shift => .{ .one = try fwdXorWithShift(alloc, op.params, input) },
        .gray_code => .{ .one = try fwdGrayCode(alloc, input) },
        .inv_gray_code => .{ .one = try fwdInvGrayCode(alloc, input) },
        .xor_prev => .{ .one = try fwdXorPrev(alloc, input) },
        .prefix_xor => .{ .one = try fwdPrefixXor(alloc, input) },
        .diff_mod => .{ .one = try fwdDiffMod(alloc, op.params, input) },
        .cumsum_mod => .{ .one = try fwdCumsumMod(alloc, op.params, input) },
        .split_field => blk: {
            const r = try fwdSplitField(alloc, op.params, input);
            break :blk .{ .two = .{ r.hi, r.lo } };
        },
        else => error.NotImplemented,
    };
}

pub const ForwardResult = union(enum) {
    one: Stream,
    two: [2]Stream, // for split_field: [hi_field, lo_field]
};

/// Inverse: given the output(s) of `op`, reconstruct the input. The caller
/// passes the same output type variant that `forward` returned.
pub fn inverseOne(alloc: Allocator, op: LowOp, out: Stream) !Stream {
    return switch (op.kind) {
        .xor_const => fwdXorConst(alloc, op.params, out), // self-inverse
        .add_const_mod => invAddConstMod(alloc, op.params, out),
        .rotate_bits => invRotateBits(alloc, op.params, out),
        .bit_swap_pair => fwdBitSwapPair(alloc, op.params, out), // self-inverse
        .bit_reverse => fwdBitReverse(alloc, out), // self-inverse
        .negate_mod => fwdNegateMod(alloc, out), // self-inverse
        .mul_const_odd_mod => invMulConstOddMod(alloc, op.params, out),
        .xor_with_shift => invXorWithShift(alloc, op.params, out),
        .gray_code => fwdInvGrayCode(alloc, out),
        .inv_gray_code => fwdGrayCode(alloc, out),
        .xor_prev => fwdPrefixXor(alloc, out),
        .prefix_xor => fwdXorPrev(alloc, out),
        .diff_mod => invDiffMod(alloc, op.params, out),
        .cumsum_mod => invCumsumMod(alloc, op.params, out),
        else => error.NotImplemented,
    };
}

pub fn inverseTwo(alloc: Allocator, op: LowOp, hi: Stream, lo: Stream) !Stream {
    if (op.kind != .split_field) return error.NotImplemented;
    return invSplitField(alloc, op.params, hi, lo);
}

// =================== implementations ===================

fn fwdXorConst(alloc: Allocator, p: Params, s: Stream) !Stream {
    const c: u32 = p.raw;
    const out = try alloc.alloc(u8, s.data.len);
    const out_stream: Stream = .{ .data = out, .count = s.count, .bits_per_elem = s.bits_per_elem };
    var i: usize = 0;
    while (i < s.count) : (i += 1) {
        out_stream.setU32(i, s.getU32(i) ^ c);
    }
    return out_stream;
}

fn fwdAddConstMod(alloc: Allocator, p: Params, s: Stream) !Stream {
    const c: u32 = p.raw;
    const k_pow2: u32 = if (s.bits_per_elem >= 32) 0 else (@as(u32, 1) << @intCast(s.bits_per_elem));
    const out = try alloc.alloc(u8, s.data.len);
    const out_stream: Stream = .{ .data = out, .count = s.count, .bits_per_elem = s.bits_per_elem };
    var i: usize = 0;
    while (i < s.count) : (i += 1) {
        const v = s.getU32(i);
        const r = if (k_pow2 == 0) v +% c else (v + c) % k_pow2;
        out_stream.setU32(i, r);
    }
    return out_stream;
}

fn invAddConstMod(alloc: Allocator, p: Params, s: Stream) !Stream {
    const c: u32 = p.raw;
    const k_pow2: u32 = if (s.bits_per_elem >= 32) 0 else (@as(u32, 1) << @intCast(s.bits_per_elem));
    const out = try alloc.alloc(u8, s.data.len);
    const out_stream: Stream = .{ .data = out, .count = s.count, .bits_per_elem = s.bits_per_elem };
    var i: usize = 0;
    while (i < s.count) : (i += 1) {
        const v = s.getU32(i);
        const r = if (k_pow2 == 0) v -% c else (v + k_pow2 - (c % k_pow2)) % k_pow2;
        out_stream.setU32(i, r);
    }
    return out_stream;
}

fn fwdRotateBits(alloc: Allocator, p: Params, s: Stream) !Stream {
    const r_bits: u5 = @intCast(p.raw & 0x1F);
    const k: u8 = s.bits_per_elem;
    const mask: u32 = if (k >= 32) 0xFFFFFFFF else (@as(u32, 1) << @intCast(k)) - 1;
    const out = try alloc.alloc(u8, s.data.len);
    const out_stream: Stream = .{ .data = out, .count = s.count, .bits_per_elem = s.bits_per_elem };
    var i: usize = 0;
    while (i < s.count) : (i += 1) {
        const v = s.getU32(i) & mask;
        const r_eff: u5 = @intCast(@as(u8, r_bits) % @max(k, 1));
        const k_minus_r: u5 = @intCast((@as(u32, k) - @as(u32, r_eff)) & 0x1F);
        const rotated = ((v << r_eff) | (v >> k_minus_r)) & mask;
        out_stream.setU32(i, rotated);
    }
    return out_stream;
}

fn invRotateBits(alloc: Allocator, p: Params, s: Stream) !Stream {
    const r_bits: u5 = @intCast(p.raw & 0x1F);
    const k: u8 = s.bits_per_elem;
    const inv_r: u32 = (@as(u32, k) - @as(u32, r_bits) % @max(k, 1)) % @max(k, 1);
    const new_p: Params = .{ .raw = inv_r & 0x1F };
    return fwdRotateBits(alloc, new_p, s);
}

fn fwdBitSwapPair(alloc: Allocator, p: Params, s: Stream) !Stream {
    const i_pos: u5 = @intCast(p.raw & 0x1F);
    const j_pos: u5 = @intCast((p.raw >> 5) & 0x1F);
    const out = try alloc.alloc(u8, s.data.len);
    const out_stream: Stream = .{ .data = out, .count = s.count, .bits_per_elem = s.bits_per_elem };
    var i: usize = 0;
    while (i < s.count) : (i += 1) {
        const v = s.getU32(i);
        const bi = (v >> i_pos) & 1;
        const bj = (v >> j_pos) & 1;
        var r = v & ~((@as(u32, 1) << i_pos) | (@as(u32, 1) << j_pos));
        r |= bi << j_pos;
        r |= bj << i_pos;
        out_stream.setU32(i, r);
    }
    return out_stream;
}

fn fwdBitReverse(alloc: Allocator, s: Stream) !Stream {
    const k: u8 = s.bits_per_elem;
    const out = try alloc.alloc(u8, s.data.len);
    const out_stream: Stream = .{ .data = out, .count = s.count, .bits_per_elem = s.bits_per_elem };
    var i: usize = 0;
    while (i < s.count) : (i += 1) {
        const v = s.getU32(i);
        var rev: u32 = 0;
        var b: u8 = 0;
        while (b < k) : (b += 1) {
            rev |= ((v >> @intCast(b)) & 1) << @intCast(k - 1 - b);
        }
        out_stream.setU32(i, rev);
    }
    return out_stream;
}

fn fwdNegateMod(alloc: Allocator, s: Stream) !Stream {
    const k_pow2: u64 = if (s.bits_per_elem >= 32) 0 else (@as(u64, 1) << @intCast(s.bits_per_elem));
    const out = try alloc.alloc(u8, s.data.len);
    const out_stream: Stream = .{ .data = out, .count = s.count, .bits_per_elem = s.bits_per_elem };
    var i: usize = 0;
    while (i < s.count) : (i += 1) {
        const v = s.getU32(i);
        const r: u32 = if (k_pow2 == 0) 0 -% v else @intCast((k_pow2 - @as(u64, v)) % k_pow2);
        out_stream.setU32(i, r);
    }
    return out_stream;
}

fn fwdMulConstOddMod(alloc: Allocator, p: Params, s: Stream) !Stream {
    const c: u32 = p.raw | 1; // force odd
    const k_pow2: u64 = if (s.bits_per_elem >= 32) 0 else (@as(u64, 1) << @intCast(s.bits_per_elem));
    const mask: u32 = if (s.bits_per_elem >= 32) 0xFFFFFFFF else (@as(u32, 1) << @intCast(s.bits_per_elem)) - 1;
    const out = try alloc.alloc(u8, s.data.len);
    const out_stream: Stream = .{ .data = out, .count = s.count, .bits_per_elem = s.bits_per_elem };
    var i: usize = 0;
    while (i < s.count) : (i += 1) {
        const v = s.getU32(i) & mask;
        const r: u32 = if (k_pow2 == 0) v *% c else @intCast((@as(u64, v) *% @as(u64, c)) & ((@as(u64, 1) << @intCast(s.bits_per_elem)) - 1));
        out_stream.setU32(i, r);
    }
    return out_stream;
}

fn invMulConstOddMod(alloc: Allocator, p: Params, s: Stream) !Stream {
    // Multiplicative inverse mod 2^k of an odd c, via Newton's iteration.
    // Starting guess: c itself (good for k ≤ 3); iterate inv = inv * (2 − c*inv) mod 2^k.
    const c: u32 = p.raw | 1;
    const k: u8 = s.bits_per_elem;
    const mask: u64 = if (k >= 64) 0xFFFFFFFFFFFFFFFF else (@as(u64, 1) << @intCast(k)) - 1;
    var inv: u64 = c; // accurate to 3 bits
    var iter: u8 = 0;
    while (iter < 6) : (iter += 1) { // 6 iterations gives ≥ 96 bits of accuracy
        inv = (inv *% (2 -% (@as(u64, c) *% inv))) & mask;
    }
    return fwdMulConstOddMod(alloc, .{ .raw = @intCast(inv & 0xFFFFFFFF) }, s);
}

fn fwdXorWithShift(alloc: Allocator, p: Params, s: Stream) !Stream {
    const shift: u5 = @intCast(p.raw & 0x1F);
    const out = try alloc.alloc(u8, s.data.len);
    const out_stream: Stream = .{ .data = out, .count = s.count, .bits_per_elem = s.bits_per_elem };
    var i: usize = 0;
    while (i < s.count) : (i += 1) {
        const v = s.getU32(i);
        out_stream.setU32(i, v ^ (v >> shift));
    }
    return out_stream;
}

fn invXorWithShift(alloc: Allocator, p: Params, s: Stream) !Stream {
    // Inverse of x → x ⊕ (x >> s) on a k-bit value: apply iteratively
    //   y, y ⊕ (y >> 2s), y ⊕ (y >> 2s) ⊕ (y >> 4s), ... until s exceeds k.
    const shift: u5 = @intCast(p.raw & 0x1F);
    const k: u8 = s.bits_per_elem;
    const out = try alloc.alloc(u8, s.data.len);
    const out_stream: Stream = .{ .data = out, .count = s.count, .bits_per_elem = s.bits_per_elem };
    var i: usize = 0;
    while (i < s.count) : (i += 1) {
        var v = s.getU32(i);
        var current_shift: u8 = shift;
        while (current_shift < k) {
            v ^= v >> @intCast(current_shift);
            current_shift *= 2;
        }
        out_stream.setU32(i, v);
    }
    return out_stream;
}

fn fwdGrayCode(alloc: Allocator, s: Stream) !Stream {
    const out = try alloc.alloc(u8, s.data.len);
    const out_stream: Stream = .{ .data = out, .count = s.count, .bits_per_elem = s.bits_per_elem };
    var i: usize = 0;
    while (i < s.count) : (i += 1) {
        const v = s.getU32(i);
        out_stream.setU32(i, v ^ (v >> 1));
    }
    return out_stream;
}

fn fwdInvGrayCode(alloc: Allocator, s: Stream) !Stream {
    const k: u8 = s.bits_per_elem;
    const out = try alloc.alloc(u8, s.data.len);
    const out_stream: Stream = .{ .data = out, .count = s.count, .bits_per_elem = s.bits_per_elem };
    var i: usize = 0;
    while (i < s.count) : (i += 1) {
        var v = s.getU32(i);
        var sh: u8 = 1;
        while (sh < k) : (sh *= 2) v ^= v >> @intCast(sh);
        out_stream.setU32(i, v);
    }
    return out_stream;
}

fn fwdXorPrev(alloc: Allocator, s: Stream) !Stream {
    const out = try alloc.alloc(u8, s.data.len);
    const out_stream: Stream = .{ .data = out, .count = s.count, .bits_per_elem = s.bits_per_elem };
    if (s.count == 0) return out_stream;
    out_stream.setU32(0, s.getU32(0));
    var i: usize = 1;
    while (i < s.count) : (i += 1) {
        out_stream.setU32(i, s.getU32(i) ^ s.getU32(i - 1));
    }
    return out_stream;
}

fn fwdPrefixXor(alloc: Allocator, s: Stream) !Stream {
    const out = try alloc.alloc(u8, s.data.len);
    const out_stream: Stream = .{ .data = out, .count = s.count, .bits_per_elem = s.bits_per_elem };
    if (s.count == 0) return out_stream;
    var acc: u32 = s.getU32(0);
    out_stream.setU32(0, acc);
    var i: usize = 1;
    while (i < s.count) : (i += 1) {
        acc ^= s.getU32(i);
        out_stream.setU32(i, acc);
    }
    return out_stream;
}

fn fwdDiffMod(alloc: Allocator, p: Params, s: Stream) !Stream {
    _ = p;
    const k_pow2: u64 = if (s.bits_per_elem >= 32) 0 else (@as(u64, 1) << @intCast(s.bits_per_elem));
    const out = try alloc.alloc(u8, s.data.len);
    const out_stream: Stream = .{ .data = out, .count = s.count, .bits_per_elem = s.bits_per_elem };
    if (s.count == 0) return out_stream;
    out_stream.setU32(0, s.getU32(0));
    var prev: u32 = s.getU32(0);
    var i: usize = 1;
    while (i < s.count) : (i += 1) {
        const cur = s.getU32(i);
        const diff: u64 = if (k_pow2 == 0) (@as(u64, cur) -% @as(u64, prev)) & 0xFFFFFFFF
                          else (@as(u64, cur) + k_pow2 - @as(u64, prev)) % k_pow2;
        out_stream.setU32(i, @intCast(diff));
        prev = cur;
    }
    return out_stream;
}

fn invDiffMod(alloc: Allocator, p: Params, s: Stream) !Stream {
    _ = p;
    const k_pow2: u64 = if (s.bits_per_elem >= 32) 0 else (@as(u64, 1) << @intCast(s.bits_per_elem));
    const out = try alloc.alloc(u8, s.data.len);
    const out_stream: Stream = .{ .data = out, .count = s.count, .bits_per_elem = s.bits_per_elem };
    if (s.count == 0) return out_stream;
    var acc: u32 = s.getU32(0);
    out_stream.setU32(0, acc);
    var i: usize = 1;
    while (i < s.count) : (i += 1) {
        const d = s.getU32(i);
        acc = if (k_pow2 == 0) acc +% d else @intCast((@as(u64, acc) + @as(u64, d)) % k_pow2);
        out_stream.setU32(i, acc);
    }
    return out_stream;
}

fn fwdCumsumMod(alloc: Allocator, p: Params, s: Stream) !Stream {
    return invDiffMod(alloc, p, s);
}

fn invCumsumMod(alloc: Allocator, p: Params, s: Stream) !Stream {
    return fwdDiffMod(alloc, p, s);
}

const SplitOut = struct { hi: Stream, lo: Stream };

fn fwdSplitField(alloc: Allocator, p: Params, s: Stream) !SplitOut {
    const start: u8 = @intCast(p.raw & 0xFF);
    const n_bits: u8 = @intCast((p.raw >> 8) & 0xFF);
    const k: u8 = s.bits_per_elem;
    if (start >= k or n_bits == 0 or start + n_bits > k) return error.BadSplitParams;
    const lo_bits: u8 = k - n_bits;

    const hi_bytes_per: usize = @divExact(types.roundUpToPow2(n_bits), 8);
    const lo_bytes_per: usize = if (lo_bits == 0) 0 else @divExact(types.roundUpToPow2(lo_bits), 8);

    const hi_buf = try alloc.alloc(u8, s.count * hi_bytes_per);
    const hi_stream: Stream = .{ .data = hi_buf, .count = s.count, .bits_per_elem = n_bits };

    const lo_stream: Stream = if (lo_bits > 0) blk: {
        const lo_buf = try alloc.alloc(u8, s.count * lo_bytes_per);
        break :blk .{ .data = lo_buf, .count = s.count, .bits_per_elem = lo_bits };
    } else .{ .data = try alloc.alloc(u8, 0), .count = 0, .bits_per_elem = 1 };

    const hi_mask: u32 = if (n_bits >= 32) 0xFFFFFFFF else (@as(u32, 1) << @intCast(n_bits)) - 1;
    const lo_mask: u32 = if (lo_bits == 0) 0 else if (lo_bits >= 32) 0xFFFFFFFF else (@as(u32, 1) << @intCast(lo_bits)) - 1;

    var i: usize = 0;
    while (i < s.count) : (i += 1) {
        const v = s.getU32(i);
        // hi field: bits [start, start + n_bits)
        const hi_v = (v >> @intCast(start)) & hi_mask;
        hi_stream.setU32(i, hi_v);
        if (lo_bits > 0) {
            // lo field: bits [0, start) ++ bits [start + n_bits, k), packed back to lo_bits-wide.
            const below: u32 = if (start == 0) 0 else (v & ((@as(u32, 1) << @intCast(start)) - 1));
            const above: u32 = v >> @intCast(start + n_bits);
            const lo_v = ((above << @intCast(start)) | below) & lo_mask;
            lo_stream.setU32(i, lo_v);
        }
    }
    return .{ .hi = hi_stream, .lo = lo_stream };
}

fn invSplitField(alloc: Allocator, p: Params, hi: Stream, lo: Stream) !Stream {
    const start: u8 = @intCast(p.raw & 0xFF);
    const n_bits: u8 = @intCast((p.raw >> 8) & 0xFF);
    const k: u8 = @intCast((p.raw >> 16) & 0xFF);
    if (k == 0 or start >= k or n_bits == 0 or start + n_bits > k) return error.BadSplitParams;
    const lo_bits: u8 = k - n_bits;

    const elem_bytes: usize = @divExact(types.roundUpToPow2(k), 8);
    const buf = try alloc.alloc(u8, hi.count * elem_bytes);
    const out: Stream = .{ .data = buf, .count = hi.count, .bits_per_elem = k };

    var i: usize = 0;
    while (i < hi.count) : (i += 1) {
        const hv = hi.getU32(i);
        const lv: u32 = if (lo_bits > 0) lo.getU32(i) else 0;
        const below: u32 = if (start == 0) 0 else lv & ((@as(u32, 1) << @intCast(start)) - 1);
        const above: u32 = if (lo_bits == 0) 0 else lv >> @intCast(start);
        const v = below | (hv << @intCast(start)) | (above << @intCast(start + n_bits));
        out.setU32(i, v);
    }
    return out;
}
