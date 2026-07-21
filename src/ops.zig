//! Reversible DSL operators. `forward` fans a stream out into 1..n streams and
//! records whatever `inverse` needs in `SideInfo`. Terminals (raw/bitpack/
//! huffman/rans) are not handled here — program.zig drives codec.zig directly.

const std = @import("std");
const builtin = @import("builtin");
const types = @import("types.zig");
const codec = @import("codec.zig");

const Allocator = std.mem.Allocator;
const Stream = types.Stream;
const Dtype = types.Dtype;

pub const K_TRANSFORM_LAYERS: u8 = 2;
pub const MAX_ENTROPY_BPE: u8 = 16;
pub const MAX_NODES: usize = 12;
pub const MAX_EXPANSIONS: usize = 256;
pub const MAX_REALIZATIONS: usize = 32;
pub const SEARCH_SAMPLE_ELEMS: usize = 4096;
pub const PLAN_CANDIDATES: usize = 8;
pub const PLAN_PROBE_BLOCKS: usize = 4;
pub const UNIFORM_SCORE: u32 = 1024 * 8;

pub const OpKind = enum(u8) {
    raw = 0,
    bitpack = 1,
    huffman = 2,
    rans = 3,

    xor_const = 10,
    add_const_mod = 11,
    xor_prev = 12,
    diff_mod = 14,
    zigzag = 16,
    gray = 18,
    rotate_bits = 20,
    bit_reverse = 21,

    split_field = 30,
    topk_codebook = 31,
    rle = 32,
    deinterleave = 33,

    split_float = 40,

    bit_plane = 50,
    byte_plane = 51,

    pub fn isTerminal(self: OpKind) bool {
        return @intFromEnum(self) < 10;
    }

    pub fn isTransform(self: OpKind) bool {
        return !self.isTerminal();
    }

    /// Bijections on the symbol alphabet: the output histogram is a permutation
    /// of the input's, so zeroth-order entropy is exactly preserved and the
    /// child's bound equals the parent's. Context-using ops are not in this set.
    pub fn isAlphabetPermutation(self: OpKind) bool {
        return switch (self) {
            .xor_const, .add_const_mod, .zigzag, .gray, .rotate_bits, .bit_reverse => true,
            else => false,
        };
    }
};

pub fn opMask(op: OpKind) u64 {
    return @as(u64, 1) << @as(u6, @intCast(@intFromEnum(op)));
}

pub const ALL_OPS_MASK: u64 = blk: {
    var mask: u64 = 0;
    for (@typeInfo(OpKind).@"enum".fields) |field| mask |= @as(u64, 1) << @intCast(field.value);
    break :blk mask;
};

/// Number of child streams. Depends on input width for the bit/byte planes.
pub fn arity(op: OpKind, in_bpe: u8) usize {
    return switch (op) {
        .raw, .bitpack, .huffman, .rans => 0,
        .xor_const,
        .add_const_mod,
        .xor_prev,
        .diff_mod,
        .zigzag,
        .gray,
        .rotate_bits,
        .bit_reverse,
        => 1,
        .split_field, .topk_codebook, .rle, .deinterleave => 2,
        .split_float => 3,
        .bit_plane => in_bpe,
        .byte_plane => (@as(usize, in_bpe) + 7) / 8,
    };
}

pub const SideInfo = union(enum) {
    none,
    terminal: struct { count: usize, bits_per_elem: u8 },
    huffman: struct { table: codec.HuffmanTable, count: usize, bits_per_elem: u8 },
    rans: struct { table: codec.RansTable, count: usize, bits_per_elem: u8 },
    codebook: struct { syms: []u32, count: usize, bits_per_elem: u8 },
    rle: struct { count: usize, bits_per_elem: u8 },
    split: struct { count: usize, bits_per_elem: u8 },
    sfloat: struct { dtype: Dtype, count: usize },

    pub fn deinit(self: *SideInfo, alloc: Allocator) void {
        switch (self.*) {
            .huffman => |*h| h.table.deinit(alloc),
            .rans => |*r| r.table.deinit(alloc),
            .codebook => |*c| alloc.free(c.syms),
            else => {},
        }
        self.* = .none;
    }

    pub fn clone(self: SideInfo, alloc: Allocator) !SideInfo {
        return switch (self) {
            .huffman => |h| .{ .huffman = .{
                .table = try h.table.clone(alloc),
                .count = h.count,
                .bits_per_elem = h.bits_per_elem,
            } },
            .rans => |r| .{ .rans = .{
                .table = try r.table.clone(alloc),
                .count = r.count,
                .bits_per_elem = r.bits_per_elem,
            } },
            .codebook => |c| .{ .codebook = .{
                .syms = try alloc.dupe(u32, c.syms),
                .count = c.count,
                .bits_per_elem = c.bits_per_elem,
            } },
            else => self,
        };
    }
};

// ==================== forward / inverse ====================

pub fn forward(
    alloc: Allocator,
    op: OpKind,
    params: u32,
    in: Stream,
    out: *std.ArrayList(Stream),
    side: *SideInfo,
) !void {
    std.debug.assert(op.isTransform());
    side.* = .none;
    const c = ctxOf(in, params);

    switch (op) {
        .xor_const => try out.append(alloc, try mapElems(alloc, in, c, fXor)),
        .add_const_mod => try out.append(alloc, try mapElems(alloc, in, c, fAdd)),
        .zigzag => try out.append(alloc, try mapElems(alloc, in, c, fZigzag)),
        .gray => try out.append(alloc, try mapElems(alloc, in, c, fGray)),
        .bit_reverse => try out.append(alloc, try mapElems(alloc, in, c, fBitRev)),
        .rotate_bits => try out.append(alloc, try mapElems(alloc, in, rotCtx(in, params, false), fRotl)),

        .xor_prev => try out.append(alloc, try xorPrev(alloc, in)),
        .diff_mod => try out.append(alloc, try diffMod(alloc, in)),

        .split_field => {
            try splitFieldFwd(alloc, in, params, out);
            side.* = .{ .split = .{ .count = in.count, .bits_per_elem = in.bits_per_elem } };
        },
        .deinterleave => {
            try deinterleaveFwd(alloc, in, params, out);
            side.* = .{ .split = .{ .count = in.count, .bits_per_elem = in.bits_per_elem } };
        },
        .bit_plane => {
            try bitPlaneFwd(alloc, in, out);
            side.* = .{ .split = .{ .count = in.count, .bits_per_elem = in.bits_per_elem } };
        },
        .byte_plane => {
            try bytePlaneFwd(alloc, in, out);
            side.* = .{ .split = .{ .count = in.count, .bits_per_elem = in.bits_per_elem } };
        },
        .rle => {
            try rleFwd(alloc, in, out);
            side.* = .{ .rle = .{ .count = in.count, .bits_per_elem = in.bits_per_elem } };
        },
        .topk_codebook => {
            const syms = try topkFwd(alloc, in, @intCast(params & 0xFF), out);
            side.* = .{ .codebook = .{
                .syms = syms,
                .count = in.count,
                .bits_per_elem = in.bits_per_elem,
            } };
        },
        .split_float => {
            const dt: Dtype = @enumFromInt(params & 0xFF);
            try splitFloatFwd(alloc, in, dt, out);
            side.* = .{ .sfloat = .{ .dtype = dt, .count = in.count } };
        },

        .raw, .bitpack, .huffman, .rans => unreachable,
    }
}

pub fn inverse(
    alloc: Allocator,
    op: OpKind,
    params: u32,
    ins: []const Stream,
    side: SideInfo,
) !Stream {
    std.debug.assert(op.isTransform());
    std.debug.assert(ins.len >= 1);
    const in = ins[0];
    const c = ctxOf(in, params);

    return switch (op) {
        .xor_const => try mapElems(alloc, in, c, fXor),
        .add_const_mod => try mapElems(alloc, in, c, fSub),
        .zigzag => try mapElems(alloc, in, c, fUnzigzag),
        .gray => try mapElems(alloc, in, c, fInvGray),
        .bit_reverse => try mapElems(alloc, in, c, fBitRev),
        .rotate_bits => try mapElems(alloc, in, rotCtx(in, params, true), fRotl),

        .xor_prev => try prefixXor(alloc, in),
        .diff_mod => try cumsumMod(alloc, in),

        .split_field => try splitFieldInv(alloc, ins, params, side.split.count, side.split.bits_per_elem),
        .deinterleave => try deinterleaveInv(alloc, ins, params, side.split.count, side.split.bits_per_elem),
        .bit_plane => try bitPlaneInv(alloc, ins, side.split.count, side.split.bits_per_elem),
        .byte_plane => try bytePlaneInv(alloc, ins, side.split.count, side.split.bits_per_elem),
        .rle => try rleInv(alloc, ins, side.rle.count, side.rle.bits_per_elem),
        .topk_codebook => try topkInv(alloc, ins, side.codebook),
        .split_float => try splitFloatInv(alloc, ins, side.sfloat.dtype, side.sfloat.count),

        .raw, .bitpack, .huffman, .rans => unreachable,
    };
}

// ==================== element-wise 1->1 ====================

const Ctx = struct { m: u32, k: u8, c: u32 };

fn maskOf(bpe: u8) u32 {
    return if (bpe >= 32) 0xFFFF_FFFF else (@as(u32, 1) << @intCast(bpe)) - 1;
}

fn ctxOf(s: Stream, c: u32) Ctx {
    return .{ .m = maskOf(s.bits_per_elem), .k = s.bits_per_elem, .c = c };
}

fn rotCtx(s: Stream, params: u32, invert: bool) Ctx {
    const k = s.bits_per_elem;
    const r: u32 = (params & 0xFF) % k;
    return .{ .m = maskOf(k), .k = k, .c = if (invert) (k - r) % k else r };
}

fn mapElems(alloc: Allocator, in: Stream, c: Ctx, comptime f: fn (Ctx, u32) u32) !Stream {
    var o = try Stream.init(alloc, in.count, in.bits_per_elem);
    for (0..in.count) |i| o.setU32(i, f(c, in.getU32(i)));
    return o;
}

fn fXor(c: Ctx, v: u32) u32 {
    return (v ^ c.c) & c.m;
}

fn fAdd(c: Ctx, v: u32) u32 {
    return (v +% c.c) & c.m;
}

fn fSub(c: Ctx, v: u32) u32 {
    return (v -% c.c) & c.m;
}

fn fZigzag(c: Ctx, v: u32) u32 {
    // `v >> (k-1)` is the sign bit; broadcast it so this inverts fUnzigzag.
    return ((v << 1) ^ (0 -% (v >> @intCast(c.k - 1)))) & c.m;
}

fn fUnzigzag(c: Ctx, v: u32) u32 {
    return ((v >> 1) ^ (0 -% (v & 1))) & c.m;
}

fn fGray(c: Ctx, v: u32) u32 {
    return (v ^ (v >> 1)) & c.m;
}

fn fInvGray(c: Ctx, v: u32) u32 {
    var x = v;
    var s: u8 = 1;
    while (s < c.k) : (s *= 2) x ^= x >> @intCast(s);
    return x & c.m;
}

fn fRotl(c: Ctx, v: u32) u32 {
    if (c.c == 0) return v & c.m;
    const r: u5 = @intCast(c.c);
    return ((v << r) | ((v & c.m) >> @intCast(c.k - c.c))) & c.m;
}

fn fBitRev(c: Ctx, v: u32) u32 {
    var r: u32 = 0;
    var b: u8 = 0;
    while (b < c.k) : (b += 1) r |= ((v >> @intCast(b)) & 1) << @intCast(c.k - 1 - b);
    return r;
}

// ==================== cross-element 1->1 ====================

fn xorPrev(alloc: Allocator, in: Stream) !Stream {
    var o = try Stream.init(alloc, in.count, in.bits_per_elem);
    if (in.count == 0) return o;
    o.setU32(0, in.getU32(0));
    for (1..in.count) |i| o.setU32(i, in.getU32(i) ^ in.getU32(i - 1));
    return o;
}

fn prefixXor(alloc: Allocator, in: Stream) !Stream {
    var o = try Stream.init(alloc, in.count, in.bits_per_elem);
    var acc: u32 = 0;
    for (0..in.count) |i| {
        acc ^= in.getU32(i);
        o.setU32(i, acc);
    }
    return o;
}

fn diffMod(alloc: Allocator, in: Stream) !Stream {
    var o = try Stream.init(alloc, in.count, in.bits_per_elem);
    if (in.count == 0) return o;
    const m = maskOf(in.bits_per_elem);
    o.setU32(0, in.getU32(0));
    for (1..in.count) |i| o.setU32(i, (in.getU32(i) -% in.getU32(i - 1)) & m);
    return o;
}

fn cumsumMod(alloc: Allocator, in: Stream) !Stream {
    var o = try Stream.init(alloc, in.count, in.bits_per_elem);
    const m = maskOf(in.bits_per_elem);
    var acc: u32 = 0;
    for (0..in.count) |i| {
        acc = (acc +% in.getU32(i)) & m;
        o.setU32(i, acc);
    }
    return o;
}

// ==================== split_field ====================

fn splitFieldFwd(alloc: Allocator, in: Stream, params: u32, out: *std.ArrayList(Stream)) !void {
    const start: u8 = @intCast(params & 0xFF);
    const n: u8 = @intCast((params >> 8) & 0xFF);
    const k = in.bits_per_elem;
    std.debug.assert(n > 0 and start < k and start + n <= k);
    const lo_bits = k - n;

    var hi = try Stream.init(alloc, in.count, n);
    var lo = if (lo_bits > 0)
        try Stream.init(alloc, in.count, lo_bits)
    else
        try Stream.init(alloc, 0, 1);

    const hi_mask = maskOf(n);
    const below_mask = maskOf(start);
    for (0..in.count) |i| {
        const v = in.getU32(i);
        hi.setU32(i, (v >> @intCast(start)) & hi_mask);
        if (lo_bits > 0) {
            const above: u32 = if (start + n < k) v >> @intCast(start + n) else 0;
            lo.setU32(i, (v & below_mask) | (above << @intCast(start)));
        }
    }
    try out.append(alloc, hi);
    try out.append(alloc, lo);
}

fn splitFieldInv(alloc: Allocator, ins: []const Stream, params: u32, count: usize, k: u8) !Stream {
    const start: u8 = @intCast(params & 0xFF);
    const n: u8 = @intCast((params >> 8) & 0xFF);
    std.debug.assert(n > 0 and start < k and start + n <= k);
    const lo_bits = k - n;
    const below_mask = maskOf(start);

    var o = try Stream.init(alloc, count, k);
    for (0..count) |i| {
        var v: u32 = ins[0].getU32(i) << @intCast(start);
        if (lo_bits > 0) {
            const lv = ins[1].getU32(i);
            v |= lv & below_mask;
            if (start + n < k) v |= (lv >> @intCast(start)) << @intCast(start + n);
        }
        o.setU32(i, v & maskOf(k));
    }
    return o;
}

// ==================== deinterleave ====================

fn deinterleaveFwd(alloc: Allocator, in: Stream, params: u32, out: *std.ArrayList(Stream)) !void {
    const period: usize = params & 0xFFFF;
    const take: usize = (params >> 16) & 0xFF;
    std.debug.assert(take > 0 and take < period);

    var n0: usize = 0;
    for (0..in.count) |i| {
        if (i % period < take) n0 += 1;
    }
    var a = try Stream.init(alloc, n0, in.bits_per_elem);
    var b = try Stream.init(alloc, in.count - n0, in.bits_per_elem);
    var ia: usize = 0;
    var ib: usize = 0;
    for (0..in.count) |i| {
        if (i % period < take) {
            a.setU32(ia, in.getU32(i));
            ia += 1;
        } else {
            b.setU32(ib, in.getU32(i));
            ib += 1;
        }
    }
    try out.append(alloc, a);
    try out.append(alloc, b);
}

fn deinterleaveInv(alloc: Allocator, ins: []const Stream, params: u32, count: usize, bpe: u8) !Stream {
    const period: usize = params & 0xFFFF;
    const take: usize = (params >> 16) & 0xFF;
    std.debug.assert(take > 0 and take < period);

    var o = try Stream.init(alloc, count, bpe);
    var ia: usize = 0;
    var ib: usize = 0;
    for (0..count) |i| {
        if (i % period < take) {
            o.setU32(i, ins[0].getU32(ia));
            ia += 1;
        } else {
            o.setU32(i, ins[1].getU32(ib));
            ib += 1;
        }
    }
    return o;
}

// ==================== bit / byte planes ====================

fn bitPlaneFwd(alloc: Allocator, in: Stream, out: *std.ArrayList(Stream)) !void {
    for (0..in.bits_per_elem) |p| {
        var pl = try Stream.init(alloc, in.count, 1);
        for (0..in.count) |i| pl.data[i] = @intCast((in.getU32(i) >> @intCast(p)) & 1);
        try out.append(alloc, pl);
    }
}

fn bitPlaneInv(alloc: Allocator, ins: []const Stream, count: usize, bpe: u8) !Stream {
    std.debug.assert(ins.len == bpe);
    var o = try Stream.init(alloc, count, bpe);
    for (0..count) |i| {
        var v: u32 = 0;
        for (ins, 0..) |pl, p| v |= (pl.getU32(i) & 1) << @intCast(p);
        o.setU32(i, v);
    }
    return o;
}

fn bytePlaneFwd(alloc: Allocator, in: Stream, out: *std.ArrayList(Stream)) !void {
    const nb = (@as(usize, in.bits_per_elem) + 7) / 8;
    for (0..nb) |b| {
        var pl = try Stream.init(alloc, in.count, 8);
        for (0..in.count) |i| pl.data[i] = @truncate(in.getU32(i) >> @intCast(b * 8));
        try out.append(alloc, pl);
    }
}

fn bytePlaneInv(alloc: Allocator, ins: []const Stream, count: usize, bpe: u8) !Stream {
    std.debug.assert(ins.len == (@as(usize, bpe) + 7) / 8);
    var o = try Stream.init(alloc, count, bpe);
    const m = maskOf(bpe);
    var i: usize = 0;

    if (bpe == 8) {
        @memcpy(o.data, ins[0].data[0..count]);
        return o;
    }

    if (bpe > 8 and comptime builtin.cpu.arch.endian() == .little) {
        if (bpe <= 16) {
            const V = @Vector(8, u16);
            const Vb = @Vector(8, u8);
            const Vs = @Vector(8, u4);
            const mask: V = @splat(@intCast(m));
            const sh: Vs = @splat(8);
            const blocked = count - count % 8;
            while (i < blocked) : (i += 8) {
                const lo: V = @intCast(@as(Vb, @bitCast(ins[0].data[i..][0..8].*)));
                const hi: V = @intCast(@as(Vb, @bitCast(ins[1].data[i..][0..8].*)));
                o.data[i * 2 ..][0..16].* = @bitCast((lo | (hi << sh)) & mask);
            }
        } else {
            const V = @Vector(4, u32);
            const Vb = @Vector(4, u8);
            const Vs = @Vector(4, u5);
            const mask: V = @splat(m);
            const sh8: Vs = @splat(8);
            const sh16: Vs = @splat(16);
            const sh24: Vs = @splat(24);
            const blocked = count - count % 4;
            while (i < blocked) : (i += 4) {
                const b0: V = @intCast(@as(Vb, @bitCast(ins[0].data[i..][0..4].*)));
                const b1: V = @intCast(@as(Vb, @bitCast(ins[1].data[i..][0..4].*)));
                const b2: V = @intCast(@as(Vb, @bitCast(ins[2].data[i..][0..4].*)));
                const b3: V = if (ins.len == 4)
                    @intCast(@as(Vb, @bitCast(ins[3].data[i..][0..4].*)))
                else
                    @splat(0);
                const raw = b0 | (b1 << sh8) | (b2 << sh16) | (b3 << sh24);
                o.data[i * 4 ..][0..16].* = @bitCast(raw & mask);
            }
        }
    }

    while (i < count) : (i += 1) {
        var v: u32 = 0;
        for (ins, 0..) |pl, b| v |= (pl.getU32(i) & 0xFF) << @intCast(b * 8);
        o.setU32(i, v & m);
    }
    return o;
}

// ==================== rle ====================

fn rleFwd(alloc: Allocator, in: Stream, out: *std.ArrayList(Stream)) !void {
    var runs: usize = 0;
    var i: usize = 0;
    while (i < in.count) {
        i += runLen(in, i);
        runs += 1;
    }

    var vals = try Stream.init(alloc, runs, in.bits_per_elem);
    var lens = try Stream.init(alloc, runs, 8);
    i = 0;
    var j: usize = 0;
    while (i < in.count) : (j += 1) {
        const l = runLen(in, i);
        vals.setU32(j, in.getU32(i));
        lens.data[j] = @intCast(l);
        i += l;
    }
    try out.append(alloc, vals);
    try out.append(alloc, lens);
}

fn runLen(in: Stream, start: usize) usize {
    const v = in.getU32(start);
    var l: usize = 1;
    while (start + l < in.count and l < 255 and in.getU32(start + l) == v) l += 1;
    return l;
}

fn rleInv(alloc: Allocator, ins: []const Stream, count: usize, bpe: u8) !Stream {
    var o = try Stream.init(alloc, count, bpe);
    var i: usize = 0;
    for (0..ins[0].count) |j| {
        const v = ins[0].getU32(j);
        const l = ins[1].getU32(j);
        for (0..l) |_| {
            o.setU32(i, v);
            i += 1;
        }
    }
    std.debug.assert(i == count);
    return o;
}

// ==================== topk_codebook ====================

const SymFreq = struct { sym: u32, n: u64 };

fn freqDesc(_: void, a: SymFreq, b: SymFreq) bool {
    if (a.n != b.n) return a.n > b.n;
    return a.sym < b.sym;
}

fn topkFwd(alloc: Allocator, in: Stream, k_top: u8, out: *std.ArrayList(Stream)) ![]u32 {
    std.debug.assert(k_top > 0);

    var freq: std.AutoHashMapUnmanaged(u32, u64) = .empty;
    defer freq.deinit(alloc);
    for (0..in.count) |i| {
        const gop = try freq.getOrPut(alloc, in.getU32(i));
        if (!gop.found_existing) gop.value_ptr.* = 0;
        gop.value_ptr.* += 1;
    }

    const pairs = try alloc.alloc(SymFreq, freq.count());
    defer alloc.free(pairs);
    var j: usize = 0;
    var it = freq.iterator();
    while (it.next()) |kv| : (j += 1) pairs[j] = .{ .sym = kv.key_ptr.*, .n = kv.value_ptr.* };
    std.mem.sort(SymFreq, pairs, {}, freqDesc);

    const n = @min(@as(usize, k_top), pairs.len);
    const syms = try alloc.alloc(u32, n);
    errdefer alloc.free(syms);
    for (0..n) |i| syms[i] = pairs[i].sym;
    std.mem.sort(u32, syms, {}, std.sort.asc(u32));

    var index: std.AutoHashMapUnmanaged(u32, u32) = .empty;
    defer index.deinit(alloc);
    for (syms, 0..) |s, i| try index.put(alloc, s, @intCast(i));

    var n_esc: usize = 0;
    for (0..in.count) |i| {
        if (!index.contains(in.getU32(i))) n_esc += 1;
    }

    var idx = try Stream.init(alloc, in.count, indexBits(n));
    var esc = try Stream.init(alloc, n_esc, in.bits_per_elem);
    var e: usize = 0;
    for (0..in.count) |i| {
        const v = in.getU32(i);
        if (index.get(v)) |ix| {
            idx.setU32(i, ix);
        } else {
            idx.setU32(i, @intCast(n));
            esc.setU32(e, v);
            e += 1;
        }
    }
    try out.append(alloc, idx);
    try out.append(alloc, esc);
    return syms;
}

/// Bits needed to hold values 0..n inclusive (n == the escape index).
fn indexBits(n: usize) u8 {
    if (n == 0) return 1;
    return @intCast(32 - @clz(@as(u32, @intCast(n))));
}

fn topkInv(alloc: Allocator, ins: []const Stream, cb: anytype) !Stream {
    var o = try Stream.init(alloc, cb.count, cb.bits_per_elem);
    const esc_idx: u32 = @intCast(cb.syms.len);
    var e: usize = 0;
    for (0..cb.count) |i| {
        const ix = ins[0].getU32(i);
        if (ix == esc_idx) {
            o.setU32(i, ins[1].getU32(e));
            e += 1;
        } else {
            o.setU32(i, cb.syms[ix]);
        }
    }
    return o;
}

// ==================== split_float ====================

fn splitFloatFwd(alloc: Allocator, in: Stream, dt: Dtype, out: *std.ArrayList(Stream)) !void {
    const f = dt.floatFields().?;
    std.debug.assert(in.bits_per_elem == f.total);

    var sign = try Stream.init(alloc, in.count, 1);
    var exp = try Stream.init(alloc, in.count, f.exp);
    var mant = try Stream.init(alloc, in.count, f.mant);

    const emask = maskOf(f.exp);
    const mmask = maskOf(f.mant);
    var i: usize = 0;

    if (f.total == 16 and comptime builtin.cpu.arch.endian() == .little) {
        const V = @Vector(8, u16);
        const Vb = @Vector(8, u8);
        const Vs = @Vector(8, u4);
        const ev: V = @splat(@intCast(emask));
        const mv: V = @splat(@intCast(mmask));
        const msh: Vs = @splat(@intCast(f.mant));
        const ssh: Vs = @splat(15);
        const wide_mant = mant.elemBytes() == 2;
        const blocked = in.count - in.count % 8;
        while (i < blocked) : (i += 8) {
            const raw: V = @bitCast(in.data[i * 2 ..][0..16].*);
            const s: Vb = @truncate(raw >> ssh);
            const e: Vb = @truncate((raw >> msh) & ev);
            const m: V = raw & mv;
            sign.data[i..][0..8].* = s;
            exp.data[i..][0..8].* = e;
            if (wide_mant) {
                mant.data[i * 2 ..][0..16].* = @bitCast(m);
            } else {
                const mb: Vb = @truncate(m);
                mant.data[i..][0..8].* = mb;
            }
        }
    }

    while (i < in.count) : (i += 1) {
        const v = in.getU32(i);
        sign.setU32(i, v >> @intCast(f.total - 1));
        exp.setU32(i, (v >> @intCast(f.mant)) & emask);
        mant.setU32(i, v & mmask);
    }

    try out.append(alloc, sign);
    try out.append(alloc, exp);
    try out.append(alloc, mant);
}

fn splitFloatInv(alloc: Allocator, ins: []const Stream, dt: Dtype, count: usize) !Stream {
    const f = dt.floatFields().?;
    var o = try Stream.init(alloc, count, f.total);
    const emask = maskOf(f.exp);
    const mmask = maskOf(f.mant);
    var i: usize = 0;

    if (comptime builtin.cpu.arch.endian() == .little) {
        if (f.total == 16) {
            const V = @Vector(8, u16);
            const Vb = @Vector(8, u8);
            const Vs = @Vector(8, u4);
            const esh: Vs = @splat(@intCast(f.mant));
            const ssh: Vs = @splat(15);
            const blocked = count - count % 8;
            while (i < blocked) : (i += 8) {
                const s: V = @intCast(@as(Vb, @bitCast(ins[0].data[i..][0..8].*)));
                const e: V = @intCast(@as(Vb, @bitCast(ins[1].data[i..][0..8].*)));
                const m: V = if (ins[2].elemBytes() == 2)
                    @bitCast(ins[2].data[i * 2 ..][0..16].*)
                else
                    @intCast(@as(Vb, @bitCast(ins[2].data[i..][0..8].*)));
                const raw = (s << ssh) | (e << esh) | m;
                o.data[i * 2 ..][0..16].* = @bitCast(raw);
            }
        } else if (f.total == 32) {
            const V = @Vector(4, u32);
            const Vb = @Vector(4, u8);
            const Vs = @Vector(4, u5);
            const esh: Vs = @splat(@intCast(f.mant));
            const ssh: Vs = @splat(31);
            const blocked = count - count % 4;
            while (i < blocked) : (i += 4) {
                const s: V = @intCast(@as(Vb, @bitCast(ins[0].data[i..][0..4].*)));
                const e: V = @intCast(@as(Vb, @bitCast(ins[1].data[i..][0..4].*)));
                const m: V = @bitCast(ins[2].data[i * 4 ..][0..16].*);
                const raw = (s << ssh) | (e << esh) | m;
                o.data[i * 4 ..][0..16].* = @bitCast(raw);
            }
        }
    }

    while (i < count) : (i += 1) {
        const s = ins[0].getU32(i) & 1;
        const e = ins[1].getU32(i) & emask;
        const m = ins[2].getU32(i) & mmask;
        o.setU32(i, (s << @intCast(f.total - 1)) | (e << @intCast(f.mant)) | m);
    }
    return o;
}
