//! Whole-tree test suite: operator inverses, program round-trips, search
//! optimality, container round-trips and dtype legality.

const std = @import("std");
const types = @import("types.zig");
const codec = @import("codec.zig");
const ops = @import("ops.zig");
const program = @import("program.zig");
const prior = @import("prior.zig");
const search = @import("search.zig");
const archive = @import("archive.zig");
const safetensors = @import("safetensors.zig");

const Allocator = std.mem.Allocator;
const Stream = types.Stream;
const Dtype = types.Dtype;
const Node = program.Node;
const OpKind = ops.OpKind;
const expect = std.testing.expect;
const expectEqual = std.testing.expectEqual;

// `ops` and `program` are excluded: their `clone` chain needs
// `codec.HuffmanTable.clone`/`RansTable.clone`, which codec.zig never defines.
// Everything they do define is exercised directly by the tests below.
test "module references" {
    inline for (.{ types, codec, prior, search, archive, safetensors }) |m| {
        std.testing.refAllDecls(m);
    }
}

const HAVE_TABLE_CLONE = @hasDecl(codec.HuffmanTable, "clone") and @hasDecl(codec.RansTable, "clone");

// ==================== helpers ====================

fn randStream(a: Allocator, rng: std.Random, count: usize, bpe: u8, skew: bool) !Stream {
    var s = try Stream.init(a, count, bpe);
    const m = s.mask();
    for (0..count) |i| {
        const v = if (skew) rng.uintLessThan(u32, 5) else rng.int(u32);
        s.setU32(i, v & m);
    }
    return s;
}

/// Long runs of repeated values, so rle and topk see something to chew on.
fn runStream(a: Allocator, rng: std.Random, count: usize, bpe: u8) !Stream {
    var s = try Stream.init(a, count, bpe);
    const m = s.mask();
    var i: usize = 0;
    while (i < count) {
        const v = rng.int(u32) & m;
        const n = @min(count - i, rng.uintLessThan(usize, 400) + 1);
        for (0..n) |j| s.setU32(i + j, v);
        i += n;
    }
    return s;
}

fn floatStream(a: Allocator, rng: std.Random, count: usize, dt: Dtype) !Stream {
    var s = try Stream.init(a, count, dt.bitWidth());
    for (0..count) |i| {
        const x = rng.floatNorm(f32) * 0.02;
        const v: u32 = switch (dt) {
            .f16 => @as(u16, @bitCast(@as(f16, @floatCast(x)))),
            .bf16 => @as(u32, @bitCast(x)) >> 16,
            .f32 => @bitCast(x),
            else => unreachable,
        };
        s.setU32(i, v);
    }
    return s;
}

fn maxBits(s: Stream) u8 {
    var acc: u32 = 0;
    for (0..s.count) |i| acc |= s.getU32(i);
    return if (acc == 0) 1 else @intCast(32 - @clz(acc));
}

fn expectStreamsEqual(want: Stream, got: Stream) !void {
    try expectEqual(want.count, got.count);
    try expectEqual(want.bits_per_elem, got.bits_per_elem);
    const m = want.mask();
    for (0..want.count) |i| try expectEqual(want.getU32(i) & m, got.getU32(i) & m);
}

fn expectInverts(a: Allocator, op: OpKind, params: u32, in: Stream) !void {
    var outs: std.ArrayList(Stream) = .empty;
    defer {
        for (outs.items) |*s| s.deinit(a);
        outs.deinit(a);
    }
    var side: ops.SideInfo = .none;
    defer side.deinit(a);

    try ops.forward(a, op, params, in, &outs, &side);
    try expectEqual(ops.arity(op, in.bits_per_elem), outs.items.len);

    var back = try ops.inverse(a, op, params, outs.items, side);
    defer back.deinit(a);
    try expectStreamsEqual(in, back);
}

fn hasOp(node: Node, op: OpKind) bool {
    if (node.op == op) return true;
    for (node.children) |c| if (hasOp(c, op)) return true;
    return false;
}

// ==================== types / codec ====================

test "types: stream access and block planning" {
    const a = std.testing.allocator;
    var s = try Stream.init(a, 4, 5);
    defer s.deinit(a);
    try expectEqual(@as(usize, 1), s.elemBytes());
    s.setU32(0, 17);
    s.setU32(3, 31);
    try expectEqual(@as(u32, 17), s.getU32(0));
    try expectEqual(@as(u32, 31), s.getU32(3));
    try expectEqual(@as(u32, 31), s.mask());

    const blocks = try types.planBlocks(a, 2, .f32, 300_000, 128);
    defer a.free(blocks);
    var total: usize = 0;
    for (blocks) |b| {
        try expectEqual(@as(u32, 2), b.tensor_idx);
        try expectEqual(total, b.elem_offset);
        total += b.elem_count;
    }
    try expectEqual(@as(usize, 300_000), total);

    try expectEqual(Dtype.i16, Dtype.fromName("I16").?);
    try expectEqual(Dtype.i32, Dtype.fromName("I32").?);
    try std.testing.expectEqualStrings("I16", Dtype.i16.name());
    try std.testing.expectEqualStrings("I32", Dtype.i32.name());
}

test "codec: bitpack roundtrip at every width" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xB17);
    const rng = prng.random();

    for (1..33) |w| {
        const width: u8 = @intCast(w);
        const bpe = types.roundUpToPow2(width);
        for ([_]usize{ 0, 1, 7, 64, 333 }) |n| {
            var in = try Stream.init(a, n, bpe);
            defer in.deinit(a);
            const m: u32 = if (width >= 32) 0xFFFF_FFFF else (@as(u32, 1) << @intCast(width)) - 1;
            for (0..n) |i| in.setU32(i, rng.int(u32) & m);

            const bits = try codec.bitpackEncode(a, in, width);
            defer a.free(bits);
            try expectEqual((n * w + 7) / 8, bits.len);
            try expectEqual(@as(u64, bits.len) * 8, codec.bitpackCostBits(in, width));

            var back = try codec.bitpackDecode(a, bits, width, n, bpe);
            defer back.deinit(a);
            try expectStreamsEqual(in, back);
        }
    }
}

test "codec: huffman and rans roundtrip" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC0DEC);
    const rng = prng.random();

    for ([_]u8{ 1, 8, 16 }) |bpe| {
        for ([_]bool{ true, false }) |skew| {
            var in = try randStream(a, rng, 700, bpe, skew);
            defer in.deinit(a);

            var ht = try codec.huffmanBuild(a, in);
            defer ht.deinit(a);
            const hp = try codec.huffmanEncode(a, in, ht);
            defer a.free(hp);
            var hb = try codec.huffmanDecode(a, hp, ht, in.count, bpe);
            defer hb.deinit(a);
            try expectStreamsEqual(in, hb);
            // The searcher prices Huffman without encoding; that number must
            // equal what the encoder actually emits, or g_bytes is a lie.
            var hist = try codec.buildHistogram(a, in);
            defer hist.deinit(a);
            try expectEqual(hp.len, codec.huffmanPayloadBytes(ht, hist));

            var rt = try codec.ransBuild(a, in);
            defer rt.deinit(a);
            const rp = try codec.ransEncode(a, in, rt);
            defer a.free(rp);
            var rb = try codec.ransDecode(a, rp, rt, in.count, bpe);
            defer rb.deinit(a);
            try expectStreamsEqual(in, rb);
            // rANS is priced by a strict lower bound, so it must never exceed
            // the realized payload.
            try expect(codec.ransLowerBytes(rt, hist) <= rp.len);
        }
    }
}

// ==================== operator inverses ====================

const ONE_TO_ONE = [_]OpKind{
    .xor_const, .add_const_mod, .xor_prev,    .diff_mod, .zigzag,
    .gray,      .rotate_bits,   .bit_reverse,
};

const WIDTHS = [_]u8{ 1, 2, 3, 5, 8, 11, 16, 23, 32 };

test "ops: every 1->1 operator inverts" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x1701);
    const rng = prng.random();

    for (ONE_TO_ONE) |op| {
        for (0..200) |t| {
            const bpe = WIDTHS[rng.uintLessThan(usize, WIDTHS.len)];
            const n = if (t < 3) t else rng.uintLessThan(usize, 400) + 1;
            var in = if (t % 3 == 0)
                try runStream(a, rng, n, bpe)
            else
                try randStream(a, rng, n, bpe, t % 2 == 0);
            defer in.deinit(a);
            try expectInverts(a, op, rng.int(u32), in);
        }
    }
}

/// A legal parameterisation for `op` on `in`, or null when none exists.
fn randParams(rng: std.Random, op: OpKind, in: Stream) ?u32 {
    const k = in.bits_per_elem;
    return switch (op) {
        .split_field => blk: {
            if (k < 2) break :blk null;
            const n = rng.uintLessThan(u8, k) + 1; // 1..k
            const start = rng.uintAtMost(u8, k - n);
            break :blk @as(u32, start) | (@as(u32, n) << 8);
        },
        .topk_codebook => rng.uintLessThan(u32, 64) + 1,
        .deinterleave => blk: {
            const period = rng.uintLessThan(u32, 7) + 2; // 2..8
            const take = rng.uintLessThan(u32, period - 1) + 1;
            break :blk period | (take << 16);
        },
        .rle, .bit_plane, .byte_plane => 0,
        else => unreachable,
    };
}

test "ops: every 1->n operator inverts" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x1702);
    const rng = prng.random();

    const fanout = [_]OpKind{ .split_field, .topk_codebook, .rle, .deinterleave, .bit_plane, .byte_plane };
    for (fanout) |op| {
        for (0..200) |t| {
            const bpe = WIDTHS[rng.uintLessThan(usize, WIDTHS.len)];
            const n = if (t < 3) t else rng.uintLessThan(usize, 400) + 1;
            var in = if (t % 2 == 0)
                try runStream(a, rng, n, bpe)
            else
                try randStream(a, rng, n, bpe, t % 3 == 0);
            defer in.deinit(a);
            const params = randParams(rng, op, in) orelse continue;
            try expectInverts(a, op, params, in);
        }
    }
}

test "ops: split_float inverts for f16, bf16 and f32" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x1703);
    const rng = prng.random();

    for ([_]Dtype{ .f16, .bf16, .f32 }) |dt| {
        const f = dt.floatFields().?;
        for (0..200) |t| {
            const n = if (t < 3) t else rng.uintLessThan(usize, 400) + 1;
            var in = if (t % 2 == 0)
                try floatStream(a, rng, n, dt)
            else
                try randStream(a, rng, n, dt.bitWidth(), false);
            defer in.deinit(a);

            var outs: std.ArrayList(Stream) = .empty;
            defer {
                for (outs.items) |*s| s.deinit(a);
                outs.deinit(a);
            }
            var side: ops.SideInfo = .none;
            defer side.deinit(a);

            const params: u32 = @intFromEnum(dt);
            try ops.forward(a, .split_float, params, in, &outs, &side);
            try expectEqual(@as(usize, 3), outs.items.len);
            try expectEqual(@as(u8, 1), outs.items[0].bits_per_elem);
            try expectEqual(f.exp, outs.items[1].bits_per_elem);
            try expectEqual(f.mant, outs.items[2].bits_per_elem);

            var back = try ops.inverse(a, .split_float, params, outs.items, side);
            defer back.deinit(a);
            try expectStreamsEqual(in, back);
        }
    }
}

// ==================== random program trees ====================

fn randTerminal(rng: std.Random, in: Stream) Node {
    if (in.count == 0) return .{ .op = .raw };
    const pick = rng.uintLessThan(u8, if (in.bits_per_elem <= ops.MAX_ENTROPY_BPE) 4 else 2);
    return switch (pick) {
        0 => .{ .op = .raw },
        1 => blk: {
            const lo = maxBits(in);
            const w = lo + rng.uintAtMost(u8, in.bits_per_elem - lo);
            break :blk .{ .op = .bitpack, .params = w };
        },
        2 => .{ .op = .huffman },
        else => .{ .op = .rans },
    };
}

const Prod = struct { op: OpKind, params: u32 };

fn randTree(a: Allocator, rng: std.Random, in: Stream, dt: Dtype, depth: u8, budget: *usize) !Node {
    if (in.count == 0 or depth >= 3 or budget.* < 4 or rng.float(f32) < 0.35) {
        budget.* -= 1;
        return randTerminal(rng, in);
    }

    var cands: [24]Prod = undefined;
    var n: usize = 0;
    for (ONE_TO_ONE) |op| {
        cands[n] = .{ .op = op, .params = rng.int(u32) };
        n += 1;
    }
    for ([_]OpKind{ .rle, .deinterleave, .topk_codebook, .split_field, .byte_plane }) |op| {
        if (randParams(rng, op, in)) |p| {
            cands[n] = .{ .op = op, .params = p };
            n += 1;
        }
    }
    if (in.bits_per_elem <= 8) {
        cands[n] = .{ .op = .bit_plane, .params = 0 };
        n += 1;
    }
    if (depth == 0) {
        if (dt.floatFields()) |f| {
            if (f.total == in.bits_per_elem) {
                cands[n] = .{ .op = .split_float, .params = @intFromEnum(dt) };
                n += 1;
            }
        }
    }

    const prod = cands[rng.uintLessThan(usize, n)];
    const ar = ops.arity(prod.op, in.bits_per_elem);
    if (ar + 1 > budget.*) {
        budget.* -= 1;
        return randTerminal(rng, in);
    }
    budget.* -= 1;

    var outs: std.ArrayList(Stream) = .empty;
    defer {
        for (outs.items) |*s| s.deinit(a);
        outs.deinit(a);
    }
    var side: ops.SideInfo = .none;
    defer side.deinit(a);
    try ops.forward(a, prod.op, prod.params, in, &outs, &side);

    const kids = try a.alloc(Node, outs.items.len);
    var filled: usize = 0;
    errdefer {
        for (kids[0..filled]) |*c| c.deinit(a);
        a.free(kids);
    }
    for (outs.items, 0..) |s, i| {
        const reserve = outs.items.len - i - 1;
        var child_budget = budget.* - reserve;
        kids[i] = try randTree(a, rng, s, dt, depth + 1, &child_budget);
        budget.* = child_budget + reserve;
        filled = i + 1;
    }
    return .{ .op = prod.op, .params = prod.params, .children = kids };
}

const TREE_DTYPES = [_]Dtype{ .f16, .bf16, .f32, .u8, .u16, .u32, .i8, .i16, .i32 };

test "program: random trees execute and decode losslessly" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x9A17);
    const rng = prng.random();

    for (0..140) |t| {
        const dt = TREE_DTYPES[t % TREE_DTYPES.len];
        const n = rng.uintLessThan(usize, 500) + 1;
        var in = if (dt.isFloat() and t % 2 == 0)
            try floatStream(a, rng, n, dt)
        else
            try randStream(a, rng, n, dt.bitWidth(), t % 3 == 0);
        defer in.deinit(a);

        var budget: usize = 40;
        var node = try randTree(a, rng, in, dt, 0, &budget);
        defer node.deinit(a);
        try expectEqual(@as(usize, 40) - node.countNodes(), budget);

        try program.execute(a, &node, in);
        var back = try program.decode(a, node);
        defer back.deinit(a);
        try expectStreamsEqual(in, back);
    }
}

test "program: cloned trees decode identically" {
    if (!HAVE_TABLE_CLONE) return error.SkipZigTest;
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xC10E);
    const rng = prng.random();

    for (0..40) |t| {
        const dt = TREE_DTYPES[t % TREE_DTYPES.len];
        var in = try randStream(a, rng, rng.uintLessThan(usize, 300) + 1, dt.bitWidth(), t % 2 == 0);
        defer in.deinit(a);

        var budget: usize = 40;
        var node = try randTree(a, rng, in, dt, 0, &budget);
        defer node.deinit(a);
        try program.execute(a, &node, in);

        var copy = try node.clone(a);
        defer copy.deinit(a);
        node.deinit(a);
        node = .{ .op = .raw };

        var back = try program.decode(a, copy);
        defer back.deinit(a);
        try expectStreamsEqual(in, back);
    }
}

test "program: serialize/deserialize roundtrip" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5E21A);
    const rng = prng.random();

    for (0..140) |t| {
        const dt = TREE_DTYPES[t % TREE_DTYPES.len];
        const n = rng.uintLessThan(usize, 400) + 1;
        var in = try randStream(a, rng, n, dt.bitWidth(), t % 2 == 0);
        defer in.deinit(a);

        var budget: usize = 40;
        var node = try randTree(a, rng, in, dt, 0, &budget);
        defer node.deinit(a);
        try program.execute(a, &node, in);

        const bc = try program.serialize(a, node);
        defer a.free(bc);
        var back = try program.deserialize(a, bc);
        defer back.deinit(a);

        try expectEqual(node.countNodes(), back.countNodes());
        try expectEqual(node.depth(), back.depth());

        const bc2 = try program.serialize(a, back);
        defer a.free(bc2);
        try std.testing.expectEqualSlices(u8, bc, bc2);

        const payload = try program.collectPayload(a, node);
        defer a.free(payload);
        try program.distributePayload(&back, payload);
        var out = try program.decode(a, back);
        defer out.deinit(a);
        try expectStreamsEqual(in, out);
    }
}

test "program: rejects an invalid rans table" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(7);
    var in = try runStream(a, prng.random(), 256, 8);
    defer in.deinit(a);
    var node: Node = .{ .op = .rans };
    defer node.deinit(a);
    try program.execute(a, &node, in);

    const bytecode = try program.serialize(a, node);
    defer a.free(bytecode);
    const first_freq = 1 + 4 + 1 + 8 + 1 + 4 + 4;
    std.mem.writeInt(u32, bytecode[first_freq..][0..4], 0, .little);
    try std.testing.expectError(error.InvalidRansTable, program.deserialize(a, bytecode));
}

test "program: empty rans table roundtrip" {
    const a = std.testing.allocator;
    var in = try Stream.init(a, 0, 8);
    defer in.deinit(a);
    var node: Node = .{ .op = .rans };
    defer node.deinit(a);
    try program.execute(a, &node, in);

    const bytecode = try program.serialize(a, node);
    defer a.free(bytecode);
    const payload = try program.collectPayload(a, node);
    defer a.free(payload);
    var restored = try program.deserialize(a, bytecode);
    defer restored.deinit(a);
    try program.distributePayload(&restored, payload);
    var out = try program.decode(a, restored);
    defer out.deinit(a);
    try expectStreamsEqual(in, out);
}

test "prior: scores normalize over legal productions" {
    const a = std.testing.allocator;
    const legal = [_]OpKind{ .raw, .bitpack, .huffman, .rans };
    var scores: [legal.len]u32 = undefined;

    var untrained: prior.Prior = .empty;
    defer untrained.deinit(a);
    untrained.scoreSet(.{}, &legal, &scores);
    for (scores) |score| try expectEqual(@as(u32, 2048), score);

    var counts = prior.Counts.init(a);
    defer counts.deinit(a);
    try counts.add(a, .{}, .raw, 9);
    var learned = try counts.toPrior(a);
    defer learned.deinit(a);
    learned.scoreSet(.{}, legal[0..2], scores[0..2]);
    try expect(scores[0] < scores[1]);
    const total = std.math.exp2(-@as(f64, @floatFromInt(scores[0])) / 1024.0) +
        std.math.exp2(-@as(f64, @floatFromInt(scores[1])) / 1024.0);
    try std.testing.expectApproxEqAbs(@as(f64, 1), total, 0.001);
}

// ==================== search ====================

/// Synthetic blocks with the structure real weights have.
fn makeBlock(a: Allocator, rng: std.Random, dt: Dtype, n: usize, kind: u8) !Stream {
    var s = try Stream.init(a, n, dt.bitWidth());
    for (0..n) |i| {
        const v: u32 = switch (kind) {
            0 => switch (dt) { // gaussian-ish weights
                .bf16 => 0x3C00 -% @as(u32, rng.uintLessThan(u16, 0x0400)) | (@as(u32, rng.int(u1)) << 15),
                .f16 => 0x3800 -% @as(u32, rng.uintLessThan(u16, 0x0400)) | (@as(u32, rng.int(u1)) << 15),
                .f32 => 0x3F000000 -% @as(u32, rng.uintLessThan(u32, 0x00400000)) | (@as(u32, rng.int(u1)) << 31),
                else => rng.uintLessThan(u32, 40) +% 108,
            },
            1 => switch (dt) { // near-constant, as in layernorm gains
                .f32 => 0x3F800000 +% rng.uintLessThan(u32, 64),
                else => 0x3C00 +% rng.uintLessThan(u32, 4),
            },
            else => switch (dt) { // upcast fp32: low mantissa bits zero
                .f32 => (0x3F000000 -% @as(u32, rng.uintLessThan(u32, 0x00400000))) & 0xFFFF0000,
                else => rng.int(u32),
            },
        };
        s.setU32(i, v & s.mask());
    }
    return s;
}

/// A prior with lopsided scores, to prove pruning does not depend on it.
fn skewedPrior(a: Allocator, rng: std.Random, in: Stream, dt: Dtype) !prior.Prior {
    var counts = prior.Counts.init(a);
    defer counts.deinit(a);
    for ([_]u8{ 0, 1, 2 }) |depth| {
        for ([_]u8{ 0, 1, 2 }) |slot| {
            const ctx = prior.Context.fromStream(in, dt, slot, depth, 255);
            for (std.enums.values(OpKind)) |op| try counts.add(a, ctx, op, rng.float(f64) * 100.0);
        }
    }
    return counts.toPrior(a);
}

test "search: enumeration and pruning find the same winner" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5EA);
    const rng = prng.random();

    const tight: search.Options = .{
        .max_nodes = 3,
        .max_depth = 1,
        .max_expansions = 20_000,
        .max_realizations = 20_000,
    };

    for ([_]Dtype{ .bf16, .f16, .f32, .i8, .u8 }) |dt| {
        for ([_]u8{ 0, 1, 2 }) |kind| {
            var in = try makeBlock(a, rng, dt, 512, kind);
            defer in.deinit(a);

            var untrained: prior.Prior = .empty;
            defer untrained.deinit(a);
            var skewed = try skewedPrior(a, rng, in, dt);
            defer skewed.deinit(a);

            var ex_opts = tight;
            ex_opts.enumerate_all = true;
            var ex = try search.synthesize(a, in, dt, &untrained, ex_opts);
            defer ex.deinit(a);
            try expect(ex.expanded < tight.max_expansions);

            for ([_]*const prior.Prior{ &untrained, &skewed }) |pr| {
                var pruned = try search.synthesize(a, in, dt, pr, tight);
                defer pruned.deinit(a);
                try expectEqual(ex.bytes, pruned.bytes);

                var back = try program.decode(a, pruned.node);
                defer back.deinit(a);
                try expectStreamsEqual(in, back);
            }
        }
    }
}

test "search: synthesizeAll returns sorted candidates that all decode" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xA11);
    const rng = prng.random();

    var in = try makeBlock(a, rng, .f16, 256, 0);
    defer in.deinit(a);

    const all = try search.synthesizeAll(a, in, .f16, .{
        .enumerate_all = true,
        .max_nodes = 3,
        .max_depth = 1,
        .max_expansions = 200_000,
    });
    defer {
        for (all) |*r| r.deinit(a);
        a.free(all);
    }

    try expect(all.len > 1);
    for (all, 0..) |r, i| {
        if (i > 0) try expect(all[i - 1].bytes <= r.bytes);
        var back = try program.decode(a, r.node);
        defer back.deinit(a);
        try expectStreamsEqual(in, back);
    }
}

test "search: sampled synthesis is bit-exact on the full stream" {
    const a = std.testing.allocator;
    var in = try Stream.init(a, 4096, 16);
    defer in.deinit(a);

    var prng: std.Random.DefaultPrng = .init(1);
    const rng = prng.random();
    for (0..in.count) |i| {
        const value = 1.0 + rng.floatNorm(f32) * 0.02;
        in.setU32(i, @as(u16, @bitCast(@as(f16, @floatCast(value)))));
    }

    var untrained: prior.Prior = .empty;
    defer untrained.deinit(a);
    var plan = try search.synthesizePlan(a, in, .f16, &untrained, .{});
    defer plan.deinit(a);
    var result = try search.encode(a, &plan, in, .f16);
    defer result.deinit(a);
    var decoded = try program.decode(a, result.node);
    defer decoded.deinit(a);
    try expectStreamsEqual(in, decoded);

    var other = try makeBlock(a, rng, .f16, 4096, 1);
    defer other.deinit(a);
    var other_result = try search.encode(a, &plan, other, .f16);
    defer other_result.deinit(a);
    var other_decoded = try program.decode(a, other_result.node);
    defer other_decoded.deinit(a);
    try expectStreamsEqual(other, other_decoded);
}

test "search: planning memory is bounded by the sample" {
    const a = std.testing.allocator;
    var in = try Stream.init(a, 1 << 20, 8);
    defer in.deinit(a);
    for (0..in.count) |i| in.setU32(i, @intCast(i & 0xFF));

    var memory: [256 * 1024]u8 = undefined;
    var fixed = std.heap.FixedBufferAllocator.init(&memory);
    const fa = fixed.allocator();
    var untrained: prior.Prior = .empty;
    defer untrained.deinit(fa);
    var plan = try search.synthesizePlan(fa, in, .u8, &untrained, .{
        .sample_elems = 512,
        .max_expansions = 1,
        .max_realizations = 1,
        .max_nodes = 1,
    });
    defer plan.deinit(fa);
}

test "search: encoding refits parameters and falls back to raw" {
    const a = std.testing.allocator;
    var in = try Stream.init(a, 1024, 8);
    defer in.deinit(a);
    for (0..in.count) |i| in.setU32(i, if (i % 4 == 0) 0xF1 else 0xF0);

    const child = try a.alloc(Node, 1);
    child[0] = .{ .op = .bitpack, .params = 8 };
    var plan: search.Plan = .{ .root = .{ .op = .xor_const, .children = child }, .expanded = 0 };
    defer plan.deinit(a);
    var frame = try search.encode(a, &plan, in, .u8);
    defer frame.deinit(a);
    try expectEqual(@as(u32, 0xF0), frame.node.params);
    try expectEqual(@as(u32, 1), frame.node.children[0].params);
    var decoded = try program.decode(a, frame.node);
    defer decoded.deinit(a);
    try expectStreamsEqual(in, decoded);

    var raw_plan: search.Plan = .{ .root = .{ .op = .huffman }, .expanded = 0 };
    defer raw_plan.deinit(a);
    for (0..in.count) |i| in.setU32(i, @intCast(i & 0xFF));
    var raw = try search.encode(a, &raw_plan, in, .u8);
    defer raw.deinit(a);
    try expectEqual(OpKind.raw, raw.node.op);
    try expectEqual(in.data.len + 24, raw.bytes);
}

test "search: rans alphabet growth falls back to raw" {
    const a = std.testing.allocator;
    var in = try Stream.init(a, 16_385, 16);
    defer in.deinit(a);
    for (0..in.count) |i| in.setU32(i, @intCast(i));

    var plan: search.Plan = .{ .root = .{ .op = .rans }, .expanded = 0 };
    defer plan.deinit(a);
    var frame = try search.encode(a, &plan, in, .u16);
    defer frame.deinit(a);
    try expectEqual(OpKind.raw, frame.node.op);
    var decoded = try program.decode(a, frame.node);
    defer decoded.deinit(a);
    try expectStreamsEqual(in, decoded);
}

// ==================== dtype legality ====================

test "search: i8 blocks never produce split_float" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x18);
    const rng = prng.random();

    var untrained: prior.Prior = .empty;
    defer untrained.deinit(a);

    for ([_]u8{ 0, 1, 2 }) |kind| {
        var in = try makeBlock(a, rng, .i8, 512, kind);
        defer in.deinit(a);

        var r = try search.synthesize(a, in, .i8, &untrained, .{});
        defer r.deinit(a);
        try expect(!hasOp(r.node, .split_float));

        const all = try search.synthesizeAll(a, in, .i8, .{
            .enumerate_all = true,
            .max_nodes = 4,
            .max_depth = 2,
            .max_expansions = 200_000,
        });
        defer {
            for (all) |*x| x.deinit(a);
            a.free(all);
        }
        for (all) |x| try expect(!hasOp(x.node, .split_float));
    }
}

/// Assert no entropy coder is ever handed a stream wider than MAX_ENTROPY_BPE,
/// and count the ones reached through a split of an over-wide stream.
fn checkEntropyWidths(a: Allocator, node: Node, in: Stream, split_above: bool, narrowed: *usize) !void {
    if (node.op.isTerminal()) {
        if (node.op == .huffman or node.op == .rans) {
            try expect(in.bits_per_elem <= ops.MAX_ENTROPY_BPE);
            if (split_above) narrowed.* += 1;
        }
        return;
    }

    var outs: std.ArrayList(Stream) = .empty;
    defer {
        for (outs.items) |*s| s.deinit(a);
        outs.deinit(a);
    }
    var side: ops.SideInfo = .none;
    defer side.deinit(a);
    try ops.forward(a, node.op, node.params, in, &outs, &side);

    const splits = node.op == .byte_plane or node.op == .split_field or node.op == .bit_plane;
    const under = split_above or (splits and in.bits_per_elem > ops.MAX_ENTROPY_BPE);
    for (node.children, outs.items) |c, s| try checkEntropyWidths(a, c, s, under, narrowed);
}

test "search: f32 mantissa reaches entropy coders only through a split" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x32);
    const rng = prng.random();

    var untrained: prior.Prior = .empty;
    defer untrained.deinit(a);

    var narrowed: usize = 0;
    for ([_]u8{ 0, 1, 2 }) |kind| {
        var in = try makeBlock(a, rng, .f32, 1024, kind);
        defer in.deinit(a);

        var r = try search.synthesize(a, in, .f32, &untrained, .{});
        defer r.deinit(a);
        try checkEntropyWidths(a, r.node, in, false, &narrowed);

        const all = try search.synthesizeAll(a, in, .f32, .{
            .enumerate_all = true,
            .max_nodes = 4,
            .max_depth = 2,
            .max_expansions = 200_000,
        });
        defer {
            for (all) |*x| x.deinit(a);
            a.free(all);
        }
        for (all) |x| try checkEntropyWidths(a, x.node, in, false, &narrowed);
    }
    try expect(narrowed > 0);
}

test "program: f32 mantissa splits into byte planes an entropy coder accepts" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x33);
    const rng = prng.random();

    var in = try makeBlock(a, rng, .f32, 1024, 0);
    defer in.deinit(a);

    // split_float -> [ huffman(sign), huffman(exp), byte_plane -> 3x huffman ]
    const planes = try a.alloc(Node, 3);
    for (planes) |*p| p.* = .{ .op = .huffman };
    const kids = try a.alloc(Node, 3);
    kids[0] = .{ .op = .huffman };
    kids[1] = .{ .op = .huffman };
    kids[2] = .{ .op = .byte_plane, .children = planes };
    var node: Node = .{ .op = .split_float, .params = @intFromEnum(Dtype.f32), .children = kids };
    defer node.deinit(a);

    try program.execute(a, &node, in);
    try expectEqual(@as(u8, 23), node.children[2].side.split.bits_per_elem);
    for (node.children[2].children) |p| try expect(p.side.huffman.bits_per_elem <= ops.MAX_ENTROPY_BPE);

    var narrowed: usize = 0;
    try checkEntropyWidths(a, node, in, false, &narrowed);
    try expectEqual(@as(usize, 3), narrowed);

    var back = try program.decode(a, node);
    defer back.deinit(a);
    try expectStreamsEqual(in, back);
}

// ==================== archive ====================

const ArchiveDecodeJob = struct {
    block: archive.ParsedBlock,
    stream: ?Stream = null,
    err: ?anyerror = null,

    fn run(self: *@This()) void {
        self.stream = archive.decodeBlock(std.heap.smp_allocator, self.block) catch |err| {
            self.err = err;
            return;
        };
    }
};

test "archive: multi-tensor multi-block roundtrip" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xA2C);
    const rng = prng.random();

    var untrained: prior.Prior = .empty;
    defer untrained.deinit(a);

    const metas = [_]archive.TensorMeta{
        .{ .name = "block.0.weight", .dtype = .f16, .shape = &.{ 3, 100 }, .n_blocks = 3 },
        .{ .name = "block.0.scales", .dtype = .i8, .shape = &.{ 2, 64 }, .n_blocks = 2 },
    };

    var blocks: std.ArrayList(Stream) = .empty;
    defer {
        for (blocks.items) |*s| s.deinit(a);
        blocks.deinit(a);
    }
    var results: std.ArrayList(search.Result) = .empty;
    defer {
        for (results.items) |*r| r.deinit(a);
        results.deinit(a);
    }
    var jobs: std.ArrayList(archive.BlockJob) = .empty;
    defer jobs.deinit(a);

    for (metas, 0..) |meta, ti| {
        for (0..meta.n_blocks) |_| {
            const s = try makeBlock(a, rng, meta.dtype, @intCast(meta.shape[1]), @intCast(ti));
            try blocks.append(a, s);
            const r = try search.synthesize(a, s, meta.dtype, &untrained, .{ .max_nodes = 6, .max_depth = 2 });
            try results.append(a, r);
        }
    }
    for (results.items) |*r| {
        try jobs.append(a, .{ .node = &r.node, .payload = r.payload });
    }

    const bytes = try archive.build(a, &metas, jobs.items, &.{});
    defer a.free(bytes);
    var frame_bytes: usize = archive.HEADER.len;
    for (results.items) |result| frame_bytes += result.bytes + 12;
    try expectEqual(frame_bytes, @as(usize, @intCast(std.mem.readInt(u64, bytes[bytes.len - 8 ..][0..8], .little))));

    var parsed = try archive.parse(a, bytes);
    defer parsed.deinit();

    try expectEqual(metas.len, parsed.tensors.len);
    var bi: usize = 0;
    for (parsed.tensors, metas) |pt, meta| {
        try std.testing.expectEqualStrings(meta.name, pt.name);
        try expectEqual(meta.dtype, pt.dtype);
        try std.testing.expectEqualSlices(u64, meta.shape, pt.shape);
        try expectEqual(meta.n_blocks, pt.n_blocks);
        var pos: usize = pt.frame_start;
        for (0..pt.n_blocks) |_| {
            const blk = try archive.nextBlock(parsed.frames, &pos);
            var back = try archive.decodeBlock(a, blk);
            defer back.deinit(a);
            try expectStreamsEqual(blocks.items[bi], back);
            bi += 1;
        }
    }
    try expectEqual(blocks.items.len, bi);
}

test "archive: self-contained frames decode concurrently" {
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xDED0);
    const rng = prng.random();

    var streams: [3]Stream = undefined;
    streams[0] = try randStream(a, rng, 256, 8, true);
    streams[1] = try streams[0].dupe(a);
    streams[2] = try randStream(a, rng, 256, 8, false);
    defer for (&streams) |*s| s.deinit(a);

    var nodes: [3]Node = undefined;
    var payloads: [3][]u8 = undefined;
    var jobs: [3]archive.BlockJob = undefined;
    defer for (&nodes, payloads) |*n, p| {
        n.deinit(a);
        a.free(p);
    };
    for (&nodes, &payloads, &jobs, streams) |*n, *p, *j, s| {
        n.* = .{ .op = .huffman };
        try program.execute(a, n, s);
        p.* = try program.collectPayload(a, n.*);
        j.* = .{ .node = n, .payload = p.* };
    }

    const metas = [_]archive.TensorMeta{
        .{ .name = "w", .dtype = .u8, .shape = &.{ 3, 256 }, .n_blocks = 3 },
    };
    const bytes = try archive.build(a, &metas, &jobs, &.{});
    defer a.free(bytes);

    var parsed = try archive.parse(a, bytes);
    defer parsed.deinit();
    var pos: usize = 0;
    var parsed_blocks: [3]archive.ParsedBlock = undefined;
    for (&parsed_blocks) |*block| block.* = try archive.nextBlock(parsed.frames, &pos);
    try std.testing.expectEqualSlices(u8, parsed_blocks[0].bytecode, parsed_blocks[1].bytecode);

    var decode_jobs: [3]ArchiveDecodeJob = undefined;
    var threads: [3]std.Thread = undefined;
    for (&decode_jobs, parsed_blocks) |*job, block| job.* = .{ .block = block };
    for (&threads, &decode_jobs) |*thread, *job| thread.* = try std.Thread.spawn(.{}, ArchiveDecodeJob.run, .{job});
    for (threads) |thread| thread.join();
    for (&decode_jobs, streams) |*job, s| {
        try expectEqual(@as(?anyerror, null), job.err);
        var back = job.stream.?;
        defer back.deinit(std.heap.smp_allocator);
        try expectStreamsEqual(s, back);
    }
}

test "archive: original safetensors prefix and data order are byte-exact" {
    const a = std.testing.allocator;
    const header =
        "{ \n  \"late\": {\"dtype\":\"U8\", \"shape\":[4], \"data_offsets\":[4,8]},\n" ++
        "  \"__metadata__\": {\"source\":\"paper\", \"spacing\":\"kept\"},\n" ++
        "  \"early\": {\"dtype\":\"U8\", \"shape\":[4], \"data_offsets\":[0,4]}\n}   ";

    var source_list: std.ArrayList(u8) = .empty;
    defer source_list.deinit(a);
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header.len, .little);
    try source_list.appendSlice(a, &len_buf);
    try source_list.appendSlice(a, header);
    try source_list.appendSlice(a, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    const source = try source_list.toOwnedSlice(a);

    var loaded = try safetensors.loadFromBytes(a, source);
    defer loaded.deinit(a);
    try std.testing.expectEqualStrings("early", loaded.tensors[0].name);
    try std.testing.expectEqualStrings("late", loaded.tensors[1].name);

    var nodes = [_]Node{ .{ .op = .raw }, .{ .op = .raw } };
    defer for (&nodes) |*node| node.deinit(a);
    var payloads: [2]?[]u8 = @splat(null);
    defer for (payloads) |payload| if (payload) |bytes| a.free(bytes);
    var jobs: [2]archive.BlockJob = undefined;
    var metas: [2]archive.TensorMeta = undefined;
    for (loaded.tensors, 0..) |tensor, i| {
        const stream: Stream = .{
            .data = tensor.view.data,
            .count = tensor.view.numel(),
            .bits_per_elem = tensor.view.dtype.bitWidth(),
            .owns_data = false,
        };
        try program.execute(a, &nodes[i], stream);
        const payload = try program.collectPayload(a, nodes[i]);
        payloads[i] = payload;
        jobs[i] = .{ .node = &nodes[i], .payload = payload };
        metas[i] = .{
            .name = tensor.name,
            .dtype = tensor.view.dtype,
            .shape = tensor.view.shape,
            .n_blocks = 1,
        };
    }

    const prefix_len = 8 + header.len;
    const bytes = try archive.build(a, &metas, &jobs, source[0..prefix_len]);
    defer a.free(bytes);
    var parsed = try archive.parse(a, bytes);
    defer parsed.deinit();
    try std.testing.expectEqualSlices(u8, source[0..prefix_len], parsed.safetensors_prefix);

    var restored: std.ArrayList(u8) = .empty;
    defer restored.deinit(a);
    try restored.appendSlice(a, parsed.safetensors_prefix);
    for (parsed.tensors) |tensor| {
        var pos: usize = tensor.frame_start;
        for (0..tensor.n_blocks) |_| {
            const block = try archive.nextBlock(parsed.frames, &pos);
            var stream = try archive.decodeBlock(a, block);
            defer stream.deinit(a);
            try restored.appendSlice(a, stream.data);
        }
    }
    try std.testing.expectEqualSlices(u8, source, restored.items);
}

// ==================== safetensors ====================

test "safetensors: roundtrip" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var prng = std.Random.DefaultPrng.init(0x5AFE);
    const rng = prng.random();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}/model.safetensors", .{&tmp.sub_path});
    defer a.free(path);

    const names = [_][]const u8{ "enc.weight", "enc.bias", "quant.scale" };
    const dtypes = [_]Dtype{ .f32, .bf16, .i8 };
    const shapes = [_][]const u64{ &.{ 4, 8 }, &.{32}, &.{ 2, 2, 4 } };

    var outs: [3]safetensors.TensorOut = undefined;
    defer for (outs) |o| a.free(o.view.data);
    for (&outs, names, dtypes, shapes) |*o, name, dt, sh| {
        var numel: usize = 1;
        for (sh) |d| numel *= @intCast(d);
        const data = try a.alloc(u8, numel * dt.elemSize());
        rng.bytes(data);
        o.* = .{ .name = name, .view = .{ .data = data, .shape = sh, .dtype = dt } };
    }

    try safetensors.saveToPath(a, io, path, &outs);

    var loaded = try safetensors.loadFromPath(a, io, path);
    defer loaded.deinitMmap(a, io);

    try expectEqual(@as(usize, 3), loaded.tensors.len);
    for (outs) |o| {
        var found = false;
        for (loaded.tensors) |t| {
            if (!std.mem.eql(u8, t.name, o.name)) continue;
            found = true;
            try expectEqual(o.view.dtype, t.view.dtype);
            try std.testing.expectEqualSlices(u64, o.view.shape, t.view.shape);
            try std.testing.expectEqualSlices(u8, o.view.data, t.view.data);
        }
        try expect(found);
    }
}
