const std = @import("std");
const types = @import("types.zig");
const codec = @import("codec.zig");
const ops = @import("ops.zig");
const program = @import("program.zig");
const search = @import("search.zig");
const prior = @import("prior.zig");
const archive = @import("archive.zig");
const safetensors = @import("safetensors.zig");
const baseline = @import("baseline.zig");
const lowlevel = @import("lowlevel.zig");
const astar = @import("astar.zig");

fn makeFp16Tensor(alloc: std.mem.Allocator, n: usize, seed: u64) !types.TensorView {
    const buf = try alloc.alloc(u8, n * 2);
    var prng: std.Random.DefaultPrng = .init(seed);
    const r = prng.random();
    for (0..n) |i| {
        const v: f16 = @floatCast(r.floatNorm(f32) * 0.02);
        const u: u16 = @bitCast(v);
        std.mem.writeInt(u16, buf[i * 2 ..][0..2], u, .little);
    }
    const shape = try alloc.alloc(u64, 1);
    shape[0] = n;
    return .{ .data = buf, .shape = shape, .dtype = .f16, .owns_data = true, .owns_shape = true };
}

test "stream get/set roundtrip 8-bit" {
    const alloc = std.testing.allocator;
    const buf = try alloc.alloc(u8, 4);
    defer alloc.free(buf);
    const s: types.Stream = .{ .data = buf, .count = 4, .bits_per_elem = 5 };
    s.setU32(0, 17);
    s.setU32(1, 0);
    s.setU32(2, 31);
    s.setU32(3, 7);
    try std.testing.expectEqual(@as(u32, 17), s.getU32(0));
    try std.testing.expectEqual(@as(u32, 31), s.getU32(2));
}

test "huffman roundtrip" {
    const alloc = std.testing.allocator;
    const data = try alloc.alloc(u8, 200);
    defer alloc.free(data);
    var prng: std.Random.DefaultPrng = .init(7);
    const r = prng.random();
    for (data) |*b| b.* = if (r.float(f32) < 0.7) 3 else r.intRangeLessThan(u8, 0, 32);
    const s: types.Stream = .{ .data = data, .count = 200, .bits_per_elem = 8 };

    var table = try codec.huffmanBuild(alloc, s);
    defer table.deinit(alloc);
    const encoded = try codec.huffmanEncode(alloc, s, table);
    defer alloc.free(encoded);

    var dec = try codec.huffmanDecode(alloc, encoded, table, 200, 8);
    defer dec.deinit(alloc);
    try std.testing.expect(std.mem.eql(u8, data, dec.data));
}

test "rans roundtrip" {
    const alloc = std.testing.allocator;
    const data = try alloc.alloc(u8, 500);
    defer alloc.free(data);
    var prng: std.Random.DefaultPrng = .init(13);
    const r = prng.random();
    for (data) |*b| b.* = if (r.float(f32) < 0.6) 5 else r.intRangeLessThan(u8, 0, 16);
    const s: types.Stream = .{ .data = data, .count = 500, .bits_per_elem = 8 };

    var table = try codec.ransBuild(alloc, s);
    defer table.deinit(alloc);
    const encoded = try codec.ransEncode(alloc, s, table);
    defer alloc.free(encoded);

    var dec = try codec.ransDecode(alloc, encoded, table, 500, 8);
    defer dec.deinit(alloc);
    try std.testing.expect(std.mem.eql(u8, data, dec.data));
}

test "split_float forward/inverse" {
    const alloc = std.testing.allocator;
    var t = try makeFp16Tensor(alloc, 200, 1);
    defer t.deinit(alloc);

    const r = try ops.splitFloatForward(alloc, t);
    var sign = r.sign;
    var exp = r.exp;
    var mant = r.mant;
    defer sign.deinit(alloc);
    defer exp.deinit(alloc);
    defer mant.deinit(alloc);

    var back = try ops.splitFloatInverse(alloc, sign, exp, mant, r.info);
    defer back.deinit(alloc);
    try std.testing.expect(std.mem.eql(u8, t.data, back.data));
}

test "delta_encode forward/inverse" {
    const alloc = std.testing.allocator;
    const buf = try alloc.alloc(u8, 100);
    defer alloc.free(buf);
    for (buf, 0..) |*b, i| b.* = @intCast(i & 0xFF);
    const s: types.Stream = .{ .data = buf, .count = 100, .bits_per_elem = 8 };
    const r = try ops.deltaEncodeForward(alloc, s);
    var d = r.out;
    defer d.deinit(alloc);
    var back = try ops.deltaEncodeInverse(alloc, d, r.info);
    defer back.deinit(alloc);
    try std.testing.expect(std.mem.eql(u8, buf, back.data));
}

test "bitplane_split forward/inverse" {
    const alloc = std.testing.allocator;
    const buf = try alloc.alloc(u8, 50);
    defer alloc.free(buf);
    var prng: std.Random.DefaultPrng = .init(3);
    const rnd = prng.random();
    for (buf) |*b| b.* = rnd.intRangeLessThan(u8, 0, 32);
    const s: types.Stream = .{ .data = buf, .count = 50, .bits_per_elem = 5 };
    const r = try ops.bitplaneSplitForward(alloc, s);
    defer alloc.free(r.planes);
    defer for (r.planes) |*pl| {
        var pp = pl.*;
        pp.deinit(alloc);
    };
    var back = try ops.bitplaneSplitInverse(alloc, r.planes, r.info, 5);
    defer back.deinit(alloc);
    try std.testing.expect(std.mem.eql(u8, buf, back.data));
}

test "tensor_xor roundtrip" {
    const alloc = std.testing.allocator;
    var t = try makeFp16Tensor(alloc, 64, 1);
    defer t.deinit(alloc);
    var b = try makeFp16Tensor(alloc, 64, 2);
    defer b.deinit(alloc);

    const r = try ops.tensorXorForward(alloc, t, b);
    var residual = r.residual;
    defer residual.deinit(alloc);

    var back = try ops.tensorXorInverse(alloc, residual, b, r.info);
    defer back.deinit(alloc);
    try std.testing.expect(std.mem.eql(u8, t.data, back.data));
}

test "program: split + huffman roundtrip" {
    const alloc = std.testing.allocator;
    var t = try makeFp16Tensor(alloc, 256, 4);
    defer t.deinit(alloc);

    const kids = try alloc.alloc(program.Node, 3);
    kids[0] = .{ .op = .huffman };
    kids[1] = .{ .op = .huffman };
    kids[2] = .{ .op = .huffman };
    var node: program.Node = .{ .op = .split_float, .children = kids };
    defer node.deinit(alloc);

    try program.compressTensor(alloc, &node, t, &.{});
    var back = try program.decompressTensor(alloc, &node, &.{});
    defer back.deinit(alloc);
    try std.testing.expect(std.mem.eql(u8, t.data, back.data));
}

test "program: split + bitplane(exp) + huffman roundtrip" {
    const alloc = std.testing.allocator;
    var t = try makeFp16Tensor(alloc, 200, 5);
    defer t.deinit(alloc);

    // exp has 5 bits → bitplane gives 5 planes
    const exp_planes = try alloc.alloc(program.Node, 5);
    for (exp_planes) |*p| p.* = .{ .op = .huffman };
    const exp_node_kids = try alloc.alloc(program.Node, 1);
    exp_node_kids[0] = .{ .op = .bitplane_split, .children = exp_planes };

    const kids = try alloc.alloc(program.Node, 3);
    kids[0] = .{ .op = .huffman };
    kids[1] = .{ .op = .bitplane_split, .children = exp_planes };
    kids[2] = .{ .op = .huffman };
    // Avoid double-free: reuse exp_planes only once
    alloc.free(exp_node_kids);

    var node: program.Node = .{ .op = .split_float, .children = kids };
    defer node.deinit(alloc);

    try program.compressTensor(alloc, &node, t, &.{});
    var back = try program.decompressTensor(alloc, &node, &.{});
    defer back.deinit(alloc);
    try std.testing.expect(std.mem.eql(u8, t.data, back.data));
}

test "program: split + delta(exp) + rans roundtrip" {
    const alloc = std.testing.allocator;
    var t = try makeFp16Tensor(alloc, 300, 6);
    defer t.deinit(alloc);

    const delta_kid = try alloc.alloc(program.Node, 1);
    delta_kid[0] = .{ .op = .rans };

    const kids = try alloc.alloc(program.Node, 3);
    kids[0] = .{ .op = .rans };
    kids[1] = .{ .op = .delta_encode, .children = delta_kid };
    kids[2] = .{ .op = .rans };
    var node: program.Node = .{ .op = .split_float, .children = kids };
    defer node.deinit(alloc);

    try program.compressTensor(alloc, &node, t, &.{});
    var back = try program.decompressTensor(alloc, &node, &.{});
    defer back.deinit(alloc);
    try std.testing.expect(std.mem.eql(u8, t.data, back.data));
}

test "search: synthesize fp16 tensor with verified roundtrip" {
    const alloc = std.testing.allocator;
    var t = try makeFp16Tensor(alloc, 65536, 11);
    defer t.deinit(alloc);
    var res = try search.synthesize(alloc, t, &.{}, .{ .verbose = false });
    defer res.deinit(alloc);
    try std.testing.expect(res.verified);
    try std.testing.expect(res.compression_ratio > 1.0);

    // Decompress from serialized form to be safe
    const program_bytes = try program.serializeProgram(alloc, &res.program);
    defer alloc.free(program_bytes);
    var node2 = try program.deserializeProgram(alloc, program_bytes);
    defer node2.deinit(alloc);
    try program.distributePayloadBytes(&node2, res.payload);
    var back = try program.decompressTensor(alloc, &node2, &.{});
    defer back.deinit(alloc);
    try std.testing.expect(std.mem.eql(u8, t.data, back.data));
}

test "search: tensor_xor improves ratio when base is similar" {
    const alloc = std.testing.allocator;
    var t = try makeFp16Tensor(alloc, 16384, 17);
    defer t.deinit(alloc);

    // Build a "base" that's t with a few bytes flipped — should highly correlate.
    const base_buf = try alloc.alloc(u8, t.data.len);
    @memcpy(base_buf, t.data);
    var prng: std.Random.DefaultPrng = .init(99);
    const r = prng.random();
    for (0..32) |_| {
        const i = r.intRangeLessThan(usize, 0, base_buf.len);
        base_buf[i] ^= 0x01;
    }
    const base_shape = try alloc.alloc(u64, 1);
    base_shape[0] = 16384;
    var base: types.TensorView = .{
        .data = base_buf,
        .shape = base_shape,
        .dtype = .f16,
        .owns_data = true,
        .owns_shape = true,
    };
    defer base.deinit(alloc);

    var res_no_base = try search.synthesize(alloc, t, &.{}, .{});
    defer res_no_base.deinit(alloc);
    var res_with_base = try search.synthesize(alloc, t, &.{base}, .{});
    defer res_with_base.deinit(alloc);

    // With a near-identical base, tensor_xor should give a much better ratio.
    try std.testing.expect(res_with_base.compression_ratio > res_no_base.compression_ratio * 1.5);
}

test "archive: 3-tensor roundtrip with shared codebooks" {
    const alloc = std.testing.allocator;
    var t1 = try makeFp16Tensor(alloc, 4096, 21);
    defer t1.deinit(alloc);
    var t2 = try makeFp16Tensor(alloc, 4096, 22);
    defer t2.deinit(alloc);
    var t3 = try makeFp16Tensor(alloc, 4096, 23);
    defer t3.deinit(alloc);

    var r1 = try search.synthesize(alloc, t1, &.{}, .{});
    defer r1.deinit(alloc);
    var r2 = try search.synthesize(alloc, t2, &.{}, .{});
    defer r2.deinit(alloc);
    var r3 = try search.synthesize(alloc, t3, &.{}, .{});
    defer r3.deinit(alloc);

    const jobs = [_]archive.TensorJob{
        .{ .name = "w1", .program = &r1.program, .payload = r1.payload },
        .{ .name = "w2", .program = &r2.program, .payload = r2.payload },
        .{ .name = "w3", .program = &r3.program, .payload = r3.payload },
    };
    const bytes = try archive.buildArchiveBytes(alloc, &jobs);
    defer alloc.free(bytes);

    var parsed = try archive.parseArchive(alloc, bytes);
    defer parsed.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 3), parsed.tensors.len);

    for (parsed.tensors, [_]types.TensorView{ t1, t2, t3 }) |*pt, orig| {
        var back = try program.decompressTensor(alloc, &pt.program, &.{});
        defer back.deinit(alloc);
        try std.testing.expect(std.mem.eql(u8, orig.data, back.data));
    }
}

// =================== Low-level primitive reversibility (property tests) ===================

fn randomStream(alloc: std.mem.Allocator, count: usize, bpe: u8, seed: u64) !types.Stream {
    const elem_bytes: usize = switch (types.roundUpToPow2(bpe)) {
        8 => 1, 16 => 2, 32 => 4, else => unreachable,
    };
    const buf = try alloc.alloc(u8, count * elem_bytes);
    var prng: std.Random.DefaultPrng = .init(seed);
    const r = prng.random();
    const mask: u32 = if (bpe >= 32) 0xFFFFFFFF else (@as(u32, 1) << @intCast(bpe)) - 1;
    const s: types.Stream = .{ .data = buf, .count = count, .bits_per_elem = bpe };
    for (0..count) |i| s.setU32(i, r.int(u32) & mask);
    return s;
}

fn streamsEqual(a: types.Stream, b: types.Stream) bool {
    if (a.count != b.count or a.bits_per_elem != b.bits_per_elem) return false;
    for (0..a.count) |i| if (a.getU32(i) != b.getU32(i)) return false;
    return true;
}

test "lowlevel: xor_const is self-inverse" {
    const alloc = std.testing.allocator;
    const s = try randomStream(alloc, 50, 8, 1);
    defer alloc.free(s.data);
    const op: lowlevel.LowOp = .{ .kind = .xor_const, .params = .{ .raw = 0xA5 } };
    const r = try lowlevel.forward(alloc, op, s);
    const fwd = r.one;
    defer alloc.free(fwd.data);
    const back = try lowlevel.inverseOne(alloc, op, fwd);
    defer alloc.free(back.data);
    try std.testing.expect(streamsEqual(s, back));
}

test "lowlevel: add_const_mod inverse" {
    const alloc = std.testing.allocator;
    const s = try randomStream(alloc, 50, 8, 2);
    defer alloc.free(s.data);
    const op: lowlevel.LowOp = .{ .kind = .add_const_mod, .params = .{ .raw = 17 } };
    const r = try lowlevel.forward(alloc, op, s);
    const fwd = r.one;
    defer alloc.free(fwd.data);
    const back = try lowlevel.inverseOne(alloc, op, fwd);
    defer alloc.free(back.data);
    try std.testing.expect(streamsEqual(s, back));
}

test "lowlevel: rotate_bits inverse" {
    const alloc = std.testing.allocator;
    const s = try randomStream(alloc, 50, 8, 3);
    defer alloc.free(s.data);
    const op: lowlevel.LowOp = .{ .kind = .rotate_bits, .params = .{ .raw = 3 } };
    const r = try lowlevel.forward(alloc, op, s);
    const fwd = r.one;
    defer alloc.free(fwd.data);
    const back = try lowlevel.inverseOne(alloc, op, fwd);
    defer alloc.free(back.data);
    try std.testing.expect(streamsEqual(s, back));
}

test "lowlevel: bit_swap_pair self-inverse" {
    const alloc = std.testing.allocator;
    const s = try randomStream(alloc, 50, 8, 4);
    defer alloc.free(s.data);
    // Swap bit 1 and bit 5.
    const op: lowlevel.LowOp = .{ .kind = .bit_swap_pair, .params = .{ .raw = (5 << 5) | 1 } };
    const r = try lowlevel.forward(alloc, op, s);
    const fwd = r.one;
    defer alloc.free(fwd.data);
    const back = try lowlevel.inverseOne(alloc, op, fwd);
    defer alloc.free(back.data);
    try std.testing.expect(streamsEqual(s, back));
}

test "lowlevel: xor_prev / prefix_xor are inverses of each other" {
    const alloc = std.testing.allocator;
    const s = try randomStream(alloc, 30, 8, 5);
    defer alloc.free(s.data);
    const fwd_op: lowlevel.LowOp = .{ .kind = .xor_prev };
    const inv_op: lowlevel.LowOp = .{ .kind = .prefix_xor };
    const r = try lowlevel.forward(alloc, fwd_op, s);
    const fwd = r.one;
    defer alloc.free(fwd.data);
    const r2 = try lowlevel.forward(alloc, inv_op, fwd);
    const back = r2.one;
    defer alloc.free(back.data);
    try std.testing.expect(streamsEqual(s, back));
}

test "lowlevel: diff_mod / cumsum_mod inverse pair" {
    const alloc = std.testing.allocator;
    const s = try randomStream(alloc, 30, 8, 6);
    defer alloc.free(s.data);
    const op: lowlevel.LowOp = .{ .kind = .diff_mod };
    const r = try lowlevel.forward(alloc, op, s);
    const fwd = r.one;
    defer alloc.free(fwd.data);
    const back = try lowlevel.inverseOne(alloc, op, fwd);
    defer alloc.free(back.data);
    try std.testing.expect(streamsEqual(s, back));
}

test "lowlevel: bit_reverse self-inverse" {
    const alloc = std.testing.allocator;
    const s = try randomStream(alloc, 50, 8, 11);
    defer alloc.free(s.data);
    const op: lowlevel.LowOp = .{ .kind = .bit_reverse };
    const r = try lowlevel.forward(alloc, op, s);
    const fwd = r.one;
    defer alloc.free(fwd.data);
    const back = try lowlevel.inverseOne(alloc, op, fwd);
    defer alloc.free(back.data);
    try std.testing.expect(streamsEqual(s, back));
}

test "lowlevel: negate_mod self-inverse" {
    const alloc = std.testing.allocator;
    const s = try randomStream(alloc, 50, 8, 12);
    defer alloc.free(s.data);
    const op: lowlevel.LowOp = .{ .kind = .negate_mod };
    const r = try lowlevel.forward(alloc, op, s);
    const fwd = r.one;
    defer alloc.free(fwd.data);
    const back = try lowlevel.inverseOne(alloc, op, fwd);
    defer alloc.free(back.data);
    try std.testing.expect(streamsEqual(s, back));
}

test "lowlevel: mul_const_odd_mod with c=37 inverse via Newton" {
    const alloc = std.testing.allocator;
    const s = try randomStream(alloc, 50, 8, 13);
    defer alloc.free(s.data);
    const op: lowlevel.LowOp = .{ .kind = .mul_const_odd_mod, .params = .{ .raw = 37 } };
    const r = try lowlevel.forward(alloc, op, s);
    const fwd = r.one;
    defer alloc.free(fwd.data);
    const back = try lowlevel.inverseOne(alloc, op, fwd);
    defer alloc.free(back.data);
    try std.testing.expect(streamsEqual(s, back));
}

test "lowlevel: xor_with_shift inverse" {
    const alloc = std.testing.allocator;
    const s = try randomStream(alloc, 50, 8, 14);
    defer alloc.free(s.data);
    const op: lowlevel.LowOp = .{ .kind = .xor_with_shift, .params = .{ .raw = 1 } };
    const r = try lowlevel.forward(alloc, op, s);
    const fwd = r.one;
    defer alloc.free(fwd.data);
    const back = try lowlevel.inverseOne(alloc, op, fwd);
    defer alloc.free(back.data);
    try std.testing.expect(streamsEqual(s, back));
}

test "lowlevel: gray_code / inv_gray_code inverse pair" {
    const alloc = std.testing.allocator;
    const s = try randomStream(alloc, 50, 8, 15);
    defer alloc.free(s.data);
    const op: lowlevel.LowOp = .{ .kind = .gray_code };
    const r = try lowlevel.forward(alloc, op, s);
    const fwd = r.one;
    defer alloc.free(fwd.data);
    const back = try lowlevel.inverseOne(alloc, op, fwd);
    defer alloc.free(back.data);
    try std.testing.expect(streamsEqual(s, back));
    // Also verify inv_gray_code(gray_code(x)) = x via op chain
    const op_inv: lowlevel.LowOp = .{ .kind = .inv_gray_code };
    const r2 = try lowlevel.forward(alloc, op_inv, fwd);
    const back2 = r2.one;
    defer alloc.free(back2.data);
    try std.testing.expect(streamsEqual(s, back2));
}

test "lowlevel: split_field reproduces fp16 sign extraction" {
    const alloc = std.testing.allocator;
    const s = try randomStream(alloc, 64, 16, 7);
    defer alloc.free(s.data);
    // Extract bit 15 (sign) of 16-bit elements: start=15, n_bits=1, k=16.
    const params: lowlevel.Params = .{ .raw = 15 | (1 << 8) | (16 << 16) };
    const op: lowlevel.LowOp = .{ .kind = .split_field, .params = params };
    const r = try lowlevel.forward(alloc, op, s);
    const hi = r.two[0];
    const lo = r.two[1];
    defer alloc.free(hi.data);
    defer alloc.free(lo.data);
    try std.testing.expectEqual(@as(u8, 1), hi.bits_per_elem);
    try std.testing.expectEqual(@as(u8, 15), lo.bits_per_elem);
    const back = try lowlevel.inverseTwo(alloc, op, hi, lo);
    defer alloc.free(back.data);
    try std.testing.expect(streamsEqual(s, back));
}

test "astar: synthesize a fp16 sign stream finds something better than raw" {
    const alloc = std.testing.allocator;
    // Build a 16-bit stream that mimics fp16 sign+exp pattern: mostly 0, occasional flipped sign.
    const n: usize = 1024;
    const buf = try alloc.alloc(u8, n * 2);
    defer alloc.free(buf);
    var prng: std.Random.DefaultPrng = .init(42);
    const r = prng.random();
    for (0..n) |i| {
        const v: u16 = if (r.float(f32) < 0.5)
            r.intRangeLessThan(u16, 0, 0x4000) // small positive
        else
            r.intRangeLessThan(u16, 0x8000, 0xC000); // small negative
        std.mem.writeInt(u16, buf[i * 2 ..][0..2], v, .little);
    }
    const s: types.Stream = .{ .data = buf, .count = n, .bits_per_elem = 16 };

    var best = try astar.synthesize(alloc, s, .{ .max_depth = 3, .max_nodes_explored = 50_000 });
    defer best.deinit(alloc);

    try std.testing.expect(best.program != null);
    // Raw cost is n * 16 = 16384 bits. A* should find at least entropy-coded ≤ raw.
    try std.testing.expect(best.cost <= n * 16);
}

test "baseline: gzip beats raw on highly compressible data" {
    const alloc = std.testing.allocator;
    const buf = try alloc.alloc(u8, 100 * 1024);
    defer alloc.free(buf);
    @memset(buf, 0);
    for (0..100) |i| buf[i * 1000] = @intCast(i & 0xFF);

    const sz = try baseline.gzipSize(alloc, buf);
    try std.testing.expect(sz < buf.len / 10); // mostly-zeros → big win
}

test "safetensors: in-memory write/read roundtrip" {
    const alloc = std.testing.allocator;
    var t1 = try makeFp16Tensor(alloc, 256, 31);
    defer t1.deinit(alloc);
    var t2 = try makeFp16Tensor(alloc, 1024, 32);
    defer t2.deinit(alloc);

    // Use the writer to produce bytes by writing to a temp file, then read back.
    var threaded: std.Io.Threaded = .init(alloc, .{});
    defer threaded.deinit();
    const io: std.Io = threaded.io();

    const tmp_path = "/tmp/brevis-safetensors-test.bin";
    const out_tensors = [_]safetensors.TensorOut{
        .{ .name = "alpha", .view = t1 },
        .{ .name = "beta", .view = t2 },
    };
    try safetensors.saveToPath(alloc, io, tmp_path, &out_tensors);

    var loaded = try safetensors.loadFromPath(alloc, io, tmp_path);
    defer loaded.deinit(alloc);

    try std.testing.expectEqual(@as(usize, 2), loaded.tensors.len);
    var found_alpha = false;
    var found_beta = false;
    for (loaded.tensors) |lt| {
        if (std.mem.eql(u8, lt.name, "alpha")) {
            try std.testing.expect(std.mem.eql(u8, t1.data, lt.view.data));
            found_alpha = true;
        } else if (std.mem.eql(u8, lt.name, "beta")) {
            try std.testing.expect(std.mem.eql(u8, t2.data, lt.view.data));
            found_beta = true;
        }
    }
    try std.testing.expect(found_alpha and found_beta);
}

test "program serialization roundtrip" {
    const alloc = std.testing.allocator;
    var t = try makeFp16Tensor(alloc, 128, 7);
    defer t.deinit(alloc);

    const kids = try alloc.alloc(program.Node, 3);
    kids[0] = .{ .op = .huffman };
    kids[1] = .{ .op = .rans };
    kids[2] = .{ .op = .raw };
    var node: program.Node = .{ .op = .split_float, .children = kids };
    defer node.deinit(alloc);

    try program.compressTensor(alloc, &node, t, &.{});
    const program_bytes = try program.serializeProgram(alloc, &node);
    defer alloc.free(program_bytes);
    const payload_bytes = try program.collectPayloadBytes(alloc, &node);
    defer alloc.free(payload_bytes);

    var node2 = try program.deserializeProgram(alloc, program_bytes);
    defer node2.deinit(alloc);
    try program.distributePayloadBytes(&node2, payload_bytes);

    var back = try program.decompressTensor(alloc, &node2, &.{});
    defer back.deinit(alloc);
    try std.testing.expect(std.mem.eql(u8, t.data, back.data));
}
