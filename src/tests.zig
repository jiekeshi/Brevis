const std = @import("std");
const types = @import("types.zig");
const codec = @import("codec.zig");
const safetensors = @import("safetensors.zig");
const baseline = @import("baseline.zig");
const lowlevel = @import("lowlevel.zig");
const astar = @import("astar.zig");
const pnode_archive = @import("pnode_archive.zig");
const phog_mod = @import("phog.zig");
const grammar = @import("grammar.zig");

fn uniformPhog() phog_mod.PHOG {
    var p = phog_mod.PHOG.empty();
    p.computeFixpoint();
    return p;
}

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

fn randomStream(alloc: std.mem.Allocator, count: usize, bpe: u8, seed: u64) !types.Stream {
    const elem_bytes: usize = switch (types.roundUpToPow2(bpe)) {
        8 => 1,
        16 => 2,
        32 => 4,
        else => unreachable,
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

// ============== codec ==============

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

// ============== low-level primitives reversibility ==============

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

// ============== unified A* + realize + decompress ==============

test "astar: realize + decompress roundtrip on a fp16 sign-pattern stream" {
    const alloc = std.testing.allocator;
    const n: usize = 1024;
    const buf = try alloc.alloc(u8, n * 2);
    defer alloc.free(buf);
    var prng: std.Random.DefaultPrng = .init(43);
    const r = prng.random();
    for (0..n) |i| {
        const v: u16 = if (r.float(f32) < 0.5)
            r.intRangeLessThan(u16, 0, 0x4000)
        else
            r.intRangeLessThan(u16, 0x8000, 0xC000);
        std.mem.writeInt(u16, buf[i * 2 ..][0..2], v, .little);
    }
    const s: types.Stream = .{ .data = buf, .count = n, .bits_per_elem = 16 };

    const phog = uniformPhog();
    var best = try astar.synthesize(alloc, s, phog, .{ .max_pops = 5_000, .realize_top_k = 3 });
    defer best.deinit(alloc);
    try std.testing.expect(best.program != null);

    // synthesize already realized — verify by decompress alone.
    const back = try astar.decompress(alloc, best.program.?);
    defer alloc.free(back.data);
    try std.testing.expectEqual(n, back.count);
    try std.testing.expect(std.mem.eql(u8, buf, back.data));
}

test "astar: fp32 realize + decompress roundtrip (near-constant)" {
    const alloc = std.testing.allocator;
    const n: usize = 1024;
    const buf = try alloc.alloc(u8, n * 4);
    defer alloc.free(buf);
    var prng: std.Random.DefaultPrng = .init(7);
    const r = prng.random();
    for (0..n) |i| {
        const v: f32 = 1.0 + r.floatNorm(f32) * 0.01; // near-constant ⇒ skewed exponent
        const u: u32 = @bitCast(v);
        std.mem.writeInt(u32, buf[i * 4 ..][0..4], u, .little);
    }
    const s: types.Stream = .{ .data = buf, .count = n, .bits_per_elem = 32 };
    const phog = uniformPhog();
    var best = try astar.synthesize(alloc, s, phog, .{ .max_pops = 20_000, .realize_top_k = 6 });
    defer best.deinit(alloc);
    try std.testing.expect(best.program != null);
    // The bpe-aware search must reconstruct 32-bit words bit-exactly.
    const back = try astar.decompress(alloc, best.program.?);
    defer alloc.free(back.data);
    try std.testing.expectEqual(n, back.count);
    try std.testing.expect(std.mem.eql(u8, buf, back.data));
    // Near-constant fp32 has a skewed exponent ⇒ must beat raw (32 bits/elem).
    try std.testing.expect(best.cost_bits < n * 32);
}

test "astar: synthesize a fp16 sign stream finds something better than raw" {
    const alloc = std.testing.allocator;
    const n: usize = 1024;
    const buf = try alloc.alloc(u8, n * 2);
    defer alloc.free(buf);
    var prng: std.Random.DefaultPrng = .init(42);
    const r = prng.random();
    for (0..n) |i| {
        const v: u16 = if (r.float(f32) < 0.5)
            r.intRangeLessThan(u16, 0, 0x4000)
        else
            r.intRangeLessThan(u16, 0x8000, 0xC000);
        std.mem.writeInt(u16, buf[i * 2 ..][0..2], v, .little);
    }
    const s: types.Stream = .{ .data = buf, .count = n, .bits_per_elem = 16 };
    const phog = uniformPhog();
    var best = try astar.synthesize(alloc, s, phog, .{ .max_pops = 5_000, .realize_top_k = 3 });
    defer best.deinit(alloc);
    try std.testing.expect(best.program != null);
    try std.testing.expect(best.cost_bits <= n * 16);
}

test "e2e: archive via file" {
    const alloc = std.testing.allocator;
    var t1 = try makeFp16Tensor(alloc, 65536, 99); // 128KB ~= fixture attn_proj size
    defer t1.deinit(alloc);
    var t2 = try makeFp16Tensor(alloc, 4096, 100);
    defer t2.deinit(alloc);
    var t3 = try makeFp16Tensor(alloc, 262144, 101); // 512KB
    defer t3.deinit(alloc);
    const stream1: types.Stream = .{ .data = t1.data, .count = 65536, .bits_per_elem = 16 };
    const stream2: types.Stream = .{ .data = t2.data, .count = 4096, .bits_per_elem = 16 };
    const stream3: types.Stream = .{ .data = t3.data, .count = 262144, .bits_per_elem = 16 };
    const phog = uniformPhog();
    var best1 = try astar.synthesize(alloc, stream1, phog, .{ .max_pops = 8_000, .realize_top_k = 4 });
    defer best1.deinit(alloc);
    var best2 = try astar.synthesize(alloc, stream2, phog, .{ .max_pops = 8_000, .realize_top_k = 4 });
    defer best2.deinit(alloc);
    var best3 = try astar.synthesize(alloc, stream3, phog, .{ .max_pops = 8_000, .realize_top_k = 4 });
    defer best3.deinit(alloc);
    const jobs = [_]pnode_archive.TensorJob{
        .{ .name = "a", .dtype = t1.dtype, .shape = t1.shape, .program = best1.program.? },
        .{ .name = "b", .dtype = t2.dtype, .shape = t2.shape, .program = best2.program.? },
        .{ .name = "c", .dtype = t3.dtype, .shape = t3.shape, .program = best3.program.? },
    };
    const bytes = try pnode_archive.buildArchiveBytes(alloc, &jobs);
    defer alloc.free(bytes);

    // Write + read back from file
    var threaded: std.Io.Threaded = .init(alloc, .{});
    defer threaded.deinit();
    const io: std.Io = threaded.io();
    const cwd = std.Io.Dir.cwd();
    const tmp = "/tmp/brevis-e2e-archive.bin";
    const fw = try cwd.createFile(io, tmp, .{});
    var wb: [4096]u8 = undefined;
    var wf = fw.writer(io, &wb);
    try wf.interface.writeAll(bytes);
    try wf.interface.flush();
    fw.close(io);

    const fr = try cwd.openFile(io, tmp, .{});
    const stat = try fr.stat(io);
    const got = try alloc.alloc(u8, @intCast(stat.size));
    defer alloc.free(got);
    var rb: [4096]u8 = undefined;
    var rdr = fr.reader(io, &rb);
    _ = try rdr.interface.readSliceAll(got);
    fr.close(io);

    try std.testing.expect(std.mem.eql(u8, bytes, got));
    var parsed = try pnode_archive.parseArchive(alloc, got);
    defer parsed.deinit(alloc);
    for (parsed.tensors) |*pt| {
        const back = try astar.decompress(alloc, pt.program);
        defer alloc.free(back.data);
        const orig: []const u8 = if (std.mem.eql(u8, pt.name, "a")) t1.data
            else if (std.mem.eql(u8, pt.name, "b")) t2.data
            else t3.data;
        try std.testing.expect(std.mem.eql(u8, orig, back.data));
    }
}

test "e2e: archive roundtrip with larger tensors (CLI-like)" {
    const alloc = std.testing.allocator;
    var t1 = try makeFp16Tensor(alloc, 4096, 99);
    defer t1.deinit(alloc);
    var t2 = try makeFp16Tensor(alloc, 16384, 100);
    defer t2.deinit(alloc);
    const stream1: types.Stream = .{ .data = t1.data, .count = 4096, .bits_per_elem = 16 };
    const stream2: types.Stream = .{ .data = t2.data, .count = 16384, .bits_per_elem = 16 };
    const phog = uniformPhog();
    var best1 = try astar.synthesize(alloc, stream1, phog, .{ .max_pops = 8_000, .realize_top_k = 4 });
    var best2 = try astar.synthesize(alloc, stream2, phog, .{ .max_pops = 8_000, .realize_top_k = 4 });
    const jobs = [_]pnode_archive.TensorJob{
        .{ .name = "a", .dtype = t1.dtype, .shape = t1.shape, .program = best1.program.? },
        .{ .name = "b", .dtype = t2.dtype, .shape = t2.shape, .program = best2.program.? },
    };
    const bytes = try pnode_archive.buildArchiveBytes(alloc, &jobs);
    defer alloc.free(bytes);
    defer best1.deinit(alloc);
    defer best2.deinit(alloc);
    var parsed = try pnode_archive.parseArchive(alloc, bytes);
    defer parsed.deinit(alloc);
    for (parsed.tensors) |*pt| {
        const stream = try astar.decompress(alloc, pt.program);
        defer alloc.free(stream.data);
        const orig: []const u8 = if (std.mem.eql(u8, pt.name, "a")) t1.data else t2.data;
        try std.testing.expect(std.mem.eql(u8, orig, stream.data));
    }
}

test "pnode_archive: 2-tensor v2 archive roundtrip bit-exact" {
    const alloc = std.testing.allocator;
    var t1 = try makeFp16Tensor(alloc, 256, 31);
    defer t1.deinit(alloc);
    var t2 = try makeFp16Tensor(alloc, 1024, 32);
    defer t2.deinit(alloc);

    // Synthesize each (synthesize already realizes).
    const stream1: types.Stream = .{ .data = t1.data, .count = 256, .bits_per_elem = 16 };
    const stream2: types.Stream = .{ .data = t2.data, .count = 1024, .bits_per_elem = 16 };
    const phog = uniformPhog();
    var best1 = try astar.synthesize(alloc, stream1, phog, .{ .max_pops = 5_000, .realize_top_k = 3 });
    defer best1.deinit(alloc);
    var best2 = try astar.synthesize(alloc, stream2, phog, .{ .max_pops = 5_000, .realize_top_k = 3 });
    defer best2.deinit(alloc);

    const jobs = [_]pnode_archive.TensorJob{
        .{ .name = "alpha", .dtype = t1.dtype, .shape = t1.shape, .program = best1.program.? },
        .{ .name = "beta", .dtype = t2.dtype, .shape = t2.shape, .program = best2.program.? },
    };
    const bytes = try pnode_archive.buildArchiveBytes(alloc, &jobs);
    defer alloc.free(bytes);

    var parsed = try pnode_archive.parseArchive(alloc, bytes);
    defer parsed.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 2), parsed.tensors.len);

    for (parsed.tensors) |*pt| {
        const stream = try astar.decompress(alloc, pt.program);
        defer alloc.free(stream.data);
        const orig: []const u8 = if (std.mem.eql(u8, pt.name, "alpha")) t1.data else t2.data;
        try std.testing.expect(std.mem.eql(u8, orig, stream.data));
    }
    // Free programs that were freshly built during synthesize.
    best1.program = jobs[0].program;
    best2.program = jobs[1].program;
    best1.deinit(alloc);
    best2.deinit(alloc);
}

// ============== misc ==============

test "baseline: gzip beats raw on highly compressible data" {
    const alloc = std.testing.allocator;
    const buf = try alloc.alloc(u8, 100 * 1024);
    defer alloc.free(buf);
    @memset(buf, 0);
    for (0..100) |i| buf[i * 1000] = @intCast(i & 0xFF);
    const sz = try baseline.gzipSize(alloc, buf);
    try std.testing.expect(sz < buf.len / 10);
}

test "safetensors: in-memory write/read roundtrip" {
    const alloc = std.testing.allocator;
    var t1 = try makeFp16Tensor(alloc, 256, 31);
    defer t1.deinit(alloc);
    var t2 = try makeFp16Tensor(alloc, 1024, 32);
    defer t2.deinit(alloc);

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
