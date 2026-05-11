//! brevis — unified low-level program-synthesis compressor.
//!
//! Subcommands:
//!   brevis compress     <model.safetensors>  <out.brv>
//!   brevis decompress   <model.brv>          <out.safetensors>
//!   brevis verify       <model.brv>          <orig.safetensors>
//!   brevis bench        <model.safetensors>     [synthesize, no archive write]
//!   brevis baseline     <model.safetensors>     [compare brevis vs gzip vs zstd]
//!   brevis demo
//!   brevis make-fixture <out.safetensors>
//!   brevis train        <model.safetensors>  <out.json>     [B&B + macro mining]

const std = @import("std");
const types = @import("types.zig");
const codec = @import("codec.zig");
const safetensors = @import("safetensors.zig");
const baseline_mod = @import("baseline.zig");
const lowlevel = @import("lowlevel.zig");
const lowlevel_training = @import("lowlevel_training.zig");
const astar = @import("astar.zig");
const pnode_archive = @import("pnode_archive.zig");

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &stdout_buf);
    var stderr_buf: [4096]u8 = undefined;
    var stderr = std.Io.File.stderr().writer(io, &stderr_buf);
    const out = &stdout.interface;
    const err = &stderr.interface;

    var args_list: std.ArrayList([]u8) = .empty;
    defer {
        for (args_list.items) |a| alloc.free(a);
        args_list.deinit(alloc);
    }
    var it = init.minimal.args.iterate();
    defer it.deinit();
    while (it.next()) |a| {
        const owned = try alloc.alloc(u8, a.len);
        @memcpy(owned, a);
        try args_list.append(alloc, owned);
    }

    if (args_list.items.len < 2) return usageExit(err);

    const cmd = args_list.items[1];
    if (std.mem.eql(u8, cmd, "compress")) {
        if (args_list.items.len != 4) return usageExit(err);
        try cmdCompress(io, out, args_list.items[2], args_list.items[3]);
    } else if (std.mem.eql(u8, cmd, "decompress")) {
        if (args_list.items.len != 4) return usageExit(err);
        try cmdDecompress(io, out, args_list.items[2], args_list.items[3]);
    } else if (std.mem.eql(u8, cmd, "verify")) {
        if (args_list.items.len != 4) return usageExit(err);
        try cmdVerify(io, out, args_list.items[2], args_list.items[3]);
    } else if (std.mem.eql(u8, cmd, "bench")) {
        if (args_list.items.len != 3) return usageExit(err);
        try cmdBench(io, out, args_list.items[2]);
    } else if (std.mem.eql(u8, cmd, "baseline")) {
        if (args_list.items.len != 3) return usageExit(err);
        try cmdBaseline(io, out, args_list.items[2]);
    } else if (std.mem.eql(u8, cmd, "demo")) {
        try cmdDemo(out);
    } else if (std.mem.eql(u8, cmd, "make-fixture")) {
        if (args_list.items.len != 3) return usageExit(err);
        try cmdMakeFixture(io, out, args_list.items[2]);
    } else if (std.mem.eql(u8, cmd, "train")) {
        if (args_list.items.len != 4) return usageExit(err);
        try cmdTrain(io, out, args_list.items[2], args_list.items[3]);
    } else {
        try err.print("brevis: unknown command '{s}'\n", .{cmd});
        return usageExit(err);
    }
    try out.flush();
}

fn usageExit(err: *std.Io.Writer) !void {
    try err.writeAll(
        \\brevis 0.2.0 — unified low-level program-synthesis compressor
        \\
        \\Usage:
        \\  brevis compress     <model.safetensors>  <out.brv>
        \\  brevis decompress   <model.brv>          <out.safetensors>
        \\  brevis verify       <model.brv>          <orig.safetensors>
        \\  brevis bench        <model.safetensors>
        \\  brevis baseline     <model.safetensors>     [vs gzip / zstd]
        \\  brevis demo
        \\  brevis make-fixture <out.safetensors>
        \\  brevis train        <model.safetensors>  <out.json>
        \\
    );
    try err.flush();
    std.process.exit(2);
}

// ============== compress (parallel, low-level via astar + macros) ==============

fn countPayloadBits(node: *const astar.PNode) u64 {
    return switch (node.*) {
        .terminal => |t| @as(u64, t.payload.len) * 8,
        .chain => |c| countPayloadBits(c.next),
        .split => |s| countPayloadBits(s.hi) + countPayloadBits(s.lo),
        .macro => 0,
    };
}

const SynthOut = struct {
    prog: ?*astar.PNode = null,
    dtype: types.Dtype = .f16,
    shape: []const u64 = &.{},
    name: []const u8 = &.{},
    raw_bits: u64 = 0,
    payload_bits: u64 = 0,
};

const SynthWorker = struct {
    next: std.atomic.Value(usize),
    tensors: []const safetensors.Tensor,
    outs: []SynthOut,

    fn run(self: *@This()) void {
        const wa = std.heap.smp_allocator;
        while (true) {
            const i = self.next.fetchAdd(1, .acq_rel);
            if (i >= self.tensors.len) return;
            const t = self.tensors[i];
            if (!t.view.dtype.isFloat16Like()) continue;
            const count: usize = @divExact(t.view.data.len, 2);
            const stream: types.Stream = .{
                .data = t.view.data,
                .count = count,
                .bits_per_elem = 16,
                .owns_data = false,
            };
            // depth 1 + built-in macros = O(1)-ish per tensor (the macros are
            // atomic actions that already encode 3-deep structure).
            var best = astar.synthesize(wa, stream, .{
                .max_depth = 1,
                .max_nodes_explored = 20_000,
            }) catch continue;
            const prog = best.program orelse continue;
            best.program = null;
            astar.realize(wa, prog, stream) catch continue;
            self.outs[i] = .{
                .prog = prog,
                .dtype = t.view.dtype,
                .shape = t.view.shape,
                .name = t.name,
                .raw_bits = @as(u64, t.view.data.len) * 8,
                .payload_bits = countPayloadBits(prog),
            };
        }
    }
};

fn synthesizeAll(work_alloc: std.mem.Allocator, io: std.Io, tensors: []const safetensors.Tensor) ![]SynthOut {
    const n_threads = std.Thread.getCpuCount() catch 8;
    const outs = try work_alloc.alloc(SynthOut, tensors.len);
    for (outs) |*o| o.* = .{};
    var worker: SynthWorker = .{
        .next = .init(0),
        .tensors = tensors,
        .outs = outs,
    };
    const threads = try work_alloc.alloc(std.Thread, n_threads);
    defer work_alloc.free(threads);
    for (threads) |*th| th.* = try std.Thread.spawn(.{}, SynthWorker.run, .{&worker});
    for (threads) |th| th.join();
    _ = io;
    return outs;
}

fn freeOuts(work_alloc: std.mem.Allocator, outs: []SynthOut) void {
    for (outs) |o| if (o.prog) |p| {
        p.deinit(work_alloc);
        work_alloc.destroy(p);
    };
    work_alloc.free(outs);
}

fn cmdCompress(io: std.Io, out: *std.Io.Writer, in_path: []const u8, out_path: []const u8) !void {
    const wa = std.heap.smp_allocator;
    var loaded = try safetensors.loadFromPath(wa, io, in_path);
    defer loaded.deinitMmap(wa, io);
    const t_start = std.Io.Timestamp.now(io, .awake);
    try out.print("compress: {d} tensors\n", .{loaded.tensors.len});
    try out.flush();

    const outs = try synthesizeAll(wa, io, loaded.tensors);
    defer freeOuts(wa, outs);

    var jobs: std.ArrayList(pnode_archive.TensorJob) = .empty;
    defer jobs.deinit(wa);
    var total_raw: u64 = 0;
    var total_bits: u64 = 0;
    for (outs) |o| {
        if (o.prog == null) continue;
        try jobs.append(wa, .{
            .name = o.name,
            .dtype = o.dtype,
            .shape = o.shape,
            .program = o.prog.?,
        });
        total_raw += o.raw_bits;
        total_bits += o.payload_bits;
    }

    const bytes = try pnode_archive.buildArchiveBytes(wa, jobs.items);
    defer wa.free(bytes);
    const cwd = std.Io.Dir.cwd();
    const f = try cwd.createFile(io, out_path, .{});
    defer f.close(io);
    var wb: [4096]u8 = undefined;
    var wf = f.writer(io, &wb);
    const chunk: usize = 1 << 30;
    var off: usize = 0;
    while (off < bytes.len) {
        const n = @min(chunk, bytes.len - off);
        try wf.interface.writeAll(bytes[off .. off + n]);
        off += n;
    }
    try wf.interface.flush();

    const elapsed = t_start.durationTo(.now(io, .awake)).toMilliseconds();
    try out.print("wrote {s} ({d} bytes) in {d}ms\n", .{ out_path, bytes.len, elapsed });
    if (total_raw > 0) {
        const ratio: f64 = @as(f64, @floatFromInt(total_raw)) / @as(f64, @floatFromInt(total_bits));
        try out.print("ratio: {d:.3}x ({d} → {d} bits)\n", .{ ratio, total_raw, total_bits });
    }
}

// ============== decompress ==============

fn cmdDecompress(io: std.Io, out: *std.Io.Writer, in_path: []const u8, out_path: []const u8) !void {
    const wa = std.heap.smp_allocator;
    const cwd = std.Io.Dir.cwd();
    const f = try cwd.openFile(io, in_path, .{});
    defer f.close(io);
    const stat = try f.stat(io);
    const bytes = try wa.alloc(u8, @intCast(stat.size));
    defer wa.free(bytes);
    var rb: [4096]u8 = undefined;
    var rdr = f.reader(io, &rb);
    _ = try rdr.interface.readSliceAll(bytes);

    var parsed = try pnode_archive.parseArchive(wa, bytes);
    defer parsed.deinit(wa);

    var out_tensors: std.ArrayList(safetensors.TensorOut) = .empty;
    var owned_data: std.ArrayList([]u8) = .empty;
    defer {
        for (owned_data.items) |d| wa.free(d);
        owned_data.deinit(wa);
        out_tensors.deinit(wa);
    }
    for (parsed.tensors) |*pt| {
        const stream = try astar.decompress(wa, pt.program);
        try owned_data.append(wa, stream.data);
        const view: types.TensorView = .{
            .data = stream.data,
            .shape = pt.shape,
            .dtype = pt.dtype,
            .owns_data = false,
            .owns_shape = false,
        };
        try out_tensors.append(wa, .{ .name = pt.name, .view = view });
    }
    try safetensors.saveToPath(wa, io, out_path, out_tensors.items);
    try out.print("decompress: wrote {d} tensors → {s}\n", .{ parsed.tensors.len, out_path });
}

// ============== verify ==============

fn cmdVerify(io: std.Io, out: *std.Io.Writer, brv_path: []const u8, orig_path: []const u8) !void {
    const wa = std.heap.smp_allocator;
    const cwd = std.Io.Dir.cwd();
    const f = try cwd.openFile(io, brv_path, .{});
    defer f.close(io);
    const stat = try f.stat(io);
    const bytes = try wa.alloc(u8, @intCast(stat.size));
    defer wa.free(bytes);
    var rb: [4096]u8 = undefined;
    var rdr = f.reader(io, &rb);
    _ = try rdr.interface.readSliceAll(bytes);

    var parsed = try pnode_archive.parseArchive(wa, bytes);
    defer parsed.deinit(wa);
    var orig = try safetensors.loadFromPath(wa, io, orig_path);
    defer orig.deinitMmap(wa, io);

    var ok: usize = 0;
    var fail: usize = 0;
    for (parsed.tensors) |*pt| {
        var found: ?types.TensorView = null;
        for (orig.tensors) |ot| if (std.mem.eql(u8, ot.name, pt.name)) {
            found = ot.view;
            break;
        };
        if (found == null) {
            try out.print("  MISSING in original: {s}\n", .{pt.name});
            fail += 1;
            continue;
        }
        const stream = try astar.decompress(wa, pt.program);
        defer wa.free(stream.data);
        if (std.mem.eql(u8, stream.data, found.?.data)) {
            ok += 1;
        } else {
            try out.print("  MISMATCH: {s}\n", .{pt.name});
            fail += 1;
        }
    }
    try out.print("verify: {d}/{d} bit-exact\n", .{ ok, ok + fail });
    if (fail > 0) std.process.exit(1);
}

// ============== bench (synthesize without writing archive) ==============

fn cmdBench(io: std.Io, out: *std.Io.Writer, in_path: []const u8) !void {
    const wa = std.heap.smp_allocator;
    var loaded = try safetensors.loadFromPath(wa, io, in_path);
    defer loaded.deinitMmap(wa, io);
    try out.print("=== brevis bench: {s} ({d} tensors) ===\n", .{ in_path, loaded.tensors.len });
    try out.flush();

    const t_start = std.Io.Timestamp.now(io, .awake);
    const outs = try synthesizeAll(wa, io, loaded.tensors);
    defer freeOuts(wa, outs);
    const elapsed = t_start.durationTo(.now(io, .awake)).toMilliseconds();

    var total_raw: u64 = 0;
    var total_bits: u64 = 0;
    for (outs) |o| {
        if (o.prog == null) continue;
        total_raw += o.raw_bits;
        total_bits += o.payload_bits;
    }
    if (total_raw > 0) {
        const ratio: f64 = @as(f64, @floatFromInt(total_raw)) / @as(f64, @floatFromInt(total_bits));
        try out.print("overall: {d} → {d} bits ({d:.3}x) in {d}ms wall\n", .{ total_raw, total_bits, ratio, elapsed });
    }
}

// ============== baseline (vs gzip / zstd) ==============

fn cmdBaseline(io: std.Io, out: *std.Io.Writer, in_path: []const u8) !void {
    const wa = std.heap.smp_allocator;
    var loaded = try safetensors.loadFromPath(wa, io, in_path);
    defer loaded.deinitMmap(wa, io);

    var raw_total: usize = 0;
    for (loaded.tensors) |t| raw_total += t.view.data.len;
    const raw_concat = try wa.alloc(u8, raw_total);
    defer wa.free(raw_concat);
    var off: usize = 0;
    for (loaded.tensors) |t| {
        @memcpy(raw_concat[off .. off + t.view.data.len], t.view.data);
        off += t.view.data.len;
    }

    try out.print("=== brevis baseline: {s} ({d} tensors, {d} bytes raw) ===\n\n", .{ in_path, loaded.tensors.len, raw_total });

    // brevis (synth + archive)
    const outs = try synthesizeAll(wa, io, loaded.tensors);
    defer freeOuts(wa, outs);
    var jobs: std.ArrayList(pnode_archive.TensorJob) = .empty;
    defer jobs.deinit(wa);
    for (outs) |o| {
        if (o.prog == null) continue;
        try jobs.append(wa, .{ .name = o.name, .dtype = o.dtype, .shape = o.shape, .program = o.prog.? });
    }
    const brevis_bytes = try pnode_archive.buildArchiveBytes(wa, jobs.items);
    defer wa.free(brevis_bytes);

    const gz_size = try baseline_mod.gzipSize(wa, raw_concat);
    const zstd_3 = try baseline_mod.zstdSize(wa, io, raw_concat, 3);
    const zstd_19 = try baseline_mod.zstdSize(wa, io, raw_concat, 19);

    const f = struct {
        fn ratio(orig: usize, comp: usize) f64 {
            return @as(f64, @floatFromInt(orig)) / @as(f64, @floatFromInt(comp));
        }
    };

    try out.writeAll("                       size (bytes)        ratio\n");
    try out.writeAll("                       -------------       --------\n");
    try out.print("raw                    {d:>13}       1.000x\n", .{raw_total});
    try out.print("gzip -9                {d:>13}       {d:.3}x  (DEFLATE)\n", .{ gz_size, f.ratio(raw_total, gz_size) });
    if (zstd_3) |z| try out.print("zstd -3                {d:>13}       {d:.3}x  (zstd default)\n", .{ z, f.ratio(raw_total, z) });
    if (zstd_19) |z| try out.print("zstd -19               {d:>13}       {d:.3}x  (zstd best)\n", .{ z, f.ratio(raw_total, z) });
    try out.print("brevis (.brv archive)  {d:>13}       {d:.3}x  (low-level synth + macros)\n", .{ brevis_bytes.len, f.ratio(raw_total, brevis_bytes.len) });
}

// ============== train (low-level B&B + macro mining) ==============

fn cmdTrain(io: std.Io, out: *std.Io.Writer, in_path: []const u8, out_path: []const u8) !void {
    const wa = std.heap.smp_allocator;
    var loaded = try safetensors.loadFromPath(wa, io, in_path);
    defer loaded.deinitMmap(wa, io);

    var picked: std.ArrayList(types.TensorView) = .empty;
    defer picked.deinit(wa);
    for (loaded.tensors) |t| {
        if (!t.view.dtype.isFloat16Like()) continue;
        try picked.append(wa, t.view);
    }
    try out.print("train: {d} tensors picked, subsampling to 5K elem for B&B\n", .{picked.items.len});
    try out.flush();

    const t_start = std.Io.Timestamp.now(io, .awake);
    var result = try lowlevel_training.trainOnTensors(wa, picked.items, .{
        .max_depth = 4,
        .max_nodes_explored = 30_000,
        .macro_def_cost = 5,
        .min_count_for_promotion = 2,
        .subsample_elements = 5_000,
    });
    defer result.deinit();
    const elapsed = t_start.durationTo(.now(io, .awake)).toMilliseconds();
    try out.print("training done in {d}ms wall\n", .{elapsed});
    try out.print("  n_tensors processed: {d}\n", .{result.counters.n_tensors});
    try out.print("  unique subtrees: {d}\n", .{result.counters.subtree_counts.count()});
    try out.print("  promoted macros: {d}\n", .{result.promoted.len});
    try out.flush();

    const cwd = std.Io.Dir.cwd();
    const f = try cwd.createFile(io, out_path, .{});
    defer f.close(io);
    var wb: [4096]u8 = undefined;
    var wf = f.writer(io, &wb);
    try lowlevel_training.dumpReport(wa, &result, &wf.interface);
    try out.print("wrote report → {s}\n", .{out_path});
    try out.print("\nNext step: python3 tools/emit_macros.py {s} src/discovered_macros.zig && zig build -Doptimize=ReleaseFast\n", .{out_path});
}

// ============== demo (synthetic data) ==============

fn cmdDemo(out: *std.Io.Writer) !void {
    const wa = std.heap.smp_allocator;
    try out.writeAll("=== brevis demo: synthetic Transformer-like tensors ===\n\n");
    try runOne(wa, out, "attention_proj", 512 * 512, 0.02, 0, false);
    try runOne(wa, out, "layernorm_gamma", 4096, 0.01, 1, true);
    try runOne(wa, out, "embedding", 1024 * 512, 0.05, 2, false);
}

fn runOne(wa: std.mem.Allocator, out: *std.Io.Writer, name: []const u8, n: usize, sigma: f32, seed: u64, near_one: bool) !void {
    const buf = try wa.alloc(u8, n * 2);
    defer wa.free(buf);
    var prng: std.Random.DefaultPrng = .init(seed);
    const r = prng.random();
    for (0..n) |i| {
        const v: f16 = if (near_one)
            @floatCast(1.0 + r.floatNorm(f32) * sigma)
        else
            @floatCast(r.floatNorm(f32) * sigma);
        const u: u16 = @bitCast(v);
        std.mem.writeInt(u16, buf[i * 2 ..][0..2], u, .little);
    }
    const stream: types.Stream = .{ .data = buf, .count = n, .bits_per_elem = 16 };
    var best = try astar.synthesize(wa, stream, .{ .max_depth = 1, .max_nodes_explored = 20_000 });
    defer best.deinit(wa);
    if (best.program) |prog| {
        try astar.realize(wa, prog, stream);
        const pb = countPayloadBits(prog);
        const ratio: f64 = @as(f64, @floatFromInt(stream.data.len * 8)) / @as(f64, @floatFromInt(pb));
        try out.print("{s:<24} n={d:<8}  ratio={d:.3}x\n", .{ name, n, ratio });
    }
}

// ============== make-fixture (generate synthetic safetensors) ==============

fn cmdMakeFixture(io: std.Io, out: *std.Io.Writer, path: []const u8) !void {
    const wa = std.heap.smp_allocator;
    var views: std.ArrayList(types.TensorView) = .empty;
    defer {
        for (views.items) |*v| v.deinit(wa);
        views.deinit(wa);
    }
    try views.append(wa, try makeFp16Synthetic(wa, 512, 512, 0.02, 0, false));
    try views.append(wa, try makeFp16Synthetic(wa, 4096, 1, 0.01, 1, true));
    try views.append(wa, try makeFp16Synthetic(wa, 1024, 512, 0.05, 2, false));
    try views.append(wa, try makeFp16Synthetic(wa, 768, 768, 0.02, 3, false));
    const names = [_][]const u8{ "attn_proj.weight", "norm_1.gamma", "embed.weight", "ffn_in.weight" };
    var outs: std.ArrayList(safetensors.TensorOut) = .empty;
    defer outs.deinit(wa);
    for (names, views.items) |n, v| try outs.append(wa, .{ .name = n, .view = v });
    try safetensors.saveToPath(wa, io, path, outs.items);
    try out.print("wrote fixture: {s} ({d} tensors)\n", .{ path, outs.items.len });
}

fn makeFp16Synthetic(wa: std.mem.Allocator, d0: u64, d1: u64, sigma: f32, seed: u64, near_one: bool) !types.TensorView {
    const n: usize = @intCast(d0 * d1);
    const buf = try wa.alloc(u8, n * 2);
    var prng: std.Random.DefaultPrng = .init(seed);
    const r = prng.random();
    for (0..n) |i| {
        const v: f16 = if (near_one)
            @floatCast(1.0 + r.floatNorm(f32) * sigma)
        else
            @floatCast(r.floatNorm(f32) * sigma);
        const u: u16 = @bitCast(v);
        std.mem.writeInt(u16, buf[i * 2 ..][0..2], u, .little);
    }
    const shape = if (d1 == 1) blk: {
        const s = try wa.alloc(u64, 1);
        s[0] = d0;
        break :blk s;
    } else blk: {
        const s = try wa.alloc(u64, 2);
        s[0] = d0;
        s[1] = d1;
        break :blk s;
    };
    return .{ .data = buf, .shape = shape, .dtype = .f16, .owns_data = true, .owns_shape = true };
}
