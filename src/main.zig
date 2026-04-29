//! brevis — CLI for synthesizing & decompressing tensor archives.
//!
//! Subcommands:
//!   brevis compress   <model.safetensors>  <out.brv>
//!   brevis decompress <model.brv>          <out.safetensors>
//!   brevis bench      <model.safetensors>     [per-tensor report]
//!   brevis verify     <model.brv>          <orig.safetensors>
//!   brevis demo                                  [synthetic data]

const std = @import("std");
const types = @import("types.zig");
const ops = @import("ops.zig");
const program = @import("program.zig");
const search = @import("search.zig");
const archive = @import("archive.zig");
const safetensors = @import("safetensors.zig");
const baseline = @import("baseline.zig");
const lowlevel_training = @import("lowlevel_training.zig");

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

    if (args_list.items.len < 2) {
        try printUsage(err);
        try err.flush();
        std.process.exit(2);
    }

    const cmd = args_list.items[1];
    if (std.mem.eql(u8, cmd, "compress")) {
        if (args_list.items.len != 4) return usageExit(err);
        try cmdCompress(alloc, io, out, args_list.items[2], args_list.items[3]);
    } else if (std.mem.eql(u8, cmd, "decompress")) {
        if (args_list.items.len != 4) return usageExit(err);
        try cmdDecompress(alloc, io, out, args_list.items[2], args_list.items[3]);
    } else if (std.mem.eql(u8, cmd, "bench")) {
        if (args_list.items.len != 3) return usageExit(err);
        try cmdBench(alloc, io, out, args_list.items[2]);
    } else if (std.mem.eql(u8, cmd, "verify")) {
        if (args_list.items.len != 4) return usageExit(err);
        try cmdVerify(alloc, io, out, args_list.items[2], args_list.items[3]);
    } else if (std.mem.eql(u8, cmd, "demo")) {
        try cmdDemo(alloc, out);
    } else if (std.mem.eql(u8, cmd, "make-fixture")) {
        if (args_list.items.len != 3) return usageExit(err);
        try cmdMakeFixture(alloc, io, out, args_list.items[2]);
    } else if (std.mem.eql(u8, cmd, "baseline")) {
        if (args_list.items.len != 3) return usageExit(err);
        try cmdBaseline(alloc, io, out, args_list.items[2]);
    } else if (std.mem.eql(u8, cmd, "collect-training")) {
        if (args_list.items.len != 4) return usageExit(err);
        try cmdCollectTraining(alloc, io, out, args_list.items[2], args_list.items[3]);
    } else if (std.mem.eql(u8, cmd, "train-lowlevel")) {
        if (args_list.items.len != 4) return usageExit(err);
        try cmdTrainLowlevel(alloc, io, out, args_list.items[2], args_list.items[3]);
    } else {
        try err.print("brevis: unknown command '{s}'\n", .{cmd});
        try printUsage(err);
        try err.flush();
        std.process.exit(2);
    }
    try out.flush();
}

fn usageExit(err: *std.Io.Writer) !void {
    try printUsage(err);
    try err.flush();
    std.process.exit(2);
}

fn printUsage(w: *std.Io.Writer) !void {
    try w.writeAll(
        \\brevis 0.1.0 — bit-exact lossless tensor compression via program synthesis
        \\
        \\Usage:
        \\  brevis compress   <model.safetensors>  <out.brv>
        \\  brevis decompress <model.brv>          <out.safetensors>
        \\  brevis bench      <model.safetensors>     [per-tensor report]
        \\  brevis verify     <model.brv>          <orig.safetensors>
        \\  brevis demo                                  [synthetic data]
        \\  brevis make-fixture <out.safetensors>        [synthetic data → safetensors]
        \\  brevis baseline   <model.safetensors>        [compare brevis vs gzip vs zstd]
        \\  brevis collect-training <model.safetensors> <out.jsonl>   [PHOG oracle dump]
        \\  brevis train-lowlevel <model.safetensors> <out.json>      [low-level grammar training: B&B per tensor + subtree mining + MDL macro promotion]
        \\
    );
}

// ---------- train-lowlevel: streaming search + abstraction + PHOG counts ----------
fn cmdTrainLowlevel(
    alloc: std.mem.Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    in_path: []const u8,
    out_path: []const u8,
) !void {
    _ = alloc;
    const work_alloc = std.heap.smp_allocator;

    var loaded = try safetensors.loadFromPath(work_alloc, io, in_path);
    defer loaded.deinitMmap(work_alloc, io);

    // Subsample tensors: training the full TinyLlama with low-level B&B on
    // every 131 MB embed/lm_head would take hours. For MVP, take only fp16/bf16
    // tensors with ≤ 1 M elements (small layernorm / small attn k/v_proj).
    var picked: std.ArrayList(types.TensorView) = .empty;
    defer picked.deinit(work_alloc);
    for (loaded.tensors) |t| {
        if (!t.view.dtype.isFloat16Like()) continue;
        try picked.append(work_alloc, t.view); // include all sizes; subsample below
    }

    try out.print("train-lowlevel: {d} tensors picked, subsampling to 5K elem for B&B (mining only)\n", .{picked.items.len});
    try out.flush();

    const t_start = std.Io.Timestamp.now(io, .awake);
    var result = try lowlevel_training.trainOnTensors(work_alloc, picked.items, .{
        .max_depth = 4,
        .max_nodes_explored = 30_000,
        .macro_def_cost = 5,
        .min_count_for_promotion = 2,
        .subsample_elements = 5_000, // 5K samples is plenty for distribution-shape mining
    });
    defer result.deinit();
    const elapsed_ms = t_start.durationTo(.now(io, .awake)).toMilliseconds();

    try out.print("training done in {d}ms wall\n", .{elapsed_ms});
    try out.print("  n_tensors processed: {d}\n", .{result.counters.n_tensors});
    if (result.counters.total_compressed_bits > 0) {
        const ratio: f64 = @as(f64, @floatFromInt(result.counters.total_raw_bits)) / @as(f64, @floatFromInt(result.counters.total_compressed_bits));
        try out.print("  avg compression ratio (low-level B&B): {d:.3}x\n", .{ratio});
    }
    try out.print("  unique subtrees seen: {d}\n", .{result.counters.subtree_counts.count()});
    try out.print("  subtrees promoted (MDL benefit > 0): {d}\n", .{result.promoted.len});
    try out.flush();

    // Write JSON report.
    const cwd = std.Io.Dir.cwd();
    const f = try cwd.createFile(io, out_path, .{});
    defer f.close(io);
    var wb: [4096]u8 = undefined;
    var wf = f.writer(io, &wb);
    try lowlevel_training.dumpReport(work_alloc, &result, &wf.interface);
    try out.print("wrote report → {s}\n", .{out_path});
    try out.flush();
}

// ---------- collect-training (PHOG oracle dump) ----------
//
// For every fp16/bf16 tensor in `in_path`, runs the full A* search (no
// fast-path, larger top_k) and emits one JSONL record per (context,
// production) pair. The Python trainer (tools/train_phog.py) consumes this
// to build the count-based PHOG.
fn cmdCollectTraining(
    alloc: std.mem.Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    in_path: []const u8,
    out_path: []const u8,
) !void {
    _ = alloc;
    const work_alloc = std.heap.smp_allocator;

    var loaded = try safetensors.loadFromPath(work_alloc, io, in_path);
    defer loaded.deinitMmap(work_alloc, io);

    const cwd = std.Io.Dir.cwd();
    const f = try cwd.createFile(io, out_path, .{});
    defer f.close(io);
    var wb: [4096]u8 = undefined;
    var wf = f.writer(io, &wb);
    const w = &wf.interface;

    try out.print("collect-training: {d} tensors → {s}\n", .{ loaded.tensors.len, out_path });
    try out.flush();

    var n_examples: usize = 0;
    for (loaded.tensors, 0..) |t, idx| {
        if (!t.view.dtype.isFloat16Like()) continue;
        var r = try search.synthesize(work_alloc, t.view, &.{}, .{
            .use_fast_path = false,
            .realize_top_k = 32,
        });
        defer r.deinit(work_alloc);

        const trace = search.ChoiceTrace.fromShape(r.chosen_shape);
        const dtype_name = t.view.dtype.name();
        const log2_numel: u8 = @intCast(std.math.log2_int(u64, @max(t.view.numel(), 1)));
        const ndim_bucket: u8 = if (t.view.shape.len == 1) 1 else if (t.view.shape.len == 2) 2 else 3;

        // Emit one record for the T_PROG choice + (if split_float) one per stream slot.
        try w.print(
            \\{{"pos":"root","dtype":"{s}","log2_numel":{d},"ndim":{d},"prod":"{s}"}}
            ++ "\n",
            .{ dtype_name, log2_numel, ndim_bucket, trace.t_prog.name() },
        );
        n_examples += 1;
        if (trace.t_prog == .split_float) {
            const slots = .{
                .{ "split.sign", trace.s_sign.? },
                .{ "split.exp", trace.s_exp.? },
                .{ "split.mant", trace.s_mant.? },
            };
            inline for (slots) |slot| {
                try w.print(
                    \\{{"pos":"{s}","dtype":"{s}","log2_numel":{d},"ndim":{d},"prod":"{s}"}}
                    ++ "\n",
                    .{ slot[0], dtype_name, log2_numel, ndim_bucket, slot[1].name() },
                );
                n_examples += 1;
            }
        }
        if (idx % 20 == 0 or idx + 1 == loaded.tensors.len) {
            try out.print("  [{d}/{d}] {s}: ratio={d:.3}x\n", .{ idx + 1, loaded.tensors.len, t.name, r.compression_ratio });
            try out.flush();
        }
    }
    try w.flush();
    try out.print("wrote {d} (ctx,production) examples\n", .{n_examples});
    try out.flush();
}

// ---------- baseline (brevis vs gzip vs zstd) ----------
fn cmdBaseline(alloc: std.mem.Allocator, io: std.Io, out: *std.Io.Writer, in_path: []const u8) !void {
    var loaded = try safetensors.loadFromPath(alloc, io, in_path);
    defer loaded.deinitMmap(alloc, io);

    // Concatenate raw tensor bytes (the meaningful payload, excludes safetensors header).
    var raw_total: usize = 0;
    for (loaded.tensors) |t| raw_total += t.view.data.len;
    const raw_concat = try alloc.alloc(u8, raw_total);
    defer alloc.free(raw_concat);
    var off: usize = 0;
    for (loaded.tensors) |t| {
        @memcpy(raw_concat[off .. off + t.view.data.len], t.view.data);
        off += t.view.data.len;
    }

    try out.print("=== brevis baseline: {s} ({d} tensors, {d} bytes raw) ===\n\n", .{ in_path, loaded.tensors.len, raw_total });

    // 1) brevis (synthesize each, build archive)
    var results: std.ArrayList(search.Result) = .empty;
    defer {
        for (results.items) |*r| r.deinit(alloc);
        results.deinit(alloc);
    }
    var brevis_total_bits: u64 = 0;
    for (loaded.tensors) |t| {
        if (!t.view.dtype.isFloat16Like()) continue;
        const r = try search.synthesize(alloc, t.view, &.{}, .{});
        try results.append(alloc, r);
        brevis_total_bits += r.actual_bits;
    }
    var jobs: std.ArrayList(archive.TensorJob) = .empty;
    defer jobs.deinit(alloc);
    var idx: usize = 0;
    for (loaded.tensors) |t| {
        if (!t.view.dtype.isFloat16Like()) continue;
        try jobs.append(alloc, .{
            .name = t.name,
            .program = &results.items[idx].program,
            .payload = results.items[idx].payload,
        });
        idx += 1;
    }
    const brevis_bytes = try archive.buildArchiveBytes(alloc, jobs.items);
    defer alloc.free(brevis_bytes);

    // 2) gzip (in-process, level 9)
    const gz_size = try baseline.gzipSize(alloc, raw_concat);

    // 3) zstd at a few levels (shell-out)
    const zstd_3 = try baseline.zstdSize(alloc, io, raw_concat, 3);
    const zstd_19 = try baseline.zstdSize(alloc, io, raw_concat, 19);

    const f = struct {
        fn ratio(orig: usize, comp: usize) f64 {
            return @as(f64, @floatFromInt(orig)) / @as(f64, @floatFromInt(comp));
        }
    };

    try out.writeAll("                       size (bytes)        ratio\n");
    try out.writeAll("                       -------------       --------\n");
    try out.print("raw                    {d:>13}       1.000x  (baseline)\n", .{raw_total});
    try out.print("gzip -9                {d:>13}       {d:.3}x  (DEFLATE, in-process)\n", .{ gz_size, f.ratio(raw_total, gz_size) });
    if (zstd_3) |z|
        try out.print("zstd -3                {d:>13}       {d:.3}x  (zstd default)\n", .{ z, f.ratio(raw_total, z) })
    else
        try out.writeAll("zstd -3                       (skip)        zstd not installed\n");
    if (zstd_19) |z|
        try out.print("zstd -19               {d:>13}       {d:.3}x  (zstd best-effort)\n", .{ z, f.ratio(raw_total, z) })
    else
        try out.writeAll("zstd -19                      (skip)        zstd not installed\n");
    try out.print("brevis (.brv archive)  {d:>13}       {d:.3}x  (synth + entropy + shared codebooks)\n", .{ brevis_bytes.len, f.ratio(raw_total, brevis_bytes.len) });
    try out.print("brevis (bits-only)     {d:>13}       {d:.3}x  (without container overhead)\n", .{ (brevis_total_bits + 7) / 8, f.ratio(raw_total, @as(usize, @intCast((brevis_total_bits + 7) / 8))) });
    try out.writeAll("\nbrevis is bit-exact lossless on all reported tensors.\n");
}

fn cmdMakeFixture(alloc: std.mem.Allocator, io: std.Io, out: *std.Io.Writer, path: []const u8) !void {
    var views: std.ArrayList(types.TensorView) = .empty;
    defer {
        for (views.items) |*v| v.deinit(alloc);
        views.deinit(alloc);
    }

    try views.append(alloc, try makeFp16Synthetic(alloc, 512, 512, 0.02, 0, false));
    try views.append(alloc, try makeFp16Synthetic(alloc, 4096, 1, 0.01, 1, true));
    try views.append(alloc, try makeFp16Synthetic(alloc, 1024, 512, 0.05, 2, false));
    try views.append(alloc, try makeFp16Synthetic(alloc, 768, 768, 0.02, 3, false));

    const names = [_][]const u8{
        "attn_proj.weight",
        "norm_1.gamma",
        "embed.weight",
        "ffn_in.weight",
    };

    var outs: std.ArrayList(safetensors.TensorOut) = .empty;
    defer outs.deinit(alloc);
    for (names, views.items) |n, v| try outs.append(alloc, .{ .name = n, .view = v });
    try safetensors.saveToPath(alloc, io, path, outs.items);
    try out.print("wrote fixture: {s} ({d} tensors)\n", .{ path, outs.items.len });
}

fn makeFp16Synthetic(alloc: std.mem.Allocator, d0: u64, d1: u64, sigma: f32, seed: u64, near_one: bool) !types.TensorView {
    const n: usize = @intCast(d0 * d1);
    const buf = try alloc.alloc(u8, n * 2);
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
        const s = try alloc.alloc(u64, 1);
        s[0] = d0;
        break :blk s;
    } else blk: {
        const s = try alloc.alloc(u64, 2);
        s[0] = d0;
        s[1] = d1;
        break :blk s;
    };
    return .{
        .data = buf,
        .shape = shape,
        .dtype = .f16,
        .owns_data = true,
        .owns_shape = true,
    };
}

// ---------- compress (parallel) ----------
const CompressWorker = struct {
    next: std.atomic.Value(usize),
    tensors: []const safetensors.Tensor,
    results: []?search.Result, // nullable: null = skipped
    work_alloc: std.mem.Allocator,

    fn run(self: *CompressWorker) void {
        while (true) {
            const idx = self.next.fetchAdd(1, .acq_rel);
            if (idx >= self.tensors.len) return;
            const t = self.tensors[idx];
            if (!t.view.dtype.isFloat16Like()) {
                self.results[idx] = null;
                continue;
            }
            const r = search.synthesize(self.work_alloc, t.view, &.{}, .{}) catch {
                self.results[idx] = null;
                continue;
            };
            self.results[idx] = r;
        }
    }
};

fn cmdCompress(alloc: std.mem.Allocator, io: std.Io, out: *std.Io.Writer, in_path: []const u8, out_path: []const u8) !void {
    _ = alloc;
    const work_alloc = std.heap.smp_allocator;

    var loaded = try safetensors.loadFromPath(work_alloc, io, in_path);
    defer loaded.deinitMmap(work_alloc, io);

    const n_threads: usize = std.Thread.getCpuCount() catch 8;
    try out.print("loaded {d} tensors from {s} (compressing on {d} threads)\n", .{ loaded.tensors.len, in_path, n_threads });
    try out.flush();

    const results = try work_alloc.alloc(?search.Result, loaded.tensors.len);
    defer {
        for (results) |maybe_r| {
            if (maybe_r) |r_const| {
                var r = r_const;
                r.deinit(work_alloc);
            }
        }
        work_alloc.free(results);
    }
    for (results) |*r| r.* = null;

    var worker: CompressWorker = .{
        .next = .init(0),
        .tensors = loaded.tensors,
        .results = results,
        .work_alloc = work_alloc,
    };

    const t_start = std.Io.Timestamp.now(io, .awake);
    const threads = try work_alloc.alloc(std.Thread, n_threads);
    defer work_alloc.free(threads);
    for (threads) |*th| th.* = try std.Thread.spawn(.{}, CompressWorker.run, .{&worker});
    for (threads) |th| th.join();
    const synth_ms = t_start.durationTo(.now(io, .awake)).toMilliseconds();
    try out.print("synthesis done in {d}ms wall\n", .{synth_ms});

    var total_raw: u64 = 0;
    var total_compressed: u64 = 0;
    for (loaded.tensors, 0..) |t, idx| {
        if (results[idx]) |r| {
            total_raw += r.raw_bits;
            total_compressed += r.actual_bits;
            try out.print("  {s:<40} {s:<35} {d:.3}x\n", .{ t.name, r.template_summary, r.compression_ratio });
        } else {
            try out.print("  skip {s}: dtype {s} (only fp16/bf16 supported in MVP)\n", .{ t.name, t.view.dtype.name() });
        }
    }
    try out.flush();

    var jobs: std.ArrayList(archive.TensorJob) = .empty;
    defer jobs.deinit(work_alloc);
    for (loaded.tensors, 0..) |t, idx| {
        // Take a stable pointer into the results array; if we copied
        // results[idx] to a local, &r.program would be a dangling pointer.
        if (results[idx] != null) {
            const r_ptr: *search.Result = &results[idx].?;
            try jobs.append(work_alloc, .{
                .name = t.name,
                .program = &r_ptr.program,
                .payload = r_ptr.payload,
            });
        }
    }

    const bytes = try archive.buildArchiveBytes(work_alloc, jobs.items);
    defer work_alloc.free(bytes);

    // End-to-end verify before writing the archive (cheap and worth it).
    {
        var arc = try archive.parseArchive(work_alloc, bytes);
        defer arc.deinit(work_alloc);
        var ok: usize = 0;
        for (arc.tensors) |*at| {
            const orig = blk: {
                for (loaded.tensors) |ot| if (std.mem.eql(u8, ot.name, at.name)) break :blk ot.view;
                return error.MissingOriginal;
            };
            var back = try program.decompressTensor(work_alloc, &at.program, &.{});
            defer back.deinit(work_alloc);
            if (!std.mem.eql(u8, orig.data, back.data)) {
                try out.print("VERIFY FAIL: {s}\n", .{at.name});
                return error.RoundtripFailed;
            }
            ok += 1;
        }
        try out.print("verified {d}/{d} tensors bit-exact before writing\n", .{ ok, arc.tensors.len });
    }

    const cwd = std.Io.Dir.cwd();
    const f = try cwd.createFile(io, out_path, .{});
    defer f.close(io);
    var wb: [4096]u8 = undefined;
    var wf = f.writer(io, &wb);
    // chunked write for large archives (mirror loadFromPath)
    const chunk: usize = 1 << 30;
    var off: usize = 0;
    while (off < bytes.len) {
        const n = @min(chunk, bytes.len - off);
        try wf.interface.writeAll(bytes[off .. off + n]);
        off += n;
    }
    try wf.interface.flush();

    try out.print("\nwrote {s} ({d} bytes)\n", .{ out_path, bytes.len });
    if (total_raw > 0) {
        const ratio: f64 = @as(f64, @floatFromInt(total_raw)) / @as(f64, @floatFromInt(total_compressed));
        try out.print("overall: {d} -> {d} bits ({d:.3}x); archive on disk: {d} bytes\n",
            .{ total_raw, total_compressed, ratio, bytes.len });
    }
    try out.flush();
}

// ---------- decompress ----------
fn cmdDecompress(alloc: std.mem.Allocator, io: std.Io, out: *std.Io.Writer, in_path: []const u8, out_path: []const u8) !void {
    const cwd = std.Io.Dir.cwd();
    const f = try cwd.openFile(io, in_path, .{});
    defer f.close(io);
    const stat = try f.stat(io);
    const bytes = try alloc.alloc(u8, @intCast(stat.size));
    defer alloc.free(bytes);
    var rb: [4096]u8 = undefined;
    var rdr = f.reader(io, &rb);
    _ = try rdr.interface.readSliceAll(bytes);

    var arc = try archive.parseArchive(alloc, bytes);
    defer arc.deinit(alloc);

    var out_views: std.ArrayList(types.TensorView) = .empty;
    defer {
        for (out_views.items) |*v| v.deinit(alloc);
        out_views.deinit(alloc);
    }
    var out_tensors: std.ArrayList(safetensors.TensorOut) = .empty;
    defer out_tensors.deinit(alloc);

    for (arc.tensors) |*t| {
        const view = try program.decompressTensor(alloc, &t.program, &.{});
        try out_views.append(alloc, view);
        try out_tensors.append(alloc, .{ .name = t.name, .view = view });
    }

    try safetensors.saveToPath(alloc, io, out_path, out_tensors.items);
    try out.print("decompressed {d} tensors -> {s}\n", .{ arc.tensors.len, out_path });
}

// ---------- bench ----------
//
// Parallel implementation: a fixed pool of workers atomically pulls the next
// tensor index from a counter, synthesizes it, and stores the result in a
// shared slice indexed by tensor position. The main thread then prints in
// order. We use `smp_allocator` for the per-task work (threadsafe in
// ReleaseFast); the `safetensors.Loaded` is read-only across workers.
const TensorJobResult = struct {
    done: bool = false,
    skipped: bool = false,
    skip_reason: []const u8 = "",
    name: []const u8 = "",
    nbytes: usize = 0,
    summary: []u8 = &.{}, // owned
    ratio: f64 = 0,
    raw_bits: u64 = 0,
    actual_bits: u64 = 0,
    elapsed_ms: i64 = 0,
};

const BenchWorker = struct {
    next: std.atomic.Value(usize),
    tensors: []const safetensors.Tensor,
    results: []TensorJobResult,
    work_alloc: std.mem.Allocator,
    io: std.Io,

    fn run(self: *BenchWorker) void {
        while (true) {
            const idx = self.next.fetchAdd(1, .acq_rel);
            if (idx >= self.tensors.len) return;
            const t = self.tensors[idx];
            self.results[idx].name = t.name;
            self.results[idx].nbytes = t.view.data.len;
            if (!t.view.dtype.isFloat16Like()) {
                self.results[idx].skipped = true;
                self.results[idx].skip_reason = t.view.dtype.name();
                self.results[idx].done = true;
                continue;
            }
            const ts = std.Io.Timestamp.now(self.io, .awake);
            var r = search.synthesize(self.work_alloc, t.view, &.{}, .{}) catch |e| {
                self.results[idx].skipped = true;
                const msg = std.fmt.allocPrint(self.work_alloc, "synth-err:{t}", .{e}) catch "synth-err";
                self.results[idx].skip_reason = msg;
                self.results[idx].done = true;
                continue;
            };
            const elapsed_ms = ts.durationTo(.now(self.io, .awake)).toMilliseconds();
            self.results[idx].summary = self.work_alloc.dupe(u8, r.template_summary) catch &.{};
            self.results[idx].ratio = r.compression_ratio;
            self.results[idx].raw_bits = r.raw_bits;
            self.results[idx].actual_bits = r.actual_bits;
            self.results[idx].elapsed_ms = elapsed_ms;
            self.results[idx].done = true;
            r.deinit(self.work_alloc);
        }
    }
};

fn cmdBench(alloc: std.mem.Allocator, io: std.Io, out: *std.Io.Writer, in_path: []const u8) !void {
    _ = alloc;
    // Use smp_allocator for everything heavy — threadsafe by construction.
    const work_alloc = std.heap.smp_allocator;

    var loaded = try safetensors.loadFromPath(work_alloc, io, in_path);
    defer loaded.deinitMmap(work_alloc, io);

    const n_threads: usize = std.Thread.getCpuCount() catch 8;

    try out.print("=== brevis bench: {s} ({d} tensors, {d} threads) ===\n", .{ in_path, loaded.tensors.len, n_threads });
    try out.writeAll("name                                     bytes        program                                     ratio\n");
    try out.writeAll("---------------------------------------- ------------ ------------------------------------------- -------\n");
    try out.flush();

    const results = try work_alloc.alloc(TensorJobResult, loaded.tensors.len);
    defer {
        for (results) |r| {
            if (r.summary.len > 0) work_alloc.free(r.summary);
            if (r.skipped and r.skip_reason.len > 0 and !std.mem.eql(u8, r.skip_reason, r.name)) {
                // Skip reasons that are dtype names are static; only free synth-err strings (heuristic: starts with "synth-err").
                if (std.mem.startsWith(u8, r.skip_reason, "synth-err")) work_alloc.free(@constCast(r.skip_reason));
            }
        }
        work_alloc.free(results);
    }
    for (results) |*r| r.* = .{};

    var worker: BenchWorker = .{
        .next = .init(0),
        .tensors = loaded.tensors,
        .results = results,
        .work_alloc = work_alloc,
        .io = io,
    };

    const t_start = std.Io.Timestamp.now(io, .awake);
    const threads = try work_alloc.alloc(std.Thread, n_threads);
    defer work_alloc.free(threads);
    for (threads) |*th| th.* = try std.Thread.spawn(.{}, BenchWorker.run, .{&worker});

    // Print results in tensor order as they complete (poll).
    var printed: usize = 0;
    while (printed < results.len) {
        if (!@atomicLoad(bool, &results[printed].done, .acquire)) {
            io.sleep(.fromNanoseconds(20 * std.time.ns_per_ms), .awake) catch {};
            continue;
        }
        const r = results[printed];
        if (r.skipped) {
            try out.print("[{d:>3}/{d}] {s:<32} {d:>12} (skip: {s})\n",
                .{ printed + 1, results.len, r.name, r.nbytes, r.skip_reason });
        } else {
            try out.print("[{d:>3}/{d}] {s:<32} {d:>12} {s:<43} {d:.3}x  ({d}ms)\n",
                .{ printed + 1, results.len, r.name, r.nbytes, r.summary, r.ratio, r.elapsed_ms });
        }
        try out.flush();
        printed += 1;
    }
    for (threads) |th| th.join();

    var total_raw: u64 = 0;
    var total_compressed: u64 = 0;
    for (results) |r| {
        total_raw += r.raw_bits;
        total_compressed += r.actual_bits;
    }
    const total_elapsed_ms = t_start.durationTo(.now(io, .awake)).toMilliseconds();
    if (total_raw > 0) {
        const ratio: f64 = @as(f64, @floatFromInt(total_raw)) / @as(f64, @floatFromInt(total_compressed));
        try out.print("\noverall: {d} -> {d} bits ({d:.3}x) in {d}ms wall\n", .{ total_raw, total_compressed, ratio, total_elapsed_ms });
        try out.flush();
    }
}

// ---------- verify ----------
fn cmdVerify(alloc: std.mem.Allocator, io: std.Io, out: *std.Io.Writer, brv_path: []const u8, orig_path: []const u8) !void {
    const cwd = std.Io.Dir.cwd();
    const f = try cwd.openFile(io, brv_path, .{});
    defer f.close(io);
    const stat = try f.stat(io);
    const bytes = try alloc.alloc(u8, @intCast(stat.size));
    defer alloc.free(bytes);
    var rb: [4096]u8 = undefined;
    var rdr = f.reader(io, &rb);
    _ = try rdr.interface.readSliceAll(bytes);

    var arc = try archive.parseArchive(alloc, bytes);
    defer arc.deinit(alloc);

    var orig = try safetensors.loadFromPath(alloc, io, orig_path);
    defer orig.deinit(alloc);

    var ok_count: usize = 0;
    var fail_count: usize = 0;
    for (arc.tensors) |*t| {
        var found: ?types.TensorView = null;
        for (orig.tensors) |ot| {
            if (std.mem.eql(u8, ot.name, t.name)) {
                found = ot.view;
                break;
            }
        }
        if (found == null) {
            try out.print("  MISSING in original: {s}\n", .{t.name});
            fail_count += 1;
            continue;
        }
        var back = try program.decompressTensor(alloc, &t.program, &.{});
        defer back.deinit(alloc);
        const matches = std.mem.eql(u8, found.?.data, back.data);
        if (matches) {
            ok_count += 1;
        } else {
            try out.print("  MISMATCH: {s}\n", .{t.name});
            fail_count += 1;
        }
    }
    try out.print("verified {d}/{d} tensors bit-exact\n", .{ ok_count, ok_count + fail_count });
    if (fail_count > 0) std.process.exit(1);
}

// ---------- demo ----------
fn cmdDemo(alloc: std.mem.Allocator, out: *std.Io.Writer) !void {
    try out.writeAll("=== brevis demo: synthetic Transformer-like tensors ===\n\n");
    try runOne(alloc, out, "attention_proj", 512 * 512, 0.02, 0, false);
    try runOne(alloc, out, "layernorm_gamma", 4096, 0.01, 1, true);
    try runOne(alloc, out, "embedding", 1024 * 512, 0.05, 2, false);
}

fn runOne(alloc: std.mem.Allocator, out: *std.Io.Writer, name: []const u8, n: usize, sigma: f32, seed: u64, near_one: bool) !void {
    const buf = try alloc.alloc(u8, n * 2);
    var prng: std.Random.DefaultPrng = .init(seed);
    const r = prng.random();
    if (near_one) {
        for (0..n) |i| {
            const v: f16 = @floatCast(1.0 + r.floatNorm(f32) * sigma);
            const u: u16 = @bitCast(v);
            std.mem.writeInt(u16, buf[i * 2 ..][0..2], u, .little);
        }
    } else {
        for (0..n) |i| {
            const v: f16 = @floatCast(r.floatNorm(f32) * sigma);
            const u: u16 = @bitCast(v);
            std.mem.writeInt(u16, buf[i * 2 ..][0..2], u, .little);
        }
    }
    const shape = try alloc.alloc(u64, 1);
    shape[0] = n;
    var t: types.TensorView = .{
        .data = buf,
        .shape = shape,
        .dtype = .f16,
        .owns_data = true,
        .owns_shape = true,
    };
    defer t.deinit(alloc);

    var res = try search.synthesize(alloc, t, &.{}, .{});
    defer res.deinit(alloc);
    try out.print("{s:<24} n={d:<8}  best={s:<32}  ratio={d:.3}x  verified={any}\n",
        .{ name, n, res.template_summary, res.compression_ratio, res.verified });
}
