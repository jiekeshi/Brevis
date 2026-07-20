//! brevis CLI.

const std = @import("std");
const types = @import("types.zig");
const ops = @import("ops.zig");
const program = @import("program.zig");
const prior = @import("prior.zig");
const search = @import("search.zig");
const archive = @import("archive.zig");
const safetensors = @import("safetensors.zig");
const baseline = @import("baseline.zig");

const Allocator = std.mem.Allocator;
const Dtype = types.Dtype;
const Block = types.Block;
const Stream = types.Stream;

const N_DTYPE: usize = 9;
const ROOT_PARENT: u8 = 255;
/// Calibration candidate enumeration is superlinear in block length; a prefix this
/// long already saturates the context statistics.
const CALIB_ELEMS: usize = 16 * 1024;

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &stdout_buf);
    const out = &stdout.interface;
    var stderr_buf: [1024]u8 = undefined;
    var stderr = std.Io.File.stderr().writer(io, &stderr_buf);
    const err = &stderr.interface;

    var argv: std.ArrayList([]u8) = .empty;
    defer {
        for (argv.items) |a| alloc.free(a);
        argv.deinit(alloc);
    }
    var it = init.minimal.args.iterate();
    defer it.deinit();
    while (it.next()) |a| try argv.append(alloc, try alloc.dupe(u8, a));

    if (argv.items.len < 2) try usage(err);
    const cmd = argv.items[1];

    var pos: std.ArrayList([]const u8) = .empty;
    defer pos.deinit(alloc);
    var opt_prior: ?[]const u8 = null;
    var opt_jobs: ?usize = null;
    var opt_blocks: usize = ops.CALIBRATE_BLOCKS;

    const rest = argv.items[2..];
    var i: usize = 0;
    while (i < rest.len) : (i += 1) {
        const a = rest[i];
        if (!std.mem.startsWith(u8, a, "--")) {
            try pos.append(alloc, a);
            continue;
        }
        i += 1;
        if (i >= rest.len) try usage(err);
        const v = rest[i];
        if (std.mem.eql(u8, a, "--prior")) {
            opt_prior = v;
        } else if (std.mem.eql(u8, a, "--jobs")) {
            opt_jobs = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--blocks")) {
            opt_blocks = try std.fmt.parseInt(usize, v, 10);
        } else try usage(err);
    }
    const p = pos.items;

    if (std.mem.eql(u8, cmd, "calibrate")) {
        if (p.len != 2) try usage(err);
        try cmdCalibrate(io, out, p[0], p[1], opt_blocks);
    } else if (std.mem.eql(u8, cmd, "compress")) {
        if (p.len != 2) try usage(err);
        try cmdCompress(io, out, p[0], p[1], opt_prior, opt_jobs);
    } else if (std.mem.eql(u8, cmd, "decompress")) {
        if (p.len != 2) try usage(err);
        try cmdDecompress(io, out, p[0], p[1], opt_jobs);
    } else if (std.mem.eql(u8, cmd, "verify")) {
        if (p.len != 2) try usage(err);
        try cmdVerify(alloc, io, out, p[0], p[1]);
    } else if (std.mem.eql(u8, cmd, "bench")) {
        if (p.len != 1) try usage(err);
        try cmdBench(io, out, p[0], opt_prior, opt_jobs);
    } else if (std.mem.eql(u8, cmd, "baseline")) {
        if (p.len != 1) try usage(err);
        try cmdBaseline(io, out, p[0], opt_jobs);
    } else if (std.mem.eql(u8, cmd, "demo")) {
        try cmdDemo(io, out);
    } else if (std.mem.eql(u8, cmd, "make-fixture")) {
        if (p.len != 1) try usage(err);
        try cmdMakeFixture(alloc, io, out, p[0]);
    } else {
        try err.print("brevis: unknown command '{s}'\n", .{cmd});
        try usage(err);
    }
    try out.flush();
}

fn usage(w: *std.Io.Writer) !noreturn {
    try w.writeAll(
        \\brevis — bit-exact lossless tensor compression via program synthesis
        \\
        \\  brevis calibrate   <model.safetensors> <prior.bin> [--blocks N]
        \\  brevis compress    <model.safetensors> <out.brv> [--prior p.bin] [--jobs N]
        \\  brevis decompress  <in.brv> <out.safetensors> [--jobs N]
        \\  brevis verify      <in.brv> <orig.safetensors>
        \\  brevis bench       <model.safetensors> [--prior p.bin] [--jobs N]
        \\  brevis baseline    <model.safetensors> [--jobs N]
        \\  brevis demo
        \\  brevis make-fixture <out.safetensors>
        \\
    );
    try w.flush();
    std.process.exit(2);
}

// ==================== shared pipeline ====================

fn checkTensors(tensors: []const safetensors.Tensor) !void {
    for (tensors) |t| {
        if (t.view.numel() * t.view.dtype.elemSize() != t.view.data.len) return error.ShapeDataMismatch;
    }
}

fn planAll(alloc: Allocator, tensors: []const safetensors.Tensor) ![]Block {
    var list: std.ArrayList(Block) = .empty;
    errdefer list.deinit(alloc);
    for (tensors, 0..) |t, ti| {
        const numel = t.view.numel();
        if (numel == 0) continue;
        const inner: usize = if (t.view.shape.len == 0) 1 else @intCast(t.view.shape[t.view.shape.len - 1]);
        const bs = try types.planBlocks(alloc, @intCast(ti), t.view.dtype, numel, inner);
        defer alloc.free(bs);
        try list.appendSlice(alloc, bs);
    }
    return list.toOwnedSlice(alloc);
}

fn loadPrior(alloc: Allocator, path: ?[]const u8) !prior.Prior {
    if (path) |pth| return prior.Prior.load(alloc, pth);
    return prior.Prior.initUniform(alloc);
}

const PlanJob = struct {
    next: std.atomic.Value(usize),
    fails: std.atomic.Value(usize),
    tensors: []const safetensors.Tensor,
    tensor_indices: []const usize,
    plans: []?search.Plan,
    pr: *const prior.Prior,
    alloc: Allocator,

    fn run(self: *PlanJob) void {
        while (true) {
            const next = self.next.fetchAdd(1, .acq_rel);
            if (next >= self.tensor_indices.len) return;
            const tensor_idx = self.tensor_indices[next];
            const view = self.tensors[tensor_idx].view;
            const stream: Stream = .{
                .data = view.data,
                .count = view.numel(),
                .bits_per_elem = view.dtype.bitWidth(),
                .owns_data = false,
            };
            self.plans[tensor_idx] = search.synthesizePlan(self.alloc, stream, view.dtype, self.pr, .{}) catch |err| {
                std.debug.print("tensor {d} dtype {s}: {t}\n", .{ tensor_idx, @tagName(view.dtype), err });
                _ = self.fails.fetchAdd(1, .acq_rel);
                continue;
            };
        }
    }
};

const EncodeJob = struct {
    next: std.atomic.Value(usize),
    fails: std.atomic.Value(usize),
    tensors: []const safetensors.Tensor,
    blocks: []const Block,
    results: []?search.Result,
    plans: []const ?search.Plan,
    alloc: Allocator,

    fn run(self: *EncodeJob) void {
        while (true) {
            const i = self.next.fetchAdd(1, .acq_rel);
            if (i >= self.blocks.len) return;
            const b = self.blocks[i];
            const s = b.asStream(self.tensors[b.tensor_idx].view.data);
            const plan = &self.plans[b.tensor_idx].?;
            var result = search.encode(self.alloc, plan, s, b.dtype) catch |err| {
                std.debug.print("block {d} tensor {d} dtype {s}: {t}\n", .{ i, b.tensor_idx, @tagName(b.dtype), err });
                _ = self.fails.fetchAdd(1, .acq_rel);
                continue;
            };
            result.expanded = plan.expanded;
            self.results[i] = result;
        }
    }
};

fn runWorkers(alloc: Allocator, n_threads: usize, job: anytype, comptime run: anytype) !void {
    const threads = try alloc.alloc(std.Thread, n_threads);
    defer alloc.free(threads);
    var spawned: usize = 0;
    errdefer for (threads[0..spawned]) |thread| thread.join();
    for (threads) |*thread| {
        thread.* = try std.Thread.spawn(.{}, run, .{job});
        spawned += 1;
    }
    for (threads) |thread| thread.join();
}

fn BatchPool(comptime Job: type) type {
    return struct {
        const Self = @This();

        alloc: Allocator,
        io: std.Io,
        threads: []std.Thread,
        mutex: std.Io.Mutex = .init,
        ready: std.Io.Condition = .init,
        done: std.Io.Condition = .init,
        job: ?*Job = null,
        epoch: usize = 0,
        finished: usize = 0,
        stopping: bool = false,

        fn init(self: *Self, alloc: Allocator, io: std.Io, n_threads: usize) !void {
            self.* = .{
                .alloc = alloc,
                .io = io,
                .threads = if (n_threads > 1) try alloc.alloc(std.Thread, n_threads) else &.{},
            };
            var spawned: usize = 0;
            errdefer {
                self.stop();
                for (self.threads[0..spawned]) |thread| thread.join();
                if (self.threads.len > 0) alloc.free(self.threads);
            }
            for (self.threads) |*thread| {
                thread.* = try std.Thread.spawn(.{}, worker, .{self});
                spawned += 1;
            }
        }

        fn deinit(self: *Self) void {
            self.stop();
            for (self.threads) |thread| thread.join();
            if (self.threads.len > 0) self.alloc.free(self.threads);
        }

        fn run(self: *Self, job: *Job) void {
            if (self.threads.len == 0) return Job.run(job);
            self.mutex.lockUncancelable(self.io);
            self.job = job;
            self.finished = 0;
            self.epoch += 1;
            self.ready.broadcast(self.io);
            while (self.finished < self.threads.len) self.done.waitUncancelable(self.io, &self.mutex);
            self.mutex.unlock(self.io);
        }

        fn stop(self: *Self) void {
            self.mutex.lockUncancelable(self.io);
            self.stopping = true;
            self.ready.broadcast(self.io);
            self.mutex.unlock(self.io);
        }

        fn worker(self: *Self) void {
            var seen: usize = 0;
            while (true) {
                self.mutex.lockUncancelable(self.io);
                while (!self.stopping and self.epoch == seen)
                    self.ready.waitUncancelable(self.io, &self.mutex);
                if (self.stopping) {
                    self.mutex.unlock(self.io);
                    return;
                }
                seen = self.epoch;
                const job = self.job.?;
                self.mutex.unlock(self.io);

                Job.run(job);

                self.mutex.lockUncancelable(self.io);
                self.finished += 1;
                if (self.finished == self.threads.len) self.done.signal(self.io);
                self.mutex.unlock(self.io);
            }
        }
    };
}

fn synthesizePlans(
    alloc: Allocator,
    tensors: []const safetensors.Tensor,
    pr: *const prior.Prior,
    n_threads: usize,
) ![]?search.Plan {
    const plans = try alloc.alloc(?search.Plan, tensors.len);
    errdefer freePlans(alloc, plans);
    for (plans) |*plan| plan.* = null;

    var tensor_indices: std.ArrayList(usize) = .empty;
    defer tensor_indices.deinit(alloc);
    for (tensors, 0..) |tensor, i| {
        if (tensor.view.numel() > 0) try tensor_indices.append(alloc, i);
    }

    var plan_job: PlanJob = .{
        .next = .init(0),
        .fails = .init(0),
        .tensors = tensors,
        .tensor_indices = tensor_indices.items,
        .plans = plans,
        .pr = pr,
        .alloc = alloc,
    };
    if (tensor_indices.items.len > 0) {
        const n = @max(@as(usize, 1), @min(n_threads, tensor_indices.items.len));
        try runWorkers(alloc, n, &plan_job, PlanJob.run);
    }
    if (plan_job.fails.load(.acquire) > 0) return error.SynthesisFailed;
    return plans;
}

const EncodePool = BatchPool(EncodeJob);

fn encodeBlocks(
    alloc: Allocator,
    tensors: []const safetensors.Tensor,
    blocks: []const Block,
    plans: []const ?search.Plan,
    n_threads: usize,
    pool: ?*EncodePool,
) ![]?search.Result {
    const results = try alloc.alloc(?search.Result, blocks.len);
    errdefer freeResults(alloc, results);
    for (results) |*result| result.* = null;
    var encode_job: EncodeJob = .{
        .next = .init(0),
        .fails = .init(0),
        .tensors = tensors,
        .blocks = blocks,
        .results = results,
        .plans = plans,
        .alloc = alloc,
    };
    if (blocks.len > 0) {
        const n = @max(@as(usize, 1), @min(n_threads, blocks.len));
        if (pool) |p| p.run(&encode_job) else try runWorkers(alloc, n, &encode_job, EncodeJob.run);
    }

    if (encode_job.fails.load(.acquire) > 0) return error.SynthesisFailed;
    return results;
}

fn synthesizeBlocks(
    alloc: Allocator,
    tensors: []const safetensors.Tensor,
    blocks: []const Block,
    pr: *const prior.Prior,
    n_threads: usize,
) ![]?search.Result {
    const plans = try synthesizePlans(alloc, tensors, pr, n_threads);
    defer freePlans(alloc, plans);
    return encodeBlocks(alloc, tensors, blocks, plans, n_threads, null);
}

fn freeResults(alloc: Allocator, results: []?search.Result) void {
    for (results) |*r| if (r.*) |*v| v.deinit(alloc);
    alloc.free(results);
}

fn freePlans(alloc: Allocator, plans: []?search.Plan) void {
    for (plans) |*plan| if (plan.*) |*value| value.deinit(alloc);
    alloc.free(plans);
}

fn tensorMetas(
    alloc: Allocator,
    tensors: []const safetensors.Tensor,
    blocks: []const Block,
) ![]archive.TensorMeta {
    const metas = try alloc.alloc(archive.TensorMeta, tensors.len);
    var bi: usize = 0;
    for (tensors, 0..) |t, ti| {
        var n: u32 = 0;
        while (bi + n < blocks.len and blocks[bi + n].tensor_idx == ti) n += 1;
        metas[ti] = .{
            .name = t.name,
            .dtype = t.view.dtype,
            .shape = t.view.shape,
            .n_blocks = n,
        };
        bi += n;
    }
    std.debug.assert(bi == blocks.len);
    return metas;
}

fn buildArchive(
    alloc: Allocator,
    tensors: []const safetensors.Tensor,
    blocks: []const Block,
    results: []?search.Result,
) ![]u8 {
    const metas = try tensorMetas(alloc, tensors, blocks);
    defer alloc.free(metas);
    var jobs: std.ArrayList(archive.BlockJob) = .empty;
    defer jobs.deinit(alloc);

    for (results, 0..) |_, i| {
        const r = &results[i].?;
        try jobs.append(alloc, .{ .node = &r.node, .payload = r.payload });
    }
    return archive.build(alloc, metas, jobs.items, &.{});
}

fn ratio(orig: u64, comp: u64) f64 {
    if (comp == 0) return 0;
    return @as(f64, @floatFromInt(orig)) / @as(f64, @floatFromInt(comp));
}

fn rawBytes(tensors: []const safetensors.Tensor) u64 {
    var n: u64 = 0;
    for (tensors) |t| n += t.view.data.len;
    return n;
}

fn threadCount(opt: ?usize) usize {
    return opt orelse (std.Thread.getCpuCount() catch 8);
}

// ==================== calibrate ====================

fn stratifiedSample(alloc: Allocator, blocks: []const Block, n: usize, seed: u64) ![]Block {
    if (blocks.len <= n) return alloc.dupe(Block, blocks);

    var strata: std.AutoHashMapUnmanaged(u16, std.ArrayList(u32)) = .empty;
    defer {
        var vit = strata.valueIterator();
        while (vit.next()) |l| l.deinit(alloc);
        strata.deinit(alloc);
    }
    for (blocks, 0..) |b, i| {
        const lg: u16 = @intCast(std.math.log2_int(usize, @max(b.elem_count, 1)));
        const key = (@as(u16, @intFromEnum(b.dtype)) << 8) | lg;
        const gop = try strata.getOrPut(alloc, key);
        if (!gop.found_existing) gop.value_ptr.* = .empty;
        try gop.value_ptr.append(alloc, @intCast(i));
    }

    var keys: std.ArrayList(u16) = .empty;
    defer keys.deinit(alloc);
    var kit = strata.keyIterator();
    while (kit.next()) |k| try keys.append(alloc, k.*);
    std.mem.sort(u16, keys.items, {}, std.sort.asc(u16));

    var prng: std.Random.DefaultPrng = .init(seed);
    const rnd = prng.random();
    for (keys.items) |k| rnd.shuffle(u32, strata.getPtr(k).?.items);

    var out: std.ArrayList(Block) = .empty;
    errdefer out.deinit(alloc);
    var cursor: usize = 0;
    while (out.items.len < n) {
        var progressed = false;
        for (keys.items) |k| {
            const l = strata.getPtr(k).?;
            if (cursor >= l.items.len) continue;
            try out.append(alloc, blocks[l.items[cursor]]);
            progressed = true;
            if (out.items.len == n) break;
        }
        if (!progressed) break;
        cursor += 1;
    }
    return out.toOwnedSlice(alloc);
}

fn accumulate(
    alloc: Allocator,
    counts: *prior.Counts,
    node: program.Node,
    in: Stream,
    dtype: Dtype,
    slot: u8,
    depth: u8,
    parent_op: u8,
    w: f64,
) anyerror!void {
    const ctx = prior.Context.fromStream(in, dtype, slot, depth, parent_op);
    try counts.add(alloc, ctx, node.op, w);
    if (node.op.isTerminal()) return;

    var outs: std.ArrayList(Stream) = .empty;
    defer {
        for (outs.items) |*s| s.deinit(alloc);
        outs.deinit(alloc);
    }
    var side: ops.SideInfo = .none;
    defer side.deinit(alloc);
    try ops.forward(alloc, node.op, node.params, in, &outs, &side);

    for (node.children, outs.items, 0..) |c, s, k|
        try accumulate(alloc, counts, c, s, dtype, @intCast(k), depth + 1, @intFromEnum(node.op), w);
}

const CalibJob = struct {
    next: std.atomic.Value(usize),
    fails: std.atomic.Value(usize),
    tensors: []const safetensors.Tensor,
    picked: []const Block,
    counts: []prior.Counts,
    alloc: Allocator,

    fn run(self: *CalibJob, slot: usize) void {
        while (true) {
            const i = self.next.fetchAdd(1, .acq_rel);
            if (i >= self.picked.len) return;
            self.one(slot, self.picked[i]) catch |e| {
                std.debug.print("calibrate block {d}: {t}\n", .{ i, e });
                _ = self.fails.fetchAdd(1, .acq_rel);
            };
        }
    }

    fn one(self: *CalibJob, slot: usize, b: Block) !void {
        const alloc = self.alloc;
        var s = b.asStream(self.tensors[b.tensor_idx].view.data);
        s.count = @min(s.count, CALIB_ELEMS);
        s.data = s.data[0 .. s.count * s.elemBytes()];

        const cands = try search.synthesizeAll(alloc, s, b.dtype, .{
            .enumerate_all = true,
            .max_nodes = 4,
            .max_depth = 2,
        });
        defer {
            for (cands) |*c| c.deinit(alloc);
            alloc.free(cands);
        }
        if (cands.len == 0) return;

        const best: f64 = @floatFromInt(cands[0].bytes);
        for (cands) |c| {
            const w = @exp(-(@as(f64, @floatFromInt(c.bytes)) - best) / ops.TAU);
            if (w < 0.01) break;
            try accumulate(alloc, &self.counts[slot], c.node, s, b.dtype, 0, 0, ROOT_PARENT, w);
        }
    }
};

fn mergeCounts(alloc: Allocator, dst: *prior.Counts, src: prior.Counts) !void {
    for (0..3) |lv| {
        var it = src.levels[lv].iterator();
        while (it.next()) |e| {
            const gop = try dst.levels[lv].getOrPut(alloc, e.key_ptr.*);
            if (!gop.found_existing) gop.value_ptr.* = @splat(0);
            for (gop.value_ptr, e.value_ptr.*) |*d, v| d.* += v;
        }
    }
}

fn cmdCalibrate(io: std.Io, out: *std.Io.Writer, in_path: []const u8, prior_path: []const u8, n_sample: usize) !void {
    const alloc = std.heap.smp_allocator;

    var loaded = try safetensors.loadFromPath(alloc, io, in_path);
    defer loaded.deinitMmap(alloc, io);
    try checkTensors(loaded.tensors);

    const blocks = try planAll(alloc, loaded.tensors);
    defer alloc.free(blocks);

    const picked = try stratifiedSample(alloc, blocks, n_sample, 0x5EED_B10C);
    defer alloc.free(picked);

    const n_threads = @max(@as(usize, 1), @min(threadCount(null), picked.len));
    try out.print("calibrate: {d} tensors, {d} blocks, sampling {d} on {d} threads\n", .{
        loaded.tensors.len, blocks.len, picked.len, n_threads,
    });
    try out.flush();

    const per_thread = try alloc.alloc(prior.Counts, n_threads);
    defer {
        for (per_thread) |*c| c.deinit(alloc);
        alloc.free(per_thread);
    }
    for (per_thread) |*c| c.* = prior.Counts.init(alloc);

    var job: CalibJob = .{
        .next = .init(0),
        .fails = .init(0),
        .tensors = loaded.tensors,
        .picked = picked,
        .counts = per_thread,
        .alloc = alloc,
    };

    const t0 = std.Io.Timestamp.now(io, .awake);
    const threads = try alloc.alloc(std.Thread, n_threads);
    defer alloc.free(threads);
    for (threads, 0..) |*t, slot| t.* = try std.Thread.spawn(.{}, CalibJob.run, .{ &job, slot });
    for (threads) |t| t.join();
    const ms = t0.durationTo(.now(io, .awake)).toMilliseconds();

    if (job.fails.load(.acquire) > 0) return error.CalibrationFailed;

    var counts = prior.Counts.init(alloc);
    defer counts.deinit(alloc);
    for (per_thread) |c| try mergeCounts(alloc, &counts, c);

    var pr = try counts.toPrior(alloc);
    defer pr.deinit(alloc);
    try pr.save(alloc, prior_path);

    try out.print("calibrated in {d}ms; contexts L0={d} L1={d} L2={d} -> {s}\n", .{
        ms, pr.levels[0].count(), pr.levels[1].count(), pr.levels[2].count(), prior_path,
    });
}

// ==================== compress ====================

fn cmdCompress(
    io: std.Io,
    out: *std.Io.Writer,
    in_path: []const u8,
    out_path: []const u8,
    prior_path: ?[]const u8,
    jobs: ?usize,
) !void {
    const alloc = std.heap.smp_allocator;

    var loaded = try safetensors.loadFromPath(alloc, io, in_path);
    defer loaded.deinitMmap(alloc, io);
    try checkTensors(loaded.tensors);

    var pr = try loadPrior(alloc, prior_path);
    defer pr.deinit(alloc);

    const blocks = try planAll(alloc, loaded.tensors);
    defer alloc.free(blocks);

    const n_threads = threadCount(jobs);
    try out.print("compress: {d} tensors, {d} blocks, {d} threads, prior={s}\n", .{
        loaded.tensors.len, blocks.len, n_threads, prior_path orelse "uniform",
    });
    try out.flush();

    const t0 = std.Io.Timestamp.now(io, .awake);
    const plans = try synthesizePlans(alloc, loaded.tensors, &pr, n_threads);
    defer freePlans(alloc, plans);
    const metas = try tensorMetas(alloc, loaded.tensors, blocks);
    defer alloc.free(metas);

    var atomic = try std.Io.Dir.cwd().createFileAtomic(io, out_path, .{ .replace = true });
    defer atomic.deinit(io);
    var file_buf: [64 * 1024]u8 = undefined;
    var writer = atomic.file.writer(io, &file_buf);
    try writer.interface.writeAll(&archive.HEADER);

    var encode_pool: EncodePool = undefined;
    try encode_pool.init(alloc, io, n_threads);
    defer encode_pool.deinit();

    var file_off: u64 = archive.HEADER.len;
    const batch_size = @max(@as(usize, 1), n_threads) * 16;
    var first: usize = 0;
    while (first < blocks.len) {
        const last = @min(first + batch_size, blocks.len);
        const batch = blocks[first..last];
        const results = try encodeBlocks(alloc, loaded.tensors, batch, plans, n_threads, &encode_pool);
        defer freeResults(alloc, results);
        for (results) |maybe_result| {
            const result = maybe_result.?;
            const frame_header = try archive.frameHeader(alloc, result.node, result.payload.len);
            defer alloc.free(frame_header);
            try writer.interface.writeAll(frame_header);
            try writer.interface.writeAll(result.payload);
            file_off += frame_header.len + result.payload.len;
        }
        first = last;
    }

    const header_len: usize = @intCast(std.mem.readInt(u64, loaded.bytes[0..8], .little));
    const tail = try archive.makeFooter(alloc, metas, file_off, loaded.bytes[0 .. 8 + header_len]);
    defer alloc.free(tail);
    try writer.interface.writeAll(tail);
    try writer.interface.flush();
    const written = file_off + tail.len;
    try atomic.replace(io);
    const ms = t0.durationTo(.now(io, .awake)).toMilliseconds();

    const raw = rawBytes(loaded.tensors);
    try out.print("synthesized and wrote in {d}ms\n", .{ms});
    try out.print("wrote {s}: {d} -> {d} bytes ({d:.3}x)\n", .{ out_path, raw, written, ratio(raw, written) });
}

// ==================== decompress / verify ====================

const DecodeJob = struct {
    next: std.atomic.Value(usize),
    fails: std.atomic.Value(usize),
    blocks: []const archive.ParsedBlock,
    streams: []?Stream,
    alloc: Allocator,

    fn run(self: *DecodeJob) void {
        while (true) {
            const i = self.next.fetchAdd(1, .acq_rel);
            if (i >= self.blocks.len) return;
            self.streams[i] = archive.decodeBlock(self.alloc, self.blocks[i]) catch |err| {
                std.debug.print("decode block {d}: {t}\n", .{ i, err });
                _ = self.fails.fetchAdd(1, .acq_rel);
                continue;
            };
        }
    }
};

const DecodePool = BatchPool(DecodeJob);

fn decodeBlocks(
    alloc: Allocator,
    blocks: []const archive.ParsedBlock,
    n_threads: usize,
    pool: ?*DecodePool,
) ![]?Stream {
    const streams = try alloc.alloc(?Stream, blocks.len);
    errdefer freeStreams(alloc, streams);
    for (streams) |*stream| stream.* = null;

    var job: DecodeJob = .{
        .next = .init(0),
        .fails = .init(0),
        .blocks = blocks,
        .streams = streams,
        .alloc = alloc,
    };
    if (blocks.len > 0) {
        const n = @min(n_threads, blocks.len);
        if (pool) |p| p.run(&job) else if (n == 1) job.run() else try runWorkers(alloc, n, &job, DecodeJob.run);
    }
    if (job.fails.load(.acquire) > 0) return error.DecompressionFailed;
    return streams;
}

fn freeStreams(alloc: Allocator, streams: []?Stream) void {
    for (streams) |*stream| if (stream.*) |*value| value.deinit(alloc);
    alloc.free(streams);
}

fn cmdDecompress(io: std.Io, out: *std.Io.Writer, in_path: []const u8, out_path: []const u8, jobs: ?usize) !void {
    const alloc = std.heap.smp_allocator;
    var loaded = try archive.loadFromPath(alloc, io, in_path);
    defer loaded.deinit(alloc, io);

    const n_threads = @max(@as(usize, 1), threadCount(jobs));
    const t0 = std.Io.Timestamp.now(io, .awake);
    var decode_pool: DecodePool = undefined;
    try decode_pool.init(alloc, io, n_threads);
    defer decode_pool.deinit();

    const metas = try alloc.alloc(safetensors.TensorMeta, loaded.parsed.tensors.len);
    defer alloc.free(metas);
    for (loaded.parsed.tensors, metas) |tensor, *meta| {
        var count: usize = 1;
        for (tensor.shape) |dim| count *= @intCast(dim);
        meta.* = .{
            .name = tensor.name,
            .dtype = tensor.dtype,
            .shape = tensor.shape,
            .byte_len = count * tensor.dtype.elemSize(),
        };
    }
    var atomic = try std.Io.Dir.cwd().createFileAtomic(io, out_path, .{ .replace = true });
    defer atomic.deinit(io);
    var file_buf: [64 * 1024]u8 = undefined;
    var writer = atomic.file.writer(io, &file_buf);
    if (loaded.parsed.safetensors_prefix.len > 0) {
        try writer.interface.writeAll(loaded.parsed.safetensors_prefix);
    } else {
        const header = try safetensors.buildHeader(alloc, metas);
        defer alloc.free(header);
        var len_buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &len_buf, header.len, .little);
        try writer.interface.writeAll(&len_buf);
        try writer.interface.writeAll(header);
    }

    for (loaded.parsed.tensors, metas) |tensor, meta| {
        var written: usize = 0;
        const batch_size = n_threads * 16;
        var frame_pos: usize = 0;
        var remaining: usize = tensor.n_blocks;
        while (remaining > 0) {
            const n = @min(batch_size, remaining);
            const blocks = try alloc.alloc(archive.ParsedBlock, n);
            defer alloc.free(blocks);
            for (blocks) |*block| block.* = try archive.nextBlock(tensor.frames, &frame_pos);
            const streams = try decodeBlocks(alloc, blocks, n_threads, &decode_pool);
            defer freeStreams(alloc, streams);
            for (streams) |maybe_stream| {
                const stream = maybe_stream.?;
                const data = stream.data[0 .. stream.count * stream.elemBytes()];
                try writer.interface.writeAll(data);
                written += data.len;
            }
            remaining -= n;
        }
        std.debug.assert(frame_pos == tensor.frames.len);
        if (written != meta.byte_len) return error.ShapeDataMismatch;
    }
    try writer.interface.flush();
    try atomic.replace(io);
    const ms = t0.durationTo(.now(io, .awake)).toMilliseconds();
    try out.print("decompressed {d} tensors on {d} threads in {d}ms -> {s}\n", .{
        loaded.parsed.tensors.len, n_threads, ms, out_path,
    });
}

fn cmdVerify(alloc: Allocator, io: std.Io, out: *std.Io.Writer, brv_path: []const u8, orig_path: []const u8) !void {
    var loaded = try archive.loadFromPath(alloc, io, brv_path);
    defer loaded.deinit(alloc, io);

    var orig = try safetensors.loadFromPath(alloc, io, orig_path);
    defer orig.deinitMmap(alloc, io);

    const header_len: usize = @intCast(std.mem.readInt(u64, orig.bytes[0..8], .little));
    const header_matches = loaded.parsed.safetensors_prefix.len == 0 or
        std.mem.eql(u8, loaded.parsed.safetensors_prefix, orig.bytes[0 .. 8 + header_len]);
    if (!header_matches) try out.writeAll("  SAFETENSORS HEADER MISMATCH\n");

    var ok: usize = 0;
    var bad: usize = 0;
    if (loaded.parsed.tensors.len != orig.tensors.len) {
        try out.print("  TENSOR COUNT: archive {d}, original {d}\n", .{ loaded.parsed.tensors.len, orig.tensors.len });
        bad += if (loaded.parsed.tensors.len > orig.tensors.len)
            loaded.parsed.tensors.len - orig.tensors.len
        else
            orig.tensors.len - loaded.parsed.tensors.len;
    }
    for (loaded.parsed.tensors) |*t| {
        const found: ?types.TensorView = blk: {
            for (orig.tensors) |ot| {
                if (std.mem.eql(u8, ot.name, t.name)) break :blk ot.view;
            }
            break :blk null;
        };
        if (found == null) {
            try out.print("  MISSING in original: {s}\n", .{t.name});
            bad += 1;
            continue;
        }
        if (t.dtype != found.?.dtype or !std.mem.eql(u64, t.shape, found.?.shape)) {
            try out.print("  METADATA MISMATCH: {s}\n", .{t.name});
            bad += 1;
            continue;
        }
        var off: usize = 0;
        var matches = true;
        var frame_pos: usize = 0;
        for (0..t.n_blocks) |_| {
            const block = try archive.nextBlock(t.frames, &frame_pos);
            var stream = try archive.decodeBlock(alloc, block);
            defer stream.deinit(alloc);
            const data = stream.data[0 .. stream.count * stream.elemBytes()];
            if (data.len > found.?.data.len - off or !std.mem.eql(u8, data, found.?.data[off..][0..data.len])) {
                matches = false;
                break;
            }
            off += data.len;
        }
        std.debug.assert(frame_pos == t.frames.len);
        if (matches and off == found.?.data.len) {
            ok += 1;
        } else {
            try out.print("  MISMATCH: {s}\n", .{t.name});
            bad += 1;
        }
    }
    try out.print("verified {d}/{d} tensors bit-exact\n", .{ ok, ok + bad });
    try out.flush();
    if (bad > 0 or !header_matches) std.process.exit(1);
}

// ==================== bench ====================

fn renderProgram(alloc: Allocator, out: *std.ArrayList(u8), node: program.Node) Allocator.Error!void {
    try out.appendSlice(alloc, @tagName(node.op));
    if (node.children.len == 0) return;
    try out.append(alloc, '(');
    for (node.children, 0..) |c, i| {
        if (i > 0) try out.append(alloc, ',');
        try renderProgram(alloc, out, c);
    }
    try out.append(alloc, ')');
}

const DtypeStat = struct {
    blocks: usize = 0,
    raw: u64 = 0,
    comp: u64 = 0,
    shapes: std.StringHashMapUnmanaged(usize) = .empty,
};

const ShapeCount = struct { name: []const u8, n: usize };

fn moreCount(_: void, a: ShapeCount, b: ShapeCount) bool {
    return a.n > b.n;
}

fn report(alloc: Allocator, out: *std.Io.Writer, blocks: []const Block, results: []const ?search.Result) !void {
    var arena: std.heap.ArenaAllocator = .init(alloc);
    defer arena.deinit();
    const a = arena.allocator();

    var stats: [N_DTYPE]DtypeStat = @splat(.{});
    var buf: std.ArrayList(u8) = .empty;

    for (blocks, results) |b, maybe| {
        const r = maybe orelse continue;
        const s = &stats[@intFromEnum(b.dtype)];
        s.blocks += 1;
        s.raw += b.byteLen();
        s.comp += r.bytes;

        buf.clearRetainingCapacity();
        try renderProgram(a, &buf, r.node);
        const gop = try s.shapes.getOrPut(a, buf.items);
        if (!gop.found_existing) {
            gop.key_ptr.* = try a.dupe(u8, buf.items);
            gop.value_ptr.* = 0;
        }
        gop.value_ptr.* += 1;
    }

    var total_raw: u64 = 0;
    var total_comp: u64 = 0;
    for (0..N_DTYPE) |di| {
        const s = stats[di];
        if (s.blocks == 0) continue;
        total_raw += s.raw;
        total_comp += s.comp;

        const dt: Dtype = @enumFromInt(di);
        try out.print("\n{s}: {d} blocks, {d} -> {d} bytes ({d:.3}x)\n", .{
            dt.name(), s.blocks, s.raw, s.comp, ratio(s.raw, s.comp),
        });

        var top: std.ArrayList(ShapeCount) = .empty;
        defer top.deinit(a);
        var it = s.shapes.iterator();
        while (it.next()) |e| try top.append(a, .{ .name = e.key_ptr.*, .n = e.value_ptr.* });
        std.mem.sort(ShapeCount, top.items, {}, moreCount);

        for (top.items[0..@min(10, top.items.len)]) |sc| {
            try out.print("  {d:>6}  {s}\n", .{ sc.n, sc.name });
        }
        try out.flush();
    }

    try out.print("\noverall: {d} -> {d} bytes ({d:.3}x)\n", .{ total_raw, total_comp, ratio(total_raw, total_comp) });
}

fn cmdBench(io: std.Io, out: *std.Io.Writer, in_path: []const u8, prior_path: ?[]const u8, jobs: ?usize) !void {
    const alloc = std.heap.smp_allocator;

    var loaded = try safetensors.loadFromPath(alloc, io, in_path);
    defer loaded.deinitMmap(alloc, io);
    try checkTensors(loaded.tensors);

    var pr = try loadPrior(alloc, prior_path);
    defer pr.deinit(alloc);

    const blocks = try planAll(alloc, loaded.tensors);
    defer alloc.free(blocks);

    const n_threads = threadCount(jobs);
    try out.print("=== brevis bench: {s} ({d} tensors, {d} blocks, {d} threads) ===\n", .{
        in_path, loaded.tensors.len, blocks.len, n_threads,
    });
    try out.flush();

    const t0 = std.Io.Timestamp.now(io, .awake);
    const results = try synthesizeBlocks(alloc, loaded.tensors, blocks, &pr, n_threads);
    defer freeResults(alloc, results);
    const ms = t0.durationTo(.now(io, .awake)).toMilliseconds();

    try report(alloc, out, blocks, results);
    try out.print("synthesis wall time: {d}ms\n", .{ms});
}

// ==================== baseline ====================

fn cmdBaseline(io: std.Io, out: *std.Io.Writer, in_path: []const u8, jobs: ?usize) !void {
    const alloc = std.heap.smp_allocator;

    var loaded = try safetensors.loadFromPath(alloc, io, in_path);
    defer loaded.deinitMmap(alloc, io);
    try checkTensors(loaded.tensors);

    const raw_total = rawBytes(loaded.tensors);
    const raw_concat = try alloc.alloc(u8, @intCast(raw_total));
    defer alloc.free(raw_concat);
    var off: usize = 0;
    for (loaded.tensors) |t| {
        @memcpy(raw_concat[off..][0..t.view.data.len], t.view.data);
        off += t.view.data.len;
    }

    try out.print("=== brevis baseline: {s} ({d} tensors, {d} bytes raw) ===\n\n", .{
        in_path, loaded.tensors.len, raw_total,
    });
    try out.flush();

    var pr = prior.Prior.initUniform(alloc);
    defer pr.deinit(alloc);
    const blocks = try planAll(alloc, loaded.tensors);
    defer alloc.free(blocks);
    const results = try synthesizeBlocks(alloc, loaded.tensors, blocks, &pr, threadCount(jobs));
    defer freeResults(alloc, results);
    const brv = try buildArchive(alloc, loaded.tensors, blocks, results);
    defer alloc.free(brv);

    const gz = try baseline.gzipSize(alloc, raw_concat);
    const z3 = try baseline.zstdSize(alloc, io, raw_concat, 3);
    const z19 = try baseline.zstdSize(alloc, io, raw_concat, 19);

    try out.writeAll("                       size (bytes)      ratio\n");
    try out.print("raw                    {d:>13}     1.000x\n", .{raw_total});
    try out.print("gzip -9                {d:>13}     {d:.3}x\n", .{ gz, ratio(raw_total, gz) });
    if (z3) |z|
        try out.print("zstd -3                {d:>13}     {d:.3}x\n", .{ z, ratio(raw_total, z) })
    else
        try out.writeAll("zstd -3                      (skip)     zstd not installed\n");
    if (z19) |z|
        try out.print("zstd -19               {d:>13}     {d:.3}x\n", .{ z, ratio(raw_total, z) })
    else
        try out.writeAll("zstd -19                     (skip)     zstd not installed\n");
    try out.print("brevis (.brv)          {d:>13}     {d:.3}x\n", .{ brv.len, ratio(raw_total, brv.len) });
}

// ==================== synthetic data ====================

fn synthTensor(alloc: Allocator, dtype: Dtype, dims: []const u64, seed: u64, near_one: bool) !types.TensorView {
    var n: usize = 1;
    for (dims) |d| n *= @intCast(d);

    const buf = try alloc.alloc(u8, n * dtype.elemSize());
    errdefer alloc.free(buf);
    var prng: std.Random.DefaultPrng = .init(seed);
    const r = prng.random();

    var counter: u32 = 0;
    for (0..n) |i| {
        const f: f32 = if (near_one) 1.0 + r.floatNorm(f32) * 0.02 else r.floatNorm(f32) * 0.02;
        switch (dtype) {
            .f16 => std.mem.writeInt(u16, buf[i * 2 ..][0..2], @bitCast(@as(f16, @floatCast(f))), .little),
            .bf16 => std.mem.writeInt(u16, buf[i * 2 ..][0..2], @truncate(@as(u32, @bitCast(f)) >> 16), .little),
            .f32 => std.mem.writeInt(u32, buf[i * 4 ..][0..4], @bitCast(f), .little),
            .u8 => buf[i] = r.intRangeAtMost(u8, 0, 31),
            .i8 => buf[i] = @bitCast(r.intRangeAtMost(i8, -16, 15)),
            .u16 => std.mem.writeInt(u16, buf[i * 2 ..][0..2], r.intRangeAtMost(u16, 0, 1023), .little),
            .i16 => std.mem.writeInt(u16, buf[i * 2 ..][0..2], @bitCast(r.intRangeAtMost(i16, -512, 511)), .little),
            .u32 => {
                counter +%= r.intRangeAtMost(u32, 0, 7);
                std.mem.writeInt(u32, buf[i * 4 ..][0..4], counter, .little);
            },
            .i32 => std.mem.writeInt(u32, buf[i * 4 ..][0..4], @bitCast(r.intRangeAtMost(i32, -512, 511)), .little),
        }
    }

    const shape = try alloc.dupe(u64, dims);
    return .{ .data = buf, .shape = shape, .dtype = dtype, .owns_data = true, .owns_shape = true };
}

const FixtureSpec = struct { name: []const u8, dtype: Dtype, dims: []const u64, near_one: bool = false };

const fixtures = [_]FixtureSpec{
    .{ .name = "attn.q_proj.weight", .dtype = .f16, .dims = &.{ 512, 512 } },
    .{ .name = "norm.gamma", .dtype = .f16, .dims = &.{4096}, .near_one = true },
    .{ .name = "embed.weight", .dtype = .bf16, .dims = &.{ 1024, 256 } },
    .{ .name = "ffn.up.weight", .dtype = .f32, .dims = &.{ 256, 256 } },
    .{ .name = "quant.scales", .dtype = .i8, .dims = &.{8192} },
    .{ .name = "router.ids", .dtype = .u32, .dims = &.{4096} },
};

fn makeFixtureTensors(alloc: Allocator, views: *std.ArrayList(types.TensorView), tensors: *std.ArrayList(safetensors.Tensor)) !void {
    for (fixtures, 0..) |spec, i| {
        try views.append(alloc, try synthTensor(alloc, spec.dtype, spec.dims, i, spec.near_one));
    }
    for (fixtures, views.items) |spec, v| {
        try tensors.append(alloc, .{ .name = spec.name, .view = v });
    }
}

fn cmdMakeFixture(alloc: Allocator, io: std.Io, out: *std.Io.Writer, path: []const u8) !void {
    var views: std.ArrayList(types.TensorView) = .empty;
    defer {
        for (views.items) |*v| v.deinit(alloc);
        views.deinit(alloc);
    }
    var tensors: std.ArrayList(safetensors.Tensor) = .empty;
    defer tensors.deinit(alloc);
    try makeFixtureTensors(alloc, &views, &tensors);

    var outs: std.ArrayList(safetensors.TensorOut) = .empty;
    defer outs.deinit(alloc);
    for (tensors.items) |t| try outs.append(alloc, .{ .name = t.name, .view = t.view });

    try safetensors.saveToPath(alloc, io, path, outs.items);
    try out.print("wrote fixture {s} ({d} tensors)\n", .{ path, outs.items.len });
}

fn cmdDemo(io: std.Io, out: *std.Io.Writer) !void {
    const alloc = std.heap.smp_allocator;

    var views: std.ArrayList(types.TensorView) = .empty;
    defer {
        for (views.items) |*v| v.deinit(alloc);
        views.deinit(alloc);
    }
    var tensors: std.ArrayList(safetensors.Tensor) = .empty;
    defer tensors.deinit(alloc);
    try makeFixtureTensors(alloc, &views, &tensors);

    var pr = prior.Prior.initUniform(alloc);
    defer pr.deinit(alloc);

    const blocks = try planAll(alloc, tensors.items);
    defer alloc.free(blocks);

    try out.print("=== brevis demo: {d} synthetic tensors, {d} blocks ===\n", .{ tensors.items.len, blocks.len });
    try out.flush();

    const t0 = std.Io.Timestamp.now(io, .awake);
    const results = try synthesizeBlocks(alloc, tensors.items, blocks, &pr, threadCount(null));
    defer freeResults(alloc, results);
    const ms = t0.durationTo(.now(io, .awake)).toMilliseconds();

    try report(alloc, out, blocks, results);
    try out.print("synthesis wall time: {d}ms\n", .{ms});
}
