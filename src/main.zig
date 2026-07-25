//! brevis CLI.

const std = @import("std");
const types = @import("types.zig");
const ops = @import("ops.zig");
const program = @import("program.zig");
const prior = @import("prior.zig");
const calibrate = @import("calibrate.zig");
const search = @import("search.zig");
const archive = @import("archive.zig");
const safetensors = @import("safetensors.zig");
const pool_mod = @import("pool.zig");
const report_mod = @import("report.zig");
const macro = @import("macro.zig");

const Allocator = std.mem.Allocator;
const Dtype = types.Dtype;
const Block = types.Block;
const Stream = types.Stream;

const PlanMode = enum { search, fixed };

/// Evaluation names the three comparable configurations this way; the
/// reporting module receives the resolved name rather than this enum.
fn modeName(mode: PlanMode, learned_prior: bool) []const u8 {
    return if (mode == .fixed) "fixed" else if (learned_prior) "phog" else "uniform";
}
const ReportFormat = enum { text, json };

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
    var opt_macros: ?[]const u8 = null;
    var opt_jobs: ?usize = null;
    var opt_tensors: usize = calibrate.DEFAULT_TENSORS;
    var opt_plan: PlanMode = .search;
    var opt_format: ReportFormat = .text;
    var opt_search: search.Options = .{};
    var saw_prior = false;
    var saw_jobs = false;
    var saw_tensors = false;
    var saw_plan = false;
    var saw_format = false;
    var saw_search_option = false;

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
            saw_prior = true;
            opt_prior = v;
        } else if (std.mem.eql(u8, a, "--jobs")) {
            saw_jobs = true;
            opt_jobs = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--tensors")) {
            saw_tensors = true;
            opt_tensors = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--plan")) {
            saw_plan = true;
            if (std.mem.eql(u8, v, "search")) opt_plan = .search else if (std.mem.eql(u8, v, "fixed")) opt_plan = .fixed else try usage(err);
        } else if (std.mem.eql(u8, a, "--format")) {
            saw_format = true;
            if (std.mem.eql(u8, v, "text")) opt_format = .text else if (std.mem.eql(u8, v, "json")) opt_format = .json else try usage(err);
        } else if (std.mem.eql(u8, a, "--max-expansions")) {
            saw_search_option = true;
            opt_search.max_expansions = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--max-nodes")) {
            saw_search_option = true;
            opt_search.max_nodes = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--max-depth")) {
            saw_search_option = true;
            opt_search.max_depth = try std.fmt.parseInt(u8, v, 10);
        } else if (std.mem.eql(u8, a, "--sample-elems")) {
            saw_search_option = true;
            opt_search.sample_elems = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--rerank-candidates")) {
            saw_search_option = true;
            opt_search.rerank_candidates = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--rerank-blocks")) {
            saw_search_option = true;
            opt_search.rerank_blocks = try std.fmt.parseInt(usize, v, 10);
        } else if (std.mem.eql(u8, a, "--macros")) {
            saw_search_option = true;
            opt_macros = v;
        } else if (std.mem.eql(u8, a, "--disable-op")) {
            saw_search_option = true;
            const op = std.meta.stringToEnum(ops.OpKind, v) orelse try usage(err);
            if (op == .raw) try usage(err);
            opt_search.enabled_ops &= ~ops.opMask(op);
        } else try usage(err);
    }
    var library: ?macro.Library = null;
    defer if (library) |*loaded| loaded.deinit();
    if (opt_macros) |path| {
        library = macro.load(alloc, io, path) catch |e| {
            try err.print("brevis: cannot load macro library '{s}': {t}\n", .{ path, e });
            try err.flush();
            std.process.exit(2);
        };
        opt_search.macros = library.?.macros;
    }

    if (opt_search.max_expansions == 0 or opt_search.max_nodes == 0) try usage(err);
    if (opt_jobs) |jobs| if (jobs == 0) try usage(err);
    if (saw_tensors and opt_tensors == 0) try usage(err);
    const is_calibrate = std.mem.eql(u8, cmd, "calibrate");
    const is_compress = std.mem.eql(u8, cmd, "compress");
    const is_decompress = std.mem.eql(u8, cmd, "decompress");
    const is_bench = std.mem.eql(u8, cmd, "bench");
    const is_config = std.mem.eql(u8, cmd, "config");
    if (saw_prior and !(is_compress or is_bench)) try usage(err);
    if (saw_jobs and !(is_calibrate or is_compress or is_decompress or is_bench)) try usage(err);
    if (saw_tensors and !is_calibrate) try usage(err);
    if (saw_plan and !(is_compress or is_bench)) try usage(err);
    if (saw_format and !(is_calibrate or is_bench)) try usage(err);
    const accepts_search_options = std.mem.eql(u8, cmd, "calibrate") or
        is_compress or is_bench or is_config;
    if (saw_search_option and !accepts_search_options) try usage(err);
    const p = pos.items;

    if (std.mem.eql(u8, cmd, "calibrate")) {
        if (p.len != 2) try usage(err);
        try cmdCalibrate(io, out, p[0], p[1], opt_tensors, opt_jobs, opt_search, opt_format);
    } else if (std.mem.eql(u8, cmd, "compress")) {
        if (p.len != 2) try usage(err);
        try cmdCompress(io, out, p[0], p[1], opt_prior, opt_jobs, opt_plan, opt_search);
    } else if (std.mem.eql(u8, cmd, "decompress")) {
        if (p.len != 2) try usage(err);
        try cmdDecompress(io, out, p[0], p[1], opt_jobs);
    } else if (std.mem.eql(u8, cmd, "verify")) {
        if (p.len != 2) try usage(err);
        try cmdVerify(alloc, io, out, p[0], p[1]);
    } else if (std.mem.eql(u8, cmd, "bench")) {
        if (p.len != 1) try usage(err);
        try cmdBench(io, out, p[0], opt_prior, opt_jobs, opt_plan, opt_format, opt_search);
    } else if (std.mem.eql(u8, cmd, "config")) {
        if (p.len != 0) try usage(err);
        try cmdConfig(out, opt_search);
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
        \\  brevis calibrate   <model.safetensors> <prior.bin> [--tensors N] [--jobs N] [--format text|json]
        \\  brevis compress    <model.safetensors> <out.brv> [--plan search|fixed] [--prior p.bin] [--jobs N]
        \\  brevis decompress  <in.brv> <out.safetensors> [--jobs N]
        \\  brevis verify      <in.brv> <orig.safetensors>
        \\  brevis bench       <model.safetensors> [--plan search|fixed] [--prior p.bin] [--jobs N] [--format text|json]
        \\  brevis config
        \\  brevis demo
        \\  brevis make-fixture <out.safetensors>
        \\
        \\Search options (calibrate, compress, bench, and config):
        \\  --max-expansions N --max-nodes N --max-depth N --sample-elems N
        \\  --rerank-candidates N --rerank-blocks N --disable-op NAME (repeatable)
        \\  --macros <library.json>   learned subtrees offered as single productions
        \\
    );
    try w.flush();
    std.process.exit(2);
}


fn cmdConfig(out: *std.Io.Writer, options: search.Options) !void {
    var json: std.json.Stringify = .{
        .writer = out,
        .options = .{ .whitespace = .indent_2 },
    };
    try json.beginObject();
    try report_mod.writeSearchConfigFields(&json, options);
    try report_mod.writeOperatorTable(&json);
    try json.endObject();
    try out.writeByte('\n');
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
    if (path) |prior_path| return prior.Prior.load(alloc, prior_path);
    return .empty;
}

fn sha256File(io: std.Io, path: []const u8) ![64]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    var reader_buffer: [64 * 1024]u8 = undefined;
    var chunk: [64 * 1024]u8 = undefined;
    var reader = file.reader(io, &reader_buffer);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    while (true) {
        const n = try reader.interface.readSliceShort(&chunk);
        hash.update(chunk[0..n]);
        if (n < chunk.len) break;
    }
    var digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    hash.final(&digest);
    return std.fmt.bytesToHex(digest, .lower);
}

const PlanJob = struct {
    next: std.atomic.Value(usize),
    fails: std.atomic.Value(usize),
    tensors: []const safetensors.Tensor,
    tensor_indices: []const usize,
    plans: []?search.Plan,
    pr: *const prior.Prior,
    mode: PlanMode,
    search_options: search.Options,
    alloc: Allocator,

    pub fn run(self: *PlanJob) void {
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
            const inner: usize = if (view.shape.len == 0) 1 else @intCast(view.shape[view.shape.len - 1]);
            self.plans[tensor_idx] = switch (self.mode) {
                .search => search.synthesizeTensorPlan(
                    self.alloc,
                    stream,
                    view.dtype,
                    inner,
                    self.pr,
                    self.search_options,
                ),
                .fixed => search.fixedPlan(self.alloc, view.dtype),
            } catch |err| {
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

    pub fn run(self: *EncodeJob) void {
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


fn synthesizePlans(
    alloc: Allocator,
    tensors: []const safetensors.Tensor,
    pr: *const prior.Prior,
    n_threads: usize,
    mode: PlanMode,
    search_options: search.Options,
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
        .mode = mode,
        .search_options = search_options,
        .alloc = alloc,
    };
    if (tensor_indices.items.len > 0) {
        const n = @max(@as(usize, 1), @min(n_threads, tensor_indices.items.len));
        try pool_mod.runWorkers(alloc, n, &plan_job, PlanJob.run);
    }
    if (plan_job.fails.load(.acquire) > 0) return error.SynthesisFailed;
    return plans;
}

const EncodePool = pool_mod.BatchPool(EncodeJob);

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
        if (pool) |p| p.run(&encode_job) else try pool_mod.runWorkers(alloc, n, &encode_job, EncodeJob.run);
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
    mode: PlanMode,
    search_options: search.Options,
) ![]?search.Result {
    const plans = try synthesizePlans(alloc, tensors, pr, n_threads, mode, search_options);
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


fn rawBytes(tensors: []const safetensors.Tensor) u64 {
    var n: u64 = 0;
    for (tensors) |t| n += t.view.data.len;
    return n;
}

fn threadCount(opt: ?usize) usize {
    return @max(@as(usize, 1), opt orelse (std.Thread.getCpuCount() catch 8));
}

fn cmdCalibrate(
    io: std.Io,
    out: *std.Io.Writer,
    in_path: []const u8,
    prior_path: []const u8,
    n_sample: usize,
    jobs: ?usize,
    search_options: search.Options,
    format: ReportFormat,
) !void {
    const alloc = std.heap.smp_allocator;

    var loaded = try safetensors.loadFromPath(alloc, io, in_path);
    defer loaded.deinitMmap(alloc, io);
    try checkTensors(loaded.tensors);

    const n_threads = threadCount(jobs);
    if (format == .text) {
        try out.print("calibrate: {d} tensors, sampling up to {d} on at most {d} threads\n", .{
            loaded.tensors.len, n_sample, n_threads,
        });
        try out.flush();
    }

    const t0 = std.Io.Timestamp.now(io, .awake);
    var trained = try calibrate.train(alloc, loaded.tensors, .{
        .max_tensors = n_sample,
        .threads = n_threads,
        .search_options = search_options,
    });
    defer trained.prior.deinit(alloc);
    const ms = t0.durationTo(.now(io, .awake)).toMilliseconds();
    try trained.prior.save(alloc, prior_path);
    const context_counts: [3]usize = .{
        trained.prior.levels[0].count(),
        trained.prior.levels[1].count(),
        trained.prior.levels[2].count(),
    };
    switch (format) {
        .text => try out.print("calibrated {d} tensors on {d} threads in {d}ms; contexts L0={d} L1={d} L2={d} -> {s}\n", .{
            trained.sampled,   trained.threads_used, ms,
            context_counts[0], context_counts[1],    context_counts[2],
            prior_path,
        }),
        .json => {
            const input_digest = try sha256File(io, in_path);
            const prior_digest = try sha256File(io, prior_path);
            var json: std.json.Stringify = .{
                .writer = out,
                .options = .{ .whitespace = .indent_2 },
            };
            try json.beginObject();
            try json.objectField("schema");
            try json.write(1);
            try json.objectField("kind");
            try json.write("brevis.calibration-report");
            try json.objectField("input");
            try json.beginObject();
            try json.objectField("path");
            try json.write(in_path);
            try json.objectField("size_bytes");
            try json.write(loaded.bytes.len);
            try json.objectField("sha256");
            try json.write(input_digest[0..]);
            try json.endObject();
            try json.objectField("output_prior");
            try json.beginObject();
            try json.objectField("path");
            try json.write(prior_path);
            try json.objectField("sha256");
            try json.write(prior_digest[0..]);
            try json.objectField("nonempty");
            try json.write(!trained.prior.isEmpty());
            try json.objectField("context_counts_by_backoff_level");
            try json.write(context_counts);
            try json.endObject();
            try json.objectField("configuration");
            try json.beginObject();
            try json.objectField("max_tensors");
            try json.write(n_sample);
            try json.objectField("seed");
            try json.write(calibrate.DEFAULT_SEED);
            try json.objectField("requested_threads");
            try json.write(n_threads);
            try json.objectField("threads_used");
            try json.write(trained.threads_used);
            try json.objectField("search");
            try json.beginObject();
            try report_mod.writeSearchConfigFields(&json, search_options);
            try json.endObject();
            try json.endObject();
            try json.objectField("observed");
            try json.beginObject();
            try json.objectField("available_tensors");
            try json.write(loaded.tensors.len);
            try json.objectField("sampled_tensors");
            try json.write(trained.sampled);
            try json.objectField("training_wall_ms");
            try json.write(ms);
            try json.endObject();
            try json.endObject();
            try out.writeByte('\n');
        },
    }
}

// ==================== compress ====================

fn cmdCompress(
    io: std.Io,
    out: *std.Io.Writer,
    in_path: []const u8,
    out_path: []const u8,
    prior_path: ?[]const u8,
    jobs: ?usize,
    mode: PlanMode,
    search_options: search.Options,
) !void {
    const alloc = std.heap.smp_allocator;

    var loaded = try safetensors.loadFromPath(alloc, io, in_path);
    defer loaded.deinitMmap(alloc, io);
    try checkTensors(loaded.tensors);

    var pr = try loadPrior(alloc, if (mode == .search) prior_path else null);
    defer pr.deinit(alloc);

    const blocks = try planAll(alloc, loaded.tensors);
    defer alloc.free(blocks);

    const n_threads = threadCount(jobs);
    try out.print("compress: {d} tensors, {d} blocks, {d} threads, plan={s}, prior={s}\n", .{
        loaded.tensors.len,
        blocks.len,
        n_threads,
        @tagName(mode),
        if (mode == .search) prior_path orelse "uniform" else "none",
    });
    try out.flush();

    const t0 = std.Io.Timestamp.now(io, .awake);
    const plans = try synthesizePlans(alloc, loaded.tensors, &pr, n_threads, mode, search_options);
    defer freePlans(alloc, plans);
    const metas = try archive.tensorMetas(alloc, loaded.tensors, blocks);
    defer alloc.free(metas);

    var atomic = try std.Io.Dir.cwd().createFileAtomic(io, out_path, .{ .replace = true });
    defer atomic.deinit(io);
    const file_buf = try alloc.alloc(u8, 4 << 20);
    defer alloc.free(file_buf);
    var writer = atomic.file.writer(io, file_buf);
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
    try out.print("wrote {s}: {d} -> {d} bytes ({d:.3}x)\n", .{ out_path, raw, written, report_mod.ratio(raw, written) });
}

// ==================== decompress / verify ====================

const DecodeJob = struct {
    next: std.atomic.Value(usize),
    fails: std.atomic.Value(usize),
    blocks: []const archive.ParsedBlock,
    streams: []?Stream,
    alloc: Allocator,

    pub fn run(self: *DecodeJob) void {
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

const DecodePool = pool_mod.BatchPool(DecodeJob);

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
        if (pool) |p| p.run(&job) else if (n == 1) job.run() else try pool_mod.runWorkers(alloc, n, &job, DecodeJob.run);
    }
    if (job.fails.load(.acquire) > 0) return error.DecompressionFailed;
    return streams;
}

fn freeStreams(alloc: Allocator, streams: []?Stream) void {
    for (streams) |*stream| if (stream.*) |*value| value.deinit(alloc);
    alloc.free(streams);
}

/// Drains decoded batches on a background thread so the next batch decodes
/// while the current one is still going to disk. One writer at a time keeps
/// the output ordered.
const WritePipe = struct {
    alloc: Allocator,
    writer: *std.Io.Writer,
    thread: ?std.Thread = null,
    batch: []?Stream = &.{},
    written: u64 = 0,
    err: ?anyerror = null,

    fn write(self: *WritePipe, batch: []?Stream) !void {
        for (batch) |maybe| {
            const stream = maybe.?;
            const data = stream.data[0 .. stream.count * stream.elemBytes()];
            try self.writer.writeAll(data);
            self.written += data.len;
        }
    }

    fn drain(self: *WritePipe) void {
        self.write(self.batch) catch |e| {
            self.err = e;
        };
    }

    fn join(self: *WritePipe) void {
        const thread = self.thread orelse return;
        thread.join();
        self.thread = null;
        freeStreams(self.alloc, self.batch);
        self.batch = &.{};
    }

    fn submit(self: *WritePipe, batch: []?Stream) !void {
        self.join();
        if (self.err) |e| {
            freeStreams(self.alloc, batch);
            return e;
        }
        self.batch = batch;
        self.thread = std.Thread.spawn(.{}, drain, .{self}) catch |e| {
            freeStreams(self.alloc, self.batch);
            self.batch = &.{};
            return e;
        };
    }

    fn finish(self: *WritePipe) !void {
        self.join();
        if (self.err) |e| return e;
    }
};

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
    const file_buf = try alloc.alloc(u8, 4 << 20);
    defer alloc.free(file_buf);
    var writer = atomic.file.writer(io, file_buf);
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

    var expected: u64 = 0;
    var remaining: usize = 0;
    for (loaded.parsed.tensors, metas) |tensor, meta| {
        remaining += tensor.n_blocks;
        expected += meta.byte_len;
    }

    var pipe: WritePipe = .{ .alloc = alloc, .writer = &writer.interface };
    defer pipe.join();

    const batch_size = n_threads * 16;
    var frame_pos: usize = 0;
    var lengths: archive.TensorLengths = .{};
    while (remaining > 0) {
        const n = @min(batch_size, remaining);
        const blocks = try alloc.alloc(archive.ParsedBlock, n);
        defer alloc.free(blocks);
        for (blocks) |*block| block.* = try archive.nextBlock(loaded.parsed.frames, &frame_pos);
        const streams = try decodeBlocks(alloc, blocks, n_threads, &decode_pool);
        lengths.accept(loaded.parsed.tensors, streams) catch |err| {
            freeStreams(alloc, streams);
            return err;
        };
        if (n_threads == 1) {
            defer freeStreams(alloc, streams);
            try pipe.write(streams);
        } else {
            try pipe.submit(streams);
        }
        remaining -= n;
    }
    try pipe.finish();
    try lengths.finish(loaded.parsed.tensors);
    if (pipe.written != expected) return error.ShapeDataMismatch;
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
        var frame_pos: usize = t.frame_start;
        for (0..t.n_blocks) |_| {
            const block = try archive.nextBlock(loaded.parsed.frames, &frame_pos);
            var stream = try archive.decodeBlock(alloc, block);
            defer stream.deinit(alloc);
            const data = stream.data[0 .. stream.count * stream.elemBytes()];
            if (data.len > found.?.data.len - off or !std.mem.eql(u8, data, found.?.data[off..][0..data.len])) {
                matches = false;
                break;
            }
            off += data.len;
        }
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


fn cmdBench(
    io: std.Io,
    out: *std.Io.Writer,
    in_path: []const u8,
    prior_path: ?[]const u8,
    jobs: ?usize,
    mode: PlanMode,
    format: ReportFormat,
    search_options: search.Options,
) !void {
    const alloc = std.heap.smp_allocator;

    var loaded = try safetensors.loadFromPath(alloc, io, in_path);
    defer loaded.deinitMmap(alloc, io);
    try checkTensors(loaded.tensors);
    var input_digest: ?[64]u8 = null;
    if (format == .json) input_digest = try sha256File(io, in_path);

    var pr = try loadPrior(alloc, if (mode == .search) prior_path else null);
    defer pr.deinit(alloc);
    const learned_prior = mode == .search and !pr.isEmpty();
    var prior_digest: ?[64]u8 = null;
    if (mode == .search) {
        if (prior_path) |path| prior_digest = try sha256File(io, path);
    }
    const prior_counts: [3]usize = .{
        pr.levels[0].count(),
        pr.levels[1].count(),
        pr.levels[2].count(),
    };
    const prior_digest_slice: ?[]const u8 = if (prior_digest) |*digest| digest[0..] else null;

    const blocks = try planAll(alloc, loaded.tensors);
    defer alloc.free(blocks);

    const n_threads = threadCount(jobs);
    if (format == .text) {
        var nonempty_tensors: usize = 0;
        for (loaded.tensors) |tensor| nonempty_tensors += @intFromBool(tensor.view.numel() > 0);
        try out.print("=== brevis bench: {s} ({d} tensors, {d} blocks, {d} requested threads, {d}/{d} planning/encoding workers, {s}) ===\n", .{
            in_path,
            loaded.tensors.len,
            blocks.len,
            n_threads,
            if (nonempty_tensors == 0) 0 else @min(n_threads, nonempty_tensors),
            if (blocks.len == 0) 0 else @min(n_threads, blocks.len),
            modeName(mode, learned_prior),
        });
        try out.flush();
    }

    const planning_start = std.Io.Timestamp.now(io, .awake);
    const plans = try synthesizePlans(alloc, loaded.tensors, &pr, n_threads, mode, search_options);
    defer freePlans(alloc, plans);
    const planning_ms = planning_start.durationTo(.now(io, .awake)).toMilliseconds();

    const encoding_start = std.Io.Timestamp.now(io, .awake);
    const results = try encodeBlocks(alloc, loaded.tensors, blocks, plans, n_threads, null);
    defer freeResults(alloc, results);
    const encoding_ms = encoding_start.durationTo(.now(io, .awake)).toMilliseconds();

    switch (format) {
        .text => {
            try report_mod.report(alloc, out, blocks, results);
            try out.print("planning wall time: {d}ms\nencoding wall time: {d}ms\n", .{ planning_ms, encoding_ms });
        },
        .json => try report_mod.reportJson(
            alloc,
            out,
            in_path,
            loaded.bytes,
            loaded.bytes.len,
            if (input_digest) |*digest| digest[0..] else unreachable,
            loaded.tensors,
            blocks,
            plans,
            results,
            report_mod.Mode{
                .name = modeName(mode, learned_prior),
                .plan_is_search = mode == .search,
            },
            prior_path,
            prior_digest_slice,
            learned_prior,
            prior_counts,
            n_threads,
            search_options,
            planning_ms,
            encoding_ms,
        ),
    }
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
            .u8, .f8_e4m3, .f8_e5m2 => buf[i] = r.intRangeAtMost(u8, 0, 31),
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

    var pr: prior.Prior = .empty;
    defer pr.deinit(alloc);

    const blocks = try planAll(alloc, tensors.items);
    defer alloc.free(blocks);

    try out.print("=== brevis demo: {d} synthetic tensors, {d} blocks ===\n", .{ tensors.items.len, blocks.len });
    try out.flush();

    const t0 = std.Io.Timestamp.now(io, .awake);
    const results = try synthesizeBlocks(alloc, tensors.items, blocks, &pr, threadCount(null), .search, .{});
    defer freeResults(alloc, results);
    const ms = t0.durationTo(.now(io, .awake)).toMilliseconds();

    try report_mod.report(alloc, out, blocks, results);
    try out.print("synthesis wall time: {d}ms\n", .{ms});
}
