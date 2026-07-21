const std = @import("std");
const types = @import("types.zig");
const ops = @import("ops.zig");
const program = @import("program.zig");
const prior = @import("prior.zig");
const search = @import("search.zig");
const safetensors = @import("safetensors.zig");

const Allocator = std.mem.Allocator;
const Stream = types.Stream;
const Dtype = types.Dtype;
const ROOT_PARENT: u8 = 255;
const TOP_CANDIDATES: usize = 8;
const PROBE_BLOCKS: usize = 4;
pub const DEFAULT_TENSORS: usize = 200;

pub const Options = struct {
    max_tensors: usize = DEFAULT_TENSORS,
    threads: usize = 1,
    seed: u64 = 0x5EED_B10C,
};

pub const Result = struct {
    prior: prior.Prior,
    sampled: usize,
};

fn pickTensors(alloc: Allocator, tensors: []const safetensors.Tensor, n: usize, seed: u64) ![]u32 {
    var strata: std.AutoHashMapUnmanaged(u16, std.ArrayList(u32)) = .empty;
    defer {
        var values = strata.valueIterator();
        while (values.next()) |list| list.deinit(alloc);
        strata.deinit(alloc);
    }
    for (tensors, 0..) |tensor, i| {
        const count = tensor.view.numel();
        if (count == 0) continue;
        const lg: u16 = @intCast(std.math.log2_int(usize, count));
        const key = (@as(u16, @intFromEnum(tensor.view.dtype)) << 8) | lg;
        const entry = try strata.getOrPut(alloc, key);
        if (!entry.found_existing) entry.value_ptr.* = .empty;
        try entry.value_ptr.append(alloc, @intCast(i));
    }

    var keys: std.ArrayList(u16) = .empty;
    defer keys.deinit(alloc);
    var key_it = strata.keyIterator();
    while (key_it.next()) |key| try keys.append(alloc, key.*);
    std.mem.sort(u16, keys.items, {}, std.sort.asc(u16));

    var prng = std.Random.DefaultPrng.init(seed);
    for (keys.items) |key| prng.random().shuffle(u32, strata.getPtr(key).?.items);

    var picked: std.ArrayList(u32) = .empty;
    errdefer picked.deinit(alloc);
    var cursor: usize = 0;
    while (picked.items.len < n) : (cursor += 1) {
        var progressed = false;
        for (keys.items) |key| {
            const group = strata.getPtr(key).?.items;
            if (cursor >= group.len) continue;
            try picked.append(alloc, group[cursor]);
            progressed = true;
            if (picked.items.len == n) break;
        }
        if (!progressed) break;
    }
    return picked.toOwnedSlice(alloc);
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
    weight: f64,
) !void {
    try counts.add(alloc, prior.Context.fromStream(in, dtype, slot, depth, parent_op), node.op, weight);
    if (node.op.isTerminal()) return;

    var outs: std.ArrayList(Stream) = .empty;
    defer {
        for (outs.items) |*stream| stream.deinit(alloc);
        outs.deinit(alloc);
    }
    var side: ops.SideInfo = .none;
    defer side.deinit(alloc);
    try ops.forward(alloc, node.op, node.params, in, &outs, &side);
    for (node.children, outs.items, 0..) |child, stream, child_slot|
        try accumulate(alloc, counts, child, stream, dtype, @intCast(child_slot), depth + 1, @intFromEnum(node.op), weight);
}

const Job = struct {
    failed: std.atomic.Value(usize) = .init(0),
    alloc: Allocator,
    tensors: []const safetensors.Tensor,
    picked: []const u32,
    counts: []prior.Counts,

    fn run(self: *Job, slot: usize) void {
        var next = slot;
        while (next < self.picked.len) : (next += self.counts.len) {
            self.trainOne(slot, self.tensors[self.picked[next]]) catch |err| {
                std.debug.print("calibrate tensor {d}: {t}\n", .{ self.picked[next], err });
                _ = self.failed.fetchAdd(1, .acq_rel);
            };
        }
    }

    fn candidateCost(self: *Job, tensor: safetensors.Tensor, blocks: []const types.Block, candidate: search.Result) !u64 {
        const n = @min(PROBE_BLOCKS, blocks.len);
        const plan: search.Plan = .{ .root = candidate.node, .expanded = candidate.expanded };
        var total: u64 = 0;
        for (0..n) |probe| {
            const index = if (n == 1) 0 else probe * (blocks.len - 1) / (n - 1);
            const stream = blocks[index].asStream(tensor.view.data);
            var encoded = try search.encode(self.alloc, &plan, stream, tensor.view.dtype);
            defer encoded.deinit(self.alloc);
            total += encoded.bytes;
        }
        return total;
    }

    fn trainOne(self: *Job, slot: usize, tensor: safetensors.Tensor) !void {
        const view = tensor.view;
        const full: Stream = .{
            .data = view.data,
            .count = view.numel(),
            .bits_per_elem = view.dtype.bitWidth(),
            .owns_data = false,
        };
        var sample = try search.planningSample(self.alloc, full, ops.SEARCH_SAMPLE_ELEMS);
        defer sample.deinit(self.alloc);
        const candidates = try search.synthesizeAll(self.alloc, sample, view.dtype, .{
            .enumerate_all = true,
            .sample_elems = 0,
        });
        defer {
            for (candidates) |*candidate| candidate.deinit(self.alloc);
            self.alloc.free(candidates);
        }

        const inner: usize = if (view.shape.len == 0) 1 else @intCast(view.shape[view.shape.len - 1]);
        const blocks = try types.planBlocks(self.alloc, 0, view.dtype, view.numel(), inner);
        defer self.alloc.free(blocks);
        var best_index: usize = 0;
        var best_cost = try self.candidateCost(tensor, blocks, candidates[0]);
        for (candidates[1..@min(TOP_CANDIDATES, candidates.len)], 1..) |candidate, i| {
            const cost = try self.candidateCost(tensor, blocks, candidate);
            if (cost < best_cost) {
                best_cost = cost;
                best_index = i;
            }
        }
        try accumulate(self.alloc, &self.counts[slot], candidates[best_index].node, sample, tensor.view.dtype, 0, 0, ROOT_PARENT, 1);
    }
};

fn mergeCounts(alloc: Allocator, dst: *prior.Counts, src: prior.Counts) !void {
    for (0..3) |level| {
        var it = src.levels[level].iterator();
        while (it.next()) |entry| {
            const target = try dst.levels[level].getOrPut(alloc, entry.key_ptr.*);
            if (!target.found_existing) target.value_ptr.* = @splat(0);
            for (target.value_ptr, entry.value_ptr.*) |*value, add| value.* += add;
        }
    }
}

pub fn train(alloc: Allocator, tensors: []const safetensors.Tensor, options: Options) !Result {
    const picked = try pickTensors(alloc, tensors, options.max_tensors, options.seed);
    defer alloc.free(picked);
    const n_threads = @max(@as(usize, 1), @min(options.threads, @max(picked.len, 1)));
    const per_thread = try alloc.alloc(prior.Counts, n_threads);
    defer {
        for (per_thread) |*counts| counts.deinit(alloc);
        alloc.free(per_thread);
    }
    for (per_thread) |*counts| counts.* = prior.Counts.init(alloc);

    var job: Job = .{ .alloc = alloc, .tensors = tensors, .picked = picked, .counts = per_thread };
    const threads = try alloc.alloc(std.Thread, n_threads);
    defer alloc.free(threads);
    var spawned: usize = 0;
    errdefer for (threads[0..spawned]) |thread| thread.join();
    for (threads, 0..) |*thread, slot| {
        thread.* = try std.Thread.spawn(.{}, Job.run, .{ &job, slot });
        spawned += 1;
    }
    for (threads) |thread| thread.join();
    if (job.failed.load(.acquire) != 0) return error.CalibrationFailed;

    var counts = prior.Counts.init(alloc);
    defer counts.deinit(alloc);
    for (per_thread) |thread_counts| try mergeCounts(alloc, &counts, thread_counts);
    return .{ .prior = try counts.toPrior(alloc), .sampled = picked.len };
}

test "calibration trains a tensor planning sample" {
    const alloc = std.testing.allocator;
    const data = try alloc.alloc(u8, 8192);
    defer alloc.free(data);
    for (data, 0..) |*byte, i| byte.* = @truncate(i *% 29);
    const shape = [_]u64{data.len};
    const tensor: safetensors.Tensor = .{
        .name = "sample",
        .view = .{ .data = data, .shape = &shape, .dtype = .u8 },
    };

    var trained = try train(alloc, &.{tensor}, .{});
    defer trained.prior.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 1), trained.sampled);
    try std.testing.expect(trained.prior.levels[0].count() > 0);
}
