//! Streaming training pipeline over the low-level grammar.
//!
//! For each training tensor:
//!   1. Run `astar.synthesize` (B&B with admissible heuristic) on the
//!      tensor's bytes (treated as a 16-bit stream for fp16/bf16).
//!   2. Walk the resulting PNode tree. Update:
//!         - PHOG counts: per (parent_op, child_op_kind) pair
//!         - Subtree counts: per canonical-form subtree of size 2..4
//!   3. After all tensors processed:
//!         - For each subtree, evaluate MDL benefit
//!         - Mark top-K as "promoted" macros
//!         - Emit a JSON report
//!
//! No Zig source regeneration in this MVP — the report tells the user
//! *what would* be promoted; integrating back into the runtime grammar is
//! left for the next iteration.

const std = @import("std");
const types = @import("types.zig");
const astar = @import("astar.zig");
const lowlevel = @import("lowlevel.zig");

const Allocator = types.Allocator;

// =================== canonical subtree key ===================
//
// We canonicalize a (sub-)PNode tree into a deterministic byte string so it
// can be used as a HashMap key. Format:
//     terminal:   0x00 OpKind
//     chain:      0x01 OpKind params(u32 LE) <children...>
//     split:      0x02 OpKind params(u32 LE) <hi> <lo>
// Leaves of the canonicalized form (when we limit to size N) are marked
// with 0xFF and treated as wildcards.

fn writeCanonical(node: *const astar.PNode, out: *std.ArrayList(u8), alloc: Allocator, depth: u8) Allocator.Error!void {
    if (depth == 0) {
        try out.append(alloc, 0xFF); // wildcard
        return;
    }
    switch (node.*) {
        .terminal => |t| {
            try out.append(alloc, 0x00);
            try out.append(alloc, @intFromEnum(t.kind));
        },
        .macro => {
            // Macros are atomic in canonical form: 0x03 macro_idx (u32 LE)
            try out.append(alloc, 0x03);
            var idx_buf: [4]u8 = undefined;
            std.mem.writeInt(u32, &idx_buf, node.macro.macro_idx, .little);
            try out.appendSlice(alloc, &idx_buf);
        },
        .chain => |c| {
            try out.append(alloc, 0x01);
            try out.append(alloc, @intFromEnum(c.op.kind));
            var pbuf: [4]u8 = undefined;
            std.mem.writeInt(u32, &pbuf, c.op.params.raw, .little);
            try out.appendSlice(alloc, &pbuf);
            try writeCanonical(c.next, out, alloc, depth - 1);
        },
        .split => |s| {
            try out.append(alloc, 0x02);
            try out.append(alloc, @intFromEnum(s.op.kind));
            var pbuf: [4]u8 = undefined;
            std.mem.writeInt(u32, &pbuf, s.op.params.raw, .little);
            try out.appendSlice(alloc, &pbuf);
            try writeCanonical(s.hi, out, alloc, depth - 1);
            try writeCanonical(s.lo, out, alloc, depth - 1);
        },
    }
}

fn nodeSize(node: *const astar.PNode) u32 {
    return switch (node.*) {
        .terminal => 1,
        .macro => 1, // counted as a single atom for subtree mining purposes
        .chain => |c| 1 + nodeSize(c.next),
        .split => |s| 1 + nodeSize(s.hi) + nodeSize(s.lo),
    };
}

// =================== PHOG and subtree counters ===================

pub const Counters = struct {
    /// (parent_kind << 8 | child_kind) → count
    phog_pair_counts: std.AutoHashMap(u16, u32),
    /// canonical_subtree_bytes → (count, size)
    subtree_counts: std.StringHashMap(SubtreeStats),
    /// total tensors processed
    n_tensors: u32 = 0,
    /// total bits across all tensors (for context)
    total_compressed_bits: u64 = 0,
    total_raw_bits: u64 = 0,
    alloc: Allocator,

    pub const SubtreeStats = struct {
        count: u32,
        size: u32,
    };

    pub fn init(alloc: Allocator) Counters {
        return .{
            .phog_pair_counts = .init(alloc),
            .subtree_counts = .init(alloc),
            .alloc = alloc,
        };
    }

    pub fn deinit(self: *Counters) void {
        self.phog_pair_counts.deinit();
        var it = self.subtree_counts.iterator();
        while (it.next()) |e| self.alloc.free(e.key_ptr.*);
        self.subtree_counts.deinit();
    }

    fn bumpPair(self: *Counters, parent: lowlevel.OpKind, child: lowlevel.OpKind) !void {
        const key = (@as(u16, @intFromEnum(parent)) << 8) | @as(u16, @intFromEnum(child));
        const gop = try self.phog_pair_counts.getOrPut(key);
        if (!gop.found_existing) gop.value_ptr.* = 0;
        gop.value_ptr.* += 1;
    }

    fn bumpSubtree(self: *Counters, key_bytes: []const u8, size: u32) !void {
        const gop = try self.subtree_counts.getOrPut(key_bytes);
        if (!gop.found_existing) {
            // Need owned key.
            const owned = try self.alloc.alloc(u8, key_bytes.len);
            @memcpy(owned, key_bytes);
            gop.key_ptr.* = owned;
            gop.value_ptr.* = .{ .count = 0, .size = size };
        }
        gop.value_ptr.count += 1;
    }
};

// =================== walk a PNode and update counters ===================

fn walk(node: *const astar.PNode, parent_kind: ?lowlevel.OpKind, c: *Counters) !void {
    // Bump phog pair count.
    if (parent_kind) |pk| {
        const maybe_child: ?lowlevel.OpKind = switch (node.*) {
            .terminal => |t| t.kind,
            .chain => |ch| ch.op.kind,
            .split => |s| s.op.kind,
            .macro => null, // macros aren't tracked in PHOG pair counts in this MVP
        };
        if (maybe_child) |ck| try c.bumpPair(pk, ck);
    }

    // Bump subtree count (cap depth at 3 for MVP — limits canonical size).
    var key_buf: std.ArrayList(u8) = .empty;
    defer key_buf.deinit(c.alloc);
    try writeCanonical(node, &key_buf, c.alloc, 3);
    const sz = nodeSize(node);
    if (sz >= 2 and sz <= 4) {
        try c.bumpSubtree(key_buf.items, sz);
    }

    // Recurse.
    switch (node.*) {
        .terminal => {},
        .macro => {},
        .chain => |ch| try walk(ch.next, ch.op.kind, c),
        .split => |s| {
            try walk(s.hi, s.op.kind, c);
            try walk(s.lo, s.op.kind, c);
        },
    }
}

// =================== main training entry ===================

pub const TrainResult = struct {
    counters: Counters,
    /// MDL-promoted subtrees, sorted by net benefit descending.
    promoted: []Promoted,
    alloc: Allocator,

    pub const Promoted = struct {
        canonical_key_hex: []u8, // hex-encoded for JSON dump
        size: u32,
        count: u32,
        mdl_benefit: i64, // count * (size - 1) - macro_def_cost
    };

    pub fn deinit(self: *TrainResult) void {
        self.counters.deinit();
        for (self.promoted) |p| self.alloc.free(p.canonical_key_hex);
        self.alloc.free(self.promoted);
    }
};

pub const TrainOpts = struct {
    max_depth: u8 = 4,
    max_nodes_explored: u64 = 200_000,
    /// MDL: macro definition is roughly this many "production tokens" in the
    /// emitted Zig source. We charge this as the cost of adding any macro.
    macro_def_cost: u32 = 5,
    /// Hard floor on subtree count to consider for promotion.
    min_count_for_promotion: u32 = 5,
    /// For large tensors: subsample to this many elements before running B&B.
    /// The resulting program is what would be best for this distribution; we
    /// don't actually compress the full tensor with it (training just cares
    /// about which programs/subtrees emerge). 0 = no subsample.
    subsample_elements: usize = 0,
};

// Per-tensor result that workers populate; main thread merges into Counters
// at the end. We can't share Counters directly because HashMap isn't threadsafe.
const PerTensor = struct {
    done: bool = false,
    program: ?*astar.PNode = null,
    cost: u64 = 0,
    raw_bits: u64 = 0,
    valid: bool = false,
};

const Worker = struct {
    next: std.atomic.Value(usize),
    tensors: []const types.TensorView,
    out: []PerTensor,
    opts: TrainOpts,

    fn run(self: *Worker) void {
        const work_alloc = std.heap.smp_allocator;
        while (true) {
            const idx = self.next.fetchAdd(1, .acq_rel);
            if (idx >= self.tensors.len) return;
            const t = self.tensors[idx];
            if (!t.dtype.isFloat16Like()) {
                @atomicStore(bool, &self.out[idx].done, true, .release);
                continue;
            }
            const total_count: usize = @divExact(t.data.len, 2);
            const use_count: usize = if (self.opts.subsample_elements > 0 and total_count > self.opts.subsample_elements)
                self.opts.subsample_elements
            else
                total_count;
            const stream: types.Stream = .{
                .data = t.data[0 .. use_count * 2],
                .count = use_count,
                .bits_per_elem = 16,
                .owns_data = false,
            };
            const best = astar.synthesize(work_alloc, stream, .{
                .max_depth = self.opts.max_depth,
                .max_nodes_explored = self.opts.max_nodes_explored,
            }) catch {
                @atomicStore(bool, &self.out[idx].done, true, .release);
                continue;
            };
            self.out[idx].program = best.program;
            self.out[idx].cost = best.cost;
            // raw_bits should match what B&B actually saw (the subsample),
            // otherwise ratio = (full / subsample-compressed) is misleading.
            self.out[idx].raw_bits = @as(u64, use_count) * @as(u64, stream.bits_per_elem);
            self.out[idx].valid = best.program != null;
            @atomicStore(bool, &self.out[idx].done, true, .release);
        }
    }
};

pub fn trainOnTensors(
    alloc: Allocator,
    tensors: []const types.TensorView,
    opts: TrainOpts,
) !TrainResult {
    const work_alloc = std.heap.smp_allocator;
    var counters = Counters.init(alloc);

    // 1. Parallel B&B per tensor.
    const n_threads: usize = std.Thread.getCpuCount() catch 8;
    const per_tensor = try work_alloc.alloc(PerTensor, tensors.len);
    defer work_alloc.free(per_tensor);
    for (per_tensor) |*p| p.* = .{};

    var worker: Worker = .{
        .next = .init(0),
        .tensors = tensors,
        .out = per_tensor,
        .opts = opts,
    };
    const threads = try work_alloc.alloc(std.Thread, n_threads);
    defer work_alloc.free(threads);
    for (threads) |*th| th.* = try std.Thread.spawn(.{}, Worker.run, .{&worker});
    for (threads) |th| th.join();

    // 2. Sequentially merge per-tensor results into the (non-threadsafe) counters.
    for (per_tensor) |p| {
        if (!p.valid) continue;
        const root = p.program.?;
        defer {
            root.deinit(alloc);
            alloc.destroy(root);
        }
        counters.n_tensors += 1;
        counters.total_raw_bits += p.raw_bits;
        counters.total_compressed_bits += p.cost;
        try walk(root, null, &counters);
    }

    // MDL promotion.
    var promoted_list: std.ArrayList(TrainResult.Promoted) = .empty;
    defer promoted_list.deinit(alloc);

    var it = counters.subtree_counts.iterator();
    while (it.next()) |e| {
        const stats = e.value_ptr.*;
        if (stats.count < opts.min_count_for_promotion) continue;
        const benefit: i64 = @as(i64, stats.count) * @as(i64, @intCast(stats.size - 1)) - @as(i64, opts.macro_def_cost);
        if (benefit <= 0) continue;
        const key = e.key_ptr.*;
        // Hex-encode.
        const hex = try alloc.alloc(u8, key.len * 2);
        const hex_chars = "0123456789abcdef";
        for (key, 0..) |b, i| {
            hex[i * 2] = hex_chars[b >> 4];
            hex[i * 2 + 1] = hex_chars[b & 0x0F];
        }
        try promoted_list.append(alloc, .{
            .canonical_key_hex = hex,
            .size = stats.size,
            .count = stats.count,
            .mdl_benefit = benefit,
        });
    }

    // Sort by benefit desc.
    std.mem.sort(TrainResult.Promoted, promoted_list.items, {}, struct {
        fn lt(_: void, a: TrainResult.Promoted, b: TrainResult.Promoted) bool {
            return a.mdl_benefit > b.mdl_benefit;
        }
    }.lt);

    return .{
        .counters = counters,
        .promoted = try promoted_list.toOwnedSlice(alloc),
        .alloc = alloc,
    };
}

// =================== JSON dump ===================

pub fn dumpReport(alloc: Allocator, result: *const TrainResult, w: *std.Io.Writer) !void {
    try w.print("{{\n  \"n_tensors\": {d},\n", .{result.counters.n_tensors});
    try w.print("  \"total_raw_bits\": {d},\n", .{result.counters.total_raw_bits});
    try w.print("  \"total_compressed_bits\": {d},\n", .{result.counters.total_compressed_bits});
    if (result.counters.total_compressed_bits > 0) {
        const ratio: f64 = @as(f64, @floatFromInt(result.counters.total_raw_bits)) / @as(f64, @floatFromInt(result.counters.total_compressed_bits));
        try w.print("  \"avg_ratio\": {d:.4},\n", .{ratio});
    }

    // PHOG pairs.
    try w.writeAll("  \"phog_pair_counts\": [\n");
    var pit = result.counters.phog_pair_counts.iterator();
    var first = true;
    while (pit.next()) |e| {
        if (!first) try w.writeAll(",\n");
        first = false;
        const parent: u8 = @intCast((e.key_ptr.* >> 8) & 0xFF);
        const child: u8 = @intCast(e.key_ptr.* & 0xFF);
        try w.print("    {{\"parent\": {d}, \"child\": {d}, \"count\": {d}}}", .{ parent, child, e.value_ptr.* });
    }
    try w.writeAll("\n  ],\n");

    // Promoted subtrees.
    try w.print("  \"promoted_macros\": {{\n    \"count\": {d},\n    \"items\": [\n", .{result.promoted.len});
    for (result.promoted, 0..) |p, i| {
        if (i > 0) try w.writeAll(",\n");
        _ = alloc;
        try w.print("      {{\"size\": {d}, \"count\": {d}, \"mdl_benefit\": {d}, \"canonical\": \"{s}\"}}", .{ p.size, p.count, p.mdl_benefit, p.canonical_key_hex });
    }
    try w.writeAll("\n    ]\n  }\n}\n");
    try w.flush();
}
