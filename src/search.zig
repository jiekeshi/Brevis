//! PHOG-guided A*: grammar description length orders expansion; an independent
//! encoded-byte lower bound prunes programs that cannot beat the incumbent.

const std = @import("std");
const types = @import("types.zig");
const ops = @import("ops.zig");
const codec = @import("codec.zig");
const program = @import("program.zig");
const prior = @import("prior.zig");

const Allocator = std.mem.Allocator;
const Stream = types.Stream;
const Dtype = types.Dtype;
const Node = program.Node;
const OpKind = ops.OpKind;

/// Smallest serialized size any node can have.
const MIN_NODE_BYTES: u64 = 5;
/// Exact serialized header size: op | params | side tag | n_children.
const NODE_HDR: usize = 7;
const ROOT_PARENT: u8 = 255;
const MAX_ARITY: usize = 32;
const MAX_PRODUCTIONS: usize = 32;

pub const Options = struct {
    enumerate_all: bool = false,
    max_expansions: usize = ops.MAX_EXPANSIONS,
    /// Ignored when collecting candidates within the expansion budget.
    max_realizations: usize = ops.MAX_REALIZATIONS,
    max_nodes: usize = ops.MAX_NODES,
    max_depth: u8 = ops.K_TRANSFORM_LAYERS,
    /// Search on at most this many elements, then realize the winner on the
    /// full block. Every candidate costs O(sample), not O(block). 0 disables.
    sample_elems: usize = ops.SEARCH_SAMPLE_ELEMS,
};

pub const Result = struct {
    node: Node,
    payload: []u8,
    bytes: usize,
    expanded: usize,

    pub fn deinit(self: *Result, alloc: Allocator) void {
        self.node.deinit(alloc);
        alloc.free(self.payload);
        self.payload = &.{};
    }
};

pub const Plan = struct {
    root: Node,
    expanded: usize,

    pub fn deinit(self: *Plan, alloc: Allocator) void {
        self.root.deinit(alloc);
    }
};

// ==================== partial programs ====================

const Hole = struct {
    depth: u8,
    slot: u8,
    parent_op: u8,
    context: ?prior.Context,
    score_lb: u32,
    /// Bits this hole is guaranteed to cost. Non-zero only once transforms are
    /// no longer legal here: above that layer a reversible transform can drive
    /// the empirical entropy arbitrarily low, so the only honest bound is 0.
    lb_bits: u64,
};

const PNode = union(enum) {
    hole: Hole,
    filled: struct { op: OpKind, params: u32, kids: []PNode },

    fn deinit(self: *PNode, alloc: Allocator) void {
        switch (self.*) {
            .hole => {},
            .filled => |*f| {
                for (f.kids) |*k| k.deinit(alloc);
                if (f.kids.len > 0) alloc.free(f.kids);
                f.kids = &.{};
            },
        }
    }

    fn clone(self: PNode, alloc: Allocator) Allocator.Error!PNode {
        const f = switch (self) {
            .hole => return self,
            .filled => |f| f,
        };
        const kids = try alloc.alloc(PNode, f.kids.len);
        var filled: usize = 0;
        errdefer {
            for (kids[0..filled]) |*k| k.deinit(alloc);
            alloc.free(kids);
        }
        for (f.kids, 0..) |k, i| {
            kids[i] = try k.clone(alloc);
            filled = i + 1;
        }
        return .{ .filled = .{ .op = f.op, .params = f.params, .kids = kids } };
    }
};

const Partial = struct {
    root: PNode,
    p: u64,
    f: u64,
    g_bytes: usize,
    lb_sum: u64,
    score_lb_sum: u64,
    n_nodes: usize,
    n_holes: usize,

    fn deinit(self: *Partial, alloc: Allocator) void {
        self.root.deinit(alloc);
    }
};

fn cmpPartial(_: void, a: Partial, b: Partial) std.math.Order {
    if (a.f != b.f) return std.math.order(a.f, b.f);
    const ab = boundBytes(a.g_bytes, a.lb_sum, a.n_holes);
    const bb = boundBytes(b.g_bytes, b.lb_sum, b.n_holes);
    if (ab != bb) return std.math.order(ab, bb);
    return std.math.order(a.p, b.p);
}

const Queue = std.PriorityQueue(Partial, void, cmpPartial);

// ==================== bounds ====================

/// Empirical zeroth-order entropy of `s` in bits, rounded down.
fn shannonBits(alloc: Allocator, s: Stream) !u64 {
    if (s.count == 0) return 0;
    var hist = try codec.buildHistogram(alloc, s);
    defer hist.deinit(alloc);
    return codec.entropyBits(hist, s.count);
}

fn holeBits(alloc: Allocator, s: Stream, depth: u8) !u64 {
    if (depth < ops.K_TRANSFORM_LAYERS) return 0;
    return shannonBits(alloc, s);
}

fn lowerBound(lb_sum: u64, n_holes: usize) u64 {
    return lb_sum + @as(u64, n_holes) * MIN_NODE_BYTES * 8;
}

fn boundBytes(g_bytes: usize, lb_sum: u64, n_holes: usize) usize {
    return g_bytes + @as(usize, @intCast((lowerBound(lb_sum, n_holes) + 7) / 8));
}

/// Serialized bytes a terminal contributes: node header + side info + the
/// length-prefixed payload. Never encodes; see codec's closed-form section.
/// Returns null when the production cannot apply to this stream.
fn terminalCost(alloc: Allocator, prod: OpKind, params: u32, s: Stream, hist: codec.Histogram) !?usize {
    var payload: u64 = 0;
    var side: usize = 0;
    switch (prod) {
        .raw => {
            payload = s.data.len;
            side = 9;
        },
        .bitpack => {
            payload = (codec.bitpackCostBits(s, @intCast(params & 0xFF)) + 7) / 8;
            side = 9;
        },
        .huffman => {
            var t = try codec.huffmanFromHist(alloc, hist, s.bits_per_elem);
            defer t.deinit(alloc);
            payload = codec.huffmanPayloadBytes(t, hist);
            side = 13 + 5 * t.entries.len;
        },
        .rans => {
            var t = codec.ransFromHist(alloc, hist, s.count) catch |e| switch (e) {
                error.AlphabetTooLarge => return null,
                else => return e,
            };
            defer t.deinit(alloc);
            payload = codec.ransLowerBytes(t, hist);
            side = 13 + 8 * t.symbols.len;
        },
        else => unreachable,
    }
    return NODE_HDR + side + 8 + @as(usize, @intCast(payload));
}

/// Serialized side body size, tag byte excluded. Mirrors program.writeSide.
fn sideBody(side: ops.SideInfo) usize {
    return switch (side) {
        .none => 0,
        .terminal, .rle, .split => 9,
        .huffman => |h| 13 + 5 * h.table.entries.len,
        .rans => |r| 13 + 8 * r.table.symbols.len,
        .codebook => |c| 13 + 4 * c.syms.len,
        .sfloat => 9,
    };
}

// ==================== productions ====================

fn addProduction(out: *[MAX_PRODUCTIONS]OpKind, len: *usize, op: OpKind) void {
    out[len.*] = op;
    len.* += 1;
}

fn legalProductions(hole_bpe: u8, depth: u8, dtype: Dtype, is_root: bool, out: *[MAX_PRODUCTIONS]OpKind) []const OpKind {
    var len: usize = 0;
    addProduction(out, &len, .raw);
    addProduction(out, &len, .bitpack);
    if (hole_bpe <= ops.MAX_ENTROPY_BPE) {
        addProduction(out, &len, .huffman);
        addProduction(out, &len, .rans);
    }
    if (depth >= ops.K_TRANSFORM_LAYERS) return out[0..len];

    const elementwise = [_]OpKind{
        .xor_const, .add_const_mod, .xor_prev,    .diff_mod, .zigzag,
        .gray,      .rotate_bits,   .bit_reverse, .rle,      .deinterleave,
    };
    for (elementwise) |op| addProduction(out, &len, op);

    if (hole_bpe > 1) {
        addProduction(out, &len, .split_field);
        addProduction(out, &len, .topk_codebook);
        addProduction(out, &len, .bit_plane);
    }
    if (hole_bpe > 8) addProduction(out, &len, .byte_plane);
    if (is_root and dtype.isFloat()) addProduction(out, &len, .split_float);
    return out[0..len];
}

fn completionScoreLowerBound(
    pr: *const prior.Prior,
    ctx: prior.Context,
    bpe: u8,
    depth: u8,
    dtype: Dtype,
    is_root: bool,
) u32 {
    var storage: [MAX_PRODUCTIONS]OpKind = undefined;
    const productions = legalProductions(bpe, depth, dtype, is_root, &storage);
    var scores: [MAX_PRODUCTIONS]u32 = undefined;
    pr.scoreSet(ctx, productions, scores[0..productions.len]);
    var lower = scores[0];
    for (scores[1..productions.len]) |score| lower = @min(lower, score);
    return lower;
}

const Feat = struct { mode: u32, max_bits: u8 };

fn featOf(hist: codec.Histogram) Feat {
    var acc: u32 = 0;
    var mode: u32 = 0;
    var best: u64 = 0;
    for (hist.pairs) |p| {
        acc |= p.sym;
        if (p.count > best) {
            best = p.count;
            mode = p.sym;
        }
    }
    return .{ .mode = mode, .max_bits = if (acc == 0) 1 else @intCast(32 - @clz(acc)) };
}

/// One parameterisation per production, derived from the hole's own data.
/// Null means the op degenerates into the identity here and is skipped.
fn chooseParams(op: OpKind, s: Stream, dtype: Dtype, f: Feat) ?u32 {
    const k = s.bits_per_elem;
    return switch (op) {
        .bitpack => @min(f.max_bits, k),
        .xor_const => if (f.mode == 0) null else f.mode,
        .add_const_mod => blk: {
            const m = s.mask();
            const c = (m -% f.mode +% 1) & m;
            break :blk if (c == 0) null else c;
        },
        .rotate_bits => if (k < 2) null else @as(u32, k / 2),
        .split_field => blk: {
            if (k < 2) break :blk null;
            if (dtype.floatFields()) |ff| {
                if (ff.total == k) break :blk @as(u32, ff.mant) | (@as(u32, ff.exp) << 8);
            }
            const start: u32 = k / 2;
            break :blk start | ((@as(u32, k) - start) << 8);
        },
        .topk_codebook => 15,
        .deinterleave => if (s.count < 2) null else 2 | (1 << 16),
        .split_float => blk: {
            const ff = dtype.floatFields() orelse break :blk null;
            break :blk if (ff.total == k) @intFromEnum(dtype) else null;
        },
        else => 0,
    };
}

// ==================== skeleton -> program ====================

fn toNode(alloc: Allocator, sk: PNode) Allocator.Error!Node {
    const f = sk.filled;
    const kids = try alloc.alloc(Node, f.kids.len);
    var filled: usize = 0;
    errdefer {
        for (kids[0..filled]) |*c| c.deinit(alloc);
        alloc.free(kids);
    }
    for (f.kids, 0..) |k, i| {
        kids[i] = try toNode(alloc, k);
        filled = i + 1;
    }
    return .{ .op = f.op, .params = f.params, .children = kids };
}

fn dropPayloads(alloc: Allocator, n: *Node) void {
    if (n.payload_owned) alloc.free(n.payload);
    n.payload = &.{};
    n.payload_owned = false;
    for (n.children) |*c| dropPayloads(alloc, c);
}

fn dropRuntime(alloc: Allocator, node: *Node) void {
    if (node.payload_owned) alloc.free(node.payload);
    node.payload = &.{};
    node.payload_owned = false;
    node.side.deinit(alloc);
    for (node.children) |*child| dropRuntime(alloc, child);
}

/// Encode for real: exact bytes = bytecode + packed payload. The returned
/// node's payloads are non-owning views into `payload`.
fn encodeReal(alloc: Allocator, sk: PNode, in: Stream) !Result {
    var node = try toNode(alloc, sk);
    errdefer node.deinit(alloc);
    try program.execute(alloc, &node, in);

    const payload = try program.collectPayload(alloc, node);
    errdefer alloc.free(payload);
    const bc = try program.serialize(alloc, node);
    const bytes = payload.len + bc.len;
    alloc.free(bc);

    dropPayloads(alloc, &node);
    try program.distributePayload(&node, payload);
    return .{ .node = node, .payload = payload, .bytes = bytes, .expanded = 0 };
}

fn fitParams(alloc: Allocator, node: *Node, in: Stream, dtype: Dtype) !void {
    switch (node.op) {
        .bitpack => {
            var acc: u32 = 0;
            for (0..in.count) |i| acc |= in.getU32(i);
            node.params = if (acc == 0) 1 else @intCast(32 - @clz(acc));
        },
        .xor_const, .add_const_mod => {
            var hist = try codec.buildHistogram(alloc, in);
            defer hist.deinit(alloc);
            if (chooseParams(node.op, in, dtype, featOf(hist))) |params| node.params = params;
        },
        else => {},
    }
}

fn fitAndExecute(alloc: Allocator, node: *Node, in: Stream, dtype: Dtype) !void {
    try fitParams(alloc, node, in, dtype);
    if (node.op.isTerminal()) return program.execute(alloc, node, in);

    var outs: std.ArrayList(Stream) = .empty;
    defer {
        for (outs.items) |*s| s.deinit(alloc);
        outs.deinit(alloc);
    }
    try ops.forward(alloc, node.op, node.params, in, &outs, &node.side);
    std.debug.assert(outs.items.len == node.children.len);
    for (node.children, outs.items) |*child, out| try fitAndExecute(alloc, child, out, dtype);
}

fn encodeTemplate(alloc: Allocator, template: Node, in: Stream, dtype: Dtype) !Result {
    var node = try template.clone(alloc);
    errdefer node.deinit(alloc);
    dropRuntime(alloc, &node);
    try fitAndExecute(alloc, &node, in, dtype);

    const payload = try program.collectPayload(alloc, node);
    errdefer alloc.free(payload);
    const bytecode = try program.serialize(alloc, node);
    const bytes = payload.len + bytecode.len;
    alloc.free(bytecode);

    dropPayloads(alloc, &node);
    try program.distributePayload(&node, payload);
    return .{ .node = node, .payload = payload, .bytes = bytes, .expanded = 0 };
}

pub fn encode(alloc: Allocator, plan: *const Plan, in: Stream, dtype: Dtype) !Result {
    var result = encodeTemplate(alloc, plan.root, in, dtype) catch |err| switch (err) {
        error.AlphabetTooLarge, error.SymbolNotInTable => return encodeTemplate(alloc, .{ .op = .raw }, in, dtype),
        else => return err,
    };
    if (plan.root.op == .raw or result.bytes < in.data.len + 24) return result;
    result.deinit(alloc);
    return encodeTemplate(alloc, .{ .op = .raw }, in, dtype);
}

// ==================== tree navigation ====================

fn firstHolePath(alloc: Allocator, sk: PNode, path: *std.ArrayList(u8)) !bool {
    switch (sk) {
        .hole => return true,
        .filled => |f| {
            for (f.kids, 0..) |k, i| {
                try path.append(alloc, @intCast(i));
                if (try firstHolePath(alloc, k, path)) return true;
                _ = path.pop();
            }
            return false;
        },
    }
}

fn nodePtr(root: *PNode, path: []const u8) *PNode {
    var cur = root;
    for (path) |i| cur = &cur.filled.kids[i];
    return cur;
}

/// Replay the transforms along `path` to rebuild the stream feeding that hole.
/// Partials store hole metadata only, never streams, so memory stays O(depth).
fn streamAt(alloc: Allocator, root: PNode, in: Stream, path: []const u8) !Stream {
    var cur = try in.dupe(alloc);
    errdefer cur.deinit(alloc);
    var sk = root;

    for (path) |idx| {
        const f = sk.filled;
        var outs: std.ArrayList(Stream) = .empty;
        defer {
            for (outs.items) |*s| s.deinit(alloc);
            outs.deinit(alloc);
        }
        var side: ops.SideInfo = .none;
        defer side.deinit(alloc);

        try ops.forward(alloc, f.op, f.params, cur, &outs, &side);
        const taken = outs.items[idx];
        outs.items[idx].owns_data = false;
        cur.deinit(alloc);
        cur = taken;
        sk = f.kids[idx];
    }
    return cur;
}

// ==================== search ====================

pub fn synthesizePlan(alloc: Allocator, in: Stream, dtype: Dtype, pr: *const prior.Prior, opts: Options) !Plan {
    var sample: ?Stream = null;
    defer if (sample) |*stream| stream.deinit(alloc);
    if (opts.sample_elems > 0 and in.count > opts.sample_elems)
        sample = try planningSample(alloc, in, opts.sample_elems);

    var picked = try run(alloc, sample orelse in, dtype, pr, opts, null);
    dropRuntime(alloc, &picked.node);
    alloc.free(picked.payload);
    return .{ .root = picked.node, .expanded = picked.expanded };
}

pub fn synthesize(alloc: Allocator, in: Stream, dtype: Dtype, pr: *const prior.Prior, opts: Options) !Result {
    var plan = try synthesizePlan(alloc, in, dtype, pr, opts);
    defer plan.deinit(alloc);
    var result = try encode(alloc, &plan, in, dtype);
    result.expanded = plan.expanded;
    return result;
}

/// Contiguous windows spread over the block. Contiguity matters: strided
/// sampling would break neighbour relations and make xor_prev/diff_mod look
/// useless when they are in fact the right answer.
pub fn planningSample(alloc: Allocator, in: Stream, want: usize) !Stream {
    const total = @min(want, in.count);
    if (total == in.count) return in.dupe(alloc);
    var out = try Stream.init(alloc, total, in.bits_per_elem);
    errdefer out.deinit(alloc);
    if (total == 0) return out;

    const n_windows = @min(@as(usize, 4), total);
    const skipped = in.count - total;
    var written: usize = 0;
    for (0..n_windows) |window| {
        const len = total / n_windows + @intFromBool(window < total % n_windows);
        const gap = if (n_windows == 1) skipped / 2 else window * skipped / (n_windows - 1);
        const start = written + gap;
        for (0..len) |i| {
            out.setU32(written, in.getU32(start + i));
            written += 1;
        }
    }
    std.debug.assert(written == total);
    return out;
}

/// Complete candidates reached within the expansion budget, ascending by bytes.
pub fn synthesizeAll(alloc: Allocator, in: Stream, dtype: Dtype, opts: Options) ![]Result {
    var untrained: prior.Prior = .empty;
    defer untrained.deinit(alloc);

    var all: std.ArrayList(Result) = .empty;
    errdefer {
        for (all.items) |*r| r.deinit(alloc);
        all.deinit(alloc);
    }
    var best = try run(alloc, in, dtype, &untrained, opts, &all);
    var best_in_all = false;
    errdefer if (!best_in_all) best.deinit(alloc);
    try all.append(alloc, best);
    best_in_all = true;

    for (all.items) |*r| r.expanded = best.expanded;
    std.mem.sort(Result, all.items, {}, lessBytes);
    return all.toOwnedSlice(alloc);
}

fn lessBytes(_: void, a: Result, b: Result) bool {
    return a.bytes < b.bytes;
}

fn run(
    alloc: Allocator,
    in: Stream,
    dtype: Dtype,
    pr: *const prior.Prior,
    opts: Options,
    all: ?*std.ArrayList(Result),
) !Result {
    const learned = !pr.isEmpty();
    const raw_sk: PNode = .{ .filled = .{ .op = .raw, .params = 0, .kids = &.{} } };
    var incumbent = try encodeReal(alloc, raw_sk, in);
    errdefer incumbent.deinit(alloc);

    var q: Queue = .empty;
    defer {
        while (q.pop()) |popped| {
            var p = popped;
            p.deinit(alloc);
        }
        q.deinit(alloc);
    }

    var path: std.ArrayList(u8) = .empty;
    defer path.deinit(alloc);

    const root_bits = try holeBits(alloc, in, 0);
    const root_ctx = if (learned) prior.Context.fromStream(in, dtype, 0, 0, ROOT_PARENT) else prior.Context{};
    const root_score_lb = completionScoreLowerBound(pr, root_ctx, in.bits_per_elem, 0, dtype, true);
    try q.push(alloc, .{
        .root = .{ .hole = .{
            .depth = 0,
            .slot = 0,
            .parent_op = ROOT_PARENT,
            .context = if (learned) root_ctx else null,
            .score_lb = root_score_lb,
            .lb_bits = root_bits,
        } },
        .p = 0,
        .f = root_score_lb,
        .g_bytes = 0,
        .lb_sum = root_bits,
        .score_lb_sum = root_score_lb,
        .n_nodes = 0,
        .n_holes = 1,
    });

    var expanded: usize = 0;
    var realized: usize = 0;
    while (q.pop()) |popped| {
        var part = popped;

        if (part.n_holes == 0) {
            if (!opts.enumerate_all and realized >= opts.max_realizations) {
                part.deinit(alloc);
                break;
            }
            realized += 1;
            var res = encodeReal(alloc, part.root, in) catch |e| switch (e) {
                error.AlphabetTooLarge => {
                    part.deinit(alloc);
                    continue;
                },
                else => return e,
            };
            std.debug.assert(res.bytes >= part.g_bytes); // g_bytes is a lower bound
            part.deinit(alloc);
            if (res.bytes < incumbent.bytes) {
                if (all) |l| try l.append(alloc, incumbent) else incumbent.deinit(alloc);
                incumbent = res;
            } else {
                if (all) |l| try l.append(alloc, res) else res.deinit(alloc);
            }
            continue;
        }

        if (expanded >= opts.max_expansions) {
            part.deinit(alloc);
            break;
        }
        expanded += 1;
        defer part.deinit(alloc);

        path.clearRetainingCapacity();
        const found = try firstHolePath(alloc, part.root, &path);
        std.debug.assert(found);
        const hole = nodePtr(&part.root, path.items).hole;

        var hs = try streamAt(alloc, part.root, in, path.items);
        defer hs.deinit(alloc);

        var hist = try codec.buildHistogram(alloc, hs);
        defer hist.deinit(alloc);

        const feat = featOf(hist);
        const ctx = if (learned)
            hole.context orelse prior.Context.fromStream(hs, dtype, hole.slot, hole.depth, hole.parent_op)
        else
            prior.Context{};

        var prod_storage: [MAX_PRODUCTIONS]OpKind = undefined;
        const prods = legalProductions(hs.bits_per_elem, hole.depth, dtype, hole.depth == 0, &prod_storage);
        var scores: [MAX_PRODUCTIONS]u32 = undefined;
        pr.scoreSet(ctx, prods, scores[0..prods.len]);

        for (prods, scores[0..prods.len]) |prod, score| {
            if (part.n_nodes == 0 and prod == .raw) continue;
            if (hs.count == 0 and prod != .raw) continue;
            const params = chooseParams(prod, hs, dtype, feat) orelse continue;
            const a = ops.arity(prod, hs.bits_per_elem);
            std.debug.assert(a <= MAX_ARITY);
            if (part.n_nodes + part.n_holes + a > opts.max_nodes) continue;
            if (a > 0 and hole.depth + 1 > opts.max_depth) continue;

            var kid_bits: [MAX_ARITY]u64 = undefined;
            var kid_scores: [MAX_ARITY]u32 = undefined;
            var kid_contexts: [MAX_ARITY]?prior.Context = undefined;
            var add: usize = undefined;

            if (prod.isTerminal()) {
                // Closed-form: never encode. Exact for raw/bitpack/huffman,
                // a strict lower bound for rans.
                add = terminalCost(alloc, prod, params, hs, hist) catch |e| switch (e) {
                    error.AlphabetTooLarge => continue,
                    else => return e,
                } orelse continue;
            } else if (prod.isAlphabetPermutation()) {
                // Histogram is permuted, entropy identical: no forward needed.
                add = NODE_HDR;
                kid_bits[0] = try holeBits(alloc, hs, hole.depth + 1);
                kid_scores[0] = 0;
                kid_contexts[0] = null;
            } else {
                var outs: std.ArrayList(Stream) = .empty;
                defer {
                    for (outs.items) |*s| s.deinit(alloc);
                    outs.deinit(alloc);
                }
                var side: ops.SideInfo = .none;
                defer side.deinit(alloc);
                try ops.forward(alloc, prod, params, hs, &outs, &side);
                add = NODE_HDR + sideBody(side);
                for (outs.items, 0..) |s, i| {
                    kid_bits[i] = try holeBits(alloc, s, hole.depth + 1);
                    const child_ctx = if (learned)
                        prior.Context.fromStream(s, dtype, @intCast(i), hole.depth + 1, @intFromEnum(prod))
                    else
                        prior.Context{};
                    kid_contexts[i] = if (learned) child_ctx else null;
                    kid_scores[i] = completionScoreLowerBound(
                        pr,
                        child_ctx,
                        s.bits_per_elem,
                        hole.depth + 1,
                        dtype,
                        false,
                    );
                }
            }

            const g = part.g_bytes + add;
            const n_holes = part.n_holes - 1 + a;
            var lb_sum = part.lb_sum - hole.lb_bits;
            for (kid_bits[0..a]) |b| lb_sum += b;
            var score_lb_sum = part.score_lb_sum - hole.score_lb;
            for (kid_scores[0..a]) |child_score| score_lb_sum += child_score;

            if (!opts.enumerate_all and boundBytes(g, lb_sum, n_holes) >= incumbent.bytes) continue;

            var root = try part.root.clone(alloc);
            errdefer root.deinit(alloc);
            const kids = try alloc.alloc(PNode, a);
            for (kids, 0..) |*k, i| k.* = .{ .hole = .{
                .depth = hole.depth + 1,
                .slot = @intCast(i),
                .parent_op = @intFromEnum(prod),
                .context = kid_contexts[i],
                .score_lb = kid_scores[i],
                .lb_bits = kid_bits[i],
            } };
            nodePtr(&root, path.items).* = .{ .filled = .{ .op = prod, .params = params, .kids = kids } };

            const p = part.p + score;
            try q.push(alloc, .{
                .root = root,
                .p = p,
                .f = p + score_lb_sum,
                .g_bytes = g,
                .lb_sum = lb_sum,
                .score_lb_sum = score_lb_sum,
                .n_nodes = part.n_nodes + 1,
                .n_holes = n_holes,
            });
        }
    }

    incumbent.expanded = expanded;
    return incumbent;
}

test "uniform root completion score uses cheapest legal production" {
    const alloc = std.testing.allocator;
    var stream = try Stream.init(alloc, 16, 8);
    defer stream.deinit(alloc);
    for (0..stream.count) |i| stream.setU32(i, @intCast(i));
    var untrained: prior.Prior = .empty;
    defer untrained.deinit(alloc);
    const ctx = prior.Context.fromStream(stream, .i8, 0, 0, ROOT_PARENT);
    try std.testing.expectEqual(@as(u32, 4186), completionScoreLowerBound(&untrained, ctx, 8, 0, .i8, true));
}
