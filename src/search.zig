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

/// Exact serialized header size: op | params | side tag | n_children.
const NODE_HDR: usize = 7;
/// Every hole must end in a terminal header, side record, and payload length.
const MIN_HOLE_BYTES: u64 = NODE_HDR + 9 + 8;
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
    /// Bound the stream used by A*. Tensor planning realizes only the shortlist
    /// on full blocks. Zero searches the complete input.
    sample_elems: usize = ops.SEARCH_SAMPLE_ELEMS,
    /// Number of sample-ranked candidates to rerank on representative full
    /// blocks. Zero disables full-block reranking and keeps the sample winner.
    rerank_candidates: usize = ops.PLAN_CANDIDATES,
    /// Number of full blocks, spread across the tensor, used for reranking.
    /// Zero disables reranking, as does rerank_candidates == 0.
    rerank_blocks: usize = ops.PLAN_PROBE_BLOCKS,
    /// Bit mask over OpKind values. Raw remains an implicit fallback even for
    /// direct API callers that clear its bit.
    enabled_ops: u64 = ops.ALL_OPS_MASK,
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
    candidates_realized: usize = 0,
    candidates_reranked: usize = 0,
    probe_blocks_used: usize = 0,
    selected_sample_rank: usize = 0,

    pub fn deinit(self: *Plan, alloc: Allocator) void {
        self.root.deinit(alloc);
    }
};

fn splitParams(dtype: Dtype, bits: u8) u32 {
    if (dtype.floatFields()) |fields| {
        if (fields.total == bits)
            return @as(u32, fields.mant) | (@as(u32, fields.exp) << 8);
    }
    const start: u32 = bits / 2;
    return start | ((@as(u32, bits) - start) << 8);
}

pub fn fixedPlan(alloc: Allocator, dtype: Dtype) !Plan {
    const children = try alloc.alloc(Node, 2);
    children[0] = .{ .op = .rans };
    children[1] = .{ .op = .bitpack };
    return .{
        .root = .{
            .op = .split_field,
            .params = splitParams(dtype, dtype.bitWidth()),
            .children = children,
        },
        .expanded = 0,
    };
}

// ==================== partial programs ====================

const Hole = struct {
    depth: u8,
    slot: u8,
    parent_op: u8,
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

fn holeBits(alloc: Allocator, s: Stream, depth: u8, max_depth: u8) !u64 {
    if (depth < max_depth) return 0;
    return shannonBits(alloc, s);
}

fn lowerBound(lb_sum: u64, n_holes: usize) u64 {
    return lb_sum + @as(u64, n_holes) * MIN_HOLE_BYTES * 8;
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
            var t = codec.huffmanFromHist(alloc, hist, s.bits_per_elem) catch |e| switch (e) {
                error.HuffmanCodeTooLong => return null,
                else => return e,
            };
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

fn addEnabledProduction(
    out: *[MAX_PRODUCTIONS]OpKind,
    len: *usize,
    op: OpKind,
    enabled_ops: u64,
) void {
    if (enabled_ops & ops.opMask(op) != 0) addProduction(out, len, op);
}

fn legalProductions(
    hole_bpe: u8,
    depth: u8,
    max_depth: u8,
    dtype: Dtype,
    is_root: bool,
    enabled_ops: u64,
    out: *[MAX_PRODUCTIONS]OpKind,
) []const OpKind {
    var len: usize = 0;
    // Raw is a structural safety net and cannot be removed by an ablation.
    addProduction(out, &len, .raw);
    addEnabledProduction(out, &len, .bitpack, enabled_ops);
    if (hole_bpe <= ops.MAX_ENTROPY_BPE) {
        addEnabledProduction(out, &len, .huffman, enabled_ops);
        addEnabledProduction(out, &len, .rans, enabled_ops);
    }
    if (depth >= max_depth) return out[0..len];

    const elementwise = [_]OpKind{
        .xor_const, .add_const_mod, .xor_prev,    .diff_mod, .zigzag,
        .gray,      .rotate_bits,   .bit_reverse, .rle,      .deinterleave,
    };
    for (elementwise) |op| addEnabledProduction(out, &len, op, enabled_ops);

    if (hole_bpe > 1) {
        addEnabledProduction(out, &len, .split_field, enabled_ops);
        addEnabledProduction(out, &len, .topk_codebook, enabled_ops);
        addEnabledProduction(out, &len, .bit_plane, enabled_ops);
    }
    if (hole_bpe > 8) addEnabledProduction(out, &len, .byte_plane, enabled_ops);
    if (is_root and dtype.isFloat()) addEnabledProduction(out, &len, .split_float, enabled_ops);
    return out[0..len];
}

fn completionScoreLowerBound(
    pr: *const prior.Prior,
    bpe: u8,
    depth: u8,
    max_depth: u8,
    dtype: Dtype,
    is_root: bool,
    enabled_ops: u64,
) u32 {
    var storage: [MAX_PRODUCTIONS]OpKind = undefined;
    const productions = legalProductions(bpe, depth, max_depth, dtype, is_root, enabled_ops, &storage);
    return pr.scoreLowerBound(productions.len);
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
            break :blk splitParams(dtype, k);
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
        error.AlphabetTooLarge, error.HuffmanCodeTooLong, error.SymbolNotInTable => return encodeTemplate(alloc, .{ .op = .raw }, in, dtype),
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

fn candidateCost(
    alloc: Allocator,
    candidate: Result,
    full: Stream,
    blocks: []const types.Block,
    dtype: Dtype,
    probe_blocks: usize,
) !u64 {
    const plan: Plan = .{ .root = candidate.node, .expanded = candidate.expanded };
    const n = @min(probe_blocks, blocks.len);
    std.debug.assert(n > 0);
    var total: u64 = 0;
    for (0..n) |probe| {
        const index = if (n == 1) 0 else probe * (blocks.len - 1) / (n - 1);
        var encoded = try encode(alloc, &plan, blocks[index].asStream(full.data), dtype);
        defer encoded.deinit(alloc);
        total += encoded.bytes;
    }
    return total;
}

pub fn synthesizeTensorPlan(
    alloc: Allocator,
    full: Stream,
    dtype: Dtype,
    inner: usize,
    pr: *const prior.Prior,
    opts: Options,
) !Plan {
    var sample: ?Stream = null;
    defer if (sample) |*stream| stream.deinit(alloc);
    if (opts.sample_elems > 0 and full.count > opts.sample_elems)
        sample = try planningSample(alloc, full, opts.sample_elems);

    var candidate_opts = opts;
    // A sample-byte incumbent cannot safely prune a real-block objective.
    candidate_opts.enumerate_all = true;
    candidate_opts.sample_elems = 0;
    const candidates = try synthesizeCandidates(alloc, sample orelse full, dtype, pr, candidate_opts);
    errdefer {
        for (candidates) |*candidate| candidate.deinit(alloc);
        alloc.free(candidates);
    }

    var best: usize = 0;
    var candidates_reranked: usize = 0;
    var probe_blocks_used: usize = 0;
    if (opts.rerank_candidates > 0 and opts.rerank_blocks > 0) {
        const blocks = try types.planBlocks(alloc, 0, dtype, full.count, inner);
        defer alloc.free(blocks);
        candidates_reranked = @min(opts.rerank_candidates, candidates.len);
        probe_blocks_used = @min(opts.rerank_blocks, blocks.len);
        var best_cost = try candidateCost(alloc, candidates[0], full, blocks, dtype, probe_blocks_used);
        for (candidates[1..candidates_reranked], 1..) |candidate, i| {
            const cost = try candidateCost(alloc, candidate, full, blocks, dtype, probe_blocks_used);
            if (cost < best_cost) {
                best = i;
                best_cost = cost;
            }
        }
    }

    const candidates_realized = candidates.len;
    for (candidates, 0..) |*candidate, i| {
        if (i != best) candidate.deinit(alloc);
    }
    var picked = candidates[best];
    alloc.free(candidates);
    dropRuntime(alloc, &picked.node);
    alloc.free(picked.payload);
    return .{
        .root = picked.node,
        .expanded = picked.expanded,
        .candidates_realized = candidates_realized,
        .candidates_reranked = candidates_reranked,
        .probe_blocks_used = probe_blocks_used,
        .selected_sample_rank = best,
    };
}

pub fn synthesize(alloc: Allocator, in: Stream, dtype: Dtype, pr: *const prior.Prior, opts: Options) !Result {
    var plan = try synthesizePlan(alloc, in, dtype, pr, opts);
    defer plan.deinit(alloc);
    var result = try encode(alloc, &plan, in, dtype);
    result.expanded = plan.expanded;
    return result;
}

/// A centered contiguous window. Contiguity preserves the true neighbour
/// relations used by xor_prev, diff_mod, and PHOG delta features. Concatenating
/// disjoint windows would create artificial transitions at their boundaries.
pub fn planningSample(alloc: Allocator, in: Stream, want: usize) !Stream {
    const total = @min(want, in.count);
    if (total == in.count) return in.dupe(alloc);
    var out = try Stream.init(alloc, total, in.bits_per_elem);
    errdefer out.deinit(alloc);
    if (total == 0) return out;
    const start = (in.count - total) / 2;
    for (0..total) |i| out.setU32(i, in.getU32(start + i));
    return out;
}

/// Complete candidates reached within the expansion budget, ascending by bytes.
pub fn synthesizeAll(alloc: Allocator, in: Stream, dtype: Dtype, opts: Options) ![]Result {
    var untrained: prior.Prior = .empty;
    defer untrained.deinit(alloc);

    return synthesizeCandidates(alloc, in, dtype, &untrained, opts);
}

fn synthesizeCandidates(
    alloc: Allocator,
    in: Stream,
    dtype: Dtype,
    pr: *const prior.Prior,
    opts: Options,
) ![]Result {
    var all: std.ArrayList(Result) = .empty;
    errdefer {
        for (all.items) |*r| r.deinit(alloc);
        all.deinit(alloc);
    }
    var best = try run(alloc, in, dtype, pr, opts, &all);
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

    const root_bits = try holeBits(alloc, in, 0, opts.max_depth);
    const root_score_lb = completionScoreLowerBound(
        pr,
        in.bits_per_elem,
        0,
        opts.max_depth,
        dtype,
        true,
        opts.enabled_ops,
    );
    try q.push(alloc, .{
        .root = .{ .hole = .{
            .depth = 0,
            .slot = 0,
            .parent_op = ROOT_PARENT,
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
            prior.Context.fromStream(hs, dtype, hole.slot, hole.depth, hole.parent_op)
        else
            prior.Context{};

        var prod_storage: [MAX_PRODUCTIONS]OpKind = undefined;
        const prods = legalProductions(
            hs.bits_per_elem,
            hole.depth,
            opts.max_depth,
            dtype,
            hole.depth == 0,
            opts.enabled_ops,
            &prod_storage,
        );
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
                kid_bits[0] = try holeBits(alloc, hs, hole.depth + 1, opts.max_depth);
                kid_scores[0] = 0;
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
                    kid_bits[i] = try holeBits(alloc, s, hole.depth + 1, opts.max_depth);
                    kid_scores[i] = completionScoreLowerBound(
                        pr,
                        s.bits_per_elem,
                        hole.depth + 1,
                        opts.max_depth,
                        dtype,
                        false,
                        opts.enabled_ops,
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
    try std.testing.expectEqual(
        @as(u32, 4186),
        completionScoreLowerBound(
            &untrained,
            8,
            0,
            ops.K_TRANSFORM_LAYERS,
            .i8,
            true,
            ops.ALL_OPS_MASK,
        ),
    );
}

test "runtime depth limit controls transform productions" {
    var shallow_storage: [MAX_PRODUCTIONS]OpKind = undefined;
    const shallow = legalProductions(8, 2, 2, .u8, false, ops.ALL_OPS_MASK, &shallow_storage);
    for (shallow) |production| try std.testing.expect(production.isTerminal());

    var deeper_storage: [MAX_PRODUCTIONS]OpKind = undefined;
    const deeper = legalProductions(8, 2, 3, .u8, false, ops.ALL_OPS_MASK, &deeper_storage);
    var found_transform = false;
    for (deeper) |production| found_transform = found_transform or production.isTransform();
    try std.testing.expect(found_transform);
}

test "operator mask removes productions but preserves raw fallback" {
    const enabled = ops.ALL_OPS_MASK & ~ops.opMask(.huffman) & ~ops.opMask(.diff_mod);
    var storage: [MAX_PRODUCTIONS]OpKind = undefined;
    const productions = legalProductions(8, 0, 2, .u8, true, enabled, &storage);
    var found_raw = false;
    for (productions) |production| {
        found_raw = found_raw or production == .raw;
        try std.testing.expect(production != .huffman);
        try std.testing.expect(production != .diff_mod);
    }
    try std.testing.expect(found_raw);

    var raw_only_storage: [MAX_PRODUCTIONS]OpKind = undefined;
    const raw_only = legalProductions(8, 0, 2, .u8, true, 0, &raw_only_storage);
    try std.testing.expectEqualSlices(OpKind, &.{.raw}, raw_only);
}

test "hole lower bound includes a terminal frame" {
    try std.testing.expectEqual(@as(usize, 24), boundBytes(0, 0, 1));
}
