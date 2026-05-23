//! Euphony-style A* search over the sentential-form graph of the brevis
//! grammar (`grammar.zig`), with edge weights `−log₂ q(A → β | c)` taken
//! from PHOG (`phog.zig`). Admissible heuristic h is the PHOG fixpoint
//! `neg_log_h_S` applied to each remaining hole.
//!
//! After search returns the next complete sentential form (i.e. a candidate
//! program), it's "realized" — actually run through the encoder to measure
//! true compressed bytes. The caller can realize the top K candidates and
//! pick the one with smallest actual bytes; A* gives them in decreasing
//! PHOG likelihood.
//!
//! Two layers in this file:
//!   1. SForm + State + bestForStream — the search.
//!   2. PNode + realize + decompress — the program-tree representation used
//!      at compress/decompress time. Same as before; A* converts its final
//!      SForm into a PNode for realize/decompress.

const std = @import("std");
const types = @import("types.zig");
const codec = @import("codec.zig");
const lowlevel = @import("lowlevel.zig");
const grammar = @import("grammar.zig");
const phog_mod = @import("phog.zig");

const Allocator = types.Allocator;
const Stream = types.Stream;
const PHOG = phog_mod.PHOG;

// =================== Sentential-form representation ===================

/// One node in a sentential-form tree. Either an unexpanded hole (carries its
/// context for PHOG) or a partially/fully-expanded op (carries the chosen
/// op + concrete params + indices of its children in the same array).
pub const SFormNode = union(enum) {
    hole: struct {
        ctx_parent_op: i16, // -1 if root, otherwise OpKind value
        ctx_slot: u8,
        bpe: u8, // bits-per-elem of the stream that will fill this hole
    },
    op: struct {
        kind: lowlevel.OpKind,
        params_raw: u32,
        child_hi: i32, // node index in array; -1 if unused
        child_lo: i32,
    },
};

pub const State = struct {
    /// Flat array of nodes; root is `nodes[0]`.
    nodes: []SFormNode,
    n_holes: u32,
    /// Sum of −log₂ q(production | ctx) along the chosen edges.
    g: f64,
    /// `n_holes × neg_log_h_S` — sum of the admissible fixpoint bound for
    /// each remaining hole.
    h: f64,

    pub fn deinit(self: *State, alloc: Allocator) void {
        alloc.free(self.nodes);
    }

    pub fn priority(self: State) f64 {
        return self.g + self.h;
    }

    pub fn isComplete(self: State) bool {
        return self.n_holes == 0;
    }
};

fn cloneState(alloc: Allocator, s: State) !State {
    const buf = try alloc.alloc(SFormNode, s.nodes.len);
    @memcpy(buf, s.nodes);
    return .{ .nodes = buf, .n_holes = s.n_holes, .g = s.g, .h = s.h };
}

/// Find the leftmost hole in the tree (DFS from root).
fn leftmostHole(nodes: []const SFormNode) ?u32 {
    return leftmostHoleAt(nodes, 0);
}

fn leftmostHoleAt(nodes: []const SFormNode, idx: u32) ?u32 {
    if (idx >= nodes.len) return null;
    return switch (nodes[idx]) {
        .hole => idx,
        .op => |o| blk: {
            if (o.child_hi >= 0) {
                if (leftmostHoleAt(nodes, @intCast(o.child_hi))) |h| break :blk h;
            }
            if (o.child_lo >= 0) {
                if (leftmostHoleAt(nodes, @intCast(o.child_lo))) |h| break :blk h;
            }
            break :blk null;
        },
    };
}

// =================== A* search core ===================

pub const Best = struct {
    /// Realized program tree (PNode), or null if no candidate was found.
    program: ?*PNode = null,
    /// Actual compressed bits after realization (not the search g).
    cost_bits: u64 = std.math.maxInt(u64),

    pub fn deinit(self: *Best, alloc: Allocator) void {
        if (self.program) |p| {
            p.deinit(alloc);
            alloc.destroy(p);
        }
        self.program = null;
    }
};

pub const Opts = struct {
    /// Stop after this many state expansions, even if no goal found.
    max_pops: u32 = 200_000,
    /// Realize at most this many complete candidates; keep the one with
    /// smallest actual encoded bytes (which need not be the highest-likelihood).
    realize_top_k: u32 = 8,
};

const StateCmp = struct {
    fn order(_: void, a: State, b: State) std.math.Order {
        return std.math.order(a.priority(), b.priority());
    }
};

pub fn synthesize(alloc: Allocator, input: Stream, phog: PHOG, opts: Opts) !Best {
    // Initial state: one hole with root context, at the input stream's width.
    var nodes0 = try alloc.alloc(SFormNode, 1);
    nodes0[0] = .{ .hole = .{ .ctx_parent_op = -1, .ctx_slot = 0, .bpe = input.bits_per_elem } };
    const init_state: State = .{
        .nodes = nodes0,
        .n_holes = 1,
        .g = 0.0,
        .h = phog.neg_log_h_S,
    };

    var heap: std.PriorityQueue(State, void, StateCmp.order) = .empty;
    defer {
        while (heap.pop()) |s_const| {
            var s = s_const;
            s.deinit(alloc);
        }
        heap.deinit(alloc);
    }
    try heap.push(alloc, init_state);

    // Equivalence-class pruning (Euphony §3.4.1, simplified): dedup states by
    // a hash of the sentential-form structure. Two states with identical sform
    // ARE equivalent (same partial program → same future). The hash table
    // tracks the lowest g+h seen for each structure; pushes with higher cost
    // are skipped.
    var seen: std.AutoHashMap(u64, f64) = .init(alloc);
    defer seen.deinit();

    var best: Best = .{};
    errdefer best.deinit(alloc);
    var realized: u32 = 0;
    var pops: u32 = 0;
    var pruned_dup: u32 = 0;
    _ = &pruned_dup;

    while (heap.pop()) |s_const| {
        var s = s_const;
        pops += 1;
        if (pops > opts.max_pops) {
            s.deinit(alloc);
            break;
        }

        if (s.isComplete()) {
            // Convert to PNode and realize.
            const prog = sformToPNode(alloc, s.nodes, 0) catch {
                s.deinit(alloc);
                continue;
            };
            const actual = realizeAndMeasure(alloc, prog, input) catch {
                prog.deinit(alloc);
                alloc.destroy(prog);
                s.deinit(alloc);
                continue;
            };
            if (actual < best.cost_bits) {
                if (best.program) |p| {
                    p.deinit(alloc);
                    alloc.destroy(p);
                }
                best.program = prog;
                best.cost_bits = actual;
            } else {
                prog.deinit(alloc);
                alloc.destroy(prog);
            }
            realized += 1;
            s.deinit(alloc);
            if (realized >= opts.realize_top_k) break;
            continue;
        }

        // Expand leftmost hole.
        const h_idx = leftmostHole(s.nodes).?;
        const hole = s.nodes[h_idx].hole;
        const ctx: grammar.Context = .{
            .parent_op = if (hole.ctx_parent_op < 0) null else @as(lowlevel.OpKind, @enumFromInt(@as(u8, @intCast(hole.ctx_parent_op)))),
            .slot = hole.ctx_slot,
        };

        const hole_bpe = hole.bpe;
        for (grammar.ALL_OPS) |op| {
            // Enumerate parameter values valid for this hole's bit width.
            const params = grammar.paramChoices(op, hole_bpe);
            if (params.len == 0) continue;
            const edge_cost = phog.negLogProb(ctx, op);

            for (params) |pv| {
                // Build new state by cloning, then replace the hole with the op.
                const a = grammar.arity(op);
                const new_n_holes: u32 = @intCast(@as(i32, @intCast(s.n_holes)) - 1 + @as(i32, a));

                // For split_field, force the stored `k` to this hole's bpe so
                // that fwd (which uses the live stream width) and inv (which
                // uses the stored `k`) agree → bit-exact at any width. Child
                // widths follow from the split: hi = n_bits, lo = bpe − n_bits.
                const n_bits: u8 = @intCast((pv >> 8) & 0xFF);
                const params_raw: u32 = if (op == .split_field)
                    (pv & 0xFFFF) | (@as(u32, hole_bpe) << 16)
                else
                    pv;
                const hi_bpe: u8 = if (op == .split_field) n_bits else hole_bpe;
                const lo_bpe: u8 = if (op == .split_field) hole_bpe - n_bits else 0;

                // Allocate child node slots if needed.
                const new_node_count = s.nodes.len + @as(usize, a);
                var new_nodes = try alloc.alloc(SFormNode, new_node_count);
                @memcpy(new_nodes[0..s.nodes.len], s.nodes);

                var child_hi: i32 = -1;
                var child_lo: i32 = -1;
                if (a >= 1) {
                    child_hi = @intCast(s.nodes.len);
                    new_nodes[s.nodes.len] = .{ .hole = .{
                        .ctx_parent_op = @intCast(@intFromEnum(op)),
                        .ctx_slot = 0,
                        .bpe = hi_bpe,
                    } };
                }
                if (a == 2) {
                    child_lo = @intCast(s.nodes.len + 1);
                    new_nodes[s.nodes.len + 1] = .{ .hole = .{
                        .ctx_parent_op = @intCast(@intFromEnum(op)),
                        .ctx_slot = 1,
                        .bpe = lo_bpe,
                    } };
                }

                new_nodes[h_idx] = .{ .op = .{
                    .kind = op,
                    .params_raw = params_raw,
                    .child_hi = child_hi,
                    .child_lo = child_lo,
                } };

                const new_state: State = .{
                    .nodes = new_nodes,
                    .n_holes = new_n_holes,
                    .g = s.g + edge_cost,
                    .h = @as(f64, @floatFromInt(new_n_holes)) * phog.neg_log_h_S,
                };

                // Prune if already worse than best.
                if (new_state.priority() >= bitsToCost(best.cost_bits)) {
                    alloc.free(new_nodes);
                    continue;
                }
                // Equivalence pruning: hash the sform structure; skip if a
                // cheaper-priority state with the same structure was seen.
                const fp = hashSForm(new_nodes);
                if (seen.get(fp)) |prev_p| {
                    if (prev_p <= new_state.priority()) {
                        alloc.free(new_nodes);
                        pruned_dup += 1;
                        continue;
                    }
                }
                try seen.put(fp, new_state.priority());
                try heap.push(alloc, new_state);
            }
        }
        s.deinit(alloc);
    }

    return best;
}

fn hashSForm(nodes: []const SFormNode) u64 {
    var h: std.hash.XxHash64 = .init(0xC0FFEE);
    var buf: [16]u8 = undefined;
    for (nodes) |n| switch (n) {
        .hole => |hh| {
            buf[0] = 0;
            std.mem.writeInt(i16, buf[1..3], hh.ctx_parent_op, .little);
            buf[3] = hh.ctx_slot;
            buf[4] = hh.bpe;
            h.update(buf[0..5]);
        },
        .op => |o| {
            buf[0] = 1;
            buf[1] = @intFromEnum(o.kind);
            std.mem.writeInt(u32, buf[2..6], o.params_raw, .little);
            std.mem.writeInt(i32, buf[6..10], o.child_hi, .little);
            std.mem.writeInt(i32, buf[10..14], o.child_lo, .little);
            h.update(buf[0..14]);
        },
    };
    return h.final();
}

fn bitsToCost(bits: u64) f64 {
    if (bits == std.math.maxInt(u64)) return std.math.inf(f64);
    // bits is actual encoded length; priority is -log q which is incomparable
    // dimensionally. We use it only as an upper bound for pruning *unlikely
    // candidates that would also be worse*. In practice priority/bits both
    // increase with worse programs, so this is a useful (if imperfect) prune.
    return @as(f64, @floatFromInt(bits));
}

// =================== SForm → PNode conversion + realize + decompress ===================
//
// PNode is the runtime program-tree representation. It carries side_info
// (Huffman/rANS tables) and encoded payload bytes on its terminal nodes
// after `realize()`. `decompress()` is its inverse.

pub const TerminalSide = union(enum) {
    huffman: struct { table: codec.HuffmanTable, count: usize, bits_per_elem: u8 },
    rans: struct { table: codec.RansTable, count: usize, bits_per_elem: u8 },
    raw: struct { count: usize, bits_per_elem: u8 },

    pub fn deinit(self: *TerminalSide, alloc: Allocator) void {
        switch (self.*) {
            .huffman => |*h| h.table.deinit(alloc),
            .rans => |*r| r.table.deinit(alloc),
            .raw => {},
        }
    }
};

pub const PNode = union(enum) {
    terminal: struct {
        kind: lowlevel.OpKind,
        bits: u64,
        side_info: TerminalSide = .{ .raw = .{ .count = 0, .bits_per_elem = 0 } },
        payload: []u8 = &.{},
        payload_owned: bool = false,
    },
    chain: struct {
        op: lowlevel.LowOp,
        next: *PNode,
    },
    split: struct {
        op: lowlevel.LowOp,
        hi: *PNode,
        lo: *PNode,
    },

    pub fn deinit(self: *PNode, alloc: Allocator) void {
        switch (self.*) {
            .terminal => |*t| {
                t.side_info.deinit(alloc);
                if (t.payload_owned and t.payload.len > 0) alloc.free(t.payload);
            },
            .chain => |c| {
                c.next.deinit(alloc);
                alloc.destroy(c.next);
            },
            .split => |s| {
                s.hi.deinit(alloc);
                alloc.destroy(s.hi);
                s.lo.deinit(alloc);
                alloc.destroy(s.lo);
            },
        }
    }
};

fn sformToPNode(alloc: Allocator, nodes: []const SFormNode, idx: u32) Allocator.Error!*PNode {
    const n = try alloc.create(PNode);
    errdefer alloc.destroy(n);
    switch (nodes[idx]) {
        .hole => return error.OutOfMemory, // shouldn't happen on a complete sform
        .op => |o| {
            if (grammar.isTerminal(o.kind)) {
                n.* = .{ .terminal = .{ .kind = o.kind, .bits = 0 } };
                return n;
            }
            const op: lowlevel.LowOp = .{ .kind = o.kind, .params = .{ .raw = o.params_raw } };
            if (grammar.arity(o.kind) == 2) {
                const hi = try sformToPNode(alloc, nodes, @intCast(o.child_hi));
                const lo = try sformToPNode(alloc, nodes, @intCast(o.child_lo));
                n.* = .{ .split = .{ .op = op, .hi = hi, .lo = lo } };
            } else {
                const next = try sformToPNode(alloc, nodes, @intCast(o.child_hi));
                n.* = .{ .chain = .{ .op = op, .next = next } };
            }
            return n;
        },
    }
}

/// Realize the program tree on `input`: run forward at non-terminals,
/// build encoder tables + encode bytes at terminals. Fills side_info+payload.
pub fn realize(alloc: Allocator, node: *PNode, input: Stream) !void {
    switch (node.*) {
        .terminal => |*t| switch (t.kind) {
            .huffman => {
                const table = try codec.huffmanBuild(alloc, input);
                const payload = codec.huffmanEncode(alloc, input, table) catch |e| {
                    var tt = table;
                    tt.deinit(alloc);
                    return e;
                };
                t.side_info = .{ .huffman = .{ .table = table, .count = input.count, .bits_per_elem = input.bits_per_elem } };
                t.payload = payload;
                t.payload_owned = true;
                t.bits = @as(u64, payload.len) * 8;
            },
            .rans => {
                const table = try codec.ransBuild(alloc, input);
                const payload = codec.ransEncode(alloc, input, table) catch |e| {
                    var tt = table;
                    tt.deinit(alloc);
                    return e;
                };
                t.side_info = .{ .rans = .{ .table = table, .count = input.count, .bits_per_elem = input.bits_per_elem } };
                t.payload = payload;
                t.payload_owned = true;
                t.bits = @as(u64, payload.len) * 8;
            },
            .raw => {
                const payload = try alloc.alloc(u8, input.data.len);
                @memcpy(payload, input.data);
                t.side_info = .{ .raw = .{ .count = input.count, .bits_per_elem = input.bits_per_elem } };
                t.payload = payload;
                t.payload_owned = true;
                t.bits = @as(u64, payload.len) * 8;
            },
            else => return error.NotATerminal,
        },
        .chain => |c| {
            const r = try lowlevel.forward(alloc, c.op, input);
            switch (r) {
                .one => |out| {
                    defer alloc.free(out.data);
                    try realize(alloc, c.next, out);
                },
                .two => return error.ArityMismatch,
            }
        },
        .split => |s| {
            const r = try lowlevel.forward(alloc, s.op, input);
            switch (r) {
                .one => return error.ArityMismatch,
                .two => |outs| {
                    defer alloc.free(outs[0].data);
                    defer alloc.free(outs[1].data);
                    try realize(alloc, s.hi, outs[0]);
                    try realize(alloc, s.lo, outs[1]);
                },
            }
        },
    }
}

fn realizeAndMeasure(alloc: Allocator, node: *PNode, input: Stream) !u64 {
    try realize(alloc, node, input);
    return totalBits(node);
}

fn totalBits(node: *const PNode) u64 {
    return switch (node.*) {
        .terminal => |t| t.bits,
        .chain => |c| totalBits(c.next),
        .split => |s| totalBits(s.hi) + totalBits(s.lo),
    };
}

/// Reverse of realize: walk a fully-realized program tree backwards,
/// reconstructing the original stream from the payloads + side_info.
pub fn decompress(alloc: Allocator, node: *const PNode) !Stream {
    return switch (node.*) {
        .terminal => |t| switch (t.side_info) {
            .huffman => |h| codec.huffmanDecode(alloc, t.payload, h.table, h.count, h.bits_per_elem),
            .rans => |r| codec.ransDecode(alloc, t.payload, r.table, r.count, r.bits_per_elem),
            .raw => |i| blk: {
                const buf = try alloc.alloc(u8, t.payload.len);
                @memcpy(buf, t.payload);
                break :blk types.Stream{ .data = buf, .count = i.count, .bits_per_elem = i.bits_per_elem };
            },
        },
        .chain => |c| blk: {
            const inner = try decompress(alloc, c.next);
            defer alloc.free(inner.data);
            break :blk lowlevel.inverseOne(alloc, c.op, inner);
        },
        .split => |s| blk: {
            const hi = try decompress(alloc, s.hi);
            defer alloc.free(hi.data);
            const lo = try decompress(alloc, s.lo);
            defer alloc.free(lo.data);
            break :blk lowlevel.inverseTwo(alloc, s.op, hi, lo);
        },
    };
}

/// Walk a realized PNode tree, observing each (parent → child) production
/// edge in PHOG counts. Used during training.
pub fn observeProgram(p: *const PNode, parent_op: ?lowlevel.OpKind, slot: u8, phog: *PHOG) void {
    const ctx: grammar.Context = .{ .parent_op = parent_op, .slot = slot };
    switch (p.*) {
        .terminal => |t| phog.observe(ctx, t.kind),
        .chain => |c| {
            phog.observe(ctx, c.op.kind);
            observeProgram(c.next, c.op.kind, 0, phog);
        },
        .split => |s| {
            phog.observe(ctx, s.op.kind);
            observeProgram(s.hi, s.op.kind, 0, phog);
            observeProgram(s.lo, s.op.kind, 1, phog);
        },
    }
}
