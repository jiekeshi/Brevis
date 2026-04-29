//! Branch-and-bound program synthesis over the low-level grammar.
//!
//! Strictly speaking this is B&B, not classic A* (we DFS rather than
//! priority-queue expand) — but it carries the A* spirit: an admissible
//! Shannon-entropy heuristic + best-so-far pruning. For small grammars and
//! depth ≤ 6, B&B is *more* memory-efficient than A* with partial states
//! (peak memory = depth × stream size, not branching-factor^depth × stream
//! size).
//!
//! Design
//! ======
//! `bestCost(stream, depth)` returns the optimal (cost, program) for
//! encoding `stream` with up to `depth` non-terminal layers above it. It
//! recurses by:
//!   - trying every terminal at this stream (huffman / rans / raw)
//!   - trying every non-terminal action (xor_const(c), rotate_bits(r),
//!     diff_mod, ..., split_field(start, n_bits)) — for each, compute the
//!     output stream and recurse with depth−1.
//! The incumbent best cost is tracked across the recursion; any subtree
//! whose admissible h already exceeds best is pruned.
//!
//! Action space is hard-limited per op type to keep the recursion tractable
//! (see `enumerateActions` for the list).

const std = @import("std");
const types = @import("types.zig");
const codec = @import("codec.zig");
const lowlevel = @import("lowlevel.zig");
const discovered = @import("discovered_macros.zig");

const Allocator = types.Allocator;
const Stream = types.Stream;

// =================== program tree representation ===================

pub const PNode = union(enum) {
    terminal: struct {
        kind: lowlevel.OpKind, // huffman | rans | raw
        bits: u64,
    },
    chain: struct {
        op: lowlevel.LowOp,
        next: *PNode,
    },
    split: struct {
        op: lowlevel.LowOp, // split_field
        hi: *PNode,
        lo: *PNode,
    },
    /// A discovered macro applied to the input. Stored as opaque atomic step
    /// (the runtime knows how to execute it via `discovered.DISCOVERED`).
    macro: struct {
        macro_idx: u32,
        bits: u64,
    },

    pub fn deinit(self: *PNode, alloc: Allocator) void {
        switch (self.*) {
            .terminal => {},
            .macro => {},
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

    pub fn clone(self: *const PNode, alloc: Allocator) Allocator.Error!*PNode {
        const n = try alloc.create(PNode);
        n.* = switch (self.*) {
            .terminal => |t| .{ .terminal = t },
            .macro => |m| .{ .macro = m },
            .chain => |c| .{ .chain = .{ .op = c.op, .next = try c.next.clone(alloc) } },
            .split => |s| .{ .split = .{
                .op = s.op,
                .hi = try s.hi.clone(alloc),
                .lo = try s.lo.clone(alloc),
            } },
        };
        return n;
    }

    pub fn pretty(self: *const PNode, alloc: Allocator) Allocator.Error![]u8 {
        return switch (self.*) {
            .terminal => |t| std.fmt.allocPrint(alloc, "{s}", .{lowlevelOpName(t.kind)}),
            .macro => |m| std.fmt.allocPrint(alloc, "MACRO[{s}]", .{discovered.DISCOVERED[m.macro_idx].name}),
            .chain => |c| blk: {
                const inner = try c.next.pretty(alloc);
                defer alloc.free(inner);
                break :blk std.fmt.allocPrint(alloc, "{s}({d}) -> {s}", .{ lowlevelOpName(c.op.kind), c.op.params.raw, inner });
            },
            .split => |s| blk: {
                const hi = try s.hi.pretty(alloc);
                defer alloc.free(hi);
                const lo = try s.lo.pretty(alloc);
                defer alloc.free(lo);
                break :blk std.fmt.allocPrint(alloc, "split_field(p={x}) [hi={s}, lo={s}]", .{ s.op.params.raw, hi, lo });
            },
        };
    }
};

fn lowlevelOpName(k: lowlevel.OpKind) []const u8 {
    return switch (k) {
        .xor_const => "xor_const",
        .add_const_mod => "add_const_mod",
        .rotate_bits => "rotate",
        .bit_swap_pair => "bit_swap",
        .xor_prev => "xor_prev",
        .prefix_xor => "prefix_xor",
        .diff_mod => "diff",
        .cumsum_mod => "cumsum",
        .split_field => "split_field",
        .huffman => "huffman",
        .rans => "rans",
        .raw => "raw",
    };
}

// =================== action enumeration (hard-limited for MVP) ===================

fn enumerateNonTerminals(stream_bpe: u8, buf: *std.ArrayList(lowlevel.LowOp), alloc: Allocator) !void {
    // Parameter-free ops that always make sense.
    try buf.append(alloc, .{ .kind = .xor_prev });
    try buf.append(alloc, .{ .kind = .diff_mod });
    try buf.append(alloc, .{ .kind = .gray_code });
    try buf.append(alloc, .{ .kind = .inv_gray_code });
    try buf.append(alloc, .{ .kind = .bit_reverse });
    try buf.append(alloc, .{ .kind = .negate_mod });

    // xor_with_shift with a few shift amounts (good for highly-correlated bits).
    if (stream_bpe >= 4) {
        const shifts: []const u32 = if (stream_bpe == 8) &.{ 1, 2, 4 } else &.{ 1, 2, 4, 8 };
        for (shifts) |s| try buf.append(alloc, .{ .kind = .xor_with_shift, .params = .{ .raw = s } });
    }

    // mul_const_odd_mod with a few small odd constants.
    {
        const odds: []const u32 = &.{ 3, 5, 7, 11, 17 };
        for (odds) |c| try buf.append(alloc, .{ .kind = .mul_const_odd_mod, .params = .{ .raw = c } });
    }

    // xor_const with a few common bit patterns.
    const xor_consts: []const u32 = if (stream_bpe == 8)
        &.{ 0xFF, 0xAA, 0x55, 0x80 }
    else if (stream_bpe == 16)
        &.{ 0xFFFF, 0xAAAA, 0x5555, 0x8000 }
    else
        &.{};
    for (xor_consts) |c| try buf.append(alloc, .{ .kind = .xor_const, .params = .{ .raw = c } });

    // rotate_bits: a few sensible rotations.
    if (stream_bpe >= 4) {
        const rotates: []const u32 = if (stream_bpe == 8) &.{ 1, 3, 4 } else &.{ 1, 4, 8, 12 };
        for (rotates) |r| try buf.append(alloc, .{ .kind = .rotate_bits, .params = .{ .raw = r } });
    }

    // split_field: a few "natural" split points based on bpe.
    if (stream_bpe == 16) {
        const splits: []const struct { start: u8, n: u8 } = &.{
            .{ .start = 15, .n = 1 }, // sign of fp16/bf16
            .{ .start = 10, .n = 5 }, // exp of fp16
            .{ .start = 7, .n = 8 }, // exp of bf16
            .{ .start = 8, .n = 8 }, // bf16 (sign+exp) vs (mant)
        };
        for (splits) |sp| {
            const params: u32 = sp.start | (@as(u32, sp.n) << 8) | (@as(u32, stream_bpe) << 16);
            try buf.append(alloc, .{ .kind = .split_field, .params = .{ .raw = params } });
        }
    } else if (stream_bpe == 8) {
        const splits: []const struct { start: u8, n: u8 } = &.{
            .{ .start = 7, .n = 1 },
            .{ .start = 4, .n = 4 },
            .{ .start = 0, .n = 4 },
            .{ .start = 6, .n = 2 },
            .{ .start = 0, .n = 2 },
        };
        for (splits) |sp| {
            const params: u32 = sp.start | (@as(u32, sp.n) << 8) | (@as(u32, stream_bpe) << 16);
            try buf.append(alloc, .{ .kind = .split_field, .params = .{ .raw = params } });
        }
    }
}

// =================== heuristic: Shannon entropy lower bound ===================

fn shannonBits(stream: Stream) u64 {
    if (stream.count == 0) return 0;
    if (stream.bits_per_elem > 16) return @as(u64, stream.count) * @as(u64, stream.bits_per_elem);
    var counts: [65536]u32 = undefined;
    @memset(&counts, 0);
    for (0..stream.count) |i| counts[stream.getU32(i) & 0xFFFF] += 1;
    var H: f64 = 0;
    const n_f: f64 = @floatFromInt(stream.count);
    for (counts) |c| {
        if (c == 0) continue;
        const p: f64 = @as(f64, @floatFromInt(c)) / n_f;
        H += -p * std.math.log2(p);
    }
    return @intFromFloat(@ceil(H * n_f));
}

// =================== macro execution ===================
//
// A discovered macro is a (small) program tree we found high-frequency-enough
// during training to lambda-fy. To use it as a single A* action, we simulate
// running it on the input stream and accumulate the encoding cost of all
// terminals in its tree.

fn applyMacroCost(alloc: Allocator, macro: discovered.Macro, root_idx: u32, input: Stream) !u64 {
    const node = macro.nodes[root_idx];
    if (node.is_terminal) {
        return terminalCost(alloc, node.op_kind, input);
    }
    const op: lowlevel.LowOp = .{ .kind = node.op_kind, .params = .{ .raw = node.params_raw } };
    const r = try lowlevel.forward(alloc, op, input);
    switch (r) {
        .one => |out| {
            defer alloc.free(out.data);
            if (node.child_hi < 0) return error.BadMacroLink;
            return try applyMacroCost(alloc, macro, @intCast(node.child_hi), out);
        },
        .two => |outs| {
            defer alloc.free(outs[0].data);
            defer alloc.free(outs[1].data);
            if (node.child_hi < 0 or node.child_lo < 0) return error.BadMacroLink;
            const hi_c = try applyMacroCost(alloc, macro, @intCast(node.child_hi), outs[0]);
            const lo_c = try applyMacroCost(alloc, macro, @intCast(node.child_lo), outs[1]);
            return hi_c + lo_c;
        },
    }
}

// =================== terminal cost estimation ===================

fn terminalCost(alloc: Allocator, kind: lowlevel.OpKind, stream: Stream) !u64 {
    _ = alloc;
    return switch (kind) {
        .raw => @as(u64, stream.data.len) * 8,
        .huffman => codec.huffmanCostBits(stream),
        .rans => codec.ransCostBits(stream),
        else => unreachable,
    };
}

// =================== main entry: bestCost recursive search ===================

pub const Best = struct {
    cost: u64,
    program: ?*PNode,

    pub fn deinit(self: *Best, alloc: Allocator) void {
        if (self.program) |p| {
            p.deinit(alloc);
            alloc.destroy(p);
        }
        self.program = null;
    }
};

pub const Opts = struct {
    max_depth: u8 = 4,
    /// Soft cap on outer compute — when exceeded, return whatever is best so far.
    max_nodes_explored: u64 = std.math.maxInt(u64),
};

/// Returns the optimal (cost, plan) for encoding `input`, considering up to
/// `depth_left` non-terminal layers. `incumbent` is the best cost found
/// elsewhere in the search — used for pruning.
pub fn bestForStream(
    alloc: Allocator,
    input: Stream,
    depth_left: u8,
    incumbent: u64,
    nodes_explored: *u64,
    node_budget: u64,
) Allocator.Error!Best {
    nodes_explored.* += 1;
    var local: Best = .{ .cost = std.math.maxInt(u64), .program = null };

    if (nodes_explored.* >= node_budget) return local;

    // Admissible bound.
    const h = shannonBits(input);
    if (h >= incumbent) return local;

    // Try each terminal.
    const terminals: []const lowlevel.OpKind = &.{ .huffman, .rans, .raw };
    for (terminals) |term| {
        const c = try terminalCost(alloc, term, input);
        if (c < local.cost) {
            local.cost = c;
            if (local.program) |p| {
                p.deinit(alloc);
                alloc.destroy(p);
            }
            const node = try alloc.create(PNode);
            node.* = .{ .terminal = .{ .kind = term, .bits = c } };
            local.program = node;
        }
    }

    // Try each discovered macro (as an atomic action — the macro body
    // already encodes a multi-step program). Macros only apply if they're
    // calibrated for this stream's bpe (or any-bpe = 0).
    for (discovered.DISCOVERED, 0..) |macro, m_idx| {
        if (macro.input_bpe != 0 and macro.input_bpe != input.bits_per_elem) continue;
        const c = applyMacroCost(alloc, macro, 0, input) catch continue;
        if (c < local.cost) {
            local.cost = c;
            if (local.program) |p| {
                p.deinit(alloc);
                alloc.destroy(p);
            }
            const node = try alloc.create(PNode);
            node.* = .{ .macro = .{ .macro_idx = @intCast(m_idx), .bits = c } };
            local.program = node;
        }
    }

    if (depth_left == 0) return local;

    // Try each non-terminal.
    var actions: std.ArrayList(lowlevel.LowOp) = .empty;
    defer actions.deinit(alloc);
    enumerateNonTerminals(input.bits_per_elem, &actions, alloc) catch return local;

    for (actions.items) |op| {
        const new_incumbent: u64 = @min(incumbent, local.cost);
        const r = lowlevel.forward(alloc, op, input) catch continue;
        switch (r) {
            .one => |out| {
                defer alloc.free(out.data);
                var sub = try bestForStream(alloc, out, depth_left - 1, new_incumbent, nodes_explored, node_budget);
                defer sub.deinit(alloc);
                if (sub.program == null) continue;
                if (sub.cost >= local.cost) continue;
                // Wrap in chain.
                const node = try alloc.create(PNode);
                const sub_program = sub.program.?;
                sub.program = null; // transfer ownership
                node.* = .{ .chain = .{ .op = op, .next = sub_program } };
                if (local.program) |p| {
                    p.deinit(alloc);
                    alloc.destroy(p);
                }
                local.cost = sub.cost;
                local.program = node;
            },
            .two => |two_outs| {
                const outs = two_outs;
                defer alloc.free(outs[0].data);
                defer alloc.free(outs[1].data);
                // Solve hi and lo independently.
                var hi_sub = try bestForStream(alloc, outs[0], depth_left - 1, new_incumbent, nodes_explored, node_budget);
                defer hi_sub.deinit(alloc);
                if (hi_sub.program == null) continue;
                if (hi_sub.cost >= new_incumbent) continue;
                const lo_budget: u64 = if (new_incumbent > hi_sub.cost) new_incumbent - hi_sub.cost else 0;
                var lo_sub = try bestForStream(alloc, outs[1], depth_left - 1, lo_budget, nodes_explored, node_budget);
                defer lo_sub.deinit(alloc);
                if (lo_sub.program == null) continue;
                const total = hi_sub.cost + lo_sub.cost;
                if (total >= local.cost) continue;
                const node = try alloc.create(PNode);
                const hi_p = hi_sub.program.?;
                const lo_p = lo_sub.program.?;
                hi_sub.program = null;
                lo_sub.program = null;
                node.* = .{ .split = .{ .op = op, .hi = hi_p, .lo = lo_p } };
                if (local.program) |p| {
                    p.deinit(alloc);
                    alloc.destroy(p);
                }
                local.cost = total;
                local.program = node;
            },
        }
    }
    return local;
}

/// Convenience: run the search from a fresh state with default opts.
pub fn synthesize(alloc: Allocator, input: Stream, opts: Opts) !Best {
    var nodes: u64 = 0;
    return bestForStream(alloc, input, opts.max_depth, std.math.maxInt(u64), &nodes, opts.max_nodes_explored);
}
