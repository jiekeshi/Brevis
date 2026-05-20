//! Probabilistic Higher-Order Grammar (PHOG) for brevis. Implements Euphony's
//! `q(A → β | c)` over the low-level grammar.
//!
//! Counts are stored as `counts[context_encode][op_kind]: u32`, smoothed with
//! Laplace `α = 1`. Probabilities are recomputed on demand from counts (lazy).
//!
//! Provides the `hFixpoint()` heuristic of Theorem 3.3 — a lower bound on the
//! cost (in `-log₂ q`) of completing any expansion of a non-terminal. With a
//! single non-terminal `S` this collapses to a scalar `h_S` that the A* search
//! adds for each remaining hole.

const std = @import("std");
const lowlevel = @import("lowlevel.zig");
const grammar = @import("grammar.zig");

pub const LAPLACE_ALPHA: f64 = 1.0;

pub const PHOG = struct {
    /// counts[ctx][op_kind] = times this production was chosen at this context
    counts: [grammar.N_CONTEXTS][grammar.N_OP_KINDS]u32,
    /// row totals — cached to avoid recomputation
    row_totals: [grammar.N_CONTEXTS]u32,
    /// h(S) = max likelihood (probability) of any complete S-derivation,
    /// computed once after training via fixpoint iteration. In log space.
    /// `neg_log_h_S` is `-log₂ h(S)`. Initially +∞ until `computeFixpoint` is called.
    neg_log_h_S: f64,

    pub fn empty() PHOG {
        var p: PHOG = .{
            .counts = undefined,
            .row_totals = undefined,
            .neg_log_h_S = std.math.inf(f64),
        };
        @memset(std.mem.asBytes(&p.counts), 0);
        @memset(std.mem.asBytes(&p.row_totals), 0);
        return p;
    }

    pub fn observe(self: *PHOG, ctx: grammar.Context, op: lowlevel.OpKind) void {
        const ci: usize = @intCast(ctx.encode());
        const oi: usize = @intCast(@intFromEnum(op));
        if (ci >= grammar.N_CONTEXTS or oi >= grammar.N_OP_KINDS) return;
        self.counts[ci][oi] += 1;
        self.row_totals[ci] += 1;
    }

    /// q(op | ctx) with Laplace smoothing. Returns a value in (0, 1].
    pub fn prob(self: PHOG, ctx: grammar.Context, op: lowlevel.OpKind) f64 {
        const ci: usize = @intCast(ctx.encode());
        const oi: usize = @intCast(@intFromEnum(op));
        if (ci >= grammar.N_CONTEXTS or oi >= grammar.N_OP_KINDS) return 1.0 / @as(f64, @floatFromInt(grammar.N_OP_KINDS));
        const c: f64 = @floatFromInt(self.counts[ci][oi]);
        const tot: f64 = @floatFromInt(self.row_totals[ci]);
        const k: f64 = @floatFromInt(grammar.N_OP_KINDS);
        return (c + LAPLACE_ALPHA) / (tot + LAPLACE_ALPHA * k);
    }

    /// -log₂ q(op | ctx). Non-negative.
    pub fn negLogProb(self: PHOG, ctx: grammar.Context, op: lowlevel.OpKind) f64 {
        return -std.math.log2(self.prob(ctx, op));
    }

    /// Compute h(S) by fixpoint (Euphony §3.3, Theorem 3.3).
    ///   h(A) = max over (A → β, c) of  q(A → β | c) × ∏ h(βᵢ)
    /// In our grammar A = S and β is determined by the chosen op_kind:
    ///   terminal op (huffman/rans/raw): h_term = q(op | c) × 1
    ///   chain op:                       h_chain = q(op | c) × h(S)
    ///   split op:                       h_split = q(op | c) × h(S) × h(S)
    /// We pick the max h(S) consistent with these. Equivalent log-space:
    ///   −log h(S) = min over (op, c) of [−log q + (children) × (−log h(S))]
    /// Initialize h(S) = 0 → −log h(S) = +∞. Repeated relaxation converges.
    pub fn computeFixpoint(self: *PHOG) void {
        var neg_log_h: f64 = std.math.inf(f64);
        // The best context to evaluate at; since we only have (parent, slot)
        // and root context is (null, 0), use that for the initial expansion of S.
        // (Properly we should consider all contexts that can reach an S, but
        // for one non-terminal this is fine.)
        const root_ctx = grammar.Context.ROOT;
        var iter: usize = 0;
        while (iter < 64) : (iter += 1) {
            var best: f64 = std.math.inf(f64);
            for (grammar.ALL_OPS) |op| {
                const cost = self.negLogProb(root_ctx, op);
                const a = grammar.arity(op);
                const child_cost: f64 = if (a == 0)
                    0.0
                else if (neg_log_h == std.math.inf(f64))
                    std.math.inf(f64)
                else
                    @as(f64, @floatFromInt(a)) * neg_log_h;
                const total = cost + child_cost;
                if (total < best) best = total;
            }
            if (best >= neg_log_h - 1e-9 and !(neg_log_h == std.math.inf(f64))) break; // converged
            neg_log_h = best;
        }
        self.neg_log_h_S = neg_log_h;
    }

    pub fn negLogH(self: PHOG, _: grammar.NonTerminal) f64 {
        return self.neg_log_h_S;
    }
};

test "PHOG: uniform prior gives equal probs" {
    var p = PHOG.empty();
    p.computeFixpoint();
    const p1 = p.prob(grammar.Context.ROOT, .huffman);
    const p2 = p.prob(grammar.Context.ROOT, .rans);
    try std.testing.expectApproxEqRel(p1, p2, 1e-9);
}

test "PHOG: observed counts increase prob" {
    var p = PHOG.empty();
    p.observe(grammar.Context.ROOT, .rans);
    p.observe(grammar.Context.ROOT, .rans);
    p.observe(grammar.Context.ROOT, .rans);
    const p_rans = p.prob(grammar.Context.ROOT, .rans);
    const p_huff = p.prob(grammar.Context.ROOT, .huffman);
    try std.testing.expect(p_rans > p_huff);
}

test "PHOG: fixpoint finite once trained" {
    var p = PHOG.empty();
    // Bias toward terminals so fixpoint converges to a finite value.
    var i: u32 = 0;
    while (i < 100) : (i += 1) p.observe(grammar.Context.ROOT, .rans);
    p.computeFixpoint();
    try std.testing.expect(p.neg_log_h_S != std.math.inf(f64));
}
