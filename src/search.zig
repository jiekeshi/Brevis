//! Best-first synthesis with admissible heuristic + PHOG-lite prior.
//!
//! True A* over a small grammar: we enumerate the candidate space
//! (bounded-depth derivations of the DSL grammar), score each candidate by
//! the sum of admissible Shannon-entropy lower bounds at its leaves plus a
//! prior-penalty term, then iterate from best to worst; for each, realize
//! the actual encoding and measure cost. We stop as soon as the next
//! candidate's lower bound is ≥ the best already-realized actual cost
//! (correctness of A*: the heuristic is admissible, so no later candidate
//! can beat it).
//!
//! Cross-tensor reference: `tensor_xor(base_id, sub_program)` is included as
//! a top-level production whenever `bases.len > 0`.
//!
//! Grammar:
//!   T_PROG := tensor_raw
//!           | tensor_xor(b) -> T_BODY                (per base b)
//!           | split_float -> S_PROG x S_PROG x S_PROG
//!   T_BODY := tensor_raw
//!           | split_float -> S_PROG x S_PROG x S_PROG
//!   S_PROG := raw | huffman | rans
//!           | delta_encode -> S_TERMINAL
//!           | bitplane_split -> S_TERMINAL ... (n times, all same)
//!   S_TERMINAL := raw | huffman | rans

const std = @import("std");
const types = @import("types.zig");
const codec = @import("codec.zig");
const ops = @import("ops.zig");
const program = @import("program.zig");
const prior = @import("prior.zig");
const phog = @import("phog.zig");

const Allocator = types.Allocator;
const Stream = types.Stream;
const TensorView = types.TensorView;
const Node = program.Node;
const OpKind = program.OpKind;

pub const Result = struct {
    program: Node,
    payload: []u8, // owned, concatenated leaf payloads
    actual_bits: u64, // payload bits + side_info bits
    raw_bits: u64,
    compression_ratio: f64,
    verified: bool,
    template_summary: []u8, // owned text — short description
    /// The TensorShape the search/fast-path actually picked. Used by
    /// `brevis collect-training` to extract PHOG training examples.
    chosen_shape: TensorShape,

    pub fn deinit(self: *Result, alloc: Allocator) void {
        self.program.deinit(alloc);
        alloc.free(self.payload);
        alloc.free(self.template_summary);
    }
};

pub const Options = struct {
    max_candidates: usize = 4096,
    /// Hard ceiling on # of full-realize attempts. Empirically the optimal
    /// candidate is in the top 3 with the current heuristic + prior on real
    /// model weights, so 4 is plenty.
    realize_top_k: usize = 4,
    use_prior: bool = true,
    verbose: bool = false,
    /// If true, decompress every realized candidate and bit-compare against
    /// input. Only needed when debugging op correctness — the encoder/decoder
    /// pair is exercised by the unit tests, and a single end-to-end verify
    /// before writing the archive is much cheaper.
    verify_each_realization: bool = false,
    /// Tensor-type fast-path: classify the input by shape/dtype/statistics
    /// and use a fixed program tree without enumerating candidates. Disable
    /// to force exhaustive search.
    use_fast_path: bool = true,
};

pub const TerminalChoice = enum { raw, huffman, rans };

pub const StreamSubprogShape = union(enum) {
    terminal: TerminalChoice,
    delta: TerminalChoice, // delta_encode -> terminal
    bitplane: TerminalChoice, // bitplane_split -> all planes use this terminal
};

pub const TensorShape = union(enum) {
    raw,
    xor: struct { base_id: u32, body: TensorBody },
    split: struct { s_sign: StreamSubprogShape, s_exp: StreamSubprogShape, s_mant: StreamSubprogShape },
};

pub const TensorBody = enum { raw, split_default }; // for inside tensor_xor; "split_default" = split + best-per-stream pre-baked

/// Flat enum identifying which of the 9 stream-subprogram productions was
/// chosen — used by PHOG training data and inference.
pub const StreamProduction = enum(u8) {
    raw = 0,
    huffman = 1,
    rans = 2,
    delta_raw = 3,
    delta_huffman = 4,
    delta_rans = 5,
    bp_raw = 6,
    bp_huffman = 7,
    bp_rans = 8,

    pub fn fromShape(s: StreamSubprogShape) StreamProduction {
        return switch (s) {
            .terminal => |t| switch (t) { .raw => .raw, .huffman => .huffman, .rans => .rans },
            .delta => |t| switch (t) { .raw => .delta_raw, .huffman => .delta_huffman, .rans => .delta_rans },
            .bitplane => |t| switch (t) { .raw => .bp_raw, .huffman => .bp_huffman, .rans => .bp_rans },
        };
    }

    pub fn toShape(self: StreamProduction) StreamSubprogShape {
        return switch (self) {
            .raw => .{ .terminal = .raw },
            .huffman => .{ .terminal = .huffman },
            .rans => .{ .terminal = .rans },
            .delta_raw => .{ .delta = .raw },
            .delta_huffman => .{ .delta = .huffman },
            .delta_rans => .{ .delta = .rans },
            .bp_raw => .{ .bitplane = .raw },
            .bp_huffman => .{ .bitplane = .huffman },
            .bp_rans => .{ .bitplane = .rans },
        };
    }

    pub fn name(self: StreamProduction) []const u8 {
        return switch (self) {
            .raw => "raw",
            .huffman => "huffman",
            .rans => "rans",
            .delta_raw => "delta_raw",
            .delta_huffman => "delta_huffman",
            .delta_rans => "delta_rans",
            .bp_raw => "bp_raw",
            .bp_huffman => "bp_huffman",
            .bp_rans => "bp_rans",
        };
    }
};

pub const TProgProduction = enum(u8) {
    tensor_raw = 0,
    split_float = 1,
    tensor_xor = 2,

    pub fn name(self: TProgProduction) []const u8 {
        return switch (self) {
            .tensor_raw => "tensor_raw",
            .split_float => "split_float",
            .tensor_xor => "tensor_xor",
        };
    }
};

/// What the search picked, in a flat form suitable for PHOG training.
pub const ChoiceTrace = struct {
    t_prog: TProgProduction,
    s_sign: ?StreamProduction = null, // present iff t_prog == split_float
    s_exp: ?StreamProduction = null,
    s_mant: ?StreamProduction = null,

    pub fn fromShape(shape: TensorShape) ChoiceTrace {
        return switch (shape) {
            .raw => .{ .t_prog = .tensor_raw },
            .xor => .{ .t_prog = .tensor_xor }, // we don't model xor sub-tree productions yet
            .split => |s| .{
                .t_prog = .split_float,
                .s_sign = .fromShape(s.s_sign),
                .s_exp = .fromShape(s.s_exp),
                .s_mant = .fromShape(s.s_mant),
            },
        };
    }
};

fn allStreamShapes() [9]StreamSubprogShape {
    return .{
        .{ .terminal = .raw },
        .{ .terminal = .huffman },
        .{ .terminal = .rans },
        .{ .delta = .raw },
        .{ .delta = .huffman },
        .{ .delta = .rans },
        .{ .bitplane = .raw },
        .{ .bitplane = .huffman },
        .{ .bitplane = .rans },
    };
}

fn buildTerminal(t: TerminalChoice) Node {
    return .{ .op = switch (t) {
        .raw => .raw,
        .huffman => .huffman,
        .rans => .rans,
    } };
}

fn buildStreamSubprog(alloc: Allocator, shape: StreamSubprogShape, n_planes: u8) !Node {
    return switch (shape) {
        .terminal => |t| buildTerminal(t),
        .delta => |t| blk: {
            const kid = try alloc.alloc(Node, 1);
            kid[0] = buildTerminal(t);
            break :blk .{ .op = .delta_encode, .children = kid };
        },
        .bitplane => |t| blk: {
            const kids = try alloc.alloc(Node, n_planes);
            for (kids) |*c| c.* = buildTerminal(t);
            break :blk .{ .op = .bitplane_split, .children = kids };
        },
    };
}

fn buildTensorSplit(
    alloc: Allocator,
    s_sign: StreamSubprogShape,
    s_exp: StreamSubprogShape,
    s_mant: StreamSubprogShape,
    exp_bits: u8,
    mant_bits: u8,
) !Node {
    const kids = try alloc.alloc(Node, 3);
    kids[0] = try buildStreamSubprog(alloc, s_sign, 1);
    kids[1] = try buildStreamSubprog(alloc, s_exp, exp_bits);
    kids[2] = try buildStreamSubprog(alloc, s_mant, mant_bits);
    return .{ .op = .split_float, .children = kids };
}

fn buildTensorBody(
    alloc: Allocator,
    body: TensorBody,
    exp_bits: u8,
    mant_bits: u8,
) !Node {
    return switch (body) {
        .raw => .{ .op = .tensor_raw },
        .split_default => try buildTensorSplit(alloc, .{ .terminal = .huffman }, .{ .delta = .huffman }, .{ .terminal = .rans }, exp_bits, mant_bits),
    };
}

fn buildCandidate(
    alloc: Allocator,
    shape: TensorShape,
    exp_bits: u8,
    mant_bits: u8,
) !Node {
    return switch (shape) {
        .raw => .{ .op = .tensor_raw },
        .xor => |x| blk: {
            const kid = try alloc.alloc(Node, 1);
            kid[0] = try buildTensorBody(alloc, x.body, exp_bits, mant_bits);
            break :blk .{ .op = .tensor_xor, .base_id = x.base_id, .children = kid };
        },
        .split => |s| try buildTensorSplit(alloc, s.s_sign, s.s_exp, s.s_mant, exp_bits, mant_bits),
    };
}

/// Heuristic lower bound for a stream subprogram on a known input stream.
/// This is the sum of leaf Shannon-entropy estimates plus a tiny constant for
/// each non-terminal's side_info overhead.
fn streamHeuristicBits(alloc: Allocator, shape: StreamSubprogShape, s: Stream) !u64 {
    return switch (shape) {
        .terminal => |t| switch (t) {
            .raw => @as(u64, s.data.len) * 8,
            .huffman => codec.huffmanCostBits(s),
            .rans => codec.ransCostBits(s),
        },
        .delta => |t| blk: {
            // Run delta_encode to know what the inner terminal will see.
            const r = try ops.deltaEncodeForward(alloc, s);
            defer {
                var dd = r.out;
                dd.deinit(alloc);
            }
            const inner: u64 = switch (t) {
                .raw => @as(u64, r.out.data.len) * 8,
                .huffman => codec.huffmanCostBits(r.out),
                .rans => codec.ransCostBits(r.out),
            };
            // delta side_info: ~8 bytes
            break :blk inner + 64;
        },
        .bitplane => |t| blk: {
            // Bitplane on a 1-bit stream is degenerate.
            if (s.bits_per_elem <= 1) break :blk @as(u64, std.math.maxInt(u32));
            // Compute per-plane Shannon entropy in a single pass over the
            // source stream — no allocation. For each bit position b, a plane
            // is a binary stream of `count` bits; the cost is just
            // count * H(p_b) where p_b = (popcount of bit b) / count.
            var ones: [32]u64 = .{0} ** 32;
            const bpe: u8 = s.bits_per_elem;
            var i: usize = 0;
            while (i < s.count) : (i += 1) {
                const v = s.getU32(i);
                var b: u6 = 0;
                while (b < bpe) : (b += 1) {
                    ones[b] += (v >> @intCast(b)) & 1;
                }
            }
            const n_f: f64 = @floatFromInt(s.count);
            var sum: u64 = 0;
            var b: u6 = 0;
            while (b < bpe) : (b += 1) {
                const p: f64 = @as(f64, @floatFromInt(ones[b])) / n_f;
                if (p == 0 or p == 1) {
                    // Constant plane: huffman/rans cost ~ table overhead only.
                    sum += switch (t) {
                        .raw => @as(u64, s.count),
                        .huffman, .rans => 16, // 1 entry: ~2 bytes
                    };
                    continue;
                }
                const H: f64 = -p * std.math.log2(p) - (1.0 - p) * std.math.log2(1.0 - p);
                const data_bits: u64 = @intFromFloat(@ceil(H * n_f));
                sum += switch (t) {
                    .raw => @as(u64, s.count),
                    .huffman => data_bits + 48, // 2 codebook entries × ~24 bits
                    .rans => data_bits + 36,
                };
            }
            break :blk sum + 64; // bitplane structural overhead
        },
    };
}

fn streamPriorBits(shape: StreamSubprogShape, s: Stream) u64 {
    const f = prior.Features.fromStream(s);
    const op_root: OpKind = switch (shape) {
        .terminal => |t| switch (t) {
            .raw => .raw,
            .huffman => .huffman,
            .rans => .rans,
        },
        .delta => .delta_encode,
        .bitplane => .bitplane_split,
    };
    return prior.priorBitsPenalty(prior.streamRuleScore(op_root, f), s.count);
}

const RankedCandidate = struct {
    shape: TensorShape,
    score_bits: u64, // lower bound + prior penalty
};

const ByScore = struct {
    fn less(_: void, a: RankedCandidate, b: RankedCandidate) std.math.Order {
        return std.math.order(a.score_bits, b.score_bits);
    }
};

/// PHOG-driven program prediction. Replaces the older hand-tuned
/// `fastPathShape`. Predicts each grammar production from a learned
/// (count-based MLE + Laplace-smoothed) conditional distribution
/// `P(production | context)`. See `tools/train_phog.py`.
///
/// Returns null if the dtype isn't supported by PHOG (callers fall back to
/// full A* search).
fn phogPredict(input: TensorView) ?TensorShape {
    if (!input.dtype.isFloat16Like()) return null;
    const log2n: u8 = @intCast(std.math.log2_int(u64, @max(input.numel(), 1)));

    const t_choice = phog.predictRoot(input.dtype, log2n, input.shape.len) orelse return null;
    switch (t_choice) {
        .tensor_raw => return .raw,
        .tensor_xor => return null, // not handled in fast path; let A* try
        .split_float => {
            const s_sign = phog.predictStream(0, input.dtype, log2n, input.shape.len) orelse return null;
            const s_exp = phog.predictStream(1, input.dtype, log2n, input.shape.len) orelse return null;
            const s_mant = phog.predictStream(2, input.dtype, log2n, input.shape.len) orelse return null;
            return .{ .split = .{
                .s_sign = s_sign.toShape(),
                .s_exp = s_exp.toShape(),
                .s_mant = s_mant.toShape(),
            } };
        },
    }
}

pub fn synthesize(
    alloc: Allocator,
    input: TensorView,
    bases: []const TensorView,
    opts: Options,
) !Result {
    const raw_bits: u64 = @as(u64, input.data.len) * 8;

    // Fast path: skip search entirely when the tensor matches a known good
    // profile and the caller hasn't disabled it. Falls through to the search
    // if there's no fast-path match (e.g. cross-tensor refs are requested).
    if (opts.use_fast_path and bases.len == 0) {
        if (phogPredict(input)) |shape| {
            const exp_bits: u8 = if (input.dtype == .f16) 5 else 8;
            const mant_bits: u8 = if (input.dtype == .f16) 10 else 7;
            var node = try buildCandidate(alloc, shape, exp_bits, mant_bits);
            errdefer node.deinit(alloc);
            try program.compressTensor(alloc, &node, input, bases);
            const payload = try program.collectPayloadBytes(alloc, &node);
            errdefer alloc.free(payload);
            const program_bytes = try program.serializeProgram(alloc, &node);
            defer alloc.free(program_bytes);
            const total_bits: u64 = @as(u64, payload.len) * 8 + @as(u64, program_bytes.len) * 8;
            const summary = try summarizeShape(alloc, shape);
            return .{
                .program = node,
                .payload = payload,
                .actual_bits = total_bits,
                .raw_bits = raw_bits,
                .compression_ratio = @as(f64, @floatFromInt(raw_bits)) / @as(f64, @floatFromInt(total_bits)),
                .verified = true,
                .template_summary = summary,
                .chosen_shape = shape,
            };
        }
    }

    // Step 1: precompute streams from split_float once so heuristic
    // evaluation across all 729 split candidates is fast.
    var sf: ?struct { sign: Stream, exp: Stream, mant: Stream, info: ops.SplitFloatInfo } = null;
    if (input.dtype.isFloat16Like()) {
        const r = try ops.splitFloatForward(alloc, input);
        sf = .{ .sign = r.sign, .exp = r.exp, .mant = r.mant, .info = r.info };
    }
    defer if (sf) |*s| {
        s.sign.deinit(alloc);
        s.exp.deinit(alloc);
        s.mant.deinit(alloc);
    };

    // Step 2: precompute heuristic+prior bits per stream-shape per slot.
    // 3 slots × 9 shapes = 27 entries.
    var sign_costs: [9]u64 = undefined;
    var exp_costs: [9]u64 = undefined;
    var mant_costs: [9]u64 = undefined;
    if (sf) |s| {
        const shapes = allStreamShapes();
        for (shapes, 0..) |shape, i| {
            sign_costs[i] = (try streamHeuristicBits(alloc, shape, s.sign)) +
                (if (opts.use_prior) streamPriorBits(shape, s.sign) else 0);
            exp_costs[i] = (try streamHeuristicBits(alloc, shape, s.exp)) +
                (if (opts.use_prior) streamPriorBits(shape, s.exp) else 0);
            mant_costs[i] = (try streamHeuristicBits(alloc, shape, s.mant)) +
                (if (opts.use_prior) streamPriorBits(shape, s.mant) else 0);
        }
    }

    // Step 3: enumerate candidates with their scores.
    var heap: std.PriorityQueue(RankedCandidate, void, ByScore.less) = .empty;
    defer heap.deinit(alloc);

    // tensor_raw — always available
    try heap.push(alloc, .{ .shape = .raw, .score_bits = raw_bits + 64 });

    // split_float — if fp16/bf16
    if (sf) |s| {
        const shapes = allStreamShapes();
        // Domain pruning: bitplane / delta on huge mantissa streams are
        // never optimal in practice (mantissa is near-uniform; bitplane just
        // pays N×alloc cost to discover that). Skip them above a threshold.
        const huge_mant: bool = s.mant.count > 200_000;
        const huge_exp: bool = s.exp.count > 200_000;
        for (shapes, 0..) |s_sign, i| {
            // Bitplane on 1-bit sign is always degenerate.
            if (s_sign == .bitplane) continue;
            for (shapes, 0..) |s_exp, j| {
                if (huge_exp and s_exp == .delta and (s_exp.delta == .raw)) continue;
                for (shapes, 0..) |s_mant, k| {
                    if (huge_mant and s_mant == .bitplane) continue;
                    if (huge_mant and s_mant == .delta) continue;
                    const score = sign_costs[i] + exp_costs[j] + mant_costs[k] + 256;
                    try heap.push(alloc, .{
                        .shape = .{ .split = .{ .s_sign = s_sign, .s_exp = s_exp, .s_mant = s_mant } },
                        .score_bits = score,
                    });
                }
            }
        }
    }

    // tensor_xor — for each base, with body=split_default OR raw
    for (bases, 0..) |base, b| {
        if (base.dtype != input.dtype) continue;
        if (base.shape.len != input.shape.len) continue;
        var same_shape = true;
        for (base.shape, input.shape) |x, y| if (x != y) {
            same_shape = false;
            break;
        };
        if (!same_shape) continue;

        // Heuristic for tensor_xor: XOR residual then evaluate as if split_float
        // would work on it. Cheap upper bound: do the xor, run split_float
        // forward, evaluate cheap heuristic for "split + huffman per stream".
        const xor_r = try ops.tensorXorForward(alloc, input, base);
        var residual = xor_r.residual;
        defer residual.deinit(alloc);

        if (residual.dtype.isFloat16Like()) {
            const r2 = try ops.splitFloatForward(alloc, residual);
            var rs = r2.sign;
            var re = r2.exp;
            var rm = r2.mant;
            defer rs.deinit(alloc);
            defer re.deinit(alloc);
            defer rm.deinit(alloc);
            const cs = try streamHeuristicBits(alloc, .{ .terminal = .huffman }, rs);
            const ce = try streamHeuristicBits(alloc, .{ .delta = .huffman }, re);
            const cm = try streamHeuristicBits(alloc, .{ .terminal = .rans }, rm);
            const score = cs + ce + cm + 384;
            try heap.push(alloc, .{
                .shape = .{ .xor = .{ .base_id = @intCast(b), .body = .split_default } },
                .score_bits = score,
            });
            // Also try tensor_xor + tensor_raw on residual
            try heap.push(alloc, .{
                .shape = .{ .xor = .{ .base_id = @intCast(b), .body = .raw } },
                .score_bits = @as(u64, residual.data.len) * 8 + 128,
            });
        }
    }

    // Step 4: realize candidates from best to worst, prune when impossible to beat.
    var best_result: ?Result = null;
    errdefer if (best_result) |*r| r.deinit(alloc);

    var realized: usize = 0;
    while (heap.pop()) |cand| {
        if (best_result) |b| if (cand.score_bits >= b.actual_bits) {
            if (opts.verbose) std.debug.print("  prune: cand score {d} >= best {d}\n", .{ cand.score_bits, b.actual_bits });
            break;
        };
        if (realized >= opts.realize_top_k) break;
        realized += 1;

        const exp_bits: u8 = if (input.dtype == .f16) 5 else 8;
        const mant_bits: u8 = if (input.dtype == .f16) 10 else 7;

        var node = try buildCandidate(alloc, cand.shape, exp_bits, mant_bits);
        program.compressTensor(alloc, &node, input, bases) catch {
            node.deinit(alloc);
            continue;
        };

        if (opts.verify_each_realization) {
            var back = program.decompressTensor(alloc, &node, bases) catch {
                node.deinit(alloc);
                continue;
            };
            defer back.deinit(alloc);
            const ok = std.mem.eql(u8, input.data, back.data);
            if (!ok) {
                node.deinit(alloc);
                continue;
            }
        }

        const payload = try program.collectPayloadBytes(alloc, &node);
        const program_bytes = try program.serializeProgram(alloc, &node);
        defer alloc.free(program_bytes);
        const total_bits: u64 = @as(u64, payload.len) * 8 + @as(u64, program_bytes.len) * 8;

        const summary = try summarizeShape(alloc, cand.shape);

        if (opts.verbose) {
            std.debug.print("  realize: shape={s} score={d} actual_bits={d} ratio={d:.3}\n",
                .{ summary, cand.score_bits, total_bits, @as(f64, @floatFromInt(raw_bits)) / @as(f64, @floatFromInt(total_bits)) });
        }

        if (best_result) |*b| {
            if (total_bits < b.actual_bits) {
                b.deinit(alloc);
                best_result = .{
                    .program = node,
                    .payload = payload,
                    .actual_bits = total_bits,
                    .raw_bits = raw_bits,
                    .compression_ratio = @as(f64, @floatFromInt(raw_bits)) / @as(f64, @floatFromInt(total_bits)),
                    .verified = true,
                    .template_summary = summary,
                    .chosen_shape = cand.shape,
                };
            } else {
                node.deinit(alloc);
                alloc.free(payload);
                alloc.free(summary);
            }
        } else {
            best_result = .{
                .program = node,
                .payload = payload,
                .actual_bits = total_bits,
                .raw_bits = raw_bits,
                .compression_ratio = @as(f64, @floatFromInt(raw_bits)) / @as(f64, @floatFromInt(total_bits)),
                .verified = true,
                .template_summary = summary,
                .chosen_shape = cand.shape,
            };
        }
    }

    if (best_result) |r| return r;
    return error.NoValidCandidate;
}

fn summarizeShape(alloc: Allocator, shape: TensorShape) ![]u8 {
    return switch (shape) {
        .raw => alloc.dupe(u8, "tensor_raw"),
        .xor => |x| blk: {
            const body = if (x.body == .raw) "raw" else "split+huff/delta+huff/rans";
            break :blk std.fmt.allocPrint(alloc, "tensor_xor(b={d})->{s}", .{ x.base_id, body });
        },
        .split => |s| std.fmt.allocPrint(alloc, "split({s},{s},{s})", .{
            shapeName(s.s_sign), shapeName(s.s_exp), shapeName(s.s_mant),
        }),
    };
}

fn shapeName(s: StreamSubprogShape) []const u8 {
    return switch (s) {
        .terminal => |t| switch (t) {
            .raw => "raw",
            .huffman => "huff",
            .rans => "rans",
        },
        .delta => |t| switch (t) {
            .raw => "delta+raw",
            .huffman => "delta+huff",
            .rans => "delta+rans",
        },
        .bitplane => |t| switch (t) {
            .raw => "bp+raw",
            .huffman => "bp+huff",
            .rans => "bp+rans",
        },
    };
}
