//! PHOG-lite: a hand-tuned probabilistic prior over DSL productions
//! conditioned on the input's statistical features.
//!
//! A real PHOG (Probabilistic Higher-Order Grammar — Bielik, Raychev, Vechev 2016)
//! learns rule probabilities from a corpus by conditioning each production on a
//! tree-context summary. We don't have a corpus of labelled-best programs, so
//! we encode a few well-known rules of thumb instead:
//!
//!   * exponent streams of trained weights have local correlation
//!     → delta_encode usually helps
//!   * mantissa streams in fp16 are near-uniform
//!     → only entropy coding helps, structural transforms usually hurt
//!   * sign streams are very skewed (≈50/50 if zero-centered)
//!     → huffman gives 1 bit/elem (the floor); other transforms add overhead
//!   * exp streams have <= 8 bits → bitplane_split can help isolate the
//!     near-constant high bits
//!
//! The prior returns log2-probability mass that we ADD to the admissible
//! lower bound when ranking candidates. A larger negative log means lower
//! probability and so a higher cost penalty.

const std = @import("std");
const types = @import("types.zig");
const program = @import("program.zig");
const Stream = types.Stream;

pub const Features = struct {
    bits_per_elem: u8,
    count: usize,
    entropy_bits: f64, // 0..bits_per_elem
    unique_count: u32, // distinct symbols seen
    autocorr_decile: f64, // 0..1, fraction of |x[i]-x[i-1]| <= 1 (rough local-corr proxy)

    pub fn fromStream(s: Stream) Features {
        if (s.count == 0) return .{
            .bits_per_elem = s.bits_per_elem,
            .count = 0,
            .entropy_bits = 0,
            .unique_count = 0,
            .autocorr_decile = 0,
        };

        var counts: [65536]u32 = undefined;
        @memset(&counts, 0);
        var unique: u32 = 0;
        for (0..s.count) |i| {
            const v = s.getU32(i) & 0xFFFF;
            if (counts[v] == 0) unique += 1;
            counts[v] += 1;
        }
        const total_f: f64 = @floatFromInt(s.count);
        var H: f64 = 0;
        for (counts) |c| {
            if (c == 0) continue;
            const p = @as(f64, @floatFromInt(c)) / total_f;
            H += -p * std.math.log2(p);
        }

        var close: u32 = 0;
        var i: usize = 1;
        while (i < s.count) : (i += 1) {
            const a = @as(i64, s.getU32(i));
            const b = @as(i64, s.getU32(i - 1));
            const d = if (a > b) a - b else b - a;
            if (d <= 1) close += 1;
        }
        const ac: f64 = if (s.count > 1) @as(f64, @floatFromInt(close)) / @as(f64, @floatFromInt(s.count - 1)) else 0;

        return .{
            .bits_per_elem = s.bits_per_elem,
            .count = s.count,
            .entropy_bits = H,
            .unique_count = unique,
            .autocorr_decile = ac,
        };
    }
};

/// Return a "log-probability" score (always ≤ 0) for choosing this op as the
/// root of a stream subprogram given the input stream features.
/// More negative = less likely.
pub fn streamRuleScore(op: program.OpKind, f: Features) f64 {
    return switch (op) {
        .raw => -2.0, // always usable but rarely optimal — ~25% prior
        .huffman => switch (f.unique_count) {
            0 => -10.0,
            1...16 => -0.3, // strongly prefers low-cardinality alphabets
            17...64 => -1.0,
            else => -2.0,
        },
        .rans => switch (f.unique_count) {
            0 => -10.0,
            1 => -3.0,
            2...16 => -0.7,
            17...256 => -0.5, // close-to-Shannon performance dominates here
            else => -1.5,
        },
        .delta_encode => blk: {
            // Boost when local correlation high
            if (f.autocorr_decile > 0.5) break :blk -0.3;
            if (f.autocorr_decile > 0.2) break :blk -1.0;
            break :blk -3.0;
        },
        .bitplane_split => blk: {
            // Best when bits_per_elem > 1 and the alphabet is heavily skewed,
            // because the high-order planes will be near-constant.
            if (f.bits_per_elem <= 1) break :blk -100.0; // useless on 1-bit
            if (f.entropy_bits / @as(f64, @floatFromInt(f.bits_per_elem)) < 0.5) break :blk -0.5;
            break :blk -2.5;
        },
        else => -10.0,
    };
}

/// Convert a log-probability score (in nats of -log(p)) into bits to add to
/// the heuristic. Cap to a bounded penalty so admissibility on small streams
/// isn't catastrophically violated.
pub fn priorBitsPenalty(log_score: f64, count: usize) u64 {
    // Bits = -log2(prior) but expressed cheaply. We weight by sqrt(count)
    // so the prior matters less for huge streams (where data dominates) and
    // more for tiny ones (where the choice is mostly heuristic).
    const sqrt_count: f64 = std.math.sqrt(@as(f64, @floatFromInt(@max(count, 1))));
    const penalty = -log_score * sqrt_count * 0.5;
    if (penalty < 0) return 0;
    return @intFromFloat(@min(penalty, 1.0e9));
}
