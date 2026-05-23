//! Grammar for the program-synthesis search, formulated the Euphony way:
//! a context-free grammar `G = ⟨N, Σ, R, S⟩` over which PHOG defines
//! `q(A → β | c)`. Brevis uses a tiny grammar — a single non-terminal `S`
//! (stream program) and one production per low-level op (the op's parameter
//! values are enumerated lazily at expansion time, in the spirit of the
//! pivot-grammar abstraction `param⋆` from the paper).
//!
//! Productions are identified by `lowlevel.OpKind` directly (since each op
//! has fixed arity, the production it generates is uniquely determined).
//! Arity is 0 for terminals (huffman/rans/raw), 1 for chain ops, 2 for
//! `split_field`.

const std = @import("std");
const lowlevel = @import("lowlevel.zig");

/// Sole non-terminal for now.
pub const NonTerminal = enum(u8) { S = 0 };

/// Total number of OpKind variants. Used to size PHOG tables.
pub const N_OP_KINDS: usize = 19; // OpKind: 0..14 + 16..18 (gap at 15)

/// True if `op` produces a complete program (no further sentential-form holes).
pub fn isTerminal(op: lowlevel.OpKind) bool {
    return switch (op) {
        .huffman, .rans, .raw => true,
        else => false,
    };
}

/// Number of NT children this op's production introduces.
pub fn arity(op: lowlevel.OpKind) u8 {
    return switch (op) {
        .huffman, .rans, .raw => 0,
        .split_field => 2,
        else => 1,
    };
}

/// Concrete parameter values to try for `op` when expanding it at the search
/// frontier. Returns an empty list for parameter-free ops — but the search
/// will then enumerate ONE expansion with `params.raw = 0`. The (op_kind ×
/// param_value) cross-product is the actual branching factor at a hole.
///
/// `bpe` is the bits-per-elem of the input stream at the expansion point.
/// Some param ranges depend on it (e.g. rotate amount is mod bpe).
pub fn paramChoices(op: lowlevel.OpKind, bpe: u8) []const u32 {
    return switch (op) {
        .xor_const => if (bpe <= 8)
            &.{ 0xFF, 0xAA, 0x55, 0x80 }
        else
            &.{ 0xFFFF, 0xAAAA, 0x5555, 0x8000 },
        .add_const_mod => if (bpe <= 8)
            &.{ 1, 17, 0x55 }
        else
            &.{ 1, 17, 0x100 },
        .rotate_bits => if (bpe == 8) &.{ 1, 3, 4 } else &.{ 1, 4, 8 },
        .bit_swap_pair => &.{}, // too combinatorial — don't enumerate in MVP
        .mul_const_odd_mod => &.{ 3, 5, 7, 11, 17 },
        .xor_with_shift => if (bpe == 8) &.{ 1, 2, 4 } else &.{ 1, 2, 4, 8 },
        .split_field => splitFieldParams(bpe),
        else => &.{0}, // parameter-free ops use raw=0
    };
}

fn splitFieldParams(bpe: u8) []const u32 {
    return switch (bpe) {
        32 => &SPLIT_PARAMS_32,
        31 => &SPLIT_PARAMS_31,
        23 => &SPLIT_PARAMS_23,
        16 => &SPLIT_PARAMS_16,
        15 => &SPLIT_PARAMS_15,
        8 => &SPLIT_PARAMS_8,
        7 => &SPLIT_PARAMS_7,
        else => &.{},
    };
}

fn mkSplit(start: u32, n: u32, k: u32) u32 {
    return start | (n << 8) | (k << 16);
}

// Note: `param.raw = start | (n_bits << 8) | (k << 16)` per lowlevel.zig.
// The search overrides the `k` field with the hole's actual bits-per-elem at
// expansion time, so the `k` written here is only documentation of intent.
const SPLIT_PARAMS_32 = [_]u32{
    mkSplit(31, 1, 32), // sign vs rest (fp32: 1/8/23)
    mkSplit(23, 9, 32), // sign+exponent together vs mantissa
};
const SPLIT_PARAMS_31 = [_]u32{
    mkSplit(23, 8, 31), // exponent (fp32) on the sign-stripped 31-bit word
};
const SPLIT_PARAMS_23 = [_]u32{
    mkSplit(16, 7, 23), // top 7 mantissa bits vs low 16 (16-bit chunk → raw/rANS)
    mkSplit(0, 14, 23), // low 14 (rANS-able) vs high 9
};
const SPLIT_PARAMS_16 = [_]u32{
    mkSplit(15, 1, 16), // sign vs rest (fp16/bf16)
    mkSplit(10, 5, 16), // exp (fp16)
    mkSplit(7, 8, 16), // exp (bf16) — but bpe=16 means we keep the high split
    mkSplit(8, 8, 16),
};
const SPLIT_PARAMS_15 = [_]u32{
    mkSplit(10, 5, 15), // exp (fp16) on the "rest" stream
    mkSplit(7, 8, 15), // exp (bf16)
};
const SPLIT_PARAMS_8 = [_]u32{
    mkSplit(7, 1, 8), // top bit
    mkSplit(4, 4, 8), // nibble
    mkSplit(0, 4, 8),
    mkSplit(0, 2, 8),
    mkSplit(6, 2, 8),
};
const SPLIT_PARAMS_7 = [_]u32{
    mkSplit(6, 1, 7),
    mkSplit(0, 3, 7),
    mkSplit(4, 3, 7),
};

/// Context for PHOG: just (parent_op, slot) — slot ∈ {0, 1} disambiguates the
/// two children of split_field. Root has parent_op = null.
pub const Context = struct {
    parent_op: ?lowlevel.OpKind,
    slot: u8 = 0,

    pub const ROOT: Context = .{ .parent_op = null, .slot = 0 };

    /// Encode to a small integer for hashmap lookup. `(N_OP_KINDS + 1) × 2`
    /// distinct values.
    pub fn encode(self: Context) u16 {
        const op_id: u16 = if (self.parent_op) |op| @as(u16, @intFromEnum(op)) + 1 else 0;
        return (op_id << 1) | (self.slot & 1);
    }
};

pub const N_CONTEXTS: usize = (N_OP_KINDS + 1) * 2;

/// All op kinds — used for iteration during training and h-fixpoint computation.
/// Includes the inverse forms (prefix_xor, cumsum_mod) so that PHOG and h treat
/// them as valid (though search typically only enumerates the forward forms).
pub const ALL_OPS: []const lowlevel.OpKind = &.{
    .xor_const,
    .add_const_mod,
    .rotate_bits,
    .bit_swap_pair,
    .bit_reverse,
    .negate_mod,
    .mul_const_odd_mod,
    .xor_with_shift,
    .gray_code,
    .inv_gray_code,
    .xor_prev,
    .prefix_xor,
    .diff_mod,
    .cumsum_mod,
    .split_field,
    .huffman,
    .rans,
    .raw,
};
