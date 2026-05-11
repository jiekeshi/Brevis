//! Hand-coded macros covering common high-frequency patterns for fp16/bf16
//! tensors. These are merged with the auto-discovered macros (see
//! `discovered_macros.zig`) and exposed to A* as atomic actions.
//!
//! Each macro is a small program tree in `MacroNode` indexed form
//! (same encoding as `discovered_macros.zig`).
//!
//! Why hand-coded? Two reasons:
//!   1. Day-1 ratio without needing a training run.
//!   2. A* depth limit means the system would never reach the equivalent
//!      structure from primitives alone for fp16 (depth 3+).

const lowlevel = @import("lowlevel.zig");
const discovered = @import("discovered_macros.zig");

pub const Macro = discovered.Macro;
pub const MacroNode = discovered.MacroNode;

// Param helpers (matching split_field encoding in lowlevel.zig):
// raw = start | (n_bits << 8) | (k << 16)
fn splitFieldParams(start: u32, n_bits: u32, k: u32) u32 {
    return start | (n_bits << 8) | (k << 16);
}

/// bf16 / fp16 stream → 3 sub-streams (sign:1, exp:5 or 8, mant:7 or 10),
/// each rans-encoded. This is the "split_float + 3×rans" of the high-level
/// pipeline, but in low-level macro form.
pub const BUILT_IN: []const Macro = &.{
    .{
        .name = "split_bf16_huff_rans_rans",
        .mdl_benefit = 1_000_000, // pinned high so search prefers it for 16-bit streams
        .input_bpe = 16,
        // Tree:
        //   0: split_field(15, 1, 16)   → (sign:1, rest:15)
        //   1: huffman                   for sign
        //   2: split_field(7, 8, 15)    rest → (exp:8, mant:7)
        //   3: rans                      for exp
        //   4: rans                      for mant
        .nodes = &.{
            .{ .op_kind = .split_field, .params_raw = splitFieldParams(15, 1, 16), .child_hi = 1, .child_lo = 2, .is_terminal = false },
            .{ .op_kind = .huffman, .params_raw = 0, .child_hi = -1, .child_lo = -1, .is_terminal = true },
            .{ .op_kind = .split_field, .params_raw = splitFieldParams(7, 8, 15), .child_hi = 3, .child_lo = 4, .is_terminal = false },
            .{ .op_kind = .rans, .params_raw = 0, .child_hi = -1, .child_lo = -1, .is_terminal = true },
            .{ .op_kind = .rans, .params_raw = 0, .child_hi = -1, .child_lo = -1, .is_terminal = true },
        },
    },
    .{
        .name = "split_fp16_huff_rans_rans",
        .mdl_benefit = 900_000,
        .input_bpe = 16,
        .nodes = &.{
            .{ .op_kind = .split_field, .params_raw = splitFieldParams(15, 1, 16), .child_hi = 1, .child_lo = 2, .is_terminal = false },
            .{ .op_kind = .huffman, .params_raw = 0, .child_hi = -1, .child_lo = -1, .is_terminal = true },
            .{ .op_kind = .split_field, .params_raw = splitFieldParams(10, 5, 15), .child_hi = 3, .child_lo = 4, .is_terminal = false },
            .{ .op_kind = .rans, .params_raw = 0, .child_hi = -1, .child_lo = -1, .is_terminal = true },
            .{ .op_kind = .rans, .params_raw = 0, .child_hi = -1, .child_lo = -1, .is_terminal = true },
        },
    },
    .{
        // For LayerNorm-gamma-like tensors (very low entropy after split).
        .name = "split_bf16_rans_rans_rans",
        .mdl_benefit = 500_000,
        .input_bpe = 16,
        .nodes = &.{
            .{ .op_kind = .split_field, .params_raw = splitFieldParams(15, 1, 16), .child_hi = 1, .child_lo = 2, .is_terminal = false },
            .{ .op_kind = .rans, .params_raw = 0, .child_hi = -1, .child_lo = -1, .is_terminal = true },
            .{ .op_kind = .split_field, .params_raw = splitFieldParams(7, 8, 15), .child_hi = 3, .child_lo = 4, .is_terminal = false },
            .{ .op_kind = .rans, .params_raw = 0, .child_hi = -1, .child_lo = -1, .is_terminal = true },
            .{ .op_kind = .rans, .params_raw = 0, .child_hi = -1, .child_lo = -1, .is_terminal = true },
        },
    },
};
