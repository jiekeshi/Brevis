//! Public API for the paper-aligned Brevis implementation.
//!
//! The four primary seams are synthesis, canonical program I/O, and exact
//! execution. Archive and calibration modules compose these seams without
//! changing their semantics.

const std = @import("std");

pub const types = @import("types.zig");
pub const dsl = @import("dsl.zig");
pub const grammar = @import("grammar.zig");
pub const grammar_prior = @import("grammar_prior.zig");
pub const calibration = @import("calibration.zig");
pub const checkpoint = @import("checkpoint.zig");
pub const safetensors = @import("safetensors.zig");
/// Low-level BRTA framing. Its `parseStructural` seam deliberately does not
/// cross-bind record metadata to the embedded safetensors header; use
/// `checkpoint.decompress*` for full archive validation and reconstruction.
pub const tensor_archive = @import("tensor_archive.zig");

const interpreter = @import("interpreter.zig");
const program_format = @import("program_format.zig");
const synthesizer = @import("synthesizer.zig");

pub const Program = dsl.Program;
pub const TensorProgram = dsl.TensorProgram;
pub const SynthesisOptions = synthesizer.Options;
pub const SynthesisResult = synthesizer.Result;
pub const ProgramDecodeLimits = program_format.DecodeLimits;

/// Find the smallest exact program encountered within the configured finite
/// search budget. The optional PHOG in `options.rule_model` orders the queue
/// and selects the bounded terminal-completion frontier; exact size still
/// decides between completed candidates.
pub fn synthesize(
    alloc: std.mem.Allocator,
    target: types.Stream,
    dtype: types.Dtype,
    options: SynthesisOptions,
) !SynthesisResult {
    return synthesizer.synthesize(alloc, target, dtype, options);
}

/// Emit the unique canonical representation of one semantic program.
pub fn writeProgram(
    alloc: std.mem.Allocator,
    program: Program,
) ![]u8 {
    return program_format.serialize(alloc, program);
}

/// Count exactly the bytes that `writeProgram` will emit.
pub fn serializedProgramSize(
    alloc: std.mem.Allocator,
    program: Program,
) !usize {
    return program_format.serializedSize(alloc, program);
}

/// Read exactly one canonical semantic program under explicit resource limits.
pub fn readProgram(
    alloc: std.mem.Allocator,
    bytes: []const u8,
    limits: ProgramDecodeLimits,
) !Program {
    return program_format.deserialize(alloc, bytes, limits);
}

/// Execute a semantic program into a newly allocated physical-word stream.
pub fn execute(
    alloc: std.mem.Allocator,
    program: Program,
) !types.Stream {
    return interpreter.execute(alloc, program);
}

/// Execute into caller-owned physical-word storage.
pub fn executeInto(
    alloc: std.mem.Allocator,
    program: Program,
    output: types.Stream,
) !void {
    return interpreter.executeInto(alloc, program, output);
}
