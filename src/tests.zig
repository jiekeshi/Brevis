//! Paper-aligned Brevis test suite.
//!
//! Tests are grouped by behavioral seam so the legacy block/template modules
//! are neither compiled nor able to redefine the implementation contract.

test "paper-aligned module surface compiles" {
    const std = @import("std");
    inline for (.{
        @import("brevis.zig"),
        @import("codec.zig"),
        @import("decomposition.zig"),
        @import("dsl.zig"),
        @import("grammar.zig"),
        @import("grammar_prior.zig"),
        @import("interpreter.zig"),
        @import("literal_encoding.zig"),
        @import("paper_calibration.zig"),
        @import("paper_pipeline.zig"),
        @import("program_format.zig"),
        @import("safetensors.zig"),
        @import("semantics.zig"),
        @import("synthesizer.zig"),
        @import("tensor_archive.zig"),
        @import("types.zig"),
    }) |module| {
        std.testing.refAllDecls(module);
    }
}
