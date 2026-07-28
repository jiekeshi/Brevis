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
        @import("phog.zig"),
        @import("interpreter.zig"),
        @import("literal_encoding.zig"),
        @import("calibration.zig"),
        @import("checkpoint.zig"),
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
