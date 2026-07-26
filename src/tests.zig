//! Paper-aligned Brevis test suite.
//!
//! Tests are grouped by behavioral seam so the legacy block/template modules
//! are neither compiled nor able to redefine the implementation contract.

const _api = @import("api_tests.zig");
const _codec = @import("codec_tests.zig");
const _grammar = @import("grammar_tests.zig");
const _grammar_prior = @import("grammar_prior_tests.zig");
const _interpreter = @import("interpreter_tests.zig");
const _literal_encoding = @import("literal_encoding_tests.zig");
const _main = @import("main.zig");
const _paper_calibration = @import("paper_calibration_tests.zig");
const _paper_pipeline = @import("paper_pipeline_tests.zig");
const _paper = @import("paper_tests.zig");
const _program_format = @import("program_format_tests.zig");
const _safetensors = @import("safetensors_tests.zig");
const _synthesizer = @import("synthesizer_tests.zig");
const _tensor_archive = @import("tensor_archive_tests.zig");

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
