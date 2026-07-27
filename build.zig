const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    _ = b.addModule("brevis", .{
        .root_source_file = b.path("src/brevis.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const exe = b.addExecutable(.{
        .name = "brevis",
        .root_module = exe_mod,
    });
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run brevis CLI");
    run_step.dependOn(&run_cmd.step);

    const test_step = b.step("test", "Run all tests");
    const test_roots = [_][]const u8{
        "src/tests.zig",
        "src/api_tests.zig",
        "src/codec_tests.zig",
        "src/grammar_tests.zig",
        "src/grammar_prior_tests.zig",
        "src/interpreter_tests.zig",
        "src/literal_encoding_tests.zig",
        "src/main.zig",
        "src/paper_calibration_tests.zig",
        "src/paper_pipeline_tests.zig",
        "src/paper_tests.zig",
        "src/program_format_tests.zig",
        "src/safetensors_tests.zig",
        "src/synthesizer_tests.zig",
        "src/tensor_archive_tests.zig",
    };
    for (test_roots) |root| {
        const test_mod = b.createModule(.{
            .root_source_file = b.path(root),
            .target = target,
            .optimize = optimize,
        });
        const tests = b.addTest(.{ .root_module = test_mod });
        const run_tests = b.addRunArtifact(tests);
        test_step.dependOn(&run_tests.step);
    }
}
