//! Command-line interface for the paper-aligned whole-tensor implementation.

const std = @import("std");
const grammar = @import("grammar.zig");
const phog = @import("phog.zig");
const calibration = @import("calibration.zig");
const checkpoint = @import("checkpoint.zig");
const safetensors = @import("safetensors.zig");
const synthesizer = @import("synthesizer.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const MAX_PRIOR_BYTES: usize = 512 * 1024 * 1024;
const DEFAULT_MAX_TOTAL_BYTES: usize = types.defaultLargeByteLimit;
/// One embedding matrix of a large-vocabulary checkpoint already exceeds a
/// gigabyte, so a smaller cap rejects ordinary models rather than bad input.
const DEFAULT_MAX_TENSOR_BYTES: usize = 4 * 1024 * 1024 * 1024;
const DEFAULT_MAX_PREFIX_BYTES: usize = 64 * 1024 * 1024;

const ResourceLimits = struct {
    max_total_bytes: usize = DEFAULT_MAX_TOTAL_BYTES,
    max_tensor_bytes: usize = DEFAULT_MAX_TENSOR_BYTES,
    max_prefix_bytes: usize = DEFAULT_MAX_PREFIX_BYTES,
};

const Command = enum {
    compress,
    decompress,
    verify,
    calibrate,
    config,
};

const Arguments = struct {
    command: Command,
    positional: std.ArrayList([]const u8) = .empty,
    prior_path: ?[]const u8 = null,
    max_tensors: usize = calibration.DEFAULT_TENSORS,
    synthesis: synthesizer.Options = .{ .seed_float_fields = false },
    resources: ResourceLimits = .{},
    workers: usize = checkpoint.DEFAULT_WORKERS,
    saw_prior: bool = false,
    saw_tensors: bool = false,
    saw_workers: bool = false,
    saw_search_option: bool = false,

    fn deinit(self: *Arguments, alloc: Allocator) void {
        self.positional.deinit(alloc);
    }
};

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout_file = std.Io.File.stdout().writer(io, &stdout_buffer);
    const stdout = &stdout_file.interface;
    var stderr_buffer: [4096]u8 = undefined;
    var stderr_file = std.Io.File.stderr().writer(io, &stderr_buffer);
    const stderr = &stderr_file.interface;

    var argv: std.ArrayList([]u8) = .empty;
    defer {
        for (argv.items) |arg| alloc.free(arg);
        argv.deinit(alloc);
    }
    var iterator = init.minimal.args.iterate();
    defer iterator.deinit();
    while (iterator.next()) |arg| {
        const owned_arg = try alloc.dupe(u8, arg);
        argv.append(alloc, owned_arg) catch |err| {
            alloc.free(owned_arg);
            return err;
        };
    }

    run(alloc, io, stdout, argv.items) catch |err| {
        if (err == error.InvalidArguments) {
            try usage(stderr);
            try stderr.flush();
            std.process.exit(2);
        }
        try stderr.print("brevis: {s}\n", .{@errorName(err)});
        try stderr.flush();
        std.process.exit(1);
    };
    try stdout.flush();
}

fn run(
    alloc: Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    argv: []const []const u8,
) !void {
    if (argv.len < 2) return error.InvalidArguments;
    if (std.mem.eql(u8, argv[1], "--help") or
        std.mem.eql(u8, argv[1], "-h"))
    {
        try usage(out);
        return;
    }

    var args = try parseArguments(alloc, argv[1..]);
    defer args.deinit(alloc);
    if (!args.saw_workers)
        args.workers = std.Thread.getCpuCount() catch checkpoint.DEFAULT_WORKERS;
    try validateArguments(args);

    switch (args.command) {
        .compress => try commandCompress(
            alloc,
            io,
            out,
            args.positional.items[0],
            args.positional.items[1],
            args.prior_path,
            args.max_tensors,
            args.synthesis,
            args.resources,
            args.workers,
        ),
        .decompress => try commandDecompress(
            alloc,
            io,
            out,
            args.positional.items[0],
            args.positional.items[1],
            args.resources,
            args.workers,
        ),
        .verify => try commandVerify(
            alloc,
            io,
            out,
            args.positional.items[0],
            args.positional.items[1],
            args.resources,
            args.workers,
        ),
        .calibrate => try commandCalibrate(
            alloc,
            io,
            out,
            args.positional.items[0],
            args.positional.items[1],
            args.max_tensors,
            args.synthesis,
            args.resources,
            args.workers,
        ),
        .config => try commandConfig(
            out,
            args.synthesis,
            args.resources,
            args.workers,
        ),
    }
}

fn parseArguments(
    alloc: Allocator,
    words: []const []const u8,
) !Arguments {
    if (words.len == 0) return error.InvalidArguments;
    var args = Arguments{
        .command = std.meta.stringToEnum(Command, words[0]) orelse
            return error.InvalidArguments,
    };
    errdefer args.deinit(alloc);

    var index: usize = 1;
    while (index < words.len) {
        const word = words[index];
        if (!std.mem.startsWith(u8, word, "--")) {
            try args.positional.append(alloc, word);
            index += 1;
            continue;
        }
        if (std.mem.eql(u8, word, "--help"))
            return error.InvalidArguments;
        if (index + 1 >= words.len) return error.InvalidArguments;
        const value = words[index + 1];
        index += 2;

        if (std.mem.eql(u8, word, "--prior")) {
            args.saw_prior = true;
            args.prior_path = value;
        } else if (std.mem.eql(u8, word, "--tensors")) {
            args.saw_tensors = true;
            args.max_tensors = try parseUnsigned(usize, value);
        } else if (std.mem.eql(u8, word, "--max-expansions")) {
            args.saw_search_option = true;
            args.synthesis.max_expansions = try parseUnsigned(usize, value);
        } else if (std.mem.eql(u8, word, "--max-nodes")) {
            args.saw_search_option = true;
            args.synthesis.max_nodes = try parseUnsigned(usize, value);
        } else if (std.mem.eql(u8, word, "--seed-float-fields")) {
            args.saw_search_option = true;
            args.synthesis.seed_float_fields =
                try parseUnsigned(u1, value) == 1;
        } else if (std.mem.eql(u8, word, "--max-depth")) {
            args.saw_search_option = true;
            args.synthesis.grammar_options.max_depth =
                try parseUnsigned(u8, value);
        } else if (std.mem.eql(u8, word, "--max-repeat-period")) {
            args.saw_search_option = true;
            args.synthesis.grammar_options.max_repeat_period =
                try parseUnsigned(usize, value);
        } else if (std.mem.eql(u8, word, "--max-concat-splits")) {
            args.saw_search_option = true;
            args.synthesis.grammar_options.max_concat_splits =
                try parseUnsigned(usize, value);
        } else if (std.mem.eql(u8, word, "--max-map-constants")) {
            args.saw_search_option = true;
            args.synthesis.grammar_options.max_map_constants =
                try parseUnsigned(usize, value);
        } else if (std.mem.eql(u8, word, "--max-rotations")) {
            args.saw_search_option = true;
            args.synthesis.grammar_options.max_rotations =
                try parseUnsigned(usize, value);
        } else if (std.mem.eql(u8, word, "--max-field-splits")) {
            args.saw_search_option = true;
            args.synthesis.grammar_options.max_field_splits =
                try parseUnsigned(usize, value);
        } else if (std.mem.eql(u8, word, "--max-total-bytes")) {
            args.resources.max_total_bytes =
                try parseUnsigned(usize, value);
        } else if (std.mem.eql(u8, word, "--max-tensor-bytes")) {
            args.resources.max_tensor_bytes =
                try parseUnsigned(usize, value);
            args.synthesis.max_decomposition_bytes =
                args.resources.max_tensor_bytes;
        } else if (std.mem.eql(u8, word, "--max-prefix-bytes")) {
            args.resources.max_prefix_bytes =
                try parseUnsigned(usize, value);
        } else if (std.mem.eql(u8, word, "--workers")) {
            args.saw_workers = true;
            args.workers = try parseUnsigned(usize, value);
        } else {
            return error.InvalidArguments;
        }
    }
    return args;
}

fn validateArguments(args: Arguments) !void {
    const expected_positionals: usize = switch (args.command) {
        .compress, .decompress, .verify, .calibrate => 2,
        .config => 0,
    };
    if (args.positional.items.len != expected_positionals)
        return error.InvalidArguments;
    if (args.synthesis.max_nodes == 0)
        return error.InvalidArguments;
    if (args.workers == 0)
        return error.InvalidArguments;
    const archive_path: ?[]const u8 = switch (args.command) {
        .compress => args.positional.items[1],
        .decompress, .verify => args.positional.items[0],
        .calibrate, .config => null,
    };
    if (archive_path) |path|
        if (!std.mem.endsWith(u8, path, ".brv"))
            return error.InvalidArguments;
    const grammar_options = args.synthesis.grammar_options;
    if (grammar_options.max_repeat_period > grammar.HARD_MAX_REPEAT_PERIOD or
        grammar_options.max_concat_splits > grammar.HARD_MAX_CONCAT_SPLITS or
        grammar_options.max_map_constants > grammar.HARD_MAX_MAP_CONSTANTS or
        grammar_options.max_rotations > grammar.HARD_MAX_ROTATIONS or
        grammar_options.max_field_splits > grammar.HARD_MAX_FIELD_SPLITS)
    {
        return error.InvalidArguments;
    }
    if (args.saw_prior and args.command != .compress)
        return error.InvalidArguments;
    if (args.saw_tensors and
        args.command != .compress and
        args.command != .calibrate)
    {
        return error.InvalidArguments;
    }
    if (args.saw_search_option and
        args.command != .compress and
        args.command != .calibrate and
        args.command != .config)
    {
        return error.InvalidArguments;
    }
}

fn parseUnsigned(comptime T: type, text: []const u8) !T {
    return std.fmt.parseInt(T, text, 10) catch error.InvalidArguments;
}

fn parseForAllocationFailureCheck(alloc: Allocator) !void {
    var args = try parseArguments(alloc, &.{
        "compress",
        "in.safetensors",
        "out.brv",
        "--max-expansions",
        "64",
        "--max-tensor-bytes",
        "1048576",
    });
    defer args.deinit(alloc);
    try validateArguments(args);
}

fn commandCompress(
    alloc: Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    source_path: []const u8,
    archive_path: []const u8,
    prior_path: ?[]const u8,
    max_calibration_tensors: usize,
    base_options: synthesizer.Options,
    resources: ResourceLimits,
    workers: usize,
) !void {
    var prior: ?phog.Prior = if (prior_path) |path|
        try loadPrior(alloc, io, path)
    else
        null;
    defer if (prior) |*model| model.deinit(alloc);

    var synthesis = base_options;
    if (prior) |*model| synthesis.phog_prior = model;
    var summary = try checkpoint.compressFile(
        alloc,
        io,
        source_path,
        archive_path,
        .{
            .synthesis = synthesis,
            .max_calibration_tensors = max_calibration_tensors,
            .workers = workers,
            .max_source_bytes = resources.max_total_bytes,
            .max_prefix_bytes = resources.max_prefix_bytes,
            .max_tensor_bytes = resources.max_tensor_bytes,
            .max_archive_bytes = resources.max_total_bytes,
        },
    );
    defer summary.deinit(alloc);

    var expanded: usize = 0;
    var completed: usize = 0;
    var fallback: usize = 0;
    var budget_exhausted: usize = 0;
    for (summary.tensors) |tensor| {
        expanded = std.math.add(usize, expanded, tensor.expanded) catch
            return error.IntegerOverflow;
        completed = std.math.add(
            usize,
            completed,
            tensor.completed_candidates,
        ) catch return error.IntegerOverflow;
        fallback += @intFromBool(tensor.used_literal_fallback);
        budget_exhausted += @intFromBool(
            tensor.status == .budget_exhausted,
        );
    }

    try out.print(
        "compressed {d} tensors: {d} -> {d} bytes ({d:.3}x)\n",
        .{
            summary.tensors.len,
            summary.source_bytes,
            summary.archive_bytes,
            compressionRatio(summary.source_bytes, summary.archive_bytes),
        },
    );
    try out.print(
        "search: expanded={d}, completed={d}, budget_exhausted={d}, literal_fallback={d}, prior={s}\n",
        .{
            expanded,
            completed,
            budget_exhausted,
            fallback,
            prior_path orelse if (base_options.max_expansions != 0 and
                max_calibration_tensors != 0)
                "checkpoint-local"
            else
                "uniform",
        },
    );
}

fn commandDecompress(
    alloc: Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    archive_path: []const u8,
    output_path: []const u8,
    resources: ResourceLimits,
    workers: usize,
) !void {
    const summary = try checkpoint.decompressFile(
        alloc,
        io,
        archive_path,
        output_path,
        decompressLimits(resources, workers),
    );
    try out.print(
        "decompressed {d} tensors: {d} -> {d} bytes\n",
        .{ summary.tensor_count, summary.archive_bytes, summary.output_bytes },
    );
}

fn commandVerify(
    alloc: Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    archive_path: []const u8,
    source_path: []const u8,
    resources: ResourceLimits,
    workers: usize,
) !void {
    const summary = try checkpoint.verifyFile(
        alloc,
        io,
        archive_path,
        source_path,
        decompressLimits(resources, workers),
    );
    try out.print(
        "verified {d} tensors and {d} reconstructed bytes exactly\n",
        .{ summary.tensor_count, summary.output_bytes },
    );
}

fn commandCalibrate(
    alloc: Allocator,
    io: std.Io,
    out: *std.Io.Writer,
    source_path: []const u8,
    prior_path: []const u8,
    max_tensors: usize,
    synthesis: synthesizer.Options,
    resources: ResourceLimits,
    workers: usize,
) !void {
    var loaded = try safetensors.loadFromPathWithLimits(
        alloc,
        io,
        source_path,
        .{
            .max_file_bytes = resources.max_total_bytes,
            .max_header_bytes = resources.max_prefix_bytes,
            .max_tensor_bytes = resources.max_tensor_bytes,
        },
    );
    defer loaded.deinit(alloc);

    var result = try calibration.trainParallel(
        alloc,
        io,
        loaded.tensors,
        .{
            .max_tensors = max_tensors,
            .synthesis = synthesis,
        },
        workers,
    );
    defer result.deinit(alloc);
    const encoded = try result.serialize(alloc);
    defer alloc.free(encoded);
    try writeFileAtomic(io, prior_path, encoded);

    try out.print(
        "calibrated {d}/{d} complete tensors -> {s} ({d} bytes)\n",
        .{
            result.observed_tensors,
            loaded.tensors.len,
            prior_path,
            encoded.len,
        },
    );
    try out.print(
        "search: expanded={d}, completed={d}, budget_exhausted={d}, literal_fallback={d}\n",
        .{
            result.expanded,
            result.completed_candidates,
            result.budget_exhausted_tensors,
            result.literal_fallback_tensors,
        },
    );
}

fn commandConfig(
    out: *std.Io.Writer,
    options: synthesizer.Options,
    resources: ResourceLimits,
    workers: usize,
) !void {
    var json: std.json.Stringify = .{
        .writer = out,
        .options = .{ .whitespace = .indent_2 },
    };
    try json.beginObject();
    try json.objectField("synthesis_unit");
    try json.write("complete_tensor");
    try json.objectField("objective");
    try json.write("canonical_program_bytes");
    try json.objectField("literal_fallback");
    try json.write(true);
    try json.objectField("phog_role");
    try json.write("queue_order_and_terminal_frontier");
    try json.objectField("archive");
    try json.write("BRTA-v3");
    try json.objectField("workers");
    try json.write(workers);
    try json.objectField("max_expansions");
    try json.write(options.max_expansions);
    try json.objectField("max_nodes");
    try json.write(options.max_nodes);
    try json.objectField("seed_float_fields");
    try json.write(options.seed_float_fields);
    try json.objectField("max_decomposition_bytes");
    try json.write(options.max_decomposition_bytes);
    try json.objectField("max_depth");
    try json.write(options.grammar_options.max_depth);
    try json.objectField("max_repeat_period");
    try json.write(options.grammar_options.max_repeat_period);
    try json.objectField("max_concat_splits");
    try json.write(options.grammar_options.max_concat_splits);
    try json.objectField("max_map_constants");
    try json.write(options.grammar_options.max_map_constants);
    try json.objectField("max_rotations");
    try json.write(options.grammar_options.max_rotations);
    try json.objectField("max_field_splits");
    try json.write(options.grammar_options.max_field_splits);
    try json.objectField("max_total_bytes");
    try json.write(resources.max_total_bytes);
    try json.objectField("max_tensor_bytes");
    try json.write(resources.max_tensor_bytes);
    try json.objectField("max_prefix_bytes");
    try json.write(resources.max_prefix_bytes);
    try json.endObject();
    try out.writeByte('\n');
}

fn decompressLimits(
    resources: ResourceLimits,
    workers: usize,
) checkpoint.DecompressLimits {
    const program_slack: usize = 16 * 1024 * 1024;
    const program_bytes = @min(
        resources.max_total_bytes,
        std.math.add(
            usize,
            resources.max_tensor_bytes,
            program_slack,
        ) catch std.math.maxInt(usize),
    );
    return .{
        .workers = workers,
        .max_archive_bytes = resources.max_total_bytes,
        .max_output_bytes = resources.max_total_bytes,
        .archive = .{
            .max_prefix_bytes = resources.max_prefix_bytes,
            .max_record_bytes = resources.max_total_bytes,
            .max_program_bytes = program_bytes,
            .max_tensor_output_bytes = resources.max_tensor_bytes,
            .program = .{
                .max_nodes = 1_000_000,
                .max_depth = 256,
                .max_output_bytes = resources.max_tensor_bytes,
                .max_literal_bytes = resources.max_tensor_bytes,
            },
        },
    };
}

fn loadPrior(
    alloc: Allocator,
    io: std.Io,
    path: []const u8,
) !phog.Prior {
    const encoded = try std.Io.Dir.cwd().readFileAlloc(
        io,
        path,
        alloc,
        .limited(MAX_PRIOR_BYTES),
    );
    defer alloc.free(encoded);
    return phog.Prior.deserialize(alloc, encoded);
}

fn writeFileAtomic(
    io: std.Io,
    path: []const u8,
    bytes: []const u8,
) !void {
    var atomic = try std.Io.Dir.cwd().createFileAtomic(
        io,
        path,
        .{ .replace = true },
    );
    defer atomic.deinit(io);
    var buffer: [64 * 1024]u8 = undefined;
    var writer = atomic.file.writer(io, &buffer);
    try writer.interface.writeAll(bytes);
    try writer.interface.flush();
    try atomic.file.sync(io);
    try atomic.replace(io);
}

fn compressionRatio(source_bytes: usize, archive_bytes: usize) f64 {
    if (archive_bytes == 0) return 0;
    return @as(f64, @floatFromInt(source_bytes)) /
        @as(f64, @floatFromInt(archive_bytes));
}

fn usage(writer: *std.Io.Writer) !void {
    try writer.writeAll(
        \\Brevis — exact whole-tensor program synthesis
        \\
        \\  brevis compress   <model.safetensors> <model.brv> [--prior model.brvp] [--tensors N] [search options]
        \\  brevis decompress <model.brv> <restored.safetensors>
        \\  brevis verify     <model.brv> <model.safetensors>
        \\  brevis calibrate  <model.safetensors> <model.brvp> [--tensors N] [search options]
        \\  brevis config [search options]
        \\
        \\Search options:
        \\  --max-expansions N
        \\  --max-nodes N
        \\  --seed-float-fields 0|1
        \\  --max-depth N
        \\  --max-repeat-period N
        \\  --max-concat-splits N
        \\  --max-map-constants N
        \\  --max-rotations N
        \\  --max-field-splits N
        \\
        \\Parallel file execution and calibration:
        \\  --workers N
        \\
        \\Resource limits (bytes):
        \\  --max-total-bytes N
        \\  --max-tensor-bytes N
        \\  --max-prefix-bytes N
        \\
    );
}

test "CLI accepts only paper-aligned whole-tensor controls" {
    const alloc = std.testing.allocator;
    var args = try parseArguments(alloc, &.{
        "compress",
        "model.safetensors",
        "model.brv",
        "--prior",
        "model.brvp",
        "--tensors",
        "7",
        "--max-expansions",
        "0",
        "--max-depth",
        "2",
        "--seed-float-fields",
        "0",
        "--max-concat-splits",
        "0",
        "--max-tensor-bytes",
        "1048576",
        "--workers",
        "4",
    });
    defer args.deinit(alloc);
    try validateArguments(args);
    try std.testing.expectEqual(Command.compress, args.command);
    try std.testing.expectEqual(@as(usize, 0), args.synthesis.max_expansions);
    try std.testing.expect(!args.synthesis.seed_float_fields);
    try std.testing.expectEqual(
        @as(u8, 2),
        args.synthesis.grammar_options.max_depth,
    );
    try std.testing.expectEqual(
        @as(usize, 0),
        args.synthesis.grammar_options.max_concat_splits,
    );
    try std.testing.expectEqualStrings("model.brvp", args.prior_path.?);
    try std.testing.expectEqual(@as(usize, 7), args.max_tensors);
    try std.testing.expectEqual(
        @as(usize, 1048576),
        args.resources.max_tensor_bytes,
    );
    try std.testing.expectEqual(
        @as(usize, 1048576),
        args.synthesis.max_decomposition_bytes,
    );
    try std.testing.expectEqual(@as(usize, 4), args.workers);

    try std.testing.expectError(
        error.InvalidArguments,
        parseArguments(alloc, &.{
            "compress",
            "in",
            "out",
            "--plan",
            "fixed",
        }),
    );
}

test "CLI defaults leave one-expansion search to PHOG" {
    const alloc = std.testing.allocator;
    var args = try parseArguments(alloc, &.{"config"});
    defer args.deinit(alloc);
    try validateArguments(args);
    try std.testing.expectEqual(@as(usize, 1), args.synthesis.max_expansions);
    try std.testing.expect(!args.synthesis.seed_float_fields);

    var legacy_suffix = try parseArguments(alloc, &.{
        "compress",
        "model.safetensors",
        "model.brta",
    });
    defer legacy_suffix.deinit(alloc);
    try std.testing.expectError(
        error.InvalidArguments,
        validateArguments(legacy_suffix),
    );
}

test "CLI rejects command-specific flags and an impossible node cap" {
    const alloc = std.testing.allocator;
    var decode_args = try parseArguments(alloc, &.{
        "decompress",
        "model.brv",
        "out.safetensors",
        "--max-depth",
        "1",
    });
    defer decode_args.deinit(alloc);
    try std.testing.expectError(
        error.InvalidArguments,
        validateArguments(decode_args),
    );

    var zero_nodes = try parseArguments(alloc, &.{
        "config",
        "--max-nodes",
        "0",
    });
    defer zero_nodes.deinit(alloc);
    try std.testing.expectError(
        error.InvalidArguments,
        validateArguments(zero_nodes),
    );

    var excessive_fanout = try parseArguments(alloc, &.{
        "config",
        "--max-concat-splits",
        "17",
    });
    defer excessive_fanout.deinit(alloc);
    try std.testing.expectError(
        error.InvalidArguments,
        validateArguments(excessive_fanout),
    );

    var zero_workers = try parseArguments(alloc, &.{
        "decompress",
        "model.brv",
        "out.safetensors",
        "--workers",
        "0",
    });
    defer zero_workers.deinit(alloc);
    try std.testing.expectError(
        error.InvalidArguments,
        validateArguments(zero_workers),
    );

    var calibration_workers = try parseArguments(alloc, &.{
        "calibrate",
        "model.safetensors",
        "model.brvp",
        "--workers",
        "4",
    });
    defer calibration_workers.deinit(alloc);
    try validateArguments(calibration_workers);
    try std.testing.expectEqual(@as(usize, 4), calibration_workers.workers);
}

test "CLI decoder defaults cap one materialized tensor and can be tightened" {
    const alloc = std.testing.allocator;
    var defaults = try parseArguments(alloc, &.{
        "decompress",
        "model.brv",
        "out.safetensors",
    });
    defer defaults.deinit(alloc);
    const default_limits = decompressLimits(
        defaults.resources,
        defaults.workers,
    );
    try std.testing.expectEqual(
        DEFAULT_MAX_TENSOR_BYTES,
        default_limits.archive.max_tensor_output_bytes,
    );
    try std.testing.expectEqual(
        DEFAULT_MAX_TENSOR_BYTES,
        default_limits.archive.program.max_output_bytes,
    );

    var tightened = try parseArguments(alloc, &.{
        "decompress",
        "model.brv",
        "out.safetensors",
        "--max-total-bytes",
        "4096",
        "--max-tensor-bytes",
        "1024",
        "--max-prefix-bytes",
        "512",
    });
    defer tightened.deinit(alloc);
    try validateArguments(tightened);
    const tight_limits = decompressLimits(
        tightened.resources,
        tightened.workers,
    );
    try std.testing.expectEqual(
        @as(usize, 4096),
        tight_limits.max_output_bytes,
    );
    try std.testing.expectEqual(
        @as(usize, 1024),
        tight_limits.archive.max_tensor_output_bytes,
    );
    try std.testing.expectEqual(
        @as(usize, 512),
        tight_limits.archive.max_prefix_bytes,
    );
}

test "CLI parser releases every allocation on injected OOM" {
    try std.testing.checkAllAllocationFailures(
        std.testing.allocator,
        parseForAllocationFailureCheck,
        .{},
    );
}
