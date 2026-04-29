//! Baseline lossless compressors for comparison against brevis.
//!
//! These run on the *raw tensor bytes* of a safetensors file (so headers are
//! ignored), giving a fair "what would a generic byte compressor do?"
//! comparison.
//!
//! Two backends:
//!   * gzip (DEFLATE level 9) — in-process via std.compress.flate
//!   * zstd                   — shells out to `zstd` if installed
//!
//! Both return only the compressed byte count — we never need to decompress
//! since we trust upstream tools.

const std = @import("std");

pub const BaselineErr = error{
    OutOfMemory,
    ZstdNotInstalled,
    ZstdFailed,
    TempFileFailed,
} || std.process.RunError || std.Io.File.OpenError || std.Io.File.WriteError;

/// In-process gzip (DEFLATE level 9). Returns compressed byte count.
pub fn gzipSize(alloc: std.mem.Allocator, input: []const u8) !usize {
    const initial_cap: usize = @max(@as(usize, 4096), input.len / 8);
    var out: std.Io.Writer.Allocating = try .initCapacity(alloc, initial_cap);
    defer out.deinit();

    var work: [std.compress.flate.max_window_len]u8 = undefined;
    var c = try std.compress.flate.Compress.init(&out.writer, &work, .gzip, .level_9);
    try c.writer.writeAll(input);
    try c.finish();
    return out.writer.end;
}

/// zstd at the given level. Returns compressed byte count, or null if zstd is
/// not installed.
///
/// Implementation: write input to a temp file, run `zstd -<level> --stdout
/// <file>`, count returned bytes.
pub fn zstdSize(alloc: std.mem.Allocator, io: std.Io, input: []const u8, level: u8) !?usize {
    // Probe for zstd by trying `zstd --version`. If it's missing, return null.
    const probe = std.process.run(alloc, io, .{
        .argv = &.{ "zstd", "--version" },
    }) catch return null;
    defer alloc.free(probe.stdout);
    defer alloc.free(probe.stderr);

    // Write input to a temp file.
    const cwd = std.Io.Dir.cwd();
    // Single-process serial use, so a fixed name is fine.
    const tmp_path = try alloc.dupe(u8, "/tmp/brevis-baseline.tmp");
    defer alloc.free(tmp_path);

    const f = try cwd.createFile(io, tmp_path, .{});
    {
        defer f.close(io);
        var wb: [4096]u8 = undefined;
        var wf = f.writer(io, &wb);
        try wf.interface.writeAll(input);
        try wf.interface.flush();
    }
    defer cwd.deleteFile(io, tmp_path) catch {};

    var lvl_buf: [8]u8 = undefined;
    const lvl_arg = try std.fmt.bufPrint(&lvl_buf, "-{d}", .{level});

    const r = try std.process.run(alloc, io, .{
        .argv = &.{ "zstd", lvl_arg, "--stdout", "-q", tmp_path },
        .stdout_limit = .unlimited,
    });
    defer alloc.free(r.stderr);
    defer alloc.free(r.stdout);

    return r.stdout.len;
}
