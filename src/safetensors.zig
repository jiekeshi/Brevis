//! Minimal safetensors reader/writer.
//!
//! File format:
//!   [8]  N (u64 LE) — header length in bytes
//!   [N]  UTF-8 JSON header
//!   […]  tensor data, in the offsets defined by the header
//!
//! Header schema (relevant subset):
//!   {
//!     "tensor_name": {
//!       "dtype": "F16" | "BF16" | "F32" | "U8" | "U16" | "U32" | "I8" | "I16" | "I32",
//!       "shape": [d1, d2, ...],
//!       "data_offsets": [start, end]
//!     },
//!     "__metadata__": { ... }
//!   }

const std = @import("std");
const types = @import("types.zig");

const Allocator = types.Allocator;
const TensorView = types.TensorView;
const Dtype = types.Dtype;

pub const Tensor = struct {
    name: []const u8, // borrowed from input bytes; copy if you need ownership
    view: TensorView, // .data borrows from input; not owned
};

fn beforeData(_: void, a: Tensor, b: Tensor) bool {
    return @intFromPtr(a.view.data.ptr) < @intFromPtr(b.view.data.ptr);
}

pub const Loaded = struct {
    /// The full file bytes. May be either an allocator-owned buffer (the
    /// fallback `read()` path) or borrow into a `MemoryMap` (the mmap path).
    bytes: []u8,
    tensors: []Tensor, // owned
    /// Set when `bytes` is mmap'd. Caller must keep `io` alive long enough
    /// to call `deinit` with the same `io` used to load.
    mmap: ?std.Io.File.MemoryMap = null,

    pub fn deinit(self: *Loaded, alloc: Allocator) void {
        for (self.tensors) |*t| {
            alloc.free(t.view.shape);
            alloc.free(t.name);
        }
        alloc.free(self.tensors);
        if (self.mmap == null) alloc.free(self.bytes);
        // `bytes` borrow into `mmap.memory`; we just need an io to destroy
        // the map. Caller is responsible: see `deinitMmap`.
    }

    /// Use this variant if the file was loaded via mmap. Required to release
    /// the kernel mapping (the regular `deinit` doesn't have access to `io`).
    pub fn deinitMmap(self: *Loaded, alloc: Allocator, io: std.Io) void {
        for (self.tensors) |*t| {
            alloc.free(t.view.shape);
            alloc.free(t.name);
        }
        alloc.free(self.tensors);
        if (self.mmap) |*mm| {
            mm.destroy(io);
        } else {
            alloc.free(self.bytes);
        }
    }
};

pub fn loadFromBytes(alloc: Allocator, bytes: []u8) !Loaded {
    if (bytes.len < 8) return error.SafetensorsTooShort;
    const header_len = std.mem.readInt(u64, bytes[0..8], .little);
    if (8 + header_len > bytes.len) return error.SafetensorsHeaderOverflow;

    const header_json = bytes[8 .. 8 + @as(usize, @intCast(header_len))];
    const data_start: usize = 8 + @as(usize, @intCast(header_len));

    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header_json, .{});
    defer parsed.deinit();
    const root = parsed.value;
    if (root != .object) return error.MalformedSafetensors;

    var tensor_list: std.ArrayList(Tensor) = .empty;
    defer tensor_list.deinit(alloc);

    var it = root.object.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        if (std.mem.eql(u8, name, "__metadata__")) continue;
        const obj = entry.value_ptr.*;
        if (obj != .object) continue;

        const dtype_v = obj.object.get("dtype") orelse return error.MalformedSafetensors;
        if (dtype_v != .string) return error.MalformedSafetensors;
        const dtype = Dtype.fromName(dtype_v.string) orelse return error.UnsupportedDtype;

        const shape_v = obj.object.get("shape") orelse return error.MalformedSafetensors;
        if (shape_v != .array) return error.MalformedSafetensors;
        const shape_buf = try alloc.alloc(u64, shape_v.array.items.len);
        for (shape_v.array.items, 0..) |item, i| {
            if (item != .integer) return error.MalformedSafetensors;
            shape_buf[i] = @intCast(item.integer);
        }

        const off_v = obj.object.get("data_offsets") orelse return error.MalformedSafetensors;
        if (off_v != .array or off_v.array.items.len != 2) return error.MalformedSafetensors;
        const begin: u64 = @intCast(off_v.array.items[0].integer);
        const end: u64 = @intCast(off_v.array.items[1].integer);

        const start_abs = data_start + @as(usize, @intCast(begin));
        const end_abs = data_start + @as(usize, @intCast(end));
        if (end_abs > bytes.len) return error.SafetensorsDataOverflow;

        const name_buf = try alloc.alloc(u8, name.len);
        @memcpy(name_buf, name);

        try tensor_list.append(alloc, .{
            .name = name_buf,
            .view = .{
                .data = bytes[start_abs..end_abs],
                .shape = shape_buf,
                .dtype = dtype,
                .owns_data = false,
                .owns_shape = true,
            },
        });
    }

    std.mem.sort(Tensor, tensor_list.items, {}, beforeData);
    return .{ .bytes = bytes, .tensors = try tensor_list.toOwnedSlice(alloc) };
}

pub fn loadFromPath(alloc: Allocator, io: std.Io, path: []const u8) !Loaded {
    const cwd = std.Io.Dir.cwd();
    const f = try cwd.openFile(io, path, .{});
    const stat = try f.stat(io);

    // Fast path: mmap the file read-only. Saves the ~600 ms of explicit
    // memcpy that read() into a 2 GB buffer would cost. Bytes are
    // demand-paged on first access (which we'd pay either way during
    // sequential scan).
    if (std.Io.File.MemoryMap.create(io, f, .{
        .len = @intCast(stat.size),
        .protection = .{ .read = true, .write = false },
        .populate = false,
    })) |mm| {
        // Note: we keep the file fd open for the life of the mapping. macOS
        // and Linux both allow closing the fd without invalidating the mmap,
        // but we err on the safe side here — the cost is one fd until
        // `deinitMmap`.
        var loaded = try loadFromBytes(alloc, mm.memory);
        loaded.mmap = mm;
        f.close(io);
        return loaded;
    } else |_| {
        // Fallback to chunked read() — used when the IO backend doesn't
        // implement mmap, or on filesystems that don't support it.
        defer f.close(io);
        const buf = try alloc.alloc(u8, @intCast(stat.size));
        var rb: [4096]u8 = undefined;
        var rdr = f.reader(io, &rb);
        const chunk: usize = 1 << 30;
        var off: usize = 0;
        while (off < buf.len) {
            const n = @min(chunk, buf.len - off);
            try rdr.interface.readSliceAll(buf[off .. off + n]);
            off += n;
        }
        return loadFromBytes(alloc, buf);
    }
}

pub const TensorOut = struct {
    name: []const u8,
    view: TensorView,
};

pub const TensorMeta = struct {
    name: []const u8,
    dtype: Dtype,
    shape: []const u64,
    byte_len: usize,
};

pub fn buildHeader(alloc: Allocator, tensors: []const TensorMeta) ![]u8 {
    var offset: u64 = 0;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(alloc);

    try out.append(alloc, '{');
    for (tensors, 0..) |tensor, i| {
        if (i > 0) try out.append(alloc, ',');
        try out.append(alloc, '"');
        try out.appendSlice(alloc, tensor.name);
        try out.appendSlice(alloc, "\":{\"dtype\":\"");
        try out.appendSlice(alloc, tensor.dtype.name());
        try out.appendSlice(alloc, "\",\"shape\":[");
        for (tensor.shape, 0..) |dim, k| {
            if (k > 0) try out.append(alloc, ',');
            const value = try std.fmt.allocPrint(alloc, "{d}", .{dim});
            defer alloc.free(value);
            try out.appendSlice(alloc, value);
        }
        try out.appendSlice(alloc, "],\"data_offsets\":[");
        const offsets = try std.fmt.allocPrint(alloc, "{d},{d}", .{ offset, offset + tensor.byte_len });
        defer alloc.free(offsets);
        try out.appendSlice(alloc, offsets);
        try out.appendSlice(alloc, "]}");
        offset += tensor.byte_len;
    }
    try out.append(alloc, '}');
    return out.toOwnedSlice(alloc);
}

/// Write tensors to a safetensors file at `path`. Tensor data is written in
/// the order given.
pub fn saveToPath(alloc: Allocator, io: std.Io, path: []const u8, tensors: []const TensorOut) !void {
    const metas = try alloc.alloc(TensorMeta, tensors.len);
    defer alloc.free(metas);
    for (tensors, metas) |tensor, *meta| meta.* = .{
        .name = tensor.name,
        .dtype = tensor.view.dtype,
        .shape = tensor.view.shape,
        .byte_len = tensor.view.data.len,
    };
    const header = try buildHeader(alloc, metas);
    defer alloc.free(header);

    const cwd = std.Io.Dir.cwd();
    const f = try cwd.createFile(io, path, .{});
    defer f.close(io);
    var wb: [4096]u8 = undefined;
    var wf = f.writer(io, &wb);

    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header.len, .little);
    try wf.interface.writeAll(&len_buf);
    try wf.interface.writeAll(header);
    for (tensors) |t| try wf.interface.writeAll(t.view.data);
    try wf.interface.flush();
}
