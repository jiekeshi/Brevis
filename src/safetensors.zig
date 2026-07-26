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

/// Resource limits enforced while parsing, before safetensors-specific owned
/// metadata or tensor buffers are allocated. `max_header_bytes` also bounds
/// the generic JSON DOM used by Zig's parser.
pub const Limits = struct {
    max_file_bytes: usize = types.defaultLargeByteLimit,
    max_header_bytes: usize = 64 * 1024 * 1024,
    max_tensors: usize = 1_000_000,
    max_name_bytes: usize = 1024 * 1024,
    max_dimensions: usize = 1024,
    max_tensor_bytes: usize = types.defaultLargeByteLimit,
};

pub const Tensor = struct {
    name: []const u8, // owned by `Loaded`
    view: TensorView, // .data borrows from input; not owned
};

/// Tensor metadata decoded from an exact safetensors prefix (`8 + N`
/// bytes). `name` and `shape` are owned by the containing `HeaderMetadata`.
pub const HeaderTensor = struct {
    name: []u8,
    dtype: Dtype,
    shape: []u64,
    data_offsets: [2]u64,

    pub fn expectedByteLen(self: HeaderTensor) !usize {
        return checkedShapeBytes(self.shape, self.dtype);
    }

    fn deinit(self: *HeaderTensor, alloc: Allocator) void {
        alloc.free(self.name);
        alloc.free(self.shape);
    }
};

/// Owned metadata for a safetensors header. Tensors are ordered by
/// `data_offsets`, matching `Loaded.tensors`, rather than JSON key order.
pub const HeaderMetadata = struct {
    tensors: []HeaderTensor,
    /// Number of data bytes implied by the contiguous tensor ranges.
    data_len: u64,

    pub fn deinit(self: *HeaderMetadata, alloc: Allocator) void {
        for (self.tensors) |*tensor| tensor.deinit(alloc);
        alloc.free(self.tensors);
    }
};

const PendingTensor = struct {
    tensor: Tensor,
    begin: usize,
    end: usize,
    source_order: usize,
};

fn beforeData(_: void, a: PendingTensor, b: PendingTensor) bool {
    if (a.begin != b.begin) return a.begin < b.begin;
    // Empty tensors at a shared boundary must precede a non-empty tensor,
    // otherwise a valid [x,x], [x,y] pair would look like an overlap.
    if (a.end != b.end) return a.end < b.end;
    return a.source_order < b.source_order;
}

fn checkedShapeBytes(shape: []const u64, dtype: Dtype) !usize {
    for (shape) |dim| {
        if (dim == 0) return 0;
    }
    var numel: usize = 1;
    for (shape) |raw_dim| {
        const dim = std.math.cast(usize, raw_dim) orelse
            return error.ShapeOverflow;
        numel = std.math.mul(usize, numel, dim) catch
            return error.ShapeOverflow;
    }
    return std.math.mul(usize, numel, dtype.elemSize()) catch
        return error.ShapeOverflow;
}

fn nonNegativeInteger(value: std.json.Value) !u64 {
    if (value != .integer or value.integer < 0)
        return error.MalformedSafetensors;
    return @intCast(value.integer);
}

fn parseHeaderTensor(
    alloc: Allocator,
    name: []const u8,
    value: std.json.Value,
    limits: Limits,
) !HeaderTensor {
    if (name.len > limits.max_name_bytes)
        return error.NameLimitExceeded;
    if (value != .object) return error.MalformedSafetensors;
    const obj = value.object;

    const dtype_v = obj.get("dtype") orelse
        return error.MalformedSafetensors;
    if (dtype_v != .string) return error.MalformedSafetensors;
    const dtype = Dtype.fromName(dtype_v.string) orelse
        return error.UnsupportedDtype;

    const shape_v = obj.get("shape") orelse
        return error.MalformedSafetensors;
    if (shape_v != .array) return error.MalformedSafetensors;
    if (shape_v.array.items.len > limits.max_dimensions)
        return error.DimensionLimitExceeded;
    const shape_buf = try alloc.alloc(u64, shape_v.array.items.len);
    errdefer alloc.free(shape_buf);
    for (shape_v.array.items, 0..) |item, i| {
        shape_buf[i] = try nonNegativeInteger(item);
    }

    const off_v = obj.get("data_offsets") orelse
        return error.MalformedSafetensors;
    if (off_v != .array or off_v.array.items.len != 2)
        return error.MalformedSafetensors;
    const begin = try nonNegativeInteger(off_v.array.items[0]);
    const end = try nonNegativeInteger(off_v.array.items[1]);
    if (begin > end) return error.InvalidDataOffsets;

    const expected_len = try checkedShapeBytes(shape_buf, dtype);
    if (expected_len > limits.max_tensor_bytes)
        return error.TensorDataLimitExceeded;
    const expected_len_u64 = std.math.cast(u64, expected_len) orelse
        return error.ShapeOverflow;
    if (end - begin != expected_len_u64)
        return error.ShapeDataMismatch;

    const name_buf = try alloc.dupe(u8, name);
    errdefer alloc.free(name_buf);
    return .{
        .name = name_buf,
        .dtype = dtype,
        .shape = shape_buf,
        .data_offsets = .{ begin, end },
    };
}

fn parseTensor(
    alloc: Allocator,
    bytes: []const u8,
    data_start: usize,
    data_len: usize,
    name: []const u8,
    value: std.json.Value,
    source_order: usize,
    limits: Limits,
) !PendingTensor {
    var metadata = try parseHeaderTensor(alloc, name, value, limits);
    errdefer metadata.deinit(alloc);
    const begin_u64 = metadata.data_offsets[0];
    const end_u64 = metadata.data_offsets[1];
    const begin = std.math.cast(usize, begin_u64) orelse
        return error.SafetensorsDataOverflow;
    const end = std.math.cast(usize, end_u64) orelse
        return error.SafetensorsDataOverflow;
    if (end > data_len) return error.SafetensorsDataOverflow;

    const start_abs = std.math.add(usize, data_start, begin) catch
        return error.SafetensorsDataOverflow;
    const end_abs = std.math.add(usize, data_start, end) catch
        return error.SafetensorsDataOverflow;
    if (end_abs > bytes.len) return error.SafetensorsDataOverflow;

    return .{
        .tensor = .{
            .name = metadata.name,
            .view = .{
                .data = bytes[start_abs..end_abs],
                .shape = metadata.shape,
                .dtype = metadata.dtype,
                .owns_data = false,
                .owns_shape = true,
            },
        },
        .begin = begin,
        .end = end,
        .source_order = source_order,
    };
}

fn validateMetadata(value: std.json.Value) !void {
    if (value != .object) return error.MalformedSafetensors;
    var metadata_it = value.object.iterator();
    while (metadata_it.next()) |metadata_entry| {
        if (metadata_entry.value_ptr.* != .string)
            return error.MalformedSafetensors;
    }
}

const PendingHeaderTensor = struct {
    tensor: HeaderTensor,
    source_order: usize,
};

fn beforeHeaderData(_: void, a: PendingHeaderTensor, b: PendingHeaderTensor) bool {
    const a_begin = a.tensor.data_offsets[0];
    const b_begin = b.tensor.data_offsets[0];
    if (a_begin != b_begin) return a_begin < b_begin;
    const a_end = a.tensor.data_offsets[1];
    const b_end = b.tensor.data_offsets[1];
    if (a_end != b_end) return a_end < b_end;
    return a.source_order < b.source_order;
}

pub const Loaded = struct {
    /// The full file bytes. May be either an allocator-owned buffer (the
    /// fallback `read()` path) or borrow into a `MemoryMap` (the mmap path).
    bytes: []const u8,
    tensors: []Tensor, // owned
    /// Set when `bytes` is mmap'd. Caller must keep `io` alive long enough
    /// to call `deinit` with the same `io` used to load.
    mmap: ?std.Io.File.MemoryMap = null,
    /// Captured by `loadFromPath` so the ordinary `deinit` is sufficient for
    /// both allocator-owned buffers and successful mmap loads.
    mmap_io: ?std.Io = null,

    pub fn deinit(self: *Loaded, alloc: Allocator) void {
        for (self.tensors) |*t| {
            alloc.free(t.view.shape);
            alloc.free(t.name);
        }
        alloc.free(self.tensors);
        if (self.mmap) |*mapping| {
            mapping.destroy(self.mmap_io orelse
                @panic("mmap-backed Loaded is missing its io"));
        } else {
            alloc.free(@constCast(self.bytes));
        }
        self.* = undefined;
    }

    /// Compatibility helper for callers that manually attach an mmap after
    /// `loadFromBytes`. Values returned by `loadFromPath` only need `deinit`.
    pub fn deinitMmap(self: *Loaded, alloc: Allocator, io: std.Io) void {
        if (self.mmap != null and self.mmap_io == null)
            self.mmap_io = io;
        self.deinit(alloc);
    }
};

/// Parse and validate an exact safetensors prefix: the 8-byte little-endian
/// header length followed by exactly that many JSON bytes, with no tensor
/// payload. This is useful to validate a streamed reconstruction before its
/// data bytes have been materialized.
pub fn parsePrefixWithLimits(
    alloc: Allocator,
    prefix: []const u8,
    limits: Limits,
) !HeaderMetadata {
    if (prefix.len > limits.max_file_bytes)
        return error.SafetensorsFileLimitExceeded;
    if (prefix.len < 8) return error.SafetensorsTooShort;
    const header_len_u64 = std.mem.readInt(u64, prefix[0..8], .little);
    const header_len = std.math.cast(usize, header_len_u64) orelse
        return error.SafetensorsHeaderOverflow;
    const prefix_len = std.math.add(usize, 8, header_len) catch
        return error.SafetensorsHeaderOverflow;
    if (header_len > limits.max_header_bytes)
        return error.SafetensorsHeaderLimitExceeded;
    if (prefix_len != prefix.len)
        return error.InvalidSafetensorsPrefix;

    const parsed = try std.json.parseFromSlice(
        std.json.Value,
        alloc,
        prefix[8..prefix_len],
        .{},
    );
    defer parsed.deinit();
    const root = parsed.value;
    if (root != .object) return error.MalformedSafetensors;

    var pending: std.ArrayList(PendingHeaderTensor) = .empty;
    defer pending.deinit(alloc);
    errdefer for (pending.items) |*item| item.tensor.deinit(alloc);

    var it = root.object.iterator();
    var source_order: usize = 0;
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        const value = entry.value_ptr.*;
        if (std.mem.eql(u8, name, "__metadata__")) {
            try validateMetadata(value);
            continue;
        }
        if (source_order >= limits.max_tensors)
            return error.TensorLimitExceeded;
        var tensor = try parseHeaderTensor(alloc, name, value, limits);
        errdefer tensor.deinit(alloc);
        try pending.append(alloc, .{
            .tensor = tensor,
            .source_order = source_order,
        });
        source_order += 1;
    }

    std.mem.sort(PendingHeaderTensor, pending.items, {}, beforeHeaderData);
    var cursor: u64 = 0;
    for (pending.items) |item| {
        if (item.tensor.data_offsets[0] != cursor)
            return error.NonContiguousTensorData;
        cursor = item.tensor.data_offsets[1];
    }

    const tensors = try alloc.alloc(HeaderTensor, pending.items.len);
    errdefer alloc.free(tensors);
    for (pending.items, tensors) |item, *tensor| tensor.* = item.tensor;
    return .{ .tensors = tensors, .data_len = cursor };
}

pub fn parsePrefix(alloc: Allocator, prefix: []const u8) !HeaderMetadata {
    return parsePrefixWithLimits(alloc, prefix, .{});
}

/// Descriptive alias for callers that treat the prefix as header metadata.
pub const parseHeaderMetadata = parsePrefix;

pub fn validatePrefix(alloc: Allocator, prefix: []const u8) !void {
    var metadata = try parsePrefix(alloc, prefix);
    metadata.deinit(alloc);
}

/// Parse `bytes` and transfer ownership of the complete buffer to the
/// returned `Loaded` on success. On error, ownership remains with the caller.
pub fn loadFromBytesWithLimits(
    alloc: Allocator,
    bytes: []u8,
    limits: Limits,
) !Loaded {
    if (bytes.len > limits.max_file_bytes)
        return error.SafetensorsFileLimitExceeded;
    if (bytes.len < 8) return error.SafetensorsTooShort;
    const header_len_u64 = std.mem.readInt(u64, bytes[0..8], .little);
    const header_len = std.math.cast(usize, header_len_u64) orelse
        return error.SafetensorsHeaderOverflow;
    const data_start = std.math.add(usize, 8, header_len) catch
        return error.SafetensorsHeaderOverflow;
    if (header_len > limits.max_header_bytes)
        return error.SafetensorsHeaderLimitExceeded;
    if (data_start > bytes.len) return error.SafetensorsHeaderOverflow;

    const header_json = bytes[8..data_start];
    const data_len = bytes.len - data_start;

    const parsed = try std.json.parseFromSlice(std.json.Value, alloc, header_json, .{});
    defer parsed.deinit();
    const root = parsed.value;
    if (root != .object) return error.MalformedSafetensors;

    var tensor_list: std.ArrayList(PendingTensor) = .empty;
    defer tensor_list.deinit(alloc);
    errdefer for (tensor_list.items) |tensor| {
        alloc.free(tensor.tensor.view.shape);
        alloc.free(tensor.tensor.name);
    };

    var it = root.object.iterator();
    var source_order: usize = 0;
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        const value = entry.value_ptr.*;
        if (std.mem.eql(u8, name, "__metadata__")) {
            try validateMetadata(value);
            continue;
        }
        if (source_order >= limits.max_tensors)
            return error.TensorLimitExceeded;

        const tensor = try parseTensor(
            alloc,
            bytes,
            data_start,
            data_len,
            name,
            value,
            source_order,
            limits,
        );
        errdefer {
            alloc.free(tensor.tensor.view.shape);
            alloc.free(tensor.tensor.name);
        }
        try tensor_list.append(alloc, tensor);
        source_order += 1;
    }

    std.mem.sort(PendingTensor, tensor_list.items, {}, beforeData);
    var cursor: usize = 0;
    for (tensor_list.items) |tensor| {
        if (tensor.begin != cursor)
            return error.NonContiguousTensorData;
        cursor = tensor.end;
    }
    if (cursor != data_len) return error.UnclaimedTensorData;

    const tensors = try alloc.alloc(Tensor, tensor_list.items.len);
    errdefer alloc.free(tensors);
    for (tensor_list.items, tensors) |pending, *tensor| {
        tensor.* = pending.tensor;
    }
    return .{ .bytes = bytes, .tensors = tensors };
}

pub fn loadFromBytes(alloc: Allocator, bytes: []u8) !Loaded {
    return loadFromBytesWithLimits(alloc, bytes, .{});
}

pub fn loadFromPathWithLimits(
    alloc: Allocator,
    io: std.Io,
    path: []const u8,
    limits: Limits,
) !Loaded {
    const cwd = std.Io.Dir.cwd();
    const f = try cwd.openFile(io, path, .{});
    defer f.close(io);
    const stat = try f.stat(io);
    const file_len = std.math.cast(usize, stat.size) orelse
        return error.SafetensorsFileTooLarge;
    if (file_len > limits.max_file_bytes)
        return error.SafetensorsFileLimitExceeded;

    // Fast path: mmap the file read-only. Saves the ~600 ms of explicit
    // memcpy that read() into a 2 GB buffer would cost. Bytes are
    // demand-paged on first access (which we'd pay either way during
    // sequential scan).
    if (std.Io.File.MemoryMap.create(io, f, .{
        .len = file_len,
        .protection = .{ .read = true, .write = false },
        .populate = false,
    })) |mm| {
        var owned_map = mm;
        errdefer owned_map.destroy(io);
        var loaded = try loadFromBytesWithLimits(
            alloc,
            owned_map.memory,
            limits,
        );
        loaded.mmap = owned_map;
        loaded.mmap_io = io;
        return loaded;
    } else |_| {
        // Fallback to chunked read() — used when the IO backend doesn't
        // implement mmap, or on filesystems that don't support it.
        const buf = try alloc.alloc(u8, file_len);
        errdefer alloc.free(buf);
        var rb: [4096]u8 = undefined;
        var rdr = f.reader(io, &rb);
        const chunk: usize = 1 << 30;
        var off: usize = 0;
        while (off < buf.len) {
            const n = @min(chunk, buf.len - off);
            try rdr.interface.readSliceAll(buf[off .. off + n]);
            off += n;
        }
        return loadFromBytesWithLimits(alloc, buf, limits);
    }
}

pub fn loadFromPath(alloc: Allocator, io: std.Io, path: []const u8) !Loaded {
    return loadFromPathWithLimits(alloc, io, path, .{});
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

fn appendDecimal(out: *std.ArrayList(u8), alloc: Allocator, value: anytype) !void {
    var buf: [32]u8 = undefined;
    const text = try std.fmt.bufPrint(&buf, "{d}", .{value});
    try out.appendSlice(alloc, text);
}

fn appendJsonString(out: *std.ArrayList(u8), alloc: Allocator, value: []const u8) !void {
    if (!std.unicode.utf8ValidateSlice(value))
        return error.InvalidTensorName;
    const hex = "0123456789abcdef";
    try out.append(alloc, '"');
    for (value) |byte| {
        switch (byte) {
            '"' => try out.appendSlice(alloc, "\\\""),
            '\\' => try out.appendSlice(alloc, "\\\\"),
            0...0x1f => {
                try out.appendSlice(alloc, "\\u00");
                try out.append(alloc, hex[byte >> 4]);
                try out.append(alloc, hex[byte & 0xf]);
            },
            else => try out.append(alloc, byte),
        }
    }
    try out.append(alloc, '"');
}

pub fn buildHeader(alloc: Allocator, tensors: []const TensorMeta) ![]u8 {
    var offset: u64 = 0;
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(alloc);

    try out.append(alloc, '{');
    for (tensors, 0..) |tensor, i| {
        if (std.mem.eql(u8, tensor.name, "__metadata__"))
            return error.ReservedTensorName;
        for (tensors[0..i]) |prior| {
            if (std.mem.eql(u8, tensor.name, prior.name))
                return error.DuplicateTensorName;
        }
        const expected_len = try checkedShapeBytes(tensor.shape, tensor.dtype);
        if (expected_len != tensor.byte_len)
            return error.ShapeDataMismatch;
        const byte_len_u64 = std.math.cast(u64, tensor.byte_len) orelse
            return error.OffsetOverflow;
        const next_offset = std.math.add(u64, offset, byte_len_u64) catch
            return error.OffsetOverflow;

        if (i > 0) try out.append(alloc, ',');
        try appendJsonString(&out, alloc, tensor.name);
        try out.appendSlice(alloc, ":{\"dtype\":\"");
        try out.appendSlice(alloc, tensor.dtype.name());
        try out.appendSlice(alloc, "\",\"shape\":[");
        for (tensor.shape, 0..) |dim, k| {
            if (k > 0) try out.append(alloc, ',');
            try appendDecimal(&out, alloc, dim);
        }
        try out.appendSlice(alloc, "],\"data_offsets\":[");
        try appendDecimal(&out, alloc, offset);
        try out.append(alloc, ',');
        try appendDecimal(&out, alloc, next_offset);
        try out.appendSlice(alloc, "]}");
        offset = next_offset;
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

    var atomic = try std.Io.Dir.cwd().createFileAtomic(
        io,
        path,
        .{ .replace = true },
    );
    defer atomic.deinit(io);
    var wb: [4096]u8 = undefined;
    var wf = atomic.file.writer(io, &wb);

    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header.len, .little);
    try wf.interface.writeAll(&len_buf);
    try wf.interface.writeAll(header);
    for (tensors) |t| try wf.interface.writeAll(t.view.data);
    try wf.interface.flush();
    try atomic.file.sync(io);
    try atomic.replace(io);
}
