//! Streamable .brv container. The reader also accepts legacy back-references.

const std = @import("std");
const types = @import("types.zig");
const safetensors = @import("safetensors.zig");
const program = @import("program.zig");

const Allocator = std.mem.Allocator;
const Dtype = types.Dtype;
const Node = program.Node;

const LEGACY_HEADER: [8]u8 = .{ 'B', 'R', 'V', 3, 5, 0, 0, 0 };
pub const HEADER: [8]u8 = .{ 'B', 'R', 'V', 3, 6, 0, 0, 0 };
const FOOTER_MAGIC: [4]u8 = .{ 'B', 'R', 'V', 'F' };

pub const BlockJob = struct { node: *Node, payload: []const u8 };
pub const TensorMeta = struct { name: []const u8, dtype: Dtype, shape: []const u64, n_blocks: u32 };

pub fn frameHeader(alloc: Allocator, node: Node, payload_len: usize) ![]u8 {
    const bytecode = try program.serialize(alloc, node);
    defer alloc.free(bytecode);
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(alloc);
    try w32(alloc, &out, @intCast(bytecode.len));
    try out.appendSlice(alloc, bytecode);
    try w64(alloc, &out, @intCast(payload_len));
    return out.toOwnedSlice(alloc);
}

/// Group planned blocks under their tensors so a writer or a report can
/// describe the archive footer without re-deriving block ownership.
pub fn tensorMetas(
    alloc: Allocator,
    tensors: []const safetensors.Tensor,
    blocks: []const types.Block,
) ![]TensorMeta {
    const metas = try alloc.alloc(TensorMeta, tensors.len);
    var bi: usize = 0;
    for (tensors, 0..) |t, ti| {
        var n: u32 = 0;
        while (bi + n < blocks.len and blocks[bi + n].tensor_idx == ti) n += 1;
        metas[ti] = .{
            .name = t.name,
            .dtype = t.view.dtype,
            .shape = t.view.shape,
            .n_blocks = n,
        };
        bi += n;
    }
    std.debug.assert(bi == blocks.len);
    return metas;
}

pub fn makeFooter(alloc: Allocator, tensors: []const TensorMeta, index_off: u64, safetensors_prefix: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(alloc);
    try w64(alloc, &out, @intCast(safetensors_prefix.len));
    try out.appendSlice(alloc, safetensors_prefix);
    try w32(alloc, &out, @intCast(tensors.len));
    for (tensors) |tensor| {
        try w16(alloc, &out, @intCast(tensor.name.len));
        try out.appendSlice(alloc, tensor.name);
        try out.append(alloc, @intFromEnum(tensor.dtype));
        try out.append(alloc, @intCast(tensor.shape.len));
        for (tensor.shape) |dim| try w64(alloc, &out, dim);
        try w32(alloc, &out, tensor.n_blocks);
    }
    try out.appendSlice(alloc, &FOOTER_MAGIC);
    try w64(alloc, &out, index_off);
    return out.toOwnedSlice(alloc);
}

pub fn build(alloc: Allocator, tensors: []const TensorMeta, jobs: []const BlockJob, safetensors_prefix: []const u8) ![]u8 {
    var n_blocks: usize = 0;
    for (tensors) |tensor| n_blocks += tensor.n_blocks;
    if (n_blocks != jobs.len) return error.BlockCountMismatch;

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(alloc);
    try out.appendSlice(alloc, &HEADER);

    for (jobs) |job| {
        const header = try frameHeader(alloc, job.node.*, job.payload.len);
        defer alloc.free(header);
        try out.appendSlice(alloc, header);
        try out.appendSlice(alloc, job.payload);
    }

    const index_off = out.items.len;
    const tail = try makeFooter(alloc, tensors, index_off, safetensors_prefix);
    defer alloc.free(tail);
    try out.appendSlice(alloc, tail);
    return out.toOwnedSlice(alloc);
}

fn w16(alloc: Allocator, out: *std.ArrayList(u8), v: u16) !void {
    var b: [2]u8 = undefined;
    std.mem.writeInt(u16, &b, v, .little);
    try out.appendSlice(alloc, &b);
}

fn w32(alloc: Allocator, out: *std.ArrayList(u8), v: u32) !void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    try out.appendSlice(alloc, &b);
}

fn w64(alloc: Allocator, out: *std.ArrayList(u8), v: u64) !void {
    var b: [8]u8 = undefined;
    std.mem.writeInt(u64, &b, v, .little);
    try out.appendSlice(alloc, &b);
}

pub const ParsedBlock = struct {
    bytecode: []const u8,
    payload: []const u8,
};

pub const ParsedTensor = struct {
    name: []const u8,
    dtype: Dtype,
    shape: []const u64,
    frame_start: usize,
    n_blocks: u32,

    fn byteLen(self: ParsedTensor) usize {
        var count: usize = 1;
        for (self.shape) |dim| count *= @intCast(dim);
        return count * self.dtype.elemSize();
    }
};

pub const TensorLengths = struct {
    index: usize = 0,
    blocks: u32 = 0,
    bytes: usize = 0,

    fn skipEmpty(self: *TensorLengths, tensors: []const ParsedTensor) !void {
        while (self.index < tensors.len and tensors[self.index].n_blocks == 0) : (self.index += 1) {
            if (tensors[self.index].byteLen() != 0) return error.ShapeDataMismatch;
        }
    }

    pub fn accept(self: *TensorLengths, tensors: []const ParsedTensor, streams: []const ?types.Stream) !void {
        try self.skipEmpty(tensors);
        for (streams) |maybe| {
            if (self.index >= tensors.len) return error.ShapeDataMismatch;
            const stream = maybe.?;
            self.blocks += 1;
            self.bytes += stream.count * stream.elemBytes();
            if (self.blocks == tensors[self.index].n_blocks) {
                if (self.bytes != tensors[self.index].byteLen()) return error.ShapeDataMismatch;
                self.index += 1;
                self.blocks = 0;
                self.bytes = 0;
                try self.skipEmpty(tensors);
            }
        }
    }

    pub fn finish(self: *TensorLengths, tensors: []const ParsedTensor) !void {
        try self.skipEmpty(tensors);
        if (self.index != tensors.len or self.blocks != 0) return error.ShapeDataMismatch;
    }
};

pub const Parsed = struct {
    tensors: []ParsedTensor,
    frames: []const u8,
    safetensors_prefix: []const u8,
    arena: std.heap.ArenaAllocator,

    pub fn deinit(self: *Parsed) void {
        self.arena.deinit();
    }
};

pub const Loaded = struct {
    parsed: Parsed,
    bytes: []u8,
    mmap: ?std.Io.File.MemoryMap,

    pub fn deinit(self: *Loaded, alloc: Allocator, io: std.Io) void {
        self.parsed.deinit();
        if (self.mmap) |*mapping| mapping.destroy(io) else alloc.free(self.bytes);
    }
};

pub fn nextBlock(frames: []const u8, pos: *usize) !ParsedBlock {
    var at = pos.*;
    var logical_end: ?usize = null;
    while (true) {
        var r: Reader = .{ .b = frames, .pos = at };
        const program_len = std.math.cast(usize, try r.u32v()) orelse return error.Truncated;
        if (program_len == 0) {
            const ref = std.math.cast(usize, try r.u64v()) orelse return error.Truncated;
            if (ref >= at) return error.InvalidProgram;
            if (logical_end == null) logical_end = r.pos;
            at = ref;
            continue;
        }
        const bytecode = try r.take(program_len);
        const payload_len = std.math.cast(usize, try r.u64v()) orelse return error.Truncated;
        const payload = try r.take(payload_len);
        pos.* = logical_end orelse r.pos;
        return .{ .bytecode = bytecode, .payload = payload };
    }
}

pub fn decodeBlock(alloc: Allocator, block: ParsedBlock) !types.Stream {
    var node = try program.deserialize(alloc, block.bytecode);
    defer node.deinit(alloc);
    try program.distributePayload(&node, block.payload);
    return program.decode(alloc, node);
}

const Reader = struct {
    b: []const u8,
    pos: usize = 0,

    fn take(self: *Reader, n: usize) ![]const u8 {
        if (self.pos > self.b.len or n > self.b.len - self.pos) return error.Truncated;
        defer self.pos += n;
        return self.b[self.pos..][0..n];
    }

    fn u8v(self: *Reader) !u8 {
        return (try self.take(1))[0];
    }

    fn u16v(self: *Reader) !u16 {
        return std.mem.readInt(u16, (try self.take(2))[0..2], .little);
    }

    fn u32v(self: *Reader) !u32 {
        return std.mem.readInt(u32, (try self.take(4))[0..4], .little);
    }

    fn u64v(self: *Reader) !u64 {
        return std.mem.readInt(u64, (try self.take(8))[0..8], .little);
    }

    fn checkCount(self: Reader, n: u32, per_elem: usize) !usize {
        if (@as(u64, n) * per_elem > self.b.len - self.pos) return error.Truncated;
        return n;
    }
};

pub fn parse(alloc: Allocator, bytes: []const u8) !Parsed {
    if (bytes.len < HEADER.len + 12) return error.Truncated;
    if (!std.mem.eql(u8, bytes[0..HEADER.len], &HEADER) and
        !std.mem.eql(u8, bytes[0..LEGACY_HEADER.len], &LEGACY_HEADER))
        return error.BadMagic;

    const footer = bytes.len - 12;
    if (!std.mem.eql(u8, bytes[footer..][0..4], &FOOTER_MAGIC)) return error.BadFooter;
    const index_off = std.math.cast(usize, std.mem.readInt(u64, bytes[footer + 4 ..][0..8], .little)) orelse
        return error.Truncated;
    if (index_off < HEADER.len or index_off > footer) return error.Truncated;

    var arena: std.heap.ArenaAllocator = .init(alloc);
    errdefer arena.deinit();
    const a = arena.allocator();
    var r: Reader = .{ .b = bytes[0..footer], .pos = index_off };

    const prefix_len = std.math.cast(usize, try r.u64v()) orelse return error.Truncated;
    const safetensors_prefix = try r.take(prefix_len);
    if (prefix_len > 0 and (prefix_len < 8 or
        std.mem.readInt(u64, safetensors_prefix[0..8], .little) != prefix_len - 8))
        return error.InvalidSafetensorsPrefix;

    const n_tensors = try r.checkCount(try r.u32v(), 8);
    const tensors = try a.alloc(ParsedTensor, n_tensors);
    for (tensors) |*tensor| {
        const name_len = try r.u16v();
        tensor.name = try a.dupe(u8, try r.take(name_len));
        tensor.dtype = std.enums.fromInt(Dtype, try r.u8v()) orelse return error.InvalidDtype;
        const ndim = try r.u8v();
        const shape = try a.alloc(u64, ndim);
        for (shape) |*dim| dim.* = try r.u64v();
        tensor.shape = shape;
        tensor.n_blocks = try r.u32v();
        tensor.frame_start = 0;
    }
    if (r.pos != footer) return error.TrailingIndexData;

    const frames = bytes[HEADER.len..index_off];
    var pos: usize = 0;
    for (tensors) |*tensor| {
        tensor.frame_start = pos;
        for (0..tensor.n_blocks) |_| _ = try nextBlock(frames, &pos);
    }
    if (pos != frames.len) return error.TrailingFrameData;

    return .{ .tensors = tensors, .frames = frames, .safetensors_prefix = safetensors_prefix, .arena = arena };
}

pub fn loadFromPath(alloc: Allocator, io: std.Io, path: []const u8) !Loaded {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const size: usize = @intCast((try file.stat(io)).size);
    if (std.Io.File.MemoryMap.create(io, file, .{
        .len = size,
        .protection = .{ .read = true, .write = false },
    })) |mapped| {
        var mapping = mapped;
        errdefer mapping.destroy(io);
        return .{ .parsed = try parse(alloc, mapping.memory), .bytes = mapping.memory, .mmap = mapping };
    } else |_| {
        const bytes = try alloc.alloc(u8, size);
        errdefer alloc.free(bytes);
        var buffer: [64 * 1024]u8 = undefined;
        var reader = file.reader(io, &buffer);
        try reader.interface.readSliceAll(bytes);
        return .{ .parsed = try parse(alloc, bytes), .bytes = bytes, .mmap = null };
    }
}
