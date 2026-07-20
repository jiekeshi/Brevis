//! Streamable .brv container: payloads first, a compact dictionary/index footer
//! last. Program bytecode and entropy tables are content-addressed.

const std = @import("std");
const types = @import("types.zig");
const codec = @import("codec.zig");
const ops = @import("ops.zig");
const program = @import("program.zig");

const Allocator = std.mem.Allocator;
const Dtype = types.Dtype;
const Node = program.Node;

pub const HEADER: [8]u8 = .{ 'B', 'R', 'V', 3, 4, 0, 0, 0 };
const FOOTER_MAGIC: [4]u8 = .{ 'B', 'R', 'V', 'F' };

pub const TableKind = enum(u8) { huffman = 0, rans = 1 };

pub const BlockJob = struct { node: *Node, payload: []const u8 };
pub const TensorMeta = struct { name: []const u8, dtype: Dtype, shape: []const u64, n_blocks: u32 };

// ==================== build ====================

const Interner = struct {
    ids: std.AutoHashMapUnmanaged(u64, u32) = .empty,
    items: std.ArrayList([]u8) = .empty,

    fn deinit(self: *Interner, alloc: Allocator) void {
        self.ids.deinit(alloc);
        for (self.items.items) |b| alloc.free(b);
        self.items.deinit(alloc);
    }

    fn intern(self: *Interner, alloc: Allocator, bytes: []const u8) !u32 {
        const gop = try self.ids.getOrPut(alloc, std.hash.XxHash64.hash(0, bytes));
        if (gop.found_existing and std.mem.eql(u8, self.items.items[gop.value_ptr.*], bytes)) {
            return gop.value_ptr.*;
        }
        const id: u32 = @intCast(self.items.items.len);
        try self.items.append(alloc, try alloc.dupe(u8, bytes));
        if (!gop.found_existing) gop.value_ptr.* = id;
        return id;
    }
};

fn collectTableSides(alloc: Allocator, node: *Node, out: *std.ArrayList(*ops.SideInfo)) Allocator.Error!void {
    switch (node.side) {
        .huffman, .rans => try out.append(alloc, &node.side),
        else => {},
    }
    for (node.children) |*c| try collectTableSides(alloc, c, out);
}

fn encodeTable(alloc: Allocator, out: *std.ArrayList(u8), side: ops.SideInfo) !void {
    switch (side) {
        .huffman => |h| {
            try out.append(alloc, @intFromEnum(TableKind.huffman));
            try w32(alloc, out, @intCast(h.table.entries.len));
            for (h.table.entries) |e| {
                try w32(alloc, out, e.sym);
                try out.append(alloc, e.len);
            }
        },
        .rans => |r| {
            try out.append(alloc, @intFromEnum(TableKind.rans));
            try w32(alloc, out, @intCast(r.table.symbols.len));
            for (r.table.symbols, r.table.info) |s, inf| {
                try w32(alloc, out, s);
                try w32(alloc, out, inf.freq);
            }
        },
        else => unreachable,
    }
}

fn strippedSide(side: ops.SideInfo) ops.SideInfo {
    return switch (side) {
        .huffman => |h| .{ .huffman = .{
            .table = .{ .entries = &.{} },
            .count = h.count,
            .bits_per_elem = h.bits_per_elem,
        } },
        .rans => |r| .{ .rans = .{
            .table = .{ .symbols = &.{}, .info = &.{} },
            .count = r.count,
            .bits_per_elem = r.bits_per_elem,
        } },
        else => unreachable,
    };
}

const BlockRec = struct { program_id: u32, refs: []u32, payload_off: u64, payload_len: u64 };

pub const Builder = struct {
    alloc: Allocator,
    tables: Interner = .{},
    progs: Interner = .{},
    recs: std.ArrayList(BlockRec) = .empty,

    pub fn init(alloc: Allocator) Builder {
        return .{ .alloc = alloc };
    }

    pub fn deinit(self: *Builder) void {
        self.tables.deinit(self.alloc);
        self.progs.deinit(self.alloc);
        for (self.recs.items) |rec| self.alloc.free(rec.refs);
        self.recs.deinit(self.alloc);
    }

    pub fn add(self: *Builder, job: BlockJob, payload_off: u64) !void {
        const alloc = self.alloc;
        var sides: std.ArrayList(*ops.SideInfo) = .empty;
        defer sides.deinit(alloc);
        try collectTableSides(alloc, job.node, &sides);

        const refs = try alloc.alloc(u32, sides.items.len);
        errdefer alloc.free(refs);
        const saved = try alloc.alloc(ops.SideInfo, sides.items.len);
        defer alloc.free(saved);
        var scratch: std.ArrayList(u8) = .empty;
        defer scratch.deinit(alloc);

        for (sides.items, saved) |sp, *side| side.* = sp.*;
        defer {
            for (sides.items, saved) |sp, side| sp.* = side;
        }
        for (sides.items, 0..) |sp, i| {
            try encodeTable(alloc, &scratch, sp.*);
            refs[i] = try self.tables.intern(alloc, scratch.items);
            scratch.clearRetainingCapacity();
            sp.* = strippedSide(sp.*);
        }

        const bytecode = try program.serialize(alloc, job.node.*);
        defer alloc.free(bytecode);
        try self.recs.append(alloc, .{
            .program_id = try self.progs.intern(alloc, bytecode),
            .refs = refs,
            .payload_off = payload_off,
            .payload_len = job.payload.len,
        });
    }

    pub fn finish(self: *Builder, tensors: []const TensorMeta, index_off: u64, safetensors_prefix: []const u8) ![]u8 {
        const alloc = self.alloc;
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(alloc);

        try w64(alloc, &out, @intCast(safetensors_prefix.len));
        try out.appendSlice(alloc, safetensors_prefix);

        try w32(alloc, &out, @intCast(self.tables.items.items.len));
        for (self.tables.items.items) |table| try out.appendSlice(alloc, table);

        try w32(alloc, &out, @intCast(self.progs.items.items.len));
        for (self.progs.items.items) |prog| {
            try w32(alloc, &out, @intCast(prog.len));
            try out.appendSlice(alloc, prog);
        }

        try w32(alloc, &out, @intCast(tensors.len));
        var rec_i: usize = 0;
        for (tensors) |tensor| {
            try w16(alloc, &out, @intCast(tensor.name.len));
            try out.appendSlice(alloc, tensor.name);
            try out.append(alloc, @intFromEnum(tensor.dtype));
            try out.append(alloc, @intCast(tensor.shape.len));
            for (tensor.shape) |dim| try w64(alloc, &out, dim);
            try w32(alloc, &out, tensor.n_blocks);

            for (0..tensor.n_blocks) |_| {
                const rec = self.recs.items[rec_i];
                try w32(alloc, &out, rec.program_id);
                try w32(alloc, &out, @intCast(rec.refs.len));
                for (rec.refs) |id| try w32(alloc, &out, id);
                try w64(alloc, &out, rec.payload_off);
                try w64(alloc, &out, rec.payload_len);
                rec_i += 1;
            }
        }
        std.debug.assert(rec_i == self.recs.items.len);
        try out.appendSlice(alloc, &FOOTER_MAGIC);
        try w64(alloc, &out, index_off);
        return out.toOwnedSlice(alloc);
    }
};

pub fn build(alloc: Allocator, tensors: []const TensorMeta, jobs: []const BlockJob, safetensors_prefix: []const u8) ![]u8 {
    var builder = Builder.init(alloc);
    defer builder.deinit();
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(alloc);
    try out.appendSlice(alloc, &HEADER);
    for (jobs) |job| {
        try builder.add(job, out.items.len);
        try out.appendSlice(alloc, job.payload);
    }
    const tail = try builder.finish(tensors, out.items.len, safetensors_prefix);
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

// ==================== parse ====================

pub const ParsedBlock = struct {
    node: *Node,
    tables: []const ParsedTable,
    refs: []const u32,
    payload: []const u8,
};

pub const ParsedTensor = struct { name: []u8, dtype: Dtype, shape: []u64, blocks: []ParsedBlock };

pub const Parsed = struct {
    tensors: []ParsedTensor,
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

const ParsedTable = union(TableKind) {
    huffman: codec.HuffmanTable,
    rans: codec.RansTable,
};

pub fn decodeBlock(alloc: Allocator, block: ParsedBlock) !types.Stream {
    var node = try cloneBorrowed(alloc, block.node.*);
    defer deinitBorrowed(alloc, &node);
    try attachTables(&node, block.tables, block.refs);
    try program.distributePayload(&node, block.payload);
    return program.decode(alloc, node);
}

fn cloneBorrowed(alloc: Allocator, node: Node) Allocator.Error!Node {
    const children: []Node = if (node.children.len == 0) &.{} else try alloc.alloc(Node, node.children.len);
    var filled: usize = 0;
    errdefer {
        for (children[0..filled]) |*child| deinitBorrowed(alloc, child);
        if (children.len > 0) alloc.free(children);
    }
    for (node.children, 0..) |child, i| {
        children[i] = try cloneBorrowed(alloc, child);
        filled += 1;
    }
    return .{
        .op = node.op,
        .params = node.params,
        .children = children,
        .side = node.side,
    };
}

fn deinitBorrowed(alloc: Allocator, node: *Node) void {
    for (node.children) |*child| deinitBorrowed(alloc, child);
    if (node.children.len > 0) alloc.free(node.children);
}

const Reader = struct {
    b: []const u8,
    pos: usize = 0,

    fn take(self: *Reader, n: usize) ![]const u8 {
        if (n > self.b.len - self.pos) return error.Truncated;
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
    /// Reject a count whose minimum encoding cannot fit in what remains.
    fn checkCount(self: Reader, n: u32, per_elem: usize) !usize {
        if (@as(u64, n) * per_elem > self.b.len - self.pos) return error.Truncated;
        return n;
    }
};

pub fn parse(alloc: Allocator, bytes: []const u8) !Parsed {
    if (bytes.len < HEADER.len + 12) return error.Truncated;
    if (!std.mem.eql(u8, bytes[0..HEADER.len], &HEADER)) return error.BadMagic;
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

    const n_tables = try r.checkCount(try r.u32v(), 5);
    const tables = try a.alloc(ParsedTable, n_tables);
    for (tables) |*t| {
        const kind = std.enums.fromInt(TableKind, try r.u8v()) orelse return error.InvalidTableKind;
        switch (kind) {
            .huffman => {
                const n = try r.checkCount(try r.u32v(), 5);
                const entries = try a.alloc(codec.HuffmanTable.Entry, n);
                for (entries) |*e| {
                    e.sym = try r.u32v();
                    e.len = try r.u8v();
                }
                t.* = .{ .huffman = .{ .entries = entries } };
            },
            .rans => {
                const n = try r.checkCount(try r.u32v(), 8);
                const symbols = try a.alloc(u32, n);
                const info = try a.alloc(codec.RansSymbol, n);
                // Frequencies must tile [0, RANS_PROB_SCALE) exactly, or decoding
                // would index the cum→symbol map out of range.
                var cum: u64 = 0;
                for (symbols, info) |*s, *inf| {
                    s.* = try r.u32v();
                    inf.freq = try r.u32v();
                    if (inf.freq == 0) return error.InvalidRansTable;
                    inf.cum = @intCast(cum);
                    cum += inf.freq;
                    if (cum > codec.RANS_PROB_SCALE) return error.InvalidRansTable;
                }
                if (n > 0 and cum != codec.RANS_PROB_SCALE) return error.InvalidRansTable;
                t.* = .{ .rans = .{ .symbols = symbols, .info = info } };
            },
        }
    }

    const n_programs = try r.checkCount(try r.u32v(), 4);
    const programs = try a.alloc(Node, n_programs);
    for (programs) |*p| p.* = try program.deserialize(a, try r.take(try r.u32v()));

    const n_tensors = try r.checkCount(try r.u32v(), 8);
    const out_tensors = try a.alloc(ParsedTensor, n_tensors);
    for (out_tensors) |*t| {
        const name_len = try r.u16v();
        t.name = try a.dupe(u8, try r.take(name_len));
        t.dtype = std.enums.fromInt(Dtype, try r.u8v()) orelse return error.InvalidDtype;
        const ndim = try r.u8v();
        const shape = try a.alloc(u64, ndim);
        for (shape) |*d| d.* = try r.u64v();
        t.shape = shape;

        const n_blocks = try r.checkCount(try r.u32v(), 16);
        t.blocks = try a.alloc(ParsedBlock, n_blocks);
        for (t.blocks) |*block| {
            const pid = try r.u32v();
            if (pid >= programs.len) return error.InvalidProgramId;
            const n_refs = try r.checkCount(try r.u32v(), 4);
            const refs = try a.alloc(u32, n_refs);
            for (refs) |*id| id.* = try r.u32v();
            const off = std.math.cast(usize, try r.u64v()) orelse return error.Truncated;
            const len = std.math.cast(usize, try r.u64v()) orelse return error.Truncated;
            if (off < HEADER.len or off > index_off or len > index_off - off) return error.Truncated;

            const node = &programs[pid];
            try attachTables(node, tables, refs);
            const payload = bytes[off..][0..len];
            try program.distributePayload(node, payload);
            block.* = .{ .node = node, .tables = tables, .refs = refs, .payload = payload };
        }
    }

    return .{ .tensors = out_tensors, .safetensors_prefix = safetensors_prefix, .arena = arena };
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

fn attachTables(node: *Node, tables: []const ParsedTable, refs: []const u32) !void {
    var ref_i: usize = 0;
    try attachTablesRec(node, tables, refs, &ref_i);
    if (ref_i != refs.len) return error.TableRefMismatch;
}

fn attachTablesRec(node: *Node, tables: []const ParsedTable, refs: []const u32, ref_i: *usize) !void {
    switch (node.side) {
        .huffman => |*h| {
            if (ref_i.* >= refs.len) return error.TableRefMismatch;
            const id = refs[ref_i.*];
            ref_i.* += 1;
            if (id >= tables.len) return error.InvalidTableId;
            if (std.meta.activeTag(tables[id]) != .huffman) return error.TableKindMismatch;
            h.table.entries = tables[id].huffman.entries;
        },
        .rans => |*r| {
            if (ref_i.* >= refs.len) return error.TableRefMismatch;
            const id = refs[ref_i.*];
            ref_i.* += 1;
            if (id >= tables.len) return error.InvalidTableId;
            if (std.meta.activeTag(tables[id]) != .rans) return error.TableKindMismatch;
            r.table.symbols = tables[id].rans.symbols;
            r.table.info = tables[id].rans.info;
        },
        else => {},
    }
    for (node.children) |*child| try attachTablesRec(child, tables, refs, ref_i);
}
