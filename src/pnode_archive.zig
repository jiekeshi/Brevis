//! Brevis v2 archive: stores `astar.PNode` trees (the unified low-level
//! grammar's programs) rather than the legacy high-level `program.Node`.
//!
//! File layout (little-endian throughout):
//!
//!   [4]    MAGIC = "BRV\x02"
//!   [2]    VERSION (u16 = 2)
//!   [2]    FLAGS (u16, reserved)
//!   [4]    N_TABLES (u32)
//!   per table:
//!     [1]  KIND (0 = huffman, 1 = rans)
//!     [4]  N_ENTRIES (u32)
//!     huffman: each entry = u32 sym + u8 len  (5 bytes)
//!     rans:    each entry = u32 sym + u32 freq (8 bytes)
//!   [4]    N_TENSORS (u32)
//!   per tensor:
//!     [2]  NAME_LEN (u16)
//!     [N]  NAME (utf-8)
//!     [1]  DTYPE (u8)
//!     [1]  NDIM (u8)
//!     [N]  SHAPE (u64 × NDIM)
//!     [4]  PROGRAM_LEN (u32)
//!     [N]  PROGRAM_BYTES (PNode tree, see below)
//!
//! PNode encoding (recursive pre-order):
//!   TAG = 0x00 (terminal):
//!     KIND: u8 (huffman / rans / raw, from lowlevel.OpKind tag)
//!     COUNT: u64 (number of logical elements this stream had)
//!     BPE: u8 (bits_per_elem)
//!     TABLE_ID: u32 (only for huffman/rans; raw uses u32 sentinel 0xFFFFFFFF)
//!     PAYLOAD_LEN: u64
//!     PAYLOAD: bytes
//!   TAG = 0x01 (chain):
//!     OP_KIND: u8
//!     OP_PARAMS: u32
//!     CHILD: recursive
//!   TAG = 0x02 (split):
//!     OP_KIND: u8
//!     OP_PARAMS: u32
//!     HI: recursive
//!     LO: recursive

const std = @import("std");
const types = @import("types.zig");
const codec = @import("codec.zig");
const astar = @import("astar.zig");
const lowlevel = @import("lowlevel.zig");

const Allocator = types.Allocator;

pub const MAGIC: [4]u8 = .{ 'B', 'R', 'V', 2 };
pub const VERSION: u16 = 2;

pub const TableKind = enum(u8) { huffman = 0, rans = 1 };

pub const SharedTable = union(TableKind) {
    huffman: codec.HuffmanTable,
    rans: codec.RansTable,

    pub fn deinit(self: *SharedTable, alloc: Allocator) void {
        switch (self.*) {
            .huffman => |*h| h.deinit(alloc),
            .rans => |*r| r.deinit(alloc),
        }
    }
};

pub const TensorJob = struct {
    name: []const u8,
    dtype: types.Dtype,
    shape: []const u64,
    program: *astar.PNode, // already realized (carries side_info + payload)
};

const TableMap = struct {
    by_fp: std.AutoHashMap(u64, u32),
    tables: std.ArrayList(SharedTable),
    alloc: Allocator,

    fn init(alloc: Allocator) TableMap {
        return .{
            .by_fp = .init(alloc),
            .tables = .empty,
            .alloc = alloc,
        };
    }
    fn deinit(self: *TableMap) void {
        self.by_fp.deinit();
        for (self.tables.items) |*t| t.deinit(self.alloc);
        self.tables.deinit(self.alloc);
    }

    fn internHuffman(self: *TableMap, t: codec.HuffmanTable) !u32 {
        const fp = fingerprintHuffman(t);
        if (self.by_fp.get(fp)) |id| return id;
        const cloned = try self.alloc.alloc(codec.HuffmanTable.Entry, t.entries.len);
        @memcpy(cloned, t.entries);
        const id: u32 = @intCast(self.tables.items.len);
        try self.tables.append(self.alloc, .{ .huffman = .{ .entries = cloned } });
        try self.by_fp.put(fp, id);
        return id;
    }

    fn internRans(self: *TableMap, t: codec.RansTable) !u32 {
        const fp = fingerprintRans(t);
        if (self.by_fp.get(fp)) |id| return id;
        const cs = try self.alloc.alloc(u32, t.symbols.len);
        @memcpy(cs, t.symbols);
        const ci = try self.alloc.alloc(codec.RansSymbol, t.info.len);
        @memcpy(ci, t.info);
        const id: u32 = @intCast(self.tables.items.len);
        try self.tables.append(self.alloc, .{ .rans = .{ .symbols = cs, .info = ci } });
        try self.by_fp.put(fp, id);
        return id;
    }
};

fn fingerprintHuffman(t: codec.HuffmanTable) u64 {
    var h: std.hash.XxHash64 = .init(0xBEEF0001);
    var n_bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &n_bytes, @intCast(t.entries.len), .little);
    h.update(&n_bytes);
    for (t.entries) |e| {
        var sb: [4]u8 = undefined;
        std.mem.writeInt(u32, &sb, e.sym, .little);
        h.update(&sb);
        h.update(&[_]u8{e.len});
    }
    return h.final();
}
fn fingerprintRans(t: codec.RansTable) u64 {
    var h: std.hash.XxHash64 = .init(0xBEEF0002);
    var n_bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &n_bytes, @intCast(t.symbols.len), .little);
    h.update(&n_bytes);
    for (t.symbols, t.info) |s, info| {
        var sb: [4]u8 = undefined;
        std.mem.writeInt(u32, &sb, s, .little);
        h.update(&sb);
        std.mem.writeInt(u32, &sb, info.freq, .little);
        h.update(&sb);
    }
    return h.final();
}

// =================== build archive ===================

pub fn buildArchiveBytes(alloc: Allocator, jobs: []const TensorJob) ![]u8 {
    var map = TableMap.init(alloc);
    defer map.deinit();

    // Phase 1: walk every program, intern tables.
    for (jobs) |job| try internTablesInPNode(&map, job.program);

    // Phase 2: write the archive.
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(alloc);

    try out.appendSlice(alloc, &MAGIC);
    try writeU16(alloc, &out, VERSION);
    try writeU16(alloc, &out, 0);

    try writeU32(alloc, &out, @intCast(map.tables.items.len));
    for (map.tables.items) |t| switch (t) {
        .huffman => |ht| {
            try out.append(alloc, @intFromEnum(TableKind.huffman));
            try writeU32(alloc, &out, @intCast(ht.entries.len));
            for (ht.entries) |e| {
                try writeU32(alloc, &out, e.sym);
                try out.append(alloc, e.len);
            }
        },
        .rans => |rt| {
            try out.append(alloc, @intFromEnum(TableKind.rans));
            try writeU32(alloc, &out, @intCast(rt.symbols.len));
            for (rt.symbols, rt.info) |s, info| {
                try writeU32(alloc, &out, s);
                try writeU32(alloc, &out, info.freq);
            }
        },
    };

    try writeU32(alloc, &out, @intCast(jobs.len));
    for (jobs) |job| {
        try writeU16(alloc, &out, @intCast(job.name.len));
        try out.appendSlice(alloc, job.name);
        try out.append(alloc, @intFromEnum(job.dtype));
        try out.append(alloc, @intCast(job.shape.len));
        for (job.shape) |d| try writeU64(alloc, &out, d);

        var prog_bytes: std.ArrayList(u8) = .empty;
        defer prog_bytes.deinit(alloc);
        try writePNode(alloc, &prog_bytes, job.program, &map);
        try writeU32(alloc, &out, @intCast(prog_bytes.items.len));
        try out.appendSlice(alloc, prog_bytes.items);
    }

    return out.toOwnedSlice(alloc);
}

fn internTablesInPNode(map: *TableMap, node: *const astar.PNode) !void {
    switch (node.*) {
        .terminal => |t| switch (t.side_info) {
            .huffman => |h| _ = try map.internHuffman(h.table),
            .rans => |r| _ = try map.internRans(r.table),
            .raw => {},
        },
        .chain => |c| try internTablesInPNode(map, c.next),
        .split => |s| {
            try internTablesInPNode(map, s.hi);
            try internTablesInPNode(map, s.lo);
        },
    }
}

fn writePNode(alloc: Allocator, out: *std.ArrayList(u8), node: *const astar.PNode, map: *TableMap) !void {
    switch (node.*) {
        .terminal => |t| {
            try out.append(alloc, 0x00);
            try out.append(alloc, @intFromEnum(t.kind));
            switch (t.side_info) {
                .huffman => |h| {
                    try writeU64(alloc, out, h.count);
                    try out.append(alloc, h.bits_per_elem);
                    const id = try map.internHuffman(h.table);
                    try writeU32(alloc, out, id);
                },
                .rans => |r| {
                    try writeU64(alloc, out, r.count);
                    try out.append(alloc, r.bits_per_elem);
                    const id = try map.internRans(r.table);
                    try writeU32(alloc, out, id);
                },
                .raw => |info| {
                    try writeU64(alloc, out, info.count);
                    try out.append(alloc, info.bits_per_elem);
                    try writeU32(alloc, out, 0xFFFFFFFF);
                },
            }
            try writeU64(alloc, out, @intCast(t.payload.len));
            try out.appendSlice(alloc, t.payload);
        },
        .chain => |c| {
            try out.append(alloc, 0x01);
            try out.append(alloc, @intFromEnum(c.op.kind));
            try writeU32(alloc, out, c.op.params.raw);
            try writePNode(alloc, out, c.next, map);
        },
        .split => |s| {
            try out.append(alloc, 0x02);
            try out.append(alloc, @intFromEnum(s.op.kind));
            try writeU32(alloc, out, s.op.params.raw);
            try writePNode(alloc, out, s.hi, map);
            try writePNode(alloc, out, s.lo, map);
        },
    }
}

// =================== parse archive ===================

pub const ParsedTensor = struct {
    name: []u8,
    dtype: types.Dtype,
    shape: []u64,
    program: *astar.PNode,

    pub fn deinit(self: *ParsedTensor, alloc: Allocator) void {
        alloc.free(self.name);
        alloc.free(self.shape);
        self.program.deinit(alloc);
        alloc.destroy(self.program);
    }
};

pub const Parsed = struct {
    tables: []SharedTable,
    tensors: []ParsedTensor,

    pub fn deinit(self: *Parsed, alloc: Allocator) void {
        for (self.tables) |*t| t.deinit(alloc);
        alloc.free(self.tables);
        for (self.tensors) |*t| t.deinit(alloc);
        alloc.free(self.tensors);
    }
};

const Reader = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn readU8(self: *Reader) u8 {
        const v = self.bytes[self.pos];
        self.pos += 1;
        return v;
    }
    fn readU16(self: *Reader) u16 {
        const v = std.mem.readInt(u16, self.bytes[self.pos..][0..2], .little);
        self.pos += 2;
        return v;
    }
    fn readU32(self: *Reader) u32 {
        const v = std.mem.readInt(u32, self.bytes[self.pos..][0..4], .little);
        self.pos += 4;
        return v;
    }
    fn readU64(self: *Reader) u64 {
        const v = std.mem.readInt(u64, self.bytes[self.pos..][0..8], .little);
        self.pos += 8;
        return v;
    }
    fn slice(self: *Reader, n: usize) []const u8 {
        const s = self.bytes[self.pos .. self.pos + n];
        self.pos += n;
        return s;
    }
};

pub fn parseArchive(alloc: Allocator, bytes: []const u8) !Parsed {
    var r: Reader = .{ .bytes = bytes };
    if (r.bytes.len < 12 or !std.mem.eql(u8, r.slice(4), &MAGIC)) return error.BadMagic;
    const ver = r.readU16();
    if (ver != VERSION) return error.UnsupportedVersion;
    _ = r.readU16(); // flags

    const n_tables = r.readU32();
    const tables = try alloc.alloc(SharedTable, n_tables);
    errdefer alloc.free(tables);
    for (tables) |*t| {
        const kind: TableKind = @enumFromInt(r.readU8());
        const n_entries = r.readU32();
        switch (kind) {
            .huffman => {
                const entries = try alloc.alloc(codec.HuffmanTable.Entry, n_entries);
                for (entries) |*e| {
                    e.sym = r.readU32();
                    e.len = r.readU8();
                }
                t.* = .{ .huffman = .{ .entries = entries } };
            },
            .rans => {
                const syms = try alloc.alloc(u32, n_entries);
                const info = try alloc.alloc(codec.RansSymbol, n_entries);
                var cum: u32 = 0;
                for (syms, info) |*s, *inf| {
                    s.* = r.readU32();
                    inf.freq = r.readU32();
                    inf.cum = cum;
                    cum += inf.freq;
                }
                t.* = .{ .rans = .{ .symbols = syms, .info = info } };
            },
        }
    }

    const n_tensors = r.readU32();
    const tensors = try alloc.alloc(ParsedTensor, n_tensors);
    for (tensors) |*t| {
        const name_len = r.readU16();
        const name_buf = try alloc.alloc(u8, name_len);
        @memcpy(name_buf, r.slice(name_len));
        const dtype: types.Dtype = @enumFromInt(r.readU8());
        const ndim = r.readU8();
        const shape = try alloc.alloc(u64, ndim);
        for (shape) |*d| d.* = r.readU64();
        const prog_len = r.readU32();
        const prog_end_pos = r.pos + prog_len;
        const prog = try readPNode(alloc, &r, tables);
        if (r.pos != prog_end_pos) return error.MalformedProgram;
        t.* = .{ .name = name_buf, .dtype = dtype, .shape = shape, .program = prog };
    }

    return .{ .tables = tables, .tensors = tensors };
}

fn readPNode(alloc: Allocator, r: *Reader, tables: []const SharedTable) !*astar.PNode {
    const tag = r.readU8();
    const node = try alloc.create(astar.PNode);
    switch (tag) {
        0x00 => {
            const kind: lowlevel.OpKind = @enumFromInt(r.readU8());
            const count = r.readU64();
            const bpe = r.readU8();
            const table_id = r.readU32();
            const payload_len = r.readU64();
            const payload_buf = try alloc.alloc(u8, @intCast(payload_len));
            @memcpy(payload_buf, r.slice(@intCast(payload_len)));
            const side_info: astar.TerminalSide = switch (kind) {
                .huffman => blk: {
                    const orig = tables[table_id].huffman.entries;
                    const cloned = try alloc.alloc(codec.HuffmanTable.Entry, orig.len);
                    @memcpy(cloned, orig);
                    break :blk .{ .huffman = .{ .table = .{ .entries = cloned }, .count = @intCast(count), .bits_per_elem = bpe } };
                },
                .rans => blk: {
                    const orig_syms = tables[table_id].rans.symbols;
                    const orig_info = tables[table_id].rans.info;
                    const cs = try alloc.alloc(u32, orig_syms.len);
                    const ci = try alloc.alloc(codec.RansSymbol, orig_info.len);
                    @memcpy(cs, orig_syms);
                    @memcpy(ci, orig_info);
                    break :blk .{ .rans = .{ .table = .{ .symbols = cs, .info = ci }, .count = @intCast(count), .bits_per_elem = bpe } };
                },
                .raw => .{ .raw = .{ .count = @intCast(count), .bits_per_elem = bpe } },
                else => return error.BadTerminalKind,
            };
            node.* = .{ .terminal = .{
                .kind = kind,
                .bits = @as(u64, payload_buf.len) * 8,
                .side_info = side_info,
                .payload = payload_buf,
                .payload_owned = true,
            } };
        },
        0x01 => {
            const op_kind: lowlevel.OpKind = @enumFromInt(r.readU8());
            const op_params = r.readU32();
            const next = try readPNode(alloc, r, tables);
            node.* = .{ .chain = .{
                .op = .{ .kind = op_kind, .params = .{ .raw = op_params } },
                .next = next,
            } };
        },
        0x02 => {
            const op_kind: lowlevel.OpKind = @enumFromInt(r.readU8());
            const op_params = r.readU32();
            const hi = try readPNode(alloc, r, tables);
            const lo = try readPNode(alloc, r, tables);
            node.* = .{ .split = .{
                .op = .{ .kind = op_kind, .params = .{ .raw = op_params } },
                .hi = hi,
                .lo = lo,
            } };
        },
        else => return error.BadPNodeTag,
    }
    return node;
}

// =================== helpers ===================
fn writeU16(alloc: Allocator, out: *std.ArrayList(u8), v: u16) !void {
    var buf: [2]u8 = undefined;
    std.mem.writeInt(u16, &buf, v, .little);
    try out.appendSlice(alloc, &buf);
}
fn writeU32(alloc: Allocator, out: *std.ArrayList(u8), v: u32) !void {
    var buf: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf, v, .little);
    try out.appendSlice(alloc, &buf);
}
fn writeU64(alloc: Allocator, out: *std.ArrayList(u8), v: u64) !void {
    var buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &buf, v, .little);
    try out.appendSlice(alloc, &buf);
}
