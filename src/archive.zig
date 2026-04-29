//! Brevis archive container (.brv) — bundles many compressed tensors with
//! cross-tensor codebook sharing.
//!
//! File layout (little-endian throughout):
//!
//!   [4]    MAGIC = "BRV\x01"
//!   [2]    VERSION (u16)
//!   [2]    FLAGS (u16) — reserved
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
//!     [2]  BASE_NAME_LEN (u16) — 0 if no base referenced
//!     [N]  BASE_NAME (utf-8)
//!     [4]  PROGRAM_LEN (u32)
//!     [N]  PROGRAM_BYTES (uses table-id refs instead of inline tables)
//!     [8]  PAYLOAD_LEN (u64)
//!     [N]  PAYLOAD_BYTES
//!
//! Codebook sharing:
//! Every huffman/rans table in any tensor's program is interned in the file's
//! shared table list, identified by content fingerprint. Programs reference
//! tables by index; identical tables across tensors are stored once.

const std = @import("std");
const types = @import("types.zig");
const codec = @import("codec.zig");
const program = @import("program.zig");
const ops = @import("ops.zig");

const Allocator = types.Allocator;

pub const MAGIC: [4]u8 = .{ 'B', 'R', 'V', 1 };
pub const VERSION: u16 = 1;

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

pub const TensorEntry = struct {
    name: []const u8, // owned
    base_name: []const u8, // owned, may be empty
    program_bytes: []u8, // owned
    payload_bytes: []u8, // owned

    pub fn deinit(self: *TensorEntry, alloc: Allocator) void {
        alloc.free(self.name);
        alloc.free(self.base_name);
        alloc.free(self.program_bytes);
        alloc.free(self.payload_bytes);
    }
};

pub const Archive = struct {
    tables: []SharedTable, // owned
    tensors: []TensorEntry, // owned

    pub fn deinit(self: *Archive, alloc: Allocator) void {
        for (self.tables) |*t| t.deinit(alloc);
        alloc.free(self.tables);
        for (self.tensors) |*t| t.deinit(alloc);
        alloc.free(self.tensors);
    }
};

// ---------- Hashing tables for dedup ----------

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

// ---------- Build dedup map by walking all programs ----------

const TableMap = struct {
    by_fp: std.AutoHashMap(u64, u32), // fingerprint → table index
    tables: std.ArrayList(SharedTable),

    fn init(alloc: Allocator) TableMap {
        return .{
            .by_fp = .init(alloc),
            .tables = .empty,
        };
    }
    fn deinit(self: *TableMap, alloc: Allocator) void {
        self.by_fp.deinit();
        // tables are transferred out via toOwnedSlice; if not, free them
        for (self.tables.items) |*t| t.deinit(alloc);
        self.tables.deinit(alloc);
    }

    fn internHuffman(self: *TableMap, alloc: Allocator, t: codec.HuffmanTable) !u32 {
        const fp = fingerprintHuffman(t);
        if (self.by_fp.get(fp)) |id| return id;
        // Clone the table and insert.
        const cloned_entries = try alloc.alloc(codec.HuffmanTable.Entry, t.entries.len);
        @memcpy(cloned_entries, t.entries);
        const id: u32 = @intCast(self.tables.items.len);
        try self.tables.append(alloc, .{ .huffman = .{ .entries = cloned_entries } });
        try self.by_fp.put(fp, id);
        return id;
    }

    fn internRans(self: *TableMap, alloc: Allocator, t: codec.RansTable) !u32 {
        const fp = fingerprintRans(t);
        if (self.by_fp.get(fp)) |id| return id;
        const cloned_syms = try alloc.alloc(u32, t.symbols.len);
        @memcpy(cloned_syms, t.symbols);
        const cloned_info = try alloc.alloc(codec.RansSymbol, t.info.len);
        @memcpy(cloned_info, t.info);
        const id: u32 = @intCast(self.tables.items.len);
        try self.tables.append(alloc, .{ .rans = .{ .symbols = cloned_syms, .info = cloned_info } });
        try self.by_fp.put(fp, id);
        return id;
    }
};

fn collectTables(alloc: Allocator, node: *const program.Node, map: *TableMap, ids: *std.ArrayList(u32)) !void {
    switch (node.side_info) {
        .huffman => |h| {
            const id = try map.internHuffman(alloc, h.table);
            try ids.append(alloc, id);
        },
        .rans => |r| {
            const id = try map.internRans(alloc, r.table);
            try ids.append(alloc, id);
        },
        else => {},
    }
    for (node.children) |*c| try collectTables(alloc, c, map, ids);
}

// ---------- Program serialization with shared table refs ----------
//
// Mirrors program.zig's serializeProgram but writes a u32 table id for
// huffman/rans nodes instead of the inline entries. The id is consumed in
// pre-order from the `ids` slice (parallel to collectTables).

fn writeProgramShared(alloc: Allocator, out: *std.ArrayList(u8), node: *const program.Node, ids: []const u32, ids_pos: *usize) !void {
    try out.append(alloc, @intFromEnum(node.op));
    var buf4: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf4, node.base_id, .little);
    try out.appendSlice(alloc, &buf4);
    try writeSideInfoShared(alloc, out, node, ids, ids_pos);
    try out.append(alloc, @intCast(node.children.len));
    for (node.children) |*c| try writeProgramShared(alloc, out, c, ids, ids_pos);
}

fn writeSideInfoShared(alloc: Allocator, out: *std.ArrayList(u8), node: *const program.Node, ids: []const u32, ids_pos: *usize) !void {
    switch (node.side_info) {
        .none => {},
        .split_float => |i| {
            try out.append(alloc, @intFromEnum(i.dtype));
            try out.append(alloc, i.exp_bits);
            try out.append(alloc, i.mant_bits);
            try out.append(alloc, i.ndim);
            for (0..i.ndim) |k| try writeU64(alloc, out, i.shape[k]);
        },
        .bitplane_split => |i| {
            try out.append(alloc, i.n_planes);
            try writeU64(alloc, out, i.count);
        },
        .delta_encode => |i| {
            try writeU32(alloc, out, i.first);
            try writeU64(alloc, out, i.count);
            try out.append(alloc, i.bits_per_elem);
        },
        .huffman => |i| {
            try writeU64(alloc, out, i.count);
            try out.append(alloc, i.bits_per_elem);
            const id = ids[ids_pos.*];
            ids_pos.* += 1;
            try writeU32(alloc, out, id);
        },
        .rans => |i| {
            try writeU64(alloc, out, i.count);
            try out.append(alloc, i.bits_per_elem);
            const id = ids[ids_pos.*];
            ids_pos.* += 1;
            try writeU32(alloc, out, id);
        },
        .raw => |i| {
            try writeU64(alloc, out, i.count);
            try out.append(alloc, i.bits_per_elem);
        },
        .tensor_raw => |i| {
            try out.append(alloc, @intFromEnum(i.dtype));
            try out.append(alloc, i.ndim);
            for (0..i.ndim) |k| try writeU64(alloc, out, i.shape[k]);
        },
        .tensor_xor => |i| {
            try out.append(alloc, @intFromEnum(i.dtype));
            try out.append(alloc, i.ndim);
            for (0..i.ndim) |k| try writeU64(alloc, out, i.shape[k]);
        },
    }
}

fn readProgramShared(alloc: Allocator, r: *program.ProgramReader, shared: []const SharedTable) !program.Node {
    const op: program.OpKind = @enumFromInt(r.readU8());
    const base_id = r.readU32();
    var side_info: program.SideInfo = .none;
    switch (op) {
        .split_float => {
            const dtype: types.Dtype = @enumFromInt(r.readU8());
            const exp_bits = r.readU8();
            const mant_bits = r.readU8();
            const ndim = r.readU8();
            var info: ops.SplitFloatInfo = .{ .dtype = dtype, .exp_bits = exp_bits, .mant_bits = mant_bits, .ndim = ndim, .shape = .{0} ** 8 };
            for (0..ndim) |k| info.shape[k] = r.readU64();
            side_info = .{ .split_float = info };
        },
        .bitplane_split => {
            const n_planes = r.readU8();
            const count = r.readU64();
            side_info = .{ .bitplane_split = .{ .n_planes = n_planes, .count = count } };
        },
        .delta_encode => {
            const first = r.readU32();
            const count = r.readU64();
            const bpe = r.readU8();
            side_info = .{ .delta_encode = .{ .first = first, .count = count, .bits_per_elem = bpe } };
        },
        .huffman => {
            const count = r.readU64();
            const bpe = r.readU8();
            const id = r.readU32();
            // Clone the shared entries so this Node owns them — required because
            // the Node's deinit will free them, and the same shared table may be
            // referenced by other nodes too.
            const orig = shared[id].huffman.entries;
            const cloned = try alloc.alloc(codec.HuffmanTable.Entry, orig.len);
            @memcpy(cloned, orig);
            side_info = .{ .huffman = .{ .table = .{ .entries = cloned }, .count = count, .bits_per_elem = bpe } };
        },
        .rans => {
            const count = r.readU64();
            const bpe = r.readU8();
            const id = r.readU32();
            const orig_syms = shared[id].rans.symbols;
            const orig_info = shared[id].rans.info;
            const cs = try alloc.alloc(u32, orig_syms.len);
            const ci = try alloc.alloc(codec.RansSymbol, orig_info.len);
            @memcpy(cs, orig_syms);
            @memcpy(ci, orig_info);
            side_info = .{ .rans = .{ .table = .{ .symbols = cs, .info = ci }, .count = count, .bits_per_elem = bpe } };
        },
        .raw => {
            const count = r.readU64();
            const bpe = r.readU8();
            side_info = .{ .raw = .{ .count = count, .bits_per_elem = bpe } };
        },
        .tensor_raw => {
            const dtype: types.Dtype = @enumFromInt(r.readU8());
            const ndim = r.readU8();
            var info: ops.TensorRawInfo = .{ .dtype = dtype, .ndim = ndim, .shape = .{0} ** 8 };
            for (0..ndim) |k| info.shape[k] = r.readU64();
            side_info = .{ .tensor_raw = info };
        },
        .tensor_xor => {
            const dtype: types.Dtype = @enumFromInt(r.readU8());
            const ndim = r.readU8();
            var info: ops.TensorXorInfo = .{ .dtype = dtype, .ndim = ndim, .shape = .{0} ** 8 };
            for (0..ndim) |k| info.shape[k] = r.readU64();
            side_info = .{ .tensor_xor = info };
        },
    }
    const n_kids = r.readU8();
    const kids = try alloc.alloc(program.Node, n_kids);
    for (kids) |*c| c.* = try readProgramShared(alloc, r, shared);
    return .{ .op = op, .children = kids, .base_id = base_id, .side_info = side_info };
}

// ---------- Build archive from a list of (name, program, payload, base_name?) ----------
pub const TensorJob = struct {
    name: []const u8,
    base_name: []const u8 = &.{},
    program: *const program.Node,
    payload: []const u8,
};

pub fn buildArchiveBytes(alloc: Allocator, jobs: []const TensorJob) ![]u8 {
    var map = TableMap.init(alloc);
    defer map.deinit(alloc);

    // Phase 1: collect IDs per job in pre-order to parallel writeProgramShared.
    var per_job_ids: std.ArrayList(std.ArrayList(u32)) = .empty;
    defer {
        for (per_job_ids.items) |*ids| ids.deinit(alloc);
        per_job_ids.deinit(alloc);
    }
    for (jobs) |job| {
        var ids: std.ArrayList(u32) = .empty;
        try collectTables(alloc, job.program, &map, &ids);
        try per_job_ids.append(alloc, ids);
    }

    // Phase 2: write archive.
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(alloc);

    try out.appendSlice(alloc, &MAGIC);
    try writeU16(alloc, &out, VERSION);
    try writeU16(alloc, &out, 0);
    try writeU32(alloc, &out, @intCast(map.tables.items.len));
    for (map.tables.items) |t| {
        try out.append(alloc, @intFromEnum(@as(TableKind, t)));
        switch (t) {
            .huffman => |ht| {
                try writeU32(alloc, &out, @intCast(ht.entries.len));
                for (ht.entries) |e| {
                    try writeU32(alloc, &out, e.sym);
                    try out.append(alloc, e.len);
                }
            },
            .rans => |rt| {
                try writeU32(alloc, &out, @intCast(rt.symbols.len));
                for (rt.symbols, rt.info) |s, info| {
                    try writeU32(alloc, &out, s);
                    try writeU32(alloc, &out, info.freq);
                }
            },
        }
    }

    try writeU32(alloc, &out, @intCast(jobs.len));
    for (jobs, 0..) |job, ji| {
        try writeU16(alloc, &out, @intCast(job.name.len));
        try out.appendSlice(alloc, job.name);
        try writeU16(alloc, &out, @intCast(job.base_name.len));
        try out.appendSlice(alloc, job.base_name);

        var prog_bytes: std.ArrayList(u8) = .empty;
        defer prog_bytes.deinit(alloc);
        var pos: usize = 0;
        try writeProgramShared(alloc, &prog_bytes, job.program, per_job_ids.items[ji].items, &pos);
        try writeU32(alloc, &out, @intCast(prog_bytes.items.len));
        try out.appendSlice(alloc, prog_bytes.items);

        try writeU64(alloc, &out, @intCast(job.payload.len));
        try out.appendSlice(alloc, job.payload);
    }

    return out.toOwnedSlice(alloc);
}

// ---------- Read archive from bytes ----------
pub const ParsedTensor = struct {
    name: []const u8, // owned
    base_name: []const u8, // owned
    program: program.Node, // owned
    payload: []u8, // owned (a copy into a separate buffer to keep sharing-by-ref intact for distributePayloadBytes)

    pub fn deinit(self: *ParsedTensor, alloc: Allocator) void {
        alloc.free(self.name);
        alloc.free(self.base_name);
        self.program.deinit(alloc);
        alloc.free(self.payload);
    }
};

pub const ParsedArchive = struct {
    shared: []SharedTable,
    tensors: []ParsedTensor,

    pub fn deinit(self: *ParsedArchive, alloc: Allocator) void {
        for (self.shared) |*t| t.deinit(alloc);
        alloc.free(self.shared);
        for (self.tensors) |*t| t.deinit(alloc);
        alloc.free(self.tensors);
    }
};

pub fn parseArchive(alloc: Allocator, bytes: []const u8) !ParsedArchive {
    var pos: usize = 0;
    if (bytes.len < 12) return error.ArchiveTooShort;
    if (!std.mem.eql(u8, bytes[0..4], &MAGIC)) return error.BadMagic;
    pos = 4;
    const version = std.mem.readInt(u16, bytes[pos..][0..2], .little);
    pos += 2;
    if (version != VERSION) return error.UnsupportedVersion;
    pos += 2; // flags

    const n_tables = std.mem.readInt(u32, bytes[pos..][0..4], .little);
    pos += 4;

    const shared = try alloc.alloc(SharedTable, n_tables);
    errdefer alloc.free(shared);

    for (shared) |*tbl| {
        const kind: TableKind = @enumFromInt(bytes[pos]);
        pos += 1;
        const n_entries = std.mem.readInt(u32, bytes[pos..][0..4], .little);
        pos += 4;
        switch (kind) {
            .huffman => {
                const entries = try alloc.alloc(codec.HuffmanTable.Entry, n_entries);
                for (entries) |*e| {
                    e.sym = std.mem.readInt(u32, bytes[pos..][0..4], .little);
                    pos += 4;
                    e.len = bytes[pos];
                    pos += 1;
                }
                tbl.* = .{ .huffman = .{ .entries = entries } };
            },
            .rans => {
                const syms = try alloc.alloc(u32, n_entries);
                const info = try alloc.alloc(codec.RansSymbol, n_entries);
                var cum: u32 = 0;
                for (syms, info) |*s, *inf| {
                    s.* = std.mem.readInt(u32, bytes[pos..][0..4], .little);
                    pos += 4;
                    inf.freq = std.mem.readInt(u32, bytes[pos..][0..4], .little);
                    pos += 4;
                    inf.cum = cum;
                    cum += inf.freq;
                }
                tbl.* = .{ .rans = .{ .symbols = syms, .info = info } };
            },
        }
    }

    const n_tensors = std.mem.readInt(u32, bytes[pos..][0..4], .little);
    pos += 4;

    const tensors = try alloc.alloc(ParsedTensor, n_tensors);
    for (tensors) |*t| {
        const name_len = std.mem.readInt(u16, bytes[pos..][0..2], .little);
        pos += 2;
        const name_buf = try alloc.alloc(u8, name_len);
        @memcpy(name_buf, bytes[pos .. pos + name_len]);
        pos += name_len;

        const base_len = std.mem.readInt(u16, bytes[pos..][0..2], .little);
        pos += 2;
        const base_buf = try alloc.alloc(u8, base_len);
        @memcpy(base_buf, bytes[pos .. pos + base_len]);
        pos += base_len;

        const prog_len = std.mem.readInt(u32, bytes[pos..][0..4], .little);
        pos += 4;
        var pr: program.ProgramReader = .{ .bytes = bytes[pos .. pos + prog_len] };
        var prog_node = try readProgramShared(alloc, &pr, shared);
        pos += prog_len;

        const pay_len = std.mem.readInt(u64, bytes[pos..][0..8], .little);
        pos += 8;
        const pay_buf = try alloc.alloc(u8, @intCast(pay_len));
        @memcpy(pay_buf, bytes[pos .. pos + @as(usize, @intCast(pay_len))]);
        pos += @intCast(pay_len);

        // Wire up payloads into terminal nodes.
        try program.distributePayloadBytes(&prog_node, pay_buf);

        t.* = .{ .name = name_buf, .base_name = base_buf, .program = prog_node, .payload = pay_buf };
    }

    return .{ .shared = shared, .tensors = tensors };
}

// ---------- helpers ----------
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
