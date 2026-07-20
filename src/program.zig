//! Program trees: a tree of `ops.OpKind` nodes. Transforms fan a stream out
//! into their children; terminals encode a stream into bytes via codec.zig.
//! `execute` runs top-down, `decode` bottom-up.

const std = @import("std");
const types = @import("types.zig");
const codec = @import("codec.zig");
const ops = @import("ops.zig");

const Allocator = std.mem.Allocator;
const Stream = types.Stream;
const Dtype = types.Dtype;
const SideTag = std.meta.Tag(ops.SideInfo);

pub const Node = struct {
    op: ops.OpKind,
    params: u32 = 0,
    children: []Node = &.{},
    side: ops.SideInfo = .none,
    payload: []u8 = &.{},
    payload_owned: bool = false,

    pub fn deinit(self: *Node, alloc: Allocator) void {
        for (self.children) |*c| c.deinit(alloc);
        if (self.children.len > 0) alloc.free(self.children);
        self.children = &.{};
        if (self.payload_owned) alloc.free(self.payload);
        self.payload = &.{};
        self.payload_owned = false;
        self.side.deinit(alloc);
    }

    pub fn clone(self: Node, alloc: Allocator) Allocator.Error!Node {
        var side = try self.side.clone(alloc);
        errdefer side.deinit(alloc);

        const payload = if (self.payload_owned) try alloc.dupe(u8, self.payload) else self.payload;
        errdefer if (self.payload_owned) alloc.free(payload);

        const kids = try alloc.alloc(Node, self.children.len);
        var filled: usize = 0;
        errdefer {
            for (kids[0..filled]) |*c| c.deinit(alloc);
            alloc.free(kids);
        }
        for (self.children, 0..) |c, i| {
            kids[i] = try c.clone(alloc);
            filled = i + 1;
        }

        return .{
            .op = self.op,
            .params = self.params,
            .children = kids,
            .side = side,
            .payload = payload,
            .payload_owned = self.payload_owned,
        };
    }

    pub fn countNodes(self: Node) usize {
        var n: usize = 1;
        for (self.children) |c| n += c.countNodes();
        return n;
    }

    /// Height in edges: a terminal alone is 0.
    pub fn depth(self: Node) u8 {
        var d: u8 = 0;
        for (self.children) |c| d = @max(d, c.depth() + 1);
        return d;
    }
};

// ==================== execute ====================

pub const ExecuteError = Allocator.Error || error{ SymbolNotInTable, AlphabetTooLarge };

pub fn execute(alloc: Allocator, node: *Node, in: Stream) ExecuteError!void {
    node.side.deinit(alloc);
    if (node.payload_owned) alloc.free(node.payload);
    node.payload = &.{};
    node.payload_owned = false;

    if (node.op.isTerminal()) {
        std.debug.assert(node.children.len == 0);
        const term = ops.SideInfo{ .terminal = .{ .count = in.count, .bits_per_elem = in.bits_per_elem } };
        switch (node.op) {
            .raw => {
                node.payload = try alloc.dupe(u8, in.data[0 .. in.count * in.elemBytes()]);
                node.side = term;
            },
            .bitpack => {
                const w: u8 = @intCast(node.params & 0xFF);
                std.debug.assert(w > 0 and w <= in.bits_per_elem);
                node.payload = try codec.bitpackEncode(alloc, in, w);
                node.side = term;
            },
            .huffman => {
                var table = try codec.huffmanBuild(alloc, in);
                errdefer table.deinit(alloc);
                node.payload = try codec.huffmanEncode(alloc, in, table);
                node.side = .{ .huffman = .{
                    .table = table,
                    .count = in.count,
                    .bits_per_elem = in.bits_per_elem,
                } };
            },
            .rans => {
                var table = try codec.ransBuild(alloc, in);
                errdefer table.deinit(alloc);
                node.payload = try codec.ransEncode(alloc, in, table);
                node.side = .{ .rans = .{
                    .table = table,
                    .count = in.count,
                    .bits_per_elem = in.bits_per_elem,
                } };
            },
            else => unreachable,
        }
        node.payload_owned = true;
        return;
    }

    var outs: std.ArrayList(Stream) = .empty;
    defer {
        for (outs.items) |*s| s.deinit(alloc);
        outs.deinit(alloc);
    }
    try ops.forward(alloc, node.op, node.params, in, &outs, &node.side);
    std.debug.assert(outs.items.len == ops.arity(node.op, in.bits_per_elem));
    std.debug.assert(node.children.len == outs.items.len);

    for (node.children, outs.items) |*c, s| try execute(alloc, c, s);
}

// ==================== decode ====================

pub const DecodeError = Allocator.Error || error{
    CorruptHuffmanStream,
    CorruptRansStream,
    CorruptBitpackStream,
    CorruptPayload,
};

pub fn decode(alloc: Allocator, node: Node) DecodeError!Stream {
    switch (node.op) {
        .raw => {
            const t = node.side.terminal;
            var s = try Stream.init(alloc, t.count, t.bits_per_elem);
            errdefer s.deinit(alloc);
            if (node.payload.len < s.data.len) return error.CorruptPayload;
            @memcpy(s.data, node.payload[0..s.data.len]);
            return s;
        },
        .bitpack => {
            const t = node.side.terminal;
            const w: u8 = @intCast(node.params & 0xFF);
            return codec.bitpackDecode(alloc, node.payload, w, t.count, t.bits_per_elem);
        },
        .huffman => {
            const h = node.side.huffman;
            return codec.huffmanDecode(alloc, node.payload, h.table, h.count, h.bits_per_elem);
        },
        .rans => {
            const r = node.side.rans;
            return codec.ransDecode(alloc, node.payload, r.table, r.count, r.bits_per_elem);
        },
        else => {},
    }

    const ins = try alloc.alloc(Stream, node.children.len);
    var filled: usize = 0;
    defer {
        for (ins[0..filled]) |*s| s.deinit(alloc);
        alloc.free(ins);
    }
    for (node.children, 0..) |c, i| {
        ins[i] = try decode(alloc, c);
        filled = i + 1;
    }
    return ops.inverse(alloc, node.op, node.params, ins, node.side);
}

// ==================== payload packing ====================

/// DFS pre-order concatenation of terminal payloads, each prefixed by a u64 length.
pub fn collectPayload(alloc: Allocator, node: Node) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(alloc);
    try appendPayload(alloc, &out, node);
    return out.toOwnedSlice(alloc);
}

fn appendPayload(alloc: Allocator, out: *std.ArrayList(u8), node: Node) Allocator.Error!void {
    if (node.op.isTerminal()) {
        try wU64(alloc, out, node.payload.len);
        try out.appendSlice(alloc, node.payload);
        return;
    }
    for (node.children) |c| try appendPayload(alloc, out, c);
}

pub const DistributeError = error{ PayloadUnderrun, PayloadOverrun };

/// Re-attach a flat buffer produced by `collectPayload` as non-owning slices.
pub fn distributePayload(node: *Node, buf: []const u8) DistributeError!void {
    var pos: usize = 0;
    try distributeRec(node, buf, &pos);
    if (pos != buf.len) return error.PayloadOverrun;
}

fn distributeRec(node: *Node, buf: []const u8, pos: *usize) DistributeError!void {
    if (node.op.isTerminal()) {
        if (pos.* + 8 > buf.len) return error.PayloadUnderrun;
        const len: usize = @intCast(std.mem.readInt(u64, buf[pos.*..][0..8], .little));
        pos.* += 8;
        if (len > buf.len - pos.*) return error.PayloadUnderrun;
        if (node.payload_owned) return error.PayloadOverrun;
        node.payload = @constCast(buf[pos.*..][0..len]);
        pos.* += len;
        return;
    }
    for (node.children) |*c| try distributeRec(c, buf, pos);
}

// ==================== serialization ====================
//
// Per node: u8 op | u32 params | u8 side_tag | side body | u8 n_children | children...

pub fn serialize(alloc: Allocator, node: Node) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(alloc);
    try writeNode(alloc, &out, node);
    return out.toOwnedSlice(alloc);
}

fn writeNode(alloc: Allocator, out: *std.ArrayList(u8), node: Node) Allocator.Error!void {
    try out.append(alloc, @intFromEnum(node.op));
    try wU32(alloc, out, node.params);
    try writeSide(alloc, out, node.side);
    try out.append(alloc, @intCast(node.children.len));
    for (node.children) |c| try writeNode(alloc, out, c);
}

fn writeSide(alloc: Allocator, out: *std.ArrayList(u8), side: ops.SideInfo) Allocator.Error!void {
    try out.append(alloc, @intFromEnum(@as(SideTag, side)));
    switch (side) {
        .none => {},
        .terminal => |t| {
            try wU64(alloc, out, t.count);
            try out.append(alloc, t.bits_per_elem);
        },
        .rle => |t| {
            try wU64(alloc, out, t.count);
            try out.append(alloc, t.bits_per_elem);
        },
        .split => |t| {
            try wU64(alloc, out, t.count);
            try out.append(alloc, t.bits_per_elem);
        },
        .huffman => |h| {
            try wU64(alloc, out, h.count);
            try out.append(alloc, h.bits_per_elem);
            try wU32(alloc, out, @intCast(h.table.entries.len));
            for (h.table.entries) |e| {
                try wU32(alloc, out, e.sym);
                try out.append(alloc, e.len);
            }
        },
        .rans => |r| {
            try wU64(alloc, out, r.count);
            try out.append(alloc, r.bits_per_elem);
            try wU32(alloc, out, @intCast(r.table.symbols.len));
            for (r.table.symbols, r.table.info) |sym, inf| {
                try wU32(alloc, out, sym);
                try wU32(alloc, out, inf.freq);
            }
        },
        .codebook => |c| {
            try wU64(alloc, out, c.count);
            try out.append(alloc, c.bits_per_elem);
            try wU32(alloc, out, @intCast(c.syms.len));
            for (c.syms) |s| try wU32(alloc, out, s);
        },
        .sfloat => |f| {
            try out.append(alloc, @intFromEnum(f.dtype));
            try wU64(alloc, out, f.count);
        },
    }
}

fn wU32(alloc: Allocator, out: *std.ArrayList(u8), v: u32) Allocator.Error!void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    try out.appendSlice(alloc, &b);
}

fn wU64(alloc: Allocator, out: *std.ArrayList(u8), v: u64) Allocator.Error!void {
    var b: [8]u8 = undefined;
    std.mem.writeInt(u64, &b, v, .little);
    try out.appendSlice(alloc, &b);
}

pub const DeserializeError = Allocator.Error || error{
    Truncated,
    InvalidOpcode,
    InvalidSideTag,
    InvalidDtype,
    InvalidRansTable,
};

const Reader = struct {
    b: []const u8,
    pos: usize = 0,

    fn take(self: *Reader, n: usize) DeserializeError![]const u8 {
        if (n > self.b.len - self.pos) return error.Truncated;
        defer self.pos += n;
        return self.b[self.pos..][0..n];
    }
    fn u8v(self: *Reader) DeserializeError!u8 {
        return (try self.take(1))[0];
    }
    fn u32v(self: *Reader) DeserializeError!u32 {
        return std.mem.readInt(u32, (try self.take(4))[0..4], .little);
    }
    fn u64v(self: *Reader) DeserializeError!u64 {
        return std.mem.readInt(u64, (try self.take(8))[0..8], .little);
    }
    /// Reject counts that cannot possibly fit, so a corrupt length can't force a huge alloc.
    fn checkCount(self: Reader, n: u32, per_elem: usize) DeserializeError!usize {
        if (@as(u64, n) * per_elem > self.b.len - self.pos) return error.Truncated;
        return n;
    }
};

pub fn deserialize(alloc: Allocator, bytes: []const u8) DeserializeError!Node {
    var r: Reader = .{ .b = bytes };
    return readNode(alloc, &r);
}

fn readNode(alloc: Allocator, r: *Reader) DeserializeError!Node {
    const op = std.enums.fromInt(ops.OpKind, try r.u8v()) orelse return error.InvalidOpcode;
    const params = try r.u32v();

    var side = try readSide(alloc, r);
    errdefer side.deinit(alloc);

    const n_kids = try r.u8v();
    const kids = try alloc.alloc(Node, n_kids);
    var filled: usize = 0;
    errdefer {
        for (kids[0..filled]) |*c| c.deinit(alloc);
        alloc.free(kids);
    }
    for (kids) |*c| {
        c.* = try readNode(alloc, r);
        filled += 1;
    }

    return .{ .op = op, .params = params, .children = kids, .side = side };
}

fn readSide(alloc: Allocator, r: *Reader) DeserializeError!ops.SideInfo {
    const tag = std.enums.fromInt(SideTag, try r.u8v()) orelse return error.InvalidSideTag;
    switch (tag) {
        .none => return .none,
        .terminal, .rle, .split => {
            const count: usize = @intCast(try r.u64v());
            const bpe = try r.u8v();
            if (tag == .terminal) return .{ .terminal = .{ .count = count, .bits_per_elem = bpe } };
            if (tag == .rle) return .{ .rle = .{ .count = count, .bits_per_elem = bpe } };
            return .{ .split = .{ .count = count, .bits_per_elem = bpe } };
        },
        .huffman => {
            const count: usize = @intCast(try r.u64v());
            const bpe = try r.u8v();
            const n = try r.checkCount(try r.u32v(), 5);
            const entries = try alloc.alloc(codec.HuffmanTable.Entry, n);
            errdefer alloc.free(entries);
            for (entries) |*e| {
                e.sym = try r.u32v();
                e.len = try r.u8v();
            }
            return .{ .huffman = .{
                .table = .{ .entries = entries },
                .count = count,
                .bits_per_elem = bpe,
            } };
        },
        .rans => {
            const count: usize = @intCast(try r.u64v());
            const bpe = try r.u8v();
            const n = try r.checkCount(try r.u32v(), 8);
            const symbols = try alloc.alloc(u32, n);
            errdefer alloc.free(symbols);
            const info = try alloc.alloc(codec.RansSymbol, n);
            errdefer alloc.free(info);
            var cum: u64 = 0;
            for (symbols, info) |*s, *inf| {
                s.* = try r.u32v();
                inf.freq = try r.u32v();
                if (inf.freq == 0 or cum + inf.freq > codec.RANS_PROB_SCALE) return error.InvalidRansTable;
                inf.cum = @intCast(cum);
                cum += inf.freq;
            }
            if ((n == 0 and count != 0) or (n > 0 and cum != codec.RANS_PROB_SCALE))
                return error.InvalidRansTable;
            return .{ .rans = .{
                .table = .{ .symbols = symbols, .info = info },
                .count = count,
                .bits_per_elem = bpe,
            } };
        },
        .codebook => {
            const count: usize = @intCast(try r.u64v());
            const bpe = try r.u8v();
            const n = try r.checkCount(try r.u32v(), 4);
            const syms = try alloc.alloc(u32, n);
            errdefer alloc.free(syms);
            for (syms) |*s| s.* = try r.u32v();
            return .{ .codebook = .{ .syms = syms, .count = count, .bits_per_elem = bpe } };
        },
        .sfloat => {
            const dtype = std.enums.fromInt(Dtype, try r.u8v()) orelse return error.InvalidDtype;
            return .{ .sfloat = .{ .dtype = dtype, .count = @intCast(try r.u64v()) } };
        },
    }
}
