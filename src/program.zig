//! Program tree representation + compress/decompress executor + binary
//! serialization.
//!
//! A program is a tree of Nodes. Inner nodes apply a reversible op and
//! recurse on the op's outputs (e.g. split_float has three children, one per
//! sub-stream). Leaf nodes are terminal entropy coders that produce bytes.
//!
//! Compress: input flows top-down; payloads accumulate on leaves.
//! Decompress: payloads flow up from leaves; the inverse of each op
//! reassembles the input.
//!
//! The total payload of a tree is the concatenation of its leaf payloads in
//! DFS-pre-order, length-prefixed for slicing.

const std = @import("std");
const types = @import("types.zig");
const codec = @import("codec.zig");
const ops = @import("ops.zig");

const Allocator = types.Allocator;
const Stream = types.Stream;
const TensorView = types.TensorView;
const Dtype = types.Dtype;

// ---------- Op kinds + arity table ----------
pub const OpKind = enum(u8) {
    tensor_raw = 0, // terminal, tensor → bytes
    tensor_xor = 1, // tensor → tensor (cross-tensor; needs base)
    split_float = 2, // tensor → 3 streams (sign, exp, mant)
    bitplane_split = 3, // stream → N streams
    delta_encode = 4, // stream → stream
    huffman = 5, // terminal, stream → bytes
    rans = 6, // terminal, stream → bytes
    raw = 7, // terminal, stream → bytes

    pub fn isTerminal(self: OpKind) bool {
        return switch (self) {
            .tensor_raw, .huffman, .rans, .raw => true,
            else => false,
        };
    }

    /// Whether this op consumes a TensorView (vs. a Stream).
    pub fn consumesTensor(self: OpKind) bool {
        return switch (self) {
            .tensor_raw, .tensor_xor, .split_float => true,
            else => false,
        };
    }
};

// ---------- SideInfo (per-op metadata stored in the tree) ----------
pub const SideInfo = union(enum) {
    none,
    split_float: ops.SplitFloatInfo,
    bitplane_split: ops.BitplaneInfo,
    delta_encode: ops.DeltaInfo,
    huffman: HuffmanSide,
    rans: RansSide,
    raw: ops.RawStreamInfo,
    tensor_raw: ops.TensorRawInfo,
    tensor_xor: ops.TensorXorInfo,

    pub const HuffmanSide = struct {
        table: codec.HuffmanTable,
        count: usize,
        bits_per_elem: u8,
    };
    pub const RansSide = struct {
        table: codec.RansTable,
        count: usize,
        bits_per_elem: u8,
    };

    pub fn deinit(self: *SideInfo, alloc: Allocator) void {
        switch (self.*) {
            .huffman => |*h| h.table.deinit(alloc),
            .rans => |*r| r.table.deinit(alloc),
            else => {},
        }
        self.* = .none;
    }
};

// ---------- Node ----------
pub const Node = struct {
    op: OpKind,
    children: []Node = &.{}, // owned
    base_id: u32 = 0, // for tensor_xor
    side_info: SideInfo = .none,
    payload: []u8 = &.{}, // for terminals; owned iff payload_owned
    payload_owned: bool = false,

    pub fn deinit(self: *Node, alloc: Allocator) void {
        for (self.children) |*c| c.deinit(alloc);
        if (self.children.len > 0) alloc.free(self.children);
        self.children = &.{};
        if (self.payload_owned and self.payload.len > 0) alloc.free(self.payload);
        self.payload = &.{};
        self.payload_owned = false;
        self.side_info.deinit(alloc);
    }

    /// Allocate and return a deep copy of this node (sans payload + side_info).
    pub fn cloneSkeleton(self: Node, alloc: Allocator) !Node {
        const kids = try alloc.alloc(Node, self.children.len);
        for (self.children, 0..) |c, i| kids[i] = try c.cloneSkeleton(alloc);
        return .{ .op = self.op, .children = kids, .base_id = self.base_id };
    }
};

pub fn makeLeaf(op: OpKind) Node {
    return .{ .op = op };
}

pub fn makeNode(alloc: Allocator, op: OpKind, children: []const Node) !Node {
    const buf = try alloc.alloc(Node, children.len);
    @memcpy(buf, children);
    return .{ .op = op, .children = buf };
}

// ---------- Compress ----------
//
// Each call mutates `node` to fill side_info (and for terminals, payload).
// `input` is either TensorView (for tensor-consuming ops) or Stream (for
// stream-consuming ops). Use the appropriate variant.
//
// For terminals, payload bytes are stored on the terminal node itself. The
// caller can later collect them via `collectPayloadBytes`.

pub const CompressErr = error{
    OutOfMemory,
    NotFloat16,
    DtypeMismatch,
    ShapeMismatch,
    UnknownBaseId,
    SymbolNotInTable,
};

pub fn compressTensor(
    alloc: Allocator,
    node: *Node,
    input: TensorView,
    bases: []const TensorView,
) CompressErr!void {
    switch (node.op) {
        .tensor_raw => {
            const r = try ops.tensorRawForward(alloc, input);
            node.side_info = .{ .tensor_raw = r.info };
            node.payload = r.bytes;
            node.payload_owned = true;
        },
        .tensor_xor => {
            if (node.base_id >= bases.len) return error.UnknownBaseId;
            const r = try ops.tensorXorForward(alloc, input, bases[node.base_id]);
            node.side_info = .{ .tensor_xor = r.info };
            std.debug.assert(node.children.len == 1);
            var residual = r.residual;
            defer residual.deinit(alloc);
            try compressTensor(alloc, &node.children[0], residual, bases);
        },
        .split_float => {
            const r = try ops.splitFloatForward(alloc, input);
            node.side_info = .{ .split_float = r.info };
            std.debug.assert(node.children.len == 3);
            var sign = r.sign;
            var exp = r.exp;
            var mant = r.mant;
            defer sign.deinit(alloc);
            defer exp.deinit(alloc);
            defer mant.deinit(alloc);
            try compressStream(alloc, &node.children[0], sign);
            try compressStream(alloc, &node.children[1], exp);
            try compressStream(alloc, &node.children[2], mant);
        },
        else => unreachable, // not a tensor-consuming op
    }
}

pub fn compressStream(alloc: Allocator, node: *Node, input: Stream) CompressErr!void {
    switch (node.op) {
        .huffman => {
            var table = try codec.huffmanBuild(alloc, input);
            const bytes = codec.huffmanEncode(alloc, input, table) catch |e| {
                table.deinit(alloc);
                return e;
            };
            node.side_info = .{ .huffman = .{
                .table = table,
                .count = input.count,
                .bits_per_elem = input.bits_per_elem,
            } };
            node.payload = bytes;
            node.payload_owned = true;
        },
        .rans => {
            var table = try codec.ransBuild(alloc, input);
            const bytes = codec.ransEncode(alloc, input, table) catch |e| {
                table.deinit(alloc);
                return e;
            };
            node.side_info = .{ .rans = .{
                .table = table,
                .count = input.count,
                .bits_per_elem = input.bits_per_elem,
            } };
            node.payload = bytes;
            node.payload_owned = true;
        },
        .raw => {
            const r = try ops.rawForward(alloc, input);
            node.side_info = .{ .raw = r.info };
            node.payload = r.bytes;
            node.payload_owned = true;
        },
        .delta_encode => {
            const r = try ops.deltaEncodeForward(alloc, input);
            node.side_info = .{ .delta_encode = r.info };
            std.debug.assert(node.children.len == 1);
            var d = r.out;
            defer d.deinit(alloc);
            try compressStream(alloc, &node.children[0], d);
        },
        .bitplane_split => {
            const r = try ops.bitplaneSplitForward(alloc, input);
            defer alloc.free(r.planes);
            node.side_info = .{ .bitplane_split = r.info };
            std.debug.assert(node.children.len == r.info.n_planes);
            for (r.planes, 0..) |pl, i| {
                var plane = pl;
                defer plane.deinit(alloc);
                try compressStream(alloc, &node.children[i], plane);
            }
        },
        else => unreachable, // not a stream-consuming op
    }
}

// ---------- Decompress ----------

pub const DecompressErr = error{
    OutOfMemory,
    UnknownBaseId,
    DecodeFailed,
    CorruptHuffmanStream,
    CorruptRansStream,
};

pub fn decompressTensor(
    alloc: Allocator,
    node: *const Node,
    bases: []const TensorView,
) DecompressErr!TensorView {
    switch (node.op) {
        .tensor_raw => {
            const info = node.side_info.tensor_raw;
            return ops.tensorRawInverse(alloc, node.payload, info) catch error.DecodeFailed;
        },
        .tensor_xor => {
            if (node.base_id >= bases.len) return error.UnknownBaseId;
            var residual = try decompressTensor(alloc, &node.children[0], bases);
            defer residual.deinit(alloc);
            const info = node.side_info.tensor_xor;
            return ops.tensorXorInverse(alloc, residual, bases[node.base_id], info) catch error.DecodeFailed;
        },
        .split_float => {
            var sign = try decompressStream(alloc, &node.children[0]);
            defer sign.deinit(alloc);
            var exp = try decompressStream(alloc, &node.children[1]);
            defer exp.deinit(alloc);
            var mant = try decompressStream(alloc, &node.children[2]);
            defer mant.deinit(alloc);
            const info = node.side_info.split_float;
            return ops.splitFloatInverse(alloc, sign, exp, mant, info) catch error.DecodeFailed;
        },
        else => unreachable,
    }
}

pub fn decompressStream(alloc: Allocator, node: *const Node) DecompressErr!Stream {
    switch (node.op) {
        .huffman => {
            const h = node.side_info.huffman;
            return codec.huffmanDecode(alloc, node.payload, h.table, h.count, h.bits_per_elem) catch |e| switch (e) {
                error.CorruptHuffmanStream => error.CorruptHuffmanStream,
                error.OutOfMemory => error.OutOfMemory,
            };
        },
        .rans => {
            const r = node.side_info.rans;
            return codec.ransDecode(alloc, node.payload, r.table, r.count, r.bits_per_elem) catch |e| switch (e) {
                error.CorruptRansStream => error.CorruptRansStream,
                error.OutOfMemory => error.OutOfMemory,
            };
        },
        .raw => {
            const info = node.side_info.raw;
            return ops.rawInverse(alloc, node.payload, info) catch error.DecodeFailed;
        },
        .delta_encode => {
            var inner = try decompressStream(alloc, &node.children[0]);
            defer inner.deinit(alloc);
            const info = node.side_info.delta_encode;
            return ops.deltaEncodeInverse(alloc, inner, info) catch error.DecodeFailed;
        },
        .bitplane_split => {
            const info = node.side_info.bitplane_split;
            const planes = try alloc.alloc(Stream, info.n_planes);
            defer {
                for (planes) |*pl| pl.deinit(alloc);
                alloc.free(planes);
            }
            for (node.children, 0..) |*c, i| {
                planes[i] = try decompressStream(alloc, c);
            }
            // Determine output bits-per-elem from n_planes.
            const out_bpe: u8 = info.n_planes;
            return ops.bitplaneSplitInverse(alloc, planes, info, out_bpe) catch error.DecodeFailed;
        },
        else => unreachable,
    }
}

// ---------- Cost estimation (admissible heuristic for A*) ----------
//
// Lower-bound estimate of total payload bits for this node when applied to
// `input`. For terminals we use the codec's Shannon-entropy estimate. For
// inner nodes we sum the children's lower bounds on the corresponding
// output streams.

pub fn costLowerBoundTensor(node: *const Node, input: TensorView, bases: []const TensorView) u64 {
    return switch (node.op) {
        .tensor_raw => @as(u64, input.data.len) * 8,
        .tensor_xor => blk: {
            if (node.base_id >= bases.len) break :blk @as(u64, input.data.len) * 8;
            // XOR doesn't change length; recurse on residual is approximated
            // by entropy of XOR'd bytes, which we don't compute here without
            // doing the xor — return upper bound.
            break :blk @as(u64, input.data.len) * 8;
        },
        .split_float => blk: {
            // Lower bound = sum over children of Shannon entropy of the
            // matching output stream. We can't actually run forward without
            // allocating, so we use a rough fp-data heuristic.
            // Float16/bf16: sign ~ 1 bit/elem, exp ~ entropy * elem, mant ~ uniform.
            // For an admissible bound we assume best-case 1+1+1=3 bits per elem.
            break :blk input.numel() * 3;
        },
        else => @as(u64, input.data.len) * 8,
    };
}

pub fn costLowerBoundStream(node: *const Node, input: Stream) u64 {
    return switch (node.op) {
        .huffman => codec.huffmanCostBits(input),
        .rans => codec.ransCostBits(input),
        .raw => @as(u64, input.data.len) * 8,
        .delta_encode, .bitplane_split => @as(u64, input.count) * @as(u64, input.bits_per_elem) / 2, // optimistic
        else => unreachable,
    };
}

// ---------- Collect leaf payloads in DFS-pre-order ----------
pub fn collectPayloadBytes(alloc: Allocator, node: *const Node) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(alloc);
    try appendPayloads(alloc, &out, node);
    return out.toOwnedSlice(alloc);
}

fn appendPayloads(alloc: Allocator, out: *std.ArrayList(u8), node: *const Node) !void {
    if (node.op.isTerminal()) {
        var len_buf: [8]u8 = undefined;
        std.mem.writeInt(u64, &len_buf, @intCast(node.payload.len), .little);
        try out.appendSlice(alloc, &len_buf);
        try out.appendSlice(alloc, node.payload);
        return;
    }
    for (node.children) |*c| try appendPayloads(alloc, out, c);
}

/// Inverse of collectPayloadBytes: distribute a flat payload buffer over the
/// terminals of `node`. Mutates terminal payload pointers (non-owning view
/// into `payload`).
pub fn distributePayloadBytes(node: *Node, payload: []const u8) !void {
    var pos: usize = 0;
    try distributeRec(node, payload, &pos);
    if (pos != payload.len) return error.PayloadOverrun;
}

fn distributeRec(node: *Node, payload: []const u8, pos: *usize) !void {
    if (node.op.isTerminal()) {
        if (pos.* + 8 > payload.len) return error.PayloadUnderrun;
        const len: u64 = std.mem.readInt(u64, payload[pos.*..][0..8], .little);
        pos.* += 8;
        if (pos.* + len > payload.len) return error.PayloadUnderrun;
        node.payload = @constCast(payload[pos.* .. pos.* + @as(usize, @intCast(len))]);
        node.payload_owned = false;
        pos.* += @intCast(len);
        return;
    }
    for (node.children) |*c| try distributeRec(c, payload, pos);
}

// ---------- Program tree binary serialization (skeleton + side_info, no payload) ----------
//
// Binary format:
//   u8  op
//   u32 base_id
//   variable side_info (per-op encoding, see writeSideInfo)
//   u8  n_children
//   ... children recursively ...

pub fn serializeProgram(alloc: Allocator, node: *const Node) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(alloc);
    try writeNode(alloc, &out, node);
    return out.toOwnedSlice(alloc);
}

fn writeNode(alloc: Allocator, out: *std.ArrayList(u8), node: *const Node) !void {
    try out.append(alloc, @intFromEnum(node.op));
    var buf4: [4]u8 = undefined;
    std.mem.writeInt(u32, &buf4, node.base_id, .little);
    try out.appendSlice(alloc, &buf4);
    try writeSideInfo(alloc, out, node);
    try out.append(alloc, @intCast(node.children.len));
    for (node.children) |*c| try writeNode(alloc, out, c);
}

fn writeSideInfo(alloc: Allocator, out: *std.ArrayList(u8), node: *const Node) !void {
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
            try writeU32(alloc, out, @intCast(i.table.entries.len));
            for (i.table.entries) |e| {
                try writeU32(alloc, out, e.sym);
                try out.append(alloc, e.len);
            }
        },
        .rans => |i| {
            try writeU64(alloc, out, i.count);
            try out.append(alloc, i.bits_per_elem);
            try writeU32(alloc, out, @intCast(i.table.symbols.len));
            for (i.table.symbols, i.table.info) |sym, inf| {
                try writeU32(alloc, out, sym);
                try writeU32(alloc, out, inf.freq);
            }
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

pub const ProgramReader = struct {
    bytes: []const u8,
    pos: usize = 0,

    pub fn readU8(self: *ProgramReader) u8 {
        const v = self.bytes[self.pos];
        self.pos += 1;
        return v;
    }
    pub fn readU32(self: *ProgramReader) u32 {
        const v = std.mem.readInt(u32, self.bytes[self.pos..][0..4], .little);
        self.pos += 4;
        return v;
    }
    pub fn readU64(self: *ProgramReader) u64 {
        const v = std.mem.readInt(u64, self.bytes[self.pos..][0..8], .little);
        self.pos += 8;
        return v;
    }
};

pub fn deserializeProgram(alloc: Allocator, bytes: []const u8) !Node {
    var r: ProgramReader = .{ .bytes = bytes };
    return try readNode(alloc, &r);
}

fn readNode(alloc: Allocator, r: *ProgramReader) !Node {
    const op: OpKind = @enumFromInt(r.readU8());
    const base_id = r.readU32();
    var side_info: SideInfo = .none;
    switch (op) {
        .split_float => {
            const dtype: Dtype = @enumFromInt(r.readU8());
            const exp_bits = r.readU8();
            const mant_bits = r.readU8();
            const ndim = r.readU8();
            var info: ops.SplitFloatInfo = .{
                .dtype = dtype,
                .exp_bits = exp_bits,
                .mant_bits = mant_bits,
                .ndim = ndim,
                .shape = .{0} ** 8,
            };
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
            const n_entries = r.readU32();
            const entries = try alloc.alloc(codec.HuffmanTable.Entry, n_entries);
            for (entries) |*e| {
                e.sym = r.readU32();
                e.len = r.readU8();
            }
            side_info = .{ .huffman = .{
                .table = .{ .entries = entries },
                .count = count,
                .bits_per_elem = bpe,
            } };
        },
        .rans => {
            const count = r.readU64();
            const bpe = r.readU8();
            const n_entries = r.readU32();
            const symbols = try alloc.alloc(u32, n_entries);
            const info = try alloc.alloc(codec.RansSymbol, n_entries);
            var cum: u32 = 0;
            for (symbols, info) |*s, *inf| {
                s.* = r.readU32();
                inf.freq = r.readU32();
                inf.cum = cum;
                cum += inf.freq;
            }
            side_info = .{ .rans = .{
                .table = .{ .symbols = symbols, .info = info },
                .count = count,
                .bits_per_elem = bpe,
            } };
        },
        .raw => {
            const count = r.readU64();
            const bpe = r.readU8();
            side_info = .{ .raw = .{ .count = count, .bits_per_elem = bpe } };
        },
        .tensor_raw => {
            const dtype: Dtype = @enumFromInt(r.readU8());
            const ndim = r.readU8();
            var info: ops.TensorRawInfo = .{ .dtype = dtype, .ndim = ndim, .shape = .{0} ** 8 };
            for (0..ndim) |k| info.shape[k] = r.readU64();
            side_info = .{ .tensor_raw = info };
        },
        .tensor_xor => {
            const dtype: Dtype = @enumFromInt(r.readU8());
            const ndim = r.readU8();
            var info: ops.TensorXorInfo = .{ .dtype = dtype, .ndim = ndim, .shape = .{0} ** 8 };
            for (0..ndim) |k| info.shape[k] = r.readU64();
            side_info = .{ .tensor_xor = info };
        },
    }
    const n_kids = r.readU8();
    const kids = try alloc.alloc(Node, n_kids);
    for (kids) |*c| c.* = try readNode(alloc, r);
    return .{ .op = op, .children = kids, .base_id = base_id, .side_info = side_info };
}
