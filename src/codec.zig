//! Entropy coders: Huffman (canonical) + rANS (range Asymmetric Numeral Systems).
//!
//! Both operate on streams of u8/u16/u32 symbols. Each encoder produces a
//! compact byte payload + a small "table" stored as side_info.
//!
//! Design notes
//! ============
//! * Huffman is canonical: we transmit only the per-symbol bit length, not
//!   the codes themselves. Decoder reconstructs codes deterministically.
//! * rANS uses an 8-bit-renormalized 32-bit state with a 14-bit precision
//!   frequency table. This is a common, simple choice that's close to
//!   Shannon entropy and faster than Huffman for skewed distributions.
//! * Both encoders are lossless and bit-exact by construction.

const std = @import("std");
const types = @import("types.zig");
const Allocator = types.Allocator;
const Stream = types.Stream;

// ==================== BitWriter / BitReader (MSB-first) ====================
//
// MSB-first: writeBits(value, n) places bit (n-1) of value first in the stream,
// then bit (n-2), ..., then bit 0. This is the natural order for prefix codes
// because reading the stream left-to-right yields the code bits in the order
// you'd traverse a Huffman tree from the root.
pub const BitWriter = struct {
    out: *std.ArrayList(u8),
    alloc: Allocator,
    /// 64-bit accumulator. Bits are filled from the MSB downward; `n` is the
    /// count already written. We flush whole bytes from the top of `cur` when
    /// `n >= 8`.
    cur: u64 = 0,
    n: u8 = 0,

    pub fn init(alloc: Allocator, out: *std.ArrayList(u8)) BitWriter {
        return .{ .out = out, .alloc = alloc };
    }

    /// Append the low `nbits` of `value`, MSB-first. Up to 56 bits per call.
    pub fn writeBits(self: *BitWriter, value: u64, nbits: u8) !void {
        // Mask off the relevant bits and shift into the accumulator's free area.
        const masked: u64 = if (nbits == 64) value else value & ((@as(u64, 1) << @intCast(nbits)) - 1);
        const shift: u8 = 64 - self.n - nbits;
        self.cur |= masked << @intCast(shift);
        self.n += nbits;
        // Flush whole bytes from the top.
        while (self.n >= 8) {
            const byte: u8 = @intCast(self.cur >> 56);
            try self.out.append(self.alloc, byte);
            self.cur <<= 8;
            self.n -= 8;
        }
    }

    pub fn flush(self: *BitWriter) !void {
        if (self.n > 0) {
            const byte: u8 = @intCast(self.cur >> 56);
            try self.out.append(self.alloc, byte);
            self.cur = 0;
            self.n = 0;
        }
    }
};

pub const BitReader = struct {
    bytes: []const u8,
    byte_pos: usize = 0,
    bits: u64 = 0,
    nbits: u8 = 0,

    pub fn init(bytes: []const u8) BitReader {
        return .{ .bytes = bytes };
    }

    inline fn fill(self: *BitReader) void {
        while (self.nbits <= 56 and self.byte_pos < self.bytes.len) {
            self.bits |= @as(u64, self.bytes[self.byte_pos]) << @intCast(56 - self.nbits);
            self.nbits += 8;
            self.byte_pos += 1;
        }
    }

    pub inline fn readBits(self: *BitReader, n: u8) ?u64 {
        std.debug.assert(n > 0 and n <= 32);
        self.fill();
        if (self.nbits < n) return null;
        const value = self.bits >> @intCast(64 - n);
        self.bits <<= @intCast(n);
        self.nbits -= n;
        return value;
    }

    pub inline fn peek12(self: *BitReader) u12 {
        self.fill();
        return @truncate(self.bits >> 52);
    }
};

// ==================== Canonical Huffman ====================

/// Code lengths up to this decode through a flat lookup table.
const LUT_BITS = 12;

pub const HuffmanTable = struct {
    /// code_length[symbol] = bits used; 0 means "symbol unused".
    /// Symbols are u32 keyed; sparse via a sorted (sym, len) list.
    entries: []Entry, // sorted by (length asc, symbol asc) — canonical order

    pub const Entry = struct { sym: u32, len: u8 };

    pub fn deinit(self: *HuffmanTable, alloc: Allocator) void {
        alloc.free(self.entries);
        self.entries = &.{};
    }

    pub fn clone(self: HuffmanTable, alloc: Allocator) !HuffmanTable {
        return .{ .entries = try alloc.dupe(Entry, self.entries) };
    }
};

/// Histogram a stream into (sym, count) pairs. For 8/16-bit storage the
/// fast path uses 4 parallel counter arrays to break the read-modify-write
/// dependency between iterations (a 2-4× speed-up on a single core, no
/// SIMD needed — modern OoO engines pipeline the independent increments).
/// For wider alphabets, falls back to a HashMap.
pub const Histogram = struct {
    /// (sym, count) pairs, ascending sym; only present symbols are listed.
    pairs: []Pair,
    pub const Pair = struct { sym: u32, count: u64 };

    pub fn requiredBits(self: Histogram) u8 {
        if (self.pairs.len == 0) return 1;
        const maximum = self.pairs[self.pairs.len - 1].sym;
        return if (maximum == 0) 1 else @intCast(32 - @clz(maximum));
    }

    pub fn deinit(self: *Histogram, alloc: Allocator) void {
        alloc.free(self.pairs);
        self.pairs = &.{};
    }
};

/// Below this length a dense alphabet-sized histogram costs far more to zero
/// than the stream costs to sort. The search evaluates many short streams, so
/// this path dominates in practice.
const SMALL_STREAM: usize = 8192;
/// Physical entropy candidates with larger alphabets cannot justify their
/// table memory in this implementation. `Lit` remains total through raw and
/// bitpack, while Huffman/rANS report themselves inapplicable.
pub const MAX_HISTOGRAM_SYMBOLS: usize = 65_536;

fn histogramBySort(alloc: Allocator, stream: Stream) !Histogram {
    const tmp = try alloc.alloc(u32, stream.count);
    defer alloc.free(tmp);
    for (0..stream.count) |i| tmp[i] = stream.getU32(i);
    std.mem.sort(u32, tmp, {}, std.sort.asc(u32));

    var unique: usize = 0;
    for (tmp, 0..) |v, i| {
        if (i == 0 or v != tmp[i - 1]) unique += 1;
    }

    const pairs = try alloc.alloc(Histogram.Pair, unique);
    var j: usize = 0;
    var i: usize = 0;
    while (i < tmp.len) {
        var run: usize = 1;
        while (i + run < tmp.len and tmp[i + run] == tmp[i]) run += 1;
        pairs[j] = .{ .sym = tmp[i], .count = @intCast(run) };
        j += 1;
        i += run;
    }
    return .{ .pairs = pairs };
}

pub fn buildHistogram(alloc: Allocator, stream: Stream) !Histogram {
    const bpe_pow2 = types.roundUpToPow2(stream.bits_per_elem);
    if (bpe_pow2 != 8 and stream.count < SMALL_STREAM) return histogramBySort(alloc, stream);
    if (bpe_pow2 == 8) {
        var c0: [256]u64 = .{0} ** 256;
        var c1: [256]u64 = .{0} ** 256;
        var c2: [256]u64 = .{0} ** 256;
        var c3: [256]u64 = .{0} ** 256;
        const data = stream.data;
        var i: usize = 0;
        const n = stream.count;
        while (n - i >= 4) : (i += 4) {
            c0[data[i + 0]] += 1;
            c1[data[i + 1]] += 1;
            c2[data[i + 2]] += 1;
            c3[data[i + 3]] += 1;
        }
        while (i < n) : (i += 1) c0[data[i]] += 1;
        var unique: usize = 0;
        var k: usize = 0;
        while (k < 256) : (k += 1) {
            if (c0[k] + c1[k] + c2[k] + c3[k] > 0) unique += 1;
        }
        const pairs = try alloc.alloc(Histogram.Pair, unique);
        var j: usize = 0;
        k = 0;
        while (k < 256) : (k += 1) {
            const tot = c0[k] + c1[k] + c2[k] + c3[k];
            if (tot > 0) {
                pairs[j] = .{ .sym = @intCast(k), .count = tot };
                j += 1;
            }
        }
        return .{ .pairs = pairs };
    }
    if (bpe_pow2 == 16) {
        // Four u64 lanes avoid counter overflow on very large trusted inputs.
        // Keep the 2 MiB table off the stack.
        const slots: usize = 65536;
        const pool = try alloc.alloc(u64, slots * 4);
        defer alloc.free(pool);
        @memset(pool, 0);
        const c0 = pool[0..slots];
        const c1 = pool[slots .. 2 * slots];
        const c2 = pool[2 * slots .. 3 * slots];
        const c3 = pool[3 * slots .. 4 * slots];
        var i: usize = 0;
        const n = stream.count;
        while (n - i >= 4) : (i += 4) {
            c0[std.mem.readInt(u16, stream.data[(i + 0) * 2 ..][0..2], .little)] += 1;
            c1[std.mem.readInt(u16, stream.data[(i + 1) * 2 ..][0..2], .little)] += 1;
            c2[std.mem.readInt(u16, stream.data[(i + 2) * 2 ..][0..2], .little)] += 1;
            c3[std.mem.readInt(u16, stream.data[(i + 3) * 2 ..][0..2], .little)] += 1;
        }
        while (i < n) : (i += 1) c0[std.mem.readInt(u16, stream.data[i * 2 ..][0..2], .little)] += 1;
        var unique: usize = 0;
        var k: usize = 0;
        while (k < slots) : (k += 1) {
            if (c0[k] + c1[k] + c2[k] + c3[k] > 0) unique += 1;
        }
        const pairs = try alloc.alloc(Histogram.Pair, unique);
        var j: usize = 0;
        k = 0;
        while (k < slots) : (k += 1) {
            const tot = c0[k] + c1[k] + c2[k] + c3[k];
            if (tot > 0) {
                pairs[j] = .{ .sym = @intCast(k), .count = tot };
                j += 1;
            }
        }
        return .{ .pairs = pairs };
    }
    // 32-bit alphabet: HashMap fallback.
    var map: std.AutoHashMap(u32, u64) = .init(alloc);
    defer map.deinit();
    for (0..stream.count) |i| {
        const sym = stream.getU32(i);
        const gop = try map.getOrPut(sym);
        if (!gop.found_existing) {
            if (map.count() > MAX_HISTOGRAM_SYMBOLS)
                return error.AlphabetTooLarge;
            gop.value_ptr.* = 0;
        }
        gop.value_ptr.* += 1;
    }
    const pairs = try alloc.alloc(Histogram.Pair, map.count());
    var j: usize = 0;
    var it = map.iterator();
    while (it.next()) |kv| : (j += 1) {
        pairs[j] = .{ .sym = kv.key_ptr.*, .count = kv.value_ptr.* };
    }
    std.mem.sort(Histogram.Pair, pairs, {}, struct {
        fn less(_: void, l: Histogram.Pair, r: Histogram.Pair) bool {
            return l.sym < r.sym;
        }
    }.less);
    return .{ .pairs = pairs };
}

/// Build a canonical Huffman table accepted by the decoder.
pub fn huffmanBuild(alloc: Allocator, stream: Stream) !HuffmanTable {
    var hist = try buildHistogram(alloc, stream);
    defer hist.deinit(alloc);
    return huffmanFromHist(alloc, hist, stream.bits_per_elem);
}

/// Code lengths from a histogram alone. No data pass.
pub fn huffmanFromHist(alloc: Allocator, hist: Histogram, bits_per_elem: u8) !HuffmanTable {
    _ = bits_per_elem;

    if (hist.pairs.len > MAX_HISTOGRAM_SYMBOLS)
        return error.AlphabetTooLarge;
    if (hist.pairs.len == 0) {
        return .{ .entries = try alloc.alloc(HuffmanTable.Entry, 0) };
    }
    if (hist.pairs.len == 1) {
        // Degenerate: a single symbol gets a 1-bit code (canonical).
        const e = try alloc.alloc(HuffmanTable.Entry, 1);
        e[0] = .{ .sym = hist.pairs[0].sym, .len = 1 };
        return .{ .entries = e };
    }

    // Build the ordinary optimal tree; overlong codes are rejected below.
    const Node = struct { freq: u64, left: ?*@This(), right: ?*@This(), sym: u32, is_leaf: bool };
    var arena: std.heap.ArenaAllocator = .init(alloc);
    defer arena.deinit();
    const a = arena.allocator();

    const Item = struct { freq: u64, idx: u32 };
    const Less = struct {
        fn less(_: void, lhs: Item, rhs: Item) std.math.Order {
            if (lhs.freq != rhs.freq) return std.math.order(lhs.freq, rhs.freq);
            return std.math.order(lhs.idx, rhs.idx);
        }
    };

    var heap: std.PriorityQueue(Item, void, Less.less) = .empty;
    defer heap.deinit(alloc);

    var nodes: std.ArrayList(*Node) = .empty;
    defer nodes.deinit(alloc);

    for (hist.pairs) |p| {
        const node = try a.create(Node);
        node.* = .{ .freq = p.count, .left = null, .right = null, .sym = p.sym, .is_leaf = true };
        try nodes.append(alloc, node);
        try heap.push(alloc, .{ .freq = p.count, .idx = @intCast(nodes.items.len - 1) });
    }

    while (heap.count() > 1) {
        const lo = heap.pop().?;
        const hi = heap.pop().?;
        const merged = try a.create(Node);
        merged.* = .{
            .freq = lo.freq + hi.freq,
            .left = nodes.items[lo.idx],
            .right = nodes.items[hi.idx],
            .sym = 0,
            .is_leaf = false,
        };
        try nodes.append(alloc, merged);
        try heap.push(alloc, .{ .freq = merged.freq, .idx = @intCast(nodes.items.len - 1) });
    }

    // Step 3: walk tree to extract code lengths.
    var lengths: std.AutoHashMap(u32, u8) = .init(alloc);
    defer lengths.deinit();

    const root = nodes.items[heap.pop().?.idx];
    try walkLengths(root, 0, &lengths);

    // Step 4: build sorted entry list (canonical).
    const entries = try alloc.alloc(HuffmanTable.Entry, lengths.count());
    var i: usize = 0;
    var lit = lengths.iterator();
    while (lit.next()) |kv| : (i += 1) {
        entries[i] = .{ .sym = kv.key_ptr.*, .len = kv.value_ptr.* };
    }
    std.mem.sort(HuffmanTable.Entry, entries, {}, struct {
        fn less(_: void, lhs: HuffmanTable.Entry, rhs: HuffmanTable.Entry) bool {
            if (lhs.len != rhs.len) return lhs.len < rhs.len;
            return lhs.sym < rhs.sym;
        }
    }.less);

    return .{ .entries = entries };
}

fn walkLengths(node: anytype, depth: u8, lengths: *std.AutoHashMap(u32, u8)) !void {
    if (depth > 32) return error.HuffmanCodeTooLong;
    if (node.is_leaf) {
        try lengths.put(node.sym, if (depth == 0) 1 else depth);
        return;
    }
    if (node.left) |l| try walkLengths(l, depth + 1, lengths);
    if (node.right) |r| try walkLengths(r, depth + 1, lengths);
}

/// Generate canonical codes from sorted (sym,len) table.
pub fn huffmanEncode(alloc: Allocator, stream: Stream, table: HuffmanTable) ![]u8 {
    var output: std.ArrayList(u8) = .empty;
    errdefer output.deinit(alloc);
    try output.ensureTotalCapacity(alloc, stream.count);
    try huffmanEncodeInto(alloc, stream, table, &output);
    return output.toOwnedSlice(alloc);
}

fn huffmanEncodeInto(
    alloc: Allocator,
    stream: Stream,
    table: HuffmanTable,
    output: *std.ArrayList(u8),
) !void {
    var writer = BitWriter.init(alloc, output);
    try huffmanEncodeWithWriter(alloc, stream, table, &writer);
}

pub fn huffmanEncodeIntoSlice(
    alloc: Allocator,
    stream: Stream,
    table: HuffmanTable,
    output: []u8,
) !void {
    var writer = FixedBitWriter{ .output = output };
    try huffmanEncodeWithWriter(alloc, stream, table, &writer);
    if (writer.position != output.len)
        return error.HuffmanPayloadSizeMismatch;
}

const FixedBitWriter = struct {
    output: []u8,
    position: usize = 0,
    cur: u64 = 0,
    n: u8 = 0,

    inline fn writeBits(
        self: *FixedBitWriter,
        value: u64,
        nbits: u8,
    ) !void {
        const masked = value & ((@as(u64, 1) << @intCast(nbits)) - 1);
        self.cur |= masked << @intCast(64 - self.n - nbits);
        self.n += nbits;
        while (self.n >= 8) {
            self.output[self.position] = @intCast(self.cur >> 56);
            self.position += 1;
            self.cur <<= 8;
            self.n -= 8;
        }
    }

    fn flush(self: *FixedBitWriter) !void {
        if (self.n == 0) return;
        self.output[self.position] = @intCast(self.cur >> 56);
        self.position += 1;
        self.cur = 0;
        self.n = 0;
    }
};

inline fn writeHuffmanSymbol(
    writer: anytype,
    packed_codes: []const u64,
    symbol: u32,
) !void {
    const packed_code = packed_codes[symbol];
    const len: u8 = @intCast(packed_code >> 32);
    if (len == 0) return error.SymbolNotInTable;
    try writer.writeBits(
        packed_code & std.math.maxInt(u32),
        len,
    );
}

fn huffmanEncodeWithWriter(
    alloc: Allocator,
    stream: Stream,
    table: HuffmanTable,
    writer: anytype,
) !void {
    const max_sym: u32 = blk: {
        var m: u32 = 0;
        for (table.entries) |e| if (e.sym > m) {
            m = e.sym;
        };
        break :blk m;
    };
    const direct_path: bool = max_sym <= 0xFFFF;
    var packed_codes: []u64 = &.{};
    defer if (direct_path) alloc.free(packed_codes);
    const Code = struct { bits: u64, len: u8 };
    var code_map: std.AutoHashMap(u32, Code) = .init(alloc);
    defer code_map.deinit();

    var code: u64 = 0;
    var previous_len: u8 = if (table.entries.len == 0)
        0
    else
        table.entries[0].len;
    if (direct_path) {
        const n_slots: usize = @as(usize, max_sym) + 1;
        packed_codes = try alloc.alloc(u64, n_slots);
        @memset(packed_codes, 0);
        for (table.entries) |e| {
            if (e.len > previous_len) {
                code <<= @intCast(e.len - previous_len);
                previous_len = e.len;
            }
            packed_codes[e.sym] = (@as(u64, e.len) << 32) |
                @as(u32, @truncate(code));
            code += 1;
        }
    } else {
        for (table.entries) |e| {
            if (e.len > previous_len) {
                code <<= @intCast(e.len - previous_len);
                previous_len = e.len;
            }
            try code_map.put(e.sym, .{ .bits = code, .len = e.len });
            code += 1;
        }
    }

    if (direct_path) {
        switch (types.roundUpToPow2(stream.bits_per_elem)) {
            8 => for (stream.data[0..stream.count]) |symbol|
                try writeHuffmanSymbol(writer, packed_codes, symbol),
            16 => for (0..stream.count) |i| {
                const symbol = std.mem.readInt(
                    u16,
                    stream.data[i * 2 ..][0..2],
                    .little,
                );
                try writeHuffmanSymbol(writer, packed_codes, symbol);
            },
            32 => for (0..stream.count) |i| {
                const symbol = std.mem.readInt(
                    u32,
                    stream.data[i * 4 ..][0..4],
                    .little,
                );
                try writeHuffmanSymbol(writer, packed_codes, symbol);
            },
            else => unreachable,
        }
    } else {
        for (0..stream.count) |i| {
            const sym = stream.getU32(i);
            const symbol_code = code_map.get(sym) orelse
                return error.SymbolNotInTable;
            try writer.writeBits(symbol_code.bits, symbol_code.len);
        }
    }
    try writer.flush();
}

pub fn huffmanDecode(alloc: Allocator, payload: []const u8, table: HuffmanTable, count: usize, bits_per_elem: u8) !Stream {
    if (bits_per_elem == 0 or bits_per_elem > 32) return error.CorruptHuffmanStream;
    const elem_bytes: usize = switch (types.roundUpToPow2(bits_per_elem)) {
        8 => 1,
        16 => 2,
        32 => 4,
        else => unreachable,
    };
    if (count != 0 and table.entries.len == 0) return error.CorruptHuffmanStream;

    var counts = [_]u32{0} ** 33;
    var max_len: u8 = 0;
    const max_symbol: u32 = if (bits_per_elem == 32)
        std.math.maxInt(u32)
    else
        (@as(u32, 1) << @intCast(bits_per_elem)) - 1;
    for (table.entries, 0..) |e, i| {
        if (e.len == 0 or e.len > 32 or e.sym > max_symbol) return error.CorruptHuffmanStream;
        if (i != 0) {
            const prev = table.entries[i - 1];
            if (e.len < prev.len or (e.len == prev.len and e.sym <= prev.sym))
                return error.CorruptHuffmanStream;
        }
        counts[e.len] += 1;
        max_len = e.len;
    }

    var first_code = [_]u64{0} ** 33;
    var first_idx = [_]u32{0} ** 33;
    var code: u64 = 0;
    var idx: u32 = 0;
    for (1..@as(usize, max_len) + 1) |len| {
        if (code + counts[len] > @as(u64, 1) << @intCast(len))
            return error.CorruptHuffmanStream;
        first_code[len] = code;
        first_idx[len] = idx;
        idx += counts[len];
        code = (code + counts[len]) << 1;
    }

    const min_len = if (table.entries.len == 0) 0 else table.entries[0].len;
    if (@as(u128, count) * min_len > @as(u128, payload.len) * 8)
        return error.CorruptHuffmanStream;
    const output_len = std.math.mul(usize, count, elem_bytes) catch return error.CorruptHuffmanStream;
    const buf = try alloc.alloc(u8, output_len);
    var s: Stream = .{ .data = buf, .count = count, .bits_per_elem = bits_per_elem };
    errdefer s.deinit(alloc);
    if (count == 0) return s;

    var br = BitReader.init(payload);

    var lut_sym: [1 << LUT_BITS]u32 = undefined;
    var lut_len: [1 << LUT_BITS]u8 = @splat(0);
    var next_code: u64 = 0;
    var prev_len: u8 = table.entries[0].len;
    for (table.entries) |e| {
        if (e.len > prev_len) {
            next_code <<= @intCast(e.len - prev_len);
            prev_len = e.len;
        }
        if (e.len <= LUT_BITS) {
            const shift: u5 = @intCast(LUT_BITS - e.len);
            const base: usize = @intCast(next_code << shift);
            const end = base + (@as(usize, 1) << shift);
            @memset(lut_sym[base..end], e.sym);
            @memset(lut_len[base..end], e.len);
        }
        next_code += 1;
    }

    for (0..count) |i| {
        const prefix = br.peek12();
        const short_len = lut_len[prefix];
        if (short_len != 0) {
            if (br.readBits(short_len) == null) return error.CorruptHuffmanStream;
            s.setU32(i, lut_sym[prefix]);
            continue;
        }
        if (max_len <= LUT_BITS) return error.CorruptHuffmanStream;
        var fallback_code = br.readBits(LUT_BITS) orelse return error.CorruptHuffmanStream;
        var len: u8 = LUT_BITS;
        while (len < max_len) {
            fallback_code = (fallback_code << 1) |
                (br.readBits(1) orelse return error.CorruptHuffmanStream);
            len += 1;
            const n = counts[len];
            if (n != 0 and fallback_code >= first_code[len] and fallback_code - first_code[len] < n) {
                s.setU32(i, table.entries[first_idx[len] + @as(u32, @intCast(fallback_code - first_code[len]))].sym);
                break;
            }
        } else return error.CorruptHuffmanStream;
    }
    return s;
}

// ==================== Bit packing ====================
//
// Each element is written as exactly `width` bits, MSB-first. No table.

pub fn bitpackEncode(alloc: Allocator, stream: Stream, width: u8) ![]u8 {
    if (width == 0 or width > 32) return error.InvalidBitpackWidth;

    var out: std.ArrayList(u8) = .empty;
    try out.ensureTotalCapacity(
        alloc,
        try bitpackByteCount(stream.count, width),
    );
    defer out.deinit(alloc);
    var bw = BitWriter.init(alloc, &out);

    const m: u64 = (@as(u64, 1) << @intCast(width)) - 1;
    for (0..stream.count) |i| {
        try bw.writeBits(@as(u64, stream.getU32(i)) & m, width);
    }
    try bw.flush();

    return out.toOwnedSlice(alloc);
}

pub fn bitpackDecode(alloc: Allocator, payload: []const u8, width: u8, count: usize, out_bpe: u8) !Stream {
    if (width == 0 or width > 32) return error.InvalidBitpackWidth;
    const required_payload = try bitpackByteCount(count, width);
    if (payload.len < required_payload) return error.CorruptBitpackStream;

    var s = try Stream.initUninitialized(alloc, count, out_bpe);
    errdefer s.deinit(alloc);
    if (count == 0) return s;

    var br = BitReader.init(payload);
    for (0..count) |i| {
        s.setU32(i, @intCast(br.readBits(width).?));
    }
    return s;
}

pub fn bitpackCostBits(stream: Stream, width: u8) u64 {
    const count = std.math.cast(u64, stream.count) orelse
        return std.math.maxInt(u64) - 7;
    const bits = std.math.mul(u64, count, width) catch
        return std.math.maxInt(u64) - 7;
    const rounded = std.math.add(u64, bits, 7) catch
        return std.math.maxInt(u64) - 7;
    return (rounded / 8) * 8;
}

fn bitpackByteCount(count: usize, width: u8) !usize {
    if (width == 0 or width > 32) return error.InvalidBitpackWidth;
    const count_u64 = std.math.cast(u64, count) orelse
        return error.BitpackSizeOverflow;
    const bit_count = std.math.mul(
        u64,
        count_u64,
        @as(u64, width),
    ) catch return error.BitpackSizeOverflow;
    const rounded = std.math.add(u64, bit_count, 7) catch
        return error.BitpackSizeOverflow;
    return std.math.cast(usize, rounded / 8) orelse
        error.BitpackSizeOverflow;
}

// ==================== rANS ====================
//
// 32-bit state, 8-bit renormalization, 14-bit probabilities.
// Reference: Pasco/Duda; this is the "byte-streaming" variant.

pub const RANS_PROB_BITS: u6 = 14;
pub const RANS_PROB_SCALE: u32 = 1 << RANS_PROB_BITS;
pub const RANS_L: u32 = 1 << 23; // lower bound on state
pub const RANS_BYTE_M: u32 = 1 << 8;

pub const RansSymbol = struct {
    freq: u32, // quantized to RANS_PROB_SCALE
    cum: u32, // cumulative
};

const RansEncoderSymbol = struct {
    x_max: u32,
    reciprocal: u32,
    bias: u32,
    complement: u16,
    shift: u5,

    fn init(info: RansSymbol) RansEncoderSymbol {
        std.debug.assert(info.freq > 0);
        std.debug.assert(info.cum + info.freq <= RANS_PROB_SCALE);

        if (info.freq == 1) return .{
            .x_max = ((RANS_L >> RANS_PROB_BITS) << 8),
            .reciprocal = std.math.maxInt(u32),
            .bias = info.cum + RANS_PROB_SCALE - 1,
            .complement = RANS_PROB_SCALE - 1,
            .shift = 0,
        };

        const shift = std.math.log2_int_ceil(u32, info.freq);
        // The encoder keeps state below 2^31, making this reciprocal exact.
        const numerator = (@as(u64, 1) << @intCast(shift + 31)) +
            info.freq - 1;
        return .{
            .x_max = ((RANS_L >> RANS_PROB_BITS) << 8) * info.freq,
            .reciprocal = @intCast(numerator / info.freq),
            .bias = info.cum,
            .complement = @intCast(RANS_PROB_SCALE - info.freq),
            .shift = @intCast(shift - 1),
        };
    }

    inline fn advance(self: RansEncoderSymbol, state: u32) u32 {
        const quotient: u32 = @intCast(
            (@as(u64, state) * self.reciprocal) >> 32,
        );
        return state + self.bias +
            (quotient >> self.shift) * @as(u32, self.complement);
    }
};

pub const RansTable = struct {
    symbols: []u32, // present symbol values, sorted ascending
    info: []RansSymbol, // 1:1 with `symbols`

    pub fn deinit(self: *RansTable, alloc: Allocator) void {
        alloc.free(self.symbols);
        alloc.free(self.info);
        self.symbols = &.{};
        self.info = &.{};
    }

    pub fn clone(self: RansTable, alloc: Allocator) !RansTable {
        const syms = try alloc.dupe(u32, self.symbols);
        errdefer alloc.free(syms);
        return .{ .symbols = syms, .info = try alloc.dupe(RansSymbol, self.info) };
    }
};

pub fn ransBuild(alloc: Allocator, stream: Stream) !RansTable {
    var hist = try buildHistogram(alloc, stream);
    defer hist.deinit(alloc);
    return ransFromHist(alloc, hist, stream.count);
}

/// Quantized frequency table from a histogram alone. No data pass.
pub fn ransFromHist(alloc: Allocator, hist: Histogram, count: usize) !RansTable {
    if (hist.pairs.len == 0) {
        return .{
            .symbols = try alloc.alloc(u32, 0),
            .info = try alloc.alloc(RansSymbol, 0),
        };
    }

    // Every observed symbol needs at least one slot, so an alphabet wider than
    // the probability scale cannot be represented at all.
    if (hist.pairs.len > RANS_PROB_SCALE) return error.AlphabetTooLarge;

    // Pairs are already sorted by symbol ascending.
    const n = hist.pairs.len;
    const syms = try alloc.alloc(u32, n);
    errdefer alloc.free(syms);
    for (hist.pairs, 0..) |p, k| syms[k] = p.sym;

    // Quantize frequencies to RANS_PROB_SCALE.
    const raw_total: f64 = @floatFromInt(count);
    const info = try alloc.alloc(RansSymbol, n);
    var quantized_total: u32 = 0;
    for (hist.pairs, 0..) |p, ii| {
        const prob: f64 = @as(f64, @floatFromInt(p.count)) / raw_total;
        var q: u32 = @intFromFloat(@round(prob * @as(f64, @floatFromInt(RANS_PROB_SCALE))));
        if (q == 0) q = 1; // every observed symbol gets at least one slot
        info[ii] = .{ .freq = q, .cum = 0 };
        quantized_total += q;
    }
    // Make the total exactly RANS_PROB_SCALE. Surplus must be spread across
    // symbols proportionally to their slack above 1: dumping it all on the
    // largest symbol underflows whenever the distribution is flat (e.g. 10k
    // equiprobable symbols all quantize to 2, leaving a surplus far larger
    // than any single frequency).
    if (quantized_total > RANS_PROB_SCALE) {
        var over: u32 = quantized_total - RANS_PROB_SCALE;
        var slack: u64 = 0;
        for (info) |x| slack += x.freq - 1;
        std.debug.assert(slack >= over); // holds because n <= RANS_PROB_SCALE
        const target = over;
        for (info) |*ip| {
            if (over == 0) break;
            const s = ip.freq - 1;
            if (s == 0) continue;
            const share: u32 = @intCast((@as(u64, s) * target) / slack);
            const cut = @min(over, @min(share, s));
            ip.freq -= cut;
            over -= cut;
        }
        while (over > 0) {
            for (info) |*ip| {
                if (over == 0) break;
                if (ip.freq > 1) {
                    ip.freq -= 1;
                    over -= 1;
                }
            }
        }
    } else if (quantized_total < RANS_PROB_SCALE) {
        var max_idx: usize = 0;
        for (info, 0..) |x, ii| if (x.freq > info[max_idx].freq) {
            max_idx = ii;
        };
        info[max_idx].freq += RANS_PROB_SCALE - quantized_total;
    }

    // Cumulative.
    var cum: u32 = 0;
    for (info) |*ip| {
        ip.cum = cum;
        cum += ip.freq;
    }

    return .{ .symbols = syms, .info = info };
}

const RansAdvance = enum { division, reciprocal };

inline fn ransAdvanceReference(state: u32, info: RansSymbol) u32 {
    return ((state / info.freq) << RANS_PROB_BITS) +
        (state % info.freq) + info.cum;
}

inline fn ransAdvance(
    comptime method: RansAdvance,
    state: u32,
    info: RansSymbol,
    encoder: RansEncoderSymbol,
) u32 {
    return switch (method) {
        .division => ransAdvanceReference(state, info),
        .reciprocal => encoder.advance(state),
    };
}

pub fn ransEncode(alloc: Allocator, stream: Stream, table: RansTable) ![]u8 {
    return ransEncodeImpl(alloc, stream, table, .reciprocal);
}

fn ransEncodeImpl(
    alloc: Allocator,
    stream: Stream,
    table: RansTable,
    comptime method: RansAdvance,
) ![]u8 {
    var lookup = try RansSymbolLookup.init(alloc, table);
    defer lookup.deinit();

    // Encode in REVERSE so decoder reads forward.
    var out: std.ArrayList(u8) = .empty;
    try out.ensureTotalCapacity(alloc, stream.count + 8);
    defer out.deinit(alloc);

    var state: u32 = RANS_L;
    var i: usize = stream.count;
    while (i > 0) {
        i -= 1;
        const sym = stream.getU32(i);
        const idx = try lookup.get(sym);
        const info = table.info[idx];
        const encoder = lookup.encoders[idx];

        const x_max = if (method == .division)
            ((RANS_L >> RANS_PROB_BITS) << 8) * info.freq
        else
            encoder.x_max;
        while (state >= x_max) {
            try out.append(alloc, @intCast(state & 0xFF));
            state >>= 8;
        }

        state = ransAdvance(method, state, info, encoder);
    }
    try out.append(alloc, @intCast(state & 0xFF));
    try out.append(alloc, @intCast((state >> 8) & 0xFF));
    try out.append(alloc, @intCast((state >> 16) & 0xFF));
    try out.append(alloc, @intCast((state >> 24) & 0xFF));

    const owned = try out.toOwnedSlice(alloc);
    std.mem.reverse(u8, owned);
    return owned;
}

pub fn ransEncodedSize(
    alloc: Allocator,
    stream: Stream,
    table: RansTable,
) !usize {
    return ransEncodedSizeImpl(alloc, stream, table, .reciprocal);
}

fn ransEncodedSizeImpl(
    alloc: Allocator,
    stream: Stream,
    table: RansTable,
    comptime method: RansAdvance,
) !usize {
    var lookup = try RansSymbolLookup.init(alloc, table);
    defer lookup.deinit();

    var size: usize = 4;
    var state: u32 = RANS_L;
    var i = stream.count;
    while (i > 0) {
        i -= 1;
        const idx = try lookup.get(stream.getU32(i));
        const info = table.info[idx];
        const encoder = lookup.encoders[idx];
        const x_max = if (method == .division)
            ((RANS_L >> RANS_PROB_BITS) << 8) * info.freq
        else
            encoder.x_max;
        while (state >= x_max) {
            size = std.math.add(usize, size, 1) catch
                return error.RansSizeOverflow;
            state >>= 8;
        }
        state = ransAdvance(method, state, info, encoder);
    }
    return size;
}

const RansSymbolLookup = struct {
    alloc: Allocator,
    direct: bool,
    array: []u16 = &.{},
    map: std.AutoHashMap(u32, u16),
    encoders: []RansEncoderSymbol = &.{},

    fn init(alloc: Allocator, table: RansTable) !RansSymbolLookup {
        var max_symbol: u32 = 0;
        for (table.symbols) |symbol| max_symbol = @max(max_symbol, symbol);

        var lookup = RansSymbolLookup{
            .alloc = alloc,
            .direct = max_symbol <= std.math.maxInt(u16),
            .map = .init(alloc),
        };
        errdefer lookup.deinit();
        lookup.encoders = try alloc.alloc(RansEncoderSymbol, table.info.len);
        for (lookup.encoders, table.info) |*encoder, info|
            encoder.* = .init(info);
        if (lookup.direct) {
            lookup.array = try alloc.alloc(u16, @as(usize, max_symbol) + 1);
            @memset(lookup.array, std.math.maxInt(u16));
            for (table.symbols, 0..) |symbol, index|
                lookup.array[symbol] = @intCast(index);
        } else {
            for (table.symbols, 0..) |symbol, index|
                try lookup.map.put(symbol, @intCast(index));
        }
        return lookup;
    }

    fn deinit(self: *RansSymbolLookup) void {
        if (self.array.len > 0) self.alloc.free(self.array);
        if (self.encoders.len > 0) self.alloc.free(self.encoders);
        self.map.deinit();
        self.array = &.{};
        self.encoders = &.{};
    }

    fn get(self: RansSymbolLookup, symbol: u32) !u16 {
        if (!self.direct)
            return self.map.get(symbol) orelse error.SymbolNotInTable;
        if (symbol >= self.array.len) return error.SymbolNotInTable;
        const index = self.array[symbol];
        if (index == std.math.maxInt(u16))
            return error.SymbolNotInTable;
        return index;
    }
};

pub const ransTesting = if (@import("builtin").is_test) struct {
    pub const Step = struct {
        state: u32,
        emitted: [4]u8,
        emitted_len: u3,
    };

    pub fn stepReference(state: u32, info: RansSymbol) Step {
        return step(state, info, .division);
    }

    pub fn stepReciprocal(state: u32, info: RansSymbol) Step {
        return step(state, info, .reciprocal);
    }

    pub fn encodeReference(
        alloc: Allocator,
        stream: Stream,
        table: RansTable,
    ) ![]u8 {
        return ransEncodeImpl(alloc, stream, table, .division);
    }

    pub fn encodedSizeReference(
        alloc: Allocator,
        stream: Stream,
        table: RansTable,
    ) !usize {
        return ransEncodedSizeImpl(alloc, stream, table, .division);
    }

    pub fn decodeReference(
        alloc: Allocator,
        payload: []const u8,
        table: RansTable,
        count: usize,
        bits_per_elem: u8,
    ) !RansDecodeResult {
        return ransDecodeWithStateImpl(
            alloc,
            payload,
            table,
            count,
            bits_per_elem,
            .runtime,
        );
    }

    fn step(
        initial_state: u32,
        info: RansSymbol,
        comptime method: RansAdvance,
    ) Step {
        const encoder: RansEncoderSymbol = .init(info);
        const x_max = if (method == .division)
            ((RANS_L >> RANS_PROB_BITS) << 8) * info.freq
        else
            encoder.x_max;
        var result = Step{
            .state = initial_state,
            .emitted = undefined,
            .emitted_len = 0,
        };
        while (result.state >= x_max) {
            result.emitted[result.emitted_len] = @intCast(result.state & 0xff);
            result.emitted_len += 1;
            result.state >>= 8;
        }
        result.state = ransAdvance(method, result.state, info, encoder);
        return result;
    }
} else struct {};

pub const RansDecodeResult = struct {
    stream: Stream,
    consumed_bytes: usize,
    final_state: u32,
};

const RansDecodeStorage = enum { runtime, byte, word, dword };

pub fn ransDecodeWithState(
    alloc: Allocator,
    payload: []const u8,
    table: RansTable,
    count: usize,
    bits_per_elem: u8,
) !RansDecodeResult {
    if (bits_per_elem == 0 or bits_per_elem > 32)
        return error.InvalidWordWidth;
    return switch (types.roundUpToPow2(bits_per_elem)) {
        8 => ransDecodeWithStateImpl(
            alloc,
            payload,
            table,
            count,
            bits_per_elem,
            .byte,
        ),
        16 => ransDecodeWithStateImpl(
            alloc,
            payload,
            table,
            count,
            bits_per_elem,
            .word,
        ),
        32 => ransDecodeWithStateImpl(
            alloc,
            payload,
            table,
            count,
            bits_per_elem,
            .dword,
        ),
        else => unreachable,
    };
}

fn ransDecodeWithStateImpl(
    alloc: Allocator,
    payload: []const u8,
    table: RansTable,
    count: usize,
    bits_per_elem: u8,
    comptime storage: RansDecodeStorage,
) !RansDecodeResult {
    if (bits_per_elem == 0 or bits_per_elem > 32)
        return error.InvalidWordWidth;
    if (count != 0 and payload.len < 4) return error.CorruptRansStream;
    // Build cum->sym lookup locally so the table stays read-only.
    const cum2sym = try alloc.alloc(u16, RANS_PROB_SCALE);
    defer alloc.free(cum2sym);
    for (table.info, 0..) |info, ii| {
        const end = info.cum + info.freq;
        var c = info.cum;
        while (c < end) : (c += 1) cum2sym[c] = @intCast(ii);
    }

    var s = try Stream.initUninitialized(alloc, count, bits_per_elem);
    errdefer s.deinit(alloc);
    if (count == 0) return .{
        .stream = s,
        .consumed_bytes = 0,
        .final_state = RANS_L,
    };

    var pos: usize = 0;
    var state: u32 = std.mem.readInt(u32, payload[pos..][0..4], .big);
    pos += 4;

    var i: usize = 0;
    while (i < count) : (i += 1) {
        const slot = state & (RANS_PROB_SCALE - 1);
        const idx = cum2sym[slot];
        const sym = table.symbols[idx];
        switch (storage) {
            .runtime => s.setU32(i, sym),
            .byte => s.data[i] = @truncate(sym),
            .word => std.mem.writeInt(
                u16,
                s.data[i * 2 ..][0..2],
                @truncate(sym),
                .little,
            ),
            .dword => std.mem.writeInt(
                u32,
                s.data[i * 4 ..][0..4],
                sym,
                .little,
            ),
        }
        const f = table.info[idx].freq;
        const c = table.info[idx].cum;
        state = f * (state >> RANS_PROB_BITS) + slot - c;
        // Renormalize.
        while (state < RANS_L) {
            if (pos >= payload.len) return error.CorruptRansStream;
            state = (state << 8) | payload[pos];
            pos += 1;
        }
    }
    return .{
        .stream = s,
        .consumed_bytes = pos,
        .final_state = state,
    };
}

pub fn ransDecode(
    alloc: Allocator,
    payload: []const u8,
    table: RansTable,
    count: usize,
    bits_per_elem: u8,
) !Stream {
    return (try ransDecodeWithState(
        alloc,
        payload,
        table,
        count,
        bits_per_elem,
    )).stream;
}

// ==================== closed-form terminal costs ====================
//
// The searcher prices candidates without encoding. For Huffman the payload
// size is EXACT: a prefix code writes sum(len[s]*count[s]) bits and flushes,
// so this is byte-identical to what huffmanEncode would emit. For rANS the
// quantized table gives a strict lower bound (the coder cannot beat the
// probabilities it was built from), which is what the branch-and-bound needs.

/// Exact payload bytes huffmanEncode would produce for this histogram.
pub fn huffmanPayloadBytes(table: HuffmanTable, hist: Histogram) u64 {
    // entries are canonical order (len, sym); hist.pairs is sym-ascending.
    var bits: u64 = 0;
    for (table.entries) |e| {
        var lo: usize = 0;
        var hi: usize = hist.pairs.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            if (hist.pairs[mid].sym < e.sym) lo = mid + 1 else hi = mid;
        }
        if (lo < hist.pairs.len and hist.pairs[lo].sym == e.sym) {
            const contribution = std.math.mul(
                u64,
                @as(u64, e.len),
                hist.pairs[lo].count,
            ) catch return std.math.maxInt(u64);
            bits = std.math.add(u64, bits, contribution) catch
                return std.math.maxInt(u64);
        }
    }
    const rounded = std.math.add(u64, bits, 7) catch
        return std.math.maxInt(u64);
    return rounded / 8;
}

/// Strict lower bound on ransEncode's payload: sum count[s]*log2(SCALE/freq[s]).
/// The trailing 4 state bytes are deliberately omitted to keep it a bound.
pub fn ransLowerBytes(table: RansTable, hist: Histogram) u64 {
    var bits: f64 = 0;
    for (hist.pairs, 0..) |p, k| {
        const f = table.info[k].freq;
        if (f == 0) continue;
        const q = @as(f64, @floatFromInt(f)) / @as(f64, @floatFromInt(RANS_PROB_SCALE));
        bits += @as(f64, @floatFromInt(p.count)) * -std.math.log2(q);
    }
    return @intFromFloat(@floor(bits / 8.0));
}

/// Zeroth-order entropy in bits, from a histogram. O(alphabet), no data pass.
pub fn entropyBits(hist: Histogram, count: usize) u64 {
    if (count == 0) return 0;
    const total: f64 = @floatFromInt(count);
    var h: f64 = 0;
    for (hist.pairs) |p| {
        const q = @as(f64, @floatFromInt(p.count)) / total;
        h -= q * std.math.log2(q);
    }
    return @intFromFloat(@floor(h * total));
}
