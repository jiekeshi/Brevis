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

    pub fn deinit(self: *Histogram, alloc: Allocator) void {
        alloc.free(self.pairs);
        self.pairs = &.{};
    }
};

/// Below this length a dense alphabet-sized histogram costs far more to zero
/// than the stream costs to sort. The search evaluates many short streams, so
/// this path dominates in practice.
const SMALL_STREAM: usize = 8192;

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
        var run: u64 = 1;
        while (i + run < tmp.len and tmp[i + run] == tmp[i]) run += 1;
        pairs[j] = .{ .sym = tmp[i], .count = run };
        j += 1;
        i += run;
    }
    return .{ .pairs = pairs };
}

pub fn buildHistogram(alloc: Allocator, stream: Stream) !Histogram {
    const bpe_pow2 = types.roundUpToPow2(stream.bits_per_elem);
    if (bpe_pow2 != 8 and stream.count < SMALL_STREAM) return histogramBySort(alloc, stream);
    if (bpe_pow2 == 8) {
        var c0: [256]u32 = .{0} ** 256;
        var c1: [256]u32 = .{0} ** 256;
        var c2: [256]u32 = .{0} ** 256;
        var c3: [256]u32 = .{0} ** 256;
        const data = stream.data;
        var i: usize = 0;
        const n = stream.count;
        while (i + 4 <= n) : (i += 4) {
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
            const tot: u64 = @as(u64, c0[k]) + c1[k] + c2[k] + c3[k];
            if (tot > 0) {
                pairs[j] = .{ .sym = @intCast(k), .count = tot };
                j += 1;
            }
        }
        return .{ .pairs = pairs };
    }
    if (bpe_pow2 == 16) {
        // 4 × 65536 × u32 = 1 MB on the heap (don't put on the stack).
        const slots: usize = 65536;
        const pool = try alloc.alloc(u32, slots * 4);
        defer alloc.free(pool);
        @memset(pool, 0);
        const c0 = pool[0..slots];
        const c1 = pool[slots .. 2 * slots];
        const c2 = pool[2 * slots .. 3 * slots];
        const c3 = pool[3 * slots .. 4 * slots];
        var i: usize = 0;
        const n = stream.count;
        while (i + 4 <= n) : (i += 4) {
            c0[std.mem.readInt(u16, stream.data[(i + 0) * 2 ..][0..2], .little)] += 1;
            c1[std.mem.readInt(u16, stream.data[(i + 1) * 2 ..][0..2], .little)] += 1;
            c2[std.mem.readInt(u16, stream.data[(i + 2) * 2 ..][0..2], .little)] += 1;
            c3[std.mem.readInt(u16, stream.data[(i + 3) * 2 ..][0..2], .little)] += 1;
        }
        while (i < n) : (i += 1) c0[std.mem.readInt(u16, stream.data[i * 2 ..][0..2], .little)] += 1;
        var unique: usize = 0;
        var k: usize = 0;
        while (k < slots) : (k += 1) {
            if (@as(u64, c0[k]) + c1[k] + c2[k] + c3[k] > 0) unique += 1;
        }
        const pairs = try alloc.alloc(Histogram.Pair, unique);
        var j: usize = 0;
        k = 0;
        while (k < slots) : (k += 1) {
            const tot: u64 = @as(u64, c0[k]) + c1[k] + c2[k] + c3[k];
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
        if (!gop.found_existing) gop.value_ptr.* = 0;
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
fn huffmanCodes(alloc: Allocator, table: HuffmanTable) !std.AutoHashMap(u32, u64) {
    var codes: std.AutoHashMap(u32, u64) = .init(alloc);
    if (table.entries.len == 0) return codes;
    var code: u64 = 0;
    var prev_len: u8 = table.entries[0].len;
    for (table.entries) |e| {
        if (e.len > prev_len) {
            code <<= @intCast(e.len - prev_len);
            prev_len = e.len;
        }
        try codes.put(e.sym, code);
        code += 1;
    }
    return codes;
}

pub fn huffmanEncode(alloc: Allocator, stream: Stream, table: HuffmanTable) ![]u8 {
    var codes = try huffmanCodes(alloc, table);
    defer codes.deinit();

    // Fast path for 8/16-bit alphabets: use direct-indexed arrays for code
    // and length lookup. This avoids ~20 ns/elem of HashMap overhead.
    const max_sym: u32 = blk: {
        var m: u32 = 0;
        for (table.entries) |e| if (e.sym > m) {
            m = e.sym;
        };
        break :blk m;
    };
    const direct_path: bool = max_sym <= 0xFFFF;
    var code_arr: []u64 = &.{};
    var len_arr: []u8 = &.{};
    defer if (direct_path) {
        alloc.free(code_arr);
        alloc.free(len_arr);
    };
    var len_map: std.AutoHashMap(u32, u8) = undefined;
    var have_len_map: bool = false;
    defer if (have_len_map) len_map.deinit();

    if (direct_path) {
        const n_slots: usize = @as(usize, max_sym) + 1;
        code_arr = try alloc.alloc(u64, n_slots);
        len_arr = try alloc.alloc(u8, n_slots);
        @memset(len_arr, 0);
        for (table.entries) |e| {
            const c = codes.get(e.sym).?;
            code_arr[e.sym] = c;
            len_arr[e.sym] = e.len;
        }
    } else {
        len_map = .init(alloc);
        have_len_map = true;
        for (table.entries) |e| try len_map.put(e.sym, e.len);
    }

    var out: std.ArrayList(u8) = .empty;
    try out.ensureTotalCapacity(alloc, stream.count); // ~1 byte/elem upper bound is loose; fine for grow
    defer out.deinit(alloc);
    var bw = BitWriter.init(alloc, &out);

    if (direct_path) {
        for (0..stream.count) |i| {
            const sym = stream.getU32(i);
            const len = len_arr[sym];
            if (len == 0) return error.SymbolNotInTable;
            try bw.writeBits(code_arr[sym], len);
        }
    } else {
        for (0..stream.count) |i| {
            const sym = stream.getU32(i);
            const len = len_map.get(sym) orelse return error.SymbolNotInTable;
            const code = codes.get(sym).?;
            try bw.writeBits(code, len);
        }
    }
    try bw.flush();

    return out.toOwnedSlice(alloc);
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
    std.debug.assert(width > 0 and width <= 32);

    var out: std.ArrayList(u8) = .empty;
    try out.ensureTotalCapacity(alloc, (stream.count * width + 7) / 8);
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
    std.debug.assert(width > 0 and width <= 32);

    const buf = try alloc.alloc(u8, count * (types.roundUpToPow2(out_bpe) / 8));
    var s: Stream = .{ .data = buf, .count = count, .bits_per_elem = out_bpe };
    errdefer s.deinit(alloc);
    if (count == 0) return s;

    if (payload.len * 8 < count * @as(usize, width)) return error.CorruptBitpackStream;

    var br = BitReader.init(payload);
    for (0..count) |i| {
        s.setU32(i, @intCast(br.readBits(width).?));
    }
    return s;
}

pub fn bitpackCostBits(stream: Stream, width: u8) u64 {
    const bits = @as(u64, stream.count) * @as(u64, width);
    return ((bits + 7) / 8) * 8;
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

pub fn ransEncode(alloc: Allocator, stream: Stream, table: RansTable) ![]u8 {
    // Direct-indexed sym -> index array for 8/16-bit alphabets (the common case).
    const max_sym: u32 = blk: {
        var m: u32 = 0;
        for (table.symbols) |s| if (s > m) {
            m = s;
        };
        break :blk m;
    };
    const direct_path = max_sym <= 0xFFFF;
    var s2i_arr: []u16 = &.{};
    defer if (direct_path) alloc.free(s2i_arr);
    var s2i_map: std.AutoHashMap(u32, u16) = undefined;
    var have_map = false;
    defer if (have_map) s2i_map.deinit();
    if (direct_path) {
        s2i_arr = try alloc.alloc(u16, @as(usize, max_sym) + 1);
        @memset(s2i_arr, std.math.maxInt(u16));
        for (table.symbols, 0..) |s, idx| s2i_arr[s] = @intCast(idx);
    } else {
        s2i_map = .init(alloc);
        have_map = true;
        for (table.symbols, 0..) |s, idx| try s2i_map.put(s, @intCast(idx));
    }

    // Encode in REVERSE so decoder reads forward.
    var out: std.ArrayList(u8) = .empty;
    try out.ensureTotalCapacity(alloc, stream.count + 8);
    defer out.deinit(alloc);

    var state: u32 = RANS_L;
    var i: usize = stream.count;
    while (i > 0) {
        i -= 1;
        const sym = stream.getU32(i);
        const idx: u16 = if (direct_path) s2i_arr[sym] else (s2i_map.get(sym) orelse return error.SymbolNotInTable);
        if (direct_path and idx == std.math.maxInt(u16)) return error.SymbolNotInTable;
        const f = table.info[idx].freq;
        const c = table.info[idx].cum;

        const x_max = ((RANS_L >> RANS_PROB_BITS) << 8) * f;
        while (state >= x_max) {
            try out.append(alloc, @intCast(state & 0xFF));
            state >>= 8;
        }

        state = ((state / f) << RANS_PROB_BITS) + (state % f) + c;
    }
    try out.append(alloc, @intCast(state & 0xFF));
    try out.append(alloc, @intCast((state >> 8) & 0xFF));
    try out.append(alloc, @intCast((state >> 16) & 0xFF));
    try out.append(alloc, @intCast((state >> 24) & 0xFF));

    const owned = try out.toOwnedSlice(alloc);
    std.mem.reverse(u8, owned);
    return owned;
}

pub fn ransDecode(alloc: Allocator, payload: []const u8, table: RansTable, count: usize, bits_per_elem: u8) !Stream {
    if (count != 0 and payload.len < 4) return error.CorruptRansStream;
    // Build cum->sym lookup locally so the table stays read-only.
    const cum2sym = try alloc.alloc(u16, RANS_PROB_SCALE);
    defer alloc.free(cum2sym);
    for (table.info, 0..) |info, ii| {
        const end = info.cum + info.freq;
        var c = info.cum;
        while (c < end) : (c += 1) cum2sym[c] = @intCast(ii);
    }

    const elem_bytes: usize = switch (types.roundUpToPow2(bits_per_elem)) {
        8 => 1,
        16 => 2,
        32 => 4,
        else => unreachable,
    };
    const buf = try alloc.alloc(u8, count * elem_bytes);
    var s: Stream = .{ .data = buf, .count = count, .bits_per_elem = bits_per_elem };
    errdefer s.deinit(alloc);
    if (count == 0) return s;

    var pos: usize = 0;
    var state: u32 = std.mem.readInt(u32, payload[pos..][0..4], .big);
    pos += 4;

    var i: usize = 0;
    while (i < count) : (i += 1) {
        const slot = state & (RANS_PROB_SCALE - 1);
        const idx = cum2sym[slot];
        const sym = table.symbols[idx];
        s.setU32(i, sym);
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
    return s;
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
            bits += @as(u64, e.len) * hist.pairs[lo].count;
        }
    }
    return (bits + 7) / 8;
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
