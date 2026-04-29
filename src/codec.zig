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
    bit_pos: usize = 0,

    pub fn init(bytes: []const u8) BitReader {
        return .{ .bytes = bytes };
    }

    /// Read one bit MSB-first.
    pub fn readBit(self: *BitReader) u1 {
        const byte_i = self.bit_pos >> 3;
        const bit_i: u3 = @intCast(7 - (self.bit_pos & 7));
        const b: u1 = @intCast((self.bytes[byte_i] >> bit_i) & 1);
        self.bit_pos += 1;
        return b;
    }
};

// ==================== Canonical Huffman ====================

pub const HuffmanTable = struct {
    /// code_length[symbol] = bits used; 0 means "symbol unused".
    /// Symbols are u32 keyed; sparse via a sorted (sym, len) list.
    entries: []Entry, // sorted by (length asc, symbol asc) — canonical order

    pub const Entry = struct { sym: u32, len: u8 };

    pub fn deinit(self: *HuffmanTable, alloc: Allocator) void {
        alloc.free(self.entries);
        self.entries = &.{};
    }
};

/// Histogram a stream into (sym, count) pairs. For 8/16-bit storage the
/// fast path uses 4 parallel counter arrays to break the read-modify-write
/// dependency between iterations (a 2-4× speed-up on a single core, no
/// SIMD needed — modern OoO engines pipeline the independent increments).
/// For wider alphabets, falls back to a HashMap.
const Histogram = struct {
    /// (sym, count) pairs, ascending sym; only present symbols are listed.
    pairs: []Pair,
    pub const Pair = struct { sym: u32, count: u64 };

    pub fn deinit(self: *Histogram, alloc: Allocator) void {
        alloc.free(self.pairs);
        self.pairs = &.{};
    }
};

fn buildHistogram(alloc: Allocator, stream: Stream) !Histogram {
    const bpe_pow2 = types.roundUpToPow2(stream.bits_per_elem);
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

/// Build a length-limited Huffman tree (max 32 bits) from a stream.
/// Returns the canonical table.
pub fn huffmanBuild(alloc: Allocator, stream: Stream) !HuffmanTable {
    var hist = try buildHistogram(alloc, stream);
    defer hist.deinit(alloc);

    if (hist.pairs.len == 0) {
        return .{ .entries = try alloc.alloc(HuffmanTable.Entry, 0) };
    }
    if (hist.pairs.len == 1) {
        // Degenerate: a single symbol gets a 1-bit code (canonical).
        const e = try alloc.alloc(HuffmanTable.Entry, 1);
        e[0] = .{ .sym = hist.pairs[0].sym, .len = 1 };
        return .{ .entries = e };
    }

    // Step 2: standard package-merge would be ideal for length-limiting,
    // but plain Huffman with 32-bit codes is enough for our alphabet sizes.
    // We build a heap of nodes and merge.
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
    const elem_bytes: usize = switch (types.roundUpToPow2(bits_per_elem)) {
        8 => 1,
        16 => 2,
        32 => 4,
        else => unreachable,
    };
    const buf = try alloc.alloc(u8, count * elem_bytes);
    var s: Stream = .{ .data = buf, .count = count, .bits_per_elem = bits_per_elem };

    if (count == 0) return s;

    var codes = try huffmanCodes(alloc, table);
    defer codes.deinit();

    // Build (code,len) -> sym lookup. For our table sizes a HashMap keyed by
    // (len << 32 | code) is plenty. But we use a per-length lookup for speed.
    const max_len = table.entries[table.entries.len - 1].len;
    var lookup = try alloc.alloc(std.AutoHashMap(u64, u32), @as(usize, max_len) + 1);
    defer alloc.free(lookup);
    for (lookup, 0..) |*lk, i| {
        _ = i;
        lk.* = .init(alloc);
    }
    defer for (lookup) |*lk| lk.deinit();

    for (table.entries) |e| {
        const c = codes.get(e.sym).?;
        try lookup[e.len].put(c, e.sym);
    }

    var br = BitReader.init(payload);

    var i: usize = 0;
    while (i < count) : (i += 1) {
        var code: u64 = 0;
        var len: u8 = 0;
        var found = false;
        while (len < max_len) {
            const bit: u64 = br.readBit();
            code = (code << 1) | bit;
            len += 1;
            if (lookup[len].get(code)) |sym| {
                s.setU32(i, sym);
                found = true;
                break;
            }
        }
        if (!found) return error.CorruptHuffmanStream;
    }
    return s;
}

/// Cost estimate (bits) for Huffman-coding a stream — Shannon entropy plus
/// table overhead. Admissible-ish lower bound for the search heuristic.
pub fn huffmanCostBits(stream: Stream) u64 {
    if (stream.count == 0) return 0;
    var counts: [65536]u32 = undefined;
    @memset(&counts, 0);
    var n: u32 = 0;

    if (stream.bits_per_elem <= 16) {
        for (0..stream.count) |i| {
            const sym = stream.getU32(i) & 0xFFFF;
            counts[sym] += 1;
            n += 1;
        }
        var H: f64 = 0.0;
        const fn_total: f64 = @floatFromInt(n);
        var unique: u32 = 0;
        for (counts) |c| {
            if (c == 0) continue;
            unique += 1;
            const p: f64 = @as(f64, @floatFromInt(c)) / fn_total;
            H += -p * std.math.log2(p);
        }
        const data_bits: u64 = @intFromFloat(@ceil(H * fn_total));
        const table_overhead: u64 = @as(u64, unique) * 24; // ~3 bytes per entry
        return data_bits + table_overhead;
    }
    // Fallback: just bound by raw size — entropy of u32 alphabet is too
    // expensive to compute in the heuristic.
    return @as(u64, stream.count) * @as(u64, stream.bits_per_elem);
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
    // For decode: cum -> symbol_index lookup of length RANS_PROB_SCALE.
    cum2sym: ?[]u16 = null,

    pub fn deinit(self: *RansTable, alloc: Allocator) void {
        alloc.free(self.symbols);
        alloc.free(self.info);
        if (self.cum2sym) |cs| alloc.free(cs);
        self.symbols = &.{};
        self.info = &.{};
        self.cum2sym = null;
    }

    pub fn buildCum2Sym(self: *RansTable, alloc: Allocator) !void {
        const cs = try alloc.alloc(u16, RANS_PROB_SCALE);
        for (self.info, 0..) |info, i| {
            const end = info.cum + info.freq;
            var c = info.cum;
            while (c < end) : (c += 1) cs[c] = @intCast(i);
        }
        self.cum2sym = cs;
    }
};

pub fn ransBuild(alloc: Allocator, stream: Stream) !RansTable {
    var hist = try buildHistogram(alloc, stream);
    defer hist.deinit(alloc);

    if (hist.pairs.len == 0) {
        return .{
            .symbols = try alloc.alloc(u32, 0),
            .info = try alloc.alloc(RansSymbol, 0),
        };
    }

    // Pairs are already sorted by symbol ascending.
    const n = hist.pairs.len;
    const syms = try alloc.alloc(u32, n);
    for (hist.pairs, 0..) |p, k| syms[k] = p.sym;

    // Quantize frequencies to RANS_PROB_SCALE.
    const raw_total: f64 = @floatFromInt(stream.count);
    const info = try alloc.alloc(RansSymbol, n);
    var quantized_total: u32 = 0;
    for (hist.pairs, 0..) |p, ii| {
        const prob: f64 = @as(f64, @floatFromInt(p.count)) / raw_total;
        var q: u32 = @intFromFloat(@round(prob * @as(f64, @floatFromInt(RANS_PROB_SCALE))));
        if (q == 0) q = 1; // every observed symbol gets at least one slot
        info[ii] = .{ .freq = q, .cum = 0 };
        quantized_total += q;
    }
    // Adjust the largest frequency to make the total exactly RANS_PROB_SCALE.
    var max_idx: usize = 0;
    for (info, 0..) |x, ii| if (x.freq > info[max_idx].freq) {
        max_idx = ii;
    };
    if (quantized_total > RANS_PROB_SCALE) {
        const over = quantized_total - RANS_PROB_SCALE;
        info[max_idx].freq -= over;
    } else if (quantized_total < RANS_PROB_SCALE) {
        const under = RANS_PROB_SCALE - quantized_total;
        info[max_idx].freq += under;
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
    if (count == 0) return s;

    if (payload.len < 4) return error.CorruptRansStream;

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

/// Optimistic lower bound for rANS cost — Shannon entropy * count.
pub fn ransCostBits(stream: Stream) u64 {
    if (stream.count == 0) return 0;
    if (stream.bits_per_elem > 16) {
        return @as(u64, stream.count) * @as(u64, stream.bits_per_elem);
    }
    var counts: [65536]u32 = undefined;
    @memset(&counts, 0);
    for (0..stream.count) |i| {
        counts[stream.getU32(i) & 0xFFFF] += 1;
    }
    var H: f64 = 0.0;
    const total_f: f64 = @floatFromInt(stream.count);
    var unique: u32 = 0;
    for (counts) |c| {
        if (c == 0) continue;
        unique += 1;
        const p: f64 = @as(f64, @floatFromInt(c)) / total_f;
        H += -p * std.math.log2(p);
    }
    const data_bits: u64 = @intFromFloat(@ceil(H * total_f));
    // Smaller table overhead than Huffman because we transmit just freqs.
    const table_overhead: u64 = @as(u64, unique) * 18;
    return data_bits + table_overhead;
}
