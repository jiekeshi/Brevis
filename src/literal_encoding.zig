//! Self-contained physical encodings for a semantic DSL literal.
//!
//! `bits_per_elem` and `count` belong to the literal's semantic type and are
//! supplied by its enclosing program. The body emitted here contains every
//! additional byte required by the selected physical encoding.
//!
//! All multi-byte integers are little-endian. Stable wire bodies are:
//!   raw:     tag | storage bytes
//!   bitpack: tag | width:u8 | packed bytes
//!   huffman: tag | entry_count:u32 | (symbol:u32, length:u8)*
//!                  | payload_bit_count:u64 | payload
//!   rANS:    tag | entry_count:u32 | (symbol:u32, frequency:u32)*
//!                  | payload_byte_count:u64 | payload
//!
//! The Huffman entries are in canonical (length, symbol) order. rANS entries
//! are in ascending symbol order; cumulative frequencies are reconstructed.

const std = @import("std");
const codec = @import("codec.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const Stream = types.Stream;

/// Values are permanent wire identifiers and also define the stable tie-break
/// order when two complete bodies have the same byte length.
pub const Tag = enum(u8) {
    raw = 0,
    bitpack = 1,
    huffman = 2,
    rans = 3,
};

pub const OwnedEncoding = struct {
    tag: Tag,
    /// Complete wire body, including `tag`.
    body: []u8,

    pub fn deinit(self: *OwnedEncoding, alloc: Allocator) void {
        alloc.free(self.body);
        self.body = &.{};
    }

    pub fn wireSize(self: OwnedEncoding) usize {
        return self.body.len;
    }

    pub fn emitBody(
        self: OwnedEncoding,
        alloc: Allocator,
        out: *std.ArrayList(u8),
    ) Allocator.Error!void {
        try out.appendSlice(alloc, self.body);
    }
};

pub fn wireSize(encoding: OwnedEncoding) usize {
    return encoding.body.len;
}

pub fn emitBody(
    alloc: Allocator,
    out: *std.ArrayList(u8),
    encoding: OwnedEncoding,
) Allocator.Error!void {
    try encoding.emitBody(alloc, out);
}

/// Encode all applicable physical representations and select by the exact
/// complete body length. Equal lengths use the stable numeric `Tag` order.
/// Raw is constructed first and is therefore an unconditional fallback for
/// every valid stream.
pub fn encodeBest(alloc: Allocator, stream: Stream) !OwnedEncoding {
    try validateStream(stream);

    var best = try encodeRaw(alloc, stream);
    errdefer best.deinit(alloc);
    if (stream.count == 0) return best;

    var bitpack_candidate = try encodeBitpack(alloc, stream);
    consider(alloc, &best, &bitpack_candidate);

    if (try encodeHuffman(alloc, stream)) |candidate_value| {
        var candidate = candidate_value;
        consider(alloc, &best, &candidate);
    }

    if (try encodeRans(alloc, stream)) |candidate_value| {
        var candidate = candidate_value;
        consider(alloc, &best, &candidate);
    }

    return best;
}

/// Decode an owned result from `encodeBest`.
pub fn decode(
    alloc: Allocator,
    bits_per_elem: u8,
    count: usize,
    encoding: OwnedEncoding,
) !Stream {
    if (encoding.body.len == 0 or
        encoding.body[0] != @intFromEnum(encoding.tag))
        return error.CorruptLiteralEncoding;
    return decodeBody(alloc, bits_per_elem, count, encoding.body);
}

/// Decode a complete emitted body. This is the strict archive-facing form of
/// `decode`: besides validating the selected codec's representation, it
/// requires the body to equal `encodeBest`'s globally smallest representation
/// (including the stable numeric-tag tie-break).
pub fn decodeBody(
    alloc: Allocator,
    bits_per_elem: u8,
    count: usize,
    body: []const u8,
) !Stream {
    var decoded = try decodeBodyUnchecked(
        alloc,
        bits_per_elem,
        count,
        body,
    );
    errdefer decoded.deinit(alloc);

    var canonical = try encodeBest(alloc, decoded);
    defer canonical.deinit(alloc);
    if (!std.mem.eql(u8, canonical.body, body))
        return error.NonCanonicalLiteralEncoding;

    return decoded;
}

/// Decode and validate one selected physical codec without comparing it
/// against the other codecs. Keep this private so persisted inputs can only
/// enter through the globally canonical `decodeBody` seam above.
fn decodeBodyUnchecked(
    alloc: Allocator,
    bits_per_elem: u8,
    count: usize,
    body: []const u8,
) !Stream {
    _ = try checkedStorageBytes(bits_per_elem, count);
    var reader: Reader = .{ .bytes = body };
    const tag = std.enums.fromInt(Tag, try reader.byte()) orelse
        return error.UnknownLiteralEncoding;

    return switch (tag) {
        .raw => decodeRaw(alloc, bits_per_elem, count, &reader),
        .bitpack => decodeBitpack(alloc, bits_per_elem, count, &reader),
        .huffman => decodeHuffman(alloc, bits_per_elem, count, &reader),
        .rans => decodeRans(alloc, bits_per_elem, count, &reader),
    };
}

fn validateStream(stream: Stream) !void {
    const expected = try checkedStorageBytes(stream.bits_per_elem, stream.count);
    if (stream.data.len != expected) return error.InvalidLiteralStream;

    const mask = stream.mask();
    for (0..stream.count) |index| {
        if (stream.getU32(index) & ~mask != 0)
            return error.InvalidLiteralStream;
    }
}

fn checkedStorageBytes(bits_per_elem: u8, count: usize) !usize {
    if (bits_per_elem == 0 or bits_per_elem > 32)
        return error.InvalidLiteralWidth;
    const elem_bytes: usize = types.roundUpToPow2(bits_per_elem) / 8;
    return std.math.mul(usize, count, elem_bytes) catch
        return error.LiteralSizeOverflow;
}

fn consider(
    alloc: Allocator,
    best: *OwnedEncoding,
    candidate: *OwnedEncoding,
) void {
    const candidate_is_better =
        candidate.body.len < best.body.len or
        (candidate.body.len == best.body.len and
            @intFromEnum(candidate.tag) < @intFromEnum(best.tag));
    if (candidate_is_better) {
        best.deinit(alloc);
        best.* = candidate.*;
        candidate.body = &.{};
    } else {
        candidate.deinit(alloc);
    }
}

fn encodeRaw(alloc: Allocator, stream: Stream) !OwnedEncoding {
    const body_len = std.math.add(usize, 1, stream.data.len) catch
        return error.LiteralSizeOverflow;
    const body = try alloc.alloc(u8, body_len);
    errdefer alloc.free(body);
    body[0] = @intFromEnum(Tag.raw);
    @memcpy(body[1..], stream.data);
    return .{ .tag = .raw, .body = body };
}

fn encodeBitpack(alloc: Allocator, stream: Stream) !OwnedEncoding {
    const width = requiredBits(stream);
    const payload = try codec.bitpackEncode(alloc, stream, width);
    defer alloc.free(payload);

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(alloc);
    try out.ensureTotalCapacity(alloc, 2 + payload.len);
    try out.append(alloc, @intFromEnum(Tag.bitpack));
    try out.append(alloc, width);
    try out.appendSlice(alloc, payload);
    return .{ .tag = .bitpack, .body = try out.toOwnedSlice(alloc) };
}

fn requiredBits(stream: Stream) u8 {
    var combined: u32 = 0;
    for (0..stream.count) |index| combined |= stream.getU32(index);
    return if (combined == 0) 1 else @intCast(32 - @clz(combined));
}

fn encodeHuffman(alloc: Allocator, stream: Stream) !?OwnedEncoding {
    var histogram = codec.buildHistogram(alloc, stream) catch |err| switch (err) {
        error.AlphabetTooLarge => return null,
        else => return err,
    };
    defer histogram.deinit(alloc);

    var table = codec.huffmanFromHist(
        alloc,
        histogram,
        stream.bits_per_elem,
    ) catch |err| switch (err) {
        error.HuffmanCodeTooLong, error.AlphabetTooLarge => return null,
        else => return err,
    };
    defer table.deinit(alloc);

    const payload = try codec.huffmanEncode(alloc, stream, table);
    defer alloc.free(payload);
    const payload_bits = try huffmanPayloadBits(table, histogram);

    if (table.entries.len > std.math.maxInt(u32))
        return null;

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(alloc);
    try out.append(alloc, @intFromEnum(Tag.huffman));
    try appendU32(alloc, &out, @intCast(table.entries.len));
    for (table.entries) |entry| {
        try appendU32(alloc, &out, entry.sym);
        try out.append(alloc, entry.len);
    }
    try appendU64(alloc, &out, payload_bits);
    try out.appendSlice(alloc, payload);
    return .{
        .tag = .huffman,
        .body = try out.toOwnedSlice(alloc),
    };
}

fn huffmanPayloadBits(
    table: codec.HuffmanTable,
    histogram: codec.Histogram,
) !u64 {
    var bits: u64 = 0;
    for (table.entries) |entry| {
        var low: usize = 0;
        var high = histogram.pairs.len;
        while (low < high) {
            const middle = low + (high - low) / 2;
            if (histogram.pairs[middle].sym < entry.sym)
                low = middle + 1
            else
                high = middle;
        }
        if (low < histogram.pairs.len and
            histogram.pairs[low].sym == entry.sym)
        {
            const symbol_bits = std.math.mul(
                u64,
                @as(u64, entry.len),
                histogram.pairs[low].count,
            ) catch return error.LiteralSizeOverflow;
            bits = std.math.add(u64, bits, symbol_bits) catch
                return error.LiteralSizeOverflow;
        }
    }
    return bits;
}

fn encodeRans(alloc: Allocator, stream: Stream) !?OwnedEncoding {
    var table = codec.ransBuild(alloc, stream) catch |err| switch (err) {
        error.AlphabetTooLarge => return null,
        else => return err,
    };
    defer table.deinit(alloc);

    const payload = try codec.ransEncode(alloc, stream, table);
    defer alloc.free(payload);
    if (table.symbols.len > std.math.maxInt(u32))
        return null;

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(alloc);
    try out.append(alloc, @intFromEnum(Tag.rans));
    try appendU32(alloc, &out, @intCast(table.symbols.len));
    for (table.symbols, table.info) |symbol, info| {
        try appendU32(alloc, &out, symbol);
        try appendU32(alloc, &out, info.freq);
    }
    try appendU64(alloc, &out, @intCast(payload.len));
    try out.appendSlice(alloc, payload);
    return .{
        .tag = .rans,
        .body = try out.toOwnedSlice(alloc),
    };
}

fn decodeRaw(
    alloc: Allocator,
    bits_per_elem: u8,
    count: usize,
    reader: *Reader,
) !Stream {
    const expected = try checkedStorageBytes(bits_per_elem, count);
    const payload = try reader.take(expected);
    try reader.finish();

    var stream = try Stream.init(alloc, count, bits_per_elem);
    errdefer stream.deinit(alloc);
    @memcpy(stream.data, payload);
    try validateStream(stream);
    return stream;
}

fn decodeBitpack(
    alloc: Allocator,
    bits_per_elem: u8,
    count: usize,
    reader: *Reader,
) !Stream {
    const width = try reader.byte();
    if (width == 0 or width > bits_per_elem or width > 32)
        return error.CorruptLiteralEncoding;
    const bit_count = std.math.mul(usize, count, @as(usize, width)) catch
        return error.LiteralSizeOverflow;
    const payload_len = std.math.add(usize, bit_count, 7) catch
        return error.LiteralSizeOverflow;
    const payload = try reader.take(payload_len / 8);
    try reader.finish();
    try validateZeroPadding(payload, bit_count);

    return codec.bitpackDecode(
        alloc,
        payload,
        width,
        count,
        bits_per_elem,
    );
}

fn decodeHuffman(
    alloc: Allocator,
    bits_per_elem: u8,
    count: usize,
    reader: *Reader,
) !Stream {
    const entry_count_u32 = try reader.readU32();
    const entry_count: usize = @intCast(entry_count_u32);
    if ((count == 0 and entry_count != 0) or
        (count != 0 and (entry_count == 0 or entry_count > count)))
        return error.CorruptLiteralEncoding;
    if (entry_count > codec.MAX_HISTOGRAM_SYMBOLS or
        @as(u64, entry_count) > maxAlphabet(bits_per_elem))
        return error.CorruptLiteralEncoding;
    if (entry_count > reader.remaining() / 5)
        return error.TruncatedLiteralEncoding;

    const entries = try alloc.alloc(codec.HuffmanTable.Entry, entry_count);
    var table: codec.HuffmanTable = .{ .entries = entries };
    defer table.deinit(alloc);
    var seen_symbols: std.AutoHashMap(u32, void) = .init(alloc);
    defer seen_symbols.deinit();
    for (entries) |*entry| {
        entry.* = .{
            .sym = try reader.readU32(),
            .len = try reader.byte(),
        };
        if (seen_symbols.contains(entry.sym))
            return error.CorruptLiteralEncoding;
        try seen_symbols.put(entry.sym, {});
    }

    const payload_bits = try reader.readU64();
    const payload_len_u64 = std.math.add(u64, payload_bits, 7) catch
        return error.LiteralSizeOverflow;
    const payload_len = std.math.cast(usize, payload_len_u64 / 8) orelse
        return error.LiteralSizeOverflow;
    const payload = try reader.take(payload_len);
    try reader.finish();
    const payload_bit_count = std.math.cast(usize, payload_bits) orelse
        return error.LiteralSizeOverflow;
    try validateZeroPadding(payload, payload_bit_count);

    var decoded = try codec.huffmanDecode(
        alloc,
        payload,
        table,
        count,
        bits_per_elem,
    );
    errdefer decoded.deinit(alloc);

    var histogram = try codec.buildHistogram(alloc, decoded);
    defer histogram.deinit(alloc);
    if (try huffmanPayloadBits(table, histogram) != payload_bits)
        return error.CorruptLiteralEncoding;
    return decoded;
}

fn decodeRans(
    alloc: Allocator,
    bits_per_elem: u8,
    count: usize,
    reader: *Reader,
) !Stream {
    const entry_count_u32 = try reader.readU32();
    const entry_count: usize = @intCast(entry_count_u32);
    if ((count == 0 and entry_count != 0) or
        (count != 0 and (entry_count == 0 or entry_count > count)) or
        entry_count > codec.RANS_PROB_SCALE)
        return error.CorruptLiteralEncoding;
    if (@as(u64, entry_count) > maxAlphabet(bits_per_elem))
        return error.CorruptLiteralEncoding;
    if (entry_count > reader.remaining() / 8)
        return error.TruncatedLiteralEncoding;

    var table = try readRansTable(
        alloc,
        reader,
        bits_per_elem,
        entry_count,
    );
    defer table.deinit(alloc);

    const payload_len_u64 = try reader.readU64();
    const payload_len = std.math.cast(usize, payload_len_u64) orelse
        return error.LiteralSizeOverflow;
    const payload = try reader.take(payload_len);
    try reader.finish();
    if (payload_len < 4)
        return error.CorruptLiteralEncoding;

    const result = try codec.ransDecodeWithState(
        alloc,
        payload,
        table,
        count,
        bits_per_elem,
    );
    var decoded = result.stream;
    errdefer decoded.deinit(alloc);
    if (result.consumed_bytes != payload.len or
        result.final_state != codec.RANS_L)
        return error.CorruptLiteralEncoding;

    // Re-encoding additionally establishes a unique byte representation for
    // every stream under this table.
    const canonical_payload = try codec.ransEncode(alloc, decoded, table);
    defer alloc.free(canonical_payload);
    if (!std.mem.eql(u8, payload, canonical_payload))
        return error.CorruptLiteralEncoding;
    return decoded;
}

fn readRansTable(
    alloc: Allocator,
    reader: *Reader,
    bits_per_elem: u8,
    entry_count: usize,
) !codec.RansTable {
    const symbols = try alloc.alloc(u32, entry_count);
    errdefer alloc.free(symbols);
    const info = try alloc.alloc(codec.RansSymbol, entry_count);
    errdefer alloc.free(info);

    var cumulative: u32 = 0;
    for (symbols, info, 0..) |*symbol, *symbol_info, index| {
        symbol.* = try reader.readU32();
        const frequency = try reader.readU32();
        if (frequency == 0 or
            (index != 0 and symbol.* <= symbols[index - 1]) or
            @as(u64, symbol.*) >= maxAlphabet(bits_per_elem))
            return error.CorruptLiteralEncoding;
        if (frequency > codec.RANS_PROB_SCALE - cumulative)
            return error.CorruptLiteralEncoding;
        symbol_info.* = .{ .freq = frequency, .cum = cumulative };
        cumulative += frequency;
    }
    if (entry_count != 0 and cumulative != codec.RANS_PROB_SCALE)
        return error.CorruptLiteralEncoding;
    return .{ .symbols = symbols, .info = info };
}

fn maxAlphabet(bits_per_elem: u8) u64 {
    return @as(u64, 1) << @intCast(bits_per_elem);
}

fn validateZeroPadding(payload: []const u8, bit_count: usize) !void {
    const used_in_last = bit_count & 7;
    if (used_in_last == 0 or payload.len == 0) return;
    const padding_bits: u3 = @intCast(8 - used_in_last);
    const padding_mask: u8 = (@as(u8, 1) << padding_bits) - 1;
    if (payload[payload.len - 1] & padding_mask != 0)
        return error.CorruptLiteralEncoding;
}

fn appendU32(
    alloc: Allocator,
    out: *std.ArrayList(u8),
    value: u32,
) Allocator.Error!void {
    var bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &bytes, value, .little);
    try out.appendSlice(alloc, &bytes);
}

fn appendU64(
    alloc: Allocator,
    out: *std.ArrayList(u8),
    value: u64,
) Allocator.Error!void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, value, .little);
    try out.appendSlice(alloc, &bytes);
}

const Reader = struct {
    bytes: []const u8,
    pos: usize = 0,

    fn remaining(self: Reader) usize {
        return self.bytes.len - self.pos;
    }

    fn take(self: *Reader, count: usize) ![]const u8 {
        if (count > self.remaining())
            return error.TruncatedLiteralEncoding;
        defer self.pos += count;
        return self.bytes[self.pos..][0..count];
    }

    fn byte(self: *Reader) !u8 {
        return (try self.take(1))[0];
    }

    fn readU32(self: *Reader) !u32 {
        return std.mem.readInt(u32, (try self.take(4))[0..4], .little);
    }

    fn readU64(self: *Reader) !u64 {
        return std.mem.readInt(u64, (try self.take(8))[0..8], .little);
    }

    fn finish(self: Reader) !void {
        if (self.pos != self.bytes.len)
            return error.TrailingLiteralBytes;
    }
};
