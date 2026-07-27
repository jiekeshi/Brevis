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
const builtin = @import("builtin");
const codec = @import("codec.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const Stream = types.Stream;

const AnalysisMetrics = struct {
    threadlocal var elements: usize = 0;
    threadlocal var rans_payload_encodes: usize = 0;
    threadlocal var validation_elements: usize = 0;
};

pub const testing = if (builtin.is_test) struct {
    pub fn resetWidthScan() void {
        AnalysisMetrics.elements = 0;
    }

    pub fn widthScanElements() usize {
        return AnalysisMetrics.elements;
    }

    pub fn resetRansPayloadEncodes() void {
        AnalysisMetrics.rans_payload_encodes = 0;
    }

    pub fn ransPayloadEncodes() usize {
        return AnalysisMetrics.rans_payload_encodes;
    }

    pub fn resetValidationScan() void {
        AnalysisMetrics.validation_elements = 0;
    }

    pub fn validationScanElements() usize {
        return AnalysisMetrics.validation_elements;
    }
} else struct {};

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

/// Select by exact wire size, then materialize only the winning body.
pub fn encodeBest(alloc: Allocator, stream: Stream) !OwnedEncoding {
    var prepared = try prepareBest(alloc, stream);
    defer prepared.deinit(alloc);
    var output: std.ArrayList(u8) = .empty;
    errdefer output.deinit(alloc);
    try prepared.emitBody(alloc, &output);
    return .{
        .tag = prepared.tag(),
        .body = try output.toOwnedSlice(alloc),
    };
}

pub fn encodedSize(alloc: Allocator, stream: Stream) !usize {
    var analysis = (try analyzeBest(alloc, stream, NO_LIMIT)).?;
    defer analysis.deinit(alloc);
    return analysis.best.size;
}

pub const PreparedEncoding = struct {
    stream: Stream,
    analysis: Analysis,

    pub fn deinit(self: *PreparedEncoding, alloc: Allocator) void {
        self.analysis.deinit(alloc);
    }

    pub fn tag(self: PreparedEncoding) Tag {
        return self.analysis.best.tag;
    }

    pub fn wireSize(self: PreparedEncoding) usize {
        return self.analysis.best.size;
    }

    pub fn emitBody(
        self: PreparedEncoding,
        alloc: Allocator,
        output: *std.ArrayList(u8),
    ) !void {
        const start = output.items.len;
        try output.ensureUnusedCapacity(alloc, self.wireSize());
        switch (self.tag()) {
            .raw => try emitRaw(alloc, self.stream, output),
            .bitpack => try emitBitpack(
                alloc,
                self.stream,
                self.analysis.bitpack_width,
                output,
            ),
            .huffman => try emitHuffman(
                alloc,
                self.stream,
                self.analysis.huffman.?,
                self.analysis.huffman_payload_bits,
                output,
            ),
            .rans => try emitRans(
                alloc,
                self.stream,
                self.analysis.rans.?,
                output,
            ),
        }
        if (output.items.len - start != self.wireSize())
            return error.LiteralSizeMismatch;
    }
};

pub fn prepareBest(
    alloc: Allocator,
    stream: Stream,
) !PreparedEncoding {
    return .{
        .stream = stream,
        .analysis = (try analyzeBest(alloc, stream, NO_LIMIT)).?,
    };
}

/// Prepare only when the exact body stays below `limit`. A null result means
/// every encoding provably reaches `limit`, so the caller can abandon the
/// enclosing candidate without paying for any exact payload size.
pub fn prepareBestWithin(
    alloc: Allocator,
    stream: Stream,
    limit: usize,
) !?PreparedEncoding {
    const analysis = (try analyzeBest(alloc, stream, limit)) orelse return null;
    return .{ .stream = stream, .analysis = analysis };
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

/// Decode a complete emitted body and validate the selected codec.
pub fn decodeBody(
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
    try validateStreamStorage(stream);
    try validateStreamValues(stream);
}

fn validateStreamStorage(stream: Stream) !void {
    const expected = try checkedStorageBytes(stream.bits_per_elem, stream.count);
    if (stream.data.len != expected) return error.InvalidLiteralStream;
}

fn validateStreamValues(stream: Stream) !void {
    if (stream.bits_per_elem == types.roundUpToPow2(stream.bits_per_elem)) return;
    if (builtin.is_test) AnalysisMetrics.validation_elements += stream.count;
    if (!stream.valuesFitWidth()) return error.InvalidLiteralStream;
}

fn checkedStorageBytes(bits_per_elem: u8, count: usize) !usize {
    if (bits_per_elem == 0 or bits_per_elem > 32)
        return error.InvalidLiteralWidth;
    const elem_bytes: usize = types.roundUpToPow2(bits_per_elem) / 8;
    return std.math.mul(usize, count, elem_bytes) catch
        return error.LiteralSizeOverflow;
}

fn emitRaw(
    alloc: Allocator,
    stream: Stream,
    output: *std.ArrayList(u8),
) !void {
    try output.append(alloc, @intFromEnum(Tag.raw));
    try output.appendSlice(alloc, stream.data);
}

fn emitBitpack(
    alloc: Allocator,
    stream: Stream,
    width: u8,
    output: *std.ArrayList(u8),
) !void {
    const payload = try codec.bitpackEncode(alloc, stream, width);
    defer alloc.free(payload);

    try output.append(alloc, @intFromEnum(Tag.bitpack));
    try output.append(alloc, width);
    try output.appendSlice(alloc, payload);
}

fn requiredBits(stream: Stream) u8 {
    var combined: u32 = 0;
    const top_bit = @as(u32, 1) <<
        @intCast(stream.bits_per_elem - 1);
    for (0..stream.count) |index| {
        if (builtin.is_test) AnalysisMetrics.elements += 1;
        combined |= stream.getU32(index);
        if (combined & top_bit != 0) return stream.bits_per_elem;
    }
    return if (combined == 0) 1 else @intCast(32 - @clz(combined));
}

const Best = struct {
    tag: Tag,
    size: usize,
};

const Analysis = struct {
    best: Best,
    bitpack_width: u8 = 1,
    huffman: ?codec.HuffmanTable = null,
    huffman_payload_bits: u64 = 0,
    rans: ?codec.RansTable = null,

    fn deinit(self: *Analysis, alloc: Allocator) void {
        if (self.huffman) |*table| table.deinit(alloc);
        if (self.rans) |*table| table.deinit(alloc);
    }
};

pub const NO_LIMIT: usize = std.math.maxInt(usize);

/// Zero-order entropy of `histogram`, rounded down and shaded by one byte so
/// accumulated floating-point error can never raise it above the true value.
/// No symbol-wise encoding can code the stream below it, so it lower-bounds
/// every payload this module can select.
fn entropyLowerBytes(histogram: codec.Histogram, count: usize) usize {
    const total: f64 = @floatFromInt(count);
    var bits: f64 = 0;
    for (histogram.pairs) |pair| {
        const occurrences: f64 = @floatFromInt(pair.count);
        bits -= occurrences * std.math.log2(occurrences / total);
    }
    const bytes: usize = @intFromFloat(@floor(bits / 8.0));
    return bytes -| 1;
}

fn analyzeBest(
    alloc: Allocator,
    stream: Stream,
    limit: usize,
) !?Analysis {
    try validateStreamStorage(stream);

    var analysis = Analysis{
        .best = .{
            .tag = .raw,
            .size = try addSize(1, stream.data.len),
        },
    };
    errdefer analysis.deinit(alloc);
    if (stream.count == 0)
        return if (analysis.best.size >= limit) null else analysis;

    var histogram = codec.buildHistogram(alloc, stream) catch |err| switch (err) {
        error.AlphabetTooLarge => {
            try validateStreamValues(stream);
            analysis.bitpack_width = requiredBits(stream);
            try analyzeBitpack(stream, &analysis);
            return if (analysis.best.size >= limit) null else analysis;
        },
        else => return err,
    };
    defer histogram.deinit(alloc);
    if (histogram.requiredBits() > stream.bits_per_elem)
        return error.InvalidLiteralStream;
    if (limit != NO_LIMIT and
        1 + entropyLowerBytes(histogram, stream.count) >= limit)
        return null;
    analysis.bitpack_width = histogram.requiredBits();
    try analyzeBitpack(stream, &analysis);

    if (codec.huffmanFromHist(
        alloc,
        histogram,
        stream.bits_per_elem,
    ) catch |err| switch (err) {
        error.HuffmanCodeTooLong, error.AlphabetTooLarge => null,
        else => return err,
    }) |table_value| {
        analysis.huffman = table_value;
        analysis.huffman_payload_bits = try huffmanPayloadBits(
            analysis.huffman.?,
            histogram,
        );
        const payload_bytes_u64 = std.math.add(
            u64,
            analysis.huffman_payload_bits,
            7,
        ) catch return error.LiteralSizeOverflow;
        const payload_bytes = std.math.cast(
            usize,
            payload_bytes_u64 / 8,
        ) orelse return error.LiteralSizeOverflow;
        const entries_bytes = std.math.mul(
            usize,
            analysis.huffman.?.entries.len,
            5,
        ) catch return error.LiteralSizeOverflow;
        selectSmaller(
            &analysis.best.tag,
            &analysis.best.size,
            .huffman,
            try addSize(try addSize(13, entries_bytes), payload_bytes),
        );
    }

    if (codec.ransFromHist(
        alloc,
        histogram,
        stream.count,
    ) catch |err| switch (err) {
        error.AlphabetTooLarge => null,
        else => return err,
    }) |table_value| {
        analysis.rans = table_value;
        const entries_bytes = std.math.mul(
            usize,
            analysis.rans.?.symbols.len,
            8,
        ) catch return error.LiteralSizeOverflow;
        const fixed_bytes = try addSize(13, entries_bytes);
        const lower_payload = std.math.cast(
            usize,
            codec.ransLowerBytes(analysis.rans.?, histogram),
        ) orelse std.math.maxInt(usize);
        const lower_size = std.math.add(
            usize,
            fixed_bytes,
            lower_payload,
        ) catch std.math.maxInt(usize);
        if (lower_size < @min(analysis.best.size, limit)) {
            const payload_size = try codec.ransEncodedSize(
                alloc,
                stream,
                analysis.rans.?,
            );
            selectSmaller(
                &analysis.best.tag,
                &analysis.best.size,
                .rans,
                try addSize(fixed_bytes, payload_size),
            );
        }
    }

    if (analysis.best.size >= limit) {
        analysis.deinit(alloc);
        return null;
    }
    return analysis;
}

fn analyzeBitpack(stream: Stream, analysis: *Analysis) !void {
    const bit_count = std.math.mul(
        usize,
        stream.count,
        @as(usize, analysis.bitpack_width),
    ) catch return error.LiteralSizeOverflow;
    const packed_bytes = try addSize(bit_count, 7) / 8;
    selectSmaller(
        &analysis.best.tag,
        &analysis.best.size,
        .bitpack,
        try addSize(2, packed_bytes),
    );
}

fn addSize(left: usize, right: usize) !usize {
    return std.math.add(usize, left, right) catch
        error.LiteralSizeOverflow;
}

fn selectSmaller(
    best: *Tag,
    best_size: *usize,
    candidate: Tag,
    candidate_size: usize,
) void {
    if (candidate_size < best_size.* or
        (candidate_size == best_size.* and
            @intFromEnum(candidate) < @intFromEnum(best.*)))
    {
        best.* = candidate;
        best_size.* = candidate_size;
    }
}

fn emitHuffman(
    alloc: Allocator,
    stream: Stream,
    table: codec.HuffmanTable,
    payload_bits: u64,
    output: *std.ArrayList(u8),
) !void {
    const payload_bytes = std.math.cast(
        usize,
        (std.math.add(u64, payload_bits, 7) catch
            return error.LiteralSizeOverflow) / 8,
    ) orelse return error.LiteralSizeOverflow;
    try output.append(alloc, @intFromEnum(Tag.huffman));
    try appendU32(alloc, output, @intCast(table.entries.len));
    for (table.entries) |entry| {
        try appendU32(alloc, output, entry.sym);
        try output.append(alloc, entry.len);
    }
    try appendU64(alloc, output, payload_bits);
    const payload = output.addManyAsSliceAssumeCapacity(payload_bytes);
    try codec.huffmanEncodeIntoSlice(alloc, stream, table, payload);
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

fn emitRans(
    alloc: Allocator,
    stream: Stream,
    table: codec.RansTable,
    output: *std.ArrayList(u8),
) !void {
    const payload = try codec.ransEncode(alloc, stream, table);
    defer alloc.free(payload);
    if (builtin.is_test)
        AnalysisMetrics.rans_payload_encodes += 1;

    try output.append(alloc, @intFromEnum(Tag.rans));
    try appendU32(alloc, output, @intCast(table.symbols.len));
    for (table.symbols, table.info) |symbol, info| {
        try appendU32(alloc, output, symbol);
        try appendU32(alloc, output, info.freq);
    }
    try appendU64(alloc, output, @intCast(payload.len));
    try output.appendSlice(alloc, payload);
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

    var stream = try Stream.initUninitialized(alloc, count, bits_per_elem);
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
