//! Focused tests for the physical entropy-codec primitives.
//!
//! These tests intentionally depend only on the current `types` and `codec`
//! modules. Canonical wire framing belongs to `literal_encoding`: in
//! particular, low-level `bitpackDecode` extracts exactly the requested
//! logical bits, while the literal decoder rejects non-zero tail padding.

const std = @import("std");
const codec = @import("codec.zig");
const types = @import("types.zig");

const Stream = types.Stream;

fn expectStreamsEqual(expected: Stream, actual: Stream) !void {
    try std.testing.expectEqual(expected.count, actual.count);
    try std.testing.expectEqual(expected.bits_per_elem, actual.bits_per_elem);
    for (0..expected.count) |index| {
        try std.testing.expectEqual(
            expected.getU32(index) & expected.mask(),
            actual.getU32(index) & actual.mask(),
        );
    }
}

fn widthMask(width: u8) u32 {
    return if (width == 32)
        std.math.maxInt(u32)
    else
        (@as(u32, 1) << @intCast(width)) - 1;
}

fn tailPaddingMask(bit_count: usize) u8 {
    const used: u8 = @intCast(bit_count & 7);
    if (used == 0) return 0;
    return (@as(u8, 1) << @intCast(8 - used)) - 1;
}

test "bitpack round trips every width and rejects truncation" {
    const alloc = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0xB17C_0DEC);
    const random = prng.random();

    for (1..33) |width_value| {
        const width: u8 = @intCast(width_value);
        const storage_width = types.roundUpToPow2(width);
        for ([_]usize{ 0, 1, 2, 7, 8, 9, 63, 257 }) |count| {
            var input = try Stream.init(alloc, count, storage_width);
            defer input.deinit(alloc);
            const mask = widthMask(width);
            for (0..count) |index|
                input.setU32(index, random.int(u32) & mask);

            const payload = try codec.bitpackEncode(alloc, input, width);
            defer alloc.free(payload);
            const bit_count = count * @as(usize, width);
            try std.testing.expectEqual((bit_count + 7) / 8, payload.len);
            try std.testing.expectEqual(
                @as(u64, @intCast(payload.len * 8)),
                codec.bitpackCostBits(input, width),
            );

            var decoded = try codec.bitpackDecode(
                alloc,
                payload,
                width,
                count,
                storage_width,
            );
            defer decoded.deinit(alloc);
            try expectStreamsEqual(input, decoded);

            if (payload.len != 0) {
                try std.testing.expectError(
                    error.CorruptBitpackStream,
                    codec.bitpackDecode(
                        alloc,
                        payload[0 .. payload.len - 1],
                        width,
                        count,
                        storage_width,
                    ),
                );
            }

            // The encoder always emits canonical zero padding. The low-level
            // decoder deliberately ignores bits outside `count * width`;
            // literal_encoding is the wire layer that rejects non-zero
            // padding before calling it.
            const padding_mask = tailPaddingMask(bit_count);
            if (padding_mask != 0) {
                try std.testing.expectEqual(
                    @as(u8, 0),
                    payload[payload.len - 1] & padding_mask,
                );
                const noncanonical = try alloc.dupe(u8, payload);
                defer alloc.free(noncanonical);
                noncanonical[noncanonical.len - 1] |= 1;
                try std.testing.expect(
                    noncanonical[noncanonical.len - 1] & padding_mask != 0,
                );
                var logical = try codec.bitpackDecode(
                    alloc,
                    noncanonical,
                    width,
                    count,
                    storage_width,
                );
                defer logical.deinit(alloc);
                try expectStreamsEqual(input, logical);
            }
        }
    }
}

test "bitpack size arithmetic rejects host-size overflow before allocation" {
    try std.testing.expectError(
        error.BitpackSizeOverflow,
        codec.bitpackDecode(
            std.testing.allocator,
            &.{},
            32,
            std.math.maxInt(usize),
            32,
        ),
    );
    try std.testing.expectError(
        error.InvalidBitpackWidth,
        codec.bitpackDecode(
            std.testing.allocator,
            &.{},
            0,
            1,
            8,
        ),
    );
}

test "canonical Huffman tables and payloads round trip deterministically" {
    const alloc = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x4855_4646);
    const random = prng.random();

    for ([_]u8{ 8, 16, 32 }) |bits_per_elem| {
        var input = try Stream.init(alloc, 1537, bits_per_elem);
        defer input.deinit(alloc);
        for (0..input.count) |index| {
            const value = if (index % 11 < 7)
                @as(u32, @intCast(index % 3))
            else if (bits_per_elem == 32)
                random.int(u32)
            else
                random.int(u32) & widthMask(bits_per_elem);
            input.setU32(index, value);
        }

        var first = try codec.huffmanBuild(alloc, input);
        defer first.deinit(alloc);
        var second = try codec.huffmanBuild(alloc, input);
        defer second.deinit(alloc);
        try std.testing.expectEqualSlices(
            codec.HuffmanTable.Entry,
            first.entries,
            second.entries,
        );
        for (first.entries, 0..) |entry, index| {
            try std.testing.expect(entry.len >= 1 and entry.len <= 32);
            if (index != 0) {
                const previous = first.entries[index - 1];
                try std.testing.expect(
                    previous.len < entry.len or
                        (previous.len == entry.len and previous.sym < entry.sym),
                );
            }
        }

        const first_payload = try codec.huffmanEncode(alloc, input, first);
        defer alloc.free(first_payload);
        const second_payload = try codec.huffmanEncode(alloc, input, second);
        defer alloc.free(second_payload);
        try std.testing.expectEqualSlices(u8, first_payload, second_payload);

        var decoded = try codec.huffmanDecode(
            alloc,
            first_payload,
            first,
            input.count,
            input.bits_per_elem,
        );
        defer decoded.deinit(alloc);
        try expectStreamsEqual(input, decoded);

        var histogram = try codec.buildHistogram(alloc, input);
        defer histogram.deinit(alloc);
        try std.testing.expectEqual(
            @as(u64, @intCast(first_payload.len)),
            codec.huffmanPayloadBytes(first, histogram),
        );

        try std.testing.expect(first_payload.len > 1);
        try std.testing.expectError(
            error.CorruptHuffmanStream,
            codec.huffmanDecode(
                alloc,
                first_payload[0 .. first_payload.len - 1],
                first,
                input.count,
                input.bits_per_elem,
            ),
        );
    }
}

test "canonical Huffman long-code fallback preserves sparse symbols" {
    const alloc = std.testing.allocator;
    const frequencies = [_]usize{
        1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987,
    };
    var total: usize = 0;
    for (frequencies) |frequency| total += frequency;

    var input = try Stream.init(alloc, total, 32);
    defer input.deinit(alloc);
    var position: usize = 0;
    for (frequencies, 0..) |frequency, symbol_index| {
        const symbol = @as(u32, @intCast(symbol_index)) * 0x0101_0101;
        for (0..frequency) |_| {
            input.setU32(position, symbol);
            position += 1;
        }
    }

    var table = try codec.huffmanBuild(alloc, input);
    defer table.deinit(alloc);
    try std.testing.expect(table.entries[table.entries.len - 1].len > 12);

    const payload = try codec.huffmanEncode(alloc, input, table);
    defer alloc.free(payload);
    var decoded = try codec.huffmanDecode(
        alloc,
        payload,
        table,
        input.count,
        input.bits_per_elem,
    );
    defer decoded.deinit(alloc);
    try expectStreamsEqual(input, decoded);
}

test "Huffman decoder rejects malicious tables and truncated streams" {
    const alloc = std.testing.allocator;

    const empty: codec.HuffmanTable = .{ .entries = &.{} };
    try std.testing.expectError(
        error.CorruptHuffmanStream,
        codec.huffmanDecode(alloc, &.{}, empty, 1, 8),
    );

    var zero_entries = [_]codec.HuffmanTable.Entry{
        .{ .sym = 0, .len = 0 },
    };
    try std.testing.expectError(
        error.CorruptHuffmanStream,
        codec.huffmanDecode(
            alloc,
            &.{0},
            .{ .entries = &zero_entries },
            1,
            8,
        ),
    );

    var long_entries = [_]codec.HuffmanTable.Entry{
        .{ .sym = 0, .len = 33 },
    };
    try std.testing.expectError(
        error.CorruptHuffmanStream,
        codec.huffmanDecode(
            alloc,
            &.{0},
            .{ .entries = &long_entries },
            1,
            8,
        ),
    );

    var unsorted_entries = [_]codec.HuffmanTable.Entry{
        .{ .sym = 1, .len = 2 },
        .{ .sym = 0, .len = 1 },
    };
    try std.testing.expectError(
        error.CorruptHuffmanStream,
        codec.huffmanDecode(
            alloc,
            &.{0},
            .{ .entries = &unsorted_entries },
            1,
            8,
        ),
    );

    var oversubscribed_entries = [_]codec.HuffmanTable.Entry{
        .{ .sym = 0, .len = 1 },
        .{ .sym = 1, .len = 1 },
        .{ .sym = 2, .len = 1 },
    };
    try std.testing.expectError(
        error.CorruptHuffmanStream,
        codec.huffmanDecode(
            alloc,
            &.{0},
            .{ .entries = &oversubscribed_entries },
            1,
            8,
        ),
    );

    var wide_entries = [_]codec.HuffmanTable.Entry{
        .{ .sym = 256, .len = 1 },
    };
    try std.testing.expectError(
        error.CorruptHuffmanStream,
        codec.huffmanDecode(
            alloc,
            &.{0},
            .{ .entries = &wide_entries },
            1,
            8,
        ),
    );

    var one_bit_entries = [_]codec.HuffmanTable.Entry{
        .{ .sym = 7, .len = 1 },
    };
    try std.testing.expectError(
        error.CorruptHuffmanStream,
        codec.huffmanDecode(
            alloc,
            &.{0},
            .{ .entries = &one_bit_entries },
            9,
            8,
        ),
    );
}

test "Huffman builder rejects code lengths unsupported by the decoder" {
    const alloc = std.testing.allocator;
    var pairs: [34]codec.Histogram.Pair = undefined;
    var low: u64 = 1;
    var high: u64 = 1;
    for (&pairs, 0..) |*pair, symbol| {
        pair.* = .{ .sym = @intCast(symbol), .count = low };
        const next = low + high;
        low = high;
        high = next;
    }
    const histogram: codec.Histogram = .{ .pairs = &pairs };
    try std.testing.expectError(
        error.HuffmanCodeTooLong,
        codec.huffmanFromHist(alloc, histogram, 8),
    );
}

test "rANS round trip reports canonical state and exact consumption" {
    const alloc = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(0x5241_4e53);
    const random = prng.random();

    for ([_]u8{ 8, 16, 32 }) |bits_per_elem| {
        var input = try Stream.init(alloc, 4097, bits_per_elem);
        defer input.deinit(alloc);
        const mask = widthMask(bits_per_elem);
        for (0..input.count) |index| {
            const value = if (index % 17 < 13)
                @as(u32, @intCast(index % 5))
            else if (bits_per_elem == 32)
                random.int(u32)
            else
                random.int(u32) & mask;
            input.setU32(index, value);
        }

        var table = try codec.ransBuild(alloc, input);
        defer table.deinit(alloc);
        try std.testing.expectEqual(table.symbols.len, table.info.len);
        var cumulative: u32 = 0;
        for (table.info, 0..) |info, index| {
            try std.testing.expect(info.freq > 0);
            try std.testing.expectEqual(cumulative, info.cum);
            cumulative += info.freq;
            if (index != 0)
                try std.testing.expect(table.symbols[index - 1] < table.symbols[index]);
        }
        try std.testing.expectEqual(codec.RANS_PROB_SCALE, cumulative);

        const first_payload = try codec.ransEncode(alloc, input, table);
        defer alloc.free(first_payload);
        const second_payload = try codec.ransEncode(alloc, input, table);
        defer alloc.free(second_payload);
        try std.testing.expectEqualSlices(u8, first_payload, second_payload);

        var decoded = try codec.ransDecodeWithState(
            alloc,
            first_payload,
            table,
            input.count,
            input.bits_per_elem,
        );
        defer decoded.stream.deinit(alloc);
        try expectStreamsEqual(input, decoded.stream);
        try std.testing.expectEqual(first_payload.len, decoded.consumed_bytes);
        try std.testing.expectEqual(codec.RANS_L, decoded.final_state);

        const with_trailing = try alloc.alloc(u8, first_payload.len + 1);
        defer alloc.free(with_trailing);
        @memcpy(with_trailing[0..first_payload.len], first_payload);
        with_trailing[with_trailing.len - 1] = 0xa5;
        var bounded = try codec.ransDecodeWithState(
            alloc,
            with_trailing,
            table,
            input.count,
            input.bits_per_elem,
        );
        defer bounded.stream.deinit(alloc);
        try expectStreamsEqual(input, bounded.stream);
        try std.testing.expectEqual(first_payload.len, bounded.consumed_bytes);
        try std.testing.expectEqual(codec.RANS_L, bounded.final_state);

        try std.testing.expectError(
            error.CorruptRansStream,
            codec.ransDecodeWithState(
                alloc,
                first_payload[0..3],
                table,
                input.count,
                input.bits_per_elem,
            ),
        );
        try std.testing.expectError(
            error.CorruptRansStream,
            codec.ransDecodeWithState(
                alloc,
                first_payload[0 .. first_payload.len - 1],
                table,
                input.count,
                input.bits_per_elem,
            ),
        );

        var histogram = try codec.buildHistogram(alloc, input);
        defer histogram.deinit(alloc);
        try std.testing.expect(
            codec.ransLowerBytes(table, histogram) <= first_payload.len,
        );
    }
}
