//! Contract tests for the self-contained physical encodings of DSL literals.

const std = @import("std");
const codec = @import("codec.zig");
const literal = @import("literal_encoding.zig");
const types = @import("types.zig");

const Stream = types.Stream;

fn streamFromWords(
    alloc: std.mem.Allocator,
    bits_per_elem: u8,
    words: []const u32,
) !Stream {
    var stream = try Stream.init(alloc, words.len, bits_per_elem);
    for (words, 0..) |word, index| stream.setU32(index, word);
    return stream;
}

fn expectStreamsEqual(expected: Stream, actual: Stream) !void {
    try std.testing.expectEqual(expected.bits_per_elem, actual.bits_per_elem);
    try std.testing.expectEqual(expected.count, actual.count);
    try std.testing.expectEqualSlices(u8, expected.data, actual.data);
}

fn expectWireAndDecode(
    alloc: std.mem.Allocator,
    input: Stream,
    encoding: literal.OwnedEncoding,
) !void {
    var emitted: std.ArrayList(u8) = .empty;
    defer emitted.deinit(alloc);

    try literal.emitBody(alloc, &emitted, encoding);
    try std.testing.expectEqual(literal.wireSize(encoding), emitted.items.len);
    try std.testing.expectEqual(encoding.wireSize(), emitted.items.len);
    try std.testing.expectEqualSlices(u8, encoding.body, emitted.items);

    var decoded = try literal.decode(
        alloc,
        input.bits_per_elem,
        input.count,
        encoding,
    );
    defer decoded.deinit(alloc);
    try expectStreamsEqual(input, decoded);

    var decoded_body = try literal.decodeBody(
        alloc,
        input.bits_per_elem,
        input.count,
        emitted.items,
    );
    defer decoded_body.deinit(alloc);
    try expectStreamsEqual(input, decoded_body);
}

test "uniform bytes choose the stable raw fallback" {
    const alloc = std.testing.allocator;
    var input = try Stream.init(alloc, 256, 8);
    defer input.deinit(alloc);
    for (0..input.count) |index| input.setU32(index, @intCast(index));

    var encoding = try literal.encodeBest(alloc, input);
    defer encoding.deinit(alloc);

    try std.testing.expectEqual(literal.Tag.raw, encoding.tag);
    try expectWireAndDecode(alloc, input, encoding);
}

test "equal complete body sizes use stable tag order" {
    const alloc = std.testing.allocator;
    var input = try streamFromWords(alloc, 8, &.{ 8, 0 });
    defer input.deinit(alloc);

    var encoding = try literal.encodeBest(alloc, input);
    defer encoding.deinit(alloc);

    // raw is tag 0 and costs 1 + 2 bytes. Four-bit packing is also 3 bytes:
    // tag + width + one payload byte.
    try std.testing.expectEqual(@as(usize, 3), encoding.wireSize());
    try std.testing.expectEqual(literal.Tag.raw, encoding.tag);
    try expectWireAndDecode(alloc, input, encoding);
}

test "small nonnegative values choose bitpack" {
    const alloc = std.testing.allocator;
    var input = try Stream.init(alloc, 128, 8);
    defer input.deinit(alloc);
    for (0..input.count) |index| input.setU32(index, @intCast(index & 1));

    var encoding = try literal.encodeBest(alloc, input);
    defer encoding.deinit(alloc);

    try std.testing.expectEqual(literal.Tag.bitpack, encoding.tag);
    try expectWireAndDecode(alloc, input, encoding);
}

test "balanced wide binary alphabet chooses canonical Huffman" {
    const alloc = std.testing.allocator;
    var input = try Stream.init(alloc, 512, 16);
    defer input.deinit(alloc);
    for (0..input.count) |index|
        input.setU32(index, if (index & 1 == 0) 0 else 0xffff);

    var encoding = try literal.encodeBest(alloc, input);
    defer encoding.deinit(alloc);

    try std.testing.expectEqual(literal.Tag.huffman, encoding.tag);
    try expectWireAndDecode(alloc, input, encoding);
}

test "very skewed wide alphabet chooses rANS" {
    const alloc = std.testing.allocator;
    var input = try Stream.init(alloc, 4096, 16);
    defer input.deinit(alloc);
    for (0..input.count) |index|
        input.setU32(index, if (index % 1024 == 0) 0xffff else 0);

    var encoding = try literal.encodeBest(alloc, input);
    defer encoding.deinit(alloc);

    try std.testing.expectEqual(literal.Tag.rans, encoding.tag);
    try expectWireAndDecode(alloc, input, encoding);
}

test "an inapplicable rANS alphabet is skipped without losing the literal" {
    const alloc = std.testing.allocator;
    const count: usize = codec.RANS_PROB_SCALE + 1;
    var input = try Stream.init(alloc, count, 32);
    defer input.deinit(alloc);
    for (0..input.count) |index| input.setU32(index, @intCast(index));

    var encoding = try literal.encodeBest(alloc, input);
    defer encoding.deinit(alloc);

    try std.testing.expect(encoding.tag != .rans);
    try expectWireAndDecode(alloc, input, encoding);
}

test "empty and boundary-width literals round trip canonically" {
    const alloc = std.testing.allocator;

    var empty = try Stream.init(alloc, 0, 1);
    defer empty.deinit(alloc);
    var empty_encoding = try literal.encodeBest(alloc, empty);
    defer empty_encoding.deinit(alloc);
    try std.testing.expectEqual(literal.Tag.raw, empty_encoding.tag);
    try std.testing.expectEqualSlices(
        u8,
        &.{@intFromEnum(literal.Tag.raw)},
        empty_encoding.body,
    );
    try expectWireAndDecode(alloc, empty, empty_encoding);

    var wide = try streamFromWords(
        alloc,
        32,
        &.{ 0, std.math.maxInt(u32) },
    );
    defer wide.deinit(alloc);
    var wide_encoding = try literal.encodeBest(alloc, wide);
    defer wide_encoding.deinit(alloc);
    try expectWireAndDecode(alloc, wide, wide_encoding);
}

test "Huffman body rejects a symbol repeated at different code lengths" {
    const alloc = std.testing.allocator;
    const body = [_]u8{
        @intFromEnum(literal.Tag.huffman),
        2, 0, 0, 0, // entry count
        0, 0, 0, 0, 1, // symbol 0, length 1
        0, 0, 0, 0, 2, // symbol 0 again, length 2
        3, 0, 0, 0, 0, 0, 0, 0, // claimed payload bits
        0, // zero-padded payload
    };

    try std.testing.expectError(
        error.CorruptLiteralEncoding,
        literal.decodeBody(alloc, 8, 1, &body),
    );
}

test "Huffman decoder rejects an inapplicable alphabet before allocation" {
    const entry_count: u32 = codec.MAX_HISTOGRAM_SYMBOLS + 1;
    var body = [_]u8{ @intFromEnum(literal.Tag.huffman), 0, 0, 0, 0 };
    std.mem.writeInt(u32, body[1..5], entry_count, .little);
    var no_allocation_storage: [1]u8 = undefined;
    var fixed = std.heap.FixedBufferAllocator.init(&no_allocation_storage);

    try std.testing.expectError(
        error.CorruptLiteralEncoding,
        literal.decodeBody(
            fixed.allocator(),
            32,
            entry_count,
            &body,
        ),
    );
}

test "public decoder rejects a valid Huffman body when a smaller codec wins" {
    const alloc = std.testing.allocator;
    // One zero encoded by a valid, canonical one-symbol Huffman table. The
    // Huffman representation itself is unambiguous, but raw costs only two
    // bytes and is therefore the globally canonical literal body.
    const non_minimal = [_]u8{
        @intFromEnum(literal.Tag.huffman),
        1, 0, 0, 0, // entry count
        0, 0, 0, 0, 1, // symbol 0, code length 1
        1, 0, 0, 0, 0, 0, 0, 0, // payload bits
        0, // the one-bit canonical code, followed by zero padding
    };

    try std.testing.expectError(
        error.NonCanonicalLiteralEncoding,
        literal.decodeBody(alloc, 8, 1, &non_minimal),
    );
}

test "bitpacked literal rejects non-zero tail padding" {
    const alloc = std.testing.allocator;
    const noncanonical = [_]u8{
        @intFromEnum(literal.Tag.bitpack),
        3,
        0b1010_0001,
    };
    try std.testing.expectError(
        error.CorruptLiteralEncoding,
        literal.decodeBody(alloc, 8, 1, &noncanonical),
    );
}

test "rANS body rejects payload bytes not consumed by the canonical stream" {
    const alloc = std.testing.allocator;
    var input = try Stream.init(alloc, 4096, 16);
    defer input.deinit(alloc);
    for (0..input.count) |index|
        input.setU32(index, if (index % 1024 == 0) 0xffff else 0);

    var encoding = try literal.encodeBest(alloc, input);
    defer encoding.deinit(alloc);
    try std.testing.expectEqual(literal.Tag.rans, encoding.tag);

    const corrupt = try alloc.alloc(u8, encoding.body.len + 1);
    defer alloc.free(corrupt);
    @memcpy(corrupt[0..encoding.body.len], encoding.body);
    corrupt[corrupt.len - 1] = 0;

    const entry_count: usize = @intCast(std.mem.readInt(
        u32,
        corrupt[1..5],
        .little,
    ));
    const payload_len_offset = 5 + entry_count * 8;
    const old_payload_len = std.mem.readInt(
        u64,
        corrupt[payload_len_offset..][0..8],
        .little,
    );
    std.mem.writeInt(
        u64,
        corrupt[payload_len_offset..][0..8],
        old_payload_len + 1,
        .little,
    );

    try std.testing.expectError(
        error.CorruptLiteralEncoding,
        literal.decodeBody(
            alloc,
            input.bits_per_elem,
            input.count,
            corrupt,
        ),
    );
}

test "large 32-bit alphabets retain the total raw-bitpack literal fallback" {
    const alloc = std.testing.allocator;
    var input = try Stream.init(
        alloc,
        codec.MAX_HISTOGRAM_SYMBOLS + 1,
        32,
    );
    defer input.deinit(alloc);
    for (0..input.count) |index| input.setU32(index, @intCast(index));

    try std.testing.expectError(
        error.AlphabetTooLarge,
        codec.buildHistogram(alloc, input),
    );

    var encoding = try literal.encodeBest(alloc, input);
    defer encoding.deinit(alloc);
    try std.testing.expect(
        encoding.tag == .raw or encoding.tag == .bitpack,
    );
    var decoded = try literal.decodeBody(
        alloc,
        input.bits_per_elem,
        input.count,
        encoding.body,
    );
    defer decoded.deinit(alloc);
    try expectStreamsEqual(input, decoded);
}
