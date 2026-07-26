const std = @import("std");
const archive = @import("tensor_archive.zig");
const dsl = @import("dsl.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;

const safetensors_prefix = [_]u8{
    2,   0,   0, 0, 0, 0, 0, 0,
    '{', '}',
};

fn literalTensor(
    alloc: Allocator,
    dtype: types.Dtype,
    shape: []const u64,
    words: []const u32,
) !dsl.TensorProgram {
    var root = try dsl.Program.literal(alloc, dtype.bitWidth(), words);
    errdefer root.deinit(alloc);
    return dsl.TensorProgram.init(alloc, dtype, shape, root);
}

fn testReadUleb(bytes: []const u8, position: *usize) !u64 {
    var result: u64 = 0;
    for (0..10) |index| {
        if (position.* >= bytes.len) return error.Truncated;
        const byte = bytes[position.*];
        position.* += 1;
        result |= @as(u64, byte & 0x7f) << @intCast(index * 7);
        if (byte & 0x80 == 0) return result;
    }
    return error.IntegerOverflow;
}

fn appendTestUleb(
    alloc: Allocator,
    output: *std.ArrayList(u8),
    value: usize,
) !void {
    var remaining: u64 = @intCast(value);
    while (true) {
        var byte: u8 = @truncate(remaining & 0x7f);
        remaining >>= 7;
        if (remaining != 0) byte |= 0x80;
        try output.append(alloc, byte);
        if (remaining == 0) return;
    }
}

test "whole-tensor archive round trips multiple independent records and prefix" {
    const alloc = std.testing.allocator;
    var first = try literalTensor(
        alloc,
        .u8,
        &.{ 2, 2 },
        &.{ 1, 2, 3, 4 },
    );
    defer first.deinit(alloc);
    var second = try literalTensor(
        alloc,
        .f32,
        &.{3},
        &.{ 0x3f80_0000, 0xbf80_0000, 0x8000_0000 },
    );
    defer second.deinit(alloc);

    const encoded = try archive.build(alloc, &safetensors_prefix, &.{
        .{ .name = "small", .tensor_program = first },
        .{ .name = "weights", .tensor_program = second },
    });
    var encoded_owned = true;
    defer if (encoded_owned) alloc.free(encoded);

    var parsed = try archive.parseStructural(alloc, encoded, .{});
    defer parsed.deinit(alloc);

    // `Parsed` is fully owned: tensor records remain usable when the source
    // archive buffer is gone.
    alloc.free(encoded);
    encoded_owned = false;

    try std.testing.expectEqualSlices(
        u8,
        &safetensors_prefix,
        parsed.safetensors_prefix,
    );
    try std.testing.expectEqual(@as(usize, 2), parsed.records.len);
    try std.testing.expectEqualStrings("small", parsed.records[0].name);
    try std.testing.expectEqual(types.Dtype.u8, parsed.records[0].tensor_program.dtype);
    try std.testing.expectEqualSlices(
        u64,
        &.{ 2, 2 },
        parsed.records[0].tensor_program.shape,
    );

    var first_output = try archive.executeVerified(alloc, parsed.records[0]);
    defer first_output.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 4), first_output.count);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, first_output.data);

    var second_output = try archive.executeVerified(alloc, parsed.records[1]);
    defer second_output.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 3), second_output.count);
    try std.testing.expectEqual(@as(u32, 0x3f80_0000), second_output.getU32(0));
    try std.testing.expectEqual(@as(u32, 0xbf80_0000), second_output.getU32(1));
    try std.testing.expectEqual(@as(u32, 0x8000_0000), second_output.getU32(2));
}

test "segmented header and tensor frame APIs compose without hidden state" {
    const alloc = std.testing.allocator;
    var tensor = try literalTensor(alloc, .u16, &.{2}, &.{ 0x1234, 0xabcd });
    defer tensor.deinit(alloc);

    const header_bytes = try archive.encodeHeader(alloc, &safetensors_prefix, 1);
    defer alloc.free(header_bytes);
    const frame = try archive.encodeTensorRecord(alloc, "x", tensor);
    defer alloc.free(frame);

    var output: std.ArrayList(u8) = .empty;
    defer output.deinit(alloc);
    try output.appendSlice(alloc, header_bytes);
    try output.appendSlice(alloc, frame);

    const header = try archive.parseHeader(output.items, .{});
    try std.testing.expectEqual(@as(usize, 1), header.tensor_count);
    try std.testing.expectEqualSlices(
        u8,
        &safetensors_prefix,
        header.safetensors_prefix,
    );

    var position = header.next_offset;
    var streamed = try archive.nextTensorRecord(
        alloc,
        output.items,
        &position,
        .{},
    );
    defer streamed.deinit(alloc);
    try std.testing.expectEqual(output.items.len, position);

    var isolated = try archive.decodeTensorRecord(alloc, frame, .{});
    defer isolated.deinit(alloc);
    try std.testing.expectEqualStrings("x", isolated.name);
    var decoded = try archive.executeVerified(alloc, isolated);
    defer decoded.deinit(alloc);
    try std.testing.expectEqual(@as(u32, 0x1234), decoded.getU32(0));
    try std.testing.expectEqual(@as(u32, 0xabcd), decoded.getU32(1));
}

test "verified record API returns the checked stream without a second execution" {
    const alloc = std.testing.allocator;
    var tensor = try literalTensor(alloc, .u16, &.{2}, &.{ 0x1234, 0xabcd });
    defer tensor.deinit(alloc);

    const frame = try archive.encodeTensorRecord(alloc, "x", tensor);
    defer alloc.free(frame);

    var position: usize = 0;
    var verified = try archive.nextVerifiedTensorRecord(
        alloc,
        frame,
        &position,
        .{},
    );
    defer verified.deinit(alloc);
    try std.testing.expectEqual(frame.len, position);
    try std.testing.expectEqualStrings("x", verified.record.name);
    try std.testing.expectEqual(@as(u32, 0x1234), verified.decoded.getU32(0));
    try std.testing.expectEqual(@as(u32, 0xabcd), verified.decoded.getU32(1));

    const corrupted = try alloc.dupe(u8, frame);
    defer alloc.free(corrupted);
    corrupted[corrupted.len - 1] ^= 0x80;
    position = 0;
    try std.testing.expectError(
        error.ChecksumMismatch,
        archive.nextVerifiedTensorRecord(
            alloc,
            corrupted,
            &position,
            .{},
        ),
    );
    try std.testing.expectEqual(@as(usize, 0), position);
}

test "zero dimension makes later overflowing dimensions an empty tensor" {
    const alloc = std.testing.allocator;
    var tensor = try literalTensor(
        alloc,
        .u8,
        &.{ std.math.maxInt(u64), 2, 0 },
        &.{},
    );
    defer tensor.deinit(alloc);

    const frame = try archive.encodeTensorRecord(alloc, "empty", tensor);
    defer alloc.free(frame);
    var verified = try archive.decodeTensorRecord(alloc, frame, .{});
    defer verified.deinit(alloc);
    try std.testing.expectEqualSlices(
        u64,
        &.{ std.math.maxInt(u64), 2, 0 },
        verified.tensor_program.shape,
    );
    var output = try archive.executeVerified(alloc, verified);
    defer output.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 0), output.count);
    try std.testing.expectEqual(@as(usize, 0), output.data.len);
}

test "record rejects checksum corruption, program corruption, truncation, and trailing data" {
    const alloc = std.testing.allocator;
    var tensor = try literalTensor(alloc, .u8, &.{4}, &.{ 7, 8, 9, 10 });
    defer tensor.deinit(alloc);

    const clean = try archive.encodeTensorRecord(alloc, "x", tensor);
    defer alloc.free(clean);

    const bad_checksum = try alloc.dupe(u8, clean);
    defer alloc.free(bad_checksum);
    bad_checksum[bad_checksum.len - 1] ^= 0x80;
    try std.testing.expectError(
        error.ChecksumMismatch,
        archive.decodeTensorRecord(alloc, bad_checksum, .{}),
    );

    const bad_program = try alloc.dupe(u8, clean);
    defer alloc.free(bad_program);
    const magic_at = std.mem.indexOf(u8, bad_program, "BRPG") orelse
        return error.TestExpectedProgramMagic;
    bad_program[magic_at] = 'X';
    try std.testing.expectError(
        error.BadMagic,
        archive.decodeTensorRecord(alloc, bad_program, .{}),
    );

    try std.testing.expectError(
        error.Truncated,
        archive.decodeTensorRecord(alloc, clean[0 .. clean.len - 1], .{}),
    );

    const trailing = try alloc.alloc(u8, clean.len + 1);
    defer alloc.free(trailing);
    @memcpy(trailing[0..clean.len], clean);
    trailing[clean.len] = 0;
    try std.testing.expectError(
        error.TrailingBytes,
        archive.decodeTensorRecord(alloc, trailing, .{}),
    );
}

test "record rejects shape that disagrees with its program" {
    const alloc = std.testing.allocator;
    var tensor = try literalTensor(alloc, .u8, &.{4}, &.{ 1, 2, 3, 4 });
    defer tensor.deinit(alloc);

    const clean = try archive.encodeTensorRecord(alloc, "x", tensor);
    defer alloc.free(clean);
    const malformed = try alloc.dupe(u8, clean);
    defer alloc.free(malformed);

    var position: usize = 0;
    _ = try testReadUleb(malformed, &position); // body length
    position += 1; // record tag
    const name_len = try testReadUleb(malformed, &position);
    position += @intCast(name_len);
    position += 1; // dtype
    try std.testing.expectEqual(@as(u64, 1), try testReadUleb(malformed, &position));
    try std.testing.expectEqual(@as(u8, 4), malformed[position]);
    // A larger declared shape lets the program decode within the record
    // allocation cap, then exercises the exact TensorProgram binding check.
    malformed[position] = 5;

    try std.testing.expectError(
        error.TensorLengthMismatch,
        archive.decodeTensorRecord(alloc, malformed, .{}),
    );
}

test "archive limits and canonical ULEB checks reject adversarial framing" {
    const alloc = std.testing.allocator;
    var tensor = try literalTensor(alloc, .u8, &.{1}, &.{42});
    defer tensor.deinit(alloc);

    const encoded = try archive.build(alloc, &safetensors_prefix, &.{
        .{ .name = "x", .tensor_program = tensor },
    });
    defer alloc.free(encoded);

    try std.testing.expectError(
        error.TensorLimitExceeded,
        archive.parseHeader(encoded, .{ .max_tensors = 0 }),
    );
    try std.testing.expectError(
        error.PrefixLimitExceeded,
        archive.parseHeader(encoded, .{ .max_prefix_bytes = 9 }),
    );

    const header = try archive.parseHeader(encoded, .{});
    var position = header.next_offset;
    try std.testing.expectError(
        error.RecordLimitExceeded,
        archive.nextTensorRecord(
            alloc,
            encoded,
            &position,
            .{ .max_record_bytes = 0 },
        ),
    );

    const overlong_count = [_]u8{
        'B',  'R',  'T', 'A', archive.VERSION,
        0x80, 0x00,
    };
    try std.testing.expectError(
        error.OverlongUleb128,
        archive.parseHeader(&overlong_count, .{}),
    );

    const overflowing_count = [_]u8{
        'B',  'R',  'T',  'A',  archive.VERSION,
        0x80, 0x80, 0x80, 0x80, 0x80,
        0x80, 0x80, 0x80, 0x80, 0x80,
    };
    try std.testing.expectError(
        error.IntegerOverflow,
        archive.parseHeader(&overflowing_count, .{}),
    );
}

test "record shape clamps program allocations before tensor binding" {
    const alloc = std.testing.allocator;
    const million_zero_program = [_]u8{
        'B', 'R', 'P', 'G', 0x01,
        0x01, 0x08, // Lit<u8>
        0xc0, 0x84, 0x3d, // count = 1,000,000
        0x19, // literal body length = 25
        0x03, // rANS
        0x01, 0x00, 0x00, 0x00, // one table entry
        0x00, 0x00, 0x00, 0x00, // symbol 0
        0x00, 0x40, 0x00, 0x00, // frequency = RANS_PROB_SCALE
        0x04, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x80, 0x00, 0x00, // final state = RANS_L
    };

    var body: std.ArrayList(u8) = .empty;
    defer body.deinit(alloc);
    try body.append(alloc, 1); // record tag
    try appendTestUleb(alloc, &body, 1);
    try body.append(alloc, 'x');
    try body.append(alloc, 0x04); // stable U8 dtype id
    try appendTestUleb(alloc, &body, 1);
    try appendTestUleb(alloc, &body, 1); // shape = [1]
    try appendTestUleb(alloc, &body, million_zero_program.len);
    try body.appendSlice(alloc, &million_zero_program);
    try body.appendNTimes(alloc, 0, 32); // checksum, never reached

    var frame: std.ArrayList(u8) = .empty;
    defer frame.deinit(alloc);
    try appendTestUleb(alloc, &frame, body.items.len);
    try frame.appendSlice(alloc, body.items);

    var bounded_storage: [128]u8 = undefined;
    var fixed = std.heap.FixedBufferAllocator.init(&bounded_storage);
    try std.testing.expectError(
        error.OutputLimitExceeded,
        archive.decodeTensorRecord(fixed.allocator(), frame.items, .{}),
    );
}
