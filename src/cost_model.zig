//! Encoder-only estimate of the bytes an open hole still owes.
//!
//! Grammar description length carries no information about the objective: rule
//! costs are nonnegative and paid per node, so a score built from them alone is
//! monotone in derivation length and always prefers the trivial `Lit`. This is
//! the missing data term, modelled over the same three regimes the literal
//! codec chooses between: raw, minimum-width bit packing, and a table-driven
//! entropy body. It reorders exploration only.

const std = @import("std");
const dsl = @import("dsl.zig");
const grammar = @import("grammar.zig");
const literal_encoding = @import("literal_encoding.zig");
const semantics = @import("semantics.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const Dtype = types.Dtype;
const Stream = types.Stream;

/// Estimates are Q10 bits so a per-element entropy keeps sub-bit resolution.
pub const BIT_SCALE: u64 = 1024;

pub const DEFAULT_SAMPLE_ELEMENTS: usize = 4096;
const MIN_SAMPLE_ELEMENTS: usize = 16;
const MAX_SAMPLE_ELEMENTS: usize = 1 << 16;

/// Widest arity in the grammar: one bit plane per bit of a 32-bit word.
pub const MAX_CHILDREN: usize = 32;

/// Tag, width, length, and codec framing. Keeps many-hole completions from
/// looking free.
const FRAMING_BITS: u64 = 8 * 8 * BIT_SCALE;

/// Sampling and histogram scratch for one synthesis call. The histogram is
/// open-addressed on the full word, so widths above 16 bits stay exact.
pub const Estimator = struct {
    alloc: Allocator,
    keys: []u32,
    counts: []u32,
    touched: []u32,
    sample: []u32,
    /// Holds the parent words while a merge extracts one field at a time.
    scratch: []u32,
    /// Backing storage for the exact-measurement stream view.
    words: []u8,
    slot_mask: usize,

    pub fn init(alloc: Allocator, sample_elements: usize) !Estimator {
        const limit = std.math.clamp(
            sample_elements,
            MIN_SAMPLE_ELEMENTS,
            MAX_SAMPLE_ELEMENTS,
        );
        const slots = try std.math.ceilPowerOfTwo(usize, limit * 2);

        const keys = try alloc.alloc(u32, slots);
        errdefer alloc.free(keys);
        const counts = try alloc.alloc(u32, slots);
        errdefer alloc.free(counts);
        const touched = try alloc.alloc(u32, limit);
        errdefer alloc.free(touched);
        const sample = try alloc.alloc(u32, limit);
        errdefer alloc.free(sample);
        const scratch = try alloc.alloc(u32, limit);
        errdefer alloc.free(scratch);
        const words = try alloc.alloc(u8, limit * 4);
        errdefer alloc.free(words);

        @memset(counts, 0);
        return .{
            .alloc = alloc,
            .keys = keys,
            .counts = counts,
            .touched = touched,
            .sample = sample,
            .scratch = scratch,
            .words = words,
            .slot_mask = slots - 1,
        };
    }

    pub fn deinit(self: *Estimator, alloc: Allocator) void {
        alloc.free(self.keys);
        alloc.free(self.counts);
        alloc.free(self.touched);
        alloc.free(self.sample);
        alloc.free(self.scratch);
        alloc.free(self.words);
        self.* = undefined;
    }

    /// Q10 bits to generate `stream` as one canonical `Lit`. A stream the
    /// sample covers completely is measured exactly: the sampled model's error
    /// on a short stream exceeds the gap between competing candidates.
    pub fn literalBits(self: *Estimator, stream: Stream) !u64 {
        if (stream.count == 0) return FRAMING_BITS;
        if (stream.count <= self.sample.len) return exactBits(
            try literal_encoding.encodedSize(self.alloc, stream),
        );
        const taken = self.fillSample(stream, 0, stream.count);
        return self.estimateWords(
            self.sample[0..taken],
            stream.count,
            stream.bits_per_elem,
        );
    }

    /// Estimated Q10 bits owed by each child target of `choice`, without
    /// materializing any child stream. Returns the number of children written
    /// to `out`; a terminal production writes none.
    pub fn childBits(
        self: *Estimator,
        choice: grammar.Choice,
        target: Stream,
        dtype: Dtype,
        out: []u64,
    ) !usize {
        const shapes = grammar.childShapesForProposal(choice, target);
        const children = shapes.slice();
        if (children.len == 0 or children.len > out.len) return 0;

        switch (choice) {
            .literal, .constant => return 0,
            .repeat => |times| {
                if (times == 0 or target.count % times != 0) return 0;
                const period = target.count / times;
                out[0] = try self.literalBits(subStream(target, 0, period));
                return 1;
            },
            .concat => |split| {
                if (split == 0 or split >= target.count) return 0;
                out[0] = try self.literalBits(subStream(target, 0, split));
                out[1] = try self.literalBits(subStream(target, split, target.count));
                return 2;
            },
            .map_xor,
            .map_add_mod,
            .map_zigzag,
            .map_gray,
            .map_rotate_left,
            .map_bit_reverse,
            => {
                const operation = choice.mapOperation().?;
                const prepared = semantics.PreparedMap.init(
                    operation,
                    target.bits_per_elem,
                ) catch return 0;
                const taken = self.fillSample(target, 0, target.count);
                for (self.sample[0..taken]) |*word|
                    word.* = prepared.inverseWord(word.*);
                out[0] = try self.estimateWords(
                    self.sample[0..taken],
                    target.count,
                    target.bits_per_elem,
                );
                return 1;
            },
            .scan_xor, .scan_add_mod => {
                if (target.count < 2) return 0;
                const operation = choice.scanOperation().?;
                const prepared = semantics.PreparedScan.init(
                    operation,
                    target.bits_per_elem,
                ) catch return 0;
                // Updates are indexed by the target position they produce, so
                // the sample is drawn from positions one and above.
                const updates = target.count - 1;
                const taken = @min(updates, self.sample.len);
                for (0..taken) |ordinal| {
                    const index = 1 + sampleIndex(updates, taken, ordinal);
                    self.sample[ordinal] = prepared.updateWord(
                        target.getU32(index - 1),
                        target.getU32(index),
                    );
                }
                out[0] = try self.estimateWords(
                    self.sample[0..taken],
                    updates,
                    target.bits_per_elem,
                );
                return 1;
            },
            .merge_fields,
            .merge_float_fields,
            .merge_bit_planes,
            .merge_byte_planes,
            => {
                const taken = self.fillSample(target, 0, target.count);
                if (taken == 0) return 0;
                // Each field extraction overwrites `self.sample`, so keep the
                // parent words aside first.
                const parent = self.scratch;
                @memcpy(parent[0..taken], self.sample[0..taken]);

                var shift: u8 = 0;
                const float = switch (choice) {
                    .merge_float_fields => dtype.floatFields(),
                    else => null,
                };
                for (children, 0..) |child, index| {
                    const child_mask = widthMask(child.bits);
                    const child_shift: u8 = if (float) |fields| switch (index) {
                        0 => fields.total - 1,
                        1 => fields.mant,
                        else => 0,
                    } else shift;
                    for (parent[0..taken], 0..) |word, ordinal|
                        self.sample[ordinal] =
                            (word >> @intCast(child_shift)) & child_mask;
                    out[index] = try self.estimateWords(
                        self.sample[0..taken],
                        child.count,
                        child.bits,
                    );
                    shift += child.bits;
                }
                return children.len;
            },
        }
    }

    /// Materialize `values` as a stream of `bits`-wide words for exact costing.
    fn viewWords(self: *Estimator, values: []const u32, bits: u8) Stream {
        var view = Stream{
            .data = self.words[0 .. values.len * (types.roundUpToPow2(bits) / 8)],
            .count = values.len,
            .bits_per_elem = bits,
            .owns_data = false,
        };
        for (values, 0..) |word, index| view.setU32(index, word);
        return view;
    }

    /// Deterministic strided sample of `stream[start..end)`.
    fn fillSample(
        self: *Estimator,
        stream: Stream,
        start: usize,
        end: usize,
    ) usize {
        const span = end - start;
        const taken = @min(span, self.sample.len);
        for (0..taken) |ordinal|
            self.sample[ordinal] =
                stream.getU32(start + sampleIndex(span, taken, ordinal)) &
                stream.mask();
        return taken;
    }

    /// Charging the entropy table is what keeps a high-cardinality child from
    /// looking free, which otherwise makes every split appear profitable.
    fn estimateWords(
        self: *Estimator,
        values: []const u32,
        total_count: usize,
        bits: u8,
    ) !u64 {
        if (total_count == 0) return FRAMING_BITS;
        const raw_bits: u64 = types.roundUpToPow2(bits);
        const raw_total = scaleBits(total_count, raw_bits * BIT_SCALE);
        if (values.len == 0) return saturatingAdd(FRAMING_BITS, raw_total);
        // The sample is the whole stream, so measure rather than model it.
        if (values.len == total_count)
            return exactBits(try literal_encoding.encodedSize(
                self.alloc,
                self.viewWords(values, bits),
            ));

        var maximum: u32 = 0;
        for (values) |value| maximum = @max(maximum, value);
        const packed_bits: u64 = @max(@as(u64, 1), requiredBits(maximum));
        const packed_total = scaleBits(total_count, packed_bits * BIT_SCALE);

        var best = @min(raw_total, packed_total);

        const distribution = self.measure(values, total_count, raw_bits);
        if (distribution.distinct <= MAX_ENTROPY_ALPHABET) {
            const table_total = scaleBits(
                distribution.distinct,
                (raw_bits + 8) * BIT_SCALE,
            );
            const entropy_total = saturatingAdd(
                scaleBits(total_count, distribution.entropy_q10),
                table_total,
            );
            best = @min(best, entropy_total);
        }
        return saturatingAdd(FRAMING_BITS, best);
    }

    const Distribution = struct {
        /// Zeroth-order entropy in Q10 bits per element.
        entropy_q10: u64,
        /// Distinct words expected across the whole stream.
        distinct: usize,
    };

    /// A sample of a longer stream understates a flat distribution: with every
    /// sampled word distinct the plug-in estimate saturates at log2(|sample|).
    /// Raising it by the unsampled spread in proportion to the distinct ratio
    /// leaves a peaked distribution alone and restores a flat one to raw width.
    fn measure(
        self: *Estimator,
        values: []const u32,
        total_count: usize,
        raw_bits: u64,
    ) Distribution {
        const sampled = values.len;
        var distinct: usize = 0;
        for (values) |value| {
            var slot = hashWord(value) & self.slot_mask;
            while (self.counts[slot] != 0 and self.keys[slot] != value)
                slot = (slot + 1) & self.slot_mask;
            if (self.counts[slot] == 0) {
                self.keys[slot] = value;
                self.touched[distinct] = @intCast(slot);
                distinct += 1;
            }
            self.counts[slot] += 1;
        }

        const log_sampled = log2Q10(sampled);
        var weighted: u128 = 0;
        for (self.touched[0..distinct]) |slot| {
            const count = self.counts[slot];
            weighted += @as(u128, count) *
                @as(u128, log_sampled - log2Q10(count));
            self.counts[slot] = 0;
        }
        const plug_in: u64 = @intCast(weighted / sampled);
        if (sampled >= total_count)
            return .{ .entropy_q10 = plug_in, .distinct = distinct };

        const coverage = (@as(u64, distinct) * BIT_SCALE) / @as(u64, sampled);
        const spread = log2Q10(total_count) - log_sampled;
        const adjusted = saturatingAdd(plug_in, (coverage * spread) / BIT_SCALE);
        const extrapolated = std.math.mul(usize, distinct, total_count) catch
            total_count;
        return .{
            .entropy_q10 = @min(adjusted, raw_bits * BIT_SCALE),
            .distinct = @min(total_count, extrapolated / sampled),
        };
    }
};

/// The literal codec defines its table-driven bodies only up to this many
/// distinct physical words; past it only raw and bit packing remain.
const MAX_ENTROPY_ALPHABET: usize = 65_536;

fn exactBits(bytes: usize) u64 {
    return scaleBits(bytes, 8 * BIT_SCALE);
}

fn subStream(target: Stream, start: usize, end: usize) Stream {
    const width = target.elemBytes();
    return .{
        .data = target.data[start * width .. end * width],
        .count = end - start,
        .bits_per_elem = target.bits_per_elem,
        .owns_data = false,
    };
}

fn widthMask(bits: u8) u32 {
    if (bits >= 32) return std.math.maxInt(u32);
    return (@as(u32, 1) << @intCast(bits)) - 1;
}

fn requiredBits(value: u32) u64 {
    if (value == 0) return 1;
    return 32 - @as(u64, @clz(value));
}

fn sampleIndex(count: usize, sample_count: usize, ordinal: usize) usize {
    if (sample_count <= 1) return 0;
    return @intCast(
        (@as(u128, ordinal) * @as(u128, count - 1)) /
            @as(u128, sample_count - 1),
    );
}

fn hashWord(word: u32) usize {
    return @as(usize, word *% 2_654_435_761) >> 8;
}

fn scaleBits(count: usize, per_element_q10: u64) u64 {
    return std.math.mul(u64, @as(u64, count), per_element_q10) catch
        std.math.maxInt(u64);
}

fn saturatingAdd(left: u64, right: u64) u64 {
    return std.math.add(u64, left, right) catch std.math.maxInt(u64);
}

/// Deterministic integer log2 in Q10. Encoder policy must be host-reproducible.
fn log2Q10(value: usize) u64 {
    std.debug.assert(value >= 1);
    const wide: u64 = @intCast(value);
    const integer_part: u64 = 63 - @as(u64, @clz(wide));
    if (wide == @as(u64, 1) << @intCast(integer_part))
        return integer_part * BIT_SCALE;

    // Normalize the mantissa into [1, 2) as Q32 and extract ten bits by
    // repeated squaring.
    var mantissa: u128 = (@as(u128, wide) << 32) >> @intCast(integer_part);
    const one: u128 = @as(u128, 1) << 32;
    var fraction: u64 = 0;
    for (0..10) |_| {
        mantissa = (mantissa * mantissa) >> 32;
        fraction <<= 1;
        if (mantissa >= one << 1) {
            mantissa >>= 1;
            fraction |= 1;
        }
    }
    return integer_part * BIT_SCALE + fraction;
}

test "log2 is exact on powers of two and monotone between them" {
    try std.testing.expectEqual(@as(u64, 0), log2Q10(1));
    try std.testing.expectEqual(@as(u64, BIT_SCALE), log2Q10(2));
    try std.testing.expectEqual(@as(u64, 3 * BIT_SCALE), log2Q10(8));
    try std.testing.expectEqual(@as(u64, 10 * BIT_SCALE), log2Q10(1024));

    // log2(3) = 1.58496...
    const three = log2Q10(3);
    try std.testing.expect(three > 1622 and three < 1625);

    var previous: u64 = 0;
    for (1..2048) |value| {
        const current = log2Q10(value);
        try std.testing.expect(current >= previous);
        previous = current;
    }
}

test "a constant stream costs far less than a random one" {
    const alloc = std.testing.allocator;
    var estimator = try Estimator.init(alloc, DEFAULT_SAMPLE_ELEMENTS);
    defer estimator.deinit(alloc);

    var uniform = try Stream.init(alloc, 4096, 16);
    defer uniform.deinit(alloc);
    for (0..uniform.count) |index| uniform.setU32(index, 0x1234);

    var noisy = try Stream.init(alloc, 4096, 16);
    defer noisy.deinit(alloc);
    var random = std.Random.DefaultPrng.init(0x5eed);
    for (0..noisy.count) |index|
        noisy.setU32(index, random.random().int(u16));

    const uniform_bits = try estimator.literalBits(uniform);
    const noisy_bits = try estimator.literalBits(noisy);
    try std.testing.expect(uniform_bits < noisy_bits / 8);
    // A full-width random stream must not be estimated below its raw size.
    try std.testing.expect(noisy_bits >= scaleBits(noisy.count, 15 * BIT_SCALE));
}

test "splitting a float target into fields lowers the estimate" {
    const alloc = std.testing.allocator;
    var estimator = try Estimator.init(alloc, DEFAULT_SAMPLE_ELEMENTS);
    defer estimator.deinit(alloc);

    // Realistic BF16 weights: one sign bit, a narrow exponent band, and a
    // mantissa that is close to noise.
    var target = try Stream.init(alloc, 8192, 16);
    defer target.deinit(alloc);
    var random = std.Random.DefaultPrng.init(0xb16);
    for (0..target.count) |index| {
        const sign: u32 = random.random().int(u1);
        const exponent: u32 = 0x7c + random.random().uintLessThan(u32, 3);
        const mantissa: u32 = random.random().int(u7);
        target.setU32(index, (sign << 15) | (exponent << 7) | mantissa);
    }

    const whole = try estimator.literalBits(target);

    var children: [MAX_CHILDREN]u64 = undefined;
    const count = try estimator.childBits(
        .{ .merge_float_fields = .bf16 },
        target,
        .bf16,
        &children,
    );
    try std.testing.expectEqual(@as(usize, 3), count);

    var split: u64 = 0;
    for (children[0..count]) |bits| split += bits;
    try std.testing.expect(split < whole);

    // The exponent field is the one that collapses.
    try std.testing.expect(children[1] < children[2]);
}

test "a scan child is cheaper than its running-sum parent" {
    const alloc = std.testing.allocator;
    var estimator = try Estimator.init(alloc, DEFAULT_SAMPLE_ELEMENTS);
    defer estimator.deinit(alloc);

    var target = try Stream.init(alloc, 4096, 16);
    defer target.deinit(alloc);
    var accumulator: u32 = 0;
    var random = std.Random.DefaultPrng.init(0x5ca7);
    for (0..target.count) |index| {
        accumulator = (accumulator + random.random().uintLessThan(u32, 4)) &
            0xffff;
        target.setU32(index, accumulator);
    }

    const whole = try estimator.literalBits(target);
    var children: [MAX_CHILDREN]u64 = undefined;
    const count = try estimator.childBits(
        .{ .scan_add_mod = target.getU32(0) },
        target,
        .u16,
        &children,
    );
    try std.testing.expectEqual(@as(usize, 1), count);
    try std.testing.expect(children[0] < whole / 2);
}

test "a bijective map pays off only through the packed-width regime" {
    const alloc = std.testing.allocator;
    var estimator = try Estimator.init(alloc, DEFAULT_SAMPLE_ELEMENTS);
    defer estimator.deinit(alloc);

    // A shared high pattern over a wide low field. The alphabet is too large
    // for an entropy table to pay, so minimum-width bit packing decides: the
    // shared high bits force a 16-bit width until the XOR clears them.
    var target = try Stream.init(alloc, 8192, 16);
    defer target.deinit(alloc);
    var prng = std.Random.DefaultPrng.init(0x2162);
    for (0..target.count) |index|
        target.setU32(index, 0xf800 | prng.random().uintLessThan(u32, 2048));

    const whole = try estimator.literalBits(target);
    var children: [MAX_CHILDREN]u64 = undefined;
    const count = try estimator.childBits(
        .{ .map_xor = 0xf800 },
        target,
        .u16,
        &children,
    );
    try std.testing.expectEqual(@as(usize, 1), count);
    try std.testing.expect(children[0] < whole);

    // A bijection cannot change the zeroth-order entropy itself, so a target
    // whose entropy body already wins sees no improvement from any map.
    var narrow = try Stream.init(alloc, 8192, 16);
    defer narrow.deinit(alloc);
    for (0..narrow.count) |index|
        narrow.setU32(index, prng.random().uintLessThan(u32, 4));
    const narrow_whole = try estimator.literalBits(narrow);
    const narrow_count = try estimator.childBits(
        .map_zigzag,
        narrow,
        .i16,
        &children,
    );
    try std.testing.expectEqual(@as(usize, 1), narrow_count);
    try std.testing.expect(children[0] >= narrow_whole);
}
