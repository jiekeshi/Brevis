//! Finite, deterministic production registry for the paper's semantic DSL.
//!
//! This module is target-directed: `propose` returns concrete productions that
//! can generate `target`, and `childTargets` derives the exact targets for their
//! holes. It deliberately contains no queue, score, or search policy.

const std = @import("std");
const decomposition = @import("decomposition.zig");
const dsl = @import("dsl.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const Stream = types.Stream;
const Dtype = types.Dtype;

/// Stable semantic production identifiers. Values are wire-safe identities;
/// gaps reserve room for future productions without renumbering existing ones.
pub const ProductionId = enum(u16) {
    literal = 0,
    constant = 1,
    repeat = 2,
    concat = 3,

    map_xor = 10,
    map_add_mod = 11,
    map_zigzag = 12,
    map_gray = 13,
    map_rotate_left = 14,
    map_bit_reverse = 15,

    scan_xor = 20,
    scan_add_mod = 21,

    merge_fields = 30,
    merge_float_fields = 31,
    merge_bit_planes = 32,
    merge_byte_planes = 33,
};

/// A concrete production and its bounded parameter proposal.
pub const Choice = union(ProductionId) {
    literal,
    constant: u32,
    /// Number of repetitions; `childTargets` returns the minimal period.
    repeat: u32,
    /// Element offset of the right child in a binary Concat.
    concat: usize,

    map_xor: u32,
    map_add_mod: u32,
    map_zigzag,
    map_gray,
    map_rotate_left: u8,
    map_bit_reverse,

    /// The scan's explicit initial word.
    scan_xor: u32,
    scan_add_mod: u32,

    /// Width of the least-significant child field.
    merge_fields: u8,
    merge_float_fields: Dtype,
    merge_bit_planes,
    merge_byte_planes,

    pub fn id(self: Choice) ProductionId {
        return std.meta.activeTag(self);
    }

    pub fn mapOperation(self: Choice) ?dsl.MapOp {
        return switch (self) {
            .map_xor => |parameter| .{ .xor = parameter },
            .map_add_mod => |parameter| .{ .add_mod = parameter },
            .map_zigzag => .zigzag,
            .map_gray => .gray,
            .map_rotate_left => |amount| .{ .rotate_left = amount },
            .map_bit_reverse => .bit_reverse,
            else => null,
        };
    }

    pub fn scanOperation(self: Choice) ?dsl.ScanOp {
        return switch (self) {
            .scan_xor => .xor,
            .scan_add_mod => .add_mod,
            else => null,
        };
    }

    pub fn mergeOperation(self: Choice) ?dsl.MergeOp {
        return switch (self) {
            .merge_fields => |low_bits| .{ .fields = low_bits },
            .merge_float_fields => |dtype| .{ .float_fields = dtype },
            .merge_bit_planes => .bit_planes,
            .merge_byte_planes => .byte_planes,
            else => null,
        };
    }
};

/// Hard caps make the grammar finite even when callers pass untrusted options.
pub const HARD_MAX_REPEAT_PERIOD: usize = 64;
pub const HARD_MAX_CONCAT_SPLITS: usize = 16;
pub const HARD_MAX_MAP_CONSTANTS: usize = 8;
pub const HARD_MAX_ROTATIONS: usize = 8;
pub const HARD_MAX_FIELD_SPLITS: usize = 8;
pub const MAX_PROPOSALS: usize = 59;

pub const Options = struct {
    max_depth: u8 = 4,
    /// Only a minimal repeat period no longer than this is proposed.
    max_repeat_period: usize = 32,
    /// Evenly spaced binary Concat split points.
    max_concat_splits: usize = 3,
    /// Representative target words used for XOR/add parameters.
    max_map_constants: usize = 2,
    /// Evenly spaced rotations in `[1, bits)`.
    max_rotations: usize = 3,
    /// Evenly spaced low-field widths in `[1, bits)`.
    max_field_splits: usize = 3,
};

pub const LegalProductions = struct {
    storage: [16]ProductionId = undefined,
    len: usize = 0,

    pub fn slice(self: *const LegalProductions) []const ProductionId {
        return self.storage[0..self.len];
    }
};

/// Enumerate legal production families in stable ProductionId order. The result
/// is fixed-capacity and allocation-free; valid targets always include Literal.
pub fn legal(
    target: Stream,
    dtype: Dtype,
    hole_depth: u8,
    options: Options,
) LegalProductions {
    var result: LegalProductions = .{};
    if (!targetIsValid(target)) return result;
    for (std.enums.values(ProductionId)) |production| {
        if (!productionIsLegal(production, target, dtype, hole_depth, options))
            continue;
        result.storage[result.len] = production;
        result.len += 1;
    }
    return result;
}

/// Whether one production family has a concrete target-directed proposal.
pub fn isLegal(
    production: ProductionId,
    target: Stream,
    dtype: Dtype,
    hole_depth: u8,
    options: Options,
) bool {
    if (!targetIsValid(target)) return false;
    return productionIsLegal(production, target, dtype, hole_depth, options);
}

fn productionIsLegal(
    production: ProductionId,
    target: Stream,
    dtype: Dtype,
    hole_depth: u8,
    options: Options,
) bool {
    if (target.count == 0) return production == .literal;
    return switch (production) {
        .literal => true,
        .constant => uniformWord(target) != null,
        else => if (hole_depth >= options.max_depth)
            false
        else switch (production) {
            .repeat => minimalRepeatTimes(target, options.max_repeat_period) != null,
            .concat => target.count >= 2 and bounded(
                options.max_concat_splits,
                HARD_MAX_CONCAT_SPLITS,
            ) > 0,
            .map_xor, .map_add_mod => bounded(
                options.max_map_constants,
                HARD_MAX_MAP_CONSTANTS,
            ) > 0,
            .map_zigzag, .map_gray, .map_bit_reverse => true,
            .map_rotate_left => bounded(options.max_rotations, HARD_MAX_ROTATIONS) > 0,
            .scan_xor, .scan_add_mod => target.count >= 2,
            .merge_fields => target.bits_per_elem > 1 and bounded(
                options.max_field_splits,
                HARD_MAX_FIELD_SPLITS,
            ) > 0,
            .merge_float_fields => if (dtype.floatFields()) |fields|
                fields.total == target.bits_per_elem
            else
                false,
            .merge_bit_planes => target.bits_per_elem > 1,
            .merge_byte_planes => target.bits_per_elem > 8,
            .literal, .constant => unreachable,
        },
    };
}

/// Return all bounded proposals in stable ProductionId order and deterministic
/// parameter order. For a valid target, element zero is always Literal.
pub fn propose(
    alloc: Allocator,
    target: Stream,
    dtype: Dtype,
    hole_depth: u8,
    options: Options,
) (Allocator.Error || dsl.ValidationError)![]Choice {
    try validateTarget(target);

    var choices: std.ArrayList(Choice) = .empty;
    errdefer choices.deinit(alloc);
    try choices.ensureTotalCapacity(alloc, MAX_PROPOSALS);
    try choices.append(alloc, .literal);
    if (target.count == 0)
        return choices.toOwnedSlice(alloc);

    if (uniformWord(target)) |word|
        try choices.append(alloc, .{ .constant = word });

    if (hole_depth >= options.max_depth)
        return choices.toOwnedSlice(alloc);

    if (minimalRepeatTimes(target, options.max_repeat_period)) |times|
        try choices.append(alloc, .{ .repeat = times });

    try appendConcatChoices(alloc, &choices, target.count, options.max_concat_splits);

    var representative: [HARD_MAX_MAP_CONSTANTS]u32 = undefined;
    const representative_count = representativeWords(
        target,
        options.max_map_constants,
        &representative,
    );
    for (representative[0..representative_count]) |word|
        try choices.append(alloc, .{ .map_xor = word });

    var add_parameters: [HARD_MAX_MAP_CONSTANTS]u32 = undefined;
    var add_count: usize = 0;
    const mask = target.mask();
    for (representative[0..representative_count]) |word| {
        const parameter = (0 -% word) & mask;
        if (!contains(u32, add_parameters[0..add_count], parameter)) {
            add_parameters[add_count] = parameter;
            add_count += 1;
            try choices.append(alloc, .{ .map_add_mod = parameter });
        }
    }

    try choices.append(alloc, .map_zigzag);
    try choices.append(alloc, .map_gray);
    try appendRotationChoices(
        alloc,
        &choices,
        target.bits_per_elem,
        options.max_rotations,
    );
    try choices.append(alloc, .map_bit_reverse);

    if (target.count >= 2) {
        try choices.append(alloc, .{ .scan_xor = target.getU32(0) });
        try choices.append(alloc, .{ .scan_add_mod = target.getU32(0) });
    }

    try appendFieldChoices(
        alloc,
        &choices,
        target.bits_per_elem,
        options.max_field_splits,
    );
    if (dtype.floatFields()) |fields| {
        if (fields.total == target.bits_per_elem)
            try choices.append(alloc, .{ .merge_float_fields = dtype });
    }
    if (target.bits_per_elem > 1)
        try choices.append(alloc, .merge_bit_planes);
    if (target.bits_per_elem > 8)
        try choices.append(alloc, .merge_byte_planes);

    std.debug.assert(choices.items.len <= MAX_PROPOSALS);
    return choices.toOwnedSlice(alloc);
}

pub const ChildTargets = struct {
    streams: []Stream,

    pub fn deinit(self: *ChildTargets, alloc: Allocator) void {
        for (self.streams) |*stream| stream.deinit(alloc);
        alloc.free(self.streams);
        self.streams = &.{};
    }
};

pub const ChildTargetError = Allocator.Error || dsl.ValidationError || error{
    InvalidChoice,
};

/// Exact storage occupied by the child target streams of `choice`, without
/// allocating them. Search uses this as a structural memory bound; it does not
/// alter production legality or the PHOG distribution.
pub fn childTargetStorageBytes(
    choice: Choice,
    target: Stream,
    dtype: Dtype,
) (dsl.ValidationError || error{ InvalidChoice, IntegerOverflow })!usize {
    try validateTarget(target);
    const elem_bytes = target.elemBytes();
    return switch (choice) {
        .literal => 0,
        .constant => |word| if (uniformWord(target) == word)
            0
        else
            error.InvalidChoice,
        .repeat => |times| blk: {
            if (times < 2 or target.count == 0 or target.count % times != 0)
                return error.InvalidChoice;
            const period = target.count / times;
            for (period..target.count) |index|
                if (target.getU32(index) != target.getU32(index % period))
                    return error.InvalidChoice;
            break :blk std.math.mul(usize, period, elem_bytes) catch
                return error.IntegerOverflow;
        },
        .concat => |split| if (split == 0 or split >= target.count)
            error.InvalidChoice
        else
            target.data.len,
        .map_xor,
        .map_add_mod,
        .map_zigzag,
        .map_gray,
        .map_rotate_left,
        .map_bit_reverse,
        => blk: {
            try choice.mapOperation().?.validate(target.bits_per_elem);
            break :blk target.data.len;
        },
        .scan_xor, .scan_add_mod => |initial| blk: {
            if (target.count < 2 or initial != target.getU32(0))
                return error.InvalidChoice;
            break :blk std.math.mul(
                usize,
                target.count - 1,
                elem_bytes,
            ) catch return error.IntegerOverflow;
        },
        .merge_float_fields => |choice_dtype| if (choice_dtype != dtype)
            error.InvalidChoice
        else
            mergeTargetStorageBytes(target, choice.mergeOperation().?),
        .merge_fields, .merge_bit_planes, .merge_byte_planes => mergeTargetStorageBytes(
            target,
            choice.mergeOperation().?,
        ),
    };
}

/// Derive the exact child targets for a concrete production. An empty result
/// means the choice is a terminal (`Lit` or `Const`).
pub fn childTargets(
    alloc: Allocator,
    choice: Choice,
    target: Stream,
    dtype: Dtype,
) ChildTargetError!ChildTargets {
    try validateTarget(target);

    return switch (choice) {
        .literal => emptyTargets(alloc),
        .constant => |word| blk: {
            if (uniformWord(target) != word) return error.InvalidChoice;
            break :blk emptyTargets(alloc);
        },
        .repeat => |times| blk: {
            const period = (try decomposition.repeat(alloc, target, times)) orelse
                return error.InvalidChoice;
            break :blk oneTarget(alloc, period);
        },
        .concat => |split| concatTargets(alloc, target, split),
        .map_xor,
        .map_add_mod,
        .map_zigzag,
        .map_gray,
        .map_rotate_left,
        .map_bit_reverse,
        => blk: {
            const operation = choice.mapOperation().?;
            break :blk oneTarget(
                alloc,
                try decomposition.map(alloc, target, operation),
            );
        },
        .scan_xor, .scan_add_mod => |initial| blk: {
            var parts = (try decomposition.scan(
                alloc,
                target,
                choice.scanOperation().?,
            )) orelse return error.InvalidChoice;
            if (parts.initial != initial) {
                parts.deinit(alloc);
                return error.InvalidChoice;
            }
            break :blk oneTarget(alloc, parts.updates);
        },
        .merge_float_fields => |choice_dtype| blk: {
            if (choice_dtype != dtype) return error.InvalidChoice;
            break :blk mergeTargets(alloc, target, choice.mergeOperation().?);
        },
        .merge_fields, .merge_bit_planes, .merge_byte_planes => mergeTargets(
            alloc,
            target,
            choice.mergeOperation().?,
        ),
    };
}

fn emptyTargets(alloc: Allocator) Allocator.Error!ChildTargets {
    return .{ .streams = try alloc.alloc(Stream, 0) };
}

fn oneTarget(alloc: Allocator, stream: Stream) Allocator.Error!ChildTargets {
    const streams = alloc.alloc(Stream, 1) catch |err| {
        var owned = stream;
        owned.deinit(alloc);
        return err;
    };
    streams[0] = stream;
    return .{ .streams = streams };
}

fn concatTargets(
    alloc: Allocator,
    target: Stream,
    split: usize,
) ChildTargetError!ChildTargets {
    if (split == 0 or split >= target.count) return error.InvalidChoice;
    const streams = try alloc.alloc(Stream, 2);
    var initialized: usize = 0;
    errdefer {
        for (streams[0..initialized]) |*stream| stream.deinit(alloc);
        alloc.free(streams);
    }
    streams[0] = try copyRange(alloc, target, 0, split);
    initialized += 1;
    streams[1] = try copyRange(alloc, target, split, target.count);
    initialized += 1;
    return .{ .streams = streams };
}

fn mergeTargets(
    alloc: Allocator,
    target: Stream,
    operation: dsl.MergeOp,
) ChildTargetError!ChildTargets {
    return .{ .streams = try decomposition.merge(alloc, target, operation) };
}

fn mergeTargetStorageBytes(
    target: Stream,
    operation: dsl.MergeOp,
) (dsl.ValidationError || error{IntegerOverflow})!usize {
    var widths: [32]u8 = undefined;
    const width_count: usize = switch (operation) {
        .fields => |low_bits| blk: {
            if (low_bits == 0 or low_bits >= target.bits_per_elem)
                return error.InvalidParameter;
            widths[0] = low_bits;
            widths[1] = target.bits_per_elem - low_bits;
            break :blk 2;
        },
        .float_fields => |dtype| blk: {
            const fields = dtype.floatFields() orelse
                return error.InvalidParameter;
            if (fields.total != target.bits_per_elem)
                return error.TypeMismatch;
            widths[0] = 1;
            widths[1] = fields.exp;
            widths[2] = fields.mant;
            break :blk 3;
        },
        .bit_planes => blk: {
            for (0..target.bits_per_elem) |index| widths[index] = 1;
            break :blk target.bits_per_elem;
        },
        .byte_planes => blk: {
            const count = (@as(usize, target.bits_per_elem) + 7) / 8;
            for (0..count - 1) |index| widths[index] = 8;
            widths[count - 1] = target.bits_per_elem -
                @as(u8, @intCast((count - 1) * 8));
            break :blk count;
        },
    };

    var total: usize = 0;
    for (widths[0..width_count]) |bits| {
        const child_bytes = std.math.mul(
            usize,
            target.count,
            types.roundUpToPow2(bits) / 8,
        ) catch return error.IntegerOverflow;
        total = std.math.add(usize, total, child_bytes) catch
            return error.IntegerOverflow;
    }
    return total;
}

fn copyRange(
    alloc: Allocator,
    target: Stream,
    start: usize,
    end: usize,
) (Allocator.Error || dsl.ValidationError)!Stream {
    var output = try Stream.init(alloc, end - start, target.bits_per_elem);
    for (start..end, 0..) |source, destination|
        output.setU32(destination, target.getU32(source));
    return output;
}

fn appendConcatChoices(
    alloc: Allocator,
    choices: *std.ArrayList(Choice),
    count: usize,
    requested: usize,
) Allocator.Error!void {
    if (count < 2) return;
    const n = @min(
        bounded(requested, HARD_MAX_CONCAT_SPLITS),
        count - 1,
    );
    for (1..n + 1) |index| {
        const split = evenInteriorPoint(count - 1, index, n);
        try choices.append(alloc, .{ .concat = split });
    }
}

fn appendRotationChoices(
    alloc: Allocator,
    choices: *std.ArrayList(Choice),
    bits: u8,
    requested: usize,
) Allocator.Error!void {
    const possible: usize = if (bits == 1) 1 else bits - 1;
    const n = @min(bounded(requested, HARD_MAX_ROTATIONS), possible);
    for (1..n + 1) |index| {
        const amount: u8 = if (bits == 1)
            0
        else
            @intCast(evenInteriorPoint(bits - 1, index, n));
        try choices.append(alloc, .{ .map_rotate_left = amount });
    }
}

fn appendFieldChoices(
    alloc: Allocator,
    choices: *std.ArrayList(Choice),
    bits: u8,
    requested: usize,
) Allocator.Error!void {
    if (bits < 2) return;
    const possible: usize = bits - 1;
    const n = @min(bounded(requested, HARD_MAX_FIELD_SPLITS), possible);
    for (1..n + 1) |index| {
        const low_bits: u8 = @intCast(evenInteriorPoint(possible, index, n));
        try choices.append(alloc, .{ .merge_fields = low_bits });
    }
}

/// `index` is 1-based and the result is a distinct point in `1...maximum`.
fn evenInteriorPoint(maximum: usize, index: usize, count: usize) usize {
    const numerator = @as(u128, index) * (@as(u128, maximum) + 1);
    return @intCast(numerator / (@as(u128, count) + 1));
}

fn representativeWords(
    target: Stream,
    requested: usize,
    output: *[HARD_MAX_MAP_CONSTANTS]u32,
) usize {
    const n = @min(
        @min(bounded(requested, HARD_MAX_MAP_CONSTANTS), target.count),
        output.len,
    );
    var written: usize = 0;
    for (0..n) |index| {
        const target_index: usize = if (n == 1)
            0
        else
            @intCast(
                (@as(u128, index) * @as(u128, target.count - 1)) /
                    @as(u128, n - 1),
            );
        const word = target.getU32(target_index);
        if (!contains(u32, output[0..written], word)) {
            output[written] = word;
            written += 1;
        }
    }
    return written;
}

fn minimalRepeatTimes(target: Stream, requested_period: usize) ?u32 {
    if (target.count < 2) return null;
    const limit = @min(
        @min(bounded(requested_period, HARD_MAX_REPEAT_PERIOD), target.count / 2),
        target.count,
    );
    for (1..limit + 1) |period| {
        if (target.count % period != 0) continue;
        var exact = true;
        for (period..target.count) |index| {
            if (target.getU32(index) != target.getU32(index % period)) {
                exact = false;
                break;
            }
        }
        if (!exact) continue;
        return std.math.cast(u32, target.count / period);
    }
    return null;
}

fn uniformWord(target: Stream) ?u32 {
    if (target.count == 0) return null;
    const word = target.getU32(0);
    for (1..target.count) |index|
        if (target.getU32(index) != word) return null;
    return word;
}

fn bounded(requested: usize, hard_max: usize) usize {
    return @min(requested, hard_max);
}

fn contains(comptime T: type, values: []const T, needle: T) bool {
    for (values) |value| if (value == needle) return true;
    return false;
}

fn targetIsValid(target: Stream) bool {
    validateTarget(target) catch return false;
    return true;
}

fn validateTarget(target: Stream) dsl.ValidationError!void {
    if (target.bits_per_elem == 0 or target.bits_per_elem > 32)
        return error.InvalidWordWidth;
    const required = std.math.mul(
        usize,
        target.count,
        target.elemBytes(),
    ) catch return error.LengthOverflow;
    if (target.data.len != required) return error.InvalidLiteralValue;
    const mask = target.mask();
    for (0..target.count) |index|
        if (target.getU32(index) & ~mask != 0)
            return error.InvalidLiteralValue;
}
