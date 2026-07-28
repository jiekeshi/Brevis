//! PHOG-inspired contextual prior for the paper DSL grammar.
//!
//! This module is encoder-only policy. It assigns integer description costs
//! to productions already admitted by `grammar`; it never decides legality and
//! it never participates in the final canonical-byte comparison.

const std = @import("std");
const dsl = @import("dsl.zig");
const grammar = @import("grammar.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const Dtype = types.Dtype;
const ProductionId = grammar.ProductionId;
const Stream = types.Stream;

pub const Cost = u32;
pub const COST_SCALE: Cost = 1024;
pub const MAX_DEPTH_BUCKET: u8 = 7;
pub const ROOT_PARENT_WIRE: u16 = 0xffff;

const MAGIC = types.MAGIC;
const KIND = types.Kind.prior;
const VERSION: u16 = 1;
const LEVEL_COUNT: usize = 3;
const MAX_SERIALIZED_ROWS: usize = 262_144;
const SAMPLE_MAX: usize = 256;

/// Stable production ordering. This intentionally does not depend on enum
/// declaration order or on the gaps between the grammar's stable IDs.
pub const PRODUCTIONS = [_]ProductionId{
    .literal,
    .constant,
    .repeat,
    .concat,
    .map_xor,
    .map_add_mod,
    .map_zigzag,
    .map_gray,
    .map_rotate_left,
    .map_bit_reverse,
    .scan_xor,
    .scan_add_mod,
    .merge_fields,
    .merge_float_fields,
    .merge_bit_planes,
    .merge_byte_planes,
};
pub const PRODUCTION_COUNT: usize = PRODUCTIONS.len;

/// The program and tensor context attached to one open synthesis hole.
///
/// `parent == null` is the root sentinel. Continuous target features are
/// deliberately reduced to small deterministic buckets.
pub const Context = struct {
    parent: ?ProductionId,
    child_slot: u8,
    depth_bucket: u8,
    dtype: Dtype,
    target_bits: u8,
    length_bucket: u8,
    zero_bucket: u8,
    distinct_bucket: u8,
    repetition_bucket: u8,
    difference_entropy_bucket: u8,

    pub fn fromTarget(
        target: Stream,
        dtype: Dtype,
        parent: ?ProductionId,
        child_slot: u8,
        depth: usize,
    ) Context {
        var context: Context = .{
            .parent = parent,
            .child_slot = child_slot,
            .depth_bucket = depthBucket(depth),
            .dtype = dtype,
            .target_bits = target.bits_per_elem,
            .length_bucket = lengthBucket(target.count),
            .zero_bucket = 0,
            .distinct_bucket = 0,
            .repetition_bucket = 0,
            .difference_entropy_bucket = 0,
        };

        if (!streamStorageIsReadable(target)) return context;
        const sample_count = @min(target.count, SAMPLE_MAX);
        if (sample_count == 0) return context;

        var distinct_values: [SAMPLE_MAX]u32 = undefined;
        var distinct_count: usize = 0;
        var zero_count: usize = 0;
        var repeated_count: usize = 0;
        for (0..sample_count) |ordinal| {
            const index = sampleIndex(target.count, sample_count, ordinal);
            const value = target.getU32(index) & target.mask();
            zero_count += @intFromBool(value == 0);
            if (ordinal > 0)
                repeated_count += @intFromBool(
                    value == (target.getU32(index - 1) & target.mask()),
                );

            var seen = false;
            for (distinct_values[0..distinct_count]) |known| {
                if (known == value) {
                    seen = true;
                    break;
                }
            }
            if (!seen) {
                distinct_values[distinct_count] = value;
                distinct_count += 1;
            }
        }

        context.zero_bucket = fractionBucket(zero_count, sample_count);
        context.distinct_bucket = fractionBucket(distinct_count, sample_count);
        context.repetition_bucket = if (sample_count < 2)
            0
        else
            fractionBucket(repeated_count, sample_count - 1);
        context.difference_entropy_bucket = differenceEntropyBucket(
            target,
            sample_count,
        );
        return context;
    }
};

pub const Config = struct {
    /// Rational mixing weight λ for the learned distribution.
    learned_numerator: u32 = 1,
    learned_denominator: u32 = 1,
    /// Additive smoothing β. It is integral so stored models and scoring are
    /// reproducible without floating-point count serialization.
    smoothing: u64 = 1,

    pub fn validate(self: Config) error{InvalidConfig}!void {
        if (self.learned_denominator == 0 or
            self.learned_numerator > self.learned_denominator or
            self.smoothing == 0)
            return error.InvalidConfig;
    }
};

const ContextKey = packed struct {
    parent_wire: u16,
    child_slot: u8,
    depth_bucket: u8,
    dtype_wire: u8,
    target_bits: u8,
    length_bucket: u8,
    zero_bucket: u8,
    distinct_bucket: u8,
    repetition_bucket: u8,
    difference_entropy_bucket: u8,
};

const Row = [PRODUCTION_COUNT]u64;
const LevelMap = std.AutoHashMapUnmanaged(ContextKey, Row);

pub const Counts = struct {
    levels: [LEVEL_COUNT]LevelMap = .{ .empty, .empty, .empty },

    pub fn init() Counts {
        return .{};
    }

    pub fn deinit(self: *Counts, alloc: Allocator) void {
        for (&self.levels) |*level| level.deinit(alloc);
        self.* = .{};
    }

    /// Add a weighted observation at all three backoff levels.
    pub fn observe(
        self: *Counts,
        alloc: Allocator,
        context: Context,
        production: ProductionId,
        weight: u64,
    ) (Allocator.Error || error{ InvalidWeight, CountOverflow })!void {
        if (weight == 0) return error.InvalidWeight;
        const index = productionIndex(production);

        // Check arithmetic before mutating any existing row.
        for (0..LEVEL_COUNT) |level| {
            const key = keyFor(context, @intCast(level));
            if (self.levels[level].get(key)) |row| {
                _ = std.math.add(u64, row[index], weight) catch
                    return error.CountOverflow;
            }
        }

        for (0..LEVEL_COUNT) |level| {
            const key = keyFor(context, @intCast(level));
            const entry = try self.levels[level].getOrPut(alloc, key);
            if (!entry.found_existing) entry.value_ptr.* = @splat(0);
            entry.value_ptr[index] += weight;
        }
    }

    /// Recursively observe one exact semantic program. Child targets are
    /// derived with the same target-directed decomposition used by synthesis.
    /// No counts are changed unless the complete tree is validated first.
    pub fn observeProgram(
        self: *Counts,
        alloc: Allocator,
        program: dsl.Program,
        target: Stream,
        dtype: Dtype,
        weight: u64,
    ) !void {
        if (weight == 0) return error.InvalidWeight;
        const program_type = try program.typeOf();
        if (program_type.bits != target.bits_per_elem or
            program_type.len != target.count)
            return error.TargetTypeMismatch;

        var observations: std.ArrayList(Observation) = .empty;
        defer observations.deinit(alloc);
        var nodes: usize = 0;
        try collectObservations(
            alloc,
            &observations,
            program,
            target,
            dtype,
            null,
            0,
            0,
            &nodes,
        );
        for (observations.items) |observation|
            try self.observe(
                alloc,
                observation.context,
                observation.production,
                weight,
            );
    }

    pub fn toPrior(
        self: *const Counts,
        alloc: Allocator,
        config: Config,
    ) !Prior {
        try config.validate();
        var prior: Prior = .{ .config = config };
        errdefer prior.deinit(alloc);
        for (0..LEVEL_COUNT) |level| {
            try prior.levels[level].ensureTotalCapacity(
                alloc,
                self.levels[level].count(),
            );
            var iterator = self.levels[level].iterator();
            while (iterator.next()) |entry| {
                prior.levels[level].putAssumeCapacity(
                    entry.key_ptr.*,
                    entry.value_ptr.*,
                );
                prior.includeMaximumCounts(entry.value_ptr.*);
            }
        }
        return prior;
    }
};

pub const ScoreError = error{
    EmptyLegalSet,
    OutputLengthMismatch,
    DuplicateProduction,
    InvalidAdmittedLowerBound,
    InvalidConfig,
};

pub const Prior = struct {
    config: Config = .{},
    levels: [LEVEL_COUNT]LevelMap = .{ .empty, .empty, .empty },
    /// Cached maxima make Equation-18 lower bounds independent of the number
    /// of tensors searched with this immutable prior.
    maximum_counts: Row = @splat(0),

    pub const empty: Prior = .{};

    pub fn deinit(self: *Prior, alloc: Allocator) void {
        for (&self.levels) |*level| level.deinit(alloc);
        self.* = .{};
    }

    pub fn isEmpty(self: *const Prior) bool {
        for (self.levels) |level|
            if (level.count() != 0) return false;
        return true;
    }

    /// Score exactly the caller-supplied admitted set. An untrained or
    /// unmatched model is strictly uniform. The method neither consults nor
    /// changes grammar legality.
    pub fn scoreSet(
        self: *const Prior,
        context: Context,
        admitted: []const ProductionId,
        output: []Cost,
    ) ScoreError!void {
        try self.config.validate();
        if (admitted.len == 0) return error.EmptyLegalSet;
        if (output.len != admitted.len) return error.OutputLengthMismatch;
        try validateAdmitted(admitted);

        const row = self.findRow(context) orelse {
            @memset(output, uniformCost(admitted.len));
            return;
        };

        var total: u128 = 0;
        for (admitted) |production|
            total += row[productionIndex(production)];
        const n: u128 = admitted.len;
        const beta: u128 = self.config.smoothing;
        const smoothed_total = total + beta * n;
        const learned_numerator: u128 = self.config.learned_numerator;
        const learned_denominator: u128 = self.config.learned_denominator;

        for (admitted, output) |production, *cost| {
            const count: u128 = row[productionIndex(production)];
            const numerator =
                learned_numerator * n * (count + beta) +
                (learned_denominator - learned_numerator) * smoothed_total;
            const denominator =
                learned_denominator * n * smoothed_total;
            cost.* = negativeLog2Ratio(numerator, denominator);
        }
    }

    /// Return null when `production` is not in the admitted set. This makes it
    /// impossible to turn an illegal production into a legal one by scoring.
    pub fn expansionCost(
        self: *const Prior,
        context: Context,
        production: ProductionId,
        admitted: []const ProductionId,
    ) ScoreError!?Cost {
        var selected: ?usize = null;
        for (admitted, 0..) |candidate, index| {
            if (candidate == production) selected = index;
        }
        if (selected == null) {
            try validateAdmitted(admitted);
            if (admitted.len == 0) return error.EmptyLegalSet;
            return null;
        }

        var costs: [PRODUCTION_COUNT]Cost = undefined;
        if (admitted.len > costs.len) return error.DuplicateProduction;
        try self.scoreSet(context, admitted, costs[0..admitted.len]);
        return costs[selected.?];
    }

    /// Compute a context-independent lower bound for every production cost.
    ///
    /// `minimum_admitted` is a proved lower bound on the number of productions
    /// in every concrete admitted set under consideration. For a stored row,
    /// the learned probability of production `r` is at most
    ///
    ///   (max_count(r) + beta) /
    ///   (max_count(r) + beta + (minimum_admitted - 1) * beta).
    ///
    /// This deliberately assumes that every competing production has no
    /// observations and that `r` simultaneously attains its largest count in
    /// any backoff row. Both assumptions can only increase its probability.
    /// The uniform mixture term is likewise maximized at
    /// `minimum_admitted`. Unmatched contexts are uniform and are covered by
    /// the same bound. Applying the deterministic Q10 logarithm therefore
    /// yields a rigorous lower bound on `scoreSet` for every context and every
    /// admitted set of at least this size.
    pub fn contextualCostLowerBounds(
        self: *const Prior,
        minimum_admitted: usize,
        output: []Cost,
    ) ScoreError!void {
        try self.config.validate();
        if (minimum_admitted == 0 or minimum_admitted > PRODUCTION_COUNT)
            return error.InvalidAdmittedLowerBound;
        if (output.len != PRODUCTION_COUNT)
            return error.OutputLengthMismatch;

        const n: u128 = minimum_admitted;
        const beta: u128 = self.config.smoothing;
        const learned_numerator: u128 = self.config.learned_numerator;
        const learned_denominator: u128 = self.config.learned_denominator;
        for (self.maximum_counts, output) |maximum_count, *cost| {
            const preferred = @as(u128, maximum_count) + beta;
            const relaxed_total = preferred + (n - 1) * beta;
            const numerator =
                learned_numerator * n * preferred +
                (learned_denominator - learned_numerator) * relaxed_total;
            const denominator =
                learned_denominator * n * relaxed_total;
            cost.* = negativeLog2Ratio(numerator, denominator);
        }
    }

    /// Canonical, insertion-order-independent model bytes.
    pub fn serialize(self: *const Prior, alloc: Allocator) ![]u8 {
        try self.config.validate();
        var output: std.ArrayList(u8) = .empty;
        errdefer output.deinit(alloc);

        try output.appendSlice(alloc, &MAGIC);
        try output.append(alloc, @intFromEnum(KIND));
        try appendInt(u16, alloc, &output, VERSION);
        try appendInt(u16, alloc, &output, COST_SCALE);
        try appendInt(u32, alloc, &output, self.config.learned_numerator);
        try appendInt(u32, alloc, &output, self.config.learned_denominator);
        try appendInt(u64, alloc, &output, self.config.smoothing);
        try output.append(alloc, LEVEL_COUNT);

        for (0..LEVEL_COUNT) |level| {
            if (self.levels[level].count() > MAX_SERIALIZED_ROWS)
                return error.TooManyRows;
            try output.append(alloc, @intCast(level));
            try appendInt(
                u32,
                alloc,
                &output,
                std.math.cast(u32, self.levels[level].count()) orelse
                    return error.TooManyRows,
            );

            var rows: std.ArrayList(SerializableRow) = .empty;
            defer rows.deinit(alloc);
            try rows.ensureTotalCapacity(alloc, self.levels[level].count());
            var iterator = self.levels[level].iterator();
            while (iterator.next()) |entry|
                rows.appendAssumeCapacity(.{
                    .key = entry.key_ptr.*,
                    .row = entry.value_ptr.*,
                });
            std.sort.heap(
                SerializableRow,
                rows.items,
                @as(u8, @intCast(level)),
                SerializableRow.lessThan,
            );

            for (rows.items) |item| {
                try validateKey(item.key, @intCast(level));
                try writeKey(alloc, &output, item.key, @intCast(level));
                var entry_count: usize = 0;
                for (item.row) |count|
                    entry_count += @intFromBool(count != 0);
                if (entry_count == 0) return error.EmptyRow;
                try output.append(alloc, @intCast(entry_count));
                for (PRODUCTIONS, 0..) |production, index| {
                    const count = item.row[index];
                    if (count == 0) continue;
                    try appendInt(
                        u16,
                        alloc,
                        &output,
                        @intFromEnum(production),
                    );
                    try appendInt(u64, alloc, &output, count);
                }
            }
        }
        return output.toOwnedSlice(alloc);
    }

    pub fn deserialize(alloc: Allocator, bytes: []const u8) !Prior {
        var reader: Reader = .{ .bytes = bytes };
        if (!std.mem.eql(u8, try reader.take(4), &MAGIC))
            return error.BadMagic;
        if (try reader.readByte() != @intFromEnum(KIND)) return error.BadMagic;
        if (try reader.readInt(u16) != VERSION) return error.BadVersion;
        if (try reader.readInt(u16) != COST_SCALE)
            return error.BadCostScale;

        const config: Config = .{
            .learned_numerator = try reader.readInt(u32),
            .learned_denominator = try reader.readInt(u32),
            .smoothing = try reader.readInt(u64),
        };
        try config.validate();
        if (try reader.readByte() != LEVEL_COUNT)
            return error.BadLevelCount;

        var prior: Prior = .{ .config = config };
        errdefer prior.deinit(alloc);
        for (0..LEVEL_COUNT) |level| {
            if (try reader.readByte() != level)
                return error.NonCanonicalLevelOrder;
            const row_count: usize = try reader.readInt(u32);
            if (row_count > MAX_SERIALIZED_ROWS) return error.TooManyRows;
            const minimum_row_bytes = keyWireSize(@intCast(level)) + 1 + 2 + 8;
            if (row_count > reader.remaining().len / minimum_row_bytes)
                return error.Truncated;
            try prior.levels[level].ensureTotalCapacity(
                alloc,
                @intCast(row_count),
            );

            var previous: ?ContextKey = null;
            for (0..row_count) |_| {
                const key = try readKey(&reader, @intCast(level));
                if (previous) |old| {
                    if (keysEqual(old, key, @intCast(level)))
                        return error.DuplicateContext;
                    if (!keyLessThan(@intCast(level), old, key))
                        return error.NonCanonicalContextOrder;
                }
                previous = key;

                const entry_count = try reader.readByte();
                if (entry_count == 0 or entry_count > PRODUCTION_COUNT)
                    return error.InvalidEntryCount;
                var row: Row = @splat(0);
                var previous_id: ?u16 = null;
                for (0..entry_count) |_| {
                    const wire_id = try reader.readInt(u16);
                    const production = std.enums.fromInt(
                        ProductionId,
                        wire_id,
                    ) orelse return error.InvalidProductionId;
                    if (previous_id) |old_id| {
                        if (wire_id == old_id)
                            return error.DuplicateProduction;
                        if (wire_id < old_id)
                            return error.NonCanonicalProductionOrder;
                    }
                    previous_id = wire_id;
                    const count = try reader.readInt(u64);
                    if (count == 0) return error.InvalidCount;
                    row[productionIndex(production)] = count;
                }
                prior.levels[level].putAssumeCapacity(key, row);
                prior.includeMaximumCounts(row);
            }
        }
        if (reader.position != bytes.len) return error.TrailingData;
        return prior;
    }

    fn includeMaximumCounts(self: *Prior, row: Row) void {
        for (row, 0..) |count, index|
            self.maximum_counts[index] = @max(
                self.maximum_counts[index],
                count,
            );
    }

    fn findRow(self: *const Prior, context: Context) ?Row {
        for (0..LEVEL_COUNT) |level| {
            if (self.levels[level].get(keyFor(
                context,
                @intCast(level),
            ))) |row| return row;
        }
        return null;
    }
};

pub fn uniformCost(admitted_count: usize) Cost {
    std.debug.assert(admitted_count > 0);
    std.debug.assert(admitted_count <= PRODUCTION_COUNT);
    return negativeLog2Ratio(1, admitted_count);
}

const Observation = struct {
    context: Context,
    production: ProductionId,
};

const MAX_OBSERVE_DEPTH: usize = 64;
const MAX_OBSERVE_NODES: usize = 1_000_000;

fn collectObservations(
    alloc: Allocator,
    observations: *std.ArrayList(Observation),
    program: dsl.Program,
    target: Stream,
    dtype: Dtype,
    parent: ?ProductionId,
    child_slot: u8,
    depth: usize,
    nodes: *usize,
) !void {
    if (depth > MAX_OBSERVE_DEPTH) return error.ProgramTooDeep;
    nodes.* = std.math.add(usize, nodes.*, 1) catch
        return error.TooManyNodes;
    if (nodes.* > MAX_OBSERVE_NODES) return error.TooManyNodes;

    const choice = try choiceForProgram(program);
    const production = choice.id();
    const context = Context.fromTarget(
        target,
        dtype,
        parent,
        child_slot,
        depth,
    );

    if (choice == .literal) {
        if (program.children.len != 0) return error.InvalidProgramArity;
        const literal = program.kind.literal;
        if (!literal.eql(target)) return error.ProgramTargetMismatch;
    }

    var child_targets = try grammar.childTargets(
        alloc,
        choice,
        target,
        dtype,
    );
    defer child_targets.deinit(alloc);
    if (child_targets.streams.len != program.children.len)
        return error.InvalidProgramArity;

    try observations.append(alloc, .{
        .context = context,
        .production = production,
    });
    for (
        program.children,
        child_targets.streams,
        0..,
    ) |child, child_target, slot| {
        const child_type = try child.typeOf();
        if (child_type.bits != child_target.bits_per_elem or
            child_type.len != child_target.count)
            return error.TargetTypeMismatch;
        if (slot > std.math.maxInt(u8)) return error.TooManyChildren;
        try collectObservations(
            alloc,
            observations,
            child,
            child_target,
            dtype,
            production,
            @intCast(slot),
            depth + 1,
            nodes,
        );
    }
}

fn choiceForProgram(program: dsl.Program) !grammar.Choice {
    return switch (program.kind) {
        .literal => .literal,
        .constant => |constant| .{ .constant = constant.word },
        .concat => blk: {
            if (program.children.len != 2) return error.UnsupportedConcatArity;
            const left_type = try program.children[0].typeOf();
            break :blk .{ .concat = left_type.len };
        },
        .repeat => |times| .{ .repeat = times },
        .map => |operation| switch (operation) {
            .xor => |parameter| .{ .map_xor = parameter },
            .add_mod => |parameter| .{ .map_add_mod = parameter },
            .zigzag => .map_zigzag,
            .gray => .map_gray,
            .rotate_left => |amount| .{ .map_rotate_left = amount },
            .bit_reverse => .map_bit_reverse,
        },
        .scan => |scan| switch (scan.operation) {
            .xor => .{ .scan_xor = scan.initial },
            .add_mod => .{ .scan_add_mod = scan.initial },
        },
        .merge => |operation| switch (operation) {
            .fields => |low_bits| .{ .merge_fields = low_bits },
            .float_fields => |dtype| .{ .merge_float_fields = dtype },
            .bit_planes => .merge_bit_planes,
            .byte_planes => .merge_byte_planes,
        },
    };
}

fn validateAdmitted(admitted: []const ProductionId) ScoreError!void {
    if (admitted.len == 0) return error.EmptyLegalSet;
    if (admitted.len > PRODUCTION_COUNT) return error.DuplicateProduction;
    var seen: [PRODUCTION_COUNT]bool = @splat(false);
    for (admitted) |production| {
        const index = productionIndex(production);
        if (seen[index]) return error.DuplicateProduction;
        seen[index] = true;
    }
}

fn productionIndex(production: ProductionId) usize {
    for (PRODUCTIONS, 0..) |known, index|
        if (known == production) return index;
    unreachable;
}

fn depthBucket(depth: usize) u8 {
    return @intCast(@min(depth, MAX_DEPTH_BUCKET));
}

fn lengthBucket(length: usize) u8 {
    if (length == 0) return 0;
    var value = length;
    var bucket: u8 = 0;
    while (value > 1 and bucket < 63) : (bucket += 1)
        value >>= 1;
    return bucket;
}

fn fractionBucket(numerator: usize, denominator: usize) u8 {
    if (denominator == 0) return 0;
    return @intCast(@min(
        @as(usize, 3),
        (numerator * 4) / denominator,
    ));
}

/// Empirical Shannon entropy of adjacent modular differences, normalized by
/// the maximum entropy for the sampled difference count and quantized to four
/// deterministic buckets. Arithmetic uses the same fixed-point log routine as
/// PHOG costs so context extraction is platform-independent.
fn differenceEntropyBucket(target: Stream, sample_count: usize) u8 {
    if (sample_count < 2) return 0;
    const difference_count = sample_count - 1;
    var values: [SAMPLE_MAX - 1]u32 = undefined;
    var counts: [SAMPLE_MAX - 1]u16 = undefined;
    var distinct: usize = 0;
    for (1..sample_count) |ordinal| {
        const index = sampleIndex(target.count, sample_count, ordinal);
        const value = target.getU32(index) & target.mask();
        const previous = target.getU32(index - 1) & target.mask();
        const difference = (value -% previous) & target.mask();

        var slot: ?usize = null;
        for (values[0..distinct], 0..) |known, candidate| {
            if (known == difference) {
                slot = candidate;
                break;
            }
        }
        if (slot) |existing| {
            counts[existing] += 1;
        } else {
            values[distinct] = difference;
            counts[distinct] = 1;
            distinct += 1;
        }
    }

    if (distinct <= 1) return 0;
    var weighted_entropy: u64 = 0;
    for (counts[0..distinct]) |count| {
        const cost = negativeLog2Ratio(count, difference_count);
        weighted_entropy += @as(u64, count) * cost;
    }
    const entropy_q10 = weighted_entropy / difference_count;
    const maximum_q10 = negativeLog2Ratio(1, difference_count);
    if (maximum_q10 == 0) return 0;
    return @intCast(@min(
        @as(u64, 3),
        (entropy_q10 * 4) / maximum_q10,
    ));
}

fn sampleIndex(count: usize, sample_count: usize, ordinal: usize) usize {
    if (sample_count <= 1) return 0;
    return @intCast(
        (@as(u128, ordinal) * @as(u128, count - 1)) /
            @as(u128, sample_count - 1),
    );
}

fn streamStorageIsReadable(stream: Stream) bool {
    if (stream.bits_per_elem == 0 or stream.bits_per_elem > 32)
        return false;
    const required = std.math.mul(
        usize,
        stream.count,
        stream.elemBytes(),
    ) catch return false;
    return stream.data.len == required;
}

fn keyFor(context: Context, level: u8) ContextKey {
    const parent_wire = if (context.parent) |parent|
        @intFromEnum(parent)
    else
        ROOT_PARENT_WIRE;
    return switch (level) {
        0 => .{
            .parent_wire = parent_wire,
            .child_slot = context.child_slot,
            .depth_bucket = context.depth_bucket,
            .dtype_wire = dtypeWire(context.dtype),
            .target_bits = context.target_bits,
            .length_bucket = context.length_bucket,
            .zero_bucket = context.zero_bucket,
            .distinct_bucket = context.distinct_bucket,
            .repetition_bucket = context.repetition_bucket,
            .difference_entropy_bucket = context.difference_entropy_bucket,
        },
        1 => .{
            .parent_wire = parent_wire,
            .child_slot = context.child_slot,
            .depth_bucket = context.depth_bucket,
            .dtype_wire = dtypeWire(context.dtype),
            .target_bits = context.target_bits,
            .length_bucket = 0,
            .zero_bucket = 0,
            .distinct_bucket = 0,
            .repetition_bucket = 0,
            .difference_entropy_bucket = 0,
        },
        2 => .{
            .parent_wire = parent_wire,
            .child_slot = context.child_slot,
            .depth_bucket = 0,
            .dtype_wire = 0,
            .target_bits = 0,
            .length_bucket = 0,
            .zero_bucket = 0,
            .distinct_bucket = 0,
            .repetition_bucket = 0,
            .difference_entropy_bucket = 0,
        },
        else => unreachable,
    };
}

const SerializableRow = struct {
    key: ContextKey,
    row: Row,

    fn lessThan(level: u8, left: SerializableRow, right: SerializableRow) bool {
        return keyLessThan(level, left.key, right.key);
    }
};

fn keyLessThan(level: u8, left: ContextKey, right: ContextKey) bool {
    if (left.parent_wire != right.parent_wire)
        return left.parent_wire < right.parent_wire;
    if (left.child_slot != right.child_slot)
        return left.child_slot < right.child_slot;
    if (level == 2) return false;
    if (left.depth_bucket != right.depth_bucket)
        return left.depth_bucket < right.depth_bucket;
    if (left.dtype_wire != right.dtype_wire)
        return left.dtype_wire < right.dtype_wire;
    if (left.target_bits != right.target_bits)
        return left.target_bits < right.target_bits;
    if (level == 1) return false;
    if (left.length_bucket != right.length_bucket)
        return left.length_bucket < right.length_bucket;
    if (left.zero_bucket != right.zero_bucket)
        return left.zero_bucket < right.zero_bucket;
    if (left.distinct_bucket != right.distinct_bucket)
        return left.distinct_bucket < right.distinct_bucket;
    if (left.repetition_bucket != right.repetition_bucket)
        return left.repetition_bucket < right.repetition_bucket;
    return left.difference_entropy_bucket <
        right.difference_entropy_bucket;
}

fn keysEqual(left: ContextKey, right: ContextKey, level: u8) bool {
    return !keyLessThan(level, left, right) and
        !keyLessThan(level, right, left);
}

fn validateKey(key: ContextKey, level: u8) !void {
    if (key.parent_wire != ROOT_PARENT_WIRE and
        std.enums.fromInt(ProductionId, key.parent_wire) == null)
        return error.InvalidParentProductionId;
    if (level < 2) {
        if (key.depth_bucket > MAX_DEPTH_BUCKET)
            return error.InvalidDepthBucket;
        _ = dtypeFromWire(key.dtype_wire) orelse return error.InvalidDtype;
        if (key.target_bits == 0 or key.target_bits > 32)
            return error.InvalidTargetWidth;
    }
    if (level == 0 and
        (key.length_bucket > 63 or
            key.zero_bucket > 3 or
            key.distinct_bucket > 3 or
            key.repetition_bucket > 3 or
            key.difference_entropy_bucket > 3))
        return error.InvalidFeatureBucket;
}

fn writeKey(
    alloc: Allocator,
    output: *std.ArrayList(u8),
    key: ContextKey,
    level: u8,
) Allocator.Error!void {
    try appendInt(u16, alloc, output, key.parent_wire);
    try output.append(alloc, key.child_slot);
    if (level == 2) return;
    try output.append(alloc, key.depth_bucket);
    try output.append(alloc, key.dtype_wire);
    try output.append(alloc, key.target_bits);
    if (level == 1) return;
    try output.append(alloc, key.length_bucket);
    try output.append(alloc, key.zero_bucket);
    try output.append(alloc, key.distinct_bucket);
    try output.append(alloc, key.repetition_bucket);
    try output.append(alloc, key.difference_entropy_bucket);
}

fn keyWireSize(level: u8) usize {
    return switch (level) {
        0 => 11,
        1 => 6,
        2 => 3,
        else => unreachable,
    };
}

fn readKey(reader: *Reader, level: u8) !ContextKey {
    var key: ContextKey = .{
        .parent_wire = try reader.readInt(u16),
        .child_slot = try reader.readByte(),
        .depth_bucket = 0,
        .dtype_wire = 0,
        .target_bits = 0,
        .length_bucket = 0,
        .zero_bucket = 0,
        .distinct_bucket = 0,
        .repetition_bucket = 0,
        .difference_entropy_bucket = 0,
    };
    if (level < 2) {
        key.depth_bucket = try reader.readByte();
        key.dtype_wire = try reader.readByte();
        key.target_bits = try reader.readByte();
    }
    if (level == 0) {
        key.length_bucket = try reader.readByte();
        key.zero_bucket = try reader.readByte();
        key.distinct_bucket = try reader.readByte();
        key.repetition_bucket = try reader.readByte();
        key.difference_entropy_bucket = try reader.readByte();
    }
    try validateKey(key, level);
    return key;
}

fn appendInt(
    comptime T: type,
    alloc: Allocator,
    output: *std.ArrayList(u8),
    value: T,
) Allocator.Error!void {
    var bytes: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &bytes, value, .little);
    try output.appendSlice(alloc, &bytes);
}

const Reader = struct {
    bytes: []const u8,
    position: usize = 0,

    fn take(self: *Reader, count: usize) error{Truncated}![]const u8 {
        if (count > self.bytes.len - self.position) return error.Truncated;
        defer self.position += count;
        return self.bytes[self.position..][0..count];
    }

    fn readByte(self: *Reader) error{Truncated}!u8 {
        return (try self.take(1))[0];
    }

    fn readInt(self: *Reader, comptime T: type) error{Truncated}!T {
        return std.mem.readInt(T, (try self.take(@sizeOf(T)))[0..@sizeOf(T)], .little);
    }

    fn remaining(self: *const Reader) []const u8 {
        return self.bytes[self.position..];
    }
};

/// Deterministically round `-log2(numerator / denominator)` to 1/1024 bit.
/// All probability arithmetic and logarithm extraction are integer-only.
fn negativeLog2Ratio(numerator: u128, denominator: u128) Cost {
    std.debug.assert(numerator > 0);
    std.debug.assert(numerator <= denominator);

    const Wide = u512;
    const fractional_bits = 10;
    const working_bits = 192;
    const one: Wide = @as(Wide, 1) << working_bits;
    const two: Wide = one << 1;
    const wide_denominator: Wide = denominator;

    var scaled_numerator: Wide = numerator;
    var integer_part: Cost = 0;
    while ((scaled_numerator << 1) <= wide_denominator) {
        scaled_numerator <<= 1;
        integer_part += 1;
    }

    // x is the normalized ratio in [1, 2), represented as Q192.
    var x: Wide =
        (wide_denominator << working_bits) / scaled_numerator;
    var fraction_with_guard: Cost = 0;
    for (0..fractional_bits + 1) |_| {
        x = (x * x) >> working_bits;
        fraction_with_guard <<= 1;
        if (x >= two) {
            x >>= 1;
            fraction_with_guard |= 1;
        }
    }
    const rounded_fraction =
        (fraction_with_guard >> 1) + (fraction_with_guard & 1);
    return integer_part * COST_SCALE + rounded_fraction;
}

fn dtypeWire(dtype: Dtype) u8 {
    return switch (dtype) {
        .f16 => 0,
        .bf16 => 1,
        .f32 => 2,
        .u8 => 3,
        .u16 => 4,
        .u32 => 5,
        .i8 => 6,
        .i16 => 7,
        .i32 => 8,
        .f8_e4m3 => 9,
        .f8_e5m2 => 10,
    };
}

fn dtypeFromWire(wire: u8) ?Dtype {
    return switch (wire) {
        0 => .f16,
        1 => .bf16,
        2 => .f32,
        3 => .u8,
        4 => .u16,
        5 => .u32,
        6 => .i8,
        7 => .i16,
        8 => .i32,
        9 => .f8_e4m3,
        10 => .f8_e5m2,
        else => null,
    };
}
