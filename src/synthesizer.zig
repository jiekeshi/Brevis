//! Budgeted A* synthesis over target-directed semantic programs.
//!
//! Search order is determined by grammar description length. Complete
//! candidates are compared only by their exact canonical serialized bytes.

const std = @import("std");
const dsl = @import("dsl.zig");
const grammar = @import("grammar.zig");
const grammar_prior = @import("grammar_prior.zig");
const interpreter = @import("interpreter.zig");
const program_format = @import("program_format.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const Dtype = types.Dtype;
const Stream = types.Stream;

pub const Options = struct {
    max_expansions: usize = 512,
    max_nodes: usize = 64,
    /// Maximum total storage of simultaneously open target streams. Choices
    /// that would exceed it are pruned before decomposition allocation.
    max_decomposition_bytes: usize = 512 * 1024 * 1024,
    grammar_options: grammar.Options = .{},
    /// Encoder-only PHOG policy. It changes queue order but never legality,
    /// canonical byte cost, or complete-candidate selection.
    rule_model: ?*const grammar_prior.Prior = null,
};

pub const SearchStatus = enum {
    /// The finite frontier induced by the configured grammar and structural
    /// limits was exhausted or safely pruned.
    proven_optimal,
    /// Search stopped because the partial-program expansion budget was spent.
    budget_exhausted,
};

pub const Result = struct {
    program: dsl.Program,
    serialized_bytes: usize,
    expanded: usize,
    completed_candidates: usize,
    status: SearchStatus,
    used_literal_fallback: bool,

    pub fn deinit(self: *Result, alloc: Allocator) void {
        self.program.deinit(alloc);
    }
};

const Partial = struct {
    choices: []grammar.Choice,
    holes: usize,
    /// g(s): rule cost already paid by filled productions.
    grammar_cost: u64,
    /// f(s) = g(s) + h(s), where h is an admissible completion bound.
    priority_cost: u64,
    /// Exact encoded bytes of the file prefix and every selected node's fixed
    /// instruction/parameter fields, plus literal framing lower bounds.
    fixed_bytes: usize,
    size_lower_bound: usize,
    serial: u64,

    fn deinit(self: *Partial, alloc: Allocator) void {
        alloc.free(self.choices);
        self.choices = &.{};
    }
};

fn comparePartial(_: void, left: Partial, right: Partial) std.math.Order {
    if (left.priority_cost != right.priority_cost)
        return std.math.order(left.priority_cost, right.priority_cost);
    if (left.size_lower_bound != right.size_lower_bound)
        return std.math.order(left.size_lower_bound, right.size_lower_bound);
    return std.math.order(left.serial, right.serial);
}

const Queue = std.PriorityQueue(Partial, void, comparePartial);

const Hole = struct {
    target: Stream,
    depth: u8,
    parent: ?grammar.ProductionId,
    child_slot: u8,
};

const OpenHoles = struct {
    items: std.ArrayList(Hole) = .empty,

    fn deinit(self: *OpenHoles, alloc: Allocator) void {
        for (self.items.items) |*hole| hole.target.deinit(alloc);
        self.items.deinit(alloc);
    }

    fn leftmost(self: *OpenHoles) *Hole {
        return &self.items.items[self.items.items.len - 1];
    }
};

/// Synthesize the smallest exact program encountered within `options`.
/// `target` is borrowed for the duration of the call.
pub fn synthesize(
    alloc: Allocator,
    target: Stream,
    dtype: Dtype,
    options: Options,
) !Result {
    if (target.bits_per_elem != dtype.bitWidth())
        return error.TensorWidthMismatch;

    // The universal fallback is a complete semantic program, not an external
    // raw block mode.
    var best = try dsl.Program.literalFromStream(alloc, target);
    errdefer best.deinit(alloc);
    var best_bytes = try program_format.serializedSize(alloc, best);
    var used_literal_fallback = true;
    var completed_candidates: usize = 0;
    var expanded: usize = 0;

    var queue = Queue.initContext({});
    defer {
        while (queue.pop()) |partial_value| {
            var partial = partial_value;
            partial.deinit(alloc);
        }
        queue.deinit(alloc);
    }

    var serial: u64 = 0;
    const initial_choices = try alloc.alloc(grammar.Choice, 0);
    var relaxed_heuristic = try RelaxedGrammarHeuristic.init(
        alloc,
        dtype,
        options,
    );
    defer relaxed_heuristic.deinit(alloc);
    const initial_heuristic = relaxed_heuristic.completionCost(
        target.bits_per_elem,
        target.count,
        0,
    );
    try queue.push(alloc, .{
        .choices = initial_choices,
        .holes = 1,
        .grammar_cost = 0,
        .priority_cost = initial_heuristic,
        .fixed_bytes = program_format.MAGIC.len + 1,
        .size_lower_bound = program_format.MAGIC.len + 1 + 1,
        .serial = serial,
    });

    while (queue.count() > 0 and expanded < options.max_expansions) {
        var partial = queue.pop().?;
        defer partial.deinit(alloc);

        if (partial.size_lower_bound >= best_bytes) continue;

        if (partial.holes == 0) {
            var candidate = try buildProgram(
                alloc,
                partial.choices,
                target,
                dtype,
            );
            var candidate_owned = true;
            defer if (candidate_owned) candidate.deinit(alloc);

            // Complete states are independently checked. The decomposition
            // contract makes failures exceptional, but correctness does not
            // rely on trusting proposal code.
            var output = try interpreter.execute(alloc, candidate);
            defer output.deinit(alloc);
            if (!streamsEqual(target, output)) return error.InvalidCandidate;

            completed_candidates += 1;
            const candidate_bytes = try program_format.serializedSize(
                alloc,
                candidate,
            );
            if (candidate_bytes < best_bytes) {
                best.deinit(alloc);
                best = candidate;
                candidate_owned = false;
                best_bytes = candidate_bytes;
                used_literal_fallback = switch (best.kind) {
                    .literal => true,
                    else => false,
                };
            }
            continue;
        }

        if (partial.choices.len + partial.holes > options.max_nodes)
            continue;

        var open_holes = try replayOpenHoles(
            alloc,
            target,
            dtype,
            partial.choices,
            options.max_decomposition_bytes,
        );
        defer open_holes.deinit(alloc);
        if (open_holes.items.items.len != partial.holes)
            return error.InvalidPartialProgram;
        const hole = open_holes.leftmost().*;

        expanded += 1;
        const choices = try grammar.propose(
            alloc,
            hole.target,
            dtype,
            hole.depth,
            options.grammar_options,
        );
        defer alloc.free(choices);
        const legal = grammar.legal(
            hole.target,
            dtype,
            hole.depth,
            options.grammar_options,
        );
        if (legal.len == 0) return error.NoLiteralFallback;
        const admitted = legal.slice();
        var production_costs: [grammar_prior.PRODUCTION_COUNT]grammar_prior.Cost = undefined;
        const context = grammar_prior.Context.fromTarget(
            hole.target,
            dtype,
            hole.parent,
            hole.child_slot,
            hole.depth,
        );
        if (options.rule_model) |rule_model| {
            try rule_model.scoreSet(
                context,
                admitted,
                production_costs[0..admitted.len],
            );
        } else {
            @memset(
                production_costs[0..admitted.len],
                grammar_prior.uniformCost(admitted.len),
            );
        }

        const current_open_storage = try openTargetStorageBytes(&open_holes);
        var retained_heuristic: u64 = 0;
        for (open_holes.items.items[0 .. open_holes.items.items.len - 1]) |other| {
            retained_heuristic = saturatingCostAdd(
                retained_heuristic,
                relaxed_heuristic.completionCost(
                    other.target.bits_per_elem,
                    other.target.count,
                    other.depth,
                ),
            );
        }
        for (choices) |choice| {
            const child_storage = grammar.childTargetStorageBytes(
                choice,
                hole.target,
                dtype,
            ) catch |err| switch (err) {
                error.IntegerOverflow => continue,
                else => return err,
            };
            const retained_storage = current_open_storage -
                hole.target.data.len;
            const next_storage = std.math.add(
                usize,
                retained_storage,
                child_storage,
            ) catch continue;
            if (next_storage > options.max_decomposition_bytes)
                continue;

            var child_targets = try grammar.childTargets(
                alloc,
                choice,
                hole.target,
                dtype,
            );
            defer child_targets.deinit(alloc);

            const new_holes = std.math.add(
                usize,
                partial.holes - 1,
                child_targets.streams.len,
            ) catch continue;
            const new_filled = std.math.add(
                usize,
                partial.choices.len,
                1,
            ) catch continue;
            const minimum_final_nodes = std.math.add(
                usize,
                new_filled,
                new_holes,
            ) catch continue;
            if (minimum_final_nodes > options.max_nodes) continue;

            const node_fixed_bytes = choiceFixedBytes(
                choice,
                hole.target,
                child_targets.streams.len,
            );
            const fixed_bytes = std.math.add(
                usize,
                partial.fixed_bytes,
                node_fixed_bytes,
            ) catch continue;
            const size_lower_bound = std.math.add(
                usize,
                fixed_bytes,
                new_holes,
            ) catch continue;
            if (size_lower_bound >= best_bytes) continue;

            const next_choices = try appendChoice(
                alloc,
                partial.choices,
                choice,
            );
            serial +%= 1;
            const grammar_cost = std.math.add(
                u64,
                partial.grammar_cost,
                costForProduction(
                    admitted,
                    production_costs[0..admitted.len],
                    choice.id(),
                ),
            ) catch std.math.maxInt(u64);
            var heuristic = retained_heuristic;
            const child_depth = std.math.add(
                u8,
                hole.depth,
                1,
            ) catch return error.DepthOverflow;
            for (child_targets.streams, 0..) |child, child_slot| {
                _ = child_slot;
                heuristic = saturatingCostAdd(
                    heuristic,
                    relaxed_heuristic.completionCost(
                        child.bits_per_elem,
                        child.count,
                        child_depth,
                    ),
                );
            }
            queue.push(alloc, .{
                .choices = next_choices,
                .holes = new_holes,
                .grammar_cost = grammar_cost,
                .priority_cost = saturatingCostAdd(
                    grammar_cost,
                    heuristic,
                ),
                .fixed_bytes = fixed_bytes,
                .size_lower_bound = size_lower_bound,
                .serial = serial,
            }) catch |err| {
                alloc.free(next_choices);
                return err;
            };
        }
    }

    const status: SearchStatus = if (queue.count() == 0)
        .proven_optimal
    else
        .budget_exhausted;
    return .{
        .program = best,
        .serialized_bytes = best_bytes,
        .expanded = expanded,
        .completed_candidates = completed_candidates,
        .status = status,
        .used_literal_fallback = used_literal_fallback,
    };
}

fn appendChoice(
    alloc: Allocator,
    existing: []const grammar.Choice,
    choice: grammar.Choice,
) Allocator.Error![]grammar.Choice {
    const output = try alloc.alloc(grammar.Choice, existing.len + 1);
    @memcpy(output[0..existing.len], existing);
    output[existing.len] = choice;
    return output;
}

/// Bytes fixed by one selected production. Literal payload data remains a
/// lower bound (one codec tag), while all chosen operation ids and parameters
/// are charged exactly as BRPG v1 emits them.
fn choiceFixedBytes(
    choice: grammar.Choice,
    target: Stream,
    child_count: usize,
) usize {
    return switch (choice) {
        .literal => 1 + 1 + uleb128Size(target.count) + 1 + 1,
        .constant => |word| 1 + 1 +
            uleb128Size(target.count) + uleb128Size(word),
        .repeat => |times| 1 + uleb128Size(times),
        .concat => 1 + uleb128Size(child_count),
        .map_xor => |parameter| 1 + 1 + uleb128Size(parameter),
        .map_add_mod => |parameter| 1 + 1 + uleb128Size(parameter),
        .map_zigzag, .map_gray, .map_bit_reverse => 1 + 1,
        .map_rotate_left => |amount| 1 + 1 + uleb128Size(amount),
        .scan_xor, .scan_add_mod => |initial| 1 + 1 +
            uleb128Size(initial),
        .merge_fields => |low_bits| 1 + 1 +
            uleb128Size(low_bits) + uleb128Size(child_count),
        .merge_float_fields => 1 + 1 + 1 + uleb128Size(child_count),
        .merge_bit_planes, .merge_byte_planes => 1 + 1 +
            uleb128Size(child_count),
    };
}

fn uleb128Size(value: anytype) usize {
    var remaining: u64 = @intCast(value);
    var size: usize = 1;
    while (remaining >= 0x80) : (size += 1)
        remaining >>= 7;
    return size;
}

fn costForProduction(
    admitted: []const grammar.ProductionId,
    costs: []const grammar_prior.Cost,
    production: grammar.ProductionId,
) grammar_prior.Cost {
    for (admitted, costs) |candidate, cost|
        if (candidate == production) return cost;
    unreachable;
}

fn saturatingCostAdd(left: u64, right: anytype) u64 {
    return std.math.add(
        u64,
        left,
        @as(u64, right),
    ) catch std.math.maxInt(u64);
}

fn saturatingCostMul(left: u64, right: usize) u64 {
    return std.math.mul(
        u64,
        left,
        std.math.cast(u64, right) orelse return std.math.maxInt(u64),
    ) catch std.math.maxInt(u64);
}

const RELAXED_WIDTH_COUNT: usize = 33;
const RELAXED_LENGTH_CLASS_COUNT: usize = 2;

const RelaxedLengthClass = enum(u1) {
    one,
    many,
};

/// Costs from Equation 18, conservatively lowered for this implementation's
/// target-conditioned PHOG normalization.
///
/// For a positive target below the depth cap, `Lit`, `Map(zigzag)`,
/// `Map(gray)`, and `Map(bit_reverse)` are always admitted. A length-one
/// target also admits `Const`; a longer target admits both `Scan` families.
/// Hence every such set has at least five families. A family outside the
/// first group implies at least one additional family. At the depth cap only
/// `Lit`, and conditionally `Const`, remain. These facts supply the admitted
/// set lower bounds used by `Prior.contextualCostLowerBounds`.
const RelaxedRuleCosts = struct {
    one: [grammar_prior.PRODUCTION_COUNT]grammar_prior.Cost,
    two: [grammar_prior.PRODUCTION_COUNT]grammar_prior.Cost,
    five: [grammar_prior.PRODUCTION_COUNT]grammar_prior.Cost,
    six: [grammar_prior.PRODUCTION_COUNT]grammar_prior.Cost,
    seven: [grammar_prior.PRODUCTION_COUNT]grammar_prior.Cost,

    fn init(options: Options) !RelaxedRuleCosts {
        var result: RelaxedRuleCosts = undefined;
        try fill(options, 1, &result.one);
        try fill(options, 2, &result.two);
        try fill(options, 5, &result.five);
        try fill(options, 6, &result.six);
        try fill(options, 7, &result.seven);
        return result;
    }

    fn fill(
        options: Options,
        minimum_admitted: usize,
        output: *[grammar_prior.PRODUCTION_COUNT]grammar_prior.Cost,
    ) !void {
        if (options.rule_model) |model| {
            try model.contextualCostLowerBounds(
                minimum_admitted,
                output,
            );
        } else {
            @memset(output, grammar_prior.uniformCost(minimum_admitted));
        }
    }

    fn at(
        self: *const RelaxedRuleCosts,
        production: grammar.ProductionId,
        minimum_admitted: usize,
    ) grammar_prior.Cost {
        const costs = switch (minimum_admitted) {
            1 => &self.one,
            2 => &self.two,
            5 => &self.five,
            6 => &self.six,
            7 => &self.seven,
            else => unreachable,
        };
        for (grammar_prior.PRODUCTIONS, 0..) |known, index|
            if (known == production) return costs[index];
        unreachable;
    }

    fn belowDepthCap(
        self: *const RelaxedRuleCosts,
        production: grammar.ProductionId,
        length_class: RelaxedLengthClass,
    ) grammar_prior.Cost {
        const minimum_admitted: usize = switch (length_class) {
            .one => switch (production) {
                .literal,
                .constant,
                .map_zigzag,
                .map_gray,
                .map_bit_reverse,
                => 5,
                else => 6,
            },
            .many => switch (production) {
                .literal,
                .map_zigzag,
                .map_gray,
                .map_bit_reverse,
                .scan_xor,
                .scan_add_mod,
                => 6,
                else => 7,
            },
        };
        return self.at(production, minimum_admitted);
    }
};

/// Equation 19 over a finite relaxed typed grammar.
///
/// The concrete type `b[n]` is relaxed to three cardinality classes:
/// `b[0]`, `b[1]`, and `b[2+]`. Width is retained, while exact lengths and
/// concrete parameters are dropped. `Concat` and `Scan` may choose either
/// positive child class whenever exact `n` would select one; this only adds
/// derivations. Depth is retained because it is an explicit structural search
/// limit and because the concrete PHOG is normalized over a different
/// admitted set at the cap. Node and decomposition limits are ignored, which
/// is again a relaxation.
///
/// The recurrence is acyclic in depth. Unary transforms recurse to the same
/// width at the next depth, Concat produces two positive children, and Merge
/// recurses to narrower widths. `b[0]` is deliberately outside this table:
/// the implementation's empty-stream extension admits only `Lit`, whose
/// single-production cost is zero.
const RelaxedGrammarHeuristic = struct {
    costs: []u64,
    max_depth: u8,

    fn init(
        alloc: Allocator,
        dtype: Dtype,
        options: Options,
    ) !RelaxedGrammarHeuristic {
        const depth_count = @as(usize, options.grammar_options.max_depth) + 1;
        const costs = try alloc.alloc(
            u64,
            try std.math.mul(
                usize,
                try std.math.mul(
                    usize,
                    depth_count,
                    RELAXED_WIDTH_COUNT,
                ),
                RELAXED_LENGTH_CLASS_COUNT,
            ),
        );
        errdefer alloc.free(costs);
        @memset(costs, 0);

        var result: RelaxedGrammarHeuristic = .{
            .costs = costs,
            .max_depth = options.grammar_options.max_depth,
        };
        const rule_costs = try RelaxedRuleCosts.init(options);

        var depth_cursor = depth_count;
        while (depth_cursor > 0) {
            depth_cursor -= 1;
            const depth: u8 = @intCast(depth_cursor);
            for (1..RELAXED_WIDTH_COUNT) |bits_index| {
                const bits: u8 = @intCast(bits_index);
                inline for (std.enums.values(RelaxedLengthClass)) |length_class| {
                    var best: u64 = if (depth == result.max_depth)
                        rule_costs.at(
                            .literal,
                            if (length_class == .one) 2 else 1,
                        )
                    else
                        rule_costs.belowDepthCap(.literal, length_class);

                    const constant_cost: u64 = if (depth == result.max_depth)
                        rule_costs.at(.constant, 2)
                    else
                        rule_costs.belowDepthCap(.constant, length_class);
                    best = @min(best, constant_cost);

                    if (depth < result.max_depth) {
                        const child_depth = depth + 1;
                        const same_width_child = result.positiveCost(
                            bits,
                            child_depth,
                            length_class,
                        );
                        const cheapest_positive_child = @min(
                            result.positiveCost(
                                bits,
                                child_depth,
                                .one,
                            ),
                            result.positiveCost(
                                bits,
                                child_depth,
                                .many,
                            ),
                        );
                        if (length_class == .many and @min(
                            options.grammar_options.max_repeat_period,
                            grammar.HARD_MAX_REPEAT_PERIOD,
                        ) > 0) {
                            best = @min(
                                best,
                                saturatingCostAdd(
                                    rule_costs.belowDepthCap(
                                        .repeat,
                                        length_class,
                                    ),
                                    cheapest_positive_child,
                                ),
                            );
                        }
                        if (length_class == .many and @min(
                            options.grammar_options.max_concat_splits,
                            grammar.HARD_MAX_CONCAT_SPLITS,
                        ) > 0) {
                            best = @min(
                                best,
                                saturatingCostAdd(
                                    rule_costs.belowDepthCap(
                                        .concat,
                                        length_class,
                                    ),
                                    saturatingCostMul(
                                        cheapest_positive_child,
                                        2,
                                    ),
                                ),
                            );
                        }
                        if (@min(
                            options.grammar_options.max_map_constants,
                            grammar.HARD_MAX_MAP_CONSTANTS,
                        ) > 0) {
                            inline for (
                                .{ .map_xor, .map_add_mod },
                            ) |production| {
                                best = @min(
                                    best,
                                    saturatingCostAdd(
                                        rule_costs.belowDepthCap(
                                            production,
                                            length_class,
                                        ),
                                        same_width_child,
                                    ),
                                );
                            }
                        }
                        inline for (.{
                            grammar.ProductionId.map_zigzag,
                            grammar.ProductionId.map_gray,
                            grammar.ProductionId.map_bit_reverse,
                        }) |production| {
                            best = @min(
                                best,
                                saturatingCostAdd(
                                    rule_costs.belowDepthCap(
                                        production,
                                        length_class,
                                    ),
                                    same_width_child,
                                ),
                            );
                        }
                        if (@min(
                            options.grammar_options.max_rotations,
                            grammar.HARD_MAX_ROTATIONS,
                        ) > 0) {
                            best = @min(
                                best,
                                saturatingCostAdd(
                                    rule_costs.belowDepthCap(
                                        .map_rotate_left,
                                        length_class,
                                    ),
                                    same_width_child,
                                ),
                            );
                        }
                        if (length_class == .many) {
                            inline for (.{
                                grammar.ProductionId.scan_xor,
                                grammar.ProductionId.scan_add_mod,
                            }) |production| {
                                best = @min(
                                    best,
                                    saturatingCostAdd(
                                        rule_costs.belowDepthCap(
                                            production,
                                            length_class,
                                        ),
                                        cheapest_positive_child,
                                    ),
                                );
                            }
                        }

                        if (bits > 1 and @min(
                            options.grammar_options.max_field_splits,
                            grammar.HARD_MAX_FIELD_SPLITS,
                        ) > 0) {
                            var low_bits: u8 = 1;
                            while (low_bits < bits) : (low_bits += 1) {
                                const children = saturatingCostAdd(
                                    result.positiveCost(
                                        low_bits,
                                        child_depth,
                                        length_class,
                                    ),
                                    result.positiveCost(
                                        bits - low_bits,
                                        child_depth,
                                        length_class,
                                    ),
                                );
                                best = @min(
                                    best,
                                    saturatingCostAdd(
                                        rule_costs.belowDepthCap(
                                            .merge_fields,
                                            length_class,
                                        ),
                                        children,
                                    ),
                                );
                            }
                        }
                        if (dtype.floatFields()) |fields| {
                            if (fields.total == bits) {
                                var children = result.positiveCost(
                                    1,
                                    child_depth,
                                    length_class,
                                );
                                children = saturatingCostAdd(
                                    children,
                                    result.positiveCost(
                                        fields.exp,
                                        child_depth,
                                        length_class,
                                    ),
                                );
                                children = saturatingCostAdd(
                                    children,
                                    result.positiveCost(
                                        fields.mant,
                                        child_depth,
                                        length_class,
                                    ),
                                );
                                best = @min(
                                    best,
                                    saturatingCostAdd(
                                        rule_costs.belowDepthCap(
                                            .merge_float_fields,
                                            length_class,
                                        ),
                                        children,
                                    ),
                                );
                            }
                        }
                        if (bits > 1) {
                            best = @min(
                                best,
                                saturatingCostAdd(
                                    rule_costs.belowDepthCap(
                                        .merge_bit_planes,
                                        length_class,
                                    ),
                                    saturatingCostMul(
                                        result.positiveCost(
                                            1,
                                            child_depth,
                                            length_class,
                                        ),
                                        bits,
                                    ),
                                ),
                            );
                        }
                        if (bits > 8) {
                            const full_bytes: usize = bits / 8;
                            const tail_bits: u8 = bits % 8;
                            var children = saturatingCostMul(
                                result.positiveCost(
                                    8,
                                    child_depth,
                                    length_class,
                                ),
                                full_bytes,
                            );
                            if (tail_bits != 0) {
                                children = saturatingCostAdd(
                                    children,
                                    result.positiveCost(
                                        tail_bits,
                                        child_depth,
                                        length_class,
                                    ),
                                );
                            }
                            best = @min(
                                best,
                                saturatingCostAdd(
                                    rule_costs.belowDepthCap(
                                        .merge_byte_planes,
                                        length_class,
                                    ),
                                    children,
                                ),
                            );
                        }
                    }
                    result.setPositiveCost(
                        bits,
                        depth,
                        length_class,
                        best,
                    );
                }
            }
        }
        return result;
    }

    fn deinit(self: *RelaxedGrammarHeuristic, alloc: Allocator) void {
        alloc.free(self.costs);
        self.* = undefined;
    }

    fn completionCost(
        self: *const RelaxedGrammarHeuristic,
        bits: u8,
        count: usize,
        depth: u8,
    ) u64 {
        // The explicit empty Lit extension has q=1 and therefore zero cost.
        if (count == 0) return 0;
        if (depth > self.max_depth or bits == 0 or bits >= RELAXED_WIDTH_COUNT)
            return 0;
        return self.positiveCost(
            bits,
            depth,
            if (count == 1) .one else .many,
        );
    }

    fn positiveCost(
        self: *const RelaxedGrammarHeuristic,
        bits: u8,
        depth: u8,
        length_class: RelaxedLengthClass,
    ) u64 {
        return self.costs[
            (@as(usize, depth) * RELAXED_WIDTH_COUNT + @as(usize, bits)) *
                RELAXED_LENGTH_CLASS_COUNT +
                @intFromEnum(length_class)
        ];
    }

    fn setPositiveCost(
        self: *RelaxedGrammarHeuristic,
        bits: u8,
        depth: u8,
        length_class: RelaxedLengthClass,
        cost: u64,
    ) void {
        self.costs[
            (@as(usize, depth) * RELAXED_WIDTH_COUNT + @as(usize, bits)) *
                RELAXED_LENGTH_CLASS_COUNT +
                @intFromEnum(length_class)
        ] = cost;
    }
};

test "relaxed c chooses a recursive derivation and h orders the queue" {
    const alloc = std.testing.allocator;
    var target = try Stream.init(alloc, 4, 8);
    defer target.deinit(alloc);
    target.setU32(0, 1);
    target.setU32(1, 2);
    target.setU32(2, 3);
    target.setU32(3, 4);
    const context = grammar_prior.Context.fromTarget(
        target,
        .u8,
        null,
        0,
        0,
    );

    var counts = grammar_prior.Counts.init();
    defer counts.deinit(alloc);
    try counts.observe(alloc, context, .map_zigzag, 10_000);
    var learned = try counts.toPrior(alloc, .{
        .learned_numerator = 19,
        .learned_denominator = 20,
    });
    defer learned.deinit(alloc);

    const options: Options = .{
        .rule_model = &learned,
        .grammar_options = .{
            .max_depth = 1,
            .max_repeat_period = 0,
            .max_concat_splits = 0,
            .max_map_constants = 0,
            .max_rotations = 0,
            .max_field_splits = 0,
        },
    };
    const rule_costs = try RelaxedRuleCosts.init(options);
    var heuristic = try RelaxedGrammarHeuristic.init(
        alloc,
        .u8,
        options,
    );
    defer heuristic.deinit(alloc);

    const leaf_cost = heuristic.completionCost(8, target.count, 1);
    const recursive_cost = saturatingCostAdd(
        rule_costs.belowDepthCap(.map_zigzag, .many),
        leaf_cost,
    );
    const root_cost = heuristic.completionCost(8, target.count, 0);
    const direct_literal_cost =
        rule_costs.belowDepthCap(.literal, .many);

    // This is a genuine two-rule shortest derivation:
    // Map(zigzag, Lit(_)), not the cheapest first rule renamed as c(A).
    try std.testing.expectEqual(recursive_cost, root_cost);
    try std.testing.expect(root_cost < direct_literal_cost);

    var queue = Queue.initContext({});
    defer queue.deinit(alloc);
    const high_choices = try alloc.alloc(grammar.Choice, 0);
    errdefer alloc.free(high_choices);
    const low_choices = try alloc.alloc(grammar.Choice, 0);
    errdefer alloc.free(low_choices);
    const paid_cost: u64 = 100;
    try queue.push(alloc, .{
        .choices = high_choices,
        .holes = 1,
        .grammar_cost = paid_cost,
        .priority_cost = saturatingCostAdd(
            paid_cost,
            direct_literal_cost,
        ),
        .fixed_bytes = 0,
        .size_lower_bound = 0,
        .serial = 0,
    });
    try queue.push(alloc, .{
        .choices = low_choices,
        .holes = 1,
        .grammar_cost = paid_cost,
        .priority_cost = saturatingCostAdd(paid_cost, root_cost),
        .fixed_bytes = 0,
        .size_lower_bound = 0,
        // If h were ignored, the earlier serial above would win.
        .serial = 1,
    });

    var first = queue.pop().?;
    defer first.deinit(alloc);
    var second = queue.pop().?;
    defer second.deinit(alloc);
    try std.testing.expectEqual(@as(u64, 1), first.serial);
    try std.testing.expectEqual(@as(u64, 0), second.serial);
}

fn replayOpenHoles(
    alloc: Allocator,
    root_target: Stream,
    dtype: Dtype,
    choices: []const grammar.Choice,
    max_decomposition_bytes: usize,
) !OpenHoles {
    var result: OpenHoles = .{};
    errdefer result.deinit(alloc);

    var root_view = root_target;
    root_view.owns_data = false;
    try result.items.append(alloc, .{
        .target = root_view,
        .depth = 0,
        .parent = null,
        .child_slot = 0,
    });

    for (choices) |choice| {
        if (result.items.items.len == 0)
            return error.InvalidPartialProgram;
        const parent_index = result.items.items.len - 1;
        const parent = result.items.items[parent_index];
        const current_storage = try openTargetStorageBytes(&result);
        const child_storage = try grammar.childTargetStorageBytes(
            choice,
            parent.target,
            dtype,
        );
        const next_storage = std.math.add(
            usize,
            current_storage - parent.target.data.len,
            child_storage,
        ) catch return error.DecompositionLimitExceeded;
        if (next_storage > max_decomposition_bytes)
            return error.DecompositionLimitExceeded;
        var children = try grammar.childTargets(
            alloc,
            choice,
            parent.target,
            dtype,
        );
        errdefer children.deinit(alloc);
        const child_depth = std.math.add(u8, parent.depth, 1) catch
            return error.DepthOverflow;

        try result.items.ensureUnusedCapacity(alloc, children.streams.len);
        result.items.items.len = parent_index;
        var parent_target = parent.target;
        parent_target.deinit(alloc);

        var child_index = children.streams.len;
        while (child_index > 0) {
            child_index -= 1;
            result.items.appendAssumeCapacity(.{
                .target = children.streams[child_index],
                .depth = child_depth,
                .parent = choice.id(),
                .child_slot = @intCast(child_index),
            });
        }
        alloc.free(children.streams);
        children.streams = &.{};
    }
    return result;
}

fn openTargetStorageBytes(open_holes: *const OpenHoles) !usize {
    var total: usize = 0;
    for (open_holes.items.items) |hole| {
        total = std.math.add(
            usize,
            total,
            hole.target.data.len,
        ) catch return error.IntegerOverflow;
    }
    return total;
}

fn buildProgram(
    alloc: Allocator,
    choices: []const grammar.Choice,
    target: Stream,
    dtype: Dtype,
) !dsl.Program {
    var cursor: usize = 0;
    var program = try buildNode(alloc, choices, &cursor, target, dtype);
    errdefer program.deinit(alloc);
    if (cursor != choices.len) return error.InvalidPartialProgram;
    return program;
}

fn buildNode(
    alloc: Allocator,
    choices: []const grammar.Choice,
    cursor: *usize,
    target: Stream,
    dtype: Dtype,
) !dsl.Program {
    if (cursor.* >= choices.len) return error.InvalidPartialProgram;
    const choice = choices[cursor.*];
    cursor.* += 1;

    var child_targets = try grammar.childTargets(alloc, choice, target, dtype);
    defer child_targets.deinit(alloc);

    if (child_targets.streams.len == 0) {
        return switch (choice) {
            .literal => dsl.Program.literalFromStream(alloc, target),
            .constant => |word| dsl.Program.constant(
                target.bits_per_elem,
                target.count,
                word,
            ),
            else => error.InvalidPartialProgram,
        };
    }

    const children = try alloc.alloc(
        dsl.Program,
        child_targets.streams.len,
    );
    var initialized: usize = 0;
    var adopted = false;
    errdefer if (!adopted) {
        for (children[0..initialized]) |*child| child.deinit(alloc);
        alloc.free(children);
    };
    for (child_targets.streams, 0..) |child_target, index| {
        children[index] = try buildNode(
            alloc,
            choices,
            cursor,
            child_target,
            dtype,
        );
        initialized += 1;
    }

    return switch (choice) {
        .repeat => |times| blk: {
            if (children.len != 1) return error.InvalidPartialProgram;
            const parent = try dsl.Program.repeat(alloc, times, children[0]);
            initialized = 0;
            alloc.free(children);
            adopted = true;
            break :blk parent;
        },
        .concat => blk: {
            const parent = try dsl.Program.concatOwned(children);
            adopted = true;
            break :blk parent;
        },
        .map_xor,
        .map_add_mod,
        .map_zigzag,
        .map_gray,
        .map_rotate_left,
        .map_bit_reverse,
        => blk: {
            if (children.len != 1) return error.InvalidPartialProgram;
            const parent = try dsl.Program.map(
                alloc,
                choice.mapOperation().?,
                children[0],
            );
            initialized = 0;
            alloc.free(children);
            adopted = true;
            break :blk parent;
        },
        .scan_xor, .scan_add_mod => |initial| blk: {
            if (children.len != 1) return error.InvalidPartialProgram;
            const parent = try dsl.Program.scan(
                alloc,
                choice.scanOperation().?,
                initial,
                children[0],
            );
            initialized = 0;
            alloc.free(children);
            adopted = true;
            break :blk parent;
        },
        .merge_fields,
        .merge_float_fields,
        .merge_bit_planes,
        .merge_byte_planes,
        => blk: {
            const parent = try dsl.Program.mergeOwned(
                choice.mergeOperation().?,
                children,
            );
            adopted = true;
            break :blk parent;
        },
        .literal, .constant => error.InvalidPartialProgram,
    };
}

fn streamsEqual(left: Stream, right: Stream) bool {
    if (left.bits_per_elem != right.bits_per_elem or
        left.count != right.count)
        return false;
    for (0..left.count) |index|
        if (left.getU32(index) != right.getU32(index)) return false;
    return true;
}
