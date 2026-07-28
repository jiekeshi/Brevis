//! Budgeted best-first synthesis over target-directed semantic programs.
//!
//! A PHOG supplies contextual rule costs, A* adds an admissible relaxed-grammar
//! heuristic, and complete candidates are compared by exact canonical bytes.

const std = @import("std");
const builtin = @import("builtin");
const dsl = @import("dsl.zig");
const grammar = @import("grammar.zig");
const phog = @import("phog.zig");
const interpreter = @import("interpreter.zig");
const program_format = @import("program_format.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const Dtype = types.Dtype;
const Stream = types.Stream;

pub const Options = struct {
    max_expansions: usize = 1,
    max_nodes: usize = 64,
    astar_heuristic: bool = true,
    seed_float_fields: bool = false,
    /// Maximum total storage of simultaneously open target streams. Choices
    /// that would exceed it are pruned before decomposition allocation.
    max_decomposition_bytes: usize = 512 * 1024 * 1024,
    grammar_options: grammar.Options = .{},
    /// Encoder-only PHOG policy. It changes queue order but never legality,
    /// canonical byte cost, or complete-candidate selection.
    phog_prior: ?*const phog.Prior = null,
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
    serialized_program: []u8,
    serialized_bytes: usize,
    expanded: usize,
    completed_candidates: usize,
    status: SearchStatus,
    used_literal_fallback: bool,

    pub fn deinit(self: *Result, alloc: Allocator) void {
        self.program.deinit(alloc);
        alloc.free(self.serialized_program);
        self.serialized_program = &.{};
    }
};

const ChoicePath = struct {
    parent: ?*const ChoicePath,
    choice: grammar.Choice,
};

fn extendChoicePath(
    path_alloc: Allocator,
    parent: ?*const ChoicePath,
    choice: grammar.Choice,
) Allocator.Error!*const ChoicePath {
    const node = try path_alloc.create(ChoicePath);
    node.* = .{
        .parent = parent,
        .choice = choice,
    };
    return node;
}

const Partial = struct {
    path: ?*const ChoicePath,
    choice_count: usize,
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

const ContextBuildCounter = struct {
    var value: std.atomic.Value(usize) = .init(0);
};

fn contextForHole(hole: Hole, dtype: Dtype) phog.Context {
    if (builtin.is_test)
        _ = ContextBuildCounter.value.fetchAdd(1, .monotonic);
    return phog.Context.fromTarget(
        hole.target,
        dtype,
        hole.parent,
        hole.child_slot,
        hole.depth,
    );
}

pub const testing = if (builtin.is_test) struct {
    pub fn resetContextBuildCount() void {
        ContextBuildCounter.value.store(0, .monotonic);
    }

    pub fn contextBuildCount() usize {
        return ContextBuildCounter.value.load(.monotonic);
    }
} else struct {};

const OpenHoles = struct {
    items: std.ArrayList(Hole) = .empty,
    storage_bytes: usize = 0,

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
    return synthesizeImpl(alloc, target, dtype, options, .own, .serialized);
}

/// Encoder-only synthesis whose returned root `Lit` may view `target`.
/// The target must outlive every read or serialization of that result.
/// Structured results remain self-contained.
pub fn synthesizeBorrowingTarget(
    alloc: Allocator,
    target: Stream,
    dtype: Dtype,
    options: Options,
) !Result {
    return synthesizeImpl(alloc, target, dtype, options, .borrow, .serialized);
}

/// Search for callers that only inspect the winning program and its exact
/// canonical length. `serialized_program` is empty and `target` is borrowed.
pub fn synthesizeUnserialized(
    alloc: Allocator,
    target: Stream,
    dtype: Dtype,
    options: Options,
) !Result {
    return synthesizeImpl(alloc, target, dtype, options, .borrow, .size_only);
}

const RootLiteralStorage = enum {
    own,
    borrow,
};

const OutputMode = enum {
    serialized,
    size_only,
};

fn synthesizeImpl(
    alloc: Allocator,
    target: Stream,
    dtype: Dtype,
    options: Options,
    root_literal_storage: RootLiteralStorage,
    output_mode: OutputMode,
) !Result {
    if (target.bits_per_elem != dtype.bitWidth())
        return error.TensorWidthMismatch;

    // The universal fallback is a complete semantic program, not an external
    // raw block mode.
    var best = try borrowedLiteral(target);
    errdefer best.deinit(alloc);

    if (options.max_expansions == 0) {
        const serialized_program: []u8 = switch (output_mode) {
            .serialized => try program_format.serialize(alloc, best),
            .size_only => &.{},
        };
        errdefer alloc.free(serialized_program);
        const exact_bytes = switch (output_mode) {
            .serialized => serialized_program.len,
            .size_only => try program_format.serializedSize(alloc, best),
        };
        if (root_literal_storage == .own) {
            const owned = try dsl.Program.literalFromStream(alloc, target);
            best.deinit(alloc);
            best = owned;
        }
        return .{
            .program = best,
            .serialized_program = serialized_program,
            .serialized_bytes = exact_bytes,
            .expanded = 0,
            .completed_candidates = 0,
            .status = .budget_exhausted,
            .used_literal_fallback = true,
        };
    }

    var best_serialized: []u8 = &.{};
    errdefer alloc.free(best_serialized);
    var best_prepared: ?program_format.PreparedSerialization = null;
    defer if (best_prepared) |*value| value.deinit(alloc);
    var best_bytes = switch (output_mode) {
        .serialized => blk: {
            best_serialized = try program_format.serialize(alloc, best);
            break :blk best_serialized.len;
        },
        .size_only => try program_format.serializedSize(alloc, best),
    };
    var used_literal_fallback = true;
    var completed_candidates: usize = 0;
    var expanded: usize = 0;
    var cutoff_size_lower_bound: usize = std.math.maxInt(usize);
    var rollout_frontier: ?Partial = null;
    var shallow_float_fields_evaluated = false;

    if (try shallowFloatFieldsCandidate(alloc, target, dtype, options)) |candidate_value| {
        shallow_float_fields_evaluated = true;
        var candidate = candidate_value;
        var candidate_owned = true;
        defer if (candidate_owned) candidate.deinit(alloc);
        completed_candidates += 1;
        if (try adoptIfSmaller(
            alloc,
            &best,
            &best_serialized,
            &best_bytes,
            &best_prepared,
            candidate,
        )) {
            candidate_owned = false;
            used_literal_fallback = false;
        }
    }

    var path_arena = std.heap.ArenaAllocator.init(alloc);
    defer path_arena.deinit();
    const path_alloc = path_arena.allocator();
    var queue = Queue.initContext({});
    defer queue.deinit(alloc);

    var serial: u64 = 0;
    var search_heuristic = try SearchHeuristic.init(
        alloc,
        dtype,
        options,
    );
    defer search_heuristic.deinit(alloc);
    const initial_heuristic = search_heuristic.completionCost(
        target.bits_per_elem,
        target.count,
        0,
    );
    const prefix_bytes = program_format.MAGIC.len + 1;
    try queue.push(alloc, .{
        .path = null,
        .choice_count = 0,
        .holes = 1,
        .grammar_cost = 0,
        .priority_cost = initial_heuristic,
        .fixed_bytes = prefix_bytes,
        .size_lower_bound = prefix_bytes + 1,
        .serial = serial,
    });

    while (queue.count() > 0) {
        const partial = queue.pop().?;

        if (partial.size_lower_bound >= best_bytes) continue;

        if (partial.holes == 0) {
            if (isRootLiteral(partial.path, partial.choice_count)) {
                completed_candidates += 1;
                continue;
            }
            const choices = try materializeChoices(
                alloc,
                partial.path,
                partial.choice_count,
            );
            defer alloc.free(choices);
            var candidate = try buildProgram(
                alloc,
                choices,
                target,
                dtype,
            );
            var candidate_owned = true;
            defer if (candidate_owned) candidate.deinit(alloc);

            completed_candidates += 1;
            if (try adoptIfSmaller(
                alloc,
                &best,
                &best_serialized,
                &best_bytes,
                &best_prepared,
                candidate,
            )) {
                candidate_owned = false;
                used_literal_fallback = false;
            }
            continue;
        }

        if (partial.choice_count + partial.holes > options.max_nodes)
            continue;

        if (expanded == options.max_expansions) {
            cutoff_size_lower_bound = @min(
                cutoff_size_lower_bound,
                partial.size_lower_bound,
            );
            retainPreferredPartial(&rollout_frontier, partial);
            continue;
        }

        var open_holes = try replayOpenHoles(
            alloc,
            target,
            dtype,
            partial.path,
            options.max_decomposition_bytes,
        );
        defer open_holes.deinit(alloc);
        if (open_holes.items.items.len != partial.holes)
            return error.InvalidPartialProgram;
        const hole = open_holes.leftmost().*;

        expanded += 1;
        const choices = if (hole.depth == 0)
            try grammar.propose(
                alloc,
                hole.target,
                dtype,
                hole.depth,
                options.grammar_options,
            )
        else
            try grammar.proposeKnownValid(
                alloc,
                hole.target,
                dtype,
                hole.depth,
                options.grammar_options,
            );
        defer alloc.free(choices);
        const legal = grammar.families(choices);
        if (legal.len == 0) return error.NoLiteralFallback;
        const admitted = legal.slice();
        var production_costs: [phog.PRODUCTION_COUNT]phog.Cost = undefined;
        if (options.phog_prior) |phog_prior| {
            try phog_prior.scoreSet(
                contextForHole(hole, dtype),
                admitted,
                production_costs[0..admitted.len],
            );
        } else {
            @memset(
                production_costs[0..admitted.len],
                phog.uniformCost(admitted.len),
            );
        }

        const current_open_storage = open_holes.storage_bytes;
        var retained_heuristic: u64 = 0;
        for (open_holes.items.items[0 .. open_holes.items.items.len - 1]) |other| {
            retained_heuristic = saturatingCostAdd(
                retained_heuristic,
                search_heuristic.completionCost(
                    other.target.bits_per_elem,
                    other.target.count,
                    other.depth,
                ),
            );
        }
        for (choices) |choice| {
            const child_storage = grammar.childTargetStorageBytesForProposal(
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

            const child_shapes = grammar.childShapesForProposal(
                choice,
                hole.target,
            );
            const child_count = child_shapes.len;
            const new_holes = std.math.add(
                usize,
                partial.holes - 1,
                child_count,
            ) catch continue;
            const new_filled = std.math.add(
                usize,
                partial.choice_count,
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
                child_count,
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
            for (child_shapes.slice()) |child| {
                heuristic = saturatingCostAdd(
                    heuristic,
                    search_heuristic.completionCost(
                        child.bits,
                        child.count,
                        child_depth,
                    ),
                );
            }

            var next: Partial = .{
                .path = partial.path,
                .choice_count = new_filled,
                .holes = new_holes,
                .grammar_cost = grammar_cost,
                .priority_cost = saturatingCostAdd(
                    grammar_cost,
                    heuristic,
                ),
                .fixed_bytes = fixed_bytes,
                .size_lower_bound = size_lower_bound,
                .serial = serial,
            };
            if (new_holes > options.max_expansions - expanded) {
                cutoff_size_lower_bound = @min(
                    cutoff_size_lower_bound,
                    size_lower_bound,
                );
                if (rollout_frontier == null or
                    comparePartial({}, next, rollout_frontier.?) == .lt)
                {
                    next.path = try extendChoicePath(
                        path_alloc,
                        partial.path,
                        choice,
                    );
                    rollout_frontier = next;
                }
                continue;
            }

            next.path = try extendChoicePath(
                path_alloc,
                partial.path,
                choice,
            );
            serial +%= 1;
            next.serial = serial;
            queue.push(alloc, next) catch |err| return err;
        }
    }

    if (rollout_frontier) |partial| if (partial.size_lower_bound < best_bytes and
        !(shallow_float_fields_evaluated and
            isShallowFloatFieldsCompletion(partial, dtype)))
    {
        const choices = try materializeLiteralCompletion(
            alloc,
            partial,
        );
        defer alloc.free(choices);
        var candidate = try buildProgram(
            alloc,
            choices,
            target,
            dtype,
        );
        var candidate_owned = true;
        defer if (candidate_owned) candidate.deinit(alloc);

        completed_candidates += 1;
        if (try adoptIfSmaller(
            alloc,
            &best,
            &best_serialized,
            &best_bytes,
            &best_prepared,
            candidate,
        )) {
            candidate_owned = false;
            used_literal_fallback = false;
        }
    };

    const status: SearchStatus = if (queue.count() == 0 and
        cutoff_size_lower_bound >= best_bytes)
        .proven_optimal
    else
        .budget_exhausted;
    if (root_literal_storage == .own) switch (best.kind) {
        .literal => |literal| if (!literal.owns_data) {
            const owned = try dsl.Program.literalFromStream(alloc, literal);
            best.deinit(alloc);
            best = owned;
        },
        else => {},
    };
    // The root literal stores the target words themselves, so only a program
    // that replaced it has to be executed and checked.
    if (!used_literal_fallback) {
        var output = try interpreter.execute(alloc, best);
        defer output.deinit(alloc);
        if (!target.eql(output)) return error.InvalidCandidate;
    }
    const serialized_program: []u8 = switch (output_mode) {
        .serialized => if (best_serialized.len != 0)
            best_serialized
        else if (best_prepared) |prepared|
            try program_format.emitPrepared(alloc, best, prepared)
        else
            try program_format.serialize(alloc, best),
        .size_only => &.{},
    };
    best_serialized = &.{};
    std.debug.assert(output_mode == .size_only or
        serialized_program.len == best_bytes);
    return .{
        .program = best,
        .serialized_program = serialized_program,
        .serialized_bytes = best_bytes,
        .expanded = expanded,
        .completed_candidates = completed_candidates,
        .status = status,
        .used_literal_fallback = used_literal_fallback,
    };
}

fn isRootLiteral(path: ?*const ChoicePath, choice_count: usize) bool {
    if (choice_count != 1) return false;
    return switch (path.?.choice) {
        .literal => true,
        else => false,
    };
}

fn isShallowFloatFieldsCompletion(partial: Partial, dtype: Dtype) bool {
    if (partial.choice_count != 1 or partial.holes != 3) return false;
    return switch (partial.path.?.choice) {
        .merge_float_fields => |candidate_dtype| candidate_dtype == dtype,
        else => false,
    };
}

fn shallowFloatFieldsCandidate(
    alloc: Allocator,
    target: Stream,
    dtype: Dtype,
    options: Options,
) !?dsl.Program {
    if (!options.seed_float_fields or
        options.max_expansions == 0 or
        options.max_nodes < 4 or
        options.grammar_options.max_depth == 0 or
        target.count == 0 or
        dtype.floatFields() == null)
    {
        return null;
    }

    const root: grammar.Choice = .{ .merge_float_fields = dtype };
    if (try grammar.childTargetStorageBytes(root, target, dtype) >
        options.max_decomposition_bytes)
    {
        return null;
    }
    const choices = [_]grammar.Choice{
        root,
        .literal,
        .literal,
        .literal,
    };
    return try buildProgram(alloc, &choices, target, dtype);
}

fn materializeChoices(
    alloc: Allocator,
    path: ?*const ChoicePath,
    count: usize,
) Allocator.Error![]grammar.Choice {
    const output = try alloc.alloc(grammar.Choice, count);
    var cursor = count;
    var current = path;
    while (current) |node| {
        cursor -= 1;
        output[cursor] = node.choice;
        current = node.parent;
    }
    std.debug.assert(cursor == 0);
    return output;
}

fn materializeLiteralCompletion(
    alloc: Allocator,
    partial: Partial,
) Allocator.Error![]grammar.Choice {
    const count = partial.choice_count + partial.holes;
    const output = try alloc.alloc(grammar.Choice, count);
    var cursor = partial.choice_count;
    var current = partial.path;
    while (current) |node| {
        cursor -= 1;
        output[cursor] = node.choice;
        current = node.parent;
    }
    std.debug.assert(cursor == 0);
    for (output[partial.choice_count..]) |*choice| choice.* = .literal;
    return output;
}

fn retainPreferredPartial(slot: *?Partial, candidate: Partial) void {
    if (slot.* == null or comparePartial({}, candidate, slot.*.?) == .lt)
        slot.* = candidate;
}

/// Bytes fixed by one selected production. Literal payload data remains a
/// lower bound (one codec tag), while all chosen operation ids and parameters
/// are charged exactly as BRPG emits them.
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
    costs: []const phog.Cost,
    production: grammar.ProductionId,
) phog.Cost {
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
/// For widths above one, a positive target below the depth cap always admits
/// `Lit`, `Map(zigzag)`, `Map(gray)`, and `Map(bit_reverse)`. A length-one
/// target also admits `Const`; a longer target admits both `Scan` families.
/// Thus the corresponding minima are five and six families, and an admitted
/// family outside either base set raises its minimum by one. At width one,
/// normal-form pruning leaves `Lit` plus `Const` for length one, or `Lit` plus
/// `Scan(xor)` for longer targets, so only a two-family bound is valid. At the
/// depth cap only `Lit`, and conditionally `Const`, remain.
const RelaxedRuleCosts = struct {
    one: [phog.PRODUCTION_COUNT]phog.Cost,
    two: [phog.PRODUCTION_COUNT]phog.Cost,
    five: [phog.PRODUCTION_COUNT]phog.Cost,
    six: [phog.PRODUCTION_COUNT]phog.Cost,
    seven: [phog.PRODUCTION_COUNT]phog.Cost,

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
        output: *[phog.PRODUCTION_COUNT]phog.Cost,
    ) !void {
        if (options.phog_prior) |prior| {
            try prior.contextualCostLowerBounds(
                minimum_admitted,
                output,
            );
        } else {
            @memset(output, phog.uniformCost(minimum_admitted));
        }
    }

    fn at(
        self: *const RelaxedRuleCosts,
        production: grammar.ProductionId,
        minimum_admitted: usize,
    ) phog.Cost {
        const costs = switch (minimum_admitted) {
            1 => &self.one,
            2 => &self.two,
            5 => &self.five,
            6 => &self.six,
            7 => &self.seven,
            else => unreachable,
        };
        for (phog.PRODUCTIONS, 0..) |known, index|
            if (known == production) return costs[index];
        unreachable;
    }

    fn belowDepthCap(
        self: *const RelaxedRuleCosts,
        production: grammar.ProductionId,
        bits: u8,
        length_class: RelaxedLengthClass,
    ) phog.Cost {
        if (bits == 1) return self.at(production, 2);
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
                        rule_costs.belowDepthCap(
                            .literal,
                            bits,
                            length_class,
                        );

                    const constant_cost: u64 = if (depth == result.max_depth)
                        rule_costs.at(.constant, 2)
                    else
                        rule_costs.belowDepthCap(
                            .constant,
                            bits,
                            length_class,
                        );
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
                                        bits,
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
                                        bits,
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
                                            bits,
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
                                        bits,
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
                                        bits,
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
                                            bits,
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
                                            bits,
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
                                            bits,
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
                                        bits,
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
                                        bits,
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

const SearchHeuristic = struct {
    relaxed: ?RelaxedGrammarHeuristic,

    fn init(
        alloc: Allocator,
        dtype: Dtype,
        options: Options,
    ) !SearchHeuristic {
        return .{
            .relaxed = if (options.astar_heuristic)
                try RelaxedGrammarHeuristic.init(alloc, dtype, options)
            else
                null,
        };
    }

    fn deinit(self: *SearchHeuristic, alloc: Allocator) void {
        if (self.relaxed) |*relaxed| relaxed.deinit(alloc);
        self.* = undefined;
    }

    fn completionCost(
        self: *const SearchHeuristic,
        bits: u8,
        count: usize,
        depth: u8,
    ) u64 {
        const relaxed = self.relaxed orelse return 0;
        return relaxed.completionCost(bits, count, depth);
    }
};

test "disabled A-star heuristic contributes zero completion cost" {
    const alloc = std.testing.allocator;
    var heuristic = try SearchHeuristic.init(
        alloc,
        .u8,
        .{ .astar_heuristic = false },
    );
    defer heuristic.deinit(alloc);
    try std.testing.expectEqual(
        @as(u64, 0),
        heuristic.completionCost(8, 1024, 0),
    );
}

test "relaxed c chooses a recursive derivation and h orders the queue" {
    const alloc = std.testing.allocator;
    var target = try Stream.init(alloc, 4, 8);
    defer target.deinit(alloc);
    target.setU32(0, 1);
    target.setU32(1, 2);
    target.setU32(2, 3);
    target.setU32(3, 4);
    const context = phog.Context.fromTarget(
        target,
        .u8,
        null,
        0,
        0,
    );

    var counts = phog.Counts.init();
    defer counts.deinit(alloc);
    try counts.observe(alloc, context, .map_zigzag, 10_000);
    var learned = try counts.toPrior(alloc, .{
        .learned_numerator = 19,
        .learned_denominator = 20,
    });
    defer learned.deinit(alloc);

    const options: Options = .{
        .phog_prior = &learned,
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
        rule_costs.belowDepthCap(.map_zigzag, 8, .many),
        leaf_cost,
    );
    const root_cost = heuristic.completionCost(8, target.count, 0);
    const direct_literal_cost =
        rule_costs.belowDepthCap(.literal, 8, .many);

    // This is a genuine two-rule shortest derivation:
    // Map(zigzag, Lit(_)), not the cheapest first rule renamed as c(A).
    try std.testing.expectEqual(recursive_cost, root_cost);
    try std.testing.expect(root_cost < direct_literal_cost);

    var queue = Queue.initContext({});
    defer queue.deinit(alloc);
    const paid_cost: u64 = 100;
    try queue.push(alloc, .{
        .path = null,
        .choice_count = 0,
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
        .path = null,
        .choice_count = 0,
        .holes = 1,
        .grammar_cost = paid_cost,
        .priority_cost = saturatingCostAdd(paid_cost, root_cost),
        .fixed_bytes = 0,
        .size_lower_bound = 0,
        // If h were ignored, the earlier serial above would win.
        .serial = 1,
    });

    const first = queue.pop().?;
    const second = queue.pop().?;
    try std.testing.expectEqual(@as(u64, 1), first.serial);
    try std.testing.expectEqual(@as(u64, 0), second.serial);
}

test "one-bit relaxed rule costs use the two-family admitted lower bound" {
    const rule_costs = try RelaxedRuleCosts.init(.{});
    inline for (std.enums.values(RelaxedLengthClass)) |length_class| {
        for (phog.PRODUCTIONS) |production|
            try std.testing.expectEqual(
                rule_costs.at(production, 2),
                rule_costs.belowDepthCap(production, 1, length_class),
            );
    }
    try std.testing.expectEqual(
        rule_costs.at(.literal, 5),
        rule_costs.belowDepthCap(.literal, 8, .one),
    );
    try std.testing.expectEqual(
        rule_costs.at(.literal, 6),
        rule_costs.belowDepthCap(.literal, 8, .many),
    );
    try std.testing.expectEqual(
        rule_costs.at(.repeat, 7),
        rule_costs.belowDepthCap(.repeat, 8, .many),
    );
}

fn replayOpenHoles(
    alloc: Allocator,
    root_target: Stream,
    dtype: Dtype,
    path: ?*const ChoicePath,
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
    result.storage_bytes = root_view.data.len;

    try replayChoicePath(
        alloc,
        &result,
        path,
        dtype,
        max_decomposition_bytes,
    );
    return result;
}

fn replayChoicePath(
    alloc: Allocator,
    open_holes: *OpenHoles,
    path: ?*const ChoicePath,
    dtype: Dtype,
    max_decomposition_bytes: usize,
) !void {
    const node = path orelse return;
    try replayChoicePath(
        alloc,
        open_holes,
        node.parent,
        dtype,
        max_decomposition_bytes,
    );
    try applyChoiceToOpenHoles(
        alloc,
        open_holes,
        node.choice,
        dtype,
        max_decomposition_bytes,
    );
}

fn applyChoiceToOpenHoles(
    alloc: Allocator,
    open_holes: *OpenHoles,
    choice: grammar.Choice,
    dtype: Dtype,
    max_decomposition_bytes: usize,
) !void {
    if (open_holes.items.items.len == 0)
        return error.InvalidPartialProgram;
    const parent_index = open_holes.items.items.len - 1;
    const parent = open_holes.items.items[parent_index];
    const current_storage = open_holes.storage_bytes;
    const child_storage = try grammar.childTargetStorageBytesForProposal(
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
    var children = try grammar.childTargetsForProposal(
        alloc,
        choice,
        parent.target,
        dtype,
    );
    errdefer children.deinit(alloc);
    const child_depth = std.math.add(u8, parent.depth, 1) catch
        return error.DepthOverflow;

    try open_holes.items.ensureUnusedCapacity(alloc, children.streams.len);
    open_holes.items.items.len = parent_index;
    var parent_target = parent.target;
    parent_target.deinit(alloc);

    var child_index = children.streams.len;
    while (child_index > 0) {
        child_index -= 1;
        open_holes.items.appendAssumeCapacity(.{
            .target = children.streams[child_index],
            .depth = child_depth,
            .parent = choice.id(),
            .child_slot = @intCast(child_index),
        });
    }
    alloc.free(children.streams);
    children.streams = &.{};
    open_holes.storage_bytes = next_storage;
}

fn buildProgram(
    alloc: Allocator,
    choices: []const grammar.Choice,
    target: Stream,
    dtype: Dtype,
) !dsl.Program {
    var cursor: usize = 0;
    var root_target = target;
    root_target.owns_data = false;
    var program = try buildNode(
        alloc,
        choices,
        &cursor,
        &root_target,
        dtype,
        false,
    );
    errdefer program.deinit(alloc);
    if (cursor != choices.len) return error.InvalidPartialProgram;
    return program;
}

fn buildNode(
    alloc: Allocator,
    choices: []const grammar.Choice,
    cursor: *usize,
    target: *Stream,
    dtype: Dtype,
    adopt_literal_target: bool,
) !dsl.Program {
    if (cursor.* >= choices.len) return error.InvalidPartialProgram;
    const choice = choices[cursor.*];
    cursor.* += 1;

    var child_targets = try grammar.childTargetsForProposal(
        alloc,
        choice,
        target.*,
        dtype,
    );
    defer child_targets.deinit(alloc);

    if (child_targets.streams.len == 0) {
        return switch (choice) {
            .literal => if (adopt_literal_target)
                adoptLiteralTarget(alloc, target)
            else
                dsl.Program.literalFromStream(alloc, target.*),
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
    for (child_targets.streams, 0..) |*child_target, index| {
        children[index] = try buildNode(
            alloc,
            choices,
            cursor,
            child_target,
            dtype,
            true,
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

/// Adopt `candidate` when it is strictly smaller, keeping the literal codec
/// analyses that sizing already produced so the final serialization does not
/// repeat them.
fn adoptIfSmaller(
    alloc: Allocator,
    winner: *dsl.Program,
    serialized: *[]u8,
    byte_size: *usize,
    prepared: *?program_format.PreparedSerialization,
    candidate: dsl.Program,
) !bool {
    var candidate_prepared = (try program_format.prepareAtMost(
        alloc,
        candidate,
        byte_size.*,
    )) orelse return false;
    errdefer candidate_prepared.deinit(alloc);

    winner.deinit(alloc);
    alloc.free(serialized.*);
    if (prepared.*) |*old| old.deinit(alloc);
    winner.* = candidate;
    serialized.* = &.{};
    byte_size.* = candidate_prepared.size;
    prepared.* = candidate_prepared;
    return true;
}

fn borrowedLiteral(target: Stream) !dsl.Program {
    var view = target;
    view.owns_data = false;
    const program: dsl.Program = .{ .kind = .{ .literal = view } };
    _ = try program.typeOf();
    return program;
}

fn adoptLiteralTarget(
    alloc: Allocator,
    target: *Stream,
) !dsl.Program {
    if (!target.owns_data)
        return dsl.Program.literalFromStream(alloc, target.*);
    const program: dsl.Program = .{ .kind = .{ .literal = target.* } };
    _ = try program.typeOf();
    target.owns_data = false;
    return program;
}
