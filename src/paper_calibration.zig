//! Deterministic input-local PHOG calibration for the paper DSL.
//!
//! Calibration is an encoder policy step, not part of the archive or decoder.
//! Every observation comes from an exact winning program produced by uniform
//! synthesis over one complete physical tensor stream. A caller-supplied rule
//! model is deliberately ignored so a model can never train itself.

const std = @import("std");
const grammar_prior = @import("grammar_prior.zig");
const safetensors = @import("safetensors.zig");
const synthesizer = @import("synthesizer.zig");
const types = @import("types.zig");

const Allocator = std.mem.Allocator;
const Io = std.Io;

pub const DEFAULT_TENSORS: usize = 32;
const DTYPE_COUNT: usize = std.enums.values(types.Dtype).len;
const SIZE_BUCKET_COUNT: usize = 64;

pub const Options = struct {
    /// Maximum number of complete tensors observed. When a corpus is larger,
    /// deterministic strata cover dtype and physical-size ranges first.
    max_tensors: usize = DEFAULT_TENSORS,
    /// Search limits and grammar bounds used to obtain each exact program.
    /// `rule_model` is always forced to null during calibration.
    synthesis: synthesizer.Options = .{},
    /// Configuration embedded in the returned learned prior.
    prior_config: grammar_prior.Config = .{},
};

pub const Result = struct {
    prior: grammar_prior.Prior,
    observed_tensors: usize,
    expanded: usize,
    completed_candidates: usize,
    budget_exhausted_tensors: usize,
    literal_fallback_tensors: usize,

    pub fn deinit(self: *Result, alloc: Allocator) void {
        self.prior.deinit(alloc);
        self.* = undefined;
    }

    /// Canonical policy bytes for callers that want to persist or inspect the
    /// learned model. No tensor words or semantic programs are archived here.
    pub fn serialize(self: *const Result, alloc: Allocator) ![]u8 {
        return self.prior.serialize(alloc);
    }
};

/// Train a PHOG prior from complete safetensors tensor views.
///
/// Selection is deterministic. If `max_tensors >= tensors.len`, every tensor
/// is observed in input order. Otherwise, the first tensor in every
/// `(dtype, log2 physical bytes)` stratum is considered in a dtype-round-robin
/// order that alternates small and large ranges. Remaining capacity covers
/// source-order endpoints. A zero cap and an empty corpus both produce an
/// empty, serializable prior.
pub fn train(
    alloc: Allocator,
    tensors: []const safetensors.Tensor,
    options: Options,
) !Result {
    try options.prior_config.validate();

    var counts = grammar_prior.Counts.init();
    defer counts.deinit(alloc);

    var uniform_options = options.synthesis;
    uniform_options.rule_model = null;

    const selected = try selectTensorIndices(
        alloc,
        tensors,
        options.max_tensors,
    );
    defer alloc.free(selected);
    var totals: Totals = .{};

    for (selected) |index| {
        const tensor = tensors[index];
        const target = try physicalStream(tensor.view);

        var synthesis = try synthesizer.synthesize(
            alloc,
            target,
            tensor.view.dtype,
            uniform_options,
        );
        defer synthesis.deinit(alloc);

        // `observeProgram` independently validates the complete program
        // against the exact target-directed decomposition before adding rows.
        try counts.observeProgram(
            alloc,
            synthesis.program,
            target,
            tensor.view.dtype,
            1,
        );
        try totals.add(synthesis);
    }

    return finish(alloc, &counts, options.prior_config, selected.len, totals);
}

pub fn trainParallel(
    alloc: Allocator,
    io: Io,
    tensors: []const safetensors.Tensor,
    options: Options,
    workers: usize,
) !Result {
    if (workers == 0) return error.InvalidWorkerCount;
    if (workers == 1 or tensors.len <= 1 or options.max_tensors <= 1)
        return train(alloc, tensors, options);
    try options.prior_config.validate();

    const selected = try selectTensorIndices(
        alloc,
        tensors,
        options.max_tensors,
    );
    defer alloc.free(selected);

    var uniform_options = options.synthesis;
    uniform_options.rule_model = null;
    var counts = grammar_prior.Counts.init();
    defer counts.deinit(alloc);
    var totals: Totals = .{};

    const Completion = union(enum) {
        synthesis: SynthesisOutcome,
    };
    const window = @min(workers, selected.len);
    const buffer = try alloc.alloc(Completion, window);
    defer alloc.free(buffer);
    var select = Io.Select(Completion).init(io, buffer);
    defer while (select.cancel()) |completion_value| {
        var completion = completion_value;
        completion.synthesis.deinit(std.heap.smp_allocator);
    };

    var launched: usize = 0;
    var pending: usize = 0;
    while (launched < window) : (launched += 1) {
        const index = selected[launched];
        select.async(
            .synthesis,
            synthesizeTask,
            .{ index, tensors[index], uniform_options },
        );
        pending += 1;
    }

    while (pending != 0) {
        var completion = try select.await();
        pending -= 1;
        defer completion.synthesis.deinit(std.heap.smp_allocator);
        switch (completion.synthesis) {
            .success => |*success| {
                if (launched < selected.len) {
                    const index = selected[launched];
                    select.async(
                        .synthesis,
                        synthesizeTask,
                        .{ index, tensors[index], uniform_options },
                    );
                    launched += 1;
                    pending += 1;
                }
                const tensor = tensors[success.tensor_index];
                const target = try physicalStream(tensor.view);
                try counts.observeProgram(
                    alloc,
                    success.synthesis.program,
                    target,
                    tensor.view.dtype,
                    1,
                );
                try totals.add(success.synthesis);
            },
            .failure => |err| return err,
        }
    }

    return finish(alloc, &counts, options.prior_config, selected.len, totals);
}

const Totals = struct {
    expanded: usize = 0,
    completed_candidates: usize = 0,
    budget_exhausted_tensors: usize = 0,
    literal_fallback_tensors: usize = 0,

    fn add(self: *Totals, synthesis: synthesizer.Result) !void {
        self.expanded = std.math.add(
            usize,
            self.expanded,
            synthesis.expanded,
        ) catch return error.IntegerOverflow;
        self.completed_candidates = std.math.add(
            usize,
            self.completed_candidates,
            synthesis.completed_candidates,
        ) catch return error.IntegerOverflow;
        self.budget_exhausted_tensors += @intFromBool(
            synthesis.status == .budget_exhausted,
        );
        self.literal_fallback_tensors += @intFromBool(
            synthesis.used_literal_fallback,
        );
    }
};

const SelectedSynthesis = struct {
    tensor_index: usize,
    synthesis: synthesizer.Result,
};

const SynthesisOutcome = union(enum) {
    success: SelectedSynthesis,
    failure: anyerror,

    fn deinit(self: *SynthesisOutcome, alloc: Allocator) void {
        switch (self.*) {
            .success => |*success| success.synthesis.deinit(alloc),
            .failure => {},
        }
    }
};

fn synthesizeTask(
    tensor_index: usize,
    tensor: safetensors.Tensor,
    options: synthesizer.Options,
) SynthesisOutcome {
    const alloc = std.heap.smp_allocator;
    const target = physicalStream(tensor.view) catch |err|
        return .{ .failure = err };
    return .{ .success = .{
        .tensor_index = tensor_index,
        .synthesis = synthesizer.synthesize(
            alloc,
            target,
            tensor.view.dtype,
            options,
        ) catch |err| return .{ .failure = err },
    } };
}

fn finish(
    alloc: Allocator,
    counts: *const grammar_prior.Counts,
    config: grammar_prior.Config,
    observed_tensors: usize,
    totals: Totals,
) !Result {
    return .{
        .prior = try counts.toPrior(alloc, config),
        .observed_tensors = observed_tensors,
        .expanded = totals.expanded,
        .completed_candidates = totals.completed_candidates,
        .budget_exhausted_tensors = totals.budget_exhausted_tensors,
        .literal_fallback_tensors = totals.literal_fallback_tensors,
    };
}

fn physicalStream(view: types.TensorView) !types.Stream {
    const count = try view.numelChecked();
    const expected_bytes = std.math.mul(
        usize,
        count,
        view.dtype.elemSize(),
    ) catch return error.IntegerOverflow;
    if (view.data.len != expected_bytes)
        return error.TensorByteLengthMismatch;
    return .{
        // Synthesis treats this borrowed target as immutable. Keep the
        // const-removal local rather than exposing writable mmap bytes through
        // the public TensorView API.
        .data = @constCast(view.data),
        .count = count,
        .bits_per_elem = view.dtype.bitWidth(),
        .owns_data = false,
    };
}

fn selectTensorIndices(
    alloc: Allocator,
    tensors: []const safetensors.Tensor,
    maximum: usize,
) Allocator.Error![]usize {
    const selected_count = @min(maximum, tensors.len);
    const selected = try alloc.alloc(usize, selected_count);
    if (selected_count == 0) return selected;
    if (selected_count == tensors.len) {
        for (selected, 0..) |*index, value| index.* = value;
        return selected;
    }

    var first_by_stratum: [DTYPE_COUNT * SIZE_BUCKET_COUNT]?usize =
        @splat(null);
    for (tensors, 0..) |tensor, index| {
        const dtype_index: usize = @intFromEnum(tensor.view.dtype);
        const bucket = sizeBucket(tensor.view.data.len);
        const stratum = dtype_index * SIZE_BUCKET_COUNT + bucket;
        if (first_by_stratum[stratum] == null)
            first_by_stratum[stratum] = index;
    }

    var buckets_by_dtype: [DTYPE_COUNT][SIZE_BUCKET_COUNT]u8 = undefined;
    var bucket_counts: [DTYPE_COUNT]usize = @splat(0);
    for (0..DTYPE_COUNT) |dtype_index| {
        for (0..SIZE_BUCKET_COUNT) |bucket| {
            const stratum = dtype_index * SIZE_BUCKET_COUNT + bucket;
            if (first_by_stratum[stratum] == null) continue;
            buckets_by_dtype[dtype_index][bucket_counts[dtype_index]] =
                @intCast(bucket);
            bucket_counts[dtype_index] += 1;
        }
    }

    var written: usize = 0;
    for (0..SIZE_BUCKET_COUNT) |round| {
        for (0..DTYPE_COUNT) |dtype_index| {
            const count = bucket_counts[dtype_index];
            if (round >= count or written == selected_count) continue;
            const half = round / 2;
            const ordinal = if (round % 2 == 0)
                half
            else
                count - 1 - half;
            const bucket = buckets_by_dtype[dtype_index][ordinal];
            selected[written] = first_by_stratum[
                dtype_index * SIZE_BUCKET_COUNT + bucket
            ].?;
            written += 1;
        }
        if (written == selected_count) return selected;
    }

    // More capacity than distinct strata: fill deterministically from the
    // checkpoint's two source-order endpoints inward.
    for (0..tensors.len) |ordinal| {
        if (written == selected_count) break;
        const candidate = if (ordinal % 2 == 0)
            ordinal / 2
        else
            tensors.len - 1 - ordinal / 2;
        if (containsIndex(selected[0..written], candidate)) continue;
        selected[written] = candidate;
        written += 1;
    }
    std.debug.assert(written == selected_count);
    return selected;
}

fn sizeBucket(bytes: usize) usize {
    if (bytes == 0) return 0;
    return @min(
        @as(usize, SIZE_BUCKET_COUNT - 1),
        std.math.log2_int(usize, bytes),
    );
}

fn containsIndex(indices: []const usize, candidate: usize) bool {
    for (indices) |index|
        if (index == candidate) return true;
    return false;
}
