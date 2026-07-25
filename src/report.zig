//! Human-readable and schema-4 JSON reporting for `brevis bench`.
//!
//! Reporting is kept out of main.zig so the CLI owns argument handling and
//! orchestration only. This module never plans, encodes, or decodes; it reads
//! finished plans and results and serializes them.

const std = @import("std");
const types = @import("types.zig");
const ops = @import("ops.zig");
const prior = @import("prior.zig");
const program = @import("program.zig");
const search = @import("search.zig");
const safetensors = @import("safetensors.zig");
const archive = @import("archive.zig");

const Allocator = std.mem.Allocator;
const Dtype = types.Dtype;
const Block = types.Block;

const N_DTYPE: usize = @typeInfo(Dtype).@"enum".fields.len;

/// What the CLI selected, reduced to what a report needs. Keeping the CLI's
/// `PlanMode` enum out of this module leaves the reporting contract stable when
/// plan selection changes.
pub const Mode = struct {
    /// "fixed", "uniform", or "phog".
    name: []const u8,
    /// Whether the run used the search planner rather than the dtype template.
    plan_is_search: bool,
};

pub fn ratio(orig: u64, comp: u64) f64 {
    if (comp == 0) return 0;
    return @as(f64, @floatFromInt(orig)) / @as(f64, @floatFromInt(comp));
}

pub fn writeSearchConfigFields(json: *std.json.Stringify, options: search.Options) !void {
    try json.objectField("transform_layers");
    try json.write(options.max_depth);
    try json.objectField("max_depth");
    try json.write(options.max_depth);
    try json.objectField("max_depth_semantics");
    try json.write("maximum_transform_layers");
    try json.objectField("max_entropy_bpe");
    try json.write(ops.MAX_ENTROPY_BPE);
    try json.objectField("max_nodes");
    try json.write(options.max_nodes);
    try json.objectField("max_nodes_semantics");
    try json.write("transforms_plus_terminals_per_program");
    try json.objectField("max_expansions");
    try json.write(options.max_expansions);
    try json.objectField("max_expansions_scope");
    try json.write("partial_program_pops_per_tensor");
    try json.objectField("max_realizations");
    try json.write(options.max_realizations);
    try json.objectField("max_realizations_scope");
    try json.write("single_stream_search_only");
    try json.objectField("tensor_search_uses_max_realizations");
    try json.write(false);
    try json.objectField("sample_elems");
    try json.write(options.sample_elems);
    try json.objectField("sampling_policy");
    try json.write("single_centered_contiguous_window");
    try json.objectField("target_block_bytes");
    try json.write(types.TARGET_BLOCK_BYTES);
    try json.objectField("rerank_candidates");
    try json.write(options.rerank_candidates);
    try json.objectField("rerank_blocks");
    try json.write(options.rerank_blocks);
    try json.objectField("rerank_enabled");
    try json.write(options.rerank_candidates > 0 and options.rerank_blocks > 0);
    try json.objectField("enabled_ops_mask");
    try json.write(options.enabled_ops);
    try json.objectField("enabled_ops");
    try json.beginArray();
    for (std.enums.values(ops.OpKind)) |op| {
        if (options.enabled_ops & ops.opMask(op) != 0) try json.write(@tagName(op));
    }
    try json.endArray();
    try json.objectField("disabled_ops");
    try json.beginArray();
    for (std.enums.values(ops.OpKind)) |op| {
        if (options.enabled_ops & ops.opMask(op) == 0) try json.write(@tagName(op));
    }
    try json.endArray();
    try json.objectField("uniform_score");
    try json.write(ops.UNIFORM_SCORE);
    try json.objectField("phog_weight");
    try json.write(prior.PHOG_WEIGHT);
    try json.objectField("candidate_collection");
    try json.write("all_within_expansion_budget");
    try json.objectField("sample_byte_pruning");
    try json.write(false);
}

fn renderProgram(alloc: Allocator, out: *std.ArrayList(u8), node: program.Node) Allocator.Error!void {
    try out.appendSlice(alloc, @tagName(node.op));
    if (node.children.len == 0) return;
    try out.append(alloc, '(');
    for (node.children, 0..) |c, i| {
        if (i > 0) try out.append(alloc, ',');
        try renderProgram(alloc, out, c);
    }
    try out.append(alloc, ')');
}

fn writeProgramTree(json: *std.json.Stringify, node: program.Node) !void {
    try json.beginObject();
    try json.objectField("op");
    try json.write(@tagName(node.op));
    try json.objectField("params_u32");
    try json.write(node.params);
    try json.objectField("terminal");
    try json.write(node.op.isTerminal());
    try json.objectField("children");
    try json.beginArray();
    for (node.children) |child| try writeProgramTree(json, child);
    try json.endArray();
    try json.endObject();
}

const DtypeStat = struct {
    blocks: usize = 0,
    raw: u64 = 0,
    comp: u64 = 0,
    shapes: std.StringHashMapUnmanaged(usize) = .empty,
};

const ShapeCount = struct { name: []const u8, n: usize };

fn moreCount(_: void, a: ShapeCount, b: ShapeCount) bool {
    return a.n > b.n;
}

pub fn report(alloc: Allocator, out: *std.Io.Writer, blocks: []const Block, results: []const ?search.Result) !void {
    var arena: std.heap.ArenaAllocator = .init(alloc);
    defer arena.deinit();
    const a = arena.allocator();

    var stats: [N_DTYPE]DtypeStat = @splat(.{});
    var buf: std.ArrayList(u8) = .empty;

    for (blocks, results) |b, maybe| {
        const r = maybe orelse continue;
        const s = &stats[@intFromEnum(b.dtype)];
        s.blocks += 1;
        s.raw += b.byteLen();
        s.comp += r.bytes;

        buf.clearRetainingCapacity();
        try renderProgram(a, &buf, r.node);
        const gop = try s.shapes.getOrPut(a, buf.items);
        if (!gop.found_existing) {
            gop.key_ptr.* = try a.dupe(u8, buf.items);
            gop.value_ptr.* = 0;
        }
        gop.value_ptr.* += 1;
    }

    var total_raw: u64 = 0;
    var total_comp: u64 = 0;
    for (0..N_DTYPE) |di| {
        const s = stats[di];
        if (s.blocks == 0) continue;
        total_raw += s.raw;
        total_comp += s.comp;

        const dt: Dtype = @enumFromInt(di);
        try out.print("\n{s}: {d} blocks, {d} -> {d} bytes ({d:.3}x)\n", .{
            dt.name(), s.blocks, s.raw, s.comp, ratio(s.raw, s.comp),
        });

        var top: std.ArrayList(ShapeCount) = .empty;
        defer top.deinit(a);
        var it = s.shapes.iterator();
        while (it.next()) |e| try top.append(a, .{ .name = e.key_ptr.*, .n = e.value_ptr.* });
        std.mem.sort(ShapeCount, top.items, {}, moreCount);

        for (top.items[0..@min(10, top.items.len)]) |sc| {
            try out.print("  {d:>6}  {s}\n", .{ sc.n, sc.name });
        }
        try out.flush();
    }

    try out.print("\noverall: {d} -> {d} bytes ({d:.3}x)\n", .{ total_raw, total_comp, ratio(total_raw, total_comp) });
}

fn programNodes(node: program.Node) usize {
    var n: usize = 1;
    for (node.children) |child| n += programNodes(child);
    return n;
}

fn programDepth(node: program.Node) usize {
    var depth: usize = 1;
    for (node.children) |child| depth = @max(depth, 1 + programDepth(child));
    return depth;
}

fn programTransformDepth(node: program.Node) usize {
    if (node.op.isTerminal()) return 0;
    var depth: usize = 1;
    for (node.children) |child| depth = @max(depth, 1 + programTransformDepth(child));
    return depth;
}

fn programTerminals(node: program.Node) usize {
    if (node.op.isTerminal()) return 1;
    var n: usize = 0;
    for (node.children) |child| n += programTerminals(child);
    return n;
}


const PROGRAM_SEQUENCE_SPEC_ID = "brevis.program-bytecode-sequence.v1";

const ProgramBytecodeEvidence = struct {
    lengths: []u32,
    sha256_by_block: [][64]u8,
    sequence_sha256: [64]u8,
};

fn collectProgramBytecodeEvidence(
    alloc: Allocator,
    results: []const ?search.Result,
) !ProgramBytecodeEvidence {
    const lengths = try alloc.alloc(u32, results.len);
    const digests = try alloc.alloc([64]u8, results.len);
    var sequence = std.crypto.hash.sha2.Sha256.init(.{});
    sequence.update(PROGRAM_SEQUENCE_SPEC_ID);
    sequence.update(&.{0});
    var count_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &count_bytes, @intCast(results.len), .little);
    sequence.update(&count_bytes);

    for (results, 0..) |maybe, index| {
        const result = maybe orelse return error.SynthesisFailed;
        const bytecode = try program.serialize(alloc, result.node);
        defer alloc.free(bytecode);
        if (bytecode.len != result.bytes - result.payload.len) return error.InvalidProgram;
        lengths[index] = std.math.cast(u32, bytecode.len) orelse return error.Overflow;
        var raw_digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
        var block_hash = std.crypto.hash.sha2.Sha256.init(.{});
        block_hash.update(bytecode);
        block_hash.final(&raw_digest);
        digests[index] = std.fmt.bytesToHex(raw_digest, .lower);

        var length_bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &length_bytes, lengths[index], .little);
        sequence.update(&length_bytes);
        sequence.update(&raw_digest);
    }
    var sequence_digest: [std.crypto.hash.sha2.Sha256.digest_length]u8 = undefined;
    sequence.final(&sequence_digest);
    return .{
        .lengths = lengths,
        .sha256_by_block = digests,
        .sequence_sha256 = std.fmt.bytesToHex(sequence_digest, .lower),
    };
}

pub fn reportJson(
    alloc: Allocator,
    out: *std.Io.Writer,
    in_path: []const u8,
    input_bytes: []const u8,
    input_size_bytes: usize,
    input_sha256: []const u8,
    tensors: []const safetensors.Tensor,
    blocks: []const Block,
    plans: []const ?search.Plan,
    results: []const ?search.Result,
    mode: Mode,
    prior_path: ?[]const u8,
    prior_sha256: ?[]const u8,
    learned_prior: bool,
    prior_counts: [3]usize,
    n_threads: usize,
    search_options: search.Options,
    planning_ms: i64,
    encoding_ms: i64,
) !void {
    var arena: std.heap.ArenaAllocator = .init(alloc);
    defer arena.deinit();
    const a = arena.allocator();
    var buf: std.ArrayList(u8) = .empty;
    const program_evidence = try collectProgramBytecodeEvidence(a, results);

    var json: std.json.Stringify = .{
        .writer = out,
        .options = .{ .whitespace = .indent_2 },
    };
    try json.beginObject();
    try json.objectField("schema");
    try json.write(4);
    try json.objectField("kind");
    try json.write("brevis.bench-report");
    try json.objectField("input");
    try json.write(in_path);
    try json.objectField("input_size_bytes");
    try json.write(input_size_bytes);
    try json.objectField("input_sha256");
    try json.write(input_sha256);
    try json.objectField("timing_scope");
    try json.write("planning_and_block_encoding_only; input/prior hashing and JSON/program-evidence serialization excluded");
    try json.objectField("cache_preconditioning");
    try json.write("full input SHA-256 scan completed before planning");
    try json.objectField("mode");
    try json.write(mode.name);
    try json.objectField("prior");
    try json.beginObject();
    try json.objectField("supplied");
    try json.write(prior_path != null);
    try json.objectField("loaded");
    try json.write(mode.plan_is_search and prior_path != null);
    try json.objectField("applied");
    try json.write(learned_prior);
    try json.objectField("guidance_active");
    try json.write(learned_prior);
    try json.objectField("path");
    if (prior_path) |path| try json.write(path) else try json.write(null);
    try json.objectField("sha256");
    if (prior_sha256) |digest| try json.write(digest) else try json.write(null);
    try json.objectField("nonempty");
    try json.write(learned_prior);
    try json.objectField("context_counts_by_backoff_level");
    try json.write(prior_counts);
    try json.endObject();
    try json.objectField("threads");
    try json.write(n_threads);
    try json.objectField("requested_threads");
    try json.write(n_threads);
    var nonempty_tensors: usize = 0;
    for (tensors) |tensor| nonempty_tensors += @intFromBool(tensor.view.numel() > 0);
    try json.objectField("planning_workers_used");
    try json.write(if (nonempty_tensors == 0) 0 else @min(n_threads, nonempty_tensors));
    try json.objectField("encoding_workers_used");
    try json.write(if (blocks.len == 0) 0 else @min(n_threads, blocks.len));
    try json.objectField("target_block_bytes");
    try json.write(types.TARGET_BLOCK_BYTES);
    try json.objectField("search_options_applied");
    try json.write(mode.plan_is_search);
    try json.objectField("search");
    try json.beginObject();
    try writeSearchConfigFields(&json, search_options);
    try json.endObject();
    try json.objectField("planning_wall_ms");
    try json.write(planning_ms);
    try json.objectField("encoding_wall_ms");
    try json.write(encoding_ms);

    var total_raw: u64 = 0;
    var total_encoded: u64 = 0;
    for (blocks, results) |block, maybe| {
        total_raw += block.byteLen();
        if (maybe) |result| total_encoded += result.bytes;
    }
    const block_frame_bytes = total_encoded + @as(u64, @intCast(blocks.len)) * 12;
    const index_offset = @as(u64, archive.HEADER.len) + block_frame_bytes;
    const metas = try archive.tensorMetas(a, tensors, blocks);
    const safetensors_header_len: usize = @intCast(std.mem.readInt(u64, input_bytes[0..8], .little));
    const safetensors_prefix_bytes = 8 + safetensors_header_len;
    const footer = try archive.makeFooter(a, metas, index_offset, input_bytes[0..safetensors_prefix_bytes]);
    try json.objectField("raw_bytes");
    try json.write(total_raw);
    try json.objectField("tensor_data_bytes");
    try json.write(total_raw);
    try json.objectField("safetensors_prefix_bytes");
    try json.write(safetensors_prefix_bytes);
    try json.objectField("encoded_bytes_without_frame_headers");
    try json.write(total_encoded);
    try json.objectField("block_frame_bytes_excluding_container_header_footer");
    try json.write(block_frame_bytes);
    try json.objectField("container_header_bytes");
    try json.write(archive.HEADER.len);
    try json.objectField("container_footer_bytes");
    try json.write(footer.len);
    try json.objectField("projected_archive_bytes");
    try json.write(index_offset + footer.len);
    try json.objectField("size_accounting");
    try json.write("raw_bytes and tensor_data_bytes exclude the safetensors prefix, while input_size_bytes includes it; packed terminal payloads include an 8-byte length per terminal; each block frame adds a 4-byte bytecode length and 8-byte packed-payload length; projected_archive_bytes adds the exact .brv container header and footer for this diagnostic replay, but an actual .brv file remains the effectiveness measurement");
    try json.objectField("raw_block_classification");
    try json.write("planned_raw means the selected tensor plan has a raw root; fallback_raw means a non-raw tensor plan produced a raw-root block because it did not beat raw or could not be applied; raw terminals below a transform root are not classified as raw-root blocks");
    try json.objectField("program_tree_semantics");
    try json.write("tensor program_tree is the selected planning template and its parameters come from planning; block program_tree is the realized archive program after per-block parameter refitting");
    try json.objectField("program_bytecode_evidence");
    try json.beginObject();
    try json.objectField("version");
    try json.write(1);
    try json.objectField("hash");
    try json.write("sha256");
    try json.objectField("sequence_spec_id");
    try json.write(PROGRAM_SEQUENCE_SPEC_ID);
    try json.objectField("block_count");
    try json.write(blocks.len);
    try json.objectField("sequence_sha256");
    try json.write(program_evidence.sequence_sha256[0..]);
    try json.endObject();

    try json.objectField("tensors");
    try json.beginArray();
    var block_index: usize = 0;
    for (tensors, 0..) |tensor, tensor_index| {
        const first_block = block_index;
        var encoded: u64 = 0;
        var framed: u64 = 0;
        var raw_root_blocks: usize = 0;
        var planned_raw_root_blocks: usize = 0;
        var fallback_raw_root_blocks: usize = 0;
        const plan_is_raw = if (plans[tensor_index]) |plan| plan.root.op == .raw else false;
        while (block_index < blocks.len and blocks[block_index].tensor_idx == tensor_index) : (block_index += 1) {
            if (results[block_index]) |result| {
                encoded += result.bytes;
                framed += result.bytes + 12;
                if (result.node.op == .raw) {
                    raw_root_blocks += 1;
                    if (plan_is_raw) planned_raw_root_blocks += 1 else fallback_raw_root_blocks += 1;
                }
            }
        }

        try json.beginObject();
        try json.objectField("index");
        try json.write(tensor_index);
        try json.objectField("name");
        try json.write(tensor.name);
        try json.objectField("dtype");
        try json.write(tensor.view.dtype.name());
        try json.objectField("shape");
        try json.write(tensor.view.shape);
        try json.objectField("numel");
        try json.write(tensor.view.numel());
        try json.objectField("raw_bytes");
        try json.write(tensor.view.data.len);
        const input_start = @intFromPtr(input_bytes.ptr);
        const input_end = input_start + input_bytes.len;
        const tensor_start = @intFromPtr(tensor.view.data.ptr);
        if (tensor_start < input_start or tensor_start > input_end or
            tensor.view.data.len > input_end - tensor_start) return error.TensorOutsideInput;
        const file_data_start = tensor_start - input_start;
        try json.objectField("file_data_start_byte");
        try json.write(file_data_start);
        try json.objectField("file_data_end_byte_exclusive");
        try json.write(file_data_start + tensor.view.data.len);
        try json.objectField("encoded_bytes_without_frame_headers");
        try json.write(encoded);
        try json.objectField("block_frame_bytes_excluding_container_header_footer");
        try json.write(framed);
        try json.objectField("block_start");
        try json.write(first_block);
        try json.objectField("block_count");
        try json.write(block_index - first_block);
        try json.objectField("raw_root_blocks");
        try json.write(raw_root_blocks);
        try json.objectField("planned_raw_root_blocks");
        try json.write(planned_raw_root_blocks);
        try json.objectField("fallback_raw_root_blocks");
        try json.write(fallback_raw_root_blocks);
        if (plans[tensor_index]) |plan| {
            buf.clearRetainingCapacity();
            try renderProgram(a, &buf, plan.root);
            try json.objectField("search_expansions");
            try json.write(plan.expanded);
            try json.objectField("candidates_realized");
            if (mode.plan_is_search) try json.write(plan.candidates_realized) else try json.write(null);
            try json.objectField("candidates_reranked");
            if (mode.plan_is_search) try json.write(plan.candidates_reranked) else try json.write(null);
            try json.objectField("probe_blocks_used");
            if (mode.plan_is_search) try json.write(plan.probe_blocks_used) else try json.write(null);
            try json.objectField("selected_sample_rank_zero_based");
            if (mode.plan_is_search) try json.write(plan.selected_sample_rank) else try json.write(null);
            try json.objectField("program");
            try json.write(buf.items);
            try json.objectField("root_operator");
            try json.write(@tagName(plan.root.op));
            try json.objectField("program_tree");
            try writeProgramTree(&json, plan.root);
            try json.objectField("program_nodes");
            try json.write(programNodes(plan.root));
            try json.objectField("program_depth");
            try json.write(programDepth(plan.root));
            try json.objectField("program_node_depth");
            try json.write(programDepth(plan.root));
            try json.objectField("program_transform_depth");
            try json.write(programTransformDepth(plan.root));
            try json.objectField("terminal_count");
            try json.write(programTerminals(plan.root));
        } else {
            try json.objectField("search_expansions");
            try json.write(null);
            try json.objectField("candidates_realized");
            try json.write(null);
            try json.objectField("candidates_reranked");
            try json.write(null);
            try json.objectField("probe_blocks_used");
            try json.write(null);
            try json.objectField("selected_sample_rank_zero_based");
            try json.write(null);
            try json.objectField("program");
            try json.write(null);
            try json.objectField("root_operator");
            try json.write(null);
            try json.objectField("program_tree");
            try json.write(null);
            try json.objectField("program_nodes");
            try json.write(null);
            try json.objectField("program_depth");
            try json.write(null);
            try json.objectField("program_node_depth");
            try json.write(null);
            try json.objectField("program_transform_depth");
            try json.write(null);
            try json.objectField("terminal_count");
            try json.write(null);
        }
        try json.endObject();
    }
    try json.endArray();

    try json.objectField("blocks");
    try json.beginArray();
    for (blocks, results, 0..) |block, maybe, index| {
        const result = maybe orelse continue;
        buf.clearRetainingCapacity();
        try renderProgram(a, &buf, result.node);
        try json.beginObject();
        try json.objectField("index");
        try json.write(index);
        try json.objectField("tensor_index");
        try json.write(block.tensor_idx);
        try json.objectField("element_offset");
        try json.write(block.elem_offset);
        try json.objectField("element_count");
        try json.write(block.elem_count);
        try json.objectField("raw_bytes");
        try json.write(block.byteLen());
        try json.objectField("encoded_bytes_without_frame_headers");
        try json.write(result.bytes);
        try json.objectField("program_bytecode_bytes");
        try json.write(result.bytes - result.payload.len);
        try json.objectField("program_bytecode_sha256");
        try json.write(program_evidence.sha256_by_block[index][0..]);
        try json.objectField("packed_terminal_payload_bytes");
        try json.write(result.payload.len);
        try json.objectField("frame_header_bytes");
        try json.write(result.bytes - result.payload.len + 12);
        try json.objectField("framed_bytes");
        try json.write(result.bytes + 12);
        const block_plan_is_raw = if (plans[block.tensor_idx]) |plan| plan.root.op == .raw else false;
        try json.objectField("raw_classification");
        if (result.node.op != .raw)
            try json.write(null)
        else if (block_plan_is_raw)
            try json.write("planned_raw")
        else
            try json.write("fallback_raw");
        try json.objectField("program");
        try json.write(buf.items);
        try json.objectField("root_operator");
        try json.write(@tagName(result.node.op));
        try json.objectField("program_tree");
        try writeProgramTree(&json, result.node);
        try json.objectField("program_nodes");
        try json.write(programNodes(result.node));
        try json.objectField("program_depth");
        try json.write(programDepth(result.node));
        try json.objectField("program_node_depth");
        try json.write(programDepth(result.node));
        try json.objectField("program_transform_depth");
        try json.write(programTransformDepth(result.node));
        try json.objectField("terminal_count");
        try json.write(programTerminals(result.node));
        try json.endObject();
    }
    try json.endArray();
    try json.endObject();
    try out.writeByte('\n');
}
