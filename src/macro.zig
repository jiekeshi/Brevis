//! Learned macros: named subtrees of existing reversible operators that the
//! search may apply as a single production.
//!
//! A macro carries no data. Its body is a tree of `ops.OpKind` nodes whose
//! leaves are either holes (the search continues there) or terminals (the
//! branch is closed). Because every operator in the body is already reversible
//! and every byte still enters through a terminal payload, a macro cannot hide
//! payload inside the grammar and cannot change what is representable — it
//! changes only what the search reaches within its expansion budget.
//!
//! Macros are expanded before a program is serialized, so an archive written
//! with a library is byte-identical to one written without it whenever both
//! select the same program, and the decoder never learns that macros exist.
//! That is what keeps the archive schema and the decode path unchanged.

const std = @import("std");
const ops = @import("ops.zig");

const Allocator = std.mem.Allocator;

pub const SCHEMA = "brevis.macro-library.v1";

/// Cap on library size. Every macro is offered at every hole, so the library
/// spends the caller's expansion budget; this bounds that dilution.
pub const MAX_MACROS: usize = 64;
pub const MAX_BODY_NODES: usize = 16;

pub const ParseError = error{
    BadSchema,
    BadMacroField,
    DuplicateMacroName,
    TooManyMacros,
    BodyTooLarge,
    UnknownOperator,
    VariableArityOperator,
    WrongChildCount,
    TerminalHasChildren,
    NoHolesOrTerminals,
} || Allocator.Error || std.json.ParseError(std.json.Scanner);

/// How a body node's `params` word is chosen at instantiation.
pub const ParamSpec = union(enum) {
    /// Derive from the stream reaching this node, exactly as the search does
    /// for a primitive production. This is what lets one macro apply across
    /// dtypes and widths.
    auto,
    literal: u32,
};

pub const Body = union(enum) {
    /// The search fills this position; it is a normal hole with normal guards.
    hole,
    node: Node,
};

pub const Node = struct {
    op: ops.OpKind,
    params: ParamSpec,
    /// Exactly `ops.arity(op, _)` entries; empty for terminals.
    children: []const Body,
};

pub const Macro = struct {
    name: []const u8,
    body: Node,
    /// Operator nodes in the body, holes excluded. Charged against `max_nodes`.
    node_count: usize,
    /// Holes the body leaves open.
    hole_count: usize,
    /// Transform layers the body itself occupies. Charged against `max_depth`,
    /// so `--max-depth` keeps meaning "transform layers" after expansion.
    transform_depth: u8,
    /// Every operator the body uses. A macro is offered only when the caller
    /// has all of them enabled, so an ablation still removes what it names.
    ops_mask: u64,
};

/// Owns the storage for a parsed library. `macros` stays valid until `deinit`.
pub const Library = struct {
    arena: std.heap.ArenaAllocator,
    macros: []const Macro,

    pub fn deinit(self: *Library) void {
        self.arena.deinit();
    }
};

/// Width-dependent arity would make a body's shape depend on the stream it
/// meets, so bodies are restricted to operators whose arity is fixed.
pub fn hasFixedArity(op: ops.OpKind) bool {
    return switch (op) {
        .bit_plane, .byte_plane => false,
        else => true,
    };
}

fn arityOf(op: ops.OpKind) usize {
    std.debug.assert(hasFixedArity(op));
    // Any width answers for a fixed-arity operator.
    return ops.arity(op, 32);
}

// ==================== parsing ====================

pub fn parse(alloc: Allocator, bytes: []const u8) ParseError!Library {
    var parsed = try std.json.parseFromSlice(std.json.Value, alloc, bytes, .{});
    defer parsed.deinit();

    var arena = std.heap.ArenaAllocator.init(alloc);
    errdefer arena.deinit();
    const a = arena.allocator();

    const root = switch (parsed.value) {
        .object => |o| o,
        else => return error.BadMacroField,
    };
    const schema = root.get("schema") orelse return error.BadSchema;
    switch (schema) {
        .string => |s| if (!std.mem.eql(u8, s, SCHEMA)) return error.BadSchema,
        else => return error.BadSchema,
    }

    const listed = switch (root.get("macros") orelse return error.BadMacroField) {
        .array => |arr| arr,
        else => return error.BadMacroField,
    };
    if (listed.items.len > MAX_MACROS) return error.TooManyMacros;

    const macros = try a.alloc(Macro, listed.items.len);
    for (listed.items, 0..) |entry, i| {
        macros[i] = try parseMacro(a, entry);
        for (macros[0..i]) |earlier| {
            if (std.mem.eql(u8, earlier.name, macros[i].name)) return error.DuplicateMacroName;
        }
    }

    return .{ .arena = arena, .macros = macros };
}

fn parseMacro(a: Allocator, entry: std.json.Value) ParseError!Macro {
    const object = switch (entry) {
        .object => |o| o,
        else => return error.BadMacroField,
    };
    const name = switch (object.get("name") orelse return error.BadMacroField) {
        .string => |s| try a.dupe(u8, s),
        else => return error.BadMacroField,
    };
    const body_value = object.get("body") orelse return error.BadMacroField;
    const body = switch (try parseBody(a, body_value)) {
        .hole => return error.BadMacroField, // a bare hole is the identity
        .node => |n| n,
    };

    const stats = bodyStats(.{ .node = body });
    if (stats.nodes > MAX_BODY_NODES) return error.BodyTooLarge;
    // A body of transforms with no holes and no terminals cannot be completed.
    if (stats.holes == 0 and stats.terminals == 0) return error.NoHolesOrTerminals;

    return .{
        .name = name,
        .body = body,
        .node_count = stats.nodes,
        .hole_count = stats.holes,
        .transform_depth = stats.transform_depth,
        .ops_mask = stats.ops_mask,
    };
}

fn parseBody(a: Allocator, value: std.json.Value) ParseError!Body {
    const object = switch (value) {
        .object => |o| o,
        else => return error.BadMacroField,
    };
    const op_name = switch (object.get("op") orelse return error.BadMacroField) {
        .string => |s| s,
        else => return error.BadMacroField,
    };
    if (std.mem.eql(u8, op_name, "hole")) return .hole;

    const op = std.meta.stringToEnum(ops.OpKind, op_name) orelse return error.UnknownOperator;
    if (!hasFixedArity(op)) return error.VariableArityOperator;

    const params: ParamSpec = switch (object.get("params") orelse std.json.Value{ .string = "auto" }) {
        .string => |s| if (std.mem.eql(u8, s, "auto")) .auto else return error.BadMacroField,
        .integer => |v| if (v < 0 or v > std.math.maxInt(u32))
            return error.BadMacroField
        else
            .{ .literal = @intCast(v) },
        else => return error.BadMacroField,
    };

    const want = arityOf(op);
    const listed = switch (object.get("children") orelse std.json.Value{ .array = .init(a) }) {
        .array => |arr| arr,
        else => return error.BadMacroField,
    };
    if (op.isTerminal() and listed.items.len != 0) return error.TerminalHasChildren;
    if (listed.items.len != want) return error.WrongChildCount;

    const children = try a.alloc(Body, want);
    for (listed.items, 0..) |child, i| children[i] = try parseBody(a, child);

    return .{ .node = .{ .op = op, .params = params, .children = children } };
}

const Stats = struct {
    nodes: usize = 0,
    holes: usize = 0,
    terminals: usize = 0,
    transform_depth: u8 = 0,
    ops_mask: u64 = 0,
};

fn bodyStats(body: Body) Stats {
    const node = switch (body) {
        .hole => return .{ .holes = 1 },
        .node => |n| n,
    };
    var stats: Stats = .{ .nodes = 1, .ops_mask = ops.opMask(node.op) };
    if (node.op.isTerminal()) {
        stats.terminals = 1;
        return stats;
    }
    var deepest: u8 = 0;
    for (node.children) |child| {
        const sub = bodyStats(child);
        stats.nodes += sub.nodes;
        stats.holes += sub.holes;
        stats.terminals += sub.terminals;
        stats.ops_mask |= sub.ops_mask;
        deepest = @max(deepest, sub.transform_depth);
    }
    stats.transform_depth = deepest + 1;
    return stats;
}

// ==================== loading ====================

pub fn load(alloc: Allocator, io: std.Io, path: []const u8) !Library {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(1 << 24));
    defer alloc.free(bytes);
    return parse(alloc, bytes);
}

// ==================== rendering ====================

/// `name(child,child)` in the same shape `report.renderProgram` uses, so a
/// library and a measured program can be read side by side.
pub fn render(alloc: Allocator, out: *std.ArrayList(u8), body: Body) Allocator.Error!void {
    const node = switch (body) {
        .hole => return out.appendSlice(alloc, "?"),
        .node => |n| n,
    };
    try out.appendSlice(alloc, @tagName(node.op));
    if (node.children.len == 0) return;
    try out.append(alloc, '(');
    for (node.children, 0..) |child, i| {
        if (i > 0) try out.append(alloc, ',');
        try render(alloc, out, child);
    }
    try out.append(alloc, ')');
}
