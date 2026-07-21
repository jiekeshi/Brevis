//! PHOG-style prior over DSL productions. Each expansion site is summarised by
//! a `Context` (tree position + bucketed statistics of the stream being
//! expanded); scores are conditional on the productions legal at that hole and
//! measured as -log2(p) in 1/1024 bits.

const std = @import("std");
const types = @import("types.zig");
const ops = @import("ops.zig");

const Allocator = std.mem.Allocator;
const Stream = types.Stream;
const Dtype = types.Dtype;

pub const N_PROD: usize = 64;

const SAMPLE_MAX: usize = ops.SEARCH_SAMPLE_ELEMS;
const HIST_SIZE: usize = 1 << 12;
const PHOG_WEIGHT: f64 = 1.0 / 20.0;
const MAGIC = "BRVP";
const VERSION: u32 = 2;

pub const Context = struct {
    slot: u8 = 0,
    depth: u8 = 0,
    parent_op: u8 = 0,
    dtype: u8 = 0,
    bpe_bucket: u8 = 0,
    entropy_bucket: u8 = 0,
    zero_bucket: u8 = 0,
    delta_bucket: u8 = 0,

    pub fn fromStream(s: Stream, dtype: Dtype, slot: u8, depth: u8, parent_op: u8) Context {
        var ctx: Context = .{
            .slot = slot,
            .depth = depth,
            .parent_op = parent_op,
            .dtype = @intFromEnum(dtype),
            .bpe_bucket = bpeBucket(s.bits_per_elem),
        };
        if (s.count == 0) return ctx;

        const n = @min(s.count, SAMPLE_MAX);
        const m = s.mask();

        var values: [HIST_SIZE]u16 = @splat(0);
        var deltas: [HIST_SIZE]u16 = @splat(0);
        var zeros: usize = 0;
        var prev = s.getU32(0) & m;
        for (0..n) |i| {
            const v = s.getU32(i) & m;
            if (v == 0) zeros += 1;
            values[fold(v)] += 1;
            deltas[fold(if (i == 0) v else (v -% prev) & m)] += 1;
            prev = v;
        }
        const h0 = entropy(&values, n);
        const h_delta = entropy(&deltas, n);

        const bpe: f64 = @floatFromInt(s.bits_per_elem);
        const zero_frac = @as(f64, @floatFromInt(zeros)) / @as(f64, @floatFromInt(n));
        const delta_ratio = if (h0 > 0) h_delta / h0 else 1.0;

        ctx.entropy_bucket = @intFromFloat(std.math.clamp(h0 / bpe * 8.0, 0.0, 7.0));
        ctx.zero_bucket = @intFromFloat(std.math.clamp(zero_frac * 4.0, 0.0, 3.0));
        ctx.delta_bucket = @intFromFloat(std.math.clamp(delta_ratio * 2.0, 0.0, 3.0));
        return ctx;
    }

    /// Level 0 uses every field, level 1 drops the statistics, level 2 keeps
    /// only the tree position.
    pub fn hash(self: Context, level: u2) u64 {
        var b: [9]u8 = @splat(0);
        b[0] = level;
        b[1] = self.slot;
        b[2] = self.parent_op;
        if (level < 2) {
            b[3] = self.depth;
            b[4] = self.dtype;
            b[5] = self.bpe_bucket;
        }
        if (level == 0) {
            b[6] = self.entropy_bucket;
            b[7] = self.zero_bucket;
            b[8] = self.delta_bucket;
        }
        return std.hash.XxHash64.hash(0, &b);
    }
};

fn fold(v: u32) usize {
    return (v ^ (v >> 12) ^ (v >> 24)) & (HIST_SIZE - 1);
}

fn bpeBucket(bpe: u8) u8 {
    if (bpe <= 1) return 0;
    if (bpe <= 8) return 1;
    if (bpe <= 16) return 2;
    return 3;
}

fn entropy(hist: []const u16, n: usize) f64 {
    const total: f64 = @floatFromInt(n);
    var h: f64 = 0;
    for (hist) |c| {
        if (c == 0) continue;
        const p = @as(f64, @floatFromInt(c)) / total;
        h -= p * std.math.log2(p);
    }
    return h;
}

pub const Prior = struct {
    levels: [3]std.AutoHashMapUnmanaged(u64, [N_PROD]u32),

    /// No observations; `scoreSet` then returns a uniform conditional prior.
    pub const empty: Prior = .{ .levels = .{ .empty, .empty, .empty } };

    pub fn deinit(self: *Prior, alloc: Allocator) void {
        for (&self.levels) |*m| m.deinit(alloc);
    }

    pub fn isEmpty(self: Prior) bool {
        for (self.levels) |level| if (level.count() != 0) return false;
        return true;
    }

    pub fn scoreLowerBound(self: Prior, production_count: usize) u32 {
        std.debug.assert(production_count > 0);
        const n: f64 = @floatFromInt(production_count);
        const probability = if (self.isEmpty()) 1.0 / n else PHOG_WEIGHT + (1.0 - PHOG_WEIGHT) / n;
        return @intFromFloat(@round(-std.math.log2(probability) * 1024.0));
    }

    fn findRow(self: Prior, ctx: Context) ?*const [N_PROD]u32 {
        for (0..3) |level| {
            if (self.levels[level].getPtr(ctx.hash(@intCast(level)))) |scores| return scores;
        }
        return null;
    }

    pub fn scoreSet(self: Prior, ctx: Context, productions: []const ops.OpKind, out: []u32) void {
        std.debug.assert(productions.len == out.len and productions.len > 0);
        const learned_row = self.findRow(ctx);
        var floor: u32 = std.math.maxInt(u32);
        for (productions, out) |op, *stored| {
            stored.* = if (learned_row) |row_scores| row_scores[@intFromEnum(op)] else ops.UNIFORM_SCORE;
            floor = @min(floor, stored.*);
        }

        var sum: f64 = 0;
        for (out) |stored| {
            const delta: f64 = @floatFromInt(stored - floor);
            sum += std.math.exp2(-delta / 1024.0);
        }
        for (out) |*dst| {
            const delta: f64 = @floatFromInt(dst.* - floor);
            const learned = std.math.exp2(-delta / 1024.0) / sum;
            const probability = PHOG_WEIGHT * learned + (1.0 - PHOG_WEIGHT) / @as(f64, @floatFromInt(productions.len));
            dst.* = @intFromFloat(@round(-std.math.log2(probability) * 1024.0));
        }
    }

    pub fn save(self: Prior, alloc: Allocator, path: []const u8) !void {
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(alloc);

        try out.appendSlice(alloc, MAGIC);
        try wU32(alloc, &out, VERSION);
        for (0..3) |lv| {
            try wU32(alloc, &out, @intCast(self.levels[lv].count()));
            var it = self.levels[lv].iterator();
            while (it.next()) |e| {
                try wU64(alloc, &out, e.key_ptr.*);
                for (e.value_ptr.*) |s| try wU32(alloc, &out, s);
            }
        }

        var threaded: std.Io.Threaded = .init(alloc, .{});
        defer threaded.deinit();
        const io = threaded.io();
        const f = try std.Io.Dir.cwd().createFile(io, path, .{});
        defer f.close(io);
        var wb: [4096]u8 = undefined;
        var w = f.writer(io, &wb);
        try w.interface.writeAll(out.items);
        try w.interface.flush();
    }

    pub fn load(alloc: Allocator, path: []const u8) !Prior {
        var threaded: std.Io.Threaded = .init(alloc, .{});
        defer threaded.deinit();
        const io = threaded.io();
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, alloc, .limited(1 << 30));
        defer alloc.free(bytes);

        var r: Reader = .{ .b = bytes };
        if (!std.mem.eql(u8, try r.take(4), MAGIC)) return error.BadMagic;
        if ((try r.u32v()) != VERSION) return error.BadVersion;

        var p: Prior = .empty;
        errdefer p.deinit(alloc);
        for (0..3) |lv| {
            const n = try r.u32v();
            if (@as(u64, n) * (8 + N_PROD * 4) > r.b.len - r.pos) return error.Truncated;
            try p.levels[lv].ensureTotalCapacity(alloc, n);
            for (0..n) |_| {
                const key = try r.u64v();
                var row: [N_PROD]u32 = undefined;
                for (&row) |*s| s.* = try r.u32v();
                p.levels[lv].putAssumeCapacity(key, row);
            }
        }
        return p;
    }
};

pub const Counts = struct {
    levels: [3]std.AutoHashMapUnmanaged(u64, [N_PROD]f64),

    pub fn init(alloc: Allocator) Counts {
        _ = alloc;
        return .{ .levels = .{ .empty, .empty, .empty } };
    }

    pub fn deinit(self: *Counts, alloc: Allocator) void {
        for (&self.levels) |*m| m.deinit(alloc);
    }

    pub fn add(self: *Counts, alloc: Allocator, ctx: Context, op: ops.OpKind, w: f64) !void {
        const idx: usize = @intFromEnum(op);
        std.debug.assert(idx < N_PROD);
        for (0..3) |lv| {
            const gop = try self.levels[lv].getOrPut(alloc, ctx.hash(@intCast(lv)));
            if (!gop.found_existing) gop.value_ptr.* = @splat(0);
            gop.value_ptr[idx] += w;
        }
    }

    /// Laplace-smoothed (α=1) production probabilities as -log2(p) * 1024.
    pub fn toPrior(self: Counts, alloc: Allocator) !Prior {
        var p: Prior = .empty;
        errdefer p.deinit(alloc);
        for (0..3) |lv| {
            try p.levels[lv].ensureTotalCapacity(alloc, self.levels[lv].count());
            var it = self.levels[lv].iterator();
            while (it.next()) |e| {
                const productions = std.enums.values(ops.OpKind);
                var total: f64 = @floatFromInt(productions.len);
                for (productions) |op| total += e.value_ptr.*[@intFromEnum(op)];
                var row: [N_PROD]u32 = @splat(ops.UNIFORM_SCORE);
                for (productions) |op| {
                    const idx: usize = @intFromEnum(op);
                    const c = e.value_ptr.*[idx];
                    const prob = (c + 1.0) / total;
                    row[idx] = @intFromFloat(@round(@min(-std.math.log2(prob) * 1024.0, 1.0e9)));
                }
                p.levels[lv].putAssumeCapacity(e.key_ptr.*, row);
            }
        }
        return p;
    }
};

fn wU32(alloc: Allocator, out: *std.ArrayList(u8), v: u32) Allocator.Error!void {
    var b: [4]u8 = undefined;
    std.mem.writeInt(u32, &b, v, .little);
    try out.appendSlice(alloc, &b);
}

fn wU64(alloc: Allocator, out: *std.ArrayList(u8), v: u64) Allocator.Error!void {
    var b: [8]u8 = undefined;
    std.mem.writeInt(u64, &b, v, .little);
    try out.appendSlice(alloc, &b);
}

const Reader = struct {
    b: []const u8,
    pos: usize = 0,

    fn take(self: *Reader, n: usize) error{Truncated}![]const u8 {
        if (n > self.b.len - self.pos) return error.Truncated;
        defer self.pos += n;
        return self.b[self.pos..][0..n];
    }
    fn u32v(self: *Reader) error{Truncated}!u32 {
        return std.mem.readInt(u32, (try self.take(4))[0..4], .little);
    }
    fn u64v(self: *Reader) error{Truncated}!u64 {
        return std.mem.readInt(u64, (try self.take(8))[0..8], .little);
    }
};

test "context-free score floor bounds learned production scores" {
    const alloc = std.testing.allocator;
    var counts = Counts.init(alloc);
    defer counts.deinit(alloc);
    const ctx: Context = .{};
    try counts.add(alloc, ctx, .raw, 1000);
    var learned = try counts.toPrior(alloc);
    defer learned.deinit(alloc);

    const productions = [_]ops.OpKind{ .raw, .bitpack, .huffman, .rans };
    var scores: [productions.len]u32 = undefined;
    learned.scoreSet(ctx, &productions, &scores);
    var actual = scores[0];
    for (scores[1..]) |score| actual = @min(actual, score);
    try std.testing.expect(learned.scoreLowerBound(productions.len) <= actual);
    try std.testing.expectEqual(@as(u32, 2048), Prior.empty.scoreLowerBound(productions.len));
}
