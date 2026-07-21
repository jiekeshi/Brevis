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

const SAMPLE_MAX: usize = 64 * 1024;
const MAGIC = "BRVP";
const VERSION: u32 = 1;

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

        // Folded to 16 bits: at most SAMPLE_MAX distinct symbols are observable
        // anyway, and the buckets are coarse.
        var hist: [1 << 16]u32 = undefined;
        @memset(&hist, 0);

        var zeros: usize = 0;
        for (0..n) |i| {
            const v = s.getU32(i) & m;
            if (v == 0) zeros += 1;
            hist[fold(v)] += 1;
        }
        const h0 = entropy(&hist, n);

        @memset(&hist, 0);
        var prev = s.getU32(0) & m;
        hist[fold(prev)] += 1;
        for (1..n) |i| {
            const v = s.getU32(i) & m;
            hist[fold((v -% prev) & m)] += 1;
            prev = v;
        }
        const h_delta = entropy(&hist, n);

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
    return (v ^ (v >> 16)) & 0xFFFF;
}

fn bpeBucket(bpe: u8) u8 {
    if (bpe <= 1) return 0;
    if (bpe <= 8) return 1;
    if (bpe <= 16) return 2;
    return 3;
}

fn entropy(hist: []const u32, n: usize) f64 {
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

    /// No observations. `score` then reports every production as equally
    /// likely, which is the correct posterior for an unseen context -- it is
    /// not a stand-in for a trained prior.
    pub const empty: Prior = .{ .levels = .{ .empty, .empty, .empty } };

    pub fn deinit(self: *Prior, alloc: Allocator) void {
        for (&self.levels) |*m| m.deinit(alloc);
    }

    pub fn score(self: Prior, ctx: Context, op: ops.OpKind) u32 {
        const idx: usize = @intFromEnum(op);
        std.debug.assert(idx < N_PROD);
        for (0..3) |lv| {
            if (self.levels[lv].get(ctx.hash(@intCast(lv)))) |row| return row[idx];
        }
        return ops.UNIFORM_SCORE;
    }

    pub fn scoreSet(self: Prior, ctx: Context, productions: []const ops.OpKind, out: []u32) void {
        std.debug.assert(productions.len == out.len and productions.len > 0);
        var floor: u32 = std.math.maxInt(u32);
        for (productions) |op| floor = @min(floor, self.score(ctx, op));

        var sum: f64 = 0;
        for (productions) |op| {
            const delta: f64 = @floatFromInt(self.score(ctx, op) - floor);
            sum += std.math.exp2(-delta / 1024.0);
        }
        const log_sum = std.math.log2(sum) * 1024.0;
        for (productions, out) |op, *dst| {
            const delta: f64 = @floatFromInt(self.score(ctx, op) - floor);
            dst.* = @intFromFloat(@round(delta + log_sum));
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
