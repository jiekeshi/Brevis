//! Per-compression spare-core budget shared by tensor workers and the codec.
//!
//! The scheduler runs one tensor per core, which leaves a checkpoint whose
//! sizes are skewed — one embedding matrix can outweigh every other tensor —
//! finishing on a single core while the rest idle. Whatever cores no tensor
//! holds are offered here so per-tensor work can fan out into them instead.
//!
const std = @import("std");

pub const Budget = struct {
    spare: std.atomic.Value(usize),

    pub fn init(spare_cores: usize) Budget {
        return .{ .spare = .init(spare_cores) };
    }

    pub fn claim(self: *Budget, wanted: usize) usize {
        var available = self.spare.load(.monotonic);
        while (available != 0) {
            const granted = @min(wanted, available);
            if (self.spare.cmpxchgWeak(
                available,
                available - granted,
                .monotonic,
                .monotonic,
            )) |current| {
                available = current;
                continue;
            }
            return granted;
        }
        return 0;
    }

    pub fn release(self: *Budget, cores: usize) void {
        if (cores != 0) _ = self.spare.fetchAdd(cores, .monotonic);
    }
};

threadlocal var active: ?*Budget = null;

pub fn bind(budget: ?*Budget) ?*Budget {
    const previous = active;
    active = budget;
    return previous;
}

pub fn claim(wanted: usize) usize {
    return if (active) |budget| budget.claim(wanted) else 0;
}

pub fn release(cores: usize) void {
    if (active) |budget| budget.release(cores);
}
