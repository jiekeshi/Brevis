//! Worker-thread helpers shared by planning, encoding, and decoding.
//!
//! `runWorkers` spawns one thread per job for a single batch. `BatchPool`
//! keeps a fixed set of threads alive across batches so the CLI can decode the
//! next batch while the current one is written.

const std = @import("std");

const Allocator = std.mem.Allocator;

pub fn runWorkers(alloc: Allocator, n_threads: usize, job: anytype, comptime run: anytype) !void {
    const threads = try alloc.alloc(std.Thread, n_threads);
    defer alloc.free(threads);
    var spawned: usize = 0;
    errdefer for (threads[0..spawned]) |thread| thread.join();
    for (threads) |*thread| {
        thread.* = try std.Thread.spawn(.{}, run, .{job});
        spawned += 1;
    }
    for (threads) |thread| thread.join();
}

pub fn BatchPool(comptime Job: type) type {
    return struct {
        const Self = @This();

        alloc: Allocator,
        io: std.Io,
        threads: []std.Thread,
        mutex: std.Io.Mutex = .init,
        ready: std.Io.Condition = .init,
        done: std.Io.Condition = .init,
        job: ?*Job = null,
        epoch: usize = 0,
        finished: usize = 0,
        stopping: bool = false,

        pub fn init(self: *Self, alloc: Allocator, io: std.Io, n_threads: usize) !void {
            self.* = .{
                .alloc = alloc,
                .io = io,
                .threads = if (n_threads > 1) try alloc.alloc(std.Thread, n_threads) else &.{},
            };
            var spawned: usize = 0;
            errdefer {
                self.stop();
                for (self.threads[0..spawned]) |thread| thread.join();
                if (self.threads.len > 0) alloc.free(self.threads);
            }
            for (self.threads) |*thread| {
                thread.* = try std.Thread.spawn(.{}, worker, .{self});
                spawned += 1;
            }
        }

        pub fn deinit(self: *Self) void {
            self.stop();
            for (self.threads) |thread| thread.join();
            if (self.threads.len > 0) self.alloc.free(self.threads);
        }

        pub fn run(self: *Self, job: *Job) void {
            if (self.threads.len == 0) return Job.run(job);
            self.mutex.lockUncancelable(self.io);
            self.job = job;
            self.finished = 0;
            self.epoch += 1;
            self.ready.broadcast(self.io);
            while (self.finished < self.threads.len) self.done.waitUncancelable(self.io, &self.mutex);
            self.mutex.unlock(self.io);
        }

        pub fn stop(self: *Self) void {
            self.mutex.lockUncancelable(self.io);
            self.stopping = true;
            self.ready.broadcast(self.io);
            self.mutex.unlock(self.io);
        }

        fn worker(self: *Self) void {
            var seen: usize = 0;
            while (true) {
                self.mutex.lockUncancelable(self.io);
                while (!self.stopping and self.epoch == seen)
                    self.ready.waitUncancelable(self.io, &self.mutex);
                if (self.stopping) {
                    self.mutex.unlock(self.io);
                    return;
                }
                seen = self.epoch;
                const job = self.job.?;
                self.mutex.unlock(self.io);

                Job.run(job);

                self.mutex.lockUncancelable(self.io);
                self.finished += 1;
                if (self.finished == self.threads.len) self.done.signal(self.io);
                self.mutex.unlock(self.io);
            }
        }
    };
}
