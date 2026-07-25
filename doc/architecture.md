# Architecture

Brevis compresses neural-network tensors by **synthesizing a short program per
tensor** in a finite, typed language whose operators are reversible by
construction. An archive stores the program plus its leaf data. Decompression
performs no search: it executes the stored program.

The repository has two layers with deliberately different characters. `src/` is
a small, tightly layered Zig engine. `eval/` is a much larger Python measurement
apparatus whose complexity buys auditability, not features.

```
                   safetensors input
                          |
   [ types ] block planning, dtype and stream model
                          |
   [ search ] A* over reversible programs, guided by [ prior ]
                          |            reranked on real blocks
   [ program ] execute the winning tree
                          |
   [ codec ] raw / bitpack / Huffman / rANS on the leaves
                          |
   [ archive ] stream independent block frames to .brv
```

## `src/` — the engine

Line counts are the shape of the module, not a target.

| File | Lines | Responsibility |
| --- | ---: | --- |
| `types.zig` | 210 | `Dtype`, `Stream` (1–32 bit logical elements), `TensorView`, `planBlocks`. The vocabulary everything else shares. |
| `codec.zig` | 844 | Terminal encoders: bitpack, canonical Huffman, rANS, histograms, and their exact cost functions. |
| `ops.zig` | 758 | The reversible operator set. Every non-terminal defines `forward`/`inverse`; `arity()` and the budget constants live here. |
| `program.zig` | 430 | The `Node` tree: execute, decode, serialize, and split payload from bytecode. |
| `prior.zig` | 309 | `Context`, three-level backoff, `Counts` → `Prior`. The PHOG grammar model. |
| `search.zig` | 913 | PHOG-guided A*, byte lower bounds, `fixedPlan`, and per-tensor planning with full-block reranking. |
| `calibrate.zig` | 217 | Input-local stratified sampling and prior fitting. Needs no external corpus. |
| `archive.zig` | 317 | Schema-6 `.brv` writer and backward-compatible reader; `tensorMetas` groups blocks under tensors. |
| `safetensors.zig` | 241 | Memory-mapped safetensors I/O. |
| `pool.zig` | 103 | `runWorkers` and `BatchPool`: the only concurrency primitives. |
| `report.zig` | 578 | Text and schema-4 JSON reporting for `bench`. Reads finished plans; never plans or encodes. |
| `main.zig` | 1055 | CLI: argument handling, orchestration, batching. |
| `tests.zig` | 1531 | 39 tests over operators, programs, archives, codecs, and dtypes. |

### The dependency graph is an acyclic layering

```
types  <- codec <- ops <- program, prior <- search <- calibrate, archive
pool   (no local dependencies)
report <- archive, ops, prior, program, safetensors, search, types
main   <- everything
```

`types.zig` and `pool.zig` depend on nothing local. No cycles. Each module
depends only downward, so a change to `search.zig` cannot ripple into `codec.zig`.

`report.zig` is the one wide-fan-in module, which is intrinsic: a report
describes everything. It stays acyclic because it only *reads* — it takes
finished `Plan` and `Result` values and serializes them. It deliberately does
not import the CLI's `PlanMode`; it receives a small `report.Mode { name,
plan_is_search }` instead, so changing how the CLI selects a plan does not
change the reporting contract.

### Invariants that span modules

These cannot be seen from one file and are the ones worth protecting.

**Two independent costs.** `p` — grammar description length from the prior —
decides only *which* partial program A* expands next. `g_bytes + lowerBound` is
a bound on serialized bytes and is the objective. The prior never changes the
objective, which is exactly what makes `uniform` vs `phog` a meaningful
measurement.

**The byte lower bound is valid only at deep holes.** Empirical entropy is not a
bound while transforms remain available, because a reversible transform can cut
it sharply. Shallow holes count only required terminal-frame cost; the Shannon
payload term is added once `max_depth` is reached.
`OpKind.isAlphabetPermutation` marks operators that provably preserve
zeroth-order entropy, so a child's bound equals its parent's.

**`raw` is never removable.** The CLI rejects `--disable-op raw`, and API callers
who clear its mask bit still get it as an implicit fallback. This is what
guarantees every supported tensor has at least one valid program.

**Arity depends on width.** `bit_plane` has `in_bpe` children, `byte_plane` has
`ceil(in_bpe/8)`. Tree shape is not determined by the operator alone;
`ops.arity()` is the authority.

**Single-stream search and tensor planning differ.** `synthesize` prunes on the
byte bound and honors `max_realizations`. `synthesizeTensorPlan` ignores
`max_realizations` on purpose: it collects every candidate completed within the
expansion budget, then reranks the top few on representative full blocks using
actual encoded bytes.

**Calibration uses one centered contiguous window per tensor.** Concatenating
disjoint windows would fabricate transitions that delta-based transforms and
features would then read as real structure.

**Archive frames are independent.** Schema 6 writes no cross-block
back-references, so each frame decodes alone and compression can stream frames
out in batches instead of holding the archive in memory.

## `eval/` — the measurement apparatus

21 K lines across seven modules, each with a matching `test_*.py` (159 tests).
It is larger than the engine because it enforces gates rather than merely
running commands.

| File | Role |
| --- | --- |
| `run_eval.py` | End-to-end fixed / uniform / phog against gzip, zstd, xz, OpenZL. |
| `benchmarking.py` | Repeated **serial** generic-codec registry (19 rows). Also the de-facto shared library — see *Known tensions*. |
| `brevis_benchmarking.py` | Repeated Brevis-system harness: builds and hashes a ReleaseFast binary, requires a clean tree, verifies every decode by full byte comparison, and checks realized program digests against an independent scan of the `.brv` frames. |
| `campaign_runner.py` | Expands the frozen matrix into hash-addressed tasks and executes **at most one**. Planning is read-only; execution regenerates and revalidates rather than trusting a saved plan. |
| `analyze_generated_dsl.py` | Bench-schema-4 evidence → canonical report/tensor/block/node JSONL. |
| `summarize_*.py` | Aggregation that refuses to count technical repetitions as new observations. |

The gates are the point. A run that disables the integrity, clean-tree, or
harness-build gate is machine-labelled `pilot` and is permanently ineligible for
formal results. `eval/PROTOCOL.md` and `eval/models-tiered.json` are
preregistered; changing them is a protocol amendment, not an edit.

## Supporting files

| Path | Role |
| --- | --- |
| `setup_env.sh` | Cluster modules, Zig toolchain lookup, node-local virtual environment. |
| `requirements.txt` | numpy, needed only by `eval/tensor_stats.py`. |
| `tools/model_cache.py` | Fetch / verify / drop one checkpoint at a time (13 tests). |
| `doc/` | This file, engineering principles, cluster pitfalls, checkpoint acquisition. |
| `CLAUDE.md` | Guidance for AI coding agents: cross-module invariants and evaluation gates. |

## Known tensions

Recorded rather than hidden, measured against
[`engineering-principles.md`](engineering-principles.md).

**`benchmarking.py` has two reasons to change.** It is both the generic-codec
harness and the shared library that `campaign_runner.py` and
`brevis_benchmarking.py` import as `common` (44 and 45 call sites). The right
fix is to extract `eval/common.py`. It has not been done because `eval/` is
audited: moving code changes harness hashes and would invalidate the comparison
basis of already-archived results. Do this only alongside a protocol amendment.

**The aggregation layer is large for the evidence available.**
`analyze_generated_dsl.py` plus three `summarize_*.py` modules total ~6.3 K lines
while exactly one formal artifact exists. Revisit for YAGNI once multiple models
have completed.

**`main.zig` is still the largest non-test file at 1055 lines.** Reporting and
concurrency have been extracted; what remains is argument handling plus one
command function per subcommand, which is cohesive but not small.

**`eval/tensor_stats.py` has no tests**, unlike every other `eval/` module.
