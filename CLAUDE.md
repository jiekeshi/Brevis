# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Brevis synthesizes a short reversible program per tensor instead of applying a fixed
codec. `src/` is the Zig 0.16 engine (~6.6 K lines); `eval/` is the audited measurement
apparatus (~25 K lines) and holds most of the repository's invariants.

Read first:

- [`README.md`](README.md) — what Brevis is, cluster setup, CLI, pipeline, source layout.
- [`doc/architecture.md`](doc/architecture.md) — per-file responsibilities, the
  acyclic module layering, and known tensions worth not re-litigating.
- [`doc/engineering-principles.md`](doc/engineering-principles.md) — **follow these when
  writing or refactoring code here.**
- [`doc/cluster-pitfalls.md`](doc/cluster-pitfalls.md) — environment traps, and the one
  known environment-sensitive test failure. Check it before debugging a setup problem.

## Commands

`source setup_env.sh` first; see README *Cluster Setup*. Build and test:

```bash
zig build -Doptimize=ReleaseFast          # binary at ./zig-out/bin/brevis
zig build test -Doptimize=ReleaseFast     # 39 tests in src/tests.zig
python3 -m unittest discover -s eval  -p 'test_*.py'
python3 -m unittest discover -s tools -p 'test_*.py'
```

Two invocation traps:

- `build.zig` does not forward `b.args` to the test runner, so `zig build test -- …`
  cannot filter. Run one Zig test with
  `zig test src/tests.zig --test-filter "codec: huffman and rans roundtrip"`.
- `eval/` modules import each other by bare name, so a single Python test must run from
  inside `eval/`: `cd eval && python3 -m unittest test_campaign_runner -k some_case`.
  The dotted form fails on import.

`brevis config <search flags>` prints how a flag combination resolves without running
anything. Use it before spending a long run.

## Invariants

These require reading several files to reconstruct, and breaking them is silent.

**The two costs are independent, and conflating them is the classic bug here.**
`p` (grammar description length from the prior) determines only *which* partial program
A* expands next. `g_bytes + lowerBound` is a lower bound on serialized bytes. The prior
never changes the compression objective — that is what makes `uniform` vs `phog` a
meaningful measurement. Keep them separate in any new code.

**The byte lower bound is only valid at deep holes.** Empirical entropy is *not* a valid
bound while transforms remain available, because a reversible transform can cut it
sharply. At shallow holes the bound includes only required terminal-frame costs; a
Shannon payload bound is added only once `max_depth` is reached.
`OpKind.isAlphabetPermutation` marks the ops that provably preserve zeroth-order
entropy, so a child's bound equals its parent's.

**Single-stream search and tensor planning behave differently.** `synthesize` prunes
with the serialized-byte bound and honors `max_realizations`. `synthesizeTensorPlan`
does *not*: it collects every candidate completed within the expansion budget in grammar
order, then reranks the top `rerank_candidates` on `rerank_blocks` full blocks using
actual encoded bytes. `max_realizations` is deliberately ignored there — see
`tensor_search_uses_max_realizations: false` in the JSON config dump.

**`raw` is never removable.** The CLI rejects `--disable-op raw`, and direct API callers
that clear its mask bit still get it as an implicit fallback. This guarantees every
supported tensor has at least one valid program. `--plan fixed` ignores the search mask
entirely and always executes the dtype template.

**`--max-depth` counts transform layers only.** Reports expose both
`program_transform_depth` (matches the flag) and `program_node_depth` (includes the
terminal layer). Do not use them interchangeably.

**Arity is width-dependent.** `bit_plane` has `in_bpe` children and `byte_plane` has
`ceil(in_bpe/8)`, so tree shape is not fixed by the op alone. `arity()` in `ops.zig` is
the authority. Grammar guards live in `legalProductions` (`search.zig`), keyed on width:
Huffman and rANS only through `MAX_ENTROPY_BPE` (16) bits, `bit_plane` only above 1 bit,
`byte_plane` only above 8, `split_float` only at a floating root.

**Calibration searches one centered contiguous window per tensor.** That single window is
load-bearing — concatenating disjoint windows would fabricate transitions that
delta-based transforms and features then read as real. The prior is deliberately weak:
mixed with uniform at `PHOG_WEIGHT = 1/20` over the admitted production slice.

**Archive frames are independent.** Schema 6 writes no cross-block back-references, so
every frame decodes on its own; the reader still accepts legacy schema 5/6 references but
rejects forward refs, self-refs, and bad lengths. Compression streams frames out in
batches rather than holding the archive in memory, which is what makes sharded
multi-hundred-GB models tractable.

## Evaluation is audited, not ad hoc

`eval/` is not a scratch benchmarking area. The harnesses enforce gates, and casually
relaxing one silently downgrades a result from formal to worthless.

- **A run that disables the integrity, clean-tree, or harness-build gate is labelled
  `pilot` and is permanently ineligible for formal results.** There is no way to promote
  it afterward.
- **Formal execution is two steps with the same filters and roots**: plan to get a
  `task_semantic_sha256`, then `--execute-one <sha>`. Do not pass the plan file back to
  the execution command.
- **Any plan or result written inside the repository must be under a Git-ignored
  directory**, or formal execution aborts before measuring anything.
- Codec-affecting environment variables (`OMP_NUM_THREADS`, `XZ_OPT`, `ZSTD_NBTHREADS`,
  …) must be *truly unset*, not set to a safe value. Pass `--jobs` instead.
- Technical repetitions are not independent tensors. The harnesses fingerprint repeated
  reports and a mismatch invalidates the run.
- `eval/PROTOCOL.md` and `eval/models-tiered.json` are **preregistered**. Treat changes
  as protocol amendments, not edits — dated and justified in the file.
- Result files predating the current schema must be rerun, not compared against.

`eval/cache/` is git-ignored. `eval/results/formal/` holds committed, audited artifacts.

## Repository context

- `main` carries the current engine plus the full evaluation apparatus. The remote branch
  `unify-low-level-pipeline` is a superseded earlier design (different module names, v2
  PNode archive format, no `eval/` at all) — not a pending feature branch, do not merge.
- `paper/` is a separate git repo (an Overleaf clone of the AAAI-27 manuscript) and is
  git-ignored here. `paper/Synthzip-draft/notes/claim-ledger.md` maps every number in the
  manuscript to the artifact that supports it, and explicitly lists what is *not*
  supported. If asked to change a reported number, reconcile the ledger too.
