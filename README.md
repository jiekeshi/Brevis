# Brevis

Brevis provides bit-exact lossless compression for neural-network tensors.

Instead of using a fixed, hand-written codec, it synthesizes a short program for each tensor in a finite, typed language whose operations are reversible by construction.

An archive contains the program, leaf data, and required metadata. Decompression performs no search; it executes the stored program to reconstruct the original safetensors file exactly.

Brevis uses Zig 0.16.

## Cluster Setup (Alliance / Nibi)

`source setup_env.sh` prepares everything. It loads the modules, puts Zig on
`PATH`, and creates a node-local virtual environment on first use.

| Component | Source |
| --- | --- |
| `StdEnv/2023`, `python/3.12` | Lmod |
| gzip, bzip2, xz, Zstandard, LZ4, Brotli | already in the `gentoo/2023` base environment; no module needed |
| Zig 0.16.0 | no Lmod module exists — install once (below) |
| numpy | `requirements.txt`; needed only by `eval/tensor_stats.py` |

Install the Zig toolchain once. `setup_env.sh` looks for `$ZIG_ROOT`, then
`.toolchain/zig-0.16.0` in the repository, then `~/software/zig-0.16.0`:

```bash
mkdir -p ~/software && cd ~/software     # or: mkdir -p .toolchain && cd .toolchain
curl -LO https://ziglang.org/download/0.16.0/zig-x86_64-linux-0.16.0.tar.xz
tar xf zig-x86_64-linux-0.16.0.tar.xz && mv zig-x86_64-linux-0.16.0 zig-0.16.0
```

The toolchain is 19.5 K files and 347 MB. Keep it out of `/project`, whose
inode quota is the tight one.

Build and evaluate on a compute node, not a login node:

```bash
salloc --account=def-zhouyang --time=2:00:00 --cpus-per-task=16 --mem=64G
source setup_env.sh && zig build -Doptimize=ReleaseFast
```

Two environment details matter and are handled by `setup_env.sh`:

- **Do not export `OMP_NUM_THREADS`, `XZ_OPT`, `ZSTD_NBTHREADS`, or the other
  names in `benchmarking.py`'s `CODEC_ENVIRONMENT_VARIABLES`.** Formal campaign
  execution requires them to be truly unset and aborts otherwise. Use `--jobs`.
- Some Alliance sessions export `PIP_PREFIX`, which silently redirects `pip
  install` out of the active virtual environment while still reporting success.
  `setup_env.sh` clears it along with `PYTHONPATH`.

The build cache and virtual environment go to `$SLURM_TMPDIR`. The parallel
filesystem is slow on many small files: a cold ReleaseFast build takes about
four minutes of wall time for well under one minute of CPU.

[`doc/cluster-pitfalls.md`](doc/cluster-pitfalls.md) records these traps in full,
plus one known environment-sensitive test failure.

## Quick Start

```bash
zig build -Doptimize=ReleaseFast

# Optional: calibrate a grammar prior from the current input
./zig-out/bin/brevis calibrate model.safetensors prior.bin

# Search with the learned prior; omit --prior for uniform A*
./zig-out/bin/brevis compress model.safetensors model.brv --plan search --prior prior.bin

# Fixed typed-DSL template with the normal raw fallback (no search)
./zig-out/bin/brevis compress model.safetensors fixed.brv --plan fixed
./zig-out/bin/brevis decompress model.brv restored.safetensors
./zig-out/bin/brevis verify model.brv model.safetensors

# Set the thread count or number of calibration tensors
./zig-out/bin/brevis compress model.safetensors model.brv --jobs 12
./zig-out/bin/brevis calibrate model.safetensors prior.bin --tensors 200

# Emit a machine-readable per-tensor and per-block DSL/search report
./zig-out/bin/brevis bench model.safetensors --prior prior.bin --format json > bench.json

# Override search and full-block reranking for controlled ablations
./zig-out/bin/brevis bench model.safetensors --format json \
  --max-expansions 64 --max-depth 1 --max-nodes 8 \
  --sample-elems 2048 --rerank-candidates 0 \
  --disable-op huffman --disable-op rle > ablation.json

# Print the effective runtime configuration (the same flags are accepted)
./zig-out/bin/brevis config --max-expansions 64 --max-depth 1
```

```bash
zig build test -Doptimize=ReleaseFast
python3 -m unittest discover -s eval  -p 'test_*.py'
python3 -m unittest discover -s tools -p 'test_*.py'
```

To run a single test, note two invocation constraints. `build.zig` does not
forward `b.args` to the test runner, so `zig build test -- …` cannot filter; and
the `eval/` modules import each other by bare name, so a single Python test must
run from inside `eval/`.

```bash
zig test src/tests.zig --test-filter "codec: huffman and rans roundtrip"
cd eval && python3 -m unittest test_campaign_runner -k stage_gate
```

## Pipeline

```text
safetensors
  → memory-map the input and plan blocks per tensor
  → search for reversible program candidates on tensor samples
  → rerank candidates on representative full blocks and select one per tensor
  → fit small per-block parameters and execute the program
  → encode terminal streams with raw / bitpack / Huffman / rANS
  → stream output to .brv

.brv
  → read the program, side information, and payload
  → decode terminal streams
  → apply reversible transforms in reverse order
  → restore the original safetensors header and tensor bytes
```

Inputs and archives are memory-mapped when read. Compression generates frames in batches and writes them in order instead of retaining the complete archive in memory, allowing bounded-memory processing of large sharded models.

## Reversible Language

A search state is a typed program tree with holes. Every nonterminal operator defines a `forward` transform and its inverse; terminal operators encode the final integer streams as bytes.

Transforms include constant XOR, modular addition, adjacent XOR, differencing, Gray coding, bit rotation, field and floating-point field splitting, run-length encoding, codebooks, bit planes, and byte planes.

Terminals are `raw`, `bitpack`, canonical Huffman, and rANS.

Operator applicability depends on input bit width, dtype, tree depth, arity, and the runtime operator mask. Repeating `--disable-op NAME` removes individual transforms or terminal codecs from search for controlled ablations. The CLI does not allow `raw` to be disabled; direct API callers also retain it as an implicit reversible fallback, so every supported tensor has at least one valid program. Fixed-plan mode deliberately ignores the search mask and continues to execute the same typed template.

### Learned macros

`--macros <library.json>` offers named subtrees of the operators above as single productions. A macro holds operators and holes, never data, and every node in its body is checked against the same legality rules it would face as a primitive production — so a macro cannot reach a program the grammar could not, and `--disable-op` still removes every macro that uses that operator. The engine expands macros into primitives before serializing, so archives and the decoder are unaffected; what a macro changes is how many search expansions a program costs to find. `brevis config --macros L` prints how a library resolves, and the JSON bench report attributes each tensor's plan to the macros that built it. `autodsl/` is the loop that proposes and gates them — see [`doc/autodsl.md`](doc/autodsl.md).

## Grammar-Guided A* Search

Brevis uses a probabilistic higher-order grammar (PHOG) to order the search. It maintains two independent costs:

- `p` is the grammar description length. It only determines which partial program A* expands first.
- `g_bytes + lowerBound` is a lower bound on serialized bytes for the stream being searched.

PHOG does not change the compression objective. It only changes candidate order within a bounded search budget. Without a prior, Brevis uses a uniform distribution so the prior's contribution can be measured directly.

Single-stream search can prune with the serialized-byte bound; tensor planning instead collects candidates in grammar order under a per-tensor expansion budget without sample-incumbent byte pruning, because it ultimately compares them on representative full blocks. The defaults are 256 partial-program expansions, at most 2 transform layers, 12 total transform-and-terminal nodes, and a 4,096-element sample. A sample size of zero searches the complete tensor. These limits are runtime options so budget and depth can be swept without rebuilding the binary. The single-stream `max_realizations` guard is not applied to tensor planning, which collects every completed candidate reached within the expansion budget.

Evaluation separates three modes: `fixed` executes a dtype-specific DSL template with the normal raw fallback, `uniform` runs A* without a learned prior, and `phog` runs the same A* search with an input-local prior. Comparing them isolates the value of the reversible language, adaptive search, and grammar guidance.

Empirical entropy is not a valid lower bound at shallow holes where transforms remain available, because a reversible transform may reduce it sharply.

At those holes, the bound includes only required terminal-frame costs. A Shannon payload bound is added only after the maximum transform depth is reached.

### Input-Local Calibration

Calibration requires no external corpus. It stratifies tensors by dtype and size, searches one centered contiguous window per selected tensor, then, by default, reranks the top eight candidates on four representative full blocks using actual encoded bytes. A single window preserves true adjacency for delta-based transforms and features; concatenating disjoint windows would introduce artificial transitions. Setting either `--rerank-candidates 0` or `--rerank-blocks 0` disables full-block reranking and retains the candidate that is best on the search sample. Calibration, compression, and benchmarking receive the same runtime search options.

`calibrate --format json` records the input and prior SHA-256 digests, sampled tensor count, requested and actual worker counts, seed, context counts, wall time, and full search configuration. A subsequent benchmark identifies the applied prior by the same digest, so calibration and evaluation settings can be audited rather than inferred from a filename.

Winning programs become `(context, production)` counts. A context captures tree position, dtype, bit width, and bucketed entropy, zero rate, and delta features. Three-level backoff reduces sparsity.

The final prior is mixed with the uniform distribution to avoid overfitting sparse statistics.

## Archive and Decoding

The current writer produces schema 6 archives. Each block frame stores program bytecode, side information, and payload. The footer stores tensor names, dtypes, shapes, block counts, and the original safetensors header.

New archives contain no cross-block back-references, keeping every frame independent and easy to parallelize. The reader still supports legacy schema 5/6 back-references and rejects forward references, self-references, and invalid lengths.

Single-threaded decoding runs directly on the calling thread. Multithreaded mode reuses a fixed worker pool and decodes the next batch while the current batch is written.

Output order and per-tensor lengths are checked before and after writing.

## Evaluation

[`eval/run_eval.py`](eval/run_eval.py) evaluates fixed, uniform, and PHOG-guided plans together with gzip, Zstandard, xz, and OpenZL. Schema-3 results record the Git commit and dirty state, full command, binary and evaluation-script hashes, the complete manifest hash, each input hash and their ordered aggregate, per-shard prior hashes, thread counts, search settings, and the command mode behind every result field. When the manifest supplies expected byte counts and SHA-256 digests, the evaluator enforces them before invoking a codec and includes the integrity decision in the shard record.

[`eval/benchmarking.py`](eval/benchmarking.py) is the repeated generic-codec harness. It benchmarks raw copy plus speed-, default-, and ratio-oriented gzip, bzip2, xz, Zstandard, LZ4, and Brotli configurations. The default protocol uses one warmup and six measured repetitions in seeded, paired forward/reverse orders. Every iteration stores raw compression and decompression time, direct-child peak RSS, archive size and SHA-256, complete round-trip verification, and structured failures. Model-labelled runs must match the declared source size and SHA-256 and a SHA-verified manifest entry. This registry is intentionally a serial-codec track; matched-worker and scaling experiments are reported separately.

[`eval/brevis_benchmarking.py`](eval/brevis_benchmarking.py) is the repeated Brevis-system harness. It integrity-binds one shard to the tiered manifest, builds and hashes a ReleaseFast binary by default, requires a clean tree, times input-local calibration separately, and benchmarks raw-terminal, fixed, uniform, and PHOG configurations in paired orders. All timed archive compression/decompression pipelines finish before the independent diagnostic replays begin, so a method's search replay cannot precondition a timed archive task. Each operation records process-level time and peak RSS, and every decode is followed by a complete untimed byte comparison. The harness checks measured archive reproducibility, exact size projection, and every realized program-bytecode digest against an independent scan of the actual `.brv` frames. For each configuration, the first complete successful warmup report, or the first complete successful measured report when there is no such warmup, supplies the single canonical schema-4 `tensors`/`blocks` detail. Matching repetitions retain every other top-level field, including internal timing, plus a semantic SHA-256 and a checkpoint-local canonical reference. The fingerprint excludes only the two internal wall times and machine-local input/prior paths. A mismatch retains its full detail and invalidates the run, so repeated diagnostics cannot be counted as independent tensors. Runs that disable integrity, clean-tree, or harness-build gates are machine-labelled `pilot` and are ineligible for formal results.

[`eval/campaign_runner.py`](eval/campaign_runner.py) expands the frozen matrix
into hash-addressed comparison tasks and executes at most one selected task.
Formal Brevis and generic-codec tasks share a fixed advisory machine lock. The
generic branch runs the exact 19-row registry once, requires a clean bound
commit and unset codec overrides, checks a 30% disk reserve before and after the
run, and publishes an immutable raw checkpoint plus a separate validation
receipt. The receipt distinguishes a trustworthy failed attempt, method
success, and eligibility for formal aggregation. Planning is read-only by
default; execution regenerates and revalidates the selected task rather than
trusting a saved plan.

```bash
python3 eval/campaign_runner.py \
  --campaign small-core-system-v1 --model-tag small-bert-f32 \
  --cache-root eval/cache \
  --staging-root eval/cache/formal-staging/frozen-v1 \
  --plan-output /tmp/brevis-bert-plan.json

TASK_SHA256=$(jq -r '.tasks[0].task_semantic_sha256' /tmp/brevis-bert-plan.json)
python3 eval/campaign_runner.py \
  --campaign small-core-system-v1 --model-tag small-bert-f32 \
  --cache-root eval/cache \
  --staging-root eval/cache/formal-staging/frozen-v1 \
  --execute-one "$TASK_SHA256"
```

Use the same filters and roots for planning and execution. Do not pass the
already-created plan path to the execution command. A plan or result written
inside the repository must be under a Git-ignored directory; formal execution
otherwise fails before measurement.

```bash
python3 eval/benchmarking.py eval/cache/.../model.safetensors \
  --output eval/results/generic-small-bert.json \
  --model-tag small-bert-f32 --model-repo google-bert/bert-base-uncased \
  --model-revision 86b5e0934494bd15c9632b12f734a8a67f723594 \
  --manifest eval/models-tiered.json --shard model.safetensors \
  --expected-source-size 440449768 \
  --expected-source-sha256 68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3 \
  --expected-manifest-sha256 84202aee827632724d7441e9ae4725633778cdafacc4f31eaf4449a044b988ad
```

[`eval/models-tiered.json`](eval/models-tiered.json) and [`eval/PROTOCOL.md`](eval/PROTOCOL.md) preregister the heterogeneous model matrix, staged resource gates, repetition policy, and the non-extrapolating GLM-5.2 shard sample. The older compact manifests remain available for smoke tests and historical reruns.

`brevis bench --format json` schema 4 separates tensor planning from block encoding time and records tensor file offsets, dtype and shape, search counters, structured program trees and parameters, and planned-raw versus fallback-raw roots. Block records separate program bytecode, packed terminal payload, and frame overhead. They also bind the realized program bytecode through per-block SHA-256 digests and an ordered aggregate digest. Top-level fields give exact header, footer, and projected archive accounting. `program_node_depth` includes the terminal layer, whereas `program_transform_depth` counts only transform layers and matches `--max-depth`. The report is generated by a diagnostic replay; the system harness verifies its program digests against the actual archive, while complete `.brv` file sizes remain the effectiveness measurement.

[`eval/analyze_generated_dsl.py`](eval/analyze_generated_dsl.py) converts only
complete system-schema-2 checkpoints with bench-schema-4 evidence into
canonical report-, tensor-, block-, and node-level JSONL. It independently
resolves and rehashes compact checkpoint-local report references, analyzes each
configuration once, and excludes pilots by default. Tensor roles come from the
SHA-bound, versioned name rules in
[`eval/tensor_role_rules_v1.json`](eval/tensor_role_rules_v1.json); they are
heuristic strata rather than architecture ground truth. The analyzer reports
terminal presence but does not invent per-terminal payload shares that schema 4
does not expose.

```bash
python3 eval/analyze_generated_dsl.py \
  eval/cache/formal-staging/small-bert-f32-core-system-v1.json \
  --output-dir eval/cache/formal-staging/small-bert-f32-dsl
```

[`eval/summarize_generated_dsl.py`](eval/summarize_generated_dsl.py) validates
one or more formal DSL-analysis directories and aggregates program length,
depth, bytecode, operator/terminal combinations, tensor strata, and
search-counter relationships without treating technical repetitions as new
tensors. [`eval/summarize_generic_benchmarks.py`](eval/summarize_generic_benchmarks.py)
accepts only raw/validation-receipt pairs from the formal generic campaign. It
rechecks every iteration and archive digest, keeps bzip2's equivalent `-9` and
`--best` labels visible but gives them one independent-observation identity,
and reports per-input, pooled per-model, model-equal, and raw-byte views. For a
multi-shard model, serial wall time is the sum of shard means and peak RSS is
the maximum over measured shard trials; no unsupported cross-shard standard
deviation is synthesized.

```bash
python3 eval/summarize_generated_dsl.py \
  eval/results/formal/system/small-core-system-v1/small-bert-f32/model.safetensors \
  --output /tmp/bert-dsl-aggregate.json

python3 eval/summarize_generic_benchmarks.py \
  --pair /path/to/task.json /path/to/task.validation.json \
  --output /tmp/generic-aggregate.json
```

The evaluator rebuilds the ReleaseFast binary so recorded source settings match the executable. Existing result files predate this schema and must be rerun before comparison with the current planner.

## Source Layout

```text
src/types.zig        dtypes, Stream, TensorView, and block planning
src/ops.zig          reversible language operators
src/macro.zig        learned macro libraries offered as single productions
src/program.zig      program execution, inversion, and serialization
src/search.zig       PHOG-guided A* and byte lower bounds
src/prior.zig        contexts and three-level backoff prior
src/calibrate.zig    input-local sampling and prior fitting
src/codec.zig        bitpack, Huffman, and rANS codecs
src/archive.zig      streaming .brv format and compatible reader
src/safetensors.zig  memory-mapped safetensors I/O
src/pool.zig         worker threads and the reusable batch pool
src/report.zig       text and schema-4 JSON bench reporting
src/main.zig         CLI, argument handling, and orchestration
eval/                end-to-end multishard evaluation
autodsl/             self-extending grammar loop: mine, propose, verify, gate
doc/                 engineering principles and cluster notes
tools/model_cache.py fetch / verify / drop evaluation checkpoints one at a time
setup_env.sh         cluster modules, Zig toolchain, virtual environment
requirements.txt     Python dependencies (numpy, for eval/tensor_stats.py only)
```

[`doc/architecture.md`](doc/architecture.md) describes every module, the
dependency layering, and the invariants that span files.
[`doc/engineering-principles.md`](doc/engineering-principles.md) states the
design principles code in this repository is expected to follow.
[`doc/cluster-pitfalls.md`](doc/cluster-pitfalls.md) documents the environment
traps behind `setup_env.sh`.
[`doc/checkpoint_acquisition.md`](doc/checkpoint_acquisition.md) covers where the
evaluation inputs come from, how to verify them, and the storage gate that
applies before downloading.
[`doc/autodsl.md`](doc/autodsl.md) describes the self-extending grammar loop. `CLAUDE.md` is guidance for AI coding agents: the
search and archive invariants that are silent to break, and the evaluation gates.

## Correctness

Tests cover randomized round trips for every operator, program and archive round trips, uniform equivalence, search pruning, FP8 and u32 data types, malformed Huffman and rANS streams, legacy references, and serial and parallel decoding.

Large evaluations perform complete byte comparisons for fixed, uniform, and PHOG-guided plans, `--jobs 1`, configured parallel decoding, and every baseline.

## License

See [LICENSE](LICENSE).
