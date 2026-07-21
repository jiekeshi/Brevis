# Brevis

Brevis provides bit-exact lossless compression for neural-network tensors.

Instead of using a fixed, hand-written codec, it synthesizes a short program for each tensor in a finite, typed language whose operations are reversible by construction.

An archive contains the program, leaf data, and required metadata. Decompression performs no search; it executes the stored program to reconstruct the original safetensors file exactly.

Brevis uses Zig 0.16.

## Quick Start

```bash
zig build -Doptimize=ReleaseFast

# Optional: calibrate a grammar prior from the current input
./zig-out/bin/brevis calibrate model.safetensors prior.bin

# Search with the learned prior; omit --prior for uniform A*
./zig-out/bin/brevis compress model.safetensors model.brv --plan search --prior prior.bin

# Fixed typed-DSL program baseline (no search)
./zig-out/bin/brevis compress model.safetensors fixed.brv --plan fixed
./zig-out/bin/brevis decompress model.brv restored.safetensors
./zig-out/bin/brevis verify model.brv model.safetensors

# Set the thread count or number of calibration tensors
./zig-out/bin/brevis compress model.safetensors model.brv --jobs 12
./zig-out/bin/brevis calibrate model.safetensors prior.bin --tensors 200
```

```bash
zig build test -Doptimize=ReleaseFast
python3 -m unittest discover -s eval -p 'test_*.py'
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

Operator applicability depends on input bit width, dtype, tree depth, and arity. `raw` is always available as a reversible fallback, so every supported tensor has at least one valid program.

## Grammar-Guided A* Search

Brevis uses a probabilistic higher-order grammar (PHOG) to order the search. It maintains two independent costs:

- `p` is the grammar description length. It only determines which partial program A* expands first.
- `g_bytes + lowerBound` is a lower bound on serialized bytes for the stream being searched.

PHOG does not change the compression objective. It only changes candidate order within a bounded search budget. Without a prior, Brevis uses a uniform distribution so the prior's contribution can be measured directly.

Single-stream search can prune with the serialized-byte bound; tensor planning instead collects candidates in grammar order under the 256-expansion budget without sample-incumbent byte pruning, because it ultimately compares them on representative full blocks.

Evaluation separates three modes: `fixed` executes a dtype-specific program expressed in the same DSL, `uniform` runs A* without a learned prior, and `phog` runs the same A* search with an input-local prior. Comparing them isolates the value of the reversible language, adaptive search, and grammar guidance.

Empirical entropy is not a valid lower bound at shallow holes where transforms remain available, because a reversible transform may reduce it sharply.

At those holes, the bound includes only required terminal-frame costs. A Shannon payload bound is added only after the maximum transform depth is reached.

### Input-Local Calibration

Calibration requires no external corpus. It stratifies samples by dtype and tensor size, searches four contiguous windows per tensor, then reranks the top eight candidates on four representative blocks using actual encoded bytes.

Winning programs become `(context, production)` counts. A context captures tree position, dtype, bit width, and bucketed entropy, zero rate, and delta features. Three-level backoff reduces sparsity.

The final prior is mixed with the uniform distribution to avoid overfitting sparse statistics.

## Archive and Decoding

The current writer produces schema 6 archives. Each block frame stores program bytecode, side information, and payload. The footer stores tensor names, dtypes, shapes, block counts, and the original safetensors header.

New archives contain no cross-block back-references, keeping every frame independent and easy to parallelize. The reader still supports legacy schema 5/6 back-references and rejects forward references, self-references, and invalid lengths.

Single-threaded decoding runs directly on the calling thread. Multithreaded mode reuses a fixed worker pool and decodes the next batch while the current batch is written.

Output order and per-tensor lengths are checked before and after writing.

## Evaluation

[`eval/run_eval.py`](eval/run_eval.py) evaluates fixed, uniform, and PHOG-guided plans together with gzip, Zstandard, xz, and OpenZL. New result files record the Git commit and dirty state, binary and evaluation-script hashes, each input hash and their ordered aggregate, the configured model manifest, per-shard prior hashes, thread counts, search settings, and the command mode behind every result field.

The evaluator rebuilds the ReleaseFast binary so recorded source settings match the executable.

### Recorded 8B Run

The five bfloat16 shards of Qwen3-8B-Base total 16,381,516,776 bytes. Full results on a 12-core Apple M4 Pro are:

| Search mode | Archive size | Compression time |
| --- | ---: | ---: |
| PHOG, first use | 10,922,326,421 bytes | 8.19 s calibration + 11.04 s compression |
| PHOG, reused prior | 10,922,326,421 bytes | 11.04 s |
| Uniform | 10,922,803,714 bytes | 24.60 s |

Bit-exact decompression takes 6.49 seconds with 12 threads and 40.41 seconds with one thread.

In this run, PHOG produced an archive 477,293 bytes smaller than uniform search. This single workload does not establish a stable compression or search-time improvement; the effect of grammar guidance is workload-dependent and must be evaluated against both uniform search and the fixed-program baseline.

Per-shard results are in [`eval/results-large.json`](eval/results-large.json). This historical snapshot predates the provenance schema used by new runs and should not be combined with newly generated measurements without rerunning it.

## Source Layout

```text
src/types.zig        dtypes, Stream, TensorView, and block planning
src/ops.zig          reversible language operators
src/program.zig      program execution, inversion, and serialization
src/search.zig       PHOG-guided A* and byte lower bounds
src/prior.zig        contexts and three-level backoff prior
src/calibrate.zig    input-local sampling and prior fitting
src/codec.zig        bitpack, Huffman, and rANS codecs
src/archive.zig      streaming .brv format and compatible reader
src/safetensors.zig  memory-mapped safetensors I/O
src/main.zig         CLI, batching, and parallel pipeline
eval/                end-to-end multishard evaluation
```

## Correctness

Tests cover randomized round trips for every operator, program and archive round trips, uniform equivalence, search pruning, FP8 and u32 data types, malformed Huffman and rANS streams, legacy references, and serial and parallel decoding.

Large evaluations perform complete byte comparisons for fixed, uniform, and PHOG-guided plans, `--jobs 1`, configured parallel decoding, and every baseline.

## License

See [LICENSE](LICENSE).
