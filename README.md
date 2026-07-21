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

Operator applicability depends on input bit width, dtype, tree depth, arity, and the runtime operator mask. Repeating `--disable-op NAME` removes individual transforms or terminal codecs from search for controlled ablations. The CLI does not allow `raw` to be disabled; direct API callers also retain it as an implicit reversible fallback, so every supported tensor has at least one valid program. Fixed-plan mode deliberately ignores the search mask and continues to execute the same typed template.

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

`brevis bench --format json` separates tensor planning from block encoding time and records every tensor's dtype, shape, selected program, search expansions, realized and reranked candidate counts, selected sample rank, encoded bytes, and raw-fallback count. It also records the realized program and encoded bytes for every block, plus the applied prior's path, SHA-256 digest, and context counts. `program_node_depth` includes the terminal layer, whereas `program_transform_depth` counts only transform layers and matches `--max-depth`. These byte counts exclude archive frame headers and are intended for generated-DSL analysis; use complete `.brv` file sizes for storage comparisons.

The evaluator rebuilds the ReleaseFast binary so recorded source settings match the executable. Existing result files predate this schema and must be rerun before comparison with the current planner.

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
