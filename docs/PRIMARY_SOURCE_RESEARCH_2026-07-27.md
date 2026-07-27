# Primary-source research for Brevis DSL, search, compression, and systems work

Date: 2026-07-27

This note studies whether the current Brevis tensor-generating-program DSL has plausible value on real checkpoints, which small extensions are worth testing, how PHOG-guided A* should be evaluated, how the generic-compressor comparison should be configured, and which systems optimizations have primary-source support.

It does not report new Brevis measurements. External results establish plausible mechanisms, not gains on the Brevis corpus.

## Evidence labels

- **Source fact**: stated or implemented in a primary source such as a paper, official manual, specification, or upstream source tree.
- **Repository fact**: observed in the Brevis checkout on 2026-07-27.
- **Candidate**: a Brevis hypothesis that still requires a controlled experiment.

## Executive answer

There is good reason to test the existing abstract syntax more thoroughly before replacing it.

- **Source fact:** ZipNN finds that separating floating-point exponent fields and, for some FP32 checkpoints, byte groups exposes distributions that an entropy coder can exploit. Its implementation uses independent 256 KiB chunks for parallel work. See the [ZipNN paper](https://arxiv.org/pdf/2411.05239).
- **Source fact:** Bitshuffle and ndzip show that bit transposition, integer-domain reversible prediction, and cache-sized independent blocks can be effective and fast on typed numerical arrays. Their evidence is from scientific and radio data, not neural checkpoints. See the [Bitshuffle paper](https://arxiv.org/pdf/1503.00638), [pinned Bitshuffle implementation](https://github.com/kiyo-masui/bitshuffle/tree/526440a16baff44bd405e0741ebd285858a5408d), [ndzip paper](https://dps.uibk.ac.at/~fabian/publications/2021-ndzip-a-high-throughput-parallel-lossless-compressor-for-scientific-data.pdf), and [pinned ndzip implementation](https://github.com/celerity/ndzip/tree/ff4e6702bf0abb86d4aeef8249bd30344dfeef75).
- **Repository fact:** `MergeFloatFields`, `MergeBytePlanes`, `MergeBitPlanes`, and `Scan` already express the main transform families supported by those sources. `Literal` preserves total coverage when no transform helps.
- **Conclusion:** external work makes the current syntax plausible, but it does not show that Brevis selects these operators on real checkpoints or that they save net archive bytes after program and table overhead.

The recommended order is:

1. Add operator-level utilization and counterfactual measurements.
2. Sweep search breadth and PHOG guidance on a held-out, dtype-stratified checkpoint set.
3. Keep the current grammar unless a specific missing pattern appears.
4. If axis or stride correlations are measured, test one small generalization of `Scan`, not a broad collection of new mechanisms.
5. Optimize the measured hot paths. The strongest low-risk candidates are rANS reciprocal multiplication, allocation and scratch reuse, SIMD plane transforms and maps, a bounded per-tensor worker pipeline, and preallocated sequential I/O.

## 1. What the existing DSL already covers

| Brevis family | Primary-source evidence | Implication |
| --- | --- | --- |
| `MergeFloatFields` | ZipNN separates exponent bits from sign and fraction bits before entropy coding. It reports the benefit on BF16 and FP32 model data. The result is data dependent. [Paper](https://arxiv.org/pdf/2411.05239) | Retain it. Measure selection and net byte savings by dtype and tensor role. |
| `MergeBytePlanes` | ZipNN groups FP32 bytes into separate streams for checkpoints whose fraction bytes are compressible. [Paper](https://arxiv.org/pdf/2411.05239) | Retain it. It is already a simple form of byte transposition. |
| `MergeBitPlanes` | Bitshuffle transposes a bit matrix in roughly L1-sized blocks and uses SSE/AVX implementations. It explicitly requires correlations among nearby elements. [Paper](https://arxiv.org/pdf/1503.00638), [implementation](https://github.com/kiyo-masui/bitshuffle/tree/526440a16baff44bd405e0741ebd285858a5408d) | Retain it, but do not infer checkpoint benefit from radio/scientific data. Its interpreter is a strong SIMD target. |
| `ScanXor`, `ScanAddMod` | ndzip avoids floating-point subtraction and applies exact integer-domain prediction before bit transposition. fpzip also motivates neighborhood prediction for smooth multidimensional floating-point arrays. [ndzip paper](https://dps.uibk.ac.at/~fabian/publications/2021-ndzip-a-high-throughput-parallel-lossless-compressor-for-scientific-data.pdf), [pinned ndzip source](https://github.com/celerity/ndzip/tree/ff4e6702bf0abb86d4aeef8249bd30344dfeef75), [fpzip](https://computing.llnl.gov/projects/fpzip) | Retain the simple one-dimensional scans. Measure adjacency and shape-axis correlations before adding a multidimensional predictor. |
| `MapZigzag`, rotations, Gray code, bit reverse | ndzip uses exact integer recoding before bit transposition, and bit-transpose implementations rely on inexpensive bit operations. The sources do not establish that every Brevis map helps checkpoints. | Treat each map as an ablation target. Remove only after held-out measurements show no value and search cost is material. |
| `Constant`, `Repeat`, `Concat`, `Literal` | These are basic finite-sequence constructors. `Literal` makes every supported finite stream representable. | Keep this small base. Do not call it a proof of compression benefit or of practical search completeness. |

### Current theoretical boundary

**Repository fact:** `src/grammar.zig` has a finite, target-directed production registry with explicit hard caps. `src/dsl.zig` validates widths, lengths, arities, and parameters at construction. This supports the phrases **finite, bounded synthesis space** and **typed-by-construction**.

It does not support stronger claims that the implementation is a conventional statically typed language, that a budgeted run explores the whole bounded space, or that every returned program is globally optimal. `src/synthesizer.zig` distinguishes `proven_optimal` from `budget_exhausted`; the paper should preserve that distinction.

PHOG should affect ranking only. It should not create legal productions, remove the literal fallback, or decide the final exact archive-size comparison.

## 2. How to determine real DSL utilization

Counting production names in the selected AST is necessary but insufficient. An operator can appear while saving no net bytes, or it can be valuable but missed by the search budget.

For every tensor, record:

- checkpoint, tensor name, dtype, shape, byte size, and a preregistered tensor-role label;
- selected AST, node count, depth, and production-family counts;
- exact serialized program bytes, terminal-table bytes, terminal payload bytes, and total tensor-record bytes;
- terminal codec chosen for every literal leaf;
- search status, expansions, completed candidates, peak frontier, search time, and peak memory;
- compression and decompression CPU time and wall time;
- byte-exact round-trip status.

Add three counterfactuals:

1. **Terminal-only counterfactual:** encode the same tensor with no semantic transform and the same terminal-codec choice rules. Report `terminal_only_bytes - selected_program_bytes`.
2. **Leave-one-family-out search:** disable one production family while keeping the same search budget, prior, candidate objective, and terminal coders. Report the change in exact archive bytes, expansions, and time.
3. **One-step opportunity scan:** evaluate each legal top-level transform followed immediately by terminal coding. This separates “the transform had no opportunity” from “A* failed to reach the opportunity.”

Aggregate at both tensor and model level. Model-level totals are the primary storage result. Tensor-level distributions explain where the result comes from. Do not treat thousands of tensors from one checkpoint as thousands of independent model observations.

### Dataset design

Larger parameter count alone is not enough. Use a stratified corpus that changes architecture, tensor role, and representation:

- dense decoder, mixture-of-experts decoder, encoder or vision model, and diffusion model;
- small pilot files, complete medium files, and complete large or sharded models;
- BF16, F16, F32, FP8 E4M3, FP8 E5M2, I8/U8, and the scale tensors paired with quantized weights;
- embeddings, attention projections, MLP or expert weights, normalization parameters, biases, optimizer states where available, and quantization scales;
- final weights and multiple training checkpoints when the study explicitly includes checkpoint evolution.

Use tensor-level stratified pilots to find mechanisms cheaply, then verify all claimed ratios on complete files. A partial shard is evidence about that shard only.

Split PHOG calibration and evaluation by model or model family, not by randomly mixing tensors from the same checkpoint. This prevents near-duplicate tensor roles and shards from leaking into both sets.

## 3. A minimal grammar extension, only if measurements justify it

The strongest missing pattern suggested by scientific compressors is prediction along a tensor stride rather than only along flattened adjacent elements.

### Candidate: generalized strided scan

A compact form is:

```text
ScanStride[op, k](initials, deltas)
```

with the following static conditions:

- `op` is XOR or addition modulo the element width;
- `1 <= k < output_length`;
- `initials` has length `k`;
- `deltas` has length `output_length - k`;
- both children and the result have the same element width;
- for `i >= k`, output element `i` is reconstructed from output element `i-k` and the corresponding delta.

`k = 1` is the current first-order scan. Restrict candidate strides to a small, deterministic set derived from tensor shapes, plus a hard cap. PHOG predicts the production family; the target-directed grammar proposes only legal stride parameters.

This operator is:

- reversible in the integer bit domain;
- simple enough to explain as a tensor-generating program;
- compatible with typed-by-construction validation;
- parallel across residue classes and amenable to block-prefix execution;
- still only a **candidate**, because fpzip and ndzip operate on smooth scientific arrays and do not show that neural weight tensors have useful axis correlations.

Adopt it only if measured shape-axis mutual information or one-step exact-byte tests show held-out gains that exceed its program metadata and search cost.

### Extensions not justified yet

- **Cross-checkpoint XOR references:** the low-precision compression preprint and ZipNN discuss delta checkpoints, but this changes the artifact from a standalone checkpoint into a reference-dependent object. Keep it outside the main Brevis format unless the paper introduces and evaluates a clearly separate delta mode. [Low-precision preprint](https://arxiv.org/pdf/2508.19263)
- **General Lorenzo or arbitrary multidimensional predictors:** the ndzip and fpzip results are domain specific. A broad predictor family would expand the grammar and weaken the paper's simple mechanism.
- **Arbitrary affine maps or permutations:** no reviewed source here establishes checkpoint benefit, and unconstrained parameters increase branching.
- **A separate operator for chunking:** first use chunking as an execution and entropy-stream implementation detail. Promote it into the semantic DSL only if chunk boundaries materially improve exact size and must be represented for decoding.

## 4. Dtype expansion

The pinned safetensors source currently enumerates BOOL, F4, two F6 formats, U8/I8, several FP8 formats, 16-bit types, 32-bit types, C64, F64, I64, and U64. It marks the enum non-exhaustive and exposes bit sizes, including sub-byte widths. See the [pinned dtype definition](https://github.com/safetensors/safetensors/blob/6eb4dc9a28ebce297606e0f4836bbf28839cacef/safetensors/src/tensor.rs#L808-L901).

**Repository fact:** Brevis currently supports F16, BF16, F32, U8/U16/U32, I8/I16/I32, F8_E4M3, and F8_E5M2. Its stream words are at most 32 bits.

Recommended sequence:

1. **Low-risk coverage:** add BOOL and byte-aligned FP8 encodings as opaque 8-bit words. Only give a dtype `float_fields` semantics when its exact bit layout is implemented and tested.
2. **Corpus-driven coverage:** add a dtype only when an evaluation checkpoint contains it. Unsupported but common metadata should be reported explicitly rather than silently reinterpreted.
3. **64-bit project:** F64, I64, U64, and C64 require a coherent 64-bit stream, DSL parameter, interpreter, terminal-codec, and wire-format design. They are not a one-line enum extension.
4. **Sub-byte project:** F4 and F6 need bit-addressed element handling. The safetensors implementation rejects tensor views whose total bit count is not byte aligned, which shows why physical byte storage and logical element indexing must be specified carefully. See [`TensorView::new`](https://github.com/safetensors/safetensors/blob/6eb4dc9a28ebce297606e0f4836bbf28839cacef/safetensors/src/tensor.rs#L748-L769).

The 2025 low-precision preprint reports skewed FP8 exponent distributions in its tested models, while its tested packed FP4 values were close to uniform and the associated scale factors were more compressible. Treat this as narrow preprint evidence, not a universal property. It supports testing FP8 and scale tensors first. [Preprint](https://arxiv.org/pdf/2508.19263)

## 5. PHOG and wider A* search

### What the original sources establish

The original PHOG conditions productions on context obtained by navigating the partially generated AST, rather than only on the parent nonterminal. It learns the context function and then estimates rule probabilities by counting. It also warns through its model-selection setup that larger contexts face data sparsity. See the [PMLR paper](https://proceedings.mlr.press/v48/bielik16.pdf).

The synthesis work based on PHOG:

- assigns production edges cost `-log2(probability)`;
- constructs the sentential-form graph on the fly;
- derives a lower-bound heuristic for A* from production probabilities;
- uses equivalence-class pruning while retaining the most likely representative in its CEGIS setting;
- introduces transfer learning to reduce overfitting across synthesis specifications.

See [Accelerating Search-Based Program Synthesis using Learned Probabilistic Models](https://prosys.kaist.ac.kr/publications/pldi18b.pdf) and its [DOI record](https://doi.org/10.1145/3192366.3192410).

### How this maps to Brevis

**Repository fact:** `src/grammar_prior.zig` uses a fixed, hand-designed context containing parent, child slot, depth, dtype, target width, length, zero fraction, distinct fraction, repetition, and difference-entropy buckets, with deterministic backoff and smoothing.

This is a defensible **PHOG-inspired contextual prior**. It is not the original paper's learned AST-navigation context function. The paper should use the narrower description unless that function-learning step is implemented and evaluated.

The fixed features may be preferable for this paper because they are small, interpretable, and checkpoint-specific. More context is not automatically better.

### Required PHOG evaluation

Compare at least:

- literal or raw terminal baseline;
- fixed hand-designed policy if one exists;
- uniform grammar costs;
- contextual PHOG-inspired prior.

For uniform and PHOG:

- use the identical legal grammar;
- use identical parameter proposals, terminal coders, exact-byte candidate objective, and search budgets;
- change only queue guidance;
- train or calibrate only on the training split;
- report held-out production negative log likelihood, expansions to first complete candidate, expansions to best candidate, total search time, best exact bytes, and status.

PHOG is useful if it reaches equal or smaller exact programs with fewer expansions or less time on held-out checkpoints. A better rule-prediction loss alone is not enough.

### Broader search protocol

Use geometric budget sweeps so scaling is visible. A practical starting grid, based on the current defaults, is:

- expansions: 512, 2,048, 8,192, 32,768;
- maximum nodes: 64, 96, 128;
- maximum depth: 4, 5, 6;
- parameter proposals: current defaults, then individually widened caps, then selected combinations up to the existing hard caps.

First vary one dimension at a time. Run the full selected combinations only after eliminating inactive dimensions. For every point, report exact bytes, elapsed time, peak memory, expansions, complete candidates, and `proven_optimal` or `budget_exhausted`.

Use a fixed time-limited wide run on a stratified tensor sample as an empirical oracle. It is an oracle relative to that run, not a proof of global optimality.

### Safe pruning candidate

The PLDI paper's equivalence relation is tied to its CEGIS semantics and should not be copied blindly.

A Brevis-specific candidate is a transposition table keyed by the complete future-relevant state:

- ordered open-hole target bytes and stream types;
- parent production, child slot, and depth for every hole;
- dtype and any other PHOG context feature;
- filled-node count and remaining structural limits.

Retain a Pareto frontier over exact fixed bytes and accumulated grammar cost. Prune only when one state is no worse in every future-relevant resource. A proof and differential search tests are required before this can preserve `proven_optimal`.

## 6. Generic-compressor comparison

The current three-profile matrix in `eval/benchmarking.py` is well grounded in official manuals:

| Compressor | Speed | Explicit default | Ratio | Source-backed note |
| --- | --- | --- | --- | --- |
| gzip | `-1` | `-6` | `-9` | GNU gzip defines `-1` as fastest, `-9` as best, and `-6` as default. [`gzip` manual](https://www.gnu.org/software/gzip/manual/gzip.html) |
| bzip2 | `-1` | `-9` | `--best` | Blocks range from 100 KiB to 900 KiB; 900 KiB is default. `--best` aliases `-9`, so the default and ratio rows are one independent configuration, not replication. [`bzip2` manual](https://sourceware.org/bzip2/manual/manual.html) |
| xz | `-1 -T1` | `-6 -T1` | `-9e -T1` | `-6` is default. Extreme mode may improve ratio but can be much slower and is not guaranteed to improve every input. [`xz` manual](https://tukaani.org/xz/man/xz.1.html) |
| Zstandard | `-1` | `-3` | `-19` | Levels 1 through 19 are normal, default is 3. `--single-thread` serializes I/O and compression and is different from `-T1`. [Pinned CLI manual](https://github.com/facebook/zstd/blob/5c7b7bad26808e6b40ac3b3d0075466e27738a9d/programs/zstd.1.md) |
| LZ4 | `--fast=5` | `-1` | `-12` | Level 1 is the recommended default and `--best` is level 12. [Pinned CLI manual](https://github.com/lz4/lz4/blob/0774d05537f9762f838f7ab541b7765f1a729cb5/programs/lz4.1.md) |
| Brotli | `-q 1` | `-q 11` | `-q 11 -w 24` | The official API defines default and maximum quality as 11, default window as 22, and maximum standard window as 24. [Official encoder API](https://brotli.org/encode.html) |

### One missing ratio ceiling

Zstandard `-19` is not its maximum ratio setting. Add a separately labeled **ceiling**, not a replacement for the preregistered ratio profile:

```text
zstd --single-thread --no-asyncio --ultra -22
```

Optionally test `--long` in another clearly named configuration with an explicit memory limit. The official manual says `--ultra` unlocks levels 20 through 22 with much more memory, `--max` is extremely slow and resource intensive, and `--long` increases compressor and decompressor memory. These are ceiling experiments, not ordinary operating points.

### Fairness rules

- Compress the exact complete safetensors file and charge every frame, checksum, table, program, and archive byte.
- Hash and byte-compare the complete reconstructed file.
- Pin executable version, executable hash, command line, thread setting, host, and input hash.
- Keep compression and decompression timing separate.
- Report raw input bytes divided by operation time.
- Preserve speed, explicit-default, ratio, and ceiling profiles as different configurations.
- Keep a serial track and a separate matched-worker scaling track.
- In the parallel track, use each tool's native threading where available and record actual worker count and memory.
- Do not rename pigz or pbzip2 as gzip or bzip2. They are separate programs and should appear as separate methods if included.
- Do not mix Zstandard `--single-thread` with `-T1`; the official manual says their I/O overlap and compressed output differ.

No source supports a promise that Brevis will match every general-purpose compressor on every checkpoint. Report where it wins and loses in ratio, compression speed, decompression speed, and memory.

## 7. Engineering optimization priorities

### Priority 0: profile the complete pipeline

Before changing algorithms, measure cycles and bytes for:

- safetensors parsing and archive I/O;
- target decomposition and A*;
- histogram construction and table normalization;
- Huffman and rANS encode/decode;
- every map, scan, and merge interpreter;
- allocation, copying, hashing, and output buffering;
- worker idle time and maximum resident memory.

Use release builds and record compiler version, target, and CPU features. Zig 0.16 states that vector operations normally lower to SIMD when supported and fall back to element-wise execution otherwise. [`@Vector` documentation](https://ziglang.org/documentation/0.16.0/#Vectors)

### Priority 1: no semantic or wire-format change

1. **Replace rANS division with exact reciprocal multiplication.** The pinned `ryg_rans` reference precomputes `x_max`, reciprocal frequency, bias, complement frequency, and shift so its fast encoder uses multiply-high instead of integer division. [Pinned source](https://github.com/rygorous/ryg_rans/blob/c9d162d996fd600315af9ae8eb89d832576cb32d/rans_byte.h#L1105-L1331)

   Brevis currently performs division and remainder per rANS symbol. A compatible implementation may preserve identical bytes if it produces the exact same quotient and state transition. Verify it differentially against the old encoder across every normalized frequency, boundary state, and randomized stream before removing the reference path.

2. **Reuse decoder and histogram scratch memory.** Avoid allocator traffic for the cumulative-to-symbol table, direct symbol lookup arrays, histograms, transform buffers, and per-worker temporary storage. Keep scratch local to a worker to avoid false sharing.

3. **SIMD the width-preserving kernels.** Use `@Vector` for XOR, modular add, rotations, byte extraction, byte merge, and suitable bit-plane transpose stages. Use explicit scalar tails. The Bitshuffle implementation is evidence that a blocked SSE/AVX transpose can be fast, not evidence of a Brevis speedup.

4. **Block interpreter execution for cache locality.** Pointwise maps and merges can process cache-sized ranges without changing whole-tensor program semantics. Keep block size an implementation parameter and tune it on held-out hardware.

5. **Parallelize independent tensors with a bounded reorder buffer.** Workers synthesize and encode tensors independently; one ordered writer emits canonical archive order. Bound completed-but-not-written bytes to cap memory and apply backpressure.

6. **Use mmap for read views and preallocate output when size is known.** The pinned safetensors implementation shows [mmap-backed parsing](https://github.com/safetensors/safetensors/blob/6eb4dc9a28ebce297606e0f4836bbf28839cacef/safetensors/src/tensor.rs#L427-L443) and [an output path](https://github.com/safetensors/safetensors/blob/6eb4dc9a28ebce297606e0f4836bbf28839cacef/safetensors/src/tensor.rs#L301-L335) that sets final length, writes through a 1 MiB buffer, and atomically renames a sibling temporary file. On macOS it also enables `F_NOCACHE`, with an upstream source comment reporting a source-specific improvement.

7. **Use positional I/O for truly parallel output.** `pread` and `pwrite` do not mutate the shared file offset and are intended for multithreaded access. Preallocate the exact output extent first. [`pread(2)` and `pwrite(2)`](https://man7.org/linux/man-pages/man2/pread.2.html), [`fallocate(2)`](https://man7.org/linux/man-pages/man2/fallocate.2.html)

### Priority 2: format-versioned speed work

1. **Two- or four-state interleaved rANS.** The reference rANS implementation states that independent states can share one byte stream without extra per-symbol signaling and can expose instruction-level parallelism on superscalar out-of-order CPUs. It costs a few state bytes. [Pinned source](https://github.com/rygorous/ryg_rans/blob/c9d162d996fd600315af9ae8eb89d832576cb32d/rans_byte.h#L883-L943)

   This changes the payload format. Version it, compare exact size, and report single-core encode and decode cycles per byte.

2. **Independent entropy chunks for large tensors.** ZipNN uses 256 KiB chunks and parallelizes across chunks and byte groups. ndzip also uses small independent blocks. Chunking permits parallel entropy coding and random access but repeats tables or state and may reduce ratio. Sweep chunk sizes and charge all metadata.

3. **Parallel scans.** Implement block-local XOR or modular scans, compute block prefixes, and apply them in a second pass. This preserves exact semantics but is worthwhile only for sufficiently large scan nodes.

4. **Versioned 64-bit and sub-byte support.** Treat these as coherent format changes, not incidental parser patches.

### Priority 3: optional, platform-specific experiments

- **Page-cache advice:** for a one-pass mapped input, test `MADV_SEQUENTIAL` and possibly `MADV_DONTNEED` after consumption. On Linux these are performance hints and do not guarantee behavior. [`madvise(2)`](https://man7.org/linux/man-pages/man2/madvise.2.html)
- **File advice:** test `POSIX_FADV_SEQUENTIAL` or `POSIX_FADV_NOREUSE` for large streaming files. Linux only gave `NOREUSE` operational semantics again in 6.3, so record kernel version. [`posix_fadvise(2)`](https://man7.org/linux/man-pages/man2/posix_fadvise.2.html)
- **Direct I/O:** Linux `O_DIRECT` bypasses the page cache but imposes filesystem and alignment constraints and may fall back or fail. It is an opt-in benchmark candidate, not a default optimization. [Linux kernel direct-I/O documentation](https://www.kernel.org/doc/html/latest/filesystems/iomap/operations.html#direct-i-o)
- **Software prefetch:** Zig's `@prefetch` is a no-op when unsupported and affects only performance. Add it only after a measured cache-miss problem. [`@prefetch` documentation](https://ziglang.org/documentation/0.16.0/#prefetch)
- **NUMA-aware workers:** ZipNN reports better scaling from multiple workers confined to NUMA nodes on its 224-core host. This motivates an affinity experiment on comparable multi-socket hardware, not a general Brevis performance claim. [ZipNN paper](https://arxiv.org/pdf/2411.05239)

Benchmark warm-cache CPU throughput separately from cold or streaming end-to-end throughput. Do not drop system caches or enable direct I/O for only one method in a comparison.

## 8. Clean implementation constraints

Performance work can remain small and auditable:

- one scalar reference kernel and one selected fast kernel per transform;
- feature selection at build time or a small dispatch boundary;
- reusable per-worker scratch rather than many defensive wrappers;
- bounded queues and explicit ownership rather than global mutable caches;
- comments only where an invariant, wire compatibility rule, or non-obvious arithmetic derivation needs explanation;
- delete old paths only after differential tests and benchmark evidence.

Do not delete raw data, evaluation manifests, or documentation merely because it is unused by a hot path. Remove generated or obsolete artifacts only when provenance and reproducibility remain intact.

## 9. Decision gates

### Keep an existing production when

- it saves net bytes on at least one held-out model stratum, or
- it is part of the small completeness base and has negligible search cost.

### Remove an existing production when

- it has zero one-step opportunity and zero selected net savings across the preregistered held-out corpus;
- leave-one-family-out search is no worse in exact bytes;
- removal materially reduces search or implementation cost.

### Add a production when

- a measured residual pattern is not expressible compactly by the existing grammar;
- a one-step prototype beats terminal-only encoding after all metadata;
- held-out search improves model-level bytes under a fixed resource budget;
- well-formedness and reversibility are local and easy to validate;
- PHOG still ranks only legal productions and the literal fallback remains available.

### Accept a systems optimization when

- byte-exact output or a deliberate format-version change is documented;
- the benchmark isolates compression, decompression, and I/O effects;
- single-core and matched-worker results are both reported where relevant;
- peak memory and compressed size do not silently regress.

## 10. Paper claim guardrails

Until the measurements exist, the paper can say:

- the DSL contains transform families motivated by prior lossless compression work;
- programs are validated and generated in a finite, bounded search configuration;
- the contextual prior is PHOG-inspired and guides A* ordering;
- exact serialized size selects among completed candidates;
- a budgeted search may terminate without exhausting its configured frontier.

It should not yet say:

- the non-literal syntax is widely used on real checkpoints;
- PHOG improves search on held-out models;
- a larger search budget improves compression;
- SIMD, multithreading, or direct I/O reaches a particular throughput;
- Brevis matches or beats a general-purpose compressor;
- a budget-exhausted result is globally optimal.

Every final quantitative sentence should point to a committed configuration, complete result record, exact input hash, and byte-exact verification.

## Primary sources

- Bielik, Raychev, and Vechev. [PHOG: Probabilistic Model for Code](https://proceedings.mlr.press/v48/bielik16.pdf). ICML 2016.
- Lee, Heo, Alur, and Naik. [Accelerating Search-Based Program Synthesis using Learned Probabilistic Models](https://prosys.kaist.ac.kr/publications/pldi18b.pdf). PLDI 2018.
- Hershcovitch et al. [ZipNN: Lossless Compression for AI Models](https://arxiv.org/pdf/2411.05239).
- Heilper and Singer. [Lossless Compression of Neural Network Components in Low-Precision Formats](https://arxiv.org/pdf/2508.19263). arXiv v1 preprint.
- Masui et al. [A Compression Scheme for Radio Data in High Performance Computing](https://arxiv.org/pdf/1503.00638) and the [pinned Bitshuffle source](https://github.com/kiyo-masui/bitshuffle/tree/526440a16baff44bd405e0741ebd285858a5408d).
- Knorr et al. [ndzip: A High-Throughput Parallel Lossless Compressor for Scientific Data](https://dps.uibk.ac.at/~fabian/publications/2021-ndzip-a-high-throughput-parallel-lossless-compressor-for-scientific-data.pdf), [DOI](https://doi.org/10.1109/DCC50243.2021.00018), and [pinned implementation](https://github.com/celerity/ndzip/tree/ff4e6702bf0abb86d4aeef8249bd30344dfeef75).
- LLNL. [fpzip official project page](https://computing.llnl.gov/projects/fpzip).
- Duda. [Asymmetric Numeral Systems](https://arxiv.org/pdf/1311.2540) and Giesen's [pinned byte-rANS reference](https://github.com/rygorous/ryg_rans/blob/c9d162d996fd600315af9ae8eb89d832576cb32d/rans_byte.h).
- safetensors. [Pinned tensor format implementation](https://github.com/safetensors/safetensors/blob/6eb4dc9a28ebce297606e0f4836bbf28839cacef/safetensors/src/tensor.rs).
- Zig. [0.16.0 language reference](https://ziglang.org/documentation/0.16.0/).
- GNU gzip. [Official manual](https://www.gnu.org/software/gzip/manual/gzip.html).
- bzip2. [Official manual](https://sourceware.org/bzip2/manual/manual.html).
- XZ Utils. [Official `xz` manual](https://tukaani.org/xz/man/xz.1.html).
- Zstandard. [Pinned CLI manual](https://github.com/facebook/zstd/blob/5c7b7bad26808e6b40ac3b3d0075466e27738a9d/programs/zstd.1.md) and [official API manual](https://facebook.github.io/zstd/doc/api_manual_latest.html).
- LZ4. [Pinned CLI manual](https://github.com/lz4/lz4/blob/0774d05537f9762f838f7ab541b7765f1a729cb5/programs/lz4.1.md).
- Brotli. [Official encoder API](https://brotli.org/encode.html).
- Linux kernel and man-pages project. [`madvise(2)`](https://man7.org/linux/man-pages/man2/madvise.2.html), [`posix_fadvise(2)`](https://man7.org/linux/man-pages/man2/posix_fadvise.2.html), [`pread(2)` and `pwrite(2)`](https://man7.org/linux/man-pages/man2/pread.2.html), [`fallocate(2)`](https://man7.org/linux/man-pages/man2/fallocate.2.html), and [direct I/O](https://www.kernel.org/doc/html/latest/filesystems/iomap/operations.html#direct-i-o).
