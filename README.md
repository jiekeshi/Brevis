# Brevis — Weight Compression via Program Synthesis

Bit-exact lossless compression of neural network weights. Each tensor is
compressed by **synthesizing a short program** in a small DSL of reversible
transforms; the program plus the entropy-coded leaf data *is* the compressed
artifact. Decompression replays the program in reverse — bit-exact by
construction.

Implemented in **Zig 0.16**.

## Build & run

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/brevis demo                                # synthetic tensors
./zig-out/bin/brevis make-fixture model.safetensors      # writes a small fp16 model
./zig-out/bin/brevis bench       model.safetensors       # per-tensor synthesis report
./zig-out/bin/brevis baseline    model.safetensors       # brevis vs gzip vs zstd
./zig-out/bin/brevis compress    model.safetensors model.brv
./zig-out/bin/brevis decompress  model.brv          recovered.safetensors
./zig-out/bin/brevis verify      model.brv          model.safetensors
zig build test                                            # unit tests (16 currently)
```

## Real-model benchmark: TinyLlama-1.1B (BF16, 2.05 GB)

```
                       size (bytes)        ratio       savings    wall time
                       -------------       --------    --------   --------------
raw safetensors          2,200,119,864     1.000×       0.0 %     —
gzip -9                  1,748,045,335     1.259×      20.6 %     2 min     (1 core)
zstd -3                  1,716,676,668     1.281×      21.9 %     1.8 sec
zstd -19                 1,669,393,188     1.318×      24.1 %     4 min     (1 core)
brevis bench (synth)     1,451,860,876     1.515×      34.0 %     1.69 sec  (12 cores, mmap input)
brevis (.brv archive)    1,451,869,414     1.515×      34.0 %     ~24 sec   (incl. write + e2e verify)

brevis verified 201/201 tensors bit-exact (`./zig-out/bin/brevis verify model.brv model.safetensors`)
```

**brevis is now in the same speed class as zstd -3 while compressing 18 % better.**
Earlier versions of brevis took 5 min wall on this model — see
[Performance journey](#performance-journey) below for what changed.

**brevis beats zstd-19 by ~14.9 % and gzip-9 by ~20.3 %** on TinyLlama. Per
tensor type, brevis picks different programs:

| tensor type | typical program | typical ratio |
| --- | --- | --- |
| LayerNorm `gamma` (4 KB BF16, near-1) | `split(rans,rans,bp+rans)` | 1.8–2.8× |
| Attention proj (8 MB BF16) | `split(huff,rans,bp+rans)` | 1.50× |
| MLP proj (23 MB BF16) | `split(huff,rans,rans)` | 1.52× |
| Embedding / lm_head (131 MB BF16) | `split(huff,rans,rans)` | 1.51× |

**Why brevis wins on weights**: gzip and zstd are general-purpose byte
compressors that don't know that the bytes are fp16/bf16 floats. brevis's
DSL explicitly splits the float into (sign, exponent, mantissa) and applies
a different encoder per component — exponent gets bitplane-split, mantissa
goes to rANS, etc. On generic data (text, images, mixed bytes) zstd
dominates easily; on numeric tensors the structure-aware approach pays off.
The gap widens on skewed distributions: a single LayerNorm gamma vector
compresses 1.8–2.8× standalone.

## Baseline comparison (synthetic fixture)

`brevis baseline` runs gzip and (if installed) zstd over the *raw tensor
bytes* of the input safetensors file and reports them next to brevis. On
the small synthetic fp16 fixture (`brevis make-fixture`):

```
                       size (bytes)        ratio
                       -------------       --------
raw                          2,760,704     1.000x  (baseline)
gzip -9                      2,548,154     1.083x  (DEFLATE, in-process)
zstd -3                      2,533,356     1.090x  (zstd default)
zstd -19                     2,537,473     1.088x  (zstd best-effort)
brevis (.brv archive)        2,330,874     1.184x  (synth + entropy + shared codebooks)
```

(Real models compress harder than this synthetic fixture because trained
weights have stronger structural regularities — see TinyLlama numbers above.)

## DreamCoder-style abstraction from low-level primitives (MVP)

There's a separate experimental track that goes one level deeper. Instead of
the hand-designed DSL (`split_float`, `bitplane_split`, `delta_encode`, ...),
this track starts from a **low-level reversible primitive library** and
**discovers** higher-level ops automatically.

### Files

- [src/lowlevel.zig](src/lowlevel.zig) — 10 low-level reversible primitives + execution VM
  (xor_const, add_const_mod, rotate_bits, bit_swap_pair, xor_prev, prefix_xor,
  diff_mod, cumsum_mod, split_field, terminals). All by-construction reversible;
  property tests (7 of them) verify `inverse(forward(x)) == x` on random inputs.
- [src/astar.zig](src/astar.zig) — branch-and-bound program synthesis with
  Shannon-entropy admissible heuristic + best-so-far pruning over the
  low-level grammar. (Strictly B&B not classic A*, but uses the A* spirit.)
- [src/lowlevel_training.zig](src/lowlevel_training.zig) — streaming training:
  for each input tensor, run B&B → walk the resulting tree → update PHOG
  pair counts + per-subtree counts → MDL-promote subtrees whose
  `count × (size−1) − macro_def_cost > 0`.
- CLI: `brevis train-lowlevel <model.safetensors> <out.json>`

### Example output (TinyLlama mini, 5 tensors, ~6 min on 1 thread)

```
train-lowlevel: 5 tensors picked (size ≤ 10M elem) for B&B training
training done in 343976ms wall
  n_tensors processed:    5
  avg compression ratio:  1.467×    (low-level B&B vs high-level fast-path's 1.515×)
  unique subtrees seen:   5
  subtrees promoted:      1

Promoted macro:
  size=3, count=3, MDL benefit=+1
  decoded: split_field(start=4, n_bits=4, k=8) → (rans, rans)
  meaning: "for 8-bit streams, split into two 4-bit nibbles and rans-encode each"
```

The system **automatically discovered** that splitting 8-bit streams into
nibbles + rans on each is a winning strategy — without any hand-coded
knowledge of `split_field` arity or the bf16/fp16 layout.

### Status: MVP

What works:
- ✅ Low-level primitives compile, are reversible, pass property tests
- ✅ B&B search finds non-trivial programs (1.467× on small bf16 tensors)
- ✅ Streaming training pipeline runs end-to-end
- ✅ Macros emerge with positive MDL benefit
- ✅ JSON report shows promoted macros + PHOG pair counts

What's NOT yet integrated:
- ⛔ Discovered macros are **not** auto-emitted as Zig source (the user
  reads the JSON and decides what to add manually)
- ⛔ Runtime A* doesn't use the learned PHOG over the dynamic grammar
- ⛔ B&B search is single-threaded and limited to depth 4 (slow on big tensors)

These are all "wire it back into runtime" steps left for future iteration.
The current MVP demonstrates **end-to-end discovery**: from primitive
reversible ops + raw tensors → emergent abstractions + measurable MDL gain.

### Ratio gap explanation

1.467× (low-level) vs 1.515× (high-level fast-path) on the same tensors —
the gap is because:
- Low-level B&B is depth-limited to 4 layers (shallower than what high-level
  ops can express — `split_float >> bitplane_split >> huffman` is naturally 3
  layers but each layer here is more powerful than a single low-level op)
- Action space is hard-limited (only a few `xor_const` constants tried, etc.)
- No shared codebooks across tensors yet in the low-level path

Both gaps would close with more compute budget (deeper search) and more
training data (more tensors → better MDL signal).

## PHOG learned fast-path

Earlier versions used a hand-tuned `fastPathShape(numel)` decision tree
(thresholds picked from eyeballing TinyLlama bench data). That's now
replaced by a real PHOG (Probabilistic Higher-Order Grammar):

```
                                       ┌── tools/train_phog.py
                                       │   (offline, run-once)
brevis collect-training X.safetensors  │
        │                              ▼
        │   training.jsonl   ─→   src/phog.zig (Zig const lookup table)
        │   (one record per                                │
        │    context/production)                           │
        │                                                  │
        ▼                                                  ▼
  oracle: full A* search                          runtime predictRoot()
  with realize_top_k = 32                         + predictStream()
  per tensor                                      → TensorShape directly
                                                  → compress
```

Productions modeled (12 total):
* T_PROG: `tensor_raw`, `split_float`, `tensor_xor`
* S_PROG (per stream slot): `raw / huffman / rans` × `terminal / delta+ / bitplane+`

Context features (deliberately cheap — no per-stream stats needed):
* position ∈ {root, split.sign, split.exp, split.mant}
* dtype ∈ {f16, bf16}
* log₂(numel) bucket ∈ [0, 8)
* ndim bucket ∈ {1, 2, 3+}

Total: 4 × 2 × 8 × 3 = 192 contexts. Each maps to a fixed-point `-log₂ p`
table over the 9 (or 3) productions for that position. Contexts not seen
during training stay uniform (Laplace `α=1`) → runtime detects the uniform
case and falls back to A* search.

After training on TinyLlama (201 BF16 tensors → 804 (context,production)
examples → 16 populated contexts), inference is **constant-time argmin over
≤ 9 u32s** — essentially free. Compression ratio matches the hand-tuned
fast-path exactly (1.515× both); the value of PHOG is that you can
**re-train it on a new model** in seconds without touching code:

```bash
./zig-out/bin/brevis collect-training new-model.safetensors training.jsonl
python tools/train_phog.py training.jsonl src/phog.zig
zig build -Doptimize=ReleaseFast
```

Future work: enrich the context with per-stream features (entropy, unique
count) so PHOG can distinguish e.g. LayerNorm-gamma from a same-size
attention bias — currently PHOG bins them together by `numel`.

## Performance journey

TinyLlama-1.1B wall time as each optimization landed (12-core M-class):

| step | wall | factor | ratio |
| --- | --- | --- | --- |
| 1. baseline (single-thread, full A\* search, per-candidate verify) | ~60 min | 1×  | 1.515× |
| 2. + 12-thread tensor-level parallelism (`std.Thread.spawn` + atomic counter, `smp_allocator`) | 3 min 36 s | 17× | 1.515× |
| 3. + faster inner loops (canonical Huffman / rANS encoders use direct-indexed arrays, `BitWriter` accumulates 64 bits) | 3 min 30 s | 17× | 1.515× |
| 4. + skip per-candidate verify; one end-to-end verify before writing instead | 3 min   | 20× | 1.515× |
| 5. + tighten search: `realize_top_k = 4`, drop bitplane on tensors with >200 K mantissa elements | — | — | — |
| 6. + tensor-type **fast-path** (no search at all when shape/size matches a profile) | **3.6 s** | 1000× | 1.515× |
| 7. + SIMD-style **histogram** (4 parallel counter arrays in `huffmanBuild` / `ransBuild`) | 2.8 s | 1300× | 1.515× |
| 8. + `@Vector(8, u16)` **SIMD `split_float`** (ARM NEON / SSE2 in one source path) | 1.76 s | 2000× | 1.515× |
| 9. + **mmap** the input safetensors file (skip the explicit user-space 2 GB memcpy) | 1.69 s | 2100× | 1.515× |
| 10. + **PHOG** (count-based MLE, learned from TinyLlama oracle dump) replaces hand-tuned fast-path | **~1.8 s** (incl. ~50 ms per-call PHOG argmin overhead) | 2000× | 1.515× |

The big takeaways:
* **Most search time was wasted re-finding the same answer.** A 30-line
  classifier (`fastPathShape` in `src/search.zig`) covers >95 % of trained
  weights without losing measurable ratio. Search only runs as a fallback.
* **Histogram was the dominant `huffmanBuild` cost**, not the tree walk.
  Replacing `AutoHashMap` with 4 parallel `u32[256]` counters (which the
  compiler keeps in registers and the OoO engine pipelines across iterations)
  dropped 482 ms / 65 M-element stream to 15 ms — **32×** without any
  intrinsics.
* **Zig's `@Vector(N, T)` is plenty for SIMD splitFloat** — one source path
  compiles to NEON on Apple Silicon and SSE2/AVX2 on x86-64 with no
  cfg-dispatch.

Other knobs:
* Bitplane heuristic does popcount per bit position in a single pass — no
  temporary planes are allocated during scoring.
* Per-tensor synthesis still runs in parallel; the worker pool size is
  `std.Thread.getCpuCount()`.
* Input safetensors are **mmap'd** (`std.Io.File.MemoryMap.create` with
  `populate=false`); the kernel demand-pages on first access. This shaves
  ~70 ms wall + ~750 ms kernel CPU vs. the read-into-buffer fallback.
  Falls back to chunked `read()` if the IO backend doesn't support mmap.

## What's in the box (next-steps from the original README → done)

| original next step | status | location |
| --- | --- | --- |
| **#1 A\* over DSL grammar** | ✅ done | `src/search.zig` — best-first search with admissible Shannon-entropy heuristic + early pruning when `next.h ≥ best.actual`. Uses a real `std.PriorityQueue`. |
| **#2 PHOG-style prior** | ✅ done (count-based MLE + Laplace, learned from oracle dump) | `src/phog.zig` (auto-generated) + `tools/train_phog.py`. PHOG predicts each grammar production from a discretized context (position × dtype × log₂(numel) × ndim). Replaces hand-tuned fast-path. Untrained contexts return `null` and trigger A\* fallback. See [PHOG section](#phog-learned-fast-path). |
| **#3 AMaze outer DSL search** | ⛔ skipped | Out of MVP scope (needs trained cost model + corpus). |
| **#4 Cross-tensor reference** | ✅ done | `src/ops.zig::tensorXorForward/Inverse` + `src/program.zig::OpKind.tensor_xor`. The search enumerates `tensor_xor(b)` for every supplied base tensor and only keeps it if the residual compresses better than the standalone tensor. |
| **#5 Real safetensors** | ✅ done | `src/safetensors.zig` — full reader + writer of the HuggingFace .safetensors format (8-byte header length, JSON header, raw little-endian payload). |
| **#6 Better entropy coder** | ✅ done | `src/codec.zig` — both canonical Huffman *and* a 32-bit / 14-bit-precision rANS coder. The synthesizer picks per-stream. |
| **#7 Shared codebooks** | ✅ done | `src/archive.zig` — every Huffman / rANS table is interned by content fingerprint into a single shared section in the .brv file. Identical tables across tensors are stored once. |

## Project layout

```
src/
  types.zig         dtype, Stream, TensorView
  codec.zig         BitWriter/Reader, canonical Huffman, rANS
  ops.zig           reversible DSL ops (forward + inverse, no tree dependency)
  program.zig       Node tree, compress/decompress, binary serialization
  search.zig        A* synthesis with admissible heuristic + prior
  prior.zig         PHOG-lite: rule scores from stream features
  archive.zig       .brv container with shared codebook section
  safetensors.zig   HuggingFace safetensors reader/writer
  baseline.zig      gzip (in-process) + zstd (shell-out) for comparison
  main.zig          CLI: compress / decompress / bench / verify / baseline / demo / make-fixture
  tests.zig         unit + roundtrip + e2e tests
build.zig
build.zig.zon
```

## Architecture

```
   safetensors file              .brv archive on disk
        │                                ▲
        ▼                                │
   load tensors            ┌──────────────────┐
        │                  │ shared codebooks │
        │                  │  + per-tensor    │
        ▼                  │  programs        │
   for each tensor t:      └──────────────────┘
       search.synthesize(t, bases)              ▲
            │                                   │
            ▼                                   │
       enumerate candidates over grammar:        │
         T_PROG := tensor_raw                    │
                 | tensor_xor(b) -> T_BODY       │
                 | split_float -> S × S × S       │
         S_PROG := raw | huffman | rans          │
                 | delta_encode -> S_TERMINAL    │
                 | bitplane_split -> S_TERMINAL  │
            │                                   │
            ▼                                   │
       score by ∑(Shannon entropy lower bound) +│
                ∑(prior log-penalty)             │
            │                                   │
            ▼                                   │
       priority queue → realize best, verify    │
                       roundtrip, prune ←─ admissible
            │
            ▼
       Result { program, payload, ratio, verified=true }
            │
            ▼                                    
       collect → archive.buildArchiveBytes ─────┘
```

## DSL grammar

```
Tensor program:
  T_PROG := tensor_raw                                 -- store raw bytes
          | tensor_xor(base_id) -> T_BODY              -- W ^ base then sub-program
          | split_float -> S_PROG × S_PROG × S_PROG    -- (sign, exp, mant)

Stream program (depth ≤ 1):
  S_PROG := raw | huffman | rans
          | delta_encode  -> S_TERMINAL                -- first-order diff
          | bitplane_split -> S_TERMINAL × n           -- n = bits_per_elem
  S_TERMINAL := raw | huffman | rans
```

The depth-1 limit on stream programs is deliberate (the gains from deeper
nesting are small and the search space explodes). The cap is a single line
in `src/search.zig` if you want to lift it.

## A\* + prior, in concrete terms

* **State:** a fully-derived candidate program (no non-terminals).
* **g(s):** zero before realization (we don't have actual bits yet).
* **h(s):** sum over leaves of Shannon-entropy + small table overhead. This
  is admissible because no entropy coder beats the Shannon bound.
* **p(s):** the PHOG-lite log-prob penalty. This is *not* admissible in the
  strict sense, but it's bounded: priors are capped per node and damped by
  `√count` so the prior matters less for long streams (where data dominates)
  and more for short ones (where the choice is mostly heuristic).
* **Prune rule:** once one candidate is fully realized with cost `c*`, any
  remaining candidate with `h+p ≥ c*` is dropped — Shannon admissibility
  guarantees they can't beat it. With the synthetic fixture, ~75% of the 730
  candidates are pruned in practice.

## Cross-tensor compression

`tensor_xor(b)` encodes `W ^ W_base` (XOR of bit patterns), which is exactly
reversible and produces low-entropy output when `W ≈ W_base`. The current
CLI doesn't expose base selection (it only synthesizes per-tensor), but the
`search.synthesize(input, bases)` API takes any list of base tensors. There
is a unit test (`search: tensor_xor improves ratio when base is similar`)
demonstrating ~3-5× ratio improvement when a near-identical base is
available.

To take advantage of cross-tensor compression at the CLI level you'd
typically pre-cluster tensors (e.g. layer-pairs of an LLM) and pick a base
per cluster — that's left as a follow-up.

## Verification

`brevis verify <archive.brv> <original.safetensors>` decompresses every
tensor in the archive and bit-exact-compares to the matching name in the
original file. Returns non-zero on any mismatch. The test suite includes
roundtrip tests at every layer (codec, op, program, archive, safetensors,
end-to-end) — run `zig build test --summary all`.

## What's NOT done

* AMaze-style outer DSL search (would need a learned cost predictor + a corpus).
* True PHOG learning (the prior is hand-tuned; a real PHOG would condition
  on tree context and learn from labelled-best programs).
* Lifting the `S_PROG` depth limit beyond 1.
* Quantized weight handling (`U8` / `I8` / `Q4` etc.) — the safetensors
  reader skips non-fp16/bf16 tensors with a warning rather than synthesizing
  for them.
* Streaming compress/decompress (current code holds the whole archive in memory).
