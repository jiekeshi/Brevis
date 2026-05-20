# Brevis — Lossless Tensor Compression via Program Synthesis

Bit-exact lossless compression of neural-network weights, implemented as a
faithful port of the **Euphony** algorithm (Lee, Heo, Alur, Naik. *Accelerating
Search-Based Program Synthesis using Learned Probabilistic Models*. PLDI'18)
to the domain of fp16/bf16 tensor encoding.

Implemented in **Zig 0.16**.

## What the algorithm does

```
Grammar  G = ⟨N, Σ, R, S⟩
   N = {S}                     (single non-terminal: stream program)
   Σ = 16 reversible primitives
       (xor_const, diff_mod, split_field, …) + 3 terminals (huffman/rans/raw)
   R = S → op β    for each op

PHOG   q(A → β | c)             learned conditional production probability
                                c = (parent_op, slot) (left-sibling/parent
                                    summary, the simplest non-trivial context)
                                trained from solved instances, Laplace-smoothed

A* search on the sentential-form derivation graph G(G_q):
   nodes  = sentential forms (partial programs with holes)
   edges  n →ᴬ⁻ᵝ n′  weight  −log₂ q(A → β | c)
   start  = S
   goals  = complete programs (no holes)
   g(n)   = path weight = −log₂ Pr(partial program)
   h(n)   = Σ over open holes of  neg_log_h_S
            where  h(S) = max(q(A→β|c) · Π h(βᵢ))  via fixpoint  (Eq. of §3.3)
            so h(n) is an admissible underestimate of completion cost.
   Pop lowest g+h; expand the leftmost hole; for each (op, param) compute
   the production edge and push the new state. Goal nodes are realized
   (Huffman/rANS encoded) and the minimum-bytes one is returned.

Training (`brevis train`):
   Round k:
     for each training tensor: synth → realized program P with current PHOG
     walk P depth-first; for each parent→child edge: PHOG.observe(ctx, op)
   PHOG.computeFixpoint()
   Repeat for several rounds; later rounds find deeper programs because
   high-likelihood productions get sharper.
```

This is the same conceptual loop as Euphony Algorithm 1 (CEGIS-with-guided-
search), specialized to our domain — instead of "verify program against
spec, add counterexample", brevis "realize program → measure actual encoded
bytes, keep min of top-K".

## Build & run

```bash
zig build -Doptimize=ReleaseFast

./zig-out/bin/brevis make-fixture model.safetensors        # synth fp16 model
./zig-out/bin/brevis train       model.safetensors brevis.phog
                                                            # 3 rounds of synth+observe
./zig-out/bin/brevis compress    model.safetensors model.brv
                                                            # auto-loads brevis.phog if present
./zig-out/bin/brevis decompress  model.brv          recovered.safetensors
./zig-out/bin/brevis verify      model.brv          model.safetensors
./zig-out/bin/brevis bench       model.safetensors
./zig-out/bin/brevis baseline    model.safetensors          # vs gzip / zstd
./zig-out/bin/brevis demo

zig build test                                              # 25/25 pass
```

## Source tree

```
src/
├── types.zig         Stream, TensorView, Dtype
├── codec.zig         Huffman + rANS + SIMD histogram (encoder backends)
├── lowlevel.zig      16 reversible primitives + property tests
├── grammar.zig       CFG: NonTerminal S, productions, context, param choices
├── phog.zig          PHOG q(A→β|c) + Laplace + h(S) fixpoint
├── astar.zig         A* on sentential-form graph + PNode + realize + decompress
├── pnode_archive.zig v2 BRV format (PNode trees + shared codebooks)
├── safetensors.zig   I/O with mmap fast-path
├── baseline.zig      gzip / zstd comparison
├── main.zig          8 CLI subcommands
└── tests.zig         25 unit tests
```

About 3.5 K LOC of Zig.

## End-to-end example

```
$ brevis make-fixture x.safetensors
wrote fixture: x.safetensors (3 tensors)

$ brevis train x.safetensors brevis.phog
train: 3 tensors, 3 rounds of synth → observe → fixpoint
  round 1: 3 programs, ratio 1.197x, h(S)=2.87 bits, 62ms
  round 2: 3 programs, ratio 1.197x, h(S)=2.32 bits, 46ms
  round 3: 3 programs, ratio 1.197x, h(S)=2.00 bits, 42ms
wrote trained PHOG → brevis.phog

$ brevis compress x.safetensors x.brv
compress: 3 tensors
wrote x.brv (705414 bytes) in 43ms
ratio: 1.197x (5259264 → 4392808 bits)

$ brevis verify x.brv x.safetensors
verify: 3/3 bit-exact
```

## Component detail

### `grammar.zig` — the CFG

```
S → huffman | rans | raw                          (terminals, arity 0)
S → xor_const(c⋆) S                               (chain, arity 1)
S → diff_mod S
S → split_field(start⋆, n_bits⋆, k⋆) S S          (split, arity 2)
…
```

There's one non-terminal `S` ("stream program"). Each production is identified
by its `lowlevel.OpKind`. Parameters (`c`, `start`, etc.) are enumerated lazily
at expansion time from a per-op concrete list — this is the pivot-grammar
`param⋆` abstraction from Euphony §4.3, in its simplest form: PHOG counts
aggregate over all concrete parameter values of an op.

Context for PHOG: `(parent_op, slot)` where `slot ∈ {0, 1}` distinguishes
the two children of `split_field`. Root has `parent_op = null`.

### `phog.zig` — the probabilistic model

```zig
counts[ctx_encoded][op_kind] : u32    // observation count
q(op | ctx) = (count + α) / (row_total + α × N_OP_KINDS)   // Laplace, α=1
neg_log_q(op | ctx) = -log₂ q(op | ctx)

h(S) fixpoint (Theorem 3.3):
    init  -log h(S) = +∞
    iter  -log h(S) = min over (op, c) of [-log q(op | c) + arity(op) × (-log h(S))]
    until convergence (typically <30 iter for our grammar)
```

The fixpoint is computed once after training and stored as
`phog.neg_log_h_S`. The A* search adds it for each remaining hole.

### `astar.zig` — the search engine

The state is a flat `[]SFormNode` array (root at index 0). Each `hole`
carries its context; each `op` carries kind + concrete params + child
indices. State expansion:

1. Find leftmost hole.
2. For each `(op, param_value)` in `grammar.paramChoices(op, bpe)`:
   a. Build a new state by cloning, replacing the hole with the op node and
      adding up to 2 new hole nodes (one per child non-terminal).
   b. New `g = old.g + neg_log_q(op | ctx)`.
   c. New `h = new_n_holes × neg_log_h_S`.
   d. Push to PQ.
3. When popping a state with `n_holes == 0`: convert to `PNode`, realize
   (run Huffman/rANS encoders, fill side_info + payload), measure actual
   compressed bytes. Keep min over top-K realized.

After search returns the `Best.program`, it's already realized and ready
for archiving or decompression.

### Archive format (`pnode_archive.zig`, magic `BRV\x02`)

```
MAGIC "BRV\x02" + version u16 + flags u16
N_TABLES u32
  (kind:u8, n_entries:u32, entries…) × N        (shared Huffman + rANS tables)
N_TENSORS u32
  (name, dtype, shape, program_bytes, payloads)  × N
```

Tables are deduplicated across tensors (hash by content fingerprint).

## What's NOT done (vs the full Euphony paper)

| Component | Done? | Notes |
|---|---|---|
| Weighted A\* search | ✅ | Algorithm 2 from the paper |
| PHOG `q(A→β\|c)` | ✅ | with Laplace smoothing |
| `h(A)` fixpoint heuristic | ✅ | Theorem 3.3 |
| Equivalence-class pruning (§3.4.1) | ⛔ | next-step optimization |
| Pivot grammar for transfer learning (§4) | ⚠️ partial | params are merged via `param⋆`, but no `const_I`/`const_O`/etc. abstract symbols since brevis has no semantic spec to filter on |
| Divide-and-conquer enumeration (§3.4.2) | ⛔ | not applicable — no predicates in our grammar |

## Caveats

* Without a trained PHOG (uniform Laplace prior), the search is essentially
  blind and produces poor ratios. **You must run `brevis train` first.**
* On small fixtures the .brv archive overhead (Huffman/rANS tables per
  tensor) can exceed the encoding savings, giving on-disk ratios <1.0.
  Real-world models with 100+ similar tensors amortize this cost away.
* Search is hard-bounded by `max_pops` and `realize_top_k` in the runtime
  `Opts` — tuning these trades latency for ratio.

## License

See [LICENSE](LICENSE).
