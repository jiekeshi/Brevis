# Brevis

Brevis is an exact, lossless compressor for neural-network tensors. It synthesizes a small typed program that regenerates each tensor's physical words, then stores that program in a versioned archive.

The paper manuscript is the normative design.

[`docs/PAPER_IMPLEMENTATION_SPEC.md`](docs/PAPER_IMPLEMENTATION_SPEC.md) is its engineering companion. When older code, formats, or evaluations disagree with them, the paper-aligned implementation wins.

Brevis targets Zig 0.16.

## Core contract

For a tensor with dtype `d`, shape `s`, word width `b(d)`, and `n = product(s)` elements, Brevis operates on the flattened physical words `Bits(X)`. It never interprets floats or integers numerically during reconstruction.

Compression produces exactly one self-contained `TensorProgram` per complete tensor:

```text
TensorProgram {
    dtype,
    shape,
    root: Program,
}
```

The root must have type `b(d)[n]`. Executing it must reproduce `Bits(X)` byte for byte. The archive does not contain hidden tensor templates, implicit block programs, or an encoder-side rule model.

The universal fallback is a semantic `Lit` program. Therefore every valid supported tensor remains representable, including when the search budget is zero.

## Semantic DSL

The program language has seven node kinds:

| Node | Type and execution rule |
| --- | --- |
| `Lit[b](words)` | Emits the stored `b`-bit words. An empty literal is supported so zero-element tensors remain representable. |
| `Const[b,n](word)` | Emits `n >= 1` copies of one valid `b`-bit word. |
| `Concat(P1,...,Pk)` | Requires `k >= 2`, equal child widths, and positive child lengths; emits children in order. |
| `Repeat[k](P)` | Requires `k >= 2`; repeats the complete child stream. |
| `Map[op,params](P)` | Applies a total width-preserving bijection to each child word. |
| `Scan[op](initial,P)` | Emits the explicit initial word, then accumulates exactly `n - 1` child updates. |
| `Merge[op](P1,...,Pk)` | Pointwise combines equal-length children whose widths sum to the output width. |

`Map` supports XOR, modular addition, ZigZag, Gray code, left rotation, and bit reversal.

`Scan` supports previous-word XOR and modular-addition updates. Its initial word is part of the node, so decoding requires no external state.

`Merge` supports contiguous fields, IEEE-like floating-point fields, bit planes, and byte planes.

Every node has one derived stream type `b[n]`. Constructors, deserialization, and execution validate widths, lengths, arities, parameters, shape products, and arithmetic overflow.

All transforms are deterministic and exactly reversible at their fixed word width. Malformed programs are ordinary input errors; correctness does not depend on debug assertions.

## Target-directed synthesis

Each open search hole contains both a required type and the exact target stream it must generate. A production is admitted only when its decomposition can recompose to that target exactly.

The finite grammar proposes constants, periods, split points, map parameters, rotations, and field layouts from the target under explicit caps. Depth, node, and expansion limits make every search terminate.

The proposal set is kept in a small normal form. Search omits identity maps,
one-bit aliases, period-one repeats already represented by `Const`, and a
parameterized field split when it is identical to the shorter parameter-free
byte-plane or bit-plane form. This removes redundant derivations without
changing the semantic language or archive format.

Search begins with `Lit(target)` as the incumbent. It expands the leftmost hole,
compares complete candidates by canonical serialized size, and continues after
the first completion. A structured final winner is executed and checked against
the exact target. The root literal is exact by construction and returns
directly.

The winner is the smallest correct complete program encountered, measured by exact canonical serialized bytes. A budget limit bounds work; it is not a claim of finding the globally shortest possible program.

The queue key is an estimate of the completed program's serialized size: the
exact bits of every instruction field already selected, plus the data cost each
open hole's target still owes. A target the estimator's sample covers completely
is measured with the real literal codec; a longer one is modelled from a bounded
deterministic sample over the same three regimes the codec chooses between - raw
storage, minimum-width bit packing, and a table-driven entropy body.

This data term is what makes structure reachable. Contextual rule cost alone is
nonnegative and paid per node, so a score built only from it is monotone in
derivation length: the trivial `Lit` is always cheapest, every additional hole
is pure loss, and a wide `Merge` ranks last precisely when splitting is what
collapses its children's entropy. Ordering by estimated size instead lets the
encoder see the payoff at the moment a production is proposed.

Grammar description length remains the secondary key, ordering states whose
size estimates agree. For Equations 18--20 the encoder derives conservative
per-production costs across every PHOG context, then solves a recursive
shortest-derivation problem in a relaxed typed grammar. The relaxation retains
width, depth, and the length classes `b[0]`, `b[1]`, and `b[2+]`, while dropping
target guards, exact positive lengths, and concrete parameters. Exact
serialized-size lower bounds break remaining ties and prune states that cannot
improve the incumbent.

At budget exhaustion the encoder retains several best-ranked open states rather
than one, completes each with `Lit`, and measures them by exact canonical bytes.
Retaining a single state makes the whole program depend on one guess that was
never measured, and it lets a larger budget return a worse program than a
smaller one.

`--data-cost-ordering 0 --frontier-candidates 1` restores the published
grammar-cost search exactly.

The explicit empty-stream extension is isolated as `b[0]`: only `Lit(empty)`
is admitted and its completion cost is zero. Universal `Lit`, deterministic
Q10 costs, and the depth-indexed dynamic program keep the heuristic finite and
admissible even when node and decomposition limits further restrict concrete
search.

An independent decomposition-storage budget is checked before child targets
are allocated. It bounds the total storage of simultaneously open target
streams, so amplifying transforms such as 32-bit bit planes can be skipped
without changing DSL legality or the universal `Lit` fallback.

### What the rule prior contributes

The prior does not compete with the size estimate; it covers where the estimate
is weakest. The estimate is a bounded zeroth-order sample of each target, and on
homogeneous BF16 weights the decomposition that actually wins can sit 20-30%
behind in estimated size, outside any affordable estimate-ranked frontier. The
prior has measured which decomposition won on sibling tensors of the same
checkpoint, so its nominations reach states the estimate ranks too low to try.

Exact-measurement slots are therefore split between the two. Spending them all
on the estimate is not the best use:

```text
CodeLlama-7B shard 2 (BF16)          archive bytes      time
  4 estimate slots, 0 prior       2,301,221,029      21.5 s
  1 estimate slot,  2 prior       2,301,221,029      15.4 s
  1 estimate slot,  3 prior       2,298,153,444      23.5 s
```

Three prior slots reach an archive no estimate-ranked frontier reaches at any
width, and two prior slots reach the estimate's best result in 1.4x less time.
The default is one estimate slot and two prior slots, which is never worse in
bytes than four estimate slots and is faster on most checkpoints. Raising
`--prior-frontier-candidates` to 3 buys a further 0.13% on some BF16
checkpoints for roughly 1.5x the encode time.

Where the estimate already ranks well the prior is simply neutral: GPT-2 (F32)
and TinyLlama-15M (F16) produce identical archives with and without it.

### PHOG ordering

An optional probabilistic higher-order grammar (PHOG) assigns description costs to legal productions. Context includes tree position, dtype, width, length, and bounded target features.

The PHOG orders the queue and selects the one open frontier completed with
`Lit` when the expansion budget is exhausted. It cannot add or remove a
production, alter a decomposition, change a literal encoding, affect
correctness checks, or replace exact byte size as the final objective.

An empty or unmatched prior gives legal productions a uniform cost. With enough budget, uniform and learned ordering compare completed candidates by the same canonical bytes.

## Physical cost and canonical literals

`Lit` is one semantic node. Raw, bit-packed, canonical Huffman, and rANS are physical encodings of that node, not separate DSL productions.

The encoder computes the exact complete size of every applicable literal
representation and materializes only the winner. The comparison includes its
tag, tables, lengths, and payload. Exact lower bounds skip an rANS payload pass
when its framing and entropy bound already cannot beat the incumbent.
Size-only program costing reuses the same analysis without producing a
temporary payload.

Raw and bitpack are total. To bound table-building memory, Huffman and rANS are
defined as applicable only up to 65,536 distinct physical words; larger
alphabets deterministically retain the raw/bitpack fallback.

Equal-size literal encodings use a stable wire-tag order. The decoder validates the selected codec's tables, padding, lengths, and rANS state without rerunning all encoder-side codec analyses.

`serializedSize(program)` and `serialize(program)` share the same emitter and literal selection path. The reported cost therefore matches the bytes that are actually written.

## Formats and exact decoding

Semantic programs use canonical `BRPG` version 1 bytecode. Node, operation, and dtype IDs are explicit stable wire values rather than in-memory enum ordinals.

Whole-tensor archives use `BRTA` version 2. The archive header stores the original safetensors prefix, followed by one length-delimited record for each tensor.

Each record contains the tensor name, dtype, shape, canonical program bytecode,
and an XXH3-64 checksum of the decoded physical bytes. The checksum detects
accidental corruption; it is not a cryptographic authenticator. Records are
independently executable and contain no cross-record references.

Decoding follows this path:

```text
BRTA v2
  -> validate limits and safetensors metadata
  -> decode and type-check one BRPG v1 program per tensor
  -> execute the program
  -> verify the tensor XXH3-64 checksum
  -> restore the original prefix and tensor bytes
```

The complete `checkpoint.decompress*` readers reject unknown versions and
IDs, truncation, trailing bytes, overlong integers, overflow, invalid programs,
metadata disagreement, checksum failures, and configured resource-limit
violations. The deliberately named low-level
`tensor_archive.parseStructural` API checks framing, programs, and checksums
only; it does not cross-bind records to the embedded safetensors metadata.

## Input-local calibration

For every nonzero search budget, compression learns an encoder-only PHOG prior
from the loaded checkpoint before running its tensor A* searches. Calibration
uses a deterministic subset that covers dtype and logarithmic physical-size
strata before filling from source-order endpoints, and records productions only
after exact program validation. Budget zero remains a strict no-calibration,
no-search terminal path.

At the engineering default of one expansion, a small teacher searches up to six
expansions on at most four tensors. The real tensor searches still expand
exactly one state, with the learned PHOG choosing the frontier whose remaining
holes are completed by `Lit`. Larger requested budgets use up to 32 calibration
tensors by default; `compress --tensors N` changes the cap, the manuscript
setting is 256, and zero disables automatic calibration.

Counts use three context backoff levels, additive smoothing, and a configurable
learned/uniform mixture. The default uses the smoothed learned distribution
directly (`lambda = 1`); unseen contexts remain uniform.

Prior bytes are canonical and insertion-order independent. They influence
compression search order and bounded terminal-frontier selection, and are never
required for decompression.
File compression and the standalone `calibrate` command synthesize calibration
tensors concurrently, then merge exact rule counts on one thread. The resulting
prior and search statistics are identical across worker counts.

## Command line

Build the optimized executable:

```bash
zig build -Doptimize=ReleaseFast
```

Inspect the effective search configuration:

```bash
./zig-out/bin/brevis config
```

Compress and restore a safetensors file:

```bash
./zig-out/bin/brevis compress model.safetensors model.brv
./zig-out/bin/brevis decompress model.brv restored.safetensors
./zig-out/bin/brevis verify model.brv model.safetensors
```

The CLI defaults to the detected hardware-thread count. Set `--workers 1` for
single-core measurements or choose an explicit count for scaling experiments;
the manuscript setting is 32. Workers synthesize or execute independent
complete-tensor programs through a bounded two-window completion queue.
Records are written in source order and remain deterministic across worker
counts.

Every nonzero search budget calibrates from the input checkpoint. A separately
persisted prior overrides that checkpoint-local model:

```bash
./zig-out/bin/brevis compress model.safetensors model.brv \
  --max-expansions 512 --tensors 256 --workers 32
./zig-out/bin/brevis calibrate model.safetensors model.brgp --tensors 256
./zig-out/bin/brevis compress model.safetensors model.brv --prior model.brgp
```

The main compute controls are:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--max-expansions` | `1` | Maximum expanded partial programs per tensor. |
| `--max-nodes` | `64` | Maximum nodes in a completed program. |
| `--max-depth` | `4` | Maximum grammar depth. |
| `--max-repeat-period` | `32` | Largest proposed minimal repeat period. |
| `--max-concat-splits` | `3` | Maximum proposed binary split points. |
| `--max-map-constants` | `2` | Maximum representative XOR/add constants. |
| `--max-rotations` | `3` | Maximum proposed rotation amounts. |
| `--max-field-splits` | `3` | Maximum proposed contiguous-field splits. |
| `--tensors` | `32` | Maximum checkpoint tensors used for input-local PHOG calibration; the one-expansion teacher caps this at 4. |
| `--workers` | hardware threads | Maximum concurrent tensor jobs in calibration, compression, decompression, and verification. |

These limits trade search coverage for time and memory. They do not weaken the exact fallback or decoding checks.
Use `--max-expansions 0` to measure the canonical terminal-codec path and
`--max-expansions 1` for the fast shallow-search path. Reproduce the manuscript
setting explicitly with `--max-expansions 512 --tensors 256 --workers 32`.
The CLI enforces finite safety ceilings of 64 repeat-period elements, 16
Concat splits, and 8 proposals for each constant, rotation, and field-split
family.

The byte-resource controls, accepted by all commands, are:

| Option | Default | Meaning |
| --- | ---: | --- |
| `--max-total-bytes` | `17179869184` on 64-bit hosts | Maximum source, archive, or reconstructed file size; saturates to the address-space maximum on narrower hosts. |
| `--max-tensor-bytes` | `4294967296` | Maximum decoded bytes for one tensor; also bounds literal storage and open decomposition targets. |
| `--max-prefix-bytes` | `67108864` | Maximum safetensors prefix/JSON-header size. |

The default caps any single tensor at 4 GiB, which admits the embedding
matrices of large-vocabulary checkpoints while still rejecting absurd headers.
Lower it explicitly for untrusted input. SafeTensors limits are enforced
before format-specific metadata allocations. File commands truncate and
stream directly to their destination; an error may therefore leave a
partial file. Parallel paths bound in-flight work to two worker windows, so
peak memory scales with those windows and the active tensor sizes rather than
the complete checkpoint. Compression preserves source record
order, and a root `Lit` may borrow its tensor bytes while its source view is
alive. File compression retains it through record serialization; calibration
uses the same seam without materializing bytecode. The zero-budget terminal
path prepares the canonical encoding once and writes its bytecode directly
into archive framing. The public `synthesize` API remains owning.

Canonical program readers additionally default to 512 MiB of output/literal
storage and 4 GiB of cumulative interpreter byte-work, so a small deeply
nested program cannot force unbounded repeated passes. These byte limits are
not a process-RSS quota: compression retains a target/program candidate and
constructs literal and record candidates, so peak memory can be several times
the largest tensor per active worker. The file API avoids copying the complete
source and archive. For untrusted workloads, lower `--max-tensor-bytes`, reduce
`--workers`, and apply an OS/container memory limit.

## Build and test

Run the complete test suite in debug and checked optimized modes:

```bash
zig build test
zig build test -Doptimize=ReleaseSafe
```

Tests cover DSL typing and execution, target decomposition, canonical literal codecs, program round trips, bounded synthesis, PHOG behavior, archive validation, safetensors validation, calibration, and complete byte-for-byte reconstruction.

## Source layout

```text
src/types.zig                    physical streams, dtypes, and tensor views
src/dsl.zig                      typed seven-node semantic program tree
src/semantics.zig                fixed-width reversible operations
src/decomposition.zig            exact inverse decompositions
src/grammar.zig                  finite target-directed proposals
src/grammar_prior.zig            PHOG contexts, scoring, and prior format
src/synthesizer.zig              budgeted A* and exact incumbent selection
src/literal_encoding.zig         raw, bitpack, Huffman, and rANS lowering
src/codec.zig                    physical entropy-codec primitives
src/program_format.zig           canonical BRPG v1 serialization
src/interpreter.zig              validated program execution
src/tensor_archive.zig           whole-tensor BRTA v2 framing and checksums
src/safetensors.zig              strict safetensors parsing and writing
src/checkpoint.zig               end-to-end compression and decompression
src/calibration.zig              deterministic input-local prior training
src/brevis.zig                   public behavioral seams
src/main.zig                     command-line interface
src/*_tests.zig                  focused acceptance and adversarial tests
```

The central public seams are `synthesize`, `writeProgram`, `readProgram`, and `execute`. Archive and checkpoint code compose those behaviors without redefining DSL semantics.

## Compatibility

`BRTA` version 2 is intentionally incompatible with version 1 and legacy
schema 5 and 6 archives. Decode an older archive with its matching Brevis
revision before recompressing it. The paper-aligned reader does not retain the
old mutable bytecode, block framing, or back-reference model.

To migrate an old archive, decode it with the matching legacy Brevis revision, recover the safetensors file, then compress that file with this version.

## License

See [LICENSE](LICENSE).
