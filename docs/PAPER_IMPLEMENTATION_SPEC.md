# Brevis paper implementation specification

This document is the engineering specification for the Brevis core. The
normative source is `Brevis-draft/main.pdf`, especially Equations (3)-(25),
Figure 3, and Algorithm 1. When this document and the legacy implementation
disagree, this document wins.

## 1. Compression model

For a tensor `X` with dtype `d`, shape `s`, physical word width `b(d)`, and
`n = product(s)` elements, Brevis operates on the flattened physical words
`Bits(X)`, not on their numerical interpretation.

Compression returns one self-contained tensor record:

```text
TensorProgram {
    dtype,
    shape,
    program,
}
```

The program is well formed only when it has type `b(d)[n]`. Executing it must
produce `Bits(X)` exactly. The learned rule model and every other encoder-only
object are excluded from the archive and from decoding.

## 2. Semantic DSL

The required semantic core is:

```text
P ::=
    Lit[b](words)
  | Const[b, n](word)
  | Concat(P1, ..., Pk)
  | Repeat[k](P)
  | Map[operation, parameters](P)
  | Scan[operation, parameters](initial, P)
  | Merge[operation, parameters](P1, ..., Pk)
```

The implementation may add a production only when it:

1. has a total, deterministic execution semantics;
2. has a target-directed decomposition whose accepted result reconstructs the
   parent target exactly;
3. has an explicit well-formedness rule;
4. has a stable canonical serialization whose complete byte cost is charged;
5. is independently useful and cannot be represented economically by the core
   productions.

The initial concrete operation families are:

- `Map`: XOR constant, modular addition, ZigZag permutation, Gray-code
  permutation, bit rotation, and bit reversal.
- `Scan`: previous-word XOR and modular-difference updates. The initial word is
  stored by the `Scan` node; its child generates exactly `n - 1` updates.
- `Merge`: contiguous bit fields, IEEE-like floating-point fields, bit planes,
  and byte planes.

Variable-length run generation and positional interleaving are useful
extensions, but they are separate semantic productions rather than aliases for
`Repeat` or `Merge`.

## 3. Required semantics and typing

Every program has exactly one derived stream type `b[n]`.

- `Lit[b](v)` has type `b[len(v)]` and executes to `v`. The implementation
  admits `Lit[b]([])` as the unique zero-length production so valid empty
  safetensors tensors remain representable without inventing a special
  out-of-language archive case.
- `Const[b, n](v)` requires `n >= 1` and `v < 2^b`; it emits `n` copies.
- `Concat(P1, ..., Pk)` requires `k >= 2`, equal child widths, and positive
  child lengths; it emits the children in order and sums their lengths.
- `Repeat[k](P)` requires `k >= 2`; it repeats the complete child stream and
  multiplies its length by `k`.
- `Map` requires a width-preserving bijection and preserves the child type.
- `Scan(initial, P)` requires a child of type `b[n - 1]`, `n >= 2`, and a
  bijection in the update argument for each previous word. It emits `initial`
  followed by the accumulated updates.
- `Merge` requires children of a common length, child widths whose sum is the
  output width, and a pointwise bijection from the child words to an output
  word.

Malformed programs are normal input errors. The decoder must never rely on
debug assertions for archive validation.

## 4. Literal representation

`Lit` is the universal semantic fallback. Raw, bit-packed, canonical Huffman,
and rANS payloads are canonical wire encodings of a literal, not separate
semantic program productions.

The program encoder selects the smallest valid literal wire encoding by its
complete encoded size, including the encoding tag, tables, lengths, and
payload. Deterministic tie-breaking is part of the format.

Raw and bit-packed bodies are applicable to every valid literal. As a bounded
resource extension, Huffman and rANS are applicable only when the literal has
at most 65,536 distinct physical words. Crossing that cap cannot make a tensor
unrepresentable: it deterministically removes those two table-based candidates
and leaves the universal raw/bit-packed path.

The encoder performs this global minimization once. The decoder validates the
selected codec and reconstructs its stream; it does not recompress the stream
to prove that no other literal codec would have been smaller.

This separation ensures that:

- the DSL remains a tensor-generating language;
- entropy coders do not masquerade as generative operations;
- one semantic program has a deterministic stored size;
- `serializedSize(P)` and the bytes emitted by `write(P)` have one source of
  truth.

## 5. Target-directed synthesis

Each search hole is a pair `H(type, target)`. Expanding a production with
parameters applies its target-directed decomposition. An expansion is admitted
only when recomposing its child targets yields the parent target exactly.

The synthesizer:

1. initializes the incumbent to `Lit(target)`;
2. pops partial programs by the lexicographic key
   `(grammar_cost_so_far + admissible_grammar_heuristic,
   serialized_size_lower_bound)`;
3. expands the leftmost hole;
4. evaluates every completed program by its exact canonical serialized size;
5. continues after the first completion;
6. stops at the expansion budget or an empty queue;
7. returns the smallest correct program encountered.

When a partial state cannot fit the remaining expansion budget, the
implementation retains the PHOG/A*-preferred open state. At budget exhaustion
it completes every remaining hole in that one state with `Lit`, evaluates the
result by exact canonical size, and compares it with the incumbent. This
bounded terminal completion is what lets a learned prior affect a one-expansion
search without pretending that the recursively completed nodes were expanded.

The engineering default is one expansion per tensor. The manuscript
configuration explicitly requests 512; changing the engineering default does
not change the manuscript setting or the search semantics at a fixed budget.

The rule prior changes exploration order and, at budget exhaustion, selects the
one open frontier completed with `Lit`. It must not change:

- production applicability;
- literal encoding;
- exact serialized size;
- correctness checks;
- final comparison between complete candidates.

Before the checkpoint's tensor searches, the encoder trains an input-local
PHOG over a deterministic subset of complete tensors. Budget zero skips this
step. At the one-expansion engineering default, a teacher searches up to six
expansions on at most four tensors, while the requested searches remain at one
expansion and disable the shallow float-field incumbent. Larger budgets use the
engineering cap of 32; the manuscript setting is 256. The resulting immutable
prior guides queue order and selects the one terminal-completed frontier. A
caller-supplied prior overrides automatic calibration; it is encoder-only and
never enters the archive.
Independent calibration searches may run concurrently. Rule observations are
reduced exactly, so worker count cannot change the prior or aggregate search
statistics.

The target-directed proposal generator uses a semantics-preserving normal form.
It omits identity maps, one-bit aliases, uniform period-one repeats already
covered by `Const`, and parameterized field splits identical to the shorter
parameter-free byte-plane or bit-plane form. This reduces redundant search
paths without removing the corresponding programs from the DSL or wire format.
PHOG normalization uses exactly the distinct production families admitted by
the proposal generator.

Finite parameter proposals plus explicit depth and node limits guarantee
termination. These limits do not imply global optimality.

The implementation follows Equations 18--20 with a finite relaxed typed
grammar. A concrete type `b[n]` maps to one of:

- `b[0]`, the explicit empty-stream extension;
- `b[1]`;
- `b[2+]`.

The table also retains the word width and configured depth. It forgets the
exact positive length, target-specific decomposition guards, and concrete
parameters. `Concat` and `Scan` may choose any child cardinality class
consistent with some concrete positive length. Node and decomposition-memory
limits are omitted from the table. These relaxations add derivations and
therefore cannot raise the shortest completion cost. The empty class has only
the extension `Lit(empty)`; its cost is zero.

The concrete PHOG normalizes over the productions admitted for the current
target, so Equation 18 cannot be implemented by scoring every production
against one fixed full rule set. For each typed-depth rule class, the
implementation proves a minimum admitted-set cardinality `m`. For production
`r`, let `Nmax(r)` be its largest count in any stored PHOG or backoff row.
With smoothing `beta` and learned mixture weight `lambda`, every concrete
probability is bounded above by

```text
p_upper(r,m) =
  lambda * (Nmax(r) + beta)
           / (Nmax(r) + beta + (m - 1) * beta)
  + (1 - lambda) / m .
```

The Q10 rule-cost lower bound is
`w_lower(r,m) = -log2(p_upper(r,m))`, using the same deterministic integer
logarithm as concrete scoring. The bound pessimistically gives all competing
rules only their smoothing counts and lets `r` attain its maximum count in
every context. An unmatched context is uniform and is covered by the same
bound. Thus `w_lower` is no greater than the concrete cost paid by any
expansion.

Dynamic programming then computes the recursive shortest derivation

```text
c(A) = min_r (w_lower(r) + sum(c(A_child)))
h(s) = sum(c(A)) for A in open_holes(s).
```

Depth makes the implemented recurrence acyclic, and universal `Lit` gives
every nonempty type a finite base case. The relaxation can select a genuinely
multi-rule derivation when, for example, a learned unary transform followed
by `Lit` is cheaper than direct `Lit`.

The second A* key charges the mandatory program prefix, every fixed
instruction/operation/parameter field already selected, conservative literal
framing, and one mandatory node tag for each open hole. Unselected payload
bytes are omitted, so this byte bound cannot overestimate a completion.

A separate structural resource check estimates child-target storage before
decomposition. Search skips an expansion when replacing the current hole
would exceed the configured total storage of open target streams. This is not
a grammar or PHOG restriction: the same production remains semantically
legal, and the complete target `Lit` incumbent remains available.

## 6. Whole-tensor programs and internal chunking

The archive stores one program per tensor. Fixed-size blocks are not archive
records and are not the unit of synthesis.

File compression and decompression may process independent tensor records in
parallel. The manuscript configuration uses 32 workers, each handling one
complete tensor at a time. A completion queue schedules at most one task per
worker with bounded two-window lookahead, avoiding source-order head-of-line
stalls without buffering the whole checkpoint. Emission remains source
ordered, and parallel execution must produce the same canonical archive as one
worker.

An implementation may internally bound memory by lowering a tensor program to
streamable regions. When regions need independent subprograms, that structure
must be explicit in the tensor program:

```text
Concat(P_region_1, ..., P_region_k)
```

The region boundaries, child program instructions, parameters, and literal
payloads are therefore all charged by the same program serialization. A
decoder executes the tensor program; it does not reconstruct an unstored
tensor-level template from block programs.

## 7. Canonical program format

The new format must:

- use explicit stable wire IDs independent of internal enum ordering;
- be versioned;
- reject truncated data, trailing data, unknown IDs, integer overflow,
  excessive recursion, excessive allocation, invalid parameters, invalid
  arity, and inconsistent derived types;
- satisfy `Read(code(P)) = P` for every encoder-produced program;
- provide a counting writer and a byte writer that share the same encoding
  implementation;
- reject all formats other than the new `BRTA` archive version explicitly.

The compatibility-removal decision is deliberate: schema-5/schema-6 archives
encode the superseded block/template language, so adapting them into the new
whole-tensor semantic program would either change their meaning or preserve two
competing implementations. They must be decoded with the matching legacy
Brevis revision and recompressed. The new decoder does not guess formats or
silently dispatch into legacy bytecode.

Archive framing, tensor names, dtypes, shapes, and original safetensors framing
are counted in complete-checkpoint results, even though they do not affect the
choice between programs for one fixed tensor.

## 8. Public seams and acceptance tests

The core exposes four behavioral seams:

```text
synthesize(target, dtype, rule_model, budget) -> Program
writeProgram(Program) -> bytes
readProgram(bytes) -> Program
execute(Program) -> exact physical-word stream
```

The archive module composes these seams but does not define DSL semantics.
`synthesize` returns an owning program. File encoding and calibration may call
explicit borrowing variants whose root `Lit` refers to the input tensor only
while its source view is alive; every structured program owns its leaf storage.
File synthesis also returns the canonical bytecode selected for the winner, so
the zero-budget terminal path does not repeat its serialized-size and
serialization passes during archive framing. Calibration requests only its
exact length. A root `Lit` is exact by construction; structured winners are
executed and checked. These optimizations do not change program bytes or the
decoder interface.

Required acceptance properties:

1. Every DSL production passes worked-example and randomized execution tests.
2. Every accepted target decomposition recomposes to the exact parent target.
3. The FP32 alternating-sequence example is represented by a real `Repeat`
   program and executes to the six required physical words.
4. `execute(readProgram(writeProgram(P))) == execute(P)`.
5. `serializedSize(P) == writeProgram(P).len`.
6. `synthesize` always returns a well-formed exact program, including with a
   zero search budget.
7. Learned and uniform rule models may change search order, but exact size
   always selects the incumbent.
8. A tensor record is independently executable and regenerates exactly the
   number of bytes implied by its dtype and shape.
9. Complete safetensors archives round-trip byte for byte.
10. Corrupt or adversarial bytecode fails safely without assertion, out-of-
    bounds access, or attacker-controlled unbounded allocation.

## 9. Explicit non-goals

- Claiming a globally shortest program.
- Treating the current block-template planner as normative behavior.
- Selecting a program by sample size, estimated entropy, or grammar
  probability instead of complete canonical bytes.
- Requiring the rule model during decompression.
- Preserving internal Zig types or opcode ordinals from the legacy core.
- Reading schema-5/schema-6 block archives in the paper-aligned decoder.
