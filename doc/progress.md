# Progress log

Running record of what has been built and measured on the `autoDSL` branch.
Design and reference material lives in [`autodsl.md`](autodsl.md); this file is
the log.

Development measurements throughout. `eval/` is the audited apparatus and
enforces gates none of this does — nothing here is a formal result.

## The goal, and how success is judged

> Evolve reusable, verifiable, model-specific DSL rules automatically. Success
> is a **smaller total archive on unseen checkpoints**, after charging for the
> new rule library, while decoding stays bit-exact and the running cost stays
> acceptable. Failure is anything else: gains that exist only on the data the
> rules were learned from, overhead that cancels the gain, search that costs too
> much, or correctness that cannot be guaranteed.

That criterion is enforced in code, not just documented:

| Clause | Where |
| --- | --- |
| bit-exact | `verify.roundtrip` — a real `.brv` is written and verified |
| unseen checkpoints | `evaluate.verdict` measures the holdout split only |
| library overhead charged | `evaluate.objective` = archive bytes + `\|Serialize(L)\|` |
| acceptable running cost | `evaluate.MAX_PLANNING_SLOWDOWN` = 1.25× planning wall |
| not just training data | acceptance needs a develop gain *and* no holdout regression |

`python3 loop.py verdict` prints SUCCESS or FAILURE against exactly this and
exits non-zero on failure.

---

## 2026-07-25 — Self-extending DSL: mechanism built and gated

### Where the work started

The proposal was a two-level loop: search programs in the current DSL, then
evolve the DSL itself, under an MDL objective that charges for the grammar as
well as the programs. Two corrections came out of reading the code and
measuring it.

**The operator set is not `Repeat`/`Scan`/`Merge`.** `src/ops.zig` defines four
terminals (`raw`, `bitpack`, `huffman`, `rans`) and fifteen transforms
(`xor_const`, `add_const_mod`, `xor_prev`, `diff_mod`, `zigzag`, `gray`,
`rotate_bits`, `bit_reverse`, `split_field`, `topk_codebook`, `rle`,
`deinterleave`, `split_float`, `bit_plane`, `byte_plane`). Operators are
1→n stream fan-outs, not sequence combinators, and `bit_plane`/`byte_plane`
have width-dependent arity.

**`|Serialize(L)| + Σ|Serialize(Pᵢ|L)|` is the wrong objective here.** Measured
on SmolLM: program bytecode is **0.128%** of the archive and the library
contributes **0 bytes** (macros expand before serialization). Both terms are
rounding errors. What a library actually spends is *search budget*. Full
argument in [`autodsl.md`](autodsl.md).

### Diagnosis before design

All at `--rerank-candidates 64 --rerank-blocks 8`, `projected_archive_bytes`.
Measurement is deterministic, including across thread counts.

Search is budget-saturated: every SmolLM tensor exhausts its 256 expansions,
and 18 distinct programs cover 1245 blocks. But spending more buys almost
nothing there:

| SmolLM | archive bytes | vs. baseline |
| --- | ---: | ---: |
| depth 2, 256 exp | 171,839,429 | — |
| depth 2, 1024 exp | 171,835,502 | −0.002% |
| depth 3, 1024 exp | 171,839,281 | −0.000% |

On BERT it does, and depth adds as much again:

| BERT | archive bytes | vs. baseline |
| --- | ---: | ---: |
| depth 2, 256 exp | 366,006,869 | — |
| depth 2, 2048 exp | 365,682,062 | −0.089% |
| depth 3, 2048 exp | 365,368,461 | −0.174% |

Inspecting the depth-3 winners showed the depth-2/2048 gain was one program the
256-expansion budget never reaches — `split_float(rans,rans,bitpack)`, which is
*depth 1*. A budget problem, not a depth problem, and exactly what a macro
addresses.

### Built

**Engine (Zig).**

- `src/macro.zig` — library schema `brevis.macro-library.v1`, parsing,
  validation, body statistics. No search, no dependencies beyond `ops`.
- `src/search.zig` — `fitMacroBody` grafts a body into a hole, checking every
  node against `legalProductions` where it lands; the expansion loop offers each
  macro as one production charged at what its expansion would cost at best.
  `Partial`/`Result`/`Plan` carry a `macros_used` bitmask.
- `src/report.zig` — `writeOperatorTable` so tooling reads the DSL from the
  engine instead of redeclaring it; macro summary in `brevis config`;
  per-tensor `macros_used` in the JSON report.
- `src/main.zig` — `--macros <library.json>` on every command that takes search
  options.

**Outer loop (Python, `autodsl/`).** `library.py`, `engine.py`, `mine.py`,
`backend.py`, `propose.py`, `verify.py`, `evaluate.py`, `ledger.py`, `loop.py`.

**Tests.** 45 Zig (6 new, covering parse failures, bit-exactness of
macro-derived programs, and that ablations still bite through a macro) and 96
Python.

### Verified

- Bit-exact roundtrip with macros on SmolLM: `compress` → `verify` →
  `decompress` → `cmp` against the original 219,795,840-byte checkpoint.
- The three archived BERT anchors still reproduce without macros.
- All three LLM backends reachable through the CloseAI proxy: `claude-opus-5`
  (Anthropic native), `qwen3.7-max`, `qwen3.6-plus` (OpenAI-compatible).
  `claude-opus-5` rejects `temperature`, so the field is now omitted by default.

### The cost model was wrong, and it was the whole problem

First measurement of the hand-written seed library: **BERT 10,741 bytes smaller,
SmolLM 92,674 bytes larger**, and macro attribution showed why — on SmolLM *no
tensor selected a macro-derived program at all*. Three live proposals from
`claude-opus-5` (all structurally valid and plausible) were rejected for the
same reason. Nothing ever fired.

The cause was in how a macro was priced for A\*. It was charged
`node_count × production_floor` — "what its expansion would cost at best" —
on the reasoning that a macro should gain no unearned ordering advantage. That
reasoning is wrong. A\* explores by `f`, so a 2-node macro at `2 × floor` sits
behind the *entire* single-production frontier, and every one of those pops
pushes ~15 more children. Within 256 expansions the macro is never reached.

The fix is also the more principled reading: **a macro is one symbol of the
extended grammar, so it is charged one production.** That is what an
abstraction *is* — it shortens the description of the programs that use it. The
byte objective is untouched either way, so the "two costs are independent"
invariant still holds.

| Seed library | archive | vs. no macros | tensors built by a macro |
| --- | ---: | ---: | ---: |
| SmolLM, per-node pricing | 171,932,437 | +92,674 ❌ | 0 / 483 |
| SmolLM, one-production pricing | 171,835,441 | **−3,988** | **210 / 483** |
| BERT, one-production pricing | 366,000,915 | **−5,954** | 0 / 206 |

Bit-exactness re-confirmed after the change: a real `.brv` at exactly the
projected 171,835,441 bytes, `verify` clean on 483/483 tensors, and `cmp`
identical after a full decompress.

Two further consequences that shaped the gates:

1. `verify.fires` rejects an inert macro before measuring its bytes — an
   unfired macro is not neutral, it dilutes a fixed budget.
2. The gate holds a wide rerank shortlist fixed across the A/B, because with the
   engine's default 8-candidate shortlist *any* added production reshuffles
   which candidates reach the full-block rerank, and that swing is larger than
   the effect being measured.

Two bugs the tests caught, both real: `verify.check` evaluated its four gates as
a tuple, so all of them ran before the first refusal could return; and
`mine.evidence` read `search_options_applied["max_expansions"]` when the engine
puts the resolved configuration under `search` — a wrong test fixture had been
hiding it.

Two bugs the tests caught, both real:

- `verify.check` built its gates as a tuple, so all four ran before the first
  refusal could return — a structurally invalid macro still spawned processes
  and an inert one still wrote an archive. Now thunks.
- `search.zig` needed macro-usage telemetry to distinguish "offered" from
  "chosen"; without it the SmolLM regression looked like noise.

### Result: the criterion is met

Two macros survived every gate.

**Mined, no model involved.** Bootstrap put 8 mined shapes through the full
gate; seven were refused (four never fired, three cost more than they earned):

```
zigzag_split_field_h_h  =  zigzag(split_field(?,?))
```

Holdout alone: −48,767 bytes, 0.0028%. Real but negligible — mining recovers
shapes the search mostly finds already.

**Proposed by `claude-opus-5`.** Given only the measurements, the operator
table, and the current library, it proposed:

```
split_float_sign_exp_split_mantissa
  =  split_float(bitpack, rans, split_field(rans, bitpack))
```

with this justification, which is a correct reading of the evidence it was
shown:

> 113 MB of BF16 (embedding/lm_head) currently settles for
> `split_field(rans,raw)`, which pays ~8 flat bits for the mantissa and never
> isolates the 1-bit sign; this delivers sign→bitpack, exponent→rans and a
> second split of the mantissa so its high bits are entropy-coded while only
> the noise bits are packed. A 3-arity root with a nested transform under one
> child is four operators deep and is exactly what a 256-expansion, 12-node
> search never assembles.

That last sentence is the thesis of the whole design, arrived at independently:
the structure is legal in the primitive grammar and simply out of reach of a
bounded search. Its two siblings that round were rejected for never firing.

**Final verdict, holdout split only:**

```
SUCCESS — unseen checkpoints shrank by 2125950 bytes after charging the
          547-byte library, at 0.89x planning cost
  holdout 1,744,896,153 -> 1,742,769,656 bytes   (0.122%)
  library 2 macros, 547 bytes, sha256=ec3a6338
  models  google__vit-base-patch16-224, TinyLlama__TinyLlama-1.1B-Chat-v1.0
```

Every clause holds: bit-exact on both holdout models via real `.brv` archives,
measured on checkpoints the macros were never selected against, library
overhead charged, and planning *faster* rather than slower (0.89×) because a
macro reaches a good program in fewer expansions.

| Library | holdout Δ (charged) | share | planning |
| --- | ---: | ---: | ---: |
| mined only, 1 macro | −48,767 | 0.0028% | 0.95× |
| mined + proposed, 2 macros | **−2,125,950** | **0.122%** | 0.89× |

The proposed macro is worth 44× the mined one. That is the interesting result:
mining recovers what the search already finds, while the proposer reached a
structure that was never in the measurements because the search could never
assemble it.

**Caveats that belong next to the number.** Two holdout checkpoints is a weak
split. 0.122% is a real, verified, generalizing gain, and it is still small
against the 99.87% of an archive that is payload — macros change *which*
program is found, not how the residual bits are modelled. A materially larger
gain needs new primitives.

### Open

- LLM `propose` rounds on top of the mined library — in progress.
- Greedy acceptance can miss a good *set*: the hand-written 4-macro seed beat
  the baseline on both develop models, while several of its members are
  rejected individually. Worth a beam or pairwise pass.
- Mining pools develop models by bytes, so the larger checkpoint's shapes
  dominate the candidate list. `diversify(per_family=…)` limits this but does
  not balance across models.
- Not built, deliberately: new primitives (Zig `forward`/`inverse` pairs are a
  proof obligation, not something to put behind an automatic gate),
  width-dependent operators in bodies, cross-tensor structure (breaks the
  independent-frame invariant), and learning the prior alongside the library.

---

## Earlier

Environment, checkpoint acquisition, and the `main.zig` split are recorded in
`git log` and in [`cluster-pitfalls.md`](cluster-pitfalls.md),
[`checkpoint_acquisition.md`](checkpoint_acquisition.md), and
[`architecture.md`](architecture.md).

Blocked, unchanged: `campaign_runner.py`'s formal path has never run — its
30%-of-filesystem disk gate fails on `/scratch` and `/home`. ZipNN and DFloat11
are not installed. See [`checkpoint_acquisition.md`](checkpoint_acquisition.md).
