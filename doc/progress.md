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
| unseen checkpoints | `evaluate.verdict` measures the locked `test` tier only, and logs every touch |
| library overhead charged | `evaluate.objective` = archive bytes + `\|Serialize(L)\|` |
| acceptable running cost | `evaluate.MAX_PLANNING_SLOWDOWN` = 1.25× planning wall |
| not just training data | acceptance needs a develop gain *and* no validation regression |

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

Three bugs the tests and the telemetry caught, all real:

- `verify.check` built its gates as a tuple, so all four ran before the first
  refusal could return — a structurally invalid macro still spawned processes
  and an inert one still wrote an archive. Now thunks.
- `mine.evidence` read `search_options_applied["max_expansions"]` when the
  engine puts the resolved configuration under `search`. A wrong test fixture
  had been hiding it; the fixture now mirrors the engine.
- `search.zig` needed macro-usage telemetry to tell "offered" from "chosen".
  Without it the SmolLM regression looked like noise instead of a cost-model
  bug.

### Result, and what a closer look did to it

Two macros survived every gate. **One mined**, with no model involved —
`zigzag(split_field(?,?))`, admitted after seven of eight mined shapes were
refused. **One proposed by `claude-opus-5`**, given only measurements, the
operator table, and the current library:

```
split_float_sign_exp_split_mantissa
  =  split_float(bitpack, rans, split_field(rans, bitpack))
```

Its own justification is a correct reading of the evidence it was shown, and
its last sentence is the thesis of the design, arrived at independently:

> ... A 3-arity root with a nested transform under one child is four operators
> deep and is exactly what a 256-expansion, 12-node search never assembles.

The first reported number was **−2,125,950 bytes (0.122%)** on ViT + TinyLlama.
Four follow-up experiments changed how that number should be read.

#### The reported set was validation, not test

`evaluate.evaluate` vetoed any candidate that regressed the "holdout", so the
holdout participated in every acceptance decision — roughly twelve times — and
`verdict` was run on it twice. That makes it **validation**. The tiers are now
explicit and enforced in `evaluate.Split`:

| Tier | Models | Role |
| --- | --- | --- |
| develop | SmolLM (I8+BF16), BERT (F32) | mined, shown to the proposer, must improve |
| validation | ViT (F32), TinyLlama (BF16) | acceptance veto — steers the library |
| test | Whisper (F16), SDXL (F16, 4 components), Qwen3-8B (BF16, 5 shards) | read only by `verdict`; every touch appended to `runs/test-log.jsonl` |

The proposer itself saw **only SmolLM**: `cmd_propose` benches
`split.develop[0]` and nothing else.

#### The contribution is not the same on every model

`experiments.py budget-curve` sweeps 64…4096 expansions for both grammars.

| Model | Primitive best in sweep | Macro @256 | Verdict |
| --- | ---: | ---: | --- |
| SmolLM | 171,835,356 @512 | **171,726,489** | primitive never matches: **reachability** |
| BERT | 365,681,984 @4096 | 366,000,885 | primitive matches at 512, in *less* wall time: **acceleration at best** |
| ViT | 288,079,353 @512 | 288,080,233 | same: **acceleration at best** |

Two things this exposes that the headline hid:

**At low budgets the effect is large.** At 64 expansions macros are worth
−47.9 MB on BERT (−10.9%) and −36.7 MB on ViT (−10.6%). The honest claim is
about reaching a good program *cheaply*, not about a better ceiling.

**More search is not monotone.** The primitive arm gets *worse* from 512 to
2048 expansions on SmolLM (171.84 → 172.35 MB) and on ViT. "Just give the
baseline more budget" is not a clean control.

#### The gain is a BF16 rule, and only on BF16 is it broad

`experiments.py per-tensor`:

| Model | dtype | macro-built | better/worse | median gain | top-5 share | total |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TinyLlama | BF16 | 174/201 | 175/15 | **8,094 B** | **15.2%** | −2,121,427 |
| SmolLM | I8+BF16 | 212/483 | 196/5 | 7 B | 98.9% | −112,940 |
| BERT | F32 | **0** | 166/31 | 7 B | 46.2% | −5,984 |
| ViT | F32 | **0** | 156/34 | 7 B | 26.6% | −5,070 |

On TinyLlama the effect is genuinely broad — 174 tensors select a macro-built
program, the median improved tensor saves 8 KB, the worst regression is 107
bytes, and the top five tensors are only 15% of the gain. On SmolLM the same
library is 99% five tensors and a 7-byte median, which is bytecode noise. On
both F32 models **the macro never fires at all**; those deltas are reordering,
with 31 and 34 tensors respectively getting *worse*.

So what was learned is a **BF16 float-field rule**, not a general one. I8
tensors gain exactly zero.

#### The LLM-versus-mining comparison does not support a general claim

| Origin | Candidates | Structurally valid | Survived verify | Paid |
| --- | ---: | ---: | ---: | ---: |
| mined | 14 | 14 | 7 | 1 |
| proposed | 6 | 6 | 1 | 1 |

Every one of the six model-written macros was structurally valid — no invented
operator, no arity error. But this is one winner against one winner at unequal
candidate budgets, so "44×" compares *these two macros*, not the two methods.
Worse, **three of the six proposals were judged under the broken cost model**
and rejected for "never fired", which was partly an artifact; those rejections
are not evidence and are still sitting in the ledger's do-not-retry list.

#### The locked test set, measured once

Three checkpoint families never read by mining, by the proposer, or by any
acceptance decision — Whisper (F16, speech), SDXL (F16, diffusion, 4
components), Qwen3-8B (BF16, 5 shards). 26.4 GB, 10 files.

```
SUCCESS — the test tier shrank by 7357344 bytes after charging the
          547-byte library, at 0.93x planning cost
  test  19,390,217,369 -> 19,382,859,478 bytes   (0.038%)
  bit-exact on all 10 files, via real .brv archives
```

The corpus total is not the whole story, and the per-file breakdown is why the
gate now has a per-model bound:

| File | Δ bytes | better/worse tensors | worst tensor |
| --- | ---: | ---: | ---: |
| Qwen3-8B shard 1 | **+4,429,751** | 59 / 8 | +2,179,130 |
| Whisper | **+165,305** | 1183 / 59 | +258,575 |
| SDXL text_encoder_2 | **+91,792** | 489 / 20 | +127,323 |
| Qwen3-8B shards 2–5 | −11,880,057 | 272 / 8 | +271 |
| SDXL unet / vae / text_encoder | −164,682 | 1770 / 196 | +6,945 |
| **total** | **−7,357,891** | | |

Per model: Qwen3-8B −7,450,306, SDXL −72,890, **Whisper +165,305**. So **two of
three models improved and seven of ten files**, the whole net gain is Qwen, and
one Qwen shard is badly hurt while its siblings carry the result.

A corpus-only gate would have waved that through. `MAX_MODEL_REGRESSION` now
caps what any single measured file may lose, at 0.2% of its own archive.

#### Reproducing this

| | |
| --- | --- |
| commits | `25d98ff` engine, `65e1801` loop, `171a9f4` docs |
| toolchain | Zig 0.16.0, Python 3.12.4, `source setup_env.sh` |
| node | c170.nibi.sharcnet, Xeon 6972P, 12 threads (`--jobs 12`) |
| binary | sha256 `583ac71fd32c1de0c7be77aca3c1ed574323a1bc6cbd36cc407a9a5fe1772f15` |
| library | `autodsl/libraries/learned.json`, canonical sha256 `ec3a633804a966bf…` |
| determinism | no RNG in search or codec; identical bytes across repeats *and* across thread counts |
| checkpoints | pinned repo + revision + per-file sha256 in `eval/models-tiered.json` |

```bash
source setup_env.sh && zig build -Doptimize=ReleaseFast
cd autodsl
python3 loop.py bootstrap --top 8 --per-family 1      # mined half, no model
python3 loop.py propose --rounds 1 --wanted 3         # model half
python3 loop.py verdict --tier test                   # the claim
python3 experiments.py --run exp budget-curve         # acceleration vs reachability
python3 experiments.py --run exp per-tensor           # where the bytes came from
python3 experiments.py --run exp archives             # real .brv, hashed both ways
```

Cost accounting, so "0.93× planning" is not read as free:

- A macro costs one A\* pop, but `fitMacroBody` runs `ops.forward` for every
  body node and builds a histogram per node, so its *per-pop* work is higher
  than a primitive's. `planning_wall_ms` measures the whole planner, so the
  0.93× is after paying that.
- The library-discovery phase is offline and is not in any reported number:
  ~40 engine invocations for the bootstrap, 2 model calls, and roughly two
  hours of wall clock on this node.
- The baseline is held at the same expansion budget. `budget-curve` is the
  place where equal-wall-time is compared instead, and it is reported there.

---

## 2026-07-25 (later) — From greedy acceptance to a population search

The loop up to here proposed one macro at a time and kept each that paid on its
own. That cannot find a *set*: two macros can cover different tensors, and a
macro that loses alone can win beside another. Directly measured here — a
hand-written four-macro library beat the baseline on both develop models while
most of its members are refused individually.

### What changed

A **library**, not a macro, now carries a fitness. `variation.py` supplies the
moves — random generation, four body mutations, member dropping, and crossover
between two libraries — all pure functions of `(library, table, rng)`, so the
loop is reproducible from a seed and every operator is testable without the
engine. `search.py` runs a population of libraries, recombines them, feeds the
measured history back to the proposer, and stops on patience or budget.

**Every arm spends the same currency.** One *evaluation* is one fitness
measurement of one library over the develop tier. The cache is shared across
arms, so revisiting a library is free for everyone. When an arm's budget is
gone it stops, whatever it was doing. Validation is applied once per arm, to
its final library, as a veto.

### The arena: five arms, 30 evaluations each, one seed

| Arm | develop Δ | validation Δ | evals | rounds | macros |
| --- | ---: | ---: | ---: | ---: | ---: |
| **search** | **−735,741** | −2,132,909 | 30 | 4 | 2 |
| greedy | −32,095 | **−2,296,109** | 18 | 6 | 1 |
| mining | −10,637 | −48,978 | 5 | 5 | 1 |
| random | 0 | 0 | 4 | 4 | 0 |
| one_shot | 0 | 0 | 1 | 1 | 0 |

**The gain is not a single lucky hit.** The search arm's best-so-far by round:

```
537,823,854 → 537,727,913 → 537,727,869 → 537,110,605
       −22,492      −95,941           −44        −617,264
```

Four rounds, improving at each, and the largest step is the *last* one. It also
exhausted its budget rather than converging, so 30 evaluations is a floor on
what it can find, not a ceiling.

**The winning macro came from mutation, not from the model.** The final library
is `zigzag(split_field(?,?))` (mined) plus:

```
split_float(?, rans, split_field(?, bitpack))        [mutated]
```

which is the search generalising the model's earlier
`split_float(bitpack, rans, split_field(rans, bitpack))` — two pinned terminals
opened back into holes. The generalisation is worth 23× the proposal it came
from. Neither mining nor the proposer would have produced it: mining only
recovers shapes already being found, and the model wrote the specific form.

**One result cuts against the ranking.** Greedy won *validation* by a little
while losing develop by 23×. Develop order is therefore not transfer order, and
the search arm may be fitting develop. Only the frozen test tier settles it.

### The frozen tier disagrees with develop, and that is the result

`search.py --judge` measured every arm's final library once on the locked test
tier (Whisper F16, SDXL F16, Qwen3-8B BF16 — 26.4 GB, 10 files), against one
shared baseline, with a real `.brv` roundtrip per non-empty library.

| Arm | develop Δ | validation Δ | **test Δ** | files better | worst file | planning | bit-exact |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| greedy | −32,095 ② | −2,296,109 ① | **−8,430,499 ①** | 6/10 | +4,131,850 | 0.95× | yes |
| search | **−735,741 ①** | −2,132,909 ② | **−7,379,859 ②** | 7/10 | +4,427,091 | 1.22× | yes |
| mining | −10,637 ③ | −48,978 ③ | **+7,954,920** | 7/10 | +8,233,466 | 0.98× | yes |
| random | 0 | 0 | 0 | — | — | — | — |
| one_shot | 0 | 0 | 0 | — | — | — | — |

**The population search won develop by 23× and lost the frozen tier to plain
greedy by 1.05 MB.** Develop order is anti-correlated with test order at the
top. It also spent 1.22× the planning time against greedy's 0.95×.

Against the standard set for this work — *stably better than one-shot, greedy,
random and pure mining at a fixed compute budget* — the answer is **no**. It
beats one-shot, random and mining; it does not beat greedy.

The mechanism is visible rather than mysterious. `zigzag(split_field(?,?))`,
the mined macro, is **+7,954,920 bytes on the test tier all by itself**. The
search arm's library contains it. The search has a drop mutation and never
used it, because dropping that macro costs develop bytes — it is worth a great
deal on SmolLM and nothing on unseen models. A fitness that sums the tier lets
one model buy the whole score.

So the failure is in the objective, not the machinery: two develop models
summed is far too weak a signal for a search with population, mutation and
crossover behind it. The next run replaces the summary with **minimax** — a
library scores as well as its *worst* develop model, which makes
"win big on one, lose on the rest" unscoreable — and moves ViT into the
fitness pool so there are three models to be worst over. ViT had already been
read once per candidate as an acceptance veto, so it was never test material.
The test tier is unchanged.

### Honest notes on the arena

- `random` and `one_shot` produced nothing at all, so "beats random" and
  "beats one-shot" here means "beats zero". That is a real result for those
  arms under this budget, not a strong comparison.
- `mining` and `random` stopped on patience after 4–5 evaluations, well under
  the cap. They could not spend the budget; that is a property of the method,
  and the cap was identical for all five.
- One seed. Variance across seeds is unmeasured.

### Open

- **The library is BF16-shaped.** It never fires on F32 and makes one F16 model
  worse. Either gate per dtype, or learn a library per dtype family.
- **Three test models is not eight to twelve.** Cached and unused: the 30B MoE
  pair (BF16 and FP8) would add MoE and FP8 coverage. No optimizer state or
  non-weight tensors anywhere in the corpus.
- **The mining-versus-proposing comparison is not controlled.** Equal candidate
  budgets, plus random-legal-macro and human-designed-macro arms, are needed
  before any claim that one method beats the other.
- **Three early proposals were judged under the broken cost model** and are
  still on the ledger's do-not-retry list. They should be re-run.
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
