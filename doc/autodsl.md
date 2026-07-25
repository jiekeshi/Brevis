# Self-extending DSL (`autodsl/`)

Brevis searches for a program per tensor inside a fixed operator set. That set
is an inductive bias: a regularity the grammar cannot name is a regularity the
search cannot find, no matter how good the prior is. `autodsl/` closes the
outer loop — it proposes additions to the grammar, verifies them, and keeps the
ones that measurably pay.

```
inner loop   A* searches for a program per tensor, in the current DSL
outer loop   mine + propose new macros -> verify -> measure -> keep or reject
```

The model is confined to *proposing*. Everything that decides what survives is
deterministic and replayable from the ledger.

## What a macro is

A macro is a named subtree of **existing** operators whose leaves are holes
(the search continues there) or terminals (the branch is closed).

```json
{"name": "zigzag_split",
 "body": {"op": "zigzag", "children": [
   {"op": "split_field", "children": [{"op": "hole"}, {"op": "hole"}]}]}}
```

Three properties follow from that definition and matter more than anything
else in the design:

**A macro cannot hide data.** Its body holds operators and holes, never
payload. The degenerate `GenerateTensor()` rule — where a "program" is three
bytes because the tensor moved into the grammar — is not penalised here, it is
unrepresentable. Every byte still leaves through a terminal payload and is
counted in full.

**A macro cannot change what is expressible.** Every body node is checked
against `legalProductions` at the position it lands in, so a macro can only
build programs the primitive grammar could also build. Width guards, the depth
budget, and `--disable-op` all still apply; an ablation that removes `zigzag`
also removes every macro containing it.

**A macro cannot change an archive.** The engine expands macros into primitives
*before* serializing. The `.brv` schema, the bytecode, and the decoder are
untouched, and an archive written with a library is byte-identical to one
written without it whenever both select the same program. Decompression never
learns that macros exist.

What a macro *does* change is **reachability under a fixed budget**. Applying
one costs a single A\* expansion and delivers its whole subtree.

## Why the obvious MDL objective is the wrong one here

The natural formulation for a self-extending grammar is

    minimize  |Serialize(L)| + Σᵢ |Serialize(Pᵢ | L)|

Measured on this codebase, both terms are noise:

| Term | Share of the archive |
| --- | ---: |
| terminal payloads | 99.87% |
| program bytecode, all blocks | 0.13% |
| the library itself | 0% — expanded before serialization |

Optimizing either would be optimizing a rounding error. Shortening a program's
bytecode by a third would move an archive by 0.04%.

What a library actually spends is **search budget**. Every macro is offered at
every hole, so a larger library explores fewer distinct structures within a
fixed expansion budget. So the enforced objective is

    minimize  Σᵢ archive_bytes(tensorᵢ | L)     at a fixed search budget B

`|Serialize(L)|` is still computed and recorded on every decision, because it
is the term that becomes binding the moment a library ships inside an archive,
and because a claim about MDL should show the term it claims is negligible.

This substitution is the one real departure from the design as originally
proposed, and it is a measurement result, not a preference.

## The three facts the design rests on

Measured, not assumed. The numbers behind them and the running log are in
[`progress.md`](progress.md).

**The search is budget-saturated.** Every tensor on SmolLM exhausts its
256-expansion budget, and 18 distinct programs cover 1245 blocks. Yet raising
the budget or the depth buys ~0% there, while on BERT it buys 0.17% — and
inspecting the winners showed that gain came from a *depth-1* program the small
budget simply never reached. The limit is what the search gets to, not what the
grammar can say. That is the gap a macro closes.

**Measurement is deterministic**, including across thread counts, so any
non-zero delta is signal. The acceptance margin exists to filter *unimportant*
gains, not noisy ones.

**Adding productions can lose.** A hand-written seed library made BERT 10,741
bytes smaller and SmolLM 92,674 bytes larger — and on SmolLM *no tensor
selected a macro-derived program at all*. The macros never won; they consumed
expansions that would have found something better. Hence `verify.fires`, and
hence "the macro looks sensible" is not evidence.

## The gates

A candidate must survive all of these, cheapest first:

1. **Structural** — arity, known operators, no width-dependent operator in a
   body, unique name, unique shape. No process is started if this fails.
2. **Engine accepts** — the library loads and every macro survives parsing.
3. **Fires** — some tensor actually selects a program the macro built.
   Rejecting here is cheap and catches the common failure.
4. **Bit-exact** — a real `.brv` is written and `brevis verify` confirms it.
   This should never fail, which is why it is worth running: a failure means an
   assumption broke.
5. **Pays** — archive bytes must fall on the develop set by more than the
   margin, *and* must not rise on the holdout set by more than the tolerance.

Proposals are written against develop-set evidence, so gating on develop alone
would reward memorizing it. With four cached checkpoints this is a weak split
and every ledger entry records it as such.

## Running it

```bash
source setup_env.sh
zig build -Doptimize=ReleaseFast
cd autodsl

python3 loop.py evidence                 # where this model's bytes are going
python3 loop.py mine --top 20            # shapes the search keeps rediscovering
python3 loop.py bootstrap --top 6        # gate mined shapes; no model involved
python3 loop.py propose --rounds 3 --model-name claude-opus-5
python3 loop.py status --measure
```

`--model-name` takes `claude-opus-5` (Anthropic protocol), `qwen3.7-max`, or
`qwen3.6-plus` (OpenAI-compatible), all through the CloseAI proxy. The key is
read from `llm.credential` (git-ignored) or `$BREVIS_LLM_API_KEY`.
`--dry-run-reply` exercises the whole loop with no network and no key.

To use a library outside the loop, any command that takes search options takes
`--macros`:

```bash
./zig-out/bin/brevis bench model.safetensors --macros autodsl/libraries/learned.json
./zig-out/bin/brevis config --macros autodsl/libraries/learned.json
```

## Layout

| File | Role |
| --- | --- |
| `src/macro.zig` | Library parsing and validation. No search, no I/O beyond loading. |
| `src/search.zig` | `fitMacroBody` grafts a body into a hole; the expansion loop offers each macro. |
| `autodsl/library.py` | The macro model, canonical hashing, validation. |
| `autodsl/engine.py` | The only place the `brevis` binary is invoked. |
| `autodsl/mine.py` | Frequent-subtree mining, and the evidence a proposer is shown. |
| `autodsl/backend.py` | LLM transport: Anthropic native and OpenAI-compatible. |
| `autodsl/propose.py` | Prompt construction and reply parsing. |
| `autodsl/verify.py` | Gates 1–4. |
| `autodsl/evaluate.py` | Gate 5, and the MDL accounting. |
| `autodsl/ledger.py` | Append-only record of every proposal and its fate. |
| `autodsl/loop.py` | The driver. |

The operator table in the prompt and in every validator comes from
`brevis config`, so the DSL is declared once, in `src/ops.zig`.
`propose.SEMANTICS` is the one hand-written table — what an operator *means* is
not in the config dump — and `test_propose.py` fails if the engine grows an
operator it does not describe.

## Where it stands

Two macros have survived every gate, one mined and one proposed by
`claude-opus-5`. On the holdout split they are worth **−2,125,950 bytes
(0.122%)** after charging the 547-byte library, bit-exact, at 0.89× planning
cost. The proposed macro accounts for 44× what the mined one does — mining
recovers shapes the search mostly finds anyway, while the proposer reached
`split_float(bitpack,rans,split_field(rans,bitpack))`, a structure legal in the
primitive grammar that a 256-expansion budget never assembles. Numbers, the
run log, and the caveats are in [`progress.md`](progress.md).

## Status and limits

Development apparatus, not audited measurement. `eval/` enforces gates this
does not (clean tree, harness hashing, repetition fingerprints, unset codec
environment variables) and nothing produced here may be reported as a formal
result. `autodsl/runs/` is git-ignored; `autodsl/libraries/` is not, because a
learned library is a deliverable.

Deliberately not built yet:

- **New primitives.** Only macros over existing operators. A genuinely new
  operator needs a `forward`/`inverse` pair in `ops.zig`, which is Zig code and
  a real proof obligation — it does not belong behind an automatic gate.
- **Width-dependent operators in bodies.** `bit_plane` and `byte_plane` have
  arity that depends on the stream, so a body containing one has no fixed
  shape. They cover ~1% of observed winning programs.
- **Cross-tensor structure.** Shared codebooks, expert-to-expert references,
  and low-rank residuals all require breaking the independent-frame invariant
  that makes sharded models tractable. That is an archive-format decision, not
  a grammar one.
- **Learning the prior alongside the library.** `brevis calibrate` still fits
  the PHOG over primitives only; a macro is scored by what its expansion would
  cost at best, so it gains no ordering advantage it did not earn.
