# BRTA v1 engineering pilot

Date: 2026-07-27

This note records engineering evidence from a dirty working tree based on
commit `ad2d71ef798a3caf65c69ee14a630da2986092ba`. It is not a source of
quantitative paper claims. Every archive cited below was verified against its
complete input before its structure was inspected.

The executed `ReleaseFast` binary has SHA-256
`012b7b9265aec91921b2a2d8e713f37d7d832ddc7b8674ec101c52fae2e96e6d`.
The machine was an Apple M2 with eight logical cores and 24 GiB of memory.

## Main conclusion

The existing grammar has genuine utilization space. Wider search selected
programs with two to four semantic operations on real BF16, F32, and I8 tensors,
and every strictly improved composition beat the same tensor's fully explored
depth-one counterfactual in exact serialized bytes. A PHOG prior calibrated on
32 Llama tensors both increased composition depth and transferred to a
different SmolLM checkpoint.

No new production was added. The current evidence favors better reachability,
guidance, canonicalization, and execution over a broader DSL:

- Llama BF16 reached 102 composed programs out of 145, with as many as three
  semantic operations.
- SmolLM BF16 reached 148 composed programs out of 271, with as many as four
  semantic operations under a prior learned on Llama.
- BERT F32 reached 20 composed programs out of 129.
- The tested I8 subset reached two composed programs out of 16, but their
  combined gain over depth one was only 14 bytes.

This supports the paper's narrow mechanism: a small tensor-generating-program
language searched by PHOG-guided bounded A*. It does not support a global
optimality claim.

## Inputs

| Input | Complete file bytes | Tensors | Physical dtype in subset | SHA-256 |
| --- | ---: | ---: | --- | --- |
| Llama small-tensor subset | 904,752 | 145 | BF16 | `0a694dba8fdff8c4db69886ec8ddd943a23de41dec1f3ed6ec37148eac082d98` |
| SmolLM tensors up to 64 KiB | 410,888 | 271 | BF16 | `b2be2a4823f7d9836dc69331d0a096abe2ad206d3b7cccc7b643ed39eff15f77` |
| BERT tensors up to 64 KiB | 524,000 | 129 | F32 | `6092dae8312c4568075599926e9a9ce01d4ff041e3610674da774201c72d8bd5` |
| SmolLM smallest 16 I8 tensors | 1,771,216 | 16 | I8 | `fbb821b1621e0c3a5057baf22debf372eb9e48e622790a8247070d07a5a5fc0b` |
| Complete SmolLM W8A8/BF16 | 219,850,592 | 483 | I8 and BF16 | `1f6ee9463560573bcccb6b48dfd89827ad9d49a4f50b6123a35b6f4efafb3baf` |
| Complete BERT | 440,449,768 | 206 | F32 | `68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3` |
| Complete Llama-3.2-1B FP8 dynamic | 2,024,671,096 | 259 | FP8 and BF16 | `e24862bf8898e483b30c924721debd7b622ef69e0f78d6552c3bd605b5d7cd19` |

Subsets were repacked in source-header order. Their byte totals include their
complete safetensors headers and data. They are mechanism probes, not
complete-model compression ratios.

## Real composed programs

`semantic operations` count `Concat`, `Repeat`, `Map*`, `Scan*`, and `Merge*`.
`Literal` and `Constant` are terminals. The depth-one counterfactual exhausts
the configured depth-one search space; the deeper runs stop at their expansion
budgets.

### Llama BF16

| Guidance and budget | Archive bytes | Program bytes | Tensors with at least two semantic operations | Maximum semantic operations |
| --- | ---: | ---: | ---: | ---: |
| terminal only | 533,127 | 504,845 | 0 | 0 |
| exhausted depth one | 433,957 | 405,675 | 0 | 1 |
| uniform, 1,024 expansions | 433,083 | 404,801 | 58 | 2 |
| PHOG, 1,024 expansions | 431,741 | 403,459 | 102 | 3 |

All 58 uniform compositions were strictly smaller than their own depth-one
programs, saving 874 program bytes in total. All 102 PHOG compositions were
strictly smaller, saving 2,216 bytes. Examples include
`MapAdd(MergeFields(...))`, `MapAdd(ScanXor(...))`, nested field merges, and
`MergeFields(MapAdd(MergeFields(...)), ...)`.

The PHOG prior used deterministic calibration on 32 of the 145 tensors,
uniform search with 512 expansions, and no shallow float-field seed. It is
1,823 bytes with SHA-256
`ecd314bfecc75f449edb6e916bc0fc56a62f04a8f17c2115c00551b3212a7aac`.
On the remaining 113 tensors, the same-budget PHOG programs used 316,113
bytes, versus 317,091 bytes under uniform guidance. PHOG therefore saved 978
held-out program bytes and found 75 composed programs instead of 40. This is a
within-checkpoint tensor split, not an independent-checkpoint result.

### Cross-model PHOG transfer to SmolLM BF16

| Guidance and budget | Archive bytes | Program bytes | Tensors with at least two semantic operations | Maximum semantic operations |
| --- | ---: | ---: | ---: | ---: |
| terminal only | 349,268 | 296,493 | 0 | 0 |
| exhausted depth one | 246,818 | 194,043 | 0 | 1 |
| uniform, 1,024 expansions | 245,857 | 193,082 | 85 | 2 |
| Llama PHOG, 1,024 expansions | 244,944 | 192,169 | 148 | 4 |

The transferred prior saved 913 archive and program bytes relative to uniform
guidance at the same budget. Of its 148 composed programs, 145 were strictly
smaller than their depth-one counterparts and three tied; none was larger.
The total gain over depth one was 1,874 program bytes. The final archive
SHA-256 is
`cea9539f53f9a52f036725a5d72fa81d702d8e701dabd68df1ce127c159826b7`.

This is the strongest pilot evidence that the contextual prior does more than
memorize one checkpoint's tensor order. It remains one transfer pair, so it is
not a cross-model aggregate. Relative to uniform guidance, PHOG improved 89
tensors, tied 180, and regressed two, for the stated net gain. It is not
per-tensor dominant.

### F32 and I8

| Input and search | Terminal archive | Exhausted depth-one archive | 1,024-expansion archive | Composed tensors | Composition gain over depth one |
| --- | ---: | ---: | ---: | ---: | ---: |
| BERT F32 | 533,041 | 445,946 | 445,751 | 20 / 129 | 195 bytes |
| SmolLM I8 | 1,620,133 | 1,597,597 | 1,597,583 | 2 / 16 | 14 bytes |

The F32 compositions were 13 `MapAdd(MergeFields(...))` and seven
`MapGray(MergeFields(...))` programs; all 20 were strictly smaller than depth
one. The two I8 compositions were `MapAdd(MergeFields(...))`, each seven bytes
smaller than its depth-one counterpart. Larger budgets therefore expose real
composition across these dtypes, but the incremental exact-byte gains are
small.

## Search and grammar changes motivated by the pilot

- The final permitted expansion is now evaluated for complete candidates
  before reporting budget exhaustion.
- PHOG costs are applied to the production families actually proposed for the
  current target.
- The default learned/uniform mixture uses `lambda = 1`. Additive smoothing
  keeps every admitted production family at nonzero probability, and contexts
  unmatched at every backoff level use uniform.
- A semantics-preserving search normal form omits identities and
  target-specific aliases. It retains the shorter parameter-free
  `MergeBytePlanes` or `MergeBitPlanes` form when it is identical to a
  parameterized field split, and does not change the accepted DSL or wire
  format.
- F32 search may initialize its incumbent with the existing shallow
  float-field program. Exact serialized size still selects the winner, and A*
  continues after initialization.
- `Concat` children must have positive lengths. Only `Literal` may represent
  an empty stream, so recursive well-formedness and target decomposition agree.

The evidence does not justify adding RLE, sparse, palette, or arbitrary
multidimensional productions. A shape-derived strided scan remains a
measurement-gated future candidate, not part of this revision.

## Complete-model checks

These runs test whole-file behavior and larger tensor sizes. Search-heavy
composition experiments remain on the deterministic subsets above because
repeatedly scanning hundreds-of-megabytes targets is expensive.

| Complete input | Configuration | Archive bytes | Input saving | Exact |
| --- | --- | ---: | ---: | --- |
| BERT F32 | shallow F32 seed, one expansion, 8 workers | 365,059,643 | 17.12% | yes |
| BERT F32 | no seed, one expansion, 8 workers | 440,467,190 | -0.004% | yes |
| Llama-3.2-1B FP8/BF16 | terminal only, serial | 1,486,925,059 | 26.56% | yes |
| Qwen2.5-7B shard 1, 3.95 GB | terminal only | 2,637,595,487 | 33.15% | yes |

The BERT seeded archive SHA-256 is
`3f4cc84375bd523a8f7b0656525b3cdef1d960a4ff25b42fba4b9614160db0de`.
The full Llama checkpoint establishes current FP8 and BF16 terminal coverage,
not useful composition on every large weight matrix. The Qwen result applies
only to shard 1, not the complete four-shard checkpoint.

## Execution and I/O work

The implementation changes preserve canonical archive bytes:

- independent tensor records are compressed and decompressed in a bounded
  source-ordered sliding window, with at most one active outcome per worker;
- each emitted result immediately reuses its window slot, removing the fixed
  batch barrier without raising the window above the worker count;
- the file encoder has an explicit borrowing seam for a winning root
  `Literal`, while the public synthesis API remains owning;
- zero-budget terminal synthesis produces the winner's canonical bytecode once
  and archive framing reuses it;
- structured literal leaves adopt decomposition buffers instead of copying
  them again;
- decoded root literals refer directly to their record payload until execution;
- rANS encoding replaces hot-loop division with exact reciprocal
  multiplication and has differential tests over every normalized frequency;
- rANS decoding uses the declared canonical body length and no longer
  re-encodes a decoded literal;
- rANS decoding dispatches once on 8-, 16-, or 32-bit output storage instead of
  testing the storage width for every symbol;
- uniform search skips PHOG target-context feature construction while keeping
  its production costs and candidate order unchanged;
- file output uses synchronized atomic replacement;
- the generic harness uses explicit 1 MiB staging copies without clone or
  sparse-seek shortcuts and includes output `fsync` in compressor wall time.

The reciprocal rANS microbenchmark reduced the measured single-core encode loop
from about 0.60 to 0.38 seconds on this machine. This isolated result is
diagnostic; the end-to-end record below is the primary engineering timing.

## Repeated SmolLM throughput record

The machine-readable result is
`eval/results/engineering/brta-v1-smollm-terminal-m2-2026-07-27.json`, SHA-256
`eaa6a0dcd05c0b4a6b76ff068aaa2771df94398b088d93ae2afa650ddb44e945`.
It uses one warmup and three measured runs per worker setting, alternated in
forward and reverse order. Hashing and exact comparison are outside the timed
process. BRTA's synchronized atomic output is inside its measured command.

| Workers | Archive bytes | Median compression | Median decompression | Maximum measured compression RSS | Maximum measured decompression RSS |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 171,762,137 | 3.271 s | 2.237 s | 294,682,624 | 235,814,912 |
| 8 | 171,762,137 | 1.640 s | 1.097 s | 267,386,880 | 196,329,472 |

All six measured archives are byte-identical, with SHA-256
`509fb5709d7f8cb51ed93d80dd6580a202a809eb628802d5a9137b29c85ee7e5`,
and all six restored files are exact. Median speedup is 1.99x for compression
and 2.04x for decompression. This is useful multicore scaling, not linear
scaling. The harness records best-effort buffered I/O and does not claim an
isolated host or fixed cache state.

## Follow-up performance audit

The machine-readable records are under
`eval/results/engineering/performance-audit-2026-07-27/`. They use the same
complete SmolLM input and retain every observation. The stable medians below
come from one warmup and three to five measured repetitions:

| State | Workers | Compression | Decompression |
| --- | ---: | ---: | ---: |
| frozen terminal baseline | 1 | 1.682 s | 1.248 s |
| frozen terminal baseline | 8 | 0.602 s | 0.482 s |
| optimized successful path | 1 | 1.121 s | 1.224 s |
| optimized successful path | 8 | 0.404 s | 0.436 s |

All archives remain 171,762,137 bytes with the same SHA-256 stated above, and
every restore is bit-exact. The largest single-worker gain came from producing
the zero-budget terminal winner's canonical bytecode once and reusing it in
archive framing: 1.682 to 1.137 seconds, or 32.44%. The source-ordered sliding
window then reduced the prepared eight-worker result from 0.440 to 0.404
seconds for compression and from 0.481 to 0.436 seconds for decompression.
Dispatching rANS decode storage width once changed single-worker decompression
by only 0.54%.

Uniform 64-expansion search on the Llama BF16 subset also stopped constructing
unused PHOG target contexts. Its five-run wall median fell from 1.18 to 0.73
seconds while archive bytes, search counters, and candidate order remained
identical.

The current source additionally tracks active future slots explicitly so
eager-completed tasks and truncated initial decode windows are always canceled
or consumed exactly once. This is an error-cleanup repair; a quiet-host timing
of that exact later binary remains pending. `final-current.json` binds the
later ReleaseFast binary and verifies six exact round trips, but its wall
samples are visibly unstable under unrelated host load. Those samples and the
earlier retained contended diagnostic are excluded from the table rather than
averaged into the stable measurements.

## General-purpose compressors

The repeated BRTA record embeds a matched-input zstd 1.5.7 default-level
comparison. The separate ultra record is
`eval/results/engineering/generic-smollm-zstd-ultra22-m2-2026-07-27.json`,
SHA-256
`dfaba56fa0581484a6f992ccea6dcf25fb3252fee52b632ac35c6f0cc2596496`.
Both generic configurations use `--single-thread --no-asyncio`, synchronize
the archive and restored file before stopping their wall clocks, and verify the
complete restored input.

| SmolLM method | Archive bytes | Input saving | Compression | Decompression | Repetitions |
| --- | ---: | ---: | ---: | ---: | ---: |
| zstd `-3` | 185,705,518 | 15.53% | 1.316 s median | 0.658 s median | 3 after 1 warmup |
| BRTA terminal, 1 worker | 171,762,137 | 21.87% | 3.271 s median | 2.237 s median | 3 after 1 warmup |
| BRTA terminal, 8 workers | 171,762,137 | 21.87% | 1.640 s median | 1.097 s median | 3 after 1 warmup |
| zstd ultra `-22` | 140,744,958 | 35.98% | 138.143 s | 0.623 s | 1 |

BRTA is 13,943,381 bytes, or 7.51%, smaller than the zstd default archive, but
zstd default is much faster. Zstd ultra is 31,017,179 bytes smaller than BRTA
and decodes faster in this one-shot run, but takes roughly 42 times
the BRTA serial compression time. A previous one-shot xz `-9e` result was also
smaller than BRTA. The correct conclusion is a tradeoff, not that BRTA
dominates general-purpose compressors.

On complete BERT F32, the seeded BRTA archive is 365,059,643 bytes. A one-shot
zstd ultra `-22` archive is 407,128,615 bytes, so BRTA is 42,068,972 bytes
smaller on that input. The corresponding input savings are 17.12% and 7.57%.
The zstd run took about 212 seconds to compress. This favorable F32 result and
the unfavorable SmolLM ultra result both need repeated frozen evaluation before
paper use.

Applying zstd `-3` outside the SmolLM BRTA archive saved only 132,200 bytes, or
0.077%. This does not justify adding an outer compressor to the format.

## Evidence boundary

The AST inventory is structural evidence only; `brevis verify` remains the
semantic check. The small-subset search sweeps are deterministic but mostly
one-shot. The repeated throughput records bind the input, binary, harness,
commands, executable hashes, output hashes, durability status, and dirty-tree
provenance, but they still do not constitute a clean paper campaign.

The manuscript therefore reports the corrected method and evaluation protocol
without importing these numerical outcomes. A paper result still requires a
clean frozen revision, preregistered model splits, repeated schedules, and the
declared resource controls.
