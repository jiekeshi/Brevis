# BRTA v1 performance audit

These are dirty-worktree engineering measurements on an Apple M2. They are not
paper-eligible results. The input is the complete 219,850,592-byte SmolLM W8A8
checkpoint with SHA-256
`1f6ee9463560573bcccb6b48dfd89827ad9d49a4f50b6123a35b6f4efafb3baf`.
Every measured restore is bit-exact. All BRTA archives below are 171,762,137
bytes with SHA-256
`509fb5709d7f8cb51ed93d80dd6580a202a809eb628802d5a9137b29c85ee7e5`.

The harness runs complete child processes, includes Brevis's synchronized
atomic output in the command time, and performs hashing and byte comparison
after timing. Each stable record uses one warmup. Medians are seconds.

| Record | Workers | Compression | Decompression | Repetitions |
| --- | ---: | ---: | ---: | ---: |
| `terminal-baseline.json` | 1 | 1.682 | 1.248 | 3 |
| `terminal-baseline.json` | 8 | 0.602 | 0.482 | 3 |
| `prepared-bytecode.json` | 1 | 1.137 | 1.231 | 3 |
| `prepared-bytecode.json` | 8 | 0.440 | 0.481 | 3 |
| `sliding-window-w8.json` | 8 | 0.404 | 0.436 | 5 |
| `rans-width-before.json` | 1 | 1.132 | 1.230 | 3 |
| `rans-width-after.json` | 1 | 1.121 | 1.224 | 3 |

The zero-budget prepared-bytecode path reduced single-worker compression wall
time by 32.44% and eight-worker compression by 26.95% relative to the frozen
baseline. Replacing fixed batches with a source-ordered window no larger than
the worker count reduced the prepared eight-worker result by another 8.13% for
compression and 9.45% for decompression. Combined against the baseline,
eight-worker compression fell by 32.89% and decompression by 9.63%.

The rANS output-width specialization changed single-worker decompression by
only 0.54%. It is a small branch cleanup, not a material system speedup.

The final successful-path binary represented by `rans-width-after.json` and
`sliding-window-w8.json` had SHA-256
`881d909f82612429ee1711ff1b3569db1b8989c1cd054733c8c05c607c7f75b1`.
The later source revision adds explicit active-slot cleanup for eager-completed
futures and truncated initial windows. That repair affects error cleanup and
adds only per-slot bookkeeping. `final-current.json` binds that exact later
ReleaseFast binary, SHA-256
`299610330e0873ba302316c1025fb178602b64bdf812d7ced812bf94cef578ef`,
and verifies six more exact round trips. Its timing remains unsuitable for an
absolute quiet-host result: single-worker compression ranged from 1.948 to
3.107 seconds and eight-worker compression from 0.784 to 1.107 seconds while
unrelated host processes were active.

## Uniform-search context A/B

A separate five-run A/B used the 904,752-byte Llama BF16 subset with SHA-256
`0a694dba8fdff8c4db69886ec8ddd943a23de41dec1f3ed6ec37148eac082d98`,
uniform guidance, 64 expansions, no shallow seed, and eight workers. Moving
PHOG target-context construction inside the learned-prior branch changed the
wall median from 1.18 to 0.73 seconds, a 38.1% reduction. User CPU fell from
6.26 to 3.58 seconds. All ten archives were 434,304 bytes with SHA-256
`dd545f3e3860131aa24e08191cc6bb64f5abcf8d36b043040f7b7be5c34567a7`;
search counters were also identical.

## Contended diagnostic

`contended-final-diagnostic.json` and `final-current.json` document rejected
timing runs. During them, unrelated host load coincided with raised Brevis CPU
time and unstable samples. Their exactness and archive checks passed, but their
wall times are not used in the stable table.
The first file's embedded zstd 1.5.7 default runs completed exactly with a
185,705,518-byte archive and medians of 0.518 seconds for compression and 0.432
seconds for decompression. Because host conditions changed between the Brevis
and zstd phases, it does not support a paired speed-dominance claim.

## File hashes

| File | SHA-256 |
| --- | --- |
| `terminal-baseline.json` | `e61275fa33b5ca8dc4b2335389aa47bf832fe3d308a5407ae65f073a59a79f0c` |
| `prepared-bytecode.json` | `d16f80998d2218a46d5d81a174f520e1dc9d831ccc43ea38f38a4e4c70901358` |
| `sliding-window-w8.json` | `0bb240673e4461e3ac49f3e35e5a312c2581db744d69e7d7583071223ef3ebad` |
| `rans-width-before.json` | `f725234e45e2c7efa749f5daa6b7b2628e6358cff3b0c27d4f724a5c54ef815d` |
| `rans-width-after.json` | `45570ceaf71742feeabe5dd26869daff757a503295699e52c3d5e37dbaddabae` |
| `contended-final-diagnostic.json` | `5951d98d26b70a457c37f55a6145145fcb7560d14c0f6761a9ce186fb76f0d84` |
| `final-current.json` | `61c20107f621bcd84b4b1c684d41eeac1fac55cfe0e91adb9f7fccfbcc14e927` |
