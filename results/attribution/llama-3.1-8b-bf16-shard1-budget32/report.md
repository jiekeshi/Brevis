# Brevis tensor/operator attribution

## Exact whole-archive accounting

- Archives: 1; tensors: 82.
- Source: 4.63 GiB; archive: 3.05 GiB; ratio: 1.5185×; saved: 1.58 GiB.
- Tensor payload: 4.63 GiB; tensor records: 3.05 GiB; ratio: 1.5185×.
- Source prefixes: 9.30 KiB; BRTA headers including those prefixes: 9.30 KiB; outer framing overhead: 8.00 B.
- Root `Lit` fallback: 1/82 tensors, 1002.00 MiB source bytes.
- Every embedded safetensors prefix matched one supplied source file exactly.

Role and dtype rows compare exact tensor payload bytes with exact complete record bytes. They exclude the one global BRTA header; whole-archive accounting above includes it.

## By tensor role

| group | tensors | source payload | archive records | ratio | saved | root-Lit |
|---|---:|---:|---:|---:|---:|---:|
| attention | 36 | 720.00 MiB | 478.88 MiB | 1.5035× | 241.12 MiB | 0 |
| embedding | 1 | 1002.00 MiB | 659.88 MiB | 1.5184× | 342.12 MiB | 1 |
| mlp | 27 | 2.95 GiB | 1.94 GiB | 1.5222× | 1.01 GiB | 0 |
| norm | 18 | 144.00 KiB | 60.27 KiB | 2.3894× | 83.73 KiB | 0 |

## By dtype

| group | tensors | source payload | archive records | ratio | saved | root-Lit |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 82 | 4.63 GiB | 3.05 GiB | 1.5185× | 1.58 GiB | 1 |

## By fallback versus synthesized selection

| group | tensors | source payload | archive records | ratio | saved | root-Lit |
|---|---:|---:|---:|---:|---:|---:|
| literal_fallback | 1 | 1002.00 MiB | 659.88 MiB | 1.5184× | 342.12 MiB | 1 |
| synthesized | 81 | 3.66 GiB | 2.41 GiB | 1.5186× | 1.25 GiB | 0 |

## By selected root program

| group | tensors | source payload | archive records | ratio | saved | root-Lit |
|---|---:|---:|---:|---:|---:|---:|
| literal_fallback | 1 | 1002.00 MiB | 659.88 MiB | 1.5184× | 342.12 MiB | 1 |
| synthesized:merge.byte_planes | 1 | 8.00 KiB | 5.77 KiB | 1.3871× | 2.23 KiB | 0 |
| synthesized:merge.fields | 80 | 3.66 GiB | 2.41 GiB | 1.5186× | 1.25 GiB | 0 |

## Program wire composition

| operator | nodes | tensors | exclusive wire bytes | share of all program bytes | root selections |
|---|---:|---:|---:|---:|---:|
| literal | 164 | 82 | 3.05 GiB | 100.000% | 1 |
| merge.byte_planes | 1 | 1 | 3.00 B | 0.000% | 1 |
| merge.fields | 81 | 80 | 324.00 B | 0.000% | 80 |

Exclusive wire bytes are the bytes syntactically owned by a node (its tag/parameters and, for `Lit`, its encoded body), excluding child subtrees. They sum with the five-byte BRPG header per tensor to exact program bytes. They are storage composition, not causal savings.

## Literal physical codecs

| codec | literals | tensors | body bytes | body share | semantic leaf storage | payload bytes |
|---|---:|---:|---:|---:|---:|---:|
| bitpack | 14 | 14 | 538.03 KiB | 0.017% | 4.05 MiB | 538.00 KiB |
| huffman | 17 | 16 | 659.91 MiB | 21.114% | 1002.11 MiB | 659.86 MiB |
| rans | 132 | 69 | 2.41 GiB | 78.869% | 5.48 GiB | 2.41 GiB |
| raw | 1 | 1 | 4.00 KiB | 0.000% | 4.00 KiB | 4.00 KiB |

## Scope and non-identifiable quantities

- Exact here: BRTA/BRPG versions and framing, embedded safetensors metadata, record name/dtype/shape binding, static stream bits/length, tensor source bytes, complete record/program/framing bytes, root fallback versus synthesized selection, node counts, node-exclusive wire bytes, and literal codec/body framing sizes.
- Not performed here: program execution; decoder output/literal/execution-work resource-limit validation; complete entropy table, payload, or padding validation; XXH3 checksum verification; or comparison of tensor payload bytes with the source. Run `brevis verify ARCHIVE SOURCE` independently for those guarantees.
- Not identifiable from BRTA alone: bytes saved *by an individual operator*. The archive persists only the winning program, not valid counterfactual program costs or the search trace. `operators.csv` therefore reports exact occupied wire bytes and root-program cohort results, never invented per-operator savings.
- `literal_semantic_storage_bytes` can exceed tensor source bytes for decompositions such as bit planes. It describes leaf streams and must not be presented as an alternate model size.
