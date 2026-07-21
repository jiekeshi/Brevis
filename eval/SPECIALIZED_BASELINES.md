# Specialized Lossless Baseline Protocol

Status: preregistered on 2026-07-21; no baseline in this document has been
installed or run. The machine-readable source of truth is
`eval/specialized-baselines.json`. A planned configuration is not a result.

## Comparison boundary

The primary object is an exact file named by `eval/models-tiered.json`, not an
abstract tensor collection and not a model that merely produces the same task
outputs. A successful decompressor must reproduce the complete safetensors file
byte for byte. This includes the length word, JSON header, metadata, tensor
order, padding or gaps, tensor payloads, and trailing bytes. The manifest-bound
input and restored output must have the same length and SHA-256.

The charged archive size is the sum of every retained file or byte range needed
for a fresh decode without the original. It includes codec payloads, raw
fallbacks, unsupported dtypes, headers, codebooks, offset and dtype tables,
pipeline descriptions, and input-specific decoder artifacts. A pinned,
input-independent executable may be excluded in the same way that the gzip and
Brevis executables are excluded. If an input-specific generated executable is
the only way to decode a file, that executable is part of the archive cost.

This boundary is stricter than tensor equality. In particular, a tool that
loads a safetensors file and later calls `save_file` may preserve tensor values
while changing the serialized header or order. Such a path is not eligible
until a strict adapter restores the original bytes.

## Fixed sources and scope

| Method | Fixed source | Preregistered scope | `domain_match` interpretation | Current state |
| --- | --- | --- | --- | --- |
| ZipNN | 0.5.4, commit `6009704271394f0497dd6352f72aedc6b947bfc0`; sdist SHA-256 `abf45d...ce650` | All manifest files through a strict adapter | F32, F16, and BF16 are native; I8, FP8, and other dtypes are raw passthrough and fully charged | Adapter required; not run |
| DFloat11 | 0.5.0, commit `457733886ce6ebc6d8dda1621fad1ffa2661e028` | `medium-tinyllama-bf16` only | BF16 Llama-family, architecture-specific | Adapter and GPU gates required; not run |
| FPcompress | 1.0.3, commit `97f037249bd28682bfd83462ea1129e14df36d81` | `small-bert-f32`, `small-vit-f32` | Predominantly F32, with the container header included | Planned; not run |
| SPDP | 1.1, source SHA-256 `c0b6ca...63116` | Every manifest file | BERT and ViT are predominantly in-domain F32; every other row is labeled `out_of_domain_but_byte_exact` | Planned; not run |
| fpzip | 1.3.0, commit `4a539c06d98b1c029b08324a086d4b75689a2b72` | `small-bert-f32`, `small-vit-f32` | F32 dtype matches, but a whole checkpoint is an unstructured 1D stream and includes a container header | Planned; not run |
| AdaptiveFC/LC | LC 1.2, commit `0553cd874ceabd7189653dd5d28958c68256bf3b` | `small-bert-f32`, `small-vit-f32` | Paper-domain F32 protocol, with the container header included | Resource-gated; not run |
| `weight-compression` | commit `b9510aaac657b04e11d2f0d0d51a9b94af159590` | Audit only | BF16 research artifact | Explicitly not an experimental baseline |

The abbreviated hashes in the table are for readability. The JSON file stores
every full commit and SHA-256, along with canonical URLs. Its model-manifest
binding is SHA-256
`84202aee827632724d7441e9ae4725633778cdafacc4f31eaf4449a044b988ad`.

## Method-specific configurations

### ZipNN

The fixed profile is the pinned release's safetensors-oriented Huffman method
with one thread for the serial comparison. The upstream 0.5.4 safetensors
scripts load tensors and serialize a new file. Therefore, their built-in tensor
path is evidence about tensor losslessness, but it is not by itself evidence of
exact original-file restoration.

Before a formal run, a committed strict adapter must retain all original
non-tensor bytes, compress only F32, F16, and BF16 payloads, preserve raw any
unsupported or non-beneficial payload, and charge its complete wrapper. It must
decode in a fresh process and match the input file SHA-256. A mixed checkpoint
such as SmolLM W8A8 or Qwen FP8 remains eligible only under the label
`mixed_native_and_raw_passthrough`; its I8 or FP8 bytes must not be attributed to
native ZipNN compression. Matched-worker scaling, if run, changes only the
recorded `--threads` value.

Sources: [repository](https://github.com/zipnn/zipnn),
[paper](https://arxiv.org/abs/2411.05239).

### DFloat11

DFloat11 is conditionally evaluated only on the pinned TinyLlama BF16 file. We
will compress that input, not substitute a publisher-precompressed DFloat11
checkpoint. The profile fixes `compress_model`, `save_single_file=true`, and
`check_correctness=true`. Its architecture-specific `pattern_dict` must be
implemented, tested, and frozen before the first formal run.

The strict adapter must charge every generated compressed tensor, lookup table,
configuration field, and sidecar. It must also retain enough original container
information to materialize the exact input safetensors file. On-the-fly GPU
weight reconstruction or bit-identical model outputs are useful separate
systems measurements, but neither replaces archive decompression and whole-file
SHA verification.

The preregistered run follows the pinned README's CUDA 12 installation path.
The gate records the GPU, compute capability, driver, CUDA runtime, PyTorch, and CuPy versions;
verifies both `decode.ptx` and `decode.cu`; loads the PTX on the actual GPU; and
bit-checks a deterministic BF16 decode smoke test. A PTX-JIT or kernel-image
failure is `failed_gpu_gate`, not a missing row. Archive decompression and
on-the-fly GPU decode are timed separately, with CUDA synchronization around
the latter.

Sources: [repository](https://github.com/LeanModels/DFloat11),
[paper](https://arxiv.org/abs/2504.11651),
[OpenReview](https://openreview.net/forum?id=xdNAVP7TGy).

### FPcompress

The CPU profiles are fixed to the artifact's two single-precision pipelines:

* `SPspeed`: `DIFFMS_4 HCLOG_4`;
* `SPratio`: `DIFFMS_4 BIT_4 RZE_1`.

Both receive the complete BERT or ViT file directly. The pinned programs hold
file sizes in a signed `int`, so an input must be positive, smaller than
2,147,483,648 bytes, divisible by four, and manifest-verified. Both selected
files meet the preregistered numeric gates. Formal execution must still verify
them at runtime. `OMP_NUM_THREADS=1` fixes the serial comparison; other worker
counts belong to a separately labeled scaling run.

The pinned `compile.py` can continue after an individual compiler failure.
Consequently, a successful script exit is insufficient. The harness must
require all four CPU executables and hash them before use.

Sources: [repository](https://github.com/burtscher/FPcompress),
[paper and artifact appendix](https://userweb.cs.txstate.edu/~burtscher/papers/asplos25.pdf).

### SPDP

SPDP receives the complete file through stdin and produces the complete charged
stream through stdout. Levels 0, 5, and 9 are fixed as the speed, middle, and
ratio-oriented profiles. The build is `gcc -O3 SPDP_11.c -o spdp`, with compiler,
source, command, and executable hashes retained.

SPDP is tailored to 32-bit and 64-bit IEEE-754 data but can process arbitrary
bytes. The two F32 checkpoint rows are labeled predominantly in-domain because
their headers are also processed. Every non-F32 checkpoint is labeled
`out_of_domain_but_byte_exact`, reported separately, and never used as evidence
of a matched floating-point-domain advantage.

Source: [SPDP 1.1 page and code](https://userweb.cs.txstate.edu/~burtscher/research/SPDPcompressor/).

### fpzip

fpzip is limited to BERT and ViT. Each complete file is treated as one F32 1D
array at full 32-bit precision. The fixed element counts are 110,112,442 for
BERT and 86,573,463 for ViT, exactly the preregistered file sizes divided by
four. Compression uses `-t float -p 32 -1 <count>`; no reduced-precision mode is
allowed.

This is dtype-matched but not a spatial-data match. fpzip's documentation says
it targets correlated floating-point arrays and is unsuitable for unstructured
streams. Results therefore remain labeled
`dtype_matched_but_unstructured_1d_and_container_header`.

Sources: [repository](https://github.com/LLNL/fpzip),
[project page](https://computing.llnl.gov/projects/fpzip).

### AdaptiveFC through LC

The paper-inspired genetic search is fixed to five stages, 140 generations, a
population of 20, mutation rate 0.8, elitism cutoff 0.1, tournament selection,
masked crossover, no preprocessor, and the pinned LC 1.2 lossless component
list. It applies only to the two F32 small-model files. SmolLM and every GB-scale
non-F32 entry are `not_applicable` due to the preregistered dtype/domain scope,
not `skipped_resource`.

The first gate runs seed 0 on each eligible file. Only if both pilots pass the
whole-file check and the recorded time, memory, and disk budget permits the
complete expansion do we run seeds 0 through 8. The AdaptiveFC paper describes
nine runs but does not publish those nine seed values. Thus 0 through 8 are new
Brevis preregistered seeds, and the experiment is
`reproduction_inspired_not_exact_seed_replication`. A partial expansion retains
all completed rows, marks remaining rows `skipped_resource`, and is never called
a nine-seed reproduction.

Search is not free. Raw records retain framework build, GA search, unique
pipelines evaluated, selected pipeline, code generation, selected-codec build,
archive compression, and archive decompression. End-to-end compression includes
search, input-specific generation/build, and one compression. Steady-state
encoding and decoding are separate metrics. Every completed seed is reported;
the primary result does not select the best seed. A best-of-nine analysis is
allowed only as secondary evidence and must charge all nine searches.

The strict adapter must put the selected pipeline and framing in the archive. If
an input-specific generated executable is required by the decoder, it must also
be charged unless a pinned input-independent decoder can consume the archived
pipeline description.

Sources: [LC repository](https://github.com/burtscher/LC-framework),
[AdaptiveFC paper](https://userweb.cs.txstate.edu/~burtscher/papers/essa24.pdf).

## Why `weight-compression` is not a baseline

The pinned repository is useful related-work and design evidence, but it does
not meet this protocol's executable whole-file boundary. Its K15 charged-format
number is an estimate that was not independently serialized and decoded at GLM
scale. Its byte-split path verifies tensor round trips but does not expose the
matched timed complete-file archive/decompress interface. Its dense 12-bit GPU
prototype does not include a fused sparse exact correction and is not a complete
exact serving path. Reporting any of these as an observed archive baseline
would mix estimates or partial prototypes with measured exact archives.

Source: [repository](https://github.com/brianbell-x/weight-compression).

## Repetitions, failures, and reporting

After all source, build, adapter, domain, and hardware gates pass, small and
medium files use one unreported warm-up plus six measured repetitions. Large
files and the GLM positional sample use one warm-up plus four measured
repetitions. The method order uses seeded forward/reverse pairs with seed 2701.
The primary CPU comparison fixes one worker. Scaling runs are separate.

AdaptiveFC is the exception: stochastic variability is measured with its fixed
seed protocol. The seed-0 selected pipeline additionally receives the normal
small-file steady-state timing schedule; every completed seed gets at least one
full correctness round trip.

Every preregistered method/profile/file combination receives a row. Valid
runtime statuses are `success`, the phase-specific `failed_*` values, `timeout`,
`oom`, `skipped_resource`, and `not_applicable`, as enumerated in the JSON.
Failures retain the phase, exact command and environment, source and executable
hashes available at failure, resource limits, exit code or signal, diagnostic
hashes, and reason. We do not omit a method on a model because it expands,
crashes, or performs poorly.

Timing records keep source verification, build, adapter preparation, search,
code generation, compression, decompression, and post-timing verification
separate. Throughput uses exact raw bytes divided by the matching operation's
wall time. Report every observation plus mean, sample standard deviation,
median, range, and coefficient of variation. The cache condition is only
best-effort buffered-cache conditioning.

## Validation

The protocol can be checked without installing a baseline:

```bash
python3 -m json.tool eval/specialized-baselines.json >/dev/null
python3 -m unittest eval.test_specialized_baselines -v
```

These commands validate protocol structure and pinned identities. They do not
produce compression results.
