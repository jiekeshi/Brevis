# Tiered Model Evaluation Protocol

Status: preregistered evaluation design, not an experimental result. The model
identities and the GLM-5.2 shard rule below were fixed on 2026-07-21 before
inspecting Brevis compression results for the newly added models.

Protocol amendment, 2026-07-21, before any formal tiered timing result was
inspected: measured repetition counts are even so that each seeded method order
can be paired with its reverse. Small and medium variants use six measured
runs; large variants and the GLM-5.2 sample use four. The generic-codec harness
also binds every model label to the matching SHA-verified manifest entry. This
amendment follows harness safety review and does not depend on compression
outcomes.

`models-tiered.json` is the machine-readable source of truth for this protocol.
It remains compatible with `run_eval.py`: the evaluator consumes `tag`, `repo`,
`revision`, `files`, and `note`, and ignores the additional audit metadata. The
older `models.json` and `models-large.json` remain useful minimal manifests; the
tiered manifest reuses all four of their model revisions and files.

## 1. Evaluation question and lossless boundary

Brevis is evaluated on the exact bytes of public safetensors weight files. A
successful lossless run must reproduce each input file byte for byte, including
its header, tensor order, tensor values, and metadata. For an input of `R` bytes
and an archive of `C` bytes, report:

* compression ratio `R / C`;
* storage saving `1 - C / R`; and
* both exact byte counts, including archive overhead.

FP8 and INT8 entries are already quantized by their publishers. Brevis does not
create those representations. “Lossless” means lossless relative to the stored
FP8 or INT8 input, not equivalence to an unquantized checkpoint. Likewise, the
SDXL image-generation pipeline may contain a lossy autoencoder, but compression
of its stored weight files must still be byte exact.

No accuracy or task-quality result is needed to establish byte-exact weight
compression. If a later experiment compares a quantized variant with its BF16
counterpart, it must describe the different input semantics and must not count
upstream quantization savings as Brevis savings.

## 2. Fixed scale tiers

Tiers use the serialized size of the canonical weight variant, because storage
is the resource under study. The cutoffs were chosen before measuring Brevis and
do not depend on compressibility:

| Tier | Canonical safetensors weight bytes |
| --- | ---: |
| small | less than 1 GiB |
| medium | 1 GiB to less than 10 GiB |
| large | 10 GiB to less than 100 GiB |
| ultra | at least 100 GiB |

Parameter counts are reported separately. They do not define the tier because
the number of bytes per parameter changes with dtype, quantization metadata, and
tensor duplication. `canonical_variant_bytes` describes the complete selected
weight variant. `selected_bytes` describes bytes that will actually be run. They
differ only for the explicitly sampled GLM-5.2 entry.

The preregistered matrix is:

| Tag | Tier and scope | Architecture / purpose | Stored dtype(s) | Canonical GB | Selected GB |
| --- | --- | --- | --- | ---: | ---: |
| `small-bert-f32` | small, complete | BERT encoder / masked language modeling | F32 | 0.440 | 0.440 |
| `small-vit-f32` | small, complete | ViT encoder / image classification | F32 | 0.346 | 0.346 |
| `small-smollm-w8a8` | small, complete | quantized dense decoder / text generation | I8, BF16 | 0.220 | 0.220 |
| `medium-tinyllama-bf16` | medium, complete | dense decoder / text generation | BF16 | 2.200 | 2.200 |
| `medium-whisper-v3-f16` | medium, complete | speech encoder-decoder / ASR | F16 | 3.087 | 3.087 |
| `medium-sdxl-base-f16` | medium, complete variant | latent diffusion / text-to-image | F16 | 6.938 | 6.938 |
| `large-qwen3-8b-bf16` | large, complete | dense decoder / text generation | BF16 | 16.382 | 16.382 |
| `large-qwen3-30b-a3b-bf16` | large, complete | MoE decoder / text generation | BF16 | 61.067 | 61.067 |
| `large-qwen3-30b-a3b-fp8` | large, complete | quantized MoE decoder / text generation | F8_E4M3, F32 | 32.449 | 32.449 |
| `ultra-glm52-bf16-positional-3shard` | ultra, three-shard sample | MoE decoder / text generation | BF16 in selected shards | 1,506.667 | 10.997 |

GB values in this table are decimal and rounded only for display. Analyses must
use the exact integer fields in the manifest and the observed local file sizes.

### Selection rationale and fairness

The matrix covers text and vision encoders, dense and MoE decoders, a speech
encoder-decoder, and a diffusion pipeline. It covers F32, BF16, F16, FP8, INT8,
and mixed-dtype checkpoints across four storage tiers. BERT, TinyLlama, the W8A8
SmolLM, and Qwen3-8B are retained from the earlier manifests rather than replaced
after seeing their results. The additional models fill missing cells in the
architecture, purpose, dtype, and scale matrix.

Selection is not based on measured compression ratio. Eligibility requires a
publicly readable Hugging Face repository, safetensors weights, a model card or
config that identifies the architecture and purpose, and an immutable commit
revision. A failed or unfavorable model remains in the accounting. It may not be
replaced by a similar model without a manifest revision, a dated reason, and a
separate result series.

This is a deliberately heterogeneous benchmark, not a random sample of all
neural networks. Report per-model results before any aggregate. Report both:

* a byte-weighted corpus aggregate, computed from sums of raw and compressed
  bytes; and
* an unweighted distribution across complete variants.

Do not merge the GLM-5.2 sample into either whole-model aggregate. The Qwen3 MoE
pair controls architecture and nominal parameter scale better than the rest of
the matrix, but its BF16 and publisher-produced FP8 variants still store
different tensors and semantics. Other comparisons may confound architecture,
task, scale, and dtype, so they are descriptive rather than causal.

## 3. Repository and dtype verification

Every `revision` is a 40-hex Git commit returned by the official Hugging Face Hub
API. Each `files` entry records the filename, remote byte size, and LFS/Xet
SHA-256 reported by the API at that revision. Architecture, task, and canonical
parameter dtype counts come from the same revision's API response, config, and
model card. The revision-specific API and model-card URLs are stored under each
entry's `verification` object.

The SDXL F16 count was obtained from safetensors headers through the official
`huggingface_hub.parse_safetensors_file_metadata` range-request helper, version
1.18.0. Its four selected files are the two text encoders, UNet, and VAE named by
the pinned `model_index.json`; the pinned model card explicitly documents the
`variant="fp16"` and `use_safetensors=True` loading path. Duplicate F32 files, the
example LoRA, `vae_1_0`, and the separately hosted refiner are not part of this
base-pipeline variant.

Whisper's `model.safetensors` is the complete F16 variant. The two files named
`model.fp32-*.safetensors` in the same repository are an alternate FP32 variant,
not extra shards of the selected F16 input.

Before a formal run:

1. Parse the manifest as JSON and reject duplicate tags, duplicate filenames
   within a tag, non-40-hex revisions, nonpositive sizes, or invalid SHA-256
   strings.
2. Query each stored revision-specific API URL. Require the returned `sha` to
   equal `revision`, and require every selected filename, size, and remote SHA-256
   to match the manifest. This is a metadata request and must not download model
   weights.
3. After a file is fetched, require its local size and SHA-256 to match the
   corresponding manifest entry before running any compressor.
4. Require `run_eval.py`'s per-shard `input_sha256` to equal that same digest.
   A mismatch is a failed integrity gate, not a new result.
5. Inspect actual safetensors headers and record per-tensor dtypes. If they
   disagree with the preregistered dtype inventory, retain the input and record
   the discrepancy; do not relabel it silently.

Never move a tag from its pinned revision to a newer `main`. A publisher update
requires a new revision and result namespace so that old results remain
reproducible.

## 4. Complete variants, components, and shard samples

`complete_weight_variant` means that all safetensors files needed for the named
stored variant are included. It does not mean every alternate precision,
training artifact, adapter, or duplicate representation in the repository is
included. Results for complete sharded variants are aggregated by summing raw
bytes, compressed bytes, and elapsed time across all listed shards; the ratio is
then computed from the byte sums, not by averaging shard ratios.

SDXL is a complete F16 base-pipeline weight variant with four components. Results
must retain component-level rows as well as their byte-weighted sum. Do not call
it the base-plus-refiner ensemble because the refiner is not selected.

### Preregistered GLM-5.2 rule

The pinned BF16 GLM-5.2 variant contains 282 safetensors shards totaling
1,506,667,387,408 remote file bytes. The pinned index reports
1,506,659,919,872 tensor-payload bytes; the difference is container headers. The
current filesystem snapshot on 2026-07-21 has 2,147,483,648,000 bytes total and
`workspace_is_volume=false`. A full GLM input alone consumes about 70% of that
filesystem, before the rest of the corpus, partial downloads, archives, restored
verification files, repetitions, and failure-recovery headroom. The evaluation
policy reserves at least 30% of filesystem capacity after caching inputs.
Therefore a complete BF16 GLM-5.2 run is not suitable on this instance. This is
an operational exclusion, not a claim that the raw files are mathematically too
large to fit once.

The fixed sample selects file indices:

```text
1, floor((1 + 282) / 2) = 141, 282
```

Selection uses only shard count and immutable lexicographic file numbering; it
does not use compression output. The selected filenames, hashes, and sizes are
already materialized in `models-tiered.json`. Do not substitute another shard if
one fails or compresses poorly.

This positional sample is useful for exercising ultra-model artifacts, but it is
not statistically representative of all 282 shards. Inspection of the pinned
weight index, performed only to document this limitation, shows that shard 1
contains embeddings, `lm_head`, and early-layer tensors; shard 141 contains many
layer-45 expert tensors; and the much smaller shard 282 contains five layer-9 or
final-normalization tensors. File numbering therefore does not uniformly sample
layers, tensor roles, shapes, or byte mass.

The only valid aggregate is a byte-weighted aggregate over these three named
files, labeled “GLM-5.2 BF16 positional three-shard sample.” Never report it as a
full GLM-5.2 ratio, whole-model saving, runtime, or throughput estimate. Do not
multiply its saving by 282, and do not construct a whole-model confidence
interval from these three nonrandom shards.

## 5. Staged execution and resource gates

Run one tag at a time and write to a tag- and repetition-specific result path.
`run_eval.py` overwrites its result document, so reusing one result path across
separate `--tag` invocations would lose prior rows.

| Stage | Tags | Gate before advancing |
| --- | --- | --- |
| 0 | metadata only | API identity, filename, size, SHA, disk, and supported-dtype checks pass |
| 1 | all `small-*` | every method either passes byte-exact verification or has a recorded failure |
| 2 | all `medium-*` | sufficient disk and wall-time budget remain; component rows retained |
| 3 | `large-qwen3-8b-bf16` | complete five-shard result and repeatability checks pass |
| 4 | both `large-qwen3-30b-a3b-*` | both complete variants are run under matching resource limits |
| 5 | `ultra-glm52-bf16-positional-3shard` | sample-only label and no-extrapolation guard are present in analysis output |

A typical isolated invocation is:

```bash
python eval/run_eval.py \
  --models eval/models-tiered.json \
  --tag small-bert-f32 \
  --results eval/raw/<run-id>/small-bert-f32/rep-01.json \
  --jobs <fixed-jobs> \
  --tensors <fixed-calibration-tensors>
```

The literal run ID, job count, calibration count, and command must be recorded,
not copied from this placeholder. Long experiments start only after the exact
manifest, command configuration, and code revision are committed. Formal runs
use a clean tree, declared input/manifest integrity, and a ReleaseFast build
performed by the harness. Disabling any gate machine-labels the result
`run_class=pilot`, records the exact ineligibility reasons, and excludes it from
paper numbers. Preserve the diff separately for a dirty pilot.

Before each stage, record available bytes, expected input bytes, largest expected
concurrent temporary footprint, and the stop threshold. Do not begin a download
that violates the 30% free-space reserve. A resource-gated skip is a recorded
`skipped_resource` result, not a missing row.

## 6. Effectiveness, efficiency, and repetition

### Effectiveness

For every selected shard and every method, retain raw bytes, compressed bytes,
ratio, storage saving, and byte-exact verification status. A model-level result
must be derivable from its shard rows. A corpus-level result must be derivable
from complete model rows. Report archive overhead and never discard expansion:
if an archive is larger than raw, its negative storage saving remains in the
table.

### Efficiency

Separate and label:

* prior calibration time;
* program-search and compression time;
* fixed-program compression time;
* decompression time at one worker and at the configured worker count;
* generic or specialized baseline compression and decompression time; and
* peak resident memory for each measured operation.

Throughput is raw input bytes divided by the corresponding operation's elapsed
time. Downloads, hash verification, and tensor-statistics scans are outside codec
time and must be timed separately if reported. Do not combine calibration and
compression unless the metric is explicitly labeled end to end.

The legacy evaluator gives one timing observation per invocation. One
observation is adequate for deterministic size validation but not for a final
speed claim. Use the following fixed timing schedule:

* small and medium complete variants: one unreported warm-up plus six measured
  repetitions;
* large complete variants: one unreported warm-up plus four measured
  repetitions; and
* the GLM-5.2 three-shard sample: one unreported warm-up plus four measured
  repetitions.

If resources prevent the specified repetitions, publish the completed raw runs,
mark the timing cell incomplete, and do not replace the missing variance with an
estimate. For each timing metric report all observations, arithmetic mean,
sample standard deviation with `ddof=1`, median, minimum, maximum, and coefficient
of variation. Size results should be identical across repetitions; any mismatch
is a correctness failure.

This container cannot reliably flush host page cache. Hashing, independent copy
staging, and warm-ups condition buffered I/O, but they do not guarantee that
every page remains resident. Record this best-effort cache condition and do not
label the measurements cold I/O throughput or guaranteed warm-cache throughput.

For parallel scaling, preregister worker counts `1, 2, 4, 8, ...` up to the
smaller of the available physical cores and the evaluator's supported maximum.
Use the same input, search budget, and method configuration at every count, with
three measured repetitions per point. Report speedup relative to one worker,
parallel efficiency, peak RSS, and the exact worker set. Do not mix results from
different machines into one scaling curve.

## 7. Search modes, ablations, and baseline fairness

At minimum, run Brevis `raw`, `fixed`, `uniform`, and `phog` configurations on the
same bytes. `fixed` uses the fixed DSL template with the normal raw fallback;
`uniform` and `phog` use the same search implementation and budget, differing
only in whether the learned prior guides expansion. Search depth, node budget,
candidate collection and reranking settings, calibration tensor count, terminal
codec set, reversible-operator set, worker count, and random seed, if any, belong
in the raw provenance.

The system harness names its internal raw configuration `raw-terminal`: it
disables every non-raw production exposed by the running binary and verifies
the effective mask. This is a framed Brevis archive, not the generic track's
unframed `raw/copy` entry.

For each ablation, change exactly the named factor where the implementation
allows it. If disabling an operator or codec changes the valid-program space,
record that fact. If the current CLI cannot express a requested ablation, label
it `not_implemented`; do not infer its contribution from another comparison.

Every external compressor receives the identical complete safetensors file,
including its header. Give all methods the same CPU-worker and memory policy as
far as their interfaces allow, and record exceptions. General-purpose tools must
have preregistered default, speed-oriented, and ratio-oriented settings where
available. Specialized weight compressors are experimental baselines only if
they accept the stored dtype and provide byte-exact reconstruction under a
comparable scope. A method that is unavailable, crashes, exceeds a resource
limit, or cannot preserve bytes receives an explicit failure row. Never omit it
only from models where it performs poorly.

Tool names and profile labels are insufficient provenance. Save executable
version, full arguments, environment variables that alter behavior, worker
count, input digest, output digest, and verification command. Results from a
different compression level are a different configuration, not additional
timing repetitions of the same configuration.

`benchmarking.py` implements the serial generic-codec track with raw copy and
three pinned profiles for gzip, bzip2, xz, Zstandard, LZ4, and Brotli. It uses
seeded forward/reverse order pairs, independent staging inodes, process-group
timeouts, direct-child RSS, per-run archive hashes, and complete post-timing
round-trip verification. Measured archive sizes and hashes must agree across
repetitions. This serial registry is not a substitute for the separate
matched-worker scaling track.

`brevis_benchmarking.py` implements repeated Brevis runs with independent
calibration, a canonical measured prior, paired method orders, process-level
RSS, complete archives, and post-timing byte verification. It executes every
timed archive pipeline before any diagnostic replay. Thus, bench planning time
is not an exact decomposition of compression wall time, and method-specific
diagnostics cannot precondition timed archive tasks. The initial checkpoint
contains the complete calibration/archive/diagnostic schedule and records task
state transitions. Work and checkpoint filesystems receive separate capacity
checks, including the old-plus-new peak of atomic checkpoint replacement.

Schema-4 diagnostic detail is deduplicated within each Brevis configuration.
The first complete successful warmup is canonical; if none exists, the first
complete successful measured repetition is canonical. Its `tensors` and
`blocks` arrays remain inline. Matching repetitions remove only those two arrays
and retain the complete remaining top-level report, internal planning/encoding
times, a semantic SHA-256, and a reference that resolves inside the same
checkpoint. The semantic fingerprint excludes `/planning_wall_ms`,
`/encoding_wall_ms`, `/input`, and `/prior/path`, while preserving every other
known or future field. Reports that fail validation are retained without being
eligible as canonical. A semantic mismatch retains its complete arrays and
fails closed. DSL analysis must resolve the one canonical detail once per
configuration; diagnostic repetitions are technical replicates, not independent
tensors or models.

`analyze_generated_dsl.py` enforces this rule for system schema 2 and bench
schema 4. It revalidates every compact reference and full-file/archive evidence,
then emits one semantic report per configuration plus canonical tensor, block,
and ordered program-node records. Old or future schemas, incomplete
configurations, inconsistent technical repetitions, and pilots without an
explicit opt-in fail closed. Even with the opt-in, pilot records remain
ineligible for paper metrics. Tensor-name roles are assigned independently on
orthogonal axes by the frozen SHA-bound rule file; these labels are reported as
heuristics, not model-architecture annotations.

## 8. Per-tensor and generated-DSL records

Console summaries are not raw data. For every tensor, retain a machine-readable
record keyed by:

```text
(manifest SHA-256, model tag, repository revision, shard filename,
 input SHA-256, tensor name)
```

The record must include dtype, shape, element and byte counts, tensor byte range,
basic statistics used by search, selected program bytecode or a stable textual
form, program length and depth, operator sequence/tree, terminal codec,
candidate count, nodes expanded, reranking measurements, calibration/search/
encode time, compressed bytes, and fallback reason. Preserve unsuccessful
searches and raw fallbacks.

Generated-DSL analysis should report program length, structure, operator
combinations, dtype and tensor-role preferences, and their association with
compression saving and search cost. Report byte-weighted and tensor-weighted
views because many small normalization tensors otherwise dominate tensor
counts, while embeddings and expert matrices dominate bytes. Associations are
descriptive unless a statistical model and its assumptions were preregistered.
Do not treat tensors from the same model as independent model-level replicates.

Bench schema 4 provides tensor offsets, structured programs and parameters,
search counters, raw-root classification, per-block payload and framing bytes,
and exact projected archive accounting. Per-block and ordered aggregate
bytecode digests bind the diagnostic programs to an independent scan of the
actual archive frames. The harness accepts this schema exactly for formal DSL
evidence and preserves the full report. Tensor-level search time, per-terminal
payload attribution in multi-leaf programs, and tensor entropy/zero/delta
statistics are not present; use search counters as a declared cost proxy and
do not infer the missing quantities.
`tensor_stats.py` remains a sampled stdout summary, and the current tools still
lack some requested per-tensor search statistics and per-tensor timings. Claims
requiring those fields remain incomplete.

## 9. Environment, failures, and limitations

Each run directory must contain or point to:

* the manifest bytes and SHA-256;
* Brevis Git commit, dirty status, binary SHA-256, evaluator SHA-256, and build
  mode;
* complete commands and stdout/stderr logs;
* CPU model, physical/logical cores, RAM, filesystem type and free bytes, GPU
  model if used, kernel, OS, Zig and Python versions;
* compressor versions and arguments;
* search and ablation configuration;
* repetition number, start/end UTC timestamps, and exit status; and
* raw per-shard, per-tensor, timing, memory, and verification records.

Record a failed attempt with at least `tag`, `revision`, `file`, `input_sha256`
when known, method, configuration hash, stage, status, exit code or signal,
stderr-log path, elapsed time, peak RSS when available, and a concise reason. Use
distinguishable statuses such as `failed_integrity`, `failed_codec`,
`failed_verify`, `unsupported_dtype`, `timeout`, `oom`, and
`skipped_resource`. Keep partial artifacts only when safe and useful, and label
them non-results.

Known design limitations that must accompany the eventual evaluation include:

* only publicly readable safetensors weights are eligible; optimizer states,
  proprietary checkpoints, and other containers are out of scope;
* the model matrix is purposive rather than randomly sampled;
* architecture, task, scale, dtype, and training history are confounded outside
  the Qwen3 MoE variant pair;
* publisher-created INT8 and FP8 variants do not measure lossless conversion
  from higher precision;
* the GLM-5.2 result is a structurally biased three-shard sample; and
* timing uses best-effort buffered I/O and is machine-specific; neither
  cold-cache nor guaranteed warm-cache residency is established.

## 10. Result-to-paper traceability and change control

Use immutable raw result paths, for example:

```text
eval/raw/<UTC-date>/<brevis-commit>/<tag>/<configuration-hash>/rep-<NN>.json
```

Derived tables must record the analysis script commit, input result paths and
hashes, filters, aggregation rule, and output hash. Every quantitative paper
claim should map to a row or figure datum, which maps to exact raw result files,
which map through `input_sha256` to a manifest file and pinned Hugging Face
revision. Keep failures and exclusions in the same accounting table.

Changes to the matrix, tier boundaries, GLM sample, repetition count, baseline
settings, or aggregation rules require a dated protocol amendment before the
affected new results are inspected. Preserve this version and its prior results.
Do not rewrite a preregistered rule in place to improve an observed outcome.

As of this document's date, `models-tiered.json` records planned inputs only. It
does not assert that the weights were downloaded, that any new model completed,
or that Brevis achieved a particular compression ratio or speed.
