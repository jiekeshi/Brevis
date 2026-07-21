# BERT F32 exclusive core-system result

This directory preserves the hash-addressed `small-bert-f32` task executed by
the formal campaign runner at commit
`411b13393d647c7cb189877764ee5de1e63d7fa0`. The runner acquired the shared
advisory lock at `/tmp/brevis-machine-benchmark.lock` before any calibration or
benchmark process. No other Brevis or generic-codec benchmark, build, test, or
paper compilation was run in this container until the task completed. The lock
cannot exclude uncooperative external processes, as documented in
`campaign-plan.json`.

The input is the manifest-bound `google-bert/bert-base-uncased`
`model.safetensors` object at revision
`86b5e0934494bd15c9632b12f734a8a67f723594`: 440,449,768 bytes with SHA-256
`68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3`.
The task used one worker, one warmup and six measured repetitions, schedule seed
2701, a 3,600 s process timeout, and 200 tensors for the separately timed PHOG
prior calibration.

All 56 scheduled configuration steps completed: 28 timed archive pipelines and
28 later diagnostic replays. Every archive decoded bit exactly, every archive
remained unchanged during decoding, and the six measured archives within each
configuration had identical sizes and SHA-256 digests. All diagnostic archive
size projections and realized-program bytecode sequences matched the actual
archives. The seven calibration priors were identical. All 63 run-local
directories and the enclosing artifact directory were removed successfully.
The result is classified `formal_eligible: true`.

The exact measured means and sample standard deviations are:

| configuration | archive bytes | saving | compress (s) | decompress (s) | peak compression RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw-terminal | 440,530,878 | -0.0184% | 0.893 ± 0.058 | 0.404 ± 0.013 | 428.1 |
| fixed | 366,004,621 | 16.9021% | 2.320 ± 0.047 | 1.973 ± 0.044 | 436.2 |
| uniform | 366,011,656 | 16.9005% | 57.332 ± 0.929 | 2.315 ± 0.051 | 662.2 |
| PHOG | 366,006,757 | 16.9016% | 50.160 ± 1.107 | 2.157 ± 0.043 | 727.0 |

PHOG calibration took 37.419 ± 0.577 s and is not included in compression
time. Compression time is the complete CLI archive pipeline. Planning and
encoding fields in the diagnostic replay are retained separately and are not
treated as a decomposition of that pipeline.

`result.json` is the immutable raw schema-2 checkpoint. `summary.json` is the
strict trace-preserving system summary. `campaign-plan.json` binds the exact
task semantics and commands; `campaign-receipt.json` is the JSON object emitted
at terminal success. The DSL directory contains the single canonical semantic
report per configuration. Large JSONL files and the aggregate are stored as
Zstandard frames; `artifact-manifest.json` records stored and logical hashes.
The new DSL records reproduce the earlier BERT program structures exactly.

To revalidate the raw result and regenerate derived output without replacing
the preserved files:

```bash
PYTHONPATH=eval python3 eval/summarize_system_benchmarks.py \
  eval/results/formal/system/small-core-system-v1/small-bert-f32/model.safetensors/exclusive-86cdd920/result.json \
  --output /tmp/bert-exclusive-system-summary.json

python3 eval/analyze_generated_dsl.py \
  eval/results/formal/system/small-core-system-v1/small-bert-f32/model.safetensors/exclusive-86cdd920/result.json \
  --output-dir /tmp/bert-exclusive-dsl

python3 eval/summarize_generated_dsl.py \
  eval/results/formal/system/small-core-system-v1/small-bert-f32/model.safetensors/exclusive-86cdd920 \
  --output /tmp/bert-exclusive-dsl-aggregate.json
```
