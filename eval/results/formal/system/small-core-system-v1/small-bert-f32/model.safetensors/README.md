# BERT F32 core-system result

This directory preserves the first complete repeated system result for the
frozen `small-bert-f32` shard. The raw result passed all schema-2 formal gates:
the manifest and input hashes match, the repository was clean at commit
`5aa37351d582e7aa2034a85844083358069097c2`, the harness built and retained one
ReleaseFast binary, all 28 archive round trips were bit exact, six measured
archives per configuration were byte-identical, and all schema-4 archive-size
and program-bytecode checks passed.

`result.json` is the immutable raw checkpoint. `summary.json` retains every
measured timing trial and links every aggregate to an exact JSON pointer in the
raw result. The DSL directory contains the canonical analysis manifest and
report records. Its larger tensor, block, and node JSONL files are stored as
Zstandard frames. `dsl/manifest.json` gives their uncompressed byte counts and
SHA-256 digests. `dsl/aggregate.json.zst` is a derived, fail-closed summary of
program length and depth, serialized bytecode, structures, operator and
terminal combinations, tensor strata, and search-counter relationships. It
does not treat the four configurations as independent models.

This run used the direct formal harness before the campaign executor acquired
its machine-level advisory lock. No other model benchmark was run concurrently,
but lightweight orchestration, unit tests, and a LaTeX compile occurred in the
same container. Effectiveness, reconstruction, and DSL evidence are eligible.
Timing trials are retained with their full variance and environment records,
but headline cross-method speed claims require confirmation under the exclusive
campaign runner.

To revalidate the raw result and regenerate derived files:

```bash
PYTHONPATH=eval python3 eval/summarize_system_benchmarks.py \
  eval/results/formal/system/small-core-system-v1/small-bert-f32/model.safetensors/result.json

python3 eval/analyze_generated_dsl.py \
  eval/results/formal/system/small-core-system-v1/small-bert-f32/model.safetensors/result.json \
  --output-dir /tmp/brevis-bert-dsl-check

python3 eval/summarize_generated_dsl.py \
  eval/results/formal/system/small-core-system-v1/small-bert-f32/model.safetensors \
  --output /tmp/brevis-bert-dsl-aggregate.json
```

`artifact-manifest.json` records exact stored and uncompressed hashes, the
compression command, and the allowed evidence uses.
