# Evaluation status

Except for the BRTA v1 runner named below, the older scripts, protocols, and
result files in this directory describe the superseded block/template
implementation and are retained as historical artifacts. They invoke removed
CLI modes and report per-block concepts that do not exist in the paper-aligned
`BRTA` version 1 implementation.

Do not use these artifacts as evidence for the current compressor. In
particular, legacy `fixed`/`uniform`/`phog` comparisons, block reports, schema
5/6 archive sizes, and recorded program inventories are not comparable to the
whole-tensor semantic DSL.

A replacement evaluation must bind results to:

- the exact source revision and Zig toolchain;
- full input paths, sizes, and SHA-256 digests;
- the canonical `BRTA` archive digest and size;
- every synthesis and grammar limit;
- the optional canonical `BRGP` prior digest;
- complete byte-for-byte decompression verification;
- separately measured calibration, compression, and decompression resources.

`brta_benchmarking.py` is the replacement BRTA v1 engineering runner. It saves
the complete `brevis config` JSON, Git and binary identity, input and archive
hashes, exact commands, worker counts, every wall-time observation, and a full
post-timing byte comparison. Results are deliberately machine-labelled
`run_class=engineering` and `paper_metrics_eligible=false`, including results
from a clean tree. Its binary SHA-256 binds the executed artifact, but the
runner does not rebuild that binary or infer its Zig compiler and optimization
flags. A formal campaign must bind those separately.

For example:

```bash
python eval/brta_benchmarking.py model.safetensors \
  --output result.json \
  --workers 1 --workers 8 \
  --warmups 1 --repetitions 3 \
  --max-expansions 0 \
  --expected-input-size <bytes> \
  --expected-input-sha256 <sha256>
```

Use `--seed-float-fields 0` for an explicit seed ablation. The implementation
also skips this seed whenever `--max-expansions 0`, even if the effective
configuration retains its default value `true`.

Pass `--prior model.brgp` to bind an existing PHOG prior, or
`--calibrate-tensors N` to time and hash one input-local calibration before the
repeated runs. `--generic zstd/default` and other registry identifiers from
`benchmarking.py` optionally embed a separate general-compressor track.
`zstd/ultra` pins `--ultra -22` as an explicitly extreme ratio ceiling.

`brta_program_inventory.py` streams one BRTA v1 archive and reports root
operations, AST depth, semantic-operation depth, operation sequences, and
program bytes per tensor. It does not execute programs or validate tensor
checksums. Run `brevis verify` first, then use the inventory only as structural
evidence:

```bash
./zig-out/bin/brevis verify model.brta model.safetensors
python eval/brta_program_inventory.py model.brta > inventory.json
```

`benchmarking.py` is format-independent and can still run pilot comparisons
against generic file compressors. Its legacy result files do not validate
BRTA v1. The generic and BRTA schedules are separate, and neither script by
itself certifies a paper result.

Generic-process wall time includes a successful file `fsync` of both the
archive and restored output. The JSON records the durability status and fails
the run if synchronization fails. BRTA file commands perform their own
synchronized atomic replacement inside the measured process.

This timing contract begins with generic result schema version 3. BRTA process
records mark wrapper-level durability as not requested because the child
command performs the sync; the BRTA result's `configuration.io_policy` records
that distinction explicitly.

Generic runs stage source and archive inputs with the current Python
interpreter and explicit 1 MiB reads and writes. This avoids clone and
sparse-seek shortcuts and keeps the harness portable across GNU and BSD
systems. Staging remains outside the timed compressor process and is recorded
in the JSON.

The current dirty-tree engineering pilot is recorded in
`BRTA_V1_PILOT_2026-07-27.md`. The raw performance A/B records and the rejected
contended diagnostic are under
`results/engineering/performance-audit-2026-07-27/`. Their numbers are useful
for implementation decisions but are not protocol-complete paper evidence.
Formal paper tables still require a clean, frozen campaign with its declared
repetition and resource controls.
