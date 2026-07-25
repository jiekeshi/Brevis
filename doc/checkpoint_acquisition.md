# Checkpoint acquisition

Nothing is vendored. `eval/cache/` is Git-ignored and starts empty; as of this
writing no checkpoint has been downloaded.

## Source of truth

[`eval/models-tiered.json`](../eval/models-tiered.json) is the machine-readable
manifest and is **preregistered** — see `eval/PROTOCOL.md`. For every model it
pins `repo`, `revision` (an immutable commit, never a branch), and for every file
its exact `bytes` and `sha256`. Do not edit it to make an acquisition step
easier; that is a protocol amendment.

The older `eval/models.json` and `eval/models-large.json` are minimal manifests
kept for smoke tests and historical reruns.

## Preregistered matrix

`execution_stage` is the intended download and run order; stage 1 is cheapest.

| Stage | Tag | Repo | Files | Bytes | dtype |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | `small-bert-f32` | `google-bert/bert-base-uncased` | 1 | 0.440 GB | F32 |
| 1 | `small-vit-f32` | `google/vit-base-patch16-224` | 1 | 0.346 GB | F32 |
| 1 | `small-smollm-w8a8` | `RedHatAI/SmolLM-135M-Instruct-quantized.w8a8` | 1 | 0.220 GB | I8 + BF16 |
| 2 | `medium-tinyllama-bf16` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1 | 2.200 GB | BF16 |
| 2 | `medium-whisper-v3-f16` | `openai/whisper-large-v3` | 1 | 3.087 GB | F16 |
| 2 | `medium-sdxl-base-f16` | `stabilityai/stable-diffusion-xl-base-1.0` | 4 | 6.938 GB | F16 |
| 3 | `large-qwen3-8b-bf16` | `Qwen/Qwen3-8B-Base` | 5 | 16.382 GB | BF16 |
| 4 | `large-qwen3-30b-a3b-bf16` | `Qwen/Qwen3-30B-A3B-Base` | 16 | 61.067 GB | BF16 |
| 4 | `large-qwen3-30b-a3b-fp8` | `Qwen/Qwen3-30B-A3B-FP8` | 7 | 32.449 GB | F8_E4M3 + F32 |
| 5 | `ultra-glm52-bf16-positional-3shard` | `zai-org/GLM-5.2` | 3 | 10.997 GB | BF16 |

**Total to download: 134.13 GB (≈124.9 GiB) across 40 files.**

Two entries need care:

- **SDXL** is a four-component pipeline (two text encoders, UNet, VAE) at the
  `fp16` variant paths named by the pinned `model_index.json`. The duplicate F32
  files, the LoRA example, the alternate VAE, and the refiner repository are
  deliberately excluded.
- **GLM-5.2** is a **positional three-shard sample**, not the model. The full
  BF16 variant is 1,506.667 GB and is excluded on storage grounds. The rule —
  shards 1, `floor((1+282)/2)=141`, and 282 — was fixed before any compression
  result was inspected. Never label results from it as a complete GLM-5.2 result
  and never extrapolate a whole-model ratio from it.

## Rolling one model at a time

The corpus does not have to be resident at once, and on a bounded quota it
should not be. Checkpoints are re-fetchable at any time because the manifest
pins an immutable revision and a SHA-256, so a dropped model is recoverable
exactly. Reports are not — they are the thing worth keeping.

```bash
python3 tools/model_cache.py status                    # what is cached, and its cost
python3 tools/model_cache.py fetch  medium-tinyllama-bf16
#   ... run the harness, write the report outside eval/cache/ ...
python3 tools/model_cache.py drop   medium-tinyllama-bf16
```

`fetch` verifies size and SHA-256 and fails loudly rather than leaving a bad
file; `drop` deletes only that model's cached weights. Peak residency is then
one model, so the whole tiered matrix needs 61 GB of headroom (the largest
single entry) rather than 134 GB.

Two rules for the loop:

- **Do not drop a model before its report is written and stored outside
  `eval/cache/`.** Re-fetching costs bandwidth; re-running costs the measurement.
- Do not delete between repetitions of the same model. Repetition counts are
  preregistered per model, and a re-fetch mid-sequence changes page-cache state
  the harness records.

## Where files land

`eval/run_eval.py` resolves the cache root from `$BREVIS_EVAL_CACHE`, defaulting
to `eval/cache/`, and stores each file at:

```
<cache-root>/<repo with / replaced by __>/<revision>/<file>
```

`fetch()` downloads to a `.part` file and `os.replace`s it into position, so an
interrupted download never leaves a truncated file that looks complete.

There is no download-only entry point. `run_eval.py` fetches lazily as it runs,
so acquisition is normally done ahead of time with the manifest and `curl`:

```bash
python3 - <<'PY' | bash
import json, pathlib, shlex
cache = pathlib.Path("eval/cache")
for m in json.load(open("eval/models-tiered.json")):
    if m["execution_stage"] != 1:      # one stage at a time
        continue
    for f in m["files"]:
        dst = cache / m["repo"].replace("/", "__") / m["revision"] / f["file"]
        url = f"https://huggingface.co/{m['repo']}/resolve/{m['revision']}/{f['file']}"
        print(f"mkdir -p {shlex.quote(str(dst.parent))} && "
              f"curl -fL --progress-bar -o {shlex.quote(str(dst))} {shlex.quote(url)}")
PY
```

Then verify against the manifest before running anything:

```bash
python3 - <<'PY'
import hashlib, json, pathlib
cache = pathlib.Path("eval/cache")
for m in json.load(open("eval/models-tiered.json")):
    for f in m["files"]:
        p = cache / m["repo"].replace("/", "__") / m["revision"] / f["file"]
        if not p.exists():
            continue
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        size_ok, sha_ok = p.stat().st_size == f["bytes"], got == f["sha256"]
        print(f"{'OK ' if size_ok and sha_ok else 'BAD'} {m['tag']:38s} {f['file']}")
PY
```

The harnesses re-check size and SHA-256 themselves and refuse to measure a
mismatch, so this is an early-failure convenience, not the integrity gate.

## Storage and the disk gate

Quota is not the constraint. The 134 GB of inputs fits comfortably in a 1024 GiB
`/scratch` quota, and 40 files is nothing against a 1000 K inode limit.

The constraint is `campaign_runner.py`'s generic-codec gate. It calls
`shutil.disk_usage(output.parent)` and requires 30% of the **filesystem** to
remain free — not 30% of your quota. On a shared parallel filesystem that is a
property of the whole cluster:

| Path | Total | Free | 30% reserve | Gate |
| --- | ---: | ---: | ---: | --- |
| `/scratch` | 5000.1 TB | 349.7 TB | 1500.0 TB | **FAIL** |
| `/home` | 219.9 TB | 55.4 TB | 66.0 TB | **FAIL** |
| `/project` | 1.0 TB | 0.7 TB | 0.3 TB | PASS |
| `$SLURM_TMPDIR` | 3.8 TB | 3.5 TB | 1.1 TB | PASS |

The rule was written for a dedicated ~2 TiB workspace volume
(`PROTOCOL.md`, `workspace_is_volume=false`), where "30% of the filesystem" and
"30% of my space" coincide. They do not coincide here.

This blocks only the generic-codec branch. `brevis_benchmarking.py` (the Brevis
system harness) has no such gate. `PROTOCOL.md` also forbids *starting* a
download that would violate the reserve, so resolve this before acquiring stage 4.

Options, in ascending order of commitment: point `--cache-root` and
`--staging-root` at a filesystem that passes; run only the Brevis system track;
or file a dated protocol amendment redefining the reserve against quota rather
than filesystem capacity. The third is a protocol decision, not a code fix.
