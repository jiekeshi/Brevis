# Cluster pitfalls

Traps hit while bringing Brevis up on Nibi (Alliance). Each one cost real time
and none of them announces itself clearly, so they are recorded here rather than
rediscovered. `setup_env.sh` already handles all of them; this file explains why
the script looks the way it does.

## 1. `OMP_NUM_THREADS` silently disqualifies formal runs

**Symptom.** 13 errors and 2 failures in `python3 -m unittest discover -s eval`,
all in `test_campaign_runner` / `test_brevis_benchmarking`:

```
CampaignError: generic formal execution requires codec environment
variables to be truly unset; present: OMP_NUM_THREADS
```

**Cause.** `OMP_NUM_THREADS` is one of `CODEC_ENVIRONMENT_VARIABLES` in
`eval/benchmarking.py` (alongside `GZIP`, `BZIP2`, `XZ_DEFAULTS`, `XZ_OPT`,
`ZSTD_CLEVEL`, `ZSTD_NBTHREADS`, `LZ4_CLEVEL`, `BROTLI_PARAM_*`). Those names
change codec behaviour from outside the command line, so the formal gate demands
they be *absent from the environment*, not merely set to a safe value.

The general HPC advice to `export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK` is
exactly wrong here.

**Fix.** Never export any of those names in a shell that will run the harness.
Pass thread counts explicitly with `brevis --jobs N`. The gate is doing its job:
a serial-codec timing track is only meaningful if nothing external retunes the
codecs.

## 2. `PIP_PREFIX` redirects installs out of the active venv, and pip says OK

**Symptom.** `virtualenv` succeeds, `pip install --no-index -r requirements.txt`
prints `Successfully installed numpy-2.4.2+computecanada` and exits 0, and then:

```
$ python -c "import numpy"
ModuleNotFoundError: No module named 'numpy'
$ pip list
pip 26.0.1+computecanada          # numpy is not there
```

**Cause.** Some Alliance sessions — JupyterHub-launched ones in particular —
export `PIP_PREFIX=$SLURM_TMPDIR`. pip honours it over the active virtual
environment, so the package lands in `$PIP_PREFIX/lib/python3.12/site-packages`
while the venv stays empty. Nothing in the output hints at this. `PYTHONPATH` is
exported the same way and leaks cvmfs site-packages into the venv, which can
mask a missing dependency.

**Fix.** `unset PIP_PREFIX PYTHONPATH` before creating or using the venv.
Confirm placement rather than trusting pip's exit code:

```bash
python -c "import numpy; print(numpy.__file__)"   # must be inside the venv
```

## 3. Zig has no Lmod module

`module spider zig` returns only R packages (`zigg`, `RcppZiggurat`). Zig 0.16.0
must be installed by hand; see README *Cluster Setup*. It is a per-user install,
not a system one.

Do **not** put it under `/project`: that quota is close to its inode limit, and
the toolchain is 19.5 K files / 347 MB.

## 4. Small-file I/O dominates build time

A cold `zig build -Doptimize=ReleaseFast` took **3m41s wall for 38s of user
CPU**. Extracting the Zig tarball onto `/home` took minutes for 55 MB. Both are
the parallel filesystem handling many small files, not slow compilation.

`setup_env.sh` points `ZIG_GLOBAL_CACHE_DIR` and the venv at `$SLURM_TMPDIR`
(node-local disk). Incremental rebuilds then run in well under a minute.

## 5. Single Python tests must be run from inside `eval/`

The `eval/` modules import each other by bare name (`import benchmarking`), so
they need `eval/` on `sys.path`. `python3 -m unittest discover -s eval` arranges
that; the dotted form does not:

```bash
python3 -m unittest eval.test_campaign_runner -k stage_gate   # ImportError
cd eval && python3 -m unittest test_campaign_runner -k stage_gate   # OK
```

A failure here looks like a broken test, but is `ModuleNotFoundError: No module
named 'benchmarking'` a few lines up.

## 6. Known environment-sensitive test failure

`test_brevis_benchmarking.BrevisSystemBenchmarkTests.
test_timeout_is_retained_without_losing_valid_archive_evidence` fails on this
cluster with `IndexError: list index out of range`.

It is not a Brevis defect and not a setup error. The test hardcodes
`timeout_seconds=0.05`, but the harness first runs a configuration probe that
launches a Python fake `brevis`. Interpreter startup here is ~27 ms median off
cvmfs, so the probe exceeds the 50 ms budget and dies before the bench stage the
test wants to exercise, leaving `configurations` empty:

```
configuration probe timed out after 0.05 seconds
```

Raising the budget confirms the assertions themselves hold:

| `timeout_seconds` | result |
| --- | --- |
| 0.05 | probe dies, `configurations == []` |
| 0.30 | `timed_out=True`, `bit_exact=True` |
| 1.00 | `timed_out=True`, `bit_exact=True` |

The test is unmodified. Changing a threshold in the audited `eval/` tree is a
protocol decision, not a cleanup. Current status is **158/159 passing**.
