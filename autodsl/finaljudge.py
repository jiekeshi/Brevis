#!/usr/bin/env python3
"""Measure the ceiling library, and its subset without the suspect macro, on
the frozen test tier. Bytes first for both, then a bit-exact roundtrip on the
winner only, because a 26 GB roundtrip costs more than the measurement does."""
from __future__ import annotations
import json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import engine, evaluate, loop, verify
from library import Library

budget = engine.Budget()
split = evaluate.Split(develop=(), validation=(),
                       test=tuple(loop.find_models(loop.TEST, required=False)))
workdir = loop.RUNS / "ceiling"
full = Library.read(workdir / "library-ceiling.json")
trimmed = full.without(next(m.name for m in full.macros
                            if m.shape() == "zigzag(split_field(?,?))"))

before = evaluate.measure_corpus(split.test, budget, None)
print(f"test baseline {before.archive_bytes}\n", flush=True)

results = []
for name, library in (("ceiling-3", full), ("ceiling-2", trimmed)):
    path = workdir / f"{name}.json"
    library.write(path)
    after = evaluate.measure_corpus(split.test, budget, path)
    serialized = len(library.canonical_bytes())
    delta = after.archive_bytes + serialized - before.archive_bytes
    improved = sum(1 for k, v in before.per_model.items()
                   if k in after.per_model and after.per_model[k] < v)
    worst = max((after.per_model[k] - v for k, v in before.per_model.items()
                 if k in after.per_model), default=0)
    slow = after.planning_wall_ms / max(before.planning_wall_ms, 1)
    results.append((delta, name, library, improved, worst, slow))
    print(f"{name:11s} {delta:>+12d}  {improved}/{len(before.per_model)} files "
          f"better  worst +{worst}  planning {slow:.2f}x  "
          f"{[m.shape() for m in library.macros]}", flush=True)

results.sort()
delta, name, library, improved, worst, slow = results[0]
print(f"\nbest: {name} at {delta:+d}; verifying bit-exactness on all "
      f"{len(split.test)} files", flush=True)
check = verify.roundtrip(library, list(split.test), budget, workdir)
print(f"bit-exact: {check.ok} — {check.reason}", flush=True)
record = {"tier": "test", "source": "ceiling", "arm": name, "delta": delta,
          "bit_exact": check.ok, "files_improved": improved,
          "files": len(before.per_model), "worst_file_regression": worst,
          "planning_slowdown": round(slow, 3),
          "test_before": before.archive_bytes,
          "macros": [m.shape() for m in library.macros],
          "library_sha256": library.sha256()}
with (loop.RUNS / "test-log.jsonl").open("a") as handle:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
print(json.dumps(record, indent=2))
