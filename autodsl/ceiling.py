#!/usr/bin/env python3
"""Greedy forward selection over every macro this project has ever found.

Not a method — a ceiling. The question it answers is "how much is reachable at
all with macros over the existing operator set", which is the number that
decides whether the direction is worth more effort. It cheats on purpose: the
pool is the union of everything discovered across every run, including runs
whose libraries were later rejected.
"""
from __future__ import annotations
import json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import engine, evaluate, loop, search
from library import Library

budget = engine.Budget()
split = evaluate.Split(develop=tuple(loop.find_models(loop.DEVELOP)),
                       validation=tuple(loop.find_models(loop.VALIDATION)),
                       test=tuple(loop.find_models(loop.TEST, required=False)))
workdir = loop.RUNS / "ceiling"
workdir.mkdir(parents=True, exist_ok=True)
pool = Library.read(loop.RUNS / "pool.json")
out = workdir / "trace.jsonl"

evaluator = search.Evaluator(split, budget, workdir, cap=10**6, mode="minimax")
baseline = evaluator.measure(Library(), free=True)
evaluator.baseline = baseline
print(f"baseline develop {baseline.develop_bytes}", flush=True)

chosen, best = Library(), baseline
remaining = list(pool.macros)
while remaining:
    scored = []
    for macro in remaining:
        grown = chosen.with_macro(macro)
        fitness = evaluator.measure(grown)
        scored.append((fitness, macro))
        print(f"  + {macro.shape():52s} {fitness.objective - baseline.objective:>+10d}"
              f"  worst {fitness.worst_relative_gain:+.6f}"
              f"  {'' if fitness.feasible else '[INFEASIBLE]'}", flush=True)
    scored.sort(key=lambda pair: (not pair[0].feasible, pair[0].score()))
    fitness, macro = scored[0]
    if not fitness.better_than(best):
        print(f"\nno remaining macro improves; stopping at {len(chosen.macros)}",
              flush=True)
        break
    chosen, best = chosen.with_macro(macro), fitness
    remaining = [m for m in remaining if m.shape() != macro.shape()]
    print(f"=> take {macro.shape()}   develop {best.objective - baseline.objective:+d}\n",
          flush=True)
    with out.open("a") as handle:
        handle.write(json.dumps({"size": len(chosen.macros),
                                 "shapes": [m.shape() for m in chosen.macros],
                                 "develop_delta": best.objective - baseline.objective,
                                 "worst": best.worst_relative_gain}) + "\n")

loop.save_library(chosen, workdir / "library-ceiling.json", budget, split, "ceiling")
print(f"\nceiling library: {len(chosen.macros)} macros, "
      f"develop {best.objective - baseline.objective:+d}")
for m in chosen.macros:
    print(f"  [{m.origin}] {m.shape()}")
