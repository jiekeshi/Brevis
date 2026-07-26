#!/usr/bin/env python3
"""Experiments that characterise what a macro library actually contributes.

The acceptance gate answers "is this library worth keeping". These answer the
harder questions a reader should ask before believing it.

    budget-curve   Does the primitive grammar find the same programs if you
                   simply give it more search? This is the experiment that
                   decides what kind of contribution a macro is:
                     - primitive catches up at a larger budget  -> acceleration
                     - primitive never catches up, at equal expansions AND at
                       equal wall time                          -> reachability
                     - neither, until a new operator is added   -> expressiveness
    per-tensor     Is the gain broad or concentrated? A single enormous tensor
                   producing 99% of the saving is a finding about that tensor,
                   not a reusable rule.
    archives       Write real `.brv` archives both ways and hash them, so a
                   claim about bytes is checkable rather than reported.

Results stream to JSONL as they are produced, so a long sweep is analysable
before it finishes and survives being interrupted.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import engine  # noqa: E402
import loop  # noqa: E402
from library import Library  # noqa: E402

DEFAULT_BUDGETS = (64, 128, 256, 512, 1024, 2048, 4096)


def _append(path: pathlib.Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _macro_tensors(report: dict) -> int:
    return sum(1 for t in report.get("tensors", []) if t.get("macros_used"))


def _tensor_bytes(report: dict) -> dict[str, int]:
    return {
        t["name"]: t["encoded_bytes_without_frame_headers"]
        for t in report.get("tensors", [])
    }


# ==================== budget curve ====================


def cmd_budget_curve(args) -> int:
    library_path = args.library if Library.read(args.library).macros else None
    if library_path is None:
        raise SystemExit(f"{args.library} has no macros; nothing to compare against")
    out = loop.RUNS / args.run / "budget-curve.jsonl"
    models = loop.find_models(args.models)

    for model in models:
        for expansions in args.budgets:
            for arm, macros in (("primitive", None), ("macro", library_path)):
                budget = engine.Budget(
                    max_depth=args.max_depth,
                    max_expansions=expansions,
                    max_nodes=args.max_nodes,
                    rerank_candidates=args.rerank_candidates,
                    rerank_blocks=args.rerank_blocks,
                    jobs=args.jobs,
                )
                started = time.monotonic()
                measurement = engine.bench(model, budget, macros)
                wall = time.monotonic() - started
                record = {
                    "model": engine.label(model),
                    "arm": arm,
                    "max_expansions": expansions,
                    "archive_bytes": measurement.archive_bytes,
                    "planning_wall_ms": measurement.planning_wall_ms,
                    "process_wall_s": round(wall, 3),
                    "macro_built_tensors": _macro_tensors(measurement.report),
                    "tensors": len(measurement.report["tensors"]),
                    "distinct_block_programs": len(
                        {b["program"] for b in measurement.report["blocks"]}
                    ),
                }
                _append(out, record)
                print(
                    f"{record['model'][:34]:34s} {arm:9s} e={expansions:<5d} "
                    f"{record['archive_bytes']:>12d}  "
                    f"{record['planning_wall_ms']:>7d} ms  "
                    f"macro-built {record['macro_built_tensors']}",
                    flush=True,
                )
    print(f"\nwrote {out}")
    return 0


def cmd_report_curve(args) -> int:
    """Read a budget-curve JSONL and answer the acceleration question."""
    path = loop.RUNS / args.run / "budget-curve.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    by_model: dict[str, dict] = collections.defaultdict(dict)
    for row in rows:
        by_model[row["model"]][(row["arm"], row["max_expansions"])] = row

    for model, cells in sorted(by_model.items()):
        budgets = sorted({key[1] for key in cells})
        print(f"\n{model}")
        print(f"  {'expansions':>10} {'primitive':>14} {'macro':>14} {'delta':>12} "
              f"{'prim ms':>9} {'macro ms':>9} {'macro-built':>12}")
        macro_at_reference = None
        for expansions in budgets:
            primitive = cells.get(("primitive", expansions))
            macro = cells.get(("macro", expansions))
            if not primitive or not macro:
                continue
            if expansions == args.reference:
                macro_at_reference = macro["archive_bytes"]
            print(f"  {expansions:>10} {primitive['archive_bytes']:>14d} "
                  f"{macro['archive_bytes']:>14d} "
                  f"{macro['archive_bytes'] - primitive['archive_bytes']:>+12d} "
                  f"{primitive['planning_wall_ms']:>9d} "
                  f"{macro['planning_wall_ms']:>9d} "
                  f"{macro['macro_built_tensors']:>12d}")

        if macro_at_reference is None:
            continue
        # Does the primitive grammar ever reach what macros reach at the
        # reference budget, given any budget in the sweep?
        caught_up = [
            expansions for expansions in budgets
            if ("primitive", expansions) in cells
            and cells[("primitive", expansions)]["archive_bytes"] <= macro_at_reference
        ]
        reference_cell = cells.get(("macro", args.reference))
        if caught_up:
            first = min(caught_up)
            cost = cells[("primitive", first)]["planning_wall_ms"]
            print(f"  -> primitive matches macro@{args.reference} at "
                  f"{first} expansions ({cost} ms vs "
                  f"{reference_cell['planning_wall_ms']} ms): ACCELERATION")
        else:
            best = min(cells[("primitive", e)]["archive_bytes"]
                       for e in budgets if ("primitive", e) in cells)
            print(f"  -> primitive never matches macro@{args.reference} within "
                  f"{max(budgets)} expansions (best {best}, "
                  f"{best - macro_at_reference:+d}): REACHABILITY")
    return 0


# ==================== per-tensor distribution ====================


def cmd_per_tensor(args) -> int:
    library = Library.read(args.library)
    if not library.macros:
        raise SystemExit(f"{args.library} has no macros")
    out = loop.RUNS / args.run / "per-tensor.jsonl"
    budget = loop.budget_from(args)

    for model in loop.find_models(args.models):
        name = engine.label(model)
        without = engine.bench(model, budget, None)
        with_macros = engine.bench(model, budget, args.library)
        before = _tensor_bytes(without.report)
        after = _tensor_bytes(with_macros.report)
        meta = {t["name"]: t for t in with_macros.report["tensors"]}

        deltas = []
        for tensor, base in before.items():
            if tensor not in after:
                continue
            delta = after[tensor] - base
            info = meta[tensor]
            deltas.append({
                "tensor": tensor,
                "dtype": info["dtype"],
                "shape": info["shape"],
                "raw_bytes": info["raw_bytes"],
                "before": base,
                "after": after[tensor],
                "delta": delta,
                "macros_used": info.get("macros_used") or [],
            })

        improved = [d for d in deltas if d["delta"] < 0]
        worsened = [d for d in deltas if d["delta"] > 0]
        unchanged = [d for d in deltas if d["delta"] == 0]
        total = sum(d["delta"] for d in deltas)
        improved.sort(key=lambda d: d["delta"])

        # How concentrated is the saving? If one tensor carries it, this is a
        # finding about that tensor rather than a reusable rule.
        gains = [-d["delta"] for d in improved] or [0]
        top1 = gains[0] if gains else 0
        top5 = sum(gains[:5])

        summary = {
            "model": name,
            "total_delta": total,
            "tensors": len(deltas),
            "improved": len(improved),
            "worsened": len(worsened),
            "unchanged": len(unchanged),
            "macro_built_tensors": _macro_tensors(with_macros.report),
            "best_gain": top1,
            "top1_share_of_gain": round(top1 / max(sum(gains), 1), 4),
            "top5_share_of_gain": round(top5 / max(sum(gains), 1), 4),
            "median_gain": int(statistics.median(gains)),
            "mean_gain": int(statistics.fmean(gains)),
            "worst_regression": max((d["delta"] for d in worsened), default=0),
            "by_dtype": _group(deltas, "dtype"),
            "top_contributors": improved[:10],
            "top_regressions": sorted(worsened, key=lambda d: -d["delta"])[:10],
        }
        _append(out, summary)

        print(f"\n{name}")
        print(f"  total {total:+d} bytes over {len(deltas)} tensors")
        print(f"  improved {len(improved)}  worsened {len(worsened)}  "
              f"unchanged {len(unchanged)}  macro-built {summary['macro_built_tensors']}")
        print(f"  gain concentration: top-1 {summary['top1_share_of_gain']:.1%}, "
              f"top-5 {summary['top5_share_of_gain']:.1%}")
        print(f"  gain per improved tensor: median {summary['median_gain']}, "
              f"mean {summary['mean_gain']}, best {summary['best_gain']}")
        print(f"  worst regression {summary['worst_regression']} bytes")
        for dtype, entry in sorted(summary["by_dtype"].items()):
            print(f"    {dtype:9s} {entry['tensors']:4d} tensors  "
                  f"{entry['delta']:+12d} bytes  "
                  f"{entry['improved']} better / {entry['worsened']} worse")
    print(f"\nwrote {out}")
    return 0


def _group(deltas: list[dict], key: str) -> dict:
    out: dict[str, dict] = {}
    for entry in deltas:
        bucket = out.setdefault(
            entry[key], {"tensors": 0, "delta": 0, "improved": 0, "worsened": 0}
        )
        bucket["tensors"] += 1
        bucket["delta"] += entry["delta"]
        bucket["improved"] += entry["delta"] < 0
        bucket["worsened"] += entry["delta"] > 0
    return out


# ==================== real archives ====================


def cmd_archives(args) -> int:
    """Write both archives for real and hash them. Bytes you can check."""
    library = Library.read(args.library)
    out = loop.RUNS / args.run / "archives.jsonl"
    budget = loop.budget_from(args)
    workdir = loop.RUNS / args.run / "archives"
    workdir.mkdir(parents=True, exist_ok=True)

    for model in loop.find_models(args.models):
        name = engine.label(model)
        record = {"model": name, "budget": budget.describe(),
                  "library_sha256": library.sha256()}
        for arm, macros in (("primitive", None), ("macro", args.library)):
            path = workdir / f"{name.replace('/', '__')}.{arm}.brv"
            size = engine.compress_and_verify(model, path, budget, macros)
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 22), b""):
                    digest.update(chunk)
            record[arm] = {"bytes": size, "sha256": digest.hexdigest(),
                           "verified_bit_exact": True}
            if not args.keep:
                path.unlink()
        record["delta"] = record["macro"]["bytes"] - record["primitive"]["bytes"]
        _append(out, record)
        print(f"{name}\n  primitive {record['primitive']['bytes']:>12d}  "
              f"{record['primitive']['sha256']}\n"
              f"  macro     {record['macro']['bytes']:>12d}  "
              f"{record['macro']['sha256']}\n"
              f"  delta     {record['delta']:>+12d}", flush=True)
    print(f"\nwrote {out}")
    return 0


# ==================== argument handling ====================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", type=pathlib.Path,
                        default=loop.LIBRARIES / "learned.json")
    parser.add_argument("--run", default="experiments")
    # Comma-separated rather than nargs="+": a greedy list swallows the
    # subcommand name, and argparse then reports a confusingly unrelated error.
    parser.add_argument("--models", type=lambda v: [s for s in v.split(",") if s],
                        default=loop.DEVELOP + loop.VALIDATION)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-expansions", type=int, default=256)
    parser.add_argument("--max-nodes", type=int, default=12)
    parser.add_argument("--rerank-candidates", type=int, default=64)
    parser.add_argument("--rerank-blocks", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=12)
    sub = parser.add_subparsers(dest="command", required=True)

    curve = sub.add_parser("budget-curve",
                           help="sweep the search budget for both grammars")
    curve.add_argument("--budgets", type=lambda v: [int(x) for x in v.split(",")],
                       default=list(DEFAULT_BUDGETS))
    curve.set_defaults(func=cmd_budget_curve)

    report = sub.add_parser("report-curve", help="read a sweep and classify it")
    report.add_argument("--reference", type=int, default=256,
                        help="the macro budget the primitive grammar must match")
    report.set_defaults(func=cmd_report_curve)

    per_tensor = sub.add_parser("per-tensor", help="where the bytes came from")
    per_tensor.set_defaults(func=cmd_per_tensor)

    archives = sub.add_parser("archives", help="write and hash real .brv both ways")
    archives.add_argument("--keep", action="store_true")
    archives.set_defaults(func=cmd_archives)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
