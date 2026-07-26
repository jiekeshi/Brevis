#!/usr/bin/env python3
"""Drive the self-extending grammar loop.

    inner loop   the engine searches for a program per tensor, in the current DSL
    outer loop   this file proposes new macros, verifies them, and keeps the
                 ones that make the corpus smaller at a fixed search budget

The model is confined to `propose`. Everything that decides what survives is
deterministic and re-runnable from the ledger.

Subcommands:
    evidence   what the measurements say about one model
    mine       macro candidates the search already keeps rediscovering
    bootstrap  admit mined candidates, one at a time, through the full gate
    propose    one round: measure, mine, ask a model, gate each proposal
    run        several propose rounds
    status     the library, the ledger, and the corpus as they stand
    verdict    success or failure: did unseen checkpoints shrink, bit-exactly,
               after the rule library was charged?

Outputs land in `autodsl/runs/`, which is git-ignored. These are development
measurements: `eval/` is the audited apparatus and nothing here is a formal
result.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import backend  # noqa: E402
import engine  # noqa: E402
import evaluate  # noqa: E402
import mine as mine_mod  # noqa: E402
import propose as propose_mod  # noqa: E402
import verify as verify_mod  # noqa: E402
from ledger import Ledger  # noqa: E402
from library import Library, OperatorTable  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
RUNS = HERE / "runs"
LIBRARIES = HERE / "libraries"
CACHE = REPO / "eval" / "cache"

# Three tiers. `develop` is mined and shown to the proposer; `validation` vetoes
# a candidate that helps develop but hurts elsewhere; `test` is read only by
# `verdict` and by nothing else, so it is the only tier a generalization claim
# may rest on. See evaluate.Split.
# ViT sits in `develop` rather than `validation`: a minimax fitness needs more
# than two models to be worth minimising over, and ViT had already been read
# once per candidate as a veto, so it was never test material. The test tier is
# unchanged and has been read only by `verdict` and `search.py --judge`.
DEVELOP = [
    "RedHatAI__SmolLM-135M-Instruct-quantized.w8a8",   # I8 + BF16
    "google-bert__bert-base-uncased",                  # F32
    "google__vit-base-patch16-224",                    # F32, vision
]
VALIDATION = [
    "TinyLlama__TinyLlama-1.1B-Chat-v1.0",             # BF16
]
TEST = [
    "openai__whisper-large-v3",                        # F16, speech
    "stabilityai__stable-diffusion-xl-base-1.0",       # F16, diffusion
    "Qwen__Qwen3-8B-Base",                             # BF16, larger LM
]


def find_models(names: list[str], *, required: bool = True) -> list[pathlib.Path]:
    found = []
    for name in names:
        if not required and not (CACHE / name).exists():
            continue
        matches = sorted((CACHE / name).rglob("*.safetensors")) if (CACHE / name).exists() else []
        if not matches:
            raise SystemExit(
                f"no checkpoint for {name!r} under {CACHE}. "
                f"Fetch it with: python3 tools/model_cache.py fetch <tag>"
            )
        found.extend(matches)
    return found


def load_table() -> OperatorTable:
    return OperatorTable.from_config(engine.config())


def load_library(path: pathlib.Path) -> Library:
    return Library.read(path) if path.exists() else Library()


def save_library(library: Library, path: pathlib.Path, budget, split, run: str) -> None:
    """Write the library with enough provenance to be read on its own.

    A library is the deliverable, so it has to say what search budget it was
    gated at and which checkpoints it never saw. Without that it is a set of
    assertions.
    """
    import dataclasses

    stamped = dataclasses.replace(library, provenance={
        "produced_by": "autodsl/loop.py",
        "run": run,
        "budget": budget.describe(),
        "split": split.describe(),
        "measurement_status": "development_only_not_formal",
        "note": (
            "Macros are expanded into primitive operators before serialization, "
            "so an archive written with this library is byte-identical to one "
            "written without it whenever both select the same program."
        ),
    })
    stamped.write(path)


def budget_from(args) -> engine.Budget:
    return engine.Budget(
        max_depth=args.max_depth,
        max_expansions=args.max_expansions,
        max_nodes=args.max_nodes,
        rerank_candidates=args.rerank_candidates,
        rerank_blocks=args.rerank_blocks,
        jobs=args.jobs,
    )


def proposer_evidence_source(split: evaluate.Split) -> pathlib.Path:
    """The only checkpoint a proposer is ever shown.

    Kept as a named function rather than an inline `split.develop[0]` so the
    containment is a property that can be asserted: `test_loop.py` fails if
    this ever returns a validation or test checkpoint. Everything the proposer
    sees is derived from this one bench report.
    """
    return split.develop[0]


def split_from(args) -> evaluate.Split:
    """Resolve the three tiers. Test checkpoints that are not cached are simply
    absent — a claim then rests on fewer models, which the verdict records."""
    return evaluate.Split(
        develop=tuple(find_models(args.develop)),
        validation=tuple(find_models(args.validation)),
        test=tuple(find_models(args.test, required=False)),
    )


# ==================== subcommands ====================


def cmd_evidence(args) -> int:
    table = load_table()
    budget = budget_from(args)
    library = load_library(args.library)
    macros = args.library if library.macros else None
    for model in find_models([args.model]):
        measurement = engine.bench(model, budget, macros)
        print(json.dumps(mine_mod.evidence(measurement.report, table), indent=2))
    return 0


def cmd_mine(args) -> int:
    table = load_table()
    budget = budget_from(args)
    library = load_library(args.library)
    macros = args.library if library.macros else None
    for model in find_models([args.model]):
        measurement = engine.bench(model, budget, macros)
        candidates = mine_mod.mine(measurement.report, table, library)
        print(f"# {model.parent.parent.name}")
        for candidate in candidates[: args.top]:
            print(
                f"  {candidate.bytes_governed / 1e6:9.1f} MB  "
                f"{candidate.blocks:6d} blocks  nodes={candidate.node_count} "
                f"holes={candidate.hole_count}  {candidate.shape}"
            )
    return 0


def _consider(
    macro,
    library: Library,
    table: OperatorTable,
    split: evaluate.Split,
    budget: engine.Budget,
    workdir: pathlib.Path,
    ledger: Ledger,
    baseline,
    origin: str,
) -> tuple[Library, object, bool]:
    """Run one candidate through every gate, recording whatever happens."""
    print(f"  candidate {macro.name}: {macro.shape()}")
    verdict = verify_mod.check(
        macro, library, table, list(split.develop), budget,
        workdir=workdir, roundtrip_models=list(split.develop[:1]),
    )
    if not verdict.ok:
        print(f"    rejected — {verdict.reason}")
        ledger.append(
            "rejected", name=macro.name, shape=macro.shape(), origin=origin,
            reason=verdict.reason, stage="verify", detail=verdict.detail,
            evaluator=evaluate.VERSION,
        )
        return library, baseline, False

    candidate = library.with_macro(macro)
    decision, after = evaluate.evaluate(
        library, candidate, split, budget, workdir, baseline=baseline
    )
    print(f"    {'accepted' if decision.accept else 'rejected'} — {decision.reason}")
    ledger.append(
        "accepted" if decision.accept else "rejected",
        name=macro.name, shape=macro.shape(), origin=origin, why=macro.why,
        reason=decision.reason, stage="evaluate", evaluator=evaluate.VERSION,
        decision=decision.describe(), verify=verdict.detail,
    )
    if not decision.accept:
        return library, baseline, False
    return candidate, after, True


def cmd_bootstrap(args) -> int:
    """Admit mined shapes through the same gate a proposal faces.

    This is the loop with the model removed. It answers a question worth
    answering before spending a single token: how much of the available gain is
    reachable by mining alone?
    """
    table = load_table()
    budget = budget_from(args)
    split = split_from(args)
    library = load_library(args.library)
    workdir = RUNS / args.run
    workdir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(workdir / "ledger.jsonl")

    macros_path = args.library if library.macros else None
    baseline = (
        evaluate.measure_corpus(split.develop, budget, macros_path),
        evaluate.measure_corpus(split.validation, budget, macros_path),
    )
    print(
        f"baseline: develop {baseline[0].archive_bytes} bytes, "
        f"validation {baseline[1].archive_bytes} bytes"
    )
    ledger.append(
        "baseline", budget=budget.describe(), split=split.describe(),
        develop_bytes=baseline[0].archive_bytes,
        validation_bytes=baseline[1].archive_bytes,
        library_sha256=library.sha256(), evaluator=evaluate.VERSION,
        environment=engine.environment_notes(),
    )

    pooled: dict[str, mine_mod.Candidate] = {}
    for model in split.develop:
        measurement = engine.bench(model, budget, macros_path)
        for candidate in mine_mod.mine(measurement.report, table, library):
            existing = pooled.get(candidate.shape)
            if existing is None or candidate.bytes_governed > existing.bytes_governed:
                pooled[candidate.shape] = candidate
    ranked = mine_mod.diversify(
        sorted(pooled.values(), key=lambda c: -c.bytes_governed),
        per_family=args.per_family,
    )[: args.top]
    print(f"{len(ranked)} mined candidates to consider")

    for candidate in ranked:
        name = mine_mod.suggest_name(candidate.shape, library.names())
        macro = mine_mod.to_macro(candidate, name, origin="mined")
        library, baseline, _ = _consider(
            macro, library, table, split, budget, workdir, ledger, baseline, "mined"
        )
        save_library(library, args.library, budget, split, args.run)
    _report(library, baseline, args.library, ledger, budget, split, args.run)
    return 0


def cmd_propose(args) -> int:
    table = load_table()
    budget = budget_from(args)
    split = split_from(args)
    library = load_library(args.library)
    workdir = RUNS / args.run
    workdir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(workdir / "ledger.jsonl")

    llm = (
        backend.ScriptedBackend(replies=[args.dry_run_reply])
        if args.dry_run_reply
        else backend.Backend(model=args.model_name)
    )

    macros_path = args.library if library.macros else None
    baseline = (
        evaluate.measure_corpus(split.develop, budget, macros_path),
        evaluate.measure_corpus(split.validation, budget, macros_path),
    )
    ledger.append(
        "baseline", budget=budget.describe(), split=split.describe(),
        develop_bytes=baseline[0].archive_bytes,
        validation_bytes=baseline[1].archive_bytes,
        library_sha256=library.sha256(), backend=llm.describe(),
        evaluator=evaluate.VERSION, environment=engine.environment_notes(),
    )

    for round_index in range(args.rounds):
        print(f"\n=== round {round_index + 1}/{args.rounds} ===")
        macros_path = args.library if library.macros else None
        measurement = engine.bench(
            proposer_evidence_source(split), budget, macros_path)
        evidence = mine_mod.evidence(measurement.report, table)
        mined = mine_mod.mine(measurement.report, table, library)

        try:
            proposal = propose_mod.propose(
                llm, evidence, library, table,
                budget=budget.describe(), mined=mined,
                rejected=ledger.rejected(evaluate.VERSION), wanted=args.wanted,
            )
        except Exception as exc:  # a bad reply ends the round, not the run
            print(f"  proposal failed: {type(exc).__name__}: {exc}")
            ledger.append("proposal_failed", round=round_index, error=str(exc)[:2000])
            continue

        (workdir / f"reply-{round_index:02d}.txt").write_text(proposal.raw_reply)
        (workdir / f"prompt-{round_index:02d}.txt").write_text(proposal.prompt)
        print(f"  {len(proposal.macros)} macros proposed")
        ledger.append(
            "proposed", round=round_index, backend=llm.describe(),
            macros=[{"name": m.name, "shape": m.shape(), "why": m.why}
                    for m in proposal.macros],
        )

        for macro in proposal.macros:
            library, baseline, kept = _consider(
                macro, library, table, split, budget, workdir, ledger,
                baseline, "proposed",
            )
            if kept:
                save_library(library, args.library, budget, split, args.run)

    _report(library, baseline, args.library, ledger, budget, split, args.run)
    return 0


def cmd_verdict(args) -> int:
    """Answer the only question that settles this: did unseen checkpoints get
    smaller, bit-exactly, after the rule library was paid for?"""
    budget = budget_from(args)
    split = split_from(args)
    library = load_library(args.library)
    workdir = RUNS / args.run
    workdir.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(workdir / "ledger.jsonl")

    models = getattr(split, args.tier)
    bit_exact = True
    detail = {}
    if library.macros and models:
        check = verify_mod.roundtrip(library, list(models), budget, workdir)
        bit_exact, detail = check.ok, check.detail
        print(f"bit-exactness on {args.tier}: {check.reason}")

    result = evaluate.verdict(
        library, split, budget, workdir, bit_exact=bit_exact,
        tier=args.tier, test_log=RUNS / "test-log.jsonl",
    )
    print(f"\n{'SUCCESS' if result.success else 'FAILURE'} — {result.reason}")
    print(f"  {result.tier:10s} {result.before} -> {result.after} bytes "
          f"({result.delta:+d} charged)")
    print(f"  library    {len(library.macros)} macros, "
          f"{result.library_serialized_bytes} bytes, "
          f"sha256={library.sha256()[:16]}")
    print(f"  models     {', '.join(result.models) or '(none cached)'}")
    if result.prior_measurements_of_this_tier:
        print(f"  NOTE: the {result.tier} tier has been measured "
              f"{result.prior_measurements_of_this_tier} time(s) before; "
              f"see autodsl/runs/test-log.jsonl")
    ledger.append("verdict", library_sha256=library.sha256(),
                  budget=budget.describe(), split=split.describe(),
                  roundtrip=detail, **result.describe())
    return 0 if result.success else 1


def cmd_status(args) -> int:
    budget = budget_from(args)
    split = split_from(args)
    library = load_library(args.library)
    ledger = Ledger(RUNS / args.run / "ledger.jsonl")
    print(f"library {args.library} sha256={library.sha256()[:16]} "
          f"macros={len(library.macros)}")
    for macro in library.macros:
        print(f"  {macro.name:32s} {macro.shape():44s} [{macro.origin}]")
    print(f"ledger: {json.dumps(ledger.summary())}")
    if args.measure:
        macros_path = args.library if library.macros else None
        for half, models in (("develop", split.develop),
                             ("validation", split.validation),
                             ("test", split.test)):
            if not models:
                continue
            result = evaluate.measure_corpus(models, budget, macros_path)
            print(f"  {half:10s} {result.archive_bytes:>14d} bytes  "
                  f"ratio {result.ratio:.4f}  "
                  f"({len(models)} models)")
    return 0


def _report(library: Library, baseline, path: pathlib.Path, ledger: Ledger,
            budget, split, run: str) -> None:
    save_library(library, path, budget, split, run)
    print(
        f"\nlibrary: {len(library.macros)} macros, sha256={library.sha256()[:16]}\n"
        f"  develop {baseline[0].archive_bytes} bytes   "
        f"validation {baseline[1].archive_bytes} bytes\n"
        f"  ledger  {json.dumps(ledger.summary())}\n"
        f"  written to {path}"
    )


# ==================== argument handling ====================


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", type=pathlib.Path,
                        default=LIBRARIES / "learned.json")
    parser.add_argument("--run", default="latest", help="subdirectory under autodsl/runs/")
    parser.add_argument("--develop", nargs="+", default=DEVELOP)
    parser.add_argument("--validation", nargs="+", default=VALIDATION)
    parser.add_argument("--test", nargs="+", default=TEST)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-expansions", type=int, default=256)
    parser.add_argument("--max-nodes", type=int, default=12)
    parser.add_argument("--rerank-candidates", type=int, default=64)
    parser.add_argument("--rerank-blocks", type=int, default=8)
    parser.add_argument("--jobs", type=int, default=12)
    sub = parser.add_subparsers(dest="command", required=True)

    evidence = sub.add_parser("evidence", help="what the measurements say")
    evidence.add_argument("model", default=DEVELOP[0], nargs="?")
    evidence.set_defaults(func=cmd_evidence)

    mine_cmd = sub.add_parser("mine", help="macro candidates from measured programs")
    mine_cmd.add_argument("model", default=DEVELOP[0], nargs="?")
    mine_cmd.add_argument("--top", type=int, default=25)
    mine_cmd.set_defaults(func=cmd_mine)

    bootstrap = sub.add_parser("bootstrap", help="gate mined candidates, no model")
    bootstrap.add_argument("--top", type=int, default=8)
    bootstrap.add_argument("--per-family", type=int, default=1,
                           help="candidates to keep per transform skeleton")
    bootstrap.set_defaults(func=cmd_bootstrap)

    propose = sub.add_parser("propose", help="ask a model for macros and gate them")
    propose.add_argument("--rounds", type=int, default=1)
    propose.add_argument("--wanted", type=int, default=4)
    propose.add_argument("--model-name", default=backend.DEFAULT_MODEL,
                         help=f"one of {', '.join(backend.KNOWN_MODELS)}")
    propose.add_argument("--dry-run-reply", default=None,
                         help="use this text instead of calling a model")
    propose.set_defaults(func=cmd_propose)

    run = sub.add_parser("run", help="several propose rounds")
    run.add_argument("--rounds", type=int, default=3)
    run.add_argument("--wanted", type=int, default=4)
    run.add_argument("--model-name", default=backend.DEFAULT_MODEL)
    run.add_argument("--dry-run-reply", default=None)
    run.set_defaults(func=cmd_propose)

    verdict = sub.add_parser(
        "verdict", help="did unseen checkpoints shrink, bit-exactly, after "
                        "charging the library?")
    verdict.add_argument("--tier", default="test",
                         choices=("test", "validation", "develop"),
                         help="which tier to judge on; anything but `test` is "
                              "a diagnostic, not a generalization claim")
    verdict.set_defaults(func=cmd_verdict)

    status = sub.add_parser("status", help="library, ledger, and corpus")
    status.add_argument("--measure", action="store_true")
    status.set_defaults(func=cmd_status)

    args = parser.parse_args(argv)
    args.library.parent.mkdir(parents=True, exist_ok=True)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
