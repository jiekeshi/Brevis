"""Decide whether a candidate macro is safe and whether it does anything.

Three questions, in increasing cost, each of which can reject on its own:

  1. Does the engine accept the library at all?
  2. Does a real archive written with it still decode bit-exact?
  3. Does the macro ever actually get chosen?

(1) and (2) are safety. A macro is a composition of operators that are already
reversible, so (2) should never fail — which is exactly why it is worth
running: a failure means an assumption broke, not that this macro was unlucky.

(3) is not safety but it is the cheapest way to reject: a macro that is never
selected still costs an expansion at every hole, so an inert macro is a
strictly negative change and there is no reason to measure its bytes.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import tempfile

import engine
from library import Library, LibraryError, Macro, OperatorTable, validate_macro


@dataclasses.dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str
    detail: dict = dataclasses.field(default_factory=dict)


def write_library(library: Library, directory: pathlib.Path, stem: str) -> pathlib.Path:
    path = directory / f"{stem}.json"
    library.write(path)
    return path


def structural(macro: Macro, table: OperatorTable, library: Library) -> Verdict:
    try:
        validate_macro(macro, table)
    except LibraryError as exc:
        return Verdict(False, f"structural: {exc}")
    if macro.name in library.names():
        return Verdict(False, f"structural: name {macro.name!r} already used")
    if macro.shape() in library.shapes():
        return Verdict(False, f"structural: body {macro.shape()} already in library")
    return Verdict(True, "structural ok")


def engine_accepts(library: Library, workdir: pathlib.Path) -> Verdict:
    path = write_library(library, workdir, "candidate")
    try:
        config = engine.config(macros=path)
    except engine.EngineError as exc:
        return Verdict(False, f"engine rejected the library: {exc}")
    if config["macro_count"] != len(library.macros):
        return Verdict(
            False,
            f"engine loaded {config['macro_count']} of {len(library.macros)} macros",
        )
    return Verdict(True, "engine accepts", {"macro_count": config["macro_count"]})


def roundtrip(
    library: Library,
    models: list[pathlib.Path],
    budget: engine.Budget,
    workdir: pathlib.Path,
) -> Verdict:
    """Write a real archive per model and prove each decodes bit-exact."""
    path = write_library(library, workdir, "candidate")
    sizes = {}
    for model in models:
        archive = workdir / f"{model.stem}-{library.sha256()[:12]}.brv"
        try:
            sizes[engine.label(model)] = engine.compress_and_verify(
                model, archive, budget, path
            )
        except engine.EngineError as exc:
            return Verdict(False, f"roundtrip failed on {engine.label(model)}: {exc}")
        finally:
            archive.unlink(missing_ok=True)
    return Verdict(True, "bit-exact on every probe", {"archive_bytes": sizes})


def macros_selected(report: dict) -> dict[str, int]:
    """How many tensors chose a program that a given macro built."""
    counts: dict[str, int] = {}
    for tensor in report.get("tensors", []):
        for name in tensor.get("macros_used") or []:
            counts[name] = counts.get(name, 0) + 1
    return counts


def fires(
    library: Library,
    macro: Macro,
    models: list[pathlib.Path],
    budget: engine.Budget,
    workdir: pathlib.Path,
) -> Verdict:
    path = write_library(library, workdir, "candidate")
    total = 0
    per_model = {}
    for model in models:
        measurement = engine.bench(model, budget, path)
        selected = macros_selected(measurement.report).get(macro.name, 0)
        per_model[engine.label(model)] = selected
        total += selected
    if total == 0:
        return Verdict(
            False,
            "never selected on any probe model, so it can only dilute the budget",
            {"selected_tensors": per_model},
        )
    return Verdict(True, f"selected by {total} tensors", {"selected_tensors": per_model})


def check(
    macro: Macro,
    library: Library,
    table: OperatorTable,
    models: list[pathlib.Path],
    budget: engine.Budget,
    *,
    workdir: pathlib.Path | None = None,
    roundtrip_models: list[pathlib.Path] | None = None,
) -> Verdict:
    """Run every gate in order, stopping at the first refusal."""
    candidate = library.with_macro(macro)
    with _workdir(workdir) as directory:
        # Each gate is a thunk: a tuple of calls would run all four before the
        # loop could refuse, so a malformed body would still spawn processes
        # and an inert macro would still write an archive.
        gates = (
            lambda: structural(macro, table, library),
            lambda: engine_accepts(candidate, directory),
            lambda: fires(candidate, macro, models, budget, directory),
            lambda: roundtrip(
                candidate, roundtrip_models or models[:1], budget, directory
            ),
        )
        verdict = Verdict(True, "no gates ran")
        for gate in gates:
            verdict = gate()
            if not verdict.ok:
                return verdict
        return verdict


class _workdir:
    def __init__(self, given: pathlib.Path | None):
        self.given = given
        self.tmp = None

    def __enter__(self) -> pathlib.Path:
        if self.given is not None:
            self.given.mkdir(parents=True, exist_ok=True)
            return self.given
        self.tmp = tempfile.TemporaryDirectory(prefix="autodsl-")
        return pathlib.Path(self.tmp.name)

    def __exit__(self, *exc) -> None:
        if self.tmp is not None:
            self.tmp.cleanup()


def summarize(verdict: Verdict) -> str:
    return json.dumps({"ok": verdict.ok, "reason": verdict.reason, **verdict.detail})
