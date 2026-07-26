"""An append-only record of every macro that was proposed and what happened.

The loop's output is a library, but its *evidence* is this file. Without it a
library is a set of assertions; with it, every macro can be traced to the
measurement that admitted it, and every rejection is available so the proposer
is not asked to rediscover the same dead end.

Entries are never rewritten. A macro that is later dropped gets a new entry
saying so.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
from typing import Iterator


@dataclasses.dataclass
class Ledger:
    path: pathlib.Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, kind: str, **fields) -> dict:
        entry = {"kind": kind, "sequence": self.count(), **fields}
        with self.path.open("a") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def entries(self) -> Iterator[dict]:
        if not self.path.exists():
            return iter(())
        return (
            json.loads(line)
            for line in self.path.read_text().splitlines()
            if line.strip()
        )

    def count(self) -> int:
        return sum(1 for _ in self.entries())

    def rejected(self, evaluator: str | None = None) -> list[dict]:
        """What to tell the proposer not to try again, newest last.

        A rejection is only binding while the evaluator that produced it still
        stands. Three candidates in this repository's own history were refused
        for "never fired" under a macro-pricing bug; carrying those forward
        would keep a whole family of shapes permanently off the table for a
        reason that no longer exists. Pass the current evaluator version to
        drop anything judged under a different one.
        """
        out: list[dict] = []
        seen: set[str] = set()
        for entry in self.entries():
            if entry.get("kind") != "rejected":
                continue
            if evaluator is not None and entry.get("evaluator") != evaluator:
                continue
            shape = entry.get("shape", "")
            if shape in seen:
                continue
            seen.add(shape)
            out.append({"shape": shape, "reason": entry.get("reason", "")})
        return out

    def stale(self, evaluator: str) -> list[dict]:
        """Decisions taken under a different evaluator, so no longer evidence."""
        return [
            {"kind": e["kind"], "name": e.get("name", ""), "shape": e.get("shape", ""),
             "evaluator": e.get("evaluator", "unversioned"),
             "reason": e.get("reason", "")}
            for e in self.entries()
            if e.get("kind") in ("accepted", "rejected")
            and e.get("evaluator") != evaluator
        ]

    def accepted_names(self) -> list[str]:
        live: list[str] = []
        for entry in self.entries():
            if entry.get("kind") == "accepted":
                live.append(entry["name"])
            elif entry.get("kind") == "dropped" and entry.get("name") in live:
                live.remove(entry["name"])
        return live

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for entry in self.entries():
            counts[entry["kind"]] = counts.get(entry["kind"], 0) + 1
        return counts
