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

    def rejected(self) -> list[dict]:
        """What to tell the proposer not to try again, newest last."""
        out: list[dict] = []
        seen: set[str] = set()
        for entry in self.entries():
            if entry.get("kind") != "rejected":
                continue
            shape = entry.get("shape", "")
            if shape in seen:
                continue
            seen.add(shape)
            out.append({"shape": shape, "reason": entry.get("reason", "")})
        return out

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
