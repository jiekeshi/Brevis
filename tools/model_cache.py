#!/usr/bin/env python3
"""Fetch, verify, and drop evaluation checkpoints one model at a time.

The tiered matrix is 134 GB. On a bounded quota it is cheaper to hold one model
at a time than to stage the whole corpus, so the intended loop is:

    fetch <tag> -> run the harness -> keep the report -> drop <tag>

Every path, byte count, and digest comes from eval/models-tiered.json. This
tool never writes to the manifest and never touches results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "eval" / "models-tiered.json"
CACHE = ROOT / "eval" / "cache"
CHUNK = 1 << 22


def models(manifest: pathlib.Path) -> dict[str, dict]:
    return {m["tag"]: m for m in json.loads(manifest.read_text())}


def paths(model: dict, cache: pathlib.Path) -> list[tuple[pathlib.Path, dict]]:
    base = cache / model["repo"].replace("/", "__") / model["revision"]
    return [(base / f["file"], f) for f in model["files"]]


def digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def check(path: pathlib.Path, spec: dict) -> str:
    if not path.exists():
        return "missing"
    if path.stat().st_size != spec["bytes"]:
        return "wrong-size"
    return "ok" if digest(path) == spec["sha256"] else "wrong-sha256"


def cmd_status(args, index) -> int:
    cache = args.cache_root
    total = 0
    for tag, model in index.items():
        states = [check(p, s) if args.verify else
                  ("ok" if p.exists() and p.stat().st_size == s["bytes"] else
                   "missing" if not p.exists() else "wrong-size")
                  for p, s in paths(model, cache)]
        held = sum(p.stat().st_size for p, _ in paths(model, cache) if p.exists())
        total += held
        state = "complete" if all(s == "ok" for s in states) else \
                "absent" if all(s == "missing" for s in states) else "partial"
        print(f"{tag:38s} {state:9s} {held / 1e9:8.3f} GB held / "
              f"{model['selected_bytes'] / 1e9:8.3f} GB total")
    print(f"{'':38s} {'':9s} {total / 1e9:8.3f} GB cached")
    usage = shutil.disk_usage(cache if cache.exists() else ROOT)
    print(f"filesystem free: {usage.free / 1e12:.1f} TB "
          f"(quota is separate; check with diskusage_report)")
    return 0


def cmd_fetch(args, index) -> int:
    model = index[args.tag]
    for path, spec in paths(model, args.cache_root):
        if check(path, spec) == "ok":
            print(f"have {path.name}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        url = (f"https://huggingface.co/{model['repo']}/resolve/"
               f"{model['revision']}/{spec['file']}")
        part = path.with_suffix(path.suffix + ".part")
        print(f"fetching {spec['file']} ({spec['bytes'] / 1e9:.3f} GB)")
        try:
            subprocess.run(["curl", "-fL", "--progress-bar", "-o", str(part), url],
                           check=True)
            part.replace(path)
        finally:
            part.unlink(missing_ok=True)
        state = check(path, spec)
        if state != "ok":
            print(f"integrity failure on {spec['file']}: {state}", file=sys.stderr)
            return 1
    print(f"{args.tag}: complete and verified")
    return 0


def cmd_verify(args, index) -> int:
    bad = 0
    for tag in ([args.tag] if args.tag else index):
        for path, spec in paths(index[tag], args.cache_root):
            state = check(path, spec)
            bad += state not in ("ok", "missing")
            if state != "missing" or args.tag:
                print(f"{'OK ' if state == 'ok' else 'BAD'} {tag:38s} "
                      f"{spec['file']} {state}")
    return 1 if bad else 0


def cmd_drop(args, index) -> int:
    model = index[args.tag]
    freed = 0
    for path, _ in paths(model, args.cache_root):
        if path.exists():
            freed += path.stat().st_size
            path.unlink()
    for parent in (args.cache_root / model["repo"].replace("/", "__") /
                   model["revision"], args.cache_root /
                   model["repo"].replace("/", "__")):
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    print(f"{args.tag}: freed {freed / 1e9:.3f} GB")
    print("keep the harness report; the checkpoint is re-fetchable from the "
          "pinned revision at any time")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=pathlib.Path, default=MANIFEST)
    parser.add_argument("--cache-root", type=pathlib.Path, default=CACHE)
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="what is cached, and how much space it uses")
    status.add_argument("--verify", action="store_true",
                        help="hash every cached file instead of checking size only")
    status.set_defaults(func=cmd_status)

    fetch = sub.add_parser("fetch", help="download one model and verify it")
    fetch.add_argument("tag")
    fetch.set_defaults(func=cmd_fetch)

    verify = sub.add_parser("verify", help="re-hash cached files against the manifest")
    verify.add_argument("tag", nargs="?")
    verify.set_defaults(func=cmd_verify)

    drop = sub.add_parser("drop", help="delete one model's cached files")
    drop.add_argument("tag")
    drop.set_defaults(func=cmd_drop)

    args = parser.parse_args(argv)
    index = models(args.manifest)
    if getattr(args, "tag", None) and args.tag not in index:
        parser.error(f"unknown tag {args.tag!r}; known: {', '.join(index)}")
    return args.func(args, index)


if __name__ == "__main__":
    raise SystemExit(main())
