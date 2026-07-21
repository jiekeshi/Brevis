#!/usr/bin/env python3
"""End-to-end eval over complete single-file or sharded models."""

import argparse
import filecmp
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
BREVIS = pathlib.Path(os.environ.get("BREVIS_BIN", ROOT / "zig-out" / "bin" / "brevis"))
CACHE = pathlib.Path(os.environ.get("BREVIS_EVAL_CACHE", ROOT / "eval" / "cache"))
RESULTS = ROOT / "eval" / "results.json"
GENERIC_BASELINES = (
    ("gzip", ["-9"], ["-d"], "gzip"),
    ("zstd", ["-19", "-T0", "-q"], ["-d", "-q"], "zstd"),
    ("xz", ["-9"], ["-d"], "xz"),
)
BASELINE_KEYS = ("gzip", "zstd", "xz", "openzl")


class EvalError(RuntimeError):
    pass


def sh(*cmd, **kw):
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(f"{path}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def run_fingerprint():
    digest = hashlib.sha256()
    digest.update(BREVIS.read_bytes())
    digest.update(pathlib.Path(__file__).read_bytes())
    return digest.hexdigest()


def fetch(repo, revision, fname, url=None):
    dst = CACHE / repo.replace("/", "__") / revision / fname
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = url or f"https://huggingface.co/{repo}/resolve/{revision}/{fname}"
    part = pathlib.Path(f"{dst}.part")
    print(f"  downloading {url}")
    try:
        sh("curl", "-fL", "--progress-bar", "-o", part, url)
        os.replace(part, dst)
    finally:
        part.unlink(missing_ok=True)
    return dst


def timed(*cmd):
    started = time.time()
    sh(*cmd, stdout=subprocess.DEVNULL)
    return time.time() - started


def stream_matches(path, command):
    proc = subprocess.Popen([str(c) for c in command], stdout=subprocess.PIPE)
    exact = True
    try:
        with path.open("rb") as expected:
            while chunk := expected.read(1 << 20):
                if proc.stdout.read(len(chunk)) != chunk:
                    exact = False
                    break
            if exact and proc.stdout.read(1):
                exact = False
        if not exact:
            proc.kill()
        code = proc.wait()
        if exact and code != 0:
            raise subprocess.CalledProcessError(code, command)
        return exact
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()


def baseline(path, tool, cargs, dargs, out):
    if not shutil.which(tool):
        return None, None, None, None
    try:
        started = time.time()
        with out.open("wb") as compressed:
            subprocess.run([tool, *cargs, "-c", str(path)], check=True, stdout=compressed)
        t_cmp = time.time() - started
        size = out.stat().st_size
        started = time.time()
        exact = stream_matches(path, [tool, *dargs, "-c", out])
        return size, t_cmp, time.time() - started, exact
    finally:
        out.unlink(missing_ok=True)


def openzl(path, out):
    zli = os.environ.get("OPENZL_ZLI") or shutil.which("zli")
    if not zli:
        return None, None, None, None
    try:
        started = time.time()
        sh(zli, "compress", "--profile", "serial", "-f", path, "-o", out,
           stdout=subprocess.DEVNULL)
        t_cmp = time.time() - started
        size = out.stat().st_size
        started = time.time()
        exact = stream_matches(path, [zli, "decompress", "-f", out, "-o", "/dev/stdout"])
        return size, t_cmp, time.time() - started, exact
    finally:
        out.unlink(missing_ok=True)


def model_files(model):
    entries = model.get("files")
    if entries is None:
        entries = [model["file"]]
    if not entries:
        raise EvalError(f"{model['tag']}: no safetensors files configured")
    files = []
    for entry in entries:
        if isinstance(entry, str):
            files.append((entry, None))
        else:
            files.append((entry["file"], entry.get("url")))
    if len({name for name, _ in files}) != len(files):
        raise EvalError(f"{model['tag']}: duplicate safetensors file")
    return files


def require_equal(source, restored, label):
    if not filecmp.cmp(source, restored, shallow=False):
        raise EvalError(f"{label}: restored bytes differ")


def evaluate_shard(model, src, fname, index, work, run_baselines=True):
    prefix = f"{model['tag'].replace('/', '_')}.{index:05d}"
    uniform_brv = work / f"{prefix}.uniform.brv"
    prior = work / f"{prefix}.prior"
    brv = work / f"{prefix}.brv"
    restored = work / f"{prefix}.rec.safetensors"
    artifacts = (uniform_brv, prior, brv, restored)
    for path in artifacts:
        path.unlink(missing_ok=True)

    try:
        raw = src.stat().st_size
        t_uniform = timed(BREVIS, "compress", src, uniform_brv)
        sh(BREVIS, "verify", uniform_brv, src)
        uniform_size = uniform_brv.stat().st_size
        uniform_brv.unlink()

        t_cal = timed(BREVIS, "calibrate", src, prior)
        t_cmp = timed(BREVIS, "compress", src, brv, "--prior", prior)
        prior.unlink()
        sh(BREVIS, "verify", brv, src)
        brevis_size = brv.stat().st_size

        t_dec_jobs1 = timed(BREVIS, "decompress", brv, restored, "--jobs", 1)
        require_equal(src, restored, f"{model['tag']}:{fname}:jobs=1")
        restored.unlink()
        t_dec = timed(BREVIS, "decompress", brv, restored)
        require_equal(src, restored, f"{model['tag']}:{fname}:parallel")
        restored.unlink()
        brv.unlink()

        row = {
            "file": fname,
            "raw": raw,
            "brevis": brevis_size,
            "uniform": uniform_size,
            "t_uniform_cmp": t_uniform,
            "uniform_verified": True,
            "t_cal": t_cal,
            "t_cmp": t_cmp,
            "t_dec_jobs1": t_dec_jobs1,
            "t_dec": t_dec,
            "bitexact_jobs1": True,
            "bitexact": True,
        }
        if run_baselines:
            for tool, cargs, dargs, key in GENERIC_BASELINES:
                values = baseline(src, tool, cargs, dargs, work / f"{prefix}.{key}")
                row[key], row[f"t_{key}"], row[f"t_{key}_dec"], row[f"{key}_exact"] = values
                if values[3] is False:
                    raise EvalError(f"{model['tag']}:{fname}:{key}: restored bytes differ")
            values = openzl(src, work / f"{prefix}.openzl")
            row["openzl"], row["t_openzl"], row["t_openzl_dec"], row["openzl_exact"] = values
            if values[3] is False:
                raise EvalError(f"{model['tag']}:{fname}:openzl: restored bytes differ")
        return row
    finally:
        for path in artifacts:
            path.unlink(missing_ok=True)


def sum_field(shards, field):
    values = [shard.get(field) for shard in shards]
    return sum(values) if all(value is not None for value in values) else None


def all_field(shards, field):
    values = [shard.get(field) for shard in shards]
    return all(values) if all(value is not None for value in values) else None


def evaluate_model(model, work, previous=None, resumed=None, checkpoint=None, fingerprint=None):
    files = model_files(model)
    names = [name for name, _ in files]
    reuse = (
        previous is not None
        and previous.get("repo") == model["repo"]
        and previous.get("revision") == model["revision"]
        and previous.get("files") == names
    )
    old_shards = {shard["file"]: shard for shard in previous.get("shards", [])} if reuse else {}
    resume_matches = resumed is not None and (resumed.get("repo"), resumed.get("revision")) == (
        model["repo"], model["revision"])
    if fingerprint is not None:
        resume_matches = resume_matches and resumed.get("fingerprint") == fingerprint
    if resume_matches:
        completed = {shard["file"]: shard for shard in resumed["shards"]}
    else:
        completed = {}
    shards = []
    for index, (fname, url) in enumerate(files, 1):
        if fname in completed:
            print(f"  [{index}/{len(files)}] {fname} (checkpoint)")
            shards.append(completed[fname])
            continue
        print(f"  [{index}/{len(files)}] {fname}")
        src = fetch(model["repo"], model["revision"], fname, url)
        sh(sys.executable, ROOT / "eval" / "tensor_stats.py", src)
        shard = evaluate_shard(model, src, fname, index, work, not reuse)
        if fname in old_shards:
            for key in BASELINE_KEYS:
                for field in (key, f"t_{key}", f"t_{key}_dec", f"{key}_exact"):
                    shard[field] = old_shards[fname].get(field)
        shards.append(shard)
        if checkpoint is not None:
            checkpoint(shards)

    row = {
        "tag": model["tag"],
        "repo": model["repo"],
        "revision": model["revision"],
        "files": names,
        "raw": sum_field(shards, "raw"),
        "brevis": sum_field(shards, "brevis"),
        "uniform": sum_field(shards, "uniform"),
        "t_uniform_cmp": sum_field(shards, "t_uniform_cmp"),
        "uniform_verified": all_field(shards, "uniform_verified"),
        "t_cal": sum_field(shards, "t_cal"),
        "t_cmp": sum_field(shards, "t_cmp"),
        "t_dec_jobs1": sum_field(shards, "t_dec_jobs1"),
        "t_dec": sum_field(shards, "t_dec"),
        "bitexact_jobs1": all_field(shards, "bitexact_jobs1"),
        "bitexact": all_field(shards, "bitexact"),
        "shards": shards,
    }
    for key in BASELINE_KEYS:
        fields = (key, f"t_{key}", f"t_{key}_dec", f"{key}_exact")
        if reuse:
            for field in fields:
                row[field] = previous.get(field)
        else:
            row[key] = sum_field(shards, key)
            row[f"t_{key}"] = sum_field(shards, f"t_{key}")
            row[f"t_{key}_dec"] = sum_field(shards, f"t_{key}_dec")
            row[f"{key}_exact"] = all_field(shards, f"{key}_exact")
    if reuse:
        row["baselines_reused"] = True
    return row


def print_summary(rows):
    print("\n" + "=" * 119)
    print(f"{'model':12} {'shards':>6} {'raw':>14} {'brevis':>14} {'ratio':>7} {'uniform':>8} "
          f"{'gzip-9':>7} {'zstd-19':>8} {'xz-9':>7} {'openzl':>7} {'cal s':>7} {'cmp s':>7} {'dec s':>7}")
    print("=" * 119)
    for row in rows:
        def ratio(key):
            return f"{row['raw'] / row[key]:.3f}" if row.get(key) else "-"

        print(f"{row['tag']:12} {len(row['files']):>6} {row['raw']:>14,} {row['brevis']:>14,} "
              f"{row['raw'] / row['brevis']:>7.3f} {ratio('uniform'):>8} {ratio('gzip'):>7} "
              f"{ratio('zstd'):>8} {ratio('xz'):>7} {ratio('openzl'):>7} "
              f"{row['t_cal']:>7.1f} {row['t_cmp']:>7.1f} {row['t_dec']:>7.1f}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=pathlib.Path, default=ROOT / "eval" / "models.json")
    parser.add_argument("--results", type=pathlib.Path, default=RESULTS)
    parser.add_argument("--tag", action="append", help="run only this model tag; repeatable")
    args = parser.parse_args(argv)
    if not BREVIS.exists():
        parser.error(f"build first: cd {ROOT} && zig build -Doptimize=ReleaseFast")
    fingerprint = run_fingerprint()

    models = json.loads(args.models.read_text())
    if args.tag:
        wanted = set(args.tag)
        models = [model for model in models if model["tag"] in wanted]
        missing = wanted - {model["tag"] for model in models}
        if missing:
            parser.error(f"unknown tag(s): {', '.join(sorted(missing))}")

    work = CACHE / "work"
    work.mkdir(parents=True, exist_ok=True)
    previous = {}
    if os.environ.get("BREVIS_REUSE_BASELINES") == "1" and args.results.exists():
        previous = {row["tag"]: row for row in json.loads(args.results.read_text())}
    checkpoint_path = pathlib.Path(f"{args.results}.checkpoint")
    checkpoints = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else []
    checkpoints = {(row["repo"], row["revision"], row.get("fingerprint")): row for row in checkpoints}

    rows = []
    for model in models:
        print(f"\n=== {model['tag']}: {model['repo']} — {model['note']}")
        identity = model["repo"], model["revision"]
        for stale in [key for key in checkpoints if key[:2] == identity and key[2] != fingerprint]:
            del checkpoints[stale]
        key = (*identity, fingerprint)

        def save_checkpoint(shards):
            checkpoints[key] = {
                "repo": model["repo"],
                "revision": model["revision"],
                "fingerprint": fingerprint,
                "shards": shards,
            }
            write_json(checkpoint_path, list(checkpoints.values()))

        row = evaluate_model(
            model, work, previous.get(model["tag"]), checkpoints.get(key), save_checkpoint, fingerprint
        )
        rows.append(row)
        write_json(args.results, rows)
        print(f"  model total: {row['brevis']:,} bytes ({row['raw'] / row['brevis']:.3f}x), bit-exact")

    for model in models:
        checkpoints.pop((model["repo"], model["revision"], fingerprint), None)
    if checkpoints:
        write_json(checkpoint_path, list(checkpoints.values()))
    else:
        checkpoint_path.unlink(missing_ok=True)
    print_summary(rows)
    print(f"\nwrote {args.results}")


if __name__ == "__main__":
    main()
