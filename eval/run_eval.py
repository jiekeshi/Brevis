#!/usr/bin/env python3
"""End-to-end eval over complete single-file or sharded models."""

import argparse
import datetime as dt
import filecmp
import hashlib
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
BREVIS = ROOT / "zig-out" / "bin" / "brevis"
CACHE = pathlib.Path(os.environ.get("BREVIS_EVAL_CACHE", ROOT / "eval" / "cache"))
RESULTS = ROOT / "eval" / "results.json"
GENERIC_BASELINES = (
    ("gzip", ["-9"], ["-d"], "gzip"),
    ("zstd", ["-19", "-T0", "-q"], ["-d", "-q"], "zstd"),
    ("xz", ["-9"], ["-d"], "xz"),
)
BASELINE_KEYS = ("gzip", "zstd", "xz", "openzl")
RESULT_SCHEMA = 3
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class EvalError(RuntimeError):
    pass


def baseline_fields(key):
    return key, f"t_{key}", f"t_{key}_dec", f"{key}_exact"


def sh(*cmd, **kw):
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pathlib.Path(f"{path}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def sha256_path(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as source:
        while chunk := source.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def named_digest(records, field):
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["file"].encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(record[field]))
    return digest.hexdigest()


def model_manifest_sha256(model):
    """Hash the complete model entry, including expected file identities.

    A filename-only digest would allow a changed expected size or SHA-256 to
    reuse an incompatible checkpoint or baseline result.
    """

    encoded = json.dumps(model, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def binary_config():
    return json.loads(subprocess.check_output((BREVIS, "config"), text=True))


def tool_version(tool):
    executable = os.environ.get("OPENZL_ZLI") if tool == "zli" else None
    executable = executable or shutil.which(tool)
    if not executable:
        return None
    result = subprocess.run((executable, "--version"), text=True, capture_output=True)
    lines = (result.stdout or result.stderr).splitlines()
    return lines[0] if lines else None


def git_output(*args):
    return subprocess.check_output(("git", "-C", ROOT, *args), text=True).strip()


def provenance(jobs, calibration_tensors):
    search = binary_config()
    search["calibration_tensors"] = calibration_tensors
    return {
        "result_schema": RESULT_SCHEMA,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_dirty": bool(git_output("status", "--porcelain")),
        "binary_sha256": sha256_path(BREVIS),
        "eval_sha256": sha256_path(__file__),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "zig_version": subprocess.check_output(("zig", "version"), text=True).strip(),
        "threads": {
            "calibration": jobs,
            "compression": jobs,
            "parallel_decompression": jobs,
            "serial_decompression": 1,
        },
        "search": search,
        "modes": {
            "fixed": {"plan": "fixed", "prior": None, "fallback": "raw"},
            "uniform": {"plan": "search", "prior": None},
            "phog": {"plan": "search", "prior": "shard-local"},
        },
        "baselines": {**{
            key: {
                "tool": tool,
                "version": tool_version(tool),
                "compress_args": cargs,
                "decompress_args": dargs,
            }
            for tool, cargs, dargs, key in GENERIC_BASELINES
        }, "openzl": {
            "tool": "zli",
            "version": tool_version("zli"),
            "compress_args": ["--profile", "serial"],
        }},
    }


def run_fingerprint(metadata):
    return hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()


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
    started = time.perf_counter()
    sh(*cmd, stdout=subprocess.DEVNULL)
    return time.perf_counter() - started


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


def normalized_model_files(model):
    entries = model.get("files")
    if entries is None:
        if "file" not in model:
            raise EvalError(f"{model.get('tag', '<unknown>')}: no safetensors files configured")
        entries = [model["file"]]
    if not entries:
        raise EvalError(f"{model['tag']}: no safetensors files configured")
    files = []
    for entry in entries:
        if isinstance(entry, str):
            normalized = {"file": entry, "url": None, "bytes": None, "sha256": None}
        elif isinstance(entry, dict):
            normalized = {
                "file": entry.get("file"),
                "url": entry.get("url"),
                "bytes": entry.get("bytes"),
                "sha256": entry.get("sha256"),
            }
        else:
            raise EvalError(f"{model['tag']}: file entry must be a string or object")
        if not isinstance(normalized["file"], str) or not normalized["file"]:
            raise EvalError(f"{model['tag']}: invalid safetensors filename")
        if normalized["bytes"] is not None and (
            not isinstance(normalized["bytes"], int) or normalized["bytes"] <= 0
        ):
            raise EvalError(f"{model['tag']}:{normalized['file']}: invalid expected byte size")
        if normalized["sha256"] is not None and (
            not isinstance(normalized["sha256"], str) or not HEX64.fullmatch(normalized["sha256"])
        ):
            raise EvalError(f"{model['tag']}:{normalized['file']}: invalid expected SHA-256")
        files.append(normalized)
    if len({entry["file"] for entry in files}) != len(files):
        raise EvalError(f"{model['tag']}: duplicate safetensors file")
    return files


def model_files(model):
    """Return the legacy ``(filename, URL)`` view used by callers and tests."""

    return [(entry["file"], entry["url"]) for entry in normalized_model_files(model)]


def validate_models(models):
    if not isinstance(models, list) or not models:
        raise EvalError("model manifest must be a nonempty JSON array")
    tags = set()
    for model in models:
        if not isinstance(model, dict):
            raise EvalError("every model manifest entry must be an object")
        tag = model.get("tag")
        if not isinstance(tag, str) or not tag:
            raise EvalError("every model manifest entry needs a nonempty tag")
        if tag in tags:
            raise EvalError(f"duplicate model tag: {tag}")
        tags.add(tag)
        if not isinstance(model.get("repo"), str) or not model["repo"]:
            raise EvalError(f"{tag}: missing repository")
        revision = model.get("revision")
        if not isinstance(revision, str) or not HEX40.fullmatch(revision):
            raise EvalError(f"{tag}: revision must be a 40-character lowercase Git SHA")
        files = normalized_model_files(model)
        selected_bytes = model.get("selected_bytes")
        if selected_bytes is not None:
            known_sizes = [entry["bytes"] for entry in files]
            if any(size is None for size in known_sizes) or sum(known_sizes) != selected_bytes:
                raise EvalError(f"{tag}: selected_bytes does not match the file-size sum")


def validate_local_file(model, entry, path):
    actual_size = path.stat().st_size
    expected_size = entry["bytes"]
    if expected_size is not None and actual_size != expected_size:
        raise EvalError(
            f"{model['tag']}:{entry['file']}: expected {expected_size} bytes, found {actual_size}"
        )
    actual_sha256 = sha256_path(path)
    expected_sha256 = entry["sha256"]
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise EvalError(
            f"{model['tag']}:{entry['file']}: SHA-256 mismatch; "
            f"expected {expected_sha256}, found {actual_sha256}"
        )
    return {
        "expected_size_bytes": expected_size,
        "observed_size_bytes": actual_size,
        "expected_sha256": expected_sha256,
        "observed_sha256": actual_sha256,
        "verified": (expected_size is None or expected_size == actual_size)
        and (expected_sha256 is None or expected_sha256 == actual_sha256),
    }


def require_equal(source, restored, label):
    if not filecmp.cmp(source, restored, shallow=False):
        raise EvalError(f"{label}: restored bytes differ")


def evaluate_shard(
    model, src, fname, index, work, run_baselines=True, jobs=None, calibration_tensors=None,
    integrity=None,
):
    prefix = f"{model['tag'].replace('/', '_')}.{index:05d}"
    fixed_brv = work / f"{prefix}.fixed.brv"
    uniform_brv = work / f"{prefix}.uniform.brv"
    prior = work / f"{prefix}.prior"
    brv = work / f"{prefix}.brv"
    restored = work / f"{prefix}.rec.safetensors"
    artifacts = (fixed_brv, uniform_brv, prior, brv, restored)
    for path in artifacts:
        path.unlink(missing_ok=True)

    try:
        raw = src.stat().st_size
        input_sha256 = integrity["observed_sha256"] if integrity is not None else sha256_path(src)
        job_args = () if jobs is None else ("--jobs", jobs)
        tensor_args = () if calibration_tensors is None else ("--tensors", calibration_tensors)

        t_fixed = timed(BREVIS, "compress", src, fixed_brv, "--plan", "fixed", *job_args)
        sh(BREVIS, "verify", fixed_brv, src)
        fixed_size = fixed_brv.stat().st_size
        fixed_brv.unlink()

        t_uniform = timed(BREVIS, "compress", src, uniform_brv, "--plan", "search", *job_args)
        sh(BREVIS, "verify", uniform_brv, src)
        uniform_size = uniform_brv.stat().st_size
        uniform_brv.unlink()

        t_cal = timed(BREVIS, "calibrate", src, prior, *tensor_args, *job_args)
        prior_sha256 = sha256_path(prior)
        t_cmp = timed(BREVIS, "compress", src, brv, "--plan", "search", "--prior", prior, *job_args)
        prior.unlink()
        sh(BREVIS, "verify", brv, src)
        brevis_size = brv.stat().st_size

        t_dec = timed(BREVIS, "decompress", brv, restored, *job_args)
        require_equal(src, restored, f"{model['tag']}:{fname}:parallel")
        restored.unlink()
        t_dec_jobs1 = timed(BREVIS, "decompress", brv, restored, "--jobs", 1)
        require_equal(src, restored, f"{model['tag']}:{fname}:jobs=1")
        restored.unlink()
        brv.unlink()

        row = {
            "file": fname,
            "input_sha256": input_sha256,
            "input_integrity": integrity,
            "prior_sha256": prior_sha256,
            "raw": raw,
            "fixed": fixed_size,
            "t_fixed_cmp": t_fixed,
            "fixed_verified": True,
            "phog": brevis_size,
            "uniform": uniform_size,
            "t_uniform_cmp": t_uniform,
            "uniform_verified": True,
            "t_cal": t_cal,
            "t_phog_cmp": t_cmp,
            "t_phog_dec_jobs1": t_dec_jobs1,
            "t_phog_dec": t_dec,
            "phog_bitexact_jobs1": True,
            "phog_bitexact": True,
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


def evaluate_model(
    model, work, previous=None, resumed=None, checkpoint=None, fingerprint=None,
    jobs=None, calibration_tensors=None,
):
    file_entries = normalized_model_files(model)
    names = [entry["file"] for entry in file_entries]
    manifest_sha256 = model_manifest_sha256(model)
    reuse = (
        previous is not None
        and previous.get("manifest_sha256") == manifest_sha256
    )
    old_shards = {shard["file"]: shard for shard in previous.get("shards", [])} if reuse else {}
    resume_matches = resumed is not None and (
        resumed.get("repo"), resumed.get("revision"), resumed.get("manifest_sha256")
    ) == (model["repo"], model["revision"], manifest_sha256)
    if fingerprint is not None:
        resume_matches = resume_matches and resumed.get("fingerprint") == fingerprint
    if resume_matches:
        completed = {shard["file"]: shard for shard in resumed["shards"]}
    else:
        completed = {}
    shards = []
    for index, entry in enumerate(file_entries, 1):
        fname = entry["file"]
        url = entry["url"]
        if fname in completed:
            expected_sha256 = entry["sha256"]
            if expected_sha256 is not None and completed[fname].get("input_sha256") != expected_sha256:
                raise EvalError(f"{model['tag']}:{fname}: checkpoint input digest violates manifest")
            print(f"  [{index}/{len(file_entries)}] {fname} (checkpoint)")
            shards.append(completed[fname])
            continue
        print(f"  [{index}/{len(file_entries)}] {fname}")
        src = fetch(model["repo"], model["revision"], fname, url)
        integrity = validate_local_file(model, entry, src)
        sh(sys.executable, ROOT / "eval" / "tensor_stats.py", src)
        shard = evaluate_shard(
            model, src, fname, index, work, not reuse, jobs, calibration_tensors, integrity,
        )
        if fname in old_shards:
            for key in BASELINE_KEYS:
                for field in baseline_fields(key):
                    shard[field] = old_shards[fname].get(field)
        shards.append(shard)
        if checkpoint is not None:
            checkpoint(shards)

    row = {
        "tag": model["tag"],
        "repo": model["repo"],
        "revision": model["revision"],
        "files": names,
        "manifest_sha256": manifest_sha256,
        "model_sha256": named_digest(shards, "input_sha256"),
        "raw": sum_field(shards, "raw"),
        "fixed": sum_field(shards, "fixed"),
        "t_fixed_cmp": sum_field(shards, "t_fixed_cmp"),
        "fixed_verified": all_field(shards, "fixed_verified"),
        "phog": sum_field(shards, "phog"),
        "uniform": sum_field(shards, "uniform"),
        "t_uniform_cmp": sum_field(shards, "t_uniform_cmp"),
        "uniform_verified": all_field(shards, "uniform_verified"),
        "t_cal": sum_field(shards, "t_cal"),
        "t_phog_cmp": sum_field(shards, "t_phog_cmp"),
        "t_phog_dec_jobs1": sum_field(shards, "t_phog_dec_jobs1"),
        "t_phog_dec": sum_field(shards, "t_phog_dec"),
        "phog_bitexact_jobs1": all_field(shards, "phog_bitexact_jobs1"),
        "phog_bitexact": all_field(shards, "phog_bitexact"),
        "shards": shards,
    }
    for key in BASELINE_KEYS:
        fields = baseline_fields(key)
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
    print("\n" + "=" * 128)
    print(f"{'model':12} {'shards':>6} {'raw':>14} {'phog':>14} {'ratio':>7} {'fixed':>8} {'uniform':>8} "
          f"{'gzip-9':>7} {'zstd-19':>8} {'xz-9':>7} {'openzl':>7} {'cal s':>7} {'cmp s':>7} {'dec s':>7}")
    print("=" * 128)
    for row in rows:
        def ratio(key):
            return f"{row['raw'] / row[key]:.3f}" if row.get(key) else "-"

        print(f"{row['tag']:12} {len(row['files']):>6} {row['raw']:>14,} {row['phog']:>14,} "
              f"{row['raw'] / row['phog']:>7.3f} {ratio('fixed'):>8} {ratio('uniform'):>8} {ratio('gzip'):>7} "
              f"{ratio('zstd'):>8} {ratio('xz'):>7} {ratio('openzl'):>7} "
              f"{row['t_cal']:>7.1f} {row['t_phog_cmp']:>7.1f} {row['t_phog_dec']:>7.1f}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=pathlib.Path, default=ROOT / "eval" / "models.json")
    parser.add_argument("--results", type=pathlib.Path, default=RESULTS)
    parser.add_argument("--tag", action="append", help="run only this model tag; repeatable")
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--tensors", type=int, default=200,
                        help="maximum tensors used to train each input-local prior")
    invocation_args = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(invocation_args)
    sh("zig", "build", "-Doptimize=ReleaseFast", cwd=ROOT)
    if not BREVIS.exists():
        parser.error(f"build first: cd {ROOT} && zig build -Doptimize=ReleaseFast")
    run = provenance(args.jobs, args.tensors)
    run["command"] = [sys.executable, str(pathlib.Path(__file__).resolve()), *invocation_args]
    run["model_manifest"] = {
        "path": str(args.models.resolve()),
        "sha256": sha256_path(args.models),
    }

    models = json.loads(args.models.read_text())
    validate_models(models)
    if args.tag:
        wanted = set(args.tag)
        models = [model for model in models if model["tag"] in wanted]
        missing = wanted - {model["tag"] for model in models}
        if missing:
            parser.error(f"unknown tag(s): {', '.join(sorted(missing))}")

    work = CACHE / "work"
    work.mkdir(parents=True, exist_ok=True)
    previous = {}
    reuse_baselines = os.environ.get("BREVIS_REUSE_BASELINES") == "1" and args.results.exists()
    run["reuse_baselines"] = reuse_baselines
    if reuse_baselines:
        previous_document = json.loads(args.results.read_text())
        if isinstance(previous_document, dict) and previous_document.get("schema") == RESULT_SCHEMA:
            previous = {row["tag"]: row for row in previous_document["models"]}
            run["baseline_reuse"] = {"results_sha256": sha256_path(args.results)}
    fingerprint = run_fingerprint(run)
    run["fingerprint"] = fingerprint
    run["started_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
    checkpoint_path = pathlib.Path(f"{args.results}.checkpoint")
    checkpoints = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else []
    checkpoints = {
        (row["repo"], row["revision"], row.get("manifest_sha256"), row.get("fingerprint")): row
        for row in checkpoints
    }

    rows = []
    for model in models:
        print(f"\n=== {model['tag']}: {model['repo']} — {model.get('note', '')}")
        manifest_sha256 = model_manifest_sha256(model)
        identity = model["repo"], model["revision"], manifest_sha256
        for stale in [key for key in checkpoints if key[:3] == identity and key[3] != fingerprint]:
            del checkpoints[stale]
        key = (*identity, fingerprint)

        def save_checkpoint(shards):
            checkpoints[key] = {
                "repo": model["repo"],
                "revision": model["revision"],
                "manifest_sha256": manifest_sha256,
                "fingerprint": fingerprint,
                "shards": shards,
            }
            write_json(checkpoint_path, list(checkpoints.values()))

        row = evaluate_model(
            model, work, previous.get(model["tag"]), checkpoints.get(key), save_checkpoint, fingerprint,
            args.jobs, args.tensors,
        )
        rows.append(row)
        write_json(args.results, {
            "schema": RESULT_SCHEMA,
            "created_at_utc": run["started_at_utc"],
            "provenance": run,
            "models": rows,
        })
        print(f"  model total: {row['phog']:,} bytes ({row['raw'] / row['phog']:.3f}x), bit-exact")

    for model in models:
        manifest_sha256 = model_manifest_sha256(model)
        checkpoints.pop((model["repo"], model["revision"], manifest_sha256, fingerprint), None)
    if checkpoints:
        write_json(checkpoint_path, list(checkpoints.values()))
    else:
        checkpoint_path.unlink(missing_ok=True)
    print_summary(rows)
    print(f"\nwrote {args.results}")


if __name__ == "__main__":
    main()
