#!/usr/bin/env python3
"""End-to-end eval: real model tensors, four baselines, bit-exact round-trips."""
import filecmp, json, os, pathlib, shutil, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parent.parent
BREVIS = ROOT / "zig-out" / "bin" / "brevis"
CACHE = pathlib.Path(os.environ.get("BREVIS_EVAL_CACHE", ROOT / "eval" / "cache"))
RESULTS = ROOT / "eval" / "results.json"


def sh(*cmd, **kw):
    return subprocess.run([str(c) for c in cmd], check=True, **kw)


def fetch(repo, revision, fname):
    dst = CACHE / repo.replace("/", "__") / revision / fname
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{fname}"
    print(f"  downloading {url}")
    sh("curl", "-fL", "--progress-bar", "-o", str(dst) + ".part", url)
    os.replace(str(dst) + ".part", dst)
    return dst


def timed(*cmd):
    t = time.time()
    sh(*cmd, stdout=subprocess.DEVNULL)
    return time.time() - t


def baseline(path, tool, cargs, dargs, out):
    if not shutil.which(tool):
        return None, None, None, None
    t = time.time()
    with open(out, "wb") as f:
        subprocess.run([tool, *cargs, "-c", str(path)], check=True, stdout=f)
    t_cmp = time.time() - t
    n = out.stat().st_size
    rec = pathlib.Path(str(out) + ".rec")
    t = time.time()
    with open(rec, "wb") as f:
        subprocess.run([tool, *dargs, "-c", str(out)], check=True, stdout=f)
    t_dec = time.time() - t
    exact = filecmp.cmp(path, rec, shallow=False)
    rec.unlink()
    out.unlink()
    return n, t_cmp, t_dec, exact


def openzl(path, out):
    zli = os.environ.get("OPENZL_ZLI") or shutil.which("zli")
    if not zli:
        return None, None, None, None
    t = time.time()
    sh(zli, "compress", "--profile", "serial", "-f", path, "-o", out,
       stdout=subprocess.DEVNULL)
    t_cmp = time.time() - t
    n = out.stat().st_size
    rec = pathlib.Path(str(out) + ".rec")
    t = time.time()
    sh(zli, "decompress", "-f", out, "-o", rec, stdout=subprocess.DEVNULL)
    t_dec = time.time() - t
    exact = filecmp.cmp(path, rec, shallow=False)
    rec.unlink()
    out.unlink()
    return n, t_cmp, t_dec, exact


def main():
    if not BREVIS.exists():
        sys.exit(f"build first: cd {ROOT} && zig build -Doptimize=ReleaseFast")

    models = json.loads((ROOT / "eval" / "models.json").read_text())
    work = CACHE / "work"
    work.mkdir(parents=True, exist_ok=True)
    previous = {}
    if os.environ.get("BREVIS_REUSE_BASELINES") == "1":
        old_path = RESULTS if RESULTS.exists() else work / "results.json"
        if old_path.exists():
            previous = {row["tag"]: row for row in json.loads(old_path.read_text())}
    rows = []

    for m in models:
        print(f"\n=== {m['tag']}: {m['repo']} — {m['note']}")
        src = fetch(m["repo"], m["revision"], m["file"])
        raw = src.stat().st_size

        sh(sys.executable, ROOT / "eval" / "tensor_stats.py", src)

        prior = work / f"{m['tag']}.prior"
        uniform_brv = work / f"{m['tag']}.uniform.brv"
        brv = work / f"{m['tag']}.brv"
        rec = work / f"{m['tag']}.rec.safetensors"

        t_uniform = timed(BREVIS, "compress", src, uniform_brv)
        sh(BREVIS, "verify", uniform_brv, src)
        n_uniform = uniform_brv.stat().st_size

        t_cal = timed(BREVIS, "calibrate", src, prior)
        t_cmp = timed(BREVIS, "compress", src, brv, "--prior", prior)
        sh(BREVIS, "verify", brv, src)
        t_dec_jobs1 = timed(BREVIS, "decompress", brv, rec, "--jobs", 1)
        exact_jobs1 = subprocess.run(["cmp", "-s", str(src), str(rec)]).returncode == 0
        t_dec = timed(BREVIS, "decompress", brv, rec)
        exact = subprocess.run(["cmp", "-s", str(src), str(rec)]).returncode == 0
        rec.unlink(missing_ok=True)

        n_brv = brv.stat().st_size
        row = {"tag": m["tag"], "repo": m["repo"], "revision": m["revision"], "raw": raw, "brevis": n_brv,
               "uniform": n_uniform, "t_uniform_cmp": t_uniform, "uniform_verified": True,
               "t_cal": t_cal, "t_cmp": t_cmp, "t_dec_jobs1": t_dec_jobs1, "t_dec": t_dec,
               "bitexact_jobs1": exact_jobs1, "bitexact": exact,
               "openzl_profile": "serial"}

        old = previous.get(m["tag"])
        if old and old.get("raw") == raw and old.get("revision") == m["revision"]:
            for key in ("gzip", "zstd", "xz", "openzl"):
                for field in (key, f"t_{key}", f"t_{key}_dec", f"{key}_exact"):
                    row[field] = old[field]
        else:
            for tool, cargs, dargs, key in (
                ("gzip", ["-9"], ["-d"], "gzip"),
                ("zstd", ["-19", "-T0", "-q"], ["-d", "-q"], "zstd"),
                ("xz", ["-9"], ["-d"], "xz"),
            ):
                n, tc, td, ex = baseline(src, tool, cargs, dargs, work / f"{m['tag']}.{key}")
                row[key], row[f"t_{key}"], row[f"t_{key}_dec"], row[f"{key}_exact"] = n, tc, td, ex

            n, tc, td, ex = openzl(src, work / f"{m['tag']}.zl")
            row["openzl"], row["t_openzl"], row["t_openzl_dec"], row["openzl_exact"] = n, tc, td, ex

        rows.append(row)
        print(f"  brevis {n_brv:,} ({raw/n_brv:.3f}x)  bit-exact={exact}")

    print("\n" + "=" * 116)
    print(f"{'dtype':6} {'raw':>14} {'brevis':>14} {'ratio':>7} {'uniform':>8} {'gzip-9':>7} "
          f"{'zstd-19':>8} {'xz-9':>7} {'openzl':>7} {'cal s':>7} {'cmp s':>7} {'dec s':>7} {'exact':>6}")
    print("=" * 116)
    for r in rows:
        def rat(k):
            return f"{r['raw']/r[k]:.3f}" if r.get(k) else "-"
        print(f"{r['tag']:6} {r['raw']:>14,} {r['brevis']:>14,} {r['raw']/r['brevis']:>7.3f} "
              f"{rat('uniform'):>8} {rat('gzip'):>7} {rat('zstd'):>8} {rat('xz'):>7} {rat('openzl'):>7} "
              f"{r['t_cal']:>7.1f} {r['t_cmp']:>7.1f} {r['t_dec']:>7.1f} {str(r['bitexact']):>6}")

    RESULTS.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {RESULTS}")
    if not all(r["bitexact"] and r["bitexact_jobs1"] and
               all(r.get(f"{k}_exact") is not False for k in ("gzip", "zstd", "xz", "openzl")) for r in rows):
        sys.exit("FAIL: not bit-exact")


if __name__ == "__main__":
    main()
