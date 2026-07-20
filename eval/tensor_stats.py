#!/usr/bin/env python3
"""Report safetensors dtype inventory, entropy, and FP32 upcast evidence."""
import json, math, struct, sys
from collections import Counter, defaultdict

import numpy as np


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return hdr, 8 + n


DT = {"F16": np.uint16, "BF16": np.uint16, "F32": np.uint32,
      "I8": np.uint8, "U8": np.uint8, "I32": np.uint32, "I64": np.uint64}


def load(path, hdr, base, name, cap):
    info = hdr[name]
    lo, hi = info["data_offsets"]
    hi = min(hi, lo + cap)
    with open(path, "rb") as f:
        f.seek(base + lo)
        raw = f.read(hi - lo)
    dt = DT.get(info["dtype"])
    if dt is None:
        return None
    return np.frombuffer(raw[: len(raw) // dt().itemsize * dt().itemsize], dtype=dt)


def h0(a, bits):
    """Order-0 entropy in bits/symbol over `bits`-wide symbols."""
    v = a.astype(np.uint32) & ((1 << bits) - 1) if bits < 32 else a.astype(np.uint32)
    _, c = np.unique(v, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def main(path, cap=8 << 20):
    hdr, base = read_header(path)
    names = [k for k in hdr if k != "__metadata__"]

    inv = Counter()
    nbytes = defaultdict(int)
    for k in names:
        d = hdr[k]["dtype"]
        lo, hi = hdr[k]["data_offsets"]
        inv[d] += 1
        nbytes[d] += hi - lo
    total = sum(nbytes.values())

    print(f"{path}")
    print(f"  tensors={len(names)}  bytes={total:,}")
    for d, n in inv.most_common():
        print(f"    {d:5} {n:5} tensors  {nbytes[d]:>14,} B  ({100*nbytes[d]/total:5.1f}%)")

    # sample the largest tensor of each dtype
    by_dt = defaultdict(list)
    for k in names:
        by_dt[hdr[k]["dtype"]].append(k)

    print("  entropy (largest tensor per dtype, first %d MiB):" % (cap >> 20))
    for d, ks in by_dt.items():
        if d not in DT:
            continue
        k = max(ks, key=lambda x: hdr[x]["data_offsets"][1] - hdr[x]["data_offsets"][0])
        a = load(path, hdr, base, k, cap)
        if a is None or a.size == 0:
            continue
        bits = {"I8": 8, "U8": 8}.get(d, 16 if d in ("F16", "BF16") else 32)
        if d == "F32":
            w = a
            low16 = np.uint32(0xFFFF)
            frac_zero = float((w & low16 == 0).mean())
            tz = np.zeros(24)
            nz = w[w != 0]
            for b in range(24):
                tz[b] = float((nz & ((1 << b) - 1) == 0).mean()) if nz.size else 0.0
            print(f"    {d:5} {k[:44]:44} H0(hi16)={h0(w >> 16, 16):5.2f}b")
            print(f"          low16-zero={100*frac_zero:6.2f}%   "
                  f"tz@13={100*tz[13]:5.2f}%  tz@16={100*tz[16]:5.2f}%  "
                  f"(true-fp32 baseline @16 = {100*2**-16:.4f}%)")
            verdict = ("UPCAST (bf16 stored as fp32)" if frac_zero > 0.99 else
                       "MIXED / partial precision" if frac_zero > 0.01 else "TRUE FP32")
            print(f"          => {verdict}")
        else:
            e = h0(a, bits)
            print(f"    {d:5} {k[:44]:44} H0={e:5.2f}b / {bits}b  -> ceiling {bits/e:4.2f}x")
            if bits == 8:
                d1 = np.diff(a.astype(np.int16), prepend=0).astype(np.uint8)
                x1 = (a ^ np.roll(a, 1)).astype(np.uint8)
                print(f"          H0(diff)={h0(d1,8):5.2f}b  H0(xor_prev)={h0(x1,8):5.2f}b")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        main(p)
        print()
