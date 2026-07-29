#!/usr/bin/env python3
"""Render a checkpoint compression case study as a static, print-ready figure.

Three panels over one checkpoint, laid out as the network itself is laid out —
one column per transformer layer, one row per tensor role:

  (a) how well each tensor compressed
  (b) which DSL program the search selected for it
  (c) where the bytes came from, split into what entropy coding alone achieves
      and what the structural operators add on top

(a) and (b) share a grid on purpose. Read together they show that the
compression ratio has real structure while the *choice of program* mostly does
not: for weight matrices several programs are near-ties and a few bytes decide
the winner, whereas the normalisation tensors are consistently handled
differently. Colouring only by program would imply a structure that is not
there.

Emits SVG, and PDF/PNG when `rsvg-convert` is available. No plotting library:
the cluster's Python has numpy and nothing else, so the SVG is written out
directly.

    python3 tools/checkpoint_figure.py --search a.json --literal b.json \
        --title "TinyLlama-1.1B (BF16)" --out figures/case-study
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil
import subprocess

# ==================== design tokens ====================

SANS = "Liberation Sans, Helvetica, Arial, sans-serif"
MONO = "Liberation Mono, DejaVu Sans Mono, monospace"

INK = "#1a1a1a"          # primary text
INK_SOFT = "#5c5c5c"     # secondary text
RULE = "#d8d8d8"         # hairlines
PAPER = "#ffffff"

# Sequential ramp for compression ratio: light warm grey to deep teal. Chosen
# to stay legible in greyscale, which is how half of print readers will see it.
RATIO_RAMP = ["#f3f2ef", "#dbe6e4", "#b9d3d1", "#8fbcbb", "#5f9ea3",
              "#3d7f8a", "#265f6d", "#16414f"]

# Categorical palette for the root operator. Okabe-Ito derived, reordered so
# the two largest categories are the most separable.
PROGRAM_COLOURS = {
    "add_const_mod": "#0072b2",
    "rotate_bits":   "#e69f00",
    "zigzag":        "#009e73",
    "gray":          "#cc79a7",
    "xor_const":     "#56b4e9",
    "bit_reverse":   "#d55e00",
    "diff_mod":      "#8c8c8c",
    "xor_prev":      "#7a5195",
}
PROGRAM_FALLBACK = "#bdbdbd"

# Role rows, in the order a reader thinks about a transformer block.
ROLE_ORDER = [
    ("self_attn.q_proj", "Q proj"),
    ("self_attn.k_proj", "K proj"),
    ("self_attn.v_proj", "V proj"),
    ("self_attn.o_proj", "O proj"),
    ("mlp.gate_proj", "MLP gate"),
    ("mlp.up_proj", "MLP up"),
    ("mlp.down_proj", "MLP down"),
    ("input_layernorm", "Norm (in)"),
    ("post_attention_layernorm", "Norm (post-attn)"),
]

LAYER_RE = re.compile(r"(?:model\.)?layers?\.(\d+)\.(.+?)(?:\.weight)?$")

# Long, repetitive role names carry little information past the part that
# distinguishes them.
ROLE_SHORTHAND = [
    ("self_attn.q_proj", "Q proj"), ("self_attn.k_proj", "K proj"),
    ("self_attn.v_proj", "V proj"), ("self_attn.o_proj", "O proj"),
    ("attention.attention.query", "Q proj"), ("attention.attention.key", "K proj"),
    ("attention.attention.value", "V proj"), ("attention.output.dense", "O proj"),
    ("mlp.gate_proj", "MLP gate"), ("mlp.up_proj", "MLP up"),
    ("mlp.down_proj", "MLP down"),
    ("intermediate.dense", "MLP in"), ("output.dense", "MLP out"),
    ("input_layernorm", "Norm (in)"),
    ("post_attention_layernorm", "Norm (post-attn)"),
    ("layernorm_before", "Norm (pre)"), ("layernorm_after", "Norm (post)"),
]


def pretty_role(role: str) -> str:
    suffix = ""
    for tail, mark in ((".weight_scale", " · scale"), (".bias", " · bias"),
                       (".weight", "")):
        if role.endswith(tail):
            role, suffix = role[: -len(tail)], mark
            break
    for pattern, label in ROLE_SHORTHAND:
        if role == pattern:
            return label + suffix
    return role.replace("_", " ") + suffix


# ==================== data ====================


def load(search: pathlib.Path, literal: pathlib.Path) -> dict:
    """Join the two runs by tensor name and index them by (layer, role)."""
    run = json.loads(search.read_text())
    lit = {t["name"]: t["encoded_bytes_without_frame_headers"]
           for t in json.loads(literal.read_text())["tensors"]}

    grid, outside, layers = {}, [], set()
    for tensor in run["tensors"]:
        if tensor["raw_bytes"] == 0:
            continue
        entry = {
            "name": tensor["name"],
            "raw": tensor["raw_bytes"],
            "encoded": tensor["encoded_bytes_without_frame_headers"],
            "literal": lit.get(tensor["name"], tensor["raw_bytes"]),
            "program": tensor["program"],
            "root": tensor["root_operator"],
            "dtype": tensor["dtype"],
            "shape": tensor["shape"],
        }
        entry["ratio"] = entry["raw"] / max(entry["encoded"], 1)
        match = LAYER_RE.search(tensor["name"])
        if match:
            layer, role = int(match.group(1)), match.group(2)
            layers.add(layer)
            grid[(layer, role)] = entry
        else:
            outside.append(entry)

    roles = [(key, label) for key, label in ROLE_ORDER
             if any(k[1] == key for k in grid)]
    seen = {key for key, _ in roles}
    for (_, role) in sorted(grid):
        if role not in seen:
            seen.add(role)
            roles.append((role, pretty_role(role)))

    return {
        "run": run, "grid": grid, "outside": outside,
        "layers": sorted(layers), "roles": roles,
        "archive": run["projected_archive_bytes"],
        "raw": run["raw_bytes"],
        "literal_total": sum(lit.values()),
    }


def role_breakdown(data: dict) -> list[dict]:
    """Per role: raw bytes, what literal coding alone gets, what structure adds."""
    out = []
    for key, label in data["roles"]:
        cells = [v for (_, role), v in data["grid"].items() if role == key]
        if not cells:
            continue
        raw = sum(c["raw"] for c in cells)
        out.append({
            "label": label,
            "raw": raw,
            "literal_share": (raw - sum(c["literal"] for c in cells)) / raw,
            "structure_share": (sum(c["literal"] for c in cells)
                                - sum(c["encoded"] for c in cells)) / raw,
        })
    return out


# ==================== svg helpers ====================


def esc(text: str) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def text(x, y, s, *, size=8.5, fill=INK, anchor="start", family=SANS,
         weight="normal", spacing=None, opacity=None) -> str:
    extra = f' letter-spacing="{spacing}"' if spacing else ""
    extra += f' opacity="{opacity}"' if opacity else ""
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-family="{family}" '
            f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}"{extra}>{esc(s)}</text>')


def rect(x, y, w, h, fill, *, stroke="none", rx=0.0, sw=0.5, opacity=None) -> str:
    extra = f' opacity="{opacity}"' if opacity is not None else ""
    return (f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'rx="{rx}"{extra}/>')


def quantile_edges(values: list[float], bins: int) -> list[float]:
    """Bin boundaries that put an equal count in each bin.

    A linear ramp is useless here: in a transformer nearly every weight matrix
    lands within a few percent of the same ratio while two normalisation rows
    stretch the range by 2x, so a linear scale paints the entire weight stack
    one colour. Quantile classification spends the palette where the tensors
    actually are, and the legend prints the real boundaries so nothing is
    hidden by the choice.
    """
    ordered = sorted(values)
    return [ordered[min(len(ordered) - 1, round(i * len(ordered) / bins))]
            for i in range(1, bins)]


def ramp_colour(value: float, edges: list[float]) -> str:
    index = 0
    while index < len(edges) and value >= edges[index]:
        index += 1
    return RATIO_RAMP[min(index, len(RATIO_RAMP) - 1)]


def si(nbytes: int) -> str:
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if nbytes >= scale:
            return f"{nbytes / scale:.1f} {unit}"
    return f"{nbytes} B"


# ==================== panels ====================

GAP = 1.6
# Sized from the longest label the checkpoint actually has: a quantised model
# carries roles like `mlp.down_proj.weight_scale`, which at a fixed width runs
# left out of its own panel and into the neighbouring grid.
MIN_label_w, MAX_label_w = 52.0, 96.0
# 7 inches, the full text width of a two-column AAAI page. Cells are sized to
# fit; type stays at its designed size, because a figure that is legible only
# before LaTeX scales it is not legible.
DEFAULT_WIDTH_PT = 504.0


def label_width(data: dict) -> float:
    longest = max((len(label) for _, label in data["roles"]), default=8)
    return min(MAX_label_w, max(MIN_label_w, longest * 3.55 + 9))


def grid_panel(data: dict, x0: float, y0: float, *, mode: str, cell: float,
               label_w: float, caption: str, subtitle: str) -> tuple[str, float, float]:
    """One layer x role heat grid. `mode` is 'ratio' or 'program'."""
    parts = [text(x0, y0, caption, size=9.0, weight="bold"),
             text(x0, y0 + 10.5, subtitle, size=7.0, fill=INK_SOFT)]
    top = y0 + 24
    step = cell + GAP
    edges = quantile_edges([c["ratio"] for c in data["grid"].values()],
                           len(RATIO_RAMP))

    for row, (key, label) in enumerate(data["roles"]):
        cy = top + row * step
        parts.append(text(x0 + label_w - 5, cy + cell - (cell - 5.6) / 2,
                          label, size=7.0, anchor="end", fill=INK_SOFT))
        for col, layer in enumerate(data["layers"]):
            entry = data["grid"].get((layer, key))
            cx = x0 + label_w + col * step
            if entry is None:
                parts.append(rect(cx, cy, cell, cell, "none", stroke=RULE, sw=0.4))
                continue
            fill = (ramp_colour(entry["ratio"], edges) if mode == "ratio"
                    else PROGRAM_COLOURS.get(entry["root"], PROGRAM_FALLBACK))
            parts.append(rect(cx, cy, cell, cell, fill, rx=min(1.4, cell / 6)))

    bottom = top + len(data["roles"]) * step
    ticks = [c for c, layer in enumerate(data["layers"]) if layer % 4 == 0]
    last = len(data["layers"]) - 1
    if last - ticks[-1] >= 2:
        ticks.append(last)
    for col in ticks:
        cx = x0 + label_w + col * step + cell / 2
        parts.append(text(cx, bottom + 7.5, str(data["layers"][col]), size=6.8,
                          anchor="middle", fill=INK_SOFT))
    width = label_w + len(data["layers"]) * step
    parts.append(text(x0 + label_w, bottom + 16.5, "transformer layer",
                      size=6.9, fill=INK_SOFT, opacity="0.85"))
    return "\n".join(parts), width, bottom + 26


def ratio_legend(data: dict, x0: float, y0: float, width: float) -> str:
    ratios = [c["ratio"] for c in data["grid"].values()]
    edges = quantile_edges(ratios, len(RATIO_RAMP))
    swatch = width / len(RATIO_RAMP)
    parts = [text(x0, y0, "compression ratio  (equal-count bins)",
                  size=7.0, fill=INK_SOFT)]
    for i, colour in enumerate(RATIO_RAMP):
        parts.append(rect(x0 + i * swatch, y0 + 5, swatch, 6.5, colour))
    digits = 2
    while digits < 4 and len({f"{e:.{digits}f}" for e in edges}) < len(edges):
        digits += 1
    parts.append(text(x0, y0 + 20, f"{min(ratios):.{digits}f}×", size=6.5,
                      fill=INK_SOFT))
    for i, edge in enumerate(edges):
        if i % 2 == 1:
            parts.append(text(x0 + (i + 1) * swatch, y0 + 20,
                              f"{edge:.{digits}f}", size=6.5, anchor="middle",
                              fill=INK_SOFT))
    parts.append(text(x0 + width, y0 + 20, f"{max(ratios):.{digits}f}×",
                      size=6.5, anchor="end", fill=INK_SOFT))
    return "\n".join(parts)


def ratio_distribution(data: dict, x0: float, y0: float, width: float) -> str:
    """Every tensor as a tick on a linear ratio axis.

    The equal-count bins in (a) deliberately hide how uneven the underlying
    distribution is, so this shows it: almost every tensor sits on one narrow
    spike and the normalisation tensors trail off alone. Without it a reader
    could take the eight colours in (a) to mean eight comparable groups.
    """
    ratios = sorted(c["ratio"] for c in data["grid"].values())
    lo, hi = ratios[0], ratios[-1]
    height = 16.0
    parts = [text(x0, y0, "distribution of those ratios, one tick per tensor",
                  size=6.9, fill=INK_SOFT),
             f'<line x1="{x0:.2f}" y1="{y0 + 6 + height:.2f}" '
             f'x2="{x0 + width:.2f}" y2="{y0 + 6 + height:.2f}" '
             f'stroke="{RULE}" stroke-width="0.6"/>']
    for ratio in ratios:
        cx = x0 + width * (ratio - lo) / max(hi - lo, 1e-9)
        parts.append(f'<line x1="{cx:.2f}" y1="{y0 + 6:.2f}" x2="{cx:.2f}" '
                     f'y2="{y0 + 6 + height:.2f}" stroke="#265f6d" '
                     f'stroke-width="0.7" opacity="0.30"/>')
    parts.append(text(x0, y0 + height + 14, f"{lo:.2f}×", size=6.5, fill=INK_SOFT))
    parts.append(text(x0 + width, y0 + height + 14, f"{hi:.2f}×", size=6.5,
                      anchor="end", fill=INK_SOFT))
    weights = sum(1 for r in ratios if r < 1.55)
    parts.append(text(x0 + width / 2, y0 + height + 14,
                      f"{weights} of {len(ratios)} below 1.55×", size=6.5,
                      anchor="middle", fill=INK_SOFT))
    return "\n".join(parts)


def program_legend(data: dict, x0: float, y0: float, width: float) -> str:
    counts: dict[str, int] = {}
    for cell in data["grid"].values():
        counts[cell["root"]] = counts.get(cell["root"], 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])
    parts = [text(x0, y0, "root operator of the selected program",
                  size=7.0, fill=INK_SOFT)]
    for i, (root, n) in enumerate(ordered[:8]):
        cy = y0 + 8 + i * 9.4
        parts.append(rect(x0, cy, 6.4, 6.4,
                          PROGRAM_COLOURS.get(root, PROGRAM_FALLBACK), rx=1.1))
        parts.append(text(x0 + 9.5, cy + 5.6, f"{root}", size=6.8,
                          fill=INK_SOFT, family=MONO))
        parts.append(text(x0 + width, cy + 5.6, str(n), size=6.8,
                          fill=INK_SOFT, family=MONO, anchor="end"))
    return "\n".join(parts)


def decomposition_panel(data: dict, x0: float, y0: float, width: float) -> tuple[str, float]:
    """Per role, the share of raw bytes removed by literal coding vs structure."""
    rows = role_breakdown(data)
    parts = [text(x0, y0, "(c)  Where the bytes go, by tensor role",
                  size=9.5, weight="bold"),
             text(x0, y0 + 11,
                  "share of raw bytes removed — entropy coding alone, "
                  "then what the structural operators add on top",
                  size=7.6, fill=INK_SOFT)]
    label_w, bar_h, gap = 96.0, 9.5, 4.6
    gutter_total, gutter_struct, gutter_size = 46.0, 92.0, 62.0
    bar_w = width - label_w - (gutter_total + gutter_struct + gutter_size)
    top = y0 + 26
    scale = max(max(r["literal_share"], 0.0) + max(r["structure_share"], 0.0)
                for r in rows) * 1.02

    # Reference grid every 10%, behind the bars.
    tick = 0.10
    while tick < scale:
        gx = x0 + label_w + bar_w * tick / scale
        parts.append(f'<line x1="{gx:.2f}" y1="{top - 4:.2f}" x2="{gx:.2f}" '
                     f'y2="{top + len(rows) * (bar_h + gap) - gap + 2:.2f}" '
                     f'stroke="{RULE}" stroke-width="0.5"/>')
        parts.append(text(gx, top - 7, f"{tick * 100:.0f}%", size=6.4,
                          anchor="middle", fill=INK_SOFT))
        tick += 0.10

    for i, row in enumerate(rows):
        cy = top + i * (bar_h + gap)
        parts.append(text(x0 + label_w - 6, cy + bar_h - 2.0, row["label"],
                          size=7.4, anchor="end", fill=INK_SOFT))
        # On a 32-bit dtype the entropy coders are illegal, so literal-only
        # comes out *larger* than raw and the share goes slightly negative.
        # Clamp it rather than emitting a negative-width rectangle; the legend
        # below says what a zero-width light segment means.
        w1 = max(0.0, bar_w * row["literal_share"] / scale)
        w2 = max(0.0, bar_w * row["structure_share"] / scale)
        parts.append(rect(x0 + label_w, cy, w1, bar_h, "#b9c9cf"))
        parts.append(rect(x0 + label_w + w1, cy, w2, bar_h, "#16414f"))
        total = row["literal_share"] + row["structure_share"]
        cursor = x0 + label_w + bar_w
        parts.append(text(cursor + gutter_total - 6, cy + bar_h - 2.0,
                          f"{100 * total:.1f}%", size=7.3, family=MONO,
                          anchor="end"))
        cursor += gutter_total
        parts.append(text(cursor + gutter_struct - 6, cy + bar_h - 2.0,
                          f"+{100 * row['structure_share']:.1f}% from structure",
                          size=6.9, family=MONO, fill=INK_SOFT, anchor="end"))
        cursor += gutter_struct
        parts.append(text(cursor + gutter_size - 6, cy + bar_h - 2.0,
                          si(row["raw"]), size=6.9, anchor="end", fill=INK_SOFT))

    bottom = top + len(rows) * (bar_h + gap) + 6
    parts.append(rect(x0 + label_w, bottom, 8, 6.5, "#b9c9cf", rx=1.2))
    starved = all(r["literal_share"] <= 0 for r in rows)
    parts.append(text(x0 + label_w + 12, bottom + 5.6,
                      "entropy coding only (all structural operators disabled)"
                      + ("  —  zero here: rANS and Huffman are legal only up "
                         "to 16-bit elements" if starved else ""),
                      size=6.9, fill=INK_SOFT))
    parts.append(rect(x0 + label_w + 258, bottom, 8, 6.5, "#16414f", rx=1.2))
    parts.append(text(x0 + label_w + 270, bottom + 5.6,
                      "added by the synthesised program", size=6.9, fill=INK_SOFT))
    return "\n".join(parts), bottom + 14


# ==================== figure ====================


def render(data: dict, title: str, width: float) -> str:
    margin = 18.0
    columns = len(data["layers"])
    label_w = label_width(data)
    cell = max(3.2, ((width - margin * 2 - 26) / 2 - label_w) / columns - GAP)
    step = cell + GAP
    grid_w = label_w + columns * step

    head = [
        text(margin, margin + 4, title, size=12.5, weight="bold"),
        text(margin, margin + 16,
             f"{len(data['grid']) + len(data['outside'])} tensors, "
             f"{si(data['raw'])} raw → {si(data['archive'])} archived "
             f"({data['raw'] / data['archive']:.3f}×). "
             f"One column per transformer layer, one row per tensor role.",
             size=8.0, fill=INK_SOFT),
        f'<line x1="{margin}" y1="{margin + 24}" x2="{width - margin}" '
        f'y2="{margin + 24}" stroke="{RULE}" stroke-width="0.8"/>',
    ]

    top = margin + 34
    left, w_left, y_left = grid_panel(
        data, margin, top, mode="ratio", cell=cell, label_w=label_w,
        caption="(a)  Compression achieved",
        subtitle="darker is a smaller archive for that tensor")
    right, w_right, y_right = grid_panel(
        data, margin + grid_w + 26, top, mode="program", cell=cell,
        label_w=label_w,
        caption="(b)  Program the search selected",
        subtitle="colour is the program's outermost operator")

    legends = "\n".join([
        ratio_legend(data, margin + label_w, y_left, w_left - label_w),
        ratio_distribution(data, margin + label_w, y_left + 32,
                           w_left - label_w),
        program_legend(data, margin + grid_w + 26 + label_w, y_right,
                       w_right - label_w),
    ])

    y_after = max(y_left, y_right) + 12 + 9.4 * 6 + 14
    body, y_end = decomposition_panel(data, margin, y_after, width - margin * 2)

    outside = sorted(data["outside"], key=lambda t: -t["raw"])[:3]
    note_y = y_end + 12
    notes = [f'<line x1="{margin}" y1="{note_y - 8}" x2="{width - margin}" '
             f'y2="{note_y - 8}" stroke="{RULE}" stroke-width="0.8"/>']
    notes.append(text(margin, note_y + 2, "Outside the layer stack:",
                      size=7.0, fill=INK_SOFT))
    span = (width - margin * 2 - 104) / max(len(outside), 1)
    for i, tensor in enumerate(outside):
        notes.append(text(margin + 104 + i * span, note_y + 2,
                          f"{tensor['name'].split('.')[-2] if '.' in tensor['name'] else tensor['name']}"
                          f"  {si(tensor['raw'])}  {tensor['ratio']:.2f}×",
                          size=6.9, family=MONO, fill=INK_SOFT))
    height = note_y + 16

    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
            f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">\n'
            f'{rect(0, 0, width, height, PAPER)}\n'
            + "\n".join(head) + "\n" + left + "\n" + right + "\n"
            + legends + "\n" + body + "\n" + "\n".join(notes) + "\n</svg>\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--search", type=pathlib.Path, required=True,
                        help="bench --format json with the full operator set")
    parser.add_argument("--literal", type=pathlib.Path, required=True,
                        help="bench --format json with every transform disabled")
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True,
                        help="path without extension")
    parser.add_argument("--width", type=float, default=DEFAULT_WIDTH_PT,
                        help="figure width in points; 504 is a full-width "
                             "two-column AAAI figure")
    args = parser.parse_args(argv)

    data = load(args.search, args.literal)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    svg = args.out.with_suffix(".svg")
    svg.write_text(render(data, args.title, args.width))
    print(f"wrote {svg}")

    converter = shutil.which("rsvg-convert")
    if converter:
        for fmt, scale in (("pdf", None), ("png", "3")):
            target = args.out.with_suffix(f".{fmt}")
            cmd = [converter, "-f", fmt, "-o", str(target)]
            if scale:
                cmd += ["-z", scale]
            subprocess.run(cmd + [str(svg)], check=True)
            print(f"wrote {target}")
    else:
        print("rsvg-convert not found; SVG only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
