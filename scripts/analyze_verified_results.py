#!/usr/bin/env python3
"""Produce paper-oriented statistics from a verified-ratios JSON summary.

This script is deliberately downstream of ``summarize_verified_ratios.py``:
it never reads checkpoint payloads, invokes a codec, or selects benchmark runs.
Unknown checkpoints fail closed so that domain and numeric-format groupings are
never silently inferred from a checkpoint name.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

BREVIS = "brevis"

# Scientific grouping choices are explicit and emitted in the JSON report.
# "image-generation" groups text-to-image model weights, not image datasets.
CHECKPOINT_METADATA = {
    "bert-fp32": ("language", "FP32"),
    "whisper-large-v3-f16": ("audio", "FP16"),
    "sdxl-base-1.0-f16": ("image-generation", "FP16"),
    "llama-3.1-8b-bf16": ("language", "BF16"),
    "ministral-3-8b-base-2512-bf16": ("language", "BF16"),
    "qwen3-32b-fp8": ("language", "FP8"),
    "qwen3-32b-bf16": ("language", "BF16"),
    "llama-3.1-70b-bf16": ("language", "BF16"),
    "mixtral-8x22b-v0.1-bf16": ("language", "BF16"),
    "glm-5.2-bf16": ("language", "BF16"),
    "voxtral-mini-3b-2507-bf16": ("audio", "BF16"),
    "qwen-image-bf16": ("image-generation", "BF16"),
}

PAIRWISE_COLUMNS = (
    "checkpoint",
    "domain",
    "numeric_format",
    "baseline",
    "brevis_exactness",
    "baseline_exactness",
    "source_bytes",
    "brevis_output_bytes",
    "baseline_output_bytes",
    "brevis_compression_ratio",
    "baseline_compression_ratio",
    "compression_advantage",
    "relative_archive_saving_fraction",
    "relative_archive_saving_percent",
    "archive_bytes_saved",
    "winner",
)

AGGREGATE_COLUMNS = (
    "group",
    "baseline",
    "models",
    "wins",
    "ties",
    "losses",
    "win_rate",
    "win_rate_ci_low",
    "win_rate_ci_high",
    "brevis_geomean_compression_ratio",
    "baseline_geomean_compression_ratio",
    "geomean_compression_advantage",
    "geomean_compression_advantage_ci_low",
    "geomean_compression_advantage_ci_high",
    "brevis_total_byte_compression_ratio",
    "baseline_total_byte_compression_ratio",
    "total_byte_compression_advantage",
    "total_byte_compression_advantage_ci_low",
    "total_byte_compression_advantage_ci_high",
    "macro_archive_saving_fraction",
    "macro_archive_saving_percent",
    "macro_archive_saving_fraction_ci_low",
    "macro_archive_saving_fraction_ci_high",
    "pooled_archive_saving_fraction",
    "pooled_archive_saving_percent",
    "pooled_archive_saving_fraction_ci_low",
    "pooled_archive_saving_fraction_ci_high",
    "total_archive_bytes_saved",
)

DEFAULT_BOOTSTRAP_SAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20_260_729


class AnalysisError(RuntimeError):
    """Raised when a verified summary is not safe to compare."""


@dataclass(frozen=True)
class Cell:
    checkpoint: str
    method: str
    exactness: str
    source_bytes: int
    output_bytes: int
    compression_ratio: float


@dataclass(frozen=True)
class PairwiseRow:
    checkpoint: str
    domain: str
    numeric_format: str
    baseline: str
    brevis_exactness: str
    baseline_exactness: str
    source_bytes: int
    brevis_output_bytes: int
    baseline_output_bytes: int
    brevis_compression_ratio: float
    baseline_compression_ratio: float
    compression_advantage: float
    relative_archive_saving_fraction: float
    relative_archive_saving_percent: float
    archive_bytes_saved: int
    winner: str


@dataclass(frozen=True)
class Aggregate:
    group: str
    baseline: str
    models: int
    wins: int
    ties: int
    losses: int
    win_rate: float
    win_rate_ci_low: float
    win_rate_ci_high: float
    brevis_geomean_compression_ratio: float
    baseline_geomean_compression_ratio: float
    geomean_compression_advantage: float
    geomean_compression_advantage_ci_low: float
    geomean_compression_advantage_ci_high: float
    brevis_total_byte_compression_ratio: float
    baseline_total_byte_compression_ratio: float
    total_byte_compression_advantage: float
    total_byte_compression_advantage_ci_low: float
    total_byte_compression_advantage_ci_high: float
    macro_archive_saving_fraction: float
    macro_archive_saving_percent: float
    macro_archive_saving_fraction_ci_low: float
    macro_archive_saving_fraction_ci_high: float
    pooled_archive_saving_fraction: float
    pooled_archive_saving_percent: float
    pooled_archive_saving_fraction_ci_low: float
    pooled_archive_saving_fraction_ci_high: float
    total_archive_bytes_saved: int


def geometric_mean(values: Iterable[float]) -> float:
    values = tuple(values)
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise AnalysisError("geometric mean requires finite positive values")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _positive_int(cell: dict[str, Any], key: str) -> int:
    value = cell.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise AnalysisError(
            f"{cell.get('checkpoint')}/{cell.get('method')}: "
            f"{key} must be a positive integer"
        )
    return value


def load_cells(path: Path) -> tuple[dict[tuple[str, str], Cell], list[str], dict[str, Any]]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read verified summary {path}: {exc}") from exc

    if payload.get("schema_version") != 1:
        raise AnalysisError("verified summary must use schema_version 1")
    if payload.get("all_complete") is not True:
        raise AnalysisError("verified summary is incomplete; refusing paper analysis")
    methods = payload.get("methods")
    raw_cells = payload.get("cells")
    if not (
        isinstance(methods, list)
        and all(isinstance(method, str) for method in methods)
        and BREVIS in methods
    ):
        raise AnalysisError("verified summary has an invalid methods list")
    if not isinstance(raw_cells, list) or not raw_cells:
        raise AnalysisError("verified summary has no cells")

    cells: dict[tuple[str, str], Cell] = {}
    for raw in raw_cells:
        if not isinstance(raw, dict):
            raise AnalysisError("verified summary contains a non-object cell")
        checkpoint = raw.get("checkpoint")
        method = raw.get("method")
        exactness = raw.get("exactness")
        if not all(isinstance(value, str) for value in (checkpoint, method, exactness)):
            raise AnalysisError("cell identity and exactness must be strings")
        if raw.get("status") != "complete":
            raise AnalysisError(f"{checkpoint}/{method}: cell is not complete")
        source_bytes = _positive_int(raw, "source_bytes")
        output_bytes = _positive_int(raw, "output_bytes")
        ratio = raw.get("compression_ratio")
        if not isinstance(ratio, (int, float)) or isinstance(ratio, bool):
            raise AnalysisError(f"{checkpoint}/{method}: invalid compression_ratio")
        expected_ratio = source_bytes / output_bytes
        if not math.isclose(float(ratio), expected_ratio, rel_tol=1e-12):
            raise AnalysisError(f"{checkpoint}/{method}: compression_ratio is inconsistent")
        identity = (checkpoint, method)
        if identity in cells:
            raise AnalysisError(f"duplicate cell: {checkpoint}/{method}")
        cells[identity] = Cell(
            checkpoint,
            method,
            exactness,
            source_bytes,
            output_bytes,
            expected_ratio,
        )

    checkpoints = list(dict.fromkeys(cell.checkpoint for cell in cells.values()))
    unknown = sorted(set(checkpoints) - set(CHECKPOINT_METADATA))
    if unknown:
        raise AnalysisError(
            "checkpoint taxonomy is missing explicit entries for: " + ", ".join(unknown)
        )
    expected = {(checkpoint, method) for checkpoint in checkpoints for method in methods}
    missing = sorted(expected - set(cells))
    unexpected = sorted(set(cells) - expected)
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing " + ", ".join(f"{c}/{m}" for c, m in missing))
        if unexpected:
            details.append("unexpected " + ", ".join(f"{c}/{m}" for c, m in unexpected))
        raise AnalysisError("; ".join(details))
    for checkpoint in checkpoints:
        sizes = {cells[checkpoint, method].source_bytes for method in methods}
        if len(sizes) != 1:
            raise AnalysisError(f"{checkpoint}: methods use different source sizes")

    provenance = {
        key: payload.get(key)
        for key in (
            "runs",
            "runs_sha256",
            "record_count",
            "preset",
            "complete_cells",
            "expected_cells",
            "exactness_policy",
        )
    }
    return cells, methods, provenance


def make_pairwise_rows(
    cells: dict[tuple[str, str], Cell],
    methods: list[str],
) -> list[PairwiseRow]:
    checkpoints = list(dict.fromkeys(checkpoint for checkpoint, _ in cells))
    rows = []
    for checkpoint in checkpoints:
        brevis = cells[checkpoint, BREVIS]
        domain, numeric_format = CHECKPOINT_METADATA[checkpoint]
        for baseline in methods:
            if baseline == BREVIS:
                continue
            other = cells[checkpoint, baseline]
            saved = other.output_bytes - brevis.output_bytes
            winner = BREVIS if saved > 0 else baseline if saved < 0 else "tie"
            saving = saved / other.output_bytes
            rows.append(
                PairwiseRow(
                    checkpoint=checkpoint,
                    domain=domain,
                    numeric_format=numeric_format,
                    baseline=baseline,
                    brevis_exactness=brevis.exactness,
                    baseline_exactness=other.exactness,
                    source_bytes=brevis.source_bytes,
                    brevis_output_bytes=brevis.output_bytes,
                    baseline_output_bytes=other.output_bytes,
                    brevis_compression_ratio=brevis.compression_ratio,
                    baseline_compression_ratio=other.compression_ratio,
                    compression_advantage=(
                        brevis.compression_ratio / other.compression_ratio
                    ),
                    relative_archive_saving_fraction=saving,
                    relative_archive_saving_percent=100.0 * saving,
                    archive_bytes_saved=saved,
                    winner=winner,
                )
            )
    return rows


def percentile(values: list[float], probability: float) -> float:
    """Return a linearly interpolated empirical percentile (R type 7)."""

    if not values or not 0.0 <= probability <= 1.0:
        raise AnalysisError("percentile requires values and a probability in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def point_estimates(rows: list[PairwiseRow]) -> dict[str, float]:
    source_bytes = sum(row.source_bytes for row in rows)
    brevis_outputs = sum(row.brevis_output_bytes for row in rows)
    baseline_outputs = sum(row.baseline_output_bytes for row in rows)
    savings = [row.relative_archive_saving_fraction for row in rows]
    return {
        "win_rate": sum(row.winner == BREVIS for row in rows) / len(rows),
        "brevis_geomean_compression_ratio": geometric_mean(
            row.brevis_compression_ratio for row in rows
        ),
        "baseline_geomean_compression_ratio": geometric_mean(
            row.baseline_compression_ratio for row in rows
        ),
        "geomean_compression_advantage": geometric_mean(
            row.compression_advantage for row in rows
        ),
        "brevis_total_byte_compression_ratio": source_bytes / brevis_outputs,
        "baseline_total_byte_compression_ratio": source_bytes / baseline_outputs,
        "total_byte_compression_advantage": baseline_outputs / brevis_outputs,
        "macro_archive_saving_fraction": math.fsum(savings) / len(savings),
        "pooled_archive_saving_fraction": (
            (baseline_outputs - brevis_outputs) / baseline_outputs
        ),
    }


def bootstrap_intervals(
    rows: list[PairwiseRow],
    *,
    samples: int,
    seed: int,
    group: str,
    baseline: str,
) -> dict[str, tuple[float, float]]:
    """Model-resampling percentile intervals for aggregate comparison metrics."""

    if samples <= 0:
        raise AnalysisError("bootstrap samples must be positive")
    seed_material = f"{seed}\0{group}\0{baseline}".encode()
    derived_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    generator = random.Random(derived_seed)
    metric_names = (
        "win_rate",
        "geomean_compression_advantage",
        "total_byte_compression_advantage",
        "macro_archive_saving_fraction",
        "pooled_archive_saving_fraction",
    )
    distributions = {name: [] for name in metric_names}
    for _ in range(samples):
        resample = [rows[generator.randrange(len(rows))] for _ in rows]
        estimates = point_estimates(resample)
        for name in metric_names:
            distributions[name].append(estimates[name])
    return {
        name: (percentile(values, 0.025), percentile(values, 0.975))
        for name, values in distributions.items()
    }


def aggregate(
    rows: list[PairwiseRow],
    group: str,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> Aggregate:
    if not rows:
        raise AnalysisError(f"cannot aggregate empty group {group!r}")
    baselines = {row.baseline for row in rows}
    if len(baselines) != 1:
        raise AnalysisError(f"group {group!r} mixes baselines")
    baseline = next(iter(baselines))
    brevis_outputs = sum(row.brevis_output_bytes for row in rows)
    baseline_outputs = sum(row.baseline_output_bytes for row in rows)
    estimates = point_estimates(rows)
    intervals = bootstrap_intervals(
        rows,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
        group=group,
        baseline=baseline,
    )
    return Aggregate(
        group=group,
        baseline=baseline,
        models=len(rows),
        wins=sum(row.winner == BREVIS for row in rows),
        ties=sum(row.winner == "tie" for row in rows),
        losses=sum(row.winner == baseline for row in rows),
        win_rate=estimates["win_rate"],
        win_rate_ci_low=intervals["win_rate"][0],
        win_rate_ci_high=intervals["win_rate"][1],
        brevis_geomean_compression_ratio=estimates[
            "brevis_geomean_compression_ratio"
        ],
        baseline_geomean_compression_ratio=estimates[
            "baseline_geomean_compression_ratio"
        ],
        geomean_compression_advantage=estimates[
            "geomean_compression_advantage"
        ],
        geomean_compression_advantage_ci_low=intervals[
            "geomean_compression_advantage"
        ][0],
        geomean_compression_advantage_ci_high=intervals[
            "geomean_compression_advantage"
        ][1],
        brevis_total_byte_compression_ratio=estimates[
            "brevis_total_byte_compression_ratio"
        ],
        baseline_total_byte_compression_ratio=estimates[
            "baseline_total_byte_compression_ratio"
        ],
        total_byte_compression_advantage=estimates[
            "total_byte_compression_advantage"
        ],
        total_byte_compression_advantage_ci_low=intervals[
            "total_byte_compression_advantage"
        ][0],
        total_byte_compression_advantage_ci_high=intervals[
            "total_byte_compression_advantage"
        ][1],
        macro_archive_saving_fraction=estimates["macro_archive_saving_fraction"],
        macro_archive_saving_percent=(
            100.0 * estimates["macro_archive_saving_fraction"]
        ),
        macro_archive_saving_fraction_ci_low=intervals[
            "macro_archive_saving_fraction"
        ][0],
        macro_archive_saving_fraction_ci_high=intervals[
            "macro_archive_saving_fraction"
        ][1],
        pooled_archive_saving_fraction=estimates[
            "pooled_archive_saving_fraction"
        ],
        pooled_archive_saving_percent=(
            100.0 * estimates["pooled_archive_saving_fraction"]
        ),
        pooled_archive_saving_fraction_ci_low=intervals[
            "pooled_archive_saving_fraction"
        ][0],
        pooled_archive_saving_fraction_ci_high=intervals[
            "pooled_archive_saving_fraction"
        ][1],
        total_archive_bytes_saved=baseline_outputs - brevis_outputs,
    )


def grouped_aggregates(
    rows: list[PairwiseRow],
    methods: list[str],
    attribute: str | None,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> list[Aggregate]:
    output = []
    baselines = [method for method in methods if method != BREVIS]
    if attribute is None:
        groups = ["all"]
    else:
        groups = sorted({str(getattr(row, attribute)) for row in rows})
    for group in groups:
        for baseline in baselines:
            selected = [
                row
                for row in rows
                if row.baseline == baseline
                and (attribute is None or getattr(row, attribute) == group)
            ]
            if selected:
                output.append(
                    aggregate(
                        selected,
                        group,
                        bootstrap_samples=bootstrap_samples,
                        bootstrap_seed=bootstrap_seed,
                    )
                )
    return output


def write_csv(path: Path, rows: Iterable[Any], columns: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _percent(value: float) -> str:
    return f"{value:.2f}%"


def _ratio(value: float) -> str:
    return f"{value:.4f}×"


def _ratio_interval(value: float, low: float, high: float) -> str:
    return f"{_ratio(value)} [{_ratio(low)}, {_ratio(high)}]"


def _fraction_interval(value: float, low: float, high: float) -> str:
    return (
        f"{_percent(100.0 * value)} "
        f"[{_percent(100.0 * low)}, {_percent(100.0 * high)}]"
    )


def write_markdown(
    path: Path,
    *,
    input_path: Path,
    rows: list[PairwiseRow],
    overall: list[Aggregate],
    by_domain: list[Aggregate],
    by_format: list[Aggregate],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    lines = [
        "# Paper-oriented analysis of verified compression results",
        "",
        f"Input: `{input_path}`",
        "",
        "All statistics use complete, verification-gated checkpoint cells. "
        "A win means strictly fewer compressed output bytes for Brevis. "
        "Win rate is wins divided by all models (ties remain in the denominator).",
        f"Bracketed 95% CIs use {bootstrap_samples:,} checkpoint-resampling "
        f"percentile bootstrap replicates with fixed seed {bootstrap_seed}.",
        "",
        "## Overall pairwise results",
        "",
        "Compression ratios below are always `source bytes / compressed output "
        "bytes` and are displayed with `×`. “GM” is the model-macro geometric "
        "mean. “Total-byte” sums source and output bytes before taking their ratio.",
        "",
        "| Baseline | W–T–L | Win rate [95% CI] | Brevis GM CR | Baseline GM CR | "
        "GM advantage [95% CI] | Brevis total-byte CR | Baseline total-byte CR | "
        "Total-byte advantage [95% CI] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in overall:
        lines.append(
            f"| {item.baseline} | {item.wins}–{item.ties}–{item.losses} | "
            f"{_fraction_interval(item.win_rate, item.win_rate_ci_low, item.win_rate_ci_high)} | "
            f"{_ratio(item.brevis_geomean_compression_ratio)} | "
            f"{_ratio(item.baseline_geomean_compression_ratio)} | "
            f"{_ratio_interval(item.geomean_compression_advantage, item.geomean_compression_advantage_ci_low, item.geomean_compression_advantage_ci_high)} | "
            f"{_ratio(item.brevis_total_byte_compression_ratio)} | "
            f"{_ratio(item.baseline_total_byte_compression_ratio)} | "
            f"{_ratio_interval(item.total_byte_compression_advantage, item.total_byte_compression_advantage_ci_low, item.total_byte_compression_advantage_ci_high)} |"
        )
    lines.extend(
        [
            "",
            "| Baseline | Macro archive saving [95% CI] | "
            "Pooled archive saving [95% CI] | Total bytes saved |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in overall:
        lines.append(
            f"| {item.baseline} | "
            f"{_fraction_interval(item.macro_archive_saving_fraction, item.macro_archive_saving_fraction_ci_low, item.macro_archive_saving_fraction_ci_high)} | "
            f"{_fraction_interval(item.pooled_archive_saving_fraction, item.pooled_archive_saving_fraction_ci_low, item.pooled_archive_saving_fraction_ci_high)} | "
            f"{item.total_archive_bytes_saved:,} |"
        )

    def group_table(title: str, aggregates: list[Aggregate]) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| Group | Baseline | N | W–T–L | GM advantage [95% CI] | "
                "Total-byte advantage [95% CI] | Macro archive saving [95% CI] | "
                "Pooled archive saving [95% CI] |",
                "|---|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for item in aggregates:
            lines.append(
                f"| {item.group} | {item.baseline} | {item.models} | "
                f"{item.wins}–{item.ties}–{item.losses} | "
                f"{_ratio_interval(item.geomean_compression_advantage, item.geomean_compression_advantage_ci_low, item.geomean_compression_advantage_ci_high)} | "
                f"{_ratio_interval(item.total_byte_compression_advantage, item.total_byte_compression_advantage_ci_low, item.total_byte_compression_advantage_ci_high)} | "
                f"{_fraction_interval(item.macro_archive_saving_fraction, item.macro_archive_saving_fraction_ci_low, item.macro_archive_saving_fraction_ci_high)} | "
                f"{_fraction_interval(item.pooled_archive_saving_fraction, item.pooled_archive_saving_fraction_ci_low, item.pooled_archive_saving_fraction_ci_high)} |"
            )

    group_table("By model domain", by_domain)
    group_table("By numeric format", by_format)

    counterexamples = [row for row in rows if row.winner != BREVIS]
    lines.extend(["", "## Counterexamples", ""])
    if counterexamples:
        lines.extend(
            [
                "| Checkpoint | Baseline winner | Brevis ratio | Baseline ratio | "
                "Brevis archive saving |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for row in counterexamples:
            lines.append(
                f"| {row.checkpoint} | {row.winner} | "
                f"{_ratio(row.brevis_compression_ratio)} | "
                f"{_ratio(row.baseline_compression_ratio)} | "
                f"{_percent(row.relative_archive_saving_percent)} |"
            )
    else:
        lines.append("No ties or Brevis losses occur in these measured pairs.")

    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- The geometric-mean advantage is the geometric mean of "
            "`Brevis compression ratio / baseline compression ratio`; values "
            "above 1 favor Brevis.",
            "- Total-byte compression ratio is `sum(source bytes) / "
            "sum(compressed output bytes)`. It is size-weighted and differs from "
            "the equal-model geometric mean.",
            "- Macro archive saving gives every checkpoint equal weight. Pooled "
            "archive saving sums bytes first and is dominated by large checkpoints.",
            "- Bracketed 95% intervals are percentile bootstrap intervals with "
            "checkpoints as the resampling unit. They quantify corpus composition "
            "sensitivity, not repeated-run measurement noise.",
            "- ZipNN is tensor-exact while Brevis and the other listed baselines "
            "are byte-exact. ZipNN comparisons therefore do not have identical "
            "container-layout preservation requirements.",
            "- Domain groups with two models and numeric-format groups with one or "
            "two models are descriptive slices, not evidence of broad generalization.",
            "- These are deterministic archive-size observations from one fixed "
            "corpus and configuration. The bootstrap describes sensitivity to "
            "checkpoint composition, not repeated-run or population uncertainty; "
            "the results do not support throughput, memory, or significance claims.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def _latex_text(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _latex_ratio(value: float) -> str:
    return f"${value:.4f}\\times$"


def _latex_ratio_interval(value: float, low: float, high: float) -> str:
    return (
        f"{_latex_ratio(value)} "
        f"[{_latex_ratio(low)}, {_latex_ratio(high)}]"
    )


def _latex_fraction_interval(value: float, low: float, high: float) -> str:
    return (
        f"{100.0 * value:.2f}\\% "
        f"[{100.0 * low:.2f}\\%, {100.0 * high:.2f}\\%]"
    )


def write_latex_tables(
    path: Path,
    *,
    overall: list[Aggregate],
    by_domain: list[Aggregate],
    by_format: list[Aggregate],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> None:
    lines = [
        "% Generated by scripts/analyze_verified_results.py.",
        r"% Requires \usepackage{graphicx} for \resizebox.",
        "% CR = source bytes / compressed output bytes.",
        "% GM = equal-checkpoint geometric mean; TB = total-byte ratio.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccccc}",
        r"\hline",
        r"Baseline & W/T/L & GM CR (B/base) & GM advantage [95\% CI] & "
        r"TB CR (B/base) & TB advantage [95\% CI] & "
        r"Macro saving [95\% CI] & Pooled saving [95\% CI] \\",
        r"\hline",
    ]
    for item in overall:
        lines.append(
            f"{_latex_text(item.baseline)} & "
            f"{item.wins}/{item.ties}/{item.losses} & "
            f"{_latex_ratio(item.brevis_geomean_compression_ratio)} / "
            f"{_latex_ratio(item.baseline_geomean_compression_ratio)} & "
            f"{_latex_ratio_interval(item.geomean_compression_advantage, item.geomean_compression_advantage_ci_low, item.geomean_compression_advantage_ci_high)} & "
            f"{_latex_ratio(item.brevis_total_byte_compression_ratio)} / "
            f"{_latex_ratio(item.baseline_total_byte_compression_ratio)} & "
            f"{_latex_ratio_interval(item.total_byte_compression_advantage, item.total_byte_compression_advantage_ci_low, item.total_byte_compression_advantage_ci_high)} & "
            f"{_latex_fraction_interval(item.macro_archive_saving_fraction, item.macro_archive_saving_fraction_ci_low, item.macro_archive_saving_fraction_ci_high)} & "
            f"{_latex_fraction_interval(item.pooled_archive_saving_fraction, item.pooled_archive_saving_fraction_ci_low, item.pooled_archive_saving_fraction_ci_high)} \\\\"
        )
    lines.extend(
        [
            r"\hline",
            r"\end{tabular}",
            r"}",
            (
                r"\caption{Verified checkpoint compression. B denotes Brevis. "
                r"Bracketed intervals are model-resampling percentile bootstrap "
                f"95\\% CIs ({bootstrap_samples:,} resamples; seed "
                f"{bootstrap_seed}).}}"
            ),
            r"\label{tab:verified-compression-overall}",
            r"\end{table*}",
        ]
    )

    def grouped_table(
        title: str,
        label: str,
        aggregates: list[Aggregate],
    ) -> None:
        lines.extend(
            [
                "",
                r"\begin{table*}[t]",
                r"\centering",
                r"\small",
                r"\setlength{\tabcolsep}{4pt}",
                r"\resizebox{\textwidth}{!}{%",
                r"\begin{tabular}{llrccccc}",
                r"\hline",
                r"Group & Baseline & $N$ & W/T/L & GM advantage [95\% CI] & "
                r"TB advantage [95\% CI] & Macro saving [95\% CI] & "
                r"Pooled saving [95\% CI] \\",
                r"\hline",
            ]
        )
        for item in aggregates:
            lines.append(
                f"{_latex_text(item.group)} & {_latex_text(item.baseline)} & "
                f"{item.models} & {item.wins}/{item.ties}/{item.losses} & "
                f"{_latex_ratio_interval(item.geomean_compression_advantage, item.geomean_compression_advantage_ci_low, item.geomean_compression_advantage_ci_high)} & "
                f"{_latex_ratio_interval(item.total_byte_compression_advantage, item.total_byte_compression_advantage_ci_low, item.total_byte_compression_advantage_ci_high)} & "
                f"{_latex_fraction_interval(item.macro_archive_saving_fraction, item.macro_archive_saving_fraction_ci_low, item.macro_archive_saving_fraction_ci_high)} & "
                f"{_latex_fraction_interval(item.pooled_archive_saving_fraction, item.pooled_archive_saving_fraction_ci_low, item.pooled_archive_saving_fraction_ci_high)} \\\\"
            )
        lines.extend(
            [
                r"\hline",
                r"\end{tabular}",
                r"}",
                f"\\caption{{{title}.}}",
                f"\\label{{{label}}}",
                r"\end{table*}",
            ]
        )

    grouped_table(
        "Verified compression comparisons by model domain",
        "tab:verified-compression-domain",
        by_domain,
    )
    grouped_table(
        "Verified compression comparisons by numeric format",
        "tab:verified-compression-format",
        by_format,
    )
    path.write_text("\n".join(lines) + "\n")


def analyze(
    input_path: Path,
    output_dir: Path,
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    cells, methods, provenance = load_cells(input_path)
    rows = make_pairwise_rows(cells, methods)
    overall = grouped_aggregates(
        rows,
        methods,
        None,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    by_domain = grouped_aggregates(
        rows,
        methods,
        "domain",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    by_format = grouped_aggregates(
        rows,
        methods,
        "numeric_format",
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    counterexamples = [asdict(row) for row in rows if row.winner != BREVIS]

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "pairwise-by-model.csv", rows, PAIRWISE_COLUMNS)
    write_csv(output_dir / "pairwise-overall.csv", overall, AGGREGATE_COLUMNS)
    write_csv(output_dir / "pairwise-by-domain.csv", by_domain, AGGREGATE_COLUMNS)
    write_csv(
        output_dir / "pairwise-by-numeric-format.csv",
        by_format,
        AGGREGATE_COLUMNS,
    )
    report = {
        "schema_version": 2,
        "input": str(input_path),
        "source_summary": provenance,
        "bootstrap": {
            "unit": "checkpoint",
            "method": "percentile",
            "confidence_level": 0.95,
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "seed_derivation": "sha256(seed, group, baseline)",
        },
        "definitions": {
            "win": "brevis_output_bytes < baseline_output_bytes",
            "win_rate": "wins / all compared models; ties stay in denominator",
            "compression_ratio": "source_bytes / output_bytes",
            "geomean_compression_ratio": (
                "equal-model geometric mean of compression_ratio"
            ),
            "geomean_compression_advantage": (
                "geometric_mean(brevis_compression_ratio / "
                "baseline_compression_ratio)"
            ),
            "total_byte_compression_ratio": (
                "sum(source_bytes) / sum(output_bytes)"
            ),
            "total_byte_compression_advantage": (
                "brevis_total_byte_compression_ratio / "
                "baseline_total_byte_compression_ratio; equivalently "
                "sum(baseline_output_bytes) / sum(brevis_output_bytes)"
            ),
            "relative_archive_saving_fraction": (
                "(baseline_output_bytes - brevis_output_bytes) / "
                "baseline_output_bytes"
            ),
            "macro_archive_saving_fraction": (
                "arithmetic mean of per-model relative archive savings"
            ),
            "pooled_archive_saving_fraction": (
                "(sum(baseline_output_bytes) - sum(brevis_output_bytes)) / "
                "sum(baseline_output_bytes)"
            ),
        },
        "taxonomy": {
            checkpoint: {"domain": domain, "numeric_format": numeric_format}
            for checkpoint, (domain, numeric_format) in CHECKPOINT_METADATA.items()
            if any(row.checkpoint == checkpoint for row in rows)
        },
        "overall": [asdict(item) for item in overall],
        "by_domain": [asdict(item) for item in by_domain],
        "by_numeric_format": [asdict(item) for item in by_format],
        "counterexamples": counterexamples,
        "caveats": [
            "ZipNN is tensor-exact; the other methods are byte-exact.",
            "Macro averages weight checkpoints equally; pooled savings weight bytes.",
            "Bootstrap intervals resample checkpoints, not repeated codec runs.",
            "Small groups are descriptive and do not establish generalization.",
            "Archive sizes alone do not establish speed, memory, or significance.",
        ],
    }
    (output_dir / "paper-analysis.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    write_markdown(
        output_dir / "paper-analysis.md",
        input_path=input_path,
        rows=rows,
        overall=overall,
        by_domain=by_domain,
        by_format=by_format,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    write_latex_tables(
        output_dir / "paper-tables.tex",
        overall=overall,
        by_domain=by_domain,
        by_format=by_format,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze an existing verification-gated ratio summary."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="path to verified-ratios.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="defaults to an analysis directory beside the input summary",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help=f"model-resampling replicates (default: {DEFAULT_BOOTSTRAP_SAMPLES})",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help=f"fixed bootstrap seed (default: {DEFAULT_BOOTSTRAP_SEED})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.input.parent / "analysis"
    try:
        report = analyze(
            args.input,
            output_dir,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
        )
    except AnalysisError as exc:
        print(f"analysis error: {exc}")
        return 2
    print(
        f"analyzed {len(report['taxonomy'])} checkpoints; "
        f"wrote paper tables under {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
