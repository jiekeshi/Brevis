#!/usr/bin/env python3
"""Build paper-ready summaries for the Brevis extra experiments.

This is a read-only post-processing step.  It consumes the append-only
benchmark log, the archive-attribution JSON, and the BF16 information-analysis
JSON; it never invokes a codec or changes any benchmark artifact.

Only successful compression attempts named by a successful exact-verification
record are admitted from ``runs.jsonl``.  The benchmark rows are therefore
measured, exact-verified, hot-cache, representative-single-shard pilot results
with n=1 per configuration.  Attribution covers that representative shard,
while the information-theory tables cover the full Llama-3.1-8B checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW = (
    ROOT
    / "results"
    / "llama31-8b-shard1-extra-minimal"
    / "raw"
    / "runs.jsonl"
)
DEFAULT_ATTRIBUTION = (
    ROOT
    / "results"
    / "attribution"
    / "llama-3.1-8b-bf16-shard1-budget32"
)
DEFAULT_INFORMATION = (
    ROOT
    / "results"
    / "information"
    / "llama-3.1-8b-bf16"
)
DEFAULT_OUTPUT = ROOT / "results" / "paper-extra-analysis"

PARETO_BUDGETS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256)
WORKER_COUNTS = (1, 4, 8, 16, 32)
ABLATION_VARIANTS = (
    "full",
    "no-astar",
    "no-phog",
    "no-phog-no-astar",
)
VARIANT_ORDER = {
    variant: index for index, variant in enumerate(ABLATION_VARIANTS)
}

BENCHMARK_SCOPE = "representative single shard"
BENCHMARK_MEASUREMENT = "measured; exact-verified; n=1 pilot"
ATTRIBUTION_SCOPE = "representative single-shard archive"
ATTRIBUTION_MEASUREMENT = (
    "measured archive accounting; n=1 archive; static attribution"
)
INFORMATION_SCOPE = "full-model BF16 checkpoint scan"
INFORMATION_MEASUREMENT = "measured full-model scan; n=1 checkpoint"

PARETO_COLUMNS = (
    "status",
    "measurement",
    "scope",
    "n",
    "checkpoint",
    "shard",
    "cache",
    "budget",
    "workers",
    "calibration_tensors",
    "source_bytes",
    "output_bytes",
    "compression_ratio_x",
    "wall_seconds",
    "throughput_mib_per_second",
    "saved_bytes_vs_budget0",
    "saved_mib_vs_budget0",
    "saved_percent_vs_budget0",
    "peak_rss_bytes",
    "peak_rss_gib",
    "exact_verified",
    "run_id",
    "attempt_id",
)
WORKER_COLUMNS = (
    "status",
    "measurement",
    "scope",
    "n",
    "checkpoint",
    "shard",
    "cache",
    "workers",
    "budget",
    "calibration_tensors",
    "source_bytes",
    "output_bytes",
    "compression_ratio_x",
    "wall_seconds",
    "throughput_mib_per_second",
    "speedup_vs_1_worker",
    "parallel_efficiency_percent",
    "peak_rss_bytes",
    "peak_rss_gib",
    "exact_verified",
    "run_id",
    "attempt_id",
)
ABLATION_COLUMNS = (
    "status",
    "measurement",
    "scope",
    "n",
    "checkpoint",
    "shard",
    "cache",
    "search_budget",
    "variant",
    "phog_enabled",
    "astar_enabled",
    "workers",
    "calibration_tensors",
    "source_bytes",
    "output_bytes",
    "compression_ratio_x",
    "wall_seconds",
    "throughput_mib_per_second",
    "extra_output_bytes_vs_full",
    "compression_ratio_delta_vs_full",
    "speedup_vs_full",
    "selected_budget_for_paper",
    "selected_for_paper",
    "peak_rss_bytes",
    "peak_rss_gib",
    "exact_verified",
    "run_id",
    "attempt_id",
)
ATTRIBUTION_COLUMNS = (
    "status",
    "measurement",
    "scope",
    "n",
    "search_budget",
    "role",
    "tensor_count",
    "source_tensor_bytes",
    "archive_record_bytes",
    "saved_bytes_vs_record",
    "compression_ratio_x",
    "record_percent_of_source",
    "program_bytes",
    "record_framing_bytes",
    "literal_fallback_tensors",
    "source_prefix_matched",
    "program_execution_performed",
)
ENTROPY_COLUMNS = (
    "status",
    "measurement",
    "scope",
    "n",
    "group",
    "tensor_count",
    "parameters",
    "parameter_share_percent",
    "nominal_bits_per_weight",
    "distinct_bf16_symbols",
    "empirical_bf16_symbol_h0_bpw",
    "distinct_exponents",
    "empirical_exponent_h0_bpw",
    "idealized_8_plus_iid_exponent_h0_bpw",
    "empirical_adjacent_exponent_h1_bpw",
    "idealized_8_plus_finite_adjacent_exponent_bpw",
    "manifest_sha256_verified",
)
METHOD_COLUMNS = (
    "status",
    "measurement",
    "scope",
    "n",
    "method",
    "source_file_bytes_basis",
    "whole_output_bytes",
    "compression_ratio_x",
    "amortized_whole_output_bpw",
    "delta_bpw_vs_empirical_bf16_h0",
    "delta_bpw_vs_8_plus_iid_exponent_h0",
    "delta_bpw_vs_finite_adjacent_exponent_reference",
    "output_size_input_kind",
    "output_size_is_exact_integer_input",
)


class ExtraAnalysisError(RuntimeError):
    """Raised when an input is missing or does not have the expected shape."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExtraAnalysisError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ExtraAnalysisError(f"expected a JSON object: {path}")
    return value


def read_jsonl_snapshot(path: Path) -> list[dict[str, Any]]:
    """Read a concurrent append-only log, tolerating a partial final line."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExtraAnalysisError(f"cannot read benchmark log: {path}") from exc
    records: list[dict[str, Any]] = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            partial_tail = index == len(lines) - 1 and not text.endswith("\n")
            if partial_tail:
                break
            raise ExtraAnalysisError(
                f"invalid JSONL record {index + 1}: {path}"
            ) from exc
        if not isinstance(value, dict):
            raise ExtraAnalysisError(
                f"JSONL record {index + 1} is not an object: {path}"
            )
        records.append(value)
    return records


def exact_verified_compressions(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return successful compressions admitted by exact verify records."""

    verified: set[tuple[str, str]] = set()
    for record in records:
        if (
            record.get("operation") != "verify"
            or record.get("status") != "ok"
            or record.get("exact") is not True
        ):
            continue
        attempts = record.get("verified_attempts")
        if not isinstance(attempts, list):
            continue
        for identity in attempts:
            if (
                isinstance(identity, list)
                and len(identity) == 2
                and all(isinstance(value, str) for value in identity)
            ):
                verified.add((identity[0], identity[1]))

    admitted = []
    for record in records:
        identity = (record.get("run_id"), record.get("attempt_id"))
        if (
            record.get("operation") == "compress"
            and record.get("status") == "ok"
            and identity in verified
        ):
            admitted.append(record)
    return admitted


def require_number(
    record: dict[str, Any],
    field: str,
    *,
    positive: bool = False,
) -> int | float:
    value = record.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExtraAnalysisError(
            f"verified record {record.get('run_id', '<unknown>')} has "
            f"invalid {field}"
        )
    if positive and value <= 0:
        raise ExtraAnalysisError(
            f"verified record {record.get('run_id', '<unknown>')} has "
            f"non-positive {field}"
        )
    return value


def latest_by_key(
    records: Iterable[dict[str, Any]],
    key: Callable[[dict[str, Any]], Any],
) -> dict[Any, dict[str, Any]]:
    latest: dict[Any, dict[str, Any]] = {}
    for record in records:
        record_key = key(record)
        previous = latest.get(record_key)
        if previous is None or str(record.get("finished_at", "")) >= str(
            previous.get("finished_at", "")
        ):
            latest[record_key] = record
    return latest


def benchmark_values(record: dict[str, Any]) -> dict[str, Any]:
    source = require_number(record, "source_bytes", positive=True)
    output = require_number(record, "output_bytes", positive=True)
    seconds = require_number(record, "wall_seconds", positive=True)
    peak_rss = require_number(record, "peak_rss_bytes", positive=True)
    return {
        "status": "complete",
        "measurement": BENCHMARK_MEASUREMENT,
        "scope": BENCHMARK_SCOPE,
        "n": 1,
        "checkpoint": record.get("checkpoint", ""),
        "shard": record.get("shard", ""),
        "cache": record.get("cache", ""),
        "source_bytes": source,
        "output_bytes": output,
        "compression_ratio_x": source / output,
        "wall_seconds": seconds,
        "throughput_mib_per_second": source / seconds / 1024**2,
        "peak_rss_bytes": peak_rss,
        "peak_rss_gib": peak_rss / 1024**3,
        "exact_verified": True,
        "run_id": record.get("run_id", ""),
        "attempt_id": record.get("attempt_id", ""),
    }


def missing_benchmark_values() -> dict[str, Any]:
    return {
        "status": "missing",
        "measurement": "not measured",
        "scope": BENCHMARK_SCOPE,
        "n": 0,
        "checkpoint": "",
        "shard": "",
        "cache": "",
        "source_bytes": None,
        "output_bytes": None,
        "compression_ratio_x": None,
        "wall_seconds": None,
        "throughput_mib_per_second": None,
        "peak_rss_bytes": None,
        "peak_rss_gib": None,
        "exact_verified": False,
        "run_id": "",
        "attempt_id": "",
    }


def build_pareto_rows(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = latest_by_key(
        (
            record
            for record in records
            if record.get("stage") == "pareto"
            and isinstance(record.get("max_expansions"), int)
        ),
        lambda record: record["max_expansions"],
    )
    baseline = selected.get(0)
    baseline_output = (
        require_number(baseline, "output_bytes", positive=True)
        if baseline is not None
        else None
    )
    rows = []
    for budget in PARETO_BUDGETS:
        record = selected.get(budget)
        values = (
            benchmark_values(record)
            if record is not None
            else missing_benchmark_values()
        )
        output = values["output_bytes"]
        if baseline_output is not None and output is not None:
            saved = baseline_output - output
            saved_mib = saved / 1024**2
            saved_percent = saved / baseline_output * 100
        else:
            saved = saved_mib = saved_percent = None
        rows.append(
            {
                **values,
                "budget": budget,
                "workers": record.get("workers") if record else None,
                "calibration_tensors": (
                    record.get("calibration_tensors") if record else None
                ),
                "saved_bytes_vs_budget0": saved,
                "saved_mib_vs_budget0": saved_mib,
                "saved_percent_vs_budget0": saved_percent,
            }
        )
    return rows


def build_worker_rows(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = latest_by_key(
        (
            record
            for record in records
            if record.get("stage") == "workers"
            and isinstance(record.get("workers"), int)
        ),
        lambda record: record["workers"],
    )
    one_worker = selected.get(1)
    one_worker_seconds = (
        require_number(one_worker, "wall_seconds", positive=True)
        if one_worker is not None
        else None
    )
    rows = []
    for workers in WORKER_COUNTS:
        record = selected.get(workers)
        values = (
            benchmark_values(record)
            if record is not None
            else missing_benchmark_values()
        )
        seconds = values["wall_seconds"]
        speedup = (
            one_worker_seconds / seconds
            if one_worker_seconds is not None and seconds is not None
            else None
        )
        rows.append(
            {
                **values,
                "workers": workers,
                "budget": (
                    record.get("max_expansions") if record else None
                ),
                "calibration_tensors": (
                    record.get("calibration_tensors") if record else None
                ),
                "speedup_vs_1_worker": speedup,
                "parallel_efficiency_percent": (
                    speedup / workers * 100
                    if speedup is not None
                    else None
                ),
            }
        )
    return rows


def choose_ablation_budget(
    records_by_key: dict[tuple[int, str], dict[str, Any]],
) -> int | None:
    by_budget: dict[int, set[str]] = {}
    for budget, variant in records_by_key:
        by_budget.setdefault(budget, set()).add(variant)
    complete = [
        budget
        for budget, variants in by_budget.items()
        if all(variant in variants for variant in ABLATION_VARIANTS)
    ]
    if 32 in complete:
        return 32
    if complete:
        return max(complete)
    if not by_budget:
        return None
    return max(
        by_budget,
        key=lambda budget: (
            len(by_budget[budget]),
            budget == 32,
            budget,
        ),
    )


def build_ablation_rows(
    records: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int | None]:
    candidates = (
        record
        for record in records
        if record.get("stage") == "ablation"
        and isinstance(record.get("max_expansions"), int)
        and record.get("variant") in VARIANT_ORDER
    )
    selected = latest_by_key(
        candidates,
        lambda record: (
            record["max_expansions"],
            record["variant"],
        ),
    )
    selected_budget = choose_ablation_budget(selected)
    rows = []
    for (budget, variant), record in sorted(
        selected.items(),
        key=lambda item: (item[0][0], VARIANT_ORDER[item[0][1]]),
    ):
        values = benchmark_values(record)
        full = selected.get((budget, "full"))
        if full is not None:
            full_output = require_number(full, "output_bytes", positive=True)
            full_seconds = require_number(full, "wall_seconds", positive=True)
            full_ratio = require_number(full, "source_bytes", positive=True) / (
                full_output
            )
            output = values["output_bytes"]
            seconds = values["wall_seconds"]
            extra_output = output - full_output
            ratio_delta = values["compression_ratio_x"] - full_ratio
            speedup = full_seconds / seconds
        else:
            extra_output = ratio_delta = speedup = None
        rows.append(
            {
                **values,
                "search_budget": budget,
                "variant": variant,
                "phog_enabled": record.get("phog"),
                "astar_enabled": record.get("astar_heuristic"),
                "workers": record.get("workers"),
                "calibration_tensors": record.get("calibration_tensors"),
                "extra_output_bytes_vs_full": extra_output,
                "compression_ratio_delta_vs_full": ratio_delta,
                "speedup_vs_full": speedup,
                "selected_budget_for_paper": selected_budget,
                "selected_for_paper": budget == selected_budget,
            }
        )
    return rows, selected_budget


def attribution_budget(data: dict[str, Any]) -> int | None:
    archives = data.get("archives")
    if not isinstance(archives, list) or len(archives) != 1:
        return None
    archive = archives[0]
    if not isinstance(archive, dict):
        return None
    match = re.search(r"(?:^|/)budget-(\d+)(?:/|\\.)", str(
        archive.get("archive_path", "")
    ))
    return int(match.group(1)) if match else None


def build_attribution_rows(directory: Path) -> list[dict[str, Any]]:
    path = directory / "attribution.json"
    data = read_json(path)
    groups = data.get("groups")
    totals = data.get("totals")
    validation = data.get("validation_scope")
    if (
        not isinstance(groups, dict)
        or not isinstance(groups.get("role"), list)
        or not isinstance(totals, dict)
        or not isinstance(validation, dict)
    ):
        raise ExtraAnalysisError(f"incomplete attribution JSON: {path}")
    budget = attribution_budget(data)
    rows = []
    for source in groups["role"]:
        if not isinstance(source, dict) or not isinstance(
            source.get("group"), str
        ):
            raise ExtraAnalysisError(f"invalid role row: {path}")
        required = (
            "tensor_count",
            "source_tensor_bytes",
            "archive_record_bytes",
            "saved_bytes_vs_record",
            "source_over_record_ratio",
            "record_percent_of_source",
            "program_bytes",
            "record_framing_bytes",
            "literal_fallback_tensors",
        )
        if any(not isinstance(source.get(field), (int, float)) for field in required):
            raise ExtraAnalysisError(f"invalid role metrics: {path}")
        rows.append(
            {
                "status": "complete",
                "measurement": ATTRIBUTION_MEASUREMENT,
                "scope": ATTRIBUTION_SCOPE,
                "n": 1,
                "search_budget": budget,
                "role": source["group"],
                "tensor_count": source["tensor_count"],
                "source_tensor_bytes": source["source_tensor_bytes"],
                "archive_record_bytes": source["archive_record_bytes"],
                "saved_bytes_vs_record": source["saved_bytes_vs_record"],
                "compression_ratio_x": source[
                    "source_over_record_ratio"
                ],
                "record_percent_of_source": source[
                    "record_percent_of_source"
                ],
                "program_bytes": source["program_bytes"],
                "record_framing_bytes": source["record_framing_bytes"],
                "literal_fallback_tensors": source[
                    "literal_fallback_tensors"
                ],
                "source_prefix_matched": validation.get(
                    "source_prefix_match"
                ),
                "program_execution_performed": validation.get(
                    "program_execution"
                ),
            }
        )
    return rows


def build_information_rows(
    directory: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = directory / "information-analysis.json"
    data = read_json(path)
    groups = data.get("groups")
    methods = data.get("methods")
    checkpoint = data.get("checkpoint")
    if (
        not isinstance(groups, list)
        or not isinstance(methods, list)
        or not isinstance(checkpoint, dict)
    ):
        raise ExtraAnalysisError(f"incomplete information JSON: {path}")
    discovery = checkpoint.get("discovery")
    manifest_verified = (
        discovery.get("manifest_reported_sha256_verified")
        if isinstance(discovery, dict)
        else None
    )

    entropy_rows = []
    for source in groups:
        if not isinstance(source, dict) or not isinstance(
            source.get("group"), str
        ):
            raise ExtraAnalysisError(f"invalid entropy group: {path}")
        entropy_rows.append(
            {
                "status": "complete",
                "measurement": INFORMATION_MEASUREMENT,
                "scope": INFORMATION_SCOPE,
                "n": 1,
                "group": source["group"],
                "tensor_count": source.get("tensor_count"),
                "parameters": source.get("parameters"),
                "parameter_share_percent": source.get(
                    "parameter_share_percent"
                ),
                "nominal_bits_per_weight": source.get(
                    "nominal_bits_per_weight"
                ),
                "distinct_bf16_symbols": source.get(
                    "distinct_bf16_symbols"
                ),
                "empirical_bf16_symbol_h0_bpw": source.get(
                    "empirical_bf16_symbol_h0_bits_per_weight"
                ),
                "distinct_exponents": source.get("distinct_exponents"),
                "empirical_exponent_h0_bpw": source.get(
                    "empirical_exponent_h0_bits_per_exponent"
                ),
                "idealized_8_plus_iid_exponent_h0_bpw": source.get(
                    "idealized_raw_sign_mantissa_plus_iid_exponent_bpw"
                ),
                "empirical_adjacent_exponent_h1_bpw": source.get(
                    "empirical_adjacent_exponent_h1_conditional_bits_per_exponent"
                ),
                "idealized_8_plus_finite_adjacent_exponent_bpw": source.get(
                    "idealized_raw_sign_mantissa_plus_finite_adjacent_"
                    "exponent_reference_bpw"
                ),
                "manifest_sha256_verified": manifest_verified,
            }
        )

    method_rows = []
    for source in methods:
        if not isinstance(source, dict) or not isinstance(
            source.get("method"), str
        ):
            raise ExtraAnalysisError(f"invalid information method: {path}")
        method_rows.append(
            {
                "status": "complete",
                "measurement": (
                    "explicit whole-output byte count joined to measured "
                    "full-model scan; n=1 checkpoint"
                ),
                "scope": INFORMATION_SCOPE,
                "n": 1,
                "method": source["method"],
                "source_file_bytes_basis": source.get(
                    "source_file_bytes_basis"
                ),
                "whole_output_bytes": source.get("whole_output_bytes"),
                "compression_ratio_x": source.get(
                    "compression_ratio_source_over_output"
                ),
                "amortized_whole_output_bpw": source.get(
                    "amortized_whole_output_bpw_per_analyzed_bf16_weight"
                ),
                "delta_bpw_vs_empirical_bf16_h0": source.get(
                    "signed_amortized_bpw_difference_from_empirical_bf16_symbol_h0"
                ),
                "delta_bpw_vs_8_plus_iid_exponent_h0": source.get(
                    "signed_amortized_bpw_difference_from_idealized_"
                    "8_plus_iid_exponent_h0"
                ),
                "delta_bpw_vs_finite_adjacent_exponent_reference": source.get(
                    "signed_amortized_bpw_difference_from_finite_"
                    "adjacent_exponent_reference"
                ),
                "output_size_input_kind": source.get("input_kind"),
                "output_size_is_exact_integer_input": source.get(
                    "whole_output_bytes_is_exact_integer_input"
                ),
            }
        )
    return entropy_rows, method_rows


def write_csv(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(
            {
                column: "" if row.get(column) is None else row.get(column)
                for column in columns
            }
            for row in rows
        )


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    if value in (None, ""):
        return "—"
    return str(value)


def ratio(value: Any) -> str:
    return f"{value:.6f}×" if isinstance(value, (int, float)) else "—"


def markdown_cell(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def markdown_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend(
        "| "
        + " | ".join(markdown_cell(value) for value in values)
        + " |"
        for values in rows
    )
    return lines


def write_markdown(
    path: Path,
    tables: dict[str, list[dict[str, Any]]],
    selected_ablation_budget: int | None,
) -> None:
    pareto = tables["pareto"]
    workers = tables["workers"]
    ablation = tables["ablation"]
    attribution = tables["attribution"]
    entropy = tables["entropy"]
    methods = tables["methods"]
    selected_ablation = [
        row for row in ablation if row["selected_for_paper"]
    ]
    lines = [
        "# Brevis extra experiments",
        "",
        "All benchmark points below are measured hot-cache runs on one "
        "representative Llama-3.1-8B BF16 shard, exact-verified, with n=1 "
        "per configuration. They are pilot measurements, not full-model "
        "timings or multi-run uncertainty estimates.",
        "",
        "Attribution is measured static accounting of the representative "
        "budget-32 archive. It does not execute the stored programs. The "
        "information-theory section is a separate full-model BF16 scan "
        "(n=1 checkpoint); empirical entropy values are descriptive "
        "references, not arbitrary-structure lower bounds.",
        "",
        "## Search-budget Pareto sweep",
        "",
        *markdown_table(
            (
                "Budget",
                "Ratio",
                "Time (s)",
                "MiB/s",
                "Saved vs B=0 (MiB)",
                "Saved vs B=0 (%)",
                "Peak RSS (GiB)",
                "Status",
            ),
            (
                (
                    row["budget"],
                    ratio(row["compression_ratio_x"]),
                    fmt(row["wall_seconds"]),
                    fmt(row["throughput_mib_per_second"]),
                    fmt(row["saved_mib_vs_budget0"]),
                    fmt(row["saved_percent_vs_budget0"], 4),
                    fmt(row["peak_rss_gib"]),
                    row["status"],
                )
                for row in pareto
            ),
        ),
        "",
        "Compression ratio is source bytes / archive bytes; larger is "
        "better. Savings are relative to the measured budget-0 archive.",
        "",
        "## Worker scaling",
        "",
        *markdown_table(
            (
                "Workers",
                "Time (s)",
                "MiB/s",
                "Speedup vs 1",
                "Efficiency (%)",
                "Peak RSS (GiB)",
                "Ratio",
                "Status",
            ),
            (
                (
                    row["workers"],
                    fmt(row["wall_seconds"]),
                    fmt(row["throughput_mib_per_second"]),
                    fmt(row["speedup_vs_1_worker"]),
                    fmt(row["parallel_efficiency_percent"]),
                    fmt(row["peak_rss_gib"]),
                    ratio(row["compression_ratio_x"]),
                    row["status"],
                )
                for row in workers
            ),
        ),
        "",
        "## Ablation",
        "",
        (
            f"The paper-facing table selects search budget "
            f"{selected_ablation_budget}. Budget 32 is preferred when its "
            "four expected variants are complete; otherwise the largest "
            "complete budget is selected. `ablation.csv` retains every "
            "measured budget."
            if selected_ablation_budget is not None
            else "No complete verified ablation measurements are available."
        ),
        "",
        *markdown_table(
            (
                "Budget",
                "Variant",
                "Ratio",
                "Time (s)",
                "MiB/s",
                "Extra bytes vs full",
                "Speedup vs full",
            ),
            (
                (
                    row["search_budget"],
                    row["variant"],
                    ratio(row["compression_ratio_x"]),
                    fmt(row["wall_seconds"]),
                    fmt(row["throughput_mib_per_second"]),
                    fmt(row["extra_output_bytes_vs_full"]),
                    fmt(row["speedup_vs_full"]),
                )
                for row in selected_ablation
            ),
        ),
        "",
        "## Archive attribution by tensor role",
        "",
        *markdown_table(
            (
                "Role",
                "Tensors",
                "Source bytes",
                "Record bytes",
                "Saved bytes",
                "Ratio",
                "Literal fallbacks",
            ),
            (
                (
                    row["role"],
                    fmt(row["tensor_count"]),
                    fmt(row["source_tensor_bytes"]),
                    fmt(row["archive_record_bytes"]),
                    fmt(row["saved_bytes_vs_record"]),
                    ratio(row["compression_ratio_x"]),
                    fmt(row["literal_fallback_tensors"]),
                )
                for row in attribution
            ),
        ),
        "",
        "## Full-model BF16 information analysis",
        "",
        *markdown_table(
            (
                "Group",
                "Weights",
                "Share (%)",
                "BF16 H0 (bpw)",
                "Exponent H0 (bpw)",
                "8 + exponent H0",
                "8 + adjacent exponent ref.",
            ),
            (
                (
                    row["group"],
                    fmt(row["parameters"]),
                    fmt(row["parameter_share_percent"]),
                    fmt(row["empirical_bf16_symbol_h0_bpw"]),
                    fmt(row["empirical_exponent_h0_bpw"]),
                    fmt(row["idealized_8_plus_iid_exponent_h0_bpw"]),
                    fmt(
                        row[
                            "idealized_8_plus_finite_adjacent_exponent_bpw"
                        ]
                    ),
                )
                for row in entropy
            ),
        ),
        "",
        "## Full-model method BPW",
        "",
        *markdown_table(
            (
                "Method",
                "Ratio",
                "Output BPW",
                "Δ vs BF16 H0",
                "Δ vs 8+exp H0",
                "Δ vs adjacent ref.",
            ),
            (
                (
                    row["method"],
                    ratio(row["compression_ratio_x"]),
                    fmt(row["amortized_whole_output_bpw"]),
                    fmt(row["delta_bpw_vs_empirical_bf16_h0"]),
                    fmt(row["delta_bpw_vs_8_plus_iid_exponent_h0"]),
                    fmt(
                        row[
                            "delta_bpw_vs_finite_adjacent_exponent_reference"
                        ]
                    ),
                )
                for row in methods
            ),
        ),
        "",
        "Method BPW uses each explicit whole-output byte count recorded by "
        "the information analyzer and the measured full-model BF16 weight "
        "count; this summarizer does not re-run the codecs.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def latex_text(value: Any) -> str:
    rendered = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "×": r"$\times$",
        "—": "--",
        "Δ": r"$\Delta$",
    }
    return "".join(
        replacements.get(character, character) for character in rendered
    )


def latex_table(
    *,
    columns: str,
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    caption: str,
    label: str,
    wide: bool = False,
) -> list[str]:
    environment = "table*" if wide else "table"
    lines = [
        rf"\begin{{{environment}}}[t]",
        r"\centering",
        r"\scriptsize",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(latex_text(value) for value in headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(
        " & ".join(latex_text(value) for value in values) + r" \\"
        for values in rows
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            rf"\caption{{{latex_text(caption)}}}",
            rf"\label{{{label}}}",
            rf"\end{{{environment}}}",
            "",
        ]
    )
    return lines


def write_latex(
    path: Path,
    tables: dict[str, list[dict[str, Any]]],
    selected_ablation_budget: int | None,
) -> None:
    pareto = tables["pareto"]
    workers = tables["workers"]
    ablation = [
        row for row in tables["ablation"] if row["selected_for_paper"]
    ]
    attribution = tables["attribution"]
    entropy = tables["entropy"]
    methods = tables["methods"]
    lines = [
        "% Generated by scripts/build_extra_paper_analysis.py",
        "% Requires booktabs. All benchmark configurations use n=1.",
        "",
        *latex_table(
            columns="rrrrrrr",
            headers=(
                "Budget",
                "Ratio",
                "Time (s)",
                "MiB/s",
                "Saved (MiB)",
                "Saved (%)",
                "RSS (GiB)",
            ),
            rows=(
                (
                    row["budget"],
                    ratio(row["compression_ratio_x"]).replace("—", "--"),
                    fmt(row["wall_seconds"]).replace("—", "--"),
                    fmt(row["throughput_mib_per_second"]).replace("—", "--"),
                    fmt(row["saved_mib_vs_budget0"]).replace("—", "--"),
                    fmt(row["saved_percent_vs_budget0"], 4).replace(
                        "—", "--"
                    ),
                    fmt(row["peak_rss_gib"]).replace("—", "--"),
                )
                for row in pareto
            ),
            caption=(
                "Search-budget trade-off on one representative "
                "Llama-3.1-8B BF16 shard (measured hot-cache pilot, n=1, "
                "exact-verified). Ratio is source/output."
            ),
            label="tab:brevis-budget-pilot",
            wide=True,
        ),
        *latex_table(
            columns="rrrrrr",
            headers=(
                "Workers",
                "Time (s)",
                "MiB/s",
                "Speedup",
                "Efficiency (%)",
                "RSS (GiB)",
            ),
            rows=(
                (
                    row["workers"],
                    fmt(row["wall_seconds"]).replace("—", "--"),
                    fmt(row["throughput_mib_per_second"]).replace("—", "--"),
                    fmt(row["speedup_vs_1_worker"]).replace("—", "--"),
                    fmt(row["parallel_efficiency_percent"]).replace(
                        "—", "--"
                    ),
                    fmt(row["peak_rss_gib"]).replace("—", "--"),
                )
                for row in workers
            ),
            caption=(
                "Worker scaling on one representative shard (measured "
                "hot-cache pilot, n=1 per configuration, exact-verified)."
            ),
            label="tab:brevis-worker-pilot",
        ),
        *latex_table(
            columns="llrrrr",
            headers=(
                "Budget",
                "Variant",
                "Ratio",
                "Time (s)",
                "Extra bytes",
                "Speedup",
            ),
            rows=(
                (
                    row["search_budget"],
                    row["variant"],
                    ratio(row["compression_ratio_x"]),
                    fmt(row["wall_seconds"]),
                    fmt(row["extra_output_bytes_vs_full"]),
                    fmt(row["speedup_vs_full"]),
                )
                for row in ablation
            ),
            caption=(
                f"Brevis ablation at search budget "
                f"{selected_ablation_budget} on one representative shard "
                "(measured hot-cache pilot, n=1, exact-verified)."
            ),
            label="tab:brevis-ablation-pilot",
        ),
        *latex_table(
            columns="lrrrrrr",
            headers=(
                "Role",
                "Tensors",
                "Source bytes",
                "Record bytes",
                "Saved bytes",
                "Ratio",
                "Fallbacks",
            ),
            rows=(
                (
                    row["role"],
                    row["tensor_count"],
                    row["source_tensor_bytes"],
                    row["archive_record_bytes"],
                    row["saved_bytes_vs_record"],
                    ratio(row["compression_ratio_x"]),
                    row["literal_fallback_tensors"],
                )
                for row in attribution
            ),
            caption=(
                "Static archive attribution by tensor role for one measured "
                "representative-shard archive (n=1 archive)."
            ),
            label="tab:brevis-attribution",
            wide=True,
        ),
        *latex_table(
            columns="lrrrrrr",
            headers=(
                "Group",
                "Weights",
                "Share (%)",
                "BF16 H0",
                "Exp. H0",
                "8+Exp. H0",
                "8+Adjacent",
            ),
            rows=(
                (
                    row["group"],
                    row["parameters"],
                    fmt(row["parameter_share_percent"]),
                    fmt(row["empirical_bf16_symbol_h0_bpw"]),
                    fmt(row["empirical_exponent_h0_bpw"]),
                    fmt(row["idealized_8_plus_iid_exponent_h0_bpw"]),
                    fmt(
                        row[
                            "idealized_8_plus_finite_adjacent_exponent_bpw"
                        ]
                    ),
                )
                for row in entropy
            ),
            caption=(
                "Measured full-model BF16 empirical information references "
                "(n=1 checkpoint scan; values in bits per weight)."
            ),
            label="tab:brevis-information",
            wide=True,
        ),
        *latex_table(
            columns="lrrrrr",
            headers=(
                "Method",
                "Ratio",
                "Output BPW",
                "Delta BF16 H0",
                "Delta 8+Exp.",
                "Delta adjacent",
            ),
            rows=(
                (
                    row["method"],
                    ratio(row["compression_ratio_x"]),
                    fmt(row["amortized_whole_output_bpw"]),
                    fmt(row["delta_bpw_vs_empirical_bf16_h0"]),
                    fmt(row["delta_bpw_vs_8_plus_iid_exponent_h0"]),
                    fmt(
                        row[
                            "delta_bpw_vs_finite_adjacent_exponent_reference"
                        ]
                    ),
                )
                for row in methods
            ),
            caption=(
                "Whole-checkpoint method size normalized by the measured "
                "full-model BF16 weight count."
            ),
            label="tab:brevis-method-bpw",
        ),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_extra_analysis(
    *,
    raw_path: Path,
    attribution_dir: Path,
    information_dir: Path,
    output_dir: Path,
) -> dict[str, list[dict[str, Any]]]:
    raw_records = read_jsonl_snapshot(raw_path)
    verified = exact_verified_compressions(raw_records)
    pareto = build_pareto_rows(verified)
    workers = build_worker_rows(verified)
    ablation, selected_ablation_budget = build_ablation_rows(verified)
    attribution = build_attribution_rows(attribution_dir)
    entropy, methods = build_information_rows(information_dir)
    tables = {
        "pareto": pareto,
        "workers": workers,
        "ablation": ablation,
        "attribution": attribution,
        "entropy": entropy,
        "methods": methods,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, columns, rows in (
        ("pareto-budget.csv", PARETO_COLUMNS, pareto),
        ("worker-scaling.csv", WORKER_COLUMNS, workers),
        ("ablation.csv", ABLATION_COLUMNS, ablation),
        ("attribution-roles.csv", ATTRIBUTION_COLUMNS, attribution),
        ("information-entropy.csv", ENTROPY_COLUMNS, entropy),
        ("information-method-bpw.csv", METHOD_COLUMNS, methods),
    ):
        write_csv(output_dir / filename, columns, rows)
    write_markdown(
        output_dir / "extra-experiments.md",
        tables,
        selected_ablation_budget,
    )
    write_latex(
        output_dir / "extra-experiments.tex",
        tables,
        selected_ablation_budget,
    )
    return tables


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument(
        "--attribution-dir",
        type=Path,
        default=DEFAULT_ATTRIBUTION,
    )
    parser.add_argument(
        "--information-dir",
        type=Path,
        default=DEFAULT_INFORMATION,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    tables = build_extra_analysis(
        raw_path=args.raw,
        attribution_dir=args.attribution_dir,
        information_dir=args.information_dir,
        output_dir=args.output_dir,
    )
    selected = {
        row["selected_budget_for_paper"]
        for row in tables["ablation"]
        if row["selected_for_paper"]
    }
    selected_label = str(next(iter(selected))) if selected else "none"
    print(
        f"wrote {sum(len(rows) for rows in tables.values())} rows to "
        f"{args.output_dir}; paper ablation budget={selected_label}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
