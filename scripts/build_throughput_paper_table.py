#!/usr/bin/env python3
"""Merge the Llama-70B throughput runs into a compact paper table.

The three input experiments were intentionally split: strict Brevis, strict
Zstd, and fast baselines whose final round-trip scan is disabled.  This script
reads immutable environment metadata plus the latest raw attempt for each
measured repetition.  It never launches a codec or reads checkpoint weights.

Incomplete methods remain in the output with blank measurements and an
explicit reason, so the command can be rerun while the experiments progress.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "results" / "throughput-llama70-paper-table"

METHOD_LABELS = {
    "brevis": "Brevis",
    "zstd-9": "Zstd-9",
    "zipnn": "ZipNN",
    "lz4-hc-9": "LZ4-HC-9",
    "libdeflate-1": "libdeflate-1",
    "snappy": "Snappy",
}

CSV_COLUMNS = (
    "status",
    "method",
    "method_label",
    "checkpoint",
    "profile",
    "cache_mode",
    "process_thread_config",
    "execution_model",
    "declared_cpu_slots",
    "configured_shard_jobs",
    "effective_shard_jobs",
    "processes_per_command",
    "effective_processes",
    "codec_workers_per_process",
    "expected_threads_per_process",
    "compress_seconds",
    "decompress_seconds",
    "compress_gb_per_second",
    "decompress_gb_per_second",
    "compress_gib_per_second",
    "decompress_gib_per_second",
    "compression_ratio_x",
    "round_trip_policy",
    "round_trip_check_performed",
    "round_trip_check_status",
    "round_trip_provenance",
    "n",
    "expected_n",
    "measurement_label",
    "missing_reason",
    "source_result_dir",
    "experiment_id",
)


class PaperTableError(RuntimeError):
    pass


@dataclass(frozen=True)
class InputSpec:
    role: str
    directory: Path
    methods: tuple[str, ...]
    strict_round_trip: bool


def default_inputs(args: argparse.Namespace) -> tuple[InputSpec, ...]:
    return (
        InputSpec(
            "brevis_strict",
            args.brevis_results,
            ("brevis",),
            True,
        ),
        InputSpec(
            "zstd_strict",
            args.zstd_results,
            ("zstd-9",),
            True,
        ),
        InputSpec(
            "fast_baselines",
            args.fast_baseline_results,
            ("zipnn", "lz4-hc-9", "libdeflate-1", "snappy"),
            False,
        ),
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperTableError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PaperTableError(f"expected a JSON object: {path}")
    return value


def environment_path(directory: Path) -> Path | None:
    current = directory / "throughput-environment.json"
    if current.is_file():
        return current
    snapshots = sorted(directory.glob("throughput-environment-*.json"))
    if not snapshots:
        return None
    if len(snapshots) > 1:
        raise PaperTableError(
            f"{directory} has multiple environment snapshots but no "
            "throughput-environment.json selector"
        )
    return snapshots[0]


def read_jsonl_snapshot(path: Path) -> list[dict[str, Any]]:
    """Read an append-only JSONL log, tolerating only a partial final line."""

    if not path.is_file():
        return []
    try:
        text = path.read_text()
    except OSError as exc:
        raise PaperTableError(f"cannot read raw log: {path}") from exc
    records = []
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            is_partial_tail = index == len(lines) - 1 and not text.endswith("\n")
            if is_partial_tail:
                break
            raise PaperTableError(
                f"invalid JSONL record {index + 1}: {path}"
            ) from exc
        if not isinstance(value, dict):
            raise PaperTableError(
                f"raw record {index + 1} is not an object: {path}"
            )
        records.append(value)
    return records


def require_environment_shape(
    environment: dict[str, Any],
    path: Path,
) -> tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    experiment_id = environment.get("experiment_id")
    protocol = environment.get("protocol")
    resource_matrix = environment.get("resource_matrix")
    checkpoints = environment.get("checkpoints")
    if (
        not isinstance(experiment_id, str)
        or not experiment_id
        or not isinstance(protocol, dict)
        or not isinstance(resource_matrix, dict)
        or not isinstance(checkpoints, list)
        or len(checkpoints) != 1
        or not isinstance(checkpoints[0], dict)
    ):
        raise PaperTableError(f"incomplete throughput environment: {path}")
    checkpoint = checkpoints[0]
    if (
        not isinstance(checkpoint.get("name"), str)
        or not isinstance(checkpoint.get("files"), list)
        or not checkpoint["files"]
    ):
        raise PaperTableError(f"invalid checkpoint descriptor: {path}")
    return experiment_id, protocol, resource_matrix, checkpoint


def process_configuration(
    method: str,
    resource: dict[str, Any] | None,
    shard_count: int | None,
) -> dict[str, Any]:
    if not resource or not shard_count:
        return {
            "process_thread_config": "",
            "execution_model": "",
            "declared_cpu_slots": None,
            "configured_shard_jobs": None,
            "effective_shard_jobs": None,
            "processes_per_command": None,
            "effective_processes": None,
            "codec_workers_per_process": None,
            "expected_threads_per_process": None,
        }
    configured_jobs = resource.get("shard_jobs")
    processes_per_command = resource.get("processes_per_command")
    codec_workers = resource.get("codec_workers")
    expected_threads = resource.get("expected_threads_per_process")
    if not all(
        isinstance(value, int) and value >= 1
        for value in (
            configured_jobs,
            processes_per_command,
            codec_workers,
            expected_threads,
        )
    ):
        raise PaperTableError(f"invalid resource metadata for {method}")
    effective_jobs = min(configured_jobs, shard_count)
    effective_processes = effective_jobs * processes_per_command
    if method == "libdeflate-1" and processes_per_command == 2:
        label = (
            f"{effective_jobs} shard jobs; {effective_processes} processes "
            f"(adapter + codec); {codec_workers} codec thread/job"
        )
    elif resource.get("execution_model") == "single_process_internal_threads":
        label = f"1 process × {codec_workers} codec threads"
    else:
        unit = "worker" if method == "brevis" else "codec thread"
        suffix = "" if codec_workers == 1 else "s"
        label = (
            f"{effective_jobs} shard processes × {codec_workers} "
            f"{unit}{suffix}"
        )
    return {
        "process_thread_config": label,
        "execution_model": resource.get("execution_model", ""),
        "declared_cpu_slots": resource.get("declared_cpu_slots"),
        "configured_shard_jobs": configured_jobs,
        "effective_shard_jobs": effective_jobs,
        "processes_per_command": processes_per_command,
        "effective_processes": effective_processes,
        "codec_workers_per_process": codec_workers,
        "expected_threads_per_process": expected_threads,
    }


def latest_attempts_for_method(
    records: Sequence[dict[str, Any]],
    *,
    experiment_id: str,
    method: str,
    checkpoint: str,
    profile: str,
) -> dict[int, list[dict[str, Any]]]:
    attempts: dict[
        tuple[int, str],
        list[tuple[int, dict[str, Any]]],
    ] = {}
    for index, record in enumerate(records):
        if (
            record.get("experiment_id") != experiment_id
            or record.get("method") != method
            or record.get("checkpoint") != checkpoint
            or record.get("profile") != profile
            or record.get("sample_kind") != "measured"
            or not isinstance(record.get("sample_index"), int)
            or not isinstance(record.get("attempt_id"), str)
        ):
            continue
        key = (record["sample_index"], record["attempt_id"])
        attempts.setdefault(key, []).append((index, record))

    latest: dict[int, tuple[int, list[dict[str, Any]]]] = {}
    for (sample_index, _attempt_id), indexed in attempts.items():
        last_index = max(index for index, _ in indexed)
        current = latest.get(sample_index)
        rows = [record for _, record in sorted(indexed)]
        if current is None or last_index > current[0]:
            latest[sample_index] = (last_index, rows)
    return {
        sample_index: rows
        for sample_index, (_last_index, rows) in latest.items()
    }


def inspect_attempt(
    records: Sequence[dict[str, Any]],
    *,
    strict_round_trip: bool,
) -> tuple[dict[str, Any] | None, str]:
    operations: dict[str, dict[str, Any]] = {}
    for record in records:
        operation = record.get("operation")
        if operation in ("compress", "decompress"):
            operations[operation] = record
    if set(operations) != {"compress", "decompress"}:
        return None, "latest attempt has not completed both phases"
    phases = (operations["compress"], operations["decompress"])
    if any(record.get("status") != "ok" for record in phases):
        return None, "latest attempt contains a failed phase"

    if strict_round_trip:
        if any(record.get("round_trip_check_performed") is False for record in phases):
            return None, "strict source explicitly skipped its round-trip check"
        if any(record.get("round_trip_exact") is not True for record in phases):
            return None, "strict round-trip check is absent or failed"
        round_trip_provenance = (
            "explicit_check_flag"
            if all(
                record.get("round_trip_check_performed") is True
                for record in phases
            )
            else "legacy_strict_log_with_exact_true"
        )
    else:
        if any(
            record.get("round_trip_check_performed") is not False
            for record in phases
        ):
            return None, "fast source lacks explicit check=false provenance"
        if any(record.get("round_trip_exact") is not None for record in phases):
            return None, "fast source incorrectly claims a round-trip result"
        if any(record.get("exactness_scope") != "not_checked" for record in phases):
            return None, "fast source lacks exactness_scope=not_checked"
        round_trip_provenance = "explicit_fast_mode_not_checked"

    compress = operations["compress"]
    decompress = operations["decompress"]
    source_bytes = compress.get("logical_uncompressed_bytes")
    decompress_source_bytes = decompress.get("logical_uncompressed_bytes")
    archive_bytes = compress.get("output_bytes")
    compress_seconds = compress.get("phase_makespan_seconds")
    decompress_seconds = decompress.get("phase_makespan_seconds")
    if (
        not isinstance(source_bytes, int)
        or source_bytes <= 0
        or decompress_source_bytes != source_bytes
        or not isinstance(archive_bytes, int)
        or archive_bytes <= 0
        or not isinstance(compress_seconds, (int, float))
        or compress_seconds <= 0
        or not isinstance(decompress_seconds, (int, float))
        or decompress_seconds <= 0
    ):
        return None, "latest attempt lacks valid timing or byte counts"
    return {
        "compress_seconds": float(compress_seconds),
        "decompress_seconds": float(decompress_seconds),
        "compress_bytes_per_second": source_bytes / compress_seconds,
        "decompress_bytes_per_second": source_bytes / decompress_seconds,
        "compression_ratio_x": source_bytes / archive_bytes,
        "round_trip_provenance": round_trip_provenance,
    }, ""


def missing_row(
    *,
    spec: InputSpec,
    method: str,
    reason: str,
    experiment_id: str = "",
    checkpoint: str = "",
    profile: str = "",
    cache_mode: str = "",
    expected_n: int | None = None,
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = "strict_exact" if spec.strict_round_trip else "skip_requested"
    status = (
        "pending (strict check required)"
        if spec.strict_round_trip
        else "not checked (configured)"
    )
    return {
        "status": "missing",
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "checkpoint": checkpoint,
        "profile": profile,
        "cache_mode": cache_mode,
        **(configuration or process_configuration(method, None, None)),
        "compress_seconds": None,
        "decompress_seconds": None,
        "compress_gb_per_second": None,
        "decompress_gb_per_second": None,
        "compress_gib_per_second": None,
        "decompress_gib_per_second": None,
        "compression_ratio_x": None,
        "round_trip_policy": policy,
        "round_trip_check_performed": None,
        "round_trip_check_status": status,
        "round_trip_provenance": "",
        "n": 0,
        "expected_n": expected_n,
        "measurement_label": (
            f"missing (0/{expected_n} complete)"
            if expected_n is not None
            else "missing"
        ),
        "missing_reason": reason,
        "source_result_dir": str(spec.directory),
        "experiment_id": experiment_id,
    }


def rows_for_input(spec: InputSpec) -> list[dict[str, Any]]:
    path = environment_path(spec.directory)
    if path is None:
        return [
            missing_row(
                spec=spec,
                method=method,
                reason="result directory or environment snapshot is missing",
            )
            for method in spec.methods
        ]
    environment = read_json(path)
    experiment_id, protocol, resource_matrix, checkpoint_data = (
        require_environment_shape(environment, path)
    )
    profiles = protocol.get("profiles")
    repetitions = protocol.get("repetitions")
    methods = protocol.get("methods")
    cache_mode = protocol.get("cache_mode")
    if (
        not isinstance(profiles, list)
        or len(profiles) != 1
        or not isinstance(profiles[0], str)
        or not isinstance(repetitions, int)
        or repetitions < 1
        or not isinstance(methods, list)
        or not isinstance(cache_mode, str)
    ):
        raise PaperTableError(f"unsupported throughput protocol: {path}")
    profile = profiles[0]
    checkpoint = checkpoint_data["name"]
    shard_count = len(checkpoint_data["files"])
    records = read_jsonl_snapshot(
        spec.directory / "raw" / "throughput-runs.jsonl"
    )
    rows = []
    for method in spec.methods:
        resource = resource_matrix.get(f"{profile}/{method}")
        configuration = process_configuration(method, resource, shard_count)
        common = {
            "experiment_id": experiment_id,
            "checkpoint": checkpoint,
            "profile": profile,
            "cache_mode": cache_mode,
            "expected_n": repetitions,
            "configuration": configuration,
        }
        if method not in methods:
            rows.append(
                missing_row(
                    spec=spec,
                    method=method,
                    reason="method is absent from the environment plan",
                    **common,
                )
            )
            continue
        attempts = latest_attempts_for_method(
            records,
            experiment_id=experiment_id,
            method=method,
            checkpoint=checkpoint,
            profile=profile,
        )
        measured = []
        reasons = []
        for sample_index in range(repetitions):
            attempt = attempts.get(sample_index)
            if attempt is None:
                reasons.append(f"repetition {sample_index}: not started")
                continue
            measurement, reason = inspect_attempt(
                attempt,
                strict_round_trip=spec.strict_round_trip,
            )
            if measurement is None:
                reasons.append(f"repetition {sample_index}: {reason}")
            else:
                measured.append(measurement)
        if len(measured) != repetitions:
            rows.append(
                missing_row(
                    spec=spec,
                    method=method,
                    reason="; ".join(reasons) or "incomplete measured repetitions",
                    **common,
                )
            )
            continue

        compress_seconds = median(
            item["compress_seconds"] for item in measured
        )
        decompress_seconds = median(
            item["decompress_seconds"] for item in measured
        )
        compress_bps = median(
            item["compress_bytes_per_second"] for item in measured
        )
        decompress_bps = median(
            item["decompress_bytes_per_second"] for item in measured
        )
        if repetitions == 1:
            measurement_label = "n=1; single-run peak-throughput estimate"
        else:
            measurement_label = f"n={repetitions}; median of repetitions"
        provenance_values = {
            item["round_trip_provenance"] for item in measured
        }
        rows.append(
            {
                "status": "complete",
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "checkpoint": checkpoint,
                "profile": profile,
                "cache_mode": cache_mode,
                **configuration,
                "compress_seconds": compress_seconds,
                "decompress_seconds": decompress_seconds,
                "compress_gb_per_second": compress_bps / 1_000_000_000,
                "decompress_gb_per_second": (
                    decompress_bps / 1_000_000_000
                ),
                "compress_gib_per_second": compress_bps / 1024**3,
                "decompress_gib_per_second": decompress_bps / 1024**3,
                "compression_ratio_x": median(
                    item["compression_ratio_x"] for item in measured
                ),
                "round_trip_policy": (
                    "strict_exact"
                    if spec.strict_round_trip
                    else "skip_requested"
                ),
                "round_trip_check_performed": spec.strict_round_trip,
                "round_trip_check_status": (
                    "exact"
                    if spec.strict_round_trip
                    else "not_checked"
                ),
                "round_trip_provenance": ",".join(
                    sorted(provenance_values)
                ),
                "n": repetitions,
                "expected_n": repetitions,
                "measurement_label": measurement_label,
                "missing_reason": "",
                "source_result_dir": str(spec.directory),
                "experiment_id": experiment_id,
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(
            {
                key: "" if row.get(key) is None else row.get(key)
                for key in CSV_COLUMNS
            }
            for row in rows
        )


def format_number(value: Any, digits: int = 2) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value:.{digits}f}"


def markdown_text(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def write_markdown(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    headers = (
        "Method",
        "Process/thread configuration",
        "Compress (s)",
        "Decompress (s)",
        "Compress (GB/s)",
        "Decompress (GB/s)",
        "Compress (GiB/s)",
        "Decompress (GiB/s)",
        "Ratio",
        "Round-trip check",
        "Evidence",
        "Status",
    )
    lines = [
        "# Llama-3.1-70B BF16 throughput",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        ratio = (
            f"{row['compression_ratio_x']:.3f}×"
            if isinstance(row.get("compression_ratio_x"), (int, float))
            else "—"
        )
        values = (
            row["method_label"],
            row["process_thread_config"] or "—",
            format_number(row["compress_seconds"]),
            format_number(row["decompress_seconds"]),
            format_number(row["compress_gb_per_second"]),
            format_number(row["decompress_gb_per_second"]),
            format_number(row["compress_gib_per_second"]),
            format_number(row["decompress_gib_per_second"]),
            ratio,
            row["round_trip_check_status"],
            row["measurement_label"],
            row["status"],
        )
        lines.append(
            "| " + " | ".join(markdown_text(value) for value in values) + " |"
        )
    lines.extend(
        [
            "",
            "GB/s uses decimal GB (10^9 bytes); GiB/s uses 2^30 bytes. "
            "Throughput uses original checkpoint bytes divided by whole-"
            "checkpoint phase makespan.",
            "",
            "`single-run peak-throughput estimate` means n=1 under the "
            "listed practical concurrency; it is not a multi-run uncertainty "
            "estimate.",
            "",
            "## Missing or incomplete",
            "",
        ]
    )
    missing = [row for row in rows if row["status"] != "complete"]
    if missing:
        lines.extend(
            f"- {markdown_text(row['method_label'])}: "
            f"{markdown_text(row['missing_reason'])}"
            for row in missing
        )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Sources",
            "",
            *(
                f"- `{markdown_text(spec)}`"
                for spec in dict.fromkeys(
                    row["source_result_dir"] for row in rows
                )
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines))


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
    }
    return "".join(replacements.get(character, character) for character in rendered)


def write_latex(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    lines = [
        "% Generated by scripts/build_throughput_paper_table.py",
        "% Requires the booktabs package.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{llrrrrrrrlll}",
        r"\toprule",
        (
            r"Method & Process/thread configuration & C (s) & D (s) & "
            r"C (GB/s) & D (GB/s) & C (GiB/s) & D (GiB/s) & Ratio & "
            r"Round-trip & Evidence & Status \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        ratio = (
            f"{row['compression_ratio_x']:.3f}x"
            if isinstance(row.get("compression_ratio_x"), (int, float))
            else "--"
        )
        values = (
            row["method_label"],
            row["process_thread_config"] or "--",
            format_number(row["compress_seconds"]).replace("—", "--"),
            format_number(row["decompress_seconds"]).replace("—", "--"),
            format_number(row["compress_gb_per_second"]).replace("—", "--"),
            format_number(row["decompress_gb_per_second"]).replace("—", "--"),
            format_number(row["compress_gib_per_second"]).replace("—", "--"),
            format_number(row["decompress_gib_per_second"]).replace("—", "--"),
            ratio,
            row["round_trip_check_status"],
            row["measurement_label"],
            row["status"],
        )
        lines.append(
            " & ".join(latex_text(value) for value in values) + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Llama-3.1-70B BF16 whole-checkpoint throughput. "
                r"GB/s is decimal and GiB/s is binary. Rows marked as "
                r"single-run peak-throughput estimates have $n=1$.}"
            ),
            r"\label{tab:llama70-throughput}",
            r"\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def build_paper_table(
    inputs: Iterable[InputSpec],
    output: Path,
) -> list[dict[str, Any]]:
    rows = [
        row
        for spec in inputs
        for row in rows_for_input(spec)
    ]
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "throughput-paper-table.csv", rows)
    write_markdown(output / "throughput-paper-table.md", rows)
    write_latex(output / "throughput-paper-table.tex", rows)
    return rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--brevis-results",
        type=Path,
        default=ROOT / "results" / "throughput-llama70-brevis-max30",
    )
    parser.add_argument(
        "--zstd-results",
        type=Path,
        default=ROOT / "results" / "throughput-llama70-max30",
    )
    parser.add_argument(
        "--fast-baseline-results",
        type=Path,
        default=ROOT / "results" / "throughput-llama70-fast-baselines",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = build_paper_table(default_inputs(args), args.output)
    complete = sum(row["status"] == "complete" for row in rows)
    print(
        f"wrote {len(rows)} paper-table rows ({complete} complete, "
        f"{len(rows) - complete} missing) to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
