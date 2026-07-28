#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZIG_VERSION=0.16.0

usage() {
  cat <<'EOF'
Usage:
  scripts/run_paper_benchmark.sh MODELS_ROOT CORE_MODEL RESULTS [SPECIALIZED_CONFIG]

Environment:
  PROGRESS_INTERVAL=5     Compression heartbeat frequency; 0 disables it.
  PAPER_TIMING=0          Set to 1 to force PROGRESS_INTERVAL=0.
  DEADLINE_HOURS=12       Overall benchmark deadline.
  WORKERS=                Override worker and shard-job counts.
  DOWNLOAD_WORKERS=8      Concurrent checkpoint downloads.
  SKIP_MODEL_DOWNLOAD=0   Set to 1 when the frozen corpus is already present.
  SKIP_SYSTEM_INSTALL=0   Set to 1 when system tools are already installed.
  DROP_CACHES_COMMAND=    Non-root cache-drop command, if configured.

Example:
  scripts/run_paper_benchmark.sh \
    /data/brevis-checkpoints /data/Qwen2.5-7B /data/brevis-results
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if (( $# < 3 || $# > 4 )); then
  usage >&2
  exit 2
fi

absolute_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PWD" "$1" ;;
  esac
}

MODELS_ROOT=$(absolute_path "$1")
CORE_MODEL=$(absolute_path "$2")
RESULTS=$(absolute_path "$3")
SPECIALIZED_CONFIG=${4:+$(absolute_path "$4")}
PROGRESS_INTERVAL=${PROGRESS_INTERVAL:-5}
DEADLINE_HOURS=${DEADLINE_HOURS:-12}
DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS:-8}

as_root() {
  if (( EUID == 0 )); then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    echo "Need root or sudo to install system dependencies." >&2
    exit 1
  fi
}

install_system_tools() {
  if [[ "${SKIP_SYSTEM_INSTALL:-0}" == 1 ]]; then
    return
  fi
  local missing=()
  local tool
  for tool in python3 curl git xz zstd lz4 libdeflate-gzip; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
  done
  python3 -c 'import ensurepip, venv' >/dev/null 2>&1 ||
    missing+=(python3-venv)
  (( ${#missing[@]} == 0 )) && return

  echo "Installing missing system tools: ${missing[*]}"
  case "$(uname -s)" in
    Linux)
      if command -v apt-get >/dev/null 2>&1; then
        as_root apt-get update
        as_root env DEBIAN_FRONTEND=noninteractive apt-get install -y \
          ca-certificates curl git xz-utils python3 python3-pip python3-venv \
          zstd lz4 libdeflate-tools libsnappy-dev
      elif command -v dnf >/dev/null 2>&1; then
        as_root dnf install -y \
          ca-certificates curl git xz python3 python3-pip \
          zstd lz4 libdeflate-utils snappy-devel
      else
        echo "Unsupported Linux package manager; install: ${missing[*]}" >&2
        exit 1
      fi
      ;;
    Darwin)
      command -v brew >/dev/null 2>&1 || {
        echo "Homebrew is required to install: ${missing[*]}" >&2
        exit 1
      }
      brew install python curl git xz zstd lz4 libdeflate snappy
      ;;
    *)
      echo "Unsupported OS; install: ${missing[*]}" >&2
      exit 1
      ;;
  esac
}

ensure_zig() {
  if command -v zig >/dev/null 2>&1 && [[ "$(zig version)" == "$ZIG_VERSION" ]]; then
    return
  fi

  local os arch target checksum
  case "$(uname -s)" in
    Linux) os=linux ;;
    Darwin) os=macos ;;
    *) echo "No Zig $ZIG_VERSION binary for this OS." >&2; exit 1 ;;
  esac
  case "$(uname -m)" in
    x86_64|amd64) arch=x86_64 ;;
    arm64|aarch64) arch=aarch64 ;;
    *) echo "No Zig $ZIG_VERSION binary for this architecture." >&2; exit 1 ;;
  esac
  target="$arch-$os"
  case "$target" in
    x86_64-linux) checksum=70e49664a74374b48b51e6f3fdfbf437f6395d42509050588bd49abe52ba3d00 ;;
    aarch64-linux) checksum=ea4b09bfb22ec6f6c6ceac57ab63efb6b46e17ab08d21f69f3a48b38e1534f17 ;;
    x86_64-macos) checksum=0387557ed1877bc6a2e1802c8391953baddba76081876301c522f52977b52ba7 ;;
    aarch64-macos) checksum=b23d70deaa879b5c2d486ed3316f7eaa53e84acf6fc9cc747de152450d401489 ;;
  esac

  local zig_dir="$ROOT/.tools/zig-$target-$ZIG_VERSION"
  if [[ ! -x "$zig_dir/zig" ]]; then
    local archive
    archive="$(mktemp "${TMPDIR:-/tmp}/zig-$ZIG_VERSION.XXXXXX.tar.xz")"
    trap 'rm -f "${archive:-}"' EXIT
    curl -fL \
      "https://ziglang.org/download/$ZIG_VERSION/zig-$target-$ZIG_VERSION.tar.xz" \
      -o "$archive"
    if command -v sha256sum >/dev/null 2>&1; then
      printf '%s  %s\n' "$checksum" "$archive" | sha256sum -c -
    else
      [[ "$(shasum -a 256 "$archive" | awk '{print $1}')" == "$checksum" ]]
    fi
    mkdir -p "$ROOT/.tools"
    tar -xJf "$archive" -C "$ROOT/.tools"
    rm -f "$archive"
    trap - EXIT
  fi
  export PATH="$zig_dir:$PATH"
}

install_system_tools
ensure_zig

VENV=${BREVIS_BENCH_VENV:-"$ROOT/.venv-benchmark"}
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install -U pip
python -m pip install \
  huggingface_hub hf-xet \
  zipnn==0.5.4 python-snappy==0.7.3 safetensors torch

cd "$ROOT"

validate_specialized_config() {
  python - "$1" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
for method, settings in config.items():
    for name in ("version_command", "compress_command", "validate_command"):
        command = settings.get(name)
        executable = Path(command[0]) if command else None
        if (
            executable
            and executable.name in {"python", "python3"}
            and not executable.is_absolute()
        ):
            sys.exit(
                f"{method}.{name} must use an absolute environment-specific "
                "Python path or an explicit environment runner"
            )
PY
}

paper_methods=(brevis zstd-9 zipnn lz4-hc-9 libdeflate-1 snappy)
benchmark_args=()
if [[ -n "$SPECIALIZED_CONFIG" ]]; then
  [[ -f "$SPECIALIZED_CONFIG" ]] || {
    echo "SPECIALIZED_CONFIG does not exist: $SPECIALIZED_CONFIG" >&2
    exit 1
  }
  validate_specialized_config "$SPECIALIZED_CONFIG"
  paper_methods+=(dfloat11 ecf8)
  benchmark_args+=(--specialized-config "$SPECIALIZED_CONFIG")
fi

if [[ ! -e "$CORE_MODEL" ]]; then
  echo "CORE_MODEL does not exist: $CORE_MODEL" >&2
  exit 1
fi
if [[ "${SKIP_MODEL_DOWNLOAD:-0}" != 1 ]]; then
  python -u scripts/download_benchmark_checkpoints.py \
    --output "$MODELS_ROOT" \
    --workers "$DOWNLOAD_WORKERS"
elif [[ ! -d "$MODELS_ROOT" ]]; then
  echo "MODELS_ROOT does not exist: $MODELS_ROOT" >&2
  exit 1
fi

if [[ "${PAPER_TIMING:-0}" == 1 ]]; then
  PROGRESS_INTERVAL=0
fi
if [[ -n "${WORKERS:-}" ]]; then
  benchmark_args+=(--workers "$WORKERS" --shard-jobs "$WORKERS")
fi
if [[ -n "${DROP_CACHES_COMMAND:-}" ]]; then
  benchmark_args+=(--drop-caches-command "$DROP_CACHES_COMMAND")
fi

mkdir -p "$RESULTS"
echo "Starting benchmark; console log: $RESULTS/benchmark-console.log"
python -u scripts/run_benchmarks.py all \
  --models-root "$MODELS_ROOT" \
  --core-model "$CORE_MODEL" \
  --results "$RESULTS" \
  --methods "${paper_methods[@]}" \
  --deadline-hours "$DEADLINE_HOURS" \
  --progress-interval "$PROGRESS_INTERVAL" \
  "${benchmark_args[@]}" 2>&1 | tee -a "$RESULTS/benchmark-console.log"
