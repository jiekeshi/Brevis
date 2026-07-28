#!/usr/bin/env python3
"""Normalize the official DFloat11 and ECF8 model converters for the harness."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


LLM_LINEAR_PATHS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
DFLOAT11_MODEL_CLASSES = {"LlamaForCausalLM", "Qwen3ForCausalLM"}


def run_dfloat11(
    source: Path,
    output: Path,
    workers: int,
    validate: bool,
) -> None:
    os.environ["OMP_NUM_THREADS"] = str(workers)
    import torch
    from dfloat11 import compress_model
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(workers)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    if model.__class__.__name__ not in DFLOAT11_MODEL_CLASSES:
        raise SystemExit(f"unsupported DFloat11 model: {model.__class__.__name__}")
    compress_model(
        model=model,
        pattern_dict={
            r"model\.embed_tokens": (),
            r"model\.layers\.\d+": LLM_LINEAR_PATHS,
            r"lm_head": (),
        },
        save_path=str(output),
        save_single_file=False,
        check_correctness=validate,
    )


def ecf8_output_path(source: Path, root: Path) -> Path:
    parts = str(source).split("/")
    parts[0] = "DFloat11"
    return root / f"models--{'--'.join(parts)}-DF6.5"


def run_ecf8(
    source: Path,
    output: Path,
    workers: int,
    upstream: Path,
    validate: bool,
) -> None:
    scripts = upstream / "scripts"
    if not (scripts / "compress.py").is_file():
        raise SystemExit(f"invalid ECF8 checkout: {upstream}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".ecf8-",
        dir=output.parent,
    ) as temporary:
        temporary_path = Path(temporary)
        model_root = temporary_path / "models"
        config = temporary_path / "config.toml"
        config.write_text(
            f"[download]\ndfloat_dir = {json.dumps(str(model_root))}\n"
        )
        command = [
            sys.executable,
            "compress.py",
            "--repo_id",
            str(source),
            "--config",
            str(config),
            "--save_model",
            "--n_processes",
            str(workers),
        ]
        if validate:
            command.append("--validate_cuda")
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            filter(None, (str(upstream), environment.get("PYTHONPATH")))
        )
        subprocess.run(command, cwd=scripts, env=environment, check=True)
        converted = ecf8_output_path(source, model_root)
        shutil.rmtree(converted / "cache")
        converted.rename(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("dfloat11", "ecf8"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--validate-cuda", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if args.method == "dfloat11":
        run_dfloat11(source, output, args.workers, args.validate_cuda)
    elif args.upstream is None:
        parser.error("ecf8 requires --upstream")
    else:
        run_ecf8(
            source,
            output,
            args.workers,
            args.upstream.expanduser().resolve(),
            args.validate_cuda,
        )


if __name__ == "__main__":
    main()
