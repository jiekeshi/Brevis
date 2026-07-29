import csv
import importlib.util
import io
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_benchmarks as bench
import benchmark_codecs
import specialized_baselines


def write_safetensors(
    path: Path,
    tensors: list[tuple[str, str, list[int], bytes]],
    metadata: dict[str, str] | None = None,
) -> None:
    offset = 0
    header = {"__metadata__": metadata} if metadata is not None else {}
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        payload.extend(data)
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as source:
        return list(csv.DictReader(source))


class BenchmarkHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_read_only_linux_drop_caches_is_not_advertised(self):
        with (
            patch.object(bench.sys, "platform", "linux"),
            patch.object(bench.os, "geteuid", return_value=0),
            patch.object(bench.Path, "exists", return_value=True),
            patch.object(bench.os, "access", return_value=False),
        ):
            self.assertFalse(bench.cache_control_available(None))

    def test_manifest_loads_only_declared_shards(self):
        model = self.root / "model"
        model.mkdir()
        first = model / "model-00001-of-00002.safetensors"
        second = model / "model-00002-of-00002.safetensors"
        write_safetensors(first, [("a", "U8", [2], b"\x01\x02")])
        write_safetensors(second, [("b", "U8", [1], b"\x03")])
        write_safetensors(model / "consolidated.safetensors", [("x", "U8", [1], b"x")])
        manifest = {
            "name": "fixture",
            "repo_id": "test/fixture",
            "revision": "a" * 40,
            "weights": [
                {"path": first.name},
                {"path": second.name},
            ],
        }
        (model / "download-manifest.json").write_text(json.dumps(manifest))

        checkpoint = bench.load_manifest_checkpoint(model)

        self.assertEqual("fixture", checkpoint.name)
        self.assertEqual((first, second), checkpoint.files)
        self.assertNotIn(model / "consolidated.safetensors", checkpoint.files)
        with self.assertRaisesRegex(bench.BenchmarkError, "non-paper checkpoint"):
            bench.load_corpus(self.root, None)
        self.assertEqual(
            ["fixture"],
            [
                item.name
                for item in bench.load_corpus(
                    self.root,
                    None,
                    allow_custom=True,
                )
            ],
        )

    def test_tensor_exact_ignores_header_order_but_not_payload_changes(self):
        source = self.root / "source.safetensors"
        reordered = self.root / "reordered.safetensors"
        corrupt = self.root / "corrupt.safetensors"
        write_safetensors(
            source,
            [
                ("a", "U8", [2], b"\x01\x02"),
                ("b", "U8", [2], b"\x03\x04"),
            ],
        )
        write_safetensors(
            reordered,
            [
                ("b", "U8", [2], b"\x03\x04"),
                ("a", "U8", [2], b"\x01\x02"),
            ],
        )
        write_safetensors(
            corrupt,
            [
                ("b", "U8", [2], b"\x03\x05"),
                ("a", "U8", [2], b"\x01\x02"),
            ],
        )

        self.assertTrue(bench.tensor_exact(source, reordered))
        self.assertFalse(bench.byte_exact(source, reordered))
        self.assertFalse(bench.tensor_exact(source, corrupt))

    def test_brevis_commands_freeze_paper_controls(self):
        source = self.root / "model.safetensors"
        archive = self.root / "model.brv"
        restored = self.root / "restored.safetensors"
        source.write_bytes(b"x" * 128)
        archive.write_bytes(b"y" * 64)
        config = bench.BrevisConfig(
            workers=4,
            max_expansions=512,
            tensors=0,
            astar_heuristic=False,
        )

        compress = bench.command_for(
            "brevis",
            "compress",
            source,
            archive,
            4,
            Path("/bin/brevis"),
            config,
        )
        decompress = bench.command_for(
            "brevis",
            "decompress",
            archive,
            restored,
            4,
            Path("/bin/brevis"),
            config,
        )

        self.assertIn("--max-expansions", compress)
        self.assertEqual("0", compress[compress.index("--tensors") + 1])
        self.assertEqual("0", compress[compress.index("--astar-heuristic") + 1])
        self.assertNotIn("--max-expansions", decompress)

        libdeflate = bench.command_for(
            "libdeflate-1",
            "compress",
            source,
            archive,
            1,
            Path("/bin/brevis"),
            config,
        )
        self.assertEqual("libdeflate-1", libdeflate[2])
        self.assertEqual("compress", libdeflate[3])

        missing_archive = self.root / "dry-run-only.brv"
        dry_run_decompress = bench.command_for(
            "brevis",
            "decompress",
            missing_archive,
            restored,
            4,
            Path("/bin/brevis"),
            config,
        )
        self.assertEqual(str(missing_archive), dry_run_decompress[2])

    def test_measure_displays_live_compression_progress(self):
        output = self.root / "archive.brv"
        logs = self.root / "logs"
        script = (
            "import pathlib,sys,time;"
            "time.sleep(0.03);"
            "pathlib.Path(sys.argv[1]).write_bytes(b'x' * 25);"
            "time.sleep(0.05)"
        )
        display = io.StringIO()

        with redirect_stdout(display):
            measurement = bench.measure(
                [sys.executable, "-c", script, str(output)],
                logs,
                "progress",
                output,
                progress=bench.ProgressDisplay(
                    "fixture/compress",
                    source_bytes=100,
                    interval_seconds=0.01,
                ),
            )

        rendered = display.getvalue()
        self.assertIn("progress fixture/compress:", rendered)
        self.assertIn("output/input=25.0%", rendered)
        self.assertIn("done fixture/compress:", rendered)
        self.assertGreater(measurement.wall_seconds, 0)

    @unittest.skipUnless(
        shutil.which("libdeflate-gzip"),
        "libdeflate is not installed",
    )
    def test_libdeflate_round_trip(self):
        source = self.root / "source.bin"
        archive = self.root / "archive.brv"
        restored = self.root / "restored.bin"
        source.write_bytes(bytes(range(256)) * 1024)

        benchmark_codecs.CODECS["libdeflate-1"].compress(source, archive, 1)
        benchmark_codecs.CODECS["libdeflate-1"].decompress(archive, restored, 1)

        self.assertEqual(source.read_bytes(), restored.read_bytes())

    @unittest.skipUnless(
        importlib.util.find_spec("zipnn"),
        "zipnn is not installed",
    )
    def test_zipnn_round_trip_preserves_raw_fallback_tensors(self):
        source = self.root / "source.safetensors"
        archive = self.root / "archive.safetensors"
        restored = self.root / "restored.safetensors"
        write_safetensors(
            source,
            [
                ("tiny_float", "F32", [2], struct.pack("<ff", 0.0211, -0.0021)),
                ("integer", "U8", [4], b"\x01\x02\x03\x04"),
            ],
            metadata={"format": "pt"},
        )

        benchmark_codecs.CODECS["zipnn"].compress(source, archive, 2)
        benchmark_codecs.CODECS["zipnn"].decompress(archive, restored, 2)

        self.assertTrue(bench.tensor_exact(source, restored))

    @unittest.skipUnless(
        importlib.util.find_spec("zipnn"),
        "zipnn is not installed",
    )
    def test_zipnn_round_trip_preserves_absent_metadata(self):
        source = self.root / "source.safetensors"
        archive = self.root / "archive.safetensors"
        restored = self.root / "restored.safetensors"
        write_safetensors(
            source,
            [
                ("tiny_float", "F32", [2], struct.pack("<ff", 0.0211, -0.0021)),
                ("integer", "U8", [4], b"\x01\x02\x03\x04"),
            ],
        )

        benchmark_codecs.CODECS["zipnn"].compress(source, archive, 2)
        benchmark_codecs.CODECS["zipnn"].decompress(archive, restored, 2)

        source_header, _ = bench.safetensors_header(source)
        restored_header, _ = bench.safetensors_header(restored)
        self.assertNotIn("__metadata__", source_header)
        self.assertNotIn("__metadata__", restored_header)
        self.assertTrue(bench.tensor_exact(source, restored))

    def test_corpus_archive_can_be_discarded_after_verification(self):
        source = self.root / "model.safetensors"
        source.write_bytes(b"fixture")
        checkpoint = bench.Checkpoint("fixture", self.root, (source,))
        args = SimpleNamespace(
            corpus_cache="hot",
            corpus_max_expansions=1,
            corpus_tensors=32,
            discard_corpus_archives=True,
            results=self.root / "results",
            workers=4,
        )

        with patch.object(bench, "run_pair") as run_pair:
            bench.size_shard(args, object(), checkpoint, source, "zstd-9")

        self.assertFalse(run_pair.call_args.kwargs["keep_archive"])
        self.assertEqual("hot", run_pair.call_args.args[5])
        self.assertEqual(1, run_pair.call_args.args[9].max_expansions)
        self.assertEqual(32, run_pair.call_args.args[9].tensors)

    def test_analysis_sweeps_use_explicit_quick_controls(self):
        args = SimpleNamespace(
            search_budgets=(0, 32),
            pareto_workers=4,
            pareto_tensors=16,
            worker_sweep=(1, 8),
            worker_sweep_max_expansions=32,
            worker_sweep_tensors=8,
            discard_analysis_archives=True,
        )
        checkpoint = bench.Checkpoint("fixture", self.root, ())

        with patch.object(bench, "run_brevis_variant") as run_variant:
            bench.run_sweeps(args, object(), checkpoint)

        calls = run_variant.call_args_list
        self.assertEqual(
            ["budget-0", "budget-32", "workers-1", "workers-8"],
            [call.args[4] for call in calls],
        )
        self.assertEqual(
            [
                bench.BrevisConfig(4, 0, 16, True),
                bench.BrevisConfig(4, 32, 16, True),
                bench.BrevisConfig(1, 32, 8, True),
                bench.BrevisConfig(8, 32, 8, True),
            ],
            [call.args[5] for call in calls],
        )
        self.assertEqual(
            [False, False, False, False],
            [call.kwargs["keep"] for call in calls],
        )

    def test_ablation_runs_all_four_explicit_variants(self):
        args = SimpleNamespace(
            ablation_workers=4,
            ablation_max_expansions=32,
            ablation_tensors=16,
            discard_analysis_archives=False,
        )
        checkpoint = bench.Checkpoint("fixture", self.root, ())

        with patch.object(bench, "run_brevis_variant") as run_variant:
            bench.run_ablation(args, object(), checkpoint)

        calls = run_variant.call_args_list
        self.assertEqual(
            ["full", "no-phog", "no-astar", "no-phog-no-astar"],
            [call.args[4] for call in calls],
        )
        self.assertEqual(
            [
                bench.BrevisConfig(4, 32, 16, True),
                bench.BrevisConfig(4, 32, 0, True),
                bench.BrevisConfig(4, 32, 16, False),
                bench.BrevisConfig(4, 32, 0, False),
            ],
            [call.args[5] for call in calls],
        )
        self.assertTrue(all(call.kwargs["keep"] for call in calls))

    def test_discard_analysis_archives_removes_verified_kept_archive(self):
        source = self.root / "model.safetensors"
        source.write_bytes(b"fixture")
        checkpoint = bench.Checkpoint("fixture", self.root, (source,))
        results = self.root / "results"
        archive = (
            results
            / "archives"
            / "fixture"
            / "budget-32"
            / "model.budget-32.brv"
        )
        archive.parent.mkdir(parents=True)
        archive.write_bytes(b"verified")
        config = bench.BrevisConfig(4, 32, 16, True)
        provenance = {"method_version": "fixture"}
        identity = bench.operation_identity(
            checkpoint,
            source,
            "brevis",
            "compress",
            "hot",
            4,
            "pareto",
            provenance,
            config,
            "budget-32",
        )
        log = bench.ResultLog(results / "raw" / "runs.jsonl")
        log.append(
            {
                "run_id": f"{bench.run_id(identity)}-verify",
                "status": "ok",
                "exact": True,
            }
        )
        args = SimpleNamespace(
            results=results,
            run_provenance={"brevis": provenance},
            rerun=False,
            dry_run=False,
        )

        with patch.object(bench, "execute_operation") as execute:
            bench.run_brevis_variant(
                args,
                log,
                checkpoint,
                "pareto",
                "budget-32",
                config,
                keep=False,
            )

        execute.assert_not_called()
        self.assertFalse(archive.exists())

    def test_parse_args_exposes_quick_analysis_controls(self):
        arguments = [
            "run_benchmarks.py",
            "preflight",
            "--pareto-workers",
            "4",
            "--pareto-tensors",
            "16",
            "--worker-sweep",
            "1,8",
            "--worker-sweep-max-expansions",
            "32",
            "--worker-sweep-tensors",
            "8",
            "--ablation-max-expansions",
            "32",
            "--ablation-tensors",
            "16",
            "--ablation-workers",
            "4",
            "--discard-analysis-archives",
        ]

        with patch.object(sys, "argv", arguments):
            args = bench.parse_args()

        self.assertEqual((1, 8), args.worker_sweep)
        self.assertEqual(4, args.pareto_workers)
        self.assertEqual(16, args.pareto_tensors)
        self.assertEqual(32, args.worker_sweep_max_expansions)
        self.assertEqual(8, args.worker_sweep_tensors)
        self.assertEqual(32, args.ablation_max_expansions)
        self.assertEqual(16, args.ablation_tensors)
        self.assertEqual(4, args.ablation_workers)
        self.assertTrue(args.discard_analysis_archives)

    def test_one_click_launcher_exposes_progress_and_current_methods(self):
        launcher = Path(__file__).with_name("run_paper_benchmark.sh")
        result = subprocess.run(
            ["bash", str(launcher), "--help"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )

        self.assertIn("PROGRESS_INTERVAL=5", result.stdout)
        self.assertIn("PAPER_TIMING=0", result.stdout)
        script = launcher.read_text()
        self.assertIn("from run_benchmarks import GENERIC_METHODS", script)
        self.assertIn("SPECIALIZED_METHODS", script)
        self.assertNotIn("paper_methods+=(dfloat11 ecf8)", script)
        self.assertIn("import ensurepip, venv", script)

    def test_specialized_config_requires_explicit_python_environment(self):
        config = self.root / "specialized.json"
        example = json.loads(
            (
                Path(__file__).resolve().parents[1]
                / "configs"
                / "specialized-baselines.example.json"
            ).read_text()
        )
        settings = example["dfloat11"]
        settings["compress_command"][0] = "python3"
        config.write_text(json.dumps({"dfloat11": settings}))

        with self.assertRaisesRegex(bench.BenchmarkError, "environment-specific"):
            bench.load_specialized_config(config)

        settings["compress_command"][0] = sys.executable
        config.write_text(json.dumps({"dfloat11": settings}))
        self.assertIn("dfloat11", bench.load_specialized_config(config))

    def test_run_identity_is_bound_to_method_provenance(self):
        source = self.root / "model.safetensors"
        source.write_bytes(b"x")
        checkpoint = bench.Checkpoint("fixture", self.root, (source,))
        arguments = (
            checkpoint,
            source,
            "zstd-9",
            "compress",
            "hot",
            1,
            "core",
        )

        first = bench.operation_identity(
            *arguments,
            {"method_version": "zstd 1.5.6"},
        )
        second = bench.operation_identity(
            *arguments,
            {"method_version": "zstd 1.5.7"},
        )
        other_host = bench.operation_identity(
            *arguments,
            {
                "method_version": "zstd 1.5.6",
                "host": {"hostname": "other"},
            },
        )

        self.assertNotEqual(bench.run_id(first), bench.run_id(second))
        self.assertNotEqual(bench.run_id(first), bench.run_id(other_host))

    def test_progress_interval_does_not_change_execution_identity(self):
        host = {"hostname": "fixture"}
        first = SimpleNamespace(host_context=host, progress_interval=1.0)
        second = SimpleNamespace(host_context=host, progress_interval=30.0)

        self.assertEqual(
            bench.execution_provenance(first, {"zstd-9": "version"}, {}),
            bench.execution_provenance(second, {"zstd-9": "version"}, {}),
        )

    def test_specialized_baselines_require_model_assets(self):
        with self.assertRaisesRegex(SystemExit, "missing model config"):
            specialized_baselines.require_model_config(self.root)
        (self.root / "config.json").write_text("{}")
        specialized_baselines.require_model_config(self.root)
        with self.assertRaisesRegex(SystemExit, "missing tokenizer assets"):
            specialized_baselines.require_tokenizer_assets(self.root)
        (self.root / "tokenizer_config.json").write_text("{}")
        (self.root / "tokenizer.json").write_text("{}")
        specialized_baselines.require_tokenizer_assets(self.root)

    def test_environment_discards_other_hosts_and_retired_methods(self):
        results = self.root / "results"
        results.mkdir()
        (results / "environment.json").write_text(
            json.dumps(
                {
                    "host": {"hostname": "old"},
                    "method_versions": {"zstd-9": "old"},
                    "run_provenance": {
                        "zstd-9": {"host": {"hostname": "old"}}
                    },
                }
            )
        )
        current_host = {
            "hostname": "current",
            "platform": "test",
            "python": "test",
            "logical_cpus": 1,
            "physical_cores": 1,
            "ram_bytes": 1,
        }
        args = SimpleNamespace(
            results=results,
            run_provenance={
                "brevis": {
                    "host": current_host,
                    "brevis_revision": "revision",
                }
            },
            host_context=current_host,
            workers=1,
            shard_jobs=1,
            allow_custom_corpus=False,
            cold_available=False,
            drop_caches_command=None,
        )

        bench.write_environment(args, {"brevis": "revision"}, [])

        environment = json.loads((results / "environment.json").read_text())
        self.assertEqual({"brevis"}, set(environment["run_provenance"]))
        self.assertEqual({"brevis"}, set(environment["method_versions"]))

        environment["run_provenance"]["libdeflate-6"] = {
            "host": current_host
        }
        environment["method_versions"]["libdeflate-6"] = "retired"
        (results / "environment.json").write_text(json.dumps(environment))

        bench.write_environment(args, {"brevis": "revision"}, [])

        migrated = json.loads((results / "environment.json").read_text())
        self.assertEqual({"brevis"}, set(migrated["run_provenance"]))
        self.assertEqual({"brevis"}, set(migrated["method_versions"]))

    def test_summary_uses_only_explicit_analysis_stage_records(self):
        results = self.root / "results"
        log = bench.ResultLog(results / "raw" / "runs.jsonl")
        log.append(
            {
                "run_id": "full",
                "attempt_id": "old",
                "status": "ok",
                "stage": "core",
                "checkpoint": "qwen2.5-7b-local",
                "shard": "model.safetensors",
                "method": "brevis",
                "operation": "compress",
                "cache": "hot",
                "workers": 1,
                "max_expansions": 512,
                "source_bytes": 100,
                "output_bytes": 70,
                "wall_seconds": 2,
                "peak_rss_bytes": 1024,
            }
        )
        log.append(
            {
                "run_id": "full",
                "attempt_id": "latest",
                "status": "ok",
                "stage": "core",
                "checkpoint": "qwen2.5-7b-local",
                "shard": "model.safetensors",
                "method": "brevis",
                "operation": "compress",
                "cache": "hot",
                "workers": 1,
                "max_expansions": 512,
                "source_bytes": 100,
                "output_bytes": 60,
                "wall_seconds": 1,
                "peak_rss_bytes": 1024,
            }
        )
        log.append(
            {
                "run_id": "unverified",
                "attempt_id": "unverified-attempt",
                "status": "ok",
                "stage": "core",
                "checkpoint": "qwen2.5-7b-local",
                "shard": "model.safetensors",
                "method": "zstd-9",
                "operation": "compress",
                "cache": "hot",
                "workers": 1,
                "source_bytes": 100,
                "output_bytes": 50,
                "wall_seconds": 1,
                "peak_rss_bytes": 1024,
            }
        )
        log.append(
            {
                "run_id": "full-verify",
                "status": "ok",
                "stage": "core",
                "checkpoint": "qwen2.5-7b-local",
                "shard": "model.safetensors",
                "method": "brevis",
                "operation": "verify",
                "cache": "hot",
                "workers": 1,
                "exact": True,
                "verified_attempts": [["full", "latest"]],
            }
        )
        analysis_rows = (
            ("pareto-full", "pareto", "budget-512", 1, 512, 32, 70),
            ("workers-one", "workers", "workers-1", 1, 32, 16, 80),
            ("ablation-full", "ablation", "full", 4, 32, 16, 90),
        )
        for (
            identifier,
            stage,
            variant,
            workers,
            max_expansions,
            calibration_tensors,
            output_bytes,
        ) in analysis_rows:
            log.append(
                {
                    "run_id": identifier,
                    "attempt_id": identifier,
                    "status": "ok",
                    "stage": stage,
                    "checkpoint": "qwen2.5-7b-local",
                    "shard": "model.safetensors",
                    "method": "brevis",
                    "operation": "compress",
                    "cache": "hot",
                    "variant": variant,
                    "workers": workers,
                    "max_expansions": max_expansions,
                    "calibration_tensors": calibration_tensors,
                    "phog": calibration_tensors > 0,
                    "astar_heuristic": True,
                    "source_bytes": 100,
                    "output_bytes": output_bytes,
                    "wall_seconds": 1,
                    "peak_rss_bytes": 1024,
                }
            )
            log.append(
                {
                    "run_id": f"{identifier}-verify",
                    "status": "ok",
                    "stage": stage,
                    "checkpoint": "qwen2.5-7b-local",
                    "shard": "model.safetensors",
                    "method": "brevis",
                    "operation": "verify",
                    "cache": "hot",
                    "workers": workers,
                    "exact": True,
                    "verified_attempts": [[identifier, identifier]],
                }
            )

        bench.summarize(results)

        figure1 = read_csv(
            results / "tables" / "figure1-archive-size-vs-time.csv"
        )
        figure2 = read_csv(
            results / "tables" / "figure2-throughput-rss-vs-workers.csv"
        )
        table4 = read_csv(results / "tables" / "table4-ablation.csv")
        table3 = read_csv(results / "tables" / "table3-end-to-end.csv")
        self.assertEqual("budget-512", figure1[0]["variant"])
        self.assertEqual("32", figure1[0]["calibration_tensors"])
        self.assertEqual("70.0", figure1[0]["archive_percent"])
        self.assertEqual("1", figure2[0]["workers"])
        self.assertEqual("16", figure2[0]["calibration_tensors"])
        self.assertEqual("80.0", figure2[0]["archive_percent"])
        self.assertEqual("full", table4[0]["variant"])
        self.assertEqual("4", table4[0]["workers"])
        self.assertEqual("90.0", table4[0]["archive_percent"])
        self.assertEqual({"brevis"}, {row["method"] for row in table3})

    def test_summary_uses_only_current_method_provenance(self):
        results = self.root / "results"
        results.mkdir()
        current = {"method_version": "current"}
        (results / "environment.json").write_text(
            json.dumps(
                {
                    "method_versions": {"brevis": "current"},
                    "run_provenance": {"brevis": current},
                    "corpus": [],
                }
            )
        )
        log = bench.ResultLog(results / "raw" / "runs.jsonl")
        for run_id, provenance, output_bytes in (
            ("old", {"method_version": "old"}, 20),
            ("current", current, 60),
        ):
            log.append(
                {
                    "run_id": run_id,
                    "attempt_id": run_id,
                    "status": "ok",
                    "stage": "core",
                    "checkpoint": "fixture",
                    "shard": "model.safetensors",
                    "method": "brevis",
                    "operation": "compress",
                    "cache": "hot",
                    "workers": 1,
                    "max_expansions": 512,
                    "provenance": provenance,
                    "source_bytes": 100,
                    "output_bytes": output_bytes,
                    "wall_seconds": 1,
                    "peak_rss_bytes": 1024,
                }
            )
            log.append(
                {
                    "run_id": f"{run_id}-verify",
                    "status": "ok",
                    "stage": "core",
                    "checkpoint": "fixture",
                    "shard": "model.safetensors",
                    "method": "brevis",
                    "operation": "verify",
                    "cache": "hot",
                    "workers": 1,
                    "provenance": provenance,
                    "exact": True,
                    "verified_attempts": [[run_id, run_id]],
                }
            )

        bench.summarize(results)

        table = read_csv(results / "tables" / "table3-end-to-end.csv")
        self.assertEqual("60.0", table[0]["archive_percent"])


if __name__ == "__main__":
    unittest.main()
