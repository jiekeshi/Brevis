import csv
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_benchmarks as bench
import specialized_baselines


def write_safetensors(path: Path, tensors: list[tuple[str, str, list[int], bytes]]) -> None:
    offset = 0
    header = {}
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

    def test_environment_does_not_merge_results_from_another_host(self):
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

    def test_summary_reuses_core_full_for_all_three_sweeps(self):
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
        self.assertEqual("60.0", figure1[0]["archive_percent"])
        self.assertEqual("1", figure2[0]["workers"])
        self.assertEqual("full", table4[0]["variant"])
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
