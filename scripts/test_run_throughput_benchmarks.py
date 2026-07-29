import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_benchmarks as base
import run_throughput_benchmarks as throughput


class ThroughputHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_environment(
        self,
        *,
        results,
        experiment_id,
        checkpoint,
        profile,
        method,
        resource,
        repetitions,
        cache_mode="unconditioned",
        round_trip_check_performed=True,
    ):
        descriptor = throughput.checkpoint_descriptor(checkpoint)
        verification = (
            throughput.exactness_protocol(method)
            if round_trip_check_performed
            else throughput.skipped_exactness_protocol()
        )
        environment = {
            "schema_version": throughput.SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "checkpoints": [descriptor],
            "protocol": {
                "profiles": [profile],
                "methods": [method],
                "cache_mode": cache_mode,
                "repetitions": repetitions,
                "round_trip_check": {
                    "performed": round_trip_check_performed,
                    "strict_check_is_default": True,
                    "skip_requested": not round_trip_check_performed,
                    "protocols": {method: verification},
                },
            },
            "resource_matrix": {
                f"{profile}/{method}": asdict(resource),
            },
        }
        throughput.write_environment(results, environment)

    def test_distribution_uses_documented_type7_quartiles(self):
        stats = throughput.distribution([1.0, 2.0, 3.0, 4.0])

        self.assertEqual(4, stats["n"])
        self.assertEqual(2.5, stats["median"])
        self.assertEqual(1.75, stats["q1"])
        self.assertEqual(3.25, stats["q3"])
        self.assertEqual(1.5, stats["iqr"])

    def test_resource_profiles_distinguish_zipnn_threads_and_processes(self):
        common = {
            "matched_workers": 8,
            "practical_workers": 32,
            "brevis_workers": 32,
            "zipnn_workers": 16,
            "cpu_affinity": None,
        }
        threaded = throughput.resolve_resource(
            "zipnn",
            "practical",
            zipnn_execution="threads",
            **common,
        )
        processes = throughput.resolve_resource(
            "zipnn",
            "practical",
            zipnn_execution="processes",
            **common,
        )
        matched_brevis = throughput.resolve_resource(
            "brevis",
            "resource-matched",
            zipnn_execution="threads",
            **common,
        )
        matched_zstd = throughput.resolve_resource(
            "zstd-9",
            "resource-matched",
            zipnn_execution="threads",
            **common,
        )

        self.assertEqual("single_process_internal_threads", threaded.execution_model)
        self.assertEqual(1, threaded.shard_jobs)
        self.assertEqual(16, threaded.codec_workers)
        self.assertEqual(16, threaded.declared_cpu_slots)
        self.assertEqual(
            {
                "torch_intra_op_threads": 1,
                "torch_inter_op_threads": 1,
                "zipnn_native_threads": 16,
            },
            threaded.runtime_controls,
        )

        self.assertEqual(
            "one_thread_process_pool_across_shards",
            processes.execution_model,
        )
        self.assertEqual(16, processes.shard_jobs)
        self.assertEqual(1, processes.codec_workers)
        self.assertEqual(16, processes.declared_cpu_slots)
        self.assertEqual(
            3,
            processes.provenance(shard_count=3)["effective_process_concurrency"],
        )

        self.assertEqual(8, matched_brevis.codec_workers)
        self.assertEqual(1, matched_brevis.shard_jobs)
        self.assertEqual(8, matched_zstd.shard_jobs)
        self.assertEqual(1, matched_zstd.codec_workers)
        self.assertEqual(
            matched_brevis.declared_cpu_slots,
            matched_zstd.declared_cpu_slots,
        )

    def test_warmups_repetitions_phase_makespan_and_exactness(self):
        first = self.root / "first.bin"
        second = self.root / "second.bin"
        first.write_bytes(b"a" * 4096)
        second.write_bytes(b"b" * 8192)
        checkpoint = base.Checkpoint(
            name="fixture",
            directory=self.root,
            files=(first, second),
        )
        results = self.root / "results"
        resource = throughput.resolve_resource(
            "brevis",
            "resource-matched",
            matched_workers=2,
            practical_workers=2,
            brevis_workers=2,
            zipnn_workers=2,
            zipnn_execution="threads",
            cpu_affinity=None,
            brevis_execution="processes",
        )
        self.assertEqual(2, resource.shard_jobs)
        self.assertEqual(1, resource.codec_workers)
        self.assertEqual(
            "one_worker_brevis_process_pool_across_shards",
            resource.execution_model,
        )

        def copy_builder(
            _method,
            _operation,
            source,
            output,
            _resource,
            _brevis_bin,
            _brevis_config,
        ):
            program = (
                "import pathlib,shutil,sys,time;"
                "time.sleep(0.08);"
                "pathlib.Path(sys.argv[2]).parent.mkdir(parents=True,exist_ok=True);"
                "shutil.copyfile(sys.argv[1],sys.argv[2])"
            )
            return [sys.executable, "-c", program, str(source), str(output)]

        records = throughput.execute_schedule(
            results=results,
            checkpoints=[checkpoint],
            profiles=["resource-matched"],
            methods=["brevis"],
            resources={("resource-matched", "brevis"): resource},
            experiment_id="fixture-experiment",
            warmups=1,
            repetitions=3,
            cache_mode="unconditioned",
            drop_caches_command=None,
            brevis_bin=Path("/unused/brevis"),
            brevis_config=base.BrevisConfig(workers=1),
            keep_artifacts=False,
            rerun=False,
            order_seed=7,
            command_builder=copy_builder,
        )

        self.assertEqual(8, len(records))
        self.assertEqual(
            ["compress", "decompress"] * 4,
            [record["operation"] for record in records],
        )
        self.assertTrue(all(record["round_trip_exact"] for record in records))
        self.assertTrue(
            all(record["round_trip_check_performed"] for record in records)
        )
        self.assertTrue(
            all(
                record["execution_model"]
                == "one_worker_brevis_process_pool_across_shards"
                for record in records
            )
        )
        for record in records:
            command_wall_sum = record[
                "sum_command_wall_seconds_diagnostic_only"
            ]
            self.assertLess(
                record["phase_makespan_seconds"],
                command_wall_sum * 0.8,
            )

        raw = base.read_jsonl(results / "raw" / "throughput-runs.jsonl")
        self.assertEqual(12, len(raw))
        for pair_id in {record["pair_id"] for record in records}:
            pair = [record for record in records if record["pair_id"] == pair_id]
            self.assertEqual(1, len({record["attempt_id"] for record in pair}))
        self.write_environment(
            results=results,
            experiment_id="fixture-experiment",
            checkpoint=checkpoint,
            profile="resource-matched",
            method="brevis",
            resource=resource,
            repetitions=3,
        )
        rows = throughput.summarize(results, "fixture-experiment")
        self.assertEqual(2, len(rows))
        self.assertEqual(
            {"compress", "decompress"},
            {row["operation"] for row in rows},
        )
        self.assertTrue(all(row["n"] == 3 for row in rows))
        self.assertTrue(all(row["all_round_trips_exact"] for row in rows))
        self.assertTrue(
            all(row["round_trip_check_performed"] for row in rows)
        )
        compression = next(row for row in rows if row["operation"] == "compress")
        self.assertEqual(1.0, compression["median_compression_ratio_x"])

        payload = json.loads(
            (results / "tables" / "throughput-summary.json").read_text()
        )
        self.assertTrue(payload["statistics"]["warmups_excluded"])
        self.assertIn("makespan", payload["statistics"]["time_basis"])

        work_files = [
            path
            for path in (results / "work").rglob("*")
            if path.is_file()
        ]
        self.assertEqual([], work_files)

        throughput.execute_schedule(
            results=results,
            checkpoints=[checkpoint],
            profiles=["resource-matched"],
            methods=["brevis"],
            resources={("resource-matched", "brevis"): resource},
            experiment_id="fixture-experiment",
            warmups=1,
            repetitions=3,
            cache_mode="unconditioned",
            drop_caches_command=None,
            brevis_bin=Path("/unused/brevis"),
            brevis_config=base.BrevisConfig(workers=1),
            keep_artifacts=False,
            rerun=False,
            order_seed=7,
            command_builder=copy_builder,
        )
        self.assertEqual(
            12,
            len(base.read_jsonl(results / "raw" / "throughput-runs.jsonl")),
        )

    def test_fast_mode_times_both_phases_without_claiming_exactness(self):
        source = self.root / "fast-source.bin"
        source.write_bytes(b"fast-throughput-payload")
        checkpoint = base.Checkpoint("fast-fixture", self.root, (source,))
        descriptor = throughput.checkpoint_descriptor(checkpoint)
        results = self.root / "fast-results"
        resource = throughput.resolve_resource(
            "brevis",
            "single-core",
            matched_workers=1,
            practical_workers=1,
            brevis_workers=1,
            zipnn_workers=1,
            zipnn_execution="threads",
            cpu_affinity=None,
        )

        def copy_builder(
            _method,
            _operation,
            input_path,
            output_path,
            _resource,
            _brevis_bin,
            _brevis_config,
        ):
            program = (
                "import pathlib,shutil,sys;"
                "pathlib.Path(sys.argv[2]).parent.mkdir(parents=True,exist_ok=True);"
                "shutil.copyfile(sys.argv[1],sys.argv[2])"
            )
            return [
                sys.executable,
                "-c",
                program,
                str(input_path),
                str(output_path),
            ]

        with patch.object(
            throughput,
            "verify_round_trip",
            side_effect=AssertionError("fast mode must not verify"),
        ):
            records = throughput.execute_schedule(
                results=results,
                checkpoints=[checkpoint],
                profiles=["single-core"],
                methods=["brevis"],
                resources={("single-core", "brevis"): resource},
                experiment_id="fast-experiment",
                warmups=0,
                repetitions=2,
                cache_mode="unconditioned",
                drop_caches_command=None,
                brevis_bin=Path("/unused/brevis"),
                brevis_config=base.BrevisConfig(workers=1),
                keep_artifacts=False,
                skip_round_trip_check=True,
                rerun=False,
                order_seed=0,
                checkpoint_descriptors={"fast-fixture": descriptor},
                command_builder=copy_builder,
            )

        self.assertEqual(
            ["compress", "decompress", "compress", "decompress"],
            [record["operation"] for record in records],
        )
        for record in records:
            self.assertEqual("ok", record["status"])
            self.assertFalse(record["round_trip_check_performed"])
            self.assertIsNone(record["round_trip_exact"])
            self.assertEqual("not_checked", record["exactness_scope"])
            self.assertGreater(record["phase_makespan_seconds"], 0)

        raw = base.read_jsonl(results / "raw" / "throughput-runs.jsonl")
        self.assertTrue(
            all(
                record["round_trip_check_performed"] is False
                for record in raw
            )
        )
        self.assertTrue(
            all(record["exactness_scope"] == "not_checked" for record in raw)
        )
        phases = [
            record for record in raw if record["operation"] != "pair"
        ]
        self.assertTrue(
            all(record["round_trip_exact"] is None for record in phases)
        )

        self.write_environment(
            results=results,
            experiment_id="fast-experiment",
            checkpoint=checkpoint,
            profile="single-core",
            method="brevis",
            resource=resource,
            repetitions=2,
            round_trip_check_performed=False,
        )
        rows = throughput.summarize(results, "fast-experiment")
        self.assertEqual(2, len(rows))
        for row in rows:
            self.assertEqual("complete", row["summary_status"])
            self.assertEqual(2, row["n"])
            self.assertFalse(row["round_trip_check_performed"])
            self.assertIsNone(row["all_round_trips_exact"])
            self.assertEqual("not_checked", row["exactness_scope"])
            self.assertEqual(
                "not_performed_by_explicit_throughput_fast_mode",
                row["verification_algorithm"],
            )
            self.assertIn("median_seconds", row)
            self.assertIn("median_gib_per_second", row)

        parsed = throughput.parse_args(
            [
                "run",
                "--checkpoint",
                str(source),
                "--skip-round-trip-check",
                "--dry-run",
            ]
        )
        self.assertTrue(parsed.skip_round_trip_check)

    def test_byte_exact_verification_hashes_only_restored_shard(self):
        source = self.root / "source.bin"
        restored = self.root / "restored.bin"
        source.write_bytes(b"identical bytes")
        restored.write_bytes(source.read_bytes())
        checkpoint = base.Checkpoint("hash-fixture", self.root, (source,))
        descriptor = throughput.checkpoint_descriptor(checkpoint)
        real_hash = base.file_sha256
        hashed_paths = []

        def track_hash(path):
            hashed_paths.append(path)
            return real_hash(path)

        with (
            patch.object(base, "file_sha256", side_effect=track_hash),
            patch.object(
                base,
                "byte_exact",
                side_effect=AssertionError("must not reread source"),
            ),
        ):
            exact, verification = throughput.verify_round_trip(
                method="brevis",
                checkpoint=checkpoint,
                descriptor=descriptor,
                restored=(restored,),
            )

        self.assertTrue(exact)
        self.assertEqual([restored], hashed_paths)
        self.assertFalse(
            verification["source_reread_during_final_verification"]
        )
        self.assertTrue(
            verification["restored_reread_during_final_verification"]
        )

    def test_zipnn_verification_keeps_tensor_exact_comparison(self):
        source = self.root / "source.safetensors"
        restored = self.root / "restored.safetensors"
        source.write_bytes(b"source")
        restored.write_bytes(b"restored")
        checkpoint = base.Checkpoint("zipnn-fixture", self.root, (source,))
        descriptor = throughput.checkpoint_descriptor(checkpoint)

        with (
            patch.object(base, "tensor_exact", return_value=True) as checker,
            patch.object(
                base,
                "file_sha256",
                side_effect=AssertionError("ZipNN must use tensor comparison"),
            ),
        ):
            exact, verification = throughput.verify_round_trip(
                method="zipnn",
                checkpoint=checkpoint,
                descriptor=descriptor,
                restored=(restored,),
            )

        self.assertTrue(exact)
        checker.assert_called_once_with(source, restored)
        self.assertEqual(
            "safetensors_tensor_bit_exact",
            verification["scope"],
        )
        self.assertTrue(
            verification["source_reread_during_final_verification"]
        )

    def test_cpu_list_parser_counts_unique_cpus(self):
        rendered, count = throughput.parse_cpu_list("0-3,2,8")
        self.assertEqual("0-3,2,8", rendered)
        self.assertEqual(5, count)
        with self.assertRaises(Exception):
            throughput.parse_cpu_list("4-2")

    def test_cli_dry_run_exposes_zipnn_process_resource_shape(self):
        model = self.root / "model"
        model.mkdir()
        (model / "model.safetensors").write_bytes(b"x" * 16)
        rendered = io.StringIO()

        with redirect_stdout(rendered):
            status = throughput.main(
                [
                    "run",
                    "--checkpoint",
                    str(model),
                    "--methods",
                    "zipnn",
                    "--profiles",
                    "practical",
                    "--zipnn-execution",
                    "processes",
                    "--zipnn-workers",
                    "2",
                    "--dry-run",
                ]
            )

        self.assertEqual(0, status)
        output = rendered.getvalue()
        self.assertIn('"configured_process_concurrency": 2', output)
        self.assertIn('"effective_process_concurrency": 1', output)
        self.assertIn("--threads 1", output)

    def test_thirty_shards_use_thirty_practical_processes(self):
        common = {
            "matched_workers": 32,
            "practical_workers": 32,
            "brevis_workers": 32,
            "zipnn_workers": 32,
            "cpu_affinity": None,
        }
        brevis = throughput.resolve_resource(
            "brevis",
            "practical",
            zipnn_execution="processes",
            brevis_execution="processes",
            **common,
        )
        zipnn = throughput.resolve_resource(
            "zipnn",
            "practical",
            zipnn_execution="processes",
            **common,
        )
        generic = throughput.resolve_resource(
            "zstd-9",
            "practical",
            zipnn_execution="processes",
            **common,
        )
        threaded_brevis = throughput.resolve_resource(
            "brevis",
            "practical",
            zipnn_execution="processes",
            **common,
        )

        for resource in (brevis, zipnn, generic):
            provenance = resource.provenance(shard_count=30)
            self.assertEqual(32, resource.shard_jobs)
            self.assertEqual(30, provenance["effective_shard_jobs"])
            self.assertEqual(30, provenance["effective_process_concurrency"])
            self.assertEqual(30, provenance["effective_cpu_slot_upper_bound"])
        self.assertEqual(1, brevis.codec_workers)
        self.assertEqual(32, threaded_brevis.codec_workers)
        self.assertEqual(1, threaded_brevis.shard_jobs)

    def test_brevis_process_dry_run_uses_one_worker_per_shard(self):
        model = self.root / "thirty-shard-model"
        model.mkdir()
        for index in range(30):
            (model / f"model-{index:05d}.safetensors").write_bytes(b"x" * 16)
        rendered = io.StringIO()

        with (
            patch.object(
                throughput,
                "scheduler_cpu_set",
                return_value=tuple(range(64)),
            ),
            patch.object(
                throughput.shutil,
                "which",
                return_value="/usr/bin/taskset",
            ),
            redirect_stdout(rendered),
        ):
            status = throughput.main(
                [
                    "run",
                    "--checkpoint",
                    str(model),
                    "--methods",
                    "brevis",
                    "--profiles",
                    "practical",
                    "--brevis-execution",
                    "processes",
                    "--brevis-workers",
                    "32",
                    "--dry-run",
                ]
            )

        self.assertEqual(0, status)
        output = rendered.getvalue()
        self.assertIn('"configured_process_concurrency": 32', output)
        self.assertIn('"effective_process_concurrency": 30', output)
        self.assertIn('"effective_cpu_slot_upper_bound": 30', output)
        self.assertIn("--workers 1", output)
        self.assertIn(".brv", output)

    def test_brevis_archive_suffix_and_artifact_containment(self):
        source = self.root / "model.safetensors"
        source.write_bytes(b"x" * 16)
        checkpoint = base.Checkpoint("..", self.root, (source,))
        results = self.root / "safe-results"
        workspace = throughput.create_artifact_workspace(
            results=results,
            checkpoint=checkpoint,
            method="brevis",
            profile="resource-matched",
            sample_kind="measured",
            sample_index=0,
            pair_id="pair",
            attempt_id="attempt",
        )

        self.assertEqual(".brv", workspace.archives[0].suffix)
        self.assertNotIn("..", workspace.root.relative_to(results / "work").parts)
        outside = self.root / "outside"
        outside.write_bytes(b"keep")
        with self.assertRaisesRegex(base.BenchmarkError, "escapes"):
            throughput.safe_unlink_artifact(workspace, outside)
        self.assertEqual(b"keep", outside.read_bytes())
        throughput.cleanup_artifact_workspace(workspace)
        self.assertFalse(workspace.marker.exists())

        symlink_results = self.root / "symlink-results"
        symlink_results.mkdir()
        target = self.root / "outside-work"
        target.mkdir()
        (symlink_results / "work").symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(base.BenchmarkError, "symlink"):
            throughput.create_artifact_workspace(
                results=symlink_results,
                checkpoint=checkpoint,
                method="brevis",
                profile="resource-matched",
                sample_kind="measured",
                sample_index=0,
                pair_id="pair",
                attempt_id="attempt",
            )

    def test_checkpoint_fingerprint_rehashes_and_matches_manifest(self):
        verified_dir = self.root / "verified"
        verified_dir.mkdir()
        verified_file = verified_dir / "model.safetensors"
        verified_file.write_bytes(b"verified-content")
        digest = hashlib.sha256(verified_file.read_bytes()).hexdigest()
        manifest = {
            "name": "verified",
            "repo_id": "test/verified",
            "revision": "a" * 40,
            "sha256_verified": True,
            "weights": [
                {
                    "path": verified_file.name,
                    "size": verified_file.stat().st_size,
                    "sha256": digest,
                }
            ],
        }
        manifest_path = verified_dir / "download-manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        checkpoint = base.Checkpoint(
            "verified",
            verified_dir,
            (verified_file,),
            "test/verified",
            "a" * 40,
        )
        real_hash = base.file_sha256
        hashed_paths = []

        def track_hash(path):
            hashed_paths.append(path)
            return real_hash(path)

        with patch.object(base, "file_sha256", side_effect=track_hash):
            descriptor = throughput.checkpoint_descriptor(checkpoint)

        self.assertTrue(descriptor["manifest_reported_sha256_verified"])
        self.assertTrue(descriptor["content_sha256_recomputed"])
        self.assertEqual(
            "computed_content_sha256_matched_manifest",
            descriptor["identity_source"],
        )
        self.assertIn(manifest_path, hashed_paths)
        self.assertIn(verified_file, hashed_paths)
        self.assertEqual(digest, descriptor["files"][0]["sha256"])

        same_content_dir = self.root / "same-content-no-manifest"
        same_content_dir.mkdir()
        same_content_file = same_content_dir / "model.safetensors"
        same_content_file.write_bytes(verified_file.read_bytes())
        same_content_descriptor = throughput.checkpoint_descriptor(
            base.Checkpoint(
                "different-label",
                same_content_dir,
                (same_content_file,),
            )
        )
        self.assertEqual(
            descriptor["fingerprint"],
            same_content_descriptor["fingerprint"],
        )

        manifest["weights"][0]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(base.BenchmarkError, "SHA-256 mismatch"):
            throughput.checkpoint_descriptor(checkpoint)

        first_dir = self.root / "fallback-a"
        second_dir = self.root / "fallback-b"
        first_dir.mkdir()
        second_dir.mkdir()
        first = first_dir / "model.safetensors"
        second = second_dir / "model.safetensors"
        first.write_bytes(b"same-size-a")
        second.write_bytes(b"same-size-b")
        first_descriptor = throughput.checkpoint_descriptor(
            base.Checkpoint("fallback", first_dir, (first,))
        )
        second_descriptor = throughput.checkpoint_descriptor(
            base.Checkpoint("fallback", second_dir, (second,))
        )
        self.assertEqual(first.stat().st_size, second.stat().st_size)
        self.assertNotEqual(
            first_descriptor["fingerprint"],
            second_descriptor["fingerprint"],
        )
        self.assertEqual(
            "computed_content_sha256",
            first_descriptor["identity_source"],
        )

    def test_failed_rerun_invalidates_old_pair_and_summary_fails_closed(self):
        source = self.root / "source.bin"
        source.write_bytes(b"payload")
        checkpoint = base.Checkpoint("attempt-fixture", self.root, (source,))
        results = self.root / "attempt-results"
        resource = throughput.resolve_resource(
            "brevis",
            "single-core",
            matched_workers=1,
            practical_workers=1,
            brevis_workers=1,
            zipnn_workers=1,
            zipnn_execution="threads",
            cpu_affinity=None,
        )

        def copy_builder(
            _method,
            _operation,
            input_path,
            output_path,
            _resource,
            _brevis_bin,
            _brevis_config,
        ):
            program = (
                "import pathlib,shutil,sys;"
                "pathlib.Path(sys.argv[2]).parent.mkdir(parents=True,exist_ok=True);"
                "shutil.copyfile(sys.argv[1],sys.argv[2])"
            )
            return [
                sys.executable,
                "-c",
                program,
                str(input_path),
                str(output_path),
            ]

        common = {
            "results": results,
            "checkpoints": [checkpoint],
            "profiles": ["single-core"],
            "methods": ["brevis"],
            "resources": {("single-core", "brevis"): resource},
            "experiment_id": "attempt-experiment",
            "warmups": 0,
            "repetitions": 1,
            "cache_mode": "unconditioned",
            "drop_caches_command": None,
            "brevis_bin": Path("/unused/brevis"),
            "brevis_config": base.BrevisConfig(workers=1),
            "keep_artifacts": False,
            "order_seed": 0,
        }
        throughput.execute_schedule(
            **common,
            rerun=False,
            command_builder=copy_builder,
        )
        self.write_environment(
            results=results,
            experiment_id="attempt-experiment",
            checkpoint=checkpoint,
            profile="single-core",
            method="brevis",
            resource=resource,
            repetitions=1,
        )
        self.assertTrue(
            all(
                row["summary_status"] == "complete"
                for row in throughput.summarize(
                    results,
                    "attempt-experiment",
                )
            )
        )

        def fail_decompress(
            method,
            operation,
            input_path,
            output_path,
            resource_spec,
            brevis_bin,
            brevis_config,
        ):
            if operation == "decompress":
                return [sys.executable, "-c", "raise SystemExit(9)"]
            return copy_builder(
                method,
                operation,
                input_path,
                output_path,
                resource_spec,
                brevis_bin,
                brevis_config,
            )

        with self.assertRaises(base.BenchmarkError):
            throughput.execute_schedule(
                **common,
                rerun=True,
                command_builder=fail_decompress,
            )
        self.assertEqual(
            [],
            list((results / "work").rglob(".brevis-throughput-owner.json")),
        )
        with self.assertRaisesRegex(base.BenchmarkError, "incomplete"):
            throughput.summarize(results, "attempt-experiment")
        provisional = throughput.summarize(
            results,
            "attempt-experiment",
            allow_incomplete=True,
        )
        self.assertTrue(
            all(row["summary_status"] == "incomplete" for row in provisional)
        )
        self.assertTrue(all(row["n"] == 0 for row in provisional))
        self.assertTrue(
            all(row["invalid_repetition_indices"] == [0] for row in provisional)
        )

        rerun_records = throughput.execute_schedule(
            **common,
            rerun=False,
            command_builder=copy_builder,
        )
        self.assertEqual(2, len(rerun_records))
        self.assertTrue(
            all(
                row["summary_status"] == "complete"
                for row in throughput.summarize(
                    results,
                    "attempt-experiment",
                )
            )
        )

        with self.assertRaises(base.BenchmarkError):
            throughput.execute_schedule(
                **common,
                rerun=True,
                keep_failed_artifacts=True,
                command_builder=fail_decompress,
            )
        retained_markers = list(
            (results / "work").rglob(".brevis-throughput-owner.json")
        )
        self.assertEqual(1, len(retained_markers))
        self.assertTrue(
            any(path.suffix == ".brv" for path in retained_markers[0].parent.rglob("*"))
        )

    def test_resource_affinity_and_libdeflate_process_metadata(self):
        matched = throughput.resolve_resource(
            "brevis",
            "resource-matched",
            matched_workers=2,
            practical_workers=4,
            brevis_workers=4,
            zipnn_workers=4,
            zipnn_execution="threads",
            cpu_affinity=None,
        )
        bound = throughput.bind_resource_affinity(
            {("resource-matched", "brevis"): matched},
            (3, 7, 11),
            taskset_available=True,
        )
        self.assertEqual(
            "3,7",
            bound[("resource-matched", "brevis")].cpu_affinity,
        )
        with self.assertRaisesRegex(base.BenchmarkError, "requires taskset"):
            throughput.bind_resource_affinity(
                {("resource-matched", "brevis"): matched},
                (3, 7),
                taskset_available=False,
            )

        libdeflate = throughput.resolve_resource(
            "libdeflate-1",
            "resource-matched",
            matched_workers=4,
            practical_workers=4,
            brevis_workers=4,
            zipnn_workers=4,
            zipnn_execution="threads",
            cpu_affinity=None,
        )
        provenance = libdeflate.provenance(shard_count=3)
        self.assertEqual(2, libdeflate.processes_per_command)
        self.assertEqual(1, libdeflate.cpu_slots_per_command)
        self.assertEqual(6, provenance["effective_process_concurrency"])
        self.assertEqual(3, provenance["effective_cpu_slot_upper_bound"])

    def test_zipnn_wrapper_sets_torch_pools_to_one(self):
        source = self.root / "source"
        output = self.root / "output"
        source.write_bytes(b"zipnn")
        calls = []
        fake_torch = SimpleNamespace(
            set_num_threads=lambda value: calls.append(("intra", value)),
            set_num_interop_threads=lambda value: calls.append(("inter", value)),
        )

        def compress(input_path, output_path, threads):
            calls.append(("codec", threads))
            output_path.write_bytes(input_path.read_bytes())

        fake_codecs = SimpleNamespace(
            CODECS={
                "zipnn": SimpleNamespace(
                    compress=compress,
                    decompress=compress,
                )
            }
        )
        with patch.dict(
            sys.modules,
            {"torch": fake_torch, "benchmark_codecs": fake_codecs},
        ):
            status = throughput.zipnn_adapter_main(
                [
                    "compress",
                    str(source),
                    str(output),
                    "--threads",
                    "16",
                ]
            )

        self.assertEqual(0, status)
        self.assertEqual(
            [("intra", 1), ("inter", 1), ("codec", 16)],
            calls,
        )

    def test_brevis_provenance_does_not_call_invalid_version_command(self):
        binary = self.root / "brevis"
        binary.write_bytes(b"binary")
        with patch.object(
            base,
            "command_version",
            side_effect=AssertionError("must not call --version"),
        ):
            provenance = throughput.method_provenance(
                "brevis",
                binary,
                "revision",
            )

        self.assertIsNone(provenance["version_command"])
        self.assertEqual("revision", provenance["repository_revision"])
        self.assertEqual(
            hashlib.sha256(b"binary").hexdigest(),
            provenance["binary_sha256"],
        )

    def test_environment_plan_detects_never_started_repetitions(self):
        key = (
            "empty-experiment",
            "checkpoint",
            "fingerprint",
            "brevis",
            "resource-matched",
            "hot",
            True,
        )
        rows = throughput.summary_rows(
            [],
            "empty-experiment",
            {
                key: {
                    "expected_repetitions": 3,
                    "resource": None,
                    "source_bytes": 10,
                    "round_trip_check_performed": True,
                    "exactness_verification": (
                        throughput.exactness_protocol("brevis")
                    ),
                }
            },
        )
        self.assertEqual(2, len(rows))
        self.assertTrue(all(row["n"] == 0 for row in rows))
        self.assertTrue(
            all(row["missing_repetition_indices"] == [0, 1, 2] for row in rows)
        )

    def test_strict_summary_requires_environment_snapshot(self):
        results = self.root / "no-environment"
        with self.assertRaisesRegex(
            base.BenchmarkError,
            "requires a matching immutable environment",
        ):
            throughput.summarize(results, "missing")

        rows = throughput.summarize(
            results,
            "missing",
            allow_incomplete=True,
        )
        self.assertEqual([], rows)
        payload = json.loads(
            (results / "tables" / "throughput-summary.json").read_text()
        )
        self.assertEqual(
            "provisional_raw_without_environment",
            payload["summary_mode"],
        )

    def test_results_symlink_is_rejected_before_environment_or_raw_write(self):
        target = self.root / "target"
        target.mkdir()
        results_link = self.root / "results-link"
        results_link.symlink_to(target, target_is_directory=True)
        environment = {
            "schema_version": throughput.SCHEMA_VERSION,
            "experiment_id": "unsafe",
        }

        with self.assertRaisesRegex(base.BenchmarkError, "must not be a symlink"):
            throughput.write_environment(results_link, environment)
        self.assertEqual([], list(target.iterdir()))

        results = self.root / "raw-symlink-results"
        raw = results / "raw"
        raw.mkdir(parents=True)
        outside = self.root / "outside-raw"
        outside.write_text("unchanged")
        (raw / "throughput-runs.jsonl").symlink_to(outside)
        source = self.root / "raw-source"
        source.write_bytes(b"source")
        checkpoint = base.Checkpoint("raw-fixture", self.root, (source,))
        resource = throughput.resolve_resource(
            "brevis",
            "single-core",
            matched_workers=1,
            practical_workers=1,
            brevis_workers=1,
            zipnn_workers=1,
            zipnn_execution="threads",
            cpu_affinity=None,
        )
        with self.assertRaisesRegex(base.BenchmarkError, "symlink"):
            throughput.execute_schedule(
                results=results,
                checkpoints=[checkpoint],
                profiles=["single-core"],
                methods=["brevis"],
                resources={("single-core", "brevis"): resource},
                experiment_id="raw-unsafe",
                warmups=0,
                repetitions=1,
                cache_mode="unconditioned",
                drop_caches_command=None,
                brevis_bin=Path("/unused"),
                brevis_config=base.BrevisConfig(workers=1),
                keep_artifacts=False,
                rerun=False,
                order_seed=0,
            )
        self.assertEqual("unchanged", outside.read_text())


if __name__ == "__main__":
    unittest.main()
