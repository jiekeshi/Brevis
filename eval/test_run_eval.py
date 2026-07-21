import json
import pathlib
import tempfile
import unittest
from unittest import mock

import run_eval


class ModelEvalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmp.name)
        self.work = self.root / "work"
        self.work.mkdir()
        self.sources = {}
        for name, size in (("model-00001-of-00002.safetensors", 10), ("model-00002-of-00002.safetensors", 14)):
            path = self.root / name
            path.write_bytes(bytes(range(size)))
            self.sources[name] = path
        self.model = {
            "tag": "mock-2shard",
            "repo": "test/model",
            "revision": "a" * 40,
            "files": list(self.sources),
            "note": "offline fixture",
        }
        self.archive_source = {}
        self.commands = []

    def tearDown(self):
        self.tmp.cleanup()

    def fetch(self, repo, revision, name, url=None):
        self.assertEqual("test/model", repo)
        self.assertEqual("a" * 40, revision)
        self.assertIsNone(url)
        return self.sources[name]

    def command(self, *cmd):
        self.commands.append(tuple(map(str, cmd)))
        action = str(cmd[1])
        if action == "calibrate":
            pathlib.Path(cmd[3]).write_bytes(b"prior")
        elif action == "compress":
            src, out = pathlib.Path(cmd[2]), pathlib.Path(cmd[3])
            divisor = 4 if ".fixed." in out.name else 2 if ".uniform." in out.name else 3
            out.write_bytes(b"x" * (src.stat().st_size // divisor + 1))
            self.archive_source[out] = src
        elif action == "decompress":
            archive, out = pathlib.Path(cmd[2]), pathlib.Path(cmd[3])
            out.write_bytes(self.archive_source[archive].read_bytes())
        return 1.0

    @staticmethod
    def baseline(path, tool, cargs, dargs, out):
        divisors = {"gzip": 2, "zstd": 3, "xz": 4}
        return path.stat().st_size // divisors[tool], 0.25, 0.125, True

    @staticmethod
    def openzl(path, out):
        return path.stat().st_size // 5, 0.25, 0.125, True

    def patches(self):
        return (
            mock.patch.object(run_eval, "fetch", side_effect=self.fetch),
            mock.patch.object(run_eval, "timed", side_effect=self.command),
            mock.patch.object(run_eval, "sh"),
            mock.patch.object(run_eval, "baseline", side_effect=self.baseline),
            mock.patch.object(run_eval, "openzl", side_effect=self.openzl),
        )

    def test_multishard_pipeline_aggregates_and_cleans_artifacts(self):
        patches = self.patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            row = run_eval.evaluate_model(self.model, self.work, jobs=3, calibration_tensors=17)

        self.assertEqual(24, row["raw"])
        self.assertEqual(7, row["fixed"])
        self.assertEqual(14, row["uniform"])
        self.assertEqual(9, row["phog"])
        self.assertEqual(12, row["gzip"])
        self.assertEqual(7, row["zstd"])
        self.assertEqual(5, row["xz"])
        self.assertEqual(4, row["openzl"])
        self.assertEqual(2.0, row["t_cal"])
        self.assertEqual(2.0, row["t_phog_cmp"])
        self.assertEqual(2.0, row["t_phog_dec_jobs1"])
        self.assertEqual(2.0, row["t_phog_dec"])
        self.assertTrue(row["phog_bitexact_jobs1"])
        self.assertTrue(row["phog_bitexact"])
        self.assertEqual(run_eval.sha256_path(self.sources[next(iter(self.sources))]),
                         row["shards"][0]["input_sha256"])
        self.assertEqual(run_eval.sha256_path(self.sources[list(self.sources)[1]]),
                         row["shards"][1]["input_sha256"])
        self.assertEqual(run_eval.named_digest(row["shards"], "input_sha256"), row["model_sha256"])
        self.assertEqual(run_eval.model_manifest_sha256(self.model), row["manifest_sha256"])
        compress = [cmd for cmd in self.commands if cmd[1] == "compress"]
        fixed = next(cmd for cmd in compress if ".fixed.brv" in cmd[3])
        uniform = next(cmd for cmd in compress if ".uniform.brv" in cmd[3])
        phog = next(cmd for cmd in compress if cmd[3].endswith(".brv") and ".fixed." not in cmd[3]
                    and ".uniform." not in cmd[3])
        self.assertEqual("fixed", fixed[fixed.index("--plan") + 1])
        self.assertEqual("search", uniform[uniform.index("--plan") + 1])
        self.assertNotIn("--prior", uniform)
        self.assertEqual("search", phog[phog.index("--plan") + 1])
        self.assertIn("--prior", phog)
        self.assertTrue(all(cmd[cmd.index("--jobs") + 1] == "3" for cmd in compress))
        calibration = next(cmd for cmd in self.commands if cmd[1] == "calibrate")
        self.assertEqual("17", calibration[calibration.index("--tensors") + 1])
        self.assertEqual("3", calibration[calibration.index("--jobs") + 1])
        self.assertEqual(list(self.sources), [shard["file"] for shard in row["shards"]])
        self.assertFalse([path for path in self.work.rglob("*") if path.is_file()])

    def test_failed_shard_raises_and_cleans_every_temporary_file(self):
        calls = 0

        def corrupt_second_shard(*cmd):
            nonlocal calls
            elapsed = self.command(*cmd)
            if str(cmd[1]) == "decompress":
                calls += 1
                if calls == 3:
                    pathlib.Path(cmd[3]).write_bytes(b"corrupt")
            return elapsed

        patches = self.patches()
        patches = (patches[0], mock.patch.object(run_eval, "timed", side_effect=corrupt_second_shard), *patches[2:])
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            with self.assertRaises(run_eval.EvalError):
                run_eval.evaluate_model(self.model, self.work)

        self.assertFalse([path for path in self.work.rglob("*") if path.is_file()])

    def test_reused_baselines_stay_attached_to_each_shard(self):
        patches = self.patches()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            previous = run_eval.evaluate_model(self.model, self.work)

        patches = self.patches()
        no_baseline = mock.patch.object(run_eval, "baseline", side_effect=AssertionError("baseline reran"))
        no_openzl = mock.patch.object(run_eval, "openzl", side_effect=AssertionError("OpenZL reran"))
        with patches[0], patches[1], patches[2], no_baseline, no_openzl:
            row = run_eval.evaluate_model(self.model, self.work, previous=previous)

        self.assertTrue(row["baselines_reused"])
        for old, new in zip(previous["shards"], row["shards"]):
            for key in run_eval.BASELINE_KEYS:
                self.assertEqual(old[key], new[key])
                self.assertEqual(old[f"{key}_exact"], new[f"{key}_exact"])

    def test_main_resumes_completed_shards_from_checkpoint(self):
        models = self.root / "models.json"
        results = self.root / "results.json"
        brevis = self.root / "brevis"
        models.write_text(json.dumps([self.model]))
        brevis.touch()
        calls = 0

        def corrupt_second_shard(*cmd):
            nonlocal calls
            elapsed = self.command(*cmd)
            if str(cmd[1]) == "decompress":
                calls += 1
                if calls == 3:
                    pathlib.Path(cmd[3]).write_bytes(b"corrupt")
            return elapsed

        patches = self.patches()
        patches = (patches[0], mock.patch.object(run_eval, "timed", side_effect=corrupt_second_shard), *patches[2:])
        config = {"max_nodes": 12, "candidate_collection": "all_within_expansion_budget",
                  "sample_byte_pruning": False}
        with mock.patch.object(run_eval, "BREVIS", brevis), mock.patch.object(run_eval, "CACHE", self.root / "cache"), \
                mock.patch.object(run_eval, "binary_config", return_value=config):
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                with self.assertRaises(run_eval.EvalError):
                    run_eval.main(["--models", str(models), "--results", str(results)])

        checkpoint = pathlib.Path(f"{results}.checkpoint")
        saved = json.loads(checkpoint.read_text())
        self.assertEqual([next(iter(self.sources))], [shard["file"] for shard in saved[0]["shards"]])
        unselected = {"repo": "other/model", "revision": "c" * 40, "fingerprint": "other", "shards": []}
        checkpoint.write_text(json.dumps([*saved, unselected]))

        compressed = []

        def record_compress(*cmd):
            if str(cmd[1]) == "compress":
                compressed.append(pathlib.Path(cmd[2]).name)
            return self.command(*cmd)

        patches = self.patches()
        patches = (patches[0], mock.patch.object(run_eval, "timed", side_effect=record_compress), *patches[2:])
        with mock.patch.object(run_eval, "BREVIS", brevis), mock.patch.object(run_eval, "CACHE", self.root / "cache"), \
                mock.patch.object(run_eval, "binary_config", return_value=config):
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                run_eval.main(["--models", str(models), "--results", str(results)])

        second = list(self.sources)[1]
        self.assertEqual([second, second, second], compressed)
        document = json.loads(results.read_text())
        self.assertEqual(2, document["schema"])
        provenance = document["provenance"]
        self.assertEqual({"fixed", "uniform", "phog"}, set(provenance["modes"]))
        self.assertEqual("fixed", provenance["modes"]["fixed"]["plan"])
        self.assertEqual("raw", provenance["modes"]["fixed"]["fallback"])
        self.assertEqual("search", provenance["modes"]["phog"]["plan"])
        self.assertEqual("shard-local", provenance["modes"]["phog"]["prior"])
        self.assertEqual(64, len(provenance["binary_sha256"]))
        self.assertIn("machine", provenance["platform"])
        self.assertTrue(provenance["zig_version"])
        self.assertIn("version", provenance["baselines"]["gzip"])
        self.assertIn("max_nodes", provenance["search"])
        self.assertEqual("all_within_expansion_budget", provenance["search"]["candidate_collection"])
        self.assertEqual(list(self.sources), [shard["file"] for shard in document["models"][0]["shards"]])
        self.assertEqual([unselected], json.loads(checkpoint.read_text()))

    def test_incompatible_checkpoint_is_ignored(self):
        compressed = []

        def record_compress(*cmd):
            if str(cmd[1]) == "compress":
                compressed.append(pathlib.Path(cmd[2]).name)
            return self.command(*cmd)

        first = next(iter(self.sources))
        names = list(self.sources)
        cases = (
            ({"repo": self.model["repo"], "revision": "b" * 40, "shards": [{"file": first}]}, None),
            ({**self.model, "fingerprint": "old", "shards": [{"file": first}]}, "new"),
        )
        for resumed, fingerprint in cases:
            with self.subTest(fingerprint=fingerprint):
                patches = self.patches()
                patches = (patches[0], mock.patch.object(run_eval, "timed", side_effect=record_compress), *patches[2:])
                with patches[0], patches[1], patches[2], patches[3], patches[4]:
                    run_eval.evaluate_model(self.model, self.work, resumed=resumed, fingerprint=fingerprint)
                self.assertEqual([names[0], names[0], names[0], names[1], names[1], names[1]], compressed)
                compressed.clear()


if __name__ == "__main__":
    unittest.main()
