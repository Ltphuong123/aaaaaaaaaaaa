import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from voice_studio import app as studio
from voice_studio.read_cache import ReadCache, engine_fingerprint


class ReadingCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = studio.VoiceStore(Path(self.directory.name))
        self.voice_id = "a" * 32
        self.voice_dir = self.store.voices / self.voice_id
        self.voice_dir.mkdir()
        self.reference = self.voice_dir / "reference.wav"
        sf.write(self.reference, np.ones(1600) * .1, 16000)
        self.metadata = {"id": self.voice_id, "name": "Original voice", "transcript": "Sample text.", "seed": 42}
        self.store.write_json(self.voice_dir / "voice.json", self.metadata)
        self.client = Mock()
        self.client.predict.return_value = (str(self.reference), 42)
        self.session = studio.BackendSession()
        self.cache = ReadCache()
        self.patches = [patch.object(studio, "store", self.store), patch.object(studio, "jobs", {}),
                        patch.object(studio, "backend_session", self.session), patch.object(studio, "read_cache", self.cache),
                        patch.object(studio, "Client", return_value=self.client),
                        patch.object(studio, "engine_fingerprint", return_value=("pid", "weights"))]
        for item in self.patches:
            item.start()
        self.counter = 0

    def tearDown(self):
        self.session.close()
        for item in reversed(self.patches):
            item.stop()
        self.directory.cleanup()

    def generate(self, **changes):
        values = {"kind": "speak", "text": "Read this sentence.", "voice_id": self.voice_id,
                  "read_mode": "balanced", "seed": 42}
        values.update(changes)
        request = studio.GenerateRequest(**values)
        self.counter += 1
        job_id = f"{self.counter:032x}"
        studio.jobs[job_id] = {"id": job_id, "kind": request.kind, "status": "queued"}
        studio.generate(job_id, request)
        self.assertEqual(studio.jobs[job_id]["status"], "completed", studio.jobs[job_id])
        return self.store.get_render(job_id)

    def test_identical_read_copies_exact_audio_to_new_history_id_and_current_name(self):
        first = self.generate()
        self.metadata["name"] = "Renamed voice"
        self.store.write_json(self.voice_dir / "voice.json", self.metadata)
        second = self.generate()
        self.assertFalse(first["cache_hit"])
        self.assertTrue(second["cache_hit"])
        self.assertNotEqual(first["id"], second["id"])
        self.assertNotEqual(first["created_at"], second["created_at"])
        self.assertEqual(second["voice_name"], "Renamed voice")
        self.assertEqual(second["audio_url"], f"/api/renders/{second['id']}/audio")
        for suffix in (".wav", ".sample.wav"):
            self.assertEqual((self.store.renders / f"{first['id']}{suffix}").read_bytes(),
                             (self.store.renders / f"{second['id']}{suffix}").read_bytes())
        self.assertEqual(first["generation"], second["generation"])
        self.client.predict.assert_called_once()

    def test_text_seed_mode_and_quality_settings_have_distinct_keys(self):
        self.generate()
        for changes in [{"text": "Another sentence."}, {"seed": 7}, {"read_mode": "fast"},
                        {"read_mode": "faithful"}, {"cfg_value": 2.5}, {"do_normalize": True}]:
            with self.subTest(changes=changes):
                before = self.client.predict.call_count
                result = self.generate(**changes)
                self.assertFalse(result["cache_hit"])
                self.assertEqual(self.client.predict.call_count, before + 1)

    def test_reference_content_and_transcript_changes_invalidate(self):
        self.generate()
        sf.write(self.reference, np.ones(1600) * .2, 16000)
        self.assertFalse(self.generate()["cache_hit"])
        self.metadata["transcript"] = "Changed sample text."
        self.store.write_json(self.voice_dir / "voice.json", self.metadata)
        self.assertFalse(self.generate()["cache_hit"])
        self.assertEqual(self.client.predict.call_count, 3)

    def test_deleted_source_and_cleared_history_regenerate_without_restoring_old_ids(self):
        first = self.generate()
        studio.delete_reading(first["id"])
        second = self.generate()
        self.assertFalse(second["cache_hit"])
        self.assertFalse((self.store.renders / f"{first['id']}.json").exists())
        studio.clear_history()
        third = self.generate()
        self.assertFalse(third["cache_hit"])
        self.assertEqual([item["id"] for item in self.store.history()], [third["id"]])
        self.assertEqual(self.client.predict.call_count, 3)

    def test_overwritten_source_audio_and_missing_sample_regenerate(self):
        first = self.generate()
        sf.write(self.store.renders / f"{first['id']}.wav", np.zeros(3200), 16000)
        second = self.generate()
        self.assertFalse(second["cache_hit"])
        (self.store.renders / f"{second['id']}.sample.wav").unlink()
        self.assertFalse(self.generate()["cache_hit"])
        self.assertEqual(self.client.predict.call_count, 3)

    def test_random_denoising_and_design_bypass_cache(self):
        for changes in [{"random_seed": True}, {"denoise": True}, {"kind": "design", "voice_id": None}]:
            with self.subTest(changes=changes):
                self.generate(**changes)
                self.assertFalse(self.generate(**changes)["cache_hit"])
        self.assertEqual(self.client.predict.call_count, 6)
        self.assertFalse(self.cache.entries)

    def test_engine_change_and_unknown_engine_invalidate(self):
        self.generate()
        with patch.object(studio, "engine_fingerprint", return_value=("newpid", "weights")):
            self.assertFalse(self.generate()["cache_hit"])
        with patch.object(studio, "engine_fingerprint", return_value=None):
            self.generate()
            self.assertFalse(self.generate()["cache_hit"])
        self.assertEqual(self.client.predict.call_count, 4)

    def test_prediction_failure_clears_other_cached_reads_and_adds_no_result(self):
        first = self.generate()
        self.client.predict.side_effect = ConnectionError("Backend stopped")
        job_id = "f" * 32
        studio.jobs[job_id] = {"status": "queued"}
        with self.assertLogs(studio.log, level="ERROR"):
            studio.generate(job_id, studio.GenerateRequest(kind="speak", text="Uncached text.", voice_id=self.voice_id))
        self.assertEqual(studio.jobs[job_id]["status"], "failed")
        self.assertFalse(self.cache.entries)
        self.assertFalse((self.store.renders / f"{job_id}.json").exists())
        self.assertTrue((self.store.renders / f"{first['id']}.json").exists())

    def test_lru_is_bounded_and_refreshes_recent_entries(self):
        first = self.generate()
        cache = ReadCache(max_entries=2)
        for key in ("a", "b"):
            cache.put(key, first["id"], self.store.renders)
        cache.get("a", self.store.renders)
        cache.put("c", first["id"], self.store.renders)
        self.assertEqual(list(cache.entries), ["a", "c"])
        cache.clear()
        self.assertFalse(cache.entries)

    def test_non_object_source_metadata_is_ignored(self):
        first = self.generate()
        path = self.store.renders / f"{first['id']}.json"
        path.write_text("[]", encoding="utf-8")
        self.cache.put("corrupt", first["id"], self.store.renders)
        self.assertIsNone(self.cache.get("corrupt", self.store.renders))


class EngineFingerprintTests(unittest.TestCase):
    def test_custom_backend_and_missing_local_install_disable_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertIsNone(engine_fingerprint(root, "https://remote.example"))
            self.assertIsNone(engine_fingerprint(root, "http://127.0.0.1:8808"))

    def test_process_model_and_code_updates_change_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "pretrained_models" / "VoxCPM2"
            code = root / "src" / "voxcpm"
            marker = root / ".runtime" / "server.pid"
            for path in (model, code, marker.parent):
                path.mkdir(parents=True)
            marker.write_text("123", encoding="utf-8")
            (root / "app.py").write_text("initial")
            for name in ["model.safetensors", "audiovae.pth", "config.json", "tokenizer.json",
                         "tokenizer_config.json", "special_tokens_map.json"]:
                (model / name).write_text("initial")
            source = code / "core.py"
            source.write_text("initial")
            previous = engine_fingerprint(root, "http://127.0.0.1:8808")
            self.assertIsNotNone(previous)
            for path in [marker, model / "model.safetensors", source]:
                path.write_text("45678" if path == marker else "updated contents")
                current = engine_fingerprint(root, "http://127.0.0.1:8808")
                self.assertNotEqual(previous, current)
                previous = current


if __name__ == "__main__":
    unittest.main()
