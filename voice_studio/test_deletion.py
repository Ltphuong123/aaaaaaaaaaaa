import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from fastapi import HTTPException
from fastapi.testclient import TestClient

from voice_studio import app as studio


class DeletionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = studio.VoiceStore(Path(self.directory.name))
        self.render_id = "a" * 32
        sf.write(self.store.renders / f"{self.render_id}.wav", np.ones(1600) * .1, 16000)
        self.store.write_json(self.store.renders / f"{self.render_id}.json", {
            "id": self.render_id, "kind": "design", "description": "Test voice", "text": "Xin chào.",
            "seed": 42, "duration": .1, "sample_rate": 16000,
        })
        self.voice = self.store.save_voice(self.render_id, "Giọng thử xóa")
        self.preset = self.store.save_voice(self.render_id, "Giọng có sẵn", origin="preset")
        self.store_patch = patch.object(studio, "store", self.store)
        self.jobs_patch = patch.object(studio, "jobs", {})
        self.store_patch.start()
        self.jobs_patch.start()
        self.client = TestClient(studio.app)

    def tearDown(self):
        self.client.close()
        self.jobs_patch.stop()
        self.store_patch.stop()
        self.directory.cleanup()

    def test_delete_only_created_voice_and_keep_original_audio_and_history(self):
        render_id = "b" * 32
        shutil.copyfile(self.store.renders / f"{self.render_id}.wav", self.store.renders / f"{render_id}.wav")
        self.store.write_json(self.store.renders / f"{render_id}.json", {
            "id": render_id, "kind": "speak", "voice_id": self.voice["id"], "voice_name": self.voice["name"],
            "created_at": studio.now(), "text": "Bản đọc đã tạo.",
        })
        original_files = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.store.renders.glob("*")}
        response = self.client.delete(f"/api/voices/{self.voice['id']}")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertFalse((self.store.voices / self.voice["id"]).exists())
        self.assertEqual([v["id"] for v in self.store.list_voices()], [self.preset["id"]])
        self.assertEqual(self.client.get(self.voice["audio_url"]).status_code, 404)
        self.assertEqual(self.store.history()[0]["id"], render_id)
        self.assertEqual(original_files, {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.store.renders.glob("*")})
        self.assertEqual(self.client.delete(f"/api/voices/{self.voice['id']}").status_code, 404)
        # Removing a voice also frees its name for a newly saved voice.
        self.assertEqual(self.store.save_voice(self.render_id, self.voice["name"])["name"], self.voice["name"])

    def test_presets_and_invalid_paths_cannot_be_deleted(self):
        self.assertEqual(self.client.delete(f"/api/voices/{self.preset['id']}").status_code, 403)
        self.assertTrue(self.store.audio_path(self.preset["id"]).is_file())
        for invalid in ("../outside", "x" * 32, "a" * 31):
            with self.assertRaises(HTTPException):
                self.store.delete_voice(invalid)

    def test_queued_and_running_voice_cannot_be_deleted(self):
        with patch.object(studio, "executor"):
            response = self.client.post("/api/jobs", json={"kind": "speak", "text": "Xin chào", "voice_id": self.voice["id"]})
        self.assertEqual(response.status_code, 202)
        job_id = response.json()["id"]
        for status in ("queued", "running"):
            studio.jobs[job_id]["status"] = status
            self.assertEqual(self.client.delete(f"/api/voices/{self.voice['id']}").status_code, 409)
            self.assertTrue(self.store.audio_path(self.voice["id"]).is_file())
        studio.jobs[job_id]["status"] = "completed"
        self.assertEqual(self.client.delete(f"/api/voices/{self.voice['id']}").status_code, 200)
        with patch.object(studio, "executor") as executor:
            response = self.client.post("/api/jobs", json={"kind": "speak", "text": "Xin chào", "voice_id": self.voice["id"]})
            self.assertEqual(response.status_code, 404)
            executor.submit.assert_not_called()

    def test_directory_link_cannot_delete_outside_the_voice_library(self):
        voice_id = "e" * 32
        target = self.store.root / "outside" / voice_id
        target.mkdir(parents=True)
        (target / "keep.txt").write_text("Keep this file", encoding="utf-8")
        link = self.store.voices / voice_id
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            if os.name != "nt":
                self.skipTest("Directory links unavailable")
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
        try:
            with self.assertRaises(HTTPException) as error:
                self.store.delete_voice(voice_id)
            self.assertEqual(error.exception.status_code, 403)
            self.assertEqual((target / "keep.txt").read_text(encoding="utf-8"), "Keep this file")
        finally:
            if link.is_symlink():
                link.unlink()
            else:
                link.rmdir()


if __name__ == "__main__":
    unittest.main()
