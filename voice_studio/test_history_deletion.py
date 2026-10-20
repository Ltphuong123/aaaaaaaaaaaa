import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from voice_studio import app as studio


class HistoryDeletionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = studio.VoiceStore(Path(self.directory.name))
        self.design = self.make_render(100, kind="design")
        self.voice = self.store.save_voice(self.design["id"], "Giọng giữ lại")
        self.reading = self.make_render(1)
        self.other = self.make_render(2)
        self.store.write_json(self.store.references / ("e" * 32 + ".json"), {"text": "Lời mẫu giữ lại"})
        self.protected = self.snapshot(exclude_readings=True)
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

    def make_render(self, number, kind="speak"):
        render_id = f"{number:032x}"
        render = {
            "id": render_id, "kind": kind, "text": f"Nội dung bản đọc {number}.",
            "description": "Giọng thử", "voice_id": None, "voice_name": "Giọng thử",
            "seed": 42, "duration": .1, "sample_rate": 16000, "created_at": studio.now(),
            "audio_url": f"/api/renders/{render_id}/audio", "download_url": f"/api/renders/{render_id}/download",
        }
        for suffix in (".wav", ".sample.wav"):
            sf.write(self.store.renders / f"{render_id}{suffix}", np.zeros(1600), 16000)
        self.store.write_json(self.store.renders / f"{render_id}.json", render)
        return render

    def snapshot(self, exclude_readings=False):
        return {
            str(path.relative_to(self.store.root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in self.store.root.rglob("*") if path.is_file()
            and (not exclude_readings or path.parent != self.store.renders or path.name.startswith(self.design["id"]))
        }

    def test_delete_one_removes_audio_text_and_completed_job_without_affecting_other_data(self):
        render_id = self.reading["id"]
        studio.jobs[render_id] = {"id": render_id, "status": "completed", "result": self.reading}
        response = self.client.delete(f"/api/history/{render_id}")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["deleted"], [render_id])
        self.assertFalse(list(self.store.renders.glob(f"{render_id}.*")))
        for endpoint in (f"/api/jobs/{render_id}", self.reading["audio_url"], self.reading["download_url"]):
            self.assertEqual(self.client.get(endpoint).status_code, 404)
        self.assertEqual([r["id"] for r in self.client.get("/api/history").json()], [self.other["id"]])
        self.assertEqual(self.snapshot(exclude_readings=True), self.protected)
        self.assertEqual(self.client.delete(f"/api/history/{render_id}").status_code, 404)

    def test_clear_all_includes_more_than_the_twenty_returned_and_keeps_designs_and_voices(self):
        for number in range(3, 27):
            self.make_render(number)
        studio.jobs[self.design["id"]] = {"status": "completed", "result": self.design}
        for render in self.store.history(limit=None):
            studio.jobs[render["id"]] = {"status": "completed", "result": render}
        self.assertEqual(len(self.client.get("/api/history").json()), 20)
        response = self.client.delete("/api/history")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["count"], 26)
        self.assertEqual(response.json()["failed"], [])
        self.assertEqual(self.snapshot(), self.protected)
        self.assertEqual(list(studio.jobs), [self.design["id"]])
        self.assertEqual(self.client.get("/api/history").json(), [])
        self.assertEqual(self.client.get(self.voice["audio_url"]).status_code, 200)
        self.assertEqual(self.client.delete("/api/history").json()["count"], 0)

    def test_in_progress_readings_are_protected_and_design_jobs_do_not_block_clear(self):
        with patch.object(studio, "executor"):
            job = self.client.post("/api/jobs", json={"kind": "speak", "voice_id": self.voice["id"], "text": "Xin chào"}).json()
        self.assertEqual(job["kind"], "speak")
        original = self.snapshot()
        for status in ("queued", "running"):
            studio.jobs[job["id"]]["status"] = status
            self.assertEqual(self.client.delete("/api/history").status_code, 409)
            self.assertEqual(self.client.delete(f"/api/history/{job['id']}").status_code, 409)
            self.assertEqual(self.snapshot(), original)
        # Removing an existing reading is safe while another one is being made.
        self.assertEqual(self.client.delete(f"/api/history/{self.reading['id']}").status_code, 200)
        studio.jobs[job["id"]]["status"] = "failed"
        studio.jobs[self.design["id"]] = {"kind": "design", "status": "running"}
        self.assertEqual(self.client.delete("/api/history").status_code, 200)
        self.assertTrue(self.store.audio_path(self.voice["id"]).is_file())

    def test_designs_invalid_ids_and_directory_links_are_protected(self):
        original = self.snapshot()
        self.assertEqual(self.client.delete(f"/api/history/{self.design['id']}").status_code, 403)
        self.assertEqual(self.client.delete("/api/history/not-an-id").status_code, 404)
        self.assertEqual(self.snapshot(), original)
        render_id = self.reading["id"]
        link = self.store.renders / f"{render_id}.sample.wav"
        link.unlink()
        target = self.store.root / "outside"
        target.mkdir()
        (target / "keep.txt").write_text("Keep this file", encoding="utf-8")
        try:
            link.symlink_to(target, target_is_directory=True)
        except OSError:
            if os.name != "nt":
                self.skipTest("Directory links unavailable")
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
        try:
            self.assertEqual(self.client.delete(f"/api/history/{render_id}").status_code, 403)
            self.assertTrue((self.store.renders / f"{render_id}.wav").is_file())
            self.assertEqual((target / "keep.txt").read_text(encoding="utf-8"), "Keep this file")
        finally:
            if link.is_symlink():
                link.unlink()
            else:
                link.rmdir()

    def test_partial_clear_reports_locked_files_and_can_be_retried(self):
        render_id = self.reading["id"]
        studio.jobs[render_id] = {"status": "completed", "result": self.reading}
        studio.jobs[self.other["id"]] = {"status": "completed", "result": self.other}
        unlink = Path.unlink

        def locked(path, *args, **kwargs):
            if path.name == f"{render_id}.wav":
                raise PermissionError("File in use")
            return unlink(path, *args, **kwargs)

        with patch.object(Path, "unlink", locked):
            result = self.client.delete("/api/history").json()
        self.assertEqual(result["deleted"], [self.other["id"]])
        self.assertEqual(result["failed"][0]["id"], render_id)
        self.assertIn(render_id, studio.jobs)
        self.assertNotIn(self.other["id"], studio.jobs)
        self.assertEqual(self.client.delete("/api/history").json()["count"], 1)
        self.assertEqual(self.snapshot(), self.protected)


if __name__ == "__main__":
    unittest.main()
