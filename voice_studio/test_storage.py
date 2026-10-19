"""Meaningful checks for durable voices and complete long-text chunking."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import HTTPException

from voice_studio.app import VoiceStore, split_text


class StorageTests(unittest.TestCase):
    def test_saved_voice_survives_new_store_and_source_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            store = VoiceStore(Path(directory))
            render_id = "a" * 32
            sf.write(store.renders / f"{render_id}.wav", np.ones(1600) * 0.1, 16000)
            store.write_json(store.renders / f"{render_id}.json", {
                "kind": "design", "description": "Giọng ấm", "text": "Xin chào.",
                "seed": 42, "duration": 0.1, "sample_rate": 16000,
            })
            voice = store.save_voice(render_id, "Giọng tiếng Việt")
            (store.renders / f"{render_id}.wav").unlink()
            reopened = VoiceStore(Path(directory))
            self.assertEqual(reopened.list_voices()[0]["name"], "Giọng tiếng Việt")
            self.assertEqual(reopened.get_voice(voice["id"])["transcript"], "Xin chào.")
            self.assertTrue(reopened.audio_path(voice["id"]).is_file())

    def test_paths_cannot_escape_library(self):
        with tempfile.TemporaryDirectory() as directory:
            store = VoiceStore(Path(directory))
            for value in ("../config", "../../README.md", "x" * 32, "a" * 31):
                with self.assertRaises(HTTPException):
                    store.get_voice(value)

    def test_long_vietnamese_text_preserves_every_word(self):
        text = ("Xin chào, đây là một câu chuyện mới.\nChúng ta hãy cùng lắng nghe! " * 30).strip()
        chunks = split_text(text)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(" ".join(chunks).split(), text.split())
        self.assertTrue(all(len(chunk) <= 240 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
