import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from pydantic import ValidationError

from voice_studio import app as studio


class ReadingTests(unittest.TestCase):
    def test_presets_preserve_saved_voice_and_report_real_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            store = studio.VoiceStore(Path(directory))
            voice_id = "a" * 32
            voice_dir = store.voices / voice_id
            voice_dir.mkdir()
            fixture = voice_dir / "reference.wav"
            sf.write(fixture, np.sin(np.arange(4800) * .1) * .1, 48000)
            metadata = {"id": voice_id, "name": "Saved voice", "transcript": "Original sample.", "seed": 0}
            store.write_json(voice_dir / "voice.json", metadata)
            with patch.object(studio, "store", store), patch.object(studio, "Client") as client:
                client.return_value.predict.return_value = (str(fixture), 0)
                for mode, steps, ultimate in [("fast", 4, False), ("balanced", 6, True), ("faithful", 10, True)]:
                    with self.subTest(mode=mode):
                        job_id = {"fast": "b", "balanced": "c", "faithful": "d"}[mode] * 32
                        studio.jobs[job_id] = {"status": "queued"}
                        request = studio.GenerateRequest(kind="speak", text="A new sentence.", voice_id=voice_id, seed=0, read_mode=mode)
                        studio.generate(job_id, request)
                        self.assertEqual(studio.jobs[job_id]["status"], "completed", studio.jobs[job_id])
                        call = client.return_value.predict.call_args.kwargs
                        self.assertEqual(call["dit_steps"], steps)
                        self.assertEqual(call["use_prompt_text"], ultimate)
                        self.assertEqual(call["prompt_text_value"], "Original sample." if ultimate else "")
                        self.assertEqual(call["seed_value"], 0)
                        self.assertIn(voice_id, call["ref_wav"]["path"])
                        result = store.get_render(job_id)
                        self.assertEqual(result["generation"]["read_mode"], mode)
                        self.assertEqual(result["generation"]["mode"], "ultimate" if ultimate else "clone")
                        self.assertGreaterEqual(result["processing_seconds"], 0)
                        self.assertEqual(json.loads((voice_dir / "voice.json").read_text()), metadata)

    def test_default_is_balanced_and_invalid_modes_are_rejected(self):
        self.assertEqual(studio.GenerateRequest(kind="speak", text="Hello").read_mode, "balanced")
        with self.assertRaises(ValidationError):
            studio.GenerateRequest(kind="speak", text="Hello", read_mode="unknown")


if __name__ == "__main__":
    unittest.main()
