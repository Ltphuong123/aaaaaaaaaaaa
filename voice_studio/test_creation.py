import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from fastapi import HTTPException
from pydantic import ValidationError

from voice_studio import app as studio


class CreationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.store = studio.VoiceStore(Path(self.directory.name))
        self.fixture = self.store.root / "fixture.wav"
        sf.write(self.fixture, np.sin(np.arange(4800) * .1) * .1, 48000)
        self.store_patch = patch.object(studio, "store", self.store)
        self.store_patch.start()

    def tearDown(self):
        self.store_patch.stop()
        self.directory.cleanup()

    def run_generation(self, request):
        job_id = "b" * 32
        studio.jobs[job_id] = {"status": "queued"}
        with patch.object(studio, "Client") as client:
            client.return_value.predict.return_value = (str(self.fixture), 77)
            studio.generate(job_id, request)
            calls = list(client.return_value.predict.call_args_list)
        self.assertEqual(studio.jobs[job_id]["status"], "completed", studio.jobs[job_id])
        return self.store.get_render(job_id), calls

    def test_controls_and_actual_seed_reach_model_and_saved_voice(self):
        reference_id = "c" * 32
        sf.write(self.store.references / f"{reference_id}.wav", np.ones(16000) * .1, 16000)
        self.store.write_json(self.store.references / f"{reference_id}.json", {"id": reference_id})
        result, calls = self.run_generation(studio.GenerateRequest(
            kind="design", text="A completely new sentence.", description="ignored in ultimate mode",
            reference_id=reference_id, use_prompt_text=True, prompt_text="Reference sentence.",
            seed=0, cfg_value=2.7, dit_steps=23, do_normalize=True, denoise=True,
        ))
        kwargs = calls[0].kwargs
        self.assertEqual(kwargs["control_instruction"], "")
        self.assertEqual(kwargs["prompt_text_value"], "Reference sentence.")
        self.assertTrue(kwargs["use_prompt_text"])
        self.assertEqual((kwargs["cfg_value"], kwargs["dit_steps"], kwargs["seed_value"]), (2.7, 23, 0))
        self.assertTrue(kwargs["do_normalize"] and kwargs["denoise"])
        self.assertEqual(result["seed"], 77)  # A retry may succeed with a different seed.
        voice = self.store.save_voice(result["id"], "Giọng kiểm tra")
        reopened = studio.VoiceStore(self.store.root).get_voice(voice["id"])
        self.assertEqual(reopened["generation"]["dit_steps"], 23)
        self.assertEqual(reopened["source_reference_id"], reference_id)
        self.assertEqual(reopened["transcript"], "A completely new sentence.")

    def test_optional_description_and_long_output_keep_one_voice(self):
        text = "Một câu chuyện mới và những điều bình dị " * 15
        result, calls = self.run_generation(studio.GenerateRequest(kind="design", text=text))
        self.assertGreater(len(calls), 1)
        self.assertIsNone(calls[0].kwargs["ref_wav"])
        self.assertEqual(calls[0].kwargs["control_instruction"], "")
        self.assertTrue(calls[1].kwargs["use_prompt_text"])
        self.assertEqual(calls[1].kwargs["prompt_text_value"], studio.split_text(text)[0])
        voice = self.store.save_voice(result["id"], "Giọng dài")
        self.assertEqual(voice["transcript"], studio.split_text(text)[0])
        self.assertLess(voice["duration"], result["duration"])

    def test_ultimate_requires_reference_and_matching_transcript(self):
        with self.assertRaises(HTTPException) as error:
            studio.create_job(studio.GenerateRequest(kind="design", text="Xin chào", use_prompt_text=True))
        self.assertIn("âm thanh mẫu", error.exception.detail)
        reference_id = "d" * 32
        self.store.write_json(self.store.references / f"{reference_id}.json", {"id": reference_id})
        with self.assertRaises(HTTPException) as error:
            studio.create_job(studio.GenerateRequest(kind="design", text="Xin chào", use_prompt_text=True, reference_id=reference_id))
        self.assertIn("lời", error.exception.detail)

    def test_model_control_ranges_match_voxcpm(self):
        for settings in ({"cfg_value": 3.1}, {"dit_steps": 51}, {"seed": -1}):
            with self.assertRaises(ValidationError):
                studio.GenerateRequest(kind="design", text="Hello", **settings)


if __name__ == "__main__":
    unittest.main()
