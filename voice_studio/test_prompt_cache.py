import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch

from voxcpm.core import VoxCPM
from voxcpm.model.voxcpm2 import VoxCPM2Model
from voxcpm.utils.prompt_cache import PromptFeatureCache


class PromptCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.audio = Path(self.directory.name) / "sample.wav"
        self.audio.write_bytes(b"reference audio")

    def tearDown(self):
        self.directory.cleanup()

    def test_identical_uploads_reuse_features_and_edits_invalidate(self):
        cache = PromptFeatureCache(2)
        key = cache.key("Original words.", str(self.audio), str(self.audio))
        features = {"audio_feat": torch.ones(2, 4, 64), "mode": "ref_continuation"}
        cache.put(key, features)
        copied = self.audio.with_name("upload-copy.wav")
        shutil.copyfile(self.audio, copied)
        self.assertIsNotNone(cache.get(cache.key("Original words.", str(copied), str(copied))))
        self.assertIsNone(cache.get(cache.key("Different words.", str(copied), str(copied))))
        self.audio.write_bytes(b"changed audio!!")
        self.assertIsNone(cache.get(cache.key("Original words.", str(self.audio), str(self.audio))))
        features["audio_feat"].zero_()
        self.assertEqual(cache.get(key)["audio_feat"].sum().item(), 512)
        self.assertEqual(cache.get(key)["audio_feat"].device.type, "cpu")

    def test_entry_and_byte_limits_evict_old_features(self):
        cache = PromptFeatureCache(2, max_bytes=32)
        cache.put("a", {"features": torch.zeros(4)})
        cache.put("b", {"features": torch.zeros(4)})
        cache.get("a")
        cache.put("c", {"features": torch.zeros(4)})
        self.assertIsNone(cache.get("b"))
        self.assertIsNotNone(cache.get("a"))
        cache.put("oversize", {"features": torch.zeros(100)})
        self.assertIsNone(cache.get("oversize"))
        cache.clear()
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache._bytes, 0)
        byte_limited = PromptFeatureCache(8, max_bytes=32)
        for key in ("a", "b", "c"):
            byte_limited.put(key, {"features": torch.zeros(4)})
        self.assertIsNone(byte_limited.get("a"))
        self.assertIsNotNone(byte_limited.get("c"))
        self.assertLessEqual(byte_limited._bytes, 32)

    def test_generator_reuses_sample_for_new_text_without_changing_quality_settings(self):
        pipeline = VoxCPM.__new__(VoxCPM)
        pipeline.prompt_feature_cache = PromptFeatureCache(2)
        pipeline.denoiser = None
        pipeline.text_normalizer = None
        pipeline.tts_model = Mock(spec=VoxCPM2Model)
        features = {"audio_feat": torch.ones(2, 4, 64), "ref_audio_feat": torch.ones(2, 4, 64), "mode": "ref_continuation"}
        pipeline.tts_model.build_prompt_cache.return_value = features
        def generate(**kwargs):
            yield torch.ones(1, 480), None, None
        pipeline.tts_model._generate_with_prompt_cache.side_effect = generate
        settings = dict(prompt_text="Sample words.", prompt_wav_path=str(self.audio),
                        reference_wav_path=str(self.audio), inference_timesteps=10, cfg_value=2.0, seed=42)
        first = pipeline.generate(text="First sentence.", **settings)
        second = pipeline.generate(text="New sentence.", **settings)
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(pipeline.tts_model.build_prompt_cache.call_count, 1)
        calls = pipeline.tts_model._generate_with_prompt_cache.call_args_list
        self.assertEqual(calls[1].kwargs["target_text"], "New sentence.")
        for call in calls:
            self.assertEqual(call.kwargs["inference_timesteps"], 10)
            self.assertEqual(call.kwargs["cfg_value"], 2.0)
            self.assertEqual(call.kwargs["seed"], 42)
            self.assertTrue(call.kwargs["retry_badcase"])
            self.assertEqual(call.kwargs["prompt_cache"]["mode"], "ref_continuation")
        # Denoising continues to process each sample through the original path.
        pipeline.generate(text="Denoised.", denoise=True, **settings)
        pipeline.generate(text="Denoised again.", denoise=True, **settings)
        self.assertEqual(pipeline.tts_model.build_prompt_cache.call_count, 3)

    def test_cache_disabled_keeps_original_behavior(self):
        disabled = PromptFeatureCache(0)
        self.assertIsNone(disabled.key("Text", str(self.audio), str(self.audio)))


if __name__ == "__main__":
    unittest.main()
