import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from voxcpm import zipenhancer


class FakeCudaPositions(torch.Tensor):
    @staticmethod
    def __new__(cls, values):
        return torch.Tensor._make_subclass(cls, values.clone(), False)

    @property
    def is_cuda(self):
        return True

    def cpu(self):
        return self.as_subclass(torch.Tensor).clone()


class FakeDenoiser(torch.nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events
        self.location = "cuda:0"
        self.encoder = torch.nn.Module()
        self.encoder.pe = torch.arange(8, dtype=torch.float32)
        self.fail_cleanup = False

    def to(self, device):
        self.events.append(str(device))
        if str(device) == "cpu" and self.fail_cleanup:
            raise RuntimeError("cleanup failure")
        self.location = str(device)
        return self


class FakePipeline:
    def __init__(self, wrapped=False, device="cuda:0"):
        self.events = []
        self.module = FakeDenoiser(self.events)
        self.module.location = device
        self.model = SimpleNamespace(model=self.module) if wrapped else self.module
        self.device = torch.device(device)
        self._model_prepare = True
        self.fail_enhance = False

    def __call__(self, input_path, output_path):
        if self.module.location != str(self.device):
            raise AssertionError("Denoiser weights were not restored to the input device")
        self.events.append("enhance")
        if self.device.type == "cuda":
            self.module.encoder.pe = FakeCudaPositions(self.module.encoder.pe)
        Path(output_path).write_bytes(b"identical enhanced audio")
        if self.fail_enhance:
            raise ValueError("original enhancement failure")


class DenoiserMemoryTests(unittest.TestCase):
    def test_optional_offload_restores_each_call_and_keeps_cached_position_bits(self):
        for wrapped in (False, True):
            with self.subTest(wrapped=wrapped), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "source.wav"
                source.write_bytes(b"input")
                pipeline = FakePipeline(wrapped=wrapped)
                with patch.object(zipenhancer, "pipeline", return_value=pipeline), patch.object(zipenhancer.torch.cuda, "empty_cache") as release:
                    enhancer = zipenhancer.ZipEnhancer("local")
                    self.assertFalse(enhancer.offload_after_use)
                    enhancer.offload_after_use = True
                    expected_positions = pipeline.module.encoder.pe.clone()
                    outputs = []
                    for index in range(2):
                        output = Path(directory) / f"output-{index}.wav"
                        outputs.append(Path(enhancer.enhance(str(source), str(output), normalize_loudness=False)).read_bytes())
                        self.assertEqual(pipeline.module.location, "cpu")
                        self.assertFalse(pipeline.module.encoder.pe.is_cuda)
                        self.assertTrue(torch.equal(pipeline.module.encoder.pe, expected_positions))
                    self.assertEqual(outputs[0], outputs[1])
                    self.assertEqual(pipeline.events, ["cuda:0", "enhance", "cpu"] * 2)
                    self.assertEqual(release.call_count, 2)
                    self.assertEqual(pipeline.device, torch.device("cuda:0"))
                    self.assertTrue(pipeline._model_prepare)

    def test_default_and_cpu_paths_keep_original_behavior(self):
        for device, enabled in (("cuda:0", False), ("cpu", True)):
            with self.subTest(device=device, enabled=enabled), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "source.wav"
                output = Path(directory) / "output.wav"
                source.write_bytes(b"input")
                pipeline = FakePipeline(device=device)
                with patch.object(zipenhancer, "pipeline", return_value=pipeline), patch.object(zipenhancer.torch.cuda, "empty_cache") as release:
                    enhancer = zipenhancer.ZipEnhancer("local")
                    enhancer.offload_after_use = enabled
                    enhancer.enhance(str(source), str(output), normalize_loudness=False)
                    self.assertEqual(pipeline.events, ["enhance"])
                    release.assert_not_called()

    def test_failed_enhancement_offloads_and_preserves_original_error(self):
        for fail_cleanup in (False, True):
            with self.subTest(fail_cleanup=fail_cleanup), tempfile.TemporaryDirectory() as directory:
                source = Path(directory) / "source.wav"
                output = Path(directory) / "output.wav"
                source.write_bytes(b"input")
                pipeline = FakePipeline()
                pipeline.fail_enhance = True
                pipeline.module.fail_cleanup = fail_cleanup
                with patch.object(zipenhancer, "pipeline", return_value=pipeline), patch.object(zipenhancer.torch.cuda, "empty_cache") as release:
                    enhancer = zipenhancer.ZipEnhancer("local")
                    enhancer.offload_after_use = True
                    if fail_cleanup:
                        with self.assertLogs(zipenhancer.log.name, level="ERROR"):
                            with self.assertRaisesRegex(RuntimeError, "original enhancement failure"):
                                enhancer.enhance(str(source), str(output), normalize_loudness=False)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "original enhancement failure"):
                            enhancer.enhance(str(source), str(output), normalize_loudness=False)
                        self.assertEqual(pipeline.module.location, "cpu")
                    self.assertEqual(pipeline.events, ["cuda:0", "enhance", "cpu"])
                    release.assert_called_once()
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
