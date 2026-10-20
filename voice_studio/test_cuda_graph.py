import unittest
from unittest.mock import patch

import torch

from voxcpm.utils.cuda_graph import GraphedEstimator


class GraphTests(unittest.TestCase):
    def test_cpu_inputs_use_original_function(self):
        calls = []
        def original(x):
            calls.append(x)
            return x * 2
        wrapper = GraphedEstimator(original)
        value = torch.arange(4, dtype=torch.float32)
        self.assertTrue(torch.equal(wrapper(value), value * 2))
        self.assertEqual(len(calls), 1)
        self.assertIsNone(wrapper.graph)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_stateful_replay_updates_positions_and_survives_cache_reset(self):
        state = torch.zeros(1, 16, device="cuda")
        def step(value, position):
            state[:, position] = value
            mask = torch.arange(16, device=state.device) <= position
            return (state * mask).sum(dim=1)
        wrapper = GraphedEstimator(step, name="Stateful test")
        expected = torch.zeros_like(state)
        for index in range(4):
            value = torch.tensor([float(index + 1)], device="cuda")
            position = torch.tensor([index], device="cuda")
            expected[:, position] = value
            self.assertTrue(torch.equal(wrapper(value, position), expected.sum(dim=1)))
        self.assertFalse(wrapper.disabled)
        state.zero_()
        self.assertTrue(torch.equal(wrapper(torch.tensor([8.0], device="cuda"), torch.tensor([0], device="cuda")), torch.tensor([8.0], device="cuda")))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_variable_prefill_bypasses_single_token_graph(self):
        wrapper = GraphedEstimator(lambda x: x * 2, capture_condition=lambda x: x.shape[1] == 1)
        prefill = torch.ones(1, 20, 4, device="cuda")
        self.assertTrue(torch.equal(wrapper(prefill), prefill * 2))
        self.assertIsNone(wrapper.graph)
        token = torch.ones(1, 1, 4, device="cuda")
        self.assertTrue(torch.equal(wrapper(token), token * 2))
        self.assertIsNotNone(wrapper.graph)
        self.assertTrue(torch.equal(wrapper(prefill), prefill * 2))
        self.assertTrue(torch.equal(wrapper(token * 3), token * 6))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_pause_releases_workspaces_and_resume_recaptures(self):
        wrapper = GraphedEstimator(lambda value: value * 3)
        value = torch.ones(4, device="cuda")
        self.assertTrue(torch.equal(wrapper(value), value * 3))
        self.assertIsNotNone(wrapper.graph)
        torch.cuda.synchronize()
        wrapper.pause()
        self.assertIsNone(wrapper.graph)
        self.assertIsNone(wrapper.inputs)
        self.assertIsNone(wrapper.output)
        self.assertTrue(torch.equal(wrapper(value * 2), value * 6))
        self.assertIsNone(wrapper.graph)
        wrapper.resume()
        self.assertTrue(torch.equal(wrapper(value * 3), value * 9))
        self.assertIsNotNone(wrapper.graph)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_replay_uses_new_inputs_and_preserves_previous_results(self):
        def original(x, y):
            return (x @ y).sin()
        wrapper = GraphedEstimator(original)
        x = torch.ones(4, 4, device="cuda")
        y = torch.eye(4, device="cuda")
        with torch.inference_mode():
            first = wrapper(x, y)
            self.assertFalse(wrapper.disabled)
            self.assertIsNotNone(wrapper.graph)
            self.assertTrue(torch.equal(first, original(x, y)))
            saved = first.clone()
            second = wrapper(x * 2, y)
            self.assertTrue(torch.equal(second, original(x * 2, y)))
            self.assertTrue(torch.equal(first, saved))
            # A different shape bypasses replay without replacing its buffers.
            larger = torch.ones(5, 5, device="cuda")
            self.assertTrue(torch.equal(wrapper(larger, larger), original(larger, larger)))
            self.assertTrue(torch.equal(wrapper(x, y), original(x, y)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_capture_failure_keeps_original_estimator_available(self):
        wrapper = GraphedEstimator(lambda x: x * 2)
        value = torch.ones(4, device="cuda")
        with patch.object(wrapper, "_capture", side_effect=RuntimeError("Unavailable")) as capture:
            with self.assertLogs("voxcpm.utils.cuda_graph", level="ERROR"):
                self.assertTrue(torch.equal(wrapper(value), value * 2))
            self.assertTrue(torch.equal(wrapper(value * 3), value * 6))
            self.assertTrue(wrapper.disabled)
            self.assertEqual(capture.call_count, 1)


if __name__ == "__main__":
    unittest.main()
