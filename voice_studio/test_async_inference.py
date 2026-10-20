import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from voxcpm.model import voxcpm2
from voxcpm.modules.minicpm4.cache import StaticKVCache
from voxcpm.utils import async_inference
from voxcpm.utils.async_inference import AsyncInferenceState, create_async_inference_state
from voxcpm.utils.cuda_graph import GraphedEstimator


class FakeEvent:
    def __init__(self, events):
        self.events = events
        self.ready = False

    def record(self, stream):
        self.events.append("record")
        self.ready = False

    def synchronize(self):
        self.events.append("wait")
        self.ready = True


class FakePinnedScalar:
    def __init__(self, event, events):
        self.event = event
        self.events = events
        self.pending = None

    def copy_(self, value, non_blocking):
        self.events.append("copy")
        if not non_blocking:
            raise AssertionError("Stop transfer must be nonblocking")
        self.pending = int(value.item())
        return self

    def item(self):
        if not self.event.ready:
            raise AssertionError("Pinned value read before its event completed")
        self.events.append("read")
        return self.pending


def fake_async_state(events):
    state = AsyncInferenceState.__new__(AsyncInferenceState)
    state.device = torch.device("cuda")
    state.base_position = torch.empty(1, dtype=torch.long)
    state.residual_position = torch.empty(1, dtype=torch.long)
    state.stop_ready = FakeEvent(events)
    state.stop_value = FakePinnedScalar(state.stop_ready, events)
    return state


class TinyLM:
    def __init__(self):
        self.kv_cache = StaticKVCache(1, 1, 2, 1, "cpu", torch.float32, max_length=16)
        self.positions = []
        self.position_buffers = []

    def embed_tokens(self, text):
        return torch.zeros(*text.shape, 2)

    def __call__(self, inputs_embeds, is_causal):
        hidden = inputs_embeds + 1
        keys = hidden.unsqueeze(1)
        return hidden, [(keys, keys)]

    def forward_step(self, value, position):
        self.positions.append(int(position.item()))
        self.position_buffers.append(position)
        return value + position.unsqueeze(-1)


def tiny_model(events, enabled=False):
    base, residual = TinyLM(), TinyLM()
    base.forward_step = GraphedEstimator(base.forward_step, name="Tiny Base")
    residual.forward_step = GraphedEstimator(residual.forward_step, name="Tiny Residual")
    base.forward_step.paused = True
    stop_flags = iter((1, 1, 0, 1))
    stop_inputs = []
    patches = []

    def encoder(feat):
        events.append("encoder")
        return feat.mean(dim=(-1, -2), keepdim=False).unsqueeze(-1).expand(-1, -1, 2)

    def decoder(**kwargs):
        events.append("patch")
        patches.append(len(patches) + 1)
        return torch.tensor([[[float(patches[-1])]]])

    def stop_head(hidden):
        events.append("stop")
        stop_inputs.append(hidden.clone())
        flag = next(stop_flags)
        return torch.tensor([[1 - flag, flag]])

    model = SimpleNamespace(
        async_inference=enabled,
        config=SimpleNamespace(lm_config=SimpleNamespace(use_mup=False)),
        patch_size=1,
        feat_encoder=encoder,
        enc_to_lm_proj=lambda value: value,
        base_lm=base,
        residual_lm=residual,
        fsq_layer=lambda value: value,
        fusion_concat_proj=lambda value: value[..., :2] + value[..., 2:],
        lm_to_dit_proj=lambda value: value,
        res_to_dit_proj=lambda value: value,
        feat_decoder=decoder,
        stop_proj=lambda value: value,
        stop_actn=lambda value: value,
        stop_head=stop_head,
    )
    return model, stop_inputs, patches


def run_tiny(model, streaming=False):
    text = torch.zeros(1, 2, dtype=torch.long)
    text_mask = torch.ones_like(text)
    feat = torch.zeros(1, 2, 1, 1)
    feat_mask = torch.zeros_like(text)
    return list(voxcpm2.VoxCPM2Model._inference(
        model, text, text_mask, feat, feat_mask, min_len=1, max_len=8, streaming=streaming,
    ))


class AsyncInferenceTests(unittest.TestCase):
    def test_factory_preserves_cpu_streaming_and_default_paths(self):
        with patch.object(async_inference, "AsyncInferenceState") as constructor:
            for enabled, streaming, device in (
                (False, False, "cuda"), (True, True, "cuda"), (True, False, "cpu"),
            ):
                self.assertIsNone(create_async_inference_state(enabled, streaming, device))
            constructor.assert_not_called()
            result = create_async_inference_state(True, False, "cuda:0")
            self.assertIs(result, constructor.return_value)
            constructor.assert_called_once_with("cuda:0")

    def test_stop_read_waits_for_transfer_event_and_positions_reuse_storage(self):
        events = []
        state = fake_async_state(events)
        random_state = torch.get_rng_state().clone()
        with patch.object(async_inference.torch.cuda, "current_stream", return_value=object()) as stream:
            for flag in (0, 1, 0):
                state.queue_stop(torch.tensor(flag))
                self.assertEqual(state.wait_stop(), flag)
            stream.assert_called_with(torch.device("cuda"))
        self.assertEqual(events, ["copy", "record", "wait", "read"] * 3)
        base_pointer = state.base_position.data_ptr()
        residual_pointer = state.residual_position.data_ptr()
        for value in (2, 3, 7):
            self.assertEqual(state.base_step(value).item(), value)
            self.assertEqual(state.residual_step(value + 1).item(), value + 1)
            self.assertEqual(state.base_position.data_ptr(), base_pointer)
            self.assertEqual(state.residual_position.data_ptr(), residual_pointer)
        self.assertNotEqual(base_pointer, residual_pointer)
        self.assertTrue(torch.equal(random_state, torch.get_rng_state()))

    def test_simulated_async_loop_keeps_terminal_patch_stop_inputs_and_rng(self):
        baseline_events, async_events = [], []
        baseline, baseline_stop, baseline_patches = tiny_model(baseline_events)
        candidate, candidate_stop, candidate_patches = tiny_model(async_events, enabled=True)
        state = fake_async_state(async_events)
        random_state = torch.get_rng_state().clone()
        with patch.object(voxcpm2, "tqdm", side_effect=lambda value: value):
            expected = run_tiny(baseline)
            with patch.object(voxcpm2, "create_async_inference_state", return_value=state), patch.object(async_inference.torch.cuda, "current_stream", return_value=object()):
                actual = run_tiny(candidate)
        self.assertTrue(torch.equal(expected[0][0], actual[0][0]))
        self.assertTrue(torch.equal(expected[0][1], actual[0][1]))
        self.assertEqual(expected[0][2], actual[0][2])
        self.assertEqual(baseline_patches, [1, 2, 3, 4])
        self.assertEqual(candidate_patches, baseline_patches)
        for first, second in zip(baseline_stop, candidate_stop):
            self.assertTrue(torch.equal(first, second))
        self.assertLess(async_events.index("stop"), async_events.index("patch"))
        self.assertLess(async_events.index("record"), async_events.index("patch"))
        self.assertLess(async_events.index("encoder", async_events.index("patch")), async_events.index("wait"))
        for original, optimized in ((baseline.base_lm, candidate.base_lm), (baseline.residual_lm, candidate.residual_lm)):
            self.assertEqual(optimized.positions, [2, 3, 4])
            self.assertEqual(optimized.positions, original.positions)
            self.assertTrue(all(value is optimized.position_buffers[0] for value in optimized.position_buffers))
            self.assertEqual(optimized.kv_cache.current_length, original.kv_cache.current_length)
        self.assertTrue(candidate.base_lm.forward_step.paused)
        self.assertFalse(candidate.residual_lm.forward_step.paused)
        self.assertTrue(torch.equal(random_state, torch.get_rng_state()))

    def test_cpu_and_streaming_inference_do_not_construct_cuda_state(self):
        with patch.object(async_inference, "AsyncInferenceState") as constructor, patch.object(voxcpm2, "tqdm", side_effect=lambda value: value):
            for streaming in (False, True):
                model, _, patches = tiny_model([], enabled=True)
                outputs = run_tiny(model, streaming=streaming)
                self.assertEqual(len(patches), 4)
                self.assertEqual(len(outputs), 4 if streaming else 1)
                self.assertTrue(model.base_lm.forward_step.paused)
            constructor.assert_not_called()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_stop_transfers_positions_and_rng(self):
        state = create_async_inference_state(True, False, "cuda:0")
        random_state = torch.cuda.get_rng_state().clone()
        self.assertTrue(state.stop_value.is_pinned())
        base_pointer = state.base_position.data_ptr()
        residual_pointer = state.residual_position.data_ptr()
        for position, flag in ((2, 0), (3, 1), (4, 0), (8, 1)):
            state.queue_stop(torch.tensor(flag, dtype=torch.long, device="cuda:0"))
            base = state.base_step(position)
            residual = state.residual_step(position + 1)
            self.assertEqual(state.wait_stop(), flag)
            self.assertEqual(base.item(), position)
            self.assertEqual(residual.item(), position + 1)
            self.assertEqual(base.data_ptr(), base_pointer)
            self.assertEqual(residual.data_ptr(), residual_pointer)
        self.assertTrue(torch.equal(random_state, torch.cuda.get_rng_state()))


if __name__ == "__main__":
    unittest.main()
