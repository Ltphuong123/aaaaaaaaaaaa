import unittest
from unittest.mock import patch

import torch

from voxcpm.modules.locdit.unified_cfm import CfmConfig, UnifiedCFM
from voxcpm.utils.cfm_graph import GraphedCFMSolver
from voxcpm.utils.cuda_graph import GraphedEstimator


class TestEstimator(torch.nn.Module):
    def forward(self, x, mu, t, cond, dt):
        return x * .2 + cond * .3 + mu.mean(dim=1)[:, None, None] + t[:, None, None] * .1


class CFMGraphTests(unittest.TestCase):
    def test_time_grids_reuse_exact_values_without_changing_rng_or_mode(self):
        model = UnifiedCFM(2, CfmConfig(), TestEstimator()).eval()
        model.inference_time_grid_cache = {}
        mu, cond = torch.ones(1, 6), torch.ones(1, 2, 4)
        for steps, sway in ((4, 1.0), (6, 1.0), (10, 1.0), (6, .5), (4, 1.0)):
            torch.manual_seed(42)
            actual = model(mu, steps, 4, cond, sway_sampling_coef=sway)
            state_after_cached = torch.get_rng_state()
            cache, model.inference_time_grid_cache = model.inference_time_grid_cache, None
            torch.manual_seed(42)
            expected = model(mu, steps, 4, cond, sway_sampling_coef=sway)
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(state_after_cached, torch.get_rng_state()))
            model.inference_time_grid_cache = cache
        self.assertEqual(len(cache), 4)
        original = {key: value.clone() for key, value in cache.items()}
        model(mu, 4, 4, cond)
        self.assertTrue(all(torch.equal(value, cache[key]) for key, value in original.items()))
        for steps in range(11, 21):
            model(mu, steps, 4, cond)
        self.assertEqual(len(cache), 8)

    def make_solver(self):
        model = UnifiedCFM(2, CfmConfig(), TestEstimator()).eval()
        estimator = GraphedEstimator(model.estimator.forward, name="Test estimator")
        model.estimator.forward = estimator
        return model, GraphedCFMSolver(model.solve_euler, estimator_graph=estimator)

    def inputs(self, steps=6, device="cuda", batch=1):
        return (torch.rand(batch, 2, 4, device=device), torch.linspace(1, 0, steps + 1, device=device),
                torch.rand(batch, 6, device=device), torch.rand(batch, 2, 4, device=device))

    def eager(self, model, *args, **kwargs):
        child = model.estimator.forward
        was_paused, child.paused = child.paused, True
        try:
            return model.solve_euler(*args, **kwargs)
        finally:
            child.paused = was_paused

    def test_cpu_retains_original_solver(self):
        model, solver = self.make_solver()
        args = self.inputs(device="cpu")
        self.assertTrue(torch.equal(solver(*args), model.solve_euler(*args)))
        self.assertIsNone(solver.replay)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_all_modes_and_cfg_match_eager_with_new_inputs_and_rng_unchanged(self):
        model, solver = self.make_solver()
        for steps in (4, 6, 10, 4):
            for cfg, zero_star in ((2.0, True), (1.5, False)):
                for _ in range(2):
                    args = self.inputs(steps)
                    before = torch.cuda.get_rng_state()
                    expected = self.eager(model, *args, cfg_value=cfg, use_cfg_zero_star=zero_star)
                    result = solver(*args, cfg_value=cfg, use_cfg_zero_star=zero_star)
                    self.assertTrue(torch.equal(result, expected))
                    self.assertTrue(torch.equal(before, torch.cuda.get_rng_state()))
                    self.assertIsNotNone(solver.replay.graph)
                    self.assertFalse(solver.disabled_signatures)
                    self.assertIsNone(model.estimator.forward.graph)
                    saved = result.clone()
                    solver(*self.inputs(steps), cfg_value=cfg, use_cfg_zero_star=zero_star)
                    self.assertTrue(torch.equal(result, saved))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_pause_resume_shape_change_and_unsupported_steps(self):
        model, solver = self.make_solver()
        args = self.inputs(4)
        self.assertTrue(torch.equal(solver(*args), model.solve_euler(*args)))
        solver.pause()
        self.assertIsNone(solver.replay)
        self.assertTrue(torch.equal(solver(*args), model.solve_euler(*args)))
        self.assertIsNone(solver.replay)
        solver.resume()
        self.assertTrue(torch.equal(solver(*args), model.solve_euler(*args)))
        self.assertIsNotNone(solver.replay.graph)
        changed = self.inputs(6, batch=2)
        self.assertTrue(torch.equal(solver(*changed), model.solve_euler(*changed)))
        signature = solver.signature
        unsupported = self.inputs(3)
        self.assertTrue(torch.equal(solver(*unsupported), model.solve_euler(*unsupported)))
        self.assertEqual(signature, solver.signature)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_failed_capture_falls_back_and_does_not_retry_every_patch(self):
        model, solver = self.make_solver()
        args = self.inputs()
        model.estimator.forward.pause()
        with patch.object(GraphedEstimator, "_capture", side_effect=RuntimeError("Unavailable")) as capture:
            with self.assertLogs("voxcpm.utils.cuda_graph", level="ERROR"):
                self.assertTrue(torch.equal(solver(*args), model.solve_euler(*args)))
            self.assertTrue(torch.equal(solver(*args), model.solve_euler(*args)))
            self.assertEqual(capture.call_count, 1)
        self.assertIsNone(solver.replay)
        self.assertTrue(model.estimator.forward.paused)


if __name__ == "__main__":
    unittest.main()
