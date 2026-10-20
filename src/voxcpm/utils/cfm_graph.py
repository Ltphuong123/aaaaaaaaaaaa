"""Replay the deterministic CFM solver without changing noise or Euler steps."""

import torch

from .cuda_graph import GraphedEstimator


class GraphedCFMSolver:
    """Keep a single fixed-shape solver graph, releasing it on mode changes.

    Noise is sampled by UnifiedCFM.forward before this callable. Capturing the
    solver preserves the RNG sequence as well as the original operation order.
    Unsupported step counts and failed captures retain the eager solver.
    """

    def __init__(self, forward, estimator_graph=None):
        self.forward = forward
        self.estimator_graph = estimator_graph
        self.replay = None
        self.signature = None
        self.paused = False
        self.disabled_signatures = set()

    def pause(self):
        self.paused = True
        self.replay = self.signature = None

    def resume(self):
        self.paused = False

    @torch.inference_mode()
    def __call__(self, x, t_span, mu, cond, cfg_value=1.0, use_cfg_zero_star=True):
        args = (x, t_span, mu, cond)
        options = dict(cfg_value=cfg_value, use_cfg_zero_star=use_cfg_zero_star)
        if self.paused or len(t_span) - 1 not in (4, 6, 10) or not all(value.is_cuda for value in args):
            return self.forward(*args, **options)
        signature = (GraphedEstimator._signature(args), float(cfg_value), bool(use_cfg_zero_star))
        if signature in self.disabled_signatures:
            return self.forward(*args, **options)
        if signature != self.signature:
            # Never accumulate a private CUDA memory pool for each read mode.
            self.replay = None
            self.signature = signature
            self.replay = GraphedEstimator(
                lambda *values: self.forward(*values, **options),
                name=f"CFM {len(t_span) - 1}-step solver",
            )
        if self.replay.graph is not None:
            return self.replay(*args)

        # The large graph records raw estimator operations, never nested graph
        # creation/replay. Its fallback can still use the smaller estimator graph.
        child = self.estimator_graph
        was_paused = child.paused if child is not None else False
        if child is not None:
            child.pause()
        try:
            result = self.replay(*args)
        finally:
            if child is not None and not was_paused:
                child.resume()
        if self.replay.disabled:
            self.disabled_signatures.add(signature)
            self.replay = self.signature = None
        return result
