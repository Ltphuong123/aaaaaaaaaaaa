"""Optional CUDA replay for fixed-shape inference blocks."""

import logging

import torch

log = logging.getLogger(__name__)


class GraphedEstimator:
    """Replay GPU operations; keep RNG and variable-length prefill outside capture.

    Stateful blocks must retain their buffers and perform idempotent writes at
    the supplied position. CPU counters must be advanced outside this callable.
    """

    def __init__(self, forward, name="LocDiT", capture_condition=None):
        self.forward = forward
        self.name = name
        self.capture_condition = capture_condition
        self.graph = None
        self.inputs = None
        self.output = None
        self.signature = None
        self.disabled = False
        self.paused = False

    def pause(self):
        """Release private GPU workspaces before memory-intensive preprocessing."""
        self.paused = True
        self.graph = self.inputs = self.output = self.signature = None

    def resume(self):
        self.paused = False

    @staticmethod
    def _signature(args):
        return tuple((tuple(x.shape), x.dtype, x.device) for x in args)

    @torch.inference_mode()
    def __call__(self, *args):
        if self.disabled or self.paused or not args or not all(isinstance(x, torch.Tensor) and x.is_cuda for x in args):
            return self.forward(*args)
        if self.capture_condition is not None and not self.capture_condition(*args):
            return self.forward(*args)
        signature = self._signature(args)
        if self.graph is None:
            try:
                self._capture(args)
                self.signature = signature
            except Exception:
                log.exception("%s CUDA replay disabled; using the original block", self.name)
                self.disabled = True
                self.graph = self.inputs = self.output = None
                torch.cuda.empty_cache()
                return self.forward(*args)
        if signature != self.signature:
            return self.forward(*args)
        for static, value in zip(self.inputs, args):
            static.copy_(value)
        self.graph.replay()
        return self.output.clone()

    def _capture(self, args):
        self.inputs = tuple(x.clone() for x in args)
        stream = torch.cuda.Stream(device=args[0].device)
        current = torch.cuda.current_stream(args[0].device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.forward(*self.inputs)
        current.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = self.forward(*self.inputs)
        graph.replay()
        expected = self.forward(*args)
        if not torch.equal(expected, output):
            raise RuntimeError("Captured block differs from original output")
        self.graph, self.output = graph, output
        log.warning("%s CUDA replay ready; output matches original exactly", self.name)
