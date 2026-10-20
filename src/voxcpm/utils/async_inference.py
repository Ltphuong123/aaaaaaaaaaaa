"""Optional inference-loop transfers with unchanged model operations."""

import torch


class AsyncInferenceState:
    """Reuse position buffers and wait only for the early stop-head transfer."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.base_position = torch.empty(1, dtype=torch.long, device=self.device)
        self.residual_position = torch.empty(1, dtype=torch.long, device=self.device)
        self.stop_value = torch.empty((), dtype=torch.long, device="cpu", pin_memory=True)
        self.stop_ready = torch.cuda.Event()

    def queue_stop(self, value):
        self.stop_value.copy_(value, non_blocking=True)
        self.stop_ready.record(torch.cuda.current_stream(self.device))

    def wait_stop(self):
        self.stop_ready.synchronize()
        return int(self.stop_value.item())

    def base_step(self, position):
        return self.base_position.fill_(position)

    def residual_step(self, position):
        return self.residual_position.fill_(position)


def create_async_inference_state(enabled, streaming, device):
    if not enabled or streaming or torch.device(device).type != "cuda":
        return None
    return AsyncInferenceState(device)
