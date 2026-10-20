"""Bounded CPU storage for deterministic reference-audio features."""

from collections import OrderedDict
import hashlib
from pathlib import Path
import threading

import torch


class PromptFeatureCache:
    def __init__(self, max_entries: int = 0, max_bytes: int = 16 * 1024 * 1024):
        self.max_entries = max(0, max_entries)
        self.max_bytes = max(0, max_bytes)
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    @staticmethod
    def _audio_hash(path: str | None) -> str | None:
        if path is None:
            return None
        digest = hashlib.sha256()
        with Path(path).open("rb") as audio:
            for chunk in iter(lambda: audio.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def key(self, prompt_text, prompt_path, reference_path):
        if not self.max_entries or not self.max_bytes:
            return None
        # Content hashes also work when Gradio copies an upload to a new path.
        # Editing/replacing audio, even with the same filename, invalidates it.
        prompt_hash = self._audio_hash(prompt_path)
        ref_hash = prompt_hash if prompt_path == reference_path else self._audio_hash(reference_path)
        return prompt_text, prompt_hash, ref_hash

    def get(self, key):
        if key is None:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return dict(entry[0])

    def put(self, key, features: dict):
        if key is None or not self.max_entries or not self.max_bytes:
            return
        # The model returns CPU tensors. Keep an independent copy and never
        # retain GPU activations or grow VRAM usage with the number of voices.
        value = {
            name: item.detach().cpu().clone() if isinstance(item, torch.Tensor) else item
            for name, item in features.items()
        }
        size = sum(
            item.numel() * item.element_size() if isinstance(item, torch.Tensor)
            else len(str(item).encode("utf-8"))
            for item in value.values()
        )
        if size > self.max_bytes:
            return
        with self._lock:
            previous = self._entries.pop(key, None)
            if previous:
                self._bytes -= previous[1]
            while self._entries and (len(self._entries) >= self.max_entries or self._bytes + size > self.max_bytes):
                _, (_, old_size) = self._entries.popitem(last=False)
                self._bytes -= old_size
            self._entries[key] = value, size
            self._bytes += size

    def clear(self):
        with self._lock:
            self._entries.clear()
            self._bytes = 0
