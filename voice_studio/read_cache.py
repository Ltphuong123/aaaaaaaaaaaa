"""Small index of reusable reads; audio remains in normal, deletable history."""

import json
from collections import OrderedDict
from pathlib import Path


def file_fingerprint(paths):
    try:
        result = []
        for path in paths:
            if not path.is_file():
                return None
            stat = path.stat()
            result.append((str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns))
        return tuple(result)
    except OSError:
        return None


def engine_fingerprint(root: Path, backend: str):
    # Only the bundled local service has a known process marker and weights.
    if backend not in {"http://127.0.0.1:8808", "http://localhost:8808"}:
        return None
    model = root / "pretrained_models" / "VoxCPM2"
    marker = root / ".runtime" / "server.pid"
    paths = [marker, root / "app.py", model / "model.safetensors", model / "config.json",
             model / "tokenizer.json", model / "tokenizer_config.json", model / "special_tokens_map.json"]
    paths.extend(path for path in (model / "audiovae.pth", model / "audiovae.safetensors") if path.is_file())
    if not any(path.name.startswith("audiovae.") for path in paths):
        return None
    paths.extend(sorted((root / "src" / "voxcpm").rglob("*.py")))
    fingerprint = file_fingerprint(paths)
    try:
        process = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return (process, fingerprint) if fingerprint and process.isdecimal() else None


class ReadCache:
    def __init__(self, max_entries=32):
        self.max_entries = max_entries
        self.entries = OrderedDict()

    @staticmethod
    def paths(renders: Path, render_id: str):
        return [renders / f"{render_id}.json", renders / f"{render_id}.wav", renders / f"{render_id}.sample.wav"]

    def clear(self):
        self.entries.clear()

    def get(self, key, renders: Path):
        entry = self.entries.pop(key, None)
        if entry is None:
            return None
        render_id, fingerprint = entry
        paths = self.paths(renders, render_id)
        if file_fingerprint(paths) != fingerprint:
            return None
        try:
            render = json.loads(paths[0].read_text(encoding="utf-8"))
            if not isinstance(render, dict) or render.get("kind") != "speak" or render.get("id") != render_id:
                return None
        except (OSError, ValueError):
            return None
        self.entries[key] = entry
        return render

    def put(self, key, render_id: str, renders: Path):
        fingerprint = file_fingerprint(self.paths(renders, render_id))
        if fingerprint is None:
            return
        self.entries.pop(key, None)
        self.entries[key] = (render_id, fingerprint)
        while len(self.entries) > self.max_entries:
            self.entries.popitem(last=False)
