from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from gradio_client import Client, handle_file
from pydantic import BaseModel, Field

from voice_studio.read_cache import ReadCache, engine_fingerprint

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("VOICE_STUDIO_DATA", ROOT / "data" / "voice_studio"))
BACKEND = os.environ.get("VOXCPM_URL", "http://127.0.0.1:8808").rstrip("/")
log = logging.getLogger(__name__)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class VoiceStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.voices = self.root / "voices"
        self.renders = self.root / "renders"
        self.references = self.root / "references"
        self.voices.mkdir(parents=True, exist_ok=True)
        self.renders.mkdir(parents=True, exist_ok=True)
        self.references.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def valid_id(value: str) -> str:
        if not re.fullmatch(r"[a-f0-9]{32}", value):
            raise HTTPException(404, "Không tìm thấy dữ liệu.")
        return value

    @staticmethod
    def write_json(path: Path, value: dict):
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def list_voices(self) -> list[dict]:
        voices = []
        for path in self.voices.glob("*/voice.json"):
            try:
                voice = json.loads(path.read_text(encoding="utf-8"))
                if (path.parent / "reference.wav").is_file():
                    voices.append(voice)
            except (OSError, ValueError):
                log.warning("Could not read voice metadata: %s", path)
        return sorted(voices, key=lambda v: (v["origin"] != "preset", v["created_at"]))

    def get_voice(self, voice_id: str) -> dict:
        path = self.voices / self.valid_id(voice_id) / "voice.json"
        if not path.is_file():
            raise HTTPException(404, "Giọng này không còn trong thư viện. Hãy chọn lại giọng.")
        return json.loads(path.read_text(encoding="utf-8"))

    def audio_path(self, voice_id: str) -> Path:
        self.get_voice(voice_id)
        return self.voices / voice_id / "reference.wav"

    def delete_voice(self, voice_id: str):
        voice_id = self.valid_id(voice_id)
        directory = self.voices / voice_id
        resolved = directory.resolve()
        # Only remove this UUID directory, never a junction/symlink target or
        # a path outside the voice library. Renders/references stay independent.
        if resolved.parent != self.voices.resolve() or resolved.name != voice_id or resolved != directory.absolute():
            raise HTTPException(403, "Không thể xóa thư mục giọng này.")
        voice = self.get_voice(voice_id)
        if voice.get("origin") != "created":
            raise HTTPException(403, "Chỉ có thể xóa giọng do bạn tạo. Giọng có sẵn được giữ trong thư viện.")
        try:
            shutil.rmtree(resolved)
        except OSError as error:
            raise HTTPException(409, "Chưa xóa được giọng. Hãy dừng phát giọng mẫu rồi thử lại.") from error

    def get_render(self, render_id: str) -> dict:
        path = self.renders / f"{self.valid_id(render_id)}.json"
        if not path.is_file():
            raise HTTPException(404, "Không tìm thấy đoạn âm thanh này.")
        return json.loads(path.read_text(encoding="utf-8"))

    def get_reference(self, reference_id: str) -> dict:
        path = self.references / f"{self.valid_id(reference_id)}.json"
        if not path.is_file():
            raise HTTPException(404, "Không tìm thấy âm thanh mẫu. Hãy tải lên hoặc thu âm lại.")
        return json.loads(path.read_text(encoding="utf-8"))

    def reference_path(self, reference_id: str) -> Path:
        self.get_reference(reference_id)
        return self.references / f"{reference_id}.wav"

    def save_voice(self, render_id: str, name: str, origin: str = "created") -> dict:
        render = self.get_render(render_id)
        if render["kind"] != "design":
            raise HTTPException(400, "Hãy tạo và nghe thử một giọng trước khi lưu.")
        name = " ".join(name.split())
        if not name:
            raise HTTPException(400, "Bạn chưa đặt tên cho giọng.")
        if any(v["name"].casefold() == name.casefold() for v in self.list_voices()):
            raise HTTPException(409, "Tên giọng đã tồn tại. Hãy chọn một tên khác.")
        voice_id = uuid.uuid4().hex
        directory = self.voices / voice_id
        directory.mkdir()
        sample = self.renders / f"{render_id}.sample.wav"
        shutil.copyfile(sample if sample.is_file() else self.renders / f"{render_id}.wav", directory / "reference.wav")
        sample_info = sf.info(directory / "reference.wav")
        voice = {
            "id": voice_id, "name": name, "description": render["description"],
            "transcript": render.get("voice_sample_text", render["text"]), "seed": render["seed"], "origin": origin,
            "duration": round(sample_info.duration, 2), "sample_rate": sample_info.samplerate,
            "generation": render.get("generation", {}),
            "source_reference_id": render.get("reference_id"),
            "created_at": now(), "audio_url": f"/api/voices/{voice_id}/audio",
        }
        self.write_json(directory / "voice.json", voice)
        return voice

    def history(self, limit: int | None = 20) -> list[dict]:
        rows = []
        for path in self.renders.glob("*.json"):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                if (isinstance(item, dict) and item.get("kind") == "speak"
                        and re.fullmatch(r"[a-f0-9]{32}", path.stem) and item.get("id") == path.stem):
                    rows.append(item)
            except (OSError, ValueError):
                continue
        return sorted(rows, key=lambda r: r.get("created_at", ""), reverse=True)[:limit]

    def delete_reading(self, render_id: str):
        render_id = self.valid_id(render_id)
        # Validate every exact file before removing anything. Never follow links
        # or remove the separate design renders, reference uploads or voice library.
        paths = [self.renders / f"{render_id}{suffix}" for suffix in (".wav", ".sample.wav", ".json")]
        for path in paths:
            if path.resolve().parent != self.renders.resolve() or path.resolve() != path.absolute():
                raise HTTPException(403, "Không thể xóa file bản đọc này.")
        if self.get_render(render_id).get("kind") != "speak":
            raise HTTPException(403, "Chỉ có thể xóa bản đọc văn bản trong lịch sử.")
        try:
            # Metadata goes last so an interrupted deletion can be retried.
            for path in paths:
                path.unlink(missing_ok=True)
        except OSError as error:
            raise HTTPException(409, "Chưa xóa được bản đọc. Hãy dừng phát hoặc tải file WAV rồi thử lại.") from error


def split_text(text: str, limit: int = 240) -> list[str]:
    """Keep every word, while bounding the model context on an 8 GB GPU."""
    chunks, current = [], ""
    sentences = re.split(r"(?<=[.!?…;])\s+|\n+", text.strip())
    for sentence in sentences:
        for word in sentence.split():
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > limit:
                chunks.append(current)
                current = word
            else:
                current = candidate
        if current and len(current) >= limit // 2:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


class GenerateRequest(BaseModel):
    kind: Literal["design", "speak"]
    text: str = Field(min_length=1, max_length=6000)
    description: str = Field(default="", max_length=500)
    voice_id: str | None = None
    seed: int = Field(default=42, ge=0, le=4294967295)
    random_seed: bool = False
    reference_id: str | None = None
    use_prompt_text: bool = False
    prompt_text: str = Field(default="", max_length=12000)
    cfg_value: float = Field(default=2.0, ge=1.0, le=3.0)
    dit_steps: int = Field(default=10, ge=1, le=50)
    do_normalize: bool = False
    denoise: bool = False
    read_mode: Literal["fast", "balanced", "faithful"] = "balanced"


# Keep all original creation controls independent from the reading presets.
READ_MODES = {
    "fast": {"dit_steps": 4, "use_prompt_text": False},
    "balanced": {"dit_steps": 6, "use_prompt_text": True},
    "faithful": {"dit_steps": 10, "use_prompt_text": True},
}


class TranscribeRequest(BaseModel):
    language: Literal["auto", "vi", "en", "zh", "ja", "ko", "fr", "de", "es"] = "auto"


class SaveRequest(BaseModel):
    render_id: str
    name: str = Field(min_length=1, max_length=60)


store = VoiceStore(DATA)
read_cache = ReadCache(max_entries=32)


class BackendSession:
    """Reuse a Gradio connection from the single serialized TTS worker."""

    def __init__(self):
        self.client = None
        self.key = None

    def close(self):
        with job_lock:
            read_cache.clear()
        client, self.client, self.key = self.client, None, None
        if client is not None:
            try:
                client.close()
            except Exception:
                log.debug("Could not close the previous backend session", exc_info=True)
            try:
                client.executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                log.debug("Could not stop the previous backend executor", exc_info=True)

    def predict(self, **kwargs):
        # Include the factory so patched clients and alternate data stores stay
        # isolated in tests. A new server connection fetches its own API schema.
        key = (BACKEND, store.renders.resolve(), Client)
        if self.client is None or self.key != key:
            self.close()
            self.client = Client(BACKEND, verbose=False, download_files=store.renders / "downloads", analytics_enabled=False)
            self.key = key
        try:
            return self.client.predict(**kwargs)
        except Exception:
            # A restarted backend may invalidate its event stream. Reconnect
            # for the next submission; never automatically synthesize twice.
            self.close()
            raise


backend_session = BackendSession()
jobs: dict[str, dict] = {}
job_lock = threading.Lock()
save_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-studio")
asr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="voice-transcript")
asr_model = None
app = FastAPI(title="Giọng Nhà", docs_url=None, redoc_url=None)


@app.middleware("http")
async def refresh_ui_assets(request, call_next):
    response = await call_next(request)
    if request.url.path in {"/", "/index.html", "/app.js", "/styles.css"}:
        response.headers["Cache-Control"] = "no-cache"
    return response


def update_job(job_id: str, **changes):
    with job_lock:
        jobs[job_id].update(changes)


def reading_cache_key(request: GenerateRequest, reference_path: Path | None, prompt_text: str, settings: dict):
    if request.kind != "speak" or request.random_seed or request.denoise or reference_path is None:
        return None
    engine = engine_fingerprint(ROOT, BACKEND)
    if engine is None:
        return None
    with reference_path.open("rb") as reference:
        reference_hash = hashlib.file_digest(reference, "sha256").hexdigest()
    values = request.model_dump()
    values["text"] = request.text.strip()
    return (str(store.root), BACKEND, engine, reference_hash, prompt_text,
            json.dumps(values, sort_keys=True, ensure_ascii=False), json.dumps(settings, sort_keys=True))


def reuse_read(job_id: str, request: GenerateRequest, voice: dict, cache_key, started: float):
    if cache_key is None:
        return False
    # History deletion uses the same lock, so an indexed source cannot disappear
    # between verification and copying. No audio lives outside normal history.
    with job_lock:
        previous = read_cache.get(cache_key, store.renders)
        if previous is None:
            return False
        for suffix in (".wav", ".sample.wav"):
            shutil.copyfile(store.renders / f"{previous['id']}{suffix}", store.renders / f"{job_id}{suffix}")
        elapsed = time.perf_counter() - started
        render = {**previous, "id": job_id, "voice_id": voice["id"], "voice_name": voice["name"],
                  "text": request.text.strip(), "description": request.description.strip(),
                  "created_at": now(), "audio_url": f"/api/renders/{job_id}/audio",
                  "download_url": f"/api/renders/{job_id}/download", "cache_hit": True,
                  "processing_seconds": round(elapsed, 2),
                  "real_time_factor": round(elapsed / max(previous["duration"], .01), 2)}
        store.write_json(store.renders / f"{job_id}.json", render)
        read_cache.put(cache_key, job_id, store.renders)
        jobs[job_id].update(status="completed", message="Đã dùng lại bản đọc đã tạo.", result=render)
        return True


def generate(job_id: str, request: GenerateRequest):
    started = time.perf_counter()
    try:
        update_job(job_id, status="running", message="Đang kết nối mô hình…")
        voice = store.get_voice(request.voice_id) if request.kind == "speak" else None
        reference_path = store.audio_path(voice["id"]) if voice else (
            store.reference_path(request.reference_id) if request.reference_id else None
        )
        prompt_text = voice["transcript"] if voice else request.prompt_text.strip()
        settings = READ_MODES[request.read_mode] if voice else {
            "dit_steps": request.dit_steps, "use_prompt_text": request.use_prompt_text,
        }
        use_prompt = settings["use_prompt_text"]
        requested_seed = secrets.randbits(32) if request.random_seed else request.seed
        cache_key = reading_cache_key(request, reference_path, prompt_text, settings)
        if voice and reuse_read(job_id, request, voice, cache_key, started):
            return
        parts = split_text(request.text)
        waves, sample_rate, actual_seeds = [], None, []
        for index, part in enumerate(parts):
            update_job(job_id, message=f"Đang đọc đoạn {index + 1}/{len(parts)}…", current=index + 1, total=len(parts))
            result = backend_session.predict(
                text=part,
                control_instruction=request.description if not use_prompt else "",
                ref_wav=handle_file(str(reference_path)) if reference_path else None,
                use_prompt_text=use_prompt,
                prompt_text_value=prompt_text if use_prompt else "",
                cfg_value=request.cfg_value, do_normalize=request.do_normalize, denoise=request.denoise,
                dit_steps=settings["dit_steps"], seed_value=requested_seed, api_name="/generate",
            )
            actual_seeds.append(int(result[1]) if result[1] is not None else requested_seed)
            audio, rate = sf.read(result[0], dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if not len(audio) or not np.isfinite(audio).all():
                raise ValueError("Mô hình chưa tạo được âm thanh hợp lệ. Hãy thử lại.")
            if sample_rate is not None and sample_rate != rate:
                raise ValueError("Các đoạn âm thanh có tần số khác nhau. Hãy thử lại.")
            if waves:
                waves.append(np.zeros(int(rate * 0.18), dtype=np.float32))
            waves.append(audio)
            sample_rate = rate
            if index == 0:
                sample_path = store.renders / f"{job_id}.sample.wav"
                sf.write(sample_path, audio, rate, subtype="PCM_16")
                # Keep the newly designed voice consistent across subsequent chunks.
                if reference_path is None:
                    reference_path, prompt_text, use_prompt = sample_path, part, True
        audio = np.concatenate(waves)
        output = store.renders / f"{job_id}.wav"
        sf.write(output, audio, sample_rate, subtype="PCM_16")
        render = {
            "id": job_id, "kind": request.kind, "text": request.text.strip(),
            "description": request.description.strip(), "voice_id": request.voice_id,
            "voice_name": voice["name"] if voice else "Giọng vừa tạo", "seed": actual_seeds[0],
            "seeds": actual_seeds, "requested_seed": requested_seed,
            "reference_id": request.reference_id, "voice_sample_text": parts[0],
            "generation": {
                "cfg_value": request.cfg_value, "dit_steps": settings["dit_steps"],
                "do_normalize": request.do_normalize, "denoise": request.denoise,
                "random_seed": request.random_seed, "use_prompt_text": settings["use_prompt_text"],
                "prompt_text": prompt_text if settings["use_prompt_text"] else "",
                "mode": "ultimate" if settings["use_prompt_text"] else "clone" if voice or request.reference_id else "design",
                "read_mode": request.read_mode if voice else None,
            },
            "duration": round(len(audio) / sample_rate, 2), "sample_rate": sample_rate, "cache_hit": False,
            "processing_seconds": round(time.perf_counter() - started, 2),
            "real_time_factor": round((time.perf_counter() - started) / (len(audio) / sample_rate), 2),
            "created_at": now(), "audio_url": f"/api/renders/{job_id}/audio",
            "download_url": f"/api/renders/{job_id}/download",
        }
        with job_lock:
            store.write_json(store.renders / f"{job_id}.json", render)
            if cache_key is not None:
                read_cache.put(cache_key, job_id, store.renders)
            jobs[job_id].update(status="completed", message="Âm thanh đã sẵn sàng.", result=render)
    except Exception as error:
        log.exception("Voice generation failed: %s", job_id)
        message = str(error)
        if isinstance(error, (ConnectionError, OSError)) or "Could not fetch" in message or "Connection" in message:
            message = "Chưa kết nối được VoxCPM. Hãy chạy start-voice-studio.cmd rồi thử lại."
        elif "out of memory" in message.lower():
            message = "GPU đang thiếu bộ nhớ. Hãy đóng ứng dụng dùng GPU và thử một đoạn ngắn hơn."
        elif isinstance(error, HTTPException):
            message = error.detail
        update_job(job_id, status="failed", message=message[:500])


@app.get("/api/health")
def health():
    with job_lock:
        active = sum(j["status"] in ("queued", "running") for j in jobs.values())
    return {"ok": True, "name": "Giọng Nhà", "active_jobs": active}


@app.get("/api/voices")
def voices():
    return store.list_voices()


@app.get("/api/history")
def history():
    return store.history()


@app.delete("/api/history")
def clear_history():
    deleted, failed = [], []
    with job_lock:
        if any(job["status"] in ("queued", "running") and job.get("kind") == "speak" for job in jobs.values()):
            raise HTTPException(409, "Đang tạo bản đọc mới. Hãy đợi đọc xong rồi xóa toàn bộ lịch sử.")
        # The list shown on screen is limited; clearing includes older readings.
        for render in store.history(limit=None):
            render_id = render["id"]
            try:
                store.delete_reading(render_id)
            except HTTPException as error:
                failed.append({"id": render_id, "message": error.detail})
                continue
            jobs.pop(render_id, None)
            deleted.append(render_id)
    return {"deleted": deleted, "count": len(deleted), "failed": failed}


@app.delete("/api/history/{render_id}")
def delete_reading(render_id: str):
    store.valid_id(render_id)
    with job_lock:
        if jobs.get(render_id, {}).get("status") in ("queued", "running"):
            raise HTTPException(409, "Bản đọc này đang được tạo. Hãy đợi đọc xong rồi xóa.")
        store.delete_reading(render_id)
        # A completed in-memory job must not restore deleted history after reload.
        jobs.pop(render_id, None)
    return {"deleted": [render_id], "count": 1, "failed": []}


@app.post("/api/jobs", status_code=202)
def create_job(request: GenerateRequest):
    request.text = request.text.strip()
    request.description = request.description.strip()
    request.prompt_text = request.prompt_text.strip()
    if not request.text:
        raise HTTPException(400, "Bạn chưa nhập nội dung cần đọc.")
    if request.kind == "design":
        if request.reference_id:
            store.get_reference(request.reference_id)
        if request.use_prompt_text and not request.reference_id:
            raise HTTPException(400, "Chế độ sao chép chi tiết cần một âm thanh mẫu.")
        if request.use_prompt_text and not request.prompt_text:
            raise HTTPException(400, "Hãy nhập hoặc nhận dạng lời của âm thanh mẫu trước khi tạo giọng.")
    elif not request.voice_id:
        raise HTTPException(400, "Hãy chọn một giọng trong thư viện.")
    with job_lock:
        if request.kind == "speak":
            store.get_voice(request.voice_id)
        pending = sum(j["status"] in ("queued", "running") for j in jobs.values())
        if pending >= 5:
            raise HTTPException(429, "Đã có nhiều yêu cầu đang chờ. Hãy đợi một chút.")
        job_id = uuid.uuid4().hex
        job = {"id": job_id, "kind": request.kind, "status": "queued", "message": "Đang chờ đến lượt…", "current": 0, "total": 0, "created_at": now(), "voice_id": request.voice_id if request.kind == "speak" else None}
        jobs[job_id] = job
    executor.submit(generate, job_id, request)
    return dict(job)


@app.post("/api/references", status_code=201)
async def upload_reference(file: UploadFile = File(...)):
    suffix = Path(file.filename or "sample.webm").suffix.lower()
    if suffix not in {".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".oga", ".webm", ".flac", ".aac", ".opus", ".wma", ".aiff", ".aif"}:
        raise HTTPException(400, "Hãy chọn file âm thanh WAV, MP3, M4A, OGG, FLAC hoặc WebM.")
    reference_id = uuid.uuid4().hex
    source = store.references / f"{reference_id}.upload{suffix}"
    output = store.references / f"{reference_id}.wav"
    try:
        total = 0
        with source.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > 30 * 1024 * 1024:
                    raise HTTPException(413, "Âm thanh mẫu tối đa 30 MB.")
                destination.write(chunk)
        if not total:
            raise HTTPException(400, "File âm thanh đang trống.")
        # Conversion happens off the event loop so uploads do not block job polling.
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(convert_reference, source, output, reference_id, file.filename)
    finally:
        source.unlink(missing_ok=True)
        await file.close()


def convert_reference(source: Path, output: Path, reference_id: str, filename: str | None) -> dict:
    ffmpeg = shutil.which("ffmpeg") or next((str(p) for p in (ROOT / "tools").glob("*/bin/ffmpeg.exe")), None)
    if not ffmpeg:
        raise HTTPException(503, "Chưa tìm thấy FFmpeg để xử lý âm thanh mẫu.")
    try:
        conversion = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
             "-protocol_whitelist", "file,pipe", "-i", str(source), "-map", "0:a:0",
             "-t", "61", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(output)],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=45,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        if conversion.returncode:
            raise ValueError("invalid audio")
        info = sf.info(output)
        if not 0.5 <= info.duration <= 60:
            raise HTTPException(400, "Hãy dùng đoạn giọng mẫu dài từ 0,5 đến 60 giây; khoảng 5–15 giây thường dễ sử dụng.")
        reference = {"id": reference_id, "name": Path(filename or "Bản thu âm").name,
                     "duration": round(info.duration, 2), "sample_rate": info.samplerate,
                     "transcript": "", "created_at": now(), "audio_url": f"/api/references/{reference_id}/audio"}
        store.write_json(output.with_suffix(".json"), reference)
        return reference
    except HTTPException:
        output.unlink(missing_ok=True)
        raise
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired):
        output.unlink(missing_ok=True)
        raise HTTPException(400, "Không đọc được âm thanh. Hãy thử file WAV, MP3 hoặc một bản thu khác.")


@app.get("/api/references/{reference_id}")
def get_reference(reference_id: str):
    return store.get_reference(reference_id)


@app.get("/api/references/{reference_id}/audio")
def reference_audio(reference_id: str):
    return FileResponse(store.reference_path(reference_id), media_type="audio/wav")


def transcribe_reference(job_id: str, reference_id: str, language: str):
    global asr_model
    try:
        update_job(job_id, status="running", message="Đang nhận dạng lời mẫu trên CPU…")
        if asr_model is None:
            from faster_whisper import WhisperModel
            model_path = Path(os.environ.get("VOICE_STUDIO_ASR_MODEL", ROOT / "models" / "whisper-base"))
            asr_model = WhisperModel(str(model_path), device="cpu", compute_type="int8",
                                     cpu_threads=min(4, os.cpu_count() or 2), num_workers=1, local_files_only=True)
        segments, info = asr_model.transcribe(str(store.reference_path(reference_id)),
            language=None if language == "auto" else language, beam_size=5, condition_on_previous_text=False)
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
        if not transcript:
            raise ValueError("Chưa nhận ra lời nói. Bạn có thể nhập lời mẫu bằng tay hoặc thu một đoạn rõ hơn.")
        reference = store.get_reference(reference_id)
        reference.update(transcript=transcript, language=info.language)
        store.write_json(store.references / f"{reference_id}.json", reference)
        update_job(job_id, status="completed", message="Đã nhận dạng xong. Hãy kiểm tra và sửa lời mẫu nếu cần.",
                   result={"reference_id": reference_id, "transcript": transcript, "language": info.language})
    except Exception as error:
        log.exception("Reference transcription failed")
        message = error.detail if isinstance(error, HTTPException) else str(error)
        if isinstance(error, (ImportError, OSError, RuntimeError)):
            message = "Chưa nạp được bộ nhận dạng lời mẫu. Bạn vẫn có thể nhập lời mẫu bằng tay."
        update_job(job_id, status="failed", message=message[:500])


@app.post("/api/references/{reference_id}/transcribe", status_code=202)
def start_transcription(reference_id: str, request: TranscribeRequest):
    store.get_reference(reference_id)
    with job_lock:
        active = sum(j.get("type") == "transcribe" and j["status"] in ("queued", "running") for j in jobs.values())
        if active >= 3:
            raise HTTPException(429, "Đang nhận dạng nhiều mẫu. Hãy đợi một chút rồi thử lại.")
        job_id = uuid.uuid4().hex
        job = {"id": job_id, "type": "transcribe", "status": "queued", "message": "Đang chờ nhận dạng…"}
        jobs[job_id] = job
    asr_executor.submit(transcribe_reference, job_id, reference_id, request.language)
    return dict(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    store.valid_id(job_id)
    with job_lock:
        if job_id in jobs:
            return dict(jobs[job_id])
    render = store.get_render(job_id)
    return {"id": job_id, "status": "completed", "result": render, "message": "Âm thanh đã sẵn sàng."}


@app.post("/api/voices", status_code=201)
def save_voice(request: SaveRequest):
    with save_lock:
        return store.save_voice(request.render_id, request.name)


@app.get("/api/voices/{voice_id}/audio")
def voice_audio(voice_id: str):
    return FileResponse(store.audio_path(voice_id), media_type="audio/wav")


@app.delete("/api/voices/{voice_id}")
def delete_voice(voice_id: str):
    with save_lock, job_lock:
        store.valid_id(voice_id)
        if any(job["status"] in ("queued", "running") and job.get("voice_id") == voice_id for job in jobs.values()):
            raise HTTPException(409, "Giọng này đang được dùng để đọc văn bản. Hãy đợi đọc xong rồi xóa.")
        store.delete_voice(voice_id)
    return {"deleted": voice_id}


@app.get("/api/renders/{render_id}/audio")
def render_audio(render_id: str):
    store.get_render(render_id)
    return FileResponse(store.renders / f"{render_id}.wav", media_type="audio/wav")


@app.get("/api/renders/{render_id}/download")
def download(render_id: str):
    store.get_render(render_id)
    return FileResponse(store.renders / f"{render_id}.wav", media_type="audio/wav", filename=f"giong-nha-{render_id[:8]}.wav")


app.mount("/", StaticFiles(directory=Path(__file__).parent / "web", html=True), name="web")
