"""Create the initial AI voice collection once, using the running local tool."""
import json
import time
from urllib.request import Request, urlopen

from voice_studio.app import store

URL = "http://127.0.0.1:8809"
SAMPLE = (
    "Xin chào, rất vui được gặp bạn. Mỗi ngày là một cơ hội để bắt đầu một câu chuyện mới. "
    "Hãy cùng tôi chậm lại một chút, lắng nghe và tận hưởng những điều giản dị quanh mình."
)
PRESETS = [
    ("Nữ dịu dàng", "Giọng nữ trẻ, dịu dàng, ấm áp, nói tiếng Việt rõ ràng, nhịp đọc nhẹ nhàng.", 132),
    ("Nam trầm ấm", "Giọng nam trưởng thành, trầm ấm, điềm tĩnh, nói tiếng Việt rõ ràng, phong cách kể chuyện.", 267),
    ("Nữ tươi sáng", "Giọng nữ trẻ, tươi sáng, vui vẻ và thân thiện, nói tiếng Việt rõ ràng, nhịp đọc tự nhiên.", 518),
]


def api(path, data=None):
    payload = json.dumps(data).encode("utf-8") if data is not None else None
    request = Request(URL + path, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=15) as response:
        return json.load(response)


if __name__ == "__main__":
    existing = {voice["name"] for voice in store.list_voices()}
    for name, description, seed in PRESETS:
        if name in existing:
            print(f"Already saved: {name}", flush=True)
            continue
        print(f"Creating: {name}", flush=True)
        job = api("/api/jobs", {"kind": "design", "text": SAMPLE, "description": description, "seed": seed})
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            current = api(f"/api/jobs/{job['id']}")
            if current["status"] == "failed":
                raise RuntimeError(current["message"])
            if current["status"] == "completed":
                voice = store.save_voice(job["id"], name, origin="preset")
                print(f"Saved: {voice['name']} ({voice['duration']} seconds)", flush=True)
                break
            time.sleep(1)
        else:
            raise TimeoutError(f"Timed out creating {name}")
