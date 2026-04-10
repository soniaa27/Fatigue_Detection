"""Flask backend for the fatigue detection dashboard."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread

from flask import Flask, Response, jsonify, request, send_file
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_FILE = BASE_DIR / "frontend.html"
ALERT_FILE = BASE_DIR / "alert.py"
FRAME_FILE = Path(tempfile.gettempdir()) / "fatigue_latest_frame.jpg"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)

_process: subprocess.Popen | None = None
_started_at: datetime | None = None
_log_lines: deque[str] = deque(maxlen=300)
_state_lock = Lock()
_metrics: dict = {}
_uploaded_video: Path | None = None
_uploaded_video_name: str | None = None

_BOUNDARY = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
_OFFLINE_JPEG: bytes | None = None


def _parse_metrics_from_line(line: str) -> None:
    import re

    updates: dict = {}
    low = line.lower()

    for key, pat in [
        ("ear", r"ear=(-?\d+\.\d+)"),
        ("perclos", r"perclos=(-?\d+\.\d+)"),
        ("gru_prob", r"gru=(-?\d+\.\d+)"),
        ("deviation", r"dev=(-?\d+\.\d+)"),
        ("sustained", r"sustained=(\d+)/(\d+)"),
    ]:
        m = re.search(pat, low)
        if not m:
            continue
        if key == "sustained":
            updates["sustained"] = int(m.group(1))
            updates["sustained_max"] = int(m.group(2))
        else:
            updates[key] = float(m.group(1))

    if "fatigue alert" in low and "⚠" in line:
        updates["alert_active"] = True
    elif "sustained=" in low:
        current_sustained = updates.get("sustained", _metrics.get("sustained", 0))
        sustained_max = updates.get("sustained_max", _metrics.get("sustained_max", 8))
        if current_sustained < sustained_max // 2:
            updates["alert_active"] = False

    if updates:
        with _state_lock:
            _metrics.update(updates)


def _append_log(line: str) -> None:
    line = line.rstrip("\n")
    if not line:
        return
    with _state_lock:
        _log_lines.append(line)
    _parse_metrics_from_line(line)


def _stream_reader(proc: subprocess.Popen) -> None:
    if proc.stdout is None:
        return
    for raw in proc.stdout:
        _append_log(raw)


def _start_backend() -> dict:
    global _process, _started_at, _metrics

    with _state_lock:
        if _process is not None and _process.poll() is None:
            return {"ok": True, "message": "Already running"}

        if _uploaded_video is None or not _uploaded_video.exists():
            return {"ok": False, "message": "Upload a video first"}

        FRAME_FILE.unlink(missing_ok=True)

        cmd = [
            sys.executable,
            "-u",
            "-X",
            "utf8",
            str(ALERT_FILE),
            "--headless",
            "--web-frame-path",
            str(FRAME_FILE),
            "--input-video",
            str(_uploaded_video),
        ]
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        _process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )
        _started_at = datetime.now()
        _metrics = {}
        _log_lines.clear()
        _log_lines.append(f"[WEB] Started alert.py (video mode): {_uploaded_video_name or _uploaded_video.name}")

        Thread(target=_stream_reader, args=(_process,), daemon=True).start()

    return {"ok": True, "message": "Backend started"}


def _stop_backend() -> dict:
    global _process

    with _state_lock:
        if _process is None or _process.poll() is not None:
            return {"ok": True, "message": "Already stopped"}
        proc = _process

    try:
        if os.name == "nt":
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()

    with _state_lock:
        _log_lines.append("[WEB] Stopped alert.py")
        _process = None
        _metrics.clear()

    return {"ok": True, "message": "Backend stopped"}


def _is_running() -> bool:
    with _state_lock:
        return _process is not None and _process.poll() is None


def _status_payload() -> dict:
    with _state_lock:
        running = _process is not None and _process.poll() is None
        pid = _process.pid if running and _process else None
        started = _started_at.isoformat(timespec="seconds") if _started_at else None
        log = list(_log_lines)
        metrics = dict(_metrics)
        source = _uploaded_video_name
    return {
        "status": {"running": running, "pid": pid, "started_at": started},
        "log": log,
        "metrics": metrics,
        "source": {"type": "video" if source else None, "name": source},
    }


def _offline_placeholder() -> bytes:
    global _OFFLINE_JPEG
    if _OFFLINE_JPEG is not None:
        return _OFFLINE_JPEG

    try:
        import cv2
        import numpy as np

        img = np.full((180, 320, 3), 15, dtype=np.uint8)
        cv2.putText(img, "CAMERA OFFLINE", (42, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (60, 60, 60), 2)
        _, buf = cv2.imencode(".jpg", img)
        _OFFLINE_JPEG = buf.tobytes()
    except Exception:
        _OFFLINE_JPEG = b""
    return _OFFLINE_JPEG


def _mjpeg_generator():
    while True:
        if _is_running() and FRAME_FILE.exists():
            try:
                data = FRAME_FILE.read_bytes()
                if data:
                    yield _BOUNDARY + data + b"\r\n"
                    time.sleep(1 / 25)
                    continue
            except OSError:
                pass
        placeholder = _offline_placeholder()
        if placeholder:
            yield _BOUNDARY + placeholder + b"\r\n"
        time.sleep(0.2)


@app.get("/")
def index():
    return send_file(FRONTEND_FILE)


@app.get("/video_feed")
def video_feed():
    return Response(_mjpeg_generator(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/status")
def api_status():
    resp = jsonify(_status_payload())
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.post("/api/upload")
def api_upload():
    global _uploaded_video, _uploaded_video_name

    file = request.files.get("video")

    if file is None or not file.filename:
        return jsonify({"ok": False, "message": "No video file provided"}), 400

    if _is_running():
        return jsonify({"ok": False, "message": "Stop the current analysis before uploading a new video"}), 409

    safe_name = secure_filename(file.filename)
    if not safe_name:
        safe_name = f"video_{int(time.time())}.mp4"

    target = UPLOAD_DIR / f"{int(time.time())}_{safe_name}"
    file.save(target)

    with _state_lock:
        _uploaded_video = target
        _uploaded_video_name = file.filename
        _metrics.clear()
        _log_lines.clear()
        _log_lines.append(f"[WEB] Uploaded video: {file.filename}")

    return jsonify({"ok": True, "message": "Video uploaded", "filename": file.filename})


@app.post("/api/start")
def api_start():
    return jsonify(_start_backend())


@app.post("/api/stop")
def api_stop():
    return jsonify(_stop_backend())


if __name__ == "__main__":
    if not FRONTEND_FILE.exists():
        raise FileNotFoundError(f"Missing: {FRONTEND_FILE}")
    if not ALERT_FILE.exists():
        raise FileNotFoundError(f"Missing: {ALERT_FILE}")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
