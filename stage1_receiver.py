#!/usr/bin/env python3
"""
Stage-1 ingestion server for Arduino/ESP32-CAM uploads.

Endpoints:
- GET  /stage1/health
- POST /stage1/frame
- POST /stage1/close
- GET  /stage1/sessions/<session_id>

`/stage1/frame` expects raw JPEG bytes in request body and these headers:
- X-Device-Id   (optional)
- X-Location    (optional)
- X-Session-Id  (optional; created automatically when missing)
- X-Timestamp-Ms(optional; sender clock)
"""

import argparse
import csv
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, request


app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INBOX = PROJECT_ROOT / "stage1_inbox"
INBOX_DIR = Path(os.environ.get("STAGE1_INBOX_DIR", DEFAULT_INBOX))
INBOX_DIR.mkdir(parents=True, exist_ok=True)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize(value: str, fallback: str) -> str:
    if value is None:
        return fallback
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return cleaned or fallback


def build_session_id(device_id: str, provided_session_id: str | None) -> str:
    if provided_session_id:
        return sanitize(provided_session_id, "session")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{sanitize(device_id, 'device')}_{stamp}_{uuid.uuid4().hex[:6]}"


def get_session_paths(session_id: str) -> tuple[Path, Path, Path, Path]:
    session_dir = INBOX_DIR / session_id
    frames_dir = session_dir / "frames"
    metadata_csv = session_dir / "frames.csv"
    counter_txt = session_dir / "counter.txt"
    return session_dir, frames_dir, metadata_csv, counter_txt


def next_frame_index(counter_txt: Path) -> int:
    if counter_txt.exists():
        raw = counter_txt.read_text(encoding="utf-8").strip()
        current = int(raw) if raw else 0
    else:
        current = 0

    new_index = current + 1
    counter_txt.write_text(str(new_index), encoding="utf-8")
    return new_index


def append_metadata(
    metadata_csv: Path,
    captured_at_utc: str,
    timestamp_ms: str,
    device_id: str,
    location: str,
    frame_file: str,
    size_bytes: int,
) -> None:
    file_exists = metadata_csv.exists()
    with metadata_csv.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not file_exists:
            writer.writerow(
                [
                    "captured_at_utc",
                    "timestamp_ms",
                    "device_id",
                    "location",
                    "frame_file",
                    "size_bytes",
                ]
            )
        writer.writerow(
            [
                captured_at_utc,
                timestamp_ms,
                device_id,
                location,
                frame_file,
                size_bytes,
            ]
        )


@app.get("/stage1/health")
def health() -> tuple[dict, int]:
    return {"status": "ok", "inbox_dir": str(INBOX_DIR)}, 200


@app.post("/stage1/frame")
def upload_frame():
    raw = request.get_data(cache=False)
    if not raw:
        return jsonify({"error": "Empty frame payload"}), 400

    device_id = sanitize(request.headers.get("X-Device-Id"), "arduino-device")
    location = sanitize(request.headers.get("X-Location"), "unknown")
    session_header = request.headers.get("X-Session-Id")
    timestamp_ms = request.headers.get("X-Timestamp-Ms", "")
    captured_at_utc = utc_now_iso()

    session_id = build_session_id(device_id=device_id, provided_session_id=session_header)
    session_dir, frames_dir, metadata_csv, counter_txt = get_session_paths(session_id)
    frames_dir.mkdir(parents=True, exist_ok=True)

    frame_index = next_frame_index(counter_txt)
    frame_name = f"frame_{frame_index:06d}.jpg"
    frame_path = frames_dir / frame_name
    frame_path.write_bytes(raw)

    append_metadata(
        metadata_csv=metadata_csv,
        captured_at_utc=captured_at_utc,
        timestamp_ms=timestamp_ms,
        device_id=device_id,
        location=location,
        frame_file=frame_name,
        size_bytes=len(raw),
    )

    # Keep response lightweight for microcontroller parsing.
    response_text = f"session_id={session_id};frame_index={frame_index}"
    return response_text, 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.post("/stage1/close")
def close_session():
    payload = request.get_json(silent=True) or {}
    session_id = (
        payload.get("session_id")
        or request.form.get("session_id")
        or request.args.get("session_id")
        or request.headers.get("X-Session-Id")
    )
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400

    session_id = sanitize(session_id, "session")
    session_dir, frames_dir, metadata_csv, _ = get_session_paths(session_id)
    if not session_dir.exists():
        return jsonify({"error": "session not found", "session_id": session_id}), 404

    frame_count = len(list(frames_dir.glob("frame_*.jpg"))) if frames_dir.exists() else 0
    closed_manifest = {
        "session_id": session_id,
        "closed_at_utc": utc_now_iso(),
        "frame_count": frame_count,
        "metadata_csv": str(metadata_csv),
        "status": "ready_for_stage1",
    }
    (session_dir / "session_closed.json").write_text(
        json.dumps(closed_manifest, indent=2),
        encoding="utf-8",
    )
    (session_dir / "_READY").write_text("", encoding="utf-8")

    return jsonify(closed_manifest), 200


@app.get("/stage1/sessions/<session_id>")
def session_info(session_id: str):
    session_id = sanitize(session_id, "session")
    session_dir, frames_dir, metadata_csv, _ = get_session_paths(session_id)
    if not session_dir.exists():
        return jsonify({"error": "session not found", "session_id": session_id}), 404

    frame_count = len(list(frames_dir.glob("frame_*.jpg"))) if frames_dir.exists() else 0
    info = {
        "session_id": session_id,
        "session_dir": str(session_dir),
        "metadata_csv": str(metadata_csv),
        "frame_count": frame_count,
        "ready": (session_dir / "_READY").exists(),
    }
    return jsonify(info), 200


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-1 frame receiver server.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=7000, type=int)
    parser.add_argument(
        "--inbox-dir",
        default=str(INBOX_DIR),
        help="Where received sessions are stored.",
    )
    return parser.parse_args()


def main() -> None:
    global INBOX_DIR
    args = parse_args()
    INBOX_DIR = Path(args.inbox_dir).resolve()
    INBOX_DIR.mkdir(parents=True, exist_ok=True)
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
