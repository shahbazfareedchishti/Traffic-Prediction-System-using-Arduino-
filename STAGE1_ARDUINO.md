# Stage-1 Arduino Integration (ESP32-CAM)

This setup removes manual handoff between capture and Stage-1.

Flow:
1. ESP32-CAM records a session as timed JPEG frames.
2. Frames are uploaded to `stage1_receiver.py`.
3. `stage1_process_session.py` runs YOLO on that session and creates LSTM-ready CSV.
4. `pipeline.py` can be triggered automatically to update/retrain LSTM artifacts.

## Files added

- Arduino sketch: `arduino/esp32_cam_stage1_sender/esp32_cam_stage1_sender.ino`
- Stage-1 receiver API: `stage1_receiver.py`
- Stage-1 YOLO processor: `stage1_process_session.py`

## 1) Run Stage-1 receiver

```bash
cd /home/shahbazfareed/Downloads/Real-Time Traffic Flow Prediction
pip install -r requirements.txt
python stage1_receiver.py --host 0.0.0.0 --port 7000
```

Health check:

```bash
curl http://127.0.0.1:7000/stage1/health
```

Uploaded sessions will be stored in:
- `stage1_inbox/<session_id>/frames/*.jpg`
- `stage1_inbox/<session_id>/frames.csv`

## 2) Flash Arduino sketch

Open:
- `arduino/esp32_cam_stage1_sender/esp32_cam_stage1_sender.ino`

Set these values:
- `WIFI_SSID`
- `WIFI_PASSWORD`
- `STAGE1_BASE_URL` (IP of machine running `stage1_receiver.py`)
- `DEVICE_ID`
- `LOCATION_NAME`

Optional timing:
- `FRAME_INTERVAL_MS`
- `RECORD_DURATION_MS`

## 3) Process one session with YOLO

Default model file used automatically: `UVH-26-MV-YOLOv11-X.pt`

Install YOLO dependency (if not installed):

```bash
pip install -r requirements-stage1.txt
```

Process session:

```bash
python stage1_process_session.py \
  --session-id <SESSION_ID>
```

Outputs:
- `stage1_output/<SESSION_ID>_frame_level.csv`
- `stage1_output/<SESSION_ID>_lstm_ready.csv`

## 4) Auto-run pipeline after YOLO processing

```bash
python stage1_process_session.py \
  --session-id <SESSION_ID> \
  --auto-pipeline
```

This calls `pipeline.py` internally and updates:
- `historical_traffic_hourly.csv`
- `best_lstm_traffic_model.pth`
- `scaler.pkl`

## 5) Fully automatic background mode (recommended for demo)

Run a worker that auto-processes all closed sessions (`_READY` markers):

```bash
python stage1_worker.py \
  --auto-pipeline
```

Then Arduino upload + close is enough; no manual command per session.

## Notes

- ESP32-CAM cannot reliably encode full MP4 on-device; this design streams frame sequence to represent video sessions.
- You can still produce a real video from frames later if needed for demo/reporting.
