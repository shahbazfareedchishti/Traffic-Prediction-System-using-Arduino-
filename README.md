# Real-Time Traffic Flow Prediction

Real-Time Traffic Flow Prediction is an end-to-end traffic analytics project that combines:

- Stage-1 vehicle detection from camera frames (YOLO)
- Automated data processing and hourly aggregation
- LSTM-based traffic forecasting
- A Flask web app to visualize predicted hourly traffic and peak periods

This repository was developed as a Final Year Project (FYP) and is structured so it can run as a demo or be extended for production-style workflows.

## Project Highlights

- End-to-end pipeline from camera capture to model retraining
- Stage-1 ingestion API for ESP32-CAM frame uploads
- Session-based YOLO processing with CSV outputs
- Automatic merge/update of historical traffic dataset
- LSTM retraining with early stopping and artifact export
- Flask UI for day-level hourly predictions with charts and peak-hour detection
- Artifact hot-reload in Flask app (new model/scaler are picked up automatically)

## Model Attribution

- This project uses a **pretrained YOLO model from Hugging Face**.
- Model name used in this project: **UVH-26**
- Local weights file used by the scripts: `UVH-26-MV-YOLOv11-X.pt`

If you publish this project, add your exact Hugging Face model URL in this section for complete attribution.

## Repository Structure

```text
.
├── app.py                              # Flask prediction app
├── templates/index.html                # UI template
├── pipeline.py                         # YOLO CSV -> hourly dataset -> LSTM retraining
├── historical_traffic_hourly.csv       # Main hourly dataset
├── best_lstm_traffic_model.pth         # Trained LSTM weights
├── scaler.pkl                          # MinMax scaler for model features
├── stage1_receiver.py                  # Ingestion server for ESP32-CAM frames
├── stage1_process_session.py           # Runs YOLO on one uploaded session
├── stage1_worker.py                    # Background auto-processor for ready sessions
├── arduino/esp32_cam_stage1_sender/
│   └── esp32_cam_stage1_sender.ino     # ESP32-CAM sender sketch
├── UVH-26-MV-YOLOv11-X.pt              # YOLO weights file (UVH-26)
├── PIPELINE.md                         # Pipeline usage notes
├── STAGE1_ARDUINO.md                   # Stage-1 + Arduino workflow notes
├── requirements.txt
└── Containerfile
```

## System Flow

1. ESP32-CAM captures frames and sends them to `stage1_receiver.py`.
2. Uploaded frames are stored in `stage1_inbox/<session_id>/frames`.
3. A session is closed (`/stage1/close`) and marked ready.
4. `stage1_process_session.py` runs YOLO on frames and creates:
   - frame-level detections
   - LSTM-ready minute CSV
5. `pipeline.py` aggregates minute detections to hourly totals, updates history, and retrains the LSTM model.
6. `app.py` uses latest artifacts to generate hourly predictions and peak-hour insights.

## Tech Stack

- Python 3.12
- Flask
- PyTorch
- scikit-learn
- pandas / numpy
- matplotlib
- ultralytics (YOLO runtime)
- ESP32-CAM (Arduino sketch for frame capture)

## Installation

### 1) Clone the repository

```bash
git clone <your-repo-url>
cd "Real-Time Traffic Flow Prediction"
```

### 2) Create virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Flask Prediction App

```bash
python app.py
```

Open: `http://127.0.0.1:5000`

What the UI does:

- Accepts a start date/time
- Forecasts hourly traffic for the selected day
- Shows:
  - prediction chart
  - prediction table
  - peak hours above dynamic threshold

## Stage-1 Ingestion API (ESP32-CAM Uploads)

Start receiver:

```bash
python stage1_receiver.py --host 0.0.0.0 --port 7000
```

Health check:

```bash
curl http://127.0.0.1:7000/stage1/health
```

### Endpoints

- `GET /stage1/health`
- `POST /stage1/frame`
- `POST /stage1/close`
- `GET /stage1/sessions/<session_id>`

Expected headers for `/stage1/frame`:

- `X-Device-Id` (optional)
- `X-Location` (optional)
- `X-Session-Id` (optional)
- `X-Timestamp-Ms` (optional)

## Process One Uploaded Session (YOLO)

```bash
python stage1_process_session.py \
  --session-id <SESSION_ID>
```

Outputs:

- `stage1_output/<SESSION_ID>_frame_level.csv`
- `stage1_output/<SESSION_ID>_lstm_ready.csv`

Run YOLO + auto-update dataset/model in one command:

```bash
python stage1_process_session.py \
  --session-id <SESSION_ID> \
  --auto-pipeline
```

## Background Worker (Fully Automated Demo Flow)

```bash
python stage1_worker.py --auto-pipeline
```

The worker monitors `stage1_inbox/*/_READY` and processes sessions automatically.

## Pipeline: YOLO CSV to LSTM Artifacts

Run directly:

```bash
python pipeline.py \
  --yolo-csv /path/to/lstm_ready.csv \
  --dataset historical_traffic_hourly.csv \
  --aggregation sum
```

Pipeline operations:

1. Load YOLO CSV (`timestamp` + count column)
2. Aggregate to hourly rows
3. Merge into historical dataset
4. Run quality checks (coverage/missing hours)
5. Retrain LSTM (unless skipped)
6. Save:
   - `best_lstm_traffic_model.pth`
   - `scaler.pkl`
   - metadata JSON

Useful flags:

- `--count-column <name>`
- `--location <location_name>`
- `--skip-training`
- `--look-back`, `--epochs`, `--batch-size`, `--learning-rate`

## Arduino (ESP32-CAM) Setup

Open:

- `arduino/esp32_cam_stage1_sender/esp32_cam_stage1_sender.ino`

Set values:

- `WIFI_SSID`
- `WIFI_PASSWORD`
- `STAGE1_BASE_URL` (machine IP running `stage1_receiver.py`)
- `DEVICE_ID`
- `LOCATION_NAME`

Optional timing parameters:

- `FRAME_INTERVAL_MS`
- `RECORD_DURATION_MS`
- `PAUSE_BETWEEN_SESSIONS_MS`

## Data Formats

### YOLO Input CSV expected by pipeline

Required:

- `timestamp`
- one count column (auto-detected from `vehicle_count`, `count`, `Total`, `total`, `vehicles`)

Optional:

- `location` (used with `--location`)

### Historical Dataset Schema

`historical_traffic_hourly.csv` contains:

- `timestamp`, `Date`, `CarCount`, `BikeCount`, `BusCount`, `TruckCount`, `Total`, `hour`, `day_of_week`

## Docker

Build:

```bash
docker build -t traffic-flow-app -f Containerfile .
```

Run:

```bash
docker run --rm -p 5000:5000 traffic-flow-app
```

## Troubleshooting

- `ultralytics` import error:
  - Install dependencies again with `pip install -r requirements.txt`
- No model updates seen in Flask:
  - Ensure `best_lstm_traffic_model.pth`, `scaler.pkl`, and dataset timestamps changed
  - Submit prediction again; app reloads artifacts at prediction time
- Empty YOLO processing output:
  - Check session frames exist under `stage1_inbox/<session_id>/frames`
  - Verify `UVH-26-MV-YOLOv11-X.pt` path and confidence threshold
- Training skipped:
  - Pipeline skips training if dataset rows are below `--min-hours-for-training`

## Notes

- ESP32-CAM does not reliably encode full MP4 on-device; this design uploads frame sequences.
- You can convert stored frames into a video later for reporting/demo purposes.

## License

Add your preferred license file (`LICENSE`) before public release.

# Traffic-Prediction-System-using-Arduino-
