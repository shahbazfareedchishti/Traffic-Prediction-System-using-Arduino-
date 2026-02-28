# Automated YOLO -> LSTM Pipeline

This project now includes `pipeline.py` to remove manual handoff between:
- Stage 1: YOLO vehicle detection CSV output
- Stage 2: LSTM training artifacts used by the Flask app

## Expected YOLO CSV input

Required:
- `timestamp` (datetime)
- one count column such as `vehicle_count` (auto-detected)

Optional:
- `location` (use `--location` to train for a single location)

## One-shot run

```bash
python pipeline.py \
  --yolo-csv /path/to/lstm_ready_20260226_140000.csv \
  --dataset historical_traffic_hourly.csv \
  --aggregation sum
```

What it does:
1. Loads YOLO CSV
2. Aggregates minute-level counts into hourly `Total`
3. Merges into `historical_traffic_hourly.csv` (deduplicated by timestamp)
4. Runs quality checks (coverage/missing hours)
5. Retrains LSTM
6. Saves:
   - `best_lstm_traffic_model.pth`
   - `scaler.pkl`
   - `pipeline_metadata.json`

## Useful options

```bash
# Filter by location if YOLO CSV has multiple road segments
python pipeline.py --yolo-csv data.csv --location "Main Road"

# Only update dataset, do not retrain yet
python pipeline.py --yolo-csv data.csv --skip-training

# Force specific source count column name
python pipeline.py --yolo-csv data.csv --count-column vehicle_count
```

## Automation with cron (example)

Run every hour and process the newest uploaded YOLO CSV:

```bash
0 * * * * cd /home/shahbazfareed/Downloads/projects && python pipeline.py --yolo-csv /home/shahbazfareed/Downloads/latest_yolo.csv >> /home/shahbazfareed/Downloads/projects/pipeline.log 2>&1
```

## Flask integration

`app.py` now auto-reloads:
- `best_lstm_traffic_model.pth`
- `scaler.pkl`
- `historical_traffic_hourly.csv`

So after pipeline retraining, new predictions are used without manual code edits.

## Stage-1 capture integration

For Arduino/ESP32-CAM upload + Stage-1 receiver + YOLO session processing, see:
- `STAGE1_ARDUINO.md`

Stage-1 YOLO dependency list:
- `requirements-stage1.txt`
