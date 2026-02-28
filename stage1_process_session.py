#!/usr/bin/env python3
"""
Process one uploaded Stage-1 session:
1) Run YOLO on saved frame images.
2) Build minute-level CSV: timestamp, vehicle_count, location.
3) Optionally call pipeline.py to update/retrain LSTM.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_YOLO_MODEL = BASE_DIR / "UVH-26-MV-YOLOv11-X.pt"


def parse_class_ids(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def resolve_session_paths(inbox_dir: Path, session_id: str) -> tuple[Path, Path, Path]:
    session_dir = inbox_dir / session_id
    frames_dir = session_dir / "frames"
    metadata_csv = session_dir / "frames.csv"
    return session_dir, frames_dir, metadata_csv


def load_session_metadata(metadata_csv: Path) -> pd.DataFrame:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Session metadata not found: {metadata_csv}")

    df = pd.read_csv(metadata_csv)
    required = {"captured_at_utc", "frame_file"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {sorted(missing)}")

    df["captured_at_utc"] = pd.to_datetime(df["captured_at_utc"], errors="coerce")
    df = df.dropna(subset=["captured_at_utc", "frame_file"]).copy()
    if df.empty:
        raise ValueError("Session metadata has no valid rows.")

    if "location" not in df.columns:
        df["location"] = "unknown"

    return df.sort_values("captured_at_utc").reset_index(drop=True)


def run_yolo_on_frames(
    metadata_df: pd.DataFrame,
    frames_dir: Path,
    model_path: Path,
    confidence: float,
    vehicle_class_ids: list[int],
    frame_step: int,
) -> pd.DataFrame:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for Stage-1 processing. Install with: pip install ultralytics"
        ) from exc

    model = YOLO(str(model_path))
    rows = []

    for idx, row in metadata_df.iterrows():
        if frame_step > 1 and (idx % frame_step != 0):
            continue

        frame_path = frames_dir / str(row["frame_file"])
        if not frame_path.exists():
            continue

        result = model(str(frame_path), conf=confidence, verbose=False)[0]
        vehicle_count = 0
        if result.boxes is not None and result.boxes.cls is not None:
            classes = result.boxes.cls.detach().cpu().numpy().astype(int)
            if vehicle_class_ids:
                vehicle_count = int(np.isin(classes, vehicle_class_ids).sum())
            else:
                vehicle_count = int(len(classes))

        rows.append(
            {
                "timestamp": row["captured_at_utc"],
                "location": row.get("location", "unknown"),
                "vehicle_count": vehicle_count,
            }
        )

    if not rows:
        raise ValueError("No YOLO detections generated. Check frames/model/confidence.")

    return pd.DataFrame(rows)


def aggregate_to_minute(yolo_frame_df: pd.DataFrame) -> pd.DataFrame:
    yolo_frame_df = yolo_frame_df.copy()
    yolo_frame_df["minute"] = pd.to_datetime(yolo_frame_df["timestamp"]).dt.floor("min")

    def choose_mode(values: pd.Series) -> str:
        mode = values.mode(dropna=True)
        if mode.empty:
            return "unknown"
        return str(mode.iloc[0])

    grouped = (
        yolo_frame_df.groupby("minute", as_index=False)
        .agg(
            vehicle_count=("vehicle_count", "mean"),
            location=("location", choose_mode),
        )
        .sort_values("minute")
    )

    grouped["vehicle_count"] = grouped["vehicle_count"].clip(lower=0).round().astype(int)
    grouped["timestamp"] = grouped["minute"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return grouped[["timestamp", "vehicle_count", "location"]]


def run_pipeline_if_requested(
    enabled: bool,
    python_bin: str,
    pipeline_script: Path,
    lstm_ready_csv: Path,
    dataset: Path,
    extra_args: list[str],
) -> None:
    if not enabled:
        return

    cmd = [
        python_bin,
        str(pipeline_script),
        "--yolo-csv",
        str(lstm_ready_csv),
        "--dataset",
        str(dataset),
    ] + extra_args
    print("Running pipeline:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO on one Stage-1 session.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--model-path", default=str(DEFAULT_YOLO_MODEL), help="Path to YOLO weights (.pt)")
    parser.add_argument("--inbox-dir", default="stage1_inbox")
    parser.add_argument("--output-dir", default="stage1_output")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--vehicle-class-ids", default="2,3,5,7")
    parser.add_argument("--frame-step", type=int, default=1, help="Use every Nth frame.")
    parser.add_argument("--auto-pipeline", action="store_true")
    parser.add_argument("--pipeline-script", default=str(BASE_DIR / "pipeline.py"))
    parser.add_argument("--dataset", default="historical_traffic_hourly.csv")
    parser.add_argument(
        "--pipeline-extra-args",
        nargs="*",
        default=[],
        help="Optional passthrough args for pipeline.py, e.g. --skip-training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    session_id = args.session_id
    inbox_dir = Path(args.inbox_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_path = Path(args.model_path).resolve()
    pipeline_script = Path(args.pipeline_script).resolve()
    dataset = Path(args.dataset).resolve()
    class_ids = parse_class_ids(args.vehicle_class_ids)

    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    session_dir, frames_dir, metadata_csv = resolve_session_paths(inbox_dir=inbox_dir, session_id=session_id)
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_df = load_session_metadata(metadata_csv)
    yolo_frame_df = run_yolo_on_frames(
        metadata_df=metadata_df,
        frames_dir=frames_dir,
        model_path=model_path,
        confidence=args.confidence,
        vehicle_class_ids=class_ids,
        frame_step=max(1, args.frame_step),
    )
    minute_df = aggregate_to_minute(yolo_frame_df)

    frame_level_csv = output_dir / f"{session_id}_frame_level.csv"
    lstm_ready_csv = output_dir / f"{session_id}_lstm_ready.csv"
    yolo_frame_df.assign(timestamp=lambda d: d["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")).to_csv(
        frame_level_csv, index=False
    )
    minute_df.to_csv(lstm_ready_csv, index=False)

    print(f"Session processed: {session_id}")
    print(f"Frame-level detections: {frame_level_csv}")
    print(f"LSTM-ready CSV: {lstm_ready_csv}")

    run_pipeline_if_requested(
        enabled=args.auto_pipeline,
        python_bin=sys.executable,
        pipeline_script=pipeline_script,
        lstm_ready_csv=lstm_ready_csv,
        dataset=dataset,
        extra_args=args.pipeline_extra_args,
    )


if __name__ == "__main__":
    main()
