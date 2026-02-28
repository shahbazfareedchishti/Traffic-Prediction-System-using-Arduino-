#!/usr/bin/env python3
"""
Automated bridge from YOLO CSV output to LSTM artifacts.

Pipeline:
1) Read minute-level YOLO CSV (timestamp + vehicle_count [+ optional location]).
2) Aggregate into hourly rows required by the LSTM pipeline.
3) Merge/update historical dataset.
4) Run data quality checks.
5) Retrain LSTM and save best model + scaler.
"""

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


FEATURES = ["Total", "hour", "day_of_week"]
DATASET_COLUMNS = [
    "timestamp",
    "Date",
    "CarCount",
    "BikeCount",
    "BusCount",
    "TruckCount",
    "Total",
    "hour",
    "day_of_week",
]


class LSTMModel(nn.Module):
    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 50,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_count_column(df: pd.DataFrame) -> str:
    candidates = ["vehicle_count", "count", "Total", "total", "vehicles"]
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError(
        "Could not infer vehicle count column. Provide --count-column. "
        "Expected one of: vehicle_count, count, Total, total, vehicles."
    )


def load_yolo_csv(path: Path, count_column: Optional[str], location: Optional[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"YOLO CSV not found: {path}")

    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("YOLO CSV must contain a 'timestamp' column.")

    if count_column is None:
        count_column = infer_count_column(df)
    elif count_column not in df.columns:
        raise ValueError(f"Requested --count-column '{count_column}' not found in CSV.")

    if location:
        if "location" not in df.columns:
            raise ValueError("Location filter requested, but CSV has no 'location' column.")
        df = df[df["location"].astype(str).str.lower() == location.lower()]

    df = df[["timestamp", count_column]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df[count_column] = pd.to_numeric(df[count_column], errors="coerce")
    df = df.dropna(subset=["timestamp", count_column])
    df = df.rename(columns={count_column: "vehicle_count"})
    df = df.sort_values("timestamp").reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows found after parsing/filtering YOLO CSV.")

    return df


def aggregate_hourly(yolo_df: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    indexed = yolo_df.set_index("timestamp").sort_index()

    if aggregation == "sum":
        hourly_total = indexed["vehicle_count"].resample("h").sum(min_count=1)
    else:
        hourly_total = indexed["vehicle_count"].resample("h").mean()

    hourly = hourly_total.dropna().clip(lower=0).reset_index()
    hourly = hourly.rename(columns={"vehicle_count": "Total"})
    hourly["Total"] = hourly["Total"].round().astype(int)
    hourly["hour"] = hourly["timestamp"].dt.hour.astype(int)
    hourly["day_of_week"] = hourly["timestamp"].dt.weekday.astype(int)
    hourly["Date"] = hourly["timestamp"].dt.day.astype(int)

    # Class-wise columns are unknown from this YOLO output; keep them zeroed.
    hourly["CarCount"] = 0
    hourly["BikeCount"] = 0
    hourly["BusCount"] = 0
    hourly["TruckCount"] = 0

    return hourly[DATASET_COLUMNS]


def ensure_dataset_schema(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    if "timestamp" not in normalized.columns:
        normalized["timestamp"] = pd.NaT

    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce")

    for column in DATASET_COLUMNS:
        if column not in normalized.columns:
            normalized[column] = 0

    numeric_cols = [c for c in DATASET_COLUMNS if c != "timestamp"]
    for col in numeric_cols:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce").fillna(0)

    normalized = normalized.dropna(subset=["timestamp"])
    normalized["Total"] = normalized["Total"].clip(lower=0)
    normalized["hour"] = normalized["timestamp"].dt.hour.astype(int)
    normalized["day_of_week"] = normalized["timestamp"].dt.weekday.astype(int)
    normalized["Date"] = normalized["timestamp"].dt.day.astype(int)

    return normalized[DATASET_COLUMNS].sort_values("timestamp").reset_index(drop=True)


def load_existing_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=DATASET_COLUMNS)
    return ensure_dataset_schema(pd.read_csv(path))


def merge_datasets(existing: pd.DataFrame, incoming: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    combined = pd.concat([existing, incoming], ignore_index=True)
    combined = ensure_dataset_schema(combined)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    deduped = before - len(combined)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    return combined, deduped


def quality_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "rows": 0,
            "start": None,
            "end": None,
            "expected_hours": 0,
            "missing_hours": 0,
            "coverage_pct": 0.0,
        }

    start = df["timestamp"].min().floor("h")
    end = df["timestamp"].max().floor("h")
    expected = pd.date_range(start=start, end=end, freq="h")
    unique_hours = df["timestamp"].dt.floor("h").drop_duplicates()
    missing = max(0, len(expected) - len(unique_hours))
    coverage = 0.0 if len(expected) == 0 else (len(unique_hours) / len(expected)) * 100

    return {
        "rows": int(len(df)),
        "start": str(start),
        "end": str(end),
        "expected_hours": int(len(expected)),
        "missing_hours": int(missing),
        "coverage_pct": round(float(coverage), 2),
    }


def build_sequences(df: pd.DataFrame, look_back: int) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    values = df[FEATURES].astype(np.float32).values
    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values)

    x_data, y_data = [], []
    for idx in range(len(values_scaled) - look_back):
        x_data.append(values_scaled[idx : idx + look_back])
        y_data.append(values_scaled[idx + look_back, 0])

    x_arr = np.array(x_data, dtype=np.float32)
    y_arr = np.array(y_data, dtype=np.float32)
    return x_arr, y_arr, scaler


def train_model(
    dataset_df: pd.DataFrame,
    model_path: Path,
    scaler_path: Path,
    look_back: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    train_split: float,
    early_stop_patience: int,
) -> dict:
    x_arr, y_arr, scaler = build_sequences(dataset_df, look_back=look_back)

    if len(x_arr) < 2:
        raise ValueError(
            f"Not enough training sequences ({len(x_arr)}). "
            f"Need more hourly rows or lower --look-back."
        )

    split_idx = int(len(x_arr) * train_split)
    split_idx = max(1, min(split_idx, len(x_arr) - 1))

    x_train, y_train = x_arr[:split_idx], y_arr[:split_idx]
    x_val, y_val = x_arr[split_idx:], y_arr[split_idx:]

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=False,
    )

    model = LSTMModel(
        input_size=len(FEATURES),
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=1,
        dropout=dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    best_state = None
    best_val_loss = float("inf")
    patience = 0

    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb).squeeze(-1)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_preds = model(x_val_t).squeeze(-1)
            val_loss = criterion(val_preds, y_val_t).item()

        avg_train = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        print(f"[Epoch {epoch:03d}] train_loss={avg_train:.6f} val_loss={val_loss:.6f}")

        if val_loss < (best_val_loss - 1e-7):
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training failed to produce a valid model state.")

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    return {
        "best_val_loss": float(best_val_loss),
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
        "total_sequences": int(len(x_arr)),
        "look_back": int(look_back),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automate YOLO -> hourly dataset -> LSTM retraining."
    )
    parser.add_argument("--yolo-csv", required=True, help="Path to YOLO output CSV.")
    parser.add_argument(
        "--dataset",
        default="historical_traffic_hourly.csv",
        help="Path to historical hourly dataset CSV.",
    )
    parser.add_argument(
        "--model-path",
        default="best_lstm_traffic_model.pth",
        help="Where to write trained model weights.",
    )
    parser.add_argument(
        "--scaler-path",
        default="scaler.pkl",
        help="Where to write fitted scaler.",
    )
    parser.add_argument(
        "--metadata-path",
        default="pipeline_metadata.json",
        help="Where to write run metadata JSON.",
    )
    parser.add_argument(
        "--aggregation",
        choices=["sum", "mean"],
        default="sum",
        help="How to aggregate minute vehicle_count into hourly Total.",
    )
    parser.add_argument(
        "--count-column",
        default=None,
        help="Column in YOLO CSV to treat as vehicle count. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--location",
        default=None,
        help="Optional location filter if YOLO CSV includes a location column.",
    )
    parser.add_argument("--look-back", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=50)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--min-hours-for-training", type=int, default=96)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    yolo_csv_path = Path(args.yolo_csv).resolve()
    dataset_path = Path(args.dataset).resolve()
    model_path = Path(args.model_path).resolve()
    scaler_path = Path(args.scaler_path).resolve()
    metadata_path = Path(args.metadata_path).resolve()

    print(f"Loading YOLO CSV: {yolo_csv_path}")
    yolo_df = load_yolo_csv(
        path=yolo_csv_path,
        count_column=args.count_column,
        location=args.location,
    )
    print(f"YOLO rows loaded: {len(yolo_df)}")

    hourly_df = aggregate_hourly(yolo_df, aggregation=args.aggregation)
    print(f"Hourly rows produced: {len(hourly_df)}")

    existing = load_existing_dataset(dataset_path)
    merged, deduped = merge_datasets(existing, hourly_df)
    merged.to_csv(dataset_path, index=False)

    quality = quality_summary(merged)
    print(
        "Dataset updated:",
        f"rows={quality['rows']}",
        f"range={quality['start']} -> {quality['end']}",
        f"missing_hours={quality['missing_hours']}",
        f"coverage={quality['coverage_pct']}%",
        f"deduped={deduped}",
    )

    training_metrics = None
    if args.skip_training:
        print("Training skipped (--skip-training).")
    elif len(merged) < args.min_hours_for_training:
        print(
            "Training skipped due to low data volume:",
            f"{len(merged)} rows < min {args.min_hours_for_training}.",
        )
    else:
        print("Starting retraining...")
        training_metrics = train_model(
            dataset_df=merged,
            model_path=model_path,
            scaler_path=scaler_path,
            look_back=args.look_back,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            train_split=args.train_split,
            early_stop_patience=args.early_stop_patience,
        )
        print(f"Training complete. Best val loss: {training_metrics['best_val_loss']:.6f}")
        print(f"Saved model: {model_path}")
        print(f"Saved scaler: {scaler_path}")

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_yolo_csv": str(yolo_csv_path),
        "aggregation": args.aggregation,
        "location_filter": args.location,
        "incoming_rows": int(len(yolo_df)),
        "hourly_rows_produced": int(len(hourly_df)),
        "dataset_path": str(dataset_path),
        "dataset_quality": quality,
        "training_metrics": training_metrics,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Run metadata written: {metadata_path}")


if __name__ == "__main__":
    main()
