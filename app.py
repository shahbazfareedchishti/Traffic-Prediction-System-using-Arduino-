from flask import Flask, render_template, request, redirect, url_for
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import joblib
from datetime import datetime, timedelta
import uuid
from pathlib import Path

app = Flask(__name__)

# ── 1. Exact same LSTM model class used during training ──
class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=50, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# ── 2. Load model, scaler, and historical profiles for warm start ──
MODEL_PATH = Path("best_lstm_traffic_model.pth")
SCALER_PATH = Path("scaler.pkl")
HISTORICAL_PATH = Path("historical_traffic_hourly.csv")

device = torch.device('cpu')  # change to 'cuda' if GPU available
LOOK_BACK = 24
FEATURES = ['Total', 'hour', 'day_of_week']
PREDICTION_RESULTS = {}
model = None
scaler = None
profile_by_dow_hour = {}
fallback_total = 0.0
artifact_mtime = {'model': None, 'scaler': None, 'historical': None}


def get_mtime(path):
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def load_artifacts():
    global model, scaler

    loaded_model = LSTMModel(input_size=3)
    loaded_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    loaded_model.to(device)
    loaded_model.eval()

    loaded_scaler = joblib.load(SCALER_PATH)

    model = loaded_model
    scaler = loaded_scaler
    artifact_mtime['model'] = get_mtime(MODEL_PATH)
    artifact_mtime['scaler'] = get_mtime(SCALER_PATH)


def load_historical_profiles():
    global profile_by_dow_hour, fallback_total

    historical_df = pd.read_csv(HISTORICAL_PATH)
    profile_by_dow_hour = (
        historical_df
        .groupby(['day_of_week', 'hour'])['Total']
        .mean()
        .to_dict()
    )
    fallback_total = float(historical_df['Total'].mean()) if not historical_df.empty else 0.0
    artifact_mtime['historical'] = get_mtime(HISTORICAL_PATH)


def refresh_if_updated():
    model_changed = get_mtime(MODEL_PATH) != artifact_mtime['model']
    scaler_changed = get_mtime(SCALER_PATH) != artifact_mtime['scaler']
    historical_changed = get_mtime(HISTORICAL_PATH) != artifact_mtime['historical']

    if model is None or scaler is None or model_changed or scaler_changed:
        load_artifacts()

    if historical_changed or not profile_by_dow_hour:
        load_historical_profiles()


load_artifacts()

try:
    load_historical_profiles()
except Exception:
    profile_by_dow_hour = {}
    fallback_total = 0.0


def build_initial_sequence(start_dt):
    rows = []
    current = start_dt - timedelta(hours=LOOK_BACK)

    for _ in range(LOOK_BACK):
        hour = current.hour
        dow = current.weekday()
        total = profile_by_dow_hour.get((dow, hour), fallback_total)
        rows.append([total, hour, dow])
        current += timedelta(hours=1)

    return np.array(rows, dtype=np.float32)


def find_peak_hours(future_times, preds_original):
    if len(preds_original) == 0:
        return [], 0

    normal_mean = float(np.mean(preds_original))
    normal_std = float(np.std(preds_original))
    threshold = normal_mean + normal_std

    peak_hours = []
    for timestamp, value in zip(future_times, preds_original):
        if value >= threshold:
            peak_hours.append((timestamp.strftime("%Y-%m-%d %H:%M"), round(max(0, value))))

    # Ensure at least one peak is reported, even for very flat curves.
    if not peak_hours:
        max_value = float(np.max(preds_original))
        for timestamp, value in zip(future_times, preds_original):
            if float(value) == max_value:
                peak_hours.append((timestamp.strftime("%Y-%m-%d %H:%M"), round(max(0, value))))

    return peak_hours, round(threshold)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        return predict()

    result_id = request.args.get('result_id')
    if result_id and result_id in PREDICTION_RESULTS:
        payload = PREDICTION_RESULTS.pop(result_id)
        return render_template('index.html', **payload)

    return render_template('index.html',
                           prediction_table=None,
                           peak_hours=None,
                           normal_threshold=None,
                           plot_url=None,
                           start_str=None,
                           error=None)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return redirect(url_for('home'))

    prediction_table = None
    plot_url = None
    start_str = None
    error = None

    start_date = request.form.get('start_date')
    start_time = request.form.get('start_time', '00:00')
    peak_hours = None
    normal_threshold = None

    try:
        refresh_if_updated()

        start_dt = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M")
        start_str = start_dt.strftime("%Y-%m-%d %H:%M")

        # Forecast only within the selected date (no next-day rows).
        future_times = []
        cursor = start_dt
        while cursor.date() == start_dt.date():
            future_times.append(cursor)
            cursor += timedelta(hours=1)

        future_steps = len(future_times)
        if future_steps == 0:
            raise ValueError("Unable to build forecast window for the selected date.")

        # Build a date-aware look-back window from historical weekday/hour averages
        init_seq = build_initial_sequence(start_dt)
        data_scaled = scaler.transform(init_seq)
        current_seq = torch.from_numpy(data_scaled).float().unsqueeze(0).to(device)

        preds_scaled = []
        current_time = start_dt

        with torch.no_grad():
            for _ in range(future_steps):
                pred_scaled = model(current_seq).item()
                preds_scaled.append(pred_scaled)

                # Convert model output to raw Total before scaling the next autoregressive row.
                pred_dummy = np.zeros((1, len(FEATURES)))
                pred_dummy[0, 0] = pred_scaled
                pred_total = scaler.inverse_transform(pred_dummy)[0, 0]

                next_hour = current_time.hour
                next_dow = current_time.weekday()  # 0 = Monday ... 6 = Sunday
                new_row = np.array([[pred_total, next_hour, next_dow]], dtype=np.float32)
                new_scaled = scaler.transform(new_row)

                current_seq = torch.cat(
                    (current_seq[:, 1:, :], torch.from_numpy(new_scaled).float().unsqueeze(0).to(device)),
                    dim=1
                )

                current_time += timedelta(hours=1)

        # ── Inverse scale to real vehicle counts ──
        dummy = np.zeros((len(preds_scaled), len(FEATURES)))
        dummy[:, 0] = preds_scaled
        preds_original = scaler.inverse_transform(dummy)[:, 0]
        peak_hours, normal_threshold = find_peak_hours(future_times, preds_original)
        peak_timestamps = {timestamp for timestamp, _ in peak_hours}

        # Build table data
        prediction_table = [
            (t.strftime("%Y-%m-%d %H:%M"), round(max(0, p)), t.strftime("%Y-%m-%d %H:%M") in peak_timestamps)
            for t, p in zip(future_times, preds_original)
        ]

        # ── Generate plot ──
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(future_times, preds_original, marker='o', color='orange', linewidth=2)
        ax.set_title(f"Predicted Hourly Traffic Volume\nStarting: {start_str}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Predicted Vehicle Count")
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close(fig)

    except Exception as ex:
        error = str(ex)

    result_id = str(uuid.uuid4())
    PREDICTION_RESULTS[result_id] = {
        'prediction_table': prediction_table,
        'peak_hours': peak_hours,
        'normal_threshold': normal_threshold,
        'plot_url': plot_url,
        'start_str': start_str,
        'error': error
    }

    return redirect(url_for('home', result_id=result_id))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
