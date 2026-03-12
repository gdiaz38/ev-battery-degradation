import numpy as np
import torch
import torch.nn as nn
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

# ── Model definition (must match train.py exactly) ───────────────────────────
FEATURE_COLS = [
    'voltage_mean', 'voltage_min', 'voltage_std', 'voltage_drop', 'voltage_slope',
    'temp_mean', 'temp_max', 'temp_rise',
    'current_mean', 'current_std',
    'discharge_time', 'internal_resistance_proxy'
]
SEQUENCE_LENGTH = 10
HIDDEN = 64
LAYERS = 2

class BatteryLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=len(FEATURE_COLS),
            hidden_size=HIDDEN,
            num_layers=LAYERS,
            batch_first=True,
            dropout=0.2
        )
        self.head = nn.Sequential(
            nn.Linear(HIDDEN, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze()

# ── Load model & scalers ──────────────────────────────────────────────────────
model = BatteryLSTM()
model.load_state_dict(torch.load("battery_lstm.pt", map_location="cpu"))
model.eval()

scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

NOMINAL_CAPACITY = 2.0  # Ah — fresh battery baseline

# ── API ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="EV Battery Degradation Predictor",
    description="Predicts State of Health (SoH) and Remaining Useful Life (RUL) from discharge cycle telemetry.",
    version="1.0.0"
)

class CycleFeatures(BaseModel):
    voltage_mean:               float
    voltage_min:                float
    voltage_std:                float
    voltage_drop:               float
    voltage_slope:              float
    temp_mean:                  float
    temp_max:                   float
    temp_rise:                  float
    current_mean:               float
    current_std:                float
    discharge_time:             float
    internal_resistance_proxy:  float

class PredictionRequest(BaseModel):
    battery_id: str
    cycles: List[CycleFeatures]  # must send exactly 10 cycles

class PredictionResponse(BaseModel):
    battery_id:          str
    predicted_capacity:  float
    state_of_health_pct: float
    rul_estimate:        str
    warning:             str

@app.get("/health")
def health():
    return {"status": "ok", "model": "BatteryLSTM", "version": "1.0.0"}

@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if len(req.cycles) != SEQUENCE_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Must provide exactly {SEQUENCE_LENGTH} cycles, got {len(req.cycles)}"
        )

    # Build feature matrix
    X = np.array([[
        c.voltage_mean, c.voltage_min, c.voltage_std, c.voltage_drop, c.voltage_slope,
        c.temp_mean, c.temp_max, c.temp_rise,
        c.current_mean, c.current_std,
        c.discharge_time, c.internal_resistance_proxy
    ] for c in req.cycles], dtype=np.float32)

    # Scale and predict
    X_scaled = scaler_X.transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0)  # (1, 10, 12)

    with torch.no_grad():
        pred_scaled = model(X_tensor).item()

    predicted_capacity = float(scaler_y.inverse_transform([[pred_scaled]])[0][0])
    soh = (predicted_capacity / NOMINAL_CAPACITY) * 100

    # RUL estimate based on current SoH trend
    if soh >= 90:
        rul = "Excellent — 80%+ cycles remaining"
    elif soh >= 80:
        rul = "Good — approaching end-of-life threshold"
    elif soh >= 70:
        rul = "Warning — recommend service inspection"
    else:
        rul = "Critical — battery replacement advised"

    warning = "⚠️ Below 80% SoH — EOL threshold crossed" if soh < 80 else "✅ Within normal operating range"

    return PredictionResponse(
        battery_id=req.battery_id,
        predicted_capacity=round(predicted_capacity, 4),
        state_of_health_pct=round(soh, 2),
        rul_estimate=rul,
        warning=warning
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
