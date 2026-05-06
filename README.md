# 🔋 EV Battery Degradation Predictor

A deep learning system that predicts lithium-ion battery State of Health (SoH) and Remaining Useful Life (RUL) from discharge cycle telemetry, using a 2-layer LSTM trained on NASA battery aging data.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-EE4C2C?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-live-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📊 Live Dashboard

👉 **[View Live App](https://gdiaz38-ev-battery-degradation.streamlit.app)**

---

## Overview

Battery degradation is one of the costliest failure modes in EVs and IoT devices. This project trains a sequence-to-one LSTM on NASA Li-ion discharge cycles, learning to predict next-cycle capacity from the last 10 cycles of telemetry — then generalizes to completely unseen batteries at 8.36% MAPE.

Key question it answers: *Given the last 10 discharge cycles of a battery, how much capacity remains — and how many cycles are left before end-of-life?*

---

## Key Results

| Metric | Value |
|---|---|
| Test MAE | 0.1303 Ah |
| Test MAPE | **8.36%** on unseen batteries |
| Architecture | 2-layer LSTM → FC head |
| Generalization | Trained B0033/B0034/B0036, tested B0005/B0006/B0007 |

---

## Features

- **Real LSTM inference** — actual model weights loaded and run on every page load
- **12 engineered features** per discharge cycle: voltage statistics, temperature profile, current behavior, discharge time, internal resistance proxy
- **Fleet overview** — SoH and RUL comparison across all batteries simultaneously
- **Capacity degradation rate** — cycle-by-cycle capacity drop visualization
- **EOL threshold line** — 80% SoH end-of-life marker on all capacity charts
- **Predicted vs actual overlay** — model predictions shown against ground truth

---

## Data

| Source | Description |
|---|---|
| [NASA Battery Dataset](https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset) | Li-ion cells cycled to failure under controlled conditions |
| Batteries B0033/B0034/B0036 | Training set |
| Batteries B0005/B0006/B0007 | Test set — completely unseen during training |

---

## Project Structure

```
ev-battery-degradation/
├── dashboard.py          # Streamlit dashboard — real LSTM inference
├── train.py              # LSTM training script
├── features.py           # Cycle feature engineering from raw discharge CSVs
├── api.py                # FastAPI REST endpoint (local use)
├── battery_lstm.pt       # Trained model weights
├── scaler_X.pkl          # Feature scaler
├── scaler_y.pkl          # Target scaler
├── train_features.csv    # Engineered features — training batteries
├── test_features.csv     # Engineered features — test batteries
└── requirements.txt
```

---

## How It Works

```
Raw discharge cycle CSV (voltage, current, temperature over time)
        ↓
features.py extracts 12 features per cycle:
  voltage_mean, voltage_min, voltage_std, voltage_drop, voltage_slope
  temp_mean, temp_max, temp_rise
  current_mean, current_std
  discharge_time, internal_resistance_proxy
        ↓
LSTM looks at last 10 cycles → predicts next cycle capacity
        ↓
SoH = predicted_capacity / nominal_capacity × 100
RUL = estimated cycles remaining before SoH drops below 80%
```

---

## Model Architecture

```
Input: (batch, 10 cycles, 12 features)
        ↓
LSTM Layer 1 (hidden=64, dropout=0.2)
        ↓
LSTM Layer 2 (hidden=64, dropout=0.2)
        ↓
FC(64 → 32) → ReLU → FC(32 → 1)
        ↓
Output: predicted capacity (Ah)
```

| Hyperparameter | Value |
|---|---|
| Sequence length | 10 cycles |
| Hidden units | 64 |
| LSTM layers | 2 |
| Dropout | 0.2 |
| Optimizer | Adam (lr=0.001) |
| Epochs | 80 |
| Batch size | 32 |

---

## Feature Engineering

Each discharge cycle is reduced to 12 scalar features:

| Feature | Description |
|---|---|
| `voltage_mean` | Mean discharge voltage |
| `voltage_min` | Minimum voltage reached |
| `voltage_std` | Voltage variability |
| `voltage_drop` | Start minus end voltage |
| `voltage_slope` | Linear fit slope over time |
| `temp_mean` | Mean cell temperature |
| `temp_max` | Peak temperature |
| `temp_rise` | Temperature delta start→end |
| `current_mean` | Mean discharge current |
| `current_std` | Current variability |
| `discharge_time` | Total discharge duration (s) |
| `internal_resistance_proxy` | ΔV/ΔI median |

---

## Dashboard Sections

**KPI Row** — battery ID, latest capacity, SoH with drift delta, estimated RUL, nominal capacity

**Predicted vs Actual Capacity** — LSTM predictions overlaid on ground truth with 80% EOL line

**State of Health by Cycle** — bar chart colored green/yellow/red by tier

**Degradation Rate** — cycle-by-cycle capacity drop

**Remaining Useful Life** — estimated cycles remaining over time

**Fleet Overview** — final SoH and RUL comparison across all batteries with status coloring

---

## Local Setup

```bash
git clone https://github.com/gdiaz38/ev-battery-degradation
cd ev-battery-degradation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard.py
```

To retrain from scratch (requires NASA dataset via kagglehub):

```bash
python3 features.py   # extract cycle features
python3 train.py      # train LSTM, saves battery_lstm.pt
```

---

## Tech Stack

`Python 3.11` · `PyTorch` · `Streamlit` · `Plotly` · `Pandas` · `NumPy` · `Scikit-learn` · `joblib`

---

## Affiliation

University of California, Riverside — MS in Engineering Management
Part of a portfolio of 10 live data science projects spanning computer vision, NLP, supply chain, and healthcare ML.

---

## License

MIT
