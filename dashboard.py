import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

st.set_page_config(page_title="EV Battery Degradation", page_icon="🔋", layout="wide")

# ── Model definition (must match train.py) ────────────────────────────────────
FEATURE_COLS = [
    'voltage_mean','voltage_min','voltage_std','voltage_drop','voltage_slope',
    'temp_mean','temp_max','temp_rise',
    'current_mean','current_std',
    'discharge_time','internal_resistance_proxy'
]
SEQUENCE_LENGTH = 10
HIDDEN = 64
LAYERS = 2

class BatteryLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=len(FEATURE_COLS),
            hidden_size=HIDDEN, num_layers=LAYERS,
            batch_first=True, dropout=0.2
        )
        self.head = nn.Sequential(
            nn.Linear(HIDDEN, 32), nn.ReLU(), nn.Linear(32, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze()

BASE = os.path.dirname(__file__)

@st.cache_resource
def load_model():
    m = BatteryLSTM()
    m.load_state_dict(torch.load(
        os.path.join(BASE, "battery_lstm.pt"), map_location="cpu"))
    m.eval()
    sx = joblib.load(os.path.join(BASE, "scaler_X.pkl"))
    sy = joblib.load(os.path.join(BASE, "scaler_y.pkl"))
    return m, sx, sy

@st.cache_data
def load_features():
    train = pd.read_csv(os.path.join(BASE, "train_features.csv"))
    test  = pd.read_csv(os.path.join(BASE, "test_features.csv"))
    return pd.concat([train, test], ignore_index=True)

def predict_battery(df_battery, model, scaler_X, scaler_y):
    """Run real LSTM inference on a battery's cycle history."""
    df = df_battery.sort_values("cycle_number").copy()
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

    X = scaler_X.transform(df[FEATURE_COLS].values.astype(np.float32))
    preds, cycles, actuals = [], [], []

    for i in range(len(X) - SEQUENCE_LENGTH):
        seq   = torch.tensor(X[i:i+SEQUENCE_LENGTH], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred_scaled = model(seq).item()
        pred_cap = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        preds.append(pred_cap)
        cycles.append(df["cycle_number"].iloc[i + SEQUENCE_LENGTH])
        if "capacity" in df.columns:
            actuals.append(df["capacity"].iloc[i + SEQUENCE_LENGTH])

    nominal = df["capacity"].iloc[:5].max() if "capacity" in df.columns else 2.0
    result  = pd.DataFrame({"cycle": cycles, "predicted_capacity": preds})
    if actuals:
        result["actual_capacity"] = actuals
    result["soh"]  = (result["predicted_capacity"] / nominal * 100).round(2)
    result["rul"]  = result["soh"].apply(
        lambda s: max(0, int((s - 80) / max(
            (result["soh"].iloc[0] - result["soh"].iloc[-1]) /
            max(len(result) - 1, 1), 0.01))))
    result["nominal"] = nominal
    return result

# ── Load ──────────────────────────────────────────────────────────────────────
model, scaler_X, scaler_y = load_model()
all_features = load_features()
batteries    = sorted(all_features["battery_id"].unique())

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔋 EV Battery Degradation Predictor")
st.caption("Real LSTM inference on NASA Li-ion aging dataset  •  8.36% MAPE on unseen batteries")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔋 Battery Selection")
    battery_id = st.selectbox("Select Battery", batteries)

    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown("- 2-layer LSTM (64 hidden units)")
    st.markdown("- Trained: B0033, B0034, B0036")
    st.markdown("- Tested: B0005, B0006, B0007")
    st.markdown("- MAE: 0.1303 Ah")
    st.markdown("- MAPE: 8.36%")
    st.markdown("- Sequence length: 10 cycles")
    st.markdown("---")
    st.markdown("**Features (12)**")
    for f in FEATURE_COLS:
        st.markdown(f"- `{f}`")
    if st.button("🔄 Clear Cache"):
        st.cache_data.clear()
        st.rerun()

# ── Run inference ─────────────────────────────────────────────────────────────
df_bat = all_features[all_features["battery_id"] == battery_id]
result = predict_battery(df_bat, model, scaler_X, scaler_y)

latest  = result.iloc[-1]
soh     = latest["soh"]
nominal = latest["nominal"]

if soh >= 90:
    status = "✅ Healthy — normal operation"
    color  = "#2DC653"
elif soh >= 80:
    status = "⚠️ Degrading — schedule maintenance soon"
    color  = "#F4D03F"
else:
    status = "🔴 Critical — below 80% SoH end-of-life threshold"
    color  = "#E63946"

# ── KPIs ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Battery",           battery_id)
k2.metric("Latest Capacity",   f"{latest['predicted_capacity']:.4f} Ah")
k3.metric("State of Health",   f"{soh:.1f}%",
           delta=f"{result['soh'].iloc[-1] - result['soh'].iloc[0]:.1f}% total drift")
k4.metric("Est. RUL",          f"{int(latest['rul'])} cycles")
k5.metric("Nominal Capacity",  f"{nominal:.2f} Ah")

st.markdown(f"**Status:** <span style='color:{color}'>{status}</span>",
            unsafe_allow_html=True)
st.divider()

# ── Row 1: Capacity curve + SoH bar ──────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.subheader("Predicted vs Actual Capacity")
    fig1 = go.Figure()

    if "actual_capacity" in result.columns:
        fig1.add_trace(go.Scatter(
            x=result["cycle"], y=result["actual_capacity"],
            mode="lines", name="Actual",
            line=dict(color="#888", width=1.5, dash="dot")
        ))

    fig1.add_trace(go.Scatter(
        x=result["cycle"], y=result["predicted_capacity"],
        mode="lines+markers", name="LSTM Prediction",
        line=dict(color="#00d4aa", width=2),
        marker=dict(size=4)
    ))
    fig1.add_hline(
        y=nominal * 0.8, line_dash="dash", line_color="#E63946",
        annotation_text="80% EOL threshold"
    )
    fig1.update_layout(height=360, template="plotly_dark",
                       yaxis_title="Capacity (Ah)", xaxis_title="Cycle",
                       hovermode="x unified", legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("State of Health by Cycle")
    bar_colors = ["#E63946" if s < 80 else "#F4D03F" if s < 90 else "#2DC653"
                  for s in result["soh"]]
    fig2 = go.Figure(go.Bar(
        x=result["cycle"], y=result["soh"],
        marker_color=bar_colors, name="SoH %"
    ))
    fig2.add_hline(y=80, line_dash="dash", line_color="#E63946",
                   annotation_text="EOL 80%")
    fig2.add_hline(y=90, line_dash="dash", line_color="#F4D03F",
                   annotation_text="Warning 90%")
    fig2.update_layout(height=360, template="plotly_dark",
                       yaxis_title="SoH (%)", xaxis_title="Cycle",
                       yaxis_range=[50, 105])
    st.plotly_chart(fig2, use_container_width=True)

# ── Row 2: Degradation rate + RUL ────────────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Capacity Degradation Rate")
    result["deg_rate"] = result["predicted_capacity"].diff().abs()
    fig3 = go.Figure(go.Scatter(
        x=result["cycle"], y=result["deg_rate"],
        mode="lines", fill="tozeroy",
        line=dict(color="#F77F00", width=2),
        fillcolor="rgba(247,127,0,0.15)"
    ))
    fig3.update_layout(height=300, template="plotly_dark",
                       yaxis_title="Capacity Drop (Ah/cycle)",
                       xaxis_title="Cycle")
    st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader("Remaining Useful Life Estimate")
    fig4 = go.Figure(go.Scatter(
        x=result["cycle"], y=result["rul"],
        mode="lines+markers",
        line=dict(color="#9B5DE5", width=2),
        fill="tozeroy", fillcolor="rgba(155,93,229,0.15)"
    ))
    fig4.update_layout(height=300, template="plotly_dark",
                       yaxis_title="Est. Cycles Remaining",
                       xaxis_title="Cycle")
    st.plotly_chart(fig4, use_container_width=True)

# ── All batteries comparison ──────────────────────────────────────────────────
st.subheader("📊 Fleet Overview — All Batteries")

fleet_rows = []
for bid in batteries:
    df_b = all_features[all_features["battery_id"] == bid]
    if len(df_b) <= SEQUENCE_LENGTH:
        continue
    res = predict_battery(df_b, model, scaler_X, scaler_y)
    fleet_rows.append({
        "battery_id":    bid,
        "cycles_tested": len(res),
        "final_soh":     res["soh"].iloc[-1],
        "final_cap":     res["predicted_capacity"].iloc[-1],
        "soh_drop":      res["soh"].iloc[0] - res["soh"].iloc[-1],
        "rul":           int(res["rul"].iloc[-1]),
        "status":        "Critical" if res["soh"].iloc[-1] < 80
                         else "Degrading" if res["soh"].iloc[-1] < 90 else "Healthy"
    })

fleet = pd.DataFrame(fleet_rows)
tier_colors = {"Healthy": "#2DC653", "Degrading": "#F4D03F", "Critical": "#E63946"}

col5, col6 = st.columns(2)
with col5:
    fig5 = px_bar = go.Figure(go.Bar(
        x=fleet["battery_id"],
        y=fleet["final_soh"],
        marker_color=[tier_colors[s] for s in fleet["status"]],
        text=fleet["final_soh"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside"
    ))
    px_bar.add_hline(y=80, line_dash="dash", line_color="#E63946")
    px_bar.update_layout(height=300, template="plotly_dark",
                         title="Final SoH by Battery",
                         yaxis_title="SoH (%)", yaxis_range=[0, 115])
    st.plotly_chart(px_bar, use_container_width=True)

with col6:
    fig6 = go.Figure(go.Bar(
        x=fleet["battery_id"],
        y=fleet["rul"],
        marker_color=[tier_colors[s] for s in fleet["status"]],
        text=fleet["rul"],
        textposition="outside"
    ))
    fig6.update_layout(height=300, template="plotly_dark",
                       title="Estimated RUL by Battery",
                       yaxis_title="Cycles Remaining")
    st.plotly_chart(fig6, use_container_width=True)

st.dataframe(fleet, use_container_width=True)

# ── Raw data ──────────────────────────────────────────────────────────────────
with st.expander("📋 Raw Prediction Data"):
    st.dataframe(result, use_container_width=True)
