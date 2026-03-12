import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="EV Battery Degradation Predictor",
    page_icon="🔋",
    layout="wide"
)

API_URL = "http://localhost:8000"

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #00d4aa; }
    .sub-header  { font-size: 1rem; color: #888; margin-bottom: 2rem; }
    .metric-card { background: #1a1a2e; border-radius: 12px; padding: 1.2rem;
                   border: 1px solid #2a2a4a; text-align: center; }
    .soh-good    { color: #00d4aa; font-size: 2.5rem; font-weight: 700; }
    .soh-warn    { color: #ffa500; font-size: 2.5rem; font-weight: 700; }
    .soh-crit    { color: #ff4444; font-size: 2.5rem; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔋 EV Battery Degradation Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time State of Health monitoring using NASA battery data + LSTM neural network</div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Battery Configuration")
battery_id = st.sidebar.selectbox("Select Battery ID", ["B0005", "B0006", "B0007", "B0033"])
num_cycles  = st.sidebar.slider("Cycles to simulate", 10, 50, 20)
degradation = st.sidebar.slider("Degradation rate (demo)", 0.001, 0.010, 0.004, 0.001)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.markdown("- Architecture: 2-layer LSTM")
st.sidebar.markdown("- Trained on: B0033, B0034, B0036")
st.sidebar.markdown("- Test MAE: 0.1303 Ah")
st.sidebar.markdown("- Test MAPE: 8.36%")

# ── Generate simulated telemetry ──────────────────────────────────────────────
def generate_cycles(n, deg_rate):
    cycles = []
    for i in range(n):
        decay = i * deg_rate
        cycles.append({
            "voltage_mean":              3.71 - decay * 8,
            "voltage_min":               2.80 - decay * 4,
            "voltage_std":               0.35 + decay * 2,
            "voltage_drop":              1.20 + decay * 10,
            "voltage_slope":            -0.0012 - decay * 0.05,
            "temp_mean":                 24.5  + decay * 20,
            "temp_max":                  26.1  + decay * 25,
            "temp_rise":                 1.6   + decay * 15,
            "current_mean":             -1.05 - decay * 2,
            "current_std":               0.08  + decay * 3,
            "discharge_time":            4200  - decay * 8000,
            "internal_resistance_proxy": -0.15 - decay * 5
        })
    return cycles

# ── Run predictions across a rolling window ───────────────────────────────────
@st.cache_data
def run_predictions(battery_id, n, deg_rate):
    all_cycles = generate_cycles(n + 10, deg_rate)
    results = []
    for i in range(n):
        window = all_cycles[i:i+10]
        try:
            r = requests.post(f"{API_URL}/predict", json={
                "battery_id": battery_id,
                "cycles": window
            }, timeout=5)
            if r.status_code == 200:
                d = r.json()
                results.append({
                    "cycle":     i + 1,
                    "capacity":  d["predicted_capacity"],
                    "soh":       d["state_of_health_pct"],
                    "rul":       d["rul_estimate"],
                    "warning":   d["warning"]
                })
        except Exception:
            pass
    return pd.DataFrame(results)

if st.button("▶ Run Prediction Simulation", type="primary"):
    with st.spinner("Running LSTM inference across cycles..."):
        df = run_predictions(battery_id, num_cycles, degradation)

    if df.empty:
        st.error("Could not reach API. Make sure `python3 api.py` is running.")
    else:
        # ── KPI Row ───────────────────────────────────────────────────────────
        latest = df.iloc[-1]
        soh    = latest["soh"]
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Battery ID", battery_id)
        with col2:
            st.metric("Latest Capacity", f"{latest['capacity']:.4f} Ah")
        with col3:
            css = "soh-good" if soh >= 90 else ("soh-warn" if soh >= 80 else "soh-crit")
            st.metric("State of Health", f"{soh:.1f}%")
        with col4:
            st.metric("Cycles Simulated", len(df))

        st.markdown(f"**Status:** {latest['warning']}")
        st.markdown("---")

        # ── Main charts ───────────────────────────────────────────────────────
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Predicted Capacity Over Cycles",
                "State of Health (%)",
                "Capacity Degradation Rate",
                "SoH Distribution"
            )
        )

        # Capacity curve
        fig.add_trace(go.Scatter(
            x=df["cycle"], y=df["capacity"],
            mode="lines+markers", name="Capacity",
            line=dict(color="#00d4aa", width=2)
        ), row=1, col=1)
        fig.add_hline(
            y=df["capacity"].max() * 0.8,
            line_dash="dash", line_color="red",
            annotation_text="80% EOL threshold",
            row=1, col=1
        )

        # SoH curve
        colors = ["#ff4444" if s < 80 else "#ffa500" if s < 90 else "#00d4aa" for s in df["soh"]]
        fig.add_trace(go.Bar(
            x=df["cycle"], y=df["soh"],
            name="SoH %", marker_color=colors
        ), row=1, col=2)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=1, col=2)

        # Degradation rate
        df["deg_rate"] = df["capacity"].diff().abs()
        fig.add_trace(go.Scatter(
            x=df["cycle"], y=df["deg_rate"],
            mode="lines", name="Degradation Rate",
            line=dict(color="#ffa500", width=2),
            fill="tozeroy", fillcolor="rgba(255,165,0,0.1)"
        ), row=2, col=1)

        # SoH histogram
        fig.add_trace(go.Histogram(
            x=df["soh"], name="SoH Distribution",
            marker_color="#00d4aa", opacity=0.75
        ), row=2, col=2)

        fig.update_layout(
            height=600,
            template="plotly_dark",
            showlegend=False,
            paper_bgcolor="#0d0d1a",
            plot_bgcolor="#0d0d1a"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Raw data table ────────────────────────────────────────────────────
        with st.expander("📊 View raw prediction data"):
            st.dataframe(df, use_container_width=True)

else:
    st.info("👈 Configure the battery on the left, then click **Run Prediction Simulation**")
    st.markdown("""
    ### How this works
    1. **Telemetry** — 12 features extracted from each discharge cycle (voltage, temp, current)
    2. **LSTM** — looks at the last 10 cycles to predict next cycle's capacity
    3. **SoH** — predicted capacity ÷ nominal capacity × 100
    4. **API** — FastAPI endpoint accepts real-time JSON telemetry, returns prediction in <50ms
    
    **Cross-battery generalization:** Trained on B0033/B0034/B0036, tested on B0005/B0006/B0007 — 
    completely unseen batteries, 8.36% MAPE.
    """)
