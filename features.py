import pandas as pd
import numpy as np
import os

DATA_PATH = "/Users/gabrieldiaz/.cache/kagglehub/datasets/patrickfleith/nasa-battery-dataset/versions/2/cleaned_dataset"

def extract_cycle_features(filepath):
    """
    Takes one discharge cycle CSV, returns a single row of features.
    This is what the LSTM will learn from — one row per cycle.
    """
    df = pd.read_csv(filepath)

    # Only keep discharge portion (current is negative during discharge)
    df = df[df['Current_measured'] < 0].copy()

    if len(df) < 10:
        return None

    features = {}

    # --- Voltage features ---
    features['voltage_mean']  = df['Voltage_measured'].mean()
    features['voltage_min']   = df['Voltage_measured'].min()
    features['voltage_std']   = df['Voltage_measured'].std()
    # Voltage drop = how fast voltage fell (proxy for internal resistance)
    features['voltage_drop']  = df['Voltage_measured'].iloc[0] - df['Voltage_measured'].iloc[-1]
    # Slope of voltage over time (degraded batteries drop faster)
    features['voltage_slope'] = np.polyfit(df['Time'], df['Voltage_measured'], 1)[0]

    # --- Temperature features ---
    features['temp_mean']     = df['Temperature_measured'].mean()
    features['temp_max']      = df['Temperature_measured'].max()
    features['temp_rise']     = df['Temperature_measured'].iloc[-1] - df['Temperature_measured'].iloc[0]

    # --- Current features ---
    features['current_mean']  = df['Current_measured'].mean()
    features['current_std']   = df['Current_measured'].std()

    # --- Discharge duration (shorter = degraded battery dying faster) ---
    features['discharge_time'] = df['Time'].iloc[-1] - df['Time'].iloc[0]

    # --- Internal resistance proxy: delta_V / delta_I ---
    delta_v = df['Voltage_measured'].diff().dropna()
    delta_i = df['Current_measured'].diff().dropna()
    with np.errstate(divide='ignore', invalid='ignore'):
        ir = np.where(np.abs(delta_i) > 0.01, delta_v / delta_i, np.nan)
    features['internal_resistance_proxy'] = np.nanmedian(ir)

    return features


def build_dataset(battery_ids):
    """
    Loads all discharge cycles for a list of batteries,
    extracts features per cycle, returns a clean DataFrame.
    """
    meta = pd.read_csv(f"{DATA_PATH}/metadata.csv")
    meta['Capacity'] = pd.to_numeric(meta['Capacity'], errors='coerce')

    discharge = meta[(meta['type'] == 'discharge') & 
                     (meta['battery_id'].isin(battery_ids)) &
                     (meta['Capacity'] > 0.5)].copy()  # filter dead/bad cycles

    discharge = discharge.sort_values(['battery_id', 'start_time']).reset_index(drop=True)

    rows = []
    for _, row in discharge.iterrows():
        filepath = f"{DATA_PATH}/data/{row['filename']}"
        feats = extract_cycle_features(filepath)
        if feats is None:
            continue
        feats['battery_id']   = row['battery_id']
        feats['capacity']     = row['Capacity']
        feats['cycle_number'] = len([r for r in rows if r['battery_id'] == row['battery_id']]) + 1
        rows.append(feats)

    df = pd.DataFrame(rows)
    return df


# --- Run it ---
print("Building training dataset (B0033, B0034, B0036)...")
train_df = build_dataset(['B0033', 'B0034', 'B0036'])
print(f"Train shape: {train_df.shape}")
print(train_df.head())

print("\nBuilding test dataset (B0005, B0006, B0007)...")
test_df = build_dataset(['B0005', 'B0006', 'B0007'])
print(f"Test shape: {test_df.shape}")

# Save to CSV
train_df.to_csv("train_features.csv", index=False)
test_df.to_csv("test_features.csv", index=False)
print("\n✅ Saved train_features.csv and test_features.csv")

# Quick sanity check — does capacity actually decline?
import matplotlib
matplotlib.use('Agg')  # no display needed
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, bid in zip(axes, ['B0033', 'B0034', 'B0036']):
    subset = train_df[train_df['battery_id'] == bid]
    ax.plot(subset['cycle_number'], subset['capacity'])
    ax.axhline(subset['capacity'].max() * 0.8, color='red', linestyle='--', label='80% EOL')
    ax.set_title(f'Battery {bid}')
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Capacity (Ah)')
    ax.legend()

plt.tight_layout()
plt.savefig("capacity_curves.png")
print("✅ Saved capacity_curves.png")
