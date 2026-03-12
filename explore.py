import pandas as pd

DATA_PATH = "/Users/gabrieldiaz/.cache/kagglehub/datasets/patrickfleith/nasa-battery-dataset/versions/2/cleaned_dataset"

meta = pd.read_csv(f"{DATA_PATH}/metadata.csv")

# Filter to discharge cycles only
discharge = meta[meta['type'] == 'discharge'].copy()

print("=== DISCHARGE METADATA ===")
print(discharge[['battery_id', 'filename', 'Capacity', 'ambient_temperature']].head(20))

print("\n=== HOW MANY CYCLES PER BATTERY? ===")
cycles_per_battery = discharge.groupby('battery_id').size().sort_values(ascending=False)
print(cycles_per_battery.head(20))

print("\n=== BATTERIES WITH MOST CYCLES (best for training) ===")
top_batteries = cycles_per_battery[cycles_per_battery >= 100]
print(f"Batteries with 100+ cycles: {len(top_batteries)}")
print(top_batteries)

# Peek inside one actual data file
print("\n=== SAMPLE DATA FILE COLUMNS ===")
sample_file = discharge.iloc[0]['filename']
print(f"Loading file: {sample_file}")
df_sample = pd.read_csv(f"{DATA_PATH}/data/{sample_file}")
print(df_sample.columns.tolist())
print(df_sample.head())
print(f"\nShape: {df_sample.shape}")
