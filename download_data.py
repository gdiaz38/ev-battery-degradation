import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import os

print("Downloading NASA Battery Dataset...")

# List files in the dataset first
path = kagglehub.dataset_download("patrickfleith/nasa-battery-dataset")
print(f"Dataset downloaded to: {path}")

# Show all files
for root, dirs, files in os.walk(path):
    for file in files:
        filepath = os.path.join(root, file)
        size = os.path.getsize(filepath) / 1024
        print(f"  {filepath} ({size:.1f} KB)")
