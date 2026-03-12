import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

# ── Config ──────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 10   # look back 10 cycles to predict next capacity
FEATURE_COLS = [
    'voltage_mean', 'voltage_min', 'voltage_std', 'voltage_drop', 'voltage_slope',
    'temp_mean', 'temp_max', 'temp_rise',
    'current_mean', 'current_std',
    'discharge_time', 'internal_resistance_proxy'
]
TARGET_COL = 'capacity'
EPOCHS     = 80
LR         = 0.001
HIDDEN     = 64
LAYERS     = 2
BATCH      = 32

# ── Dataset ──────────────────────────────────────────────────────────────────
class BatteryDataset(Dataset):
    def __init__(self, df, scaler_X=None, scaler_y=None, fit=False):
        df = df.copy()
        df[FEATURE_COLS] = df[FEATURE_COLS].fillna(df[FEATURE_COLS].median())

        X = df[FEATURE_COLS].values.astype(np.float32)
        y = df[TARGET_COL].values.astype(np.float32)

        if fit:
            self.scaler_X = MinMaxScaler()
            self.scaler_y = MinMaxScaler()
            X = self.scaler_X.fit_transform(X)
            y = self.scaler_y.fit_transform(y.reshape(-1,1)).flatten()
        else:
            self.scaler_X = scaler_X
            self.scaler_y = scaler_y
            X = scaler_X.transform(X)
            y = scaler_y.transform(y.reshape(-1,1)).flatten()

        # Build sequences — one sample = 10 consecutive cycles → predict next capacity
        self.sequences, self.targets = [], []
        for battery in df['battery_id'].unique():
            mask = df['battery_id'] == battery
            Xb = X[mask]
            yb = y[mask]
            for i in range(len(Xb) - SEQUENCE_LENGTH):
                self.sequences.append(Xb[i:i+SEQUENCE_LENGTH])
                self.targets.append(yb[i+SEQUENCE_LENGTH])

        self.sequences = torch.tensor(np.array(self.sequences), dtype=torch.float32)
        self.targets   = torch.tensor(np.array(self.targets),   dtype=torch.float32)

    def __len__(self):  return len(self.targets)
    def __getitem__(self, i): return self.sequences[i], self.targets[i]


# ── Model ────────────────────────────────────────────────────────────────────
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


# ── Train ────────────────────────────────────────────────────────────────────
print("Loading data...")
train_df = pd.read_csv("train_features.csv")
test_df  = pd.read_csv("test_features.csv")

train_ds = BatteryDataset(train_df, fit=True)
test_ds  = BatteryDataset(test_df, scaler_X=train_ds.scaler_X, scaler_y=train_ds.scaler_y)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

model     = BatteryLSTM()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Training on {len(train_ds)} sequences, testing on {len(test_ds)} sequences\n")

train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    # Train
    model.train()
    epoch_loss = 0
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Validate
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Xb, yb in test_loader:
            pred = model(Xb)
            val_loss += criterion(pred, yb).item()

    train_losses.append(epoch_loss / len(train_loader))
    val_losses.append(val_loss / len(test_loader))

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | Train Loss: {train_losses[-1]:.5f} | Val Loss: {val_losses[-1]:.5f}")

# ── Evaluate ─────────────────────────────────────────────────────────────────
print("\n=== Final Evaluation on Unseen Batteries (B0005, B0006, B0007) ===")
model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for Xb, yb in test_loader:
        preds = model(Xb)
        all_preds.extend(train_ds.scaler_y.inverse_transform(
            preds.numpy().reshape(-1,1)).flatten())
        all_true.extend(train_ds.scaler_y.inverse_transform(
            yb.numpy().reshape(-1,1)).flatten())

mae  = mean_absolute_error(all_true, all_preds)
rmse = np.sqrt(mean_squared_error(all_true, all_preds))
mape = np.mean(np.abs((np.array(all_true) - np.array(all_preds)) / np.array(all_true))) * 100

print(f"MAE  : {mae:.4f} Ah")
print(f"RMSE : {rmse:.4f} Ah")
print(f"MAPE : {mape:.2f}%")

# ── Save model & scalers ─────────────────────────────────────────────────────
torch.save(model.state_dict(), "battery_lstm.pt")
joblib.dump(train_ds.scaler_X, "scaler_X.pkl")
joblib.dump(train_ds.scaler_y, "scaler_y.pkl")
print("\n✅ Saved battery_lstm.pt, scaler_X.pkl, scaler_y.pkl")

# ── Plot loss curves ──────────────────────────────────────────────────────────
plt.figure(figsize=(10,4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses,   label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('LSTM Training — Battery Degradation')
plt.legend()
plt.tight_layout()
plt.savefig("training_curves.png")
print("✅ Saved training_curves.png")
