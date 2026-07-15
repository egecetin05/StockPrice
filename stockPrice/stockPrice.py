import time
import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# ==========================================
# STEP 1: DATA COLLECTION
# ==========================================
print("Fetching Apple (AAPL) stock data...")
df_apple = yf.download("AAPL", start="2021-01-01", end="2026-01-01")

# ==========================================
# STEP 2: DATA PREPROCESSING
# ==========================================
print("Preprocessing data and creating sliding windows...")
# Use 5 features: Open, High, Low, Close, Volume
features = df_apple[['Open', 'High', 'Low', 'Close', 'Volume']].values

# Scale data to (0, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(features)

# 'Close' is at index 3
target_index = 3
window_size = 60

X, y = [], []
for i in range(len(scaled_data) - window_size):
    X.append(scaled_data[i : i + window_size])
    y.append(scaled_data[i + window_size, target_index])

X = np.array(X)
y = np.array(y).reshape(-1, 1)

# Chronological Train-Test Split (80% Train, 20% Test)
train_size = int(len(X) * 0.80)
X_train = torch.tensor(X[:train_size], dtype=torch.float32)
y_train = torch.tensor(y[:train_size], dtype=torch.float32)
X_test = torch.tensor(X[train_size:], dtype=torch.float32)
y_test = torch.tensor(y[train_size:], dtype=torch.float32)

# ==========================================
# STEP 3: MODEL DEFINITIONS
# ==========================================
class StockPredictionLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=100, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1, :])
        return predictions

class StockPredictionGRU(nn.Module):
    def __init__(self, input_size=5, hidden_size=100, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        predictions = self.linear(gru_out[:, -1, :])
        return predictions

# ==========================================
# STEP 4: MODEL TRAINING
# ==========================================
num_epochs = 200
criterion = nn.MSELoss()

# --- Train LSTM ---
print("\n--- Training LSTM Model ---")
model_lstm = StockPredictionLSTM()
optimizer_lstm = optim.Adam(model_lstm.parameters(), lr=0.001)

start_time_lstm = time.time()
for epoch in range(num_epochs):
    optimizer_lstm.zero_grad()
    loss = criterion(model_lstm(X_train), y_train)
    loss.backward()
    optimizer_lstm.step()
    if (epoch + 1) % 50 == 0:
        print(f"LSTM Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.5f}")
print(f"LSTM Training Time: {time.time() - start_time_lstm:.2f} seconds")

# --- Train GRU ---
print("\n--- Training GRU Model ---")
model_gru = StockPredictionGRU()
optimizer_gru = optim.Adam(model_gru.parameters(), lr=0.001)

start_time_gru = time.time()
for epoch in range(num_epochs):
    optimizer_gru.zero_grad()
    loss = criterion(model_gru(X_train), y_train)
    loss.backward()
    optimizer_gru.step()
    if (epoch + 1) % 50 == 0:
        print(f"GRU Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.5f}")
print(f"GRU Training Time: {time.time() - start_time_gru:.2f} seconds")

# ==========================================
# STEP 5: EVALUATION & RESULTS
# ==========================================
print("\nEvaluating models and calculating RMSE...")

# We need a scaler specifically for the 'Close' prices to inverse transform predictions
scaler_close = MinMaxScaler(feature_range=(0, 1))
scaler_close.fit(df_apple[['Close']].values)

# Evaluate LSTM
model_lstm.eval()
with torch.no_grad():
    lstm_predictions = model_lstm(X_test)
lstm_pred_prices = scaler_close.inverse_transform(lstm_predictions.numpy())

# Evaluate GRU
model_gru.eval()
with torch.no_grad():
    gru_predictions = model_gru(X_test)
gru_pred_prices = scaler_close.inverse_transform(gru_predictions.numpy())

# Actual Prices
actual_prices = scaler_close.inverse_transform(y_test.numpy())

# RMSE Calculation
rmse_lstm = np.sqrt(mean_squared_error(actual_prices, lstm_pred_prices))
rmse_gru = np.sqrt(mean_squared_error(actual_prices, gru_pred_prices))

print("\n==========================================")
print("             FINAL RESULTS                ")
print("==========================================")
print(f"LSTM Model Test RMSE: ${rmse_lstm:.2f}")
print(f"GRU Model Test RMSE:  ${rmse_gru:.2f}")
print("==========================================")