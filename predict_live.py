import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load model
model = load_model("models/lstm_model.h5")

# Load real NASA rainfall data
df = pd.read_csv("data/processed/live_input.csv", parse_dates=["date"])
df = df.sort_values("date")

# Scale rainfall for model input
import joblib
scaler = joblib.load("models/scaler.pkl")
scaled = scaler.transform(df[["rainfall_mm"]])


# Use last 30 days as input
last_30 = scaled[-30:].reshape(1, 30, 1)

# Predict next 7 days
predictions_scaled = []

for _ in range(7):
    next_day = model.predict(last_30)[0][0]
    predictions_scaled.append(next_day)

    # Slide window forward
    last_30 = np.append(last_30[:, 1:, :], [[[next_day]]], axis=1)

# Inverse transform to get rainfall in mm
predictions_scaled = np.array(predictions_scaled).reshape(-1, 1)
predictions_scaled = np.clip(predictions_scaled, 0, 1)
predictions_mm = scaler.inverse_transform(predictions_scaled).flatten()

# Create forecast DataFrame
start_date = df["date"].iloc[-1] + pd.Timedelta(days=1)
future_dates = pd.date_range(start=start_date, periods=7)

forecast_df = pd.DataFrame({
    "date": future_dates,
    "predicted_rainfall_mm": np.round(predictions_mm, 2)
})

forecast_df.to_csv("outputs/predictions_next_7_days_on_NASA_data.csv", index=False)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(forecast_df["date"], forecast_df["predicted_rainfall_mm"], marker='o')
plt.title("Predicted Rainfall for Next 7 Days (LSTM)")
plt.xlabel("Date")
plt.ylabel("Rainfall (mm)")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/plots/rainfall_forecast_next_7_days_on_NASA_data.png")
plt.show()

print("✅ Forecast saved and plotted.")
