# 📘 LSTM Rainfall Forecasting Module

This module handles the end-to-end process of preparing data, training an LSTM model on historical rainfall records, evaluating its performance, and generating rainfall forecasts for the next 7 days. It uses daily rainfall data from NASA POWER and is designed to work independently or as part of a larger smart rainwater harvesting system.

---

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/smart-rainwater-harvesting.git
cd smart-rainwater-harvesting
```

### 2. Create and Activate a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

## 📂 File Structure (LSTM Module Only)
```
src/
  preprocessing.py
  lstm_model.py
  predict_live.py
  fetch_live_data_nasa.py
models/
  lstm_model.h5
  scaler.pkl
outputs/
  predictions_next_7_days_on_NASA_data.csv
  plots/
    rainfall_forecast_next_7_days_on_NASA_data.png
requirements.txt
README_LSTM.md
```

---

## 🛠️ Step-by-Step Usage

### Step 1: Fetch the Latest Rainfall Data
Uses NASA POWER API to download the last 45 days of rainfall for the target location.
```bash
python src/fetch_live_data_nasa.py
```
Creates: `data/processed/live_input.csv`

### Step 2: Train the LSTM Model
Train a deep learning model on historical rainfall using 30-day input windows.
```bash
python src/lstm_model.py
```
Creates: `models/lstm_model.h5`, `models/scaler.pkl`
Model performance (MAE, RMSE, R²) is printed and training/validation loss is plotted.

### Step 3: Predict Rainfall for the Next 7 Days
Uses the most recent 30 days of NASA rainfall data as input and generates a 7-day rainfall forecast.
```bash
python src/predict_live.py
```
Creates:
- `outputs/predictions_next_7_days_on_NASA_data.csv`
- `outputs/plots/rainfall_forecast_next_7_days_on_NASA_data.png`

---

## 📊 Example Output Format

**CSV preview (`outputs/predictions_next_7_days_on_NASA_data.csv`):**
```
date,predicted_rainfall_mm
2025-08-06,9.84
2025-08-07,10.12
2025-08-08,11.07
2025-08-09,9.93
2025-08-10,8.64
2025-08-11,8.21
2025-08-12,7.93
```

**Plot:**
Line graph showing forecasted daily rainfall in mm over the next 7 days.

---

## 📌 Notes
- NASA POWER does not provide current-day data. The most recent data is always up to yesterday.
- Ensure `models/lstm_model.h5` and `models/scaler.pkl` are present before running predictions.
- Input to the model is a 30-day rolling window of rainfall (mm/day), scaled using MinMaxScaler.
- Forecast is generated autoregressively for 7 days.

---

## 🧪 Evaluation Metrics
These metrics are printed automatically during model evaluation:
- **MAE (Mean Absolute Error):** Average prediction error in mm
- **RMSE (Root Mean Squared Error):** Penalizes large deviations
- **R² (Explained Variance Score):** How well the model explains target variance

---

## 📬 Output Integration
The 7-day forecasted values are designed to be consumed by:
- Tank inflow simulation
- Dashboard and visualization tools
- Optimization engines (e.g., genetic algorithms)

No further dependencies or external API keys are required after initial setup.

