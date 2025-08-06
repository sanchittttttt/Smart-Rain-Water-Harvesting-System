import subprocess

print("\n🌧️  Smart Rainwater Harvesting System: Running Full Pipeline")

# Step 1: Fetch NASA rainfall data
print("\n📦 Step 1: Fetching real rainfall data from NASA...")
subprocess.run(["python", "src/fetch_live_data_nasa.py"], check=True)

# Step 2: Predict next 7 days with trained LSTM
print("\n🔮 Step 2: Predicting next 7 days of rainfall with LSTM...")
subprocess.run(["python", "src/predict_live.py"], check=True)

# Done
print("\n✅ All steps completed.")
print("📄 See: outputs/predictions_next_7_days_on_NASA_data.csv")
print("🖼️  See: outputs/plots/rainfall_forecast_next_7_days_on_NASA_data.png")
