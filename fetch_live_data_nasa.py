import pandas as pd
import requests
from datetime import datetime, timedelta

# Location coordinates for Pune
lat = 18.54
lon = 73.85

# Date range: last 30 days up to yesterday (NASA doesn't return today)
end_date = datetime.utcnow().date() - timedelta(days=1)
start_date = end_date - timedelta(days=45)

start_str = start_date.strftime("%Y%m%d")
end_str = end_date.strftime("%Y%m%d")

print(f"📅 Fetching NASA POWER rainfall from {start_str} to {end_str}...")

# NASA POWER endpoint
url = (
    f"https://power.larc.nasa.gov/api/temporal/daily/point"
    f"?start={start_str}&end={end_str}"
    f"&latitude={lat}&longitude={lon}"
    f"&community=ag&parameters=PRECTOTCORR"
    f"&format=JSON"
)

response = requests.get(url)
if response.status_code != 200:
    raise Exception(f"❌ NASA API request failed: {response.status_code} {response.text}")

data = response.json()

# Extract dates and rainfall in mm/day
daily_data = data["properties"]["parameter"]["PRECTOTCORR"]
dates = list(daily_data.keys())
rainfall_mm = list(daily_data.values())

df = pd.DataFrame({
    "date": pd.to_datetime(dates),
    "rainfall_mm": rainfall_mm
})

# Clean up: remove any rows with -999.0 (indicating no data)
df = df[df["rainfall_mm"] != -999.0]


# Save
df.to_csv("data/processed/live_input.csv", index=False)
print("✅ Saved NASA rainfall to data/processed/live_input.csv")
