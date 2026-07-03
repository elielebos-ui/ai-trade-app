import os
import glob
import numpy as np
import pandas as pd

print("⚙️ Starting the AI Trade App Multi-Asset Feature Engineering...")

# Automatically find all files we just downloaded
files = glob.glob("*_historical_data.csv")

if not files:
    print("❌ Error: No historical data files found!")
    exit()

for file_name in files:
    print(f"\n🔄 Processing {file_name}...")
    df = pd.read_csv(file_name, header=[0, 1], index_col=0, parse_dates=True)
    df.columns = df.columns.get_level_values(0)
    
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility_20d"] = df["Daily_Return"].rolling(window=20).std() * np.sqrt(252)

    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI_14"] = 100 - (100 / (1 + rs))

    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    df_clean = df.dropna().copy()
    output_name = file_name.replace('historical_data', 'features')
    df_clean.to_csv(output_name)

    print(f"✅ Features generated and saved as: {output_name}")

print("\n🎯 Multi-Asset feature engineering complete!")