import os
import glob
import numpy as np
import pandas as pd

print("⚙️ Starting the AI Trade App Multi-Asset Macro Feature Engineering...")

# Helper function to load clean macro files
def load_macro(file_path, prefix):
    if os.path.exists(file_path):
        m_df = pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)
        m_df.columns = m_df.columns.get_level_values(0)
        return pd.to_numeric(m_df["Close"], errors='coerce').to_frame(name=f"{prefix}_Close")
    return None

# Load the macro dataframes
usd_df = load_macro("us_dollar_macro.csv", "USD")
sp_df = load_macro("sp500_macro.csv", "SP500")

# Find all primary asset historical data files
files = glob.glob("*_historical_data.csv")

for file_name in files:
    print(f"\n🔄 Weaving macro clues into {file_name}...")
    df = pd.read_csv(file_name, header=[0, 1], index_col=0, parse_dates=True)
    df.columns = df.columns.get_level_values(0)
    
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Core Technical Indicators
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

    # --- ADVANCED BRAIN UPGRADE: MERGE INTER-MARKET MACRO DATA ---
    if usd_df is not None:
        df = df.join(usd_df, how="left")
        # Forward-fill gaps if forex markets have different holiday schedules
        df["USD_Close"] = df["USD_Close"].ffill()
        df["USD_Return_5d"] = df["USD_Close"].pct_change(periods=5) # 5-day directional momentum of US dollar

    if sp_df is not None:
        df = df.join(sp_df, how="left")
        df["SP500_Close"] = df["SP500_Close"].ffill()
        df["SP500_Return_5d"] = df["SP500_Close"].pct_change(periods=5) # 5-day directional momentum of stocks

    df_clean = df.dropna().copy()
    output_name = file_name.replace('historical_data', 'features')
    df_clean.to_csv(output_name)
    print(f"✅ Enhanced feature matrix saved: {output_name}")

print("\n🎯 Macro feature engineering complete!")