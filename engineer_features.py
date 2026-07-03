import os
import yfinance as yf

print("🚀 Starting the AI Trade App Data Ingestion System (Macro Upgrade)...")

assets = {
    "gold": "GC=F", 
    "crude_oil": "CL=F",
    "bitcoin": "BTC-USD",
    "apple": "AAPL",
    "nvidia": "NVDA"
}

macro_indicators = {
    "us_dollar": "DX-Y.NYB",   
    "sp500": "^GSPC"           
}

print("\n🌍 Downloading Global Macro Market Overlays...")
for name, ticker_symbol in macro_indicators.items():
    data = yf.download(ticker_symbol, period="10y")
    if not data.empty:
        data.to_csv(f"{name}_macro.csv")
        print(f"✅ Saved macro indicator: {name.upper()}")

for name, ticker_symbol in assets.items():
    print(f"\n🔄 Downloading 10 years of {name.upper()} data...")
    market_data = yf.download(ticker_symbol, period="10y")

    if not market_data.empty:
        filename = f"{name}_historical_data.csv"
        market_data.to_csv(filename)
        print(f"✅ Success! Saved {len(market_data)} days of history for {name.upper()}.")
    else:
        print(f"❌ Failed to download data for {name}.")

print("\n🎯 Macro-Informed Data Collection Complete!")
