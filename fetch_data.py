import os
import yfinance as yf

print("🚀 Starting the AI Trade App Data Ingestion System (Multi-Asset Upgrade)...")

# We can now add ANY asset in the world right here!
assets = {
    "gold": "GC=F", 
    "crude_oil": "CL=F",
    "bitcoin": "BTC-USD",
    "apple": "AAPL",
    "nvidia": "NVDA"
}

for name, ticker_symbol in assets.items():
    print(f"\n🔄 Downloading 10 years of {name.upper()} data...")
    
    market_data = yf.download(ticker_symbol, period="10y")

    if not market_data.empty:
        filename = f"{name}_historical_data.csv"
        market_data.to_csv(filename)
        print(f"✅ Success! Saved {len(market_data)} days of history for {name.upper()}.")
    else:
        print(f"❌ Failed to download data for {name}.")

print("\n🎯 Multi-Asset data collection complete!")