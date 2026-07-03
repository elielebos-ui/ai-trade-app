import os
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

print("🤖 Launching Macro-Enhanced AI Live Paper Trading Desk...")

asset_name = "crude_oil"
feature_file = f"{asset_name}_features.csv"

if not os.path.exists(feature_file):
    print(f"❌ Error: {feature_file} not found. Please run your pipeline first.")
    exit()

df = pd.read_csv(feature_file, index_col=0, parse_dates=True)

df["Future_Close"] = df["Close"].shift(-5)
df["Target"] = (df["Future_Close"] > df["Close"]).astype(int)
df_train = df.dropna().copy()

# UPGRADED: Synced with macro clues
features_list = ["SMA_20", "SMA_50", "Volatility_20d", "RSI_14", "MACD", "MACD_Signal", "USD_Return_5d", "SP500_Return_5d"]

ai_model = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=10, random_state=42)
ai_model.fit(df_train[features_list], df_train["Target"])

latest_row = df.iloc[-1]
X_live = df[features_list].iloc[[-1]]
prediction = int(ai_model.predict(X_live)[0])
current_price = float(latest_row["Close"])
current_date = str(df.index[-1].strftime('%Y-%m-%d'))

portfolio_file = "paper_portfolio.json"

if os.path.exists(portfolio_file):
    with open(portfolio_file, "r") as f:
        portfolio = json.load(f)
else:
    portfolio = {
        "cash": 10000.0,
        "shares": 0.0,
        "position": "CASH",
        "last_price": current_price,
        "history": []
    }

print(f"📂 Ledger Loaded. Current Position: Status={portfolio['position']} | Cash=${portfolio['cash']:,.2f}")

if portfolio["position"] == "LONG":
    current_valuation = portfolio["shares"] * current_price
    print(f"📊 Live Open Long Position Valuation: ${current_valuation:,.2f}")
else:
    current_valuation = portfolio["cash"]
    print(f"📊 Live Cash Balance Valuation: ${current_valuation:,.2f}")

if prediction == 1 and portfolio["position"] == "CASH":
    portfolio["shares"] = portfolio["cash"] / current_price
    portfolio["cash"] = 0.0
    portfolio["position"] = "LONG"
    portfolio["last_price"] = current_price
    portfolio["history"].append({"date": current_date, "action": "BUY", "price": current_price, "total_value": current_valuation})
    print(f"🟩 LIVE ORDER EXECUTED: AI bought shares at ${current_price:,.2f}")

elif prediction == 0 and portfolio["position"] == "LONG":
    portfolio["cash"] = portfolio["shares"] * current_price
    portfolio["shares"] = 0.0
    portfolio["position"] = "CASH"
    portfolio["last_price"] = current_price
    portfolio["history"].append({"date": current_date, "action": "SELL", "price": current_price, "total_value": portfolio["cash"]})
    print(f"🟥 LIVE ORDER EXECUTED: AI liquidated position at ${current_price:,.2f}")

else:
    print("💤 ALGORITHM HOLD: Market conditions dictate maintaining current allocations. No trade required.")

with open(portfolio_file, "w") as f:
    json.dump(portfolio, f, indent=4)

print("✅ Live Paper Trading desk execution checks complete.")