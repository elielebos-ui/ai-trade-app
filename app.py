import os
import json
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="AI Trade App Master Dashboard", page_icon="🤖", layout="wide")

st.title("🦅 AI Trade App: Multi-Asset Market Intelligence")
st.markdown("An autonomous system executing price action feature engineering, machine learning predictions, and live paper execution.")
st.divider()

ASSETS = {
    "Gold Futures": "gold_features.csv",
    "Crude Oil Futures": "crude_oil_features.csv",
    "Bitcoin (Crypto)": "bitcoin_features.csv",
    "Apple (Tech)": "apple_features.csv",
    "NVIDIA (AI)": "nvidia_features.csv"
}

st.sidebar.header("Navigation Panel")
selected_asset = st.sidebar.selectbox("Choose Asset to Analyze", list(ASSETS.keys()))
target_file = ASSETS[selected_asset]

if not os.path.exists(target_file):
    st.error(f"❌ Missing data for {selected_asset}. Please run your data pipeline first!")
else:
    df = pd.read_csv(target_file, index_col=0, parse_dates=True)

    # --- MACHINE LEARNING & BACKTEST ENGINE ---
    df["Future_Close"] = df["Close"].shift(-5)
    df["Target"] = (df["Future_Close"] > df["Close"]).astype(int)
    
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].dropna().copy()
    df_test = df.iloc[split_idx:].copy()
    
    features_list = ["SMA_20", "SMA_50", "Volatility_20d", "RSI_14", "MACD", "MACD_Signal"]
    
    ai_model = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_split=10, random_state=42)
    ai_model.fit(df_train[features_list], df_train["Target"])

    X_live = df[features_list].iloc[[-1]]
    live_prediction = ai_model.predict(X_live)[0]
    live_confidence = ai_model.predict_proba(X_live)[0]

    df_test["AI_Prediction"] = ai_model.predict(df_test[features_list])
    df_test["Market_Return"] = df_test["Close"].pct_change()
    df_test["Strategy_Return"] = df_test["AI_Prediction"].shift(1) * df_test["Market_Return"]
    df_test = df_test.dropna()

    starting_capital = 10000
    df_test["Market Growth ($)"] = (1 + df_test["Market_Return"]).cumprod() * starting_capital
    df_test["AI Strategy Growth ($)"] = (1 + df_test["Strategy_Return"]).cumprod() * starting_capital

    winning_days = len(df_test[df_test["Strategy_Return"] > 0])
    losing_days = len(df_test[df_test["Strategy_Return"] < 0])
    total_active_days = winning_days + losing_days
    win_rate = (winning_days / total_active_days * 100) if total_active_days > 0 else 0

    df_test["Peak_Wallet"] = df_test["AI Strategy Growth ($)"].cummax()
    df_test["Drawdown_Pct"] = (df_test["AI Strategy Growth ($)"] - df_test["Peak_Wallet"]) / df_test["Peak_Wallet"]
    max_drawdown = df_test["Drawdown_Pct"].min() * 100

    # --- MAIN INTERFACE ---
    st.subheader(f"📊 Live Performance Metrics: {selected_asset}")
    
    latest_row = df.iloc[-1]
    current_price = float(latest_row["Close"])
    prev_price = float(df.iloc[-2]["Close"])
    price_change = current_price - prev_price
    
    if live_prediction == 1:
        ai_status_text = "🟩 BUY / LONG"
        ai_delta_text = f"Confidence: {live_confidence[1]*100:.1f}%"
    else:
        ai_status_text = "🟥 SELL / SHORT"
        ai_delta_text = f"Confidence: {live_confidence[0]*100:.1f}%"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric(label="Current Price", value=f"${current_price:,.2f}", delta=f"${price_change:,.2f}")
    col2.metric(label="Backtest Historical Profit", value=f"${df_test['AI Strategy Growth ($)'].iloc[-1]:,.2f}", delta=f"{win_rate:.1f}% Win Rate")
    col3.metric(label="Max Backtest Drawdown", value=f"{max_drawdown:.2f}%")
    col4.metric(label="Live AI Recommendation", value=ai_status_text, delta=ai_delta_text)

    st.divider()

    # --- LIVE PAPER TRADING PORTFOLIO MONITOR ---
    st.subheader("💼 Active Forward-Testing Live Paper Account (Crude Oil Focused)")
    portfolio_file = "paper_portfolio.json"
    
    if os.path.exists(portfolio_file):
        with open(portfolio_file, "r") as f:
            port = json.load(f)
        
        # Calculate current valuation of live holdings based on current file price
        if port["position"] == "LONG":
            live_val = port["shares"] * current_price
            display_pos = f"🟢 LONG ({port['shares']:.4f} Units)"
        else:
            live_val = port["cash"]
            display_pos = "⚪ CASH (Flat Market)"
            
        p_col1, p_col2, p_col3 = st.columns(3)
        p_col1.metric(label="Live Account Value", value=f"${live_val:,.2f}", delta=f"${live_val - 10000.0:,.2f} Total Return")
        p_col2.metric(label="Current Allocation Status", value=display_pos)
        p_col3.metric(label="Initial Account Core Deposit", value="$10,000.00")
    else:
        st.info("💡 Live Paper account will initialize and display data immediately after the first night's 1:30 AM automation loop runs!")

    st.divider()

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader("📉 Technical Price Chart")
        chart_data = df[["Close", "SMA_20", "SMA_50"]].tail(150)
        st.line_chart(chart_data)
    with right_col:
        st.subheader("💰 Historical Backtest Simulator")
        backtest_chart_data = df_test[["AI Strategy Growth ($)", "Market Growth ($)"]]
        st.line_chart(backtest_chart_data)