import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil

# -----------------------------
# CONFIGURATION
# -----------------------------
START_DATE = "2019-01-01"
END_DATE = "2025-07-01"
REBALANCE = True  # Toggle quarterly rebalancing
ROLLING_WINDOW = 126  # ~6 months for rolling Sharpe

# Portfolio tickers
stocks = [
    "ROK", "EMR", "HON", "MSFT", "NVDA", "PLTR", "CRWD",
    "CGNX", "AMAT", "SNOW", "SSYS", "DDD"
]
etfs = ["ROBO", "SOXX", "ESGU", "ICLN"]
portfolio = stocks + etfs

# Portfolio weights
weights = {
    "ROK": 0.05, "EMR": 0.04, "HON": 0.04, "MSFT": 0.07, "NVDA": 0.07,
    "PLTR": 0.04, "CRWD": 0.04, "CGNX": 0.03, "AMAT": 0.04, "SNOW": 0.03,
    "SSYS": 0.03, "DDD": 0.02, "ROBO": 0.15, "SOXX": 0.10,
    "ESGU": 0.10, "ICLN": 0.05
}
weights = {k: v / sum(weights.values()) for k, v in weights.items()}

# -----------------------------
# DATA DOWNLOAD
# -----------------------------
data = yf.download(portfolio, start=START_DATE, end=END_DATE, auto_adjust=False)

# Ensure correct price column
if "Adj Close" in data.columns:
    data = data["Adj Close"]
elif "Close" in data.columns:
    data = data["Close"]
else:
    raise ValueError("No 'Adj Close' or 'Close' found in downloaded data")

# Drop missing tickers and adjust weights
data = data.dropna(axis=1)
weights = {k: v for k, v in weights.items() if k in data.columns}

if not weights:
    raise ValueError("No valid tickers found with available data!")

weight_array = np.array(list(weights.values()))
print("Using tickers:", list(weights.keys()))

# -----------------------------
# CALCULATE RETURNS
# -----------------------------
daily_returns = data.pct_change().dropna()

if REBALANCE:
    rebalance_dates = pd.date_range(start=daily_returns.index[0], end=daily_returns.index[-1], freq='QE')
    portfolio_returns = pd.Series(dtype=float)

    for i in range(len(rebalance_dates)):
        start_date = rebalance_dates[i]
        end_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else daily_returns.index[-1]
        sub_returns = daily_returns.loc[start_date:end_date].dot(weight_array)
        portfolio_returns = pd.concat([portfolio_returns, sub_returns])
else:
    portfolio_returns = daily_returns.dot(weight_array)

cumulative_returns = (1 + portfolio_returns).cumprod()

# -----------------------------
# BENCHMARKS
# -----------------------------
benchmarks = yf.download(["QQQ", "SPY", "ESGU"], start=START_DATE, end=END_DATE, auto_adjust=False)
if "Adj Close" in benchmarks.columns:
    benchmarks = benchmarks["Adj Close"]
else:
    benchmarks = benchmarks["Close"]

benchmarks = benchmarks.pct_change().dropna()
benchmarks_cum = (1 + benchmarks).cumprod()

# -----------------------------
# METRICS
# -----------------------------
cagr = (cumulative_returns.iloc[-1]) ** (252 / len(portfolio_returns)) - 1
volatility = portfolio_returns.std() * np.sqrt(252)
sharpe = cagr / volatility
max_drawdown = ((cumulative_returns / cumulative_returns.cummax()) - 1).min()

metrics = pd.DataFrame({
    "CAGR": [f"{cagr*100:.2f}%"],
    "Volatility": [f"{volatility*100:.2f}%"],
    "Sharpe Ratio": [f"{sharpe:.2f}"],
    "Max Drawdown": [f"{max_drawdown*100:.2f}%"]
})

# -----------------------------
# SAVE OUTPUTS (Excel + Charts)
# -----------------------------
os.makedirs("outputs/charts", exist_ok=True)
metrics.to_excel("outputs/portfolio_metrics.xlsx", index=False)

# Cumulative returns chart
plt.figure(figsize=(10, 6))
plt.plot(cumulative_returns, label="ESG Automation Portfolio")
for col in benchmarks_cum.columns:
    plt.plot(benchmarks_cum[col], label=col)
plt.legend()
plt.title("Cumulative Returns: ESG Automation vs Benchmarks (2019-2025)")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/charts/cumulative_returns.png")
plt.close()

# Rolling Sharpe chart
rolling_sharpe = (
    (portfolio_returns.rolling(ROLLING_WINDOW).mean() * 252) /
    (portfolio_returns.rolling(ROLLING_WINDOW).std() * np.sqrt(252))
)
plt.figure(figsize=(10, 5))
plt.plot(rolling_sharpe, label="Rolling Sharpe (6 months)")
plt.axhline(1, color="gray", linestyle="--", linewidth=1)
plt.legend()
plt.title("Rolling Sharpe Ratio")
plt.grid(True)
plt.savefig("outputs/charts/rolling_sharpe.png")
plt.close()

# Drawdown chart
drawdown = (cumulative_returns / cumulative_returns.cummax()) - 1
plt.figure(figsize=(10, 5))
plt.plot(drawdown, label="Drawdown", color="red")
plt.title("Portfolio Drawdown")
plt.grid(True)
plt.savefig("outputs/charts/drawdown.png")
plt.close()

print("Backtest complete. Results saved in outputs/ folder.")

# -----------------------------
# GENERATE GITHUB PAGES (HTML)
# -----------------------------
os.makedirs("docs/charts", exist_ok=True)

# Copy charts to docs
shutil.copy("outputs/charts/cumulative_returns.png", "docs/charts/")
shutil.copy("outputs/charts/rolling_sharpe.png", "docs/charts/")
shutil.copy("outputs/charts/drawdown.png", "docs/charts/")

# Portfolio weights table
weights_df = pd.DataFrame(list(weights.items()), columns=["Ticker", "Weight"])
weights_df["Weight"] = (weights_df["Weight"] * 100).round(2).astype(str) + "%"

# Build HTML dashboard
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ESG Automation Portfolio</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        h1 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 400px; margin-bottom: 20px; }}
        table, th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
        img {{ max-width: 600px; display: block; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>ESG Automation Portfolio Backtest</h1>

    <h2>Portfolio Weights</h2>
    {weights_df.to_html(index=False)}

    <h2>Metrics</h2>
    {metrics.to_html(index=False)}

    <h2>Charts</h2>
    <h3>Cumulative Returns</h3>
    <img src="charts/cumulative_returns.png" alt="Cumulative Returns">

    <h3>Rolling Sharpe Ratio</h3>
    <img src="charts/rolling_sharpe.png" alt="Rolling Sharpe Ratio">

    <h3>Drawdown</h3>
    <img src="charts/drawdown.png" alt="Drawdown">
</body>
</html>
"""

with open("docs/index.html", "w") as f:
    f.write(html_content)

print("GitHub Pages dashboard generated at docs/index.html")
