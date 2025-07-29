import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# Portfolio tickers
stocks = [
    "ROK", "EMR", "HON", "MSFT", "NVDA", "PLTR", "CRWD",
    "CGNX", "AMAT", "SNOW", "SSYS", "DDD"
]
etfs = ["ROBO", "SOXX", "ESGU", "ICLN"]
portfolio = stocks + etfs

# Weights
weights = {
    "ROK": 0.05, "EMR": 0.04, "HON": 0.04, "MSFT": 0.07, "NVDA": 0.07,
    "PLTR": 0.04, "CRWD": 0.04, "CGNX": 0.03, "AMAT": 0.04, "SNOW": 0.03,
    "SSYS": 0.03, "DDD": 0.02, "ROBO": 0.15, "SOXX": 0.10,
    "ESGU": 0.10, "ICLN": 0.05
}

total_weight = sum(weights.values())
weights = {k: v / total_weight for k, v in weights.items()}

# Download data
data = yf.download(portfolio, start="2019-01-01", end="2025-07-01")['Adj Close'].dropna(axis=1)

# Adjust weights for available tickers
weights = {k: v for k, v in weights.items() if k in data.columns}
weight_array = np.array(list(weights.values()))

# Daily returns
daily_returns = data.pct_change().dropna()
portfolio_returns = daily_returns.dot(weight_array)
cumulative_returns = (1 + portfolio_returns).cumprod()

# Benchmarks
benchmarks = yf.download(["QQQ", "SPY", "ESGU"], start="2019-01-01", end="2025-07-01")['Adj Close']
benchmarks = benchmarks.pct_change().dropna()
benchmarks_cum = (1 + benchmarks).cumprod()

# Metrics
cagr = (cumulative_returns[-1]) ** (252 / len(portfolio_returns)) - 1
volatility = portfolio_returns.std() * np.sqrt(252)
sharpe = cagr / volatility
max_drawdown = ((cumulative_returns / cumulative_returns.cummax()) - 1).min()

metrics = pd.DataFrame({
    "CAGR": [f"{cagr*100:.2f}%"],
    "Volatility": [f"{volatility*100:.2f}%"],
    "Sharpe Ratio": [f"{sharpe:.2f}"],
    "Max Drawdown": [f"{max_drawdown*100:.2f}%"]
})

# Save outputs
os.makedirs("outputs/charts", exist_ok=True)
metrics.to_excel("outputs/portfolio_metrics.xlsx", index=False)

# Plot cumulative returns
plt.figure(figsize=(10,6))
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

print("Backtest complete. Results saved in outputs/ folder.")
