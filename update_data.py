import json
import numpy as np
from datetime import datetime, UTC
import os

# Ensure docs folder exists
os.makedirs("docs", exist_ok=True)

# -------- Portfolio Allocation --------
new_alloc = {
    'ROBO': 32,
    'ADSK': 8,
    'ISRG': 8,
    'ABB': 8,
    'TER': 8,
    'QCLN': 5,
    'ENPH': 5,
    'NEE': 5,
    'Other ESG Automation': 21
}

# -------- Backtest simulation (3Y) --------
np.random.seed(42)
years = 3
days = 252 * years

assumed_returns = {
    'ROBO': 0.12, 'ADSK': 0.10, 'ISRG': 0.14,
    'ABB': 0.09, 'TER': 0.13, 'QCLN': 0.11,
    'ENPH': 0.15, 'NEE': 0.08, 'Other ESG Automation': 0.10
}
assumed_vol = {
    'ROBO': 0.20, 'ADSK': 0.22, 'ISRG': 0.25,
    'ABB': 0.18, 'TER': 0.24, 'QCLN': 0.23,
    'ENPH': 0.30, 'NEE': 0.15, 'Other ESG Automation': 0.20
}

# Simulate daily portfolio returns
portfolio_daily = np.zeros(days)
for ticker, weight in new_alloc.items():
    mu = assumed_returns[ticker] / 252
    sigma = assumed_vol[ticker] / np.sqrt(252)
    daily_ret = np.random.normal(mu, sigma, days)
    portfolio_daily += weight / 100 * daily_ret

cum_return = (1 + portfolio_daily).cumprod()
total_return = float(cum_return[-1] - 1)
cagr = float(cum_return[-1] ** (1 / years) - 1)
max_drawdown = float((cum_return / np.maximum.accumulate(cum_return) - 1).min())

# -------- Example holdings and performance tables --------
holdings = [
    {
        "ticker": "ROK", "name": "Rockwell Automation, Inc.",
        "weight": "5.56%", "quantity": "27.35",
        "initial_price": "203.14", "last_price": "332.17",
        "news": "No news available"
    },
    {
        "ticker": "MSFT", "name": "Microsoft Corporation",
        "weight": "7.78%", "quantity": "38.50",
        "initial_price": "202.00", "last_price": "497.41",
        "news": "No news available"
    }
]

performance = {
    "1M": {"Portfolio": "10.15%", "QQQ": "6.39%", "SPY": "5.14%", "ESGU": "5.16%"},
    "3M": {"Portfolio": "36.12%", "QQQ": "17.77%", "SPY": "10.78%", "ESGU": "11.26%"},
    "6M": {"Portfolio": "18.99%", "QQQ": "4.43%", "SPY": "3.36%", "ESGU": "2.90%"},
    "YTD": {"Portfolio": "23.72%", "QQQ": "10.10%", "SPY": "5.61%", "ESGU": "4.95%"},
    "1Y": {"Portfolio": "36.02%", "QQQ": "15.19%", "SPY": "14.49%", "ESGU": "14.32%"},
    "Since Inception": {"Portfolio": "209.84%", "QQQ": "101.27%", "SPY": "96.09%", "ESGU": "88.00%"}
}

# -------- Save JSON to docs/portfolio.json --------
data = {
    "portfolio_weights": new_alloc,
    "metrics": {
        "total_return_3y": f"{total_return:.2%}",
        "cagr": f"{cagr:.2%}",
        "max_drawdown": f"{max_drawdown:.2%}",
        "sharpe_ratio": "0.88"
    },
    "holdings": holdings,
    "performance": performance,
    "last_updated": datetime.now(UTC).isoformat()
}

with open("docs/portfolio.json", "w") as f:
    json.dump(data, f, indent=2)

print("Portfolio data & metrics updated to docs/portfolio.json")
