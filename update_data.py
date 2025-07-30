import json
import numpy as np
from datetime import datetime

# -------- Portfolio allocation --------
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
cagr = float(cum_return[-1] ** (1/years) - 1)
max_drawdown = float((cum_return / np.maximum.accumulate(cum_return) - 1).min())

# -------- Save data --------
data = {
    "portfolio_weights": new_alloc,
    "metrics": {
        "total_return_3y": f"{total_return:.2%}",
        "cagr": f"{cagr:.2%}",
        "max_drawdown": f"{max_drawdown:.2%}"
    },
    "last_updated": datetime.utcnow().isoformat() + "Z"
}

with open("data/portfolio.json", "w") as f:
    json.dump(data, f, indent=2)

print("Portfolio data & metrics updated.")
