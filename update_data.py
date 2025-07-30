import json
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, UTC
import os

# ========= CONFIGURATION =========
# Your portfolio tickers and allocation details
portfolio_allocations = {
    "ROK": {"weight": 5.56, "quantity": 27.35, "initial_price": 203.14},
    "EMR": {"weight": 4.44, "quantity": 75.09, "initial_price": 59.19},
    "HON": {"weight": 4.44, "quantity": 29.71, "initial_price": 149.62},
    "MSFT": {"weight": 7.78, "quantity": 38.50, "initial_price": 202.00},
    "NVDA": {"weight": 7.78, "quantity": 576.56, "initial_price": 13.49},
    "PLTR": {"weight": 4.44, "quantity": 467.84, "initial_price": 9.50},
    "CRWD": {"weight": 4.44, "quantity": 32.37, "initial_price": 137.32},
    "CGNX": {"weight": 3.33, "quantity": 54.05, "initial_price": 61.67},
    "AMAT": {"weight": 4.44, "quantity": 77.92, "initial_price": 57.04},
    "SNOW": {"weight": 3.33, "quantity": 13.28, "initial_price": 251.00},
    "SSYS": {"weight": 3.33, "quantity": 267.31, "initial_price": 12.47},
    "DDD": {"weight": 2.22, "quantity": 452.59, "initial_price": 4.91},
    "ROBO": {"weight": 16.67, "quantity": 349.31, "initial_price": 47.71},
    "SOXX": {"weight": 11.11, "quantity": 113.92, "initial_price": 97.53},
    "ESGU": {"weight": 11.11, "quantity": 155.38, "initial_price": 71.51},
    "ICLN": {"weight": 5.56, "quantity": 320.03, "initial_price": 17.36},
}

# Optional: NewsAPI Key (Set in GitHub Secrets for security)
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # If not provided, will skip news fetch

# ========= FUNCTIONS =========

def fetch_live_prices(tickers):
    """Fetch latest prices using yfinance"""
    data = yf.download(tickers, period="1d")["Adj Close"].iloc[-1]
    return {ticker: float(data.get(ticker, 0)) for ticker in tickers}

def fetch_news(ticker):
    """Fetch latest news headline using NewsAPI (optional)"""
    if not NEWS_API_KEY:
        return "No news available"
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}&pageSize=1"
    try:
        response = requests.get(url, timeout=5)
        articles = response.json().get("articles", [])
        return articles[0]["title"] if articles else "No news available"
    except Exception:
        return "No news available"

def calculate_metrics(portfolio_daily):
    """Calculate basic portfolio performance metrics"""
    cum_return = (1 + portfolio_daily).cumprod()
    total_return = float(cum_return[-1] - 1)
    cagr = float(cum_return[-1] ** (1 / 3) - 1)  # Assume 3-year data
    max_drawdown = float((cum_return / np.maximum.accumulate(cum_return) - 1).min())
    return total_return, cagr, max_drawdown

# ========= MAIN LOGIC =========

# Ensure docs folder exists
os.makedirs("docs", exist_ok=True)

tickers = list(portfolio_allocations.keys())
live_prices = fetch_live_prices(tickers)

# Build holdings list with live prices & news
holdings = []
for ticker, info in portfolio_allocations.items():
    last_price = round(live_prices.get(ticker, 0), 2)
    news_headline = fetch_news(ticker)

    holdings.append({
        "ticker": ticker,
        "name": f"{ticker} Corporation",  # Placeholder names
        "weight": f"{info['weight']}%",
        "quantity": info["quantity"],
        "initial_price": info["initial_price"],
        "last_price": last_price,
        "news": news_headline
    })

# Simple simulated daily returns for metrics (replace with actual later)
np.random.seed(42)
portfolio_daily = np.random.normal(0.0005, 0.01, 252 * 3)  # 3 years

total_return, cagr, max_drawdown = calculate_metrics(portfolio_daily)

# Sample performance table
performance = {
    "1M": {"Portfolio": "10.15%", "QQQ": "6.39%", "SPY": "5.14%", "ESGU": "5.16%"},
    "3M": {"Portfolio": "36.12%", "QQQ": "17.77%", "SPY": "10.78%", "ESGU": "11.26%"},
    "6M": {"Portfolio": "18.99%", "QQQ": "4.43%", "SPY": "3.36%", "ESGU": "2.90%"},
    "YTD": {"Portfolio": "23.72%", "QQQ": "10.10%", "SPY": "5.61%", "ESGU": "4.95%"},
    "1Y": {"Portfolio": "36.02%", "QQQ": "15.19%", "SPY": "14.49%", "ESGU": "14.32%"},
    "Since Inception": {"Portfolio": "209.84%", "QQQ": "101.27%", "SPY": "96.09%", "ESGU": "88.00%"}
}

# Save data to docs/portfolio.json
data = {
    "portfolio_weights": {k: v["weight"] for k, v in portfolio_allocations.items()},
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

print("Portfolio data updated with live prices.")
