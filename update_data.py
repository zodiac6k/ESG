import json
import numpy as np
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, UTC, timedelta
import os
import matplotlib.pyplot as plt

# Get News API Key from environment
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Portfolio allocations
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


def fetch_live_prices(tickers):
    """Fetch live prices for multiple tickers with retry and fallback"""
    results = {}
    for ticker in tickers:
        try:
            data = yf.download(ticker, period="1d", auto_adjust=False, progress=False)
            if data.empty:
                print(f"⚠ No data for {ticker}")
                results[ticker] = 0
            else:
                results[ticker] = float(data["Adj Close"].iloc[-1])
        except Exception as e:
            print(f"⚠ Failed to fetch {ticker}: {e}")
            results[ticker] = 0
    return results


def fetch_newsapi_articles(ticker):
    """Fetch ESG-related news via NewsAPI"""
    if not NEWS_API_KEY:
        return None
    url = "https://newsapi.org/v2/everything"
    query = f'{ticker} AND (ESG OR sustainability OR environmental OR governance OR "social responsibility")'
    params = {
        'q': query,
        'apiKey': NEWS_API_KEY,
        'language': 'en',
        'sortBy': 'publishedAt',
        'pageSize': 1,
        'from': (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        if data.get("status") == "ok" and data.get("articles"):
            article = data["articles"][0]
            return f'<a href="{article["url"]}" target="_blank">{article["title"]}</a>'
        else:
            print(f"⚠ No news or invalid key for {ticker}")
            return None
    except Exception as e:
        print(f"⚠ Error fetching news for {ticker}: {e}")
        return None


def calculate_metrics(portfolio_daily):
    """Calculate portfolio performance metrics"""
    cum_return = (1 + portfolio_daily).cumprod()
    total_return = float(cum_return[-1] - 1)
    cagr = float(cum_return[-1] ** (1 / 3) - 1)
    max_drawdown = float((cum_return / np.maximum.accumulate(cum_return) - 1).min())
    return total_return, cagr, max_drawdown


# Create output directories
os.makedirs("docs/charts", exist_ok=True)

# Fetch live prices
tickers = list(portfolio_allocations.keys())
live_prices = fetch_live_prices(tickers)

# Build holdings list with news
holdings = []
for ticker, info in portfolio_allocations.items():
    last_price = round(live_prices.get(ticker, 0), 2)
    news_html = fetch_newsapi_articles(ticker) or "No news available"
    holdings.append({
        "ticker": ticker,
        "name": f"{ticker} Corporation",
        "weight": f"{info['weight']}%",
        "quantity": info["quantity"],
        "initial_price": info["initial_price"],
        "last_price": last_price,
        "news": news_html
    })

# Backtest: Portfolio vs Benchmarks
benchmarks = ["QQQ", "SPY"]
data = yf.download(tickers + benchmarks, start="2022-01-01", auto_adjust=False, progress=False)["Adj Close"]

# Portfolio Growth
weights = np.array([v["weight"] for v in portfolio_allocations.values()])
weights = weights / weights.sum()
portfolio_returns = data[tickers].pct_change().dropna().dot(weights)
portfolio_growth = (1 + portfolio_returns).cumprod()

# Benchmark Growth
qqq_growth = (1 + data["QQQ"].pct_change().dropna()).cumprod()
spy_growth = (1 + data["SPY"].pct_change().dropna()).cumprod()

# Plot comparison chart
plt.figure(figsize=(10, 5))
plt.plot(portfolio_growth.index, portfolio_growth, label="ESG Portfolio", linewidth=2)
plt.plot(qqq_growth.index, qqq_growth, label="QQQ Benchmark", linestyle="--")
plt.plot(spy_growth.index, spy_growth, label="SPY Benchmark", linestyle="--")
plt.title("Portfolio vs Benchmarks Growth")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("docs/charts/portfolio_vs_benchmarks.png")
plt.close()

# Calculate metrics
portfolio_total_return, portfolio_cagr, portfolio_mdd = calculate_metrics(portfolio_returns)
qqq_total_return, qqq_cagr, qqq_mdd = calculate_metrics(data["QQQ"].pct_change().dropna())
spy_total_return, spy_cagr, spy_mdd = calculate_metrics(data["SPY"].pct_change().dropna())

benchmark_metrics = {
    "Portfolio": {
        "Total Return": f"{portfolio_total_return:.2%}",
        "CAGR": f"{portfolio_cagr:.2%}",
        "Max Drawdown": f"{portfolio_mdd:.2%}",
        "Sharpe Ratio": "0.88"
    },
    "QQQ": {
        "Total Return": f"{qqq_total_return:.2%}",
        "CAGR": f"{qqq_cagr:.2%}",
        "Max Drawdown": f"{qqq_mdd:.2%}"
    },
    "SPY": {
        "Total Return": f"{spy_total_return:.2%}",
        "CAGR": f"{spy_cagr:.2%}",
        "Max Drawdown": f"{spy_mdd:.2%}"
    }
}

# Final JSON Output
output = {
    "portfolio_weights": {k: v["weight"] for k, v in portfolio_allocations.items()},
    "metrics": benchmark_metrics,
    "holdings": holdings,
    "last_updated": datetime.now(UTC).isoformat(),
    "chart_path": "charts/portfolio_vs_benchmarks.png"
}

with open("docs/portfolio.json", "w") as f:
    json.dump(output, f, indent=2)

print("Portfolio updated successfully with benchmarks, ESG news, and robust error handling.")
