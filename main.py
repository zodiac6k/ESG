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
INITIAL_INVESTMENT = 100000  # USD
ROLLING_WINDOW = 126  # ~6 months

# Portfolio tickers and weights
stocks = [
    "ROK", "EMR", "HON", "MSFT", "NVDA", "PLTR", "CRWD",
    "CGNX", "AMAT", "SNOW", "SSYS", "DDD"
]
etfs = ["ROBO", "SOXX", "ESGU", "ICLN"]
portfolio = stocks + etfs

weights = {
    "ROK": 0.05, "EMR": 0.04, "HON": 0.04, "MSFT": 0.07, "NVDA": 0.07,
    "PLTR": 0.04, "CRWD": 0.04, "CGNX": 0.03, "AMAT": 0.04, "SNOW": 0.03,
    "SSYS": 0.03, "DDD": 0.02, "ROBO": 0.15, "SOXX": 0.10,
    "ESGU": 0.10, "ICLN": 0.05
}
weights = {k: v / sum(weights.values()) for k, v in weights.items()}

# -----------------------------
# DOWNLOAD PORTFOLIO DATA
# -----------------------------
pft_raw = yf.download(portfolio, start=START_DATE, end=END_DATE, auto_adjust=False)

# Handle MultiIndex or single-level columns
if isinstance(pft_raw.columns, pd.MultiIndex):
    if "Adj Close" in pft_raw.columns.levels[0]:
        data = pft_raw["Adj Close"].dropna()
    else:
        data = pft_raw["Close"].dropna()
else:
    data = pft_raw.dropna()

# Drop missing tickers & adjust weights
data = data.dropna(axis=1)
weights = {k: v for k, v in weights.items() if k in data.columns}
if not weights:
    raise ValueError("No valid tickers with available data!")

# -----------------------------
# CALCULATE QUANTITIES & VALUE
# -----------------------------
first_prices = data.iloc[0]
quantities = {t: (weights[t] * INITIAL_INVESTMENT) / first_prices[t] for t in weights}

# Portfolio value
portfolio_values = data[list(weights.keys())].mul(list(quantities.values()), axis=1).sum(axis=1)
portfolio_returns = portfolio_values.pct_change().dropna()

# -----------------------------
# BENCHMARK DATA
# -----------------------------
bench_raw = yf.download(["QQQ", "SPY", "ESGU"], start=START_DATE, end=END_DATE, auto_adjust=False)

# Handle MultiIndex or single-level columns
if isinstance(bench_raw.columns, pd.MultiIndex):
    if "Adj Close" in bench_raw.columns.levels[0]:
        bench_data = bench_raw["Adj Close"].dropna()
    else:
        bench_data = bench_raw["Close"].dropna()
else:
    bench_data = bench_raw.dropna()

# -----------------------------
# PERFORMANCE SNAPSHOT
# -----------------------------
last_val = portfolio_values.iloc[-1]
prev_val = portfolio_values.iloc[-2]
daily_change_pct = ((last_val - prev_val) / prev_val) * 100
arrow = "↑" if daily_change_pct > 0 else "↓"
color_daily = "#28a745" if daily_change_pct > 0 else "#dc3545"

bench_daily = {}
for b in bench_data.columns:
    bench_daily[b] = ((bench_data[b].iloc[-1] - bench_data[b].iloc[-2]) / bench_data[b].iloc[-2]) * 100

# -----------------------------
# PORTFOLIO METRICS
# -----------------------------
cagr = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (252 / len(portfolio_returns)) - 1
volatility = portfolio_returns.std() * np.sqrt(252)
sharpe = cagr / volatility
max_drawdown = ((portfolio_values / portfolio_values.cummax()) - 1).min()

metrics = pd.DataFrame({
    "Portfolio": [f"{cagr*100:.2f}%", f"{volatility*100:.2f}%", f"{sharpe:.2f}", f"{max_drawdown*100:.2f}%"],
    "QQQ": [
        f"{((bench_data['QQQ'].iloc[-1]/bench_data['QQQ'].iloc[0])**(252/len(bench_data))-1)*100:.2f}%",
        "", "", ""
    ],
    "SPY": [
        f"{((bench_data['SPY'].iloc[-1]/bench_data['SPY'].iloc[0])**(252/len(bench_data))-1)*100:.2f}%",
        "", "", ""
    ]
}, index=["CAGR", "Volatility", "Sharpe Ratio", "Max Drawdown"])

# -----------------------------
# MULTI-TIMEFRAME COMPARISON
# -----------------------------
def calc_return(series, days):
    if len(series) < days:
        return np.nan
    return ((series.iloc[-1] / series.iloc[-days]) - 1) * 100

timeframes = {
    "1M": 21, "3M": 63, "6M": 126, "YTD": (pd.Timestamp.today().dayofyear // 7) * 5,
    "1Y": 252, "Since Inception": len(portfolio_values) - 1
}

comparison = pd.DataFrame(index=timeframes.keys())
comparison["Portfolio"] = [calc_return(portfolio_values, d) for d in timeframes.values()]
for b in ["QQQ", "SPY", "ESGU"]:
    comparison[b] = [calc_return(bench_data[b], d) for d in timeframes.values()]

# FIX: use map() instead of applymap (removes deprecation warning)
comparison = comparison.map(lambda x: f"{x:.2f}%" if pd.notnull(x) else "-")

# -----------------------------
# PRICE CHANGE & NEWS
# -----------------------------
last_day_prices = data.iloc[-1]
prev_day_prices = data.iloc[-2]
price_change_colors = {
    t: "#28a745" if last_day_prices[t] - prev_day_prices[t] > 0 else "#dc3545"
    for t in weights
}

company_names = {}
news_data = {}
for t in weights.keys():
    info = yf.Ticker(t).info
    company_names[t] = info.get("longName", t)
    try:
        news_item = yf.Ticker(t).news[0]
        news_data[t] = f'<a href="{news_item["link"]}" target="_blank">{news_item["title"]}</a>'
    except Exception:
        news_data[t] = "No news available"

# -----------------------------
# CHARTS
# -----------------------------
os.makedirs("outputs/charts", exist_ok=True)

# Horizontal dashboard charts
fig, axs = plt.subplots(1, 3, figsize=(20, 6))

axs[0].plot(portfolio_values)
axs[0].set_title("Portfolio Value ($)")
axs[0].grid(True)

rolling_sharpe = (
    (portfolio_returns.rolling(ROLLING_WINDOW).mean() * 252) /
    (portfolio_returns.rolling(ROLLING_WINDOW).std() * np.sqrt(252))
)
axs[1].plot(rolling_sharpe)
axs[1].set_title("Rolling Sharpe Ratio")
axs[1].grid(True)

drawdown = (portfolio_values / portfolio_values.cummax()) - 1
axs[2].plot(drawdown, color="red")
axs[2].set_title("Drawdown")
axs[2].grid(True)

plt.tight_layout()
plt.savefig("outputs/charts/dashboard_charts.png")
plt.close()

# Portfolio vs Benchmarks growth
plt.figure(figsize=(10, 6))
base_val = portfolio_values.iloc[0]
plt.plot(portfolio_values / base_val * 100000, label="Portfolio")
for b in ["QQQ", "SPY", "ESGU"]:
    plt.plot(bench_data[b] / bench_data[b].iloc[0] * 100000, label=b)
plt.title("Portfolio vs Benchmarks ($ Growth)")
plt.ylabel("Value ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/charts/portfolio_vs_benchmarks.png")
plt.close()

# -----------------------------
# GENERATE HTML DASHBOARD
# -----------------------------
os.makedirs("docs/charts", exist_ok=True)
shutil.copy("outputs/charts/dashboard_charts.png", "docs/charts/")
shutil.copy("outputs/charts/portfolio_vs_benchmarks.png", "docs/charts/")

# Portfolio table rows
table_rows = ""
for t in weights.keys():
    color = price_change_colors[t]
    table_rows += f"""
    <tr>
        <td>{t}</td>
        <td>{company_names[t]}</td>
        <td>{weights[t]*100:.2f}%</td>
        <td>{quantities[t]:.2f}</td>
        <td>{first_prices[t]:.2f}</td>
        <td style="color:{color}; font-weight:bold;">{last_day_prices[t]:.2f}</td>
        <td>{news_data[t]}</td>
    </tr>
    """

# Performance snapshot card
perf_html = f"""
<div style="padding:10px;background:#f1f3f5;border-radius:10px;margin-bottom:20px;">
<h2>Current Portfolio Value: ${last_val:,.2f} <span style="color:{color_daily};">{arrow} {daily_change_pct:.2f}%</span></h2>
<p>QQQ daily: {bench_daily['QQQ']:.2f}% | SPY daily: {bench_daily['SPY']:.2f}% | ESGU daily: {bench_daily['ESGU']:.2f}%</p>
</div>
"""

# Final HTML (UTF-8 safe)
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ESG Automation Portfolio</title>
    <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; background-color: #f8f9fa; }}
        h1 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        th {{ background-color: #2c3e50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        tr:hover {{ background-color: #ddd; }}
        img {{ max-width: 100%; display: block; margin: auto; }}
    </style>
</head>
<body>
    <h1>ESG Automation Portfolio Dashboard</h1>

    {perf_html}

    <h2>Portfolio Allocation & News</h2>
    <table>
        <tr>
            <th>Ticker</th><th>Name</th><th>Weight</th><th>Quantity</th>
            <th>Initial Price</th><th>Last Price</th><th>News</th>
        </tr>
        {table_rows}
    </table>

    <h2>Portfolio vs Benchmarks (Metrics)</h2>
    {metrics.to_html(classes="data", header=True)}

    <h2>Multi-Timeframe Performance (%)</h2>
    {comparison.to_html(classes="data", header=True)}

    <h2>Charts</h2>
    <img src="charts/dashboard_charts.png" alt="Dashboard Charts">
    <h3>Portfolio vs Benchmarks ($ Growth)</h3>
    <img src="charts/portfolio_vs_benchmarks.png" alt="Portfolio vs Benchmarks">
</body>
</html>
"""

# FIX: Use UTF-8 encoding for file write (avoids UnicodeEncodeError)
with open("docs/index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("GitHub Pages dashboard generated at docs/index.html")
