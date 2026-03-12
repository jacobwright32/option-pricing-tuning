# Option Pricing Tuning

Autonomous AI-driven optimization of options pricing models and CFD trading
signals using **real stock market data** from Yahoo Finance.

## Latest Results

![Experiment Progress](progress.png)

| Metric | Value |
|---|---|
| Combined Score | 2.2524 |
| Sharpe Ratio | 3.0887 |
| MAPE | 0.002003 |
| Win Rate | 86.2% |
| Trades | 29 |
| Experiments | 850 (199 kept) |

## How It Works

An AI agent (Claude) runs an experiment loop:

1. Edits `price.py` with a new idea (better vol surface, smarter signals, etc.)
2. Runs the model against **real stock prices** (SPY, QQQ, AAPL, MSFT, NVDA, TSLA, etc.)
3. Measures **pricing accuracy** (MAPE vs ground truth) and **CFD signal quality** (Sharpe ratio)
4. Keeps improvements, discards regressions, loops

## Data

- **20 real US stocks** from Yahoo Finance (~2 years of daily prices)
- Synthetic options generated on real prices with realistic vol surfaces
- CFD simulation with Trading 212 costs (0.1% spread, ~3% annual overnight financing)

## Scoring

```
combined_score = 0.4 * pricing_score + 0.6 * signal_score
```

- **pricing_score** = `max(0, 1 - MAPE)` - accuracy vs ground truth option prices
- **signal_score** = annualized Sharpe ratio of simulated CFD trades
