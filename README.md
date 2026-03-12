# Option Pricing Tuning

Autonomous AI-driven optimization of options pricing models and CFD trading
signals, inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)
and [Atlas-GIC](https://github.com/chrisworsey55/atlas-gic).

## How It Works

An AI agent (Claude) runs an infinite experiment loop:

1. Edits `price.py` with a new idea (better vol surface, smarter signals, etc.)
2. Runs the model against a synthetic options market with realistic vol surfaces
3. Measures **pricing accuracy** (MAPE vs ground truth) and **CFD signal quality** (Sharpe ratio)
4. Keeps improvements, discards regressions, loops forever

The ground truth uses asset-specific volatility surfaces with skew, smile,
and term structure. The baseline Black-Scholes with flat vol systematically
misprices — the agent discovers how to model the vol surface and extract
profitable CFD signals.

## Why Options Pricing for CFD Trading?

Even if you only trade CFDs (e.g., on Trading 212), options data is gold:

- **Implied volatility** tells you what the market expects — set better stop-losses
- **IV rank** signals mean-reversion opportunities — go long when fear is high
- **Put-call skew** reveals institutional sentiment — steep skew = hedging = bearish
- **Vol term structure** detects regime changes before they hit price
- **Mispricing detection** spots informed flow before the move happens

## Quick Start

```bash
cd option-pricing-tuning
pip install -e .

# Run baseline
python price.py

# Or let the agent run autonomously (see program.md)
```

## Files

| File | Purpose |
|---|---|
| `price.py` | **The only file the agent edits.** Pricing model + signal generator. |
| `prepare.py` | Data generation, evaluation pipeline, CFD simulator. Read-only. |
| `program.md` | Agent instructions — the autonomous experiment protocol. |
| `analysis.py` | Post-run visualization of experiment progress. |
| `results.tsv` | Experiment log (created by agent, tab-separated). |

## Evaluation Metric

```
combined_score = 0.4 × pricing_score + 0.6 × signal_score
```

- **pricing_score** = `max(0, 1 − MAPE)` — accuracy vs ground truth option prices
- **signal_score** = annualized Sharpe ratio of simulated CFD trades

Weighted 60% toward signals because the end goal is profitable CFD trading.

## Synthetic Market

- 20 assets with unique vol dynamics (different base vol, skew, smile, term structure)
- 504 trading days (~2 years)
- Options snapshots every 5 days with 13 strikes × 6 expiries × calls + puts
- Ground truth: Black-Scholes with asset-specific vol surfaces
- Market prices: ground truth + realistic noise (0.5–3% bid-ask + microstructure)
- CFD simulation: Trading 212 costs (0.1% spread, ~3% annual overnight financing)
