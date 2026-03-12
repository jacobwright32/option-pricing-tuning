# Option Pricing Tuning — Autonomous Experiment Protocol

You are an autonomous research agent optimizing an options pricing model
and CFD signal generator. Your goal: maximize the **combined_score** metric
by iterating on `price.py`.

## Files

| File | Editable? | Purpose |
|---|---|---|
| `price.py` | **YES — this is the ONLY file you edit** | Pricing model + CFD signal generator |
| `prepare.py` | NO — never modify | Data generation, evaluation, CFD simulator |
| `program.md` | NO — never modify | This document (your instructions) |
| `results.tsv` | Append only | Experiment log (you maintain this) |
| `analysis.py` | NO — never modify | Post-run visualization |

## Metric

**`combined_score`** (higher is better):

    combined = 0.4 × pricing_score + 0.6 × signal_score

- `pricing_score = max(0, 1 − MAPE)` — how accurately you price options vs ground truth
- `signal_score = sharpe_ratio` — annualized Sharpe of CFD trades from your signals

The ground truth uses asset-specific volatility surfaces with skew, smile,
and term structure. Black-Scholes with flat vol will systematically misprice.
The baseline signal is a simple IV-vs-realized-vol ratio.

## Setup (Phase 1)

1. Propose a run tag (e.g., `mar12`). Create branch `autoresearch/<tag>`.
2. Read all in-scope files: `price.py`, `prepare.py`, `program.md`.
3. Create `results.tsv` with header:
   ```
   commit	combined_score	sharpe	mape	status	description
   ```
4. Run baseline: `python price.py > run.log 2>&1`
5. Extract `combined_score` from `run.log`, log to `results.tsv` as baseline.

## Experiment Loop (Phase 2) — LOOP FOREVER

```
LOOP:
    1. git status / git log --oneline -3   (know where you are)
    2. Think of an experiment (see Ideas below)
    3. Edit price.py
    4. git add price.py && git commit -m "<short description>"
    5. Run: python price.py > run.log 2>&1
    6. Check run.log:
       - If crash → read tail, attempt fix or SKIP
       - If timeout (>3 min) → kill and SKIP
    7. Extract combined_score, sharpe_ratio, pricing_mape
    8. Append to results.tsv
    9. If combined_score IMPROVED → KEEP (leave commit in place)
       If combined_score EQUAL or WORSE → DISCARD (git reset --hard HEAD~1)
    10. GOTO 1
```

**NEVER STOP.** Do not pause to ask the human. The human might be asleep.
The loop runs until the human interrupts you, period.

## Constraints

- Only edit `price.py`. No new files. No new dependencies beyond what's
  in `pyproject.toml`.
- Do not modify `prepare.py` or the evaluation metric.
- Each run must complete within **3 minutes** wall time. If it doesn't,
  your model is too slow — simplify.
- Keep the `PricingModel` class interface: `price_chain(chain)` and
  `generate_signal(chain, price_history)`.
- Redirect output: `> run.log 2>&1` to avoid flooding your context.
- All else being equal, **simpler is better**. A small improvement that adds
  ugly complexity is not worth it. Removing something and getting equal or
  better results is a great outcome.

## Experiment Ideas

### Pricing Improvements
- Fit a quadratic/cubic vol smile per chain instead of flat vol
- SVI (Stochastic Volatility Inspired) parameterization
- SABR model calibration
- Weight IV estimates by vega (more reliable for ATM)
- Separate call/put IV handling (put-call parity arbitrage check)
- Interpolate vol surface across strikes and expiries
- Use moneyness-binned median IV instead of global median

### Signal Improvements
- IV percentile rank over rolling window (mean-reversion signal)
- Put-call skew: (put IV − call IV) as sentiment indicator
- Term structure slope: short-dated IV vs long-dated IV
- Realized vol regime detection (high/low vol environments)
- Combine pricing mispricing z-score as signal
- Momentum overlay (trend following + vol signal)
- Vol-adjusted position sizing (Kelly or vol-target)
- Multi-timeframe signals (combine 7d, 30d, 90d IV)
- Skew change signal (steepening/flattening)

### Speed Improvements
- Rational approximation for IV instead of Newton + Brent
- Vectorize the IV solver (avoid Python loops)
- Cache intermediate calculations

## Results Format

`results.tsv` is tab-separated, one row per experiment:

```
commit	combined_score	sharpe	mape	status	description
a1b2c3d	0.234567	0.45	0.12	keep	baseline
e4f5g6h	0.289012	0.52	0.10	keep	quadratic vol smile
i7j8k9l	0.201234	0.30	0.15	discard	SABR calibration (slower, worse)
```

## CFD Context

The end user trades CFDs on Trading 212. Key implications:
- **Spread cost**: 0.1% per trade — signals must clear this hurdle
- **Overnight financing**: ~3% annualized — holding costs matter
- **Leverage**: up to 5x — position sizing is critical
- **No options trading**: all signals must translate to long/short CFD positions
- **Signal frequency**: every 5 trading days — this is a swing-trading system
- Sharpe ratio matters more than total return (risk-adjusted)
- Max drawdown should stay reasonable (< 30%)

The best experiments will improve BOTH pricing accuracy AND signal quality,
because a better pricing model → better mispricing detection → better signals.
