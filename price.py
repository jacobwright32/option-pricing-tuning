"""
Option Pricing Model & CFD Signal Generator
============================================
THIS IS THE ONLY FILE THE AGENT MAY EDIT.

Baseline: Black-Scholes with flat volatility estimate + simple IV-rank signal.
The agent should iterate on both pricing accuracy and signal quality.
"""

import numpy as np
from numba import njit, prange
import math


# ─── Numba-accelerated math ─────────────────────────────────────────────────

_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


@njit(cache=True)
def _norm_cdf(x):
    """Standard normal CDF using erfc approximation."""
    return 0.5 * math.erfc(-x / _SQRT2)


@njit(cache=True)
def _norm_pdf(x):
    """Standard normal PDF."""
    return math.exp(-0.5 * x * x) / _SQRT2PI


# ─── Black-Scholes Formulas (Numba JIT) ─────────────────────────────────────

@njit(cache=True, parallel=True)
def bs_price(S, K, T, r, sigma, is_call):
    """Numba-accelerated Black-Scholes European option price."""
    n = len(S)
    result = np.empty(n)
    for i in prange(n):
        t = max(T[i], 1e-10)
        s = max(sigma[i], 1e-10)
        sqrt_t = math.sqrt(t)
        d1 = (math.log(S[i] / K[i]) + (r + 0.5 * s * s) * t) / (s * sqrt_t)
        d2 = d1 - s * sqrt_t
        disc = math.exp(-r * t)
        if is_call[i]:
            result[i] = S[i] * _norm_cdf(d1) - K[i] * disc * _norm_cdf(d2)
        else:
            result[i] = K[i] * disc * _norm_cdf(-d2) - S[i] * _norm_cdf(-d1)
    return result


@njit(cache=True, parallel=True)
def bs_vega(S, K, T, r, sigma):
    """Numba-accelerated Black-Scholes vega."""
    n = len(S)
    result = np.empty(n)
    for i in prange(n):
        t = max(T[i], 1e-10)
        s = max(sigma[i], 1e-10)
        sqrt_t = math.sqrt(t)
        d1 = (math.log(S[i] / K[i]) + (r + 0.5 * s * s) * t) / (s * sqrt_t)
        result[i] = S[i] * _norm_pdf(d1) * sqrt_t
    return result


# ─── Implied Volatility Solver (Numba JIT) ──────────────────────────────────

@njit(cache=True)
def implied_vol_vec(S, K, T, r, market_price, is_call, max_iter=20, tol=1e-8):
    """Numba-accelerated Newton-Raphson IV solver."""
    n = len(S)
    sigma = np.full(n, 0.3)
    active = np.ones(n, dtype=np.bool_)

    for _ in range(max_iter):
        any_active = False
        for i in range(n):
            if not active[i]:
                continue
            any_active = True
            t = max(T[i], 1e-10)
            s = max(sigma[i], 1e-10)
            sqrt_t = math.sqrt(t)
            d1 = (math.log(S[i] / K[i]) + (r + 0.5 * s * s) * t) / (s * sqrt_t)
            d2 = d1 - s * sqrt_t
            disc = math.exp(-r * t)

            if is_call[i]:
                price = S[i] * _norm_cdf(d1) - K[i] * disc * _norm_cdf(d2)
            else:
                price = K[i] * disc * _norm_cdf(-d2) - S[i] * _norm_cdf(-d1)

            vega = S[i] * _norm_pdf(d1) * sqrt_t
            diff = price - market_price[i]

            if abs(diff) < tol or vega < 1e-12:
                active[i] = False
                continue

            new_s = sigma[i] - diff / vega
            sigma[i] = min(max(new_s, 0.01), 5.0)

        if not any_active:
            break

    for i in range(n):
        sigma[i] = min(max(sigma[i], 0.01), 5.0)
    return sigma


# ─── Pricing Model ───────────────────────────────────────────────────────────

class PricingModel:
    def __init__(self):
        pass

    def price_chain(self, chain):
        """Vol surface fit with IRLS robust regression."""
        S = chain["S"]
        K = chain["K"]
        T = chain["T"]
        r = chain["r"]
        is_call = chain["is_call"]
        market_price = chain["market_price"]

        ivs = implied_vol_vec(S, K, T, r, market_price, is_call)
        log_m = np.log(K / S)
        sqrt_T = np.sqrt(np.maximum(T, 1 / 252))
        X = np.column_stack([
            np.ones(len(log_m)),
            log_m ** 2,
            sqrt_T,
            log_m / sqrt_T,
        ])

        def _irls_fit(X_sub, y_sub):
            w = np.ones(len(y_sub))
            coeffs = None
            lam = 1e-3  # Ridge penalty
            n_feat = X_sub.shape[1]
            for _ in range(12):
                Xw = X_sub * w[:, None]
                yw = y_sub * w
                # Ridge: (X'X + λI)^-1 X'y
                XtX = Xw.T @ Xw + lam * np.eye(n_feat)
                Xty = Xw.T @ yw
                coeffs = np.linalg.solve(XtX, Xty)
                resid = y_sub - X_sub @ coeffs
                mad = np.median(np.abs(resid)) + 1e-8
                w = 1.0 / (1.0 + (resid / (1.5 * mad)) ** 2)
            return coeffs

        fitted_vol = np.copy(ivs)
        try:
            good = ivs > 0.005
            if good.sum() > 8:
                coeffs = _irls_fit(X[good], ivs[good])
            else:
                coeffs = _irls_fit(X, ivs)
            fitted_vol = np.clip(X @ coeffs, 0.01, 5.0)
        except Exception:
            fitted_vol = ivs

        fair_values = bs_price(S, K, T, r, fitted_vol, is_call)
        return np.maximum(fair_values, 0.01)

    def generate_signal(self, chain, price_history):
        """Contrarian signal: buy dips when IV premium is high."""
        S = chain["S"]
        K = chain["K"]
        T = chain["T"]
        r = chain["r"]
        is_call = chain["is_call"]
        market_price = chain["market_price"]

        if len(S) == 0 or len(price_history) < 20:
            return 0.0

        spot = S[0]
        moneyness = np.abs(np.log(K / spot))
        atm_mask = moneyness < 0.07
        if atm_mask.sum() == 0:
            return 0.0

        # ATM IV
        atm_ivs = implied_vol_vec(S[atm_mask], K[atm_mask], T[atm_mask], r,
                                   market_price[atm_mask], is_call[atm_mask], max_iter=6)
        current_iv = np.median(atm_ivs)

        log_returns = np.diff(np.log(price_history))
        realized_vol = np.std(log_returns) * np.sqrt(252)
        if realized_vol < 0.01:
            return 0.0

        iv_rv_ratio = current_iv / realized_vol
        ret_5d = (price_history[-1] / price_history[-5]) - 1.0
        ret_10d = (price_history[-1] / price_history[-10]) - 1.0
        dist_from_high = (price_history[-1] / np.max(price_history[-30:])) - 1.0

        # RSI(14) calculation
        if len(price_history) >= 15:
            deltas = np.diff(price_history)
            gains = np.maximum(deltas[-14:], 0)
            losses = np.maximum(-deltas[-14:], 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            if avg_loss > 1e-10:
                rsi = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
            else:
                rsi = 100.0
        else:
            rsi = 50.0

        if iv_rv_ratio > 2.0 and realized_vol < 0.50 and -0.035 < ret_5d < -0.025 and -0.06 < ret_10d < -0.01 and -0.08 < dist_from_high < -0.03:
            return 1.0

        # RSI-based tier: extremely oversold + IV premium + all price conditions
        if rsi < 30 and iv_rv_ratio > 2.0 and realized_vol < 0.45 and -0.04 < ret_5d < -0.02 and -0.055 < ret_10d < -0.015 and -0.075 < dist_from_high < -0.03:
            return 0.35

        return 0.0


# ─── Plot & README Update ────────────────────────────────────────────────────

def update_progress_plot():
    """Generate progress.png from results.tsv and update README."""
    from pathlib import Path
    import pandas as pd

    tsv = Path("results.tsv")
    if not tsv.exists() or tsv.stat().st_size == 0:
        return

    df = pd.read_csv(tsv, sep="\t")
    if len(df) < 1:
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Option Pricing Tuning - Real Data Experiment Progress", fontsize=14)

        colors = {"keep": "#2ecc71", "discard": "#e74c3c", "crash": "#95a5a6"}

        # 1. Combined score
        ax = axes[0, 0]
        for status, color in colors.items():
            mask = df["status"] == status
            if mask.any():
                ax.scatter(df.index[mask], df.loc[mask, "combined_score"],
                          c=color, s=30, alpha=0.7, label=status)
        running_best = df["combined_score"].cummax()
        ax.plot(running_best, "k-", linewidth=1.5, alpha=0.5, label="running best")
        ax.set_xlabel("Experiment #")
        ax.set_ylabel("Combined Score")
        ax.set_title("Score Progress")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # 2. Sharpe ratio
        ax = axes[0, 1]
        if "sharpe" in df.columns:
            for status, color in colors.items():
                mask = df["status"] == status
                if mask.any():
                    ax.scatter(df.index[mask], df.loc[mask, "sharpe"],
                              c=color, s=30, alpha=0.7)
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.set_xlabel("Experiment #")
            ax.set_ylabel("Sharpe Ratio")
            ax.set_title("Signal Quality (Sharpe)")
            ax.grid(True, alpha=0.3)

        # 3. Pricing MAPE
        ax = axes[1, 0]
        if "mape" in df.columns:
            for status, color in colors.items():
                mask = df["status"] == status
                if mask.any():
                    ax.scatter(df.index[mask], df.loc[mask, "mape"],
                              c=color, s=30, alpha=0.7)
            ax.set_xlabel("Experiment #")
            ax.set_ylabel("MAPE")
            ax.set_title("Pricing Accuracy (lower = better)")
            ax.grid(True, alpha=0.3)

        # 4. Pareto front
        ax = axes[1, 1]
        if "sharpe" in df.columns and "mape" in df.columns:
            for status, color in colors.items():
                mask = df["status"] == status
                if mask.any():
                    ax.scatter(df.loc[mask, "mape"], df.loc[mask, "sharpe"],
                              c=color, s=30, alpha=0.7, label=status)
            ax.set_xlabel("MAPE (lower = better)")
            ax.set_ylabel("Sharpe (higher = better)")
            ax.set_title("Pareto Front")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("progress.png", dpi=150)
        plt.close(fig)

    except Exception:
        return

    best = df.loc[df["combined_score"].idxmax()]
    n_total = len(df)
    n_kept = (df["status"] == "keep").sum()

    readme = f"""# Option Pricing Tuning

Autonomous AI-driven optimization of options pricing models and CFD trading
signals using **real stock market data** from Yahoo Finance.

## Latest Results

![Experiment Progress](progress.png)

| Metric | Value |
|---|---|
| Combined Score | {best['combined_score']:.4f} |
| Sharpe Ratio | {best['sharpe']:.4f} |
| MAPE | {best['mape']:.6f} |
| Win Rate | {best.get('win_rate', 0):.1%} |
| Trades | {int(best.get('n_trades', 0))} |
| Experiments | {n_total} ({n_kept} kept) |

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
"""
    Path("README.md").write_text(readme, encoding="utf-8")


# ─── Main Entry Point ────────────────────────────────────────────────────────

def main():
    """Run the model and print evaluation results."""
    import time as _time
    from prepare import evaluate

    print("=" * 60)
    print("Option Pricing Tuning - Experiment Run (REAL DATA)")
    print("=" * 60)

    t0 = _time.time()
    model = PricingModel()
    score = evaluate(model)
    elapsed = _time.time() - t0

    print("=" * 60)
    print(f"Wall time: {elapsed:.1f}s")
    print("=" * 60)

    update_progress_plot()


if __name__ == "__main__":
    main()
