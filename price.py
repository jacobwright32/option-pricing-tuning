"""
Option Pricing Model & CFD Signal Generator
============================================
THIS IS THE ONLY FILE THE AGENT MAY EDIT.

Baseline: Black-Scholes with flat volatility estimate + simple IV-rank signal.
The agent should iterate on both pricing accuracy and signal quality.
"""

import numpy as np
from scipy.stats import norm


# ─── Black-Scholes Formulas ──────────────────────────────────────────────────

def bs_price(S, K, T, r, sigma, is_call):
    """Vectorized Black-Scholes European option price."""
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(is_call, call, put)


def bs_vega(S, K, T, r, sigma):
    """Black-Scholes vega (sensitivity to vol)."""
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


# ─── Implied Volatility Solver ───────────────────────────────────────────────

def implied_vol_vec(S, K, T, r, market_price, is_call, max_iter=20, tol=1e-8):
    """Fully vectorized Newton-Raphson IV solver with bisection fallback."""
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    market_price = np.asarray(market_price, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)

    sigma = np.full(len(S), 0.3)
    active = np.ones(len(S), dtype=bool)

    for _ in range(max_iter):
        if not active.any():
            break
        price = bs_price(S[active], K[active], T[active], r, sigma[active], is_call[active])
        vega = bs_vega(S[active], K[active], T[active], r, sigma[active])
        diff = price - market_price[active]
        updatable = vega > 1e-12
        converged = np.abs(diff) < tol
        step = np.zeros_like(diff)
        step[updatable] = diff[updatable] / vega[updatable]
        new_sigma = sigma[active] - step
        sigma[active] = np.clip(new_sigma, 0.01, 5.0)
        done_mask = converged | ~updatable
        idx = np.where(active)[0]
        active[idx[done_mask]] = False

    return np.clip(sigma, 0.01, 5.0)


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
            lam = 2e-4  # Ridge penalty
            n_feat = X_sub.shape[1]
            for _ in range(13):
                Xw = X_sub * w[:, None]
                yw = y_sub * w
                # Ridge: (X'X + λI)^-1 X'y
                XtX = Xw.T @ Xw + lam * np.eye(n_feat)
                Xty = Xw.T @ yw
                coeffs = np.linalg.solve(XtX, Xty)
                resid = y_sub - X_sub @ coeffs
                mad = np.median(np.abs(resid)) + 1e-8
                w = 1.0 / (1.0 + (resid / (1.3 * mad)) ** 2)
            return coeffs

        fitted_vol = np.copy(ivs)
        try:
            good = (ivs > 0.10) & (ivs < 0.90)
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
        """Multi-factor contrarian signal with vol surface features."""
        S = chain["S"]
        K = chain["K"]
        T = chain["T"]
        r = chain["r"]
        is_call = chain["is_call"]
        market_price = chain["market_price"]

        if len(S) == 0 or len(price_history) < 10:
            return 0.0

        spot = S[0]
        moneyness = np.abs(np.log(K / spot))
        atm_mask = moneyness < 0.05
        if atm_mask.sum() == 0:
            atm_mask = moneyness < 0.10
        if atm_mask.sum() == 0:
            return 0.0

        # Use custom IV solver for signal with better initial guess
        S_atm = S[atm_mask]
        K_atm = K[atm_mask]
        T_atm = T[atm_mask]
        mp_atm = market_price[atm_mask]
        ic_atm = is_call[atm_mask]
        intrinsic = np.where(ic_atm, np.maximum(S_atm - K_atm, 0), np.maximum(K_atm - S_atm, 0))
        time_val = np.maximum(mp_atm - intrinsic, 0.01)
        sig0 = np.clip(time_val / (S_atm * 0.4 * np.sqrt(np.maximum(T_atm, 1e-4))), 0.05, 2.0)
        # Newton-Raphson with custom start
        sigma_atm = sig0.copy()
        for _ in range(10):
            p = bs_price(S_atm, K_atm, T_atm, r, sigma_atm, ic_atm)
            v = bs_vega(S_atm, K_atm, T_atm, r, sigma_atm)
            diff = p - mp_atm
            upd = v > 1e-12
            sigma_atm[upd] -= diff[upd] / v[upd]
            sigma_atm = np.clip(sigma_atm, 0.01, 5.0)
        atm_ivs = sigma_atm
        current_iv = np.median(atm_ivs)
        iv_std = np.std(atm_ivs)

        log_returns = np.diff(np.log(price_history))
        realized_vol = np.std(log_returns) * np.sqrt(252)
        rv_5d = np.std(log_returns[-5:]) * np.sqrt(252) if len(log_returns) >= 5 else realized_vol

        # OTM put skew
        otm_put = (~is_call) & (K < spot * 0.92)
        if otm_put.sum() > 2:
            otm_ivs = implied_vol_vec(S[otm_put], K[otm_put], T[otm_put], r, market_price[otm_put], is_call[otm_put], max_iter=8)
            put_skew = np.median(otm_ivs) - current_iv
        else:
            put_skew = 0.0

        if realized_vol < 0.01:
            return 0.0
        iv_rv_ratio = current_iv / realized_vol
        ret_5d = (price_history[-1] / price_history[-5]) - 1.0
        low_20d = np.min(price_history[-20:])
        dist_from_low = (price_history[-1] / low_20d) - 1.0

        low_scale = max(0.0, 1.0 - dist_from_low / 0.05)

        ret_10d = (price_history[-1] / price_history[-10]) - 1.0

        # Put-call IV spread (near ATM): put IV vs call IV
        near_atm = moneyness < 0.10
        atm_puts = near_atm & (~is_call)
        atm_calls = near_atm & is_call
        if atm_puts.sum() > 0 and atm_calls.sum() > 0:
            put_iv_med = np.median(implied_vol_vec(S[atm_puts], K[atm_puts], T[atm_puts], r,
                                                    market_price[atm_puts], is_call[atm_puts], max_iter=8))
            call_iv_med = np.median(implied_vol_vec(S[atm_calls], K[atm_calls], T[atm_calls], r,
                                                     market_price[atm_calls], is_call[atm_calls], max_iter=8))
            pc_spread = (put_iv_med - call_iv_med) / (current_iv + 1e-8)
            pc_boost = min(0.75, max(0.0, pc_spread * 80.0))
        else:
            pc_boost = 0.0

        # Normalized skew: put_skew / current_iv captures relative fear level
        norm_skew = put_skew / (current_iv + 1e-8)
        skew_boost = min(0.5, max(0.0, norm_skew * 1.3))
        # IV coherence: penalize when ATM IVs are very dispersed
        coherence = max(0.0, 1.0 - iv_std / (current_iv + 1e-8)) ** 1.2
        # IV term structure boost: short-term vs long-term ATM IV
        short_T = T[atm_mask] < 25/252
        long_T = T[atm_mask] > 40/252
        if short_T.sum() > 0 and long_T.sum() > 0:
            term_spread = np.median(atm_ivs[short_T]) - np.median(atm_ivs[long_T])
            term_boost = min(1.0, max(0.0, term_spread * 100))
        else:
            term_boost = 0.0
        # Short-term RV spike boost: recent vol > longer-term vol
        rv_spike = min(0.3, max(0.0, (rv_5d / (realized_vol + 1e-8) - 1.0) * 0.5))
        if iv_rv_ratio > 1.85 and ret_5d < -0.015 and dist_from_low < 0.035:
            accel1 = min(0.12, max(0.0, ret_5d / (ret_10d + 1e-8) - 0.4) * 0.7) if ret_10d < -0.01 else 0.0
            return (0.16 + skew_boost + accel1 + pc_boost + rv_spike) * low_scale * coherence
        elif iv_rv_ratio > 1.6 and ret_5d < -0.040 and dist_from_low < 0.035:
            return (0.24 + pc_boost + rv_spike) * low_scale * coherence
        elif iv_rv_ratio > 1.5 and ret_10d < -0.06 and dist_from_low < 0.02 and ret_5d < -0.005:
            # Acceleration: if most of the 10d loss is in last 5d, more recent = better
            accel = min(0.30, max(0.0, ret_5d / (ret_10d + 1e-8) - 0.3) * 0.9) if ret_10d < -0.01 else 0.0
            return (skew_boost + term_boost + accel) * low_scale * coherence
        else:
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
