"""
Option Pricing Model & CFD Signal Generator
============================================
THIS IS THE ONLY FILE THE AGENT MAY EDIT.

Baseline: Black-Scholes with flat volatility estimate + simple IV-rank signal.
The agent should iterate on both pricing accuracy and signal quality.

Ideas the agent might explore:
    - Vol surface fitting (SVI, SABR, polynomial)
    - Jump-diffusion or stochastic vol corrections
    - Better IV solver (Brent, rational approximation)
    - Skew/smile-aware pricing
    - Signal: IV percentile rank, put-call skew, term structure slope
    - Signal: mispricing z-score, vol regime detection
    - Position sizing: Kelly criterion, vol-targeting
    - Combine multiple signals with learned weights
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


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

def implied_vol(S, K, T, r, market_price, is_call, max_iter=50, tol=1e-6):
    """
    Newton-Raphson IV solver for a single option.
    Falls back to Brent's method if Newton fails.
    """
    # Intrinsic value check
    if is_call:
        intrinsic = max(S - K * np.exp(-r * T), 0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0)
    if market_price <= intrinsic + 1e-10:
        return 0.05  # floor

    # Newton-Raphson
    sigma = 0.30  # initial guess
    for _ in range(max_iter):
        price = float(bs_price(S, K, T, r, sigma, is_call))
        vega = float(bs_vega(S, K, T, r, sigma))
        if vega < 1e-12:
            break
        diff = price - market_price
        if abs(diff) < tol:
            return max(sigma, 0.01)
        sigma -= diff / vega
        sigma = max(sigma, 0.01)
        sigma = min(sigma, 5.0)

    # Fallback: Brent's method
    try:
        def objective(s):
            return float(bs_price(S, K, T, r, s, is_call)) - market_price
        sigma = brentq(objective, 0.01, 5.0, xtol=tol)
    except (ValueError, RuntimeError):
        sigma = 0.30  # give up, return default

    return max(sigma, 0.01)


def implied_vol_vec(S, K, T, r, market_price, is_call):
    """Vectorized IV solver — loops over individual options."""
    n = len(S) if hasattr(S, '__len__') else 1
    if n == 1:
        return np.array([implied_vol(S, K, T, r, market_price, is_call)])
    ivs = np.zeros(n)
    for i in range(n):
        s = S[i] if hasattr(S, '__len__') else S
        k = K[i] if hasattr(K, '__len__') else K
        t = T[i] if hasattr(T, '__len__') else T
        mp = market_price[i] if hasattr(market_price, '__len__') else market_price
        ic = is_call[i] if hasattr(is_call, '__len__') else is_call
        ivs[i] = implied_vol(s, k, t, r, mp, ic)
    return ivs


# ─── Pricing Model ───────────────────────────────────────────────────────────

class PricingModel:
    """
    The agent optimizes this class.

    Must implement:
        price_chain(chain) -> array of fair values
        generate_signal(chain, price_history) -> float in [-1, 1]
    """

    def __init__(self):
        # ── Pricing parameters ──
        self.default_vol = 0.30  # flat vol assumption (baseline)

        # ── Signal parameters ──
        self.iv_lookback = 60           # rolling window for IV percentile
        self.long_threshold = 0.25      # go long when IV percentile below this
        self.short_threshold = 0.75     # go short when IV percentile above this
        self.signal_strength = 0.5      # base signal magnitude

        # State for rolling IV tracking
        self._iv_history = {}  # asset_key -> list of ATM IVs

    def price_chain(self, chain):
        """
        Price an entire options chain for one asset on one day.

        chain: dict with keys:
            S            (n,) underlying prices (all same value)
            K            (n,) strike prices
            T            (n,) time to expiry in years
            r            float risk-free rate
            is_call      (n,) bool
            market_price (n,) observed market prices

        Returns: (n,) array of model fair values.
        """
        S = chain["S"]
        K = chain["K"]
        T = chain["T"]
        r = chain["r"]
        is_call = chain["is_call"]
        market_price = chain["market_price"]

        # Baseline: compute IV from each market price, then use flat average vol
        # to re-price. This is barely better than returning market_price directly.
        ivs = implied_vol_vec(S, K, T, r, market_price, is_call)
        avg_vol = np.median(ivs)  # single flat vol for the whole chain

        # Re-price with the flat vol estimate
        fair_values = bs_price(S, K, T, r, avg_vol, is_call)
        return np.maximum(fair_values, 0.01)

    def generate_signal(self, chain, price_history):
        """
        Generate a CFD trading signal from options data + price history.

        chain:         dict with same keys as price_chain
        price_history: (n_days,) array of recent underlying prices

        Returns: float in [-1, 1]
            positive = go long the underlying via CFD
            negative = go short the underlying via CFD
            0 = no position
        """
        S = chain["S"]
        K = chain["K"]
        T = chain["T"]
        r = chain["r"]
        is_call = chain["is_call"]
        market_price = chain["market_price"]

        if len(S) == 0 or len(price_history) < 5:
            return 0.0

        # ── Compute ATM implied vol ──
        spot = S[0]
        moneyness = np.abs(np.log(K / spot))
        # Find near-ATM, ~30-day options
        atm_mask = (moneyness < 0.05) & (T > 20 / 252) & (T < 45 / 252)
        if atm_mask.sum() == 0:
            # Broaden search
            atm_mask = moneyness < 0.10
        if atm_mask.sum() == 0:
            return 0.0

        atm_ivs = implied_vol_vec(
            S[atm_mask], K[atm_mask], T[atm_mask], r,
            market_price[atm_mask], is_call[atm_mask]
        )
        current_iv = np.median(atm_ivs)

        # ── Compute realized vol from price history ──
        if len(price_history) < 10:
            return 0.0
        log_returns = np.diff(np.log(price_history))
        realized_vol = np.std(log_returns) * np.sqrt(252)

        # ── Simple IV vs RV signal ──
        # When IV >> RV: market is pricing in more vol than realized → mean reversion
        # likely → sell vol → but for CFD we interpret as: market is fearful → contrarian long
        # When IV << RV: market is complacent → potential for vol expansion → short
        iv_rv_ratio = current_iv / max(realized_vol, 0.01)

        if iv_rv_ratio > (1.0 + self.short_threshold):
            # IV much higher than RV → market fearful → contrarian long
            signal = self.signal_strength
        elif iv_rv_ratio < (1.0 - self.long_threshold):
            # IV much lower than RV → complacent → cautious short
            signal = -self.signal_strength
        else:
            signal = 0.0

        return np.clip(signal, -1.0, 1.0)


# ─── Main Entry Point ────────────────────────────────────────────────────────

def main():
    """Run the model and print evaluation results."""
    import time as _time
    from prepare import evaluate

    print("=" * 60)
    print("Option Pricing Tuning — Experiment Run")
    print("=" * 60)

    t0 = _time.time()
    model = PricingModel()
    score = evaluate(model)
    elapsed = _time.time() - t0

    print("=" * 60)
    print(f"Wall time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
