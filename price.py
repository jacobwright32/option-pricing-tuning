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

def implied_vol_vec(S, K, T, r, market_price, is_call, max_iter=15, tol=1e-5):
    """
    Fully vectorized Newton-Raphson IV solver.
    Operates on entire arrays at once — no Python loops.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    market_price = np.asarray(market_price, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)

    sigma = np.full_like(S, 0.30)  # initial guess
    active = np.ones(len(S), dtype=bool)

    for _ in range(max_iter):
        if not active.any():
            break
        price = bs_price(S[active], K[active], T[active], r, sigma[active], is_call[active])
        vega = bs_vega(S[active], K[active], T[active], r, sigma[active])
        diff = price - market_price[active]

        # Update where vega is meaningful
        updatable = vega > 1e-12
        converged = np.abs(diff) < tol

        # Apply Newton step
        step = np.zeros_like(diff)
        step[updatable] = diff[updatable] / vega[updatable]
        new_sigma = sigma[active] - step
        sigma[active] = np.clip(new_sigma, 0.01, 5.0)

        # Mark converged or zero-vega as done
        done_mask = converged | ~updatable
        idx = np.where(active)[0]
        active[idx[done_mask]] = False

    return np.clip(sigma, 0.01, 5.0)


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
        self.long_threshold = 0.25      # go long when IV/RV ratio below (1 - this)
        self.short_threshold = 0.75     # go short when IV/RV ratio above (1 + this)
        self.signal_strength = 0.5      # base signal magnitude

    def price_chain(self, chain):
        """
        Price an entire options chain for one asset on one day.

        Uses a quadratic vol smile fit per expiry bucket:
            σ(m) = a + b·m + c·m²
        where m = ln(K/S). This captures skew and smile per maturity.
        """
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
            log_m,
            log_m ** 2,
            sqrt_T,
            log_m * sqrt_T,
        ])

        def _irls_fit(X_sub, y_sub):
            w = np.ones(len(y_sub))
            coeffs = None
            for _ in range(15):
                Xw = X_sub * w[:, None]
                yw = y_sub * w
                coeffs, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
                resid = y_sub - X_sub @ coeffs
                mad = np.median(np.abs(resid)) + 1e-8
                w = 1.0 / (1.0 + (resid / (3 * mad)) ** 2)
            return coeffs

        # Fit separate surfaces for calls and puts
        fitted_vol = np.copy(ivs)
        try:
            call_mask = is_call
            put_mask = ~is_call
            if call_mask.sum() >= 5:
                c_coeffs = _irls_fit(X[call_mask], ivs[call_mask])
                fitted_vol[call_mask] = X[call_mask] @ c_coeffs
            if put_mask.sum() >= 5:
                p_coeffs = _irls_fit(X[put_mask], ivs[put_mask])
                fitted_vol[put_mask] = X[put_mask] @ p_coeffs
            fitted_vol = np.clip(fitted_vol, 0.01, 5.0)
        except Exception:
            fitted_vol = ivs

        fair_values = bs_price(S, K, T, r, fitted_vol, is_call)
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

        if len(S) == 0 or len(price_history) < 10:
            return 0.0

        # ── Compute ATM implied vol (vectorized) ──
        spot = S[0]
        moneyness = np.abs(np.log(K / spot))
        atm_mask = (moneyness < 0.05) & (T > 20 / 252) & (T < 45 / 252)
        if atm_mask.sum() == 0:
            atm_mask = moneyness < 0.10
        if atm_mask.sum() == 0:
            return 0.0

        atm_ivs = implied_vol_vec(
            S[atm_mask], K[atm_mask], T[atm_mask], r,
            market_price[atm_mask], is_call[atm_mask]
        )
        current_iv = np.median(atm_ivs)

        # ── Compute realized vol from price history ──
        log_returns = np.diff(np.log(price_history))
        realized_vol = np.std(log_returns) * np.sqrt(252)

        # ── Simple IV vs RV signal ──
        # When IV >> RV: market pricing more vol than realized → fearful → contrarian long
        # When IV << RV: market complacent → potential vol expansion → short
        iv_rv_ratio = current_iv / max(realized_vol, 0.01)

        # Short-only: low IV/RV → complacent → short
        if iv_rv_ratio < (1.0 - self.long_threshold):
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
