"""
Option Pricing Tuning – Data & Evaluation Pipeline
===================================================
READ-ONLY: The autonomous agent must NEVER modify this file.

Generates a synthetic options market with realistic volatility surfaces,
evaluates pricing accuracy, simulates Trading 212-style CFD trades from
model signals, and returns a combined score.

The ground truth uses asset-specific vol surfaces with skew, smile, and
term structure — Black-Scholes with flat vol will systematically misprice,
giving the agent room to improve.
"""

import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

# ─── Configuration ────────────────────────────────────────────────────────────

CACHE_DIR = Path.home() / ".cache" / "option-pricing-tuning"
DATA_FILE = CACHE_DIR / "market_data.npz"
TIME_BUDGET = 120  # seconds per experiment run
SEED = 42

N_ASSETS = 20
N_DAYS = 504            # ~2 trading years
SNAPSHOT_EVERY = 5      # option snapshots every 5 days
LOOKBACK = 60           # price history provided to signal generator

STRIKES_PCT = np.array(
    [0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 1.00, 1.03, 1.05, 1.07, 1.10, 1.15, 1.20]
)
EXPIRIES_DAYS = np.array([7, 14, 30, 60, 90, 180])
RISK_FREE_RATE = 0.045

# Trading 212 CFD costs
CFD_SPREAD_PCT = 0.0010     # 0.10% one-way
CFD_OVERNIGHT_PCT = 0.00008 # per day (~2.9% annualized)
HOLD_DAYS = 5               # hold between snapshots
INITIAL_CAPITAL = 10_000.0
MAX_POSITION_PCT = 0.10     # max 10% of capital per asset

# ─── Black-Scholes (internal, for data generation) ───────────────────────────

def _bs_price(S, K, T, r, sigma, is_call):
    """Vectorized Black-Scholes European option pricing."""
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-10)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return np.where(is_call, call_price, put_price)


def _true_vol(log_moneyness, T_years, params):
    """
    Ground-truth implied volatility surface per asset.

        σ(m, T) = σ₀ + skew · ln(K/S) / √T + smile · ln(K/S)² + term · (√T − 0.3)

    Each asset has unique (σ₀, skew, smile, term) parameters, producing
    realistic equity-like vol surfaces with negative skew and convex smile.
    """
    sigma0, skew, smile, term = params
    sqrt_T = np.sqrt(np.maximum(T_years, 1.0 / 252))
    vol = sigma0 + skew * log_moneyness / sqrt_T + smile * log_moneyness ** 2 + term * (sqrt_T - 0.3)
    return np.clip(vol, 0.05, 2.0)

# ─── Data Generation ─────────────────────────────────────────────────────────

def _generate_dataset():
    """
    Generate the full synthetic market dataset.

    Returns dict with:
        prices          (N_ASSETS, N_DAYS)       underlying daily closes
        vol_params      (N_ASSETS, 4)            true vol surface parameters
        snapshot_days   (N_SNAPSHOTS,)            day indices of option snapshots
        opt_asset       (N_OPTIONS,)              asset index per option
        opt_snap        (N_OPTIONS,)              snapshot index per option
        opt_S           (N_OPTIONS,)              underlying price
        opt_K           (N_OPTIONS,)              strike price
        opt_T           (N_OPTIONS,)              time to expiry in years
        opt_is_call     (N_OPTIONS,)              True = call, False = put
        opt_true_iv     (N_OPTIONS,)              true implied vol
        opt_true_price  (N_OPTIONS,)              true option price (from vol surface)
        opt_market_price(N_OPTIONS,)              noisy "market" price
    """
    rng = np.random.default_rng(SEED)

    # ── 1. Simulate underlying prices ──
    prices = np.zeros((N_ASSETS, N_DAYS))
    for i in range(N_ASSETS):
        S0 = rng.uniform(30, 500)
        mu = rng.uniform(0.02, 0.15)       # annual drift
        base_vol = rng.uniform(0.15, 0.50)  # base annual vol

        # Mean-reverting stochastic vol (Ornstein-Uhlenbeck on log-vol)
        log_v = np.log(base_vol)
        kappa = rng.uniform(2.0, 8.0)       # mean-reversion speed
        xi = rng.uniform(0.3, 0.8)          # vol of vol
        S = S0
        for t in range(N_DAYS):
            log_v += kappa * (np.log(base_vol) - log_v) / 252 + xi / np.sqrt(252) * rng.standard_normal()
            v = np.exp(np.clip(log_v, np.log(0.05), np.log(1.5)))
            S *= np.exp((mu - 0.5 * v ** 2) / 252 + v / np.sqrt(252) * rng.standard_normal())
            prices[i, t] = S

    # ── 2. Vol surface parameters per asset ──
    vol_params = np.zeros((N_ASSETS, 4))
    for i in range(N_ASSETS):
        rv = np.std(np.diff(np.log(prices[i]))) * np.sqrt(252)  # realized vol
        vol_params[i] = [
            rv * rng.uniform(0.9, 1.3),          # σ₀: base IV (near realized)
            rng.uniform(-0.20, -0.05),            # skew: negative (equity-like)
            rng.uniform(0.05, 0.25),              # smile: convex
            rng.uniform(-0.03, 0.03),             # term structure tilt
        ]

    # ── 3. Generate option snapshots ──
    snapshot_days = np.arange(LOOKBACK, N_DAYS - HOLD_DAYS, SNAPSHOT_EVERY)
    n_snaps = len(snapshot_days)
    n_strikes = len(STRIKES_PCT)
    n_expiries = len(EXPIRIES_DAYS)
    n_per_snap = n_strikes * n_expiries * 2  # calls + puts
    n_total = N_ASSETS * n_snaps * n_per_snap

    opt_asset = np.zeros(n_total, dtype=np.int32)
    opt_snap = np.zeros(n_total, dtype=np.int32)
    opt_S = np.zeros(n_total)
    opt_K = np.zeros(n_total)
    opt_T = np.zeros(n_total)
    opt_is_call = np.zeros(n_total, dtype=bool)
    opt_true_iv = np.zeros(n_total)
    opt_true_price = np.zeros(n_total)
    opt_market_price = np.zeros(n_total)

    idx = 0
    for ai in range(N_ASSETS):
        params = vol_params[ai]
        for si, day in enumerate(snapshot_days):
            S = prices[ai, day]
            for ki, strike_pct in enumerate(STRIKES_PCT):
                K = S * strike_pct
                for ei, exp_d in enumerate(EXPIRIES_DAYS):
                    T = exp_d / 252.0
                    log_m = np.log(K / S)
                    for is_call in [True, False]:
                        iv = _true_vol(log_m, T, params)
                        tp = _bs_price(S, K, T, RISK_FREE_RATE, iv, is_call)
                        tp = max(tp, 0.01)

                        # Market price: true + bid-ask noise + microstructure
                        noise_pct = rng.uniform(0.005, 0.03)  # 0.5-3% noise
                        mp = tp * (1.0 + noise_pct * rng.standard_normal())
                        mp = max(mp, 0.01)

                        opt_asset[idx] = ai
                        opt_snap[idx] = si
                        opt_S[idx] = S
                        opt_K[idx] = K
                        opt_T[idx] = T
                        opt_is_call[idx] = is_call
                        opt_true_iv[idx] = iv
                        opt_true_price[idx] = tp
                        opt_market_price[idx] = mp
                        idx += 1

    return {
        "prices": prices,
        "vol_params": vol_params,
        "snapshot_days": snapshot_days,
        "opt_asset": opt_asset,
        "opt_snap": opt_snap,
        "opt_S": opt_S,
        "opt_K": opt_K,
        "opt_T": opt_T,
        "opt_is_call": opt_is_call,
        "opt_true_iv": opt_true_iv,
        "opt_true_price": opt_true_price,
        "opt_market_price": opt_market_price,
    }


def load_data():
    """Load cached dataset or generate fresh."""
    if DATA_FILE.exists():
        data = dict(np.load(DATA_FILE, allow_pickle=False))
        # Restore bool dtype
        data["opt_is_call"] = data["opt_is_call"].astype(bool)
        return data

    print("Generating synthetic market data (first run only)...")
    t0 = time.time()
    data = _generate_dataset()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATA_FILE, **data)
    print(f"Dataset cached to {DATA_FILE} ({time.time() - t0:.1f}s)")
    return data

# ─── CFD Trade Simulation ────────────────────────────────────────────────────

def simulate_cfd_trades(signals, prices, snapshot_days):
    """
    Simulate Trading 212-style CFD trades from model signals.

    Args:
        signals:       (N_ASSETS, n_snapshots) float in [-1, 1]
                       positive = long, negative = short, 0 = flat
        prices:        (N_ASSETS, N_DAYS) underlying daily closes
        snapshot_days: (n_snapshots,) day indices

    Returns:
        dict with: sharpe, total_return, max_drawdown, win_rate, n_trades,
                   daily_pnl (array)
    """
    n_assets, n_snaps = signals.shape
    capital = INITIAL_CAPITAL
    daily_equity = [capital]
    trade_returns = []

    for si in range(n_snaps):
        day = snapshot_days[si]
        exit_day = min(day + HOLD_DAYS, prices.shape[1] - 1)
        if exit_day <= day:
            continue

        snap_pnl = 0.0
        for ai in range(n_assets):
            sig = signals[ai, si]
            if abs(sig) < 0.05:  # dead zone
                continue

            entry_price = prices[ai, day]
            exit_price = prices[ai, exit_day]
            if entry_price <= 0:
                continue

            # Position size: signal strength × max allocation
            pos_size = abs(sig) * MAX_POSITION_PCT * capital
            direction = np.sign(sig)

            # Entry cost (half-spread)
            effective_entry = entry_price * (1.0 + direction * CFD_SPREAD_PCT)
            effective_exit = exit_price * (1.0 - direction * CFD_SPREAD_PCT)

            # Gross return
            gross_ret = direction * (effective_exit - effective_entry) / entry_price

            # Overnight financing
            hold = exit_day - day
            financing = CFD_OVERNIGHT_PCT * hold

            net_ret = gross_ret - financing
            pnl = pos_size * net_ret
            snap_pnl += pnl
            trade_returns.append(net_ret)

        capital += snap_pnl
        capital = max(capital, 1.0)  # floor to prevent zero
        daily_equity.append(capital)

    daily_equity = np.array(daily_equity)
    equity_returns = np.diff(daily_equity) / daily_equity[:-1]

    # Metrics
    trade_returns = np.array(trade_returns) if trade_returns else np.array([0.0])
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL

    # Sharpe (annualized, using snapshot-frequency returns)
    if len(equity_returns) > 1 and equity_returns.std() > 1e-10:
        periods_per_year = 252 / HOLD_DAYS
        sharpe = equity_returns.mean() / equity_returns.std() * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = np.maximum.accumulate(daily_equity)
    drawdown = (peak - daily_equity) / peak
    max_dd = drawdown.max()

    win_rate = (trade_returns > 0).mean() if len(trade_returns) > 0 else 0.0

    return {
        "sharpe": float(sharpe),
        "total_return": float(total_return),
        "max_drawdown": float(max_dd),
        "win_rate": float(win_rate),
        "n_trades": len(trade_returns),
        "daily_equity": daily_equity,
    }

# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model):
    """
    Full evaluation pipeline — the single metric the agent optimizes.

    The model must implement:
        model.price_chain(chain: dict) -> np.ndarray of fair values
        model.generate_signal(chain: dict, price_history: np.ndarray) -> float in [-1, 1]

    Prints summary to stdout (agent greps for combined_score).
    Returns the combined score (higher is better).
    """
    t0 = time.time()
    data = load_data()

    prices = data["prices"]
    snapshot_days = data["snapshot_days"]
    n_snaps = len(snapshot_days)

    # Pre-build group index: (asset, snapshot) -> array of option indices
    # This avoids repeated O(n) mask lookups in the inner loop.
    group_idx = {}
    for i in range(len(data["opt_asset"])):
        key = (int(data["opt_asset"][i]), int(data["opt_snap"][i]))
        if key not in group_idx:
            group_idx[key] = []
        group_idx[key].append(i)
    group_idx = {k: np.array(v) for k, v in group_idx.items()}

    def _make_chain(idx):
        return {
            "S": data["opt_S"][idx],
            "K": data["opt_K"][idx],
            "T": data["opt_T"][idx],
            "r": RISK_FREE_RATE,
            "is_call": data["opt_is_call"][idx],
            "market_price": data["opt_market_price"][idx],
        }

    # ── 1. Pricing accuracy ──
    model_prices = np.copy(data["opt_market_price"])  # fallback = market price

    for (ai, si), idx in group_idx.items():
        chain = _make_chain(idx)
        try:
            fv = model.price_chain(chain)
            model_prices[idx] = np.asarray(fv).flatten()
        except Exception:
            pass  # keeps market_price fallback

    # MAPE against TRUE prices (not market prices)
    true_p = data["opt_true_price"]
    mape = np.mean(np.abs(model_prices - true_p) / np.maximum(true_p, 0.01))

    # RMSE normalized by mean price
    rmse = np.sqrt(np.mean((model_prices - true_p) ** 2))
    rmse_pct = rmse / np.mean(true_p)

    # ── 2. Signal generation ──
    signals = np.zeros((N_ASSETS, n_snaps))
    for ai in range(N_ASSETS):
        for si, day in enumerate(snapshot_days):
            key = (ai, si)
            if key not in group_idx:
                continue
            chain = _make_chain(group_idx[key])
            history_start = max(0, day - LOOKBACK)
            price_history = prices[ai, history_start:day + 1]
            try:
                sig = model.generate_signal(chain, price_history)
                signals[ai, si] = float(np.clip(sig, -1.0, 1.0))
            except Exception:
                signals[ai, si] = 0.0

    # ── 3. CFD simulation ──
    cfd = simulate_cfd_trades(signals, prices, snapshot_days)

    # ── 4. Combined score ──
    # Higher is better. Pricing accuracy is rewarded (lower MAPE = higher score),
    # signal profitability is rewarded (higher Sharpe = higher score).
    # Weighting: 40% pricing, 60% signals (because the user trades CFDs).
    pricing_score = max(0, 1.0 - mape)  # 1.0 = perfect, 0.0 = 100% MAPE
    signal_score = cfd["sharpe"]         # typically -2 to +3

    combined = 0.4 * pricing_score + 0.6 * signal_score

    elapsed = time.time() - t0

    # ── Print results (agent greps these) ──
    print(f"combined_score {combined:.6f}")
    print(f"pricing_mape {mape:.6f}")
    print(f"pricing_rmse_pct {rmse_pct:.6f}")
    print(f"sharpe_ratio {cfd['sharpe']:.6f}")
    print(f"total_return {cfd['total_return']:.6f}")
    print(f"max_drawdown {cfd['max_drawdown']:.6f}")
    print(f"win_rate {cfd['win_rate']:.6f}")
    print(f"n_trades {cfd['n_trades']}")
    print(f"eval_time {elapsed:.1f}s")

    return combined
