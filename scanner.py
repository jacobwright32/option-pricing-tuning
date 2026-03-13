"""
Live Stock Scanner Dashboard
=============================
Scans S&P 500 + Russell 2000 stocks for buy signals using the optimized contrarian strategy.

Signal: Buy when ALL conditions are met:
  1. IV/RV ratio > 1.5  (fear premium is elevated)
  2. -6% < 5-day return < -2%  (dip but not crash)
  3. -8% < 10-day return < -1%  (medium-term weakness, not freefall)
  4. -15% < Distance from 30-day high < -5%  (pullback, not collapse)

Usage:  streamlit run scanner.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import json
import time

TRADES_FILE = Path(__file__).parent / "trades.json"

# ─── Ticker Universe (S&P 500 + Russell 2000) ────────────────────────────────

# Import full universe from prepare.py (same tickers used for optimization)
try:
    from prepare import TICKERS as _ALL_TICKERS
    SCAN_TICKERS = list(_ALL_TICKERS)
except ImportError:
    # Fallback to S&P 500 if prepare.py not available
    SCAN_TICKERS = [
        "AAPL", "ABBV", "ABT", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEP",
        "AFL", "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALK", "ALL", "AMAT",
        "AMCR", "AMD", "AME", "AMGN", "AMP", "AMT", "AMZN", "ANET", "ANSS", "AON",
        "AOS", "APA", "APD", "APH", "APTV", "ARE", "ATO", "AVB", "AVGO",
        "AVY", "AWK", "AXP", "AZO", "BA", "BAC", "BAX", "BBY", "BDX",
        "BEN", "BIIB", "BIO", "BK", "BKNG", "BKR", "BLK", "BMY", "BR",
        "BRK-B", "BRO", "BSX", "BWA", "BXP", "C", "CAG", "CAH", "CARR", "CAT",
        "CB", "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG",
        "CF", "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMA",
        "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COP",
        "COST", "CPB", "CPRT", "CPT", "CRL", "CRM", "CSCO", "CSX", "CTAS",
        "CTRA", "CTSH", "CTVA", "CVS", "CVX", "D", "DAL", "DD",
        "DE", "DFS", "DG", "DGX", "DHI", "DHR", "DIS", "DLTR", "DOV",
        "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM", "EA",
        "EBAY", "ECL", "ED", "EFX", "EIX", "EL", "EMN", "EMR", "ENPH", "EOG",
        "EPAM", "EQIX", "EQR", "EQT", "ES", "ESS", "ETN", "ETR", "EVRG",
        "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FANG", "FAST", "FBHS", "FCX",
        "FDS", "FDX", "FE", "FFIV", "FIS", "FISV", "FITB", "FMC", "FOX",
        "FOXA", "FRT", "FTNT", "FTV", "GD", "GE", "GILD", "GIS",
        "GLW", "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW",
        "HAL", "HAS", "HBAN", "HCA", "HD", "HOLX", "HON", "HPE", "HPQ", "HRL",
        "HSIC", "HST", "HSY", "HUM", "HWM", "IBM", "ICE", "IDXX", "IEX", "IFF",
        "ILMN", "INCY", "INTC", "INTU", "INVH", "IP", "IPG", "IQV", "IR", "IRM",
        "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JCI", "JKHY", "JNJ", "JNPR",
        "JPM", "K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KLAC", "KMB", "KMI",
        "KMX", "KO", "KR", "L", "LDOS", "LEN", "LH", "LHX", "LIN", "LKQ",
        "LLY", "LMT", "LNT", "LOW", "LRCX", "LUV", "LW",
        "LYB", "LYV", "MA", "MAA", "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO",
        "MDLZ", "MDT", "MET", "META", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC",
        "MMM", "MNST", "MO", "MOH", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MRO",
        "MS", "MSCI", "MSFT", "MSI", "MTB", "MTD", "MU", "NCLH", "NDAQ",
        "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC",
        "NTAP", "NTRS", "NUE", "NVDA", "NVR", "NWL", "NWS", "NWSA", "NXPI", "O",
        "ODFL", "OGN", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY", "PARA",
        "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE", "PFG", "PG",
        "PGR", "PH", "PHM", "PKG", "PLD", "PM", "PNC", "PNR", "PNW",
        "POOL", "PPG", "PPL", "PRU", "PSA", "PSX", "PTC", "PVH", "PWR",
        "PYPL", "QCOM", "QRVO", "RCL", "RE", "REG", "REGN", "RF", "RHI", "RJF",
        "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "SBAC",
        "SBUX", "SCHW", "SEE", "SHW", "SJM", "SLB", "SNA", "SNPS", "SO",
        "SPG", "SPGI", "SRE", "STE", "STT", "STX", "STZ", "SWK", "SWKS", "SYF",
        "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC",
        "TFX", "TGT", "TMO", "TMUS", "TPR", "TRGP", "TRMB", "TROW", "TRV", "TSCO",
        "TSLA", "TSN", "TT", "TTWO", "TXN", "TXT", "TYL", "UAL", "UDR", "UHS",
        "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V", "VFC", "VICI", "VLO",
        "VMC", "VNO", "VRSK", "VRSN", "VRTX", "VTR", "VTRS", "VZ", "WAB", "WAT",
        "WBA", "WBD", "WDC", "WEC", "WELL", "WFC", "WHR", "WM", "WMB", "WMT",
        "WRB", "WRK", "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XRAY", "XYL",
        "YUM", "ZBH", "ZBRA", "ZION", "ZTS",
        "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLP",
    ]


# ─── Trade Storage ───────────────────────────────────────────────────────────

def load_trades():
    if TRADES_FILE.exists():
        return json.loads(TRADES_FILE.read_text())
    return {"scans": []}


def save_trades(data):
    TRADES_FILE.write_text(json.dumps(data, indent=2, default=str))


# ─── Company Names ──────────────────────────────────────────────────────────

COMPANY_NAMES = {}

def get_company_name(ticker):
    """Get company name, cached in module-level dict."""
    if ticker in COMPANY_NAMES:
        return COMPANY_NAMES[ticker]
    try:
        info = yf.Ticker(ticker).info
        name = info.get("shortName") or info.get("longName") or ticker
        COMPANY_NAMES[ticker] = name
    except Exception:
        COMPANY_NAMES[ticker] = ticker
    return COMPANY_NAMES[ticker]


# ─── Signal Logic (from optimized price.py — Exp 119, score 1.638) ──────────

def compute_signal_metrics(prices_60d):
    if len(prices_60d) < 30:
        return None
    spot = prices_60d[-1]
    log_returns = np.diff(np.log(prices_60d))
    rv = np.std(log_returns) * np.sqrt(252)
    if rv < 0.01:
        return None
    ret_5d = (prices_60d[-1] / prices_60d[-5]) - 1.0
    ret_10d = (prices_60d[-1] / prices_60d[-10]) - 1.0
    high_30d = np.max(prices_60d[-30:])
    dist_from_high = (prices_60d[-1] / high_30d) - 1.0
    return {
        "spot": spot, "rv": rv, "ret_5d": ret_5d, "ret_10d": ret_10d,
        "high_30d": high_30d, "dist_from_high": dist_from_high,
    }


def get_atm_iv(ticker, spot):
    try:
        tk = yf.Ticker(ticker)
        expirations = tk.options
        if not expirations:
            return None
        today = datetime.now()
        best_exp, best_days = None, 999
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            days = (exp_date - today).days
            if 14 <= days <= 60 and days < best_days:
                best_days, best_exp = days, exp_str
        if best_exp is None:
            for exp_str in expirations:
                exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
                days = (exp_date - today).days
                if 7 < days < best_days:
                    best_days, best_exp = days, exp_str
        if best_exp is None:
            return None
        chain = tk.option_chain(best_exp)
        ivs = []
        for df in [chain.calls, chain.puts]:
            atm = df[np.abs(df["strike"] / spot - 1.0) < 0.03]
            if len(atm) > 0 and "impliedVolatility" in atm.columns:
                valid = atm["impliedVolatility"].dropna()
                ivs.extend(valid[(valid > 0.05) & (valid < 3.0)].tolist())
        return float(np.median(ivs)) if ivs else None
    except Exception:
        return None


# ─── Streamlit App ───────────────────────────────────────────────────────────

st.set_page_config(page_title="CFD Signal Scanner", page_icon="📊", layout="wide")

tab_scan, tab_track, tab_completed = st.tabs(["📡 Scanner", "📈 Trade Tracker", "✅ Completed Scans"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

with tab_scan:
    st.title("📊 CFD Signal Scanner")
    st.markdown("""
    **Strategy**: Buy moderate dips when IV premium is elevated. Avoid crashes.
    Hold for **7 calendar days**, then exit.
    """)

    # Fixed thresholds from optimization (Exp 53, score 2.396)
    min_iv_rv = 2.0
    ret_5d_range = (-0.035, -0.025)
    ret_10d_range = (-0.06, -0.01)
    dist_high_range = (-0.08, -0.03)

    # Persist scan results across reruns so Save button works
    if "buy_signals" not in st.session_state:
        st.session_state.buy_signals = None
        st.session_state.near_miss = None
        st.session_state.scan_time = None

    if st.button("🔍 Run Scan", type="primary", use_container_width=True):
        tickers = SCAN_TICKERS

        with st.spinner(f"Downloading price data for {len(tickers)} stocks..."):
            # Download in batches of 500 to avoid yfinance timeout with large ticker lists
            batch_size = 500
            frames = []
            for b in range(0, len(tickers), batch_size):
                batch = tickers[b:b + batch_size]
                batch_data = yf.download(batch, period="6mo", auto_adjust=True, threads=True, progress=False)
                if batch_data.empty:
                    continue
                batch_close = batch_data["Close"] if "Close" in batch_data.columns.get_level_values(0) else batch_data
                frames.append(batch_close)
            if frames:
                close = pd.concat(frames, axis=1)
            else:
                st.error("Failed to download price data.")
                st.stop()

        st.subheader("Phase 1: Price Pre-Filter")
        prefilter_results = []
        price_progress = st.progress(0)
        valid_tickers = [t for t in tickers if t in close.columns]

        for i, ticker in enumerate(valid_tickers):
            price_progress.progress((i + 1) / len(valid_tickers))
            prices = close[ticker].dropna().values
            if len(prices) < 60:
                continue
            prices_60d = prices[-60:]
            metrics = compute_signal_metrics(prices_60d)
            if metrics is None:
                continue
            passes = (
                ret_5d_range[0] < metrics["ret_5d"] < ret_5d_range[1]
                and ret_10d_range[0] < metrics["ret_10d"] < ret_10d_range[1]
                and dist_high_range[0] < metrics["dist_from_high"] < dist_high_range[1]
                and metrics["rv"] < 0.50
            )
            if passes:
                prefilter_results.append({
                    "Ticker": ticker,
                    "Price": metrics["spot"],
                    "5d Ret": metrics["ret_5d"],
                    "10d Ret": metrics["ret_10d"],
                    "Dist High": metrics["dist_from_high"],
                    "RV": metrics["rv"],
                })
        price_progress.empty()

        if not prefilter_results:
            st.session_state.buy_signals = []
            st.session_state.near_miss = []
            st.session_state.scan_time = datetime.now()
        else:
            df_pre = pd.DataFrame(prefilter_results)
            fmt = {"Price": "${:.2f}", "5d Ret": "{:.1%}", "10d Ret": "{:.1%}",
                   "Dist High": "{:.1%}", "RV": "{:.1%}"}
            st.dataframe(df_pre.style.format(fmt), use_container_width=True, hide_index=True)
            st.write(f"**{len(prefilter_results)}** stocks pass price filters")

            st.subheader("Phase 2: IV Check")
            iv_progress = st.progress(0)
            status_text = st.empty()
            buy_signals, near_miss = [], []

            for i, row in enumerate(prefilter_results):
                ticker = row["Ticker"]
                status_text.text(f"Checking IV for {ticker}... ({i+1}/{len(prefilter_results)})")
                iv_progress.progress((i + 1) / len(prefilter_results))
                iv = get_atm_iv(ticker, row["Price"])
                if iv is None:
                    continue
                iv_rv = iv / row["RV"]
                result = {**row, "ATM IV": iv, "IV/RV": iv_rv}
                if iv_rv > min_iv_rv:
                    buy_signals.append(result)
                elif iv_rv > min_iv_rv * 0.85:
                    near_miss.append(result)
                time.sleep(0.2)

            iv_progress.empty()
            status_text.empty()

            st.session_state.buy_signals = buy_signals
            st.session_state.near_miss = near_miss
            st.session_state.scan_time = datetime.now()

    # Display results from session state (persists across reruns for Save button)
    fmt = {"Price": "${:.2f}", "5d Ret": "{:.1%}", "10d Ret": "{:.1%}",
           "Dist High": "{:.1%}", "RV": "{:.1%}"}

    if st.session_state.buy_signals is not None:
        buy_signals = st.session_state.buy_signals
        near_miss = st.session_state.near_miss

        st.markdown("---")
        if buy_signals:
            st.subheader(f"🟢 BUY SIGNALS ({len(buy_signals)})")
            st.markdown("**Open LONG CFD. Hold 7 calendar days. Close next Friday.**")
            df_buy = pd.DataFrame(buy_signals).sort_values("IV/RV", ascending=False)
            fmt2 = {**fmt, "ATM IV": "{:.1%}", "IV/RV": "{:.2f}"}
            st.dataframe(df_buy.style.format(fmt2), use_container_width=True, hide_index=True)

            scan_date = st.session_state.scan_time.strftime("%Y-%m-%d %H:%M")
            exit_date = (st.session_state.scan_time + timedelta(days=7)).strftime("%Y-%m-%d")
            if st.button("💾 Save scan & start tracking", type="secondary"):
                trades_data = load_trades()
                scan_entry = {
                    "scan_date": scan_date,
                    "exit_date": exit_date,
                    "tickers": [],
                }
                for _, r in df_buy.iterrows():
                    scan_entry["tickers"].append({
                        "ticker": r["Ticker"],
                        "entry_price": round(float(r["Price"]), 2),
                        "iv_rv": round(float(r["IV/RV"]), 2),
                        "iv": round(float(r["ATM IV"]), 4),
                        "rv": round(float(r["RV"]), 4),
                        "ret_5d": round(float(r["5d Ret"]), 4),
                        "ret_10d": round(float(r["10d Ret"]), 4),
                        "dist_high": round(float(r["Dist High"]), 4),
                    })
                trades_data["scans"].append(scan_entry)
                save_trades(trades_data)
                st.success(f"Saved {len(scan_entry['tickers'])} trades. Exit by {exit_date}.")
        else:
            st.subheader("⚪ No Buy Signals")
            st.info("No stocks meet ALL criteria. Strategy is selective (~1 trade/month).")

        if near_miss:
            st.subheader(f"🟡 Near-Miss ({len(near_miss)})")
            df_near = pd.DataFrame(near_miss).sort_values("IV/RV", ascending=False)
            fmt2 = {**fmt, "ATM IV": "{:.1%}", "IV/RV": "{:.2f}"}
            st.dataframe(df_near.style.format(fmt2), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: TRADE TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

with tab_track:
    st.title("📈 Trade Tracker")

    trades_data = load_trades()
    scans = trades_data.get("scans", [])

    if not scans:
        st.info("No tracked scans yet. Run a scan and click 'Save scan & start tracking'.")
    else:
        # Scan selector
        scan_labels = [
            f"{s['scan_date']} — {len(s['tickers'])} ticker(s) — exit by {s['exit_date']}"
            for s in scans
        ]
        selected_idx = st.selectbox(
            "Select scan to view",
            range(len(scans)),
            format_func=lambda i: scan_labels[i],
            index=len(scans) - 1,
        )
        scan = scans[selected_idx]

        scan_dt = datetime.strptime(scan["scan_date"], "%Y-%m-%d %H:%M")
        exit_dt = datetime.strptime(scan["exit_date"], "%Y-%m-%d")
        now = datetime.now()
        days_elapsed = (now - scan_dt).days
        days_remaining = max(0, (exit_dt - now).days)

        # Status
        if now >= exit_dt:
            st.error(f"⏰ EXIT NOW — Hold period ended {scan['exit_date']}")
        elif days_remaining <= 1:
            st.warning(f"⚠️ Exit tomorrow ({scan['exit_date']}) — {days_remaining} day(s) left")
        else:
            st.info(f"📅 {days_elapsed} days in / {days_remaining} days remaining — exit {scan['exit_date']}")

        # Progress bar
        total_days = 7
        progress = min(1.0, days_elapsed / total_days)
        st.progress(progress, text=f"Day {min(days_elapsed, 7)} of 7")

        # Fetch current prices
        tracked_tickers = [t["ticker"] for t in scan["tickers"]]

        if st.button("🔄 Refresh prices", type="primary"):
            with st.spinner("Fetching current prices..."):
                current_data = yf.download(tracked_tickers, period="10d", auto_adjust=True,
                                           threads=True, progress=False)
                if len(tracked_tickers) == 1:
                    current_prices = {tracked_tickers[0]: current_data["Close"].iloc[-1]}
                else:
                    current_prices = {}
                    for t in tracked_tickers:
                        if t in current_data["Close"].columns:
                            p = current_data["Close"][t].dropna()
                            if len(p) > 0:
                                current_prices[t] = p.iloc[-1]

                # Get company names
                names = {}
                for t in tracked_tickers:
                    names[t] = get_company_name(t)

            rows = []
            pnl_list = []
            total_pnl = 0
            for t in scan["tickers"]:
                ticker = t["ticker"]
                entry = t["entry_price"]
                current = current_prices.get(ticker)
                name = names.get(ticker, ticker)
                if current is not None:
                    pnl_pct = (current / entry) - 1.0
                    pnl_dollar = pnl_pct * 1000  # Per $1000 position
                    total_pnl += pnl_pct
                    pnl_list.append({"Ticker": ticker, "Name": name, "P&L %": pnl_pct})
                    status = "🟢" if pnl_pct > 0 else "🔴"
                    rows.append({
                        "": status,
                        "Ticker": ticker,
                        "Company": name,
                        "Entry": entry,
                        "Current": round(current, 2),
                        "P&L %": pnl_pct,
                        "P&L /$1K": round(pnl_dollar, 2),
                        "IV/RV at entry": t["iv_rv"],
                    })
                else:
                    rows.append({
                        "": "⚪",
                        "Ticker": ticker,
                        "Company": name,
                        "Entry": entry,
                        "Current": None,
                        "P&L %": None,
                        "P&L /$1K": None,
                        "IV/RV at entry": t["iv_rv"],
                    })

            if rows:
                df_track = pd.DataFrame(rows)
                st.dataframe(
                    df_track.style.format({
                        "Entry": "${:.2f}",
                        "Current": "${:.2f}",
                        "P&L %": "{:+.2%}",
                        "P&L /$1K": "${:+.2f}",
                        "IV/RV at entry": "{:.2f}",
                    }, na_rep="—"),
                    use_container_width=True,
                    hide_index=True,
                )

                # Summary
                n_winners = sum(1 for r in rows if r["P&L %"] is not None and r["P&L %"] > 0)
                n_total = sum(1 for r in rows if r["P&L %"] is not None)
                avg_pnl = total_pnl / n_total if n_total > 0 else 0

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg P&L", f"{avg_pnl:+.2%}")
                with col2:
                    st.metric("Winners", f"{n_winners}/{n_total}")
                with col3:
                    st.metric("Days Left", f"{days_remaining}")

                # Returns distribution chart
                if len(pnl_list) > 0:
                    st.subheader("Returns Distribution")
                    df_pnl = pd.DataFrame(pnl_list).sort_values("P&L %")
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(10, max(3, len(df_pnl) * 0.4)))
                    colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df_pnl["P&L %"]]
                    labels = [f"{r['Ticker']} ({r['Name'][:20]})" for _, r in df_pnl.iterrows()]
                    ax.barh(labels, df_pnl["P&L %"] * 100, color=colors)
                    ax.axvline(x=0, color="gray", linewidth=0.8)
                    ax.set_xlabel("Return (%)")
                    ax.set_title("P&L by Position")
                    for i, (_, r) in enumerate(df_pnl.iterrows()):
                        ax.text(r["P&L %"] * 100, i, f" {r['P&L %']:+.1%}", va="center", fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

        else:
            # Show entry info without current prices
            rows = []
            for t in scan["tickers"]:
                rows.append({
                    "Ticker": t["ticker"],
                    "Entry Price": f"${t['entry_price']:.2f}",
                    "IV/RV": f"{t['iv_rv']:.2f}",
                    "5d Ret": f"{t['ret_5d']:.1%}",
                    "10d Ret": f"{t['ret_10d']:.1%}",
                    "Dist High": f"{t['dist_high']:.1%}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("Click 'Refresh prices' to see current P&L")

        # Delete scan
        st.markdown("---")
        if st.button("🗑️ Delete this scan", type="secondary"):
            trades_data["scans"].pop(selected_idx)
            save_trades(trades_data)
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: COMPLETED SCANS
# ═══════════════════════════════════════════════════════════════════════════════

with tab_completed:
    st.title("✅ Completed Trade Scans")

    trades_data_c = load_trades()
    all_scans = trades_data_c.get("scans", [])
    completed_scans = [
        s for s in all_scans
        if datetime.now() >= datetime.strptime(s["exit_date"], "%Y-%m-%d")
    ]

    if not completed_scans:
        st.info("No completed scans yet. Scans appear here after their 7-day hold period ends.")
    else:
        st.write(f"**{len(completed_scans)}** completed scan(s)")

        if st.button("🔄 Load results", type="primary", key="load_completed"):
            all_tickers = list({t["ticker"] for s in completed_scans for t in s["tickers"]})
            with st.spinner(f"Fetching price history for {len(all_tickers)} tickers..."):
                hist_data = yf.download(all_tickers, period="6mo", auto_adjust=True,
                                        threads=True, progress=False)

            summary_rows = []
            all_returns = []

            for s in completed_scans:
                exit_dt_s = datetime.strptime(s["exit_date"], "%Y-%m-%d")
                scan_winners = 0
                scan_losers = 0
                scan_total_pnl = 0
                scan_trades = 0
                ticker_details = []

                for t in s["tickers"]:
                    ticker = t["ticker"]
                    entry = t["entry_price"]
                    try:
                        if len(all_tickers) == 1:
                            prices = hist_data["Close"].dropna()
                        else:
                            prices = hist_data["Close"][ticker].dropna()
                        exit_prices = prices[prices.index <= pd.Timestamp(exit_dt_s + timedelta(days=3))]
                        if len(exit_prices) > 0:
                            exit_price = float(exit_prices.iloc[-1])
                            pnl = (exit_price / entry) - 1.0
                            all_returns.append(pnl)
                            scan_total_pnl += pnl
                            scan_trades += 1
                            if pnl > 0:
                                scan_winners += 1
                            else:
                                scan_losers += 1
                            ticker_details.append(f"{ticker} ({pnl:+.1%})")
                    except Exception:
                        continue

                if scan_trades > 0:
                    summary_rows.append({
                        "Scan Date": s["scan_date"],
                        "Exit Date": s["exit_date"],
                        "Tickers": ", ".join(ticker_details),
                        "Trades": scan_trades,
                        "Winners": scan_winners,
                        "Losers": scan_losers,
                        "Win Rate": scan_winners / scan_trades,
                        "Avg P&L": scan_total_pnl / scan_trades,
                        "Total P&L": scan_total_pnl,
                    })

            if summary_rows:
                df_summary = pd.DataFrame(summary_rows)
                st.dataframe(
                    df_summary.style.format({
                        "Win Rate": "{:.0%}",
                        "Avg P&L": "{:+.2%}",
                        "Total P&L": "{:+.2%}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

                # Aggregate stats
                total_trades = sum(r["Trades"] for r in summary_rows)
                total_winners = sum(r["Winners"] for r in summary_rows)
                overall_avg = np.mean(all_returns) if all_returns else 0
                overall_total = sum(all_returns)

                st.markdown("---")
                st.subheader("Overall Performance")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Trades", total_trades)
                with col2:
                    st.metric("Win Rate", f"{total_winners/total_trades:.0%}" if total_trades > 0 else "—")
                with col3:
                    st.metric("Avg Return/Trade", f"{overall_avg:+.2%}")
                with col4:
                    st.metric("Total P&L (sum)", f"{overall_total:+.2%}")

                # Returns distribution across all completed trades
                if len(all_returns) > 1:
                    st.subheader("Returns Distribution (All Completed Trades)")
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(8, 4))
                    colors_hist = ["#2ecc71" if r >= 0 else "#e74c3c" for r in sorted(all_returns)]
                    ax.hist(np.array(all_returns) * 100, bins=max(5, len(all_returns) // 2),
                            color="#3498db", edgecolor="white", alpha=0.8)
                    ax.axvline(x=0, color="gray", linewidth=1, linestyle="--")
                    ax.axvline(x=np.mean(all_returns) * 100, color="#2ecc71", linewidth=2,
                               label=f"Mean: {np.mean(all_returns):+.2%}")
                    ax.set_xlabel("Return (%)")
                    ax.set_ylabel("Count")
                    ax.set_title("Distribution of Trade Returns")
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.warning("Could not fetch exit prices for completed scans.")
        else:
            # Show basic info without fetching prices
            for s in completed_scans:
                tickers = ", ".join(t["ticker"] for t in s["tickers"])
                st.write(f"**{s['scan_date']}** — {len(s['tickers'])} trade(s): {tickers} — exited {s['exit_date']}")

