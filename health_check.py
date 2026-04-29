#!/usr/bin/env python3
"""
Model health check — runs a live_30d backtest for each symbol and compares
results against a stored baseline to detect model drift requiring retraining.

Usage:
    python health_check.py                    # compare against baseline, alert if degraded
    python health_check.py --save-baseline    # run backtests and save as new baseline
    python health_check.py --symbols MES MNQ  # check specific symbols only

Credentials (required for live data):
    Pass via --username/--apikey or set TOPSTEP_USERNAME / TOPSTEP_APIKEY env vars.

Add to crontab for weekly Sunday checks (8am):
    0 8 * * 0 cd /path/to/algoTraderAI && python health_check.py >> logs/health_check.log 2>&1
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime

BASELINE_FILE = "data/health_baseline.json"
LOG_FILE = "logs/health_check.log"
ALERT_FLAG = "logs/RETRAIN_ALERT.flag"

ALL_SYMBOLS = ["MES", "MNQ", "MGC"]

# Alert thresholds
WIN_RATE_DROP_WARN = 5.0   # warn if win rate falls this many pct-points below baseline
WIN_RATE_DROP_CRIT = 10.0  # critical if it falls this much below baseline
MIN_TRADES = 10            # ignore results with too few trades (thin data)


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_backtest_output(output: str) -> dict | None:
    """
    Extract win_rate, pnl, and trades from backtest stdout.
    Returns None if the output is missing expected metrics.
    """
    trades_m  = re.search(r"Total Trades:\s+(\d+)", output)
    wr_m      = re.search(r"Win Rate:\s+([\d.]+)%", output)
    pnl_m     = re.search(r"Total P&L \(Dollars\):\s+\$?([-\d,\.]+)", output)

    if not trades_m or not pnl_m:
        return None

    trades  = int(trades_m.group(1))
    win_rate = float(wr_m.group(1)) if wr_m else 0.0
    pnl     = float(pnl_m.group(1).replace(",", ""))

    return {"trades": trades, "win_rate": win_rate, "pnl": pnl}


# ── Backtest runner ───────────────────────────────────────────────────────────

def run_live_backtest(symbol: str, username: str, api_key: str) -> dict | None:
    """Run live_30d backtest for symbol and return parsed metrics."""
    cmd = [
        sys.executable, "backtest.py",
        "--symbol",   symbol,
        "--scenario", "live_30d",
        "--live-data",
        "--username", username,
        "--apikey",   api_key,
        "--no-target",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        combined = result.stdout + result.stderr
        return parse_backtest_output(combined)
    except Exception as exc:
        print(f"  ❌ {symbol}: backtest process failed — {exc}")
        return None


# ── Comparison ────────────────────────────────────────────────────────────────

def compare(current: dict, baseline: dict) -> tuple[str, list[str]]:
    """
    Compare current metrics against baseline.

    Returns:
        (level, reasons) where level is "ok" | "warn" | "critical"
    """
    reasons = []
    level = "ok"

    if current["trades"] < MIN_TRADES:
        reasons.append(f"only {current['trades']} trades — too few to judge")
        return "warn", reasons

    wr_drop = baseline["win_rate"] - current["win_rate"]
    if wr_drop >= WIN_RATE_DROP_CRIT:
        reasons.append(
            f"win rate dropped {wr_drop:.1f}pts  "
            f"({baseline['win_rate']:.1f}% → {current['win_rate']:.1f}%)"
        )
        level = "critical"
    elif wr_drop >= WIN_RATE_DROP_WARN:
        reasons.append(
            f"win rate dropped {wr_drop:.1f}pts  "
            f"({baseline['win_rate']:.1f}% → {current['win_rate']:.1f}%)"
        )
        level = "warn"

    if current["pnl"] < 0:
        reasons.append(f"P&L is negative (${current['pnl']:,.2f})")
        level = "critical" if level != "critical" else level

    if not reasons:
        reasons.append(
            f"win rate {current['win_rate']:.1f}%  P&L ${current['pnl']:,.2f}  trades {current['trades']}"
        )

    return level, reasons


# ── Baseline I/O ──────────────────────────────────────────────────────────────

def load_baseline() -> dict:
    if not os.path.exists(BASELINE_FILE):
        return {}
    with open(BASELINE_FILE) as f:
        return json.load(f)


def save_baseline(data: dict) -> None:
    os.makedirs(os.path.dirname(BASELINE_FILE), exist_ok=True)
    with open(BASELINE_FILE, "w") as f:
        json.dump(data, f, indent=2)


# ── Alert flag ────────────────────────────────────────────────────────────────

def set_alert_flag(message: str) -> None:
    os.makedirs(os.path.dirname(ALERT_FLAG), exist_ok=True)
    with open(ALERT_FLAG, "w") as f:
        f.write(f"{datetime.now().isoformat()}\n{message}\n")


def clear_alert_flag() -> None:
    if os.path.exists(ALERT_FLAG):
        os.remove(ALERT_FLAG)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Weekly model health check")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Run backtests and save results as the new baseline")
    parser.add_argument("--symbols", nargs="+", default=ALL_SYMBOLS,
                        choices=ALL_SYMBOLS,
                        help=f"Symbols to check (default: all — {ALL_SYMBOLS})")
    parser.add_argument("--username", default=os.environ.get("TOPSTEP_USERNAME"),
                        help="TopstepX username (or set TOPSTEP_USERNAME)")
    parser.add_argument("--apikey", default=os.environ.get("TOPSTEP_APIKEY"),
                        help="TopstepX API key (or set TOPSTEP_APIKEY)")
    args = parser.parse_args()

    if not args.username or not args.apikey:
        print("❌ Credentials required: --username/--apikey or TOPSTEP_USERNAME/TOPSTEP_APIKEY env vars.")
        sys.exit(1)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*60}")
    print(f"  MODEL HEALTH CHECK — {now}")
    if args.save_baseline:
        print("  Mode: save-baseline")
    print(f"{'='*60}\n")

    # Run backtests
    results = {}
    for symbol in args.symbols:
        print(f"  [{symbol}] Running live_30d backtest...", flush=True)
        metrics = run_live_backtest(symbol, args.username, args.apikey)
        if metrics:
            print(f"  [{symbol}] trades={metrics['trades']}  "
                  f"win_rate={metrics['win_rate']:.1f}%  "
                  f"pnl=${metrics['pnl']:,.2f}")
            results[symbol] = metrics
        else:
            print(f"  [{symbol}] ⚠ Could not parse results")

    if not results:
        print("\n❌ No results — check credentials and API connectivity.")
        sys.exit(1)

    # Save-baseline mode: store and exit
    if args.save_baseline:
        baseline = load_baseline()
        baseline.update(results)
        baseline["saved_at"] = now
        save_baseline(baseline)
        clear_alert_flag()
        print(f"\n✅ Baseline saved to {BASELINE_FILE}")
        print(json.dumps(results, indent=2))
        return

    # Compare mode: load baseline and evaluate
    baseline = load_baseline()
    if not baseline:
        print(f"\n⚠ No baseline found at {BASELINE_FILE}.")
        print("  Run with --save-baseline first to establish a reference point.")
        sys.exit(1)

    print(f"\n  Baseline from: {baseline.get('saved_at', 'unknown')}\n")

    overall_level = "ok"
    alert_lines = []

    for symbol in args.symbols:
        if symbol not in results:
            continue
        if symbol not in baseline:
            print(f"  [{symbol}] ⚠ No baseline entry — skipping comparison")
            continue

        level, reasons = compare(results[symbol], baseline[symbol])
        icon = {"ok": "✅", "warn": "⚠️ ", "critical": "🚨"}[level]
        print(f"  {icon} [{symbol}] {level.upper()}")
        for r in reasons:
            print(f"       {r}")

        if level == "critical":
            overall_level = "critical"
            alert_lines.append(f"[{symbol}] CRITICAL: {'; '.join(reasons)}")
        elif level == "warn" and overall_level != "critical":
            overall_level = "warn"
            alert_lines.append(f"[{symbol}] WARN: {'; '.join(reasons)}")

    print(f"\n{'='*60}")
    if overall_level == "critical":
        msg = f"Model retraining recommended.\n" + "\n".join(alert_lines)
        set_alert_flag(msg)
        print(f"  🚨 CRITICAL — {msg}")
        print(f"  Alert flag written to: {ALERT_FLAG}")
    elif overall_level == "warn":
        print(f"  ⚠️  WARNING — monitor closely next session")
        for line in alert_lines:
            print(f"       {line}")
        clear_alert_flag()
    else:
        print(f"  ✅ Model healthy — no action needed")
        clear_alert_flag()
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
