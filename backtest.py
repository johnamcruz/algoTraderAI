#!/usr/bin/env python3
"""
Backtest runner — run predefined market-regime scenarios against algoTrader.py.

Usage:
    python3 backtest.py                        # run all scenarios
    python3 backtest.py --scenario bull_2023   # run one scenario by key
    python3 backtest.py --list                 # list available scenarios
    python3 backtest.py --parallel             # run all in parallel (default: sequential)
    python3 backtest.py --symbol MES           # override symbol (MNQ default)
    python3 backtest.py --entry_conf 0.90      # override entry confidence
    python3 backtest.py --model models/cisd_ote_hybrid_v7.onnx  # override model
"""

import argparse
import subprocess
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

# ── Scenario definitions ─────────────────────────────────────────────────────

SCENARIOS = {
    "bear_2022": {
        "label":      "2022 Bear Market",
        "start_date": "2022-01-01",
        "end_date":   "2022-10-15",
        "note":       "Persistent downtrend; model over-fires on pullback bounces",
    },
    "recovery_2023": {
        "label":      "2023 Recovery",
        "start_date": "2023-01-01",
        "end_date":   "2023-12-31",
        "note":       "Full-year recovery; post-rate-hike rebound + AI hype",
    },
    "banking_2023": {
        "label":      "2023 Banking Crisis",
        "start_date": "2023-03-01",
        "end_date":   "2023-05-31",
        "note":       "SVB collapse, high-vol chop; CISD historically strong here",
    },
    "selloff_2024": {
        "label":      "Aug 2024 Selloff",
        "start_date": "2024-07-15",
        "end_date":   "2024-09-15",
        "note":       "Sharp selloff + recovery; high directional volatility",
    },
    "oos_2021": {
        "label":      "2021 OOS Control",
        "start_date": "2021-01-01",
        "end_date":   "2021-12-31",
        "note":       "Out-of-sample year; structural uptrend with low volatility",
    },
    "recent_120d": {
        "label":      "Last 120 Days",
        "start_date": "2025-06-27",
        "end_date":   "2025-10-24",
        "note":       "Most recent 120-day window in data; previously profitable at 0.70",
    },
    "recent_30d": {
        "label":      "Last 30 Days",
        "start_date": "2025-09-24",
        "end_date":   "2025-10-24",
        "note":       "Most recent 30-day window in data",
    },
    "recent_60d": {
        "label":      "Last 60 Days",
        "start_date": "2025-08-25",
        "end_date":   "2025-10-24",
        "note":       "Most recent 60-day window in data",
    },
    "recent_90d": {
        "label":      "Last 90 Days",
        "start_date": "2025-07-26",
        "end_date":   "2025-10-24",
        "note":       "Most recent 90-day window in data",
    },
    "recent_180d": {
        "label":      "Last 6 Months",
        "start_date": "2025-04-24",
        "end_date":   "2025-10-24",
        "note":       "Most recent 6-month window in data",
    },
    "recent_1yr": {
        "label":      "Last 1 Year",
        "start_date": "2024-10-24",
        "end_date":   "2025-10-24",
        "note":       "Most recent full year in data",
    },
}

# Live scenarios: dates computed at runtime from today.
# Only available when --live-data is passed.
LIVE_SCENARIOS = {
    "live_30d":  {"label": "Live Last 30 Days",  "days": 30},
    "live_60d":  {"label": "Live Last 60 Days",  "days": 60},
    "live_90d":  {"label": "Live Last 90 Days",  "days": 90},
    "live_180d": {"label": "Live Last 180 Days", "days": 180},
}


def _live_scenario(key: str) -> dict:
    """Build a scenario dict for a live_Nd key using today as end date."""
    days = LIVE_SCENARIOS[key]["days"]
    today = datetime.utcnow().date()
    start = today - timedelta(days=days)
    return {
        "label":      LIVE_SCENARIOS[key]["label"],
        "start_date": str(start),
        "end_date":   str(today),
        "note":       f"Live API data: last {days} days ending today ({today})",
    }

# Symbol → data file, tick_size, full contract ID for backtesting
SYMBOL_CONFIG = {
    "MNQ": {
        "data":       "data/NQ_5min.csv",
        "tick_size":  0.25,
        "contract":   "CON.F.US.MNQ.M26",
    },
    "MES": {
        "data":       "data/ES_5min.csv",
        "tick_size":  0.25,
        "contract":   "CON.F.US.MES.M26",
    },
    "MGC": {
        "data":       "data/GC_5min.csv",
        "tick_size":  0.10,
        "contract":   "CON.F.US.MGC.M26",
    },
    "SIL": {
        "data":       "data/SI_5min.csv",
        "tick_size":  0.005,
        "contract":   "CON.F.US.SIL.M26",
    },
}

DEFAULT_MODEL_V7      = "models/cisd_ote_hybrid_v7.onnx"
DEFAULT_MODEL_V10     = "models/cisd_ote_hybrid_v10.onnx"
DEFAULT_MODEL_ST      = "models/st_trend_v1.onnx"
DEFAULT_MODEL_VWAP    = "models/vwap_v1.onnx"

# Maps strategy name → default model when --model is not explicitly overridden
STRATEGY_DEFAULT_MODEL = {
    "cisd-ote7":  DEFAULT_MODEL_V7,
    "cisd-ote10": DEFAULT_MODEL_V10,
    "supertrend": DEFAULT_MODEL_ST,
    "vwap":       DEFAULT_MODEL_VWAP,
}
DEFAULT_SYMBOL        = "MNQ"
DEFAULT_ENTRY_CONF    = 0.80
DEFAULT_RISK_AMOUNT   = 200.0
DEFAULT_HIGH_CONF_MULT = 2.0
DEFAULT_MAX_CONTRACTS = 5
DEFAULT_MAX_LOSS      = 3000.0
DEFAULT_PROFIT_TARGET = 12000.0
DEFAULT_MIN_STOP_ATR  = 0.5
DEFAULT_MIN_STOP_PTS  = 1.0
DEFAULT_MIN_RISK_RR   = 2.0


# ── Build command ─────────────────────────────────────────────────────────────

def _get_scenario(scenario_key: str, args) -> dict:
    """Return the scenario dict for a key, supporting live_Nd keys."""
    if scenario_key in LIVE_SCENARIOS:
        return _live_scenario(scenario_key)
    return SCENARIOS[scenario_key]


def build_command(scenario_key: str, args) -> list[str]:
    sc = _get_scenario(scenario_key, args)
    sym_cfg = SYMBOL_CONFIG[args.symbol]

    cmd = [
        sys.executable, "algoTrader.py",
        "--backtest",
        "--contract",             sym_cfg["contract"],
        "--tick_size",            str(sym_cfg["tick_size"]),
        "--start-date",           sc["start_date"],
        "--end-date",             sc["end_date"],
        "--strategy",             args.strategy,
        "--model",                args.model,
        "--entry_conf",           str(args.entry_conf),
        "--risk_amount",          str(args.risk_amount),
        "--max_contracts",        str(args.max_contracts),
        "--high_conf_multiplier", str(args.high_conf_mult),
        "--max_loss",             str(args.max_loss),
        "--min_stop_atr",         str(args.min_stop_atr),
        "--min_stop_pts",         str(args.min_stop_pts),
        "--min_risk_rr",          str(args.min_risk_rr),
        "--quiet",
    ]

    if getattr(args, 'live_data', False):
        cmd += ["--live-data", "--username", args.username, "--apikey", args.apikey]
    else:
        cmd += ["--backtest_data", sym_cfg["data"]]

    if not args.no_breakeven:
        cmd.append("--breakeven_on_2r")
    if args.no_target:
        cmd.append("--no-profit-target")
    else:
        cmd += ["--profit_target", str(args.profit_target)]
    return cmd


# ── Run one scenario ──────────────────────────────────────────────────────────

def run_scenario(scenario_key: str, args, log_dir: str) -> dict:
    sc = _get_scenario(scenario_key, args)
    log_path = os.path.join(log_dir, f"{scenario_key}.txt")
    cmd = build_command(scenario_key, args)

    t0 = time.time()
    with open(log_path, "w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - t0

    output = open(log_path).read()
    return {
        "key":      scenario_key,
        "label":    sc["label"],
        "note":     sc["note"],
        "elapsed":  elapsed,
        "returncode": proc.returncode,
        "output":   output,
        "log_path": log_path,
    }


# ── Extract summary lines from bot output ─────────────────────────────────────

def extract_summary(output: str) -> list[str]:
    lines = output.splitlines()
    summary_lines = []
    in_summary = False
    for line in lines:
        if "SIMULATION RESULTS" in line or "BACKTEST SUMMARY" in line:
            in_summary = True
        if in_summary:
            summary_lines.append(line)
            # Stop after the closing separator that follows the stats block
            if summary_lines and line.startswith("===") and len(summary_lines) > 3:
                break
    return summary_lines


# ── Print results ─────────────────────────────────────────────────────────────

def print_results(results: list[dict], args) -> None:
    width = 70
    print()
    print("═" * width)
    print(f"  BACKTEST RESULTS — {args.symbol}  |  strategy: {args.strategy}  |  model: {os.path.basename(args.model)}")
    print(f"  entry_conf={args.entry_conf}  risk=${args.risk_amount}  max_contracts={args.max_contracts}")
    print(f"  max_loss=${args.max_loss}  profit_target={'none' if args.no_target else f'${args.profit_target}'}  min_stop_atr={args.min_stop_atr}")
    print(f"  min_risk_rr={args.min_risk_rr}  breakeven={'OFF' if args.no_breakeven else 'ON'}")
    print("═" * width)

    for r in results:
        marker = "✓" if r["returncode"] == 0 else "✗"
        print(f"\n{marker} {r['label']} ({r['key']})  [{r['elapsed']:.1f}s]")
        print(f"  {r['note']}")
        print(f"  {r['start_date']} → {r['end_date']}")
        print()
        for line in extract_summary(r["output"]):
            print(f"  {line}")
        if r["returncode"] != 0:
            print(f"  ⚠ Process exited {r['returncode']} — see {r['log_path']}")

    print()
    print("═" * width)
    print(f"  Full logs: {os.path.dirname(results[0]['log_path'])}/")
    print("═" * width)


def attach_dates(results: list[dict], args) -> list[dict]:
    for r in results:
        sc = _get_scenario(r["key"], args)
        r["start_date"] = sc["start_date"]
        r["end_date"] = sc["end_date"]
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run predefined backtest scenarios against algoTrader.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--scenario",   type=str,  default=None,
                        help="Key of a single scenario to run (omit for all)")
    parser.add_argument("--list",       action="store_true",
                        help="List available scenarios and exit")
    parser.add_argument("--parallel",   action="store_true",
                        help="Run scenarios in parallel (default: sequential)")
    parser.add_argument("--symbol",     type=str,  default=DEFAULT_SYMBOL,
                        choices=["MNQ", "MES", "MGC", "SIL"],
                        help=f"Trading symbol (default: {DEFAULT_SYMBOL})")
    parser.add_argument("--strategy",   type=str,  default="cisd-ote7",
                        choices=["cisd-ote7", "cisd-ote10", "supertrend", "vwap"],
                        help="Strategy to backtest (default: cisd-ote7)")
    parser.add_argument("--model",      type=str,  default=DEFAULT_MODEL_V7,
                        help=f"ONNX model path (default: {DEFAULT_MODEL_V7})")
    parser.add_argument("--entry_conf", type=float, default=DEFAULT_ENTRY_CONF,
                        help=f"Entry confidence threshold (default: {DEFAULT_ENTRY_CONF})")
    parser.add_argument("--risk_amount", type=float, default=DEFAULT_RISK_AMOUNT,
                        help=f"Risk per trade in $ (default: {DEFAULT_RISK_AMOUNT})")
    parser.add_argument("--max_contracts", type=int, default=DEFAULT_MAX_CONTRACTS,
                        help=f"Max contracts per trade (default: {DEFAULT_MAX_CONTRACTS})")
    parser.add_argument("--high_conf_mult", type=float, default=DEFAULT_HIGH_CONF_MULT,
                        help=f"High-confidence size multiplier (default: {DEFAULT_HIGH_CONF_MULT})")
    parser.add_argument("--max_loss", type=float, default=DEFAULT_MAX_LOSS,
                        help=f"Max loss limit in $ (default: {DEFAULT_MAX_LOSS})")
    parser.add_argument("--profit_target", type=float, default=DEFAULT_PROFIT_TARGET,
                        help=f"Profit target in $ (default: {DEFAULT_PROFIT_TARGET})")
    parser.add_argument("--min_stop_atr", type=float, default=DEFAULT_MIN_STOP_ATR,
                        help=f"Min stop ATR multiplier (default: {DEFAULT_MIN_STOP_ATR})")
    parser.add_argument("--min_stop_pts", type=float, default=DEFAULT_MIN_STOP_PTS,
                        help=f"Min stop floor in points (default: {DEFAULT_MIN_STOP_PTS})")
    parser.add_argument("--min_risk_rr", type=float, default=DEFAULT_MIN_RISK_RR,
                        help=f"(cisd-ote7) RR gate: min model-predicted RR to enter (default: {DEFAULT_MIN_RISK_RR}, 0=off)")
    parser.add_argument("--no-breakeven", action="store_true", default=False,
                        help="Disable breakeven-on-2R (default: on)")
    parser.add_argument("--no-target", action="store_true", default=False,
                        help="Disable session profit target cap — run full regime window")
    parser.add_argument("--live-data", action="store_true", default=False,
                        help="Fetch bars from the live API instead of CSV (enables live_Nd scenarios)")
    parser.add_argument("--username", type=str, default=os.environ.get("TOPSTEP_USERNAME"),
                        help="TopstepX username (or set TOPSTEP_USERNAME env var)")
    parser.add_argument("--apikey", type=str, default=os.environ.get("TOPSTEP_APIKEY"),
                        help="TopstepX API key (or set TOPSTEP_APIKEY env var)")
    args = parser.parse_args()

    # Auto-select the correct model when --model was not explicitly provided
    if args.model == DEFAULT_MODEL_V7 and args.strategy in STRATEGY_DEFAULT_MODEL:
        args.model = STRATEGY_DEFAULT_MODEL[args.strategy]

    all_scenario_keys = list(SCENARIOS.keys()) + list(LIVE_SCENARIOS.keys())

    if args.list:
        print("\nAvailable scenarios (CSV-based):")
        for key, sc in SCENARIOS.items():
            print(f"  {key:<18}  {sc['start_date']} → {sc['end_date']}  {sc['label']}")
            print(f"  {'':18}  {sc['note']}")
        print("\nLive scenarios (--live-data required):")
        for key, ls in LIVE_SCENARIOS.items():
            sc = _live_scenario(key)
            print(f"  {key:<18}  {sc['start_date']} → {sc['end_date']}  {ls['label']}")
        print()
        return

    if args.live_data and not args.scenario:
        keys_to_run = list(LIVE_SCENARIOS.keys())
    elif args.scenario:
        keys_to_run = [args.scenario]
    else:
        keys_to_run = list(SCENARIOS.keys())

    for k in keys_to_run:
        if k not in all_scenario_keys:
            print(f"Unknown scenario '{k}'. Use --list to see options.")
            sys.exit(1)

    if args.live_data and (not args.username or not args.apikey):
        print("--live-data requires --username and --apikey (or TOPSTEP_USERNAME/TOPSTEP_APIKEY env vars).")
        sys.exit(1)

    # Create log dir
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backtest_logs", run_ts)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Running {len(keys_to_run)} scenario(s) — logs → {log_dir}/")
    if args.parallel:
        print("Mode: parallel")
    else:
        print("Mode: sequential")
    print()

    results = []

    if args.parallel:
        with ThreadPoolExecutor(max_workers=len(keys_to_run)) as ex:
            futures = {ex.submit(run_scenario, k, args, log_dir): k for k in keys_to_run}
            for fut in as_completed(futures):
                r = fut.result()
                marker = "✓" if r["returncode"] == 0 else "✗"
                print(f"  {marker} Finished: {r['label']}  ({r['elapsed']:.1f}s)")
                results.append(r)
        results.sort(key=lambda r: keys_to_run.index(r["key"]))
    else:
        for k in keys_to_run:
            sc = _get_scenario(k, args)
            print(f"  → {sc['label']} ({sc['start_date']} → {sc['end_date']}) ...", flush=True)
            r = run_scenario(k, args, log_dir)
            marker = "✓" if r["returncode"] == 0 else "✗"
            print(f"     {marker} done ({r['elapsed']:.1f}s)")
            results.append(r)

    results = attach_dates(results, args)
    print_results(results, args)


if __name__ == "__main__":
    main()
