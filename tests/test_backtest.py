"""
Unit tests for backtest.py — scenario definitions, command builder, and summary extractor.
Does NOT invoke algoTrader.py (no subprocess calls, no file I/O).
"""

import sys
import os
import argparse
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backtest import (
    SCENARIOS,
    SYMBOL_CONFIG,
    DEFAULT_ENTRY_CONF,
    DEFAULT_RISK_AMOUNT,
    DEFAULT_MAX_CONTRACTS,
    DEFAULT_MAX_LOSS,
    DEFAULT_PROFIT_TARGET,
    DEFAULT_MIN_STOP_ATR,
    DEFAULT_MIN_STOP_PTS,
    DEFAULT_MIN_ENTRY_DISTANCE,
    DEFAULT_MIN_RISK_RR,
    DEFAULT_HIGH_CONF_MULT,
    DEFAULT_MODEL,
    DEFAULT_SYMBOL,
    build_command,
    extract_summary,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _default_args(**overrides):
    """Return a minimal argparse.Namespace matching backtest.py defaults."""
    base = dict(
        symbol=DEFAULT_SYMBOL,
        strategy='cisd-ote',
        model=DEFAULT_MODEL,
        entry_conf=DEFAULT_ENTRY_CONF,
        risk_amount=DEFAULT_RISK_AMOUNT,
        max_contracts=DEFAULT_MAX_CONTRACTS,
        high_conf_mult=DEFAULT_HIGH_CONF_MULT,
        max_loss=DEFAULT_MAX_LOSS,
        profit_target=DEFAULT_PROFIT_TARGET,
        min_stop_atr=DEFAULT_MIN_STOP_ATR,
        min_stop_pts=DEFAULT_MIN_STOP_PTS,
        min_entry_distance=DEFAULT_MIN_ENTRY_DISTANCE,
        min_vty_regime=0.75,
        min_risk_rr=DEFAULT_MIN_RISK_RR,
        breakeven=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ── Scenario definitions ───────────────────────────────────────────────────────

class TestScenarioDefinitions:
    def test_all_required_keys_present(self):
        for key, sc in SCENARIOS.items():
            assert "label"      in sc, f"{key} missing 'label'"
            assert "start_date" in sc, f"{key} missing 'start_date'"
            assert "end_date"   in sc, f"{key} missing 'end_date'"
            assert "note"       in sc, f"{key} missing 'note'"

    def test_dates_are_valid_iso(self):
        from datetime import date
        for key, sc in SCENARIOS.items():
            start = date.fromisoformat(sc["start_date"])
            end   = date.fromisoformat(sc["end_date"])
            assert start < end, f"{key}: start_date must be before end_date"

    def test_known_scenarios_exist(self):
        for key in ("bear_2022", "recovery_2023", "banking_2023", "selloff_2024", "oos_2021"):
            assert key in SCENARIOS, f"Expected scenario '{key}' not found"

    def test_bear_2022_ends_oct(self):
        assert SCENARIOS["bear_2022"]["end_date"] == "2022-10-15"

    def test_selloff_2024_range(self):
        sc = SCENARIOS["selloff_2024"]
        assert sc["start_date"] == "2024-07-15"
        assert sc["end_date"]   == "2024-09-15"


# ── Symbol config ─────────────────────────────────────────────────────────────

class TestSymbolConfig:
    def test_mnq_and_mes_defined(self):
        assert "MNQ" in SYMBOL_CONFIG
        assert "MES" in SYMBOL_CONFIG

    def test_required_keys(self):
        for sym, cfg in SYMBOL_CONFIG.items():
            assert "data"      in cfg, f"{sym} missing 'data'"
            assert "tick_size" in cfg, f"{sym} missing 'tick_size'"
            assert "contract"  in cfg, f"{sym} missing 'contract'"

    def test_mnq_contract_id(self):
        assert "MNQ" in SYMBOL_CONFIG["MNQ"]["contract"]

    def test_mes_contract_id(self):
        assert "MES" in SYMBOL_CONFIG["MES"]["contract"]

    def test_tick_sizes_are_positive(self):
        for sym, cfg in SYMBOL_CONFIG.items():
            assert cfg["tick_size"] > 0, f"{sym} tick_size must be positive"


# ── Command builder ───────────────────────────────────────────────────────────

class TestBuildCommand:
    def test_returns_list_of_strings(self):
        cmd = build_command("bear_2022", _default_args())
        assert isinstance(cmd, list)
        assert all(isinstance(s, str) for s in cmd)

    def test_backtest_flag_present(self):
        cmd = build_command("bear_2022", _default_args())
        assert "--backtest" in cmd

    def test_correct_data_file_for_mnq(self):
        cmd = build_command("bear_2022", _default_args(symbol="MNQ"))
        idx = cmd.index("--backtest_data")
        assert "NQ" in cmd[idx + 1]

    def test_correct_data_file_for_mes(self):
        cmd = build_command("bear_2022", _default_args(symbol="MES"))
        idx = cmd.index("--backtest_data")
        assert "ES" in cmd[idx + 1]

    def test_full_contract_id_passed(self):
        cmd = build_command("bear_2022", _default_args(symbol="MNQ"))
        idx = cmd.index("--contract")
        assert "MNQ" in cmd[idx + 1]
        assert "CON" in cmd[idx + 1]

    def test_start_and_end_dates_match_scenario(self):
        cmd = build_command("bear_2022", _default_args())
        start_idx = cmd.index("--start-date")
        end_idx   = cmd.index("--end-date")
        assert cmd[start_idx + 1] == SCENARIOS["bear_2022"]["start_date"]
        assert cmd[end_idx   + 1] == SCENARIOS["bear_2022"]["end_date"]

    def test_entry_conf_passed(self):
        cmd = build_command("bear_2022", _default_args(entry_conf=0.85))
        idx = cmd.index("--entry_conf")
        assert cmd[idx + 1] == "0.85"

    def test_risk_amount_passed(self):
        cmd = build_command("bear_2022", _default_args(risk_amount=200.0))
        idx = cmd.index("--risk_amount")
        assert cmd[idx + 1] == "200.0"

    def test_max_contracts_passed(self):
        cmd = build_command("bear_2022", _default_args(max_contracts=5))
        idx = cmd.index("--max_contracts")
        assert cmd[idx + 1] == "5"

    def test_max_loss_passed(self):
        cmd = build_command("bear_2022", _default_args(max_loss=400.0))
        idx = cmd.index("--max_loss")
        assert cmd[idx + 1] == "400.0"

    def test_profit_target_passed(self):
        cmd = build_command("bear_2022", _default_args(profit_target=6000.0))
        idx = cmd.index("--profit_target")
        assert cmd[idx + 1] == "6000.0"

    def test_min_entry_distance_passed(self):
        cmd = build_command("bear_2022", _default_args(min_entry_distance=3.0))
        idx = cmd.index("--min_entry_distance")
        assert cmd[idx + 1] == "3.0"

    def test_quiet_flag_present(self):
        cmd = build_command("bear_2022", _default_args())
        assert "--quiet" in cmd

    def test_strategy_is_cisd_ote(self):
        cmd = build_command("bear_2022", _default_args())
        idx = cmd.index("--strategy")
        assert cmd[idx + 1] == "cisd-ote"

    def test_all_scenarios_build_without_error(self):
        args = _default_args()
        for key in SCENARIOS:
            cmd = build_command(key, args)
            assert len(cmd) > 5


# ── Summary extractor ─────────────────────────────────────────────────────────

class TestExtractSummary:
    def _make_output(self, total=10, wins=7, losses=3, pnl=2100.0, extra=""):
        return f"""Logging configured to file: logs/bot_MNQ_20260421.log
🤖 Simulation Bot initialized for MNQ
📊 Strategy: CISDOTEStrategy
💰 Profit Target: $6,000.00 | Max Loss: $400.00

============================================================
📊 SIMULATION RESULTS
============================================================
Total Trades: {total}
Winning Trades: {wins}
Losing Trades: {losses}
Win Rate: {wins/total*100:.2f}%
Average P&L per Trade: ${pnl/total:.2f}
{extra}
Total P&L (Points): 123.45
Total P&L (Dollars): ${pnl:,.2f}
============================================================

📋 TRADE LOG:
"""

    def test_extracts_simulation_results_header(self):
        output = self._make_output()
        lines = extract_summary(output)
        assert any("SIMULATION RESULTS" in l for l in lines)

    def test_extracts_total_trades(self):
        output = self._make_output(total=42)
        lines = extract_summary(output)
        assert any("Total Trades: 42" in l for l in lines)

    def test_extracts_win_rate(self):
        output = self._make_output(total=10, wins=7)
        lines = extract_summary(output)
        assert any("Win Rate" in l for l in lines)

    def test_stops_before_trade_log(self):
        output = self._make_output()
        lines = extract_summary(output)
        assert not any("TRADE LOG" in l for l in lines)

    def test_empty_output_returns_empty(self):
        lines = extract_summary("")
        assert lines == []

    def test_no_summary_section_returns_empty(self):
        lines = extract_summary("just some random log output\nnothing here")
        assert lines == []

    def test_profit_target_hit_line_included(self):
        extra = "🎉 Profit Target Hit: 6,200.00 USD (Target: 6,000.00 USD)"
        output = self._make_output(extra=extra)
        lines = extract_summary(output)
        assert any("Profit Target Hit" in l for l in lines)

    def test_mll_hit_line_included(self):
        extra = "🛑 MLL Hit Zero: P&L $-400.00 (MLL: $400.00)"
        output = self._make_output(extra=extra)
        lines = extract_summary(output)
        assert any("MLL Hit" in l for l in lines)


# ── Defaults sanity ───────────────────────────────────────────────────────────

class TestDefaults:
    def test_default_risk_amount(self):
        assert DEFAULT_RISK_AMOUNT == 200.0

    def test_default_max_contracts(self):
        assert DEFAULT_MAX_CONTRACTS == 5

    def test_default_max_loss(self):
        assert DEFAULT_MAX_LOSS == 400.0

    def test_default_profit_target(self):
        assert DEFAULT_PROFIT_TARGET == 6000.0

    def test_default_entry_conf(self):
        assert DEFAULT_ENTRY_CONF == 0.70

    def test_default_symbol_is_mnq(self):
        assert DEFAULT_SYMBOL == "MNQ"

    def test_default_min_entry_distance(self):
        assert DEFAULT_MIN_ENTRY_DISTANCE == 3.0
