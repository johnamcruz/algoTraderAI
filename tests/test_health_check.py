"""
Unit tests for health_check.py

Tests cover:
  - parse_backtest_output: valid output, missing fields, zero trades, negative P&L
  - compare: ok / warn / critical thresholds, thin-data guard
  - baseline save / load round-trip
  - alert flag creation and clearing
  - main() save-baseline and compare modes via subprocess mocking
"""

import sys
import os
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock

import health_check as hc


# ── parse_backtest_output ─────────────────────────────────────────────────────

SAMPLE_OUTPUT = """
  Total Trades: 49
  Winning Trades: 18
  Losing Trades: 31
  Win Rate: 36.73%
  Average P&L per Trade: $22.67
  Total P&L (Points): 101.27
  Total P&L (Dollars): $1,110.87
"""

NEGATIVE_PNL_OUTPUT = """
  Total Trades: 65
  Win Rate: 29.23%
  Total P&L (Dollars): $-1,187.35
"""

ZERO_TRADES_OUTPUT = """
  Total Trades: 0
  Total P&L (Dollars): $0.00
"""


def test_parse_valid_output():
    result = hc.parse_backtest_output(SAMPLE_OUTPUT)
    assert result is not None
    assert result["trades"] == 49
    assert result["win_rate"] == pytest.approx(36.73)
    assert result["pnl"] == pytest.approx(1110.87)


def test_parse_negative_pnl():
    result = hc.parse_backtest_output(NEGATIVE_PNL_OUTPUT)
    assert result is not None
    assert result["pnl"] == pytest.approx(-1187.35)
    assert result["win_rate"] == pytest.approx(29.23)


def test_parse_zero_trades():
    result = hc.parse_backtest_output(ZERO_TRADES_OUTPUT)
    assert result is not None
    assert result["trades"] == 0
    assert result["win_rate"] == 0.0  # no Win Rate line → defaults to 0


def test_parse_missing_trades_returns_none():
    result = hc.parse_backtest_output("Total P&L (Dollars): $500.00")
    assert result is None


def test_parse_missing_pnl_returns_none():
    result = hc.parse_backtest_output("Total Trades: 10\nWin Rate: 50.0%")
    assert result is None


def test_parse_empty_string_returns_none():
    assert hc.parse_backtest_output("") is None


# ── compare ───────────────────────────────────────────────────────────────────

BASELINE = {"trades": 49, "win_rate": 36.73, "pnl": 1110.87}


def test_compare_ok():
    current = {"trades": 45, "win_rate": 35.0, "pnl": 800.0}
    level, reasons = hc.compare(current, BASELINE)
    assert level == "ok"


def test_compare_warn_win_rate_drop():
    current = {"trades": 45, "win_rate": 30.0, "pnl": 500.0}  # 6.73 pt drop → warn
    level, reasons = hc.compare(current, BASELINE)
    assert level == "warn"
    assert any("win rate" in r for r in reasons)


def test_compare_critical_win_rate_drop():
    current = {"trades": 45, "win_rate": 25.0, "pnl": 200.0}  # 11.73 pt drop → critical
    level, reasons = hc.compare(current, BASELINE)
    assert level == "critical"


def test_compare_critical_negative_pnl():
    current = {"trades": 45, "win_rate": 34.0, "pnl": -500.0}
    level, reasons = hc.compare(current, BASELINE)
    assert level == "critical"
    assert any("negative" in r for r in reasons)


def test_compare_warn_then_negative_pnl_stays_critical():
    """Both warn-level win rate drop AND negative P&L → critical."""
    current = {"trades": 45, "win_rate": 30.0, "pnl": -200.0}
    level, reasons = hc.compare(current, BASELINE)
    assert level == "critical"


def test_compare_thin_data_returns_warn():
    current = {"trades": 3, "win_rate": 50.0, "pnl": 100.0}
    level, reasons = hc.compare(current, BASELINE)
    assert level == "warn"
    assert any("too few" in r for r in reasons)


# ── baseline save / load ──────────────────────────────────────────────────────

def test_baseline_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "BASELINE_FILE", str(tmp_path / "baseline.json"))
    data = {"MES": {"trades": 49, "win_rate": 36.73, "pnl": 1110.87}}
    hc.save_baseline(data)
    loaded = hc.load_baseline()
    assert loaded == data


def test_load_baseline_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "BASELINE_FILE", str(tmp_path / "nonexistent.json"))
    assert hc.load_baseline() == {}


# ── alert flag ────────────────────────────────────────────────────────────────

def test_set_and_clear_alert_flag(tmp_path, monkeypatch):
    flag = str(tmp_path / "RETRAIN_ALERT.flag")
    monkeypatch.setattr(hc, "ALERT_FLAG", flag)
    hc.set_alert_flag("test message")
    assert os.path.exists(flag)
    content = open(flag).read()
    assert "test message" in content
    hc.clear_alert_flag()
    assert not os.path.exists(flag)


def test_clear_alert_flag_no_op_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(hc, "ALERT_FLAG", str(tmp_path / "no_flag.flag"))
    hc.clear_alert_flag()  # should not raise


# ── run_live_backtest (mocked subprocess) ─────────────────────────────────────

@patch("health_check.subprocess.run")
def test_run_live_backtest_returns_parsed_metrics(mock_run):
    mock_run.return_value = MagicMock(
        stdout=SAMPLE_OUTPUT,
        stderr="",
    )
    result = hc.run_live_backtest("MES", "user", "key")
    assert result is not None
    assert result["trades"] == 49
    assert result["pnl"] == pytest.approx(1110.87)


@patch("health_check.subprocess.run")
def test_run_live_backtest_returns_none_on_parse_failure(mock_run):
    mock_run.return_value = MagicMock(stdout="no metrics here", stderr="")
    result = hc.run_live_backtest("MES", "user", "key")
    assert result is None


@patch("health_check.subprocess.run", side_effect=Exception("process error"))
def test_run_live_backtest_returns_none_on_exception(mock_run):
    result = hc.run_live_backtest("MES", "user", "key")
    assert result is None
