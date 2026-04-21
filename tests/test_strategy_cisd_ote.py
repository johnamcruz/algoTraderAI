"""Unit tests for CISDOTEStrategy v5.1 — session gate, entry gate, stop/target."""

import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_cisd_ote import CISDOTEStrategy, SESSION_START_HOUR, SESSION_END_HOUR


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def strategy():
    """Instantiate strategy without loading ONNX model."""
    s = CISDOTEStrategy.__new__(CISDOTEStrategy)
    s._session_start_hour = SESSION_START_HOUR   # 7
    s._session_end_hour   = SESSION_END_HOUR      # 16
    s._min_vty_regime     = 0.75
    s._latest_vty_regime  = 1.0    # healthy regime by default
    s._active_zones       = []
    s._latest_zone_bullish = 0.0
    s._latest_cisd_features = None
    return s


def ts(hour, minute=0, tz="America/New_York"):
    """Build a timezone-aware Timestamp at the given ET hour."""
    return pd.Timestamp(f"2026-04-21 {hour:02d}:{minute:02d}:00", tz=tz)


# ── 1. Session gate (is_trading_allowed) ──────────────────────────────────────

class TestSessionGate:

    def test_session_open_allowed(self, strategy):
        assert strategy.is_trading_allowed(ts(7)) is True

    def test_mid_session_allowed(self, strategy):
        assert strategy.is_trading_allowed(ts(12)) is True

    def test_last_minute_of_session_allowed(self, strategy):
        assert strategy.is_trading_allowed(ts(15, 59)) is True

    def test_session_close_blocked(self, strategy):
        assert strategy.is_trading_allowed(ts(16)) is False

    def test_pre_session_blocked(self, strategy):
        assert strategy.is_trading_allowed(ts(6, 59)) is False

    def test_overnight_blocked(self, strategy):
        assert strategy.is_trading_allowed(ts(1, 25)) is False

    def test_after_hours_blocked(self, strategy):
        assert strategy.is_trading_allowed(ts(20)) is False

    def test_custom_session_window(self, strategy):
        strategy._session_start_hour = 9
        strategy._session_end_hour   = 11
        assert strategy.is_trading_allowed(ts(9))  is True
        assert strategy.is_trading_allowed(ts(10)) is True
        assert strategy.is_trading_allowed(ts(11)) is False
        assert strategy.is_trading_allowed(ts(8))  is False


# ── 2. Entry gate (should_enter_trade) ────────────────────────────────────────

class TestEntryGate:

    def _enter(self, strategy, prediction=1, confidence=0.80, entry_conf=0.70):
        return strategy.should_enter_trade(prediction, confidence, {}, entry_conf, 0)

    def test_above_threshold_long_enters(self, strategy):
        ok, direction = self._enter(strategy, prediction=1, confidence=0.80)
        assert ok is True
        assert direction == 'LONG'

    def test_above_threshold_short_enters(self, strategy):
        ok, direction = self._enter(strategy, prediction=2, confidence=0.80)
        assert ok is True
        assert direction == 'SHORT'

    def test_exactly_at_threshold_enters(self, strategy):
        ok, direction = self._enter(strategy, confidence=0.70, entry_conf=0.70)
        assert ok is True

    def test_below_threshold_blocked(self, strategy):
        ok, direction = self._enter(strategy, confidence=0.69, entry_conf=0.70)
        assert ok is False
        assert direction is None

    def test_hold_prediction_blocked(self, strategy):
        ok, direction = self._enter(strategy, prediction=0, confidence=0.95)
        assert ok is False
        assert direction is None

    def test_regime_gate_blocks_low_volatility(self, strategy):
        strategy._latest_vty_regime = 0.50   # below 0.75 threshold
        ok, direction = self._enter(strategy, confidence=0.90)
        assert ok is False
        assert direction is None

    def test_regime_gate_passes_healthy_volatility(self, strategy):
        strategy._latest_vty_regime = 0.80   # above 0.75
        ok, direction = self._enter(strategy, prediction=1, confidence=0.90)
        assert ok is True

    def test_regime_gate_disabled_when_zero(self, strategy):
        strategy._min_vty_regime    = 0.0
        strategy._latest_vty_regime = 0.10   # would be blocked if gate enabled
        ok, direction = self._enter(strategy, prediction=1, confidence=0.90)
        assert ok is True

    def test_high_confidence_still_blocked_by_regime(self, strategy):
        strategy._latest_vty_regime = 0.263   # matches the 92% blocked case from backtest
        ok, direction = self._enter(strategy, confidence=0.92)
        assert ok is False

    def test_confidence_below_threshold_ignores_regime(self, strategy):
        """Regime gate is never reached when confidence fails first."""
        strategy._latest_vty_regime = 0.10
        ok, direction = self._enter(strategy, confidence=0.50, entry_conf=0.70)
        assert ok is False


# ── 3. Stop / target calculation (get_stop_target_pts) ───────────────────────

class TestStopTargetPts:

    def _add_zone(self, strategy, fib_bot, fib_top, is_bullish=True):
        strategy._active_zones = [{
            'fib_bot':    fib_bot,
            'fib_top':    fib_top,
            'is_bullish': is_bullish,
            'created_bar': 0,
            'entered_zone': True,
            'signal_fired': True,
        }]

    def test_no_zones_returns_none(self, strategy):
        stop, target = strategy.get_stop_target_pts(None, 'LONG', 100.0)
        assert stop   is None
        assert target is None

    def test_long_stop_is_distance_to_fib_bot(self, strategy):
        self._add_zone(strategy, fib_bot=95.0, fib_top=100.0)
        stop, _ = strategy.get_stop_target_pts(None, 'LONG', 98.0)
        assert stop == pytest.approx(3.0)   # 98 - 95

    def test_long_target_is_2r(self, strategy):
        self._add_zone(strategy, fib_bot=95.0, fib_top=100.0)
        stop, target = strategy.get_stop_target_pts(None, 'LONG', 98.0)
        assert target == pytest.approx(stop * 2.0)

    def test_short_stop_is_distance_to_fib_top(self, strategy):
        self._add_zone(strategy, fib_bot=95.0, fib_top=100.0, is_bullish=False)
        stop, _ = strategy.get_stop_target_pts(None, 'SHORT', 97.0)
        assert stop == pytest.approx(3.0)   # 100 - 97

    def test_short_target_is_2r(self, strategy):
        self._add_zone(strategy, fib_bot=95.0, fib_top=100.0, is_bullish=False)
        stop, target = strategy.get_stop_target_pts(None, 'SHORT', 97.0)
        assert target == pytest.approx(stop * 2.0)

    def test_entry_at_zone_boundary_returns_none(self, strategy):
        """Zero stop distance — invalid zone, should return None."""
        self._add_zone(strategy, fib_bot=95.0, fib_top=100.0)
        stop, target = strategy.get_stop_target_pts(None, 'LONG', 95.0)
        assert stop   is None
        assert target is None

    def test_nearest_zone_selected_when_multiple(self, strategy):
        strategy._active_zones = [
            {'fib_bot': 80.0, 'fib_top': 85.0, 'is_bullish': True,
             'created_bar': 0, 'entered_zone': True, 'signal_fired': True},
            {'fib_bot': 95.0, 'fib_top': 100.0, 'is_bullish': True,
             'created_bar': 0, 'entered_zone': True, 'signal_fired': True},
        ]
        stop, _ = strategy.get_stop_target_pts(None, 'LONG', 97.0)
        assert stop == pytest.approx(2.0)   # 97 - 95, nearest zone selected
