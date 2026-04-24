"""Unit tests for CISDOTEStrategyV7 — entry gate, stop/target, feature vector, zone state."""

import sys
import os
import pytest
import numpy as np
import pandas as pd
from collections import deque
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# futures_foundation is an external package not installed in the test environment.
# Stub it out before importing the strategy so the module loads cleanly.
_ff_stub = MagicMock()
_ff_stub.derive_features = MagicMock(return_value=pd.DataFrame())
_ff_stub.get_model_feature_columns = MagicMock(return_value=[])
_ff_stub.INSTRUMENT_MAP = {'ES': 0, 'NQ': 1, 'RTY': 2, 'YM': 3, 'GC': 4}
sys.modules.setdefault('futures_foundation', _ff_stub)

from strategy_cisd_ote_v7 import (
    CISDOTEStrategyV7,
    ZONE_MAX_BARS, MAX_RISK_DOLLARS, POINT_VALUES,
    OPTIMAL_START_HOUR, OPTIMAL_END_HOUR,
)


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def strategy():
    """Instantiate v7 strategy without loading ONNX model."""
    s = CISDOTEStrategyV7.__new__(CISDOTEStrategyV7)
    s._instrument           = 'NQ'
    s._active_zones         = deque(maxlen=20)
    s._latest_cisd_features = None
    s._latest_zone_bullish  = 0.0
    s._latest_risk_rr       = 2.0
    s._latest_signal_meta   = {}
    s._bar_count            = 0
    return s


def _zone(fib_bot=95.0, fib_top=100.0, is_bullish=True, had_sweep=False,
          disp_strength=0.85, created_bar=0):
    return {
        'fib_bot':      fib_bot,
        'fib_top':      fib_top,
        'is_bullish':   is_bullish,
        'had_sweep':    had_sweep,
        'disp_strength': disp_strength,
        'created_bar':  created_bar,
        'entered_zone': True,
        'signal_fired': False,
    }


def _df(close=97.0, high=98.0, low=94.0, hour=10):
    """Minimal single-row DataFrame for feature vector tests."""
    idx = pd.DatetimeIndex(
        [pd.Timestamp(f'2026-04-21 {hour:02d}:00:00', tz='America/New_York')]
    )
    return pd.DataFrame(
        {'open': [97.0], 'high': [high], 'low': [low], 'close': [close], 'volume': [1000]},
        index=idx,
    )


# ── 1. Session gate (is_trading_allowed) ─────────────────────────────────────

class TestSessionGate:
    """v7 removes the hard session gate — is_trading_allowed always returns True."""

    def test_always_true_during_session(self, strategy):
        ts = pd.Timestamp('2026-04-21 10:00:00', tz='America/New_York')
        assert strategy.is_trading_allowed(ts) is True

    def test_always_true_pre_market(self, strategy):
        ts = pd.Timestamp('2026-04-21 04:00:00', tz='America/New_York')
        assert strategy.is_trading_allowed(ts) is True

    def test_always_true_after_hours(self, strategy):
        ts = pd.Timestamp('2026-04-21 20:00:00', tz='America/New_York')
        assert strategy.is_trading_allowed(ts) is True

    def test_always_true_overnight(self, strategy):
        ts = pd.Timestamp('2026-04-21 01:00:00', tz='America/New_York')
        assert strategy.is_trading_allowed(ts) is True


# ── 2. Entry gate (should_enter_trade) ───────────────────────────────────────

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
        ok, _ = self._enter(strategy, confidence=0.70, entry_conf=0.70)
        assert ok is True

    def test_below_threshold_blocked(self, strategy):
        ok, direction = self._enter(strategy, confidence=0.69, entry_conf=0.70)
        assert ok is False
        assert direction is None

    def test_hold_prediction_blocked(self, strategy):
        ok, direction = self._enter(strategy, prediction=0, confidence=0.95)
        assert ok is False
        assert direction is None

    def test_no_regime_gate_in_v7(self, strategy):
        """v7 has no volatility regime filter — high confidence always passes."""
        ok, _ = self._enter(strategy, prediction=1, confidence=0.95)
        assert ok is True

    def test_conservative_threshold(self, strategy):
        ok, direction = self._enter(strategy, confidence=0.91, entry_conf=0.90)
        assert ok is True
        assert direction == 'LONG'

    def test_below_conservative_threshold_blocked(self, strategy):
        ok, _ = self._enter(strategy, confidence=0.89, entry_conf=0.90)
        assert ok is False


# ── 3. Stop / target calculation (get_stop_target_pts) ───────────────────────

class TestStopTargetPts:

    def test_no_zones_returns_none(self, strategy):
        stop, target = strategy.get_stop_target_pts(None, 'LONG', 100.0)
        assert stop is None and target is None

    def test_long_stop_is_distance_to_fib_bot(self, strategy):
        strategy._active_zones.appendleft(_zone(fib_bot=95.0, fib_top=100.0))
        stop, _ = strategy.get_stop_target_pts(None, 'LONG', 98.0)
        assert stop == pytest.approx(3.0)   # 98 - 95

    def test_long_target_uses_risk_rr(self, strategy):
        strategy._latest_risk_rr = 2.5
        strategy._active_zones.appendleft(_zone(fib_bot=95.0, fib_top=100.0))
        stop, target = strategy.get_stop_target_pts(None, 'LONG', 98.0)
        assert target == pytest.approx(stop * 2.5)

    def test_short_stop_is_distance_to_fib_top(self, strategy):
        strategy._active_zones.appendleft(_zone(fib_bot=95.0, fib_top=100.0, is_bullish=False))
        stop, _ = strategy.get_stop_target_pts(None, 'SHORT', 97.0)
        assert stop == pytest.approx(3.0)   # 100 - 97

    def test_short_target_uses_risk_rr(self, strategy):
        strategy._latest_risk_rr = 3.0
        strategy._active_zones.appendleft(_zone(fib_bot=95.0, fib_top=100.0, is_bullish=False))
        stop, target = strategy.get_stop_target_pts(None, 'SHORT', 97.0)
        assert target == pytest.approx(stop * 3.0)

    def test_risk_rr_floored_at_1(self, strategy):
        """If model predicts RR < 1, floor to 1.0 so target never beats stop."""
        strategy._latest_risk_rr = 0.3
        strategy._active_zones.appendleft(_zone(fib_bot=95.0, fib_top=100.0))
        stop, target = strategy.get_stop_target_pts(None, 'LONG', 98.0)
        assert target == pytest.approx(stop * 1.0)

    def test_zero_stop_returns_none(self, strategy):
        strategy._active_zones.appendleft(_zone(fib_bot=95.0, fib_top=100.0))
        stop, target = strategy.get_stop_target_pts(None, 'LONG', 95.0)
        assert stop is None and target is None

    def test_nearest_zone_selected(self, strategy):
        strategy._active_zones.appendleft(_zone(fib_bot=80.0, fib_top=85.0))
        strategy._active_zones.appendleft(_zone(fib_bot=95.0, fib_top=100.0))
        stop, _ = strategy.get_stop_target_pts(None, 'LONG', 97.0)
        assert stop == pytest.approx(2.0)   # nearest: 97 - 95


# ── 4. Zone state and lifecycle ───────────────────────────────────────────────

class TestZoneState:

    def test_active_zone_count_empty(self, strategy):
        assert strategy.active_zone_count == 0

    def test_active_zone_count_reflects_zones(self, strategy):
        strategy._active_zones.appendleft(_zone())
        strategy._active_zones.appendleft(_zone(fib_bot=80.0, fib_top=85.0))
        assert strategy.active_zone_count == 2

    def test_on_trade_exit_stop_loss_clears_zones(self, strategy):
        strategy._active_zones.appendleft(_zone())
        strategy._latest_cisd_features = np.zeros(10, dtype=np.float32)
        strategy._latest_zone_bullish = 1.0
        strategy.on_trade_exit('STOP_LOSS')
        assert strategy.active_zone_count == 0
        assert strategy._latest_cisd_features is None
        assert strategy._latest_zone_bullish == 0.0

    def test_on_trade_exit_take_profit_keeps_zones(self, strategy):
        strategy._active_zones.appendleft(_zone())
        strategy.on_trade_exit('TAKE_PROFIT')
        assert strategy.active_zone_count == 1


# ── 5. CISD feature vector (_build_cisd_feature_vector) ─────────────────────

class TestCISDFeatureVector:

    def test_no_zones_returns_none(self, strategy):
        df = _df()
        result = strategy._build_cisd_feature_vector(df, abs_bar=10)
        assert result is None

    def test_returns_10_features(self, strategy):
        strategy._active_zones.appendleft(_zone(fib_bot=95.0, fib_top=100.0))
        df = _df(close=97.0)
        result = strategy._build_cisd_feature_vector(df, abs_bar=10)
        assert result is not None
        assert result.shape == (10,)

    def test_bullish_zone_direction_feature(self, strategy):
        strategy._active_zones.appendleft(_zone(is_bullish=True))
        df = _df(close=97.0)
        result = strategy._build_cisd_feature_vector(df, abs_bar=10)
        assert result[4] == pytest.approx(1.0)   # zone_is_bullish

    def test_bearish_zone_direction_feature(self, strategy):
        strategy._active_zones.appendleft(_zone(is_bullish=False))
        df = _df(close=97.0)
        result = strategy._build_cisd_feature_vector(df, abs_bar=10)
        assert result[4] == pytest.approx(-1.0)  # zone_is_bullish

    def test_sweep_feature_when_present(self, strategy):
        strategy._active_zones.appendleft(_zone(had_sweep=True))
        df = _df(close=97.0)
        result = strategy._build_cisd_feature_vector(df, abs_bar=10)
        assert result[6] == pytest.approx(1.0)   # had_liquidity_sweep

    def test_no_sweep_feature(self, strategy):
        strategy._active_zones.appendleft(_zone(had_sweep=False))
        df = _df(close=97.0)
        result = strategy._build_cisd_feature_vector(df, abs_bar=10)
        assert result[6] == pytest.approx(0.0)

    def test_in_optimal_session_flag_on(self, strategy):
        strategy._active_zones.appendleft(_zone())
        df = _df(close=97.0, hour=OPTIMAL_START_HOUR)
        result = strategy._build_cisd_feature_vector(df, abs_bar=10)
        assert result[9] == pytest.approx(1.0)

    def test_in_optimal_session_flag_off(self, strategy):
        strategy._active_zones.appendleft(_zone())
        df = _df(close=97.0, hour=OPTIMAL_END_HOUR)   # at end hour = off
        result = strategy._build_cisd_feature_vector(df, abs_bar=10)
        assert result[9] == pytest.approx(0.0)

    def test_zone_age_normalized_by_max_bars(self, strategy):
        strategy._active_zones.appendleft(_zone(created_bar=0))
        df = _df(close=97.0)
        result = strategy._build_cisd_feature_vector(df, abs_bar=ZONE_MAX_BARS)
        assert result[3] == pytest.approx(1.0)   # age = ZONE_MAX_BARS / ZONE_MAX_BARS

    def test_no_nan_in_output(self, strategy):
        strategy._active_zones.appendleft(_zone())
        df = _df(close=97.0)
        result = strategy._build_cisd_feature_vector(df, abs_bar=5)
        assert not np.any(np.isnan(result))

    def test_all_features_within_clip_bounds(self, strategy):
        strategy._active_zones.appendleft(_zone())
        df = _df(close=97.0)
        result = strategy._build_cisd_feature_vector(df, abs_bar=5)
        assert np.all(result >= -10.0) and np.all(result <= 10.0)


# ── 6. Instrument resolution ─────────────────────────────────────────────────

class TestInstrumentResolution:
    """parse_future_symbol maps micro contracts to their parent for INSTRUMENT_MAP lookup."""

    def test_mnq_maps_to_nq(self):
        from bot_utils import parse_future_symbol
        assert parse_future_symbol('MNQZ5') == 'NQ'

    def test_mes_maps_to_es(self):
        from bot_utils import parse_future_symbol
        assert parse_future_symbol('MESH5') == 'ES'

    def test_mgc_maps_to_gc(self):
        from bot_utils import parse_future_symbol
        assert parse_future_symbol('MGCZ5') == 'GC'

    def test_full_contract_id_nq(self):
        from bot_utils import parse_future_symbol
        assert parse_future_symbol('NQZ5') == 'NQ'

    def test_none_returns_none(self):
        from bot_utils import parse_future_symbol
        assert parse_future_symbol(None) is None
