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

from strategies.strategy_cisd_ote_v7 import (
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
    s._min_risk_rr          = 0.0
    s._active_zones         = deque(maxlen=20)
    s._latest_cisd_features = None
    s._latest_zone_bullish  = 0.0
    s._latest_risk_rr       = 2.0
    s._latest_signal_meta   = {}
    s._bar_count            = 0
    s.skip_stats            = {'conf_gate': 0, 'rr_gate': 0, 'hold': 0}
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

    def test_rr_gate_disabled_by_default(self, strategy):
        """min_risk_rr=0.0 (default) — low predicted RR never blocks entry."""
        strategy._min_risk_rr = 0.0
        strategy._latest_risk_rr = 0.0
        ok, _ = self._enter(strategy, prediction=1, confidence=0.95)
        assert ok is True

    def test_rr_gate_blocks_low_rr(self, strategy):
        strategy._min_risk_rr = 2.0
        strategy._latest_risk_rr = 1.5
        ok, direction = self._enter(strategy, prediction=1, confidence=0.95)
        assert ok is False
        assert direction is None

    def test_rr_gate_passes_at_threshold(self, strategy):
        strategy._min_risk_rr = 2.0
        strategy._latest_risk_rr = 2.0
        ok, _ = self._enter(strategy, prediction=1, confidence=0.95)
        assert ok is True

    def test_rr_gate_passes_above_threshold(self, strategy):
        strategy._min_risk_rr = 2.0
        strategy._latest_risk_rr = 3.5
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

    def test_long_target_below_2r_skips(self, strategy):
        """predict=1.7 → skip (None, None)."""
        strategy._latest_risk_rr = 1.7
        strategy._active_zones.appendleft(_zone(fib_bot=95.0, fib_top=100.0))
        stop, target = strategy.get_stop_target_pts(None, 'LONG', 98.0)
        assert stop is None and target is None

    def test_long_target_int_floors(self, strategy):
        """predict=2.5 → int(2.5)=2 → 2R."""
        strategy._latest_risk_rr = 2.5
        strategy._active_zones.appendleft(_zone(fib_bot=95.0, fib_top=100.0))
        stop, target = strategy.get_stop_target_pts(None, 'LONG', 98.0)
        assert target == pytest.approx(stop * 2.0)

    def test_long_target_3r(self, strategy):
        """predict=3.7 → int(3.7)=3 → 3R."""
        strategy._latest_risk_rr = 3.7
        strategy._active_zones.appendleft(_zone(fib_bot=95.0, fib_top=100.0))
        stop, target = strategy.get_stop_target_pts(None, 'LONG', 98.0)
        assert target == pytest.approx(stop * 3.0)

    def test_long_target_4r(self, strategy):
        """predict=4.5 → int(4.5)=4 → 4R."""
        strategy._latest_risk_rr = 4.5
        strategy._active_zones.appendleft(_zone(fib_bot=95.0, fib_top=100.0))
        stop, target = strategy.get_stop_target_pts(None, 'LONG', 98.0)
        assert target == pytest.approx(stop * 4.0)

    def test_long_target_above_4r_no_ceiling(self, strategy):
        """predict=7.8 → int(7.8)=7 → 7R (no ceiling)."""
        strategy._latest_risk_rr = 7.8
        strategy._active_zones.appendleft(_zone(fib_bot=95.0, fib_top=100.0))
        stop, target = strategy.get_stop_target_pts(None, 'LONG', 98.0)
        assert target == pytest.approx(stop * 7.0)

    def test_short_stop_is_distance_to_fib_top(self, strategy):
        strategy._active_zones.appendleft(_zone(fib_bot=95.0, fib_top=100.0, is_bullish=False))
        stop, _ = strategy.get_stop_target_pts(None, 'SHORT', 97.0)
        assert stop == pytest.approx(3.0)   # 100 - 97

    def test_short_target_snaps_to_tier(self, strategy):
        """predict=3.0 snaps exactly to 3R tier."""
        strategy._latest_risk_rr = 3.0
        strategy._active_zones.appendleft(_zone(fib_bot=95.0, fib_top=100.0, is_bullish=False))
        stop, target = strategy.get_stop_target_pts(None, 'SHORT', 97.0)
        assert target == pytest.approx(stop * 3.0)

    def test_risk_rr_below_2_skips(self, strategy):
        """predicted_rr < 2.0 → (None, None)."""
        strategy._latest_risk_rr = 0.3
        strategy._active_zones.appendleft(_zone(fib_bot=95.0, fib_top=100.0))
        stop, target = strategy.get_stop_target_pts(None, 'LONG', 98.0)
        assert stop is None and target is None

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


# ── 6. Skip stats ────────────────────────────────────────────────────────────

class TestSkipStats:
    """should_enter_trade increments skip_stats for each rejection reason."""

    def test_conf_gate_increments(self, strategy):
        strategy.skip_stats = {'conf_gate': 0, 'rr_gate': 0, 'hold': 0}
        strategy.should_enter_trade(1, 0.60, {}, entry_conf=0.80, adx_thresh=0)
        assert strategy.skip_stats['conf_gate'] == 1
        assert strategy.skip_stats['rr_gate'] == 0

    def test_rr_gate_increments(self, strategy):
        strategy._min_risk_rr = 2.0
        strategy._latest_risk_rr = 1.5
        strategy.skip_stats = {'conf_gate': 0, 'rr_gate': 0, 'hold': 0}
        strategy.should_enter_trade(1, 0.90, {}, entry_conf=0.80, adx_thresh=0)
        assert strategy.skip_stats['rr_gate'] == 1
        assert strategy.skip_stats['conf_gate'] == 0

    def test_hold_prediction_increments(self, strategy):
        strategy.skip_stats = {'conf_gate': 0, 'rr_gate': 0, 'hold': 0}
        strategy.should_enter_trade(0, 0.90, {}, entry_conf=0.80, adx_thresh=0)
        assert strategy.skip_stats['hold'] == 1

    def test_successful_entry_does_not_increment(self, strategy):
        strategy.skip_stats = {'conf_gate': 0, 'rr_gate': 0, 'hold': 0}
        strategy.should_enter_trade(1, 0.90, {}, entry_conf=0.80, adx_thresh=0)
        assert strategy.skip_stats['conf_gate'] == 0
        assert strategy.skip_stats['rr_gate'] == 0
        assert strategy.skip_stats['hold'] == 0

    def test_stats_accumulate_across_calls(self, strategy):
        strategy.skip_stats = {'conf_gate': 0, 'rr_gate': 0, 'hold': 0}
        for _ in range(3):
            strategy.should_enter_trade(1, 0.50, {}, entry_conf=0.80, adx_thresh=0)
        assert strategy.skip_stats['conf_gate'] == 3


# ── 7. Model compatibility ────────────────────────────────────────────────────

_V5_MODEL  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'models', 'cisd_ote_hybrid_v5_1.onnx')
_V7_MODEL  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'models', 'cisd_ote_hybrid_v7.onnx')
_MODELS_AVAILABLE = os.path.exists(_V5_MODEL) and os.path.exists(_V7_MODEL)


@pytest.mark.skipif(not _MODELS_AVAILABLE, reason="ONNX model files not present")
class TestModelCompatibility:
    """load_model() raises ValueError when the ONNX model is incompatible with v7 strategy."""

    def _make_strategy(self, model_path):
        """Build a bare strategy instance and call load_model()."""
        import onnxruntime
        s = CISDOTEStrategyV7.__new__(CISDOTEStrategyV7)
        s.model_path = model_path
        s.model = None
        s.load_model()
        return s

    def test_v7_model_loads_without_error(self):
        s = self._make_strategy(_V7_MODEL)
        assert s.model is not None

    def test_v5_model_raises_value_error(self):
        with pytest.raises(ValueError, match="incompatible with CISDOTEStrategyV7"):
            self._make_strategy(_V5_MODEL)

    def test_v5_error_message_names_missing_inputs(self):
        with pytest.raises(ValueError, match="features"):
            self._make_strategy(_V5_MODEL)

    def test_v5_error_message_suggests_v7_model(self):
        with pytest.raises(ValueError, match="v7.onnx"):
            self._make_strategy(_V5_MODEL)


# ── 8. Instrument resolution ─────────────────────────────────────────────────

class TestInstrumentResolution:
    """parse_future_symbol maps micro contracts to their parent for INSTRUMENT_MAP lookup."""

    def test_mnq_maps_to_nq(self):
        from utils.bot_utils import parse_future_symbol
        assert parse_future_symbol('MNQZ5') == 'NQ'

    def test_mes_maps_to_es(self):
        from utils.bot_utils import parse_future_symbol
        assert parse_future_symbol('MESH5') == 'ES'

    def test_mgc_maps_to_gc(self):
        from utils.bot_utils import parse_future_symbol
        assert parse_future_symbol('MGCZ5') == 'GC'

    def test_full_contract_id_nq(self):
        from utils.bot_utils import parse_future_symbol
        assert parse_future_symbol('NQZ5') == 'NQ'

    def test_none_returns_none(self):
        from utils.bot_utils import parse_future_symbol
        assert parse_future_symbol(None) is None
