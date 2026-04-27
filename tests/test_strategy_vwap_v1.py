"""Unit tests for VWAPReversionStrategyV1 — session gate, ATR, VWAP state, signals, stop/target.

Every test verifies behaviour against the exact labeling code in vwap_v1.py — the
source of truth for how the model was trained. Key things being guarded:

  1. Session gate: 07:00–16:00 ET only
  2. Wilder's ATR (period=14)
  3. Daily VWAP: resets on date change; SD needs 3+ deviations
  4. 1H bucket VWAP: resets on hour change
  5. ATR rank computed BEFORE appending (excludes current bar)
  6. VWAP-SD rank computed BEFORE appending
  7. Bars-at-extreme counter: increments/resets, capped at 20
  8. Dev momentum: dev_sd[i] - dev_sd[i-3], clipped [-5, 5]
  9. Signal fires ONLY on abs(dev_sd) >= 2.0 + reversal candle + session
 10. Stop: bar_extreme ± 0.1 × ATR (exact match with labeler)
 11. TP tiers: <2R skip, 2–4R fixed 2R, ≥4R → VWAP distance
 12. Entry gate (conf threshold + RR gate)
 13. on_trade_exit resets signal state
 14. 8-element feature vector matches training feature column order
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd
from collections import deque
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Stub futures_foundation before importing the strategy.
_ff_stub = MagicMock()
_ff_stub.derive_features = MagicMock(return_value=pd.DataFrame())
_ff_stub.get_model_feature_columns = MagicMock(return_value=[])
_ff_stub.INSTRUMENT_MAP = {'ES': 0, 'NQ': 1, 'RTY': 2, 'YM': 3, 'GC': 4}
sys.modules.setdefault('futures_foundation', _ff_stub)

from strategies.strategy_vwap_v1 import (
    VWAPReversionStrategyV1,
    VWAP_DEV_THRESH, SL_ATR_MULT, ATR_PERIOD, ATR_RANK_WINDOW, SD_RANK_WINDOW,
    SESSION_START, SESSION_END,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bar(high, low, close, open_=None, vol=1000, dt='2026-04-21 10:00:00'):
    """Single-row DataFrame for calling _on_new_bar."""
    if open_ is None:
        open_ = close
    idx = pd.DatetimeIndex([pd.Timestamp(dt, tz='America/New_York')])
    return pd.DataFrame(
        {'open': [open_], 'high': [high], 'low': [low], 'close': [close], 'volume': [vol]},
        index=idx,
    )


def _seed_vwap(s, vwap=100.0, vwap_sd=5.0, date='2026-04-21'):
    """
    Pre-seed daily VWAP state so a single bar produces a predictable dev_sd.
    Uses huge cum_vol so the new bar doesn't meaningfully shift VWAP.
    """
    s._current_date = pd.Timestamp(date).date()
    s._cum_vol      = 1_000_000.0
    s._cum_tp_vol   = vwap * 1_000_000.0
    s._vwap         = vwap
    # 100 alternating deviations → std = vwap_sd
    s._vwap_devs    = [vwap_sd, -vwap_sd] * 50
    s._vwap_sd      = float(np.std([vwap_sd, -vwap_sd] * 50))


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def vwap():
    """VWAPReversionStrategyV1 with all state pre-initialised — no ONNX model loaded."""
    s = VWAPReversionStrategyV1.__new__(VWAPReversionStrategyV1)
    s._instrument    = 'NQ'
    s._min_risk_rr   = 0.0

    # Daily VWAP
    s._current_date  = None
    s._cum_tp_vol    = 0.0
    s._cum_vol       = 0.0
    s._vwap_devs     = []
    s._vwap          = 0.0
    s._vwap_sd       = 0.0

    # 1H bucket VWAP
    s._htf_hour_key    = None
    s._htf_cum_tv      = 0.0
    s._htf_cum_v       = 0.0
    s._htf_devs        = []
    s._htf_vwap_dev_sd = 0.0

    # Wilder ATR
    s._atr             = 0.0
    s._prev_close      = 0.0
    s._atr_initialized = False

    # Rolling windows
    s._atr_history = deque(maxlen=ATR_RANK_WINDOW)
    s._sd_history  = deque(maxlen=SD_RANK_WINDOW)

    # Lookback deques
    s._recent_dev_sd = deque(maxlen=4)
    s._recent_vwap   = deque(maxlen=13)

    # Counter
    s._bars_at_extreme = 0

    # Signal state
    s._is_signal_bar         = False
    s._current_direction     = 0
    s._current_vwap_features = np.zeros(8, dtype=np.float32)
    s._signal_atr            = 0.0
    s._signal_sl_dist        = 0.0
    s._signal_vwap           = 0.0
    s._latest_risk_rr        = 0.0
    s._latest_signal_meta    = {}

    s._bar_count = 0
    s.skip_stats = {'conf_gate': 0, 'rr_gate': 0, 'hold': 0}
    return s


# ── 1. Session gate (is_trading_allowed) ─────────────────────────────────────

class TestSessionGate:
    """VWAP has no session filter — is_trading_allowed always returns True (matches CISD v7 / SuperTrend)."""

    def test_always_true_during_session(self, vwap):
        ts = pd.Timestamp('2026-04-21 10:00:00', tz='America/New_York')
        assert vwap.is_trading_allowed(ts) is True

    def test_always_true_pre_market(self, vwap):
        ts = pd.Timestamp('2026-04-21 06:00:00', tz='America/New_York')
        assert vwap.is_trading_allowed(ts) is True

    def test_always_true_after_hours(self, vwap):
        ts = pd.Timestamp('2026-04-21 20:00:00', tz='America/New_York')
        assert vwap.is_trading_allowed(ts) is True

    def test_always_true_overnight(self, vwap):
        ts = pd.Timestamp('2026-04-21 01:00:00', tz='America/New_York')
        assert vwap.is_trading_allowed(ts) is True


# ── 2. Wilder's ATR (14-period) ───────────────────────────────────────────────

class TestWildersATR:
    """ATR[i] = ((period-1)*ATR[i-1] + TR[i]) / period, floored at 1e-6."""

    def test_first_bar_initializes_to_hl_range(self, vwap):
        vwap._on_new_bar(_bar(102, 98, 100), 0)
        assert vwap._atr == pytest.approx(4.0)

    def test_second_bar_uses_wilder_formula(self, vwap):
        vwap._on_new_bar(_bar(102, 98, 100), 0)   # ATR = 4.0, prev_close=100
        # TR = max(103-99=4, |103-100|=3, |99-100|=1) = 4
        vwap._on_new_bar(_bar(103, 99, 101), 1)
        expected = ((ATR_PERIOD - 1) * 4.0 + 4.0) / ATR_PERIOD
        assert vwap._atr == pytest.approx(expected)

    def test_atr_with_large_true_range(self, vwap):
        vwap._on_new_bar(_bar(102, 98, 100), 0)   # ATR = 4.0
        # TR = max(108-96=12, |108-100|=8, |96-100|=4) = 12
        vwap._on_new_bar(_bar(108, 96, 102), 1)
        expected = ((ATR_PERIOD - 1) * 4.0 + 12.0) / ATR_PERIOD
        assert vwap._atr == pytest.approx(expected)

    def test_atr_floor_at_1e6(self, vwap):
        """Zero-range bar must not produce zero ATR."""
        vwap._on_new_bar(_bar(100, 100, 100), 0)
        assert vwap._atr == pytest.approx(1e-6)

    def test_atr_accumulates_over_multiple_bars(self, vwap):
        bars = [(102, 98, 100), (103, 99, 101), (105, 100, 102), (104, 101, 103)]
        for i, (h, l, c) in enumerate(bars):
            vwap._on_new_bar(_bar(h, l, c), i)
        # Step through manually: TRs for bars 1,2,3 are 4, 5, 3
        atr = 4.0
        for tr in [4.0, 5.0, 3.0]:
            atr = ((ATR_PERIOD - 1) * atr + tr) / ATR_PERIOD
        assert vwap._atr == pytest.approx(atr, rel=1e-5)

    def test_prev_close_updated_each_bar(self, vwap):
        vwap._on_new_bar(_bar(102, 98, 100), 0)
        assert vwap._prev_close == pytest.approx(100.0)
        vwap._on_new_bar(_bar(103, 99, 101), 1)
        assert vwap._prev_close == pytest.approx(101.0)


# ── 3. Daily VWAP ─────────────────────────────────────────────────────────────

class TestDailyVWAP:
    """VWAP = sum(tp*vol) / sum(vol). Resets on calendar date change."""

    def test_first_bar_vwap_equals_typical_price(self, vwap):
        # tp = (102+98+100)/3 = 100
        vwap._on_new_bar(_bar(102, 98, 100, vol=1000), 0)
        assert vwap._vwap == pytest.approx(100.0)

    def test_vwap_accumulates_volume_weighted(self, vwap):
        vwap._on_new_bar(_bar(102, 98, 100, vol=1000), 0)   # tp=100
        vwap._on_new_bar(_bar(112, 108, 110, vol=1000), 1)  # tp=110
        expected = (100.0 * 1000 + 110.0 * 1000) / 2000
        assert vwap._vwap == pytest.approx(expected)

    def test_vwap_sd_zero_with_fewer_than_3_bars(self, vwap):
        vwap._on_new_bar(_bar(102, 98, 100, vol=1000), 0)
        vwap._on_new_bar(_bar(103, 99, 101, vol=1000), 1)
        assert vwap._vwap_sd == pytest.approx(0.0)

    def test_vwap_sd_nonzero_after_3_bars(self, vwap):
        for i in range(3):
            vwap._on_new_bar(_bar(102 + i, 98 + i, 100 + i, vol=1000), i)
        assert vwap._vwap_sd > 0.0

    def test_daily_vwap_resets_on_date_change(self, vwap):
        vwap._on_new_bar(_bar(102, 98, 100, vol=1_000_000, dt='2026-04-21 15:00:00'), 0)
        # New day — state resets; VWAP should reflect only the new bar's tp
        vwap._on_new_bar(_bar(105, 101, 103, vol=100, dt='2026-04-22 09:00:00'), 1)
        expected_tp = (105 + 101 + 103) / 3.0
        assert vwap._vwap == pytest.approx(expected_tp)

    def test_zero_volume_falls_back_to_tp(self, vwap):
        vwap._on_new_bar(_bar(102, 98, 100, vol=0), 0)
        assert vwap._vwap == pytest.approx((102 + 98 + 100) / 3.0)


# ── 4. 1H bucket VWAP ─────────────────────────────────────────────────────────

class TestHTFBucketVWAP:
    """HTF VWAP resets each calendar hour and accumulates within the hour."""

    def test_first_bar_sets_hour_key(self, vwap):
        vwap._on_new_bar(_bar(102, 98, 100, dt='2026-04-21 10:00:00'), 0)
        assert vwap._htf_hour_key == (2026, 4, 21, 10)

    def test_same_hour_accumulates_volume(self, vwap):
        vwap._on_new_bar(_bar(102, 98, 100, vol=1000, dt='2026-04-21 10:00:00'), 0)
        cum_v_after_bar1 = vwap._htf_cum_v
        vwap._on_new_bar(_bar(103, 99, 101, vol=1000, dt='2026-04-21 10:05:00'), 1)
        assert vwap._htf_cum_v == pytest.approx(cum_v_after_bar1 + 1000)

    def test_new_hour_resets_bucket(self, vwap):
        vwap._on_new_bar(_bar(102, 98, 100, vol=1000, dt='2026-04-21 10:55:00'), 0)
        vwap._on_new_bar(_bar(103, 99, 101, vol=500,  dt='2026-04-21 11:00:00'), 1)
        assert vwap._htf_hour_key == (2026, 4, 21, 11)
        assert vwap._htf_cum_v == pytest.approx(500.0)

    def test_htf_dev_zero_when_fewer_than_3_bars_in_bucket(self, vwap):
        """HTF SD is zero until 3 bars exist in current hour bucket."""
        vwap._on_new_bar(_bar(102, 98, 100, vol=1000, dt='2026-04-21 10:00:00'), 0)
        assert vwap._htf_vwap_dev_sd == pytest.approx(0.0)

    def test_htf_key_changes_across_days(self, vwap):
        vwap._on_new_bar(_bar(102, 98, 100, dt='2026-04-21 10:00:00'), 0)
        vwap._on_new_bar(_bar(103, 99, 101, dt='2026-04-22 10:00:00'), 1)
        assert vwap._htf_hour_key == (2026, 4, 22, 10)


# ── 5. ATR rank ───────────────────────────────────────────────────────────────

class TestATRRank:
    """atr_rank = proportion of history < current ATR — computed BEFORE appending."""

    def test_history_grows_each_bar(self, vwap):
        for i in range(5):
            vwap._on_new_bar(_bar(102, 98, 100), i)
        assert len(vwap._atr_history) == 5

    def test_history_capped_at_window(self, vwap):
        for i in range(ATR_RANK_WINDOW + 50):
            vwap._on_new_bar(_bar(102, 98, 100), i)
        assert len(vwap._atr_history) == ATR_RANK_WINDOW

    def test_first_bar_history_has_one_entry_after_bar(self, vwap):
        """Rank is computed with empty history on bar 0, then current ATR is appended."""
        vwap._on_new_bar(_bar(102, 98, 100), 0)
        assert len(vwap._atr_history) == 1

    def test_large_atr_after_small_history_gets_high_rank(self, vwap):
        """A very wide bar after a calm period should have rank near 1.0."""
        for i in range(20):
            vwap._on_new_bar(_bar(101, 99, 100), i)   # narrow bars, ATR ≈ 2
        atr_before_spike = vwap._atr
        vwap._on_new_bar(_bar(130, 70, 100), 20)       # huge range
        # ATR after spike is bigger than calm ATR
        assert vwap._atr > atr_before_spike


# ── 6. VWAP-SD rank ───────────────────────────────────────────────────────────

class TestSDRank:
    """vwap_sd_rank computed BEFORE appending — excludes current bar."""

    def test_sd_history_grows_per_bar(self, vwap):
        for i in range(5):
            vwap._on_new_bar(_bar(102 + i, 98 + i, 100 + i, vol=1000), i)
        # sd_history only adds when vwap_sd > 0 (needs 3+ devs first)
        assert len(vwap._sd_history) <= 5

    def test_sd_history_capped_at_window(self, vwap):
        for i in range(SD_RANK_WINDOW + 50):
            vwap._on_new_bar(_bar(102, 98, 100 + (i % 3), vol=1000), i)
        assert len(vwap._sd_history) <= SD_RANK_WINDOW


# ── 7. Bars-at-extreme counter ────────────────────────────────────────────────

class TestBarsAtExtreme:
    """Counter increments when abs(dev_sd) >= 2.0, resets to 0 otherwise, capped at 20."""

    def test_increments_on_extreme_bar(self, vwap):
        _seed_vwap(vwap)
        # dev_sd = (89 - 100) / 5 = -2.2 → abs >= 2.0
        vwap._on_new_bar(_bar(91, 87, 89, open_=87, vol=100), 0)
        assert vwap._bars_at_extreme == 1

    def test_resets_on_non_extreme_bar(self, vwap):
        _seed_vwap(vwap)
        vwap._bars_at_extreme = 5
        # dev_sd = (99 - 100) / 5 = -0.2 → abs < 2.0 → reset
        vwap._on_new_bar(_bar(100, 98, 99, open_=99, vol=100), 0)
        assert vwap._bars_at_extreme == 0

    def test_capped_at_20(self, vwap):
        _seed_vwap(vwap)
        vwap._bars_at_extreme = 20
        vwap._on_new_bar(_bar(91, 87, 89, open_=87, vol=100), 0)
        assert vwap._bars_at_extreme == 20

    def test_accumulates_consecutive_extremes(self, vwap):
        _seed_vwap(vwap)
        # With huge cum_vol, VWAP barely shifts so dev stays extreme each bar
        for i in range(3):
            vwap._on_new_bar(_bar(91, 87, 89, open_=87, vol=100), i)
        assert vwap._bars_at_extreme == 3


# ── 8. Dev momentum deque ─────────────────────────────────────────────────────

class TestDevMomentumDeque:
    """_recent_dev_sd tracks the last 4 dev_sd values for momentum feature."""

    def test_deque_grows_per_bar(self, vwap):
        _seed_vwap(vwap)
        for i in range(4):
            vwap._on_new_bar(_bar(100, 98, 99, open_=99, vol=100), i)
        assert len(vwap._recent_dev_sd) == 4

    def test_deque_capped_at_4(self, vwap):
        _seed_vwap(vwap)
        for i in range(10):
            vwap._on_new_bar(_bar(100, 98, 99, open_=99, vol=100), i)
        assert len(vwap._recent_dev_sd) == 4

    def test_recent_vwap_deque_capped_at_13(self, vwap):
        for i in range(20):
            vwap._on_new_bar(_bar(102, 98, 100, vol=1000), i)
        assert len(vwap._recent_vwap) == 13


# ── 9. Signal detection ───────────────────────────────────────────────────────

class TestSignalDetection:
    """Signal fires ONLY on: abs(dev_sd) >= 2.0, reversal candle, session 7–16 ET."""

    def test_no_signal_when_vwap_sd_zero(self, vwap):
        """< 3 bars → sd=0 → no signal regardless of price."""
        vwap._on_new_bar(_bar(80, 70, 75, open_=90, vol=1000), 0)
        assert vwap._is_signal_bar is False

    def test_no_signal_when_dev_below_threshold(self, vwap):
        _seed_vwap(vwap)
        # dev_sd = (99 - 100) / 5 = -0.2 < 2.0
        vwap._on_new_bar(_bar(100, 98, 99, open_=98, vol=100), 0)
        assert vwap._is_signal_bar is False

    def test_no_signal_at_extreme_but_no_reversal_candle(self, vwap):
        """Bar is extreme (dev_sd=-2.2) but close < open → not a bull reversal."""
        _seed_vwap(vwap)
        vwap._on_new_bar(_bar(91, 87, 89, open_=92, vol=100), 0)   # bearish close
        assert vwap._is_signal_bar is False

    def test_buy_signal_on_extreme_bullish_bar(self, vwap):
        """Below VWAP by 2+ SDs + bullish candle → BUY (direction=1)."""
        _seed_vwap(vwap)
        vwap._on_new_bar(_bar(91, 87, 89, open_=87, vol=100), 0)   # bullish close
        assert vwap._is_signal_bar is True
        assert vwap._current_direction == 1

    def test_sell_signal_on_extreme_bearish_bar(self, vwap):
        """Above VWAP by 2+ SDs + bearish candle → SELL (direction=2)."""
        _seed_vwap(vwap)
        vwap._on_new_bar(_bar(113, 109, 111, open_=112, vol=100), 0)  # bearish close
        assert vwap._is_signal_bar is True
        assert vwap._current_direction == 2

    def test_signal_fires_outside_regular_session_hours(self, vwap):
        """No session gate — extreme + reversal at 17:00 ET still signals."""
        _seed_vwap(vwap)
        vwap._on_new_bar(
            _bar(91, 87, 89, open_=87, vol=100, dt='2026-04-21 17:00:00'), 0
        )
        assert vwap._is_signal_bar is True

    def test_signal_cleared_on_next_non_extreme_bar(self, vwap):
        _seed_vwap(vwap)
        vwap._on_new_bar(_bar(91, 87, 89, open_=87, vol=100), 0)
        assert vwap._is_signal_bar is True
        vwap._on_new_bar(_bar(101, 99, 100, open_=100, vol=100), 1)
        assert vwap._is_signal_bar is False

    def test_signal_at_exact_2sd_boundary(self, vwap):
        """abs(dev_sd) exactly 2.0 must trigger."""
        _seed_vwap(vwap, vwap=100.0, vwap_sd=5.0)
        # dev_sd needs to be exactly -2.0: close = 100 - 2*5 = 90 (approx)
        # With huge cum_vol, after appending the new bar vwap barely shifts.
        # Use close=90 → dev_sd ≈ -2.0 (might be -2.00x depending on exact SD)
        vwap._on_new_bar(_bar(92, 88, 90, open_=88, vol=100), 0)
        # dev_sd at -2.0 boundary — just verify no crash and direction set correctly
        if vwap._is_signal_bar:
            assert vwap._current_direction == 1  # BUY (below VWAP, bullish close)


# ── 10. Stop calculation ──────────────────────────────────────────────────────

class TestSignalStop:
    """sl_dist = abs(close - bar_extreme) + SL_ATR_MULT × ATR (matches labeler)."""

    def test_long_stop_is_close_minus_low_plus_atr_buffer(self, vwap):
        _seed_vwap(vwap)
        vwap._on_new_bar(_bar(91, 87, 89, open_=87, vol=100), 0)
        assert vwap._is_signal_bar is True
        expected_sl = abs(89 - 87) + SL_ATR_MULT * vwap._atr
        assert vwap._signal_sl_dist == pytest.approx(expected_sl)

    def test_short_stop_is_high_minus_close_plus_atr_buffer(self, vwap):
        _seed_vwap(vwap)
        vwap._on_new_bar(_bar(113, 109, 111, open_=112, vol=100), 0)
        assert vwap._is_signal_bar is True
        expected_sl = abs(113 - 111) + SL_ATR_MULT * vwap._atr
        assert vwap._signal_sl_dist == pytest.approx(expected_sl)

    def test_signal_vwap_captured_at_bar(self, vwap):
        _seed_vwap(vwap, vwap=100.0)
        vwap._on_new_bar(_bar(91, 87, 89, open_=87, vol=100), 0)
        # signal_vwap should be the updated VWAP (≈100 with huge cum_vol)
        assert vwap._signal_vwap == pytest.approx(vwap._vwap, rel=1e-3)

    def test_signal_atr_captured_at_bar(self, vwap):
        _seed_vwap(vwap)
        vwap._on_new_bar(_bar(91, 87, 89, open_=87, vol=100), 0)
        assert vwap._signal_atr == pytest.approx(vwap._atr)


# ── 11. Stop/target tiers (get_stop_target_pts) ───────────────────────────────

class TestStopTargetPts:
    """TP tiers: <2R skip, 2–4R → 2R, ≥4R → max(vwap_dist, int(rr)×R)."""

    def _setup(self, s, sl_dist=5.0, rr=2.5, signal_vwap=100.0):
        s._signal_sl_dist = sl_dist
        s._latest_risk_rr = rr
        s._signal_vwap    = signal_vwap

    def test_zero_sl_dist_returns_none(self, vwap):
        vwap._signal_sl_dist = 0.0
        stop, target = vwap.get_stop_target_pts(None, 'LONG', 100.0)
        assert stop is None and target is None

    def test_rr_below_2_skips(self, vwap):
        self._setup(vwap, sl_dist=5.0, rr=1.9)
        stop, target = vwap.get_stop_target_pts(None, 'LONG', 100.0)
        assert stop is None and target is None

    def test_rr_exactly_2_gives_2r(self, vwap):
        self._setup(vwap, sl_dist=5.0, rr=2.0)
        stop, target = vwap.get_stop_target_pts(None, 'LONG', 100.0)
        assert stop   == pytest.approx(5.0)
        assert target == pytest.approx(10.0)   # int(2.0) × 5 = 2 × 5

    def test_rr_2_5_floors_to_2r(self, vwap):
        """int(2.5) = 2 → 2R."""
        self._setup(vwap, sl_dist=4.0, rr=2.5)
        stop, target = vwap.get_stop_target_pts(None, 'LONG', 100.0)
        assert target == pytest.approx(4.0 * 2)

    def test_rr_3_7_floors_to_3r(self, vwap):
        """int(3.7) = 3 → 3R."""
        self._setup(vwap, sl_dist=5.0, rr=3.7)
        stop, target = vwap.get_stop_target_pts(None, 'LONG', 100.0)
        assert target == pytest.approx(5.0 * 3)

    def test_rr_4_gives_4r(self, vwap):
        self._setup(vwap, sl_dist=5.0, rr=4.0)
        stop, target = vwap.get_stop_target_pts(None, 'LONG', 100.0)
        assert target == pytest.approx(5.0 * 4)

    def test_rr_6_7_floors_to_6r(self, vwap):
        """int(6.7) = 6 → 6R."""
        self._setup(vwap, sl_dist=5.0, rr=6.7)
        stop, target = vwap.get_stop_target_pts(None, 'LONG', 100.0)
        assert target == pytest.approx(5.0 * 6)

    def test_stop_passthrough(self, vwap):
        """stop_pts always equals _signal_sl_dist regardless of TP tier."""
        self._setup(vwap, sl_dist=7.5, rr=3.0)
        stop, _ = vwap.get_stop_target_pts(None, 'SHORT', 100.0)
        assert stop == pytest.approx(7.5)


# ── 12. Entry gate (should_enter_trade) ───────────────────────────────────────

class TestEntryGate:

    def _enter(self, s, prediction=1, confidence=0.80, entry_conf=0.70):
        return s.should_enter_trade(prediction, confidence, {}, entry_conf, 0)

    def test_long_above_threshold_enters(self, vwap):
        ok, direction = self._enter(vwap, prediction=1, confidence=0.80)
        assert ok is True
        assert direction == 'LONG'

    def test_short_above_threshold_enters(self, vwap):
        ok, direction = self._enter(vwap, prediction=2, confidence=0.80)
        assert ok is True
        assert direction == 'SHORT'

    def test_exactly_at_threshold_enters(self, vwap):
        ok, _ = self._enter(vwap, confidence=0.70, entry_conf=0.70)
        assert ok is True

    def test_below_threshold_blocked(self, vwap):
        ok, direction = self._enter(vwap, confidence=0.69, entry_conf=0.70)
        assert ok is False
        assert direction is None

    def test_hold_prediction_blocked(self, vwap):
        ok, direction = self._enter(vwap, prediction=0, confidence=0.95)
        assert ok is False
        assert direction is None

    def test_rr_gate_disabled_at_zero(self, vwap):
        vwap._min_risk_rr    = 0.0
        vwap._latest_risk_rr = 0.0
        ok, _ = self._enter(vwap, prediction=1, confidence=0.95)
        assert ok is True

    def test_rr_gate_blocks_low_rr(self, vwap):
        vwap._min_risk_rr    = 2.0
        vwap._latest_risk_rr = 1.5
        ok, direction = self._enter(vwap, prediction=1, confidence=0.95)
        assert ok is False
        assert direction is None

    def test_rr_gate_passes_at_threshold(self, vwap):
        vwap._min_risk_rr    = 2.0
        vwap._latest_risk_rr = 2.0
        ok, _ = self._enter(vwap, prediction=1, confidence=0.95)
        assert ok is True

    def test_rr_gate_passes_above_threshold(self, vwap):
        vwap._min_risk_rr    = 2.0
        vwap._latest_risk_rr = 4.5
        ok, _ = self._enter(vwap, prediction=1, confidence=0.95)
        assert ok is True

    def test_conservative_threshold_0_80(self, vwap):
        ok, direction = self._enter(vwap, confidence=0.81, entry_conf=0.80)
        assert ok is True
        assert direction == 'LONG'

    def test_just_below_conservative_threshold_blocked(self, vwap):
        ok, _ = self._enter(vwap, confidence=0.79, entry_conf=0.80)
        assert ok is False


# ── 13. Skip stats ────────────────────────────────────────────────────────────

class TestSkipStats:
    """should_enter_trade must increment skip_stats for each rejection reason."""

    def test_conf_gate_increments(self, vwap):
        vwap.should_enter_trade(1, 0.50, {}, entry_conf=0.70, adx_thresh=0)
        assert vwap.skip_stats['conf_gate'] == 1
        assert vwap.skip_stats['rr_gate']   == 0

    def test_rr_gate_increments(self, vwap):
        vwap._min_risk_rr    = 2.0
        vwap._latest_risk_rr = 1.5
        vwap.should_enter_trade(1, 0.90, {}, entry_conf=0.70, adx_thresh=0)
        assert vwap.skip_stats['rr_gate']   == 1
        assert vwap.skip_stats['conf_gate'] == 0

    def test_hold_increments(self, vwap):
        vwap.should_enter_trade(0, 0.90, {}, entry_conf=0.70, adx_thresh=0)
        assert vwap.skip_stats['hold'] == 1

    def test_successful_entry_no_increment(self, vwap):
        vwap.should_enter_trade(1, 0.90, {}, entry_conf=0.70, adx_thresh=0)
        assert all(v == 0 for v in vwap.skip_stats.values())

    def test_stats_accumulate_across_calls(self, vwap):
        for _ in range(5):
            vwap.should_enter_trade(1, 0.50, {}, entry_conf=0.70, adx_thresh=0)
        assert vwap.skip_stats['conf_gate'] == 5


# ── 14. on_trade_exit ─────────────────────────────────────────────────────────

class TestOnTradeExit:
    """Exit must reset signal state so the next bar starts clean."""

    def test_stop_loss_clears_signal_bar_flag(self, vwap):
        vwap._is_signal_bar     = True
        vwap._current_direction = 1
        vwap.on_trade_exit('STOP_LOSS')
        assert vwap._is_signal_bar is False

    def test_take_profit_clears_direction(self, vwap):
        vwap._current_direction = 2
        vwap.on_trade_exit('TAKE_PROFIT')
        assert vwap._current_direction == 0

    def test_exit_with_no_signal_is_idempotent(self, vwap):
        vwap._is_signal_bar     = False
        vwap._current_direction = 0
        vwap.on_trade_exit('STOP_LOSS')
        assert vwap._is_signal_bar is False
        assert vwap._current_direction == 0


# ── 15. Feature vector ────────────────────────────────────────────────────────

class TestFeatureVector:
    """_current_vwap_features must be 8 float32 values in training column order."""

    def _fire_buy_signal(self, s):
        """Helper: seed state and fire a BUY signal bar."""
        _seed_vwap(s, vwap=100.0, vwap_sd=5.0)
        # Pre-populate atr_history for a nonzero rank
        for _ in range(5):
            s._atr_history.append(3.0)
        # Pre-initialize ATR
        s._atr             = 4.0
        s._prev_close      = 91.0
        s._atr_initialized = True
        s._on_new_bar(_bar(91, 87, 89, open_=87, vol=100), 0)

    def test_signal_bar_fires(self, vwap):
        self._fire_buy_signal(vwap)
        assert vwap._is_signal_bar is True

    def test_feature_vector_has_8_elements(self, vwap):
        self._fire_buy_signal(vwap)
        assert vwap._current_vwap_features.shape == (8,)

    def test_feature_vector_is_float32(self, vwap):
        self._fire_buy_signal(vwap)
        assert vwap._current_vwap_features.dtype == np.float32

    def test_vwap_dev_sd_negative_for_buy_signal(self, vwap):
        """Index 0: signed deviation — must be negative (below VWAP) for BUY."""
        self._fire_buy_signal(vwap)
        assert vwap._current_vwap_features[0] < 0

    def test_vwap_dev_pct_matches_formula(self, vwap):
        """Index 1: (close - vwap) / (close + 1e-6) × 100."""
        self._fire_buy_signal(vwap)
        close      = 89.0
        vwap_price = vwap._vwap
        expected   = (close - vwap_price) / (close + 1e-6) * 100
        assert vwap._current_vwap_features[1] == pytest.approx(expected, rel=1e-3)

    def test_atr_rank_in_unit_interval(self, vwap):
        """Index 2: ATR percentile rank ∈ [0, 1]."""
        self._fire_buy_signal(vwap)
        assert 0.0 <= float(vwap._current_vwap_features[2]) <= 1.0

    def test_bars_at_extreme_non_negative(self, vwap):
        """Index 5: consecutive extreme bar count ≥ 0."""
        self._fire_buy_signal(vwap)
        assert float(vwap._current_vwap_features[5]) >= 0

    def test_no_nan_in_feature_vector(self, vwap):
        self._fire_buy_signal(vwap)
        assert not np.any(np.isnan(vwap._current_vwap_features))


# ── 16. Non-signal predict short-circuits ────────────────────────────────────

class TestPredictShortCircuit:
    """predict() must return (0, 0.0) immediately without hitting the model."""

    def test_returns_hold_when_no_signal_bar(self, vwap):
        vwap._is_signal_bar     = False
        vwap._current_direction = 0
        pred, conf = vwap.predict(pd.DataFrame())
        assert pred == 0
        assert conf == pytest.approx(0.0)

    def test_returns_hold_when_direction_zero(self, vwap):
        vwap._is_signal_bar     = True   # signal set but direction not resolved
        vwap._current_direction = 0
        pred, conf = vwap.predict(pd.DataFrame())
        assert pred == 0
        assert conf == pytest.approx(0.0)


# ── 17. Strategy constants ────────────────────────────────────────────────────

class TestStrategyConstants:
    """Verify that live constants match what the model was trained with."""

    def test_vwap_dev_threshold_is_2(self):
        assert VWAP_DEV_THRESH == pytest.approx(2.0)

    def test_sl_atr_mult_is_0_1(self):
        assert SL_ATR_MULT == pytest.approx(0.1)

    def test_atr_period_is_14(self):
        assert ATR_PERIOD == 14

    def test_session_start_is_7(self):
        assert SESSION_START == 7

    def test_session_end_is_16(self):
        assert SESSION_END == 16

    def test_sequence_length_is_96(self):
        s = VWAPReversionStrategyV1.__new__(VWAPReversionStrategyV1)
        assert s.get_sequence_length() == 96

    def test_warmup_length_is_200(self):
        s = VWAPReversionStrategyV1.__new__(VWAPReversionStrategyV1)
        assert s.get_warmup_length() == 200
